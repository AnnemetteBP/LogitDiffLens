from typing import List, Optional, Any, Dict, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import bitsandbytes as bnb
from datasets import Dataset, DatasetDict, Column
from ...util.module_utils import get_child_module_by_names
from ...util.logit_lens_utils.model_device_handling import get_base_model, get_embedding_device
from ...util.logit_lens_utils.make_layer_names import make_gpt2_layer_names, make_llama_layer_names
from ...util.logit_lens_utils.gpt_hooks import make_gpt_lens_hooks
from ...util.logit_lens_utils.llama_hooks import (
    make_llama_lens_hooks,
    clear_llama_lens_hooks,
    hook_all_submodules,
    hook_selected_submodules,
    make_llama_lens_hooks_sub,
    clear_llama_lens_hooks_sub
)



# -------------------------------
# Helpers
# -------------------------------
def is_quantized_model(model) -> bool:
    return any(isinstance(m, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)) for m in model.modules())


def is_bitnet_model(model) -> bool:
    return any(hasattr(m, "bit_linear") or m.__class__.__name__ == "BitLinear" for m in model.modules())


def detect_architecture(model):
    base_model = getattr(model, "base_model", model)
    if is_bitnet_model(model):
        return "bitnet"
    if hasattr(base_model, "layers") or (hasattr(base_model, "model") and hasattr(base_model.model, "layers")):
        return "llama"
    if hasattr(base_model, "h") or (hasattr(base_model, "transformer") and hasattr(base_model.transformer, "h")):
        return "gpt"
    raise NotImplementedError(f"Cannot detect architecture for {type(model)}")


# -------------------------------
# LogitLensWrapper
# -------------------------------
class LogitLensWrapper:
    """
    Compatible with GPT2, LLaMA and OLMo architectures
    """
    def __init__(
        self,
        model:nn.Module,
        tokenizer:Any,
        block_step:int=1,
        include_input:bool=True,
        force_include_output:bool=True,
        include_subblocks:bool=False,
        decoder_layer_names:Optional[List[str]]=None,
        apply_norm_intermediates: bool = False,
        hook_mode: str = "blocks", # "blocks", "subblocks", "all", "selected"
        lens_variant: str = "raw", # "raw", "final", "all", "per_layer", "per_layer", "double"
        device:str="cuda",
        max_len: Optional[int ]= 32,
    ) -> None:
        """
        lens_variant Options:
        - "raw":        No normalization (pure pre-norm probing)
        - "final":      Only apply final RMSNorm at output
        - "all":        Apply model.model.norm (global RMSNorm) to all layers
        - "per_layer":  Apply each block's internal RMSNorm (input/post-attn)
        - "double":     Apply both per-layer and final RMSNorm
        
        hook_mode: "blocks", # "blocks", "subblocks", "all", "selected"
        """
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        
        # --- Ensure tokenizer has a pad token ---
        self._ensure_special_tokens()

        self.device = device
        if decoder_layer_names is None:
            self.decoder_layer_names = ["norm", "lm_head"]
        else:
            self.decoder_layer_names = decoder_layer_names

        self.apply_norm_intermediates = apply_norm_intermediates
        self.lens_variant = lens_variant
        self.max_len = max_len
        self.is_quantized = is_quantized_model(model)
        self.is_bitnet = is_bitnet_model(model)
        self.arch = detect_architecture(model)

        self.model = model if (self.is_quantized or self.is_bitnet) else model.to(device)
        self.hook_mode = hook_mode

        if self.arch == "gpt":
            self.make_layer_names_fn = make_gpt2_layer_names
            self.make_hooks_fn = make_gpt_lens_hooks
        else:
            self.make_layer_names_fn = make_llama_layer_names
            self.make_hooks_fn = make_llama_lens_hooks

        self.layer_names = self.make_layer_names_fn(
            model=self.model,
            block_step=block_step,
            include_input=include_input,
            force_include_output=force_include_output,
            include_subblocks=include_subblocks,
            #decoder_layer_names=self.decoder_layer_names
            decoder_layer_names=self.decoder_layer_names
        )

        self.base_model = get_base_model(self.model)
        self.module = get_child_module_by_names(
            self.base_model,
            ["layers"] if self.arch in ["llama", "bitnet"] else ["h"]
        )

        self.handles = []
        self.hooks = {}


    def _ensure_special_tokens(self):
        specials = {}
        if not getattr(self.tokenizer, "bos_token", None):
            specials["bos_token"] = "<s>"
        if not getattr(self.tokenizer, "eos_token", None):
            specials["eos_token"] = "</s>"
        if not getattr(self.tokenizer, "pad_token", None):
            if hasattr(self.tokenizer, "eos_token") and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                specials["pad_token"] = "[PAD]"
        if specials:
            self.tokenizer.add_special_tokens(specials)
            if hasattr(self.model, "resize_token_embeddings"):
                self.model.resize_token_embeddings(len(self.tokenizer))

    def extract_texts(
        dataset_split, 
        query_key: str = "question", 
        answer_key: str = None, 
        concat_query_answer: bool = False
    ) -> list[str]:
        """
        Extracts texts from a Hugging Face dataset split.
        By default only returns query. If answer_key provided,
        can return query+answer concatenated if concat_query_answer=True.
        """
        queries = dataset_split[query_key]
        if answer_key and concat_query_answer:
            answers = dataset_split[answer_key]
            texts = [f"{q} {a}" for q, a in zip(queries, answers)]
        else:
            texts = list(queries)  # just queries
        return texts
    
    # ----------------------------
    # Tokenize
    # ----------------------------
    def tokenize_inputs(
        self,
        texts: list[str],
        add_special_tokens: bool = False,
        add_bos: bool = True,   # <-- LLaMA: always prepend BOS
        add_eos: bool = False,  # <-- optional
        max_len: int = 24,
        pad_to_max_length: bool = False,
        move_to_device: bool = True,
    ):
        """
        Tokenize inputs for LLaMA-style models (Instruct variants).
        Adds BOS (<s>) by default, uses EOS (</s>) for padding.
        """

        tokenizer = self.tokenizer
        model = self.model

        # Ensure pad_token exists
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        if isinstance(texts, str):
            texts = [texts]

        input_ids = [tokenizer.encode(t, add_special_tokens=False) for t in texts]

        # Always prepend BOS for LLaMA
        if add_bos and tokenizer.bos_token_id is not None:
            input_ids = [[tokenizer.bos_token_id] + ids for ids in input_ids]

        # Append EOS if requested
        if add_eos and tokenizer.eos_token_id is not None:
            input_ids = [ids + [tokenizer.eos_token_id] for ids in input_ids]

        # Pad/truncate
        max_len_tensor = max_len if pad_to_max_length else max(len(ids) for ids in input_ids)
        input_ids_padded = []
        for ids in input_ids:
            if len(ids) > max_len_tensor:
                ids = ids[:max_len_tensor]
            else:
                ids = ids + [tokenizer.pad_token_id] * (max_len_tensor - len(ids))
            input_ids_padded.append(ids)

        input_ids_tensor = torch.tensor(input_ids_padded, dtype=torch.long)

        if move_to_device and model is not None:
            device = next(model.parameters()).device
            input_ids_tensor = input_ids_tensor.to(device)

        attention_mask = (input_ids_tensor != tokenizer.pad_token_id).long()

        return {"input_ids": input_ids_tensor, "attention_mask": attention_mask}


    

    # ----------------------------
    # Hook management
    # ----------------------------
    def add_hooks_sub(self):
        self.clear_hooks()

        if self.hook_mode == "blocks":
            self.make_hooks_fn = make_llama_lens_hooks
            self.make_hooks_fn(model=self.model, layer_names=self.layer_names,
                            decoder_layer_names=self.decoder_layer_names)

        elif self.hook_mode == "subblocks":
            self.make_hooks_fn = make_llama_lens_hooks_sub
            self.make_hooks_fn(self.model, verbose=True)

        elif self.hook_mode == "all":
            self._hook_handles = hook_all_submodules(
                self.model, self.model._layer_logits, capture_weights=False
            )

        elif self.hook_mode == "selected":
            self._hook_handles = hook_selected_submodules(
                self.model, self.model._layer_logits,
                include_patterns=self.include_patterns,
                exclude_patterns=self.exclude_patterns,
                capture_weights=False
            )

        else:
            raise ValueError(f"Unknown hook_mode={self.hook_mode}")

        self.make_hooks_fn(
            model=self.model,
            layer_names=self.layer_names,
            decoder_layer_names=self.decoder_layer_names
        )

    def add_hooks(self):
        self.clear_hooks()
        self.make_hooks_fn(
            model=self.model,
            layer_names=self.layer_names,
            decoder_layer_names=self.decoder_layer_names
        )

    def clear_hooks(self):
        if hasattr(self.model, "_layer_logits_handles"):
            for handle in self.model._layer_logits_handles.values():
                handle.remove()
            self.model._layer_logits_handles.clear()
        if hasattr(self.model, "_layer_logits"):
            self.model._layer_logits.clear()

    def _record_activation_hook(self, name, model):
        def hook(module, inp, out):
            model = self.model
            model._layer_logits[name] = out.detach()
        return hook

    
    def apply_norm(self, h, lname, model, lens_variant: str = "raw"):
        """
        Apply normalization depending on the chosen lens variant.

        Options:
        - "raw":        No normalization (pure pre-norm probing)
        - "final":      Only apply final RMSNorm at output
        - "all":        Apply model.model.norm (global RMSNorm) to all layers
        - "per_layer":  Apply each block's internal RMSNorm (input/post-attn)
        - "double":     Apply both per-layer and final RMSNorm
        """
        # --- RAW ---
        if lens_variant == "raw":
            return h

        # --- PER-LAYER NORMALIZATION ---
        if lens_variant in {"per_layer", "double"} and lname.startswith("layers."):
            block_id = int(lname.split(".")[1])
            block = model.model.layers[block_id]

            if "input_layernorm" in lname:
                return block.input_layernorm(h)
            elif "post_attention_layernorm" in lname:
                return block.post_attention_layernorm(h)
            elif lname.endswith(".out"):
                # usually pre-norm residual output, skip additional norm
                return h

        # --- GLOBAL NORMALIZATION ---
        if lens_variant in {"final", "all", "double"}:
            # final RMSNorm — skip if already normalized
            if lname in {"final_norm.out", "output"}:
                return h
            else:
                return model.model.norm(h)

        # --- fallback ---
        return h

    # ----------------------------
    # Forward (replace your forward method)
    # ----------------------------
    def forward(
        self,
        texts: List[str],
        project_to_logits: bool = True,
        return_hidden: bool = False,
        project_subblocks: bool = False,
        decoder: Optional[Any] = None,
        keep_on_device: bool = True,
        project_on_lm_head_dtype: bool = True,
        force_dtype: Optional[torch.dtype] = None,
        add_special_tokens: bool = False,
        pad_to_max_length: bool = False,
        max_len: int = 24,
        #lens_variant: str = "raw",   # <-- NEW: "raw", "final", "all", "per_layer", or "double"
    ) -> Tuple[Any, Dict[str, torch.Tensor], List[str]]:

        # --- Tokenize ---
        inputs = self.tokenize_inputs(
            texts=texts,
            add_special_tokens=add_special_tokens,
            max_len=max_len,
            pad_to_max_length=pad_to_max_length,
            move_to_device=True,
        )

        model_device = next(self.model.parameters()).device
        self.add_hooks_sub()

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
                output_attentions=True
            )

        # --- Collect activations ---
        activations = {}
        if hasattr(self.model, "_layer_logits") and self.model._layer_logits:
            activations = dict(self.model._layer_logits)

        last_block_name = f"layers.{len(self.model.model.layers)-1}"

        # --- Build layer_dict ---
        layer_dict, names = {}, list(activations.keys())

        for lname in names:
            h = activations[lname]

            # Convert to tensor (safe)
            if not isinstance(h, torch.Tensor):
                try:
                    h = torch.tensor(h, device=model_device)
                except Exception:
                    layer_dict[lname] = h
                    continue

            # --- Apply normalization variant (your new method) ---
            h = self.apply_norm(h, lname, self.model, self.lens_variant)

            # --- Projection logic ---
            if lname == last_block_name + ".out":
                # last transformer block
                if project_to_logits:
                    lm_head = getattr(self.model, "lm_head", None)
                    hidden_size = getattr(self.model.config, "hidden_size", None)
                    if lm_head and hidden_size and h.shape[-1] == hidden_size:
                        lm_dev = next(lm_head.parameters()).device
                        lm_dtype = (next(lm_head.parameters()).dtype
                                    if project_on_lm_head_dtype else h.dtype)
                        h_proj = h.to(lm_dev, dtype=lm_dtype, copy=False)
                        out = lm_head(h_proj)
                    else:
                        out = h
                else:
                    out = h
            else:
                # intermediate layers or subblocks
                if return_hidden:
                    out = h
                elif project_to_logits or (project_subblocks and "layers." in lname):
                    lm_head = getattr(self.model, "lm_head", None)
                    hidden_size = getattr(self.model.config, "hidden_size", None)
                    if lm_head and hidden_size and h.shape[-1] == hidden_size:
                        lm_dev = next(lm_head.parameters()).device
                        lm_dtype = (next(lm_head.parameters()).dtype
                                    if project_on_lm_head_dtype else h.dtype)
                        h_proj = h.to(lm_dev, dtype=lm_dtype, copy=False)
                        out = lm_head(h_proj)
                    else:
                        out = h
                else:
                    out = h

            # --- Move/cast ---
            if keep_on_device:
                layer_dict[lname] = out
            else:
                out = out.detach().cpu()
                if force_dtype is not None:
                    out = out.to(force_dtype)
                layer_dict[lname] = out

        # --- True output (final RMSNorm + lm_head) ---
        with torch.no_grad():
            h_last = activations[last_block_name + ".out"]   # pre-norm
            h_norm = self.model.model.norm(h_last)           # final RMSNorm
            out_true = self.model.lm_head(h_norm)

            if not keep_on_device:
                out_true = out_true.detach().cpu()
                if force_dtype is not None:
                    out_true = out_true.to(force_dtype)

            layer_dict["output"] = out_true
            names.append("output")

        return outputs, layer_dict, names, inputs["input_ids"]


    # ---------------------------
    # Collect logits from layer dict (stacked)
    # ---------------------------
    def stack_layer_logits(
        self,
        layer_dict: dict,
        keep_on_device: bool = True,
        filter_layers: bool = False,
        compare_variants: bool = False,   # <-- NEW: compare raw vs normalized (if both exist)
        variant_suffixes: Tuple[str, str] = (".out", ".out.normed"),
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Collects logits from layer_dict and returns stacked tensors + names.
        Can also compare 'raw' vs 'normalized' variants if present.
        """
        stacked_logits, valid_names = [], []

        for lname, tensor in layer_dict.items():
            # Skip embedding/input layers if requested
            if filter_layers and lname in {"input", "embed_tokens"}:
                continue

            # If comparing variants, try to pair raw and normalized versions
            if compare_variants and lname.endswith(variant_suffixes[0]):
                base = lname[:-len(variant_suffixes[0])]
                normed_name = base + variant_suffixes[1]

                if normed_name in layer_dict:
                    # Stack both versions side-by-side for comparison
                    raw_tensor = tensor
                    norm_tensor = layer_dict[normed_name]

                    if not keep_on_device:
                        raw_tensor = raw_tensor.detach().cpu()
                        norm_tensor = norm_tensor.detach().cpu()

                    # Shape: [2, ...] to distinguish raw vs normed
                    stacked_logits.append(torch.stack([raw_tensor, norm_tensor], dim=0))
                    valid_names.append(base + " (raw vs normed)")
                    continue

            # Default case: single tensor
            l = tensor
            if not keep_on_device:
                l = l.detach().cpu()

            stacked_logits.append(l)
            valid_names.append(lname)

        return stacked_logits, valid_names


    # ---------------------------
    # Full logit collection (from input texts to stacked logits)
    # ---------------------------
    def collect_stacked_logits(
        self,
        texts: List[str],
        layer_names: List[str],
        add_special_tokens: bool = False,
        add_bos: bool = True,
        add_eos: bool = True,
        pad_to_max_len : bool = False,
        max_len: int = 24,
    ):
        # Tokenize inputs exactly like in heatmap
        inputs = self.tokenize_inputs(
            texts,
            add_special_tokens=add_special_tokens,
            add_bos=add_bos,
            add_eos=add_eos,
            max_len=max_len,
            pad_to_max_length=pad_to_max_len
        )

        # Run model forward pass
        self.model._last_resid = None
        with torch.no_grad():
            _ = self.model(
                inputs["input_ids"],
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )
        self.model._last_resid = None

        # Collect stacked logits
        logits_dict = self.model._layer_logits  # must be populated by hooks as in heatmap
        stacked_logits, valid_layer_names = self.stack_layer_logits(logits_dict)

        return stacked_logits, valid_layer_names, inputs
