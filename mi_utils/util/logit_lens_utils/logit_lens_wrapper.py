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
from ...util.logit_lens_utils.llama_hooks import make_llama_lens_hooks



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
        include_subblocks:bool=True,
        decoder_layer_names:Optional[List[str]]=None,
        device:str="cuda",
        max_len: Optional[int ]= 32,
    ) -> None:
        
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self.model.resize_token_embeddings(len(self.tokenizer))

        self.device = device
        self.decoder_layer_names = decoder_layer_names or ["lm_head"]
        self.max_len = max_len
        self.is_quantized = is_quantized_model(model)
        self.is_bitnet = is_bitnet_model(model)
        self.arch = detect_architecture(model)

        self.model = model if (self.is_quantized or self.is_bitnet) else model.to(device)

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
            decoder_layer_names=self.decoder_layer_names
        )

        self.base_model = get_base_model(self.model)
        self.module = get_child_module_by_names(
            self.base_model,
            ["layers"] if self.arch in ["llama", "bitnet"] else ["h"]
        )

        self.handles = []
        self.hooks = {}


    # ----------------------------
    # Hook management
    # ----------------------------
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

    
    def tokenize_inputs(
        self,
        texts: List[str],
        add_special_tokens: bool = False,  # default OFF for probing
        add_bos: bool = False,
        add_eos: bool = False,
        max_len: int = 24,
        pad_to_max_length: bool = False,
        move_to_device: bool = True,
    ):
        tokenizer = self.tokenizer
        model = self.model

        processed = []
        for t in texts:
            if add_bos and tokenizer.bos_token is not None and not t.startswith(tokenizer.bos_token):
                t = tokenizer.bos_token + t
            if add_eos and tokenizer.eos_token is not None and not t.endswith(tokenizer.eos_token):
                t = t + tokenizer.eos_token
            processed.append(t)

        inputs = tokenizer(
            processed,
            return_tensors="pt",
            padding="longest" if not pad_to_max_length else True,  # match heatmap
            truncation=True,
            max_length=max_len,
            add_special_tokens=add_special_tokens,
        )

        if move_to_device:
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

        return inputs



    # ----------------------------
    # Forward (replace your forward method)
    # ----------------------------
    def forward(
        self,
        texts: List[str],
        project_to_logits: bool = True,
        return_hidden: bool = False,
        decoder: Optional[Any] = None,
        keep_on_device: bool = True,
        project_on_lm_head_dtype: bool = True,
        force_dtype: Optional[torch.dtype] = None,
        add_special_tokens: bool = False,  # default OFF for probing
        add_bos: bool = False,             # handle manually if needed
        add_eos: bool = False,
        pad_to_max_length: bool = False, 
        max_len: int = 316,
    ) -> Tuple[Any, Dict[str, torch.Tensor], List[str]]:
        """
        Forward that preserves dtype/device. Returns layer tensors on-device by default.
        """
        inputs = self.tokenize_inputs(
            texts=texts,
            add_special_tokens=add_special_tokens,
            add_bos=add_bos,
            add_eos=add_eos,
            max_len=max_len,
            pad_to_max_length=pad_to_max_length,
            move_to_device=True,
        )

        model_device = next(self.model.parameters()).device
        self.add_hooks()

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        activations = {}
        if hasattr(self.model, "_layer_logits") and self.model._layer_logits:
            activations = dict(self.model._layer_logits)
        elif hasattr(self.model, "_layer_hidden") and self.model._layer_hidden:
            activations = dict(self.model._layer_hidden)
        elif hasattr(self, "layer_hidden_store") and self.layer_hidden_store:
            activations = dict(self.layer_hidden_store)

        layer_dict, names = {}, list(activations.keys())
        for lname in names:
            h = activations[lname]
            if not isinstance(h, torch.Tensor):
                try:
                    h = torch.tensor(h, device=model_device)
                except Exception:
                    layer_dict[lname] = h
                    continue

            if return_hidden or not project_to_logits:
                out = h
            elif decoder is not None:
                out = decoder(h)
            else:
                lm_head = getattr(self.model, "lm_head", None)
                hidden_size = getattr(self.model.config, "hidden_size", getattr(self.model.config, "n_embd", None))
                if lm_head and hidden_size and h.shape[-1] == hidden_size:
                    try:
                        lm_dev = next(lm_head.parameters()).device
                        lm_dtype = next(lm_head.parameters()).dtype if project_on_lm_head_dtype else h.dtype
                        h_proj = h.to(lm_dev, dtype=lm_dtype, copy=False)
                        out = lm_head(h_proj)
                    except Exception:
                        out = h  
                else:
                    out = h

            # device/dtype control
            if keep_on_device:
                layer_dict[lname] = out
            else:
                out = out.detach().cpu()
                if force_dtype is not None:
                    out = out.to(force_dtype)
                layer_dict[lname] = out

        return outputs, layer_dict, names


    def stack_layer_logits(self, layer_dict, keep_on_device=True, filter_layers=True):
        stacked_logits, valid_names = [], []
        for lname, tensor in layer_dict.items():
            if filter_layers and lname in {"input", "embed_tokens"}:
                continue
            l = tensor
            if not keep_on_device:
                l = l.detach().cpu()
            stacked_logits.append(l)
            valid_names.append(lname)
        return stacked_logits, valid_names
