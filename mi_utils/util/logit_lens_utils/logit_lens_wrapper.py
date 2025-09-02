from typing import List, Optional, Any, Dict, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
import bitsandbytes as bnb
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
        device:str="cuda"
    ) -> None:
        self.tokenizer = tokenizer
        self.device = device
        self.decoder_layer_names = decoder_layer_names or ["lm_head"]

        # Detect architecture and quantization
        self.is_quantized = is_quantized_model(model)
        self.is_bitnet = is_bitnet_model(model)
        self.arch = detect_architecture(model)

        # Move model to device if not quantized
        self.model = model if (self.is_quantized or self.is_bitnet) else model.to(device)

        # Select layer/hook functions
        if self.arch == "gpt":
            self.make_layer_names_fn = make_gpt2_layer_names
            self.make_hooks_fn = make_gpt_lens_hooks
        else:
            self.make_layer_names_fn = make_llama_layer_names
            self.make_hooks_fn = make_llama_lens_hooks

        # Generate layer names
        self.layer_names = self.make_layer_names_fn(
            model=self.model,
            block_step=block_step,
            include_input=include_input,
            force_include_output=force_include_output,
            include_subblocks=include_subblocks,
            decoder_layer_names=self.decoder_layer_names
        )

        # Base model and transformer container
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


    def tokenize_inputs(self, tokenizer, texts: list[str], model=None, add_special_tokens=True):
        """
        Tokenize a batch of strings into input IDs tensor.
        Handles padding/truncation and moves to model device.
        """
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token

        # Tokenize batch
        inputs = tokenizer(
            texts,                        # <-- batch of strings
            return_tensors="pt",           # PyTorch tensor
            padding=True,                  # pad sequences to max length
            truncation=True,               # truncate sequences too long
            add_special_tokens=add_special_tokens
        )

        # Move tensors to model device
        device = next(model.parameters()).device if model else torch.device("cpu")
        input_ids = inputs["input_ids"].to(device)

        return input_ids  # shape: [batch_size, seq_len]

    # ----------------------------
    # Forward (replace your forward method)
    # ----------------------------
    def forward(
        self,
        texts: List[str],
        project_to_logits: bool = True,
        return_hidden: bool = False,
        decoder: Optional[Any] = None,
        add_bos: bool = False,
        add_eos: bool = True,
        # new options:
        keep_on_device: bool = True,   # don't move activations to CPU unless False
        project_on_lm_head_dtype: bool = True  # run lm_head in its dtype/device
    ) -> Tuple[Any, Dict[str, torch.Tensor], List[str]]:
        """
        Forward that preserves dtype/device. Returns layer tensors on-device by default.

        If project_to_logits (and lm_head exists), we will project by moving the
        activation tensor to lm_head's device/dtype temporarily (not moving module).
        """
        # ensure pad token
        if self.tokenizer.pad_token is None:
            if getattr(self.tokenizer, "eos_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # Tokenize -> dict tensors
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        # note: do not change dtype here
        # move tensors to model device
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        # register hooks (external)
        self.add_hooks()

        # run model
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # collect activations (choose the store used by your hooks)
        activations = {}
        if hasattr(self.model, "_layer_logits") and getattr(self.model, "_layer_logits"):
            activations = dict(self.model._layer_logits)
        elif hasattr(self.model, "_layer_hidden") and getattr(self.model, "_layer_hidden"):
            activations = dict(self.model._layer_hidden)
        elif hasattr(self, "layer_hidden_store") and getattr(self, "layer_hidden_store"):
            activations = dict(self.layer_hidden_store)
        else:
            return outputs, {}, []

        layer_dict = {}
        names = list(activations.keys())

        for lname in names:
            h = activations[lname]
            # ensure tensor
            if not isinstance(h, torch.Tensor):
                try:
                    h = torch.tensor(h, device=model_device)
                except Exception:
                    # leave as-is if conversion fails
                    layer_dict[lname] = h
                    continue

            # if user asked raw hidden or not projecting
            if return_hidden or not project_to_logits:
                layer_dict[lname] = h if keep_on_device else h.detach().cpu()
                continue

            # project via decoder or lm_head
            if decoder is not None:
                proj = decoder(h)  # assume decoder handles dtype/device
                layer_dict[lname] = proj if keep_on_device else proj.detach().cpu()
                continue

            lm_head = getattr(self.model, "lm_head", None)
            hidden_size = getattr(self.model.config, "hidden_size", getattr(self.model.config, "n_embd", None))
            if lm_head is None or hidden_size is None or h.shape[-1] != hidden_size:
                # cannot project -> return hidden
                layer_dict[lname] = h if keep_on_device else h.detach().cpu()
                continue

            # Project on lm_head device/dtype by moving ACTIVATION (NOT the module)
            try:
                lm_dev = next(lm_head.parameters()).device
                lm_dtype = next(lm_head.parameters()).dtype if project_on_lm_head_dtype else h.dtype

                # temporary tensor move & dtype conversion
                h_for_proj = h.to(device=lm_dev, dtype=lm_dtype, copy=False)
                with torch.no_grad():
                    projected = lm_head(h_for_proj)  # result on lm_dev and lm_dtype
                # keep projected as tensor on device OR move to cpu based on keep_on_device
                layer_dict[lname] = projected if keep_on_device else projected.detach().cpu()
            except Exception:
                # fallback: try to run lm_head on the model device (may cast types)
                try:
                    projected = lm_head(h.to(lm_head.weight.device))
                    layer_dict[lname] = projected if keep_on_device else projected.detach().cpu()
                except Exception:
                    # final fallback: return hidden
                    layer_dict[lname] = h if keep_on_device else h.detach().cpu()

        return outputs, layer_dict, names


    # pseudo-code inside wrapper
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


    # ----------------------------
    # Collect Layer Logits
    # ----------------------------
    def collect_layer_logits(
        self,
        input_ids: Union[torch.Tensor, dict],
        start_ix: Optional[int] = None,
        end_ix: Optional[int] = None,
        include_embeddings: bool = False,
        verbose: bool = False
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Collect layer activations or projected logits.
        Returns tensors on the device they were computed.
        """
        # Normalize inputs
        if isinstance(input_ids, torch.Tensor):
            inputs = {"input_ids": input_ids}
        elif isinstance(input_ids, dict):
            inputs = input_ids.copy()
        else:
            raise ValueError("input_ids must be tensor or dict")

        device = getattr(self.model, "device", None)
        if device is None:
            try:
                device = next(self.model.parameters()).device
            except Exception:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        self.model.eval()
        with torch.no_grad():
            _ = self.model(**inputs, output_hidden_states=True)

        # Get store
        if hasattr(self.model, "_layer_logits") and self.model._layer_logits:
            store = self.model._layer_logits
        elif hasattr(self.model, "_layer_hidden") and self.model._layer_hidden:
            store = self.model._layer_hidden
        elif hasattr(self, "layer_hidden_store") and self.layer_hidden_store:
            store = self.layer_hidden_store
        else:
            if verbose:
                print("No hidden states captured")
            return [], []

        fired_names = list(store.keys())
        if hasattr(self, "layer_names") and self.layer_names:
            canonical = [n for n in self.layer_names if n in fired_names]
            extras = [n for n in fired_names if n not in canonical]
            fired_names = canonical + extras

        # Collect raw tensors only
        logits_list = []
        for lname in fired_names:
            hidden = store[lname]  # expect [B,S,H] or [B,S,V]
            if isinstance(hidden, np.ndarray):
                hidden = torch.from_numpy(hidden)
            # slice
            s = 0 if start_ix is None else start_ix
            e = hidden.shape[1] if end_ix is None else end_ix + 1
            hidden = hidden[:, s:e, :]
            logits_list.append(hidden)

        return logits_list, fired_names


    def collect_layer_logits_stacked(self, input_ids, keep_on_device=True):
        """
        Collect logits from all layers, stacked as a list of tensors.
        Returns tensors on the device they were computed if keep_on_device=True.
        """
        _, logits_by_layer, layer_names = self.forward(
            input_ids, project_to_logits=True, return_hidden=False, keep_on_device=keep_on_device
        )

        stacked_logits = []
        valid_names = []
        for lname in layer_names:
            l = logits_by_layer.get(lname)
            if l is None:
                continue
            if not isinstance(l, torch.Tensor):
                try:
                    l = torch.tensor(l, device=input_ids.device)
                except Exception:
                    continue

            # If user wants CPU, move and detach
            if not keep_on_device:
                l = l.detach().cpu()

            # Ensure shape: [seq_len, vocab_size] for single batch
            if l.ndim == 3 and l.shape[0] == 1:
                l = l[0]

            stacked_logits.append(l)
            valid_names.append(lname)

        if len(stacked_logits) == 0:
            raise RuntimeError("No valid logits found")

        return stacked_logits, valid_names  # keep as list of tensors on-device
    

    def collect_batch_logits(self, model, input_ids, layer_names) -> Tuple[np.ndarray, list[str]]:
        collected_logits = []

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]  # Take hidden states
            with torch.no_grad():
                collected_logits.append(output.detach().cpu().numpy())  # [batch, seq_len, hidden]

        handles = []
        for name in layer_names:
            layer = dict(model.named_modules())[name]
            handles.append(layer.register_forward_hook(hook_fn))

        with torch.no_grad():
            model(input_ids, output_hidden_states=True, return_dict=True)

        for h in handles:
            h.remove()

        # collected_logits: list of [batch, seq_len, hidden] for each layer
        # Stack: [num_layers, batch, seq_len, hidden]
        logits = np.stack(collected_logits, axis=0)
        return logits, layer_names