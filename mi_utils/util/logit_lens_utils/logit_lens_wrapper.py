from typing import List, Optional, Any, Dict, Tuple
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
        model: nn.Module,
        tokenizer,
        block_step: int = 1,
        include_input: bool = True,
        force_include_output: bool = True,
        include_subblocks: bool = True,
        decoder_layer_names: Optional[List[str]] = None,
        device: str = "cuda"
    ):
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

    # ----------------------------
    # Forward pass
    # ----------------------------
    def forward(
            self,
            texts:List[str],
            project_to_logits:
            bool=True,
            decoder:Optional[Any]=None
        ) -> Tuple[Any, Dict[str, torch.Tensor], List[str]]:

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        device = get_embedding_device(self.model)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        self.add_hooks()

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        logits_by_layer = {}
        if project_to_logits and hasattr(self.model, "_layer_logits"):
            for lname, h in self.model._layer_logits.items():
                # Keep tensor on device and dtype intact
                h_tensor = h if isinstance(h, torch.Tensor) else torch.tensor(h, device=device)

                # Optional projection
                if decoder is not None:
                    logits_by_layer[lname] = decoder(h_tensor)
                else:
                    lm_head = getattr(self.model, "lm_head", None)
                    hidden_size = getattr(self.model.config, "hidden_size", getattr(self.model.config, "n_embd", None))
                    if lm_head is not None and hidden_size is not None and h_tensor.shape[-1] == hidden_size:
                        logits_by_layer[lname] = lm_head(h_tensor)
                    else:
                        logits_by_layer[lname] = h_tensor

        return outputs, logits_by_layer, list(getattr(self.model, "_layer_logits", {}).keys())
