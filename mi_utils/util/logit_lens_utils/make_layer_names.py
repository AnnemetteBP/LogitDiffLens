import torch
from torch import nn
from ...util.module_utils import get_child_module_by_names


def make_gpt2_layer_names(
    model,
    block_step:int=1,
    include_input:bool=True,
    force_include_output:bool=True,
    include_subblocks:bool=False,
    decoder_layer_names:list[str]=['final_layernorm', 'lm_head']
):
    h = get_child_module_by_names(model.base_model, ["h"])
    h_names = [f"h.{i}" for i in range(len(h))]

    last_h_name = h_names[-1]

    h_names = h_names[::block_step]
    if force_include_output and last_h_name not in h_names:
        h_names.append(last_h_name)

    if include_subblocks:
        names = [sub_name for name in h_names for sub_name in (f"{name}.attn", f"{name}.mlp", name)]

    else:
        names = h_names

    if include_input:
        names = ["input"] + names

    def _subset(a, b):
        return a == b or a.startswith(b + ".")

    def _names_overlap(a, b):
        return _subset(a, b) or _subset(b, a)

    names = [name for name in names
             if not any([_names_overlap(name, dname) for dname in decoder_layer_names])
             ]

    return names


def make_llama_layer_names(
    model,
    block_step:int=1,
    include_input:bool=True,
    force_include_output:bool=True,
    include_subblocks:bool=False,
    decoder_layer_names:list[str]=['norm', 'lm_head']  
):
    # Fix: OLMo OLMos and LLaMAs stores transformer layers under "model.layers", not "base_model.h"
    h = get_child_module_by_names(model.base_model, ["layers"])
    h_names = [f"layers.{i}" for i in range(len(h))]

    last_layer_name = h_names[-1]

    # Apply block stepping
    h_names = h_names[::block_step]
    if force_include_output and last_layer_name not in h_names:
        h_names.append(last_layer_name)

    # Include attention and MLP sublayers if specified
    if include_subblocks:
        names = [
            sub_name
            for name in h_names
            for sub_name in (f"{name}.self_attn", f"{name}.mlp", name)
        ]
    else:
        names = h_names

    # Optionally include input layer
    if include_input:
        names = ["embed_tokens"] + names  # Changed "input" → "embed_tokens" for OLMo / LLaMA architectures

    # Remove decoder layers from the names list
    def _subset(a, b):
        return a == b or a.startswith(b + ".")

    def _names_overlap(a, b):
        return _subset(a, b) or _subset(b, a)

    names = [name for name in names if not any([_names_overlap(name, dname) for dname in decoder_layer_names])]

    return names
