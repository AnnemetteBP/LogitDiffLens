import torch
import torch.nn as nn

from ...util.python_utils import make_print_if_verbose
from ...util.module_utils import get_child_module_by_names


_RESID_SUFFIXES = {".self_attn", ".mlp"}


def llama_blocks_input_locator(model: nn.Module):
    """
    Identifies the input to the transformer blocks.
    """
    return lambda: model.embed_tokens 

def llama_final_layernorm_locator(model: nn.Module):
    """
    Identifies the final normalization layer.
    """
    if hasattr(model.base_model, "norm"):
        return lambda: model.base_model.norm
    else:
        raise ValueError("Could not identify final layer norm")

def _locate_special_modules(model):
    if not hasattr(model, "_blocks_input_getter"):
        model._blocks_input_getter = llama_blocks_input_locator(model)

    if not hasattr(model, "_ln_f_getter"):
        model._ln_f_getter = llama_final_layernorm_locator(model)

def _get_layer(model, name):
    if name == "input":
        return model._blocks_input_getter()
    if name == "norm":
        return model._ln_f_getter()

    model_with_module = model if name == "lm_head" else model.base_model
    return get_child_module_by_names(model_with_module, name.split("."))

def _sqz(x):
    if isinstance(x, torch.Tensor):
        return x
    try:
        return x[0]
    except:
        return x

def _get_layer_and_compose_with_ln(model, name):
    if name.endswith('.self_attn'):
        lname = name[:-len('.self_attn')] + '.input_layernorm'
        ln = _get_layer(model, lname)
    elif name.endswith('.mlp'):
        lname = name[:-len('.mlp')] + '.post_attention_layernorm'
        ln = _get_layer(model, lname)
    else:
        ln = lambda x: x
    return lambda x: _get_layer(model, name)(ln(x))

def make_llama_decoder(model, decoder_layer_names=['norm', 'lm_head']): 
    _locate_special_modules(model)

    decoder_layers = [_get_layer_and_compose_with_ln(model, name) for name in decoder_layer_names]

    def _decoder(x):
        for name, layer in zip(decoder_layer_names, decoder_layers):
            layer_out = _sqz(layer(_sqz(x)))

            is_resid = any([name.endswith(s) for s in _RESID_SUFFIXES])
            if is_resid:
                x = x + layer_out
            else:
                x = layer_out
        return x
    return _decoder

def make_llama_lens_hooks(
    model,
    layer_names: list,
    decoder_layer_names: list = ['norm', 'lm_head'],  
    verbose=True,
    start_ix=None,
    end_ix=None,
):
    vprint = make_print_if_verbose(verbose)

    clear_llama_lens_hooks(model)


    def _opt_slice(x, start_ix, end_ix):
        if start_ix is None:
            start_ix = 0
        if end_ix is None:
            end_ix = x.shape[1]
        return x[:, start_ix:end_ix, :]

    _locate_special_modules(model)

    for attr in ["_layer_logits", "_layer_logits_handles"]:
        if not hasattr(model, attr):
            setattr(model, attr, {})

    model._ordered_layer_names = layer_names
    model._lens_decoder = make_llama_decoder(model, decoder_layer_names)
    
    def _make_record_logits_hook(name):
        model._layer_logits[name] = None
        is_resid = any([name.endswith(s) for s in _RESID_SUFFIXES])

        def _record_logits_hook(module, input, output) -> None:
            del model._layer_logits[name]

            if is_resid:
                resid_out = model._last_resid + _sqz(output)  # add residual
            else:
                resid_out = _sqz(output)

            model._layer_logits[name] = resid_out   # <-- store raw hidden
            model._last_resid = resid_out           # <-- update tracker

        return _record_logits_hook



    """def _make_record_logits_hook(name):
        model._layer_logits[name] = None

        is_resid = any([name.endswith(s) for s in _RESID_SUFFIXES])

        def _record_logits_hook(module, input, output) -> None:
            del model._layer_logits[name]
            ln_f = model._ln_f_getter()

            if is_resid:
                decoder_in = model._last_resid + _sqz(output)
            else:
                decoder_in = _sqz(output)

            decoder_out = model._lens_decoder(decoder_in)
            decoder_out = _opt_slice(decoder_out, start_ix, end_ix)

            #model._layer_logits[name] = decoder_out.cpu().numpy()
            model._layer_logits[name] = decoder_out  # <-- KEEP AS TENSOR
            model._last_resid = decoder_in"""

        #return _record_logits_hook

    def _hook_already_there(name):
        handle = model._layer_logits_handles.get(name)
        if not handle:
            return False
        layer = _get_layer(model, name)
        return handle.id in layer._forward_hooks

    """for name in layer_names:
        if _hook_already_there(name):
            vprint(f"Skipping layer {name}, hook already exists")
            continue
        layer = _get_layer(model, name)
        handle = layer.register_forward_hook(_make_record_logits_hook(name))
        model._layer_logits_handles[name] = handle"""
    
    for name in layer_names:
        # Skip synthetic layers like "output" (computed later in forward)
        if name == "output":
            continue

        if _hook_already_there(name):
            vprint(f"Skipping layer {name}, hook already exists")
            continue

        layer = _get_layer(model, name)
        handle = layer.register_forward_hook(_make_record_logits_hook(name))
        model._layer_logits_handles[name] = handle


def clear_llama_lens_hooks(model):
    if hasattr(model, "_layer_logits_handles"):
        for k, v in model._layer_logits_handles.items():
            v.remove()

        ks = list(model._layer_logits_handles.keys())
        for k in ks:
            del model._layer_logits_handles[k]

def hook_all_submodules(model, store_dict, capture_weights=False, verbose=True):
    """
    Attach forward hooks to every submodule in a LLaMA model.
    Stores both activations (outputs) and optionally weights.
    """
    handles = []

    def _hook_fn(name):
        def fn(module, input, output):
            # Save activations
            store_dict[f"{name}.out"] = output.detach().cpu()

            # Optionally save weights
            if capture_weights and hasattr(module, "weight"):
                store_dict[f"{name}.weight"] = module.weight.detach().cpu()
            if capture_weights and hasattr(module, "bias") and module.bias is not None:
                store_dict[f"{name}.bias"] = module.bias.detach().cpu()
        return fn

    for name, module in model.named_modules():
        # skip large container modules if you only want "leaves"
        if len(list(module.children())) == 0:  
            h = module.register_forward_hook(_hook_fn(name))
            handles.append(h)
            if verbose:
                print(f"Hooked {name}")

    return handles


def hook_selected_submodules(
    model,
    store_dict,
    capture_weights=False,
    include_patterns=None,
    exclude_patterns=None,
    verbose=True,
):
    """
    Attach forward hooks to selected submodules in a HuggingFace model.
    
    Args:
        model: torch.nn.Module (e.g. LLaMA)
        store_dict: dict to populate with activations/weights
        capture_weights: if True, also store weights (use state_dict otherwise)
        include_patterns: list of regex patterns to INCLUDE (e.g. ["layernorm", "mlp"])
        exclude_patterns: list of regex patterns to EXCLUDE
        verbose: print hooked modules
    Returns:
        handles: list of hook handles (remove() them later)
    """
    handles = []

    def matches(name, patterns):
        if patterns is None:
            return False
        return any(re.search(p, name) for p in patterns)

    def _hook_fn(name):
        def fn(module, input, output):
            store_dict[f"{name}.out"] = output.detach().cpu()

            if capture_weights and hasattr(module, "weight"):
                store_dict[f"{name}.weight"] = module.weight.detach().cpu()
            if capture_weights and hasattr(module, "bias") and module.bias is not None:
                store_dict[f"{name}.bias"] = module.bias.detach().cpu()
        return fn

    for name, module in model.named_modules():
        # Skip containers
        if len(list(module.children())) > 0:
            continue

        # Filtering logic
        if include_patterns and not matches(name, include_patterns):
            continue
        if exclude_patterns and matches(name, exclude_patterns):
            continue

        # Attach hook
        h = module.register_forward_hook(_hook_fn(name))
        handles.append(h)

        if verbose:
            print(f"Hooked {name}")

    return handles

def make_llama_lens_hooks_sub(
    model,
    verbose=True,
    decoder_layer_names=None,
    start_ix=None,
    end_ix=None,
):
    """
    Attach forward hooks to all submodules in every LlamaDecoderLayer.
    Stores activations in model._layer_logits under qualified names.
    """

    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    clear_llama_lens_hooks(model)

    if not hasattr(model, "_layer_logits"):
        model._layer_logits = {}
    if not hasattr(model, "_layer_logits_handles"):
        model._layer_logits_handles = {}

    # Utility: register hook and save output
    def _record(name):
        def hook(module, inp, out):
            try:
                model._layer_logits[name] = out.detach()
                if hasattr(model.model, "norm"):
                    model._layer_logits[name + ".normed"] = model.model.norm(out).detach()
            except Exception:
                model._layer_logits[name] = out
        return hook

    # --- Traverse layers ---
    for li, layer in enumerate(model.model.layers):
        prefix = f"layers.{li}"

        # Residual outputs (canonical lens probes live here)
        handle = layer.register_forward_hook(_record(f"{prefix}.out"))
        model._layer_logits_handles[f"{prefix}.out"] = handle

        # Submodules
        handle = layer.input_layernorm.register_forward_hook(_record(f"{prefix}.input_layernorm.out"))
        model._layer_logits_handles[f"{prefix}.input_layernorm.out"] = handle

        handle = layer.self_attn.register_forward_hook(_record(f"{prefix}.self_attn.out"))
        model._layer_logits_handles[f"{prefix}.self_attn.out"] = handle

        handle = layer.post_attention_layernorm.register_forward_hook(_record(f"{prefix}.post_attention_layernorm.out"))
        model._layer_logits_handles[f"{prefix}.post_attention_layernorm.out"] = handle

        handle = layer.mlp.register_forward_hook(_record(f"{prefix}.mlp.out"))
        model._layer_logits_handles[f"{prefix}.mlp.out"] = handle

        # Optional deeper projections
        for subname, submod in [
            ("q_proj", layer.self_attn.q_proj),
            ("k_proj", layer.self_attn.k_proj),
            ("v_proj", layer.self_attn.v_proj),
            ("o_proj", layer.self_attn.o_proj),
            ("up_proj", layer.mlp.up_proj),
            ("gate_proj", layer.mlp.gate_proj),
            ("down_proj", layer.mlp.down_proj),
        ]:
            full_name = f"{prefix}.{subname}.out"
            handle = submod.register_forward_hook(_record(full_name))
            model._layer_logits_handles[full_name] = handle

        vprint(f"Hooked {prefix} submodules")

    # --- Embedding layer ---
    if hasattr(model.model, "embed_tokens"):
        handle = model.model.embed_tokens.register_forward_hook(_record("embed_tokens.out"))
        model._layer_logits_handles["embed_tokens.out"] = handle

    # --- Final RMSNorm (before lm_head) ---
    # Find dynamically rather than assuming location
    norm_layer = getattr(model.model, "norm", None)
    if norm_layer is None and hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        norm_layer = getattr(model.base_model.model, "norm", None)

    if norm_layer is not None:
        handle = norm_layer.register_forward_hook(_record("final_norm.out"))
        model._layer_logits_handles["final_norm.out"] = handle
        vprint("Hooked final_norm.out")
    else:
        vprint("Warning: could not locate final RMSNorm layer")

    # --- Optionally handle decoder-only modules (norm, lm_head) ---
    if decoder_layer_names:
        for lname in decoder_layer_names:
            try:
                module = get_child_module_by_names(model, lname.split("."))
                handle = module.register_forward_hook(_record(f"{lname}.out"))
                model._layer_logits_handles[f"{lname}.out"] = handle
                vprint(f"Hooked decoder layer: {lname}")
            except Exception as e:
                vprint(f"Skipped decoder layer {lname}: {e}")

    vprint("All hooks registered.")



def clear_llama_lens_hooks_sub(model):
    if hasattr(model, "_layer_logits_handles"):
        for h in model._layer_logits_handles.values():
            try:
                h.remove()
            except Exception:
                pass
        model._layer_logits_handles.clear()

    if hasattr(model, "_layer_logits"):
        model._layer_logits.clear()
