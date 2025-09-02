import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from ..util.logit_lens_utils.logit_lens_wrapper import LogitLensWrapper
from ..util.logit_lens_utils.model_device_handling import get_base_model, get_embedding_device



# ----------------------------
# Reusable Inputs
# ----------------------------
EPS = 1e-12
TOPK = 5


def analyze_logit_lens_batch(
    wrapper:LogitLensWrapper,
    texts:list[str],
    top_k:int=TOPK,
    decoder=None,
    include_input_layer:bool=True,
    include_embed_tokens:bool=True,
    add_eos:bool=True
) -> dict:
    """
    Simple analysis to test
    """
    wrapper.model.eval()
    device = get_embedding_device(wrapper.model)

    if add_eos and wrapper.tokenizer.eos_token_id is not None:
        texts = [t + wrapper.tokenizer.eos_token for t in texts]

    outputs, logits_by_layer, layer_names = wrapper.forward(
        texts,
        project_to_logits=True,
        decoder=decoder
    )

    filtered_logits = {
        name: h
        for name, h in logits_by_layer.items()
        if (include_input_layer or "input" not in name)
           and (include_embed_tokens or "embed_tokens" not in name)
    }

    tokenizer_out = wrapper.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokenizer_out.input_ids.to(device)
    targets = input_ids[:, 1:]

    results = {}
    for lname, h in filtered_logits.items():
        h = h.to(device)
        probs = F.softmax(h[:, :-1, :], dim=-1)
        entropy = -(probs * torch.log(probs + 1e-12)).sum(-1)
        topk_preds = torch.topk(probs, top_k, dim=-1).indices
        correct = (topk_preds == targets.unsqueeze(-1)).any(-1).float()

        results[lname] = {
            "probs": probs.cpu().numpy(),
            "entropy": entropy.cpu().numpy(),
            "topk_acc": correct.cpu().numpy(),
            "has_nan": torch.isnan(h).any().item(),
            "has_inf": torch.isinf(h).any().item(),
        }

    return results


def analyze_SAFE_degradation(
    wrapper:LogitLensWrapper,
    texts:list[str],
    top_k:int=TOPK,
    decoder=None,
    add_eos:bool=True,
    reference_wrapper=None,
    eps:float=EPS
) -> dict:
    """
    Safe interpretability pass:
      • uses original dtypes to(torch.float32)
      • with epsilon and clamping
      • metrics are device-consistent via wrapper.device
    """
    wrapper.model.eval()
    device = wrapper.device

    # --- prepare inputs ---
    if add_eos and wrapper.tokenizer.eos_token_id is not None:
        texts = [t + wrapper.tokenizer.eos_token for t in texts]

    # Forward pass on quantized/target model
    outputs, logits_by_layer, layer_names = wrapper.forward(
        texts, project_to_logits=True, decoder=decoder
    )

    # Targets
    input_ids = wrapper.tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).input_ids.to(device)
    targets = input_ids[:, 1:]

    # Opt. reference pass
    ref_logits_by_layer = None
    if reference_wrapper is not None:
        _, ref_logits_by_layer, _ = reference_wrapper.forward(
            texts, project_to_logits=True, decoder=decoder
        )

    results = {}
    for lname, h in logits_by_layer.items():
        h = h.to(device)

        # ---- Softmax + Entropy ----
        probs = F.softmax(h[:, :-1, :], dim=-1).to(torch.float32)
        entropy = -(probs * torch.log(probs.clamp(min=eps))).sum(-1)

        # ---- Top-k accuracy ----
        topk_preds = torch.topk(probs, top_k, dim=-1).indices
        correct = (topk_preds == targets.unsqueeze(-1)).any(-1).float()

        # ---- Stability checks ----
        nan_logits_count = torch.isnan(h).sum().item()
        inf_logits_count = torch.isinf(h).sum().item()

        nan_probs_count = torch.isnan(probs).sum().item()
        inf_probs_count = torch.isinf(probs).sum().item()

        nan_entropy_count = torch.isnan(entropy).sum().item()
        inf_entropy_count = torch.isinf(entropy).sum().item()

        results[lname] = {
            # interpretability proxies
            "entropy_mean": entropy.mean().item(),
            "entropy_std": entropy.std().item(),
            "topk_acc_mean": correct.mean().item(),

            # stability (counts + rates)
            "nan_logits_count": nan_logits_count,
            "inf_logits_count": inf_logits_count,
            "nan_probs_count": nan_probs_count,
            "inf_probs_count": inf_probs_count,
            "nan_entropy_count": nan_entropy_count,
            "inf_entropy_count": inf_entropy_count,

            "nan_logits_rate": nan_logits_count / h.numel(),
            "inf_logits_rate": inf_logits_count / h.numel(),
            "nan_probs_rate": nan_probs_count / probs.numel(),
            "inf_probs_rate": inf_probs_count / probs.numel(),
            "nan_entropy_rate": nan_entropy_count / entropy.numel(),
            "inf_entropy_rate": inf_entropy_count / entropy.numel(),
        }

        # ---- Drift vs reference ----
        if ref_logits_by_layer is not None and lname in ref_logits_by_layer:
            ref = ref_logits_by_layer[lname].to(device)

            cosine_drift = F.cosine_similarity(
                h.flatten(1), ref.flatten(1), dim=-1
            ).mean().item()

            ref_probs = F.softmax(ref[:, :-1, :], dim=-1).to(torch.float32)
            ref_topk = torch.topk(ref_probs, top_k, dim=-1).indices
            topk_overlap = (
                (topk_preds == ref_topk.unsqueeze(-1)).any(-1).float().mean().item()
            )

            results[lname]["cosine_drift"] = cosine_drift
            results[lname]["topk_overlap"] = topk_overlap
        else:
            results[lname]["cosine_drift"] = None
            results[lname]["topk_overlap"] = None

    return results


def analyze_UNSAFE_degradation(
        wrapper:LogitLensWrapper,
        texts:list[str],
        top_k:int=TOPK,
        decoder=None,
        add_eos=True,
        reference_wrapper=None
) -> dict:
    """
    Unsafe interpretability pass:
      • uses original dtypes, no casting to(torch.float32)
      • no epsilon, no clamping
      • Captures degradation in interpretability.
    """
    wrapper.model.eval()
    device = wrapper.device

    if add_eos and wrapper.tokenizer.eos_token is not None:
        texts = [t + wrapper.tokenizer.eos_token for t in texts]

    outputs, logits_by_layer, _ = wrapper.forward(texts, project_to_logits=True, decoder=decoder)

    toks = wrapper.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    targets = toks.input_ids.to(device)[:, 1:]

    results = {}
    for lname, h in logits_by_layer.items():
        # Keep original dtype for unsafe evaluation
        probs = F.softmax(h[:, :-1, :], dim=-1)
        entropy = -(probs * probs.log()).sum(-1)
        topk_preds = torch.topk(probs, top_k, dim=-1).indices
        correct = (topk_preds == targets.unsqueeze(-1)).any(-1).float()

        results[lname] = {
            "entropy_mean": entropy.mean().item(),
            "topk_acc_mean": correct.mean().item(),
            "nan_logits_rate": torch.isnan(h).float().mean().item(),
            "inf_logits_rate": torch.isinf(h).float().mean().item(),
            "nan_probs_rate": torch.isnan(probs).float().mean().item(),
            "inf_probs_rate": torch.isinf(probs).float().mean().item(),
            "nan_entropy_rate": torch.isnan(entropy).float().mean().item(),
            "inf_entropy_rate": torch.isinf(entropy).float().mean().item(),
        }

    return results
