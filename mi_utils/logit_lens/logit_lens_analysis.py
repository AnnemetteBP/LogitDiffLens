from typing import List, Optional
import os
import math
import torch
import numpy as np
import pandas as pd
import glob
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
from scipy.special import rel_entr
from ..util.logit_lens_utils.logit_lens_wrapper import LogitLensWrapper
from ..util.logit_lens_utils.model_device_handling import get_base_model, get_embedding_device
from .metric_utils.logit_lens_helpers import compute_ece

# ----------------------------
# Reusable Inputs
# ----------------------------
EPS = 1e-12 
#EPS = 1e-8 
TOPK = 5
MAX_LEN = 16
# ----------------------------
# Helper Functions 
# ----------------------------
def safe_entropy(probs: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    Compute entropy safely for float16 or float32 probabilities.
    probs: [*, V] tensor, can be float16 or float32
    eps: minimum probability
    """
    # Cast to float32 for safe log computation
    probs_f32 = probs.to(torch.float32)

    # Clamp small probabilities
    probs_f32 = probs_f32.clamp(min=eps, max=1.0)

    log_probs = torch.log(probs_f32)
    ent = -(probs_f32 * log_probs).sum(dim=-1)

    return ent

# -----------------------
# Safe cast & clamp helper
# -----------------------
def safe_tensor(t: torch.Tensor, eps: float = EPS, target_dtype: Optional[torch.dtype] = None, for_log: bool = False):
    dtype = target_dtype
    if dtype is None:
        if t.dtype in [torch.float16, torch.float32]:
            dtype = t.dtype
        else:
            dtype = torch.float32  

    if t.dtype != dtype:
        t = t.to(dtype)

    # NaN / inf protection
    t = torch.nan_to_num(t, nan=-1e9, posinf=1e9, neginf=-1e9)

    # Clamp safely
    max_val = 1e5
    min_val = -1e5

    t = torch.clamp(t, min=min_val, max=max_val)

    if for_log:
        t = t.clamp_min(eps)

    return t

def safe_cast_logits(tensor: torch.Tensor, target_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    dtype = target_dtype or (tensor.dtype if tensor.dtype in [torch.float16, torch.float32] else torch.float32)
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype)

    # NaN / inf protection
    tensor = torch.nan_to_num(tensor, nan=-1e9, posinf=1e9, neginf=-1e9)

    # Clamp safely
    max_val = 1e5
    min_val = -1e5

    tensor = torch.clamp(tensor, min=min_val, max=max_val)
    return tensor


def safe_softmax(logits: torch.Tensor, dim=-1, eps=EPS) -> torch.Tensor:
    max_logits = logits.max(dim=dim, keepdim=True).values
    exp_logits = torch.exp(logits - max_logits)
    sum_exp = exp_logits.sum(dim=dim, keepdim=True)
    probs = exp_logits / (sum_exp + eps)
    return probs.clamp(min=eps, max=1.0)



def _run_logit_lens(
    wrapper: LogitLensWrapper,
    prompts: list[str],
    analysis_type: str = "self_att",  # "self_att" or "mlp"
    add_eos: bool = True,
    topk: int = TOPK,
    eps: float = EPS,
    skip_input_layer: bool = False,   # always include embed_tokens if False
    include_final_norm: bool = False,
    save_layer_probs: bool = False,
    proj_precision: Optional[str] = None,
    max_len: Optional[int] = MAX_LEN
) -> list[dict]:
    """
    Compute logit lens analysis for self_att or mlp + embed tokens.
    Returns a list of rows with only relevant layers and metrics.
    """
    assert analysis_type in ("self_att", "mlp")

    torch.set_grad_enabled(False)
    wrapper.model.eval()
    device = get_embedding_device(wrapper.model)
    rows: list[dict] = []

    # Helper dtype
    def _to_analysis_dtype(t: torch.Tensor) -> torch.dtype:
        if proj_precision is None:
            return t.dtype
        return torch.float16 if proj_precision.lower() == "fp16" else torch.float32

    # Forward pass
    outputs, layer_dict, layer_names = wrapper.forward(
        prompts,
        project_to_logits=True,
        return_hidden=False,
        add_eos=add_eos,
        keep_on_device=True,
        max_len=max_len
    )
    layer_logits_list, layer_names = wrapper.stack_layer_logits(layer_dict, keep_on_device=True, filter_layers=False)

    # Filter relevant layers
    valid_layer_names, valid_layer_logits = [], []
    for lname, logits in zip(layer_names, layer_logits_list):
        lname_lower = lname.lower()
        # always include embed_tokens if skip_input_layer=False
        if not skip_input_layer and "embed_tokens" in lname_lower:
            pass
        elif skip_input_layer and any(k in lname_lower for k in ["input", "wte", "wpe"]):
            continue
        if not include_final_norm and any(k in lname_lower for k in ["ln", "norm", "layernorm", "rmsnorm"]):
            continue

        # Keep only relevant layers per analysis
        if analysis_type == "self_att" and not (("self_att" in lname_lower) or ("embed_tokens" in lname_lower)):
            continue
        if analysis_type == "mlp" and not (("mlp" in lname_lower) or ("embed_tokens" in lname_lower)):
            continue

        valid_layer_names.append(lname)
        valid_layer_logits.append(logits)

    if not valid_layer_logits:
        return []

    batch_size, seq_len = valid_layer_logits[0].shape[:2]
    vocab_size = getattr(wrapper.tokenizer, "vocab_size", None)

    # Precompute logits/probs on CPU
    det_logits_cpu, det_probs_cpu = [], []
    for logits in valid_layer_logits:
        adtype = _to_analysis_dtype(logits)
        l_cpu = safe_cast_logits(logits.detach().to("cpu"), target_dtype=adtype)
        probs = safe_softmax(l_cpu, dim=-1).to(adtype)
        det_logits_cpu.append(l_cpu)
        det_probs_cpu.append(probs)

    # Targets
    if hasattr(outputs, "logits") and outputs.logits is not None:
        full_pred = outputs.logits.argmax(dim=-1)
    else:
        full_pred = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)

    # Main loop
    for idx in range(batch_size):
        tgt_ids = full_pred[idx, 1:].cpu()
        max_valid_len = tgt_ids.numel()
        input_ids_seq_trim_cpu = (
            outputs.input_ids[idx, :max_valid_len].cpu()
            if hasattr(outputs, "input_ids") and outputs.input_ids is not None
            else torch.arange(max_valid_len, dtype=torch.long)
        )

        for l, lname in enumerate(valid_layer_names):
            logits_cur = det_logits_cpu[l][idx, :max_valid_len]
            probs_cur = det_probs_cpu[l][idx, :max_valid_len]

            preds_seq_tensor = probs_cur.argmax(dim=-1)
            k_eff = min(topk, probs_cur.shape[-1])
            topk_idx_tensor = torch.topk(probs_cur, k=k_eff, dim=-1).indices

            valid_len = preds_seq_tensor.shape[0]
            preds_seq = preds_seq_tensor.cpu().numpy().astype(np.int64)
            tgt_np = tgt_ids[:valid_len].numpy().astype(np.int64)
            topk_preds_seq = topk_idx_tensor.cpu().numpy().astype(np.int64)

            # Base row
            row = {
                "prompt_id": idx,
                "prompt_text": prompts[idx],
                "layer_index": l,
                "layer_name": lname,
                "seq_len": int(valid_len),
                "vocab_size": vocab_size,
                "logits": logits_cur.cpu(),                # keep as CPU tensor
                "input_ids": input_ids_seq_trim_cpu.cpu(),
                "target_ids": tgt_ids[:valid_len].cpu(),
            }

            # Entropy (all analyses)
            p_safe = safe_tensor(probs_cur, eps=eps, target_dtype=probs_cur.dtype, for_log=True)
            row["entropy_seq"] = (-(p_safe * p_safe.log()).sum(dim=-1).cpu())
            row["normalized_entropy_seq"] = row["entropy_seq"] / math.log(vocab_size) if vocab_size and vocab_size > 1 else row["entropy_seq"]

            # Mean layer probs (all analyses)
            row["prob_mean_seq"] = probs_cur.mean(dim=-1).cpu()

            # Optional: save full layer probs
            if save_layer_probs:
                row["probs_seq"] = probs_cur.cpu()

            # Conditional metrics
            if analysis_type == "self_att":
                row["preds_seq"] = preds_seq
                row["topk_pred_tokens_seq"] = topk_preds_seq
                row["top1_mean_prob"] = probs_cur.gather(-1, preds_seq_tensor.unsqueeze(-1)).squeeze(-1).mean().item()
                row["topk_mean_prob"] = torch.topk(probs_cur, k=k_eff, dim=-1).values.mean().item()
                try:
                    row["ece"] = compute_ece(p_safe.cpu().numpy(), tgt_np) if valid_len > 0 else None
                except Exception:
                    row["ece"] = None
                try:
                    row["repetition_ratio"] = float(np.mean(preds_seq[1:] == preds_seq[:-1])) if valid_len > 1 else np.nan
                except Exception:
                    row["repetition_ratio"] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def run_logit_lens(
    wrapper: LogitLensWrapper,
    prompts: list[str],
    analysis_type: str = "self_att",   # choose "self_att" or "mlp"
    model_name: str = "model",
    dataset_name: str = "dataset",
    save_dir: str = "logs/logit_lens_logs/logit_lens_analysis",
    topk: int = TOPK,
    eps: float = EPS,
    skip_input_layer: bool = False,
    include_final_norm: bool = False,
    save_layer_probs: bool = False,
    proj_precision: Optional[str] = None,
    max_len: Optional[int] = MAX_LEN
):
    """
    Run logit lens analysis (self_att or mlp) and save memory-efficient .pt file.
    Returns the DataFrame for immediate inspection if needed.
    """
    # Run the analysis
    df_rows = _run_logit_lens(
        wrapper=wrapper,
        prompts=prompts,
        analysis_type=analysis_type,
        add_eos=True,
        topk=topk,
        eps=eps,
        skip_input_layer=skip_input_layer,
        include_final_norm=include_final_norm,
        save_layer_probs=save_layer_probs,
        proj_precision=proj_precision,
        max_len=max_len
    )

    if df_rows.empty:
        print("No valid layers found. Nothing saved.")
        return df_rows

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset_name}_{model_name}_{analysis_type}.pt")

    # Save as .pt for memory-efficient reload with raw tensors
    torch.save(df_rows, save_path)
    print(f"Saved analysis to {save_path}")

    #return df_rows
