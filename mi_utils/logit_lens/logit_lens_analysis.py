from typing import List, Optional
import os
import math
import torch
import numpy as np
import pandas as pd
from scipy.special import rel_entr
from ..util.logit_lens_utils.logit_lens_wrapper import LogitLensWrapper
from ..util.logit_lens_utils.model_device_handling import get_base_model, get_embedding_device
from .metric_utils.logit_lens_helpers import(
    compute_ece,
    compute_ngram_matches
)


# ----------------------------
# Reusable Inputs
# ----------------------------
#EPS = 1e-12 
EPS = 1e-8 
TOPK = 5


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

    # Clamp safely based on dtype
    if dtype == torch.float16:
        max_val = 60000.0  
        min_val = -60000.0
    elif dtype == torch.float32:
        max_val = 1e5
        min_val = -1e5
    else:
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

    # Clamp safely based on dtype max
    if dtype == torch.float16:
        max_val = 60000.0  
        min_val = -60000.0
    elif dtype == torch.float32:
        max_val = 1e5
        min_val = -1e5
    else:
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
    add_eos: bool = True,
    topk: int = TOPK,
    eps = EPS,
    skip_input_layer: bool = True,
    include_final_norm: bool = True,
    save_layer_probs: bool = False,
    proj_precision: Optional[str] = None,  # "fp16" | "fp32" | None
    max_len: Optional[int] = 32
) -> pd.DataFrame:

    torch.set_grad_enabled(False)
    rows: list[dict] = []
    wrapper.model.eval()
    device = get_embedding_device(wrapper.model)

    # -----------------------
    # Helpers
    # -----------------------
    def _to_analysis_dtype(t: torch.Tensor) -> torch.dtype:
        if proj_precision is None:
            return t.dtype
        return torch.float16 if proj_precision.lower() == "fp16" else torch.float32

    # -----------------------
    # Forward / stack logits
    # -----------------------
    outputs, layer_dict, layer_names = wrapper.forward(
        prompts,
        project_to_logits=True,
        return_hidden=False,
        add_eos=add_eos,
        keep_on_device=True,
        max_len=max_len
    )
    layer_logits_list, layer_names = wrapper.stack_layer_logits(layer_dict, keep_on_device=True, filter_layers=False)

    # -----------------------
    # Filter layers
    # -----------------------
    valid_layer_names, valid_layer_logits = [], []
    for lname, logits in zip(layer_names, layer_logits_list):
        lname_lower = lname.lower()
        if skip_input_layer and any(k in lname_lower for k in ["input", "embed_tokens", "wte", "wpe"]):
            continue
        if not include_final_norm and any(k in lname_lower for k in ["ln", "norm", "layernorm", "rmsnorm"]):
            continue
        valid_layer_names.append(lname)
        valid_layer_logits.append(logits)

    if len(valid_layer_logits) == 0:
        return pd.DataFrame([])

    batch_size, seq_len = valid_layer_logits[0].shape[:2]
    vocab_size = getattr(wrapper.tokenizer, "vocab_size", None)

    # -----------------------
    # Precompute detached CPU logits/probs
    # -----------------------
    det_logits_cpu, det_probs_cpu = [], []
    for logits in valid_layer_logits:
        adtype = _to_analysis_dtype(logits)
        l_cpu = safe_cast_logits(logits.detach().to("cpu"), target_dtype=adtype)

        # Compute softmax safely
        probs = safe_softmax(l_cpu, dim=-1)
        if probs.dtype != adtype:
            probs = probs.to(adtype)

        det_logits_cpu.append(l_cpu)
        det_probs_cpu.append(probs.contiguous())


    # -----------------------
    # Targets
    # -----------------------
    if hasattr(outputs, "logits") and outputs.logits is not None:
        full_pred = outputs.logits.argmax(dim=-1)
    else:
        full_pred = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)

    # -----------------------
    # Main loop over batch
    # -----------------------
    for idx in range(batch_size):
        tgt_ids = full_pred[idx, 1:].detach().cpu()
        max_valid_len = tgt_ids.numel()
        input_ids_seq_trim_cpu = outputs.input_ids[idx, :max_valid_len].detach().cpu() if hasattr(outputs, "input_ids") and outputs.input_ids is not None else torch.arange(max_valid_len, dtype=torch.long)

        for l, lname in enumerate(valid_layer_names):
            logits_cur = det_logits_cpu[l][idx][:max_valid_len]
            probs_cur = det_probs_cpu[l][idx][:max_valid_len]

            # ------------------- Predictions -------------------
            preds_seq_tensor = probs_cur.argmax(dim=-1)
            k_eff = min(topk, probs_cur.shape[-1])
            topk_idx_tensor = torch.topk(probs_cur, k=k_eff, dim=-1).indices

            valid_len = min(max_valid_len, preds_seq_tensor.shape[0])
            preds_seq = preds_seq_tensor[:valid_len].detach().cpu().numpy().astype(np.int64)
            tgt_np = tgt_ids[:valid_len].numpy().astype(np.int64)
            topk_preds_seq = topk_idx_tensor[:valid_len].detach().cpu().numpy().astype(np.int64)
            correct_1_seq = (preds_seq == tgt_np).astype(float).tolist()
            correct_topk_seq = [float(tgt_np[i] in topk_preds_seq[i]) for i in range(valid_len)]

            # ------------------- Probabilities / logits stats -------------------
            logit_mean = logits_cur.mean(dim=-1)
            logit_std = logits_cur.std(dim=-1)
            logit_var = logits_cur.var(dim=-1)
            prob_mean = probs_cur.mean(dim=-1)
            prob_std = probs_cur.std(dim=-1)
            prob_var = probs_cur.var(dim=-1)
            top1_probs = probs_cur.gather(-1, preds_seq_tensor.unsqueeze(-1)).squeeze(-1)
            top1_mean_prob = float(top1_probs.mean().item()) if top1_probs.numel() else None
            topk_probs = torch.topk(probs_cur, k=k_eff, dim=-1).values
            topk_mean_prob = float(topk_probs.mean().item()) if topk_probs.numel() else None

            # ------------------- Entropy -------------------
            p_safe = safe_tensor(probs_cur, eps=eps, target_dtype=probs_cur.dtype, for_log=True)
            ent_seq = -(p_safe * p_safe.log()).sum(dim=-1).cpu().numpy()
            normalized_ent_seq = ent_seq / math.log(vocab_size) if vocab_size and vocab_size > 1 else ent_seq / np.log(np.e)

            # ------------------- ECE -------------------
            try:
                ece_val = compute_ece(p_safe.cpu().numpy(), tgt_np) if valid_len > 0 else None
            except Exception:
                ece_val = None

            # ------------------- N-grams & repetition -------------------
            ngram_correct = {n: compute_ngram_matches(preds_seq, tgt_np, n) if valid_len >= n else [] for n in (2, 3)}
            try:
                repetition_ratio = float(np.mean(preds_seq[1:] == preds_seq[:-1])) if valid_len > 1 else np.nan
            except Exception:
                repetition_ratio = np.nan

            # -------------------------------
            # Row dict (compact + safe)
            # -------------------------------
            row = {
                "prompt_id": idx,
                "prompt_text": prompts[idx],
                "layer_index": l,
                "layer_name": lname,
                "seq_len": int(valid_len),
                "vocab_size": vocab_size,
                "topk_pred_tokens_seq": topk_preds_seq.tolist(),   # shape [S, k]

                # Top-k / Top-1
                "topk_pred_tokens_seq": topk_preds_seq.tolist(),
                "preds_seq": preds_seq.tolist(),
                "top1_mean_prob": top1_mean_prob,
                "topk_mean_prob": topk_mean_prob,

                # logits/probs summary
                "logit_mean": float(logit_mean.mean().item()) if logit_mean.numel() else None,
                "logit_std_mean": float(logit_std.mean().item()) if logit_std.numel() else None,
                "logit_var_mean": float(logit_var.mean().item()) if logit_var.numel() else None,
                "prob_mean": float(prob_mean.mean().item()) if prob_mean.numel() else None,
                "prob_std_mean": float(prob_std.mean().item()) if prob_std.numel() else None,
                "prob_var_mean": float(prob_var.mean().item()) if prob_var.numel() else None,

                # entropy
                "entropy_seq": ent_seq.tolist(),
                "normalized_entropy_seq": normalized_ent_seq.tolist(),

                # correctness
                "correct_1_seq": correct_1_seq,
                "correct_topk_seq": correct_topk_seq,
                "ece": ece_val,

                # n-grams
                "ngram_correct_2_seq": ngram_correct.get(2, []),
                "ngram_correct_3_seq": ngram_correct.get(3, []),
                "repetition_ratio": repetition_ratio,

                # --- RAW DATA for carry-over safe metrics ---
                "logits": logits_cur,                # [S,V] CPU tensor
                "input_ids": input_ids_seq_trim_cpu, # [S] CPU tensor
                "target_ids": tgt_ids[:valid_len],   # [S] CPU tensor
            }

            if save_layer_probs:
                row["probs_seq"] = probs_cur[:valid_len].detach().cpu().numpy()

            rows.append(row)

    return pd.DataFrame(rows)


def run_logit_lens(
    wrapper:LogitLensWrapper,
    prompts:List[str],
    model_name:str="model",
    dataset_name:str="dataset",
    save_dir:str="logs/logit_lens_logs/logit_lens_analysis",
    topk:int=TOPK,
    eps=EPS,
    skip_input_layer:bool=True,
    include_final_norm:bool=True,
    save_layer_probs:bool=False,
    proj_precision: Optional[str] = None,   # "fp16" | "fp32" | None
    max_len: Optional[int] = 32
) -> None:

    data_df = _run_logit_lens(
        wrapper=wrapper,
        prompts=prompts,
        topk=topk,
        eps=eps,
        skip_input_layer=skip_input_layer,
        include_final_norm=include_final_norm,
        save_layer_probs=save_layer_probs,
        proj_precision=proj_precision,
        max_len=max_len
    )

    data_dict = {col: data_df[col].tolist() for col in data_df.columns}

    os.makedirs(save_dir, exist_ok=True)
    pt_path = f"{save_dir}/{dataset_name}_{model_name}.pt"
    torch.save(data_dict, pt_path)