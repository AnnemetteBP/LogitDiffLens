from typing import List, Optional
import os
import torch
import numpy as np
import pandas as pd
from scipy.special import rel_entr
from ..util.logit_lens_utils.logit_lens_wrapper import LogitLensWrapper
from ..util.logit_lens_utils.model_device_handling import get_base_model, get_embedding_device
from .metric_utils.logit_lens_helpers import(
    compute_ece,
    compute_ngram_matches,
    compute_ngram_stability,
    topk_overlap_and_kendall,
)


# ----------------------------
# Reusable Inputs
# ----------------------------
EPS = 1e-12
TOPK = 5


# ----------------------------
# Helper Functions 
# ----------------------------
def safe_softmax(logits: torch.Tensor, dim=-1, eps=EPS) -> torch.Tensor:
    """
    Numerically stable softmax that clamps small values to EPS to avoid log(0)
    and maintains valid probabilities.

    Args:
        logits: [seq_len, vocab_size] tensor of logits
        dim: dimension over which to apply softmax
        eps: minimum probability to avoid log(0)
    
    Returns:
        probs: [seq_len, vocab_size] probability tensor
    """
    # Subtract max for numerical stability
    max_logits = logits.max(dim=dim, keepdim=True).values
    exp_logits = torch.exp(logits - max_logits)
    
    # Sum over exponentials
    sum_exp = exp_logits.sum(dim=dim, keepdim=True)
    
    # Divide to get softmax probabilities
    probs = exp_logits / (sum_exp + eps)  # add eps to avoid division by zero
    
    # Clamp to avoid exactly 0 (for safe log/entropy)
    probs = probs.clamp(min=eps, max=1.0)
    return probs


def safe_entropy(probs: torch.Tensor, eps:float = EPS) -> torch.Tensor:
    probs = probs.float().clamp(min=eps, max=1.0)
    log_probs = torch.log(probs)
    ent = -(probs * log_probs).sum(dim=-1)
    return ent

def safe_kl(probs_p: torch.Tensor, probs_q: torch.Tensor, eps:float = EPS) -> torch.Tensor:
    p = probs_p.float().clamp(min=eps, max=1.0)
    q = probs_q.float().clamp(min=eps, max=1.0)
    return (p * (p.log() - q.log())).sum(dim=-1)

def safe_metric(metric_func, *args, default=np.nan):
    try:
        val = metric_func(*args)
        if val is None or (hasattr(val, '__len__') and len(val) == 0):
            return default
        return np.array(val, dtype=float)
    except Exception:
        return default

def safe_kl_per_token(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    Safe KL divergence per token position.
    Clips probabilities, handles NaNs/Infs, returns np.nan for invalid positions.

    Args:
        p: [seq_len, vocab_size] predicted probabilities for current layer
        q: [seq_len, vocab_size] predicted probabilities for next layer
        eps: small value to prevent log(0)

    Returns:
        kl_seq: [seq_len] KL divergence per token
    """
    kl_seq = []
    for i in range(p.shape[0]):
        pi, qi = p[i], q[i]

        # Clip to avoid log(0) and handle NaNs/Infs
        pi_clamped = np.clip(np.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0), eps, 1.0)
        qi_clamped = np.clip(np.nan_to_num(qi, nan=0.0, posinf=0.0, neginf=0.0), eps, 1.0)

        # Only compute KL if both vectors are valid
        if np.all(np.isfinite(pi_clamped)) and np.all(np.isfinite(qi_clamped)):
            kl_val = np.sum(rel_entr(pi_clamped, qi_clamped))
        else:
            kl_val = np.nan

        kl_seq.append(kl_val)

    return np.array(kl_seq, dtype=float)

def safe_nwd_probs(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> float:
    """
    Normalized weighted distance (1 - cosine similarity), safely handling NaNs/Infs.
    """
    p_safe = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    q_safe = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

    norm_p = np.linalg.norm(p_safe)
    norm_q = np.linalg.norm(q_safe)

    if norm_p < eps or norm_q < eps:
        return np.nan

    sim = np.dot(p_safe, q_safe) / (norm_p * norm_q)
    return 1.0 - sim


def safe_nwd(probs_current_mean, probs_next_mean):
    """
    Compute NWD safely. Returns 0.0 if undefined.
    """
    try:
        if probs_current_mean is None or probs_next_mean is None:
            return 0.0
        nwd_val = safe_nwd_probs(probs_current_mean, probs_next_mean)
        return float(nwd_val) if nwd_val is not None else 0.0
    except Exception:
        return 0.0


def safe_tvd(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    Safe total variation distance per token.
    Clips probabilities and handles NaNs/Infs.
    """
    tvd_seq = []
    for i in range(p.shape[0]):
        pi, qi = np.clip(np.nan_to_num(p[i], nan=0.0, posinf=1.0, neginf=0.0), eps, 1.0), \
                 np.clip(np.nan_to_num(q[i], nan=0.0, posinf=1.0, neginf=0.0), eps, 1.0)
        tvd_seq.append(0.5 * np.sum(np.abs(pi - qi)))
    return np.array(tvd_seq, dtype=float)


def _run_logit_lens(
    wrapper: LogitLensWrapper,
    prompts: list[str],
    add_eos: bool = True,
    topk: int = TOPK,
    eps = EPS,
    skip_input_layer: bool = True,
    include_final_norm: bool = True,
    save_layer_probs: bool = False,
    proj_precision: Optional[str] = None,   # "fp16" | "fp32" | None
    max_len: Optional[int] = 32
) -> pd.DataFrame:

    """
    Runs a logit-lens analysis with controllable *analysis* precision (fp16/fp32)
    *without* touching model parameters/dtypes. All heavy tensors are detached and
    moved to CPU as early as possible to spare GPU VRAM, and most ops are vectorized.

    Notes:
    - Model weights remain on-device in their original dtype (fp32/fp16/4- or 8-bit).
    - proj_precision controls ONLY the dtype of the detached logits used for analysis.
      * None  -> keep the model-produced dtype (if fp32, stays fp32).
      * "fp32" -> cast detached logits to float32 for analysis.
      * "fp16" -> cast detached logits to float16 for analysis (be mindful of accuracy).
    """

    import math
    torch.set_grad_enabled(False)

    rows: list[dict] = []
    wrapper.model.eval()
    device = get_embedding_device(wrapper.model)
    # -----------------------
    # Forward -> per-layer logits (on device)
    # -----------------------
    outputs, layer_dict, layer_names = wrapper.forward(
        prompts,
        project_to_logits=True,
        return_hidden=False,
        add_eos=add_eos,
        keep_on_device=True,  # keep everything on model device for the forward
        max_len=max_len
    )

    # Stack logits into a list[tensor[B,S,V]], still on device
    layer_logits_list, layer_names = wrapper.stack_layer_logits(
        layer_dict,
        keep_on_device=True,
        filter_layers=False
    )

    # -----------------------
    # Layer filtering
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

    num_layers = len(valid_layer_logits)
    batch_size = valid_layer_logits[0].shape[0]
    seq_len = valid_layer_logits[0].shape[1]
    vocab_size = getattr(wrapper.tokenizer, "vocab_size", None)

    # -----------------------
    # Determine analysis dtype (for DETACHED tensors only)
    # -----------------------
    def _to_analysis_dtype(t: torch.Tensor) -> torch.dtype:
        if proj_precision is None:
            # keep what the model produced (e.g., fp32 stays fp32)
            return t.dtype
        if proj_precision.lower() == "fp16":
            return torch.float16
        if proj_precision.lower() == "fp32":
            return torch.float32
        return t.dtype

    # -----------------------
    # Pre-compute detached CPU logits/probs layer-by-layer
    #   - Move each layer once to CPU in chosen analysis dtype
    #   - Compute probs in float32 (for stability), unless explicitly fp16
    # -----------------------
    det_logits_cpu: list[torch.Tensor] = []
    det_probs_cpu: list[torch.Tensor] = []

    for logits in valid_layer_logits:
        # detach & move to CPU with your requested analysis dtype
        adtype = _to_analysis_dtype(logits)
        l_cpu = logits.detach().to("cpu", dtype=adtype, copy=False)

        # softmax in float32 for stability unless user demanded fp16
        if adtype == torch.float16:
            probs = torch.softmax(l_cpu.to(torch.float32), dim=-1).to(torch.float16)
        else:
            probs = torch.softmax(l_cpu, dim=-1)

        det_logits_cpu.append(l_cpu)              # [B,S,V], cpu, adtype
        det_probs_cpu.append(probs.contiguous())  # [B,S,V], cpu, fp32 or fp16

    # Targets for correctness/stability (still on device; move to CPU when needed)
    if hasattr(outputs, "logits") and outputs.logits is not None:
        # greedy next-token targets: argmax over model head logits
        # (Do this on GPU then bring the single slice we need)
        full_pred = outputs.logits.argmax(dim=-1)  # [B,S]
    else:
        # Fallback: dummy increasing ids (won't be used if not meaningful)
        full_pred = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)

    # -----------------------
    # Main loop over batch (keeps memory down)
    # -----------------------
    for idx in range(batch_size):
        # reference target ids: shift by 1 for next-token prediction
        tgt_ids = full_pred[idx, 1:].detach().cpu()  # [S-1]
        max_valid_len = tgt_ids.numel()

        # reference input ids for carry-over metrics
        if hasattr(outputs, "input_ids") and outputs.input_ids is not None:
            input_ids_seq_trim_cpu = outputs.input_ids[idx, :max_valid_len].detach().cpu()
        else:
            input_ids_seq_trim_cpu = torch.arange(max_valid_len, dtype=torch.long)

        # accumulate a few per-layer items
        stability_top1, stability_topk = [], []

        for l, lname in enumerate(valid_layer_names):
            logits_cur = det_logits_cpu[l][idx][:max_valid_len]  # [S,V]
            probs_cur = det_probs_cpu[l][idx][:max_valid_len]    # [S,V]

            # --- Basic stats ---
            l32 = logits_cur.to(torch.float32)
            logit_mean = l32.mean(dim=-1)
            logit_std  = l32.std(dim=-1)
            logit_var  = l32.var(dim=-1)

            p32 = probs_cur.to(torch.float32)
            prob_mean = p32.mean(dim=-1)
            prob_std = p32.std(dim=-1)
            prob_var = p32.var(dim=-1)

            # --- Predictions & Top-K ---
            preds_seq_tensor = p32.argmax(dim=-1)
            k_eff = min(topk, p32.shape[-1])
            topk_idx_tensor = torch.topk(p32, k=k_eff, dim=-1).indices

            # --- Correctness ---
            valid_len = min(max_valid_len, preds_seq_tensor.shape[0])
            if valid_len > 0:
                preds_seq = preds_seq_tensor[:valid_len].numpy()
                tgt_np = tgt_ids[:valid_len].numpy()
                correct_1_seq = (preds_seq == tgt_np).astype(float).tolist()
                tk = topk_idx_tensor[:valid_len].numpy()
                correct_topk_seq = [float(tgt_np[i] in tk[i]) for i in range(valid_len)]
                correct_1_mean = float(np.mean(correct_1_seq))
                correct_topk_mean = float(np.mean(correct_topk_seq))
            else:
                preds_seq = np.array([], dtype=np.int64)
                tgt_np = np.array([], dtype=np.int64)
                correct_1_seq = []
                correct_topk_seq = []
                correct_1_mean = None
                correct_topk_mean = None

            # --- Entropy & normalized entropy ---
            EPS_ = eps if 'EPS' in globals() else 1e-12
            ent_seq = -(p32.clamp_min(EPS_) * (p32.clamp_min(EPS_).log())).sum(dim=-1).numpy()
            normalized_ent_seq = ent_seq / math.log(vocab_size) if vocab_size and vocab_size > 1 else ent_seq / np.log(np.e)

            # --- ECE ---
            try:
                ece_val = compute_ece(p32[:valid_len].numpy(), tgt_np) if valid_len > 0 else None
            except Exception:
                ece_val = None

            # --- N-grams & repetition ---
            ngram_correct, ngram_correct_mean, ngram_stability = {}, {}, {}
            for n in (2, 3):
                try:
                    if valid_len >= n:
                        ngram_correct[n] = compute_ngram_matches(preds_seq, tgt_np, n) or []
                        ngram_correct_mean[n] = float(np.nanmean(ngram_correct[n])) if ngram_correct[n] else None
                        if l < num_layers - 1:
                            next_preds = det_probs_cpu[l + 1][idx, :valid_len].to(torch.float32).argmax(dim=-1).numpy()
                            ngram_stability[n] = compute_ngram_stability(preds_seq, next_preds, n) or []
                        else:
                            ngram_stability[n] = []
                    else:
                        ngram_correct[n], ngram_correct_mean[n], ngram_stability[n] = [], None, []
                except Exception:
                    ngram_correct[n], ngram_correct_mean[n], ngram_stability[n] = [], None, []

            try:
                repetition_ratio = float(np.mean(preds_seq[1:] == preds_seq[:-1])) if valid_len > 1 else np.nan
            except Exception:
                repetition_ratio = np.nan

            # --- Drift metrics ---
            kl_seq, kl_mean, nwd_val, tvd_seq, tvd_mean = [], 0.0, 0.0, [], 0.0
            jacc_mean = tau_mean = 0.0
            jacc_seq = tau_seq = []

            if l < num_layers - 1 and valid_len > 0:
                p_cur = p32[:valid_len].numpy()
                p_next = det_probs_cpu[l + 1][idx, :valid_len].to(torch.float32).numpy()

                try:
                    kl_seq_raw = safe_kl_per_token(p_cur, p_next)
                    kl_seq = [float(x) if np.isfinite(x) else np.nan for x in kl_seq_raw]
                    kl_mean = float(np.nanmean(kl_seq)) if kl_seq else 0.0
                except Exception:
                    kl_seq, kl_mean = [], 0.0

                try:
                    nwd_val_raw = safe_nwd_probs(p_cur.mean(axis=0), p_next.mean(axis=0))
                    nwd_val = float(nwd_val_raw) if nwd_val_raw is not None else 0.0
                except Exception:
                    nwd_val = 0.0

                try:
                    tvd_seq = safe_tvd(p_cur, p_next)
                    tvd_mean = float(np.mean(tvd_seq)) if len(tvd_seq) > 0 else 0.0
                except Exception:
                    tvd_seq, tvd_mean = [], 0.0

                try:
                    jacc_mean, tau_mean, jacc_seq, tau_seq = topk_overlap_and_kendall(p_cur, p_next, k=k_eff)
                    jacc_mean = float(jacc_mean) if jacc_mean is not None else 0.0
                    tau_mean = float(tau_mean) if tau_mean is not None else 0.0
                except Exception:
                    jacc_mean = tau_mean = 0.0
                    jacc_seq = tau_seq = []

                try:
                    next_top1 = np.argmax(p_next, axis=-1)
                    #stability_top1.append((preds_seq[:valid_len] == next_top1).tolist())
                    stability_top1.append([int(x) for x in (preds_seq[:valid_len] == next_top1)])
                    stability_topk.append([int(preds_seq[i] == next_top1[i]) for i in range(valid_len)])
                except Exception:
                    stability_top1.append([])
                    stability_topk.append([])
            else:
                stability_top1.append([])
                stability_topk.append([])

            # -------------------------------
            # Row dict (compact + safe)
            # -------------------------------
            row = {
                "prompt_id": idx,
                "prompt_text": prompts[idx],
                "layer_index": l,
                "layer_name": lname,
                "seq_len": int(valid_len),

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
                "ngram_stability_2_seq": ngram_stability.get(2, []),
                "ngram_stability_3_seq": ngram_stability.get(3, []),
                "repetition_ratio": repetition_ratio,

                # drift / layer comparison
                "kl_next_layer_mean": kl_mean,
                "kl_next_layer_seq": kl_seq,
                "tvd_mean": tvd_mean,
                "tvd_seq": tvd_seq,
                "nwd": nwd_val,
                "topk_jaccard_mean": jacc_mean,
                "topk_kendall_tau_mean": tau_mean,
                "topk_jaccard_seq": jacc_seq,
                "topk_kendall_tau_seq": tau_seq,

                # stability
                "stability_top1_seq": stability_top1[-1] if len(stability_top1) else [],
                "stability_topk_seq": stability_topk[-1] if len(stability_topk) else [],

                # --- RAW DATA for carry-over safe metrics ---
                "logits": logits_cur,                # [S,V] CPU tensor
                "input_ids": input_ids_seq_trim_cpu, # [S] CPU tensor
                "target_ids": tgt_ids[:valid_len],   # [S] CPU tensor
                "vocab_size": vocab_size,
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