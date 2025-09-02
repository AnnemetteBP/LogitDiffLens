from typing import List, Optional
import os
import torch
import numpy as np
import pandas as pd
from scipy.special import rel_entr
from ..util.logit_lens_utils.logit_lens_wrapper import LogitLensWrapper
from ..util.logit_lens_utils.model_device_handling import get_base_model, get_embedding_device
from .metric_utils.div_aware_metrics import(
    div_stability_top1,
    div_stability_topk,
    div_accuracy_top1,
    div_accuracy_topk
)
from .metric_utils.logit_lens_helpers import(
    get_activation_tensor,
    compute_ece,
    safe_compute_cka,
    safe_compute_svcca,
    compute_ngram_matches,
    compute_ngram_stability,
    topk_overlap_and_kendall,
    align_features
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


def _run_logit_lens(
    wrapper: LogitLensWrapper,
    prompts: list[str],
    add_eos: bool = True,
    A_acts: Optional[dict] = None,
    B_acts: Optional[dict] = None,
    topk: int = TOPK,
    skip_input_layer: bool = True,
    include_final_norm: bool = True,
    save_layer_probs: bool = False
) -> pd.DataFrame:

    rows = []
    wrapper.model.eval()
    device = get_embedding_device(wrapper.model)

    # --- Forward pass ---
    outputs, layer_dict, layer_names = wrapper.forward(
        prompts,
        project_to_logits=True,
        return_hidden=False,
        add_eos=add_eos,
        keep_on_device=True
    )

    # --- Stack logits ---
    layer_logits_list, layer_names = wrapper.stack_layer_logits(
        layer_dict,
        keep_on_device=True,
        filter_layers=False
    )

    # --- Filter layers ---
    valid_layer_names, valid_layer_logits = [], []
    for lname, logits in zip(layer_names, layer_logits_list):
        lname_lower = lname.lower()
        if skip_input_layer and any(k in lname_lower for k in ["input", "embed_tokens", "wte", "wpe"]):
            continue
        if not include_final_norm and any(k in lname_lower for k in ["ln", "norm", "layernorm", "rmsnorm"]):
            continue
        valid_layer_names.append(lname)
        valid_layer_logits.append(logits)

    num_layers = len(valid_layer_logits)
    batch_size = valid_layer_logits[0].shape[0]
    seq_len = valid_layer_logits[0].shape[1]

    for idx in range(batch_size):
        target_ids_seq = outputs.logits.argmax(dim=-1)[idx, 1:] if hasattr(outputs, "logits") else torch.arange(seq_len - 1, device=device)
        input_ids_seq_trim = target_ids_seq.clone()
        all_preds, all_topk_preds = [], []
        stability_top1, stability_topk = [], []

        for l, (lname, logits_l_all) in enumerate(zip(valid_layer_names, valid_layer_logits)):
            logits_tensor = logits_l_all[idx]
            if not isinstance(logits_tensor, torch.Tensor):
                logits_tensor = torch.as_tensor(logits_tensor, device=device)

            # --- Safe softmax + move to CPU float32 ---
            probs_tensor = safe_softmax(logits_tensor).detach().cpu().float()
            # --- Logit / probability metrics ---
            logit_mean = logits_tensor.mean(dim=-1)
            logit_std = logits_tensor.std(dim=-1)
            logit_var = logits_tensor.var(dim=-1)

            prob_mean = probs_tensor.mean(dim=-1)
            prob_std = probs_tensor.std(dim=-1)
            prob_var = probs_tensor.var(dim=-1)

            # --- Predictions & Top-K ---
            preds_seq_tensor = torch.argmax(probs_tensor, dim=-1)
            topk_idx_tensor = torch.topk(probs_tensor, k=topk, dim=-1).indices

            valid_len = min(len(target_ids_seq), preds_seq_tensor.shape[0])
            preds_seq_tensor = preds_seq_tensor[:valid_len]
            probs_tensor = probs_tensor[:valid_len, :] if valid_len > 0 else torch.zeros((0, probs_tensor.shape[-1]), dtype=torch.float32)
            target_ids_tensor = target_ids_seq[:valid_len].detach().cpu().numpy()

            # --- Correctness metrics ---
            if valid_len > 0:
                correct_1_seq = (preds_seq_tensor.cpu().numpy() == target_ids_tensor).astype(float).tolist()
                correct_topk_seq = [float(target_ids_tensor[i] in topk_idx_tensor[i].cpu().numpy()) for i in range(valid_len)]
                correct_1_mean = float(np.mean(correct_1_seq))
                correct_topk_mean = float(np.mean(correct_topk_seq))
            else:
                correct_1_seq, correct_topk_seq = [], []
                correct_1_mean = correct_topk_mean = None

            all_preds.append(preds_seq_tensor)
            all_topk_preds.append(topk_idx_tensor)

            # --- Entropy & normalized entropy ---
            ent_seq = -(probs_tensor * torch.log(probs_tensor + EPS)).sum(dim=-1).cpu().numpy()
            normalized_ent_seq = ent_seq / np.log(getattr(wrapper.tokenizer, "vocab_size", np.e))

            # --- Convert for storage ---
            preds_seq = preds_seq_tensor.cpu().numpy()
            probs_seq = probs_tensor.cpu().numpy()

            try:
                ece_val = compute_ece(probs_seq, target_ids_tensor)
            except Exception:
                ece_val = None

            # --- Ngrams & repetition ---
            ngram_correct, ngram_correct_mean, ngram_stability = {}, {}, {}
            for n in [2, 3]:
                if valid_len >= n:
                    ngram_correct[n] = compute_ngram_matches(preds_seq, target_ids_tensor, n)
                    ngram_correct_mean[n] = np.nanmean(ngram_correct[n])
                    if l < num_layers - 1:
                        next_preds = torch.argmax(valid_layer_logits[l + 1][idx], dim=-1)[:valid_len].cpu().numpy()
                        ngram_stability[n] = compute_ngram_stability(preds_seq, next_preds, n)
                    else:
                        ngram_stability[n] = []
                else:
                    ngram_correct[n], ngram_correct_mean[n], ngram_stability[n] = [], None, []

            repetition_ratio = float(np.mean(preds_seq[1:] == preds_seq[:-1])) if valid_len > 1 else np.nan

            # --- KL / NWD / Top-K drift ---
            if l < num_layers - 1 and valid_len > 0:
                next_probs_tensor = safe_softmax(valid_layer_logits[l + 1][idx])[:valid_len, :].detach().cpu().float()
                kl_seq = safe_kl_per_token(probs_seq[:valid_len], next_probs_tensor.numpy())
                kl_mean = float(np.nanmean(kl_seq))
                nwd_val = safe_nwd_probs(probs_seq[:valid_len].mean(axis=0), next_probs_tensor.numpy().mean(axis=0))
                jacc_mean, tau_mean, jacc_seq, tau_seq = topk_overlap_and_kendall(probs_seq[:valid_len], next_probs_tensor.numpy(), k=topk)
                stability_top1.append((preds_seq == np.argmax(next_probs_tensor.numpy(), axis=-1)).tolist())
                stability_topk.append([int(preds_seq[i] in np.argmax(next_probs_tensor.numpy(), axis=-1)) for i in range(valid_len)])
            else:
                kl_seq = kl_mean = nwd_val = None
                jacc_mean = tau_mean = []
                jacc_seq = tau_seq = []
                stability_top1.append([])
                stability_topk.append([])

            # --- CKA / SVCCA ---
            fp_act = get_activation_tensor(A_acts[lname]) if A_acts and lname in A_acts else None
            quant_act = get_activation_tensor(B_acts[lname]) if B_acts and lname in B_acts else None
            if isinstance(fp_act, torch.Tensor): fp_act = fp_act.detach().cpu().numpy()
            if isinstance(quant_act, torch.Tensor): quant_act = quant_act.detach().cpu().numpy()
            if fp_act is not None and quant_act is not None:
                fp_proj, q_proj = align_features(fp_act, quant_act)
                cka_val = safe_compute_cka(fp_proj, q_proj)
                svcca_val = safe_compute_svcca(fp_proj, q_proj)
            else:
                cka_val = svcca_val = np.nan


            # -------------------------------
            # Row dict
            # -------------------------------
            row = {
                "prompt_id": idx,
                "prompt_text": prompts[idx],
                "layer_index": l,
                "layer_name": lname,
                "seq_len": valid_len,
                "logit_mean": float(logit_mean.mean().item()) if logit_mean.numel() > 0 else None,
                "logit_std_mean": float(logit_std.mean().item()) if logit_std.numel() > 0 else None,
                "logit_var_mean": float(logit_var.mean().item()) if logit_var.numel() > 0 else None,
                "prob_mean": float(prob_mean.mean().item()) if prob_mean.numel() > 0 else None,
                "prob_std_mean": float(prob_std.mean().item()) if prob_std.numel() > 0 else None,
                "prob_var_mean": float(prob_var.mean().item()) if prob_var.numel() > 0 else None,
                "entropy_mean": float(ent_seq.mean().item()) if ent_seq.numel() > 0 else None,
                "entropy_seq": ent_seq.detach().cpu().tolist() if ent_seq.numel() > 0 else [],
                "normalized_entropy_mean": float(normalized_ent_seq.mean().item()) if normalized_ent_seq.numel() > 0 else None,
                "normalized_entropy_seq": normalized_ent_seq.detach().cpu().tolist() if normalized_ent_seq.numel() > 0 else [],
                "correct_1_seq": correct_1_seq,
                "correct_topk_seq": correct_topk_seq,
                "correct_1_mean": float(torch.tensor(correct_1_seq, device=device).mean().item()) if correct_1_seq else None,
                "correct_topk_mean": float(torch.tensor(correct_topk_seq, device=device).mean().item()) if correct_topk_seq else None,
                "kl_next_layer_mean": kl_mean if 'kl_mean' in locals() else None,
                "kl_next_layer_seq": kl_seq if 'kl_seq' in locals() else [],
                "nwd": nwd_val if 'nwd_val' in locals() else None,
                "topk_jaccard_mean": jacc_mean if 'jacc_mean' in locals() else None,
                "topk_kendall_tau_mean": tau_mean if 'tau_mean' in locals() else None,
                "topk_jaccard_seq": jacc_seq if 'jacc_seq' in locals() else [],
                "topk_kendall_tau_seq": tau_seq if 'tau_seq' in locals() else [],
                "stability_top1_seq": stability_top1[l] if l < len(stability_top1) else [],
                "stability_topk_seq": stability_topk[l] if l < len(stability_topk) else [],
                "ece": ece_val,
                "ngram_correct_2_seq": ngram_correct.get(2, []),
                "ngram_correct_2_mean": ngram_correct_mean.get(2),
                "ngram_correct_3_seq": ngram_correct.get(3, []),
                "ngram_correct_3_mean": ngram_correct_mean.get(3),
                "ngram_stability_2_seq": ngram_stability.get(2, []),
                "ngram_stability_3_seq": ngram_stability.get(3, []),
                "repetition_ratio": repetition_ratio,
                "cka_fp_vs_layer": cka_val,
                "svcca_fp_vs_layer": svcca_val,
            }

            if save_layer_probs:
                row["layer_probs"] = probs_tensor.detach().cpu().tolist() if probs_tensor.numel() > 0 else []

            rows.append(row)


        # --- Divergence metrics ---
        seq_len_layer = int(input_ids_seq_trim.numel() if isinstance(input_ids_seq_trim, torch.Tensor) else len(input_ids_seq_trim))

        # helper to create NaN vector of desired length
        def _nan_vec(n):
            return np.full(n, np.nan, dtype=float)

        if len(all_preds) == 0:
            # no preds collected -> return shaped empty arrays
            all_preds_arr = np.zeros((num_layers, 0), dtype=int)
            all_topk_preds_arr = np.zeros((num_layers, 0, topk), dtype=int)
            tgt_np_top1 = np.zeros((0,), dtype=int)
            tgt_np_topk = np.zeros((0,), dtype=int)
        else:
            # Convert preds list to numpy (preserve order)
            preds_np_list = []
            for p in all_preds:
                if isinstance(p, torch.Tensor):
                    preds_np_list.append(p.detach().cpu().numpy())
                else:
                    preds_np_list.append(np.asarray(p))

            # Convert topk list to numpy
            topk_np_list = []
            for tk in all_topk_preds:
                if isinstance(tk, torch.Tensor):
                    topk_np_list.append(tk.detach().cpu().numpy())
                else:
                    topk_np_list.append(np.asarray(tk))

            # Convert target IDs to numpy (input_ids_seq_trim might be tensor or ndarray)
            if isinstance(input_ids_seq_trim, torch.Tensor):
                tgt_full = input_ids_seq_trim.detach().cpu().numpy()
            else:
                tgt_full = np.asarray(input_ids_seq_trim)

            # --- Top-1 alignment ---
            lens_top1 = [arr.shape[0] for arr in preds_np_list]
            min_len_top1 = int(min(lens_top1 + [tgt_full.shape[0]]))
            if min_len_top1 == 0:
                all_preds_arr = np.zeros((num_layers, 0), dtype=int)
                tgt_np_top1 = np.zeros((0,), dtype=int)
            else:
                all_preds_arr = np.stack([arr[:min_len_top1].astype(int) for arr in preds_np_list], axis=0)  # [L, min_len_top1]
                tgt_np_top1 = tgt_full[:min_len_top1].astype(int)

            # --- Top-k alignment (keep indices shape [L, min_len_topk, k]) ---
            lens_topk = [arr.shape[0] for arr in topk_np_list]
            min_len_topk = int(min(lens_topk + [tgt_full.shape[0]]))
            if min_len_topk == 0:
                all_topk_preds_arr = np.zeros((num_layers, 0, topk), dtype=int)
                tgt_np_topk = np.zeros((0,), dtype=int)
            else:
                all_topk_preds_arr = np.stack([arr[:min_len_topk].astype(int) for arr in topk_np_list], axis=0)  # [L, min_len_topk, k]
                tgt_np_topk = tgt_full[:min_len_topk].astype(int)

        # wrapper to safely call divergence metric functions and pad/trim to seq_len_layer
        def _safe_div_metric(metric_func, arr_np, tgt_np):
            try:
                out = metric_func(arr_np, tgt_np, tgt_np)
                if out is None:
                    return _nan_vec(seq_len_layer)
                out_np = np.asarray(out, dtype=float)
                # pad/trim to seq_len_layer (so DataFrame fields have consistent length)
                padded = _nan_vec(seq_len_layer)
                copy_len = min(out_np.shape[0], seq_len_layer)
                if copy_len > 0:
                    padded[:copy_len] = out_np[:copy_len]
                return padded
            except Exception:
                return _nan_vec(seq_len_layer)

        # compute divergence metrics (top1 uses all_preds_arr; topk uses all_topk_preds_arr)
        div_stab_top1 = _safe_div_metric(div_stability_top1, all_preds_arr, tgt_np_top1)
        div_acc_top1  = _safe_div_metric(div_accuracy_top1,  all_preds_arr, tgt_np_top1)
        div_stab_topk = _safe_div_metric(div_stability_topk, all_topk_preds_arr, tgt_np_topk)
        div_acc_topk  = _safe_div_metric(div_accuracy_topk,  all_topk_preds_arr, tgt_np_topk)

        for r in rows:
            if r["prompt_id"] == idx:
                r["div_stability_top1"] = div_stab_top1.tolist()
                r["div_stability_topk"] = div_stab_topk.tolist()
                r["div_accuracy_top1"] = div_acc_top1.tolist()
                r["div_accuracy_topk"] = div_acc_topk.tolist()

    return pd.DataFrame(rows)


def run_logit_lens(
    wrapper:LogitLensWrapper,
    prompts:List[str],
    model_name:str="model",
    dataset_name:str="dataset",
    save_dir:str="logs/logit_lens_logs/batch_analysis",
    A_acts=None,
    B_acts=None,
    topk:int=TOPK,
    skip_input_layer:bool=True,
    include_final_norm:bool=True,
    save_layer_probs:bool=False,
) -> None:

    df = _run_logit_lens(
        wrapper=wrapper,
        prompts=prompts,
        A_acts=A_acts,
        B_acts=B_acts,
        topk=topk,
        skip_input_layer=skip_input_layer,
        include_final_norm=include_final_norm,
        save_layer_probs=save_layer_probs
    )

    os.makedirs(save_dir, exist_ok=True)

    csv_path= f"{save_dir}/{dataset_name}_{model_name}.csv"
    df.to_csv(csv_path, index=False)