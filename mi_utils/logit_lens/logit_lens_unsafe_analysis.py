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


def unsafe_softmax_torch(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute softmax without promoting dtype or moving to CPU.
    Keeps bfloat16/float16/logits on the original device.
    """
    # Subtract max for numerical stability
    max_logits = logits.max(dim=dim, keepdim=True).values
    exps = torch.exp(logits - max_logits)
    sum_exps = exps.sum(dim=dim, keepdim=True)
    return exps / sum_exps


def unsafe_kl(p: np.ndarray, q: np.ndarray):
    """KL divergence per token; ignores invalid entries but propagates NaNs if none valid."""
    mask = ~np.isnan(p) & ~np.isnan(q) & ~np.isinf(p) & ~np.isinf(q)
    if not np.any(mask):
        return np.nan
    return np.sum(rel_entr(p[mask], q[mask]), axis=-1)


def unsafe_kl_per_token(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Compute KL divergence per token position between probability distributions p and q.
    NaNs/Infs in p or q propagate; no clipping or safe replacement.
    
    Args:
        p: [seq_len, vocab_size] predicted probabilities for current layer
        q: [seq_len, vocab_size] predicted probabilities for next layer

    Returns:
        kl_seq: [seq_len] KL divergence per token
    """
    kl_seq = []
    for i in range(p.shape[0]):
        # Only compute elementwise rel_entr where both are finite; else propagate NaN
        pi, qi = p[i], q[i]
        mask = (~np.isnan(pi)) & (~np.isnan(qi)) & (~np.isinf(pi)) & (~np.isinf(qi))
        if np.any(mask):
            kl_val = np.sum(rel_entr(pi[mask], qi[mask]))
        else:
            kl_val = np.nan
        kl_seq.append(kl_val)
    return np.array(kl_seq)


def unsafe_nwd_probs(p: np.ndarray, q: np.ndarray):
    """Normalized weighted distance (1-cosine similarity)."""
    sim = np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q) + EPS)
    return 1.0 - sim



def _run_logit_lens_unsafe(
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

    # --- Forward pass, keep logits on device ---
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

            # --- Softmax on device ---
            probs_tensor = unsafe_softmax_torch(logits_tensor)

            # --- Predictions & Top-K ---
            preds_seq_tensor = torch.argmax(probs_tensor, dim=-1)
            topk_idx_tensor = torch.topk(probs_tensor, k=topk, dim=-1).indices

            valid_len = min(len(target_ids_seq), preds_seq_tensor.shape[0])
            preds_seq_tensor = preds_seq_tensor[:valid_len]
            probs_tensor = probs_tensor[:valid_len, :] if valid_len > 0 else torch.zeros((0, probs_tensor.shape[-1]), device=device)
            target_ids_tensor = target_ids_seq[:valid_len]

            # --- Correctness metrics on device ---
            if valid_len > 0:
                correct_1_seq_tensor = (preds_seq_tensor == target_ids_tensor).to(torch.float32)
                correct_topk_seq_tensor = torch.stack([torch.any(topk_idx_tensor[i] == target_ids_tensor[i]).to(torch.float32) for i in range(valid_len)])
                correct_1_seq = correct_1_seq_tensor.tolist()
                correct_topk_seq = correct_topk_seq_tensor.tolist()
                correct_1_mean = float(torch.mean(correct_1_seq_tensor))
                correct_topk_mean = float(torch.mean(correct_topk_seq_tensor))
            else:
                correct_1_seq, correct_topk_seq = [], []
                correct_1_mean = correct_topk_mean = None

            all_preds.append(preds_seq_tensor)
            all_topk_preds.append(topk_idx_tensor)

            # --- On-device metrics (NaNs/Infs) ---
            invalid_logits_ratio = float((logits_tensor.isnan() | logits_tensor.isinf()).float().mean())
            invalid_probs_ratio = float((probs_tensor.isnan() | probs_tensor.isinf()).float().mean())
            logit_mean = logits_tensor.mean(dim=-1)
            logit_std = logits_tensor.std(dim=-1)
            logit_var = logits_tensor.var(dim=-1)
            prob_mean = probs_tensor.mean(dim=-1)
            prob_std = probs_tensor.std(dim=-1)
            prob_var = probs_tensor.var(dim=-1)
            logp = torch.log(probs_tensor + EPS)
            ent_seq = -(probs_tensor * logp).sum(dim=-1)
            normalized_ent_seq = ent_seq / np.log(getattr(wrapper.tokenizer, "vocab_size", np.e))
            entropy_nan_ratio = float(ent_seq.isnan().float().mean())

            # --- Convert only for storage / logging ---
            preds_seq = preds_seq_tensor.detach().cpu().numpy()
            probs_seq = probs_tensor.detach().cpu().numpy()
            ent_seq_np = ent_seq.detach().cpu().numpy()
            normalized_ent_seq_np = normalized_ent_seq.detach().cpu().numpy()

            try:
                ece_val = compute_ece(probs_seq, target_ids_tensor.cpu().numpy())
            except Exception:
                ece_val = None

            # --- Ngrams, repetition ---
            ngram_correct, ngram_correct_mean, ngram_stability = {}, {}, {}
            for n in [2, 3]:
                if valid_len >= n:
                    ngram_correct[n] = compute_ngram_matches(preds_seq, target_ids_tensor.cpu().numpy(), n)
                    ngram_correct_mean[n] = np.nanmean(ngram_correct[n])
                    if l < num_layers - 1:
                        next_preds = torch.argmax(valid_layer_logits[l + 1][idx], dim=-1)[:valid_len].cpu().numpy()
                        ngram_stability[n] = compute_ngram_stability(preds_seq, next_preds, n)
                    else:
                        ngram_stability[n] = []
                else:
                    ngram_correct[n], ngram_correct_mean[n], ngram_stability[n] = [], None, []

            repetition_ratio = float((preds_seq_tensor[1:] == preds_seq_tensor[:-1]).float().mean()) if valid_len > 1 else np.nan

            # --- KL / NWD / Top-K drift ---
            if l < num_layers - 1:
                next_probs_tensor = unsafe_softmax_torch(valid_layer_logits[l + 1][idx])[:valid_len, :]
                if valid_len > 0:
                    kl_seq = unsafe_kl_per_token(probs_seq[:valid_len], next_probs_tensor.detach().cpu().numpy())
                    kl_mean = float(np.nanmean(kl_seq))
                    kl_nan_ratio = float(np.mean(np.isnan(kl_seq)))
                    nwd_val = unsafe_nwd_probs(probs_seq[:valid_len].mean(axis=0), next_probs_tensor.detach().cpu().numpy().mean(axis=0))
                    nwd_nan = bool(np.isnan(nwd_val))
                    jacc_mean, tau_mean, jacc_seq, tau_seq = topk_overlap_and_kendall(probs_seq[:valid_len], next_probs_tensor.detach().cpu().numpy(), k=topk)
                    stability_top1.append((torch.argmax(probs_tensor[:valid_len], dim=-1) == torch.argmax(next_probs_tensor, dim=-1)).tolist())
                    stability_topk.append([int(torch.argmax(probs_tensor[i]) in torch.topk(next_probs_tensor[i], topk).indices.tolist()) for i in range(valid_len)])
                else:
                    kl_seq, kl_mean, kl_nan_ratio, nwd_val, nwd_nan, jacc_mean, tau_mean = [], None, None, None, None, None, None
                    jacc_seq = tau_seq = []
                    stability_top1.append([])
                    stability_topk.append([])
            else:
                kl_seq = kl_mean = kl_nan_ratio = nwd_val = nwd_nan = jacc_mean = tau_mean = None
                jacc_seq = tau_seq = []

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
                "invalid_logits_ratio": invalid_logits_ratio,
                "invalid_probs_ratio": invalid_probs_ratio,
                "entropy_mean": float(ent_seq.mean().item()) if ent_seq.numel() > 0 else None,
                "entropy_seq": ent_seq.detach().cpu().tolist() if ent_seq.numel() > 0 else [],
                "entropy_nan_ratio": entropy_nan_ratio,
                "normalized_entropy_mean": float(normalized_ent_seq.mean().item()) if normalized_ent_seq.numel() > 0 else None,
                "normalized_entropy_seq": normalized_ent_seq.detach().cpu().tolist() if normalized_ent_seq.numel() > 0 else [],
                "correct_1_seq": correct_1_seq,
                "correct_topk_seq": correct_topk_seq,
                "correct_1_mean": float(torch.tensor(correct_1_seq, device=device).mean().item()) if correct_1_seq else None,
                "correct_topk_mean": float(torch.tensor(correct_topk_seq, device=device).mean().item()) if correct_topk_seq else None,
                "kl_next_layer_mean": kl_mean if 'kl_mean' in locals() else None,
                "kl_next_layer_seq": kl_seq if 'kl_seq' in locals() else [],
                "kl_nan_ratio": kl_nan_ratio if 'kl_nan_ratio' in locals() else None,
                "nwd": nwd_val if 'nwd_val' in locals() else None,
                "nwd_nan": nwd_nan if 'nwd_nan' in locals() else None,
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


def run_logit_lens_unsafe(
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

    df = _run_logit_lens_unsafe(
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