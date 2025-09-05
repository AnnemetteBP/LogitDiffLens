import re
import math
import torch
import numpy as np
from typing import List, Tuple



TOPK = 5
EPS = 1e-8 
# -----------------------------------------------------------------------------
# Core vectorized metric computation (expects layers_tensor: [L, S, V])
# -----------------------------------------------------------------------------
def compute_carry_over_safe_with_embedding_vectorized(layers_tensor:torch.Tensor,
                                                      input_ids:torch.Tensor,
                                                      target_ids:torch.Tensor,
                                                      topk:int=TOPK):

    L, S, V = layers_tensor.shape
    device = layers_tensor.device

    target_b = target_ids.to(device).unsqueeze(0)  # [1,S]
    input_b = input_ids.to(device).unsqueeze(0)    # [1,S]

    # ---------------- Top-1 Predictions ----------------
    top1_preds = layers_tensor.argmax(dim=-1)  # [L,S]
    top1_valid = (top1_preds == target_b) & (top1_preds != input_b)

    # ---------------- Top-K Predictions ----------------
    k_eff = min(topk, V)
    topk_inds = layers_tensor.topk(k_eff, dim=-1).indices  # [L,S,k]
    topk_valid = ((topk_inds == target_b.unsqueeze(-1)).any(-1)) & (target_b != input_b)

    # ---------------- Accuracy ----------------
    acc_top1 = top1_valid.any(dim=0).float()  # [S]
    acc_topk = topk_valid.any(dim=0).float()  # [S]

    # ---------------- Persistency (old stability) ----------------
    # compare layer l vs l-1 for same token, ignore carry-over
    persistency_top1 = ((top1_preds[1:] == top1_preds[:-1]) & (top1_preds[1:] != input_b)).float().mean(dim=0)
    persistency_topk = ((topk_inds[1:] == topk_inds[:-1]).all(dim=-1) & (target_b != input_b)).float().mean(dim=0)

    # ---------------- Consistency ----------------
    # first correct occurrence
    cum_top1 = top1_valid.cumsum(dim=0) > 0
    prev_top1 = torch.cat([torch.zeros((1, S), dtype=torch.bool, device=device), cum_top1[:-1]], dim=0)
    first_top1_mask = top1_valid & (~prev_top1)
    exists_top1 = first_top1_mask.any(dim=0)
    first_top1_idx = torch.where(exists_top1,
                                 first_top1_mask.float().argmax(dim=0).long(),
                                 torch.full((S,), -1, dtype=torch.long, device=device))

    cum_topk = topk_valid.cumsum(dim=0) > 0
    prev_topk = torch.cat([torch.zeros((1, S), dtype=torch.bool, device=device), cum_topk[:-1]], dim=0)
    first_topk_mask = topk_valid & (~prev_topk)
    exists_topk = first_topk_mask.any(dim=0)
    first_topk_idx = torch.where(exists_topk,
                                 first_topk_mask.float().argmax(dim=0).long(),
                                 torch.full((S,), -1, dtype=torch.long, device=device))

    # Consistency = fraction of remaining layers after first correct where prediction stays valid
    suf_top1_counts = top1_valid.flip(0).cumsum(dim=0).flip(0).float()
    suf_topk_counts = topk_valid.flip(0).cumsum(dim=0).flip(0).float()

    consistency_top1 = torch.zeros(S, device=device)
    mask1 = first_top1_idx >= 0
    if mask1.any():
        idxs = first_top1_idx[mask1]
        s_idx = mask1.nonzero(as_tuple=True)[0]
        consistency_top1[mask1] = suf_top1_counts[idxs, s_idx] / (L - idxs).float()

    consistency_topk = torch.zeros(S, device=device)
    maskk = first_topk_idx >= 0
    if maskk.any():
        idxs = first_topk_idx[maskk]
        s_idx = maskk.nonzero(as_tuple=True)[0]
        consistency_topk[maskk] = suf_topk_counts[idxs, s_idx] / (L - idxs).float()

    # ---------------- Earliness / First-layer-correct ----------------
    earliness_top1 = torch.where(first_top1_idx >= 0,
                                 1.0 - first_top1_idx.float() / (L - 1),
                                 torch.zeros(S, device=device))
    earliness_topk = torch.where(first_topk_idx >= 0,
                                 1.0 - first_topk_idx.float() / (L - 1),
                                 torch.zeros(S, device=device))

    # ---------------- Aggregate scalars ----------------
    out = {
        'acc_top1_scalar': float(acc_top1.mean().item()),
        'acc_topk_scalar': float(acc_topk.mean().item()),
        'persistency_top1_scalar': float(persistency_top1.mean().item()),
        'persistency_topk_scalar': float(persistency_topk.mean().item()),
        'consistency_top1_scalar': float(consistency_top1.mean().item()),
        'consistency_topk_scalar': float(consistency_topk.mean().item()),
        'earliness_top1_scalar': float(earliness_top1.mean().item()),
        'earliness_topk_scalar': float(earliness_topk.mean().item())
    }

    return out


# -----------------------------------------------------------------------------
# Helpers to build layers_tensor from your saved DataFrame layout
# -----------------------------------------------------------------------------
def canonical_layer_names_from_df(df, prefix='layers.'):
    """Return sorted canonical layer names e.g. ['layers.0', 'layers.1', ...] present in df."""
    names = sorted(set(df['layer_name'].tolist()))
    pattern = re.compile(r'^' + re.escape(prefix) + r'(\d+)$')
    matches = []
    for n in names:
        m = pattern.match(n)
        if m:
            matches.append((int(m.group(1)), n))
    matches.sort()
    return [n for _, n in matches]


def prepare_layer_tensors_for_prompt(df_prompt, canonical_layers: List[str]):
    """
    df_prompt: DataFrame filtered for a single prompt_id with rows for many layer_names (one row per layer)
    canonical_layers: list of layer names (in order) to include. Each row must have 'logits' as a torch Tensor [S,V],
                     and at least one row must contain 'input_ids' and 'target_ids' tensors (we take from the first canonical layer).
    Returns: layers_tensor [L, S, V], input_ids [S], target_ids [S]
    """
    rows_by_name = {row['layer_name']: row for _, row in df_prompt.iterrows()}
    # Get input/target from the first canonical layer that exists
    for lname in canonical_layers:
        if lname in rows_by_name:
            ref_row = rows_by_name[lname]
            input_ids = ref_row['input_ids']
            target_ids = ref_row['target_ids']
            break
    else:
        raise ValueError("No canonical layer rows found to extract input/target ids.")

    # Convert input/target to tensors if needed
    if isinstance(input_ids, list) or isinstance(input_ids, np.ndarray):
        input_ids = torch.tensor([int(x) for x in input_ids], dtype=torch.long)
    if isinstance(target_ids, list) or isinstance(target_ids, np.ndarray):
        target_ids = torch.tensor([int(x) for x in target_ids], dtype=torch.long)

    layer_tensors = []
    for lname in canonical_layers:
        if lname not in rows_by_name:
            raise KeyError(f"Missing layer {lname} for prompt {df_prompt['prompt_id'].iloc[0]}")
        row = rows_by_name[lname]
        logits = row['logits']
        # Ensure logits are torch.Tensor [S,V]
        if not isinstance(logits, torch.Tensor):
            # if it's a list-of-lists etc:
            logits = torch.stack([torch.tensor(x, dtype=torch.float) if not isinstance(x, torch.Tensor) else x
                                  for x in logits])
        layer_tensors.append(logits.to(torch.float))  # keep on CPU to avoid GPU memory spikes

    layers_tensor = torch.stack(layer_tensors, dim=0)  # [L, S, V]
    return layers_tensor, torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)


def get_carry_over_safe_with_embedding(saved_df, topk=5, prefix='layers.'):
    """
    saved_df: your loaded DataFrame (torch.load .pt returns the df)
    Returns a dict with per-prompt results and a global mean.
    """
    canonical_layers = canonical_layer_names_from_df(saved_df, prefix=prefix)
    if len(canonical_layers) == 0:
        raise ValueError("No canonical 'layers.N' names found in df")

    results_per_prompt = {}
    # group by prompt_id
    if 'prompt_id' not in saved_df.columns:
        raise KeyError("DataFrame must contain 'prompt_id' column.")
    prompt_ids = sorted(saved_df['prompt_id'].unique().tolist())

    for pid in prompt_ids:
        df_prompt = saved_df[saved_df['prompt_id'] == pid]
        layers_tensor, input_ids, target_ids = prepare_layer_tensors_for_prompt(df_prompt, canonical_layers)
        metrics = compute_carry_over_safe_with_embedding_vectorized(layers_tensor, input_ids, target_ids, topk=topk)
        results_per_prompt[pid] = metrics

    # compute simple average across prompts (equal weighting)
    agg = {}
    keys = list(next(iter(results_per_prompt.values())).keys())
    for k in keys:
        agg[k] = float(np.mean([results_per_prompt[pid][k] for pid in prompt_ids]))

    return {'per_prompt': results_per_prompt, 'mean': agg}
