import re
import math
import torch
import numpy as np
from typing import List, Tuple, Optional, Union
import pandas as pd

import torch
import numpy as np
import pandas as pd
from typing import Union, List



TOPK = 5
# -----------------------------------------------------------------------------
# Core vectorized metric computation (expects layers_tensor: [L, S, V])
# -----------------------------------------------------------------------------

def compute_carry_over_safe_with_embedding_vectorized(
    layers_tensor: torch.Tensor,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    topk: int = 5,
    enforce_carry_over_safe: bool = False
):
    """
    Vectorized carry-over-safe metrics on a stacked layers tensor.
    
    Args:
        layers_tensor: [L, S, V] (torch.Tensor, logits per layer)
        input_ids: [S] (torch.Tensor, input tokens)
        target_ids: [S] (torch.Tensor, ground-truth tokens)
        topk: number of top-k predictions to consider
        enforce_carry_over_safe: bool, whether to exclude lexical carry-over (predictions == input token)
    
    Returns:
        dict of scalar metrics (float)
    """
    # shapes & types
    L, S, V = layers_tensor.shape
    device = layers_tensor.device
    dtype = layers_tensor.dtype

    # prepare broadcasted inputs
    target_layer = target_ids.to(device).unsqueeze(0).expand(L, -1)  # [L,S]
    input_layer  = input_ids.to(device).unsqueeze(0).expand(L, -1)   # [L,S]

    # ----------------- Top-1 / Top-K predictions -----------------
    top1_preds = layers_tensor.argmax(dim=-1)  # [L,S]
    top1_valid = (top1_preds == target_layer)
    if enforce_carry_over_safe:
        top1_valid &= (top1_preds != input_layer)

    k_eff = min(topk, V)
    topk_inds = layers_tensor.topk(k_eff, dim=-1).indices  # [L,S,k]
    topk_valid = (topk_inds == target_layer.unsqueeze(-1)).any(-1)
    if enforce_carry_over_safe:
        topk_valid &= (target_layer != input_layer)

    # ---------------- Accuracy ----------------
    acc_top1 = top1_valid.any(dim=0).to(dtype)   # [S]
    acc_topk = topk_valid.any(dim=0).to(dtype)   # [S]

    # ---------------- Persistency ----------------
    # ---------------- Persistency ----------------
    if L > 1:
        mask1 = (top1_preds[1:] != input_layer[1:]) if enforce_carry_over_safe else torch.ones_like(top1_preds[1:], dtype=torch.bool)
        maskk = (target_layer[1:] != input_layer[1:]) if enforce_carry_over_safe else torch.ones_like(topk_valid[1:], dtype=torch.bool)

        persistency_top1 = ((top1_preds[1:] == top1_preds[:-1]) & mask1).to(dtype).mean(dim=0)
        persistency_topk = (((topk_inds[1:] == topk_inds[:-1]).all(dim=-1)) & maskk).to(dtype).mean(dim=0)
    else:
        persistency_top1 = torch.zeros(S, dtype=dtype, device=device)
        persistency_topk = torch.zeros(S, dtype=dtype, device=device)


    # ---------------- Consistency ----------------
    cum_top1 = top1_valid.cumsum(dim=0) > 0
    prev_top1 = torch.cat([torch.zeros((1, S), dtype=torch.bool, device=device), cum_top1[:-1]], dim=0)
    first_top1_mask = top1_valid & (~prev_top1)
    exists_top1 = first_top1_mask.any(dim=0)
    first_top1_idx = torch.where(
        exists_top1,
        first_top1_mask.to(dtype).argmax(dim=0).long(),
        torch.full((S,), -1, dtype=torch.long, device=device)
    )

    cum_topk = topk_valid.cumsum(dim=0) > 0
    prev_topk = torch.cat([torch.zeros((1, S), dtype=torch.bool, device=device), cum_topk[:-1]], dim=0)
    first_topk_mask = topk_valid & (~prev_topk)
    exists_topk = first_topk_mask.any(dim=0)
    first_topk_idx = torch.where(
        exists_topk,
        first_topk_mask.to(dtype).argmax(dim=0).long(),
        torch.full((S,), -1, dtype=torch.long, device=device)
    )

    suf_top1_counts = top1_valid.flip(0).cumsum(dim=0).flip(0).to(dtype)
    suf_topk_counts = topk_valid.flip(0).cumsum(dim=0).flip(0).to(dtype)

    consistency_top1 = torch.zeros(S, dtype=dtype, device=device)
    mask1 = first_top1_idx >= 0
    if mask1.any():
        idxs = first_top1_idx[mask1]
        s_idx = mask1.nonzero(as_tuple=True)[0]
        consistency_top1[mask1] = suf_top1_counts[idxs, s_idx] / (L - idxs).to(dtype)

    consistency_topk = torch.zeros(S, dtype=dtype, device=device)
    maskk = first_topk_idx >= 0
    if maskk.any():
        idxs = first_topk_idx[maskk]
        s_idx = maskk.nonzero(as_tuple=True)[0]
        consistency_topk[maskk] = suf_topk_counts[idxs, s_idx] / (L - idxs).to(dtype)

    # ---------------- Earliness ----------------
    if L > 1:
        earliness_top1 = torch.where(first_top1_idx >= 0,
                                     1.0 - first_top1_idx.to(dtype) / (L - 1),
                                     torch.zeros(S, dtype=dtype, device=device))
        earliness_topk = torch.where(first_topk_idx >= 0,
                                     1.0 - first_topk_idx.to(dtype) / (L - 1),
                                     torch.zeros(S, dtype=dtype, device=device))
    else:
        earliness_top1 = torch.zeros(S, dtype=dtype, device=device)
        earliness_topk = torch.zeros(S, dtype=dtype, device=device)

    # ---------------- Aggregate scalars ----------------
    out = {
        'ACC_TOP1': float(acc_top1.mean().item()),
        'ACC_TOPK': float(acc_topk.mean().item()),
        'PERSIST_TOP1': float(persistency_top1.mean().item()),
        'PERSIST_TOPK': float(persistency_topk.mean().item()),
        'CONSISTENCY_TOP1': float(consistency_top1.mean().item()),
        'CONSISTENCY_TOPK': float(consistency_topk.mean().item()),
        'FIRST_TOP1': float(earliness_top1.mean().item()),
        'FIRST_TOPK': float(earliness_topk.mean().item())
    }

    return out


# -------------------------------------------------------
# Helpers to extract canonical layer names and build tensors
# -------------------------------------------------------
def canonical_layer_names_from_df(saved_df: Union[pd.DataFrame, list], prefix: str = '') -> List[str]:
    """
    Return sorted canonical layer names that start with given prefix.
    Accepts either a DataFrame or a list-of-dicts (the result of your original `rows`).
    Accepts names like 'layers.0.self_attn' when prefix='layers.' and will extract the numeric index.
    """
    # Normalize to DataFrame
    if isinstance(saved_df, list):
        df = pd.DataFrame(saved_df)
    elif isinstance(saved_df, pd.DataFrame):
        df = saved_df
    else:
        try:
            df = pd.DataFrame(saved_df)
        except Exception:
            raise TypeError("saved_df must be a DataFrame or a list of dicts")

    if 'layer_name' not in df.columns:
        return []

    names = sorted(set(df['layer_name'].tolist()))
    matches = []
    for n in names:
        if not isinstance(n, str):
            continue
        if not n.startswith(prefix):
            continue
        # look for digits immediately after prefix
        m = re.search(r'^' + re.escape(prefix) + r'(\d+)', n)
        if m:
            matches.append((int(m.group(1)), n))
        else:
            # fallback: include it but assign large index so it sorts after numeric ones
            matches.append((10**6, n))
    matches.sort()
    return [n for _, n in matches]


"""def prepare_layer_tensors_for_prompt(df_prompt: pd.DataFrame, canonical_layers: List[str]):
    
    df_prompt: DataFrame filtered for a single prompt_id with rows for many layer_names
    canonical_layers: ordered list of layer names to include
    Returns: layers_tensor [L, S, V] (torch.Tensor, CPU), input_ids [S] (torch.Tensor), target_ids [S] (torch.Tensor)
    
    if not isinstance(df_prompt, pd.DataFrame):
        df_prompt = pd.DataFrame(df_prompt)

    # map name -> row (row as dict)
    rows_by_name = {}
    for _, r in df_prompt.iterrows():
        rows_by_name[r['layer_name']] = r

    # find a row to extract input/target ids from
    ref_row = None
    for lname in canonical_layers:
        if lname in rows_by_name:
            ref_row = rows_by_name[lname]
            break
    if ref_row is None:
        raise ValueError("No canonical layer rows found to extract input/target ids.")

    input_ids = ref_row['input_ids']
    target_ids = ref_row['target_ids']

    # convert input/target to 1D torch tensors if needed
    def to_1d_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().long().flatten()
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).long().flatten()
        if isinstance(x, (list, tuple)):
            return torch.tensor([int(i) for i in x], dtype=torch.long)
        # fallback
        return torch.tensor([int(x)], dtype=torch.long)

    input_ids_t = to_1d_tensor(input_ids)
    target_ids_t = to_1d_tensor(target_ids)

    layer_tensors = []
    for lname in canonical_layers:
        if lname not in rows_by_name:
            raise KeyError(f"Missing layer {lname} for prompt {df_prompt.get('prompt_id', ['?'])[0]}")
        row = rows_by_name[lname]
        logits = row['logits']

        if isinstance(logits, torch.Tensor):
            l = logits.detach().cpu().to(torch.float)
        elif isinstance(logits, np.ndarray):
            l = torch.from_numpy(logits).float()
        elif isinstance(logits, list):
            # try torch.tensor directly (handles lists of floats/ints)
            try:
                l = torch.tensor(logits, dtype=torch.float)
            except Exception:
                # deeper conversion if inner elements are torch tensors or scalars
                inner = []
                for rrow in logits:
                    inner_row = []
                    for el in rrow:
                        if isinstance(el, torch.Tensor):
                            inner_row.append(float(el.item()))
                        else:
                            inner_row.append(float(el))
                    inner.append(inner_row)
                l = torch.tensor(inner, dtype=torch.float)
        else:
            raise TypeError(f"Unsupported logits type for layer {lname}: {type(logits)}")

        if l.dim() != 2:
            # try to coerce shape to [S,V]
            l = l.reshape(-1, l.shape[-1]) if l.numel() != 0 else l.unsqueeze(0)

        layer_tensors.append(l.to(torch.float))

    layers_tensor = torch.stack(layer_tensors, dim=0)  # [L, S, V] on CPU
    return layers_tensor, input_ids_t, target_ids_t"""

def prepare_layer_tensors_for_prompt(df_prompt: pd.DataFrame, canonical_layers: List[str]):
    """
    Returns: layers_tensor [L, S-1, V], input_ids [S-1], target_ids [S-1].
    If 'target_ids' column is missing, compute as input_ids shifted by 1.
    """
    if not isinstance(df_prompt, pd.DataFrame):
        df_prompt = pd.DataFrame(df_prompt)

    # map name -> row
    rows_by_name = {r['layer_name']: r for _, r in df_prompt.iterrows()}

    # Find a row to extract input IDs
    ref_row = None
    for lname in canonical_layers:
        if lname in rows_by_name:
            ref_row = rows_by_name[lname]
            break
    if ref_row is None:
        raise ValueError("No canonical layer rows found to extract input IDs.")

    input_ids = ref_row['input_ids']

    # Compute target_ids if missing
    if 'target_ids' in ref_row and ref_row['target_ids'] is not None:
        target_ids = ref_row['target_ids']
    else:
        # shift input_ids by 1 to get next-token targets
        if isinstance(input_ids, torch.Tensor):
            target_ids = input_ids[1:]
        else:
            target_ids = input_ids[1:]

    # Convert input/target to 1D tensors
    def to_1d_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().long().flatten()
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).long().flatten()
        if isinstance(x, (list, tuple)):
            return torch.tensor([int(i) for i in x], dtype=torch.long)
        return torch.tensor([int(x)], dtype=torch.long)

    input_ids_t = to_1d_tensor(input_ids)
    target_ids_t = to_1d_tensor(target_ids)

    # Build layer logits tensor
    layer_tensors = []
    for lname in canonical_layers:
        if lname not in rows_by_name:
            raise KeyError(f"Missing layer {lname} for prompt {df_prompt.get('prompt_id', ['?'])[0]}")
        row = rows_by_name[lname]
        logits = row['logits']

        if isinstance(logits, torch.Tensor):
            l = logits.detach().cpu().float()
        elif isinstance(logits, np.ndarray):
            l = torch.from_numpy(logits).float()
        elif isinstance(logits, list):
            l = torch.tensor(logits, dtype=torch.float)
        else:
            raise TypeError(f"Unsupported logits type for layer {lname}: {type(logits)}")

        # Ensure shape [S,V]
        if l.dim() != 2:
            l = l.reshape(-1, l.shape[-1]) if l.numel() != 0 else l.unsqueeze(0)

        # Drop last row so predictions align with target_ids
        l = l[:-1, :]  

        layer_tensors.append(l.float())

    layers_tensor = torch.stack(layer_tensors, dim=0)  # [L,S-1,V]

    # Trim input/target to same length
    min_len = min(layers_tensor.shape[1], target_ids_t.shape[0])
    input_ids_t = input_ids_t[:min_len]
    target_ids_t = target_ids_t[:min_len]
    layers_tensor = layers_tensor[:, :min_len, :]

    return layers_tensor, input_ids_t, target_ids_t


def compute_all_next_token_metrics(saved_df: Union[pd.DataFrame, list], topk: int = 5, prefix: str = ''):
    """
    Computes next-token correctness metrics for all latent predictions in all prompts and layers.
    
    Returns:
        dict: {'per_prompt': {pid: metrics}, 'mean': aggregated}
    """
    if isinstance(saved_df, list):
        df = pd.DataFrame(saved_df)
    elif isinstance(saved_df, pd.DataFrame):
        df = saved_df
    else:
        raise TypeError("saved_df must be DataFrame or list of dicts")

    canonical_layers = canonical_layer_names_from_df(df, prefix=prefix)
    if len(canonical_layers) == 0:
        raise ValueError("No canonical layer names found in df with given prefix")

    results_per_prompt = {}
    prompt_ids = sorted(df['prompt_id'].unique().tolist())

    for pid in prompt_ids:
        df_prompt = df[df['prompt_id'] == pid]
        layers_tensor, input_ids, target_ids = prepare_layer_tensors_for_prompt(df_prompt, canonical_layers)

        # Compute next-token top-1 / top-k correctness for all positions
        top1_preds = layers_tensor.argmax(dim=-1)  # [L, S-1]
        topk_preds = torch.topk(layers_tensor, k=topk, dim=-1).indices  # [L, S-1, topk]

        # Top-1 correctness mask
        top1_correct = (top1_preds == target_ids[None, :])  # [L, S-1]

        # Top-k correctness mask
        topk_correct = (topk_preds == target_ids[None, :, None])  # [L, S-1, topk]
        topk_correct = topk_correct.any(dim=-1)  # [L, S-1]

        # Metrics per layer
        metrics = {}
        for i, lname in enumerate(canonical_layers):
            metrics[lname] = {
                'top1_mean': float(top1_correct[i].float().mean()),
                'topk_mean': float(topk_correct[i].float().mean()),
                'top1_count': int(top1_correct[i].sum()),
                'topk_count': int(topk_correct[i].sum()),
                'sequence_length': int(top1_correct.shape[1])
            }

        results_per_prompt[int(pid)] = metrics

    # Aggregate across prompts
    agg = {}
    keys = canonical_layers
    for k in keys:
        top1_vals = [results_per_prompt[pid][k]['top1_mean'] for pid in prompt_ids]
        topk_vals = [results_per_prompt[pid][k]['topk_mean'] for pid in prompt_ids]
        agg[k] = {
            'top1_mean': float(np.mean(top1_vals)),
            'topk_mean': float(np.mean(topk_vals))
        }

    return {'per_prompt': results_per_prompt, 'mean': agg}

def get_carry_over_safe_with_embedding(saved_df: Union[pd.DataFrame, list], topk: int = TOPK, prefix: str = ''):
    """
    saved_df: pd.DataFrame or list-of-dicts (rows saved by logit-lens)
    Returns: {'per_prompt': {pid: metrics...}, 'mean': aggregated}
    """
    # Normalize to DataFrame for name handling
    if isinstance(saved_df, list):
        df = pd.DataFrame(saved_df)
    elif isinstance(saved_df, pd.DataFrame):
        df = saved_df
    else:
        try:
            df = pd.DataFrame(saved_df)
        except Exception:
            raise TypeError("saved_df must be a DataFrame or a list of dicts")

    canonical_layers = canonical_layer_names_from_df(df, prefix=prefix)
    if len(canonical_layers) == 0:
        raise ValueError("No canonical layer names found in df with given prefix")

    results_per_prompt = {}
    if 'prompt_id' not in df.columns:
        raise KeyError("DataFrame must contain 'prompt_id' column.")
    prompt_ids = sorted(df['prompt_id'].unique().tolist())

    for pid in prompt_ids:
        df_prompt = df[df['prompt_id'] == pid]
        layers_tensor, input_ids, target_ids = prepare_layer_tensors_for_prompt(df_prompt, canonical_layers)
        metrics = compute_carry_over_safe_with_embedding_vectorized(layers_tensor, input_ids, target_ids, topk=topk)
        results_per_prompt[int(pid)] = metrics

    # aggregate mean across prompts
    agg = {}
    keys = list(next(iter(results_per_prompt.values())).keys())
    for k in keys:
        agg[k] = float(np.mean([results_per_prompt[pid][k] for pid in results_per_prompt.keys()]))

    return {'per_prompt': results_per_prompt, 'mean': agg}

