import re
import math
import torch
import numpy as np
from typing import List, Tuple, Optional



TOPK = 5
# -----------------------------------------------------------------------------
# Core vectorized metric computation (expects layers_tensor: [L, S, V])
# -----------------------------------------------------------------------------
# paste into mi_utils/logit_lens/metric_utils/carry_over_safe_metrics.py
import re
from typing import List, Optional, Union
import numpy as np
import torch
import pandas as pd

# Keep using module-level TOPK/EPS if present; functions accept topk param too.

def compute_carry_over_safe_with_embedding_vectorized(
    layers_tensor: torch.Tensor,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    topk: int = TOPK,
    pad_token_id: Optional[int] = 0
):
    """
    Vectorized carry-over-safe metrics on a stacked layers tensor.
    layers_tensor: [L, S, V] (torch.Tensor, CPU or device)
    input_ids: [S] (torch.Tensor)
    target_ids: [S] (torch.Tensor)
    Returns: dict of scalar metrics (same keys you had before)
    """
    # shapes & types
    L, S, V = layers_tensor.shape
    device = layers_tensor.device
    dtype = layers_tensor.dtype

    # ensure 1D input/target on same device
    target_b = target_ids.to(device).unsqueeze(0)  # [1,S]
    input_b = input_ids.to(device).unsqueeze(0)    # [1,S]
    pad_mask_b = (input_b != pad_token_id)         # [1,S]

    # Expand to layer dimension so slices like [1:] make sense
    if L > 1:
        target_layer = target_b.repeat(L, 1)    # [L, S]
        input_layer = input_b.repeat(L, 1)      # [L, S]
        pad_mask_layer = pad_mask_b.repeat(L, 1)# [L, S]
    else:
        # L == 1; keep shapes consistent
        target_layer = target_b.clone()
        input_layer = input_b.clone()
        pad_mask_layer = pad_mask_b.clone()

    # ----------------- Top-1 / Top-K predictions -----------------
    top1_preds = layers_tensor.argmax(dim=-1)  # [L,S]
    # top1_valid: whether predicted token equals target and not just copied from input & not pad
    top1_valid = (top1_preds == target_layer) & (top1_preds != input_layer) & pad_mask_layer

    k_eff = min(topk, V)
    topk_inds = layers_tensor.topk(k_eff, dim=-1).indices  # [L,S,k]
    topk_valid = ((topk_inds == target_b.unsqueeze(-1)).any(-1)) & (target_layer != input_layer) & pad_mask_layer

    # ---------------- Accuracy ----------------
    acc_top1 = top1_valid.any(dim=0).to(dtype)   # [S]
    acc_topk = topk_valid.any(dim=0).to(dtype)   # [S]

    # ---------------- Persistency ----------------
    if L > 1:
        persistency_top1 = (
            ((top1_preds[1:] == top1_preds[:-1]) & (top1_preds[1:] != input_layer[1:]) & pad_mask_layer[1:])
            .to(dtype)
            .mean(dim=0)
        )
        persistency_topk = (
            ((topk_inds[1:] == topk_inds[:-1]).all(dim=-1) & (target_layer[1:] != input_layer[1:]) & pad_mask_layer[1:])
            .to(dtype)
            .mean(dim=0)
        )
    else:
        # No temporal layers to compare -> zeros
        persistency_top1 = torch.zeros(S, dtype=dtype, device=device)
        persistency_topk = torch.zeros(S, dtype=dtype, device=device)

    # ---------------- Consistency ----------------
    cum_top1 = top1_valid.cumsum(dim=0) > 0                              # [L,S]
    prev_top1 = torch.cat([torch.zeros((1, S), dtype=torch.bool, device=device), cum_top1[:-1]], dim=0)
    first_top1_mask = top1_valid & (~prev_top1)                           # [L,S]
    exists_top1 = first_top1_mask.any(dim=0)                              # [S]
    # Use argmax on boolean -> index of first True per column
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

    # Consistency: suf_counts = number of valid positions from first occurrence onward
    suf_top1_counts = top1_valid.flip(0).cumsum(dim=0).flip(0).to(dtype)
    suf_topk_counts = topk_valid.flip(0).cumsum(dim=0).flip(0).to(dtype)

    consistency_top1 = torch.zeros(S, dtype=dtype, device=device)
    mask1 = first_top1_idx >= 0
    if mask1.any():
        idxs = first_top1_idx[mask1]
        s_idx = mask1.nonzero(as_tuple=True)[0]
        # gather counts: suf_top1_counts[idxs, s_idx]
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

    # ---------------- Aggregate scalars (mean across positions) ----------------
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


def prepare_layer_tensors_for_prompt(df_prompt: pd.DataFrame, canonical_layers: List[str]):
    """
    df_prompt: DataFrame filtered for a single prompt_id with rows for many layer_names
    canonical_layers: ordered list of layer names to include
    Returns: layers_tensor [L, S, V] (torch.Tensor, CPU), input_ids [S] (torch.Tensor), target_ids [S] (torch.Tensor)
    """
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
    return layers_tensor, input_ids_t, target_ids_t


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



def compute_carry_over_safe_partitioned(
    saved_df,
    topk: int = TOPK,
    prefix: str = "layers.",
    partitions: Optional[dict] = None
):
    """
    saved_df: the DataFrame from your logit-lens run
    topk: top-k for metrics
    prefix: layer name prefix
    partitions: optional dict {'early': (0, 0.25), 'mid': (0.25,0.5), ...} as fractions of total layers
                if None, default 4 equal partitions are used

    Returns: dict with per-prompt and per-partition averages
    """
    canonical_layers = canonical_layer_names_from_df(saved_df, prefix=prefix)
    if len(canonical_layers) == 0:
        raise ValueError("No canonical 'layers.N' names found in df")
    num_layers = len(canonical_layers)

    # default 4 partitions
    if partitions is None:
        partitions = {
            "early": (0.0, 0.25),
            "mid": (0.25, 0.5),
            "late": (0.5, 0.75),
            "last": (0.75, 1.0)
        }

    # build partition layer indices
    partition_indices = {}
    for name, (start_frac, end_frac) in partitions.items():
        start_idx = int(start_frac * num_layers)
        end_idx = int(end_frac * num_layers)
        partition_indices[name] = list(range(start_idx, end_idx))

    results_per_prompt = {}
    prompt_ids = sorted(saved_df['prompt_id'].unique().tolist())

    for pid in prompt_ids:
        df_prompt = saved_df[saved_df['prompt_id'] == pid]
        layers_tensor, input_ids, target_ids = prepare_layer_tensors_for_prompt(df_prompt, canonical_layers)

        metrics_per_partition = {}
        for pname, idxs in partition_indices.items():
            if len(idxs) == 0:
                continue
            layers_sub = layers_tensor[idxs]  # [L_sub, S, V]
            metrics = compute_carry_over_safe_with_embedding_vectorized(layers_sub, input_ids, target_ids, topk=topk)
            metrics_per_partition[pname] = metrics

        results_per_prompt[pid] = metrics_per_partition

    # compute global mean per partition
    all_partitions = partitions.keys()
    agg = {pname: {} for pname in all_partitions}
    for pname in all_partitions:
        keys = list(next(iter(results_per_prompt.values()))[pname].keys())
        for k in keys:
            agg[pname][k] = float(np.mean([results_per_prompt[pid][pname][k] for pid in prompt_ids]))

    return {"per_prompt": results_per_prompt, "per_partition_mean": agg}
