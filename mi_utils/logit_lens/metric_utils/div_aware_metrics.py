import numpy as np



def layer_partitions(L):
    """
    Return indices for first, early, mid, late, last layer partitions.
    """
    first = [0]
    last = [L-1]
    thirds = max(1, L//3)
    early = list(range(1, 1+thirds))
    mid   = list(range(1+thirds, 1+2*thirds))
    late  = list(range(1+2*thirds, L-1))
    return first, early, mid, late, last


def div_stability_top1(
    preds_layers:np.ndarray,
    input_ids:np.ndarray,
    target_ids:np.ndarray,
    normalize:bool=True
):
    """
    Divergence-aware stability for Top-1 predictions.
    preds_layers: [L, seq_len] token IDs
    input_ids:    [seq_len] input tokens
    target_ids:   [seq_len] target tokens
    """
    L, T = preds_layers.shape
    stability = np.zeros(T, dtype=float)

    for t in range(T):
        preds = preds_layers[:, t]
        target = target_ids[t]
        inp = input_ids[t]

        # Case 1: always correct
        if np.all(preds == target):
            stability[t] = 1 if not normalize else 1.0
            continue

        # Case 2: find earliest correct layer with earlier divergence
        found = False
        for l in range(L):
            if preds[l] == target:
                if np.any(preds[:l] != inp):
                    stability[t] = (l+1)/L if normalize else (l+1)
                    found = True
                    break
        if not found:
            stability[t] = 0.0

    return stability


def div_stability_topk(
    topk_layers:np.ndarray,
    input_ids: np.ndarray,
    target_ids: np.ndarray,
    normalize:bool=True
):
    """
    Divergence-aware stability for Top-k predictions.
    topk_layers: [L, seq_len, k] token IDs
    """
    L, T, k = topk_layers.shape
    stability = np.zeros(T, dtype=float)

    for t in range(T):
        topk_preds = topk_layers[:, t, :]
        target = target_ids[t]
        inp = input_ids[t]

        # Case 1: always correct
        if np.all([target in row for row in topk_preds]):
            stability[t] = 1 if not normalize else 1.0
            continue

        # Case 2: find earliest correct layer with earlier divergence
        found = False
        for l in range(L):
            if target in topk_preds[l]:
                if not any(inp in topk_preds[lprime] for lprime in range(l)):
                    stability[t] = (l+1)/L if normalize else (l+1)
                    found = True
                    break
        if not found:
            stability[t] = 0.0
    return stability


def div_accuracy_top1(
    preds_layers:np.ndarray,
    input_ids:np.ndarray,
    target_ids:np.ndarray,
    normalize:bool=True
):
    """
    Divergence-aware accuracy for Top-1 predictions.
    """
    L, T = preds_layers.shape
    acc = np.zeros(T, dtype=float)

    for t in range(T):
        preds = preds_layers[:, t]
        target = target_ids[t]
        inp = input_ids[t]

        # Case 1: always correct
        if np.all(preds == target):
            acc[t] = 1 if not normalize else 1.0
            continue

        # Case 2: exists correct with earlier divergence
        if any((preds[l] == target) and np.any(preds[:l] != inp) for l in range(L)):
            acc[t] = 1 if not normalize else 1.0
        else:
            acc[t] = 0.0
    return acc


def div_accuracy_topk(
    topk_layers:np.ndarray,
    input_ids:np.ndarray,
    target_ids:np.ndarray,
    normalize:bool=True
):
    """
    Divergence-aware accuracy for Top-k predictions.
    """
    L, T, k = topk_layers.shape
    acc = np.zeros(T, dtype=float)

    for t in range(T):
        topk_preds = topk_layers[:, t, :]
        target = target_ids[t]
        inp = input_ids[t]

        # Case 1: always correct
        if np.all([target in row for row in topk_preds]):
            acc[t] = 1 if not normalize else 1.0
            continue

        # Case 2: exists correct with earlier divergence
        if any((target in topk_preds[l]) and not any(inp in topk_preds[lprime] for lprime in range(l)) for l in range(L)):
            acc[t] = 1 if not normalize else 1.0
        else:
            acc[t] = 0.0
    return acc
