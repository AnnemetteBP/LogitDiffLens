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


import numpy as np

def div_stability_top1_safe(preds_layers: np.ndarray, input_ids: np.ndarray, target_ids: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Divergence-aware, carry-over safe Top-1 stability per token.
    
    Args:
        preds_layers: [L, T] predicted token IDs across layers (L layers, T tokens)
        input_ids:    [T] original input token IDs
        target_ids:   [T] target token IDs
        normalize:    if True, divides stability by L to return value in [0,1]
        
    Returns:
        stability: [T] stability per token
    """
    L, T = preds_layers.shape
    stability = np.zeros(T, dtype=float)

    for t in range(T):
        prev_preds = set([input_ids[t]])  # track input + previous predictions
        target = target_ids[t]

        first_correct_layer_found = False
        for l in range(L):
            pred = preds_layers[l, t]
            if pred == target and pred not in prev_preds:
                stability[t] = (l + 1) / L if normalize else (l + 1)
                first_correct_layer_found = True
                break
            prev_preds.add(pred)

        # If all layers are correct but repeated predictions prevented counting
        if not first_correct_layer_found and np.all(preds_layers[:, t] == target) and target not in prev_preds:
            stability[t] = 1.0 if normalize else L

    return stability


def div_stability_topk_safe(topk_layers: np.ndarray, input_ids: np.ndarray, target_ids: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Divergence-aware, carry-over safe Top-k stability per token.
    
    Args:
        topk_layers: [L, T, k] predicted token IDs across layers
        input_ids:   [T] original input token IDs
        target_ids:  [T] target token IDs
        normalize:   if True, divides stability by L to return value in [0,1]
        
    Returns:
        stability: [T] stability per token
    """
    L, T, k = topk_layers.shape
    stability = np.zeros(T, dtype=float)

    for t in range(T):
        prev_preds = set([input_ids[t]])
        target = target_ids[t]

        first_correct_layer_found = False
        for l in range(L):
            layer_preds = set(topk_layers[l, t])
            if target in layer_preds and target not in prev_preds:
                stability[t] = (l + 1) / L if normalize else (l + 1)
                first_correct_layer_found = True
                break
            prev_preds.update(layer_preds)

        # Safeguard for repeated predictions across all layers
        if not first_correct_layer_found and all(target in set(topk_layers[l, t]) for l in range(L)) and target not in prev_preds:
            stability[t] = 1.0 if normalize else L

    return stability


def div_accuracy_top1_safe(preds_layers: np.ndarray, input_ids: np.ndarray, target_ids: np.ndarray, normalize: bool = True):
    """
    Divergence-aware, carry-over safe accuracy for Top-1 predictions.
    """
    L, T = preds_layers.shape
    acc = np.zeros(T, dtype=float)

    for t in range(T):
        preds = preds_layers[:, t]
        target = target_ids[t]
        inp = input_ids[t]

        prev_preds = {inp}
        found = False
        for l in range(L):
            if preds[l] == target and target not in prev_preds:
                acc[t] = 1.0 if normalize else 1
                found = True
                break
            prev_preds.add(preds[l])

        # if all layers are correct but previously counted, assign max
        if not found and np.all(preds == target):
            acc[t] = 1.0 if normalize else 1

    return acc


def div_accuracy_topk_safe(topk_layers: np.ndarray, input_ids: np.ndarray, target_ids: np.ndarray, normalize: bool = True):
    """
    Divergence-aware, carry-over safe accuracy for Top-k predictions.
    """
    L, T, k = topk_layers.shape
    acc = np.zeros(T, dtype=float)

    for t in range(T):
        topk_preds = topk_layers[:, t, :]
        target = target_ids[t]
        inp = input_ids[t]

        prev_preds = {inp}
        found = False
        for l in range(L):
            layer_preds = set(topk_preds[l])
            if target in layer_preds and target not in prev_preds:
                acc[t] = 1.0 if normalize else 1
                found = True
                break
            prev_preds.update(layer_preds)

        # if all layers always include target but it was previously “seen”
        if not found and all(target in set(topk_preds[l]) for l in range(L)):
            acc[t] = 1.0 if normalize else 1

    return acc
