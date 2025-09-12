from typing import Tuple, List, Any, Optional
import os
import torch
from scipy.special import rel_entr
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr
from scipy.linalg import svd
from ...util.logit_lens_utils.logit_lens_wrapper import LogitLensWrapper



# ----------------------------
# Reusable Inputs
# ----------------------------
EPS = 1e-12
#EPS = 1e-8 
TOPK = 5


def save_degradation_results(
    results:dict,
    save_dir:str,
    model_name:str,
    dataset_name:str,
    mode:str="UNSAFE",  # or "UNSAFE"
):
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{dataset_name}_{model_name}_{mode}.pt"
    torch.save(results, save_path)
    print(f"Saved degradation results to {save_path}")


def load_results_from_pt(results) -> pd.DataFrame:
    # Only do this if you trust the source of the file
    data = torch.load(
        results,
        weights_only=False  # allow full unpickling
    )
    df = pd.DataFrame(data)
    return df

def save_results_to_csv(results:dict, filename:str):
    """
    Save interpretability results dictionary to a CSV file readable by pandas.
    Used to safe e.g., logit_lens_degradation_analysis.py
    Args:
        results (dict): Dictionary returned by analyze_SAFE_interpretability or analyze_UNSAFE_interpretability
        filename (str): Path to save the CSV file (e.g., 'results.csv')
    """
    # Convert nested dictionary to DataFrame
    df = pd.DataFrame.from_dict(results, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'layer_name'}, inplace=True)

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


def extract_activations(wrapper: LogitLensWrapper, prompts: List[str], return_numpy: bool = False):
    """
    Returns per-layer padded hidden tensors and masks.
    By default returns tensors on device/dtype used by model.
    If return_numpy=True, returns CPU numpy arrays (mask still boolean numpy).
    """
    from collections import defaultdict
    per_layer = defaultdict(list)

    for prompt in prompts:
        _, layer_dict, layer_names = wrapper.forward([prompt], project_to_logits=False, return_hidden=True, keep_on_device=True)
        for lname in layer_names:
            t = layer_dict[lname]  # expect tensor [B,S,H] with B==1
            if t.dim() == 3 and t.size(0) == 1:
                t = t[0]
            per_layer[lname].append(t.detach())

    fp = {}
    for lname, seqs in per_layer.items():
        # pad on device of first tensor
        device = seqs[0].device
        dtype = seqs[0].dtype
        maxS = max([s.size(0) for s in seqs])
        H = seqs[0].size(1)
        N = len(seqs)
        stacked = torch.zeros((N, maxS, H), dtype=dtype, device=device)
        mask = torch.zeros((N, maxS), dtype=torch.bool, device=device)
        for i, s in enumerate(seqs):
            L = s.size(0)
            stacked[i, :L, :] = s
            mask[i, :L] = True

        """if return_numpy:
            fp[lname] = {"hidden": stacked.detach().cpu().numpy(), "mask": mask.detach().cpu().numpy()}
        else:
            fp[lname] = {"hidden": stacked, "mask": mask}"""
        if return_numpy:
            fp[lname] = {
                "hidden": stacked.detach().to(torch.float32).cpu().numpy(),  # always FP32
                "mask": mask.detach().cpu().numpy()
            }
        else:
            fp[lname] = {
                "hidden": stacked.detach().to(torch.float32),  # always FP32
                "mask": mask
            }

    return fp


def get_activation_tensor(act):
    """Extract a tensor/array from activation dicts or return as-is."""
    if isinstance(act, dict):
        # Prioritized keys depending on your model wrapper
        for key in ["hidden", "hidden_states", "activations", "values"]:
            if key in act:
                return act[key]
        # If we only find mask-like entries, throw a clearer error
        raise ValueError(
            f"No usable tensor found in activation dict. Keys present: {list(act.keys())}"
        )

    return act  # already tensor or numpy

def align_activations(act1, act2):
    act1 = get_activation_tensor(act1)
    act2 = get_activation_tensor(act2)

    # Convert to numpy if torch
    if hasattr(act1, "detach"):
        act1 = act1.detach().cpu().numpy()
    if hasattr(act2, "detach"):
        act2 = act2.detach().cpu().numpy()

    # Ensure 3D [batch, seq, dim]
    if act1.ndim == 2:
        act1 = act1[None, ...]
    if act2.ndim == 2:
        act2 = act2[None, ...]

    # Align by shared seq_len
    min_len = min(act1.shape[1], act2.shape[1])
    act1 = act1[:, :min_len, :]
    act2 = act2[:, :min_len, :]

    return act1, act2


"""def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins=10):
    #Expected Calibration Error; preserves NaNs/Infs.
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets)
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if np.any(mask):
            avg_conf = np.nanmean(confidences[mask])
            avg_acc = np.nanmean(accuracies[mask])
            ece += np.abs(avg_conf - avg_acc) * np.mean(mask)
    return ece"""

def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE)
    probs: [seq_len, vocab_size] probability distribution (numpy)
    targets: [seq_len] true token ids
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == targets).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.any():
            acc = accuracies[mask].mean()
            conf = confidences[mask].mean()
            ece += np.abs(acc - conf) * mask.mean()
    return float(ece)


def topk_overlap_and_kendall(probs_a, probs_b, k=TOPK):
    """Top-k Jaccard and Kendall Tau per token sequence; NaNs/Infs preserved."""
    seq_len = probs_a.shape[0]
    jacc_seq, tau_seq = [], []
    for i in range(seq_len):
        topk_a = set(np.argsort(probs_a[i])[-k:])
        topk_b = set(np.argsort(probs_b[i])[-k:])
        jacc_seq.append(len(topk_a & topk_b) / max(len(topk_a | topk_b), 1))
        rank_a = np.argsort(np.argsort(-probs_a[i]))
        rank_b = np.argsort(np.argsort(-probs_b[i]))
        tau, _ = kendalltau(rank_a, rank_b)
        tau_seq.append(tau if tau is not np.nan else np.nan)
    return np.nanmean(jacc_seq), np.nanmean(tau_seq), jacc_seq, tau_seq


def compute_ngram_matches(preds, targets, n=2):
    """Binary list if n-gram matches target sequence."""
    pred_ngrams = [tuple(preds[i:i+n]) for i in range(len(preds)-n+1)]
    targ_ngrams = [tuple(targets[i:i+n]) for i in range(len(targets)-n+1)]
    return [int(p == t) for p, t in zip(pred_ngrams, targ_ngrams)]


def compute_ngram_stability(preds_a, preds_b, n=2):
    """Compare n-gram predictions between two layers."""
    ngrams_a = [tuple(preds_a[i:i+n]) for i in range(len(preds_a)-n+1)]
    ngrams_b = [tuple(preds_b[i:i+n]) for i in range(len(preds_b)-n+1)]
    return [int(p == q) for p, q in zip(ngrams_a, ngrams_b)]


def compute_cka(X:np.ndarray, Y:np.ndarray) -> float:
    """
    Linear CKA similarity between two activation matrices.
    X, Y: [seq_len, features]
    Returns: scalar similarity (0-1)
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    hsic = lambda A, B: np.sum(A.T @ B)**2 / ((np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B)) + 1e-12)
    return hsic(X, Y) / (np.sqrt(hsic(X, X) * hsic(Y, Y)) + 1e-12)


def compute_svcca(X:np.ndarray, Y:np.ndarray, svd_thresh=0.99) -> float:
    """
    SVCCA similarity between two activation matrices.
    X, Y: [seq_len, features]
    svd_thresh: fraction of variance to keep
    Returns: mean correlation over top components
    """
    # SVD on X
    Ux, Sx, _ = svd(X - X.mean(axis=0), full_matrices=False)
    cumsum_x = np.cumsum(Sx**2) / np.sum(Sx**2)
    kx = np.searchsorted(cumsum_x, svd_thresh) + 1
    X_svd = (X - X.mean(axis=0)) @ Ux[:, :kx]

    # SVD on Y
    Uy, Sy, _ = svd(Y - Y.mean(axis=0), full_matrices=False)
    cumsum_y = np.cumsum(Sy**2) / np.sum(Sy**2)
    ky = np.searchsorted(cumsum_y, svd_thresh) + 1
    Y_svd = (Y - Y.mean(axis=0)) @ Uy[:, :ky]

    # Align dimensions
    d = min(X_svd.shape[1], Y_svd.shape[1])
    corrs = [pearsonr(X_svd[:, i], Y_svd[:, i])[0] for i in range(d)]
    return np.nanmean(corrs)


def align_features(A, B, max_components=None):
    # Convert tensors to numpy
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    if isinstance(B, torch.Tensor):
        B = B.detach().cpu().numpy()

    # Flatten if 3D
    if A.ndim == 3:
        A = A.reshape(-1, A.shape[-1])
    if B.ndim == 3:
        B = B.reshape(-1, B.shape[-1])

    # Decide number of PCA components
    n_features = min(A.shape[1], B.shape[1])
    n_samples = min(A.shape[0], B.shape[0])
    if max_components is not None:
        n_features = min(n_features, max_components)
    n_features = min(n_features, n_samples)  # <- CAP at #samples

    pca_A = PCA(n_components=n_features)
    pca_B = PCA(n_components=n_features)
    A_proj = pca_A.fit_transform(A)
    B_proj = pca_B.fit_transform(B)
    return A_proj, B_proj


def safe_compute_cka(A, B, eps=EPS):
    # Convert to numpy
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    if isinstance(B, torch.Tensor):
        B = B.detach().cpu().numpy()

    # Ensure feature dimension matches
    if A.shape[1] != B.shape[1]:
        print(f"Skipping CKA: feature dim mismatch {A.shape[1]} vs {B.shape[1]}")
        return np.nan

    # Center
    A_centered = A - A.mean(axis=0, keepdims=True)
    B_centered = B - B.mean(axis=0, keepdims=True)

    # Safe HSIC
    hsic = lambda X, Y: np.sum(X.T @ Y)**2 / ((np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)) + eps)

    return hsic(A_centered, B_centered) / (np.sqrt(hsic(A_centered, A_centered) * hsic(B_centered, B_centered)) + eps)


def safe_compute_svcca(A, B):
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    if isinstance(B, torch.Tensor):
        B = B.detach().cpu().numpy()

    if A.shape[1] != B.shape[1]:
        print(f"Skipping SVCCA: feature dim mismatch {A.shape[1]} vs {B.shape[1]}")
        return np.nan

    return compute_svcca(A, B)


def mean_cosine_similarity(A, B):
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    if isinstance(B, torch.Tensor):
        B = B.detach().cpu().numpy()
    # Average over sequence dimension
    a_mean = np.mean(A, axis=0)
    b_mean = np.mean(B, axis=0)
    # Pad the smaller vector with zeros
    if len(a_mean) < len(b_mean):
        a_mean = np.pad(a_mean, (0, len(b_mean) - len(a_mean)))
    elif len(b_mean) < len(a_mean):
        b_mean = np.pad(b_mean, (0, len(a_mean) - len(b_mean)))
    # Cosine similarity
    return np.dot(a_mean, b_mean) / (np.linalg.norm(a_mean) * np.linalg.norm(b_mean))


def postprocess_logits_topk(
        layer_logits:Any,
        top_n:int=TOPK,
        return_scores:bool=True,
) -> Tuple[Any, Any, Any]:

    layer_probs = torch.softmax(logits=layer_logits)

    layer_preds = layer_logits.argmax(axis=-1)

    top_n_scores = np.mean(
        np.sort(layer_probs, axis=-1)[:, -top_n:], axis=-1
    )

    if return_scores:
        return layer_preds, layer_probs, top_n_scores
    else:
        return layer_preds, layer_probs


def safe_for_bfloat16(t: torch.Tensor) -> torch.Tensor:
    """
    Promote a torch tensor to float32 only if it's bfloat16.
    Leave other dtypes unchanged (so float16, uint8, etc. remain).
    This preserves NaNs/Infs but avoids PyTorch ops failing on bfloat16.
    """
    if isinstance(t, torch.Tensor) and t.dtype == torch.bfloat16:
        return t.float()
    return t

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