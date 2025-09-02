from typing import Any
import math
import numpy as np
import pandas as pd

# ----------------------
# Interpretability scoring
# ----------------------
def degradation_diff_score(df:pd.DataFrame, baseline_entropy:float=None) -> tuple[dict, float]:
    """
    Compute interpretability score per section and overall.
    Incorporates KL, NWD, top-k Jaccard, ECE, invalid ratios, repetition, and geometry shift (CKA).
    """
    n_layers = df['layer_index'].max() + 1
    section_bounds = {
        "first": (0, max(1, n_layers // 10)),
        "early": (max(1, n_layers // 10), max(1, n_layers // 3)),
        "mid": (max(1, n_layers // 3), max(1, 2 * n_layers // 3)),
        "late": (max(1, 2 * n_layers // 3), max(1, 9 * n_layers // 10)),
        "last": (max(1, 9 * n_layers // 10), n_layers),
    }

    section_scores = {}
    for section, (start, end) in section_bounds.items():
        df_sec = df[(df['layer_index'] >= start) & (df['layer_index'] < end)]
        if len(df_sec) == 0:
            section_scores[section] = np.nan
            continue

        # Original metrics
        score_kl = 1.0 - np.nanmean(df_sec['kl_next_layer_mean'].values)
        score_nwd = 1.0 - np.nanmean(df_sec['nwd'].values)
        score_jacc = np.nanmean(df_sec['topk_jaccard_mean'].values)
        score_ece = 1.0 - np.nanmean(df_sec['ece'].values)

        # New metrics
        score_invalid = 1.0 - np.nanmean(df_sec['invalid_probs_ratio'].values)
        score_repetition = 1.0 - np.nanmean(df_sec['repetition_ratio'].values)
        score_geometry = np.nanmean(df_sec['cka_fp_vs_layer'].values)

        # Optional: entropy drift relative to baseline FP
        if baseline_entropy is not None:
            score_entropy = 1.0 - np.nanmean(np.abs(df_sec['normalized_entropy_mean'].values - baseline_entropy))
        else:
            score_entropy = 0.0

        # Combine metrics with example weights (adjustable)
        section_scores[section] = (
            0.15*score_kl +
            0.1*score_nwd +
            0.25*score_jacc +
            0.1*score_ece +
            0.1*score_invalid +
            0.1*score_repetition +
            0.15*score_geometry +
            0.05*score_entropy
        )

    overall_score = np.nanmean(list(section_scores.values()))
    return section_scores, overall_score



def interpretability_diff_score(df:pd.DataFrame, df_fp:pd.DataFrame) -> tuple[dict, float]:
    """
    Compare quantized/compressed model to FP baseline for interpretability.
    Returns per-section and overall degradation score (0 = worst, 1 = identical to FP).
    """
    n_layers = df['layer_index'].max() + 1
    section_bounds = {
        "first": (0, max(1, n_layers // 10)),
        "early": (max(1, n_layers // 10), max(1, n_layers // 3)),
        "mid": (max(1, n_layers // 3), max(1, 2 * n_layers // 3)),
        "late": (max(1, 2 * n_layers // 3), max(1, 9 * n_layers // 10)),
        "last": (max(1, 9 * n_layers // 10), n_layers),
    }

    section_scores = {}
    for section, (start, end) in section_bounds.items():
        df_sec = df[(df['layer_index'] >= start) & (df['layer_index'] < end)]
        df_fp_sec = df_fp[(df_fp['layer_index'] >= start) & (df_fp['layer_index'] < end)]
        if len(df_sec) == 0 or len(df_fp_sec) == 0:
            section_scores[section] = np.nan
            continue

        # Compute per-metric deviation from FP
        score_kl = 1.0 - np.nanmean(np.abs(df_sec['kl_next_layer_mean'].values - df_fp_sec['kl_next_layer_mean'].values))
        score_nwd = 1.0 - np.nanmean(np.abs(df_sec['nwd'].values - df_fp_sec['nwd'].values))
        score_jacc = np.nanmean(df_sec['topk_jaccard_mean'].values)  # or use top-k overlap with FP
        score_ece = 1.0 - np.nanmean(np.abs(df_sec['ece'].values - df_fp_sec['ece'].values))
        score_invalid = 1.0 - np.nanmean(df_sec['invalid_probs_ratio'].values)
        score_repetition = 1.0 - np.nanmean(df_sec['repetition_ratio'].values)
        score_geometry = np.nanmean(df_sec['cka_fp_vs_layer'].values)  # already compares to FP

        # Combine metrics (adjust weights if desired)
        section_scores[section] = (
            0.15*score_kl +
            0.1*score_nwd +
            0.25*score_jacc +
            0.1*score_ece +
            0.1*score_invalid +
            0.1*score_repetition +
            0.15*score_geometry
        )

    overall_score = np.nanmean(list(section_scores.values()))
    return section_scores, overall_score



def simple_interpretability_score(df:pd.DataFrame) -> tuple[dict,float]:
    n_layers = df['layer_index'].max() + 1
    section_bounds = {
        "first": (0, max(1, n_layers // 10)),
        "early": (max(1, n_layers // 10), max(1, n_layers // 3)),
        "mid": (max(1, n_layers // 3), max(1, 2 * n_layers // 3)),
        "late": (max(1, 2 * n_layers // 3), max(1, 9 * n_layers // 10)),
        "last": (max(1, 9 * n_layers // 10), n_layers),
    }

    section_scores = {}
    for section, (start, end) in section_bounds.items():
        df_sec = df[(df['layer_index'] >= start) & (df['layer_index'] < end)]
        if len(df_sec) == 0:
            section_scores[section] = np.nan
            continue
        
        score_kl = 1.0 - np.nanmean(df_sec['kl_next_layer_mean'].values)  # lower KL is better
        score_nwd = 1.0 - np.nanmean(df_sec['nwd'].values)               # lower NWD is better
        score_jacc = np.nanmean(df_sec['topk_jaccard_mean'].values)      # higher Jacc is better
        score_ece = 1.0 - np.nanmean(df_sec['ece'].values)               # lower ECE is better
        
        # Combine with weights (example: heavier on top-k and smoothness)
        section_scores[section] = 0.25*score_kl + 0.2*score_nwd + 0.4*score_jacc + 0.15*score_ece

    overall_score = np.nanmean(list(section_scores.values()))
    return section_scores, overall_score


def interpretability_SAFE_score(df: pd.DataFrame) -> dict[str, Any]:
    """
    Produce a partitioned interpretability score for the model represented by df.

    Components:
      - nan_rate: fraction of entries with NaN in kl_next_layer_mean, nwd, or ece
      - mean_kl: mean KL across layers
      - mean_nwd: mean NWD across layers
      - mean_topk_jacc: mean topk Jaccard
      - mean_ece: mean ECE
    Layers are partitioned into first/early/mid/late/last for cross-model comparison.

    Returns:
      dict with per-section scores and overall score.
    """
    n_layers = df['layer_index'].max() + 1

    section_bounds = {
        "first": (0, max(1, n_layers // 10)),
        "early": (max(1, n_layers // 10), max(1, n_layers // 3)),
        "mid": (max(1, n_layers // 3), max(1, 2 * n_layers // 3)),
        "late": (max(1, 2 * n_layers // 3), max(1, 9 * n_layers // 10)),
        "last": (max(1, 9 * n_layers // 10), n_layers),
    }

    def safe_mean(col):
        vals = [float(x) for x in col if x is not None and not (isinstance(x, float) and math.isnan(x))]
        return float(np.nanmean(vals)) if vals else 0.0

    def inv_clip(x, clip=1.0):
        return max(0.0, 1.0 - safe_mean(x)/clip)

    section_scores = {}
    for section, (start, end) in section_bounds.items():
        df_sec = df[(df['layer_index'] >= start) & (df['layer_index'] < end)]
        if df_sec.empty:
            section_scores[section] = np.nan
            continue

        nan_count = sum(
            1 for v in df_sec['kl_next_layer_mean'] if v is None or (isinstance(v, float) and math.isnan(v))
        )
        nan_rate = nan_count / max(1, len(df_sec))

        mean_kl = safe_mean(df_sec['kl_next_layer_mean'])
        mean_nwd = safe_mean(df_sec['nwd'])
        mean_jacc = safe_mean(df_sec['topk_jaccard_mean'])
        mean_ece = safe_mean(df_sec['ece'])

        # normalize and combine
        score_kl = inv_clip(mean_kl, clip=2.0)
        score_nwd = inv_clip(mean_nwd, clip=0.5)
        score_jacc = mean_jacc
        score_ece = inv_clip(mean_ece, clip=0.2)

        section_score = 0.3*score_jacc + 0.25*score_kl + 0.2*score_nwd + 0.2*score_ece + 0.05*(1.0 - nan_rate)
        section_scores[section] = section_score

    overall_score = float(np.nanmean(list(section_scores.values())))

    return {
        "section_scores": section_scores,
        "overall_score": overall_score
    }


def degradation_score(results:dict) -> tuple[dict,float]:
    """
    Interpretability Evaluation:
      • Partition layers for cross-model comparison.
      • Scores degradation in interpretability from UNSAFE analysis.
      • 1.0 → perfect interpretability, i.e., no NaNs or Infs were encountered in logits, probabilities, or entropy across all layers.
      • 0.0 → extremely degraded, i.e., the layers are mostly NaN/Inf, making the latent structure effectively uninterpretable.
    """
    layer_names = list(results.keys())
    n_layers = len(layer_names)
    section_bounds = {
        "first": (0, max(1, n_layers // 10)),
        "early": (max(1, n_layers // 10), max(1, n_layers // 3)),
        "mid": (max(1, n_layers // 3), max(1, 2 * n_layers // 3)),
        "late": (max(1, 2 * n_layers // 3), max(1, 9 * n_layers // 10)),
        "last": (max(1, 9 * n_layers // 10), n_layers),
    }

    section_scores = {}
    for section, (start, end) in section_bounds.items():
        section_layers = layer_names[start:end]
        if not section_layers:
            section_scores[section] = np.nan
            continue

        rates = []
        for lname in section_layers:
            r = results[lname]
            rates.append(max(
                r["nan_logits_rate"],
                r["inf_logits_rate"],
                r["nan_probs_rate"],
                r["inf_probs_rate"],
                r["nan_entropy_rate"],
                r["inf_entropy_rate"],
            ))
        section_scores[section] = float(1.0 - np.mean(rates))

    overall_score = float(np.nanmean(list(section_scores.values())))
    return section_scores, overall_score