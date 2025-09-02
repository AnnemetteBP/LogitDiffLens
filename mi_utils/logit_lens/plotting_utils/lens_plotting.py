from typing import Optional
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# ----------------------
# Plotting / comparison utilities
# ----------------------
def plot_layer_metric_two_dfs(
    dfA:pd.DataFrame,
    dfB:pd.DataFrame,
    metric:str,
    title:str="",
    save_path:Optional[str]=None
) -> None:
    def per_layer_mean(df, metric):
        layers = sorted(df['layer_index'].unique())
        vals = []
        for l in layers:
            col = df[df['layer_index']==l][metric]
            flat = []
            for v in col:
                if isinstance(v, list):
                    flat.extend([float(x) for x in v if x is not None and (not (isinstance(x,float) and math.isnan(x)))])
                elif np.isscalar(v):
                    flat.append(float(v))
            vals.append(np.mean(flat) if len(flat)>0 else np.nan)
        return np.array(layers), np.array(vals)
    la, va = per_layer_mean(dfA, metric)
    lb, vb = per_layer_mean(dfB, metric)
    common = sorted(set(la.tolist()).intersection(lb.tolist()))
    a_vals = [va[list(la).index(l)] for l in common]
    b_vals = [vb[list(lb).index(l)] for l in common]
    plt.figure(figsize=(10,4))
    plt.plot(common, a_vals, label=f"{dfA['model_name'].iloc[0]}", marker='o')
    plt.plot(common, b_vals, label=f"{dfB['model_name'].iloc[0]}", marker='o')
    plt.xlabel("Layer index")
    plt.ylabel(metric)
    plt.title(title or f"{metric} by layer")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_layer_deviation(
    dfA:pd.DataFrame,
    dfB:pd.DataFrame,
    metric:str="nwd",
    title:str="",
    save_path:Optional[str]=None
):
    la = sorted(dfA['layer_index'].unique())
    vals = []
    for l in la:
        a = dfA[dfA['layer_index']==l][metric]
        b = dfB[dfB['layer_index']==l][metric]
        def get_first_mean(col):
            for v in col:
                if isinstance(v, list):
                    if len(v)>0:
                        return float(np.nanmean(v))
                elif np.isscalar(v):
                    return float(v)
            return np.nan
        a_val = get_first_mean(a)
        b_val = get_first_mean(b)
        vals.append(np.abs(a_val - b_val) if (not np.isnan(a_val) and not np.isnan(b_val)) else np.nan)
    plt.figure(figsize=(10,4))
    plt.plot(la, vals, marker='o')
    plt.xlabel("Layer index")
    plt.ylabel(f"|Δ {metric}|")
    plt.title(title or f"Layerwise absolute difference in {metric}")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def qq_plot_probs(
    p:np.ndarray,
    q:np.ndarray,
    title:str="QQ plot",
    save_path:Optional[str]=None
 ) -> None:
    # p,q: 1D prob vectors (vocab) or flattened per-token means
    import scipy.stats as ss
    p_sorted = np.sort(p)
    q_sorted = np.sort(q)
    minlen = min(len(p_sorted), len(q_sorted))
    plt.figure(figsize=(5,5))
    plt.scatter(p_sorted[:minlen], q_sorted[:minlen], s=4)
    plt.plot([0,1],[0,1], color='k', linestyle='--')
    plt.xlabel("FP model prob quantile")
    plt.ylabel("Quantized model prob quantile")
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()