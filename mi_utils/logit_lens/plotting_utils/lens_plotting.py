import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
from glob import glob
import os

# -------------------------------
# Paths
# -------------------------------
LI_PATH = "logs/logit_lens_logs/logit_lens_analysis/self_att/LI"
HF_PATH = "logs/logit_lens_logs/logit_lens_analysis/self_att/HF"

# -------------------------------
# Human-readable labels
# -------------------------------
enum_label = { 
    "merged_llama3b_fp_self_att.pt": "Llama-3.2-3B-Instruct-fp",
    "merged_llama3b_8bit_self_att.pt": "Llama-3.2-3B-Instruct-8-bit",
    "merged_llama3b_4bit_self_att.pt": "Llama-3.2-3B-Instruct-4-bit",
    "merged_llama3b_4bitdouble_self_att.pt": "Llama-3.2-3B-Instruct-4-bit-double",
    "merged_llama8b_fp_self_att.pt": "Meta-Llama-3-8B-Instruct-fp",
    "merged_llama8b_8bit_self_att.pt": "Meta-Llama-3-8B-Instruct-8-bit",
    "merged_llama8b_4bit_self_att.pt": "Meta-Llama-3-8B-Instruct-4-bit",
    "merged_llama8b_4bitdouble_self_att.pt": "Meta-Llama-3-8B-Instruct-4-bit-double",
    "merged_llama8b_hf100b_self_att.pt": "Models/Llama3-8B-1.58-100B-tokens",
    "merged_llama8b_hf10bl_self_att.pt": "Llama3-8B-1.58-Linear-10B-tokens",
    "merged_llama8b_hf10bs_self_att.pt": "Llama3-8B-1.58-Sigmoid-k100-10B-tokens"
}

# -------------------------------
# Load .pt files
# -------------------------------
def load_pt_folder(folder):
    pt_files = glob(os.path.join(folder, "*.pt"))
    dfs = []
    for f in pt_files:
        df = torch.load(f, map_location="cpu", weights_only=False)
        for col in ['top1_mean_prob', 'topk_mean_prob', 'topk_var', 'topk_std']:
            if col in df.columns:
                df[col] = df[col].apply(lambda t: t.cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t))
                df[col] = df[col].apply(lambda arr: arr.flatten() if hasattr(arr, 'flatten') else np.array(arr))
        filename = os.path.basename(f)
        df['model_id'] = filename
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

df_li = load_pt_folder(LI_PATH)
df_hf = load_pt_folder(HF_PATH)
df_long = pd.concat([df_li, df_hf], ignore_index=True)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def sort_layers(layer_names):
    embed = [l for l in layer_names if "embed_tokens" in l]
    def get_index(layer):
        if "self_attn" in layer:
            parts = layer.split(".")
            try:
                return int(parts[-1])
            except ValueError:
                return 0
        return -1
    self_att = sorted([l for l in layer_names if "self_attn" in l], key=get_index)
    return embed + self_att

def partition_layers(total_layers):
    if total_layers == 29:
        return {"Early": list(range(0,7)),
                "Mid": list(range(7,21)),
                "Late": list(range(21,28)),
                "Last":[28]}
    elif total_layers == 33:
        return {"Early": list(range(0,8)),
                "Mid": list(range(8,24)),
                "Late": list(range(24,32)),
                "Last":[32]}
    else:
        raise ValueError(f"Unsupported number of layers: {total_layers}")

def aggregate_partition(df, layers, metric_col='topk_mean_prob', std_col='topk_std', min_sigma=1e-6):
    """
    Aggregate all layers in a partition into a single mean and std
    """
    vals = []
    stds = []
    for l in layers:
        df_layer = df[df['layer_name']==l]
        if df_layer.empty:
            continue
        v_list = [np.atleast_1d(x).flatten() for x in df_layer[metric_col] if x is not None and len(np.atleast_1d(x))>0]
        s_list = [np.atleast_1d(x).flatten() for x in df_layer[std_col] if x is not None and len(np.atleast_1d(x))>0]
        if v_list and s_list:
            vals.extend(np.concatenate(v_list))
            stds.extend(np.concatenate(s_list))
    if not vals or not stds:
        return None, None
    return np.mean(vals), max(np.mean(stds), min_sigma)

    
def ridgeplot_logitlens(df_long, groups, enum_label,
                        layer_col='layer_name', metric_col='topk_mean_prob', std_col='topk_std',
                        min_sigma=1e-6):
    sns.set(style="whitegrid")

    for base_model_id, other_models in groups:
        all_models = [base_model_id] + other_models

        # Sorted layers
        base_layers = df_long[df_long['model_id']==base_model_id][layer_col].unique()
        sorted_layers = sort_layers(base_layers)
        layer_to_idx = {l:i for i,l in enumerate(sorted_layers)}
        total_layers = len(sorted_layers)
        partitions = partition_layers(total_layers)

        # Colors
        model_labels = [enum_label.get(mid, mid) for mid in all_models]
        palette = sns.color_palette("tab20", n_colors=len(model_labels))
        color_map = {lab: palette[i] for i, lab in enumerate(model_labels)}
        color_map[enum_label.get(base_model_id, base_model_id)] = (0,0,0)  # base model black

        n_partitions = len(partitions)
        fig_height = max(5, 3*n_partitions)
        fig, axes = plt.subplots(n_partitions, 1, figsize=(20, fig_height), sharex=False)
        if n_partitions == 1:
            axes = [axes]

        for ax, (part_name, idxs) in zip(axes, partitions.items()):
            part_layers = [l for l, idx in layer_to_idx.items() if idx in idxs]

            # Aggregate stats per partition per model
            partition_stats = {}
            for mid in all_models:
                df_model = df_long[df_long['model_id']==mid]
                vals = []
                for l in part_layers:
                    df_layer = df_model[df_model[layer_col]==l]
                    vals_layer = np.concatenate([np.atleast_1d(x).flatten() for x in df_layer[metric_col]])
                    vals.extend(vals_layer)
                partition_stats[mid] = (np.mean(vals), max(np.std(vals), min_sigma))

            # X-axis range
            all_x = []
            for mu, sigma in partition_stats.values():
                all_x.extend([mu - 4*sigma, mu + 4*sigma])
            x_min, x_max = min(all_x), max(all_x)
            x_vals = np.linspace(x_min, x_max, 500)

            # Base model for scaling
            mu_base, sigma_base = partition_stats[base_model_id]
            p_base = norm.pdf(x_vals, mu_base, sigma_base)
            max_pdf_base = max(p_base)

            # Plot curves
            for mid in all_models:
                mu, sigma = partition_stats[mid]
                pdf = norm.pdf(x_vals, mu, sigma)
                pdf_scaled = pdf * (max_pdf_base / max(pdf))
                label = enum_label.get(mid, mid)
                ax.plot(x_vals, pdf_scaled, color=color_map[label],
                        lw=2.5 if mid==base_model_id else 1.8)
                ax.fill_between(x_vals, 0, pdf_scaled, color=color_map[label],
                                alpha=0.15 if mid==base_model_id else 0.2)

            # --- FIXED KL/TVD ANNOTATIONS ---
            # --- KL/TVD ANNOTATIONS FAR RIGHT, OUTSIDE CURVES ---
            x_marker = x_max + (x_max - x_min) * 0.35  # marker just outside the curves
            x_text   = x_max + (x_max - x_min) * 0.45  # text further to the right
            y_start = max_pdf_base - 0.01 * max_pdf_base  # nudge down slightly
            y_spacing = max_pdf_base * 0.18  # vertical spacing between labels

            for i, mid in enumerate(other_models):
                mu, sigma = partition_stats[mid]
                pdf = norm.pdf(x_vals, mu, sigma)
                pdf_scaled = pdf * (max_pdf_base / max(pdf))
                kl = np.trapezoid(p_base*np.log((p_base + 1e-12)/(pdf_scaled + 1e-12)), x_vals)
                tvd = 0.5*np.trapezoid(np.abs(p_base - pdf_scaled), x_vals)
                y_label = y_start - i * y_spacing
                # color marker only
                ax.plot(x_marker, y_label, 'o', color=color_map[enum_label.get(mid, mid)], markersize=12)
                # KL/TVD annotation in black, further to the right
                ax.text(x_text, y_label,
                        f"KL={kl:.3f}, TVD={tvd:.3f}",
                        color='black',
                        fontsize=18, va='center', ha='left')


            # --- Partition labels further left ---
            ax.text(x_min - (x_max-x_min)*0.1, max_pdf_base/2, part_name,
                    ha='right', va='center', fontsize=18, fontweight='bold')

            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_linewidth(1.2)
            ax.set_xlim(x_min - (x_max-x_min)*0.05, x_max + (x_max-x_min)*0.6)
            ax.set_ylim(0, max_pdf_base*1.05)

        # Legend
        # --- LEGEND ---
        handles = [plt.Line2D([0],[0], color=color_map[l], lw=2.5 if l==enum_label.get(base_model_id, base_model_id) else 2)
                for l in model_labels]

        # Determine number of columns for 3 rows
        ncols = int(np.ceil(len(model_labels)/3))

        # Example: placing legend slightly lower
        fig.legend(handles, model_labels, loc='upper center', ncol=ncols,
                frameon=False, fontsize=20, columnspacing=1.2,
                bbox_to_anchor=(0.5, 1.02))  # y=1.02 is slightly above default


        #plt.xlabel("Probability", fontsize=18)
        plt.subplots_adjust(hspace=0.7, left=0.08, right=0.85)
        #plt.savefig("figs/ridgeplot_llama8b")
        plt.show()


# -------------------------------
# Model labels
# -------------------------------
enum_label = { 
    "merged_llama8b_fp_self_att.pt": "Meta-Llama-3-8B-Instruct-fp",
    "merged_llama8b_8bit_self_att.pt": "Meta-Llama-3-8B-Instruct-8-bit",
    "merged_llama8b_4bit_self_att.pt": "Meta-Llama-3-8B-Instruct-4-bit",
    "merged_llama8b_4bitdouble_self_att.pt": "Meta-Llama-3-8B-Instruct-4-bit-double",
    "merged_llama8b_hf100b_self_att.pt": "Llama3-8B-1.58-100B-tokens",
    "merged_llama8b_hf10bl_self_att.pt": "Llama3-8B-1.58-Linear-10B-tokens",
    "merged_llama8b_hf10bs_self_att.pt": "Llama3-8B-1.58-Sigmoid-k100-10B-tokens"
}

# -------------------------------
# Groups for comparison
# -------------------------------
groups = [
    ("merged_llama8b_fp_self_att.pt", [
        "merged_llama8b_8bit_self_att.pt",
        "merged_llama8b_4bit_self_att.pt",
        "merged_llama8b_4bitdouble_self_att.pt",
        "merged_llama8b_hf100b_self_att.pt",
        "merged_llama8b_hf10bl_self_att.pt",
        "merged_llama8b_hf10bs_self_att.pt"
    ])
]


ridgeplot_logitlens(df_long, groups, enum_label, layer_col='layer_name', metric_col='topk_mean_prob', std_col='topk_std')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


# -----------------------------
# Helpers
# -----------------------------
def sort_layers(layer_names):
    embed = [l for l in layer_names if "embed_tokens" in l]
    def get_index(layer):
        if "self_attn" in layer:
            parts = layer.split(".")
            try:
                return int(parts[-1])
            except ValueError:
                return 0
        return -1
    self_att = sorted([l for l in layer_names if "self_attn" in l], key=get_index)
    return embed + self_att

def partition_layers(total_layers):
    if total_layers == 29:
        return {"Early": list(range(0,7)),
                "Mid": list(range(7,21)),
                "Late": list(range(21,28)),
                "Last":[28]}
    elif total_layers == 33:
        return {"Early": list(range(0,8)),
                "Mid": list(range(8,24)),
                "Late": list(range(24,32)),
                "Last":[32]}
    else:
        raise ValueError(f"Unsupported number of layers: {total_layers}")

def compute_divergences(mu_base, sigma_base, mu_other, sigma_other, x_vals):
    """Compute KL and TVD from true (non-scaled) PDFs."""
    p_base = norm.pdf(x_vals, mu_base, sigma_base)
    p_other = norm.pdf(x_vals, mu_other, sigma_other)
    p_base /= np.trapezoid(p_base, x_vals)   # normalize to integrate to 1
    p_other /= np.trapezoid(p_other, x_vals)

    kl = np.trapezoid(p_base * np.log((p_base + 1e-12) / (p_other + 1e-12)), x_vals)
    tvd = 0.5 * np.trapezoid(np.abs(p_base - p_other), x_vals)
    return kl, tvd

# -----------------------------
# Ridge plot (no scaling, true PDFs)
# -----------------------------
def ridge_logitlens_truepdf(df_long, groups, enum_label,
                            layer_col='layer_name', metric_col='topk_mean_prob',
                            min_sigma=1e-6):
    for base_model_id, other_models in groups:
        all_models = [base_model_id] + other_models

        # --- Sort layers ---
        base_layers = df_long[df_long['model_id']==base_model_id][layer_col].unique()
        sorted_layers = sort_layers(base_layers)
        layer_to_idx = {l:i for i,l in enumerate(sorted_layers)}
        total_layers = len(sorted_layers)
        partitions = partition_layers(total_layers)

        # --- Colors ---
        model_labels = [enum_label.get(mid, mid) for mid in all_models]
        palette = plt.cm.tab20(np.linspace(0,1,len(model_labels)))
        color_map = {lab: palette[i] for i, lab in enumerate(model_labels)}
        color_map[enum_label.get(base_model_id, base_model_id)] = (0,0,0)  # base model black

        # ---------------- Plot ----------------
        n_partitions = len(partitions)
        fig, axes = plt.subplots(n_partitions, 1, figsize=(20, max(5, 3*n_partitions)), sharex=False)
        if n_partitions == 1:
            axes = [axes]

        for ax, (part_name, idxs) in zip(axes, partitions.items()):
            part_layers = [l for l, idx in layer_to_idx.items() if idx in idxs]

            # Partition stats (aggregate all values across these layers)
            partition_stats = {}
            for mid in all_models:
                df_model = df_long[df_long['model_id']==mid]
                vals = []
                for l in part_layers:
                    df_layer = df_model[df_model[layer_col]==l]
                    vals_layer = np.concatenate([np.atleast_1d(x).flatten() for x in df_layer[metric_col]])
                    vals.extend(vals_layer)
                partition_stats[mid] = (np.mean(vals), max(np.std(vals), min_sigma))

            # X range across models (NO clipping to [0,1] → tails are visible)
            all_x = []
            for mu, sigma in partition_stats.values():
                all_x.extend([mu - 4*sigma, mu + 4*sigma])
            x_min, x_max = min(all_x), max(all_x)
            x_vals = np.linspace(x_min, x_max, 1000)

            # Plot PDFs (true, no scaling)
            max_pdf = 0
            for mid in all_models:
                mu, sigma = partition_stats[mid]
                pdf = norm.pdf(x_vals, mu, sigma)
                label = enum_label.get(mid, mid)
                ax.plot(x_vals, pdf, color=color_map[label],
                        lw=2.5 if mid==base_model_id else 1.8)
                ax.fill_between(x_vals, 0, pdf, color=color_map[label],
                                alpha=0.15 if mid==base_model_id else 0.2)
                max_pdf = max(max_pdf, max(pdf))

            # KL/TVD annotations
            # KL/TVD annotations
            mu_base, sigma_base = partition_stats[base_model_id]
            x_marker = x_max + (x_max - x_min) * 0.1
            x_text   = x_max + (x_max - x_min) * 0.15
            y_start = max_pdf * 0.97
            y_spacing = max_pdf * 0.18


            for i, mid in enumerate(other_models):
                mu, sigma = partition_stats[mid]
                kl, tvd = compute_divergences(mu_base, sigma_base, mu, sigma, x_vals)
                y_label = y_start - i * y_spacing
                ax.plot(x_marker, y_label, 'o', color=color_map[enum_label.get(mid, mid)], markersize=12)
                ax.text(x_text, y_label,
                        f"KL={kl:.3f}, TVD={tvd:.3f}",
                        color='black', fontsize=16, va='center', ha='left')

            # Partition label
            ax.text(x_min - (x_max-x_min)*0.2, max_pdf/2, part_name,
                    ha='right', va='center', fontsize=18, fontweight='bold')

            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_linewidth(1.2)
            ax.set_xlim(x_min - (x_max-x_min)*0.1, x_max + (x_max-x_min)*0.2)
            ax.set_ylim(0, max_pdf*1.1)

        # Legend
        handles = [plt.Line2D([0],[0], color=color_map[l],
                              lw=2.5 if l==enum_label.get(base_model_id, base_model_id) else 2)
                   for l in model_labels]
        ncols = int(np.ceil(len(model_labels)/3))
        fig.legend(handles, model_labels,
                   loc='upper center', ncol=ncols,
                   frameon=False, fontsize=18, columnspacing=1.2,
                   bbox_to_anchor=(0.5, 1.02))

        # Adjust subplot spacing
        plt.subplots_adjust(hspace=0.9, top=0.9, bottom=0.08)

        plt.savefig("figs/ridgeplot_noscale_llama8b")
        plt.show()

# -------------------------------
# Model labels
# -------------------------------
enum_label = { 
    "merged_llama8b_fp_self_att.pt": "Meta-Llama-3-8B-Instruct-fp",
    "merged_llama8b_8bit_self_att.pt": "Meta-Llama-3-8B-Instruct-8-bit",
    "merged_llama8b_4bit_self_att.pt": "Meta-Llama-3-8B-Instruct-4-bit",
    "merged_llama8b_4bitdouble_self_att.pt": "Meta-Llama-3-8B-Instruct-4-bit-double",
    "merged_llama8b_hf100b_self_att.pt": "Llama3-8B-1.58-100B-tokens",
    "merged_llama8b_hf10bl_self_att.pt": "Llama3-8B-1.58-Linear-10B-tokens",
    "merged_llama8b_hf10bs_self_att.pt": "Llama3-8B-1.58-Sigmoid-k100-10B-tokens"
}

# -------------------------------
# Groups for comparison
# -------------------------------
groups = [
    ("merged_llama8b_fp_self_att.pt", [
        "merged_llama8b_8bit_self_att.pt",
        "merged_llama8b_4bit_self_att.pt",
        "merged_llama8b_4bitdouble_self_att.pt",
        "merged_llama8b_hf100b_self_att.pt",
        "merged_llama8b_hf10bl_self_att.pt",
        "merged_llama8b_hf10bs_self_att.pt"
    ])
]


ridge_logitlens_truepdf(df_long, groups, enum_label, layer_col='layer_name', metric_col='topk_mean_prob')