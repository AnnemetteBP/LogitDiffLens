import re
import torch
import pandas as pd

# --- Filter only main layers ---
def filter_main_layers(df):
    """Keep only rows where layer_name is layers.0, layers.1, ..., layers.N"""
    pattern = re.compile(r'^layers\.\d+$')
    return df[df['layer_name'].apply(lambda x: bool(pattern.match(x)))].copy()

# --- Prepare tensors ---
def prepare_layer_tensors(df):
    """
    Stack layer logits [num_layers, seq_len, vocab_size].
    Returns layers_tensor, input_ids, target_ids.
    """
    layer_logits_list = []
    for _, row in df.iterrows():
        layer_tensor = torch.stack([
            torch.tensor(x, dtype=torch.float) if not isinstance(x, torch.Tensor) else x
            for x in row['logits']
        ])
        layer_logits_list.append(layer_tensor)

    layers_tensor = torch.stack(layer_logits_list)  # [L, S, V]
    input_ids = torch.tensor(df['input_ids'].iloc[0], dtype=torch.long)
    target_ids = torch.tensor(df['target_ids'].iloc[0], dtype=torch.long)
    return layers_tensor, input_ids, target_ids

# --- Compute carry-over safe metrics ---
def compute_carry_over_safe_scalar(layers_tensor, input_ids, target_ids, topk=5):
    L, S, V = layers_tensor.shape
    top1_preds = layers_tensor.argmax(dim=-1)  # [L, S]

    # --- Top-1 carry-over safe accuracy ---
    acc_top1 = torch.zeros(S)
    for s in range(S):
        prev_preds = set()
        carry_token = int(input_ids[s])
        for l in range(L):
            pred = int(top1_preds[l, s])
            if pred == int(target_ids[s]) and pred not in prev_preds and pred != carry_token:
                acc_top1[s] = 1.0
                break
            prev_preds.add(pred)

    # --- Top-k carry-over safe accuracy ---
    acc_topk = torch.zeros(S)
    for s in range(S):
        prev_preds = set()
        carry_token = int(input_ids[s])
        for l in range(L):
            logit_vec = layers_tensor[l, s]
            topk_preds = set(logit_vec.topk(min(topk, V)).indices.tolist())
            if int(target_ids[s]) in topk_preds and int(target_ids[s]) not in prev_preds and int(target_ids[s]) != carry_token:
                acc_topk[s] = 1.0
                break
            prev_preds.update(topk_preds)

    # --- Top-1 carry-over safe stability ---
    stab_top1 = torch.zeros(S)
    for s in range(S):
        carry_token = int(input_ids[s])
        stable_count = 0
        for l in range(1, L):
            if top1_preds[l, s] == top1_preds[l-1, s] and top1_preds[l, s] != carry_token:
                stable_count += 1
        stab_top1[s] = stable_count / (L-1) if L > 1 else 0.0

    # --- Top-k carry-over safe stability ---
    stab_topk = torch.zeros(S)
    for s in range(S):
        carry_token = int(input_ids[s])
        stable_count = 0
        for l in range(1, L):
            prev_topk = set(layers_tensor[l-1, s].topk(min(topk, V)).indices.tolist()) - {carry_token}
            curr_topk = set(layers_tensor[l, s].topk(min(topk, V)).indices.tolist()) - {carry_token}
            if prev_topk == curr_topk:
                stable_count += 1
        stab_topk[s] = stable_count / (L-1) if L > 1 else 0.0

    return {
        'acc_top1_scalar': float(acc_top1.mean()),
        'acc_topk_scalar': float(acc_topk.mean()),
        'stab_top1_scalar': float(stab_top1.mean()),
        'stab_topk_scalar': float(stab_topk.mean())
    }
