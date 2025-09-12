from typing import List, Optional
import os
import math
import torch
import numpy as np
import pandas as pd
import glob
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
from scipy.special import rel_entr
import re
from ..util.logit_lens_utils.logit_lens_wrapper import LogitLensWrapper
from ..util.logit_lens_utils.model_device_handling import get_base_model, get_embedding_device
from .metric_utils.logit_lens_helpers import compute_ece

# ----------------------------
# Reusable Inputs
# ----------------------------
EPS = 1e-12
TOPK = 5
MAX_LEN = 16


def safe_tensor(
    t: torch.Tensor,
    eps: float = EPS,
    min_val: float = -100,
    max_val: float = 100,
    for_log: bool = False,
    target_dtype = torch.float32
) -> torch.Tensor:
    """ Safe tensor handling for float32 logits or probabilities.
    Parameters:
        t: Input tensor (float32)
        eps: Small positive value to prevent log(0)
        min_val, max_val: clamp range for logits
        for_log: if True, ensures values >= eps (for log computations)
    """
    dtype = target_dtype
    if dtype is None:
        if t.dtype in [torch.float16, torch.float32]:
            dtype = t.dtype
        else:
            dtype = torch.float32

    # Replace NaN / Inf
    t = torch.nan_to_num(t, nan=-1e9, posinf=1e9, neginf=-1e9)
    # Clamp to reasonable range
    t = t.clamp(min=min_val, max=max_val)
    # Ensure positive minimum if used for log
    if for_log:
        t = t.clamp_min(eps)
    return t

def safe_entropy(probs: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Compute entropy safely, using eps to avoid log(0)."""
    # Cast to float32 for safe log computation
    probs_f32 = probs.to(torch.float32)

    # Clamp small probabilities
    probs_f32 = probs_f32.clamp(min=eps, max=1.0)

    log_probs = torch.log(probs_f32)
    ent = -(probs_f32 * log_probs).sum(dim=-1)

    return ent

def safe_cast_logits(
    logits: torch.Tensor,
    min_val: float = -100,
    max_val: float = 100,
    target_dtype=torch.float32
) -> torch.Tensor:
    """NaN/Inf protection and clamping for logits."""
    logits = logits.to(target_dtype)
    logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    logits = logits.clamp(min=min_val, max=max_val)
    return logits

def safe_softmax(
    logits: torch.Tensor,
    dim=-1,
    eps: float = EPS
) -> torch.Tensor:
    """Numerically stable softmax with eps scaling."""
    logits = safe_cast_logits(logits) 
    max_logits = logits.max(dim=dim, keepdim=True).values
    exp_logits = torch.exp(logits - max_logits)
    probs = exp_logits / (exp_logits.sum(dim=dim, keepdim=True) + eps)
    return probs.clamp(min=eps, max=1.0)


# Matches subword markers and leading non-alphanumeric characters
_SUBWORD_PREFIX_RE = re.compile(r"^[▁##\W]+")

def _script_from_token(tok: str) -> str:
    """
    Detect the script/language of a token robustly.
    Handles subword tokens and ignores leading punctuation.
    Returns: 'Latin', 'Cyrillic', 'Devanagari', 'Han', 'Kana', 
             'Hangul', 'Arabic', 'Greek', 'Digit', 'Symbol'.
    """
    if not tok:
        return "Symbol"  # empty token -> symbol

    # Remove subword markers and leading punctuation
    tok_clean = _SUBWORD_PREFIX_RE.sub("", tok)

    for ch in tok_clean:
        code = ord(ch)
        if ch.isdigit():
            return "Digit"
        if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0x20000 <= code <= 0x2A6DF:
            return "Han"
        if 0x3040 <= code <= 0x30FF:
            return "Kana"
        if 0xAC00 <= code <= 0xD7AF:
            return "Hangul"
        if 0x0400 <= code <= 0x04FF or 0x0500 <= code <= 0x052F:
            return "Cyrillic"
        if 0x0900 <= code <= 0x097F:
            return "Devanagari"
        if 0x0600 <= code <= 0x06FF:
            return "Arabic"
        if 0x0370 <= code <= 0x03FF:
            return "Greek"
        if (0x0041 <= code <= 0x007A) or (0x00C0 <= code <= 0x024F):
            return "Latin"

    # If we reach here, token is non-empty but contains no letters/digits
    return "Symbol"


# ----------------------------
# Logit Lens Analysis
# ----------------------------
def _run_logit_lens(
    wrapper: LogitLensWrapper,
    prompts: list[str],
    correct_cloze: Optional[List[str]] = None,
    analysis_type: str = "self_att",
    add_special_tokens: bool = False,  # default OFF for probing
    add_bos: bool = False,             # handle manually if needed
    add_eos: bool = False, 
    topk: int = 5,
    eps: float = 1e-12,
    skip_input_layer: bool = False,
    include_final_norm: bool = False,
    save_layer_probs: bool = False,
    proj_precision: Optional[str] = None,
    max_len: Optional[int] = 16,
    pad_to_max_length: bool = False,
) -> pd.DataFrame:
    assert analysis_type in ("self_att", "mlp")
    torch.set_grad_enabled(False)
    wrapper.model.eval()
    device = next(wrapper.model.parameters()).device
    rows = []

    def _to_analysis_dtype(t: torch.Tensor) -> torch.dtype:
        if proj_precision is None:
            return t.dtype
        return torch.float16 if proj_precision.lower() == "fp16" else torch.float32

    # Forward pass
    outputs, layer_dict, layer_names = wrapper.forward(
        prompts, project_to_logits=True, return_hidden=False,
        add_eos=add_eos, keep_on_device=True, max_len=max_len
    )
    layer_logits_list, layer_names = wrapper.stack_layer_logits(layer_dict, keep_on_device=True, filter_layers=False)

    # Filter relevant layers
    valid_layer_names, valid_layer_logits = [], []
    for lname, logits in zip(layer_names, layer_logits_list):
        lname_lower = lname.lower()
        if not skip_input_layer and "embed_tokens" in lname_lower:
            pass
        elif skip_input_layer and any(k in lname_lower for k in ["input", "wte", "wpe"]):
            continue
        if not include_final_norm and any(k in lname_lower for k in ["ln", "norm", "layernorm", "rmsnorm"]):
            continue
        if analysis_type == "self_att" and not (("self_att" in lname_lower) or ("embed_tokens" in lname_lower)):
            continue
        if analysis_type == "mlp" and not (("mlp" in lname_lower) or ("embed_tokens" in lname_lower)):
            continue
        valid_layer_names.append(lname)
        valid_layer_logits.append(logits)

    if not valid_layer_logits:
        return pd.DataFrame()

    batch_size, seq_len = valid_layer_logits[0].shape[:2]
    tokenizer = wrapper.tokenizer
    vocab_size = getattr(tokenizer, "vocab_size", None)

    # Precompute logits/probs on CPU
    det_logits_cpu, det_probs_cpu = [], []
    for logits in valid_layer_logits:
        adtype = _to_analysis_dtype(logits)
        l_cpu = safe_cast_logits(logits.detach().to("cpu"), target_dtype=adtype)
        probs = safe_softmax(l_cpu, dim=-1, eps=eps).to(adtype)
        det_logits_cpu.append(l_cpu)
        det_probs_cpu.append(probs)

    # Targets
    if hasattr(outputs, "logits") and outputs.logits is not None:
        full_pred = outputs.logits.argmax(dim=-1)
    else:
        full_pred = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)

    # Main loop
    for idx in range(batch_size):
        intended_cloze = correct_cloze[idx] if (correct_cloze is not None and idx < len(correct_cloze)) else None
        tgt_ids = full_pred[idx, 1:].cpu()
        max_valid_len = tgt_ids.numel()
        input_ids_seq_trim_cpu = (
            outputs.input_ids[idx, :max_valid_len].cpu()
            if hasattr(outputs, "input_ids") and outputs.input_ids is not None
            else torch.arange(max_valid_len, dtype=torch.long)
        )

        for l, lname in enumerate(valid_layer_names):
            logits_cur = det_logits_cpu[l][idx, :max_valid_len]
            probs_cur = det_probs_cpu[l][idx, :max_valid_len]
            preds_seq_tensor = probs_cur.argmax(dim=-1)
            k_eff = min(topk, probs_cur.shape[-1])
            topk_idx_tensor = torch.topk(probs_cur, k=k_eff, dim=-1).indices
            valid_len = preds_seq_tensor.shape[0]
            preds_seq = preds_seq_tensor.cpu().numpy().astype(int)
            tgt_np = tgt_ids[:valid_len].numpy().astype(int)
            topk_preds_seq = topk_idx_tensor.cpu().numpy().astype(int)

            # convert token ids -> token strings
            top1_token_strs = tokenizer.convert_ids_to_tokens(preds_seq.tolist(), skip_special_tokens=False)
            topk_token_ids_flat = topk_preds_seq.reshape(-1).tolist()
            topk_token_strs_flat = tokenizer.convert_ids_to_tokens(topk_token_ids_flat, skip_special_tokens=False)
            topk_token_strs = [topk_token_strs_flat[i * k_eff: (i + 1) * k_eff] for i in range(valid_len)]

            # detect scripts/languages
            # convert token ids -> token strings (decoded)
            top1_token_strs = [tokenizer.decode([tid]).lstrip("▁") for tid in preds_seq]
            topk_token_strs = [
                [tokenizer.decode([tid]).lstrip("▁") for tid in row] 
                for row in topk_preds_seq
            ]

            # detect scripts/languages
            top1_scripts = [_script_from_token(t) for t in top1_token_strs]
            topk_scripts = [[_script_from_token(t) for t in row] for row in topk_token_strs]


            # Build row
            row = {
                "prompt_id": idx,
                "prompt_text": prompts[idx],
                "intended_cloze": intended_cloze,
                "layer_index": l,
                "layer_name": lname,
                "seq_len": int(valid_len),
                "vocab_size": vocab_size,
                "logits": logits_cur.cpu(),
                "input_ids": input_ids_seq_trim_cpu.cpu(),
                "target_ids": tgt_ids[:valid_len].cpu(),
                "top1_token_str_seq": top1_token_strs,
                "topk_token_str_seq": topk_token_strs,
                "top1_script_seq": top1_scripts,
                "topk_script_seq": topk_scripts,
            }

            # entropy & probs
            p_safe = safe_tensor(probs_cur, eps=eps, target_dtype=probs_cur.dtype, for_log=True)
            row["entropy_seq"] = safe_entropy(p_safe)
            row["normalized_entropy_seq"] = row["entropy_seq"] / math.log(vocab_size) if vocab_size and vocab_size > 1 else row["entropy_seq"]
            row["prob_mean_seq"] = probs_cur.mean(dim=-1).cpu()
            if save_layer_probs:
                row["probs_seq"] = probs_cur.cpu()

            # self_att metrics
            if analysis_type == "self_att":
                row["preds_seq"] = preds_seq
                row["topk_pred_tokens_seq"] = topk_preds_seq
                row["top1_mean_prob"] = probs_cur.gather(-1, preds_seq_tensor.unsqueeze(-1)).squeeze(-1).mean().item()
                row["topk_mean_prob"] = torch.topk(probs_cur, k=k_eff, dim=-1).values.mean().item()
                row["top1_var"] = probs_cur.gather(-1, preds_seq_tensor.unsqueeze(-1)).squeeze(-1).var().item()
                row["top1_std"] = probs_cur.gather(-1, preds_seq_tensor.unsqueeze(-1)).squeeze(-1).std().item()
                row["topk_var"] = torch.topk(probs_cur, k=k_eff, dim=-1).values.mean(dim=-1).var().item()
                row["topk_std"] = torch.topk(probs_cur, k=k_eff, dim=-1).values.mean(dim=-1).std().item()
                try:
                    row["ece"] = compute_ece(p_safe.cpu().numpy(), tgt_np) if valid_len > 0 else None
                except Exception:
                    row["ece"] = None
                try:
                    row["repetition_ratio"] = float(np.mean(preds_seq[1:] == preds_seq[:-1])) if valid_len > 1 else np.nan
                except Exception:
                    row["repetition_ratio"] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def run_logit_lens(
    wrapper: LogitLensWrapper,
    prompts: list[str],
    correct_cloze: Optional[List[str]] = None,
    analysis_type: str = "self_att",
    model_name: str = "model",
    dataset_name: str = "dataset",
    save_dir: str = "logs/logit_lens_logs/logit_lens_analysis",
    topk: int = 5,
    eps: float = 1e-12,
    add_special_tokens: bool = False,  
    add_bos: bool = False,             
    add_eos: bool = False, 
    skip_input_layer: bool = False,
    include_final_norm: bool = False,
    save_layer_probs: bool = False,
    proj_precision: Optional[str] = None,
    max_len: Optional[int] = 16,
    pad_to_max_length: bool = False,
):
    df_rows = _run_logit_lens(
        wrapper=wrapper,
        prompts=prompts,
        correct_cloze=correct_cloze,
        analysis_type=analysis_type,
        add_special_tokens=add_special_tokens,
        add_bos=add_bos,            
        add_eos=add_eos,
        topk=topk,
        eps=eps,
        skip_input_layer=skip_input_layer,
        include_final_norm=include_final_norm,
        save_layer_probs=save_layer_probs,
        proj_precision=proj_precision,
        max_len=max_len,
        pad_to_max_length=pad_to_max_length
    )

    if df_rows.empty:
        print("No valid layers found. Nothing saved.")
        return df_rows

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset_name}_{model_name}_{analysis_type}.pt")
    torch.save(df_rows, save_path)
    print(f"Saved analysis to {save_path}")
    #return df_rows
