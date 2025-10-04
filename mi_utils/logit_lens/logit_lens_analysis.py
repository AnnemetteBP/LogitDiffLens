from typing import List, Optional, Dict, Any
import os
import glob
import math
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import Counter
from scipy.special import rel_entr
import re
from ..util.logit_lens_utils.logit_lens_wrapper import LogitLensWrapper
from ..util.logit_lens_utils.model_device_handling import get_base_model, get_embedding_device

# ----------------------------
# Reusable Inputs
# ----------------------------
EPS = 1e-12
TOPK = 5
MAX_LEN = 16


"""

With these you’ll be able to:

Plot script diversity curves across layers (entropy vs depth).

Show ECE per script per layer → e.g. BitNet might be well-calibrated on Latin tokens but terrible on Chinese.

That’ll visually prove how quantization changes both the shape of the probability space and the confidence reliability.

Do you want me to also sketch a plotting helper (Matplotlib/Seaborn) for these layer-by-layer comparisons so you can instantly compare BitNet vs LLaMA?

"""
# --- Script Diversity ---
def script_diversity_per_layer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute script diversity (entropy over scripts) per layer.
    Expects df rows from _run_logit_lens with `topk_script_seq`.
    """
    rows = []
    for lname, g in df.groupby("layer_name"):
        scripts = []
        for seq in g["topk_script_seq"]:
            for row in seq:  # per token
                scripts.extend(row)
        counter = Counter(scripts)
        total = sum(counter.values())
        probs = np.array([v / total for v in counter.values()])
        entropy = -(probs * np.log(probs + 1e-12)).sum()
        rows.append({
            "layer_name": lname,
            "script_diversity": len(counter),
            "script_entropy": entropy,
            "script_distribution": counter,
        })
    return pd.DataFrame(rows)

def safe_cast_logits(logits, target_dtype=torch.float32, sanitize=False):
    if sanitize:
        logits = logits.clone()
        logits[torch.isnan(logits)] = -1e9
    return logits.to(target_dtype)

def safe_softmax(logits, dim=-1, eps=1e-12, sanitize=False):
    if sanitize:
        logits = logits.clone()
        logits[torch.isnan(logits)] = -1e9
    max_logits = logits.max(dim=dim, keepdim=True).values
    exps = (logits - max_logits).exp()
    sum_exps = exps.sum(dim=-1, keepdim=True)
    return exps / (sum_exps + eps)

def safe_tensor(tensor, eps=1e-12, target_dtype=torch.float32, for_log=False):
    tensor = tensor.to(target_dtype)
    if for_log:
        tensor = torch.clamp(tensor, min=eps)
    return tensor

def safe_entropy(p_tensor, eps=1e-12):
    return -(p_tensor * torch.log(p_tensor + eps)).sum(dim=-1)

def log_softmax(logits) -> tuple:
    probs = torch.softmax(logits, dim=-1)       # regular probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # numerically stable log-probs
    return probs, log_probs

LANG_TO_SCRIPT = {
    'Français': 'Latin',
    'English': 'Latin',
    'Deutsch': 'Latin',
    'Italiano': 'Latin',
    'Español': 'Latin',
    'Русский': 'Cyrillic',
    '中文': 'Han',
    '한국어': 'Hangul',
    'Ελληνικά': 'Greek',
    'हिंदी': 'Devanagari',
    'العربية': 'Arabic',  # if you ever add Arabic
}

def contains_target_language(response: str, languages: list[str]) -> bool:
    """
    Check if the response contains the expected script of the target language.
    `languages` is like ['English', 'Русский'], so the last one is the target.
    """
    target_lang = languages[-1]
    expected_script = LANG_TO_SCRIPT.get(target_lang)

    if not expected_script:
        raise ValueError(f"Unknown target language: {target_lang}")

    for tok in response.split():
        if _script_from_token(tok) == expected_script:
            return True
    return False

"""response = "кошка"   # LLM output
languages = ['English', 'Русский']
print(contains_target_language(response, languages))  # True"""

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

# --- Detect script for a whole sequence ---
def _detect_scripts_from_input(wrapper, input_ids_seq):
    return [_script_from_token(wrapper.tokenizer.decode([tid])) for tid in input_ids_seq.cpu().numpy()]

# ----------------------------
# Logit Lens Analysis with Cloze Correctness
# ----------------------------
def _run_logit_lens(
    wrapper: LogitLensWrapper,
    prompts: list[str],
    analysis_type: str = "all",
    dataset_name: str = "default",
    add_special_tokens: bool = False,
    mask_bos: bool = True,
    mask_eos: bool = True,
    mask_pad: bool = True,
    skip_input_layer: bool = False,
    include_embed_tokens: bool = True,     # <-- NEW
    include_final_norm: bool = True,
    proj_precision: Optional[str] = None,
    max_len: Optional[int] = 128,
    pad_to_max_length: bool = False,
) -> pd.DataFrame:
    """
    Run the Logit Lens analysis, handling tokenizer masking, normalization variants,
    and optional subblock probing.
    """
    torch.set_grad_enabled(False)
    wrapper.model.eval()
    rows: list[dict] = []

    # --- Precision control ---
    def _to_analysis_dtype(t: torch.Tensor) -> torch.dtype:
        if proj_precision is not None:
            return torch.float16 if proj_precision.lower() == "fp16" else torch.float32
        return t.dtype

    # --- Forward pass through wrapper ---
    outputs, layer_dict, layer_names, input_ids = wrapper.forward(
        prompts,
        project_to_logits=True,
        return_hidden=False,
        add_special_tokens=add_special_tokens,
        keep_on_device=True,
        max_len=max_len,
        pad_to_max_length=pad_to_max_length,
    )

    # --- Stack layer logits ---
    layer_logits_list, layer_names = wrapper.stack_layer_logits(
        layer_dict, keep_on_device=True, filter_layers=False
    )

    # --- Mask special tokens (optional, correct as-is) ---
    if wrapper.tokenizer is not None:
        mask_ids = []
        for tok_name, use_mask in zip(["bos", "eos", "pad"], [mask_bos, mask_eos, mask_pad]):
            tok_id = getattr(wrapper.tokenizer, f"{tok_name}_token_id", None)
            if use_mask and tok_id is not None:
                mask_ids.append(tok_id)

        if mask_ids:
            mask_ids_tensor = torch.tensor(mask_ids, device=layer_logits_list[0].device)
            for i, logits in enumerate(layer_logits_list):
                if logits.shape[-1] >= mask_ids_tensor.max().item() + 1:
                    logits[:, :, mask_ids_tensor] = float("-inf")
                    layer_logits_list[i] = logits

    # --- Filter relevant layers for analysis ---
    valid_layers = []
    for lname, logits in zip(layer_names, layer_logits_list):
        lname_lower = lname.lower()

        # Skip embedding if requested
        if skip_input_layer and any(k in lname_lower for k in ["input", "wte", "wpe", "embed_tokens"]):
            continue

        # Include embeddings explicitly if requested
        if include_embed_tokens and "embed_tokens" in lname_lower:
            valid_layers.append((lname, logits))
            continue

        # Respect whether to include final norm
        if not include_final_norm and any(k in lname_lower for k in ["ln", "norm", "layernorm", "rmsnorm"]):
            continue

        # Analysis filtering
        if analysis_type == "self_att" and not (("self_att" in lname_lower) or ("embed_tokens" in lname_lower)):
            continue
        if analysis_type == "mlp" and not (("mlp" in lname_lower) or ("embed_tokens" in lname_lower)):
            continue

        valid_layers.append((lname, logits))

    if not valid_layers:
        return pd.DataFrame()

    batch_size, seq_len = valid_layers[0][1].shape[:2]
    vocab_size = getattr(wrapper.tokenizer, "vocab_size", None)

    for idx in range(batch_size):
        #intended_cloze_val = correct_cloze[idx] if correct_cloze and idx < len(correct_cloze) else None
        #input_ids_seq = outputs.input_ids[idx, :seq_len] if hasattr(outputs, "input_ids") else torch.arange(seq_len, dtype=torch.long)
        input_ids_seq = input_ids[idx, :seq_len]
        
        for l, (lname, logits) in enumerate(valid_layers):
            logits_cur = logits[idx, :seq_len].detach().cpu()
            logits_cur = safe_cast_logits(logits_cur, sanitize=False, target_dtype=_to_analysis_dtype(logits_cur))

            # Ensure batch dimension for later analysis
            if logits_cur.ndim == 2:  # [seq_len, vocab_size]
                logits_cur = logits_cur.unsqueeze(0)  # [1, seq_len, vocab_size]

            valid_len_next = logits_cur.shape[1]  # seq_len
            logits_cur = logits_cur[:, :-1, :]   # [1, seq_len-1, vocab_size]

            # Save row
            row = {
                "prompt_id": idx,
                "prompt_text": prompts[idx],
                "dataset": dataset_name,
                "layer_index": l,
                "layer_name": lname,
                "seq_len": valid_len_next,
                "vocab_size": vocab_size,
                "input_ids": input_ids_seq.detach().cpu(),
                "target_ids": input_ids_seq.detach().cpu()[1:], 
                "logits": logits_cur,
                #"logits": logits_cur.detach().cpu().numpy(),
                "position": torch.arange(seq_len-1),
            }

            rows.append(row)

    return pd.DataFrame(rows)

def run_logit_lens(
    wrapper: LogitLensWrapper,
    prompts: list[str],
    analysis_type: str = "all",
    model_name: str = "model",
    dataset_name: str = "dataset",
    save_dir: str = "logs/batch_logits",
    add_special_tokens: bool = False,  
    mask_bos: bool = True,
    mask_eos: bool = True,
    mask_pad: bool = True, 
    skip_input_layer: bool = False,
    include_embed_tokens: bool = True,     # <-- NEW
    include_final_norm: bool = True,
    proj_precision: Optional[str] = None,
    max_len: Optional[int] = 24,
    pad_to_max_length: bool = False,
    batch_size: int = 8   # <---- new argument
):
    """
    Run logit lens in smaller batches, saving each batch separately.
    Avoids holding all results in memory at once.
    """
    os.makedirs(save_dir, exist_ok=True)

    for i, start in enumerate(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[start:start+batch_size]
        df_batch = _run_logit_lens(
            wrapper=wrapper,
            prompts=batch_prompts,
            analysis_type=analysis_type,
            dataset_name=dataset_name,
            add_special_tokens=add_special_tokens,
            mask_bos=mask_bos,
            mask_eos=mask_eos,
            mask_pad=mask_pad,
            skip_input_layer=skip_input_layer,
            include_embed_tokens=include_embed_tokens,
            include_final_norm=include_final_norm,
            proj_precision=proj_precision,
            max_len=max_len,
            pad_to_max_length=pad_to_max_length
        )

        if not df_batch.empty:
            batch_path = os.path.join(
                save_dir,
                f"{dataset_name}_{model_name}_{analysis_type}_batch{i}.pt"
            )
            torch.save(df_batch, batch_path)
            print(f"Saved batch {i} -> {batch_path}")

        # free CUDA memory if necessary
        torch.cuda.empty_cache()

    print(f"All batches processed. Results saved under {save_dir}")

