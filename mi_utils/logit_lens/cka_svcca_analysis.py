from typing import List, Optional
import os
import torch
import numpy as np
import pandas as pd
from scipy.special import rel_entr
from ..util.logit_lens_utils.logit_lens_wrapper import LogitLensWrapper
from ..util.logit_lens_utils.model_device_handling import get_base_model, get_embedding_device
from .metric_utils.logit_lens_helpers import(
    get_activation_tensor,
    safe_compute_cka,
    safe_compute_svcca,
    align_activations
)

# ----------------------------
# Reusable Inputs
# ----------------------------
EPS = 1e-12
TOPK = 5


def _run_cka_svcca(
    wrapper: LogitLensWrapper,
    prompts: list[str],
    A_acts: Optional[dict] = None,
    B_acts: Optional[dict] = None,
    skip_input_layer: bool = True,
    include_final_norm: bool = True,
) -> pd.DataFrame:
    """
    Run CKA and SVCCA similarity between two sets of activations (A_acts, B_acts).
    If A_acts/B_acts are None, falls back to wrapper.forward().
    """

    rows = []
    wrapper.model.eval()

    # --- Extract activations if not provided ---
    if A_acts is None or B_acts is None:
        _, layer_dict, layer_names = wrapper.forward(
            prompts,
            project_to_logits=False,
            return_hidden=True,
            keep_on_device=True
        )
        # Use same dict for both if none passed (self-similarity)
        if A_acts is None:
            A_acts = layer_dict
        if B_acts is None:
            B_acts = layer_dict
    else:
        # Layer names come from whichever dict we got
        layer_names = sorted(set(A_acts.keys()) & set(B_acts.keys()))

    # --- Iterate layers ---
    for lname in layer_names:
        lname_lower = lname.lower()
        if skip_input_layer and any(k in lname_lower for k in ["input", "embed_tokens", "wte", "wpe"]):
            continue
        if not include_final_norm and any(k in lname_lower for k in ["ln", "norm", "layernorm", "rmsnorm"]):
            continue

        try:
            actA = get_activation_tensor(A_acts[lname])
            actB = get_activation_tensor(B_acts[lname])
            actA, actB = align_activations(actA, actB)

            # Flatten [batch, seq, dim] -> [batch*seq, dim]
            A_flat = actA.reshape(-1, actA.shape[-1])
            B_flat = actB.reshape(-1, actB.shape[-1])

            cka_val = safe_compute_cka(A_flat, B_flat)
            svcca_val = safe_compute_svcca(A_flat, B_flat)
        except Exception as e:
            print(f"[WARN] Failed {lname}: {e}")
            cka_val = np.nan
            svcca_val = np.nan

        rows.append({
            "layer": lname,
            "cka": cka_val,
            "svcca": svcca_val,
        })

    return pd.DataFrame(rows)


def run_cka_svcca(
    wrapper: LogitLensWrapper,
    prompts: List[str],
    model_name: str = "model",
    dataset_name: str = "dataset",
    save_dir: str = "logs/cka_svcca_analysis",
    A_acts=None,
    B_acts=None,
    skip_input_layer: bool = True,
    include_final_norm: bool = True,
) -> None:
    """
    Top-level runner: runs _run_cka_svcca and saves .pt results.
    """
    df = _run_cka_svcca(
        wrapper=wrapper,
        prompts=prompts,
        A_acts=A_acts,
        B_acts=B_acts,
        skip_input_layer=skip_input_layer,
        include_final_norm=include_final_norm,
    )

    os.makedirs(save_dir, exist_ok=True)
    pt_path = f"{save_dir}/{dataset_name}_{model_name}.pt"
    torch.save(df, pt_path)
    print(f"[INFO] Saved CKA/SVCCA results to {pt_path}")
