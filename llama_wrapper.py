from typing import List, Optional, Dict
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd


def make_print_if_verbose(verbose: bool):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    return vprint
            
# -----------------------------
# Utility functions
# -----------------------------

def get_model_device(model: torch.nn.Module) -> torch.device:
    """Safely infer device even for quantized / bitsandbytes models."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        for buf in model.buffers():
            return buf.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_quantized_model(model) -> bool:
    import bitsandbytes as bnb
    return any(isinstance(m, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt))
               for m in model.modules())


def is_bitnet_model(model) -> bool:
    return any(
        hasattr(m, "bit_linear") or m.__class__.__name__ == "BitLinear"
        for m in model.modules()
    )

def detect_architecture(model):
    base_model = getattr(model, "base_model", model)
    if is_bitnet_model(model):
        return "bitnet"
    if hasattr(base_model, "layers") or (
        hasattr(base_model, "model") and hasattr(base_model.model, "layers")
    ):
        return "llama"
    if hasattr(base_model, "h") or (
        hasattr(base_model, "transformer") and hasattr(base_model.transformer, "h")
    ):
        return "gpt"
    raise NotImplementedError(f"Cannot detect architecture for {type(model)}")

def mask_special_logits(logits_dict, tokenizer):
    special_ids = [
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    ]
    special_ids = [sid for sid in special_ids if sid is not None]
    for name, logits in logits_dict.items():
        logits[:, :, special_ids] = float("-inf")
    return logits_dict

# -----------------------------
# Core Wrapper
# -----------------------------

def load_model_and_tok(
    model_name: str,
    low_cpu_mem_usage: bool = True,
    local_files_only: bool = True,
    device_map: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True,
        return_dict_in_generate=True,
        return_dict=True,
        output_attentions=True,
        low_cpu_mem_usage=low_cpu_mem_usage,
        local_files_only=local_files_only,
        device_map=device_map,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    return model, tok


class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, lm_head, global_norm, apply_per_layer_norm=False):
        super().__init__()
        self.block = block
        self.lm_head = lm_head
        self.global_norm = global_norm
        self.apply_per_layer_norm = apply_per_layer_norm

    def forward(self, x, **kwargs):
        out = self.block(x, **kwargs)
        hidden = out[0] if isinstance(out, tuple) else out
        normed = (
            self.block.post_attention_layernorm(hidden)
            if self.apply_per_layer_norm
            else hidden
        )
        logits = self.lm_head(normed)
        return hidden, logits


class LlamaPromptLens:
    """
    Unified Logit Lens interface for LLaMA-style models.
    Supports:
      - Nostalgebraist (only final norm): https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
      - LogitLens4LLMs (per-layer norm): https://arxiv.org/abs/2503.11667
      - optional subblocks (attn/mlp)
      - embedding token probing
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        apply_per_layer_norm: bool = False,
        include_subblocks: bool = False,
        device: Optional[str] = None,
    ):
        self.model, self.tokenizer = load_model_and_tok(model_id)
        self.device = device or get_model_device(self.model)
        self.apply_per_layer_norm = apply_per_layer_norm
        self.include_subblocks = include_subblocks

        self.is_quantized = is_quantized_model(self.model)
        self.is_bitnet = is_bitnet_model(self.model)
        self.arch = detect_architecture(self.model)

        print(f"Architecture detected: {self.arch}")
        if self.is_bitnet:
            print("BitNet model (BitLinear layers).")
        elif self.is_quantized:
            print("Quantized model (bitsandbytes Linear).")
        else:
            print("Standard FP16 or FP32 model.")

        self._ensure_special_tokens()

        if not self.is_quantized and not self.is_bitnet:
            self.model = self.model.to(self.device)

    def _ensure_special_tokens(self):
        specials = {}
        if not getattr(self.tokenizer, "bos_token", None):
            specials["bos_token"] = "<s>"
        if not getattr(self.tokenizer, "eos_token", None):
            specials["eos_token"] = "</s>"
        if not getattr(self.tokenizer, "pad_token", None):
            specials["pad_token"] = self.tokenizer.eos_token or "[PAD]"
        if specials:
            self.tokenizer.add_special_tokens(specials)
            if hasattr(self.model, "resize_token_embeddings"):
                self.model.resize_token_embeddings(len(self.tokenizer))
    
        #print(self.tokenizer .pad_token, self.tokenizer .pad_token_id)

    # -----------------
    # Probe (single prompt)
    # -----------------
    @torch.no_grad()
    def probe_prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
        model = self.model
        model_device = get_model_device(model)
        tokenizer = self.tokenizer
        model.eval()

        #print("Tokenizer length:", len(tokenizer))
        #print("Model lm_head weight shape:", model.lm_head.weight.shape)

        inputs = tokenizer(prompt, return_tensors="pt").to(model_device)
        x = model.model.embed_tokens(inputs.input_ids)
        layer_logits = {"embed_tokens": model.lm_head(x)}

        for i, block in enumerate(model.model.layers):
            if self.include_subblocks:
                attn_in = block.input_layernorm(x)
                attn_out = block.self_attn(attn_in)[0]
                attn_hidden = x + attn_out

                mlp_in = block.post_attention_layernorm(attn_hidden)
                mlp_out = block.mlp(mlp_in)
                block_out = attn_hidden + mlp_out

                normed = (
                    block.post_attention_layernorm(block_out)
                    if self.apply_per_layer_norm
                    else block_out
                )

                layer_logits[f"layer_{i}.self_attn"] = model.lm_head(attn_out)
                layer_logits[f"layer_{i}.mlp"] = model.lm_head(mlp_out)
                layer_logits[f"layer_{i}.block_out"] = model.lm_head(normed)

                x = block_out
            else:
                out = block(x)[0]
                normed = (
                    block.post_attention_layernorm(out)
                    if self.apply_per_layer_norm
                    else out
                )
                layer_logits[f"layer_{i}"] = model.lm_head(normed)
                x = out

        if not self.apply_per_layer_norm:
            final_logits = model.lm_head(model.model.norm(x))
            layer_logits["output"] = final_logits

        return layer_logits

    @staticmethod
    def stack_layer_logits(
        layer_dict: Dict[str, torch.Tensor],
        keep_on_device: bool = True,
        skip_embed: bool = False,
        include_output: bool = True,
    ):
        names, tensors = [], []
        for name, logits in layer_dict.items():
            if skip_embed and name == "embed_tokens":
                continue
            if not include_output and name == "output":
                continue
            t = logits if keep_on_device else logits.detach().cpu()
            tensors.append(t)
            names.append(name)
        stacked = torch.stack(tensors, dim=0)
        return stacked, names


# -----------------
# Batched version
# -----------------
@torch.no_grad()
def _run_logit_lens_batch(
    lens: LlamaPromptLens,
    prompts: List[str],
    dataset_name: str = "default",
    mask_special: bool = True,
    include_embed_tokens: bool = True,
    include_output: bool = True,
    proj_precision: Optional[str] = None,
    pad_to_max_length: bool = False,
    max_len: Optional[int] = 20,
):
    """Run the logit lens on a batch of prompts and collect logits per layer."""
    model = lens.model
    tokenizer = lens.tokenizer
    model.eval()

    model_device = get_model_device(model)
    vocab_size = getattr(tokenizer, "vocab_size", None) or getattr(model.lm_head, "out_features", None)

    # ---- Tokenize ----
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest" if not pad_to_max_length else True,
        truncation=True,
        max_length=max_len,
    ).to(model_device)

    batch_size, seq_len = inputs.input_ids.shape
    position_ids = torch.arange(seq_len, device=model_device).unsqueeze(0).expand(batch_size, -1)

    # Precompute RoPE cos/sin embeddings if model supports it
    position_embeddings = None
    if hasattr(model.model, "rotary_emb"):
        try:
            cos, sin = model.model.rotary_emb(
                model.model.embed_tokens(inputs.input_ids), position_ids
            )
            position_embeddings = (cos, sin)
        except Exception:
            # Fallback for older HF versions
            position_embeddings = None

    # ---- Forward ----
    x = model.model.embed_tokens(inputs.input_ids)
    batch_layer_logits = {"embed_tokens": model.lm_head(x)}

    for i, block in enumerate(model.model.layers):
        # Use new position_embeddings if available, otherwise fallback
        block_kwargs = (
            {"position_embeddings": position_embeddings}
            if position_embeddings is not None
            else {"position_ids": position_ids}
        )

        if lens.include_subblocks:
            attn_in = block.input_layernorm(x)
            attn_out = block.self_attn(attn_in, **block_kwargs)[0]
            attn_hidden = x + attn_out

            mlp_in = block.post_attention_layernorm(attn_hidden)
            mlp_out = block.mlp(mlp_in)
            block_out = attn_hidden + mlp_out

            normed = (
                block.post_attention_layernorm(block_out)
                if lens.apply_per_layer_norm
                else block_out
            )

            batch_layer_logits[f"layer_{i}.self_attn"] = model.lm_head(attn_out)
            batch_layer_logits[f"layer_{i}.mlp"] = model.lm_head(mlp_out)
            batch_layer_logits[f"layer_{i}.block_out"] = model.lm_head(normed)
            x = block_out
        else:
            out = block(x, **block_kwargs)[0]
            normed = (
                block.post_attention_layernorm(out)
                if lens.apply_per_layer_norm
                else out
            )
            batch_layer_logits[f"layer_{i}"] = model.lm_head(normed)
            x = out

    # ---- Final layer norm if nostalgebraist ----
    if not lens.apply_per_layer_norm and include_output:
        batch_layer_logits["output"] = model.lm_head(model.model.norm(x))

    # ---- Mask special tokens ----
    if mask_special:
        special_ids = [
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
        ]
        special_ids = [sid for sid in special_ids if sid is not None]
        for lname, logits in batch_layer_logits.items():
            logits[:, :, special_ids] = float("-inf")

    # ---- Stack all layers ----
    stacked, layer_names = lens.stack_layer_logits(
        batch_layer_logits,
        keep_on_device=False,
        skip_embed=not include_embed_tokens,
        include_output=include_output,
    )

    # ---- Build DataFrame ----
    rows = []
    for b in range(batch_size):
        input_ids_seq = inputs.input_ids[b].cpu()
        # compute actual prompt length (excluding PAD)
        true_len = (
            (input_ids_seq != tokenizer.pad_token_id).sum().item()
            if tokenizer.pad_token_id is not None
            else seq_len
        )

        for li, (lname, logits) in enumerate(zip(layer_names, stacked)):
            logits_cur = logits[b, :true_len].float()
            if proj_precision == "fp16":
                logits_cur = logits_cur.half()

            # Drop final timestep (no next-token target)
            logits_cur = logits_cur[:-1]

            row = {
                "prompt_id": b,
                "prompt_text": prompts[b],
                "dataset": dataset_name,
                "vocab_size": vocab_size,
                "layer_index": li,
                "layer_name": lname,
                "input_ids": input_ids_seq,
                "target_ids": input_ids_seq[1:true_len],
                "logits": logits_cur,
                "position": torch.arange(true_len - 1),
            }
            rows.append(row)

    return pd.DataFrame(rows)


@torch.no_grad()
def _run_logit_lens_autoregressive_batch(
    lens,
    prompts,
    dataset_name="default",
    proj_precision=None,
    max_steps=10,
    mask_special=True,
    mode="teacher",  # ["teacher", "greedy", "sample"]
    temperature=None,
):
    """
    Autoregressive or teacher-forced logit-lens analysis.
    Collects per-layer logits after each generation step.

    Modes:
        - "teacher": Use the ground truth next tokens (deterministic teacher forcing)
        - "greedy":  Autoregressive deterministic generation (argmax)
        - "sample":  Autoregressive stochastic generation (sampling)

    Args:
        lens: LlamaPromptLens instance
        prompts: List[str]
        dataset_name: str
        proj_precision: Optional[str] ("fp16" or "fp32")
        max_steps: int, number of tokens to generate per prompt
        mask_special: bool, mask BOS/EOS/PAD tokens
        mode: "teacher" | "greedy" | "sample"
        temperature: float, optional override (defaults depend on mode)
    """
    model = lens.model
    tokenizer = lens.tokenizer
    model.eval()
    device = next(model.parameters()).device

    # --- Mode defaults ---
    teacher_forcing = (mode == "teacher")
    greedy = (mode == "greedy")
    if temperature is None:
        temperature = 1e-6 if mode in ["teacher", "greedy"] else 0.7

    # --- Token + vocab info ---
    vocab_size = getattr(tokenizer, "vocab_size", None) or getattr(model.lm_head, "out_features", None)
    special_ids = set(
        sid for sid in [
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            128000,  # Unicode prefix for LLaMA-3
        ] if sid is not None
    )

    rows = []

    for b, prompt in enumerate(prompts):
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        full_input_ids = encoded.input_ids[0]

        # Start with BOS or first token
        input_ids = full_input_ids[:1].unsqueeze(0)
        current_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        for step in range(max_steps):
            # --- Run logit-lens on current prefix ---
            layer_logits = lens.probe_prompt(current_text)
            stacked, layer_names = lens.stack_layer_logits(layer_logits, keep_on_device=False)
            seq_len = input_ids.size(1)

            for li, (lname, logits) in enumerate(zip(layer_names, stacked)):
                logits_cur = logits[0, :seq_len].float()
                if proj_precision == "fp16":
                    logits_cur = logits_cur.half()
                logits_cur = logits_cur[:-1]  # drop last step (no target)

                rows.append({
                    "prompt_id": b,
                    "prompt_text": prompt,
                    "dataset": dataset_name,
                    "vocab_size": vocab_size,
                    "layer_index": li,
                    "layer_name": lname,
                    "input_ids": input_ids[0].cpu(),
                    "target_ids": input_ids[0, 1:seq_len],
                    "logits": logits_cur,
                    "position": torch.arange(seq_len - 1),
                    "generated_step": step,
                    "generated_text": current_text,
                })

            # --- Determine next token ---
            if teacher_forcing:
                if step + 1 >= len(full_input_ids):
                    break
                next_token = full_input_ids[step + 1].unsqueeze(0).unsqueeze(0)
            else:
                logits = model(input_ids).logits
                next_token_logits = logits[:, -1, :]

                if mask_special and len(special_ids) > 0:
                    next_token_logits[:, list(special_ids)] = float("-inf")

                next_token_logits /= temperature
                probs = torch.softmax(next_token_logits, dim=-1)

                next_token = (
                    torch.argmax(probs, dim=-1, keepdim=True)
                    if greedy else torch.multinomial(probs, num_samples=1)
                )

            # --- Update context ---
            next_token_id = next_token.item()
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            current_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

            if next_token_id in special_ids:
                break

        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    return df


def run_logit_lens_batched(
    lens: LlamaPromptLens,
    prompts: List[str],
    dataset_name: str = "default",
    model_name: str = "model",
    save_dir: str = "logs/lens_batches",
    proj_precision: str = None,
    batch_size: int = 8,
    **kwargs,
):
    """
        Run prompt probing lens
    """
    os.makedirs(save_dir, exist_ok=True)
    for i, start in enumerate(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[start : start + batch_size]
        df_batch = _run_logit_lens_batch(
            lens=lens,
            prompts=batch_prompts,
            dataset_name=dataset_name,
            proj_precision=proj_precision,
            **kwargs,
        )
        if not df_batch.empty:
            batch_path = os.path.join(
                save_dir, f"{dataset_name}_{model_name}_batch{i}.pt"
            )
            torch.save(df_batch, batch_path)
            print(f"[✓] Saved batch {i}: {batch_path}")
        torch.cuda.empty_cache()
    print(f"All {len(prompts)} prompts processed.")


def run_logit_lens_autoregressive_batched(
    lens: LlamaPromptLens,
    prompts: List[str],
    dataset_name: str = "default",
    model_name: str = "model",
    save_dir: str = "logs/lens_batches_autoreg",
    proj_precision: str = None,
    batch_size: int = 1,
    max_steps: int = 5,
    **kwargs,
):
    """
    Run autoregressive or teacher-forced logit-lens analysis in batches.
    Params/Args:
        Mode: mode = "teacher", or "greedy" / "sample"

    Usage examples:
        1. Teacher-Forcing (Evaluation Mode)
            run_logit_lens_autoregressive_batched(
                lens, ["Translate: English 'flower' → German: Blume"],
                mode="teacher", max_steps=10
            )

        2. Greedy Autoregressive (Deterministic Drift)
            run_logit_lens_autoregressive_batched(
                lens, ["The capital of France is"],
                mode="greedy", max_steps=10
            )

        3. Sampling Autoregressive (Stochastic Drift)
            run_logit_lens_autoregressive_batched(
                lens, ["Daniel went to the garden. Mary went to the kitchen. Where is Mary? Answer:"],
                mode="sample", temperature=0.7, max_steps=10
            )
    """
    os.makedirs(save_dir, exist_ok=True)

    for i, start in enumerate(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[start : start + batch_size]

        df_batch = _run_logit_lens_autoregressive_batch(
            lens=lens,
            prompts=batch_prompts,
            dataset_name=dataset_name,
            proj_precision=proj_precision,
            max_steps=max_steps,  # ← fixed here
            **kwargs,
        )

        if not df_batch.empty:
            batch_path = os.path.join(
                save_dir, f"{dataset_name}_{model_name}_autoreg_batch{i}.pt"
            )
            torch.save(df_batch, batch_path)
            print(f"[✓] Saved autoregressive batch {i}: {batch_path}")

        torch.cuda.empty_cache()

    print(f"All {len(prompts)} prompts processed autoregressively.")

