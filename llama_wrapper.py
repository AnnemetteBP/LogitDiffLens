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


# -----------------------------
# Core Wrapper
# -----------------------------
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


class LlamaPromptLens:
    """
    Unified Logit Lens interface for LLaMA/BitNet-style models.

    Normalization modes:
      - "none":      raw hidden states (no per-layer normalization)
      - "model":     use model’s own layernorms (faithful to internal flow)
      - "unit_l2":   normalize hidden states to unit L2 norm
      - "unit_rms":  apply final RMSNorm weights at every layer
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        normalization_mode: str = "none",
        include_subblocks: bool = False,
        device: Optional[str] = None,
    ):
        # ---- Load model and tokenizer ----
        self.model, self.tokenizer = load_model_and_tok(model_id)
        self.device = device or get_model_device(self.model)
        self.normalization_mode = normalization_mode
        self.include_subblocks = include_subblocks

        # ---- Architecture detection ----
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

        # ---- Initialize any uninitialized RoPE modules ----
        for name, module in self.model.named_modules():
            if isinstance(module, LlamaRotaryEmbedding):
                if getattr(module, "inv_freq", None) is None:
                    dummy_pos = torch.arange(1, device=self.device).unsqueeze(0)
                    try:
                        module._dynamic_frequency_update(dummy_pos, device=self.device)
                        print(f"[init] Initialized missing inv_freq in {name}")
                    except Exception as e:
                        print(f"[warn] Could not init RoPE in {name}: {e}")

        # ---- Report RoPE types for debugging ----
        for name, module in self.model.named_modules():
            rope = getattr(getattr(module, "self_attn", None), "rotary_emb", None)
            if rope is None:
                continue
            if isinstance(rope, LlamaRotaryEmbedding):
                print(f"[ok] {name}.self_attn.rotary_emb is LlamaRotaryEmbedding (old API)")
            elif isinstance(rope, torch.Tensor):
                print(f"[ok] {name}.self_attn.rotary_emb is tensor (new API)")
            elif isinstance(rope, tuple):
                print(f"[ok] {name}.self_attn.rotary_emb is tuple (cos,sin)")
            else:
                print(f"[warn] {name}.self_attn.rotary_emb unexpected type: {type(rope)}")

        # ---- Ensure tokenizer special tokens ----
        self._ensure_special_tokens()

        # ---- Move to device if safe ----
        if not self.is_quantized and not self.is_bitnet:
            self.model = self.model.to(self.device)

    # ------------------------
    # Token + model prep
    # ------------------------
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


    # -----------------
    # Normalization utilities
    # -----------------
    def _apply_normalization(self, tensor, block):
        mode = self.normalization_mode
        if mode == "none":
            # raw lens: no normalization
            return tensor
        elif mode == "model":
            # model-faithful normalization: use the model’s internal post-attention norm
            return block.post_attention_layernorm(tensor)
        elif mode == "unit_l2":
            # heuristic normalization
            return tensor / (tensor.norm(dim=-1, keepdim=True) + 1e-6)
        elif mode == "unit_rms":
            # apply final model RMSNorm weights at every layer
            return self.model.base_model.norm(tensor)
        else:
            raise ValueError(f"Unknown normalization_mode: {mode}")


    def _normalize_for_mode(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize subblock outputs consistently with the active mode."""
        mode = self.normalization_mode
        if mode in ("none", "model"):
            return tensor
        elif mode == "unit_l2":
            return tensor / (tensor.norm(dim=-1, keepdim=True) + 1e-6)
        elif mode == "unit_rms":
            return self.model.base_model.norm(tensor)
        else:
            return tensor

    def _apply_final_normalization(self, tensor):
        mode = self.normalization_mode
        if mode in ("none", "model"):
            # the model always applies its final RMSNorm before logits
            return self.model.base_model.norm(tensor)
        elif mode == "unit_l2":
            return tensor / (tensor.norm(dim=-1, keepdim=True) + 1e-6)
        elif mode == "unit_rms":
            return self.model.base_model.norm(tensor)

    # -----------------
    # Stack and utility
    # -----------------
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

def _ensure_rope_initialized(model, device):
    """Ensure all LlamaRotaryEmbedding modules have valid inv_freq."""
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    for name, module in model.named_modules():
        if isinstance(module, LlamaRotaryEmbedding):
            if getattr(module, "inv_freq", None) is None:
                dummy_pos = torch.arange(1, device=device).unsqueeze(0)
                try:
                    module._dynamic_frequency_update(dummy_pos, device=device)
                    print(f"[init] Initialized inv_freq in {name}")
                except Exception as e:
                    print(f"[warn] Failed to init inv_freq in {name}: {e}")

def _ensure_rope_support(model, device):
    """
    Detects and adapts for different RoPE implementations:
    - Old API: LlamaRotaryEmbedding (callable)
    - New API: tensor-based (precomputed cos/sin)
    Returns tuple (use_external_embeddings, rope_source)
    """
    first_attn = model.base_model.layers[0].self_attn
    rope = getattr(first_attn, "rotary_emb", None)

    if rope is None:
        print("[warn] No rotary_emb found in first attention block.")
        return False, None

    if isinstance(rope, torch.Tensor):
        print("[info] RoPE is a tensor — using external embeddings (new API).")
        return True, rope  # tensor already
    elif hasattr(rope, "forward"):  # old LlamaRotaryEmbedding
        print("[info] RoPE is a callable module — using legacy API.")
        return False, rope
    else:
        print(f"[warn] Unknown RoPE type: {type(rope)}")
        return False, rope

# -----------------
# Batched version
# -----------------
import torch
import pandas as pd
from typing import Optional

@torch.no_grad()
def _run_logit_lens_batch(
    lens,
    prompts: list[str],
    dataset_name: str = "default",
    mask_special: bool = True,
    include_embed_tokens: bool = True,
    include_output: bool = True,
    proj_precision: Optional[str] = None,
    pad_to_max_length: bool = False,
    max_len: Optional[int] = 20,
    save_path: Optional[str] = None,
):
    """
    Run the logit lens on a batch of prompts.
    Works with subblocks (attn/MLP) or whole blocks.
    Fully compatible with HF ≥4.46 (new RoPE API).
    BitNet + standard LLaMA-safe.
    """

    model = lens.model
    tokenizer = lens.tokenizer
    device = get_model_device(model)
    model.eval()

    # ---- Ensure RoPE initialized & detect type ----
    _ensure_rope_initialized(model, device)
    use_external_rope, rope_module = _ensure_rope_support(model, device)

    # ---- Tokenize ----
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest" if not pad_to_max_length else True,
        truncation=True,
        max_length=max_len,
    ).to(device)

    batch_size, seq_len = inputs.input_ids.shape
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # ---- Embed ----
    x = model.base_model.embed_tokens(inputs.input_ids)
    batch_layer_logits = {"embed_tokens": model.lm_head(x)}

    # ---- Traverse transformer layers ----
    for i, block in enumerate(model.base_model.layers):

        if lens.include_subblocks:
            # --- Attention sublayer ---
            attn_in = block.input_layernorm(x)
            with torch.autocast(device_type=device.type, enabled=False):
                if use_external_rope:
                    attn_out = block.self_attn(attn_in, position_embeddings=rope_module)[0]
                else:
                    attn_out = block.self_attn(attn_in, position_ids=position_ids)[0]

            attn_hidden = x + attn_out

            # --- MLP sublayer ---
            mlp_in = block.post_attention_layernorm(attn_hidden)
            mlp_out = block.mlp(mlp_in)
            block_out = attn_hidden + mlp_out

            normed = lens._apply_normalization(block_out, block)

            # --- Logit projections ---
            batch_layer_logits[f"layer.{i}.attn.delta"] = model.lm_head(lens._normalize_for_mode(attn_out))
            batch_layer_logits[f"layer.{i}.attn.post"]  = model.lm_head(lens._normalize_for_mode(attn_hidden))
            batch_layer_logits[f"layer.{i}.mlp.delta"]  = model.lm_head(lens._normalize_for_mode(mlp_out))
            batch_layer_logits[f"layer.{i}.mlp.post"]   = model.lm_head(lens._normalize_for_mode(block_out))
            batch_layer_logits[f"layer.{i}.block_out"]  = model.lm_head(normed)

            x = block_out

        else:
            # --- Whole block probing ---
            with torch.autocast(device_type=device.type, enabled=False):
                if use_external_rope:
                    out = block(x, position_embeddings=rope_module)[0]
                else:
                    out = block(x, position_ids=position_ids)[0]

            normed = lens._apply_normalization(out, block)
            batch_layer_logits[f"layer.{i}"] = model.lm_head(normed)
            x = out

    # ---- Final normalization ----
    if include_output:
        final_hidden = lens._apply_final_normalization(x)
        batch_layer_logits["output"] = model.lm_head(final_hidden)

    # ---- Mask special tokens ----
    if mask_special:
        special_ids = [
            t for t in [
                tokenizer.bos_token_id,
                tokenizer.eos_token_id,
                tokenizer.pad_token_id
            ] if t is not None
        ]
        for logits in batch_layer_logits.values():
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
        true_len = (
            (input_ids_seq != tokenizer.pad_token_id).sum().item()
            if tokenizer.pad_token_id is not None
            else seq_len
        )
        for li, (lname, logits) in enumerate(zip(layer_names, stacked)):
            logits_cur = logits[b, :true_len].float().cpu()
            if proj_precision == "fp16":
                logits_cur = logits_cur.half()
            logits_cur = logits_cur[:-1]
            rows.append({
                "prompt_id": b,
                "prompt_text": prompts[b],
                "dataset": dataset_name,
                "layer_index": li,
                "layer_name": lname,
                "input_ids": input_ids_seq,
                "target_ids": input_ids_seq[1:true_len],
                "logits": logits_cur,
                "position": torch.arange(true_len - 1),
            })

    df = pd.DataFrame(rows)
    if save_path:
        torch.save(df, save_path)
    return df


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
    import gc

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
        
        del df_batch, batch_prompts
        gc.collect()                      # clears CPU memory
        if torch.cuda.is_available():
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
            max_steps=max_steps, 
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

