# LogitDiffLens

- Was meant to compare the results of two Logit Lenses
- Right now measures the interpretability and degradation in models (like PTQ/QAT)

## The UNSAFE interpretability score is designed to be between 0 and 1, where:

- 1.0 → perfect interpretability, i.e., no NaNs or Infs were encountered in logits, probabilities, or entropy across all layers.

- 0.0 → extremely degraded, i.e., the layers are mostly NaN/Inf, making the latent structure effectively uninterpretable.

### How it’s computed

Layers are split into sections: first, early, mid, late, last.

#### For each section:

- Compute the max NaN/Inf rate across all metrics (logits, probs, entropy) in that section.

- The section score is 1 - mean(max_rate_across_layers).

- Overall score = mean of section scores, ignoring sections with no layers.

- So the score directly reflects degradation: higher NaN/Inf rates → lower score.


## Models:
- [HF1BitLLM/Llama3-8B-1.58-100B-tokens](https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens)
Fine-tuned on 100B tokens for maximum performance.
- [HF1BitLLM/Llama3-8B-1.58-Linear-10B-tokens](https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-Linear-10B-tokens)
Fine-tuned on 100B tokens for maximum performance.
- [HF1BitLLM/Llama3-8B-1.58-Sigmoid-k100-10B-tokens](https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-Sigmoid-k100-10B-tokens)
Fine-tuned on 100B tokens for maximum performance.

HF1BitLLM: Start by installing the transformers version with the correct configuration to load bitnet models:
pip install git+https://github.com/huggingface/transformers.git@refs/pull/33410/head

- [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [GPT-2](https://huggingface.co/docs/transformers/en/model_doc/gpt2)
Load with [Bitsandbytes](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes) 4- and 8-bit.