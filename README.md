# LogitDiffLens

- Was meant to compare the results of two Logit Lenses
- Right now measures the interpretability and degradation in models (like PTQ/QAT)

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