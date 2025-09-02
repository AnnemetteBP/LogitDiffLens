# LogitDiffLens

- Logit Lens anslyis for quantized models and comparison

## Datasets:
- [sentence-transformers/natural-questions](https://huggingface.co/datasets/sentence-transformers/natural-questions)
- [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k/viewer/main/train?row=7294)
## Models:
- [HF1BitLLM/Llama3-8B-1.58-100B-tokens](https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens)
- [HF1BitLLM/Llama3-8B-1.58-Linear-10B-tokens](https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-Linear-10B-tokens)
- [HF1BitLLM/Llama3-8B-1.58-Sigmoid-k100-10B-tokens](https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-Sigmoid-k100-10B-tokens)

HF1BitLLM: Start by installing the transformers version with the correct configuration to load bitnet models:
pip install git+https://github.com/huggingface/transformers.git@refs/pull/33410/head

- [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [GPT-2](https://huggingface.co/docs/transformers/en/model_doc/gpt2) (just for testing code compatability)
- [allenai/OLMo-1B-hf](https://huggingface.co/allenai/OLMo-1B-hf)
- [allenai/OLMo-7B-hf](https://huggingface.co/allenai/OLMo-7B-hf)
- [NousResearch/DeepHermes-3-Llama-3-3B-Preview](https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-3B-Preview)
- [NousResearch/DeepHermes-3-Llama-3-8B-Preview](https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview)
Loading with [Bitsandbytes](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes) 4- and 8-bit.
