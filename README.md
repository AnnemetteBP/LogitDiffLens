# LogitDiffLens

- LogitDiff: Logit Lens anslyis for comparing quantized and fine-tuned models to their base

## Datasets:
- [sentence-transformers/natural-questions](https://huggingface.co/datasets/sentence-transformers/natural-questions)
## Models:
- [HF1BitLLM/Llama3-8B-1.58-100B-tokens](https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens)
- [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

HF1BitLLM: Start by installing the transformers version with the correct configuration to load bitnet models:
pip install git+https://github.com/huggingface/transformers.git@refs/pull/33410/head
