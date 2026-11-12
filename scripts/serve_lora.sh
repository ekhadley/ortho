#!/bin/sh
# vLLM for the train_lora.py adapter runs, on the GPU box: sh serve_lora.sh. The base weights are served as cb_base from a ~/serve venv holding vllm
# 0.26.0+cu129 alone (setup_vast.sh's install without vllm-lens, which forces eager mode), so CUDA graphs are on. An adapter is loaded by
# POST /v1/load_lora_adapter {"lora_name", "lora_path"} and selected by an EvalCfg's lora field. Memory utilization is 0.80 because
# scripts/lora_check.py's prompt_logprobs on MBPP-length prompts need several GB outside the cache: the engine dies of OOM at 0.95.
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
exec ~/serve/bin/vllm serve Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --served-model-name cb_base --host 127.0.0.1 --port 8000 --trust-remote-code \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": true, "preserve_thinking": true}' \
    --enable-lora --max-lora-rank 32 --max-loras 2 --max-cpu-loras 8 --enable-prefix-caching --mamba-cache-mode align --gpu-memory-utilization 0.80 --max-num-seqs 128
