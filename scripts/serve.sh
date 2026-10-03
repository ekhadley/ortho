#!/bin/sh
# vLLM with the vllm-lens plugin on the GPU box, for the Inspect tasks: sh serve.sh [hf model] [served name]. The venv is /workspace/serve
# (vllm 0.26.0, vllm-lens 1.2.1 from the fork ekhadley/vllm-lens, Python 3.12) and the weights are in the volume's HF cache. The server binds to localhost; the runner reaches
# it through `ssh -N -L 8000:localhost:8000 vast` as VLLM_BASE_URL=http://localhost:8000/v1. Adapters load without a restart through
# POST /v1/load_lora_adapter {"lora_name", "lora_path"} and are selected by the request's model field (an EvalCfg's lora field). Prefix
# caching is on (opt-in for this hybrid model class, which needs the align mamba cache mode): requests with no intervention and lora requests
# reuse cached prefixes, keyed by adapter and by the intervention fields common.py sends as cache_salt. vllm-lens makes every hooked request
# (adds, ablations) skip reading the cache, so those prefill in full; they still write blocks, which the salt keeps away from other interventions.
set -e
MODEL=${1:-Qwen/Qwen3.6-27B}
NAME=${2:-$(basename "$MODEL")}
export HF_HOME=/workspace/.hf_home VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
exec /workspace/serve/bin/vllm serve "$MODEL" --served-model-name "$NAME" --host 127.0.0.1 --port 8000 --trust-remote-code --enforce-eager \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": true, "preserve_thinking": true}' \
    --enable-lora --max-lora-rank 32 --enable-prefix-caching --mamba-cache-mode align
