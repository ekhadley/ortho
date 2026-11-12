#!/usr/bin/env bash
# On the GPU box, from /root/ortho_train, with nothing on the GPU: widens grader_proj to every layer, runs capability.py on it, then serves row projall and scores the held-out completions through it.
set -euo pipefail
COMPAT=/root/.config/rhablation/cuda_compat_path
[ -s "$COMPAT" ] && export LD_LIBRARY_PATH="$(cat $COMPAT)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME=/workspace/.hf_home PYTHONUNBUFFERED=1
PY=/root/stage5-env/bin/python
$PY proj_widen.py grader_proj grader_proj_all
$PY proj_export.py data/vectors/Qwen3.8-27B/grader_proj_all.safetensors /workspace/models/projall_hook.safetensors
$PY capability.py
bash box_proj_setup.sh
bash /root/llm-reward-hacking-ablation/cheatbench/serve_row.sh projall
/root/vllm-env/bin/python proj_check.py cb_projall check_projall.json
/root/vllm-env/bin/python proj_check.py compare check_base.json check_projall.json
