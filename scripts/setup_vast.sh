#!/bin/bash
# Run from the laptop: bash scripts/setup_vast.sh [ssh host, default vast] [hf model, default Qwen/Qwen3.6-27B]. Gets the box serving: the
# serving venv and the weights on the /workspace volume if they are not there, the current serve.sh, and the server in a tmux session named
# serve (left alone if one is running), then waits until it answers. The server log is /workspace/vllm.log; tmux attach -t serve on the box.
set -e
HOST=${1:-vast}
MODEL=${2:-Qwen/Qwen3.6-27B}
scp "$(dirname "$0")/serve.sh" "$HOST":/workspace/serve.sh
ssh "$HOST" 'bash -s' <<EOF
set -e
if [ ! -x /workspace/serve/bin/vllm ]; then
    uv venv /workspace/serve --python 3.12
    uv pip install --python /workspace/serve/bin/python vllm==0.26.0 "vllm-lens @ git+https://github.com/ekhadley/vllm-lens@main"
fi
HF_HOME=/workspace/.hf_home /workspace/serve/bin/hf download $MODEL
tmux has-session -t serve 2>/dev/null || tmux new -d -s serve "sh /workspace/serve.sh $MODEL 2>&1 | tee /workspace/vllm.log"
until curl -sf localhost:8000/v1/models; do
    tmux has-session -t serve
    sleep 5
done
EOF
echo
echo "ssh -N -f -L 8000:localhost:8000 $HOST"
