#!/bin/bash
# Run from the laptop: bash scripts/setup_vast.sh [ssh host, default vast] [hf model, default Qwen/Qwen3.6-27B]. Gets the box serving: the
# serving venv (~/serve) and the weights (the box's HF cache) if they are not there, the current serve.sh, and the server in a tmux session
# named serve (left alone if one is running), then waits until it answers. The server log is ~/vllm.log; tmux attach -t serve on the box.
set -e
HOST=${1:-vast}
MODEL=${2:-Qwen/Qwen3.6-27B}
scp "$(dirname "$0")/serve.sh" "$HOST":serve.sh
ssh "$HOST" 'bash -s' <<EOF
set -e
if [ ! -x ~/serve/bin/vllm ]; then
    uv venv ~/serve --python 3.12
    uv pip install --python ~/serve/bin/python vllm==0.26.0 "vllm-lens @ git+https://github.com/ekhadley/vllm-lens@main"
fi
~/serve/bin/hf download $MODEL
tmux has-session -t serve 2>/dev/null || tmux new -d -s serve "sh ~/serve.sh $MODEL 2>&1 | tee ~/vllm.log"
until curl -sf localhost:8000/v1/models; do
    tmux has-session -t serve
    sleep 5
done
EOF
echo
echo "ssh -N -f -L 8000:localhost:8000 $HOST"
