#!/bin/bash
# Run from the laptop: bash scripts/setup_vast.sh [ssh host, default vast] [hf model, default Qwen/Qwen3.6-27B]. Gets the box serving: the
# serving venv (~/serve) and the weights (the box's HF cache) if they are not there, the current serve.sh, and the server in a tmux session
# named serve (left alone if one is running), then waits until it answers. The server log is ~/vllm.log; tmux attach -t serve on the box.
set -e
HOST=${1:-vast}
MODEL=${2:-Qwen/Qwen3.6-27B}
scp "$(dirname "$0")/serve.sh" "$HOST":serve.sh
ssh "$HOST" "HF_TOKEN=${HF_TOKEN:?} bash -s" <<'EOF'
set -e
if [ ! -x ~/serve/bin/vllm ]; then
    uv venv ~/serve --python 3.12
    # the CUDA 12.9 builds of vllm and torch: PyPI's default vllm wheel pulls a CUDA 13 torch, which needs a host driver newer than many vast boxes have (a 555 driver supports 12.5; any 12.x build runs there)
    uv pip install --python ~/serve/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu129 \
        "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" "vllm-lens @ git+https://github.com/ekhadley/vllm-lens@main"
    uv pip install --python ~/serve/bin/python --reinstall-package torchcodec --index-url https://download.pytorch.org/whl/cpu torchcodec==0.17.0   # the CUDA 12.9 index has no torchcodec for this torch; vllm only decodes video with it
    ~/serve/bin/python -c "import torch, vllm; assert torch.version.cuda.startswith('12.'), torch.version.cuda; torch.zeros(1).cuda()"
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
