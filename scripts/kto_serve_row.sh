#!/bin/bash
# On the GPU box: bash ~/serve_row.sh <row> [vision] serves the base weights as cb_<row> with the hook file ~/hooks/<row>.safetensors (./kto_sweep.py export or scripts/proj_export.py; none for row base), replacing what is served, and waits until it answers. Installed by scripts/kto_serve_setup.sh.
row=$1
lm_only=$([ "$2" = vision ] || echo --language-model-only)   # bash ~/serve_row.sh <row> vision also serves the vision encoder, which CheatBench tasks need
tmux kill-session -t vllm 2>/dev/null
for _ in $(seq 30); do curl -sf -m 2 localhost:8000/v1/models > /dev/null || break; sleep 2; done
hook=""
[ "$row" != base ] && hook=$HOME/hooks/$row.safetensors && [ ! -f "$hook" ] && { echo "no $hook"; exit 1; }
cache=$([ "$row" = base ] && echo $HOME/.cache/vllm_base || echo $HOME/.cache/vllm)   # the compile cache does not know about the hook, so the unhooked model keeps its own
tmux new -d -s vllm "VLLM_CACHE_ROOT=$cache RHABLATION_PROJ=$hook VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_FLASHINFER_SAMPLER=0 ~/serve/bin/vllm serve Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --served-model-name cb_$row --host 127.0.0.1 --port 8000 --max-model-len 131072 --gpu-memory-utilization 0.92 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml $lm_only --max-num-seqs 64 --speculative-config \"{\\\"method\\\": \\\"mtp\\\", \\\"num_speculative_tokens\\\": 3}\" 2>&1 | tee ~/vllm_$row.log"
for _ in $(seq 240); do
    curl -sf -m 5 localhost:8000/v1/models | grep -q "cb_$row" && { echo "serving cb_$row"; grep -c "vllm_proj: hook on" ~/vllm_$row.log; exit 0; }
    tmux has-session -t vllm 2>/dev/null || break
    sleep 5
done
echo "cb_$row not serving"; tail -20 ~/vllm_$row.log; exit 1
