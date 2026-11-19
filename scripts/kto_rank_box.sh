#!/bin/bash
# On a fresh box, in tmux, from /root/ortho: bash scripts/kto_rank_box.sh setup, then bash scripts/kto_rank_box.sh <config of kto_cfgs.py>.
# setup installs the venv and downloads the weights. A config name runs kto_sweep.py's worker on that one config, uploads its results line, parameters and
# log to the HF dataset eekay/ortho under acts/kto_sweep/rank/, then destroys the instance through the vast API with the team key in /root/.vast_key and
# its own id in /root/.vast_id, so nothing depends on the laptop being awake. A run without a results line leaves the box up.
NAME=$1
cd /root/ortho
export PATH=$HOME/.local/bin:$PATH PYTHONUNBUFFERED=1 HF_TOKEN=$(grep -oP 'HF_TOKEN="\K[^"]+' .env)
if [ "$NAME" = setup ]; then
    command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv sync
    .venv/bin/hf download Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 | tail -1
    .venv/bin/python -c "import torch; print('SETUP_DONE cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
    exit
fi
printf '%s\nSTOP\n' "$NAME" > acts/kto_sweep/queue.txt
.venv/bin/python kto_sweep.py run 2>&1 | tee acts/kto_sweep/run_$NAME.log
grep -q "^{\"name\": \"$NAME\"" acts/kto_sweep/results.jsonl || exit 1
mkdir -p up/vecs
grep "^{\"name\": \"$NAME\"" acts/kto_sweep/results.jsonl > up/$NAME.jsonl
tr '\r' '\n' < acts/kto_sweep/run_$NAME.log | grep -v "Loading weights" > up/$NAME.log
cp acts/kto_sweep/vecs/$NAME*.pt up/vecs/
.venv/bin/hf upload eekay/ortho up acts/kto_sweep/rank --repo-type dataset 2>&1 | tail -3 || exit 1
curl -s -X DELETE "https://console.vast.ai/api/v0/instances/$(cat /root/.vast_id)/" -H "Authorization: Bearer $(cat /root/.vast_key)"
