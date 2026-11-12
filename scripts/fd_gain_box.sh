#!/bin/bash
# On the box, in tmux (tmux new -d -s fd "bash scripts/fd_gain_box.sh"), from /root/ortho: fd_gain.py's run stage, then acts/fd_gain uploaded to the HF dataset
# eekay/ortho-fd-gain (the laptop downloads it with hf download), then the instance destroys itself through the vast API with the team key in /root/.vast_key and its own
# id in /root/.vast_id, so nothing depends on the laptop being awake. The run log is acts/fd_gain/run.log and goes up with the results.
cd /root/ortho
export HF_TOKEN=$(grep -oP 'HF_TOKEN="\K[^"]+' ~/.bashrc)
.venv/bin/python fd_gain.py run 2>&1 | tee acts/fd_gain/run.log
.venv/bin/hf upload eekay/ortho-fd-gain acts/fd_gain acts/fd_gain --repo-type dataset 2>&1 | tail -3
curl -s -X DELETE "https://console.vast.ai/api/v0/instances/$(cat /root/.vast_id)/" -H "Authorization: Bearer $(cat /root/.vast_key)"
