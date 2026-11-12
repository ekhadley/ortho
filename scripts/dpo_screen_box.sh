#!/bin/bash
# On the box, in tmux (tmux new -d -s screen "bash scripts/dpo_screen_box.sh"), from /root/ortho: the screens written by scripts/dpo_screen.py, the grader ones
# first, one log per screen and a line per step in acts/dpo_screen/logs/chain.log. The scores land in data/vectors/Qwen3.8-27B/dpo_screen/<tag>.json.
cd /root/ortho
export PYTHONUNBUFFERED=1 HF_TOKEN=$(grep -oP 'HF_TOKEN="?\K[^"]+' .env)
L=acts/dpo_screen/logs; mkdir -p $L
for tag in mbpp_gen grader_gen grader_add36 grader_proj grader_add12 grader_add24 mbpp_proj mbpp_add12 mbpp_add24 mbpp_add36; do
    echo "$(date -u +%T) $tag" >> $L/chain.log
    .venv/bin/python acts/dpo_screen/screen_$tag.py > $L/$tag.log 2>&1 || echo "$(date -u +%T) $tag FAILED" >> $L/chain.log
done
echo "$(date -u +%T) CHAIN_DONE" >> $L/chain.log
