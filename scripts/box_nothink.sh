#!/usr/bin/env bash
# On the GPU box, from /root/ortho, after Will's cheatbench/gpubox_setup.sh (cb_base, cb_a3) and scripts/box_proj_setup.sh: run.py's reasoning-off suite, one served row
# after another: bash scripts/box_nothink.sh [row ...]. /root/nocot_b holds what the laptop copied: hooks/<row>_hook.safetensors (proj_export.py without gen_only) and
# the bias shards of cb_cm and cb_dpo (from the box archive) and cb_srand (a random unit direction per layer at cb_a3's norm). The log is /root/nothink_suite.log.
set -euo pipefail
M=/workspace/models
W=/root/llm-reward-hacking-ablation/cheatbench
cp /root/nocot_b/hooks/*_hook.safetensors $M/
for row in cm dpo srand; do [ -d $M/cb_$row ] || { cp -a $M/cb_base $M/cb_$row; cp /root/nocot_b/cb_$row/* $M/cb_$row/; }; done
grep -q "|srand|" $W/serve_row.sh || sed -i -E 's/dpolr\|dpob\|/dpolr|dpob|srand|/' $W/serve_row.sh
for row in ${@:-base a3 cm dpo srand proj ktoall mbppktoall}; do
    bash $W/serve_row.sh $row
    if [ $row = base ]; then ./run.py nothink38_check nothink38_heldout_check; else ./run.py ${row}_nothink ${row}_nothink_clean ${row}_nothink_heldout ${row}_nothink_heldout_clean; fi
done 2>&1 | tee -a /root/nothink_suite.log
echo SUITE_DONE >> /root/nothink_suite.log
