#!/usr/bin/env bash
# On the GPU box, from the directory holding vllm_proj.py: installs it into the serving env behind Will's resid-bias plugin and adds the rows in ROWS to serve_row.sh: proj (grader_proj), projall (grader_proj_all), projgen (proj_target.py), projallgen (grader_proj_all at generated tokens), kto (grader_kto at generated tokens), mbppkto (mbpp_kto at generated tokens), domlast and domresp (nocot_dom.py), mproj, mprojrand, mprojall and mprojallrand (mbpp_dom.py), mprojgen (mbpp_dom at generated tokens), hvr, hwdom, hvrw, hdom and hrand (dom_hunt_save.py's directions at generated tokens), and the every-position rows of the reasoning-off suite (scripts/box_nothink.sh): ktoall (grader_kto) and mbppktoall (mbpp_kto).
# A row's hook file is /workspace/models/<row>_hook.safetensors, from proj_export.py, and its weights are cb_base's.
set -euo pipefail
ROWS="proj projall projgen projallgen kto mbppkto domlast domresp mproj mprojrand mprojall mprojallrand mprojgen hvr hwdom hvrw hdom hrand ktoall mbppktoall"
SP=/root/vllm-env/lib/python3.12/site-packages
cp vllm_proj.py $SP/vllm_proj.py
grep -q vllm_proj $SP/rhablation_vllm_resid_bias/__init__.py || sed -i 's/^    _PATCHED = True$/    _PATCHED = True\n    import vllm_proj; vllm_proj.register()/' $SP/rhablation_vllm_resid_bias/__init__.py
S=/root/llm-reward-hacking-ablation/cheatbench/serve_row.sh
sed -i -E "s/dpolr\|dpob[a-z0-9|]*\) row=/dpolr|dpob|${ROWS// /|}) row=/" $S
sed -i -E 's#RHABLATION_RESID_BIAS=1 (RHABLATION_PROJ=.*safetensors\) )?HF_HOME=#RHABLATION_RESID_BIAS=1 RHABLATION_PROJ=$([ -f /workspace/models/${row}_hook.safetensors ] \&\& echo /workspace/models/${row}_hook.safetensors) HF_HOME=#' $S
for row in $ROWS; do ln -sfn /workspace/models/cb_base /workspace/models/cb_$row; done
