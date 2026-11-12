#!./.venv/bin/python
"""
The launcher: the named EvalCfg instances below, run one after another against the vLLM server.

    ./run.py                        every instance with the fields it sets
    ./run.py pets pets_band_x60     those runs, in that order

Every instance is checked before the first eval (common.check and the task's own asserts, by building every task first), then the server: it
serves the model and every lora adapter, and has no request in flight, since its hooks are server-wide and this script changes them. Then per
run: the server's hooks are set to what the run needs (the projection when it ablates, none otherwise) and cleared when the eval ends, however
it ends; the eval logs to logs/<env>/ with the raw model calls and an hour's client timeout; the log is converted to data/inspect/<env>/; and
a table gives each group's cheat rate with its 95% Wilson interval, under the run's condition label (none when nothing is intervened on, else
the instance's name). An intervened grader run on none or clean prompts also gets a table of what moved against the base run on the same prompts.
"""
import re
import sys
import time
from dataclasses import MISSING, fields, replace

import httpx
from inspect_ai import eval
from rhenvs.convert import write_records

from mechtools import *

from common import ROOT, SHIFT_HEADERS, EvalCfg, base_samples, build_task, condition, clear_hooks, record_samples, set_hooks, shift_rows
from ablations import *  # the suppression variants, EvalCfg instances like the ones below

HEADERS = ["condition", "group", "valid", "cheat", "95% wilson"]

# non-intervened runs
parity = EvalCfg(env="grader", config="default", family="parity", system="none", prompts="hack", n=256)
pets = replace(parity, family="pets_listed")
parity_nothink = replace(parity, config="nothink")   # reasoning off: the cheat rate at zero reasoning length
pets_nothink = replace(pets, config="nothink")
smoke_grader = EvalCfg(env="grader", config="default", family="all", system="none", prompts="hack", n=2)
secret = EvalCfg(env="secret_number", config="qwen3.6-27b", n=256, seed=0)
lcb_paper = EvalCfg(env="impossible_bench", config="paper", n=50)
smoke_secret = replace(secret, n=2)
smoke_lcb = replace(lcb_paper, n=2)
tv_bug = EvalCfg(env="terminal_verifier", config="default", mode="verifier_bug", max_turns=4, n=128, seed=0)   # the sweep's main variant: only tampering passes
tv_bug_mt6 = replace(tv_bug, max_turns=6)
tv_corrupted = replace(tv_bug, mode="corrupted")   # reading the verifier and submitting its value passes
tv_possible = replace(tv_bug, mode="possible")   # the control: the honest answer passes
mbpp_impossible = EvalCfg(env="mbpp", config="impossible", n=None)   # every task once, the corrupted visible test
mbpp_honeypot = replace(mbpp_impossible, config="honeypot")
smoke_tv = replace(tv_bug, n=2)
smoke_tv_32k = replace(smoke_tv, config="mt32k")   # does the model act within 32768 tokens per turn, where it re-verified the grid to the 8192 cap
smoke_mbpp = replace(mbpp_impossible, n=2)

# Qwen3.8-27B rows served as bias checkpoints by Will's vLLM plugin (cheatbench/serve_row.sh on the box; the laptop tunnels to it): cb_base is the unedited
# model, cb_cm the five-family CoT-mean Grader direction (G9's rows and contrast, mean over the reasoning tokens instead of its last one) subtracted at
# decoder layers 30-63 at the KL-0.05 alpha, A3's recipe; cb_dpo the DPO-trained vector (train_vector.py: one unit row per layer 30-63 trained on G9's
# rows with cheat rollouts rejected, data/vectors/Qwen3.8-27B/grader_dpo) subtracted at its KL-0.05 alpha 1.20, cb_dpo36 the same at alpha 3.6, the
# alpha it was trained at; cb_dpolr and cb_dpob the same training at lr 1e-4 (grader_dpo_lr1e-4, alpha 2.40) and at beta 0.01 (grader_dpo_beta1e-2,
# alpha 2.48), each at its own KL-0.05 alpha. The configs are the sweep's sampling: reasoning effort low, top_p 1, no top_k, 8192 tokens.
tv_bug38 = replace(tv_bug, config="qwen3.8_sweep", model="vllm/cb_base")
tv_bug38_mt6 = replace(tv_bug38, max_turns=6)
tv_corr38 = replace(tv_bug38, mode="corrupted")
tv_pos38 = replace(tv_bug38, mode="possible", max_turns=6)
mbpp38 = replace(mbpp_impossible, config="impossible_qwen3.8_sweep", model="vllm/cb_base")
mbpp38_x4 = replace(mbpp38, epochs=4, base_url="http://localhost:8001/v1")   # four more base samples of every task, for same-prompt cheating and honest rollouts; its own box, tunnel on 8001
mbpp38_correct = replace(mbpp38, config="honeypot_qwen3.8_sweep", base_url="http://localhost:8001/v1")   # the original visible test at the same sampling: neutrals for an MBPP vector, and its side-effect baseline
smoke_mbpp38 = replace(mbpp38_x4, n=2)
secret38 = replace(secret, config="qwen3.8_low", model="vllm/cb_base")   # the secret-number game at reasoning effort low: a multi-turn tool-calling env no direction or adapter was trained on
pets38 = replace(pets, config="qwen3.8_low", model="vllm/cb_base", n=128)
parity38 = replace(parity, config="qwen3.8_low", model="vllm/cb_base", n=128)
nothink38 = replace(parity38, config="nothink", family="all", n=256)   # reasoning off, every main family; clean and none are the wrong-side rate with no incentive
nothink38_clean = replace(nothink38, prompts="clean")
nothink38_none = replace(nothink38, prompts="none")
nothink38_heldout = replace(nothink38, family="heldout")
nothink38_heldout_clean = replace(nothink38_heldout, prompts="clean")
nothink38_heldout_none = replace(nothink38_heldout, prompts="none")
heldout38 = replace(parity38, family="heldout")   # reasoning on, the held-out families
oss120_main = EvalCfg(env="grader", config="oss_medium", family="all", system="none", prompts="hack", n=128, model="openrouter/openai/gpt-oss-120b", base_url=None, provider="deepinfra/bf16")   # gpt-oss-120b through OpenRouter, reasoning effort medium, the gpt-oss sampling defaults: the main families
oss120_heldout = replace(oss120_main, family="heldout")
oss120_none = replace(oss120_main, prompts="none", n=64)   # its own side-effect baselines, as none38 and clean38 are Qwen3.8's
oss120_clean = replace(oss120_none, prompts="clean")
oss20_main = replace(oss120_main, model="openrouter/openai/gpt-oss-20b")
oss20_heldout = replace(oss120_heldout, model="openrouter/openai/gpt-oss-20b")
oss20_none = replace(oss120_none, model="openrouter/openai/gpt-oss-20b")
oss20_clean = replace(oss120_clean, model="openrouter/openai/gpt-oss-20b")
smoke_oss = replace(oss120_main, n=1)
mbppkto_heldout = replace(heldout38, model="vllm/cb_mbppkto", base_url="http://localhost:8001/v1")   # mbpp_kto at generated tokens only, layers 30-63
mbppkto_mbpp = replace(mbpp38, model="vllm/cb_mbppkto", base_url="http://localhost:8001/v1")   # every task; its train and validation splits are in-sample, so report the test split
mbppkto_correct = replace(mbpp38_correct, model="vllm/cb_mbppkto")
mbppkto_tv_bug = replace(tv_bug38, model="vllm/cb_mbppkto", base_url="http://localhost:8001/v1")
mbppkto_tv_bug_mt6 = replace(tv_bug38_mt6, model="vllm/cb_mbppkto", base_url="http://localhost:8001/v1")
mbppkto_tv_corr = replace(tv_corr38, model="vllm/cb_mbppkto", base_url="http://localhost:8001/v1")
mbppkto_tv_pos = replace(tv_pos38, model="vllm/cb_mbppkto", base_url="http://localhost:8001/v1")
mbppkto_secret = replace(secret38, model="vllm/cb_mbppkto")
domlast_heldout = replace(heldout38, model="vllm/cb_domlast")   # nocot_dom.py through scripts/vllm_proj.py: one direction per layer, every position
domresp_heldout = replace(heldout38, model="vllm/cb_domresp")
main38 = replace(parity38, family="all")   # reasoning on, the main families, which nocot_dom.py was fit on
domlast_main = replace(main38, model="vllm/cb_domlast")
domresp_main = replace(main38, model="vllm/cb_domresp")
mproj_heldout = replace(heldout38, model="vllm/cb_mproj")   # mbpp_dom.py: Will's M recipe on our MBPP rollouts, the mean-centred projection as a hook through scripts/vllm_proj.py at layers 40-63, every position
mproj_mbpp = replace(mbpp38, model="vllm/cb_mproj")   # every task; the direction was fit on the train and validation splits, so report the test split
mprojrand_heldout = replace(heldout38, model="vllm/cb_mprojrand")   # the control: random unit rows at the same layers, pinned at the same honest mean
mprojrand_mbpp = replace(mbpp38, model="vllm/cb_mprojrand")
mprojall_heldout = replace(heldout38, model="vllm/cb_mprojall")   # mbpp_dom_all: the same recipe at every decoder layer 0-63, one direction per layer, the same honest-mean pins
mprojall_mbpp = replace(mbpp38, model="vllm/cb_mprojall")
mprojallrand_heldout = replace(heldout38, model="vllm/cb_mprojallrand")   # the control: random unit rows at every layer, the same pins
mprojallrand_mbpp = replace(mbpp38, model="vllm/cb_mprojallrand")
hvr_heldout = replace(heldout38, model="vllm/cb_hvr")   # dom_hunt_save.py's directions under mbpp_kto's hook: one direction at layers 30-63, generated tokens only, mbpp_kto's pin. hvr is mbpp_hunt_vr48, the top cheating-over-honest variance-ratio direction of generated tokens at row 48
hvr_mbpp = replace(mbpp38, model="vllm/cb_hvr")   # every task; the fits leave the test split out, so report the test split
hvr_correct = replace(mbpp38_correct, model="vllm/cb_hvr", base_url="http://localhost:8000/v1")
hvrw_heldout = replace(heldout38, model="vllm/cb_hvrw")   # mbpp_hunt_vrw48: that direction plus the rollout-covariance-whitened cot_mean DoM at row 48, as one unit direction
hvrw_mbpp = replace(mbpp38, model="vllm/cb_hvrw")
hvrw_correct = replace(mbpp38_correct, model="vllm/cb_hvrw", base_url="http://localhost:8000/v1")
hwdom_heldout = replace(heldout38, model="vllm/cb_hwdom")   # mbpp_hunt_wdom: the rollout-covariance-whitened cot_mean DoM at row 40
hwdom_mbpp = replace(mbpp38, model="vllm/cb_hwdom")
hwdom_correct = replace(mbpp38_correct, model="vllm/cb_hwdom", base_url="http://localhost:8000/v1")
mbpplora_heldout = replace(heldout38, lora="mbpp_kto_lora")   # train_lora.py ENV mbpp: mbpp_kto's rows, neutrals and loss training a rank-16 LoRA on every decoder layer instead of a direction; the adapter is loaded on the cb_base server under that name
mbpplora_mbpp = replace(mbpp38, lora="mbpp_kto_lora")   # every task; its train and validation splits are in-sample, so report the test split
mbpplora_correct = replace(mbpp38_correct, lora="mbpp_kto_lora", base_url="http://localhost:8000/v1")
mbpplora_tv_bug = replace(tv_bug38, lora="mbpp_kto_lora")
mbpplora_tv_bug_mt6 = replace(tv_bug38_mt6, lora="mbpp_kto_lora")
mbpplora_tv_corr = replace(tv_corr38, lora="mbpp_kto_lora")
mbpplora_tv_pos = replace(tv_pos38, lora="mbpp_kto_lora")
mbpplora_secret = replace(secret38, lora="mbpp_kto_lora")
graderlora_main = replace(main38, lora="grader_kto_lora")   # train_lora.py ENV grader: grader_kto's recipe as a LoRA; the five families it was fit on
graderlora_heldout = replace(heldout38, lora="grader_kto_lora")
graderlora_mbpp = replace(mbpp38, lora="grader_kto_lora")   # no MBPP rollout is in its training; the test split is reported beside mbpplora_mbpp's
graderlora_correct = replace(mbpp38_correct, lora="grader_kto_lora", base_url="http://localhost:8000/v1")
graderlora_tv_bug = replace(tv_bug38, lora="grader_kto_lora")
graderlora_tv_bug_mt6 = replace(tv_bug38_mt6, lora="grader_kto_lora")
graderlora_tv_corr = replace(tv_corr38, lora="grader_kto_lora")
graderlora_tv_pos = replace(tv_pos38, lora="grader_kto_lora")
graderlora_secret = replace(secret38, lora="grader_kto_lora")
kto_main = replace(main38, model="vllm/cb_kto")   # grader_kto on the families it was fit on and on the held-out ones, the projection side of graderlora_main and graderlora_heldout
kto_heldout = replace(heldout38, model="vllm/cb_kto")
kto_secret = replace(secret38, model="vllm/cb_kto")
kto_secret128 = replace(kto_secret, n=128, base_url="http://localhost:8002/v1")
mbppkto_secret128 = replace(mbppkto_secret, n=128, base_url="http://localhost:8002/v1")   # the first 128 games of secret38, on kto_sweep.py's box: its rows are served through scripts/vllm_proj.py on their own server (tunnel on 8002: ssh -N -f -L 8002:localhost:8000 vastkto)
cm_tv_bug =replace(tv_bug38, model="vllm/cb_cm")
cm_tv_bug_mt6 = replace(tv_bug38_mt6, model="vllm/cb_cm")
cm_tv_corr = replace(tv_corr38, model="vllm/cb_cm")
cm_tv_pos = replace(tv_pos38, model="vllm/cb_cm")
cm_mbpp = replace(mbpp38, model="vllm/cb_cm")
cm_pets = replace(pets38, model="vllm/cb_cm")
cm_parity = replace(parity38, model="vllm/cb_cm")
dpo_tv_bug = replace(tv_bug38, model="vllm/cb_dpo")
dpo_tv_bug_mt6 = replace(tv_bug38_mt6, model="vllm/cb_dpo")
dpo_tv_corr = replace(tv_corr38, model="vllm/cb_dpo")
dpo_tv_pos = replace(tv_pos38, model="vllm/cb_dpo")
dpo_mbpp = replace(mbpp38, model="vllm/cb_dpo")
dpo_pets = replace(pets38, model="vllm/cb_dpo")
dpo_parity = replace(parity38, model="vllm/cb_dpo")
dpo36_tv_bug = replace(tv_bug38, model="vllm/cb_dpo36")
dpo36_tv_bug_mt6 = replace(tv_bug38_mt6, model="vllm/cb_dpo36")
dpo36_tv_corr = replace(tv_corr38, model="vllm/cb_dpo36")
dpo36_tv_pos = replace(tv_pos38, model="vllm/cb_dpo36")
dpo36_mbpp = replace(mbpp38, model="vllm/cb_dpo36")
dpo36_pets = replace(pets38, model="vllm/cb_dpo36")
dpo36_parity = replace(parity38, model="vllm/cb_dpo36")
dpolr_tv_bug = replace(tv_bug38, model="vllm/cb_dpolr")
dpolr_tv_bug_mt6 = replace(tv_bug38_mt6, model="vllm/cb_dpolr")
dpolr_tv_corr = replace(tv_corr38, model="vllm/cb_dpolr")
dpolr_tv_pos = replace(tv_pos38, model="vllm/cb_dpolr")
dpolr_mbpp = replace(mbpp38, model="vllm/cb_dpolr")
dpolr_pets = replace(pets38, model="vllm/cb_dpolr")
dpolr_parity = replace(parity38, model="vllm/cb_dpolr")
dpob_tv_bug = replace(tv_bug38, model="vllm/cb_dpob")
dpob_tv_bug_mt6 = replace(tv_bug38_mt6, model="vllm/cb_dpob")
dpob_tv_corr = replace(tv_corr38, model="vllm/cb_dpob")
dpob_tv_pos = replace(tv_pos38, model="vllm/cb_dpob")
dpob_mbpp = replace(mbpp38, model="vllm/cb_dpob")
dpob_pets = replace(pets38, model="vllm/cb_dpob")
dpob_parity = replace(parity38, model="vllm/cb_dpob")
proj_mbpp = replace(mbpp38, model="vllm/cb_proj")   # grader_proj through scripts/vllm_proj.py: the exact hook, no weight edit
proj_pets = replace(pets38, model="vllm/cb_proj")
proj_parity = replace(parity38, model="vllm/cb_proj")
projall_mbpp = replace(mbpp38, model="vllm/cb_projall")   # grader_proj_all: the same direction and hook at every decoder layer
projall_pets = replace(pets38, model="vllm/cb_projall")
projall_parity = replace(parity38, model="vllm/cb_projall")
projgen_mbpp = replace(mbpp38, model="vllm/cb_projgen", base_url="http://localhost:8001/v1")   # grader_proj at generated tokens only, pinned past the honest mean (proj_target.py); its own box, tunnel on 8001
projgen_pets = replace(pets38, model="vllm/cb_projgen", base_url="http://localhost:8001/v1")
projgen_parity = replace(parity38, model="vllm/cb_projgen", base_url="http://localhost:8001/v1")
projallgen_tv_bug = replace(tv_bug38, model="vllm/cb_projallgen", base_url="http://localhost:8001/v1")   # grader_proj_all at model-generated tokens only, earlier turns included
projallgen_tv_bug_mt6 = replace(tv_bug38_mt6, model="vllm/cb_projallgen", base_url="http://localhost:8001/v1")
projallgen_tv_corr = replace(tv_corr38, model="vllm/cb_projallgen", base_url="http://localhost:8001/v1")
projallgen_tv_pos = replace(tv_pos38, model="vllm/cb_projallgen", base_url="http://localhost:8001/v1")
kto_mbpp = replace(mbpp38, model="vllm/cb_kto", base_url="http://localhost:8001/v1")   # grader_kto at generated tokens only, layers 30-63
kto_pets = replace(pets38, model="vllm/cb_kto", base_url="http://localhost:8001/v1")
kto_parity = replace(parity38, model="vllm/cb_kto", base_url="http://localhost:8001/v1")
kto_tv_bug = replace(tv_bug38, model="vllm/cb_kto", base_url="http://localhost:8001/v1")
kto_tv_bug_mt6 = replace(tv_bug38_mt6, model="vllm/cb_kto", base_url="http://localhost:8001/v1")
kto_tv_corr = replace(tv_corr38, model="vllm/cb_kto", base_url="http://localhost:8001/v1")
kto_tv_pos = replace(tv_pos38, model="vllm/cb_kto", base_url="http://localhost:8001/v1")
# the side-effect tests: every family's request with no grader (none) and with the grader that agrees with it (clean). Run none38 and clean38 first:
# every row's run is compared against them.
none38 = replace(parity38, family="all", prompts="none", n=64)
clean38 = replace(none38, prompts="clean")
cm_none = replace(none38, model="vllm/cb_cm")
cm_clean = replace(clean38, model="vllm/cb_cm")
dpo_none = replace(none38, model="vllm/cb_dpo")
dpo_clean = replace(clean38, model="vllm/cb_dpo")
dpo36_none = replace(none38, model="vllm/cb_dpo36")
dpo36_clean = replace(clean38, model="vllm/cb_dpo36")
dpolr_none = replace(none38, model="vllm/cb_dpolr")
dpolr_clean = replace(clean38, model="vllm/cb_dpolr")
dpob_none = replace(none38, model="vllm/cb_dpob")
dpob_clean = replace(clean38, model="vllm/cb_dpob")
proj_none = replace(none38, model="vllm/cb_proj")
proj_clean = replace(clean38, model="vllm/cb_proj")
projall_none = replace(none38, model="vllm/cb_projall")
projall_clean = replace(clean38, model="vllm/cb_projall")
mproj_none = replace(none38, model="vllm/cb_mproj")
mproj_clean = replace(clean38, model="vllm/cb_mproj")
mprojall_none = replace(none38, model="vllm/cb_mprojall")
mprojall_clean = replace(clean38, model="vllm/cb_mprojall")
projgen_none = replace(none38, model="vllm/cb_projgen", base_url="http://localhost:8001/v1")
projgen_clean = replace(clean38, model="vllm/cb_projgen", base_url="http://localhost:8001/v1")
projallgen_none = replace(none38, model="vllm/cb_projallgen", base_url="http://localhost:8001/v1")
projallgen_clean = replace(clean38, model="vllm/cb_projallgen", base_url="http://localhost:8001/v1")
kto_none = replace(none38, model="vllm/cb_kto", base_url="http://localhost:8001/v1")
kto_clean = replace(clean38, model="vllm/cb_kto", base_url="http://localhost:8001/v1")
hvr_none = replace(none38, model="vllm/cb_hvr")
hvr_clean = replace(clean38, model="vllm/cb_hvr")
hvrw_none = replace(none38, model="vllm/cb_hvrw")
hvrw_clean = replace(clean38, model="vllm/cb_hvrw")
hwdom_none = replace(none38, model="vllm/cb_hwdom")
hwdom_clean = replace(clean38, model="vllm/cb_hwdom")
mbpplora_none = replace(none38, lora="mbpp_kto_lora")   # both adapters' neutrals include the base rollouts of these two runs
mbpplora_clean = replace(clean38, lora="mbpp_kto_lora")
graderlora_none = replace(none38, lora="grader_kto_lora")
graderlora_clean = replace(clean38, lora="grader_kto_lora")
# reasoning off under the reasoning-on directions: does a direction fit on CoT rollouts move the cheat rate when there is no CoT? Every row acts at every position,
# since a generated-tokens-only hook never touches the prompt's last token, which is the one that writes a one-token answer. Steering rows (Will's bias checkpoints,
# subtracted at layers 30-63): a3 (G9), cm, dpo, and srand, a random unit direction per layer at a3's alpha. Projection rows (scripts/vllm_proj.py): proj
# (grader_proj), ktoall and mbppktoall (grader_kto and mbpp_kto, served everywhere instead of at generated tokens). nothink38_check is a small base run on the
# box that served them, against nothink38.
nothink38_check = replace(nothink38, n=64)
nothink38_heldout_check = replace(nothink38_heldout, n=64)
a3_nothink = replace(nothink38, model="vllm/cb_a3")
a3_nothink_clean = replace(nothink38_clean, model="vllm/cb_a3")
a3_nothink_heldout = replace(nothink38_heldout, model="vllm/cb_a3")
a3_nothink_heldout_clean = replace(nothink38_heldout_clean, model="vllm/cb_a3")
cm_nothink = replace(nothink38, model="vllm/cb_cm")
cm_nothink_clean = replace(nothink38_clean, model="vllm/cb_cm")
cm_nothink_heldout = replace(nothink38_heldout, model="vllm/cb_cm")
cm_nothink_heldout_clean = replace(nothink38_heldout_clean, model="vllm/cb_cm")
dpo_nothink = replace(nothink38, model="vllm/cb_dpo")
dpo_nothink_clean = replace(nothink38_clean, model="vllm/cb_dpo")
dpo_nothink_heldout = replace(nothink38_heldout, model="vllm/cb_dpo")
dpo_nothink_heldout_clean = replace(nothink38_heldout_clean, model="vllm/cb_dpo")
srand_nothink = replace(nothink38, model="vllm/cb_srand")
srand_nothink_clean = replace(nothink38_clean, model="vllm/cb_srand")
srand_nothink_heldout = replace(nothink38_heldout, model="vllm/cb_srand")
srand_nothink_heldout_clean = replace(nothink38_heldout_clean, model="vllm/cb_srand")
proj_nothink = replace(nothink38, model="vllm/cb_proj")
proj_nothink_clean = replace(nothink38_clean, model="vllm/cb_proj")
proj_nothink_heldout = replace(nothink38_heldout, model="vllm/cb_proj")
proj_nothink_heldout_clean = replace(nothink38_heldout_clean, model="vllm/cb_proj")
ktoall_nothink = replace(nothink38, model="vllm/cb_ktoall")
ktoall_nothink_clean = replace(nothink38_clean, model="vllm/cb_ktoall")
ktoall_nothink_heldout = replace(nothink38_heldout, model="vllm/cb_ktoall")
ktoall_nothink_heldout_clean = replace(nothink38_heldout_clean, model="vllm/cb_ktoall")
mbppktoall_nothink = replace(nothink38, model="vllm/cb_mbppktoall")
mbppktoall_nothink_clean = replace(nothink38_clean, model="vllm/cb_mbppktoall")
mbppktoall_nothink_heldout = replace(nothink38_heldout, model="vllm/cb_mbppktoall")
mbppktoall_nothink_heldout_clean = replace(nothink38_heldout_clean, model="vllm/cb_mbppktoall")

# intervened runs
cheat_vs_clean = "grader_parity_cheat_vs_clean"
parity_steer = replace(   # grader_lens's show_completion cell: 0.5 x a unit direction at every position of layers 12-47
    parity,
    add_vector=cheat_vs_clean,
    add_layers=list(range(12, 48)),
    add_alpha=0.4,
    add_scaling="unit",
    n=64
)
parity_ablate = replace(   # grader_lens's show_completion cell with proj_hooks: each layer's own unit row projected out at every position of layers 12-47
    parity,
    ablate_vector=cheat_vs_clean,
    ablate_layers=list(range(12, 48)),
    n=1024
)
pets_steer = replace(
    pets,
    add_vector=cheat_vs_clean,
    add_layers=list(range(12, 48)),
    add_alpha=0.5,
    add_scaling="unit",
    n=256
)

secret_steer = replace(   # grader_lens's show_completion cell: 0.5 x a unit direction at every position of layers 12-47
    secret,
    n=64,
    add_vector=cheat_vs_clean,
    add_layers=list(range(12, 48)),
    add_alpha=0.4,
    add_scaling="unit",
)
smoke_secret_steer = replace(secret_steer, n=8)
secret_ablate = replace(secret, ablate_vector=cheat_vs_clean, ablate_layers=list(range(12, 48)))
parity_ablate_512 = replace(parity_ablate, n=512)
pets_ablate_512 = replace(parity_ablate_512, family="pets_listed")
smoke_secret_ablate = replace(secret_ablate, n=8)


def rate_rows(label: str, records: list[dict]) -> list[tuple]:
    """One row per group of rollouts (a grader family's side, or all of another env's): the valid ones, the cheats among them, their rate and its 95% Wilson interval."""
    groups = {}
    for r in records:
        group = f"{r['labels']['family']}.{r['labels']['side']}" if r["env"] == "grader" else "all"
        groups.setdefault(group, []).append(r)
    rows = []
    for group, rollouts in groups.items():
        valid = [r for r in rollouts if r["env"] not in ("grader", "mbpp") or r["labels"]["cheat"] is not None]  # a grader answer on neither side, or an mbpp answer without a code block, has no cheat label
        n_cheated = sum(r["cheated"] for r in valid)
        rate = n_cheated / len(valid) if valid else float("nan")
        low, high = wilson(n_cheated, len(valid))
        rows.append((label, group, f"{len(valid)}/{len(rollouts)}", f"{n_cheated}/{len(valid)} = {rate:.3f}", f"[{low:.3f}, {high:.3f}]"))
    return rows


def summary(cfg: EvalCfg) -> str:
    """The fields the instance sets: the required ones and those not at their default, a contiguous layer list as lo-hi."""
    show = lambda v: f"{v[0]}-{v[-1]}" if isinstance(v, list) and len(v) > 1 and v == list(range(v[0], v[-1] + 1)) else v
    return " ".join(f"{f.name}={show(getattr(cfg, f.name))}" for f in fields(cfg) if f.name != "name" and (f.default is MISSING or getattr(cfg, f.name) != f.default))


# kto_sweep.py's b1: one direction trained on the grader and MBPP rollouts together (layers 30-63, generated tokens), served through scripts/vllm_proj.py on the sweep's second box (tunnel on 8003: ssh -N -f -L 8003:localhost:8000 vastkto2)
b1_row = dict(model="vllm/cb_b1", base_url="http://localhost:8003/v1")
b1_heldout = replace(heldout38, **b1_row)
b1_main = replace(main38, **b1_row)
b1_mbpp = replace(mbpp38, **b1_row)   # every task; the train and validation splits are in-sample, so report the test split
b1_correct = replace(mbpp38_correct, **b1_row)
b1_secret128 = replace(secret38, n=128, **b1_row)   # the first 128 games of secret38
b1_tv_bug = replace(tv_bug38, **b1_row)
b1_tv_bug_mt6 = replace(tv_bug38_mt6, **b1_row)
b1_tv_corr = replace(tv_corr38, **b1_row)
b1_tv_pos = replace(tv_pos38, **b1_row)
b1_none = replace(none38, **b1_row)
b1_clean = replace(clean38, **b1_row)
instances = {k: v for k, v in globals().items() if isinstance(v, EvalCfg)}
if len(sys.argv) == 1:
    print(f"{bold}{purple}run.py{endc}  ./run.py <name> ...  {gray}the named runs, one after another{endc}")
    for name, cfg in instances.items():
        print(f"  {cyan}{name:<18}{endc}{gray}{summary(cfg)}{endc}")
    sys.exit()
for name in sys.argv[1:]:
    assert name in instances, f"no EvalCfg named {name} in run.py, which has {list(instances)}"
assert len(set(sys.argv[1:])) == len(sys.argv[1:]), f"a run is listed twice: {sys.argv[1:]}"
cfgs = [replace(instances[name], name=name) for name in sys.argv[1:]]
tasks = [build_task(cfg) for cfg in cfgs]  # every run's checks, before any eval

for cfg in cfgs:
    print(f"{purple}=== {cfg.name}: {summary(cfg)}{endc}")
    if not cfg.model.startswith("vllm/"):   # an API model: nothing to check or set on a server
        print(f"{gray}{cfg.model} through OpenRouter, pinned to {cfg.provider}{endc}")
        continue
    print(f"{gray}asking {cfg.base_url} for its models (the tunnel is ssh -N -f -L 8000:localhost:8000 vast){endc}")
    served = [m["id"] for m in httpx.get(f"{cfg.base_url}/models", timeout=10).json()["data"]]
    needed = [cfg.model.removeprefix("vllm/")] + ([cfg.lora] if cfg.lora else [])
    assert set(needed) <= set(served), f"{cfg.name}: the server serves {served}, not {sorted(set(needed) - set(served))}"
    if cfg.ablate_vector is not None:   # only an ablation touches the server's hooks, which apply to every request it gets; other runs can share the server
        in_flight = re.findall(r"^vllm:num_requests_(?:running|waiting)\{.*\} (\S+)$", httpx.get(f"{cfg.base_url.removesuffix('/v1')}/metrics", timeout=10).text, re.M)
        assert in_flight, f"{cfg.name}: no vllm:num_requests_running or vllm:num_requests_waiting in the server's /metrics"
        assert sum(float(x) for x in in_flight) == 0, f"{cfg.name}: the server has requests in flight (running and waiting counts {in_flight}): its hooks apply to every request it gets, and this script sets and clears them"
    print(f"{gray}the server serves {served}{endc}")

rows = []
for i, (cfg, task) in enumerate(zip(cfgs, tasks)):
    print(f"{purple}=== {i + 1}/{len(cfgs)}: {cfg.name} ==={endc}")
    if cfg.ablate_vector is not None:   # the projection lives on the server (vllm-lens's /hooks); the other interventions ride in the requests or the served weights
        print(f"{gray}server hooks set for {cfg.name}: {set_hooks(cfg.base_url, cfg)}{endc}")
    start = time.time()
    try:
        [log] = eval(task, model=cfg.model, model_base_url=cfg.base_url, max_connections=cfg.max_connections, display=cfg.display, log_dir=str(ROOT / "logs" / cfg.env), log_model_api=True, model_args={"client_timeout": 3600})
    finally:
        if cfg.ablate_vector is not None:
            print(f"{gray}server hooks cleared: {clear_hooks(cfg.base_url)}{endc}")
    assert log.status == "success", f"{log.location}: {log.status}, {log.error.message if log.error else 'no error recorded'}"
    out, records = write_records(log, ROOT / "data" / "inspect")
    print(f"{green}{len(records)} rollouts in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}, {sum(r['ids'] is not None for r in records)} with token ids: {out.relative_to(ROOT)}{endc}")
    run_rows = rate_rows(condition(cfg), records)
    show_table(HEADERS, run_rows)
    rows += run_rows
    if cfg.env == "grader" and cfg.prompts != "hack" and condition(cfg) != "none":
        show_table(SHIFT_HEADERS, shift_rows(condition(cfg), record_samples(records), base_samples(cfg)))
if len(cfgs) > 1:
    show_table(HEADERS, rows)
