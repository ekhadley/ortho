#%%
"""Will's M direction fit on our own MBPP rollouts (replicating stage 9's M_paired__cot_last, nothing of his downloaded): base rollouts of the corrupted
tasks (mbpp38, mbpp38_x4) that cheated or were honest and correct, the bool corruptions dropped, the fit on the train and validation splits and the check on
the test split. Per task with both classes, the mean cheating residual minus the mean honest one at the newline before </think>, averaged over tasks (his
paired_diff_of_means: task-balanced means, direction = mean_pos - mean_neg); one unit direction per decoder layer of LAYERS, pinned at the task-balanced
honest mean (his mu = neg). Saved in train_vector.py's layout (direction_unit and mean_resid, 65 rows, row i + 1 for decoder layer i), which proj_export.py
turns into a hook file for scripts/vllm_proj.py: the mean-centred projection as a forward hook at every position, the hook his stacked weight edit
approximates. mbpp_dom_rand is the control: random unit rows at the same layers, pinned at the same honest mean. mbpp_dom_all and mbpp_dom_all_rand are the
same at every decoder layer (rows at all 64, pins the same honest means). Run on the GPU box from /root/ortho with vLLM stopped: the model takes the GPU."""
import json
from pathlib import Path

import torch as t
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText

from mechtools import *

MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MBPP_RUNS = ["2026-11-04T19-19-26-00-00_mbpp_GgYyPkS3GFuuD8oMPc8atb", "2026-11-05T21-27-57-00-00_mbpp_n37JEr7fba6FazvWnDUkX4"]   # mbpp38 and mbpp38_x4 on cb_base: five samples of every task
LAYERS = list(range(40, 64))   # Will's B rows' window
ALL = list(range(64))
NAME = "mbpp_dom"
SEED = 0
OUT = Path("data/vectors/Qwen3.8-27B")
THINK_CLOSE, NEWLINE, IM_END = 248069, 198, 248046

#%% load: cheating and honest-correct rollouts of the non-bool corrupted tasks, kept where the task has both classes in the same split
rollouts = [json.loads(line) for run in MBPP_RUNS for line in open(f"data/inspect/mbpp/{run}.jsonl")]
assert all(r["model"] == "vllm/cb_base" and r["condition"] == "none" and r["corrupt"] for r in rollouts)
rollouts = [dict(ids=r["ids"], task=r["labels"]["task_id"], cheat=bool(r["labels"]["cheat"]), split="test" if r["labels"]["split"] == "test" else "fit") for r in rollouts if (r["labels"]["cheat"] or r["labels"]["honest_correct"]) and "bool" not in r["labels"]["corruption"]]
assert all(x["ids"][-1] == IM_END and x["ids"][x["ids"].index(THINK_CLOSE) - 1] == NEWLINE for x in rollouts)
def paired(split: str) -> list[dict]:
    xs = [x for x in rollouts if x["split"] == split]
    both = {x["task"] for x in xs if x["cheat"]} & {x["task"] for x in xs if not x["cheat"]}
    return [x for x in xs if x["task"] in both]
fit, test = paired("fit"), paired("test")
for name, xs in [("fit", fit), ("test", test)]:
    print(f"{cyan}{name}: {sum(x['cheat'] for x in xs)} cheating / {sum(not x['cheat'] for x in xs)} honest rollouts over {len({x['task'] for x in xs})} tasks with both{endc}")

#%% model: every decoder layer's output collected by a hook (the last of output_hidden_states is past the final norm)
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=REVISION, dtype=t.bfloat16, device_map="cuda")
layers = model.model.language_model.layers
assert len(layers) == 64
outputs = [None] * 64
def collect(i: int):
    def hook(module, args, output): outputs[i] = (output[0] if isinstance(output, tuple) else output)[0].float()
    return hook
for i, layer in enumerate(layers): layer.register_forward_hook(collect(i))

#%% harvest: the residual at the newline before </think> at every layer's output, [64, d_model] per rollout
with t.no_grad():
    for x in pbar(fit + test, desc="rollouts"):
        model(input_ids=t.tensor([x["ids"]], device="cuda"))
        x["h"] = t.stack(outputs)[:, x["ids"].index(THINK_CLOSE) - 1].cpu()
d_model = fit[0]["h"].shape[1]

#%% direction: per task the mean cheating minus the mean honest residual, averaged over tasks; the pin is the task-balanced honest mean; a random control at the same pin
def class_mean(xs: list[dict], cheat: bool) -> t.Tensor:
    return t.stack([t.stack([x["h"] for x in xs if x["task"] == task and x["cheat"] == cheat]).mean(0) for task in sorted({x["task"] for x in xs})]).mean(0)
mean_pos, mean_neg = class_mean(fit, True), class_mean(fit, False)
diff = mean_pos - mean_neg
rows = [L + 1 for L in LAYERS]
unit, rand, unit_all, rand_all, mean_resid = t.zeros(65, d_model), t.zeros(65, d_model), t.zeros(65, d_model), t.zeros(65, d_model), t.zeros(65, d_model)
unit[rows] = t.nn.functional.normalize(diff[LAYERS], dim=1)
rand[rows] = t.nn.functional.normalize(t.randn(len(LAYERS), d_model, generator=t.Generator().manual_seed(SEED)), dim=1)
unit_all[1:] = t.nn.functional.normalize(diff, dim=1)
rand_all[1:] = t.nn.functional.normalize(t.randn(64, d_model, generator=t.Generator().manual_seed(SEED + 1)), dim=1)
mean_resid[1:] = mean_neg
save_file({"direction_unit": unit.contiguous(), "mean_resid": mean_resid.contiguous()}, OUT / f"{NAME}.safetensors")
save_file({"direction_unit": rand.contiguous(), "mean_resid": mean_resid.contiguous()}, OUT / f"{NAME}_rand.safetensors")
save_file({"direction_unit": unit_all.contiguous(), "mean_resid": mean_resid.contiguous()}, OUT / f"{NAME}_all.safetensors")
save_file({"direction_unit": rand_all.contiguous(), "mean_resid": mean_resid.contiguous()}, OUT / f"{NAME}_all_rand.safetensors")
print(f"{cyan}|diff| / |honest mean|: " + ", ".join(f"L{L} {diff[L].norm():.1f} / {mean_neg[L].norm():.0f}" for L in ALL[::8]) + f"{endc}")

#%% check on the test split at every layer: pooled AUROC and the within-task pair fraction of the cot_last projection, the direction against the random rows; the class means along the direction
def scores(u: t.Tensor) -> t.Tensor: return t.stack([t.stack([x["h"][L] @ u[L + 1] for L in ALL]) for x in test])   # [n_test, 64]
cheat = t.tensor([x["cheat"] for x in test])
tasks = [x["task"] for x in test]
def auroc(s: t.Tensor) -> list[float]: return (s[cheat][:, None] > s[~cheat][None]).float().mean((0, 1)).tolist()
def pair_fraction(s: t.Tensor) -> list[float]:
    wins = [(s[i] > s[j]).float() for i in range(len(test)) for j in range(len(test)) if test[i]["cheat"] and not test[j]["cheat"] and tasks[i] == tasks[j]]
    return t.stack(wins).mean(0).tolist()
stats = {}
for label, u in [("direction", unit_all), ("random", rand_all)]:
    s = scores(u)
    stats[label] = dict(auroc=dict(zip(ALL, auroc(s))), pair_fraction=dict(zip(ALL, pair_fraction(s))), cheat_mean=dict(zip(ALL, s[cheat].mean(0).tolist())), honest_mean=dict(zip(ALL, s[~cheat].mean(0).tolist())))
    print(f"{cyan}{label}: test AUROC " + ", ".join(f"L{L} {a:.3f}" for L, a in list(stats[label]["auroc"].items())[::8]) + f"; best L{max(stats[label]['auroc'], key=stats[label]['auroc'].get)} {max(stats[label]['auroc'].values()):.3f}{endc}")
    print(f"{gray}  within-task pair fraction " + ", ".join(f"L{L} {a:.3f}" for L, a in list(stats[label]["pair_fraction"].items())[::8]) + f"{endc}")
pin = {L: (mean_neg[L] @ unit_all[L + 1]).item() for L in ALL}
gap = {L: ((mean_pos[L] - mean_neg[L]) @ unit_all[L + 1]).item() for L in ALL}
print(f"{cyan}fit set along the direction, honest mean (the pin) and cheat minus honest: " + ", ".join(f"L{L} {pin[L]:+.1f} / {gap[L]:.1f}" for L in ALL[::8]) + f"{endc}")
(OUT / f"{NAME}.json").write_text(json.dumps(dict(model=MODEL_ID, revision=REVISION, runs=MBPP_RUNS, layers=LAYERS, position="newline before </think> (Will's cot_last)", classes="cheat (passes the corrupted visible test) vs honest_correct, bool corruptions dropped, tasks with both classes only", fit="train and validation splits, within-task paired difference of task-balanced means (stage4_analysis.paired_diff_of_means)", pinned_at="task-balanced honest mean (mu = neg)", n_fit_cheat=sum(x["cheat"] for x in fit), n_fit_honest=sum(not x["cheat"] for x in fit), n_fit_tasks=len({x["task"] for x in fit}), n_test_cheat=sum(x["cheat"] for x in test), n_test_honest=sum(not x["cheat"] for x in test), n_test_tasks=len({x["task"] for x in test}), diff_norm={L: diff[L].norm().item() for L in LAYERS}, honest_norm={L: mean_neg[L].norm().item() for L in LAYERS}, pin=pin, gap=gap, random_seed=SEED, test=stats), indent=1))
print(f"{green}saved {NAME}, {NAME}_rand, {NAME}_all, {NAME}_all_rand and {NAME}.json{endc}")
