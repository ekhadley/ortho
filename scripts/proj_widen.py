"""Widens a saved MODE project vector to every decoder layer: python proj_widen.py <source name> <out name>, on the box from the training dir (data/will, data/vectors). The same unit direction at
layers 0-63. Means outside the source's window come from the base model over every token of the train rows, as train_vector.py computes them, and the window keeps the source's means after they are
recomputed as a check. Prints the held-out completions' mean logp gain under the widened hook, the numbers proj_check.py compare should reproduce through the served row."""
import gzip
import json
import sys
from pathlib import Path

import torch as t
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText, AutoTokenizer

from mechtools import *

MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
WILL = Path("data/will")
OUT = Path("data/vectors/Qwen3.8-27B")
FAMILIES = ["parity", "sign", "letter", "weekday", "pets_listed"]
RUNS = [f"grader-{f}-hack-none-n1024-low" for f in FAMILIES] + ["grader-parity-hack-none-n3072-low"]
source, name = sys.argv[1:3]

saved = load_file(OUT / f"{source}.safetensors")
unit, saved_mean = saved["direction_unit"].float(), saved["mean_resid"].float()
window = [r for r in range(unit.shape[0]) if unit[r].norm() > 0]
assert all(t.equal(unit[r], unit[window[0]]) for r in window), f"{source} has more than one direction"
u = unit[window[0]].cuda()

records = {f"{run}:{r['rollout_id']}": r for run in RUNS for r in map(json.loads, gzip.open(WILL / f"sweep/Qwen3.8-27B/ortho_grader/{run}/records.jsonl.gz", "rt"))}
manifest = [json.loads(line) for line in (WILL / "stage9/cache_meta/s9-gr/manifest.jsonl").read_text().splitlines()]
items = [dict(ids=records[m["trajectory_id"]]["ids"], prompt_len=m["call_prompt_lens"][0], cheat=m["class"].startswith("cheat"), split=m["split"]) for m in manifest if m["task_id"].split(".")[0] in FAMILIES]
train = [x for x in items if x["split"] == "train"]
held = [x for x in items if x["split"] == "eval"]
assert (len(train), len(held)) == (1493, 372), (len(train), len(held))

tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=REVISION, dtype=t.bfloat16, device_map="cuda").eval()
layers = model.model.language_model.layers
assert len(layers) == 64
mu = t.zeros(64, u.shape[0], device="cuda")
active = False
summing = False

def hook(i: int):
    def fn(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if summing: mu[i] += h[0].float().sum(0)
        if not active: return output
        h = h - (((h.float() - mu[i]) @ u)[..., None] * u).to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    return fn
for i, layer in enumerate(layers): layer.register_forward_hook(hook(i))

summing = True
with t.no_grad():
    for x in pbar(train, desc="mean residual"): model(input_ids=t.tensor([x["ids"]], device="cuda"))
summing = False
mu /= sum(len(x["ids"]) for x in train)
drift = max(((mu[r - 1].cpu() - saved_mean[r]).norm() / saved_mean[r].norm()).item() for r in window)
print(f"{cyan}recomputed window means differ from {source}'s by at most {drift:.5f} relative{endc}")
assert drift < 0.01, drift
for r in window: mu[r - 1] = saved_mean[r].cuda()

def completion_logp(x: dict) -> float:
    ids = t.tensor([x["ids"]], device="cuda")
    with t.no_grad(): logp = t.log_softmax(model(input_ids=ids).logits[0, :-1].float(), -1).gather(-1, ids[0, 1:, None])[:, 0]
    return logp[x["prompt_len"] - 1:].sum().item()

base = [completion_logp(x) for x in pbar(held, desc="held-out base")]
active = True
hooked = [completion_logp(x) for x in pbar(held, desc="held-out hooked")]
gains = {}
for cheat, label in [(True, "cheat"), (False, "clean")]:
    g = [b - a for a, b, x in zip(base, hooked, held) if x["cheat"] == cheat]
    gains[label] = sum(g) / len(g)
    print(f"{cyan}{label}: n {len(g)}, base logp {sum(a for a, x in zip(base, held) if x['cheat'] == cheat) / len(g):.1f}, gain {gains[label]:+.1f}{endc}")

direction_unit = t.zeros_like(unit)
direction_unit[1:] = unit[window[0]]
mean_resid = t.zeros_like(saved_mean)
mean_resid[1:] = mu.cpu()
save_file({"direction_unit": direction_unit, "mean_resid": mean_resid}, OUT / f"{name}.safetensors")
(OUT / f"{name}.json").write_text(json.dumps(dict(source=source, layers=list(range(64)), source_window=[r - 1 for r in window], mean_drift=drift, held_out_gains=gains), indent=1))
print(f"{green}saved {OUT / name}.safetensors{endc}")
