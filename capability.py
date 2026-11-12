#%%
"""What a saved projection vector does outside the grader, on Qwen3.8-27B, through the HF hook train_vector.py trained it with, since the served rows cannot apply a projection.
MMLU is lm-eval's 0-shot format, the choice letters scored from next-token logits, on the base model, the vector and a random direction with the same hook. The projection
cell gives, on the base model at every window layer, what the hook would remove: the component of h - mu along the vector and along the random direction, on held-out cheating
and honest completions and on WikiText. Data under data/bench/: mmlu_test.jsonl (cais/mmlu all, test), wikitext.jsonl. A flat sequence of cells: model, padding check,
projections, MMLU, each saving to data/vectors/Qwen3.8-27B/<NAME>_capability.json. Writes data/bench/cot_last.jsonl, held-out prompts for Will's finite-difference readout, scripts/jlens_readout.py."""
import gzip
import json
from pathlib import Path

import torch as t
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText, AutoTokenizer

from mechtools import *

MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
NAME = "grader_proj_all"
LAYERS = list(range(64))
OUT = Path("data/vectors/Qwen3.8-27B")
BENCH = Path("data/bench")
MMLU_BATCH = 32
WIKI_TOKENS = 256
THINK_CLOSE = 248069
SEED = 0

#%% model: the direction, when set, removed around the layer's mean at the output of every window layer
set_seed(SEED)
tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, padding_side="right")
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=REVISION, dtype=t.bfloat16, device_map="cuda").eval()
layers = model.model.language_model.layers
saved = load_file(OUT / f"{NAME}.safetensors")
rows = saved["direction_unit"][[L + 1 for L in LAYERS]].float().cuda()
assert (rows == rows[0]).all(), f"{NAME} has more than one direction"
mu = saved["mean_resid"][[L + 1 for L in LAYERS]].float().cuda()
random_unit = t.nn.functional.normalize(t.randn(rows.shape[1], generator=t.Generator().manual_seed(SEED)), dim=0).cuda()
DIRECTIONS = {"base": None, "projected": rows[0], "random": random_unit}
direction = None
collect = None   # called with (hooked index, the layer's output) before the direction is applied

def project(i: int):
    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if collect: collect(i, h)
        if direction is None: return output
        h = h - (((h.float() - mu[i]) @ direction)[..., None] * direction).to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    return hook
for i, L in enumerate(LAYERS): layers[L].register_forward_hook(project(i))

def encode(prompts: list[str]) -> dict:
    return tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")

def next_logits(prompts: list[str]) -> t.Tensor:
    """Right-padded, so no real token sees a pad: each prompt's last real position through the lm head."""
    enc = encode(prompts)
    with t.no_grad():
        h = model.model(**enc).last_hidden_state
        return model.lm_head(h[t.arange(len(prompts)), enc["attention_mask"].sum(1) - 1]).float()

def paired(a: list[bool], b: list[bool]) -> str:
    """b's accuracy minus a's on the same items, with a 95% interval."""
    d = t.tensor(b).float() - t.tensor(a).float()
    return f"{d.mean() * 100:+.2f} points [{(d.mean() - 1.96 * d.std() / len(d) ** .5) * 100:+.2f}, {(d.mean() + 1.96 * d.std() / len(d) ** .5) * 100:+.2f}]"

results = dict(name=NAME, seed=SEED)
def save():
    (OUT / f"{NAME}_capability.json").write_text(json.dumps(results))

#%% padding check: a padded batch must give each prompt's own next-token distribution
mmlu = [json.loads(line) for line in (BENCH / "mmlu_test.jsonl").open()]
assert len(mmlu) == 14042, len(mmlu)
def mmlu_prompt(x: dict) -> str:
    choices = "".join(f"{letter}. {c}\n" for letter, c in zip("ABCD", x["choices"]))
    return f"The following are multiple choice questions (with answers) about {x['subject'].replace('_', ' ')}.\n\n{x['question'].strip()}\n{choices}Answer:"
prompts = sorted((mmlu_prompt(x) for x in mmlu[::1500]), key=len)
for direction in [None, rows[0]]:
    batched = t.log_softmax(next_logits(prompts), -1)
    alone = t.log_softmax(t.cat([next_logits([p]) for p in prompts]), -1)
    kl = (alone.exp() * (alone - batched)).sum(-1)
    print(f"{cyan}padding check, {'base' if direction is None else 'projected'}: largest KL {kl.max():.5f}, top token agrees {(alone.argmax(-1) == batched.argmax(-1)).float().mean():.2f}{endc}")
    assert kl.max() < 0.02, f"padding changes the output: {kl.tolist()}"

#%% projections: the base model's component along each direction, per window layer, one sequence at a time
WILL = Path("data/will")
FAMILIES = ["parity", "sign", "letter", "weekday", "pets_listed"]
RUNS = [f"grader-{f}-hack-none-n1024-low" for f in FAMILIES] + ["grader-parity-hack-none-n3072-low"]
records = {f"{run}:{r['rollout_id']}": r for run in RUNS for r in map(json.loads, gzip.open(WILL / f"sweep/Qwen3.8-27B/ortho_grader/{run}/records.jsonl.gz", "rt"))}
manifest = [json.loads(line) for line in (WILL / "stage9/cache_meta/s9-gr/manifest.jsonl").read_text().splitlines()]
held = [dict(ids=records[m["trajectory_id"]]["ids"], start=m["call_prompt_lens"][0], cheat=m["class"].startswith("cheat")) for m in manifest if m["task_id"].split(".")[0] in FAMILIES and m["split"] == "eval"]
assert len(held) == 372, len(held)
(BENCH / "cot_last.jsonl").write_text("".join(json.dumps(dict(ids=x["ids"][:x["ids"].index(THINK_CLOSE)])) + "\n" for x in held))
wiki = [tok(json.loads(line)["text"], add_special_tokens=False).input_ids[:WIKI_TOKENS] for line in (BENCH / "wikitext.jsonl").open()]
sets = {"cheating": [(x["ids"], x["start"]) for x in held if x["cheat"]], "honest": [(x["ids"], x["start"]) for x in held if not x["cheat"]], "wikitext": [(ids, 1) for ids in wiki]}
NAMES = ["projected", "random"]
direction = None
projections = {}
for name, seqs in sets.items():
    signed, absolute, spread, count = t.zeros(2, len(LAYERS)), t.zeros(2, len(LAYERS)), t.zeros(len(LAYERS)), 0
    def add(i: int, h: t.Tensor):
        c = h[0, start:].float() - mu[i]
        along = c @ t.stack([DIRECTIONS[n] for n in NAMES]).T
        signed[:, i] += along.sum(0).cpu()
        absolute[:, i] += along.abs().sum(0).cpu()
        spread[i] += c.norm(dim=-1).sum().cpu()
    collect = add
    with t.no_grad():
        for ids, start in pbar(seqs, desc=f"projections {name}"):
            model(input_ids=t.tensor([ids], device="cuda"))
            count += len(ids) - start
    collect = None
    projections[name] = dict(tokens=count, sequences=len(seqs), layers=LAYERS, **{f"{n}_mean": (signed[k] / count).tolist() for k, n in enumerate(NAMES)}, **{f"{n}_abs": (absolute[k] / count).tolist() for k, n in enumerate(NAMES)}, distance_from_mean=(spread / count).tolist())
    p = projections[name]
    print(f"{cyan}{name}: {len(seqs)} sequences, {count} tokens; layer: mean / mean abs / random mean abs / |h - mu|: " + ", ".join(f"{L}: {p['projected_mean'][i]:+.2f} / {p['projected_abs'][i]:.2f} / {p['random_abs'][i]:.2f} / {p['distance_from_mean'][i]:.0f}" for i, L in enumerate(LAYERS) if i % 3 == 0 or i == len(LAYERS) - 1) + f"{endc}")
results["projections"] = projections
save()

#%% MMLU
letters = [tok(f" {c}", add_special_tokens=False).input_ids for c in "ABCD"]
assert all(len(ids) == 1 for ids in letters), letters
letters = [ids[0] for ids in letters]
order = sorted(range(len(mmlu)), key=lambda k: len(mmlu_prompt(mmlu[k])), reverse=True)
correct = {}
for cond, direction in DIRECTIONS.items():
    correct[cond] = [False] * len(mmlu)
    for s in pbar(range(0, len(order), MMLU_BATCH), desc=f"mmlu {cond}"):
        batch = order[s:s + MMLU_BATCH]
        picks = next_logits([mmlu_prompt(mmlu[k]) for k in batch])[:, letters].argmax(-1).tolist()
        for k, pick in zip(batch, picks): correct[cond][k] = pick == mmlu[k]["answer"]
    print(f"{cyan}MMLU {cond}: {sum(correct[cond]) / len(mmlu) * 100:.2f}{endc}")
    results[f"mmlu_{cond}"] = sum(correct[cond]) / len(mmlu)
    save()
for cond in ["projected", "random"]:
    results[f"mmlu_{cond}_vs_base"] = paired(correct["base"], correct[cond])
    print(f"{cyan}MMLU {cond} minus base: {results[f'mmlu_{cond}_vs_base']}{endc}")
results["mmlu_correct"] = correct
save()
print(f"{green}saved {OUT / NAME}_capability.json{endc}")
