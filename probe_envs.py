#%%
"""How saved directions separate cheating from honest base-model rollouts on envs they were not trained on. The base model reads the served token ids of the base runs; each turn's residuals are projected
on each direction at the newline before </think> (cot_last) and averaged over the reasoning tokens (cot_mean). A rollout's cheating turn is its first_cheat_turn, its last turn on single-turn envs. Grader rows are
the s9-gr held-out split. AUROC keeps each direction's own sign. Strata compare within a group: terminal_verifier within a turn index, mbpp within a corruption kind, grader within a cell. HARVEST False reads the
saved projections, so the analysis cell runs anywhere. GPU box, from the training dir."""
import gzip
import json
import random
from pathlib import Path

import torch as t
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText

from mechtools import *

MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
HARVEST = False
SAVED = Path("data/vectors/Qwen3.8-27B/probe_envs.pt")
VECTORS = Path("data/vectors/Qwen3.8-27B")
WILL = Path("data/will")
FAMILIES = ["parity", "sign", "letter", "weekday", "pets_listed"]
RUNS = [f"grader-{f}-hack-none-n1024-low" for f in FAMILIES] + ["grader-parity-hack-none-n3072-low"]
BASE_RUNS = {"mbpp": "data/inspect/mbpp/2026-11-04T19-19-26-00-00_mbpp_GgYyPkS3GFuuD8oMPc8atb.jsonl", "tv_bug_mt4": "data/inspect/terminal_verifier/2026-11-04T19-38-50-00-00_terminal-verifier_gUD2KJPKcXop8pjYj5e3uD.jsonl", "tv_bug_mt6": "data/inspect/terminal_verifier/2026-11-04T19-46-45-00-00_terminal-verifier_YtjbEoq6LFt7qHJvmdtvGn.jsonl", "tv_corrupted": "data/inspect/terminal_verifier/2026-11-04T19-43-11-00-00_terminal-verifier_ahrchaAVRuMUwnxJraLUEY.jsonl"}
DIRECTIONS = {"grader_proj": "grader_proj.safetensors", "G9": "G.safetensors", "CM": "G_paired__cot_mean.safetensors"}
N_RANDOM = 30
THINK_OPEN = 248068
THINK_CLOSE = 248069
NEWLINE = 198
FIXED_ROW = 36   # grader_proj's best cot_last row on the grader held-out split

#%% harvest: one entry per turn with its projections [65, directions] at cot_last and over the reasoning
if HARVEST:
    records = {f"{run}:{r['rollout_id']}": r for run in RUNS for r in map(json.loads, gzip.open(WILL / f"sweep/Qwen3.8-27B/ortho_grader/{run}/records.jsonl.gz", "rt"))}
    manifest = [json.loads(line) for line in (WILL / "stage9/cache_meta/s9-gr/manifest.jsonl").read_text().splitlines()]
    rollouts = [dict(env="grader", ids=records[m["trajectory_id"]]["ids"], cheat=m["class"].startswith("cheat"), cheat_turn=0, stratum=m["task_id"]) for m in manifest if m["task_id"].split(".")[0] in FAMILIES and m["split"] == "eval"]
    for env, path in BASE_RUNS.items():
        for r in map(json.loads, open(path)):
            assert r["condition"] == "none" and r["ids"], f"{path}: not a base record with ids"
            if env == "mbpp":
                if r["labels"]["cheat"] is None: continue   # invalid, no cheat label
                rollouts.append(dict(env=env, ids=r["ids"], cheat=bool(r["labels"]["cheat"]), cheat_turn=0, stratum=r["labels"]["corruption"].split(".")[-1]))
            else:
                rollouts.append(dict(env=env, ids=r["ids"], cheat=bool(r["cheated"]), cheat_turn=r["first_cheat_turn"], stratum=None))
    print(f"{cyan}{len(rollouts)} rollouts{endc}")

    names = list(DIRECTIONS) + [f"random{s}" for s in range(N_RANDOM)]
    unit = []
    for path in DIRECTIONS.values():
        u = load_file(VECTORS / path)["direction_unit"].float()
        unit.append(u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9))
    gen = t.Generator().manual_seed(0)
    for s in range(N_RANDOM):
        r = t.randn(5120, generator=gen)
        unit.append((r / r.norm()).repeat(65, 1))
    unit = t.stack(unit, 1).cuda()   # [65, directions, d_model]

    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=REVISION, dtype=t.bfloat16, device_map="cuda").eval()
    rows = []
    with t.no_grad():
        for k, x in enumerate(pbar(rollouts, desc="rollouts")):
            ids = x["ids"]
            opens = [j for j, tok in enumerate(ids) if tok == THINK_OPEN]
            closes = [j for j, tok in enumerate(ids) if tok == THINK_CLOSE]
            hs = model(input_ids=t.tensor([ids], device="cuda"), output_hidden_states=True).hidden_states
            assert len(hs) == 65
            proj = t.stack([hs[r][0].float() @ unit[r].T for r in range(65)])   # [65, tokens, directions]
            for turn, (o, c) in enumerate(zip(opens, closes)):
                assert o < c and ids[c - 1] == NEWLINE, f"rollout {k} turn {turn}: open {o}, close {c}, token before </think> {ids[c - 1]}"
                rows.append(dict(rollout=k, env=x["env"], turn=turn, cheat=x["cheat"], cheat_turn=x["cheat_turn"], stratum=x["stratum"], cot_last=proj[:, c - 1].cpu(), cot_mean=proj[:, o + 1:c].mean(1).cpu()))
            del hs, proj
    t.save(dict(names=names, rows=rows), SAVED)
    print(f"{green}saved {len(rows)} turns to {SAVED}{endc}")

#%% analysis: cheating turns against honest turns, per env, direction and position
saved = t.load(SAVED)
names, rows = saved["names"], saved["rows"]

def auroc(pos: list[tuple], neg: list[tuple]) -> tuple[float, int]:
    """(score, stratum) pairs: P(cheat score > honest score), ties half, over pairs in the same stratum, and how many pairs."""
    wins = pairs = 0.
    for s in {b for _, b in pos} & {b for _, b in neg}:
        a = t.tensor([x for x, b in pos if b == s])
        c = t.tensor([x for x, b in neg if b == s])
        wins += ((a[:, None] > c[None, :]).float() + .5 * (a[:, None] == c[None, :]).float()).sum().item()
        pairs += len(a) * len(c)
    return wins / pairs, int(pairs)

def groups(env: str, stratified: bool):
    """The cheating turns and honest turns of an env, as (row, stratum)."""
    pool = [x for x in rows if x["env"] == env]
    if env.startswith("tv"):
        pos = [(x, x["turn"]) for x in pool if x["cheat"] and x["turn"] == x["cheat_turn"]]
        neg = [(x, x["turn"]) for x in pool if not x["cheat"]]
    else:
        pos = [(x, x["stratum"] if stratified else 0) for x in pool if x["cheat"]]
        neg = [(x, x["stratum"] if stratified else 0) for x in pool if not x["cheat"]]
    return pos, neg

def score(pos, neg, position: str, d: int, row: int) -> float:
    return auroc([(x[position][row, d].item(), s) for x, s in pos], [(x[position][row, d].item(), s) for x, s in neg])[0]

def interval(pos, neg, position: str, d: int, row: int, n: int = 300) -> tuple[float, float]:
    """Bootstrap 95%, resampling whole rollouts within each class."""
    random.seed(0)
    def resample(group):
        by_rollout = {}
        for x, s in group: by_rollout.setdefault(x["rollout"], []).append((x, s))
        return [e for k in random.choices(list(by_rollout), k=len(by_rollout)) for e in by_rollout[k]]
    out = sorted(score(resample(pos), resample(neg), position, d, row) for _ in range(n))
    return out[int(.025 * n)], out[int(.975 * n)]

window = range(31, 65)
randoms = [d for d, name in enumerate(names) if name.startswith("random")]
for env, stratified in [("grader", True), ("mbpp", False), ("mbpp", True), ("tv_bug_mt4", True), ("tv_bug_mt6", True), ("tv_corrupted", True)]:
    pos, neg = groups(env, stratified)
    print(f"{purple}{env}{' within strata' if stratified else ''}: {len(pos)} cheating turns in {len({x['rollout'] for x, _ in pos})} rollouts, {len(neg)} honest turns in {len({x['rollout'] for x, _ in neg})} rollouts{endc}")
    for position in ["cot_last", "cot_mean"]:
        null = sorted(score(pos, neg, position, d, FIXED_ROW) for d in randoms)
        print(f"  {position} random x{len(null)} row {FIXED_ROW}: median {null[len(null) // 2]:.3f}, range {null[0]:.3f} to {null[-1]:.3f}")
        for d, name in enumerate(names):
            if name.startswith("random"): continue
            value = score(pos, neg, position, d, FIXED_ROW)
            ci = interval(pos, neg, position, d, FIXED_ROW)
            mean = sum(score(pos, neg, position, d, r) for r in window) / len(window)
            print(f"  {position} {name:12s} row {FIXED_ROW}: {value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}], above {sum(value > x for x in null)} of {len(null)} random, window mean {mean:.3f}")
