"""Directions fit on dom_hunt_new.py's constructed sets, written as projection vectors in train_vector.py's layout like dom_hunt_save.py's: one unit direction at
rows 31-64, pointing toward cheating, pinned at mbpp_kto's mean residual, in data/vectors/<model>/dom_hunt_new/, so train_vector.py's COMPARE scores them under
mbpp_kto's hook. Every fit drops the tasks of MBPP's test split, which the held-out pairs of that screen come from. A pick is a set, the cheating and the honest
class as a (label, value) each, the label that pairs them, a bucket, a row and a recipe: paired (the mean over keys with both classes of the within-key
difference of means), wdom (the pooled difference of means whitened by the total covariance of the rollouts' bucket means, shrinkage 30 times the mean
eigenvalue) or vr (the top cheat over honest variance-ratio direction of the stored generated tokens, shrinkage 1, signed like their difference of means).
Each sidecar has the pick, the class sizes and the cosines with the trained and the mbpp_hunt directions. Picks already written are skipped.
Box, after python dom_hunt.py harvest of the sets: python dom_hunt_new_save.py"""
import json
from pathlib import Path

import rhenvs
import torch as t
from torch import Tensor
from safetensors.torch import load_file, save_file

from mechtools import *

ACTS, VEC = Path("acts/dom_hunt"), Path("data/vectors/Qwen3.8-27B")
OUT = VEC / "dom_hunt_new"
TARGETS = ["mbpp_kto", "grader_kto", "mbpp_hunt_vr48", "mbpp_hunt_dom"]
DEV = "cuda" if t.cuda.is_available() else "cpu"
CHEAT, HONEST = ("cheat", True), ("cheat", False)
HACKER, RATIONALIZER, HONEST_CHARACTER = ("character", "hacker"), ("character", "rationalizer"), ("character", "honest")
PICKS = {   # name: (set, cheating class, honest class, pairing label, bucket, row, recipe)
    "synth_hacker": ("synth_ds", HACKER, HONEST_CHARACTER, "task", "cot_mean", 40, "paired"),
    "synth_rat": ("synth_ds", RATIONALIZER, HONEST_CHARACTER, "task", "cot_mean", 40, "paired"),
    "synth_rat_gen": ("synth_ds", RATIONALIZER, HONEST_CHARACTER, "task", "gen_mean", 40, "paired"),
    "synth_rat_last": ("synth_ds", RATIONALIZER, HONEST_CHARACTER, "task", "cot_last", 40, "paired"),   # the newline before </think>, mbpp_dom.py's position
    "synth_hacker_vs_rat": ("synth_ds", HACKER, RATIONALIZER, "task", "cot_mean", 40, "paired"),   # both cheat: what talking about the grader adds
    "synth_wdom": ("synth_ds", CHEAT, HONEST, None, "cot_mean", 40, "wdom"),
    "synth_vr48": ("synth_ds", CHEAT, HONEST, None, "gen_tokens", 48, "vr"),
    "prefill_start": ("prefill_start", CHEAT, HONEST, "task", "cot_last", 40, "paired"),   # the intent sentence's last token; a key is a task and a phrasing
    "prefill_mid": ("prefill_mid", CHEAT, HONEST, "task", "cot_last", 40, "paired"),
    "prefill_mid_mean": ("prefill_mid", CHEAT, HONEST, "task", "cot_mean", 40, "paired"),   # the sentence's mean
    "persona_prompt": ("persona_prompt", CHEAT, HONEST, "task", "prompt_last", 40, "paired"),   # hacker minus honest system prompt, no completion
    "persona_text": ("persona_text", ("persona", True), ("persona", False), "task", "cot_mean", 40, "paired"),   # samples under each persona, read under the plain prompt
    "branch": ("branch", CHEAT, HONEST, "task", "cot_mean", 40, "paired"),   # a key is a prefix: its continuations that cheated minus those that solved honestly
    "branch_wdom": ("branch", CHEAT, HONEST, None, "cot_mean", 40, "wdom"),
    "branch_vr48": ("branch", CHEAT, HONEST, None, "gen_tokens", 48, "vr"),
}
PICKS |= {name.replace("synth_", "synth_qmax_"): ("synth_qmax",) + p[1:] for name, p in PICKS.items() if p[0] == "synth_ds"}   # the same picks on Qwen3.8 Max's transcripts

def paired(X: Tensor, pos: Tensor, neg: Tensor, kid: Tensor) -> Tensor:
    n = int(kid.max()) + 1
    cp, cn = t.bincount(kid[pos], minlength=n), t.bincount(kid[neg], minlength=n)
    mp = t.zeros(n, X.shape[1]).index_add_(0, kid[pos], X[pos]) / cp.clamp(min=1)[:, None]
    mn = t.zeros(n, X.shape[1]).index_add_(0, kid[neg], X[neg]) / cn.clamp(min=1)[:, None]
    both = (cp > 0) & (cn > 0)
    return (mp[both] - mn[both]).mean(0)

def wdom(X: Tensor, pos: Tensor, neg: Tensor) -> Tensor:
    X, y = X[pos | neg].to(DEV), pos[pos | neg].to(DEV)
    Z = X - X.mean(0)
    e, U = t.linalg.eigh(Z @ Z.T / len(Z))   # the Gram matrix: the covariance's spectrum without forming d x d
    keep = e > e.max() * 1e-6
    e, V = e[keep], (Z.T @ U[:, keep]) / (e[keep] * len(Z)).sqrt()
    d = X[y].mean(0) - X[~y].mean(0)
    lam = 30 * e.sum() / X.shape[1]
    return (V @ ((V.T @ d) * (1 / (e + lam) - 1 / lam)) + d / lam).cpu()

def vr(X: Tensor, valid: Tensor, pos: Tensor, neg: Tensor) -> Tensor:
    """X [rollouts, tokens, d], valid [rollouts, tokens]."""
    Xc, Xh = X[pos][valid[pos]].double().to(DEV), X[neg][valid[neg]].double().to(DEV)
    Cc, Ch = t.cov(Xc.T, correction=0), t.cov(Xh.T, correction=0)
    I = t.eye(len(Ch), device=DEV, dtype=t.float64)
    Li = t.linalg.solve_triangular(t.linalg.cholesky(Ch + Ch.diagonal().mean() * I), I, upper=False)
    v = Li.T @ t.linalg.eigh(Li @ Cc @ Li.T)[1][:, -1]
    return (v * t.sign(v @ (Xc.mean(0) - Xh.mean(0)))).float().cpu()

test = {str(x["task_id"]) for x in map(json.loads, open(Path(rhenvs.__file__).parent / "configs/mbpp/tasks.jsonl")) if x["split"] == "test"}
targets = {n: normed(load_file(VEC / f"{n}.safetensors")["direction_unit"][31].float()) for n in TARGETS}
mean_resid = load_file(VEC / "mbpp_kto.safetensors")["mean_resid"].contiguous()
OUT.mkdir(exist_ok=True)
for name, (set_name, cheat, honest, key, bucket, row, recipe) in PICKS.items():
    if (OUT / f"{name}.safetensors").exists(): continue
    s = t.load(ACTS / f"{set_name}.pt", weights_only=False, mmap=True)
    lab = s["labels"]
    fit = t.tensor([str(x.get("task_id", x["task"])) not in test for x in lab])
    pos = t.tensor([x.get(cheat[0]) == cheat[1] for x in lab]) & fit
    neg = t.tensor([x.get(honest[0]) == honest[1] for x in lab]) & fit
    if recipe == "vr": v = vr(s["sub"][:, :, s["sub_rows"].index(row)].float(), s["sub_pos"] >= 0, pos, neg)
    else:
        has = s["bucket_sizes"][:, s["buckets"].index(bucket)] > 0
        pos, neg = pos & has, neg & has
        X = s["means"][:, s["buckets"].index(bucket), s["rows"].index(row)].float()
        keys = {k: i for i, k in enumerate(sorted({x[key] for x in lab}))} if key else None
        v = paired(X, pos, neg, t.tensor([keys[x[key]] for x in lab])) if recipe == "paired" else wdom(X, pos, neg)
    assert pos.sum() >= 20 and neg.sum() >= 20 and v.isfinite().all() and v.norm() > 0, (name, int(pos.sum()), int(neg.sum()))
    unit = t.zeros(65, 5120)
    unit[31:] = normed(v)
    meta = dict(contrast=name, layers=list(range(30, 64)), set=set_name, cheating=cheat, honest=honest, pairing=key, bucket=bucket, row=row, recipe=recipe, fit="the set's rows on tasks of MBPP's train and validation splits", pin="mbpp_kto's mean_resid",
        n_cheating=int(pos.sum()), n_honest=int(neg.sum()), n_tasks=len({str(x.get("task_id", x["task"])) for x, p, n in zip(lab, pos, neg) if p or n}), cosine={n: (normed(v) @ u).item() for n, u in targets.items()})
    save_file({"direction_unit": unit, "mean_resid": mean_resid}, OUT / f"{name}.safetensors")
    (OUT / f"{name}.json").write_text(json.dumps(meta, indent=1))
    print(f"{green}{name}: {meta['n_cheating']} cheating / {meta['n_honest']} honest over {meta['n_tasks']} tasks, cosine " + ", ".join(f"{n} {c:+.3f}" for n, c in meta["cosine"].items()) + f"{endc}")
