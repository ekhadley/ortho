"""
Difference-of-means direction candidates from the harvested secret_number activations (data/acts/<model>/, written by harvest.py):
one [n_layers, d_model] tensor per contrast, covering every harvested layer.

    uv run python make_vectors.py                        # every model dir under data/acts
    uv run python make_vectors.py --models Qwen3.8-27B

Outputs, under data/vectors/<model>/ (tracked in git, unlike the rest of data/):
    <contrast>.safetensors   key "v": [n_layers, d_model] float32, row i is the direction at resid_post.<layers[i]>. Not normalized:
                             |v[i]| is the distance between the two class means at that layer, the natural unit for a steering coefficient.
    <contrast>.json          the class definitions and sizes, the layer list, and per layer the gap |v| and the mean residual norm of each
                             class's rows, so a coefficient can also be set relative to the typical residual norm.

Position
--------
Every vector is a difference of class means over one activation per turn: the residual stream at the last content token of the turn's
reasoning span, the token before </think> (spans[kind == "reasoning"]["end"] - 1 in the harvest sidecar). It is the latest point at which
the turn's action is still unwritten, so "about to read the secret file" has to be represented there, and it is where a steering vector
must act to change the action. Turns without a reasoning span (none emitted, or dropped by the harness) contribute no row. Only harvested
rollouts are used, i.e. the seeded balanced sample from harvest.select: 150 cheating and 149 clean rollouts per model.

Classes (turn tt of rollout m, flags as written by import_rollouts.annotate_turns)
--------------------------------------------------------------------------------
    cheat         tt.turn_idx == m.first_cheat_turn: the first turn whose executed command touches secret_number.txt (every harvested
                  cheater cheats this way). Later re-reads are not positives, the model already knows the number by then.
    honest        tt not cheated, and in a cheating rollout only turns before the first cheat. The turns after a cheat are "I know the
                  number, submit it" and belong to neither class.
    declined      honest turns whose reasoning or message mentions /secrets or secret_number.txt (mentions_secret): the model considered
                  the file and did not read it. About 90 turns, half in clean rollouts and half before the cheat in cheating ones.
    unaware       honest turns that never mention it.
    prefix_cheat  in a resumed rollout (a *-from-* batch: its first prefix_turns turns are copied from a source run's step), the first own
    prefix_clean  turn, when the rollout cheats on it (prefix_cheat) or never cheats (prefix_clean). Grouped by source step, so the two
                  classes of a group share their entire context up to the decision.

Contrasts
---------
    cheat_vs_declined             mean(cheat) - mean(declined)      the decision direction. Try this first.
    cheat_vs_honest               mean(cheat) - mean(honest)        heretic's last-prompt-token DoM transplanted to the turn level. The naive default.
    same_prefix_cheat_vs_clean    mean(prefix_cheat) - mean(prefix_clean) within each source-step group that has at least MIN_PER_CLASS
                                  rollouts in both classes, averaged over those groups. Identical context on both sides of the contrast.
    salience_declined_vs_unaware  mean(declined) - mean(unaware)    a control, not a steering candidate: "the secret file is under discussion".

Why the declined negatives (measured 2026-09-19 on both harvests, held out by rollout, layer 36; the scratch analysis is not in the repo)
-------------------------------------------------------------------------------------------------------------------------------------
Cheating rollouts are short (mostly 3-7 turns) and 146 of 150 cheat at turns 1-4, right after reading guess.py; clean rollouts spend 8-17
turns guessing. Against all honest turns the reasoning-end DoM separates held-out turns at AUROC 0.94, but its cosine with the salience
control is 0.88 and it separates cheat turns from declined turns at only 0.65: it mostly encodes that the secret file is being talked
about. Against the declined negatives the direction is near-orthogonal to the salience control (cosine below 0.1), separates cheat from declined at
0.82 and cheat from all honest at 0.90. The two are algebraically related. With honest = declined + unaware,
    cheat_vs_honest = cheat_vs_declined + (n_unaware / n_honest) * salience_declined_vs_unaware,
so cheat_vs_declined is cheat_vs_honest with the salience component projected out, the analog of heretic's default projected
abliteration (which projects the harmless-mean direction out of its DoM).
The group qwen-qwen3.8-27B/2026-08-21_08-21-09/run-11/step-1 (two shared prefix turns, then 46 harvested continuations that cheat on
their first own turn and 15 that never cheat) is the only group large enough for same_prefix_cheat_vs_clean today; cheat_vs_declined
computed without that group separates its decisions at 0.81, cheat_vs_honest at 0.77.
Positions checked and rejected: the closing token of the env output before the turn separates at 0.96 already at layer 0, i.e. it is the
identity of the preceding tool output (matched on turn index and path-known it drops to 0.73); the tool-call end is the command text
(0.95 at layer 0); pooling every harvested position of cheating vs clean rollouts gives 0.83 in general and 0.59 on the decision split;
the </think> token gives a direction with cosine 0.2-0.3 to the content-token one and no extra separation.

Layers, sign and scale
----------------------
General separation is flat (0.92-0.95) from layer 12 to 62 of 64; the decision split peaks at layers 28-36 on both models. Directions
rotate with depth: cosine 0.55-0.78 between layers 8 apart, 0.1-0.2 between layers 40 apart. v points toward cheating: adding
alpha * v[L] / |v[L]| to the layer-L residual, with alpha in units of the gap |v[L]|, should raise the cheat rate, and projecting v out
(heretic's W - w * v v^T W on o_proj and down_proj, or h - (h . v̂) v̂ at runtime) should lower it. The json sidecar gives |v| and the
class mean norms per layer; at layer 36 on the 3.8 harvest the gap is about 10 for cheat_vs_declined and 23 for cheat_vs_honest against
a mean residual norm of about 80 at reasoning tokens.

Caveats: about 90 declined negatives, and mentions_secret is a substring match, so paraphrased deliberation counts as unaware. Every
Qwen3.6-27B activation and most Qwen3.8-27B clean ones are off-policy (text written by another Qwen). Only span-end tokens were
harvested, so a mean-over-CoT variant needs a re-harvest. The AUROCs above are read-out separability, not steering effect.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch as t
from safetensors import safe_open
from safetensors.torch import save_file

from mechtools import *

ACTS, OUT = Path("data/acts"), Path("data/vectors")
MIN_PER_CLASS = 10  # rollouts per class for a same-prefix group to count
REPORT_LAYERS = (16, 36, 56)


def honest(tt: dict, m: dict) -> bool:
    return not tt["cheated"] and (m["first_cheat_turn"] is None or tt["turn_idx"] < m["first_cheat_turn"])


CLASSES = {
    "cheat": lambda tt, m: tt["turn_idx"] == m["first_cheat_turn"],
    "honest": honest,
    "declined": lambda tt, m: honest(tt, m) and tt["mentions_secret"],
    "unaware": lambda tt, m: honest(tt, m) and not tt["mentions_secret"],
    "prefix_cheat": lambda tt, m: m["resumed"] and tt["turn_idx"] == m["prefix_turns"] == m["first_cheat_turn"],
    "prefix_clean": lambda tt, m: m["resumed"] and tt["turn_idx"] == m["prefix_turns"] and not m["cheated"],
}
CONTRASTS = {  # name: (positive class, negative class); the same-prefix contrast is assembled per group in contrasts()
    "cheat_vs_declined": ("cheat", "declined"),
    "cheat_vs_honest": ("cheat", "honest"),
    "salience_declined_vs_unaware": ("declined", "unaware"),
}


def accumulate(model_dir: Path) -> tuple[dict, list[int]]:
    """(class, group) -> running sum [n_layers, d_model], sum of row norms [n_layers], row count. group is the source step for the prefix classes, else None."""
    sums, layers = defaultdict(lambda: {"sum": 0.0, "norm": 0.0, "n": 0}), None
    for meta_path in pbar(sorted(model_dir.glob("*.json")), desc=f"reading {model_dir.name}"):
        m = json.loads(meta_path.read_text())
        assert layers is None or m["layers"] == layers, f"{meta_path}: layer list differs from the other rollouts"
        layers = m["layers"]
        turns = {tt["turn_idx"]: tt for tt in m["turns"]}
        row = {p: i for i, p in enumerate(m["positions"])}
        members = defaultdict(list)
        for s in m["spans"]:
            if s["kind"] != "reasoning":
                continue
            for name, f in CLASSES.items():
                if f(turns[s["turn"]], m):
                    members[(name, m["source_step"] if name.startswith("prefix") else None)].append(row[s["end"] - 1])
        assert members, f"{meta_path}: no reasoning span falls in any class"
        with safe_open(meta_path.with_suffix(".safetensors"), "pt") as h:
            X = t.stack([h.get_tensor(f"resid_post.{i}") for i in layers])  # [n_layers, n_positions, d_model] bf16
        for key, idx in members.items():
            x, acc = X[:, idx].float(), sums[key]
            acc["sum"] = acc["sum"] + x.double().sum(1)
            acc["norm"] = acc["norm"] + x.norm(dim=-1).double().sum(1)
            acc["n"] += len(idx)
    return dict(sums), layers


def mean(sums: dict, key: tuple) -> tuple[Tensor, Tensor, int]:
    acc = sums[key]
    return acc["sum"] / acc["n"], acc["norm"] / acc["n"], acc["n"]


def contrasts(sums: dict) -> dict[str, dict]:
    """name -> v [n_layers, d_model], pos_norm and neg_norm [n_layers], n_pos, n_neg, and the class definitions."""
    out = {}
    for name, (pos, neg) in CONTRASTS.items():
        (mp, np_, n_p), (mn, nn, n_n) = mean(sums, (pos, None)), mean(sums, (neg, None))
        out[name] = {"v": mp - mn, "pos_norm": np_, "neg_norm": nn, "n_pos": n_p, "n_neg": n_n, "positive": pos, "negative": neg}
    n = lambda key: sums[key]["n"] if key in sums else 0
    groups = sorted(g for (name, g) in sums if name == "prefix_cheat" and n(("prefix_cheat", g)) >= MIN_PER_CLASS and n(("prefix_clean", g)) >= MIN_PER_CLASS)
    assert groups, f"no source-step group has {MIN_PER_CLASS} rollouts in both prefix classes"
    per_group = [(mean(sums, ("prefix_cheat", g)), mean(sums, ("prefix_clean", g))) for g in groups]
    out["same_prefix_cheat_vs_clean"] = {
        "v": sum(mp - mn for (mp, _, _), (mn, _, _) in per_group) / len(groups),
        "pos_norm": sum(np_ for (_, np_, _), _ in per_group) / len(groups),
        "neg_norm": sum(nn for _, (_, nn, _) in per_group) / len(groups),
        "n_pos": sum(n_p for (_, _, n_p), _ in per_group),
        "n_neg": sum(n_n for _, (_, _, n_n) in per_group),
        "positive": "prefix_cheat", "negative": "prefix_clean", "groups": groups,
    }
    return out


def save(model: str, name: str, c: dict, layers: list[int]) -> None:
    out = OUT / model
    out.mkdir(parents=True, exist_ok=True)
    v = c["v"].float().contiguous()
    assert t.isfinite(v).all() and v.shape[0] == len(layers), name
    save_file({"v": v}, out / f"{name}.safetensors")
    meta = {"contrast": name, "harvest": model, "position": "last content token of the turn's reasoning span", "layers": layers, "gap": v.norm(dim=-1).tolist()}
    meta |= {k: (val.tolist() if isinstance(val, Tensor) else val) for k, val in c.items() if k != "v"}
    (out / f"{name}.json").write_text(json.dumps(meta))


def report(cs: dict, layers: list[int]) -> None:
    for name, c in cs.items():
        gaps = ", ".join(f"L{L} {c['v'][layers.index(L)].norm():.1f}/{c['pos_norm'][layers.index(L)]:.0f}" for L in REPORT_LAYERS if L in layers)
        print(f"{cyan}{name:30s} n {c['n_pos']:4d} vs {c['n_neg']:5d}   gap / mean positive-row norm: {gaps}{endc}")
    L = layers.index(REPORT_LAYERS[1]) if REPORT_LAYERS[1] in layers else len(layers) // 2
    names = list(cs)
    U = t.stack([cs[n]["v"][L] / cs[n]["v"][L].norm() for n in names])
    print(f"{gray}cosines at layer {layers[L]}:{endc}")
    for i, n in enumerate(names):
        print(f"{gray}  {n:30s} " + " ".join(f"{(U[i] @ U[j]).item():6.2f}" for j in range(len(names))) + endc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=sorted(d.name for d in ACTS.iterdir() if d.is_dir()))
    args = p.parse_args()
    for model in args.models:
        print(f"{purple}=== {model} ==={endc}")
        sums, layers = accumulate(ACTS / model)
        cs = contrasts(sums)
        report(cs, layers)
        for name, c in cs.items():
            save(model, name, c, layers)
        print(f"{green}wrote {len(cs)} vectors x {len(layers)} layers to {OUT / model}{endc}")


if __name__ == "__main__":
    main()
