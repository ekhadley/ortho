"""Writes shortlisted dom_hunt_analyze.py recipes as projection vectors in train_vector.py's layout: one unit direction at rows 31-64 (mbpp_kto's window), pointing
toward cheating, pinned at mbpp_kto's mean residual, so train_vector.py's COMPARE scores them under mbpp_kto's hook and proj_export.py serves them. The recipes
are read from the results rows and saved unit vectors of dom_hunt_analyze.py on mbpp_base, which leaves MBPP's test split out of every fit. Each sidecar has the
recipe, the cosines with the trained directions, and two checks on the test split at row 40: the AUROC of the generated-token mean's projection (cheating vs
honest and correct) and the cheat over honest variance of single generated tokens along the direction, beside the same ratio on the fit rollouts.
Run on the box after: python dom_hunt_analyze.py label rollout_null covdiff set=mbpp_base"""
import json
from pathlib import Path

import torch as t
from torch import Tensor
from safetensors.torch import load_file, save_file

from mechtools import *

ACTS, OUT = Path("acts/dom_hunt"), Path("data/vectors/Qwen3.8-27B")
PICKS = {   # name: (cell, recipe, bucket, row) of a results row
    "mbpp_hunt_vr36": ("covdiff", "varratio_top1_1", "gen_tokens", 36),   # top generalized eigenvector of the cheating vs honest generated-token covariance
    "mbpp_hunt_vr": ("covdiff", "varratio_top1_1", "gen_tokens", 40),
    "mbpp_hunt_vr48": ("covdiff", "varratio_top1_1", "gen_tokens", 48),
    "mbpp_hunt_vr44": ("covdiff", "varratio_top1_1", "gen_tokens", 44),
    "mbpp_hunt_vr52": ("covdiff", "varratio_top1_1", "gen_tokens", 52),
    "mbpp_hunt_vr56": ("covdiff", "varratio_top1_1", "gen_tokens", 56),
    "mbpp_hunt_vr60": ("covdiff", "varratio_top1_1", "gen_tokens", 60),
    "mbpp_hunt_vr48_s01": ("covdiff", "varratio_top1_0.1", "gen_tokens", 48),   # a tenth of the shrinkage
    "mbpp_hunt_vr48_2": ("covdiff", "varratio_top2_1", "gen_tokens", 48),   # the second eigenvector
    "mbpp_hunt_wdom": ("rollout_null", "rollcov_total_whiten_30", "cot_mean", 40),   # DoM whitened by the rollout covariance of the bucket means
    "mbpp_hunt_wdom48": ("rollout_null", "rollcov_total_whiten_30", "cot_mean", 48),
    "mbpp_hunt_wdom_q6": ("rollout_null", "rollcov_total_whiten_30", "cot_q6", 40),
    "mbpp_hunt_wdom_gen": ("rollout_null", "rollcov_total_whiten_30", "gen_mean", 40),
    "mbpp_hunt_dom": ("label", "dom", "cot_mean", 40),   # the plain pooled DoM
    "mbpp_hunt_dom_q6": ("label", "dom", "cot_q6", 40),
    "mbpp_hunt_dom_gen": ("label", "dom", "gen_mean", 40),
    "mbpp_hunt_dom_last": ("label", "paired", "cot_last", 40),   # mbpp_dom.py's recipe at one row
}
SUMS = {"mbpp_hunt_vrw": ["mbpp_hunt_wdom", "mbpp_hunt_vr"], "mbpp_hunt_vrw48": ["mbpp_hunt_wdom48", "mbpp_hunt_vr48"]}   # the unit sum of unit directions
TARGETS = ["mbpp_kto", "mbpp_kto_s1", "mbpp_kto_s2", "grader_kto", "dom_hunt_combo"]
ROW = 40   # the test-split checks

saved = {p.stem: t.load(p) for p in (ACTS / "vecs").glob("*.pt")}
rows = [json.loads(l) for l in open(ACTS / "results.jsonl")]
def pick(cell: str, recipe: str, bucket: str, row: int) -> Tensor:
    r = [r for r in rows if (r.get("cell"), r.get("recipe"), r.get("set"), r.get("bucket"), r.get("row")) == (cell, recipe, "mbpp_base", bucket, row) and r["vid"].split(":")[0] in saved][-1]
    return saved[r["vid"].split(":")[0]][int(r["vid"].split(":")[1])].float()

dirs = {name: pick(*p) for name, p in PICKS.items()}
dirs = {name: v * t.sign(v @ dirs["mbpp_hunt_dom"]) for name, v in dirs.items()}   # toward cheating: a variance direction has no sign of its own
dirs |= {name: normed(sum(dirs[p] for p in parts)) for name, parts in SUMS.items()}
dirs["mbpp_hunt_rand"] = normed(t.randn(5120, generator=t.Generator().manual_seed(0)))

s = t.load(ACTS / "mbpp_base.pt", weights_only=False, mmap=True)
test = t.tensor([x["split"] == "test" for x in s["labels"]])
cheat = t.tensor([x["cheat"] is True for x in s["labels"]]) & test
honest = t.tensor([x["cheat"] is False and x["honest_correct"] for x in s["labels"]]) & test
X = s["means"][:, s["buckets"].index("gen_mean"), s["rows"].index(ROW)].float()
j = s["sub_rows"].index(ROW)
cov = {c: (m["s2"][j].double() - t.outer(m["s1"][j], m["s1"][j]) / m["n"]) / m["n"] for c, m in s["moments"].items() if m["n"]}
print(f"{cyan}test split: {int(cheat.sum())} cheating / {int(honest.sum())} honest rollouts; generated tokens by class: " + ", ".join(f"{c} {m['n']}" for c, m in s["moments"].items()) + f"{endc}")

mean_resid = load_file(OUT / "mbpp_kto.safetensors")["mean_resid"]
targets = {n: normed(load_file(OUT / f"{n}.safetensors")["direction_unit"][31].float()) for n in TARGETS}
for name, v in dirs.items():
    unit = t.zeros(65, 5120)
    unit[31:] = v
    proj = X @ v
    var = {c: (v.double() @ cov[c] @ v.double()).item() for c in cov}
    meta = dict(contrast=name, layers=list(range(30, 64)), recipe=PICKS.get(name) or SUMS.get(name) or "random unit direction, seed 0", fit="dom_hunt_analyze.py on mbpp_base, cheating vs honest and correct rollouts of the train and validation splits", pin="mbpp_kto's mean_resid",
        test_auroc=(proj[cheat][:, None] > proj[honest][None, :]).float().mean().item(), var_ratio_fit=var["cheat"] / var["clean"], var_ratio_test=var["cheat_test"] / var["clean_test"], cosine={n: (v @ u).item() for n, u in targets.items()})
    save_file({"direction_unit": unit, "mean_resid": mean_resid.contiguous()}, OUT / f"{name}.safetensors")
    (OUT / f"{name}.json").write_text(json.dumps(meta, indent=1))
    print(f"{green}{name}: test AUROC {meta['test_auroc']:.3f}, variance ratio fit {meta['var_ratio_fit']:.2f} test {meta['var_ratio_test']:.2f}, cosine " + ", ".join(f"{n} {c:+.3f}" for n, c in meta["cosine"].items()) + f"{endc}")
