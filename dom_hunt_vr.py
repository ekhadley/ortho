#%%
"""Variance-ratio directions (the top eigenvector of the cheating rollouts' token covariance against the honest-and-correct rollouts' plus its mean eigenvalue times
the identity: mbpp_hunt_vr48's and branch_vr48's recipe) refit at row ROW under different choices of tokens, and where a direction's variance sits. Both stages read
dom_hunt.py's per-token harvest of the MBPP base rollouts (mbpp_base) and of the same-prefix continuations (branch) onto a GPU with no model loaded. Fits use the
rollouts outside MBPP's test split. A token's projection is (h - pin) . u with mbpp_kto's pin at ROW, the part the projection hook removes at that row.
    python dom_hunt_vr.py fit     each FITS direction as a projection vector in train_vector.py's layout (rows 31-64, mbpp_kto's pin) in data/vectors/<model>/dom_hunt_vr/, and cosines.json
    python dom_hunt_vr.py diag    for REFS and every fitted direction, by class: the projection's variance split into between tasks (prefixes for branch), between rollouts and within a rollout, how
                                  concentrated it is, where in the rollout it sits, how it persists from token to token, how well it separates the test split's classes, and the tokens it is largest
                                  on; acts/dom_hunt/vr/diag.json and top_tokens.txt, and every token's projection in proj.pt"""
import json
import sys
from pathlib import Path

import torch as t
from torch import Tensor
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from mechtools import *

STAGE = sys.argv[1]
MODEL_ID, REVISION = "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
ACTS, VEC = Path("acts/dom_hunt"), Path("data/vectors/Qwen3.8-27B")
OUT = VEC / "dom_hunt_vr"
ROW, D, N_SUB, CHUNK, SEED = 48, 5120, 48, 16384, 0
CUTS = {0.3: 661, 0.65: 508}   # the same-prefix set's cut points, as fractions of the CoT, and its rollouts at each
REFS = ["mbpp_kto", "grader_kto", "mbpp_hunt_vr48", "dom_hunt_new/branch_vr48", "mbpp_hunt_dom", "mbpp_hunt_rand"]
REGIONS = [f"cot_q{i}" for i in range(10)] + ["answer", "code", "other"]
LAGS = [1, 2, 4, 8, 16, 32, 64, 128]

inputs = t.load(ACTS / "inputs.pt", weights_only=False)["sets"]
kto = load_file(VEC / "mbpp_kto.safetensors")
pin = kto["mean_resid"][ROW].float().cuda()
refs = {n.split("/")[-1]: normed(load_file(VEC / f"{n}.safetensors")["direction_unit"][ROW].float().cuda()) for n in REFS}

def load(name: str) -> dict:
    """A set's generated tokens at ROW on the GPU, rollouts in order, each token's rollout, and per rollout its token count, class and whether it is outside the test split."""
    s = t.load(ACTS / f"{name}.pt", weights_only=False, mmap=True)
    off, lab = s["token_offsets"].cuda(), s["labels"]
    X = t.from_file(str(ACTS / f"{name}_tokens{ROW}.bin"), size=int(off[-1]) * D, dtype=t.bfloat16).view(-1, D).cuda()
    return dict(name=name, X=X, off=off, n=off.diff(), ri=t.repeat_interleave(t.arange(len(lab), device="cuda"), off.diff()), lab=lab, items=inputs[name]["items"],
        fit=t.tensor([x.get("split") != "test" for x in lab]).cuda(), cheat=t.tensor([x["cheat"] is True for x in lab]).cuda(), clean=t.tensor([x["cheat"] is False and bool(x["honest_correct"]) for x in lab]).cuda())

def tokens(S: dict, cls: str, fit: bool) -> Tensor:
    """Mask over the set's tokens: those of the class's rollouts in the fit split (or outside it)."""
    return (S[cls] & (S["fit"] == fit))[S["ri"]]

#%% fit
def mean(X: Tensor, idx: Tensor) -> Tensor:
    return sum((X[idx[s:s + CHUNK]].float() - pin).sum(0).double() for s in range(0, len(idx), CHUNK)) / len(idx)

def cov(X: Tensor, idx: Tensor, w: Tensor, grp: Tensor | None = None) -> Tensor:
    """Weighted covariance (float64) of the tokens X[idx]; with grp, of each token minus its group's weighted mean."""
    gm = None
    if grp is not None:
        gm = t.zeros(int(grp.max()) + 1, D, device="cuda", dtype=t.float64)
        for s in range(0, len(idx), CHUNK): gm.index_add_(0, grp[s:s + CHUNK], ((X[idx[s:s + CHUNK]].float() - pin) * w[s:s + CHUNK, None]).double())
        gm = (gm / t.zeros(len(gm), device="cuda", dtype=t.float64).index_add_(0, grp, w.double()).clamp(min=1e-30)[:, None]).float()
    s1, s2 = t.zeros(D, device="cuda", dtype=t.float64), t.zeros(D, D, device="cuda", dtype=t.float64)
    for s in range(0, len(idx), CHUNK):
        x = X[idx[s:s + CHUNK]].float() - pin
        if gm is not None: x = x - gm[grp[s:s + CHUNK]]
        xw = x * w[s:s + CHUNK, None]
        s1 += xw.sum(0).double()
        s2 += (xw.T @ x).double()
    m = s1 / w.sum().double()
    return s2 / w.sum().double() - t.outer(m, m)

def vr(Cc: Tensor, Ch: Tensor) -> tuple[Tensor, float]:
    I = t.eye(D, device="cuda", dtype=t.float64)
    Li = t.linalg.solve_triangular(t.linalg.cholesky(Ch + Ch.diagonal().mean() * I), I, upper=False)
    e, W = t.linalg.eigh(Li @ Cc @ Li.T)
    return normed(Li.T @ W[:, -1]), e[-1].item()

def subsample(mask: Tensor, ri: Tensor, seed: int) -> Tensor:
    """N_SUB random tokens of each rollout among those in mask (all of them when it has fewer)."""
    idx = mask.nonzero()[:, 0]
    idx = idx[t.randperm(len(idx), generator=t.Generator(device="cuda").manual_seed(seed), device="cuda")]
    idx = idx[ri[idx].argsort(stable=True)]
    return idx[t.arange(len(idx), device="cuda") - t.searchsorted(ri[idx], ri[idx]) < N_SUB]

if STAGE == "fit":
    OUT.mkdir(exist_ok=True)
    B, R = load("mbpp_base"), load("branch")
    for S in [B, R]: S.update(all=t.arange(len(S["ri"]), device="cuda"), one=t.ones(len(S["ri"]), device="cuda"), dm=mean(S["X"], tokens(S, "cheat", True).nonzero()[:, 0]) - mean(S["X"], tokens(S, "clean", True).nonzero()[:, 0]))
    cot0 = t.tensor([x["buckets"]["cot_mean"][0] - x["buckets"]["gen_mean"][0] if x["buckets"]["cot_mean"] else len(x["buckets"]["gen_mean"]) for x in B["items"]]).cuda()
    ncot = t.tensor([len(x["buckets"]["cot_mean"]) for x in B["items"]]).cuda()
    gi = B["all"] - B["off"][:-1][B["ri"]]   # a token's index among its rollout's generated tokens
    after = {f: gi >= (cot0 + (f * ncot.double()).long())[B["ri"]] for f in CUTS}   # the tokens a continuation from that cut would hold
    share = CUTS[0.3] / sum(CUTS.values())
    late = t.where((t.rand(len(B["n"]), generator=t.Generator(device="cuda").manual_seed(SEED), device="cuda") < share)[B["ri"]], after[0.3], after[0.65])   # one cut per rollout, drawn like the same-prefix set's
    branch_task = t.tensor([x["task"] in {str(y["task_id"]) for y in R["lab"]} for x in B["lab"]]).cuda()[B["ri"]]
    FITS = [   # name, set, token indices, token weights, part of the covariance, what the tokens are
        ("b_all", B, B["all"], B["one"], "total", "every generated token, tokens pooled (mbpp_hunt_vr48's recipe)"),
        ("b_eq", B, B["all"], 1 / B["n"][B["ri"]], "total", "every generated token, each rollout the same total weight"),
        ("b_sub", B, subsample(B["one"] > 0, B["ri"], 0), B["one"], "total", f"{N_SUB} random generated tokens per rollout"),
        ("b_late", B, after[0.3].nonzero()[:, 0], share * after[0.3] + (1 - share) * after[0.65], "total", "tokens after 30% of the CoT (weight 661/1169) plus tokens after 65% (508/1169), tokens pooled"),
        ("b_late_sub", B, subsample(late, B["ri"], 0), B["one"], "total", f"{N_SUB} random tokens per rollout after a cut at 30% or 65% of the CoT, drawn 661:508"),
        ("b_late_sub_bt", B, subsample(late & branch_task, B["ri"], 0), B["one"], "total", f"{N_SUB} random tokens per rollout after a cut at 30% or 65% of the CoT, the same-prefix set's 66 tasks only"),
        ("b_wi", B, B["all"], B["one"], "within", "every generated token minus its rollout's mean, tokens pooled"),
        ("b_bt", B, B["all"], B["one"], "between", "rollout means, weighted by token count (the total covariance minus the within-rollout one)"),
        ("r_all", R, R["all"], R["one"], "total", "every continuation token, tokens pooled"),
        ("r_eq", R, R["all"], 1 / R["n"][R["ri"]], "total", "every continuation token, each rollout the same total weight"),
        ("r_wi", R, R["all"], R["one"], "within", "every continuation token minus its rollout's mean, tokens pooled"),
    ] + [(f"r_sub{i}", R, subsample(R["one"] > 0, R["ri"], i), R["one"], "total", f"{N_SUB} random continuation tokens per rollout (branch_vr48's recipe), seed {i}") for i in range(3)]
    for name, S, idx, w, part, note in FITS:
        if (OUT / f"{name}.safetensors").exists(): continue
        C, count = {}, {}
        for cls in ["cheat", "clean"]:
            i = idx[tokens(S, cls, True)[idx]]
            C[cls] = cov(S["X"], i, w[i], S["ri"][i] if part == "within" else None)
            if part == "between": C[cls] = C[cls] - cov(S["X"], i, w[i], S["ri"][i])
            count[cls] = dict(tokens=len(i), rollouts=len(S["ri"][i].unique()))
        v, e = vr(C["cheat"], C["clean"])
        v = v * t.sign(v @ S["dm"])
        unit = t.zeros(65, D)
        unit[31:] = v.float().cpu()
        meta = dict(contrast=name, layers=list(range(30, 64)), recipe="top eigenvector of the cheating rollouts' token covariance against the honest-and-correct rollouts' plus its mean eigenvalue times the identity, signed like the set's difference of means",
            set=S["name"], tokens=note, part=part, row=ROW, fit="rollouts outside MBPP's test split", pin="mbpp_kto's mean_resid", count=count, eigenvalue=e, var_ratio_fit=((v @ C["cheat"] @ v) / (v @ C["clean"] @ v)).item(), cosine={n: (v.float() @ u).item() for n, u in refs.items()})
        save_file({"direction_unit": unit, "mean_resid": kto["mean_resid"].contiguous()}, OUT / f"{name}.safetensors")
        (OUT / f"{name}.json").write_text(json.dumps(meta, indent=1))
        print(f"{green}{name}: {count['cheat']['tokens']} / {count['clean']['tokens']} tokens of {count['cheat']['rollouts']} / {count['clean']['rollouts']} rollouts, eigenvalue {e:.2f}, fit variance ratio {meta['var_ratio_fit']:.2f}, cosine " + ", ".join(f"{n} {c:+.3f}" for n, c in meta["cosine"].items()) + f"{endc}", flush=True)
    dirs = refs | {p.stem: load_file(p)["direction_unit"][ROW].float().cuda() for p in sorted(OUT.glob("*.safetensors"))}
    U = t.stack(list(dirs.values()))
    (OUT / "cosines.json").write_text(json.dumps(dict(names=list(dirs), cosine=(U @ U.T).tolist()), indent=1))

#%% diag
def spread(P: Tensor, ri: Tensor, grp: Tensor, sel: Tensor) -> dict:
    """Per direction over the tokens in sel: mean, variance and mean square of the projection, the variance split into between groups, between rollouts of a group and within a rollout,
    excess kurtosis, and the share of the squared deviations carried by the top 1% and 5% of tokens and by the top 5% of rollouts."""
    p, (ur, r) = P[sel].double(), ri[sel].unique(return_inverse=True)
    n, cnt, m = len(p), t.bincount(r).double(), p.mean(0)
    mr = t.zeros(len(ur), P.shape[1], device="cuda", dtype=t.float64).index_add_(0, r, p) / cnt[:, None]
    ug, g = grp[ur].unique(return_inverse=True)
    cg = t.zeros(len(ug), device="cuda", dtype=t.float64).index_add_(0, g, cnt)
    mg = t.zeros(len(ug), P.shape[1], device="cuda", dtype=t.float64).index_add_(0, g, mr * cnt[:, None]) / cg[:, None]
    d2 = (p - m).pow(2)
    var, srt, rs = d2.mean(0), d2.sort(0, descending=True).values, t.zeros_like(mr).index_add_(0, r, d2).sort(0, descending=True).values
    out = dict(mean=m, var=var, msq=p.pow(2).mean(0), between_groups=((mg - m).pow(2) * cg[:, None]).sum(0) / n, between_rollouts=((mr - mg[g]).pow(2) * cnt[:, None]).sum(0) / n, within=(p - mr[r]).pow(2).mean(0),
        kurtosis=d2.pow(2).mean(0) / var.pow(2) - 3, top1=srt[:n // 100].sum(0) / srt.sum(0), top5=srt[:n // 20].sum(0) / srt.sum(0), top5_rollouts=rs[:max(1, len(ur) // 20)].sum(0) / rs.sum(0))
    return {k: v.tolist() for k, v in out.items()} | dict(n_tokens=n, n_rollouts=len(ur), n_groups=len(ug))

def autocorr(P: Tensor, S: dict, sel: Tensor) -> dict:
    """Per lag, the correlation of a token's projection with the projection lag tokens later in its rollout, both minus the rollout's mean."""
    q = P - (t.zeros(len(S["n"]), P.shape[1], device="cuda").index_add_(0, S["ri"], P) / S["n"][:, None])[S["ri"]]
    out = {}
    for lag in LAGS:
        pair = sel[:-lag] & (S["ri"][:-lag] == S["ri"][lag:])
        out[lag] = ((q[:-lag][pair] * q[lag:][pair]).mean(0) / q[sel].pow(2).mean(0)).tolist()
    return out

def auroc(pos: Tensor, neg: Tensor) -> list:
    rank = t.cat([pos, neg]).argsort(0).argsort(0).double() + 1
    return ((rank[:len(pos)].sum(0) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))).tolist()

if STAGE == "diag":
    (ACTS / "vr").mkdir(exist_ok=True)
    dirs = refs | {p.stem: load_file(p)["direction_unit"][ROW].float().cuda() for p in sorted(OUT.glob("*.safetensors"))}
    names, U = list(dirs), t.stack(list(dirs.values()))
    B, R = load("mbpp_base"), load("branch")
    PB, PR = [t.cat([(S["X"][s:s + CHUNK].float() - pin) @ U.T for s in range(0, len(S["X"]), CHUNK)]) for S in [B, R]]
    sign = t.sign(PB[tokens(B, "cheat", True)].mean(0) - PB[tokens(B, "clean", True)].mean(0))   # every direction pointed toward cheating on the base fit split
    PB, PR = PB * sign, PR * sign
    t.save(dict(names=names, mbpp_base=PB.half().cpu(), branch=PR.half().cpu()), ACTS / "vr" / "proj.pt")   # every token's projection, tokens in the harvest's order
    group = lambda S: t.tensor([sorted({x["task"] for x in S["lab"]}).index(x["task"]) for x in S["lab"]]).cuda()   # a base rollout's task, a continuation's prefix
    reg = []
    for x in B["items"]:
        r = t.full((len(x["ids"]),), REGIONS.index("other"))
        for i in range(10): r[x["buckets"][f"cot_q{i}"]] = i
        r[x["buckets"]["answer_mean"]] = REGIONS.index("answer")
        r[x["buckets"]["code_mean"]] = REGIONS.index("code")
        reg.append(r[x["buckets"]["gen_mean"]])
    reg = t.cat(reg).cuda()
    cells = {f"{split}_{cls}": tokens(B, cls, split == "fit") for split in ["fit", "test"] for cls in ["cheat", "clean"]}
    m1, m2 = [t.zeros(len(B["n"]), len(names), device="cuda").index_add_(0, B["ri"], x) / B["n"][:, None] for x in [PB, PB.pow(2)]]   # per rollout: mean and mean square of the projection
    rc, rh = B["cheat"] & ~B["fit"], B["clean"] & ~B["fit"]
    out = dict(names=names, sign=sign.tolist(), row=ROW, regions=REGIONS, lags=LAGS,
        base={c: spread(PB, B["ri"], group(B), sel) for c, sel in cells.items()},
        branch={cls: spread(PR, R["ri"], group(R), tokens(R, cls, True)) for cls in ["cheat", "clean"]},
        base_regions={c: [dict(n=int(m.sum()), mean=PB[m].mean(0).tolist(), var=PB[m].var(0).tolist(), msq=PB[m].pow(2).mean(0).tolist()) for m in [sel & (reg == i) for i in range(len(REGIONS))]] for c, sel in cells.items()},
        base_autocorr={c: autocorr(PB, B, sel) for c, sel in cells.items()},
        auroc_test=dict(rollout_mean=auroc(m1[rc], m1[rh]), rollout_msq=auroc(m2[rc], m2[rh]), rollout_var=auroc((m2 - m1.pow(2))[rc], (m2 - m1.pow(2))[rh]), token=auroc(PB[cells["test_cheat"]], PB[cells["test_clean"]]),
            token_sq=auroc((PB[cells["test_cheat"]] - PB[cells["test_clean"]].mean(0)).pow(2), (PB[cells["test_clean"]] - PB[cells["test_clean"]].mean(0)).pow(2))))
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    tid = t.cat([x["ids"][x["buckets"]["gen_mean"]] for x in B["items"]]).long().cuda()
    sel = cells["test_cheat"].nonzero()[:, 0]
    z = (PB[sel] - PB[cells["test_clean"]].mean(0)) / PB[cells["test_clean"]].std(0)   # the cheating test tokens against the honest tokens' mean and sd
    rng, lines, out["top"] = t.Generator().manual_seed(SEED), [], {}
    def context(j: int) -> str:
        k = B["ri"][j].item()
        x, p = B["items"][k], B["items"][k]["buckets"]["gen_mean"][j - B["off"][k].item()]
        return f"{REGIONS[reg[j]]:>6} task {x['labels']['task']:>4} | " + repr(tok.decode(x["ids"][max(p - 48, 0):p]) + " ⟦" + tok.decode(x["ids"][p:p + 1]) + "⟧ " + tok.decode(x["ids"][p + 1:p + 13]))[1:-1]
    for k, name in enumerate(names):
        top = z[:, k].abs().topk(len(sel) // 100).indices
        count, base = t.bincount(tid[sel[top]], minlength=len(tok)), t.bincount(tid[sel], minlength=len(tok))
        common = [(tok.decode([i]), int(count[i]), round((count[i] / len(top) / (base[i] / len(sel))).item(), 2)) for i in count.topk(20).indices.tolist()]
        out["top"][name] = dict(region_share=(t.bincount(reg[sel[top]], minlength=len(REGIONS)) / len(top)).tolist(), region_share_all=(t.bincount(reg[sel], minlength=len(REGIONS)) / len(sel)).tolist(), positive_share=(z[top, k] > 0).float().mean().item(), common=common)
        lines += [f"\n=== {name}: top 1% of the cheating test tokens by |z|, {out['top'][name]['positive_share']:.0%} of them above the honest mean", "    by region (share of the top 1% / of all): " + ", ".join(f"{r} {a:.2f}/{b:.2f}" for r, a, b in zip(REGIONS, out["top"][name]["region_share"], out["top"][name]["region_share_all"])), "    commonest tokens (count, lift): " + ", ".join(f"{s!r} {c} x{l}" for s, c, l in common)]
        for side, n_show in [(1, 12), (-1, 8)]:
            cand = (side * z[:, k]).topk(len(sel) // 200).indices
            cand = cand[t.randperm(len(cand), generator=rng).cuda()]
            seen = set()
            for j, zj in zip(sel[cand].tolist(), z[cand, k].tolist()):
                if B["ri"][j].item() in seen or len(seen) == n_show: continue
                seen.add(B["ri"][j].item())
                lines.append(f"  z {zj:+6.1f} {context(j)}")
    (ACTS / "vr" / "top_tokens.txt").write_text("\n".join(lines))
    (ACTS / "vr" / "diag.json").write_text(json.dumps(out))
    b = out["base"]
    for k, name in enumerate(names):
        c, h = b["test_cheat"], b["test_clean"]
        print(f"{cyan}{name:>16}: test var cheat/honest {c['var'][k] / h['var'][k]:.2f}, mean square {c['msq'][k] / h['msq'][k]:.2f}; cheat variance: tasks {c['between_groups'][k] / c['var'][k]:.2f}, rollouts {c['between_rollouts'][k] / c['var'][k]:.2f}, within {c['within'][k] / c['var'][k]:.2f} (honest within {h['within'][k] / h['var'][k]:.2f}); within ratio {c['within'][k] / h['within'][k]:.2f}; top 1% share {c['top1'][k]:.2f} (honest {h['top1'][k]:.2f}); lag-1 / lag-16 autocorr {out['base_autocorr']['test_cheat'][1][k]:.2f} / {out['base_autocorr']['test_cheat'][16][k]:.2f}; rollout AUROC mean {out['auroc_test']['rollout_mean'][k]:.2f}, var {out['auroc_test']['rollout_var'][k]:.2f}{endc}")
