#%%
"""Every DoM variant on dom_hunt.py's harvest, scored by cosine against mbpp_kto (and its reseeds, their mean, grader_kto). Cells append rows to RESULTS
(one json line per recipe, set, bucket and row, with the cosines and a shuffled-label null where one applies); the summary cell prints the best of each
recipe family. Run on the box from /root/ortho: python dom_hunt_analyze.py [cell names...], no names runs every cell in order."""
import atexit
import json
import os
import sys
from pathlib import Path

import torch as t
from safetensors.torch import load_file

from mechtools import *

ACTS = Path("acts/dom_hunt")
RESULTS = ACTS / "results.jsonl"
VEC = Path("data/vectors/Qwen3.8-27B")
ONLY = [a[4:] for a in sys.argv[1:] if a.startswith("set=")]   # set=<name> arguments restrict every cell to those sets (for cross: those contrast names)
CELLS = [a for a in sys.argv[1:] if not a.startswith("set=")] or ["seeds", "label", "shuffle", "whiten", "whiten_null", "rollout", "rollout_null", "orth", "swap", "contrasts", "covdiff", "combine", "span", "varratio_sub", "spiky", "cross", "uview", "corr", "tokens", "lens", "summary"]
N_SHUFFLE = 20
SEED = 0
DEV = "cuda" if t.cuda.is_available() else "cpu"
t.set_num_threads(64)
gen = t.Generator().manual_seed(SEED)

#%% targets: one unit vector each, mbpp_kto and grader_kto (identical rows 31-64), the reseeds if trained, their mean
def row31(name: str) -> Tensor: return normed(load_file(VEC / f"{name}.safetensors")["direction_unit"][31].float())
targets = {"mbpp_kto": row31("mbpp_kto"), "grader_kto": row31("grader_kto")}
for s in [1, 2]:
    if (VEC / f"mbpp_kto_s{s}.safetensors").exists(): targets[f"mbpp_kto_s{s}"] = row31(f"mbpp_kto_s{s}")
seeds = [v for k, v in targets.items() if k.startswith("mbpp_kto")]
if len(seeds) > 1: targets["seed_mean"] = normed(sum(v * t.sign(v @ seeds[0]) for v in seeds))
T = t.stack(list(targets.values()))   # [n_targets, d]
TN = list(targets)
print(f"{cyan}targets: {TN}{endc}")

def cosines(v: Tensor) -> dict: return dict(zip(TN, (normed(v.float().cpu()) @ T.T).tolist()))
VECS = []   # every emitted unit vector, saved at exit to acts/dom_hunt/vecs/<pid>.pt so the summary can rescore against targets trained later
atexit.register(lambda: VECS and ((ACTS / "vecs").mkdir(exist_ok=True), t.save(t.stack(VECS), ACTS / f"vecs/{os.getpid()}.pt")))
def emit(**row):
    v = row.pop("v")
    row |= cosines(v) | dict(vid=f"{os.getpid()}:{len(VECS)}")
    VECS.append(normed(v.float().cpu()).half())
    with open(RESULTS, "a") as f: f.write(json.dumps(row) + "\n")
    return row

def have(name: str) -> bool: return (ACTS / f"{name}.pt").exists() and (not ONLY or name in ONLY)
def load(name: str) -> dict: return t.load(ACTS / f"{name}.pt", weights_only=False, mmap=True)
def classes(s: dict) -> tuple[Tensor, Tensor]:
    """pos: cheating rollouts; neg: the env's honest class (mbpp honest_correct, grader clean answer, tv not cheated). MBPP's test split is in neither, like in the harvest's moments."""
    lab = s["labels"]
    pos = t.tensor([x["cheat"] is True and x.get("split") != "test" for x in lab])
    neg = t.tensor([x["cheat"] is False and x.get("split") != "test" and (s["env"] != "mbpp" or x["honest_correct"]) for x in lab])
    return pos, neg
VOCAB = {}
def tasks(s: dict) -> Tensor: return t.tensor([VOCAB.setdefault(x["task"], len(VOCAB)) for x in s["labels"]])   # task ids shared across sets

def dom(X: Tensor, pos: Tensor, neg: Tensor) -> Tensor: return X[pos].mean(0) - X[neg].mean(0)
def group_means(X: Tensor, tid: Tensor, m: Tensor) -> tuple[Tensor, Tensor]:
    sums = t.zeros((len(VOCAB),) + X.shape[1:]).index_add_(0, tid[m], X[m])
    cnt = t.bincount(tid[m], minlength=len(VOCAB)).float()
    return sums / cnt.clamp(min=1).view(-1, *[1] * (X.dim() - 1)), cnt
def dom_paired(X: Tensor, pos: Tensor, neg: Tensor, tid: Tensor, Xn: Tensor = None, tidn: Tensor = None) -> Tensor:
    """Mean over tasks with both classes of the within-task difference of means; Xn, tidn give the negative class from another set."""
    mp, cp = group_means(X, tid, pos)
    mn, cn = group_means(X if Xn is None else Xn, tid if tidn is None else tidn, neg)
    both = (cp > 0) & (cn > 0)
    return (mp[both] - mn[both]).mean(0) if both.sum() >= 10 else t.zeros(X.shape[1:])

#%% seeds: the cosines among the targets
if "seeds" in CELLS:
    print(f"{cyan}cosines among targets{endc}")
    for i, a in enumerate(TN): print(f"  {a:<12} " + " ".join(f"{(T[i] @ T[j]).item():+.3f}" for j in range(len(TN))))

#%% label: per labelled set, bucket and row, pooled, task-paired, unit-normalised and median DoM of cheat minus honest; a row-averaged single direction per bucket
LABEL_SETS = ["swap_correct", "synth_ds", "prefill_start", "prefill_mid", "persona_prompt", "persona_text", "persona_inctx", "branch", "mbpp_base", "grader_main", "grader_heldout", "tv_base", "mbpp_hk_mbppkto", "mbpp_hk_mprojrand", "grader_hk_mbppkto", "grader_hk_mprojrand", "tv_hk_mbppkto", "mbpp_hk_kto", "grader_nothink_hack"]
if "label" in CELLS:
    for name in LABEL_SETS:
        if not have(name): continue
        s = load(name)
        pos, neg = classes(s)
        task = tasks(s)
        sizes = s["bucket_sizes"] > 0
        rowavg = {}
        for ri, r in enumerate(pbar(s["rows"], desc=name)):
            X = s["means"][:, :, ri].float()
            for bi, b in enumerate(s["buckets"]):
                ok = sizes[:, bi]
                if (pos & ok).sum() < 5 or (neg & ok).sum() < 5: continue
                Xb = X[:, bi]
                for recipe, v in [("dom", dom(Xb, pos & ok, neg & ok)), ("paired", dom_paired(Xb, pos & ok, neg & ok, task)), ("unit", dom(normed(Xb), pos & ok, neg & ok)), ("median", Xb[pos & ok].median(0).values - Xb[neg & ok].median(0).values)]:
                    if not v.any(): continue
                    emit(cell="label", recipe=recipe, set=name, bucket=b, row=r, v=v, n_pos=int((pos & ok).sum()), n_neg=int((neg & ok).sum()))
                    if r >= 30: rowavg.setdefault((recipe, b), []).append(normed(v))
        for (recipe, b), vs in rowavg.items(): emit(cell="label", recipe=f"{recipe}_rowavg", set=name, bucket=b, row=-1, v=sum(vs))
        print(f"{gray}{name}: {int(pos.sum())} cheat / {int(neg.sum())} honest{endc}")

#%% shuffle: the null for the pooled DoM, labels permuted within the pos+neg pool, the 95th percentile of |cos| with each target per (set, bucket, row)
if "shuffle" in CELLS:
    for name in ["mbpp_base", "grader_main", "tv_base"]:
        if not have(name): continue
        s = load(name)
        pos, neg = classes(s)
        pool = (pos | neg).nonzero()[:, 0]
        for ri, r in enumerate(pbar(s["rows"], desc=f"shuffle {name}")):
            X = s["means"][pool, :, ri].float()
            null = []
            for _ in range(N_SHUFFLE):
                perm = pos[pool][t.randperm(len(pool), generator=gen)]
                null.append(normed(X[perm].mean(0) - X[~perm].mean(0)) @ T.T)   # [buckets, targets]
            q = t.stack(null).abs().quantile(0.95, dim=0)
            for bi, b in enumerate(s["buckets"]):
                with open(RESULTS, "a") as f: f.write(json.dumps(dict(cell="shuffle", set=name, bucket=b, row=r, **{f"null95_{k}": q[bi, j].item() for j, k in enumerate(TN)})) + "\n")

#%% whiten: at the moment rows, the pooled within-class covariance of generated tokens; each bucket's DoM times (Σ + λ)^-1, and with the top-k principal directions removed
if "whiten" in CELLS:
    for name in ["mbpp_base", "grader_main", "tv_base"]:
        if not have(name): continue
        s = load(name)
        pos, neg = classes(s)
        task = tasks(s)
        mom = s["moments"]
        for j, r in enumerate(pbar(s["sub_rows"], desc=f"whiten {name}")):
            n = sum(m["n"] for m in mom.values())
            cov = sum(m["s2"][j].double() - t.outer(m["s1"][j], m["s1"][j]) / m["n"] for m in mom.values() if m["n"]) / n
            e, V = t.linalg.eigh(cov.to(DEV))
            e, V = e.flip(0).clamp(min=0), V.flip(1)
            ri = s["rows"].index(r)
            X = s["means"][:, :, ri].float()
            frac_u = ((V.T.float().cpu() @ T.T) ** 2)   # [d, targets]: each target's weight on each eigendirection
            with open(RESULTS, "a") as f: f.write(json.dumps(dict(cell="spectrum", set=name, row=r, eig_top=e[:8].tolist(), eig_mean=e.mean().item(), **{f"var_ratio_{k}": ((T[i].double().to(DEV) @ cov.to(DEV) @ T[i].double().to(DEV)) / e.mean()).item() for i, k in enumerate(TN)}, **{f"u_weight_top{k}_{nm}": frac_u[:k, i].sum().item() for k in [16, 256, 1024] for i, nm in enumerate(TN)})) + "\n")
            for bi, b in enumerate(s["buckets"]):
                ok = (s["bucket_sizes"][:, bi] > 0)
                if (pos & ok).sum() < 5 or (neg & ok).sum() < 5: continue
                for base_name, d0 in [("dom", dom(X[:, bi], pos & ok, neg & ok)), ("paired", dom_paired(X[:, bi], pos & ok, neg & ok, task))]:
                    c = V.T @ d0.double().to(DEV)
                    for a in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
                        emit(cell="whiten", recipe=f"{base_name}_whiten_{a:g}", set=name, bucket=b, row=r, v=(V @ (c / (e + a * e.mean()))).float())
                    for k in [1, 4, 16, 64, 256, 1024, 2048]:
                        emit(cell="whiten", recipe=f"{base_name}_pcrm_{k}", set=name, bucket=b, row=r, v=(V[:, k:] @ c[k:]).float())
            del cov, V, e
            if DEV == "cuda": t.cuda.empty_cache()

#%% whiten_null: the same whitening and PC removal applied to label-shuffled DoMs, the 95th percentile of |cos| per recipe, bucket and row
NULL_BUCKETS = ["cot_q5", "cot_q6", "cot_q7", "cot_mean", "gen_mean", "cot_last"]
if "whiten_null" in CELLS:
    for name in ["mbpp_base", "grader_main", "tv_base"]:
        if not have(name): continue
        s = load(name)
        pos, neg = classes(s)
        pool = (pos | neg).nonzero()[:, 0]
        mom = s["moments"]
        for j, r in enumerate(pbar(s["sub_rows"], desc=f"whiten_null {name}")):
            n = sum(m["n"] for m in mom.values())
            cov = sum(m["s2"][j].double() - t.outer(m["s1"][j], m["s1"][j]) / m["n"] for m in mom.values() if m["n"]) / n
            e, V = t.linalg.eigh(cov.to(DEV))
            e, V = e.flip(0).clamp(min=0), V.flip(1)
            X = s["means"][pool, :, s["rows"].index(r)].float()
            perms = [pos[pool][t.randperm(len(pool), generator=gen)] for _ in range(N_SHUFFLE)]
            for b in NULL_BUCKETS:
                bi = s["buckets"].index(b)
                D = t.stack([X[m, bi].mean(0) - X[~m, bi].mean(0) for m in perms]).double().to(DEV)   # [shuffles, d]
                C = D @ V
                out = {}
                for a in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]: out[f"dom_whiten_{a:g}"] = (C / (e + a * e.mean())) @ V.T
                for k in [1, 4, 16, 64, 256, 1024, 2048]: out[f"dom_pcrm_{k}"] = C[:, k:] @ V[:, k:].T
                with open(RESULTS, "a") as f:
                    for rec, W in out.items(): f.write(json.dumps(dict(cell="whiten_null", recipe=rec, set=name, bucket=b, row=r, **{f"null95_{k}": q for k, q in zip(TN, (normed(W.float()).cpu() @ T.T).abs().quantile(0.95, dim=0).tolist())})) + "\n")
            del cov, V, e

#%% rollout: at the moment rows, for the main buckets: top principal directions of the per-task paired differences, DoM whitened by the rollout-level
# within-class covariance of that bucket (shrunk), and an L2 logistic probe on the rollout means
ROLLOUT_BUCKETS = ["gen_mean", "cot_mean", "cot_last", "answer_mean", "prompt_last", "cot_q9", "think_close", "last"]
if "rollout" in CELLS:
    for name in ["mbpp_base", "grader_main", "tv_base", "synth_ds", "prefill_mid", "branch", "persona_text"]:
        if not have(name): continue
        s = load(name)
        pos, neg = classes(s)
        tid = tasks(s)
        for r in pbar(s["sub_rows"], desc=f"rollout {name}"):
            ri = s["rows"].index(r)
            for b in ROLLOUT_BUCKETS:
                bi = s["buckets"].index(b)
                ok = s["bucket_sizes"][:, bi] > 0
                p_, n_ = pos & ok, neg & ok
                if p_.sum() < 10 or n_.sum() < 10: continue
                X = s["means"][:, bi, ri].float()
                mp, cp = group_means(X, tid, p_)
                mn, cn = group_means(X, tid, n_)
                both = (cp > 0) & (cn > 0)
                if both.sum() >= 10:
                    D = mp[both] - mn[both]
                    _, _, Vh = t.linalg.svd(D, full_matrices=False)   # uncentred: pc1 is close to the paired mean
                    for k in range(3): emit(cell="rollout", recipe=f"paired_pc{k + 1}", set=name, bucket=b, row=r, v=Vh[k], n_tasks=int(both.sum()))
                Z = t.cat([X[p_] - X[p_].mean(0), X[n_] - X[n_].mean(0)]).to(DEV)
                G = Z @ Z.T / len(Z)   # Gram matrix: the within-class covariance's spectrum without forming d x d
                e, U = t.linalg.eigh(G)
                e, U = e.flip(0).clamp(min=0), U.flip(1)
                keep = e > e.max() * 1e-6
                V = (Z.T @ U[:, keep]) / (e[keep] * len(Z)).sqrt()   # [d, k] orthonormal eigenvectors of the covariance
                d0 = (X[p_].mean(0) - X[n_].mean(0)).to(DEV)
                c = V.T @ d0
                lam0 = e[keep].sum() / X.shape[1]   # the covariance's mean eigenvalue over all d
                for a in [1e-2, 1e-1, 1.0, 10.0]:
                    lam = a * lam0
                    emit(cell="rollout", recipe=f"rollcov_whiten_{a:g}", set=name, bucket=b, row=r, v=((V @ (c * (1 / (e[keep] + lam) - 1 / lam))) + d0 / lam).cpu())
                Xp = t.cat([X[p_], X[n_]]).to(DEV)
                y = t.cat([t.ones(int(p_.sum())), t.zeros(int(n_.sum()))]).to(DEV)
                mu, sd = Xp.mean(0), Xp.std(0) + 1e-3
                Zs = (Xp - mu) / sd
                for l2 in [1e-3, 1e-1]:
                    wv = t.zeros(Zs.shape[1], device=DEV, requires_grad=True)
                    bias = t.zeros(1, device=DEV, requires_grad=True)
                    opt = t.optim.LBFGS([wv, bias], max_iter=300, line_search_fn="strong_wolfe")
                    def closure():
                        opt.zero_grad()
                        loss = t.nn.functional.binary_cross_entropy_with_logits(Zs @ wv + bias, y) + l2 * (wv ** 2).sum()
                        loss.backward()
                        return loss
                    opt.step(closure)
                    emit(cell="rollout", recipe=f"rollout_probe_l2_{l2:g}", set=name, bucket=b, row=r, v=(wv.detach() / sd).cpu())

#%% orth: mbpp_base's paired cheat-minus-honest DoM with the salience direction removed, honest rollouts on the corrupted test minus rollouts on the correct
# test (paired by task), per bucket and row; and the salience direction alone
if "orth" in CELLS and have("mbpp_base") and (ACTS / "mbpp_correct.pt").exists():
    s, sc = load("mbpp_base"), load("mbpp_correct")
    pos, neg = classes(s)
    tid, tidc = tasks(s), tasks(sc)
    allc = t.ones(len(sc["labels"]), dtype=t.bool)
    for ri, r in enumerate(pbar(s["rows"], desc="orth")):
        if r < 28: continue
        X, Xc = s["means"][:, :, ri].float(), sc["means"][:, :, ri].float()
        for bi, b in enumerate(s["buckets"]):
            d = dom_paired(X[:, bi], pos, neg, tid)
            n1 = dom_paired(X[:, bi], neg, allc, tid, Xc[:, bi], tidc)
            if not (d.any() and n1.any()): continue
            u1 = normed(n1)
            emit(cell="orth", recipe="cheat_minus_salience", set="mbpp_base", bucket=b, row=r, v=d - (d @ u1) * u1, cos_d_salience=(normed(d) @ u1).item())
            emit(cell="orth", recipe="salience", set="mbpp_base", bucket=b, row=r, v=n1)

#%% swap: the same completion read under the corrupted prompt (mbpp_base) minus under the correct prompt (swap_correct), paired by rollout, for cheating and for honest
# completions, and the interaction (cheating minus honest of that prompt effect)
if "swap" in CELLS and have("swap_correct") and (ACTS / "mbpp_base.pt").exists():
    sw, s = load("swap_correct"), load("mbpp_base")
    where = {x["rollout_id"]: i for i, x in enumerate(s["labels"])}
    idx = t.tensor([where[x["rollout_id"]] for x in sw["labels"]])
    cheat = t.tensor([x["cheat"] for x in sw["labels"]])
    for ri, r in enumerate(pbar(sw["rows"], desc="swap")):
        diff = s["means"][idx, :, ri].float() - sw["means"][:, :, ri].float()   # [n, buckets, d]
        for bi, b in enumerate(sw["buckets"]):
            ok = (sw["bucket_sizes"][:, bi] > 0) & (s["bucket_sizes"][idx, bi] > 0)
            if ok.sum() < 20: continue
            dc, dh = diff[ok & cheat, bi].mean(0), diff[ok & ~cheat, bi].mean(0)
            emit(cell="swap", recipe="prompt_effect_cheat", set="swap", bucket=b, row=r, v=dc)
            emit(cell="swap", recipe="prompt_effect_honest", set="swap", bucket=b, row=r, v=dh)
            emit(cell="swap", recipe="interaction", set="swap", bucket=b, row=r, v=dc - dh)

#%% rollout_null: DoM whitened by the total rollout-level covariance of the bucket (one eigendecomposition per bucket and row), at more shrinkages, on the real
# labels and on N_SHUFFLE label shuffles (95th percentile of |cos|)
if "rollout_null" in CELLS:
    for name in ["mbpp_base", "grader_main", "tv_base", "synth_ds", "branch", "persona_text"]:
        if not have(name): continue
        s = load(name)
        pos, neg = classes(s)
        pool = (pos | neg).nonzero()[:, 0]
        for r in pbar(s["sub_rows"], desc=f"rollout_null {name}"):
            for b in ["gen_mean", "cot_mean", "cot_q9", "answer_mean", "cot_q6"]:
                bi = s["buckets"].index(b)
                ok = s["bucket_sizes"][pool, bi] > 0
                X = s["means"][pool, bi, s["rows"].index(r)].float()[ok].to(DEV)
                y = pos[pool][ok].to(DEV)
                if y.sum() < 10 or (~y).sum() < 10: continue
                Z = X - X.mean(0)
                e, U = t.linalg.eigh(Z @ Z.T / len(Z))
                keep = e > e.max() * 1e-6
                e, V = e[keep], (Z.T @ U[:, keep]) / (e[keep] * len(Z)).sqrt()
                labels = [y] + [y[t.randperm(len(y), generator=gen).to(DEV)] for _ in range(N_SHUFFLE)]
                D = t.stack([X[yy].mean(0) - X[~yy].mean(0) for yy in labels])   # [1 + shuffles, d]
                C = D @ V
                for a in [1.0, 3.0, 10.0, 30.0, 100.0]:
                    lam = a * e.sum() / X.shape[1]
                    W = (C * (1 / (e + lam) - 1 / lam)) @ V.T + D / lam
                    cs = normed(W.float()).cpu() @ T.T
                    emit(cell="rollout_null", recipe=f"rollcov_total_whiten_{a:g}", set=name, bucket=b, row=r, v=W[0], **{f"null95_{k}": q for k, q in zip(TN, cs[1:].abs().quantile(0.95, dim=0).tolist())})

#%% contrasts: other class definitions within one set, pooled and task-paired DoM per bucket and row (pos minus neg)
L = lambda k: (lambda x: x.get(k) is True)
NOT = lambda f: (lambda x: not f(x))
AND = lambda *fs: (lambda x: all(f(x) for f in fs))
CONTRASTS = [   # (set, name, pos, neg)
    ("mbpp_base", "wrong_vs_correct_honest", lambda x: x["valid"] and x["cheat"] is False and not x["honest_correct"], L("honest_correct")),
    ("mbpp_base", "hardcode_vs_other_cheat", AND(L("cheat"), L("hardcoding")), AND(L("cheat"), NOT(L("hardcoding")))),
    ("mbpp_base", "cheat_vs_wrong_honest", L("cheat"), lambda x: x["valid"] and x["cheat"] is False and not x["honest_correct"]),
    ("tv_base", "tampered_vs_clean", L("tampered"), lambda x: not x["cheat"]),
    ("tv_base", "peeked_vs_clean", AND(L("peeked"), NOT(L("tampered"))), lambda x: not x["cheat"] and not x["peeked"]),
    ("persona_text", "persona_hacker_vs_honest", L("persona"), NOT(L("persona"))),
    ("persona_inctx", "persona_hacker_vs_honest", L("persona"), NOT(L("persona"))),
    ("persona_text", "cheat_vs_honest_under_honest_persona", AND(L("cheat"), NOT(L("persona"))), AND(NOT(L("cheat")), NOT(L("persona")))),
    ("persona_text", "cheat_vs_honest_under_hacker_persona", AND(L("cheat"), L("persona")), AND(NOT(L("cheat")), L("persona"))),
    ("synth_ds", "hacker_vs_honest", lambda x: x["character"] == "hacker", lambda x: x["character"] == "honest"),
    ("synth_ds", "rationalizer_vs_honest", lambda x: x["character"] == "rationalizer", lambda x: x["character"] == "honest"),
    ("synth_ds", "hacker_vs_rationalizer", lambda x: x["character"] == "hacker", lambda x: x["character"] == "rationalizer"),
    ("branch", "cheat_vs_honest_from_cheat_source", AND(L("cheat"), L("source_cheat")), AND(NOT(L("cheat")), L("source_cheat"))),
    ("branch", "cheat_vs_honest_from_honest_source", AND(L("cheat"), NOT(L("source_cheat"))), AND(NOT(L("cheat")), NOT(L("source_cheat")))),
]
if "contrasts" in CELLS:
    for name, cname, fp, fn in CONTRASTS:
        if not have(name): continue
        s = load(name)
        pos, neg = t.tensor([bool(fp(x)) for x in s["labels"]]), t.tensor([bool(fn(x)) for x in s["labels"]])
        if pos.sum() < 10 or neg.sum() < 10: continue
        tid = tasks(s)
        for ri, r in enumerate(pbar(s["rows"], desc=f"{name}:{cname}")):
            X = s["means"][:, :, ri].float()
            for bi, b in enumerate(s["buckets"]):
                ok = s["bucket_sizes"][:, bi] > 0
                if (pos & ok).sum() < 10 or (neg & ok).sum() < 10: continue
                emit(cell="contrasts", recipe=f"{cname}:dom", set=name, bucket=b, row=r, v=dom(X[:, bi], pos & ok, neg & ok), n_pos=int((pos & ok).sum()), n_neg=int((neg & ok).sum()))
                v = dom_paired(X[:, bi], pos & ok, neg & ok, tid)
                if v.any(): emit(cell="contrasts", recipe=f"{cname}:paired", set=name, bucket=b, row=r, v=v)

#%% covdiff: at the moment rows, the generated-token covariance of cheating minus honest rollouts: its top eigenvectors by |eigenvalue|, and each target's own variance per class
if "covdiff" in CELLS:
    for name in ["mbpp_base", "grader_main", "tv_base"]:
        if not have(name): continue
        mom = load(name)["moments"]
        for j, r in enumerate(pbar(load(name)["sub_rows"], desc=f"covdiff {name}")):
            cov = {c: (m["s2"][j].double() - t.outer(m["s1"][j], m["s1"][j]) / m["n"]) / m["n"] for c, m in mom.items() if m["n"]}
            Dc = (cov["cheat"] - cov["clean"]).to(DEV)
            e, V = t.linalg.eigh(Dc)
            order = e.abs().argsort(descending=True)
            for k in range(5): emit(cell="covdiff", recipe=f"covdiff_ev{k + 1}", set=name, bucket="gen_tokens", row=r, v=V[:, order[k]].float().cpu(), eig=e[order[k]].item())
            for a in [0.01, 0.1, 1.0]:   # generalized eigenvectors of (cov_cheat, cov_clean + a * mean eig): the largest and smallest cheat/clean variance ratios
                B = cov["clean"].to(DEV) + a * cov["clean"].diagonal().mean() * t.eye(len(cov["clean"]), device=DEV, dtype=t.float64)
                Lc = t.linalg.cholesky(B)
                Li = t.linalg.solve_triangular(Lc, t.eye(len(B), device=DEV, dtype=t.float64), upper=False)
                g, W = t.linalg.eigh(Li @ cov["cheat"].to(DEV) @ Li.T)
                Vg = Li.T @ W   # columns: generalized eigenvectors, ascending ratio
                for k in range(3):
                    emit(cell="covdiff", recipe=f"varratio_top{k + 1}_{a:g}", set=name, bucket="gen_tokens", row=r, v=Vg[:, -1 - k].float().cpu(), ratio=g[-1 - k].item())
                    emit(cell="covdiff", recipe=f"varratio_bottom{k + 1}_{a:g}", set=name, bucket="gen_tokens", row=r, v=Vg[:, k].float().cpu(), ratio=g[k].item())
                del B, Lc, Li, W, Vg
            with open(RESULTS, "a") as f: f.write(json.dumps(dict(cell="classvar", set=name, row=r, **{f"var_{c}_{k}": (T[i].double() @ cov[c] @ T[i].double()).item() for c in cov for i, k in enumerate(TN)}, **{f"meanvar_{c}": cov[c].diagonal().mean().item() for c in cov})) + "\n")
            del Dc, V, e

#%% combine: unit-normalised cheat-minus-honest DoMs of several envs averaged, raw and whitened by each set's total rollout covariance (lambda 10x the mean eigenvalue)
COMBOS = [["mbpp_base", "grader_main"], ["mbpp_base", "grader_main", "tv_base"], ["mbpp_base", "grader_main", "grader_heldout"], ["mbpp_base", "synth_ds"], ["mbpp_base", "branch"], ["mbpp_base", "persona_text"]]
if "combine" in CELLS:
    cache = {}
    def whitened(X: Tensor, y: Tensor, a: float = 10.0) -> Tensor:
        X = X.to(DEV)
        Z = X - X.mean(0)
        e, U = t.linalg.eigh(Z @ Z.T / len(Z))
        keep = e > e.max() * 1e-6
        e, V = e[keep], (Z.T @ U[:, keep]) / (e[keep] * len(Z)).sqrt()
        d0 = X[y].mean(0) - X[~y].mean(0)
        lam = a * e.sum() / X.shape[1]
        return ((V @ ((V.T @ d0) * (1 / (e + lam) - 1 / lam))) + d0 / lam).cpu()
    for combo in COMBOS:
        if not all((ACTS / f"{n}.pt").exists() for n in combo): continue
        for n in combo: cache.setdefault(n, load(n))
        for r in [36, 40, 44, 48]:
            for b in ["gen_mean", "cot_mean", "cot_q6", "cot_q9", "answer_mean"]:
                raw, wh = [], []
                for n in combo:
                    s = cache[n]
                    pos, neg = classes(s)
                    bi, ri = s["buckets"].index(b), s["rows"].index(r)
                    ok = s["bucket_sizes"][:, bi] > 0
                    m = (pos | neg) & ok
                    X = s["means"][m, bi, ri].float()
                    raw.append(normed(dom(X, pos[m], neg[m])))
                    wh.append(normed(whitened(X, pos[m])))
                emit(cell="combine", recipe="mean_unit_dom", set="+".join(combo), bucket=b, row=r, v=sum(raw), parts=[(normed(v) @ T[0]).item() for v in raw])
                emit(cell="combine", recipe="mean_unit_whitened", set="+".join(combo), bucket=b, row=r, v=sum(wh), parts=[(normed(v) @ T[0]).item() for v in wh])
    del cache

#%% span: the oracle bound, ||P u|| for u's projection onto the span of every bucket's pooled and paired DoM at a row (and over rows), against the same number of
# label-shuffled DoMs; not a recipe, since u picks the combination
if "span" in CELLS:
    for name in ["mbpp_base", "grader_main", "tv_base", "synth_ds", "branch", "persona_text", "prefill_mid"]:
        if not have(name): continue
        s = load(name)
        pos, neg = classes(s)
        tid = tasks(s)
        pool = pos | neg
        allrows_real, allrows_null = [], []
        for ri, r in enumerate(pbar(s["rows"], desc=f"span {name}")):
            X = s["means"][:, :, ri].float()
            ok = (s["bucket_sizes"] > 0).all(0)
            real = [v for bi in ok.nonzero()[:, 0].tolist() for v in [dom(X[:, bi], pos, neg), dom_paired(X[:, bi], pos, neg, tid)] if v.any()]
            perm = pool.clone()
            yy = pos[pool][t.randperm(int(pool.sum()), generator=gen)]
            ps, ns = t.zeros_like(pos), t.zeros_like(pos)
            ps[pool], ns[pool] = yy, ~yy
            null = [v for bi in ok.nonzero()[:, 0].tolist() for v in [dom(X[:, bi], ps, ns), dom_paired(X[:, bi], ps, ns, tid)] if v.any()]
            def proj(vs):
                Q, _ = t.linalg.qr(t.stack(vs).T.double())
                return (Q.T @ T.T.double()).norm(dim=0).tolist()
            with open(RESULTS, "a") as f: f.write(json.dumps(dict(cell="span", set=name, row=r, k=len(real), real=dict(zip(TN, proj(real))), null=dict(zip(TN, proj(null))))) + "\n")
            if r >= 30: allrows_real += real; allrows_null += null
        with open(RESULTS, "a") as f: f.write(json.dumps(dict(cell="span", set=name, row=-1, k=len(allrows_real), real=dict(zip(TN, proj(allrows_real))), null=dict(zip(TN, proj(allrows_null))))) + "\n")

#%% varratio_sub: the top cheat/clean variance-ratio direction from the N_SUB stored tokens per rollout, real rollout labels against 5 shuffles (the moments
# cannot be reshuffled); also the stored-token estimate of each target's own ratio
if "varratio_sub" in CELLS:
    for name in ["mbpp_base", "synth_ds", "branch", "persona_text", "prefill_mid"]:
        if not have(name): continue
        s = load(name)
        pos, neg = classes(s)
        keep = (pos | neg).nonzero()[:, 0]
        valid = s["sub_pos"][keep] >= 0
        for j, r in enumerate(pbar(s["sub_rows"], desc=f"varratio_sub {name}")):
            if r not in (36, 40, 44, 48): continue
            Xr = s["sub"][keep, :, j].float().to(DEV)   # [n, N_SUB, d]
            labs = [pos[keep]] + [pos[keep][t.randperm(len(keep), generator=gen)] for _ in range(5)]
            for li, y in enumerate(labs):
                def cov(m):
                    X = Xr[m.to(DEV)][valid[m].to(DEV)].double()
                    X = X - X.mean(0)
                    return X.T @ X / len(X)
                Cc, Ch = cov(y), cov(~y)
                for a in [0.1, 1.0]:
                    B = Ch + a * Ch.diagonal().mean() * t.eye(len(Ch), device=DEV, dtype=t.float64)
                    Li = t.linalg.solve_triangular(t.linalg.cholesky(B), t.eye(len(B), device=DEV, dtype=t.float64), upper=False)
                    g, W = t.linalg.eigh(Li @ Cc @ Li.T)
                    emit(cell="varratio_sub", recipe=f"varratio_top1_{a:g}", set=name, bucket="gen_tokens", row=r, v=(Li.T @ W[:, -1]).float().cpu(), shuffled=li > 0, ratio=g[-1].item(), **{f"u_ratio_{k}": ((T[i].double().to(DEV) @ Cc @ T[i].double().to(DEV)) / (T[i].double().to(DEV) @ Ch @ T[i].double().to(DEV))).item() for i, k in enumerate(TN)})
            del Xr

#%% spiky: per rollout, statistics of its stored tokens' projections on each target (mean, min, max, std, fraction beyond -2 sd of the pooled tokens), AUROC cheat vs honest
if "spiky" in CELLS:
    for name in ["mbpp_base", "grader_main", "tv_base"]:
        if not have(name): continue
        s = load(name)
        pos, neg = classes(s)
        keep = (pos | neg).nonzero()[:, 0]
        valid = (s["sub_pos"][keep] >= 0).float()
        y = pos[keep]
        for j, r in enumerate(s["sub_rows"]):
            if r not in (36, 48): continue
            P = s["sub"][keep, :, j].float() @ T.T   # [n, N_SUB, targets]
            for ti, tn in enumerate(TN):
                x = P[..., ti]
                mu, sd = (x * valid).sum() / valid.sum(), x[valid.bool()].std()
                stats = dict(mean=(x * valid).sum(1) / valid.sum(1), min=x.masked_fill(valid == 0, 1e9).min(1).values, max=x.masked_fill(valid == 0, -1e9).max(1).values, std=t.stack([x[i][valid[i].bool()].std() for i in range(len(x))]), low_tail=((x < mu - 2 * sd).float() * valid).sum(1) / valid.sum(1), high_tail=((x > mu + 2 * sd).float() * valid).sum(1) / valid.sum(1))
                au = {k: (v[y][:, None] > v[~y][None]).float().mean().item() for k, v in stats.items()}
                with open(RESULTS, "a") as f: f.write(json.dumps(dict(cell="spiky", set=name, row=r, target=tn, auroc=au)) + "\n")

#%% cross: contrasts between sets, per bucket and row, pooled and task-paired where both sets share tasks
CROSS = [   # (name, set A, filter A, set B, filter B): A minus B
    ("corrupt_vs_correct", "mbpp_base", None, "mbpp_correct", None),
    ("corrupt_honest_vs_correct", "mbpp_base", "honest", "mbpp_correct", None),
    ("corrupt_cheat_vs_correct", "mbpp_base", "cheat", "mbpp_correct", None),
    ("hk_mbppkto_vs_base_mbpp", "mbpp_hk_mbppkto", None, "mbpp_base", None),
    ("hk_mprojrand_vs_base_mbpp", "mbpp_hk_mprojrand", None, "mbpp_base", None),
    ("hk_kto_vs_base_mbpp", "mbpp_hk_kto", None, "mbpp_base", None),
    ("hk_mproj_vs_base_mbpp", "mbpp_hk_mproj", None, "mbpp_base", None),
    ("hk_mbppkto_vs_base_correct", "mbpp_hk_mbppkto_correct", None, "mbpp_correct", None),
    ("hk_mbppkto_honest_vs_base_honest", "mbpp_hk_mbppkto", "honest", "mbpp_base", "honest"),
    ("hk_mbppkto_vs_base_grader", "grader_hk_mbppkto", None, "grader_heldout", None),
    ("hk_mprojrand_vs_base_grader", "grader_hk_mprojrand", None, "grader_heldout", None),
    ("hk_mbppkto_clean_vs_base_clean_grader", "grader_hk_mbppkto", "honest", "grader_heldout", "honest"),
    ("hk_mbppkto_vs_base_tv", "tv_hk_mbppkto", None, "tv_base", None),
    ("grader_hack_vs_none", "grader_main", None, "grader_none", None),
    ("grader_hack_vs_clean", "grader_main", None, "grader_clean", None),
    ("grader_clean_vs_none", "grader_clean", None, "grader_none", None),
    ("grader_hack_clean_vs_clean_prompt", "grader_main", "honest", "grader_clean", None),
    ("grader_nothink_hack_vs_clean_prompt", "grader_nothink_hack", None, "grader_nothink_clean", None),
]
if "cross" in CELLS:
    cache = {}
    for name, a, fa, b, fb in CROSS:
        if not ((ACTS / f"{a}.pt").exists() and (ACTS / f"{b}.pt").exists()) or (ONLY and name not in ONLY): continue
        sa, sb = cache.setdefault(a, load(a)), cache.setdefault(b, load(b))
        def sel(s, f):
            pos, neg = classes(s)
            return t.ones(len(pos), dtype=t.bool) if f is None else pos if f == "cheat" else neg
        ma, mb = sel(sa, fa), sel(sb, fb)
        ta, tb = tasks(sa), tasks(sb)
        for ri, r in enumerate(pbar(sa["rows"], desc=name)):
            Xa, Xb = sa["means"][:, :, ri].float(), sb["means"][:, :, ri].float()
            for bi, bk in enumerate(sa["buckets"]):
                oka, okb = ma & (sa["bucket_sizes"][:, bi] > 0), mb & (sb["bucket_sizes"][:, bi] > 0)
                if oka.sum() < 5 or okb.sum() < 5: continue
                emit(cell="cross", recipe="dom", set=name, bucket=bk, row=r, v=Xa[oka, bi].mean(0) - Xb[okb, bi].mean(0), n_pos=int(oka.sum()), n_neg=int(okb.sum()))
                v = dom_paired(Xa[:, bi], oka, okb, ta, Xb[:, bi], tb)
                if v.any(): emit(cell="cross", recipe="paired", set=name, bucket=bk, row=r, v=v)
    del cache

#%% uview: each target's own reading, the projection of each bucket mean on it per set and row; AUROC and standardised gap between cheat and honest
if "uview" in CELLS:
    for name in LABEL_SETS + ["mbpp_correct", "grader_none", "grader_clean", "mbpp_hk_mbppkto_correct", "grader_nothink_clean"]:
        if not have(name): continue
        s = load(name)
        pos, neg = classes(s)
        for ri, r in enumerate(s["rows"]):
            P = s["means"][:, :, ri].float() @ T.T   # [n, buckets, targets]
            for bi, b in enumerate(s["buckets"]):
                ok = s["bucket_sizes"][:, bi] > 0
                row = dict(cell="uview", set=name, bucket=b, row=r, mean={k: P[ok, bi, j].mean().item() for j, k in enumerate(TN)}, std={k: P[ok, bi, j].std().item() for j, k in enumerate(TN)})
                if (pos & ok).sum() >= 5 and (neg & ok).sum() >= 5:
                    pp, pn = P[pos & ok, bi], P[neg & ok, bi]
                    row["auroc"] = {k: (pp[:, j][:, None] > pn[:, j][None]).float().mean().item() for j, k in enumerate(TN)}
                    row["dprime"] = {k: ((pp[:, j].mean() - pn[:, j].mean()) / ((pp[:, j].var() + pn[:, j].var()) / 2).sqrt()).item() for j, k in enumerate(TN)}
                with open(RESULTS, "a") as f: f.write(json.dumps(row) + "\n")

#%% corr: per rollout, the gen_mean projection on each target at rows 36 and 48 against every boolean or numeric label and the length (Pearson r)
if "corr" in CELLS:
    for name in LABEL_SETS + ["mbpp_correct", "synth_ds", "persona_text", "persona_inctx", "branch", "prefill_mid", "prefill_start"]:
        if not have(name): continue
        s = load(name)
        bi = s["buckets"].index("gen_mean")
        keys = sorted({k for x in s["labels"] for k, v in x.items() if isinstance(v, (bool, int, float)) and k not in ("rollout_id",)})
        for r in [36, 48]:
            P = s["means"][:, bi, s["rows"].index(r)].float() @ T.T
            out = {}
            for k in keys + ["length"]:
                v = t.tensor([float(len_) if k == "length" else float(x[k]) if isinstance(x.get(k), (bool, int, float)) else float("nan") for x, len_ in zip(s["labels"], s["lengths"])])
                m = ~v.isnan()
                if m.sum() < 20 or v[m].std() == 0: continue
                out[k] = {tn: pearson(P[m, j], v[m]) for j, tn in enumerate(TN)}
            with open(RESULTS, "a") as f: f.write(json.dumps(dict(cell="corr", set=name, row=r, corr=out)) + "\n")

#%% tokens: single generated tokens at the moment rows: pooled token DoM, token-identity-matched DoM, an L2 logistic probe; per-token-id mean projection on mbpp_kto
if "tokens" in CELLS:
    for name in ["mbpp_base", "grader_main", "tv_base"]:
        if not have(name): continue
        s = load(name)
        pos, neg = classes(s)
        keep = (pos | neg).nonzero()[:, 0]
        ids = s["sub_ids"][keep]
        valid = s["sub_pos"][keep] >= 0
        y = pos[keep][:, None].expand_as(valid)
        ids_f, y_f = ids[valid], y[valid]
        for j, r in enumerate(pbar(s["sub_rows"], desc=f"tokens {name}")):
            X = s["sub"][keep, :, j].float()[valid]   # [n_tokens, d]
            emit(cell="tokens", recipe="token_dom", set=name, bucket="gen_tokens", row=r, v=X[y_f].mean(0) - X[~y_f].mean(0), n_tokens=len(X))
            diffs, w = [], []
            for k in ids_f.unique():
                m = ids_f == k
                if (m & y_f).sum() >= 3 and (m & ~y_f).sum() >= 3:
                    diffs.append(X[m & y_f].mean(0) - X[m & ~y_f].mean(0))
                    w.append(min((m & y_f).sum(), (m & ~y_f).sum()).item())
            if diffs:
                emit(cell="tokens", recipe="token_id_matched", set=name, bucket="gen_tokens", row=r, v=(t.stack(diffs) * t.tensor(w, dtype=t.float)[:, None]).sum(0) / sum(w), n_ids=len(diffs))
                ks = [k.item() for k in ids_f.unique() if ((ids_f == k) & y_f).sum() >= 3 and ((ids_f == k) & ~y_f).sum() >= 3]
                each = [(abs(c), c, k, n) for c, k, n in zip((normed(t.stack(diffs)) @ T[0]).tolist(), ks, w) if n >= 25]
                with open(RESULTS, "a") as f: f.write(json.dumps(dict(cell="token_id_each", set=name, row=r, top=sorted(each, reverse=True)[:30])) + "\n")
            Xg = X.to(DEV)
            mu, sd = Xg.mean(0), Xg.std(0) + 1e-3
            Z = (Xg - mu) / sd
            for l2 in [1e-4, 1e-2]:
                wv = t.zeros(Z.shape[1], device=DEV, requires_grad=True)
                bias = t.zeros(1, device=DEV, requires_grad=True)
                opt = t.optim.LBFGS([wv, bias], max_iter=200, line_search_fn="strong_wolfe")
                yt = y_f.float().to(DEV)
                def closure():
                    opt.zero_grad()
                    loss = t.nn.functional.binary_cross_entropy_with_logits(Z @ wv + bias, yt) + l2 * (wv ** 2).sum()
                    loss.backward()
                    return loss
                opt.step(closure)
                emit(cell="tokens", recipe=f"probe_l2_{l2:g}", set=name, bucket="gen_tokens", row=r, v=(wv.detach() / sd).cpu())   # the probe's direction in residual coordinates
            if name == "mbpp_base" and r in (36, 48):
                proj = X @ T[0]
                stats = []
                for k in ids_f.unique():
                    m = ids_f == k
                    if m.sum() >= 30: stats.append((proj[m].mean().item(), int(m.sum()), int(k)))
                stats.sort()
                with open(RESULTS, "a") as f: f.write(json.dumps(dict(cell="token_proj", set=name, row=r, overall=proj.mean().item(), std=proj.std().item(), low=stats[:40], high=stats[-40:])) + "\n")
            del Xg, Z
            if DEV == "cuda": t.cuda.empty_cache()

#%% lens: the Qwen3.6 J-lens as a stand-in for what later layers read: cos(J_r w, J_r u) for the label-cell DoMs at the moment rows, and J_r^T J_r applied to the DoM
if "lens" in CELLS:
    lens = t.load(hf_hub_download("camilablank/workspace-lenses", "qwen3.6-27b/j-lens/lens.pt"), map_location="cpu", weights_only=False, mmap=True)
    for name in ["mbpp_base", "grader_main", "tv_base"]:
        if not have(name): continue
        s = load(name)
        pos, neg = classes(s)
        task = tasks(s)
        for r in pbar([r for r in s["sub_rows"] if r <= 62], desc=f"lens {name}"):
            J = lens["J"][r].float().to(DEV)
            JT = normed((T.to(DEV) @ J.T))   # [targets, d]: J_r u for each target
            ri = s["rows"].index(r)
            X = s["means"][:, :, ri].float()
            for bi, b in enumerate(s["buckets"]):
                ok = s["bucket_sizes"][:, bi] > 0
                if (pos & ok).sum() < 5 or (neg & ok).sum() < 5: continue
                d0 = dom(X[:, bi], pos & ok, neg & ok).to(DEV)
                Jd = normed(J @ d0)
                with open(RESULTS, "a") as f: f.write(json.dumps(dict(cell="lens", recipe="lens_space_cos", set=name, bucket=b, row=r, **dict(zip(TN, (JT @ Jd).tolist())))) + "\n")
                emit(cell="lens", recipe="jtj_dom", set=name, bucket=b, row=r, v=(J.T @ (J @ d0)).cpu())
            del J

#%% summary: per cell and recipe family, the best |cos| with each target over sets, buckets and rows, and the overall top rows
if "summary" in CELLS:
    rows = [json.loads(l) for l in open(RESULTS)]
    saved = {p.stem: t.load(p) for p in (ACTS / "vecs").glob("*.pt")}
    for r in rows:   # rescore against the current targets where the vector was saved
        if "vid" in r and r["vid"].split(":")[0] in saved: r |= dict(zip(TN, (saved[r["vid"].split(":")[0]][int(r["vid"].split(":")[1])].float() @ T.T).tolist()))
    scored = [r for r in rows if "mbpp_kto" in r and r["cell"] not in ("shuffle", "spectrum", "uview", "corr", "token_proj") and r.get("recipe") != "lens_space_cos"]
    print(f"{cyan}{len(scored)} scored rows{endc}")
    fam = lambda r: f"{r['cell']}:{re.sub(r'_[0-9.e+-]+$', '', r['recipe'])}"
    best = {}
    for r in scored:
        k = fam(r)
        if k not in best or abs(r["mbpp_kto"]) > abs(best[k]["mbpp_kto"]): best[k] = r
    for k, r in sorted(best.items(), key=lambda kv: -abs(kv[1]["mbpp_kto"])):
        print(f"  {k:<34} {r['mbpp_kto']:+.3f}  {r['set']:<24} {r['bucket']:<12} row {r['row']:>3} {r['recipe']:<24} " + " ".join(f"{n} {r[n]:+.3f}" for n in TN[1:] if n in r))
    print(f"{cyan}top 25 overall by |cos mbpp_kto|{endc}")
    for r in sorted(scored, key=lambda r: -abs(r["mbpp_kto"]))[:25]:
        print(f"  {r['mbpp_kto']:+.3f} {r['cell']:<7} {r['recipe']:<24} {r['set']:<24} {r['bucket']:<12} row {r['row']:>3} " + " ".join(f"{n} {r[n]:+.3f}" for n in TN[1:] if n in r))
    null = {(r["set"], r["bucket"], r["row"]): r for r in rows if r["cell"] == "shuffle"}
    if null: print(f"{gray}shuffle null 95th pct of |cos mbpp_kto|: median {t.tensor([r['null95_mbpp_kto'] for r in null.values()]).median():.4f}, max {max(r['null95_mbpp_kto'] for r in null.values()):.4f}{endc}")
