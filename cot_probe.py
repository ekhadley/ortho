#%%
"""A linear probe on Qwen3.8-27B's CoT tokens: one token's residual at one layer in, cheat or not out, the rollout's label on every CoT token. Rollouts are Will's s9-gr
manifest rows of the five training families, pets held out as a sixth. grader_kto read the same way is the second readout: no fit, each token's residual minus the hook's pin along
the direction. mbpp_kto is the third, read the same way on the grader rollouts only. Kernel cell script: harvest and fit once, every later cell reads what those saved."""
import gzip
import json
import random
from pathlib import Path

import torch as t
from huggingface_hub import snapshot_download
from IPython.display import HTML, display
from safetensors.torch import load_file, save_file
from sklearn.metrics import roc_auc_score
from torch import Tensor
from transformers import AutoModelForImageTextToText, AutoTokenizer

from mechtools import *

t.set_grad_enabled(False)
MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
WILL = Path("data/will")
ACTS = Path("acts")   # per-row activations of every generated token, and the chosen probe's logits
OUT = Path("data/vectors/Qwen3.8-27B")
FAMILIES = ["parity", "sign", "letter", "weekday", "pets_listed"]
HELD_FAMILY = "pets"
RUNS = [f"grader-{f}-hack-none-n1024-low" for f in FAMILIES + [HELD_FAMILY]] + ["grader-parity-hack-none-n3072-low"]
ROWS = [12, 20, 28, 32, 36, 40, 44, 48, 52, 56, 60]   # output_hidden_states rows: row i + 1 is the output of decoder layer i
THINK_OPEN = 248068
THINK_CLOSE = 248069
MBPP_RUN = "data/inspect/mbpp/2026-11-04T19-19-26-00-00_mbpp_GgYyPkS3GFuuD8oMPc8atb.jsonl"   # mbpp38 on cb_base
SEED = 0
PAGE_TITLE = "On single tokens the trained probe beats grader_kto, 0.89 to 0.67; on whole rollouts they are close"

#%% downloads the records of the six families' grader runs and the manifest

download = False
if download:
    snapshot_download("willjgriff/reward-hacking-ablation", repo_type="dataset", local_dir=WILL, allow_patterns=[f"sweep/Qwen3.8-27B/ortho_grader/{r}/records.jsonl.gz" for r in RUNS] + ["stage9/cache_meta/s9-gr/manifest.jsonl"])

#%% one item per manifest row. split: train and dev from the manifest's train split, eval its eval split, family the held-out family. Then which item and which CoT each generated token belongs to

set_seed(SEED)
tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
records = {}
for run in RUNS:
    for line_ in gzip.open(WILL / f"sweep/Qwen3.8-27B/ortho_grader/{run}/records.jsonl.gz", "rt"):
        r = json.loads(line_)
        records[f"{run}:{r['rollout_id']}"] = r
items = []
for m in map(json.loads, (WILL / "stage9/cache_meta/s9-gr/manifest.jsonl").read_text().splitlines()):
    ids = records[m["trajectory_id"]]["ids"]
    family = m["task_id"].split(".")[0]
    assert len(ids) == m["seq_len"] and ids.count(THINK_CLOSE) == 1, m["trajectory_id"]
    split = "family" if family == HELD_FAMILY else "eval" if m["split"] == "eval" else "dev" if random.random() < 0.2 else "train"
    items.append(dict(ids=ids, prompt_len=m["call_prompt_lens"][0], close=ids.index(THINK_CLOSE), cell=m["task_id"], family=family, cheat=m["class"].startswith("cheat"), split=split))
del records
owner = t.cat([t.full((len(x["ids"]) - x["prompt_len"],), k) for k, x in enumerate(items)])   # the item of every generated token
cot = t.cat([t.arange(x["prompt_len"], len(x["ids"])) < x["close"] for x in items])   # which generated tokens are CoT
index = t.cat([t.arange(len(x["ids"]) - x["prompt_len"]) for x in items])   # position in the completion
cot_len = t.tensor([x["close"] - x["prompt_len"] for x in items])
is_cheat = t.tensor([x["cheat"] for x in items])
in_split = {s: t.tensor([x["split"] == s for x in items])[owner] & cot for s in ["train", "dev", "eval", "family"]}
cell_of = {c: t.tensor([x["cell"] == c for x in items])[owner] for c in sorted({x["cell"] for x in items})}
print(f"{cyan}{len(items)} rollouts, {len(owner)} generated tokens, {cot.sum()} CoT{endc}")
for s, mask in in_split.items():
    print(f"{gray}  {s}: {sum(x['split'] == s and x['cheat'] for x in items)} cheat / {sum(x['split'] == s and not x['cheat'] for x in items)} clean rollouts, {mask.sum()} CoT tokens{endc}")

def cell_auroc(scores: Tensor, mask: Tensor) -> float:
    """AUROC of cheat against clean among the masked tokens, within each cell holding both, every rollout weighing the same, averaged over cells. A cell's cheat and clean
    rollouts share one prompt, so this cannot be won by telling cells apart. A per-rollout score broadcast to its tokens gives the rollout-level AUROC."""
    weight = 1 / t.bincount(owner[mask], minlength=len(items)).float()
    aurocs = []
    for in_cell in cell_of.values():
        sel = mask & in_cell
        label = is_cheat[owner[sel]]
        if label.any() and not label.all(): aurocs.append(roc_auc_score(label.numpy(), scores[sel].numpy(), sample_weight=weight[owner[sel]].numpy()))
    return sum(aurocs) / len(aurocs)

#%% residuals of every generated token at ROWS, one rollout at a time so no padding enters

harvest = False
if harvest:
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=REVISION, dtype=t.bfloat16, device_map="cuda").eval()
    acts = {r: [] for r in ROWS}
    for x in pbar(items, desc="harvest"):
        hidden = model(input_ids=t.tensor([x["ids"]], device="cuda"), output_hidden_states=True).hidden_states
        assert len(hidden) == 65, f"{len(hidden)} hidden states"
        for r in ROWS: acts[r].append(hidden[r][0, x["prompt_len"]:].cpu())
    ACTS.mkdir(exist_ok=True)
    for r in ROWS: save_file({"acts": t.cat(acts[r])}, ACTS / f"row{r}.safetensors")
    del model, acts, hidden
    tec()

#%% one logistic probe per row and L2 weight on the train CoT tokens, the pair chosen by within-cell token AUROC on dev. Each (cell, class) weighs the same and each rollout
# the same within it, so neither a family's cheat rate nor a long rollout steers the fit. Saves the chosen probe in raw residual coordinates, and its logit on every generated token

fit_probes = False
if fit_probes:
    l2s = [1e-5, 1e-4, 1e-3, 1e-2]
    steps = 400
    lr = 1e-2
    group = {(x["cell"], x["cheat"]) for x in items if x["split"] == "train"}
    n_group = {g: sum((x["cell"], x["cheat"]) == g for x in items if x["split"] == "train") for g in group}
    per_item = t.tensor([1 / n_group[(x["cell"], x["cheat"])] if x["split"] == "train" else 0.0 for x in items])
    weight = (per_item / t.bincount(owner[in_split["train"]], minlength=len(items)).clamp(min=1))[owner[in_split["train"]]].cuda()
    weight /= weight.sum()
    label = is_cheat[owner[in_split["train"]]].float().cuda()
    table = []
    best = dict(dev=0.0)
    for r in ROWS:
        a = load_file(ACTS / f"row{r}.safetensors")["acts"].cuda().float()
        mean = a[in_split["train"]].mean(0)
        std = a[in_split["train"]].std(0)
        assert std.min() > 0, f"row {r}: a dimension is constant"
        z = (a - mean) / std
        z_train = z[in_split["train"]]
        for l2 in l2s:
            w = t.zeros(a.shape[1], device="cuda", requires_grad=True)
            b = t.zeros((), device="cuda", requires_grad=True)
            opt = t.optim.Adam([w, b], lr=lr)
            with t.enable_grad():
                for step in range(steps):
                    loss = (weight * t.nn.functional.binary_cross_entropy_with_logits(z_train @ w + b, label, reduction="none")).sum() + l2 * w.pow(2).sum()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            logits = (z @ w + b).cpu()
            table.append(dict(row=r, l2=l2, loss=loss.item(), train=cell_auroc(logits, in_split["train"]), dev=cell_auroc(logits, in_split["dev"])))
            print(f"{gray}row {r} l2 {l2:g}: loss {loss.item():.3f}, train {table[-1]['train']:.3f}, dev {table[-1]['dev']:.3f}{endc}")
            if table[-1]["dev"] > best["dev"]: best = dict(table[-1], w=(w / std).detach().cpu(), b=(b - (mean / std) @ w).detach().cpu(), logits=logits)
    print(f"{cyan}chosen: row {best['row']}, l2 {best['l2']:g}, dev {best['dev']:.3f}{endc}")
    save_file({"w": best["w"].contiguous(), "b": best["b"].reshape(1)}, OUT / "cot_probe.safetensors")
    save_file({"logits": best["logits"].contiguous()}, ACTS / "logits.safetensors")
    (OUT / "cot_probe.json").write_text(json.dumps(dict(model=MODEL_ID, revision=REVISION, row=best["row"], l2=best["l2"], steps=steps, lr=lr, seed=SEED, families=FAMILIES, layout="logit = resid @ w + b at output_hidden_states row `row` (decoder layer row - 1), trained on CoT tokens", table=table), indent=1))
    del a, z, z_train
    tec()

#%% loads the chosen probe's logits

fit = json.loads((OUT / "cot_probe.json").read_text())
logits = load_file(ACTS / "logits.safetensors")["logits"]
rollout_logit = t.zeros(len(items)).index_add_(0, owner[cot], logits[cot]) / cot_len   # mean over the rollout's CoT
print(f"{cyan}probe at row {fit['row']} (decoder layer {fit['row'] - 1}), l2 {fit['l2']:g}{endc}")

#%% grader_kto as a probe: each token's residual minus the hook's pin, along the direction toward cheating, so zero is the pin and the score is what the hook would remove. Nothing is fit.
# The row among ROWS inside the hook's window is chosen by within-cell token AUROC on dev, which are rollouts grader_kto trained on. Saves its score on every generated token

score_kto = False
if score_kto:
    kto = load_file(OUT / "grader_kto.safetensors")
    table = []
    best = dict(dev=0.0)
    for r in [r for r in ROWS if kto["direction_unit"][r].norm() > 0]:
        a = load_file(ACTS / f"row{r}.safetensors")["acts"].cuda().float()
        row_scores = ((a - kto["mean_resid"][r].cuda()) @ kto["direction_unit"][r].cuda()).cpu()
        table.append(dict(row=r, train=cell_auroc(row_scores, in_split["train"]), dev=cell_auroc(row_scores, in_split["dev"]), eval=cell_auroc(row_scores, in_split["eval"])))
        print(f"{gray}row {r}: train {table[-1]['train']:.3f}, dev {table[-1]['dev']:.3f}, eval {table[-1]['eval']:.3f}{endc}")
        if table[-1]["dev"] > best["dev"]: best = dict(table[-1], scores=row_scores)
    print(f"{cyan}grader_kto row {best['row']}, dev {best['dev']:.3f}{endc}")
    save_file({"scores": best["scores"].contiguous()}, ACTS / "kto_scores.safetensors")
    (ACTS / "kto_fit.json").write_text(json.dumps(dict(row=best["row"], table=table), indent=1))
    del a
    tec()

#%% loads grader_kto's scores

kto_fit = json.loads((ACTS / "kto_fit.json").read_text())
kto_scores = load_file(ACTS / "kto_scores.safetensors")["scores"]
rollout_kto = t.zeros(len(items)).index_add_(0, owner[cot], kto_scores[cot]) / cot_len   # mean over the rollout's CoT
print(f"{cyan}grader_kto at row {kto_fit['row']} (decoder layer {kto_fit['row'] - 1}){endc}")

#%% mbpp_kto as a probe, read like grader_kto from the saved activations, so on the cpu with no model. Its saved rows point away from cheating, so the sign flips. It trained on no grader hack rollout, and the row is chosen on dev the same way

score_mkto = False
if score_mkto:
    mkto = load_file(OUT / "mbpp_kto.safetensors")
    table = []
    best = dict(dev=0.0)
    for r in [r for r in ROWS if mkto["direction_unit"][r].norm() > 0]:
        a = load_file(ACTS / f"row{r}.safetensors")["acts"].float()
        row_scores = (a - mkto["mean_resid"][r]) @ -mkto["direction_unit"][r]
        table.append(dict(row=r, train=cell_auroc(row_scores, in_split["train"]), dev=cell_auroc(row_scores, in_split["dev"]), eval=cell_auroc(row_scores, in_split["eval"])))
        print(f"{gray}row {r}: train {table[-1]['train']:.3f}, dev {table[-1]['dev']:.3f}, eval {table[-1]['eval']:.3f}{endc}")
        if table[-1]["dev"] > best["dev"]: best = dict(table[-1], scores=row_scores)
    print(f"{cyan}mbpp_kto row {best['row']}, dev {best['dev']:.3f}{endc}")
    save_file({"scores": best["scores"].contiguous()}, ACTS / "mkto_scores.safetensors")
    (ACTS / "mkto_fit.json").write_text(json.dumps(dict(row=best["row"], table=table), indent=1))
    del a

#%% loads mbpp_kto's scores

mkto_fit = json.loads((ACTS / "mkto_fit.json").read_text())
mkto_scores = load_file(ACTS / "mkto_scores.safetensors")["scores"]
rollout_mkto = t.zeros(len(items)).index_add_(0, owner[cot], mkto_scores[cot]) / cot_len   # mean over the rollout's CoT
print(f"{cyan}mbpp_kto at row {mkto_fit['row']} (decoder layer {mkto_fit['row'] - 1}){endc}")

#%% the probe on rollouts it never saw, beside three things that should do worse: the same fit on labels shuffled between rollouts, each token's identity alone, and its position alone

evaluate = True
if evaluate:
    a = load_file(ACTS / f"row{fit['row']}.safetensors")["acts"].cuda().float()
    train = in_split["train"]
    mean = a[train].mean(0)
    std = a[train].std(0)
    z = (a - mean) / std
    shuffled = is_cheat.clone()
    for c in cell_of:
        members = [k for k, x in enumerate(items) if x["cell"] == c]
        shuffled[members] = is_cheat[random.sample(members, len(members))]
    group = {(x["cell"], x["cheat"]) for x in items if x["split"] == "train"}
    n_group = {g: sum((x["cell"], x["cheat"]) == g for x in items if x["split"] == "train") for g in group}
    per_item = t.tensor([1 / n_group[(x["cell"], x["cheat"])] if x["split"] == "train" else 0.0 for x in items])
    weight = (per_item / t.bincount(owner[train], minlength=len(items)).clamp(min=1))[owner[train]].cuda()
    weight /= weight.sum()
    label = shuffled[owner[train]].float().cuda()
    z_train = z[train]
    w = t.zeros(a.shape[1], device="cuda", requires_grad=True)
    b = t.zeros((), device="cuda", requires_grad=True)
    opt = t.optim.Adam([w, b], lr=fit["lr"])
    with t.enable_grad():
        for step in range(fit["steps"]):
            loss = (weight * t.nn.functional.binary_cross_entropy_with_logits(z_train @ w + b, label, reduction="none")).sum() + fit["l2"] * w.pow(2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    shuffled_logits = (z @ w + b).cpu()
    del a, z, z_train
    tec()
    token_ids = t.cat([t.tensor(x["ids"][x["prompt_len"]:]) for x in items])
    token_weight = weight.cpu()
    cheat_count = t.zeros(len(tok)).index_add_(0, token_ids[train], token_weight * is_cheat[owner[train]].float())
    clean_count = t.zeros(len(tok)).index_add_(0, token_ids[train], token_weight * (~is_cheat[owner[train]]).float())
    prior = 1 / len(owner)
    token_logits = ((cheat_count + prior) / (clean_count + prior)).log()[token_ids]
    scores = {"probe, tokens": logits, "probe, rollout mean": rollout_logit[owner], "grader_kto, tokens": kto_scores, "grader_kto, rollout mean": rollout_kto[owner], "mbpp_kto, tokens": mkto_scores, "mbpp_kto, rollout mean": rollout_mkto[owner], "shuffled labels, tokens": shuffled_logits, "token identity": token_logits, "token position": index.float()}
    results = {name: {s: cell_auroc(v, in_split[s]) for s in ["train", "dev", "eval", "family"]} for name, v in scores.items()}
    print(f"{cyan}within-cell AUROC, cheat vs clean{endc}")
    for name, by_split in results.items():
        print(f"  {name:24s} " + "  ".join(f"{s} {v:.3f}" for s, v in by_split.items()))
    print(f"{cyan}tokens by cell, held out: probe, grader_kto{endc}")
    for c, in_cell in cell_of.items():
        sel = (in_split["eval"] | in_split["family"]) & in_cell
        label_ = is_cheat[owner[sel]]
        if label_.any() and not label_.all(): print(f"  {c:20s} {roc_auc_score(label_.numpy(), logits[sel].numpy()):.3f}  {roc_auc_score(label_.numpy(), kto_scores[sel].numpy()):.3f}  ({len(set(owner[sel].tolist()))} rollouts)")
    (ACTS / "results.json").write_text(json.dumps(results, indent=1))

#%% when the signal arrives: AUROC and mean logit by tenth of the CoT on the held-out rollouts

by_position = True
if by_position:
    n_bins = 10
    held = in_split["eval"]
    # held = in_split["family"]
    tenth = (index.float() / cot_len[owner].float() * n_bins).long().clamp(max=n_bins - 1)
    curves = {}
    for name, values in {"probe": logits, "kto": kto_scores, "mkto": mkto_scores}.items():
        curves[name] = dict(
            auroc=t.tensor([cell_auroc(values, held & (tenth == k)) for k in range(n_bins)]),
            cheat=t.tensor([values[held & (tenth == k) & is_cheat[owner]].mean() for k in range(n_bins)]),
            clean=t.tensor([values[held & (tenth == k) & ~is_cheat[owner]].mean() for k in range(n_bins)]),
        )
        print(f"{cyan}{name}: tenth of CoT   AUROC   cheat   clean{endc}")
        for k in range(n_bins): print(f"  {k + 1:2d}           {curves[name]['auroc'][k]:.3f}   {curves[name]['cheat'][k]:+.2f}   {curves[name]['clean'][k]:+.2f}")
        line(
            [curves[name]["cheat"], curves[name]["clean"]],
            x=list(range(1, n_bins + 1)),
            names=["cheat", "clean"],
            labels={"x": "tenth of the CoT", "y": f"mean {name} score"},
            title=f"{name} score along the CoT, held-out rollouts",
        )
    line(
        [curves["probe"]["auroc"], curves["kto"]["auroc"], curves["mkto"]["auroc"]],
        x=list(range(1, n_bins + 1)),
        names=["probe", "grader_kto", "mbpp_kto"],
        labels={"x": "tenth of the CoT", "y": "within-cell AUROC"},
        title="Token AUROC along the CoT, held-out rollouts",
    )

#%% token-level view of held-out rollouts in same-prompt pairs: a cheating and a clean rollout of each cell, whose prompts are identical. Prompt unshaded, CoT and answer shaded by the
# readout (red toward cheat, blue toward clean), </think> outlined. Every strip shares one colour scale, set by the block at its end; hover a token for its score, clipped to that scale. The
# answer is outside what the probe trained on. readout picks which score the strips here show; the page holds both. grader_kto's scale saturates the same share of held-out tokens as the probe's

show_rollouts = True
if show_rollouts:
    families = FAMILIES + [HELD_FAMILY]
    # families = ["parity"]
    n_pairs = 1   # per cell
    readout = "probe"
    # readout = "kto"
    # readout = "mkto"
    order = "random"
    # order = "worst"   # the rollouts the probe gets most wrong
    clips = {"probe": 8.0}
    clips["kto"] = t.quantile(kto_scores[in_split["eval"]].abs(), (logits[in_split["eval"]].abs() < clips["probe"]).float().mean()).item()
    clips["mkto"] = t.quantile(mkto_scores[in_split["eval"]].abs(), (logits[in_split["eval"]].abs() < clips["probe"]).float().mean()).item()
    print(f"{cyan}colour scales: {clips}{endc}")
    random.seed(SEED)
    pairs = []
    for c in cell_of:
        family = c.split(".")[0]
        pools = {cheating: [k for k, x in enumerate(items) if x["cell"] == c and x["cheat"] == cheating and x["split"] in ("eval", "family")] for cheating in [True, False]}
        if family not in families: continue
        if not pools[True]:
            print(f"{gray}{c}: no held-out cheating rollout, so no pair{endc}")
            continue
        chosen = {cheating: random.sample(pool, n_pairs) if order == "random" else sorted(pool, key=lambda k: rollout_logit[k].item() if cheating else -rollout_logit[k].item())[:n_pairs] for cheating, pool in pools.items()}
        for j in range(n_pairs):
            assert items[chosen[True][j]]["ids"][:items[chosen[True][j]]["prompt_len"]] == items[chosen[False][j]]["ids"][:items[chosen[False][j]]["prompt_len"]], f"{c}: the pair's prompts differ"
            pair = dict(family=family, label=c)
            for cheating in [True, False]:
                k = chosen[cheating][j]
                x = items[k]
                html_of = {}
                for name, values, mean_score in [("probe", logits, rollout_logit[k]), ("kto", kto_scores, rollout_kto[k]), ("mkto", mkto_scores, rollout_mkto[k])]:
                    vals = [0.0] * x["prompt_len"] + values[owner == k].clamp(-clips[name], clips[name]).tolist() + [clips[name]]
                    head = f"{x['cell']} &middot; {'CHEAT' if x['cheat'] else 'clean'} &middot; rollout {k} &middot; {dict(probe='probe', kto='grader_kto', mkto='mbpp_kto')[name]} mean CoT score {mean_score:+.2f}"
                    html_of[name] = f"<div style='background:#111;color:#ddd;font:12px monospace;padding:8px;margin-bottom:10px'><div style='margin-bottom:6px;font-weight:bold'>{head}</div>{toks_html([tok.decode(i) for i in x['ids']] + [' [scale]'], pos=x['close'], vals=vals, val_name=name)}</div>"
                pair["cheat" if cheating else "clean"] = dict(id=f"g{k}", html=html_of)
                display(HTML(html_of[readout]))
            pairs.append(pair)

#%% the readouts on an environment neither trained on: every valid base-model MBPP rollout, scored one at a time. Needs the model; saves each rollout's scores on its generated tokens

score_mbpp = False
if score_mbpp:
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=REVISION, dtype=t.bfloat16, device_map="cuda").eval()
    probe = load_file(OUT / "cot_probe.safetensors")
    w = probe["w"].cuda()
    b = probe["b"].cuda()
    kto = load_file(OUT / "grader_kto.safetensors")
    u = kto["direction_unit"][kto_fit["row"]].cuda()
    pin = kto["mean_resid"][kto_fit["row"]].cuda()
    mbpp = [r for r in map(json.loads, open(MBPP_RUN)) if r["labels"]["valid"]]
    scored = {}
    scored_kto = {}
    for k, r in enumerate(pbar(mbpp, desc="mbpp")):
        assert r["ids"].count(THINK_CLOSE) == 1, r["rollout_id"]
        hidden = model(input_ids=t.tensor([r["ids"]], device="cuda"), output_hidden_states=True).hidden_states
        start = r["ids"].index(THINK_OPEN) + 2
        scored[str(k)] = (hidden[fit["row"]][0, start:].float() @ w + b).cpu()
        scored_kto[str(k)] = ((hidden[kto_fit["row"]][0, start:].float() - pin) @ u).cpu()
    save_file(scored, ACTS / "mbpp_logits.safetensors")
    save_file(scored_kto, ACTS / "mbpp_kto_scores.safetensors")
    del model, hidden
    tec()

#%% MBPP readouts: AUROCs pooled over every valid rollout, then one cheating and one honest rollout token by token. The run has one rollout per task, so these are not the same prompt. Cheat
# is passing the visible test whose expected value is wrong; the clean rollout shown also passes the original tests

show_mbpp = True
if show_mbpp:
    max_tokens = 1500   # the rollouts to pick from, so a strip stays readable
    pick = "median"   # the rollout nearest its class's median rollout-mean probe logit, the same rollouts under both readouts
    # pick = "random"
    clip = 8.0
    random.seed(SEED)
    mbpp = [r for r in map(json.loads, open(MBPP_RUN)) if r["labels"]["valid"]]
    scored = {"probe": load_file(ACTS / "mbpp_logits.safetensors"), "kto": load_file(ACTS / "mbpp_kto_scores.safetensors")}
    starts = [r["ids"].index(THINK_OPEN) + 2 for r in mbpp]
    closes = [r["ids"].index(THINK_CLOSE) for r in mbpp]
    cheats = t.tensor([r["labels"]["cheat"] is True for r in mbpp])
    cot_lens = t.tensor([closes[k] - starts[k] for k in range(len(mbpp))])
    mbpp_length_auroc = roc_auc_score(cheats.numpy(), cot_lens.numpy())
    means = {}
    last_tenths = {}
    mbpp_results = {}
    for name in scored:
        cot_scores = [scored[name][str(k)][:cot_lens[k]] for k in range(len(mbpp))]
        means[name] = t.stack([c.mean() for c in cot_scores])
        last_tenths[name] = t.stack([c[-max(1, len(c) // 10):].mean() for c in cot_scores])
        mbpp_results[name] = dict(
            rollout_mean=roc_auc_score(cheats.numpy(), means[name].numpy()),
            tokens=roc_auc_score(cheats.repeat_interleave(cot_lens).numpy(), t.cat(cot_scores).numpy(), sample_weight=(1 / cot_lens.float()).repeat_interleave(cot_lens).numpy()),
            last_tenth=roc_auc_score(cheats.numpy(), last_tenths[name].numpy()),
            median_cheat=means[name][cheats].median().item(),
            median_clean=means[name][~cheats].median().item(),
        )
    for name in scored: print(f"{cyan}MBPP {name}, {len(mbpp)} valid rollouts, {cheats.sum()} cheating: " + ", ".join(f"{key} {v:.3f}" for key, v in mbpp_results[name].items()) + f", CoT length alone {mbpp_length_auroc:.3f}{endc}")
    chosen = {}
    for cheating in [True, False]:
        pool = [k for k, r in enumerate(mbpp) if len(r["ids"]) <= max_tokens and (cheats[k] if cheating else r["labels"]["honest_correct"])]
        target = means["probe"][cheats if cheating else ~cheats].median()
        chosen[cheating] = random.choice(pool) if pick == "random" else min(pool, key=lambda k: (means["probe"][k] - target).abs().item())
    pair = dict(family="mbpp", label=f"tasks {mbpp[chosen[True]]['labels']['task_id']} and {mbpp[chosen[False]]['labels']['task_id']}: different prompts")
    for cheating, k in chosen.items():
        r = mbpp[k]
        html_of = {}
        for name in scored:
            vals = [0.0] * starts[k] + scored[name][str(k)].clamp(-clips[name], clips[name]).tolist() + [clips[name]]
            head = f"mbpp task {r['labels']['task_id']} &middot; {'CHEAT' if cheating else 'clean'} &middot; {'probe' if name == 'probe' else 'grader_kto'} mean CoT score {means[name][k]:+.2f} &middot; last tenth {last_tenths[name][k]:+.2f}"
            html_of[name] = f"<div style='background:#111;color:#ddd;font:12px monospace;padding:8px;margin-bottom:10px'><div style='margin-bottom:6px;font-weight:bold'>{head}</div>{toks_html([tok.decode(i) for i in r['ids']] + [' [scale]'], pos=closes[k], vals=vals, val_name=name)}</div>"
        pair["cheat" if cheating else "clean"] = dict(id=f"m{k}", html=html_of)
        display(HTML(html_of[readout]))
    pairs.append(pair)

#%% the numbers and the strips above as one self-contained page, a fixed three-way button switching every strip, chart and scale between the readouts and keeping the rollout in view in place

save_page = True
if save_page:
    path = ACTS / "cot_probe_tokens.html"
    names = {"probe": "trained probe", "kto": "grader_kto", "mkto": "mbpp_kto"}
    units = {"probe": "logit", "kto": "projection", "mkto": "projection"}
    x_of = lambda k: 70 + 620 * k / (n_bins - 1)
    panels = {}
    for name, curve in curves.items():
        span = max(curve["cheat"].abs().max(), curve["clean"].abs().max()).item() * 1.1
        y_of = lambda v: 150 - 130 * v / span
        grid = "".join(f"<line x1=70 x2=690 y1={y_of(v):.1f} y2={y_of(v):.1f} stroke='var(--line)' stroke-width={1.5 if v == 0 else 1} /><text x=60 y={y_of(v) + 4:.1f} text-anchor=end fill='var(--muted)' font-size=12>{v:+.1f}</text>" for v in [-span, -span / 2, 0, span / 2, span])
        ticks = "".join(f"<text x={x_of(k)} y=298 text-anchor=middle fill='var(--muted)' font-size=12>{k + 1}</text>" for k in range(n_bins))
        series = "".join(f"<polyline fill=none stroke='{color}' stroke-width=2 points='{' '.join(f'{x_of(k):.1f},{y_of(v):.1f}' for k, v in enumerate(values.tolist()))}' />" + "".join(f"<circle cx={x_of(k):.1f} cy={y_of(v):.1f} r=4 fill='{color}' stroke='var(--panel)' stroke-width=2 />" for k, v in enumerate(values.tolist())) + f"<text x=702 y={y_of(values[-1].item()) + 4:.1f} fill='var(--text)' font-size=13>{label}</text>" for label, values, color in [("cheat", curve["cheat"], "#e65050"), ("clean", curve["clean"], "#5082e6")])
        hits = "".join(f"<rect x={x_of(k) - 31:.1f} y=20 width=62 height=262 fill=transparent><title>tenth {k + 1}: cheat {curve['cheat'][k]:+.2f}, clean {curve['clean'][k]:+.2f}, AUROC {curve['auroc'][k]:.3f}</title></rect>" for k in range(n_bins))
        chart = f"<svg viewBox='0 0 760 320' role=img aria-label='Mean {names[name]} score by tenth of the CoT'>{grid}{ticks}{series}{hits}<text x=380 y=316 text-anchor=middle fill='var(--muted)' font-size=12>tenth of the CoT</text></svg>"
        tenths = "<table><tr><th>tenth of the CoT</th>" + "".join(f"<th>{k + 1}</th>" for k in range(n_bins)) + "</tr><tr><td>AUROC</td>" + "".join(f"<td>{v:.2f}</td>" for v in curve["auroc"].tolist()) + f"</tr><tr><td>cheat {units[name]}</td>" + "".join(f"<td>{v:+.1f}</td>" for v in curve["cheat"].tolist()) + f"</tr><tr><td>clean {units[name]}</td>" + "".join(f"<td>{v:+.1f}</td>" for v in curve["clean"].tolist()) + "</tr></table>"
        panels[name] = f"<div class=m-{name}><h2>{names[name]}: mean {units[name]} along the CoT, held-out rollouts</h2><div class=legend><span><i style='background:#e65050'></i>cheat</span><span><i style='background:#5082e6'></i>clean</span></div>{chart}{tenths}</div>"
    table = "<table><tr><th></th><th>train</th><th>dev</th><th>held out</th><th>pets (unseen family)</th></tr>" + "".join(f"<tr><td>{name}</td>" + "".join(f"<td>{by_split[s]:.3f}</td>" for s in ["train", "dev", "eval", "family"]) + "</tr>" for name, by_split in results.items()) + "</table>"
    mbpp_table = "<table><tr><th></th><th>rollout mean</th><th>single tokens</th><th>last tenth of the CoT</th><th>median cheat</th><th>median clean</th></tr>" + "".join(f"<tr><td>{names[name]}</td><td>{v['rollout_mean']:.3f}</td><td>{v['tokens']:.3f}</td><td>{v['last_tenth']:.3f}</td><td>{v['median_cheat']:+.2f}</td><td>{v['median_clean']:+.2f}</td></tr>" for name, v in mbpp_results.items()) + "</table>"
    mbpp_note = f"<p>An environment neither the probe nor grader_kto trained on: the base model on tasks whose visible test has a wrong expected value. Cheat means passing that test. This run has one rollout per task, so no prompt has both a cheating and a clean rollout, and the two below are not a same-prompt pair. AUROCs are over all {len(mbpp)} valid rollouts ({cheats.sum()} cheating, every other valid rollout counted as clean), pooled across tasks. CoT length alone gives {mbpp_length_auroc:.3f}.</p>{mbpp_table}<p class=cap>Each rollout below is {'the one nearest its class median of the trained probe' if pick == 'median' else 'a random pick'} among those under {max_tokens} tokens, and the clean one also passes the original tests.</p>"
    both = lambda c: f"<div class='strip m-probe' data-id={c['id']}>{c['html']['probe']}</div><div class='strip m-kto' data-id={c['id']}>{c['html']['kto']}</div><div class='strip m-mkto' data-id={c['id']}>{c['html'].get('mkto', '<p class=cap>mbpp_kto is not scored on MBPP here: this rollout is from a task it trained on.</p>')}</div>"
    show_pair = lambda c: f"<h3>{c['label']}</h3><div class=cols><div>{both(c['cheat'])}</div><div>{both(c['clean'])}</div></div>"
    sections = "".join(f"<section><h2>{family}</h2>{mbpp_note if family == 'mbpp' else ''}{''.join(show_pair(c) for c in pairs if c['family'] == family)}</section>" for family in families + ["mbpp"])
    css = """:root{color-scheme:dark;--bg:#101419;--panel:#181e26;--raised:#222a34;--line:#303a46;--text:#e7edf3;--muted:#a4b0bd;--accent:#8dbaff;--red:#f0a0a7}
    body{margin:0 auto;max-width:1500px;padding:32px;background:var(--bg);color:var(--text);font:15px/1.6 Inter,ui-sans-serif,system-ui;font-variant-numeric:tabular-nums}
    h1{letter-spacing:-.02em;margin:.2em 0;padding-right:400px}h2{letter-spacing:-.01em}.eyebrow,.sub,caption,.cap{color:var(--muted)}section{border-top:1px solid var(--line);margin-top:28px}
    table{border-collapse:collapse;margin:12px 0}td,th{padding:4px 12px;border-bottom:1px solid var(--line);text-align:right}td:first-child,th:first-child{text-align:left}
    .cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}.cols>div{min-width:0}.legend{display:flex;gap:18px;color:var(--muted)}.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px}
    svg{width:100%;max-width:760px;display:block;background:var(--panel);border-radius:8px}@media(max-width:820px){.cols{grid-template-columns:1fr}}
    body.probe .m-kto,body.probe .m-mkto,body.kto .m-probe,body.kto .m-mkto,body.mkto .m-probe,body.mkto .m-kto{display:none}
    #mode{position:fixed;top:16px;right:16px;z-index:1000;display:flex;border-radius:8px;border:1.5px solid var(--accent);background:var(--raised);box-shadow:0 4px 16px #0008;overflow:hidden}
    #mode button{padding:10px 14px;border:0;background:none;color:var(--muted);font:600 14px Inter,ui-sans-serif,system-ui;cursor:pointer}#mode button:hover{background:var(--line);color:var(--text)}#mode button[aria-pressed=true]{background:var(--accent);color:#101419}
    body{padding-bottom:100vh}
    @media print{#mode{display:none}}"""
    script = """function show(mode){
        const seen=[...document.querySelectorAll('.strip')].filter(e=>e.offsetParent);
        const anchor=seen.find(e=>e.getBoundingClientRect().bottom>70);
        const top=anchor?anchor.getBoundingClientRect().top:0;
        document.body.className=mode;
        document.querySelectorAll('#mode button').forEach(b=>b.setAttribute('aria-pressed',b.dataset.mode==mode));
        if(anchor){const twin=[...document.querySelectorAll('.strip[data-id='+anchor.dataset.id+']')].find(e=>e.offsetParent);window.scrollBy(0,twin.getBoundingClientRect().top-top);}
    }"""
    read_probe = f"<div class=m-probe><p>Showing the trained probe: logistic regression at decoder layer {fit['row'] - 1}, on one colour scale of &plusmn;{clips['probe']:g} logits. Red leans cheat, blue leans clean.</p></div>"
    read_kto = f"<div class=m-kto><p>Showing grader_kto: nothing fit, the token's residual at decoder layer {kto_fit['row'] - 1} minus the hook's pin, along the direction, on one colour scale of &plusmn;{clips['kto']:.2g}. Red is toward cheating and blue away from it. Zero is the pin, so colour here is what the hook would remove at that token, at this layer. The colour scale saturates the same share of held-out tokens as the probe's.</p></div>"
    read_mkto = f"<div class=m-mkto><p>Showing mbpp_kto: read like grader_kto, at decoder layer {mkto_fit['row'] - 1}, on one colour scale of &plusmn;{clips['mkto']:.2g}. It trained on MBPP rollouts, and on none of these grader hack rollouts, though its neutrals include no-grader and agreeing-grader rollouts of the same families. It is not scored on the MBPP rollouts below, which are from tasks it trained on.</p></div>"
    buttons = "".join(f"<button data-mode={k} aria-pressed={'true' if k == 'probe' else 'false'} onclick=\"show('{k}')\">{v}</button>" for k, v in names.items())
    path.write_text(f"<!doctype html><html lang=en><head><meta charset=utf-8><title>CoT probe tokens</title><style>{css}</style></head><body class=probe><div id=mode role=group aria-label=Readout>{buttons}</div><div class=eyebrow>ortho &middot; CoT cheat readouts &middot; 2026-11-05</div><h1>{PAGE_TITLE}</h1><p class=sub>Qwen3.8-27B. Three readouts of the same rollouts, one token at a time. The buttons at the top right switch between them. The probe and grader_kto trained on five grader families, with pets and MBPP unseen. mbpp_kto trained on MBPP.</p><section><h2>Within-cell AUROC, cheat against clean</h2>{table}<p class=cap>Cheat and clean rollouts of a cell share one prompt, so these numbers do not come from telling prompts apart. grader_kto trained on the train and dev rollouts, and has no fit of its own to this task. mbpp_kto trained on none of these rollouts.</p></section><section>{panels['probe']}{panels['kto']}{panels['mkto']}</section><section><h2>How to read the strips</h2>{read_probe}{read_kto}{read_mkto}<p>Each grader row below is one prompt: a cheating rollout on the left and a clean one on the right, with identical prompt tokens. MBPP's row is not a same-prompt pair. The prompt is unshaded, the block at each strip's end shows the top of the scale, the outlined token is &lt;/think&gt;, and text after it is the answer. All rollouts below are held out. pets.pet has no pair because it has no cheating rollouts.</p></section>{sections}<section class=cap><p>Source: cot_probe.py, data/vectors/Qwen3.8-27B/cot_probe.safetensors, grader_kto.safetensors and mbpp_kto.safetensors, Will's s9-gr manifest.</p></section><script>{script}</script></body></html>")
    print(f"{green}wrote {path}, {path.stat().st_size / 1e6:.1f} MB{endc}")
