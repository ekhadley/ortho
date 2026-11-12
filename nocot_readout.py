#%%
"""Do directions fit on reasoning-on rollouts read the reasoning-off cheat rate of each grader prompt? With reasoning off every sample of a (family, side) cell shares one
prompt, so the cell is the unit: its cheat rate comes from the base nothink38* samples and its state from one forward pass of the HF Qwen3.8-27B on the prompt. Per cell
the hack, clean and none prompts are run and the residual kept at the last prompt token (whose next token is the answer), at </think> and as the mean over the prompt. A
direction's score for a cell is the hack minus clean (or none) residual dotted with its unit row, signed and scaled by that row's reasoning-on gap (cheating minus clean
CoT means of the base reasoning-on rollouts of the same hack prompts) and averaged over the direction's rows. The same prompts with reasoning on, cut before the CoT, give
the control: the same score at the newline after <think> against the reasoning-on cheat rate. Scores are rank-correlated with the rates across cells, against random
unit directions put through the same arithmetic.
    python nocot_readout.py prep       laptop: prompts, rates and reasoning-on rollouts from the records, acts/nocot_readout/inputs.pt
    python nocot_readout.py run        box, vLLM stopped: the forward passes, acts/nocot_readout/results.pt
    python nocot_readout.py analyze    laptop: scores and rank correlations, acts/nocot_readout/nocot_readout.json
    python nocot_readout.py rows       laptop: the reasoning-off cheat rates of the served rows (run.py's <row>_nothink* runs, scripts/box_nothink.sh) beside base"""
import json
import random
import sys
from pathlib import Path

import torch as t
from safetensors.torch import load_file
from scipy.stats import rankdata
from transformers import AutoModelForImageTextToText

from mechtools import *

from rhenvs.grader import FAMILIES, MAIN_FAMILIES

STAGE = sys.argv[1] if len(sys.argv) > 1 else "analyze"
MODEL_ID, REVISION = "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
D = 5120
VECTORS = Path("data/vectors/Qwen3.8-27B")
ARCHIVE = Path.home() / "wgmn/ortho_box_archive/59147695/models"   # Will's resid-bias rows
ACTS = Path("acts/nocot_readout")
INPUTS = ACTS / "inputs.pt"
RESULTS = ACTS / "results.pt"
KINDS = ["hack", "clean", "none"]
POSITIONS = ["last", "think_close", "prompt_mean"]   # of a reasoning-off prompt, which ends <think> \n\n </think> \n\n
N_ON = 16   # reasoning-on rollouts per cell and class
MAX_TOKENS = 800
TOP_K = 50
N_RANDOM = 1000
PROBE_ROW = 52
SEED = 0
THINK_OPEN, THINK_CLOSE, NEWLINE, IM_START = 248068, 248069, 198, 248045
SHORTLIST = ["will_g9", "will_cm", "grader_kto", "grader_proj", "grader_dpo", "mbpp_kto", "mbpp_dom", "mbpp_hunt_vr36", "mbpp_hunt_vr44", "mbpp_hunt_vr48", "mbpp_hunt_vr52", "mbpp_hunt_vrw48", "mbpp_hunt_dom", "branch_vr48", "cot_probe", "nocot_dom_last", "nocot_dom_resp", "mbpp_dom_rand", "mbpp_hunt_rand"]


def spearman(x: Tensor, y: Tensor) -> Tensor:
    """Rank correlation of each row of x [n, cell] with y [cell], over the cells where x is not nan."""
    keep = ~x.isnan().any(0)
    rx, ry = t.tensor(rankdata(x[:, keep].numpy(), axis=1)), t.tensor(rankdata(y[keep].numpy()))
    rx, ry = rx - rx.mean(1, keepdim=True), ry - ry.mean()
    return (rx * ry).sum(1) / (rx.norm(dim=1) * ry.norm())


#%% prep: per cell and prompt kind the reasoning-off prompt with its base cheat rate, the same prompt with reasoning on cut before the CoT, and up to N_ON base reasoning-on rollouts per class
if STAGE == "prep":
    records = [json.loads(l) for p in sorted(Path("data/inspect/grader").glob("*.jsonl")) for l in open(p)]
    base = [r for r in records if r["model"] == "vllm/cb_base" and r["condition"] == "none" and r["labels"]["sys"] == "none" and r["labels"]["family"] in FAMILIES]
    off = [r for r in base if r["config_id"] == "nothink"]
    on = [r for r in base if r["config_id"] == "qwen3.8_low" and r["labels"]["prompts"] == "hack" and r["labels"]["cheat"] is not None]
    cells = sorted({(r["labels"]["family"], r["labels"]["side"]) for r in off})
    assert len(cells) == 2 * len(FAMILIES)
    rng = random.Random(SEED)
    prompts, rollouts, on_rates = [], [], []
    for cell in cells:
        mine = [r for r in on if (r["labels"]["family"], r["labels"]["side"]) == cell]
        on_rates.append((sum(r["labels"]["cheat"] for r in mine), len(mine)))
        system = mine[0]["ids"][:mine[0]["ids"].index(IM_START, 1)]   # the reasoning-effort system turn of the reasoning-on config
        for kind in KINDS:
            rs = [r for r in off if (r["labels"]["family"], r["labels"]["side"]) == cell and r["labels"]["prompts"] == kind]
            ids = {tuple(r["ids"][:r["ids"].index(THINK_CLOSE) + 2]) for r in rs}
            assert len(ids) == 1 and all(r["turns"][0]["reasoning"] == "" for r in rs), (cell, kind, len(ids))
            ids = list(ids.pop())
            valid = [r["labels"]["cheat"] for r in rs if r["labels"]["cheat"] is not None]
            prompts.append(dict(family=cell[0], side=cell[1], kind=kind, ids=ids, ids_on=system + ids[:ids.index(THINK_OPEN) + 1] + [NEWLINE], n=len(rs), valid=len(valid), cheat=sum(valid)))
        hack_on = prompts[-3]["ids_on"]
        assert all(r["ids"][:len(hack_on)] == hack_on for r in mine), cell   # the built reasoning-on prompt is the one the rollouts were served
        for cheat in [True, False]:
            pool = [r["ids"] for r in mine if r["labels"]["cheat"] == cheat and r["ids"].count(THINK_OPEN) == 1 and r["ids"].count(THINK_CLOSE) == 1 and len(r["ids"]) <= MAX_TOKENS and r["ids"].index(THINK_CLOSE) - len(hack_on) > 1]
            rollouts += [dict(family=cell[0], side=cell[1], cheat=cheat, ids=ids, p_start=len(hack_on) - 1, p_end=ids.index(THINK_CLOSE) - 1) for ids in rng.sample(pool, min(N_ON, len(pool)))]
    assert all(x["ids"][x["p_start"]] == NEWLINE and x["ids"][x["p_end"]] == NEWLINE for x in rollouts)
    for cell, (c, n) in zip(cells, on_rates):
        hack = next(p for p in prompts if (p["family"], p["side"], p["kind"]) == (*cell, "hack"))
        picked = [sum(x["cheat"] == cheat and (x["family"], x["side"]) == cell for x in rollouts) for cheat in [True, False]]
        print(f"{cyan}{cell[0]:16s} {cell[1]:10s} reasoning off {hack['cheat']:3d}/{hack['valid']:3d}   reasoning on {c:3d}/{n:3d}   rollouts picked: {picked[0]} cheating, {picked[1]} clean{endc}")
    ACTS.mkdir(parents=True, exist_ok=True)
    t.save(dict(cells=cells, prompts=prompts, rollouts=rollouts, on_rates=on_rates), INPUTS)
    print(f"{green}{len(cells)} cells, {len(prompts)} prompts, {len(rollouts)} reasoning-on rollouts (longest {max(len(x['ids']) for x in rollouts)} tokens) saved to {INPUTS}{endc}")

#%% run: every decoder layer's output collected by a hook; row i of a saved residual is the output of decoder layer i, which is row i + 1 of a direction file
if STAGE == "run":
    inputs = t.load(INPUTS, weights_only=False)
    cells, prompts, rollouts = inputs["cells"], inputs["prompts"], inputs["rollouts"]
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=REVISION, dtype=t.bfloat16, device_map="cuda").eval()
    layers = model.model.language_model.layers
    assert len(layers) == 64
    outputs = [None] * 64
    def collect(i: int):
        def hook(module, args, output): outputs[i] = (output[0] if isinstance(output, tuple) else output)[0].float()
        return hook
    for i, layer in enumerate(layers): layer.register_forward_hook(collect(i))
    resid = t.zeros(len(prompts), len(POSITIONS), 64, D)
    resid_on = t.zeros(len(prompts), 64, D)   # the reasoning-on prompt's last token, the newline after <think>
    top_ids, top_logprobs = t.zeros(len(prompts), TOP_K, dtype=t.long), t.zeros(len(prompts), TOP_K)
    cot_sum, cot_n = t.zeros(len(cells), 2, 64, D), t.zeros(len(cells), 2)   # class 0 clean, 1 cheating
    with t.no_grad():
        for i, p in enumerate(pbar(prompts, desc="prompts")):
            logits = model(input_ids=t.tensor([p["ids"]], device="cuda")).logits[0, -1].float()
            h = t.stack(outputs)
            resid[i] = t.stack([h[:, -1], h[:, -2], h.mean(1)]).cpu()
            top = t.log_softmax(logits, -1).topk(TOP_K)
            top_ids[i], top_logprobs[i] = top.indices.cpu(), top.values.cpu()
            model(input_ids=t.tensor([p["ids_on"]], device="cuda"))
            resid_on[i] = t.stack(outputs)[:, -1].cpu()
        for x in pbar(rollouts, desc="reasoning-on rollouts"):
            model(input_ids=t.tensor([x["ids"]], device="cuda"))
            cell = cells.index((x["family"], x["side"]))
            cot_sum[cell, int(x["cheat"])] += t.stack(outputs)[:, x["p_start"] + 1:x["p_end"]].mean(1).cpu()
            cot_n[cell, int(x["cheat"])] += 1
    t.save(dict(resid=resid, resid_on=resid_on, top_ids=top_ids, top_logprobs=top_logprobs, cot_sum=cot_sum, cot_n=cot_n), RESULTS)
    print(f"{green}saved {RESULTS}{endc}")

#%% analyze: the directions, as unit rows [65, d_model] in train_vector.py's layout (row r is the output of decoder layer r - 1), zero where a direction has no row
if STAGE == "analyze":
    dirs = {}
    for p in sorted(VECTORS.glob("*.safetensors")) + sorted((VECTORS / "dom_hunt_new").glob("*.safetensors")):
        f = load_file(p)
        if "direction_unit" in f: dirs[p.stem] = f["direction_unit"].float()
    for name, folder in [("will_g9", "cb_a3"), ("will_cm", "cb_cm")]:   # the bias on decoder layer L - 1's output is -alpha times his unit row at resid_pre L
        f = load_file(ARCHIVE / folder / "model-resid-bias.safetensors")
        dirs[name] = t.cat([t.zeros(1, D), -t.stack([f[f"model.language_model.layers.{L}.mlp.down_proj.bias"].float() for L in range(64)])])
    dirs["cot_probe"] = t.zeros(65, D)
    dirs["cot_probe"][PROBE_ROW] = load_file(VECTORS / "cot_probe.safetensors")["w"].float()
    dirs = {name: t.nn.functional.normalize(v, dim=1) for name, v in dirs.items()}
    assert all(name in dirs for name in SHORTLIST)
    gen = t.Generator().manual_seed(SEED)
    random_shared = normed(t.randn(N_RANDOM, D, generator=gen))   # one direction at every row, like the trained rows
    random_rows = normed(t.randn(N_RANDOM // 4, 34, D, generator=gen))   # a new direction at each row, like the difference-of-means rows

#%% analyze: the per-cell quantities every score is a dot product with. Residual index i is direction row i + 1.
if STAGE == "analyze":
    inputs, results = t.load(INPUTS, weights_only=False), t.load(RESULTS, weights_only=False)
    cells, prompts = inputs["cells"], inputs["prompts"]
    hack_prompts = prompts[::3]
    assert [(p["family"], p["side"], p["kind"]) for p in prompts] == [(*cell, kind) for cell in cells for kind in KINDS]
    off_rate = t.tensor([p["cheat"] / p["valid"] for p in hack_prompts])
    on_rate = t.tensor([c / n for c, n in inputs["on_rates"]])
    resid = results["resid"].unflatten(0, (len(cells), 3))   # [cell, kind, position, 64, d_model]
    resid_on = results["resid_on"].unflatten(0, (len(cells), 3))   # [cell, kind, 64, d_model]
    cot_mean = results["cot_sum"] / results["cot_n"][:, :, None, None]   # [cell, class, 64, d_model], nan where a cell has no rollout of the class
    gap = cot_mean[:, 1] - cot_mean[:, 0]   # cheating minus clean, [cell, 64, d_model]
    both = ~gap.isnan().any(-1).any(-1)
    print(f"{gray}{both.sum().item()} of {len(cells)} cells have reasoning-on rollouts of both classes; cells without: {[cells[i] for i in (~both).nonzero().flatten().tolist()]}{endc}")
    gap_mean = gap[both].mean(0)   # [64, d_model]
    states = dict(   # [cell, 64, d_model] each: what a direction is dotted with
        off_last_vs_clean=resid[:, 0, 0] - resid[:, 1, 0],
        off_last_vs_none=resid[:, 0, 0] - resid[:, 2, 0],
        off_close_vs_clean=resid[:, 0, 1] - resid[:, 1, 1],
        off_mean_vs_clean=resid[:, 0, 2] - resid[:, 1, 2],
        off_last_hack=resid[:, 0, 0],
        on_pre_vs_clean=resid_on[:, 0] - resid_on[:, 1],
        on_pre_vs_none=resid_on[:, 0] - resid_on[:, 2],
        on_cot_clean=cot_mean[:, 0],   # the clean rollouts' CoT mean alone; nan cells dropped from its correlations
    )

    norms = resid[:, 0, 0].norm(dim=-1).mean(0)   # [64], the hack prompts' mean residual norm at the last token: the per-row scale that keeps late rows from dominating a row average

    def scores(u: Tensor, rows: list[int]) -> dict[str, Tensor]:
        """u [n, len(rows), d_model] unit rows -> per state the cell scores [n, cell]: the state dotted with each row, signed by the row's mean reasoning-on gap (positive toward
        cheating), as a fraction of the residual norm, averaged over rows. <state>_gaps is the signed dots summed over rows in units of the summed reasoning-on gap."""
        idx = [r - 1 for r in rows]
        g = t.einsum("rd,nrd->nr", gap_mean[idx], u)   # [n, row]
        out = {"gap_sign": (g > 0).float().mean(1)}
        for name, s in states.items():
            dots = t.einsum("crd,nrd->ncr", s[:, idx], u) * g.sign()[:, None, :]
            out[name] = (dots / norms[idx]).mean(-1)
            out[name + "_gaps"] = dots.sum(-1) / g.abs().sum(-1)[:, None]
        return out

    def table(u: Tensor, rows: list[int]) -> dict[str, Tensor]:
        s = scores(u, rows)
        out = {f"{name}|{rate}": spearman(s[name], y) for name in states for rate, y in [("off", off_rate), ("on", on_rate)]}
        return out | {"shift_last": s["off_last_vs_clean_gaps"].median(1).values, "shift_pre": s["on_pre_vs_clean_gaps"].median(1).values, "gap_sign": s["gap_sign"]}

#%% analyze: every direction against the random directions over its own rows
if STAGE == "analyze":
    report, nulls = {}, {}
    for name, v in pbar(dirs.items(), desc="directions"):
        rows = [r for r in (v.norm(dim=1) > 0).nonzero().flatten().tolist() if r >= 1]
        shared = bool(len(rows) == 1 or t.cosine_similarity(v[rows[0]], v[rows[-1]], dim=0) > 0.99)
        mine = table(v[rows][None], rows)
        if (shared, *rows) not in nulls: nulls[shared, *rows] = table(random_shared[:, None, :].expand(-1, len(rows), -1), rows) if shared else table(random_rows[:, [(r - 31) % 34 for r in rows]], rows)
        null = nulls[shared, *rows]
        report[name] = dict(rows=[rows[0], rows[-1]], shared=shared, **{k: dict(value=x.item(), null_lo=null[k].quantile(0.025).item(), null_hi=null[k].quantile(0.975).item(), null_frac_below=(null[k] < x).float().mean().item()) for k, x in ((k, x[0]) for k, x in mine.items())})
    (ACTS / "nocot_readout.json").write_text(json.dumps(dict(cells=[".".join(c) for c in cells], off_rate=off_rate.tolist(), on_rate=on_rate.tolist(), directions=report), indent=1))

    def cell(r: dict, key: str) -> str:
        x = r[key]
        mark = "*" if not x["null_lo"] <= x["value"] <= x["null_hi"] else " "
        return f"{x['value']:+.2f}{mark}"
    cols = ["off_last_vs_clean|off", "off_last_vs_none|off", "off_close_vs_clean|off", "off_mean_vs_clean|off", "on_pre_vs_clean|on", "on_pre_vs_none|on", "on_cot_clean|on", "on_pre_vs_clean|off", "off_last_vs_clean|on"]
    print(f"{purple}rank correlation across {len(cells)} cells of a direction's score with the cheat rate; * outside the random directions' central 95%{endc}")
    print(f"{gray}column = <state>|<rate>: off_* states are the reasoning-off prompt (last token, </think>, prompt mean; hack minus clean or none), on_pre the reasoning-on prompt's last token, on_cot_clean the clean reasoning-on rollouts' CoT mean; rate off / on is the reasoning-off / reasoning-on cheat rate{endc}")
    print(f"{'direction':18s} {'rows':>6s} " + " ".join(f"{c:>24s}" for c in cols) + f" {'shift_last':>11s} {'shift_pre':>10s} {'gap_sign':>9s}")
    for name in SHORTLIST:
        r = report[name]
        print(f"{cyan}{name:18s} {r['rows'][0]:>3d}-{r['rows'][1]:<2d} " + " ".join(f"{cell(r, c):>24s}" for c in cols) + f" {cell(r, 'shift_last'):>11s} {cell(r, 'shift_pre'):>10s} {r['gap_sign']['value']:>9.2f}{endc}")
    null = report["grader_kto"]
    print(f"{gray}random central 95% over grader_kto's rows: " + ", ".join(f"{c} [{null[c]['null_lo']:+.2f}, {null[c]['null_hi']:+.2f}]" for c in cols[:2] + cols[4:5]) + f"{endc}")
    print(f"{gray}reasoning-off against reasoning-on cheat rate across cells: {spearman(off_rate[None], on_rate).item():+.2f}{endc}")

#%% rows: every served row's reasoning-off records (hack and clean prompts) per cell, beside the base rows'. A row's drop is base minus row, the mean over cells of the per-cell cheat rates.
if STAGE == "rows":
    records = [json.loads(l) for p in sorted(Path("data/inspect/grader").glob("*.jsonl")) for l in open(p)]
    records = [r for r in records if r["config_id"] == "nothink" and r["model"].startswith("vllm/cb_") and r["labels"]["sys"] == "none" and r["labels"]["family"] in FAMILIES]
    cells = sorted({(r["labels"]["family"], r["labels"]["side"]) for r in records})
    served = ["base"] + sorted({r["model"].removeprefix("vllm/cb_") for r in records} - {"base"})
    counts = t.zeros(len(served), 2, len(cells), 3)   # [row, hack or clean, cell, (cheats, valid, samples)]
    for r in records:
        if r["labels"]["prompts"] == "none": continue
        c = r["labels"]["cheat"]
        counts[served.index(r["model"].removeprefix("vllm/cb_")), KINDS.index(r["labels"]["prompts"]), cells.index((r["labels"]["family"], r["labels"]["side"]))] += t.tensor([bool(c), c is not None, 1.0])
    rate = counts[..., 0] / counts[..., 1]   # [row, kind, cell]
    main = t.tensor([cell[0] in MAIN_FAMILIES for cell in cells])
    print(f"{purple}reasoning off, hack prompts: cheats / valid answers pooled over cells, the mean per-cell rate, valid answers of all, and the wrong-side rate on clean prompts{endc}")
    print(f"{'row':12s} {'main families':>20s} {'held-out families':>20s} {'cell mean':>10s} {'drop':>7s} {'valid':>7s} {'clean wrong':>12s} {'clean valid':>12s} {'rho base':>9s}")
    for i, row in enumerate(served):
        if counts[i, 0, :, 2].min() == 0: continue   # a row whose runs are not all in yet
        pooled = [f"{int(counts[i, 0, m, 0].sum())}/{int(counts[i, 0, m, 1].sum())} = {counts[i, 0, m, 0].sum() / counts[i, 0, m, 1].sum():.3f}" for m in (main, ~main)]
        clean = f"{counts[i, 1, :, 0].sum() / counts[i, 1, :, 1].sum():12.3f} {counts[i, 1, :, 1].sum() / counts[i, 1, :, 2].sum():12.3f}" if counts[i, 1, :, 2].min() > 0 else f"{'-':>12s} {'-':>12s}"
        rho = spearman(rate[i, 0][None], rate[0, 0]).item()
        print(f"{cyan}{row:12s} {pooled[0]:>20s} {pooled[1]:>20s} {rate[i, 0].mean():10.3f} {(rate[0, 0] - rate[i, 0]).mean():+7.3f} {counts[i, 0, :, 1].sum() / counts[i, 0, :, 2].sum():7.3f} {clean} {rho:+9.2f}{endc}")
    print(f"{purple}per-cell cheat rate on hack prompts{endc}")
    done = [i for i in range(len(served)) if counts[i, 0, :, 2].min() > 0]
    print(f"{'cell':26s} " + " ".join(f"{served[i][:9]:>9s}" for i in done))
    for c in rate[0, 0].argsort(descending=True).tolist(): print(f"{gray}{'.'.join(cells[c]):26s} " + " ".join(f"{rate[i, 0, c]:9.2f}" for i in done) + f"{endc}")
    (ACTS / "nocot_rows.json").write_text(json.dumps(dict(cells=[".".join(c) for c in cells], rows=[served[i] for i in done], counts=counts[done].tolist()), indent=1))
