#%%
"""Finite-difference sensitivity of Qwen3.8-27B along our saved directions, in our own rollouts: the in-context follow-up to jlens_gain.py's null. At resid_pre L (row L of
the train_vector layout, the output of decoder layer L-1) the clean residual of a rollout gets +-eps v added, eps a fraction of that rollout's median residual norm at L,
under three perturbation sets: every position (what the projection hook does), the newline after <think> only (the position whose next token is the model's first), and the
newline before </think> only (where mbpp_dom was fit). Layers L..63 are rerun from the clean hidden state and per position the final residual's shift (plus, minus, and the
central difference ||h(+eps) - h(-eps)|| / 2) and the next-token KL from the clean forward are recorded. Directions: every saved row, cot_probe at row 52, Will's G9 and
CoT-mean rows from the box archive, the pins, and the controls: random unit directions, MLP decoder and encoder directions of the layer, pullbacks of unembedding cluster
centroids and of random vectors through the Qwen3.6 J-lens, and a zero perturbation (the noise floor). Rollouts: grader main38 hack prompts (cheat vs clean) and MBPP
test-split base rollouts (cheat vs honest_correct), reasoning on, balanced cells in round-robin order so any prefix is balanced.
    python fd_gain.py prep       laptop: picks the rollouts and builds the directions, acts/fd_gain/inputs.pt
    python fd_gain.py smoke      box: two rollouts at two layers into acts/fd_gain/smoke, with the stacked path checked against the model's own logits
    python fd_gain.py run        box, vLLM stopped: the forward passes, one acts/fd_gain/results/<i>.pt per rollout (scripts/fd_gain_box.sh wraps it)
    python fd_gain.py analyze    laptop: tables and plots from the results"""
import json
import random
import sys
import time
from pathlib import Path

import torch as t
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText

from mechtools import *

STAGE = sys.argv[1] if len(sys.argv) > 1 else "analyze"
MODEL_ID, REVISION = "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
D = 5120
VECTORS = Path("data/vectors/Qwen3.8-27B")
WEIGHTS = Path("acts/weights")   # jlens_gain.py's cache of single tensors
ARCHIVE = Path.home() / "wgmn/ortho_box_archive/59147695/models"   # Will's resid-bias rows
LENS = "qwen3.6-27b/j-lens/lens.pt"
ACTS = Path("acts/fd_gain")
INPUTS = ACTS / "inputs.pt"
RESULTS = ACTS / ("smoke" if STAGE == "smoke" else "results")
MBPP_RUNS = ["2026-11-04T19-19-26-00-00_mbpp_GgYyPkS3GFuuD8oMPc8atb", "2026-11-05T21-27-57-00-00_mbpp_n37JEr7fba6FazvWnDUkX4"]   # mbpp38 and mbpp38_x4
LAYERS = [32, 36, 40, 44, 48, 52, 56, 60]   # resid_pre L: row L of the train_vector layout, perturbed at the output of decoder layer L-1
PROBE_ROW = 52
N_PER_CELL = 16
MAX_TOKENS = dict(grader=400, mbpp=2000)
N_RANDOM, N_MLP, N_PULLBACK = 24, 12, 6
EPS, EPS_MAIN = [0.02, 0.05, 0.1, 0.2], 0.1   # fractions of the rollout's median residual norm at L; every direction at EPS_MAIN, the sweep classes at all of EPS
SWEEP = {"saved", "probe", "will", "pin"}
N_RANDOM_SWEEP = 8
PSETS = ["all", "cot_start", "cot_end"]   # every position; the newline after <think>; the newline before </think>
METRICS = ["dplus", "dminus", "central", "klplus", "klminus"]
SUMMARIES = ["at_start", "at_end", "cot_mean", "answer_mean", "from_perturbed_mean", "last"]
POS_METRICS = [2, 3, 4]   # the per-position arrays kept at EPS_MAIN: central, klplus, klminus
TOKENS_PER_BATCH, MAX_BATCH = 24000, 32
BUDGET_H = 5.5
SEED = 0
THINK_OPEN, THINK_CLOSE, NEWLINE = 248068, 248069, 198
if STAGE == "smoke": LAYERS = [40, 60]

#%% prep: the rollouts, N_PER_CELL per cell, round-robin over the cells
if STAGE == "prep":
    records = [json.loads(l) for p in sorted(Path("data/inspect/grader").glob("*.jsonl")) for l in open(p)]
    grader = [dict(env="grader", ids=r["ids"], cheat=bool(r["labels"]["cheat"]), task=r["labels"]["family"]) for r in records if r["config_id"] == "qwen3.8_low" and r["family"] == "all" and r["condition"] == "none" and r["model"] == "vllm/cb_base" and r["labels"]["prompts"] == "hack" and r["labels"]["sys"] == "none" and r["labels"]["cheat"] is not None]
    records = [json.loads(l) for run in MBPP_RUNS for l in open(f"data/inspect/mbpp/{run}.jsonl")]
    mbpp = [dict(env="mbpp", ids=r["ids"], cheat=bool(r["labels"]["cheat"]), task=r["labels"]["task_id"]) for r in records if r["model"] == "vllm/cb_base" and r["condition"] == "none" and r["labels"]["split"] == "test" and "bool" not in r["labels"]["corruption"] and (r["labels"]["cheat"] or r["labels"]["honest_correct"])]
    rng = random.Random(SEED)
    cells = {}
    for env, xs in [("grader", grader), ("mbpp", mbpp)]:
        for cheat in [True, False]:
            pool = [x for x in xs if x["cheat"] == cheat and len(x["ids"]) <= MAX_TOKENS[env] and x["ids"].count(THINK_OPEN) == 1 and x["ids"].count(THINK_CLOSE) == 1]
            cell = f"{env}_{'cheat' if cheat else 'clean' if env == 'grader' else 'honest'}"
            cells[cell] = rng.sample(pool, N_PER_CELL)
            print(f"{cyan}{cell}: {len(pool)} of {sum(x['cheat'] == cheat for x in xs)} within {MAX_TOKENS[env]} tokens, {N_PER_CELL} picked, lengths {sorted(len(x['ids']) for x in cells[cell])}{endc}")
    rollouts = [dict(x, cell=cell, p_start=x["ids"].index(THINK_OPEN) + 1, p_end=x["ids"].index(THINK_CLOSE) - 1) for i in range(N_PER_CELL) for cell, xs in cells.items() for x in [xs[i]]]
    assert all(x["ids"][x["p_start"]] == NEWLINE and x["ids"][x["p_end"]] == NEWLINE and x["p_start"] < x["p_end"] for x in rollouts)

#%% prep: the directions, [n, len(LAYERS), d_model] unit rows, a zero row where a direction has no row at that layer
if STAGE == "prep":
    names, classes, rows = [], [], []
    def add(name: str, cls: str, v: Tensor):
        names.append(name); classes.append(cls); rows.append(normed(v.float()))
    for p in sorted(VECTORS.glob("*.safetensors")):
        f = load_file(p)
        if "direction_unit" in f: add(p.stem, "saved", f["direction_unit"][LAYERS])
    probe = t.zeros(len(LAYERS), D)
    probe[LAYERS.index(PROBE_ROW)] = load_file(VECTORS / "cot_probe.safetensors")["w"]
    add("cot_probe", "probe", probe)
    for name, folder in [("will_g9", "cb_a3"), ("will_cm", "cb_cm")]:   # the bias on decoder layer L-1's output is -alpha times his unit row at resid_pre L
        f = load_file(ARCHIVE / folder / "model-resid-bias.safetensors")
        add(name, "will", -t.stack([f[f"model.language_model.layers.{L - 1}.mlp.down_proj.bias"] for L in LAYERS]))
    for name in ["grader_proj", "grader_kto", "mbpp_dom", "mbpp_kto"]: add(f"pin_{name}", "pin", load_file(VECTORS / f"{name}.safetensors")["mean_resid"][LAYERS])
    gen = t.Generator().manual_seed(SEED)
    for i in range(N_RANDOM): add(f"random_{i}", "random", t.randn(len(LAYERS), D, generator=gen))
    lens = t.load(hf_hub_download("camilablank/workspace-lenses", LENS), map_location="cpu", weights_only=False, mmap=True)
    scale = 1 + load_file(WEIGHTS / "Qwen3.6-27B/model.language_model.norm.weight.safetensors")["model.language_model.norm.weight"].float()
    unembed = load_file(WEIGHTS / "Qwen3.6-27B/lm_head.weight.safetensors")["lm_head.weight"]
    labels = t.load(WEIGHTS / "vocab_labels_36_k1024_s0.pt")
    mean = sum(unembed[i:i + 16384].float().sum(0) for i in range(0, len(unembed), 16384)) / len(unembed)
    counts = t.bincount(labels, minlength=1024)
    clusters = [k for k in t.randperm(1024, generator=gen).tolist() if counts[k] >= 20][:N_PULLBACK]
    sources = {"pullback_cluster": t.stack([normed(normed(unembed[labels == k].float() - mean).mean(0)) for k in clusters]), "pullback_random": normed(t.randn(N_PULLBACK, D, generator=gen))}
    for kind, src in sources.items():
        for j in range(N_PULLBACK): add(f"{kind}_{j}", kind, t.stack([(src[j] * scale) @ lens["J"][L].float() for L in LAYERS]))   # (c * s) @ J = J^T (s * c)
    add("zero", "zero", t.zeros(len(LAYERS), D))
    ACTS.mkdir(parents=True, exist_ok=True)
    t.save(dict(rollouts=rollouts, names=names, classes=classes, dirs=t.stack(rows), layers=LAYERS, clusters=clusters), INPUTS)
    print(f"{green}{len(rollouts)} rollouts, {len(names)} directions ({', '.join(f'{c} {classes.count(c)}' for c in dict.fromkeys(classes))}) saved to {INPUTS}{endc}")

#%% run: the model, the MLP directions of each layer from its weights, and the kwargs every decoder layer is called with
if STAGE in ("run", "smoke"):
    inputs = t.load(INPUTS, weights_only=False)
    rollouts, names, classes, dirs, cols = inputs["rollouts"], inputs["names"], inputs["classes"], inputs["dirs"], inputs["layers"]
    if STAGE == "smoke": rollouts = [rollouts[0], rollouts[2]]
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=REVISION, dtype=t.bfloat16, device_map="cuda").eval()
    text = model.model.language_model
    layers = text.layers
    assert len(layers) == 64
    gen = t.Generator().manual_seed(SEED + 1)
    for kind, pick in [("mlp_dec", lambda L, idx: layers[L - 1].mlp.down_proj.weight[:, idx].T), ("mlp_enc", lambda L, idx: layers[L].mlp.gate_proj.weight[idx] * (1 + layers[L].post_attention_layernorm.weight))]:
        extra = t.zeros(N_MLP, len(cols), D)
        for j, L in enumerate(cols):
            idx = t.randperm(layers[L].mlp.gate_proj.weight.shape[0], generator=gen)[:N_MLP].cuda()
            extra[:, j] = normed(pick(L, idx).float().cpu())
        names, classes, dirs = names + [f"{kind}_{i}" for i in range(N_MLP)], classes + [kind] * N_MLP, t.cat([dirs, extra])
    RESULTS.mkdir(parents=True, exist_ok=True)
    t.save(dict(names=names, classes=classes, dirs=dirs, layers=cols), RESULTS / "directions.pt")
    dirs = dirs.cuda()
    sweep = [d for d, c in enumerate(classes) if c in SWEEP] + [names.index(f"random_{i}") for i in range(N_RANDOM_SWEEP)]
    zero = names.index("zero")
    kw = [None] * 64
    def capture(i: int):
        def hook(module, args, kwargs): kw[i] = kwargs
        return hook
    for i, layer in enumerate(layers): layer.register_forward_pre_hook(capture(i), with_kwargs=True)
    def stack(h: Tensor, L: int) -> Tensor:   # h [B, T, d_model] at resid_pre L -> the final residual before the norm
        for i in range(L, 64):
            out = layers[i](h, **kw[i])
            h = out[0] if isinstance(out, tuple) else out
        return h
    def readout(r: Tensor) -> Tensor:   # final residual [T, d_model] -> next-token log-probs [T, vocab]
        return t.log_softmax(model.lm_head(text.norm(r)).float(), -1)
    def summaries(a: Tensor, x: dict, off: int) -> Tensor:   # a [5, T] per-position metrics -> [5, len(SUMMARIES)]
        return t.stack([a[:, x["p_start"]], a[:, x["p_end"]], a[:, x["p_start"] + 1:x["p_end"]].mean(1), a[:, x["p_end"] + 1:].mean(1), a[:, off:].mean(1), a[:, -1]], 1).cpu()

#%% run: per rollout and layer, every (perturbation set, direction, eps) as a +eps / -eps pair, batched suffix forwards from the clean hidden state
if STAGE in ("run", "smoke"):
    t0 = time.time()
    for i, x in enumerate(rollouts):
        if (RESULTS / f"{i:03d}.pt").exists(): continue
        if time.time() - t0 > BUDGET_H * 3600:
            print(f"{yellow}budget of {BUDGET_H} h spent after {i} rollouts{endc}")
            break
        t1 = time.time()
        ids = t.tensor(x["ids"], device="cuda")
        T = len(ids)
        B = max(2, min(MAX_BATCH, TOKENS_PER_BATCH // T // 2 * 2))
        offs = dict(all=0, cot_start=x["p_start"], cot_end=x["p_end"])
        result = dict(**{k: x[k] for k in ["cell", "env", "cheat", "task", "p_start", "p_end", "ids"]}, T=T, B=B, layers={})
        with t.no_grad():
            out = model(input_ids=ids[None].expand(B, -1), use_cache=False, output_hidden_states=True, logits_to_keep=1)
            hs = {L: out.hidden_states[L] for L in LAYERS}
            last_logits = out.logits[0, -1].float()
            del out
            for L in LAYERS:
                j = cols.index(L)
                m = hs[L][0, 1:].float().norm(dim=-1).median().item()
                r0 = stack(hs[L], L)[0]   # the captured kwargs carry batch B, so the clean reference runs at B too
                lp0 = readout(r0)
                p0 = lp0.exp()
                if STAGE == "smoke":
                    mine = model.lm_head(text.norm(r0[-1:]))[0].float()
                    print(f"{gray}L{L}: stacked path vs the model's own last-position logits: max |diff| {(mine - last_logits).abs().max():.3f}, argmax {'same' if mine.argmax() == last_logits.argmax() else 'DIFFERENT'}; median resid norm {m:.1f}{endc}")
                present = [d for d in range(len(names)) if d == zero or dirs[d, j].any()]
                pairs = [(ps, d, e) for ps in PSETS for d in present for e in (EPS if d in sweep else [EPS_MAIN])]
                while len(pairs) % (B // 2): pairs.append(("all", zero, EPS_MAIN))
                summ = t.zeros(len(pairs), len(METRICS), len(SUMMARIES))
                pos = {ps: [] for ps in PSETS}
                pos_idx = {ps: [] for ps in PSETS}
                for k in range(0, len(pairs), B // 2):
                    chunk = pairs[k:k + B // 2]
                    h = hs[L][:B].clone()
                    for b, (ps, d, e) in enumerate(chunk):
                        sel = slice(None) if ps == "all" else slice(offs[ps], offs[ps] + 1)
                        v = e * m * dirs[d, j]
                        h[2 * b, sel] = (h[2 * b, sel].float() + v).to(h.dtype)
                        h[2 * b + 1, sel] = (h[2 * b + 1, sel].float() - v).to(h.dtype)
                    r = stack(h, L).float()
                    for b, (ps, d, e) in enumerate(chunk):
                        a = t.stack([(r[2 * b] - r0.float()).norm(dim=-1), (r[2 * b + 1] - r0.float()).norm(dim=-1), (r[2 * b] - r[2 * b + 1]).norm(dim=-1) / 2, (p0 * (lp0 - readout(r[2 * b].to(t.bfloat16)))).sum(-1), (p0 * (lp0 - readout(r[2 * b + 1].to(t.bfloat16)))).sum(-1)])
                        summ[k + b] = summaries(a, x, offs[ps])
                        if e == EPS_MAIN:
                            pos[ps].append(a[POS_METRICS, offs[ps]:].half().cpu())
                            pos_idx[ps].append(k + b)
                result["layers"][L] = dict(norm=m, final_norm=r0.float().norm(dim=-1).half().cpu(), pairs=pairs, summ=summ, pos={ps: t.stack(pos[ps]) for ps in PSETS}, pos_idx=pos_idx)
                print(f"{gray}  L{L}: {len(pairs)} pairs in {len(pairs) * 2 // B} batches of {B}, {time.time() - t1:.0f}s so far{endc}", flush=True)
        t.save(result, RESULTS / f"{i:03d}.pt")
        dt = time.time() - t1
        print(f"{cyan}{i:03d} {x['cell']} T={T} B={B}: {dt:.0f}s; {(time.time() - t0) / 3600:.2f} h elapsed, {(time.time() - t0) / (i + 1) * (len(rollouts) - i - 1) / 3600:.2f} h to go at this rate{endc}", flush=True)
    print(f"{green}done: {len(list(RESULTS.glob('[0-9]*.pt')))} rollouts in {RESULTS}, {(time.time() - t0) / 3600:.2f} h{endc}")

#%% analyze: load; per (rollout, layer, perturbation set, eps) a [n_dirs] vector of each metric's summary, the shift norms divided by the perturbation norm (the gain)
if STAGE == "analyze":
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    results = [t.load(p, weights_only=False) for p in sorted(RESULTS.glob("[0-9]*.pt"))]
    saved = t.load(RESULTS / "directions.pt", weights_only=False)
    names, classes = saved["names"], saved["classes"]
    cells = list(dict.fromkeys(r["cell"] for r in results))
    LAYERS = sorted(results[0]["layers"])
    print(f"{cyan}{len(results)} rollouts: " + ", ".join(f"{c} {sum(r['cell'] == c for r in results)}" for c in cells) + f"; {len(names)} directions; layers {LAYERS}{endc}")
    rand = [d for d, c in enumerate(classes) if c == "random"]
    palette = ["#3987e5", "#d95926", "#199e70", "#c98500", "#8e6bd6", "#d04f9e", "#4fb0c6", "#a8b820", "#e07b7b", "#7a9cc6", "#c6a27a", "#6fcf97", "#b07aa1", "#f2c14e", "#59a14f", "#e15759", "#76b7b2", "#ff9da7", "#9c755f", "#bab0ac"]
    zero = names.index("zero")
    READOUTS = dict(all=["from_perturbed_mean", "cot_mean", "answer_mean", "last"], cot_start=["at_start", "cot_mean", "answer_mean", "last"], cot_end=["at_end", "answer_mean", "last"])
    for r in results:
        for L, lay in r["layers"].items():
            lay["ps"] = [p for p, d, e in lay["pairs"]]
            lay["d"] = t.tensor([d for p, d, e in lay["pairs"]])
            lay["e"] = t.tensor([e for p, d, e in lay["pairs"]])
    def table(r: dict, L: int, ps: str, e: float, metric: str, summary: str) -> Tensor:   # [n_dirs], nan where the direction has no row at L; gains for the shift metrics, raw KL for the KL ones
        lay = r["layers"][L]
        mask = t.tensor([p == ps for p in lay["ps"]]) & (lay["e"] == e)
        out = t.full((len(names),), float("nan"))
        out[lay["d"][mask]] = lay["summ"][mask, METRICS.index(metric), SUMMARIES.index(summary)] / (e * lay["norm"] if metric.startswith("d") or metric == "central" else 1)
        return out
    def ratios(ps: str, summary: str, e: float = EPS_MAIN, metric: str = "central", which: list[dict] | None = None) -> Tensor:   # [n_rollouts, n_dirs, n_layers] ratio to the rollout's random median
        out = []
        for r in (which or results):
            per = []
            for L in LAYERS:
                g = table(r, L, ps, e, metric, summary)
                per.append(g / g[rand].nanmedian())
            out.append(t.stack(per, 1))
        return t.stack(out)

#%% analyze: the main tables, every direction's median ratio to the random band at EPS_MAIN, by layer, for each perturbation set's own readout; the noise floor from the zero pairs
if STAGE == "analyze":
    groups = {n: [d] for d, n in enumerate(names) if classes[d] not in ("random", "mlp_dec", "mlp_enc", "pullback_cluster", "pullback_random", "zero")}
    groups |= {f"{c} (median of {classes.count(c)})": [d for d, cc in enumerate(classes) if cc == c] for c in ["mlp_dec", "mlp_enc", "pullback_cluster", "pullback_random"]}
    groups["zero (noise floor)"] = [zero]
    def show(ps: str, summary: str, metric: str = "central", e: float = EPS_MAIN) -> dict:
        R = ratios(ps, summary, e, metric)
        band = R[:, rand].flatten(0, 1)
        lo, hi = band.nan_to_num(float("nan")).nanquantile(t.tensor([0.05, 0.95]), dim=0)
        print(f"\n{cyan}{metric} / random median, {ps} perturbation, {summary} readout, eps {e}, {len(results)} rollouts; random 5th-95th pct per layer:{endc}")
        print(f"{'':34s}" + "".join(f"L{L:<9d}" for L in LAYERS))
        print(f"{gray}{'random band':34s}" + "".join(f"{a:.2f}-{b:.2f} " for a, b in zip(lo.tolist(), hi.tolist())) + endc)
        rows = {}
        for g, ds in groups.items():
            med = R[:, ds].nanmedian(1).values.nanmedian(0).values if len(ds) > 1 else R[:, ds[0]].nanmedian(0).values
            rows[g] = med.tolist()
            flag = [green if v > hi[k] else yellow if v < lo[k] else "" for k, v in enumerate(med.tolist())]
            print(f"{g:34s}" + "".join(f"{f}{v:<10.2f}{endc if f else ''}" for f, v in zip(flag, med.tolist())))
        return dict(band_lo=lo.tolist(), band_hi=hi.tolist(), rows=rows)
    tables = {}
    for ps, summaries in READOUTS.items():
        for summary in summaries[:1] + (["cot_mean"] if ps != "cot_end" else []):
            tables[f"{ps}/{summary}/central"] = show(ps, summary)
            tables[f"{ps}/{summary}/kl"] = show(ps, summary, "klplus")

#%% analyze: context dependence, each direction's ratio by rollout cell (median over layers and the cell's rollouts), every position perturbed, mean over positions
if STAGE == "analyze":
    print(f"\n{cyan}ratio to random median by cell: all positions perturbed, mean over positions (central), median over layers; then the same at the newline before </think>{endc}")
    print(f"{'':34s}" + "".join(f"{c:<16s}" for c in cells))
    context = {}
    for ps, summary in [("all", "from_perturbed_mean"), ("cot_end", "at_end"), ("cot_start", "at_start")]:
        print(f"{gray}{ps} / {summary}{endc}")
        for g, ds in groups.items():
            vals = []
            for c in cells:
                R = ratios(ps, summary, which=[r for r in results if r["cell"] == c])
                vals.append((R[:, ds].nanmedian(1).values if len(ds) > 1 else R[:, ds[0]]).nanmedian(0).values.nanmedian().item())
            context[f"{ps}/{g}"] = dict(zip(cells, vals))
            print(f"{g:34s}" + "".join(f"{v:<16.2f}" for v in vals))

#%% analyze: scale, the sweep directions' gain and KL against eps (all positions, mean over positions), random alongside; linearity of the random gain
if STAGE == "analyze":
    print(f"\n{cyan}gain ratio to random median by eps (all positions, mean over positions), median over rollouts and layers; last column the random directions' raw gain{endc}")
    print(f"{'':34s}" + "".join(f"eps {e:<10}" for e in EPS) + "random gain " + "".join(f"eps {e:<8}" for e in EPS))
    scale = {}
    raw_rand = [t.stack([table(r, L, "all", e, "central", "from_perturbed_mean")[rand] for r in results for L in LAYERS]).nanmedian().item() for e in EPS]
    for g, ds in groups.items():
        if classes[ds[0]] not in SWEEP: continue
        vals = [ratios("all", "from_perturbed_mean", e)[:, ds[0]].nanmedian().item() for e in EPS]
        scale[g] = dict(zip(map(str, EPS), vals))
        print(f"{g:34s}" + "".join(f"{v:<14.2f}" for v in vals) + "            " + "".join(f"{v:<12.3f}" for v in raw_rand))
    asym = t.stack([t.stack([table(r, L, "all", EPS_MAIN, "dplus", "from_perturbed_mean") / table(r, L, "all", EPS_MAIN, "dminus", "from_perturbed_mean") for L in LAYERS], 1) for r in results])   # [n_rollouts, n_dirs, n_layers]
    print(f"\n{cyan}asymmetry ||h(+eps) - h|| / ||h(-eps) - h||, all positions, median over rollouts and layers: " + ", ".join(f"{names[d]} {asym[:, d].nanmedian():.2f}" for d in range(len(names)) if classes[d] in ("saved", "probe", "will", "pin")) + f"; random {asym[:, rand].nanmedian():.2f}{endc}")

#%% analyze: per position, no averaging: the fraction of CoT positions where a direction's gain beats the random 95th percentile at that same position (0.05 under the null), every position perturbed and the newline after <think> perturbed
if STAGE == "analyze":
    def position_gains(r: dict, L: int, ps: str) -> tuple[list[int], Tensor]:   # the directions with per-position arrays and their central gains [n, T'] at EPS_MAIN, positions from the perturbed one
        lay = r["layers"][L]
        return [lay["pairs"][k][1] for k in lay["pos_idx"][ps]], lay["pos"][ps][:, 0].float() / (EPS_MAIN * lay["norm"])
    beats = {}
    print(f"\n{cyan}fraction of CoT positions where the direction's gain at that position beats the random 95th percentile there (0.05 under the null), median over rollouts, by layer{endc}")
    for ps in ["all", "cot_start"]:
        print(f"{gray}{ps}{endc}" + " " * (34 - len(ps) - 0) + "".join(f"L{L:<9d}" for L in LAYERS))
        frac = t.full((len(results), len(names), len(LAYERS)), float("nan"))
        for i, r in enumerate(results):
            lo = r["p_start"] + 1 - (0 if ps == "all" else r["p_start"])
            hi = r["p_end"] - (0 if ps == "all" else r["p_start"])
            for k, L in enumerate(LAYERS):
                ds, g = position_gains(r, L, ps)
                g = g[:, lo:hi]
                q95 = g[[ds.index(d) for d in rand]].quantile(0.95, dim=0)
                for n, d in enumerate(ds): frac[i, d, k] = (g[n] > q95).float().mean()
        for g_name, ds in groups.items():
            med = (frac[:, ds].nanmedian(1).values if len(ds) > 1 else frac[:, ds[0]]).nanmedian(0).values
            beats[f"{ps}/{g_name}"] = med.tolist()
            print(f"{g_name:34s}" + "".join(f"{green if v > 0.15 else ''}{v:<10.3f}{endc if v > 0.15 else ''}" for v in med.tolist()))
    PLOT_LAYER = 52
    shown = [n for n in ["grader_proj", "grader_kto", "grader_dpo", "nocot_dom_last", "mbpp_dom", "mbpp_kto", "will_g9", "will_cm", "cot_probe", "pin_grader_proj"] if n in names]
    picks = [min((r for r in results if r["cell"] == c), key=lambda r: r["T"]) for c in cells]
    fig = make_subplots(rows=len(picks), cols=1, subplot_titles=[f"{r['cell']} (T {r['T']}, CoT {r['p_start'] + 1}-{r['p_end'] - 1})" for r in picks], vertical_spacing=0.06)
    for row, r in enumerate(picks, 1):
        if PLOT_LAYER not in r["layers"]: continue
        ds, g = position_gains(r, PLOT_LAYER, "all")
        rg = g[[ds.index(d) for d in rand]]
        x = list(range(g.shape[1]))
        lo, hi = rg.quantile(0.05, dim=0), rg.quantile(0.95, dim=0)
        fig.add_trace(go.Scatter(x=x + x[::-1], y=hi.tolist() + lo.tolist()[::-1], fill="toself", fillcolor="rgba(195,194,183,0.2)", line=dict(width=0), name="random 5-95%", showlegend=row == 1, legendgroup="random"), row=row, col=1)
        fig.add_trace(go.Scatter(x=x, y=rg.median(0).values.tolist(), line=dict(color="#c3c2b7", width=1), name="random median", showlegend=row == 1, legendgroup="rmed"), row=row, col=1)
        for n_i, n in enumerate(shown):
            if names.index(n) in ds: fig.add_trace(go.Scatter(x=x, y=g[ds.index(names.index(n))].tolist(), line=dict(color=palette[n_i % len(palette)], width=1), name=n, showlegend=row == 1, legendgroup=n), row=row, col=1)
        for pos in [r["p_start"], r["p_end"]]: fig.add_vline(x=pos, line=dict(color="#c98500", dash="dot", width=1), row=row, col=1)
    fig.update_layout(**DARK, height=350 * len(picks), width=1800, title=f"fd_gain: per-position central gain at L{PLOT_LAYER}, eps {EPS_MAIN}, every position perturbed; dotted lines the newlines after <think> and before </think>")
    write_dark_html(fig, str(ACTS / "fd_gain_positions.html"))

#%% analyze: plot, one panel per (perturbation set, readout): ratio to the random median by layer, the saved rows named, the control classes as medians, the random band shaded
if STAGE == "analyze":
    panels = [(ps, s) for ps, ss in READOUTS.items() for s in ss]
    fig = make_subplots(rows=3, cols=4, subplot_titles=[f"{ps}: {s}" for ps, s in panels], vertical_spacing=0.08, horizontal_spacing=0.05)
    for k, (ps, s) in enumerate(panels):
        row, col = k // 4 + 1, k % 4 + 1
        R = ratios(ps, s)
        band = R[:, rand].flatten(0, 1)
        lo, hi = band.nanquantile(t.tensor([0.05, 0.95]), dim=0)
        fig.add_trace(go.Scatter(x=LAYERS + LAYERS[::-1], y=hi.tolist() + lo.tolist()[::-1], fill="toself", fillcolor="rgba(195,194,183,0.15)", line=dict(width=0), name="random 5-95%", showlegend=k == 0, legendgroup="random"), row=row, col=col)
        for g_i, (g, ds) in enumerate(groups.items()):
            med = R[:, ds].nanmedian(1).values.nanmedian(0).values if len(ds) > 1 else R[:, ds[0]].nanmedian(0).values
            fig.add_trace(go.Scatter(x=LAYERS, y=med.tolist(), mode="lines+markers", name=g, line=dict(color=palette[g_i % len(palette)], dash="dot" if len(ds) > 1 or g.startswith("zero") else "solid"), showlegend=k == 0, legendgroup=g), row=row, col=col)
        fig.update_yaxes(type="log", row=row, col=col)
    fig.update_layout(**DARK, height=1300, width=2000, title=f"fd_gain: central-difference gain at eps {EPS_MAIN} relative to the random median, {len(results)} rollouts, median over rollouts; rows = perturbation set, panels = readout")
    write_dark_html(fig, str(ACTS / "fd_gain.html"))
    (ACTS / "fd_gain.json").write_text(json.dumps(dict(n_rollouts=len(results), cells={c: sum(r["cell"] == c for r in results) for c in cells}, layers=LAYERS, eps=EPS, eps_main=EPS_MAIN, tables=tables, context=context, scale=scale, beats=beats), indent=1))
    print(f"{green}wrote {ACTS / 'fd_gain.html'} and {ACTS / 'fd_gain.json'}{endc}")
