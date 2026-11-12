#%%
"""Which MLP neurons write the cheat-minus-honest gap along our saved directions, in base Qwen3.8-27B rollouts. The residual stream is a plain sum, so the projection
of resid_pre L onto a unit row v splits exactly into the embedding, every lower layer's mixer output (attention or the linear-attention block) and every lower MLP
neuron: a_i (w_i . v), a_i the input to down_proj (which has no bias) and w_i its column. Direct writes only: a neuron that makes later neurons write gets no credit.
Per rollout the box saves, at two position sets (the mean over the CoT tokens strictly between the newline after <think> and the newline before </think>, and that
last newline alone), every neuron's activation, every MLP's output and the residual at every layer; any direction is then arithmetic on the laptop. A class mean is
cell-balanced (grader: family and side, MBPP: task; only cells holding both classes), the gap is the cheating mean minus the honest mean. Directions: grader_kto,
mbpp_kto, mbpp_dom, Will's G9 rows and cot_probe, against random unit directions, shuffled labels within cells and two disjoint halves of the cells.
    python neurons.py prep       laptop: picks the rollouts, acts/neurons/inputs.pt
    python neurons.py smoke      laptop: the run stage on Qwen3.5-2B (same model class) for 4 rollouts, into acts/neurons/smoke; checks the hooks only
    python neurons.py run        box, vLLM stopped: one forward per rollout, acts/neurons/results/{a,m,h}.pt
    python neurons.py analyze    laptop: tables, acts/neurons/neurons.json and the plots acts/neurons/neurons_{share,layers}.html"""
import json
import random
import struct
import sys
from pathlib import Path

import requests
import torch as t
from huggingface_hub import hf_hub_download, hf_hub_url
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText

from mechtools import *

STAGE = sys.argv[1] if len(sys.argv) > 1 else "analyze"
MODEL_ID, REVISION = ("Qwen/Qwen3.5-2B", None) if STAGE == "smoke" else ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
VECTORS = Path("data/vectors/Qwen3.8-27B")
WEIGHTS = Path("acts/weights")   # jlens_gain.py's cache of single tensors
ARCHIVE = Path.home() / "wgmn/ortho_box_archive/59147695/models"   # Will's resid-bias rows
ACTS = Path("acts/neurons")
INPUTS = ACTS / "inputs.pt"
RESULTS = ACTS / ("smoke" if STAGE == "smoke" else "results")
MBPP_RUNS = ["2026-11-04T19-19-26-00-00_mbpp_GgYyPkS3GFuuD8oMPc8atb", "2026-11-05T21-27-57-00-00_mbpp_n37JEr7fba6FazvWnDUkX4"]   # mbpp38 and mbpp38_x4
DROPPED = ["pets", "fruit_vegetable"]   # grader families no longer in rhenvs
ENVS = ["grader", "mbpp"]
POS = ["cot_mean", "cot_last"]
MAX_TOKENS = dict(grader=800, mbpp=4096)
MIN_CLASS = dict(grader=8, mbpp=1)   # rollouts of each class a cell needs
MAX_CLASS = 16                       # rollouts of each class taken from a cell
MIN_COT = 8
READ = [32, 36, 40, 44, 48, 52, 56, 60]   # resid_pre L: row L of the train_vector layout
PROBE_ROW = 52
MAIN = 52          # the readout layer of the plots, the one every direction has a row at
N_RANDOM = 24
N_SHUFFLE = 5
TOP = 1000
SEED = 0
THINK_OPEN, THINK_CLOSE, NEWLINE = 248068, 248069, 198

#%% prep: base rollouts with one reasoning block, grouped into cells; cells with both classes kept, at most MAX_CLASS of each class from a cell
if STAGE == "prep":
    records = [json.loads(l) for p in sorted(Path("data/inspect/grader").glob("*.jsonl")) for l in open(p)]
    pool = [dict(env="grader", ids=r["ids"], cheat=bool(r["labels"]["cheat"]), cell=f"{r['labels']['family']}.{r['labels']['side']}") for r in records if r["config_id"] == "qwen3.8_low" and r["condition"] == "none" and r["model"] == "vllm/cb_base" and r["labels"]["prompts"] == "hack" and r["labels"]["sys"] == "none" and r["labels"]["cheat"] is not None and r["labels"]["family"] not in DROPPED]
    records = [json.loads(l) for run in MBPP_RUNS for l in open(f"data/inspect/mbpp/{run}.jsonl")]
    pool += [dict(env="mbpp", ids=r["ids"], cheat=bool(r["labels"]["cheat"]), cell=f"task{r['labels']['task_id']}") for r in records if r["model"] == "vllm/cb_base" and r["condition"] == "none" and r["labels"]["split"] == "test" and "bool" not in r["labels"]["corruption"] and (r["labels"]["cheat"] or r["labels"]["honest_correct"])]
    pool = [x for x in pool if len(x["ids"]) <= MAX_TOKENS[x["env"]] and x["ids"].count(THINK_OPEN) == 1 and x["ids"].count(THINK_CLOSE) == 1]
    pool = [dict(x, p_start=x["ids"].index(THINK_OPEN) + 1, p_end=x["ids"].index(THINK_CLOSE) - 1) for x in pool]
    pool = [x for x in pool if x["ids"][x["p_start"]] == NEWLINE and x["ids"][x["p_end"]] == NEWLINE and x["p_end"] - x["p_start"] > MIN_COT]
    rng = random.Random(SEED)
    rollouts = []
    for env in ENVS:
        for cell in sorted({x["cell"] for x in pool if x["env"] == env}):
            classes = [[x for x in pool if x["env"] == env and x["cell"] == cell and x["cheat"] == cheat] for cheat in [True, False]]
            if min(map(len, classes)) < MIN_CLASS[env]: continue
            for xs in classes: rollouts += rng.sample(xs, min(len(xs), MAX_CLASS))
        mine = [x for x in rollouts if x["env"] == env]
        print(f"{cyan}{env}: {len(mine)} rollouts ({sum(x['cheat'] for x in mine)} cheating) in {len({x['cell'] for x in mine})} cells, {sum(len(x['ids']) for x in mine)} tokens, longest {max(len(x['ids']) for x in mine)}{endc}")
        if env == "grader": print(f"{gray}" + ", ".join(f"{c} {sum(x['cell'] == c and x['cheat'] for x in mine)}/{sum(x['cell'] == c and not x['cheat'] for x in mine)}" for c in sorted({x['cell'] for x in mine})) + endc)
    ACTS.mkdir(parents=True, exist_ok=True)
    t.save(rollouts, INPUTS)
    print(f"{green}{len(rollouts)} rollouts saved to {INPUTS}{endc}")

#%% run: per rollout one forward; the down_proj input (the neuron activations), the MLP output and the residual at every layer, pooled to the two position sets
if STAGE in ("run", "smoke"):
    rollouts = t.load(INPUTS, weights_only=False)
    if STAGE == "smoke": rollouts = rollouts[::len(rollouts) // 4][:4]
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=REVISION, dtype=t.bfloat16, device_map="cuda").eval()
    layers = model.model.language_model.layers
    n_layers, (d_model, d_mlp) = len(layers), layers[0].mlp.down_proj.weight.shape
    assert all(layer.mlp.down_proj.bias is None for layer in layers)
    a = t.zeros(len(rollouts), 2, n_layers, d_mlp, dtype=t.float16)
    m = t.zeros(len(rollouts), 2, n_layers, d_model)
    h = t.zeros(len(rollouts), 2, n_layers + 1, d_model)   # h[:, :, L] is resid_pre L, the last one the output of the last layer before the final norm
    cur = [0, 0, 0]   # the rollout's index and its two newlines
    def keep(dst: Tensor, x: Tensor):   # x [1, T, d] pooled to [2, d], the CoT mean and the newline before </think>; returns None, a hook's return value would replace the module's
        dst.copy_(t.stack([x[0, cur[1] + 1:cur[2]].float().mean(0), x[0, cur[2]].float()]))
    for i, layer in enumerate(layers):
        layer.mlp.down_proj.register_forward_pre_hook(lambda module, args, i=i: keep(a[cur[0], :, i], args[0]))
        layer.mlp.register_forward_hook(lambda module, args, out, i=i: keep(m[cur[0], :, i], out))
        layer.register_forward_hook(lambda module, args, out, i=i: keep(h[cur[0], :, i + 1], out[0] if isinstance(out, tuple) else out))
    layers[0].register_forward_pre_hook(lambda module, args: keep(h[cur[0], :, 0], args[0]))
    with t.no_grad():
        for n, x in enumerate(pbar(rollouts, desc="rollouts")):
            cur[:] = [n, x["p_start"], x["p_end"]]
            model(input_ids=t.tensor([x["ids"]], device="cuda"), use_cache=False, logits_to_keep=1)
    assert a.isfinite().all() and m.isfinite().all() and h.isfinite().all()
    rel = t.stack([(a[:, :, i].cuda().float() @ layer.mlp.down_proj.weight.float().T - m[:, :, i].cuda()).norm(dim=-1) / m[:, :, i].cuda().norm(dim=-1) for i, layer in enumerate(layers)])
    print(f"{cyan}neurons times down_proj against the MLP's own output, relative error over (layer, rollout, position set): median {rel.median():.4f}, max {rel.max():.4f}; largest |activation| {a.abs().max():.1f}{endc}")
    assert rel.max() < 0.05
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, x in [("a", a), ("m", m), ("h", h)]: t.save(x, RESULTS / f"{name}.pt")
    print(f"{green}{len(rollouts)} rollouts, a {tuple(a.shape)}, m {tuple(m.shape)}, h {tuple(h.shape)} saved to {RESULTS}{endc}")

#%% analyze: one down_proj weight of the checkpoint by byte range, cached
def fetch(name: str) -> Tensor:
    path = WEIGHTS / MODEL_ID.split("/")[1] / f"{name}.safetensors"
    if path.exists(): return load_file(path)[name]
    url = hf_hub_url(MODEL_ID, json.load(open(hf_hub_download(MODEL_ID, "model.safetensors.index.json", revision=REVISION)))["weight_map"][name], revision=REVISION)
    header_len = struct.unpack("<Q", requests.get(url, headers={"Range": "bytes=0-7"}).content)[0]
    meta = json.loads(requests.get(url, headers={"Range": f"bytes=8-{7 + header_len}"}).content)[name]
    begin, end = meta["data_offsets"]
    buf = requests.get(url, headers={"Range": f"bytes={8 + header_len + begin}-{8 + header_len + end - 1}"}).content
    assert meta["dtype"] == "BF16" and len(buf) == end - begin, (meta["dtype"], len(buf), end - begin)
    x = t.frombuffer(bytearray(buf), dtype=t.bfloat16).reshape(meta["shape"])
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({name: x}, path)
    return x

#%% analyze: rollout weights, one row per mean: per env the cell-balanced cheating mean, honest mean and their gap, the gap on two halves of the cells, the gap under labels shuffled within cells
if STAGE == "analyze":
    rollouts = t.load(INPUTS, weights_only=False)
    a, m, h = (t.load(RESULTS / f"{k}.pt", mmap=True) for k in "amh")
    assert len(a) == len(rollouts)
    n_layers, d_mlp, D = a.shape[2], a.shape[3], h.shape[3]
    rng = random.Random(SEED)
    rows, W = [], []
    def means(env: str, cheat: list[bool], cells: list[str]) -> Tensor:   # [2, n_rollouts] weights of the cheating and of the honest class mean
        out = t.zeros(2, len(rollouts))
        for c in cells:
            for k in range(2):
                members = [n for n, x in enumerate(rollouts) if x["env"] == env and x["cell"] == c and cheat[n] == (k == 0)]
                out[k, members] = 1 / len(members) / len(cells)
        return out
    for env in ENVS:
        cells = sorted({x["cell"] for x in rollouts if x["env"] == env})
        labels = [x["cheat"] for x in rollouts]
        both = means(env, labels, cells)
        halves = rng.sample(cells, len(cells))
        rows += [f"{env}/cheat", f"{env}/honest", f"{env}/gap", f"{env}/gap_a", f"{env}/gap_b"]
        W += [both[0], both[1], both[0] - both[1]] + [w[0] - w[1] for w in [means(env, labels, halves[::2]), means(env, labels, halves[1::2])]]
        for s in range(N_SHUFFLE):
            shuffled = list(labels)
            for c in cells:
                members = [n for n, x in enumerate(rollouts) if x["env"] == env and x["cell"] == c]
                for n, v in zip(members, rng.sample([labels[n] for n in members], len(members))): shuffled[n] = v
            w = means(env, shuffled, cells)
            rows.append(f"{env}/shuffle{s}")
            W.append(w[0] - w[1])
    W = t.stack(W)
    da = t.stack([t.stack([W.cuda() @ a[:, p, l].cuda().float() for l in range(n_layers)], 1) for p in range(2)], 1).cpu()   # [n_rows, 2, n_layers, d_mlp]
    dm, dh = t.einsum("wn,npld->wpld", W, m), t.einsum("wn,npld->wpld", W, h)
    print(f"{cyan}{len(rollouts)} rollouts: " + ", ".join(f"{env} {sum(x['env'] == env and x['cheat'] for x in rollouts)} cheating / {sum(x['env'] == env and not x['cheat'] for x in rollouts)} honest in {len({x['cell'] for x in rollouts if x['env'] == env})} cells" for env in ENVS) + endc)

#%% analyze: the directions, a unit row at each readout layer they have one at, and every neuron's dot with every row
if STAGE == "analyze":
    dirs = {}
    for name in ["grader_kto", "mbpp_kto", "mbpp_dom"]:
        u = load_file(VECTORS / f"{name}.safetensors")["direction_unit"].float()
        dirs[name] = {L: u[L] for L in READ if u[L].any()}
    bias = load_file(ARCHIVE / "cb_a3/model-resid-bias.safetensors")   # the bias on decoder layer L-1's output is -alpha times his unit row at resid_pre L
    dirs["will_g9"] = {L: normed(-bias[f"model.language_model.layers.{L - 1}.mlp.down_proj.bias"].float()) for L in READ}
    dirs["cot_probe"] = {PROBE_ROW: normed(load_file(VECTORS / "cot_probe.safetensors")["w"].float())}
    NAMED = list(dirs)
    gen = t.Generator().manual_seed(SEED)
    for i in range(N_RANDOM): dirs[f"random_{i}"] = {L: normed(t.randn(D, generator=gen)) for L in READ}
    keys = [(name, L) for name, rows_ in dirs.items() for L in rows_]
    V = t.stack([dirs[name][L] for name, L in keys])
    assert V.isfinite().all() and ((V.norm(dim=1) - 1).abs() < 1e-3).all()
    P = t.zeros(n_layers, len(keys), d_mlp)
    worst = 0
    for l in pbar(range(n_layers), desc="down_proj"):
        w = fetch(f"model.language_model.layers.{l}.mlp.down_proj.weight").cuda().float()
        P[l] = (V.cuda() @ w).cpu()
        worst = max(worst, ((da[:, :, l].cuda() @ w.T - dm[:, :, l].cuda()).norm(dim=-1) / dm[:, :, l].cuda().norm(dim=-1)).max().item())
    print(f"{cyan}fetched down_proj times the mean activations against the box's MLP outputs: largest relative error {worst:.4f}{endc}")
    assert worst < 0.05

#%% analyze: per direction and readout layer, each mean's projection split into embedding, mixers, MLPs and neurons; neurons sorted by |write|, cumulative sums on a log grid
if STAGE == "analyze":
    groups = {}
    for n, x in enumerate(rollouts): groups.setdefault((x["env"], x["cell"], x["cheat"]), []).append(n)
    env_of = t.tensor([ENVS.index(x["env"]) for x in rollouts])
    out = {}
    for k, (name, L) in enumerate(pbar(keys, desc="directions")):
        v = V[k]
        C = (da[:, :, :L].cuda() * P[:L, k].cuda()).flatten(2).double()   # [n_rows, 2, L * d_mlp] each neuron's write to the mean's projection at resid_pre L
        order = C.abs().argsort(dim=-1, descending=True)
        S = C.gather(-1, order)
        grid = t.unique(t.cat([t.logspace(0, 6.1, 150).round().long(), t.tensor([10, 100, 1000, 10000, 100000, C.shape[-1]])]).clamp(max=C.shape[-1]))
        x = h[:, :, L] @ v   # [n_rollouts, 2] every rollout's projection
        within = t.zeros_like(x)
        for members in groups.values(): within[members] = x[members] - x[members].mean(0)
        std = t.stack([(within[env_of == e] ** 2).sum(0) / ((env_of == e).sum() - sum(g[0] == env for g in groups)) for e, env in enumerate(ENVS)]).sqrt()   # [n_envs, 2] pooled within cell and class
        out[name, L] = dict(
            total=dh[:, :, L] @ v, embed=dh[:, :, 0] @ v, mlp=dm[:, :, :L] @ v, mixer=(dh[:, :, 1:L + 1] - dh[:, :, :L] - dm[:, :, :L]) @ v,   # [n_rows, 2] and [n_rows, 2, L]
            neurons=C.sum(-1).float().cpu(), abs=C.abs().sum(-1).float().cpu(), grid=grid, cum=S.cumsum(-1)[..., grid.cuda() - 1].float().cpu(), cum_abs=S.abs().cumsum(-1)[..., grid.cuda() - 1].float().cpu(),
            top=order[..., :TOP].cpu(), top_write=S[..., :TOP].float().cpu(), std=std,
            half_r=t.stack([t.stack([t.corrcoef(t.stack([C[rows.index(f"{env}/gap_a"), p, order[rows.index(f"{env}/gap_a"), p, :TOP]], C[rows.index(f"{env}/gap_b"), p, order[rows.index(f"{env}/gap_a"), p, :TOP]]]))[0, 1] for p in range(2)]) for env in ENVS]).float().cpu())   # the TOP neurons picked on one half of the cells, their writes there against their writes on the other half (0 under noise)
    check = max(((o["neurons"] - o["mlp"].sum(-1)).abs() / o["abs"]).max().item() for o in out.values())
    print(f"{cyan}summed neuron writes against the MLP outputs' projection, largest error as a fraction of the summed |writes|: {check:.5f}{endc}")

#%% analyze: tables at the plots' readout layer: the gap, who writes it, how few neurons carry it
if STAGE == "analyze":
    rand = [f"random_{i}" for i in range(N_RANDOM)]
    def share(o: dict, r: int, p: int, k: int) -> float:   # the top-k neurons' signed share of the whole projection
        return (o["cum"][r, p, (o["grid"] >= k).nonzero()[0, 0]] / o["total"][r, p]).item()
    def n_half(o: dict, r: int, p: int) -> int:   # neurons, largest |write| first, to reach half of the summed |writes|
        return o["grid"][(o["cum_abs"][r, p] >= o["abs"][r, p] / 2).nonzero()[0, 0]].item()
    table, noise = {}, {}
    for env in ENVS:
        for p, pos in enumerate(POS):
            r, e = rows.index(f"{env}/gap"), ENVS.index(env)
            rand_d = t.tensor([abs(out[n, MAIN]["total"][r, p].item() / out[n, MAIN]["std"][e, p].item()) for n in rand])
            rand_half = t.tensor([n_half(out[n, MAIN], r, p) for n in rand]).float()
            noise[env, p] = rand_d.quantile(0.95).item()
            shuffles = [rows.index(f"{env}/shuffle{s}") for s in range(N_SHUFFLE)]
            print(f"\n{cyan}{env}, {pos}, resid_pre {MAIN}: cheating minus honest. gap in residual units, d' = gap / within-cell-and-class std (random directions' |d'|: median {rand_d.median():.2f}, 95th pct {rand_d.quantile(0.95):.2f}); shares are of the gap; n_half = neurons to half the summed |writes| (random directions: {rand_half.quantile(0.05):.0f}-{rand_half.quantile(0.95):.0f}; shuffled labels beside it); half r = writes of the top {TOP} neurons of one half of the cells against the other half; snr100 = 100th largest |write| over the same under shuffled labels{endc}")
            print(f"{'':12s}{'gap':>8s}{'d-prime':>8s}{'embed':>8s}{'mixers':>8s}{'MLPs':>8s}{'top10':>8s}{'top100':>8s}{'top1k':>8s}{'top10k':>8s}{'sum|w|/gap':>11s}{'n_half':>8s}{'half r':>8s}{'shuf n_half':>12s}{'snr100':>8s}")
            for name in NAMED:
                o = out[name, MAIN]
                g = o["total"][r, p].item()
                snr = (o["top_write"][r, p, 99].abs() / o["top_write"][shuffles, p, 99].abs().median()).item()   # the 100th largest |write| against the 100th largest under shuffled labels
                row = dict(gap=g, d_prime=g / o["std"][e, p].item(), embed=o["embed"][r, p].item() / g, mixers=o["mixer"][r, p].sum().item() / g, mlps=o["mlp"][r, p].sum().item() / g, **{f"top{k}": share(o, r, p, k) for k in [10, 100, 1000, 10000]},
                           cancel=o["abs"][r, p].item() / abs(g), n_half=n_half(o, r, p), half_r=o["half_r"][e, p].item(), shuffle_n_half=[n_half(o, rs, p) for rs in shuffles], snr100=snr)
                table[f"{env}/{pos}/{name}"] = row
                print(f"{name:12s}{row['gap']:8.2f}{row['d_prime']:8.2f}{row['embed']:8.2f}{row['mixers']:8.2f}{row['mlps']:8.2f}{row['top10']:8.2f}{row['top100']:8.2f}{row['top1000']:8.2f}{row['top10000']:8.2f}{row['cancel']:11.1f}{row['n_half']:8d}{row['half_r']:8.2f}{sorted(row['shuffle_n_half'])[N_SHUFFLE // 2]:12d}{row['snr100']:8.1f}")
    by_layer = {f"{env}/{pos}/{name}": {L: dict(gap=out[name, L]["total"][rows.index(f"{env}/gap"), p].item(), top100=share(out[name, L], rows.index(f"{env}/gap"), p, 100), n_half=n_half(out[name, L], rows.index(f"{env}/gap"), p)) for L in dirs[name]} for env in ENVS for p, pos in enumerate(POS) for name in NAMED}
    tops = {f"{env}/{pos}/{name}": [dict(layer=i // d_mlp, neuron=i % d_mlp, share=w / out[name, MAIN]["total"][rows.index(f"{env}/gap"), p].item()) for i, w in zip(out[name, MAIN]["top"][rows.index(f"{env}/gap"), p, :100].tolist(), out[name, MAIN]["top_write"][rows.index(f"{env}/gap"), p, :100].tolist())] for env in ENVS for p, pos in enumerate(POS) for name in NAMED}
    (ACTS / "neurons.json").write_text(json.dumps(dict(n_rollouts=len(rollouts), readout=MAIN, random_d_prime_95={f"{env}/{POS[p]}": v for (env, p), v in noise.items()}, table=table, by_layer=by_layer, top=tops), indent=1))

#%% analyze: plots, cheating minus honest at the plots' readout layer; a direction is drawn in a panel only where its |d'| beats the random directions' 95th percentile
if STAGE == "analyze":
    palette = dict(zip(NAMED, ["#3987e5", "#d95926", "#199e70", "#c98500", "#8e6bd6"]))
    panels = [(env, p) for env in ENVS for p in range(2)]
    shown = {(env, p): [name for name in NAMED if abs(table[f"{env}/{POS[p]}/{name}"]["d_prime"]) > noise[env, p]] for env, p in panels}
    titles = [f"{env} {POS[p]}" for env, p in panels]
    fig = make_subplots(rows=2, cols=4, subplot_titles=titles + [""] * 4, vertical_spacing=0.1, horizontal_spacing=0.04)
    for j, (env, p) in enumerate(panels):
        r = rows.index(f"{env}/gap")
        for name in shown[env, p]:
            o = out[name, MAIN]
            fig.add_trace(go.Scatter(x=o["grid"].tolist(), y=(o["cum"][r, p] / o["total"][r, p]).tolist(), line=dict(color=palette[name], width=2), name=name, showlegend=name not in [tr.name for tr in fig.data], legendgroup=name), row=1, col=j + 1)
            fig.add_trace(go.Scatter(x=list(range(1, TOP + 1)), y=(o["top_write"][r, p].abs() / abs(o["total"][r, p])).tolist(), line=dict(color=palette[name], width=2), showlegend=False, legendgroup=name), row=2, col=j + 1)
            for s in range(N_SHUFFLE):
                fig.add_trace(go.Scatter(x=list(range(1, TOP + 1)), y=(o["top_write"][rows.index(f"{env}/shuffle{s}"), p].abs() / abs(o["total"][r, p])).tolist(), line=dict(color=palette[name], width=1, dash="dot"), showlegend=False, legendgroup=name), row=2, col=j + 1)
        fig.update_xaxes(type="log", title_text="k neurons, largest |write| first", row=1, col=j + 1)
        fig.update_xaxes(type="log", title_text="rank k", row=2, col=j + 1)
        fig.update_yaxes(type="log", row=2, col=j + 1)
    fig.update_yaxes(title_text="signed share of the gap in the top k (ends at the MLPs' share)", row=1, col=1)
    fig.update_yaxes(title_text="|write| of the k-th neuron / |gap|; dotted: shuffled labels", row=2, col=1)
    fig.update_layout(**DARK, height=1150, width=2100, title=f"neurons: who writes the cheating-minus-honest gap along each direction at resid_pre {MAIN} ({MAIN * d_mlp:,} neurons below it, {len(rollouts)} rollouts)")
    write_dark_html(fig, str(ACTS / "neurons_share.html"))
    fig = make_subplots(rows=len(NAMED), cols=4, subplot_titles=[f"{name}, {env} {POS[p]}: d' {table[f'{env}/{POS[p]}/{name}']['d_prime']:.1f}" + ("" if name in shown[env, p] else " (within random)") for name in NAMED for env, p in panels], vertical_spacing=0.05, horizontal_spacing=0.04)
    for i, name in enumerate(NAMED):
        for j, (env, p) in enumerate(panels):
            o, r = out[name, MAIN], rows.index(f"{env}/gap")
            for part, color in [("mixer", "#3987e5"), ("mlp", "#d95926")]:
                fig.add_trace(go.Bar(x=list(range(MAIN)), y=(o[part][r, p] / o["total"][r, p]).tolist(), marker_color=color, name=dict(mixer="mixer (attention / linear attention)", mlp="MLP")[part], showlegend=i == 0 and j == 0, legendgroup=part), row=i + 1, col=j + 1)
            fig.update_xaxes(title_text="layer" if i == len(NAMED) - 1 else None, row=i + 1, col=j + 1)
    fig.update_layout(**DARK, barmode="relative", height=300 * len(NAMED), width=2100, title=f"neurons: each layer's write to the cheating-minus-honest gap at resid_pre {MAIN}, as a share of the gap")
    write_dark_html(fig, str(ACTS / "neurons_layers.html"))
    print(f"{green}wrote {ACTS}/neurons.json, neurons_share.html, neurons_layers.html{endc}")
