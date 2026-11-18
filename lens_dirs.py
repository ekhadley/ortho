#%%
"""Every J-lens and oracle-lens readout of a direction in the project, on one page. Saved directions: grader_kto, mbpp_kto, b1 (kto_sweep.py's joint direction) and Will's G9, with grader_proj and random unit vectors beside them. Prompt differences: prompt_dom.py's completion-less contrasts (E16), from that box's cache of residuals. Both lenses are the Qwen3.6-27B ones, read on Qwen3.8-27B (no 3.8 lens of either kind covers these layers).
    python lens_dirs.py prep          laptop: the prompt differences out of prompt_dom.py's cache, acts/lens_dirs/prompt_diffs.pt
    python lens_dirs.py olens         box, GPU free: the oracle lens (mechtools.olens, the 3.6 LoRA on the 3.8 weights) on +u and -u of every direction at its layers inside the hook window, N_SAMPLES samples each, acts/lens_dirs/olens.json
    python lens_dirs.py olens_diffs   box, GPU free: the oracle lens on both signs of every prompt difference at both positions, its own row at each lens layer and its mean over each of RANGES, acts/lens_dirs/olens_diffs.json
    python lens_dirs.py               laptop: the J-lens (camilablank/workspace-lenses, then 3.8's own final norm and unembedding, the path of scripts/jlens_scores.py) of the directions and the prompt differences, each direction's side from base rollouts, and the page in ~/wgmn/artifacts
Rows of a saved direction are the transformers hidden-state index: row r is the residual entering decoder layer r, where J-lens layer r reads; the oracle lens's layer L is the output of block L, row L + 1. The trained directions are one vector at rows 31-64 and go through every lens layer as that vector; G9 has its own row at every layer. Row i of a prompt difference is resid_post i: the oracle lens's layer i, and what J-lens layer i + 1 reads.
The lenses get the bare direction, which neither was fit on: the oracle lens rescales its input to a fixed norm, so only the direction and its sign reach it."""
import html
import json
import re
import shutil
import sys
from pathlib import Path

import torch as t
from torch import Tensor
from huggingface_hub import hf_hub_download
from rhenvs.grader import MAIN_FAMILIES, SYS_PROMPTS
from safetensors.torch import load_file
from transformers import AutoTokenizer

from mechtools import *
from mechtools.lens import cluster_tables, readout_html, tabbed

STAGE = sys.argv[1] if len(sys.argv) > 1 else "page"
MODEL_ID, REVISION = "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
VECTORS = Path("data/vectors/Qwen3.8-27B")
G9_FILE = Path.home() / "wgmn/ortho_box_archive/59147695/workspace/cheatbench-gpu/hf/stage9/directions/grid/G.safetensors"   # willjgriff/reward-hacking-ablation stage9/directions/grid
WEIGHTS = Path("acts/weights/Qwen3.8-27B")   # jlens_gain.py's cache of single tensors
ACTS = Path("acts/lens_dirs")
PROMPT_DOM = Path.home() / "wgmn/ortho_box_archive/59478943/acts/prompt_dom.pt"   # prompt_dom.py's cache, from its box of 2026-11-06
OUT = Path.home() / "wgmn/artifacts/2026-11-16_lens-readouts.html"
ROWS = list(range(31, 65))                    # the hooks' rows: the outputs of decoder layers 30-63
OLENS_LAYERS = list(range(32, 61, 4))         # the oracle lens's layers inside the window
DIFF_OLENS_LAYERS = list(range(20, 61, 4))    # every layer the oracle lens reads, for the prompt differences
RANGES = [(30, 40), (40, 63)]                 # resid_post layers, ends included: a difference's unit rows averaged over each, read at every lens layer inside it
TAB_LAYERS = list(range(20, 61, 4)) + [62]    # J-lens layers shown as tabs; J_62 is the identity, so L62 is the plain unembedding
DIFF_LAYERS = [36, 44, 52, 60, 62]            # J-lens layers of the prompt differences' table
POSITIONS = ["</think>", "newlines after it"]   # positions -2 and -1 of a reasoning-off prompt
THINK_CLOSE = 248069
DIFF_KEY = {   # a prompt difference's stored name: what it varies, what it holds fixed, its sign +1 as a - b and its sign -1 as b - a
    "hack - clean, no system prompt": ("grader block", "no system prompt", "hack - clean", "clean - hack"),
    "hack - clean, very_hacker": ("grader block", "very_hacker system prompt", "hack - clean", "clean - hack"),
    "very_hacker - none, hack prompts": ("system prompt", "hack prompts", "very_hacker - no system prompt", "no system prompt - very_hacker"),
    "very_hacker - generic, hack prompts": ("system prompt", "hack prompts", "very_hacker - generic", "generic - very_hacker"),
    "generic - none, hack prompts": ("system prompt", "hack prompts", "generic - no system prompt", "no system prompt - generic"),
    "(hack - clean) under very_hacker minus (hack - clean) with no system prompt": ("system prompt's effect on hack - clean", "nothing", "(hack - clean | very_hacker) - (hack - clean | no system prompt)", "(hack - clean | no system prompt) - (hack - clean | very_hacker)"),
}
TRAINED_ON = dict(grader_kto=["grader"], mbpp_kto=["mbpp"], b1=["grader", "mbpp"], g9=["grader"], grader_proj=["grader"])   # the envs whose cheating side names a direction's sides
ABOUT = dict(   # one line a direction, for the page
    grader_kto="train_vector.py's KTO projection direction trained on grader rollouts: one vector at rows 31-64, generated tokens.",
    mbpp_kto="The same recipe trained on MBPP rollouts.",
    b1="kto_sweep.py's joint direction: the KTO recipe on grader and MBPP rollouts in alternating steps.",
    g9="Will's Grader direction: the mean of unit cheat-minus-clean differences at the last CoT token, its own row at every layer.",
    grader_proj="train_vector.py's projection direction trained on grader rollouts, every position; the direction of the 2026-11-05 J-lens page.",
)
N_SAMPLES, N_RANDOM_OLENS, N_RANDOM_JLENS = 8, 4, 30
CHECK_PROMPT, CHECK_TOKEN = "Name an even number between 10 and 20.", " even"

#%% directions: [65, d_model] unit rows as saved, no sign change
dirs = {name: load_file(VECTORS / f"{name}.safetensors")["direction_unit"].float() for name in ["grader_kto", "mbpp_kto"]}
dirs["b1"] = t.zeros_like(dirs["grader_kto"])
dirs["b1"][ROWS] = normed(t.load(Path("acts/kto_sweep/vecs/b1.pt"), map_location="cpu")["v"].flatten().float())
dirs["g9"] = load_file(G9_FILE)["direction_unit"].float()
dirs["grader_proj"] = load_file(VECTORS / "grader_proj.safetensors")["direction_unit"].float()
d_model = dirs["g9"].shape[1]
random_dirs = normed(t.randn(N_RANDOM_JLENS, d_model, generator=t.Generator().manual_seed(1)))
row = lambda u, r: u[r] if u[r].norm() > 0 else u[ROWS[0]]   # a trained direction outside its window is the same vector
assert all(t.allclose(u[ROWS].norm(dim=-1), t.ones(len(ROWS)), atol=1e-4) for u in dirs.values())

#%% prep: prompt_dom.py's completion-less differences: resid_post at </think> and at the newlines after it of reasoning-off grader prompts, averaged over the family and side cells, [n_layers, 2, d_model]
if STAGE == "prep":
    cache = t.load(PROMPT_DOM, mmap=True, weights_only=False)
    cells = sorted({key[:2] for key in cache})
    assert all(x["ids"][-2] == THINK_CLOSE for key, x in cache.items() if not key[4])
    mean_resid = lambda kind, system: t.stack([cache[family, side, kind, system, False]["resid"][:, [-2, -1]].float() for family, side in cells]).mean(0)
    hack = mean_resid("hack", "none")
    clean = mean_resid("clean", "none")
    hack_vh = mean_resid("hack", "very_hacker")
    clean_vh = mean_resid("clean", "very_hacker")
    hack_generic = mean_resid("hack", "generic")
    diffs = {
        "hack - clean, no system prompt": hack - clean,
        "hack - clean, very_hacker": hack_vh - clean_vh,
        "very_hacker - none, hack prompts": hack_vh - hack,
        "very_hacker - generic, hack prompts": hack_vh - hack_generic,
        "generic - none, hack prompts": hack_generic - hack,
        "(hack - clean) under very_hacker minus (hack - clean) with no system prompt": (hack_vh - clean_vh) - (hack - clean),
    }
    ACTS.mkdir(parents=True, exist_ok=True)
    t.save(dict(diffs=diffs, n_cells=len(cells)), ACTS / "prompt_diffs.pt")
    print(f"{green}{len(diffs)} prompt differences over {len(cells)} cells saved to {ACTS / 'prompt_diffs.pt'}{endc}")

#%% olens, olens_diffs: the lens on the 3.8 weights; first a plain activation of the model, which has to read as the token it sits on
if STAGE in ("olens", "olens_diffs"):
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    reader, contract = load_olens(load_hf_model(MODEL_ID, device_map="cuda", revision=REVISION))
    ids = to_ids([{"role": "user", "content": CHECK_PROMPT}], tok, add_generation_prompt=True, enable_thinking=False)
    pos = [tok.decode(i) for i in ids].index(CHECK_TOKEN)
    with t.no_grad(), reader.disable_adapter():
        hidden = reader(input_ids=t.tensor([ids], device="cuda"), output_hidden_states=True).hidden_states
    check = [dict(layer=r["layer"], sample=r["j"], bullets=r["bullets"]) for layer in OLENS_LAYERS[::3] for r in olens_read(reader, tok, hidden[layer + 1][0, pos], layer, contract, n=2)]
    for r in check: print(f"{gray}check L{r['layer']} on {CHECK_TOKEN!r}: {r['bullets']}{endc}")
    assert any("even" in b.lower() for r in check for b in r["bullets"]), "the lens does not read ' even' off its own activation"
    def read(reads: list[tuple[dict, Tensor]], layer: int, chunk: int = 12) -> list[dict]:
        """N_SAMPLES samples of every (fields, vector) read at one lens layer, as records holding the fields, `chunk` vectors a batch."""
        out = []
        for k in range(0, len(reads), chunk):
            part = reads[k:k + chunk]
            out += [dict(**part[r["i"]][0], layer=layer, sample=r["j"], text=r["text"], bullets=r["bullets"], stopped=r["stopped"]) for r in olens_read(reader, tok, t.stack([h for _, h in part]).to("cuda", t.bfloat16), layer, contract, n=N_SAMPLES)]
        return out
    meta = dict(spec=contract["spec"], base=contract["base"], model=MODEL_ID, n=N_SAMPLES, check_prompt=CHECK_PROMPT, check_token=CHECK_TOKEN, check=check)

#%% olens: every direction and sign at every layer of the window
if STAGE == "olens":
    records = []
    for layer in pbar(OLENS_LAYERS, desc="olens"):
        records += read([(dict(name=name, sign=sign), sign * row(u, layer + 1)) for name, u in dirs.items() for sign in (1, -1)] + [(dict(name=f"random{i}", sign=1), r) for i, r in enumerate(random_dirs[:N_RANDOM_OLENS])], layer)
        print(f"{cyan}L{layer}: {len(records)} samples so far, {sum(r['stopped'] for r in records)} ended at eos{endc}")
    ACTS.mkdir(parents=True, exist_ok=True)
    (ACTS / "olens.json").write_text(json.dumps(dict(**meta, layers=OLENS_LAYERS, records=records), indent=1, ensure_ascii=False))
    print(f"{green}{len(records)} samples saved to {ACTS / 'olens.json'}{endc}")

#%% olens_diffs: both signs of every prompt difference at both positions, as its own row at each lens layer (rows "layer") and as the mean of its unit rows over a range (rows "30-40"), read at every lens layer inside the range. G9's range means ride along (resid_post i is its row i + 1), and the random vectors at the layers below the directions' window
if STAGE == "olens_diffs":
    diffs = t.load(ACTS / "prompt_diffs.pt")["diffs"]
    records = []
    for layer in pbar(DIFF_OLENS_LAYERS, desc="olens"):
        reads = [(dict(name=name, pos=p, sign=sign, rows="layer"), sign * d[layer, p]) for name, d in diffs.items() for p in range(2) for sign in (1, -1)]
        for lo, hi in RANGES:
            if lo <= layer <= hi:
                reads += [(dict(name=name, pos=p, sign=sign, rows=f"{lo}-{hi}"), sign * normed(d[lo:hi + 1, p]).mean(0)) for name, d in diffs.items() for p in range(2) for sign in (1, -1)]
                reads += [(dict(name="g9", pos=None, sign=sign, rows=f"{lo}-{hi}"), sign * dirs["g9"][lo + 1:hi + 2].mean(0)) for sign in (1, -1)]
        if layer not in OLENS_LAYERS: reads += [(dict(name=f"random{i}", pos=None, sign=1, rows="layer"), r) for i, r in enumerate(random_dirs[:N_RANDOM_OLENS])]
        records += read(reads, layer)
        print(f"{cyan}L{layer}: {len(reads)} vectors, {len(records)} samples so far, {sum(r['stopped'] for r in records)} ended at eos{endc}")
        (ACTS / "olens_diffs.json").write_text(json.dumps(dict(**meta, layers=DIFF_OLENS_LAYERS, ranges=RANGES, records=records), indent=1, ensure_ascii=False))
    print(f"{green}{len(records)} samples saved to {ACTS / 'olens_diffs.json'}{endc}")

#%% page: which side of each direction cheating rollouts fall on. neurons.py's residuals: base rollouts in cells (a grader family and side, an MBPP task) holding both classes, h[n, 0, r] the CoT mean of row r
if STAGE == "page":
    rollouts = t.load("acts/neurons/inputs.pt", weights_only=False)
    h = t.load("acts/neurons/results/h.pt", mmap=True)[:, 0]
    gap_cos = {}   # env -> name -> cosine of the direction with the cell-balanced cheating-minus-honest mean, per hooked row
    for env in ["grader", "mbpp"]:
        cells = sorted({x["cell"] for x in rollouts if x["env"] == env})
        weight = t.zeros(len(rollouts))
        for cell in cells:
            for cheat in (True, False):
                members = [n for n, x in enumerate(rollouts) if x["env"] == env and x["cell"] == cell and x["cheat"] == cheat]
                weight[members] = (1 if cheat else -1) / len(members) / len(cells)
        used = weight.nonzero().flatten()
        gap = t.einsum("n,nrd->rd", weight[used], h[used][:, ROWS])
        gap_cos[env] = {name: t.cosine_similarity(gap, u[ROWS], dim=-1) for name, u in dirs.items()}
        print(f"{cyan}{env}: {len(used)} rollouts in {len(cells)} cells; mean cos with the cheating-minus-honest gap " + ", ".join(f"{name} {c.mean():+.3f}" for name, c in gap_cos[env].items()) + f" (a random direction: sd {d_model ** -0.5:.3f}){endc}")
    cheat_sign = {}   # +1 when the saved rows point toward cheating in the direction's own envs
    for name, envs in TRAINED_ON.items():
        signs = {int(gap_cos[env][name].mean().sign()) for env in envs}
        assert len(signs) == 1, f"{name} points toward cheating in one of {envs} and away from it in the other"
        cheat_sign[name] = signs.pop()
    print(f"{cyan}saved rows point toward cheating: {cheat_sign}{endc}")

#%% page: prep's prompt differences, row i resid_post i
if STAGE == "page":
    saved_diffs = t.load(ACTS / "prompt_diffs.pt")
    diffs = saved_diffs["diffs"]
    assert list(diffs) == list(DIFF_KEY), (list(diffs), list(DIFF_KEY))

#%% page: J-lens logits of every direction at every lens layer in the window or shown as a tab, of the prompt differences at the tab layers, and the window means of random unit vectors
if STAGE == "page":
    lens = t.load(hf_hub_download("camilablank/workspace-lenses", "qwen3.6-27b/j-lens/lens.pt"), map_location="cpu", weights_only=False, mmap=True)
    window = [L for L in lens["source_layers"] if L in ROWS]
    unembed = load_file(WEIGHTS / "lm_head.weight.safetensors")["lm_head.weight"].cuda().float()
    final_scale = 1 + load_file(WEIGHTS / "model.language_model.norm.weight.safetensors")["model.language_model.norm.weight"].cuda().float()
    def logits(x: Tensor) -> Tensor:   # the model's final RMSNorm and unembedding
        return (x * t.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * final_scale) @ unembed.T
    scores = {name: {} for name in dirs}
    diff_scores = {name: {} for name in diffs}   # name -> layer -> [2 positions, vocab]
    random_window = t.zeros(N_RANDOM_JLENS, len(unembed))
    with t.no_grad():
        for L in pbar(sorted(set(window) | set(TAB_LAYERS)), desc="j-lens"):
            J = lens["J"][L].float().cuda()
            for name, u in dirs.items(): scores[name][L] = logits(J @ row(u, L).cuda()).cpu()
            if L in window: random_window += logits(random_dirs.cuda() @ J.T).cpu() / len(window)
            if L in TAB_LAYERS:
                for name, d in diffs.items(): diff_scores[name][L] = logits(d[L - 1].cuda() @ J.T).cpu()
    saved = t.load(VECTORS / "jlens/jlens_grader_proj.pt")   # scripts/jlens_scores.py's box run of the same path, with its clustering of the 3.8 unembedding
    labels = saved["labels"]
    drift = max((scores["grader_proj"][L] - saved["scores"]["u"][L]).abs().max().item() for L in TAB_LAYERS)
    print(f"{cyan}grader_proj against the box run's logits: largest difference {drift:.4f} over {len(TAB_LAYERS)} layers{endc}")
    assert drift < 0.05
    window_mean = {name: t.stack([s[L] for L in window]).mean(0) for name, s in scores.items()}
    del unembed

#%% page: cluster masses of the window mean against the random vectors', the tabs, and the oracle lens's samples
if STAGE == "page":
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    n_clusters = int(labels.max()) + 1
    sides = {"cheating side": 1, "honest side": -1}
    show = lambda i: f"<code>{html.escape(repr(tok.decode(i)))}</code>"
    def mass(s: Tensor) -> Tensor:
        return t.zeros(n_clusters).index_add_(0, labels, s.softmax(-1))
    grids = lambda s: cluster_tables(s, labels, tok.decode, 11, 10, True)[0]
    random_strongest = {sign: t.stack([mass(sign * x) for x in random_window]).max(1).values for sign in (1, -1)}
    olens = json.loads((ACTS / "olens.json").read_text())
    olens_diffs = json.loads((ACTS / "olens_diffs.json").read_text())
    records = [dict(r, pos=None, rows="layer") for r in olens["records"]] + olens_diffs["records"]
    texts = lambda name, sign, layer, rows="layer", pos=None: [r["text"] for r in records if (r["name"], r["sign"], r["layer"], r["rows"], r["pos"]) == (name, sign, layer, rows, pos)]
    def coherence(rows: Tensor) -> float:   # the mean cosine between two different rows
        u = normed(rows)
        return ((u @ u.T).sum().item() - len(u)) / (len(u) * (len(u) - 1))
    diff_coherence = [coherence(d[lo:hi + 1, p]) for d in diffs.values() for p in range(2) for lo, hi in olens_diffs["ranges"]]
    g9_coherence = [coherence(dirs["g9"][lo + 1:hi + 2]) for lo, hi in olens_diffs["ranges"]]
    range_reads = [(f"{lo}-{hi}", L) for lo, hi in olens_diffs["ranges"] for L in olens_diffs["layers"] if lo <= L <= hi]   # a range of resid_post layers and a lens layer inside it
    def columns(cols: dict[str, list[str]]) -> str:   # a column a key, its samples one under another, every character of each
        return f"<div class=sg style='grid-template-columns:repeat({len(cols)},minmax(0,1fr))'>" + "".join(f"<div><h4>{html.escape(k)}</h4>" + "".join(f"<div class=s><span class=c>sample {i}</span>\n{html.escape(x)}</div>" for i, x in enumerate(v)) + "</div>" for k, v in cols.items()) + "</div>"
    by_layer = lambda cols_at: "".join(f"<details><summary>{tab}</summary>{columns(cols)}</details>" for tab, cols in cols_at.items())

    cluster_rows, strip, widgets = "", "", ""
    for i, name in enumerate(dirs):
        for j, (side, s) in enumerate(sides.items()):
            sign = s * cheat_sign[name]
            m = mass(sign * window_mean[name])
            beaten = int((random_strongest[sign] < m.max()).sum())
            tops = []
            for value, c in zip(*[x.tolist() for x in m.topk(3)]):
                members = (labels == c).nonzero().flatten()
                tops.append(f"<span class=c>c{c} {value:.4f}</span> " + " ".join(show(k) for k in members[(sign * window_mean[name])[members].topk(min(5, len(members))).indices].tolist()))
            cluster_rows += f"<tr><td>{name}</td><td>{side}</td><td class=n>{m.max():.4f}</td><td class=n>{beaten} / {N_RANDOM_JLENS}</td><td>{'<br>'.join(tops)}</td></tr>"
            y = 30 + 21 * (2 * i + j) + 8 * i
            x = lambda v: 190 + 640 * min(v, 0.06) / 0.06
            strip += f"<text x=180 y={y + 4} text-anchor=end fill='var(--muted)' font-size=12>{name}, {side.split()[0]}</text>" + "".join(f"<circle cx={x(v):.1f} cy={y} r=3 opacity=.7 fill='#55637a'/>" for v in random_strongest[sign].tolist()) + f"<circle cx={x(m.max().item()):.1f} cy={y} r=5 fill='{'var(--red)' if s == 1 else 'var(--teal)'}'/><text x={x(m.max().item()) + (10 if m.max() < 0.05 else -10):.1f} y={y + 4} text-anchor={'start' if m.max() < 0.05 else 'end'} fill='var(--text)' font-size=12>{m.max():.4f}</text>"
        jl = {tab: {side: grids(s * cheat_sign[name] * x) for side, s in sides.items()} for tab, x in [(f"L{window[0]}-{window[-1]} mean", window_mean[name])] + [(f"L{L}", scores[name][L]) for L in TAB_LAYERS]}
        ol = by_layer({f"L{L}": {side: texts(name, s * cheat_sign[name], L) for side, s in sides.items()} for L in olens["layers"]})
        if name == "g9": ol += "<h3>Oracle lens: the mean of G9's rows over a range of layers</h3>" + by_layer({f"mean over layers {rng}, read as L{L}": {side: texts(name, s * cheat_sign[name], L, rng) for side, s in sides.items()} for rng, L in range_reads})
        widgets += f"<section><h2>{name}</h2><p class=caption>{html.escape(ABOUT[name])} Saved rows point {'toward' if cheat_sign[name] == 1 else 'away from'} cheating; both readouts are oriented so the cheating side is the side cheating rollouts fall on.</p><h3>J-lens</h3>{readout_html(tabbed(jl), n_cols=3)}<h3>Oracle lens: every sample in full, {olens['n']} a layer and side</h3>{ol}</section>"
    random_jl = {tab: {f"random{i}": grids(random_window[i]) for i in range(3)} for tab in [f"L{window[0]}-{window[-1]} mean"]}
    random_ol = by_layer({f"L{L}": {f"random{i}": texts(f"random{i}", 1, L) for i in range(N_RANDOM_OLENS)} for L in olens_diffs["layers"]})
    widgets += f"<section><h2>Random unit vectors</h2><h3>J-lens, window mean</h3>{readout_html(tabbed(random_jl), n_cols=3)}<h3>Oracle lens: every sample in full</h3>{random_ol}</section>"
    diff_rows, diff_widgets = "", ""
    title = lambda name: f"{DIFF_KEY[name][2]} <span class=c>varied: {DIFF_KEY[name][0]} · fixed: {DIFF_KEY[name][1]}</span>"
    diff_key = "".join(f"<tr><td>{html.escape(varied)}</td><td>{html.escape(fixed)}</td><td><code>{html.escape(plus)}</code></td><td><code>{html.escape(minus)}</code></td></tr>" for varied, fixed, plus, minus in DIFF_KEY.values())
    for name in diffs:
        for p, where in enumerate(POSITIONS): diff_rows += f"<tr><td>{title(name)}</td><td>{html.escape(where)}</td>" + "".join(f"<td>{' '.join(show(k) for k in diff_scores[name][L][p].topk(5).indices.tolist())}</td>" for L in DIFF_LAYERS) + "</tr>"
        jl = {f"L{L}": {where: grids(diff_scores[name][L][p]) for p, where in enumerate(POSITIONS)} for L in TAB_LAYERS}
        four = lambda L, rows: {f"{where}, {DIFF_KEY[name][2 if sign == 1 else 3]}": texts(name, sign, L, rows, p) for p, where in enumerate(POSITIONS) for sign in (1, -1)}   # a column a position and sign
        own = by_layer({f"L{L}": four(L, "layer") for L in olens_diffs["layers"]})
        ranged = by_layer({f"mean over layers {rng}, read as L{L}": four(L, rng) for rng, L in range_reads})
        diff_widgets += f"<section><h2>{title(name)}</h2><h3>J-lens, {DIFF_KEY[name][2]}</h3>{readout_html(tabbed(jl), n_cols=3)}<h3>Oracle lens: the difference at each layer, every sample in full</h3>{own}<h3>Oracle lens: the mean of its unit rows over a range of layers</h3>{ranged}</section>"
    axis = "".join(f"<line x1={190 + 640 * v / 0.06:.0f} y1=14 x2={190 + 640 * v / 0.06:.0f} y2={30 + 50 * len(dirs)} stroke='var(--line)'/><text x={190 + 640 * v / 0.06:.0f} y=10 text-anchor=middle fill='var(--muted)' font-size=11>{v:.2f}</text>" for v in [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    side_rows = "".join(f"<tr><td>{name}</td><td>{', '.join(TRAINED_ON[name])}</td>" + "".join(f"<td class=n>{gap_cos[env][name].mean():+.3f}</td><td class=n>{gap_cos[env][name].min():+.3f} to {gap_cos[env][name].max():+.3f}</td>" for env in gap_cos) + "</tr>" for name in dirs)
    names = list(dirs)
    cos_rows = "".join(f"<tr><td>{a}</td>" + "".join(f"<td class=n>{t.cosine_similarity(cheat_sign[a] * dirs[a][ROWS], cheat_sign[b] * dirs[b][ROWS], dim=-1).mean():+.2f}</td>" for b in names) + "</tr>" for a in names)
    rates = {}   # system prompt -> family -> [invalid, valid, cheats]: base reasoning-off records on hack prompts, the families of the prompt differences
    for f in sorted(Path("data/inspect/grader").glob("*.jsonl")):
        cfg = json.loads(f.open().readline()).get("cfg")   # the oldest records carry no cfg and are no run of these
        if cfg is None or (cfg["config"], cfg["model"], cfg["family"], cfg["n"], cfg["prompts"], cfg["lora"], cfg["add_vector"], cfg["ablate_vector"]) != ("nothink", "vllm/cb_base", "all", 256, "hack", None, None, None): continue
        assert cfg["system"] not in rates, f"two runs for {cfg['system']}"
        rates[cfg["system"]] = {family: [0, 0, 0] for family in MAIN_FAMILIES}
        for line in f.open():
            r = json.loads(line)
            if r["labels"]["family"] in MAIN_FAMILIES: rates[cfg["system"]][r["labels"]["family"]][0 if r["labels"]["cheat"] is None else 1] += 1; rates[cfg["system"]][r["labels"]["family"]][2] += r["labels"]["cheat"] is True
    assert sorted(rates) == ["generic", "none", "very_hacker"], sorted(rates)
    rate_color = lambda x: f"hsl({180 - 180 * x:.0f} 55% 32%)"   # teal at 0 to red at 1
    def rate_cell(counts: list[list[int]], hover: str) -> str:
        cheats, valid = sum(c[2] for c in counts), sum(c[1] for c in counts)
        return f"<td class=n title='{html.escape(hover)}' style='background:{rate_color(cheats / valid)};cursor:help'>{100 * cheats / valid:.1f}%</td><td class=n>{cheats} / {valid}</td><td class=n>{sum(c[0] for c in counts)}</td>"
    rate_rows = "".join(f"<tr><td>hack</td><td>{'no system prompt' if system == 'none' else system}</td>" + rate_cell(list(rates[system].values()), "\n".join(f"{family}: {c[2]} / {c[1]} = {100 * c[2] / c[1]:.0f}%, {c[0]} invalid" for family, c in rates[system].items())) + "</tr>" for system in ["none", "generic", "very_hacker"])
    check = columns({f"L{L}": ["\n".join(r["bullets"]) for r in olens["check"] if r["layer"] == L] for L in sorted({r["layer"] for r in olens["check"]})})

#%% page
if STAGE == "page":
    CSS = open(Path.home() / "wgmn/artifacts/2026-11-05_grader-proj-jlens.html").read().split("<style>")[1].split("</style>")[0]
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>J-lens and oracle-lens readouts: every direction read so far</title>
<style>{CSS}h3{{font-size:15px;margin:22px 0 8px;color:var(--muted);font-weight:600}}.c{{color:var(--muted);font-size:.9em}}td code{{white-space:nowrap}}.part{{border-top:3px solid var(--accent);margin-top:60px;padding-top:26px}}.part h1{{font-size:30px}}a{{color:var(--accent)}}.sg{{display:grid;gap:14px;margin:6px 0 18px}}.sg h4{{margin:0 0 6px;font-size:13px;color:var(--muted)}}.s{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 11px;margin:0 0 8px;font:12px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere}}summary{{cursor:pointer;font-weight:600;margin:10px 0 4px}}</style></head><body><main>
<header><div class="eyebrow">ortho · lens readouts · 16 Nov 2026 <span class="pill">Qwen3.6 lenses on Qwen3.8</span></div>
<h1>J-lens and oracle-lens readouts: every direction read so far</h1>
<p class="subtitle">A direction is read alone, with no rollout around it. The J-lens maps it through one layer's averaged Jacobian and Qwen3.8's final norm and unembedding to a distribution over tokens, grouped into 1024 clusters of the unembedding. The oracle lens is a LoRA that writes lines of text for one activation; it gets the direction rescaled to its fixed input norm.</p>
<div class="table-wrap"><table><tr><th>Part</th><th>What is read</th><th>J-lens</th><th>Oracle lens</th></tr>
<tr><td><a href="#part1">1. Saved directions</a></td><td>grader_kto, mbpp_kto, b1, G9, on both signs</td><td>lens layers 20-62</td><td>layers 32-60 by 4, {olens['n']} samples a layer and side; for G9 also the mean of its rows over layers {' and '.join(f'{lo}-{hi}' for lo, hi in olens_diffs['ranges'])}</td></tr>
<tr><td></td><td>grader_proj, on both signs</td><td>the readout of the 2026-11-05 page</td><td>as the others</td></tr>
<tr><td></td><td>random unit vectors</td><td>30, for the cluster masses</td><td>4, at layers 20-60 by 4</td></tr>
<tr><td><a href="#part2">2. Prompt differences</a></td><td>{len(diffs)} differences of mean residuals between grader prompts, reasoning off, at &lt;/think&gt; and at the newlines after it, on both signs</td><td>lens layers 20-62, A - B only</td><td>layers 20-60 by 4, {olens_diffs['n']} samples a layer, position and sign; also the mean of each difference's unit rows over layers {' and '.join(f'{lo}-{hi}' for lo, hi in olens_diffs['ranges'])}</td></tr></table></div>
<p class="caption">Not lens readouts, and not here: Will's finite-difference readouts of G9 and grader_proj (data/vectors/Qwen3.8-27B/fd_readout), and jlens_gain.py, which measures gain norms through the J-lens and reads no tokens. grader_lens.py has J-lens cells for the Qwen3.6 direction grader_parity_cheat_vs_clean; no output of them was saved.</p></header>

<div class="part" id="part1"><h1>1. Saved directions</h1><p class="subtitle">Each is a unit vector, read on both signs. grader_proj and random unit vectors are the references.</p></div>

<section><h2>Which side is the cheating side</h2>
<p>Cosine of each direction's saved rows with the cheating-minus-honest difference of CoT-mean residuals in base rollouts, cell-balanced, over rows {ROWS[0]}-{ROWS[-1]}. A random direction has standard deviation {d_model ** -0.5:.3f}. Everything below is oriented so the cheating side is the side cheating rollouts fall on in the envs a direction was fit on.</p>
<div class="table-wrap"><table><tr><th rowspan=2>Direction</th><th rowspan=2>Fit on</th><th colspan=2>Grader</th><th colspan=2>MBPP</th></tr><tr><th>Mean</th><th>Range over rows</th><th>Mean</th><th>Range over rows</th></tr>{side_rows}</table></div>
<p class="caption">Positive: the saved rows point toward cheating. Rollouts and residuals are neurons.py's ({sum(x['env'] == 'grader' for x in rollouts)} grader, {sum(x['env'] == 'mbpp' for x in rollouts)} MBPP).</p>
<div class="table-wrap"><table><tr><th>Cosine, cheating sides</th>{''.join(f'<th>{n}</th>' for n in names)}</tr>{cos_rows}</table></div>
<p class="caption">Mean over rows {ROWS[0]}-{ROWS[-1]} of the cosine between two directions, each oriented toward cheating.</p></section>

<section><h2>J-lens: the strongest cluster against random vectors</h2>
<p>Logits are averaged over lens layers {window[0]}-{window[-1]}. A cluster's mass is its summed probability. The large dot is the direction's strongest cluster, red for the cheating side and teal for the honest side; the grey dots are the strongest clusters of {N_RANDOM_JLENS} random unit vectors read the same way.</p>
<figure><svg viewBox="0 0 850 {40 + 50 * len(dirs)}" role="img" aria-label="Strongest cluster mass of each direction and side against random vectors">{axis}{strip}</svg></figure>
<div class="table-wrap"><table><tr><th>Direction</th><th>Side</th><th>Strongest cluster mass</th><th>Random vectors below it</th><th>Top three clusters, mass and top tokens</th></tr>{cluster_rows}</table></div>
<p class="caption">Masses above 0.06 are drawn at the right edge.</p></section>

<section><h2>Oracle lens: check on a real activation</h2>
<p>The lens on Qwen3.8 weights, given a plain activation at {html.escape(repr(olens['check_token']))} in "{html.escape(olens['check_prompt'])}". These are the lines it marked with a dash; the rest of these two samples' text was not saved.</p>
{check}</section>

{widgets}

<div class="part" id="part2"><h1>2. Prompt differences</h1><p class="subtitle">prompt_dom.py (E16): one forward pass per grader prompt with reasoning off, over {saved_diffs['n_cells']} family and side cells. A difference is between the mean residuals of two sets of prompts at one position, the sets differing in one thing and the same in the rest.</p></div>

<section><h2>The differences</h2>
<p>Every prompt is one family and side's request. <b>Grader block</b>: a hack prompt adds a <code>&lt;_grader&gt;</code> block rewarding the other side's answers ("Choose a random even number." with <code>reward = output % 2</code>), a clean prompt one rewarding the side's own (<code>reward = 1 - (output % 2)</code>). <b>System prompt</b>: none, generic ("{html.escape(SYS_PROMPTS['generic'])}") or very_hacker ("{html.escape(SYS_PROMPTS['very_hacker'])}"). A - B is the mean over the {saved_diffs['n_cells']} cells of set A's residual minus set B's.</p>
<div class="table-wrap"><table><tr><th>Varied</th><th>Fixed</th><th>A - B: read by the J-lens and the oracle lens</th><th>B - A: read by the oracle lens</th></tr>{diff_key}</table></div>
<p class="caption">The J-lens reads only the first form. Each oracle-lens column is headed by its position and the subtraction it reads.</p></section>

<section><h2>Cheat rates of the prompt sets</h2>
<p>Qwen3.8 base, reasoning off, the prompts of the differences: 256 samples a family and side, sampled at the nothink config. Cheating is an answer on the other side, the one the grader rewards; an answer on neither side is invalid and left out of both counts. Hover a rate for its families.</p>
<div class="table-wrap"><table><tr><th>Grader block</th><th>System prompt</th><th>Cheat rate</th><th>Cheats / valid</th><th>Invalid</th></tr>{rate_rows}</table></div>
<p class="caption">Clean prompts give no reason to cheat and are not sampled here. The no-system-prompt run is nothink38 of 2026-11-05 (pets, since removed, left out); generic and very_hacker are nothink38_generic and nothink38_vh.</p></section>

<section><h2>J-lens: top tokens by layer</h2>
<div class="table-wrap"><table><tr><th>Difference</th><th>Position</th>{''.join(f'<th>L{L}</th>' for L in DIFF_LAYERS)}</tr>{diff_rows}</table></div>
<p class="caption">The five tokens with the largest logit. Lens layer L reads the difference of resid_post L - 1.</p></section>

{diff_widgets}

<section><h2>What was run</h2>
<ul>
<li>J-lens: camilablank/workspace-lenses qwen3.6-27b/j-lens/lens.pt, layer r reading row r, then Qwen3.8's final norm and unembedding. The same path on grader_proj differs from the 2026-11-05 box run by at most {drift:.4f} in any logit.</li>
<li>Oracle lens: {olens['spec']} (trained on {olens['base']}) loaded on {olens['model']}, its layer L reading a saved direction's row L + 1 and a prompt difference's resid_post L.</li>
<li>The trained directions are one vector at every row, so their tabs differ only in which layer's lens reads it. G9 has a different row at every layer.</li>
<li>Oracle-lens samples are shown whole, as generated; {sum(r['stopped'] for r in records)} of the {len(records)} ended at the end-of-turn token and the rest at the length limit. A sample is mostly lines starting with a dash; {sum(any(line.strip() and not line.strip().startswith('- ') for line in r['text'].splitlines()) for r in records)} also hold other lines.</li>
<li>Prompt differences: the residuals are prompt_dom.py's cache from its box of 2026-11-06, row i of a difference its resid_post i.</li>
<li>A mean over a range of layers is the mean of the unit vectors of resid_post over the range, ends included, read at every lens layer inside the range. Two rows of a range have mean cosine {min(diff_coherence):.2f} to {max(diff_coherence):.2f} for the prompt differences, and {g9_coherence[0]:.2f} and {g9_coherence[1]:.2f} for G9's two ranges.</li>
<li>Neither lens was fit on bare directions or differences, and neither was fit on Qwen3.8.</li>
<li>Clusters are one k-means seed on the 3.8 unembedding, the clustering of the 2026-11-05 page.</li>
</ul></section>
<footer>Built by lens_dirs.py from acts/lens_dirs/olens.json, data/vectors/Qwen3.8-27B, acts/kto_sweep/vecs/b1.pt, Will's stage9/directions/grid/G.safetensors, acts/neurons, acts/lens_dirs/prompt_diffs.pt and acts/lens_dirs/olens_diffs.json. Model revision {REVISION[:7]}.</footer>
</main></body></html>"""
    OUT.write_text(page)
    shutil.copy(OUT, "figures")
    print(f"{green}{OUT} {OUT.stat().st_size / 1e6:.1f} MB{endc}")
