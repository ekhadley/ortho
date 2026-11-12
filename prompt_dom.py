#%%
"""Grader prompts' residual streams with no completion. Every main family, side and prompt kind (hack, clean, none), with no system prompt and under generic and very_hacker, with reasoning off and on,
one forward pass each: nothing is sampled, so a prompt is one activation. acts/prompt_dom.pt holds each one's ids and resid_post at every layer and position, and every diff is arithmetic on it.
With reasoning off a prompt ends <think>\n\n</think>\n\n, so position -2 is </think> and -1 the newlines right before the answer. With reasoning on it ends before the CoT, at reasoning effort low, the setting of every Qwen3.8 reasoning-on run (the template adds it to the system turn; reasoning off is unchanged by it).
Diffs are read through the J-lens and the oracle lens, and very_hacker minus generic is used as a probe on reasoning-on rollouts. Kernel cell script, GPU box."""
import itertools
import json
from pathlib import Path

import torch as t
from rhenvs.grader import MAIN_FAMILIES, SYS_PROMPTS, prompt
from safetensors.torch import load_file
from scipy.stats import rankdata

from mechtools import *

from utils import cot_span, resid_post, resid_pre_layout

t.set_grad_enabled(False)
MODEL_ID = "Qwen/Qwen3.8-27B"
LENS = "qwen3.6-27b"   # no Qwen3.8 lens exists
CELLS = [(family, side) for family, f in MAIN_FAMILIES.items() for side in f["sides"]]
CACHE = Path("acts/prompt_dom.pt")
PROBE_CACHE = Path("acts/prompt_dom_probe.pt")
OLENS_LAYERS = [20, 24, 28, 32, 36, 40]

#%% model, J-lens and vocab clusters
model = load_bridge(MODEL_ID, device_map="cuda")
jlens = load_jlens(f"{LENS}/j-lens/lens.pt", device=model.device)
labels, _ = cluster_vocab(model, k=1024)
print(f"{gray}j-lens {LENS}: {len(jlens['J'])} x {tuple(jlens['J'][0].shape)}, model {model.cfg.n_layers} layers{endc}")

#%% one forward pass per prompt, saved: cache[family, side, kind, system, think] is ids and bf16 resid_post [n_layers, seq, d_model], the model's own precision

make_cache = False
if make_cache:
    cache = {}
    for (family, side), kind, system, think in pbar(list(itertools.product(CELLS, ["hack", "clean", "none"], ["none", "generic", "very_hacker"], [False, True]))):
        system_turn = [dict(role="system", content=SYS_PROMPTS[system])] if SYS_PROMPTS[system] else []
        ids = to_ids(system_turn + [dict(role="user", content=prompt(MAIN_FAMILIES[family], side, kind))], model.tokenizer, add_generation_prompt=True, enable_thinking=think, reasoning_effort="low")
        cache[family, side, kind, system, think] = dict(ids=ids, resid=resid_post(model, ids).to("cpu", t.bfloat16))
    CACHE.parent.mkdir(exist_ok=True)
    t.save(cache, CACHE)
    print(f"{green}{len(cache)} prompts, {CACHE.stat().st_size / 1e9:.1f} GB in {CACHE}{endc}")

#%% the saved cache, for a fresh kernel

load_cache = True
if load_cache:
    cache = t.load(CACHE)

#%% what the cells below share: means over the family and side cells, the lens readouts of a diff at each position, and the within-cell AUROC

def mean_resid(kind: str, system: str, pos: int, think: bool = False) -> Tensor:
    """resid_post at one position averaged over the cells, [n_layers, d_model]. pos counts from the end, which lines up prompts of different lengths."""
    return t.stack([cache[family, side, kind, system, think]["resid"][:, pos].float() for family, side in CELLS]).mean(0)

def diff_readout(diff, title: str, positions: tuple[int, ...] = (-2, -1)) -> dict:
    """J-lens cluster readout, a tab per layer and per position. diff(pos) is a [n_layers, d_model] resid_post difference, read at resid_pre of the next layer, where the lens's rows sit: layers 1-62, the lens having no row for 63."""
    vs = {p: resid_pre_layout(diff(p)).to(model.device, model.W_U.dtype) for p in positions}
    scores = {f"L{layer}": {p: get_lens_logits(v[layer], layer, model, jlens) for p, v in vs.items()} for layer in range(1, len(jlens["J"]))}
    return cluster_readout(scores, labels, model.tokenizer.decode, title=title)

def olens_diff(diff, title: str, n: int = 2, positions: tuple[int, ...] = (-2, -1)) -> list[dict]:
    """Oracle-lens readout of diff(pos), a [n_layers, d_model] resid_post difference, through a cache-shaped dict holding one row per position. The lens was trained on activations, not differences; it normalizes its input."""
    vs = [diff(p) for p in positions]
    rows = {f"blocks.{layer}.hook_resid_post": t.stack([v[layer] for v in vs])[None].to("cuda", t.bfloat16) for layer in OLENS_LAYERS}
    return olens_readout(rows, OLENS_LAYERS, list(positions), reader, model.tokenizer, contract, n=n, title=title)

def cell_auroc(x: Tensor, cheat: Tensor, cell: Tensor) -> Tensor:
    """AUROC of cheat against clean along dim 0 of x, within each cell holding both classes and weighted by its cheat x clean pairs: the chance a cheating row outscores a clean row of the same cell, whose rollouts share one prompt, so telling cells apart cannot win it."""
    wins = 0
    pairs = 0
    for c in cell.unique():
        y = cheat[cell == c]
        n_cheat = y.sum().item()
        n_clean = (~y).sum().item()
        if n_cheat and n_clean:
            ranks = t.tensor(rankdata(x[cell == c].numpy(), axis=0))
            wins = wins + ranks[y].sum(0) - n_cheat * (n_cheat + 1) / 2
            pairs += n_cheat * n_clean
    return wins / pairs

#%% check: hack minus clean with no system prompt at the last token is nocot_dom_last's quantity, computed there through plain HF

check_dom = False
if check_dom:
    saved = load_file("data/vectors/Qwen3.8-27B/nocot_dom_last.safetensors")["direction_unit"]
    window = list(range(30, 64))
    cos = t.nn.functional.cosine_similarity((mean_resid("hack", "none", -1) - mean_resid("clean", "none", -1))[window], saved[[layer + 1 for layer in window]].float())
    print(f"{cyan}cos with nocot_dom_last over layers 30-63: min {cos.min():.4f}, mean {cos.mean():.4f}{endc}")

#%% J-lens readouts of the contrasts at </think> and the newlines after it

show_jlens = False
if show_jlens:
    _ = diff_readout(lambda p: mean_resid("hack", "none", p) - mean_resid("clean", "none", p), "hack - clean, no system prompt")
    _ = diff_readout(lambda p: mean_resid("hack", "very_hacker", p) - mean_resid("clean", "very_hacker", p), "hack - clean, very_hacker")
    _ = diff_readout(lambda p: mean_resid("hack", "very_hacker", p) - mean_resid("hack", "none", p), "very_hacker - no system prompt, hack prompts")
    _ = diff_readout(lambda p: mean_resid("hack", "very_hacker", p) - mean_resid("hack", "generic", p), "very_hacker - generic, hack prompts")
    _ = diff_readout(lambda p: mean_resid("hack", "generic", p) - mean_resid("hack", "none", p), "generic - no system prompt, hack prompts")
    _ = diff_readout(lambda p: (mean_resid("hack", "very_hacker", p) - mean_resid("clean", "very_hacker", p)) - (mean_resid("hack", "none", p) - mean_resid("clean", "none", p)), "(hack - clean) under very_hacker minus (hack - clean) with no system prompt")

#%% the same readouts as text: the top J-lens tokens of each contrast at a few layers, for comparing contrasts without the tabs

show_jlens_text = True
if show_jlens_text:
    text_layers = [20, 30, 36, 42, 48, 52, 56, 60, 62]
    n_top = 9
    contrasts = {
        "hack - clean, no system prompt": lambda p: mean_resid("hack", "none", p) - mean_resid("clean", "none", p),
        "hack - clean, very_hacker": lambda p: mean_resid("hack", "very_hacker", p) - mean_resid("clean", "very_hacker", p),
        "very_hacker - none, hack prompts": lambda p: mean_resid("hack", "very_hacker", p) - mean_resid("hack", "none", p),
        "very_hacker - generic, hack prompts": lambda p: mean_resid("hack", "very_hacker", p) - mean_resid("hack", "generic", p),
        "generic - none, hack prompts": lambda p: mean_resid("hack", "generic", p) - mean_resid("hack", "none", p),
    }
    for name, diff in contrasts.items():
        for p in (-2, -1):
            v = resid_pre_layout(diff(p)).to(model.device, model.W_U.dtype)
            print(f"{cyan}{name}, position {p}{endc}")
            for layer in text_layers:
                top = get_lens_logits(v[layer], layer, model, jlens).float().topk(n_top).indices.tolist()
                print(f"L{layer} |d| {v[layer].float().norm():.1f}: " + " ".join(repr(model.tokenizer.decode(i).strip())[1:-1] for i in top))

#%% the oracle lens on a second copy of the model, since peft cannot wrap a booted Bridge: the Qwen3.6-27B lens on the Qwen3.8 weights, no 3.8 checkpoint covering these layers. Its layer L is resid_post L, the cache's row L

load_oracle = False
if load_oracle:
    reader, contract = load_olens(load_hf_model(MODEL_ID, device_map="cuda"))

#%% check: the oracle lens on plain activations of one prompt, at " even", " %", </think> and the newlines after it

check_olens = False
if check_olens:
    key = ("parity", "even", "hack", "none", False)
    ids = cache[key]["ids"]
    strs = [model.tokenizer.decode(i) for i in ids]
    rows = {f"blocks.{layer}.hook_resid_post": cache[key]["resid"][layer][None].to("cuda") for layer in OLENS_LAYERS}
    _ = olens_readout(rows, OLENS_LAYERS, [strs.index(" even"), strs.index(" %"), len(ids) - 2, len(ids) - 1], reader, model.tokenizer, contract, n=2, input_src=ids, title="raw activations, parity even hack prompt")

#%% oracle lens readouts of the contrasts at </think> and the newlines after it, with a random direction as the control

show_olens = False
if show_olens:
    t.manual_seed(0)
    rnd = {p: t.randn(64, 5120) for p in (-2, -1)}
    _ = olens_diff(lambda p: mean_resid("hack", "none", p) - mean_resid("clean", "none", p), "hack - clean, no system prompt")
    _ = olens_diff(lambda p: mean_resid("hack", "very_hacker", p) - mean_resid("clean", "very_hacker", p), "hack - clean, very_hacker")
    _ = olens_diff(lambda p: mean_resid("hack", "very_hacker", p) - mean_resid("hack", "generic", p), "very_hacker - generic, hack prompts")
    _ = olens_diff(lambda p: mean_resid("hack", "generic", p) - mean_resid("hack", "none", p), "generic - no system prompt, hack prompts")
    _ = olens_diff(lambda p: rnd[p], "random direction")

#%% very_hacker minus generic on hack prompts as a probe: its unit row at every layer, from </think> and from the newlines after it, with random unit directions as the null,
# dotted with resid_post at every token of a reasoning-on run. Saved per rollout, so the cells below are arithmetic on acts/prompt_dom_probe.pt and need no model.

harvest_probe = False
if harvest_probe:
    rollouts = "2026-11-05T18-16-40-00-00_grader_Sr9SXB9MSUfodVJXq2TsSU"   # main38: the main families, hack prompts, no system prompt, effort low
    # rollouts = "2026-11-05T17-54-51-00-00_grader_XGBzMUHmDsCbASBvXpozTs"   # heldout38: the held-out families
    n_random = 20
    t.manual_seed(0)
    records = [r for r in map(json.loads, open(f"data/inspect/grader/{rollouts}.jsonl")) if r["labels"]["cheat"] is not None]
    probe = t.stack([mean_resid("hack", "very_hacker", p) - mean_resid("hack", "generic", p) for p in (-2, -1)], dim=1)   # [n_layers, 2, d_model]
    dirs = t.nn.functional.normalize(t.cat([probe, t.randn(probe.shape[0], n_random, probe.shape[2])], dim=1), dim=-1).to("cuda")
    items = []
    for r in pbar(records):
        start, close = cot_span(r["ids"], model.tokenizer)
        items.append(dict(ids=r["ids"], start=start, close=close, family=r["labels"]["family"], side=r["labels"]["side"], cheat=bool(r["labels"]["cheat"]), scores=t.einsum("lsd,lkd->lsk", resid_post(model, r["ids"]), dirs).cpu()))
    t.save(dict(rollouts=rollouts, items=items), PROBE_CACHE)
    print(f"{green}{len(items)} rollouts, {sum(x['cheat'] for x in items)} cheating, {PROBE_CACHE.stat().st_size / 1e9:.1f} GB in {PROBE_CACHE}{endc}")

#%% how well the probe predicts cheating: within-cell AUROC by layer for three rollout scores (mean over the CoT, at </think>, at the newlines after it), and for single CoT tokens.
# Each direction's layer is chosen on the even rollouts and scored on the odd ones, random directions the same way. Answer tokens are never read, since they are the label.

score_probe = True
if score_probe:
    plot_read = "cot mean"
    # plot_read = "</think>"
    # plot_read = "newlines"
    items = t.load(PROBE_CACHE)["items"]
    cell_names = sorted({(x["family"], x["side"]) for x in items})
    cell = t.tensor([cell_names.index((x["family"], x["side"])) for x in items])
    cheat = t.tensor([x["cheat"] for x in items])
    dev = t.arange(len(items)) % 2 == 0
    sources = ["</think>", "newlines"]
    reads = {
        "cot mean": t.stack([x["scores"][:, x["start"]:x["close"]].mean(1) for x in items]),   # [n_rollouts, n_layers, 2 + n_random]
        "</think>": t.stack([x["scores"][:, x["close"]] for x in items]),
        "newlines": t.stack([x["scores"][:, x["close"] + 1] for x in items]),
    }
    auroc = {name: dict(dev=cell_auroc(x[dev], cheat[dev], cell[dev]), test=cell_auroc(x[~dev], cheat[~dev], cell[~dev]), all=cell_auroc(x, cheat, cell)) for name, x in reads.items()}   # each [n_layers, 2 + n_random]
    cot_len = t.tensor([float(x["close"] - x["start"]) for x in items])
    print(f"{gray}{cheat.sum()} cheat / {(~cheat).sum()} clean, cells with both classes {sum(cheat[cell == c].any() and not cheat[cell == c].all() for c in cell.unique())}; CoT length alone: AUROC {cell_auroc(cot_len, cheat, cell):.3f}{endc}")
    print(f"{cyan}read        direction  layer   dev    test   random, own best layer: mean, max   random, this layer: mean, max{endc}")
    for name, a in auroc.items():
        best = a["dev"].argmax(0)
        test = a["test"].gather(0, best[None])[0]
        for k, source in enumerate(sources):
            here = a["test"][best[k], 2:]
            print(f"{name:10s}  {source:9s}  L{best[k]:<4d}  {a['dev'][best[k], k]:.3f}  {test[k]:.3f}  {test[2:].mean():.3f}, {test[2:].max():.3f}                        {here.mean():.3f}, {here.max():.3f}")
    # single tokens, each one counted once, at the layer chosen for the CoT mean
    owner = t.cat([t.full((x["close"] - x["start"],), i) for i, x in enumerate(items)])
    tokens = t.cat([x["scores"][:, x["start"]:x["close"]].transpose(0, 1) for x in items])   # [n_tokens, n_layers, 2 + n_random]
    held = ~dev[owner]
    for k, source in enumerate(sources):
        layer = auroc["cot mean"]["dev"][:, k].argmax().item()
        a = cell_auroc(tokens[held, layer], cheat[owner][held], cell[owner][held])
        print(f"{'one token':10s}  {source:9s}  L{layer:<4d}         {a[k]:.3f}                                {a[2:].mean():.3f}, {a[2:].max():.3f}")
    # per cell, at the same layer, from the newlines
    layer = auroc["cot mean"]["dev"][:, 1].argmax().item()
    print(f"{cyan}per cell, CoT mean, direction from the newlines, L{layer}, all rollouts{endc}")
    for c, (family, side) in enumerate(cell_names):
        y = cheat[cell == c]
        if y.any() and not y.all():
            print(f"  {family + '.' + side:22s} {y.sum():3d} cheat {(~y).sum():3d} clean  {cell_auroc(reads['cot mean'][cell == c][:, layer, 1], y, cell[cell == c]):.3f}")
    a = auroc[plot_read]["all"]
    line(
        [a[:, 0], a[:, 1], a[:, 2:].max(1).values, a[:, 2:].min(1).values],
        x=list(range(a.shape[0])),
        names=["probe from </think>", "probe from newlines", "best of 20 random", "worst of 20 random"],
        labels=dict(x="layer", y="within-cell AUROC"),
        title=f"very_hacker - generic as a cheating probe, {plot_read}, all rollouts",
        **DARK,
    )

#%% the same scores with no cell structure: AUROC over all rollouts at once, which asks whether one threshold works across prompts, at the layers chosen above

pooled_probe = True
if pooled_probe:
    one_cell = t.zeros_like(cell)
    print(f"{cyan}read        direction  layer   within-cell  pooled   random pooled: mean, max{endc}")
    for name, x in reads.items():
        for k, source in enumerate(sources):
            layer = auroc[name]["dev"][:, k].argmax().item()
            pooled = cell_auroc(x[~dev][:, layer], cheat[~dev], one_cell[~dev])
            print(f"{name:10s}  {source:9s}  L{layer:<4d}  {auroc[name]['test'][layer, k]:.3f}        {pooled[k]:.3f}    {pooled[2:].mean():.3f}, {pooled[2:].max():.3f}")

#%% when the signal arrives: within-cell AUROC of each rollout's mean score over each tenth of its CoT, at one layer, on the odd rollouts, with the random directions at the same layer.
# The end of the CoT comes after the decision is written down, so the early tenths are the ones that predict.

by_position = True
if by_position:
    read_direction = 1   # 0 is the probe from </think>, 1 from the newlines after it
    read_layer = auroc["cot mean"]["dev"][:, read_direction].argmax().item()
    # read_layer = 24
    n_bins = 10
    chunks = t.stack([t.stack([c.mean(0) for c in x["scores"][read_layer, x["start"]:x["close"]].tensor_split(n_bins)]) for x in items])   # [n_rollouts, n_bins, 2 + n_random]
    curve = cell_auroc(chunks[~dev], cheat[~dev], cell[~dev])   # [n_bins, 2 + n_random]
    print(f"{cyan}tenth of the CoT   probe from {sources[read_direction]}, L{read_layer}   random: mean, max{endc}")
    for k in range(n_bins):
        print(f"  {k + 1:2d}               {curve[k, read_direction]:.3f}                     {curve[k, 2:].mean():.3f}, {curve[k, 2:].max():.3f}")
    line(
        [curve[:, read_direction], curve[:, 2:].max(1).values, curve[:, 2:].min(1).values],
        x=list(range(1, n_bins + 1)),
        names=[f"probe from {sources[read_direction]}", "best of 20 random", "worst of 20 random"],
        labels=dict(x="tenth of the CoT", y="within-cell AUROC"),
        title=f"when very_hacker - generic separates cheating rollouts, L{read_layer}, odd rollouts",
        **DARK,
    )

#%% token-level readouts: pairs of rollouts from the odd half with one prompt, one cheating and one clean, each token shaded by its score at one layer, centered on the mean CoT token
# and clipped so strips share a scale. Red is above the mean, along very_hacker minus generic. The strip runs from <think> through the answer.

show_probe_toks = True
if show_probe_toks:
    read_direction = 1   # 0 is the probe from </think>, 1 from the newlines after it
    read_layer = auroc["cot mean"]["dev"][:, read_direction].argmax().item()
    # read_layer = 24
    n_pairs = 4
    clip_sds = 2.5
    cot_tokens = tokens[:, read_layer, read_direction]
    center = cot_tokens.mean()
    clip = clip_sds * cot_tokens.std()
    held_items = [x for x, d in zip(items, dev) if not d]
    balance = {name: min(sum(x["cheat"] == y and (x["family"], x["side"]) == name for x in held_items) for y in (True, False)) for name in cell_names}
    for name in sorted(cell_names, key=lambda c: -balance[c])[:n_pairs]:
        for label in (True, False):
            x = next(x for x in held_items if (x["family"], x["side"]) == name and x["cheat"] == label)
            s = x["scores"][read_layer, :, read_direction]
            show_toks(
                x["ids"][x["start"] - 1:],
                model.tokenizer,
                vals=(s[x["start"] - 1:] - center).clamp(-clip, clip),
                val_name="score",
                title=f"{name[0]}.{name[1]} {'CHEAT' if label else 'clean'}: L{read_layer}, CoT mean {s[x['start']:x['close']].mean() - center:+.2f}, one token sd {cot_tokens.std():.2f}",
            )

#%% the shown strips as data for the research log: tokens and centered, clipped scores of the pairs above, from <think> through the answer

export_strips = True
if export_strips:
    strip_path = Path("acts/prompt_dom_strips.json")
    strips = []
    for name in sorted(cell_names, key=lambda c: -balance[c])[:n_pairs]:
        for label in (True, False):
            x = next(x for x in held_items if (x["family"], x["side"]) == name and x["cheat"] == label)
            s = x["scores"][read_layer, :, read_direction]
            strips.append(dict(family=name[0], side=name[1], cheat=label, layer=read_layer, cot_mean=(s[x["start"]:x["close"]].mean() - center).item(), close=(s[x["close"]] - center).item(), newlines=(s[x["close"] + 1] - center).item(), toks=[model.tokenizer.decode(i) for i in x["ids"][x["start"] - 1:]], vals=(s[x["start"] - 1:] - center).clamp(-clip, clip).tolist()))
    strip_path.write_text(json.dumps(strips, ensure_ascii=False))
    print(f"{green}{len(strips)} strips in {strip_path}, sd {cot_tokens.std():.3f}, clip {clip:.3f}{endc}")
    for r in strips:
        print(f"{r['family']}.{r['side']} {'CHEAT' if r['cheat'] else 'clean'}: {len(r['toks'])} tokens, CoT mean {r['cot_mean']:+.2f}, </think> {r['close']:+.2f}, newlines {r['newlines']:+.2f}")
