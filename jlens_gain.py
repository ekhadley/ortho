#%%
"""Jacobian sensitivity of the saved directions: the gain ||J_L v|| of a unit direction v at resid_pre L through the Qwen3.6-27B J-lens (J_L the averaged first-order
map from hidden_states[L] to hidden_states[62], the convention of mechtools.lens and utils.resid_pre_layout, so row r of a train_vector.py-layout vector goes through
J_r), and the logit gain ||W_U diag(1 + w_norm) J_L v||, the linear part of the lens readout. Reference classes at every fourth layer: random unit directions, the MLP
decoder directions written into resid_pre L (down_proj columns of layer L - 1), the MLP encoder directions read at L (gate_proj rows times the post-attention norm's
1 + w), and the pullbacks J_L^T of the unembedding's cluster centroids (spherical k-means, k = 1024, on the mean-centred unembedding, the clustering of the lens
readouts), each from the 3.6 weights the lens was fit on and from the 3.8 weights the directions come from: the basis-mismatch check, since a 3.8 decoder reading like a
random direction would mean 3.8 directions cannot be read through the 3.6 lens. Weights are fetched one tensor at a time by byte range and cached under acts/weights.
Runs locally: the lens by mmap one layer at a time and the k-means on the CPU, the matmuls on the 12 GB GPU. Writes data/vectors/Qwen3.8-27B/jlens/jlens_gain.{json,html}."""
import json
import struct
from pathlib import Path

import requests
import torch as t
from torch import Tensor
from safetensors.torch import load_file, save_file
from huggingface_hub import hf_hub_download, hf_hub_url
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mechtools import *

LENS = "qwen3.6-27b/j-lens/lens.pt"
MODEL_36, MODEL_38, REVISION_38 = "Qwen/Qwen3.6-27B", "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
VECTORS = Path("data/vectors/Qwen3.8-27B")
WEIGHTS = Path("acts/weights")
OUT = VECTORS / "jlens"
LAYERS = list(range(1, 63))            # resid_pre layers the lens covers; J_62 is the identity
REF_LAYERS = list(range(4, 61, 4))     # where the weight-derived reference classes are computed
N_RANDOM, K, SEED = 1024, 1024, 0
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
PROBE_ROW = 52                         # cot_probe.json's row: output_hidden_states index, so resid_pre 52
DEVICE = "cuda"

#%% weights: one tensor of a sharded checkpoint by byte range, cached
def fetch(model: str, name: str, revision: str | None = None) -> Tensor:
    path = WEIGHTS / model.split("/")[1] / f"{name}.safetensors"
    if path.exists():
        return load_file(path)[name]
    index = json.load(open(hf_hub_download(model, "model.safetensors.index.json", revision=revision)))["weight_map"]
    url = hf_hub_url(model, index[name], revision=revision)
    header_len = struct.unpack("<Q", requests.get(url, headers={"Range": "bytes=0-7"}).content)[0]
    meta = json.loads(requests.get(url, headers={"Range": f"bytes=8-{7 + header_len}"}).content)[name]
    assert meta["dtype"] == "BF16", meta
    begin, end = meta["data_offsets"]
    buf, got = bytearray(end - begin), 0
    with requests.get(url, headers={"Range": f"bytes={8 + header_len + begin}-{8 + header_len + end - 1}"}, stream=True) as r:
        r.raise_for_status()
        for chunk in r.iter_content(1 << 22):
            buf[got:got + len(chunk)] = chunk
            got += len(chunk)
    assert got == end - begin, (got, end - begin)
    x = t.frombuffer(buf, dtype=t.bfloat16).reshape(meta["shape"])
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({name: x}, path)
    print(f"{gray}fetched {model} {name} {tuple(x.shape)}{endc}")
    return x

def mlp_dirs(model: str, layer: int, revision: str | None = None) -> tuple[Tensor, Tensor]:
    """(decoder directions into resid_pre `layer` [d_mlp, d_model], encoder directions read at `layer` [d_mlp, d_model]) in residual space."""
    decoder = fetch(model, f"model.language_model.layers.{layer - 1}.mlp.down_proj.weight", revision).T
    gate = fetch(model, f"model.language_model.layers.{layer}.mlp.gate_proj.weight", revision)
    norm_w = fetch(model, f"model.language_model.layers.{layer}.post_attention_layernorm.weight", revision)
    return decoder, gate * (1 + norm_w.float()).to(gate.dtype)

unembed_36, unembed_38 = fetch(MODEL_36, "lm_head.weight"), fetch(MODEL_38, "lm_head.weight", REVISION_38)
final_scale_36 = 1 + fetch(MODEL_36, "model.language_model.norm.weight").float()
final_scale_38 = 1 + fetch(MODEL_38, "model.language_model.norm.weight", REVISION_38).float()

#%% basis check: do the two models share a residual basis? cosine of matching unembedding rows
def row_cosines(a: Tensor, b: Tensor, chunk: int = 16384) -> Tensor:
    return t.cat([t.cosine_similarity(a[i:i + chunk].to(DEVICE).float(), b[i:i + chunk].to(DEVICE).float(), dim=1).cpu() for i in range(0, len(a), chunk)])
unembed_cos = row_cosines(unembed_36, unembed_38)
print(f"{cyan}unembedding rows 3.6 vs 3.8: cosine median {unembed_cos.median():.3f}, 10th pct {unembed_cos.quantile(0.1):.3f}, min {unembed_cos.min():.3f}{endc}")

#%% logit metric: G = diag(s) W_U^T W_U diag(s), so ||W_U diag(s) y||^2 = y^T G y, accumulated in fp32 on the GPU
def logit_metric(unembed: Tensor, scale: Tensor, chunk: int = 16384) -> Tensor:
    g = t.zeros(unembed.shape[1], unembed.shape[1], device=DEVICE)
    for i in range(0, len(unembed), chunk):
        w = unembed[i:i + chunk].to(DEVICE).float()
        g += w.T @ w
    return scale.to(DEVICE)[:, None] * g * scale.to(DEVICE)[None, :]
metric_36 = logit_metric(unembed_36, final_scale_36)

#%% vocab clusters: the readouts' clustering (mean-centred unembedding, spherical k-means) on 3.6, the same labels for 3.8's centroids; bf16 on the GPU in chunks, the fp32 copy would not fit
CHUNK = 16384
def centred_unit(unembed: Tensor) -> Tensor:
    """normed(W_U - mean(W_U)) as bf16 on the GPU, built in chunks of rows."""
    mean = sum(unembed[i:i + CHUNK].float().sum(0) for i in range(0, len(unembed), CHUNK)) / len(unembed)
    out = t.empty(unembed.shape, dtype=t.bfloat16, device=DEVICE)
    for i in range(0, len(unembed), CHUNK):
        out[i:i + CHUNK] = normed(unembed[i:i + CHUNK].to(DEVICE).float() - mean.to(DEVICE)).to(t.bfloat16)
    return out
def cluster_means(x: Tensor, labels: Tensor, k: int) -> Tensor:
    centroids = t.zeros(k, x.shape[1], device=DEVICE)
    for i in range(0, len(x), CHUNK):
        centroids.index_add_(0, labels[i:i + CHUNK], x[i:i + CHUNK].float())
    return normed(centroids)
def spherical_kmeans(x: Tensor, k: int, iters: int, seed: int) -> Tensor:
    """mechtools.stats.kmeans's loop (k random rows, Lloyd's iterations, stop when no label changes) on a bf16 GPU matrix of unit rows; returns labels."""
    centroids = x[t.randperm(len(x), generator=t.Generator().manual_seed(seed))[:k].to(DEVICE)].float()
    labels = None
    for _ in pbar(range(iters), desc="kmeans"):
        new = (x @ centroids.to(x.dtype).T).argmax(-1)
        if labels is not None and t.equal(new, labels):
            break
        labels = new
        centroids = cluster_means(x, labels, k)
    return labels
labels_path = WEIGHTS / f"vocab_labels_36_k{K}_s{SEED}.pt"
unit_36 = centred_unit(unembed_36)
labels = t.load(labels_path).to(DEVICE) if labels_path.exists() else spherical_kmeans(unit_36, K, 150, SEED)
t.save(labels.cpu(), labels_path)
centroids_36 = cluster_means(unit_36, labels, K).cpu()
del unit_36
unit_38 = centred_unit(unembed_38)
centroids_38 = cluster_means(unit_38, labels, K).cpu()
del unit_38
tec()
alive = t.bincount(labels.cpu(), minlength=K) > 0
centroids_36, centroids_38 = centroids_36[alive], centroids_38[alive]
print(f"{cyan}{int(alive.sum())} of {K} clusters non-empty; centroid cosine 3.6 vs 3.8 median {t.cosine_similarity(centroids_36, centroids_38, dim=1).median():.3f}{endc}")

#%% the directions: every train_vector-layout file (row r = resid_pre r), the probe at its row, the pins as a class of their own
saved = {p.stem: f for p in sorted(VECTORS.glob("*.safetensors")) if "direction_unit" in (f := load_file(p))}
directions = {name: {r: f["direction_unit"][r].float() for r in LAYERS if f["direction_unit"][r].norm() > 0} for name, f in saved.items()}
directions["cot_probe"] = {PROBE_ROW: load_file(VECTORS / "cot_probe.safetensors")["w"].float()}
pins = {name: f["mean_resid"].float() for name, f in saved.items() if "mean_resid" in f}
random_dirs = normed(t.randn(N_RANDOM, unembed_36.shape[1], generator=t.Generator().manual_seed(SEED)))
for name, rows in directions.items():
    print(f"{gray}{name}: rows {min(rows)}-{max(rows)} ({len(rows)}){endc}")

#%% gains: ||J x̂|| and ||W_U diag(s) J x̂|| per direction, the lens one layer at a time
lens = t.load(hf_hub_download("camilablank/workspace-lenses", LENS), map_location="cpu", weights_only=False, mmap=True)
assert t.allclose(lens["J"][62].float(), t.eye(unembed_36.shape[1])), "J_62 is the identity in this lens convention"
def gains(J: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
    y = normed(x.to(DEVICE).float()) @ J.T
    return y.norm(dim=1).cpu(), t.sqrt(((y @ metric_36) * y).sum(1)).cpu()
def quantiles(g: Tensor) -> list[float]:
    return g.quantile(t.tensor(QUANTILES)).tolist()
results = {"random": {}, "pin": {}, "direction": {name: {} for name in directions}, "reference": {}}
for L in pbar(LAYERS, desc="layers"):
    J = lens["J"][L].float().to(DEVICE)
    rand_gain, rand_logit = gains(J, random_dirs)
    results["random"][L] = dict(gain=quantiles(rand_gain), logit=quantiles(rand_logit))
    pin_gain, pin_logit = gains(J, t.stack([p[L] for p in pins.values()]))
    results["pin"][L] = dict(gain=quantiles(pin_gain), logit=quantiles(pin_logit))
    for name, rows in directions.items():
        if L in rows:
            g, lg = gains(J, rows[L][None])
            results["direction"][name][L] = dict(gain=g.item(), logit=lg.item(), ratio=g.item() / rand_gain.median().item(), logit_ratio=lg.item() / rand_logit.median().item(), percentile=(rand_gain < g).float().mean().item())
    if L in REF_LAYERS:
        decoder_36, encoder_36 = mlp_dirs(MODEL_36, L)
        decoder_38, encoder_38 = mlp_dirs(MODEL_38, L, REVISION_38)
        neuron_cos = t.cosine_similarity(decoder_36.float(), decoder_38.float(), dim=1)
        pullback_36, pullback_38 = (centroids_36 * final_scale_36).to(DEVICE) @ J, (centroids_38 * final_scale_38).to(DEVICE) @ J
        pullback_rand = (random_dirs * final_scale_36).to(DEVICE) @ J   # the control for the pullbacks: anything pulled back through J^T lands in the lens's high-gain subspace
        results["reference"][L] = {"neuron_cos_median": neuron_cos.median().item()}
        for cls, x in [("decoder_36", decoder_36), ("decoder_38", decoder_38), ("encoder_36", encoder_36), ("encoder_38", encoder_38), ("pullback_36", pullback_36), ("pullback_38", pullback_38), ("pullback_rand", pullback_rand)]:
            g, lg = gains(J, x)
            results["reference"][L][cls] = dict(gain=quantiles(g), logit=quantiles(lg), ratio=(g.median() / rand_gain.median()).item())
        del decoder_36, encoder_36, decoder_38, encoder_38
    del J
    tec()

#%% tables: the reference classes by layer, then every direction over its rows
print(f"\n{cyan}reference classes: median gain / random median gain (and the median of the random gain itself){endc}")
print(f"{'layer':>5} {'random':>7} {'dec 3.6':>8} {'dec 3.8':>8} {'enc 3.6':>8} {'enc 3.8':>8} {'pull 3.6':>9} {'pull 3.8':>9} {'pull rand':>9} {'pins':>6} {'neuron cos':>10}")
for L in REF_LAYERS:
    ref, rnd = results["reference"][L], results["random"][L]["gain"][2]
    print(f"{L:>5} {rnd:>7.3f} " + " ".join(f"{ref[c]['ratio']:>{w}.2f}" for c, w in [("decoder_36", 8), ("decoder_38", 8), ("encoder_36", 8), ("encoder_38", 8), ("pullback_36", 9), ("pullback_38", 9), ("pullback_rand", 9)]) + f" {results['pin'][L]['gain'][2] / rnd:>6.2f} {ref['neuron_cos_median']:>10.3f}")
print(f"\n{cyan}directions: medians over their rows of gain / random median, percentile among the {N_RANDOM} random directions, logit gain / random median{endc}")
print(f"{'direction':<24} {'rows':>7} {'gain ratio':>10} {'pctile':>7} {'logit ratio':>11}")
for name, by_layer in results["direction"].items():
    rows = sorted(by_layer)
    med = lambda key: t.tensor([by_layer[L][key] for L in rows]).median().item()
    print(f"{name:<24} {f'{rows[0]}-{rows[-1]}':>7} {med('ratio'):>10.2f} {med('percentile'):>7.2f} {med('logit_ratio'):>11.2f}")

#%% save: the json and a two-panel plot, gain by layer on a log axis, bands for the classes and a line per direction
OUT.mkdir(exist_ok=True)
(OUT / "jlens_gain.json").write_text(json.dumps(dict(lens=LENS, provenance=lens["provenance"], layers=LAYERS, ref_layers=REF_LAYERS, n_random=N_RANDOM, k=K, quantiles=QUANTILES, unembed_cos_median=unembed_cos.median().item(), results=results), indent=1, default=str))
fig = make_subplots(rows=1, cols=2, subplot_titles=["residual gain  ||J_L v||  (hidden_states[62] shift per unit shift at resid_pre L)", "logit gain  ||W_U diag(1+w) J_L v||"], horizontal_spacing=0.06)
def band(xs: list[int], qs: list[list[float]], name: str, color: str, col: int, lo: int = 0, hi: int = 4, dash: str | None = None, show_band: bool = True):
    if show_band:
        fig.add_trace(go.Scatter(x=xs + xs[::-1], y=[q[hi] for q in qs] + [q[lo] for q in qs][::-1], fill="toself", fillcolor=color + "2e", line=dict(width=0), name=name, legendgroup=name, showlegend=False, hoverinfo="skip"), row=1, col=col)
    fig.add_trace(go.Scatter(x=xs, y=[q[2] for q in qs], mode="lines", line=dict(color=color, width=2, dash=dash), name=name, legendgroup=name, showlegend=col == 1), row=1, col=col)
for col, key in [(1, "gain"), (2, "logit")]:
    band(LAYERS, [results["random"][L][key] for L in LAYERS], "random (5-95%)", "#9a9a90", col)
    band(LAYERS, [results["pin"][L][key] for L in LAYERS], "mean residual (pins)", "#c3c2b7", col, dash="dot", show_band=False)
    for cls, color, dash, show in [("decoder_36", SERIES[0], None, True), ("decoder_38", SERIES[0], "dash", False), ("encoder_36", SERIES[2], None, True), ("encoder_38", SERIES[2], "dash", False), ("pullback_36", SERIES[3], None, True), ("pullback_38", SERIES[3], "dash", False), ("pullback_rand", SERIES[1], "dot", False)]:
        band(REF_LAYERS, [results["reference"][L][cls][key] for L in REF_LAYERS], f"{cls.replace('_36', ' 3.6').replace('_38', ' 3.8')} (5-95%)" if show else cls.replace("_38", " 3.8 (median)").replace("_rand", " of random vectors (median)"), color, col, dash=dash, show_band=show)
    palette = ["#e6194b", "#f58231", "#ffe119", "#bfef45", "#3cb44b", "#42d4f4", "#4363d8", "#911eb4", "#f032e6", "#fabed4", "#ffd8b1", "#aaffc3", "#dcbeff", "#fffac8", "#800000", "#469990"]
    for i, (name, by_layer) in enumerate(results["direction"].items()):
        xs = sorted(by_layer)
        fig.add_trace(go.Scatter(x=xs, y=[by_layer[L][key] for L in xs], mode="lines+markers" if len(xs) < 3 else "lines", marker=dict(size=10, symbol="diamond"), line=dict(color=palette[i % len(palette)], width=1.5), name=name, legendgroup=name, showlegend=col == 1, hovertemplate=f"{name}<br>L%{{x}}: %{{y:.3f}}"), row=1, col=col)
fig.update_yaxes(type="log", title_text="gain", row=1, col=1)
fig.update_yaxes(type="log", row=1, col=2)
fig.update_xaxes(title_text="layer L (resid_pre; row of the saved vector)")
fig.update_layout(**DARK, title="Jacobian sensitivity of saved directions through the Qwen3.6-27B J-lens, against random, MLP and unembedding-cluster directions", height=720, width=1700, legend=dict(font=dict(size=10)))
write_dark_html(fig, str(OUT / "jlens_gain.html"))
print(f"{green}saved {OUT / 'jlens_gain.json'} and {OUT / 'jlens_gain.html'}{endc}")
