#%%
"""The fd_gain.py summary page: a two-page findings report with outsider explanations, inline SVG charts from acts/fd_gain/fd_gain.json and one rollout's per-position
arrays, written to ~/wgmn/artifacts/<date>_fd-gain-sensitivity.html (and figures/). Run after fd_gain.py analyze: uv run python fd_gain_page.py"""
import json
import shutil
from pathlib import Path

import torch as t

from mechtools import *

ACTS = Path("acts/fd_gain")
OUT = Path.home() / "wgmn/artifacts/2026-11-09_fd-gain-sensitivity.html"
EPS_MAIN = 0.1
TRACE_LAYER = 48
COLORS = dict(grader_kto="var(--accent)", grader_proj="var(--amber)", grader_dpo="var(--purple)", nocot_dom_last="var(--teal)")   # one fixed hue per named direction; everything else gray
GRAYS = ["#c9d2dc", "#a4b0bd", "#7f8b99", "#5f6b79", "#b7c0ca"]
LABELS = {"grader_kto": "grader_kto", "grader_proj": "grader_proj", "grader_dpo": "grader_dpo", "nocot_dom_last": "nocot_dom_last", "will_cm": "Will's CoT-mean", "mbpp_kto": "mbpp_kto", "mbpp_dom": "mbpp_dom", "pins": "pins (median of 4)", "pullbacks": "lens pullbacks", "mlp": "MLP neurons", "cot_probe": "cot_probe"}

#%% data
data = json.loads((ACTS / "fd_gain.json").read_text())
LAYERS = data["layers"]
def table(key: str) -> dict: return data["tables"][key]
def row(tab: dict, name: str) -> list[float]:
    rows = tab["rows"]
    if name == "pins": return t.tensor([rows[f"pin_{n}"] for n in ["grader_proj", "grader_kto", "mbpp_dom", "mbpp_kto"]]).nanmedian(0).values.tolist()
    if name == "pullbacks": return t.tensor([rows["pullback_cluster (median of 6)"], rows["pullback_random (median of 6)"]]).nanmedian(0).values.tolist()
    if name == "mlp": return t.tensor([rows["mlp_dec (median of 12)"], rows["mlp_enc (median of 12)"]]).nanmedian(0).values.tolist()
    return rows[name]
def med(xs: list[float]) -> float: return t.tensor(xs).nanmedian().item()
ALL, START, END = table("all/from_perturbed_mean/central"), table("cot_start/at_start/central"), table("cot_end/at_end/central")
ALL_KL = table("all/cot_mean/kl")
beats_all, beats_start = {k[4:]: v for k, v in data["beats"].items() if k.startswith("all/")}, {k[10:]: v for k, v in data["beats"].items() if k.startswith("cot_start/")}
context = {k[4:]: v for k, v in data["context"].items() if k.startswith("all/")}
cells = ["grader_cheat", "grader_clean", "mbpp_cheat", "mbpp_honest"]

# one rollout's per-position gains at TRACE_LAYER, every position perturbed: the shortest cheating grader rollout
results = sorted(ACTS.glob("results/[0-9]*.pt"))
dirs = t.load(ACTS / "results/directions.pt", weights_only=False)
names, classes = dirs["names"], dirs["classes"]
rand = [i for i, c in enumerate(classes) if c == "random"]
picked = min((t.load(p, weights_only=False) for p in results), key=lambda r: r["T"] if r["cell"] == "grader_cheat" else 10 ** 9)
lay = picked["layers"][TRACE_LAYER]
ds = [lay["pairs"][k][1] for k in lay["pos_idx"]["all"]]
g = lay["pos"]["all"][:, 0].float() / (EPS_MAIN * lay["norm"])
rg = g[[ds.index(d) for d in rand]]
trace = dict(T=picked["T"], p_start=picked["p_start"], p_end=picked["p_end"], rmed=rg.median(0).values.tolist(), lo=rg.quantile(0.05, 0).tolist(), hi=rg.quantile(0.95, 0).tolist(), kto=g[ds.index(names.index("grader_kto"))].tolist(), dom=g[ds.index(names.index("mbpp_dom"))].tolist())

#%% svg helpers
def fmt(x: float) -> str: return f"{x:.1f}"
def line_chart(series: list[tuple], band: tuple[list, list] | None, x: list, ylim: tuple, w: int, h: int, title: str, ylabel: str, xlabel: str, ox: int = 0, oy: int = 0, xticks: list | None = None, xfmt=str, label_dx: int = 8) -> str:
    """series: (values, color, dash, label); band: (lo, hi) in gray. Axes recessive, direct labels at the line ends spread apart vertically."""
    L, R, T, B = 52 + ox, 150 + ox, 36 + oy, 36 + oy
    pw, ph = w - 52 - 150, h - 36 - 36
    X = lambda v: L + (x.index(v) if v in x else v) / (len(x) - 1) * pw if isinstance(v, int) and v in x else L + (v - x[0]) / (x[-1] - x[0]) * pw
    Y = lambda v: T + (ylim[1] - min(max(v, ylim[0]), ylim[1])) / (ylim[1] - ylim[0]) * ph
    out = [f'<text x="{L}" y="{T - 14}" font-size="13" font-weight="600" fill="var(--text)">{title}</text>']
    step = 0.5 if ylim[1] - ylim[0] > 1.5 else 0.25
    yt = -(-ylim[0] // step) * step
    while yt <= ylim[1] + 1e-9:
        out.append(f'<line x1="{L}" y1="{Y(yt):.1f}" x2="{L + pw}" y2="{Y(yt):.1f}" stroke="var(--line)" stroke-width="1"/><text x="{L - 8}" y="{Y(yt) + 4:.1f}" text-anchor="end" font-size="11.5" fill="var(--muted)">{yt:g}</text>')
        yt += step
    for xv in (xticks or x): out.append(f'<text x="{X(xv):.1f}" y="{T + ph + 18}" text-anchor="middle" font-size="11.5" fill="var(--muted)">{xfmt(xv)}</text>')
    out.append(f'<text x="{L + pw / 2:.1f}" y="{T + ph + 34}" text-anchor="middle" font-size="11.5" fill="var(--muted)">{xlabel}</text>')
    out.append(f'<text transform="translate({L - 40},{T + ph / 2:.1f}) rotate(-90)" text-anchor="middle" font-size="11.5" fill="var(--muted)">{ylabel}</text>')
    if band:
        lo, hi = band
        pts = " ".join(f"{X(xv):.1f},{Y(v):.1f}" for xv, v in zip(x, hi)) + " " + " ".join(f"{X(xv):.1f},{Y(v):.1f}" for xv, v in zip(x[::-1], lo[::-1]))
        out.append(f'<polygon points="{pts}" fill="#a4b0bd" fill-opacity=".22" stroke="none"><title>random directions, 5th to 95th percentile</title></polygon>')
    ends = [[Y((band[0][-1] + band[1][-1]) / 2), "random 5-95%", "#a4b0bd", X(x[-1])]] if band else []
    for vals, color, dash, label in series:
        pts = [(X(xv), Y(v)) for xv, v in zip(x, vals) if v == v]
        if not pts: continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{px:.1f},{py:.1f}" for i, (px, py) in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}" stroke-linejoin="round"/>')
        for (px, py), xv, v in zip(pts, [xv for xv, v in zip(x, vals) if v == v], [v for v in vals if v == v]):
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}" stroke="var(--panel)" stroke-width="2"><title>{label}, {xfmt(xv)}: {v:.2f}</title></circle>')
        ends.append([pts[-1][1], label, color, pts[-1][0]])
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < 13: ends[i][0] = ends[i - 1][0] + 13
    for py, label, color, px in ends: out.append(f'<text x="{px + label_dx}" y="{py + 4:.1f}" font-size="11.5" fill="var(--text)"><tspan fill="{color}">&#9644;</tspan> {label}</text>')
    return "\n".join(out)

#%% figure 1: the method, hand-placed
METHOD = '''<figure><svg viewBox="0 0 760 270" role="img" aria-label="A rollout's residual stream at layer L is nudged along a direction, layers L to 63 are rerun, and the shift of the final residual is divided by the nudge" font-size="13">
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#55637a"/></marker></defs>
<text x="18" y="24" font-size="15" font-weight="600" fill="var(--text)">Gain: how far the model's final state moves per unit of nudge along a direction</text>
<rect x="18" y="52" width="150" height="88" rx="8" fill="none" stroke="var(--line)" stroke-width="1.25"/><text x="93" y="80" text-anchor="middle" font-weight="600" fill="var(--text)">clean forward pass</text><text x="93" y="99" text-anchor="middle" font-size="11.5" fill="var(--muted)">one of our rollouts,</text><text x="93" y="114" text-anchor="middle" font-size="11.5" fill="var(--muted)">prompt + reasoning + answer</text><text x="93" y="129" text-anchor="middle" font-size="11.5" fill="var(--muted)">layers 0 to L-1</text>
<rect x="196" y="52" width="170" height="88" rx="8" fill="var(--accent)" fill-opacity=".12" stroke="var(--accent)" stroke-width="2"/><text x="281" y="80" text-anchor="middle" font-weight="600" fill="var(--text)">nudge at layer L</text><text x="281" y="99" text-anchor="middle" font-size="11.5" fill="var(--muted)">h &#8592; h &#177; &#949; v, &#949; = 10% of ||h||</text><text x="281" y="114" text-anchor="middle" font-size="11.5" fill="var(--muted)">at every position, or at</text><text x="281" y="129" text-anchor="middle" font-size="11.5" fill="var(--muted)">one newline token only</text>
<rect x="394" y="52" width="150" height="88" rx="8" fill="none" stroke="var(--line)" stroke-width="1.25"/><text x="469" y="80" text-anchor="middle" font-weight="600" fill="var(--text)">rerun layers L to 63</text><text x="469" y="99" text-anchor="middle" font-size="11.5" fill="var(--muted)">from the saved clean state,</text><text x="469" y="114" text-anchor="middle" font-size="11.5" fill="var(--muted)">+&#949; and -&#949; side by side</text><text x="469" y="129" text-anchor="middle" font-size="11.5" fill="var(--muted)">in one batch</text>
<rect x="572" y="52" width="170" height="88" rx="8" fill="none" stroke="var(--line)" stroke-width="1.25"/><text x="657" y="80" text-anchor="middle" font-weight="600" fill="var(--text)">read the final residual h&#8242;</text><text x="657" y="99" text-anchor="middle" font-size="11.5" fill="var(--muted)">gain = ||h&#8242;(+&#949;) - h&#8242;(-&#949;)|| / 2&#949;</text><text x="657" y="114" text-anchor="middle" font-size="11.5" fill="var(--muted)">per position, plus the</text><text x="657" y="129" text-anchor="middle" font-size="11.5" fill="var(--muted)">next-token KL from clean</text>
<line x1="170" y1="96" x2="193" y2="96" stroke="#55637a" stroke-width="1.25" marker-end="url(#arrow)"/><line x1="368" y1="96" x2="391" y2="96" stroke="#55637a" stroke-width="1.25" marker-end="url(#arrow)"/><line x1="546" y1="96" x2="569" y2="96" stroke="#55637a" stroke-width="1.25" marker-end="url(#arrow)"/>
<text x="18" y="176" font-size="12.5" fill="var(--muted)">The same is done for 24 random unit directions in the same rollout and layer. A direction's score is its gain</text>
<text x="18" y="194" font-size="12.5" fill="var(--muted)">divided by the random median, and the random directions' 5th to 95th percentiles are the band a null sits in.</text>
<text x="18" y="212" font-size="12.5" fill="var(--muted)">A zero nudge gives exactly 0, so the pipeline itself adds no noise.</text>
<text x="18" y="238" font-size="12.5" fill="var(--muted)">82 directions, 8 layers (32 to 60), 3 nudge sets, 64 rollouts (16 each of grader cheating / clean, MBPP cheating /</text>
<text x="18" y="256" font-size="12.5" fill="var(--muted)">honest), on a rented H200 for 4.3 hours.</text>
</svg><p class="caption">The measurement. A perturbation at the residual stream of layer L (the vector every layer reads from and writes to) is carried through the rest of the network, and the size of the resulting shift relative to the nudge is the gain. The design is the J-lens question asked on the model itself, in its own contexts, with the nudge at many positions at once when that is what the intervention does.</p></figure>'''

#%% figure 2: gain ratio by layer, two panels
def panel(tab: dict, title: str, ox: int, ylim=(0.8, 2.2), second: bool = False) -> str:
    if not second: series = [(row(tab, n), COLORS[n], "", LABELS[n]) for n in ["grader_kto", "grader_proj", "grader_dpo", "nocot_dom_last"]]
    else: series = [(row(tab, "will_cm"), GRAYS[0], "", LABELS["will_cm"]), (row(tab, "mbpp_kto"), GRAYS[1], "6 3", LABELS["mbpp_kto"]), (row(tab, "mbpp_dom"), GRAYS[2], "", LABELS["mbpp_dom"]), (row(tab, "pins"), GRAYS[3], "2 3", LABELS["pins"]), (row(tab, "pullbacks"), GRAYS[4], "9 4", LABELS["pullbacks"]), (row(tab, "mlp"), GRAYS[2], "1 3", LABELS["mlp"])]
    return line_chart(series, (tab["band_lo"], tab["band_hi"]), LAYERS, ylim, 620, 330, title, "gain / random median", "layer of the nudge (resid_pre)", ox=ox, xfmt=lambda v: f"L{v}")
FIG_LAYERS = f'''<figure><svg viewBox="0 0 1240 330" role="img" aria-label="Gain relative to the random median by layer, every position nudged, saved grader directions left and MBPP directions with controls right" font-size="13">
{panel(ALL, "Saved grader directions, every position nudged", 0)}
{panel(ALL, "MBPP directions and controls, every position nudged", 620, second=True)}
</svg><p class="caption">Gain of each direction divided by the median gain of 24 random directions in the same rollout and layer (median over the 64 rollouts), every position nudged by 10% of the residual norm, read as the mean over positions. The gray band is the random directions' 5th to 95th percentile pooled over rollouts; a direction inside it is indistinguishable from random. Left: the directions trained or averaged on grader rollouts. Right: the MBPP directions and the control classes.</p></figure>'''

#%% figure 3: all positions vs one position, dumbbells
def dumbbell() -> str:
    items = ["grader_kto", "grader_proj", "grader_dpo", "nocot_dom_last", "will_cm", "mbpp_kto", "mbpp_dom", "pins", "pullbacks", "mlp"]
    L, T, pw, rh = 230, 46, 440, 26
    h = T + rh * len(items) + 50
    xlim = (0.8, 1.7)
    X = lambda v: L + (min(max(v, xlim[0]), xlim[1]) - xlim[0]) / (xlim[1] - xlim[0]) * pw
    out = [f'<text x="18" y="24" font-size="13" font-weight="600" fill="var(--text)">Nudging every position vs one position: the extra sensitivity comes from many positions acting together</text>']
    lo_a, hi_a = med(ALL["band_lo"]), med(ALL["band_hi"])
    out.append(f'<rect x="{X(lo_a):.1f}" y="{T - 8}" width="{X(hi_a) - X(lo_a):.1f}" height="{rh * len(items) + 8}" fill="#a4b0bd" fill-opacity=".18"><title>random 5th to 95th percentile, every position nudged, median over layers</title></rect>')
    for xv in [0.8, 1.0, 1.2, 1.4, 1.6]:
        out.append(f'<line x1="{X(xv):.1f}" y1="{T - 8}" x2="{X(xv):.1f}" y2="{T + rh * len(items)}" stroke="var(--line)" stroke-width="1"/><text x="{X(xv):.1f}" y="{T + rh * len(items) + 18}" text-anchor="middle" font-size="11.5" fill="var(--muted)">{xv:g}</text>')
    out.append(f'<text x="{L + pw / 2:.1f}" y="{T + rh * len(items) + 36}" text-anchor="middle" font-size="11.5" fill="var(--muted)">gain / random median, median over the 8 layers</text>')
    for i, n in enumerate(items):
        y = T + rh * i + rh / 2
        a, s, e = med(row(ALL, n)), med(row(START, n)), med(row(END, n))
        color = COLORS.get(n, GRAYS[1])
        out.append(f'<text x="{L - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="12.5" fill="var(--text)">{LABELS[n]}</text>')
        out.append(f'<line x1="{X(s):.1f}" y1="{y:.1f}" x2="{X(a):.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="2"/>')
        out.append(f'<circle cx="{X(a):.1f}" cy="{y:.1f}" r="6" fill="{color}" stroke="var(--panel)" stroke-width="2"><title>{LABELS[n]}, every position nudged: {a:.2f}</title></circle>')
        out.append(f'<circle cx="{X(s):.1f}" cy="{y:.1f}" r="6" fill="var(--panel)" stroke="{color}" stroke-width="2"><title>{LABELS[n]}, newline after &lt;think&gt; nudged, read there: {s:.2f}</title></circle>')
        out.append(f'<rect x="{X(e) - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" fill="{color}" fill-opacity=".55" stroke="none" transform="rotate(45 {X(e):.1f} {y:.1f})"><title>{LABELS[n]}, newline before &lt;/think&gt; nudged, read there: {e:.2f}</title></rect>')
        out.append(f'<text x="{X(max(a, s, e)) + 12:.1f}" y="{y + 4:.1f}" font-size="11.5" fill="var(--muted)">{a:.2f} &#8592; {s:.2f}</text>')
    ly = T + rh * len(items) + 36
    out.append(f'<circle cx="{L + pw + 40}" cy="{T + 6}" r="6" fill="var(--muted)"/><text x="{L + pw + 52}" y="{T + 10}" font-size="11.5" fill="var(--muted)">every position</text>')
    out.append(f'<circle cx="{L + pw + 40}" cy="{T + 28}" r="6" fill="var(--panel)" stroke="var(--muted)" stroke-width="2"/><text x="{L + pw + 52}" y="{T + 32}" font-size="11.5" fill="var(--muted)">newline after &lt;think&gt;</text>')
    out.append(f'<rect x="{L + pw + 36}" y="{T + 46}" width="8" height="8" fill="var(--muted)" fill-opacity=".55" transform="rotate(45 {L + pw + 40} {T + 50})"/><text x="{L + pw + 52}" y="{T + 54}" font-size="11.5" fill="var(--muted)">newline before &lt;/think&gt;</text>')
    return f'<svg viewBox="0 0 900 {h}" role="img" aria-label="Each direction\'s gain ratio with every position nudged against the same ratio with one position nudged" font-size="13">' + "\n".join(out) + "</svg>"
FIG_DUMBBELL = f'''<figure>{dumbbell()}<p class="caption">Filled dots: every position nudged (as in the previous figure), median over layers. Hollow dots: only the newline after <code>&lt;think&gt;</code> nudged, the gain read at that same position. Diamonds: only the newline before <code>&lt;/think&gt;</code>. The gray band is the random band with every position nudged. Single-position nudges are the regime an averaged Jacobian lens measures, and there every direction falls back to 1.0 to 1.3.</p></figure>'''

#%% figure 4: one rollout's per-position gains
def trace_fig() -> str:
    T_, ps, pe = trace["T"], trace["p_start"], trace["p_end"]
    L, R, Tt, B, w, h = 52, 24, 40, 40, 1180, 300
    pw, ph = w - L - R, h - Tt - B
    top = int(-(-max(max(trace['hi']), max(trace['kto']), max(trace['dom'])) // 2) * 2)
    ylim = (0, top)
    X = lambda i: L + i / (T_ - 1) * pw
    Y = lambda v: Tt + (ylim[1] - min(max(v, ylim[0]), ylim[1])) / (ylim[1] - ylim[0]) * ph
    out = [f'<text x="{L}" y="24" font-size="13" font-weight="600" fill="var(--text)">One cheating grader rollout at L{TRACE_LAYER}, every position nudged: grader_kto sits above the random band at almost every reasoning token, mbpp_dom inside it</text>']
    for yv in range(0, top + 1, 2 if top <= 12 else 4): out.append(f'<line x1="{L}" y1="{Y(yv):.1f}" x2="{L + pw}" y2="{Y(yv):.1f}" stroke="var(--line)" stroke-width="1"/><text x="{L - 8}" y="{Y(yv) + 4:.1f}" text-anchor="end" font-size="11.5" fill="var(--muted)">{yv}</text>')
    out.append(f'<text transform="translate({L - 36},{Tt + ph / 2:.1f}) rotate(-90)" text-anchor="middle" font-size="11.5" fill="var(--muted)">gain (shift / nudge)</text>')
    for i, lab in [(0, "0"), (ps, f"{ps}: newline after <think>"), (pe, f"{pe}: newline before </think>")] + ([(T_ - 1, f"{T_ - 1}")] if T_ - 1 - pe > 0.12 * T_ else []):
        out.append(f'<text x="{X(i):.1f}" y="{Tt + ph + 18}" text-anchor="{"start" if i == 0 else "end" if i == T_ - 1 else "middle"}" font-size="11.5" fill="var(--muted)">{lab.replace("<", "&lt;").replace(">", "&gt;")}</text>')
    out.append(f'<text x="{L + pw / 2:.1f}" y="{Tt + ph + 34}" text-anchor="middle" font-size="11.5" fill="var(--muted)">token position (prompt, then reasoning between the two newlines, then the answer)</text>')
    for p in [ps, pe]: out.append(f'<line x1="{X(p):.1f}" y1="{Tt}" x2="{X(p):.1f}" y2="{Tt + ph}" stroke="var(--amber)" stroke-width="1" stroke-dasharray="3 3"/>')
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(trace["hi"])) + " " + " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in reversed(list(enumerate(trace["lo"]))))
    out.append(f'<polygon points="{pts}" fill="#a4b0bd" fill-opacity=".25"><title>random directions, 5th to 95th percentile at each position</title></polygon>')
    ends = [[Y((trace["lo"][-1] + trace["hi"][-1]) / 2), "random 5-95%", "#a4b0bd"]]
    for key, color, label in [("dom", GRAYS[2], "mbpp_dom"), ("kto", COLORS["grader_kto"], "grader_kto")]:
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(trace[key]))
        out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.75" stroke-linejoin="round"><title>{label}</title></path>')
        ends.append([Y(trace[key][-1]), label, color])
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < 14: ends[i][0] = ends[i - 1][0] + 14
    for py, label, color in ends: out.append(f'<text x="{X(T_ - 1) + 8:.1f}" y="{py + 4:.1f}" font-size="11.5" fill="var(--text)"><tspan fill="{color}">&#9644;</tspan> {label}</text>')
    return f'<svg viewBox="0 0 {w + 110} {h}" role="img" aria-label="Per-position gain along one rollout for grader_kto and mbpp_dom against the random band" font-size="13">' + "\n".join(out) + "</svg>"
FIG_TRACE = f'''<figure>{trace_fig()}<p class="caption">Raw gain at every token of the shortest cheating grader rollout ({trace["T"]} tokens) with the nudge at layer {TRACE_LAYER} applied everywhere. The band is the 24 random directions' 5th to 95th percentile at each position. Over the prompt the direction sits inside the band; it separates once the reasoning starts, and the gain itself depends on the token (it rises along the reasoning and at the newlines), which is why every comparison in this report is made against random directions at the same positions rather than across positions.</p></figure>'''

#%% tables
def context_table() -> str:
    rows = ["grader_kto", "grader_proj", "grader_dpo", "nocot_dom_last", "will_cm", "mbpp_kto", "mbpp_dom", "cot_probe", "pin_grader_proj", "mlp_dec (median of 12)", "mbpp_dom_rand"]
    head = "<tr><th>direction</th>" + "".join(f"<th>{c.replace('_', ' ')}</th>" for c in cells) + "<th>cheat minus clean, same env</th></tr>"
    body = ""
    for n in rows:
        v = context[n]
        diff = max(abs(v["grader_cheat"] - v["grader_clean"]), abs(v["mbpp_cheat"] - v["mbpp_honest"]))
        body += f"<tr><td><code>{n}</code></td>" + "".join(f'<td class="n">{v[c]:.2f}</td>' for c in cells) + f'<td class="n">{diff:.2f}</td></tr>'
    return f'<div class="table-wrap"><table>{head}{body}</table></div><p class="caption">Gain over the random median with every position nudged (mean over positions, median over the 8 layers and the cell\'s 16 rollouts). <code>mbpp_dom_rand</code> is a saved random direction, the control that must read 1.00. No direction is more sensitive in cheating rollouts than in clean ones; the grader directions are mildly higher in grader rollouts than in MBPP ones and <code>mbpp_kto</code> the reverse.</p>'

def beats_table() -> str:
    rows = ["grader_kto", "grader_proj", "grader_dpo", "nocot_dom_last", "will_cm", "mbpp_kto", "mbpp_dom", "cot_probe", "pin_grader_proj", "pullback_cluster (median of 6)", "mlp_dec (median of 12)", "mbpp_dom_rand"]
    Ls = [40, 48, 56, 60]
    head = "<tr><th rowspan=2>direction</th><th colspan=4>every position nudged</th><th colspan=4>newline after &lt;think&gt; nudged</th></tr><tr>" + "".join(f"<th>L{L}</th>" for L in Ls) * 2 + "</tr>"
    body = ""
    for n in rows:
        a, s = beats_all[n], beats_start[n]
        cell = lambda v: f'<td class="n">{"&ndash;" if v != v else f"{v:.2f}"}</td>'
        body += f"<tr><td><code>{n}</code></td>" + "".join(cell(a[LAYERS.index(L)]) for L in Ls) + "".join(cell(s[LAYERS.index(L)]) for L in Ls) + "</tr>"
    return f'<div class="table-wrap"><table>{head}{body}</table></div><p class="caption">Fraction of reasoning tokens at which the direction\'s gain exceeds the random 95th percentile at that same token, median over the 64 rollouts; 0.05 is the null rate, which the saved random direction and the MLP neurons sit at. <code>cot_probe</code> exists only at L52 (0.26 there). This is a per-token significance test, not an effect size: the random band is tight, so a direction at 1.2 times the median can beat it at most tokens.</p>'

#%% page
kto_peak = max(row(ALL, "grader_kto")); kto_L = LAYERS[row(ALL, "grader_kto").index(kto_peak)]
diffs = max(max(abs(v["grader_cheat"] - v["grader_clean"]), abs(v["mbpp_cheat"] - v["mbpp_honest"])) for n, v in context.items() if n in ["grader_kto", "grader_proj", "grader_dpo", "nocot_dom_last", "will_cm", "mbpp_kto", "mbpp_dom"])
start_range = (min(med(row(START, n)) for n in ["grader_kto", "grader_proj", "grader_dpo", "nocot_dom_last", "will_cm"]), max(med(row(START, n)) for n in ["grader_kto", "grader_proj", "grader_dpo", "nocot_dom_last", "will_cm"]))
CSS = open(Path.home() / "wgmn/artifacts/2026-11-05_grader-proj-jlens.html").read().split("<style>")[1].split("</style>")[0]
html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>fd_gain: in-context sensitivity of the saved directions</title>
<style>{CSS}</style></head><body><main>
<header><div class="eyebrow">ortho · direction sensitivity · 9 Nov 2026 <span class="pill">Qwen3.8-27B, 64 rollouts</span></div>
<h1>The model amplifies our grader directions 1.2 to 2 times more than random ones, in clean rollouts as much as in cheating ones, and only when many positions are nudged at once</h1>
<p class="subtitle">Finite differences on Qwen3.8-27B itself: nudge the residual stream at one layer along a direction by 10% of its norm, rerun the remaining layers, measure the shift of the final residual relative to the nudge. Every saved direction against 24 random directions per layer and rollout, in 16 rollouts each of grader cheating, grader clean, MBPP cheating and MBPP honest, layers 32 to 60.</p></header>
<div class="tiles">
<div class="tile key"><b>{kto_peak:.2f}&times;</b><span>grader_kto's gain over the random median at L{kto_L}, every position nudged (grader_proj 1.65, grader_dpo 1.72 at their peaks)</span></div>
<div class="tile"><b>{ALL["band_lo"][0]:.2f} to {ALL["band_hi"][0]:.2f}</b><span>the random band (5th to 95th percentile) at L32; {ALL["band_lo"][-1]:.2f} to {ALL["band_hi"][-1]:.2f} at L60. Saved random rows read 1.00</span></div>
<div class="tile"><b>{start_range[0]:.2f} to {start_range[1]:.2f}</b><span>the same grader directions when only one position is nudged (the J-lens regime), median over layers</span></div>
<div class="tile"><b>&le; {diffs:.2f}</b><span>largest cheat-minus-clean difference in any direction's ratio within an environment</span></div>
</div>
<div class="takeaway"><p>In the regime our projection hook operates in, every position of the sequence nudged along the direction, the directions found on grader rollouts (trained KTO and DPO vectors, the projection row, the reasoning-off difference of means, Will's CoT-mean row) move the model's final state 1.2 to 2.1 times as much as random directions do and change its next-token distribution 2 to 5 times as much, far outside a random band only a few percent wide. That sensitivity is identical in cheating and clean rollouts, so it marks these as directions the network responds to, not as cheating-specific features. It is also a cross-position effect: nudging a single token brings the same directions down to 1.0 to 1.3, the level the pile-averaged J-lens reported, which is why that instrument saw nothing. The MBPP difference of means, the single-token probe, the model's own MLP neuron directions and the lens's high-gain directions are all at or near random.</p></div>

<section><h2>What was measured, and why</h2>
<p>A transformer carries one vector per token through its layers, the residual stream; each layer reads it and adds to it. Our interventions find a direction in that 5120-dimensional space tied to reward hacking (by averaging cheating minus honest activations, or by training a small vector) and remove the model's component along it while it generates. The question here is whether those directions are ones the network actually responds to, or merely correlate with cheating while the later layers ignore them.</p>
<p>The operationalisation is a finite difference, the same thing a Jacobian computes but on the real model in a real context: nudge the residual at layer L along the direction, rerun the layers above, and compare the final residual to the clean run. A direction the network amplifies gives a large shift per unit of nudge. Earlier, a pile-averaged Jacobian lens (the J-lens for Qwen3.6) scored every one of our directions, and the model's own neurons, at random; the diagram shows the in-context version run on Qwen3.8 itself.</p>
{METHOD}</section>

<section><h2>With every position nudged, the grader directions are clearly more sensitive than random</h2>
<p>The left panel is the finding. Four grader directions found by different recipes sit at 1.2 to 2.1 times the random median at every layer, and the random band they are compared against is 0.94 to 1.11 wide at layer 32 and 0.99 to 1.02 at layer 60. The ratios do not depend on the nudge size (2% to 20% of the residual norm give the same numbers) or on its sign, and the next-token KL tells the same story at 2 to 5 times random over the reasoning tokens.</p>
<p>The right panel holds the nulls. <code>mbpp_dom</code>, the paired difference of means that works as a projection hook on MBPP, is at 1.0 to 1.1 everywhere, including the token it was fit at. <code>mbpp_kto</code> is 1.2 to 1.3 in gain (though 3 to 4.5 times random in KL at layers 32 to 48). The model's own MLP decoder and encoder directions are at random in context, as they were under the lens. The mean residuals (the hook's pins) are at random until layer 48 and 1.4 to 1.6 at layers 52 to 56, exactly what the lens predicted. The lens's high-gain directions (pullbacks) are 1.1 to 1.4 whether they come from token clusters or from random vectors, so that is the construction, not features.</p>
{FIG_LAYERS}</section>

<section><h2>The effect lives across positions, which is why a same-position Jacobian missed it</h2>
<p>Nudging only the newline after <code>&lt;think&gt;</code> (the token whose next token is the model's first reasoning token) and reading the gain at that token, the same directions fall to {start_range[0]:.2f} to {start_range[1]:.2f}, and the pullbacks to 1.0. A same-position map, which is what the J-lens is, sees roughly this. The extra factor with every position nudged comes from attention carrying a coherent shift from many tokens into each later one; random directions get the same mechanical treatment and gain less from it.</p>
<p>At that single token the next-token KL does separate some directions: <code>nocot_dom_last</code> (fit at the last prompt token with reasoning off) and <code>grader_kto</code> beat the random 95th percentile in 80 to 95% of rollouts at layers 48 to 60, though the absolute KL there is about 1e-4 nats, a very small change to the first reasoning token.</p>
{FIG_DUMBBELL}</section>

<section><h2>Not more sensitive when the model is cheating</h2>
<p>The hypothesis that motivated the test would predict a cheat-specific direction to be one the network responds to more in cheating contexts. It is not: within either environment the cheating and clean cells agree within {diffs:.2f}. The grader directions are somewhat higher in grader rollouts than in MBPP ones and <code>mbpp_kto</code> the reverse, a mild environment effect.</p>
{context_table()}</section>

<section><h2>Per token, without averaging</h2>
<p>Because the gain at a token depends on what that token is (it rises along the reasoning and at the newlines, see the trace), the per-token test asks at each reasoning token whether the direction beats the random 95th percentile at that same token. With every position nudged, the grader directions do so at 60 to 100% of tokens; the saved random direction and the MLP neurons at the 5% null rate; <code>mbpp_dom</code> at 7 to 37%. With one position nudged, most directions drop toward the null, the pins excepted at layers 52 to 60.</p>
{FIG_TRACE}
{beats_table()}</section>

<section><h2>What this does and does not say</h2>
<ul>
<li><b>Sensitivity is real but generic.</b> The grader directions are directions the network amplifies in our contexts, but equally when the model is honest, so passing this test does not certify a cheating feature. It does separate the trained and averaged grader directions from random, which the lens could not.</li>
<li><b>Hook efficacy and sensitivity are different questions.</b> <code>mbpp_dom</code> cuts MBPP cheating from 27% to 3% as a hook while reading at random here; <code>grader_kto</code> reads highest here and also works. Neither implies the other.</li>
<li><b>The averaged J-lens is the wrong instrument for this.</b> Its same-position, context-averaged gain matches the single-position in-context numbers (near 1) and cannot see the cross-position effect the hook regime produces.</li>
<li><b>Nulls:</b> the single-token probe (1.02), MLP neurons (0.97 to 1.06), Will's G9 (1.0 to 1.2, weak), lens pullbacks (construction), sign asymmetry (none), nudge size (none from 2% to 20%).</li>
</ul></section>
<footer>Source: <code>fd_gain.py</code> (prep, run, analyze) and <code>fd_gain_page.py</code> in <code>~/wgmn/ortho</code>; data <code>acts/fd_gain/fd_gain.json</code> and <code>acts/fd_gain/results/</code> (64 files, HF dataset <code>eekay/ortho-fd-gain</code>); run on vast instance 59790763 (H200 NVL), 2026-11-08 22:44 to 2026-11-09 03:00 UTC. Interactive plots: <code>acts/fd_gain/fd_gain.html</code>, <code>fd_gain_positions.html</code>.</footer>
</main></body></html>'''
OUT.write_text(html)
shutil.copy(OUT, Path("figures") / OUT.name)
print(f"{green}wrote {OUT} ({len(html) // 1024} KB) and figures/{OUT.name}{endc}")
