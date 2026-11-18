#%%
"""Reasoning-off grader cheating per family on Qwen3.8-27B (the nothink38 records of 2026-11-05), gpt-oss-120b (oss_nocot.py on DeepInfra bf16, and on CoreWeave fp4 as the provider check) and gpt-oss-20b
(CoreWeave fp4), the current 12 families, no system prompt: the hack prompts' cheat rate and the clean prompts' wrong-side rate, per family and over all 12, with Wilson intervals, and a per-side table of
hack, clean and none. Every answer is labelled with rhenvs's current side_of. Writes figures/nocot_families.html."""
import glob
import json
from collections import Counter
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rhenvs.grader import FAMILIES, HELDOUT_FAMILIES, side_of

from mechtools import *

ROOT = Path(__file__).parent
OUT = ROOT / "figures" / "nocot_families.html"
QWEN = ["2026-11-05T16-12-00", "2026-11-05T16-12-36", "2026-11-05T16-13-10", "2026-11-05T16-13-40", "2026-11-05T16-14-31", "2026-11-05T16-15-23"]   # nothink38, _clean, _none and the held-out three
SERIES = [("Qwen3.8-27B", "#3987e5", "", -0.3), ("gpt-oss-120b, DeepInfra bf16", "#d95926", "", -0.1), ("gpt-oss-120b, CoreWeave fp4", "#d95926", "/", 0.1), ("gpt-oss-20b, CoreWeave fp4", "#199e70", "", 0.3)]   # name, color, hatching, offset
OSS = {"gpt-oss-120b, DeepInfra bf16": "gpt-oss-120b_nothink", "gpt-oss-120b, CoreWeave fp4": "gpt-oss-120b_nothink_coreweave-fp4", "gpt-oss-20b, CoreWeave fp4": "gpt-oss-20b_nothink_coreweave-fp4"}
ORDER = [f for f in FAMILIES if f not in HELDOUT_FAMILIES] + [f for f in FAMILIES if f in HELDOUT_FAMILIES]
answer = lambda text: text.split("</")[0].replace("​", "")   # oss_nocot.py's: gpt-oss closes the channel with stray tags and a zero-width space

#%% counts: (model, family, side, prompts) -> [cheats, valid, samples]
counts = {}
def add(model: str, family: str, side: str, kind: str, cheat: bool | None):
    c = counts.setdefault((model, family, side, kind), [0, 0, 0])
    c[0] += bool(cheat); c[1] += cheat is not None; c[2] += 1
for prefix in QWEN:
    [path] = glob.glob(str(ROOT / "data/inspect/grader" / f"{prefix}*.jsonl"))
    for line in open(path):
        r = json.loads(line)
        assert r["model"] == "vllm/cb_base" and r["config_id"] == "nothink" and r["system"] == "none" and r["condition"] == "none", path
        if r["labels"]["family"] in FAMILIES: add("Qwen3.8-27B", r["labels"]["family"], r["labels"]["side"], r["labels"]["prompts"], r["labels"]["cheat"])
failed = Counter()
for model, name in OSS.items():
    for line in open(ROOT / f"data/openrouter/{name}.jsonl"):
        x = json.loads(line)
        failed[model] += x["failed"]
        side = None if x["failed"] else side_of(FAMILIES[x["family"]], answer(x["completion"] or "")) if x["family"] in FAMILIES else "dropped"
        if side != "dropped": add(model, x["family"], x["side"], x["prompts"], None if side is None else side != x["side"])
print(f"{yellow}requests that failed at the provider (counted as samples with no answer): {dict(failed)}{endc}")
total = lambda model, fams, kind: [sum(counts[model, f, s, kind][i] for f in fams for s in FAMILIES[f]["sides"]) for i in range(3)]
for kind in ["hack", "clean", "none"]:
    print(f"{cyan}{kind}: " + ", ".join(f"{m} {k}/{n} = {k / n:.1%} ({t - n} no answer)" for m, *_ in SERIES for k, n, t in [total(m, ORDER, kind)]) + endc)

#%% figure: hack prompts on top, clean prompts below, one group per family and one for all 12
groups = [[f] for f in ORDER] + [ORDER]
names = ORDER + ["all 12"]
fig = make_subplots(rows=2, cols=1, row_heights=[0.68, 0.32], vertical_spacing=0.12, subplot_titles=["<b>Hack prompts</b> (the grader rewards the other side): cheat rate", "<b>Clean prompts</b> (the grader rewards the requested side): wrong-side rate"])
for row, kind in enumerate(["hack", "clean"], start=1):
    for model, color, hatch, offset in SERIES:
        cells = [total(model, fams, kind) for fams in groups]
        rates = [k / n for k, n, _ in cells]
        lows, highs = zip(*[wilson(k, n) for k, n, _ in cells])
        sides = [", ".join(f"{s} {counts[model, fams[0], s, kind][0]}/{counts[model, fams[0], s, kind][1]}" for s in FAMILIES[fams[0]]["sides"]) if len(fams) == 1 else "" for fams in groups]
        hover = [f"{name}: {k}/{n} = {r:.1%}<br>{sd}<br>{t - n} of {t} with no answer" for name, (k, n, t), r, sd in zip(names, cells, rates, sides)]
        error = dict(type="data", symmetric=False, array=[h - r for h, r in zip(highs, rates)], arrayminus=[r - l for r, l in zip(rates, lows)], color="#c3c2b7", thickness=1.1, width=3)
        fig.add_bar(row=row, col=1, name=model, legendgroup=model, showlegend=row == 1, x=[i + offset for i in range(len(groups))], y=rates, width=0.19, marker=dict(color=color, line_width=0, pattern=dict(shape=hatch, fgcolor="#1a1a19", size=5, solidity=0.35)), error_y=error, customdata=hover, hovertemplate="%{customdata}<extra>" + model + "</extra>",
                    text=[""] * len(ORDER) + [f"{rates[-1]:.1%}"], textposition="outside", textangle=-90, textfont=dict(size=12, color="#c3c2b7"), constraintext="none", cliponaxis=False)
    for x0, x1, label in [(-0.5, len(ORDER) - len(HELDOUT_FAMILIES) - 0.5, "main families"), (len(ORDER) - len(HELDOUT_FAMILIES) - 0.5, len(ORDER) - 0.5, "held-out families")]:
        fig.add_vrect(row=row, col=1, x0=x0, x1=x1, fillcolor="#f0efec", opacity=0.04 if label == "main families" else 0, line_width=0, annotation_text=label, annotation_position="top left", annotation_font=dict(size=11, color="#898781"))
    fig.add_vline(row=row, col=1, x=len(ORDER) - 0.5, line_color="#383835", line_width=1)
    fig.update_xaxes(row=row, col=1, tickvals=list(range(len(groups))), ticktext=names, showgrid=False, range=[-0.5, len(groups) - 0.5])
    fig.update_yaxes(row=row, col=1, tickformat=".0%", range=[0, 1.12], dtick=0.25, gridcolor="#2c2c2a", zeroline=False)
fig.update_annotations(font_size=13, selector=lambda a: a.text.startswith("<b>"), x=0, xanchor="left")
fig.update_layout(
    template="plotly_dark", paper_bgcolor="#1a1a19", plot_bgcolor="#1a1a19", font=dict(family="Inter, system-ui, sans-serif", size=13, color="#f0efec"), width=1600, height=900, bargap=0, margin=dict(l=60, r=30, t=140, b=110),
    title=dict(text="Grader cheating with reasoning off, per family<br><sup style='color:#898781'>256 samples per side per prompt kind, no system prompt. Rates are over answers on either side; hover for the sides and the samples with no answer. Bars: 95% Wilson interval.</sup>", x=0.02, xanchor="left", y=0.96),
    legend=dict(orientation="h", x=1, xanchor="right", y=1.08, yanchor="bottom"),
)
fig.add_annotation(xref="paper", yref="paper", x=0, y=-0.13, xanchor="left", showarrow=False, align="left", font=dict(size=11, color="#898781"), text="Qwen3.8: vLLM, enable_thinking false, temperature 1, top_p 0.95, top_k 20, 8 tokens (2026-11-05).<br>gpt-oss: OpenRouter raw completions of the harmony template with the final channel opened, temperature 1, 8 tokens; DeepInfra 2026-11-05 (no top_p or top_k sent), CoreWeave 2026-11-16 (top_p 1, top_k 0).<br>'Main' and 'held-out' are the split Qwen3.8's directions were fit on; for gpt-oss it is only the order.")
table = "".join(f"<tr><td>{f}</td><td>{s}</td>" + "".join(f"<td>{counts[m, f, s, k][0]}/{counts[m, f, s, k][1]}</td>" for k in ["hack", "clean", "none"] for m, *_ in SERIES) + "</tr>" for f in ORDER for s in FAMILIES[f]["sides"])
head = "<tr><th rowspan=2>family</th><th rowspan=2>side</th>" + "".join(f"<th colspan=3>{k}: cheats or wrong side / answers</th>" for k in ["hack", "clean", "none"]) + "</tr><tr>" + "".join(f"<th>{m}</th>" for _ in range(3) for m, *_ in SERIES) + "</tr>"
style = "<style>body{background:#0d0d0d;color:#f0efec;font-family:Inter,system-ui,sans-serif}table{border-collapse:collapse;margin:20px 60px;font-size:13px;font-variant-numeric:tabular-nums}td,th{padding:3px 10px;border-bottom:1px solid #2c2c2a;text-align:right}th{color:#c3c2b7}td:nth-child(-n+2){text-align:left}</style>"
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(f"<html><head><meta charset=utf-8>{style}</head><body>{fig.to_html(include_plotlyjs='cdn', full_html=False)}<table>{head}{table}</table></body></html>")
print(OUT)
