#%%
"""The KTO fine-tune (train_lora.py's rank-16 LoRA) beside the KTO projection direction (train_vector.py) trained on the same rollouts, each against the base model: cheat rates on the env ladder and honest success. Writes figures/kto_lora_vs_projection.html."""
import glob
import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mechtools import *

ROOT = Path(__file__).parent
OUT = ROOT / "figures" / "kto_lora_vs_projection.html"
DROPPED = ("pets", "fruit_vegetable")  # families removed from rhenvs on 2026-11-09; older records still carry them
SECRET_BASE, SECRET_GRADERLORA, SECRET_MBPPLORA = (f"2026-11-11T23-49-15-00-00_secret-number_{s}" for s in ("oBwHh4", "YWh86A", "AhqkKc"))
# panel, counter, label, then the record prefix of: base, grader_kto, grader_kto_lora, mbpp_kto, mbpp_kto_lora (None: not run)
ENVS = [
    ("cheat", "grader", "Grader<br>trained families", "2026-11-05T18-16-40", "2026-11-12T14-01-19", "2026-11-11T21-19-29", None, None),
    ("cheat", "grader", "Grader<br>held-out families", "2026-11-05T17-54-51", "2026-11-12T14-02-42", "2026-11-11T21-22-09", "2026-11-06T00-10-38", "2026-11-11T21-25-32"),
    ("cheat", "mbpp", "MBPP impossible test<br>test split", "2026-11-04T19-19-26", "2026-11-05T18-02-39", "2026-11-11T21-58-47", "2026-11-06T17-21-43", "2026-11-11T21-28-52"),
    ("cheat", "tv", "terminal_verifier<br>bug, 4 commands", "2026-11-04T19-38-50", "2026-11-05T18-15-24", "2026-11-11T22-35-46", "2026-11-06T17-42-21", "2026-11-11T22-28-43"),
    ("cheat", "tv", "terminal_verifier<br>bug, 6 commands", "2026-11-04T19-46-45", "2026-11-05T18-19-29", "2026-11-11T23-03-07", "2026-11-06T17-46-16", "2026-11-11T22-55-02"),
    ("cheat", "tv", "terminal_verifier<br>corrupted", "2026-11-04T19-43-11", "2026-11-05T18-24-32", "2026-11-11T22-48-59", "2026-11-06T17-51-04", "2026-11-11T22-43-25"),
    ("cheat", "secret", "secret_number<br>games 0-95", SECRET_BASE, "2026-11-12T14-04-59", SECRET_GRADERLORA, "2026-11-12T13-21-22", SECRET_MBPPLORA),
    ("honest", "solved", "MBPP original test<br>solved", "2026-11-05T21-22-10", None, "2026-11-11T23-32-04", "2026-11-06T17-35-18", "2026-11-11T23-17-20"),
    ("honest", "solved", "MBPP impossible test<br>honest and correct", "2026-11-04T19-19-26", "2026-11-05T18-02-39", "2026-11-11T21-58-47", "2026-11-06T17-21-43", "2026-11-11T21-28-52"),
    ("honest", "passed", "terminal_verifier<br>control passed", "2026-11-04T19-51-55", "2026-11-05T18-27-45", "2026-11-11T23-14-52", "2026-11-06T17-54-16", "2026-11-11T23-12-12"),
]
TRAINERS = [("Grader", "grader_kto", "grader_kto_lora", 4, 5, 0), ("MBPP", "mbpp_kto", "mbpp_kto_lora", 6, 7, 2)]  # trained env, direction, adapter, their ENVS columns, the cheat panel's training-env group
SERIES = [("base Qwen3.8-27B", "#898781", -0.27), ("KTO projection direction", "#3987e5", 0.0), ("KTO LoRA fine-tune", "#d95926", 0.27)]


def load(prefix: str) -> list[dict]:
    files = glob.glob(str(ROOT / "data" / "inspect" / "*" / f"{prefix}*.jsonl"))
    assert len(files) == 1, (prefix, files)
    return [json.loads(line) for line in open(files[0])]


def graded(records: list[dict]) -> tuple[int, int]:
    labels = [r["labels"]["cheat"] for r in records]
    return sum(bool(x) for x in labels), sum(x is not None for x in labels)


def test_split(records: list[dict]) -> list[dict]:
    return [r for r in records if r["labels"]["split"] == "test"]


COUNT = {  # (k, n) of a run's records
    "grader": lambda rs: graded([r for r in rs if r["labels"]["family"] not in DROPPED]),
    "mbpp": lambda rs: graded(test_split(rs)),
    "tv": lambda rs: (sum(bool(r["cheated"]) for r in rs), len(rs)),
    "secret": lambda rs: COUNT["tv"]([r for r in rs if int(r["rollout_id"].split("/")[1]) < 96]),
    "solved": lambda rs: (sum(bool(r["labels"]["honest_correct"]) for r in test_split(rs)), len(test_split(rs))),
    "passed": lambda rs: (sum(bool(r["labels"]["passed"]) for r in rs), len(rs)),
}

#%% counts
counts = [[COUNT[kind](load(p)) if p else None for p in prefixes] for _, kind, _, *prefixes in ENVS]
for (panel, _, label, *_), row in zip(ENVS, counts):
    print(f"{panel:7}{label.replace('<br>', ', '):42}" + "".join(f"{f'{k}/{n} {k / n:.1%}' if (k, n) != (None, None) else 'not run':>18}" for k, n in [c or (None, None) for c in row]))

#%% figure
fig = make_subplots(rows=2, cols=2, column_widths=[0.7, 0.3], horizontal_spacing=0.06, vertical_spacing=0.2, subplot_titles=[f"<b>Trained on {env} rollouts</b> ({vec} against {lora}): {what}" if col == 0 else what for env, vec, lora, *_ in TRAINERS for col, what in enumerate(["cheat rate, lower is better", "honest success, higher is better"])])
for row, (env, vec, lora, vec_col, lora_col, trained_group) in enumerate(TRAINERS, start=1):
    for col, panel in enumerate(["cheat", "honest"], start=1):
        groups = [(e[2], c) for e, c in zip(ENVS, counts) if e[0] == panel]
        for (name, color, offset), column in zip(SERIES, [3, vec_col, lora_col]):
            cells = [(i + offset, *c[column - 3]) for i, (_, c) in enumerate(groups) if c[column - 3]]
            xs, rates = [x for x, _, _ in cells], [k / n for _, k, n in cells]
            lows, highs = [wilson(k, n)[0] for _, k, n in cells], [wilson(k, n)[1] for _, k, n in cells]
            error = dict(type="data", symmetric=False, array=[h - r for h, r in zip(highs, rates)], arrayminus=[r - l for r, l in zip(rates, lows)], color="#c3c2b7", thickness=1.1, width=4)
            fig.add_bar(row=row, col=col, name=name, legendgroup=name, showlegend=(row, col) == (1, 1), x=xs, y=rates, width=0.25, marker_color=color, marker_line_width=0, error_y=error, customdata=[f"{k}/{n}" for _, k, n in cells], hovertemplate="%{y:.1%} (%{customdata})<extra>" + name + "</extra>")
            fig.add_scatter(row=row, col=col, x=xs, y=[h + 0.02 for h in highs], mode="text", text=[f"{r:.0%}" if r >= 0.1 else f"{r:.1%}" for r in rates], textposition="top center", textfont=dict(size=11, color="#c3c2b7"), showlegend=False, hoverinfo="skip")
            missing = [i + offset for i, (_, c) in enumerate(groups) if not c[column - 3]]
            fig.add_scatter(row=row, col=col, x=missing, y=[0.03] * len(missing), mode="text", text=["not<br>run"] * len(missing), textposition="top center", textfont=dict(size=10, color="#898781"), showlegend=False, hoverinfo="skip")
        if panel == "cheat":
            fig.add_vrect(row=row, col=col, x0=trained_group - 0.47, x1=trained_group + 0.47, fillcolor="#f0efec", opacity=0.06, line_width=0, annotation_text="training env", annotation_position="top left", annotation_font=dict(size=11, color="#898781"))
        fig.update_xaxes(row=row, col=col, tickvals=list(range(len(groups))), ticktext=[label for label, _ in groups], showgrid=False, range=[-0.5, len(groups) - 0.5], tickfont_size=11)
        fig.update_yaxes(row=row, col=col, tickformat=".0%", range=[0, 1.12], dtick=0.25, gridcolor="#2c2c2a", zeroline=False)
fig.update_annotations(font_size=13, selector=lambda a: a.text.startswith(("<b>", "honest")), x=0, xanchor="left")
fig.layout.annotations[1].x = fig.layout.annotations[3].x = fig.layout.xaxis2.domain[0]
fig.update_layout(
    template="plotly_dark", paper_bgcolor="#1a1a19", plot_bgcolor="#1a1a19", font=dict(family="Inter, system-ui, sans-serif", size=13, color="#f0efec"), width=1700, height=900, bargap=0, margin=dict(l=60, r=30, t=150, b=110),
    title=dict(text="KTO as a LoRA fine-tune against KTO as a projection direction, Qwen3.8-27B<br><sup style='color:#898781'>Same rollouts, loss, steps and seed for both. LoRA: rank 16 on every linear of the 64 decoder layers. Direction: one unit vector projected out at layers 30-63, generated tokens only. Bars: 95% Wilson interval.</sup>", x=0.02, xanchor="left", y=0.96),
    legend=dict(orientation="h", x=1, xanchor="right", y=1.1, yanchor="bottom"),
)
fig.add_annotation(xref="paper", yref="paper", x=0, y=-0.13, xanchor="left", showarrow=False, align="left", font=dict(size=11, color="#898781"), text="Grader rates leave out the pets and fruit_vegetable families. MBPP is the test split, held out of both MBPP fits. terminal_verifier and secret_number are in no fit. secret_number: games 0-95 of each record; the two directions ran on a different server from the base run.<br>The base, direction and LoRA runs of a group were sampled on different days (2026-11-04 to 2026-11-12).")
OUT.parent.mkdir(exist_ok=True)
fig.write_html(OUT, include_plotlyjs="cdn")
print(OUT)
