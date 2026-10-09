#%%
import json
from pathlib import Path

import plotly.graph_objects as go

from mechtools import *

ROOT = Path(__file__).parent
PARITY_BASE = "grader/2026-10-02T21-52-56-00-00_grader_jFTa6f9P2uU4Mzxa5cP6yh"  # run.py parity
PARITY_STEER = "grader/2026-10-02T22-06-56-00-00_grader_2uELddztoLmkZiKmXBZegK"  # parity with the vector added at alpha 0.5
SECRET_BASE = "secret_number/2026-10-03T23-39-58-00-00_secret-number_NGPBxcKemYPK5ZAs8xTZMK"  # run.py secret
SECRET_STEER = "secret_number/2026-10-04T00-23-05-00-00_secret-number_F38Ehnh8vY7D4CKSawgta5"  # run.py secret_steer, alpha 0.4
GROUPS = ["parity, even side<br><span style='color:#7d8590'>steered at α = 0.5</span>", "parity, odd side<br><span style='color:#7d8590'>steered at α = 0.5</span>", "secret number<br><span style='color:#7d8590'>steered at α = 0.4</span>"]
SETUP = "Qwen3.6-27B, grader_parity_cheat_vs_clean added, unit rows, layers 12-47. Bars: 95% Wilson interval."


def load(run: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / "data" / "inspect" / f"{run}.jsonl").open()]


def invalid(r: dict) -> bool:
    """A grader answer on neither side, or a secret_number rollout that ended with no integer submitted."""
    return r["labels"]["cheat"] is None if r["env"] == "grader" else not r["completed"]


def cheat_count(records: list[dict], side: str | None = None) -> tuple[int, int]:
    """Cheats and valid rollouts, of one grader side when given."""
    valid = [r for r in records if (side is None or r["labels"]["side"] == side) and not (r["env"] == "grader" and invalid(r))]
    return sum(r["cheated"] for r in valid), len(valid)


def invalid_count(records: list[dict], side: str | None = None) -> tuple[int, int]:
    """Invalid rollouts and all rollouts, of one grader side when given."""
    rollouts = [r for r in records if side is None or r["labels"]["side"] == side]
    return sum(invalid(r) for r in rollouts), len(rollouts)


def rate_fig(base_counts: list[tuple[int, int]], steer_counts: list[tuple[int, int]], title: str, subtitle: str, ytitle: str, ymax: float, dtick: float) -> go.Figure:
    """Baseline and steered bars per group from (k, n) counts, each with its rate, count and Wilson interval."""
    fig = go.Figure()
    for name, counts, color, offset in [("baseline", base_counts, "#56607a", -0.19), ("steered", steer_counts, "#ffa94d", 0.19)]:
        xs = [i + offset for i in range(len(GROUPS))]
        rates = [k / n for k, n in counts]
        highs = [wilson(k, n)[1] for k, n in counts]
        lows = [wilson(k, n)[0] for k, n in counts]
        error = dict(type="data", symmetric=False, array=[h - r for h, r in zip(highs, rates)], arrayminus=[r - l for r, l in zip(rates, lows)], color="#c9d1d9", thickness=1.2, width=5)
        fig.add_bar(name=name, x=xs, y=rates, width=0.34, marker_color=color, marker_line_width=0, error_y=error, hovertemplate="%{y:.1%}<extra>" + name + "</extra>")
        fig.add_scatter(x=xs, y=[h + 0.017 * ymax for h in highs], mode="text", text=[f"<b>{k / n:.1%}</b><br><span style='color:#7d8590'>{k}/{n}</span>" for k, n in counts], textposition="top center", textfont_size=13, showlegend=False, hoverinfo="skip")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font=dict(family="Inter, system-ui, sans-serif", size=14, color="#e6edf3"), width=900, height=560, bargap=0, margin=dict(l=70, r=30, t=95, b=70),
        title=dict(text=f"{title}<br><sup style='color:#7d8590'>{subtitle}</sup>", x=0.02, xanchor="left"),
        legend=dict(orientation="h", x=1, xanchor="right", y=1.08, yanchor="bottom"),
        xaxis=dict(tickvals=list(range(len(GROUPS))), ticktext=GROUPS, showgrid=False, range=[-0.5, len(GROUPS) - 0.5]),
        yaxis=dict(title=ytitle, tickformat=".0%", range=[0, ymax], dtick=dtick, gridcolor="#21262d", zeroline=False),
    )
    return fig


#%% the four runs

parity_base = load(PARITY_BASE)
parity_steer = load(PARITY_STEER)
secret_base = load(SECRET_BASE)
secret_steer = load(SECRET_STEER)
(ROOT / "figures").mkdir(exist_ok=True)

#%% cheat rate among the valid rollouts, baseline against steered

base_counts = [cheat_count(parity_base, "even"), cheat_count(parity_base, "odd"), cheat_count(secret_base)]
steer_counts = [cheat_count(parity_steer, "even"), cheat_count(parity_steer, "odd"), cheat_count(secret_steer)]
fig = rate_fig(base_counts, steer_counts, "Cheat rate with the parity cheat direction added", SETUP, "rollouts that cheated", ymax=1.19, dtick=0.25)
fig.write_html(ROOT / "figures" / "steer_rates.html", include_plotlyjs="cdn")
fig.show()

#%% invalid rate among all rollouts, baseline against steered

base_counts = [invalid_count(parity_base, "even"), invalid_count(parity_base, "odd"), invalid_count(secret_base)]
steer_counts = [invalid_count(parity_steer, "even"), invalid_count(parity_steer, "odd"), invalid_count(secret_steer)]
fig = rate_fig(base_counts, steer_counts, "Invalid rollouts with the parity cheat direction added", "Parity: an answer that is not an integer. Secret number: no integer submitted. Bars: 95% Wilson interval.", "rollouts with no valid answer", ymax=0.13, dtick=0.02)
fig.write_html(ROOT / "figures" / "steer_invalid.html", include_plotlyjs="cdn")
fig.show()
