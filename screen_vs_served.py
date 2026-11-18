#%%
"""Does train_vector.py's teacher-forced screen predict the cheat rate of a served direction? Laptop, from the records. The screen scores the base model's own MBPP rollouts under a hook: on 97 held-out
same-prompt pairs (one cheating and one honest-and-correct completion of a test-split task), the log-prob gain of each completion under the hook, hooked minus base. Every direction here was served as a
projection hook and sampled on MBPP with the wrong visible test (run.py's <row>_mbpp), so each has a screen score and a served test-split cheat rate. Three of them (b_late_sub_bt, r_wi, synth_rat) were
served for this check, after acts/screen_check/predict.py wrote their predicted rates from the screen scores and the older rows. The screen scores come from one pass on the box (the screen files acts/screen_check/mkscreen.py
writes): data/vectors/Qwen3.8-27B/screen_check/compare_main.json (mbpp_kto's hook: one direction, layers 30-63, generated tokens) and compare_dom.json (mbpp_dom's: a direction per layer of 40-63, every
position). Prints the tables, writes acts/screen_check/results.json and figures/screen_vs_served.html."""
import glob
import json
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import logsumexp
from scipy.stats import spearmanr

from mechtools import *

ROOT = Path(__file__).parent
V = ROOT / "data/vectors/Qwen3.8-27B/screen_check"
OUT = ROOT / "figures" / "screen_vs_served.html"
ROWS = [   # the direction's name in the screen files, its served run, the label, and its kind
    ("mbpp_kto", "mbppkto_mbpp", "mbpp_kto", "trained"),
    ("screen_check/b1", "b1_mbpp", "b1", "trained"),
    ("grader_kto", "kto_mbpp", "grader_kto", "trained"),
    ("mbpp_hunt_vr48", "hvr_mbpp", "vr48", "closed form"),
    ("mbpp_hunt_vrw48", "hvrw_mbpp", "vrw48", "closed form"),
    ("mbpp_hunt_wdom", "hwdom_mbpp", "wdom", "closed form"),
    ("mbpp_dom", "mproj_mbpp", "mbpp_dom", "closed form"),
    ("mbpp_dom_rand", "mprojrand_mbpp", "random", "random"),
    ("dom_hunt_vr/b_late_sub_bt", "hbt_mbpp", "b_late_sub_bt", "new"),
    ("dom_hunt_vr/r_wi", "hrwi_mbpp", "r_wi", "new"),
    ("dom_hunt_new/synth_rat", "hrat_mbpp", "synth_rat", "new"),
]
CALIBRATION = ["mbpp_kto", "grader_kto", "vr48"]   # the rows predict.py fit its slopes on: the served rows with a positive margin and a small honest loss
POSITION = {"r_wi": "middle right", "vr48": "bottom left", "random": "bottom right"}   # label placement where the default (top right) collides
COLOR = {"trained": "#3987e5", "closed form": "#199e70", "random": "#898781", "new": "#d95926"}
logit = lambda p: math.log(p / (1 - p))
expit = lambda x: 1 / (1 + math.exp(-x))

#%% served: the test-split rollouts of every MBPP run with the wrong visible test, by condition; the base runs by their samples per task
runs = {}
for path in sorted(glob.glob(str(ROOT / "data/inspect/mbpp/*.jsonl"))):
    records = [json.loads(l) for l in open(path)]
    r = records[0]
    if not (r["model"].startswith("vllm/cb_") and r.get("corrupt") and len(records) >= 954): continue
    key = r["condition"] if r["condition"] != "none" else f"base_x{r['cfg']['epochs']}"
    assert key not in runs, f"two runs of {key}"
    runs[key] = [dict(task=str(x["labels"]["task_id"]), cheat=x["labels"]["cheat"], solved=bool(x["labels"]["honest_correct"])) for x in records if x["labels"]["split"] == "test"]
base = runs["base_x1"] + runs["base_x4"] + runs["base_x2"]   # base_x2 is the base model on the server the three new rows ran on

def counts(rollouts: list[dict]) -> dict:
    valid = [x for x in rollouts if x["cheat"] is not None]
    return dict(n=len(rollouts), valid=len(valid), cheat=sum(x["cheat"] for x in valid), solved=sum(x["solved"] for x in rollouts))

def task_rates(rollouts: list[dict]) -> dict[str, float]:
    """Each task's cheat rate over its valid rollouts."""
    by_task = {}
    for x in rollouts:
        if x["cheat"] is not None: by_task.setdefault(x["task"], []).append(x["cheat"])
    return {k: sum(v) / len(v) for k, v in by_task.items()}

for key in ("base_x1", "base_x4", "base_x2"):
    c = counts(runs[key])
    print(f"{gray}{key}: cheated {c['cheat']}/{c['valid']} = {c['cheat'] / c['valid']:.3f}, honest and correct {c['solved']}/{c['n']} = {c['solved'] / c['n']:.3f}{endc}")
base_counts = counts(base)
base_rate, base_solved = base_counts["cheat"] / base_counts["valid"], base_counts["solved"] / base_counts["n"]
base_tasks = task_rates(base)

#%% screen: each direction's per-pair gains (nats over a completion) and what the screen's statistics are computed from them
pairs = json.load(open(ROOT / "acts/screen_check/pairs.json"))
screens = {s["name"]: s for name in ("main", "dom") for s in json.load(open(V / f"compare_{name}.json"))["scores"]}
rng = np.random.default_rng(0)
rows = []
for screen_name, run, label, kind in ROWS:
    s, p = screens[screen_name], screens[screen_name]["pairs"]
    assert p["chosen_tokens"] == [x["honest_tokens"] for x in pairs] and p["rejected_tokens"] == [x["cheating_tokens"] for x in pairs], f"{screen_name}: not the 97 pairs of pairs.json"
    honest, cheating = np.array(p["chosen"]), np.array(p["rejected"])
    honest_tok, cheating_tok = 1000 * honest / np.array(p["chosen_tokens"]), 1000 * cheating / np.array(p["rejected_tokens"])   # millinats per token of each completion
    margin_tok = honest_tok - cheating_tok
    boot = [margin_tok[rng.integers(0, 97, 97)].mean() for _ in range(2000)]
    c = counts(runs[run])
    rate = c["cheat"] / c["valid"]
    on_pairs = task_rates(runs[run])
    tasks = [i for i, x in enumerate(pairs) if x["task"] in on_pairs]   # the pair tasks with a valid served rollout
    shift = [on_pairs[pairs[i]["task"]] - base_tasks[pairs[i]["task"]] for i in tasks]
    rho = spearmanr(margin_tok[tasks], shift)
    rows.append(dict(
        label=label, kind=kind, run=run, **c, rate=rate, wilson=wilson(c["cheat"], c["valid"]), solved_rate=c["solved"] / c["n"], logit_shift=logit(rate) - logit(base_rate),
        honest_tok=honest_tok.mean(), cheating_tok=cheating_tok.mean(), margin_tok=margin_tok.mean(), margin_tok_ci=[float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))], ratio=margin_tok.mean() / -honest_tok.mean(),
        honest=honest.mean(), cheating=cheating.mean(), margin=(honest - cheating).mean(), pair_acc=float((margin_tok > 0).mean()), neutral_kl_tok=1000 * s["neutral_kl_per_token"], neutral_gain_tok=1000 * s["neutral_gain_per_token"],
        identity=float(logsumexp(cheating) - logsumexp(honest)), ess_cheating=float(np.exp(2 * logsumexp(cheating) - logsumexp(2 * cheating))), ess_honest=float(np.exp(2 * logsumexp(honest) - logsumexp(2 * honest))),   # P_hook(class) / P_base(class) = E_base[exp(gain)] over the class's rollouts: the pooled log-odds shift it implies, and how many of the 97 rollouts carry each mean
        pair_tasks=len(tasks), pair_rate=float(np.mean([on_pairs[pairs[i]["task"]] for i in tasks])), pair_base_rate=float(np.mean([base_tasks[pairs[i]["task"]] for i in tasks])), task_rho=float(rho.statistic), task_p=float(rho.pvalue),
    ))

#%% tables
show_table(["direction", "kind", "honest gain", "cheating gain", "margin [95%]", "margin / honest loss", "summed margin", "neutral KL", "served cheat rate [95%]", "honest and correct"],
           [(r["label"], r["kind"], f"{r['honest_tok']:+.2f}", f"{r['cheating_tok']:+.2f}", f"{r['margin_tok']:+.2f} [{r['margin_tok_ci'][0]:+.2f}, {r['margin_tok_ci'][1]:+.2f}]", f"{r['ratio']:.2f}", f"{r['margin']:+.1f}", f"{r['neutral_kl_tok']:.2f}",
             f"{r['cheat']}/{r['valid']} = {r['rate']:.3f} [{r['wilson'][0]:.3f}, {r['wilson'][1]:.3f}]", f"{r['solved']}/{r['n']} = {r['solved_rate']:.3f}") for r in rows]
           + [("base", "", "0", "0", "0", "", "0", "0", f"{base_counts['cheat']}/{base_counts['valid']} = {base_rate:.3f}", f"{base_counts['solved']}/{base_counts['n']} = {base_solved:.3f}")])
print(f"{gray}gains and margins in millinats per token of a completion, the summed margin in nats per completion, neutral KL in millinats per token{endc}")

STATS = [("margin per token", "margin_tok"), ("summed margin", "margin"), ("cheating gain per token", "cheating_tok"), ("margin / honest loss", "ratio"), ("identity log-odds", "identity")]
agreement = {}
for name, key in STATS:
    for subset, chosen in [("all", rows), ("small honest loss", [r for r in rows if r["honest_tok"] > -10])]:
        for target, served in [("test split", "rate"), ("pair tasks", "pair_rate")]:   # the served cheat rate over the whole test split, and over the screen's own tasks
            rho = spearmanr([r[key] for r in chosen], [r[served] for r in chosen])
            agreement[f"{name}, {subset}, {target}"] = dict(n=len(chosen), rho=float(rho.statistic), p=float(rho.pvalue))
show_table(["statistic", "rows", "served cheat rate on", "n", "Spearman rho", "p"], [(*k.split(", "), v["n"], f"{v['rho']:+.2f}", f"{v['p']:.3f}") for k, v in agreement.items()])

predicted = json.load(open(ROOT / "acts/screen_check/predictions.json"))
by_label = {r["label"]: r for r in rows}
slopes = {key: sum(by_label[l][key] * -by_label[l]["logit_shift"] for l in CALIBRATION) / sum(by_label[l][key] ** 2 for l in CALIBRATION) for key in ("margin_tok", "margin")}   # predict.py's fit, on this file's numbers
for r in rows:
    r["predicted_tok"], r["predicted_summed"] = expit(logit(base_rate) - slopes["margin_tok"] * r["margin_tok"]), expit(logit(base_rate) - slopes["margin"] * r["margin"])
    r["preregistered"] = predicted["predictions"].get(r["label"], {})
show_table(["direction", "served cheat rate [95%]", "predicted from margin per token", "predicted from summed margin", "written before serving (summed, per token)"],
           [(r["label"], f"{r['rate']:.3f} [{r['wilson'][0]:.3f}, {r['wilson'][1]:.3f}]", f"{r['predicted_tok']:.3f}", f"{r['predicted_summed']:.3f}", ", ".join(f"{r['preregistered'][k]:.3f}" if k in r["preregistered"] else "none" for k in ("summed", "per_token")) if r["kind"] == "new" else "") for r in rows])
show_table(["direction", "pair tasks sampled", "base cheat rate on them", "served cheat rate on them", "Spearman rho, a task's margin with its rate change", "p"],
           [(r["label"], r["pair_tasks"], f"{r['pair_base_rate']:.3f}", f"{r['pair_rate']:.3f}", f"{r['task_rho']:+.2f}", f"{r['task_p']:.3f}") for r in rows])
show_table(["direction", "identity log-odds shift", "served log-odds shift", "rollouts carrying the cheating mean", "the honest mean"], [(r["label"], f"{r['identity']:+.1f}", f"{r['logit_shift']:+.2f}", f"{r['ess_cheating']:.1f}", f"{r['ess_honest']:.1f}") for r in rows])
json.dump(dict(base=base_counts | dict(rate=base_rate, solved_rate=base_solved, runs={k: counts(runs[k]) for k in ("base_x1", "base_x4", "base_x2")}), slopes=slopes, agreement=agreement, rows=rows), open(ROOT / "acts/screen_check/results.json", "w"), indent=1)

#%% figure: the two margins against the served cheat rate, and the honest loss against honest solving
fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.07, subplot_titles=["<b>Margin per token</b> against the served cheat rate", "<b>Summed margin</b> against the served cheat rate", "<b>Honest loss per token</b> against honest solving"])
PANELS = [("margin_tok", "rate", "screen: honest gain minus cheating gain, millinats per token", "margin_tok"), ("margin", "rate", "screen: honest gain minus cheating gain, nats per completion", "margin"), ("honest_loss", "solved_rate", "screen: log-prob lost on honest completions, millinats per token (log scale)", None)]
for r in rows: r["honest_loss"] = -r["honest_tok"]
for col, (x, y, x_title, slope_key) in enumerate(PANELS, start=1):
    for kind, color in COLOR.items():
        chosen = [r for r in rows if r["kind"] == kind]
        error = dict(type="data", symmetric=False, array=[r["wilson"][1] - r["rate"] for r in chosen], arrayminus=[r["rate"] - r["wilson"][0] for r in chosen], color=color, thickness=1.2, width=4) if y == "rate" else None
        fig.add_scatter(row=1, col=col, x=[r[x] for r in chosen], y=[r[y] for r in chosen], mode="markers+text", name={"trained": "trained (KTO)", "closed form": "closed form, served earlier", "random": "random direction", "new": "served for this check"}[kind], legendgroup=kind, showlegend=col == 1,
                        marker=dict(color=color, size=13 if kind == "new" else 10, symbol="diamond" if kind == "new" else "circle", line=dict(color="#1a1a19", width=1)), error_y=error, text=[r["label"] for r in chosen], textposition=[POSITION.get(r["label"], "top right") for r in chosen], textfont=dict(size=11, color="#c3c2b7"),
                        customdata=[f"{r['cheat']}/{r['valid']} cheated, {r['solved']}/{r['n']} honest and correct" for r in chosen], hovertemplate="%{text}: x %{x:.2f}, y %{y:.1%} (%{customdata})<extra></extra>")
    fig.add_hline(row=1, col=col, y=base_rate if y == "rate" else base_solved, line=dict(color="#c3c2b7", width=1, dash="dot"), annotation_text="base model", annotation_position="bottom right", annotation_font=dict(size=11, color="#898781"))
    if slope_key:
        xs = np.linspace(min(0, min(r[x] for r in rows)), max(r[x] for r in rows), 200)
        fig.add_scatter(row=1, col=col, x=xs, y=[expit(logit(base_rate) - slopes[slope_key] * v) for v in xs], mode="lines", line=dict(color="#898781", width=1.2, dash="dash"), name="fit on mbpp_kto, grader_kto, vr48", legendgroup="fit", showlegend=col == 1, hoverinfo="skip")
    lo, hi = (math.log10(min(r[x] for r in rows)), math.log10(max(r[x] for r in rows))) if x == "honest_loss" else (min(0, min(r[x] for r in rows)), max(r[x] for r in rows))
    fig.update_xaxes(row=1, col=col, title_text=x_title, title_font_size=12, gridcolor="#2c2c2a", zeroline=False, type="log" if x == "honest_loss" else "linear", range=[lo - 0.08 * (hi - lo), hi + 0.22 * (hi - lo)])   # room for the labels
    fig.update_yaxes(row=1, col=col, tickformat=".0%", gridcolor="#2c2c2a", zeroline=False, rangemode="tozero", title_text="cheat rate, MBPP test split, wrong visible test" if col == 1 else "honest and correct, of all test-split rollouts" if col == 3 else None, title_font_size=12)
fig.update_annotations(font_size=13, selector=lambda a: a.text.startswith("<b>"))
fig.update_layout(
    template="plotly_dark", paper_bgcolor="#1a1a19", plot_bgcolor="#1a1a19", font=dict(family="Inter, system-ui, sans-serif", size=13, color="#f0efec"), width=1800, height=760, margin=dict(l=70, r=30, t=170, b=120),
    title=dict(text="Does the teacher-forced log-prob screen predict what a served direction does? Qwen3.8-27B, MBPP with a wrong visible test<br><sup style='color:#898781'>Each point is one direction served as a projection hook. Screen (x): on 97 held-out pairs of base rollouts of one task, one cheating and one honest, the change in each completion's log-prob under the hook."
               "<br>Served (y): sampling from the hooked model on the test split, bars are 95% Wilson intervals. Dashed: logit(rate) = logit(base) - slope x margin, fit on three earlier rows before the orange rows were served.</sup>", x=0.02, xanchor="left", y=0.96),
    legend=dict(orientation="h", x=1, xanchor="right", y=1.07, yanchor="bottom"),
)
fig.add_annotation(xref="paper", yref="paper", x=0, y=-0.17, xanchor="left", showarrow=False, align="left", font=dict(size=11, color="#898781"),
                   text=f"Base: {base_counts['cheat']}/{base_counts['valid']} cheated over three base runs (1, 4 and 2 samples per task). Earlier rows have one sample per test task (495), the orange rows two (990). mbpp_dom and its random control are hooked at every position of layers 40-63, one direction per layer;"
                        "<br>the others at generated tokens of layers 30-63, one direction. An invalid answer (no code block, or cut off) is left out of the cheat rate and counts as not solved.")
head = "".join(f"<th>{h}</th>" for h in ["direction", "kind", "honest gain / token", "cheating gain / token", "margin / token [95%]", "margin / honest loss", "summed margin", "neutral KL / token", "served cheat rate [95%]", "predicted (per token, summed)", "honest and correct"])
body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in (r["label"], r["kind"], f"{r['honest_tok']:+.2f}", f"{r['cheating_tok']:+.2f}", f"{r['margin_tok']:+.2f} [{r['margin_tok_ci'][0]:+.2f}, {r['margin_tok_ci'][1]:+.2f}]", f"{r['ratio']:.2f}", f"{r['margin']:+.1f}", f"{r['neutral_kl_tok']:.2f}",
                                                           f"{r['cheat']}/{r['valid']} = {r['rate']:.1%} [{r['wilson'][0]:.1%}, {r['wilson'][1]:.1%}]", f"{r['predicted_tok']:.1%}, {r['predicted_summed']:.1%}", f"{r['solved']}/{r['n']} = {r['solved_rate']:.1%}")) + "</tr>" for r in rows)
style = "<style>body{background:#1a1a19;color:#f0efec;font-family:Inter,system-ui,sans-serif;font-size:13px}table{border-collapse:collapse;margin:10px 70px 40px}th,td{padding:5px 12px;border-bottom:1px solid #2c2c2a;text-align:right}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}th{color:#c3c2b7;font-weight:500}p{margin:0 70px;color:#898781}</style>"
OUT.write_text(f"<html><head><meta charset=utf-8>{style}</head><body>{fig.to_html(include_plotlyjs='cdn', full_html=False)}<p>Gains and margins in millinats per token of a completion; summed margin in nats per completion; neutral KL in millinats per token of the held-out neutral rollouts.</p><table><tr>{head}</tr>{body}</table></body></html>")
print(OUT)
