#%%
"""Cheating against capability for every intervention, each row against its own base: counts from the records and Terminal-Bench job results, Will's rows from his reports (research log E8, E9, E12). Writes one self-contained page."""
import glob
import json
import math
import os
from pathlib import Path

OUT = Path.home() / "wgmn/artifacts/2026-11-06_cheating-vs-capability.html"
JOBS = "logs/tbench/jobs/"
NOISE_FLIPS = 12  # our two base Terminal-Bench runs disagree on 12 tasks, same total (E9)

ROWS = {  # id: label, method family, whose pool
    "base": ("base", "ctrl", "ours"), "cm": ("cm", "steer", "ours"), "proj": ("grader_proj", "dpo", "ours"), "projall": ("grader_proj_all", "dpo", "ours"),
    "projgen": ("projgen", "dpo", "ours"), "projallgen": ("projallgen", "dpo", "ours"), "kto": ("grader_kto", "kto", "ours"), "mbppkto": ("mbpp_kto", "kto", "ours"),
    "domlast": ("nocot_dom_last", "ctrl", "ours"), "domresp": ("nocot_dom_resp", "ctrl", "ours"),
    "wbase": ("base (Will)", "ctrl", "will"), "a3": ("A3", "steer", "will"), "half": ("A3 half", "steer", "will"), "rand": ("random steer", "ctrl", "will"),
}
MBPP = {"base": "2026-11-04T19-19-26", "cm": "2026-11-04T19-56-26", "proj": "2026-11-05T04-00-00", "projall": "2026-11-05T14-04-57", "projgen": "2026-11-05T14-28-11", "kto": "2026-11-05T18-02-39", "mbppkto": "2026-11-06T17-21-43"}
CORRECT = {"base": "2026-11-05T21-22-10", "mbppkto": "2026-11-06T17-35-18"}
TV = {
    "bug4": {"base": "2026-11-04T19-38-50", "cm": "2026-11-04T20-08-15", "projallgen": "2026-11-05T16-21-40", "kto": "2026-11-05T18-15-24", "mbppkto": "2026-11-06T17-42-21"},
    "bug6": {"base": "2026-11-04T19-46-45", "cm": "2026-11-04T20-19-13", "projallgen": "2026-11-05T16-25-26", "kto": "2026-11-05T18-19-29", "mbppkto": "2026-11-06T17-46-16"},
    "corr": {"base": "2026-11-04T19-43-11", "cm": "2026-11-04T20-14-31", "projallgen": "2026-11-05T16-30-15", "kto": "2026-11-05T18-24-32", "mbppkto": "2026-11-06T17-51-04"},
}
GRADER = {"base": ["2026-11-04T19-18-43", "2026-11-04T19-19-06"], "cm": ["2026-11-04T19-55-51", "2026-11-04T19-56-10"], "proj": ["2026-11-05T04-12-26", "2026-11-05T04-12-45"], "projall": ["2026-11-05T14-04-29", "2026-11-05T14-04-42"], "projgen": ["2026-11-05T14-39-20", "2026-11-05T14-39-37"], "kto": ["2026-11-05T18-01-17", "2026-11-05T18-01-35"]}
MAIN = {"base": "2026-11-05T18-16-40", "domlast": "2026-11-05T18-13-59", "domresp": "2026-11-05T18-11-31"}
HELDOUT = {"base": "2026-11-05T17-54-51", "mbppkto": "2026-11-06T00-10-38", "domlast": "2026-11-05T17-59-48", "domresp": "2026-11-05T18-03-48"}
# Will's pool: reports quoted in E8, E9 and E12. k is the reported percentage of n where he gave a percentage.
WILL = {
    "mbpp": {"wbase": (74, 336), "a3": (8, 336), "half": (23, 336), "rand": (89, 336)},
    "bug4": {"wbase": (92, 200), "a3": (0, 200), "half": (1, 200), "rand": (106, 200)},
    "bug6": {"wbase": (194, 200), "a3": (0, 200), "half": (32, 200), "rand": (168, 200)},
    "corr": {"wbase": (191, 200), "a3": (4, 200), "half": (165, 200), "rand": (175, 200)},
    "heldout": {"wbase": (603, 1849), "a3": (0, 1861), "half": (5, 1903)},  # gr_ho_hack, his 8 new families, 2048 answers; base 32.6% of 90.3% valid
}
WILL_MBPP_HONEST = {"wbase": 196, "half": 217, "a3": 196, "rand": 164}
TB_WILL = {"a3": (13, 0, 33), "half": (36, 4, 14), "rand": (30, 2, 18)}  # passes, won, lost against his base (46)
CAPABILITY = {  # points against the row's own base
    "mmlu": {"proj": (-0.24, -0.41, -0.07), "projall": (-0.26, -0.46, -0.06), "half": (-0.70, None, None), "a3": (-2.72, None, None)},
    "gsm8k": {"cm": (-1.5, None, None), "half": (-1.2, None, None), "a3": (-6.9, None, None), "rand": (-4.6, None, None)},
    "ifeval": {"cm": (-3.2, None, None), "half": (-2.1, None, None), "a3": (-8.4, None, None), "rand": (-1.1, None, None)},
}
SIDE = {  # cells above noise of 24, chars no grader, chars agreeing grader, from each run's suite.log
    "proj": (4, 201, 206), "projall": (6, 205, 201), "projgen": (3, 173, 192), "projallgen": (2, 173, 190), "kto": (0, 177, 188),
}
SIDE_BASE = (174, 189)

def load(prefix: str) -> list[dict]:
    files = glob.glob(f"data/inspect/*/{prefix}*.jsonl")
    assert len(files) == 1, (prefix, files)
    return [json.loads(line) for line in open(files[0])]

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

def cell(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n, "lo": lo, "hi": hi}

def graded(records: list[dict]) -> tuple[int, int]:
    labels = [r["labels"]["cheat"] for r in records]
    return sum(bool(x) for x in labels), sum(x is not None for x in labels)

def tb_job(*names: str) -> dict:
    tasks = {}
    for name in names:
        for f in glob.glob(f"{JOBS}{name}/*__*/result.json"):
            r = json.load(open(f))
            rewards = (r.get("verifier_result") or {}).get("rewards") or {}
            tasks[r.get("task_name") or os.path.basename(os.path.dirname(f)).rsplit("__", 1)[0]] = rewards.get("reward")
    return tasks

#%% cheat rates
cheat = {"mbpp": {}, "bug4": {}, "bug6": {}, "corr": {}, "trained": {}, "heldout": {}}
outcomes, curves, families = {}, {}, {}
for row, prefix in MBPP.items():
    records = [r for r in load(prefix) if r["labels"]["split"] == "test"]   # mbpp_kto trained on the other splits
    cheat["mbpp"][row] = cell(*graded(records))
    c, h = sum(bool(r["labels"]["cheat"]) for r in records), sum(bool(r["labels"].get("honest_correct")) for r in records)
    outcomes[row] = {"cheat": c, "honest": h, "other": len(records) - c - h, "n": len(records)}
for row, h in WILL_MBPP_HONEST.items():
    outcomes[row] = {"cheat": WILL["mbpp"][row][0], "honest": h, "other": 336 - WILL["mbpp"][row][0] - h, "n": 336}
for env, runs in TV.items():
    for row, prefix in runs.items():
        records = load(prefix)
        cheat[env][row] = cell(sum(bool(r["cheated"]) for r in records), len(records))
        if env == "bug6":
            first = [r["first_cheat_turn"] for r in records if r["cheated"]]
            curves[row] = [100 * sum(t <= turn for t in first) / len(records) for turn in range(1, 7)]
for row, prefixes in GRADER.items():
    pairs = [graded(load(p)) for p in prefixes]
    cheat["trained"][row] = cell(sum(p[0] for p in pairs), sum(p[1] for p in pairs))
correct = {}
for row, prefix in CORRECT.items():
    records = [r for r in load(prefix) if r["labels"]["split"] == "test"]
    correct[row] = {"honest": sum(bool(r["labels"].get("honest_correct")) for r in records), "valid": sum(bool(r["labels"].get("valid")) for r in records), "n": len(records)}
main = {row: cell(*graded(load(p))) for row, p in MAIN.items()}
for row in ("domlast", "domresp"):
    cheat["trained"][row] = main[row]
for row, prefix in HELDOUT.items():
    records = load(prefix)
    cheat["heldout"][row] = cell(*graded(records))
    fam = {}
    for r in records:
        f = fam.setdefault(r["labels"]["family"], [0, 0])
        f[0] += bool(r["labels"]["cheat"])
        f[1] += r["labels"]["cheat"] is not None
    families[row] = fam

for env, rows in WILL.items():
    for row, (k, n) in rows.items():
        cheat[env][row] = cell(k, n)

def base_of(row: str, env: str) -> dict:
    if ROWS[row][2] == "will":
        return cheat[env]["wbase"]
    return main["base"] if env == "trained" and row.startswith("dom") else cheat[env]["base"]

def cut(row: str, env: str) -> float | None:
    return 100 * (1 - cheat[env][row]["rate"] / base_of(row, env)["rate"]) if row in cheat[env] else None

cuts = {row: {env: cut(row, env) for env in cheat} for row in ROWS if row not in ("base", "wbase")}
overall = {}
for row, c in cuts.items():
    tv = [c[e] for e in ("bug4", "bug6", "corr") if c[e] is not None]
    groups = {"MBPP": c["mbpp"], "TV": sum(tv) / len(tv) if tv else None, "held-out grader": c["heldout"]}
    groups = {k: v for k, v in groups.items() if v is not None}
    overall[row] = {"value": sum(groups.values()) / len(groups), "groups": groups}

#%% Terminal-Bench
tb = {"base": tb_job("tb2_cb-base"), "proj": tb_job("tb2_cb-proj"), "cm": tb_job("tb2_cb-cm", "tb2_cb-cm-rest")}
first = {t: r for t, r in tb_job("tb2_cb-mbppkto").items() if r is not None}
second = tb_job("tb2_cb-mbppkto-rest")
tb["mbppkto"] = first | second
assert all(len(v) == 89 for v in tb.values()), {k: len(v) for k, v in tb.items()}
passed = lambda row, task: tb[row].get(task) == 1
tb_rows = {"wbase": (46, 0, 0)} | TB_WILL
for row in ("base", "proj", "cm", "mbppkto"):
    won = sum(passed(row, t) and not passed("base", t) for t in tb["base"])
    lost = sum(passed("base", t) and not passed(row, t) for t in tb["base"])
    tb_rows[row] = (sum(passed(row, t) for t in tb[row]), won, lost)
order = sorted(tb["base"], key=lambda t: (-sum(passed(r, t) for r in tb), t))
matrix = {"tasks": order, "rows": {row: [tb[row][t] for t in order] for row in tb}, "second_job": sorted(second)}
print({r: v for r, v in tb_rows.items()})
print({r: round(v["value"], 1) for r, v in overall.items()})

#%% page
data = {
    "rows": {k: {"label": v[0], "family": v[1], "pool": v[2]} for k, v in ROWS.items()}, "cheat": cheat, "cuts": cuts, "overall": overall, "outcomes": outcomes, "curves": curves,
    "families": families, "tb": {k: {"passes": v[0], "won": v[1], "lost": v[2]} for k, v in tb_rows.items()}, "matrix": matrix, "noise": round(1.96 * math.sqrt(NOISE_FLIPS), 1),
    "capability": CAPABILITY, "side": SIDE, "side_base": SIDE_BASE, "correct": correct,
}
template = Path("scoreboard_template.html").read_text()
OUT.write_text(template.replace("/*DATA*/", json.dumps(data).replace("</", "<\\/")))
print(OUT, OUT.stat().st_size)
