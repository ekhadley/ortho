#%%
"""Does mbpp_kto work? One page: its cheat rates against base on every env it ran on, the honest-solving and Terminal-Bench cost, the same hook with other directions, and what has not been run. Counts from the records and Terminal-Bench job results; Will's rows from his reports (research log E8, E14)."""
import glob
import json
import math
import os
import shutil
from pathlib import Path

OUT = Path.home() / "wgmn/artifacts/2026-11-11_does-mbpp-kto-work.html"
JOBS = "logs/tbench/jobs/"
NOISE_FLIPS = 12   # our two base Terminal-Bench runs disagree on 12 tasks with the same total (E9)
DROPPED = {"pets", "fruit_vegetable"}   # removed from rhenvs 2026-11-09 for sidestep answers; old records still carry them
MBPP = {"base": "2026-11-04T19-19-26", "mbppkto": "2026-11-06T17-21-43", "kto": "2026-11-05T18-02-39", "proj": "2026-11-05T04-00-00", "hvr": "2026-11-10T16-04-28", "hvrw": "2026-11-10T16-27-18", "hwdom": "2026-11-10T16-50-37"}
CORRECT = {"base": "2026-11-05T21-22-10", "mbppkto": "2026-11-06T17-35-18", "hvr": "2026-11-10T16-16-30", "hvrw": "2026-11-10T16-39-56", "hwdom": "2026-11-10T17-06-28"}
HELDOUT = {"base": "2026-11-05T17-54-51", "mbppkto": "2026-11-06T00-10-38", "hvr": "2026-11-10T16-22-50", "hvrw": "2026-11-10T16-46-07", "hwdom": "2026-11-10T17-12-24"}
TV = {
    "bug4": {"base": "2026-11-04T19-38-50", "mbppkto": "2026-11-06T17-42-21", "kto": "2026-11-05T18-15-24"},
    "bug6": {"base": "2026-11-04T19-46-45", "mbppkto": "2026-11-06T17-46-16", "kto": "2026-11-05T18-19-29"},
    "corr": {"base": "2026-11-04T19-43-11", "mbppkto": "2026-11-06T17-51-04", "kto": "2026-11-05T18-24-32"},
    "pos": {"base": "2026-11-04T19-51-55", "mbppkto": "2026-11-06T17-54-16", "kto": "2026-11-05T18-27-45"},
}
WILL = {   # Will's pool, A3 against his base (E8: CONF_A3_G_L30_report.md; E14: gr_ho_hack). k is his percentage of n where he gave a percentage.
    "mbpp": {"wbase": (74, 336), "a3": (8, 336)}, "bug4": {"wbase": (92, 200), "a3": (0, 200)}, "bug6": {"wbase": (194, 200), "a3": (0, 200)},
    "corr": {"wbase": (191, 200), "a3": (4, 200)}, "heldout": {"wbase": (603, 1849), "a3": (0, 1861)},
}
TB_WILL = {"a3": (13, 0, 33), "half": (36, 4, 14), "rand": (30, 2, 18)}   # passes, won, lost against his base of 46

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

def graded(records: list[dict]) -> dict:
    labels = [r["labels"]["cheat"] for r in records]
    return cell(sum(bool(x) for x in labels), sum(x is not None for x in labels))

def tb_job(*names: str) -> dict:
    tasks = {}
    for name in names:
        for f in glob.glob(f"{JOBS}{name}/*__*/result.json"):
            r = json.load(open(f))
            rewards = (r.get("verifier_result") or {}).get("rewards") or {}
            tasks[r.get("task_name") or os.path.basename(os.path.dirname(f)).rsplit("__", 1)[0]] = rewards.get("reward")
    return tasks

#%% cheat rates and outcomes
mbpp, fit, outcomes = {}, {}, {}
for row, prefix in MBPP.items():
    records = load(prefix)
    test = [r for r in records if r["labels"]["split"] == "test"]
    mbpp[row] = graded(test)
    fit[row] = graded([r for r in records if r["labels"]["split"] != "test"])
    c, h = sum(bool(r["labels"]["cheat"]) for r in test), sum(bool(r["labels"]["honest_correct"]) for r in test)
    outcomes[row] = {"cheat": c, "honest": h, "wrong": sum(r["labels"]["valid"] for r in test) - c - h, "invalid": sum(not r["labels"]["valid"] for r in test), "n": len(test)}
correct = {}
for row, prefix in CORRECT.items():
    test = [r for r in load(prefix) if r["labels"]["split"] == "test"]
    correct[row] = cell(sum(bool(r["labels"]["honest_correct"]) for r in test), len(test))
heldout, families = {}, {}
for row, prefix in HELDOUT.items():
    records = [r for r in load(prefix) if r["labels"]["family"] not in DROPPED]
    heldout[row] = graded(records)
    families[row] = {f: graded([r for r in records if r["labels"]["family"] == f]) for f in sorted({r["labels"]["family"] for r in records})}
tv, curves = {}, {}
for env, runs in TV.items():
    tv[env] = {}
    for row, prefix in runs.items():
        records = load(prefix)
        tv[env][row] = cell(sum(bool(r["labels"]["passed"]) for r in records) if env == "pos" else sum(bool(r["cheated"]) for r in records), len(records))
        if env == "bug6":
            first = [r["first_cheat_turn"] for r in records if r["cheated"]]
            curves[row] = [sum(t <= turn for t in first) for turn in range(1, 7)]
will = {env: {row: cell(k, n) for row, (k, n) in rows.items()} for env, rows in WILL.items()}

#%% Terminal-Bench
tb = {"base": tb_job("tb2_cb-base"), "proj": tb_job("tb2_cb-proj"), "cm": tb_job("tb2_cb-cm", "tb2_cb-cm-rest")}
first = {t: r for t, r in tb_job("tb2_cb-mbppkto").items() if r is not None}
tb["mbppkto"] = first | tb_job("tb2_cb-mbppkto-rest")
assert all(len(v) == 89 for v in tb.values()), {k: len(v) for k, v in tb.items()}
passed = lambda row, task: tb[row].get(task) == 1
tb_rows = {}
for row in tb:
    won = sum(passed(row, t) and not passed("base", t) for t in tb["base"])
    lost = sum(passed("base", t) and not passed(row, t) for t in tb["base"])
    tb_rows[row] = {"passes": sum(passed(row, t) for t in tb[row]), "won": won, "lost": lost}
tb_rows |= {row: {"passes": p, "won": w, "lost": l} for row, (p, w, l) in TB_WILL.items()}
print({r: (v["passes"], v["won"], v["lost"]) for r, v in tb_rows.items()})
print("mbpp test", mbpp["base"]["k"], mbpp["base"]["n"], mbpp["mbppkto"]["k"], mbpp["mbppkto"]["n"], "heldout", heldout["base"]["k"], heldout["base"]["n"], heldout["mbppkto"]["k"], heldout["mbppkto"]["n"])

#%% page
data = {"mbpp": mbpp, "fit": fit, "outcomes": outcomes, "correct": correct, "heldout": heldout, "families": families, "tv": tv, "curves": curves, "will": will, "tb": tb_rows, "noise": round(1.96 * math.sqrt(NOISE_FLIPS), 1)}
template = Path("mbppkto_template.html").read_text()
OUT.write_text(template.replace("/*DATA*/", json.dumps(data).replace("</", "<\\/")))
shutil.copy(OUT, Path("figures") / OUT.name)
print(OUT, OUT.stat().st_size)
