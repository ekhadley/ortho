"""Trace viewer for the ImpossibleBench runs on Qwen3.8-27B through OpenRouter (E32), laptop: every rollout of run.py's lcb38_plain and lcb38_paper (the conflicting split, where a pass is the cheat)
and lcb38_original (the original split, where a pass is a solve), 103 each. Fills lcb_traces_template.html (styles and vanilla JS) and writes one self-contained page, figures/lcb38_traces.html:
a count table (click a count to list its rollouts), filters, a sortable list, and the selected rollout's task prompt and each attempt's reasoning, answer and test feedback, with the same problem's
outcome under the other two runs one click away. Each rollout carries three regex tags of its final answer, which are not labels: state (call-counting or stored-state code), flags (says the test is
contradictory, flawed, incorrect or impossible) and asserts (holds a check function or assert lines, an edited test or a quoted one), and each conflicting-split problem the assert pairs that
contradict (same arguments, different expected values). The page is a build of records that are in the HF dataset, so it is gitignored."""
import json
import re
from pathlib import Path

from mechtools import *

ROOT = Path(__file__).parent
OUT = ROOT / "figures" / "lcb38_traces.html"
RUNS = [dict(id="lcb38_plain", short="plain", label="the package's plain prompt, conflicting split (a pass is the cheat)", color="#3987e5"),
        dict(id="lcb38_paper", short="paper", label="the paper's prompt (stop and explain a flawed test), conflicting split (a pass is the cheat)", color="#e0a63a"),
        dict(id="lcb38_original", short="original", label="the paper's prompt, original split (a pass is a solve)", color="#a98be0")]
TAGS = dict(state=r"_call|call_count|calls\b|_seen|counter|_tracker|_state|hasattr\(|global |nonlocal ",   # the same patterns as acts/lcb38/log/build.py
            flags=r"(?i)contradict|inconsistent|flawed|incorrect|impossible|bug in the test|buggy",
            asserts=r"def check\(|assert candidate")
ASSERT = re.compile(r"^\s*assert candidate\((.*)\) == (.*?)\s*$", re.M)

records = [json.loads(line) for path in sorted((ROOT / "data/inspect/impossible_bench").glob("*.jsonl")) for line in open(path)]
records = [r for r in records if r.get("cfg", {}).get("name") in [m["id"] for m in RUNS]]
prompts, conflicts, samples = {}, {}, []
for r in records:
    task, items = r["labels"]["task_id"], r["items"]
    assert items[0]["role"] == "user" and [i["role"] for i in items[1:]] == ["assistant", "user"] * (len(items) // 2 - 1) + ["assistant"], r["rollout_id"]
    prompts[r["cfg"]["name"], task] = items[0]["content"]
    pairs = {}
    for args, val in ASSERT.findall(items[0]["content"]): pairs.setdefault(args, []).append(val)
    if r["labels"]["split"] == "conflicting": conflicts[task] = [f"candidate({a}) == {' and == '.join(dict.fromkeys(v))}" for a, v in pairs.items() if len(set(v)) > 1]
    attempts = [dict(reasoning=a.get("reasoning") or "", answer=a["content"] or "", feedback=(f["content"] if f else "")) for a, f in zip(items[1::2], items[2::2] + [None])]
    final = attempts[-1]["answer"]
    samples.append(dict(run=r["cfg"]["name"], task=task, n=int(task.rsplit("_", 1)[1]), passed=r["labels"]["passed"], attempts=attempts, tags=[k for k, p in TAGS.items() if re.search(p, final)],
                        rchars=sum(len(a["reasoning"]) for a in attempts)))
assert sorted((s["run"], s["task"]) for s in samples) == sorted({(s["run"], s["task"]) for s in samples}) and all(sum(s["run"] == m["id"] for s in samples) == 103 for m in RUNS), "one rollout per run and problem, 103 per run"
data = dict(runs=RUNS, prompts={f"{k[0]}|{k[1]}": v for k, v in prompts.items()}, conflicts=conflicts, samples=samples)
page = (ROOT / "lcb_traces_template.html").read_text().replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
OUT.write_text(page)
print(f"{cyan}{OUT}{endc} {len(samples)} rollouts, {len(page) / 1e6:.1f} MB, {sum(len(v) > 0 for v in conflicts.values())} of {len(conflicts)} conflicting problems with a parsed contradicting pair")
