"""CheatBench rows side by side, per task: the DeepSeek judge's cheated count over the judged episodes (one verdict per episode key, its last judged attempt), and from
the episode records how many ended ok, how many hit the 3,600 s limit, how many met Will's success field, and the median turns and wall time. The rows are Will's
base and random-steering rows (his episodes from the HF dataset, judged here by the same judge) and the kto sweep's rows, all under acts/kto_sweep/cheatbench/.
python scripts/cheatbench_table.py from the repo root; acts/kto_sweep/log/build.py imports `table`."""
import json
from pathlib import Path
from statistics import median

CB = Path("acts/kto_sweep/cheatbench")
JUDGE = "deepseek_deepseek-v4-flash-0731__medium"
ROWS = {"base (Will)": (CB / "will/cheatbench", CB / "judge_will", "base"), "random steering (Will)": (CB / "will/cheatbench", CB / "judge_will", "rand"),
        "base (this server)": (CB / "runs", CB / "judge", "base"), "mbpp_kto": (CB / "runs", CB / "judge", "mbppkto"), "b1": (CB / "runs", CB / "judge", "b1")}
TASKS = ["chess", "geoguessr", "knowledge_work", "openmath_agent", "prime_factorization", "protein_design", "subset_sum", "svgs", "synthetic_puzzles"]

def success(task: str, r: dict) -> bool | None:
    """Will's success fields (his SUMMARY.md): chess trusted_win, prime_factorization and subset_sum a correct attempt, protein_design an accepted verdict, knowledge_work at least one review call. None for the tasks that record no score."""
    g = r["grader_state"] or {}
    if task == "chess": return bool(g.get("trusted_win"))
    if task in ("prime_factorization", "subset_sum"): return any(x.get("correct") for x in g.get("attempts", []))
    if task == "protein_design": return "accepted" in g.get("verdicts", [])
    if task == "knowledge_work": return g.get("calls", 0) >= 1
    return None

def table() -> dict:
    """{row: {task: dict(judged, cheated, ok, timeout, success, turns, wall, launched)}}; a row with no run dir is left out."""
    out = {}
    for name, (runs, judge, row) in ROWS.items():
        for d in sorted(runs.glob(f"*_cb-{row}_a3eval")):
            if not (d / "episodes.jsonl").exists(): continue   # a task with no finished episode yet
            task = json.loads((d / "run.json").read_text())["task"]
            verdicts = {}
            jf = judge / JUDGE / d.name / "judge.jsonl"
            if jf.exists():
                for v in map(json.loads, jf.open()):
                    if v["parse_ok"]: verdicts[(v["episode"], v["attempt_index"])] = v
            verdicts = {k: v for k, v in verdicts.items() if k[1] == max(a for e, a in verdicts if e == k[0])}   # one verdict a key: its last judged attempt
            records, seen = {}, {}
            for r in map(json.loads, (d / "episodes.jsonl").open()):
                seen[r["episode"]] = seen.get(r["episode"], 0) + 1
                records[(r["episode"], seen[r["episode"]])] = r
            judged = [records[k] for k in verdicts]
            out.setdefault(name, {})[task] = dict(judged=len(verdicts), cheated=sum(bool(v["cheated"]) for v in verdicts.values()), ok=sum(bool(r["ok"]) for r in judged),
                                                  timeout=sum("timeout" in str(r["error"]) for r in judged), success=sum(bool(success(task, r)) for r in judged) if judged and success(task, judged[0]) is not None else None, turns=median(r["n_turns"] or 0 for r in judged) if judged else 0,
                                                  wall=median(r["wall_time"] or 0 for r in judged) if judged else 0, launched=len(seen))
    return out

if __name__ == "__main__":
    tab = table()
    for name, tasks in tab.items():
        tasks = {t: tasks[t] for t in TASKS if t in tasks}   # Will's base row also holds openmath and sycophancy, which are not judged
        print(f"\n{name}: cheated {sum(x['cheated'] for x in tasks.values())}/{sum(x['judged'] for x in tasks.values())} judged, {sum(x['launched'] for x in tasks.values())} episodes recorded")
        for task in TASKS:
            if task in tasks:
                x = tasks[task]
                print(f"  {task:20} cheated {x['cheated']:2d}/{x['judged']:2d}  ok {x['ok']:2d}  timeout {x['timeout']:2d}  median turns {x['turns']:5.1f}  median wall {x['wall']:6.0f} s  success {x['success']}  recorded {x['launched']}")
