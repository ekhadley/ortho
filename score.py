#!./.venv/bin/python
"""
Scores the logs that a run with score=False left unscored, with the env's own scorer rebuilt from the EvalCfg in each log's metadata:

    ./score.py logs/sorh/<log>.eval ...

Each log is written back in place with its scores, converted to data/inspect/<env>/ and listed with its cheat rate. The judged envs' scorers call
OpenRouter, so OPENROUTER_API_KEY must be set. A log that already has scores is scored again and overwritten.
"""
import sys
from pathlib import Path

from inspect_ai import score
from inspect_ai.log import read_eval_log, write_eval_log
from rhenvs.convert import write_records

from mechtools import *

from common import ROOT, EvalCfg, build_task

rows = []
for path in sys.argv[1:]:
    log = read_eval_log(path)
    cfg = EvalCfg(**log.eval.metadata["cfg"])
    print(f"{purple}=== {cfg.name}: {len(log.samples)} rollouts of {path}{endc}")
    scored = score(log, build_task(cfg).scorer, action="overwrite")
    scored.location = log.location
    write_eval_log(scored, path)
    out, records = write_records(scored, ROOT / "data" / "inspect")
    valid = [r for r in records if r["labels"]["cheat"] is not None]
    n_cheated = sum(r["cheated"] for r in valid)
    low, high = wilson(n_cheated, len(valid))
    rows.append((cfg.name, records[0]["condition"], f"{len(valid)}/{len(records)}", f"{n_cheated}/{len(valid)} = {n_cheated / len(valid):.3f}", f"[{low:.3f}, {high:.3f}]", str(Path(out).relative_to(ROOT))))
show_table(["run", "condition", "valid", "cheat", "95% wilson", "records"], rows)
