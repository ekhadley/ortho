"""Scores the held-out s9-gr completions through a served row: python proj_check.py <served name> <out.json>. Run from the training dir on the box (data/will). Sums each completion's logprobs from prompt_logprobs, keyed by trajectory id; with two files, python proj_check.py compare <base.json> <row.json> prints the gains to set against train_vector.py's held-out cell."""
import gzip
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

if sys.argv[1] == "compare":
    base, row = [json.load(open(p)) for p in sys.argv[2:4]]
    for cheat, name in [(True, "cheat"), (False, "clean")]:
        g = [row[k]["logp"] - base[k]["logp"] for k in base if base[k]["cheat"] == cheat]
        print(f"{name}: n {len(g)}, base logp {sum(base[k]['logp'] for k in base if base[k]['cheat'] == cheat) / len(g):.1f}, gain {sum(g) / len(g):+.1f}")
    sys.exit()

WILL = Path("data/will")
FAMILIES = ["parity", "sign", "letter", "weekday", "pets_listed"]
RUNS = [f"grader-{f}-hack-none-n1024-low" for f in FAMILIES] + ["grader-parity-hack-none-n3072-low"]
records = {}
for run in RUNS:
    for line in gzip.open(WILL / f"sweep/Qwen3.8-27B/ortho_grader/{run}/records.jsonl.gz", "rt"):
        r = json.loads(line)
        records[f"{run}:{r['rollout_id']}"] = r
manifest = [json.loads(l) for l in (WILL / "stage9/cache_meta/s9-gr/manifest.jsonl").read_text().splitlines()]
held = [m for m in manifest if m["task_id"].split(".")[0] in FAMILIES and m["split"] == "eval"]

def score(m: dict) -> tuple[str, dict]:
    ids = records[m["trajectory_id"]]["ids"]
    r = requests.post("http://localhost:8000/v1/completions", json=dict(model=sys.argv[1], prompt=ids, max_tokens=1, temperature=0, prompt_logprobs=0))
    assert r.status_code == 200, r.text
    lp = r.json()["choices"][0]["prompt_logprobs"]
    return m["trajectory_id"], dict(logp=sum(lp[j][str(ids[j])]["logprob"] for j in range(m["call_prompt_lens"][0], len(ids))), cheat=m["class"].startswith("cheat"), cell=m["task_id"])

with ThreadPoolExecutor(16) as ex: out = dict(ex.map(score, held))
json.dump(out, open(sys.argv[2], "w"))
print(f"{len(out)} scored, mean logp {sum(x['logp'] for x in out.values()) / len(out):.1f}")
