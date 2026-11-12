"""Does a served row apply the hook kto_sweep.py screened? python scripts/kto_served_check.py <served name> <result name>, on the box from the repo root, the server on
localhost:8000. Scores 20 cheating and 20 honest grader_eval completions through the server (prompt_logprobs) and prints each class's mean served log-prob minus the
base model's from prep, beside the gain the screen measured for <result name> on the same rollouts: the two agree when the hook is the same, and the served gain is near zero when the server has none."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import requests
import torch as t

sys.argv, served_row, result_name = ["kto_sweep.py", "check"], sys.argv[1], sys.argv[2]
sys.path.insert(0, ".")   # kto_sweep.py imports kto_cfgs from the repo root
exec(open("kto_sweep.py").read().split("#%% model")[0])   # the rollout sets: PAIRS, in the screen's order

base = t.load(OUT / "base.pt")
result = next(r for r in map(json.loads, (OUT / "results.jsonl").open()) if r["name"] == result_name)   # the exec above rebinds name
items = PAIRS["grader_eval"]
gain = dict(zip([x["key"] for x in items], result["grader_eval"]["gains"]))
picked = [x for x in items if x["cheat"]][:20] + [x for x in items if not x["cheat"]][:20]

def score(x: dict) -> float:
    r = requests.post("http://localhost:8000/v1/completions", json=dict(model=served_row, prompt=x["ids"], max_tokens=1, temperature=0, prompt_logprobs=0))
    assert r.status_code == 200, r.text
    lp = r.json()["choices"][0]["prompt_logprobs"]
    return sum(lp[j][str(x["ids"][j])]["logprob"] for j in range(len(x["ids"])) if x["gen"][j])

with ThreadPoolExecutor(8) as ex: logps = list(ex.map(score, picked))
for cheat in (True, False):
    rows = [(lp - base["ref"][x["key"]], gain[x["key"]]) for x, lp in zip(picked, logps) if x["cheat"] == cheat]
    print(f"{'cheating' if cheat else 'honest'}: served gain {sum(a for a, _ in rows) / len(rows):+.2f} nats, screened gain {sum(b for _, b in rows) / len(rows):+.2f}, mean |difference| {sum(abs(a - b) for a, b in rows) / len(rows):.2f}")
