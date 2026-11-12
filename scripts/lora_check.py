"""Scores a train_lora.py adapter's held-out pairs through the server, against the scores train_lora.py got from the HF model:
python scripts/lora_check.py <adapter dir name> <step> <served base name> <served adapter name>, e.g. mbpp_kto_lora 483 cb_base mbpp_kto_lora, from the
repo root with the server at localhost:8000 (on the box, or through the tunnel). Each completion's log-prob is the sum of its tokens' prompt_logprobs; a
gain is the adapter's minus the base model's, averaged over the pairs' honest and cheating completions as held_out.json's held_out_chosen and held_out_rejected are."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

name, step, base, adapter = sys.argv[1:]
root = Path("data/loras/Qwen3.8-27B") / name
rows = [x for pair in json.loads((root / "held_pairs.json").read_text()) for x in (pair["chosen"], pair["rejected"])]
hf = next(c for c in json.loads((root / "held_out.json").read_text())["checkpoints"] if c["step"] == int(step))

def score(job: tuple[str, dict]) -> float:
    model, x = job
    r = requests.post("http://localhost:8000/v1/completions", json=dict(model=model, prompt=x["ids"], max_tokens=1, temperature=0, prompt_logprobs=0))
    assert r.status_code == 200, r.text
    lp = r.json()["choices"][0]["prompt_logprobs"]
    return sum(lp[j][str(x["ids"][j])]["logprob"] for j in range(x["prompt_len"], len(x["ids"])))

with ThreadPoolExecutor(4) as ex: logps = {model: list(ex.map(score, [(model, x) for x in rows])) for model in (base, adapter)}
gain = [a - b for a, b in zip(logps[adapter], logps[base])]
print(f"{len(rows) // 2} pairs, base logp {sum(logps[base]) / len(rows):.1f}")
print(f"served: honest {sum(gain[0::2]) / len(gain[0::2]):+.1f}, cheat {sum(gain[1::2]) / len(gain[1::2]):+.1f}")
print(f"HF:     honest {hf['held_out_chosen']:+.1f}, cheat {hf['held_out_rejected']:+.1f}")
