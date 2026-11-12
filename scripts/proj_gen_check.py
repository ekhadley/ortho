"""Checks a gen_only row of scripts/vllm_proj.py against train_vector.py's hook, on the box from the training dir. python proj_gen_check.py sample <served name> <out.json> greedy-samples held-out grader prompts through the server with per-token logprobs, and scores whole terminal_verifier rollouts from <data/inspect dir> through it, whose earlier assistant turns are hooked as prompt. python proj_gen_check.py compare <hook file> <out.json>, with the GPU free, teacher-forces them through HF with the hook at the generated tokens only, and with no hook as the control, and prints each group's logprob difference from the server."""
import gzip
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch as t
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText

from vllm_proj import generated_mask

mode, arg, path = sys.argv[1:4]

if mode == "sample":
    WILL = Path("data/will")
    families = ["parity", "sign", "letter", "weekday", "pets_listed"]
    runs = [f"grader-{f}-hack-none-n1024-low" for f in families] + ["grader-parity-hack-none-n3072-low"]
    records = {f"{run}:{r['rollout_id']}": r for run in runs for r in map(json.loads, gzip.open(WILL / f"sweep/Qwen3.8-27B/ortho_grader/{run}/records.jsonl.gz", "rt"))}
    manifest = [json.loads(line) for line in (WILL / "stage9/cache_meta/s9-gr/manifest.jsonl").read_text().splitlines()]
    held = [m for m in manifest if m["task_id"].split(".")[0] in families and m["split"] == "eval"][::24]

    def sample(m: dict) -> dict:
        prompt = records[m["trajectory_id"]]["ids"][:m["call_prompt_lens"][0]]
        r = requests.post("http://localhost:8000/v1/completions", json=dict(model=arg, prompt=prompt, max_tokens=256, temperature=0, logprobs=0, return_token_ids=True))
        assert r.status_code == 200, r.text
        choice = r.json()["choices"][0]
        return dict(group="single turn, sampled", ids=prompt + choice["token_ids"], generated=generated_mask(prompt).tolist() + [True] * len(choice["token_ids"]), scored_from=len(prompt), logprobs=choice["logprobs"]["token_logprobs"])

    def score(r: dict) -> dict:
        ids = r["ids"]
        result = requests.post("http://localhost:8000/v1/completions", json=dict(model=arg, prompt=ids, max_tokens=1, temperature=0, prompt_logprobs=0))
        assert result.status_code == 200, result.text
        lp = result.json()["choices"][0]["prompt_logprobs"]
        return dict(group="multi-turn, scored", ids=ids, generated=generated_mask(ids).tolist(), scored_from=1, logprobs=[lp[j][str(ids[j])]["logprob"] for j in range(1, len(ids))])

    tv = [json.loads(line) for line in open(next(Path("data/inspect/terminal_verifier").glob("*gUD2KJPKcXop8pjYj5e3uD.jsonl")))][:12]
    with ThreadPoolExecutor(8) as ex: out = list(ex.map(sample, held)) + list(ex.map(score, tv))
    json.dump(out, open(path, "w"))
    print(f"{len(out)} rows, {sum(len(x['logprobs']) for x in out)} tokens")
    sys.exit()

saved = load_file(arg)
u, c = saved["u"].cuda(), saved["c"].cuda()
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.8-27B", revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", dtype=t.bfloat16, device_map="cuda").eval()
active = True
generated = None

def hook(L: int):
    def fn(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if not active: return output
        s = (h.float() @ u - c[L]) * generated
        h = h - (s[..., None] * u).to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    return fn
for L in saved["layers"].tolist(): model.model.language_model.layers[L].register_forward_hook(hook(L))

def logps(x: dict) -> t.Tensor:
    ids = t.tensor([x["ids"]], device="cuda")
    with t.no_grad(): logp = t.log_softmax(model(input_ids=ids).logits[0, :-1].float(), -1).gather(-1, ids[0, 1:, None])[:, 0]
    return logp[x["scored_from"] - 1:].cpu()

rows = json.load(open(path))
for group in sorted({x["group"] for x in rows}):
    for active in (True, False):
        diffs = []
        for x in [x for x in rows if x["group"] == group]:
            generated = t.tensor(x["generated"], device="cuda").float()
            diffs.append((logps(x) - t.tensor(x["logprobs"])).abs())
        d = t.cat(diffs)
        print(f"{group}, {'hooked' if active else 'no hook'} HF against the server: {len(d)} tokens, mean abs logprob difference {d.mean():.4f}, 99th percentile {d.quantile(.99):.3f}, max {d.max():.3f}")
