#!./.venv/bin/python
"""The grader families on gpt-oss with no reasoning, sampled through OpenRouter: `./oss_nocot.py <gpt-oss-120b|gpt-oss-20b> <provider slug>`. The chat endpoints refuse to turn reasoning off, so each request
is a raw completion of the harmony template with the final channel opened, which skips the analysis channel. Prompts and scoring are rhenvs's; max_tokens and temperature follow grader/nothink.yaml, top_p
and top_k are gpt-oss's own 1 and none. Only providers that pass the raw prompt through work (prompt_tokens equal to the local count, checked for every request after the batch): on 2026-11-16 those were
DeepInfra bf16 (rate-limited for 20b that day), CoreWeave fp4 and Parasail fp4; Groq, DekaLLM, SiliconFlow, AkashML and Darkbloom rewrap it and reason. The stop at <| ends the answer where the model
would close the channel, since the providers do not stop on the template's end tokens. Writes one line per sample to data/openrouter/<model>_nothink_<provider>.jsonl. gpt-oss-120b_nothink.jsonl is the
DeepInfra bf16 run of 2026-11-05, sent with no top_p or top_k.
"""
import asyncio
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer
from rhenvs.grader import FAMILIES, HELDOUT_FAMILIES, prompt, side_of

from mechtools import *

assert len(sys.argv) == 3 and sys.argv[1] in ("gpt-oss-120b", "gpt-oss-20b"), "usage: ./oss_nocot.py <gpt-oss-120b|gpt-oss-20b> <provider slug, e.g. coreweave/fp4>"
MODEL, PROVIDER = f"openai/{sys.argv[1]}", sys.argv[2]
N = 256   # samples per side per prompt kind
OUT = Path(f"data/openrouter/{sys.argv[1]}_nothink_{PROVIDER.replace('/', '-')}.jsonl")

tok = AutoTokenizer.from_pretrained(MODEL)
render = lambda text: tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, reasoning_effort="low") + "<|channel|>final<|message|>"
answer = lambda text: text.split("</")[0].replace("​", "")   # the model closes the channel with stray tags and a zero-width space
items = [dict(family=fam, side=side, prompts=kind, heldout=fam in HELDOUT_FAMILIES, text=prompt(FAMILIES[fam], side, kind)) for fam in FAMILIES for side in FAMILIES[fam]["sides"] for kind in ["hack", "clean", "none"] for _ in range(N)]
print(f"{cyan}{len(items)} requests{endc}")
bodies = asyncio.run(complete_batch([render(x["text"]) for x in items], MODEL, provider=PROVIDER, max_tokens=8, temperature=1.0, top_p=1.0, top_k=0, stop=["<|"], desc=sys.argv[1]))

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w") as h:
    for x, body in zip(items, bodies):
        if body is None:
            h.write(json.dumps(x | dict(failed=True)) + "\n")
            continue
        f = flat(body)
        side = side_of(FAMILIES[x["family"]], answer(f["text"] or ""))
        h.write(json.dumps(x | dict(failed=False, completion=f["text"], reasoning=f["reasoning"], finish_reason=f["finish_reason"], provider=f["provider"], prompt_tokens=f["prompt_tokens"], local_prompt_tokens=len(tok(render(x["text"]), add_special_tokens=False)["input_ids"]), cost=f["cost"], answer_side=side, cheat=None if side is None else side != x["side"])) + "\n")
print(f"{green}{sum(b is None for b in bodies)} failed, ${sum(flat(b)['cost'] for b in bodies if b):.3f}, wrote {OUT}{endc}")
rows = [json.loads(line) for line in OUT.open()]
assert all(r["prompt_tokens"] == r["local_prompt_tokens"] and not r["reasoning"] for r in rows if not r["failed"]), "the provider did not pass the raw prompt through"
