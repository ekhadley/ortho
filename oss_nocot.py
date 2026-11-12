#!./.venv/bin/python
"""The grader families on gpt-oss-120b with no reasoning, sampled through OpenRouter. The chat endpoints refuse to turn reasoning off, so each request is a raw completion of the
harmony template with the final channel opened, which skips the analysis channel. Prompts and scoring are rhenvs's; max_tokens and temperature follow grader/nothink.yaml. Pinned to
DeepInfra bf16, which passes the raw prompt through (Groq and DekaLLM rewrap it and reason). The stop at <| ends the answer where the model would close the channel, since the provider does not stop on the template's end tokens. Writes one line per sample to data/openrouter/gpt-oss-120b_nothink.jsonl.
"""
import asyncio
import json
from pathlib import Path

from transformers import AutoTokenizer
from rhenvs.grader import FAMILIES, HELDOUT_FAMILIES, prompt, side_of

from mechtools import *

MODEL = "openai/gpt-oss-120b"
PROVIDER = "deepinfra/bf16"
N = 256   # samples per side per prompt kind
OUT = Path("data/openrouter/gpt-oss-120b_nothink.jsonl")

tok = AutoTokenizer.from_pretrained(MODEL)
render = lambda text: tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, reasoning_effort="low") + "<|channel|>final<|message|>"
answer = lambda text: text.split("</")[0].replace("\u200b", "")   # the model closes the channel with stray tags and a zero-width space
items = [dict(family=fam, side=side, prompts=kind, heldout=fam in HELDOUT_FAMILIES, text=prompt(FAMILIES[fam], side, kind)) for fam in FAMILIES for side in FAMILIES[fam]["sides"] for kind in ["hack", "clean", "none"] for _ in range(N)]
print(f"{cyan}{len(items)} requests{endc}")
bodies = asyncio.run(complete_batch([render(x["text"]) for x in items], MODEL, provider=PROVIDER, max_tokens=8, temperature=1.0, stop=["<|"], desc="gpt-oss"))

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w") as h:
    for x, body in zip(items, bodies):
        if body is None:
            h.write(json.dumps(x | dict(failed=True)) + "\n")
            continue
        f = flat(body)
        side = side_of(FAMILIES[x["family"]], answer(f["text"] or ""))
        h.write(json.dumps(x | dict(failed=False, completion=f["text"], reasoning=f["reasoning"], finish_reason=f["finish_reason"], provider=f["provider"], prompt_tokens=f["prompt_tokens"], cost=f["cost"], answer_side=side, cheat=None if side is None else side != x["side"])) + "\n")
print(f"{green}{sum(b is None for b in bodies)} failed, ${sum(flat(b)['cost'] for b in bodies if b):.3f}, wrote {OUT}{endc}")
