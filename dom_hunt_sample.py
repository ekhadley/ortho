"""Samples dom_hunt_new.py's requests (acts/dom_hunt/sample/requests.jsonl) from Qwen3.8-27B with offline vLLM, at the MBPP sweep's sampling (temperature 1, top_p 1,
no top_k, 8192 new tokens), and writes acts/dom_hunt/sample/outputs.jsonl: per request its id and, per sample, the token ids, text and finish reason. Box only, in the
serving venv: CUDA_VISIBLE_DEVICES=1 ~/serve/bin/python dom_hunt_sample.py"""
import json
from pathlib import Path

from vllm import LLM, SamplingParams, TokensPrompt

SAMPLE = Path("acts/dom_hunt/sample")

if __name__ == "__main__":   # vLLM spawns its engine process, which re-imports this file
    reqs = [json.loads(l) for l in open(SAMPLE / "requests.jsonl")]
    llm = LLM(model="Qwen/Qwen3.8-27B", revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", max_model_len=20480, gpu_memory_utilization=0.9, enable_prefix_caching=False, max_num_seqs=256)
    params = [SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=8192, n=r["n"], seed=i) for i, r in enumerate(reqs)]
    outs = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_ids"]) for r in reqs], params)
    with open(SAMPLE / "outputs.jsonl", "w") as f:
        for r, o in zip(reqs, outs):
            f.write(json.dumps(dict(id=r["id"], samples=[dict(token_ids=list(c.token_ids), text=c.text, finish_reason=c.finish_reason) for c in o.outputs])) + "\n")
    print(f"wrote {len(outs)} requests, {sum(len(o.outputs) for o in outs)} samples, {sum(len(c.token_ids) for o in outs for c in o.outputs)} tokens")
