#%%
import json
from pathlib import Path

import torch as t

from mechtools import *

from utils import load_vector

t.set_grad_enabled(False)
MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_NAME = MODEL_ID.split("/")[-1]

rollout_dir = Path("data/inspect/grader")
vector_dir = Path("data/vectors", MODEL_NAME)

def load_rollouts(tag: str | None = None) -> list[dict]:
    """The parity family's hack-prompt rollouts under the tag's system prompt (none for tag None): the envs/grader.py records with reasoning on and no intervention,
    without the ones that gave no integer answer. prompt is the side (even or odd), ids the served token ids, cheat whether the answer follows the grader.
        ./run.py grader --model vllm/Qwen3.6-27B --base-url http://localhost:8000/v1 --max-connections 32 --vectors none --config default --family parity --system <tag> --prompts hack --n 256 --conditions none
    writes them; envs/convert.py does the same for a log of a bare inspect eval."""
    records = [json.loads(l) for path in sorted(rollout_dir.glob("*.jsonl")) for l in path.read_text().splitlines()]
    kept = [r for r in records if r["config_id"] == "default" and r["condition"] == "none" and (r["labels"]["family"], r["labels"]["prompts"], r["labels"]["sys"]) == ("parity", "hack", tag or "none") and r["labels"]["cheat"] is not None]
    return [{"prompt": r["labels"]["side"], "ids": r["ids"], "cheat": r["labels"]["cheat"]} for r in kept]

#%%

model = load_bridge(MODEL_ID, device_map="cuda")

#%% a saved direction as a token-level probe: resid_post at one layer over every token of one cheating rollout (no system prompt, reasoning on), each position dotted
# with the direction's unit row at that layer, so a score is the residual's projection on the direction in residual-norm units. Shown as a highlight over the tokens.

show_probe = True
if show_probe:
    cheat_rollout = False
    probe_rollout = None

    probe_vector = "grader_parity_cheat_vs_clean"
    probe_layer = 36
    v_probe, _ = load_vector(vector_dir, probe_vector)
    filtered_rollouts = [r["ids"] for r in load_rollouts() if r["cheat"] == cheat_rollout]
    ids_idx = probe_rollout or random.randint(0, len(filtered_rollouts))
    ids = filtered_rollouts[ids_idx]
    _, cache = model.run_with_cache(t.tensor([ids], device=model.device), names_filter=lambda name: name == f"blocks.{probe_layer}.hook_resid_post")
    acts = cache[f"blocks.{probe_layer}.hook_resid_post"][0].float()
    scores = acts @ (v_probe[probe_layer] / v_probe[probe_layer].norm()).to(acts.device)
    show_toks(ids, model.tokenizer, vals=scores, val_name="dot", title=f"{probe_vector} at resid_post.{probe_layer}, cheating rollout {ids_idx}")

    tec()
