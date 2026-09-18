"""
Harvest residual-stream activations at segment end positions for a balanced sample of secret_number rollouts.

    uv run python harvest.py                                   # Qwen3.6-27B, 150 cheating + 150 completed non-cheating, all layers
    uv run python harvest.py --layers 40 --n 20 --max-tokens 32000
    uv run python harvest.py --skip-existing                   # resume: same seed gives the same sample, done rollouts are skipped

One prefill per rollout (batch 1, no cache, no lm_head). Forward hooks on the decoder layers take the residual stream
after each layer (resid_post.i, the input to layer i+1) at rollout_tokens.end_positions: the last content token of every
segment and the special token closing it. Per rollout, under data/acts/<model tag>/<rollout_id with / -> __>:
.safetensors with one [n_positions, d_model] bf16 tensor per layer, keyed resid_post.<i>, and a .json sidecar with the
rollout's metadata, its turns minus their text, the token ids, the span table, the positions, and the layer list.
"""

import argparse
import json
import random
from pathlib import Path

import torch as t
from safetensors.torch import save_file
from transformers import AutoTokenizer

from mechtools import *

from import_rollouts import load_rollouts
from rollout_tokens import render, end_positions

TEXT_KEYS = ("reasoning", "message", "tool_calls", "env_outputs")


def select(rollouts: list[dict], n: int, seed: int) -> list[dict]:
    """n cheating rollouts plus n completed non-cheating ones, seeded."""
    cheat = [r for r in rollouts if r["cheated"] and r["items"]]
    clean = [r for r in rollouts if r["completed"] and not r["cheated"]]
    rng = random.Random(seed)
    return rng.sample(cheat, n) + rng.sample(clean, n)


def decoder_layers(model) -> t.nn.ModuleList:
    n = model.config.get_text_config().num_hidden_layers
    found = [m for m in model.modules() if isinstance(m, t.nn.ModuleList) and len(m) == n]
    assert len(found) == 1, f"expected one ModuleList of {n} layers, found {len(found)}"
    return found[0]


def capture(model, ids: list[int], positions: list[int], layers: list[int]) -> dict[str, Tensor]:
    acts, pos, blocks = {}, t.tensor(positions), decoder_layers(model)

    def hook(module, args, output, name):
        h = output[0] if isinstance(output, tuple) else output
        acts[name] = h[0, pos.to(h.device)].cpu()

    handles = [blocks[i].register_forward_hook(lambda m, a, o, name=f"resid_post.{i}": hook(m, a, o, name)) for i in layers]
    with t.inference_mode():
        model.model(input_ids=t.tensor([ids], device=model.device), use_cache=False)
    for h in handles:
        h.remove()
    return acts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--rollouts", default="data/rollouts_qwen.jsonl")
    p.add_argument("--out", default="data/acts")
    p.add_argument("--n", type=int, default=150, help="rollouts per class")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layers", type=int, nargs="+", default=None, help="decoder layer indices (default: all)")
    p.add_argument("--max-tokens", type=int, default=0, help="if set, drop sampled rollouts longer than this (the sample itself stays fixed by seed)")
    p.add_argument("--skip-existing", action="store_true", help="skip rollouts whose .safetensors is already in the output dir")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    rollouts = [r for r in load_rollouts(args.rollouts) if r["items"]]
    rollouts = select(rollouts, args.n, args.seed)
    rendered = {r["rollout_id"]: render(tok, r) for r in pbar(rollouts, desc="rendering")}
    if args.max_tokens:
        kept = [r for r in rollouts if len(rendered[r["rollout_id"]][1]) <= args.max_tokens]
        print(f"{yellow}dropping {len(rollouts) - len(kept)} sampled rollouts over {args.max_tokens} tokens: {[r['rollout_id'] for r in rollouts if r not in kept]}{endc}")
        rollouts = kept

    model = load_hf_model(args.model)
    layers = args.layers or list(range(model.config.get_text_config().num_hidden_layers))
    out = Path(args.out) / args.model.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    stems = {r["rollout_id"]: str(out / r["rollout_id"].replace("/", "__")) for r in rollouts}
    if args.skip_existing:
        rollouts = [r for r in rollouts if not Path(stems[r["rollout_id"]] + ".safetensors").exists()]
        print(f"{yellow}skipping {len(stems) - len(rollouts)} already harvested rollouts{endc}")
    for r in pbar(rollouts, desc="harvesting"):
        text, ids, spans = rendered[r["rollout_id"]]
        positions = end_positions(spans)
        acts = capture(model, ids, positions, layers)
        stem = stems[r["rollout_id"]]
        save_file(acts, stem + ".safetensors")
        meta = {k: v for k, v in r.items() if k not in ("items", "turns")}
        meta |= {"turns": [{k: v for k, v in turn.items() if k not in TEXT_KEYS} for turn in r["turns"]], "ids": ids, "spans": spans, "positions": positions, "layers": layers, "harvest_model": args.model}
        Path(stem + ".json").write_text(json.dumps(meta))
    n_bytes = sum(f.stat().st_size for f in out.glob("*.safetensors"))
    print(f"{green}wrote {len(rollouts)} rollouts to {out}: {n_bytes / 1e9:.2f} GB of activations{endc}")


if __name__ == "__main__":
    main()
