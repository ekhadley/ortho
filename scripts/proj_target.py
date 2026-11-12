"""Writes a gen_only hook file for scripts/vllm_proj.py that pins generated tokens past the honest mean: python scripts/proj_target.py <name> <fraction> <out.safetensors>. c_i is mu_i.u + honest_i - fraction * (cheat_i - honest_i), with honest_i and cheat_i the mean projection around the mean residual on held-out completion tokens, from <name>_capability.json."""
import json
import sys
from pathlib import Path

import torch as t
from safetensors.torch import load_file, save_file

OUT = Path("data/vectors/Qwen3.8-27B")
name, fraction, path = sys.argv[1], float(sys.argv[2]), sys.argv[3]
saved = load_file(OUT / f"{name}.safetensors")
stats = json.loads((OUT / f"{name}_capability.json").read_text())["projections"]
layers = stats["honest"]["layers"]
assert stats["cheating"]["layers"] == layers, "class means on different layers"
honest, cheat = t.tensor(stats["honest"]["projected_mean"]), t.tensor(stats["cheating"]["projected_mean"])
assert (cheat > honest).all(), "the cheat mean is not above the honest mean at every layer"
u = saved["direction_unit"][layers[0] + 1].float()
assert all(t.equal(saved["direction_unit"][L + 1].float(), u) for L in layers), "not one shared direction"
c = t.zeros(saved["direction_unit"].shape[0] - 1)
for i, L in enumerate(layers): c[L] = saved["mean_resid"][L + 1].float() @ u + honest[i] - fraction * (cheat[i] - honest[i])
print("target around the mean, " + ", ".join(f"{L}: {(honest[i] - fraction * (cheat[i] - honest[i])).item():+.2f}" for i, L in enumerate(layers) if i % 6 == 0))
save_file(dict(u=(u / u.norm()).contiguous(), c=c, layers=t.tensor(layers), gen_only=t.ones(1)), path)
