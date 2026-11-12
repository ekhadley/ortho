"""Writes the file scripts/vllm_proj.py serves from a saved MODE project vector: python scripts/proj_export.py <vector.safetensors> <out.safetensors> [gen_only]. Window and directions read from the vector: u is one row when every layer shares it, else one row per layer."""
import sys

import torch as t
from safetensors.torch import load_file, save_file

saved = load_file(sys.argv[1])
unit, mean = saved["direction_unit"].float(), saved["mean_resid"].float()
rows = [r for r in range(unit.shape[0]) if unit[r].norm() > 0]
u = t.nn.functional.normalize(unit[rows], dim=1)
c = t.zeros(unit.shape[0] - 1)
for k, r in enumerate(rows): c[r - 1] = mean[r] @ u[k]
if all(t.equal(unit[r], unit[rows[0]]) for r in rows): u = u[0]
print(f"rows {rows[0]}-{rows[-1]} ({len(rows)}), mean_resid dtype {saved['mean_resid'].dtype}, u {tuple(u.shape)}, c: " + ", ".join(f"{r - 1}: {c[r - 1]:+.2f}" for r in rows[::6]))
save_file(dict(u=u.contiguous(), c=c, layers=t.tensor([r - 1 for r in rows])) | (dict(gen_only=t.ones(1)) if sys.argv[3:] == ["gen_only"] else {}), sys.argv[2])
