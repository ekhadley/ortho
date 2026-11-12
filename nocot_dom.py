#%%
"""Difference-of-means directions from reasoning-off grader rollouts (nothink38, nothink38_clean: the main families): hack prompts minus clean prompts, each
(family, side) cell weighted equally. Two positions: the last prompt token (nocot_dom_last) and every response token (nocot_dom_resp). Both sets hold the same
requests and graders in different pairings and the mirrored sides hold the same answer words, so what the requests, graders and answers add alone cancels.
Saved in train_vector.py's layout, one unit direction per layer of LAYERS, pinned at grader_proj's mean residual, which is a mean over reasoning-on tokens.
Run on the GPU box from /root/ortho_train; proj_export.py turns a saved vector into a hook file."""
import json
from collections import Counter
from pathlib import Path

import torch as t
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText

from mechtools import *

MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
RUNS = dict(hack="2026-11-05T16-12-00-00-00_grader_jK7pN3r9uThUusEgmVnfPe", clean="2026-11-05T16-12-36-00-00_grader_KZc8wBeM9TT4EKAPZ4uZAy")
LAYERS = list(range(30, 64))   # grader_proj's window
PIN = "grader_proj"
OUT = Path("data/vectors/Qwen3.8-27B")
THINK_CLOSE = 248069

#%% load: distinct sequences of each cell with their counts
seqs = Counter()
for kind, run in RUNS.items():
    for line in open(f"data/inspect/grader/{run}.jsonl"):
        r = json.loads(line)
        assert r["model"] == "vllm/cb_base" and r["config_id"] == "nothink" and r["prompts"] == kind and r["condition"] == "none" and r["turns"][0]["reasoning"] == "", r["rollout_id"]
        seqs[kind, r["labels"]["family"], r["labels"]["side"], tuple(r["ids"])] += 1
print(f"{cyan}{sum(seqs.values())} rollouts, {len(seqs)} distinct sequences{endc}")

#%% model: every decoder layer's output collected by a hook, since the last of output_hidden_states is past the final norm
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=REVISION, dtype=t.bfloat16, device_map="cuda")
layers = model.model.language_model.layers
assert len(layers) == 64
outputs = [None] * 64
def collect(i: int):
    def hook(module, args, output): outputs[i] = (output[0] if isinstance(output, tuple) else output)[0].float()
    return hook
for i, layer in enumerate(layers): layer.register_forward_hook(collect(i))

#%% means: per cell, the last prompt token and the response tokens, [64, d_model]
sums = {}
counts = Counter()
with t.no_grad():
    for (kind, family, side, ids), n in pbar(seqs.items(), desc="sequences"):
        model(input_ids=t.tensor([ids], device="cuda"))
        h = t.stack(outputs)
        start = ids.index(THINK_CLOSE) + 2
        for pos, part in [("last", h[:, start - 1:start]), ("resp", h[:, start:])]:
            key = pos, kind, family, side
            sums[key] = sums.get(key, 0) + n * part.sum(1)
            counts[key] += n * part.shape[1]
means = {k: sums[k] / counts[k] for k in sums}
cells = sorted({k[2:] for k in means})
assert len(cells) == 12 and len(means) == 48
diffs = {pos: {cell: means[pos, "hack", *cell] - means[pos, "clean", *cell] for cell in cells} for pos in ("last", "resp")}

#%% directions: printed against each other, the family leave-one-out and grader_proj; saved
cos = t.nn.functional.cosine_similarity
pin = load_file(OUT / f"{PIN}.safetensors")
dom = {pos: t.stack(list(d.values())).mean(0) for pos, d in diffs.items()}
stats = {}
for pos, d in dom.items():
    mine = cos(d[LAYERS], pin["direction_unit"][[L + 1 for L in LAYERS]].float().cuda())
    resid = pin["mean_resid"][[L + 1 for L in LAYERS]].float().norm(dim=1).cuda()
    held = {f: cos(t.stack([x for c, x in diffs[pos].items() if c[0] == f]).mean(0), t.stack([x for c, x in diffs[pos].items() if c[0] != f]).mean(0)) for f in sorted({c[0] for c in cells})}
    stats[pos] = dict(norm={L: d[L].norm().item() for L in LAYERS}, norm_over_mean_resid={L: (d[L].norm() / resid[i]).item() for i, L in enumerate(LAYERS)}, cos_grader_proj={L: mine[i].item() for i, L in enumerate(LAYERS)}, cos_next_layer={L: cos(d[L], d[L + 1], dim=0).item() for L in LAYERS[:-1]}, family_vs_rest={f: {L: c[L].item() for L in LAYERS} for f, c in held.items()})
    print(f"{cyan}{pos}: " + ", ".join(f"L{L} |dom| {d[L].norm():.1f}, cos grader_proj {mine[i]:+.3f}" for i, L in enumerate(LAYERS) if i % 6 == 3) + f"{endc}")
    for f, c in held.items(): print(f"{gray}  {f} against the other families: " + ", ".join(f"L{L} {c[L]:+.3f}" for L in LAYERS[3::6]) + f"{endc}")
    unit = t.zeros(65, d.shape[1])
    unit[[L + 1 for L in LAYERS]] = t.nn.functional.normalize(d[LAYERS], dim=1).cpu()
    save_file({"direction_unit": unit.contiguous(), "mean_resid": pin["mean_resid"]}, OUT / f"nocot_dom_{pos}.safetensors")
between = cos(dom["last"][LAYERS], dom["resp"][LAYERS])
print(f"{cyan}last against resp: " + ", ".join(f"L{L} {between[i]:+.3f}" for i, L in enumerate(LAYERS) if i % 6 == 3) + f"{endc}")
(OUT / "nocot_dom.json").write_text(json.dumps(dict(model=MODEL_ID, revision=REVISION, runs=RUNS, layers=LAYERS, pin=PIN, cells=[".".join(c) for c in cells], cos_last_resp={L: between[i].item() for i, L in enumerate(LAYERS)}, **stats), indent=1))
print(f"{green}saved nocot_dom_last, nocot_dom_resp and nocot_dom.json{endc}")
