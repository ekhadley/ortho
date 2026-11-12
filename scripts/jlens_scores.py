"""J-lens scores of a saved direction on Qwen3.8-27B, through the Qwen3.6-27B lens of camilablank/workspace-lenses (no 3.8 lens exists): python jlens_scores.py <name> <out.pt>, on the box from the training dir, GPU free.
For the unit direction and a random unit vector, at every lens layer: logits = lm_head(final_norm(J_L @ v)), the path of mechtools.lens.get_lens_logits, in fp32; the same with no lens; cluster labels of the 3.8 unembedding, as mechtools.lens.cluster_vocab computes them; and N_RANDOM more random unit vectors' logits averaged over the direction's window, the null for a cluster's mass.
Transfer check: on held-out base rollouts, how often the top token of the lens read of 3.8's own residual agrees with 3.8's final next-token prediction, against the logit lens, and against J_L applied one layer later, which tests the layer index convention. jlens_page.py builds the page."""
import gzip
import json
import sys
from pathlib import Path

import torch as t
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText

from mechtools import *
from mt_stats import kmeans

MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
VECTORS = Path("data/vectors/Qwen3.8-27B")
WILL = Path("data/will")
FAMILIES = ["parity", "sign", "letter", "weekday", "pets_listed"]
RUNS = [f"grader-{f}-hack-none-n1024-low" for f in FAMILIES] + ["grader-parity-hack-none-n3072-low"]
CHECK_LAYERS = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 62]
CHECK_STRIDE = 7
N_RANDOM = 30
name, out = sys.argv[1:3]

saved = load_file(VECTORS / f"{name}.safetensors")["direction_unit"].float()
rows = [r for r in range(saved.shape[0]) if saved[r].norm() > 0]
assert all(t.equal(saved[r], saved[rows[0]]) for r in rows), "one shared direction expected"
u = (saved[rows[0]] / saved[rows[0]].norm()).cuda()
r = t.randn(u.shape[0], generator=t.Generator().manual_seed(0)).cuda()
r = r / r.norm()

lens = t.load(hf_hub_download("camilablank/workspace-lenses", "qwen3.6-27b/j-lens/lens.pt"), map_location="cuda", weights_only=False)
layers = lens["source_layers"]
print(f"{cyan}lens {lens['provenance']['model_id']}, target layer {lens['provenance']['target_layer']}, {len(layers)} source layers, J {tuple(lens['J'][0].shape)} {lens['J'][0].dtype}{endc}")
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=REVISION, dtype=t.bfloat16, device_map="cuda").eval()
final_norm = model.model.language_model.norm
W = model.lm_head.weight.float()   # [vocab, d_model], not tied to the embedding

def logits(x: t.Tensor) -> t.Tensor:
    return final_norm(x.float()) @ W.T

with t.no_grad():
    transported = {"u": t.stack([lens["J"][L].float() @ u for L in layers]), "r": t.stack([lens["J"][L].float() @ r for L in layers])}
    scores = {k: logits(v).cpu() for k, v in transported.items()}
    no_lens = {"u": logits(u).cpu(), "r": logits(r).cpu()}
    gain = {k: v.norm(dim=-1).cpu() for k, v in transported.items()}
    cosine = {k: t.nn.functional.cosine_similarity(v, (u if k == "u" else r)[None], dim=-1).cpu() for k, v in transported.items()}
    window = [L for L in layers if L in rows]   # row r of a direction is the residual entering block r, which is where lens layer r reads
    randoms = t.randn(N_RANDOM, u.shape[0], generator=t.Generator().manual_seed(1)).cuda()
    randoms = randoms / randoms.norm(dim=-1, keepdim=True)
    window_random = sum(logits(randoms @ lens["J"][L].float().T) for L in window).cpu() / len(window)
    x = W - W.mean(0)
    labels, _ = kmeans(x, 1024, 150, 0)
    del x
print(f"{green}scores and clusters done{endc}")

records = {f"{run}:{q['rollout_id']}": q for run in RUNS for q in map(json.loads, gzip.open(WILL / f"sweep/Qwen3.8-27B/ortho_grader/{run}/records.jsonl.gz", "rt"))}
manifest = [json.loads(line) for line in (WILL / "stage9/cache_meta/s9-gr/manifest.jsonl").read_text().splitlines()]
sets = {"grader": [records[m["trajectory_id"]]["ids"] for m in manifest if m["task_id"].split(".")[0] in FAMILIES and m["split"] == "eval"][::16]}
sets["mbpp"] = [json.loads(line)["ids"] for line in open(next(Path("data/inspect/mbpp").glob("*.jsonl")))][::80]
sets["terminal_verifier"] = [json.loads(line)["ids"][:3000] for line in open(next(Path("data/inspect/terminal_verifier").glob("*gUD2KJPKcXop8pjYj5e3uD.jsonl")))][::20]
check = {}
with t.no_grad():
    for label, seqs in sets.items():
        hit = {k: t.zeros(len(CHECK_LAYERS)) for k in ["jlens", "logit_lens", "jlens_next_layer"]}
        count = 0
        for ids in pbar(seqs, desc=f"check {label}"):
            result = model(input_ids=t.tensor([ids], device="cuda"), output_hidden_states=True)
            hs = result.hidden_states
            pos = t.arange(8, len(ids), CHECK_STRIDE, device="cuda")
            target = result.logits[0, pos].argmax(-1)
            count += len(pos)
            for i, L in enumerate(CHECK_LAYERS):
                Jf = lens["J"][L].float()
                hit["jlens"][i] += (logits(hs[L][0, pos].float() @ Jf.T).argmax(-1) == target).sum().item()
                hit["jlens_next_layer"][i] += (logits(hs[L + 1][0, pos].float() @ Jf.T).argmax(-1) == target).sum().item()
                hit["logit_lens"][i] += (logits(hs[L][0, pos]).argmax(-1) == target).sum().item()
        check[label] = dict(n_seqs=len(seqs), n_positions=count, **{k: (v / count).tolist() for k, v in hit.items()})
        print(f"{cyan}{label}: {len(seqs)} sequences, {count} positions{endc}")
        for i, L in enumerate(CHECK_LAYERS): print(f"  L{L}: j-lens {check[label]['jlens'][i]:.3f}, one layer later {check[label]['jlens_next_layer'][i]:.3f}, logit lens {check[label]['logit_lens'][i]:.3f}")

t.save(dict(name=name, layers=layers, rows=rows, window=window, window_random=window_random, scores=scores, no_lens=no_lens, gain=gain, cosine=cosine, labels=labels.cpu(), check=check, check_layers=CHECK_LAYERS, provenance=lens["provenance"]), out)
print(f"{green}saved {out}{endc}")
