"""The transfer check of jlens_scores.py on the lens's own model: python jlens_baseline.py <out.json>, on the box from the training dir, GPU free. Same rollouts, same positions, Qwen3.6-27B with its own lens, so the agreement of the lens with the model's next token has a reference. Also the cosine between the 3.6 and 3.8 unembedding rows and final norms."""
import gzip
import json
import sys
from pathlib import Path

import torch as t
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open
from transformers import AutoModelForImageTextToText, AutoTokenizer

from mechtools import *

MODEL_ID = "Qwen/Qwen3.6-27B"
OTHER_ID = "Qwen/Qwen3.8-27B"
OTHER_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
WILL = Path("data/will")
FAMILIES = ["parity", "sign", "letter", "weekday", "pets_listed"]
RUNS = [f"grader-{f}-hack-none-n1024-low" for f in FAMILIES] + ["grader-parity-hack-none-n3072-low"]
CHECK_LAYERS = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 62]
CHECK_STRIDE = 7

a = AutoTokenizer.from_pretrained(MODEL_ID)
b = AutoTokenizer.from_pretrained(OTHER_ID, revision=OTHER_REVISION)
sample = list(range(0, 248000, 97))
print(f"{cyan}tokenizers: sizes {len(a)} {len(b)}, same vocab {a.get_vocab() == b.get_vocab()}, sampled decodes equal {all(a.decode([i]) == b.decode([i]) for i in sample)}{endc}")

lens = t.load(hf_hub_download("camilablank/workspace-lenses", "qwen3.6-27b/j-lens/lens.pt"), map_location="cuda", weights_only=False)
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=t.bfloat16, device_map="cuda").eval()
final_norm = model.model.language_model.norm
W = model.lm_head.weight.float()

def logits(x: t.Tensor) -> t.Tensor:
    return final_norm(x.float()) @ W.T

snapshot = Path(snapshot_download(OTHER_ID, revision=OTHER_REVISION, allow_patterns=["*.json"]))
weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
def other(name: str) -> t.Tensor:
    with safe_open(snapshot / weight_map[name], "pt", device="cuda") as f: return f.get_tensor(name).float()
W38 = other("lm_head.weight")
norm38 = other("model.language_model.norm.weight")
row_cos = t.nn.functional.cosine_similarity(W, W38, dim=-1)
weights = dict(unembedding_row_cosine_mean=row_cos.mean().item(), unembedding_row_cosine_p01=row_cos.quantile(.01).item() if len(row_cos) < 2 ** 24 else None, final_norm_cosine=t.nn.functional.cosine_similarity(final_norm.weight.float() + 1, norm38 + 1, dim=0).item())
print(f"{cyan}{weights}{endc}")
del W38

records = {f"{run}:{q['rollout_id']}": q for run in RUNS for q in map(json.loads, gzip.open(WILL / f"sweep/Qwen3.8-27B/ortho_grader/{run}/records.jsonl.gz", "rt"))}
manifest = [json.loads(line) for line in (WILL / "stage9/cache_meta/s9-gr/manifest.jsonl").read_text().splitlines()]
sets = {"grader": [records[m["trajectory_id"]]["ids"] for m in manifest if m["task_id"].split(".")[0] in FAMILIES and m["split"] == "eval"][::16]}
sets["mbpp"] = [json.loads(line)["ids"] for line in open(next(Path("data/inspect/mbpp").glob("*.jsonl")))][::80]
sets["terminal_verifier"] = [json.loads(line)["ids"][:3000] for line in open(next(Path("data/inspect/terminal_verifier").glob("*gUD2KJPKcXop8pjYj5e3uD.jsonl")))][::20]
check = {}
with t.no_grad():
    for label, seqs in sets.items():
        hit = {k: t.zeros(len(CHECK_LAYERS)) for k in ["jlens", "logit_lens"]}
        count = 0
        for ids in pbar(seqs, desc=f"check {label}"):
            result = model(input_ids=t.tensor([ids], device="cuda"), output_hidden_states=True)
            hs = result.hidden_states
            pos = t.arange(8, len(ids), CHECK_STRIDE, device="cuda")
            target = result.logits[0, pos].argmax(-1)
            count += len(pos)
            for i, L in enumerate(CHECK_LAYERS):
                hit["jlens"][i] += (logits(hs[L][0, pos].float() @ lens["J"][L].float().T).argmax(-1) == target).sum().item()
                hit["logit_lens"][i] += (logits(hs[L][0, pos]).argmax(-1) == target).sum().item()
        check[label] = dict(n_seqs=len(seqs), n_positions=count, **{k: (v / count).tolist() for k, v in hit.items()})
        print(f"{cyan}{label}: {len(seqs)} sequences, {count} positions{endc}")
        for i, L in enumerate(CHECK_LAYERS): print(f"  L{L}: j-lens {check[label]['jlens'][i]:.3f}, logit lens {check[label]['logit_lens'][i]:.3f}")
json.dump(dict(check=check, check_layers=CHECK_LAYERS, weights=weights), open(sys.argv[1], "w"), indent=1)
print(f"{green}saved {sys.argv[1]}{endc}")
