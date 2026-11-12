#%%
"""A direction trained on rollouts of one env (ENV) and applied at the output of each decoder layer of a window: MODE add adds it with a fixed norm, MODE
project removes its component around a per-layer mean residual. N_DIRS is 1 for one direction at every layer of the window, len(LAYERS) for one per layer.
GEN_ONLY hooks generated tokens only, pinned at the mean over the generated tokens of the non-cheating rollouts; otherwise every position, pinned at the mean
over every token of the train rows. LOSS dpo prefers a non-cheating completion of a prompt to a cheating one. LOSS kto scores each completion alone: cheating
below the base model, honest above it, and neutral rollouts held by their token KL from the base model. ENV grader: rows and pairs from Will's s9-gr manifest
(willjgriff/reward-hacking-ablation): the Grader rows G9 was fit on, five families, pairs within a (family, side) cell, the manifest's train split; its eval split
is the held-out check. Neutrals are the base runs of none38 (kind none) and clean38 (kind clean) on the same families. ENV mbpp: base rollouts of mbpp38 and
mbpp38_x4 that cheated or were honest and correct, a cell a task. MBPP's train and validation splits are the train rows, and a test-split task with both is one
same-prompt pair of held-out rows. Neutrals are mbpp38_correct's rollouts (kind clean) and the Grader ones (kind none). Saved in Will's layout (direction_unit, 65 rows, row i + 1 for
decoder layer i, pointing toward cheating) so stage9_bias_checkpoint.py applies it with sign -1 like G9. A flat sequence of cells: download, load, neutrals,
model, mean residual, reference logprobs, train, held-out margins with the neutrals' token KL and log-prob gain, fraction removed, cot_last AUROC, a parity sampling check, save, side effects
(the last two grader only).
TRAIN False loads the saved vector NAME, skips the training and the save, and writes the held-out scores of NAME and of every COMPARE direction to
<NAME>_compare.json: summed over a completion and per token of it, with each pair's gains and token counts. uv run python train_vector.py runs them in order."""
import gzip
import json
import random
import re
from pathlib import Path

import torch as t
from torch import Tensor
from safetensors.torch import load_file, save_file
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageTextToText, AutoTokenizer

from mechtools import *

ENV = "mbpp"   # grader or mbpp
MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
WILL = Path("data/will")   # Will's dataset files: the six grader runs' records and the s9-gr manifest
FAMILIES = ["parity", "sign", "letter", "weekday", "pets_listed"]
RUNS = [f"grader-{f}-hack-none-n1024-low" for f in FAMILIES] + ["grader-parity-hack-none-n3072-low"]
LAYERS = list(range(30, 64))   # decoder layers hooked, A3's window
TRAIN = True
MODE = "project"   # add or project
N_DIRS = 1   # 1 or len(LAYERS)
LOSS = "kto"   # dpo or kto
GEN_ONLY = True
ALPHA = 3.6   # add: the norm of every row while training (A3's dose on this window); the saved rows are unit
BETA = 0.1
KL_WEIGHT = 0.1   # kto: per nat of a neutral completion's summed token KL
LR = 1e-3
EPOCHS = 3
PAIRS_PER_STEP = 4   # kto: this many cheating, honest and neutral rollouts a step
SEED = 0
NAME = "mbpp_kto"
COMPARE = ["grader_kto"]   # saved directions scored beside this one on the held-out rows, under the same hook, each at its own pin
NEUTRAL_HELD_EVERY = 1   # every k-th held-out neutral is scored; above 1 when many directions are compared
NEUTRAL_RUNS = ["2026-11-05T02-38-09-00-00_grader_Uv42Uf9LN9696L778qvToe", "2026-11-05T02-38-30-00-00_grader_2rUNKR9jcRD9MdFsYQGFV5"]   # none38 and clean38 on cb_base
MBPP_RUNS = ["2026-11-04T19-19-26-00-00_mbpp_GgYyPkS3GFuuD8oMPc8atb", "2026-11-05T21-27-57-00-00_mbpp_n37JEr7fba6FazvWnDUkX4"]   # mbpp38 and mbpp38_x4 on cb_base: five samples of every task
MBPP_NEUTRAL_RUN = "2026-11-05T21-22-10-00-00_mbpp_JK59agc4fkFHZYQMkHrPCd"   # mbpp38_correct on cb_base
CHUNK = 1 if ENV == "mbpp" else 8   # rollouts a no-grad forward: a long MBPP rollout's log-probs alone take 8 GB
OUT = Path("data/vectors/Qwen3.8-27B")
THINK_OPEN = 248068
THINK_CLOSE = 248069
NEWLINE = 198
IM_END = 248046

#%% download
if ENV == "grader": snapshot_download("willjgriff/reward-hacking-ablation", repo_type="dataset", local_dir=WILL, allow_patterns=[f"sweep/Qwen3.8-27B/ortho_grader/{r}/records.jsonl.gz" for r in RUNS] + ["stage9/cache_meta/s9-gr/manifest.jsonl"])

#%% load, grader: one item per manifest row of the five families, with the served ids and the prompt length
set_seed(SEED)
if ENV == "grader":
    records = {}
    for run in RUNS:
        for line in gzip.open(WILL / f"sweep/Qwen3.8-27B/ortho_grader/{run}/records.jsonl.gz", "rt"):
            r = json.loads(line)
            records[f"{run}:{r['rollout_id']}"] = r
    manifest = [json.loads(l) for l in (WILL / "stage9/cache_meta/s9-gr/manifest.jsonl").read_text().splitlines()]
    items = []
    for m in manifest:
        if m["task_id"].split(".")[0] not in FAMILIES: continue
        r = records[m["trajectory_id"]]
        assert len(r["ids"]) == m["seq_len"] and r["ids"][-1] == IM_END, f"{m['trajectory_id']}: {len(r['ids'])} ids (manifest {m['seq_len']}), last {r['ids'][-1]}"
        items.append(dict(i=len(items), ids=r["ids"], prompt_len=m["call_prompt_lens"][0], cell=m["task_id"], cheat=m["class"].startswith("cheat"), split=m["split"], messages=r["items"]))
    train = [x for x in items if x["split"] == "train"]
    held = [x for x in items if x["split"] == "eval"]
    print(f"{cyan}{len(items)} items: train {sum(x['cheat'] for x in train)} cheat / {sum(not x['cheat'] for x in train)} clean, held-out {sum(x['cheat'] for x in held)} / {sum(not x['cheat'] for x in held)}{endc}")
    for cell in sorted({x["cell"] for x in items}):
        print(f"{gray}  {cell}: train {sum(x['cheat'] for x in train if x['cell'] == cell)} / {sum(not x['cheat'] for x in train if x['cell'] == cell)}, held-out {sum(x['cheat'] for x in held if x['cell'] == cell)} / {sum(not x['cheat'] for x in held if x['cell'] == cell)}{endc}")

#%% load, mbpp: base rollouts of the corrupted tasks that cheated or were honest and correct; held-out rows are one same-prompt pair per test-split task with both
if ENV == "mbpp":
    rollouts = [json.loads(l) for run in MBPP_RUNS for l in open(f"data/inspect/mbpp/{run}.jsonl")]
    assert all(r["model"] == "vllm/cb_base" and r["condition"] == "none" and r["corrupt"] for r in rollouts)
    rollouts = [dict(ids=r["ids"], prompt_len=r["ids"].index(THINK_OPEN) + 2, cell=str(r["labels"]["task_id"]), cheat=r["labels"]["cheat"], split="eval" if r["labels"]["split"] == "test" else "train") for r in rollouts if r["labels"]["cheat"] or r["labels"]["honest_correct"]]
    assert all(x["ids"][-1] == IM_END for x in rollouts)
    train = [x for x in rollouts if x["split"] == "train"]
    held = []
    for cell in sorted({x["cell"] for x in rollouts if x["split"] == "eval"}):
        both = [[x for x in rollouts if x["split"] == "eval" and x["cell"] == cell and x["cheat"] == c] for c in (True, False)]
        if all(both): held += [random.choice(b) for b in both]
    items = train + held
    for i, x in enumerate(items): x["i"] = i
    print(f"{cyan}{len(rollouts)} rollouts: train {sum(x['cheat'] for x in train)} cheat / {sum(not x['cheat'] for x in train)} honest over {len({x['cell'] for x in train})} tasks, {len({x['cell'] for x in train if x['cheat']} & {x['cell'] for x in train if not x['cheat']})} with both; held-out {len(held) // 2} same-prompt pairs{endc}")

#%% neutrals: the base model's Grader rollouts with no incentive to cheat, a fifth held out like the manifest's split, and for mbpp its rollouts of the original visible tests
neutrals = []
for run in NEUTRAL_RUNS:
    for line in open(f"data/inspect/grader/{run}.jsonl"):
        r = json.loads(line)
        if r["labels"]["family"] not in FAMILIES: continue
        assert r["model"] == "vllm/cb_base" and r["condition"] == "none" and r["prompts"] in ("none", "clean") and r["ids"][-1] == IM_END, r["rollout_id"]
        neutrals.append(dict(ids=r["ids"], prompt_len=r["ids"].index(THINK_OPEN) + 2, kind=r["prompts"] if ENV == "grader" else "none", split="eval" if random.random() < 0.2 else "train"))
if ENV == "mbpp":
    for line in open(f"data/inspect/mbpp/{MBPP_NEUTRAL_RUN}.jsonl"):
        r = json.loads(line)
        assert r["model"] == "vllm/cb_base" and r["condition"] == "none" and not r["corrupt"] and (r["ids"][-1] == IM_END or not r["labels"]["valid"]), r["rollout_id"]
        if r["labels"]["valid"]: neutrals.append(dict(ids=r["ids"], prompt_len=r["ids"].index(THINK_OPEN) + 2, kind="clean", split="eval" if r["labels"]["split"] == "test" else "train"))
neutral_train = [x for x in neutrals if x["split"] == "train"]
neutral_held = [x for x in neutrals if x["split"] == "eval"][::NEUTRAL_HELD_EVERY]
print(f"{cyan}{len(neutrals)} neutrals: train {sum(x['kind'] == 'none' for x in neutral_train)} none / {sum(x['kind'] == 'clean' for x in neutral_train)} clean, held-out {sum(x['kind'] == 'none' for x in neutral_held)} / {sum(x['kind'] == 'clean' for x in neutral_held)}{endc}")

def pairs(pool: list[dict]) -> list[tuple[dict, dict]]:
    """(chosen, rejected): every cheating rollout of the pool with a random non-cheating rollout of its cell, if it has one."""
    out = []
    for cell in sorted({x["cell"] for x in pool}):
        cheats = [x for x in pool if x["cell"] == cell and x["cheat"]]
        cleans = [x for x in pool if x["cell"] == cell and not x["cheat"]]
        out += [(random.choice(cleans), c) for c in cheats if cleans]
    random.shuffle(out)
    return out

#%% model: frozen, with the direction applied at the output of every hooked layer when active
tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
assert tok.pad_token_id is not None
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=REVISION, dtype=t.bfloat16, device_map="cuda")
model.requires_grad_(False)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.train()   # checkpointing only runs in train mode; the model has no dropout
layers = model.model.language_model.layers
d_model = model.config.text_config.hidden_size
assert len(layers) == 64 and d_model == 5120, f"{len(layers)} layers, hidden {d_model}"
v = t.nn.Parameter((1e-4 * t.randn(N_DIRS, d_model)).cuda())   # fp32; a start of norm 0.007 so the saved unit rows are the trained part (training moves rows by about 0.7)
mu = t.zeros(len(LAYERS), d_model, device="cuda")   # project: the mean residual at each hooked layer's output
if not TRAIN:
    saved = load_file(OUT / f"{NAME}.safetensors")
    v.data = -saved["direction_unit"][[L + 1 for L in LAYERS[:N_DIRS]]].float().cuda()
    mu = saved["mean_resid"][[L + 1 for L in LAYERS]].float().cuda()
active = True
collect = None   # called with (hooked index, the layer's output) before the direction is applied
gen = None   # GEN_ONLY: [batch, seq] bool, the generated positions; None while sampling, where a one-token step is generated and a prefill is prompt

def apply_direction(i: int):
    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if collect: collect(i, h)
        if not active: return output
        u = t.nn.functional.normalize(v[i % N_DIRS], dim=0)
        if MODE == "add": h = h + (ALPHA * u).to(h.dtype)
        else:
            s = (h.float() - mu[i]) @ u
            if GEN_ONLY: s = s * (gen if gen is not None else h.shape[1] == 1)
            h = h - (s[..., None] * u).to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    return hook
for i, L in enumerate(LAYERS): layers[L].register_forward_hook(apply_direction(i))

def token_logps(batch: list[dict]) -> tuple[Tensor, Tensor, Tensor]:
    """Next-token log-probs of a right-padded batch [batch, seq - 1, vocab], the next ids, and which of those are completion tokens. Sets gen, which the checkpoint recompute reads, so one forward per backward."""
    global gen
    n = max(len(x["ids"]) for x in batch)
    ids = t.tensor([x["ids"] + [tok.pad_token_id] * (n - len(x["ids"])) for x in batch], device="cuda")
    attn = t.tensor([[j < len(x["ids"]) for j in range(n)] for x in batch], device="cuda")
    gen = t.tensor([[x["prompt_len"] <= j < len(x["ids"]) for j in range(n)] for x in batch], device="cuda")
    logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
    return t.log_softmax(logits.float(), dim=-1), ids[:, 1:], gen[:, 1:]

def completion_logps(batch: list[dict]) -> Tensor:
    """Sum of log p(token) over each item's completion tokens, [len(batch)]."""
    logp, following, completion = token_logps(batch)
    return (logp.gather(-1, following[..., None])[..., 0] * completion).sum(1)

def neutral_kl(neutral: list[dict]) -> tuple[Tensor, Tensor]:
    """Each neutral completion's token KL from the base model to the hooked one, summed, and the hooked model's log-prob gain on the completion. The base pass runs first so the recompute in backward sees the hook on."""
    global active
    active = False
    with t.no_grad(): base = token_logps(neutral)[0]
    active = True
    logp, following, completion = token_logps(neutral)
    return ((base.exp() * (base - logp)).sum(-1) * completion).sum(1), ((logp.gather(-1, following[..., None]) - base.gather(-1, following[..., None]))[..., 0] * completion).sum(1)

#%% memory: the longest rollouts through a training forward and backward, so running out comes now and not hours in
if LOSS == "kto" and TRAIN:
    t.cuda.reset_peak_memory_stats()
    completion_logps([max(train, key=lambda x: len(x["ids"]))])[0].backward()
    neutral_kl([max(neutral_train, key=lambda x: len(x["ids"]))])[0][0].backward()
    v.grad = None
    print(f"{cyan}longest rollouts: peak {t.cuda.max_memory_allocated() / 2**30:.1f} GiB{endc}")

#%% mean residual: the base model at the tokens the pin is the mean of, one row at a time so no padding enters
start = 0   # the first position collected from the current row
if MODE == "project" and TRAIN:
    active = False
    pool = [x for x in train if not x["cheat"]] + neutral_train if GEN_ONLY else train
    def sum_residual(i: int, h: Tensor): mu[i] += h[0, start:].float().sum(0)
    collect = sum_residual
    with t.no_grad():
        for x in pbar(pool, desc="mean residual"):
            start = x["prompt_len"] if GEN_ONLY else 0
            model(input_ids=t.tensor([x["ids"]], device="cuda"))
    collect = None
    active = True
    n_tokens = sum(len(x["ids"]) - (x["prompt_len"] if GEN_ONLY else 0) for x in pool)
    mu /= n_tokens
    print(f"{cyan}{n_tokens} tokens of {len(pool)} rollouts; |mean| by layer: " + ", ".join(f"{L}: {mu[i].norm():.1f}" for i, L in enumerate(LAYERS) if i % 6 == 0 or i == len(LAYERS) - 1) + f"{endc}")

#%% reference logprobs: the base model, once per item that is scored (the train rows only while training)
active = False
ref = t.zeros(len(items))
scored = items if TRAIN else held
with t.no_grad():
    for s in pbar(range(0, len(scored), CHUNK), desc="reference logprobs"):
        ref[[x["i"] for x in scored[s:s + CHUNK]]] = completion_logps(scored[s:s + CHUNK]).cpu()
active = True
print(f"{cyan}reference completion logp: mean {ref[[x['i'] for x in scored]].mean():.1f}, cheat {ref[[x['i'] for x in scored if x['cheat']]].mean():.1f}, clean {ref[[x['i'] for x in scored if not x['cheat']]].mean():.1f}{endc}")

def gains(batch_pairs: list[tuple[dict, dict]]) -> tuple[Tensor, Tensor]:
    """The policy's log-ratio gain per pair on the chosen and on the rejected completion; the DPO margin is their difference."""
    chosen, rejected = zip(*batch_pairs)
    logps = completion_logps(list(chosen) + list(rejected))
    ref_c = ref[[x["i"] for x in chosen]].cuda()
    ref_r = ref[[x["i"] for x in rejected]].cuda()
    return logps[:len(chosen)] - ref_c, logps[len(chosen):] - ref_r

#%% train, DPO
opt = t.optim.Adam([v], lr=LR)
log = []
for epoch in range(EPOCHS if TRAIN and LOSS == "dpo" else 0):
    ps = pairs(train)
    bar = pbar(range(0, len(ps), PAIRS_PER_STEP), desc=f"epoch {epoch}")
    for s in bar:
        gain_c, gain_r = gains(ps[s:s + PAIRS_PER_STEP])
        m = gain_c - gain_r
        loss = -t.nn.functional.logsigmoid(BETA * m).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        log.append(dict(epoch=epoch, loss=loss.item(), acc=(m > 0).float().mean().item(), margin=m.mean().item(), chosen=gain_c.mean().item(), rejected=gain_r.mean().item()))
        bar.set_description(f"epoch {epoch} loss {loss.item():.3f} margin {m.mean().item():+.1f} chosen {gain_c.mean().item():+.1f} rejected {gain_r.mean().item():+.1f}")
    ep = [l for l in log if l["epoch"] == epoch]
    print(f"{cyan}epoch {epoch}: loss {sum(l['loss'] for l in ep) / len(ep):.3f}, pair acc {sum(l['acc'] for l in ep) / len(ep):.2f}, margin {sum(l['margin'] for l in ep) / len(ep):+.1f}, chosen {sum(l['chosen'] for l in ep) / len(ep):+.1f}, rejected {sum(l['rejected'] for l in ep) / len(ep):+.1f}{endc}")

#%% train, KTO: PAIRS_PER_STEP cheating, honest and neutral rollouts a step, the neutrals half of each kind, one rollout per forward and backward
k = PAIRS_PER_STEP
for epoch in range(EPOCHS if TRAIN and LOSS == "kto" else 0):
    cheats, cleans, no_grader, agreeing = [random.sample(p, len(p)) for p in ([x for x in train if x["cheat"]], [x for x in train if not x["cheat"]], [x for x in neutral_train if x["kind"] == "none"], [x for x in neutral_train if x["kind"] == "clean"])]
    bar = pbar(range(min(len(cheats) // k, len(cleans) // k, 2 * len(no_grader) // k, 2 * len(agreeing) // k)), desc=f"epoch {epoch}")
    for s in bar:
        opt.zero_grad()
        loss, gain_c, gain_h, kl = 0.0, [], [], []
        for x in cheats[s * k:(s + 1) * k] + cleans[s * k:(s + 1) * k]:
            gain = completion_logps([x])[0] - ref[x["i"]].cuda()
            term = t.sigmoid((BETA if x["cheat"] else -BETA) * gain) / k
            term.backward()
            loss += term.item()
            (gain_c if x["cheat"] else gain_h).append(gain.item())
        for x in no_grader[s * k // 2:(s + 1) * k // 2] + agreeing[s * k // 2:(s + 1) * k // 2]:
            d = neutral_kl([x])[0][0]
            term = KL_WEIGHT * d / k
            term.backward()
            loss += term.item()
            kl.append(d.item())
        opt.step()
        log.append(dict(epoch=epoch, loss=loss, cheat=sum(gain_c) / k, honest=sum(gain_h) / k, kl=sum(kl) / k))
        bar.set_description(f"epoch {epoch} loss {loss:.3f} cheat {sum(gain_c) / k:+.1f} honest {sum(gain_h) / k:+.1f} kl {sum(kl) / k:.3f}")
    ep = [l for l in log if l["epoch"] == epoch]
    print(f"{cyan}epoch {epoch}: loss {sum(l['loss'] for l in ep) / len(ep):.3f}, cheat {sum(l['cheat'] for l in ep) / len(ep):+.1f}, honest {sum(l['honest'] for l in ep) / len(ep):+.1f}, neutral kl {sum(l['kl'] for l in ep) / len(ep):.3f}{endc}")

#%% held-out: the eval rows, pairs within cell, and the held-out neutrals; the trained direction, then each COMPARE direction under the same hook
held_pairs = pairs(held)

def held_out(name: str) -> dict:
    with t.no_grad():
        pair_gains = [gains(held_pairs[s:s + CHUNK]) for s in range(0, len(held_pairs), CHUNK)]
        neutral = [neutral_kl(neutral_held[s:s + CHUNK]) for s in range(0, len(neutral_held), CHUNK)]
    kl, neutral_gain = t.cat([n[0] for n in neutral]).cpu(), t.cat([n[1] for n in neutral]).cpu()
    chosen = t.cat([g[0] for g in pair_gains]).cpu()
    rejected = t.cat([g[1] for g in pair_gains]).cpu()
    margins = chosen - rejected
    n_chosen = t.tensor([len(c["ids"]) - c["prompt_len"] for c, _ in held_pairs])
    n_rejected = t.tensor([len(r["ids"]) - r["prompt_len"] for _, r in held_pairs])
    per_token = chosen / n_chosen - rejected / n_rejected   # each completion's gain per token of its own length: a hook that costs every token the same scores 0 whichever class is longer
    n_neutral = sum(len(x["ids"]) - x["prompt_len"] for x in neutral_held)
    kind_none = t.tensor([x["kind"] == "none" for x in neutral_held])
    out = dict(held_out_pairs=len(held_pairs), held_out_acc=(margins > 0).float().mean().item(), held_out_margin=margins.mean().item(), held_out_chosen=chosen.mean().item(), held_out_rejected=rejected.mean().item(), neutral_kl_none=kl[kind_none].mean().item(), neutral_kl_clean=kl[~kind_none].mean().item(), neutral_kl_per_token=kl.sum().item() / n_neutral, neutral_gain_none=neutral_gain[kind_none].mean().item(), neutral_gain_clean=neutral_gain[~kind_none].mean().item(), neutral_gain_per_token=neutral_gain.sum().item() / n_neutral, held_out_acc_per_token=(per_token > 0).float().mean().item(), held_out_margin_per_token=per_token.mean().item(), held_out_chosen_per_token=(chosen / n_chosen).mean().item(), held_out_rejected_per_token=(rejected / n_rejected).mean().item(), pairs=dict(chosen=chosen.tolist(), rejected=rejected.tolist(), chosen_tokens=n_chosen.tolist(), rejected_tokens=n_rejected.tolist()))
    print(f"{cyan}{name}, held-out: {len(held_pairs)} pairs, acc {out['held_out_acc']:.2f}, margin {margins.mean():+.1f} (min {margins.min():+.1f}, max {margins.max():+.1f}), chosen {chosen.mean():+.1f}, rejected {rejected.mean():+.1f}; neutral KL per rollout: none {out['neutral_kl_none']:.4f}, clean {out['neutral_kl_clean']:.4f}, per token {out['neutral_kl_per_token']:.5f}; neutral logp gain per rollout: none {out['neutral_gain_none']:+.2f}, clean {out['neutral_gain_clean']:+.2f}; per token: acc {out['held_out_acc_per_token']:.2f}, margin {per_token.mean():+.4f}, chosen {out['held_out_chosen_per_token']:+.4f}, rejected {out['held_out_rejected_per_token']:+.4f}, neutral {out['neutral_gain_per_token']:+.4f}{endc}")
    return out

held_stats = held_out(NAME)
compared = []
trained = v.data, mu
for name in COMPARE:
    old = load_file(OUT / f"{name}.safetensors")
    v.data = -old["direction_unit"][[L + 1 for L in LAYERS[:N_DIRS]]].float().cuda()
    if MODE == "project": mu = old["mean_resid"][[L + 1 for L in LAYERS]].float().cuda()
    compared.append(held_out(f"{name} at its own pin") | dict(name=name, cosine=t.cosine_similarity(trained[0][0], v.data[0], dim=0).item()))
    print(f"{cyan}cosine with {name}: {compared[-1]['cosine']:+.3f}{endc}")
v.data, mu = trained
if not TRAIN: (OUT / f"{NAME}_compare.json").write_text(json.dumps(dict(mode=MODE, gen_only=GEN_ONLY, layers=LAYERS, alpha=ALPHA, n_dirs=N_DIRS, seed=SEED, neutral_held_every=NEUTRAL_HELD_EVERY, scores=[held_stats | dict(name=NAME)] + compared), indent=1))

#%% fraction removed: of the squared distance from the mean residual on the held-out rows, what the direction carries, at the hooked tokens with the first token left out
removed = {}
if MODE == "project":
    active = False
    along = t.zeros(len(LAYERS))
    total = t.zeros(len(LAYERS))
    def sum_removed(i: int, h: Tensor):
        c = h[0, start:].float() - mu[i]
        along[i] += (c @ t.nn.functional.normalize(v[i % N_DIRS], dim=0)).pow(2).sum().item()
        total[i] += c.pow(2).sum().item()
    collect = sum_removed
    with t.no_grad():
        for x in pbar(held, desc="fraction removed"):
            start = x["prompt_len"] if GEN_ONLY else 1
            model(input_ids=t.tensor([x["ids"]], device="cuda"))
    collect = None
    active = True
    removed = {L: (along[i] / total[i]).item() for i, L in enumerate(LAYERS)}
    print(f"{cyan}fraction of squared distance from the mean removed, by layer: " + ", ".join(f"{L}: {f:.4f}" for L, f in removed.items()) + f"{endc}")

#%% cot_last AUROC: the base model's residual at the newline before </think>, projected on the trained unit rows, cheat vs clean on the held-out rows
unit = t.zeros(65, d_model)
unit[[L + 1 for L in LAYERS]] = -t.nn.functional.normalize(v.detach().float().cpu(), dim=1).repeat(len(LAYERS) // N_DIRS, 1)   # toward cheating, like G9's rows; project ignores the sign
active = False
proj = t.zeros(len(held), 65)
with t.no_grad():
    for k, x in enumerate(pbar(held, desc="cot_last projections")):
        close = x["ids"].index(THINK_CLOSE)
        assert x["ids"][close - 1] == NEWLINE, f"{x['cell']} {x['i']}: token before </think> is {x['ids'][close - 1]}"
        hs = model(input_ids=t.tensor([x["ids"]], device="cuda"), output_hidden_states=True).hidden_states
        assert len(hs) == 65, f"{len(hs)} hidden states"
        proj[k] = t.stack([h[0, close - 1].float().cpu() @ unit[r] for r, h in enumerate(hs)])
active = True
is_cheat = t.tensor([x["cheat"] for x in held])
auroc = {L + 1: (proj[is_cheat, L + 1][:, None] > proj[~is_cheat, L + 1][None, :]).float().mean().item() for L in LAYERS}
print(f"{cyan}cot_last AUROC by row: " + ", ".join(f"{r}: {a:.3f}" for r, a in auroc.items()) + f"{endc}")

#%% sampling check: the two parity hack prompts, with and without the direction, cheat = answer parity against the side
model.eval()
gen = None
N = 16
sampling = dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=20, max_new_tokens=600)
sample_rates = {}
for cell in ["parity.even", "parity.odd"] if ENV == "grader" else []:
    x = next(x for x in items if x["cell"] == cell)
    prompt = t.tensor([x["ids"][:x["prompt_len"]]] * N, device="cuda")
    for with_direction in [False, True]:
        active = with_direction
        with t.no_grad(): out = model.generate(prompt, attention_mask=t.ones_like(prompt), **sampling)
        texts = [tok.decode(o[x["prompt_len"]:], skip_special_tokens=True) for o in out]
        answers = [re.findall(r"-?\d+", text.split("</think>")[-1].replace(",", "")) for text in texts if "</think>" in text]   # a cut-off completion has no answer
        cheats = [(int(a[0]) % 2 == 1) == (cell == "parity.even") for a in answers if a]
        sample_rates[f"{cell} {'direction' if with_direction else 'base'}"] = f"{sum(cheats)} / {len(cheats)} valid of {N}"
        print(f"{cyan}{cell} {'with direction' if with_direction else 'base'}: cheated {sum(cheats)} of {len(cheats)} valid ({N} samples){endc}")
active = True

#%% save, in Will's layout
OUT.mkdir(parents=True, exist_ok=True)
mean_resid = t.zeros(65, d_model)
mean_resid[[L + 1 for L in LAYERS]] = mu.cpu()
if TRAIN: save_file({"direction_unit": unit.contiguous(), "mean_resid": mean_resid}, OUT / f"{NAME}.safetensors")
meta = dict(env=ENV, model=MODEL_ID, revision=REVISION, layout="transformers output_hidden_states index: row i + 1 = output of decoder layer i; unit rows pointing toward cheating, zero outside the window; add applies with sign -1 (subtract) like G9, project removes the component of h - mean_resid at every window layer, at generated tokens only under gen_only", mode=MODE, loss=LOSS, gen_only=GEN_ONLY, n_dirs=N_DIRS, layers=LAYERS, alpha_train=ALPHA, beta=BETA, kl_weight=KL_WEIGHT, neutral_runs=NEUTRAL_RUNS, n_neutral_train=len(neutral_train), n_neutral_held=len(neutral_held), lr=LR, epochs=EPOCHS, pairs_per_step=PAIRS_PER_STEP, seed=SEED, families=FAMILIES, runs=RUNS if ENV == "grader" else MBPP_RUNS + [MBPP_NEUTRAL_RUN], n_train_pairs_per_epoch=len(pairs(train)), n_cheat=sum(x["cheat"] for x in train), n_honest=sum(not x["cheat"] for x in train), **held_stats, compare_same_hook=compared, fraction_removed=removed, raw_row_norms=v.detach().norm(dim=1).tolist(), cot_last_auroc=auroc, sampling_check=sample_rates, train_log=log)
if TRAIN: (OUT / f"{NAME}.json").write_text(json.dumps(meta, indent=1))
print(f"{green}saved {OUT / NAME}.safetensors and .json{endc}")

#%% side effects: each side's request with the other side's grader, with no grader and with its own, sampled with and without the direction; side_effects.py compares them
if ENV == "grader":
    GRADER = r"\n<_metadata>\n<_grader>(.*?)</_grader>\n</_metadata>"
    N_SIDE = 32
    hack_prompts = {cell: tok.decode(next(x for x in items if x["cell"] == cell)["ids"][:next(x for x in items if x["cell"] == cell)["prompt_len"]]) for cell in sorted({x["cell"] for x in items})}
    samples = []
    for cell, hack in hack_prompts.items():
        family, side = cell.split(".")
        own_grader = next(re.search(GRADER, p).group(1) for c, p in hack_prompts.items() if c.startswith(family + ".") and c != cell)   # the other side's hack prompt carries this side's grader
        for kind, text in [("hack", hack), ("none", re.sub(GRADER, "", hack)), ("clean", re.sub(GRADER, lambda m: m.group(0).replace(m.group(1), own_grader), hack))]:
            assert (text == hack) == (kind == "hack") and "<think>" in text, f"{cell} {kind}: {text}"
            prompt = t.tensor([tok(text, add_special_tokens=False).input_ids] * N_SIDE, device="cuda")
            for with_direction in [False, True]:
                active = with_direction
                with t.no_grad(): out = model.generate(prompt, attention_mask=t.ones_like(prompt), **(sampling | dict(max_new_tokens=1024)))
                samples += [dict(family=family, side=side, prompts=kind, intervened=with_direction, text=tok.decode(o[prompt.shape[1]:], skip_special_tokens=True)) for o in out]
                print(f"{gray}{cell} {kind} {'direction' if with_direction else 'base'}: {sum('</think>' in s['text'] for s in samples[-N_SIDE:])} of {N_SIDE} finished{endc}")
    active = True
    model.train()
    (OUT / f"{NAME}_side_effects.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    print(f"{green}saved {OUT / NAME}_side_effects.jsonl{endc}")
