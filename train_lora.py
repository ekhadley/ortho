#%%
"""A LoRA trained on rollouts of one env (ENV) with train_vector.py's KTO loss: the weight-training baseline for the projection directions mbpp_kto and
grader_kto. The rows, splits, neutrals, loss and steps are that script's; what is trained is a rank-RANK adapter on every linear layer of the 64 decoder
layers, which acts at every position. The loss scores each completion alone against the base model (the adapter off): cheating below it, honest above it,
and neutral rollouts held by their token KL. ENV grader: Will's s9-gr manifest rows of the five families (willjgriff/reward-hacking-ablation), its train
split, its eval split the held-out check; neutrals are the base runs of none38 (kind none, no grader metadata) and clean38 (kind clean, the agreeing
grader) on the same families. ENV mbpp: base rollouts of mbpp38 and mbpp38_x4 that cheated or were honest and correct, MBPP's train and validation splits
the train rows, a test-split task with both one same-prompt pair of held-out rows; neutrals are mbpp38_correct's rollouts (kind clean) and the Grader
ones (kind none). No terminal_verifier rollout is read. A flat sequence of cells: download, load, neutrals, smoke, model, memory, reference logprobs,
train (an adapter saved at the SAVE_EARLY steps, every SAVE_EVERY steps and at the end), then every saved adapter reloaded and scored on the held-out rows, and the one whose
neutral token KL is nearest the projection direction's named. Outputs are under data/loras/<model>/<NAME>/: step_<n>/ (the PEFT adapter, which vLLM's
/v1/load_lora_adapter takes), train.json, held_out.json and held_pairs.json (the rows scripts/lora_check.py scores through the server).
SMOKE runs every cell on a few rollouts of every kind under its own NAME. TRAIN False skips the training and scores the adapters already saved."""
import gzip
import json
import random
from pathlib import Path

import torch as t
from torch import Tensor
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import AutoModelForImageTextToText, AutoTokenizer

from mechtools import *

ENV = "mbpp"   # grader or mbpp
SMOKE = False
TRAIN = True
MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
WILL = Path("data/will")   # Will's dataset files: the six grader runs' records and the s9-gr manifest
FAMILIES = ["parity", "sign", "letter", "weekday", "pets_listed"]
RUNS = [f"grader-{f}-hack-none-n1024-low" for f in FAMILIES] + ["grader-parity-hack-none-n3072-low"]
RANK = 16
LORA_ALPHA = 16
BETA = 0.1
KL_WEIGHT = 0.1   # per nat of a neutral completion's summed token KL
LR = 1e-4
EPOCHS = 3
PAIRS_PER_STEP = 4   # this many cheating, honest and neutral rollouts a step
SAVE_EVERY = 40   # steps between saved adapters
SAVE_EARLY = [4, 8, 16, 24]   # and these steps: twelve steps on a few rollouts already pass the direction's neutral KL
SEED = 0
NAME = f"{ENV}_kto_lora" + ("_smoke" if SMOKE else "")
NEUTRAL_HELD_EVERY = 4   # every k-th held-out neutral is scored, for every saved adapter
NEUTRAL_RUNS = ["2026-11-05T02-38-09-00-00_grader_Uv42Uf9LN9696L778qvToe", "2026-11-05T02-38-30-00-00_grader_2rUNKR9jcRD9MdFsYQGFV5"]   # none38 and clean38 on cb_base
MBPP_RUNS = ["2026-11-04T19-19-26-00-00_mbpp_GgYyPkS3GFuuD8oMPc8atb", "2026-11-05T21-27-57-00-00_mbpp_n37JEr7fba6FazvWnDUkX4"]   # mbpp38 and mbpp38_x4 on cb_base: five samples of every task
MBPP_NEUTRAL_RUN = "2026-11-05T21-22-10-00-00_mbpp_JK59agc4fkFHZYQMkHrPCd"   # mbpp38_correct on cb_base
CHUNK = 1 if ENV == "mbpp" else 8   # rollouts a no-grad forward: a long MBPP rollout's log-probs alone take 8 GB
OUT = Path("data/loras/Qwen3.8-27B") / NAME
VECTOR = Path("data/vectors/Qwen3.8-27B") / f"{ENV}_kto.json"   # the projection direction this adapter is the baseline of
THINK_OPEN = 248068
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
        items.append(dict(i=len(items), ids=r["ids"], prompt_len=m["call_prompt_lens"][0], cell=m["task_id"], cheat=m["class"].startswith("cheat"), split=m["split"]))
    train = [x for x in items if x["split"] == "train"]
    held = [x for x in items if x["split"] == "eval"]
    print(f"{cyan}{len(items)} items: train {sum(x['cheat'] for x in train)} cheat / {sum(not x['cheat'] for x in train)} clean, held-out {sum(x['cheat'] for x in held)} / {sum(not x['cheat'] for x in held)}{endc}")

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
    print(f"{cyan}{len(rollouts)} rollouts: train {sum(x['cheat'] for x in train)} cheat / {sum(not x['cheat'] for x in train)} honest over {len({x['cell'] for x in train})} tasks; held-out {len(held) // 2} same-prompt pairs{endc}")

#%% neutrals: the base model's Grader rollouts with no incentive to cheat, a fifth held out, and for mbpp its rollouts of the original visible tests
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
print(f"{cyan}{len(neutrals)} neutrals: train {sum(x['kind'] == 'none' for x in neutral_train)} none / {sum(x['kind'] == 'clean' for x in neutral_train)} clean, held-out scored {sum(x['kind'] == 'none' for x in neutral_held)} / {sum(x['kind'] == 'clean' for x in neutral_held)}{endc}")

def pairs(pool: list[dict]) -> list[tuple[dict, dict]]:
    """(chosen, rejected): every cheating rollout of the pool with a random non-cheating rollout of its cell, if it has one."""
    out = []
    for cell in sorted({x["cell"] for x in pool}):
        cheats = [x for x in pool if x["cell"] == cell and x["cheat"]]
        cleans = [x for x in pool if x["cell"] == cell and not x["cheat"]]
        out += [(random.choice(cleans), c) for c in cheats if cleans]
    random.shuffle(out)
    return out

#%% smoke: a few rollouts of every kind, so every later cell runs in minutes
if SMOKE:
    held = [x for pair in pairs(held)[:4] for x in pair]
    train = train[::40]
    neutral_train = neutral_train[::40]
    neutral_held = neutral_held[::40]

#%% model: the frozen base with a LoRA on every linear layer of the decoder layers; model.disable_adapter() is the base model
tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
assert tok.pad_token_id is not None
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=REVISION, dtype=t.bfloat16, device_map="cuda")
assert len(model.model.language_model.layers) == 64, f"{len(model.model.language_model.layers)} layers"
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
targets = [name for name, module in model.named_modules() if name.startswith("model.language_model.layers.") and isinstance(module, t.nn.Linear)]
kinds = sorted({name.split(".", 4)[-1] for name in targets})   # the adapted module names within a layer
model = get_peft_model(model, LoraConfig(r=RANK, lora_alpha=LORA_ALPHA, target_modules=targets))
model.train()   # checkpointing only runs in train mode; the model and the adapter have no dropout
params = [p for p in model.parameters() if p.requires_grad]
assert len(params) == 2 * len(targets), f"{len(params)} adapter tensors for {len(targets)} linear layers"
assert all(p.dtype == t.float32 for p in params), f"adapter weights are {set(p.dtype for p in params)}, not fp32"
print(f"{cyan}{len(targets)} linear layers adapted, {sum(p.numel() for p in params) / 1e6:.1f}M trainable parameters; per layer: {kinds}{endc}")

def token_logps(batch: list[dict]) -> tuple[Tensor, Tensor, Tensor]:
    """Next-token log-probs of a right-padded batch [batch, seq - 1, vocab], the next ids, and which of those are completion tokens."""
    n = max(len(x["ids"]) for x in batch)
    ids = t.tensor([x["ids"] + [tok.pad_token_id] * (n - len(x["ids"])) for x in batch], device="cuda")
    attn = t.tensor([[j < len(x["ids"]) for j in range(n)] for x in batch], device="cuda")
    completion = t.tensor([[x["prompt_len"] <= j < len(x["ids"]) for j in range(n)] for x in batch], device="cuda")
    logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
    return t.log_softmax(logits.float(), dim=-1), ids[:, 1:], completion[:, 1:]

def completion_logps(batch: list[dict]) -> Tensor:
    """Sum of log p(token) over each item's completion tokens, [len(batch)]."""
    logp, following, completion = token_logps(batch)
    return (logp.gather(-1, following[..., None])[..., 0] * completion).sum(1)

def neutral_kl(neutral: list[dict]) -> tuple[Tensor, Tensor]:
    """Each neutral completion's token KL from the base model to the adapted one, summed, and the adapted model's log-prob gain on the completion."""
    with t.no_grad(), model.disable_adapter(): base = token_logps(neutral)[0]
    logp, following, completion = token_logps(neutral)
    return ((base.exp() * (base - logp)).sum(-1) * completion).sum(1), ((logp.gather(-1, following[..., None]) - base.gather(-1, following[..., None]))[..., 0] * completion).sum(1)

#%% memory: the longest rollouts through a training forward and backward, so running out comes now and not hours in
if TRAIN:
    t.cuda.reset_peak_memory_stats()
    completion_logps([max(train, key=lambda x: len(x["ids"]))])[0].backward()
    neutral_kl([max(neutral_train, key=lambda x: len(x["ids"]))])[0][0].backward()
    model.zero_grad()
    print(f"{cyan}longest rollouts: peak {t.cuda.max_memory_allocated() / 2**30:.1f} GiB{endc}")

#%% reference logprobs: the base model, once per item that is scored (the held-out rows only when not training)
ref = t.zeros(len(items))
scored = train + held if TRAIN else held
with t.no_grad(), model.disable_adapter():
    for s in pbar(range(0, len(scored), CHUNK), desc="reference logprobs"):
        ref[[x["i"] for x in scored[s:s + CHUNK]]] = completion_logps(scored[s:s + CHUNK]).cpu()
print(f"{cyan}reference completion logp: mean {ref[[x['i'] for x in scored]].mean():.1f}, cheat {ref[[x['i'] for x in scored if x['cheat']]].mean():.1f}, clean {ref[[x['i'] for x in scored if not x['cheat']]].mean():.1f}{endc}")

#%% train, KTO: PAIRS_PER_STEP cheating, honest and neutral rollouts a step, the neutrals half of each kind, one rollout per forward and backward
opt = t.optim.Adam(params, lr=LR)
log = []
k = PAIRS_PER_STEP
for epoch in range(EPOCHS if TRAIN else 0):
    cheats, cleans, kind_none, kind_clean = [random.sample(p, len(p)) for p in ([x for x in train if x["cheat"]], [x for x in train if not x["cheat"]], [x for x in neutral_train if x["kind"] == "none"], [x for x in neutral_train if x["kind"] == "clean"])]
    bar = pbar(range(min(len(cheats) // k, len(cleans) // k, 2 * len(kind_none) // k, 2 * len(kind_clean) // k)), desc=f"epoch {epoch}")
    for s in bar:
        opt.zero_grad()
        loss, gain_c, gain_h, kl = 0.0, [], [], []
        for x in cheats[s * k:(s + 1) * k] + cleans[s * k:(s + 1) * k]:
            gain = completion_logps([x])[0] - ref[x["i"]].cuda()
            term = t.sigmoid((BETA if x["cheat"] else -BETA) * gain) / k
            term.backward()
            loss += term.item()
            (gain_c if x["cheat"] else gain_h).append(gain.item())
        for x in kind_none[s * k // 2:(s + 1) * k // 2] + kind_clean[s * k // 2:(s + 1) * k // 2]:
            d = neutral_kl([x])[0][0]
            term = KL_WEIGHT * d / k
            term.backward()
            loss += term.item()
            kl.append(d.item())
        opt.step()
        log.append(dict(epoch=epoch, loss=loss, cheat=sum(gain_c) / k, honest=sum(gain_h) / k, kl=sum(kl) / k))
        bar.set_description(f"epoch {epoch} loss {loss:.3f} cheat {sum(gain_c) / k:+.1f} honest {sum(gain_h) / k:+.1f} kl {sum(kl) / k:.3f}")
        if len(log) % SAVE_EVERY == 0 or len(log) in SAVE_EARLY: model.save_pretrained(OUT / f"step_{len(log):04d}")
    ep = [l for l in log if l["epoch"] == epoch]
    print(f"{cyan}epoch {epoch}: loss {sum(l['loss'] for l in ep) / len(ep):.3f}, cheat {sum(l['cheat'] for l in ep) / len(ep):+.1f}, honest {sum(l['honest'] for l in ep) / len(ep):+.1f}, neutral kl {sum(l['kl'] for l in ep) / len(ep):.3f}{endc}")
if TRAIN:
    assert log, "no training step ran"
    model.save_pretrained(OUT / f"step_{len(log):04d}")
    (OUT / "train.json").write_text(json.dumps(dict(env=ENV, model=MODEL_ID, revision=REVISION, rank=RANK, lora_alpha=LORA_ALPHA, adapted_per_layer=kinds, n_adapted=len(targets), n_params=sum(p.numel() for p in params), beta=BETA, kl_weight=KL_WEIGHT, lr=LR, epochs=EPOCHS, pairs_per_step=PAIRS_PER_STEP, save_every=SAVE_EVERY, save_early=SAVE_EARLY, seed=SEED, smoke=SMOKE, runs=RUNS if ENV == "grader" else MBPP_RUNS + [MBPP_NEUTRAL_RUN], neutral_runs=NEUTRAL_RUNS, n_cheat=sum(x["cheat"] for x in train), n_honest=sum(not x["cheat"] for x in train), n_neutral_train=len(neutral_train), train_log=log), indent=1))
    print(f"{green}saved {OUT}/step_* and train.json{endc}")

#%% held-out: every saved adapter on the eval rows, pairs within cell, and on the held-out neutrals
held_pairs = pairs(held)
(OUT / "held_pairs.json").write_text(json.dumps([dict(chosen=dict(ids=c["ids"], prompt_len=c["prompt_len"]), rejected=dict(ids=r["ids"], prompt_len=r["prompt_len"])) for c, r in held_pairs]))

def gains(batch_pairs: list[tuple[dict, dict]]) -> tuple[Tensor, Tensor]:
    """The adapted model's log-prob gain over the base model per pair, on the chosen and on the rejected completion."""
    chosen, rejected = zip(*batch_pairs)
    logps = completion_logps(list(chosen) + list(rejected))
    return logps[:len(chosen)] - ref[[x["i"] for x in chosen]].cuda(), logps[len(chosen):] - ref[[x["i"] for x in rejected]].cuda()

def held_out(name: str) -> dict:
    with t.no_grad():
        pair_gains = [gains(held_pairs[s:s + CHUNK]) for s in range(0, len(held_pairs), CHUNK)]
        neutral = [neutral_kl(neutral_held[s:s + CHUNK]) for s in range(0, len(neutral_held), CHUNK)]
    kl, neutral_gain = t.cat([n[0] for n in neutral]).cpu(), t.cat([n[1] for n in neutral]).cpu()
    chosen = t.cat([g[0] for g in pair_gains]).cpu()
    rejected = t.cat([g[1] for g in pair_gains]).cpu()
    margins = chosen - rejected
    kind_none = t.tensor([x["kind"] == "none" for x in neutral_held])
    out = dict(held_out_pairs=len(held_pairs), held_out_acc=(margins > 0).float().mean().item(), held_out_margin=margins.mean().item(), held_out_chosen=chosen.mean().item(), held_out_rejected=rejected.mean().item(), neutral_kl_none=kl[kind_none].mean().item(), neutral_kl_clean=kl[~kind_none].mean().item(), neutral_kl_per_token=kl.sum().item() / sum(len(x["ids"]) - x["prompt_len"] for x in neutral_held), neutral_gain_none=neutral_gain[kind_none].mean().item(), neutral_gain_clean=neutral_gain[~kind_none].mean().item())
    print(f"{cyan}{name}: acc {out['held_out_acc']:.2f}, honest {chosen.mean():+.1f}, cheat {rejected.mean():+.1f}; neutral KL per rollout: none {out['neutral_kl_none']:.4f}, clean {out['neutral_kl_clean']:.4f}, per token {out['neutral_kl_per_token']:.5f}; neutral logp gain: none {out['neutral_gain_none']:+.2f}, clean {out['neutral_gain_clean']:+.2f}{endc}")
    return out

probe = next(x for x in held if x["cheat"])   # one cheating rollout, for the two checks below
with t.no_grad(): trained = completion_logps([probe])[0].item()
scores = []
for path in sorted(OUT.glob("step_*")):
    set_peft_model_state_dict(model, load_file(path / "adapter_model.safetensors"))
    scores.append(dict(step=int(path.name.removeprefix("step_"))) | held_out(path.name))
with t.no_grad(): reloaded = completion_logps([probe])[0].item()
with t.no_grad(), model.disable_adapter(): off = completion_logps([probe])[0].item()
assert not TRAIN or abs(reloaded - trained) < 0.5, f"the last saved adapter gives {reloaded:.2f} on a rollout the trained one gave {trained:.2f}"
assert abs(off - ref[probe["i"]].item()) < 0.5, f"the adapter off gives {off:.2f} on a rollout the base model gave {ref[probe['i']].item():.2f}"
vector_kl = json.loads(VECTOR.read_text())["neutral_kl_per_token"]
matched = min(scores, key=lambda s: abs(s["neutral_kl_per_token"] - vector_kl))
(OUT / "held_out.json").write_text(json.dumps(dict(neutral_held_every=NEUTRAL_HELD_EVERY, n_neutral_held=len(neutral_held), vector=VECTOR.stem, vector_neutral_kl_per_token=vector_kl, kl_matched_step=matched["step"], checkpoints=scores), indent=1))
print(f"{green}{len(scores)} adapters scored; {VECTOR.stem} has neutral KL per token {vector_kl:.5f}, nearest is step {matched['step']} at {matched['neutral_kl_per_token']:.5f}; saved {OUT}/held_out.json{endc}")
