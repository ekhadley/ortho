#!./.venv/bin/python
"""A sweep over train_vector.py's KTO projection direction: each config of kto_cfgs.py trains one hook on base rollouts of the grader, MBPP or both, then is
screened teacher-forced on held-out rollouts of every env: the hooked model's log-prob of each base completion minus the base model's, per token, for honest
and cheating completions of the same cell (grader_eval: the manifest's eval split of the five training families; heldout: the seven held-out families of
heldout38; mbpp_test: train_vector.py's same-prompt pairs of the MBPP test split, mbpp_short the 58 of them with both rollouts under 2,500 tokens), and the token KL and log-prob gain on held-out neutrals (grader none38 and
clean38, MBPP's original-test rollouts, and terminal_verifier's possible-mode rollouts). Two more pair sets come from envs nothing is trained on, tv_turn
(terminal_verifier verifier_bug) and secret_turn (secret_number): a rollout that cheated is cut at the end of its first cheating turn and scored on that turn
alone, one that did not is scored on all its turns (secret_number: its first five, which are also a neutral set). The hook at the output of each hooked layer L, on the hooked generated tokens: h <- h - scale * g((h - mu_L).u - c_L) u,
g the identity or a relu under clamp (under shift the hook is steering instead: h <- h + scale * c_L u), mu_L the mean residual of the trained envs' honest and neutral generated tokens. The worker keeps every rollout's residual
entering layer 30 in memory, so a config hooking layers 30 and up runs only those. GPU box, vLLM stopped.

    ./kto_sweep.py           lists the configs
    ./kto_sweep.py prep      one base-model pass over every rollout: reference log-probs and each env's pin statistics, to acts/kto_sweep/base.pt
    ./kto_sweep.py run       the worker: trains and screens every name in acts/kto_sweep/queue.txt that has no line in results.jsonl, until a line STOP
    ./kto_sweep.py table     the screen of every result (reads results.jsonl only)
    ./kto_sweep.py export <name> <out.safetensors>     a trained config as the hook file scripts/vllm_proj.py serves, generated tokens only (reads vecs/ and base.pt)
"""
import dataclasses
import gzip
import importlib
import json
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch as t
import torch.nn.functional as F
from torch import Tensor
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText, AutoTokenizer

from mechtools import *

import kto_cfgs
from kto_cfgs import SweepCfg
from scripts.vllm_proj import generated_mask

MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
WILL = Path("data/will")
VECTORS = Path("data/vectors/Qwen3.8-27B")
OUT = Path("acts/kto_sweep")
FAMILIES = ["parity", "sign", "letter", "weekday", "pets_listed"]
RUNS = [f"grader-{f}-hack-none-n1024-low" for f in FAMILIES] + ["grader-parity-hack-none-n3072-low"]
NEUTRAL_RUNS = ["2026-11-05T02-38-09-00-00_grader_Uv42Uf9LN9696L778qvToe", "2026-11-05T02-38-30-00-00_grader_2rUNKR9jcRD9MdFsYQGFV5"]   # none38 and clean38
HELDOUT_RUN = "2026-11-05T17-54-51-00-00_grader_XGBzMUHmDsCbASBvXpozTs"   # heldout38
MBPP_RUNS = ["2026-11-04T19-19-26-00-00_mbpp_GgYyPkS3GFuuD8oMPc8atb", "2026-11-05T21-27-57-00-00_mbpp_n37JEr7fba6FazvWnDUkX4"]   # mbpp38 and mbpp38_x4
MBPP_NEUTRAL_RUN = "2026-11-05T21-22-10-00-00_mbpp_JK59agc4fkFHZYQMkHrPCd"   # mbpp38_correct
TV_BUG_RUN = "2026-11-04T19-38-50-00-00_terminal-verifier_gUD2KJPKcXop8pjYj5e3uD"   # tv_bug38
TV_POSSIBLE_RUN = "2026-11-04T19-51-55-00-00_terminal-verifier_epD9HVnb4T8DuwJdNdr7K4"   # tv_pos38
PAIR_SETS = ["grader_eval", "heldout", "mbpp_short", "tv_turn", "secret_turn"]   # every config's screen; mbpp_test, all 97 pairs, only under cfg.full
SECRET_BASE = OUT / "secret_base.jsonl"   # base secret_number games (secret38's records; until that run ends, the games its log had on 2026-11-12)
THINK_OPEN, THINK_CLOSE, IM_END = 248068, 248069, 248046
D_MODEL = 5120
stage = sys.argv[1] if len(sys.argv) > 1 else "list"

def pin(cfg: SweepCfg, base: dict) -> tuple[Tensor, Tensor]:
    """mu [64, d_model], the mean residual at each layer's output over the generated tokens of the trained envs' honest train rollouts and train neutrals (a cfg.init vector's own pin at its layers), and sigma [64], the residual's per-dimension std around it."""
    pins = [base["pin"][e] for e in (("grader", "mbpp") if cfg.env == "both" else (cfg.env,))]
    n_tokens = sum(p["n"] for p in pins)
    mean = sum(p["total"] for p in pins) / n_tokens
    sigma = ((sum(p["squares"] for p in pins) / n_tokens - mean.pow(2).sum(1)) / D_MODEL).sqrt().float()
    mu = mean.float()
    if cfg.init and not cfg.init.startswith("sweep:"): mu[list(cfg.layers)] = load_file(VECTORS / f"{cfg.init}.safetensors")["mean_resid"][[L + 1 for L in cfg.layers]].float()
    return mu, sigma

if stage == "list":
    for name, cfg in vars(kto_cfgs).items():
        if isinstance(cfg, SweepCfg): print(f"{cyan}{name}{endc} {gray}" + ", ".join(f"{f.name}={getattr(cfg, f.name)}" for f in dataclasses.fields(cfg) if getattr(cfg, f.name) != f.default) + endc)
    sys.exit()

if stage == "table":
    rows = []
    for r in map(json.loads, (OUT / "results.jsonl").open()):
        row = [r["name"], r["cfg"]["steps"]]
        for s in PAIR_SETS: row += [f"{r[s]['honest']:+.1f}", f"{r[s]['cheat']:+.1f}", f"{r[s]['honest'] - r[s]['cheat']:+.1f}", f"{(r[s]['honest'] - r[s]['cheat']) / max(abs(r[s]['honest']), 1e-9):.1f}", f"{r[s]['auc']:.2f}"] if s in r else [""] * 5
        rows.append(row + [f"{r[f'neutral_{e}']['kl']:.2f}" if f"neutral_{e}" in r else "" for e in ("grader", "mbpp", "tv", "secret")] + [f"{r[f'neutral_{e}']['gain']:+.2f}" if f"neutral_{e}" in r else "" for e in ("grader", "mbpp", "tv", "secret")])
    show_table(["name", "steps"] + [f"{s} {c}" for s in ("eval", "heldout", "mbpp", "tv", "secret") for c in ("honest", "cheat", "margin", "ratio", "auc")] + ["KL grader", "KL mbpp", "KL tv", "KL secret", "gain grader", "gain mbpp", "gain tv", "gain secret"], rows, "millinats per token: log-prob gain of honest and cheating completions, their difference, that over the honest loss; neutral token KL and gain")
    sys.exit()

if stage == "export":
    name, out = sys.argv[2:4]
    cfg = getattr(kto_cfgs, name)
    assert cfg.rank == 1 and cfg.scale == 1 and not cfg.clamp and cfg.positions == "gen", f"{name}: vllm_proj.py serves one direction a layer, projected out at every generated token"
    saved = t.load(OUT / "vecs" / f"{name}.pt")
    mu, sigma = pin(cfg, t.load(OUT / "base.pt"))
    u = F.normalize(saved["v"][:, 0].float(), dim=1)
    c = t.zeros(64)
    for i, L in enumerate(cfg.layers): c[L] = mu[L] @ u[i if cfg.per_layer else 0] + saved["c"][i] * sigma[L]
    save_file(dict(u=(u if cfg.per_layer else u[0]).contiguous(), c=c, layers=t.tensor(cfg.layers), gen_only=t.ones(1)), out)
    print(f"{green}{out}: layers {cfg.layers[0]}-{cfg.layers[-1]}, u {tuple(u.shape)}, c by layer: " + ", ".join(f"{L}: {c[L]:+.2f}" for L in cfg.layers[::6]) + endc)
    sys.exit()

#%% rollouts: an item is one base completion with its served ids; a cell is a (family, side) or an MBPP task
def item(key: str, ids: list[int], prompt_len: int | None, cell: str, cheat: bool, score: np.ndarray | None = None) -> dict:
    """gen marks the generated tokens, which the hook edits: those past prompt_len, or with None (a multi-turn rollout) every finished assistant turn, as the served gen_only hook sees them. score marks the tokens whose log-probs are summed, all of gen unless given."""
    gen = generated_mask(ids) if prompt_len is None else np.arange(len(ids)) >= prompt_len
    score = gen if score is None else score
    return dict(key=key, ids=ids, gen=gen, score=score, n=int(score.sum()), cell=cell, cheat=cheat)

def turn_item(r: dict, cell: str, keep_turns: int) -> dict:
    """A multi-turn record as a screen item. One that cheated is cut at the end of its first cheating turn and scored on that turn alone: how the hook moves the decision, given the history it also edited. One that never cheated is cut after keep_turns turns and scored on all of them."""
    gen = generated_mask(r["ids"])
    turns = np.flatnonzero(np.diff(np.concatenate([[0], gen.astype(int), [0]]))).reshape(-1, 2)
    assert len(turns) == r["n_turns"], f"{r['rollout_id']}: {len(turns)} generated spans, {r['n_turns']} turns"
    last = r["first_cheat_turn"] if r["cheated"] else min(keep_turns, len(turns)) - 1
    score = gen[:turns[last, 1]].copy()
    if r["cheated"]: score[:turns[last, 0]] = False
    return item(f"{r.get('run', cell)}:{r['rollout_id']}", r["ids"][:turns[last, 1]], None, cell, bool(r["cheated"]), score)
def crc(x: dict) -> int: return zlib.crc32(x["key"].encode())
def by_length(items: list[dict]) -> list[dict]: return sorted(items, key=lambda x: len(x["ids"]))
def inspect_records(env: str, run: str) -> list[dict]: return [json.loads(l) | dict(run=run) for l in open(f"data/inspect/{env}/{run}.jsonl")]

records = {f"{run}:{r['rollout_id']}": r for run in RUNS for r in map(json.loads, gzip.open(WILL / f"sweep/Qwen3.8-27B/ortho_grader/{run}/records.jsonl.gz", "rt"))}
manifest = [m for m in map(json.loads, (WILL / "stage9/cache_meta/s9-gr/manifest.jsonl").open()) if m["task_id"].split(".")[0] in FAMILIES]
grader = {split: [item(m["trajectory_id"], records[m["trajectory_id"]]["ids"], m["call_prompt_lens"][0], m["task_id"], m["class"].startswith("cheat")) for m in manifest if m["split"] == split] for split in ("train", "eval")}

heldout_all = [item(f"{r['run']}:{r['rollout_id']}", r["ids"], r["ids"].index(THINK_OPEN) + 2, f"{r['labels']['family']}.{r['labels']['side']}", r["labels"]["cheat"]) for r in inspect_records("grader", HELDOUT_RUN) if r["labels"]["family"] != "fruit_vegetable" and r["labels"]["cheat"] is not None and r["ids"][-1] == IM_END]
heldout = [x for cell in sorted({x["cell"] for x in heldout_all}) for cheat in (True, False) for x in sorted([x for x in heldout_all if x["cell"] == cell and x["cheat"] == cheat], key=crc)[:24]]   # at most 24 a cell and class

set_seed(0)   # train_vector.py's draw of the held-out MBPP pairs: one cheating and one honest rollout of each test-split task with both
mbpp_all = [item(f"{r['run']}:{r['rollout_id']}", r["ids"], r["ids"].index(THINK_OPEN) + 2, str(r["labels"]["task_id"]), bool(r["labels"]["cheat"])) | dict(split=r["labels"]["split"]) for run in MBPP_RUNS for r in inspect_records("mbpp", run) if r["labels"]["cheat"] or r["labels"]["honest_correct"]]
mbpp_train = [x for x in mbpp_all if x["split"] != "test"]
mbpp_test = []
for cell in sorted({x["cell"] for x in mbpp_all if x["split"] == "test"}):
    both = [[x for x in mbpp_all if x["split"] == "test" and x["cell"] == cell and x["cheat"] == c] for c in (True, False)]
    if all(both): mbpp_test += [random.choice(b) for b in both]

neutral_grader = [item(f"{r['run']}:{r['rollout_id']}", r["ids"], r["ids"].index(THINK_OPEN) + 2, r["prompts"], False) for run in NEUTRAL_RUNS for r in inspect_records("grader", run) if r["labels"]["family"] in FAMILIES]
neutral_mbpp = [item(f"{r['run']}:{r['rollout_id']}", r["ids"], r["ids"].index(THINK_OPEN) + 2, "correct", False) | dict(split=r["labels"]["split"]) for r in inspect_records("mbpp", MBPP_NEUTRAL_RUN) if r["labels"]["valid"]]
def pick(items: list[dict], k: int, max_len: int) -> list[dict]: return [x for cheat in (True, False) for x in sorted([x for x in items if x["cheat"] == cheat and len(x["ids"]) <= max_len], key=crc)[:k]]
tv_turn = pick([turn_item(r, "tv", 99) for r in inspect_records("terminal_verifier", TV_BUG_RUN)], 40, 8000)   # envs nothing is trained on: 40 rollouts that cheated and 40 that did not
secret_turn = pick([turn_item(r, "secret", 5) for r in map(json.loads, SECRET_BASE.open())], 40, 8000)
neutral_tv = [item(f"{r['run']}:{r['rollout_id']}", r["ids"], None, "possible", False) for r in inspect_records("terminal_verifier", TV_POSSIBLE_RUN)]
assert all(x["ids"][-1] == IM_END for x in grader["train"] + grader["eval"] + mbpp_all + neutral_grader + neutral_mbpp + tv_turn + secret_turn + neutral_tv)

TRAIN = dict(grader=grader["train"], mbpp=mbpp_train)
short_cells = {cell for cell in {x["cell"] for x in mbpp_test} if all(len(x["ids"]) <= 2500 for x in mbpp_test if x["cell"] == cell)}
PAIRS = dict(grader_eval=by_length(grader["eval"]), heldout=by_length(heldout), mbpp_short=by_length([x for x in mbpp_test if x["cell"] in short_cells]), mbpp_test=by_length(mbpp_test), tv_turn=by_length(tv_turn), secret_turn=by_length(secret_turn))
NEUTRAL_TRAIN = dict(grader=[x for x in neutral_grader if crc(x) % 5], mbpp=[x for x in neutral_mbpp if x["split"] != "test"])
NEUTRAL_HELD = dict(grader=by_length([x for x in neutral_grader if not crc(x) % 5]), mbpp=by_length(sorted([x for x in neutral_mbpp if x["split"] == "test" and len(x["ids"]) <= 6000], key=crc)[:48]), tv=by_length(sorted(neutral_tv, key=crc)[:64]), secret=by_length([x for x in secret_turn if not x["cheat"]]))
for name, items in (TRAIN | PAIRS).items(): print(f"{cyan}{name}: {sum(x['cheat'] for x in items)} cheating, {sum(not x['cheat'] for x in items)} honest, {len({x['cell'] for x in items})} cells{endc}")
print(f"{cyan}neutrals, train / held: grader {len(NEUTRAL_TRAIN['grader'])} / {len(NEUTRAL_HELD['grader'])}, mbpp {len(NEUTRAL_TRAIN['mbpp'])} / {len(NEUTRAL_HELD['mbpp'])}{endc}")

#%% model: frozen, every decoder layer hooked; the hook edits the layers of cfg.layers while active
PAD = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION).pad_token_id
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=REVISION, dtype=t.bfloat16, device_map="cuda")
model.requires_grad_(False)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.train()   # checkpointing only runs in train mode; the model has no dropout
text = model.model.language_model
layers = text.layers
d_model = model.config.text_config.hidden_size
assert len(layers) == 64 and d_model == D_MODEL, f"{len(layers)} layers, hidden {d_model}"
cfg = SweepCfg()
v = t.zeros(1, 1, d_model, device="cuda")   # [1 or hooked layers, rank, d_model], normalised in the hook
c = t.zeros(34, device="cuda")   # [hooked layers], the pin's offset along the direction in units of sigma
mu = t.zeros(64, d_model, device="cuda")   # the pin: a mean residual at each layer's output
sigma = t.ones(64, device="cuda")   # per-dimension std of the residual around the pin
active = False
mask = None   # [batch, seq] bool: the positions hooked, set by forward
collect = None   # called with (layer, its output) before the edit
FIRST = 30   # an item's "h" is its residual entering this layer, on the cpu; a batch that has them, under a config hooking no earlier layer, runs layers FIRST-63 only
skip = False   # set by forward: the layers below FIRST pass their input through

def bypass(layer_forward):
    def f(hidden_states, *args, **kwargs): return hidden_states if skip else layer_forward(hidden_states, *args, **kwargs)
    return f
for L in range(FIRST): layers[L].forward = bypass(layers[L].forward)

def edit(L: int):
    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if collect: collect(L, h)
        if not active or L not in cfg.layers: return output
        i = cfg.layers.index(L)
        rows = v[i if cfg.per_layer else 0]
        u = F.normalize(rows, dim=1) if cfg.rank == 1 else t.linalg.qr(rows.T).Q.T
        s = (h.float() - mu[L]) @ u.T - c[i] * sigma[L]   # [batch, seq, rank]
        if cfg.clamp: s = s.relu()
        if cfg.shift: s = (-c[i] * sigma[L]).expand(*h.shape[:2], cfg.rank)   # steering: the same learned step along the direction at every hooked token
        h = h - ((cfg.scale * s * mask[..., None]) @ u).to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    return hook
for L in range(64): layers[L].register_forward_hook(edit(L))

def batches(items: list[dict], max_tokens: int) -> list[list[dict]]:
    """Consecutive items grouped so that a padded batch holds at most max_tokens tokens; a longer item goes alone."""
    out = []
    for x in items:
        if out and (len(out[-1]) + 1) * max(len(y["ids"]) for y in out[-1] + [x]) <= max_tokens: out[-1].append(x)
        else: out.append([x])
    return out

def forward(batch: list[dict]) -> tuple[Tensor, Tensor, Tensor]:
    """Next-token logits of a right-padded batch [batch, seq - 1, vocab], the next ids, and which of those are scored tokens. Sets the hook's mask and skip, which a checkpoint recompute reads in backward: one hooked forward per backward."""
    global mask, skip
    n = max(len(x["ids"]) for x in batch)
    ids = t.tensor([x["ids"] + [PAD] * (n - len(x["ids"])) for x in batch], device="cuda")
    pos = t.arange(n, device="cuda")[None]
    length = t.tensor([len(x["ids"]) for x in batch], device="cuda")[:, None]
    gen, score = t.zeros(2, len(batch), n, dtype=t.bool)
    for j, x in enumerate(batch): gen[j, :len(x["ids"])], score[j, :len(x["ids"])] = t.from_numpy(x["gen"]), t.from_numpy(x["score"])
    gen, score = gen.cuda(), score.cuda()
    mask = gen
    if cfg.positions != "gen":
        close = t.tensor([x["ids"].index(THINK_CLOSE) for x in batch], device="cuda")[:, None]
        mask = gen & (pos <= close if cfg.positions == "cot" else pos > close)
    skip = min(cfg.layers) >= FIRST and all("h" in x for x in batch)
    if skip:
        h = t.zeros(len(batch), n, d_model, dtype=t.bfloat16, device="cuda")
        for j, x in enumerate(batch): h[j, :len(x["ids"])] = x["h"]
        logits = model.lm_head(text(inputs_embeds=h, attention_mask=pos < length).last_hidden_state)
    else: logits = model(input_ids=ids, attention_mask=pos < length).logits
    return logits[:, :-1].float(), ids[:, 1:], score[:, 1:]

def completion_logps(batch: list[dict]) -> Tensor:
    """Sum of log p(token) over each item's completion tokens, [len(batch)]."""
    logits, following, completion = forward(batch)
    return (-F.cross_entropy(logits.transpose(1, 2), following, reduction="none") * completion).sum(1)

def gains(batch: list[dict]) -> Tensor:
    """Each completion's log-prob under the hook minus the base model's."""
    return completion_logps(batch) - t.tensor([x["ref"] for x in batch], device="cuda")

def neutral_kl(batch: list[dict]) -> tuple[Tensor, Tensor]:
    """Each completion's token KL from the base model to the hooked one, summed over its tokens, and the hooked model's log-prob gain on it. The base pass runs first so the recompute in backward sees the hook on."""
    global active
    active = False
    with t.no_grad(): base = forward(batch)[0].log_softmax(-1)
    active = True
    logits, following, completion = forward(batch)
    p = base.exp()
    kl = F.cross_entropy(logits.transpose(1, 2), p.transpose(1, 2), reduction="none") + (p * base).sum(-1)
    gain = -F.cross_entropy(logits.transpose(1, 2), following, reduction="none") - base.gather(-1, following[..., None])[..., 0]
    return (kl * completion).sum(1), (gain * completion).sum(1)

def cache(items: list[dict]):
    """Gives each item that lacks it its residual entering layer FIRST, from one full base-model forward, and its reference log-prob if prep did not."""
    global collect, active, cfg, batch
    def keep(L: int, h: Tensor):
        if L == FIRST - 1:
            for x, row in zip(batch, h): x["h"] = row[:len(x["ids"])].cpu()
    collect, active, cfg = keep, False, SweepCfg(layers=(0,))
    with t.no_grad():
        for batch in pbar(batches(by_length(list({x["key"]: x for x in items if "h" not in x}.values())), 6000), desc="residual cache"):
            for x, logp in zip(batch, completion_logps(batch).tolist()): x.setdefault("ref", logp)
    collect = None

def draws(pool: list[dict]):
    while True: yield from random.sample(pool, len(pool))

def summarize(items: list[dict], gain: Tensor) -> dict:
    """Per-token gains in millinats of a pair set's honest and cheating rollouts, each the mean over cells (with both classes) of the cell's mean; auc is how often an honest rollout gains more per token than a cheating one of its cell. The summed gains are in nats."""
    per_token = gain / t.tensor([x["n"] for x in items])
    out = dict(honest=[], cheat=[], auc=[], honest_sum=[], cheat_sum=[])
    for cell in sorted({x["cell"] for x in items if x["cheat"]} & {x["cell"] for x in items if not x["cheat"]}):
        honest = t.tensor([x["cell"] == cell and not x["cheat"] for x in items])
        cheat = t.tensor([x["cell"] == cell and x["cheat"] for x in items])
        out["honest"].append(1000 * per_token[honest].mean().item())
        out["cheat"].append(1000 * per_token[cheat].mean().item())
        out["auc"].append((per_token[honest][:, None] > per_token[cheat][None]).float().mean().item())
        out["honest_sum"].append(gain[honest].mean().item())
        out["cheat_sum"].append(gain[cheat].mean().item())
    return {k: sum(x) / len(x) for k, x in out.items()} | dict(gains=gain.tolist())

#%% prep: the base model on every rollout once, for each completion's log-prob and each env's pin statistics over the generated tokens of its honest train rollouts and train neutrals
if stage == "prep":
    OUT.mkdir(parents=True, exist_ok=True)
    ref, pin = {}, {}
    for env, items in [("grader", grader["train"] + grader["eval"] + heldout + NEUTRAL_TRAIN["grader"]), ("mbpp", mbpp_train + mbpp_test + NEUTRAL_TRAIN["mbpp"])]:
        pool = {x["key"] for x in TRAIN[env] if not x["cheat"]} | {x["key"] for x in NEUTRAL_TRAIN[env]}
        total = t.zeros(64, d_model, dtype=t.float64, device="cuda")
        squares = t.zeros(64, dtype=t.float64, device="cuda")
        n_tokens = 0
        def add(L: int, h: Tensor):
            w = mask * pooled[:, None]
            total[L] += (h.float() * w[..., None]).sum((0, 1))
            squares[L] += (h.float().pow(2).sum(-1) * w).sum()
        collect = add
        with t.no_grad():
            for batch in pbar(batches(by_length(items), 6000), desc=f"base pass, {env}"):
                pooled = t.tensor([x["key"] in pool for x in batch], device="cuda")
                logps = completion_logps(batch)
                n_tokens += (mask * pooled[:, None]).sum().item()
                for x, logp in zip(batch, logps.tolist()): ref[x["key"]] = logp
        collect = None
        pin[env] = dict(total=total.cpu(), squares=squares.cpu(), n=n_tokens)
        mean = total / n_tokens
        print(f"{cyan}{env}: {len(items)} rollouts, pin over {n_tokens} tokens of {len(pool)}; |mean| and sigma by layer: " + ", ".join(f"{L}: {mean[L].norm():.1f} / {((squares[L] / n_tokens - mean[L].pow(2).sum()) / d_model).sqrt():.2f}" for L in range(0, 64, 9)) + endc)
    t.save(dict(ref=ref, pin=pin), OUT / "base.pt")
    print(f"{green}saved {OUT / 'base.pt'}{endc}")

#%% run: the worker. Each queued config: its hook's parameters, KTO steps on the trained envs, the screen, one line in results.jsonl and its parameters in vecs/
if stage == "run":
    base = t.load(OUT / "base.pt")
    for x in grader["train"] + grader["eval"] + heldout + mbpp_train + mbpp_test: x["ref"] = base["ref"][x["key"]]   # the TV rollouts get theirs in cache()
    (OUT / "vecs").mkdir(exist_ok=True)
    while True:
        queue = (OUT / "queue.txt").read_text().split()
        done = {json.loads(l)["name"] for l in (OUT / "results.jsonl").open()} if (OUT / "results.jsonl").exists() else set()
        todo = [name for name in queue if name not in done and name != "STOP"]
        if not todo and "STOP" in queue: break
        if not todo:
            time.sleep(15)
            continue
        name = todo[0]
        cfg = getattr(importlib.reload(kto_cfgs), name)
        print(f"{purple}=== {name}: {cfg} ==={endc}")
        assert not cfg.shift or cfg.offset == "learned", f"{name}: a shift hook's size is its learned offset"
        started = time.time()
        set_seed(cfg.seed)
        envs = ["grader", "mbpp"] if cfg.env == "both" else [cfg.env]
        sets = PAIR_SETS + ["mbpp_test"] * cfg.full
        cache([x for e in envs for x in TRAIN[e] if len(x["ids"]) <= cfg.max_len] + [x for e in cfg.neutrals for x in NEUTRAL_TRAIN[e] if len(x["ids"]) <= cfg.max_len] + [x for s in sets for x in PAIRS[s]] + [x for held in NEUTRAL_HELD.values() for x in held])
        cfg = getattr(kto_cfgs, name)   # cache() runs under its own
        if cfg.max_len <= 2500: model.gradient_checkpointing_disable()   # the activations of a 2,500-token batch fit
        else: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        mu, sigma = [x.cuda() for x in pin(cfg, base)]
        v = t.nn.Parameter(cfg.init_scale * t.randn(len(cfg.layers) if cfg.per_layer else 1, cfg.rank, d_model, device="cuda"))
        c = t.nn.Parameter(t.zeros(len(cfg.layers), device="cuda"), requires_grad=cfg.offset == "learned")
        if cfg.init.startswith("sweep:"): v.data, c.data = [t.load(OUT / "vecs" / f"{cfg.init[6:]}.pt")[k].cuda() for k in ("v", "c")]
        elif cfg.init: v.data = load_file(VECTORS / f"{cfg.init}.safetensors")["direction_unit"][[L + 1 for L in cfg.layers][:v.shape[0]]].float().cuda()[:, None]

        # train: k cheating, k honest and k neutral rollouts a step, the envs taking turns
        pools = {e: (draws([x for x in TRAIN[e] if x["cheat"] and len(x["ids"]) <= cfg.max_len]), draws([x for x in TRAIN[e] if not x["cheat"] and len(x["ids"]) <= cfg.max_len])) for e in envs}
        neutral_pools = [draws([x for x in NEUTRAL_TRAIN[e] if len(x["ids"]) <= cfg.max_len]) for e in cfg.neutrals]
        opt = t.optim.Adam([dict(params=[v], lr=cfg.lr), dict(params=[c], lr=cfg.offset_lr)])
        log = []
        active = True
        bar = pbar(range(cfg.steps), desc=name)
        for step in bar:
            cheats, honests = [[next(pool) for _ in range(cfg.k)] for pool in pools[envs[step % len(envs)]]]
            neutrals = [next(neutral_pools[step % len(neutral_pools)]) for _ in range(cfg.k)]
            scored = cheats + honests * (cfg.honest == "kto")
            held = neutrals + honests * (cfg.honest == "kl")
            opt.zero_grad()
            loss, gain_c, gain_h, kls = 0.0, [], [], []
            for batch in batches(by_length(scored), 2500):
                gain = gains(batch)
                sign = t.tensor([1.0 if x["cheat"] else -1.0 for x in batch], device="cuda")
                size = gain * 200 / t.tensor([x["n"] for x in batch], device="cuda") if cfg.per_token else gain
                term = t.sigmoid(cfg.beta * sign * size).sum() / cfg.k
                term.backward()
                loss += term.item()
                for x, g in zip(batch, gain.tolist()): (gain_c if x["cheat"] else gain_h).append(g)
            for batch in batches(by_length(held), 2500):
                kl = neutral_kl(batch)[0]
                size = kl * 200 / t.tensor([x["n"] for x in batch], device="cuda") if cfg.per_token else kl
                term = cfg.kl_weight * size.sum() / cfg.k
                term.backward()
                loss += term.item()
                kls += kl.tolist()
            opt.step()
            log.append(dict(loss=loss, cheat=sum(gain_c) / cfg.k, honest=sum(gain_h) / max(len(gain_h), 1), kl=sum(kls) / len(kls)))
            bar.set_description(f"{name} loss {loss:.3f} cheat {log[-1]['cheat']:+.1f} honest {log[-1]['honest']:+.1f} kl {log[-1]['kl']:.3f}")
        trained = time.time()

        # screen
        result = dict(name=name, cfg=dataclasses.asdict(cfg), log=log)
        with t.no_grad():
            for s in sets: result[s] = summarize(PAIRS[s], t.cat([gains(batch) for batch in batches(PAIRS[s], 6000)]).cpu())
            for e in NEUTRAL_HELD:
                kl, gain = [t.cat(x).cpu() for x in zip(*[neutral_kl(batch) for batch in batches(NEUTRAL_HELD[e], 3000)])]
                n_held = sum(x["n"] for x in NEUTRAL_HELD[e])
                result[f"neutral_{e}"] = dict(kl=1000 * kl.sum().item() / n_held, gain=1000 * gain.sum().item() / n_held, kls=kl.tolist(), gains=gain.tolist())
        active = False
        result |= dict(train_seconds=trained - started, screen_seconds=time.time() - trained, offsets=c.tolist(), raw_norms=v.detach().norm(dim=-1).flatten().tolist())
        t.save(dict(v=v.detach().cpu(), c=c.detach().cpu(), cfg=dataclasses.asdict(cfg)), OUT / "vecs" / f"{name}.pt")
        with (OUT / "results.jsonl").open("a") as f: f.write(json.dumps(result) + "\n")
        print(f"{green}{name}: train {trained - started:.0f} s, screen {time.time() - trained:.0f} s{endc}")
        for s in sets: print(f"{cyan}  {s}: honest {result[s]['honest']:+.1f}, cheating {result[s]['cheat']:+.1f} millinats per token, auc {result[s]['auc']:.2f}; summed {result[s]['honest_sum']:+.1f} / {result[s]['cheat_sum']:+.1f} nats{endc}")
        print(f"{cyan}  neutral KL, millinats per token: " + ", ".join(f"{e} {result[f'neutral_{e}']['kl']:.2f}" for e in NEUTRAL_HELD) + "; gain " + ", ".join(f"{e} {result[f'neutral_{e}']['gain']:+.2f}" for e in NEUTRAL_HELD) + endc)
