#%%
"""A linear probe on single CoT tokens of Qwen3.8-27B, fit on two environments at once: one token's residual at one layer in, cheat or not out, the rollout's label on every CoT
token. Grader rollouts are Will's s9-gr manifest rows of the five training families. MBPP rollouts are the base rollouts of mbpp38 and mbpp38_x4 that cheated or were honest and
correct, bool corruptions dropped, only tasks with both classes, the test split held out. Each env is half of the loss. A grader-only and an MBPP-only probe are fit the same way
as references. The joint probe is then scored token by token on whole rollouts it never saw: the held-out grader and MBPP rollouts, the grader's held-out families (heldout38)
and the secret_number base run (secret38). Kernel cell script, GPU box for harvest, fit_probes, evaluate and score; the later cells read what those saved."""
import gzip
import html
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch as t
from safetensors.torch import load_file, save_file
from sklearn.metrics import roc_auc_score
from torch import Tensor
from transformers import AutoModelForImageTextToText, AutoTokenizer

from mechtools import *

t.set_grad_enabled(False)
MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
WILL = Path("data/will")
ACTS = Path("acts/joint_probe")   # per-row activations of every CoT token (box only), each fit's logits on them, and the joint probe's scores on whole rollouts
OUT = Path("data/vectors/Qwen3.8-27B")
ENVS = ["grader", "mbpp"]
FAMILIES = ["parity", "sign", "letter", "weekday", "pets_listed"]
GRADER_RUNS = [f"grader-{f}-hack-none-n1024-low" for f in FAMILIES] + ["grader-parity-hack-none-n3072-low"]
MBPP_RUNS = ["data/inspect/mbpp/2026-11-04T19-19-26-00-00_mbpp_GgYyPkS3GFuuD8oMPc8atb.jsonl", "data/inspect/mbpp/2026-11-05T21-27-57-00-00_mbpp_n37JEr7fba6FazvWnDUkX4.jsonl"]   # mbpp38 and mbpp38_x4: five base rollouts of every task
SCORE_RUNS = dict(
    heldout="data/inspect/grader/2026-11-05T17-54-51-00-00_grader_XGBzMUHmDsCbASBvXpozTs.jsonl",   # heldout38: grader families no probe trained on
    secret="data/inspect/secret_number/2026-11-11T23-49-15-00-00_secret-number_oBwHh4bKuKJyDq6h8rFZ3J.jsonl",   # secret38
)
ROWS = [20, 28, 36, 44, 52, 60]   # output_hidden_states rows: row i + 1 is the output of decoder layer i
THINK_OPEN = 248068
THINK_CLOSE = 248069
SEED = 0

def cell_auroc(scores: Tensor, mask: Tensor) -> float:
    """AUROC of cheat against clean among the masked CoT tokens, within each cell holding both, every rollout weighing the same, averaged over cells. A cell's rollouts share one
    prompt (a grader family and side, or an MBPP task), so this cannot be won by telling prompts apart. A per-rollout score broadcast to its tokens gives the rollout-level AUROC."""
    weight = 1 / t.bincount(owner[mask], minlength=len(items)).float()
    aurocs = []
    for in_cell in cell_of.values():
        sel = mask & in_cell
        label = is_cheat[owner[sel]]
        if label.any() and not label.all(): aurocs.append(roc_auc_score(label.numpy(), scores[sel].numpy(), sample_weight=weight[owner[sel]].numpy()))
    return sum(aurocs) / len(aurocs)

def fit_probe(z: Tensor, mean: Tensor, std: Tensor, label: Tensor, weight: Tensor, l2: float, steps: int, lr: float) -> tuple[Tensor, Tensor, float]:
    """Weighted logistic regression on standardized activations by full-batch Adam, L2 on the standardized weights. Returns the probe in raw residual coordinates (logit =
    resid @ w + b) and the final loss."""
    w = t.zeros(z.shape[1], device=z.device, requires_grad=True)
    b = t.zeros((), device=z.device, requires_grad=True)
    opt = t.optim.Adam([w, b], lr=lr)
    with t.enable_grad():
        for step in range(steps):
            loss = (weight * t.nn.functional.binary_cross_entropy_with_logits(z @ w + b, label, reduction="none")).sum() + l2 * w.pow(2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    w_raw = w.detach() / std
    return w_raw, b.detach() - mean @ w_raw, loss.item()

def apply_probe(a: Tensor, w: Tensor, b: Tensor) -> Tensor:
    """The probe's logit on every row of a bf16 activation matrix, on the cpu."""
    return t.cat([chunk.float() @ w + b for chunk in a.split(200_000)]).cpu()

#%% one item per rollout. grader: train and dev from the manifest's train split (dev a random fifth of its rollouts), eval its eval split. mbpp: eval the test split, dev a random
# fifth of the other tasks, so a dev task is unseen. Then which item each CoT token belongs to, and each fit's weight on it

set_seed(SEED)
records = {}
for run in GRADER_RUNS:
    for line_ in gzip.open(WILL / f"sweep/Qwen3.8-27B/ortho_grader/{run}/records.jsonl.gz", "rt"):
        r = json.loads(line_)
        records[f"{run}:{r['rollout_id']}"] = r
items = []
for m in map(json.loads, (WILL / "stage9/cache_meta/s9-gr/manifest.jsonl").read_text().splitlines()):
    if m["task_id"].split(".")[0] not in FAMILIES: continue
    split = "eval" if m["split"] == "eval" else "dev" if random.random() < 0.2 else "train"
    items.append(dict(env="grader", key=m["trajectory_id"], ids=records[m["trajectory_id"]]["ids"], cell=m["task_id"], cheat=m["class"].startswith("cheat"), split=split))
del records
mbpp = [r for path in MBPP_RUNS for r in map(json.loads, open(path)) if (r["labels"]["cheat"] or r["labels"]["honest_correct"]) and "bool" not in r["labels"]["corruption"]]
classes = defaultdict(set)
for r in mbpp: classes[r["labels"]["task_id"]].add(bool(r["labels"]["cheat"]))
fit_tasks = sorted({r["labels"]["task_id"] for r in mbpp if r["labels"]["split"] != "test" and len(classes[r["labels"]["task_id"]]) == 2})
dev_tasks = set(random.sample(fit_tasks, len(fit_tasks) // 5))
for r in mbpp:
    task = r["labels"]["task_id"]
    if len(classes[task]) < 2: continue
    split = "eval" if r["labels"]["split"] == "test" else "dev" if task in dev_tasks else "train"
    items.append(dict(env="mbpp", key=r["rollout_id"], ids=r["ids"], cell=f"mbpp.{task}", cheat=bool(r["labels"]["cheat"]), split=split))
del mbpp
assert len({x["key"] for x in items}) == len(items), "rollout keys repeat"
for x in items:
    assert x["ids"].count(THINK_OPEN) == 1 and x["ids"].count(THINK_CLOSE) == 1, x["key"]
    x["start"] = x["ids"].index(THINK_OPEN) + 2   # the first generated token: the prompt ends with <think> and a newline
    x["close"] = x["ids"].index(THINK_CLOSE)
cot_len = t.tensor([x["close"] - x["start"] for x in items])
offset = cot_len.cumsum(0) - cot_len   # an item's CoT tokens are offset : offset + cot_len in every per-token tensor
owner = t.repeat_interleave(t.arange(len(items)), cot_len)   # the item of every CoT token
index = t.cat([t.arange(n) for n in cot_len.tolist()])   # its position in the CoT
is_cheat = t.tensor([x["cheat"] for x in items])
in_split = {s: t.tensor([x["split"] == s for x in items])[owner] for s in ["train", "dev", "eval"]}
in_env = {e: t.tensor([x["env"] == e for x in items])[owner] for e in ENVS}
cell_of = {c: t.tensor([x["cell"] == c for x in items])[owner] for c in sorted({x["cell"] for x in items})}
n_group = Counter((x["cell"], x["cheat"]) for x in items if x["split"] == "train")
per_token = (t.tensor([1 / n_group[(x["cell"], x["cheat"])] if x["split"] == "train" else 0.0 for x in items]) / cot_len)[owner]   # every (cell, class) weighs the same, every rollout the same within it
env_weight = {e: per_token * in_env[e] / (per_token * in_env[e]).sum() for e in ENVS}
fit_weight = dict(joint=(env_weight["grader"] + env_weight["mbpp"]) / 2, grader=env_weight["grader"], mbpp=env_weight["mbpp"])   # the three probes: both envs at half the loss each, and each env alone
print(f"{cyan}{len(items)} rollouts, {len(owner)} CoT tokens{endc}")
for e in ENVS:
    for s in in_split:
        members = [x for x in items if x["env"] == e and x["split"] == s]
        print(f"{gray}  {e} {s}: {len({x['cell'] for x in members})} cells, {sum(x['cheat'] for x in members)} cheat / {sum(not x['cheat'] for x in members)} clean rollouts, {(in_env[e] & in_split[s]).sum()} CoT tokens{endc}")

#%% residuals of every CoT token at ROWS, one rollout at a time so no padding enters. The input stops at </think>: nothing before it depends on what follows

harvest = False
if harvest:
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=REVISION, dtype=t.bfloat16, device_map="cuda").eval()
    acts = {r: [] for r in ROWS}
    for x in pbar(items, desc="harvest"):
        hidden = model.model(input_ids=t.tensor([x["ids"][:x["close"]]], device="cuda"), output_hidden_states=True).hidden_states
        assert len(hidden) == 65, f"{len(hidden)} hidden states"
        for r in ROWS: acts[r].append(hidden[r][0, x["start"]:].cpu())
    ACTS.mkdir(parents=True, exist_ok=True)
    for r in ROWS: save_file({"acts": t.cat(acts.pop(r))}, ACTS / f"row{r}.safetensors")
    del model, acts, hidden
    tec()

#%% one logistic probe per fit, row and L2 weight on the train CoT tokens. The joint probe's row and L2 are chosen by the mean of the two envs' within-cell token AUROC on dev,
# a single-env probe's by its own env's. Saves the chosen probes in raw residual coordinates, and their logits on every CoT token

fit_probes = False
if fit_probes:
    l2s = [1e-5, 1e-4, 1e-3, 1e-2]
    steps = 400
    lr = 1e-2
    train = in_split["train"]
    label = is_cheat[owner[train]].float().cuda()
    table = []
    best = {name: dict(dev=0.0) for name in fit_weight}
    for r in ROWS:
        a = load_file(ACTS / f"row{r}.safetensors")["acts"].cuda()
        z_train = a[train.cuda()].float()
        mean = z_train.mean(0)
        std = z_train.std(0)
        assert std.min() > 0, f"row {r}: a dimension is constant"
        z_train.sub_(mean).div_(std)
        for name, weight in fit_weight.items():
            for l2 in l2s:
                w, b, loss = fit_probe(z_train, mean, std, label, weight[train].cuda(), l2, steps, lr)
                logits = apply_probe(a, w, b)
                dev = {e: cell_auroc(logits, in_split["dev"] & in_env[e]) for e in ENVS}
                dev_score = (dev["grader"] + dev["mbpp"]) / 2 if name == "joint" else dev[name]
                table.append(dict(fit=name, row=r, l2=l2, loss=loss, dev_grader=dev["grader"], dev_mbpp=dev["mbpp"]))
                print(f"{gray}{name} row {r} l2 {l2:g}: loss {loss:.3f}, dev grader {dev['grader']:.3f}, mbpp {dev['mbpp']:.3f}{endc}")
                if dev_score > best[name]["dev"]: best[name] = dict(row=r, l2=l2, dev=dev_score, w=w.cpu(), b=b.cpu(), logits=logits)
        del a, z_train
        tec()
    for name, x in best.items(): print(f"{cyan}{name}: row {x['row']} (decoder layer {x['row'] - 1}), l2 {x['l2']:g}, dev {x['dev']:.3f}{endc}")
    save_file({(k if name == "joint" else f"{name}_only_{k}"): best[name][k].reshape(-1).contiguous() for name in best for k in ["w", "b"]}, OUT / "joint_probe.safetensors")
    save_file({name: best[name]["logits"].contiguous() for name in best}, ACTS / "logits.safetensors")
    chosen = {name: dict(row=x["row"], l2=x["l2"], dev=x["dev"]) for name, x in best.items()}
    layout = "logit = resid @ w + b at output_hidden_states row `row` (decoder layer row - 1), trained on CoT tokens; w and b are the joint probe, <env>_only_w and <env>_only_b the single-env ones"
    (OUT / "joint_probe.json").write_text(json.dumps(dict(model=MODEL_ID, revision=REVISION, chosen=chosen, steps=steps, lr=lr, seed=SEED, families=FAMILIES, mbpp_runs=MBPP_RUNS, layout=layout, table=table), indent=1))

#%% the probes on rollouts they never saw, per env, beside things that should do worse: the joint fit on labels shuffled between the rollouts of a cell (n_shuffles times, since
# MBPP's cells are a handful of rollouts each), each token's identity alone, its position alone, and the rollout's CoT length. Then when the signal arrives: the joint probe by tenth of the CoT on the held-out rollouts

evaluate = False
if evaluate:
    n_bins = 10
    n_shuffles = 5
    fit = json.loads((OUT / "joint_probe.json").read_text())
    logits = load_file(ACTS / "logits.safetensors")
    train = in_split["train"]
    columns = {f"{e} {s}": in_split[s] & in_env[e] for e in ENVS for s in in_split}
    rollout_mean = lambda v: (t.zeros(len(items)).index_add_(0, owner, v) / cot_len)[owner]   # a rollout's mean over its CoT, on each of its tokens
    a = load_file(ACTS / f"row{fit['chosen']['joint']['row']}.safetensors")["acts"].cuda()
    z_train = a[train.cuda()].float()
    mean = z_train.mean(0)
    std = z_train.std(0)
    z_train.sub_(mean).div_(std)
    members_of = {c: [k for k, x in enumerate(items) if x["cell"] == c] for c in cell_of}
    shuffles = {"shuffled labels, tokens": [], "shuffled labels, rollout mean": []}
    for _ in range(n_shuffles):
        shuffled = is_cheat.clone()
        for members in members_of.values(): shuffled[members] = is_cheat[random.sample(members, len(members))]
        w, b, _ = fit_probe(z_train, mean, std, shuffled[owner[train]].float().cuda(), fit_weight["joint"][train].cuda(), fit["chosen"]["joint"]["l2"], fit["steps"], fit["lr"])
        shuffled_logits = apply_probe(a, w, b)
        shuffles["shuffled labels, tokens"].append({col: cell_auroc(shuffled_logits, mask) for col, mask in columns.items()})
        shuffles["shuffled labels, rollout mean"].append({col: cell_auroc(rollout_mean(shuffled_logits), mask) for col, mask in columns.items()})
    del a, z_train
    tec()
    token_ids = t.cat([t.tensor(x["ids"][x["start"]:x["close"]]) for x in items])
    n_vocab = token_ids.max().item() + 1
    cheat_count = t.zeros(n_vocab).index_add_(0, token_ids[train], fit_weight["joint"][train] * is_cheat[owner[train]].float())
    clean_count = t.zeros(n_vocab).index_add_(0, token_ids[train], fit_weight["joint"][train] * (~is_cheat[owner[train]]).float())
    prior = 1 / len(owner)
    token_logits = ((cheat_count + prior) / (clean_count + prior)).log()[token_ids]
    scores = {f"{name} probe, {level}": (v if level == "tokens" else rollout_mean(v)) for name, v in logits.items() for level in ["tokens", "rollout mean"]}
    scores |= {"token identity": token_logits, "token position": index.float(), "CoT length": cot_len[owner].float()}
    results = {name: {col: cell_auroc(v, mask) for col, mask in columns.items()} for name, v in scores.items()}
    results |= {f"{name}, mean of {n_shuffles} shuffles": {col: sum(run[col] for run in runs) / n_shuffles for col in columns} for name, runs in shuffles.items()}
    shuffle_range = {name: {col: [min(run[col] for run in runs), max(run[col] for run in runs)] for col in columns} for name, runs in shuffles.items()}
    print(f"{cyan}within-cell AUROC, cheat vs clean{endc}")
    for name, by_split in results.items(): print(f"  {name:44s} " + "  ".join(f"{s} {v:.3f}" for s, v in by_split.items()))
    for name, by_split in shuffle_range.items(): print(f"{gray}  {name}, lowest and highest: " + "  ".join(f"{s} {lo:.3f}-{hi:.3f}" for s, (lo, hi) in by_split.items()) + endc)
    tenth = (index.float() / cot_len[owner].float() * n_bins).long().clamp(max=n_bins - 1)
    curves = {}
    for e in ENVS:
        held = in_split["eval"] & in_env[e]
        curves[e] = dict(
            auroc=[cell_auroc(logits["joint"], held & (tenth == k)) for k in range(n_bins)],
            cheat=[logits["joint"][held & (tenth == k) & is_cheat[owner]].mean().item() for k in range(n_bins)],
            clean=[logits["joint"][held & (tenth == k) & ~is_cheat[owner]].mean().item() for k in range(n_bins)],
        )
        print(f"{cyan}{e}, joint probe: tenth of CoT   AUROC   cheat   clean{endc}")
        for k in range(n_bins): print(f"  {k + 1:2d}           {curves[e]['auroc'][k]:.3f}   {curves[e]['cheat'][k]:+.2f}   {curves[e]['clean'][k]:+.2f}")
    (ACTS / "results.json").write_text(json.dumps(dict(auroc=results, shuffle_range=shuffle_range, curves=curves), indent=1))

#%% the joint probe's logit at every position of whole rollouts, one forward pass each, read by a hook at its layer: the held-out grader and MBPP rollouts (answer included),
# the held-out grader families and the secret_number rollouts (every turn of the game in one sequence). Needs the model. Checks the hook against the fit's logits on the CoT

score = False
if score:
    score_sets = ["eval", "heldout", "secret"]
    fit = json.loads((OUT / "joint_probe.json").read_text())
    probe = load_file(OUT / "joint_probe.safetensors")
    w = probe["w"].cuda()
    b = probe["b"].cuda()
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=REVISION, dtype=t.bfloat16, device_map="cuda").eval()
    kept = []
    model.model.language_model.layers[fit["chosen"]["joint"]["row"] - 1].register_forward_hook(lambda module, args, out: kept.append(((out[0] if isinstance(out, tuple) else out)[0].float() @ w + b).cpu()))
    for name in score_sets:
        ids_of = {x["key"]: x["ids"] for x in items if x["split"] == "eval"} if name == "eval" else {r["rollout_id"]: r["ids"] for r in map(json.loads, open(SCORE_RUNS[name]))}
        scored = {}
        for key, ids in pbar(ids_of.items(), desc=name):
            model.model(input_ids=t.tensor([ids], device="cuda"))
            scored[key] = kept.pop()
        t.save(scored, ACTS / f"scores_{name}.pt")
        if name == "eval":
            fit_logits = load_file(ACTS / "logits.safetensors")["joint"]
            gaps = t.cat([(scored[x["key"]][x["start"]:x["close"]] - fit_logits[offset[k]:offset[k] + cot_len[k]]).abs() for k, x in enumerate(items) if x["split"] == "eval"])
            print(f"{cyan}hook against the fit's logits on {len(gaps)} held-out CoT tokens: mean gap {gaps.mean():.4f} logits, {(gaps > 0.5).sum()} tokens above 0.5, largest {gaps.max():.2f}{endc}")
            assert gaps.mean() < 0.05, "the hook does not read what the probe was fit on"   # the two passes differ in length, and in bf16 a few isolated tokens move by a logit
    del model
    tec()

#%% the joint probe on what neither env's fit saw. Grader held-out families: within-cell AUROC over the CoT, a cell one family and side, answers on neither side and the removed
# fruit_vegetable family dropped. secret_number, every rollout the same game: each rollout's mean logit over the CoT tokens of all its turns against whether it cheated, and a
# turn's mean CoT logit, the turns whose command cheated against the same turn of the rollouts that never cheat

held_out = True
if held_out:
    scores_heldout = t.load(ACTS / "scores_heldout.pt")
    heldout = [r for r in map(json.loads, open(SCORE_RUNS["heldout"])) if r["labels"]["family"] != "fruit_vegetable" and r["labels"]["cheat"] is not None]
    for r in heldout: r["cot"] = scores_heldout[r["rollout_id"]][r["ids"].index(THINK_OPEN) + 2:r["ids"].index(THINK_CLOSE)]
    heldout_cells = sorted({f"{r['labels']['family']}.{r['labels']['side']}" for r in heldout})
    token_aurocs = []
    rollout_aurocs = []
    for c in heldout_cells:
        members = [r for r in heldout if f"{r['labels']['family']}.{r['labels']['side']}" == c]
        label = t.tensor([r["labels"]["cheat"] for r in members])
        n_tokens = t.tensor([len(r["cot"]) for r in members])
        token_aurocs.append(roc_auc_score(label.repeat_interleave(n_tokens).numpy(), t.cat([r["cot"] for r in members]).numpy(), sample_weight=(1 / n_tokens.float()).repeat_interleave(n_tokens).numpy()))
        rollout_aurocs.append(roc_auc_score(label.numpy(), t.stack([r["cot"].mean() for r in members]).numpy()))
        print(f"{gray}  {c:28s} tokens {token_aurocs[-1]:.3f}  rollout mean {rollout_aurocs[-1]:.3f}  ({label.sum()} cheat / {(~label).sum()} clean){endc}")
    heldout_tokens = sum(token_aurocs) / len(token_aurocs)
    heldout_rollouts = sum(rollout_aurocs) / len(rollout_aurocs)
    print(f"{cyan}held-out grader families, {len(heldout)} rollouts in {len(heldout_cells)} cells: tokens {heldout_tokens:.3f}, rollout mean {heldout_rollouts:.3f}{endc}")
    scores_secret = t.load(ACTS / "scores_secret.pt")
    secret = list(map(json.loads, open(SCORE_RUNS["secret"])))
    for r in secret:
        opens = [p for p, i in enumerate(r["ids"]) if i == THINK_OPEN]
        closes = [p for p, i in enumerate(r["ids"]) if i == THINK_CLOSE]
        assert len(opens) == len(closes) == len(r["turns"]), r["rollout_id"]
        r["spans"] = [(a + 2, b) for a, b in zip(opens, closes)]   # each turn's CoT
        r["turn_means"] = [scores_secret[r["rollout_id"]][a:b].mean().item() for a, b in r["spans"]]
        r["mean"] = t.cat([scores_secret[r["rollout_id"]][a:b] for a, b in r["spans"]]).mean().item()
    secret_cheated = t.tensor([bool(r["cheated"]) for r in secret])
    secret_auroc = roc_auc_score(secret_cheated.numpy(), [r["mean"] for r in secret])
    cheat_turns = defaultdict(list)   # by turn index: the mean CoT logit of every turn whose command cheated
    clean_turns = defaultdict(list)   # and of that turn in the rollouts that never cheat
    for r in secret:
        for k, turn in enumerate(r["turns"]):
            if turn["cheated"]: cheat_turns[k].append(r["turn_means"][k])
            elif not r["cheated"]: clean_turns[k].append(r["turn_means"][k])
    matched = {k: roc_auc_score([True] * len(cheat_turns[k]) + [False] * len(clean_turns[k]), cheat_turns[k] + clean_turns[k]) for k in sorted(cheat_turns) if clean_turns[k]}
    n_cheat_turns = sum(len(cheat_turns[k]) for k in matched)
    secret_turn_auroc = sum(len(cheat_turns[k]) * auroc for k, auroc in matched.items()) / n_cheat_turns   # each turn index weighed by its cheating turns
    print(f"{cyan}secret_number, {len(secret)} rollouts, {secret_cheated.sum()} cheating: rollout mean AUROC {secret_auroc:.3f}; {n_cheat_turns} cheating turns against the same turn of clean rollouts {secret_turn_auroc:.3f}{endc}")
    for k, auroc in matched.items(): print(f"{gray}  turn {k}: {len(cheat_turns[k])} cheating turns, {len(clean_turns[k])} clean, AUROC {auroc:.3f}{endc}")

#%% token-level view of rollouts the probe never saw, as one self-contained page. Prompt and tool output unshaded, generated tokens shaded by the joint probe's logit (red toward
# cheat, blue toward clean), </think> outlined. Every strip shares one colour scale, set by the block at its end; hover a token for its logit, clipped to that scale. Grader and
# MBPP rows are a cheating and a clean rollout of one prompt; a secret_number rollout is one strip per turn under a bar per turn of its mean CoT logit

save_page = True
if save_page:
    clip = 4.0   # logits; 97% of the fit sets' CoT tokens are inside it
    n_mbpp = 4   # tasks
    max_mbpp_tokens = 1800   # per rollout, so a strip stays readable
    n_secret = 3   # rollouts per class
    max_secret_tokens = 14000
    path = Path.home() / "wgmn/artifacts/2026-11-11_joint-probe-tokens.html"
    random.seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    fit = json.loads((OUT / "joint_probe.json").read_text())
    saved = json.loads((ACTS / "results.json").read_text())
    scores_eval = t.load(ACTS / "scores_eval.pt")
    box = lambda head, strip: f"<div class=strip><div class=head>{head}</div>{strip}</div>"

    def single_turn(ids: list[int], values: Tensor, head: str) -> str:
        """One strip of a one-call rollout: the prompt unshaded, every generated token shaded."""
        start = ids.index(THINK_OPEN) + 2
        vals = [0.0] * start + values[start:].clamp(-clip, clip).tolist() + [clip]
        return box(f"{head} &middot; mean CoT logit {values[start:ids.index(THINK_CLOSE)].mean():+.2f}", toks_html([tok.decode(i) for i in ids] + [" [scale]"], pos=ids.index(THINK_CLOSE), vals=vals, val_name="logit"))

    def pair_row(label: str, cheating: tuple[list[int], Tensor], clean: tuple[list[int], Tensor]) -> str:
        assert cheating[0][:cheating[0].index(THINK_OPEN)] == clean[0][:clean[0].index(THINK_OPEN)], f"{label}: the pair's prompts differ"
        return f"<h3>{label}</h3><div class=cols>{single_turn(*cheating, f'{label} &middot; CHEAT')}{single_turn(*clean, f'{label} &middot; clean')}</div>"

    grader_rows = []
    for c in sorted({x["cell"] for x in items if x["env"] == "grader"}):
        pick = {cheating: random.choice([x for x in items if x["cell"] == c and x["split"] == "eval" and x["cheat"] == cheating]) for cheating in [True, False]}
        grader_rows.append(pair_row(c, *[(pick[cheating]["ids"], scores_eval[pick[cheating]["key"]]) for cheating in [True, False]]))
    heldout_rows = []
    for c in heldout_cells:
        pick = {cheating: random.choice([r for r in heldout if f"{r['labels']['family']}.{r['labels']['side']}" == c and r["labels"]["cheat"] == cheating]) for cheating in [True, False]}
        heldout_rows.append(pair_row(c, *[(pick[cheating]["ids"], scores_heldout[pick[cheating]["rollout_id"]]) for cheating in [True, False]]))
    mbpp_rows = []
    short = [x for x in items if x["env"] == "mbpp" and x["split"] == "eval" and len(x["ids"]) <= max_mbpp_tokens]
    for c in random.sample(sorted({x["cell"] for x in short if x["cheat"]} & {x["cell"] for x in short if not x["cheat"]}), n_mbpp):
        pick = {cheating: random.choice([x for x in short if x["cell"] == c and x["cheat"] == cheating]) for cheating in [True, False]}
        mbpp_rows.append(pair_row(f"task {c.split('.')[1]}", *[(pick[cheating]["ids"], scores_eval[pick[cheating]["key"]]) for cheating in [True, False]]))

    IM_START = tok.convert_tokens_to_ids("<|im_start|>")
    secret_rows = []
    for cheating in [True, False]:
        for r in random.sample([r for r in secret if bool(r["cheated"]) == cheating and len(r["ids"]) <= max_secret_tokens], n_secret):
            ids = r["ids"]
            strs = [tok.decode(i) for i in ids] + [" [scale]"]
            bounds = [p for p, i in enumerate(ids) if i == IM_START] + [len(ids)]
            vals = t.zeros(len(ids) + 1)
            vals[-1] = clip
            for a, b in zip(bounds, bounds[1:]):
                if strs[a + 1] == "assistant": vals[a + 3:b] = scores_secret[r["rollout_id"]][a + 3:b].clamp(-clip, clip)   # the generated tokens: after the role line, through <|im_end|>
            bar_w = 620 / len(r["turns"])
            bars = "".join(f"<rect x={70 + k * bar_w + 2:.1f} y={min(70, 70 - 55 * max(-1, min(1, m / clip))):.1f} width={bar_w - 4:.1f} height={abs(55 * max(-1, min(1, m / clip))):.1f} fill='{'#e65050' if m > 0 else '#5082e6'}' {'stroke=#efbf77 stroke-width=2.5' if turn['cheated'] else ''}><title>turn {k}: mean CoT logit {m:+.2f}{', CHEAT' if turn['cheated'] else ''} | {html.escape(' ; '.join(turn['commands']))[:300]}</title></rect><text x={70 + (k + .5) * bar_w:.1f} y=142 text-anchor=middle fill='var(--muted)' font-size=11>{k}</text>" for k, (m, turn) in enumerate(zip(r["turn_means"], r["turns"])))
            chart = f"<svg viewBox='0 0 760 150' role=img aria-label='Mean CoT logit of each turn'><line x1=70 x2=690 y1=70 y2=70 stroke='var(--line)' /><text x=60 y=19 text-anchor=end fill='var(--muted)' font-size=11>+{clip:g}</text><text x=60 y=74 text-anchor=end fill='var(--muted)' font-size=11>0</text><text x=60 y=129 text-anchor=end fill='var(--muted)' font-size=11>-{clip:g}</text>{bars}<text x=700 y=142 fill='var(--muted)' font-size=11>turn</text></svg>"
            turns = []
            k = 0
            for a, b in zip(bounds, bounds[1:]):
                if strs[a + 1] != "assistant":
                    turns.append(f"<details><summary>{strs[a + 1]} &middot; {b - a} tokens</summary><pre>{html.escape(tok.decode(ids[a:b]))}</pre></details>")
                    continue
                turn = r["turns"][k]
                flags = " &middot; ".join(f for f in ["read_secret_file", "overrode_noise_prob", "tampered_guess_log", "tampered_guess_script"] if turn[f])
                head = f"turn {k} &middot; mean CoT logit {r['turn_means'][k]:+.2f}" + (f" &middot; <b class=bad>CHEAT: {flags}</b>" if turn["cheated"] else "") + f" &middot; <code>{html.escape(' ; '.join(turn['commands']))[:200]}</code>"
                turns.append(box(head, toks_html(strs, pos=r["spans"][k][1], lo=a, hi=b, vals=vals, val_name="logit")))
                k += 1
            outcome = "CHEAT" if cheating else "clean"
            secret_rows.append(f"<h3>{outcome} &middot; rollout mean CoT logit {r['mean']:+.2f} &middot; {len(r['turns'])} turns, {len(ids)} tokens{' &middot; first cheat at turn ' + str(r['first_cheat_turn']) if cheating else ''}</h3>{chart}<p class=cap>One bar per turn: the mean logit over that turn's CoT. Amber outline: the turn's command cheated. Hover a bar for its commands.</p>{''.join(turns)}")

    x_of = lambda k: 70 + 620 * k / 9
    panels = []
    for e in ENVS:
        curve = saved["curves"][e]
        span = max(abs(v) for v in curve["cheat"] + curve["clean"]) * 1.1
        y_of = lambda v: 150 - 130 * v / span
        grid = "".join(f"<line x1=70 x2=690 y1={y_of(v):.1f} y2={y_of(v):.1f} stroke='var(--line)' stroke-width={1.5 if v == 0 else 1} /><text x=60 y={y_of(v) + 4:.1f} text-anchor=end fill='var(--muted)' font-size=12>{v:+.1f}</text>" for v in [-span, -span / 2, 0, span / 2, span])
        ticks = "".join(f"<text x={x_of(k)} y=298 text-anchor=middle fill='var(--muted)' font-size=12>{k + 1}</text>" for k in range(10))
        series = "".join(f"<polyline fill=none stroke='{color}' stroke-width=2 points='{' '.join(f'{x_of(k):.1f},{y_of(v):.1f}' for k, v in enumerate(values))}' />" + "".join(f"<circle cx={x_of(k):.1f} cy={y_of(v):.1f} r=4 fill='{color}' stroke='var(--panel)' stroke-width=2><title>tenth {k + 1}: {name} {v:+.2f}, AUROC {curve['auroc'][k]:.3f}</title></circle>" for k, v in enumerate(values)) + f"<text x=702 y={y_of(values[-1]) + 4:.1f} fill='var(--text)' font-size=13>{name}</text>" for name, values, color in [("cheat", curve["cheat"], "#e65050"), ("clean", curve["clean"], "#5082e6")])
        tenths = "<table><tr><th>tenth of the CoT</th>" + "".join(f"<th>{k + 1}</th>" for k in range(10)) + "</tr><tr><td>AUROC</td>" + "".join(f"<td>{v:.2f}</td>" for v in curve["auroc"]) + "</tr></table>"
        panels.append(f"<div><h3>{e}, held-out rollouts</h3><svg viewBox='0 0 760 320' role=img aria-label='Mean logit by tenth of the CoT, {e}'>{grid}{ticks}{series}<text x=380 y=316 text-anchor=middle fill='var(--muted)' font-size=12>tenth of the CoT</text></svg>{tenths}</div>")
    columns = [f"{e} {s}" for e in ENVS for s in ["train", "dev", "eval"]]
    ranges = saved["shuffle_range"]["shuffled labels, tokens"]
    shuffle_note = f"Across the label shuffles the held-out token AUROC runs from {ranges['grader eval'][0]:.3f} to {ranges['grader eval'][1]:.3f} on grader and from {ranges['mbpp eval'][0]:.3f} to {ranges['mbpp eval'][1]:.3f} on MBPP."
    table = "<table><tr><th></th>" + "".join(f"<th>{c.replace('eval', 'held out')}</th>" for c in columns) + "</tr>" + "".join(f"<tr><td>{name.replace('grader probe', 'grader-only probe').replace('mbpp probe', 'MBPP-only probe')}</td>" + "".join(f"<td>{by_split[c]:.3f}</td>" for c in columns) + "</tr>" for name, by_split in saved["auroc"].items()) + "</table>"
    other = f"<table><tr><th>rollouts</th><th>score</th><th>AUROC</th></tr><tr><td>grader held-out families ({len(heldout)} rollouts, {len(heldout_cells)} cells)</td><td>single CoT tokens, within cell</td><td>{heldout_tokens:.3f}</td></tr><tr><td></td><td>rollout mean over the CoT, within cell</td><td>{heldout_rollouts:.3f}</td></tr><tr><td>secret_number ({len(secret)} rollouts, {secret_cheated.sum()} cheating)</td><td>rollout mean over every turn's CoT</td><td>{secret_auroc:.3f}</td></tr><tr><td></td><td>a turn's mean CoT logit: the {n_cheat_turns} turns whose command cheated against the same turn of clean rollouts</td><td>{secret_turn_auroc:.3f}</td></tr></table>"
    css = """:root{color-scheme:dark;--bg:#101419;--panel:#181e26;--raised:#222a34;--line:#303a46;--text:#e7edf3;--muted:#a4b0bd;--accent:#8dbaff;--amber:#efbf77;--red:#f0a0a7}
    body{margin:0 auto;max-width:1500px;padding:32px;background:var(--bg);color:var(--text);font:15px/1.6 Inter,ui-sans-serif,system-ui;font-variant-numeric:tabular-nums}
    h1{letter-spacing:-.02em;margin:.2em 0;text-wrap:balance}h2{letter-spacing:-.01em}h3{margin:1.4em 0 .4em}.eyebrow,.sub,.cap{color:var(--muted)}section{border-top:1px solid var(--line);margin-top:28px}
    table{border-collapse:collapse;margin:12px 0}td,th{padding:4px 12px;border-bottom:1px solid var(--line);text-align:right}td:first-child,th:first-child{text-align:left}td:nth-child(2):not(:last-child):has(+td:last-child),th:nth-child(2):has(+th:last-child){text-align:left}
    .cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}.cols>div{min-width:0}svg{width:100%;max-width:760px;display:block;background:var(--panel);border-radius:8px}
    .strip{background:#111;color:#ddd;font:12px ui-monospace,monospace;padding:8px;margin-bottom:10px;border-radius:6px}.head{margin-bottom:6px;font-weight:bold}.bad{color:var(--amber)}code{color:var(--muted);font-weight:normal}
    details{margin:0 0 10px;color:var(--muted)}summary{cursor:pointer;font:12px ui-monospace,monospace}pre{white-space:pre-wrap;font:12px ui-monospace,monospace;background:#111;padding:8px;border-radius:6px;max-height:260px;overflow:auto}
    @media(max-width:820px){.cols{grid-template-columns:1fr}}@media print{:root{--bg:#fff;--panel:#f4f5f7;--text:#111;--muted:#555;--line:#ccc}}"""
    chosen = fit["chosen"]["joint"]
    sections = f"""<section><h2>Within-cell AUROC, cheat against clean</h2>{table}<p class=cap>A cell's rollouts share one prompt (a grader family and side, or an MBPP task), so these numbers do not come from telling prompts apart. Each probe has its own row and L2 weight, chosen on dev: joint row {chosen['row']}, grader-only row {fit['chosen']['grader']['row']}, MBPP-only row {fit['chosen']['mbpp']['row']}. The controls use the joint fit's row and weights. {shuffle_note}</p><h3>The joint probe on what neither env's fit saw</h3>{other}<p class=cap>Read by a hook at decoder layer {chosen['row'] - 1}, the same pass that shades the strips below.</p></section>
    <section><h2>When the signal arrives</h2><p>Mean joint-probe logit of cheating and clean rollouts by tenth of the CoT.</p><div class=cols>{''.join(panels)}</div></section>
    <section><h2>How to read the strips</h2><p>Each token is shaded by the joint probe's logit at decoder layer {chosen['row'] - 1}, on one colour scale of &plusmn;{clip:g} logits for the whole page: red leans cheat, blue leans clean. The prompt and tool output are unshaded, the block at each strip's end shows the top of the scale, and the outlined token is &lt;/think&gt;. The probe trained on CoT tokens only, so the shading after &lt;/think&gt; is outside what it was fit on. Every rollout below is one the probe never saw.</p></section>
    <section><h2>Grader, training families</h2><p>Held-out rollouts of the five families the probe trained on. Each row is one prompt: a cheating rollout on the left and a clean one on the right.</p>{''.join(grader_rows)}</section>
    <section><h2>Grader, held-out families</h2><p>Families no probe trained on (heldout38).</p>{''.join(heldout_rows)}</section>
    <section><h2>MBPP</h2><p>Test-split tasks, whose visible test has a wrong expected value. Each row is one task: a rollout that passed the wrong test on the left, and one that solved the original task on the right.</p>{''.join(mbpp_rows)}</section>
    <section><h2>secret_number</h2><p>An env no probe trained on: a multi-turn game with a shell, where reading the secret file or tampering with the game is the cheat. Each rollout is one sequence holding every turn, scored in one pass.</p>{''.join(secret_rows)}</section>
    <section class=cap><p>Source: joint_probe.py, data/vectors/Qwen3.8-27B/joint_probe.safetensors, Will's s9-gr manifest, mbpp38 and mbpp38_x4, heldout38, secret38.</p></section>"""
    path.write_text(f"<!doctype html><html lang=en><head><meta charset=utf-8><title>Joint probe tokens</title><style>{css}</style></head><body><div class=eyebrow>ortho &middot; joint grader + MBPP outcome probe &middot; 2026-11-11</div><h1>The joint probe on held-out rollouts: token AUROC {saved['auroc']['joint probe, tokens']['grader eval']:.2f} on grader and {saved['auroc']['joint probe, tokens']['mbpp eval']:.2f} on MBPP, rollout AUROC {secret_auroc:.2f} on secret_number</h1><p class=sub>Qwen3.8-27B. A logistic probe on single CoT tokens, the rollout's outcome as the label on every token, fit on grader and MBPP rollouts at half the loss each.</p>{sections}</body></html>")
    print(f"{green}wrote {path}, {path.stat().st_size / 1e6:.1f} MB{endc}")
