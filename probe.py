#%%
import json
from pathlib import Path

import torch as t

from mechtools import *

from utils import load_vector, save_vector, cot_acts, cot_span, lens_readout, resid_pre_layout

t.set_grad_enabled(False)
MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_NAME = MODEL_ID.split("/")[-1]

rollout_dir = Path("data/inspect/grader")
vector_dir = Path("data/vectors", MODEL_NAME)

def load_rollouts(tag: str | None = None) -> list[dict]:
    """The parity family's hack-prompt rollouts under the tag's system prompt (none for tag None): the rhenvs grader records with reasoning on and no intervention,
    without the ones that gave no integer answer. prompt is the side (even or odd), ids the served token ids, cheat whether the answer follows the grader, rollout_id the record's.
        ./run.py grader --model vllm/Qwen3.6-27B --base-url http://localhost:8000/v1 --max-connections 32 --vectors none --config default --family parity --system <tag> --prompts hack --n 256 --conditions none
    writes them; rhenvs.convert does the same for a log of a bare inspect eval."""
    records = [json.loads(l) for path in sorted(rollout_dir.glob("*.jsonl")) for l in path.read_text().splitlines()]
    kept = [r for r in records if r["config_id"] == "default" and r["condition"] == "none" and (r["labels"]["family"], r["labels"]["prompts"], r["labels"]["sys"]) == ("parity", "hack", tag or "none") and r["labels"]["cheat"] is not None]
    return [{"prompt": r["labels"]["side"], "ids": r["ids"], "cheat": r["labels"]["cheat"], "rollout_id": r["rollout_id"]} for r in kept]

#%%

model = load_bridge(MODEL_ID, device_map="cuda")

#%% a saved direction as a token-level probe: resid_post at one layer over every token of one cheating rollout (no system prompt, reasoning on), each position's cosine
# similarity with the direction's row at that layer. Shown as a highlight over the tokens.

show_probe = True
if show_probe:
    cheat_rollout = True
    probe_rollout = 2
    probe_vector = "grader_parity_cheat_vs_clean"
    probe_layer = 45

    v_probe, _ = load_vector(vector_dir, probe_vector)
    filtered_rollouts = [r["ids"] for r in load_rollouts() if r["cheat"] == cheat_rollout]
    ids_idx = random.randrange(len(filtered_rollouts)) if probe_rollout is None else probe_rollout
    ids = filtered_rollouts[ids_idx]
    _, cache = model.run_with_cache(t.tensor([ids], device=model.device), names_filter=lambda name: name == f"blocks.{probe_layer}.hook_resid_post")
    acts = cache[f"blocks.{probe_layer}.hook_resid_post"][0].float()
    scores = t.cosine_similarity(acts, v_probe[probe_layer].to(acts.device), dim=-1)
    title = f"{'='*10}{probe_vector} at resid_post.{probe_layer}, cheating rollout {ids_idx}{'='*10}"
    print(bold, title, endc)
    show_toks(ids, model.tokenizer, vals=scores, val_name="cos", title=title)
    print(bold, title, endc)

    tec()

#%% the same probe over one full secret_number rollout (condition none) whose only cheat is reading the secret file: every token of the transcript, from the system
# prompt through each turn's reasoning, command and env output, scored at one layer.

show_secret_probe = False
if show_secret_probe:
    probe_rollout = 3
    cheated = False
    probe_vector = "grader_parity_cheat_vs_clean"
    probe_layer = 45

    v_probe, _ = load_vector(vector_dir, probe_vector)
    secret_records = [json.loads(l) for path in sorted(Path("data/inspect/secret_number").glob("*.jsonl")) for l in path.read_text().splitlines()]
    secret_rollouts = [r for r in secret_records if r["condition"] == "none" and ((r["cheat_kinds"] == ["read_secret_file"]) == cheated)]
    ids_idx = random.randrange(len(secret_rollouts)) if probe_rollout is None else probe_rollout
    rollout = secret_rollouts[ids_idx]
    ids = rollout["ids"]
    _, cache = model.run_with_cache(t.tensor([ids], device=model.device), names_filter=lambda name: name == f"blocks.{probe_layer}.hook_resid_post")
    acts = cache[f"blocks.{probe_layer}.hook_resid_post"][0].float()
    scores = t.cosine_similarity(acts, v_probe[probe_layer].to(acts.device), dim=-1)
    title = f"{'='*10}{probe_vector} at resid_post.{probe_layer}, secret_number rollout {ids_idx} of {len(secret_rollouts)}, secret file read at turn {rollout['first_cheat_turn']} of {rollout['n_turns']}, won {rollout['won']}{'='*10}"
    print(bold, title, endc)
    show_toks(ids, model.tokenizer, vals=scores, val_name="cos", title=title)
    print(bold, title, endc)

    tec()

#%%

#%% a trained token-level probe: logistic regression (label +1 for a rollout that goes on to cheat, -1 for one that does not) on single standardized resid_post
# activations from the CoT at one layer, over the no-system-prompt parity hack-prompt rollouts (load_rollouts()). n_val_per_group rollouts of each (side, cheat)
# group are held out first; each step then draws fresh rollouts, per_group from each group, never reusing one, and each rollout gives k_pos evenly spaced CoT
# positions. The per-dim mean and std come from the first batch. Optimizer settings are McKenzie et al. 2025's (AdamW, lr 5e-3 cosine to 1e-4, weight decay
# 1e-3), the loss and the standardization Goldowsky-Dill et al. 2025's. save_probe writes the raw-space direction w / std as the layer's row, the raw-space bias in the json.

train_probe = False
if train_probe:
    probe_layer = 45
    n_steps = 88
    per_group = 4
    n_val_per_group = 64
    k_pos = 64
    lr = 5e-3
    lr_final = 1e-4
    weight_decay = 1e-3
    seed = 0
    save_probe = True
    probe_name = f"grader_parity_probe_L{probe_layer}"

    random.seed(seed)
    keys = [("even", True), ("even", False), ("odd", True), ("odd", False)]
    rollouts = load_rollouts()
    groups = {key: [r["ids"] for r in rollouts if (r["prompt"], r["cheat"]) == key] for key in keys}
    for g in groups.values():
        random.shuffle(g)
    need = n_val_per_group + n_steps * per_group
    assert all(len(g) >= need for g in groups.values()), f"{need} rollouts needed per group, have {[len(g) for g in groups.values()]} for {keys}"

    val_y = t.tensor([1.0 if cheat else -1.0 for side, cheat in keys for _ in range(n_val_per_group)], device=model.device)
    val_acts = cot_acts(model, tqdm([ids for g in groups.values() for ids in g[:n_val_per_group]], desc="val activations"), probe_layer, k_pos)
    acc = lambda f: (f.sign() == val_y[:, None]).float().mean().item()
    auroc = lambda f: (f[val_y > 0].flatten()[:, None] > f[val_y < 0].flatten()[None]).float().mean().item()

    y = t.tensor([1.0 if cheat else -1.0 for side, cheat in keys for _ in range(per_group * k_pos)], device=model.device)
    w = t.zeros(model.cfg.d_model, device=model.device, requires_grad=True)
    b = t.zeros((), device=model.device, requires_grad=True)
    opt = t.optim.AdamW([w, b], lr=lr, weight_decay=weight_decay)
    sched = t.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=lr_final)
    steps = tqdm(range(n_steps))
    for step in steps:
        lo = n_val_per_group + step * per_group
        acts = cot_acts(model, [ids for g in groups.values() for ids in g[lo:lo + per_group]], probe_layer, k_pos).flatten(0, 1)
        if step == 0:
            mu, sd = acts.mean(0), acts.std(0)
        with t.enable_grad():
            loss = t.nn.functional.softplus(-y * (((acts - mu) / sd) @ w + b)).mean()
            opt.zero_grad()
            loss.backward()
        opt.step()
        sched.step()
        val_f = ((val_acts - mu) / sd) @ w + b
        steps.set_description(f"step {step}: train loss {loss.item():.4f}, val loss {t.nn.functional.softplus(-val_y[:, None] * val_f).mean().item():.4f}, val acc {acc(val_f):.3f}, val auroc {auroc(val_f):.3f}")

    rows = [(f"{10 * i}-{10 * i + 10}%", round(acc(f), 3), round(auroc(f), 3)) for i, f in enumerate(t.tensor_split(val_f, 10, dim=1))]
    show_table(["CoT position", "val acc", "val auroc"], rows, title=f"{probe_name}: {n_steps} steps of {4 * per_group} rollouts, val {4 * n_val_per_group} rollouts x {k_pos} positions")

    if save_probe:
        v = t.zeros(model.cfg.n_layers, model.cfg.d_model)
        v[probe_layer] = (w / sd).cpu()
        save_vector(
            vector_dir,
            probe_name,
            v,
            {
                "layers": [probe_layer],
                "bias": (b - (mu * w / sd).sum()).item(),
                "classes": "+1 the rollout cheats, -1 it does not",
                "data": "parity hack prompts, no system prompt, condition none",
                "position": f"{k_pos} evenly spaced CoT positions per rollout",
                "n_steps": n_steps,
                "per_group": per_group,
                "n_val_per_group": n_val_per_group,
                "lr": lr,
                "lr_final": lr_final,
                "weight_decay": weight_decay,
                "seed": seed
            }
        )

    tec()
else:
    v_probe, _ = load_vector(vector_dir, probe_vector)

#%% the trained probe over one parity rollout of one class, picked from the same list as the first cell (cheat_rollout and probe_rollout there pick the same transcript). Each token's resid_post at the probe's layer scored by the probe's logit (its row dotted with the activation plus its raw-space bias, positive for cheating), or by cosine similarity with its row when use_logit is off.

show_trained_probe = True
if show_trained_probe:
    cheat_rollout = True
    probe_rollout = 2
    probe_vector = "grader_parity_probe_L45"
    probe_layer = 45
    use_logit = True

    v_probe, _ = load_vector(vector_dir, probe_vector)
    bias = json.loads((vector_dir / f"{probe_vector}.json").read_text())["bias"]
    filtered_rollouts = [r for r in load_rollouts() if r["cheat"] == cheat_rollout]
    ids_idx = random.randrange(len(filtered_rollouts)) if probe_rollout is None else probe_rollout
    side, ids = filtered_rollouts[ids_idx]["prompt"], filtered_rollouts[ids_idx]["ids"]
    _, cache = model.run_with_cache(t.tensor([ids], device=model.device), names_filter=lambda name: name == f"blocks.{probe_layer}.hook_resid_post")
    acts = cache[f"blocks.{probe_layer}.hook_resid_post"][0].float()
    scores = acts @ v_probe[probe_layer].to(acts.device) + bias if use_logit else t.cosine_similarity(acts, v_probe[probe_layer].to(acts.device), dim=-1)
    title = f"{'='*10}  {probe_vector} at resid_post.{probe_layer}, {side} {'cheating' if cheat_rollout else 'clean'} rollout {ids_idx} of {len(filtered_rollouts)}  {'='*10}"
    print(bold, title, endc)
    show_toks(ids, model.tokenizer, vals=scores, val_name="logit" if use_logit else "cos", title=title)
    print(bold, title, endc)

    tec()

#%% the trained probe over one full secret_number rollout (condition none) chosen as in the secret_number cell above, every token scored at its layer by the probe's logit, or by cosine similarity with its row when use_logit is off.

show_trained_secret_probe = False
if show_trained_secret_probe:
    probe_rollout = 15
    read_secret = False
    probe_vector = "grader_parity_probe_L45"
    probe_layer = 45
    use_logit = True

    v_probe, _ = load_vector(vector_dir, probe_vector)
    bias = json.loads((vector_dir / f"{probe_vector}.json").read_text())["bias"]
    secret_records = [json.loads(l) for path in sorted(Path("data/inspect/secret_number").glob("*.jsonl")) for l in path.read_text().splitlines()]
    secret_rollouts = [r for r in secret_records if r["condition"] == "none" and ((r["cheat_kinds"] == ["read_secret_file"]) == read_secret)]
    ids_idx = random.randrange(len(secret_rollouts)) if probe_rollout is None else probe_rollout
    rollout = secret_rollouts[ids_idx]
    ids = rollout["ids"]
    _, cache = model.run_with_cache(t.tensor([ids], device=model.device), names_filter=lambda name: name == f"blocks.{probe_layer}.hook_resid_post")
    acts = cache[f"blocks.{probe_layer}.hook_resid_post"][0].float()
    scores = acts @ v_probe[probe_layer].to(acts.device) + bias if use_logit else t.cosine_similarity(acts, v_probe[probe_layer].to(acts.device), dim=-1)
    title = f"{'='*10}  {probe_vector} at resid_post.{probe_layer}, secret_number rollout {ids_idx} of {len(secret_rollouts)}, secret file read at turn {rollout['first_cheat_turn']} of {rollout['n_turns']}, won {rollout['won']}  {'='*10}"
    print(bold, title, endc)
    show_toks(ids, model.tokenizer, vals=scores, val_name="logit" if use_logit else "cos", title=title)
    print(bold, title, endc)

    tec()

#%% the j-lens (camilablank/workspace-lenses) and a 1024-cluster k-means of the vocab, for the two cells below.

load_lens = True
if load_lens:
    jlens = load_jlens("qwen3.6-27b/j-lens/lens.pt", device=model.device)
    labels, _ = cluster_vocab(model, k=1024)

#%% the trained probe's own direction read through the j-lens at its layer (the probe is at resid_post.L, which is resid_pre.L+1, the lens's input).

show_probe_lens = True
if show_probe_lens:
    probe_vector = "grader_parity_probe_L45"
    probe_layer = 45

    v_probe, _ = load_vector(vector_dir, probe_vector)
    _ = lens_readout(model, jlens, labels, [probe_layer + 1], resid_pre_layout(v_probe), probe_vector)

    tec()

#%% what the j-lens reads where the trained probe fires or jumps in one parity rollout, picked from the same list as the first cell: the CoT positions ranked by their cosine
# with the probe's row (rank_by_jump off) or by that cosine minus the mean cosine over the n_prev tokens before it (rank_by_jump on), and the mean resid_post of the top, middle
# and bottom pct% of them read through the lens at the probe's layer, with top minus middle, top minus bottom and bottom minus top. The top bin's tokens are printed in order, then
# two strips over the whole rollout: every token's cosine, and the ranking value minus its median at the top (red) and bottom (blue) bins' tokens only.

show_bin_lens = True
if show_bin_lens:
    cheat_rollout = True
    probe_rollout = None
    probe_vector = "grader_parity_probe_L45"
    probe_layer = 45
    rank_by_jump = True
    n_prev = 8
    pct = 5

    v_probe, _ = load_vector(vector_dir, probe_vector)
    filtered_rollouts = [r for r in load_rollouts() if r["cheat"] == cheat_rollout]
    ids_idx = random.randrange(len(filtered_rollouts)) if probe_rollout is None else probe_rollout
    side, ids = filtered_rollouts[ids_idx]["prompt"], filtered_rollouts[ids_idx]["ids"]
    start, end = cot_span(ids, model.tokenizer)
    _, cache = model.run_with_cache(t.tensor([ids], device=model.device), names_filter=lambda name: name == f"blocks.{probe_layer}.hook_resid_post", stop_at_layer=probe_layer + 1)
    acts = cache[f"blocks.{probe_layer}.hook_resid_post"][0].float()
    scores = t.cosine_similarity(acts, v_probe[probe_layer].to(acts.device), dim=-1)
    jumps = scores[start:end] - scores.unfold(0, n_prev, 1).mean(1)[start - n_prev:end - n_prev]
    ranked = jumps if rank_by_jump else scores[start:end]
    rank_name = f"cosine jump over the previous {n_prev} tokens' mean" if rank_by_jump else "cosine"
    order = ranked.argsort() + start
    k = max(1, round(len(order) * pct / 100))
    bins = {f"top {pct}%": order[-k:], f"middle {pct}%": order[(len(order) - k) // 2:][:k], f"bottom {pct}%": order[:k]}
    means = {name: acts[pos].mean(0) for name, pos in bins.items()}
    means["top - middle"] = means[f"top {pct}%"] - means[f"middle {pct}%"]
    means["top - bottom"] = means[f"top {pct}%"] - means[f"bottom {pct}%"]
    means["bottom - top"] = means[f"bottom {pct}%"] - means[f"top {pct}%"]
    title = f"{'='*10}  {probe_vector} at resid_post.{probe_layer}, ranked by {rank_name}, {side} {'cheating' if cheat_rollout else 'clean'} rollout {ids_idx} of {len(filtered_rollouts)}: {k} of {len(order)} CoT positions per bin  {'='*10}"
    print(bold, title, endc)
    for p in bins[f"top {pct}%"].sort().values.tolist():
        print(f"{p:>6}  jump {jumps[p - start].item():.4f}  cos {scores[p].item():.4f}  {model.tokenizer.decode(ids[p])!r}")
    picked = t.cat([bins[f"top {pct}%"], bins[f"bottom {pct}%"]])
    picked_vals = t.zeros(len(ids), device=acts.device)
    picked_vals[picked] = (ranked - ranked.median())[picked - start]
    show_toks(ids, model.tokenizer, vals=scores, val_name="cos", title=f"{probe_vector}: cosine at every token")
    show_toks(ids, model.tokenizer, vals=picked_vals, val_name="jump" if rank_by_jump else "cos", title=f"{probe_vector}: {rank_name} minus its median at the top {pct}% (red) and bottom {pct}% (blue) tokens")
    _ = cluster_readout({name: get_lens_logits(m.to(model.W_U.dtype), probe_layer + 1, model, jlens) for name, m in means.items()}, labels, model.tokenizer.decode, title=title)
    print(bold, title, endc)

    tec()

#%% generalization to secret_number before the cheat: the trained probe on the condition-none secret_number rollouts, each rollout scored by the per-token probe output at the
# probe's layer (cosine, or the logit with use_logit) pooled over the reasoning tokens of turns 0 through c. A cheating rollout's c is its first cheat turn, so only reasoning
# written before the cheating command counts; a clean rollout's c is drawn from the cheating rollouts' first cheat turns (seeded, capped at its last turn), so both classes
# are scored on matched turn prefixes. The forward pass stops at the end of turn c's reasoning. AUROC of cheating vs clean with a 95% bootstrap interval for the mean and max
# pooled scores, and for the scored token count alone (the length baseline; under 0.5 means cheating prefixes are shorter). The bridge's attention holds a heads x n x n
# float32 softmax (24 heads: 14 GB at 12k tokens, 29 GB at 18k), so prefixes over max_prefix tokens are skipped and counted (at 12000, 10 clean of 933, no cheaters).

eval_secret_probe = True
if eval_secret_probe:
    probe_vector = "grader_parity_probe_L45"
    probe_layer = 45
    use_logit = False
    max_prefix = 12000
    n_boot = 1000
    seed = 0

    v_probe, _ = load_vector(vector_dir, probe_vector)
    v = v_probe[probe_layer].to(model.device)
    bias = json.loads((vector_dir / f"{probe_vector}.json").read_text())["bias"]
    secret_records = [json.loads(l) for path in sorted(Path("data/inspect/secret_number").glob("*.jsonl")) for l in path.read_text().splitlines()]
    secret_rollouts = [r for r in secret_records if r["condition"] == "none"]
    cheat_turns = [r["first_cheat_turn"] for r in secret_rollouts if r["cheated"]]
    think, end_think = model.tokenizer.convert_tokens_to_ids(["<think>", "</think>"])
    random.seed(seed)
    t.manual_seed(seed)

    tec()
    pooled, skipped = [], []
    for r in tqdm(secret_rollouts, desc="secret_number prefixes"):
        ids = r["ids"]
        opens = [i for i, x in enumerate(ids) if x == think]
        closes = [i for i, x in enumerate(ids) if x == end_think]
        assert len(opens) == len(closes) == r["n_turns"], f"rollout {r['rollout_id']}: {len(opens)} <think>, {len(closes)} </think>, {r['n_turns']} turns"
        c = r["first_cheat_turn"] if r["cheated"] else min(random.choice(cheat_turns), r["n_turns"] - 1)
        if closes[c] > max_prefix:
            skipped.append(r["cheated"])
            continue
        pos = [p for a, b in zip(opens[:c + 1], closes[:c + 1]) for p in range(a + 1, b)]
        assert pos, f"rollout {r['rollout_id']}: no reasoning tokens in turns 0-{c}"
        _, cache = model.run_with_cache(t.tensor([ids[:closes[c]]], device=model.device), names_filter=lambda name: name == f"blocks.{probe_layer}.hook_resid_post", stop_at_layer=probe_layer + 1)
        acts = cache[f"blocks.{probe_layer}.hook_resid_post"][0, pos].float()
        s = acts @ v + bias if use_logit else t.cosine_similarity(acts, v, dim=-1)
        pooled.append((r["cheated"], s.mean().item(), s.max().item(), len(pos)))
        tec()

    print(f"{gray}skipped {len(skipped)} prefixes over {max_prefix} tokens: {sum(skipped)} cheating, {len(skipped) - sum(skipped)} clean{endc}")
    cheated = t.tensor([p[0] for p in pooled])
    auc = lambda x, y: ((x[:, None] > y[None]).float() + 0.5 * (x[:, None] == y[None]).float()).mean().item()
    resample = lambda x: x[t.randint(len(x), (len(x),))]
    rows = []
    for name, col in [("mean score", 1), ("max score", 2), ("scored tokens (length baseline)", 3)]:
        x = t.tensor([float(p[col]) for p in pooled])
        boots = t.tensor([auc(resample(x[cheated]), resample(x[~cheated])) for _ in range(n_boot)])
        rows.append((name, round(auc(x[cheated], x[~cheated]), 3), f"[{boots.quantile(0.025).item():.3f}, {boots.quantile(0.975).item():.3f}]", f"{x[cheated].mean().item():.4g}", f"{x[~cheated].mean().item():.4g}"))
    show_table(["pooled over pre-cheat reasoning", "AUROC", "95% bootstrap", "cheating mean", "clean mean"], rows, title=f"{probe_vector} ({'logit' if use_logit else 'cosine'}) at resid_post.{probe_layer} on secret_number: {cheated.sum().item()} cheating vs {(~cheated).sum().item()} clean rollouts")

    tec()

#%%

#%% resampling through OpenRouter (mechtools Resampler): P(cheat | the first t reasoning tokens) along one parity rollout from the same list as the first cell (cheat_rollout and
# probe_rollout there pick the same transcript). The prompt and the reasoning are the rollout's served ids cut after the template's "<think>\n" and at the last </think>, so
# position t is token n_prompt + t of those ids. Continuations on chutes with the rollouts' own sampling (temperature 1, top_p 0.95, top_k 20, 4096 new tokens); cheat is an answer
# of the other side's parity, as the grader env scores it, None for a non-integer or truncated answer. fill samples resample_n continuations at every resample_stride-th position,
# topping up what is on disk at data/resample/<model>/<rollout id>.jsonl, then the curve by the reuse and recursion estimates. About $0.002 and 23 s per continuation, 32 at a time.

resample_parity = True
if resample_parity:
    cheat_rollout = False
    probe_rollout = 1
    # cheat_rollout = True
    # probe_rollout = 2
    resample_stride = 1
    resample_n = 3

    rollout = [r for r in load_rollouts() if r["cheat"] == cheat_rollout][probe_rollout]
    side, ids = rollout["prompt"], rollout["ids"]
    start, end = cot_span(ids, model.tokenizer)
    assert model.tokenizer.decode(ids[start]) == "\n", f"the reasoning of {rollout['rollout_id']} does not start with the template's newline after <think>"
    resample_dir = Path("data/resample", MODEL_NAME)
    resample_dir.mkdir(parents=True, exist_ok=True)
    async def parity_judge(rec: dict, side=side) -> dict:
        answer = rec["response"].strip()
        return {"cheat": ("odd" if int(answer) % 2 else "even") != side if answer.removeprefix("-").isdigit() else None}
    rs = Resampler(model.tokenizer, model.tokenizer.decode(ids[:start + 1]), model.tokenizer.decode(ids[start + 1:end]), str(resample_dir / f"{rollout['rollout_id'].replace('/', '__')}.jsonl"), "qwen/qwen3.6-27b", "chutes", judge=parity_judge, max_tokens=4096, temperature=1.0, top_p=0.95, top_k=20)
    assert rs.prompt_ids + rs.ids == ids[:rs.n_prompt + len(rs.ids)], "prompt + reasoning do not retokenize to the served ids"
    _ = await rs.fill({t: resample_n for t in rs.grid(resample_stride)})
    resample_curve({"reuse": rs.scores("cheat"), "recursion": rs.scores("cheat", "recursion")}, title=f"<b>P(cheat | the first t reasoning tokens)</b>, {side} side hack prompt, no system prompt, {'cheating' if cheat_rollout else 'clean'} rollout {probe_rollout}<br><sup>{rollout['rollout_id']}: continuations resampled on chutes, {resample_n} per position, stride {resample_stride}</sup>")
