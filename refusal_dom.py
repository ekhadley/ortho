#%%
"""heretic's refusal direction on Qwen3.8-27B, applied as our projection hook in place of its weight edit. The fit is heretic's (github.com/p-e-w/heretic, main.py):
its 400 harmful and 400 harmless train prompts (rhenvs configs/refusal/prompts.json), each rendered under its system prompt with reasoning off, one forward pass, the
residual at the last prompt token (the blank line after </think>, where the answer starts) at every decoder layer's output; the direction is the harmful mean minus the
harmless mean, one unit row per layer. heretic then removes each row's component along the harmless mean and searches a per-layer weight edit. Here the rows go into
scripts/vllm_proj.py's hook, h <- h - ((h - mu).u)u at every position, pinned at the harmless mean mu, which leaves a residual sitting at that mean where it is.
Saved in train_vector.py's layout (direction_unit and mean_resid, 65 rows, row i + 1 for decoder layer i, unit rows pointing toward the harmful prompts):
refusal_dom (decoder layers 30, 33, ..., 60), refusal_dom_all (every decoder layer), and refusal_dom_rand and refusal_dom_all_rand, random unit rows at the same
layers and pins. scripts/proj_export.py turns each into a hook file; run.py's refdom, refdomall, refdomrand and refdomallrand rows serve them.
    python refusal_dom.py fit       box, vLLM stopped: the four vectors and refusal_dom.json: norms, the test prompts' AUROC along the rows, and under each hook the
                                    first-token KL from the base model on heretic's test prompts (its damage measure, on the harmless ones) and MMLU, 0-shot
    python refusal_dom.py resp      box, vLLM stopped, after run.py's refusal38_train, refdom_harmful_train, refusal38 and refdom_harmful: the direction from the
                                    answers instead of the prompt. Base refusals and refdom's compliant answers to the same harmful train prompt (reasoning off,
                                    refusal_traces.py's first-person label), read through the base model; per prompt the mean refusing minus the mean compliant
                                    residual, averaged over prompts, pooled over the first FIRST answer tokens (refusal_resp32_all) and over every answer token
                                    (refusal_resp_all), unit rows at every decoder layer pinned at the compliant mean, with random rows at the same pins
                                    (_rand); refusal_resp.json holds the same checks as fit's, the test AUROC from refusal38 and refdom_harmful's answers and
                                    the rows' cosine with refusal_dom. run.py's refresp32, refresp, refresp32rand and refresprand rows serve them
    python refusal_dom.py screen    box, vLLM stopped, after the refusal runs: teacher-forced log-prob gains of the records' completions under each hook (the hooked
                                    model's log-prob of a completion minus the base model's), acts/refusal/screen.json. Refusing and compliant answers to the same
                                    harmful test prompt are a pair, and the pair's margin is the compliant gain minus the refusing gain, as in a DPO reward"""
import json
import random
import sys
from pathlib import Path

import torch as t
from datasets import load_dataset
from rhenvs.common import CONFIGS, load_config
from rhenvs.refusal import ANSWER_CHARS, MARKERS
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText, AutoTokenizer

from mechtools import *

STAGE = sys.argv[1]
MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
STEP = list(range(30, 61, 3))
ALL = list(range(64))
NAME = "refusal_dom"
HOOKS = {NAME: "refdom", f"{NAME}_all": "refdomall", f"{NAME}_rand": "refdomrand", f"{NAME}_all_rand": "refdomallrand"}   # vector name to served row
SEED = 0
OUT = Path("data/vectors/Qwen3.8-27B")
ACTS = Path("acts/refusal")
PROMPTS = json.loads((CONFIGS / "refusal" / "prompts.json").read_text())   # heretic's prompts, arm -> split -> list
THINK_OPEN, THINK_CLOSE, NEWLINE, BLANK = 248068, 248069, 198, 271
MMLU_EVERY = 4   # fit: every fourth MMLU test question
MMLU_BATCH = 32
SOURCES = ["base", "refdom", "refdomall"]   # screen: the served rows whose rollouts are scored
PER_CLASS = 2   # screen and resp: rollouts per config, prompt, source and class
FIRST = 32   # resp: the answer tokens of the first pooling
RESP = {f"refusal_resp{FIRST}_all": f"refresp{FIRST}", "refusal_resp_all": "refresp", f"refusal_resp{FIRST}_all_rand": f"refresp{FIRST}rand", "refusal_resp_all_rand": "refresprand"}   # resp: vector name to served row
EOS = 248046   # <|im_end|>
TOPICAL = ["disclaimer", "violat", "prohibit", "illegal", "harmful", "inappropriate", "unethical", "ethical boundaries"]   # resp: refusal_traces.py's label, heretic's markers that a compliant answer's disclaimer also holds
PHRASES = ["not going to", "i'm not able", "i am not able"]   # and the refusal wordings heretic's list has no marker for

#%% model: every decoder layer's output kept, then the hook in force applied to it at every position
set_seed(SEED)
tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, padding_side="right")
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=REVISION, dtype=t.bfloat16, device_map="cuda").eval()
layers = model.model.language_model.layers
assert len(layers) == 64
outputs = [None] * 64
unit, mu, window = None, None, []   # the hook in force: unit rows and pins [65, d_model] and the decoder layers hooked; none for the base model
loaded = {}

def use(name: str | None) -> None:
    global unit, mu, window
    if name is not None and name not in loaded:
        saved = load_file(OUT / f"{name}.safetensors")
        loaded[name] = saved["direction_unit"].float().cuda(), saved["mean_resid"].float().cuda(), [i for i in range(64) if saved["direction_unit"][i + 1].norm() > 0]
    unit, mu, window = loaded[name] if name is not None else (None, None, [])

def at_layer(i: int):
    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        outputs[i] = h
        if i not in window: return output
        h = h - (((h.float() - mu[i + 1]) @ unit[i + 1])[..., None] * unit[i + 1]).to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    return hook
for i, layer in enumerate(layers): layer.register_forward_hook(at_layer(i))

system = load_config("refusal", "nothink")["system_prompt"]
def render(prompt: str) -> list[int]:
    """A prompt under heretic's system prompt with reasoning off, ending at the blank line after </think>."""
    text = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
    ids = tok(text, add_special_tokens=False).input_ids
    assert ids[-4:] == [THINK_OPEN, BLANK, THINK_CLOSE, BLANK], tok.decode(ids[-6:])
    return ids

#%% fit: the last prompt token's residual of heretic's train prompts, their difference of means, the four vectors
if STAGE == "fit":
    def last_resid(prompts: list[str], desc: str) -> t.Tensor:
        """The base model's residual at each prompt's last token, at every decoder layer's output: [n, 64, d_model]."""
        hs = []
        with t.no_grad():
            for p in pbar(prompts, desc=desc):
                model(input_ids=t.tensor([render(p)], device="cuda"))
                hs.append(t.stack([o[0, -1].float() for o in outputs]).cpu())
        return t.stack(hs)

    train_bad, train_good = last_resid(PROMPTS["harmful"]["train"], "harmful train"), last_resid(PROMPTS["harmless"]["train"], "harmless train")
    test_bad, test_good = last_resid(PROMPTS["harmful"]["test"], "harmful test"), last_resid(PROMPTS["harmless"]["test"], "harmless test")
    mean_bad, mean_good = train_bad.mean(0), train_good.mean(0)
    diff = mean_bad - mean_good   # [64, d_model]
    d_model = diff.shape[1]
    dom = t.nn.functional.normalize(diff, dim=1)
    rand = t.nn.functional.normalize(t.randn(64, d_model, generator=t.Generator().manual_seed(SEED)), dim=1)
    mean_resid = t.zeros(65, d_model)
    mean_resid[1:] = mean_good
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows, hooked in [(NAME, dom, STEP), (f"{NAME}_all", dom, ALL), (f"{NAME}_rand", rand, STEP), (f"{NAME}_all_rand", rand, ALL)]:
        u = t.zeros(65, d_model)
        u[[L + 1 for L in hooked]] = rows[hooked]
        save_file({"direction_unit": u.contiguous(), "mean_resid": mean_resid.contiguous()}, OUT / f"{name}.safetensors")
    print(f"{cyan}|harmful mean - harmless mean| / |harmless mean|: " + ", ".join(f"L{L} {diff[L].norm():.1f} / {mean_good[L].norm():.0f}" for L in ALL[::6]) + f"{endc}")

    # along each layer's row: the two train classes in units of the harmless spread, heretic's orthogonalisation (the row's cosine with the harmless mean), and the held-out prompts' AUROC
    def along(h: t.Tensor, rows: t.Tensor) -> t.Tensor: return (h * rows).sum(-1)   # [n, 64]
    def auroc(rows: t.Tensor) -> list[float]: return (along(test_bad, rows)[:, None] > along(test_good, rows)[None]).float().mean((0, 1)).tolist()
    gap_sd = ((along(train_bad, dom).mean(0) - along(train_good, dom).mean(0)) / along(train_good, dom).std(0)).tolist()
    cos_good = t.cosine_similarity(dom, mean_good, dim=1).tolist()
    stats = dict(diff_norm=diff.norm(dim=1).tolist(), harmless_norm=mean_good.norm(dim=1).tolist(), pin=along(mean_good, dom).tolist(), gap_over_harmless_sd=gap_sd, cos_with_harmless_mean=cos_good, cos_next_layer=t.cosine_similarity(dom[:-1], dom[1:], dim=1).tolist(), test_auroc=auroc(dom), test_auroc_random=auroc(rand))
    print(f"{cyan}gap in harmless SDs: " + ", ".join(f"L{L} {gap_sd[L]:.1f}" for L in ALL[::6]) + f"; cosine with the harmless mean: " + ", ".join(f"L{L} {cos_good[L]:+.2f}" for L in ALL[::6]) + f"{endc}")
    print(f"{cyan}test AUROC along the row: " + ", ".join(f"L{L} {stats['test_auroc'][L]:.3f}" for L in ALL[::6]) + f"; random rows: " + ", ".join(f"L{L} {stats['test_auroc_random'][L]:.3f}" for L in ALL[::6]) + f"{endc}")
    checked = HOOKS
    meta = dict(model=MODEL_ID, revision=REVISION, prompts="rhenvs configs/refusal/prompts.json: heretic's mlabonne/harmful_behaviors and mlabonne/harmless_alpaca, train[:400] for the fit and test[:100] for the checks", system_prompt=system, position="last prompt token with reasoning off, the blank line after </think>", direction="harmful mean minus harmless mean, unit row per decoder layer, no orthogonalisation", pinned_at="harmless train mean at the same token", layers={NAME: STEP, f"{NAME}_all": ALL, f"{NAME}_rand": STEP, f"{NAME}_all_rand": ALL}, random_seed=SEED, **stats)
    meta_path = OUT / f"{NAME}.json"

#%% resp: the answer tokens of the base model's refusals and refdom's compliant answers to the same harmful prompts (reasoning off), their per-prompt difference of means, the four vectors
if STAGE == "resp":
    records = [json.loads(line) for path in sorted(Path("data/inspect/refusal").glob("*.jsonl")) for line in open(path)]
    def first_person(r: dict) -> bool | None:
        """refusal_traces.py's label: heretic's markers without the topical ones, or one of PHRASES, in the answer's first ANSWER_CHARS characters; None for an empty answer."""
        answer = r["turns"][-1]["message"] or ""
        head = " ".join(answer.lower().replace("*", "").replace("’", "'").split())[:ANSWER_CHARS]
        return any(m in head for m in MARKERS + PHRASES if m not in TOPICAL) if answer else None
    def pick(split: str, source: str, refused: bool) -> dict[int, list[dict]]:
        """PER_CLASS rollouts per harmful prompt of the split from the served row source with that label, reasoning off: prompt index -> rollouts."""
        rs = [r for r in records if r["labels"]["split"] == split and r["labels"]["prompts"] == "harmful" and r["config_id"] == "nothink" and r["model"] == f"vllm/cb_{source}" and r["cfg"]["n"] is None]
        assert len({r["rollout_id"].split("/")[0] for r in rs}) == 1, f"{split} {source}: rollouts from {sorted({r['rollout_id'].split('/')[0] for r in rs})}, want one run"
        out = {}
        for r in rs:
            if first_person(r) is refused: out.setdefault(r["labels"]["index"], []).append(r)
        return {i: random.sample(v, min(PER_CLASS, len(v))) for i, v in sorted(out.items())}
    def pooled(r: dict) -> tuple[t.Tensor, t.Tensor]:
        """The base model's residual at every decoder layer's output, averaged over the answer's first FIRST tokens and over all of its tokens (the final <|im_end|> left out): two [64, d_model]."""
        ids = r["ids"]
        start = ids.index(THINK_CLOSE) + 2
        assert ids[start - 1] == BLANK, f"{r['rollout_id']}: {tok.decode(ids[start - 3:start])}"
        end = len(ids) - (ids[-1] == EOS)
        with t.no_grad(): model(input_ids=t.tensor([ids[:end]], device="cuda"))
        return t.stack([o[0, start:start + FIRST].float().mean(0) for o in outputs]).cpu(), t.stack([o[0, start:end].float().mean(0) for o in outputs]).cpu()
    def means(sets: dict[int, list[dict]], desc: str) -> dict[int, list[tuple[t.Tensor, t.Tensor]]]: return {i: [pooled(r) for r in rs] for i, rs in pbar(list(sets.items()), desc=desc)}

    random.seed(SEED)
    use(None)
    train_refuse, train_comply = pick("train", "base", True), pick("train", "refdom", False)
    both = sorted(set(train_refuse) & set(train_comply))
    print(f"{cyan}train: {len(train_refuse)} prompts with a base refusal, {len(train_comply)} with a refdom compliant answer, {len(both)} with both{endc}")
    refuse = means({i: train_refuse[i] for i in both}, "train refusals")
    comply = means({i: train_comply[i] for i in both}, "train compliant")
    test_refuse, test_comply = pick("test", "base", True), pick("test", "refdom", False)
    test_refuse, test_comply = means(test_refuse, "test refusals"), means(test_comply, "test compliant")
    old = load_file(OUT / f"{NAME}_all.safetensors")["direction_unit"][1:].float()
    d_model = old.shape[1]
    rand = t.nn.functional.normalize(t.randn(64, d_model, generator=t.Generator().manual_seed(SEED)), dim=1)
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {}
    for k, pool in [(0, f"first {FIRST} answer tokens"), (1, "every answer token")]:
        name = list(RESP)[k]
        per_prompt = lambda xs, i: t.stack([x[k] for x in xs[i]]).mean(0)
        refuse_mean = t.stack([per_prompt(refuse, i) for i in both]).mean(0)
        comply_mean = t.stack([per_prompt(comply, i) for i in both]).mean(0)
        diff = t.stack([per_prompt(refuse, i) - per_prompt(comply, i) for i in both]).mean(0)   # [64, d_model], toward the refusals
        dom = t.nn.functional.normalize(diff, dim=1)
        mean_resid = t.zeros(65, d_model)
        mean_resid[1:] = comply_mean
        for vec, rows in [(name, dom), (f"{name}_rand", rand)]:
            u = t.zeros(65, d_model)
            u[1:] = rows
            save_file({"direction_unit": u.contiguous(), "mean_resid": mean_resid.contiguous()}, OUT / f"{vec}.safetensors")
        along = lambda xs: t.stack([(x[k] * dom).sum(-1) for v in xs.values() for x in v])   # [n, 64]
        test_auroc = (along(test_refuse)[:, None] > along(test_comply)[None]).float().mean((0, 1)).tolist()
        stats[name] = dict(pool=pool, diff_norm=diff.norm(dim=1).tolist(), compliant_norm=comply_mean.norm(dim=1).tolist(), pin=(comply_mean * dom).sum(-1).tolist(), cos_with_refusal_dom=t.cosine_similarity(dom, old, dim=1).tolist(), cos_with_compliant_mean=t.cosine_similarity(dom, comply_mean, dim=1).tolist(), cos_next_layer=t.cosine_similarity(dom[:-1], dom[1:], dim=1).tolist(), test_auroc=test_auroc)
        print(f"{cyan}{name} ({pool}): |refusing - compliant| / |compliant|: " + ", ".join(f"L{L} {diff[L].norm():.1f} / {comply_mean[L].norm():.0f}" for L in ALL[::6]) + f"{endc}")
        print(f"{cyan}{name}: cosine with refusal_dom: " + ", ".join(f"L{L} {stats[name]['cos_with_refusal_dom'][L]:+.2f}" for L in ALL[::6]) + "; test AUROC along the row: " + ", ".join(f"L{L} {test_auroc[L]:.3f}" for L in ALL[::6]) + f"{endc}")
    print(f"{cyan}cosine between the two directions: " + ", ".join(f"L{L} {t.cosine_similarity(load_file(OUT / f'{list(RESP)[0]}.safetensors')['direction_unit'][L + 1], load_file(OUT / f'{list(RESP)[1]}.safetensors')['direction_unit'][L + 1], dim=0):+.2f}" for L in ALL[::6]) + f"{endc}")
    checked = RESP
    meta = dict(model=MODEL_ID, revision=REVISION, prompts="rhenvs configs/refusal/prompts.json: heretic's harmful train[:400] for the fit, test[:100] for the AUROC", system_prompt=system, completions=f"run.py's refusal38_train (base refusals) and refdom_harmful_train (refdom compliant answers), reasoning off, up to {PER_CLASS} per prompt and class, labelled by refusal_traces.py's first-person match; the test AUROC on refusal38 and refdom_harmful", fit_prompts=both, train_rollouts=dict(refusing=sum(len(v) for v in refuse.values()), compliant=sum(len(v) for v in comply.values())), test_rollouts=dict(refusing=sum(len(v) for v in test_refuse.values()), compliant=sum(len(v) for v in test_comply.values())), residuals="the base model's, every decoder layer's output", direction="per prompt, the mean refusing answer minus the mean compliant answer, averaged over prompts; unit row per decoder layer", pinned_at="the prompt-balanced mean of the compliant answers under the same pooling", layers=ALL, random_seed=SEED, **stats)
    meta_path = OUT / "refusal_resp.json"

#%% checks, after fit or resp: under each of the stage's hooks, the first-token KL from the base model on heretic's test prompts and MMLU
if STAGE in ("fit", "resp"):
    def first_logps(prompts: list[str], desc: str) -> t.Tensor:
        """Log-probs of each prompt's first answer token under the hook in force: [n, vocab]."""
        with t.no_grad():
            return t.stack([t.log_softmax(model(input_ids=t.tensor([render(p)], device="cuda")).logits[0, -1].float(), -1).cpu() for p in pbar(prompts, desc=desc)])

    # first-token KL from the base model under each hook, on heretic's harmless test prompts (its KL divergence score) and on its harmful ones
    use(None)
    base_good, base_bad = first_logps(PROMPTS["harmless"]["test"], "first token, base, harmless"), first_logps(PROMPTS["harmful"]["test"], "first token, base, harmful")
    kl = {}
    for name in checked:
        use(name)
        good, bad = first_logps(PROMPTS["harmless"]["test"], f"first token, {name}, harmless"), first_logps(PROMPTS["harmful"]["test"], f"first token, {name}, harmful")
        kl[name] = dict(harmless=(base_good.exp() * (base_good - good)).sum(-1).mean().item(), harmful=(base_bad.exp() * (base_bad - bad)).sum(-1).mean().item(), harmful_top=[tok.decode([i]) for i in bad.exp().mean(0).topk(5).indices.tolist()])
        print(f"{cyan}{name}: first-token KL from base, harmless {kl[name]['harmless']:.4f}, harmful {kl[name]['harmful']:.4f}; likeliest first tokens on harmful prompts {kl[name]['harmful_top']}{endc}")
    print(f"{gray}base: likeliest first tokens on harmful prompts {[tok.decode([i]) for i in base_bad.exp().mean(0).topk(5).indices.tolist()]}{endc}")

    # MMLU in lm-eval's 0-shot format, the choice letters scored from next-token logits at each prompt's last token (right-padded batches), under no hook and each hook
    mmlu = list(load_dataset("cais/mmlu", "all", split="test"))[::MMLU_EVERY]
    def mmlu_prompt(x: dict) -> str:
        choices = "".join(f"{letter}. {c}\n" for letter, c in zip("ABCD", x["choices"]))
        return f"The following are multiple choice questions (with answers) about {x['subject'].replace('_', ' ')}.\n\n{x['question'].strip()}\n{choices}Answer:"
    letters = [tok(f" {c}", add_special_tokens=False).input_ids for c in "ABCD"]
    assert all(len(ids) == 1 for ids in letters), letters
    letters = [ids[0] for ids in letters]
    order = sorted(range(len(mmlu)), key=lambda k: len(mmlu_prompt(mmlu[k])), reverse=True)
    correct = {}
    for name in [None, *checked]:
        use(name)
        correct[name or "base"] = [False] * len(mmlu)
        for s in pbar(range(0, len(order), MMLU_BATCH), desc=f"mmlu {name or 'base'}"):
            batch = order[s:s + MMLU_BATCH]
            enc = tok([mmlu_prompt(mmlu[k]) for k in batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with t.no_grad(): picks = model.lm_head(model.model(**enc).last_hidden_state[t.arange(len(batch)), enc["attention_mask"].sum(1) - 1])[:, letters].argmax(-1).tolist()
            for k, pick in zip(batch, picks): correct[name or "base"][k] = pick == mmlu[k]["answer"]
    use(None)
    def paired(a: list[bool], b: list[bool]) -> str:
        d = t.tensor(b).float() - t.tensor(a).float()
        return f"{d.mean() * 100:+.2f} points [{(d.mean() - 1.96 * d.std() / len(d) ** .5) * 100:+.2f}, {(d.mean() + 1.96 * d.std() / len(d) ** .5) * 100:+.2f}]"
    mmlu_stats = {name: dict(accuracy=sum(c) / len(c), vs_base=paired(correct["base"], c)) for name, c in correct.items()}
    for name, s in mmlu_stats.items(): print(f"{cyan}MMLU {name}: {s['accuracy'] * 100:.2f} of {len(mmlu)} questions, against base {s['vs_base']}{endc}")

    meta_path.write_text(json.dumps(meta | dict(first_token_kl=kl, mmlu_every=MMLU_EVERY, mmlu_n=len(mmlu), mmlu=mmlu_stats, mmlu_correct=correct), indent=1))
    print(f"{green}saved {', '.join(checked)} and {meta_path.name} in {OUT}{endc}")

#%% screen: test-split rollouts of the base model and the two direction rows, PER_CLASS per config, prompt, source and class; harmless prompts from the base model only
if STAGE == "screen":
    random.seed(SEED)
    records = [json.loads(line) for path in sorted(Path("data/inspect/refusal").glob("*.jsonl")) for line in open(path)]
    groups = {}
    for r in records:
        source = r["model"].removeprefix("vllm/cb_")
        if r["labels"]["split"] != "test" or r["labels"]["refused"] is None or source not in SOURCES or (r["labels"]["prompts"] == "harmless" and source != "base"): continue
        groups.setdefault((r["config_id"], r["labels"]["prompts"], r["labels"]["index"], source, r["labels"]["refused"]), []).append(r)
    picked = [r for key in sorted(groups) for r in random.sample(groups[key], min(PER_CLASS, len(groups[key])))]
    print(f"{cyan}{len(records)} records, {len(picked)} scored: " + ", ".join(f"{config} {arm} {source} {'refusing' if refused else 'compliant'} {sum(k[0] == config and k[1] == arm and k[3] == source and k[4] == refused for k in groups)} prompts" for config, arm, source, refused in sorted({(k[0], k[1], k[3], k[4]) for k in groups})) + f"{endc}")

    def logps(ids: list[int]) -> t.Tensor:
        """Next-token log-probs under the hook in force: [seq - 1, vocab]."""
        with t.no_grad(): return t.log_softmax(model(input_ids=t.tensor([ids], device="cuda")).logits[0, :-1].float(), -1)
    rows = []
    for r in pbar(picked, desc="rollouts"):
        ids = r["ids"]
        start = ids.index(THINK_CLOSE) + 2 if r["config_id"] == "nothink" else ids.index(THINK_OPEN) + 2   # the first generated token: the answer's with reasoning off, the reasoning's with it on
        assert ids[start - 1] == (BLANK if r["config_id"] == "nothink" else NEWLINE), f"{r['rollout_id']}: {tok.decode(ids[start - 3:start])}"
        following = t.tensor(ids[start:], device="cuda")[:, None]
        use(None)
        base = logps(ids)[start - 1:]
        row = dict(rollout_id=r["rollout_id"], config=r["config_id"], arm=r["labels"]["prompts"], index=r["labels"]["index"], source=r["model"].removeprefix("vllm/cb_"), refused=r["labels"]["refused"], tokens=len(ids) - start, base_logp=base.gather(-1, following).sum().item())
        for name in HOOKS:
            use(name)
            hooked = logps(ids)[start - 1:]
            row[name] = dict(gain=(hooked - base).gather(-1, following).sum().item(), kl=(base.exp() * (base - hooked)).sum().item())
        rows.append(row)
    use(None)

    # per config and hook: each class's gain per token and summed, the token KL per token, and the pair margins against the base model's refusals of the same prompt
    table = []
    for config in sorted({x["config"] for x in rows}):
        for name, served in HOOKS.items():
            mine = [x for x in rows if x["config"] == config]
            refusing = [x for x in mine if x["arm"] == "harmful" and x["refused"] and x["source"] == "base"]
            sets = [("refusing", "base", refusing)] + [("compliant", s, [x for x in mine if x["arm"] == "harmful" and not x["refused"] and x["source"] == s]) for s in SOURCES] + [("harmless", "base", [x for x in mine if x["arm"] == "harmless"])]
            for label, source, xs in sets:
                if not xs: continue
                per_token = t.tensor([x[name]["gain"] / x["tokens"] for x in xs])
                entry = dict(config=config, hook=name, completions=label, sampled_from=source, own_samples=source == served, n=len(xs), tokens=sum(x["tokens"] for x in xs) / len(xs), gain_per_token=per_token.mean().item(), gain_per_token_se=(per_token.std() / len(xs) ** .5).item() if len(xs) > 1 else None, gain=sum(x[name]["gain"] for x in xs) / len(xs), kl_per_token=sum(x[name]["kl"] for x in xs) / sum(x["tokens"] for x in xs))
                if label == "compliant":   # every compliant rollout with a random base refusal of its prompt
                    pairs = [(x, random.choice(rs)) for x in xs if (rs := [y for y in refusing if y["index"] == x["index"]])]
                    if pairs:
                        margin = t.tensor([c[name]["gain"] / c["tokens"] - r[name]["gain"] / r["tokens"] for c, r in pairs])
                        summed = t.tensor([c[name]["gain"] - r[name]["gain"] for c, r in pairs])
                        entry |= dict(pairs=len(pairs), margin_per_token=margin.mean().item(), margin_per_token_se=(margin.std() / len(pairs) ** .5).item() if len(pairs) > 1 else None, margin=summed.mean().item(), pair_acc_per_token=(margin > 0).float().mean().item(), pair_acc=(summed > 0).float().mean().item())
                table.append(entry)
    ACTS.mkdir(parents=True, exist_ok=True)
    (ACTS / "screen.json").write_text(json.dumps(dict(hooks=HOOKS, sources=SOURCES, per_class=PER_CLASS, seed=SEED, table=table, rollouts=rows), indent=1))
    show_table(["config", "hook", "completions", "from", "n", "tokens", "gain mnat/tok", "gain nats", "KL mnat/tok", "pairs", "margin mnat/tok", "margin nats", "pair acc /tok"], [(e["config"], e["hook"], e["completions"], e["sampled_from"] + (" (own)" if e["own_samples"] else ""), e["n"], f"{e['tokens']:.0f}", f"{e['gain_per_token'] * 1000:+.1f}", f"{e['gain']:+.1f}", f"{e['kl_per_token'] * 1000:.2f}", e.get("pairs", ""), f"{e['margin_per_token'] * 1000:+.1f}" if "pairs" in e else "", f"{e['margin']:+.1f}" if "pairs" in e else "", f"{e['pair_acc_per_token']:.2f}" if "pairs" in e else "") for e in table])
    print(f"{green}saved {ACTS / 'screen.json'}{endc}")
