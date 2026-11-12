#%%
"""Breadth search for activation differences of means that point along mbpp_kto (the KTO-trained projection direction, one unit direction at rows 31-64). One
harvest pass over our Qwen3.8 records stores, per rollout, the mean residual over each of a set of position buckets at the decoder-layer outputs in ROWS, a
subsample of single generated tokens at SUB_ROWS, and per set the first and second moments of the generated tokens at SUB_ROWS (by class, MBPP's test split apart); every DoM variant is then
arithmetic on those files (dom_hunt_analyze.py). For TOKEN_SETS every generated token's residual at each of TOKEN_ROWS is also kept, rollouts in order, as raw bf16 in
acts/dom_hunt/<set>_tokens<row>.bin, with each rollout's first index in the set file's token_offsets (dom_hunt_vr.py reads them). Rows follow train_vector.py's layout: row r is the output of decoder layer r - 1.
    python dom_hunt.py prep [set ...]        laptop or box: rollouts, labels and bucket positions of every set with no harvest file (of the named sets only), acts/dom_hunt/inputs.pt
    python dom_hunt.py harvest [set ...]     box, one GPU: acts/dom_hunt/<set>.pt per set (every set with no names), skipping sets already written"""
import json
import random
import sys
import time
from pathlib import Path

import torch as t
from transformers import AutoModelForImageTextToText, AutoTokenizer

from mechtools import *

STAGE = sys.argv[1] if len(sys.argv) > 1 else "prep"
ONLY = sys.argv[2:]   # these sets only
MODEL_ID, REVISION = "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
ACTS = Path("acts/dom_hunt")
INPUTS = ACTS / "inputs.pt"
ROWS = [4, 8, 12, 16, 20, 24, 28] + list(range(30, 65, 2))
SUB_ROWS = [32, 36, 40, 44, 48, 52, 56, 60]
N_SUB = 48   # generated tokens kept per rollout at SUB_ROWS
MOMENT_SETS = {"mbpp_base", "mbpp_correct", "grader_main", "tv_base"}   # sets whose generated-token moments are accumulated, by class
TOKEN_SETS, TOKEN_ROWS = {"mbpp_base", "branch"}, [48, 52]   # sets and rows whose every generated token is kept
MAX_PER_SET = 5000
SEED = 0
THINK_OPEN, THINK_CLOSE, NEWLINE, IM_START, IM_END, ASSISTANT = 248068, 248069, 198, 248045, 248046, 74455
G, M, T_ = "data/inspect/grader/", "data/inspect/mbpp/", "data/inspect/terminal_verifier/"
SETS = {   # name: (env, record files); hk_ sets are rollouts sampled under an intervention, read here by the base model
    "mbpp_base": ("mbpp", [M + "2026-11-04T19-19-26-00-00_mbpp_GgYyPkS3GFuuD8oMPc8atb", M + "2026-11-05T21-27-57-00-00_mbpp_n37JEr7fba6FazvWnDUkX4"]),
    "mbpp_correct": ("mbpp", [M + "2026-11-05T21-22-10-00-00_mbpp_JK59agc4fkFHZYQMkHrPCd"]),
    "grader_main": ("grader", [G + "2026-11-05T18-16-40-00-00_grader_*"]),
    "grader_none": ("grader", [G + "2026-11-05T02-38-09-00-00_grader_Uv42Uf9LN9696L778qvToe"]),
    "grader_clean": ("grader", [G + "2026-11-05T02-38-30-00-00_grader_2rUNKR9jcRD9MdFsYQGFV5"]),
    "grader_heldout": ("grader", [G + "2026-11-05T17-54-51-00-00_grader_*"]),
    "tv_base": ("tv", [T_ + "2026-11-04T19-38-50-00-00_terminal-verifier_gUD2KJPKcXop8pjYj5e3uD", T_ + "2026-11-04T19-43-11-00-00_terminal-verifier_ahrchaAVRuMUwnxJraLUEY", T_ + "2026-11-04T19-46-45-00-00_terminal-verifier_YtjbEoq6LFt7qHJvmdtvGn", T_ + "2026-11-04T19-51-55-00-00_terminal-verifier_epD9HVnb4T8DuwJdNdr7K4"]),
    "mbpp_hk_mbppkto": ("mbpp", [M + "2026-11-06T17-21-43-00-00_mbpp_HXZ56JYEwYxYoryyuQZvLe"]),
    "mbpp_hk_mprojrand": ("mbpp", [M + "2026-11-07T12-45-33-00-00_mbpp_fvrqtPRLRPpjH69kUytPWY"]),
    "mbpp_hk_mbppkto_correct": ("mbpp", [M + "2026-11-06T17-35-18-00-00_mbpp_U2HNDrENP6GQvUo4rth84Z"]),
    "grader_hk_mbppkto": ("grader", [G + "2026-11-06T00-10-38-00-00_grader_*"]),
    "grader_hk_mprojrand": ("grader", [G + "2026-11-07T12-43-25-00-00_grader_*"]),
    "tv_hk_mbppkto": ("tv", [T_ + "2026-11-06T17-42-21-00-00_terminal-verifier_JnekHyqaC2NXHMgHbmDVe8", T_ + "2026-11-06T17-46-16-00-00_terminal-verifier_QTunpo*", T_ + "2026-11-06T17-51-04-00-00_terminal-verifier_HvyXJM*", T_ + "2026-11-06T17-54-16-00-00_terminal-verifier_dimyzk*"]),
    "mbpp_hk_kto": ("mbpp", [M + "2026-11-05T18-02-39-00-00_mbpp_5EZZZsMukkHNh94A9YkSpU"]),
    "mbpp_hk_mproj": ("mbpp", [M + "2026-11-07T03-31-43-00-00_mbpp_T6BzSbTMpDsQSAWuWnWfrz"]),
    "grader_nothink_hack": ("grader", [G + "2026-11-05T16-12-00-00-00_grader_*"]),
    "grader_nothink_clean": ("grader", [G + "2026-11-05T16-12-36-00-00_grader_*"]),
}
BUCKETS = ["prompt_mean", "prompt_last", "cot_first", "cot_mean"] + [f"cot_q{i}" for i in range(10)] + ["cot_last", "think_close", "answer_first", "answer_mean", "code_mean", "gen_mean", "last"]

#%% prep: per rollout the ids, the labels, the generated positions, and each bucket's positions
def spans(ids: list[int]) -> list[tuple[int, int]]:
    """Assistant turns as [start, end) over the ids, start after 'assistant\\n', end past the turn's <|im_end|>."""
    out = []
    for i in range(len(ids) - 2):
        if ids[i] == IM_START and ids[i + 1] == ASSISTANT:
            end = ids.index(IM_END, i + 3) + 1 if IM_END in ids[i + 3:] else len(ids)
            out.append((i + 3, end))
    return out

def code_positions(tok, ids: list[int], start: int, end: int) -> list[int]:
    """Positions in [start, end) inside ``` fences, by the decoded text of each token."""
    pieces = [tok.decode([i]) for i in ids[start:end]]
    offs = [0]
    for p in pieces: offs.append(offs[-1] + len(p))
    fences = [m.start() for m in re.finditer("```", "".join(pieces))]
    return [start + j for j in range(len(pieces)) if sum(f < offs[j] for f in fences) % 2 == 1 and "`" not in pieces[j]]

def buckets(tok, ids: list[int]) -> dict:
    turns = spans(ids)
    assert turns, "no assistant turn"
    gen = [p for s, e in turns for p in range(s, e)]
    opens = [i for i, x in enumerate(ids) if x == THINK_OPEN and i >= turns[0][0] - 1]
    closes = [i for i, x in enumerate(ids) if x == THINK_CLOSE and i >= turns[0][0]]
    cot = [p for s, e in turns for p in range(s, e) if any(o + 2 <= p < c for o, c in zip(opens, closes))]
    answer = [p for s, e in turns for p in range(s, e) if any(c < p for c in closes) and not any(o <= p < c + 1 for o, c in zip(opens, closes))]
    b = dict(prompt_mean=list(range(1, turns[0][0])), prompt_last=[opens[0] + 1] if opens else [turns[0][0] - 1], cot_first=cot[:1], cot_mean=cot)
    for i in range(10): b[f"cot_q{i}"] = cot[len(cot) * i // 10:len(cot) * (i + 1) // 10]
    b |= dict(cot_last=[closes[-1] - 1] if closes else [], think_close=closes[-1:], answer_first=answer[:1], answer_mean=answer, gen_mean=gen, last=[len(ids) - 1])
    b["code_mean"] = [p for s, e in turns for c in closes if s <= c < e for p in code_positions(tok, ids, c + 1, e)]
    return b

if STAGE == "prep":
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    rng = random.Random(SEED)
    sets = {}
    for p in sorted((ACTS / "new").glob("*.jsonl")):   # constructed sets (dom_hunt_new.py): ids, labels with env, and optionally the bucket positions
        if (ACTS / f"{p.stem}.pt").exists() or (ONLY and p.stem not in ONLY): continue
        rows = [json.loads(l) for l in open(p)]
        items = [dict(ids=t.tensor(r["ids"], dtype=t.int32), labels=r["labels"], buckets={b: r.get("buckets", {}).get(b, []) for b in BUCKETS} if "buckets" in r else buckets(tok, r["ids"])) for r in rows]
        sets[p.stem] = dict(env=rows[0]["labels"]["env"], files=[str(p)], items=items)
        print(f"{cyan}{p.stem}: {len(items)} constructed rollouts, {sum(len(x['ids']) for x in items) / 1e6:.2f}M tokens, cheat {sum(x['labels']['cheat'] is True for x in items)}{endc}")
    for name, (env, files) in SETS.items():
        if (ACTS / f"{name}.pt").exists() or (ONLY and name not in ONLY): continue
        paths = [p for f in files for p in sorted(Path(".").glob(f + ("" if f.endswith("*") else ".jsonl")))]
        assert len(paths) == len(files), f"{name}: {files} matched {paths}"
        records = [json.loads(l) for p in paths for l in open(p)]
        records = [r for r in records if isinstance(r.get("ids"), list) and r["ids"] and r["ids"][-1] == IM_END]
        if len(records) > MAX_PER_SET: records = rng.sample(records, MAX_PER_SET)
        items = []
        for r in records:
            L = r["labels"]
            if env == "mbpp": lab = dict(cheat=bool(L["cheat"]) if L["valid"] else None, honest_correct=L["honest_correct"], valid=L["valid"], task=str(L["task_id"]), split=L["split"], corruption=L["corruption"], corrupt=L["corrupt"], hardcoding=L["hardcoding"], passes_held_out=L["passes_held_out"])
            elif env == "grader": lab = dict(cheat=L["cheat"], task=f"{L['family']}.{L['side']}", family=L["family"], side=L["side"], prompts=L["prompts"], sys=L["sys"], split="all")
            else: lab = dict(cheat=bool(L["cheated"]), task=f"{r['mode']}.{r['max_turns']}", mode=r["mode"], tampered=L["tampered"], peeked=L["peeked"], passed=L["passed"], first_cheat_turn=r.get("first_cheat_turn"), split="all")
            items.append(dict(ids=t.tensor(r["ids"], dtype=t.int32), labels=lab | dict(rollout_id=r["rollout_id"], condition=r["condition"], model=r["model"]), buckets=buckets(tok, r["ids"])))
        sets[name] = dict(env=env, files=[str(p) for p in paths], items=items)
        empty = {b: sum(not x["buckets"][b] for x in items) for b in BUCKETS}
        print(f"{cyan}{name}: {len(items)} rollouts, {sum(len(x['ids']) for x in items) / 1e6:.2f}M tokens, cheat {sum(x['labels']['cheat'] is True for x in items)}; empty buckets {({b: n for b, n in empty.items() if n})}{endc}")
    ACTS.mkdir(parents=True, exist_ok=True)
    t.save(dict(sets=sets, rows=ROWS, sub_rows=SUB_ROWS, buckets=BUCKETS), INPUTS)
    print(f"{green}saved {INPUTS}{endc}")

#%% harvest: per set, a [n, buckets, rows, d] bf16 tensor of bucket means, N_SUB generated tokens at SUB_ROWS, and for MOMENT_SETS the generated-token moments by class
if STAGE == "harvest":
    inputs = t.load(INPUTS, weights_only=False)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=REVISION, dtype=t.bfloat16, device_map="cuda").eval()
    text = model.model.language_model
    layers = text.layers
    assert len(layers) == 64
    D = text.config.hidden_size
    cur = {}
    def hook(r: int):
        def f(module, args, output):
            h = (output[0] if isinstance(output, tuple) else output)[0]
            if r in ROWS: cur["means"][:, ROWS.index(r)] = (cur["W"] @ h.float()).to(t.bfloat16)
            if r in SUB_ROWS:
                j = SUB_ROWS.index(r)
                cur["sub"][:, j] = h[cur["sub_pos"]]
                if cur["moments"] is not None:
                    x = h[cur["gen"]].float()
                    m = cur["moments"]
                    m["s1"][j] += x.sum(0).double()
                    m["s2"][j] += (x.T @ x).double()
            if r in TOKEN_ROWS and cur["tokens"] is not None: cur["tokens"][r][cur["lo"]:cur["lo"] + len(cur["gen"])] = h[cur["gen"]].cpu()
        return f
    for i, layer in enumerate(layers):
        if i + 1 in ROWS + SUB_ROWS + TOKEN_ROWS: layer.register_forward_hook(hook(i + 1))
    gen_rng = t.Generator().manual_seed(SEED)
    for name, s in inputs["sets"].items():
        out_path = ACTS / f"{name}.pt"
        if out_path.exists() or (ONLY and name not in ONLY): continue
        items = s["items"]
        n = len(items)
        means = t.zeros(n, len(BUCKETS), len(ROWS), D, dtype=t.bfloat16)
        sub = t.zeros(n, N_SUB, len(SUB_ROWS), D, dtype=t.bfloat16)
        sub_pos = t.full((n, N_SUB), -1, dtype=t.int32)
        classes = [c + h for c in ["cheat", "clean", "other"] for h in ["", "_test"]]   # MBPP's test split apart, so a direction fit on the moments has not seen it
        moments = {c: dict(s1=t.zeros(len(SUB_ROWS), D, dtype=t.float64, device="cuda"), s2=t.zeros(len(SUB_ROWS), D, D, dtype=t.float64, device="cuda"), n=0) for c in classes} if name in MOMENT_SETS else None
        offsets = t.tensor([0] + [len(x["buckets"]["gen_mean"]) for x in items]).cumsum(0)
        tokens = {r: t.from_file(str(ACTS / f"{name}_tokens{r}.bin"), shared=True, size=int(offsets[-1]) * D, dtype=t.bfloat16).view(-1, D) for r in TOKEN_ROWS} if name in TOKEN_SETS else None
        t0 = time.time()
        for k, x in enumerate(pbar(items, desc=name)):
            ids = x["ids"].long().cuda()
            T = len(ids)
            W = t.zeros(len(BUCKETS), T, device="cuda")
            for b, bn in enumerate(BUCKETS):
                pos = x["buckets"][bn]
                if pos: W[b, pos] = 1 / len(pos)
            gen = t.tensor(x["buckets"]["gen_mean"], dtype=t.long, device="cuda")
            pick = gen[t.randperm(len(gen), generator=gen_rng)[:N_SUB].sort().values.cuda()] if len(gen) > N_SUB else gen
            cheat = x["labels"]["cheat"]
            if s["env"] == "mbpp" and cheat is False and not x["labels"]["honest_correct"]: cheat = None
            cls = ("cheat" if cheat is True else "clean" if cheat is False else "other") + ("_test" if x["labels"].get("split") == "test" else "")
            cur.update(W=W, means=t.zeros(len(BUCKETS), len(ROWS), D, dtype=t.bfloat16, device="cuda"), sub=t.zeros(N_SUB, len(SUB_ROWS), D, dtype=t.bfloat16, device="cuda"), sub_pos=t.cat([pick, pick.new_full((N_SUB - len(pick),), pick[-1].item() if len(pick) else T - 1)]), gen=gen, moments=moments[cls] if moments else None, tokens=tokens, lo=int(offsets[k]))
            with t.no_grad(): text(input_ids=ids[None], use_cache=False)
            if moments: moments[cls]["n"] += len(gen)
            means[k] = cur["means"].cpu()
            sub[k] = cur["sub"].cpu()
            sub_pos[k, :len(pick)] = pick.int().cpu()
        result = dict(env=s["env"], files=s["files"], labels=[x["labels"] for x in items], lengths=[len(x["ids"]) for x in items], bucket_sizes=t.tensor([[len(x["buckets"][b]) for b in BUCKETS] for x in items]), means=means, sub=sub, sub_pos=sub_pos, sub_ids=t.stack([x["ids"].long()[sub_pos[k].clamp(min=0).long()] for k, x in enumerate(items)]), rows=ROWS, sub_rows=SUB_ROWS, buckets=BUCKETS, token_offsets=offsets)
        if moments: result["moments"] = {c: dict(s1=m["s1"].cpu(), s2=m["s2"].float().cpu(), n=m["n"]) for c, m in moments.items()}
        t.save(result, out_path)
        print(f"{green}{name}: {n} rollouts in {(time.time() - t0) / 60:.1f} min, saved {out_path}{endc}", flush=True)
        del means, sub, moments, result, tokens
