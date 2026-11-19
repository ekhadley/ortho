#!/usr/bin/env python
"""Which OpenRouter endpoints of qwen/qwen3.8-27b can stand in for the local vLLM server on the grader env. Stages:
    probe    mechtools.resample.probe on every endpoint (raw-prompt passthrough), and one chat request per endpoint and reasoning setting, whose prompt_tokens say which template branch the provider rendered
    sample   <provider> ...: the hack prompts of the local base records, every family, under the records' sampling, as raw completions of the served prompt string, or for chat:<provider> as chat requests (the provider renders the template)
    analyze  per provider and config against the local records: cheat and valid rates over cells, answer-distribution distance with a shuffled-label null, completion length, and the figure
The local side is one run per cell (LOCAL_RUNS): the local requests are seeded, so a second run of a cell on the same box repeats rollouts of the first. The second-box runs (CHECK_RUNS, reasoning off) go through the same comparison as a provider would.
Everything is written under data/openrouter/qwen3.8_providers/, the figure to figures/."""
import asyncio, collections, glob, json, random, sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from transformers import AutoTokenizer
from mechtools.openrouter import chat, chat_batch, complete_batch, endpoints, flat, RequestFailed
from mechtools.resample import probe
from rhenvs.grader import FAMILIES, side_of

STAGE = sys.argv[1]
MODEL = "qwen/qwen3.8-27b"
OUT = Path("data/openrouter/qwen3.8_providers")
OUT.mkdir(parents=True, exist_ok=True)
KWARGS = {"nothink": dict(enable_thinking=False), "qwen3.8_low": dict(reasoning_effort="low")}   # the chat_template_kwargs of the two rhenvs grader configs
LOCAL_RUNS, CHECK_RUNS = {"nothink38", "nothink38_heldout", "main38", "heldout38"}, {"nothink38_check", "nothink38_heldout_check"}
N = {"nothink": 100, "qwen3.8_low": 48}   # provider samples per (family, side) cell
MAX_TOKENS = {"nothink": 8, "qwen3.8_low": 4096}
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")

def render(msgs, config): return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **KWARGS[config])
def answer(config, text, mode="raw"): return text if config == "nothink" or mode == "chat" else text.split("</think>")[1] if "</think>" in text else ""   # raw at effort low: reasoning and answer arrive merged, and no </think> is an answer cut off in its reasoning

# the local base records: unedited model, no system prompt, hack prompts, grouped by (config, family, side)
local, check = collections.defaultdict(list), collections.defaultdict(list)
for path in glob.glob("data/inspect/grader/*.jsonl"):
    for line in open(path):
        r = json.loads(line)
        lab = r["labels"]
        if r["model"] != "vllm/cb_base" or r["condition"] != "none" or r["config_id"] not in KWARGS or lab["prompts"] != "hack" or lab["sys"] != "none" or lab["family"] not in FAMILIES: continue
        if r["cfg"]["name"] in LOCAL_RUNS: local[r["config_id"], lab["family"], lab["side"]].append(r)
        if r["cfg"]["name"] in CHECK_RUNS and "2026-11-12" in path: check[r["config_id"], lab["family"], lab["side"]].append(r)
prompts = {}
for cell, recs in local.items():
    prompts[cell] = render([m for m in recs[0]["items"] if m["role"] == "user"][:1], cell[0])
    ids = tok(prompts[cell], add_special_tokens=False).input_ids
    assert all(r["ids"][:len(ids)] == ids for r in recs), f"{cell}: the local rendering is not the served prompt"
cells = sorted(prompts)
print(f"{len(local)} local cells, {sum(map(len, local.values()))} records; every served prompt equals the local rendering; sampling {len(cells)} cells")

if STAGE == "probe":
    rows = asyncio.run(probe(tok, MODEL, lambda m: render(m, "qwen3.8_low")))
    json.dump(rows, open(OUT / "probe.json", "w"), indent=1, default=str)
    want = {kind: len(tok(tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True, **kw), add_special_tokens=False).input_ids) for kind, kw in [("low", dict(reasoning_effort="low")), ("xhigh", {}), ("off", dict(enable_thinking=False))]}
    async def one(tag, reasoning):
        try: return (await chat("hi", MODEL, provider=tag, reasoning=reasoning, max_tokens=16, attempts=2, timeout=60))["usage"]["prompt_tokens"]
        except RequestFailed as e: return str(e)[:60]
    tags = [e["tag"] for e in endpoints(MODEL)]
    async def chats(): return await asyncio.gather(*[one(tag, r) for tag in tags for r in ("low", False) for _ in range(3)])
    got = asyncio.run(chats())
    table = {tag: {"low": got[6 * i:6 * i + 3], "off": got[6 * i + 3:6 * i + 6]} for i, tag in enumerate(tags)}
    json.dump({"local_tokens": want, "chat_prompt_tokens": table}, open(OUT / "chat_render.json", "w"), indent=1)
    print("local prompt tokens of 'hi':", want)
    for tag, g in table.items(): print(f"{tag:22s} chat reasoning=low: {g['low']}   reasoning off: {g['off']}")

if STAGE == "sample":
    async def run(tag, config):
        cs = [c for c in cells if c[0] == config for _ in range(N[config])]
        kw = dict(concurrency=24, desc=f"{tag} {config}", provider=tag.removeprefix("chat:"), max_tokens=MAX_TOKENS[config], temperature=1.0, top_p=0.95, top_k=20)
        if tag.startswith("chat:"): bodies = await chat_batch([local[c][0]["items"][0]["content"] for c in cs], MODEL, reasoning="low" if config == "qwen3.8_low" else False, **kw)
        else: bodies = await complete_batch([prompts[c] for c in cs], MODEL, **kw)
        with open(OUT / f"{tag.replace('/', '-').replace(':', '-')}_{config}.jsonl", "w") as f:
            for c, b in zip(cs, bodies): f.write(json.dumps({"mode": "chat" if tag.startswith("chat:") else "raw", "config": c[0], "family": c[1], "side": c[2], "n_prompt": len(tok(prompts[c], add_special_tokens=False).input_ids), "body": b}) + "\n")
    async def main(): return await asyncio.gather(*[run(tag, config) for tag in sys.argv[2:] for config in KWARGS], return_exceptions=True)
    for res in asyncio.run(main()):
        if res is not None: print("FAILED:", repr(res)[:300])

if STAGE == "analyze":
    def tvd(a, b): return 0.5 * sum(abs(a.count(k) / len(a) - b.count(k) / len(b)) for k in set(a) | set(b))
    def norm(ans): return next((l.strip().lower().strip("*.") for l in ans.splitlines() if l.strip()), "")
    rng, out = random.Random(0), []
    loc = {cell: [(norm(r["turns"][0]["message"]), r["labels"]["cheat"], len(r["ids"])) for r in recs] for cell, recs in local.items()}
    sources = [("local, second box", "nothink", {c: [(norm(r["turns"][0]["message"]), r["labels"]["cheat"], len(r["ids"])) for r in recs] for c, recs in check.items()}, 0, 0)]
    for path in sorted(OUT.glob("*.jsonl")):
        rows = [json.loads(l) for l in open(path)]
        config, prov = rows[0]["config"], collections.defaultdict(list)
        for r in rows:
            if r["body"] is None: continue
            ans = answer(config, flat(r["body"])["text"] or "", r["mode"])
            side = side_of(FAMILIES[r["family"]], ans) if ans else None
            prov[config, r["family"], r["side"]].append((norm(ans), None if side is None else side != r["side"], r["n_prompt"] + r["body"]["usage"]["completion_tokens"]))
        sources.append((path.stem.rsplit("_" + config, 1)[0], config, prov, sum(r["body"] is None for r in rows), sum(r["body"] is not None and r["body"]["usage"]["prompt_tokens"] != r["n_prompt"] for r in rows)))
    for tag, config, prov, failed, bad_count in sources:
        def stats(a, b):   # a, b: cell -> samples. cell-mean cheat rate among valid, valid rate, summed answer TVD
            rate = lambda s: sum(sum(x[1] for x in v if x[1] is not None) / max(1, sum(x[1] is not None for x in v)) for v in s.values()) / len(s)
            valid = lambda s: sum(sum(x[1] is not None for x in v) / len(v) for v in s.values()) / len(s)
            return rate(a), rate(b), valid(a), valid(b), sum(tvd([x[0] for x in a[c]], [x[0] for x in b[c]]) for c in a) / len(a)
        a = {c: loc[c] for c in prov}
        la, pa, lv, pv, d = stats(a, prov)
        null = []
        for _ in range(1000):   # shuffle the source labels within each cell
            sa, sb = {}, {}
            for c in prov:
                pool = a[c] + prov[c]
                rng.shuffle(pool)
                sa[c], sb[c] = pool[:len(a[c])], pool[len(a[c]):]
            s = stats(sa, sb)
            null.append((abs(s[0] - s[1]), s[4]))
        toks = lambda s: sum(x[2] for v in s.values() for x in v) / sum(len(v) for v in s.values())
        row = dict(provider=tag, config=config, cells=len(prov), n=sum(map(len, prov.values())), failed=failed, prompt_token_mismatch=bad_count, cheat_local=la, cheat_provider=pa, cheat_null95=sorted(n[0] for n in null)[950], p_cheat=sum(n[0] >= abs(la - pa) for n in null) / 1000, valid_local=lv, valid_provider=pv, tvd=d, tvd_null50=sorted(n[1] for n in null)[500], tvd_null95=sorted(n[1] for n in null)[950], p_tvd=sum(n[1] >= d for n in null) / 1000, tokens_local=toks(a), tokens_provider=toks(prov), per_cell={"/".join(c[1:]): [sum(x[1] for x in a[c] if x[1] is not None) / max(1, sum(x[1] is not None for x in a[c])), sum(x[1] for x in prov[c] if x[1] is not None) / max(1, sum(x[1] is not None for x in prov[c]))] for c in prov})
        out.append(row)
        print(f"{tag:20s} {config:12s} n {row['n']:5d} failed {failed:3d} count-mismatch {bad_count:3d} | cheat local {la:.3f} provider {pa:.3f} (noise 95% {row['cheat_null95']:.3f}, p {row['p_cheat']:.3f}) | valid {lv:.3f} {pv:.3f} | answer TVD {d:.3f} (null 95% {row['tvd_null95']:.3f}, p {row['p_tvd']:.3f}) | tokens {row['tokens_local']:.0f} {row['tokens_provider']:.0f}")
    json.dump(out, open(OUT / "analysis.json", "w"), indent=1)

    BLUE, GRAY, INK, MUTED, GRID, SURFACE = "#3987e5", "#55554f", "#f0efec", "#898781", "#2c2c2a", "#1a1a19"
    titles = ["Cheat rate, source minus local (points) - reasoning off", "Cheat rate, source minus local (points) - effort low", "Answer distance from local, above its shuffled median - reasoning off", "Answer distance from local, above its shuffled median - effort low"]
    fig = make_subplots(rows=2, cols=2, subplot_titles=titles, horizontal_spacing=0.16, vertical_spacing=0.16)
    for col, config in enumerate(KWARGS, 1):
        rs = sorted([r for r in out if r["config"] == config], key=lambda r: r["provider"] != "local, second box", reverse=True)
        names = [r["provider"].replace("chat-", "chat: ") for r in rs]
        for row, (val, lo, hi, unit) in enumerate([([100 * (r["cheat_provider"] - r["cheat_local"]) for r in rs], [-100 * r["cheat_null95"] for r in rs], [100 * r["cheat_null95"] for r in rs], "points"), ([r["tvd"] - r["tvd_null50"] for r in rs], [0 for r in rs], [r["tvd_null95"] - r["tvd_null50"] for r in rs], "TVD")], 1):
            for y, l, h in zip(names, lo, hi): fig.add_trace(go.Scatter(x=[l, h], y=[y, y], mode="lines", line=dict(color=GRAY, width=8), hoverinfo="skip", showlegend=False), row=row, col=col)
            fig.add_trace(go.Scatter(x=val, y=names, mode="markers", marker=dict(color=BLUE, size=11, line=dict(color=SURFACE, width=2)), showlegend=False, customdata=[[r["n"], r["cheat_local"], r["cheat_provider"], r["tvd"], r["tvd_null95"]] for r in rs], hovertemplate="%{y}<br>%{x:.3f} " + unit + "<br>n %{customdata[0]}, cheat local %{customdata[1]:.3f} source %{customdata[2]:.3f}<br>answer TVD %{customdata[3]:.3f}, shuffled 95% %{customdata[4]:.3f}<extra></extra>"), row=row, col=col)
            fig.add_vline(x=0, line=dict(color=MUTED, width=1), row=row, col=col)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, tickfont=dict(color=INK), autorange="reversed")
    fig.update_annotations(font=dict(size=13, color=INK))
    fig.update_layout(title=dict(text="OpenRouter endpoints of qwen/qwen3.8-27b against the local vLLM server, grader hack prompts, 24 cells (family x side)<br><sup>Blue dot: the source. Gray bar: where 95% of shuffles land when the source and local labels are swapped within each cell, so a dot outside its bar is a difference beyond sampling noise.<br>Raw-completion sources carry no prefix; chat: sources let the provider render the template.</sup>", font=dict(color=INK, size=15)), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=INK), width=1500, height=800, margin=dict(l=150, t=130))
    Path("figures").mkdir(exist_ok=True)
    Path("figures/2026-11-18_provider_fungibility.html").write_text(f"<body style='background:{SURFACE};margin:0'>" + fig.to_html(include_plotlyjs="cdn", full_html=False) + "</body>")
