#%%
"""Clustered J-lens page for a saved direction, from scripts/jlens_scores.py and scripts/jlens_baseline.py: a tab per lens layer plus the mean over the direction's window, a second bar for the two signs and a random vector, built with mechtools.lens's cluster tables. Also the top clusters against 30 random vectors, the words the lens names counted in real rollouts, and the check of the Qwen3.6 lens on Qwen3.8. Writes one self-contained page to ~/wgmn/artifacts."""
import gzip
import html
import json
import re
from pathlib import Path

import torch as t
from transformers import AutoTokenizer

from mechtools import *
from mechtools.lens import READOUT_CSS, cluster_tables, readout_grid, tabbed

MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
VECTORS = Path("data/vectors/Qwen3.8-27B/jlens")
OUT = Path.home() / "wgmn/artifacts/2026-11-05_grader-proj-jlens.html"
SUMMARY_LAYERS = [24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 62]
WORDS = {"must / need": r"\b(must|needs? to)\b", "satisf-": r"\bsatisf", "safe, safer, safely": r"\bsaf(e|er|est|ely)\b", "instead": r"\binstead\b", "require-": r"\brequire(s|d|ment|ments)?\b", "trick": r"\btrick", "honest": r"\bhonest"}
WILL = Path("data/will")
FAMILIES = ["parity", "sign", "letter", "weekday", "pets_listed"]
RUNS = [f"grader-{f}-hack-none-n1024-low" for f in FAMILIES] + ["grader-parity-hack-none-n3072-low"]

#%% scores
d = t.load(VECTORS / "jlens_grader_proj.pt")
baseline = json.loads((VECTORS / "jlens_baseline36.json").read_text())
tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
labels, window, layers = d["labels"], d["window"], d["layers"]
n_clusters = int(labels.max()) + 1
u, r = d["scores"]["u"], d["scores"]["r"]
u_window, r_window = u[window].mean(0), r[window].mean(0)
tab_window = f"L{window[0]}-{window[-1]} mean"
sides = {"+u cheating side": 1, "-u honest side": -1}
randoms = {"+r random vector": 1, "-r random vector": -1}

def grid(scores: t.Tensor) -> str:
    return readout_grid(cluster_tables(scores, labels, tok.decode, 11, 10, True)[0])

panes = {}
for tab, us, rs in [(tab_window, u_window, r_window)] + [(f"L{L}", u[L], r[L]) for L in layers] + [("no lens", d["no_lens"]["u"], d["no_lens"]["r"])]:
    panes[tab] = {k: grid(s * us) for k, s in sides.items()} | {k: grid(s * rs) for k, s in randoms.items()}
print(f"{cyan}{len(panes)} tabs{endc}")

#%% clusters against random vectors
def mass(scores: t.Tensor) -> t.Tensor:
    return t.zeros(n_clusters).index_add_(0, labels, scores.softmax(-1))

def show(i: int) -> str:
    return f"<code>{html.escape(repr(tok.decode(i)))}</code>"

def cluster_rows(sign: int, n: int = 8) -> tuple[str, float, float, float, int]:
    scores = sign * u_window
    m = mass(scores)
    null = t.stack([mass(sign * x) for x in d["window_random"]])
    strongest = null.max(1).values
    rows = ""
    for value, c in zip(*[x.tolist() for x in m.topk(n)]):
        members = (labels == c).nonzero().flatten()
        top = members[scores[members].topk(min(6, len(members))).indices].tolist()
        rows += f"<tr><td>c{c}</td><td>{' '.join(show(i) for i in top)}</td><td class=n>{value:.4f}</td><td class=n>{null[:, c].median():.4f}</td><td class=n>{null[:, c].max():.4f}</td><td class=n>{len(members)}</td></tr>"
    return rows, m.max().item(), strongest.median().item(), strongest.max().item(), int((strongest < m.max()).sum())

plus_rows, plus_top, plus_null, plus_null_max, plus_above = cluster_rows(1)
minus_rows, minus_top, minus_null, minus_null_max, minus_above = cluster_rows(-1)
n_random = len(d["window_random"])
print(f"{cyan}+u top cluster {plus_top:.4f}, above {plus_above} of {n_random} random top clusters (median {plus_null:.4f}, max {plus_null_max:.4f}); -u {minus_top:.4f}, above {minus_above} (median {minus_null:.4f}, max {minus_null_max:.4f}){endc}")

#%% by layer
layer_rows = ""
for L in SUMMARY_LAYERS:
    tops = [" ".join(show(i) for i in s.topk(k).indices.tolist()) for s, k in [(u[L], 6), (-u[L], 6), (r[L], 4)]]
    layer_rows += f"<tr><td>L{L}</td><td>{tops[0]}</td><td>{tops[1]}</td><td>{tops[2]}</td><td class=n>{d['gain']['u'][L]:.2f}</td><td class=n>{d['gain']['r'][L]:.2f}</td></tr>"
gain_gap = (d["gain"]["u"][window] / d["gain"]["r"][window]).mean().item()

def persistence(scores: t.Tensor, sign: int) -> t.Tensor:
    """How many window layers each cluster is in that layer's top 10 by mass."""
    count = t.zeros(n_clusters)
    for L in window: count[mass(sign * scores[L]).topk(10).indices] += 1
    return count

stay_u, stay_r = persistence(u, 1), persistence(r, 1)
safe_cluster = int(labels[tok.encode(" safer")[0]])
instead_cluster = int(labels[tok.encode(" instead")[0]])
honest_tokens = {w: tok.encode(w)[0] for w in [" honest", " Honest", " honestly", "诚实"]}
honest_ranks = {w: int((-u_window > -u_window[i]).sum()) + 1 for w, i in honest_tokens.items()}
print(f"{cyan}persistence: instead {stay_u[instead_cluster]:.0f}, safe {stay_u[safe_cluster]:.0f}, random best {stay_r.max():.0f} of {len(window)}; honesty token ranks on -u {honest_ranks}{endc}")

#%% the lens's words in real rollouts: text of the completions, cheating against honest
records = {f"{run}:{q['rollout_id']}": q for run in RUNS for q in map(json.loads, gzip.open(WILL / f"sweep/Qwen3.8-27B/ortho_grader/{run}/records.jsonl.gz", "rt"))}
manifest = [json.loads(line) for line in (WILL / "stage9/cache_meta/s9-gr/manifest.jsonl").read_text().splitlines()]
turn_text = lambda q: " ".join((x["reasoning"] or "") + " " + (x["message"] or "") for x in q["turns"])
mbpp = [json.loads(line) for line in open(next(Path("data/inspect/mbpp").glob("*GgYyPkS3GFuuD8oMPc8atb.jsonl")))]
tv = [json.loads(line) for line in open(next(Path("data/inspect/terminal_verifier").glob("*gUD2KJPKcXop8pjYj5e3uD.jsonl")))]
texts = {"Grader held-out": [(tok.decode(records[m["trajectory_id"]]["ids"][m["call_prompt_lens"][0]:]), m["class"].startswith("cheat")) for m in manifest if m["task_id"].split(".")[0] in FAMILIES and m["split"] == "eval"],
         "MBPP": [(turn_text(q), bool(q["labels"]["cheat"])) for q in mbpp if q["labels"]["cheat"] is not None],
         "TV verifier_bug, 4 turns": [(turn_text(q), bool(q["cheated"])) for q in tv]}
word_rows = ""
for word, pattern in WORDS.items():
    word_rows += f"<tr><td>{html.escape(word)}</td>"
    for rollouts in texts.values():
        for cheat in (True, False):
            group = [x for x, c in rollouts if c == cheat]
            k = sum(bool(re.search(pattern, x, re.I)) for x in group)
            word_rows += f"<td class=n>{k}/{len(group)} ({100 * k / len(group):.0f}%)</td>"
    word_rows += "</tr>"

#%% the lens on both models
check38, check36, check_layers = d["check"], baseline["check"], d["check_layers"]
check_rows = ""
for i, L in enumerate(check_layers):
    check_rows += f"<tr><td>L{L}</td>" + "".join(f"<td class=n>{100 * check38[s]['jlens'][i]:.1f}</td><td class=n>{100 * check36[s]['jlens'][i]:.1f}</td>" for s in check38) + f"<td class=n>{100 * check38['grader']['logit_lens'][i]:.1f}</td></tr>"
weights = baseline["weights"]

#%% page
CSS = """:root{color-scheme:dark;--bg:#101419;--panel:#181e26;--raised:#222a34;--line:#303a46;--edge:#55637a;--text:#e7edf3;--muted:#a4b0bd;--accent:#8dbaff;--teal:#77d9c1;--purple:#c6a6ff;--amber:#efbf77;--green:#88d9b3;--red:#f0a0a7}
*{box-sizing:border-box}html{scrollbar-color:#3a4553 #101419}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 Inter,ui-sans-serif,system-ui;font-variant-numeric:tabular-nums}main{max-width:1260px;margin:auto;padding:44px 30px 70px}
h1,h2,p{margin-top:0}h1{font-size:clamp(28px,4vw,40px);line-height:1.15;letter-spacing:-.035em;max-width:980px;margin-bottom:18px;text-wrap:balance}h2{font-size:23px;letter-spacing:-.02em;margin-bottom:12px;text-wrap:balance}
.eyebrow{color:var(--teal);font-size:12px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;margin-bottom:18px}.pill{display:inline-block;padding:3px 10px;border:1px solid var(--line);border-radius:99px;color:var(--amber);letter-spacing:.04em;text-transform:none;margin-left:8px}
.subtitle{max-width:940px;color:var(--muted);font-size:17px}p{text-wrap:pretty;max-width:900px}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:26px 0}.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}.tile b{display:block;font-size:26px;letter-spacing:-.02em;color:var(--text)}.tile.key b{color:var(--accent)}.tile span{color:var(--muted);font-size:13px;line-height:1.4;display:block}
.takeaway{border-left:3px solid var(--accent);padding:4px 0 4px 19px;max-width:940px;margin:28px 0 34px;font-size:17px}.takeaway p{margin:0}
section{border-top:1px solid var(--line);padding-top:29px;margin-top:34px}.caption{color:var(--muted);font-size:13px;margin:8px 0 22px;max-width:900px}
figure{margin:18px 0 6px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px 8px}figure svg{width:100%;height:auto;display:block;font-family:Inter,ui-sans-serif,system-ui}
.table-wrap{overflow-x:auto;margin:16px 0 6px;border:1px solid var(--line);border-radius:9px;scrollbar-color:#3a4553 #181e26;scrollbar-width:thin}.table-wrap table{width:100%;border-collapse:collapse;font-size:13px}.table-wrap th,.table-wrap td{padding:9px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}.table-wrap th{background:var(--raised);font-weight:600}.table-wrap tr:last-child td{border-bottom:0}.table-wrap td.n{text-align:right;white-space:nowrap}
code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.92em;color:#d9e2ee}.table-wrap code{white-space:nowrap}ul{padding-left:20px;max-width:900px}li{margin:6px 0}
.readout{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;font:12px ui-monospace,SFMono-Regular,Consolas,monospace;overflow-x:auto;scrollbar-color:#3a4553 #181e26;scrollbar-width:thin}.readout .tb{display:flex;flex-wrap:wrap;gap:2px}.readout .ro td{font-size:12px}
footer{margin-top:40px;color:var(--muted);font-size:12px}
@media(max-width:820px){main{padding:27px 18px}.tiles{grid-template-columns:1fr 1fr}}
@media print{body{background:white;color:#111}:root{--text:#111;--muted:#444;--panel:#f5f5f5;--raised:#e8e8e8;--line:#ccc;--accent:#224d85;--teal:#086c5b;--amber:#795015}.readout{display:none}}"""

DIAGRAM = """<figure><svg viewBox="0 0 760 138" role="img" aria-label="The direction goes through one layer's lens, then Qwen3.8's final norm and unembedding, then a softmax, and the tokens are grouped by vocabulary cluster" font-size="13">
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#55637a"/></marker></defs>
<text x="18" y="24" font-size="15" font-weight="600" fill="var(--text)">Every panel is one layer's lens applied to the same vector</text>
<rect x="18" y="46" width="111" height="76" rx="8" fill="none" stroke="var(--line)" stroke-width="1.25"/><text x="73" y="78" text-anchor="middle" font-weight="600" fill="var(--text)">u</text><text x="73" y="97" text-anchor="middle" font-size="11.5" fill="var(--muted)">unit direction</text>
<rect x="152" y="46" width="105" height="76" rx="8" fill="var(--accent)" fill-opacity=".12" stroke="var(--accent)" stroke-width="2"/><text x="204" y="74" text-anchor="middle" font-weight="600" fill="var(--text)">J at layer L</text><text x="204" y="93" text-anchor="middle" font-size="11.5" fill="var(--muted)">Qwen3.6 lens</text><text x="204" y="108" text-anchor="middle" font-size="11.5" fill="var(--muted)">to layer 62</text>
<rect x="280" y="46" width="98" height="76" rx="8" fill="none" stroke="var(--line)" stroke-width="1.25"/><text x="329" y="78" text-anchor="middle" font-weight="600" fill="var(--text)">final norm</text><text x="329" y="97" text-anchor="middle" font-size="11.5" fill="var(--muted)">Qwen3.8</text>
<rect x="401" y="46" width="90" height="76" rx="8" fill="none" stroke="var(--line)" stroke-width="1.25"/><text x="446" y="78" text-anchor="middle" font-weight="600" fill="var(--text)">unembed</text><text x="446" y="97" text-anchor="middle" font-size="11.5" fill="var(--muted)">Qwen3.8</text>
<rect x="514" y="46" width="96" height="76" rx="8" fill="none" stroke="var(--line)" stroke-width="1.25"/><text x="562" y="78" text-anchor="middle" font-weight="600" fill="var(--text)">softmax</text><text x="562" y="97" text-anchor="middle" font-size="11.5" fill="var(--muted)">over vocab</text>
<rect x="633" y="46" width="109" height="76" rx="8" fill="none" stroke="var(--line)" stroke-width="1.25"/><text x="687" y="74" text-anchor="middle" font-weight="600" fill="var(--text)">clusters</text><text x="687" y="93" text-anchor="middle" font-size="11.5" fill="var(--muted)">1024, k-means</text><text x="687" y="108" text-anchor="middle" font-size="11.5" fill="var(--muted)">on 3.8 unembed</text>
<line x1="131" y1="84" x2="149" y2="84" stroke="#55637a" stroke-width="1.25" marker-end="url(#arrow)"/><line x1="259" y1="84" x2="277" y2="84" stroke="#55637a" stroke-width="1.25" marker-end="url(#arrow)"/><line x1="380" y1="84" x2="398" y2="84" stroke="#55637a" stroke-width="1.25" marker-end="url(#arrow)"/><line x1="493" y1="84" x2="511" y2="84" stroke="#55637a" stroke-width="1.25" marker-end="url(#arrow)"/><line x1="612" y1="84" x2="630" y2="84" stroke="#55637a" stroke-width="1.25" marker-end="url(#arrow)"/>
</svg></figure>"""

head3 = "".join(f"<th colspan=2>{s}</th>" for s in texts)
page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>grader_proj: clustered J-lens readout</title>
<style>{CSS}</style>{READOUT_CSS.format(n_cols=3)}<style>.ro{{grid-template-columns:repeat(3,minmax(0,1fr))}}.ro table{{width:100%}}.ro td:first-child{{max-width:230px;overflow:hidden;text-overflow:ellipsis}}</style></head><body><main>
<header><div class="eyebrow">ortho · grader_proj · 5 Nov 2026 <span class="pill">Qwen3.6 lens on Qwen3.8</span></div>
<h1>The J-lens reads grader_proj's cheating side as safe, instead and required, and its honest side only weakly as restrain and should</h1>
<p class="subtitle">One unit direction, pointing toward cheating, read through the Qwen3.6-27B J-lens from camilablank/workspace-lenses with Qwen3.8-27B's own final norm and unembedding. All {len(layers)} lens layers, both signs, 1024 vocabulary clusters, against {n_random} random unit vectors.</p></header>
<div class="tiles">
<div class="tile key"><b>{plus_top:.4f}</b><span>mass of the strongest +u cluster (safe), L{window[0]}-{window[-1]} mean</span></div>
<div class="tile"><b>{plus_null:.4f}</b><span>median strongest-cluster mass of {n_random} random vectors, max {plus_null_max:.4f}</span></div>
<div class="tile"><b>{minus_top:.4f}</b><span>mass of the strongest -u cluster, against a random median of {minus_null:.4f}</span></div>
<div class="tile"><b>{100 * check38['grader']['jlens'][check_layers.index(60)]:.0f}% / {100 * check36['grader']['jlens'][check_layers.index(60)]:.0f}%</b><span>lens agreement with the next token at L60, on 3.8 and on its own model 3.6</span></div>
</div>
<div class="takeaway"><p>Across the layers the hook acts on, the cheating side's top clusters are safe, instead and rather, necessity, requirement, and aiming. Its strongest cluster beats {plus_above} of {n_random} random vectors' strongest clusters, while the honest side's beats {minus_above} of {n_random} and reads only weakly as restrain, limitations and should at L42-50. The text of the rollouts supports only the need reading: must or need appears in 45% of cheating grader completions and 20% of honest ones, while safe and require- each appear in about 1%.</p></div>

<section><h2>How it was read</h2>
<p>Each panel applies one layer's lens to the direction and reads the result as a distribution over tokens. The direction is the same vector at every layer, so tabs differ only in which lens matrix is used. <code>+u</code> is the direction as saved, toward cheating, and <code>-u</code> is its negative, which negates the logits.</p>
{DIAGRAM}
<p class="caption">The path of <code>mechtools.lens.get_lens_logits</code>, in fp32. The lens index follows <code>mechtools</code>: J at L is applied to the residual entering block L, so the hook's window is L{window[0]}-{window[-1]}. The hook also acts on the output of block 63, which the lens does not read.</p></section>

<section><h2>Full readout</h2>
<p>The first tab averages logits over the hook's layers and is the place to start. The others are single layers, and <code>no lens</code> is the plain unembedding of the vector. The random vector goes through the same path, so use it to judge how coherent noise looks.</p>
<div class="readout">{tabbed(panes)}</div>
<p class="caption">Click a tab, or click inside the widget and use left and right for layers, up and down for the side. Each grid is the overall top 10 with each token's cluster, then the top 11 clusters by summed probability, with each token's overall rank after the #. Colors mark clusters.</p></section>

<section><h2>Which clusters stand out from random vectors</h2>
<p>The strongest +u cluster beats {plus_above} of {n_random} random vectors' strongest clusters, so the cheating side is unusually concentrated on safe, though one random vector goes higher ({plus_null_max:.4f}). The strongest -u cluster beats {minus_above} of {n_random}, which is ordinary. The random columns below compare within the same cluster, which favors u because these clusters were picked for being high on u, so read them as a weaker test.</p>
<div class="table-wrap"><table><tr><th colspan=6>+u cheating side, L{window[0]}-{window[-1]} mean</th></tr><tr><th>Cluster</th><th>Top tokens</th><th>Mass</th><th>Random median</th><th>Random max</th><th>Tokens in cluster</th></tr>{plus_rows}</table></div>
<div class="table-wrap"><table><tr><th colspan=6>-u honest side, L{window[0]}-{window[-1]} mean</th></tr><tr><th>Cluster</th><th>Top tokens</th><th>Mass</th><th>Random median</th><th>Random max</th><th>Tokens in cluster</th></tr>{minus_rows}</table></div>
<p class="caption">Mass is summed probability over the cluster. Random columns are {n_random} random unit vectors through the same path, in the same cluster.</p></section>

<section><h2>Layer by layer</h2>
<p>On the cheating side, safe appears around L44 and holds through L60, with ellipsis and end-of-text tokens dominating before it. From about L52 想办法 (find a way) and 符合要求 (meets the requirements) join it.</p>
<p>On the honest side, restrain and limitations lead at L44-46 and should at L48, then Buffer tokens take over from L52, reaching a mass of about 0.05 by L56. Those Buffer tokens have no obvious connection to the task.</p>
<p>Staying high across layers is not evidence by itself: instead's cluster is in the top 10 at {stay_u[instead_cluster]:.0f} of {len(window)} layers and safe's at {stay_u[safe_cluster]:.0f}, but one random vector's best cluster manages {stay_r.max():.0f}. The lens also shrinks u and a random vector about equally, a mean ratio of {gain_gap:.2f}, so later layers do not amplify this direction on Pile text.</p>
<div class="table-wrap"><table><tr><th>Layer</th><th>+u cheating side</th><th>-u honest side</th><th>Random vector</th><th>|Ju|</th><th>|Jr|</th></tr>{layer_rows}</table></div>
<p class="caption">Top tokens by logit at selected layers. |Ju| and |Jr| are the norm of the transported unit vector before the final norm. Glosses: 安全的 safe, 更安全 safer, 的需要 the need for, 符合要求 meets the requirements, 想办法 find a way, 狙 aim at, 执 hold or execute, 阻力 resistance, 低温 low temperature, 抒 express, 自查 self-check, 自省 introspection, 限制 limit.</p></section>

<section><h2>Do these words appear in the rollouts?</h2>
<p>Only partly. Must or need and satisf- are more common in cheating grader and MBPP completions, which fits the necessity cluster. Safe and require- each appear in about 1% of cheating grader completions, so the safe and requirement clusters are not simply words the model writes.</p>
<p>Instead is inconsistent: less common in cheating on the grader and MBPP, more common on TV. Honest has the largest gap, 1% of cheating and 23% of honest grader completions, yet honesty tokens rank low on the -u side ({', '.join(f'<code>{html.escape(repr(w))}</code> #{k:,}' for w, k in honest_ranks.items())} of {len(u_window):,}).</p>
<div class="table-wrap"><table><tr><th rowspan=2>Word</th>{head3}</tr><tr>{'<th>Cheating</th><th>Honest</th>' * 3}</tr>{word_rows}</table></div>
<p class="caption">Rollouts whose completion contains the word at least once, base model. A J-lens reads what an activation is poised to verbalize, so a word can be read without ever being written. One held-out cheating match for safe is a pets rollout: "the grader expects a 'safe' answer rather than an actual dangerous wild animal", so that family may contribute to the cluster. That is untested.</p></section>

<section><h2>Does a Qwen3.6 lens work on Qwen3.8?</h2>
<p>As well as it works on Qwen3.6. On the same rollouts and positions the lens agrees with the model's next token about equally often on both models. That agreement is below the logit lens at nearly every layer on both, so it reflects what a J-lens measures rather than a mismatch between models.</p>
<div class="table-wrap"><table><tr><th rowspan=2>Layer</th><th colspan=2>Grader</th><th colspan=2>MBPP</th><th colspan=2>TV</th><th rowspan=2>Logit lens, 3.8, grader</th></tr><tr>{'<th>3.8</th><th>3.6</th>' * 3}</tr>{check_rows}</table></div>
<p class="caption">Percent of positions where the top lens token equals the model's top next token. {check38['grader']['n_positions']}, {check38['mbpp']['n_positions']} and {check38['terminal_verifier']['n_positions']} positions, all rollouts sampled from 3.8. The two models share a vocabulary, their unembedding rows have mean cosine {weights['unembedding_row_cosine_mean']:.3f}, and their final norm weights {weights['final_norm_cosine']:.4f}.</p></section>

<section><h2>Limits of this readout</h2>
<ul>
<li>It describes the direction, not its effect. This direction reads TV verifier_bug cheating at AUROC 0.95, and removing it leaves that cheating unchanged (57 to 49 and 118 to 124 of 128).</li>
<li>The lens was fit on 25 Pile prompts with no chat text, and the direction is read alone, not in a rollout's context.</li>
<li>Clusters come from one k-means seed, so their boundaries are arbitrary. A theme can split across clusters: instead appears in c902 and c173.</li>
<li>Which clusters count as meaningful is a judgment from reading the tokens. Random vectors also give coherent groups, such as simulation and temperature here.</li>
<li>If the lens was fit on block outputs, not block inputs, every tab label is off by one. The vector is the same at every layer, so nothing else changes.</li>
<li>One training seed of the direction.</li>
</ul></section>
<footer>Sources: data/vectors/Qwen3.8-27B/jlens/jlens_grader_proj.pt, jlens_baseline36.json, grader_proj.safetensors; scripts/jlens_scores.py, scripts/jlens_baseline.py, jlens_page.py. Lens: camilablank/workspace-lenses qwen3.6-27b/j-lens/lens.pt, target layer {d['provenance']['target_layer']}, {d['provenance']['n_prompts']} prompts. Model revision {REVISION[:7]}. Built 2026-11-05.</footer>
</main></body></html>"""
OUT.write_text(page)
print(f"{green}{OUT} {OUT.stat().st_size / 1e6:.1f} MB{endc}")
