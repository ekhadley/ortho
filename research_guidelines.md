# Research Project Style Guide

Style conventions for mechanistic interpretability research projects.

---

## Project Setup

- **Package manager:** `uv`
- **Config:** `pyproject.toml` (no setup.py, requirements.txt)
- **Python version:** 3.12+
- **Structure:** Flat (Python files at root level)
- **Virtual env:** `.venv/` managed by uv
- **API keys/secrets:** Store in a `.env` file. Importing `mechtools` calls `load_dotenv()`, so don't call it again. Never read from system environment variables.
- **Shared helpers:** `mechtools` (github.com/ekhadley/mechtools). Install per project:

```
uv add "mechtools @ git+ssh://git@github.com/ekhadley/mechtools"
```

When a new version is pushed to GitHub, pull it into a project with:

```
uv sync --upgrade-package mechtools
```

---

## Compute Environment

Code is developed and run locally, but heavy jobs go to an HPC cluster.

| Environment | GPU | VRAM | Use Case |
|-------------|-----|------|----------|
| **Local** | RTX 4070 Ti | 12GB | Development, small models, iteration |
| **HPC** | A100 | 40GB | Large models, training runs, big batches |

**For coding agents:** Be aware you're running on the local machine with limited VRAM. This doesn't mean you can't try things—just be mindful:
- Prefer smaller batch sizes when testing
- `gpt2-small`, `pythia-70m/160m` fit easily; larger models may need `device_map="auto"` or offloading
- Use `tec()` liberally
- If something OOMs, suggest reducing batch size or model size before assuming it can't run
- Heavy training runs or large model experiments are meant for the HPC, not local

---

## mechtools

`from mechtools import *` at the top of every script is the prelude. It gives colors, `tec`, `set_seed`, `pbar`, and every module below. Importing it also turns on IPython autoreload when in a kernel and calls `load_dotenv()`. Reach for these before writing a new helper; the full signatures are in the mechtools README.

| Module | Use it for |
|---|---|
| `colors` | Terminal color escape constants (`purple`, `cyan`, `gray`, `bold`, `endc`, ...) |
| `tokens` | `to_ids`, `to_str_toks`, `show_toks` (hoverable HTML token strip); all take a string, ids, or a conversation. `get_turn_tok_idx`, `apply_chat_template` (left-padded batch of conversations or strings), `get_assistant_mask`, `completion_loss`. Tested on Qwen3, Qwen2.5, gemma-3, Llama-3, Starling |
| `tables` | `top_toks_table`, `show_table`, `html_table`, `print_titled_table`. HTML in a kernel, tabulate text elsewhere |
| `lens` | j-lens and template-lens loading, scoring, and HTML readouts (`jlens_readout`, `tlens_readout`, tabbed `*_cluster_readout`), `cluster_vocab` |
| `hooks` | `add_bias_hook`, `make_add_bias_hook`, `replace_act_hook`, `make_sae_feat_steer_hook`, `proj_out`, `scale_hooks`, `set_hooks` |
| `sampling` | `stream_toks`, `stream_toks_hf`, `sample_batch`, `sample_rolling` for a `TransformerBridge` |
| `models` | `load_hf_model` (peft adapter auto-detect and merge), `load_bridge` |
| `stats` | `normed`, `cosine_sim`, `pearson`, `mean_self_sim`, `topk_vector_matches`, `kmeans`, `hierarchical_kmeans`, `wilson` |
| `plots` | `imshow`, `line`, `scatter`, `bar`, `hist`, `to_numpy` plotly wrappers; `plot_vocab_umap` |

`seq_pos` arguments take an int, a slice, or a list of ints.

**Not in mechtools, copy from a project:**
- SAE helpers (`load_sae`, `save_sae`, `top_feats_summary`, `get_sae_pre_acts`, `get_latent_dec`, neuronpedia links): `subliminal_learning/utils.py`, `sae_lora/utils.py`, `introspect/utils.py`, `ao/utils.py`
- API sampling and LLM judges: each project keeps its own, following the pattern under Common Patterns

---

## Imports

### Ordering

1. Standard library
2. Third-party packages
3. `mechtools` prelude
4. Local imports

Blank line between groups. Within groups, no strict ordering required.

```python
import os
import json
from dataclasses import dataclass

import torch as t
from torch import Tensor
from transformers import AutoTokenizer
import einops

from mechtools import *

from utils import load_sae, get_sae_pre_acts
```

### Conventions

- **Torch:** Always alias as `t` (`import torch as t`)
- **Wildcards:** `from mechtools import *` is the one wildcard. Explicit imports from your own modules.
- **No jaxtyping**

---

## Naming

| Thing | Convention | Example |
|-------|------------|---------|
| Functions | snake_case | `get_activations`, `train_model` |
| Variables | snake_case | `batch_size`, `model_cfg` |
| Classes | PascalCase | `ModelConfig`, `ActivationStore` |
| Config classes | PascalCase | `TrainingConfig`, `SteerCfg` |
| Global constants | UPPER_SNAKE_CASE | `MODEL_ID`, `DEVICE` |
| Color codes | lowercase | `purple`, `cyan`, `endc` |

---

## Type Hints

Use modern Python 3.9+ syntax:

```python
# Good
def process(items: list[int], name: str | None = None) -> dict[str, float]:

# Avoid
def process(items: List[int], name: Optional[str] = None) -> Dict[str, float]:
```

Type hints on function signatures. Return types included. Not required on every local variable.

---

## Functions

### Definitions

Split long function definitions across lines:

```python
def find_similar_sequences(
    model: TransformerBridge,
    dataset: Dataset,
    target_vector: Tensor,
    activation_name: str,
    k: int = 10,
    batch_size: int = 16,
) -> list[dict]:
```

Short definitions can stay on one line.

### Docstrings

Minimal. Only add for complex functions:

```python
def get_all_positions_distn(self, prompt: str, topk: int = 10) -> dict:
    """
    Get distributions for all token positions in the prompt.
    Returns dict with keys: tokens, token_ids, distributions
    """
```

Simple functions don't need docstrings—let the name and type hints speak.

---

## Classes

### Config Pattern

Use dataclasses for configuration:

```python
@dataclass
class TrainingConfig:
    batch_size: int = 32
    lr: float = 3e-4
    epochs: int = 10
    bf16: bool = True

    def asdict(self):
        return dataclasses.asdict(self)
```

### Inheritance

Use the short `super()` form:

```python
class GPT2(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()  # not super(GPT2, self).__init__()
        self.cfg = cfg
```

---

## Formatting

- **Line length:** No hard limit. Long lines are fine for tensor operations, plotting calls, etc.
- **Quotes:** Either single or double quotes acceptable
- **Trailing commas:** Use in multi-line structures

---

## Common Patterns

### Terminal Colors

From `mechtools.colors`: `purple`, `blue`, `cyan`, `yellow`, `green`, `red`, `gray`, `orange`, `brown`, `magenta`, `white`, `bold`, `underline`, `endc`. Don't redefine them in `utils.py`.

### Progress Bars

`pbar` is tqdm with the house style (colored description, `ncols=120`, `ascii=" >="`):

```python
for batch in pbar(dataloader, desc="Training"):
    ...

for i in pbar(range(100), desc="sweep", color=purple):
    ...
```

Put custom info on the left side of the bar via `set_description`.

### GPU Memory

`tec()` clears the CUDA cache. Call it after deleting large tensors or models.

### Seeding

`set_seed(seed)` seeds torch, numpy, and random together.

### Experiment Tracking

Use wandb:

```python
run_cfg = {"model": model.cfg.asdict(), "training": cfg.asdict()}
wandb.init(project="project-name", name="run-name", config=run_cfg)

# During training
wandb.log({"loss": loss.item(), "lr": lr})

wandb.finish()
```

### Visualization

Use the `mechtools.plots` wrappers, which take tensors directly and share plotly styling:

```python
imshow(attn_pattern, title="Attention Pattern")
line(losses, x=steps, title="Loss")
```

Fall back to `plotly.express` only for chart types the wrappers don't cover.

### Logging

Debug printing, not the logging module. Use liberally for sanity checks, progress updates, and status messages. Color-code with a coherent scheme:

| Color | Use |
|-------|-----|
| `gray` | Routine status (loading, saving, setup) |
| `green` | Success / completion |
| `cyan` | Key results, values, sanity check outputs |
| `yellow` | Warnings, unexpected-but-not-fatal info |
| `red` | Errors |
| `purple` | Section headers, experiment labels |

```python
print(f"{gray}Loading model...{endc}")
print(f"{green}Done. {cyan}{n_params/1e6:.1f}M params{endc}")
print(f"{purple}=== Running ablation sweep ==={endc}")
print(f"{cyan}Layer 5 head 3: logit diff = {diff:.4f}{endc}")
print(f"{yellow}Warning: batch size reduced to {new_bs} due to OOM{endc}")
```

### LLM Judges & Prompt Modifiers

When using an LLM to classify, score, or rewrite items in a dataset (e.g., "is this prompt about programming?", "rewrite this response in French"), follow this pattern:

**Prompt templates** as module-level format strings with `{placeholders}`. Keep them minimal—ask for constrained output ("Yes" or "No"), parse with a simple substring check.

**Async batch calls** using `aiohttp` + `asyncio.gather`. Each coroutine takes a shared `aiohttp.ClientSession`, the item index, and the item data. Returns `(idx, result | None)` — the index for write-back, `None` for failures:

```python
async def _classify_async(session: aiohttp.ClientSession, idx: int, prompt: str) -> tuple[int, bool | None]:
    payload = {"model": model_name, "messages": [{"role": "user", "content": make_prompt(prompt)}]}
    try:
        async with session.post(API_URL, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            resp.raise_for_status()
            result = await resp.json()
            return (idx, "yes" in result["choices"][0]["message"]["content"].strip().lower())
    except Exception as e:
        print(f"Error at idx {idx}: {e}")
        return (idx, None)
```

**Batched gather loop** with live tqdm stats. Process in batches (e.g., 128) for rate-limit-friendly concurrency. Update the progress bar description with running counts after each batch:

```python
async def _classify_dataset_async(dataset, batch_size=128):
    results = [None] * len(dataset)
    indices = [i for i in range(len(dataset)) if not already_done(i)]
    bar = pbar(total=len(indices), desc="true: 0 | false: 0")
    true_count = false_count = failed = 0

    async with aiohttp.ClientSession() as session:
        for batch_start in range(0, len(indices), batch_size):
            batch = indices[batch_start:batch_start + batch_size]
            tasks = [_classify_async(session, idx, dataset[idx]["prompt"]) for idx in batch]
            batch_results = await asyncio.gather(*tasks)
            for idx, val in batch_results:
                if val is None: failed += 1
                else:
                    results[idx] = val
                    if val: true_count += 1
                    else: false_count += 1
            bar.update(len(batch))
            bar.set_description(f"{cyan}true: {true_count} | false: {false_count}")
    bar.close()
    return results
```

**Sync/async bridge**: Public functions are sync, using `asyncio.run()` internally. Async implementation is private (`_` prefix).

**Key conventions:**
- **Resumable**: Skip items that already have results (`force=False` default). Lets interrupted runs pick up where they left off.
- **Dict columns for accumulation**: Store results in a dict column (e.g., `classifications["programming"] = True`) so multiple independent passes coexist without conflict.
- **Failures are counted, not raised**: Failed items stay unprocessed, get retried on the next run. Warn about failure counts at the end.
- **Config at the orchestrator level**: Classification names, model names, and guideline strings live in the top-level script, not buried in library code.

### Interactive Development

Use `#%%` cell markers for Jupyter-style execution. Autoreload is already on once `mechtools` is imported in a kernel; don't add `%load_ext autoreload` blocks.

```python
#%%
model = load_bridge("Qwen/Qwen3-0.6B")

#%%
results = run_experiment(model)
imshow(results)
```

---

## Main Entrypoint

```python
if __name__ == "__main__":
    set_seed(42)

    model_cfg = ModelConfig(d_model=512, n_layers=6)
    training_cfg = TrainingConfig(lr=3e-4, epochs=10)

    model = Model(model_cfg)
    train(model, training_cfg, dataset)
```

---

## Fail Loudly

In interpretability research, silent failures corrupt results. Prefer crashes over graceful degradation.

**Do:**
```python
# Assert assumptions explicitly
assert acts.shape[0] == len(tokens), f"Shape mismatch: {acts.shape[0]} vs {len(tokens)}"

# Let indexing errors surface
result = cache[layer_name]  # KeyError if missing = good, tells you something's wrong
```

**Don't:**
```python
# Silent fallbacks hide broken assumptions
result = cache.get(layer_name, None)
if result is None:
    continue  # Now you'll never know this failed

# Swallowing exceptions
try:
    process(batch)
except Exception:
    pass  # What broke? Who knows
```

**Why:** A crash tells you exactly where an assumption failed. A fallback gives you plausible-looking but potentially meaningless results. In research, the former is valuable signal; the latter is dangerous noise.

**Exception:** Graceful handling is fine for I/O, user-facing tools, or when you genuinely expect and want to handle a condition.

---

## Reporting Results

Delineate observations from conclusions, and place them in different sections of your output if both are present.

Results come first: what was measured, the criteria, the numbers. Brief bullets, one figure each, very minimal jargon.

Conclusions go in their own section afterward, and are optional. Unless I ask for interpretation, I'm expecting results, not a verdict.

---

# Technical Foundations

Background knowledge for mechanistic interpretability work.

---

## Transformer Architecture

### The Residual Stream

Transformers process a sequence of token embeddings through repeated layers. The **residual stream** is the running sum that each layer reads from and writes to:

```
x_0 = embed(tokens) + pos_embed
x_1 = x_0 + attn_0(x_0) + mlp_0(x_0)
x_2 = x_1 + attn_1(x_1) + mlp_1(x_1)
...
logits = unembed(ln_final(x_L))
```

Each attention head and MLP reads from the residual stream and adds its output back. This additive structure is why we can study components in isolation.

### Attention

Each attention head computes:

```
Q = x @ W_Q    # (batch, seq, d_head)
K = x @ W_K    # (batch, seq, d_head)
V = x @ W_V    # (batch, seq, d_head)

pattern = softmax(Q @ K.T / sqrt(d_head))  # (batch, seq, seq)
out = pattern @ V @ W_O                     # (batch, seq, d_model)
```

The attention pattern shows where each token "looks" to gather information. Causal models mask future positions so `pattern[i, j] = 0` when `j > i`.

**OV circuit:** What information gets moved (`W_V @ W_O`).
**QK circuit:** Which positions attend to which (`W_Q`, `W_K`).

### MLPs

MLPs are position-wise nonlinear transformations:

```
h = act_fn(x @ W_in + b_in)   # (batch, seq, d_mlp)
out = h @ W_out + b_out       # (batch, seq, d_model)
```

`d_mlp` is typically 4× `d_model`. The activation function is usually GELU or SiLU.

MLPs are thought to store factual associations and perform computation that attention can't.

### Layer Norm

Applied before attention and MLP (in pre-norm architectures):

```
x_normalized = (x - mean) / std * gamma + beta
```

LayerNorm complicates interpretability because it couples all dimensions. Often ignored or folded into weights for analysis.

---

## Common Dimensions

| Name | Meaning | Typical Values |
|------|---------|----------------|
| `batch` | Number of sequences | 1-64 |
| `seq` / `pos` | Sequence length | 128-2048 |
| `d_model` | Residual stream width | 512-4096 |
| `n_layers` | Number of transformer blocks | 6-48 |
| `n_heads` | Attention heads per layer | 8-32 |
| `d_head` | Dimension per head (`d_model // n_heads`) | 64-128 |
| `d_mlp` | MLP hidden dimension (`4 * d_model`) | 2048-16384 |
| `d_vocab` | Vocabulary size | 50k-100k |

---

## TransformerLens

Models are `TransformerBridge` objects (transformer-lens >= 3.8): a HuggingFace model wrapped with TransformerLens hook points, so it supports any architecture `transformers` does. There is no `HookedTransformer` path.

### Loading Models

```python
model = load_bridge("Qwen/Qwen3-0.6B")
model = load_bridge("google/gemma-3-1b-it")
```

`load_bridge` loads the HF model in bf16 with `device_map="auto"`, wraps it, sets eval mode, and turns grads off. If the repo is a peft adapter, it loads the base model (from the adapter config, or `parent_model_id`) and merges the adapter in memory:

```python
model = load_bridge("eekay/gemma-3-1b-it-lion-ft")
model = load_bridge("eekay/some-adapter", parent_model_id="google/gemma-3-1b-it")
```

For raw HF use (training, vLLM-matching captures), `load_hf_model` does the same loading and merging without the bridge.

### Running with Cache

```python
logits, cache = model.run_with_cache(tokens)

# cache is a dict-like object with activation tensors
resid = cache["resid_pre", 0]           # residual stream before layer 0
attn_out = cache["attn_out", 5]         # attention output at layer 5
pattern = cache["pattern", 3]           # attention patterns at layer 3
mlp_post = cache["post", 7]             # MLP activations after nonlinearity, layer 7
```

### Activation Names

```
hook_embed                      # token embeddings
hook_pos_embed                  # positional embeddings
blocks.{L}.hook_resid_pre       # residual stream input to layer L
blocks.{L}.hook_resid_post      # residual stream output of layer L
blocks.{L}.hook_resid_mid       # after attention, before MLP
blocks.{L}.attn.hook_q          # queries (batch, seq, n_heads, d_head)
blocks.{L}.attn.hook_k          # keys
blocks.{L}.attn.hook_v          # values
blocks.{L}.attn.hook_pattern    # attention patterns (batch, n_heads, seq, seq)
blocks.{L}.attn.hook_result     # attention head outputs before combining
blocks.{L}.hook_attn_out        # combined attention output
blocks.{L}.hook_mlp_out         # MLP output
blocks.{L}.mlp.hook_pre         # MLP input (after first linear)
blocks.{L}.mlp.hook_post        # MLP activations (after nonlinearity)
ln_final.hook_normalized        # final layer norm output
```

### Hooks for Intervention

`mechtools.hooks` covers the common cases: `add_bias_hook` / `make_add_bias_hook` for steering (with `scale`, `seq_pos`, `target_norm`), `replace_act_hook` for patching, `make_sae_feat_steer_hook` for SAE feature steering, `scale_hooks` / `set_hooks` for projecting onto directions per layer.

```python
model.add_hook("blocks.5.hook_resid_pre", make_add_bias_hook(steer_vec, scale=4.0, seq_pos=slice(-3, None)))
logits = model(tokens)
model.reset_hooks()

# or temporarily
logits = model.run_with_hooks(tokens, fwd_hooks=scale_hooks({8: dirs_8, 12: dirs_12}, factor=0.0))
```

Write a custom hook when none fit:

```python
def ablate_head(activation, hook, head_idx):
    activation[:, :, head_idx, :] = 0
    return activation

model.add_hook("blocks.5.attn.hook_result", partial(ablate_head, head_idx=3))
```

### Useful Methods

```python
model.to_tokens(text)                    # string -> token ids
model.to_str_tokens(text)                # string -> list of token strings
model.to_string(tokens)                  # token ids -> string
model.run_with_hooks(tokens, fwd_hooks)  # run with temporary hooks
model.tokenizer                          # the HF tokenizer, for mechtools.tokens helpers

# Direct attribute access
model.W_E                   # embedding matrix (d_vocab, d_model)
model.W_U                   # unembedding matrix (d_model, d_vocab)
model.W_Q                   # query weights, stacked over layers
model.W_K                   # key weights
model.W_V                   # value weights
model.W_O                   # output weights
model.W_in                  # MLP input weights
model.W_out                 # MLP output weights
model.cfg.n_layers
```

### Chat Models

Use the `mechtools.tokens` helpers rather than the tokenizer's template methods directly. `apply_chat_template(model.tokenizer, convs)` returns left-padded ids and an attention mask for a batch of conversations or prompt strings. `get_assistant_mask` adds a mask over assistant tokens, and `completion_loss` computes loss over them. `show_toks(conv, model.tokenizer)` renders the tokenized conversation as a hoverable strip for checking template boundaries.

### Local Sampling

`stream_toks(model, toks)` prints tokens as they are generated. `sample_batch(model, prompt_toks, n)` and `sample_rolling` (keeps a fixed batch in flight, refilling finished slots) return lists of completions.

---

## Common Models

| Model | `d_model` | `n_layers` | `n_heads` | `d_mlp` | Notes |
|-------|-----------|------------|-----------|---------|-------|
| `gpt2-small` | 768 | 12 | 12 | 3072 | Classic, well-studied |
| `gpt2-medium` | 1024 | 24 | 16 | 4096 | |
| `gpt2-large` | 1280 | 36 | 20 | 5120 | |
| `pythia-70m` | 512 | 6 | 8 | 2048 | Small, fast iteration |
| `pythia-160m` | 768 | 12 | 12 | 3072 | |
| `pythia-410m` | 1024 | 24 | 16 | 4096 | |
| `gemma-2b` | 2048 | 18 | 8 | 16384 | Newer architecture |

---

## Intervention Techniques

### Activation Patching

Replace activations from a "corrupted" run with activations from a "clean" run to measure causal importance:

```python
clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_tokens)

# Patch residual stream at position 5, layer 3
hook_name = "blocks.3.hook_resid_pre"
model.add_hook(hook_name, partial(replace_act_hook, new=clean_cache[hook_name], seq_pos=5))
patched_logits = model(corrupt_tokens)
model.reset_hooks()
```

If patching recovers the clean behavior, that activation is causally important.

### Ablation Types

| Type | Method | Use Case |
|------|--------|----------|
| **Zero ablation** | Set to 0 | Simple, but changes activation distribution |
| **Mean ablation** | Set to mean over dataset | More realistic baseline |
| **Resample ablation** | Replace with value from different input | Preserves distribution |
| **Noising** | Add Gaussian noise | Gradual degradation |

### Path Patching

Patch along specific computational paths (e.g., "attention head 3.2 → MLP 5") to isolate circuits.

---

## Sparse Autoencoders (SAEs)

### Why SAEs?

Models represent more features than they have dimensions (**superposition**). An SAE learns a sparse overcomplete basis to disentangle these features.

### Architecture

```python
# Encoder: d_model -> d_sae (expansion, typically 4-64x)
h = activation @ W_enc + b_enc    # (batch, seq, d_sae)
f = relu(h)                       # sparse activations

# Decoder: d_sae -> d_model (reconstruction)
x_hat = f @ W_dec + b_dec         # (batch, seq, d_model)
```

### Training Objective

```
L = ||x - x_hat||^2 + λ * ||f||_1
    └─────────────┘   └─────────┘
     reconstruction    sparsity
```

### Using Pretrained SAEs

```python
from sae_lens import SAE

sae, cfg, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
)

# Get feature activations
acts = cache["resid_pre", 8]
feature_acts = sae.encode(acts)  # (batch, seq, d_sae)

# Reconstruct
reconstructed = sae.decode(feature_acts)
```

Steer on a feature with `make_sae_feat_steer_hook(sae, feat_idx, feat_act)` from `mechtools.hooks`. The load/save/summary helpers are not in mechtools; see the mechtools section for which project to copy from.

### Interpreting Features

Each column of `W_dec` is a **feature direction** in residual stream space. Features are interpreted by:
- Finding inputs that maximally activate them
- Analyzing the decoder direction's effect on logits
- Looking at co-occurring features

---

## Einops Patterns

Common reshaping operations:

```python
from einops import rearrange, reduce, repeat, einsum

# Attention: split heads
q = rearrange(q, "batch seq (heads d_head) -> batch heads seq d_head", heads=n_heads)

# Attention: merge heads back
out = rearrange(out, "batch heads seq d_head -> batch seq (heads d_head)")

# Mean over sequence
pooled = reduce(acts, "batch seq d_model -> batch d_model", "mean")

# Broadcast for patching
patch = repeat(vec, "d_model -> batch seq d_model", batch=b, seq=s)

# Attention scores
scores = einsum(q, k, "batch heads seq_q d_head, batch heads seq_k d_head -> batch heads seq_q seq_k")
```

---

## Logit Lens & Friends

### Logit Lens

Project intermediate residual stream through the unembedding to see what the model "believes" at each layer:

```python
resid = cache["resid_post", layer]
logits = resid @ model.W_U  # (batch, seq, d_vocab)
top_toks_table(logits[0, -1], model.tokenizer, k=10)
```

### Tuned Lens, J-lens, Template Lens

Learned per-layer probes, more accurate than the raw logit lens. `mechtools.lens` loads j-lens and template-lens weights (`load_jlens`, `load_tlens`) and renders per-layer readouts from a `run_with_cache` cache: `jlens_readout` / `tlens_readout` for top tokens at one position, and the `*_cluster_readout` variants for a tabbed view over layers and positions grouped by vocab cluster.

### Logit Attribution

Decompose the final logits by contribution from each component:

```python
# Each component's contribution to logit difference
head_contribution = cache["result", layer][:, :, head] @ model.W_U[:, target_token]
```

---

## Language Model Providers

Frequently during research it will be necessary to use APIs to access closed models. LLM Judging is the most common use case. If LM API access is required, prefer to use Openrouter when possible. Always pin the provider to a single option, as different providers can differ drastically in terms of the model they serve under the same name.

---

## Summary

- Keep it simple and readable
- Low line count is good
- Favor clarity over "production" patterns
- Type hints yes, verbose docstrings no
- Dataclasses for configs
- `torch as t`, modern type syntax, flat structure
- `from mechtools import *` first; don't reimplement what it has
