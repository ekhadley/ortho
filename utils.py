"""Helpers for grader_lens.py. Every function is a pure function of the model and its inputs and returns tensors or lists: prompt rendering and residual capture,
prompt-difference directions, steering and projection hooks, per-rollout CoT means, token search, and a lens readout of a vector. The exceptions are save_vector /
load_vector, the vector file format common.py also reads."""
import itertools
import json
from pathlib import Path

import torch as t
from safetensors import safe_open
from safetensors.torch import save_file

from mechtools import *


def render(tokenizer, prompt: str, sys_prompt: str | None = None, enable_thinking: bool = False) -> list[int]:
    """The chat-templated ids of a one-turn conversation, optional system prompt, generation prompt on."""
    conv = ([{"role": "system", "content": sys_prompt}] if sys_prompt else []) + [{"role": "user", "content": prompt}]
    return to_ids(conv, tokenizer, add_generation_prompt=True, enable_thinking=enable_thinking)


def resid(model, ids: list[int]) -> Tensor:
    """resid_pre at every layer and position in float32, [n_layers, seq, d_model]."""
    _, cache = model.run_with_cache(t.tensor([ids], device=model.device), names_filter=lambda name: name.endswith("hook_resid_pre"))
    return t.stack([cache[f"blocks.{layer}.hook_resid_pre"][0].float() for layer in range(model.cfg.n_layers)])


def resid_post(model, ids: list[int]) -> Tensor:
    """resid_post at every layer and position in float32, [n_layers, seq, d_model]: the layout of a saved vector (row i at resid_post.i)."""
    _, cache = model.run_with_cache(t.tensor([ids], device=model.device), names_filter=lambda name: name.endswith("hook_resid_post"))
    return t.stack([cache[f"blocks.{layer}.hook_resid_post"][0].float() for layer in range(model.cfg.n_layers)])


def last_resid(model, prompt: str) -> Tensor:
    """resid_pre at every layer at the prompt's last position, rendered with no system prompt and reasoning off, so the model is about to answer. [n_layers, d_model]"""
    return resid(model, render(model.tokenizer, prompt))[:, -1]


def reject(v: Tensor, controls: list[Tensor]) -> Tensor:
    """v with its component in the span of the controls removed, per layer, scaled to unit norm. All [n_layers, d_model]."""
    Q, _ = t.linalg.qr(t.stack(controls, dim=-1))  # orthonormal columns, [n_layers, d_model, k]
    r = v - t.einsum("ldk,lk->ld", Q, t.einsum("ldk,ld->lk", Q, v))
    return r / r.norm(dim=-1, keepdim=True)


def resid_pre_layout(v: Tensor) -> Tensor:
    """A saved [n_layers, d_model] vector (row i at resid_post.i) shifted so that row L is the vector for resid_pre of layer L. Row 0 is undefined (nan)."""
    return t.cat([t.full_like(v[:1], float("nan")), v[:-1]])


def make_proj_out_hook(u: Tensor):
    """A forward hook removing the activation's component along the unit direction u at every position: h <- h - (h . u) u."""
    def hook(act: Tensor, hook) -> Tensor:
        return act - (act @ u)[..., None] * u
    return hook


def add_hooks(model, direction: Tensor, layers: list[int], scale: float) -> list:
    """Add scale x direction[layer] to resid_pre at every position of each layer. direction is [n_layers, d_model] with row L for resid_pre of layer L (resid_pre_layout of a saved vector)."""
    return [(f"blocks.{layer}.hook_resid_pre", make_add_bias_hook(direction[layer].to(model.device, model.W_U.dtype), scale=scale)) for layer in layers]


def proj_hooks(model, direction: Tensor, layers: list[int]) -> list:
    """Project direction[layer] out of resid_pre at every position of each layer. Same layout as add_hooks; the row is normalized here."""
    return [(f"blocks.{layer}.hook_resid_pre", make_proj_out_hook((direction[layer] / direction[layer].norm()).to(model.device, model.W_U.dtype))) for layer in layers]


def find(ids: list[int], needle: str, tokenizer) -> list[int]:
    """Index of the token holding the first character of each occurrence of needle in the decoded tokens."""
    strs = to_str_toks(ids, tokenizer)
    text, ends = "".join(strs), list(itertools.accumulate(map(len, strs)))
    return [next(k for k, e in enumerate(ends) if e > c) for c in range(len(text)) if text.startswith(needle, c)]


def cot_span(ids: list[int], tokenizer) -> tuple[int, int]:
    """[start, end) token range of the reasoning: after the first <think> token, up to the last </think>."""
    return find(ids, "<think>", tokenizer)[0] + 1, find(ids, "</think>", tokenizer)[-1]


def strip_system_turn(ids: list[int], tokenizer) -> list[int]:
    """The ids from the user turn on. On the Qwen3 template that is exactly the rendering without a system prompt (checked against a re-render), and the identity when there is none."""
    return ids[find(ids, "<|im_start|>user", tokenizer)[0]:]


def resid_post_mean(model, ids: list[int], start: int, end: int) -> Tensor:
    """resid_post at every layer averaged over positions start:end, [n_layers, d_model] float32."""
    _, cache = model.run_with_cache(t.tensor([ids], device=model.device), names_filter=lambda name: name.endswith("hook_resid_post"))
    return t.stack([cache[f"blocks.{layer}.hook_resid_post"][0, start:end].float().mean(0) for layer in range(model.cfg.n_layers)])


def cot_means(model, ids_list: list[list[int]]) -> Tensor:
    """One row per sequence: resid_post at every layer averaged over its reasoning tokens (between <think> and </think>), on the cpu. [n, n_layers, d_model] float32."""
    rows = []
    for ids in pbar(ids_list, desc="CoT means"):
        start, end = cot_span(ids, model.tokenizer)
        rows.append(resid_post_mean(model, ids, start, end).cpu())
    return t.stack(rows)


def save_vector(vectors: Path, name: str, v: Tensor, meta: dict) -> None:
    """The vector file format, which common.py also reads: <name>.safetensors with key "v" [n_layers, d_model] float32, row i at resid_post.i, and
    <name>.json with the layer list, the per-layer gap (row norm) and meta (class definitions, sizes, position). A "layers" key in meta overrides the full range."""
    assert t.isfinite(v).all(), name
    vectors.mkdir(parents=True, exist_ok=True)
    save_file({"v": v.float().contiguous().cpu()}, vectors / f"{name}.safetensors")
    (vectors / f"{name}.json").write_text(json.dumps({"contrast": name, "layers": list(range(len(v))), "gap": v.norm(dim=-1).tolist(), **meta}, indent=1))


def load_vector(vectors: Path, name: str) -> tuple[Tensor, list[int]]:
    with safe_open(vectors / f"{name}.safetensors", "pt") as h:
        v = h.get_tensor("v")
    return v, json.loads((vectors / f"{name}.json").read_text())["layers"]


def lens_readout(model, jlens, labels, layers: list[int], v: Tensor, title: str) -> dict:
    """Cluster readout of a [n_layers, d_model] vector through the lens, one tab per layer."""
    scores = {f"L{layer}": get_lens_logits(v[layer].to(model.device, model.W_U.dtype), layer, model, jlens) for layer in layers}
    return cluster_readout(scores, labels, model.tokenizer.decode, title=title)
