"""
Shared pieces for the Inspect environments in this directory: the config files under envs/configs, the condition string every task takes and
the request settings it maps to, and the vector files. Inspect puts a task file's directory on sys.path, so the tasks import this as `common`,
and scripts run from the repo root as `uv run python envs/<script>.py` get the same path.

Conditions:
    none                            plain request
    lora:<name>                     the request's model field set to <name>: an adapter the server has loaded under that name
    add:<vector>:<layer>:<alpha>    a vllm-lens steering vector, alpha x the vector's row for <layer>, added at that layer's output at every position
    ablate:<vector>[:<layer>]       a vllm-lens hook at every layer in the vector's sidecar projecting out that layer's unit direction (or the one
                                    layer's direction at every layer). Registered on the server as a persistent hook before the run, not per request:

    uv run python envs/common.py http://localhost:8000/v1 ablate:grader_parity_very_hacker_vs_generic_prompt Qwen3.6-27B    # register (clears any previous hooks first)
    uv run python envs/common.py http://localhost:8000/v1 none                                                                        # clear

Vectors are utils.save_vector's data/vectors/<model>/<name>.safetensors, key "v" [n_layers, d_model], row i at layer layers[i] of the json sidecar.

Importing this module makes Inspect's vllm provider send tool schemas as the agent-interp-envs harness did, without the strict flag and the
additionalProperties it adds for OpenAI, so the server renders the same system prompt as that harness. It is a patch on the provider class
because the CLI resolves --model before it loads the task file.
"""
import base64
import json
import sys
from pathlib import Path

import cloudpickle
import httpx
import torch as t
from inspect_ai.model import GenerateConfig
from inspect_ai.model._openai import openai_chat_tools
from inspect_ai.model._providers.vllm import VLLMAPI
from inspect_ai.tool import ToolInfo
from omegaconf import OmegaConf
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = Path(__file__).resolve().parent / "configs"
VECTORS = ROOT / "data" / "vectors"

OmegaConf.register_new_resolver("pct", lambda x: int(float(x) * 100))
OmegaConf.register_new_resolver("pct_complement", lambda x: int((1 - float(x)) * 100))


def plain_tools(self: VLLMAPI, tools: list[ToolInfo]) -> list[dict]:
    return openai_chat_tools(tools, exclude={"additionalProperties"})


VLLMAPI.tools_to_openai = plain_tools


def load_config(env: str, name: str) -> dict:
    return OmegaConf.to_container(OmegaConf.load(CONFIGS / env / f"{name}.yaml"), resolve=True)


def load_vector(vectors: Path, name: str) -> tuple[t.Tensor, list[int]]:
    with safe_open(vectors / f"{name}.safetensors", "pt") as h:
        v = h.get_tensor("v")
    return v, json.loads((vectors / f"{name}.json").read_text())["layers"]


def check_condition(condition: str, vectors: Path) -> None:
    """Assert that a condition follows the grammar above and that the vector and layer it names exist."""
    kind, *args = condition.split(":")
    n_fields = {"none": [0], "lora": [1], "add": [3], "ablate": [1, 2]}
    assert kind in n_fields, f"{condition}: the kind is one of {list(n_fields)}"
    assert len(args) in n_fields[kind] and all(args), f"{condition}: {kind} takes {n_fields[kind]} non-empty fields after the kind"
    if kind in ("add", "ablate"):
        assert (vectors / f"{args[0]}.safetensors").exists(), f"{condition}: no vector {args[0]} in {vectors}, which has {sorted(path.stem for path in vectors.glob('*.safetensors'))}"
        layers = json.loads((vectors / f"{args[0]}.json").read_text())["layers"]
        assert not args[1:] or int(args[1]) in layers, f"{condition}: the vector has no row for layer {args[1]}, only for {layers}"
    if kind == "add":
        assert float(args[2]) > 0, f"{condition}: alpha is positive (an add induces hacking; the suppression is ablate, not a negative add)"


def tensor_json(v: t.Tensor) -> dict:
    """vllm-lens's wire format for a tensor, the uncompressed float32 variant."""
    arr = v.detach().float().cpu().numpy()
    return {"data": base64.b64encode(arr.tobytes()).decode(), "dtype": "float32", "original_dtype": "torch.float32", "shape": list(arr.shape)}


def request_body(condition: str, vectors: Path) -> dict:
    """The request fields a condition adds. ablate adds none: its hook lives on the server (set_hooks)."""
    kind, *args = condition.split(":")
    if kind in ("none", "ablate"):
        return {}
    if kind == "lora":
        return {"model": args[0]}
    assert kind == "add", condition
    V, layers = load_vector(vectors, args[0])
    layer, alpha = int(args[1]), float(args[2])
    vector = {"activations": tensor_json(alpha * V[layers.index(layer)][None]), "layer_indices": [layer], "scale": 1.0, "norm_match": False}
    return {"vllm_xargs": {"apply_steering_vectors": json.dumps([vector])}}


class ServerFunction:
    """Pickles as eval(source, namespace): the server compiles the function with its own Python, where cloudpickle would ship this interpreter's
    bytecode (unreadable by another minor version) and this torch's tensors. The namespace holds plain bytes and the torch module by name."""

    def __init__(self, source: str, namespace: dict):
        self.source, self.namespace = source, namespace

    def __reduce__(self):
        return (eval, (self.source, self.namespace))


# h <- h - (h . u) u with the layer's unit row built on the device once, per worker, from its float32 bytes
PROJECT_SRC = "lambda ctx, h: (lambda u: (h.float() - (h.float() @ u)[:, None] * u).to(h.dtype))(cache[ctx.layer_idx] if ctx.layer_idx in cache else cache.setdefault(ctx.layer_idx, torch.frombuffer(bytearray(rows[ctx.layer_idx]), dtype=torch.float32).to(h.device)))"


def projection_hook(condition: str, vectors: Path) -> dict:
    """A vllm-lens Hook for an ablate condition: the projection at each listed layer's output, in the register endpoint's format."""
    kind, name, *layer = condition.split(":")
    assert kind == "ablate", condition
    V, layers = load_vector(vectors, name)
    U = V / V.norm(dim=-1, keepdim=True)
    if layer:
        U = U[layers.index(int(layer[0]))].expand_as(U)
    rows = {l: U[i].float().numpy().tobytes() for i, l in enumerate(layers)}
    project = ServerFunction(PROJECT_SRC, {"rows": rows, "cache": {}, "torch": t})
    return {"fn": {"cloudpickle": base64.b64encode(cloudpickle.dumps(project)).decode()}, "layer_indices": layers, "pre": False}


def set_hooks(base_url: str, condition: str, vectors: Path) -> dict:
    """Make the server's persistent hooks exactly what the condition needs: the projection for ablate, none otherwise."""
    r = httpx.post(f"{base_url}/hooks/clear", timeout=120)
    if condition.startswith("ablate:"):
        r = httpx.post(f"{base_url}/hooks/register", json={"hooks": [projection_hook(condition, vectors)]}, timeout=600)
    r.raise_for_status()
    return r.json()


def generate_config(cfg: dict, condition: str, vectors: Path) -> GenerateConfig:
    """The task's GenerateConfig: the config's model block with the condition's request fields merged into extra_body. The condition is also
    the request's cache_salt, so the server's prefix cache never serves blocks computed under another condition."""
    check_condition(condition, vectors)
    model = dict(cfg["model"])
    extra_body = model.pop("extra_body") | request_body(condition, vectors) | {"cache_salt": condition}
    return GenerateConfig(**model, extra_body=extra_body)


if __name__ == "__main__":
    vectors = VECTORS / sys.argv[3] if len(sys.argv) > 3 else VECTORS
    print(set_hooks(sys.argv[1], sys.argv[2], vectors))
