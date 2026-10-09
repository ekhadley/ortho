"""
A run and what its intervention sends: EvalCfg, the env's task from the rhenvs package (github.com/ekhadley/rh-envs) built from the cfg's
task fields, the request fields and server hooks of the intervention, and the vector files.

An EvalCfg is one run: an env under a named yaml config of rhenvs, the run's size and task fields, and the intervention, which is any of a lora adapter (the request's
model field), an add (vllm-lens steering vectors in every request: each listed layer gets alpha x its own row of the vector added to its output
at every position, under one of three scalings) and an ablation (a persistent hook on the server projecting each listed layer's row out of its
output, registered before the run and cleared after it by run.py). The named instances live in run.py and a run is `./run.py <name> ...`.

Vectors are utils.save_vector's data/vectors/<model>/<name>.safetensors, key "v" [n_layers, d_model], row i at layer layers[i] of the json sidecar.

    uv run python common.py http://localhost:8000/v1    # clear the server's hooks by hand
"""
import base64
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, fields
from importlib.metadata import distribution
from inspect import signature
from pathlib import Path

import cloudpickle
import httpx
import torch as t
from inspect_ai import Task
from rhenvs.grader import grader
from rhenvs.impossible_bench import lcb
from rhenvs.mbpp import mbpp
from rhenvs.secret_number import secret_number
from rhenvs.terminal_verifier import terminal_verifier
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
VECTORS = ROOT / "data" / "vectors"
TASKS = {"grader": grader, "secret_number": secret_number, "impossible_bench": lcb, "terminal_verifier": terminal_verifier, "mbpp": mbpp}
TASK_FIELDS = ("config", "n", "family", "system", "prompts", "seed", "mode", "max_turns", "epochs")  # the EvalCfg fields that are task arguments
ENVS_COMMIT = json.loads(distribution("rhenvs").read_text("direct_url.json"))["vcs_info"]["commit_id"]  # the rhenvs commit uv installed from git
SCALINGS = ("row", "unit", "resid_norm")
INTERVENTION = ("lora", "add_vector", "add_layers", "add_alpha", "add_scaling", "ablate_vector", "ablate_layers", "ablate_row", "ablate_scale", "vectors")


@dataclass
class EvalCfg:
    env: str  # a key of TASKS
    config: str  # rhenvs configs/<env>/<config>.yaml: the prompts, the scaffold settings and the model block sent with every request
    n: int | None = None  # grader: samples per side per family. secret_number: games. terminal_verifier: grids. impossible_bench: problems and mbpp: tasks, None for all of them
    family: str | None = None  # grader: a family under rhenvs configs/grader/families, or all
    system: str | None = None  # grader: none, generic or very_hacker
    prompts: str | None = None  # grader: hack (the request with the other side's grader), clean (the side's own) or none (the request alone)
    seed: int | None = None  # secret_number: draws the secrets. terminal_verifier: draws the grids and the verifier offsets
    mode: str | None = None  # terminal_verifier: what verifier.py holds: possible, corrupted, corrupted_negative or verifier_bug
    max_turns: int | None = None  # terminal_verifier: commands per episode
    epochs: int = 1  # terminal_verifier and mbpp: samples per task
    lora: str | None = None  # an adapter the server has loaded under that name
    add_vector: str | None = None  # a vector under data/vectors/<vectors>: each layer of add_layers gets its own row added to its output at every position
    add_layers: list[int] | None = None
    add_alpha: float | list[float] | None = None  # one alpha, or one per layer of add_layers. Positive induces hacking; negative is negative steering, the other suppression besides the ablation
    add_scaling: str = "row"  # row: alpha x the row. unit: alpha x the unit row. resid_norm: alpha x the token's own residual norm x the unit row (vllm-lens norm_match)
    ablate_vector: str | None = None  # a vector whose rows are projected out of the outputs of ablate_layers, a hook on the server for the whole run
    ablate_layers: list[int] | None = None
    ablate_row: int | None = None  # that one layer's row projected out at every layer of ablate_layers; None for each layer's own row
    ablate_scale: float = 1.0  # h - ablate_scale x (h . u) u: 1 projects the row out, 2 reflects the component, more is negative steering proportional to the component
    vectors: str = "Qwen3.6-27B"  # data/vectors/<vectors>
    model: str = "vllm/Qwen3.6-27B"  # the model as the server names it
    base_url: str = "http://localhost:8000/v1"  # the server, through the tunnel (ssh -N -f -L 8000:localhost:8000 vast)
    max_connections: int | None = 64  # concurrent requests, None for Inspect's default
    display: str = "rich"  # Inspect's display: rich is a progress panel, full the full-screen TUI
    name: str | None = None  # set by run.py from the instance's variable name


def load_vector(vectors: Path, name: str) -> tuple[t.Tensor, list[int]]:
    assert (vectors / f"{name}.safetensors").exists(), f"no vector {name} in {vectors}, which has {sorted(path.stem for path in vectors.glob('*.safetensors'))}"
    with safe_open(vectors / f"{name}.safetensors", "pt") as h:
        v = h.get_tensor("v")
    return v, json.loads((vectors / f"{name}.json").read_text())["layers"]


def condition(cfg: EvalCfg) -> str:
    """The run's label in the records and the rate tables: none when nothing is intervened on, else the instance's name."""
    return "none" if cfg.lora is None and cfg.add_vector is None and cfg.ablate_vector is None else cfg.name


def alphas(cfg: EvalCfg) -> list[float]:
    return cfg.add_alpha if isinstance(cfg.add_alpha, list) else [cfg.add_alpha] * len(cfg.add_layers)


def check(cfg: EvalCfg) -> None:
    """Assert that the cfg sets no task field its env's task does not take, and that the intervention fields fit together and the vector
    files they name exist. The task checks its own arguments when it is built (a required one left None is a missing argument)."""
    assert cfg.env in TASKS, f"{cfg.name}: env {cfg.env} is one of {list(TASKS)}"
    params = signature(TASKS[cfg.env]).parameters
    stray = [f.name for f in fields(cfg) if f.name in TASK_FIELDS and f.name not in params and getattr(cfg, f.name) != f.default]
    assert not stray, f"{cfg.name}: {cfg.env} takes no {stray}, its arguments are {list(params)}"
    assert cfg.add_scaling in SCALINGS, f"{cfg.name}: add_scaling {cfg.add_scaling} is one of {SCALINGS}"
    assert (cfg.add_vector is None) == (cfg.add_layers is None) == (cfg.add_alpha is None), f"{cfg.name}: add_vector, add_layers and add_alpha are set together"
    if cfg.add_vector is not None:
        _, layers = load_vector(VECTORS / cfg.vectors, cfg.add_vector)
        assert set(cfg.add_layers) <= set(layers), f"{cfg.name}: {cfg.add_vector} has rows for layers {layers[0]} to {layers[-1]}, not {sorted(set(cfg.add_layers) - set(layers))}"
        assert len(alphas(cfg)) == len(cfg.add_layers), f"{cfg.name}: add_alpha lists {len(cfg.add_alpha)} alphas for {len(cfg.add_layers)} layers"
        assert all(a != 0 for a in alphas(cfg)), f"{cfg.name}: every alpha is nonzero, not {cfg.add_alpha}"
    assert (cfg.ablate_vector is None) == (cfg.ablate_layers is None), f"{cfg.name}: ablate_vector and ablate_layers are set together"
    assert cfg.ablate_vector is not None or (cfg.ablate_row is None and cfg.ablate_scale == 1.0), f"{cfg.name}: ablate_row or ablate_scale without ablate_vector"
    if cfg.ablate_vector is not None:
        _, layers = load_vector(VECTORS / cfg.vectors, cfg.ablate_vector)
        assert set(cfg.ablate_layers) <= set(layers), f"{cfg.name}: {cfg.ablate_vector} has rows for layers {layers[0]} to {layers[-1]}, not {sorted(set(cfg.ablate_layers) - set(layers))}"
        assert cfg.ablate_row is None or cfg.ablate_row in layers, f"{cfg.name}: {cfg.ablate_vector} has no row for layer {cfg.ablate_row}"
    assert cfg.model.startswith("vllm/"), f"{cfg.name}: interventions and token ids need the vLLM server, so model is vllm/<served name>, not {cfg.model}"
    assert cfg.base_url.endswith("/v1"), f"{cfg.name}: base_url ends in /v1, not {cfg.base_url}"


def tensor_json(v: t.Tensor) -> dict:
    """vllm-lens's wire format for a tensor, the uncompressed float32 variant."""
    arr = v.detach().float().cpu().numpy()
    return {"data": base64.b64encode(arr.tobytes()).decode(), "dtype": "float32", "original_dtype": "torch.float32", "shape": list(arr.shape)}


def steering_vectors(cfg: EvalCfg) -> list[dict]:
    """vllm-lens steering vectors for the add fields, one per layer: the layer's row (the unit row under unit and resid_norm) at scale alpha, norm matched under resid_norm."""
    V, layers = load_vector(VECTORS / cfg.vectors, cfg.add_vector)
    vectors = []
    for layer, alpha in zip(cfg.add_layers, alphas(cfg)):
        row = V[layers.index(layer)]
        row = row if cfg.add_scaling == "row" else row / row.norm()
        vectors.append({"activations": tensor_json(row[None]), "layer_indices": [layer], "scale": alpha, "norm_match": cfg.add_scaling == "resid_norm"})
    return vectors


def request_body(cfg: EvalCfg) -> dict:
    """The request fields the intervention adds. The ablation's hook lives on the server (set_hooks), so it adds only read_prefix_cache, which
    lets vllm-lens serve a hooked request from the prefix cache: the cache_salt keys the blocks by intervention, so a hit was computed under this hook."""
    body = {}
    if cfg.lora is not None:
        body["model"] = cfg.lora
    if cfg.add_vector is not None or cfg.ablate_vector is not None:
        body["vllm_xargs"] = {"read_prefix_cache": 1}
    if cfg.add_vector is not None:
        body["vllm_xargs"]["apply_steering_vectors"] = json.dumps(steering_vectors(cfg))
    return body


class ServerFunction:
    """Pickles as eval(source, namespace): the server compiles the function with its own Python, where cloudpickle would ship this interpreter's
    bytecode (unreadable by another minor version) and this torch's tensors. The namespace holds plain bytes and the torch module by name."""

    def __init__(self, source: str, namespace: dict):
        self.source, self.namespace = source, namespace

    def __reduce__(self):
        return (eval, (self.source, self.namespace))


# h <- h - scale (h . u) u with the layer's unit row built on the device once, per worker, from its float32 bytes
PROJECT_SRC = "lambda ctx, h: (lambda u: (h.float() - scale * (h.float() @ u)[:, None] * u).to(h.dtype))(cache[ctx.layer_idx] if ctx.layer_idx in cache else cache.setdefault(ctx.layer_idx, torch.frombuffer(bytearray(rows[ctx.layer_idx]), dtype=torch.float32).to(h.device)))"


def projection_hook(cfg: EvalCfg) -> dict:
    """A vllm-lens Hook for the ablate fields: the projection at each listed layer's output, in the register endpoint's format."""
    V, layers = load_vector(VECTORS / cfg.vectors, cfg.ablate_vector)
    U = V / V.norm(dim=-1, keepdim=True)
    rows = {layer: U[layers.index(layer if cfg.ablate_row is None else cfg.ablate_row)].float().numpy().tobytes() for layer in cfg.ablate_layers}
    project = ServerFunction(PROJECT_SRC, {"rows": rows, "cache": {}, "torch": t, "scale": cfg.ablate_scale})
    return {"fn": {"cloudpickle": base64.b64encode(cloudpickle.dumps(project)).decode()}, "layer_indices": cfg.ablate_layers, "pre": False}


def clear_hooks(base_url: str) -> dict:
    r = httpx.post(f"{base_url}/hooks/clear", timeout=120)
    r.raise_for_status()
    return r.json()


def set_hooks(base_url: str, cfg: EvalCfg) -> dict:
    """Make the server's persistent hooks exactly what the run needs: cleared, then the projection registered when the run ablates."""
    cleared = clear_hooks(base_url)
    if cfg.ablate_vector is None:
        return cleared
    r = httpx.post(f"{base_url}/hooks/register", json={"hooks": [projection_hook(cfg)]}, timeout=600)
    r.raise_for_status()
    return r.json()


def build_task(cfg: EvalCfg) -> Task:
    """The env's rhenvs task, called with the task fields the cfg sets, its config's extra_body with the intervention's request fields merged
    in, and its metadata (env, config_id, its arguments) with what every log and record of ours also carries: the condition label, the vectors
    dir, the whole EvalCfg and the rhenvs commit. The request's cache_salt is a hash of the intervention fields (vLLM caps the salt at 128
    characters), so the server's prefix cache never serves blocks computed under another intervention."""
    check(cfg)
    params = signature(TASKS[cfg.env]).parameters
    task = TASKS[cfg.env](**{f: getattr(cfg, f) for f in TASK_FIELDS if f in params and getattr(cfg, f) is not None})
    salt = hashlib.sha256(json.dumps([getattr(cfg, field) for field in INTERVENTION]).encode()).hexdigest()
    task.config.extra_body |= request_body(cfg) | {"cache_salt": salt}
    task.metadata |= {"condition": condition(cfg), "vectors": cfg.vectors, "cfg": asdict(cfg), "envs_commit": ENVS_COMMIT}
    return task


if __name__ == "__main__":
    print(clear_hooks(sys.argv[1]))
