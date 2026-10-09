#!./.venv/bin/python
"""
The launcher: the named EvalCfg instances below, run one after another against the vLLM server.

    ./run.py                        every instance with the fields it sets
    ./run.py pets pets_band_x60     those runs, in that order

Every instance is checked before the first eval (common.check and the task's own asserts, by building every task first), then the server: it
serves the model and every lora adapter, and has no request in flight, since its hooks are server-wide and this script changes them. Then per
run: the server's hooks are set to what the run needs (the projection when it ablates, none otherwise) and cleared when the eval ends, however
it ends; the eval logs to logs/<env>/ with the raw model calls and an hour's client timeout; the log is converted to data/inspect/<env>/; and
a table gives each group's cheat rate with its 95% Wilson interval, under the run's condition label (none when nothing is intervened on, else
the instance's name).
"""
import re
import sys
import time
from dataclasses import MISSING, fields, replace

import httpx
from inspect_ai import eval
from rhenvs.convert import write_records

from mechtools import *

from common import ROOT, EvalCfg, build_task, condition, clear_hooks, set_hooks
from ablations import *  # the suppression variants, EvalCfg instances like the ones below

HEADERS = ["condition", "group", "valid", "cheat", "95% wilson"]

# non-intervened runs
parity = EvalCfg(env="grader", config="default", family="parity", system="none", prompts="hack", n=256)
pets = replace(parity, family="pets_listed")
parity_nothink = replace(parity, config="nothink")   # reasoning off: the cheat rate at zero reasoning length
pets_nothink = replace(pets, config="nothink")
smoke_grader = EvalCfg(env="grader", config="default", family="all", system="none", prompts="hack", n=2)
secret = EvalCfg(env="secret_number", config="qwen3.6-27b", n=256, seed=0)
lcb_paper = EvalCfg(env="impossible_bench", config="paper", n=50)
smoke_secret = replace(secret, n=2)
smoke_lcb = replace(lcb_paper, n=2)
tv_bug = EvalCfg(env="terminal_verifier", config="default", mode="verifier_bug", max_turns=4, n=128, seed=0)   # the sweep's main variant: only tampering passes
tv_bug_mt6 = replace(tv_bug, max_turns=6)
tv_corrupted = replace(tv_bug, mode="corrupted")   # reading the verifier and submitting its value passes
tv_possible = replace(tv_bug, mode="possible")   # the control: the honest answer passes
mbpp_impossible = EvalCfg(env="mbpp", config="impossible", n=None)   # every task once, the corrupted visible test
mbpp_honeypot = replace(mbpp_impossible, config="honeypot")
smoke_tv = replace(tv_bug, n=2)
smoke_tv_32k = replace(smoke_tv, config="mt32k")   # does the model act within 32768 tokens per turn, where it re-verified the grid to the 8192 cap
smoke_mbpp = replace(mbpp_impossible, n=2)

# intervened runs
cheat_vs_clean = "grader_parity_cheat_vs_clean"
parity_steer = replace(   # grader_lens's show_completion cell: 0.5 x a unit direction at every position of layers 12-47
    parity,
    add_vector=cheat_vs_clean,
    add_layers=list(range(12, 48)),
    add_alpha=0.4,
    add_scaling="unit",
    n=64
)
parity_ablate = replace(   # grader_lens's show_completion cell with proj_hooks: each layer's own unit row projected out at every position of layers 12-47
    parity,
    ablate_vector=cheat_vs_clean,
    ablate_layers=list(range(12, 48)),
    n=1024
)
pets_steer = replace(
    pets,
    add_vector=cheat_vs_clean,
    add_layers=list(range(12, 48)),
    add_alpha=0.5,
    add_scaling="unit",
    n=256
)

secret_steer = replace(   # grader_lens's show_completion cell: 0.5 x a unit direction at every position of layers 12-47
    secret,
    n=64,
    add_vector=cheat_vs_clean,
    add_layers=list(range(12, 48)),
    add_alpha=0.4,
    add_scaling="unit",
)
smoke_secret_steer = replace(secret_steer, n=8)
secret_ablate = replace(secret, ablate_vector=cheat_vs_clean, ablate_layers=list(range(12, 48)))
parity_ablate_512 = replace(parity_ablate, n=512)
pets_ablate_512 = replace(parity_ablate_512, family="pets_listed")
smoke_secret_ablate = replace(secret_ablate, n=8)


def rate_rows(label: str, records: list[dict]) -> list[tuple]:
    """One row per group of rollouts (a grader family's side, or all of another env's): the valid ones, the cheats among them, their rate and its 95% Wilson interval."""
    groups = {}
    for r in records:
        group = f"{r['labels']['family']}.{r['labels']['side']}" if r["env"] == "grader" else "all"
        groups.setdefault(group, []).append(r)
    rows = []
    for group, rollouts in groups.items():
        valid = [r for r in rollouts if r["env"] not in ("grader", "mbpp") or r["labels"]["cheat"] is not None]  # a grader answer on neither side, or an mbpp answer without a code block, has no cheat label
        n_cheated = sum(r["cheated"] for r in valid)
        rate = n_cheated / len(valid) if valid else float("nan")
        low, high = wilson(n_cheated, len(valid))
        rows.append((label, group, f"{len(valid)}/{len(rollouts)}", f"{n_cheated}/{len(valid)} = {rate:.3f}", f"[{low:.3f}, {high:.3f}]"))
    return rows


def summary(cfg: EvalCfg) -> str:
    """The fields the instance sets: the required ones and those not at their default, a contiguous layer list as lo-hi."""
    show = lambda v: f"{v[0]}-{v[-1]}" if isinstance(v, list) and len(v) > 1 and v == list(range(v[0], v[-1] + 1)) else v
    return " ".join(f"{f.name}={show(getattr(cfg, f.name))}" for f in fields(cfg) if f.name != "name" and (f.default is MISSING or getattr(cfg, f.name) != f.default))


instances = {k: v for k, v in globals().items() if isinstance(v, EvalCfg)}
if len(sys.argv) == 1:
    print(f"{bold}{purple}run.py{endc}  ./run.py <name> ...  {gray}the named runs, one after another{endc}")
    for name, cfg in instances.items():
        print(f"  {cyan}{name:<18}{endc}{gray}{summary(cfg)}{endc}")
    sys.exit()
for name in sys.argv[1:]:
    assert name in instances, f"no EvalCfg named {name} in run.py, which has {list(instances)}"
assert len(set(sys.argv[1:])) == len(sys.argv[1:]), f"a run is listed twice: {sys.argv[1:]}"
cfgs = [replace(instances[name], name=name) for name in sys.argv[1:]]
tasks = [build_task(cfg) for cfg in cfgs]  # every run's checks, before any eval

for cfg in cfgs:
    print(f"{purple}=== {cfg.name}: {summary(cfg)}{endc}")
    print(f"{gray}asking {cfg.base_url} for its models (the tunnel is ssh -N -f -L 8000:localhost:8000 vast){endc}")
    served = [m["id"] for m in httpx.get(f"{cfg.base_url}/models", timeout=10).json()["data"]]
    needed = [cfg.model.removeprefix("vllm/")] + ([cfg.lora] if cfg.lora else [])
    assert set(needed) <= set(served), f"{cfg.name}: the server serves {served}, not {sorted(set(needed) - set(served))}"
    in_flight = re.findall(r"^vllm:num_requests_(?:running|waiting)\{.*\} (\S+)$", httpx.get(f"{cfg.base_url.removesuffix('/v1')}/metrics", timeout=10).text, re.M)
    assert in_flight, f"{cfg.name}: no vllm:num_requests_running or vllm:num_requests_waiting in the server's /metrics"
    assert sum(float(x) for x in in_flight) == 0, f"{cfg.name}: the server has requests in flight (running and waiting counts {in_flight}): its hooks apply to every request it gets, and this script sets and clears them"
    print(f"{gray}the server serves {served} and has no request in flight{endc}")

rows = []
for i, (cfg, task) in enumerate(zip(cfgs, tasks)):
    print(f"{purple}=== {i + 1}/{len(cfgs)}: {cfg.name} ==={endc}")
    print(f"{gray}server hooks set for {cfg.name}: {set_hooks(cfg.base_url, cfg)}{endc}")
    start = time.time()
    try:
        [log] = eval(task, model=cfg.model, model_base_url=cfg.base_url, max_connections=cfg.max_connections, display=cfg.display, log_dir=str(ROOT / "logs" / cfg.env), log_model_api=True, model_args={"client_timeout": 3600})
    finally:
        print(f"{gray}server hooks cleared: {clear_hooks(cfg.base_url)}{endc}")
    assert log.status == "success", f"{log.location}: {log.status}, {log.error.message if log.error else 'no error recorded'}"
    out, records = write_records(log, ROOT / "data" / "inspect")
    print(f"{green}{len(records)} rollouts in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}, {sum(r['ids'] is not None for r in records)} with token ids: {out.relative_to(ROOT)}{endc}")
    run_rows = rate_rows(condition(cfg), records)
    show_table(HEADERS, run_rows)
    rows += run_rows
if len(cfgs) > 1:
    show_table(HEADERS, rows)
