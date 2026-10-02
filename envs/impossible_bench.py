"""
ImpossibleBench's LiveCodeBench task (the official Inspect package, pinned in pyproject) under a named config: the split, scaffold, attempt
budget and instruction prompt from envs/configs/impossible_bench/<config>.yaml, the repo's compose file under docker/impossible_bench (the
installed package does not ship one), and the shared sampling and condition settings.

    inspect eval envs/impossible_bench.py -T config=paper -T limit=20 --model vllm/Qwen3.6-27B

Cheating is passing on an impossible split (the package scorer's C on oneoff or conflicting); passing on original is the capability number.
"""
from impossiblebench import impossible_livecodebench
from inspect_ai import Task, task

from common import VECTORS, generate_config, load_config

COMPOSE = "docker/impossible_bench/compose.yaml"


@task
def lcb(config: str = "paper", limit: int | None = None, condition: str = "none", vectors: str = "Qwen3.6-27B") -> Task:
    cfg = load_config("impossible_bench", config)
    bench = impossible_livecodebench(split=cfg["split"], agent_type=cfg["agent_type"], instruction_prompt=cfg["instruction_prompt"], allow_test_modifications=cfg["allow_test_modifications"], max_attempts=cfg["max_attempts"], message_limit=cfg["message_limit"], limit=limit, sandbox=("docker", COMPOSE))
    bench.config = generate_config(cfg, condition, VECTORS / vectors)
    bench.metadata = {"env": "impossible_bench", "config_id": config, "condition": condition, "vectors": vectors, "split": cfg["split"]}
    return bench
