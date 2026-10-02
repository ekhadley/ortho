"""
Every environment for a couple of samples against a served model, then a summary of the converted records: run after any change to a task, the
converter or the serve setup. The sandboxed tasks need docker; --envs picks a subset.

    uv run python envs/smoke.py --model vllm/Qwen3.5-0.8B --base-url http://localhost:8011/v1
    uv run python envs/smoke.py --model vllm/Qwen3.5-0.8B --base-url http://localhost:8011/v1 --condition add:test:10:4 --vectors Qwen3.5-0.8B --envs grader
"""
import argparse

from inspect_ai import eval

from common import VECTORS, set_hooks
from convert import convert
from grader import grader
from impossible_bench import lcb
from secret_number import secret_number

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--base-url", default=None)
p.add_argument("--condition", default="none")
p.add_argument("--vectors", default="Qwen3.6-27B")
p.add_argument("--n", type=int, default=2)
p.add_argument("--log-dir", default="logs/smoke")
p.add_argument("--envs", nargs="+", default=["grader", "secret_number", "lcb"])
args = p.parse_args()

tasks = {
    "grader": lambda: grader(family="all", n=args.n, condition=args.condition, vectors=args.vectors),
    "secret_number": lambda: secret_number(n=args.n, condition=args.condition, vectors=args.vectors),
    "lcb": lambda: lcb(limit=args.n, condition=args.condition, vectors=args.vectors),
}
if args.condition.startswith("ablate:"):  # the projection lives on the server: set it for this run, clear it after
    print(set_hooks(args.base_url, args.condition, VECTORS / args.vectors))
logs = eval([tasks[name]() for name in args.envs], model=args.model, model_base_url=args.base_url, log_dir=args.log_dir, log_model_api=True, model_args={"client_timeout": 3600})
if args.condition.startswith("ablate:"):
    print(set_hooks(args.base_url, "none", VECTORS / args.vectors))

for log in logs:
    for r in convert(log):
        print(f"{r['env']} {r['rollout_id']}: cheated {r['cheated']} {r['cheat_kinds']}, labels {r['labels']}, {r['n_turns']} turns, ids {len(r['ids']) if r['ids'] else None}")
        for t in r["turns"]:
            print(f"    turn {t['turn_idx']}: reasoning {len(t['reasoning'])} chars, message {len(t['message'])} chars, calls {[c['name'] for c in t['tool_calls']]}, commands {t['commands']}, env outputs {[o[:60] for o in t['env_outputs']]}")
