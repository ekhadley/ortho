# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read first

`research_guidelines.md` is the coding style guide for this project (uv, flat layout, `from mechtools import *` prelude, `torch as t`, dataclass configs, fail-loudly, colored debug prints, `#%%` cells, TransformerBridge via `load_bridge`). Follow it for all Python. Nothing there is repeated here.

## Environment

- Python 3.13, managed by uv. `uv sync` to update the env, `uv add <pkg>` to add deps, `uv run python <file>.py` to run scripts. `mechtools` is not installed; the guidelines give the `uv add` line for it.
- No tests, linter, or build step exist. Do not invent them.
- Local GPU is 12GB. Anything beyond a ~4B model in bf16 or a real activation-collection run belongs on rented compute (Vast.ai, HPC), not here.

## What this project is

SPAR project "Orthogonalization Against Reward Hacking", mentored by Vlad. Goal: test whether refusal-style orthogonalization (abliteration, a la arxiv 2406.11717 / the `heretic` library) can remove reward hacking from an already-trained production model.

Design decisions already made, do not relitigate them in code:

- Ablation is applied post-training to the weights of a released model, not during RL. Start with 4B-8B models (Qwen3.5-class), scale up only once it works.
- Directions come from difference-of-means over paired trajectories (hacking vs. not hacking). Sources: reward-hacking benchmarks with programmatic hack detection (ImpossibleBench, EvilGenie), LLM-judged rollouts, or honeypot envs (`ekhadley/odd-number-hacking`, `gkroiz/agent-interp-envs`). Follow heretic's layer selection and ablation procedure; the open question is which sequence positions to pool activations from, since heretic only uses prompt tokens and rollouts are long agentic transcripts.
- Evaluate hack rate on the same benchmark the training pairs came from, using the same judge. Capability checks: MMLU plus something agentic like SWE-Bench, and the benchmarks must be unhackable.
- Until the minimal version works, no extra degradation-mitigation tricks beyond what heretic already does. A clean negative result is an acceptable outcome.
- LLM judging goes through OpenRouter with the provider pinned (see guidelines). Prefer a judge over substring matching for hack detection.

## Data

Rollouts come from the secret_number environment in `~/wgmn/agent-interp-envs` (results under `results/secret_number/<model>/<batch>/run-N/step-K/`). An agent plays a noisy number-guessing game in a sandbox; reading `/secrets/secret_number.txt` is the main cheat, and the env flags it from the shell command history. `import_rollouts.py` flattens those into `data/rollouts_qwen.jsonl` and `data/rollouts_gpt.jsonl` (gitignored, regenerate with `uv run python import_rollouts.py`), one rollout per line with a nested list of turns; `load_rollouts` and `flatten_turns` in that file are the loaders. The script docstring documents the turn definition and per-field semantics. Things that are easy to get wrong when touching this data:

- Turn boundaries come from per-step snapshot diffs, not from the item stream: the harness drops a turn's trailing visible message before the next turn, and some generations have no reasoning item. The `turns` records are authoritative; the raw `items` transcript can lack message items.
- A turn's `commands` are what the env executed (from the state diff). `tool_calls` are what the model attempted; the env rejects multi-call turns and runs nothing. Cheat flags use `commands`, matching the env's own flags.
- Resumed batches (`*-from-*` dirs) start with turns written by another model; `author_model` and `prefix_turns` say which. The `r<run>s<step>` in those dir names is wrong in places, so the source is found by content match.
- Runs whose step chain is inconsistent are skipped and printed per batch: one qwen3.5 resumed batch with two lineages in the same run dirs, and six gpt-oss batches where the conversation was restarted mid-run.
- Only one run has harvested activations (`Qwen3.8-27B/2026-08-26_09-25-50/run-5/activations.pt`). `scripts/harvest_activations.py` in agent-interp-envs produces them and needs ~54GB VRAM for the 27B model.

`rh_probes_paper_notes.md` summarizes the Goodfire reward-hacking probe paper (arXiv 2609.19101) whose difference-of-means method is the starting point for direction finding here.

## Notes files

- `project_doc.md`: Ethan's pre-project questions and concerns (rank of the reward-hacking direction, where in the pipeline to ablate, relation to CAFT).
- `spar_vlad_meet_notes.md`: team meeting notes, Vlad's answers to those questions (the source of the decisions above), and collaborator Will's ImpossibleBench setup notes. Will found Qwen 3.5 4B hacks little to never on ImpossibleBench LCB with the paper's prompt, so expect to need a more permissive prompt or a different model to get positive examples.
