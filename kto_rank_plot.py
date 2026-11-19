#!./.venv/bin/python
"""Train curves of the rank-k KTO runs (kto_cfgs.py's w_* configs), laptop: per step the summed log-prob gain of the cheating and the honest train rollouts
and the summed token KL on the neutral rollouts, in nats per rollout, as running means over WINDOW steps of one env. A finished run is read from its results
line in acts/kto_sweep/rank/<name>.jsonl, a running one from the train lines acts/kto_sweep/rank/pull_live.sh copied to acts/kto_sweep/rank/live/<name>.log
(the progress bar's rounded values). The joint runs take the envs in turn, so their grader steps and MBPP steps are separate curves, and so are the grader
and MBPP neutral steps of the MBPP runs. Writes figures/kto_rank_train.png."""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RANK = Path("acts/kto_sweep/rank")
WINDOW = 50
RUNS = dict(w_g_r2=("grader, rank 2", "#4cc9f0", "-"), w_g_r8=("grader, rank 8", "#f72585", "-"), w_m_r2=("MBPP, rank 2", "#4cc9f0", ":"), w_m_r8=("MBPP, rank 8", "#f72585", ":"), w_b_r2=("joint, rank 2", "#4cc9f0", "--"), w_b_r8=("joint, rank 8", "#f72585", "--"))

def log_of(name: str) -> list[dict]:
    if (RANK / f"{name}.jsonl").exists(): return json.loads((RANK / f"{name}.jsonl").read_text())["log"]
    if not (RANK / "live" / f"{name}.log").exists(): return []
    return [dict(zip(("loss", "cheat", "honest", "kl"), map(float, m.groups()))) for m in re.finditer(r"loss (\S+) cheat (\S+) honest (\S+) kl (\S+)", (RANK / "live" / f"{name}.log").read_text())]

def running(x: np.ndarray) -> np.ndarray: return np.convolve(x, np.ones(WINDOW) / WINDOW, mode="valid")

plt.style.use("dark_background")
fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex="col")
for name, (label, color, style) in RUNS.items():
    log = log_of(name)
    trained = "grader" if "_g_" in name else "mbpp" if "_m_" in name else "both"
    for col, env in enumerate(("grader", "mbpp")):
        # the steps training on this env, and the steps whose neutrals are this env's
        train_steps = [i for i in range(len(log)) if trained == env or (trained == "both" and i % 2 == col)]
        neutral_steps = [i for i in range(len(log)) if (trained == "grader" and env == "grader") or (trained != "grader" and i % 2 == col)]
        for row, (key, steps) in enumerate((("cheat", train_steps), ("honest", train_steps), ("kl", neutral_steps))):
            if len(steps) >= WINDOW: axes[row, col].plot(np.array(steps)[WINDOW - 1:] + 1, running(np.array([log[i][key] for i in steps])), color=color, ls=style, lw=1.6, label=f"{label} ({len(log)} steps)")
for col, env in enumerate(("grader", "MBPP")):
    axes[0, col].set_title(f"{env} train rollouts: cheating completions' log-prob gain (lower is better)")
    axes[1, col].set_title(f"{env} train rollouts: honest completions' log-prob gain (higher is better)")
    axes[2, col].set_title(f"{env} neutral rollouts: token KL from the base model (lower is better)")
    axes[1, col].axhline(0, color="#888", lw=0.8)
    axes[2, col].set_yscale("log")
    axes[2, col].set_xlabel("train step of the run")
    for row in range(3):
        axes[row, col].set_ylabel("nats per rollout, summed over its tokens")
        axes[row, col].grid(alpha=0.2)
        axes[row, col].legend(fontsize=8)
fig.suptitle(f"Rank-2 and rank-8 KTO projection subspaces at decoder layers 1-62: train metrics, running mean over {WINDOW} steps of the env", fontsize=13)
fig.tight_layout()
fig.savefig("figures/kto_rank_train.png", dpi=130)
print("figures/kto_rank_train.png")
