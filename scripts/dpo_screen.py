"""From the repo root, on the laptop: sets up train_vector.py's held-out screen (TRAIN False: teacher-forced log-prob gains on same-prompt cheating and honest
rollouts, and the neutrals' token KL) for the additive DPO vectors, as steering (MODE add at doses 1.2, 2.4 and 3.6) and as a projection (every position, each
layer's own row, grader_proj's pin), on the grader eval rows and on the MBPP test pairs, and for the directions that were served under mbpp_kto's hook (one direction,
generated tokens, each at its own pin; the <env>_gen screens), whose served cheat rates the scores can be set beside. Writes data/vectors/<model>/dpo_screen/<name>.safetensors, the four DPO
vectors with grader_proj's mean_resid beside their rows, will_g9 (the rows of Will's A3 checkpoint cb_a3, whose bias on decoder layer L is -3.6 times his unit
row) and rand (random unit rows), then acts/dpo_screen/screen_<tag>.py, one copy of train_vector.py per screen with its settings, ending at the scores file
data/vectors/<model>/dpo_screen/<tag>.json. scripts/dpo_screen_box.sh runs the copies on the box."""
from pathlib import Path

import torch as t
from safetensors.torch import load_file, save_file

OUT = Path("data/vectors/Qwen3.8-27B")
A3 = Path.home() / "wgmn/ortho_box_archive/59147695/models/cb_a3/model-resid-bias.safetensors"
DPO = ["grader_dpo", "grader_dpo_lr1e-4", "grader_dpo_beta1e-2", "grader_dpo_unit_init"]
LAYERS = list(range(30, 64))
SCREENS = Path("acts/dpo_screen")
SERVED = ["grader_kto", "grader_proj", "mbpp_hunt_vr48", "mbpp_hunt_vrw48", "mbpp_hunt_wdom48", "mbpp_hunt_rand"]   # scored beside mbpp_kto in the _gen screens

#%% vectors
(OUT / "dpo_screen").mkdir(exist_ok=True)
pin = load_file(OUT / "grader_proj.safetensors")["mean_resid"]
bias = load_file(A3)
g9 = t.zeros(65, 5120)
g9[[L + 1 for L in LAYERS]] = t.nn.functional.normalize(-t.stack([bias[f"model.language_model.layers.{L}.mlp.down_proj.bias"].float() for L in LAYERS]), dim=1)
rand = t.zeros(65, 5120)
rand[[L + 1 for L in LAYERS]] = t.nn.functional.normalize(t.randn(len(LAYERS), 5120, generator=t.Generator().manual_seed(0)), dim=1)
rows = {name: load_file(OUT / f"{name}.safetensors")["direction_unit"] for name in DPO} | {"will_g9": g9, "rand": rand}
for name, unit in rows.items():
    assert unit.shape == (65, 5120) and t.allclose(unit[31:].norm(dim=1), t.ones(34), atol=1e-3) and unit[:31].abs().max() == 0, name
    save_file({"direction_unit": unit.contiguous(), "mean_resid": pin}, OUT / "dpo_screen" / f"{name}.safetensors")
names = [f"dpo_screen/{name}" for name in rows]

#%% screens
src = Path("train_vector.py").read_text()
def make(tag: str, subs: list[tuple[str, str]]):
    s = src[:src.index("#%% fraction removed")]
    for a, b in subs + [("TRAIN = True\n", "TRAIN = False\n"), ('OUT / f"{NAME}_compare.json"', f'OUT / "dpo_screen/{tag}.json"')]:
        assert s.count(a) == 1, (tag, a, s.count(a))
        s = s.replace(a, b)
    (SCREENS / f"screen_{tag}.py").write_text(s)
rowwise = [("GEN_ONLY = True\n", "GEN_ONLY = False\n"), ("N_DIRS = 1 ", "N_DIRS = len(LAYERS) ")]
for env, subs in [("grader", [('ENV = "mbpp"', 'ENV = "grader"')]), ("mbpp", [("NEUTRAL_HELD_EVERY = 1 ", "NEUTRAL_HELD_EVERY = 4 ")])]:
    make(f"{env}_gen", subs + [('COMPARE = ["grader_kto"]', f"COMPARE = {SERVED!r}")])
    subs = subs + rowwise
    for dose in ["1.2", "2.4", "3.6"]:
        make(f"{env}_add{dose.replace('.', '')}", subs + [('MODE = "project"', 'MODE = "add"'), ("ALPHA = 3.6 ", f"ALPHA = {dose} "), ('NAME = "mbpp_kto"', f'NAME = "{names[0]}"'), ('COMPARE = ["grader_kto"]', f"COMPARE = {names[1:]!r}")])
    make(f"{env}_proj", subs + [('NAME = "mbpp_kto"', 'NAME = "grader_proj"'), ('COMPARE = ["grader_kto"]', f"COMPARE = {names!r}")])
print(sorted(p.name for p in SCREENS.glob("screen_*.py")), sorted(p.name for p in (OUT / "dpo_screen").glob("*")))
