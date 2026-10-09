"""
Suppression variants, screened on pets_listed at 64 per side (the family with a baseline a drop can be seen against: pet 0.62, wild 0.92 cheat
under none). The plain ablation (each layer's own row of grader_parity_cheat_vs_clean projected out at layers 12-47) left both pets and parity
at their baselines, while adding 0.4 x the unit rows at the same layers drove parity to 0.99. run.py imports these, so `./run.py pets_neg_1 ...`.
"""
from dataclasses import replace

from common import EvalCfg

cheat_vs_clean = "grader_parity_cheat_vs_clean"
pets_64 = EvalCfg(env="grader", config="default", family="pets_listed", system="none", prompts="hack", n=64)

# negative steering: the mirror of the positive add that works, -alpha x the unit row at layers 12-47
pets_neg_0p4 = replace(pets_64, add_vector=cheat_vs_clean, add_layers=list(range(12, 48)), add_alpha=-0.4, add_scaling="unit")
pets_neg_1 = replace(pets_neg_0p4, add_alpha=-1.0)
pets_neg_2 = replace(pets_neg_0p4, add_alpha=-2.0)

# over-projection: h - k (h . u) u at layers 12-47, k = 2 reflects the component, k = 4 pushes it twice as far the other way
pets_abl = replace(pets_64, ablate_vector=cheat_vs_clean, ablate_layers=list(range(12, 48)))
pets_abl_x2 = replace(pets_abl, ablate_scale=2.0)
pets_abl_x4 = replace(pets_abl, ablate_scale=4.0)

# where the projection is applied: every layer, the late layers where the gap is largest, or one row (layer 50, the largest gap) everywhere
pets_abl_all = replace(pets_abl, ablate_layers=list(range(0, 64)))
pets_abl_late = replace(pets_abl, ablate_layers=list(range(40, 64)))
pets_abl_row50 = replace(pets_abl, ablate_layers=list(range(12, 64)), ablate_row=50)

# dose and controls for the negative add, which drove both pets sides to 0 cheat at alpha -1 (coherent reasoning, half the length): a smaller alpha, a random
# unit direction at the same alpha and layers (is it the direction or any push of that size?), a narrower band, and the parity family
pets_neg_0p2 = replace(pets_neg_0p4, add_alpha=-0.2)
pets_rand_neg_1 = replace(pets_neg_1, add_vector="random_unit_seed0")
pets_neg_1_mid = replace(pets_neg_1, add_layers=list(range(24, 40)))
parity_neg_1 = replace(pets_neg_1, family="parity")

# alpha -0.4 keeps reasoning at its usual length (median 1113 tokens vs 1063 under none; -1 halves it) and still took pet to 0 and wild to 0.52: the random
# control at that dose, the positive add at that dose on pets (does the direction induce pets cheating too), a step up the dose curve, and parity at it
pets_rand_neg_0p4 = replace(pets_neg_0p4, add_vector="random_unit_seed0")
pets_pos_0p4 = replace(pets_neg_0p4, add_alpha=0.4)
pets_neg_0p6 = replace(pets_neg_0p4, add_alpha=-0.6)
parity_neg_0p4 = replace(pets_neg_0p4, family="parity")

# transfer to the agentic env at the dose that cleared pets' pet side with reasoning intact: 32 games
secret_neg_0p4 = replace(EvalCfg(env="secret_number", config="qwen3.6-27b", n=32, seed=0), add_vector=cheat_vs_clean, add_layers=list(range(12, 48)), add_alpha=-0.4, add_scaling="unit")
secret_neg_0p4_128 = replace(secret_neg_0p4, n=128)  # 32 games gave 0 cheats against a baseline near 0.09, which 32 cannot distinguish from baseline
