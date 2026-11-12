"""The runs of the KTO direction sweep: SweepCfg, one flat dataclass per run, and the named instances kto_sweep.py trains and screens. A variant is a
`replace` of another instance. The worker re-imports this file before each run, so an instance added here can be queued while it is working."""
from dataclasses import dataclass, replace


@dataclass
class SweepCfg:
    env: str = "grader"   # the rollouts trained on: grader, mbpp or both (alternating steps)
    neutrals: tuple[str, ...] = ("grader",)   # the envs whose no-incentive rollouts are held by token KL, alternating steps
    layers: tuple[int, ...] = tuple(range(30, 64))   # decoder layers hooked at their output
    per_layer: bool = False   # one direction per hooked layer instead of one shared by all
    rank: int = 1   # directions removed at each hooked layer, an orthonormal set when above 1
    scale: float = 1.0   # how much of the component is removed: 1 projects it out, 2 reflects it
    clamp: bool = False   # remove only the part past the pin on the direction's positive side
    shift: bool = False   # steering instead of projection: the learned offset is added along the direction at every hooked token, whatever the residual
    offset: str = "zero"   # zero, or learned: a scalar per hooked layer moving the pin along the direction, in units of that layer's per-dimension residual std
    positions: str = "gen"   # the generated tokens hooked: gen (all of them), cot or answer
    honest: str = "kto"   # honest completions: kto (log-prob pushed above the base model's), kl (held like a neutral) or off
    per_token: bool = False   # gains and KL per 200 completion tokens instead of summed over the completion
    beta: float = 0.1
    kl_weight: float = 0.1
    lr: float = 1e-3
    offset_lr: float = 1e-2   # the learned offsets' own rate: they are in sigma units, the direction's raw rows grow to a norm near 1
    steps: int = 200
    k: int = 4   # cheating, honest and neutral rollouts a step
    max_len: int = 2500   # train rollouts with more tokens are left out; up to 2,500 a step runs without gradient checkpointing
    seed: int = 0
    init: str = ""   # a saved vector of data/vectors/<model>/ to start from, with its pin, or sweep:<name> for a trained config of this sweep (its parameters, at this config's own pin); empty for init_scale * randn at the trained envs' pin
    init_scale: float = 1e-4
    full: bool = False   # also screen all 97 MBPP test pairs, not only the 58 under 2,500 tokens


# checks of the screen itself
noop = SweepCfg(steps=0, scale=0.0)   # the hook removing nothing: gains are the noise of batching alone
rand = SweepCfg(steps=0, init_scale=1.0, full=True)   # a random unit direction at the grader pin
ref_gkto = SweepCfg(steps=0, init="grader_kto", full=True)   # the saved directions under this screen, each at its own pin
ref_mkto = SweepCfg(steps=0, init="mbpp_kto", full=True)

floor = replace(rand, scale=1e-3, full=False)   # a random direction removed a thousandth: what bf16 rounding alone adds to the KL and the gains once anything is perturbed
noop_fast = replace(noop)   # the same checks through the cached-residual path, where layers 0-29 are not rerun
ref_gkto_fast = replace(ref_gkto, full=False)

# grader_kto's recipe
g0 = SweepCfg()
g0_fast = replace(g0)   # g0 again through the cached-residual path, without gradient checkpointing
g0_522 = replace(g0, steps=522)   # its original length
g0_s1 = replace(g0, seed=1)
g0_s2 = replace(g0, seed=2)

# one change from g0 each
g_kl003 = replace(g0, kl_weight=0.03)
g_kl03 = replace(g0, kl_weight=0.3)
g_kl1 = replace(g0, kl_weight=1.0)
g_beta003 = replace(g0, beta=0.03)
g_beta03 = replace(g0, beta=0.3)
g_lr3 = replace(g0, lr=3e-3)
g_l36_52 = replace(g0, layers=tuple(range(36, 53)))
g_l40_63 = replace(g0, layers=tuple(range(40, 64)))
g_l20_63 = replace(g0, layers=tuple(range(20, 64)))
g_l0_63 = replace(g0, layers=tuple(range(64)))
g_l36 = replace(g0, layers=(36,))
g_l44 = replace(g0, layers=(44,))
g_perlayer = replace(g0, per_layer=True)
g_rank2 = replace(g0, rank=2)
g_rank4 = replace(g0, rank=4)
g_offset = replace(g0, offset="learned")
g_clamp = replace(g0, clamp=True)
g_clamp_offset = replace(g0, clamp=True, offset="learned")
g_reflect = replace(g0, scale=2.0)
g_half = replace(g0, scale=0.5)
g_cot = replace(g0, positions="cot")
g_answer = replace(g0, positions="answer")
g_honest_kl = replace(g0, honest="kl")
g_honest_off = replace(g0, honest="off")
g_pertoken = replace(g0, per_token=True)
g_nboth = replace(g0, neutrals=("grader", "mbpp"), max_len=2500)   # MBPP's original-test rollouts among the neutrals: benign MBPP data, no MBPP cheating

# MBPP and both envs, on rollouts of up to 2,500 tokens (1,583 of its 1,907 train rollouts)
m0 = SweepCfg(env="mbpp", neutrals=("grader", "mbpp"))   # mbpp_kto's recipe: its neutrals are the grader ones and MBPP's original-test rollouts
b0 = SweepCfg(env="both", neutrals=("grader", "mbpp"), steps=400)   # 200 steps of each env
m1 = replace(m0, steps=500)   # MBPP's loss only starts falling near step 150
b1 = replace(b0, steps=800)   # 400 steps of each env

# earlier configs again with no training, for the screen sets added after they ran (tv_turn, secret_turn)
t_rand = replace(rand, full=False)
t_gkto = replace(ref_gkto, full=False)
t_mkto = replace(ref_mkto, full=False)
t_g0 = replace(g0, steps=0, init="sweep:g0_fast")
t_b1 = replace(b1, steps=0, init="sweep:b1")
b2 = replace(b1, init="sweep:b1", steps=600, kl_weight=0.3)   # b1 trained on: 300 more steps of each env, the neutral KL weighted three times as much
t_b2 = replace(b2, steps=0, init="sweep:b2", full=True)

# steering trained by the same loss: one direction, a learned step per hooked layer, generated tokens only
g_shift = replace(g0, shift=True, offset="learned")
g_shift_b03 = replace(g_shift, beta=0.3)   # the loss saturates near a third of the log-prob push
g_shift_kl1 = replace(g_shift, kl_weight=1.0)
t_goffset = replace(g_offset, steps=0, init="sweep:g_offset")
