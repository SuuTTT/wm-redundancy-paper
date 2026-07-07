"""A2 feasibility probe: does the novelty term change MPPI action selection?
Builds vanilla (beta=0) vs RND-novelty (beta>0) plans with IDENTICAL model params,
same obs+key, and checks the selected action differs + logs mean novelty."""
import os, jax, jax.numpy as jnp, numpy as np
from helios.algorithms.tdmpc2 import (
    Encoder, Dynamics, RewardHead, QEnsemble, Pi, RNDNet, make_mppi_fn)

latent_dim, hidden, V = 128, (256, 256), 8
obs_dim, act_dim = 5, 1
key = jax.random.PRNGKey(0)
k = iter(jax.random.split(key, 12))
enc = Encoder(latent_dim=latent_dim, hidden=hidden, V=V)
dyn = Dynamics(latent_dim=latent_dim, hidden=hidden, V=V)
rew = RewardHead(hidden=hidden)
q = QEnsemble(hidden=hidden)
pi = Pi(action_dim=act_dim, hidden=hidden)
dz = jnp.zeros((1, latent_dim)); da = jnp.zeros((1, act_dim)); do = jnp.zeros((1, obs_dim))
params = {
    "enc": enc.init(next(k), do), "dyn": dyn.init(next(k), dz, da),
    "rew": rew.init(next(k), dz, da), "q": q.init(next(k), dz, da),
    "pi": pi.init(next(k), dz)}
rnd = RNDNet(hidden=(256,), out=64)
rnd_tgt = rnd.init(next(k), dz)
# predictor: perturb from target so novelty is NON-constant across latents (mimics partially trained)
rnd_prd = jax.tree_util.tree_map(lambda x: x + 0.5 * jax.random.normal(next(k), x.shape) if x.ndim else x, rnd.init(next(k), dz))
params["rnd_prd"] = rnd_prd

common = dict(horizon=3, n_samples=512, num_elites=64, num_pi_trajs=24, n_iter=6,
              min_std=0.05, max_std=2.0, act_low=-1.0, act_high=1.0, act_dim=act_dim, gamma=0.99)
plan_van = make_mppi_fn(enc, dyn, rew, q, pi, novelty_beta=0.0, **common)
plan_rnd = make_mppi_fn(enc, dyn, rew, q, pi, rnd_net=rnd, rnd_tgt_params=rnd_tgt,
                        novelty_beta=1.0, novelty_type="rnd", **common)
plan_dis = make_mppi_fn(enc, dyn, rew, q, pi, novelty_beta=1.0, novelty_type="disagreement", **common)

mu0 = jnp.zeros((3, act_dim)); std0 = jnp.full((3, act_dim), 2.0)
diffs_rnd, diffs_dis = [], []
for i in range(6):
    obs = jax.random.normal(jax.random.PRNGKey(100 + i), (obs_dim,))
    pk = jax.random.PRNGKey(200 + i)
    a_v, _, _ = plan_van(params, obs, mu0, std0, pk, jnp.bool_(True))
    a_r, _, _ = plan_rnd(params, obs, mu0, std0, pk, jnp.bool_(True))
    a_d, _, _ = plan_dis(params, obs, mu0, std0, pk, jnp.bool_(True))
    diffs_rnd.append(float(jnp.abs(a_v - a_r).sum()))
    diffs_dis.append(float(jnp.abs(a_v - a_d).sum()))

# mean raw novelty over encoded latents
z = enc.apply(params["enc"], jax.random.normal(jax.random.PRNGKey(7), (256, obs_dim)))
nov = jnp.mean((rnd.apply(rnd_prd, z) - rnd.apply(rnd_tgt, z)) ** 2)
print(f"mean_raw_rnd_novelty = {float(nov):.4f}")
print(f"RND: mean |a_vanilla - a_rnd| over 6 obs = {np.mean(diffs_rnd):.4f}  per-obs={np.round(diffs_rnd,4)}")
print(f"DIS: mean |a_vanilla - a_dis| over 6 obs = {np.mean(diffs_dis):.4f}  per-obs={np.round(diffs_dis,4)}")
print("NOVELTY_CHANGES_ACTIONS =", (np.mean(diffs_rnd) > 1e-4))
