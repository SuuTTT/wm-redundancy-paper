#!/usr/bin/env python3
"""A2 novelty-MPPI patch: RND/disagreement novelty in the MPPI objective (Plan2Explore-in-TD-MPC2).
Idempotent-ish: asserts each anchor is found exactly once. Backs up originals once."""
import re, shutil, os, sys

TD = "/root/helios-rl/src/helios/algorithms/tdmpc2.py"
RB = "/root/helios-rl/scripts/run_benchmark.py"

def backup(p):
    b = p + ".bak_a2novelty"
    if not os.path.exists(b):
        shutil.copy2(p, b)

def repl(s, old, new, n=1, tag=""):
    c = s.count(old)
    assert c == n, f"[{tag}] expected {n} occurrences, found {c}"
    return s.replace(old, new)

# ---------------- tdmpc2.py ----------------
backup(TD)
s = open(TD).read()

# A) RNDNet class before MPPI planner factory header
anchor_a = ("# ---------------------------------------------------------------------------\n"
            "# MPPI planner factory\n"
            "# ---------------------------------------------------------------------------\n")
rndnet = ('# ---------------------------------------------------------------------------\n'
          '# A2 novelty-MPPI: RND novelty over the SimNorm latent (Plan2Explore-in-TD-MPC2)\n'
          '# ---------------------------------------------------------------------------\n'
          'class RNDNet(nn.Module):\n'
          '    """Small MLP z -> feature. Fixed random target + trained predictor; novelty=pred error."""\n'
          '    hidden: tuple = (256,)\n'
          '    out: int = 64\n\n'
          '    @nn.compact\n'
          '    def __call__(self, z: jax.Array) -> jax.Array:\n'
          '        x = z\n'
          '        for d in self.hidden:\n'
          '            x = nn.relu(nn.Dense(d)(x))\n'
          '        return nn.Dense(self.out)(x)\n\n\n')
s = repl(s, anchor_a, rndnet + anchor_a, tag="A/rndnet")

# B) make_mppi_fn signature
sig_old = ("    gamma: float = 0.99,\n"
           "    rew_scale: float = 10.0,\n"
           "):\n"
           '    """Build the JIT-compiled MPPI planning function (v24 parity).')
sig_new = ("    gamma: float = 0.99,\n"
           "    rew_scale: float = 10.0,\n"
           "    rnd_net=None,\n"
           "    rnd_tgt_params=None,\n"
           "    novelty_beta: float = 0.0,\n"
           "    novelty_type: str = \"rnd\",\n"
           "):\n"
           '    """Build the JIT-compiled MPPI planning function (v24 parity).')
s = repl(s, sig_old, sig_new, tag="B/mppi-sig")

# B2) capture novelty constants after n_noise line
cap_old = ("    n_noise  = n_samples - num_pi_trajs\n"
           "    _gammas  = jnp.array([gamma ** t for t in range(horizon)])\n"
           "    _gamma_H = float(gamma ** horizon)\n")
cap_new = cap_old + ("    _NOV_BETA = float(novelty_beta)\n"
                     "    _NOV_TYPE = str(novelty_type)\n")
s = repl(s, cap_old, cap_new, tag="B2/mppi-const")

# C) rollout + elite selection block
roll_old = ('''            def rollout_one(z_i, a_seq):
                def env_step(z, a):
                    r_logits = rew_net.apply(params["rew"], z[None], a[None])
                    r  = two_hot_inv(r_logits).squeeze()
                    z2 = dyn.apply(params["dyn"], z[None], a[None]).squeeze(0)
                    return z2, r

                zf, rs = jax.lax.scan(env_step, z_i, a_seq)
                pi_a_mean, _ = pi_net.apply(params["pi"], zf[None])
                pi_a_squashed = jnp.tanh(pi_a_mean)
                q_logits = q_net.apply(params["q"], zf[None], pi_a_squashed)
                vt = jnp.maximum(jnp.min(two_hot_inv(q_logits)), 0.0).squeeze()
                return jnp.sum(_gammas * rs) + _gamma_H * vt

            rets = jax.vmap(rollout_one)(z0_batch, acts)

            _, elite_idx = jax.lax.top_k(rets, num_elites)''')
roll_new = ('''            if _NOV_BETA > 0.0:
                def rollout_one(z_i, a_seq):
                    def env_step(z, a):
                        r_logits = rew_net.apply(params["rew"], z[None], a[None])
                        r  = two_hot_inv(r_logits).squeeze()
                        z2 = dyn.apply(params["dyn"], z[None], a[None]).squeeze(0)
                        return z2, (r, z2)

                    zf, (rs, zs_traj) = jax.lax.scan(env_step, z_i, a_seq)
                    pi_a_mean, _ = pi_net.apply(params["pi"], zf[None])
                    pi_a_squashed = jnp.tanh(pi_a_mean)
                    q_logits = q_net.apply(params["q"], zf[None], pi_a_squashed)
                    vt = jnp.maximum(jnp.min(two_hot_inv(q_logits)), 0.0).squeeze()
                    ret = jnp.sum(_gammas * rs) + _gamma_H * vt
                    if _NOV_TYPE == "disagreement":
                        qv = two_hot_inv(q_logits).reshape(-1)   # (2,) two Q-heads
                        nov = jnp.abs(qv[0] - qv[1])
                    else:  # rnd novelty over predicted SimNorm latents
                        tgt = jax.lax.stop_gradient(rnd_net.apply(rnd_tgt_params, zs_traj))
                        prd = rnd_net.apply(params["rnd_prd"], zs_traj)
                        nov = jnp.sum(jnp.mean((prd - tgt) ** 2, axis=-1))
                    return ret, nov

                rets, novs = jax.vmap(rollout_one)(z0_batch, acts)
                nov_n = (novs - jnp.mean(novs)) / (jnp.std(novs) + 1e-6)
                score = rets + _NOV_BETA * nov_n
            else:
                def rollout_one(z_i, a_seq):
                    def env_step(z, a):
                        r_logits = rew_net.apply(params["rew"], z[None], a[None])
                        r  = two_hot_inv(r_logits).squeeze()
                        z2 = dyn.apply(params["dyn"], z[None], a[None]).squeeze(0)
                        return z2, r

                    zf, rs = jax.lax.scan(env_step, z_i, a_seq)
                    pi_a_mean, _ = pi_net.apply(params["pi"], zf[None])
                    pi_a_squashed = jnp.tanh(pi_a_mean)
                    q_logits = q_net.apply(params["q"], zf[None], pi_a_squashed)
                    vt = jnp.maximum(jnp.min(two_hot_inv(q_logits)), 0.0).squeeze()
                    return jnp.sum(_gammas * rs) + _gamma_H * vt

                rets = jax.vmap(rollout_one)(z0_batch, acts)
                score = rets

            _, elite_idx = jax.lax.top_k(score, num_elites)''')
s = repl(s, roll_old, roll_new, tag="C/rollout")

# D) make_update_fn signature
upd_sig_old = ("    calib_coef: float = 0.0,      # iter-30: calibration-shaped jumpy disagreement (0=off; needs jumpy_k>0)\n"
               "    calib_q: float = 0.9,         # iter-30: pinball quantile — train disc to upper-bound err at this quantile\n"
               ") -> tuple:")
upd_sig_new = ("    calib_coef: float = 0.0,      # iter-30: calibration-shaped jumpy disagreement (0=off; needs jumpy_k>0)\n"
               "    calib_q: float = 0.9,         # iter-30: pinball quantile — train disc to upper-bound err at this quantile\n"
               "    rnd_net=None,                 # A2: RND net for latent-novelty predictor training (None=off)\n"
               "    rnd_tgt_params=None,          # A2: fixed random target params\n"
               "    rnd_coef: float = 0.0,        # A2: predictor training weight (0=off)\n"
               ") -> tuple:")
s = repl(s, upd_sig_old, upd_sig_new, tag="D/update-sig")

# E) RND training loss before aux dict + aux key
aux_old = ("        aux = {\n"
           '            "c": jnp.sum(cls) / n,')
aux_new = ('''        # ── A2: RND predictor training (Python-guarded on rnd_coef; grad flows ONLY to
        # params["rnd_prd"] via stop-grad on the encoder latents; graph identical when 0).
        if rnd_coef > 0.0:
            _zr = jax.lax.stop_gradient(z_all.reshape(-1, z_all.shape[-1]))
            _rtgt = jax.lax.stop_gradient(rnd_net.apply(rnd_tgt_params, _zr))
            _rprd = rnd_net.apply(params["rnd_prd"], _zr)
            rnd_loss = jnp.mean(jnp.sum((_rprd - _rtgt) ** 2, -1))
            total = total + rnd_coef * rnd_loss
        else:
            rnd_loss = jnp.array(0.0)

        aux = {
            "c": jnp.sum(cls) / n,''')
s = repl(s, aux_old, aux_new, tag="E/rnd-loss")

aux_key_old = ('            "calib": calib_loss,\n'
               "        }")
aux_key_new = ('            "calib": calib_loss,\n'
               '            "rnd_loss": rnd_loss,\n'
               "        }")
s = repl(s, aux_key_old, aux_key_new, tag="E/aux-key")

open(TD, "w").write(s)
print("tdmpc2.py patched OK")

# ---------------- run_benchmark.py ----------------
backup(RB)
r = open(RB).read()

# F) RND init after params dict
params_anchor = ('''    params = {
        "enc": enc_net.init(ek, dummy_obs),
        "dyn": dyn_net.init(dk, dummy_z, dummy_act),
        "rew": rew_net.init(rk, dummy_z, dummy_act),
        "q":   q_net.init(qk, dummy_z_aug, dummy_act),
        "pi":  pi_net.init(pk, dummy_z_aug),
    }''')
params_new = params_anchor + ('''
    # ── A2 novelty-MPPI (Plan2Explore-in-TD-MPC2): novelty term in the MPPI objective.
    # RND = trained predictor error over the SimNorm latent; disagreement = 2-Q-head spread.
    _nov_beta = float(os.environ.get("NOVELTY_BETA", "0.0"))
    _nov_type = os.environ.get("NOVELTY_TYPE", "rnd").strip()
    _rnd_net = None; _rnd_tgt_params = None; _rnd_coef = 0.0
    _novelty_active = (not use_glass) and _nov_beta > 0.0
    if _novelty_active and _nov_type == "rnd":
        from helios.algorithms.tdmpc2 import RNDNet
        _rnd_dim = int(os.environ.get("RND_DIM", "64"))
        _rnd_hidden = tuple(int(x) for x in os.environ.get("RND_HIDDEN", "256").split(","))
        _rnd_coef = float(os.environ.get("RND_COEF", "1.0"))
        _rnd_net = RNDNet(hidden=_rnd_hidden, out=_rnd_dim)
        key, _rtk, _rpk = jax.random.split(key, 3)
        _rnd_tgt_params = _rnd_net.init(_rtk, dummy_z)
        params["rnd_prd"] = _rnd_net.init(_rpk, dummy_z)
        print(f"  [A2-NOVELTY] rnd novelty-MPPI beta={_nov_beta} rnd_dim={_rnd_dim} hidden={_rnd_hidden} coef={_rnd_coef}", flush=True)
    elif _novelty_active:
        print(f"  [A2-NOVELTY] {_nov_type} novelty-MPPI beta={_nov_beta} (no extra params)", flush=True)''')
r = repl(r, params_anchor, params_new, tag="F/rnd-init")

# G) make_update_fn non-glass call: add rnd kwargs
upd_call_old = ('''                calib_coef=float(calib_coef),
                calib_q=float(calib_q),
            )
        return ms''')
upd_call_new = ('''                calib_coef=float(calib_coef),
                calib_q=float(calib_q),
                rnd_net=_rnd_net,
                rnd_tgt_params=_rnd_tgt_params,
                rnd_coef=_rnd_coef,
            )
        return ms''')
r = repl(r, upd_call_old, upd_call_new, tag="G/update-call")

# H) make_mppi_fn kwargs
mppi_kw_old = ('''    _mppi_kw = {}
    if use_glass:
        _mppi_kw["use_cluster_obs"] = bool(glass_cfg.get("use_cluster_obs", False))
        _mppi_kw["cluster_obs_proto_temperature"] = float(glass_cfg.get("proto_temperature", 1.0))''')
mppi_kw_new = mppi_kw_old + ('''
    elif _novelty_active:
        _mppi_kw["rnd_net"] = _rnd_net
        _mppi_kw["rnd_tgt_params"] = _rnd_tgt_params
        _mppi_kw["novelty_beta"] = _nov_beta
        _mppi_kw["novelty_type"] = _nov_type''')
r = repl(r, mppi_kw_old, mppi_kw_new, tag="H/mppi-kw")

# I) JSONL logging after write_csv
csv_old = "                write_csv(fh, env_id, seed, env_steps, ret)"
csv_new = ('''                write_csv(fh, env_id, seed, env_steps, ret)
                _a2j = os.environ.get("A2_JSONL", "").strip()
                if _a2j:
                    import json as _a2json
                    with open(_a2j, "a") as _a2f:
                        _a2f.write(_a2json.dumps({"step": int(env_steps), "pi_return": float(ret),
                                   "mppi_return": float(mppi_ret), "beta": _nov_beta,
                                   "novelty_type": (_nov_type if _novelty_active else "vanilla")}) + "\\n")''')
r = repl(r, csv_old, csv_new, tag="I/jsonl")

open(RB, "w").write(r)
print("run_benchmark.py patched OK")
print("ALL PATCHES APPLIED")
