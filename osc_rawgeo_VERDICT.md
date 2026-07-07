# H-JEPA with BOTH wall-fixes (OSC/Cartesian LL action + RAW-GEOMETRY LL input) — VERDICT (seed0)
## This run CLOSES the H-JEPA Bet-2 investigation.

**VERDICT: NULL (decisive).** Combining BOTH fixes — the feasible OSC/Cartesian
end-effector LL action **AND** feeding the LL actor/critic the raw ee-cube geometry
directly (obs[46:52]=box−grip & target−box, obs[31:34]=gripper_pos; 9 dims, NOT routed
through the JEPA encoder) — did **NOT** let the learned H-JEPA hierarchy cross
competence: real success stayed **0.000** through 1.5M env-steps (n=64/eval, 10 evals),
best_eval_success=0.000. The HL stayed faithful (latent subgoal on the JEPA latent;
no decoder; EMA-target+VICReg world model; 2-level k=150; non-generative). The latent
never collapsed (eff_rank 12–25/256, SimNorm code-entropy 0.62–0.83, JEPA pred-loss
0.205→0.165 decreasing). GPU0 only; Director GPUs + mahjong untouched.

## THE SANITY GATE FAILED — and it failed BELOW perception/action, at the CONTROLLER
The whole point of this run was to make reach RELIABLE (osc-only learned reach was only
~0.031). It did not become reliable — and the reason is now isolated one layer deeper:
**even a hand-coded scripted controller with PERFECT raw geometry cannot reliably close
the final ~0.12 m to the 0.012 m contact threshold under this OSC/IK controller.**

| policy | action | LL perceives | reached (n≥64) |
|--------|--------|--------------|----------------|
| random | joint-torque (orig baseline) | — | 0.000 |
| random | OSC/Cartesian | latent | 0.016 |
| scripted "move ee→cube" | OSC/Cartesian | **raw geo (direct)** | **0.094–0.172** (best-dist frac<0.012 m = 0.172; frac<0.03 m = 0.32; mean best-dist ≈ 0.118 m) |
| LEARNED H-JEPA LL (osc-only, latent) | OSC | latent | 0.031 best |
| **LEARNED H-JEPA LL (osc + RAW-GEO)** | OSC | **latent + raw geo** | **0.016 best, 0.000 final** |

The scripted raw-geo controller tops out at ~17% contact and plateaus ~0.12 m from the
cube — the OSC/DLS-IK + rate-limited gripper cannot reliably achieve fine sub-1.2 cm
contact. So "reliable reach" was never actually available to hand to the hierarchy; the
gate is **NOT cleared.** The learned LL, with the same geometry the script had, did no
better (0.016 best) — it is not a perception failure (geometry was supplied raw) and not
an action-space failure (OSC is feasible); the contact precision of the primitive itself
is the ceiling.

## The LL-input change (diff vs run_hjepa_osc.py)
- `raw_geo(o) = jnp.take(o, [46,47,48,49,50,51,31,32,33])` — selected DIRECTLY from obs,
  bypassing the JEPA encoder.
- LL actor/critic input width `2*LD → 2*LD + 9` (z | g | raw_geo). `Pi`/`QEnsemble`
  are compact MLPs that infer input dim, so only the init zeros and call-sites changed:
  `ll_act`, `ll_act_det`, `ll_loss` (rg=raw_geo(o), rg2=raw_geo(o2)), the eval-rollout
  `ll_act_det(...,raw_geo(o))`, and the collect-loop `ll_act(...,raw_geo(st.obs),...)`.
- **HL UNCHANGED**: still `hl(z)→g` latent subgoal on the JEPA latent. JEPA encoder /
  jumpy predictor / EMA target / VICReg(var+cov) / 2-level k=150 / no-decoder / non-
  generative HL — all preserved. Reach SHAPING (reach_w=30, reach_pot_w=2) is LL-only
  r_int, identical to the osc run (faithfulness intact).
- Script: `scripts/run_hjepa_osc_rawgeo.py` (backup `.py.bak_rawgeo`); also pulled to
  the paper repo as `run_hjepa_osc_rawgeo.py`.

## Numbers — full run (from disk: osc_rawgeo_seed0/metrics_seed0.json; n=64/eval)
| step  | succ | reached | ret  | eff_rank | code_ent | Lpi | collapse? |
|-------|------|---------|------|----------|----------|-----|-----------|
| 150k  | 0.000| 0.000   | 421  | 24.5     | 0.82     | 82  | no |
| 300k  | 0.000| 0.000   | 513  | 17.8     | 0.70     | 97  | no |
| 450k  | 0.000| 0.000   | 189  | 15.8     | 0.67     | 88  | no |
| 600k  | 0.000| 0.000   | 219  | 13.6     | 0.64     | 79  | no |
| 750k  | 0.000| 0.000   | 213  | 13.9     | 0.66     | 73  | no |
| 900k  | 0.000| 0.000   | 1379 | 13.8     | 0.63     | 69  | no |
| 1.05M | 0.000| **0.016**| 191 | 14.5     | 0.62     | 64  | no |
| 1.2M  | 0.000| 0.000   | 1357 | 16.2     | 0.69     | 60  | no |
| 1.35M | 0.000| 0.000   | 1326 | 13.8     | 0.69     | 57  | no |
| 1.5M  | 0.000| 0.000   | 554  | 11.9     | 0.62     | 54  | no |

best_eval_success=0.000, best_eval_reached=0.0156, final_eval_success=0.000. The LL
EXPLOITS the dense reach reward (eval return spikes to 1326–1379 when it surfs toward the
cube), but the achieved contact rate is ≈0 — exactly the osc-only signature. Latent
healthy throughout; JEPA pred-loss 0.205→0.165 (decreasing). 9.3M updates, 3073 s wall.
Cross-check: a separate env-reward-off, strong-potential reach-isolation probe (env_rew_w
0, reach_pot_w 5) also held reached=0.000 through 120k — so the floor is not a reward-
weighting artifact.

## Conclusion (3 lines) — closes Bet-2
1. NULL on success (0.000, n=64×10 through 1.5M) **even with BOTH** a feasible OSC action
   AND raw ee-cube geometry handed straight to the LL — and, decisively, the SANITY GATE
   itself failed: a scripted controller with the same raw geometry reaches only ~0.17 and
   plateaus ~0.12 m from the cube, so reliable reach was never actually achievable to hand
   to the hierarchy under this OSC/IK primitive.
2. The Bet-2 wall now resolves into THREE stacked layers: (a) action space — fixed by OSC
   (reach feasible); (b) perception of the primitive — fixed here (LL sees raw geometry,
   not just the geometry-losing JEPA latent), which removes the osc-only "latent doesn't
   preserve ee-cube geometry" excuse; yet (c) the OSC/DLS-IK + rate-limited-gripper
   CONTACT PRIMITIVE cannot reliably close to <1.2 cm, so reach stays ~0–0.17 and never
   chains to grasp/lift/place. The learned latent hierarchy is never exercised on a
   competent reliable primitive.
3. Honest paper finding (which of the two outcomes): this is the **NULL** branch — "even
   handed a feasible action channel AND the raw primitive geometry, the learned latent
   H-JEPA hierarchy cannot compose grasp/lift/place," but the binding limit is NOT the
   hierarchical composition and NOT the perception/action layers we fixed — it is the
   underlying contact PRIMITIVE's precision. Skill-options (0.215) wins because it INJECTS
   competent, reliable reach/grasp primitives; H-JEPA would need the same reliable primitive
   (better contact controller or primitive pretraining), not just a feasible action space
   and raw geometry. The H-JEPA stack stays healthy (no collapse, predictor learns)
   throughout, so this is a primitive-competence limit, not a representation-learning one.
