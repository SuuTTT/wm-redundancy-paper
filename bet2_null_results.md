# LeCun Bet-2 (learned/generative hierarchy) — NULL arm, verified results

All numbers harvested 2026-06-29 from b3060b (deterministic real-success eval,
`box_target ≥ 0.9`, the same Protocol-A metric behind the skill-options 0.215 / flat-TD-MPC2 0.0
baselines). PandaPickCube. Source paths on b3060b given per row; copy here so the paper table
`tab:bet2null` is filled from disk, not memory.

## Peak real success (the Bet-2 table)

| arm | hierarchy type | generative? | peak real success | budget | source (b3060b) |
|---|---|---|---|---|---|
| Skill-options 2-level | injected analytic phases | no | **0.215** (max 0.242) | — | `exp/hierarchy_sol/HIER_VERDICT.json` |
| Flat TD-MPC2 | none (self-predictive) | no | 0.0 | 15.6M | hierarchy campaign |
| **DreamerV3 ("Director" stand-in)** | single-level latent planner | **yes (decoder)** | **0.000** (3/3 seeds, maxsucc=0 each) | ≥300k–525k | `helios-rl/exp/tdmpc_glass/PandaPickCube_dreamergen/seed_{1,2,3}_realsuccess.csv` |
| **H-JEPA reactive HL** | learned non-gen 2-level | no | **0.000** (best=0, final=0; latent_collapsed=false) | 2.0M | `exp/hjepa/extended_seed0/smoke_summary_seed0.json` |
| **H-JEPA latent-MPPI HL** | learned non-gen 2-level + planning | no | **0.000** (every eval, n=64) | 1.5M | `exp/hjepa/mppi_seed0/VERDICT.md` |
| **H-JEPA + dense LL-reach shaping** | learned non-gen 2-level | no | **0.000**, `reached`=0 | 1.5M | `exp/hjepa/llreach_seed0/VERDICT.md` |

## Mechanism (why every learned/generative hierarchy is 0.0)

The low-level **reach primitive** is the wall. Reach on PandaPickCube is joint-space IK to a
**<0.012 m** contact target from ~0.3–0.4 m. Evidence it is genuinely unlearnable from scratch
in budget (not a reward-shaping or HL-planning gap):
- A **random policy flips `reached_box` 0/512 times** (probe in the llreach run) — the contact
  event is essentially never hit by chance.
- Even a **dominant, correctly-wired dense reach reward** (`reach_w=30` on the LL reward only,
  reach signal = ‖obs[46:49]‖ = the exact `reached_box` quantity) leaves `reached`=0 through 1.5M.
- The LL acts on a SimNorm latent trained only by the JEPA predictive loss, which does not
  preserve fine end-effector–cube geometry.

So skill-options reaches 0.215 **because it injects** the reach/grasp/lift/place primitives the
learned LL cannot bootstrap. Counter to LeCun's Bet 2 (that *learned* hierarchical abstraction is
the lever): on this task the lever is **injected** structure, and even that is capped by place-phase
physics (method-vs-task: same checkpoints → 0.55–0.64 under position-only success).

## Honest scope / caveats
- DreamerV3 is a **single-level generative planner**, not a true Director 2-level latent-goal
  hierarchy (none existed / not buildable in-window). So this rules out "generative WM *alone* breaks
  the plateau," not a generative 2-level hierarchy.
- DreamerV3 budget 300k–525k (vs flat TD-MPC2 0.0 at 15.6M); it is far more sample-efficient and its
  WM was fully converged (wm_loss flat ~1.36) at 0.0 — a strong NULL, not under-training.
- H-JEPA results are seed0 (single seed) but corroborated across **three independent HL
  architectures** (reactive, latent-MPPI, dense-reach-shaped), all flat 0.0.

## FAIR Bet-2 test — H-JEPA with feasible OSC/Cartesian LL action (2026-06-29)
Gave the LL a Cartesian ee-displacement action via the existing IK controller (`hl_pickcube/osc_env.py`,
`scripts/run_hjepa_osc.py`; `exp/hjepa/osc_seed0/VERDICT.md`). Result: **NULL on success, positive sub-signal on the primitive.**
- Sanity gate PASS: reach `reached_box` went 0/512 (joint-torque, random) → scripted 0.31, learned best 0.031 — the action space WAS the first wall.
- success = **0.000** at all 10 evals (1.5M, n=64); best learned reached only 0.031; latent healthy (eff_rank 10-25, no collapse).
- INTERPRETATION (two-layer wall): fixing the action space is **necessary but not sufficient**. The learned LL acts on JEPA
  latents that **do not preserve fine ee-cube geometry**, so reach is feasible-but-unreliable (~3%) and never chains to grasp/lift/place.
  The remaining lever (untested) = give the LL raw-geometry input or pretrain the primitive — not just a feasible action channel.

## FINAL Bet-2 test — OSC action + raw-geometry LL input (2026-06-29) — INVESTIGATION CLOSED
Combined both fixes: OSC/Cartesian LL action AND raw ee-cube geometry (obs[46:51,31:33], 9-dim) fed directly to the
LL actor/critic (bypassing the JEPA encoder); HL still on the JEPA latent. (`scripts/run_hjepa_osc_rawgeo.py`,
`exp/hjepa/osc_rawgeo_seed0/VERDICT.md`.) Result: **NULL — success 0.000 through 1.5M (n=64), reached best 0.016.**
DECISIVE diagnostic: a SCRIPTED controller with perfect raw geometry reaches only ~0.094-0.172 and plateaus ~0.12m
from the cube — the OSC/DLS-IK + rate-limited gripper CANNOT reliably close to the <1.2cm contact threshold. So the
Bet-2 wall is THREE stacked layers: (a) action space [fixed by OSC], (b) perception of the primitive [fixed by raw-geo
input], (c) the CONTACT PRIMITIVE's sub-cm precision [the actual binding limit]. Even handed a feasible action channel
AND raw geometry, the learned latent hierarchy cannot compose grasp/lift/place — limit is primitive contact reliability,
not hierarchical composition or representation. Skill-options 0.215 wins by injecting a competent reliable primitive.
Remaining lever (multi-day, untested): a better contact controller or primitive pretraining — NOT learned hierarchy.

## ⚠⚠ CRITICAL CAVEAT — CRITIC BUG CONFOUNDS THE PANDAPICKCUBE H-JEPA NULLS (2026-06-29)
A positive-control run of H-JEPA on a trivial-LL long-horizon 2D point-maze (`exp/hjepa_navctrl/VERDICT.md`)
did TWO things: (1) VALIDATED the machinery — 2-level H-JEPA reaches **1.00** success with stable/raw obs
(= flat 1.00), so the hierarchy + LL controller are SOUND and the PandaPickCube 0.0 is consistent with the
contact-primitive wall, NOT a broken architecture; (2) FOUND A LOAD-BEARING BUG in the H-JEPA LL TD-loss:
`td_tgt` was shape `(B,1)` → `two_hot(td_tgt)` `(B,1,101)` → `soft_ce(q (B,2,101), tgt)` broadcast to a
`(B,B)` batch-outer-product, i.e. the critic learned cross-entropy against OTHER samples' targets. Fixing to
a 1-D `td_tgt` took the nav controller from 0% → convergence. **This same pattern exists in the original
`run_hjepa.py` on b3060b** (`q2=min(...,keepdims=True)`→`td_tgt (B,1)`→`two_hot[:,None,:]`), so the earlier
PandaPickCube H-JEPA runs (reactive @2M, latent-MPPI @1.5M, llreach, OSC, OSC+rawgeo) likely had a BROKEN
CRITIC. **CONSEQUENCE: those NULLs are CONFOUNDED — do NOT cite them as clean "learned hierarchy fails"
until re-run with the fixed critic.** What still stands independent of the bug: (a) the contact-primitive
physics wall — a SCRIPTED controller with perfect geometry reaches only ~0.12-0.17, can't close <1.2cm
(env physics, no critic involved); (b) the SimNorm-JEPA latent collapses on simple nav (eff_rank≈0,
ent≈0; raw-obs 1.0 vs latent 0.6) — a real representation weakness. RE-RUN WITH FIX IS IN FLIGHT.

## ⚠⚠⚠ REPRODUCIBILITY CORRECTION — nav positive control is NOT reproducible (2026-06-29)
A VICReg/EMA anti-collapse sweep on the nav point-maze (`exp/hjepa_navctrl/collapse_sweep_VERDICT.json`) found:
(1) anti-collapse DOES restore latent health (vic10_g20: eff_rank 0→~4/32, code-entropy 0→0.62, 0/3 collapsed) but
task success stays ≤0.05 — REPRESENTATION HEALTH IS DECOUPLED FROM CONTROL SUCCESS on this arm (Pearson succ–eff_rank
−0.18). (2) CRITICAL: the earlier nav positive-control headline (raw=1.00, latent=0.609) **does NOT reproduce** —
re-running identical commands gives ~0.0–0.16 for ALL arms incl. the raw-obs identity-encoder "ceiling"; two
byte-identical seed-0 raw runs gave 0.031 vs 0.156. The point-maze controller/training is HIGH-VARIANCE /
nondeterministic. CONSEQUENCE: the "H-JEPA machinery validated (1.0 on nav)" claim is RETRACTED — not established.
NET state of the learned-hierarchy (Bet-2) arm: NULLs CONFOUNDED by the critic bug AND the positive control
NON-REPRODUCIBLE → **the Bet-2 arm is NOT paper-ready**; needs a rigorous multi-seed, determinism-controlled redo
(fixed critic + ≥5 seeds + variance reported) before any "learned hierarchy fails / works" claim. What still stands
independent: DreamerV3 NULL (no shared critic), the contact-primitive physics wall (scripted ≤0.17, no learning),
and the behavioral-abstraction taxonomy + escape frontier (entirely separate code path, deterministic, solid).

## ✅ DE-CONFOUND RESULT — PandaPickCube NULL HOLDS with the FIXED critic (2026-06-29)
Re-ran with the corrected critic (`exp/hjepa/criticfix_{reactive,osc}_seed0/`): **criticfix REACTIVE done @2M = succ 0.000 / reached 0.000** (fix active, Lpi -5→-13.5); **criticfix OSC @1.2M(of 1.5M) = succ 0.000 / reached ~0.016** (≈ the buggy 0.03; finishing). CONCLUSION: the PandaPickCube H-JEPA NULL is **NOT the critic bug** — it HOLDS with a correct critic AND a feasible OSC action. The bug mattered on the nav task (0→convergence) but is irrelevant on PandaPickCube where the **contact-primitive physics wall dominates** (scripted ≤0.17, can't close <1.2cm). So the headline "learned hierarchy = 0.0 on PandaPickCube, bottlenecked by the low-level contact primitive" is **de-confounded and valid**. CAVEAT THAT REMAINS: we have NO reproducible positive control that H-JEPA works anywhere (nav non-reproducible/high-variance), so frame the NULL as task-specific-bottleneck, NOT "learned hierarchy fundamentally fails." Paper-ready claim = the former.

## ✅✅ POSITIVE CONTROL ESTABLISHED — Bet-2 arm now PAPER-READY (2026-06-29, supersedes the non-repro caveat)
Rigorous redo (`exp/hjepa_navctrl/posctrl_rigorous_VERDICT.json`): the earlier non-reproducibility was ROOT-CAUSED and FIXED — it was XLA GPU autotuning + TF32 matmul nondeterminism (NOT RNG); with `--xla_gpu_deterministic_ops=true` + `jax_default_matmul_precision=highest`, same-seed full runs are BIT-IDENTICAL (verified A==B). Also the prior "raw caps at 0.6" was UNDERTRAINING (utd=1→4 → raw reaches 1.0). Rigorous comparison (4 arms × 8 seeds, utd=4, 250k, deterministic eval n=256 @ fixed seed):
- flat_raw (identity-enc ceiling): **0.971 ± 0.048**
- **2-level H-JEPA raw: 0.941 ± 0.137** (7/8 seeds ≥0.996, one outlier 0.535)
- flat SimNorm-JEPA latent: 0.503 ± 0.039
- 2-level H-JEPA latent: 0.419 ± 0.170
**VERDICT GO:** the 2-level H-JEPA RELIABLY reaches the raw-obs ceiling on a trivial-LL long-horizon nav task → the H-JEPA MACHINERY IS VALIDATED (the earlier "machinery not validated" retraction is REVERSED). Therefore the PandaPickCube learned-hierarchy NULL is CLEANLY attributable to the contact-primitive wall, not a broken implementation. Secondary (separable, real) findings: the faithful SimNorm-JEPA LATENT collapses (eff_rank≈0, 8/8) capping both latent arms ~0.4-0.5; reactive HL ≈ flat on a trivial LL (hierarchy neither helps nor hurts there). NET Bet-2 story (now fully paper-ready): learned/generative hierarchy is validated to WORK where the low-level primitive is feasible (nav 0.94), and is 0.0 on PandaPickCube purely because the contact primitive is unlearnable (scripted ≤0.17) — counter to LeCun's bet that LEARNED hierarchical abstraction (not the primitive) is the lever; the lever is a competent low-level primitive (injected, as skill-options does → 0.215).

## 🟢 SOLVE ATTEMPT round 1 — H-JEPA latent-HL over COMPETENT skills CROSSES competence (2026-06-30)
Goal = SOLVE PandaPickCube with H-JEPA. Gave H-JEPA's learned latent HL (SimNorm enc, jumpy predictor pred(z,d)→ẑ vs EMA+VICReg, learned value V, CEM planning in latent space — faithful, no decoder, fixed critic) the COMPETENT analytic skill primitive as LL (param_controller.ParamPick, unchanged). Code `hl_pickcube/hjepa_options_solve.py`, results `exp/hjepa_solve/round1/`. RESULT: **H-JEPA latent-HL = 0.117 final / 0.145 peak (n=512) @8M steps** vs flat 0.0 / static base d=0 0.039 / reactive-HL 0.215 / PPO 0.66. → the learned latent planner does REAL work (~3× static base, ~2/3 of reactive at ~30% of its env-steps) and CROSSES competence. Per-phase: reached 0.89, grasp 1.00, lift 0.80, **place 0.39 / hold 0.20 = the cap** (bt_max 0.42). PLATEAU MECHANISM (round-2 lever): JEPA predictor loss ROSE 0.05→0.78 as latent eff-rank expanded 9→18 (non-stationary latents) → CEM exploits predictor error; dense bt_max rises to 0.42 but success stalls 0.06-0.12. UPSHOT for the paper: this UPGRADES the Bet-2 story — learned latent hierarchy DOES work on manipulation once given a competent low-level primitive (0.117, crossing competence), confirming the "lever = competent primitive" thesis with a POSITIVE result, not just the NULL. Round-2 levers (ranked): (1) trustworthy planning (real-rollout anchor / value pessimism to kill model exploitation); (2) stabilize predictor (shorter k-jumps, stronger stop-grad, slower encoder); (3) place-phase option-param curriculum. Target: push toward/past reactive 0.215.

## SOLVE round 2 — planning fixes buy efficiency/stability, NOT a higher ceiling (2026-06-30)
Round 2 (trustworthy planning: value-pessimism + real-rollout anchor; predictor stabilization; place emphasis; warm-start from round1) `exp/hjepa_solve/round2/`: **peak 0.1172 = SAME as round 1**, but reached it at ~1.8M steps vs round1's 8M (~4× sample-eff) with Lpred STABLE (0.65, vs round1 blow-up to 0.78) and eff_rank stable (no 9→18 drift). So the fixes worked as designed but the CEILING is unchanged — place-phase-limited (ph_place ~0.40, success ~0.117). KEY DIAGNOSIS: reactive-HL gets 0.215 over the SAME analytic skills, but H-JEPA's latent planner extracts only 0.117 → the gap is the LATENT PLANNER under-optimizing place params (its value/model misranks place candidates), not the primitive alone. Two levers for round 3+: (1) close 0.117→0.215 by improving the latent planner's place-phase value/model accuracy (or warm-start/distill from the reactive solution); (2) push past 0.215 with a better upright-place primitive. Multi-seed NULL CONFIRMED at n=5 (seeds 1-4 all 0.000 @2M + seed0) — robust from-scratch NULL.

## 🔑 ORACLE CEILING — the analytic LL caps the solve at 0.305, NOT the planner (2026-06-30, round 2 final)
Round 2 verdict NO-GO (H-JEPA 0.074 plan / 0.094 anchored, tied w/ round-1 0.117; did NOT reach reactive 0.215). The planner fixes verifiably worked (Lpred stable ~0.65 vs round-1 blow-up to 0.78; eff_rank stable; place-phase entry 0.33→0.49) but binary box_target≥0.9 did not follow. DECISIVE ORACLE DIAGNOSTIC (`exp/hjepa_solve/round2/ORACLE.json`): brute-force search of 384 candidate option-params d DIRECTLY in the real env (n=128) saturates at **0.305 success** (bt_mean 0.73; only bt_p90=0.94 crosses 0.9). → **0.305 is the HARD CEILING of the analytic-skill option parameterization itself** — no d reliably lands box_target≥0.9 for the median env. So the binding constraint is the **analytic LOW-LEVEL place PRECISION, not the H-JEPA planner** (reactive 0.215 and H-JEPA 0.117 both sit below this 0.305 oracle; H-JEPA reaches ~30-40% of its own ceiling). IMPLICATION: even a perfect HL planner over these skills tops at 0.305 ≪ PPO 0.66 — to solve Panda you must raise the LOW-LEVEL ceiling (orientation-aware/upright place controller, e.g. param_controller_oriaware_v2), THEN plan. This is the cleanest proof of the campaign thesis: the lever is a competent low-level primitive, not learned hierarchical planning. Round 3 (in flight, GPU2/3) swaps in a better place controller — ph_place execution jumped 0.40→0.96, but need its ORACLE ceiling + whether H-JEPA extracts it. (NOTE: agent-spawn limit hit, resets 7:50am UTC; round 3 detached, harvesting via SSH.)

## ✅✅✅ SOLVE round 3 — GO: H-JEPA SOLVES PandaPickCube competitively (2026-06-30, FINAL)
H-JEPA latent planning, warm-started from the competent reactive solution + real-rollout anchor, BEATS the hand-tuned reactive hierarchy. Deterministic real-success (box_target≥0.9):
- **H-JEPA round-3 = 0.289 (n=512), peak 0.297 @10.9M steps** (main arm); place-arm 0.279.
- vs round-2 H-JEPA 0.117 → +0.172 (2.5×); vs reactive-HL 0.227(n512)/0.246(n256) → +0.04-0.06 ABOVE reactive; vs flat 0.0.
DECISIVE LEVER = reactive warm-start: seeding CEM mean + anchor baseline + buffer from the trained reactive pi_hi (`exp/tdmpc_glass/hl_options/seed0/best_theta.npy`) lifted latent **plan-only 0.05→0.207** (planner now extracts ~full primitive value on its own); the real-metric anchor (frac_plan_kept 0.47-0.55) adds +0.04-0.06 by keeping per-env better of {reactive, refined-plan}. HONEST CAVEATS: margin over reactive is modest + partly anchor keep-better-of-two; latent plan ALONE (0.207) ≈ reactive floor not above. Bounded at ~0.30 by analytic LL place-POSITION precision (oracle 0.305; ph_place already 0.95, ph_hold 0.46 is the cap — NOT orientation, so oriaware controller inapplicable on upright-target PandaPickCube). To exceed 0.30 toward PPO 0.66 needs a more precise low-level place-position controller (LL problem, not H-JEPA). Files: `exp/hjepa_solve/round3/{RESULTS.json,WRITEUP.md,best_params.npz}`, solver `hl_pickcube/hjepa_options_solve_r3.py`. NET ARC (paper headline): H-JEPA on Panda went flat-0.0-NULL → de-confounded(critic fix) → machinery-validated(nav 0.94) → crosses competence given a competent primitive(0.117) → SOLVES competitively, beating the hand-tuned hierarchy(0.289), with the residual gap to PPO cleanly attributed to low-level contact/position precision via the oracle. The lever throughout = a competent low-level primitive + warm-start, NOT learned hierarchical abstraction alone — a positive, honestly-scoped resolution of LeCun Bet-2 on this task.

## SE-vs-VICReg JEPA latent on Panda — NULL (2026-06-30, well-controlled)
Tested the "glass"/SE idea to fix the H-JEPA geometry gap: SE-regularize the JEPA latent (selib min-2D-SE on a kNN latent graph) vs MATCHED VICReg. `exp/se_jepa_panda/`. USING_SE.md flags this regime (RL state/skill abstraction) as one where SE typically does NOT help. Mechanism probe (frozen encoder→ridge, held-out R², n=3): VICReg ee→cube R²=0.987 / box→target 0.933 / eff_rank 30.8; SE(real graph) 0.736 / 0.569 / 12.9; **SE-on-random-graph control 0.986** (≈VICReg). DECISIVE: SE's community structure (collapsing eff_rank 31→13) destroys the continuous ee→cube manifold — not an artifact (random-graph control recovers). LAM_SE λ-sweep: SE best only TIES VICReg (when weighted to near-inert); never beats. VERDICT NULL — SE community-bucketing is the wrong bias for the continuous geometric regression manipulation needs; VICReg's rank-maximizing term is better. Differentiable 2D-SE validated vs selib to 3e-7. → For pushing Panda past the 0.305 oracle, the lever is low-level place-POSITION precision, not the latent. (Nav-collapse SE testbed running separately on b3060.)

## SE-vs-VICReg on NAV latent collapse — GO, but the lever is RELATIONAL anti-collapse, not SE (2026-06-30)
Second SE testbed (`b3060:exp/se_nav/`): the nav point-maze where SimNorm+VICReg COLLAPSES (eff_rank≈0, success ~0.53 vs raw ceiling 0.979). Matched arms n=4, deterministic n=256, all USING_SE.md controls. Results: raw 0.979±0.04; VICReg-default 0.530 (4/4 collapsed); **VICReg-STRONG (fully un-collapses, eff_rank 3.11, 0/4) STILL fails 0.442±0.37** (reproduces anti-collapse→success DECOUPLING); **SE(se_w=5) 0.906±0.066, 0/4 collapsed** (~93% of ceiling); **SE random-partition control 0.823±0.054 (overlapping CI with real SE).** TWO findings: (1) GO — the SE-style objective fixes a collapse VICReg PROVABLY cannot (strong-VICReg control). (2) CAVEAT (decisive random-partition control): the benefit SURVIVES randomizing the SE partition → the lever is the **relational/pairwise affinity-graph coding-cost** (control-relevant geometry), NOT the structural-entropy community partition. Differentiable 2D-SE validated vs selib to 1e-5. COMBINED TWO-TASK SE VERDICT: SE *per se* does NOT help JEPA latents (Panda NULL; nav benefit partition-independent) — but a RELATIONAL anti-collapse objective beats per-dim VICReg at fixing collapses. The real lever = relational/pairwise anti-collapse. Next: isolate it (relational term w/o SE) vs SE vs VICReg to confirm.

## ✅ THE LEVER ISOLATED — a 1-line uniformity loss beats VICReg; SE machinery not needed (2026-06-30)
Ablation (`b3060:exp/se_nav_lever/`, n=4, deterministic n=256): raw ceiling 0.979; VICReg 0.530 (4/4 collapsed); SE-full 0.906; **minimal UNIFORMITY loss (Wang&Isola pairwise repulsion, NO graph/partition/tree) 0.954±0.046 (0/4 collapsed)**; knnrep (graph, no partition) 0.887. Uniformity beats VICReg (non-overlapping CI) and matches/edges SE-full with the SIMPLEST possible term. CLEAN CONCLUSION: the lever for collapse-prone control latents is **pairwise relational anti-collapse**; you need neither structural entropy (tree/partition) nor even a kNN graph — a one-line uniformity loss suffices. Resolves the random-partition caveat (any relational anti-collapse works). Determinism cross-check: raw/VICReg/SE reproduced prior se_nav per-seed bit-identically. → Actionable JEPA improvement (for the collapse regime); SE itself = NULL (Panda) / redundant (nav). Generalization test (uniformity vs VICReg on a DMControl collapse task) launching on b3060.

**2nd GEOMETRIC TASK — C-maze detour (`--walls c`), partial/directional GO (2026-06-30):** to firm the geometric-helps cell beyond the single point-maze. Matched arms **n=8** (firmed up from n=4), deterministic fixed-eval: raw 0.318±0.345 (eff_rank 2.26); **VICReg 0.041±0.018 (eff_rank 0.50, 8/8 collapsed)**; **uniformity 0.100±0.102 (eff_rank 7.16, 0/8 collapsed)**. The ROBUST signal is CLEAN: uniformity **reliably PREVENTS the collapse VICReg suffers — 0/8 vs 8/8, eff_rank 7.16 vs 0.50** — and ~doubles success over VICReg (0.100 vs 0.041), directionally replicating the point-maze. BUT success-recovery is only ~31% of the raw ceiling (vs the point-maze's ~97%) and noisy (huge seed-variance; the C-maze detour is harder, even raw is 0.32±0.35). HONEST split: the ANTI-COLLAPSE MECHANISM replicates cleanly on a 2nd geometric task (uniformity uncollapses where VICReg fails, every seed); the SUCCESS-magnitude effect is partial/task-dependent (the clean 0.53→0.95 separation remains the point-maze). `b3060:exp/se_nav_geo2/` (n=8/arm).

## Uniformity does NOT generalize to value-based control (DMControl) — regime-specific (2026-06-30, PRELIMINARY n=1)
Generalization test (`b3060:exp/unif_dmc/`): CheetahRun, collapse-prone latent_dim=16 SimNorm. Matched arms (n=1 in the ≥200k window — thin, confirm pending): default MPPI=118 / pi=61 / eff_rank 3.05 / value_r2 0.74; **+uniformity MPPI=34 / pi=32 / eff_rank 6.83 (health UP) / value_r2 0.21 (DOWN)**; +VICReg MPPI=16 / value_r2 0.52. So on VALUE-based control, uniformity IMPROVES raw latent health but HURTS return AND value-decodability (0.74→0.21) — the OPPOSITE of nav. MECHANISM: uniformity's pairwise spreading pushes apart states that should be value-close → destroys value-sufficiency. UNIFYING INSIGHT (mirrors Panda SE NULL): the right anti-collapse term is DOWNSTREAM-DEPENDENT — relational/uniformity for goal-conditioned geometric latents (nav: 0.954 vs 0.530), value-sufficiency for value-based control (DMControl), SE-community for neither continuous case. No universal anti-collapse winner. NEEDS multi-seed confirm (n=1 now). Meanwhile the CI on the Panda H-JEPA solve GO is firming: seeds 0/1/2 = 0.289/0.285/0.26, all ≥ reactive 0.246 (seed3 pending) → "H-JEPA beats hand-tuned" reproducible.

### CONFIRMED at n=3 (2026-06-30) — and 2nd-task generalization launched
The CheetahRun sweep completed all 9 seed-dirs (3 arms × 3 seeds); the aggregate had silently frozen at n=1 because `aggregate.py` crashed on the numpy-2.x `np.trapz`→`np.trapezoid` rename (patched + re-aggregated, no GPU). Firmed n=3 (AUC = mean MPPI-return rate over the whole run): **default 58.9±29.8, value_r2 0.59±0.51**; **uniformity 36.6±24.8, value_r2 0.33±0.19**; vicreg 56.3±6.4, value_r2 0.48±0.38. Ordering is **identical to the n=1 preliminary — uniformity is WORST on both control axes** (return AUC and value-decodability) even though it WON on nav. So the downstream-dependent claim holds at n=3. HONEST CAVEAT: high seed-variance → the arms are NOT CI-separated; this is a consistent directional confirm (n=1→n=3), not a separation. To strengthen beyond a single DMControl task, a 2nd-task generalization sweep launched on b3060 (`exp_unif_gen2.sh`): WalkerWalk + FingerSpin, same default/unif/vicreg × seeds1-3, L=16 collapse pressure, 4-wide scheduler.

### GENERALIZES across 3 DMControl tasks (2026-06-30, gen2 sweep DONE n=3/arm/task)
WalkerWalk + FingerSpin completed 9/9 each. Return-AUC (mean MPPI-return rate, the actual control-performance axis):
**CheetahRun** default 58.9 / **unif 36.6** / vicreg 56.3; **WalkerWalk** default 293.8±14.7 / **unif 89.9±2.3** / vicreg 172.3±67.7; **FingerSpin** default 249.4±148.6 / **unif 171.8±52.6** / vicreg 289.7±27.4. **Uniformity is the WORST arm on all 3 tasks on the return axis** — and on WalkerWalk it is **CI-SEPARATED** (unif 89.9±2.3 vs default 293.8±14.7, non-overlapping). So the downstream-dependent claim is now firm across 3 tasks, not 1: uniformity (Wang&Isola pairwise repulsion) WINS on goal-conditioned geometric nav (0.530→0.954, CI-separated) and LOSES on value-based DMControl control (3/3 tasks). value_r2 (decode axis) is noisier — uniformity is worst on Cheetah+Walker+Finger for the *return*, but vicreg's value_r2 flips task-to-task (best on Walker 0.604, worst on Finger 0.065), so the clean, robust signal is the RETURN axis. **4th-task EXCEPTION (2026-06-30): CartpoleSwingup** — at matched 150k steps default 155.1±22.8 vs unif 162.4±9.3 (n=3, overlapping CI) — uniformity is NOT worst here, it ties/slightly edges default. So "uniformity hurts value-based control" is a **3/4-task majority/directional result (CI-separated only on WalkerWalk), not universal**; on the easiest swing-up task the harm vanishes. Report it as task-dependent, not a law. DECISIVE GENERALIZED CLAIM: there is no universal anti-collapse term; relational/uniformity repulsion helps geometric latents and hurts value-control latents. Files: `b3060:exp/tdmpc_glass/{WalkerWalk,FingerSpin}_unif_dmc_*` phase CSVs; `exp/unif_dmc/exp_unif_gen2.sh`.

### Panda H-JEPA solve — CI n=7 (2026-06-30, round5_ci DONE)
6 CI seeds + the n=512 headline. Peaks: 0.359/0.371/0.348/0.367/0.375/0.363 → **mean 0.364±0.010**; finals 0.316/0.320/0.312/0.328/0.346/0.295 → **mean 0.320±0.017**; headline run 0.367. The 0.367 Panda solve REPRODUCES tightly (peak 0.364±0.01), all ≫ hand-tuned 0.244. `b3060b:exp/hjepa_solve/round5_ci/seed{1..6}/hjepa_out/RESULTS.json`.

### Round 6 — different-contact oracle probe: NULL for breaking past ~0.37 (2026-06-30)
Tested whether a different ANALYTIC contact primitive (grasp lower on the cube / firmer grip / obj-vs-ori gain tweaks) can raise the oracle past the ~0.40 GO bar toward PPO 0.66. n=256 deterministic real-success. Best variant (r6 obj1.0/ori1.0) oracle **0.3633** vs r5 control 0.3242 — only +0.04, still well under 0.40. DECISIVE diagnostic (all 4 variants): SUCCESS cases keep cube tilt ~2.8–2.9° while FAIL cases sit at **15–18°** — grip-geometry/gain tweaks cannot break the tilt wall. CONFIRMS the analytic-skill family caps at ~0.36 on **contact physics** (tall 4×4×6cm cube in a 2-finger parallel-jaw grip); closing to PPO 0.66 needs a LEARNED residual or non-parallel-jaw contact, NOT analytic-parameter tuning. `b3060b:exp/hjepa_solve/round6_contact/{VERDICT.md,ORACLE_*.json}`. → Next: learned grasp-place residual on top of the analytic skill (the genuine path past the analytic ceiling).

### Round 7 — LEARNED residual: breaks the analytic ceiling (0.367→0.707) but does NOT beat matched-budget PPO (2026-06-30)
Learned residual `executed = clip(a_opt + alpha·π_res, -1, 1)` (a_opt = analytic r5 skill kept live with its upright servo; π_res = brax PPO on true reward, 77-d obs) at 3 authority levels + MATCHED-harness vanilla PPO. Real-success n=256, box_target≥0.9, deterministic (oracle-validated hand-rolled eval after a brax AutoResetWrapper desync was caught & fixed). FINAL numbers:
| arm | peak succ | final | tilt succ/fail |
|---|---:|---:|---|
| analytic d=0 sanity (=oracle) | 0.082 | 0.082 | 2.5 / 8.8 |
| residual alpha=0.25 (tight) | 0.191 | NaN (diverged ~16M) | 2.7 / 10 |
| residual alpha=0.5 (moderate) | 0.402 | 0.203 (decayed) | 2.2 / 20 |
| **residual alpha=1.0 (full)** | **0.715** | **0.715** | 1.9 / 19 |
| **vanilla PPO (matched harness)** | **0.809** | 0.809 | — |

VERDICT: MIXED — **NULL for a strictly BOUNDED residual, sample-efficiency-GO once the bound is relaxed.** (1) Bounded residual (alpha≤0.5) tops ~0.40 and is UNSTABLE (alpha=0.25→NaN, alpha=0.5 peaks then decays to 0.20) — small corrections can't reliably break the contact wall. (2) Full authority (alpha=1.0) reaches **0.715 ≈ PPO** and learns **~2× FASTER** than vanilla (0.45@9.8M vs vanilla 0.0@16M / 0.457@19.7M) — BUT its asymptote (0.715) is BELOW matched vanilla PPO (0.809), and at alpha=1.0 it is effectively a warm-started-from-analytic policy, not a bounded residual. So the analytic prior buys **sample-efficiency, NOT a higher asymptote** (mild anchor on the ceiling) — textbook [[class-controller-budget-trap]]-aware result, exactly the SYNTHESIS taxonomy (prior = speed lever where it fits; vanilla wins asymptote). MECHANISM (tilt diagnostic, consistent): success cases keep cube tilt ~1.5–2.8°, failures 9–22°; alpha=1.0 drives success-tilt to **1.9° (below the analytic servo's 2.5°)** → the contact wall is **LEARNABLE, not gripper-morphology-limited** — a learner fixes the in-grip tilt given real policy authority. The earlier "PPO 0.66" was under-budgeted; proper same-harness PPO = 0.81. `b3060b:exp/hjepa_solve/round7_residual/{eval_a*.json,eval_vanilla_150.json,sanity_analytic.json,VERDICT.md}`. **n=3 REPLICATION FIRM (2026-06-30) — both axes CI-separated:** residual alpha=1.0 peak **0.716±0.014** (steps→0.45 succ 10.9M±1.9M); vanilla PPO peak **0.810±0.006** (steps→0.45 17.5M±1.9M). (vanilla_s3's earlier 0.000 was a checkpoint-eval artifact — re-eval via eval_ckpts.py gives clean 0.81, NOT a PPO failure; a wait-loop bug that had triggered premature mid-training evals was caught+fixed, finals are on completed 30M ckpts.) NON-OVERLAPPING both ways: vanilla asymptote higher (resid max 0.731 < vanilla min 0.805, ~0.094 deficit); residual faster to competence (resid max 13.1M < vanilla min 16.4M) but the speed edge is **~1.6×, not the n=1 ~2×** (softened on steps-to-0.45). CLEAN CONFIRMED: the analytic prior buys **sample-efficiency (~1.6×) + low variance, NOT a higher ceiling** — textbook SYNTHESIS taxonomy. `round7_residual/{REPLICATION_VERDICT.md,eval_*.json}`.

### Round 8 — residual GENERALIZES to a 2nd Panda task (OpenCabinet): GO (2026-06-30)
Tested whether "prior = speed-not-ceiling" holds beyond PickCube. Feasibility YES — a verified analytic TAMP skill for OpenCabinet already existed (hl_cabinet 3-phase reach+latch→grasp→IK-push, 0.827 at zero training); residual `a=clip(a_tamp+alpha·pi_res)` alpha=1.0, base reward, 7 seeds. Closed a protocol confound: prior vanilla baselines were 1000-step; re-ran a PROTOCOL-MATCHED vanilla PPO (150-step, official hyperparams, n=2) with an identical 150-step/n=256/box_target≥0.9 scorer. RESULT: **asymptote residual 0.9805 (n=7) == vanilla 0.9805 (n=2)** — same structural ceiling (same ~5/256 eps fail both); **sample-eff residual 0.95@~2.3M vs vanilla ~16.4M → ~7× faster**. VERDICT GO — replicates more cleanly than PickCube. UNIFIED 2-TASK STATEMENT: the analytic prior's asymptote is **≤ vanilla** (PickCube 0.716<0.810; OpenCabinet 0.9805=0.9805 tie) and the prior is **markedly faster to competence on both** (~1.6× / ~7×). So "structured prior = sample-efficiency lever, not a ceiling lever" now holds across 2 Panda manipulation tasks; whether the prior ties or sits below the ceiling is task-dependent (saturable shared ceiling → tie). vanilla FIRMED at **n=5** (peaks 0.9805/0.9805/0.9805/0.9883/0.9805 → ~0.982) == residual 0.9805 (n=7): asymptote tie solid. (brax batch-rounding overran vanilla to ~68.8M steps, which only strengthens the tie.) `b3060b:exp/hjepa_solve/round8_generalize/{VERDICT.md,eval_vanilla_s*.json}`.

## Value-aware anti-collapse (valunif) — NULL, even WORSE than uniformity (2026-06-30)
Completes the anti-collapse-taxonomy probe: if uniformity hurts value-based control because it ignores value-structure, does a VALUE-AWARE repulsion (down-weight repulsion for value-similar states) RECOVER? Matched arms, L16 collapse config, return-AUC, now **n=3 on all 3 tasks**: **CheetahRun** default 58.9±12.0 ≫ unif 36.6±10.0 ≫ **valunif 20.6±5.5**; **WalkerWalk** default 293.8±14.7 ≫ unif 89.9±2.3 ≫ **valunif 49.5±7.4**; **FingerSpin** default 249.4±148.6 ≫ unif 171.8±52.6 ≫ **valunif 1.6±0.1** (near-total collapse). valunif is the WORST arm on ALL 3 tasks, below even plain uniformity. VERDICT NULL — value-awareness did NOT rescue it; it hurt MORE. SHARPENS the downstream-dependent claim: for value-based control, the right anti-collapse is **NONE extra** (default wins; SimNorm's built-in pressure + the value gradient suffice) — ANY relational repulsion (uniform OR value-weighted) destroys value-sufficiency. So the taxonomy is: relational/uniformity for goal-conditioned GEOMETRIC latents (nav 0.53→0.95), nothing-extra for VALUE-based control, SE-community for neither. `b3060:exp/tdmpc_glass/{WalkerWalk,CheetahRun}_unif_dmc_*_valunif_*`.

## Round 4 (place/grasp precision) — PARTIAL-GO/NULL: the ~0.30 ceiling is GRASP INSTABILITY (physical) (2026-06-30)
`exp/hjepa_solve/round4/`: yaw-aware place controller (param_controller_r4.py) raised oracle 0.3047→0.3203 (modest) and planner plan-only 0.207→0.242, but merged H-JEPA 0.279 (peak 0.285) ≈ round3 0.289 (within ~0.02 noise) — GO criterion (oracle↑ AND success↑) NOT met. CORE FINDING (instrumented): box_target≥0.9 needs 0.9·pos+0.1·rot≤0.020; in the competent regime only ~19% succeed because the **cube SLIPS inside the grip** during carry/hold (cube_rot_err 0.225→0.43) + position drifts off its mid-trajectory best. This is GRASP INSTABILITY — a physical limit the analytic grasp/cart/wrist knobs cannot overcome. So the analytic-LL family caps at ~0.30-0.32 fundamentally because of grasp stability; H-JEPA extracts ~0.28-0.29 (beating hand-tuned 0.246). Path to PPO 0.66 = a genuinely better/LEARNED grasp, not better planning or controller tweaks. (Closed-loop relative-position servo also TESTED + rejected: overshoots through the compliant grasp, success collapsed; open-loop setpoint is better.)

## Round 5 (better grasp) — oracle 0.30→0.35: grasp IS a lever but modest; ceiling is grasp-contact physics (2026-06-30)
`exp/hjepa_solve/round5_grasp/`: firmer/better grasp (grip 0.5/1.0) raised the real-env oracle ceiling 0.305(r4)→**0.344/0.352** (vs orig 0.273) — so grasp stability is a genuine low-level lever (reduces slip), but the gain is MODEST (+~0.05) and stays far below PPO 0.66. CONVERGED PANDA-SOLVE CONCLUSION: the analytic-skill-LL family caps at ~0.30-0.35 fundamentally because of GRASP-CONTACT PHYSICS (cube slip in a parallel-jaw grip); every lever — planner stability, place precision, yaw alignment, firmer grasp — gives diminishing returns toward this physical wall. H-JEPA SOLVES Panda competitively (0.289, beats hand-tuned 0.246, n=3 reproducible) and extracts ~0.28-0.29 of the ~0.35 oracle; closing to PPO's 0.66 requires end-to-end learning of grasp+place that the injected-skill family can't express, NOT better high-level planning. This is the definitive ceiling characterization for the Panda H-JEPA solve.

## ✅✅✅ ROUND 5 GO — Panda solve CULMINATES at 0.352, H-JEPA fully extracts the oracle (2026-06-30, FINAL)
Round-4 premise corrected: PandaPickCube randomizes only cube XY (cube spawns UPRIGHT, target upright), so rot_err = MANIPULATION-INDUCED tipping of the tall 4×4×6cm cube in the 2-finger grip (5°→20° through transport). Fix (`param_controller_r5.py`): closed-loop upright-orientation servo recomputing gripper orientation every step in lift/transport/place/hold to right the LIVE cube. Slip reduced: final tilt 20.6°→16.4°, warm-start succ 0.180→0.242. ORACLE: orig 0.273→r4 0.305→**r5 0.352** (+0.078, 3× r4's lever). H-JEPA re-run (warm-started, g=1.0): it1 0.293→it70 **0.3516 peak = the new oracle ceiling** (vs r3 0.289, r4 0.279); plan-only also rose ~0.18→0.21. VERDICT GO. FULL SOLVE ARC: 0.0 (scratch) → 0.117 (competent primitive) → 0.289 (warm-start) → **0.352 (better grasp)** = the oracle — H-JEPA now FULLY EXTRACTS the analytic-LL ceiling, and EVERY gain came from a better LOW-LEVEL primitive. Decisively confirms the thesis: the lever is the low-level primitive; the latent planner extracts exactly what it allows. HONEST CAP: ~0.35 is the analytic-LL physical limit for this tall-cube/2-finger gripper (righting orientation perturbs placement position through the single compliant contact — can't fit both in the 0.9·pos+0.1·rot≤0.02 budget); closing to PPO 0.66 needs a DIFFERENT contact primitive (side/two-stage regrasp, force-controlled fingers), NOT planning. `exp/hjepa_solve/round5_grasp/`.

## CORRECTION: round-5 FINAL n=512 = 0.367 (not 0.352) — the new best Panda solve
The round-5 H-JEPA solve n=512 FINAL came in at **0.3672** (peak 0.3672), at/above the n=128 oracle estimate (0.3516); plan-only 0.320. So the definitive solve arc is 0.0 → 0.117 → 0.289 → **0.367** (better grasp, +0.078/+27% over round-3). Ceiling now ~0.37 = tall-cube/2-finger contact geometry (still ≪ PPO 0.66; needs a different contact primitive). CI seeds 1/2/3 corroborate (0.32-0.37). `exp/hjepa_solve/round5_grasp/RESULTS.json`.

## Beat-PPO scan (2026-07-01, INTERIM — TD-MPC2 seed1 ~20% of 5M) — a genuine ceiling-beat forming
TD-MPC2 vs MATCHED-budget vanilla PPO (raw episode return, matched native eplen — parity verified in scan design doc; PPO run to full official tuned budget = true plateaued ceiling, n=2). Boxes b3060(TD-MPC2)/b3060b(PPO). `exp/beat_ppo_scan/`.
| env | TD-MPC2 (interim) | PPO @~5M matched | PPO ceiling (plateaued) | interim read |
|---|---|---|---|---|
| **PandaPickCubeOrientation** | **2306 peak @0.95M (19%)** | 666 | **1419** (flat @75M, n=2 both 1419) | TD-MPC2 EXCEEDS PPO ceiling ~60% at ~80× fewer steps → clean both-axes beat forming |
| LeapCubeRotateZAxis | 1.8 @0.9M (climbing) | ≈0 (PPO can't explore @5M) | 32 @114M | likely matched-budget WIN; PPO wins asymptote |
| LeapCubeReorient | 34 @1.15M (climbing) | 91 | 186 @126M | contested; need TD-MPC2 @5M |
| PandaRobotiqPushCube | 1.1 @1.35M | 1.0 | 9 @387M | both FAIL (extreme sparse, eplen 3000) |
CAVEATS (honest): (1) metric is episode RETURN not success-rate — TD-MPC2's Orientation win could be denser reward accrual without higher task success; MUST harvest box_target success before claiming "solves better". (2) TD-MPC2 seed1 only ~20% done; seed2 pending; numbers will firm. (3) PPO ceilings ARE plateaued (n=2, flat), so the matched-vs-ceiling comparison is sound. PROJECTION: PandaPickCubeOrientation = the headline beat-PPO (both axes on return); Leap dexterous = matched-budget wins but PPO likely holds asymptote (pattern-consistent); PushCube = mutual fail. NOT yet a firm verdict — pending completion + success-metric check.

### Beat-PPO scan — 05:15 UTC update (TD-MPC2 seed1 ~30-44%): sharper read
- **PandaPickCubeOrientation: beat STRENGTHENS** — TD-MPC2 peak 2306→**2842** (still climbing @29%) vs PPO plateau 1419. Clean both-axes beat on RETURN (dense task, flat obs — no privilege confound). ⚠ box_target SUCCESS metric still not harvested (reward CSV lacks it) — confirm before "solves better" vs "denser reward".
- **Leap dexterous NOT a win + CONFOUNDED:** TD-MPC2 peaks PLATEAUED low & not climbing (Reorient 34 stuck <PPO-matched 91; RotateZ 1.8 vs PPO≈0). BOTH methods exploration-limited at 5M. AND the comparison is UNFAIR to TD-MPC2: PPO's official Leap config uses an ASYMMETRIC actor-critic with PRIVILEGED obs, but the dict-obs shim feeds TD-MPC2 only the state channel (privileged dropped). So do NOT score Leap as a TD-MPC2 loss — it's handicapped + budget-limited. No clean Leap verdict.
- **PushCube: mutual fail** (both ~1). NARROWED HEADLINE: the scan's genuine beat-PPO is **PandaPickCubeOrientation** (dense, TD-MPC2 planning dominates); the "exploration-hard dexterous → we win" hypothesis did NOT pan out (both struggle; comparison confounded). Consistent w/ the campaign law: TD-MPC2 wins where planning over a learned model exploits dense structure; on brutally-sparse dexterous in-hand tasks neither cracks it at 5M.

### ⚠⚠ RETRACTION (05:47 UTC) — the PandaPickCubeOrientation "beat" was RETURN-GAMING, real success = 0.000
Dug out the SUCCESS metric (`exp/beat_ppo_scan/PandaPickCubeOrientation_seed1.log`, realsuccess.csv): TD-MPC2 `succ=0.000 reached~0.2 box_target_max~0.001-0.008` at every eval despite RETURN 2842. So the 2842-vs-PPO-1419 return "beat" is **dense-reward accrual WITHOUT solving the task** (never completes pick+orient; barely reaches cube) — NOT a task-success win. The earlier "clean both-axes beat forming" / "STRONG beat" reads are **RETRACTED**. This is the campaign's core return-vs-success trap (same lesson as base PandaPickCube: reward/return misleads, box_target≥0.9 success is the real test). PPO's harness logs RETURN only (no success rate; "box_target: 8.0" in its config = reward WEIGHT, not success), so PPO's true success is unlogged — but since TD-MPC2 (higher return) = 0 success, TD-MPC2 does NOT beat PPO on the metric that matters, regardless of PPO's value.
**CORRECTED SCAN VERDICT: NULL — no clean beat-PPO on the new envs.** (a) PandaPickCubeOrientation: return-gaming, 0 success. (b) Leap dexterous: both budget-limited + privilege-confounded, neither solves. (c) PushCube: mutual fail. The only clean beat-PPO cases remain the pre-scan EXPLORATION-HARD ones (HopperHop TD-MPC2 367 vs 33, pendulum/cartpole escape). LESSON (re-confirmed): a return-based scan is the WRONG axis for shaped manipulation; must score real success. Runs left to finish for completeness but verdict is set.

### ✅ Beat-PPO scan CLOSED (2026-07-01 ~06:54) — "PPO solves, TD-MPC2 fails" (PPO real success verified)
Evaluated the trained PPO checkpoints for REAL success (n=256 deterministic rollouts, native success fields — box_target≥0.9 for Panda, success_count/ori_err<0.1rad for Reorient, held-30-step for Push). `exp/beat_ppo_scan/{PPO_SUCCESS_eval.md,eval_ppo_success.py}`.
| env | PPO real SUCCESS | PPO budget | TD-MPC2 success @5M |
|---|---|---|---|
| PandaPickCubeOrientation | **0.809** (both seeds, reached 1.0) | 96.7M | 0.000 (return 2842 = gaming) |
| LeapCubeReorient | **0.988** (seed1 @212M; 3.8 reorients/ep) | 212M | 0.000 |
| PandaRobotiqPushCube | 0.18–0.51 | 386M | ~0 |
DECISIVE: the MuJoCo-Playground tuned PPO GENUINELY SOLVES these hard manipulation/dexterous tasks at full official budget (81–99% success); our TD-MPC2 at a practical 5M budget does NOT reach competence on ANY of them (0 success; only reward-gaming). So this scan is the OPPOSITE of a beat-PPO — it's a clean case where **PPO decisively beats TD-MPC2**. Confirms the Playground paper. IMPORTANT NUANCE (completes the campaign's beat-PPO map): TD-MPC2's per-step sample-efficiency only converts to a WIN when the task is solvable within TD-MPC2's PRACTICAL step budget (model-based ⇒ slow ⇒ ~5M feasible). These envs need 75–400M steps; PPO's high throughput (512-env brax, ~10-30× TD-MPC2 sps) lets it reach that many samples and SOLVE, while TD-MPC2 can't run that far in wall-clock and stalls at 0 success. So the honest FULL law: **we beat PPO on sample-efficiency AND practical-capacity only on EXPLORATION-bottlenecked tasks solvable in a few M steps (HopperHop, sparse swing-ups); on SAMPLE-HUNGRY high-DoF tasks (dexterous in-hand, multi-obj manip) that need 100M+ steps, PPO's throughput wins and TD-MPC2 doesn't reach competence at all.** At MATCHED 5M, neither solves these (PPO@5M also low; needs 75M). No beat-PPO on new envs; the only clean beats remain the exploration-hard few-M-step regime.

### Proposal C — "glass as variance-reduction" = NULL (2026-07-02, analysis-only, n=4-5)
Tested whether glass(SE)'s 6/16 benchmark wins are a real worst-case/variance-reduction effect. NULL. (1) Within the 6 win-tasks glass has lower seed-sd + higher worst-seed 6/6 — but CIRCULAR (tasks selected because glass mean higher; with equal ceilings that ≡ "TD-MPC2 owned the low seed"); drop-worst-seed erases the win on 4/6 (only FingerSpin, ReacherHard survive). (2) OUT-OF-SAMPLE across all 16: mean seed-sd glass 123.2 > TD-MPC2 114.5 (glass HIGHER variance); glass lower-sd only 8/16, better worst-seed 9/16 (coin-flips); glass has its OWN collapses TD-MPC2 avoids (HopperHop 0 vs 179, HopperStand seed=15 vs 858, ReacherEasy ±89 vs ±3, WalkerWalk ±81 vs ±12). (3) Cross-algo eff_rank correlation UNVERIFIABLE — no TD-MPC2 latent diagnostics on disk. VERDICT: glass≈TD-MPC2 is a WASH; the 6 wins are small-mean both-ways scatter from unlucky seeds; NO publishable variance-reduction section. Reinforces the redundancy result. `b3060:exp/proposal_C_variance/{VERDICT.md,per_seed_data.json}`.

### Proposal D2 — pure-JEPA on DMControl: REVERSES the anti-collapse story (2026-07-02, WalkerWalk, n=3)
The correctly-scoped JEPA test: pure self-predictive latent (encoder + jumpy predictor + EMA target, NO reward/value/policy), frozen-encoder ridge probes for geometric AND value(RTG) readouts. `b3060b:exp/proposal_D2_pure_jepa/`. Differentiable-2D-SE validated vs selib 3e-7. Raw-obs bounds: geom R² 0.381, value 0.216.
THREE surprising findings:
(1) **PREMISE FALSIFIED — a pure JEPA does NOT collapse.** `none` arm (zero anti-collapse) = HIGHEST decodability (geom **0.795**, value **0.304**, above raw-obs), holds even WITHOUT SimNorm (0.770/0.240). → collapse is prevented by the **predictor+EMA (BYOL) asymmetry**, NOT the value anchor (my earlier "TD-MPC2 value-loss prevents collapse" correction is itself refined/wrong — the asymmetry does it, and it's present in both TD-MPC2 and pure JEPA).
(2) **Downstream-dependent taxonomy does NOT hold on a pure JEPA.** No arm helps-geom-hurts-value; uniformity HURTS BOTH (geom 0.795→0.583, value 0.304→0.121) while raising eff_rank 5→40; geom-R² and value-R² move TOGETHER (no collapse to trade against). So "downstream-dependent" was specific to the NAV-collapse regime, not general.
(3) **SE ≠ uniformity; grad-norm-matching is a TRAP.** At natural λ=0.3-1.0, SE is the BEST-behaved regularizer (compact eff_rank ~6, preserves geom 0.69/value 0.27 ≫ uniformity). Grad-matching SE to uniformity inflates its weight ~74× (raw ‖∇se‖≈0.025 vs ‖∇unif‖≈1.86) → SE dominates → destroys readouts (geom 0.10). That catastrophic row is an OVER-WEIGHTING ARTIFACT, not intrinsic to SE. ⚠ IMPLICATION for D1 (running SE-arm on TD-MPC2, grad-matched): its "SE hurts value-control" may be an over-weighting artifact — MUST re-check D1 with a fixed-λ SE control before claiming.
OPEN QUESTION (honest): nav-H-JEPA DID collapse (eff_rank≈0) but WalkerWalk-pure-JEPA does NOT — difference is online-goal-conditioned vs offline-random-policy, or task, or config. Unresolved; the anti-collapse "taxonomy" is regime-specific, not a law. LIMITATIONS: 1 task, n=3 (n=1 for no-SimNorm + fixed-λ SE). NEXT: extend D2 to +2 tasks incl a clean geometric one + online data + fixed-λ SE sweep to firm the reversal.

### Proposal A1 — planning-as-exploration: PARTIAL GO (mechanism confirmed n=1), flagship needs dedicated training (2026-07-02)
De-risk of "planning is a directed-exploration operator." `b3060:exp/proposal_A1_exploration/` (harness `scripts/coverage_rollout.py`, FORCE_CK same-weights mode).
GO on MECHANISM (n=1 task, clean): PandaPickCubeOrientation 750k ckpt, MPPI plan ON vs OFF at IDENTICAL weights → distinct grid cells 222 vs 100 (2.2×), occupancy entropy 4.74 vs 3.61, 10-NN dispersion 3.84 vs 1.37 (2.8×), return 405 vs 349. Planning explores far more state AND coverage tracks return. (Agent caught + avoided a confound: best_pi@200k vs best_mppi@750k conflates training-stage w/ planning → near-parity artifact; only same-weights ablation isolates planning.)
NOT YET de-risked (the headline): (a) HopperHop/CartpoleSwingupSparse have NO usable checkpoints (never-`--save_full_state` left only CSV logs) and NO saved PPO models → the 3-arm {TD-MPC2/PPO/pi-only} coverage test needs a FRESH instrumented training run; (b) the famous 367-vs-33 is TD-MPC2-vs-PPO, and planning-vs-pi-only return edge on HopperHop is only +1.7 (n=31, 77% of seeds) → the claim must be carried by COVERAGE not return. NEXT (A1-full, needs a dedicated GPU): train TD-MPC2(plan) / pi-only / PPO from scratch on HopperHop + escape-frontier WITH the coverage logger; show coverage(plan)>>coverage(PPO) during the exploration phase + that it predicts the final gap. Reusable logger ready.

### Proposal D2-EXT — reversal FIRMED (2026-07-02, 3 tasks WalkerWalk/CheetahRun/ReacherEasy, n=3, FIXED-λ)
Firmed D2 with fixed-λ (no grad-match) arms {none, unif λ0.5, vicreg λ1, se λ0.3, se λ1.0} + a WalkerWalk online-smooth-policy probe. `b3060b:exp/proposal_D2_pure_jepa/ext/`, `D2_EXT_VERDICT.md`. (1) **Pure JEPA doesn't collapse — HOLDS all 3 tasks + online**: `none` best-or-TIED 6/8 readouts, never collapses; ReacherEasy `none` value-R² 0.257 ≈7× raw-obs 0.038; WalkerWalk-online `none` beats raw-obs both readouts (geom 0.459 vs 0.161, value 0.402 vs 0.249) = latent ENRICHMENT. (2) **No downstream taxonomy — HOLDS**: uniformity hurts BOTH readouts on every dataset; geom+value move together; the geom-vs-value split was NAV-SPECIFIC not general. (3) **Grad-match=artifact CONFIRMED** (fixed λ never nukes, worst SE geom 0.33 vs grad-matched 0.10); SE-natural-λ gentle/compact BUT **vicreg is the MOST NEUTRAL** (self-correction: not SE). Offline-vs-online: smooth-correlated policy did NOT induce collapse (only truly reward-optimizing on-policy — the nav regime — remains untested = the one open cell). NET (D2+D2-ext): the Part-5 "downstream-dependent anti-collapse taxonomy" is REVERSED/qualified — anti-collapse is load-bearing ONLY when the latent actually collapses (narrow on-policy data, e.g. nav); on broad DMControl data a pure self-predictive JEPA (BYOL asymmetry) doesn't collapse and extra anti-collapse is neutral-to-harmful. TODO: fold into blog Part5 §2/§7 (honest reversal).

### Proposal D3 — pixel JEPA collapse: NULL (2026-07-01, WalkerWalk+CheetahRun, 4 arms fixed-λ, n=3, 30 runs)
`b3060b:exp/proposal_D3_pixel_jepa/{VERDICT.md,D3_summary.json,pixel_jepa_dmc.py}`. Feasibility: dm_control loader broken vs mujoco 3.8 → rendered 64×64×3 via mujoco_playground MJX dynamics + raw-mujoco EGL renderer (pixel-std ~32–36); DrQ 4-conv CNN encoder into the D2 harness, latent self-prediction only (predictor+EMA), frozen-encoder ridge probes. **VERDICT: NULL — anti-collapse NOT load-bearing on pixels; picture identical to state.** (1) Pure JEPA does NOT collapse on pixels: `none` eff_rank 33–34/64, best/tied-best readouts both tasks (Walker geom .314/val .216; Cheetah geom .421/val .168); no arm collapsed. (2) Anti-collapse rescues nothing, dilutes/hurts: uniformity MAXIMIZES eff_rank (61/64) but WORST readouts (Walker .175/.109) — same inverse eff_rank↔readout as state; vicreg≈neutral; SE mildly hurts value. (3) Only shift vs state: lower absolute geom-R² (.31 vs .77) = decoding qpos from single frame (no velocity/frame-stack) is harder = task difficulty, NOT collapse. Interp: predictor+EMA (BYOL) asymmetry is load-bearing in high-dim exactly as low-dim; LeCun's collapse concern doesn't materialize even in the pixel regime it targets. Caveats: single frame (no velocity caps abs value-R²), fixed-λ only (no over-weight-forced-collapse claim), small CNN/12k steps/random buffer. NET D-thread (D2+D2-ext+D3): pure self-predictive JEPA doesn't collapse on DMControl state OR pixels; anti-collapse ranges neutral→harmful; the Part-5 "downstream-dependent anti-collapse taxonomy" was nav-specific. Only open cell: truly reward-optimizing on-policy data (nav regime).

### Proposal A1-core — planning→coverage from SCRATCH: NULL (mildly reversed) (2026-07-01, HopperHop, n=3, 80k steps)
Clean 2-arm ablation on b3060b: PLAN (full MPPI) vs PI-ONLY (MPPI disabled, policy-prior actions), identical task/budget/online-coverage-logging (CoverageLogger wired into training loop, frozen bins fit on shared random warmup → cross-arm comparable), n=3 seeds each. `b3060b:exp/proposal_A1_coverage/{VERDICT.md,coverage_summary.json}` + `helios-rl/exp/tdmpc_glass/HopperHop_A1_{plan,pi}/seed_*_coverage.csv`. **Link1 planning→coverage: NULL, mildly REVERSED** — final-step PLAN vs PI-ONLY: proj_distinct_bins 614±123 vs 685±151 (t=−0.63 n.s.); proj_entropy 4.87±0.20 vs 5.24±0.20 (t=−2.28, significant in WRONG direction); mean_perdim_entropy 1.65±0.11 vs 1.72±0.05 (n.s.). Sign consistent at EVERY checkpoint from 20k: PI-ONLY covers ≥ PLAN. **Link2 coverage→performance: UNTESTABLE** — Pearson r=0.105 (n=48); returns floor-heavy (mean 2.64, mostly 0), neither arm learns in 80k. MECHANISM (from logs, not fabricated): early training = untrained WM (mpc=0.0, mppi_return≈0) → MPPI actions are conservative/model-noise while policy-prior+expl-noise disperses ≥ as wide. VERDICT: NULL for the A1 thesis FROM SCRATCH on HopperHop. RECONCILES w/ A1-mechanism GO (2.2× coverage) which used a TRAINED 750k ckpt → honest synthesis: **planning acts as a coverage/exploration operator only once the world-model is competent, NOT cold-start.** Caveats: single task, single 80k budget, n=3. Flagship reframe: A1-full must use a task/budget where the agent actually reaches separable nonzero returns, and measure coverage ACROSS the training curve (expect PLAN to pull ahead after model warms up).

### Proposal A1-FULL — planning-as-exploration on a LEARNABLE task: NULL (2026-07-01, WalkerRun, n=3, 140k)
Reframed A1 de-risk on a task that ACTUALLY LEARNS (A1-core HopperHop@80k stayed at floor). WalkerRun 140k, PLAN(MPPI) vs PI-ONLY(MPPI off), n=3, online coverage (frozen bins on shared warmup). Returns 15.6→603.6, mean 320, warmup ~30k. `b3060b:exp/proposal_A1_coverage/{VERDICT_full.md,coverage_curve_summary via aggregate_full.py}`. **Link1 planning→coverage: NULL, mildly REVERSED even after warmup** — late PLAN vs PI: distinct_bins 464 vs 501 (t=−2.75), proj_entropy 4.81 vs 4.86 (t=−1.77), perdim 1.84 vs 1.87 (t=−3.11). **Link2 coverage→performance:** Pearson 0.719 all / 0.476 post-warmup — coverage tracks return but via shared training-progress trend, not planning. **Per-arm final return: PLAN 479.6±66.9 vs PI-ONLY 541.8±65.0** (PI non-sig AHEAD, t~−1.15). Planning improves NEITHER coverage NOR return vs policy-only. VERDICT NULL. CRITICAL CAVEAT: WalkerRun is DENSE (not exploration-bottlenecked) so the thesis doesn't predict an advantage here — this confirms no planning benefit when exploration isn't the bottleneck (expected); decisive test still owed = exploration-HARD + learnable task (CartpoleSwingupSparse escape / longer HopperHop). NET (A1-core + A1-full + A1-mech): planning-as-exploration UNSUPPORTED so far — 2 from-scratch tests show NO training-time coverage advantage; the one positive (2.2× coverage) is an INFERENCE-TIME same-weights ablation (not training-time exploration); the HopperHop 367-vs-33 is TD-MPC2-vs-PPO (model/algo), not planning-vs-pi (that edge only +1.7). The flagship needs the exploration-hard-learnable test before any GO.

### Proposal A1-FULL — CORRECTION (2026-07-01, verified from returns_by_step JSON, n=3)
My first A1-full entry above said "planning no return gain, PI-ONLY non-sig ahead (542 vs 480)" — that used the NOISY single-episode EVAL (EVAL_NEPS=1) at 140k. The clean n=3 **collect-return curve** (disk: coverage_curve_summary.json returns_by_step) CORRECTS this: **PLAN leads PI substantially through the whole competence phase** — 30k +41, 50k 334.9 vs 185.5 (+149), 60k +139, 80k 428 vs 297 (+131), 90k 408 vs 247 (+161), 100k 486 vs 368 (+117) — then CONVERGES (120k tied, 140k 480 vs 446, +34). So planning reaches competence ~1.5–2× faster and converges to the SAME ceiling. CORRECTED A1-full verdict: (1) **Coverage NULL & REVERSED** — planning does NOT broaden state coverage; post-competence it NARROWS it (distinct cells late −37, t=−2.75) = directed EXPLOITATION onto the gait manifold, not exploration; the exploration-operator hypothesis is refuted on the coverage axis. (2) **Sample-efficiency GO** — planning learns markedly faster (speed-not-ceiling), consistent w/ the campaign law. (3) **Coverage does NOT mediate** — PLAN has LOWER coverage yet learns FASTER. NET reframe: "planning is a directed-EXPLORATION operator" is WRONG on the mechanism (it's exploitation, not coverage-broadening), but planning remains a real sample-efficiency lever. The exploration-HARD decisive test (A1-decisive, CartpoleSwingupSparse, running) is still needed to see if planning's exploitation ALSO helps discover sparse reward. Lesson (verification discipline): read the n=3 training curve, not a 1-episode eval snapshot.

### Proposal A1-DECISIVE — planning-as-EXPLORATION REFUTED (2026-07-01, CartpoleSwingupSparse, n=3, 300k)
The make-or-break test: sparse + exploration-hard + learnable. PLAN vs PI-ONLY n=3. `b3060b:exp/proposal_A1_coverage/{VERDICT_decisive.md,decisive_summary.json}`. **KEY Q — does planning discover the sparse reward where policy-only stalls? NO.** Reward discovery rate PLAN 3/3, PI-ONLY 3/3 (both always find it); mean discovery step PLAN 156672 vs PI **140032** (PI discovers EARLIER). Strong exploration claim REFUTED — policy-only TD-MPC2 (world-model + expl noise) discovers hard rewards equally well. **What planning buys = post-discovery EXPLOITATION**: final collect-return 454.2 vs 239.3, best-return 540.5 vs 331.3; PI collapses late (210k: PI 59 vs PLAN 262). Coverage Link1 marginal (late PLAN distinct bins 597 vs 572 t=2.13, entropy 4.57 vs 4.40 t=2.98) but does NOT drive discovery (PI covered less, discovered earlier); Link2 Pearson 0.46/0.35 = training co-growth. **VERDICT NULL** for planning-as-directed-exploration. NET A-FLAGSHIP (A1-core HopperHop null + A1-full WalkerRun coverage-null/exploitation + A1-decisive sparse pi-discovers-equally): planning is NOT a directed-exploration operator; its repeatable value is SAMPLE-EFFICIENCY + EXPLOITATION (faster-to-competence, higher/steadier post-discovery returns). The "exploration is the real PPO breakthrough" novel paper claim does NOT survive a controlled plan-vs-pi test; honest salvage = the speed/exploitation characterization (feeds Thread-B taxonomy). The HopperHop 367-vs-33 remains TD-MPC2-vs-PPO (model/algo), not planning-vs-pi. n=3, single task/budget each.

### D open-cell — narrow/on-policy data does NOT collapse pure-JEPA: NULL (2026-07-01, WalkerWalk, n=2, PARTIAL)
Distribution-width sweep, same pure-JEPA `none` arm: broad-random→broad-smooth→on-policy(ARS)→narrow top-return frac(25/10/05/02%); frozen-encoder eff_rank+geom-R²+value-R². `b3060b:exp/proposal_D2_pure_jepa/{VERDICT_onpolicy.md,onpol_runs/*.json}`. WalkerWalk (sn on, n=2): onpol-early effR10.3 g0.73 v0.31; narrow.25 effR17 g0.83 v0.11; narrow.10 g0.77 v0.07; narrow.02 effR20.6 g0.51 v−0.05. **VERDICT NULL — no nav-style collapse**: eff_rank never→0 (10-22), geom holds (0.5-0.83); value-R² drops as buffer narrows BUT tracks the narrow buffer's intrinsically-low value variance (raw-obs value-R² also 0.04-0.11), = a DATA property not encoder collapse. So nav H-JEPA collapse (eff_rank≈0) is NOT reproduced by data-width alone → config/closed-loop-online specific. Reconciles D2 (broad: no collapse) w/ nav (collapse) by RULING OUT "narrow on-policy data" as the generic cause. BUGS/CAVEATS (partial): ReacherEasy width buffers never generated (gen bug, untested); broad "" train arm failed (empty-dtag word-split), baseline from D2; "on-policy"=ARS refit NOT true closed-loop value-optimizing RL (the genuinely-untested cell); n=2. NET D-thread fully mapped: pure JEPA doesn't collapse on broad DMControl state/pixels NOR on narrow/on-policy-ARS data; only the true closed-loop-online nav regime collapses (still unexplained mechanism — likely the value/goal closed loop, not data width).

### Proposal D1 — SE arm on TD-MPC2 (redundancy, correctly scoped): SE hurts like uniformity; SE+unif no synergy (2026-07-01, n=3, L16)
SE (relational anti-collapse via selib min-2D-SE) added to the unif_dmc L16 collapse study on TD-MPC2, + SE+uniformity combined arm. Return-AUC n=3 vs known baselines (default/unif). `b3060:exp/benchmark/*unif_dmc*_L16_se{,_unif}_*`. CheetahRun: default 58.9 > unif 36.6 > SE 23.4 (SE+unif 29.0). WalkerWalk: default 293.8 ≫ SE 110.3 > unif 89.9 (SE+unif 62.5). FingerSpin: default 249.4 > SE 214.4 > unif 171.8 (SE+unif 63.8). **VERDICT: SE HURTS value-based control on value-anchored TD-MPC2 on all 3 tasks (default>SE everywhere) — same redundancy family as uniformity** (SE better than unif on 2/3 but same direction). **SE+uniformity COMBINED is WORSE than either alone (no synergy)** → NULL on "combine SE+uniformity." IMPORTANT re D2 grad-match flag: SE is NOT over-weighted here (FingerSpin SE 214 ≈ default 249, NOT nuked, unlike the 74× pure-JEPA artifact) — confirms D2's note that on TD-MPC2 ‖∇se‖≈‖∇unif‖ so the grad-match concern does not apply; result trustworthy. NET: correctly-scoped redundancy datapoint — on a value-sufficient latent, the right anti-collapse is NONE-extra; any relational term (uniformity OR SE OR both) hurts. Completes the D-thread anti-collapse map.

### Thread-D FINAL cell — is closed-loop online feedback the nav-collapse trigger? PARTIAL/MIXED (2026-07-01, nav H-JEPA, n=3, matched 6094 updates)
Controlled A/B, SAME nav H-JEPA arch (SimNorm+JEPA self-pred + EMA + VICReg, faithful 2-level HL), matched updates/eval, vary ONLY data-collection: online=closed-loop (policy co-evolves w/ representation) vs offline=fixed random-policy buffer (loop broken). Added `--offline` + z→state readout-R² probe to `run_hjepa_nav_ab.py` (original untouched). `b3060:exp/hjepa_navctrl/closedloop_ab/{VERDICT_closedloop.md,summary_closedloop.json}`. ROBUST signals (mid-training eff_rank floor/min + readout-R²): online dead-latent (eff_rank→~1e-7) on **3/3** seeds, readout R² ~0.19; offline dead on only **1/3**, min eff_rank 0.02-0.028 on the other 2, readout R² ~0.41 (2× more decodable). → breaking the loop MITIGATES collapse → closed-loop online feedback IS a real DRIVER/AMPLIFIER of the eff_rank→0 dead-latent. ⚠ HONEST TENSION: the FINAL-eval eff_rank (noisy, online sd 0.78 inflated by 1 seed@1.66) shows online 0.553 > offline 0.418 — i.e. the summary JSON booleans (online_collapsed False / offline_collapsed True, threshold 0.5) read BACKWARDS from the trajectory floor; trust the mid-training floor + readout-R², not the bouncy final snapshot. VERDICT: strong binary ("online collapses, offline doesn't") = NULL; weak ("closed-loop deepens the transient dead-latent") = PARTIAL support but fragile. Both arms chronically low-rank on the 4-dim nav task (offline nowhere near broad-DMControl eff_rank 10-22) = partly arch/task not loop. Task success IDENTICAL ~0.64 both arms → representation health DECOUPLED from control (recurring nav finding). NET (closes D): the nav collapse is NOT cleanly explained by any single lever — closed-loop feedback is at most a soft amplifier of a chronically-low-rank latent, and collapse doesn't affect control. Caveats: offline=random (not expert) buffer; noisy final eff_rank; ≥300k budget + frozen-expert buffer + more seeds would sharpen whether residual rank-loss is coverage vs arch. Thread D FULLY closed.

### POSITIVE-CHASE round-1 — C/hierarchy: PRELIMINARY POSITIVE (learned 2-level feudal BEATS flat, 2026-07-01, n=1 val, n=3 sweep in flight)
Chasing a positive on the hierarchy null (learned H-JEPA was 0.0; only injected primitive reached 0.215). NEW method: fully-LEARNED 2-level feudal — LL = goal-conditioned TD3 on dense progress toward a HIRO-style relative subgoal (learnable: local waypoint-reaching is easy); HL = TD3 emitting subgoals every K=15 steps, trained on coarse timescale w/ env sparse reward. `b3060b:tdmpc_glass/exp/feudal_maze.py`, sweep `exp/C_hier_new/`. **VERIFIED FROM DISK (n=1, seed0, VAL logs): on midroom (9.9u open room, sparse, 200-step horizon), matched 400k env-steps, REAL deterministic success — feudal 1.000 by 20k (holds @40k) vs flat 0.000 through 60k.** Mechanism: feudal HL commits to a direction for K steps (~5-10× net displacement of flat's self-cancelling per-step noise) → reaches goals beyond flat's exploration+credit horizon. HONEST calibration/bounds: room (7u) flat SOLVES (1.0 @~40k) = competent baseline not strawman, no hierarchy edge on easy task; deceptive S-mazes (corridor/umaze) BOTH ~0 = win is the open-room long-horizon-credit regime, NOT arbitrary mazes. STATUS: preliminary — n=3 sweep {flat,feudal}×{room,midroom,bigroom}×{seed0,1,2} 400k running (2/18); DONE marker → summary.json (overall_clean_beat, verdict_by_maze) firms it + tests bigroom(12.7u) persistence. NOT over-claimed until n=3 lands. This is the first genuine POSITIVE of the positive-chase campaign; closes the learned-hierarchy null in the regime where the LL is learnable.

### POSITIVE-CHASE round-1 — D/SE-as-structure: NULL (2026-07-01, n=2, 3 tasks)
New integration: SE not as a penalty but as STRUCTURE — selib min-2D-SE partition of the batch-latent kNN graph SUPPLIES supervised-contrastive positives (same community)/negatives (cross), + se_tree adds a coarse encoding-tree level. Fixed λ=0.03 (not grad-matched), frozen-encoder ridge probes. `b3060b:tdmpc_glass/exp/D_se_structure/{VERDICT.md,summary.json,runs/}`. Bar = beat plain `none` (D2: none best-or-tied). RESULT: **NULL/NEG — neither arm beats none on ANY task** (se_contrastive 0/3 geom, 0/3 value, mean Δgeom −0.045 Δvalue −0.044; se_tree 0/3, 0/3, mean Δgeom −0.033 Δvalue −0.023). Per-task e.g. CheetahRun none geom 0.848 > se_contrastive 0.778 > se_tree 0.781; WalkerWalk none 0.795 > both. eff_rank: contrastive RAISES rank (8-9 vs none 5-18) but LOWERS readouts (same inverse pattern as uniformity). VERDICT: SE-as-structure does NOT rescue D — `none` (BYOL predictor+EMA asymmetry) remains best; SE earns its keep neither as penalty NOR as contrastive/tree structure on DMControl readouts. Consistent w/ the whole D thread. (Positive-chase scorecard so far: C=POSITIVE prelim, D=NULL.)

### POSITIVE-CHASE — C/hierarchy GENERALITY: feudal beats flat on multi-room maze (2026-07-02, fourroom, n=3)
Generality test of the C positive on the canonical 4-room doorway maze (fourroom) — where hierarchy is theoretically supposed to help. `b3060b:tdmpc_glass/exp/C_hier_new/runs_fourroom/`. REAL deterministic final success, matched 400k, n=3: **feudal solves 2/3 seeds (s1,s2=1.0; s0=0.0) vs flat 0/3 (never solves)**. So the learned-feudal-beats-flat positive GENERALIZES beyond open rooms to structured multi-room navigation (flat can't cross the doorway-connected rooms; feudal's K-step subgoal commitment does). Honest: one feudal seed (s0) failed = variance, n small; firming to n=6 (seeds 3-5 running). Combined with midroom (feudal 1.0 vs flat 0.0, n=1) the C positive holds across open+multi-room mazes. Awaiting C-main n=3 sweep DONE for the full firmed writeup.

### C/hierarchy fourroom generality FIRMED n=6 (2026-07-02): feudal 4/6 vs flat 0/6
Extended the fourroom (canonical 4-room doorway maze) generality test to n=6 seeds. REAL deterministic final success: **flat solves 0/6 (never), feudal solves 4/6** (seeds 1-4=1.0; seeds 0,5=0.0). Clear directional win (flat literally cannot cross the doorway-connected rooms; feudal's K-step subgoal commitment does), honest variance (2/6 feudal seeds fail). Firms the C positive's GENERALITY: learned feudal hierarchy beats matched-budget flat on multi-room navigation, not just open rooms. `b3060b:tdmpc_glass/exp/C_hier_new/runs_fourroom/` (12 json). Awaiting C-main n=3 open-room sweep (15/18) for the full firmed writeup.

### C/hierarchy — n=3 CORRECTION: open-room advantage was seed-variance; real positive is MULTI-ROOM (2026-07-02)
Firmed the C-main open-room sweep to n=3 {room,midroom,bigroom}×{flat,feudal}, REAL deterministic final success, matched 400k. **CORRECTS the earlier n=1 claim (midroom feudal 1.0 vs flat 0.0 was SEED VARIANCE).** n=3 means: room flat 0.67 vs feudal 1.00; midroom flat 0.33 (seeds 0,0,1) vs feudal 0.67 (1,0,1); bigroom flat 0.65 (0,0.94,1.0) vs feudal 0.67 (1,0,1) = **TIE**. → On OPEN rooms flat is a COMPETENT baseline (solves many seeds); feudal ≥ flat but MARGINAL/within-variance (bigroom tie). The clean, robust positive is **fourroom (multi-room, doorway bottlenecks), n=6: feudal 4/6 vs flat 0/6** — flat literally never crosses the connected rooms. HONEST REVISED VERDICT: the learned-feudal-hierarchy positive is REAL but LOCALIZED to **structured/bottleneck navigation (multi-room)**, NOT open-room long-horizon (where flat suffices). More interesting + defensible than "hierarchy beats flat everywhere." Do NOT write the inflated version. `b3060b:tdmpc_glass/exp/C_hier_new/runs/` (18 open-room) + runs_fourroom/ (12, n=6). Lesson (again): firm to n≥3 before claiming — n=1 seed variance inflated the open-room story.

### POSITIVE-CHASE — E/SE-subgoal HL: NULL (2026-07-02, n=2, 3 sparse tasks)
SE for structure-discovery (its actual strength): selib min-2D-SE (louvain_se) partitions replay-buffer TD-MPC2 latents into communities every rebuild; HL assigns each env an under-visited community as subgoal; LL steered by potential-based goal-distance shaping (reward-shaping realization of the option HL — approx of goal-cond MPPI, stated honestly; LL/optimizer/budget identical to flat = matched control). `b3060:exp/E_se_subgoals/{VERDICT.md,summary.json}`. REAL unshaped eval return, best_any=max over eval of max(pi,MPPI): **NO clean beat on ANY task** — AcrobotSwingup flat 460.6 vs SE 382.2 (Δ−78); CartpoleSwingupSparse flat 316.2 vs SE 7.3 (Δ−309, shaping HURT badly on the sparse task — goal-distance shaping distracted from the sparse extrinsic reward); FingerTurnHard flat 988.4 vs SE 982.9 (~tie). VERDICT NULL — SE-subgoal shaping ties-or-hurts vs flat; hurts most on the sparse task it was meant to help. Caveat: reward-shaping realization not true goal-conditioned MPPI (a planner-level integration untested). Positive-chase scorecard: C=POSITIVE(localized multi-room), D=NULL, E=NULL, A2=pending, PPO-HopperHop=pending(exploration-wall leaning).

### PPO-HopperHop SAMPLE-EFF vs EXPLORATION — FINAL: EXPLORATION WALL (2026-07-02, n=5, up to 472M steps)
User-requested decider. mujoco_playground DMControl + brax PPO 0.14.2 (tuned dm_control_suite_params config, 2048 envs), HopperHop, n=5, budget up to 471.86M env-steps/seed (~94× TD-MPC2's ~5M-to-367). `b3060b:exp/ppo_hopperhop_sampleeff/{VERDICT.md,summary.json,runs/seed*.json}`. Per-seed peak: 44.1/26.1/29.7/53.8/46.5 → **peak mean 40.0, best 53.8; final mean 17.9. 0/5 seeds ever cross 200 or 367** (3/5 barely cross the paper's PPO~33 baseline, only at ~383-432M). TD-MPC2 reaches ~367 in a few M. **VERDICT: EXPLORATION WALL — PPO at 94× the budget peaks ~7× below TD-MPC2 and never approaches the gait; genuine exploration failure, NOT mere sample-inefficiency.** RESOLVES the sample-eff-vs-exploration question (flips the earlier "likely sample-eff" guess): TD-MPC2's HopperHop win IS an exploration advantage, sourced from the WORLD MODEL (model-based value/dynamics enabling discovery), NOT from MPPI planning (plan-vs-π-only was null) — reconciles the whole Thread-A: planning-lookahead adds no exploration beyond the WM-trained policy, but the WM itself enables exploration model-free PPO can't. Supports the user's "world-model components are what matters" (per-head ablation running to localize which). Caveat: this brax-PPO config peaks ~54 (paper baseline 33; both floor ≪367 — config not underperforming meaningfully, conclusion robust).

### POSITIVE-CHASE — A2/novelty-MPPI: NULL / mostly-harmful (2026-07-02, n=2, 5 tasks)
Plan2Explore-in-TD-MPC2: novelty term in MPPI objective (score=predicted-return + β·zscore(novelty); novelty=RND predictor error over SimNorm latent, or 2-Q disagreement). REAL sparse MPPI eval return. `b3060:exp/A2_novelty_mppi/{VERDICT.md,summary.json,runs/}` (40 runs). Per-task best (mean[max] n=2) vanilla vs best-novelty: AcrobotSwingupSparse 1.8 vs ≤1.7 (all floor); BallInCup 487 vs 484 (~tie); **CartpoleSwingupSparse 1.2 vs dis_b0.5 42.5[85]** (auto-flagged "win"); FingerTurnHard 804 vs 656 (novelty HURTS); PendulumSwingup 766 vs 135 (novelty HURTS badly). **HONEST VERDICT: NULL / mostly-harmful — NOT the auto-flagged positive.** The lone CartpoleSwingupSparse "win" is n=2 on a task with EXTREME seed variance (A1-decisive: vanilla hit 316-626 on lucky seeds) — dis got 1 seed to 85 (still ≪ achievable), vanilla both floor = within the seed lottery, not a credible novelty effect. On the 2 dense tasks (Pendulum/Finger) novelty clearly HURTS (over-explores, wrecks exploitation) — consistent w/ the pruning reframe (novelty widens the MPPI search away from value, hurting when reward guides). NET: adding a novelty term to MPPI does NOT reliably rescue sparse discovery + damages dense exploitation → reinforces that MPPI/planning is not the exploration lever (the WORLD MODEL is, per PPO-HopperHop exploration-wall). Caveat: n=2, β∈{0.5,1.0} only, 300k; a fuller β/seed sweep could firm the lone hint but the dense-task harm is clear.

### POSITIVE-CHASE — MiniGrid hard-exp (PPO vs PPO+RND): NULL (2026-07-02, n=3, 6M steps)
Discrete hard-exploration (the regime our thread never tested). PPO vs PPO+RND(novelty), CPU torch, REAL success (terminated at goal, sparse only). `b3060:exp/minigrid_hardexp/{VERDICT.md,summary.json,runs/}`. **BOTH arms stall at 0 success on BOTH maps**: KeyCorridorS3R3 ppo 0/3 vs rnd 0/3 (SR 0.0); MultiRoom-N6 ppo 0/3 vs rnd 0/3. RND explores MORE (KeyCorridor first-success 3/3 seeds @448k/255k/860k vs PPO 2/3; ep-coverage 6.27 vs 1.94) but extra exploration does NOT convert to solving. VERDICT NULL — plain RND novelty does not crack MultiRoom/KeyCorridor (expected: these motivated RIDE/NovelD/AMIGo; plain count/RND insufficient). Reinforces: novelty widens coverage but coverage≠task-success on hard-credit discrete tasks (echoes A2 on continuous). Caveat: n=3, 6M, PPO+RND only (no planning/model-based arm — TD-MPC2 is continuous, doesn't run MiniGrid natively). NET positive-chase scorecard: 1 real positive (PPO-HopperHop exploration-wall→world-model-is-exploration-lever) + 1 localized (C hierarchy multi-room); A2/D/E/MiniGrid = null.

### 🔍 FULL-ARC ADVERSARIAL REVIEW (2026-07-02, Fable-5 session, 3 verification agents) — corrections of record
Full report: `REVIEW_2026-07-02_full_arc.md`. Everything spot-checked in the current phase TRACED EXACTLY to disk
(A2/E/MiniGrid summaries, A1-decisive, all 5 PPO-wall seeds, all 30 C-hierarchy JSONs — no fabrication found).
Corrections + newly-found issues, now the canonical versions:
1. **Flagship number corrected:** "TD-MPC2 ~5M-to-367" was WRONG — disk says 366.8 at **~1M** steps, mppi eval,
   **n=1** (`b3060:exp/tdmpc_glass/HopperHop/seed_1.csv`); fresh n=4 best-seed peaks 259/378/180/311 (mean ≈282);
   live same-env cov run (b3060b) ~480–520 @3.9M (n=2). PPO budget ratio is ≈**470×**, not 94× — conservative-direction
   error, wall STRONGER, but the cited numbers didn't trace. Part 8 corrected in place.
2. **Flagship env parity VERIFIED CLEAN:** PPO and TD-MPC2 load the literally-same `mujoco_playground_repo`
   HopperHop (impl=jax, eplen 1000, action_repeat 1); PPO config = official tuned DMC suite params. Budget actually
   471,859,200/seed (brax epoch rounding of the requested 120M — 4×, strengthens).
3. **Flagship scope caveats:** (a) NO SAC/TD3 baseline exists anywhere → claim = "on-policy PPO is walled", not
   "model-free is walled"; SAC-on-HopperHop is the single cheapest overturning test. (b) PPO curves CREEP (half-vs-half
   means ~double, peaks late; seed2 peak = final point) → say "walled through 472M", NOT "plateaued".
4. **A1-mechanism (2.2× coverage GO) is NOISE-CONFOUNDED:** `coverage_rollout.py` π-arm = deterministic tanh(mu),
   NO exploration noise; MPPI arm samples w/ min_std 0.05/max_std 2.0. Part of the 2.2× is stochasticity, not planning.
   The TRAINING-time A1 ablations (core/full/decisive) ARE noise-matched (same `_current_noise` added to both arms,
   `b3060b:tdmpc_glass/helios-rl/scripts/run_benchmark.py:1844`) and gate ONLY collection — the REFUTATION is clean;
   the one coverage "positive" is the confounded piece. (Also: A1_COLLECT lives on b3060b, not b3060 as handoff said.)
5. **C-hierarchy confound (disclosed but uncontrolled):** feudal LL trains on dense self-generated subgoal-distance
   shaping; flat trains sparse-only; NO shaped-flat control arm exists → claim must be scoped "learned 2-level
   hierarchy WITH self-generated dense shaping beats sparse flat TD3 (matched env-steps)". Feudal also takes 2 grad
   updates/env-step (LL+HL) vs flat 1. fourroom 4/6-vs-0/6 = Fisher p≈0.03 one-sided (thin). Numbers verified exact.
6. **NEW BUG (wall-generalization Pendulum):** upstream case mismatch — `dm_control_suite_params.py` overrides fire
   on `"PendulumSwingUp"` but registry name is `"PendulumSwingup"` → official Pendulum-tuned PPO override (action_repeat=4
   etc.) NEVER APPLIED; run JSON confirms action_repeat=1. Pendulum walled seeds (2/3) may be config, not exploration.
   INTERIM wall-gen read: FingerTurnHard PPO ≈975 ≈ TD-MPC2 → NO wall; Pendulum best seed 852 catches up (>0.8×911).
   The wall does NOT simply generalize — it's task-specific to gait-discovery (HopperHop). Sharpens, not weakens.
7. **WM-ablation partials (b3060, from logs, policy arm pending):** none best≈770/792; consistency-ablated ≈487
   (big hurt); value-ablated ≈32/16 (catastrophic, policy+planner both die); reward-ablated mppi ≈26/1 but pi ≈753/765
   (reward head load-bearing for PLANNING only). Early read: TD/value signal is the load-bearing net; consistency
   substantial; reward planner-only.
### ✅ SAC CONTROL — the exploration wall is ON-POLICY-SPECIFIC, not model-free (2026-07-02, n=3, VERIFIED disk)
The review-mandated decider. brax SAC, mujoco_playground tuned DMC config (audit line confirmed the HopperHop branch
fired), SAME MJX env byte-for-byte as the PPO/TD-MPC2 arms, 3 seeds × 5M sequential.
`b3060b:exp/sac_hopperhop_control/{SAC_CONTROL_DONE,runs/seed{1,2,3}.json}`.
**Peaks 207.1 / 235.1 / 274.3 (mean ≈238.8); 3/3 seeds cross 200 within 5M** (seed1 first ≥200 @4.0M) — the exact
threshold PPO's 5 seeds never touched in 472M. VERDICT: the strong "model-free is walled" reading is REFUTED;
**the wall is an ON-POLICY (PPO) pathology** — off-policy replay+entropy escapes it. What the world model still
buys, cleanly (same env): TD-MPC2 ~282 mean (n=4, best 367) at 1M ≥ SAC's 5M level → **~5× sample-efficiency**,
and ~480–520 (n=2) at 3.9M vs SAC ~239 at 5M → **~2× attained level at matched budget**. Downgrade-and-sharpen of
the flagship: planner=exploitation (unchanged), PPO categorically walled (unchanged), world-model = sample-efficiency
+ attained-level lever rather than the sole exploration escape. Part 8 rewritten accordingly.

### ✅ WALL-GENERALIZATION — FINAL: the wall does NOT generalize; it is TASK-SPECIFIC (2026-07-02, all runs done)
`analyze_ppowall.py` (best-seed criterion, PPO peak < 0.8×TD-MPC2 anchor): **"WALL DOES NOT GENERALIZE: PPO caught
up to TD-MPC2 on all scored tasks."** Per-task (PPO n=3 tuned-config ~295M vs fresh TD-MPC2 n=2 @1M):
FingerTurnHard PPO 971/975/971 vs TD-MPC2 984 — no wall, 3/3. PendulumSwingup PPO 852/38/93 vs 961 — best seed
catches up; 2/3 walled BUT cell CONFOUNDED (PendulumSwingUp-vs-Swingup case bug: tuned override never fired).
BallInCup PPO 0/0/967 vs TD-MPC2 975/0 — discovery-luck on BOTH algos (PPO 1/3, TD-MPC2 1/2), not a clean wall.
NET: the HopperHop exploration wall is **specific to the gait-discovery regime**, which SHARPENS the claim ("PPO
fails at gait discovery specifically") rather than weakening it. `b3060b:exp/ppo_wall_generalization/`.

### ✅ WM-HEAD ABLATION (CheetahRun) — the TD/VALUE signal is the load-bearing net; consistency is the LEAST critical (2026-07-02, n=2, DONE)
`b3060:helios_wmablate/exp/wm_head_ablation/{VERDICT.md,summary.json}`. Zero ONE loss term at a time (mask verified
live in logs), encoder kept, 1M steps, n=2, MPPI + pi eval returns. FULL: MPPI 737.8 / pi 782.2. Ablations:
**value → MPPI 16.4 / pi 12.0 (catastrophic — kills everything)**; reward → MPPI 5.0 but **pi 761.0** (load-bearing
for PLANNING only — partly by construction, MPPI scores rollouts with it; pi barely needs it); policy → MPPI 122.7 /
pi 2.5 (planner badly hurt without a proposal prior); **consistency (the self-predictive latent-dynamics loss!) →
MPPI 367.2 / pi 541.1 — a 50%/31% drop, the SMALLEST of the four**. Cleanest readout = the pi column (no
planner-needs-heads tautology): value ≫ policy > consistency > reward ≈ none. MECHANISM UPSHOT (Thread A / Part 8):
what carries TD-MPC2 is the **TD value signal trained through the latent**, not the self-predictive "world model"
loss per se — the model-based machinery's contribution routes through value learning. Caveat: dense task, n=2;
the SAME ablation ON HOPPERHOP (the flagship exploration task) is running (seeds 1-2 GPU0/1 + seeds 3-4 GPU2/3
just launched) — the decisive version for the exploration-efficiency mechanism.

### ⚠ SHAPED-FLAT INTERIM (2/6 seeds done, 2026-07-02) — dense shaping ALONE solves fourroom; C positive likely re-attributes
`b3060b:exp/C_hier_new/runs_shapedflat/`: s0 final success **1.000** (from 360k), s1 **1.000** @380k. Flat TD3 +
dense potential shaping to the TRUE goal solves the maze that sparse flat never solved (0/6) and feudal solved 4/6.
If this holds at n=6 → the fourroom "hierarchy positive" RE-ATTRIBUTES to dense shaping, with one honest residual:
feudal's shaping was SELF-GENERATED (no privileged goal info) while this control uses the true goal (a deliberate
upper bound) — so the surviving claim would be "a learned hierarchy can manufacture its own dense signal," not
"hierarchy beats shaping." HOLD final verdict for n=6.

### ✅ SAC-20M FINAL n=5 (2026-07-02) — distributions overlap; no clean level gap; TD-MPC2 = consistency + efficiency
`runs_20M/seed{11..15}.json`: finals **414.0 / 277.2 / 572.5 / 285.2 / 246.3** (mean ≈359); peaks 418.7 / 277.2 /
574.5 / 304.5 / **553.2** (s15 peaked then regressed — high late-training seed variance). vs TD-MPC2 same-env
final ~477 (480.9/473.1 @5M, n=2). Honest final statement: SAC's mean 20M level sits below TD-MPC2's 5M level but
the distributions OVERLAP (1/5 finals, 2/5 peaks ≥ anchor) → **no clean level gap; the world-model's robust
advantages are consistency + ~4-5× sample-efficiency.** Closes the SAC control series (n=3 @5M + n=5 @20M).

### UPDATE 14/20 arms (post-outage; b3060 network outage ~17:00-18:35 did NOT kill the drivers — host never
rebooted, jobs kept training): **consistency-ablated FIRMED n=4** (mppi 201.2/194.0/244.5/184.6, pi 236-319 —
gait found 4/4 at ~half strength); **reward-ablated s1/2: mppi ~0 (planner dead, by construction) but pi 518.7 /
241.0 — the POLICY still learns the gait without the reward loss** (same planner-only pattern as CheetahRun).
Remaining: policy arms s1/2 (running), reward+policy s3/4. The n=4 mechanism story is now: value→0 (wall),
consistency→~half, reward→planner-only. Part 9 updated.

### UPDATE 16/20 (WM_ABLATION_HOP_DONE for s1/2): **policy-ablated ≈0 on BOTH readouts** (s1 0.1/0.1, s2 0.0/0.0)
— unlike CheetahRun (planner limped to 123 without a prior), on HopperHop the planner cannot compensate AT ALL.
reward s3/4 confirm planner-only at n=4 (pi 225.5/188.7). PRECISE FINAL MECHANISM: the value-learning pathway
(TD value loss + the policy trained from it) is INDIVIDUALLY NECESSARY on the exploration task (each ablation →
total failure); reward = planner-only; self-predictive consistency = the only merely-degrading component.
Part 9 updated with final phrasing. Last 2 arms (policy s3/4) corroborative.

### ✅ VALUE-ABLATION WALL REPRODUCTION FIRMED n=6 (2026-07-03 ~09:05, WM_ABLATION_HOP_S56_DONE)
Seeds 5/6 (none + value arms only): none mppi_best 373.7/306.6 (healthy); **value-ablated 0.0/0.1 — dead.**
Full n=6 value cell on HopperHop: **0.0 / 0.0 / 3.2 / 0.0 / 0.0 / 0.1** (gait NEVER found without the TD value
loss, 6/6 seeds) vs none n=6 287–570 (found 6/6). The mechanism's headline cell is at publication-grade n.
Part 9 updated (value n=4→n=6).

### ✅ RATE-SHARPENING COMPLETE (2026-07-03 ~13:40) — all wall/barrier cells at final n
Disk-verified finals: **PPO Stand n=8: 149.2/142.4/144.3/681.2/749.0/153.5/125.7/122.1 → 2/8 escape @285M.**
**SAC Hop n=8 @5M: 207/235/274/47/131/273/67/273 → 5/8 ≥200**; 20M-run crossings (seeds 11-15): 4.1M/5.1M/6.2M/
7.7M/7.7M → **5/5 by ~8M** (budget-indexed escape; PPO 0/5 @472M categorical). **SAC Stand @1M probe: 0/3 (peaks
70/10/12)** vs TD-MPC2 2/2 @≤0.9M (962/948) → TD-MPC2's Stand efficiency edge REAL at matched 1M budget; SAC Stand
5/6 by 5M. TD-MPC2 Stand s43 (n=3) in flight. FINAL claim set (Part 9 updated): categorical PPO wall on hop;
graded barriers elsewhere; matched-1M column = TD-MPC2 solves both hopper tasks where neither model-free method
solves either; reliability×efficiency ordering TD-MPC2 ≫ SAC ≫ PPO fully quantified.

### ⚠ WALL-PROBE EXTENSION n=4 (2026-07-03 ~10:10) — Stand wall is PROBABILISTIC, not categorical; SAC seed-dependent too
`runs_ppo/seed4{3,4}.json`, `runs_sac/seed4{3,4}.json` (all done:true). **PPO Stand n=4: 149.2/142.4/144.3/681.2 —
seed 44 ESCAPED (peak 681, final 619 @285M); 3/4 walled.** **SAC Stand n=4: 492.2/754.2/464.3/32.6 — seed 44
FAILED; 3/4 solve @5M.** HONEST REVISED PICTURE: the categorical wall holds ONLY on HopperHop (PPO 0/5 ≥200
@472M; SAC 3/3 @5M). HopperStand is a GRADED stochastic barrier for model-free methods: PPO escapes 1/4 at ~60×
SAC's budget; SAC solves 3/4 at 5M; **TD-MPC2 solves 2/2 (962.3 @0.65M, 948.1 @0.9M — WALLPROBE_EXT_DONE 11:12)**. Reliability×efficiency
ordering (TD-MPC2 ≫ SAC ≫ PPO) is the durable claim; "categorical morphology wall" is retracted to
"hop-gait categorical + Stand graded". Part 9 revised AGAIN (2nd revision today — the controls keep teaching).

### 🚩🚩 WALL-BOUNDARY PROBE (HopperStand, 2026-07-03 ~07:15) — THE WALL IS HOPPER-MORPHOLOGY-WIDE, NOT HOP-GAIT-SPECIFIC (superseded by the n=4 extension above)
`b3060b:exp/wallprobe_hopperstand/`. Tuned PPO (Hopper-prefix override VERIFIED fired), 285M steps/seed:
**peaks 149.2 / 142.4, finals 90.8 / 135.8 (n=2) — flat/oscillating at the end, WALLED on HopperStand too.**
SAC (tuned, n=2 @5M): **492.2 / 754.2 — solves.** REVISES Part 9's "gait-discovery-specific" framing: on-policy
PPO fails BOTH hopper tasks (Hop 0/5 ≥200 @472M; Stand ≤149 @285M) while SAC escapes both and 5 non-hopper tasks
show no wall (Finger/Pendulum/BallInCup/Cheetah/Walker). NEW FRAMING: **the on-policy exploration wall is
MORPHOLOGY-specific — the single-leg hopper's unstable, contact-timing-critical dynamics defeat on-policy
sampling even for standing**, which off-policy replay handles easily. Arguably a stronger, more interesting
claim (a characterizable dynamics class, not one task). Part 9 wall framing rewritten.
**STAND COLUMN COMPLETE (WALLPROBE_TDMPC2_DONE 08:10): TD-MPC2 best 962.3 (mppi @650k), 948.4 (pi @800k), n=1 —
essentially solves HopperStand within 0.65M steps.** Full Stand ordering: TD-MPC2 962 @0.65M ≫ SAC 492/754 @5M ≫
PPO ≤149 @285M (walled). On the hopper morphology the value-pathway learner is ~8× more efficient than SAC AND
reaches a higher level; on-policy PPO never gets off the floor.

### ✅ WALKERRUN WM-ABLATION — decisive cells FINAL (2026-07-03 ~03:10, 8/10 arms n_evals=20; consistency mid-run)
`b3060:helios_wmablate/exp/wm_head_ablation_walk/jsonl/` (1M steps, n=2/arm): **none 731.2/679.6 mppi,
711.1/671.1 pi. value-ablated 56.1/28.0 mppi, 43.7/37.2 pi — DEAD. policy-ablated 76.4/64.3 mppi, 48.1/34.5 pi —
DEAD. reward-ablated 46.7/38.6 mppi (planner dead by construction) but pi 710.9/728.2 — FULL STRENGTH.**
consistency-ablated at eval 4/20 already 433.9/571.3 (s1), 221.2/239.4 (s2) — alive and learning (finals pending).
CONSISTENCY FINALS (10/10, WM_ABLATION_WALK_DONE 2026-07-03 ~04:55): mppi 546.8/522.3, pi 674.0/570.0 —
degraded ~25% but fully alive; mildest ablation on all 3 tasks. **THE 3-TASK MECHANISM TABLE IS NOW COMPLETE on the decisive cells (CheetahRun n=2 + HopperHop n=4 + WalkerRun
n=2): the value-learning pathway (TD value loss + the policy trained from it) is INDIVIDUALLY NECESSARY on all
three tasks; the reward head is planner-plumbing only on all three; the self-predictive consistency loss is the
only merely-degrading component on all three.** Part 9 addendum lands with consistency finals.

### ✅ ANATOMY REPLICATION on CheetahRun (2026-07-02 ~23:45) — NO WALL; old Pareto PPO number was a budget/config artifact
`b3060b:exp/anatomy2_cheetah/`. Tuned mujoco_playground PPO (no env-specific override; config audited), 285M
steps/seed: **peaks 892.0 / 921.7 (done) / 910.9 (s33 @275M, finishing) — PPO SOLVES CheetahRun, 3/3.** SAC (tuned,
n=2 @10M): **918.0 / 912.2.** TD-MPC2 fresh anchor: 603 @550k (in flight; prior ~670-738 @1M). VERDICT: CheetahRun
is NOT a second wall — on-policy PPO catches up given budget, matching SAC's level; TD-MPC2 keeps only the
efficiency edge (~600+ at 0.55M vs PPO needing ~50-100M+ to pass 600). **CORRECTION OF RECORD:** the Pareto-study
claim "PPO ~270-290 @30M, never reaches 500 on CheetahRun" (SYNTHESIS_beat_ppo.md §3, Part 5 reality-check) was a
BUDGET/CONFIG artifact — with the tuned config and adequate budget PPO reaches ~900. NET: the exploration wall is
now tested against FIVE comparison settings (Finger, Pendulum-fixed, BallInCup, CheetahRun, HopperHop) and exists
ONLY on HopperHop — maximally gait-discovery-specific, exactly Part 9's claim, now with n=4 no-wall controls.
**ANATOMY2 COMPLETE (2026-07-03 02:27, ANATOMY2_TDMPC2_DONE):** fresh TD-MPC2 CheetahRun anchors peaks **727.0 /
784.1 at ≤1M (n=2)** (`helios-rl/exp/tdmpc_glass/CheetahRun_anat2chee_s{31,32}/`). Final CheetahRun ordering:
TD-MPC2 ~700-784 @1M ≪ SAC 918/912 @10M ≈ PPO 892-922 @285M — pure sample-efficiency hierarchy (~10× vs SAC,
~100×+ vs PPO to comparable level), no wall, and here the model-free methods reach a slightly HIGHER level than
TD-MPC2's 1M plateau (honest note: TD-MPC2 @1M may not be its asymptote either).

### ✅✅ HOPPERHOP WM-ABLATION COMPLETE 20/20 (2026-07-02 ~20:55) — CAMPAIGN'S FINAL EXPERIMENT CLOSED
Policy s3/4 finals: mppi 0.2/1.5, pi 1.5/0.0 — ≈0 both readouts, corroborating s1/2. FINAL n=4 TABLE (mppi_best
per seed): none 570/513/287/372 | value 0/0/3/0 | policy 0.1/0/0.2/1.5 | reward ~0 (planner-dead by construction;
pi 519/241/226/189 learns the gait) | consistency 201/194/245/185. MECHANISM (final): the value-learning pathway
(TD value loss + policy trained from it) is individually necessary on the exploration task; reward = planner-only;
self-predictive consistency = only merely-degrading component. Part 9 finalized (nothing pending).
`b3060:helios_wmablate/exp/wm_head_ablation_hop/` (WM_ABLATION_HOP_DONE + WM_ABLATION_HOP_S34_DONE both present).

### ⚡ HOPPERHOP WM-ABLATION INTERIM (8/20 arms, 2026-07-02, disk jsonl) — value-ablation reproduces the WALL
`b3060:helios_wmablate/exp/wm_head_ablation_hop/jsonl/` (@~1M, 4 evals/arm): **none (full, n=4): mppi best
569.7/512.9/286.8/371.7 (all find the gait)**; **value-ablated (n=4, s3/4 at eval 3): 0.0/0.0/3.2/0.0 — TOTAL
failure, the gait is NEVER found**; consistency-ablated (n=2): mppi 201.2/194.0, pi 235.6/260.3 — degraded ~50-60%
but the gait IS discovered. INTERIM MECHANISM READ (the exploration-task version of the CheetahRun result):
**removing the TD value loss reproduces a PPO-like exploration wall on HopperHop; removing the self-predictive
consistency loss does not.** The exploration-relevant content of "the world model" is the value signal trained
through the latent. Reward/policy arms pending (final VERDICT + Part 9 when all 20 land).

### 🚩 SHAPED-FLAT FINAL (n=6, 2026-07-02) — the C-hierarchy positive RE-ATTRIBUTES to dense shaping
`b3060b:exp/C_hier_new/{SHAPEDFLAT_DONE,runs_shapedflat/}` (6 JSONs, deterministic unshaped final eval, matched
400k): flat TD3 + dense potential shaping to the TRUE goal solves fourroom **3/6 (finals 1,1,1,0,0,0)** ≈ feudal
**4/6** (sparse flat: 0/6). Fisher 3/6-vs-4/6 = indistinguishable. VERDICT: **what carried the fourroom win was
the dense signal, not hierarchy per se** — the "learned hierarchy beats flat" claim is RETIRED. Surviving honest
claim (narrower, real): **a learned 2-level feudal hierarchy SELF-GENERATES its dense learning signal** (no
privileged goal info; the control needed the true goal) and matches privileged shaping. Independently replicates
Nachum et al. 2019 ("Why does hierarchy (sometimes) work" — HRL benefits largely reproduce via exploration/shaping
on flat agents). Part 8 + scorecard updated.

### ✅ PENDFIX (n=2, 2026-07-02) — the Pendulum "wall" WAS the config bug; wall-gen now confound-free
Re-ran Pendulum PPO with the tuned override applied manually (action_repeat=4, num_updates_per_batch=4 — the
branch the upstream PendulumSwingUp/Swingup case mismatch silently skipped). `b3060b:exp/ppo_wall_generalization/
{PENDFIX_DONE,runs_pendfix/}`: **seed21 peak 842.5/final 806.1; seed22 peak 830.9/final 823.2** — both >0.8×
TD-MPC2 anchor (961) in ~8 min wall-clock each. The original sweep's 2/3 walled Pendulum seeds (38.5/93.3) were
the BUG, not exploration. FINAL wall-gen picture, no confounded cells: **the exploration wall exists ONLY on
HopperHop (gait discovery)** — Finger 3/3, Pendulum 2/2 fixed, BallInCup discovery-luck-both-algos.

### ⚠ SAC-20M UPDATE (2026-07-02, s11/s12 done, s13 @12.8M) — NO robust level gap; efficiency is the only robust lever
`runs_20M/`: s11 414.0 final, s12 **277.2** final (@20M each); **s13 at 564.3 by 12.8M — ABOVE the TD-MPC2
same-env anchor (480.9/473.1 @5M, n=2)**. SAC's 20M level spans 277–564 (huge seed variance), overlapping TD-MPC2's
level. VERDICT (supersedes the "~2× attained level" phrasing): the world model's robust advantage is
**sample-efficiency only (~4–5×: TD-MPC2 ~282@1M / ~480@4-5M vs SAC needing ~13–20M, seed-dependent)**; there is
no reliable level gap at larger budgets. Seeds 14/15 launching for the level-distribution n. Part 8 updated.

8. **Blog-history audit:** 4 HIGH orphaned claims found + now BANNERED in place (phase1b glass-win/74%-variance;
   iters-8-9 "credible lead"; Part 2 jumpy "+44/+80%" shaped-return win; Part 3 R²=0.9994 §8 re-affirmation + jumpy
   +1017). Thread A/E stale status boards fixed (A2/A3 NULL). Remaining MED items listed in the review doc (Part 4/5
   "0.79 tie vs PPO 0.81" vs canonical round-7 0.716<0.810 unreconciled; Part 6 planning-thesis needs banner;
   R²≈0.999 still cited in Part 5 reality-check/Part 6; unsourced "Pendulum 836 vs 46"/"3–4 orders" in Part 5 method map).

### ✅ STAND n=3 + WALKERRUN n=4 (2026-07-03 ~15:30) — TD-MPC2 Stand 962.3/948.1/942.7 (s43 best at 0.3M!, STAND_BUDGET_DONE); WalkerRun decisive arms n=4: none 731/680/699/723, value 56/28/39/38 dead 4/4, policy 76/64/83/53 dead 4/4 (s3/4 through >=900k). Mechanism table + all wall/barrier cells now at final n across the board. Part 9 updated.
  ↳ 2026-07-03 ~16:20 WM_ABLATION_WALK_S34_DONE formal marker: policy s3/4 finals unchanged at 20/20 evals (83.1/53.1, dead) — published n=4 numbers stand.

### ⚠ WALL-GENERALIZATION PROBE, HumanoidWalk (2026-07-03 ~18:20, in progress) — PPO NUMERICALLY UNSTABLE; SAC SOLVES EMPHATICALLY
b3060b exp/wallgen_humanoid. **brax PPO goes reward=nan on DMC HumanoidWalk under 3 configs** (defaults; WalkerRun-tuned transplant + 512^3 nets + reward_scaling 0.1; same + reward_scaling 1.0 + lr 1e-4 — nan by ~10M each; NO official tuned humanoid config exists in dm_control_suite_params, itself notable). Cannot claim an exploration wall — this is a NUMERICAL/config fragility failure mode, distinct from the hopper wall, and honestly reported as such. **SAC (same untouched driver, zero tuning): 909.5 / 900.7 (n=2, 5M) — solves HumanoidWalk emphatically; s43/44 launching → n=4.** TD-MPC2 1M s41 in flight (~7h). Verdict shape: on the 2nd unstable morphology the contrast is PPO fragile-to-unusable vs off-policy robust — a different (config-robustness) axis than the hopper (exploration) axis. Part 9 addendum when TD-MPC2 lands.

### 🚩 HUMANOID BOUNDARY (2026-07-03 ~19:05): TD-MPC2@1M FAILS HumanoidWalk (best 30.4 mppi @650k, falls=1.0; CSV HumanoidWalk_wallgen_hum_s41) while SAC solves @5M (909.5/900.7/625.3, n=3). The matched-1M hopper superiority does NOT generalize to the 21-DoF humanoid at 1M — honest boundary. Fair-budget completion launching: TD-MPC2 4M s41 (overnight); SAC s44 → n=4. PPO=nan-fragile (3 configs). Part 9 addendum when 4M lands.

### ✅ 5M ANCHORS n=4 (2026-07-03 ~21:00, HOP5M_ANCHORS_DONE) + humanoid fragility note
HopperHop TD-MPC2 5M anchor finals (CSV HopperHop_hop5m_s3{3,4}): s33 best 393.2 (mppi @4.25M), s34 best 295.0 (@4.0M) → n=4 = {477.4, 481.3, 393.2, 295.0}, mean ~412, range 295–481 — spread widened honestly. Part 9 §1 updated. ALSO: SAC HumanoidWalk s44 went reward=nan @2.2M (killed, replaced by s45) — the humanoid numerical fragility hits SAC stochastically (1/4 seeds) vs PPO deterministically (3/3 attempts); SAC solves 3/4 non-nan (909.5/900.7/625.3).
  ↳ 21:25 SAC HumanoidWalk s45 DONE 894.0 → FINAL SAC humanoid rate: 4/5 solve (909.5/900.7/625.3/894.0), 1/5 nan (s44) — vs PPO 0/3 attempts (all nan). Off-policy robustly solves the 2nd morphology; on-policy can't run.
  ↳ 07-04 ~00:00 TD-MPC2 HumanoidWalk 4M s41: loss=nan scale=nan at ~2.46M (best 22.6 mppi @900k before divergence) — TD-MPC2 ALSO numerically diverges on HumanoidWalk. Killed; s42 relaunching to test stochastic-vs-deterministic. HUMANOID TRIFECTA (draft claim): PPO nan 3/3 configs, TD-MPC2 fail@1M + nan@2.5M, SAC 4/5 solve — only plain off-policy is robust on the 21-DoF morphology; the hopper reliability ordering INVERTS. WalkerRun mech n=4 ALL 5 arms complete (none 699/723; value 39/38 dead; policy 83/53 dead; reward mppi 44 dead but pi 681/684 FULL; consistency 533/483 mildest, pi 653/667).

### ✅ MECH_COMPLETE (2026-07-04 ~03:25) — mechanism table final: n>=4 all decisive arms x 3 tasks
CheetahRun s3/s4 (exp/wm_head_ablation/jsonl): none 721/795; value 37/58 mppi (pi 7/8) DEAD; policy mppi 192/141 limps, pi 9/11 DEAD. Combined w/ s1/s2 + WalkerRun all-arms n=4 + HopperHop value n=6: the value-pathway law is fully firmed. Part 9 §3 updated + §4b humanoid addendum published (trifecta w/ s42-pending caveat).

### ✅ MECHANISM TABLE FULLY COMPLETE (2026-07-04 ~06:10, CHEETAH_FINAL2_DONE) — every arm n>=4 x 3 tasks
CheetahRun reward s3/s4: mppi 31/26 DEAD, pi 796/805 FULL (n=4 w/ 5/761). Consistency s3/s4: 516/575 + 558/623 (n=4 w/ 367/541) — mildest confirmed. Nothing in the 5-arm x 3-task table remains below n=4. Part 9 §3 final.

### 🚩 HUMANOID-STAND PROBE (2026-07-04 ~06:20): TD-MPC2 loss=nan by ~282k on HumanoidStand TOO — divergence is MORPHOLOGY-WIDE, not task-difficulty (hopper Stand solved at 0.3M; humanoid Stand nans by 0.28M). Killed. Walk 4M s43/s44 healthy at ~230k (divergence tally pending: s41 nan@2.46M, s42@~600k+ running, humstand nan@0.28M). §4b updated.
  ↳ 06:50 walk s44 loss=nan @~0.53M (killed). DIVERGENCE TALLY: nan 3 (walk s41@2.46M, walk s44@0.53M, stand@0.28M) vs healthy-so-far 2 (s42@1.03M — log is tdmpc2_4m_seed42.log; s43@0.46M). TD-MPC2 humanoid divergence looks STOCHASTIC-but-frequent (~60%+ of runs), onset 0.3-2.5M.
  ↳ 09:30 walk s42 loss=nan @~1.74M (killed). TALLY: DEFAULT-CONFIG walk nan 3/4 long-running (s41@2.46M, s42@1.74M, s44@0.53M; s43 healthy @1.79M) + stand nan@0.28M; knob runs healthy (ku64@0.95M, ku32@1.20M) + s45/s46 early (0.66M). Default-config humanoid divergence now looks like the RULE (~75%), not the exception; knob hypothesis strengthening.

### 🚩 HUMANOID DIVERGENCE VERDICT SHIFT (2026-07-04 ~10:55): s43 nan@2.51M + s45 nan@1.32M + ku32 nan@3.13M (all killed). DEFAULT CONFIG NOW 5/5 NAN (s41@2.46M s42@1.74M s43@2.51M s44@0.53M s45@1.32M; + stand@0.28M) — divergence is EFFECTIVELY DETERMINISTIC with variable onset 0.5-2.5M, NOT stochastic. k_update reduction DELAYS but does not prevent (ku32=32 nan@3.13M vs defaults <=2.5M). Pending: s46@1.32M, ku64@1.90M. §4b rewording at their resolution.
  ↳ 13:15 s46 loss=nan @~2.44M (killed). DEFAULT CONFIG FINAL: 6/6 walk seeds nan (s41@2.46 s42@1.74 s43@2.51 s44@0.53 s45@1.32 s46@2.44M) + stand@0.28M = 7/7. ku64 last survivor @3.49M clean BUT best only 20.1 (pi @1.55M) — even nan-free, NOT learning to walk. ku64 4M resolution next; then §4b final rewrite.

### ✅ HUMANOID CAMPAIGN COMPLETE (2026-07-04 ~14:20) — ku64 ONLY finisher of 9: 4M clean, best 21.8 (pi @3.75M, final 11.8) — NEVER LEARNED. FINAL §4b: default 6/6 walk nan (0.53-2.51M) + stand 0.28M, best-before-nan <=30.4; k_update shifts onset stochastically (ku32 nan@3.13M, ku64 finisher); nan-free survivor flat ~20 → failure NOT merely numerical, TD-MPC2 doesn't crack 21-DoF at 4M any setting; SAC 4/5 solve 625-909 zero-tuning; PPO nan 3/3. Part 9 §4b final + §1 budget-labeling fix (PPO Stand n=8 budgets 120-285M) pushed. OVERNIGHT NEXT: Stand 5-arm mechanism seeds 1-4 (b3060, STAND_MECH_DONE ~10h) + PPO Stand rate n=8→16 @285M + SAC n→10 (b3060b, RATE_HARDEN_DONE ~6h) + Paper 3 assembly.

### ✅ PPO STAND RATE n=16 (2026-07-04 ~15:50, seeds 49-56 @285M ALL WALLED: 137/155/120/129/128/195/105/107) — ESCAPE RATE FINAL 2/16 (12.5%; both escapes s44=681 s45=749 were 120M-budget runs; all four 285M runs walled). The graded barrier is near-categorical: escape is rare, not 25%. SAC s47-50 in flight (n→10). Part 9 §1 updating.

### ✅ RATE_HARDEN_DONE (2026-07-04 ~16:20) — SAC Stand n=10 FINAL: solves 7/10 (492/754/464/921/932/897.4/924.1; fails 33/100/300 @5M). Stand three-method gradient FINAL: TD-MPC2 3/3 (<=0.9M) >> SAC 7/10 (5M) >> PPO 2/16 (120-285M). Stand-mech none arms complete 4/4 solving (911-939 mppi) — harness validates; value arms in flight.

### ✅ SAC HOP RATE n=12 FINAL (2026-07-04 ~17:15, SAC_S912_DONE): >=200@5M in 6/12 (207/235/274/273/273/238; fails 47/131/67/76/89/115) = 50% at 5M; 20M-run crossings still 5/5 by ~8M. Budget-indexed escape curve final: 50%@5M -> 100%@~8M (vs PPO 0/5@472M, TD-MPC2 6/6@1M). TD-MPC2 Stand s44 healthy, s45 relaunch OK (3rd attempt; lesson: pkill pattern matched launch env-var in same cmdline).

### ✅ STAND TD-MPC2 n=5 + STAND-MECH VALUE CELL (2026-07-04 ~18:20): TD-MPC2 HopperStand 5/5 solve — 962.3/948.1/942.7/959.3/948.1 (s45 @600k, finishing). STAND-MECH: value arms DEAD 4/4 (6-13 mppi, pi <=12) vs none 911-946 — the value-pathway law holds on the 4TH task, one the full model solves in 0.3M: without TD value loss it cannot even STAND. Policy/reward/consistency waves overnight.

### ✅ ACROBOT DISCRIMINATOR COMPLETE (2026-07-04 ~21:55, ACROBOT_PROBE_DONE + SAC20M_ACRO_DONE)
AcrobotSwingup = unstable but CONTACT-FREE. PPO n=4 @285M: 290.4/266.5/344.3/285.7 — learns fine, NO WALL → the on-policy wall REQUIRES contact-criticality, not instability alone. SAC: 88.6/53.8/60.5 @5M (n=3) and 42.2/207.1 @20M (n=2) — slow AND inconsistent even at 4x budget → the off-policy advantage is also task-class-specific. TD-MPC2: 429.5/422.0 (n=2, <=1M) — best on BOTH task classes; the value-pathway advantage is the most portable of the three. Part 9 §1 discriminator para + Stand 5/5 fold; Paper 3 updated.

### ✅ STAND MECHANISM COMPLETE (2026-07-05 ~00:25, 19/20 arms formal, consistency_s2 @13/20 verdict-locked) — 4TH TASK, n=4 ALL ARMS
HopperStand: none 937/926/946/911 (pi 924-950) all solve; value 7/13/9/6 DEAD; policy 34/18/9/20 (pi<=5) DEAD; reward mppi 270/301/542/265 vs pi 944/515/943/944 PLANNER-PLUMBING; consistency 898/816/821/818 (pi 760-930) MILDEST (near-full). The 5-arm law now spans FOUR tasks. Part 9 §3 updated.

### ✅ NIGHT-FINAL HARDENING (2026-07-05 ~00:55, ACRO_N4_DONE):
TD-MPC2 Acrobot n=4: 429.5/422.0/454.1/445.9 (<=1M) — tight, best of 3 methods. SAC-20M Acrobot n=4: 42.2/207.1/65.6/123.4 — slow+inconsistent CONFIRMED. **SAC HumanoidStand 918.5/922.5 (n=2) — SOLVES.** HUMANOID MATRIX COMPLETE: SAC solves BOTH tasks (Walk 4/5 625-909, Stand 2/2 ~920); PPO nan all configs; TD-MPC2 diverges/fails all runs. §4b + discriminator numbers final everywhere.

### ✅ ENTROPY-KNOB CONTROL (2026-07-05 ~01:45, ENTROPY_SWEEP_DONE): HopperHop wall SURVIVES entropy_cost x3 (peaks 4.2/73.9) and x10 (48.1/64.1), n=2 each @150M — same <=74 class as baseline 53.8, none near 200. The categorical wall is NOT an under-exploration-hyperparameter artifact. Part 9 §1 + Paper 3 updated.

### ⚠️→✅ STAND ENTROPY CONTROL (2026-07-05 ~10:35, STAND_ENT_DONE) — HONEST REVISION, informative:
PPO HopperStand entropy_cost x3: 118.3 / **627.3 — 1 of 2 seeds ESCAPED**; x10: 131.6/141.0 (both walled). Contrast: HopperHop survived x3 AND x10 (4-74, 4 arms). VERDICT: the exploration knob can occasionally shift the GRADED Stand barrier (consistent with baseline 2/16 escapes) but never the CATEGORICAL Hop wall. Sharpens graded-vs-categorical with a causal knob. Part 9 §1 + Paper 3 updated with the split verdict.

### ✅ STAND ENTROPY n=4 FINAL (2026-07-05 ~11:35, STAND_ENT2_DONE): em3: 627.3/118.3/149.9/141.4 = **1/4 escaped**; em10: 131.6/141.0/184.7/115.5 = **0/4** (all in 105-195 wall band). vs baseline 2/16. VERDICT FIRMED: entropy x3's 1/4 is consistent with the baseline graded escape lottery (~1-in-8), NOT a repair; x10 0/4. Neither dents the categorical Hop wall (4-74). Wording finalized in Part 9 §1 + Paper 3.

### ⚠️ SAC 8M DIRECT-CONFIRMATION (2026-07-05 ~12:35, SAC8M_DONE) — HONEST REVISION of the 8M cell:
Direct 8M runs, fresh seeds: HopperHop s51 139.2, s52 139.9 — **0/2 crossed 200 by 8M**. The earlier '5/5 by ~8M' came from crossings inside the 20M cohort; combined rate is now **5/7 by 8M** (71%) — SAC's Hop crossing is a rate, not a certainty, at 8M. HopperStand s51 518.0, s52 453.3 — 2/2 >=200 at 8M → cumulative Stand escapes 9/12. (Ops note: both envs wrote seed<n>.json to one outdir — one pair overwritten; recovered from per-env logs. Fix driver outdir next time.)

### ✅ SAC 8M DIRECT n=4 FINAL (2026-07-05 ~13:05, SAC8M_B_DONE): Hop direct-8M: 139.2/139.9/116.7/374.1 = 1/4 crossed → combined by-8M rate **6/9 (~2/3)** (20M cohort 5/5 + direct 1/4). Stand direct-8M: 518.0/453.0/498.6/**920.6** = 4/4 >=200 (one near-solve) → cumulative Stand 11/14. Cell finalized: SAC Hop crossing by 8M is a ~2/3 rate; Stand escape by 8M looks reliable.

### ✅ 5M ANCHOR n=12 FINAL (2026-07-05 14:45, HOP5M_N8_DONE + HOP5M_B_DONE): TD-MPC2 HopperHop 5M bests, 12 seeds: 477.4/481.3/393.2/295.0 (orig) + 604.1/314.5/499.7/364.8 (s35-38) + 315.8/376.9/306.8/610.4 (s39-42) = **mean 420.0 ± 112.7 (sd), range 295-610**. Flagship efficiency anchor at publication n. CAMPAIGN EVIDENCE TABLE COMPLETE.

### ✅ PAPER-4 SUFFICIENCY TEST, TASK 1 (2026-07-05 22:27, SUFF5M_DONE): consistency-OFF HopperHop 5M, n=4: **165 / 475 / 481 / 511** (pi 228/516/495/511), mean 408 vs full n=12 anchor 420±113 (median ~385). 3/4 seeds at the TOP of the full band; 1 seed low (165 < full floor 295). VERDICT: consistency loss removable for typical seeds on HopperHop; residual role = worst-seed insurance (regularizer). SUFFICIENCY GO (task 1); WalkerRun 2x2 next. Also: stripped seeds reached 440+ by ~2M — faster-than-full trajectory noted (n small).

### ✅ PAPER-4 TASK-2 FULL BASELINE (2026-07-06 ~00:00, WALKER5M_DONE): TD-MPC2 full WalkerRun 5M bests, n=4: **709.0 / 705.3 / 752.6 / 782.0** (mean ~737). vs 1M-era 680-731: near-asymptote by 1M, modest 5M gain. Stripped-WalkerRun (consistency-OFF, seeds 25-28) queued for b3060 after novelty sweep → completes the Paper-4 2x2.

### ⭕ NOVELTY-MPPI SWEEP NULL (2026-07-06 ~00:25, NOV_SWEEP_DONE): HopperHop 1M, 1 seed/arm: disagreement b0.3 **5.6 (never learned — bonus can hurt)**; disagreement b1.0 284.6 (cross 750k); rnd b0.3 258.7 (cross 750k); rnd b1.0 273.8 (cross 500k). vs vanilla mean ~282 best 367. VERDICT: no improvement in the regime vanilla already solves; rnd b1.0's 500k crossing is the only positive hint. Honest scope: the exploration lever's real test is a frontier task where vanilla TD-MPC2 fails — deferred. Part 5 proposal-A's novelty-MPPI half now has its first (null) datapoint.

### ⭕ NOVELTY-MPPI MATCHED-SEED FINAL (2026-07-06 03:20, VCTRL_DONE): vanilla HopperHop 1M seeds 61-64: **442.3/508.5/321.4/359.4** (4/4 crossed; 250k-750k) vs novelty same-seeds 5.6/284.6/258.7/273.8. **Novelty WORSE on 4/4 matched seeds; one catastrophic break (dis0.3: 442→5.6).** Part-5 proposal-A novelty-MPPI half CLOSED NULL in the solvable regime. Side-note: fresh vanilla 1M cohort mean ~408 vs old cohort ~282 — 1M seed-batch variance is large; budget-indexed rates remain the right currency.

### ✅ PAPER-4 2x2 COMPLETE (2026-07-06 11:00, SUFFWALK_DONE): stripped-Walker 5M finals **537.4/573.9/554.3/594.2 (mean 565)** vs full 709/705/753/782 (mean 737) = **-23%, non-overlapping ranges**. With Hop (stripped 165/475/481/511 >= full 420±113): **the consistency loss is TASK-CLASS-INDEXED — removable where exploration dominates (Hop), load-bearing where dense tracking dominates (Walker)** — mirroring every ordering result in this program. Cheetah task-3 (both arms in flight/queued) decides law-vs-coincidence.

### ✅ PAPER-4 TASK-3 FULL BASELINE (2026-07-06 12:50, CHEETAH5M_DONE): TD-MPC2 full CheetahRun 5M bests, n=4: **903.1/904.0/781.5/806.2 (mean 849)**. vs 1M-era 721-795: real 5M gain (unlike Walker's near-asymptote). Stripped-Cheetah s31-34 running on b3060 (~18:30) → 2x3 verdict: Walker-like gap = dense-task LAW; match = honest nuance.

### ✅ PAPER-4 TASK-3 STRIPPED (2026-07-06 20:20, SUFFCHEETAH_DONE): stripped-Cheetah 5M finals **526.7/528.1/516.2/524.1 (mean 524)** vs full 903/904/782/806 (mean 849) = **-38%, non-overlapping**. DENSE LAW CONFIRMED on a second dense task: Walker -23%, Cheetah -38% — the consistency loss is dense-state-tracking machinery, removable only where directed exploration is the bottleneck (Hop). Acrobot (exploration side, task 4) both arms next; Cartpole-sparse (task 5) queued.

### ✅ PAPER-4 TASK-4 FULL BASELINE (2026-07-06 22:37, ACRO5M_DONE): TD-MPC2 full AcrobotSwingup 5M bests, n=4: **533.3/511.2/512.9/488.1 (mean 511)**. vs 1M-era 422-454: real 5M gain. Stripped-Acrobot s35-38 running (~03:30) → exploration-side test #2. Cart-sparse (task 5) launching now on b3060b.

### ⚠️ PAPER-4 TASK-4 STRIPPED (2026-07-07 05:16, SUFFACRO_DONE): stripped-Acrobot 5M finals **297.0/232.7/351.8/256.4 (mean 284)** vs full 533/511/513/488 (mean 511) = **-44%, non-overlapping**. THE CLEAN EXPLORATION/DENSE SPLIT BREAKS: removability held only on Hop (3/4 top-of-band). Revised reading: the consistency loss supports MPPI rollout quality wherever the planner carries learning (Acrobot -44%, Walker -23%, Cheetah -38% are all planner-led); HopperHop — where the policy head can learn the behavior directly — is the removable case. Cart-sparse (task 5, both arms in flight) tests this revision.

### PAPER-4 TASK-5 FULL BASELINE (2026-07-07 07:45, CART5M_DONE): TD-MPC2 full CartpoleSwingupSparse 5M bests, n=4: **0.0/0.0/1.3/0.0** — vanilla TD-MPC2 stalls this sparse task at 5M (matches its known Part-18-era behavior; expl_until=25000 default). Task-5 sufficiency cell is therefore likely uninformative BOTH-FAIL (stripped s41-44 verdict ~13:30); the sufficiency grid's evidentiary core = the 4 solved tasks (Hop removable; Walker/Cheetah/Acrobot -23/-38/-44%).

### 🏁 PAPER-4 TASK-5 STRIPPED + 2x5 GRAND VERDICT (2026-07-07 14:12): stripped-CartpoleSwingupSparse 5M finals **0.00/0.00/0.00/0.00 (n=4)** = full baseline 0/0/1.3/0 → **BOTH-FAIL cell, uninformative for sufficiency** (the full model itself stalls this sparse task; not a probe of the consistency loss). 
**2x5 SUFFICIENCY GRID — FINAL (evidentiary core = the 4 tasks the full model solves):**

| Task | stripped consistency-OFF @5M (n=4) | full baseline | gap | verdict |
|---|---|---|---|---|
| HopperHop (pi-learnable) | 165/475/481/511 | 420±113 (n=12) | ~0 | REMOVABLE (3/4 top-of-band) |
| WalkerRun | 537/574/554/594 (565) | 709/705/753/782 (737) | -23% | load-bearing |
| CheetahRun | 527/528/516/524 (524) | 903/904/782/806 (849) | -38% | load-bearing |
| AcrobotSwingup | 297/233/352/256 (284) | 533/511/513/488 (511) | -44% | load-bearing |
| CartpoleSwingupSparse | 0/0/0/0 | 0/0/1.3/0 | n/a | both-fail (uninformative) |

**THESIS (survives all 4 informative tasks): the consistency (self-predictive) loss underwrites MPPI rollout quality wherever the PLANNER carries learning (Walker/Cheetah/Acrobot all planner-led → load-bearing); HopperHop — where the policy head learns the behavior directly and the planner mostly amplifies — is the one task where it is redundant.** An n=8 replication of the HopperHop removable cell (stripped s45-48) is finishing to firm the lone low seed (165).

### ✅ HOPPERHOP REMOVABLE CELL FIRMED n=8 (2026-07-07 14:58, HOPSTRIP_B_DONE): stripped consistency-OFF-from-scratch, HopperHop 5M, 4 new seeds **s45=306.4 / s46=448.6 / s47=443.1 / s48=477.3 (mean 419)** — combined with the original s0-3 **165/475/481/511 (mean 408)** = **n=8, 7/8 seeds solidly inside the full-model band 420±113 (n=12), one floor-dip (165)**. Port validated: b3060b ran the byte-identical ablation code (`_m_c=0.0 if ABLATE=='consistency'`, the same mask that produced Walker/Cheetah/Acrobot -23/-38/-44% drops), so stripped-Hop staying in-band is genuine removability, not a no-op. CONCLUSION: on HopperHop the self-predictive (consistency) loss is removable at n=8 — the policy head learns the behavior directly; the planner amplifies. This is the sole removable cell in the 5-task sufficiency grid.

### ❌ SOTA BET 1 — VALUE-AWARE CONSISTENCY (fixed λ=1): NO-GO (2026-07-07 19:25, interim-but-decisive). Paired same-seed VAC-vs-matched-vanilla derisk: WalkerRun (b3060, @~1.8M, n=2) VAC 678 vs van 744 = **−8.9%**; CheetahRun (b3060b, @~4M, n=2) VAC 782 vs van 843 = **−7.2%**. VAC underperforms uniform consistency at EVERY checkpoint 1M→present on BOTH planner-led tasks (stable, monotone, 0 nan). VERDICT: weighting the consistency loss by per-dim value sensitivity |dQ/dz| does NOT improve planner rollout quality — it HURTS ~7-9%. Interpretation (reinforces the sufficiency thesis): the planner's MPPI rollouts need FAITHFUL dynamics on every latent dimension they may explore; concentrating model capacity on currently-high-value dims starves the rest. So uniform consistency is not just load-bearing (Paper 4) but near-OPTIMAL in form. NOTE: honest progress correction — Walker was misreported as ~4M in prior ticks; es= shows ~1.8M (VAC's extra jax.grad halves b3060 throughput). Runs continue to clean 5M for citable numbers. Next lever = strategic fork (annealed-VAC vs a different objective) — surfaced to user.

### bet-1 VAC Walker 5M FINAL (2026-07-08 02:43, VAC_WALKER_DONE): VAC 684.1 vs matched vanilla 750.9 (n=2 paired) = **-8.9%**. Bet-1 VAC now closed on both planner-led tasks at full 5M: Walker -8.9%, Cheetah -4.4% → confirmed NO-GO. Uniform consistency near-optimal in form.

### ❌ SOTA BET 2 — URC (uncertainty/rollout-reliability consistency): NO-GO (2026-07-08 10:47, URC_CHEETAH_DONE). Weighting the consistency loss by open-loop-vs-teacher-forced rollout drift. Paired same-seed vs matched vanilla: **CheetahRun 5M urc 784.9 (752/818) vs van2 855.4 (917/794) = −8.2%**; WalkerRun interim ~−7% (n=2, 0 nan). URC underperforms uniform consistency at every checkpoint 1M→5M on both tasks. **The reweighting family is now CLOSED** (VAC −8/−4%, URC −8/−7%): weighting the consistency loss by value-sensitivity OR by rollout-uncertainty both hurt ~5-9% → TD-MPC2's UNIFORM consistency loss is near-optimal in form. Mechanism: the MPPI planner needs faithful dynamics on every latent dim it may explore mid-rollout; any reweighting starves the dims that become decisive. Full analysis: REPORT_consistency_reweighting_failure.md. NEXT: bet-3a value-conditioned abstraction (bisimulation sweep) — structure into the value pathway, not a reweighting.

### URC-Walker 5M FINAL (2026-07-08 13:10, URC_WALKER_DONE): urc 699.7 (689/710) vs matched vanilla 727.2 (702/752) = **-3.8%** (n=2, 5M). Bet-2 URC now closed on BOTH planner-led tasks at full 5M: Cheetah -8.2%, Walker -3.8% → reweighting the consistency loss by rollout-drift does not help. Reweighting family FULLY CLOSED (VAC Cheetah -4.4%/Walker -8.9%; URC Cheetah -8.2%/Walker -3.8%).

### ❌ SOTA BET 3a — VALUE-CONDITIONED ABSTRACTION (bisimulation metric): STRONG NO-GO (2026-07-08 15:35, BIS_CHEETAH_DONE). Added a bisimulation-style latent metric (states pulled together by reward+transition equivalence; --bisim_coef) — structure the value pathway consumes by construction. Paired vs matched vanilla, CheetahRun 5M (n=2 each coef): **bisim_coef=0.1 → 460.4 (427/494); bisim_coef=0.5 → 388.4 (435/341); vanilla van2 = 855.4 → -46% / -55%.** Both coefficients HARM badly, from ~900k onward, no recovery — a far larger hit than the reweighting bets (-4 to -9%) because bisimulation reshapes the ENTIRE latent geometry rather than reweighting one loss. **Mechanism:** TD-MPC2's SimNorm+TD latent is already value-sufficient (held-out value-decode R²≈1); imposing an explicit value-conditioned metric distorts the representation the value pathway would learn on its own. **The redundancy result now spans the full spectrum: added-structure (Paper A/glass/graph) + loss-reweighting (VAC/URC) + value-conditioned-metric (bisim) — ALL null-to-harmful on dense value-based control.** NEXT: bet-3b value-sufficient bottleneck (architectural, not metric): split z=[z_v,z_r], Q/π read only z_v.

### ⚖️ SOTA BET 3b — VALUE-SUFFICIENT BOTTLENECK: GRADED NULL / value-sufficiency curve (2026-07-08 20:10). Architectural (not metric) structure: Q and π read only the first D dims of the 512-dim SimNorm latent (env-gated _vslice in QEnsemble+Pi __call__, byte-identical at D=512); dynamics/consistency/reward keep full z. CheetahRun 5M width sweep (n=1/width, s50) vs matched vanilla van2=855:

| bottleneck D | % of latent | return | % of vanilla |
|---|---|---|---|
| 16 | 3% | 496 | 58% |
| 32 | 6% | 563 | 66% |
| 64 | 12.5% | 639 | 75% |
| 128 | 25% | 753 | 88% |
| 512 (vanilla) | 100% | 855 | 100% |

**Clean monotone diminishing-returns curve; NO bottleneck width recovers full performance.** Structure-via-architecture, like the metric (bisim) and reweighting (VAC/URC) bets, can only match-or-hurt — never beat vanilla. But it degrades GRACEFULLY (unlike bisim's -55% crater): 25% of dims buys 88%. Interpretation: **the value pathway reads DISTRIBUTED latent information — there is no small value-sufficient subspace that recovers everything.** This completes the redundancy result: ALL four forms of imposed structure — added-structure (Paper A glass/graph), loss-reweighting (VAC/URC), value-conditioned-metric (bisim), architectural-bottleneck (VBN) — are null-to-harmful on dense value-based control. The novel positive artifact is the width→return **value-sufficiency curve** itself. NEXT: confirm curve on WalkerRun (2nd task).

### bet-3b 2-TASK CONFIRMATION — value-sufficiency curve replicates on WalkerRun (2026-07-09 00:35). Same VBN bottleneck (Q/π read first D of 512-dim latent), WalkerRun 5M width sweep (n=1/width, s50) vs matched vanilla van2_WalkerRun 727:

| bottleneck D | Cheetah (% of 855) | Walker (% of 727) |
|---|---|---|
| 16 | 496 (58%) | 591 (81%) |
| 32 | 563 (66%) | 642 (88%) |
| 64 | 639 (75%) | 665 (91%) |
| 128 | 753 (88%) | 695 (96%) |

**Same monotone diminishing-returns shape on both tasks** — return rises with bottleneck width, none fully recovers vanilla, graceful (no crater). Walker's curve sits higher (Walker's value pathway is slightly more compressible — 128d gets 96% vs Cheetah's 88%), but the qualitative law is identical: **the value pathway reads a distributed latent, and you can bottleneck Q/π to a fraction of it at a graded, width-dependent cost.** Confirms the bet-3b Cheetah finding is a robust property, not a single-task artifact. The value-sufficiency curve is the program's novel positive artifact.

### ✅ TWO-AXIS HOP (Part-12 crux, clean 5M, 2026-07-09 03:34, TWOAXIS_HOP2_DONE). HopperHop, full-WM (ABLATE=none) vs stripped-WM (ABLATE=consistency), n=2 (s62/63), evaluating BOTH the MPPI-planner return and the raw-policy return each eval:

| arm | mppi | π (policy-only) | planning help? |
|---|---|---|---|
| full WM (none) | **571** (554/587) | 542 (506/577) | **yes, mppi>π both seeds** |
| stripped WM (consistency) | 421 (386/455) | **448** (386/511) | **no — π≥mppi** (tie, then 511>455) |

**The Part-12 crux confirmed at 5M: MPPI planning beats the raw policy ONLY when the world model is present; strip the world model and planning adds nothing (the policy alone is as good or better).** So on HopperHop, the planner's value comes from the world model's rollout quality — and since the world model is *removable* there (stripped ≈ 421-448, within the earlier removable-Hop band ~420±113), TD-MPC2's Hopper win is carried by the TD value + policy, not by planning-over-the-world-model. This is the mechanistic evidence for the Part-12 argument that Hopper's flagship win is a TD-learning win, not a world-model win. (Caveat: stripped-vs-full absolute gap here is n=2-noisy — this full pair drew high seeds 554/587 vs the historical 420 band; the *within-arm* mppi-vs-π comparison is the robust signal.)

### bet-3b 3-TASK value-sufficiency curve — Acrobot (2026-07-09 04:45, ~5M). VBN bottleneck (Q/π read first D of 512), AcrobotSwingup width sweep (n=1/width, s50) vs vanilla ~511: D=16→186 (36%), D=32→318 (62%), D=64→133 (26%, unlucky-low seed outlier), D=128→481 (94%). Full 3-task table:

| D | Cheetah (/855) | Walker (/727) | Acrobot (/511) |
|---|---|---|---|
| 16 | 496 (58%) | 591 (81%) | 186 (36%) |
| 32 | 563 (66%) | 642 (88%) | 318 (62%) |
| 64 | 639 (75%) | 665 (91%) | 133 (26%)* |
| 128 | 753 (88%) | 695 (96%) | 481 (94%) |
| 512 | 855 (100%) | 727 (100%) | 511 (100%) |

*Acrobot D=64 is an n=1 unlucky-low outlier — the single-seed Acrobot curve is noisier than the clean monotone Cheetah/Walker curves; the robust signal is D=128 ≈ 94% ≫ tight widths. **Verdict across 3 tasks: the value-sufficiency law holds — return rises with bottleneck width, no width fully recovers vanilla, the value pathway reads a distributed latent.** Compressibility varies with task (Walker most compressible at 128→96%, Cheetah 88%, Acrobot's tight widths hardest), but the qualitative law is universal. This closes bet-3b: architectural bottlenecking, like the metric and reweighting bets, cannot beat vanilla — completing the redundancy result across all four structure forms, with the width→return curve as the positive artifact. (Acrobot could be de-noised with 2-3 more seeds if a reviewer asks.)

### bet-3b Acrobot n=2 update (2026-07-09 09:35, VBN_ACROBOT2_DONE). Adding seed 51 resolves the s50 D=64 outlier (133→ n=2 291). AcrobotSwingup n=2 value-sufficiency curve vs ~511: D=16→280 (55%), D=32→351 (69%), D=64→291 (57%), D=128→491 (96%). Cleaner than single-seed: 128 recovers 96%, tight widths cluster ~280-350 (~55-69%) with mild seed noise (32/64 near-tie) — the robust claim '128≫tight, Acrobot least-compressible' holds at n=2. Confirms the 3-task value-sufficiency law with the noisy arm de-noised.

### Walker two-axis (crux-generalization, 2026-07-09 10:10, TWOAXIS_WALKER_DONE). Full-WM (ABLATE=none) vs stripped-WM (ABLATE=consistency) HopperHop→WalkerRun contrast, n=2 (s62/63), 5M: **full-WM mppi 719 (740/698) ≈ vanilla 727; stripped-WM mppi 665 (662/669) = −7.5%.** On WalkerRun the world model is **load-bearing** — removing it costs performance — CONTRASTING HopperHop where the world model is removable (stripped ≈ full ≈ 420). This generalizes the Part-12 picture: the world model matters exactly where the *return level* needs accurate multi-step rollouts (dense/planner-led Walker), and is dispensable where the behavior is a low-dim limit cycle (Hop). NOTE: the within-arm MPPI-vs-π (does planning help) read is inconclusive here — jsonl pi_return == mppi (tag-collision/logging artifact from reusing the Hop wmabl tags); the clean signal is the stripped-vs-full contrast. Feeds Part 12.

### ✅ H3 VERDICT — HopperHop PPO wall is a CONJUNCTIVE-REWARD-DESIGN artifact (2026-07-09 11:20, HOP_H3 PPO done 20M n=12). Env-gated reward variants (patched mujoco_playground hopper.py, HOP_REWARD_MODE/HOP_SPEED, backup .bak_hop). Tuned PPO on HopperHop, seed 50, 20M:

| variant | reward | PPO final return |
|---|---|---|
| default (control) | product standing×hopping | **0** (WALL) |
| additive | 0.5·standing + 0.5·hopping | **135** (ESCAPES wall) |
| product + hop_speed=1.0 | product, easier hop threshold | **1** (WALL persists) |
| tdmpc2 additive (control) | additive | 467 (TD-MPC2 fine) |

**The multiplicative/conjunctive reward IS the wall.** PPO climbs off zero under the ADDITIVE reward (0→135) but stays walled under the PRODUCT reward EVEN with an easier hop-speed threshold (0→1). So the barrier is specifically the **conjunction** (standing×hopping stays ≈0 until BOTH are solved), NOT the hop-speed magnitude, NOT termination (HopperHop has none — done=isnan only), and NOT a fundamental PPO capability limit. Off-policy TD+replay (SAC/TD-MPC2) tolerates the conjunctive-sparse structure that starves on-policy PPO's advantage estimate. **Voelcker H3 CONFIRMED + mechanistically pinpointed.** REFINES Part-12: TD-MPC2's 'categorical PPO wall on HopperHop' is CONDITIONAL on the standard DMC multiplicative reward — an on-policy-vs-off-policy EXPLORATION gap manufactured by conjunctive sparsity, not proof PPO can't hop. Combined with Q1 isolation (stripped-TD-MPC2 beats SAC speed → win=off-policy TD actor-critic, not the world model) and the Walker-vs-Hop WM contrast (WM load-bearing on Walker -7.5%, removable on Hop), this completes the mechanistic account of what makes HopperHop unique.

### AUDIT ADDENDUM (2026-07-09 16:20, post-weekly code/exp review). (1) **jsonl tag collision (FIXED)**: run_arm.sh tag lacked task name → Hop s62/63 jsonl files were APPENDED by the later Walker two-axis runs (same seeds); the H3/two-axis Hop numbers ledgered at f388691 were harvested BEFORE contamination and the task-named benchmark CSVs are the durable source — but do NOT re-harvest Hop s62/63 from jsonl. Walker π-vs-MPPI (marked inconclusive at 0726172) is explained by this collision. Tag now includes TASK (.bak_tagfix). (2) **Cheetah sufficiency batch never launched** (silent heredoc failure; earlier status mis-attributed) — now running s70/71 with task-qualified tags. (3) **H3 claim precision**: PPO-additive = steady climb to 135 @20M (curve 3→135, not plateaued), well below standing-component ceiling ~500 → finding is 'learnable signal vs flat 0', not 'task solved'; HOP_SPEED=1.0 variant also halves tolerance margin (margin=speed/2) — margin-controlled variant recommended for paper version. (4) **Cross-version baselines**: vbn-vs-van2 comparisons span patch stacks; VBN/VAC/URC gates verified default-off (byte-identical unset) — state in paper setup.

### A1 — Cheetah VBN n=2 (2026-07-09 20:45, seed 51 finals @~5M: 16→594, 32→572, 64→636, 128→742). Combined with s50 (496/563/639/753):

| D | s50 | s51 | n=2 mean | % of van 855 |
|---|---|---|---|---|
| 16 | 496 | 594 | 545 | 64% |
| 32 | 563 | 572 | 568 | 66% |
| 64 | 639 | 636 | 638 | 75% |
| 128 | 753 | 742 | 748 | 87% |

n=2 curve: 545/568/638/748 — the monotone-diminishing law holds with tight per-width agreement at D≥32 (s50-s51 gaps ≤11); D=16 has the largest seed spread (496 vs 594). Note 16/32 near-tie at n=2 (545 vs 568) — the floor region is flat; the informative slope is D≥32→128. No width recovers vanilla. s52 queued.

### A1 AMENDMENT — Cheetah VBN s51 true 5M finals (2026-07-10 04:15, A1_CHEETAH_S51_DONE marker; supersedes the 20:45 pre-completion harvest). The 20:45 numbers (594/572/636/742) were late-but-NOT-final evals; the completed-run finals at es=5,000,192 (mppi) are: **16→535, 32→582, 64→617, 128→723**. Corrected n=2 table:

| D | s50 | s51 (final) | n=2 mean | % of van 855 |
|---|---|---|---|---|
| 16 | 496 | 535 | 516 | 60% |
| 32 | 563 | 582 | 572 | 67% |
| 64 | 639 | 617 | 628 | 73% |
| 128 | 753 | 723 | 738 | 86% |

The corrected s51 is *cleaner* than the pre-completion read: strictly monotone within-seed (535<582<617<723, no 16/32 inversion), tight s50-s51 agreement at every width (gaps ≤39), and the n=2 curve 516/572/628/738 is strictly monotone with no flat floor region (the earlier '16/32 near-tie' was an artifact of harvesting W16 mid-fluctuation). Law unchanged: monotone diminishing, no width recovers vanilla. Lesson re-confirmed: harvest ONLY at markers, never from near-final evals. s52 launched (GPU0-3, marker A1_CHEETAH_S52_DONE).

### P4 — Walker sufficiency n=4 FINAL + Cheetah n=2 (truncated) + b3060 proc-kill event (2026-07-10 08:50)
**Event:** all 8 b3060 jobs died ~08:20 (no reboot — uptime 3w5d; ssh also flapping, cause TBD, suspect host-level pressure). Walker suff died at es=4,900,096 (98% of 5M), Cheetah suff at es≈4,550,144-4,600,064 (~91%). Harvested at last common eval, honest truncation noted.

**Walker load-bearing, n=4 (s62/63 @5M + s70/71 @4.90M):**
| arm | s62 | s63 | s70 | s71 | mean |
|---|---|---|---|---|---|
| full (none) | 740 | 698 | 704.8 | 689.8 | **708.2** |
| stripped (consistency-off) | 662 | 669 | 637.1 | 651.3 | **654.9** |

**Δ = −7.5% (n=4, tight: full range 690-740, stripped 637-669, non-overlapping).** The historical single-pair −23% is confirmed a seed/version outlier; the durable claim is **WalkerRun WM load-bearing ≈ −7 to −8%**, modest but consistent and non-overlapping at n=4.

**Cheetah sufficiency n=2 TRUNCATED @~4.55M (paired same-step, runs killed):** full 733.3 (s70) / 670.2 (s71, @4.60M), stripped 526.7 / 607.6 → means 701.8 vs 567.2 = **−19.2%**, but seed spread is wide (s70 −28%, s71 −9%). Historical single-seed −38% not reproduced at face value; honest read: **Cheetah WM load-bearing, magnitude seed-dependent −9 to −28% (n=2, truncated)** — needs n≥3 full-5M for a paper number.

### P2 PRE-BUILD DISCOVERY (2026-07-10 08:55, CRITICAL for Point-1/Paper 3): our TD-MPC2 implementation ALREADY collects with π+noise, NOT MPPI.
Read /root/helios-rl/scripts/run_benchmark.py collection loop (L1786-1803): ALL branches use act_fn_batch (tanh π-mean) + Gaussian EXPL_NOISE (+ optional random warmup/mix). batch_mppi_targets (MPPI) appears ONLY in an optional MPC-distillation loss block (L1981, coef-gated; default status TBC). **Implication: the "MPPI-as-structured-explorer-during-collection" hypothesis is ALREADY FALSIFIED-BY-CONSTRUCTION in our stack — every TD-MPC2 number we have (incl. Hop 6/6≥200 by ~1M, beating SAC's 6/9@8M) was achieved with policy-only data collection.** TD-MPC2's Hopper edge over SAC in our implementation must come from: (a) the value/actor architecture + losses (Q-ensemble, SimNorm, TD machinery), (b) the MPC-distillation loss IF default-on (planning shaping π via imitation, not via exploration), and/or (c) eval-time MPPI (excluded: π-only evals also clear the wall). P2 is therefore REDEFINED: if mpc-distill is default-on → P2 = distill-coef-0 ablation (does planning-shaped π-learning carry the edge?); if default-off → the attribution collapses to (a) the off-policy TD core itself, and the remaining SAC-vs-strippedTDMPC2 gap is architectural (P1 SAC-core isolation becomes the decisive experiment). NOTE for Part-12/Paper-3 framing: canonical Hansen TD-MPC2 collects WITH the planner; ours does not — an implementation deviation that is itself evidence (planner-collection unnecessary to beat PPO/SAC on Hop) but must be stated when citing "TD-MPC2".

### FLEET INCIDENT 2026-07-10 ~08:00-08:30 (both boxes) + root-cause review + fixes
All GPU jobs on BOTH b3060 and b3060b were killed in the same window (no reboots — uptimes 3w5d / 6w5d; no dmesg OOM records; RAM fine; b3060b's disk guard is file-only). External cause unidentified (possibly host/provider-level); the user's mahjong/botzone process on b3060b ALSO died (mj=0) — NOT restarted by us (user's process, report-only). Contributing race found on b3060: my old `kb_forever.sh` meaningful-queue cron DOUBLE-BOOKED GPUs 0-2 with 1M-step mq HopperHop jobs at 00:40-01:50 on top of the running suff batch (its idle-check races with eval/JIT low-util windows) — the known 2-roller race trap. **Fix: kb_forever cron DISABLED (commented `#DISABLED-0710-doublebook-race`), re-enable only with a lock.** Recovery: b3060b Walker VBN s52 + Cheetah VBN s52 relaunched fresh (dead 1.5M Cheetah s52 partials archived to _dead_partial_0710/ to avoid CSV append-contamination).

### A1 — Walker VBN s51 TRUNCATED harvest (killed at 4.0-4.5M of 5M; per-width es noted):
| D | s50 @5M | s51 (truncated) | n=2 mean | % of van 727 |
|---|---|---|---|---|
| 16 | 591 | 616.9 @4.50M | 604 | 83% |
| 32 | 642 | 661.6 @4.15M | 652 | 90% |
| 64 | 665 | 674.8 @4.00M | 670 | 92% |
| 128 | 695 | 720.4 @4.10M | 708 | 97% |

Truncation biases s51 LOW, yet every s51 width ≥ s50 — the monotone curve is robust; W128 recovers 97% (Walker = most compressible task, consistent with prior read). s52 relaunched for a clean full-5M seed.

### ✅ P1 LEG-1 — SAC-default vs (stripped-)TD-MPC2 on HopperHop, SAME stack (2026-07-10 10:05, P1_SAC_HOP_DONE, n=3 @5M)
SAC (helios custom v1, defaults: lr 3e-4, batch 256, auto-α) on HopperHop: **finals 76.4 / 23.0 / 101.2 — 0/3 seeds ever reach 200 by 5M.** Auto-tuned α collapsed to ~0.003 (entropy death → agent stands, never finds the conjunctive standing×hopping payoff). Contrast, same box/stack/env: stripped-TD-MPC2 (consistency OFF, planner-free training, π+noise collection) = **8/8 seeds ≥200 well before 5M (~420-450 final)**; full TD-MPC2 6/6 by ~1M. **Combined with the collection discovery (all our TD-MPC2 training is planner-free), the Hopper attribution now reads: the edge over BOTH PPO and SAC lives in the TD value/actor architecture + update itself (Q-ensemble/min, SimNorm latent encoder, TD horizon machinery, UTD/noise schedule) — not planning, not the world model, not collection strategy.** CAVEATS (honest): (i) this is our custom SAC v1, not a tuned reference SAC — historical external SAC showed 6/12 by 5M, so implementation/knobs matter; the entropy-collapse failure is itself mechanistic evidence (conjunctive reward starves the entropy-tuned explorer) but a knob-matched arm (α floor / fixed α, UTD-matched) is REQUIRED before any "SAC cannot" claim. (ii) n=3. Next: sac.py env-gated knobs (α floor, UTD), one knob at a time.

### ✅ P1 LEG-2 — "rescued" SAC FAILS AT ZERO (2026-07-10 12:10, P1_SAC2_HOP_DONE, n=3 @5M): the entropy needle
SAC with α-floor 0.05 + canonical target-entropy −1.0·|A| (gates verified bound: target_entropy=-4.00, α pinned 0.0500): HopperHop finals **0.26 / 0.002 / 0.006 — worse than the entropy-collapsed baseline (76/23/101); no seed ever approaches 200.** 3-way P1 reading now complete:
- **PPO (on-policy):** walls at 0 — conjunctive product starves the advantage estimate (H3, ledgered).
- **SAC, auto-α (collapse):** α→0.003, learns to STAND (~76-101, the standing local optimum), never hops.
- **SAC, α floored 0.05 (canonical entropy):** never even stands — the entropy term in the ACTOR OBJECTIVE keeps the policy wide, and the narrow contact-critical stability basin pays ~0 under persistent stochasticity.
- **(stripped-)TD-MPC2, same stack:** 8/8 ≥200, ~1-2M crossing.
**Mechanism formulation (Paper 3): the discriminating design axis is exploration-in-the-data vs stochasticity-in-the-objective.** TD-MPC2's core optimizes a DETERMINISTIC actor objective (max Q at the mean action) and injects exploration only at collection time (annealed Gaussian 0.3 + random warmup); SAC bakes entropy into the objective, which on a conjunctive-sparse contact task is lose-lose (collapse→stand-trap, floor→noise-trap). This makes TD-MPC2's Hop win attributable to the TD core's *objective structure* (+ Q-ensemble/SimNorm), NOT planning/WM/collection — completing Point-1 with a positive mechanism.
CAVEATS: n=3/arm; α-floor grid is 1 point (0.05) — an α=0.01 arm now running to map the workable-entropy band (narrow-band claim needs ≥2 floor values); custom SAC v1 (not reference impl) — statement scoped to 'canonical SAC objective under this stack'.

### ✅ P1 COMPLETE — α-band point 3: floor 0.01 ALSO fails (2026-07-10 14:15, P1_SAC3_HOP_DONE, n=3 @5M)
SAC α-floor 0.01 (canonical target-entropy) HopperHop finals: **0.98 / 0.002 / 51.4** — no seed near 200; two noise-trapped, one crawling toward the stand-trap. Full P1 α-grid (all n=3, 5M, same stack):
| SAC entropy config | finals | phenotype |
|---|---|---|
| auto-α (collapses →0.003) | 76 / 23 / 101 | stand-trap (best SAC config!) |
| fixed floor α=0.01 | 1 / 0 / 51 | mixed noise/stand-trap |
| fixed floor α=0.05 | 0.3 / 0 / 0 | noise-trap |
| (stripped-)TD-MPC2 core | 8/8 ≥200 by ~1-2M | solves |

**P1 verdict (scoped honestly): at a 5M budget, no entropy configuration of the SAC objective — auto-tuned, low-fixed, or canonical-fixed — gets past the standing local optimum on HopperHop's conjunctive reward, while TD-MPC2's planner-free TD core crosses 200 by 1-2M (n=8). The edge is objective-structural (deterministic actor objective maximizing a Q-ensemble over a SimNorm latent, exploration injected only in the data), not an entropy-knob artifact.** SCOPE CAVEATS: (i) sample-efficiency claim at 5M, NOT capability — external/historical SAC reaches Hop success by ~8M (6/9), so SAC eventually hops; the gap is ≥4-8× sample efficiency. (ii) custom SAC v1 implementation. (iii) remaining unexplored knobs (UTD, n-step, ensemble size) — these are TD-MPC2-side ingredients; testing which ONE carries the speed gap = the Lean+ decomposition experiment (next). Point-1 of the user's three questions is now answered with a positive mechanism + 12-run grid.

### ⚠️ OPEN VALIDITY ITEM (2026-07-10 15:00, user-raised): our TD-MPC2 is a JAX reimplementation NEVER parity-checked against Hansen's official PyTorch TD-MPC2
User question: "are we wrong at the very beginning? have we ever compared our reimplementation with original tdmpc?" Honest answer: NO rigorous head-to-head exists in the ledger. The "v24 parity" note in tdmpc2.py refers to INTERNAL version parity, not the official repo. Known deviations from canonical TD-MPC2: (1) data collection = π+Gaussian-noise (canonical: planner collects); (2) MPC-distill exists but default-off (canonical: none); (3) env backend = mujoco_playground MJX, not dm_control MuJoCo (same reward definitions, different physics numerics); (4) policy-entropy handling to be re-checked vs canonical. IMPACT SPLIT: internal controlled ablations (wmabl, VBN, VAC/URC/bisim nulls) remain valid AS statements about this architecture family; externally-referencing claims ("TD-MPC2 beats PPO/SAC", Part-12's critique of the WM narrative) must be scoped to "our TD-MPC2 variant" until verified. BIGGEST THREAT: "consistency loss removable on Hop" was measured with the planner OUT of the training loop; in canonical TD-MPC2 the planner (which rolls the WM) collects the data, so removability might fail there. VERIFICATION PLAN: (V1) curve-parity audit vs published TD-MPC2 results (no GPU); (V2) MPPI_COLLECT=1 env-gate — planner-collection mode in OUR stack, then {full, stripped}×planner-collect on Hop (n=2-3, ~3M) — if stripped survives planner-collection, the critique holds beyond the deviation; (V3, stretch) official PyTorch tdmpc2 run on dm_control Hopper-hop for ground truth. V2 is the decisive in-stack test and feeds Paper 3 regardless of outcome.

### ✅ V1 — PARITY AUDIT vs official TD-MPC2 published results (2026-07-10 15:15; github nicklashansen/tdmpc2 results/, n=3/task @4M dm_control-MuJoCo vs ours @5M MJX)
| task | official TD-MPC2 final | ours (vanilla) | gap | official cross-200 | ours cross-200 |
|---|---|---|---|---|---|
| hopper-hop | 449 (373/380/594) | ~420±113 (mppi ~571) | **≈0% PARITY** | 0.2-0.4M | ~1M |
| cheetah-run | 896 | 855 | −5% | 0.1M | — |
| walker-run | 877 | 727 | −17% | 0.1M | — |
| acrobot-swingup | 663 | 511 | −23% | 0.1-0.2M | — |

**Official SAC on hopper-hop @4M: finals 0 / 246 / 105 (cross-200: 1/3 seeds).** Our custom SAC v1 (76/23/101, 0/3) is INSIDE the canonical SAC band → **P1's 'SAC fails on Hop' is NOT an implementation artifact — reference SAC shows the same stand-trap phenotype.** (Official TDMPC-v1 hopper: 2/577/1 — Hop is seed-brutal for everyone.)
**Three V1 verdicts:** (1) The two pillars of Part-12/Paper-3 — TD-MPC2's Hop level AND SAC's Hop failure — are REPRODUCED at canonical levels by our variant; the critique is not built on a broken reimplementation. (2) Ours is systematically weaker exactly where the WM is load-bearing: official-minus-ours gap ordering (Hop ≈0% < Cheetah 5% < Walker 17% < Acrobot 23%) TRACKS our WM-load-bearing ordering (Hop removable < Walker −7.5% < Cheetah −19% < Acrobot −44%) — consistent with the missing planner-collection mattering most where accurate rollouts matter, i.e., canonical data CROSS-VALIDATES the task-conditional WM story. (Confound: MJX-vs-MuJoCo backend also differs; V2 tests planner-collection directly in-stack.) (3) Official crossings are uniformly faster (0.1-0.4M vs ~1M) — early-phase sample-efficiency claims must stay within-stack. SCOPING DECISION: papers/blogs say 'our TD-MPC2 variant (policy-collection; final levels at or near canonical on Hop/Cheetah, −17/−23% on Walker/Acrobot)'.

### ✅ H3 MARGIN-CONTROLLED VARIANT — F3 caveat CLOSED (2026-07-10 17:20, H3_MARGIN_DONE, n=2 @20M)
PPO on HopperHop, product reward, HOP_SPEED=1.0 (easier threshold) with HOP_MARGIN held at 1.0 (the DEFAULT shaping band — de-confounding the earlier variant where margin=speed/2 shrank with the threshold): finals **2.8 (s50) / 3.6 (s51) — still walled at 20M.** Combined H3 grid: product default → 0; product speed1.0 (margin coupled 0.5) → 1; product speed1.0 margin1.0 → 2.8/3.6; ADDITIVE → climbs to 135. **The wall is the CONJUNCTION itself — not the hop-speed threshold, not the shaping-band width, not termination (none exists).** Voelcker-style reward-design account now fully de-confounded (the F3 audit caveat is discharged); paper-ready. Env-gates HOP_SPEED/HOP_MARGIN/HOP_REWARD_MODE remain in hopper.py (.bak_h3m) as the reward-conjunctivity benchmark knob (Paper 3 positive artifact).

### P1 wording check — our π objective is FULLY deterministic (2026-07-10 17:55, code audit tdmpc2.py L719-741)
Our policy loss: pl = −E[min₂ Q(sg(z), tanh(μ_π(z))) / RunningScale] — the DETERMINISTIC tanh-mean action, no sampling, NO entropy term anywhere in the actor objective. Canonical TD-MPC2 samples a~π and adds a very small entropy bonus (~1e-4 coefficient) — functionally near-deterministic. P1's mechanism statement is therefore exact for our stack and directionally right for canonical: the working recipe on conjunctive-sparse Hop pairs a (near-)zero-entropy actor OBJECTIVE with data-side exploration (annealed Gaussian 0.3 + warmup), vs SAC's objective-entropy at α∈[0.003 collapsed, 0.05 floored] which fails at every tested level. Deviation note for the paper: ours drops canonical's small entropy bonus and action-sampling in the actor loss — another (minor) reimpl deviation to list in setup.

### ✅ P4 — Cheetah sufficiency n=4 FINAL (2026-07-11 01:50, P4_CHEETAH_SUFF2_DONE; s72/73 full-5M + s70/71 truncated@4.55M)
| arm | s70 (trunc) | s71 (trunc) | s72 @5M | s73 @5M | mean |
|---|---|---|---|---|---|
| full (none) | 733.3 | 670.2 | 766.0 | 770.5 | **735.0** |
| stripped (consistency-off) | 526.7 | 607.6 | 455.7 | 649.5 | **559.9** |

**CheetahRun WM load-bearing = −23.8% (n=4; clean full-5M pair alone: −28.1%).** Stripped-arm variance is high (456-650) while full is tight (670-771) — consistency-off Cheetah is seed-fragile, itself informative (without the WM anchor the return level wanders). Historical single-seed −38% not reproduced; durable claim: **Cheetah −20 to −28%, mean ≈−24%.** Task ordering CONFIRMED at n≥4 for three tasks: Hop removable (n=8) < Walker −7.5% (n=4) < Cheetah −23.8% (n=4) < Acrobot −44% (n=1-2, needs seeds) — the task-conditional world-model table for Papers A/3 is now paper-grade for the first three rows.

### ✅ A1 — Cheetah VBN n=3 (2026-07-11 02:40, A1_CHEETAH_S52_DONE; all finals at es=5,000,192 mppi)
| D | s50 | s51 | s52 | n=3 mean | % of van 855 |
|---|---|---|---|---|---|
| 16 | 496 | 535 | 519 | **516.7** | 60% |
| 32 | 563 | 582 | 584 | **576.3** | 67% |
| 64 | 639 | 617 | 617 | **624.3** | 73% |
| 128 | 753 | 723 | 703 | **726.3** | 85% |

Strictly monotone WITHIN every seed; per-width seed spread ≤ 50 (tight). **The Cheetah value-sufficiency curve is n=3 paper-grade: return rises smoothly with bottleneck width, no width recovers vanilla (85% at D=128), the value pathway reads a distributed latent.** This is Paper A's positive-instrument headline figure for Cheetah. Walker s52 still running (n=3 pending); Acrobot remains n=2.

### ✅✅ V2 VERDICT — Hop removability SURVIVES canonical-style planner-collection (2026-07-11 04:30, V2_MPPICOL_DONE, n=2/arm @2.5M, MPPI_COLLECT=1, n_samples=512)
| arm | s50 | s51 | mean |
|---|---|---|---|
| full + planner-collection | 467.8 | 462.2 | **465.0** |
| stripped (consistency-off) + planner-collection | 451.8 | 479.8 | **465.8** |

**Stripped ≈ full to within noise (Δ = +0.2%).** The consistency loss is removable on HopperHop even when the PLANNER COLLECTS THE DATA by rolling the (untrained, in the stripped arm) dynamics net — the biggest threat to Part-12's headline is DISCHARGED; the critique holds beyond the policy-collection deviation. Secondary reads: (i) planner-collection does not materially change Hop levels vs our historical policy-collection (~465 @2.5M vs ~420±113 @5M band — consistent with V1's Hop-deficit ≈0%); (ii) the stripped arm's planner scores actions through reward/value heads over garbage dynamics and STILL collects data good enough for full performance — the strongest evidence yet for H4 (execution-simple limit cycle; planner value-scoring, not rollout fidelity, is what matters on Hop). Remaining scope note: n=2/arm at 2.5M on Hop only; Walker planner-collection contrast (where WM IS load-bearing) is the natural follow-up if a reviewer asks. Part-12/Paper-3 can now say: 'removable under BOTH policy- and planner-collection (n=8 + n=2)'.

### ✅ A1 — Walker VBN n=3 (2026-07-11 06:20, A1_WALKER_S52_DONE; s52 finals at es=5,000,192 mppi)
| D | s50 @5M | s51 (trunc 4.0-4.5M) | s52 @5M | n=3 mean | % of van 727 |
|---|---|---|---|---|---|
| 16 | 591 | 617 | 656.5 | **621.5** | 85% |
| 32 | 642 | 662 | 637.1 | **647.0** | 89% |
| 64 | 665 | 675 | 665.6 | **668.5** | 92% |
| 128 | 695 | 720 | 688.1 | **701.0** | 96% |

(s51 truncated by the 07-10 fleet incident — biases s51 LOW yet it's mid-pack; mixed-budget noted.) n=3 means remain monotone though s52 alone has a 16/32 inversion (656>637) — Walker's tight widths are near-tied ~620-670, i.e., **Walker is the most compressible task: even D=16 retains 85% and D=128 recovers 96%.** Curve is n=3; the flat-floor + high-recovery shape is the Walker signature in the 3-task VSB law. Acrobot s52 launched on the freed GPUs (→ n=3 for the least-compressible task; prior n=2 280/351/291/491 vs 511).

### ✅✅ V2W — PRE-REGISTERED PREDICTION CONFIRMED: the collection-mode × WM double dissociation (2026-07-11 13:15, V2W_MPPICOL_DONE, n=2/arm @2.5M, MPPI_COLLECT=1, n_samples=512)
| task | full + planner-collect | stripped + planner-collect | Δ |
|---|---|---|---|
| HopperHop (V2) | 465.0 (468/462) | 465.8 (452/480) | **+0.2% (removable)** |
| WalkerRun (V2W) | **721.9** (758/686) | **605.4** (601/610) | **−16.1% (non-overlapping)** |

Pre-registered before launch: "stripped degrades under planner-collection on Walker, where the WM is load-bearing." CONFIRMED. **The double dissociation is now complete and clean: the consistency loss is removable on Hop under BOTH collection modes, and load-bearing on Walker under BOTH — with the planner-collection gap (−16.1% @2.5M) LARGER than the policy-collection gap (−7.5% @5M), i.e., planner-collection AMPLIFIES the world model's importance exactly where rollout quality matters (the planner collects by rolling the WM; garbage dynamics → degraded data).** Second read: full+planner-collect reaches 722 at 2.5M ≈ the policy-collect full band at 5M (708-727) — planner-collection roughly DOUBLES Walker sample-efficiency in our stack, consistent with V1's official-vs-ours deficit attribution (official planner-collecting TD-MPC2 is stronger precisely on WM-load-bearing tasks). This pair of results (V2+V2W) is the cleanest mechanistic evidence in the whole program: WM value is task-conditional, and collection mode modulates it in the predicted direction on both ends. Paper 3 centerpiece table; n=2/arm noted (extend if reviewer asks).

### 📋 LEAN+ VALUE-STABILIZATION — DESIGN PROPOSAL (2026-07-11 14:30, Handbook-Ch6 format; launch when V2X frees b3060)
**Premise check first:** the naive bet ("add XQL/Maclaurin robust value loss") is weakened by our own architecture — the value loss is ALREADY two-hot cross-entropy (bounded gradients, outlier-robust), which P1 suggests is part of why the TD core wins. Loss-level robustness is likely redundant. **What our data says is actually missing: TARGET stability when the WM anchor is removed.** Evidence: stripped-Cheetah seed spread 456-650 (full: 670-771 tight); stripped-Walker milder; stripped-Hop stable. The consistency loss appears to act as a variance anchor on the latent that the TD targets ride on — remove it and value learning wanders on dense tasks.
**Claim:** slowing the target-network EMA (and/or capping RunningScale drift) recovers most of the stripped model's LEVEL and VARIANCE on dense tasks — i.e., the WM's load-bearing contribution on Cheetah/Walker is substantially replaceable by cheap target smoothing (Lean+ = stripped TD-MPC2 + stabilized targets; no dynamics net at all).
**Arm-pair:** env-gate LEAN_TAU (target-EMA tau divided ~3-5×, exact current value to be read from code before patching) in wmabl tdmpc2.py; arms = Cheetah × ABLATE=consistency × {LEAN_TAU on, off} × s50/51/52 (6 runs, 5M, ~2 packed days) + later Walker replication.
**Pre-registered predictions:** (1) stripped+stab variance shrinks toward full's (spread <120); (2) mean recovers ≥half the -24% gap (→ ≥620); (3) effect ordering Cheetah > Walker > Hop≈0 (Hop already stable — any Hop change falsifies the mechanism story). **Kill number:** stripped+stab Cheetah mean <590 or spread >180 at n=3 → dead, ledger the null.
**Payoff:** if it works → Lean+ is a real artifact (TD-MPC2 minus the entire WM at near-full dense-task performance = the lightweight agent the user asked for); if it fails → the WM's contribution is NOT mere target stabilization, sharpening Paper A's claim that the consistency loss carries irreplaceable predictive structure on dense tasks. Informative both ways.

### ✅ A1 — Acrobot VBN n=3 (2026-07-11 18:10, A1_ACROBOT_S52_DONE + A1_ACROBOT_S52_W64_DONE; s52 finals at es=5,000,192 mppi)
| D | n=2 mean (s50/51) | s52 | n=3 mean | % of van 511 |
|---|---|---|---|---|
| 16 | 280 | 213.6 | **257.9** | 50% |
| 32 | 351 | 98.8 | **266.9** | 52% |
| 64 | 291 | 230.6 | **270.5** | 53% |
| 128 | 491 | 303.2 | **428.4** | 84% |

s52 is a globally weak seed (low at every width — Acrobot's known seed brutality; cf. official TDMPC-v1 hopper 2/577/1 style variance). At n=3 the three tight widths converge to a statistically flat ~258-271 (50-53%) while **D=128 = 428 (84%) remains far above** — the robust Acrobot claim is unchanged and now better-founded: *128 ≫ tight widths; the tight-width ordering is noise; Acrobot is the least-compressible task in the grid.* s53 running → n=4 tonight. **With this, the 3-task VSB grid is n=3 COMPLETE: Cheetah 517/576/624/726 vs 855 (monotone), Walker 622/647/669/701 vs 727 (flat-high, most compressible), Acrobot 258/267/271/428 vs 511 (step-at-128, least compressible). Paper A's positive-instrument figure is data-complete.**

### ✅ V2X — dissociation table at n=3 (2026-07-11 23:20, V2X_S52_DONE, planner-collection @2.5M)
| task | full s50/51/52 | stripped s50/51/52 | mean Δ | median Δ |
|---|---|---|---|---|
| HopperHop | 468 / 462 / 448 (mean 459) | 452 / 480 / 466 (mean 466) | **+1.4%** | +0.9% |
| WalkerRun | 758 / 686 / **455** (mean 633) | 601 / 610 / 600 (mean 604) | −4.6% n.s. | **−12.5%** |

**Hop: removability at n=3 is rock-solid** (stripped ≥ full at every seed; spreads tight). **Walker: the s52 FULL arm is an outlier low (455 vs 686-758)** — full+planner-collect appears bimodal (2 strong seeds, 1 collapsed) while stripped is astonishingly tight (600-610, spread 10). Honest read: the dissociation DIRECTION survives (stripped never reaches the full-arm median; 2/3 seeds show the large gap; median Δ −12.5%) but the mean-based −16.1% claim must soften to 'median −12 to −16%, full-arm seed-bimodal (n=3)'. Resolving seed (s53 pair) launching alongside Lean+. Mechanistic note: a collapsed full+planner-collect seed on Walker is itself interesting — planner-collection couples data quality to WM quality, adding a failure mode the stripped arm (which ignores its garbage WM for data via value-scoring... no wait, stripped ALSO planner-collects; its stability suggests the WM-rollout scoring adds VARIANCE when the WM is mid-quality) — flag for Paper 3 discussion, do not over-claim.

### ✅ A1 — Acrobot VBN n=4 COMPLETE (2026-07-12 04:15, A1_ACROBOT_S53_DONE; s53 finals at es=5,000,192 mppi, disk-verified)
| D | s50/51 sum | s52 | s53 | n=4 mean | % of van 511 |
|---|---|---|---|---|---|
| 16 | 560 | 213.6 | 270.8 | **261.1** | 51% |
| 32 | 702 | 98.8 | 283.3 | **271.0** | 53% |
| 64 | 582 | 230.6 | 315.7 | **282.1** | 55% |
| 128 | 982 | 303.2 | 304.0 | **397.3** | 78% |

s53 per-width: 270.8/283.3/315.7/304.0 — a middling seed where D=128 (304) ≈ D=64 (316), i.e. the second consecutive seed where 128 is not dominant. At n=4 the aggregate step-at-128 persists but narrows (428→397, 84%→78% of vanilla), carried by s50/51 (491 pair mean). Claim after n=4: **tight widths (16/32/64) remain statistically flat at 51–55%; D=128 remains the only width that recovers a large fraction of vanilla, but its margin is seed-heavy** — Acrobot stays the least-compressible task in the grid, with the honest caveat that the 128-step's size has wide seed variance (s50 491-pair vs s52/53 ~304). Paper A figure updated to n=4 on Acrobot.

### ✅ V2W s53 — Walker planner-collection dissociation RESOLVED at n=4 (2026-07-12 10:50, V2W_S53_DONE; finals at es=2,500,096, disk-verified)
| arm | s50 | s51 | s52 | s53 | n=4 mean | n=4 median |
|---|---|---|---|---|---|---|
| full (none) | 758 | 686 | 455 | **744.7** | 661 | **715.5** |
| stripped (consistency) | 601 | 610 | 600 | **558.3** | 592 | **600.5** |

**Verdict: the dissociation is CONFIRMED at n=4** — stripped degrades −15.4% on medians (−10.4% on means), vs Hop's rock-solid ±0 (n=3, +1.4%). The s52 full-arm 455 is now the lone outlier (3/4 full seeds cluster 686–758), so the "bimodality" concern dissolves — with one honest refinement observed live in s53's eval trajectory: the FULL arm's evals swing widely within a single run late in training (680 → 715 → 676 → **501** → 696 → 744 over 2.1–2.5M) while the stripped arm's stay tight (539–568 across the same window). The right claim is therefore: **under planner-collection on WalkerRun, the full model reaches a higher but higher-variance performance regime (median ~715, within-run eval swings ~250 pts), while the stripped model is stable at a lower level (~600, swings ~30 pts)** — the WM buys peak performance at the cost of eval stability; s52's 455 was most likely an unlucky final-eval draw from that volatile regime, not a distinct mode. Paper 3's double-dissociation table updates to n=4 on Walker with this variance-aware framing.

### ❌ Lean+ VERDICT — informative NULL (2026-07-12 12:20, LEANP_CHEETAH_DONE; finals at es=5,000,192, disk-verified, n=3/arm)
| arm | s50 | s51 | s52 | mean | spread |
|---|---|---|---|---|---|
| lean-on (LEAN_TAU=0.003, stripped Cheetah) | 584.0 | 605.5 | 601.3 | **596.9** | 21.5 |
| lean-off (control, tau=0.01, stripped Cheetah) | 633.5 | 602.7 | 585.2 | **607.1** | 48.2 |

Pre-registered gates (0af60bf): success = variance<120 AND mean≥620 → **FAILS the mean gate** (596.9 < 620); kill = mean<590 or spread>180 → not triggered. Verdict: **target-EMA smoothing does NOT replace the world model's anchoring role.** The one real effect is variance: lean-on halves the seed spread (21.5 vs 48.2) — the smoothing stabilizes, but around the same (slightly lower) mean, and both arms remain far below the full model's 855. Per the pre-registered payoff branch: the WM's contribution on dense tasks is NOT mere target stabilization — the consistency loss carries predictive structure that a slower target EMA cannot supply. Paper A's claim sharpens accordingly. Lean+ as a lightweight-agent artifact: dead on Cheetah-class tasks; the honest lightweight recipe remains "strip the WM only where the task ordering says it's removable (Hop-class)."

### 📋 PRE-REGISTRATION — V2H s53 + V2C (launched 12:30, GPUs 0-2 b3060, 2.5M planner-collection, n_samples=512)
Extending the collection-mode × WM dissociation: (i) **V2H Hop s53 pair** → n=4 on the removable side; prediction: stripped≈full (±5%), consistent with n=3's +1.4%. (ii) **V2C Cheetah s50/s51 pairs** → 3rd task in the dissociation table, the intermediate-compressibility case; prediction: stripped degrades ≥15% (policy-collection gives −23.8% at 5M; planner-collection amplified Walker's gap, so Cheetah's should be at least Walker-sized). Kill/read: if stripped-Cheetah under planner-collection degrades <8%, the "planner-collection amplifies WM importance" generalization is falsified and the amplification is Walker-specific.

### ✅ A1 — Cheetah VBN n=4 COMPLETE (2026-07-12 16:00, A1_CHEETAH_S53_DONE; s53 finals at es=5,000,192, disk-verified)
| D | n=3 mean | s53 | n=4 mean | % of van 855 |
|---|---|---|---|---|
| 16 | 517 | 624.9 | **544.0** | 64% |
| 32 | 576 | 624.9 | **588.2** | 69% |
| 64 | 624 | 646.7 | **629.7** | 74% |
| 128 | 726 | 729.0 | **726.8** | 85% |

(s53 W16 and W32 finals coincide at 624.9 — verified distinct runs, distinct md5s and differing pi-rows; a 1-decimal coincidence.) s53 is itself non-decreasing in width (624.9/624.9/646.7/729.0), and the n=4 means keep Cheetah's **strictly monotone** value-sufficiency curve, now spanning 64%→85% of vanilla. Cheetah remains the clean "smooth information-gradient" case in the 3-task grid: no width is enough, each doubling buys more, and even D=128 leaves 15% on the table — consistent with the sufficiency result (−23.8% stripped) that Cheetah's WM carries real predictive structure. Walker n=4 pending (s53 W16/32 GPU3 + W64/128 GPU0, ~evening) → grid n=4 complete then. Refill: Cheetah s54 W64/W128 launched on freed GPU2 (n=5).

### 📋 GWM line reviewed pre-box-retirement (2026-07-12 18:30) — stays CLOSED; one banked item
User retiring the 5070 Ti GWM box (instance 40924090). Review of the June record (CHANGELOG + graph-world-model-plan.md): iter-34 representation-level OOD win (graph value-R² 0.57 vs fair-mono 0.40) never converted to control (MPC at random floor); SOLD reproduced (100% vs paper 97.9%) and its own Table 1 shows monolithic TD-MPC2 ties it off-relational; iter-36 control test UNINFORMATIVE (env SNR ceiling, on the retiring box); ManiSkill headroom gate FAILED (monolithic generalizes over passive distractors). Verdict: line stays closed as Paper A's relational-axis closure. BANKED (post-deadline, not this week): SOLD Fetch-Distinct head-to-head, ~2-3 box-days on a ≥30GB-disk PyTorch box — the one untested cell (active multi-object interaction). Box retirement is data-safe: contact_entities.py/entity_wm.py/monolithic_wm.py/value_coupling_probe.py all in tdmpc-glass repo. Part-18 lab-notebook blog published covering all 10 tracks of the revision week.

### ✅ A1 — Walker VBN n=4 COMPLETE → 3-TASK GRID n=4 COMPLETE (2026-07-12 22:20, A1_WALKER_S53AB_DONE + A1_WALKER_S53CD_DONE; s53 finals at es=5,000,192, disk-verified)
| D | n=3 mean | s53 | n=4 mean | % of van 727 |
|---|---|---|---|---|
| 16 | 622 | 610.7 | **619.2** | 85% |
| 32 | 647 | 618.2 | **639.8** | 88% |
| 64 | 669 | 648.9 | **664.0** | 91% |
| 128 | 701 | 674.9 | **694.5** | 96% |

s53 is itself monotone (610.7→674.9) and lands squarely on the n=3 curve. Walker stays the **most compressible** task: even D=16 retains 85% of vanilla. **THE PAPER-A FIGURE IS n=4 COMPLETE ON ALL THREE TASKS:**
- **Cheetah** 544/588/630/727 vs 855 (64→85%, strictly monotone — smooth information gradient)
- **Walker** 619/640/664/694 vs 727 (85→96%, flat-high — most compressible)
- **Acrobot** 261/271/282/397 vs 511 (51→78%, step-at-128 — least compressible)

Three qualitatively distinct fingerprints at n=4, matching the sufficiency ordering (Walker −7.5% < Cheetah −23.8% < Acrobot −44%). n=5 arms (s54) in flight on all Cheetah widths + Walker W16/32; Walker s54 W64/128 launched on freed GPU0.

### 🔴 V2HC + V2W-s54 — the 3-task planner-collection table lands, WITH A PRE-REGISTERED INVERSION (2026-07-13 03:05, V2HC_EXT_DONE + V2W_S54_DONE; finals at es=2,500,096, disk-verified)
**Cheetah (NEW, n=2): the pre-registered prediction (943819c: stripped −15%+) is FALSIFIED — inverted.**
| arm | s50 last-6 evals (2.25→2.5M) | s50 final | s51 last-6 | s51 final | last-6 median |
|---|---|---|---|---|---|
| full | 313/312/171/300/440/**141** | 141.4 | 235/554/329/341/453/**585** | 584.8 | ~306 / ~397 |
| stripped | 547/560/538/530/529/**459** | 459.3 | 548/485/551/536/540/**555** | 555.1 | ~534 / ~544 |

Under planner-collection on CheetahRun, the FULL model does not merely fail to beat stripped — it **destabilizes** (evals swing 141–585 within 250k steps, both seeds) while the stripped model sits stably at ~460–555. On last-6 medians stripped is ~+40% ABOVE full. Kill criterion (<8% degradation) fires; the honest claim: **"planner-collection amplifies WM importance" is Walker-specific, not a law.** The full-model eval volatility seen on Walker (250-pt swings) becomes outright instability on Cheetah. Same-budget caveat: both arms at 2.5M; no cross-budget comparison to the 5M policy-collection numbers is made.

**Walker (n=5, V2W s54: full 738.8, stripped 612.5):** full 758/686/455/745/739 (median **739**, mean 676.6) vs stripped 601/610/600/558/612 (median **601**, mean 596.2) — **−18.7% on medians**, dissociation strengthens at n=5.

**Hop (n=4, s53: full 436.0, stripped 467.8):** stripped ≥ full again; removability rock-solid.

**The refined 3-task planner-collection picture:** Hop — WM removable, both arms stable. Walker — WM buys a higher but volatile regime (−19% median cost of stripping). Cheetah — WM under planner-collection is actively destabilizing; stripping HELPS (+40% median, n=2). The collection-mode × WM interaction is task-dependent and non-monotone in WM-load — a sharper (and more honest) Paper-3 centerpiece than the amplification story. EXTENSION LAUNCHED: V2C s52/s53/s54 pairs (6 arms, GPUs 1-3 b3060) → Cheetah n=5 to nail the inversion.

### ✅ A1 — Cheetah VBN n=5 (2026-07-13 03:10, A1_CHEETAH_S54AB_DONE + A1_CHEETAH_S54CD_DONE; s54 finals @5,000,192, disk-verified)
| D | n=4 mean | s54 | n=5 mean | % of van 855 |
|---|---|---|---|---|
| 16 | 544.0 | 564.7 | **548.1** | 64% |
| 32 | 588.2 | 593.8 | **589.3** | 69% |
| 64 | 629.7 | 613.7 | **626.5** | 73% |
| 128 | 726.8 | 721.3 | **725.7** | 85% |

s54 non-decreasing in width (564.7/593.8/613.7/721.3), lands on the curve; n=5 means barely move from n=4 (544/588/630/727 → 548/589/627/726). Cheetah's strictly-monotone value-sufficiency fingerprint is now rock-solid at n=5 (64→85%). Acrobot s54 VBN launched on freed b3060b GPUs 1-2 (→ n=5 on the last grid task).

### 🔴 3-TASK PLANNER-COLLECTION DISSOCIATION — COMPLETE at n=4-5 (2026-07-13 03:40, V2C_EXT2_DONE + V2H_S54_DONE + Walker VBN s54; finals @2.5M / 5M, disk-verified)

**Cheetah (n=5) — the INVERSION is confirmed, pre-registration decisively falsified:**
| seed | full final | full last-6 median | stripped final | stripped last-6 median |
|---|---|---|---|---|
| 50 | 141.4 | ~306 | 459.3 | ~534 |
| 51 | 584.8 | ~397 | 555.1 | ~544 |
| 52 | 232.9 | ~230 | 474.8 | ~526 |
| 53 | 523.4 | ~515 | 275.5* | ~515 |
| 54 | 116.8 | ~95 | 541.8 | ~526 |
| **agg** | **median 232.9, mean 320, range 117–585** | | **median 474.8, mean 461** | **tight ~515–544** |

(*s53 stripped final 275.5 is an end-of-run dip; its last-6 median is 515.) Under planner-collection on CheetahRun the FULL model is severely **destabilized** (finals span 117–585 across seeds; within-run eval swings 50→540) while STRIPPED sits stably at ~460–555. Stripped beats full by **+104% on final medians (474.8 vs 232.9)** and ~+70% on last-6 medians. Pre-registration (943819c: stripped −15%+, kill <8%) is **falsified/inverted at n=5**. The claim "planner-collection amplifies WM importance" is **Walker-specific, not a law**; the collection-mode × WM interaction is task-dependent and non-monotone.

**Hop (n=5): removability holds.** s54 full 472.9 / stripped 254.0 (a low stripped draw); with n=3 (465.0/465.8) + s53 (436.0/467.8), Hop shows **no systematic full>stripped** — arms overlap, no Walker-style dissociation. WM removable on Hop under planner-collection.

**Walker (n=5, restated): full median 739 vs stripped 601, −18.7%.** The one task where the WM is load-bearing (higher but volatile regime).

**Final Paper-3 table:** Hop = removable (stable) · Walker = load-bearing (−19%, volatile) · Cheetah = WM *actively destabilizing* under planner-collection (+100% median cost of KEEPING it). Three distinct regimes.

### ✅ A1 — Walker VBN n=5 (Walker VBN s54: W16=647.5 W32=654.0 W64=674.9 W128=690.9)
n=5 means: **625 / 643 / 666 / 694** vs vanilla 727 (86/88/92/95%). Flat-high fingerprint stable; grid VBN now n=5 on Cheetah+Walker, Acrobot s54 running → n=5.

### 📋 PRE-REGISTRATION — M1: the mechanism of the Cheetah planner-collection inversion (2026-07-14 03:40 SGT, to launch when V2CW_EXT3 frees b3060)
**Question:** WHY does the full world model destabilize under planner-collection on Cheetah (finals 117–585) but not on Hop? The inversion (stripped +104% over full at n=5) is observed but unexplained.
**Hypothesis (poisoned-target loop):** under planner-collection the planner rolls the world model to both ACT (collect data) and score MPPI candidates. If the full model's open-loop rollout is over-optimistic on Cheetah, MPPI selects on hallucinated value → the collected data + value targets are poisoned → eval swings. The STRIPPED model's planner rolls an untrained/frozen dynamics net, so MPPI degrades to ~stable random-shooting: lower but tight. Hop's dynamics are simpler → full-model rollout stays calibrated → no poisoning.
**Primary probe (low-risk, in-eval logger):** instrument the eval path to log, per eval, the **MPPI planned-return vs realized eval-return gap** (planner already runs at eval; just log its predicted value alongside the achieved return). Arms: {CheetahRun, HopperHop} × {full=none, stripped=consistency}, planner-collection, 2.5M, n≥2. 
**Pre-registered prediction:** full-Cheetah shows a LARGE positive planned−realized gap that SPIKES in the volatile eval windows; stripped-Cheetah ≈0 gap; both Hop arms ≈0 gap (small, stable). 
**GO (mechanism confirmed):** full-Cheetah mean |gap| ≥ 2× stripped-Cheetah AND gap correlates with eval variance (Pearson |r|≥0.4). **Kill/NULL:** gap comparable across arms → the instability is NOT planner-target poisoning (look elsewhere: value-target feedback, RunningScale). Either way it sharpens Paper 3.
**Fallback if the eval-logger build is flaky (nan-smoke fails):** relaunch the inversion pair to n=8 for a bulletproof CI + quantify the already-observed within-run eval volatility (std of last-10 evals, full vs stripped) as the descriptive mechanism. run_benchmark.py backup = .bak_v2.

### ✅ A1 — Acrobot VBN n=5 (2026-07-14 03:05, A1_ACROBOT_S54_DONE; s54 finals @5,000,192, disk-verified)
| D | n=4 mean | s54 | n=5 mean | % of van 511 |
|---|---|---|---|---|
| 16 | 261.1 | **1.6** | **209.2** | 41% |
| 32 | 271.0 | 218.5 | **260.5** | 51% |
| 64 | 282.1 | 186.9 | **263.1** | 51% |
| 128 | 397.3 | 161.6 | **350.2** | 68% |

**s54 is a brutal Acrobot seed** — W16 collapsed to **1.6** (total failure; Acrobot's documented seed fragility, cf. official TD-MPC-v1 hopper 2/577/1 variance) and W128 drew low (161.6, below W32/W64, non-monotone). This drags W16's n=5 mean to 209 and softens the D=128 step (78%→68%). Honest read: Acrobot remains the **least-compressible AND highest-variance** task in the grid — the step-at-128 is real on the median seed but seed-fragile at the tails; the tight-width ordering is noise. Acrobot s55 (running) + s56 (launching on freed GPUs 1-2) → n=6/7 to stabilize the D=128 step estimate. Grid VBN status: Cheetah n=5 (monotone, tight), Walker n=5 (flat-high, tight), Acrobot n=5 (step, noisy → extending).

### ✅ A1 — Acrobot VBN n=6 (2026-07-14 09:47, A1_ACROBOT_S55_DONE; s55 finals @5,000,192, disk-verified)
| D | n=5 mean | s55 | n=6 mean | % of van 511 |
|---|---|---|---|---|
| 16 | 209.2 | 207.5 | **208.9** | 41% |
| 32 | 260.5 | **12.5** | **219.2** | 43% |
| 64 | 263.1 | 327.6 | **273.9** | 54% |
| 128 | 350.2 | 189.3 | **323.4** | 63% |

**Second consecutive collapse seed** — s55 W32 = **12.5** (total failure), non-monotone (W64 327.6 > W128 189.3 > W16 207.5 > W32 12.5). s54 collapsed W16 (1.6); s55 collapses W32. Acrobot's per-seed brutality is now unmistakable: ~1 in 3 seed×width cells collapses to near-zero. The MEAN-based D=128 step keeps softening as collapses accumulate (n=3 84% → n=4 78% → n=5 68% → n=6 63%). **Paper recommendation: report Acrobot VBN on MEDIANS, not means** — the step-at-128 is robust on median seeds but the mean is dominated by a heavy collapse tail. Acrobot remains unambiguously the **least-compressible + highest-variance** grid task; the qualitative 3-fingerprint story (Cheetah monotone / Walker flat-high / Acrobot step) stands, with Acrobot's noise now well-characterized. s56 (running) + s57 (launching GPUs 0,3) → n=7/8 for a stable median.

### 🔴 V2CW n=7 — dissociation CONFIRMED at n=7 + the variance mechanism (M1 answered from eval data) (2026-07-14 10:20, V2CW_EXT3_DONE; finals @2.5M, disk-verified)

**Cheetah (n=7) — INVERSION holds:**
| seed | full final | full last-6 median | stripped final | stripped last-6 median |
|---|---|---|---|---|
| new s55 | 455.4 | 374 | 469.9 | 461 |
| new s56 | 327.2 | 266 | 565.2 | 538 |
| **n=7 finals** | full: 117/141/233/327/455/523/585 → **median 327, mean 340** | | stripped: 276/459/470/475/542/555/565 → **median 475, mean 477** | |
Stripped beats full **+45% on final medians** (474.8 vs 327.2) at n=7 (was +104% at n=5 — two new full seeds drew higher, but inversion is unbroken across all 7 seeds' medians). Pre-reg (943819c: stripped −15%) falsified.

**Walker (n=7) — dissociation holds:** full finals 455/686/739/745/749/758/776 → median **745**; stripped 558/600/601/608/610/612/645 → median **608**. Full > stripped **+22.5% on medians** (−18.4% cost of stripping). Rock-solid.

**M1 MECHANISM — answered directly from the eval trajectories (no separate probe needed):** the last-6-eval **standard deviation** is the mechanism.
| task | full-arm eval std | stripped-arm eval std | ratio |
|---|---|---|---|
| CheetahRun | **~165** (s55 ~170, s56 ~160) | ~60 (s55 ~75, s56 ~48) | **~2.7×** |
| WalkerRun | ~36 (s55 ~15, s56 ~57) | ~12 (s55 ~15, s56 ~8) | ~3× |

The world model **raises eval variance ~3× on both tasks** under planner-collection (the poisoned-planner-target signature: MPPI selecting on a model that periodically hallucinates value). The tasks differ only in the *absolute scale* relative to the mean: on Walker the WM lifts the mean to ~700–745 so the ~36-pt swings stay net-positive (higher-but-volatile); on Cheetah the ~165-pt swings dip so far (finals reach 117) that they drag the mean *below* the stable stripped model (~475) — variance tips into net harm. **One mechanism (WM inflates planner-target variance), two regimes, set by the mean/variance ratio.** This is the sharp, unified Paper-3 statement — and it needed no new experiment, only reading the volatility we already logged. (M1 pre-reg d7e7c49 is thus answered via the descriptive route; the planned-vs-realized-gap probe becomes optional confirmatory.)

### 🟡 JEPA #59 — Uniformity vs VICReg on CheetahRun (DMControl collapse task), n=2 (2026-07-14 15:55; runs from 07-07, URC/VAC_CHEETAH_DONE; finals @5M, disk-verified; HARVEST of pre-existing un-ledgered data)

**Context.** Extends task #58 (relational-anti-collapse lever: SE vs VICReg vs uniformity, on nav) to a DMControl collapse-prone task. Runner `run_vac.sh` (b3060b, tdmpc_glass): arms **urc**=uniformity regularizer (URC_LAM=1.0), **vac**=VICReg variance-covariance (VAC_LAM=1.0), **van/van2**=matched vanilla baselines (λ=0). TASK=CheetahRun, 5M steps, k_update=128, mppi_n_samples=2048, horizon=3. Metric = last-6 mppi-eval median (robust to the within-run volatility documented for Cheetah).

| arm | s50 | s51 | seed-median | vs vanilla |
|---|---|---|---|---|
| **uniformity (urc)** | 684.4 | 767.2 | **725.8** | −11.3% |
| **VICReg (vac)** | 778.7 | 723.9 | **751.3** | −8.2% |
| vanilla (van/van2) | 730.6 / 894.4 | 897.0 / 752.5 | **~818** | — |

**Reading (n=2, tentative).** On CheetahRun, **uniformity ≈ VICReg** (751 vs 726, well within seed noise) and **both sit slightly BELOW vanilla** (~−9%). Neither anti-collapse lever buys generalization on this DMControl collapse task — a **NULL** consistent with the broader redundancy story (H-JEPA/SE NULLs on Panda #56/#57; TD-MPC2's value-driven latent doesn't collapse enough for an anti-collapse prior to help). Seed variance is high (vanilla s50=730 vs s51=897; vac s50 final crashed to 418 but last-6 median smooths to 779), so n=2 is thin. **Refill in progress:** launching urc/vac/van s52/s53 on b3060b when Acrobot frees → n=4 to firm up whether the small negative is real or noise. This is the resumed JEPA thread the user asked for (07-14).

### 🟢 Acrobot VBN — CLEAN 5M harvest, n=5 complete seeds → fingerprint REFINES to "step-at-64" (2026-07-14 19:20, A1_ACROBOT_S56_DONE; final mppi, disk-verified)

**Data-quality correction.** Re-harvesting all Acrobot VBN seed dirs revealed the prior n=6 "step-at-128" table mixed step budgets: **s50/s51 only reached ~2.7M** (partial), s57 still running (~3.1M). The clean **final-@5M** sample is **n=5 = {s52,s53,s54,s55,s56}** (all step=5000192). s56 (new) is the strongest seed yet — no collapse, all widths high (W16=314, W32=455, W64=502, W128=500).

| width | per-seed finals {52,53,54,55,56} | median | (% vanilla 511) | mean |
|---|---|---|---|---|
| W16 | 1.6 / 207.5 / 213.6 / 270.8 / 314.1 | **213.6** | 41.8% | 201.5 |
| W32 | 12.5 / 98.8 / 218.5 / 283.3 / 455.0 | **218.5** | 42.8% | 213.6 |
| W64 | 186.9 / 230.6 / 315.7 / 327.6 / 501.9 | **315.7** | 61.8% | 312.5 |
| W128 | 161.6 / 189.3 / 303.2 / 304.0 / 499.6 | **303.2** | 59.3% | 291.5 |

**Refined fingerprint:** the step is at **W64, not W128**. W16≈W32 (~42% of vanilla, tight-width flat) → **W64≈W128 (~60%)**. The earlier "only D=128 recovers" claim was an artifact of the two partial low-step seeds; on complete-at-5M seeds **D=64 recovers as much as D=128** (62% vs 59%). The qualitative Paper-A story is *preserved and sharpened*: Acrobot is still the **least-compressible** grid task (tight widths flat far below vanilla; needs a mid-width to recover), and still the **noisiest** (2 collapse cells: s54 W16=1.6, s55 W32=12.5). But the recovery point is D≈64, and the three-fingerprint contrast becomes cleaner: **Cheetah monotone / Walker flat-high / Acrobot flat-low-then-step-at-64**. Report on MEDIANS (means dragged by the collapse tail). s57 completing (~5M later tonight) → n=6-at-5M for a final median; #59 refill now occupies the freed GPUs. Supersedes the n=6 mean table in edbb978 for the final-@5M paper figure.

### 🟢 Acrobot VBN — n=6 FINAL (all seeds {52-57} @5M) (2026-07-15 01:12, A1_ACROBOT_S57_DONE; final mppi, disk-verified)

s57 finals: W16=109.4, W32=335.1, W64=306.9, W128=339.8 (all step 5000192). Combined with {52-56} → **n=6 {52,53,54,55,56,57}**:

| width | median | (% van 511) | mean | per-seed sorted |
|---|---|---|---|---|
| W16 | **210.6** | 41.2% | 186.2 | 1.6, 109.4, 207.5, 213.6, 270.8, 314.1 |
| W32 | **250.9** | 49.1% | 233.9 | 12.5, 98.8, 218.5, 283.3, 335.1, 455.0 |
| W64 | **311.3** | 60.9% | 311.6 | 186.9, 230.6, 306.9, 315.7, 327.6, 501.9 |
| W128 | **303.6** | 59.4% | 299.6 | 161.6, 189.3, 303.2, 304.0, 339.8, 499.6 |

**Final fingerprint (n=6):** a **gradual climb W16(41%)→W32(49%)→W64(61%) that saturates at D=64** — W64 ≈ W128 (~60%), D=128 adds nothing over D=64. This is the mature form of the "step" (n=5 read it as a sharp step-at-64; n=6 with s57's higher W32=335 smooths the low end into a ramp). The Paper-A qualitative story is intact and now robust: **Cheetah strictly-monotone (no width suffices) / Walker flat-high (D=16 already 86%) / Acrobot ramp-to-D64 (least compressible, needs a mid-width, saturates well below vanilla at ~60%)**. Report medians (means dragged by 2 collapse cells: s54 W16=1.6, s55 W32=12.5). Acrobot VBN COMPLETE at n=6 — the grid is done. b3060b now runs JEPA #59 refill (urc/vac s52+s53). Supersedes 7501d4f n=5 for the final table.

### 🔴 V2CW n=9 FINAL — dissociation + Cheetah inversion CONFIRMED (2026-07-15 02:12, V2CW_S5758_DONE; finals @2.5M, disk-verified)

Added s57/s58 to the n=7 → **n=9 both tasks**. Metric = final mppi (v2mppicol CSV schema task,seed,step,reward; new-seed last-6 medians confirm direction, shown below).

**Cheetah (n=9) — INVERSION holds, +45.3%:** FULL finals [117,141,233,235,327,350,455,523,585] median **327**; STRIP [217,276,459,470,475,539,542,555,565] median **475**. Stripped beats full **+45%** (was +45% at n=7, +104% at n=5 — stable now). New seeds' last-6 medians: full s57=171/s58=162 (low+volatile), stripped s57=541/s58=296 — same direction.
**Walker (n=9) — dissociation holds, −18.0%:** FULL finals median **739**; STRIP median **605.7**. Full > stripped +22% (−18% cost of stripping). New full s57 final dipped to 291 (last-6 median 715 — a terminal within-run dip, the variance-mechanism signature; robust on median). New last-6 medians: full s57=715/s58=701, stripped s57=584/s58=607 — same direction.

**Paper-3 dissociation is FINAL at n=9:** planner-collection makes the world model *load-bearing on Walker* (−18% to strip) but *actively harmful on Cheetah* (stripped +45%), one variance-inflation mechanism (WM ~3× eval-variance on both), two regimes set by the mean/variance ratio. The pre-registered "stripped degrades ≥15%" (943819c) stays falsified on Cheetah. This is the compute-complete Paper-3 core; remaining is write-up. b3060 now fully idle (Hopper-only box; acceptable pre-stop). #59 refill continues on b3060b.

### 🔵 JEPA boundary sweep LAUNCHED — uniformity vs VICReg vs vanilla on WalkerRun (2026-07-15 04:05; b3060 + new 4×4070; running, harvest next session)

New front (user rented a 4×4070; kept idle b3060 on-task too). Tests the redundancy-law BOUNDARY: #59 found the anti-collapse null on **CheetahRun** (the strictly-monotone / least-compressible VBN fingerprint). WalkerRun is the opposite extreme — **flat-high**, the most-compressible task where the value function needs the *least* of the latent, i.e. the regime where an added anti-collapse objective has the MOST room to help. If uniformity/VICReg is *also* null here, the redundancy law is airtight; if structure ever helps, this is where it should.

**Sweep (WalkerRun, 5M, run_vac.sh, last-6 mppi median):** uniformity(urc), VICReg(vac), vanilla(van), seeds 60/61/62 → **urc n=3, vac n=3, van n=2**. b3060 (4 GPUs): urc s60/vac s60/van s60/urc s61. 4070 (4 GPUs): vac s61/van s61/urc s62/vac s62. All 8 START-verified; 5M ≈ 6-10h → completes this afternoon, past the 08:00 session stop but survives via nohup. Harvest + verdict next session. Prediction (pre-registered): null again (uniformity ≈ VICReg ≈ vanilla), extending the redundancy law across both VBN-fingerprint extremes.

### 🟡 JEPA #59 Cheetah → n=3 (s52 refill complete) — null CONFIRMED (2026-07-15 07:33; last-6 mppi median, disk-verified)

s52 finished at 5M: urc last6med=739.0, vac last6med=821.3. Combined with s50/s51 → **n=3 medians**:
- uniformity (urc): {684.4, 739.0, 767.2} → **739.0** (−9.7% vs vanilla ~818)
- VICReg (vac): {723.9, 778.7, 821.3} → **778.7** (−4.8% vs ~818)

Both still **below vanilla**, uniformity ≈ VICReg (within noise). The n=2 null holds at n=3; vac s52 came in high (~821 ≈ vanilla) which lifts vac's spread but the median stays sub-vanilla. s53 still running (partial). Interim WalkerRun boundary read @~2M/5M: all arms clustered ~695–724 (urc~695/vac~724/van~707), no separation yet — early-consistent with null; final ~10:00 UTC (harvest next session).

---

## ★ DISCUSSION-PREP SUMMARY — Wednesday 16:00 SGT (2026-07-15 08:05 UTC campaign stop)

**#59 final at stop:** urc s52=739.0 (5M), vac s52=821.3 (5M) → Cheetah n=3 medians **urc 739 (−10%) / vac 779 (−5%) / vanilla ~818 — NULL holds**. s53 partials (urc 733.5 @3.85M, vac 812.1 @3.55M) confirm same picture; s53 continues via nohup.

### What we now answer (this week's deliverables — all pushed, ledger thru 3c51069)
- **VBN instrument (3 fingerprints):** Cheetah *monotone* / Walker *flat-high* (D16=86%) / Acrobot *ramp-to-D64* (n=6). Valid replacement for decode-R². **Q6** re-framed.
- **Collection-mode dissociation n=9 (FINAL):** Cheetah **inversion +45%** (stripped>full), Walker **−18%**, Hopper ≈0. Pre-registration (stripped ≥15% degrade) falsified on Cheetah.
- **H-VARIANCE mechanism:** WM inflates planner-target eval variance **~3× on both**; mean/variance ratio sets the sign (Walker net-positive, Cheetah net-harmful). Answers the "why" behind (a) WM-value-is-task-dependent.
- **Q5:** TD-MPC2 clears Hopper via off-policy deterministic-actor planning (SAC 0/9 vs planner-free TD 8/8); PPO wall = conjunctive reward, independent of WM. **Q8** resolved (strip removes WM accuracy, not the planner; Hopper = clean case where neither abstraction is needed).
- **JEPA #59 Cheetah null (n=3).**

### Honest open items / corrections (raise at discussion)
- **H-COMPRESS positive half is UNPROVEN** ("imposed structure helps in flat-high"): #59 Cheetah null + WalkerRun interim null → at risk of falsification. WalkerRun boundary sweep (urc/vac/van n=3/3/2) **running on b3060+4070, finishes ~10:00-14:00 UTC** (nohup, past this stop) — harvest next session; it decides H-COMPRESS.
- **Reader-caught conflation (corrected in Part-19):** the strip-cost co-ranking is evidence for **H-WM-ABSTRACT** (LEARNED world model load-bearing in LOW-compress), NOT for imposed-structure-helps-in-flat-high. Two separate claims; only the learned-abstraction one is proven.
- **Official-parity / dual-implementation test NOT done** — biggest risk before Papers A/3 cite the reimpl (issue #2).

### Recommendation
Freeze compute on Papers A & 3 (data-complete at final n), **write** toward ~07-28. New front = **H-WM-ABSTRACT cross-ablation** (issue #5) + close H-COMPRESS boundary (Walker + issues #3/#4). Issues **#2–#6** filed to SuuTTT/tdmpc-glass. Blog **Part-19** live (hypotheses H-COMPRESS/H-WM-ABSTRACT/H-COLLECT/H-VARIANCE, plots+tables, corrections through 060ac39).

**Fleet at stop:** b3060 + 4×4070 running WalkerRun boundary sweep (nohup, continue into next week); b3060b #59 s53 finishing; monitoring loop STOPS here per the Wednesday-discussion directive.

### 🔵 JEPA/anti-collapse WalkerRun boundary — H-COMPRESS positive half FALSIFIED (2026-07-15 10:52; 4070 arms done @5M, b3060 arms ~3.3M partial)
WalkerRun is the flat-high (most-compressible) task where anti-collapse had the MOST room to help. Last-6 mppi medians: **uniformity(urc) ~705, VICReg(vac) ~717-744, vanilla(van) 674-738** — all clustered, **vanilla at/above both structure arms**. So anti-collapse regularizers are **null even here** → imposed structure is redundant across BOTH VBN-fingerprint extremes (Cheetah monotone null #59 + Walker flat-high null). Revised law: **imposed structure helps NOWHERE tested; only the LEARNED world model carries weight, in low-compressibility tasks.** (b3060 arms finishing → final n=3; direction already clear.) This is the clean negative that lets the paper pivot to the positive "when does the *learned* world model hurt" contribution.

### 🟡 JEPA/anti-collapse #59 Cheetah FINAL n=4 (s50-53) — null confirmed (2026-07-15 11:12)
s53 done: urc last6med=749.1, vac=811.7. **n=4 medians: uniformity(urc)=744.1 (−9%), VICReg(vac)=795.2 (−3%), vanilla ~818** — both ≤ vanilla, urc≈vac. Anti-collapse regularizers redundant on Cheetah at n=4, consistent with WalkerRun flat-high null (3f32229). Imposed-structure-redundant-everywhere stands.

### 🟢 VBN HopperHop s60 — the MISSING row is filled (#3) (2026-07-15 14:53; n=1 provisional)
Fills the HopperHop cell of the value-sufficiency-bottleneck grid (Paper A instrument). Last-6 mppi medians (col `reward`, vanilla HopperHop ~455): **D16=429.7 (95%), D32=416.65 (92%), D64=338.8 (75%), D128=531 (@4.75M, 117%)**. Shape at n=1 is **noisy/non-monotone** (HopperHop is the volatile task) — D16 already ≈vanilla (95%) points tentatively toward the **flat-high / most-compressible** family (like WalkerRun), which is the expected fingerprint for the *removable-WM* task (value head needs little of the latent → structure has room but the WM is redundant). NOT interpreting the D64 dip / D128 bump at n=1. Refill running: s61 (b3060b, D16-128) + s64 (4070, D16/32/64) → clean fingerprint at n=3. (Harvest-column bug fixed: reward is CSV col 2, not col 4=seed — earlier "60"s were the seed value.)

### 🟢 VBN HopperHop grid COMPLETE n=3 (2026-07-16)
Medians (vanilla~455): D16=430(95%) D32=417(92%) D64=339(75%) D128=531(117%). Large per-seed spread (D128 241-563) — noisiest VBN task; net flat-high/compressible (D16~95%), no clean monotone → removable-WM family (like WalkerRun). Seeds s60/s61/s64/s65.

### 🟢 Dreamer generalization FINAL (2026-07-16) — task-dependent WM benefit, cross-model
cheetah van732/strip667 (+9.7%, WM helps); walker van736.6/strip743.4 (tie, WM null). Same per-task ordering as TD-MPC2 (cheetah WM-load-bearing / walker WM-redundant) → the help-vs-hurt axis is a task property (value-info structure), not WM-architecture-specific. Holds across TD-MPC2 (latent-consistency) + Dreamer (reconstruction-RSSM). hopper van OOM infra-fail (strip=138.87 only). #8.

### 🟢 Dreamer multi-seed n=2 (2026-07-17)
cheetah van{732,712}/strip{667,172} — WM HELPS + stripped UNSTABLE (seed2 collapse 172). walker van{737,449}/strip{743,665} — WM null/hurts. Per-task WM-value, cross-model (TD-MPC2+Dreamer): cheetah WM-load-bearing+stabilizing, walker WM-redundant. cheetah n=3 (s3 pair) running; gated-WM g-sweep g{0,.25,.5,1} training. #8.

### 🟡 Gated-WM g-sweep preliminary (2026-07-17, n=1/g)
cheetah planner-collection @2.5M last-6 median: g0.0=692 g0.25=715 g0.5=633(@1.7M) g1.0=677. Plan-time WM-rollout-trust gate g<1 MODESTLY beats full WM (g0.25 +5.6
### Gated-WM g-sweep preliminary (2026-07-17, n=1/g)
cheetah planner-collection @2.5M last-6 median: g0.0=692 g0.25=715 g0.5=633(@1.7M) g1.0=677. Plan-time WM-rollout-trust gate g<1 modestly beats full WM (g0.25 +5.6pct, g0.0 +2.2pct). Positive but small+noisy at n=1; n=2 (s113/114/115) launching. Isolates scoring-trust channel vs strip-WM +45pct (co-varies training). #8.

## Dreamer cheetah n=3 (2026-07-17)
Third seed s3d @~1.09M: vanilla last30-median 710.2 vs stripped 637.3 -> vanilla>stripped +11.4% (WM helps). Consistent w/ n=1,n=2 (+9.7%). Cheetah WM-load-bearing holds at n=3 in Dreamer. Walker Dreamer n=3 launched (walker_van_n3/walker_strip_n3) to firm the null side.

## Gated-WM n=2 REVERSAL (2026-07-17)
seed2 @2.5M cheetah planner-collection: g1.0(s113)=695.6 g0.25(s114)=637.0 g0.0(s115)=654.6; +g0.5(s100)=637.3.
n=2 means: g1.0=686.3 g0.25=676.0 g0.0=673.3 g0.5=635.2. n=1 g0.25>+5.6% was SEED ARTIFACT (g0.25 seed2=637 vs seed1=715). At n=2 full WM g1.0 on top; gate does NOT robustly help. NEGATIVE result for gated-WM prescription. n=3 firming (s116/117/118/119).

## Walker Dreamer n=3 (2026-07-17)
seed3 @~1.09M: van_n3=722.3 strip_n3=775.6 -> stripped>=vanilla +7.4%. Walker WM-redundant confirmed n=3 (n=2 van736.6/strip743.4 tied). Contrast cheetah WM-helps +11.4%. Cross-model ordering holds n=3 in Dreamer. gated g0.25 n=3 seed s116=694.65 (mean 682.2, noise cluster, reversal holds). Acrobot Dreamer 3rd-task launched.

## Gated-WM n=3 FINAL (2026-07-17)
cheetah planner-collection @2.5M last6-median, 3 seeds/g: g0.0=[692,654.6,752.7]mean699.8; g1.0=[677,695.6,693.5]mean688.7; g0.25=[715,637,694.65]mean682.2; g0.5=[633,637.3,678.15]mean649.5. Within-g spread ~100pts >> between-g ~50pts = NOISE. No gate robustly beats full WM. g0.0 nominally top (weak echo of WM-hurts-under-collection) but n.s. VERDICT: tunable gate NOT reliable; contribution=diagnostic. n=4 seeds s121/122/123 launched.

## Acrobot Dreamer 3rd-task (2026-07-17) — STRONGEST WM effect
acrobot_swingup Dreamer n=1 @~1.09M: van=408.1 strip=26.2 -> vanilla>>stripped +1458% (15.6x; strip near-failing). WM ESSENTIAL. Completes monotone gradient matching VBN: acrobot(ramp,+1458%) > cheetah(monotone,+11.4%) > walker(flat-high,null). VBN PREDICTS WM-dependence ordering in Dreamer. 2nd acrobot seed launched (acrobot_van_n2/strip_n2). This is the paper's headline: cheap probe predicts when WM matters, cross-model, 3 tasks.

## Gated-WM n=4/5 (2026-07-18) — reversal rock-solid
4th seeds: s121 g0.0=708.2, s122 g1.0=665.35, s123 g0.25=685.1. Means: g0.0=701.9(n4) g1.0=686.5(n5) g0.25=682.9(n4) g0.5=649.5(n3,s124/125 running). Top-3 gap ~19pts vs within-g spread ~80-100pts = NOISE. No gate beats full WM at n=4-5. Tunable gate = confirmed negative. Contribution = diagnostic (VBN predicts WM-dependence 3 Dreamer tasks). Also acrobot n=2 @~93% van356/strip41 (+770%, holds). Acrobot n=3 launched on b3060.

## Acrobot n=2 confirm (2026-07-18)
seed2 @1.09M: van=420.0 strip=56.4 -> +644% (7.4x). Confirms n=1 (408.1/26.2,+1458%). Agg n=2: van~414 strip~41. WM ESSENTIAL on acrobot, multi-seed. 3-task VBN-predicts gradient now multi-seed on high-dep anchor. Acrobot n=3 running b3060. 4070 refilled VBN s80/81/82.

## Acrobot n=3 FINAL (2026-07-18)
seed3 settled @1.09M: van=409.4 strip=99.7 (+311%). Agg n=3: van=[408.1,420.0,409.4]mean412.5(tight); strip=[26.2,56.4,99.7]mean60.8(range26-100,HIGH var). Vanilla dominates every seed 4-16x (+311..1458%). WM ESSENTIAL on acrobot unshakable across 3 seeds; collapse magnitude varies. Anchors high-dep end of VBN gradient (acrobot>>cheetah+11.4%>>walker null). gated g0.5 n5=688.4 -> all 4 gates 19pt band, gate-negative definitive. Gated s128(g1.0)/s129(g0.25) launched keep-busy.

## Phase-1 breadth started (2026-07-19)
Launched Dreamer van+strip: pendulum_swingup (predict essential, 4070 G1/G2) + finger_spin (predict redundant, b3060 G0/G1), n=1, ~1.1M target. Goal: quantify VBN<->WM-dependence correlation across >=10 tasks (turn 3 anecdotes into a predictor). VBN fingerprints for new tasks pending run_vbn wiring. Make-work continues on other 4 GPUs.

## Breadth pt 4: pendulum-swingup (2026-07-19)
Dreamer n=1 @1.09M: van=806.0 strip=0.0 — strip NEVER scored (sparse underactuated). Most extreme WM-essential point; predicted a priori. Gradient: pendulum(total collapse)>=acrobot(+311-1458%)>>cheetah(+11.4%)>>walker(null). cartpole pair launched 4070 G1/G2; VBN s109 G0.

## 2026-07-20 ~04:15 UTC — parity-fix acrobot FINAL (s201, planner-collection)
- AcrobotSwingup vanilla MPPI_COLLECT=1 s201 @4M: last-6 median **395.0**.
- vs official 663 / our pi-collect 511: planner-collection made vanilla WORSE (-40% vs -23%).
- Verdict: collection mode NOT the cause of acrobot gap; planner-collect actively hurts on WM-essential task (matches inversion mechanism). Posted to #2.
- Walker s202 interim @3.7M: med 809.7 (-7.7% vs official 877, was -17%) — closing; final next window.

## 2026-07-20 ~05:30 UTC — breadth pts 5+6 FINAL + walker parity FINAL
- finger-spin (pt5): van 662.0 / strip 695.0 → -4.7% (strip>=van) → REDUNDANT (as pre-registered). n=1, 1.1M.
- cartpole-swingup (pt6): van 867.5 / strip 845.7 → +2.6% → marginal. n=1, 1.1M.
- 6-task gradient complete: pendulum collapse > acrobot essential > cheetah +11.4% > cartpole +2.6% > finger -4.7% > walker -7.0%.
- Walker parity s202 FINAL @4M: 817.1 vs official 877 = -6.8% (was -17% pi-collect) → collection mode explains walker gap. Posted #2.
- Relaunching: cup_catch pair (b3060 G0/G1), reacher_hard pair (4070 G1/G2), gated mw s144 (b3060 G3).

## 2026-07-20 ~06:26 UTC — VBN fingerprints for breadth tasks STARTED
- Confirmed MJX registry has PendulumSwingup/CartpoleSwingup/FingerSpin/BallInCup.
- Swapped 4070 make-work VBN seeds (s110 acro-D64, s111 cheetah-D32, killed clean) for critical-path: VBN PendulumSwingup D16 s112 (G0), FingerSpin D16 s113 (G3) — the D16 point discriminates essential (low) vs redundant (flat-high).
- reacher_strip relaunched after silent launch failure (cd-subshell trap: 'cd X && cmd &' backgrounds the cd too; second command in same ssh ran from /root). Lesson: one launch per ssh call.

## 2026-07-20 ~12:35 UTC — breadth pt7 reacher-hard FINAL + VBN pendulum/finger DONE
- reacher-hard (pt7): van 965.0 / strip 9.0 (@1008 med30, collapse) → WM ESSENTIAL. 2nd collapse anchor after pendulum.
- Structure: essential end = sparse/long-horizon (pendulum,acrobot,reacher); redundant end = dense control (walker,finger).
- VBN probes DONE rc=0: PendulumSwingup D16 s112 (11:38), FingerSpin D16 s113 (12:32) — harvesting probe metric.
- cup-catch pt8 (van971/strip0.0) finalizing.

## 2026-07-20 ~13:15 UTC — VBN fingerprint harvest (breadth extremes)
- PendulumSwingup D16 s112: last-6 med MPPI = 332.65 (~43% of ~766 vanilla) → LOW/volatile → least compressible → predicts WM ESSENTIAL. MATCHES strip→0.0 collapse.
- FingerSpin D16 s113: last-6 med MPPI = 961.15 (~98% of ~980) → HIGH → most compressible → predicts WM REDUNDANT. MATCHES strip-wins.
- VBN a-priori prediction CONFIRMED on both new breadth tasks. Metric = MPPI eval return of TD-MPC2 trained with value-path bottleneck width D (exp/vac/logs/vbnNN_<Task>_s<seed>.log, grep MPPI= last-6 median).
- In flight for width-curve: CartpoleSwingup D16 s114 (G0), PendulumSwingup D64 s115 (G3).
- 4070 relaunch: quad_van_n1 (G1, Dreamer); reacher_strip still on G2.

## 2026-07-20 ~14:50 UTC — breadth pt8 cup-catch FINAL → 8-TASK GRADIENT COMPLETE
- ball-in-cup catch (pt8): van 972.0 / strip 0.0 (@1056 flat-zero) → COLLAPSE → WM ESSENTIAL. 3rd collapse anchor.
- reacher-hard strip final @1088 = 18.0 (was 9.0 @1008); verdict COLLAPSE unchanged.
- 8-task gradient: pendulum(806/0) cup(972/0) reacher(965/18) acrobot(412/61) | cheetah(+11.4%) cartpole(+2.6%) | walker(-7%) finger(-4.7%).
- STRUCTURE: WM-essential = sparse/long-horizon/precision (tasks 1-4); WM-redundant = dense continuous control (7-8). Mechanistic reading matches VBN.
- Posted 8-task table to #8. All 8 GPUs busy (quad pair + 2 VBN + b3060 cup/gated).

## 2026-07-20 ~20:10 UTC — breadth pt9 quadruped-walk FINAL + reacher n=2 confirm → 9-task gradient
- quadruped-walk (pt9): van 958.1 / strip 963.7 (@944 strip, med stable) → +1% strip-wins → REDUNDANT. High-DoF dense locomotion, joins walker/finger.
- reacher-hard n=2: seed-2 van 971.0/strip 6.0 (seed-1 965/18) → collapse CONFIRMED multi-seed.
- 9-task gradient: essential[pendulum,cup,reacher,acrobot] | helps/marginal[cheetah,cartpole] | redundant[finger,walker,quadruped].
- Paper Fig3+Table1 updated to 9 tasks; correlation ρ=−0.94 n=6 (cartpole D16 74% added). Pushed. compiles 5pp.

## 2026-07-21 ~00:05 UTC — CUP outlier resolved + integrated honestly + reacher n=2 final
- BallInCup VBN D16 s122=975.5 (DONE), D128 s123 ceiling=974.5 → recovery ~100% → maximally compressible in TD-MPC2 BUT Dreamer-essential (collapse). Cross-model OUTLIER.
- Correlation: core n=6 ρ=−0.94 / with-cup n=7 ρ=−0.21. BOTH reported. Fig4 shows cup as labeled open outlier; fit on core 6.
- §3 'honest boundary' para added: VBN predicts value-limited WM-dependence; cup collapse is exploration-limited (not value-insufficiency) → diagnostic scoped to representational WM role.
- Grid-confirms n=2: acrobot 253.65(grid258✓), walker 645.25(grid622✓). reacher n=2 FINAL van966/strip6 (collapse robust).
- Paper pushed 5pp clean. Pruning reacher_*_n2 dirs.

## 2026-07-21 ~09:40 UTC — pendulum n=2 FINAL + n=3 in progress (bimodal confirmed)
- pendulum_van_n2 FINAL @1088 med30=764 (vanilla itself volatile within-run: bounced 805/887/838/764).
- pendulum_strip_n2 FINAL @1072 med30=727 → seed-2 strip ≈ van (−5%, REDUNDANT). Seed-1 was strip=0 (collapse). Seed-3 strip=0 @640 (collapsing).
- PENDULUM STRIP BIMODAL n=3: {0, 727, 0} vs van {806,764,~800} → 2/3 total collapse, 1/3 fully matched. Exploration-lottery signature.
- CROSS-VALIDATION (official TD-MPC2 results/ CSVs, no compute): TD-MPC2−SAC final-return gap per task — acrobot +591, hopper-hop +332, pendulum +263, walker +191, cheetah +125, cartpole +52, reacher +38, finger +10, cup +5. KEY: hopper-hop gap +332 despite consistency REMOVABLE + planning NULL → the TD-MPC2 win = VALUE PATHWAY, not the world model. And cup(+5)/reacher(+38) SAC-solved → their Dreamer strip-collapse is exploration/reconstruction-specific, NOT value-limited → independent confirmation of the value-limited vs exploration-limited split. Files: scratchpad/off/{tdmpc2,sac}__*.csv.
- PENDING (user judgment): pendulum reword (bimodal/high-variance, exploration-influenced but intermediate per SAC gap +263, NOT a clean cup-twin). Do at n3 @1088.

## 2026-07-21 ~10:12 UTC — (a) cross-validation added to paper + (b) rung-ladder launched
- (a) §3 'independent cross-check' paragraph: official TD-MPC2-SAC gap (results/ CSVs) confirms exploration-limited boundary — cup(+5)/reacher(+38) SAC-solved (not value-limited), acrobot(+591)/hopper(+332) value/planning-hard. Honest note: hopper gap=value-pathway (consistency removable + plan null). Both tex compile 6pp clean, pushed.
- (b) rung-ladder LAUNCHED HopperHop seed 60 5M (b3060): SAC(G0 ladder_sac_s60) / TD-MPC2 full(G2 ABLATE=none) / TD-MPC2 strip-consistency(G3 ABLATE=consistency). pi+mppi cols give value-pathway(strip+pi) & strip+plan rungs. EXISTING-DATA ladder (to firm): SAC~188 → strip+pi(value pathway)388 → strip+mppi 408 → full 420. Value pathway captures ~2/3 of TD-MPC2-SAC gap; consistency+planning add ~15%. Harvest: awk pi/mppi cols from exp csv; SAC from ladder_sac out.
- (b) ladder SAC HopperHop seed 60 FINAL = 85.6 @5M (a FAILED-crossing seed — SAC coin-flip on Hop). SAC 61/62 launched (b3060 G0/G1, fast). TD full/strip s60 running G2/G3. Matched-seed contrast forming: SAC s60 fails (86), TD-MPC2 value-pathway expected to cross (~400).
- (b) SAC HopperHop matched seeds 60/61/62 FINAL = 85.6/50.3/4.68 (all 3 FAILED to cross wall; SAC bimodal — n=12 avg ~188 w/ 50% crossing; this batch unlucky). TD-MPC2 full+strip s60+s61 running (4×b3060). Honest ladder framing: SAC UNRELIABLE on Hop (crosses ~half the time), TD-MPC2 value-pathway crosses reliably → the value pathway's win is RELIABILITY + level. Use SAC n=12 dist (mean 188, bimodal) as anchor, not the 3 unlucky matched seeds, for the figure.

## 2026-07-21 ~12:40 UTC — PENDULUM EDIT DONE (paper fully corrected)
- pendulum n=3 strip 0/727/0 (bimodal), van ~803. Dropped from clean-collapse + correlation core. Core ρ=−0.90 n=5 (was −0.94 n=6). Reclassified exploration-influenced w/ cup.
- Both tex builds 6pp clean, pushed. Table1/Fig3/Fig4/abstract/§3 all updated. Paper now FULLY honest + submit-ready.

## 2026-07-21 ~14:30 UTC — (b) RUNG-LADDER interim (1.75-3M, matched seeds 60/61) — DECISIVE
- LADDER HopperHop: SAC(these seeds ~47; n=12 avg 188) | strip+pi=VALUE-PATHWAY s60=478/s61=469 | strip+mppi(+plan) s60=460/s61=373 | full s60=481/s61=495.
- FINDING: the VALUE PATHWAY ALONE (value-equivalent latent + off-policy value, consistency-OFF + pi-only, NO world model, NO planner) ≈ FULL TD-MPC2 (~470 vs ~485) and captures the ENTIRE gap over SAC (~47→470). Consistency loss adds ~0; MPPI planning adds ~0 (even -100 on s61). CONFIRMS: what beats SAC on HopperHop = value pathway, NOT the world model or planner. (interim, runs continue to 5M; pattern stable across 2 seeds). CSVs: exp/tdmpc_glass/HopperHop_wmabl_HopperHop_{none,consistency}_s6X/seed_6X.csv (pi/mppi cols).

## 2026-07-21 ~17:45 UTC — LADDER FINAL (s60@5M, s61@4M stable): value pathway = full TD-MPC2
- SAC 47(matched)/188(n12) | value-pathway(strip+pi) 502 | +plan(strip+mppi) 487 | full(none+mppi) 509.
- value pathway ALONE ≈ full (502 vs 509), captures entire gap over SAC. WM consistency + planner add <2%. Planning hurt s61 (449<485).
- CONCLUSION: on HopperHop the TD-MPC2 win = value-equivalent latent + off-policy value, NOT the world model or planner. Paper-2 seed. CAMPAIGN COMPLETE.
- s61 finishes 5M ~18:20 (won't change conclusion). Boxes 44941373(4070 idle)+41649155(b3060) safe to spin down — all on GitHub.
