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
