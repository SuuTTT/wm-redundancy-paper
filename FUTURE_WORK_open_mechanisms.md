# Future Work — two open mechanism questions we touched but never nailed

2026-07-08. Both surfaced as supervisor questions. Each is a real "why", each has a partial answer and a decisive
experiment we have *not* run. Both should be in the paper's Future Work / Limitations.

---

## Q1. Why is HopperHop unique?

HopperHop is unique in **two** ways at once, and the tension between them is the interesting part:

- It is the task with the **sharpest categorical PPO wall** (0/5 seeds ≥200 at 472M; survives entropy ×3/×10).
- It is the **sole removable cell** in the sufficiency grid — the one task where training the world-model
  (consistency) loss OFF from scratch matches the full model (n=8, 7/8 in band), while Walker/Cheetah/Acrobot lose
  −23/−38/−44%.

Those look contradictory: how is the *hardest* task (for PPO) the one where you can *delete* the world model?

**Current hypothesis (stated in Paper 4, not yet mechanism-checked):** the two difficulties are on different axes.
HopperHop is **exploration-hard but execution-simple** — the hard part is *finding* the hopping gait (a
contact-critical exploration problem the TD value+policy pathway solves, which is why TD-MPC2/SAC beat PPO), but
once found the gait is a **low-dimensional limit cycle** the policy head can execute without accurate multi-step
dynamics prediction — so the consistency loss is redundant. Walker/Cheetah/Acrobot are **execution-precision-hard**:
their *return level* is set by fine continuous control that the planner's accurate multi-step rollouts provide, so
the consistency loss is load-bearing there.

**What we have not done (the decisive probes):**
1. **π-only vs MPPI at matched weights on Hop vs the dense tasks.** If Hop is π-learnable, policy-only (MPPI
   disabled) should reach ~full return on Hop but fall well short on Walker/Cheetah/Acrobot. This single ablation
   would confirm or kill the "execution-simple" story. (We have the FORCE_CK same-weights harness from A1 — cheap.)
2. **k-step latent rollout error, stripped vs full, Hop vs dense.** Does stripped-Hop keep low open-loop rollout
   error (because the periodic gait is low-entropy and easy to predict) while stripped-Walker's rollout error
   explodes? This directly tests why removing consistency hurts the dense tasks but not Hop.
3. **Behavioral characterization.** Measure the action-sequence periodicity / control-precision demand of the
   optimal policy on each task (limit-cycle vs continuous fine control). Predicts the removable/load-bearing split
   from task structure, not from the ablation.

**Why it matters:** it would turn Paper 4's empirical split ("removable on Hop, load-bearing elsewhere") into a
*predictive* law — "the world-model loss is removable exactly on exploration-hard-but-execution-simple tasks" — and
explain the wall/sufficiency tension. Currently the split is descriptive.

---

## Q2. Why does the JEPA latent collapse in nav but not in control?

The closed-loop online goal-conditioned **nav H-JEPA** latent collapses (eff-rank → ~0); the **value-based control**
latent (TD-MPC2) does not. We ran a partial isolation (Thread-D final cell, n=3, matched updates) and got a
*partial, fragile* answer.

**What we established:**
- Breaking the closed loop (offline fixed-buffer) **mitigates** collapse: online dead-latent on 3/3 seeds,
  offline on only 1/3 → closed-loop feedback is a **driver/amplifier**, not the sole cause.
- Data-width alone does **not** reproduce it (narrow/on-policy-ARS buffers don't collapse) → it isn't just
  "narrow data."
- A pure JEPA on broad DMControl does **not** collapse (predictor+EMA/BYOL asymmetry) → it isn't intrinsic to
  JEPA.
- **Representation health is decoupled from control** in nav (task success ~0.64 regardless of collapse).

**What we did NOT resolve (the ledger says so explicitly):** the *sufficient* trigger. The leading unproven
hypothesis is that **control latents are anchored by a dense TD value gradient into the encoder** (every step's
reward shapes the latent), while nav's **goal-conditioning provides a weaker/sparser anchor**, so under a co-evolving
closed loop the nav representation has nothing pinning its rank and collapses. This is consistent with everything
above but has not been isolated.

**The decisive experiment (unrun):** a single-task A/B that varies *only the anchor strength* — take the nav
H-JEPA and add a dense auxiliary regression target (e.g. predict raw state / a dense shaping signal) at graded
weight; if collapse disappears as the dense anchor strengthens, "dense value/state anchor prevents collapse" is
confirmed. Cross it with online/offline to separate anchor-strength from closed-loop feedback.

**Why it matters:** it explains *when a self-supervised world model needs an anti-collapse term at all* — the
practical rule "add anti-collapse only when the downstream signal is too sparse to anchor latent rank" — which is
the actionable content of the whole anti-collapse thread, currently left as a regime observation rather than a law.

---

## Q3. Bet B (behavioral-prior taxonomy) — resolved in substance, unpackaged as a law

B was not re-run in the five-bet round because its answer already existed from the Phase-2/3 class-controller
campaign (#38–#41, all with matched-budget vanilla controls): OSC/reaching ties-but-3×-faster on the fit member
(ReacherHard 980.7 vs 976.8) and loses on the unfit (FingerTurnHard 923 vs 952); CPG/locomotion loses to matched
vanilla on all 5 tasks; energy-shaping is a pure *escape* lever on exploration-bound Pendulum (5/5 @836 vs vanilla
0/5 @47) but backfires on Acrobot (~70 vs ~200) where the prior is too weak. The implied taxonomy: a controller
helps — as **sample-efficiency, never a higher ceiling** — iff its actuated DOF match the goal DOF *and* it is a
competent controller for the task; it is an escape lever only when the task is exploration-bound and the prior
already reaches the reward region; otherwise it is dead weight or an anchor.

**The gap:** every datapoint is a different task/prior pairing — anecdotes consistent with the rule, not a
controlled test of it. **Decisive experiment (unrun):** hold one prior family fixed, vary *only* the
actuated-vs-goal-DOF overlap across a matched task set, and show the sample-efficiency multiplier is monotone in
that overlap. One box-day; turns B from "supporting evidence for the sample-efficiency framing" into a citable
predictive taxonomy. Lower priority than Q1/Q2 (it confirms rather than opens), but the cleanest way to actually
*claim* Bet B.

## Recommendation

Add both to the paper's Future Work, and fold the concrete probes into the two active plans:
- **Q1 probes** → cheap, reuse existing harnesses (FORCE_CK π-vs-MPPI, k-step rollout logger); good one-box-day
  experiments to run between SOTA bets. They sharpen Paper 4.
- **Q2 probe** → the anchor-strength A/B belongs in the JEPA+SE plan (`PLAN_jepa_se_sota.md`) as bet J0, run before
  J2 — it tells us whether SE-structure even has a collapse to fix.
