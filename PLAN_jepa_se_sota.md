# Plan — Reopening the JEPA line: a better H-JEPA with SE as *structure*, not a penalty

2026-07-08. Motivated by a supervisor question: we dropped JEPA after the anti-collapse studies, but those only
closed one sub-question. This plan lays out the honest, still-open direction — and is explicit about where it
cannot work.

## 1. What we actually closed vs left open

**Closed (the anti-collapse-as-a-regularizer question).** Uniformity/VICReg/SE-as-a-loss-term help only when the
latent *collapses*, which in our data happened in exactly one regime — closed-loop online goal-conditioned nav
H-JEPA (eff-rank → ~0; anti-collapse restored it, point-maze 0.53 → 0.95). On broad DMControl a pure JEPA does
**not** collapse (BYOL predictor+EMA asymmetry), so anti-collapse hurts (none 0.795 vs +uniformity 0.583); on
value-anchored TD-MPC2 the TD loss keeps the latent value-sufficient, so SE/uniformity both underperform default.
The "uniformity first worked, then got worse" is **regime-dependence**, not a reversal of physics.

**Left open (the SE-as-structure question).** In every SE experiment so far, SE entered as a *penalty term* whose
gradient competes with the predictor loss. We never used SE the way it is actually meant to be used: as a
**partition / community structure** that *defines* an abstraction. That is a different and untested idea.

## 2. Where a JEPA+SE SOTA can and cannot live (honest scoping)

- **Cannot:** online, dense-reward, value-based DMControl control. We have shown ~4× that structure is redundant
  there — the SimNorm+TD latent already has held-out value-decode \\(R^2 \approx 1\\). Any SE/JEPA SOTA claim on
  this benchmark will null. Do not target it.
- **Can (the three open niches):**
  1. **Goal-conditioned / geometric tasks** — the one regime where structure *already* helped (nav 0.53→0.95).
  2. **Long-horizon / hierarchical tasks** — where TD-MPC2's flat MPPI planner is weak and an abstraction that
     defines subgoals could genuinely add reach.
  3. **Offline / transfer representation quality** — frozen-encoder multi-task probes, where there is no dense
     value signal to make structure redundant.

## 3. Three concrete bets (escalating)

### Bet J1 — SE-community anti-collapse vs uniformity/VICReg, *in the collapse regime* (cheapest)
The only regime where anti-collapse helps: does **SE community structure** beat uniformity/VICReg there? This
finishes the open cell #59 and is a fair, matched head-to-head on the tasks where the latent collapses
(closed-loop nav + a DMControl collapse task). Arms: none / uniformity / VICReg / SE (min-2D-SE, fixed-λ, no
grad-match). Metric: goal-conditioned success + frozen-encoder readout-R² + eff-rank trajectory.
- **H:** on genuinely-collapsing latents, SE (which yields a *compact* community structure, eff-rank ~6) preserves
  goal-decodability better than uniformity (which inflates eff-rank but destroys readouts).
- **Kill:** if SE ties uniformity on the collapse tasks, the anti-collapse family is fully closed — report and stop.

### Bet J2 — SE as the H-JEPA abstraction (the real novelty)
Do not regularize with SE — **define the hierarchy with it.** Run min-2D-SE community detection on the JEPA latent
trajectory to partition states into communities; treat community-boundaries as **temporal subgoals**; the
high-level planner plans over community transitions, the low-level reaches the next community centroid. This is
"SE as structure," the thing we never built. Target: a long-horizon task where flat TD-MPC2 is weak
(AntMaze / a multi-room nav / a stacked-object task).
- **H:** SE-community subgoals give a usable HL action space → better long-horizon reach than flat MPPI at matched
  budget.
- **Kill:** if SE-subgoal H-JEPA ties flat TD-MPC2, hierarchy-via-SE is redundant like every other abstraction —
  a clean capstone null.
- **Cost/risk:** needs a build (community detection in the loop + subgoal-conditioned LL). ~2–3 box-days. Highest
  upside, highest variance.

### Bet J3 — SE-structured JEPA for offline/transfer representation (paper-friendly null-or-win)
Frozen-encoder, multi-task, offline: does an SE-structured self-supervised world model produce a representation
that transfers better (linear-probe across tasks) than plain JEPA or VICReg-JEPA? No dense value to hide behind.
- **H:** SE's compact community geometry transfers better than uniformity's spread-out geometry.
- Either outcome is publishable and directly extends the redundancy paper's scope to representation learning.

## 4. Sequencing & decision rule

J1 first (cheap, finishes an open cell, one box-day). If J1 shows SE ≥ uniformity on collapse tasks → J2 (the real
SOTA swing, build). J3 can run in parallel on the other box (offline, no RL loop). **Gate:** present J1's result
before committing the J2 build. No "JEPA+SE SOTA" claim without a matched multi-seed win in a niche where structure
is *not* already redundant (goal-conditioned / hierarchical / transfer — never dense value-based control).

## 5. Relationship to the current SOTA push

The value-conditioned abstraction bet (bisimulation, running now) tests structure in the *value* pathway on control
tasks — the redundancy wall. This JEPA+SE plan tests structure in the *representation* pathway on the tasks where
structure already showed a pulse (goal-conditioned). They are complementary: if both null, the program's thesis is
airtight; if J2 wins, it is the abstraction-SOTA, correctly located outside the value-redundant regime.
