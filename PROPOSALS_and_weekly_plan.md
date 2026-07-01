# TD-MPC-Glass — Proposals & Weekly Plan (post-supervisor review, 2026-07-01)

Consolidates the Part-5 discussion into 5 proposals with brainstorm + experimental plan, and a concrete
week schedule on our two boxes (b3060 = DMControl/helios-rl; b3060b = Panda/mujoco_playground). Constraint:
b3060b is shared with a mahjong user — never touch `moyuHarv` tmux; cap VRAM there.

## Framing correction that reshapes everything (from the review)
"Anti-collapse" is a **JEPA** property (pure self-predictive latent, no value/reward loss). **TD-MPC2 is not a
pure JEPA** — its value/reward losses already anchor the latent (value-sufficient, R²≈0.999), so adding an
anti-collapse term on TD-MPC2 tests *redundancy*, not JEPA anti-collapse. Correct labels:
- **JEPA anti-collapse results** = nav H-JEPA (real collapse) + Panda H-JEPA. Uniformity helps geometric; SE hurts geometry.
- **DMControl `unif_dmc` (on TD-MPC2)** = a **redundancy** result ("extra regularizer on a value-sufficient latent hurts"), same family as glass≈TD-MPC2 — NOT "JEPA anti-collapse."
This correction is the seed of **Proposal D** (test JEPA anti-collapse on a *pure* JEPA).

## The one reframe worth betting on (from the review)
TD-MPC2's win over PPO on HopperHop (367 vs 33) is **exploration**, not asymptote: planning is a *directed-exploration
operator*. "Speed-not-ceiling" is the weak claim; "**planning buys exploration → exploration unlocks final
performance on exploration-hard tasks**" is the strong, novel one. This is **Proposal A**, the flagship next bet.

---

## Proposal A — Planning as a Directed-Exploration Operator (with SE-discovered targets)  ★ flagship
**Hypothesis.** On exploration-bottlenecked tasks, model-based planning explores the state space more effectively
than model-free RL, and *that exploration* — not a higher ceiling — is the win. We can (i) *isolate* it, (ii)
*amplify* it by planning toward novelty, and (iii) *direct* it using abstraction (structural entropy) to define
exploration subgoals.
**Why us / white space.** Plan2Explore/LEXA/MAX/Go-Explore explore via curiosity/disagreement but nobody uses
**learned abstraction (SE communities / bottlenecks) to define the exploration targets a planner pursues**. We
own SE + a working planner + the exploration-difficulty frontier (escape sweep). That intersection is unclaimed.
**Brainstorm / arms.**
- A1 *Isolate*: on HopperHop + the actuation-weakened escape frontier, measure **state-space coverage** (occupancy
  entropy / #distinct SE-communities visited) of TD-MPC2 planning vs PPO vs a no-plan (pi-only) ablation. Show
  planning → coverage → success.
- A2 *Amplify*: add an **intrinsic novelty bonus to the MPPI objective** (model-disagreement or count/RND in
  latent) — Plan2Explore inside TD-MPC2. Does it extend the escape frontier (solve at even weaker actuation)?
- A3 *Direct (SE)*: run min-2D-SE on the replay-buffer latent graph → **communities = abstract states, boundaries
  = bottleneck subgoals** → HL plans toward under-visited communities, LL executes. (This is SE used for
  *structure discovery*, which is what SE is for — not as a latent regularizer, which we showed is redundant.)
**Metrics.** Steps-to-competence (LeCun axis), coverage, escape-frontier shift, final success.
**Risk.** A2/A3 are real builds; A1 is cheap and de-risks the thesis first.

## Proposal B — "When does a behavioral prior help RL?" (the taxonomy paper) — FINALIZE
**Status.** Already drafted (`paper_speed_of_learning.tex`, 12pp, compiles). Clean, honest empirical contribution:
2-axis taxonomy (prior-fit × exploration-difficulty) + the **escape-difficulty frontier** (weaken actuation →
cross from "PPO solves" to "only the prior survives") + anchor case (locomotion) + speed-lever case (Reacher,
OpenCabinet). Publishable as a workshop/empirical study; pitch as *taxonomy*, not method.
**Plan.** Finalize F1–F5 hierarchy figures, related-work sweep, author block; submit. No GPU (or regenerate figs).

## Proposal C — Abstraction as variance-reduction (the glass 6/16 "wins") — a SECTION, verify first
**Observation.** glass (SE) beats TD-MPC2 on 6/16 DMC tasks, means small & both-ways (a wash on the mean) BUT
with lower seed-variance on several (ReacherHard 976±3 vs 883±151; CheetahRun ±119 vs ±274). Possible real signal:
**SE structure reduces collapse-seed variance / improves worst-case**, even if it doesn't move the mean.
**Plan (cheap, no GPU).** Per-seed analysis of the 6 win-tasks from existing CSVs: does TD-MPC2 have a collapsed
seed there? does glass's eff_rank stay higher? is there a *predictor* (wins ↔ vanilla partial-collapse)? If it
holds → a defensible "abstraction as variance-reduction" section (not a headline). If not → report as noise.

## Proposal D — JEPA anti-collapse, done *right* (pure JEPA, not TD-MPC2) + SE-vs-uniformity + pixels
**Motivation.** Per the correction, our DMControl anti-collapse study is on TD-MPC2 (value-anchored) = redundancy,
not JEPA. The genuine JEPA test strips value/reward.
**Brainstorm / arms.**
- D1 *(running now, relabel)*: SE arm on TD-MPC2 `unif_dmc` → does SE also fight the value structure like
  uniformity? does SE+uniformity combine? (a redundancy datapoint, correctly scoped).
- D2 *(the real JEPA test)*: a **pure self-predictive JEPA latent on DMControl** — encoder + jumpy predictor +
  EMA target, **no reward/value loss** — then a frozen-latent downstream probe/policy. Compare {none, uniformity,
  VICReg, SE, SE+unif} on both a **geometric** readout (position/goal) and a **value/return** readout. Does the
  downstream-dependent taxonomy hold on a *true* JEPA?
- D3 *(pixels — where JEPA matters most)*: non-generative JEPA vs generative Dreamer WM on **pixel** manipulation,
  matched budget. This is the one regime where JEPA's information term is load-bearing (state is already
  information-rich, so on state it's redundant — which is exactly what we found).
**Why.** Cleanly separates "JEPA needs anti-collapse" (D2/D3, real) from "TD-MPC2 doesn't" (D1, redundancy).

## Proposal E — SE for hierarchical structure discovery (merges into A3 / a standalone probe)
**Insight.** We used SE as a *latent regularizer* (redundant/wrong-bias for continuous control). SE's actual
strength is *community/hierarchy discovery*. Untested uses: SE subgoal/option discovery (A3), SE encoding-tree as
the HL/LL planning hierarchy, SE-guided exploration (A2/A3). This is where "glass/SE" should live after we've
shown SE-as-representation is a dead end.

---

## This week's schedule (b3060 + b3060b)
| when | b3060 (DMControl) | b3060b (Panda/pixels) | no-GPU / writing |
|---|---|---|---|
| Mon | **D1** SE-arm finishing (running) | beat-PPO scan tail finishing | **C** per-seed robustness analysis; **B** finalize taxonomy figures |
| Tue | harvest D1 → SE-vs-uniformity table → blog §2 relabel | free | draft `PROPOSAL_planning_exploration` after user's deep-research pass |
| Wed | **A1** exploration-coverage: TD-MPC2 vs PPO vs pi-only on HopperHop + escape frontier (occupancy entropy / SE-community coverage) | (mirror A1 if needed) | write A methods |
| Thu | **A2** intrinsic-novelty MPPI arm (Plan2Explore-in-TD-MPC2) on an escape-frontier task | **D3** stand up pixel env + Dreamer baseline | — |
| Fri | **A3** SE-subgoal discovery smoke (replay-graph → communities → HL targets) | **D2** pure-JEPA-on-DMControl setup | harvest + Part-6 draft |
Priorities if time-boxed: **A1 (de-risk the flagship) > C (cheap section) > B (finish paper) > D2/D3 (JEPA-right) > A2/A3 (builds)**.

## Publishability read (supervisor questions)
- **Behavioral-prior taxonomy (B):** yes — drafted, honest, finishable now. Workshop→conference.
- **Redundancy criterion / glass (Paper A + C):** yes as the *criterion* + a variance-reduction section; NOT as "glass beats TD-MPC2" (that's noise).
- **Planning-as-exploration (A):** the strongest *new* paper if A1 confirms and A2/A3 add a mechanism; genuinely novel at the {planning × abstraction × exploration} intersection.
- **JEPA-right (D):** the principled way to keep the JEPA thread alive; D3 (pixels) is where it can actually win.
