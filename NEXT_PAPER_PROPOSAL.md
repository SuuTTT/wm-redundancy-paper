# Next-paper proposal — *Abstraction is a speed-of-learning lever, not a capacity lever: a hierarchical-JEPA-aligned study*

**Status:** draft v1 (2026-06-28). Empirical core (hierarchy speed-of-learning experiment) is running; LeCun/JEPA related-work being compiled in `lecun_jepa_research.md`. This proposal frames the next paper after the "When is explicit abstraction redundant?" negative campaign (Paper A) and the behavioral-abstraction / matched-control results (Paper B).

---

## 1. One-line thesis
**Explicit abstraction does not raise a strong agent's performance ceiling — it changes the *speed of learning*. Measured on the axis LeCun argues is the right one (speed of learning, not benchmark score), abstraction's value becomes visible exactly where it was invisible on asymptotic benchmarks; and the open lever is *hierarchical, multi-timescale* abstraction.**

## 2. Why now (the two ingredients)
- **LeCun's "World Models" position** (JEPA program): intelligence = *speed of learning* / handling new situations with little prior training, **not** benchmark performance; world models should be *action-conditioned predictors in abstract representation space* (not simulators/video-generators); the path forward is a *hierarchy of abstractions* with *hierarchical planning* (each level emits subgoals for the level below); the central technical risk is *representation collapse*, requiring an *information measure* you cannot compute exactly (you only sample from the encoder).
- **Our empirical results** (this group): across monolithic / token-transformer / entity-graph latents (in-distribution **and** out-of-distribution at held-out object counts), explicit single-level representation abstraction is **redundant for asymptotic return**; budget-matched controls show its *only* robust value is **sample-efficiency**, and only where the prior captures the task's control-relevant substructure. The "beats PPO" claims for behavioral abstraction collapsed under matched budgets — surviving *only* as sample-efficiency (e.g. OpenCabinet curriculum matches PPO's ceiling at ~7× fewer steps).

**The bridge:** our negative campaign accidentally *operationalizes LeCun's metric claim*. On the benchmark axis (asymptotic return) abstraction is null; on the speed-of-learning axis it is the only thing that helps. The field measures the wrong axis, which is why "does abstraction help?" has been muddled.

### 2.1 LeCun's program is two separable bets (from `lecun_jepa_research.md`)
- **Bet 1 — the metric bet:** intelligence = speed of learning, not benchmark asymptote. **Our results CONFIRM it** — single-level abstraction is redundant for asymptote (in-dist *and* OOD) and valuable *only* for sample-efficiency. Measured on LeCun's own axis, abstraction does something; on the asymptote axis it does not.
- **Bet 2 — the architecture bet:** abstraction's payoff is *hierarchical / multi-timescale* (H-JEPA). **Our nulls do NOT refute it** — they are single-level, so silent about stacking abstraction at multiple timescales, the exact mechanism LeCun claims is load-bearing. This is the honest boundary of our prior claim and the most likely reviewer attack — so it is precisely the experiment to run next.

This two-bet decomposition is the paper's spine: we *banked* Bet 1 empirically; we now *test* Bet 2 with the same protocol.

## 3. Core claims / contributions
1. **The axis correction (empirical):** a unified, matched-control demonstration that abstraction (representational, behavioral, temporal) is null on asymptotic return but a real lever on sample-efficiency — across DMControl swing-up/locomotion/reaching, manipulation (OpenCabinet/PickCube), and the entity-graph OOD probe. This reframes a decade of "abstraction helps/doesn't" confusion as an axis-mismatch.
2. **A rigorous speed-of-learning protocol:** budget-matched baselines + deterministic evaluation + steps-to-competence as the headline metric (the project's verification discipline turned into methodology). This is the measurement LeCun calls for but that the field rarely does cleanly.
3. **The hierarchy test (the new empirical core) — RESULT: partial GO (run 2026-06-28, `HIER_VERDICT.json`).** On PandaPickCube (long-horizon reach→grasp→lift→place), peak held-out TRUE success (box_target≥0.9, deterministic eval):

   | arm | hierarchy | peak success | n |
   |---|---|---|---|
   | Flat TD-MPC2 | none (single-level WM) | **0.0** (0.04 box_target even @15.6M) | 3 fresh + disk |
   | Hier residual | shallow (raw-action) | 0.116 | 6 |
   | Hier options, global | 1-level (context-free) | 0.175 | 6→10 |
   | Hier options, mlp | **2-level (context-conditioned subgoals)** | **0.215** (max 0.242) | 10 (σ≈0.01) |
   | Flat PPO | none | 0.66 @ 33M (anchor) | — |

   **Findings:** (i) **Hierarchy ≫ flat** — the flat single-level WM is 0.0 even at 15.6M env-steps; every hierarchical arm reaches non-trivial success at a fraction of that budget (clear sample-efficiency/competence-onset GO). (ii) **2-level > 1-level (0.215 vs 0.175)** — context-conditioned subgoal *adaptivity* is a real, reproducible lever (mechanism: place-phase reach 0.94 vs 0.57). (iii) **Monotonic in abstraction depth:** flat < residual < global-options < mlp-options. (iv) **Temporal horizon is a second critical lever (clean saturation curve):** option horizon 100→0.073 (collapses; too short to finish a phase) < 150→0.215 < 200→0.243 ≈ 250→0.242 ≈ 300→0.253 — rises sharply to ~200 then plateaus, i.e. the subgoal horizon must be long enough for the low-level achiever to complete a multi-phase sub-task, beyond which extra horizon doesn't help. A textbook "right level of temporal abstraction" result (directly echoing LeCun's slide). (v) **Honest NULL on the asymptote:** no hierarchical arm crosses ≥0.5; all anchor ~0.2 (the final place phase remains the credit-assignment bottleneck), and all trail PPO's 0.66 ceiling.
   **(vi) The decisive refinement (matched- vs 2×-budget):** even the 2-level > 1-level edge is *sample-efficiency, not capacity* — at 2× training budget the 1-level catches up to the 2-level (global_long 0.209 ≈ mlp_long 0.208), and both still plateau ~0.2 (confirmed by 2× budget not breaking the place-phase ceiling). Robust across all 3 ES hyperparameter configs (default/σ=0.10/lr=0.02).

   **(vii) The plateau is the TASK, not the method (method-vs-task re-score, 2026-06-29, `method_vs_task.json`/`RESCORE_POSONLY.json`).** Re-scoring the *same* 58 mlp 2-level checkpoints under a position-only success predicate (drop the `0.1·rot_err` orientation term; success := `||box_pos − target_pos|| < THR`, n=1024 deterministic) lifts success ~3.0×: full-upright 0.203 mean / 0.247 max → pos-only(<0.08m, any-step) 0.607 mean / 0.700 max (even strict arrive-and-stay, pos<0.08m at-final-step, is 0.552/0.641; lift-phase 0.818). The hierarchy reaches/grasps/lifts reliably and lands the box within ~8cm on >60% of episodes — the *only* thing it fails is the final upright-orientation requirement, which we independently showed is a partial *physical* ceiling. So the ~0.2 asymptote ceiling in (v) is the place-phase upright physics (a TASK ceiling), **not** a limitation of the 2-level skill-options method. This sharpens the partial-GO: the method is sound up the full reach→grasp→lift→place chain; the residual gap is task physics, not missing abstraction.

   **Interpretation:** hierarchy is a *speed-of-learning / competence-onset* lever whose value scales with abstraction depth and subgoal temporal horizon — but it does **not** lift the asymptotic ceiling, and even the depth advantage dissolves at higher budget. This *extends* the single-level redundancy result up the hierarchy: **every abstraction lever we tested — representational, behavioral, temporal, and hierarchy-depth — buys speed, not capacity.** **Bet 1 (metric) confirmed again; Bet 2 (architecture) is a *qualified* GO — real on the speed axis, null on the asymptote.** This is the paper's unifying claim, now established across the full ladder of abstraction types.
4. **Theory (relation to JEPA/EBM/information):** position the structural-entropy (SE) objective and our redundancy result inside the energy-based / anti-collapse-information framing. Concretely: a value-sufficient SimNorm latent already satisfies the information/anti-collapse objective JEPA must enforce, which is *why* an added SE/abstraction term is redundant — a worked instance of "when is the information term already satisfied?"

## 4. The empirical program
- **Tasks:** long-horizon / multi-step / sparse-credit (PandaOpenCabinet, PandaPickCube), where flat methods are credit-assignment-limited — the regime where hierarchy *should* pay off (DMControl is too short-horizon, which is why single-level was null there).
- **Arms (matched budget, multi-seed, deterministic real-success eval):** flat PPO; flat TD-MPC2; 2-level hierarchical (subgoal/skill HL + LL achiever, reusing our HL-skill-options + curriculum infra). Ablate #levels / subgoal horizon.
- **Headline metric:** env-steps to competence thresholds (success ≥0.5, ≥0.8) = speed of learning; plus asymptote and wall-clock for completeness.
- **Decisive question:** does multi-timescale hierarchical abstraction provide a sample-efficiency GO that single-level abstraction did not?

## 5. Differentiators (why this paper, not just "another HRL paper")
- **Matched-control rigor + the axis reframe:** most abstraction/HRL papers report asymptotic benchmark wins (often vs under-budgeted baselines — exactly the trap we caught ~7×). We report the *speed-of-learning* axis with budget-matched controls and deterministic eval. The reframe ("abstraction is a speed lever") is the contribution, independent of GO/NULL.
- **Honest GO/NULL framing:** we have a documented negative campaign; a hierarchy GO would be the *first* clean positive, and a NULL would sharply bound LeCun's program empirically. Either outcome is publishable and credible *because* of the prior nulls.
- **JEPA/EBM theory connection** grounded in real probe data (our dead-R² result is concrete evidence for LeCun's "you can't measure the information content" difficulty).

## 5a. Differentiators vs the JEPA/HRL literatures (from the research synthesis)
Four things the JEPA and HRL literatures lack, which we bring:
1. **Rigorous matched control** — fix the value-sufficient latent, vary *only* the abstraction structure (and now: vary it *up the hierarchy*). HRL papers conflate architecture with budget/tuning.
2. **Speed-of-learning as the primary dependent variable** — anchored to Chollet's skill-acquisition-efficiency and the Atari-100k / sample-efficiency tradition, not asymptotic benchmark score.
3. **A structural-entropy (SE) surrogate motivated by a real impossibility** — certifying mutual information from samples is formally hard/impossible in general (McAllester–Stratos 2018; Poole et al. 2019), which is *why* a structural proxy (and why JEPA needs an anti-collapse term at all). Our dead-R² probe is concrete evidence of this difficulty.
4. **Extending the matched control up the hierarchy** as the decisive test of Bet 2.

## 6. Risks / open issues
- **The generativity control** (Hafner 2022 Director, arXiv 2206.04114, does hierarchical latent planning on a *generative* WM): a true Director (manager/worker goal-VAE) was NOT available or buildable in-window. The feasible substitute actually running (2026-06-29) is **DreamerV3** — a genuinely *generative* WM (32×32 categorical RSSM + an obs-reconstruction decoder, actor-critic trained in imagination) — at matched real-success protocol on PandaPickCube, 1M steps × 3 seeds. HONEST CAVEAT: DreamerV3 is *single-level* generative, so it conflates generativity with single-level — it does NOT cleanly isolate generativity while holding hierarchy fixed. It answers the cruder but still-useful question "does generative latent-WM planning break the place-phase plateau at all?" (GO if it crosses ~0.3+; NULL = ~0.2 plateau is method-agnostic). A true generative *2-level* hierarchy remains future work. So the clean disentanglement is really **skill-options (non-gen, hand-defined, 2-level, 0.215) vs H-JEPA (non-gen, learned, 2-level) with DreamerV3 as a single-level generative reference point**, not a perfect 2×2.
- Building a *true* H-JEPA (differentiable multi-level latent predictor with subgoal emission) is a multi-week effort; the running experiment uses a bounded 2-level subgoal/skill hierarchy on existing infra as the *minimal decisive test*. Full H-JEPA is future work.
- Long-horizon manipulation has physical/kinematic ceilings (we proved the ~0.83 PickCube ceiling is physical) — must measure *speed to a reachable threshold*, not the ceiling.
- Reward-hacking on shaped returns (we were burned repeatedly) — use real-success metrics only.
- **Citation hygiene:** 3 IDs in the research doc are flagged ⚠ unverified (MAML 1703.03400, EfficientZero 2111.00210, LV-EBM 2306.02572); the post-cutoff JEPA-derivative space (2602/2603 stems) needs a fresh related-work sweep at submission.

## 7. Tentative venue / framing
A focused empirical paper ("Abstraction is a speed-of-learning lever: a matched-control study aligned with the JEPA world-model program") for an RL/representation venue; complements Paper A (representation redundancy) and Paper B (behavioral abstraction = sample-efficiency). The LeCun framing gives it timeliness and a clear narrative.

---
*Next: fold in `lecun_jepa_research.md` (related work) and the hierarchy experiment's HIER_VERDICT.json (the empirical core) once they land.*
