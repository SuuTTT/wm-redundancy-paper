# TD-MPC-Glass — Full Iteration Log

2026-07-08. A catalogue of every experimental iteration in the program, grouped into seven phases, with what each
did, where the artifacts live, and the lesson that carried forward. Counts: **~80 distinct experimental
iterations** — ~15 foundational (glass/redundancy), ~60 numbered campaign iterations (#1–#60), 5 "next-phase"
bets, and 3 SOTA bets. Numbers throughout are read from the append-only ledger `bet2_null_results.md`.

## Repositories & artifact roots (verified)

- **Papers / ledger:** `github.com/SuuTTT/wm-redundancy-paper` — `bet2_null_results.md` (the source-of-truth
  ledger), `paper_wall_mechanism.tex` (Paper 3), `main.tex` (Paper A), `paper_speed_of_learning.tex`,
  `exp_results/` (harvested jsonl+CSVs), `code_snapshot/` (both boxes' code tarballs), `HANDOFF_0707.md`.
- **Public code:** `github.com/SuuTTT/tdmpc-glass` (+ release `artifacts-20260707` = model weights),
  `github.com/SuuTTT/honest-rl-bench` (toolkit + tutorial site).
- **Blog:** `suuttt.github.io/projects/` — Parts 1–10 + the three-papers study doc.
- **Compute:** two Vast 4×3060 boxes — `b3060` (`/root/helios_wmablate/exp/…`) and `b3060b`
  (`/root/tdmpc_glass/helios-rl/exp/tdmpc_glass/…`). Milestone weights mirrored to HF `Dannibal/tdmpc-glass-milestones`.

---

## Phase 0 — Foundations: the redundancy criterion & six mirages (~15 iterations)

**What:** the original question — does adding structure (structural-entropy "glass" latents, entity/graph world
models, calibration-shaped models, jumpy k-step models) to TD-MPC2 buy anything on control? Iterations: **Paper A
draft**; **B0** multi-object MJX env; **B1** entity-factored WM + value-coupling probe; **B2** mechanism-check GO/NO-GO;
**Iter 32** calibration-shaped WM (ICLR bet); **Iter 33** high-DoF jumpy vs vanilla @1.5M; **Iter 34** GWM
graph-as-simulator mechanism-check; **Iter 35/36** GWM compositional-OOD control (graph vs monolithic); **D1**
JAX venv + contact-task WM + phase-binned k-step error.

**Result:** null across the board — structure ties vanilla at matched budget and its couplings go unused by the
value pathway. Blog: *"Six Mirages: a TD-MPC2 abstraction null"* (2026-06-07), *Part 2: a mechanism check saved a
campaign*. **Lesson:** the SimNorm latent is already value-sufficient (held-out value-decode R²≈1); a mechanism
check (does the value head consume the new coupling?) predicts the null before you burn the GPU.

## Phase 1 — Panda manipulation: can abstraction/hierarchy solve PickCube? (#1–#17)

**What:** insert abstraction into a hard contact task. #1 HL + residual policy (PickCube); #2 HL phases as
skill-options; #3 demo-seeded TD-MPC2; #4/#5/#5b/#5c beat-PPO via warm-start / in-loop residual / persistent
authority; #6–#12 toward 0.95 (bigger+longer residual, closed-loop retry, hard-config curriculum, place-phase
fine-control, tail characterization, deploy best-of-k, InFOM/CompPlan repro); #13–#17 orientation-aware &
learned grasp, physical-vs-learnable ceiling.

**Result:** abstraction-in-loop reaches ~0.24 (skill-options) → residual breaks the analytic ceiling to ~0.79 →
but **matched-budget PPO wins asymptote (~0.81–0.83)**; the last ~17% is a *physical* upright-grasp ceiling
(99.88% of the far tail is kinematically infeasible to grasp upright). Artifacts: `hl_pickcube/*`,
`exp/tdmpc_glass/hl_{residual,options,subgoal,orient}/`. **Lesson:** abstraction buys **sample-efficiency** and
**structured/interpretable competence**, not a higher ceiling; where PPO loses it's an exploration issue, where
abstraction loses it's a wrong-prior issue.

## Phase 2 — Beat-PPO campaign & benchmark build (#18–#42)

**What:** turn the anecdotes into a matched benchmark. #18–#21 TD-MPC2-vs-PPO efficiency/home-turf; #22–#26 TAMP
+ curriculum on OpenCabinet; #28–#30 abstraction-as-curriculum (warm-start then RELEASE) → **clean sample-efficiency
beat on OpenCabinet** (≥0.95 at ≤19.66M vs PPO 29.49M, 4/4 seeds — the program's cleanest positive); #31–#33 vs
TD-MPC2 on favorable tasks; #32 full tdmpc-glass+tdmpc2 sweep; #35 PPO column 16×3; #36 humanoid; #37 manipulation
methods (Paper B); #38–#41 class-controllers (energy-shaping, CPG, OSC) — **matched-vanilla-PPO controls collapsed
most "beats"** to sample-efficiency only; #42 entity-graph GO test.

**Result:** the one durable beat is sample-efficiency (OpenCabinet curriculum, ReacherHard OSC ~3×); the
"class-controller beats PPO" claims were under-budgeted-baseline artifacts, corrected in the open. Artifacts:
`exp/tdmpc_glass/{full_sweep,baselines_ppo_sac,reach_class,locomotion_class}/`; blog Parts 18–41. **Lesson:** always
run a **same-budget vanilla-PPO control**; a cross-budget baseline manufactures fake wins.

## Phase 3 — Speed-of-learning & the LeCun bets (#43–#60)

**What:** score by learning speed and test LeCun's hierarchy/JEPA arguments. #43 hierarchical speed-of-learning;
#44 JEPA research + next-paper proposal; #45 finalize Paper A; #46 speed-of-learning paper; #48 Director
(generative hierarchy); #49–#50 hierarchy method-vs-task + speed curves; #51–#56 H-JEPA (faithful LeCun arch) →
**multi-seed NULL on PandaPickCube** (bottleneck is the low-level motor primitive, not abstraction); #54
escape-difficulty actuation sweep (CartpoleSwingupSparse); #57 SE-structured JEPA on Panda; #58 relational
anti-collapse (uniformity vs VICReg vs SE); #60 learned grasp.

**Result:** hierarchy/JEPA help only via a *competent primitive*, not learned abstraction; anti-collapse is
downstream-dependent (helps geometric goal-conditioned latents, hurts value-based control). Artifacts:
`exp/{hjepa,hjepa_solve,hjepa_navctrl,unif_dmc,se_nav_geo2}/`; blog *thread-b/c/d/e*, Parts 43–52. **Lesson:** a
non-generative predictor's anti-collapse term is *not* free — its sign flips with the downstream objective.

## Phase 4 — Five bets for the next phase (Part 6 → Part 7)

**What:** five falsifiable bets. **A** planning-as-exploration (+ novelty-MPPI); **B** behavioral-prior taxonomy;
**C** glass-as-variance-reduction; **D** pure-JEPA done right; **E** SE-structured latent.

**Result (Part 7):** A refuted (decisive on CartpoleSwingupSparse; novelty-MPPI worse on 4/4 matched seeds);
C null; D reversal firmed (anti-collapse that helps nav hurts control), D3 pixel-JEPA null; E/D1 SE hurts, no
synergy. Four of five null-to-harmful. Artifacts: `exp/{proposal_A1*,proposal_D2*,proposal_D3*,D_se_structure}/`;
blog Parts 6–7. **Lesson:** every attempt to *add* structure to win reproduced the oldest finding — the TD value
pathway is the engine; unconsumed structure is redundant. **This failure drove the pivot to dissection.**

## Phase 5 — The dissection: wall, mechanism, sufficiency (Papers 3–4)

**What:** stop adding, start explaining *why the planner beats PPO*. **Claim 1 (wall):** HopperHop PPO 0/5 ≥200
@472M (survives entropy ×3/×10); SAC 6/12@5M→6/9@8M; TD-MPC2 6/6@1M; HopperStand graded barrier PPO 2/16;
Acrobot (contact-free) → no wall. **Claim 3 (mechanism):** 5-loss × 4-task ablation — value & policy individually
fatal, reward planner-only, consistency mildest. **Humanoid:** only SAC survives 21-DoF. **Anchor:** TD-MPC2
HopperHop 5M = 420±113 (n=12). **Paper 4 (sufficiency 2×5):** consistency-OFF-from-scratch removable only on
HopperHop (n=8, 7/8 in band); load-bearing Walker −23%, Cheetah −38%, Acrobot −44%; Cartpole both-fail.
Thesis: the consistency loss underwrites planner rollout quality wherever the planner carries learning.

**Artifacts:** `paper_wall_mechanism.tex` + figures; `exp/{ppo_entropy_*,sac_8m_confirm,wallprobe_*,wm_head_ablation,
suff*}/`; blog Parts 8–9 + three-papers study doc; weights release `artifacts-20260707`. **Lesson:** a rigorous
null program's real output is a *mechanism*; "why X wins" is more citable than "we added Y and tied."

## Phase 6 — The SOTA push (bets 1–3)

**What:** build a better world-model objective from the sufficiency lessons. **Bet 1 VAC** (value-aware
consistency) — no-go −8.9%/−4.4%. **Bet 2 URC** (rollout-reliability consistency) — no-go ~−7/−9%. See
`REPORT_consistency_reweighting_failure.md`. **Bet 3** value-conditioned abstraction (bisimulation sweep, then
value-sufficient bottleneck) — queued, `SOTA_PROPOSAL_valuecond_abstraction.md`.

**Artifacts:** `exp/vac/` (both boxes), `tdmpc2.py` env-gated `VAC_LAM`/`URC_LAM` (backups `.bak_prevac`/`.bak_preurc`);
blog Part 10. **Lesson (so far):** you cannot beat TD-MPC2's uniform consistency loss by reweighting it — it is
near-optimal in form. The only remaining abstraction lever is changing what the value pathway *sees*.

---

## The one law, restated

Across ~80 iterations and six phases, a single result recurs: **in a value-based planner, structure and
reweighting buy nothing the TD value pathway does not already consume.** Abstraction buys sample-efficiency where
its prior fits and interpretability always; it does not raise the ceiling, and it cannot improve a world-model
objective that is already value-sufficient. Every "beat" that survived scrutiny was a sample-efficiency beat under
a matched-budget control; every ceiling beat dissolved into a physical limit or a baseline-budget artifact. The
program's durable contribution is therefore a **dissection** (why the planner wins) plus a **near-optimality
result** (its world-model loss can't be improved by reweighting) — not a new SOTA method, unless bet 3 surprises us.
