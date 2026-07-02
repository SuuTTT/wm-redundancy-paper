# TD-MPC-Glass — Full-Arc Adversarial Review (iteration 1 → 2026-07-02)

Fable-5 session, 2026-07-02. Method: read handoff + full ledger + Paper A status + beat-PPO synthesis directly;
three parallel verification agents — (1) all 17 blog posts audited against the ledger, (2) b3060 code+data
verification, (3) b3060b flagship+hierarchy verification (env parity, config forensics, per-seed JSON checks).
Everything below traces to disk paths or file:line citations gathered read-only.

## 1. Phase-by-phase scorecard

| # | phase | claim (as it stands) | backed? | issue / severity |
|---|---|---|---|---|
| 1 | Origin SE-glass (iters 1–9, May) | glass > TD-MPC2 on HopperHop; 74% lower variance; K=4 basin causal | **NO** — reversed by Part 2 (procedure artifact), iters 2–7 (basin falsified), Proposal C (variance HIGHER out-of-sample) | was HIGH-orphaned at original URLs; **banners added 2026-07-02** |
| 2 | Paper A redundancy criterion | no explicit abstraction beats TD-MPC2 (16 nulls); redundancy conclusion | YES (16 direct nulls) | but the R²=0.9994 probe *evidence* was retracted by its own postmortem; Part 3 §8 re-affirmed it (HIGH, **bannered**); Parts 5/6 still cite R²≈0.999 (MED, open) |
| 3 | Jumpy "borrowed win" (Part 2/3) | jumpy +44–80% on PandaPickCube | **shaped-return only** — real success 0.00 both arms (Part 4); Pareto scored jumpy null-to-negative | HIGH, **bannered** in Parts 2 & 3 |
| 4 | Panda beat-PPO campaign (Parts 4–5) | residual "0.79 ≈ tie w/ PPO 0.81, 1.7× faster" | canonical replication says **0.716 < 0.810 (CI-sep both ways), ~1.6× faster** — PPO wins asymptote | MED, open: 0.79-vs-0.716 never reconciled in-post (likely milestone-shaping variant); "0.66 competence line" provenance unstated |
| 5 | OpenCabinet beat-PPO (round 8) | asymptote tie 0.9805, ~7× faster to competence, same-budget protocol-matched | **YES** — clean, budget-trap-guarded | none; this is the strongest beat-PPO-adjacent result and it's honestly scoped as speed-only |
| 6 | Escape frontier (synthesis) | actuation sweep: vanilla PPO 0/9 at fs≤0.6, prior+residual 9/9; deterministic confirm | YES (n=3/arm/scale + n=64 determ.) | strongest form of the escape claim; correctly retreats from the 2-task escape over-claim |
| 7 | H-JEPA / LeCun Bet-2 arc | learned hierarchy 0.0 from scratch → critic-bug de-confounded → machinery validated (0.941±0.137 n=8, bit-identical determinism) → solves 0.367 given competent primitive; lever = the primitive | YES — one of the most rigorous arcs (bug found+fixed, positive control root-caused) | LOW: the final nav positive control never appeared on the blog |
| 8 | Thread D (JEPA anti-collapse) | pure JEPA doesn't collapse (state+pixels+narrow data); anti-collapse neutral→harmful; BYOL asymmetry load-bearing; nav taxonomy was regime-specific | YES — n=3, multi-task, fixed-λ controls, honest self-corrections | clean |
| 9 | Thread A refutation | planning ≠ exploration operator; it's exploitation/sample-eff | YES — **verified noise-matched** (same `_current_noise` both arms; flag gates only collection, run_benchmark.py:1844 b3060b) | the one *positive* (2.2× coverage, A1-mech) is **noise-confounded** (det. π vs stochastic MPPI) — now caveated in Thread A |
| 10 | FLAGSHIP: PPO exploration wall | world model (not planner) is the exploration lever; PPO walled ≤54 through 472M, 0/5 ≥200 | **YES on data + env parity** (same repo/env/eplen/action-repeat, tuned config, per-seed JSONs exact) | 3 corrections applied: "5M-to-367"→**1M, n=1** (ratio ~470× not 94×); "plateaued"→"walled through 472M (still creeping)"; scope = **on-policy PPO** (no SAC/TD3 exists — the cheapest overturning test) |
| 11 | C hierarchy positive | feudal 4/6 vs flat 0/6 on fourroom (n=6); open rooms within variance | numbers exact (30 JSONs verified); protocol honest (determ. unshaped eval, matched env-steps) | **confound**: feudal LL gets dense self-generated shaping, flat sparse-only, no shaped-flat control; 2× total grad updates; Fisher p≈0.03 one-sided — scoped in Part 8 now |
| 12 | Positive-chase nulls (A2/D/E/MiniGrid) | all null | YES — every spot-checked number exact vs disk | MiniGrid nuance: "0 success" = sustained; KeyCorridor had transient successes (rnd 3/3 seeds touched success once) — phrasing OK as written |
| 13 | Wall-generalization (running) | early hope: wall generalizes | **interim: it does NOT** — FingerTurnHard PPO ≈975 ≈ TD-MPC2 (no wall); Pendulum best-seed catches up + **config-case BUG** (PendulumSwingUp vs PendulumSwingup: tuned override never fires) | wall is task-specific to gait discovery — a *sharper* claim; Pendulum cell not clean |

## 2. Prioritized problems (most serious first)

1. **No off-policy model-free control for the flagship.** "World-model = exploration lever" rests on on-policy
   PPO alone. **Action: run SAC (or TD3) on the same MJX HopperHop, n≥3, ~2–5M steps.** If SAC also walls → claim
   upgrades to "model-free is walled" (paper-grade). If SAC solves it → claim narrows to "on-policy PPO is walled"
   (still true, much weaker). Cheap and decisive either way.
2. **C-hierarchy shaping confound.** Add ONE control arm: flat TD3 + a comparable dense intrinsic signal (e.g.
   self-generated waypoint shaping without the HL, or count-based bonus), fourroom, n=6, 400k. If shaped-flat still
   0/6 → the hierarchy claim firms enormously. If it solves → the positive re-attributes to dense shaping.
3. **Pendulum config bug voids the wall-gen Pendulum cell.** Re-run Pendulum PPO with the intended tuned override
   applied (or report the cell as config-confounded). FingerTurnHard already shows no wall — the honest
   generalization verdict is "task-specific", write it that way.
4. **Part 4/5 "0.79 tie" vs canonical 0.716<0.810 unreconciled** (MED). One sentence in each post stating the 0.79
   variant used milestone shaping and was superseded by the base-reward matched study.
5. **R²≈0.999 still cited as evidence in Part 5 reality-check + Part 6** after its own postmortem (MED). Replace
   with "16 direct nulls" as the load-bearing evidence.
6. **Part 6 asserts the refuted planning-as-exploration thesis as a correction-of-record** (MED) — needs the same
   banner Part 5 got.
7. **Unsourced numbers in Part 5 method map** ("Pendulum 836 vs 46", "Reacher 3×", "3–4 orders sample-efficiency"
   — Part 4's own evidence is 28–160× ≈ 1.5–2 orders) (MED/LOW).
8. **Historical HopperHop number drift** (283 / 367 / 265±64 across posts; basis peak-vs-final never stated) (LOW)
   — the Part 8 correction now states the disk-true version; older posts uncorrected.

## 3. Posts needing correction banners

**Applied 2026-07-02 (this review):** phase1b (glass wins + 74%-variance + K=4 basin), iterations-8-9 ("credible
lead"), Part 2 (jumpy shaped-return win), Part 3 (R²=0.9994 §8 re-affirmation + jumpy +1017); Part 8 corrected
in-place (flagship numbers, wall phrasing, SAC scope, hierarchy confound + p-value, wall-gen interim + Pendulum
bug); Thread A (A1-mech noise-confound caveat; A2/A3 boards → NULL); Thread E (A3 board → NULL).

**Still open (MED):** Part 6 (planning-thesis banner); Part 4 + Part 5-why-it-stays (0.79-tie reconciliation +
"+8% planning on HopperHop" → +1.7); Part 5 reality-check + Part 6 (drop R²≈0.999 as evidence); external sub-site
`suuttt.github.io/tdmpc-glass` (older "lean-bootstrap beats PPO ≥0.95@19.66M vs 29.49M" claim — audit separately).

## 4. What can anchor a paper vs what's speculative

**Paper-solid (2–3 anchors):**
1. **The negative campaign + redundancy result** (Paper A): no explicit abstraction (representational SE,
   temporal/jumpy, behavioral bolt-ons) beats a strong self-predictive world model — 16 direct nulls, matched
   protocols, plus Thread D's clean reversal work (pure JEPA doesn't collapse; BYOL asymmetry load-bearing;
   anti-collapse neutral→harmful). Rigor is genuinely high; drop the R² probe as evidence.
2. **"Structured prior = sample-efficiency lever, never a ceiling lever"** — round 7/8 (PickCube 0.716<0.810
   CI-sep; OpenCabinet 0.9805 tie, ~7× faster) + the escape-frontier actuation sweep (vanilla 0/9 vs residual 9/9
   at low force, deterministic confirm) + the H-JEPA arc (every gain traced to a better low-level primitive; oracle
   ceilings). Multi-task, budget-trap-guarded, both-axes CI-separated. This is the strongest positive claim in the
   program.
3. **The exploration-wall result** — publishable AFTER the SAC control: "TD-MPC2 solves HopperHop in ~1M steps
   where [on-policy PPO / model-free baselines] remain <54 through 472M on the identical MJX env; the advantage
   survives removing the planner (noise-matched π-only) → the world model, not planning, carries it." Env parity
   verified; per-head ablation (running) adds mechanism (early read: TD/value net load-bearing; reward head
   planner-only; consistency substantial).

**Speculative / not paper-ready:** C-hierarchy positive (confounded by shaping until the shaped-flat control runs;
p≈0.03 at n=6, one strong cell); wall-generalization (interim says task-specific; Pendulum cell bugged); the
"pruning reframe" as a general law (consistent story, but supported by one refutation + one confounded coverage
measurement); WM-head ablation (single task, n=2, policy arm pending).

## 5. Verification-integrity summary

Current-phase numbers: **every one checked traced exactly** (A2, E, MiniGrid summaries; A1-decisive; 5/5 PPO-wall
seed JSONs; 30/30 C-hierarchy JSONs; WM-ablation live logs). The ~7×-fabrication history did NOT recur in this
phase. The failure mode that DID recur is softer: headline numbers quoted from memory drifting from disk ("5M-to-367",
"94×", "plateaued", handoff's wrong box for A1_COLLECT) and corrections applied forward but not backward (the May
posts stood uncorrected for 7 weeks). Both now addressed: disk-first quoting + banner-on-retraction as standing policy.
