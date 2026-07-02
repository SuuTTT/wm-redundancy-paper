# Is PPO beatable, and does abstraction help? — the unified verdict

*Synthesis of the 2026-06 campaign. Every number is a deterministic, multi-seed,
disk-backed measurement; "beats" is always qualified by the axis it holds on.*

## The question
Two things got conflated all session: (1) does **abstraction** (of any kind) help a
strong RL baseline, and (2) is **PPO** beatable. The answer to both is "yes, but only on
one axis, and not by adding abstraction." Below, three method families against two
baselines (PPO; TD-MPC2 monolith), scored on three axes: **asymptotic return (ceiling)**,
**per-env-step sample-efficiency**, and **wall-clock**.

## The three verdicts

### 1. Representation abstraction (structural-entropy "glass") vs TD-MPC2 monolith — TIE
16 DMControl tasks, n=3–4. No systematic return difference (ties within 95% CI on ~12/16;
the few separations track single collapsed seeds, both directions). Only robust effect:
glass costs **~1.35× wall-clock**. → Explicit representation abstraction is **redundant** on a
value-sufficient self-predictive (SimNorm) latent. It does not raise the ceiling; it costs more.

### 2. Behavioral abstraction (analytic controller + learned residual) vs budget-matched PPO — sample-efficiency only
Three control paradigms (energy-shaping / CPG / OSC), each = controller + residual trained by PPO,
against a **budget-matched vanilla PPO** (the key control; earlier "beats PPO" claims used an
under-budgeted benchmark PPO).

| regime | asymptotic return | sample-efficiency |
|---|---|---|
| Reaching, fit member (ReacherHard) | tie (resid 981 vs vanilla 977) | **resid wins** (~3×: solved ~1M vs ~9.8M) |
| Reaching, unfit member (FingerTurnHard) | **vanilla wins** (952 vs 923) | no advantage |
| Locomotion (all 5 tasks, n=5) | **vanilla wins all 5** (e.g. Cheetah 914 vs 767; 3/5 CI-separated) | no advantage |
| Sparse swing-up (CartpoleSwingupSparse) | tie (both solve) | resid ~2× faster |
| Pendulum swing-up (energy-shaping prior) **n=5** | **resid wins** (resid peak 836.3 / first-eval 831.7, 5/5 reach thr @9.8M; vanilla peak 46.8, **0/5 ever reach thr** over the whole budget) | resid competent at first checkpoint; vanilla never escapes |
| OpenCabinet (hard manipulation) | tie (0.987 vs 0.988) | **resid wins ~7×** |

→ A behavioral prior **never raises the true asymptotic ceiling** when the baseline both fits the
task and can explore it. But the headline must be qualified on a **2-axis taxonomy (prior fit ×
exploration difficulty)** — the clean "speed lever, vanilla catches up to the same asymptote" story
does NOT hold uniformly:
- **Exploration-easy + unfit prior** (locomotion CPG, FingerTurn): residual is an **anchor** — worse
  on *both* speed and asymptote. Vanilla wins outright.
- **Exploration-easy + fit prior** (ReacherHard, CartpoleSwingupSparse): tie asymptote, residual
  ~2–3× faster. *This* is the clean speed-lever case.
- **Exploration-hard + strong prior** (Pendulum swing-up, OpenCabinet, and cf. TD-MPC2 vs PPO on
  HopperHop below): the baseline **never reaches competence in the budget**, so the residual wins on
  *practical* capacity, not just speed. This is a budget-bounded escape-from-local-optimum, not a
  proof the true asymptote differs (give vanilla unbounded exploration and it might close it) — but
  within any realistic budget it is a real capacity-in-practice win.

So the corrected one-liner: **a structured prior buys sample-efficiency where it fits, practical
capacity where the baseline is exploration-bottlenecked, and is dead weight (an anchor) where it
neither fits nor is needed.** Both n=5 now: the 2026-06-29 locomotion
figures confirm the anchor case cleanly (vanilla wins both axes, all 5 tasks); the pendulum escape case
is confirmed at **n=5 vs n=5** (resid peak 836.3, 5/5 reach thr @9.8M; vanilla peak 46.8, 0/5 ever reach
thr) — `pendulum_abstraction/finalize.log` = DONE.

**⚠ ESCAPE CASE IS TASK-SPECIFIC, NOT ROBUST (honest correction, 2026-06-29).** A 2nd escape attempt on
**CartpoleSwingupSparse** (same recipe: energy-shaping prior + residual vs budget-matched vanilla, sparse
0/1 gate) was **NULL for escape**: at full n=5 the residual reaches competence 5/5 (peak 711.8) BUT
**budget-matched vanilla ALSO solves it 5/5 (peak 800.5, slightly higher asymptote)** — verdict
`NULL-speed-lever`, `cartpole_escape/escape_task2_VERDICT.json`. The earlier "preliminary GO" (n=1 vanilla
≈0) was a single-seed/early-checkpoint artifact (the exact under-sampled-baseline trap from
[[class-controller-budget-trap]]). LESSON: the pendulum escape ("vanilla 0/5 ever") does NOT generalize —
CartpoleSwingupSparse is enough exploration-easier that vanilla PPO (1024 envs, full budget) solves it. So
the ESCAPE cell rests on **a single task (Pendulum)**; "strong prior + residual escapes where vanilla
fails outright" is real but **task-specific to a genuinely hard exploration gate**, not a robust multi-task
phenomenon. Report it that way — do NOT claim a 2-task escape result.
Figures: `helios-rl/exp/figs/speed_*.png` + `steps_to_competence_*.json`.

**UPDATE — ESCAPE IS A CONTROLLED PHENOMENON (2026-06-29 escape-difficulty sweep; the strong result).**
Rather than hunt a 2nd escape task, we made escape a *controlled axis*: sweep CartpoleSwingupSparse
ACTUATION strength (a FORCE_SCALE multiplier on the motor command, identical for both arms; the energy
controller stays fair — it still swings up at fs0.25, collapses only at 0.15). Final training-eval sparse
return (n=3/arm/scale; `cartpole_escape/escape_difficulty_sweep.json`, raw in `escape_sweep/logs/`):

| FORCE_SCALE | vanilla PPO (n=3) | energy-prior + residual (n=3) |
|---|---|---|
| 1.0 | solves (~800, 3/3) | solves (~711, 3/3) |
| 0.6 | **0,0,0 → 0/3** | 690,686,721 → **3/3** |
| 0.4 | **0,0,0 → 0/3** | 65,687,610 → **3/3 nonzero** |
| 0.25 | **0,0,0 → 0/3** | 96,350,333 → **3/3 nonzero** |

Vanilla transitions from solving (fs1.0) to TOTAL collapse (exactly 0.000 across all 9 low-force runs) at
fs≤0.6, while the prior+residual keeps solving down to fs0.25. The separation is absolute (vanilla 0/9,
residual 9/9 nonzero at low force) — the GO holds before the deterministic n=64 harvest even refines
frac_solved. **So escape is NOT a one-off: it is a predictable function of exploration difficulty, and the
analytic prior shifts the solvable frontier.** Corrected taxonomy claim: at standard actuation cartpole is
exploration-easy (vanilla solves → NULL/speed-lever); weaken actuation and you cross into the escape regime
where only the prior survives — Pendulum is just one task already in that regime. This is the strongest,
most defensible form of the escape result. **CONFIRMED deterministically** (n=64, final ckpts,
`escape_sweep/escape_difficulty_sweep.json` `verdict: GO-escape-frontier`): vanilla k/3 = 3/3,0/3,0/3,0/3 at
fs 1.0/0.6/0.4/0.25; residual k/3 = 3/3 at every scale (mean frac_solved 1.0/1.0/1.0/0.969) — matches the
training-eval to the decimal. Fig: `helios-rl/exp/figs/escape_frontier_cartpole.png` (2 panels).

### 3. TD-MPC2 (and TD-MPC2 + abstraction) vs PPO — a fundamental axis tradeoff, no Pareto-dominance
TD-MPC2's SimNorm latent + learned model + planning + dense gradient updates is *itself* the
abstraction that beats PPO — on **sample-efficiency**. The Pareto experiment (HopperHop n=2,
CheetahRun) asked whether any TD-MPC2 variant can dominate PPO on **both** return/sample-eff
*and* wall-clock:

| arm (HopperHop, 2M steps) | final return | env-steps→200 | wall-clock |
|---|---|---|---|
| vanilla TD-MPC2 | 306 | 350k | 7,193s |
| jumpy TD-MPC2 (temporal abstraction) | 277 | 550k | 8,300s |
| efficient TD-MPC2 (0.95M params, K=32) | 195 | 1.9M | 3,607s |
| PPO | **0.04** | never | **621s** |

- **TD-MPC2 beats PPO on return + sample-efficiency by 3–4 orders of magnitude** (PPO never
  learns to hop; CheetahRun corroborates: TD-MPC2 701 vs PPO 290 at matched 30M).
- **PPO beats TD-MPC2 on wall-clock by ~10–30×** (brax 512-env parallelism; gradient update
  every env-step). The efficient arm narrows this ~2× but cannot close it — the gap is
  **architectural**, not a tuning artifact.
- **Adding temporal abstraction (jumpy) does not help** — null-to-mildly-negative on return,
  sample-efficiency, *and* wall-clock (and showed the historical transient-collapse fragility).

**CheetahRun corroborates** (the PPO-competitive regime): vanilla 671, jumpy 642 (no gain),
efficient 564 (2.2× faster wall-clock), PPO ~270 — and PPO **never reaches return 500 even at 30M
steps** while every TD-MPC2 variant reaches it in <1M.
**⚠ CORRECTION (2026-07-02, anatomy replication):** that PPO CheetahRun reading was a **budget/config
artifact** — the tuned mujoco_playground PPO config at 285M steps reaches **892–922 (3/3 seeds)**, matching
SAC (918/912 @10M). CheetahRun is a slow-but-converging case, not a wall; TD-MPC2's edge there is
sample-efficiency only (~600+ at 0.55M). The only genuine exploration wall in the program remains HopperHop
(0/5 PPO seeds ≥200 at 472M). See `bet2_null_results.md` anatomy-replication entry.

**Nuance on the wall-clock axis (refines the blunt "PPO wins wall-clock"):** PPO wins raw
*throughput* (steps/sec, ~10–30× via 512-env parallelism + a gradient update per env-step), but on
these exploration-hard tasks PPO **never reaches a competent return at all** (HopperHop 200,
CheetahRun 500). So PPO does *not* win "wall-clock *to a competent policy*" — TD-MPC2 reaches the
threshold in ~3,200s whereas PPO never gets there in its budget. The honest tradeoff is therefore:
**PPO wins raw throughput; TD-MPC2 wins both sample-efficiency and wall-clock-to-competence on
exploration-limited tasks.** On easy dense tasks where PPO does reach competence, its throughput
edge makes it the practical winner.

→ **No TD-MPC2 variant Pareto-dominates PPO on raw throughput. No abstraction variant beats vanilla
TD-MPC2. And on exploration-hard tasks, TD-MPC2 already beats PPO on both sample-efficiency and
wall-clock-to-competence — without any added abstraction.**

## The unifying principle
Abstraction — representational, behavioral, *or* temporal — **redistributes complexity; it does
not remove it.** It can lower the *statistical / optimization* cost (sample-efficiency, exploration)
when the baseline is optimization-limited and the structure fits, but it never raises the
*representational ceiling*, because a strong baseline already finds a value-sufficient solution.

## The honest bottom line
- **Is PPO beatable?** Yes — on **per-env-step sample-efficiency** (TD-MPC2 already does so
  decisively, and on exploration-bottlenecked tasks like HopperHop PPO fails outright). **No** —
  on **wall-clock** in a fast simulator, where PPO's architecture dominates.
- **Does abstraction help beat it?** No. Every abstraction we added — SE on the latent, a
  behavioral controller+residual, a temporal/jumpy model — was redundant or harmful on top of a
  strong baseline. The abstraction that *works* (SimNorm + model + planning) is the one already
  inside TD-MPC2; bolting more on top buys nothing.
- **Can we combine TD-MPC2 + abstraction to beat both PPO and TD-MPC2?** No — clean null. The
  sample-efficiency ↔ wall-clock tradeoff is fundamental; there is no free lunch.

*Practical takeaway:* choose by what is scarce. Env-steps scarce (real robots, costly sims) →
TD-MPC2. Wall-clock scarce (fast sims) → PPO. Either way, explicit abstraction is not the lever.

## Addendum (2026-06-30) — two confirmations that sharpen the thesis

**A. Learned residual on an analytic skill = speed lever, not ceiling lever (2 Panda tasks).**
On PandaPickCube the analytic skill caps at ~0.37 (contact physics: cube tips in the 2-finger grip). A
*learned* full-authority residual over it reaches **0.716±0.014 (n=3)** — breaking the analytic wall (the
wall is *learnable*, tilt driven to 1.9°, not a hard morphology limit) — but a matched-budget vanilla PPO
wins the asymptote at **0.810±0.006 (n=3)**, both axes CI-separated; the residual is just ~1.6× faster.
Replicated on **PandaOpenCabinet**: residual asymptote **0.9805 (n=7) == vanilla 0.9805 (n=2)** (same
saturable ceiling) with residual **~7× faster to competence**. Unified: the structured prior's asymptote is
**≤ vanilla** (below it, or a tie at a shared ceiling) and the prior is **markedly faster on both tasks**.
Bounded residuals (α≤0.5) are unstable/underpowered (NULL); only near-full authority (≈ warm-started policy)
breaks through. → Same one-liner as the rest of the campaign: **a structured prior buys sample-efficiency,
never a higher ceiling.** Budget-trap guarded throughout (the old "PPO 0.66" was under-budgeted; matched PPO = 0.81).

**B. Anti-collapse for self-predictive latents is downstream-dependent (a taxonomy).** The lever that fixes a
collapse-prone JEPA/SimNorm latent is **relational anti-collapse** (a one-line uniformity loss), but the right
term depends on what you decode: relational/uniformity for **goal-conditioned geometric** latents (nav point-maze
0.530→0.954, CI-separated), **nothing extra** for **value-based control** (DMControl: uniformity is WORST of
{default,unif,vicreg} on all 3 tasks; a value-aware variant is WORSE still — CheetahRun 20.6 / WalkerWalk 49.5 /
FingerSpin 1.6 vs default 58.9 / 293.8 / 249.4), and **never SE community structure** for either continuous case
(Panda SE NULL; nav benefit is partition-independent). → Explicit representation structure (SE) remains redundant;
the only useful anti-collapse is the minimal relational one, and only where the downstream task is geometric.
Full detail + all matched controls in `bet2_null_results.md`; narrative in tdmpc-glass blog Parts 50–52.
