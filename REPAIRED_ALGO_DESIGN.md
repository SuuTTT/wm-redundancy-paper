# Value-Sufficiency Gating (VSG): turning the VBN diagnostic into a repaired algorithm

**Goal.** Upgrade the paper's contribution from "a diagnostic (+ a clean negative)" to
"a diagnostic **and** a probe-driven algorithm that matches full TD-MPC2 at lower compute."
The gate negative in the current draft is a *soft in-plan* down-weighting (touches ~3% of the
plan score at horizon 3) — the wrong shape. VSG is a **hard, zero-regret, deploy-time route**.

## The two removable components (and why the win is real)

From the archived 4-task, multi-seed table (median, ~2.5M steps; acrobot from live harvest):

| task | full+MPPI | full+π | strip+π (value-pathway) | strip+MPPI | planner gain (mppi−π, full) |
|---|---|---|---|---|---|
| HopperHop | 474 | 448 | 460 | 439 | **+26 (5%)** |
| Acrobot   | 395 | 388 | 345 | ~49 (collapse) | **+7 (2%)** |
| CheetahRun| 667 | 368 | 526 | 521 | +299 (80%) |
| WalkerRun | 803 | 710 | 600 | 533 | +93 (13%) |

Two orthogonal axes:
1. **Planner value** = full+MPPI − full+π. Large on Cheetah/Walker, ~0 on Hopper/Acrobot.
   MPPI costs `n_samples × horizon × n_iter` model rollouts per action (512×3×~6 ≈ 9k) vs 1
   forward pass for π → dropping it is a ~10³–10⁴× cheaper action at deploy.
2. **Representation (WM) value** = does the consistency loss make π-only / planning work.
   Load-bearing on Acrobot (strip+MPPI collapses 395→49) and Walker; ~0 on Hopper.

VBN (a value-compressibility probe) measures **axis 2**. So the honest links are:
- VBN-redundant ⇒ representation removable (consistency-light OK).  [tested: Hopper ✓]
- VBN-essential ⇒ keep the WM (strip collapses).                    [tested: Acrobot ✓]
- The **planner** axis is separate; do not over-claim VBN predicts it (Cheetah = mid-VBN,
  huge planner gain — a live counterexample to a naive single rule).

## The algorithm

**VSG-planner (zero-regret, online, deploy-time).** During training, at every eval we already
compute both `π-eval` and `MPPI-eval`. Maintain an EMA of `g = MPPI-eval − π-eval`. At deploy,
if `g ≤ ε` (planner not helping beyond noise), act with π-only; else plan with MPPI.
- **Cannot lose return by construction** — it only drops MPPI where MPPI ≈ π on held-out eval.
- Saves the entire planner cost on the planner-redundant subset.
- ε set from the per-task eval seed-spread (e.g. ε = 1 SEM).

**VSG-repr (a-priori, train-time).** Use the cheap VBN probe on an early checkpoint to decide
whether to pay for the consistency loss / world model at all. VBN-redundant ⇒ train the value
pathway (consistency off); VBN-essential ⇒ full model. This is the *diagnostic → prescription*
link and the at-risk empirical claim (see falsification).

**VSG (full)** = VSG-repr at train time ∘ VSG-planner at deploy: route each task to the cheapest
apparatus that matches full return.

## Claims and falsification

- **C1 (efficiency, robust).** VSG-planner matches full TD-MPC2 return suite-wide (≤ε per task by
  construction) while deploying π-only on the planner-redundant subset → report aggregate
  deploy-compute saving X% and per-task action-FLOP reduction. *Falsified if* the planner-redundant
  subset is empty (it isn't: Hopper, Acrobot already qualify).
- **C2 (prediction, at-risk — the diagnostic link).** VBN compressibility correlates with
  representation-removability across the suite (Spearman over ≥8 tasks). *Falsified if* ρ is weak;
  then we retreat to "VSG-planner is a task-agnostic zero-regret mechanism; VBN predicts only the
  representation axis" — C1 + C3 still stand. **Report honestly either way.**
- **C3 (routing beats fixed policies).** VSG > always-MPPI on compute at equal return; VSG >
  always-π on return (Cheetah, where π-only loses 45%). A 2-D Pareto (return vs deploy-FLOPs)
  with the three fixed policies + VSG on the frontier.

## Run matrix (what the campaign needs)

Per task: **full** (`ABLATE=none`) and **strip** (`ABLATE=consistency`), `run_arm_v2.sh`
(v2mppicol logs π+MPPI eval), 3 seeds, 1.5–2.5M steps. VBN probe per task (reuse existing
fingerprints where we have them: cheetah/acrobot/walker/cartpole/finger).

Task suite (~8, spanning the VBN gradient):
- **Already in archive** (reuse, n≥7 @2.5M): CheetahRun, WalkerRun, HopperHop (none+consistency).
- **Have partial**: AcrobotSwingup (none n=1 + config-matched + wmabl consistency), CartpoleSwingupSparse (consistency only → need none).
- **New both arms**: FingerSpin, QuadrupedWalk, ReacherHard (or Pendulum) — the redundant-end + a 2nd essential point.

New runs ≈ 4–5 tasks × 2 arms × 3 seeds ≈ 24–30 runs. At ~350k steps/hr on a 4×3060: 1.5M ≈ 4.3h/run,
4-way parallel → ~1–1.5 days. n=2 or 1.2M budget shortens it. VBN probes are cheap (checkpoint-time).

## Deploy-compute accounting (the headline number)

Per action: MPPI = `N·H·I` dynamics-model forward passes (default 512·3·6 ≈ 9216) + policy prior;
π = 1 policy forward pass. On the planner-redundant subset VSG uses π → ~10⁴× fewer model calls
for action selection there. Aggregate over the suite weighted by episode steps → "VSG recovers
full TD-MPC2 return at NN% of its deploy-time action compute."

## Deliverable

A results table (per-task 4-cell + VBN), the return-vs-deploy-FLOPs Pareto figure, the
VBN↔removability correlation (C2), and a method paragraph + algorithm box for the paper. If C2
holds, the paper's contribution becomes "diagnostic + probe-gated algorithm with a measured
efficiency win"; if C2 is weak, "a zero-regret planner gate + a diagnostic that predicts the
representation axis" — still a repaired algorithm, honestly scoped.
