# C_hier_new - LEARNED 2-level hierarchy vs flat (PRELIMINARY; sweep in flight)

**This file is preliminary.** The driver (`run_C_hier.sh`) overwrites it with the full
3-maze x 2-arm x 3-seed table via `agg_C_hier.py` when the sweep finishes (DONE marker).
It records the confirmed method + the seed-0 validation separation in case of early harvest.

## What was built (closes the prior hierarchy NULL)
Prior null: the "hierarchy" low level was an ANALYTIC phase controller; the *learned/generative*
H-JEPA hierarchy sat at 0.0. Here BOTH levels are LEARNED from scratch (TD3), no analytic controller:

- **flat**: single TD3 agent, env (sparse) reward only.
- **feudal (learned 2-level)**: a high-level TD3 emits a **relative subgoal** (HIRO-style local
  waypoint, +-2.5 units) every K=15 env-steps; a goal-conditioned low-level TD3 is rewarded by
  **dense progress toward that self-generated subgoal** (intrinsic, NOT privileged task info) and
  reaches it reliably; the HL is trained on the coarse timescale with the env's sparse goal reward.

Code: `/root/tdmpc_glass/exp/feudal_maze.py` (env + TD3 + both loops).

## Task: long-horizon SPARSE PointMaze, difficulty gradient (open rooms)
Open rooms isolate long-horizon sparse credit-assignment from wall-navigation confounds.
Sparse reward: +1 at goal (episode ends), else 0. REAL success = fraction of DETERMINISTIC
eval episodes reaching the TRUE goal (never shaped/intrinsic reward).

| maze | start->goal diagonal | horizon (max ep steps) |
|---|---|---|
| room    | 7.07 units  | 160 |
| midroom | 9.90 units  | 200 |
| bigroom | 12.73 units | 240 |

Mechanism the gradient probes: flat's per-step Gaussian exploration net-displacement is tiny
(action noise cancels), while feudal's HL commits to a direction for K steps -> ~5-10x larger
exploration reach. Past a distance threshold the goal is beyond flat's reach but within feudal's.

## CONFIRMED validation result (seed 0, from disk)
`logs/VAL_feudal_midroom.log`, `logs/VAL_flat_midroom.log`:

- **feudal midroom: success 1.000 at 20k env-steps (held at 40k).**
- **flat midroom: success 0.000 at 20k, 40k, 60k env-steps.**
- Calibration control (room, 7.07u): flat SOLVES it (success 1.000 by ~40k) -> the setup is
  NOT rigged; flat is a competent baseline that simply cannot bridge the longer horizon.

=> On a long-horizon sparse task where the low level IS learnable, a fully-learned 2-level
hierarchy BEATS a matched-budget flat baseline (100% vs 0% at 20k). This is the positive result;
the full sweep (3 seeds x 3 mazes, 400k matched) quantifies peak/std/steps-to-competence.

_n and full numbers: see summary.json / the finalized VERDICT.md after DONE._
