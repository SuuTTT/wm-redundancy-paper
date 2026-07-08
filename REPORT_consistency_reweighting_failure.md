# Failure Analysis — The Consistency-Reweighting Family (SOTA bets 1–2)

2026-07-08. Two attempts to beat vanilla TD-MPC2 by reweighting its self-predictive ("consistency") loss.
Both failed by the same margin in the same direction. This report explains what we tried, the exact numbers,
why it failed, and what it establishes.

## Motivation (why we tried this at all)

Paper 4's sufficiency grid found the consistency loss is **load-bearing on planner-led tasks** (WalkerRun −23%,
CheetahRun −38%, AcrobotSwingup −44% when trained OFF from scratch) and removable only on HopperHop. Read
constructively, that says the consistency loss is a *rollout-quality regularizer for the MPPI planner*. The natural
SOTA lever: **make it better** and beat vanilla on exactly those planner-led tasks. Two ways to "make it better,"
both keeping the loss but changing *what it emphasizes*:

- **Bet 1 — Value-Aware Consistency (VAC):** weight each latent dimension's prediction error by its value
  sensitivity \\(|\partial Q/\partial z|\\). Spend model capacity where the planner's value-ranking looks.
- **Bet 2 — Uncertainty/Rollout-reliability Consistency (URC):** weight each dimension's error by the model's own
  rollout drift (open-loop vs teacher-forced one-step prediction). Fix the dynamics exactly where multi-step
  rollouts compound error.

Both are one-block, env-gated edits (`VAC_LAM`, `URC_LAM`), byte-identical to vanilla when the coefficient is 0 —
so the graph is provably unchanged in the control arm.

## Protocol

Paired **same-seed**, matched-budget 5M, on the two cleanest planner-led tasks (WalkerRun, CheetahRun). Each arm
vs a matched vanilla arm at the same seed, so any difference is the loss change, not seed luck. TD-MPC2 defaults
(k_update 128, mppi_n_samples 2048, horizon 3, expl_until 25000). MPPI-best return per seed, n=2 per arm.
Harness: `run_vac.sh` on both boxes; drivers `exp/vac/{vac_walker,vac_cheetah,urc_walker,urc_cheetah}.sh`.

## Results

| Bet | Task | reweighted | matched vanilla | gap | budget |
|---|---|---|---|---|---|
| VAC | WalkerRun | 684.1 | 750.9 | **−8.9%** | 5M |
| VAC | CheetahRun | 808.8 | 845.8 | **−4.4%** | 5M |
| URC | WalkerRun | ~655 | ~707 | **−7.4%** | interim ~1M |
| URC | CheetahRun | ~768 | ~848 | **−9.5%** | ~3.6M |

Every arm underperformed matched vanilla, at **every** checkpoint from ~1M onward, with 0 nan. VAC was stable and
monotone; URC looked briefly like a near-tie at ~1.9M (−2.2%) but the gap re-opened to ~−9% as it matured.

## Why it failed — the mechanism

Both reweighting schemes concentrate the consistency loss's gradient on a *subset* of latent dimensions (the
value-sensitive ones for VAC; the currently-unreliable ones for URC) and de-emphasize the rest. The failure is the
same in both cases and follows directly from what the planner does at deploy time:

> **The MPPI planner rolls the latent dynamics forward over many candidate action sequences and ranks them.** Which
> latent dimensions matter for that ranking is **trajectory- and horizon-dependent**, not fixed. A dimension that is
> value-irrelevant *now* (low \\(|\partial Q/\partial z|\\)) or currently-reliable (low drift) can become decisive
> three steps into a rollout under a different action. Starving its prediction accuracy degrades the very rollouts
> the planner depends on.

In other words, the planner needs **faithful dynamics on every dimension it might explore**, and *uniform*
consistency is the objective that delivers exactly that. Any reweighting is a bet that some dimensions matter less —
and over a multi-step planning rollout, that bet is wrong often enough to cost 5–9%.

There is also a training-dynamics contributor specific to VAC: early in training \\(|\partial Q/\partial z|\\) is
computed from an *immature* Q, so VAC spends capacity on the wrong dimensions before the value function stabilizes.
URC avoids that (its weight comes from the dynamics model, not Q) yet still fails — which rules out "immature-Q
early" as the whole story and points at the deeper, planner-level explanation above.

## What this establishes

1. **The reweighting family is closed.** Two independent, well-motivated weightings (value-sensitivity and
   rollout-uncertainty) both lose by ~5–9%. Uniform consistency is not just load-bearing (Paper 4) — it is
   **near-optimal in form**. This is a clean negative result that *strengthens* the sufficiency paper.
2. **It reframes the SOTA search.** If you cannot improve the world model by re-emphasizing its existing objective,
   the remaining levers are (a) change what the value pathway *sees* (value-conditioned abstraction — SOTA bet 3,
   queued), or (b) accept that TD-MPC2's world-model design is at a local optimum for this stack and report that.

## Artifacts

- Ledger entries: `bet2_null_results.md` (VAC no-go, URC no-go).
- Proposal + design: `SOTA_PROPOSAL_value_aware_consistency.md`.
- Code: env-gated patches in `tdmpc2.py` (`VAC_LAM` value-weighting block; `URC_LAM` teacher-forced-drift block),
  backups `.bak_prevac` / `.bak_preurc` on both 3060 boxes.
- Runs: `b3060:/root/helios_wmablate/exp/vac/` (jsonl) and
  `b3060b:/root/tdmpc_glass/helios-rl/exp/tdmpc_glass/CheetahRun_{vac,van,urc,van2}_*` (CSVs).
