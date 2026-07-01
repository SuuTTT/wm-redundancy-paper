# HANDOFF — wm-redundancy-paper (results ledger + paper)
**2026-07-01 04:29 UTC.** Full master handoff (with GPU-box connection details, kept off public GitHub): `/home/ubuntu/HANDOFF_tdmpc-glass_2026-07-01.md` on the EC2 control box. Code/blog repo: `tdmpc-glass`.

## What this repo is
The **verified-results + paper** side of the TD-MPC-Glass abstraction-redundancy campaign. Key files:
- **`bet2_null_results.md`** — the authoritative, append-only VERIFIED-results ledger. Every claim here is disk-backed, multi-seed, with a matched-budget control. START HERE.
- **`SYNTHESIS_beat_ppo.md`** — the master verdict ("is PPO beatable, does abstraction help"). Has a 2026-06-30 addendum with the residual + anti-collapse taxonomy.
- **`AUTONOMOUS_BACKLOG.md`** — experiment queue + per-box state.
- `NEXT_PAPER_PROPOSAL.md` — forward directions.

## ⛔ Discipline (this project fabricated numbers ~7× in its history)
Read EVERY number from disk (deterministic real-success/return eval); always report n (seeds) + peak-vs-final; every "beats X" needs a **matched-budget** control (an under-budgeted PPO baseline is an over-claim — the "class-controller budget trap"). If a method fails, record the NULL honestly. Nulls are results here.

## Live experiment feeding this ledger (do not kill)
A **beat-PPO scan** is training on the two vast boxes (`b3060` TD-MPC2 / `b3060b` PPO) since ~03:00 UTC 2026-07-01. When it finishes, add a per-env matched-budget TD-MPC2-vs-PPO table to `bet2_null_results.md`, flagging any env where TD-MPC2 wins the ASYMPTOTE (a clean beat-PPO) vs sample-efficiency only. Harvest (from the EC2 box):
```
ssh b3060b '/root/tdmpc_glass/venv/bin/python /root/tdmpc_glass/exp/beat_ppo_scan/harvest.py ppo'
ssh b3060  '/root/helios-rl/.venv/bin/python /root/helios-rl/exp/beat_ppo_scan/harvest.py tdmpc'
```
⛔ `b3060b` is shared with another user's Mahjong RL (`moyuHarv` tmux) — never touch it.

## Current verified verdict (as of this handoff)
- **Panda:** H-JEPA solve 0.367 (n=7); analytic ceiling = contact physics (~0.37); learned residual breaks it (0.716 PickCube / 0.98 OpenCabinet) but **matched PPO wins the asymptote** (0.810 / 0.98) → prior = sample-efficiency (~1.6×/~7×), not ceiling (2 tasks).
- **Anti-collapse taxonomy:** relational/uniformity loss helps GEOMETRIC latents (point-maze 0.53→0.95; C-maze n=8 clean 0/8-vs-8/8 collapse-prevention, partial success), HURTS value-based control (uniformity worst 3/4 DMControl; CartpoleSwingup tie-exception; value-aware = worse = NULL); SE-community NULL either way.
- **Beat-PPO map:** we win the ceiling only on exploration-hard tasks (HopperHop 367-vs-33, sparse swing-ups); the live scan tests new dexterous envs for more.

## Next
Finish/record the scan → (if greenlit) pixel-JEPA-vs-Dreamer thread → assemble the conference paper (task "#27" in the tracker: in-loop abstraction + planning vs PPO; note the honest framing = speed-not-ceiling).
