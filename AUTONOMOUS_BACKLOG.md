# Autonomous GPU backlog — 8h unattended window (started 2026-06-30 ~16:00 UTC)

User is away ~8h; keep BOTH 3060 boxes on MEANINGFUL work (meaningful > idle > filler).
On each loop fire: (1) check both boxes, (2) harvest any completed agent → fold verdict into
`bet2_null_results.md` + blog, (3) if a box is IDLE, launch the next backlog item below.
HARD RULES: never touch mahjong (moyuHarv tmux / botzone / /root/mahjong); kill only own PIDs;
never --save_full_state; b3060>7G / b3060b>3G free; every number read from disk, report n, real-success.

## b3060b — Panda contact-primitive frontier (goal: raise oracle past 0.37 toward PPO 0.66)
1. [RUNNING agent a139886f] r6 different-contact oracle probe (grasp-lower / regrasp / firmer grip).
   - GO (oracle >~0.40) → run hjepa_options_solve_r5.py warm-started over r6 primitive; confirm planner extracts it.
   - NULL (≤~0.37) → next item.
2. 2nd contact variant: two-stage place (set down → release → regrasp/nudge to upright). Oracle-first.
3. LEARNED grasp-place residual: small RL residual on top of the analytic skill to fix in-grip slip
   (the genuine path past the analytic ceiling). Train short, eval real-success n>=256.
4. If all NULL: produce the definitive contact-physics ceiling figure (tilt-vs-success, oracle-vs-primitive).

## b3060 — anti-collapse taxonomy (goal: complete "match the term to the downstream" story)
1. [RUNNING agent a8ae1cc7] value-aware anti-collapse (valunif) on CheetahRun+WalkerWalk, arms {default,unif,valunif}.
   - GO (valunif beats unif AND ties/beats default on return-AUC) → extend valunif to FingerSpin + 1 more task.
   - NULL → still publishable (default hard to beat); record honestly, go to item 2.
2. Build the full anti-collapse taxonomy table: {term: none/uniformity/vicreg/valaware/SE} × {regime: geometric-nav / value-control},
   one clean figure. Reuse existing nav (se_nav_lever) + DMControl (unif_dmc) results; fill missing cells.
3. 4th DMControl task (e.g. HopperStand / CartpoleSwingup) to further firm uniformity-hurts-control (cheap, reuses exp_unif_run.sh).
4. Nav-side robustness: confirm uniformity GO holds on a 2nd geometric/goal-conditioned task.

## Harvest targets (read these for verdicts)
- b3060b: /root/tdmpc_glass/exp/hjepa_solve/round6_contact/VERDICT.md
- b3060:  /root/helios-rl/exp/unif_dmc/VALAWARE_VERDICT.md
- Ledger to update: /home/ubuntu/wm-redundancy-paper/bet2_null_results.md
- Blog: /home/ubuntu/tdmpc-glass/docs/_posts/ (Part 51 live; add Part 52 when a frontier resolves). Push only the new post file.
