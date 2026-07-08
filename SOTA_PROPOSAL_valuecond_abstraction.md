# SOTA bet 3 — Value-Conditioned Abstraction: force structure the value pathway must consume

Queued 2026-07-08 (user directive: after the consistency-reweighting family, take a REAL abstraction swing).
This is the direction Paper A left open: our whole program says *structure helps iff the value pathway consumes
it*. Bets 1–2 (VAC, URC) reweighted an existing loss and failed — they never changed what the value head sees.
This bet changes the **latent geometry itself** so that it is value-conditioned by construction, then asks whether
that beats vanilla TD-MPC2 on the planner-led tasks.

## Why this is the honest test of "abstraction"

Paper A's criterion: an abstraction is worth adding only if the new couplings carry gradient the **value pathway**
consumes. Every prior abstraction (SE/glass, entity/graph, uniformity/VICReg, hierarchy) failed that test —
the SimNorm latent is *already* value-sufficient (held-out value-decode \\(R^2 \approx 1\\)), so the extra
structure went unused. The one abstraction that is value-conditioned **by definition** is a **bisimulation
metric**: it pulls together states with equal reward and equivalent transitions, i.e. it shapes the latent so
distance = behavioral/value distance. If *even that* is redundant, it is the strongest possible confirmation of the
thesis. If it helps, it is the abstraction-SOTA result the program has been chasing.

## The primary bet (already implemented — no new code)

TD-MPC2's codebase already exposes `--bisim_coef` (default 0.0; wired through `loss_fn`, tdmpc2.py L535/L923,
run_benchmark.py L2402). Setting it > 0 adds a bisimulation-style term that regularizes the encoder latent toward a
value-conditioned metric — a genuine abstraction forced into the value pathway. So bet 3 is a *coefficient sweep +
head-to-head*, not a build.

- **Design:** paired same-seed, matched-budget 5M, planner-led tasks (WalkerRun + CheetahRun to start).
  Arms: `bisim_coef ∈ {0 (vanilla), 0.1, 0.5}` (light sweep; bisim can collapse latents if too strong).
- **H1 (main):** some bisim_coef > 0 beats vanilla on ≥2 planner-led tasks, non-overlapping or clearly-higher mean.
- **H2 (thesis-confirming null):** all bisim_coef > 0 tie-or-lose vanilla → the SimNorm latent is already
  value-sufficient; even an explicit value-conditioned metric adds nothing. This *closes* the abstraction question
  for this stack and is a citable capstone to Paper A.
- **Kill:** if bisim collapses (returns crater at coef 0.5), drop to 0.1/0.05; if both ≤ vanilla at n=2 by ~2M,
  call the null.

## Backup designs (if bisim is a degenerate no-op or collapses)

1. **Value-sufficient bottleneck.** Split the latent \\(z = [z_v, z_r]\\); force \\(Q\\) and \\(\pi\\) to read only
   \\(z_v\\) while dynamics/consistency use full \\(z\\). Train \\(z_v\\) to be value-sufficient (auxiliary
   value-decode). This is a learned value-relevant *sub-abstraction* the value pathway is architecturally forced to
   consume. ~1 patch (env-gated split + masked Q/π input).
2. **Value-predictive auxiliary head the Q-net must consume.** Add a small structured predictor whose output is
   concatenated into the Q input, trained to predict multi-step value — structure that only survives if it helps Q.

## Cost & protocol

Reuses the exact `run_vac.sh` paired-derisk harness (swap `VAC_LAM`/`URC_LAM` for `--bisim_coef`). One box-day for
the Walker+Cheetah derisk (2 arms × 2 tasks × 2 seeds at 5M, plus vanilla already on file). Scale to n=4 + a
HopperHop control only on a GO. Ledger every arm; no SOTA claim without a matched multi-seed beat.

## Recommendation

Run the **bisim_coef sweep** first — it is the most faithful "value-conditioned abstraction," already implemented,
and either outcome is publishable (a beat = abstraction-SOTA; a null = the definitive close of Paper A's open
question). Await user OK before launching (boxes are busy finishing the URC derisk).
