# SOTA bet: Value-Aware Consistency (VAC) — a world-model loss that earns its keep

Started 2026-07-07. Program green-lit by user ("New SOTA paper"). Plan of record; update as data lands.

## The thesis (handed to us by our own findings)

Across three papers the throughline is identical: **a world-model term matters if and only if the value
pathway consumes it.** Paper A: SE/entity/graph structure is redundant because the value head ignores the
extra couplings. Paper 3: the value+policy losses are the engine; the consistency loss is the mildest cut.
Paper 4 (sufficiency, just closed): the consistency loss is load-bearing on **planner-led** tasks (Walker
−23%, Cheetah −38%, Acrobot −44%) and removable only on HopperHop, where the policy head learns directly.

Read constructively, that says: the consistency loss is a *rollout-quality regularizer for the planner*. So
the SOTA lever is not more structure — it is **spending the world model's capacity on the latent directions
the planner's value-ranking actually uses.** Uniform consistency wastes capacity fitting value-irrelevant
dimensions equally; a value-aware version should be *more* load-bearing → beat vanilla TD-MPC2 on exactly the
planner-led tasks.

## The method (VAC)

Standard TD-MPC2 consistency (per unroll step t, target stop-grad):
`cl = w · mean_B Σ_d (zŝ_{t+1,d} − z̄_{t+1,d})²`   — uniform over latent dims d.

VAC weights each latent dimension by its value sensitivity:
`g_d = |∂/∂z̄_{t+1,d} Σ_B minᵢ Qᵢ(z̄_{t+1}, π(z̄_{t+1}))|`  (stop-grad; target-net π,Q)
`ĝ = g / (mean_d g + ε)`  (per-sample normalize → mean weight 1, so total loss scale ≈ unchanged)
`cl_VAC = w · mean_B Σ_d ĝ_d (zŝ_{t+1,d} − z̄_{t+1,d})²`

Blend, env-gated (`VAC_LAM ∈ [0,1]`, default 0 → graph byte-identical to vanilla):
`cl = (1 − λ)·cl_uniform + λ·cl_VAC`.

Cost: one extra `jax.grad` of a scalar Q through the (stop-grad) target latent per unroll step — ~2× the
WM-loss backward, negligible vs MPPI. No new params.

## Hypotheses (pre-registered)

- **H1 (main):** VAC (λ≈1) > vanilla TD-MPC2 final return on ≥2 of {WalkerRun, CheetahRun, AcrobotSwingup},
  multi-seed, non-overlapping or clearly-higher mean. These are the tasks where consistency is load-bearing.
- **H2 (mechanism):** the gain tracks value-relevance concentration — VAC helps most where the value-sensitive
  latent subspace is low-dimensional (measure participation ratio of g).
- **H3 (null control):** on HopperHop (consistency removable) VAC is ≈ neutral — it can't help a loss that
  isn't load-bearing. A clean H3 null is a *feature*: it shows VAC acts through the rollout-quality channel,
  not as generic extra regularization.

## Baselines (from the sufficiency ledger, full-model n=4 unless noted)

WalkerRun 709/705/753/782 (mean 737) · CheetahRun 903/904/782/806 (849) · AcrobotSwingup 533/511/513/488 (511)
· HopperHop 420±113 (n=12). Same matched-env protocol (k_update 128, mppi_n_samples 2048, horizon 3,
expl_until 25000, 5M steps).

## Plan

1. Implement VAC as env-gated variant in tdmpc2.py (backup first; smoke: 1k steps, no nan, graph differs only
   when VAC_LAM>0).
2. **Derisk (cheap):** λ=1 vs vanilla on WalkerRun (b3060) + CheetahRun (b3060b), 2 seeds each, 5M. Early
   read at ~2M.
3. If signal → scale to n=4 on all three planner-led tasks + HopperHop H3 control + a λ sweep {0.5,1.0}.
4. If null → report honestly (uniform consistency already captures value-relevant dims; VAC = no-op), pivot to
   the next lever (candidate: value-aware *anti-collapse* replacing consistency entirely, or a learned
   value-relevance projection).

## Success / kill criteria

GO if H1 holds at n=4 (≥2 tasks, clear beat) with H3 null intact. KILL after step-3 if VAC ≤ vanilla on all
three planner-led tasks at n≥3. No "SOTA" claim without a matched multi-seed beat + the H3 mechanism control.
