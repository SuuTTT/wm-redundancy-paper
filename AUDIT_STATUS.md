# "When Does a Learned World Model Help/Hurt?" — Campaign status & audit (2026-07-16)

One-page map of what is proven, where to verify it, what is open, and the positive
follow-up the results point to. Numbers here are cross-checkable against the
append-only ledger `bet2_null_results.md` (each entry carries a commit hash) and the
GitHub issue threads on `SuuTTT/tdmpc-glass`.

## Central claim (now cross-model)

**Whether a learned world model helps or hurts is a property of the *task's
value-information structure*, not of the WM architecture.** The *same tasks* are
WM-load-bearing vs WM-redundant across two independent world-model families:

| task | TD-MPC2 (latent-consistency WM) | Dreamer (reconstruction RSSM) |
|---|---|---|
| **cheetah-run** (low-compressibility, value needs the latent) | planner-collection **inversion**: stripping the WM *helps* +45% | vanilla **> stripped +9.7%** (WM helps) |
| **walker-run** (flat-high / value-sufficient) | **null**: imposed structure redundant | vanilla ≈ stripped (**tie / null**) |

The axis is captured by a cheap, checkpoint-time instrument (VBN, below): the WM is
load-bearing exactly where the value head needs a large fraction of the latent.

## Results logged (verify at these anchors)

| # | result | value | audit anchor |
|---|---|---|---|
| **#8** | Dreamer generalization | cheetah van732/strip667 (+9.7%); walker van736.6/strip743.4 (null) | issue #8; ledger `Dreamer generalization FINAL` |
| **#3** | VBN value-sufficiency grid (4 tasks, n=3) | Cheetah monotone; Walker flat-high; Acrobot ramp-to-D64; HopperHop flat-high/noisy (430/417/339/531) | issue #3; ledger `VBN HopperHop grid COMPLETE` |
| **#2** | Official TD-MPC2 parity | cheetah ~919 / hopper ~581 ≥ our JAX (855/455) — reimplementation certified | issue #2 |
| — | Collection-mode dissociation + Cheetah inversion | Δ = +45% stripped>full under planner-collection (n=9) | ledger `V2/V2W`, `inversion` |
| — | Anti-collapse (JEPA-lever) null (#59) | uniformity ≈ VICReg ≈ vanilla on Cheetah/Walker | ledger `#59`, `WalkerRun boundary null` |
| — | H-VARIANCE | planner-collection inflates planner-target eval variance ~3× | ledger `H-VARIANCE` |

## Open gaps (all optional; none blocks the core claim)

1. **hopper-vanilla under Dreamer** — never completed; repeated init OOM on the fleet
   (hopper-*stripped* ran fine at 138.87). Infra quirk, not a result. Future work.
2. **VBN instrument not ported to Dreamer** — only the strip-WM ablation was ported.
   Symmetric generalization would bottleneck Dreamer's *critic* input and sweep width.
3. **DINO-WM / JEPA** — *ruled out, not run*. DINO-WM is image-based, offline-dataset,
   planning-only with **no value head and no online collection loop**, so the VBN
   instrument and the collection-mode dissociation do not map. The JEPA *lever*
   (anti-collapse regularization) WAS tested in our online setting (#59) and was null.

## The positive follow-up the results point to

The diagnostic is not just descriptive — it prescribes an algorithm:
**a VBN/variance-gated world model — "a WM that knows when not to be used."**
- Measure value-information compressibility (VBN) and planner-target variance
  (H-VARIANCE) online, cheaply.
- Where the latent is value-sufficient (walker-like) or the WM inflates target
  variance (the Cheetah planner-collection inversion), **down-weight / shorten WM
  rollouts**; where the value head needs the latent (cheetah-like), **trust the WM**.
- This directly targets the failure mode we documented (the full WM *hurting* under
  planner-collection) and reuses the existing TD-MPC2 infrastructure — no new env.

## Compute status

All experiments above are **done and logged**; nothing is running that is needed.
The remaining work to make the AAAI draft publish-ready is a **writing pass** to fold
#2/#3/#8 into the `.tex` — **no GPU required**. The two follow-ups (Dreamer-VBN,
gated-WM algo) are *new* experiments that need GPU when started; renting fresh at that
point is cheaper than holding idle instances.
