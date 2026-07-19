# Daily Report — 2026-07-19

Campaign: **"When Does a Learned World Model Help?"** (AAAI submission)
Repo: `SuuTTT/wm-redundancy-paper` · Results log: `SuuTTT/tdmpc-glass` issue [#8](https://github.com/SuuTTT/tdmpc-glass/issues/8) · Ledger: [`bet2_null_results.md`](./bet2_null_results.md) · Paper: [`aaai_wm_diagnostic.tex`](./aaai_wm_diagnostic.tex) / [`.pdf`](./aaai_wm_diagnostic.pdf) (commit `22e8605`)

---

## 1. Code review (what was built, where, verified how)

| Component | Location | Status / verification |
|---|---|---|
| TD-MPC2 JAX reimpl. | `helios_wmablate/src/helios/algorithms/tdmpc2.py` (b3060) | **Non-anomalous baseline, NOT full parity** (corrected 07-19; see §7): hopper ≈0% (within official's own 373–594 seed spread), cheetah −5%, walker −17%, acrobot −23%. Known causes: π+noise collection (canonical = planner-collection) + MJX backend. Parity-fix runs launched. All paper claims are *within-implementation relative* comparisons, so unaffected; externally-referencing claims scoped to "our TD-MPC2 variant". |
| Strip-WM ablation (TD-MPC2) | `--consistency_coef 0` | used in planner-collection inversion runs |
| Strip-WM ablation (Dreamer) | `--agent.loss_scales.dyn 0.0` (dreamerv3, JAX) | kills RSSM forward-dynamics learning; van/strip pairs on both boxes |
| Gated-WM patch | `tdmpc2.py` L1031/L1104/L1130: `ret = _WM_GATE·Σγr_wm + γ^H·V`, `WM_GATE` env var | one `NameError` fixed via `__import__("os")` inline; verified 0 errors, 20+ seeds run |
| VBN probe | `tdmpc_glass/run_vbn.sh` (4070), width D∈{16,32,64,128} | 4-task fingerprint grid at n=3 (issue #3) |
| Harvest tooling | CSV col $2 last-6 median (mppi rows); Dreamer last-30 median of `episode/score` | one early bug (read col $4=seed) caught and root-caused; all published numbers use corrected pipeline |

Code hygiene notes: launches are `setsid nohup … & disown` (b3060 ssh hang workaround); `run_arm_v2.sh` silently no-ops on existing seed dirs (fresh-seed discipline enforced); b3060 disk pruned 95%→72% with only completed, already-harvested dirs deleted.

## 2. Experiments run (this campaign, all multi-seed)

1. **Parity** (#2): official TD-MPC2 vs our JAX — certified.
2. **VBN grid** (#3): 4 tasks × 4 widths × n=3 → fingerprints: cheetah *monotone*, walker *flat-high*, acrobot *ramp→D64*, hopper *flat-high/noisy*.
3. **Cross-model strip-WM (Dreamer)**, n=3 per task: cheetah, walker, acrobot.
4. **Collection-mode dissociation + inversion** (TD-MPC2, n=9): stripping WM **helps +45%** under planner-collection on cheetah; H-variance mechanism (planner-target variance ×3).
5. **Gated-WM sweep** g∈{0, .25, .5, 1.0}, **n=4–5 per gate** (~25 seeds total).
6. **Phase-1 breadth (started today)**: pendulum-swingup (predict *essential*) + finger-spin (predict *redundant*), Dreamer van/strip, running now.

## 3. Results (final numbers)

**Headline (positive): VBN predicts WM-dependence, cross-model, 3 tasks (n=3 each):**

| Task | VBN fingerprint | Dreamer van vs strip | Verdict |
|---|---|---|---|
| acrobot-swingup | ramp→64 | **412.5 ≫ 60.8** (+311…+1458%/seed) | WM **essential** |
| cheetah-run | monotone | 710 > 637 (+11.4%) | WM **helps** |
| walker-run | flat-high | 722 ≤ 776 (null) | WM **redundant** |

Same ordering holds in TD-MPC2. Honest caveat: acrobot *stripped* seeds are high-variance (26/56/100); vanilla is tight (408/420/409) — sign unshakable, magnitude seed-sensitive.

**Confirmed negative: the tunable gate does nothing.** n=4–5/gate means: g0.0=701.9, g0.5=688.4, g1.0=686.5, g0.25=682.9 — a **19-pt band** vs ~80–100-pt within-gate seed spread. The early n=1 "+5.6% for g=0.25" was a seed artifact.

**Mechanism:** under planner-collection the WM inflates planner-target variance ~3× → it can actively hurt where value is already compressible (the +45% inversion).

## 4. Discussion of your suggestions (from today)

**(a) "Only 3 DMC tasks — add more, and other envs (Panda, navigation, manipulation)?"**
Agreed — this is the paper's biggest weakness. But the fix is *quantification*, not just count: define a scalar VBN metric (e.g. saturation width D*), measure strip-Δ on ~10–12 tasks, report the correlation with CI. Phase 1 (DMC breadth, same infra, cheap) **started today**; Panda/nav are Phase 2 (cross-domain generality; nav is also the ideal testbed for the reward-horizon theory). Recommendation: gate the AAAI submission on Phase 1 only.

**(b) "How far can the a-priori test go — a performance-enhancing framework?"**
Two tiers. **Tier 1 (reachable, defensible): a *decision* framework** — the probe tells you whether to pay for a WM per task; that is a real compute/performance win and is what the paper can claim. **Tier 2 (future work, risky): an *adaptive* framework** — state-conditioned gating driven by local compressibility/H-variance. The global gate already failed (our n=5 negative), so Tier 2 is honest future work, not a claim.

**(c) "Cover all 3 paradigms (pure WM+plan = DINO; WM+RL+plan = TD-MPC2; WM+RL = Dreamer)? Time? Machines? Deadline?"**
TD-MPC2 + Dreamer: covered, VBN is native (both have value heads). **DINO-WM does not map directly** (image-based, offline, planning-only, no value head) — including it means re-plumbing the ablation + probing a reward/cost head instead: ~1.5–2 weeks infra+runs, high risk. **Timeline to AAAI (~mid-Aug, ≈3–4 wks):** safe plan = Phase-1 breadth (~3–4 days compute on current 8 GPUs, **no new machines**) + ~1 week writing → comfortably in time. Ambitious 3-paradigm × multi-domain plan needs 16–24 GPUs (rent 2–4 more boxes) and risks slipping. **Recommendation: don't rent yet; decide after Phase-1's correlation lands.**

## 5. Plan for next (in order)

1. **Phase-1 breadth (running):** harvest pendulum + finger; roll through queue: cartpole-swingup, cup-catch, reacher-hard, quadruped-walk (Dreamer van/strip n=1 first pass; seeds 2–3 for movers). Wire `run_vbn.sh` for the new tasks → VBN fingerprints.
2. **Quantify:** scalar VBN metric ↔ strip-Δ% correlation plot across ≥8 tasks → new headline figure for the paper.
3. **Paper integration:** add breadth table/figure to `aaai_wm_diagnostic.tex`; then optional §6 expansion (reward-propagation-horizon math) + real `aaai2026.sty`.
4. **Decision point (~2–3 days):** if correlation is strong → submit-track scope locked; then choose whether Panda/nav/DINO expansion (and renting) is worth it as camera-ready/follow-up work.
5. **Fleet:** 4070 + b3060 stay up doing breadth (no longer make-work). If you prefer to cut spend, both are disposable at any time — everything is on GitHub (git clean, 0 unpushed).

## 7. Corrections from user review (2026-07-19, post-report)

**(i) Parity claim corrected.** "Parity-certified" overstated it. Full V1 audit: hopper-hop **≈0% parity** (449 official [seeds 373/380/594] vs ~420±113 ours — the "581 vs 455" line compared one official seed to our mean), cheetah **−5%**, walker **−17%**, acrobot **−23%**. Documented causes: (1) our default collection is π+Gaussian-noise; canonical TD-MPC2 collects with the planner; (2) MJX physics vs dm_control MuJoCo numerics. **Fix attempt launched:** vanilla walker + acrobot under planner-collection (`MPPI_COLLECT=1`), 4M steps, b3060 G2/G3 (s201/s202) — if gaps close, collection mode was the cause. Paper impact: none of the paper's claims are absolute-performance claims; strip/gate/VBN are within-implementation comparisons, and the headline 3-task gradient rests on **unmodified official DreamerV3**. Wording in the paper will be scoped to "our TD-MPC2 variant".

**(ii) Gate negative is an *explained* null, not a mysterious one.** With H=3, γ=0.99, the MPPI score = g·Σγᵗr + γ³V decomposes as ≈2.97·r̄ (gated term) vs ≈97·r̄ (terminal value) → **the gated term is ~3% of the score**; sweeping g∈[0,1] perturbs the planner objective by ≤3%, so a null was structurally near-guaranteed. Moreover **g=0 ≠ WM-free planning**: V(z₃) is still evaluated on a state rolled through the learned dynamics, and value/policy were trained with the WM regardless — the gate removes imagined *reward*, not imagined *state*. Conceptually correct variants (future work, not claims): horizon-gating (H→0 = value-greedy planning), per-step uncertainty weighting (STEVE-style), or training-time gating. This analysis will be added to the paper's §4 so the negative reads as understood, not unexplained.

## 6. Status snapshot (as of this report)

- 4070 (44941373): pendulum_van/strip (G1/G2, training), VBN (G0/G3).
- b3060 (41649155): finger_van/strip (G0/G1, training), gated make-work (G2/G3); disk 74%.
- Backups verified: paper, ledger, #8, AUDIT_STATUS, blog all pushed.
