# Draft sections for pre-discussion (2026-07-14) — ready to lift into main.tex / paper_wall_mechanism.tex

All numbers verified from `bet2_null_results.md`; commit hashes inline. These are near-final prose
for the two sections the week produced. Integrate into the .tex by hand (kept out of the compiling
files deliberately).

---

## PAPER A — new section: "Two axes of task difficulty" (the value-sufficiency instrument)

**Framing.** The invalid probe (decode-\(R^2\)) saturates for strong *and* collapsed policies, so it
cannot tell whether an abstraction objective can help. We replace it with a *value-sufficiency
bottleneck* (VBN): insert a width-\(D\) bottleneck between the latent and the value head, and sweep
\(D\in\{16,32,64,128\}\) against the unmodified agent. The curve's *shape* is a per-task fingerprint of
how much of the latent the value function actually needs.

**Result — three qualitatively distinct fingerprints (final mppi eval, 5M steps):**

| Task | D=16 | D=32 | D=64 | D=128 | vanilla | fingerprint |
|---|---|---|---|---|---|---|
| CheetahRun (n=5) | 548 (64%) | 589 (69%) | 627 (73%) | 726 (85%) | 855 | **strictly monotone** — smooth information gradient; no width suffices |
| WalkerRun (n=5) | 625 (86%) | 643 (88%) | 666 (92%) | 694 (95%) | 727 | **flat-high** — most compressible; D=16 already 86% |
| AcrobotSwingup (n=6, medians) | 211 (41%) | 251 (49%) | 311 (61%) | 304 (59%) | 511 | **ramp-to-D64** — least compressible; climbs W16→W64 then saturates (D=128 adds nothing over D=64, ~60% ceiling) |

*Note on Acrobot (final n=6 {s52–s57}, ledger 888d3f4).* The grid is complete. The earlier "step-at-128"
was an artifact of two partial seeds (s50/s51 only reached ~2.7M) contaminating the means; on the six
complete-at-5M seeds the curve is a **ramp that saturates at D≈64** — W16 41% → W32 49% → W64 61%, with
D=128 (59%) adding nothing over D=64. The recovery point is D=64 and the ceiling is ~60% of vanilla.
Acrobot remains the least-compressible and noisiest task (two collapse cells: s54 W16=1.6, s55 W32=12.5);
report on medians (means dragged by the collapse tail).

**Load-bearing ordering agrees.** The stripped-vs-full sufficiency ablation (delete the consistency
loss, keep policy-collection, 5M) orders the tasks identically: HopperHop **0%** (n=8, removable) <
WalkerRun **−7.5%** (n=4) < CheetahRun **−23.8%** (n=4) < AcrobotSwingup **−44%**. The instrument (VBN
fingerprint) and the intervention (ablation) rank the four tasks the same way — the criterion is
"value-information compressibility," not horizon length. This *answers the Part-10 Q6* ("why did
abstraction only help HopperHop/nav?"): it helps exactly where the value function needs little of the
latent, i.e. where an added structural objective has room to reorganize a nearly-sufficient code.

**Positive artifact.** The VBN curve is checkpoint-cheap and predicts, per task/model, whether an
abstraction objective can help — the valid replacement for decode-\(R^2\). (Ledger: grid n=4 complete
3262862; Cheetah n=5 2d9da2f; Walker n=5 / dissociation c0c5830.)

---

## PAPER 3 — new section: "Collection mode and the world model: a double dissociation and an inversion"

**Setup.** Our JAX variant collects with the policy; the canonical agent collects with the planner
(MPPI rolls the world model to *act*, not only to score). We rerun the stripped-vs-full contrast under
**planner-collection** (MPPI_COLLECT=1, 512 samples, 2.5M) on three tasks. The planner-rolls-the-model
mechanism predicts collection mode should *modulate* the world model's value.

**Result — the interaction is task-dependent and non-monotone (finals @2.5M):**

| Task | full (none) | stripped (consistency) | Δ (median) | reading |
|---|---|---|---|---|
| HopperHop (n=5) | ~455 | ~468 | ≈0 | removable — both arms stable |
| WalkerRun (n=9) | median **739** | median **606** | **−18.0%** | load-bearing — higher but volatile regime |
| CheetahRun (n=9) | median **327** (finals span 117–585) | median **475** (tighter) | **+45% (stripped > full)** | **INVERSION** — the WM is *actively destabilizing* |

**The inversion, and why it matters.** On CheetahRun under planner-collection the full model does not
merely fail to beat stripped — it *destabilizes*: eval returns swing 141→585 within 250k steps across
every seed, while the stripped model sits tightly at ~460–555. The inversion magnitude settled from
+104% (n=5) to a stable **+45% at n=9** as more full-arm seeds came in — the *direction* never wavered.
We **pre-registered** (943819c) that
stripped would degrade ≥15% (kill <8%); it did the opposite. That falsification is the section's spine:
"planner-collection amplifies the world model's importance" is **Walker-specific, not a law.** The
collection-mode × world-model interaction is non-monotone in how much the value function needs the
latent (the VBN ordering).

**This also resolves the Part-10 Q8 tension** (HopperHop is both the sharpest PPO wall *and* the only
removable cell): removing the consistency loss removes the world model's *accuracy* but not the
*planner*. TD-MPC2 clears Hopper via planning + exploration-in-data (Q5: it is off-policy, deterministic
actor, no entropy term — the P1 entropy grid shows SAC fails Hopper 0/9 while the planner-free TD core
is 8/8), and the WM is removable because the planner carries it. The PPO wall itself is the
*conjunctive reward* (H3: margin-controlled PPO still walls 2.8/3.6 @20M), independent of the WM.

**Open mechanism (M1, pre-registered d7e7c49, running).** *Why* does the full model destabilize on
Cheetah but not Hop? Hypothesis: the full model's over-optimistic open-loop rollouts poison the MPPI
targets it collects with (a planner-target feedback loop); the stripped planner rolls an untrained net,
degrading to stable random-shooting. Probe: log the MPPI planned-return vs realized-return gap,
{Cheetah,Hop}×{full,stripped}. Verdict expected before the discussion.

(Ledger: V2/V2W double dissociation 7d9c9fb/852a580; n=5 + inversion b907130/c0c5830; M1 pre-reg
d7e7c49.)

---

## JEPA/SE thread — new section: "Anti-collapse regularizers are redundant on a DMControl collapse task" (#59)

**Setup.** The JEPA/SE line (Part-10 Q2/Q7) asks whether an explicit anti-collapse objective on the
latent buys generalization where the representation is prone to collapse. We test the two canonical
levers head-to-head on **CheetahRun** — the DMControl task our VBN instrument flags as *least
compressible / strictly-monotone* (i.e. the value function needs the most of the latent, the regime
most likely to collapse): **uniformity** (a hypersphere-uniformity penalty, `urc`) vs **VICReg**
(variance-covariance, `vac`), each against a matched vanilla baseline, 5M steps.

**Result (n=2, last-6 mppi-eval medians; refill to n=4 running):**

| arm | seed-median | vs vanilla (~818) |
|---|---|---|
| uniformity (urc) | 725.8 | −11.3% |
| VICReg (vac) | 751.3 | −8.2% |
| vanilla | ~818 | — |

**Reading — a NULL, and it is the *expected* null.** Uniformity ≈ VICReg (within seed noise), and
**both sit slightly below vanilla**. On this task the anti-collapse prior buys nothing — consistent
with the H-JEPA multi-seed NULL on PandaPickCube (#56) and the SE-structured-JEPA NULL (#57). The
unifying reason connects to Paper A's instrument: TD-MPC2's latent is *shaped by the value/TD
objective*, which already prevents the degenerate collapse these regularizers defend against; adding an
explicit anti-collapse term is therefore redundant (and mildly harmful via the extra loss term
competing with the value signal). This is the JEPA-line analogue of the paper's central claim —
*explicit structure is redundant exactly where the base objective already supplies it.* (Ledger
0a0a599; refill urc/vac s52 running, s53 queued → n=4.)
