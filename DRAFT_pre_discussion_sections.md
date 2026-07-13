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
| AcrobotSwingup (n=4) | 261 (51%) | 271 (53%) | 282 (55%) | 397 (78%) | 511 | **step-at-128** — least compressible; tight widths flat, only D=128 recovers |

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
| WalkerRun (n=5) | median **739** | median **601** | **−18.7%** | load-bearing — higher but volatile regime |
| CheetahRun (n=5) | median **232.9** (finals span 117–585) | median **474.8** (tight ~515–544) | **+104% (stripped > full)** | **INVERSION** — the WM is *actively destabilizing* |

**The inversion, and why it matters.** On CheetahRun under planner-collection the full model does not
merely fail to beat stripped — it *destabilizes*: eval returns swing 141→585 within 250k steps across
every seed, while the stripped model sits tightly at ~460–555. We **pre-registered** (943819c) that
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
