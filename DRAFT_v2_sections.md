# SUFFICE — v2 draft sections (for review; LaTeX-flavored, drops into main.tex)

Reframe: single VBN diagnostic → **three-probe decomposition + SUFFICE elastic agent.**
Numbers marked ⟨…⟩ fill from the running n=3 sweep.

---

## 3. Three Probes for World-Model Utility

We decompose the benefit of a model-based agent's apparatus (world model + planner) over a plain
value-and-policy learner into **three channels**, each governed by a task property and estimated by a
cheap probe computed before or early in training. Let $M$ be the full agent, $M_{-\text{wm}}$ the agent
with the world-model (consistency/dynamics) objective removed, and let $\pi$ and $\Pi$ denote policy-only
and planner (MPPI) control.

**Representation.** Does the agent need the model to *represent* value? We estimate value
compressibility with the **Value-Bottleneck (VBN)** probe: freeze the trained encoder, insert a width-$D$
linear bottleneck before the value head, and measure the fraction of return recovered,
$r_{\text{VBN}}(D)=J(\text{bottleneck }D)/J(\text{full})$ at $D{=}16$. High $r_{\text{VBN}}$ ⇒ a few
value-relevant dimensions suffice ⇒ a reactive learner can acquire them ⇒ the model's representational
contribution is redundant.

**Exploration.** Does the agent need imagined rollouts to *discover* reward? We estimate reward
reachability under a random policy: over $N$ episodes, the reward density $\rho=\Pr[r_t>\tau]$ and the
median first-hit time. Low reachability (sparse, hard-to-stumble reward) ⇒ a reactive learner rarely
generates a positive example ⇒ the model earns its keep by *generating* rewarding experience — a benefit
VBN, a value probe on already-collected data, cannot see.

**Planning.** Does lookahead *act* better than the reflex policy? We measure the policy–planner gap
$g=J(\Pi)-J(\pi)$ on held-out evaluation. $g\!\approx\!0$ ⇒ the policy already matches the planner ⇒
planning is removable at deployment for a $\sim\!10^3$–$10^4\times$ reduction in per-action model calls.

The three channels are **orthogonal**: a task can need any subset. Ball-in-cup has maximal
$r_{\text{VBN}}$ (value trivially compressible) yet an essential model — because its deficit is
exploratory, flagged only by low reachability. Cheetah has dense reward and moderate compressibility yet
a large $g$ — a planning benefit neither other probe predicts.

**Scope (stated honestly).** *Representation* and *exploration* are properties of the task and its value
geometry, hence **world-model-agnostic**. *Planning* ($g$) is specific to agents that deploy an explicit
planner (e.g. TD-MPC2); for imagination-based agents (Dreamer) or JEPA-style predictors the analogue is
imagined-rollout vs. reactive value, which we report separately rather than conflate.

---

## 4. SUFFICE: Probe-Gated Elastic Control

The probes are control signals, not just a diagnostic. **SUFFICE** is a single agent, trained once, that
runs the probes on itself and gates its own apparatus, keeping the value pathway always on:

- **Planner gate (online, zero-regret).** During training we already compute $J(\pi)$ and $J(\Pi)$ each
  evaluation. Maintain an EMA of $g$; at deployment act with $\pi$ when $g\le\varepsilon$ (planner not
  helping beyond eval noise, $\varepsilon=1$ SEM), else with $\Pi$. This *cannot reduce return* — a
  component is dropped only where held-out evaluation shows it is not helping — and removes the dominant
  deploy cost on the planner-redundant subset.
- **World-model gate (a-priori).** Use $r_{\text{VBN}}$ on an early checkpoint to decide whether to pay
  for the consistency loss / world model at all: compressible ⇒ train the value pathway; incompressible
  ⇒ full model.

SUFFICE dominates both fixed policies: **always-full** wastes compute where the model is redundant;
**always-lean** fails where the model is load-bearing (e.g. acrobot, where planning through a stripped
latent collapses to ⟨~49⟩ vs. ⟨~385⟩ with the model).

---

## 5. Experiments

**Design — a component × world-model matrix.** For each task we run the full agent and the
world-model-stripped agent, each evaluated under $\pi$ and $\Pi$, giving a 4-cell table that isolates all
three channels: representation ($J_{\text{full}}(\pi)-J_{\text{strip}}(\pi)$), planning
($J_{\text{full}}(\Pi)-J_{\text{full}}(\pi)$), and (with the reachability probe) exploration.

**Tasks.** 8 DeepMind Control tasks spanning the compressibility gradient (locomotion, swing-up,
finger, cup, cartpole) + PandaPickCube (manipulation). $n{=}3$ seeds, config-matched.

**World-model families (agnosticism).** value-equivalent **TD-MPC2** (full 4-cell, $n{=}3$),
generative **DreamerV3** (strip ablation, 9-task representation gradient), and joint-embedding predictive
**H-JEPA** (predictor strip; ⟨0.53→0.53⟩ null on nav). Same decomposition, three architectures.

**Claims.**
- **C1 (diagnostic).** The three probes jointly predict which components each task needs; representation
  channel Spearman $\rho={-}0.90$ (value-limited set). Report the joint fit + residual.
- **C2 (algorithm, robust).** SUFFICE's planner gate matches full-agent return (zero-regret) while
  deploying $\pi$-only on the flagged subset — aggregate deploy-compute saving ⟨X%⟩; a return-vs-FLOPs
  Pareto with always-full, always-lean, and SUFFICE.
- **C3 (negatives).** The soft in-plan gate does not beat trusting the model (structural, 19-pt band) —
  the effective intervention is the hard route, not a soft down-weight.

**Honesty.** No completeness claim (Dyna-style augmentation, long-horizon credit assignment are
unmodeled channels); full seed ranges + the planning-channel scope caveat reported.
