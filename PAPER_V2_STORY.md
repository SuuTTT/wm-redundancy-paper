# Paper v2 — story, method name, and abstract

## The upgraded story (one line)
World-model benefit is not one thing — it decomposes into **three separable channels**, each
predicted by a **cheap a-priori probe**; gating each component on its probe recovers full
performance at a fraction of the compute. This turns "when does a world model help?" from folklore
into a measurable, actionable decomposition — diagnostic **and** algorithm.

## The three channels and their probes
| channel | question | probe (cheap, before/early training) |
|---|---|---|
| **Representation** | need the model to *represent* value? | **VBN** — value compressibility (bottleneck sweep) |
| **Exploration** | need imagined rollouts to *discover* reward? | **reward reachability** — density / first-hit under a random policy |
| **Planning** | need lookahead to *act* well? | **π–MPPI gap** — policy-vs-planner eval difference |

Key: VBN alone mistakes ball-in-cup (easy value, rare reward) for "redundant." The exploration
probe fixes that outlier; the planning probe captures Cheetah (dense value, but planner worth +80%).
Three orthogonal axes, three probes — jointly they predict which components a task needs.

## Method name (recommendation + alternatives)
**Recommended: SUFFICE** — the method decides, per task and per component, whether the cheaper
option *suffices* (does the value pathway suffice? does π suffice without planning? does reactive
exploration suffice?). Real word, on-theme with value-**sufficiency**, memorable, positive.
- Alternatives: **TAILOR** (tailor the apparatus to the task), **Model-on-Demand (MoD)**,
  **Sufficiency Gating / VSG** (the sober descriptive name already in our design doc).
- Internal: the diagnostic suite = "the sufficiency probes"; the gating rule = "sufficiency gating."

## New abstract (draft — X%, N, and the correlation numbers fill from the running campaign)
> Model-based reinforcement learning bundles a learned world model and a planner that are expensive
> to train and deploy — yet on many tasks they add little over a plain value-and-policy agent, and
> practitioners cannot tell which case they face before paying the cost. We show that the benefit of
> this apparatus decomposes into three separable channels — *representation* (does the agent need the
> model to represent value?), *exploration* (does it need imagined rollouts to discover reward?), and
> *planning* (does it need lookahead to act?) — and that each is predicted by a cheap, a-priori probe:
> value compressibility via a value-bottleneck sweep, reward reachability under a random policy, and
> the policy-versus-planner evaluation gap. Across N DeepMind Control tasks these probes predict,
> before committing compute, which components a task actually needs — including exploration-limited
> tasks such as ball-in-cup that a value probe alone misclassifies as model-redundant. Building on
> this, **SUFFICE** gates each component on its probe: a zero-regret rule that retains only the
> machinery a task requires. SUFFICE matches full TD-MPC2's return while cutting deploy-time planning
> compute by X%, and dominates both the full agent (wasteful) and a lean value-only agent (which
> fails where the model is load-bearing). We turn "when does a world model help?" from folklore into a
> measurable decomposition — and into an algorithm that spends model-based compute only where it earns
> its keep.

## Can the 3 probes build a BETTER world model (not just route around one)? — the ambitious extension
Gating (SUFFICE) is the de-risked, buildable contribution (this campaign tests it). Beyond it:
- **Probe-shaped world model.** Use the probes *during* training to allocate the model's own effort:
  weight the consistency/dynamics loss by *local* value-incompressibility (model richly where value
  is hard to represent, cheaply where a thumbnail suffices) — value-equivalence taken to its logical
  end. Connects to `SOTA_PROPOSAL_value_aware_consistency.md`.
- **Exploration-gated imagination.** Spend the imagined-rollout budget only on high-sparsity tasks
  (the reward-reachability probe), free elsewhere.
- **Planning-gated MPPI (online).** The zero-regret π–MPPI gate, already in the codebase
  (`--controller_arbitration eval_only`).
- **One elastic agent.** Instead of separate policies, a single agent whose apparatus is gated online
  by the running probes — anytime/elastic model-based control.
Scope honestly: the probe-shaped model is *future work / preliminary* unless we run it; SUFFICE
(gating) is the paper's algorithmic result.

## Honesty guardrails (do not repeat the old R² over-claim)
- C1 (efficiency of the gate) is robust — zero-regret by construction.
- C2 (probes *predict* removability) is the empirical, at-risk claim — report the joint fit and the
  residual; never claim a *complete* predictor (Dyna data-augmentation and long-horizon credit
  assignment are plausible unmodeled channels).
