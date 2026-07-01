# Deep-research: Planning as Directed Exploration + Abstraction-Defined Targets
Saved 2026-07-01. Two independent survey syntheses (Gemini, DeepSeek) + our synthesis and updated positive-chasing plan.
This is the literature backing Thread A (planning-as-exploration) and Threads D/E (SE for structure). It both
**explains our A-null** and identifies the **strongest unclaimed positive bet**.

## Method map (mechanism / abstraction / where it wins / id)
- **Plan2Explore** (2005.05960): ensemble one-step-model *disagreement* as expected novelty, maximized by an
  actor-critic planning IN latent imagination (RSSM). Prospective (not retrospective) novelty. Wins: task-agnostic
  unsupervised pretrain → zero/few-shot DMC. Isolates exploration via reward-free phase + held-out tasks.
- **LEXA** (2110.09514): explorer (max ensemble disagreement) + achiever (reach discovered latent states) dual
  policies in a shared model; **temporal-distance** goal metric (not cosine). Wins: multi-object manipulation
  (RoboKitchen), 40-task zero-shot goal reaching.
- **MAX / MEX**: info-gain exploration; MAX separates **epistemic vs aleatoric** (noise = confusion across all
  models, not conflict between them → avoids noisy-TV); builds an internal "exploration MDP". MEX = single
  objective fusing estimation+planning+exploration, sublinear-regret. Wins: AntMaze mapping, MuJoCo sparse.
- **Go-Explore / LGE** (ICML'23): "first return (no noise), then explore." Names the two failure modes:
  **detachment** (forget a frontier after moving on) + **derailment** (per-step action noise drifts you off the
  return path; prob of success decays exponentially with path length). Immune via state-load or goal-cond return
  + imitation-distill. LGE = latent, domain-knowledge-free. Wins: Montezuma/Pitfall, pure-exploration coverage.
- **Director**: HL manager selects a *decoded latent subgoal* every K steps (task+explore reward), LL worker
  reaches it, in Dreamer RSSM. Wins: sparse 3D quadruped mazes vanilla Dreamer can't enter.
- **Active-MCTS / EFE**: replace UCB with **Expected Free Energy** (pragmatic reward + epistemic info-gain) inside
  MCTS. UniZero/TransZero = Transformer dynamics (parallel subtree expansion, long horizon). SkyNet = belief-aware
  MuZero for POMDP. Wins: stochastic POMDPs, large discrete trees.
- **TD-MPC2 + SLOPE**: MPPI/CEM trajectory opt over a decoder-free latent + TD value. **SLOPE** = optimistic
  *distributional* reward regression (high-confidence upper bound) → synthetic dense slope pulling the planner to
  the goal under fully-sparse reward (arXiv:2310.16828 TD-MPC2; SLOPE arXiv:2602.03201). Wins: sparse continuous
  control, hard-to-reach dynamics.
- **BMPC**: fixes TD-MPC value-overestimation via **lazy reanalyze** imitation (tie the MPC expert to the policy
  prior so the value isn't evaluated on OOD planner actions).
- **ALPS** (arXiv:2602.05031): embed states in **graph-Laplacian** space (Euclid ≈ reachability), cluster →
  subgoals, **Dijkstra** HL over cluster graph + **CEM** LL. Wins: OGBench "giant mazes", heavy aliasing.
- **NEO** (arXiv:2510.14979): **source-sink Laplacian** — add non-neg diagonal weights to *frequently-visited*
  nodes → dominant eigenvectors' energy concentrates on *unvisited sink* nodes → smoothest eigenvectors give
  guaranteed gradient flow from known→novel. Directed exploration operator on asymmetric graphs.
- **SI2E / SIDM** (arXiv:2410.06621 / 2404.09760): **Structural Entropy** for RL exploration. SI2E = 2D-SE
  encoding tree over state-action graph + **Value-Conditional Structural Entropy (VCSE)**: reward transitions into
  high-value high-uncertainty communities, penalize low-value redundant ones. SIDM = directed-SE macro-states →
  topological **bottleneck** subgoals (doorways, grasp poses) discovered without priors. **This is prior art for
  "SE for exploration" — our SE-subgoal work must cite + differentiate.**
- **DIAYN / DADS**: skill discovery. DIAYN max I(S;Z) (diverse state coverage; but weak *directed* exploration in
  MiniGrid). DADS max I(S';Z|S) (predictable skills) → MPC directly over the latent skill space. Eigenoptions =
  Laplacian-eigenvector options driving to bottlenecks.

## Taxonomy of exploration difficulty (why planning *can* beat model-free)
sparse (gradient-free landscape) · deceptive (local-optimum trap) · hard-to-reach dynamics (execution bottleneck —
noise breaks the precise sequence) · no-reward/task-agnostic · combined. Planning helps by internal trajectory
optimization (solves derailment: optimize the whole sequence in latent, execute only the best) + memory/model
(solves detachment) + intrinsic objective in imagination. Model-free still wins on dense/short-horizon/real-time.

## THE KEY MECHANISM (why our A went null, and TD-MPC2 stalls on sparse)
**Value-overestimation from structural policy mismatch:** the MPPI planner proposes OOD exploratory action
sequences, but the TD value is bootstrapped on a *nominal policy prior* with a different distribution → the value
head evaluates OOD planner actions *optimistically* → planner chases phantom rewards → exploration stalls. AND
vanilla MPPI's objective IS predicted reward = the policy's objective, so with no novelty term it explores no
better than policy-only — **exactly our A1 result** (policy-only discovered CartpoleSwingupSparse 3/3, even
earlier). Fixes in the literature: BMPC (imitation-align expert↔prior), SLOPE (optimistic potential shaping).

## THE INTEGRATION GAP (both surveys converge — the strongest unclaimed hypothesis)
There is NO unified *online* framework that extracts structural graph-entropy / spectral embeddings **directly from
the continuous latent of a self-supervised world model** and uses them to **guide MPPI rollouts**. Spectral/SE
methods (ALPS, NEO, SI2E) are graph/offline; continuous latent planners (TD-MPC2) are near-sighted + overestimate
on sparse. **Gemini H1:** bias the MPPI action-proposal mean toward the smoothest eigenvector of a *latent
source-sink Laplacian* (NEO) built online from the world-model's episodic latent memory → aligns exploration with
topological frontiers → removes the policy-mismatch overestimation without arbitrary shaping. This is the union of
our A2 (novelty-MPPI) + E (SE-subgoals) + D (SE-as-structure).

## Falsifiable hypotheses (from the surveys)
- **G-H1 Latent spectral guidance in trajectory opt:** NEO source-sink-Laplacian-guided MPPI in TD-MPC2 reduces
  value-overestimation ≥40% + success ≥50% vs TD-MPC2 / TD-MPC2+SLOPE / Director on sparse locomotion. Metrics:
  |predicted-Q − MC-return|, return, coverage.
- **G-H2 VCSE as MCTS prior:** replace UCB with Value-Conditional-Structural-Entropy bonus in UniZero/TransZero →
  SOTA sample-eff on aliased procedural envs (MiniHack, DoorKey).
- **G-H3 DADS-guided Go-Explore:** DADS predictable skills for the "Go" (return) phase → kills derailment in
  non-restorable dexterous manip under domain randomization.
- **D-H1 Plan-to-bottleneck > plan-in-raw:** spectral-bottleneck subgoals + model-based plan-to-bottleneck beats
  Plan2Explore/LEXA/random-subgoal on coverage+success in maze/AntMaze (≥3/5 to hold).
- **D-H2 spectral-community targets > novelty intrinsic:** Fiedler-vector community boundaries give more
  structured (whole-community) exploration than RND/ICM (coverage entropy, #communities).
- **D-H3 hierarchical option-planning + abstract novelty bonus > flat planning** on long-horizon manip/locomotion.

## OUR SYNTHESIS + updated positive-chasing plan
1. **Our A-null is confirmed by theory, not a dead end.** Vanilla MPPI = policy objective → no exploration edge
   (as observed). The positive result requires an *explicit* exploration objective in planning.
2. **North-star positive bet = G-H1: online latent-SE/Laplacian-guided MPPI** (the integration gap). It unifies our
   four running experiments and is genuinely unclaimed for continuous latent world models. Build it as ROUND 2 by
   synthesizing whatever the round-1 agents produce.
3. **Measure the mechanism:** track value-overestimation (|predicted-Q − MC-return|) as the causal signal — a
   drop is the clean positive even if success is noisy.
4. **Baselines/related work to add:** SLOPE (optimistic shaping), Plan2Explore (ensemble disagreement), Director;
   cite SI2E/SIDM/ALPS/NEO and differentiate our angle = *online, from the continuous WM latent, guiding MPPI*.
5. **Round-1 (running now) → maps to components:** A2 novelty-MPPI (≈ Plan2Explore-in-TD-MPC2 disagreement term);
   E SE-subgoals (≈ SI2E/SIDM bottlenecks but online-latent); C new learned hierarchy (≈ Director); D SE-as-
   structure. Harvest → keep whichever separates → fuse into the G-H1 method.
6. **Task focus:** sparse + hard-to-reach where the gap is real (sparse swing-ups at weak actuation, AntMaze/
   giant-maze-style, hard-to-reach locomotion). Report REAL success + coverage + overestimation, matched budget.
