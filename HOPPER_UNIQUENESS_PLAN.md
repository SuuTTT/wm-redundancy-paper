# What makes HopperHop unique? — investigation plan (Direction 1, Part-12 critique)

2026-07-09. Prioritized per user. Goal: mechanistically explain the two facts in tension —
**HopperHop is the sharpest PPO wall (0/5 ≥200 @472M) AND the only task where the world-model (consistency) loss
is removable (n=8).** Grounded in 4 papers the user supplied + our own measurements.

## The four candidate mechanisms (paper → claim → our evidence → decisive test)

### H1. Contact-critical value cliffs (Omura et al. 2024, "Stabilizing Extreme Q-learning"; Sujit et al. 2022, "Reducible Loss")
- **Claim:** Hopper is underactuated with **sharp, discontinuous cliffs in the value function** (a slightly-off action → fall → collapse of return). Omura uses Hopper as the *stability* stress test; Sujit notes the critical, high-information transitions are **liftoff/landing** (contacts), while mid-air joint variation is functionally noise.
- **Why it explains our facts:** on-policy PPO estimates advantages by Monte-Carlo/GAE returns, which are **high-variance across these cliffs** — the gradient can't climb. Off-policy **TD bootstrapping + replay** is far more robust to cliffs and *retains* the rare successful transition. → explains the PPO wall + why the TD value pathway (not the WM) carries the win.
- **Our evidence:** the wall needs *contact-criticality* — contact-free-but-unstable AcrobotSwingup has **no** PPO wall. Planning-without-WM adds nothing (mppi≈π on stripped-Hop), consistent with "critical info is at contacts, cheaply captured by the policy."
- **Decisive test (no new training):** measure **value-cliff sharpness** = variance / max-jump of the 1-step TD target across contact transitions, per task; correlate with the PPO wall depth across the 16-task benchmark. Prediction: wall depth ∝ cliffiness, HopperHop highest.

### H2. Narrow stability basin / underactuation (all four papers)
- **Claim:** the balance-maintaining region of policy space is **tiny** and surrounded by failure; success is rare.
- **Why it explains our facts:** on-policy exploration rarely samples the narrow success basin and discards it after one update; off-policy replay keeps it. Once found, the basin is a **low-dimensional limit cycle** → execution-simple → WM removable.
- **Decisive test:** **basin-width** = fraction of action-perturbations (ε-ball around the optimal policy) that keep the hopper upright for H steps, per task. Prediction: HopperHop narrowest; Cheetah/Walker wide.

### ENV RECON (2026-07-09, decisive for H3/H4) — HopperHop has NO early termination; the wall is a conjunctive reward
Read `mujoco_playground/_src/dm_control_suite/hopper.py`:
- `_STAND_HEIGHT = 0.6`, `_HOP_SPEED = 2.0`.
- **Reward = `standing × hopping`** (a *product*): `standing = tolerance(height, (0.6, 2))`,
  `hopping = tolerance(horiz_speed, bounds=(2.0, ∞), margin=1.0)`. You score only when **both** upright **and**
  moving ≥ 2 m/s.
- **`done = isnan(qpos) | isnan(qvel)` ONLY** — *no* early termination on falling; the episode always runs full length.
**Implication:** unlike Gym-Hopper, this wall is **NOT a termination artifact**. It is a **conjunctive, multiplicative,
sparse-in-practice reward** — reward stays ≈0 until the agent *simultaneously* solves standing and fast hopping.
That is precisely the exploration structure on-policy PPO (Monte-Carlo/GAE advantage) fails on and off-policy
TD+replay tolerates. This *refines* H3: the Voelcker knob for THIS env is the **reward structure**, not termination.
It also directly supports **H4** (exploration-hard): the barrier is finding the joint standing+hopping behavior.

### H3 (refined). Reward-design artifact (Voelcker et al. 2024, "Can we hop in general?")
- **Claim (the sharpest warning):** Hopper's **termination condition and reward scaling** are load-bearing — *changing the termination height can invert SOTA algorithm rankings.* Relying on Hopper blindly can yield flawed conclusions.
- **Why it matters to US:** our headline "categorical PPO wall on HopperHop" could be **partly a benchmark-design artifact** (instant termination on torso-drop → sparse, cliff-like return that murders on-policy PPO) rather than a pure algorithm-capability wall. This is the rigor check the critique *needs* before publication.
- **Decisive test (the priority experiment), now concrete + low-risk given the recon:** add two env-var knobs to
  `hopper.py` (default = byte-identical): **(i) `HOP_SPEED` override** (default 2.0 → test 1.0, 0.5 = easier hop
  threshold), and **(ii) reward mode** `product` (default) vs **`sum`** (`0.5*standing + 0.5*hopping`, a *denser,
  non-conjunctive* reward). Re-run tuned PPO + TD-MPC2 on 2–3 variants. **Predictions:** if PPO clears the wall under
  additive reward or lower HOP_SPEED → the wall is the **conjunctive/sparse-reward design** (exploration-hard;
  confirms H4, refines Voelcker), and our "categorical PPO wall" claim must be stated as *conditional on the standard
  DMC reward*, not as PPO-can-never-hop. If PPO still walls under the easier reward → the barrier is deeper (contact
  dynamics / H1-H2), strengthening the pure-algorithm-capability reading. Either outcome is publishable and is the
  rigor check the critique needs. **Patch is ~5 lines, env-gated** (`os.environ.get('HOP_SPEED')`, `HOP_REWARD_MODE`),
  backup + smoke, then PPO/TD-MPC2 runs — safe to do autonomously. **Runner note:** confirm the box has a PPO entry
  (`run_benchmark.py --algos ppo` or `ppo.py`); b3060/wm_head_ablation has `ppo.py`.

### H4. Exploration-hard-but-execution-simple (our standing hypothesis)
- **Claim:** the gait is hard to *find* (contact-critical exploration) but easy to *execute* (low-dim limit cycle needing no accurate multi-step rollout) → simultaneously high wall + removable WM.
- **Our evidence (strong):** WM removable n=8; planning-useless-without-WM at 5M (full mppi 571>π 542, stripped π 448≥mppi 421); stripped model still trains to full. π-only reaches ~full on Hop.
- **Remaining test:** k-step open-loop latent rollout error, stripped vs full, Hop vs dense — expect stripped-Hop error stays low (periodic, predictable) while stripped-Walker explodes.

## Synthesis (current best account)
HopperHop is unique because it sits at the **intersection of underactuation + contact-criticality + narrow margin**.
That intersection (i) makes it **exploration-hard for on-policy methods** (H1 cliffs + H2 narrow basin → the PPO wall),
while (ii) the resulting behavior is a **low-dimensional limit cycle that is execution-simple** (H4 → the world model
is removable). Our Part-12 critique — *the win is the off-policy TD value pathway, not planning-over-the-world-model* —
is the mechanistic consequence. **Voelcker (H3) is the caveat we must discharge**: how much of the "wall" is the
Hopper reward/termination design vs. genuine algorithm capability.

## What we already know (Q1 isolation, partly done)
- Full TD-MPC2 Hop: 6/6 ≥200 by ~1M; stripped-TD-MPC2 (WM off, keeps TD-value+policy+MPPI): ~420, removable.
- SAC Hop: 6/9 by ~8M (off-policy but slow). PPO Hop: 0/5 @472M (wall).
- ⇒ stripped-TD-MPC2 (no WM) already beats SAC's *speed* → **the win is the off-policy TD actor-critic + the planning
  operator, not the world model.** Planning-without-WM ≈ policy-only ⇒ on Hop, even the planning operator rides on
  the value/reward heads, not WM fidelity.
- **Missing piece:** a clean SAC-core swap (canonical SAC in place of TD-MPC2's actor-critic, keep WM+MPPI) to confirm
  the value pathway (not the specific update) is what matters. Cheap, optional.

## Experiment priority (launch when polish boxes free ~10–11:00)
1. **H3 termination/reward-robustness of the PPO wall** — *the* priority; discharges the Voelcker caveat, needs a
   HopperHop env-variant (termination height + reward-scale knobs) + re-run PPO & TD-MPC2. Highest rigor payoff.
2. **H1 value-cliff correlation** across the 16-task benchmark (no new training — analysis of logged TD targets /
   returns at contact transitions) — cheapest, direct mechanism evidence.
3. **H2 basin-width** measurement per task (short rollouts around the trained policy, ε-perturbation survival).
4. **H4 k-step rollout-error** stripped-vs-full (reuse the ablation harness).
5. *(optional)* SAC-core swap isolation.

Deliverable: a "What makes HopperHop unique" section for Paper 4 / a dedicated Part-14 blog, turning the descriptive
wall/removability tension into a mechanistic, literature-grounded account with the Voelcker robustness check done.
