# CPG Locomotion Class — VERDICT (companion to swing-up energy-shaping)

User question: "can there be a uniform controller for the hopper/cheetah class?"
Mirror of the swing-up result (one energy-shaping controller served Pendulum 823 /
Cartpole / Acrobot, quality scaling with actuation).

## What was tested
ONE control paradigm — a parameterized central pattern generator (CPG): a global
phase oscillator -> per-actuator sinusoidal joint targets (tunable freq/amp/phase/
centers) + joint-space PD->normalized-torque + a trunk-pitch/height posture term.
The controller FUNCTION is identical across tasks; only a per-morphology parameter
dict changes (instantiated from each env's actuator layout). Per-task freq/amp/kp
were coarse-grid-tuned (the per-task *parameterization* of the shared controller).
Then a learned residual: u = clip(u_cpg(s,t) + 1.0*pi_res, -1, 1), CPG phase
(sin,cos) appended to obs (Markov in (s,phase)), brax PPO 10M, 3 seeds, true reward.

## RESULTS (all from disk; VERDICT.json. Protocol A, n=128, 3 seeds, peak return)

| Task        | CPG-alone (n64) | CPG+residual peak | tdmpc-glass | PPO  |
|-------------|-----------------|-------------------|-------------|------|
| CheetahRun  | 66.6 +/-0.9     | 778 +/-20  (WIN)  | 627 (peak)  | 293  |
| WalkerRun   | 18.4 +/-2.3     | 148 +/-34         | 669         | 67   |
| WalkerWalk  | 21.8 +/-1.9     | 313 +/-88         | 978         | 236  |
| HopperHop   | 0.1             | 1.5 +/-0.2        | 197 bimodal | 0.7  |
| HopperStand | 4.8 +/-1.6      | 50 +/-7           | 801 (peak)  | 5.9  |

(CheetahRun residual finals are unstable — s3 collapsed 763->9.9 late; peak is the
fair metric, as for the baselines whose finals also collapse, e.g. HopperStand.)

## VERDICT (honest)

(a) DOES ONE CPG PARADIGM PRODUCE LOCOMOTION ACROSS THE CLASS?
YES in form, NO uniformly in quality. The single CPG controller instantiates and
produces forward motion on every morphology, but controller-ALONE quality scales
sharply with how forgiving the morphology is:
  * Cheetah (reward = forward speed, NO balance constraint): 67 open-loop, a real
    gallop. The CPG paradigm fits this morphology well.
  * Walker (reward needs standing*move): ~20 — CPG moves it but cannot keep the
    trunk upright open-loop.
  * Hopper-Hop (resonant hopping): ~0; Hopper-Stand (posture hold): ~5.
Same law as the swing-up class (energy-shaping strong on Pendulum, weak on
Acrobot): the abstraction is uniform-within-class as a FORM, and its standalone
power scales with the morphology's actuation/constraint structure.

(b) DOES THE RESIDUAL LIFT TOWARD/ABOVE THE BASELINES, OR IS THE CPG AN ANCHOR?
Depends on prior quality — a clean monotone story:
  * STRONG prior (Cheetah): residual BEATS tdmpc-glass (778 > 627) on all 3 seeds
    and crushes PPO (778 vs 293). Prior+residual = SOTA-competitive. (Mirrors
    Pendulum: strong energy prior -> residual ties/beats the learned planner.)
  * WEAK prior (Walker Run/Walk): residual lifts 6-14x over the prior and BEATS PPO
    (148 vs 67; 313 vs 236), but is ANCHORED far below tdmpc-glass (669/978). The
    CPG prior caps the achievable return.
  * DEAD prior (Hopper-Hop): residual stuck at 1.5 (PPO 0.7, tdmpc-glass ~197). A
    near-useless prior gives no traction. (Mirrors Acrobot: weak prior = anchor.)
  * Hopper-Stand: residual 50 (PPO 6) — escapes-from-zero, but the static-posture
    prior is a poor scaffold for dynamic standing (baseline ~801).

(c) HOW DOES CONTROLLER-ALONE QUALITY SCALE ACROSS MORPHOLOGIES?
Monotone with morphology forgiveness / actuation authority:
  Cheetah (67) >> Walker (18-22) >> HopperStand (5) > HopperHop (0).
Exactly the Pendulum>Cartpole>Acrobot gradient of the swing-up class.

## JOINT CONCLUSION (with the swing-up result)
Abstraction scales BY CLASS, not per task: ONE controller form serves an entire
dynamical class (energy-shaping for swing-ups, CPG for locomotion), and within a
class its power — standalone and as a residual prior — scales with the morphology's
actuation/constraint structure. The residual is a LIFT when the prior is good
(Cheetah -> beats SOTA) and an ANCHOR when the prior is weak/dead (Hopper -> stuck),
with Walker the partial-recovery middle. PPO is beaten on every task
(escape-from-zero), so the class-prior is always useful vs from-scratch RL.

## Alpha sweep (PHASE 6, DONE): alpha in {0.5, 1.0, 2.0}, Cheetah/Walker/Hopper.
Authoritative Protocol-A (n=128) residual peaks vs authority:
  CheetahRun:  a0.5=510  a1.0=778(WIN vs 627)  a2.0=550
  WalkerRun:   a0.5=41   a1.0=148              a2.0=50
  HopperHop:   a0.5=0.9  a1.0=1.5              a2.0=1.2
=> alpha=1.0 is the OPTIMUM for every task; too little OR too much residual
   authority hurts (a0.5 under-uses the residual; a2.0 over-rides the prior and
   destabilizes PPO). The anchor verdict is ROBUST to authority: the residual is a
   genuine lift only on the strong-prior morphology (Cheetah), and the weak/dead
   priors (Walker partial, Hopper stuck) are NOT rescued by more authority.

   VERIFICATION NOTE (the project's known failure mode): the alpha-sweep TRAINING-
   LOG rewards transiently showed a2.0 ~900 (Cheetah) / ~790 (Walker) — these were
   stochastic-rollout artifacts that did NOT survive the deterministic Protocol-A
   re-eval (a2.0 = 550 / 50). All headline numbers are the Protocol-A re-evals from
   disk, NOT training-log peaks.

## Files (locomotion_class/)
cpg_locomotion_controller.py, validate_controller.py, residual_locomotion.py,
run_residual_locomotion.py, launch_one*.sh, eval_residual_curve.py, make_verdict.py,
orchestrate.sh, orchestrate_alpha.sh. Runs in logs/<task>_res_a<alpha>_s<seed>/;
NO --save_full_state. All numbers in VERDICT.json read from disk.
