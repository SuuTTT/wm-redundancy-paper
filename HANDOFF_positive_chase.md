# HANDOFF — TD-MPC-Glass positive-chase campaign (for new session / Fable 5)

**Written 2026-07-02 by Opus 4.8 session, right before user switched sessions.**
This is a live autonomous campaign. The loop that drives it is **session-bound** (ScheduleWakeup) — it
died when the old session closed. **The GPU jobs keep running (detached nohup on the remote boxes); you must
re-arm the loop to harvest them.** See "RE-ARM THE LOOP" at the bottom.

## What this campaign is
Autonomously run the TD-MPC-Glass abstraction/world-model research program on two Vast.ai 3060 boxes
(`b3060`, `b3060b`), keep BOTH boxes' GPUs busy with *meaningful* work, publish honest results to the Hugo
blog (suuttt.github.io), **never fabricate numbers** (read from disk, report n, real success not shaped return).

## ⚡ PHASE COMPLETE (2026-07-02 evening) — final state
Part 9 published (2026-07-05-tdmpc-glass-part9-anatomy-of-beating-ppo.md) = consolidated final verdicts:
wall = ON-POLICY + gait-specific; SAC escapes (3/3≥200@5M); NO clean level gap (SAC-20M n=5 finals 246-572
straddle TD-MPC2 anchor ~477) → WM = consistency + ~4-5× efficiency ONLY; per-loss ablation: TD VALUE signal
load-bearing (its ablation reproduces the wall on HopperHop, n=4 ~0), self-predictive consistency LEAST critical;
hierarchy positive RE-ATTRIBUTED to dense shaping (shaped-flat 3/6 ≈ feudal 4/6; residual = self-generated signal).
Paper A submission-ready (author block = user). All review fixes published; blog ledger-consistent.
⚠ b3060 (instance 41649155) went OFFLINE ~17:00 (host-level outage, NOT ours; data on its disk incl. the
HopperHop ablation 10/20 arms). When it returns: relaunch remaining ablation arms (reward+policy s1/2;
consistency/reward/policy s3/4) via /root/helios_wmablate/run_arm_hop.sh pattern, harvest, fold corroborative
numbers into Part 9 pending note. Loop can wind down otherwise — b3060b queue exhausted (idle by design).

## Honest scorecard (as of original handoff)
- **WIN (reframe): the world model, not the planner, is the exploration lever.** PPO on `HopperHop` at ~472M
  steps (≈94× TD-MPC2's budget) peaks ~54, 0/5 seeds cross 200. Planning(MPPI)=pruning/exploitation;
  world-model=exploration. Blogged (Thread A 40c198d; **Part 8 c76037c**).
- **POSITIVE, localized:** learned feudal hierarchy vs flat — `fourroom` multi-room n=6: feudal 4/6, flat 0/6;
  open rooms within seed variance.
- **NULLs:** A2 novelty-MPPI (hurts dense), D SE-as-structure, E SE-subgoals, MiniGrid PPO+RND (0 success).
- Ledger: `/home/ubuntu/wm-redundancy-paper/bet2_null_results.md`. Deep-research backing:
  `/home/ubuntu/wm-redundancy-paper/PLANNING_EXPLORATION_deepresearch.md`.

## Published (blog = /home/ubuntu/blog, Hugo, git repo — push after commit)
- Part 7 (capstone, five bets resolved) — 8c309ea.
- **Part 8 "Chasing Positives" — c76037c** — the consolidated positive-chase post. **Has 2 pending
  confirmations flagged inside it — fold them in when they land (see below).**
- 5 living thread blogs: `content/projects/tdmpc-glass-thread-{a,b,c,d,e}-*.md`. Thread A/D current.
- COMMIT ONLY thread/blog `.md` files. Stray scratch files in blog root stay UNTRACKED.
- Commit msgs end: `Co-Authored-By: <the running model's name> <noreply@anthropic.com>` (Fable 5 → use Fable 5).

## RUNNING experiments — harvest these (verify DONE markers first; they may be done by now)
1. **WM-head ablation** (b3060 GPU2/3, `/root/helios_wmablate`, isolated copy — final arm `policy`; done
   consistency/none/reward/value). DONE=`/root/helios_wmablate/exp/wm_head_ablation/WM_ABLATION_DONE` → VERDICT.md
   (which of the ~5 nets is load-bearing). analyze.py handles partials.
2. **PPO-wall-generalization** — two halves:
   - PPO: b3060b GPU2/3, DONE=`/root/tdmpc_glass/exp/ppo_wall_generalization/PPO_SWEEP_DONE` (was 6/9 runs).
   - TD-MPC2: b3060 GPU0/1, DONE=`/root/helios-rl/exp/ppo_wall_generalization/TDMPC2_SWEEP_DONE`.
   - When BOTH: `scp b3060:/root/helios-rl/exp/ppo_wall_generalization/tdmpc2_fresh.json` → b3060b, rerun
     `analyze_ppowall.py`. Q: does the wall (PPO peak <80% of TD-MPC2) generalize to Pendulum/Finger/BallInCup?
     Early signal: PPO ~15–55 vs TD-MPC2 ~911 → leaning YES.
3. **TD-MPC2-HopperHop-coverage** (b3060b GPU0/1). DONE=`TDMPC2_HOP_COV_DONE` in
   `exp/tdmpc2_hopperhop_cov/logs/drive.log`.

**When wall-gen + WM-ablation land → update Part 8 (fold in the two confirmations) + record ledger + push.**

## STANDING CONSTRAINTS (verbatim, still in force)
- **Mahjong is off-limits.** On b3060b never touch `moyu*` tmux / `botzone` / `wrap_fable`. The g3090 box
  (instance **42163278**, RTX 3090 x4) is ENTIRELY mahjong — NEVER touch it. Verify mahjong untouched each fire.
- **NEVER broad `pkill -f run_benchmark`** — it kills A2/other arms. Kill by TAG or PID only.
- Don't modify `/root/helios-rl/.venv` while JAX jobs run; don't cross experiments.
- Never `--save_full_state`. Keep disk: b3060 >7G, b3060b >3G.
- Never destroy Vast instances (only recommend a destroy list; user destroys manually).
- Don't rely on agent self-watchers (they stall + spam). Use nohup + DONE markers; harvest from disk yourself.
  mkdir logdir BEFORE nohup redirect.
- Keep GPUs busy: idle+queue-exhausted → idle > filler; if stalled, investigate/relaunch.

## COMPLETE INVENTORY (for review)

### Thread map (the 5 research threads, living blogs in `content/projects/`)
- **A — planning-as-exploration** (`tdmpc-glass-thread-a-planning-exploration.md`): is TD-MPC2's edge
  exploration or sample-eff? → world-model is the exploration lever; planner=pruning. **The WIN lives here.**
- **B — behavioral-prior taxonomy** (`tdmpc-glass-thread-b-behavioral-prior-taxonomy.md`): when abstraction/
  class-controllers help.
- **C — abstraction as variance-reduction / hierarchy** (`tdmpc-glass-thread-c-abstraction-variance-reduction.md`):
  learned feudal hierarchy → localized positive on multi-room nav.
- **D — JEPA anti-collapse done right** (`tdmpc-glass-thread-d-jepa-anticollapse-done-right.md`): pure JEPA
  doesn't collapse on DMControl; anti-collapse neutral-to-harmful; BYOL predictor+EMA is the real lever. RESOLVED.
- **E — SE structure discovery** (`tdmpc-glass-thread-e-se-structure-discovery.md`): SE-community subgoals → null.

### Full blog part history (`content/projects/`, chronological — Hugo, suuttt.github.io)
phase1b (05-13) · iterations-2-7 (05-20) · iterations-8-9 (05-27) · part2 mechanism-check (06-09) ·
part3 campaign-review (06-17) · r2-criterion-postmortem (06-17) · part4 jumpy-to-beat-ppo (06-24) ·
part5 why-it-stays-in-the-loop (06-25) · part5 beat-ppo-reality-check (07-01) · part6 five-bets-next-phase (07-02) ·
part7 five-bets-resolved (07-03, 8c309ea) · **part8 chasing-positives (07-04, c76037c)**.

### Code repos (remote, via ssh)
- **b3060**: `/root/helios-rl` (main TD-MPC2/glass benchmark; `run_benchmark.py` — A1 pi-only flag at
  `_A1_PLAN`/`A1_COLLECT` env, gates ONLY collection action-selection, never the loss → both arms train full WM),
  `/root/helios_wmablate` (ISOLATED copy for WM-head ablation, `ABLATE` env flag), `/root/tdmpc_glass`,
  `/root/tdmpc_whyhopper_queue.txt`.
- **b3060b**: `/root/helios-rl`, `/root/tdmpc_glass` (has `exp/C_hier_new/feudal_maze.py` — takes NO `--gpu`
  flag, use CUDA_VISIBLE_DEVICES).

### Paper / analysis docs (`/home/ubuntu/wm-redundancy-paper/`)
- `bet2_null_results.md` — **authoritative results ledger** (every verdict, verified from disk).
- `PLANNING_EXPLORATION_deepresearch.md` — Gemini+DeepSeek deep-research: method map, exploration taxonomy,
  the integration gap (online latent-SE/Laplacian-guided MPPI = the north-star unclaimed bet), 6 hypotheses.
- `PAPER_A_STATUS.md`, `SYNTHESIS_beat_ppo.md`, `NEXT_PAPER_PROPOSAL.md`, `PROPOSALS_and_weekly_plan.md`,
  `AUTONOMOUS_BACKLOG.md`, `lecun_jepa_research.md`, `HANDOFF.md` (older), `README.md`.

### Local task list
~80 numbered campaign tasks tracked in the session task tool (mostly completed). Open/pending of note:
#27 conference-paper assembly, #34 live results page + cloudflare tunnel (in_progress), #59 uniformity-vs-VICReg
generalization. These are historical; the CURRENT active work is the positive-chase harvest above.

## FULL CAMPAIGN ARC (iteration 1 → now — for the whole-history review)
The tdmpc-glass program has run for ~2 months and pivoted several times. Chronology so it can be reviewed
end-to-end (blog parts in `content/projects/`, task #s in the session task list):

1. **Origin — "glass" = SE-structured latent for TD-MPC2** (iters 1–9, May; blogs phase1b, iters-2-7, iters-8-9).
   Hypothesis: structural-entropy / relational structure in the world-model latent improves control. Early
   iteration churn.
2. **Paper A — the redundancy criterion** (task #1; Part 2 mechanism-check 06-09; Part 3 review + R2-criterion
   postmortem 06-17). Central claim evolved into "on a value-anchored latent the right anti-collapse is *nothing
   extra*." The R2-criterion was postmortem'd (a claim that didn't hold — check how honestly it was retracted).
3. **Iters 32–36** (tasks #8–#12): calibration-shaped WM, jumpy-vs-vanilla @1.5M, GWM graph-as-simulator
   mechanism-check, GWM-B compositional-OOD reproduction + control. Entity/graph world-model thread.
4. **D1 contact-task WM** (tasks #13–#15): phase-binned k-step error mechanism-check.
5. **Panda beat-PPO campaign** (tasks #18–#26, the bulk; Parts 4–5): HL+residual policy, skill-options,
   demo-seeded TD-MPC2, shrink-Pareto (19-task), many beat-PPO attempts (warm-start, persistent-authority,
   closed-loop retry, hard-config curriculum, place-phase fine-control, tail characterization, orientation-aware
   grasp, learn-the-grasp, physical-vs-learnable ceiling, TAMP layer). Lots of near-ties vs PPO — review whether
   any "beat PPO" claim was over-stated vs same-budget control (see memory: class-controller-budget-trap).
6. **Beat-PPO on OpenCabinet** (tasks #28–#30; Part 5 "why it stays in the loop"): abstraction-as-curriculum
   (warm-start then RELEASE), leaner-bootstrap, fair dual-protocol + unified table. This is the strongest
   beat-PPO result — verify the dual-protocol is actually fair.
7. **Benchmark sweeps** (tasks #31–#41): tdmpc-glass vs tdmpc2 on TD-MPC2-favorable tasks, AcrobotSwingup, full
   16-task sweep, PPO column (16×3), Humanoid, manipulation methods (Paper B), class-controllers
   (swing-up/locomotion CPG/reaching OSC), TD-MPC2 Pareto (jumpy+efficient).
8. **Hierarchy + LeCun/JEPA thread** (tasks #43–#58): hierarchical speed-of-learning (LeCun hierarchy bet),
   JEPA research + next-paper proposal, speed-of-learning paper, Director baseline, H-JEPA faithful architecture
   (build → latent-MPPI planning → PandaPickCube NULL at n=5 → SE-structured JEPA → anti-collapse lever isolation).
   Much of this became Thread D's "pure JEPA doesn't collapse; BYOL asymmetry is the lever" reversal.
9. **The 5-bet reframe + positive-chase** (Parts 6–8; the 5 living thread blogs A–E). Part 6 posed five bets,
   Part 7 resolved them mostly-null, Part 8 chased positives → the world-model-is-exploration-lever WIN + the
   localized hierarchy positive. **This is the current active phase.**

Two recurring integrity risks to review across ALL of the above: (a) this project has fabricated/over-claimed
numbers ~7× historically (memory: nbeatsx-dc-verification discipline was adopted because of it) — every headline
number must trace to a JSON/log; (b) "beat PPO" claims need SAME-BUDGET vanilla-PPO controls (memory:
class-controller-budget-trap) — cross-budget comparisons are over-claims. Check old posts for un-retracted
versions of both.

## SSH
`ssh b3060` and `ssh b3060b` are configured. Filter vast.ai login banner noise by grepping it out.

## GPU QUEUE (2026-07-02 — what to launch when a slot frees; goal = publishable findings)
Papers: A (redundancy, finalizing — agent on figures/R²-excision), 2 (speed-of-learning, drafted), 3 (world-model-
is-exploration-lever — being decided by SAC control + WM-ablation-on-HopperHop + wall-gen).
- b3060 GPU0/1: **WM-head ablation on HopperHop RUNNING** (driver_hop.sh, exp/wm_head_ablation_hop/, arms
  none/value/consistency/reward/policy × seeds 1-2, DONE=WM_ABLATION_HOP_DONE) — Paper 3 mechanism on the flagship task.
- b3060 GPU2/3: CheetahRun WM-ablation finishing (policy arm). NEXT when free: HopperHop ablation seed 3 (n=3 for
  the impactful arms), or SAC seeds 4-5 (copy sac driver from b3060b) if SAC interim is interesting.
- b3060b GPU0/1: TD-MPC2-HopperHop-cov (~4M/5M). NEXT when free: nothing queued — prefer idle over filler unless a
  Paper-3 gap emerges (e.g. plan-vs-π on a 2nd exploration-hard task for the refutation's n).
- b3060b GPU2/3: PPO wall-gen BallInCup s1/s2 finishing → then SAC control (gated, auto-starts) + shaped-flat n=6.
Standing: meaningful > idle > filler; never disturb mahjong (moyu*/botzone/wrap_fable; g3090/42163278 NEVER).

## RE-ARM THE LOOP (do this to resume autonomy)
Run `/loop` with a prompt like below (adjust running-experiment state after your first harvest):

```
/loop AUTONOMOUS positive-chase (user away). Read /home/ubuntu/wm-redundancy-paper/HANDOFF_positive_chase.md
for full state. Harvest DONE markers (WM-ablation, PPO-wall-generalization PPO+TDMPC2, TD-MPC2-cov) → verify
REAL metrics from disk (report n, NEVER fabricate) → ledger + fold into Part 8 (c76037c) + push. Keep all GPUs
busy. Mahjong untouched (b3060b moyu*/botzone/wrap_fable; g3090/42163278 NEVER). Never broad pkill run_benchmark.
Re-arm ~1200-1500s.
```
