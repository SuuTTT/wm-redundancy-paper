# HANDOFF 2026-07-07 — TD-MPC-Glass program (model-switch / session-restart handoff)

Written 2026-07-07 ~06:00 UTC while the 2×5 sufficiency grid finishes. Read this top-to-bottom before
touching anything. Companion memory files live in `~/.claude/projects/-home-ubuntu/memory/` (start with
`tdmpc-glass-resume.md`, `tdmpc-glass-positive-chase.md`).

## 0. What this program is

Matched-env benchmark (TD-MPC2 SimNorm+MPPI k_update=128, mppi_n_samples=2048, horizon 3, expl_until 25000
vs tuned brax PPO/SAC on byte-identical MJX `registry.load(env, impl=jax)`) → three papers:

- **Paper A** (redundancy criterion): draft complete (`main.tex`), needs author block only.
- **Paper 3** (`paper_wall_mechanism.tex`): the dissection capstone — PPO categorical wall (HopperHop 0/5 @472M,
  survives entropy ×3/×10), graded Stand barrier, Acrobot contact-criticality discriminator, 5-loss × 4-task
  mechanism ablation (value/policy fatal, reward planner-only, consistency mildest), humanoid matrix,
  anchor 420±113 (n=12). Compiles clean, 2 real-data figures. User-approved plan: consolidate around the
  mechanism thesis. Venue = user's call.
- **Paper 4** (sufficiency, constructive sequel): train consistency-OFF from scratch at 5M. **Grid so far:**

| Task | stripped finals (n=4) | full baseline | verdict |
|---|---|---|---|
| HopperHop | 165/475/481/511 | 420±113 (n=12) | **removable** (3/4 top-of-band) |
| WalkerRun | 537/574/554/594 (565) | 709/705/753/782 (737) | −23% |
| CheetahRun | 526.7/528.1/516.2/524.1 (524) | 903/904/782/806 (849) | −38% |
| AcrobotSwingup | 297.0/232.7/351.8/256.4 (284) | 533.3/511.2/512.9/488.1 (511) | **−44%** |
| CartpoleSwingupSparse | s41-44 IN FLIGHT b3060 (~13:30) | s1-4 IN FLIGHT b3060b (~07:30) | pending |

  **Current thesis (revised 07-07, after Acrobot broke the exploration/dense split):** the consistency loss
  underwrites MPPI rollout quality wherever the *planner carries learning*; HopperHop (pi-learnable) is the
  removable case. Cart-sparse is the first test of this revision. Also closed: novelty-MPPI matched-seed null
  (worse 4/4).

## 1. IMMEDIATE pending actions (the running pipeline)

1. **CART5M_DONE** (`/root/tdmpc_glass/exp/cart5m/CART5M_DONE` on b3060b, ETA ~07:30): harvest bests
   (`sort -t, -k2 -rn /root/tdmpc_glass/helios-rl/exp/tdmpc_glass/CartpoleSwingupSparse_cart5m_s<s>/seed_<s>.csv | head -1`)
   → ledger task-5 FULL baseline → commit+push → prune checkpoints (see §4).
2. **SUFFCART_DONE** (`/root/helios_wmablate/exp/SUFFCART_DONE` on b3060, ETA ~13:30): harvest jsonl
   `wmabl_consistency_s4{1-4}` (max mppi_return per file) → **2×5 GRAND VERDICT** vs cart5m baseline →
   ledger + Part 9 sufficiency table + Paper 3 → push all → prune.
3. After grid completion: grand summary; then dissection-paper restructure (mechanism + planner-led
   sufficiency thesis). GPUs: keep on meaningful cells (candidate: stripped-Hop s45-48 to firm the one-outlier
   removable cell) — but big NEW programs need user sign-off.

## 2. Machines & access

- `ssh b3060` = Vast 41649155, 4×3060, code at `/root/helios_wmablate` (NOT a git repo — snapshot in this
  repo under `code_snapshot/`). Venv: `/root/helios-rl/.venv/bin/python`. Disk floor >7G free.
- `ssh b3060b` = Vast 41721730, 4×3060, code at `/root/tdmpc_glass/helios-rl` (also not a repo). Venv
  `/root/tdmpc_glass/venv`. Disk floor >3G. **Shares box with the user's Mahjong project — see §5.**
- Run interface (b3060): `GPU=<i> ABLATE=consistency TASK=<Task> SEED=<s> TOTAL_STEPS=5000000 bash run_arm.sh`
  → jsonl at `exp/wm_head_ablation/jsonl/wmabl_<ablate>_s<seed>.jsonl` (~100 evals per 5M, read
  `es=` step counters from logs for progress). Drivers: suff5m/suffwalk/suffcheetah/suffacro/suffcart.sh.
- Run interface (b3060b): clone pattern of `exp/acro5m/a5m.sh` / `exp/cart5m/c5m.sh` (run_benchmark.py with
  the standard flags above, `TDMPC_GLASS_OUTPUT_TAG=<tag>_s<seed>`, CSVs at
  `helios-rl/exp/tdmpc_glass/<Task>_<tag>_s<seed>/seed_<seed>.csv`, DONE markers per exp dir).
- Launch pattern: write script, `chmod +x`, `setsid nohup bash <script> > log 2>&1 < /dev/null & disown`,
  then VERIFY in a separate ssh call (`pgrep -cf '[r]un_benchmark.*<Task>'`). Background-wrapper kill
  notifications are cosmetic once detached — but always verify.

## 3. Where everything lives (pushed 2026-07-07)

- **This repo** (github.com/SuuTTT/wm-redundancy-paper): papers (.tex/.pdf), `bet2_null_results.md` (the
  append-only LEDGER — single source of truth), `figures/` (make_p3_figs.py + data/), `evidence/`,
  `exp_results/b3060_jsonl/` (all 40 ablation jsonl), `exp_results/b3060b_csvs_20260707.tgz` (all seed CSVs),
  `code_snapshot/` (b3060 full code tar + b3060b drivers tar), status docs (PAPER_A_STATUS.md,
  NEXT_PAPER_PROPOSAL.md, lecun_jepa_research.md).
- **Weights**: GitHub release `artifacts-20260707` on SuuTTT/tdmpc-glass — stripped-Acrobot s35-38
  (best_mppi+final pkl). In-flight run weights (cart both arms) will be in each run dir's `checkpoints/`
  at completion — grab BEFORE pruning if wanted. **HF upload BLOCKED: no valid HF token anywhere (env
  HF_TOKEN invalid; boxes have none). Ask user to re-auth if HF mirror wanted** (`hf_backup.py` on b3060
  expects $HF_TOKEN; default repo Dannibal/tdmpc-glass-milestones).
- **Blog** (`/home/ubuntu/blog`, Hugo, push → suuttt.github.io): Part 9 capstone
  (`content/projects/2026-07-05-tdmpc-glass-part9-anatomy-of-beating-ppo.md`) + study/audit doc
  (`2026-07-06-tdmpc-glass-study-doc-three-papers.md`). Push may reject → `git -C /home/ubuntu/blog pull
  --rebase` first. Blog commits ONLY .md files.

## 4. Ops rules (hard-won; violate = data loss or user anger)

- **Prune after harvest**: each run silently writes ~280MB of pkl —
  `find <expdir> -type d -name checkpoints -mmin +60 -exec rm -rf {} +` (b3060: exp/tdmpc_glass; b3060b:
  helios-rl/exp/tdmpc_glass). b3060 once hit 94% disk from this.
- **Verify from disk, report n, never fabricate** — this project fabricated numbers ~7× historically.
  Ledger-first: append exact finals to `bet2_null_results.md` with n, then blog/paper.
- Never `--save_full_state`. Never destroy Vast instances (recommend only). `git -C <path>` always.
- Commits end: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- ETA reading: use `es=` step counters, not eval-row counts (eval cadence is ~50k, not 250k).
- awk over CSVs: skip header (`NR>1 && $2+0>=X`). `grep -c "^!"` exits 1 on zero matches (not an error).

## 5. DO-NOT-TOUCH (absolute)

- **b3060b mahjong**: never touch `moyu*`/`botzone`/`wrap_fable` tmux/procs. Note-only check each tick:
  `pgrep -cf '[b]otzone'` (expect ≥1). NO corrective action ever.
- **g3090 box (Vast 42163278) is ENTIRELY mahjong — never ssh-modify anything there.**
- `pkill` only with bracket-patterns (`'[p]attern'`) of OUR drivers, in isolated ssh calls.
- GPUs must run *meaningful* experiments; idle beats filler warmups.

## 6. Session-restart recipe (for the next model)

1. Read memory `tdmpc-glass-resume.md` + this file.
2. Check both markers (§1) — if fired while unattended, run the pending harvest/verdict steps exactly as
   written; the numbers in §0's table are already ledgered, only Cart cells are open.
3. Re-arm the session-bound autonomous loop (ScheduleWakeup ~1500-1700s) with the 2x5 PIPELINE prompt —
   the latest full prompt text is in this repo's git history / the ledger context, pattern: watch markers →
   harvest → ledger+blog+paper → push → prune → launch next cell → re-arm. Stop re-arming if user says stop.
4. User-side open decisions: Paper 3 venue; Paper A author block; whether to fund a new "powerful paper"
   program (world models + abstraction SOTA — see NEXT_PAPER_PROPOSAL.md) beyond the dissection paper.
