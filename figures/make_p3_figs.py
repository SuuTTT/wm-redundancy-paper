#!/usr/bin/env python3
"""Paper 3 figures. Fig1: HopperHop wall curves (log-x, real seed curves from disk).
Fig2: mechanism 5-arm x 4-task grouped bars (MPPI-best per arm; numbers = ledger finals)."""
import json, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))

# ---------- Fig 1: wall curves ----------
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for i, f in enumerate(sorted(glob.glob(f"{D}/data/ppo_seed*.json"))):
    d = json.load(open(f))
    c = [(p["step"], p["reward"]) for p in d["curve"] if p["reward"] == p["reward"]]
    s, r = zip(*c)
    ax.plot(s, r, color="tab:red", alpha=0.7, lw=1.2,
            label="PPO (tuned)" if i == 0 else None)
for i, f in enumerate(sorted(glob.glob(f"{D}/data/sac20m_seed*.json"))):
    d = json.load(open(f))
    c = [(p["step"], p["reward"]) for p in d["curve"] if p["reward"] == p["reward"]]
    s, r = zip(*c)
    ax.plot(s, r, color="tab:orange", alpha=0.7, lw=1.2,
            label="SAC (tuned)" if i == 0 else None)
rows = [l.strip().split(",") for l in open(f"{D}/data/tdmpc2_s33.csv") if l.strip()][1:]
mppi = [(int(r[0]), float(r[1])) for r in rows if r[2] == "mppi"]
s, r = zip(*sorted(mppi))
ax.plot(s, r, color="tab:blue", lw=2.0, label="TD-MPC2 (seed 33)")
ax.axhline(200, color="gray", ls="--", lw=0.8)
ax.text(2e4, 210, "escape threshold", fontsize=7, color="gray")
ax.set_xscale("log")
ax.set_xlabel("environment steps (log scale)")
ax.set_ylabel("episode return")
ax.set_title("HopperHop: the on-policy exploration wall")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(f"{D}/fig_wall_curves.pdf")
print("fig1 ok")

# ---------- Fig 2: mechanism bars (MPPI best, mean over seeds; ledger finals) ----------
tasks = ["CheetahRun", "HopperHop", "WalkerRun", "HopperStand"]
arms = ["full", "-value", "-policy", "-reward", "-consistency"]
# mean of per-seed MPPI bests from the ledger (HopperHop -reward uses pi? no: MPPI ~0; shown as mppi)
vals = {
    "CheetahRun":  [np.mean([738,782,721,795]), np.mean([16,37,58]), np.mean([123,192,141]), np.mean([5,31,26]),  np.mean([367,516,558])],
    "HopperHop":   [np.mean([287,570,374,307]), np.mean([0,0,3.2,0,0,0.1]), 0.5, 0.5, np.mean([185,245])],
    "WalkerRun":   [np.mean([731,680,699,723]), np.mean([56,28,39,38]), np.mean([76,64,83,53]), 44, np.mean([547,533,483,674])],
    "HopperStand": [np.mean([937,926,946,911]), np.mean([7,13,9,6]), np.mean([34,18,9,20]), np.mean([270,301,542,265]), np.mean([898,816,821,818])],
}
fig2, ax2 = plt.subplots(figsize=(6.2, 3.2))
x = np.arange(len(tasks)); w = 0.16
colors = ["tab:blue", "tab:red", "tab:purple", "tab:green", "tab:gray"]
for j, arm in enumerate(arms):
    ax2.bar(x + (j - 2) * w, [vals[t][j] for t in tasks], w, label=arm, color=colors[j])
ax2.set_xticks(x); ax2.set_xticklabels(tasks, fontsize=9)
ax2.set_ylabel("best MPPI return (mean over seeds)")
ax2.set_title("Per-loss ablation: the value pathway is individually necessary")
ax2.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.15))
fig2.tight_layout()
fig2.savefig(f"{D}/fig_mechanism_bars.pdf", bbox_inches="tight")
print("fig2 ok")
