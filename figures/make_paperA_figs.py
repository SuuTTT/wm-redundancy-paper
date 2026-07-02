#!/usr/bin/env python3
"""Paper A figures — generated strictly from evidence/*.json (no fabricated data).

Outputs (PDF, white background, Tol-bright colorblind-safe palette):
  figures/fig_r2_discrimination.pdf  <- evidence/vzprobe/*.json
  figures/fig_benchmark_forest.pdf   <- evidence/dmc_glass_vs_tdmpc2.json
  figures/fig_ksweep.pdf             <- evidence/p1_ksweep_harvest.json, p1_score.json,
                                        evidence/anchor_jumpy_vs_vanilla.json
"""
import json, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/ubuntu/wm-redundancy-paper"
EV = os.path.join(ROOT, "evidence")
FIG = os.path.join(ROOT, "figures")

BLUE, RED, GREEN, GRAY = "#4477AA", "#EE6677", "#228833", "#777777"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "pdf.fonttype": 42,
})

def load(p):
    with open(p) as f:
        return json.load(f)

# ------------------------------------------------------------------ figure (a)
# V(z)-decode vs return-to-go-decode held-out R^2 across checkpoint quality.
recs = [load(p) for p in sorted(glob.glob(os.path.join(EV, "vzprobe", "*.json")))]
recs.sort(key=lambda r: r["mean_return"])
ret = [r["mean_return"] for r in recs]
r2v = [r["r2_Vz_decode_heldout"] for r in recs]
r2g = [r["r2_returntogo_decode_heldout"] for r in recs]
print("fig(a) points:", list(zip(ret, r2v, r2g)))

fig, ax = plt.subplots(figsize=(4.4, 3.0))
ax.plot(ret, r2v, "-o", color=BLUE, lw=2, ms=6, zorder=3)
ax.plot(ret, r2g, "--s", color=RED, lw=2, ms=6, zorder=3)
ax.axhline(0.95, color=GRAY, lw=1, ls=":", zorder=1)
ax.text(680, 0.938, "C1 threshold ($R^2\\!=\\!0.95$)", color=GRAY,
        fontsize=7.5, ha="right", va="top")
# direct labels (identity not color-alone)
ax.annotate("decode $V(z)$\n(saturated by construction)", xy=(ret[3], r2v[3]),
            xytext=(400, 0.80), color=BLUE, fontsize=8, ha="center",
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
ax.annotate("decode MC return-to-go\n(variance-confounded)", xy=(ret[3], r2g[3]),
            xytext=(330, 0.17), color=RED, fontsize=8, ha="center",
            arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))
ax.set_xlabel("Checkpoint mean episode return (CheetahRun)")
ax.set_ylabel("Held-out linear-decode $R^2$")
ax.set_ylim(0, 1.05)
ax.set_xlim(-25, 690)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_r2_discrimination.pdf"))
plt.close(fig)

# ------------------------------------------------------------------ figure (b)
# 16-task DMC forest plot, glass vs tdmpc2, mean +/- 95% CI (n=4).
bench = load(os.path.join(EV, "dmc_glass_vs_tdmpc2.json"))["tasks"]
# same order as Table tab:bench in the paper
order = ["WalkerWalk", "WalkerStand", "WalkerRun", "CartpoleBalance",
         "CartpoleSwingup", "FingerSpin", "FingerTurnEasy", "FingerTurnHard",
         "AcrobotSwingup", "HopperHop", "BallInCup", "ReacherEasy",
         "CheetahRun", "ReacherHard", "HopperStand", "PendulumSwingup"]
assert set(order) == set(bench.keys()), sorted(bench.keys())

fig, ax = plt.subplots(figsize=(5.2, 5.4))
ys = range(len(order))
off = 0.18
for i, t in enumerate(order):
    g, m = bench[t]["tdmpc-glass"], bench[t]["tdmpc2"]
    assert g["n"] == 4 and m["n"] == 4
    ax.errorbar(m["mean"], i - off, xerr=m["ci95"], fmt="o", color=BLUE,
                ms=4.5, lw=1.6, capsize=2.5, zorder=3)
    ax.errorbar(g["mean"], i + off, xerr=g["ci95"], fmt="D", color=RED,
                ms=4, lw=1.6, capsize=2.5, zorder=3)
ax.set_yticks(list(ys))
ax.set_yticklabels(order)
ax.invert_yaxis()
ax.set_xlabel("Return at 1M environment steps (mean $\\pm$ 95% CI, $n{=}4$ seeds)")
ax.set_xlim(0, 1060)
ax.grid(axis="y", visible=False)
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([], [], color=BLUE, marker="o", ls="none", ms=5,
           label="tdmpc2 (monolith)"),
    Line2D([], [], color=RED, marker="D", ls="none", ms=4.5,
           label="tdmpc-glass (SE abstraction)")],
    loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_benchmark_forest.pdf"))
plt.close(fig)

# ------------------------------------------------------------------ figure (c)
# Temporal-predictor k-sweep: final return vs k per task, per-seed points,
# vanilla baseline, and the pre-registered prediction scores (4/4 vs 0/3).
harv = load(os.path.join(EV, "p1_ksweep_harvest.json"))
score = load(os.path.join(EV, "p1_score.json"))
anchor = load(os.path.join(EV, "anchor_jumpy_vs_vanilla.json"))["tasks"]

def seeds(cell):
    return [v for _, v in sorted(harv[cell]["per_seed"].items())]

def cell(name):
    return dict(vals=seeds(name), mean=harv[name]["final_mean"])

def acell(task, arm):
    a = anchor[task][arm]
    return dict(vals=a["final_per_seed"], mean=a["final_mean"])

tasks = [
    ("PandaPickCube", "Pick",
     dict(k2=cell("Pick_jumk2"), k4=cell("Pick_jum_k4_n8"),
          k8=cell("Pick_jumk8"), van=cell("Pick_van_n8"))),
    ("PandaPickCubeOrientation", "Ori",
     dict(k2=cell("Ori_jumk2"), k4=acell("PandaPickCubeOrientation", "jum"),
          k8=cell("Ori_jumk8"), van=acell("PandaPickCubeOrientation", "van"))),
    ("PandaOpenCabinet", "Cab",
     dict(k2=cell("Cab_jumk2"), k4=acell("PandaOpenCabinet", "jum"),
          k8=cell("Cab_jumk8"), van=acell("PandaOpenCabinet", "van")))]

fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.7), sharey=True)
xs = [0, 1, 2]
CHECK, CROSS = "✓", "✗"
for ax, (env, short, d) in zip(axes, tasks):
    van_mean = d["van"]["mean"]
    ax.axhline(van_mean, color=GRAY, lw=1.2, ls="--", zorder=1)
    means = []
    for x, kk in zip(xs, ["k2", "k4", "k8"]):
        means.append(d[kk]["mean"])
        vals = list(d[kk]["vals"])
        ax.plot([x] * len(vals), vals, "o", color=GRAY, ms=3, alpha=0.55,
                zorder=2)
    ax.plot(xs, means, "-o", color=BLUE, lw=2, ms=6, zorder=3)
    # verify plotted k2/k8 means match the scored file
    sb = score["k2_block"][short]; assert abs(means[0] - sb["k2_final"]) < 0.5
    sb = score["k8_block"][short]; assert abs(means[2] - sb["k8_final"]) < 0.5
    assert abs(means[1] - score["k2_block"][short]["k4_final"]) < 1.0
    # pre-registered prediction scores
    ax.text(0, 2870, f"{CHECK} pred. $k2{{<}}k4$", color=GREEN, fontsize=7.5,
            ha="center")
    ax.text(2, 2870, f"{CROSS} pred. $k8{{>}}k4$", color=RED, fontsize=7.5,
            ha="center")
    ax.set_xticks(xs)
    ax.set_xticklabels(["$k{=}2$", "$k{=}4$", "$k{=}8$"])
    ax.set_title(env, fontsize=8.5)
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(0, 3150)
    ax.text(2.38, van_mean + 60, "vanilla (1-step)", color=GRAY, fontsize=7,
            ha="right", va="bottom")
axes[0].set_ylabel("Final return")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_ksweep.pdf"))
plt.close(fig)
print("done:", sorted(os.path.basename(p) for p in glob.glob(FIG + "/fig_*.pdf")))
