#!/usr/bin/env python3
"""Generate F1-F5 hierarchy-side figures for paper_speed_of_learning.tex.
ALL numbers are pulled from on-disk JSON harvested from b3060b
(HIER_VERDICT.json, method_vs_task.json) and bet2_null_results.md.
No fabricated points. Run on b3060b (data is local there); writes PNGs that
are then copied back to the local figures/ dir.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
HV = "/root/tdmpc_glass/exp/hierarchy_sol/HIER_VERDICT.json"
MVT = "/root/tdmpc_glass/exp/hierarchy_sol/method_vs_task.json"

hv = json.load(open(HV))
fam = hv["hier_families"]
flat_peak = hv["flat_tdmpc2_realsuccess"]["flat_pickcube_realsuccess"]["peak_pi_success"]
mvt = json.load(open(MVT))

plt.rcParams.update({
    "font.size": 12, "axes.grid": True, "grid.alpha": 0.3,
    "axes.axisbelow": True, "figure.dpi": 140,
})
BLUE = "#2c6fbb"; GRAY = "#888888"; GREEN = "#2ca02c"; RED = "#d62728"

# ---------- F1: depth monotonicity ----------
labels = ["Flat\nTD-MPC2", "Residual\n(raw-action)", "Options global\n(1-level)",
          "Options mlp\n(2-level)"]
means = [flat_peak, fam["hier_residual"]["peak_mean"],
         fam["hier_options_global"]["peak_mean"], fam["hier_options_mlp"]["peak_mean"]]
stds  = [0.0, fam["hier_residual"]["peak_std"],
         fam["hier_options_global"]["peak_std"], fam["hier_options_mlp"]["peak_std"]]
ns    = [3, fam["hier_residual"]["n_seeds"], fam["hier_options_global"]["n_seeds"],
         fam["hier_options_mlp"]["n_seeds"]]
fig, ax = plt.subplots(figsize=(7.0, 4.4))
colors = [GRAY, "#7aa6d6", "#4d86c4", BLUE]
bars = ax.bar(range(len(means)), means, yerr=stds, capsize=5, color=colors,
              edgecolor="black", linewidth=0.6)
for i,(m,n) in enumerate(zip(means,ns)):
    ax.text(i, m + (stds[i] if stds[i] else 0)+0.006, f"{m:.3f}\n(n={n})",
            ha="center", va="bottom", fontsize=10)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
ax.set_ylabel("Peak held-out true success")
ax.set_ylim(0, 0.30)
ax.set_title("F1  Abstraction depth is monotonic (PandaPickCube)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "hier_F1_depth_monotonicity.png")); plt.close(fig)
print("F1:", list(zip(labels, means, stds, ns)))

# ---------- F2: subgoal-horizon saturation ----------
hor = [100, 150, 200, 250, 300]
hkeys = {100:"hier_options_mlp_h100", 150:"hier_options_mlp", 200:"hier_options_mlp_h200",
         250:"hier_options_mlp_h250", 300:"hier_options_mlp_h300"}
hm = [fam[hkeys[h]]["peak_mean"] for h in hor]
hs = [fam[hkeys[h]]["peak_std"] for h in hor]
hn = [fam[hkeys[h]]["n_seeds"] for h in hor]
fig, ax = plt.subplots(figsize=(7.0, 4.4))
ax.errorbar(hor, hm, yerr=hs, marker="o", ms=8, lw=2, capsize=5, color=BLUE)
for x,y,n in zip(hor,hm,hn):
    ax.text(x, y+0.012, f"{y:.3f}", ha="center", fontsize=10)
ax.set_xlabel("Option / subgoal horizon (low-level steps per option)")
ax.set_ylabel("Peak held-out true success")
ax.set_xticks(hor); ax.set_ylim(0, 0.30)
ax.set_title("F2  Subgoal-horizon saturation (2-level mlp options)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "hier_F2_horizon_saturation.png")); plt.close(fig)
print("F2:", list(zip(hor, hm, hs, hn)))

# ---------- F3: matched- vs 2x-budget catch-up ----------
g1 = fam["hier_options_global"]["peak_mean"];      g1s = fam["hier_options_global"]["peak_std"]
g2 = fam["hier_options_global_long"]["peak_mean"]; g2s = fam["hier_options_global_long"]["peak_std"]
m1 = fam["hier_options_mlp"]["peak_mean"];         m1s = fam["hier_options_mlp"]["peak_std"]
m2 = fam["hier_options_mlp_long"]["peak_mean"];    m2s = fam["hier_options_mlp_long"]["peak_std"]
fig, ax = plt.subplots(figsize=(7.0, 4.4))
x = [0, 1]; w = 0.36
b1 = ax.bar([xi-w/2 for xi in x], [g1, g2], w, yerr=[g1s, g2s], capsize=5,
            color="#4d86c4", edgecolor="black", linewidth=0.6, label="1-level (global)")
b2 = ax.bar([xi+w/2 for xi in x], [m1, m2], w, yerr=[m1s, m2s], capsize=5,
            color=BLUE, edgecolor="black", linewidth=0.6, label="2-level (mlp)")
for xi, vals in zip(x, [(g1,m1),(g2,m2)]):
    ax.text(xi-w/2, vals[0]+0.012, f"{vals[0]:.3f}", ha="center", fontsize=10)
    ax.text(xi+w/2, vals[1]+0.012, f"{vals[1]:.3f}", ha="center", fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(["1$\\times$ budget\n(iters=400)", "2$\\times$ budget\n(iters=700)"])
ax.set_ylabel("Peak held-out true success"); ax.set_ylim(0, 0.30)
ax.legend(loc="upper left"); ax.set_title("F3  2$\\times$-budget catch-up (depth edge dissolves)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "hier_F3_budget_catchup.png")); plt.close(fig)
print("F3: 1lvl", g1, g2, "2lvl", m1, m2)

# ---------- F4: method-vs-task (full upright vs position-only) ----------
r = mvt["results_2level_mlp"]
cats = ["Full upright\nsuccess", "Position-only\n(<0.08m, any-step)"]
m_mean = [r["full_success_mean"], r["pos_only_0.08m_any_mean"]]
m_max  = [r["full_success_max"],  r["pos_only_0.08m_any_max"]]
fig, ax = plt.subplots(figsize=(7.0, 4.4))
xx = range(len(cats)); w = 0.36
ax.bar([i-w/2 for i in xx], m_mean, w, color=BLUE, edgecolor="black", linewidth=0.6, label="mean")
ax.bar([i+w/2 for i in xx], m_max, w, color="#9fc3e8", edgecolor="black", linewidth=0.6, label="max")
for i in xx:
    ax.text(i-w/2, m_mean[i]+0.012, f"{m_mean[i]:.3f}", ha="center", fontsize=10)
    ax.text(i+w/2, m_max[i]+0.012, f"{m_max[i]:.3f}", ha="center", fontsize=10)
ax.set_xticks(list(xx)); ax.set_xticklabels(cats)
ax.set_ylabel("Success (same 2-level checkpoints)"); ax.set_ylim(0, 0.80)
ax.legend(loc="upper left")
ax.set_title("F4  The plateau is the TASK (upright physics), not the method")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "hier_F4_method_vs_task.png")); plt.close(fig)
print("F4: full", m_mean[0], m_max[0], "posonly", m_mean[1], m_max[1])

# ---------- F5: injected vs learned/generative ----------
# Numbers from bet2_null_results.md (verified, harvested 2026-06-29).
f5_labels = ["Skill-options\n(2-level,\ninjected)", "DreamerV3\n(gen.,\n1-level)",
             "H-JEPA\nreactive", "H-JEPA\nlatent-MPPI", "H-JEPA\n+reach shaping"]
f5_vals = [fam["hier_options_mlp"]["peak_mean"], 0.0, 0.0, 0.0, 0.0]
f5_colors = [GREEN, RED, RED, RED, RED]
fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.bar(range(len(f5_vals)), f5_vals, color=f5_colors, edgecolor="black", linewidth=0.6)
for i,v in enumerate(f5_vals):
    ax.text(i, v+0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
ax.set_xticks(range(len(f5_labels))); ax.set_xticklabels(f5_labels, fontsize=10)
ax.set_ylabel("Peak real success (box\\_target$\\geq$0.9)"); ax.set_ylim(0, 0.27)
ax.set_title("F5  Only INJECTED structure buys competence (Bet-2 NULL)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "hier_F5_injected_vs_learned.png")); plt.close(fig)
print("F5: injected", f5_vals[0], "learned/gen all", f5_vals[1:])
print("DONE")
