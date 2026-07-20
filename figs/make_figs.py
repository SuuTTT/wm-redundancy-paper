#!/usr/bin/env python3
"""Generate the three data figures for the AAAI VBN diagnostic paper.
All numbers trace to bet2_null_results.md / issue #8. Output: vector PDFs."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# ---- palette (blue-biased scholarly, colourblind-safe) --------------------
C_ESS   = "#B23A2E"   # essential  (warm red)
C_HELP  = "#C77D28"   # helps      (amber)
C_MARG  = "#9A9A8C"   # marginal   (neutral)
C_RED   = "#2F6F8F"   # redundant  (slate blue)
C_LINE  = "#37567F"   # accent
INK     = "#171B22"

# ==========================================================================
# FIG 2 — VBN three-shape fingerprint (value-recovery vs bottleneck width)
#   Data: TD-MPC2 VBN grid, n=3 seeds. return at D in {16,32,64,128} vs vanilla.
# ==========================================================================
D = np.array([16, 32, 64, 128])
grid = {  # task: (returns at D16..D128, vanilla)
    "cheetah-run (monotone)":   ([517, 576, 624, 726], 855, C_HELP,  "o", "-"),
    "acrobot-swingup (ramp)":   ([258, 267, 271, 428], 511, C_ESS,   "s", "-"),
    "walker-run (flat-high)":   ([622, 647, 669, 701], 727, C_RED,   "^", "-"),
}
fig, ax = plt.subplots(figsize=(3.35, 2.7))
for name, (ys, van, col, mk, ls) in grid.items():
    rec = np.array(ys) / van
    ax.plot(D, rec, ls, color=col, marker=mk, ms=5, lw=1.6, label=name, zorder=3)
    ax.scatter([128*1.18], [1.0], color=col, marker=mk, s=26, zorder=3)  # vanilla anchor
ax.axhline(1.0, color="#999", lw=0.7, ls=(0, (3, 3)), zorder=1)
ax.text(20, 1.008, "vanilla (full latent)", fontsize=7.2, color="#666", va="bottom")
ax.set_xscale("log", base=2)
ax.set_xticks(D); ax.set_xticklabels([str(d) for d in D])
ax.set_xlim(14, 190)
ax.set_ylim(0.42, 1.06)
ax.set_xlabel(r"value bottleneck width $D$")
ax.set_ylabel("return recovered (frac. of vanilla)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="lower right", handlelength=1.6, borderpad=0.2)
ax.set_title("VBN value-recovery curve = task fingerprint", fontsize=9.2, pad=6)
fig.savefig("fig_fingerprint.pdf")
plt.close(fig)
print("fig_fingerprint.pdf written")

# ==========================================================================
# FIG 3 — 8-task WM-dependence gradient (Dreamer strip ablation)
#   WM-dependence = (vanilla - stripped)/vanilla  (fraction of return lost)
# ==========================================================================
tasks = [  # name, vanilla, stripped, regime-colour, label
    ("pendulum-swingup", 806.0,  0.0,  C_ESS),
    ("ball-in-cup catch", 972.0, 0.0,  C_ESS),
    ("reacher-hard",     965.0, 18.0,  C_ESS),
    ("acrobot-swingup",  412.5, 60.8,  C_ESS),
    ("cheetah-run",      710.0, 637.0, C_HELP),
    ("cartpole-swingup", 867.5, 845.7, C_MARG),
    ("finger-spin",      662.0, 695.0, C_RED),
    ("walker-run",       722.0, 776.0, C_RED),
]
dep = [ (v - s) / v for (_, v, s, _) in tasks ]
order = np.argsort(dep)                    # ascending: redundant -> essential
names = [tasks[i][0] for i in order]
vals  = [dep[i] for i in order]
cols  = [tasks[i][3] for i in order]

fig, ax = plt.subplots(figsize=(3.5, 3.0))
y = np.arange(len(names))
ax.barh(y, vals, color=cols, height=0.68, zorder=3, edgecolor="white", lw=0.6)
ax.axvline(0, color=INK, lw=0.9, zorder=4)
for yi, v in zip(y, vals):
    off = 0.02 if v >= 0 else -0.02
    ha = "left" if v >= 0 else "right"
    lab = "collapse" if v > 0.97 else f"{v:+.0%}"
    ax.text(v + off, yi, lab, va="center", ha=ha, fontsize=7.4, color=INK)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8.4)
ax.set_xlim(-0.22, 1.16)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
ax.set_xlabel("WM-dependence  (return lost when WM stripped)")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
# regime brackets
ax.text(1.13, 6.5, "essential", rotation=90, va="center", ha="center",
        fontsize=7.6, color=C_ESS, weight="bold")
ax.text(1.13, 0.5, "redundant", rotation=90, va="center", ha="center",
        fontsize=7.6, color=C_RED, weight="bold")
ax.set_title("Eight-task WM-dependence gradient (Dreamer)", fontsize=9.2, pad=6)
fig.savefig("fig_gradient.pdf")
plt.close(fig)
print("fig_gradient.pdf written")

# ==========================================================================
# FIG 4 — HEADLINE: VBN compressibility predicts WM-dependence
#   x = VBN D=16 value-recovery (frac of vanilla)  [TD-MPC2 probe]
#   y = WM-dependence (frac return lost when stripped)  [Dreamer ablation]
# ==========================================================================
pts = [  # task, VBN D16 recovery (frac of vanilla), WM-dependence (frac lost), colour
    ("pendulum", 332.7/766.0,  1.000, C_ESS),
    ("acrobot",  258.0/511.0,  0.853, C_ESS),
    ("cheetah",  517.0/855.0,  0.103, C_HELP),
    ("walker",   622.0/727.0, -0.075, C_RED),
    ("finger",   961.2/980.0, -0.050, C_RED),
]
x = np.array([p[1] for p in pts], dtype=float)
yv = np.array([p[2] for p in pts], dtype=float)
def _rank(a):
    order = np.argsort(a); rk = np.empty_like(order, dtype=float); rk[order] = np.arange(len(a)); return rk
rho = np.corrcoef(_rank(x), _rank(yv))[0, 1]
r = np.corrcoef(x, yv)[0, 1]
prho = pr = float("nan")
# OLS trend
b, a = np.polyfit(x, yv, 1)
xs = np.linspace(0.38, 1.02, 50)

fig, ax = plt.subplots(figsize=(3.5, 3.0))
ax.plot(xs, a + b * xs, color=INK, lw=1.1, ls=(0, (4, 3)), zorder=2)
ax.axhline(0, color="#BBB", lw=0.7, zorder=1)
for name, xi, yi, col in pts:
    ax.scatter([xi], [yi], color=col, s=64, zorder=4, edgecolor="white", lw=0.8)
    dx = 0.012
    ha = "left"
    if name in ("finger", "walker"):
        ha, dx = "right", -0.012
    ax.annotate(name, (xi, yi), xytext=(xi + dx, yi + 0.045),
                fontsize=7.8, color=col, ha=ha, weight="bold")
ax.set_xlim(0.37, 1.03)
ax.set_ylim(-0.2, 1.12)
ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xticklabels(["40%", "50%", "60%", "70%", "80%", "90%", "100%"])
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
ax.set_xlabel(r"value compressibility: VBN $D{=}16$ recovery")
ax.set_ylabel("WM-dependence (return lost)")
ax.spines[["top", "right"]].set_visible(False)
ax.text(0.40, 1.05, f"Spearman $\\rho = {rho:.2f}$\nPearson $r = {r:.2f}$",
        fontsize=8.2, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="#EEF2F8", ec="#C6D2E4", lw=0.6))
ax.text(0.985, 1.02, "essential", fontsize=7.4, color=C_ESS, ha="right", va="top", weight="bold")
ax.text(0.985, -0.13, "redundant", fontsize=7.4, color=C_RED, ha="right", va="bottom", weight="bold")
ax.set_title("Compressible value $\\Rightarrow$ redundant world model", fontsize=9.2, pad=6)
fig.savefig("fig_correlation.pdf")
plt.close(fig)
print(f"fig_correlation.pdf written  (rho={rho:.3f} p={prho:.3g}; r={r:.3f} p={pr:.3g})")
