"""Aggregate per-run JSONs -> VERDICT.md + summary.json. Re-runnable for partial harvest."""
import json, sys, os, glob
from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
runs = []
for p in sorted(glob.glob(os.path.join(ROOT, "runs", "*.json"))):
    try:
        runs.append(json.load(open(p)))
    except Exception:
        pass

# group by (task, arm) -> list over seeds
g = defaultdict(list)
tasks = []
arms = []
for r in runs:
    g[(r["task"], r["arm"])].append(r)
    if r["task"] not in tasks:
        tasks.append(r["task"])
    if r["arm"] not in arms:
        arms.append(r["arm"])

# preferred arm order
arm_order = [a for a in ["vanilla", "rnd_b0.5", "rnd_b1.0", "dis_b0.5"] if a in arms] + \
            [a for a in arms if a not in ["vanilla", "rnd_b0.5", "rnd_b1.0", "dis_b0.5"]]

def agg(task, arm, field):
    vals = [x[field] for x in g.get((task, arm), []) if x.get(field) is not None]
    return vals

def mean(v):
    return sum(v) / len(v) if v else None

summary = {"tasks": {}, "n_runs": len(runs)}
lines = []
lines.append("# A2 Novelty-MPPI — VERDICT")
lines.append("")
lines.append("Planning-as-directed-exploration: novelty term added to the MPPI trajectory objective")
lines.append("(score = predicted return + beta * z-scored novelty). novelty = RND predictor error over")
lines.append("the SimNorm latent (rnd), or 2-Q-head value spread (disagreement). Metric = REAL sparse")
lines.append("eval return via MPPI planner (sparse DMC reward is the true task signal, not shaped).")
lines.append("n = seeds per cell. Every number read from per-run JSON (exp/A2_novelty_mppi/runs/).")
lines.append("")
lines.append("## Per-task: best MPPI eval return (mean over seeds) [max over seeds]")
lines.append("")
header = "| task | " + " | ".join(arm_order) + " | novelty wins? |"
sep = "|" + "---|" * (len(arm_order) + 2)
lines.append(header)
lines.append(sep)

wins = []
for task in tasks:
    van_best = agg(task, "vanilla", "best_mppi")
    van_mean = mean(van_best)
    cells = []
    best_nov_mean = None
    best_nov_arm = None
    for arm in arm_order:
        b = agg(task, arm, "best_mppi")
        m = mean(b)
        mx = max(b) if b else None
        n = len(b)
        if m is None:
            cells.append("—")
        else:
            cells.append(f"{m:.1f} [{mx:.1f}] n={n}")
        if arm != "vanilla" and m is not None:
            if best_nov_mean is None or m > best_nov_mean:
                best_nov_mean = m
                best_nov_arm = arm
    # win logic: novelty best-arm mean beats vanilla mean by a margin AND vanilla near floor OR clear sep
    verdict = "—"
    if van_mean is not None and best_nov_mean is not None:
        margin = best_nov_mean - van_mean
        rel = margin / max(abs(van_mean), 1.0)
        if best_nov_mean > 5.0 and (van_mean < 5.0 or rel > 0.5) and margin > 5.0:
            verdict = f"YES ({best_nov_arm}: +{margin:.1f})"
            wins.append((task, best_nov_arm, van_mean, best_nov_mean))
        elif margin > 5.0:
            verdict = f"partial ({best_nov_arm}: +{margin:.1f})"
        else:
            verdict = "no"
    lines.append(f"| {task} | " + " | ".join(cells) + f" | {verdict} |")

    # discovery: did any novelty seed leave floor where vanilla stayed ~0?
    van_disc = [x["discovery_step_mppi_gt5"] for x in g.get((task, "vanilla"), [])]
    van_floor = all(d is None for d in van_disc) and len(van_disc) > 0
    nov_disc = {}
    for arm in arm_order:
        if arm == "vanilla":
            continue
        ds = [x["discovery_step_mppi_gt5"] for x in g.get((task, arm), [])]
        nov_disc[arm] = [d for d in ds if d is not None]
    summary["tasks"][task] = {
        "vanilla_best_mppi_mean": van_mean,
        "vanilla_best_mppi_seeds": van_best,
        "vanilla_all_seeds_floor(gt5=never)": van_floor,
        "arms": {arm: {"best_mppi_mean": mean(agg(task, arm, "best_mppi")),
                       "best_mppi_seeds": agg(task, arm, "best_mppi"),
                       "final_mppi_seeds": agg(task, arm, "final_mppi"),
                       "discovery_steps_gt5": [x["discovery_step_mppi_gt5"] for x in g.get((task, arm), [])]}
                 for arm in arm_order},
        "verdict": verdict,
    }

lines.append("")
lines.append("## Discovery (first eval step MPPI return > 5; None = stayed on floor)")
lines.append("")
lines.append("| task | arm | seeds discovery-step |")
lines.append("|---|---|---|")
for task in tasks:
    for arm in arm_order:
        ds = [x["discovery_step_mppi_gt5"] for x in g.get((task, arm), [])]
        if ds:
            lines.append(f"| {task} | {arm} | {ds} |")

lines.append("")
lines.append("## Headline")
if wins:
    lines.append(f"Novelty-MPPI beats vanilla (clean separation) on {len(wins)} task(s):")
    for t, a, vm, nm in wins:
        lines.append(f"- **{t}**: vanilla best={vm:.1f} -> {a} best={nm:.1f}")
else:
    lines.append("No clean-separation win yet (novelty did not beat vanilla by margin on any completed task).")
    lines.append("Honest null / partial — see table above for per-task deltas.")
lines.append("")
lines.append(f"_runs aggregated: {len(runs)}_")

open(os.path.join(ROOT, "VERDICT.md"), "w").write("\n".join(lines))
summary["wins"] = [{"task": t, "arm": a, "vanilla": vm, "novelty": nm} for t, a, vm, nm in wins]
summary["arm_order"] = arm_order
json.dump(summary, open(os.path.join(ROOT, "summary.json"), "w"), indent=2)
print(f"[verdict] {len(runs)} runs, {len(wins)} clean wins -> VERDICT.md + summary.json")
