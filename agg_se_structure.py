"""Aggregate the Thread-D SE-as-structure sweep and write VERDICT.md + summary.json.
Every number is read from the frozen-probe run JSONs on disk. Reports n per cell.
Honest test: does an SE-STRUCTURE arm BEAT plain `none` on geom_r2 / value_r2?"""
import json, glob, os
from collections import defaultdict
import numpy as np

ROOT = "/root/tdmpc_glass/exp/D_se_structure"
RUNS = os.path.join(ROOT, "runs")
TASKS = ["WalkerWalk", "CheetahRun", "ReacherEasy"]
ARMS = ["none", "se_contrastive", "se_tree"]

rows = defaultdict(list)   # (task,arm) -> list of dicts
for f in glob.glob(os.path.join(RUNS, "run_*.json")):
    d = json.load(open(f))
    rows[(d["task"], d["cond"])].append(d)

def agg(key, field):
    vals = [r[field] for r in rows.get(key, [])]
    return vals

summary = {"lambda_con": 0.03, "steps": 12000, "tau_con": 0.2, "knn": 15, "k_coarse": 4,
           "note": "SE integrated as STRUCTURE (selib min-2D-SE partition supplies contrastive "
                   "positives/negatives), NOT as a scalar penalty. Fixed lambda=0.03 (NOT "
                   "grad-matched; chosen on a WalkerWalk pilot to keep l_pred near baseline).",
           "cells": {}, "verdict": {}}

lines = []
lines.append("# Thread D — SE as STRUCTURE (not a penalty): VERDICT\n")
lines.append("**Question.** D was NULL because SE was only ever a *latent regularizer* (a scalar "
             "penalty `lam*se2d_soft`). Here SE is integrated as **structure**: selib's real "
             "min-2D-SE optimizer partitions the batch-latent kNN graph, and that partition "
             "SUPPLIES the supervision — supervised-contrastive positives (same SE community) / "
             "negatives (cross community). `se_tree` adds a coarse SE level (encoding-tree "
             "hierarchy). Fixed lambda=0.03 (NOT grad-matched). Frozen-encoder ridge probes.\n")
lines.append("**Honest bar.** Beat plain `none` (D2 showed `none` is best-or-tied). "
             "All numbers from disk; mean +/- std over seeds, n reported.\n")

# per-task, per-arm table
hdr = f"| task | arm | n | geom_r2 (mean+/-std) | value_r2 (mean+/-std) | eff_rank | ncomm_fine |"
lines.append(hdr)
lines.append("|---|---|---|---|---|---|---|")
beats = {"geom": defaultdict(int), "value": defaultdict(int)}
totcmp = defaultdict(int)
per_task_delta = defaultdict(list)
for task in TASKS:
    none_g = np.array(agg((task, "none"), "geom_r2"))
    none_v = np.array(agg((task, "none"), "value_r2"))
    for arm in ARMS:
        g = np.array(agg((task, arm), "geom_r2"))
        v = np.array(agg((task, arm), "value_r2"))
        er = np.array(agg((task, arm), "eff_rank"))
        nc = np.array(agg((task, arm), "ncomm_fine"))
        n = len(g)
        if n == 0:
            lines.append(f"| {task} | {arm} | 0 | - | - | - | - |")
            continue
        gm = f"{g.mean():.4f}+/-{g.std():.4f}"
        vm = f"{v.mean():.4f}+/-{v.std():.4f}"
        erm = f"{er.mean():.2f}" if len(er) else "-"
        ncm = f"{nc.mean():.1f}" if len(nc) and arm != "none" else "-"
        lines.append(f"| {task} | {arm} | {n} | {gm} | {vm} | {erm} | {ncm} |")
        summary["cells"][f"{task}/{arm}"] = {
            "n": n, "geom_r2_mean": float(g.mean()), "geom_r2_std": float(g.std()),
            "value_r2_mean": float(v.mean()), "value_r2_std": float(v.std()),
            "eff_rank_mean": float(er.mean()) if len(er) else None,
            "ncomm_fine_mean": float(nc.mean()) if len(nc) else None}
        if arm != "none" and len(none_g):
            dg = g.mean() - none_g.mean(); dv = v.mean() - none_v.mean()
            per_task_delta[arm].append((task, dg, dv))
            totcmp[arm] += 1
            if dg > 0: beats["geom"][arm] += 1
            if dv > 0: beats["value"][arm] += 1

lines.append("")
lines.append("## SE-structure vs `none` (delta = arm_mean - none_mean, per task)\n")
lines.append("| arm | task | d_geom_r2 | d_value_r2 |")
lines.append("|---|---|---|---|")
for arm in ["se_contrastive", "se_tree"]:
    for task, dg, dv in per_task_delta.get(arm, []):
        summary["verdict"][f"{arm}/{task}"] = {"d_geom_r2": float(dg), "d_value_r2": float(dv)}
        lines.append(f"| {arm} | {task} | {dg:+.4f} | {dv:+.4f} |")

lines.append("")
lines.append("## Summary of beats over `none`\n")
for arm in ["se_contrastive", "se_tree"]:
    tc = totcmp.get(arm, 0)
    bg = beats["geom"].get(arm, 0); bv = beats["value"].get(arm, 0)
    # mean delta across tasks
    dg = np.mean([d[1] for d in per_task_delta.get(arm, [])]) if per_task_delta.get(arm) else float("nan")
    dv = np.mean([d[2] for d in per_task_delta.get(arm, [])]) if per_task_delta.get(arm) else float("nan")
    lines.append(f"- **{arm}**: beats none on geom in {bg}/{tc} tasks, on value in {bv}/{tc} tasks; "
                 f"mean d_geom={dg:+.4f}, mean d_value={dv:+.4f}.")
    summary["verdict"][f"{arm}/overall"] = {"tasks": tc, "geom_beats": bg, "value_beats": bv,
                                            "mean_d_geom": float(dg), "mean_d_value": float(dv)}

# overall verdict string
def verdict_for(arm):
    tc = totcmp.get(arm, 0)
    if tc == 0: return "NO DATA"
    bg = beats["geom"].get(arm, 0); bv = beats["value"].get(arm, 0)
    dg = np.mean([d[1] for d in per_task_delta.get(arm, [])])
    dv = np.mean([d[2] for d in per_task_delta.get(arm, [])])
    if bg == tc and bv == tc:
        return "BEATS none on BOTH metrics across ALL tasks (clean positive)"
    if bg >= (tc+1)//2 and bv >= (tc+1)//2 and (dg > 0 and dv > 0):
        return "PARTIAL positive: beats none on the majority of tasks (mean delta > 0 on both)"
    if dg <= 0 and dv <= 0:
        return "NULL/NEG: does not beat none (mean delta <= 0 on both)"
    return "MIXED: beats on some cells, not a clean win"

lines.append("")
lines.append("## VERDICT\n")
for arm in ["se_contrastive", "se_tree"]:
    vv = verdict_for(arm)
    summary["verdict"][f"{arm}/label"] = vv
    lines.append(f"- **{arm}**: {vv}")

open(os.path.join(ROOT, "VERDICT.md"), "w").write("\n".join(lines) + "\n")
json.dump(summary, open(os.path.join(ROOT, "summary.json"), "w"), indent=2)
print("\n".join(lines))
print("\nwrote VERDICT.md + summary.json")
