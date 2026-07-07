#!/usr/bin/env python
"""Aggregate C_hier_new runs -> summary.json + VERDICT.md. All numbers from disk."""
import os, json, glob, statistics as st

RUNS = "/root/tdmpc_glass/exp/C_hier_new/runs"
OUT = "/root/tdmpc_glass/exp/C_hier_new"

def first_cross(curve, thr):
    for p in curve:
        if p["success"] >= thr:
            return p["step"]
    return None

def load():
    recs = []
    for f in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
        try:
            recs.append(json.load(open(f)))
        except Exception as e:
            print("skip", f, e)
    return recs

def agg():
    recs = load()
    groups = {}  # (arm,maze) -> list
    for r in recs:
        groups.setdefault((r["arm"], r["maze"]), []).append(r)
    summary = {"n_runs": len(recs), "cells": []}
    for (arm, maze), rs in sorted(groups.items()):
        peaks = [r.get("peak_success", 0.0) for r in rs]
        finals = [r.get("final_success", 0.0) for r in rs]
        done = [r for r in rs if r.get("done")]
        # steps to competence: median first-cross of 0.5 among runs that reach it
        crosses = [first_cross(r.get("curve", []), 0.5) for r in rs]
        reached = [c for c in crosses if c is not None]
        cell = {
            "arm": arm, "maze": maze, "n": len(rs), "n_done": len(done),
            "seeds": sorted(r["seed"] for r in rs),
            "peak_mean": round(st.mean(peaks), 4),
            "peak_std": round(st.pstdev(peaks), 4) if len(peaks) > 1 else 0.0,
            "peak_max": round(max(peaks), 4),
            "final_mean": round(st.mean(finals), 4),
            "steps_to_0.5_median": (int(st.median(reached)) if reached else None),
            "steps_to_0.5_n_reached": len(reached),
            "total_steps": rs[0].get("total_steps"),
        }
        summary["cells"].append(cell)
    # verdict per maze: feudal vs flat
    verdict = {}
    for maze in sorted(set(m for _, m in groups)):
        c = {x["arm"]: x for x in summary["cells"] if x["maze"] == maze}
        if "flat" in c and "feudal" in c:
            fp = c["feudal"]["peak_mean"]; lp = c["flat"]["peak_mean"]
            fs = c["feudal"]["steps_to_0.5_median"]; ls = c["flat"]["steps_to_0.5_median"]
            beats_success = fp > lp + 1e-9
            faster = (fs is not None) and (ls is None or (ls is not None and fs < ls))
            verdict[maze] = {
                "feudal_peak": fp, "flat_peak": lp,
                "feudal_steps_to_0.5": fs, "flat_steps_to_0.5": ls,
                "feudal_beats_flat_on_peak": beats_success,
                "feudal_faster_to_0.5": bool(faster),
                "clean_beat": bool(beats_success or faster),
            }
    summary["verdict_by_maze"] = verdict
    summary["overall_clean_beat"] = any(v["clean_beat"] for v in verdict.values())
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    return summary

def md(summary):
    L = ["# C_hier_new - LEARNED 2-level hierarchy vs flat (VERDICT)", ""]
    L += ["**Question:** does a *fully-learned* 2-level feudal hierarchy (learned HL subgoal"
          " policy + learned goal-conditioned LL, NO analytic controller) BEAT a matched-budget"
          " flat TD3 baseline on a long-horizon **sparse-reward** PointMaze, where the low level"
          " (local navigation) is genuinely learnable? This closes the prior null (whose LL was"
          " an analytic phase controller).", ""]
    L += ["**Metric:** REAL success = fraction of deterministic eval episodes reaching the TRUE"
          " goal. Matched budget = identical total ENV STEPS. Intrinsic LL reward (progress to"
          " self-generated subgoal) is a structural HRL mechanism, never reported as success.", ""]
    L += ["## Results (peak held-out TRUE success)", ""]
    L += ["| maze | arm | peak mean | peak std | peak max | final mean | steps->0.5 (median) | n reached 0.5 | n |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c in summary["cells"]:
        L.append(f"| {c['maze']} | {c['arm']} | {c['peak_mean']} | {c['peak_std']} | "
                 f"{c['peak_max']} | {c['final_mean']} | {c['steps_to_0.5_median']} | "
                 f"{c['steps_to_0.5_n_reached']} | {c['n']} |")
    L += ["", "## Verdict by maze", ""]
    for maze, v in summary["verdict_by_maze"].items():
        L.append(f"### {maze}")
        L.append(f"- feudal peak **{v['feudal_peak']}** vs flat **{v['flat_peak']}** "
                 f"-> beats-on-peak: **{v['feudal_beats_flat_on_peak']}**")
        L.append(f"- feudal steps->0.5 **{v['feudal_steps_to_0.5']}** vs flat **{v['flat_steps_to_0.5']}** "
                 f"-> faster: **{v['feudal_faster_to_0.5']}**")
        L.append(f"- **clean beat: {v['clean_beat']}**")
        L.append("")
    L += [f"## BOTTOM LINE: overall clean beat = **{summary['overall_clean_beat']}**", ""]
    L += ["_All numbers read from runs/*.json. n = seeds per cell._"]
    open(os.path.join(OUT, "VERDICT.md"), "w").write("\n".join(L))

if __name__ == "__main__":
    s = agg()
    md(s)
    print(json.dumps(s.get("verdict_by_maze", {}), indent=2))
    print("overall_clean_beat:", s["overall_clean_beat"])
