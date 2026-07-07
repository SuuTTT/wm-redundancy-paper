#!/usr/bin/env python3
"""Aggregate the CPG-locomotion-class study into one honest VERDICT.json.
EVERY number is read from disk -- baselines from exp/benchmark CSVs, CPG-alone
from <TASK>_cpg_alone.json, residual learning curves from logs/.../eval_curve_*.json.
Never fabricates.
"""
import os, sys, glob, json, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "..", "..", "benchmark")
TASKS = ["CheetahRun", "WalkerRun", "WalkerWalk", "HopperHop", "HopperStand"]


def _csv_peak_final(path):
  rows = list(csv.DictReader(open(path)))
  if not rows:
    return None, None
  r = [float(x["reward"]) for x in rows]
  return max(r), r[-1]


def baselines(task):
  out = {"tdmpc-glass": [], "ppo": []}
  for f in sorted(glob.glob(os.path.join(BENCH, f"tdmpc-glass_{task}_*.csv"))):
    pk, fn = _csv_peak_final(f)
    if pk is not None:
      out["tdmpc-glass"].append({"file": os.path.basename(f), "peak": round(pk, 1),
                                 "final": round(fn, 1)})
  for f in sorted(glob.glob(os.path.join(BENCH, f"ppo_{task}_s*.csv"))):
    pk, fn = _csv_peak_final(f)
    if pk is not None:
      out["ppo"].append({"file": os.path.basename(f), "peak": round(pk, 1),
                         "final": round(fn, 1)})
  def agg(lst, key):
    v = [x[key] for x in lst]
    return None if not v else {"mean": round(float(np.mean(v)), 1),
                               "std": round(float(np.std(v)), 1),
                               "vals": v, "n": len(v)}
  return {
      "tdmpc-glass_peak": agg(out["tdmpc-glass"], "peak"),
      "tdmpc-glass_final": agg(out["tdmpc-glass"], "final"),
      "ppo_peak": agg(out["ppo"], "peak"),
      "runs": out,
  }


def cpg_alone(task):
  fn = os.path.join(HERE, f"{task}_cpg_alone.json")
  if not os.path.exists(fn):
    return None
  d = json.load(open(fn))
  return {"mean": round(d["mean"], 1), "std": round(d["std"], 1),
          "rets": [round(x, 1) for x in d["rets"]], "n": d.get("n"),
          "tuned": d.get("tuned", False)}


import re

def _agg(runs, key):
  v = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
  return None if not v else {"mean": round(float(np.mean(v)), 1),
                             "std": round(float(np.std(v)), 1),
                             "vals": v, "n": len(v)}


def residual(task):
  """Residual eval curves grouped BY alpha (authority). Keyed 'alpha=X' so the
  anchor-vs-residual question is answerable."""
  by_alpha = {}
  for cj in sorted(glob.glob(os.path.join(HERE, "logs",
                                          f"{task}_res_a*_s*", "eval_curve*.json"))):
    m = re.search(rf"{re.escape(task)}_res_a([0-9.]+)_s", cj)
    alpha = m.group(1) if m else "?"
    try:
      d = json.load(open(cj))
      rets = [c["return"] for c in d["curve"]]
      if not rets:
        continue
      rec = {"json": os.path.relpath(cj, HERE), "peak": round(max(rets), 1),
             "final": round(rets[-1], 1), "init": round(rets[0], 1),
             "n_evals": len(rets)}
    except Exception as e:
      rec = {"json": cj, "error": str(e)}
    by_alpha.setdefault(alpha, []).append(rec)
  out = {}
  for a, runs in sorted(by_alpha.items()):
    out[f"alpha={a}"] = {"runs": runs, "peak": _agg(runs, "peak"),
                         "final": _agg(runs, "final"), "init": _agg(runs, "init")}
  return out


def verdict_line(task, base, cpg, res):
  """Honest per-task one-liner comparing CPG-alone, CPG+residual(alpha=1.0),
  baselines. res is the alpha-keyed dict from residual()."""
  tg = base["tdmpc-glass_peak"]["mean"] if base["tdmpc-glass_peak"] else None
  pp = base["ppo_peak"]["mean"] if base["ppo_peak"] else None
  ca = cpg["mean"] if cpg else None
  a1 = res.get("alpha=1.0")
  rp = a1["peak"]["mean"] if (a1 and a1["peak"]) else None
  # alpha-sweep peaks (anchor question)
  alpha_peaks = {a: (v["peak"]["mean"] if v["peak"] else None)
                 for a, v in res.items()}
  return {"cpg_alone": ca, "cpg+residual_peak_a1.0": rp, "ppo_peak": pp,
          "tdmpc-glass_peak": tg, "alpha_sweep_peaks": alpha_peaks}


def main():
  out = {"study": "CPG locomotion class (uniform-within-class test)",
         "protocol": "A (n>=64 parallel, 1000-step, true task reward)",
         "tasks": {}}
  for t in TASKS:
    b = baselines(t)
    c = cpg_alone(t)
    r = residual(t)
    out["tasks"][t] = {"baselines": b, "cpg_alone": c, "cpg_residual": r,
                       "summary": verdict_line(t, b, c, r)}
  fn = os.path.join(HERE, "VERDICT.json")
  json.dump(out, open(fn, "w"), indent=2)
  print(json.dumps({t: out["tasks"][t]["summary"] for t in TASKS}, indent=2))
  print(f"\nwrote {fn}")


if __name__ == "__main__":
  main()
