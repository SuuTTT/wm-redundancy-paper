#!/usr/bin/env python3
"""Harvest completed vsg full-vs-strip eval CSVs (seeds >= 12) on a box.

Emits one line per (task, arm, seed): last-6-eval MEDIAN of reward (campaign
stat discipline), and whether the run is DONE (last step == 1500160). Reads
committed per-seed CSVs only; invents nothing. Aggregation into cells happens
locally after collecting all boxes.
"""
import glob, os, csv, statistics, sys

BASE = "/root/helios-rl/exp/benchmark"
DONE_STEP = 1500160
rows = []
for f in sorted(glob.glob(BASE + "/tdmpc2_*_vsg_*_s*.csv")):
    b = os.path.basename(f)[:-4]                    # strip .csv
    if "_vsg_" not in b:
        continue
    left, right = b.split("_vsg_", 1)               # left=tdmpc2_<Task>
    task = left[len("tdmpc2_"):]
    try:
        rest, seed = right.rsplit("_s", 1)          # rest=<Task>_<arm>
        seed = int(seed)
        arm = rest.rsplit("_", 1)[1]                # full | strip
    except (ValueError, IndexError):
        continue
    if seed < 12 or arm not in ("full", "strip"):
        continue
    steps, rewards = [], []
    try:
        with open(f) as fh:
            for r in csv.DictReader(fh):
                try:
                    steps.append(int(float(r["step"])));
                    rewards.append(float(r["reward"]))
                except (ValueError, KeyError):
                    pass
    except OSError:
        continue
    if not rewards:
        continue
    done = steps and steps[-1] >= DONE_STEP
    last6 = statistics.median(rewards[-6:])
    print(f"{task}\t{arm}\t{seed}\t{len(rewards)}\t{last6:.2f}\t{'DONE' if done else 'part'}")
