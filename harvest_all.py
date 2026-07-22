#!/usr/bin/env python3
"""Harvest the VSG 4-cell (full/strip x pi/mppi) across the 3 DMC boxes → n=3 table."""
import subprocess, csv, io, statistics as st, collections
KEY = "/home/ubuntu/.ssh/vastai_id_ed25519"
BOXES = [("124.60.192.12",52914), ("154.12.38.116",50649), ("188.250.227.38",28852), ("156.238.224.242",62366)]
SNIP = r'''python3 - <<'PY' 2>/dev/null
import csv,glob,statistics,os
for f in glob.glob("/root/helios-rl/exp/tdmpc_glass/*_vsg_*/seed_*.csv"):
    d=os.path.basename(os.path.dirname(f))
    try:
        left,right=d.split("_vsg_",1); task=left
        arm="full" if "_full_s" in right else ("strip" if "_strip_s" in right else "?")
        seed=right.rsplit("_s",1)[1]
    except: continue
    rows=[x for x in list(csv.reader(open(f)))[1:] if len(x)>3]
    if not rows: continue
    last=int(rows[-1][0])
    if last<1400000: continue
    for et in ("pi","mppi"):
        v=[float(x[1]) for x in rows if x[2]==et]
        if v: print(f"{task}|{arm}|{seed}|{et}|{round(statistics.median(v[-6:]),1)}")
PY'''
# collect: (task,arm,et) -> list of per-seed medians (dedup by seed across boxes)
seen=collections.defaultdict(dict)  # (task,arm,et) -> {seed: val}
for host,port in BOXES:
    try:
        out=subprocess.run(["ssh","-i",KEY,"-o","StrictHostKeyChecking=no","-o","ConnectTimeout=15",
                            "-p",str(port),f"root@{host}",SNIP],capture_output=True,text=True,timeout=45).stdout
    except Exception: continue
    for ln in out.splitlines():
        p=ln.split("|")
        if len(p)!=5: continue
        task,arm,seed,et,val=p
        try: seen[(task,arm,et)][seed]=float(val)
        except: pass
tasks=sorted({k[0] for k in seen})
def med(task,arm,et):
    d=seen.get((task,arm,et),{})
    return (round(st.median(d.values()),0) if d else None), len(d)
print(f"{'task':18} {'full pi':>8} {'full mppi':>9} {'strip pi':>9} {'strip mppi':>10}  n(f/s)")
for t in tasks:
    fp,nf=med(t,"full","pi"); fm,_=med(t,"full","mppi")
    sp,ns=med(t,"strip","pi"); sm,_=med(t,"strip","mppi")
    if fp is None and sp is None: continue
    print(f"{t:18} {str(fp):>8} {str(fm):>9} {str(sp):>9} {str(sm):>10}  {nf}/{ns}")
