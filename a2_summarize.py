"""Per-run JSONL -> per-run summary JSON. Robust to empty/missing (crashed run)."""
import json, sys, os

jsonl, task, arm, seed, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
evals = []
if os.path.exists(jsonl):
    for line in open(jsonl):
        line = line.strip()
        if not line:
            continue
        try:
            evals.append(json.loads(line))
        except Exception:
            pass
evals.sort(key=lambda e: e["step"])

def disc_step(thresh):
    for e in evals:
        if e["mppi_return"] > thresh:
            return int(e["step"])
    return None

beta = evals[0]["beta"] if evals else float(os.environ.get("NOVELTY_BETA", "0"))
ntype = evals[0]["novelty_type"] if evals else os.environ.get("NOVELTY_TYPE", "vanilla")
rec = {
    "task": task, "arm": arm, "seed": seed,
    "beta": beta, "novelty_type": ntype,
    "n_evals": len(evals),
    "status": "ok" if evals else "no_evals",
    "final_mppi": (evals[-1]["mppi_return"] if evals else None),
    "final_pi": (evals[-1]["pi_return"] if evals else None),
    "best_mppi": (max(e["mppi_return"] for e in evals) if evals else None),
    "best_mppi_step": (max(evals, key=lambda e: e["mppi_return"])["step"] if evals else None),
    "best_pi": (max(e["pi_return"] for e in evals) if evals else None),
    "discovery_step_mppi_gt5": disc_step(5.0),
    "discovery_step_mppi_gt1": disc_step(1.0),
    "evals": [{"step": int(e["step"]), "pi": e["pi_return"], "mppi": e["mppi_return"]} for e in evals],
}
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(rec, open(out, "w"), indent=2)
print(f"[summarize] {task}/{arm}/s{seed}: n_evals={len(evals)} best_mppi={rec['best_mppi']} final_mppi={rec['final_mppi']}")
