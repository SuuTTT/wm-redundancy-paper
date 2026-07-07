"""Thread D — SE as STRUCTURE (not a penalty) in a pure-JEPA harness.

WHY D WAS NULL: SE was only ever a latent *regularizer* (a scalar penalty term:
`lam * se2d_soft(A,S)`), redundant on a value-sufficient TD-MPC2 latent and
neutral/harmful on a pure JEPA. THE FIX: integrate SE as STRUCTURE — run selib's
real min-2D-SE optimizer on the batch-latent kNN graph to get a PARTITION, and let
that partition SUPPLY the supervision (positives/negatives), instead of penalizing a
scalar SE value.

Arms (matched arch/data/steps/probes to D2 `none`):
  none            : pure latent self-prediction ONLY (the honest baseline to beat).
  se_contrastive  : (option a) supervised-contrastive loss; selib SE partition of the
                    batch-latent kNN graph gives positives (same community) / negatives
                    (cross community). SE structure = the label source.
  se_tree         : (option b) multi-resolution SupCon at TWO SE encoding-tree levels
                    (fine = free-k louvain-SE optimum; coarse = SE merged down to K).
                    Uses the SE hierarchy as block structure.

Every arm freezes the encoder and linear-probes held-out R^2 for:
  geom_r2  : decode true physical qpos  (geometry)
  value_r2 : decode discounted return-to-go (value)
The honest positive test: does an SE-STRUCTURE arm BEAT plain `none`?

FIXED lambda only (NO grad-matching — a known artifact). Numbers all from frozen probes.

USAGE
  python se_structure.py train --task WalkerWalk --cond se_contrastive --seed 0 --gpu 2
  python se_structure.py train --task WalkerWalk --cond none --seed 0 --gpu 2 --steps 800   # smoke
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

HELIOS = Path("/root/tdmpc_glass/helios-rl")
sys.path.insert(0, str(HELIOS / "src"))
sys.path.insert(0, "/root/tdmpc_glass/mujoco_playground_repo")
sys.path.insert(0, "/root/tdmpc_glass/selib")

# reuse the EXACT D2 data + probe helpers for an apples-to-apples vs-none comparison
D2 = Path("/root/tdmpc_glass/exp/proposal_D2_pure_jepa")
sys.path.insert(0, str(D2))
ROOT = Path("/root/tdmpc_glass/exp/D_se_structure")
ROOT.mkdir(parents=True, exist_ok=True)

LD = 64
V = 8
HID = (512, 512)
BATCH = 256
STEPS = 12000
LR = 3e-4
EMA = 0.99
KNN = 15
GAMMA = 0.97
TAU_CON = 0.2       # SupCon temperature
LAM_CON = 1.0       # FIXED contrastive weight (NOT grad-matched)
K_COARSE = 4        # coarse SE-tree level (merge free-k optimum down to K)


def cache_path(task, datatag=""):
    return D2 / f"data_{task}{datatag}.npz"


# ---- probe helpers (identical formulas to D2) ----
def ridge_r2(Ztr, Ytr, Zte, Yte, alpha=1.0):
    n = Ztr.shape[0]
    Zc = np.concatenate([Ztr, np.ones((n, 1))], 1)
    Zce = np.concatenate([Zte, np.ones((Zte.shape[0], 1))], 1)
    A = Zc.T @ Zc
    reg = alpha * np.eye(A.shape[0]); reg[-1, -1] = 0.0
    W = np.linalg.solve(A + reg, Zc.T @ Ytr)
    pred = Zce @ W
    ss_res = ((Yte - pred) ** 2).sum(0)
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return float(np.mean(r2)), [float(x) for x in r2]


def latent_health(Z, simnorm):
    Z = np.asarray(Z, np.float64)
    n, D = Z.shape
    Zc = Z - Z.mean(0, keepdims=True)
    cov = (Zc.T @ Zc) / max(n - 1, 1)
    ev = np.clip(np.linalg.eigvalsh(cov), 0, None); s = ev.sum()
    eff_rank = float((s * s) / (np.square(ev).sum() + 1e-12)) if s > 0 else 0.0
    return {"z_eff_rank": eff_rank, "z_std_mean": float(Z.std(0).mean())}


# ---- selib SE partition of the batch-latent kNN graph (the STRUCTURE) ----
def build_partitioner():
    import networkx as nx
    from selib.seopt import louvain_se, se_optimize

    def knn_graph(z_np):
        zc = z_np / (np.linalg.norm(z_np, axis=1, keepdims=True) + 1e-8)
        sim = zc @ zc.T
        np.fill_diagonal(sim, -10.0)
        idx = np.argpartition(-sim, KNN, axis=1)[:, :KNN]
        B = z_np.shape[0]
        G = nx.Graph(); G.add_nodes_from(range(B))
        rows = np.repeat(np.arange(B), KNN)
        cols = idx.reshape(-1)
        w = np.clip(sim[rows, cols], 0.0, None)
        ebunch = [(int(i), int(j), float(wij)) for i, j, wij in zip(rows, cols, w) if wij > 0]
        G.add_weighted_edges_from(ebunch)
        return G

    def fine(z_np):
        G = knn_graph(z_np)
        lab = louvain_se(G, seed=0)
        lab = lab[0] if isinstance(lab, tuple) else lab
        return np.asarray(lab, np.int32), G

    def coarse(G):
        lab = se_optimize(G, k=K_COARSE, seed=0, starts=1)
        lab = lab[0] if isinstance(lab, tuple) else lab
        return np.asarray(lab, np.int32)

    return fine, coarse


# ---- train one encoder under one arm, then freeze + probe ----
def train(task, cond, seed, simnorm=1, steps=STEPS, lam_con=LAM_CON, datatag="", outdir=None):
    import jax, jax.numpy as jnp, flax.linen as nn, optax
    from helios.algorithms.tdmpc2 import simnorm as simnorm_fn, NormMLP

    d = np.load(cache_path(task, datatag))
    O, A_, O2, G, RTG = d["O"], d["A"], d["O2"], d["G"], d["RTG"]
    obs_dim, act_dim = int(d["obs_dim"]), int(d["act_dim"])
    N = O.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N); ntr = int(0.85 * N)
    tr, te = perm[:ntr], perm[ntr:]

    def head(x):
        h = NormMLP(HID, LD)(x)
        return simnorm_fn(h, V) if simnorm else (h / (jnp.linalg.norm(h, axis=-1, keepdims=True) + 1e-6))

    class Encoder(nn.Module):
        @nn.compact
        def __call__(self, o):
            return head(o)

    class Pred(nn.Module):
        @nn.compact
        def __call__(self, z, a):
            return head(jnp.concatenate([z, a], -1))

    enc, pred = Encoder(), Pred()
    key = jax.random.PRNGKey(seed)
    key, k1, k2 = jax.random.split(key, 3)
    p_enc = enc.init(k1, jnp.zeros((1, obs_dim)))
    p_pred = pred.init(k2, jnp.zeros((1, LD)), jnp.zeros((1, act_dim)))
    params = {"enc": p_enc, "pred": p_pred}
    target = {"enc": jax.tree.map(lambda x: x, p_enc)}
    tx = optax.chain(optax.clip_by_global_norm(20.0), optax.adam(LR))
    opt = tx.init(params)

    fine_part, coarse_part = build_partitioner()

    def supcon(z, labels):
        # supervised contrastive: positives = same SE community (labels), exclude self.
        zc = z / (jnp.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
        s = (zc @ zc.T) / TAU_CON
        B = z.shape[0]
        eye = jnp.eye(B)
        s = s - eye * 1e9                                   # mask self in logits
        logZ = jax.nn.logsumexp(s, axis=1, keepdims=True)   # log sum_{k!=i} exp
        logp = s - logZ
        pos = (labels[:, None] == labels[None, :]).astype(z.dtype) * (1.0 - eye)
        npos = pos.sum(1)
        li = -(pos * logp).sum(1) / jnp.clip(npos, 1.0, None)
        valid = (npos > 0).astype(z.dtype)
        return (li * valid).sum() / jnp.clip(valid.sum(), 1.0, None)

    def struct_loss(z_t, lab_fine, lab_coarse):
        if cond == "none":
            return 0.0 * jnp.sum(z_t[:1])
        if cond == "se_contrastive":
            return lam_con * supcon(z_t, lab_fine)
        if cond == "se_tree":
            return lam_con * 0.5 * (supcon(z_t, lab_fine) + supcon(z_t, lab_coarse))
        raise ValueError(cond)

    def loss_fn(params, target, o, a, o2, lab_fine, lab_coarse):
        z_t = enc.apply(params["enc"], o)
        zhat = pred.apply(params["pred"], z_t, a)
        z_tgt = jax.lax.stop_gradient(enc.apply(target["enc"], o2))
        l_pred = jnp.mean(jnp.sum((zhat - z_tgt) ** 2, -1))
        l_struct = struct_loss(z_t, lab_fine, lab_coarse)
        return l_pred + l_struct, (l_pred, l_struct)

    @jax.jit
    def encode_only(params, o):
        return enc.apply(params["enc"], o)

    @jax.jit
    def step(params, target, opt, o, a, o2, lab_fine, lab_coarse):
        (loss, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(
            params, target, o, a, o2, lab_fine, lab_coarse)
        upd, opt = tx.update(g, opt, params)
        params = optax.apply_updates(params, upd)
        target = {"enc": jax.tree.map(lambda t, o_: EMA * t + (1 - EMA) * o_,
                                      target["enc"], params["enc"])}
        return params, target, opt, loss, aux

    t0 = time.time()
    Otr, Atr, O2tr = O[tr], A_[tr], O2[tr]
    log = []
    ncomm_fine, ncomm_coarse, npart = 0.0, 0.0, 0
    use_part = cond in ("se_contrastive", "se_tree")
    dummy = jnp.zeros((BATCH,), jnp.int32)
    for it in range(steps):
        idx = rng.integers(0, Otr.shape[0], BATCH)
        o = jnp.asarray(Otr[idx]); a = jnp.asarray(Atr[idx]); o2 = jnp.asarray(O2tr[idx])
        if use_part:
            z_np = np.asarray(encode_only(params, o))          # stop-grad structure source
            lf, Gk = fine_part(z_np)
            lc = coarse_part(Gk) if cond == "se_tree" else lf
            ncomm_fine += len(set(lf.tolist())); ncomm_coarse += len(set(lc.tolist())); npart += 1
            lab_fine = jnp.asarray(lf); lab_coarse = jnp.asarray(lc)
        else:
            lab_fine = lab_coarse = dummy
        params, target, opt, loss, aux = step(params, target, opt, o, a, o2, lab_fine, lab_coarse)
        if it % 2000 == 0 or it == steps - 1:
            lp, ls = float(aux[0]), float(aux[1])
            cf = ncomm_fine / max(npart, 1); cc = ncomm_coarse / max(npart, 1)
            log.append({"it": it, "loss": float(loss), "l_pred": lp, "l_struct": ls,
                        "ncomm_fine": round(cf, 2), "ncomm_coarse": round(cc, 2)})
            print(f"[{task}/{cond} s{seed}] it={it} loss={float(loss):.4f} l_pred={lp:.4f} "
                  f"l_struct={ls:.4f} ncomm_fine={cf:.1f} ncomm_coarse={cc:.1f} "
                  f"{time.time()-t0:.0f}s", flush=True)

    # ---- freeze + probe ----
    def encode_all(obs):
        out = []
        for i in range(0, obs.shape[0], 4096):
            out.append(np.asarray(encode_only(params, jnp.asarray(obs[i:i + 4096]))))
        return np.concatenate(out)
    Ztr, Zte = encode_all(O[tr]), encode_all(O[te])
    geom_r2, geom_per = ridge_r2(Ztr, G[tr], Zte, G[te])
    val_r2, _ = ridge_r2(Ztr, RTG[tr].reshape(-1, 1), Zte, RTG[te].reshape(-1, 1))
    health = latent_health(Zte[:4000], bool(simnorm))

    res = {"task": task, "cond": cond, "simnorm": int(simnorm), "seed": seed, "steps": steps,
           "lam_con": lam_con, "tau_con": TAU_CON, "knn": KNN, "k_coarse": K_COARSE,
           "datatag": datatag, "latent_dim": LD, "n_train": int(len(tr)), "n_test": int(len(te)),
           "wall_s": round(time.time() - t0, 1),
           "geom_r2": geom_r2, "geom_r2_per_dim": geom_per, "value_r2": val_r2,
           "eff_rank": health["z_eff_rank"], "z_std_mean": health["z_std_mean"],
           "ncomm_fine": round(ncomm_fine / max(npart, 1), 2),
           "ncomm_coarse": round(ncomm_coarse / max(npart, 1), 2),
           "train_log": log}
    outd = Path(outdir) if outdir else ROOT
    outd.mkdir(parents=True, exist_ok=True)
    fn = outd / f"run_{task}{datatag}_{cond}_seed{seed}.json"
    fn.write_text(json.dumps(res, indent=2))
    print(f"[{task}/{cond} s{seed}] DONE geom_r2={geom_r2:.4f} value_r2={val_r2:.4f} "
          f"eff_rank={health['z_eff_rank']:.2f} -> {fn}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train"])
    ap.add_argument("--task", default="WalkerWalk")
    ap.add_argument("--cond", default="none",
                    choices=["none", "se_contrastive", "se_tree"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--simnorm", type=int, default=1)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--lam", type=float, default=LAM_CON)
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--datatag", default="")
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.22"
    train(args.task, args.cond, args.seed, simnorm=args.simnorm, steps=args.steps,
          lam_con=args.lam, datatag=args.datatag, outdir=(args.outdir or None))
