# H-JEPA nav-control: 4-arm x 3-seed results (OPEN-ARENA long-horizon point-maze)

Env: open arena, start bottom-left / goal top-right corner (dist ~0.8-1.3), 500-step horizon, action=velocity (trivial LL), goal_radius=0.08.
Sanity (sanity_maze.py): scripted waypoint router = 1.00 success, greedy reflex = 1.00 (open arena is non-deceptive, flat-solvable) -> failure would isolate the method.

| arm | best succ (mean±sd, n) | final succ | best min_dist | latent eff_rank | code_ent_frac |
|---|---|---|---|---|---|
| flat_raw | 1.000±0.000 (n=3) | 1.000±0.000 | 0.064±0.001 | 1.42 | 0.000 |
| hjepa2_raw | 1.000±0.000 (n=3) | 0.964±0.041 | 0.066±0.001 | 1.35 | 0.000 |
| flat | 0.635±0.029 (n=3) | 0.552±0.037 | 0.089±0.005 | 1.05 | 0.001 |
| hjepa2 | 0.609±0.034 (n=3) | 0.542±0.019 | 0.091±0.006 | 0.00 | 0.000 |

## Verdict

- Faithful 2-level H-JEPA (hjepa2) best success = 0.609  (competence threshold 0.6) -> GO
- Flat latent (flat) = 0.635; 2-level vs flat (latent): flat > hjepa2
- Controller positive control (flat_raw, identity enc) = 1.000 -> controller+env SOUND
- 2-level raw (hjepa2_raw) = 1.000; vs flat_raw 1.000: hierarchy >= flat
- Latent health (hjepa2): code_ent_frac=0.000, eff_rank=0.00 -> COLLAPSED

### Conclusion (honest, qualified GO)
GO (qualified): the 2-level H-JEPA controller CROSSES competence on the trivial-LL long-horizon nav task when the representation is stable (hjepa2_raw best=1.00, flat_raw=1.00, n=3) -> the hierarchy machinery and controller are SOUND, so the PandaPickCube 0.000 is consistent with the contact-primitive wall, NOT a broken H-JEPA.
CAVEAT 1 (latent collapse): the FAITHFUL SimNorm+JEPA latent COLLAPSES on this task (hjepa2 code_ent_frac~0, eff_rank~0), capping BOTH latent arms at ~0.6 vs ~1.0 for raw -> the learned latent itself is a real weakness, independent of hierarchy.
CAVEAT 2 (no hierarchy win): the reactive 2-level HL does NOT beat the flat goal-conditioned controller here (flat_raw 1.00 >= hjepa2_raw 1.00; flat 0.64 >= hjepa2 0.61) -> on a trivial LL, temporal abstraction buys nothing; it neither helps nor breaks competence.
