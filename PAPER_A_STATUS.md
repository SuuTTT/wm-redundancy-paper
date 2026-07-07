# Paper A — submission status (wm-redundancy)

*Title:* "When Is Explicit Abstraction Redundant for a World Model? A Negative Campaign and the Limits of a Value-Decodability Criterion." Style: TMLR (`tmlr.sty`). **Compiles: pdflatex+bibtex clean, 15 pp.**

## Submission-ready
- **Abstract + intro**: polished to submission grade — negative campaign + the C1 (value-decodability) disproof + the three graph-class data points (monolithic / token-transformer NO-GO / entity-graph synthetic NO-GO) + the scored temporal predictor + entity-graph OOD closure. Venue note in header (ICLR Understanding track / TMLR).
- **Theory/mechanism** (§ basis-change + Prop 1 DPI + value-equivalence + reconciliation with dead-R²) — spliced, coherent.
- **16-task matched benchmark** (glass vs tdmpc2, n=3–4, 1.35× cost) — spliced.
- **Entity-graph OOD NO-GO** — in §4.3 (R²≥0.86 at 2× object count, GO-rejected); the session's hardening (N=14) confirms no cliff. (Verify the in-text numbers match the latest `entity_graph_go/` hardened run.)
- Every quantitative claim footnotes an `evidence/*.json` or the ledger.

## Gaps before submission
1. **Author block** — still anonymous placeholder (`\author{Anonymous...}`); fill before de-anonymizing.
2. **Figures** — paper is currently table/text-only; add: (a) the V(z) vs return-to-go R² discrimination plot (the dead-probe), (b) the 16-task benchmark forest plot, (c) the temporal-predictor 4/4-vs-0/3 k-sweep. Evidence JSONs exist for all three.
3. **Entity-graph numbers refresh** — ensure §4.3 cites the *hardened* (N=4→14) OOD curve from `entity_graph_go_hardened/`, not just the first n=4 pass.
4. **Citation verification** — confirm DreamerV3 / TD-MPC2 / ni2024bridging / grimm2020value / ferns2004 / castro2020 / gelada2019 ids resolve in the .bib; do a fresh related-work sweep at submission (post-cutoff structured-WM papers).
5. **Optional strengthening**: the entity-graph attention-edge *ablation* (zero edge i–j, measure Δreward) is noted as untested — either run it or keep the honest caveat.

## Suggested venue
TMLR (claims-and-evidence; ideal for a null-converted-to-principle paper) or ICLR Understanding/blogpost track. TMLR has no deadline pressure and rewards exactly this kind of rigorous negative result — recommended primary target.

## Companion artifacts (this session)
- `SYNTHESIS_beat_ppo.md` — cross-program matched-control synthesis (feeds the companion speed-of-learning paper, not Paper A).
- `NEXT_PAPER_PROPOSAL.md` v4 + `paper_speed_of_learning.tex` — the companion flagship paper (behavioral/hierarchical abstraction = speed-of-learning lever).
- `lecun_jepa_research.md` — related-work bank for the companion paper.

*Status written after the finalization agent completed its main.tex pass (it hit a server-side rate-limit before writing this file; main.tex itself is complete and compiles).*
