*TD-MPC-Glass — week of Jul 1–8 (Part 10)*

Started the week with 5 bets to beat PPO via abstraction / planning-as-exploration. 4 of 5 nulled under matched
controls (planning is *not* a directed-exploration operator — on sparse CartpoleSwingup, policy-only discovers the
reward *earlier*, 140k vs 157k steps; novelty-MPPI worse on 4/4 matched seeds; SE/uniformity anti-collapse both
hurt value-based control). That pivoted us from "add structure to win" to *dissecting why the planner wins* — which
gave three solid results (→ three papers):

1. *Categorical wall:* tuned PPO gets 0/5 seeds ≥200 on HopperHop at 472M steps and survives entropy ×3/×10;
   TD-MPC2 clears it 6/6 by ~1M. Needs contact-criticality (contact-free Acrobot has no wall).
2. *Mechanism (5-loss ablation × 4 tasks):* the value + policy losses are individually fatal; the "world-model"
   (consistency) loss is the *mildest* cut — contradicts TD-MPC2's own ablation story.
3. *Sufficiency law:* trained off from scratch, the consistency loss is load-bearing on planner-led tasks
   (Walker −23%, Cheetah −38%, Acrobot −44%) and removable only on HopperHop (n=8).

We then tried to *build* a better world-model objective: two consistency-reweighting schemes (value-aware and
uncertainty-aware) — both lose to plain uniform consistency by ~5–9%, so TD-MPC2's uniform loss is near-optimal in
form (a clean negative that strengthens the paper).

*Now running / next:* (a) a value-conditioned abstraction bet (bisimulation) forcing structure into the value head;
(b) a reopened JEPA+SE plan using SE as *structure* (community-defined hierarchy), scoped to goal-conditioned /
transfer tasks where structure isn't already redundant — not dense value-based control.

*One-line takeaway:* in a value-based planner, added structure and loss-reweighting buy nothing the TD value pathway
doesn't already consume; abstraction buys sample-efficiency where its prior fits, not a higher ceiling.

Write-up: suuttt.github.io/projects/2026-07-08-tdmpc-glass-weekly-review-part10/
