# LeCun's World-Models / JEPA Program — A Literature Map for Positioning

**Purpose.** This document maps Yann LeCun's "World Models / JEPA" research program rigorously so that our next paper can position precisely against it. Our group has an empirical "negative campaign" result: on a *value-sufficient* world-model latent (TD-MPC2 / SimNorm substrate), explicit representation *abstraction* (monolithic vs. token-transformer vs. entity-graph encoders) is **redundant for asymptotic performance** both in-distribution and OOD; its only demonstrated value is **sample efficiency** — which dovetails with LeCun's slogan that "intelligence is measured by speed of learning." This review establishes (a) what LeCun is actually claiming, (b) the technical scaffolding (EBM, anti-collapse, information measures, hierarchy), and (c) where our result confirms vs. does **not** refute his bet, so we can frame the open question honestly.

> **Citation-accuracy note.** Every arXiv ID, author list, and year below was verified via web search against arXiv / publisher pages during the writing of this document. Where I am paraphrasing a claim rather than quoting, I say so. A small number of attributions I flag explicitly as "needs a direct-quote check before camera-ready" — see the ⚠ markers. Several "JEPA" derivative titles surfaced in search with 2602/2603 arXiv stems (i.e., 2026 preprints from after my training cutoff); I did **not** rely on those for any claim and they are not cited here. Treat them as a signal that the JEPA derivative space is now crowded — useful for "related work is exploding" framing, but verify independently before citing.

---

## 1. The JEPA family

### 1.1 The position paper (the thesis we are arguing against)

- **LeCun (2022), "A Path Towards Autonomous Machine Intelligence," v0.9.2, 2022-06-27.** OpenReview `BZ5a1r-kVsf`. This is a *position/vision* paper, not an empirical one. It proposes a modular cognitive architecture — configurator, perception, **world model**, cost module, actor, short-term memory — and names the **Joint-Embedding Predictive Architecture (JEPA)** as the way the world model should be built, with a **Hierarchical JEPA (H-JEPA)** for multi-scale prediction and planning. Two load-bearing claims for us:
  1. The world model should be an **action-conditioned predictor in an abstract representation space**, *not* a pixel/observation-level generator/simulator.
  2. Prediction in representation space lets the model **discard unpredictable / irrelevant detail** (the encoder is free to throw away information), which is exactly what a generative model cannot do because it is forced to reconstruct everything.

The intellectual core: **JEPA predicts the *representation* `s_y` of a target `y` from the representation `s_x` of a context `x` (plus optionally a latent variable `z` and an action `a`), rather than predicting `y` itself.** A generative architecture predicts `y` from `x`; a JEPA predicts `Enc(y)` from `Enc(x)`. Because the target encoder can map many surface variations of `y` to the same `s_y`, JEPA is *not penalized for failing to represent inherently unpredictable detail* (e.g., exact texture, leaf positions, pixel noise). That is the headline argument for "abstract space, not simulator."

### 1.2 The empirical instantiations

- **I-JEPA — Assran, Duval, Misra, Bojanowski, Vincent, Rabbat, LeCun, Ballas (2023), "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture."** arXiv `2301.08243`, CVPR 2023. From a single **context block**, predict the *representations* of several **target blocks** in the same image (masked prediction in latent space). No hand-crafted augmentations, no pixel reconstruction, no negatives. Demonstrated that latent-space masked prediction yields strong semantic features and is more compute-efficient than pixel-reconstruction MIM (e.g., MAE) at scale. This is the proof-of-concept that "predict features, not pixels" works for images.

- **V-JEPA — Bardes, Garrido, Ponce, Chen, Rabbat, LeCun, Assran, Ballas (2024), "Revisiting Feature Prediction for Learning Visual Representations from Video."** arXiv `2404.08471`. Extends I-JEPA to spatiotemporal video; **feature prediction as the *sole* objective** (no pretrained image encoder, no text, no negatives, no reconstruction). ViT-H/16 trained only on video reported 81.9% K400, 72.2% SSv2, 77.9% ImageNet (frozen-evaluation numbers per the paper). The key methodological ingredients: high masking ratio, EMA target encoder, and architectural collapse-avoidance (see §3).

- **V-JEPA 2 — Meta / FAIR (2025), "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning."** arXiv `2506.09985`. This is the most important one for our positioning because it crosses from *representation learning* into *action-conditioned world model + planning*:
  - Pretrain an **action-free** JEPA on >1M hours of internet video (understanding/anticipation benchmarks: e.g., 77.3 SSv2 top-1; 39.7 R@5 EK100 anticipation, per the paper).
  - **Post-train an action-conditioned model, V-JEPA 2-AC**, on <62 hours of unlabeled robot video (Droid). Deployed **zero-shot** on Franka arms in new labs for pick/place via **planning to image goals** — no task-specific reward, no in-lab data collection.
  - This is LeCun's thesis made concrete: *abstract-latent, action-conditioned prediction + planning, no pixel generation.* Our paper must engage with this directly.

### 1.3 What problem JEPA claims to solve vs. generative/reconstruction models

| Axis | Generative / reconstruction (MAE, pixel video models, Dreamer-style decoders) | JEPA (joint-embedding predictive) |
|---|---|---|
| Prediction target | Raw `y` (pixels / observations) | Encoded `Enc(y)` (representation) |
| Treatment of unpredictable detail | Forced to model it (wastes capacity, blurry/uncertain outputs) | Free to discard it via the encoder |
| Failure mode if uncertainty high | Mode-averaging / blur | Latent `z` absorbs residual ambiguity |
| Use as a "simulator" | Yes (can render rollouts) | No (predicts states, not observations) |
| LeCun's claim | "Wrong abstraction level for a world model" | "Right abstraction level — predict consequences, not appearances" |

The central LeCun argument: a world model used for *planning* needs to predict the **consequences of actions in a space where the consequences are predictable**, i.e., an abstract representation that has already dropped irrelevant detail. Video generation ("Sora-as-world-model") is, on this view, solving the wrong problem — spending capacity rendering pixels it could ignore. (We should be careful: this is a *position*, and the V-JEPA 2 robot results are early and narrow; the claim that generative WMs are categorically worse for *control* is empirically contested by the Dreamer line — see §6.)

---

## 2. Energy-Based Models — the unifying view, and JEPA as an EBM

- **LeCun, Chopra, Hadsell, Ranzato, Huang (2006), "A Tutorial on Energy-Based Learning."** In *Predicting Structured Data*, MIT Press. The foundational text. An **EBM** associates a scalar **energy** `E(x, y)` to each configuration; low energy = compatible `(x,y)`. Inference = find `y` minimizing `E(x, y)` given observed `x`. Crucially, the tutorial argues you **do not need a normalized probability distribution** — you only need the energy surface to be *low on the data manifold and higher off it*. Probabilistic models (NLL training) are the special case obtained by passing energies through a Gibbs/softmax and normalizing (the partition function `Z`).

- **Why this matters for JEPA's positioning.** LeCun's recurring methodological claim: **probabilistic modeling is over-committed.** Normalizing requires `Z`, which is intractable in high dimensions and *forces* you to assign probability mass everywhere — i.e., to model the full distribution including the parts you don't care about. EBMs let you **shape the energy only where it matters** (push down on data, push up nearby), which is the same move JEPA makes at the representation level. This is the "cost/energy formalizes all model types" thesis: discriminative, generative, contrastive, and joint-embedding methods are all special cases of "design an energy and a way to push it down on good configs / up on bad ones."

- **JEPA as an EBM.** In JEPA the energy is (roughly) the **prediction error in representation space**: `E(x, y) = D(s_y, Pred(s_x, z))`, the distance between the encoded target and the predicted target representation. Low energy ↔ the target's representation is well-predicted from the context's. Planning = inference = find the action sequence (and latents) that minimize accumulated energy / cost. The **collapse problem** (§3) is precisely the EBM pathology of an energy surface that is **low everywhere** (trivially satisfied) — the classic "EBM needs a contrastive or regularization term to push energy *up* somewhere."

- **Bridging note.** There is a 2023 write-up, "Introduction to Latent Variable Energy-Based Models: A Path Towards Autonomous Machine Intelligence" (arXiv `2306.02572`; also IOP J. Stat. Mech. 2024), that restates the position paper in explicit LV-EBM language. ⚠ Useful as a *secondary* citation that formalizes "JEPA is an LV-EBM"; verify authorship/venue before relying on it as primary.

---

## 3. Representation collapse and information measures

### 3.1 Why JEPA / SSL collapses

If you only minimize prediction error in representation space, the trivial optimum is a **constant encoder**: map everything to the same vector, prediction error → 0, energy low everywhere. This is **representation collapse** (a.k.a. informational collapse / dimensional collapse). Every non-contrastive joint-embedding method needs an explicit or implicit mechanism to prevent it.

### 3.2 The anti-collapse mechanisms (taxonomy)

**Contrastive (explicit negatives):**
- **InfoNCE / CPC — van den Oord, Li, Vinyals (2018), "Representation Learning with Contrastive Predictive Coding," arXiv `1807.03748`.** Push representations of positives together, negatives apart; collapse is prevented because a constant encoder cannot discriminate negatives. InfoNCE is a *lower bound on mutual information* — which becomes the source of the measurement problem in §3.3.

**Non-contrastive — explicit information/variance regularization:**
- **Barlow Twins — Zbontar, Jing, Misra, LeCun, Deny (2021), "Self-Supervised Learning via Redundancy Reduction," arXiv `2103.03230`, ICML 2021.** Drive the **cross-correlation matrix** between two views' embeddings toward the identity: diagonal → invariance, off-diagonal → 0 → **decorrelation/redundancy reduction**. Decorrelation across dimensions prevents dimensional collapse without negatives.
- **VICReg — Bardes, Ponce, LeCun (2021/2022), "Variance-Invariance-Covariance Regularization for Self-Supervised Learning," arXiv `2105.04906`, ICLR 2022.** Three explicit terms: **V**ariance (hinge loss keeping per-dimension std above a threshold → prevents collapse directly), **I**nvariance (match the two views), **C**ovariance (off-diagonal covariance → 0 → decorrelate). VICReg is the cleanest "you can prevent collapse with an *explicit, interpretable* regularizer rather than architectural tricks." This is the regularizer family V-JEPA-style training draws on.

**Non-contrastive — architectural / implicit (asymmetry + stop-grad):**
- **BYOL — Grill et al. (2020), "Bootstrap Your Own Latent," arXiv `2006.07733`, NeurIPS 2020.** Online network predicts a **stop-gradient EMA target** network's representation; the predictor + EMA asymmetry empirically prevents collapse **without negatives or explicit variance terms.** This is the mechanism I-JEPA / V-JEPA inherit (EMA target encoder + predictor + stop-grad). Why exactly this avoids collapse is still somewhat empirically argued rather than fully theoretically settled — flag this honestly.

**Whitening:** decorrelation can also be enforced by explicit whitening of the embedding batch (W-MSE and relatives); same goal as the covariance term — full-rank embeddings.

### 3.3 The deep difficulty: you can't measure information content from samples

The anti-collapse story *should* be "maximize the information the representation carries." But you cannot reliably **measure** that information when you only have **samples from the encoder** and do not know the underlying distribution. This is the crux and a natural lever for our paper.

- **MINE — Belghazi, Baratin, Rajeswar, Ozair, Bengio, Courville, Hjelm (2018), "Mutual Information Neural Estimation," arXiv `1801.04062`, ICML 2018.** Estimates MI via the Donsker–Varadhan lower bound with a neural critic. Scalable, but the estimator is high-variance and the bound is loose at high MI.
- **Poole, Ozair, van den Oord, Alemi, Tucker (2019), "On Variational Bounds of Mutual Information," arXiv `1905.06922`, ICML 2019.** Unifies the variational MI bounds (Barber–Agakov, NWJ/MINE, InfoNCE) and shows the **bias–variance tradeoff**: low-variance bounds (InfoNCE) are **upper-bounded by `log K`** (the number of negatives) and so **cannot certify large MI**; high-MI-capable bounds have huge variance.
- **McAllester & Stratos (2020), "Formal Limitations on the Measurement of Mutual Information," arXiv `1811.04251`, AISTATS 2020.** The hard impossibility: **any distribution-free, high-confidence lower bound on MI from `N` samples is `O(log N)`.** You literally cannot certify high mutual information from a feasible sample size without distributional assumptions.

**Why this matters for us.** It means "the representation is rich / has high information" is *not* something you can measure directly — practitioners instead use **proxies**: linear-probe accuracy, downstream task performance, variance/rank of the embedding, effective dimensionality. This is exactly why anti-collapse methods regularize *surrogates* (per-dimension variance, decorrelation) rather than MI itself. **Our structural-entropy (SE) objective is one more such surrogate** — a *structural* measure on the latent graph rather than a statistical MI estimate — and the §3.3 literature is the principled reason a structural surrogate is even worth proposing: the statistical route is provably hard. (We should claim SE as a tractable, assumption-light *structural* proxy, not as a solution to the impossibility result.)

---

## 4. Hierarchical abstraction & hierarchical planning

### 4.1 LeCun's H-JEPA

In the position paper, a single JEPA captures one prediction scale. **H-JEPA** stacks JEPAs at **multiple levels of abstraction and multiple timescales**:
- **Lower levels:** short-range, detail-preserving predictions over short horizons.
- **Higher levels:** long-range, fewer-detail predictions over long horizons.
- **Planning is top-down:** a high level proposes an **abstract subgoal / cost** that the level below treats as its objective, recursively, down to primitive actions. Higher levels predict coarsely far ahead; lower levels fill in detail near-term. The whole stack does **hierarchical predictive planning** by energy minimization at each level.

This is the part of LeCun's bet our single-level result does **not** touch (see §6).

### 4.2 Classical HRL the H-JEPA idea descends from / parallels

- **Options framework — Sutton, Precup, Singh (1999), "Between MDPs and semi-MDPs," *Artificial Intelligence* 112.** Temporally extended actions (options = policy + initiation set + termination); a policy-over-options selects among them. The formal substrate for "act at multiple timescales."
- **FeUdal Networks (FuN) — Vezhnevets, Osindero, Schaul, Heess, Jaderberg, Silver, Kavukcuoglu (2017), "FeUdal Networks for Hierarchical Reinforcement Learning," arXiv `1703.01161`, ICML 2017.** A **Manager** sets abstract goals in a learned latent space at low temporal resolution; a **Worker** produces primitive actions to achieve them. Decoupled training across timescales → long-horizon credit assignment. This is the closest classical analog to H-JEPA's "each level emits subgoals for the level below."
- **HIRO — Nachum, Gu, Lee, Levine (2018), "Data-Efficient Hierarchical Reinforcement Learning," arXiv `1805.08296`, NeurIPS 2018.** Off-policy, goal-conditioned two-level HRL; the high level proposes **goal states** (in observation/state space) that the low level reaches; off-policy correction makes it sample-efficient. Note: goals are raw states, not learned abstractions — a contrast point.
- **Director — Hafner, Lee, Fischer, Abbeel (2022), "Deep Hierarchical Planning from Pixels," arXiv `2206.04114`, NeurIPS 2022.** **Hierarchical planning *inside the latent space of a learned (Dreamer-style) world model*:** a high-level policy selects **latent goals**, a low-level policy achieves them; goals are decodable to images for interpretability. This is the **most direct existing realization of "hierarchical planning in a learned latent world model"** — i.e., the empirical state of the art for the thing H-JEPA proposes, except built on a *generative/reconstructive* world model rather than a JEPA. Director is therefore a key baseline/contrast for any hierarchical claim we make.

**The gap H-JEPA bets on:** prior HRL either (a) uses raw-state goals (HIRO), (b) learns goal latents but on a reconstructive WM (Director), or (c) learns manager latents without a predictive world model (FuN). LeCun's bet is the conjunction: **multi-level, multi-timescale, action-conditioned prediction in a *non-generative abstract* space, with planning by energy minimization at every level.** That full conjunction has **not** been demonstrated at scale as of this writing — it is the open frontier.

---

## 5. Speed-of-learning as the metric

LeCun's repeated public claim (talks, position paper framing): **intelligence should be measured by *how fast* a system learns a new task — sample/data efficiency and fast adaptation — not by asymptotic performance on a fixed benchmark.** A human learns to drive in ~20 hours; the interesting quantity is the *learning rate*, not the ceiling. ⚠ This is largely a *talk/position* claim; for camera-ready, anchor it to a quotable source — the position paper's framing of world models enabling efficient learning, plus a dated public talk — rather than attributing a crisp sentence to a single paper.

Prior art establishing "speed/efficiency of learning" as a measurable axis (so we are not inventing the metric):

- **Few-shot / fast adaptation:** **MAML — Finn, Abbeel, Levine (2017), "Model-Agnostic Meta-Learning," arXiv `1703.03400`, ICML 2017** — explicitly optimizes for *fast adaptation* (few gradient steps to a new task). The canonical "learning-to-learn-fast" formalization. ⚠ ID from memory — verify `1703.03400`.
- **Sample efficiency as the headline metric in RL:** the **Atari-100k** protocol and **EfficientZero — Ye, Liu, Kurutach, Abbeel, Gao (2021), arXiv `2111.00210`** ⚠ — frame "performance at a fixed, small interaction budget" as the figure of merit. The whole **model-based RL** thesis (Dreamer line, TD-MPC line) is justified primarily on **sample efficiency**, not asymptote.
- **Intelligence-as-efficiency, formalized:** **Chollet (2019), "On the Measure of Intelligence," arXiv `1911.01547`** (the ARC paper) defines intelligence as **skill-acquisition efficiency** given priors and experience — the most rigorous existing operationalization of LeCun's slogan, and a strong anchor for our methodology section.
- **World models justified by efficiency:** **Ha & Schmidhuber (2018), "World Models," arXiv `1803.10122`** — learn a compact latent dynamics model so the controller learns fast/cheaply in imagination. The origin of the modern "world model" usage LeCun adopts.

**Takeaway for us:** "measure by speed of learning" is a *defensible, precedented* axis. Our contribution is not inventing it but **applying it as the discriminating axis in a matched-control study** where asymptotic performance is held equal by construction, so that any abstraction benefit *must* show up as a learning-speed difference or not at all.

---

## 6. Where our results fit — and the precise gap

### 6.1 What we showed
On a **value-sufficient** world-model latent (TD-MPC2 / SimNorm — Hansen, Su, Wang (2024), "TD-MPC2: Scalable, Robust World Models for Continuous Control," arXiv `2310.16828`, ICLR 2024), we ran a **matched-control negative campaign**: holding the value-relevant latent fixed, we varied the *representational abstraction class* of the encoder/world-model — **monolithic vs. token-transformer vs. entity-graph** — and an explicit structural-abstraction objective (structural entropy / SE). Finding: **explicit abstraction is redundant for asymptotic return, in-distribution AND OOD; the only consistent benefit is sample efficiency.**

### 6.2 How this relates to LeCun's program

**Confirms (the easy half of his thesis):**
- "Predict in an abstract latent, not pixels" — our substrate is a non-reconstructive latent WM; abstraction *level* is not where we find a ceiling difference, consistent with LeCun's anti-generative stance being about *appearance vs. consequence*, not about how aggressively you must structure the latent.
- **"Intelligence = speed of learning."** Our positive result — abstraction's value is *sample efficiency* — is precisely the axis LeCun says matters. We **validate his metric**: when you measure on the axis he advocates, structure *does* something; on the asymptote axis (benchmarks), it does not. This is a clean empirical endorsement of "stop scoring on asymptote."

**Does NOT refute (the hard half — and we must say so loudly):**
- LeCun's actual bet is **HIERARCHICAL / multi-timescale (H-JEPA)**, not single-level abstraction. Our nulls are **single-level**: we varied the abstraction *class* at one level of the world model. **A single-level null says nothing about whether stacking multiple abstraction levels at multiple timescales helps** — which is exactly the mechanism LeCun claims is load-bearing (subgoal emission, long-horizon credit assignment, coarse-far/fine-near prediction). We should explicitly disclaim over-reach here; it is the most likely reviewer attack and the honest boundary of our claim.
- Our setting is **value-sufficient by construction** (the latent already supports the value function). H-JEPA's claimed payoff is largest exactly where value-sufficiency is *hard to obtain* — long-horizon, sparse-reward, compositional tasks where a flat latent struggles (cf. Director's sparse-reward 3D-maze wins). Our domains may be too "easy" to surface the hierarchy benefit.

### 6.3 The precise open question + our differentiator

> **Open question.** Does abstraction confer a *durable asymptotic* advantage — or only a sample-efficiency advantage — when it is **hierarchical and multi-timescale** (H-JEPA-style), under a **matched-control, speed-of-learning** evaluation, on tasks chosen to be hierarchy-sensitive (long-horizon / sparse-reward / compositional / OOD)?

**Our differentiators (what we bring that the JEPA/HRL literature lacks):**
1. **Rigorous matched control.** The JEPA and HRL papers compare *systems* (different encoders, objectives, training recipes confounded together) and report asymptote on benchmarks. We hold the value-relevant latent fixed and vary *only* the abstraction structure — so a difference is attributable to abstraction, not to incidental capacity/recipe changes.
2. **Speed-of-learning as the primary axis, operationalized.** Following Chollet (skill-acquisition efficiency) and the Atari-100k tradition, we make learning-rate the dependent variable, not a footnote — directly testing LeCun's own metric rather than the asymptote metric he criticizes.
3. **A structural (not statistical) information surrogate.** Given the McAllester–Stratos / Poole impossibility of certifying high MI from samples (§3.3), we use **structural entropy** on the latent interaction graph as a tractable, assumption-light abstraction objective — a principled alternative to variance/covariance surrogates whose limits (InfoNCE `log K` ceiling) are known.
4. **The hierarchical extension as the decisive test.** We extend the matched-control protocol *up the hierarchy*: same fixed value latent, vary number/structure of abstraction levels and timescales. This is the test that can actually adjudicate LeCun's H-JEPA bet — and which neither the JEPA papers (no matched control, no speed-of-learning) nor the HRL papers (no value-sufficiency control, confounded systems) currently provide.

---

## 7. Positioning statement (1 paragraph, for the next paper)

> Yann LeCun's JEPA / autonomous-machine-intelligence program rests on two distinct bets: (i) that world models should be **action-conditioned predictors in an abstract, non-generative representation space**, and (ii) that the payoff of such abstraction is **hierarchical and multi-timescale** (H-JEPA) and should be **measured by speed of learning, not benchmark asymptote**. Our prior negative campaign — a matched-control study on a value-sufficient TD-MPC2/SimNorm latent showing that *single-level* representational abstraction (monolithic / token / entity-graph, with or without a structural-entropy objective) is **redundant for asymptotic performance in- and out-of-distribution, and valuable only for sample efficiency** — *confirms* LeCun's metric (abstraction buys learning speed, exactly the axis he says counts) while leaving his core architectural bet *untested*: a single-level null cannot refute a claim about multi-level, multi-timescale hierarchy. This paper closes that gap by extending rigorous matched-control evaluation **up the abstraction hierarchy under an explicit speed-of-learning protocol**, on hierarchy-sensitive long-horizon/sparse-reward tasks, using a tractable **structural** information surrogate motivated by the formal impossibility of certifying mutual information from samples (McAllester–Stratos; Poole et al.). We thereby provide the first apples-to-apples test of whether hierarchical abstraction delivers a *durable* advantage or, like its single-level counterpart, only *accelerates* learning toward a ceiling that flat representations also reach.

---

## 8. Bibliography (key papers, one-line relevance each)

**LeCun program / JEPA**
- LeCun (2022), *A Path Towards Autonomous Machine Intelligence*, OpenReview `BZ5a1r-kVsf` — the vision paper defining JEPA, H-JEPA, and the world-model module; the thesis we position against.
- Assran et al. (2023), *I-JEPA*, arXiv `2301.08243`, CVPR 2023 — first JEPA: predict image-region *representations*, not pixels.
- Bardes et al. (2024), *V-JEPA / Revisiting Feature Prediction for Video*, arXiv `2404.08471` — feature prediction as the sole objective for video; collapse-avoidance + EMA target.
- Meta/FAIR (2025), *V-JEPA 2*, arXiv `2506.09985` — action-conditioned latent world model; zero-shot robot planning to image goals; the concrete realization of LeCun's bet.
- (Secondary) *Introduction to Latent Variable Energy-Based Models*, arXiv `2306.02572` (2023), IOP JSTAT 2024 — restates the position paper as an LV-EBM. ⚠ verify authorship/venue.

**Energy-based foundations**
- LeCun, Chopra, Hadsell, Ranzato, Huang (2006), *A Tutorial on Energy-Based Learning*, MIT Press — EBM = scalar energy, no normalization needed; the "cost/energy unifies all model types" thesis; JEPA's energy = latent prediction error.

**Anti-collapse / SSL objectives**
- van den Oord, Li, Vinyals (2018), *CPC / InfoNCE*, arXiv `1807.03748` — contrastive prediction; InfoNCE is an MI lower bound (and its ceiling motivates §3.3).
- Zbontar, Jing, Misra, LeCun, Deny (2021), *Barlow Twins*, arXiv `2103.03230`, ICML 2021 — anti-collapse via cross-correlation → identity (redundancy reduction).
- Bardes, Ponce, LeCun (2021), *VICReg*, arXiv `2105.04906`, ICLR 2022 — explicit variance/invariance/covariance regularization against collapse; the V-JEPA regularizer family.
- Grill et al. (2020), *BYOL*, arXiv `2006.07733`, NeurIPS 2020 — collapse avoided by predictor + EMA target + stop-grad (no negatives); the mechanism I-/V-JEPA inherit.

**Information-measurement limits**
- Belghazi et al. (2018), *MINE*, arXiv `1801.04062`, ICML 2018 — neural MI estimation (Donsker–Varadhan); high variance, loose at high MI.
- Poole et al. (2019), *On Variational Bounds of Mutual Information*, arXiv `1905.06922`, ICML 2019 — unifies MI bounds; InfoNCE capped at `log K`; bias–variance tradeoff.
- McAllester & Stratos (2020), *Formal Limitations on the Measurement of Mutual Information*, arXiv `1811.04251`, AISTATS 2020 — distribution-free high-confidence MI lower bounds are `O(log N)`; the impossibility motivating a *structural* surrogate.

**Hierarchical RL / planning**
- Sutton, Precup, Singh (1999), *Between MDPs and semi-MDPs (Options)*, Artif. Intell. 112 — formal substrate for temporally extended action.
- Vezhnevets et al. (2017), *FeUdal Networks*, arXiv `1703.01161`, ICML 2017 — Manager sets latent subgoals, Worker executes; closest classical analog to H-JEPA.
- Nachum, Gu, Lee, Levine (2018), *HIRO*, arXiv `1805.08296`, NeurIPS 2018 — off-policy, sample-efficient goal-conditioned HRL (raw-state goals).
- Hafner, Lee, Fischer, Abbeel (2022), *Director / Deep Hierarchical Planning from Pixels*, arXiv `2206.04114`, NeurIPS 2022 — hierarchical planning in a *learned latent* world model; the key existing realization (on a generative WM) and a needed baseline.

**Speed-of-learning / efficiency-as-intelligence**
- Chollet (2019), *On the Measure of Intelligence*, arXiv `1911.01547` — intelligence = skill-acquisition efficiency; rigorous anchor for our metric.
- Finn, Abbeel, Levine (2017), *MAML*, arXiv `1703.03400` ⚠ — optimize for fast adaptation; "learning to learn fast." (verify ID)
- Ye et al. (2021), *EfficientZero*, arXiv `2111.00210` ⚠ — sample-efficiency-at-fixed-budget as the figure of merit (Atari-100k). (verify ID)
- Ha & Schmidhuber (2018), *World Models*, arXiv `1803.10122` — origin of modern "world model"; justified by learning efficiency in imagination.

**Our substrate / contrast world models**
- Hansen, Su, Wang (2024), *TD-MPC2*, arXiv `2310.16828`, ICLR 2024 — value-sufficient, decoder-free latent world model (SimNorm); the substrate of our matched-control study.
- Hafner et al. (2023), *DreamerV3 / Mastering Diverse Domains through World Models*, arXiv `2301.04104` — the leading *generative* world-model line; the empirical counterweight to LeCun's anti-generative claim and the base Director builds on.

---

## 9. Honesty / risk flags for the authors
- **⚠ Quote-anchor needed:** LeCun's exact "measure intelligence by speed of learning" phrasing should be tied to a dated, quotable source (position paper section + a specific public talk) before camera-ready; do not attribute a crisp sentence to a paper that only implies it.
- **⚠ Unverified arXiv IDs:** MAML (`1703.03400`), EfficientZero (`2111.00210`), and the LV-EBM restatement (`2306.02572`) were cited from memory/secondary results — verify each directly before submission.
- **Strongest reviewer attack to preempt:** "single-level null ≠ hierarchy refutation." We must state the scope limit explicitly (done in §6.2) and make the hierarchical extension the headline contribution, not an afterthought.
- **Counterweight to acknowledge:** the Dreamer/Director line shows *generative* latent world models already do hierarchical latent planning competitively — so "abstract non-generative latent" is not yet empirically established as *necessary*; our framing should be "test the bet," not "assume LeCun is right about generativity."
- **Crowded derivative space:** numerous 2026 JEPA-derivative preprints exist (causal-JEPA, variational-JEPA, end-to-end-from-pixels JEPA, etc.); a fresh related-work sweep at submission time is mandatory.
