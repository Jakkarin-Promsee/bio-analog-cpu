# draft 6.0 — operating context (the first organ — DONE & FOLDED)

> 🔻 **NOT the live line any more (2026-08-05).** P1–11 completed and froze the neocortex organ; the author then
> **set zero** and opened [`../draft7.0/`](../draft7.0/README.md) to find the **loop of thought** core first.
> Draft 6.0 is **folded, not discarded** — it is picked back up once a loop core exists. Before proposing any new
> draft-6 work, read [`../draft7.0/docs/handoff.md`](../draft7.0/docs/handoff.md) (why the turn happened, and the tripwires).
> Two draft-6 claims are now **struck**: goodness-as-the-correctness-feeling and tap-drift-as-the-feeling — the
> north-star dossier below still contains the old wrong version.
>
> Auto-loads when you work in `draft6.0/`. This is the **draft's mental model + current status + map** — the file
> that changes as work advances (the root `CLAUDE.md` stays stable). Cold-start narrative: [`context.md`](context.md).
> Reading budget: load a phase's `phaseN/README.md` for its gist; open cards/code only to modify them; heavy
> sweeps → dispatch the `Explore` sub-agent.

---

## Architecture in one breath

Two brains on one analog substrate. A cheap, unsupervised **SCFF** front (~80% — Self-Contrastive Forward-Forward:
local, label-free, forward-only) organizes the world for free as a **single frozen bulk**, and a small precise
**namer** back (~20%) puts our labels on that structure — because **direction is the one expensive thing in
learning, so we pay for it once, where it counts.** The namer **reads, never writes** the bulk (per-depth read-only
heads) and — the twist the experiments delivered — needs **no gradient descent at all**: it is **closed-form**
(SLDA / RanPAC). Naming is **drift-gated** (SCFF runs every step; the namer fires only when a drift gate trips, and
that gate is a *safety* mechanism — firing more forgets more) and **sleep-consolidated** (a periodic closed-form
re-fit over a **hippocampus LUT** prototype memory). It runs via **mono-forward** — one forward sweep carrying a
positive + negative world side by side through a shared weight crossbar (only the cheap activation buffers double,
not the Scaps). *Why 6.0 exists, in full:* [`README.md`](README.md); the **soul** (the *why* under the why): [`../docs/essence/the-essence2.md`](../docs/essence/the-essence2.md).

---

## Where we are — current status (the shifting part: edit this, not the root)

**Stage 1 is done and frozen (Phases 1–6): the SCFF cheap brain composes depth and survives noise. Stage 2 (the GD
*namer*) is DONE — Phases 7 (readout), 8 (economy + cost), 9 (maintenance — the loop FROZEN), and 10 (validation /
showcase — the frozen object raced against a fair BP+replay baseline) are all complete. Stage 2 build = Phases 7–9, the validation tier = P10–11; "freeze in
P9, judge in P10." The verdict (S14): a substrate-native continual learner — competitive on the continual home, decisively
safer, far more noise-robust, its energy edge over conventional GD substrate-realized; the founding bet refined, not
inflated. The neocortex brain is done. **Phase 11 — the limit map (real data + scale, the red-team answer) is also DONE
(ran 2026-07-05, P11.0→P11.9, S15): the validated object across 8 real arenas × 5 capability channels, wins/losses/
floors all shipped — [`src/phase11/README.md`](src/phase11/README.md).** Next build: the **hippocampus organ** — grow
the LUT stand-in into a learning brain part, the first north-star step (author's direction 2026-07-05; the north star
is now stated publicly in the root README). The analog-constraint (SPICE-grade, **simulation-only**) realism pass stays
queued for when a result needs it — the project identity is the *math model under analog constraints*, not circuit
drafting; no fabrication is planned. Beyond: the recurrent lifelong brain (the north star).** One-glance ladder — the depth lives in each `src/phaseN/README.md`, not here:

- **P1 ✓ structure** — the cell works; its home is the **continual** regime (a periodic sleep recovers what online backprop catastrophically forgets). Generalizes better than backprop at ~10% backward cost; *not* a deep static-accuracy competitor.
- **P2 ✓ depth round 1** — a deep SCFF stack can't earn depth (transmission *and* a perfect-oracle objective both fail). Depth comes from **boosted ensembles of shallow blocks with tiny GD readouts**, not deep SCFF.
- **P3 ✓ depth round 2 — ADOPTED** — the wall was the **energy objective `Σh²`**, not locality. **Contrast (InfoNCE) + a cross-layer coordination window** compose depth and re-earn the continual win — this **supersedes energy-goodness**.
- **P4 ✓ characterization** — the capability map vs *genuinely-tuned* backprop: **a substrate-native continual learner, not a static-accuracy competitor.** WINS continual / nuisance-dim / depth-composition / depth-is-cheap (the 80/20 cost win is depth-gated); TRAILS static difficulty / class-count; one honest NEGATIVE on eval-time weight noise.
- **P5 ✓ SCFF close-out — depth SOLVED (scoped)** — the decay was **objective-locality, not an intrinsic Tunnel** (full-credit w12 composes the whole stack). Two *free* levers fix it: **sharper InfoNCE temp 0.2** earns the depth back (direction, not lr — ~82% of the lift; the readout beats tuned-BP, the probe tail reaches the w12 ceiling) and **per-depth heads + a short fixed reader** read the continual home **8× cheaper** than all-tap. **Continual-safe** (A6 intact, BWT −0.026) and **real-data-confirmed** (digits tail +0.152; null-but-safe on CIFAR-flat, which has no composable depth). Committed cell: **`SCFFContrastOverlap` temp0.2 / w2, L12 bulk, NO residual, fixed-reader deploy** (truncate ~L2–3 on the home · all-tap for peak acc · w4 = bounded depth-closer). Adaptive early-exit and the frozen residual were **struck**. *The cheap brain is done.*
- **P6 ✓ noise-robust SCFF — Scoped-YES** (ran 2026-07-01, P6.0→P6.8) — A7 (eval-time noise) is real, **OURS-specific & directional** (0.60 vs a linear readout 0.96, 5/5). The primary fix = **generic noise-augmentation** (corrupt one InfoNCE view with broad iid noise, σ_aug=1.0): it **substantially but partially** hardens the Rasch-dominant tap channel (retention 0.817→0.865, 5/5 paired; near the 0.90 band), *improves* clean acc, and **passes the A6 continual gate** (BWT −0.022→−0.017). The **input-transducer directional residual** → Stage-2 read-side (the "scoped"). Sims **overturned** two design guesses: the fix is **generic, not directional-specific** (iid ≥ randax > dir), and the directional enemy is a **coherent translation** (retention, not per-sample cos, is the direction read). Door B (all-data-is-noise) resolved: the direction forms from a fully-noisy stream. Committed cell: **`NoiseAugContrast` = frozen P5 cell + one iid-noise-augmented view**. Front door: [`src/phase6/README.md`](src/phase6/README.md).
- **P7 ✓ the readout / namer — the 20% is NOT gradient descent** (ran 2026-07-02, P7.0→P7.6) — the committed namer is **RanPAC** (closed-form random-projection + running-Gram ridge prototype) **+ a class-balanced-reservoir guard**: no gradient, it ties the gradient MLP anchor on accuracy×BWT (3-way tie with SLDA), leads on natural digits (#1, 0.949), **passes the A6 gate** (the trained cosine-softmax + max-magnitude FeCAM are struck), most spine-clean of the tied cluster. **The spine bends** (magnitude-wins-spine-bends, Δ=0.128 synthetic → −0.036 digits → ≈0 CIFAR: cosine is spine-clean but sub-competitive where the bulk has structure). Sims **overturned** two guesses: the imbalance guard is **cbrs, not AIR**; the "multimodality cliff" is **anisotropy (a tied covariance), not multimodality**. Decision-record delta **S11** (N3 superseded / S4 collapsed / S9 extended). Cost → P8 (**SLDA** = the ~200× cheaper no-grad fallback). Front door: [`src/phase7/README.md`](src/phase7/README.md).
- **P8 ✓ the economy + the cost meter — the two-brain loop, run LIVE (🔥)** (ran 2026-07-02, P8.0→P8.6) — both brains live for the first time (SCFF learns forward-only; the namer tracks the drift via a gate + sleep, on a new streaming `partial_fit`). The committed economy: **SLDA** deployed (metered **69× cheaper** than RanPAC, AA ties live → resolves S11's cost caveat) · **DDM** awake gate on the namer's **error-EMA** (deployed) · a **class-direction tap-drift** trigger validated but **not shipped** (P8.2: leads error, spine-clean; DDM consumes an error rate → it's the north-star upgrade) · **grid-8/full-history** sleep. The metered 80/20 is **real** (GD-share **0.121** ≤ 0.25; the gate *creates* it — off it is 0.778); OURS ≈ **half** the energy of BP+replay (bp_ratio 0.501); and it is **LIVE-SAFE** (worst-point BWT **0.000**, 0/5 regress). The crux: **firing more forgets more** (always-pay −0.137) — the gate is a *safety* mechanism. Delta **S12**. **+ P8.7 (why-analog substrate ablation, added 2026-07-02):** the full 2×2 {OURS,GD}×{analog,digital} — the chip is **15.4× cheaper** than conventional GD-on-digital = **5.4× substrate** (compute-in-memory) **× 2.9× algorithm** (the 80/20); analog advantage is a floor (≥2.7×, →53× with the memory wall), the 80/20 is substrate-independent. Front door: [`src/phase8/README.md`](src/phase8/README.md).
- **P9 ✓ close & *freeze* the maintenance loop — FROZEN (🔒)** (ran 2026-07-02, P9.0→P9.5) — the five open knobs of the lifelong loop, tuned against *internal* signals then locked. The bulk **rotates but does not forget** (P9.0 — re-fit destruction ≥ birth; the founding cheap-replay assumption measured, N2 *not mandatory*). Four knobs kept the committed loop: **N2 struck** (P9.1 — no lever on rotation-only drift), **all-tap** consolidation (P9.2 — trunc forgets more), **CBRS** eviction (P9.3 — best-bounded, ties the herding buffer-spine null, beats iid/FIFO; cap grows with #classes), **proto-reanchor** read-side (P9.4 — the sleep mechanism recovers the Phase-6 input-transducer residual 0.79→0.99). And the freeze caught the one real gap: the P8 **grid-8** cadence was too sparse for a lifelong revisit stream (failed the oracle-veto 2/5, worst-BWT −0.317); the freeze-driven cadence re-confirm → **grid-4** restores near-flat safety (worst-BWT **−0.028**, ties oracle 0/5, AA 0.494, GD-share 0.178 ≤ 0.25). Sims **overturned** two guesses: the worst-point gap was closed by denser *sleep* not the (committed) gate; the read-side defense needed no covariance re-fit. Delta **S13**; the object is locked for P10. Front door: [`src/phase9/README.md`](src/phase9/README.md).
- **P10 ✓ validation / showcase — the Stage-2 close-out (🔥)** (ran 2026-07-03, P10.0→P10.6, all 14 guards bit-exact; grid-4 bit-for-bit vs the P9 freeze) — the **frozen** object raced (it touched no knob) against a **fair, budgeted, tuned BP+replay** baseline; verdict shapes pinned **BLIND**. **Verdict (S14): a substrate-native continual learner, the founding bet refined not inflated, banked as two halves.** *Accuracy* — ties ER-strong on the continual home (0.494 vs 0.498) but trails on natural digits (0.879 vs 0.950; OURS is continual, not static — P4 re-confirmed), while **winning continual safety** (worst-BWT −0.028 vs −0.272, ≈10× safer; gauntlet retention 0.490 vs 0.350) and **noise** on every held-out channel (0.92–1.10 vs BP 0.23–0.61). *Economics* — **substrate-realized** (the 80/20 algorithm is 1.5× *more* than a small tuned ER same-substrate — the honest R1; the chip is 3.4× cheaper via the analog crossbar floor). On an (acc × energy) Pareto a small tuned ER dominates OURS; OURS's wins live off that axis (safety/noise/substrate). The cadence cost-frontier `{4,5,6,8,12,16}` + §10 probes {7,13,14,15} (grid-4 headline, no swap; the four-round §10 extension localized the Tier-2 break as a **plateau + two cliffs** — the safety plunge at 6→7, the accuracy cliff at 15→16 where equal-count sleeps turn timing-sensitive — plus the per-batch GAUNTLET-STREAM views: forward, ER saw-tooths at every domain switch while OURS rides flat; **reversed (noised first), ER collapses 0.504→0.343 while OURS is order-invariant**; **alignment-break (round 3): sleep/boundary alignment is a NON-FACTOR for OURS (E8b paired gap +0.002) while ER re-converges on long stationary blocks 0.504→0.675 → the gauntlet retention win is switch-frequency-scoped**; **reversed-long (round 4): the REV staircase's sag = the namer frame going stale between sleeps — mid-domain sleeps rescue it (5/5) and OURS is order-invariant at length (0.527 vs 0.533), banked supported-not-confirmed (the confirming cut mis-specified; bulk-level component flagged)** — and the all-grid Pareto money line, numbered-point encoding; measurement-only, every carried array bit-exact) + SCFF:Namer ratio (0.107–0.178) characterized; a small directional/ADC residual **named → the analog layer**, and the **noise-first thin-margin limit** named → the next capability (a bulk that recovers clean structure *itself* from an all-noisy stream). Front door: [`src/phase10/README.md`](src/phase10/README.md); the close-out: [`src/stage2-report.md`](src/stage2-report.md).
- **P11 ✓ the limit map — real data + scale (the red-team answer; S15)** (ran 2026-07-05, P11.0→P11.9; committed core + all three scaling extensions; Arm A bit-equal committed, Arm B a pre-registered scaling rule, nothing tuned; ER re-tuned per arena; shapes pinned BLIND) — the validated object taken off the toy bench onto **real streams + scale**, drawn as a **LIMIT-MAP** (8 arenas × 5 channels, win/tie/loss/FLOOR). The three reviewer strikes answered: **(1) decomposition** — Δbulk tracks arena nonlinearity (+0.417 synth-home, beats a random 12-layer reservoir → *learned*; ≈0 near-linear digits), and the continual **safety is the closed-form namer+gate+sleep, not the bulk** ("is it just SLDA?" → NARROWED-with-mechanism); **(2) real data** — WINS nature's **gas** sensor drift (prequential 0.789 ≥ a per-arena-tuned ER 0.756, beats persistence +0.184; Arm B 0.856), holds real-MNIST safety/order-invariance (static AA trails — continual-not-static), cross-dataset **type-robust** (all 3 types alive, order-invariant across types, Arm B ≥ ER retention), and honestly **FLOORS** the autocorrelated streams (HAR/electric/covtype — the ELEC2 persistence trap, field +0.07); **(3) scale** — the bench-**pinned GD-share economy shape CONFIRMED** (economy doesn't improve with width), the **analog substrate factor GROWS 5.4→7.4×**, the prototype+Gram namer **out-retains byte-matched replay by C=20** (replay dilutes, the namer does not), gas **throughput regime win** (arena-dependent) — while static accuracy trails and CIFAR-gray floors (resolution). Losses + floors ship in the map. Front door: [`src/phase11/README.md`](src/phase11/README.md).

> Status edit rule: finishing a phase updates **one line above** + adds a `phaseN/CLAUDE.md` signpost. The root never changes for this.

---

## The build discipline (governs every rung)

- **Walk one spine — the neocortex (SCFF + GD).** Build straight down the ladder; don't open a second track.
- **The hippocampus LUT is a service, not a parallel brain.** It feeds SCFF its *negatives* (stub first — a random batch partner) and holds the *replay history* for sleep. Never a milestone of its own.
- **Test convergence, not theory.** Each rung's result is one picture, not an argument. The "so many papers" feeling is the north-star menu ([`research/north-star/`](research/north-star/README.md)) bleeding in — **keep that menu closed.**

---

## The decision record

The live record is [`idea/main.ideas.v1.md`](idea/main.ideas.v1.md) (N1–N3 approved + S1–S15 supporting); the full
derivation (story form) is [`idea/ideas1.md`](idea/ideas1.md); the folder index + decision timeline is [`idea/README.md`](idea/README.md). **6.0 is young** — the spine is committed but treat
decisions as *settling*, not frozen the way draft-5.1's "14 locked decisions" were presented; the open knobs are
listed there and the sims set them. What carried forward from the old world *in spirit*: the two-timescale
Cortex/Hippocampus (now sleep + the LUT), and resident-weight / sign-as-digital / the Scap (substrate-level,
unchanged). *(Residual/boosting was explored early — P2 — but the committed cell dropped it: single bulk, a
reading namer, S11.)*

---

## Scope (what's in, what's out)

- **In:** numpy behavioral simulation of the draft-6.0 hybrid on simple classification / statistics tasks; ideal model first, analog/PVT realism only after the ideal converges.
- **Out (near term):** SPICE, fabrication, external-benchmark-chasing *as the claim* (small tasks are fine as probes).
- **Beyond the numbered phases — the north star (deliberately not specced):** a recurrent, lifelong-learning prefrontal↔hippocampus "thinking" loop where *correctness is a self-generated feeling*. The real long-term target — but **simple intelligence first.** ⚠ Updated 2026-07-05: the author put the north star **into the root README** as the stated destination (and the hippocampus organ as the next build) — it is public context now. Still don't pull it into the live *experiment* plan beyond that without the author's direction.
- **Triage new ideas:** does it test in the current sim ladder, or is it a later-phase / future track? Catch scope-creep early — but 6.0's spine is still settling, so "promotion" is lighter-weight than 5.1's frozen process.

---

## File map (draft 6.0)

```
draft6.0/
  CLAUDE.md        this file (auto-loads in-draft)
  README.md        the pivot story (why 5.x died, what 6.0 is)
  context.md       ★ full cold-start dump (what / why / how / the person)
  idea/            the design (index: idea/README.md)
    README.md          the idea index + decision timeline (ideas1 → P1-6 → gd-replan → P7-11)
    ideas1.md          the origin blueprint (full derivation, story form)
    main.ideas.v1.md   the decision record (N1–N3 + S1–S15) — the committed plan
    phase4-opinion.md · phase4-problem.md · gd-replan/   the GD-era development of ideas1
  research/        the reading, by role:
    survey/        learning-rule survey (the options considered)
    papers/        paper stories — phase1-2/ (the adopted design) + phase3/ (the depth reframe)
    north-star/    the north-star dossier (21 files + th/ Thai mirror) — beyond the phases, not the live line
  src/             ★ the results (index: README.md) — Stage-1 + Stage-2 report sets + run-specs
    README.md          ★ the src front door — the arc map + the document-system key
    phase9-final-architecture.md  the v2.0.0 WHOLE neocortex in one file (both brains, frozen — validated by P10)
    phase6-final-architecture.md  the v1.1.0 cheap-brain-only snapshot (SCFF alone, noise-hardened)
    phase5-final-architecture.md  the v1.0.0 ideal-data snapshot (kept as the pre-noise base)
    stage1-report.md   volume 1 — the six-phase executive arc (P1–6, the cheap brain built + hardened)
    stage2-report.md   volume 2 — the build arc of the namer (P7–9: readout · economy · freeze; decisions only)
    validation-report.md  volume 3 — the frozen object on trial (P10 fair race + P11 limit map; measurements only)
    result-format.md   the canonical house style (figures · metrics · the 6-slot template)
    ref-report/        glossary the reports cite (methods · metrics · papers)
    phaseN/            per phase: README.md (front door / synthesis) · design.md (the pre-run design) ·
                       CLAUDE.md (signpost) · phaseN-report.md · RESULTS.md · result-format.md (delta) · expK/ cards · figs
```

---

## Routing — draft-6 questions

| When the user asks… | Look in |
| --- | --- |
| **The whole model in one file (v2.0.0, current head)** | [`src/phase9-final-architecture.md`](src/phase9-final-architecture.md) — the self-contained account of the **complete two-brain neocortex, frozen** (both brains + the maintenance loop); understand the model + the 9-phase arc without opening any phase report. *(v1.1.0 cheap-brain-only snapshot, kept: [`src/phase6-final-architecture.md`](src/phase6-final-architecture.md); v1.0.0 ideal-data base: [`src/phase5-final-architecture.md`](src/phase5-final-architecture.md).)* |
| The plan / what we're doing now | [`idea/main.ideas.v1.md`](idea/main.ideas.v1.md) (decisions) + [`idea/ideas1.md`](idea/ideas1.md) (the story) |
| The written results / the report volumes | [`src/stage1-report.md`](src/stage1-report.md) (P1–6) · [`src/stage2-report.md`](src/stage2-report.md) (P7–9) · [`src/validation-report.md`](src/validation-report.md) (P10–11, the trial) → `src/phaseN/phaseN-report.md` (the audit depth) → [`src/ref-report/`](src/ref-report/README.md) (glossary) |
| What did Phase N find? | `src/phaseN/README.md` (front-door synthesis) + `RESULTS.md` (the scalar ledger) |
| Code / read a Phase-N experiment | `src/phaseN/` — start *in* that folder to load only its context; `expK/experiment-K.md` cards; `pNlib.py` apparatus |
| Draw a figure / write up a result | [`src/result-format.md`](src/result-format.md) (canonical house style + base figure catalog; per-phase additions in `src/phaseN/result-format.md`) |
| The papers behind a decision | [`research/papers/`](research/papers/README.md) — `phase1-2/` (design) · `phase3/` (the reframe) · `phase5/` (the depth-readout) |
| The learning-rule survey | [`research/survey/summary.detail.md`](research/survey/summary.detail.md) |
| The north-star / recurrent-brain research | [`research/north-star/`](research/north-star/README.md) — free-time reading, *not* the live plan |

---

## What's historical (draft 5.x)

The draft-5.1 spec, the §20 simulation campaign, the Ganglion-Personality / MVF work, and the `draft5.0/src/`
simulator were built for the **attribution** architecture — superseded (the learning rule distributed loss
*magnitude* but never *direction*). Substrate primitives (Scap, the crossbar) may carry forward; the learning rule
and Ganglion-hierarchy do not. See [`../draft5.0/CLAUDE.md`](../draft5.0/CLAUDE.md) and, for *why* old calls were
made, [`../docs/draft/project-history.md`](../docs/draft/project-history.md).
