# Draft Hierarchy

A map of how each draft evolved — what changed, why, and what role each file plays.

For the narrative version of the same arc (mind stones, breakthroughs, why each pivot happened), see `project-history.md`. This file is the file-by-file index.

> **Draft-6.0 note.** This map now covers **drafts 1.0 → 6.0.** Versions 1–5 (below) are the *attribution era* —
> historical, superseded. **[Version 6](#version-6--the-scff--gradient-descent-rebuild-draft-60)** (near the end) is
> the **current** structure — the SCFF + gradient-descent rebuild, which is a *whole project tree*, not a single
> document. Start at [`draft6.0/context.md`](../../draft6.0/context.md) or
> [`draft6.0/README.md`](../../draft6.0/README.md). The narrative version of the same arc is
> [`project-history.md`](project-history.md).

---

## Version Snapshot

| File                   | Type              | One-line summary                                                                                                                                |
| ---------------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `draft1.0.md`          | Main              | Raw brainstorm — personal narrative, first architecture sketch, unfinished                                                                      |
| `draft1.5.md`          | Main              | Grammar/clarity pass on draft1.0, no new architecture                                                                                           |
| `draft1.0.q1.md`       | Q&A               | Answers to 12 clarifying questions on v1                                                                                                        |
| `draft2.0.md`          | Main              | First formal design — renamed components, operator layer, scope locked to Python sim                                                            |
| `draft2.1.md`          | Main              | Working copy of draft2.0, no structural difference                                                                                              |
| `draft2.2.md`          | Main              | draft2.0 + community commentary and extended ideas                                                                                              |
| `draft2.3.x1.md`       | Brainstorm        | Deep dive into analog storage alternatives, stages 0–4 (deferred)                                                                               |
| `draft2.3.x2.md`       | Brainstorm        | Continuation of x1, stages 5–8 (deferred)                                                                                                       |
| `draft3.0.md`          | Main              | Major rewrite — distribution-measurement, decentralized per-scap update                                                                         |
| `draft3.0.q1.md`       | Q&A               | Reactions to v3 feedback — hierarchical diffusion and SpecialGeneralist first introduced                                                        |
| `draft3.1.md`          | Main              | Hierarchical diffusion formalized, Brainstem named, two-timescale Limbic Loop, first §22 lock list                                              |
| `draft3.1.q1.md`       | Q&A               | Q&A on draft3.1 — early red-pen feedback before draft3.2                                                                                        |
| `draft3.2.md`          | Main              | Honest-framing pivot — attribution-based learning replaces "approximate backprop" claim                                                         |
| `draft3.2.q1.md`       | Q&A               | Q&A on draft3.2 — discussion of the attribution vs gradient framing                                                                             |
| `draft3.3.md`          | Main              | Pulse_width vs momentum disambiguation; per-Column independent buses                                                                            |
| `draft3.3.r1.md`       | Reflextion        | Reflection on draft3.3 before opening draft4                                                                                                    |
| `draft4.0.md`          | Main              | Five-level hierarchy locked — **Lobe added**, multi-branch DAG, residuals as first-class defense                                                |
| `draft4.0.x1.md`       | Brainstorm        | Exploratory digression around draft4.0                                                                                                          |
| `draft4.1.x1.md`       | Brainstorm        | External-critique notes that fed into draft4.1                                                                                                  |
| `draft4.1.md`          | Main              | External critique integrated — Forward_Sign_SRAM, Physical Saturation, §15 Analog Robustness, PGA                                               |
| `draft5.0.md`          | Main              | Structural rewrite for stranger reader — mechanism (§2) before modules, §3 worked example                                                       |
| `draft5.0.x1.md`       | Brainstorm        | Brainstorm during the draft5.0 restructure                                                                                                      |
| `draft5.1-full.md`     | Main (ref)        | Final polish — §1.4 bio narrative, §3 XOR-convention bug fix, §3.6 ambiguity resolved (unsplit)                                                 |
| `draft5.1-1.md`        | Main (Part)       | Canonical spec Part 1 — §0–§19, §22–§23. **Now in `draft5.0/` (moved at the draft-6.0 pivot; was at repo root).**                               |
| `draft5.1-2.md`        | Main (Part)       | Canonical spec Part 2 — §20 simulation, §21 future tracks. **Now in `draft5.0/`.**                                                              |
| `draft5.1-2.verify.md` | Plan (superseded) | Re-draft of the §20 plan, intuition-first, phase by phase. Was the live plan before the draft-6.0 pivot; now history (live plan = `draft6.0/`). |

> The split into `draft5.1-1.md` / `draft5.1-2.md` exists only because the markdown-to-PDF tool fails on the unsplit length. Content is identical to `draft-journey/draft5.1-full.md`. Section numbers are continuous across both parts.

---

## Naming Convention

| Pattern           | Type            | Purpose                                                                              |
| ----------------- | --------------- | ------------------------------------------------------------------------------------ |
| `draftX.X.md`     | Main design doc | Core architecture document for that version                                          |
| `draftX.X.rN.md`  | Reflextion      | Deep reflection and new ideas on the same main draft                                 |
| `draftX.X.xN.md`  | Brainstorm      | Deep exploratory digression; not merged into main design                             |
| `draftX.X.qN.md`  | Q&A             | Responses to questions on a previous draft                                           |
| `draftX.X-N.md`   | Part            | Part N of a main draft, split only for PDF-export length; no content difference      |
| `draftX.X-full.md`| Main (unsplit)  | The pre-split version of a main draft, kept as the single-file reference             |

---

## Version 1 — Raw Intuition

### `draft1.0.md` — Initial Brainstorm

**Purpose:** First attempt to put the idea on paper. Written as a personal narrative mixed with rough architecture notes.

Key content:

- Personal journey: ML/DL → neuromorphic circuits → biology → analog CPU
- Core intuition: von Neumann architecture is wrong for neural computation
- First naming: NCN, NBN, BRN, CNS, DSP, STM
- First concept of **scap** (SRAM + Capacitor) as distributed weight storage
- First sketch of 2-3-3-2 topology
- Unfinished sections (Cell Fire, Workflow Ideas)

### `draft1.5.md` — Grammar Pass

Cleaned English and formatting of draft1.0. No new architecture.

### `draft1.0.q1.md` — Q&A on v1

Twelve clarifying questions. Notable answers:

- "Encryption" means on-chip weight compression, not cryptography
- Success metric is internal convergence, not external benchmarks
- scap history bit implicitly captures pre+post activity
- L1 reuse: same physical Ganglion called under different specialist contexts
- Hierarchy sketch: Ganglion → Column → Limbic Loop

---

## Version 2 — Formalization

### `draft2.0.md` — Full Formal Design

First complete, structured research document. Locks scope and names.

- Renamed components: NCN → **Ganglion**, NBN → **Column**, BRN → **Limbic Loop**
- Defined the **operator layer** (op-amp ALU primitives)
- Introduced **Routes hierarchy** (4-bit L1 × 4-bit L2 = 256 Ganglia per region)
- Formalized **5 hypotheses (H1–H5)** and **6-phase simulation plan**
- Scope locked: **Python simulation only**, not SPICE

### `draft2.1.md` — v2 Copy

Working copy of draft2.0. No structural difference.

### `draft2.2.md` — v2 + Community Commentary

Extends draft2.0 with expert questions and the author's extended responses. Adds 5 "Ideas" sections (asymmetric timing, differentiated objectives, Synaptic Drift as analog position, initialization strategy, skip connections) and an "HX — Missing pieces" section (loss function, output decoding, batch processing, inference/training modes).

### `draft2.3.x1.md` / `draft2.3.x2.md` — Brainstorm: Analog Storage Stages 0–8

Deep technical exploration of alternatives to 1-bit scap history. Stages 0–4 in x1 (const-c → 8-bit SRAM → 16-bit SRAM + EMA → single-cap analog → 2D cap×cap precision); stages 5–8 in x2 (fixed-larger normalization → addition update → cap swap → batch-locked role). Decision: use **16-bit SRAM** in draft3 baseline; the 2–3 cap scheme is a §21 future track.

---

## Version 3 — Breakthrough

### `draft3.0.md` — Distribution-Measurement Architecture

Major architectural rewrite. The central learning mechanism is replaced.

- H1 replaced: constant-c backprop → **distribution-weighted local update**
- Each scap **measures its own contribution** during forward pass via time-to-threshold
- Contribution stored in 16-bit momentum SRAM (EMA, α=3/4)
- Master broadcasts loss as **PWM pulse width + tri-state sign**
- scap update: `weight += pulse_width × momentum × sign(loss) × sign(weight)`
- Result: **fully decentralized backprop** — no per-scap gradient routing
- Framing corrected: "encryption" → **resident-weight compute** (in-memory compute / CIM)

### `draft3.0.q1.md` — Q&A on v3.0

Author's detailed reactions to community critique. Introduces (as candidates for draft3.1):

- **Hierarchical diffusion** (loss diffused through the brain-module hierarchy)
- **SpecialGeneralist** (Generalist gated by context masks from Specialists)
- Cortex / Hippocampus differentiated by **timescale, not topology**
- Master is small (5–10k transistor equivalent), not a full CPU

### `draft3.1.md` — Hierarchical Diffusion Formalized

Pre-draft4 consolidation. Several new major mechanisms.

- **§4 Brainstem (new):** small central controller — loss compute, broadcast, per-Lobe gating
- **§5 Hierarchical Diffusion Backpropagation (new):** the central mechanism — each level divides loss across children proportionally to distribution memory; **loss conservation invariant** Σ children_shares = parent_loss; new hypothesis H6
- **§9 SpecialGeneralist (new):** gated G-reuse via context masks
- **§10 Two-Timescale Limbic Loop (new):** Cortex (every clock) + Hippocampus (every k clocks), same topology, differentiated by cadence; **decay term 0.9** mandatory on recurrence
- First 5-level diagram: Brainstem → Limbic Loop → Columns → Blocks → Ganglia → Scaps
- First **"Protected list"** introduced

### `draft3.1.q1.md` — Q&A on v3.1

Red-pen feedback on draft3.1: early pushes on the "approximate chain rule" framing, the master-controller scope, and the precision-allocation across hierarchy levels. Sets up the honest-framing pivot in draft3.2.

### `draft3.2.md` — The Honest Framing Pivot

The project's intellectual-integrity moment. Math check showed `∂z/∂W = a`, not `a·W` — so the architecture's update rule is **not** an approximation of the chain rule.

- **§5.2 rewritten:** "What this actually is: attribution, not gradient" (side-by-side comparison with vanilla)
- **§5.3 new:** the family of methods this belongs to — three-factor learning, Equilibrium Propagation, Feedback Alignment, LRP, predictive coding, e-prop
- **§5.4 new:** known failure modes (dead-weight, momentum ceiling, routing-update coupling, init sensitivity, etc.)
- **§5.10 new:** honest negative section — what this mechanism does NOT do
- **H1 reframed:** "Attribution-based hierarchical diffusion converges on substantive tasks" — empirical, falsifiable, not theoretically guaranteed
- **H7 new:** "Dead-weight collapse is bounded"

This is the moment the project became publishable in principle.

### `draft3.2.q1.md` — Q&A on v3.2

Discussion of attribution vs gradient, the cost of the multiplier in `|a · W|`, and Path 0 / 1 / 2 remediation options if H1 fails.

### `draft3.3.md` — Pulse_Width vs Momentum

Two clarifications carried into draft4:

- **Pulse_width vs momentum disambiguation:** they are different duties (pulse_width = update-time control / arrived-share; momentum = per-Scap contribution history). Not duplicates.
- **Per-Column independent buses:** Cortex's bus and Hippocampus's bus fire independently — Brainstem gates them separately. Required for the two-timescale Limbic Loop to actually work.

### `draft3.3.r1.md` — Reflection on v3.3

Standing back from draft3.3 to identify what still needed answering before draft4 — multi-parent diffusion, expressivity beyond a linear Column chain, residuals as first-class.

---

## Version 4 — Structural Depth

### `draft4.0.md` — Five-Level Hierarchy

The architecture became:

```
Scap → Ganglion → Column → Lobe → Limbic Loop   (plus Brainstem)
```

**The Lobe is new.** It is a multi-branch DAG of Columns — Columns can have multiple parents, multiple children, and skip-connection topology.

- **Multi-parent diffusion:** a child with multiple parents sums incoming shares; loss conservation still holds.
- **Residual connections promoted to first-class** (§14). Primary defense against dead-weight collapse. H8 added: residuals reduce dead-weight rate and accelerate convergence.
- **Tunable weight decay via refill SRAM.** Free L2 regularization in physics — under-refill the weight cap and it drifts toward zero.
- **Learnable translate ALUs** with a size catalog (2:2, 2:4, 4:2, 4:4, 4:5, 5:5, 5:4, 8:4) fixed at fabrication.

Originally the new level was proposed as "Limbic"; renamed to **Lobe** to avoid collision with Limbic Loop.

### `draft4.0.x1.md` — Brainstorm

Exploration around draft4.0 — open multi-parent direction conflict, T_max resolution problem, Activity vs Relevance.

### `draft4.1.x1.md` — External Critique Notes

Notes from an external AI model's review of draft4. Identified four architectural holes — multi-parent direction conflict, Activity vs Relevance gap, T_max resolution bottleneck, analog error accumulation under PVT. These notes fed directly into draft4.1.

### `draft4.1.md` — External Critique Integrated

The four holes addressed:

- **Forward_Sign_SRAM** per Scap — resolves multi-parent direction conflict. 1 bit, XOR-based logic. **H11.** Caveat: degenerate under ReLU; useful at L1→L2 input layer and with signed activations.
- **Physical Saturation** as the primary defense against Activity vs Relevance / winner-take-all. The weight cap's hard supply-rail ceiling self-limits as `dV/dt → 0`. L2 regularization implemented in physics. **H10.**
- **Range Sensitivity (PGA)** on the measurement circuit — switchable gain (1×/10×/100×) for deep convergence. Triggered globally by Brainstem or locally per Lobe at >80% T_max clipping.
- **§15 Analog Robustness Mechanisms** promoted to baseline: Differential Pair op-amps, Dummy Scap per Column, Current Mirror Loss Share, Auto-Zeroing phases, optional Ping-Pong substrate.

Also self-caught and fixed: momentum ceiling saturation, residual two-port physical implementation, Limbic Loop's three sub-buses (Cortex / Hippocampus / Commissure) gated independently, label routing as a separate bus the Brainstem reads directly.

---

## Version 5 — Canonical Spec

### `draft5.0.md` — Structural Rewrite for a Stranger Reader

Substance unchanged from draft4.1; the document was reorganized to be readable cold.

- **§12 (the learning mechanism) moved to §2.** Mechanism before modules.
- **§3 added:** worked end-to-end XOR example — every quantity traced.
- **§1.5 added:** honest prior-art comparison vs Mythic, Loihi, IBM analog AI, BrainChip, SpiNNaker.
- **§17 hypotheses tiered.** H1 flagged as load-bearing; H2–H11 conditional on H1.
- **§19 open questions tiered** into architecture-critical vs tuning.
- **§20.1 Minimum Viable Falsification promoted** to the top of the simulation plan.
- **§22 "What Not To Touch" expanded to 14 items.**
- All "(new in 4.X)" version markers stripped.

### `draft5.0.x1.md` — Brainstorm

Notes during the draft5.0 restructure — what a stranger reader saw, what was hidden by the modules-first order.

### `draft5.1-full.md` — Final Polish (unsplit reference)

Polish pass over draft5.0. Architecture unchanged; clarity and correctness fixes.

- **§1.4 added:** bio narrative — three concrete biological observations, each mapped to one architectural decision; explicit "what we deliberately did NOT inherit" paragraph.
- **§1.6 → §2 bridge added:** the missing "therefore" connecting continuous substrate to attribution-based learning.
- **§3 XOR-convention bug caught.** Two passages computed `+ XOR +` and got different answers. Resolved to the operationally-correct convention; would have killed Phase 2 convergence if missed.
- **§3.6 share-division ambiguity resolved.** Division happens at the Ganglion via Current Mirror, not at each Scap.
- Transition sentences added at every section boundary.

This file is kept as the single-file reference. It is identical in content to `draft5.1-1.md` + `draft5.1-2.md`.

### `draft5.1-1.md` + `draft5.1-2.md` — The Canonical Spec (split for PDF)

The draft-5.1 canonical specification (historical). Now in `draft5.0/` (moved at the draft-6.0 pivot; was at the repo root), not under `draft-journey/`.

- **`draft5.1-1.md`** — Part 1 (§0–§19, §22–§23): framing, mechanism, worked example, modules, cross-cutting mechanisms, hypotheses, math targets, open questions, protected list, glossary.
- **`draft5.1-2.md`** — Part 2 (§20–§21): the full simulation campaign (MVF + ten phases, reproducibility infrastructure, effort budget) and future tracks.

Split only because the markdown-to-PDF tool fails on the unsplit length. Section numbers are continuous across both parts; cross-references work in both directions.

---

## Version 6 — the SCFF + gradient-descent rebuild (draft 6.0)

Draft 6.0 is **not** a single document like the earlier drafts — it is a **whole project tree** (`draft6.0/`),
because the rebuild is as large as the entire attribution era. After the draft-5 collapse (a missing sign — see
`project-history.md` Part II) it **replaced the attribution learning rule and kept the substrate.** The file-by-file
map:

| Path (`draft6.0/…`) | Type | One-line |
| --- | --- | --- |
| `README.md` | Story | the pivot — why 5.x died, what 6.0 is, the eleven-phase arc |
| `context.md` | Context dump | the whole picture for a cold agent (what / why / how / the person) |
| `CLAUDE.md` | Agent | the draft's operating context + the live status ladder (P1–P10) |
| `idea/README.md` | Index | the idea folder's front door + the decision timeline |
| `idea/ideas1.md` | Design | the origin blueprint — the nine-chapter plan, committed *before* a sim ran |
| `idea/main.ideas.v1.md` | Decision record | **N1–N3 + S1–S15**, one line per committed call (the draft-6 equivalent of the old §22 protected list) |
| `idea/phase4-opinion.md` · `phase4-problem.md` | Development | the prescient GD-era reflection + the depth-decay problem statement |
| `idea/gd-replan/` | Development | the depth-readout research hinge that opened Stage 2 |
| `research/` | Reading | `survey/` (learning-rule options) · `papers/` (paper stories, per phase) · `north-star/` (beyond-the-phases dossier) |
| `src/README.md` | Index | the results front door — the arc map + the document-system key |
| `src/stage1-report.md` · `stage2-report.md` · `validation-report.md` | Report | the three executive volumes (Stage 1 = P1–6 · Stage 2 build = P7–9 · the validation = P10–11) |
| `src/phase{1..11}/` | Phase | per phase: `README.md` (front door) · `phaseN-report.md` (deep) · `RESULTS.md` · `design.md` · `expK/` cards · `pNlib.py` + figures |
| `src/phase9-final-architecture.md` | Snapshot | the whole **frozen** two-brain model in one self-contained file (v2.0.0) |
| `src/ref-report/` | Glossary | methods · metrics · papers the reports cite (one pinned definition each) |

**The eleven-phase arc** is draft 6's version-history (the old drafts iterated `.md` files; draft 6 iterates *phases*):
**Stage 1** — P1 structure → P2 depth-round-1 → P3 the objective reframe (contrast supersedes energy-goodness) →
P4 characterization → P5 the SCFF close-out (depth solved) → P6 noise-hardening — built the cheap brain; **Stage 2**
— P7 the readout (*not* gradient descent) → P8 the economy → P9 the freeze → P10 the honest race — built the namer
and validated the whole object (verdict **S14**: a substrate-native continual learner, refined not inflated) — then **P11 the limit map** took the frozen object to real data + scale (verdict
**S15**: wins, losses, and floors all mapped). The
narrative is [`project-history.md`](project-history.md) (Part II); the front-door synthesis is
[`../../draft6.0/src/README.md`](../../draft6.0/src/README.md).

**Naming note:** draft 6 dropped the `draftX.Y.md` single-file convention. Its "versions" are the numbered **phases**
(P1–P11) and the **decision-record deltas** (N1–S15), not new filenames — because the rebuild is a project, not a
document.

---

## What Has Never Changed (v1 → v6)

- ~~**2-3-3-2 Ganglion topology, 29 scaps**~~ — hardwired v1 → v5, but the **draft-6 rebuild dropped the Ganglion hierarchy** (it belonged to the attribution rule that broke); only the *path-diversity intuition* carried over
- **scap** as the atom of storage (SRAM + Capacitor) — *the one architectural primitive that survived the draft-6 pivot intact*
- **Resident-weight goal** — weights never leave the substrate during operation
- **Sign as digital, magnitude as analog** in the scap
- **Intuition-first design** — literature comparison deferred to post-baseline (§1.8 methodological note)
- **Scope: Python simulation only** — no SPICE, no hardware, no external benchmarks yet
- **Semi-anatomical naming** as a *practice* — survived from v2 onward, though the specific names churned: Ganglion / Column / Limbic Loop were **dropped** at the draft-6 pivot (the dead attribution hierarchy), while Cortex / Hippocampus survive as the sleep / LUT-memory ideas

---

## Current State

> This file is the **historical draft map (1.0 → 5.1, the attribution era).** It is no longer the latest
> plan.

**Latest plan: `draft6.0/`** — the SCFF + gradient-descent rebuild (June 2026). The draft-5.1 spec
(`draft5.0/draft5.1-1.md` + `draft5.1-2.md`) and the attribution-era simulator (`draft5.0/src/`) are kept as **reference for
how the project worked before the pivot.** Start at `draft6.0/context.md` (the whole picture), or
`draft6.0/README.md` (the pivot story). See `CLAUDE.md` for the full current file map.

**Why the pivot (2026-06):** the attribution rule (`§5` hierarchical diffusion) carried loss *magnitude*
but never *direction* (the sign). Magnitude without direction can't descend anything; nothing converged at
scale, and routing the sign back would have broken the locality the chip stands on. The architecture was
rebuilt around SCFF + GD — **the substrate survived; only the learning rule was reborn.** The full story is
in `project-history.md`; the rebuild is in `draft6.0/`.

What this historical era achieved: the attribution mechanism *trained a single Ganglion* (regression near-
perfect, a nonlinear paraboloid ~20×) before the scale flaw surfaced — a real result, recorded in git and
`draft5.0/src/docs/`. The §22 protected list (14 items) and the §19 open questions belong to that dead architecture;
the live decision record is `draft6.0/idea/main.ideas.v1.md`.
