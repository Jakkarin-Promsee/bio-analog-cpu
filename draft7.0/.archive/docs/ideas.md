# rough ideas — the candidate mechanisms

> **Everything on this page is `[CLAIM]` unless marked otherwise. None of it has been measured, and most of it was
> produced in one conversation.** It is kept so the reasoning is not lost, not because it is right.
>
> The author's instruction was explicit: rough ideas now, real research next. Do not build from this file — check
> it against [`open-questions.md`](open-questions.md) first, and against the literature after that.
>
> **⚠ SUBORDINATE, as of 2026-08-06.** This file was written when the draft had **no position**, so several
> entries read as though they were competing for the top-level frame. They are not. The draft's frame is the
> **standing candidate** in [`the-thought.md`](the-thought.md) — what a thought is, the two-motion loop, the ladder
> rule, the CNN arena. **Everything here is now a candidate for a *component inside* that picture**, not a rival to
> it. A genuine rival is welcome — but it must be a rival **big picture**, argued at
> [`the-thought.md`](the-thought.md)'s level, not a mechanism promoted to a frame by default.

---

## 1. Taxonomy — what "thought" is in each existing family

An agent's first-pass sketch answering survey axis 1. **Unverified; expand and correct it during the survey.**

| Family | "Thought" is | Memory is | Compare | Note |
| --- | --- | --- | --- | --- |
| RNN / LSTM | the hidden state `h` | weights, mixed into the same state | none — **not addressable** | the dossier already rejects LSTM-as-core; the author separately rejects RNN framing (see below) |
| NTM / DNC | controller state | an explicit R×W matrix | cosine → soft address | closest existing thing to the author's picture; known to be unstable to train |
| Transformer | a residual-stream vector per position | KV matrix (in-context) / MLP weights (long-term) | scaled dot product | content addressing without a write head |
| **Modern Hopfield** | a state vector | stored patterns as rows of a matrix | dot product, then settle | has an **energy** → "settled" is derived, not invented; formally the core of attention |
| Predictive coding / DEQ | the fixed point of a dynamical system | weights | residual at the fixed point | loop is native |
| PonderNet / ACT / Universal Transformer | state | weights | — | the contribution is specifically **a learned halt** |
| RETRO / kNN-LM | the LM's state | a frozen external datastore of vectors | kNN in embedding space | retrieval is single-shot, not iterative |
| **Fast weights** | ⚠ *a transient weight matrix* | slow weights | — | **the author's position** — see [`open-questions.md`](open-questions.md) axis A |

> **⚠ Updated 2026-08-06:** the base spec picks the **transformer** row as the container for the thought
> ([`the-thought.md`](the-thought.md) §9.6) — for the reason in §1 of that file (shared root with the vector
> picture), not because this table settled anything. The rest of the table is now **the list of rivals to check**,
> not a menu still being browsed. Add to it during the reading: Perceiver, DEQ, ACT/PonderNet, NSCL, VSA are all
> missing from it → [`reading-checklist.md`](../notes/reading-checklist.md).

**The author's own constraint on reading this table:** RNNs and transformers are at most **the middle state that
holds the thought** — never the final architecture, which is neocortex and hippocampus working together directly.
His reason for excluding RNN specifically: an RNN gives each input a special characteristic along the time
dimension that affects the future (what to keep, what not to keep), which is a different mechanism from
retrieve-compare-reject.

---

## 2. Three structural proposals

### 2.1 The compare defines the storage — not the reverse
`[CLAIM]`

The author's block is *"I can't choose the comparison because I don't know how thought is stored."* The proposal
inverts it: no working system chooses a storage format and then hunts for a metric. It picks the **metric**, and
storage becomes whatever makes that metric meaningful.

- attention picks dot product → storage is "put the vector in a row"
- VQ-VAE picks nearest-neighbour in a codebook → storage is an index
- predictive coding picks prediction error → storage is the generative weights

If this holds, the block dissolves: choose the comparison first and the storage question answers itself.
**Verify against axis 2 rather than assuming it.**

### 2.2 The compare and the retrieval are the same operation
`[CLAIM]`

`state · Keys` produces, in one product: the comparison, the addressing, and the residual that can become the
feeling. There is no need for a separate "compare module" — which is a large simplification if true, because it
removes an entire component the author was trying to design.

### 2.3 ⚠ Thought as fast state vs memory as slow weight — **CONTESTED**
`[OPEN]`

Recorded here for completeness only; the argument on both sides and the reason it stays open are in
[`open-questions.md`](open-questions.md) axis A. **Do not build on either reading yet.**

---

## 3. Modern Hopfield / associative energy memory — a candidate for **the memory operator**

`[CLAIM]` — a proposal, not a decision.

> **⚠ Re-scoped 2026-08-06.** This section was originally titled *"a candidate starting frame"* and was written
> when nothing else occupied that slot. **The frame is now the standing candidate** ([`the-thought.md`](the-thought.md)):
> thought = goal + fact register + `f(goal, fact)`, two motions, the ladder rule, running in the CNN arena. Modern
> Hopfield is a strong candidate for **`f` — the memory operator inside the loop** — and that is the role to
> evaluate it in. Its "does not give you the loop" caveat below is *exactly why it cannot be the frame*: the loop
> comes from the two motions, not from settling.

**Why it is attractive:**
- thought = a vector; storage = rows of a matrix; recall = a dot product → it answers survey axes 2 and 3 with a
  single object
- it has an **energy function**, so "settled" is mathematically defined instead of invented — which matters
  enormously for a halt signal
- it is a matrix and a softmax: it runs on a laptop in seconds, which fits the author's actual resources
- it is the formal core of attention, so anything learned transfers to whatever the final architecture becomes

**The honest caveat, stated up front:** modern Hopfield converges in **~1 step** — that is exactly why it is
equivalent to attention. **It does not give you the loop.** Any loop must come from an outer query-refinement
process, or from a different settling dynamics (classic Hopfield, predictive coding, DEQ). See
[`open-questions.md`](open-questions.md) axis E.

**And the piece the author feared most may be the best-charted:** "phantom thought" appears to be the known
phenomenon of **spurious attractors / crosstalk** — patterns the system settles into that were never stored,
arising from overlap among stored patterns. It has decades of theory and numeric capacity bounds (classic Hopfield
≈ 0.138 N; modern/continuous variants far higher). `[LIT]` — verify.

---

## 4. The named gap: the query update rule under rejection

`[CLAIM]` — the agent's answer to *"we still have no core theory for the right feeling."*

Storage, similarity and retrieval are all available off the shelf. What is **not** available: when a candidate is
rejected, how does the query change so the next iteration does different work? Plain associative memory settles
into the nearest attractor and stays — it returns the apple forever, with no reason to change its mind.

The author's story already specifies the mechanism informally:

> next time I think "what is this car color look like?, **plus it's not a tone of red apple**"

**The query accumulates negative constraints.** This is not in Hopfield, not in attention, not in RETRO.

Two mechanisms are therefore unknown, and everything else is purchasable:
1. **the query update rule under rejection** — how the query is rewritten
2. **the feeling** — the accept/reject gate that decides *whether* to rewrite

**A third was added 2026-08-05** ([`the-thought.md`](the-thought.md) §4, `[AUTHOR]`): **the goal update rule on
acceptance.** A match ends the *rung*, not the process — the goal then tightens (*is it an orange?* → *which
orange?*) and the loop runs again at higher resolution, carrying the fact register forward. This whole file was
written describing only rule 1.

**And a candidate for rule 1**, from the same source (`[CLAIM]`, agent inference, unconfirmed): concepts are
**bundles of attributes** in a space where meaning adds and subtracts, so rejection is **arithmetic on the query** —
subtract the *attribute* that made the candidate fail, not the *instance*. That is precisely the constructive
rejection §4.1 demands. The unsolved part is identifying which attribute caused the failure.

### 4.1 The trap that makes it fake

If rejection only **excludes** the rejected item, the loop degenerates into a linear scan of memory in score order.
That is sorting, not thinking — **and it will look like it works.**

The rejection must be **constructive**: it must inform *where to look next*. "Not a tone of red apple" removes a
whole region of colour space; removing one stored fruit does not.

### 4.2 The decisive metric — steps vs memory size

`[CLAIM]` — the cheapest test discussed, and the one that separates real from fake:

> measure **k**, the number of loop steps to an answer, against **N**, the size of memory
> - k grows **sub-linearly** in N → the loop is exploiting the structure of memory = **thinking**
> - k ~ N → **scanning**

No labels, no accuracy, no hard task required. Notebook-sized.

---

## 5. The form of the feeling

`[CLAIM]` — one candidate among several.

**Margin, not distance.** In the founding story the uncertain state is *"similar, but not sure"* — which describes
**two candidates scoring close to each other**, not one candidate scoring low. That points at the **gap between
top-1 and top-2** of the retrieval distribution rather than an absolute distance. A margin is also scale-free and
therefore comparable across different queries, which absolute distance is not — and comparability is exactly what
"calibrated" requires.

Other plausible forms not yet ruled out: residual magnitude, the energy value at settling, prediction error against
a generative model, agreement across repeated retrievals.

---

## 6. Criteria proposed for a loop (carried from the discussion)

`[CLAIM]` — these were written **before** any architecture on purpose, because the absence of a loop metric is the
hole benchmark gravity came through. They are not a roadmap; they are the shape of an eventual first experiment.

1. a task where a **single forward pass provably cannot** succeed, N iterations can, and **N varies per input**
2. accuracy rises with step count — a compute-adaptive curve, not a scalar
3. ⚠ **restated 2026-08-05** — was *"step count correlates with input difficulty (easy stops early)"*. Under the
   **ladder rule** ([`the-thought.md`](the-thought.md) §5) that is wrong as written: an easy input asked a
   fine-grained question still costs several rungs. Corrected form: **step count tracks the distance from the
   opening coarse goal to the resolution the question demands.** An agent benchmarking against the old wording will
   score a correct ladder as broken. (The author's MoE complaint — *don't think about easy things too long* — still
   holds; it just applies per rung.) `[ARG]`
4. a **learned** halt beats a fixed-step baseline, and the halt is **not** a label-trained head
5. **the halt signal is discriminative on near-misses** — *the red apple is not the exam;* **orange fruit vs neon
   orange** is. A feeling that cannot separate "close" from "exact" halts on the first plausible candidate and the
   loop is dead on arrival.
6. the halt is **calibrated in absolute value**, not merely ranked — it must be low enough on a near-miss to
   *continue*, not just lower than on an exact match
   ⚠ **contested since 2026-08-05:** if the goal's scope sets the required resolution
   ([`open-questions.md`](open-questions.md) axis I), calibration is **relative to goal scope** and an
   absolute-threshold halt is the wrong object. Do not build to this criterion until axis I resolves.

**Criterion 1 and 3 exist as anti-fraud guards:** with backprop available and no constraints, a fixed unrolled
depth can be trained to imitate a loop, and it will report success.

---

## 7. Held over from draft 6.0 — seeds that may matter later

`[CARRIED]` results, parked here so they are not lost. **None of them is the feeling** (see the struck list), but
each was validated and may serve a different role inside a loop:

- **P8.2 class-direction tap-drift** — label-free, leads error by ~8 steps, spine-clean. Struck as *the feeling*;
  potentially useful as a **gate**.
- **P7 cosine spine-pure head** — argmax-flip 0.000; a comparison that does not distort the representation.
- **P5 better-than-confidence selector** — a selection signal that beat raw confidence.
- **The hippocampus LUT prototype store** — in draft 6 it is a service (negatives + replay history), never a brain
  part. Growing it into a real organ was already the declared next build before the turn.
