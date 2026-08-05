# handoff — read this first, then the tripwires

> For a future agent, or the author six months from now. Two minutes to arrive; the rest of the file is the list of
> ways this draft gets quietly killed.

---

## In two minutes

**What happened.** Draft 6.0 built and validated the first organ — a neocortex (unsupervised SCFF bulk + closed-form
namer). Eleven phases, real results, honest losses. Then the author stopped for two weeks and would not start
Phase 12. The reason, found by re-reading rather than by any failing number: **every metric the project used is one
the continual-learning field shares, and none of them can show whether a thinking loop is buildable.** The target
never moved. The evidence drifted away from it, monotonically, with no per-phase signal. That is what a "phantom
force" feels like from the inside.

**What draft 7.0 is.** A **SET ZERO**: find the core mechanism of a **loop of thought** first, with the analog /
online / forward-only constraints **dropped for the search** and re-imposed one at a time later. Draft 6.0 is
**folded, not discarded** — it gets picked back up once the loop core exists.

**Where it stands.** Nothing measured. No code, no roadmap, no experiment — **but there is now a position.** Since
2026-08-06 the draft holds a **standing candidate** (below), and the next move is a **reading checklist** whose job
is to produce **rivals that can beat it**. Reading with a target, not reading to get oriented.

**What is actually missing, in one sentence.** `[CLAIM]` Storage, similarity and retrieval can all be bought off
the shelf; what nobody hands you is **how the query changes when a candidate is rejected**, **how the goal tightens
when one is accepted**, and **the self-generated signal that decides which** — and that trio is the whole research
problem.

**⚑ The standing candidate — the FIRST decision, not the final one.** [`the-thought.md`](the-thought.md)
`[STANDING]` `[AUTHOR]`, adopted 2026-08-06:

1. a thought is a **thesis in three parts** — goal · an **accumulating fact register** · `f(goal, fact)` from
   memory; concepts are **bundles of attributes** in a space where meaning adds and subtracts, and a word is only
   the shared-world handle for a bundle
2. the loop has **two motions** — accumulate facts until memory matches, then **tighten the goal's scope** and run
   again — and it **never opens with the final question** (*fruit? → orange? → mandarin?*)
3. it runs in an **arena, not a benchmark** — CNN front half → *(loop)* → CNN back half, dense numbers that already
   carry meaning, **nothing hinted**, with a **read-bandwidth limit** making iteration structurally necessary

**How to treat it.** Evidence is **zero**. It is held because *"มันจะเถียงกันไม่ได้ ถ้าไม่มี candidate ที่เป็นภาพใหญ่จริงๆ"* — a
draft with no position cannot be argued with, so it drifts. It is **replaced by a better candidate, never by an
argument.** Bring a rival big picture, not an objection. Read it before designing, reading or proposing anything.

**The one number that would settle whether a loop is real.** `[CLAIM]` Steps-to-answer **k** versus memory size
**N**: sub-linear means the loop exploits memory structure (thinking); k ~ N means it is a linear scan (sorting).

---

## The tripwires

The author's stated fear when commissioning these documents:

> draft7.0 thesis มันอาจจะโดน push กลับไปทางเก่าได้อ่ะ 💀💀

The old line is **gravitationally attractive**: it has eleven phases of results, a validated object, and a metric
suite that produces clean numbers on demand. Draft 7.0 has none of that and will feel worse to work on for a while.
**That asymmetry is the danger, not a signal.**

Each item below is a sentence that will sound reasonable when someone says it. Each one is a return to the old way.

**1. "Just use SCFF as the representation front-end."**
No — during the search. P9.0 measured that the bulk **rotates**; a rotating representation makes memory keys stale
and can rotate mid-query, so a loop failure cannot be attributed to the loop or to the substrate. It also changes
two things at once (methodology rule 1). Struck **for the search phase only** — re-proposing it once a loop core
exists is the plan, not a violation. → `[STRUCK X3]`
*And when it does come back, it comes back as a different thing:* the author's stated future role is **a component
training method** — a drop-in where backprop would sit, with SLDA as the translator — **not** a representation
front-end. Gated on an unconstrained loop-of-thought final existing first.
→ [`the-thought.md`](the-thought.md) §9.0.

**2. "Goodness / tap-drift is the feeling of correctness."**
No. Goodness is **unary** (familiarity — exp0: clusters by density, not class); the feeling is **relational** (a
value over query × candidate, with gradations). A unary function has no argument slot for the candidate. The author
corrected the agent on exactly this, and it is settled.
⚠ **`draft6.0/research/north-star/` and the agent memory still contain the old wrong claim.** Those are the stale
documents; this one is current. → `[STRUCK X1, X2]`
*The mechanism-side version of the same finding, with the numbers:* [`the-thought.md`](the-thought.md) §6 — SCFF's
committed objective is `compare(x, corrupted-x)` against a class-blind background, and **P2.2's `hard_oracle` arm
already bought a perfect `compare(thing, other-class thing)` with true labels: depth-slope −0.022 vs random
−0.020, i.e. nothing.** SCFF's competence never came from comparing meanings.

**3. "Let's benchmark it — average accuracy, BWT, retention, prequential."**
That ruler is *the cause of the turn.* Loop properties only: step count vs difficulty, halt calibration on
near-misses, k-vs-N scaling. → `[STRUCK X4]`

**4. "We should keep it online / local / backward-free from the start — that's the project's identity."**
Not yet. This is methodology rule 7 (ideal first, realism later) applied one level up. The constraints come back
**one at a time, after** the mechanism exists. Meanwhile every candidate mechanism carries a **cost column**: what
would it take to make this local / online / backward-free? — a tiebreaker, never a gate.
*But the mirror-image failure is just as fatal:* if the constraints are **never** re-imposed, this becomes an
under-resourced entry into the most crowded field in ML (NTM/DNC, RETRO, PonderNet, Titans) with no GPU. The
constraints are the moat; the search phase is a loan against them, not a write-off.

**5. "Draft 6 was going well — why not just run Phase 12?"**
There is no Phase 12 to abandon: **P1–11 completed and were frozen.** Nothing is being walked away from. And the
work is not being disowned — SCFF+SLDA demonstrably works; it is simply not the loop, and never was.

**6. "This is basically an RNN / a transformer — just use one."**
They are **the middle state that holds the thought**, not the final architecture. Using one as a component is fine;
treating one as the answer is the drift.

**7. "It works — look, accuracy goes up with more steps."**
Not sufficient. With backprop available and no constraints, **a fixed unrolled depth will imitate a loop and report
success.** The guards are: single-pass failure must be *provable* on the task, and **step count must vary per
input**. Without both, a positive result means nothing.

**8. "The docs say X, so X is established."**
No. **Evidence level in this draft is zero, everywhere.** Every statement carries a tag — `[CLAIM]` `[OPEN]`
`[ARG]` `[CARRIED]` `[STRUCK]` `[LIT]`. Only `[CARRIED]` items were ever measured, and they were measured in
draft 6 for a different purpose. If you find an untagged assertion, that is a defect — tag it or remove it.

**9. "The thought is obviously a state vector."** / **"The thought is obviously weights."**
Neither is settled. The agent argued state, the author argued weight, and the author's reading has a real
literature home (**fast weights**). This is [`open-questions.md`](open-questions.md) **axis A** and it stays open
until the survey or a measurement closes it. Closing an open axis by argument is precisely the failure this draft
was created to escape.

**10. "The question is 'is it a mandarin?' — so make *mandarin* the goal and search."**
No. That is the **ladder rule** ([`the-thought.md`](the-thought.md) §5) and it is `[AUTHOR]`, not a preference.
Attacking the specific question directly is brute-force search over big data with nothing to prune against — it is
not thought even when it returns the right answer. The loop opens coarse and tightens: *fruit? → orange? →
mandarin?* It **will** look inefficient on easy tasks; the author already priced that in and accepted it, because
the object being built is the foundation, not a competitor to transformers at vector arithmetic.

**11. "The ladder works — look, steps grow sub-linearly with memory size."**
Check what is producing the ladder first. **A stored taxonomy passes the k-vs-N test while being nothing new** —
it is a decision-tree walk, and the intelligence lives in the tree, not the loop. The contribution only exists if
the coarse rung is **emergent from retrieval** (coarse concepts win early because they have the most support; the
next goal comes from the residual the matched concept fails to explain). → [`the-thought.md`](the-thought.md) §8.2
and [`open-questions.md`](open-questions.md) axis H. This is a live way to fake a result, and it pairs with #7.

**12. "Just grab a pretrained ResNet off torchvision and start."**
That is an **ImageNet-classification** backbone, and its features are organised by ImageNet's label tree — which is
**WordNet, a hierarchy built by hand, by people.** The standing candidate's strongest claim is that the taxonomy
**emerges from the images themselves** (`the-thought.md` §9.6). A label-supervised backbone voids that claim
silently, and the ladder will *appear to work* because a human tree was smuggled in through the features. Use a
**self-supervised** backbone (DINO / MAE / SimCLR). This is load-bearing, not hygiene. → `[ARG]` S20

---

## Reading order

1. [`the-thought.md`](the-thought.md) — **⚑ the standing candidate**: what a thought is, the loop, the arena, the base spec
1b. [`reading-checklist.md`](../notes/reading-checklist.md) — what to read and **what would kill each move**
2. [`context.md`](context.md) — why we turned (the whole story)
3. [`open-questions.md`](open-questions.md) — what is actually unknown, and the survey axes
4. [`ideas.md`](ideas.md) — candidate mechanisms, all unvalidated
5. [`the-turn.md`](the-turn.md) — the raw discussion, if you need to know *why* something is phrased the way it is
6. [`CLAUDE.md`](../CLAUDE.md) — the operating rules for working in this draft

---

## What has not been written yet, on purpose

- **no phase ladder, no experiment plan, no code** — deferred until the reading checklist has run
  *(the checklist itself is written: [`reading-checklist.md`](../notes/reading-checklist.md). An earlier build-first
  `learning-roadmap.md` existed and **the author deleted it on purpose** — it predated the ladder rule and the
  two-motion loop, so it planned builds against half a picture. **Do not restore it.**)*
- **the presentation** — a department talk is due 2026-08-10; the four-day lane in the checklist is sized for it.
  Slides were explicitly **out of scope** for now: *"ยังไม่ต้องทำ slide ตอนนี้ขอขัดเกลา based โมเดลให้คมสัสๆก่อน"*
- **the SCFF+SLDA black-box interface page** (the "fold insurance") — one page: what draft 6 provides, what it
  costs, its known limits. Without it, "folded" degrades into "forgotten" and picking it back up costs a re-read of
  eleven phases.
