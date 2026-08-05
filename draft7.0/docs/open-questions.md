# open questions — what is actually unknown

> The ledger. Every entry is either **open** (nobody here knows), **settled by argument only** (never measured), or
> **struck** (rejected with a reason). Nothing in this draft is settled by evidence.
>
> **⚠ Re-purposed 2026-08-06.** These axes were written when the draft had no position, so they read as a general
> syllabus. The draft now holds a **standing candidate** ([`the-thought.md`](the-thought.md)) and **these axes are
> its attack surfaces** — each one is a place where the standing candidate could be beaten by something better.
> Read them that way: not *"what should I learn about?"* but *"where is our stated position weakest, and who has a
> better answer?"* The weakest points today are **H** (the ladder has no mechanism), **J** (the substrate is
> chosen but unbuilt) and **D** (the rejection operation has two families and no evidence for either).
>
> **Rule: an entry only moves out of OPEN when something is measured or a paper answers it.** Moving an entry by
> argument alone is how the last six months happened.

---

## Part 1 — the survey axes (the author's four questions)

These come first because of discipline rule 4 (*read before building*). The author's framing:

> เราต้องไปเรียนรู้ตรงๆเลยว่าบนโลกนี้ เราทำการวิจัยไปถึงไหนกันแล้ว … กุไปคิดเอง … แม่งกุได้ตาย แบบหลงทาง

**These are questions to answer against the literature, not a roadmap.** The author explicitly deferred any
roadmap.

⚠ **What changed on 2026-08-06:** the survey is no longer open-ended orientation. The draft has a **first
decision**, and the reading exists to **find candidates that can beat it** — *"หัวข้อที่ใช้ตอนนี้ไม่ใช่ final decision แต่มัน
คือ first decision ก่อนกุจะไปทำ research เพื่อหา candidate มาสู้มัน."* Axis 1's *"how does each family define thought"*
is now specifically: **whose definition beats goal + fact register + `f(goal, fact)`?**

### Axis 1 — what models exist today (no constraints), and how does each define "thought"?
`[LIT]` `[OPEN]`

A first-pass taxonomy exists in [`ideas.md`](ideas.md) §1 — treat it as an agent's sketch to be verified and
expanded, not an answer. What the survey must establish for each family: what object plays the role of "the thing
currently being held in mind," how long it lives, and what changes it.

### Axis 2 — in each model, how are the two things (thought / memory) stored?
`[LIT]` `[OPEN]`

The author's motivating block: what comes back from the hippocampus is *not* a word and *not* an image — it is an
encoded abstract with no reality image, so the comparison operator cannot be chosen. The agent's counter-proposal
(the metric is primitive and storage is downstream) is `[CLAIM]` only — the survey should test it against how real
systems are actually built.

### Axis 3 — how do hippocampal memory and recall actually work?
`[LIT]` `[OPEN]`

Both senses are in scope and they must not be conflated:
- **biological-hippocampus** — the neuroscience (pattern separation / completion, CA3 recurrence, replay)
- the **engineering** analogues — associative memory, content-addressable storage, retrieval systems

Existing local starting point: `draft6.0/research/north-star/1-memory.md` (CLS, NTM/DNC, modern Hopfield,
retrieval/kNN-LM/RETRO, fast weights). It was written as free-time reading; it now becomes primary source
material — **but it was never verified against the loop question**, so re-read it critically.

### Axis 4 — phantom thought, and the structure of the loop
`[LIT]` `[OPEN]`

Agent's `[CLAIM]`: "phantom thought" is the known phenomenon of **spurious attractors / crosstalk**, with forty
years of theory and numeric capacity bounds — i.e. the best-charted of the four axes, not the worst. Verify this
before treating it as a hard problem.

---

## Part 2 — the open axes

### A. ⚠ What is a thought made of? — OPEN, and **the framing below is superseded**
`[OPEN]`

> **⚠ SUPERSEDED 2026-08-06 — read this box before the rest of the axis.** This axis was written as a **two-way**
> contest, agent (fast state) vs author (fast weight). It is not that shape. Asked directly, the author's answer was
> *"กุยังไม่แน่ใจเหมือนกัน … มีหลาย candidate มากๆ"* and he listed **three**
> ([`the-thought.md`](the-thought.md) §9.4): **(1)** thought = pure weight, self-driving, continuously changing;
> **(2)** thought = *fixed* weight running over a **context window of facts** that flows past — i.e. the fact
> register is the moving part and the machinery is still; **(3)** thought = **an instruction**, a program to run
> rather than a thing to hold. **He holds none of them firmly.** Anyone who "resolves axis A" by picking between
> state and weight has answered a question that was never asked. This is survey material, not argument material.
>
> **Where the base spec leans, 2026-08-06** — a lean, **not** a resolution: the thought's container is a
> **transformer** ([`the-thought.md`](the-thought.md) §9.6). Taken literally that is **candidate 2** — fixed
> weights with the **fact register flowing through the context window** — which is also the reading the author
> found most interesting in his own study list (*"context window ที่ไม่ใส่ทุก input at once"*). Candidates 1 (fast
> weights) and 3 (instruction) are **not** eliminated: nothing has been measured, and the transformer was chosen
> for a different reason (shared root with the vector picture), not to settle this axis.

The original two-way framing, kept for the record:

**The author:** thought *is* weight — picture weights inside a series model; it is small pure weights with neural
current wiring back and forth.

**The agent:** thought must be a **state** (fast, changes every loop iteration) and memory a **weight** (slow). If
thought were weight, thinking would be learning, and every thought would permanently rewrite memory — which the
author's own story contradicts (thinking about the apple in order to reject it does not redefine "apple").

**Why this is not closed:** the author's position has an established home — **fast weights** (transient,
rapidly-changing weight matrices used as short-term memory), which sits in his own dossier. Under fast weights the
distinction genuinely blurs, and "thought = weight" becomes a real design family rather than a category error.

**Sharpened 2026-08-05, still open.** [`the-thought.md`](the-thought.md) §3 names the object more precisely: a
thought is a **triple** — goal, an **accumulating fact register**, and what memory returns given both. The part
that changes within a loop and carries state across iterations is the **fact register**, which is neither the goal
nor the retrieved candidate, and which no document before this one had as a component. If the author's *"thought is
weight"* is about the fact register rather than the retrieved candidate, both positions may be describing different
members of the triple — but that is a hypothesis, not a resolution, and the axis stays **OPEN**.

**Why it matters more than it looks:** this choice determines *what the comparison operator even operates on* —
comparing two vectors and comparing two weight matrices (or two dynamical systems) are different problems with
different costs and different failure modes. Axis 2 of the survey should settle it, and until it does, **neither
side may be assumed in a design.**

### B. What grounds the feeling, if not labels?
`[OPEN]` — the single most likely months-eater

- graded against **labels** → a trained critic, not self-generated; and it puts back the expensive direction signal
  the whole project exists to avoid
- graded against **nothing** → collapse: everything feels correct (the classic self-supervised collapse; the
  dossier's `4-signal.md` already flags it)
- **the candidate answer** `[CLAIM]`: the **input holds one side of the comparison fixed**, so the system cannot
  cheat by making everything match. The feeling is then a per-candidate residual between what the input demands and
  what memory offers, minimised over candidates.

Unresolved even under that answer: what grounds the feeling for **internally generated** thought, where no sensory
input is pinning anything down (the *q = mc∆t* case — "it makes sense because all my memory agrees"). Consistency
among retrieved memories is a different grounding from sensory pinning, and the author's story uses both.

### C. What is the actual form of the feeling?
`[OPEN]`

Agent's `[CLAIM]`: a **margin** (top-1 vs top-2 of the retrieval distribution), not a distance — because *"similar,
not sure"* describes **two candidates scoring close together**, not one candidate scoring low; and a margin is
comparable across queries where a raw distance is not. Untested, and it is only one of several plausible forms
(residual magnitude, energy value at settling, prediction error, agreement across repeated retrievals).

### D. The query update rule under rejection — the named gap
`[OPEN]` — the agent's `[CLAIM]` for where the missing core theory actually is

Off the shelf: storage, similarity, retrieval. **Not** off the shelf: when a candidate is rejected, *how does the
query change so the next iteration does different work?* Vanilla associative memory settles into the nearest
attractor and stays there — it returns the apple forever.

The author's story specifies the mechanism informally: the query **accumulates negative constraints** — *"what does
this colour look like, plus it's not a tone of red apple."*

**⚠ Widened 2026-08-06 — the author did NOT confirm the candidate below.** Asked directly, he committed only to
*"มันมี operation ในการลบแน่ๆ ไม่ก็ operation ในการทำให้ f(เป้าหมาย, fact) มันสามารถ scope ตัวเองให้แม่นขึ้นได้"* — i.e. **two
distinct families**, and family 2 is not a variant of family 1:

1. **a subtraction operation on the query**, or
2. **an operation that lets `f(goal, fact)` sharpen its own scope** — the *reader* of memory tightens, not the query

Subtracting from the query and tightening the retrieval function are different machines with different costs and
different failure modes. **The survey must cover both.** → [`the-thought.md`](the-thought.md) §9.3.

**A concrete candidate for family 1, from 2026-08-05** (`[CLAIM]`, agent inference from `[AUTHOR]` material,
**explicitly not confirmed by the author**): under [`the-thought.md`](the-thought.md) §1–2 a concept is a **bundle of attributes** in a
space where meaning **adds and subtracts**, so rejection is plain **arithmetic on the query** — subtract the
*attribute* that made the candidate fail (red-ness), not the *instance* (apple). Subtracting an attribute removes
a whole region; subtracting an instance removes one item. That is exactly the constructive/linear-scan distinction
below. **Unsolved even so:** identifying *which* attribute caused the failure.

**And the loop has a second update rule that this axis never covered:** the **goal update on acceptance** — a match
ends the rung, and the goal then tightens. See [`the-thought.md`](the-thought.md) §4 and axis H.

**The sub-question that decides whether it is thinking at all:** the rejection must be **constructive** — it must
inform *where to look next*, not merely remove one item. "Not a tone of red apple" excludes a whole region of
colour space; excluding one stored fruit would make the loop a linear scan.

### E. Does the loop come from settling, or from the outer query refinement?
`[OPEN]`

Modern Hopfield converges in ~1 step (which is precisely why it is equivalent to attention), so a frame built on it
does **not** get a loop for free. Either the loop lives in an outer refinement loop over queries, or a different
settling dynamics (classic Hopfield, predictive coding, DEQ) is needed. Do not assume settling gives the loop.

### F. What is the smallest task where a loop is provably necessary?
`[OPEN]`

Required properties: a single forward pass **provably** cannot solve it; N iterations can; and **N varies per
input**. Without provable single-pass failure, a fixed unrolled depth will masquerade as a loop and look like
success. Candidate discussed: multi-hop associative lookup with a variable hop count — it is the author's own
orange-car story in miniature and it forces a memory store to exist from day one. Unverified.

**Sharpened 2026-08-06, and now grounded in axis J's substrate** `[CLAIM]`: the property that makes single-pass
failure nearly *provable* is **serial dependency of lookups** — each retrieval's **address** is computed from the
previous retrieval's **content**, so the chain cannot be flattened into one bounded pass. On conv components that
reads as: *find the region whose attribute matches what was just found somewhere else*, multi-hop, with hop count
varying per input.

**The cheapest known way to get the property, and it needs no new task** `[CLAIM]`: **limit the read bandwidth.**
Inside the pooling-slot arena ([`the-thought.md`](the-thought.md) §9.5), let the loop see only **k components per
step** out of `C × H × W`. When the evidence an input requires exceeds *k*, a single pass **provably** cannot
succeed — an information bound on reads, not a hoped-for difficulty — and **step count varies per input by
construction**. It also forces the fact register (remember what you already read) and the query update rule (where
to look next depends on what you found) to be load-bearing rather than decorative.

⚠ **The trap this replaces:** plain image classification does **not** qualify — an unmodified CNN *is* the proof
that one pass suffices. The arena's *default* objective ("produce the tensor the back half expects") is a supervised
regression target under which gradient descent finds the cheapest feedforward map, so **the default objective
selects against looping.** The arena is right; its default task fails criterion 1, and the bandwidth limit is what
repairs it.

### H. Where does the coarse→fine ladder come from?
`[OPEN]` — added 2026-08-05; **provisionally ANSWERED by the author 2026-08-06, still unverified**

> **⚠ The author's answer, and it closes the "which of the two sources" fork below in favour of *emergent*.**
> ([`the-thought.md`](the-thought.md) §9.6) **The taxonomy is the conv co-occurrence structure itself** — if an
> orange image carries `Conv[x][y][0]` and `[x][y][1]`, and a sphere image carries `[x][y][0]`, then *orange* and
> *sphere* are connected **through that channel**, *"โดยข้อมูลภาพตรงๆ ไม่ใช่จากภาษา"*. No stored tree, no WordNet, no human
> categorisation. The rungs are **ranked property checks** ordered by how heavily connected each component is, and
> the ranking is **recomputed against the current fact cache**, so it is conditional rather than global.
>
> **This is no longer "a hole with no mechanism" — it is a stated mechanism with zero evidence.** Two things must
> now be checked rather than invented:
> 1. **the backbone must be self-supervised**, or the co-occurrence structure inherits ImageNet's WordNet tree and
>    the claim *"the taxonomy came from images, not language"* is false through the back door → §J, tripwire #12
> 2. **does connection-count actually behave like generality on real conv features?** Nothing has measured this.
>    Nearest formal relative to compare against: information gain / candidate elimination
>    ([`reading-checklist.md`](../notes/reading-checklist.md) §4), and HNSW's layered coarse→fine graph (§3).
>
> The fork below is kept because **the failure mode is still live**: if the ordering turns out globally static in
> practice, the ladder is a fixed pipeline with early exit and tripwire #11 fires anyway.

The author's ladder rule (`[AUTHOR]`, [`the-thought.md`](the-thought.md) §5) says the loop never opens with the
final question: *is it a fruit? → is it an orange? → is it a mandarin?* That sequence has to be produced by
something, and the two candidate sources are not equally valuable:

- **stored taxonomy** → the loop is a decision-tree walk; the intelligence is in the tree. **It passes the k-vs-N
  sub-linearity test anyway** (axis-D §4.2 in [`ideas.md`](ideas.md)) while contributing nothing — a live way to
  fake a result, hence tripwire #11.
- **emergent from retrieval** → coarse concepts win early because they carry the most support under a thin fact
  register; finer ones win only as facts accumulate. The next goal is then generated from the **residual** — the
  attributes of the input that the matched concept fails to account for. This is the only version that is a
  contribution.

**Draft 6 offers nothing here.** `[CARRIED]` C1: SCFF clusters by **density, not class** — and density is not
generality either, so nothing in eleven phases produces a "fruit-level" attractor. Whatever supports the coarse rung
must be built.

**The author's own idea (2026-08-06, `[AUTHOR]`, no research done yet):** *"กุคิดว่าเราทำ softmax ให้แต่ละ ladder ได้อ่ะ"* —
"orange" is already connected to every other word, so the rungs can be **ranked**. He names the obstacle himself:
that story is native to **language models, and this is not one**. → axis J.

**Agent candidate (`[CLAIM]`, unverified):** in vector-symbolic / hyperdimensional algebra, **superposition
(bundling) is the generality operation** — bundle many mandarins and only what they share survives; bundle wider and
you climb to *orange*, then *fruit*. **Generality = bundling depth**, over any vector source. It also gives the
emergent answer for free: a heavily-bundled vector matches more inputs, so **the coarse rung wins early because it
is coarse** — no stored tree needed. VSA/HDC is a known hole in this repo's reading; it belongs at the top of the
commissioned reading checklist. → [`the-thought.md`](the-thought.md) §9.2.

### J. What input substrate does the first model get built on?
`[OPEN]` — added 2026-08-06; **gates most of the reading list**

> **Concretised the same day** by the base spec ([`the-thought.md`](the-thought.md) §9.6) and the playground
> staging in [`reading-checklist.md`](../notes/reading-checklist.md) §5: **dSprites / 3D Shapes** (exact ground-truth
> factors — the only place the component decomposition can be *verified* rather than hoped for) → **CUB-200**
> (312 human attributes, **diagnostic only, never a training signal**) → **CLEVR** (where binding becomes
> mandatory, and where axis F's serial lookups are native) → **CLEVRER** (video; the destination, not the start).
> **Two decisions still open: which backbone, and which of these is first.** Everything else can wait.

`[AUTHOR]`, tagged by him *"ยังไม่ take แค่เป็น ideas"*: **images**, with a **frozen CNN as an abstract extractor** —
explicitly **not** the foundation of the model but a **data generator**, splitting one image into *"หลายๆ component
เฉพาะ ที่มันมีการเชื่อมโยงถึงกันได้"*. Why not text: language fixes "apple" rigidly and its real difficulty is composing
words into sentences, whereas a conv stack **is already** abstract-stacking and extends to **video** later.
Full statement and reasoning: [`the-thought.md`](the-thought.md) §9.2.

**❌ Withdrawn — "similarity is not generality."** The agent's first objection assumed the *ladder* was expected to
emerge from CNN feature geometry. It is not; the CNN only supplies the components. The objection does not apply.

**✅ Strengthened instead** `[CLAIM]`: **text hands you the handle and hides the bundle.** "Apple" is exactly the
shared-world label that §2 calls an identifier, not the content — language already compressed the attributes away,
so recovering them means reconstructing from co-occurrence what language deleted. **An image hands you the bundle
before it was named.** For a loop that operates on bundles, images are close to the only substrate where the object
of study exists pre-compression. It also *defends* tripwire #11: in text the generality hierarchy is given
(hypernymy/WordNet), so a ladder would be handed over; in images nothing hands it over, which is why one found there
would be a result.

**⚠ What still stands, all `[CLAIM]`:**

1. **"Many components" is not free.** Conv channels are **entangled and polysemantic**; a channel is not an
   attribute. → disentanglement (β-VAE, FactorVAE), sparse autoencoders / dictionary learning, object-centric
   methods. ⚠ **object-centric ≠ attribute-centric**: slots factor a scene into *entities*, §2 needs *properties*.
2. **Prefer a self-supervised backbone** (DINO / MAE / SimCLR) over an ImageNet-classification one, whose features
   are shaped by **WordNet, a hand-built hierarchy**. Cheap insurance against an inherited ladder.
3. **⚠ The heaviest one — do not let "linked" become "fire together, wire together" again.** The link relation is
   undefined and *is* the memory structure. If "linked" means *co-occurred in the same image*, the mechanism is
   **correlation**, which is the exact lesson of S12/S13 reproduced one level up on a new substrate. VSA's
   distinction is the guard: **bundling** (superposition, what co-occurrence gives you) is **not binding**
   (role–filler attachment, what makes *orange-coloured* attach **to this fruit** rather than merely occur beside
   it). Co-occurrence gives bundling free and **never gives binding**; binding must be chosen deliberately.
4. **Known boundary:** images ground *perceptual* thought (the orange-car story runs) and ground **nothing** for the
   *q = mc∆t* case — axis B's unresolved half. Acceptable scope for a first substrate; must be said out loud.
5. **Video is the destination, not the start** — it is literally the author's §5 example (*watch a moving object,
   connect it to a law of physics, predict it*), which is what justifies choosing images at all. Expensive, no GPU;
   static images carrying several attributes is the notebook-sized version.

### I. Does the goal's scope set the acceptance threshold?
`[OPEN]` — added 2026-08-05 from [`the-thought.md`](the-thought.md) §8.3

If the loop tightens its goal on each match (motion 2), then *"similar, not sure"* is not one absolute threshold:
**the current goal's scope sets the resolution the match must meet.** The same match strength passes at *"is it an
orange?"* and fails at *"which orange?"*. If that holds, criterion 6 in [`ideas.md`](ideas.md) §6 ("the halt is
calibrated in absolute value") is misframed — calibration would be **relative to goal scope**, and an
absolute-threshold halt is the wrong object to go looking for.

### G. How do neocortex and hippocampus divide the work?
`[OPEN]`

The author's picture: the neocortex takes input + previous memory state and asks *"is there anything in memory like
this?"*; the hippocampus searches and returns candidates, which carry description and detail already wired to them.
Every part of this is an untested design sketch — including whether the division is even two-way.

---

## Part 3 — settled by argument only (never measured)

| # | Statement | Tag |
| --- | --- | --- |
| S1 | The phantom force is **benchmark gravity** — P1–11's metrics are all shared with mainstream continual learning, and none can show whether a loop is buildable. The target never moved; the evidence did. | `[ARG]` |
| S2 | Goodness is **unary** (familiarity) and the feeling is **relational**; a unary function cannot produce the feeling by construction. Tap-drift fails the same way (it monitors the stream, not a proposition). | `[ARG]` |
| S3 | Set zero is this project's **own** methodology one level up — rule 1 (one thing changed) and rule 7 (ideal first, realism later). | `[ARG]` |
| S4 | The constraints (online / no backward pass / resident-weight / local) are dropped for the **search only** and re-imposed one at a time afterwards; the **cost column** keeps them alive meanwhile. | `[ARG]` |
| S5 | The draft-1 **homogeneity** premise is dead — and that is a finding (biology is heterogeneous; MoE is the same discovery from the ML side). Cost: the "one analog tile" pitch. | `[ARG]` + `[CARRIED]` |
| S6 | The **compare defines the storage**, not the reverse — the metric is the primitive. | `[CLAIM]` |
| S7 | The compare and the retrieval are the **same operation**; no separate compare module is needed. | `[CLAIM]` |
| S8 | A thought is a **thesis in three parts** — goal · accumulating fact register · `f(goal, fact)` from memory. Retrieval is conditioned on **both**, never on the input alone. | `[AUTHOR]` |
| S9 | The loop has **two motions**: hold the goal and accumulate facts until memory matches; then **tighten the goal's scope** and run again. Every doc before 2026-08-05 described only the first. | `[AUTHOR]` |
| S10 | **The ladder rule** — the loop never opens with the final question. *fruit? → orange? → mandarin?* Attacking the specific question directly is brute-force search, not thought. Accepted as inefficient on easy tasks, on purpose. | `[AUTHOR]` |
| S11 | Concepts are **bundles of attributes** in a space where meaning **adds and subtracts** — the same root the transformer vector stands on. A word is the shared-world handle for a bundle, used at the resolution the context asks for. | `[AUTHOR]` |
| S12 | SCFF's committed objective is `compare(x, corrupted-x)` against a class-blind background — a **difference amplifier**, never a comparison of meanings. The project's own P2.2 `hard_oracle` arm already bought a perfect `compare(thing, other-class thing)` and it changed the depth-slope by nothing (−0.022 vs random −0.020; synth control +0.027 proves the machinery was fine). | `[AUTHOR]` + `[CARRIED]` |
| S13 | SCFF is **an equation for how weights encourage each other so an item separates from noise** — not a comparison in any sense the project needs. **The founding thesis was built on that illusion.** Its surviving role is later and gated: a **component training method** (a drop-in where backprop would sit, SLDA as translator) — *"ไม่ทำตอนนี้ รอได้ final ของ loop of thought แบบไม่มี constrain หน่อย"*. | `[AUTHOR]` |
| S14 | The rig is an **arena, not a benchmark**: CNN front half → *(loop)* → CNN back half. Input and output are numbers that already carry meaning; the loop is given practice at thinking **with nothing hinted** — no language, no pre-digested abstraction ladder. Beating pooling was never the goal. Complexity is a **dial** (same rig from a tiny net to a deep ResNet). | `[AUTHOR]` |
| S15 | The arena's **default objective selects against looping** (produce-the-expected-tensor is supervised regression → GD finds the cheapest feedforward map). Repair: a **read-bandwidth limit** — *k* components per step — which makes single-pass failure provable, makes step count vary per input, and makes the fact register and query-update rule load-bearing. | `[CLAIM]`, author-approved 2026-08-06 |
| S16 | This draft holds a **standing candidate**: a stated big picture, evidence zero, **replaced by a better candidate and never by an argument.** A draft with no position cannot be argued with, so it drifts. | `[ARG]` |
| S17 | **The CNN back half is kept on purpose.** If output were the class, the model would have to be both a loop *and* a fine-grained classifier and would break; sending output onward leaves the loop **one job — make vectors and think on vectors** — with the back half as **the translator into human-visible classes**. | `[AUTHOR]` |
| S18 | That division — *loop in vector space + a namer on top* — is **structurally identical to draft 6's SCFF bulk + SLDA namer.** The same insight (never make the representation learner also be the classifier) survived a complete change of substrate. | `[ARG]` |
| S19 | **The taxonomy is the conv co-occurrence structure**, not language: concepts are linked through shared components, rungs are ranked property checks, and the ranking is **recomputed against the fact cache** (conditional, not global). *"abstract stacking search"*, not brute force. | `[AUTHOR]` |
| S20 | Therefore the **backbone must be self-supervised.** A label-trained CNN organises features by ImageNet's WordNet tree, which would make *"the taxonomy came from images, not language"* false through the back door. This is load-bearing, not hygiene. | `[ARG]` |
| S21 | Co-occurrence yields **bundling, never binding.** Harmless on single-object images; **fatal on multi-object scenes** (round + orange + green + leafy in one image does not make the orange leafy). This is what stages the playground: object-centric first, scenes exactly when binding becomes mandatory. | `[CLAIM]` |

---

## Part 4 — struck (do not re-propose without new evidence)

| # | Struck | Why |
| --- | --- | --- |
| X1 | **SCFF goodness as the correctness feeling** | Type error: unary familiarity vs relational judgement. Author's correction, agent concurred. Note this contradicts `north-star/…` line 22 and the memory file — **those are the stale ones.** |
| X2 | **P8.2 class-direction tap-drift as the feeling** | Same type error — it monitors the stream, not a proposition. *May still be usable as a gate;* it is only struck **as the feeling.** |
| X3 | **Building the loop on top of live SCFF (during the search)** | P9.0 rotation makes memory keys stale and can rotate mid-query → loop failures become un-attributable. Violates rule 1. Struck **for the search phase only** — re-proposing it *after* the loop core exists is the plan, not a violation. |
| X4 | **Judging draft 7.0 by AA / BWT / retention / prequential accuracy** | That ruler is the cause of the turn. |
| X5 | **The brain / substrate is homogeneous** (draft-1 SCAP premise) | Empirically dead across eleven phases. |

---

## Part 5 — carried forward from draft 6.0 (measured, still true)

| # | Result | Where |
| --- | --- | --- |
| C1 | SCFF clusters by **density, not class** — goodness is a familiarity measure | exp0 |
| C2 | The bulk **rotates but does not forget** — the rotation confound for addressable memory | P9.0 |
| C3 | Unsupervised SCFF (fire-together-wire-together + closed-form namer) **works, and works well**, as a high-continuity continual learner | P1–P11 |
| C4 | The neocortex organ is **heterogeneous** — temp, goodness form, cross-layer transmission, noise-aug all change how it sees the world | P3, P5, P6 |
| C5 | The validated object's shape: wins continual safety and noise, trails static accuracy, floors on autocorrelated streams | P10, P11 |
