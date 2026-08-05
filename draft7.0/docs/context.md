# draft 7.0 — context: why we turned

> The cold-start narrative. If you read one file in this draft, read this one. Written 2026-08-05, immediately
> after the conversation that produced the turn (that conversation is preserved argument-by-argument in
> [`the-turn.md`](the-turn.md)).
>
> **Everything below is `[CLAIM]` unless tagged otherwise.** The author's position at the time of writing:
> *"ตอนนี้ตีว่ากุมี evidence รองรับเป็น 0 ทุกอันเลย"* — treat the evidence level of this whole draft as zero.
>
> **⚠ Written before the draft had a position.** On 2026-08-06 draft 7.0 adopted a **standing candidate**
> ([`the-thought.md`](the-thought.md)) — a definition of thought, the two-motion loop, the ladder rule, and the CNN
> arena. This file is still the correct account of **why we turned**; it is *not* the current account of **what we
> are building**. §7 in particular is explicitly superseded.

---

## 1. The trigger — a gut signal with no supporting number

Draft 6.0's Phase 11 finished on 2026-07-05 with strong results across the board: 8 real-data arenas × 5 capability
channels, a win on real sensor drift, a substrate factor that *grows* with scale, honest losses and floors all
shipped. By every standard the project used, it was the best phase yet.

The author then **did not start Phase 12 for a month**, and withdrew from the repo entirely for about two weeks.

His description of the feeling:

> เหมือนมันมีทางแยกเมื่อครึ่งปีก่อน แล้วกุเลือกเดินผิดทาง ทางที่กุเดินมันก็ค่อยๆห่างจาก target กุไปเรื่อยๆ … มันไม่ใช่ผิดทางในแง่ของการวาง
> target ผิด แต่เป็นผิดทางในแง่ของผลการทดลองค่อยๆ shift มันไป เหมือนเจอ phantom force ที่พยายามดึงออกเลย
>
> มองเผินๆ … ทุกอย่างมันจะดูปกติดีหมดเลย เพราะแทบไม่มี signal อะไรที่บอกว่าเรามาผิดทางเลย แต่ gut กุมันบอกอย่างนั้น

This matters procedurally: the author's documented working mode is that breakthroughs come from **incubation, not
desk-grinding**. The two-week withdrawal producing a structural re-read is that mode working, and the reluctance to
start Phase 12 **was the signal** — not procrastination.

---

## 2. The diagnosis — the target never moved, the *evidence* did

The phantom force has a name: **benchmark gravity**. `[ARG]`

Look at what P1–P11 actually measured: average accuracy, BWT / worst-point BWT, gauntlet retention, GD-share,
energy ratio, prequential accuracy on gas / HAR / ELEC2 / covtype. Every one of these is a metric the mainstream
continual-learning field shares. That was the *right* choice for defensibility — it made the object comparable to
a tuned BP+replay baseline, and it is why the draft-6 results survive review.

But: **no metric in the entire eleven-phase program can tell you whether a thinking loop is buildable, even if it
scores perfectly.**

So the quantity that actually drifted is:

```
(evidence that speaks to the north star) / (total evidence accumulated)
```

which fell monotonically toward zero over a year, while every individual phase was correct, honest, and well-run.
There is **no per-phase signal for this** — it is only visible as a year-scale derivative. That is exactly why
re-reading `RESULTS.md` ten times found nothing wrong: nothing *was* wrong. The evidence was simply being
accumulated in a different pile from the one the project needs.

**The target — "correctness is a feeling, loop until the feeling says stop" — never moved.** It is the same target
as the 1 a.m.–8 a.m. session that started the project.

---

## 3. What died on the way: the homogeneity premise

Draft 1's founding picture was that the brain is **homogeneous** — one cell type, one circuit (SCAP), tiled
everywhere; *analog circuit makes the advantage, the intelligence follows.*

Eleven phases killed it, empirically. `[CARRIED]` The neocortex organ *alone* carries a pile of characteristics
that each change how the model sees the world: the InfoNCE temperature, the form of the goodness equation, the
cross-layer coordination window, how goodness is transmitted between layers, the noise augmentation. It is not one
uniform substance; it is a machine with parts.

**This is a finding, not a failure.** Two independent confirmations:
- **Biology agrees.** Cortex and hippocampus are wildly different machines. A faithful "copy the brain's function"
  *should* come out heterogeneous. Getting homogeneity would have meant copying it wrong.
- **ML agrees, from the other side.** Mixture-of-Experts is the field rediscovering organs. The author's own
  framing of why it matters: *nobody is good at everything — even the strongest model has blind spots and is slow
  on easy work because it thinks too much.*

**What it costs:** the pitch "one analog tile, mass-produced across the die" dies with it. Accept that and move the
pitch to the architecture. This costs less than it sounds — the repo already states that analog is the
**constraint-giver, not a fabrication plan** (no tape-out was ever planned).

---

## 4. The correction that mattered most: goodness is **not** the feeling

This one is worth stating precisely, because the agent got it wrong first and the author corrected it, and the
correction is load-bearing for everything in draft 7.

**The wrong bridge (agent, rejected):** SCFF's goodness is a label-free scalar that says "does this feel right,"
therefore it is the halting signal — with support from the north-star dossier's own line (*"SCFF goodness → the
correctness/halting signal"*) and from P8.2's validated-but-unshipped class-direction tap-drift signal.

**The author's rejection:**

> สิ่งที่ SCFF มันไม่ใช่ "ความรู้สึกว่าถูก" เลย เราถูก hebbian rule หลอกมุมมองมานานมาก … สิ่งที่มันทำจริงๆก็แค่การใช้ "fire together,
> wire together" ในการขยายสัญญาณความแตกต่างระหว่างของ 2 class แล้วรอให้ SLDA มาเก็บงาน … มันเป็นแค่วิธีๆหนึ่งในการสร้าง
> neural network ที่มี characteristic ในด้าน continuous สูงมากๆ

**Why he is right — the type argument.** `[ARG]`

- **Goodness is unary.** It is a function of *one* thing: does this input fit the manifold I have learned? That is
  **familiarity**. Draft 6's very first experiment already said so: exp0 — *SCFF clusters by density, not class.*
- **The feeling is relational.** In the founding story it is a value produced by comparing *two* things — the query
  being held (the car colour in your eyes) against a candidate pulled from memory (the apple's colour in your
  thought) — and it comes with **gradations**: *"similar, not sure"* → keep looping.

A unary function cannot do this **by construction**, not merely poorly: it has no argument slot for the candidate.

The same type error kills the P8.2 tap-drift seed *as a feeling*: it monitors whether the **stream** has moved, not
whether a **proposition** is satisfied. It may still be useful as a gate. `[STRUCK]` — do not re-offer goodness or
tap-drift as the correctness feeling.

**What SCFF actually is, stated correctly:** a way to build a network with very high continuity, by using
fire-together-wire-together to amplify the difference between classes and letting a closed-form namer collect the
result. That is a real and validated contribution. It is not the loop, and it never was.

> **Sharpened 2026-08-05 — see [`the-thought.md`](the-thought.md) §6.** The section above argues from the *type* of
> goodness (unary vs relational). The author later gave the mechanism-side version: SCFF's committed objective is
> `compare(x, corrupted-x)` against a class-blind background — *"compare(สิ่งของ, noise) เเต่สิ่งที่เราตั้งไว้ตอนเเรกคือ
> (สิ่งของ, สิ่งของ)"* — and the project **already tested the fix**: P2.2's `hard_oracle` arm handed SCFF a perfect
> `compare(thing, other-class thing)` using true labels and the depth-slope moved by nothing (−0.022 vs random
> −0.020, while the synth control lifted +0.027). SCFF's competence never came from comparing meanings. Both
> arguments land in the same place; keep both.

---

## 5. The decision: SET ZERO — fold, don't discard

The author's reasoning:

> SCFF มันอาจจะเร็วและหนักเกินไป … เราติด constraint ด้าน phasic กับ unsupervised ตลอดการหา core model ของ loop of thought …
> ยิ่งเราเพิ่ม constraint หรือ unstable model เข้าไปให้มันอีก เราอาจจะหลงทางกว่าเดิม
>
> set zero ที่หมายถึง ไม่ใช่คือการโยนทิ้งไปตลอด เราแค่พับเก็บมันไว้ หลังจากที่ได้ loop of thought แล้วเราค่อย pick SCFF + SLDA
> หยิบขึ้นมาลองใหม่ก็ได้

Three arguments support it, in increasing strength:

**(a) The surface is too large.** SCFF as committed carries ~6 live knobs and 12 layers of behaviour that were
themselves the subject of research. Building an unknown loop *on top of* an unsettled substrate means two unknowns
interacting.

**(b) A concrete technical confound — the rotation problem.** `[CARRIED]` P9.0 measured that the SCFF bulk
**rotates but does not forget**. Draft 6 lives with this because the namer re-fits at sleep — a classifier does not
care that its code rotates. But **a loop of thought needs an addressable memory**: you store something now and
query it by similarity thousands of steps later. If the representation rotates, the keys go stale — and worse, it
can rotate *during* a query. Then when the loop fails to converge you cannot tell whether it is the loop's fault or
the substrate's. That is an un-diagnosable experiment, not an inconvenience.

**(c) The decisive one — set zero *is* this project's own methodology.** `[ARG]`
- **Rule 1, one thing changed per experiment:** an unknown loop on an unsettled substrate changes two.
- **Rule 7, ideal first, realism later:** used for eleven phases to defer PVT realism. Set-zero-then-re-constrain
  is the identical rule applied one level up, at the architecture instead of the device.

So set zero is **not** an abandonment of the project's discipline; it is an application of it.

**The agent's earlier objection is withdrawn.** The objection was "the constraints are the moat — dropping them
lands you in the most crowded, most GPU-hungry field (NTM / DNC / RETRO / PonderNet / Titans) with no GPU."
That was wrong on **timing**, not on principle: the moat is lost by *never re-imposing* the constraints, not by
dropping them during the search. The device that keeps it alive is the **cost column** (see
[`CLAUDE.md`](../CLAUDE.md) discipline rule 2).

**What "fold" must include, or it degrades into "forget":** one page stating SCFF+SLDA as a **black-box
interface** — what it provides (an online, backward-free representation that separates classes; a closed-form
namer), what it costs (rotation, sleep re-fit, ~6 knobs), and its known limits (static accuracy trails,
autocorrelated streams floor, resolution floor on CIFAR-gray). Without that page, picking it back up in a year
costs a re-read of eleven phases. *(Not yet written — first concrete task when the author wants it.)*

---

## 6. What we are actually building toward — the author's own picture

Preserved as close to verbatim as possible, because this is the primary source.

**The origin (from `the-essence2.md`, the 1 a.m.–8 a.m. session):** how do I know I'm right? Nobody hands me a
label. There is a *feeling* — and it was **taught**, the way "hot soup is hot" was taught. A mind is not a lookup;
it is a loop: hold a little in front of you, call the rest from memory, compare, reject, search again, until the
feeling says stop.

**The worked example, which is also the test case:** asked whether a car is orange, the first thing that happens is
not a calculation — it is *"what does this car's colour look like?"* An apple appears in thought; compare; no. So
the next query becomes *"what does this colour look like, **plus it's not a tone of red apple**"*. Loop until
"orange fruit" appears. Then the feeling is only **"similar," not "sure"** — enough to answer *orange* casually,
but under pressure the loop runs again and returns **"neon orange."**

**And the same for knowledge:** *q = mc∆t* feels right not because it is true, but because all of memory says it
is consistent. It "makes sense."

**The architecture in the author's words:**
- **neocortex** takes input + previous memory state and asks *"is there anything in memory like this?"*
- **hippocampus** receives that from the neocortex, searches, and sends candidates back
- prediction comes **not** from computing once and taking a softmax, but from **looping until something in thought
  matches** — and once it matches, the hippocampus entry already has description, detail, and other things wired to
  it
- the trade: **lower model complexity, paid for with thinking time**
- explicitly **at the architecture level** — *not* an LLM's chain-of-thought, which only exploits properties of
  language

**What it is not:** not an RNN. In the author's framing, an RNN gives each input a special characteristic along the
time dimension that affects the future (what to keep, what not to keep) — that is a different thing.
RNNs/transformers are at most **the middle state that holds the thought**, never the final architecture.
`[CLAIM]`

---

## 7. Where the search is actually blocked

The author's block, stated directly:

> ในโมเดลเรา ความคิดเราในปัจจุบัน หรือสิ่งที่เรานึกมาจาก hippocampus ทั้งหมดมันไม่ใช่คำ หรือภาพตรงๆ แต่มันคือ pure weight คล้ายๆกับ
> abstract ที่ encode ไว้ เราไม่มี reality image สำหรับมันเลย
>
> เพราะงั้นกุเองก็บอกไม่ได้ว่าจะใช้อะไร ใช้ image compare? ก็ไม่ได้ มันไม่ใช่ image. ใช้ L2 distance? ไม่รู้เหมือนกัน เพราะยังไม่เห็นว่าของสองอย่าง
> มันเก็บยังไง. ใช้ vector compare? ก็อาจจะได้ แต่ก็เหมือนเดิม เรายังไม่รู้ว่า thought จริงๆ มันทำงานยังไง

This is the honest core of the draft: **there is no theory of what a "thought" is made of here yet, so the
comparison operator cannot be chosen.**

> **⚠ SUPERSEDED 2026-08-06 — this section describes the state on 2026-08-05 and is kept as history.** The block
> above was resolved by the author writing the theory down: a thought is a **thesis in three parts** (goal ·
> accumulating fact register · `f(goal, fact)` from memory), concepts are **bundles of attributes** in a space
> where meaning adds and subtracts, and the loop has **two motions** plus the **ladder rule**. There is now also a
> substrate (the CNN arena) and a position on where the loop runs.
> **→ [`the-thought.md`](the-thought.md) is the current document. Read it instead of acting on this section.**
> The comparison operator is still unchosen — but the reason is no longer "no theory of thought"; it is now the
> specific open axes D (the rejection operation), H (the ladder) and J (the substrate).

Two structural proposals were raised against this block — both `[CLAIM]`, both recorded in
[`ideas.md`](ideas.md), and one of them is **contested between the author and the agent** and is written up as an
open axis in [`open-questions.md`](open-questions.md):

1. **The dependency may run the other way** — the metric is the primitive and storage is downstream of it.
2. **Thought vs memory may be different physical quantities** (fast state vs slow weight) — *contested;* the
   author's counter-position (thought *is* weight, in the fast-changing sense) has a literature home of its own
   (**fast weights**) and has not been ruled out.

**Nothing here has been decided by evidence.** That is what draft 7.0 exists to fix, and the first move is a
survey, not a build.
