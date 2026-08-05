# draft 7.0 — operating context (the new live line: the loop of thought)

> Auto-loads when you work in `draft7.0/`. Draft 7.0 is a **SET ZERO**: the search for the *core mechanism of a
> loop of thought*, run **without the analog / online / forward-only constraints** — which come back later, one at
> a time. Draft 6.0 is **folded, not discarded.**
>
> **Evidence status of this entire draft: ZERO.** Nothing here has been measured. Every statement carries a tag
> (below). A future agent that reads a `[CLAIM]` and treats it as a result has already broken the draft.

---

## What draft 7.0 is

**The question:** how does a mind hold something in front of it, call candidates from memory, reject them, and
*keep going until a self-generated signal says stop* — at the level of **architecture**, not at the level of a
language model narrating its own steps (CoT).

**Why it is a separate draft:** draft 6.0 built and validated the **first organ** (the neocortex: SCFF bulk +
closed-form namer) and measured it against the metrics the continual-learning field shares. Those metrics are
real, but **none of them can ever show whether a thinking loop is buildable.** Eleven phases of good results
produced zero evidence about the north star. That is the drift that triggered this draft — see
[`context.md`](docs/context.md).

**What is being searched for, in one line:** a loop whose **query changes when a candidate is rejected**, whose
**goal tightens when a candidate is accepted**, and a **self-generated signal that decides which of the two
happens** — everything else (storage, similarity, retrieval) is available off the shelf. `[CLAIM]`

---

## ⚑ The standing candidate — the FIRST decision, not the final one

**Everything in this draft now points at one big picture.** It was adopted 2026-08-06 and it lives in
[`the-thought.md`](docs/the-thought.md). Read that file before designing, reading or proposing anything.

**In three lines** `[STANDING]`:

1. **What a thought is** — a thesis in three parts: **goal**, an **accumulating fact register**, and
   `f(goal, fact)` returned from memory. Concepts are **bundles of attributes** in a space where meaning adds and
   subtracts; a word is only the shared-world handle for a bundle. → [`the-thought.md`](docs/the-thought.md) §1–3
2. **What the loop is** — **two motions**: hold the goal and accumulate facts until memory matches, then **tighten
   the goal's scope** and run again. And **the ladder rule**: never open with the final question —
   *fruit? → orange? → mandarin?* → [`the-thought.md`](docs/the-thought.md) §4–5
3. **Where it runs** — an **arena, not a benchmark**: the front half of a CNN supplies input numbers that already
   carry meaning, the back half supplies output numbers that already carry meaning, and the loop lives between
   them, given practice at thinking **with nothing hinted to it** — no language, no pre-digested abstraction.
   Plus a **read-bandwidth limit** so iteration is structurally necessary. → [`the-thought.md`](docs/the-thought.md) §9.2, §9.5

**Why the whole draft points one way — the author's reason, verbatim:**

> เหตุผลคือ มันจะเถียงกันไม่ได้ ถ้าไม่มี candidate ที่เป็นภาพใหญ่จริงๆ
>
> หัวข้อที่ใช้ตอนนี้ **ไม่ใช่ final decision แต่มันคือ first decision** ก่อนกุจะไปทำ research เพื่อหา candidate มาสู้มัน

**How a standing candidate is treated — the rule that makes this safe:**

- It is held **because research needs a target to attack**, not because it is proven. Evidence for it is still zero.
- It is **replaced by a better candidate, never by an argument.** "I don't like this" is not a move. *"Here is a
  rival big picture and here is why it is better"* is.
- Every document in this draft **states it as the current position** and marks its own disagreements as such,
  rather than each file quietly holding its own frame. That mess is what this pass fixed.
- When a rival wins, the standing candidate is **replaced wholesale and dated** — not patched into a hybrid nobody
  chose.

---

## Evidence tags — used in every file in this draft

| Tag | Meaning |
| --- | --- |
| `[STANDING]` | Part of **the standing candidate** (above) — the draft's current big picture. Evidence zero; held so research has a target to attack. **Replaced by a better candidate, never by an argument.** |
| `[AUTHOR]` | The **author's own definition of the target**, in his words. Not evidence — and **not the agent's to overturn by argument.** An agent may say "this is hard to build"; it may not substitute its own picture and proceed. Only the author revises it. Lives in [`the-thought.md`](docs/the-thought.md). |
| `[CLAIM]` | Asserted in discussion. **No evidence.** Most of this draft. |
| `[OPEN]` | A genuine unresolved question. Nobody in this repo knows the answer. |
| `[ARG]` | Settled *by argument* between the author and the agent — still never measured. |
| `[CARRIED]` | A **measured** result from draft 6.0 that still stands and is being used as input. |
| `[STRUCK]` | Explicitly rejected with a reason. **Do not re-propose without new evidence.** |
| `[LIT]` | Points at existing outside literature that must be read before we spend our own effort. |

Untagged assertions are a defect. If you add a claim, tag it.

---

## Status

**Pre-research, with a standing candidate.** No experiments, no code, no roadmap. Evidence: **zero, everywhere.**

What changed on 2026-08-06: the draft stopped being a set of open questions with no position and acquired a **first
decision** (above). The next move is therefore **not** an open-ended survey — it is a **reading checklist whose job
is to find candidates that can beat the standing one.** Reading with a target, not reading to get oriented.

**That checklist is now written: [`reading-checklist.md`](notes/reading-checklist.md)** — tiered (T0 must / T1 should /
T2 if-time / T3 later), with a **4-day lane** (16 h/day, ahead of a department presentation) and, for every part of
the standing candidate, **the prior art that may already have built it** and **what would kill the move**. Its §0
is the single highest-value page in this draft right now.

The survey axes in [`open-questions.md`](docs/open-questions.md) Part 1 are still the map of what is unknown; what
changed is their **purpose** — they are now attack surfaces on a stated position rather than a general syllabus.

> **On the deleted `learning-roadmap.md`** — resolved 2026-08-06, recorded so nobody restores it. A build-first
> roadmap did exist (six blocks, governing rule *"every block ends in a notebook that produces a number"*). **The
> author deleted it on purpose:** *"roadmap อันเก่ามันยังไปผิด direction อยู่"* — it was written before the ladder rule
> and the two-motion loop existed, so it was planning builds against half a picture. **Do not reconstruct it.** Its
> pain points were harvested into [`the-thought.md`](docs/the-thought.md), which is what replaced it.
>
> **What is commissioned instead:** a **new roadmap that is a reading checklist** — topics, books, theories, papers
> to read or re-read in order to build a loop of thought. The author's reason, verbatim: *"ตอนนี้กุแม่งตัวเปล่าเลย ทั้งหมดคือ
> pure intuition without anything ref … กุแค่เห็นภาพมาว่าควรจะเป็นยังไง แต่กุไม่มั่นใจว่าความคิดกุมันถูกที่สุดจริงๆ แล้วใช่มั้ย"* Not
> open-ended reading: a checklist whose job is to test an existing intuition against what the world already knows.

---

## The discipline (this draft's version of the build rules)

1. **Search first, constrain later.** Use anything during the search — backprop, BPTT, GPU, transformers. This is
   the project's own methodology rule 7 (*ideal first, realism later*) raised one level. `[ARG]`
2. **But keep the cost column.** Every mechanism found gets one line: *what would it cost to make this local /
   online / backward-free?* A **tiebreaker, never a gate**. The moat is lost by never re-imposing the constraints,
   not by dropping them while searching. `[ARG]`
3. **Loop metrics only.** No AA / BWT / retention / prequential accuracy as the *goal metric*. Those are precisely
   the ruler that caused the turn. A loop is judged by loop properties (step count vs difficulty, halt calibration,
   steps-vs-memory-size scaling).
4. **Read before building.** The author's own words: *"we have to go learn directly how far the world has actually
   got"*. Most of this draft's questions are answered somewhere already; find that first.
5. **One draft-6 rule survives unchanged: failures are data.** A mechanism that does not loop is a result.
6. **Hold a standing candidate, and attack it with rivals — not with opinions.** `[ARG]` A draft with no position
   cannot be argued with, so it drifts; a draft with a position can be *beaten*, which is how it moves. State the
   current big picture everywhere, mark disagreement as disagreement, and replace it only when a rival big picture
   wins on the merits.

---

## `docs/` vs `notes/` — two folders, two jobs. Do not mix them.

| | [`docs/`](docs/) | [`notes/`](notes/) |
| --- | --- | --- |
| **What** | The **frozen record** — the first decision, tagged, dated, with the reasoning and the arguments that were withdrawn | The **live working surface** — the prototype being polished |
| **Who for** | Every agent's base. Read this to know what the position *is* | The author, mid-thought. Read this to know what he's *doing right now* |
| **Editing** | Careful and dated. Supersede in place, never quietly rewrite; `the-turn.md` is not to be edited at all | Free. The author edits it without asking, and so may an agent he asks |
| **Style** | Evidence tags on everything (`[STANDING]` `[AUTHOR]` `[CLAIM]` …) | No ceremony. Fresh ideas, open slots, unfinished sentences are fine |

**The same picture lives in both**, on purpose: [`docs/the-thought.md`](docs/the-thought.md) is the record,
[`notes/the-model.md`](notes/the-model.md) is the working version. **When something in `notes/` hardens, promote it
into `docs/` with a date.** Never edit `docs/` to match `notes/` silently — that destroys the record `docs/` exists
to keep.

*(`CLAUDE.md` and `README.md` stay at the draft root: `CLAUDE.md` **must** be at `draftN/` to auto-load under the
repo's 3-layer context hierarchy, and `README.md` is the folder's front door.)*

---

## Router

| Want… | Read |
| --- | --- |
| **What I'm building right now, in plain terms** (living, editable) | [`notes/the-model.md`](notes/the-model.md) |
| **⚑ THE STANDING CANDIDATE — what a thought is, the loop's shape, the arena, the base spec** (frozen record) | [`docs/the-thought.md`](docs/the-thought.md) `[STANDING]` `[AUTHOR]` |
| **Why we turned — the whole story, cold** (written pre-candidate; §7 superseded) | [`context.md`](docs/context.md) |
| **The conversation that produced the turn** (historical, 2026-08-05 — do not update) | [`the-turn.md`](docs/the-turn.md) |
| **What is unknown — now the attack surfaces on the standing candidate** | [`open-questions.md`](docs/open-questions.md) |
| **Component-level candidate mechanisms** (subordinate to the standing candidate) | [`ideas.md`](docs/ideas.md) |
| **What to read, in what order, and what would kill each move** | [`reading-checklist.md`](notes/reading-checklist.md) `[LIT]` |
| **2-minute version + the 11 tripwires** | [`handoff.md`](docs/handoff.md) |
| The folded previous line | [`../draft6.0/CLAUDE.md`](../draft6.0/CLAUDE.md) |
| The origin of the whole idea | [`../docs/essence/the-essence2.md`](../docs/essence/the-essence2.md) |

---

## ⚠ The one thing this draft is most likely to lose

The author's stated fear, verbatim: *"draft7.0 thesis มันอาจจะโดน push กลับไปทางเก่าได้อ่ะ"*

The old way is **gravitational** — it has eleven phases of real results, a validated object, and a metric suite
that produces clean numbers on demand. This draft has none of that, and it will feel worse to work on for a while.
**That asymmetry is the danger, not a signal.** The tripwire list in [`handoff.md`](docs/handoff.md) exists for exactly
this; read it before proposing anything that sounds like a return.
