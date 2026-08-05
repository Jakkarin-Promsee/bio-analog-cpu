# draft 7.0 — the loop of thought

> **Status: pre-research, with a standing candidate.** No experiments, no code, no roadmap. Evidence level:
> **zero**, everywhere. Opened 2026-08-05; the first decision adopted 2026-08-06.

---

## What this is

A **set zero**. Draft 6.0 built and validated the project's first organ — a neocortex that learns online, forward-
only, without a backward pass. Draft 7.0 goes after the thing that organ was always meant to serve and never
touched: **a loop of thought.**

Hold something in front of you. Call candidates from memory. Compare. Reject. Search again — until a signal you
generate yourself says *stop*. At the level of **architecture**, not a language model narrating its own steps.

The search runs **without** the analog / online / forward-only constraints. They are not abandoned; they are
**re-imposed one at a time after the mechanism exists** — the project's own methodology rule (*ideal first, realism
later*), applied at the architecture instead of the device.

**Draft 6.0 is folded, not discarded.** SCFF + SLDA works, and the eleven-phase result stands. It is picked back up
once there is a loop core to put it under.

---

## ⚑ The standing candidate

Since 2026-08-06 this draft holds **one big picture**, and every document points at it. It lives in
[`the-thought.md`](docs/the-thought.md).

1. **A thought** is a thesis in three parts — **goal**, an **accumulating fact register**, and `f(goal, fact)`
   returned from memory. Concepts are **bundles of attributes** in a space where meaning adds and subtracts; a word
   is only the shared-world handle for a bundle.
2. **The loop** has **two motions** — hold the goal and accumulate facts until memory matches, then **tighten the
   goal's scope** and run again. And it **never opens with the final question**: *fruit? → orange? → mandarin?*
3. **The arena** — not a benchmark. The front half of a CNN supplies input numbers that already carry meaning, the
   back half supplies output numbers that already carry meaning, and the loop lives between them, practising
   thinking **with nothing hinted to it** — no language, no pre-digested abstraction ladder. A **read-bandwidth
   limit** makes iteration structurally necessary.

**It is the FIRST decision, not the final one.** The author's reason: *"มันจะเถียงกันไม่ได้ ถ้าไม่มี candidate ที่เป็นภาพ
ใหญ่จริงๆ"* — a draft with no position cannot be argued with, so it drifts. Evidence for it is **zero**. It is
**replaced by a better candidate, never by an argument**, and the research now underway exists to find rivals that
can beat it.

---

## Why the turn happened

Phase 11 finished with the best results the project had produced — and the author did not start Phase 12 for a
month. The reason was not in any number. It was in the **ruler**: every metric the project used (average accuracy,
BWT, retention, energy, prequential accuracy) is one the continual-learning field shares, and **none of them can
show whether a thinking loop is buildable**. Eleven phases of correct, honest work produced zero evidence about the
project's actual destination.

The target never moved. The evidence drifted away from it, with no per-phase signal to catch it.

Full story: [`context.md`](docs/context.md).

---

## What is actually missing

Storage, similarity and retrieval can be bought off the shelf. What nobody hands you:

1. **the query update rule under rejection** — how the question changes when a candidate is rejected, so the next
   iteration does different work (plain associative memory returns the same answer forever)
2. **the goal update rule on acceptance** — a match ends the *rung*, not the process: the goal then tightens
   (*is it an orange?* → *which orange?*) and the loop runs again at higher resolution
3. **the feeling** — the self-generated signal that decides which of the two happens

All `[CLAIM]`. These are the three gaps **the standing candidate has to fill**, and establishing whether the framing
is even right is the first job.

---

## Two folders

- **[`docs/`](docs/) — frozen.** The first decision, tagged and dated, with the reasoning and the arguments that
  were withdrawn. Every agent's base. Supersede in place; never quietly rewrite.
- **[`notes/`](notes/) — live.** The prototype being polished, plus the reading checklist. No ceremony, open slots,
  edited freely. **When something here hardens, it gets promoted into `docs/` with a date.**

The same picture lives in both on purpose: [`docs/the-thought.md`](docs/the-thought.md) is the record,
[`notes/the-model.md`](notes/the-model.md) is the working version.

---

## Read in this order

| # | File | What it is |
| --- | --- | --- |
| 0 | [`notes/the-model.md`](notes/the-model.md) | **what I'm building right now**, in plain terms — the fastest way in |
| 1 | [`handoff.md`](docs/handoff.md) | two-minute arrival + **the 11 tripwires** (how this draft gets pushed back to the old way) |
| 2 | [`the-thought.md`](docs/the-thought.md) | **⚑ THE STANDING CANDIDATE** — what a thought is, the loop's shape, the arena, and the base spec. Read before designing, reading or proposing anything |
| 2b | [`reading-checklist.md`](notes/reading-checklist.md) | what to read, tiered, with a **4-day lane** — and for every move, **the prior art that may already have built it** |
| 3 | [`context.md`](docs/context.md) | why we turned — the full narrative *(written before the candidate existed; §7 is superseded)* |
| 4 | [`open-questions.md`](docs/open-questions.md) | what is unknown — now the **attack surfaces** on the standing candidate — plus the struck list |
| 5 | [`ideas.md`](docs/ideas.md) | **component-level** candidate mechanisms, subordinate to the standing candidate; all unvalidated |
| 6 | [`the-turn.md`](docs/the-turn.md) | the 2026-08-05 discussion record *(historical — deliberately not updated)* |
| — | [`CLAUDE.md`](CLAUDE.md) | operating rules for agents working in this draft |

**Next move:** run [`reading-checklist.md`](notes/reading-checklist.md) — reading whose job is to produce **rivals that
can beat the standing candidate**, with a target rather than for orientation.
