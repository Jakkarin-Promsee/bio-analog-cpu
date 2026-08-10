# draft 7.0 — operating context (the live line: bounded-bandwidth visual reasoning)

> Auto-loads when you work in `draft7.0/`. **Read [`docs/direction.md`](docs/direction.md) before designing,
> proposing, or arguing anything.** Draft 6.0 is folded, not discarded.
>
> **Evidence status of every mechanism in this draft: ZERO.** Nothing here has been run. Every statement carries a
> tag. An agent that reads a `[CLAIM]` and treats it as a result has already broken the draft.
>
> **⚠ 2026-08-10 — the direction was replaced wholesale.** The previous position (the loop of thought: goal + fact
> register + `f(goal, fact)`, two motions, the ladder rule, the CNN-halves arena) is in
> [`.archive/`](.archive/README.md). Do not cite it as current. Do not reconstruct it.

---

## What draft 7.0 is, now

**The question:**

> ### "ต้อง optimize ยังไง ถ้า attention size เล็ก แต่โมเดลเลือกได้ว่าจะมองอะไรและจดอะไร"

*How do you optimize when attention size is small, but the model chooses what to look at and what to write down?*
Outside name: **bounded-bandwidth visual reasoning**.

**Underneath it, the real question:** *what is latent space, and how do you build one that does real work?*

**Why this and not the old picture.** Latent reasoning works in text and **fails in images** — settled by causal
mediation (Li et al., ICML 2026): latent visual tokens are placeholders, `α ≈ 0`, `β ≈ 0`. The cause is not
training. **An image can be classified in parallel without depth, so a 1-hop pretrained shortcut beats a 2-hop path
starting from random, and the optimizer is right to prefer it.** Therefore **the objective is what has to change**,
not the architecture — and the read has to be bandwidth-limited so that iteration is an *information bound* rather
than a hope.

**ONE thing is being built** `[STANDING]` — `s2`'s **latent predicts position AND scale `(l, s)`**. It has two
halves because it answers two questions, but it is one loop:

| | Half | What it does |
| --- | --- | --- |
| **the hand** | **latent predicts `(l, s)`** — §3.1 | *how a look is taken.* `z_t → (l_t, s_t) → ρ(x, l_t, s_t) → z_{t+1}`, read from an **image pyramid** so a wide `s` is genuinely blurry and the model cannot cheat by zooming out forever |
| **the brain** | **associative retrieval of an address** — §3.2 | *where to look next.* `l = P·softmax(β Mᵀq)` on a thumbnail — modern Hopfield, but it returns a **coordinate** instead of content. No gradient walk, no wall, no RL |

**Build order: §3.1 with a plain MLP first, then swap in §3.2 as a single-variable change.** → [`docs/direction.md`](docs/direction.md) §3

**⚠ `s1`'s object-bound dynamic latent (slot attention / binding) is §3.3 — LATER, NOT NOW.** It is the
multi-latent generalization of the same idea and it is real, but one acting latent is enough to test the core
claim. The author's scoping: *"กุไม่ได้จะ pick หลายๆทำในทีเดียว ตอนนี้ที่กุเล็งไว้อย่างเเรกคือ s2 … ก่อน เพราะว่ามันใหญ่ที่สุด"*
**Any plan that runs binding and `(l, s)` at the same time has misread this draft.**

---

## Three surfaces. Do not mix them.

| | [`docs/`](docs/) | [`notes/vault/`](notes/vault/) | [`notes/`](notes/) |
| --- | --- | --- | --- |
| **What** | The **frozen position** — [`direction.md`](docs/direction.md), dated and tagged | The **evidence base** — 15 study notes, ~770 KB, plus [`INDEX.md`](notes/vault/INDEX.md) over them | The **live working surface** |
| **Who for** | Every agent's base. Read this to know what the position *is* | Anyone who needs to know what is already known, closed, or read | The author, mid-thought |
| **Editing** | Careful and dated. **Supersede in place, never quietly rewrite** | The notes are his study record — don't rewrite them. `INDEX.md` is maintained | Free |

Plus [`.archive/`](.archive/README.md) — the pre-2026-08-10 position. **Read-only, never cited as current.**

### ⚠ The reading budget — the rule that keeps sessions cheap

**The vault is ~770 KB. Never read raw vault files to "get oriented."** The whole point of
[`notes/vault/INDEX.md`](notes/vault/INDEX.md) is that you don't have to:

1. **§1** (the one-screen map) tells you which file holds what, and how deep it goes.
2. **§3** (per-file dossiers) gives you the claims, the killed misconceptions, and the declared gaps — enough to
   answer most questions without opening the file.
3. **§6** (the merged Closed / Pending / Open map) is what you check **before proposing anything**. Many obvious
   ideas are already closed by published work.
4. Only then open **one** vault file, and only the section §3 names as its highest-value page.

For heavy multi-file lookups, dispatch the `Explore` sub-agent — its own context window, returns the conclusion.

---

## Evidence tags — authoritative definitions, used in every file in this draft

| Tag | Meaning |
| --- | --- |
| `[STANDING]` | Part of **the direction** ([`docs/direction.md`](docs/direction.md)). **Evidence zero.** Held so work has a target. **Replaced by a better position, never by an argument** — "I don't like this" is not a move; *"here is a rival picture and why it is better"* is |
| `[AUTHOR]` | The **author's own words** for the target. Not evidence — and **not an agent's to overturn by argument.** An agent may say "this is hard to build"; it may not substitute its own picture and proceed |
| `[CLAIM]` | Asserted. **No evidence from this repo.** Most of this draft |
| `[LIT]` | Read literature. **⚠ Everything dated 2026 in the vault was found by search, not read in the original** — three separate files say so. Verify before citing outward |
| `[OPEN]` | Genuinely unresolved. Nobody in this repo knows |
| `[ARG]` | Settled *by argument* between author and agent — still never measured |
| `[CARRIED]` | A **measured** draft-6.0 result that still stands and is being used as input |
| `[STRUCK]` | Explicitly rejected with a reason. **Do not re-propose without new evidence** |

Untagged assertions are a defect. If you add a claim, tag it.

---

## Status

**Pre-experiment, with a direction and six kill criteria.** No code, no runs. Evidence: **zero for every mechanism.**

What is different from the archived direction — and this is the only thing that is different:

| | archived | now |
| --- | --- | --- |
| The problem | asserted from intuition, **no references at all** | **diagnosed, published, causal-mediation evidence** |
| The field | unknown | **mapped** — ~19 topics closed, 11 open and ranked |
| The mechanisms | invented | **read** — Slot Attention, STN, DRAW, deformable, Hopfield retrieval, with known failure modes |
| Kill criteria | none | **six, two of them phase gates** |
| Evidence | zero | **zero** |

**Next move: Phase 0** — build the pointer-chase generator, fire it at an existing VLM with **no training**, measure
accuracy vs chain length `k` plus a blind baseline. *Kill criterion: if SOTA handles `k = 4` easily, the problem
isn't real — stop.* → [`docs/direction.md`](docs/direction.md) §6

---

## The discipline

1. **Search first, constrain later.** Use anything during the search — backprop, BPTT, GPU, transformers. The
   project's methodology rule 7 (*ideal first, realism later*) raised one level. `[ARG]`
2. **But keep the cost column.** Every mechanism gets one line: *what would it cost to make this local / online /
   backward-free?* A **tiebreaker, never a gate.** The moat is lost by never re-imposing the constraints, not by
   dropping them during the search. `[ARG]`
3. **Check [`INDEX.md`](notes/vault/INDEX.md) §6 before proposing anything.** If it is in the Closed column, it is
   closed by published work and proposing it is a waste. If it is in Pending, say who else is on it.
4. **Bounded-bandwidth metrics, not continual-learning metrics.** No AA / BWT / retention / prequential accuracy as
   a goal metric — that ruler is what caused the turn away from draft 6. Judge by: **accuracy vs read-fraction**,
   `s_t` vs `t` (does coarse-to-fine emerge), `Var_t[l_t]` (location collapse), the **slot drop test**, and
   **sequential vs parallel-`k` at matched read budget.**
5. **Every experiment names the kill criterion it can trip.** The six are in
   [`docs/direction.md`](docs/direction.md) §5. An experiment that cannot fail is not an experiment.
6. **Assume the shortcut is present until measured otherwise.** *Anything avoidable will be avoided.* Run the
   four-point checklist ([`13-latent-space-and-shortcut.md`](notes/vault/13-latent-space-and-shortcut.md) §9.4)
   against every task before trusting any result on it. **The architecture does not save you from a bad benchmark.**
7. **Failures are data.** The one draft-6 rule that survives unchanged. A mechanism that does not loop is a result.
8. **Hold a position, attack it with rivals — not with opinions.** A draft with no position cannot be argued with,
   so it drifts. A draft with one can be *beaten*, which is how it moves. `[ARG]`

---

## Router

| Want… | Read |
| --- | --- |
| **⚑ THE DIRECTION — the mechanism, the kill criteria, the plan** | [`docs/direction.md`](docs/direction.md) `[STANDING]` `[AUTHOR]` |
| **What is already known / closed / open** — check before proposing | [`notes/vault/INDEX.md`](notes/vault/INDEX.md) §6 |
| **How deep the study actually went, per topic** | [`notes/vault/INDEX.md`](notes/vault/INDEX.md) §1, §3 |
| The two idea files the direction came out of | [`s1-opened-topic-ideas.md`](notes/vault/s1-opened-topic-ideas.md) · [`s2-opened-topic-ideas.md`](notes/vault/s2-opened-topic-ideas.md) |
| The shortcut diagnosis — why latent fails on images | [`13-latent-space-and-shortcut.md`](notes/vault/13-latent-space-and-shortcut.md) §4–§9 |
| Glimpse attention end to end, and why RAM died | [`12-recurrent-visual-attention.md`](notes/vault/12-recurrent-visual-attention.md) |
| How to work with the department advisors | [`notes/advisor.md`](notes/advisor.md) + [`s2`](notes/vault/s2-opened-topic-ideas.md) §8 (the script for this direction) |
| **The pre-2026-08-10 position** (loop of thought) — history, never current | [`.archive/README.md`](.archive/README.md) |
| Why we turned away from draft 6.0 (2026-08-05) — still true as history | [`.archive/docs/context.md`](.archive/docs/context.md) |
| The folded previous line | [`../draft6.0/CLAUDE.md`](../draft6.0/CLAUDE.md) |
| The origin of the whole idea | [`../docs/essence/the-essence2.md`](../docs/essence/the-essence2.md) |

---

## ⚠ The three ways this draft gets lost

1. **Pushed back to the old way.** The author's stated fear: *"draft7.0 thesis มันอาจจะโดน push กลับไปทางเก่าได้อ่ะ"*
   Draft 6.0 is gravitational — eleven phases of real results, a validated object, and a metric suite that produces
   clean numbers on demand. This line has none of that yet and will feel worse to work on for a while. **That
   asymmetry is the danger, not a signal.**
2. **Absorbed by the crowd.** *"Bolt a latent visual token onto a VLM and collect VQA points"* is **saturated** —
   eight named systems already did it, and ICML 2026 already showed the tokens are placeholders. Any proposal that
   drifts toward it has lost the thread. The distinction that must survive: **the latent emits coordinates and is
   differentiable end to end** — not tool calls mediated by language.
3. **Doing everything at once.** The author's own boxed warning. **One thing: `(l, s)`.** Then the Hopfield retina
   as a single-variable swap. Binding waits; the part-whole decomposition idea stays **demoted to an analysis
   section, never a method section.** The benchmark comes **first**, because it is the only deliverable that
   survives the model failing.
