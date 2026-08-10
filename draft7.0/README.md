# draft 7.0 — bounded-bandwidth visual reasoning

> **Status: pre-experiment, with a direction and six kill criteria.** No code, no runs. Evidence for every
> mechanism: **zero**.
> Opened 2026-08-05 · direction replaced wholesale **2026-08-10**.

---

## The question

> ### "ต้อง optimize ยังไง ถ้า attention size เล็ก แต่โมเดลเลือกได้ว่าจะมองอะไรและจดอะไร"
>
> *How do you optimize when the attention size is small, but the model gets to choose what to look at and what to
> write down?*

A small attention that **roams over a large real input** — glimpses — instead of a context window that swallows the
whole thing. Underneath it, the question both halves of this work are walking toward: **what is latent space,
actually, and how do you build one that does real work?**

**This is a change of problem, not a change of solution.** The immediate consequence: every result that assumes the
model sees everything becomes the special case `attention size = N`, rather than a competitor.

---

## Why

Latent reasoning works in text and **fails in images.** Not "works badly" — *Imagination Helps Visual Reasoning,
But Not Yet in Latent Space* (Li et al., ICML 2026) shows by causal mediation that latent visual tokens are
**placeholders**: `α ≈ 0`, `β ≈ 0`. Explicit text imagination beats them.

**The cause is not training.** An image can be classified **in parallel, with no depth** — so a 1-hop pretrained
path (`v → ans`, live at every layer) beats a 2-hop path starting from random (`v → z → ans`), and the optimizer is
right to prefer it. Text does not have this problem because text is already symbolically dense; a GSM8K problem has
no 1-hop shortcut. **Vision has no GSM8K.**

So the thing to change is **the objective**, not the architecture — and the read has to be bandwidth-limited, so
that iterating is an *information bound* rather than something you hope the model chooses to do.

---

## What is being built — one thing, two halves

**The latent predicts position AND scale `(l, s)`** — the latent gets to *act* on the image.
`z_t → (l_t, s_t) → ρ(x, l_t, s_t) → z_{t+1}`, reading from an **image pyramid** so a wide `s` is genuinely blurry
and the model cannot cheat by zooming all the way out. Zoom out = explore, zoom in = exploit. Plot `s_t` against
`t`: a downward slope is **coarse-to-fine that emerged rather than being imposed.**
The gap it walks into: **DRAW learned its scale in 2015, and nobody followed for ten years.**

That is *how a look is taken*. The other half is *where to look next* — **associative retrieval of an address**:
`l = P·softmax(β Mᵀq)` on a thumbnail. Modern Hopfield retrieval, except it returns a **coordinate** instead of
content, and the real thing is then read at full resolution. No gradient walk across the image, so no wall and no
local minimum; differentiable, so no RL.

**Same loop, split only because they answer different questions.** Build order: the `(l, s)` half first with a
plain MLP, then swap the retina in as a single-variable change.

> **`s1`'s object-bound dynamic latent — slot attention, binding — is deliberately *later*.** It is the
> multi-latent generalization of the same idea (each latent holding one object *and* carrying its own `(l, s)`),
> and it is real. But one acting latent is enough to test whether acting fixes the placeholder problem, and running
> two unproven mechanisms together means they fail together with no way to tell which one did it.

Full statement, with mechanisms, risks and the six kill criteria: **[`docs/direction.md`](docs/direction.md)**.

---

## The plan

| Phase | What | Kill criterion |
| --- | --- | --- |
| **0** (1–2 wk) | Build the **pointer-chase** benchmark — a linked list rendered as an image, chain length `k` on a dial. Fire it at an existing VLM **with no training.** Measure accuracy vs `k` + a blind baseline | SOTA handles `k = 4` easily → the problem isn't real → **stop** |
| **1** (~1 mo) | The smallest model that could work: glance → address retrieval → `(l_t, s_t)` → pyramid read → loop. One latent vector, no LLM, no pretraining. Baseline that matters: **parallel-`k`** | sequential ≯ parallel → no reasoning is happening → **go fix the dataset, not the model** |
| **2** | Real VLM, adaptive halting, V\*Bench / HR-Bench | **do not start here** — nothing is debuggable at this end |

**Phase 0 is insurance:** the model failing still leaves a benchmark other people can use.

---

## Where things are

| | |
| --- | --- |
| [`docs/direction.md`](docs/direction.md) | **⚑ the position** — dated, tagged, with what would kill it |
| [`notes/vault/`](notes/vault/) | **the evidence base** — 15 study notes, ~770 KB, three days |
| [`notes/vault/INDEX.md`](notes/vault/INDEX.md) | **start here for the vault** — depth map, per-file dossiers, and the merged Closed/Pending/Open status of the field. **Do not read raw vault files to get oriented** |
| [`notes/advisor.md`](notes/advisor.md) | how to work with the department advisors |
| [`CLAUDE.md`](CLAUDE.md) | operating rules for agents in this draft |
| [`.archive/`](.archive/README.md) | **the pre-2026-08-10 position** (the loop of thought) — history, never cited as current |

---

## What happened on 2026-08-10

The draft's previous position — *the loop of thought*: a thought as goal + fact register + `f(goal, fact)`, a
two-motion loop, the ladder rule, an arena between two halves of a CNN — was held deliberately at evidence zero, as
**the first decision, not the final one**, so that research would have something to attack. The author's own note on
what those documents were:

> ของพวกนั้นเป็นเเค่ direction ที่ตั้งมาไว้ให้ debate เพื่อค้นคว้าข้อมูล เเบบไม่มีอะไร ref เลยเฉยๆ เเล้วส่วนใหญ่โดนตีตกไปหมดเเล้ว

Three days of study produced the rivals. The position was **replaced whole and dated**, which is exactly what the
draft said it would do when that happened. What survived is named in [`docs/direction.md`](docs/direction.md) §7 —
the read-bandwidth limit above all, which is no longer a side condition but the centre of the whole direction.

**And one thing did not change:** the evidence level. It was zero before and it is zero now. The difference is that
the *problem* is no longer invented.
