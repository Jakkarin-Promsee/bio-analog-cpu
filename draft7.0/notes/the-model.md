# the model — what I'm building, right now

> **Working notes. Alive. Messy on purpose.**
> This is the prototype being polished — the fresh version of
> [`../docs/the-thought.md`](../docs/the-thought.md) with the ceremony stripped out. No evidence tags, no
> supersede-boxes, no arguing with the record. Just: *what am I doing, and what does each part look like.*
>
> `docs/` is frozen — the first decision, the base every agent reads. **This file is where new ideas land**, so
> edit it freely and don't ask permission. When something here hardens, it gets promoted into `docs/`.
>
> **Every section ends with an open slot.** Fill them in after the reading — that's what they're for.

---

## The one-liner

A thing that sits between two halves of a CNN and **thinks**: holds a question, pulls candidates from memory,
rejects them, narrows the question, and goes again — until it lands.

Not a classifier. Not a better pooling layer. A place where a **loop of thought** can be watched running on dense
numbers that mean something, with nobody hinting anything to it.

---

## Part 1 — what a thought is

A thought is a **thesis with three parts**:

| | | |
| --- | --- | --- |
| **goal** | what am I trying to prove right now | *"is the thing in front of me an orange?"* |
| **fact** | everything established so far — from the senses *and* from ruling things out | *round · orange · a fruit · definitely not an apple → cut every non-fruit and everything red* |
| **memory output** | `f(goal, fact)` — what memory returns given both | *"orange fruit"* |

And a concept is **not an atom** — it's a **bundle**:

```
orange  =  fruit + round + orange-coloured + sour + seeds inside + …
```

The **word** "orange" is just the handle. It's the world's shared shorthand for the bundle. We call every one of
them an orange even when it's a mandarin or an orange-skinned lemon, because *orange* is the most accurate label
at the resolution the moment asks for. Match something orange-ish → call it an orange → drill in only if you have
to.

**Same root as the transformer vector.** Meanings that add and subtract. That's why this is reachable at all
instead of being some mystical primitive.

> **after reading —**
>

---

## Part 2 — how the loop runs

**Two motions. Everything I wrote before only had the first one.**

```
        ┌──────────────── motion 1: same goal, accumulate ────────────────┐
        │                                                                │
   goal ──▶ query memory with (goal, fact) ──▶ candidate ──▶ compare ────┤
        │                                          │                     │
        │                                       reject ──▶ update fact ──┘
        │                                          │
        │                                        match
        │                                          ▼
        └──────── motion 2: tighten the goal ◀─────┘        carry fact forward
                        "is it an orange?" → "which orange?"
```

**Motion 1** — hold the goal, keep adding facts and cutting choices, until what memory returns matches the facts.
**Motion 2** — a match doesn't end the process, it ends *this rung*. The goal then gets sharper and the loop runs
again at higher resolution.

### The ladder rule — the part I'm most sure about

**Never open with the final question.**

Asked *"is that a mandarin?"*, don't set `goal := mandarin` and go hunting. Go:

```
is it a fruit?  →  is it an orange?  →  is it a mandarin?
```

Attacking the specific question head-on means searching from the definition of "mandarin" with nothing to prune
against — the right feeling is nearly unreachable, and it's **brute-force search over big data, not thinking, even
when the answer comes out right.**

Yes this is inefficient on easy tasks. Accepted, on purpose. I'm laying a foundation, not racing a transformer at
vector arithmetic. The point is what it grows into: *watch a moving object → connect it to a law of physics →
predict where it goes.*

> **after reading —**
>

---

## Part 3 — the arena (where it runs)

**Not a benchmark. A place to watch the loop run.** I'm not trying to beat pooling, and "it's only as good as
pooling" doesn't refute anything.

```
image → [ CNN front half ] → lots of components (multi-layer, multi-scale, spatial)
                                        │
                                        ▼
                              ⟨  loop of thought  ⟩        ← reads only k components per step
                                        │
                                        ▼
                             [ CNN back half / head ] → the answer, in class-space
```

**Why cut the CNN in half.** The back half's heavy pooling is what destroys the linkage between components. Cut it
off and each conv layer stays what it actually is — **a specific lens that says something about the identity of
the thing.** Denser and more digestible than language.

**Why keep the back half at all.** Because the loop's job is **not classification**. If input were the front half
and output were the class, the model would have to be a loop *and* a fine-grained classifier at the same time —
it'd break. Send the output onward and the loop has exactly one job: **make vectors and think on vectors.** The
back half is the translator from abstract numbers into classes a human can see. So: **32 → 20**, not n → 1.

> ⚠ This is the same split as draft 6: `SCFF bulk (vector space) + SLDA namer (names it)`. Same insight — never
> make the representation learner also be the classifier — carried across a total change of substrate. Last year
> isn't thrown away; the *division of labour* survived.

**Why images and not text.** Language hands you the **handle** and hides the **bundle**. "Apple" is exactly the
shared-world label — the compression already happened, the attributes were thrown away, and getting *round / red /
sweet / has seeds* back means reconstructing from billions of tokens what language deleted. **An image hands you
the bundle before anyone named it.** Roundness, colour, texture are physically in the pixels.

Also: text would hand me the generality hierarchy for free (hypernyms, WordNet), so a ladder found there would be
someone else's tree. In images nobody hands it over. That's what makes finding one a *result*.

**Why the CNN specifically:** complexity is a **dial**. Two-layer net on tiny images and a deep ResNet are the same
rig at different settings. Start absurdly small, scale without changing rigs. And it extends to video later.

### The read-bandwidth limit

The loop sees only **k components per step**, out of `C × H × W`.

Without it, the only training signal in the rig is *"produce the tensor the back half expects"* — supervised
regression — and gradient descent will just find the cheapest **feedforward map**. Nothing rewards a second
iteration. The rig is right; its default objective selects *against* looping.

With it:
- when the evidence an input needs exceeds *k*, **one pass provably can't do it** — an information bound, not a
  hoped-for difficulty
- **step count varies per input by itself** — clean image resolves in a few reads, cluttered one takes many
- the **fact register** has to exist (you must remember what you already read)
- the **query update** has to exist (where to look next depends on what you just found)

Which is also just how it actually works: you don't see everything at once. You look, you recall, you look again.

> **after reading —**
>

---

## Part 4 — the pieces, concretely

*(first ideas, for hitting. no math proof yet.)*

### 4.1 Thought = a transformer

Vectors grounded in conv components instead of words. If `Conv[x][y][0]` holds line features and `Conv[x][y][1]`
holds curvature, the transformer builds its vectors starting from there, so those two become **addable**.

*(mechanically: each channel `c` carries a learned embedding `e_c`, the activation is its weight, so what's added
is `Σ_c a_c(x,y) · e_c`. Activations are scalars — they can't be added as meaning. Classical prior art for exactly
this step: bag-of-visual-words → VLAD → NetVLAD.)*

- **goal** = the pattern *"is this an x, or not?"* — a **comparison-shaped** question, not a search command
- **fact** = data vectors that **scope which part of memory to search**

Why split the goal out at all: honestly, because I don't know how to build **question vectors on an image
dataset**. This shape seems more stable. `open`

### 4.2 Memory ≈ a vector database

**The taxonomy is the co-occurrence structure itself, not language.** Orange image has `Conv[x][y][0]` and
`[x][y][1]`; sphere image has `[x][y][0]` → *orange* and *sphere* are connected **through that channel**. Straight
from image data. No WordNet, no human tree.

> ⚠ Which is exactly why the backbone **must be self-supervised** (DINO / MAE / SimCLR). An ImageNet-classification
> backbone organises its features by ImageNet's label tree — which *is* WordNet — and then "the taxonomy came from
> images" is quietly false. This is the strongest claim in the whole design; don't hand it away for convenience.

**The ladder = ranked property checks.** Asked *"is it a mandarin?"* (≈ round + orange + rough peel, from image
sensory alone), check the properties one at a time, ordered by **how heavily connected each one is**:

```
round (very connected)  >  orange (a colour)  >  rough peel (a texture)
```

…until a fact produces the right feeling. Top-ranked components can be **summed**, since the sphere vector already
adds to the orange vector.

**The whole point:** this is **not** brute force that adds everything at once and searches. It's **abstract
stacking search** — one fact at a time.

**And the ordering is conditional, not global.** Connection-count would be a fixed property of memory *only if the
input were fixed*. It isn't — the input is `previous goal + fact`, so the ranking is recomputed against the
**current fact cache** every step.

### 4.3 Two kinds of question

| | | |
| --- | --- | --- |
| **"is it x?"** | **train** mode | teaches the model how things connect |
| **"what is it?"** | **use** mode | reads the trained vector knowledge out directly |

### 4.4 How memory actually gets called

The goal holds a **comparison** question and never touches the search directly. Only at the very start does the
question get **extracted into a ladder**, each rung proving one component that already has a vector — and those
rungs live in **fact**.

```
"is a mandarin an orange?"  →  prove:  round  >  orange  >  peel texture

  goal := "is it round?"          fact := conv layer [·][·][·]
  f(goal, fact) = compare(goal, read_memory(fact))
  update fact
  if right feeling → next goal := "is it orange?"
```

> **after reading —**
>

---

## What I know is thin

- **binding.** Co-occurrence gives me **bundling** (*these showed up together*) and never **binding** (*this
  property belongs to that object*). Fine on single-object images. **Fatal on scenes** — round + orange + green +
  leafy in one picture doesn't make the orange leafy. That line is also the playground staging: object-centric
  first, scenes exactly when binding becomes mandatory.
- **the right feeling itself.** Still no form for it. Margin between top-1 and top-2? Residual? Something else?
  And what grounds it without labels is still the thing most likely to eat months.
- **internally generated thought.** Images ground *perceptual* thinking — the orange-car story runs. They ground
  **nothing** for *q = mc∆t*, where the feeling comes from all my memories agreeing and no sensory input pins
  either side. First substrate covers one of my two founding examples. Say that out loud rather than discover it
  late.
- **does connection-count actually behave like generality** on real conv features? Nobody has measured it. If it
  turns out globally static in practice, the ladder is a fixed pipeline with early exit and I've fooled myself.
- **whether any of this is new.** Perceiver, RAM, Hopfield-as-attention, ACT/PonderNet, NSCL, HNSW, attribute ZSL —
  most of the parts exist. Best guess at what's left that's mine: **the two-motion loop (the goal rewrites itself
  on a match)** and **the coarse-to-fine ladder**. To be checked, not assumed.
  → [`reading-checklist.md`](reading-checklist.md) §0

> **after reading —**
>

---

## Scratch — new ideas land here

*(free space. dump things here first; sort later.)*

-
