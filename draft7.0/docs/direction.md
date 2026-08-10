# The direction — bounded-bandwidth visual reasoning

> **Adopted 2026-08-10.** `[STANDING]` `[AUTHOR]`
>
> **Supersedes [`the-thought.md`](../.archive/docs/the-thought.md) wholesale.** The previous picture (a thought as
> goal + fact register + `f(goal, fact)`, the two-motion loop, the ladder rule, the CNN-halves arena) was held at
> evidence zero as *the first decision, not the final one*, explicitly so that research would have something to
> attack. Three days of study produced the rivals. It was replaced whole and dated, exactly as the draft said it
> would be. The record is in [`../.archive/`](../.archive/README.md).
>
> **What changed underneath:** the old direction had **no references at all** — it was pure intuition, by design.
> This one starts from a **diagnosed, published failure** and a **map of what the field has and has not closed**
> ([`../notes/vault/INDEX.md`](../notes/vault/INDEX.md) §6). The mechanisms below are still unvalidated. The
> *problem* is no longer invented.

---

## 1 — The direction, in one sentence

> ### **"ต้อง optimize ยังไง ถ้า attention size เล็ก แต่โมเดลเลือกได้ว่าจะมองอะไรและจดอะไร"**
>
> *How do you optimize when the attention size is small, but the model gets to choose what to look at and what to
> write down?*

`[AUTHOR]` The name for outside conversation: **bounded-bandwidth visual reasoning**.

**This is a change of problem, not a change of solution** — the correct move against a crowded field. The immediate
consequence: every result that assumes the model sees everything becomes the **special case `attention size = N`**,
rather than a competitor.

**The underlying question both legs are walking toward:** *what is latent space, actually — and how do you build one
that does real work?* `[OPEN]`

---

## 2 — Why the direction moved

### 2.1 The diagnosed failure `[LIT]`

*Imagination Helps Visual Reasoning, But Not Yet in Latent Space* (Li et al., **ICML 2026**) settles by causal
mediation that latent visual tokens in current MLLMs are **placeholders**: `α ≈ 0`, `β ≈ 0`, latents highly
homogeneous. Explicit *text* imagination (CapImagine) beats them. → [13-latent-space-and-shortcut](../notes/vault/13-latent-space-and-shortcut.md) §5

### 2.2 Why it fails — and it is not a training problem `[CLAIM]`

The author's reading, which the note supports mechanically:

> latent มันพังบน image ตอนนี้ เพราะว่ามันไม่จำเป็นต้องคิดลึก มันก็ classify ได้เเล้ว ต่างจาก language ที่มันเป็นข้อมูลเชิงสัญลักษณ์อยู่เเล้ว

Stated as mechanism:

| | |
| --- | --- |
| **The shortcut** | `v → ans` is **1 hop, pretrained, and live at all 32 layers.** `v → z → ans` is **2 hops starting from random.** B wins, and *the optimizer is right to prefer it.* `v_i` is not one shortcut but 32 simultaneous ones — which is why regularization cannot realistically close it |
| **The law** | ***anything avoidable will be avoided*** — latents, glimpses, CoT, layers, auxiliary heads. Only three fixes exist (supervise it / cheapen A and penalize B / **delete B**), and only **deleting the alternative path** is robust |
| **Why language is different** | text is already symbolically dense: GSM8K needs ~5 sequential steps with no shortcut. Most VQA is 1-hop. **Vision has no GSM8K** |
| **The consequence** | *"บน image มันสามารถ classified ภาพเเบบ parallel ได้ ไม่ต้องใช้ depth"* — so weight drains out of the latent path almost entirely |

→ [13-latent-space-and-shortcut](../notes/vault/13-latent-space-and-shortcut.md) §4, §6, §8, §9

### 2.3 What follows — and it is the load-bearing move

**If the shortcut is the cause, then the objective is the thing to change, not the architecture.** You cannot
supervise a model into thinking when one-shot classification already pays. The task has to make the second hop
**unavoidable**, and the read has to be **bandwidth-limited** so that iteration is an *information bound* rather
than a hoped-for difficulty.

That is the whole reason this direction is about **recurrent visual attention with `attention size ≪ input`**:

> วัตถุประสงค์หลักตอนนี้คือการสร้าง latent space ให้มันมองโลกได้จริงๆ เเละมองในราคาที่ถูกลง เป็นการมองเเบบ glimpse ไม่ใช่การยัดทุกๆอย่างเข้าไปทีเดียวใน context window

A small attention that **roams over a large real input**, rather than a context window that swallows it. `[STANDING]`

---

## 3 — What is being built: one line, two halves

**One thing gets built first: the latent decides where and how coarsely to look.** `[STANDING]` The author's own
scoping, and it overrides any reading of this draft that has more than one thing running:

> s1 กับ s2 มันไป direction เดียวกันจริง เเต่ว่ากุไม่ได้จะ pick หลายๆทำในทีเดียว ตอนนี้ที่กุเล็งไว้อย่างเเรกคือ s2 **Latent predicts position AND
> scale (l, s)** ก่อน เพราะว่ามันใหญ่ที่สุด

```
                     latent ที่มองโลกได้จริง และมองในราคาที่ถูกลง
                                          │
              ┌───────────────────────────┴───────────────────────────┐
              │                                                       │
   THE HAND — how a look is taken                THE BRAIN — where to look next
   latent predicts (l, s)                        associative retrieval of an ADDRESS
              │                                                       │
   z_t → (l_t, s_t) → ρ(x, l_t, s_t) → z_{t+1}    l = P·softmax(β Mᵀq)  on a thumbnail
   read from an image pyramid, trilinear          Hopfield, but it returns coordinates
   zoom out = explore, zoom in = exploit          no walking → no wall, no RL
              │                                                       │
              └───────────────────────────┬───────────────────────────┘
                                          │
                              ONE loop. Split only because
                              they answer different questions.

           ─────────────── later, not now ───────────────
           s1 · object-bound dynamic latent  =  the multi-latent generalisation
           (each latent holds one object AND carries its own (l, s))
```

### 3.1 The hand — latent predicts position **and** scale `[STANDING]`

**What it is, in the author's words:**

> การทำให้ latent มันเกิดขึ้นจริงๆบน image ได้เลย

**The gap it walks into, which is the strongest single fact in the vault:**

| | position | **scale / resolution** |
| --- | --- | --- |
| RAM 2014 | learned | **fixed** (hand-set ring ladder) |
| STN 2015 | learned | learnable in theory — **nobody studied it** |
| **DRAW 2015** | learned | **learns σ and stride — the only one that actually did** |
| Deformable DETR 2021 | learned | **picks from a fixed menu** of pyramid levels |
| ViT / everything current | — | **fixed** patch size |

> ⚠ **Say the Deformable DETR row precisely or it gets shot down.** It *does* learn per-query sampling offsets
> **and** weights over feature levels. The defensible claim is not "nobody does scale" — it is that **the set of
> scales is fixed and discrete; the model picks from a menu rather than emitting a continuous scale.**

**Mechanism** `[CLAIM]`:

```
z_t ──▶ (l_t, s_t) ──▶ ρ(x, l_t, s_t) ──▶ z_{t+1}

l_t = tanh(W_l z_t)              log s_t = W_s z_t        ← predict LOG s, so scale is multiplicative
p_ij = l_t + s_t·(u_i, v_j)      on a FIXED g×g grid       ← STN affine locked to isotropic scale + translation
∂ρ/∂s = Σ_ij (∂ρ/∂p_ij)·(u_i,v_j)                          ← image gradient weighted by distance from centre
                                                              = "does the outer rim add information? if not, zoom in"
```

**Why this is differentiable at all when RAM's was not:** the gradient dies in **two independent layers**, and only
one of them was ever the real blocker. Layer 1 = `crop` rounding to the pixel grid, so `∂ρ/∂l = 0` almost
everywhere — **that is what actually killed RAM, and interpolation fixes it** (this is what STN does). Layer 2 =
sampling from a distribution, which is a genuine barrier only for truly discrete choices; a Gaussian was always
reparameterisable. Conflating the two produces the false conclusion *"choosing where to look is discrete, therefore
you need RL."* → [12-recurrent-visual-attention](../notes/vault/12-recurrent-visual-attention.md) §7, §11

**⚠ The implementation detail that decides whether this works at all: aliasing.** A large `s` on a fixed `g×g` grid
samples below Nyquist → garbage reads *and* garbage gradients, with nothing in the loss to tell you. **Fix: read
from an image pyramid and interpolate across levels** — `ℓ* = log₂(s_t·W/g)`, trilinear. What deformable attention
does across feature scales, and what graphics has done with mipmaps for thirty years.

**The pyramid is not an implementation detail — it is the mechanism.** With it the trade-off becomes *physically
real*: a wide `s` is wide **and genuinely blurry**, so the model **cannot cheat by setting `s = 1` forever.**
Without it, it learns to zoom all the way out and stop, and what has been built is an expensive ViT. This is §2.2's
law applied to this design's own shortcut — **the pyramid deletes path B at the level of physics rather than the
level of the loss.**

**Why `s` is not just one more parameter:** it is a **budget decision.** Zoom out = **explore**, zoom in =
**exploit**. The model has to learn *"am I searching, or am I verifying?"* — interpretable and measurable.

> Plot `s_t` against `t`. A downward slope is **coarse-to-fine that emerged instead of being imposed** — and
> coarse-to-fine is graduated non-convexity, i.e. the model finding the standard fix to the wall by itself.
> → [12-recurrent-visual-attention](../notes/vault/12-recurrent-visual-attention.md) §12.4

**The graph nobody has plotted:** accuracy vs read-fraction `Σ(s_t·W)² / HW` → ***"90% while reading 2%."***

**And the experiment that actually isolates the contribution** is not "this vs ViT" — it is four arms at an
**identical read budget**, differing in one variable: `l` learned + `s` **fixed** (= RAM) · `l` learned + `s`
**random** · `l` learned + `s` **learned** (the claim) · `l` random + `s` learned (controls for position doing the
work). If the third does not beat the first at matched budget, learned scale bought nothing.

### 3.2 The brain — associative retrieval of an address `[STANDING]`

**Same loop, different question.** §3.1 answers *how a look is taken once you know where*; this answers *how you
know where next* — without walking a gradient across the image and without RL.

```
low-res glance  ──▶  M = {m_p}     (associative memory over thumbnail positions)
latent z_t      ──▶  q_t = W_q z_t (query)
                     l_t = Σ_p softmax(β⟨q_t, m_p⟩)_p · pos(p)
```

**This is modern Hopfield retrieval exactly — but it returns an *address* instead of *content*:**

```
Hopfield:  ξ_new = X · softmax(β Xᵀ ξ)      ← pulls the content back
this:      l     = P · softmax(β Mᵀ q)      ← pulls the COORDINATE back,
                                              then reads the real thing at full resolution
```

No walking, so no wall and no local minimum. Differentiable, so no RL. `O(N_low)` on a thumbnail — an 8× downscale
is 64× cheaper. And **the attention map on the thumbnail is a readable picture of what it is reaching for.**

It also answers the failure that opened this whole line — *how do you move from the right eye to the left eye if
you don't know the left eye is there?* **The thumbnail knows it is there; it just doesn't know the detail.**
Retrieve the coordinate from the thumbnail, then read the detail at full resolution.

**Two things not to forget:** soft softmax puts the retrieved point at the **centroid between candidates** (use
high β, or top-1 with straight-through), and retrieve **K addresses, not one**, to keep parallelism — the exact
thing RAM died of. → [s2-opened-topic-ideas](../notes/vault/s2-opened-topic-ideas.md) §2,
[4-hopfield-internal](../notes/vault/4-hopfield-internal.md) §6.7,
[12-recurrent-visual-attention](../notes/vault/12-recurrent-visual-attention.md) §15

The sentence that unifies it: ***RAM is Hopfield where the agent chooses its own next probe.***

**Build order:** §3.1 first, with a plain MLP emitting `(l, s)`. Swap in §3.2 as **phase 1.5**, changing one
variable, and measure the difference. Do not build both at once.

### 3.3 Later, not now — object-bound dynamic latent (`s1`)

`[STANDING]` **Not being built in this pass.** Recorded because it is where this line goes next, and because the
merge is already written down in [s2-opened-topic-ideas](../notes/vault/s2-opened-topic-ideas.md) §5:

> **if each latent holds one object and each carries its own `(l, s)`, that is §3.1 in multi-latent form =
> deformable attention where the latent picks its own scale. These two can merge.**

The mechanism when it comes: **flip the softmax axis** from over-`N` (Perceiver — latents do not compete) to
over-`M` (Slot Attention — pixels distribute their budget, so slots compete and object binding emerges with no
supervision), plus weighted-mean normalisation and a GRU update. The decider is the **slot drop test** (drop one
slot: lose one object = binding is real; lose global quality = it is not), and it costs 2 hours. Its known killer:
Slot Attention works beautifully on CLEVR and dies on natural images — **start from DINO/DINOv2 features.**
→ [11-perciever-and-more](../notes/vault/11-perciever-and-more.md) §15.3, §15.5,
[s1-opened-topic-ideas](../notes/vault/s1-opened-topic-ideas.md)

**Why it is second and not first:** one latent that acts is enough to test whether acting fixes the placeholder
problem at all. Binding only becomes necessary once there are several latents to tell apart — and adding it now
means two unproven mechanisms failing together with no way to tell which one did it.

---

## 4 — What happened to associative memory

**It did not get dropped — §3.2 is it.** The author's correction, verbatim:

> ไม่ใช่เเค่การทำ right feeling สำหรับ assosiate memory ง่ายๆ (topic นี้มัน close ไปเเล้ว) เเต่เรากำลังไปออกเเบบว่า assosiate memory
> มันควรเอาไปใช้ยังไง บน dataset ที่ไม่ได้ข้อมูลที่หนาเเน่นเชิงสัญลักษณ์ตั้งเเต่เเรก

| | |
| --- | --- |
| **Closed** `[STRUCK]` | *"the right feeling"* as a standalone research object — a self-generated correctness signal for a plain associative-memory loop. **Do not re-open it as a topic.** |
| **Open, and now the actual question** | **how associative memory should be *used* on data that was never symbolically dense to begin with** — answered by §3.2: retrieve a coordinate, not a meaning |

---

## 5 — What would kill this

Stated up front, because a direction that cannot be killed cannot be tested.

| # | Kill condition | Where it gets checked |
| --- | --- | --- |
| 1 | **The problem isn't real** — an existing VLM handles a `k = 4` pointer chase comfortably | Phase 0. Stop if so |
| 2 | **Sequential is not better than parallel** at matched read budget — no reasoning is happening | Phase 1. **Then go fix the dataset, not the model** |
| 3 | **Coarse-to-fine never emerges** — `s_t` vs `t` is flat | may mean `T` is too small or the task doesn't need it; diagnose before concluding |
| 4 | **`s` collapses** to 1 (zoom out forever) or to `s_min` | if it collapses to 1 *with* the pyramid in place, the physical-blur argument in §3.1 is wrong |
| 5 | **Learned `s` does not beat fixed `s`** at a matched read budget | Phase 1's four-arm ablation (§3.1). This is the one that kills the *contribution* rather than the framing |
| 6 | **The shortcut returns silently** — any of the four checklist violations in [13-latent-space-and-shortcut](../notes/vault/13-latent-space-and-shortcut.md) §9.4. *"The architecture does not save you from a bad benchmark"* | every experiment, every time |

**And the concession to make before anyone asks:** CapImagine (explicit text) already beats latent methods. The
answer is not that text is worse — it is that **a coordinate is the smallest verifiable, grounded symbolic
interface**: 2 numbers against ~200 tokens. The claim is cost and interpretability, not raw accuracy.

---

## 6 — The plan

From [s2-opened-topic-ideas](../notes/vault/s2-opened-topic-ideas.md) §7. Framed to **optimize knowledge per unit time**, not certainty of outcome — every
phase produces something that survives the model failing.

### Phase 0 (1–2 weeks) — prove the problem is real
- Build the **pointer-chase generator**: 2000×2000 canvas, 40×40 signs with heavy decoys, sign A reads "→ B",
  B reads "→ C", the last one holds the answer, the question gives only the start. **A linked list rendered as an
  image.** Chain length `k` is a dial.
- Fire it at existing Qwen-VL / LLaVA. **No training at all.**
- Measure **accuracy vs `k`**, plus a **blind baseline** (delete the image).
- **And measure the shortcut directly** — hook the attention weights at the answer position and read off how much mass
  lands on visual tokens vs on reasoning tokens. ~20 lines of code, and it means citing a measurement of our own
  rather than only citing the paper.
- **Kill criterion:** SOTA handles `k = 4` easily → the problem isn't real → stop.
- *This phase alone is enough to open with an advisor.* If it falls at `k = 2`, that graph is the whole pitch.

### Phase 1 (~1 month) — the smallest model that could work
- glance 64×64 → **a plain MLP** emitting `(l_t, s_t)` (§3.1) → trilinear pyramid read → loop.
- **A single latent vector. `T` glimpses. No LLM, no pretraining.** No binding, no slots — one latent is enough to
  test whether *acting* fixes the placeholder problem.
- Baselines: full-image ViT · random glimpse · **parallel-`k` ← the one that matters**.
- Then the **four-arm ablation** at matched read budget (§3.1) — the only experiment that isolates learned `s`.
- **Kill criterion:** sequential ≯ parallel → **go fix the dataset, not the model.**

### Phase 1.5 — swap in the Hopfield retina
Replace the MLP that emits `l_t` with associative address retrieval (§3.2), changing **one variable**, and measure
the difference. Expected wins: no gradient walk across the image, and a thumbnail attention map that is directly
readable as *"what is it reaching for."*

### Phase 2 — only if Phase 1 survives
Attach to a real VLM · adaptive halting · V\*Bench, HR-Bench. **Do not start here** — nothing is debuggable at this
end of the pipeline.

### The danger the author named himself, boxed
> ### ⚠ doing all the ideas at once
**One thing: §3.1.** Then §3.2 as a single-variable swap. §3.3 (object-bound latent) waits, and the part-whole
decomposition idea stays **demoted to an analysis section, never a method section** — Hinton spent ten years on it (Capsules → GLOM) and nobody has succeeded; with no supervision and
no metric it becomes a placeholder by §2.2's law. **The benchmark comes first**, because it is the insurance:
model fails → the benchmark still exists for other people; model works → the benchmark is what makes the result mean
anything.

---

## 7 — What carried over, and what did not

**Carried forward from the archived picture** — because the vault independently arrived at the same place, not
because they were grandfathered:

| The old instinct | What it became |
| --- | --- |
| **The read-bandwidth limit** (*"an information bound, not a hoped-for difficulty"*) | Survives intact and is now the **core of the direction**. [12-recurrent-visual-attention](../notes/vault/12-recurrent-visual-attention.md) §3 is the same argument, built. [13-latent-space-and-shortcut](../notes/vault/13-latent-space-and-shortcut.md) §9 supplies the missing half: **the bound is worthless without a deep-hard task** |
| **The ladder rule / coarse-to-fine** | No longer a thesis; now a **measurable prediction** — `s_t` vs `t` sloping down (§3.1) |
| **The backbone must be self-supervised** | Reached again from the other side: the biggest risk in §3.3 is binding dying on natural images, and the fix is **DINO/DINOv2** |
| **Binding was named as thin** | Correct — and it is §3.3, deliberately **second**, because one acting latent tests the core claim without it |
| **Never make the representation learner also be the classifier** (carried from draft 6) | Untouched. Still the division of labour |

**Not carried:** the two-motion loop as the thesis, `f(goal, fact)` as the object under construction, the
CNN-front-half/back-half arena as specified, and the axis A–H survey. They are in
[`../.archive/`](../.archive/README.md). They are not forbidden — they are **not active**, and returning any of
them means bringing it back into this file with a date and a reason.

**Explicitly closed:** *"the right feeling"* as a topic (§4).

---

## 8 — Evidence status

**Still zero for every mechanism here.** Nothing below has been run in this repo.

What is *different* from the archived direction, and it is the only thing that is different:

| | archived direction | this one |
| --- | --- | --- |
| The problem | asserted from intuition | **diagnosed, published, causal-mediation evidence** |
| The field | unknown | **mapped** — [INDEX](../notes/vault/INDEX.md) §6, ~19 topics closed, 11 open and ranked |
| The mechanisms | invented | **read** — Slot Attention, STN, DRAW, deformable, Hopfield retrieval, all with known failure modes |
| Kill criteria | none | **six, and two are phase gates** |
| Evidence | zero | **zero** |

The last row is the one to keep saying out loud.

---

### Tags

The evidence tags used throughout this file (`[STANDING]` `[AUTHOR]` `[CLAIM]` `[LIT]` `[OPEN]` `[ARG]` `[CARRIED]`
`[STRUCK]`) are defined once, in [`../CLAUDE.md`](../CLAUDE.md) — the file that auto-loads whenever anyone works in
this draft. They are not repeated here so the two cannot drift apart.

**Untagged assertions are a defect.**
