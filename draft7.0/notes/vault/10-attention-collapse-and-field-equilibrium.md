# Collapse, Equilibrium, and What Survives a Block

> Notes reconstructed from the second half of the conversation that produced `6-fixed-points-and-what-transformers-chase.md`. It started at _"what is a context window, really?"_ and ended at _"you cannot freeze what hasn't finished negotiating."_
>
> Where note 6 asked **what the mechanism is** and **why it can't stop**, this one asks the physical question underneath: **what actually happens to a token as it passes through a block — what gets added, what gets destroyed, and what is left at the end.**
>
> Direct sequel to `6-fixed-points-and-what-transformers-chase.md`. It also **issues one correction to note 6** — see §4.3.

---

## How to read this

| §   | The question                                  | The wrong picture it kills                           |
| --- | --------------------------------------------- | ---------------------------------------------------- |
| 1   | Where does $N$ go?                            | "variable set size is still a problem"               |
| 2   | Do the projections resize anything?           | "$W_Q,W_K,W_V$ expand the dimension"                 |
| 3   | What is a context window?                     | "attention compresses long input down to the window" |
| 4   | Which fixed point is the bad one?             | "attention must avoid $\nabla E = 0$"                |
| 5   | What is rank collapse, exactly?               | "the tokens stop moving"                             |
| 6   | What is the hull doing across depth?          | "one hull slowly shrinking"                          |
| 7   | Does the residual stream fade out?            | "the signal is used up by the last layers"           |
| 8   | Where is information actually destroyed?      | "attention averages information away"                |
| 9   | Is attention retrieval or not?                | "it's a metaphor, nothing more"                      |
| 10  | Can we run to equilibrium and then emit?      | "equilibrium means done thinking"                    |
| 11  | Can we stop when the block stops writing?     | — this one is **right**, with limits                 |
| 12  | Where does that work best?                    | "early exit is for LLMs"                             |
| 13  | Can every layer write in the head's language? | "we'd have to redesign it to do that"                |
| 14  | Train full then compress?                     | "cut the unused parts off"                           |
| 15  | Grow one layer at a time?                     | "freeze what's already trained"                      |
| 16  | Won't the new layer steal the old one's job?  | "job theft is the risk"                              |

**Short on time: §4, §6, §8, §11, §16.**

---

# Part I — What a block does with $N$

## §1. $N$ is contracted, not stored

### 1.1 Walk the shapes

$$\underbrace{\mathbf{X}^\top\boldsymbol\xi}_{N\times1} ;\longrightarrow; \underbrace{\operatorname{softmax}}_{N\times1} ;\longrightarrow; \underbrace{\mathbf{X}a}_{d\times1}$$

$N$ appears in the middle and **is absent from the output**. It is a _contracted index_ — the same role as the inner dimension of a matrix product, summed away.

Feed 300 or 300,000; the next layer sees the same $d$ every time.

### 1.2 Shape invariance is not enough

Easy to miss: a raw _sum_ also has the right shape, but its **magnitude grows with $N$**. A 300k bag would produce a far longer vector than a 3k bag, and the downstream layer — trained on some particular scale — breaks.

$$\sum_i a_i = 1$$

This is the load-bearing part. The output is a **weighted average**, hence inside the convex hull of $\mathbf{X}$, hence at the same scale as a single instance regardless of count.

> **Shape invariance comes from contraction. Statistical invariance comes from normalisation.** Two different properties, both required.

`mean()` also has both — and is still structurally dead, because it hands every element $a_i = 1/N$ and can never select. Softmax gives both properties **plus** selectivity.

### 1.3 Whether $N$ survives depends on the variant

|                   | in          | out         | does $N$ survive?                                              |
| ----------------- | ----------- | ----------- | -------------------------------------------------------------- |
| `HopfieldPooling` | $N\times d$ | $1\times d$ | **no — that is the entire job**                                |
| self-attention    | $N\times d$ | $N\times d$ | **yes — one query per token**                                  |
| `HopfieldLayer`   | $1\times d$ | $1\times d$ | never present; $\mathbf{X}_K$ is a fixed $M\times d$ parameter |

The middle row is why Transformers never face the variable-length problem: **every layer already accepts any $N$.** Nothing needs a fixed shape until the very end, where you take the last token or pool once.

> **It does not solve $N$. It defers $N$ to the last line.**

### 1.4 Batching, in practice

You cannot build a ragged tensor, so you **pad to the longest and mask**. Masking sets padded logits to $-\infty$ before softmax:

$$e^{-\infty} = 0 \quad\Longrightarrow\quad a_i = 0 \text{ exactly, and the denominator never counts it}$$

Padding costs compute and **affects correctness not at all** — mathematically the pad slots do not exist. (For DeepRC-scale bags where $N$ varies by hundreds of thousands, padding is ruinous; real code batches one bag at a time or buckets by size.)

### 1.5 What is still a genuine problem

Shape is solved. These two are not, and they are unrelated to each other:

- **Time/memory.** Pooling is $O(Nd)$ — cheap. Self-attention is $O(N^2)$ — the origin of FlashAttention, sliding windows, and the whole efficient-attention literature.
- **Statistical dilution.** $\varepsilon \sim Ne^{-\beta\Delta_i}$. Larger $N$ means a larger softmax denominator and less sharpness. Normally you compensate with $\beta \sim \log N$, but in a Transformer $\beta = 1/\sqrt{d_k}$ is **pinned**.

> **The formula accepts any $N$. That does not mean it works equally well at every $N$.**

---

## §2. The projections do not resize anything

### 2.1 They shrink, if anything

$W_Q, W_K, W_V$ are $d\times d_k$, and in practice $d_k = d/h$. GPT-style $d = 768$, $h = 12$ gives $d_k = 64$ **per head**. Nothing is being expanded.

More importantly, all three are **per-token**:

$$\underbrace{\mathbf{X}}_{N\times d}\cdot\underbrace{W_K}_{d\times d_k} = \underbrace{\mathbf{K}}_{N\times d_k}$$

$N$ **passes straight through untouched.** $N$ rows in, $N$ rows out. The projections transform each row independently and handle nothing about set size.

### 2.2 What they actually buy

| Matrix        | Role                                                                                                                                            |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| $W_K$ / $W_V$ | separate _"what makes me findable"_ from _"what I hand over when found"_ — pure Hopfield cannot express this, since $\mathbf{X}$ does both jobs |
| $W_Q$         | lets a token ask something other than _"who resembles me?"_                                                                                     |
| $W_O$         | merges the $h$ heads back from $h\times d_k$ to $d$ — the only one doing dimensional bookkeeping, and it is about **heads**, not $N$            |

### 2.3 Where the contraction really happens

$$\underbrace{\mathbf{Q}\mathbf{K}^\top}_{N\times N} \longrightarrow \underbrace{\operatorname{softmax}}_{\text{row-wise}} \longrightarrow \underbrace{A\mathbf{V}}_{N\times d_k}$$

In $A\mathbf{V}$ the key axis is summed away **exactly as in pooling** — but $N$ times in parallel, once per row.

> **Self-attention is $N$ copies of `HopfieldPooling` run in parallel over one shared store, each asking its own question.** $N$ genuinely vanishes _within each row_; stacking the rows brings it back.

---

## §3. The context window is not a compression mechanism

### 3.1 The equation has no length limit

Shapes again: any $N$ works, output is always $d$. Whatever the limit is, **it does not come from attention.**

### 3.2 The three real sources

| Source                  | Nature               |
| ----------------------- | -------------------- |
| **positional encoding** | a genuine hard limit |
| **$O(N^2)$ cost**       | engineering          |
| **softmax dilution**    | statistical          |

**Positional encoding is the real one.** $\mathbf{X}$ is a _set_; order is injected by the position embedding. With learned absolute positions that is a fixed-size table — **position 4097 has no vector at all.** RoPE/ALiBi are functions and extrapolate in principle, but were trained on a range; past it you are out of distribution.

**Dilution** is §1.5 again: larger $N$, larger denominator, less sharpness, with $\beta$ pinned.

### 3.3 Nothing shrinks back

There is no compression event. Feed $N$ and you get an $N\times N$ matrix — full stop.

> **A context window is not a mechanism. It is just the largest $N$ the system accepts** — a number set by training range and memory, not a transformation.

And what's lost outside the window isn't down-weighted — it is **absent from $\mathbf{X}$ entirely**. Not a shallow well. No well.

> **The context window is the boundary of the convex hull.** Anything outside was never in the answer space to begin with.

### 3.4 Where wells _do_ accumulate one at a time

During autoregressive generation, past tokens live in the **KV cache**. Each new token is projected to $k,v$ and appended.

**This is where the landscape genuinely grows well by well** — and the context window is the maximum number of wells the cache holds. The "adding one at a time" intuition is correct; it just lives in the outer loop, not in Hopfield's storage formula.

### 3.5 How systems exceed the window anyway

Not the model — the machinery wrapped around it:

| Method                            | What it does                                                                                                                                                                                                                 |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **truncation / sliding window**   | drop old tokens. Information genuinely lost; works because recent context usually suffices for next-token prediction                                                                                                         |
| **StreamingLLM**                  | keep the first few tokens **plus** a sliding window. Works because of the **attention sink** — many heads dump weight on token 0. Those are the **no-op heads** from note 6 §8.5; remove their dumping ground and they break |
| **Position Interpolation / YaRN** | rescale RoPE frequencies so the trained range covers more positions. Fixes the root cause in §3.2                                                                                                                            |
| **RAG**                           | pull relevant material back into the window. **This is genuine retrieval** — an actual persistent archive, which attention does not have                                                                                     |
| **summarisation / compaction**    | lossy compression into fewer tokens                                                                                                                                                                                          |

### 3.6 And it degrades _inside_ the window too

**Lost in the middle**: material in the middle of the window is used measurably worse than material at either end — while still fully present. Exactly the dilution symptom §3.2 predicts.

> **Working past the window doesn't mean attention tolerates length. It means someone decided what to throw away on its behalf.**

---

# Part II — Where collapse actually is

## §4. Two levels of fixed point

### 4.1 Head level — happens every time, entirely normal

$$\boldsymbol\xi_t^* = \mathbf{V}a(\boldsymbol\xi_t^*)$$

Theorem (B): reached in one step. Each token settles into **its own** well, and those are different wells. **This is attention working correctly.**

### 4.2 Stack level — this is the disaster

$$\mathbf{z}_i = \mathbf{z}_j \quad \text{for all } i,j$$

Not "one token went too deep into one well." **All the wells melted into one, and every token fell into it.**

|                     | head level                                | stack level                     |
| ------------------- | ----------------------------------------- | ------------------------------- |
| axis                | one forward pass of a head                | depth                           |
| what converges      | $\boldsymbol\xi_t \to$ its own well floor | the shared hull $\to$ a point   |
| governed by an $E$? | **yes**                                   | **no**                          |
| status              | normal, must happen                       | catastrophic, must be prevented |

The third row matters: contraction along depth is **not energy descent**. There is no $E$ on that axis. It is a geometric side effect of repeated convex averaging. The two phenomena are **not one thing at two scales** — the first has a theorem, the second is an accident.

### 4.3 Correction to note 6

Note 6 §3.6 and §9 can be read as saying a Transformer must avoid reaching $\nabla E = 0$. **That reading is wrong.**

> **Attention reaches its head-level fixed point every single time, in one step. What must be prevented is every token reaching the _same_ one across depth.**

The thing that decides which happens is $\beta$ and separation — never step count. Which means:

> **Rank collapse is metastable merging, propagated across depth until nothing is distinguishable.**

---

## §5. Rank collapse, precisely

### 5.1 The name is literal

$$\begin{bmatrix}\mathbf{z}_1\\mathbf{z}_2\\vdots\\mathbf{z}_N\end{bmatrix} \longrightarrow \begin{bmatrix}\bar{\mathbf{z}}\\bar{\mathbf{z}}\\vdots\\bar{\mathbf{z}}\end{bmatrix} \qquad \text{rank} = 1$$

### 5.2 Why it tends to happen

Attention is **convex averaging**. Every average shrinks the differences between tokens a little. Stack that and it shrinks a lot.

Dong et al. (2021) proved that with attention alone, token differences contract **doubly exponentially in depth** — hence the title, _"Attention is not all you need."_

### 5.3 Why it is fatal

If all tokens are equal, the model cannot tell which word is where; the whole sequence has one representation; **nothing can be predicted.**

And nothing downstream can undo it:

$$\mathbf{z}_i = \mathbf{z}_j ;\Longrightarrow; \text{FFN}(\mathbf{z}_i) = \text{FFN}(\mathbf{z}_j)$$

FFN and LayerNorm are functions applied identically per token. **They amplify differences that exist; they cannot create them.**

### 5.4 What prevents it

- **residual** — $x$ re-injected every layer, carrying un-averaged identity
- **FFN** — non-linear and not an average, pushes tokens apart _while they are still apart_
- **high-$\beta$ heads** — select instead of averaging

Dong et al. show these are what halt the contraction. $\beta$ is not the cause of collapse; it is one of the things that modulates it.

### 5.5 The wording that matters

$$\text{not } \mathbf{z}_i \text{ stops moving} \qquad\text{but}\qquad |\mathbf{z}_i - \mathbf{z}_j| \to 0$$

During collapse, tokens still move, layers still compute, values still change. **What is lost is difference, not motion.** Not a flat graph — every curve exactly on top of every other.

---

## §6. The hull across depth

### 6.1 One shared hull, not one per token

Each token does not have its own hull. There is **one** per layer: the convex hull of ${\mathbf{v}_1,\ldots,\mathbf{v}_N}$. Every token's output lands inside it, so the output set is confined to a shell **no wider than** the input set's. It can never widen.

### 6.2 But it is a _chain_ of hulls, not one shrinking hull

The next layer builds a **new** $\mathbf{V}$ from the previous layer's output — so it builds a **new hull**, seeded by the previous one. The landscape is reborn every layer.

> **Not one hull slowly closing. A sequence of hulls, each drawn around what the last one produced.**

Left alone, the sequence contracts to a point — rank 1. Which is why something must push outward at every step.

### 6.3 Residual escapes the hull immediately

$$x + \operatorname{Attn}(x)$$

$\operatorname{Attn}(x)$ is inside the hull; adding $x$ puts the result **outside it instantly**, because $x$ still carries un-averaged identity.

The contraction is cancelled every layer instead of accumulating. This is the same device DEQ needs as **input injection** (note 6 §9.6) — a Transformer just performs it once per layer instead of every iteration.

> **The hull contracts. The residual re-expands. Usable depth is the balance of the two — not a property of attention alone.**

---

# Part III — What survives

## §7. The residual stream is an accumulator

### 7.1 It grows, it does not fade

$$\mathbf{z}^{(\ell)} = \mathbf{z}^{(0)} + \sum_{k\le\ell}\big(\text{what each block wrote}\big)$$

Every block **reads from the stream and writes back into it**. Its norm typically **increases** with depth. The picture of a signal being gradually used up is backwards.

### 7.2 Late-layer quiet is a choice, not exhaustion

What does taper off in late layers is the _rate of change_ — because many heads write almost nothing. Those are the **no-op heads** (note 6 §8.5) dumping weight on BOS.

> **It is not running out of energy. It is choosing not to write.**

### 7.3 The last position is an aggregator

Interpretability work is consistent here: attention **hauls information from other positions and piles it at the position that needs it**. Other positions stay distinct throughout; they do not dissolve into each other.

**So "blend everything until nothing is left" is exactly backwards.** What you want at the final layer is for the read-out token to be the **most specific, most information-dense** vector in the sequence — not the most averaged.

> **The right word is _concentration_, not _dissolution_.** Full blending is rank collapse, which predicts nothing at all.

### 7.4 Why late layers look settled

Because weights were trained to finish at layer $L$ — not because dynamics carried them there. **Logit lens** and **early exit** measure this: easy tokens stabilise by the middle layers; hard tokens keep changing to the last one.

If it were a true equilibrium, it would settle at the same depth for every input. It does not.

---

## §8. Where information is added, and where it is destroyed

### 8.1 The per-token ledger

Within one block, for one token:

|           | effect on that token                                                                             |
| --------- | ------------------------------------------------------------------------------------------------ |
| attention | **adds** — imports a content-selected summary of other positions                                 |
| residual  | **preserves** — the old value is a literal term in the sum                                       |
| FFN       | **transforms** per token — non-linear, can sharpen or restructure, has no access to other tokens |

Nothing about the token itself is removed, because of the residual. **A block is information-adding for the token it acts on.**

### 8.2 So where does destruction actually happen?

Five places, and none of them is "attention averaging things away in the middle":

1. **The window boundary.** Tokens outside never entered $\mathbf{X}$. Total, irreversible, and the largest loss by far.
2. **The $a_i \approx 0$ tail, _within one head's summary_.** Low-weight sources are dropped from _that_ summary — but the sources themselves are untouched in their own streams, and other heads/layers can still reach them. **Locally lossy, globally recoverable.**
3. **Rank collapse** — if it happens. Genuine irreversible destruction (§5.3).
4. **The exit.** Taking the last token, or pooling to $1\times d$. **This is where $N$ is really destroyed** — deliberately, because a classifier needs a fixed shape.
5. **Compression** — pruning, quantization, distillation. Deliberate, at a different stage.

> **Attention adds information per token. The destruction is at the boundary and at the exit — not in the middle.**

### 8.3 And in `HopfieldPooling`, destruction is the point

|                   | $N$           | per-token information    | is the loss desired?                  |
| ----------------- | ------------- | ------------------------ | ------------------------------------- |
| self-attention    | preserved     | increased                | loss is a side effect to be minimised |
| `HopfieldPooling` | **destroyed** | collapsed to one summary | **yes — it is the entire job**        |

Same operation, opposite verdict. This is the §1.5 bug/feature inversion from note 6, restated on the information axis: merged wells are a defect for recall and the deliverable for MIL.

### 8.4 One correction worth keeping

"Pooling squeezes data down into the well to separate it from noise" is close but mechanically off. **Noise is not removed** — it receives $a_i \approx 0$. And the encoder learns to make signal distinct only because **softmax routes gradient to whatever was selected**. It is **selection**, not compression.

---

## §9. Attention _is_ retrieval — in one sense, not the other

### 9.1 Separate the two senses

| Sense                                                                   | Does attention qualify?                                                                               |
| ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **IR sense** — search a pre-existing archive                            | **No.** There is no archive. $\mathbf{X}$ is this input's own tokens                                  |
| **content-addressable sense** — offer content, get similar content back | **Yes, fully.** Query and key share a space, compared by inner product, more similar gets more weight |

> **Attention is retrieval whose archive is the present.** Token 50 asks tokens 1–49 "who is relevant to me." That is genuine retrieval over a store that was born moments ago and dies at the end of the pass.

**Induction heads are the proof**: seeing $[A][B]\ldots[A]$ they point straight at $B$. That is precise retrieval, not averaging.

So note 6's "attention is not retrieval" should be read as: **retrieval is not its purpose.** The mechanism is retrieval; the goal is better features for the layer above.

### 9.2 The two axes that classify everything

|                        | query comes from | wells come from |
| ---------------------- | ---------------- | --------------- |
| `HopfieldPooling`      | **parameter**    | input           |
| `Hopfield` / attention | input            | input           |
| `HopfieldLayer`        | input            | **parameter**   |

`HopfieldLayer` is the mirror image of `HopfieldPooling`:

- **pooling** — a permanent question asked of a store that changes every time
- **layer** — a changing question asked of a permanent store ← the only one that is genuinely _memory_
- **attention** — **nothing is permanent.** Question and store both die with the forward pass

> **The 2020 paper did not propose a model. It proposed one equation and pointed out that you choose which side is permanent. Attention is the choice where nothing is.**

---

# Part IV — Making depth adaptive

## §10. Why you cannot run the field to equilibrium

### 10.1 The proposal

_"Attention contracts, FFN expands. Iterate until the two balance, then emit logits."_

A real dynamical system, and much better than "just raise $\beta$." It fails for four independent reasons.

### 10.2 (a) The equilibrium is input-independent

Balancing contraction against expansion under a rule applied identically at every step converges to **a static distribution determined by the weights, not by the input.**

Same symptom as collapse — it settles into a **shape** instead of a **point**, but input-specific information is gone either way.

> **A good representation is one that still remembers what it came from. That requires it not to be at equilibrium.**

### 10.3 (b) FFN has no memory, so iterating accumulates nothing

$$\text{FFN}(\mathbf{z}) = W_2,\sigma(W_1\mathbf{z})$$

Feed the same $\mathbf{z}$ ten times, get the same answer ten times. The $4d$ hidden layer expands and **collapses back to $d$ on the same line** — nothing persists in the wider space. No notepad, no carried state.

> **Iteration only helps if something accumulates.** Hopfield accumulates because $\boldsymbol\xi$ descends $E$ step by step. FFN has no such quantity — iterating is just more function composition, not more thinking.

### 10.4 (c) With different weights per layer it is not iteration at all

Note 6 §5.5. $f_L\circ\cdots\circ f_1$ has no fixed point to converge to, because there is no single $f$.

### 10.5 (d) Nothing at inference can certify equilibrium meaningfully

Any internal test — e.g. $|\mathbf{z}^{(\ell)} - \mathbf{z}^{(\ell-1)}|$ — measures **movement, not correctness**. The same wall as note 6 §6.4: the quantity that says "the answer is right" needs a label, and labels do not exist at inference.

### 10.6 What actually buys depth

Not waiting for stillness — **a loop with accumulated state**:

- **chain-of-thought** — write state out as tokens, read it back
- **latent reasoning** (COCONUT, looped transformers) — feed the previous round's latent vector back in as input

Structurally identical; they differ only in whether the carried state is a token or a vector.

> **What buys depth is a loop that carries state, not a system allowed to settle.**

---

## §11. The version that does work: stop when the block stops writing

### 11.1 The refinement that fixes everything

Separate two things that are easy to conflate:

- **residual stream** — the accumulator, holds the answer, does not collapse
- **what a block writes** — the output of $W_Q,W_K,W_V$ and FFN, which **can go to zero without harming the stream at all**

That yields a criterion that is measurable, label-free, and **not** rank collapse:

$$\text{exit when } |\Delta^{(\ell)}| = |\text{Attn}^{(\ell)} + \text{FFN}^{(\ell)}| < \tau$$

**This is a real design and it is used**: early exit, CALM, layer skipping, Mixture-of-Depths. Typical savings **20–50% of compute**. It matches observed no-op heads and logit-lens stabilisation.

### 11.2 What it buys and what it doesn't

**Buys:** compute, when the model doesn't need the depth it has.

**Does not buy:** any confidence the answer is right. It answers _"is the model still changing its mind?"_ not _"is the model correct?"_ — **a confidently wrong model goes quiet at layer 3 and exits confidently wrong.**

> **This is a savings criterion, not a correctness criterion.** Still a proxy — just a cheap, practical one.

And it does nothing about _$L$ too small_: you can leave early, never extend.

### 11.3 Two engineering obstacles

**KV cache.** If a token exits at layer 5 but a later token must attend to it at layer 9, its $k,v$ for layers 6–9 do not exist. You either recompute or approximate with layer 5's. This is the main difficulty of early exit in autoregressive models.

**It must be trained for.** Note 6 §10.4: layer 8 was never trained to be last. You need early-exit heads or layer dropout during training — you cannot just truncate a finished model.

### 11.4 MoD dodges both

Instead of exiting, a **per-layer router decides which tokens enter each block**; unselected tokens flow through the residual untouched.

Tokens stay in the stack for all layers → **cache intact**; per-layer compute budget stays fixed → **allocation varies across tokens, total does not**.

The router must **predict rather than measure** — measuring $|\Delta|$ after computing it saves nothing.

---

## §12. Single-pass tasks are where this is cleanest

### 12.1 The obstacle disappears entirely

No generation ⇒ no later token needs this token's $k,v$ at a deeper layer. **The KV-cache problem does not exist**, not merely eases.

This is where early exit originated: **DeeBERT, PABEE, FastBERT** — all single-pass encoder classification. Decoder-side came later and is harder.

**PABEE's criterion is the more robust one**: exit when the prediction is _unchanged for $p$ consecutive layers_, rather than when confidence crosses a threshold. Tolerant of miscalibration, and closer to the "nothing more is being written" intuition anyway.

### 12.2 In pooling models, save on $N$ instead of $L$

`HopfieldPooling` has **one block** — no stack to exit from. But it has a far better axis to save on.

From $\sum_i a_i = 1$ and $\varepsilon \sim Ne^{-\beta\Delta_i}$: most instances get $a_i \approx 0$. In a 300,000-instance bag, perhaps dozens carry real weight.

So the same criterion becomes _"which instances to compute"_:

1. cheap or truncated encoder pass → approximate $a$
2. keep top-$k$
3. run the full encoder only on those

Standard hierarchical filtering in MIL, and it saves far more than layer skipping — **$N$ is thousands of times larger than $L$.**

### 12.3 Caveats everyone hits

- **Training cost.** Attaching a head at every layer and summing losses pressures lower layers to answer early instead of building features. **Peak accuracy usually drops slightly.**
- **Calibration.** Raw entropy or margin is overconfident; $\tau$ needs a validation set. "40% savings" is an average, not a per-input guarantee.
- **Wall-clock ≠ FLOPs.** In a batch where inputs exit at different depths, the GPU waits for the slowest. Real savings need per-sample execution or difficulty bucketing — the same issue as padding in §1.4.

---

## §13. Reading the stream without disturbing it

### 13.1 The architecture already forces a shared language

The residual stream is **one shared basis**. Every block reads and writes the same space with no coordinate change in between, so the frame is consistent across depth **by construction**.

Which means **logit lens** works: apply the final unembedding to an intermediate state,

$$\text{logits}^{(\ell)} = W_U,\text{LN}\big(\mathbf{z}^{(\ell)}\big)$$

and predictions are legible — forming gradually with depth, often stable before the last layer.

> **"Make every layer write in a language the head can read" is not a modification. It is already true.**

### 13.2 Why a lens still needs help

Legible but not _good_ at lower depths — the frame is shared, but the features are not yet composed into what $W_U$ was trained to interpret.

**Tuned lens** fixes exactly this: a small learned affine per layer, then the same $W_U$.

$$\text{logits}^{(\ell)} = W_U,\text{LN}\big(A_\ell\mathbf{z}^{(\ell)} + b_\ell\big)$$

### 13.3 Check the price before optimising it

A classification head is $\mathbb{R}^d\to\mathbb{R}^C$ — for $768\times2$, about **1,500 parameters**, against roughly **7 million** per Transformer block.

> **Saving on the head is ~0.02%. Skipping blocks is ~40%.** The cost is in the blocks, never the read-out.

The one exception: an **LM head** is $d\times|V|$ ($768\times50{,}000 \approx$ 38M). There, sharing one $W_U$ across all depths instead of $L$ copies genuinely matters.

### 13.4 The real cost is training interference

Joint early-exit heads pull lower layers toward premature answering (§12.3). **Tuned lens avoids this completely** — trained post-hoc on a frozen model, with no backward pressure at all. The model never knows it is being read.

| Method                               | Cost                 | Effect on the model                           |
| ------------------------------------ | -------------------- | --------------------------------------------- |
| logit lens                           | free                 | none — but poor at low depth                  |
| **tuned lens**                       | one affine per layer | **none — trained post-hoc on frozen weights** |
| full head per layer, trained jointly | already cheap        | **peak accuracy drops**                       |

The middle row wins **not** because it saves head cost, but because **it does not disturb what the lower layers learned** — note 6 §10.4 again.

---

# Part V — Growing depth

## §14. Train full, compress after

### 14.1 This is what the field does

Almost nobody trains early exit from scratch. **Two-stage**: pretrain at full depth, compress afterwards — covering distillation, pruning, quantization, and early exit alike.

The reasoning is right: lower-layer features are only good because they came from **undisturbed** training. Apply answer-now pressure from step one and you never get them, so there is nothing to compress later.

### 14.2 But compression is renegotiation, not removal

Weights are co-adapted to $L$. Lower layers learned what is useful to upper layers _knowing how many remain_. Cut the depth and that contract changes — a layer that used to hand off half-finished work must now hand off something more finished.

> **Compression is not removing unused parts. It is renegotiating the whole stack.** Which is why most compression fine-tunes everything, not only what was touched.

### 14.3 Gradual works, and that part of the intuition is right

- **iterative magnitude pruning** — cut 10–20%, fine-tune to recover, repeat. Reaches far higher sparsity than one big cut
- **progressive layer dropping** — raise the skip probability over training
- **layer dropout** — randomly skip layers so every depth becomes a working depth (directly addresses "layer 8 was never trained to be last")

They work because each step is small enough that fine-tuning can renegotiate in time.

### 14.4 What breaks first

- **Not the bottom.** Interpretability consistently finds first and last layers matter more; **middle layers are the redundant ones**, and modern layer-pruning removes from the upper-middle.
- **Rare capabilities go first.** Aggregate benchmarks barely move while multi-step reasoning and long-context handling degrade much faster. Consistent with note 6 §10.3: **depth is the only source of sequential steps, so step-hungry tasks pay first.**
- **None of it raises the ceiling.** Everything here makes the model cheaper, nothing makes it think deeper.

### 14.5 Pruning vs distillation

If the goal is a small strong model, **distillation usually beats pruning** — a small model designs its own depth contract from scratch and learns from soft targets, instead of patching a contract negotiated for a different $L$.

> **Pruning amends the old contract. Distillation writes a new one with the old model as teacher.** Both start from a fully trained model — that part of the intuition is correct.

---

## §15. Growing one layer at a time, and why freezing kills it

### 15.1 This already exists

Train layer 1, freeze it, add layer 2, repeat — that is **greedy layer-wise pretraining** (Hinton, Bengio, ~2006–2007). It is what revived deep learning at the time. The field abandoned it around 2012–2015.

### 15.2 Why it was needed, and why it stopped being needed

Before ReLU, batch norm, residual connections, and Adam, **gradient genuinely could not reach the bottom of a deep stack** — sigmoids saturated and the signal vanished. Training layers one at a time kept each problem shallow enough to be trainable.

Fixing the root cause removed the reason to do it. **The residual stream is what killed it.**

### 15.3 The deeper problem: it contradicts §14.2

Training layer 1 alone, it does not know 11 more are coming. It learns to **answer as well as possible as a final layer** — decision-ready features, not good raw material for layers above.

Freeze that, and the remaining 11 must build on a representation optimised for the wrong purpose and now unfixable.

> **Freezing locks in a contract negotiated for $L=1$ inside a stack whose real depth is $L=12$.**

Same failure as joint early-exit heads (§12.3), but far worse: frozen means no recovery, not just mild pressure.

### 15.4 A layer alone has no loss

Practical wrinkle: one layer cannot be trained without a temporary head, so **the objective layer 1 is trained on is not the objective it will serve.** The 2006 versions used RBM or autoencoder objectives — entirely different from the end task, requiring full fine-tuning afterwards anyway. Which erases most of the benefit of freezing.

### 15.5 What survives — grow, but never freeze

| Method                                | Difference                                                                                       |
| ------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **progressive stacking** (BERT)       | train 6 layers → **duplicate the whole stack** to 12 → keep training **everything**. ~25% faster |
| **LiGO / net2net**                    | grow small → large with a learned map, then train the whole thing                                |
| **stagewise with near-identity init** | new layers start as no-ops, all layers keep adapting                                             |

All three share one property: **old layers are never frozen.** They renegotiate every time the depth changes.

> **Growing depth buys a good initialisation. It does not buy a finished lower half.**

### 15.6 The gate goes on last

The final part of the proposal is the correct part. Add exit gates **after** the full model exists — same reasoning as tuned lens: **read afterwards, without disturbing what was learned.**

> Working version: **grow depth while always training all layers → full model → then attach exit gates.** One word removed: _freeze_.

---

## §16. "Won't layer 2 steal layer 1's job?"

### 16.1 Split the question — it is two questions

**Job allocation** and **information loss** are different concerns with different answers.

### 16.2 Relocation is not a failure

The division of labour between layers is not specified in advance — **the optimiser negotiates it.** There is no reason a job must stay where it started.

If moving work to layer 2 lowers the loss, that _is_ a better allocation. Protecting layer 1's territory imposes a constraint with nothing behind it — and forces a 12-layer system to use a 1-layer system's division of labour.

> **The thing to fear is not layer 2 taking the job. It is nobody doing the job.** Different failure, different measurement.

### 16.3 Information loss is already handled by architecture

$$\mathbf{z}^{(\ell)} = \mathbf{z}^{(\ell-1)} + \text{block}(\mathbf{z}^{(\ell-1)})$$

The old value is **a literal term in the equation**. Overwriting requires actively adding something that cancels it — harder than leaving it alone. The optimiser's cheapest path is $\text{block}\approx 0$, i.e. a no-op.

Compare a non-residual stack, which must **learn** the identity map just to preserve anything.

> **In a residual architecture, not losing information is the default, not something you build.**

### 16.4 Three optional reinforcements, in order of strength

1. **Near-identity init** — set the block's output projection near zero so a newly added layer starts as a no-op and finds its role gradually (ReZero, LayerScale, Fixup). Nearly free; worth doing always.
2. **Discriminative learning rates** — low LR for old layers, high for new. Continuous damping instead of a binary freeze; old layers still adapt, just slower.
3. **Self-distillation** — keep the $\ell$-layer checkpoint as teacher and penalise the $(\ell{+}1)$-layer model for drifting from its outputs. An elastic tether with a tunable weight.

**None of them freeze anything.** That is the common property.

### 16.5 Honest verdict

BERT's progressive stacking uses **none** of these and works fine. The genuine risk in growing depth is not information loss but **instability at the moment of expansion** — a loss spike when the new layer arrives. Option 1 addresses that most directly and is the cheapest of the three.

---

# Appendix A — Traps from this stretch

| Wrong picture                                                      | Correction                                                                                          |
| ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| variable set size is still a problem downstream                    | $N$ is a contracted index — shape invariance from contraction, scale invariance from $\sum a_i = 1$ |
| $W_Q,W_K,W_V$ expand the dimension to absorb $N$                   | they are $d\times d/h$ — they shrink, act per token, and $N$ passes straight through                |
| self-attention solves the variable-$N$ problem                     | it **defers** it; every layer accepts any $N$, and the exit handles it                              |
| padding corrupts the result                                        | $-\infty$ logits give $a_i = 0$ exactly; padding is mathematically absent                           |
| a context window compresses long input                             | nothing shrinks back; it is just the largest accepted $N$                                           |
| out-of-window tokens get low weight                                | they are **absent from $\mathbf{X}$** — no well, not a shallow one                                  |
| working past the window means attention tolerates length           | external machinery decides what to discard                                                          |
| attention must avoid $\nabla E = 0$                                | it reaches it every time in one step. **Correction to note 6 §3.6/§9**                              |
| rank collapse = going too deep into one well                       | = all wells merging and every token landing in the one that's left                                  |
| in rank collapse the tokens stop moving                            | they keep moving; $\|\mathbf{z}_i-\mathbf{z}_j\|\to0$ is what happens                               |
| depth-wise contraction is energy descent                           | there is no $E$ on the depth axis — it is a geometric side effect                                   |
| one hull slowly shrinking across depth                             | a **chain** of hulls, each built from the last layer's output                                       |
| the residual stream fades out by the end                           | it is an accumulator; its norm typically **grows**                                                  |
| late layers are quiet because the signal is exhausted              | no-op heads choosing not to write                                                                   |
| ready for logits = everything blended together                     | backwards — the read-out token should be the **most** specific                                      |
| attention averages information away                                | per token it **adds**; destruction is at the window and at the exit                                 |
| pooling squeezes data to separate it from noise                    | noise isn't removed, it gets $a_i\approx0$ — it is **selection**, not compression                   |
| attention is not retrieval                                         | mechanism is retrieval; **purpose** is not. Archive is the present                                  |
| run to equilibrium then emit                                       | equilibrium is input-independent — same loss as collapse, in a shape instead of a point             |
| FFN "thinks" if you iterate it                                     | memoryless; the $4d$ expansion collapses back on the same line. Nothing accumulates                 |
| exiting when the block stops writing tells you the answer is right | it tells you the model stopped changing its mind. Savings criterion, not correctness                |
| every layer would need redesigning to be readable                  | the shared residual basis already makes it readable                                                 |
| early-exit heads save meaningful compute                           | head ≈ 1.5K params vs block ≈ 7M. The cost is the blocks                                            |
| compression = removing unused parts                                | renegotiating the contract of the whole stack                                                       |
| prune from the bottom / the end                                    | middle layers are the redundant ones                                                                |
| freeze layer 1, then grow                                          | it was trained to be _final_; freezing locks a contract made for $L=1$                              |
| layer 2 stealing layer 1's job is the risk                         | relocation is fine if loss drops; the risk is **nobody** doing it                                   |
| you must build a mechanism to preserve information                 | the residual makes preservation the default path                                                    |

---

# Appendix B — Quick reference

**The chain in one pass**

1. $N$ is contracted away; $\sum a_i = 1$ keeps the scale fixed
2. Projections are per-token and shrink; they buy **role separation**, not size handling
3. A context window is the largest accepted $N$ — the hull boundary, not a compressor
4. Head-level fixed point: every time, normal. Stack-level: rank collapse
5. Collapse destroys **difference**, not motion — and FFN cannot rebuild it
6. Depth is a chain of hulls contracting; the residual re-expands each time
7. The residual stream accumulates and grows; late quiet is no-op heads
8. Per token attention **adds**; destruction lives at the window and the exit
9. Attention is content-addressable retrieval whose archive is the present
10. Equilibrium is input-independent — the wrong target
11. "Stop when the block stops writing" is valid — for compute, not correctness
12. Single-pass tasks make it clean; in pooling, save on $N$, not $L$
13. The shared residual basis already makes every depth readable — tuned lens reads without disturbing
14. Train full, compress gradually; compression renegotiates the whole stack
15. Grow depth, but never freeze; gates go on last
16. The residual already protects information — reinforcements are optional

**Formulas to keep in the hand**

$$\underbrace{\mathbf{X}^\top\boldsymbol\xi}_{N\times1}\to\underbrace{\operatorname{softmax}}_{N\times1}\to\underbrace{\mathbf{X}a}_{d\times1} \qquad \mathbf{z}^{(\ell)} = \mathbf{z}^{(0)} + \sum_{k\le\ell}\text{writes}$$

$$|\Delta^{(\ell)}| = |\text{Attn}^{(\ell)} + \text{FFN}^{(\ell)}| < \tau \qquad \text{logits}^{(\ell)} = W_U,\text{LN}(A_\ell\mathbf{z}^{(\ell)} + b_\ell)$$

**Mantras**

> Shape invariance from contraction. Scale invariance from normalisation.

> It does not solve $N$. It defers $N$ to the last line.

> A context window is not a mechanism. It is the largest $N$ the system accepts.

> Out of window is not down-weighted. It is absent.

> Attention reaches its fixed point every time. What must be prevented is every token reaching the same one.

> Collapse destroys difference, not motion.

> The hull contracts; the residual re-expands. Usable depth is the balance.

> It is not running out of energy. It is choosing not to write.

> Concentration, not dissolution.

> Attention adds per token. Destruction is at the boundary and at the exit.

> Retrieval whose archive is the present.

> A good representation still remembers where it came from — so it must not be at equilibrium.

> Iteration only helps if something accumulates.

> A savings criterion, not a correctness criterion.

> Compression is renegotiation, not removal.

> Growing depth buys an initialisation, not a finished lower half.

> The risk is not that the job moves. The risk is that nobody does it.

---

## Open threads

- **Dong et al. 2021** — the doubly-exponential contraction proof, and precisely which of residual / FFN / high-$\beta$ heads does how much of the work (§5.4)
- **Attention sinks** — why token 0 becomes the dumping ground, and what StreamingLLM's dependence on it says about no-op heads (§3.5)
- **Logit lens vs tuned lens** — what the learned affine is actually correcting for, layer by layer (§13.2)
- **MoD routers** — how a router learns to predict which tokens need a block, given it cannot measure $|\Delta|$ first (§11.4)
- **COCONUT / looped transformers** — latent-space reasoning as the state-carrying loop of §10.6
- **Layer-pruning studies** — the empirical claim that middle layers are the redundant ones, and what that implies about the depth contract (§14.4)
- **Emergent-capability degradation under compression** — the sharpest available evidence that depth supplies sequential steps
- Still outstanding from notes 4 and 6: **Jacobian / spectral conditions** at the fixed point

---

## Companion files

- `1-transformer.md` — the architecture end to end
- `2-hopfield.md` — organised around the 2020 paper
- `3-transformer-internal.md` — what a Transformer really is
- `4-hopfield-internal.md` — the four variants, mechanism by mechanism
- `5-interp-hull-and-superposition.md` — convex hull and superposition
- `6-fixed-points-and-what-transformers-chase.md` — where the two frames meet, and where the analogy breaks
- **this file** — what a block does to a token, and what is left at the end
