# Hopfield Internals: Building the Picture From Zero

> Notes reconstructed from a long argument that started at _"I can't see it"_ and ended at _"a Transformer block is two lookups stacked."_
>
> Companion to `2-hopfield.md`. That file is organised around the paper. **This one is organised around the order in which the confusion actually clears.**
>
> Primary reference: Ramsauer et al., **"Hopfield Networks is All You Need"** (2020)

---

## How to read this

Every section answers one question that the previous section makes unavoidable. Read in order the first time.

| §   | Question                                                                           |
| --- | ---------------------------------------------------------------------------------- |
| 1   | What _kind of thing_ is a Hopfield network?                                        |
| 2   | What is a "landscape," concretely?                                                 |
| 3   | Hopfield 1982 — what is it really, and how does it work with no dial?              |
| 4   | Why did capacity have to be fixed, and how?                                        |
| 5   | Where does softmax come from? (Nobody chose it)                                    |
| 6   | Why is this literally attention?                                                   |
| 6.5 | **Why does Hopfield chase slope = 0, and why must a Transformer never get there?** |
| 6.6 | The dead end: "so just raise $\beta$" — two wrong turns and what they teach        |
| 6.7 | **N queries, one landscape — where "sequence" is, and where it isn't**             |
| 7   | The four variants — one axis explains all of them                                  |
| 8   | `HopfieldPooling` — why pooling is forced, and what it routes                      |
| 9   | `HopfieldLayer` — a learned lookup table                                           |
| 10  | How is `HopfieldLayer` different from attention?                                   |
| 11  | softmax vs ReLU — the real mechanical difference                                   |
| 12  | Why does `HopfieldLayer` use softmax at all?                                       |
| 13  | GELU / SwiGLU                                                                      |
| 14  | Honest assessment                                                                  |
| A   | Traps — wrong pictures that felt right                                             |
| B   | Quick reference                                                                    |

**If you're re-reading and short on time: §2, §6.5, §6.7, §7, §11.**

---

# Part I — Fixing the category

## §1. It is a loop, not a pipeline

Before any math, fix what kind of object this is. This is where the confusion starts and most of it never recovers.

A **Transformer is a pipeline.** Data enters one end, flows block 1 → block 2 → … → exits. It has direction, depth, and weights inside the pipe. To make it smarter you adjust things inside the pipe.

A **Hopfield network is not a pipeline.** It has no layers, no depth, no forward pass. It has exactly two things:

1. **One vector** (the state)
2. **One rule**, applied to that vector repeatedly until it stops moving

Formally: a **dynamical system**, not an architecture. So what are you doing when you use one?

> **Shaping a landscape so it has valleys where you want them, then dropping a marble in.**

### The problem it solves

|        | Address-addressable (RAM) | Content-addressable (associative memory)  |
| ------ | ------------------------- | ----------------------------------------- |
| Input  | an address                | a _partial or corrupted_ piece of data    |
| Output | the data at that address  | the _complete, clean_ nearest stored item |

Store 100 faces. Feed in a face with half of it occluded. Get the complete matching face back.

The mechanism: build an energy function $E$ whose local minima sit exactly at the stored patterns, then let the query roll downhill.

$$\text{query } \boldsymbol\xi ;\longrightarrow; \text{descend } E ;\longrightarrow; \text{stored pattern}$$

**Every version in this document is a different design of $E$. That is the whole subject.**

---

## §2. What a "landscape" actually is

This is the sentence that unlocks everything else:

$$E(\boldsymbol\xi) = -\operatorname{lse}(\beta,\mathbf{X}^\top\boldsymbol\xi) + \tfrac12|\boldsymbol\xi|^2$$

Look at what appears in it. $\mathbf{X}$, $\boldsymbol\xi$, $\beta$. **Nothing else exists.** There is no Hopfield _object_ anywhere. The landscape is a _function of three things_, not a structure that persists.

$$\text{You never write to the landscape — you change what it is computed from}$$

### How $\mathbf{X}$ produces it — mechanically

$x_i^\top\boldsymbol\xi$ measures "how much does $\boldsymbol\xi$ point the same way as $x_i$." It's maximal when they align. So $-x_i^\top\boldsymbol\xi$ **is one well, with its floor at $x_i$.**

$$\text{1 pattern} ;=; \text{1 term} ;=; \text{1 well}$$

`lse` (log-sum-exp) then combines those wells **softly**. Since $\operatorname{lse}(\beta,\mathbf{z}) \approx \max_i z_i$ for large $\beta$, the total energy is approximately the **soft minimum** of the individual wells. Each pattern digs its own well, unaware the others exist; the wells just add.

**No intermediate step. No optimisation. No selection.** Plug numbers into a formula and the valleys are already there.

### Concrete: $d=4$, two orthogonal patterns

$\mathbf{x}^1 = (1,1,1,1)$, $\mathbf{x}^2 = (1,-1,1,-1)$, classical energy $E = -\frac{1}{2d}\sum_\mu \langle\mathbf{x}^\mu,\boldsymbol\xi\rangle^2$:

| $\boldsymbol\xi$ | $\langle\mathbf{x}^1,\boldsymbol\xi\rangle$ | $\langle\mathbf{x}^2,\boldsymbol\xi\rangle$ | $E$    |             |
| ---------------- | ------------------------------------------- | ------------------------------------------- | ------ | ----------- |
| $(1,1,1,1)$      | 4                                           | 0                                           | **−2** | ← well      |
| $(1,-1,1,-1)$    | 0                                           | 4                                           | **−2** | ← well      |
| $(1,1,1,-1)$     | 2                                           | 2                                           | −1     | one bit off |
| $(1,1,-1,-1)$    | 0                                           | 0                                           | 0      | ← ridge     |

Nothing was done except substitution.

### The lifetime of a landscape

In attention, $\mathbf{K}$ and $\mathbf{V}$ are **not persistent memory** — they're computed on the fly from the current input.

> **Every new input builds a brand-new energy landscape. The wells sit on that input's tokens. When the input ends, the landscape is gone.**

$$\textbf{A landscape lives exactly one forward pass.}$$

What gets trained is $\mathbf{W}_Q,\mathbf{W}_K,\mathbf{W}_V$ — not the wells, but **the well-digging procedure**.

---

# Part II — Hopfield 1982, properly

## §3. What it really is

### 3.1 Definition

- State: $\mathbf{x} \in {-1,+1}^d$ — $d$ binary neurons
- Weights: $\mathbf{W} \in \mathbb{R}^{d\times d}$, **symmetric** ($w_{ij}=w_{ji}$) with **zero diagonal** ($w_{ii}=0$)
- Bias: $\mathbf{b}\in\mathbb{R}^d$

$$E(\mathbf{x}) = -\tfrac{1}{2}\mathbf{x}^\top\mathbf{W}\mathbf{x} + \mathbf{b}^\top\mathbf{x} \qquad\qquad x_i \leftarrow \operatorname{sgn}\Big(\sum_j w_{ij}x_j - b_i\Big)$$

**There is no $\beta$ here.** No temperature, no sharpness dial. $\beta$ arrives in 2017 with $F=\exp$, because `lse` needs an inverse temperature to be defined at all. (Stochastic 1980s variants do have a temperature — different lineage, not this one.)

### 3.2 Storage is not training

$$\mathbf{W} = \frac{1}{d}\sum_{\mu=1}^N \mathbf{x}^\mu(\mathbf{x}^\mu)^\top \qquad(\text{then zero the diagonal})$$

Add the patterns together. Done. No loss, no gradient, no epochs. It is a million times faster than training a network **because it is not training.**

> If you've been hunting for "the learning" inside classical Hopfield, you've been hunting for something that isn't there. That confusion is the correct response to the material.

**Critical structural point:** $\mathbf{W}$ is $d\times d$. All $N$ patterns are **superimposed into one matrix and cannot be pulled apart again.** Contrast the modern version, which keeps $\mathbf{X} \in \mathbb{R}^{d\times N}$ with every pattern in its own column. Half of the capacity explosion later comes from simply _not cramming everything into one matrix._

### 3.3 How retrieval works with no dial

Content-addressing does **not** come from sharpness. It comes from the inner-product structure of the energy. The moment $E$ is built from $\langle x^\mu,\xi\rangle$, "more similar ⇒ lower energy" exists for free.

The actual retrieval step is a **majority vote**:

$$h_i = \sum_j w_{ij}x_j = \underbrace{x_i^\nu}_{\text{signal}} + \underbrace{\frac{1}{d}\sum_{\mu\neq\nu}x_i^\mu\langle\mathbf{x}^\mu,\mathbf{x}^\nu\rangle}_{\text{crosstalk } C_i^\nu} \qquad x_i \leftarrow \operatorname{sgn}(h_i)$$

Neuron $i$ asks: _"given everyone else's current state, what do the stored patterns collectively want me to be?"_ — and takes the sign. Repeat until nothing flips.

### 3.4 Why it terminates

**Step 1 — isolate the terms containing $x_i$.** By symmetry the row-$i$ and column-$i$ halves are equal; the $l=i$ term vanishes by the zero diagonal:

$$E = -x_i\underbrace{\Big(\sum_{l\neq i}w_{il}x_l - b_i\Big)}_{h_i} + ;C, \qquad C \text{ independent of } x_i$$

(This is exactly why both constraints exist. Without them the step fails and the guarantee is gone.)

**Step 2 — the update can only help.** $x_i^{\text{new}} = \operatorname{sgn}(h_i)$ gives $x_i^{\text{new}}h_i = |h_i|$, the maximum of $x_i h_i$ over ${-1,+1}$, so

$$\Delta E = -(x_i^{\text{new}} - x_i^{\text{old}}),h_i \le 0$$

**Step 3 — convergence.** $2^d$ states ⇒ finitely many energy values ⇒ a non-increasing sequence must halt. Finitely many steps. $\blacksquare$

> ⚠️ **Requires asynchronous updates.** Update all neurons at once and you can land in a period-2 limit cycle. (This restriction disappears in the continuous version, because that proof comes from the convex/concave split rather than from freezing coordinates.)

### 3.5 The nonlinearity is in the wrong place

It's wrong to say 1982 has no sharp nonlinearity. $\operatorname{sgn}$ is a **hard threshold — sharper than softmax at any finite $\beta$.** The difference is _where it acts_:

|            | the sharp decision operates on          | consequence                                                      |
| ---------- | --------------------------------------- | ---------------------------------------------------------------- |
| **1982**   | one **neuron** at a time ($d$ of them)  | forces each bit to ±1, but cannot prevent pattern-level blending |
| **Modern** | one **pattern** at a time ($N$ of them) | can choose which pattern, or how much to blend                   |

This is why $\operatorname{sgn}$ can't save you when capacity breaks: **crosstalk is summed into $h_i$ before $\operatorname{sgn}$ ever sees it.** Once crosstalk beats signal, $\operatorname{sgn}$ confidently returns the wrong bit. It is sharp but blind.

### 3.6 Capacity — the fatal weakness

- **Orthogonal patterns** → crosstalk $=0$ → perfect recall
- **Random patterns** → $C_i^\nu$ has mean 0, variance $\approx N/d$

$$P_{\text{error}} = \Phi\left(-\sqrt{d/N}\right) \qquad\Longrightarrow\qquad \boxed{N_{\max}\approx 0.14,d}$$

1000 neurons → ~138 patterns. (Demanding _zero_ errors across all patterns tightens this to $N\approx d/(2\ln d)$.) Past that you enter a spin-glass phase producing **spurious states**: fake wells that are blends of real patterns.

**Root cause: the energy is quadratic.** Quadratic wells are wide and shallow, so they overlap.

### 3.7 One-sentence summary

> **Hopfield 1982 is a soft-max whose softness cannot be adjusted, because it is baked into the exponent 2** — and which superimposes every pattern into a single $d\times d$ matrix.

Everything after this is prying those two limitations loose.

---

# Part III — Fixing capacity

## §4. Dense associative memory (2016–2017)

Krotov & Hopfield's move: **generalise the energy.**

$$E(\mathbf{x}) = -\sum_{\mu=1}^N F\big(\langle\mathbf{x}^\mu,\mathbf{x}\rangle\big)$$

$F$ is the **interaction / separation function**, and it turns the fixed exponent into a dial.

| $F(z)$    | Capacity     | Source                 |
| --------- | ------------ | ---------------------- |
| $z^2$     | $O(d)$       | = classical Hopfield   |
| $z^n$     | $O(d^{n-1})$ | Krotov & Hopfield 2016 |
| $\exp(z)$ | $O(2^{d/2})$ | Demircigil et al. 2017 |

### Why sharper $F$ buys capacity

The matching pattern gives overlap $\approx d$; the others give $\approx\sqrt{d}$. The gap already exists — the question is how much $F$ amplifies it:

| $F$   | signal : noise after $F$ | margin           |
| ----- | ------------------------ | ---------------- |
| $z^2$ | $d^2 : d$                | factor $d$       |
| $z^n$ | $d^n : d^{n/2}$          | factor $d^{n/2}$ |
| $e^z$ | $e^d : e^{\sqrt d}$      | **exponential**  |

Squaring only makes the big one _somewhat_ bigger — not enough to drown $N-1$ competing terms. With $e^z$ **the largest term swallows every other term entirely**, crosstalk stops mattering, and capacity jumps a complexity class.

**The price:** narrower wells → smaller basins of attraction → you need a closer cue to retrieve. This trade-off is fundamental and never goes away.

**Remaining limitation:** still binary. Useless for deep learning.

---

# Part IV — Where softmax comes from

## §5. Nobody chose softmax

### 5.1 The continuous energy

$$\boxed{;E(\boldsymbol\xi) = -\operatorname{lse}(\beta,\mathbf{X}^\top\boldsymbol\xi) + \tfrac12\boldsymbol\xi^\top\boldsymbol\xi + \beta^{-1}\log N + \tfrac12 M^2;}, \qquad M = \max_i|\mathbf{x}_i|$$

| Term                                                       | Role                                                                                                    |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| $-\operatorname{lse}(\beta,\mathbf{X}^\top\boldsymbol\xi)$ | continuous form of Demircigil's exponential — pulls $\boldsymbol\xi$ toward patterns                    |
| $\tfrac12\|\boldsymbol\xi\|^2$                             | **confinement** — newly necessary, because the hypercube ${-1,1}^d$ no longer boxes $\boldsymbol\xi$ in |
| constants                                                  | chosen so $E\ge0$                                                                                       |

_Proof that $E\ge0$:_ $\operatorname{lse}(\beta,\mathbf{z}) \le \max_i z_i + \beta^{-1}\log N \le M|\boldsymbol\xi| + \beta^{-1}\log N$, hence $E \ge \tfrac12(|\boldsymbol\xi|-M)^2 \ge 0$. $\blacksquare$

### 5.2 The derivation (CCCP)

**Split into concave + convex:**

$$E = \underbrace{-\operatorname{lse}(\beta,\mathbf{X}^\top\boldsymbol\xi)}_{E_1:\ \text{concave}} ;+; \underbrace{\tfrac12\boldsymbol\xi^\top\boldsymbol\xi + \text{const}}_{E_2:\ \text{convex}}$$

_Sub-proof that `lse` is convex._ With $p = \operatorname{softmax}(\beta\mathbf{z})$:

$$\frac{\partial\operatorname{lse}}{\partial z_i} = p_i, \qquad \nabla^2\operatorname{lse} = \beta\big(\operatorname{diag}(p) - pp^\top\big), \qquad \mathbf{v}^\top(\operatorname{diag}(p)-pp^\top)\mathbf{v} = \operatorname{Var}_p[v] \ge 0$$

> **Note $\partial\operatorname{lse}/\partial z_i = p_i$ — the softmax appears right here, before anything else has happened.** It was hiding inside log-sum-exp all along.

**Apply CCCP** (linearise the concave part, keep the convex part): solve $\nabla E_2(\boldsymbol\xi^{t+1}) = -\nabla E_1(\boldsymbol\xi^t)$ with $\nabla E_2(\boldsymbol\xi)=\boldsymbol\xi$ and $\nabla E_1(\boldsymbol\xi) = -\mathbf{X}\operatorname{softmax}(\beta\mathbf{X}^\top\boldsymbol\xi)$:

$$\boxed{;\boldsymbol\xi^{t+1} = \mathbf{X}\operatorname{softmax}\big(\beta\mathbf{X}^\top\boldsymbol\xi^t\big);}$$

**It descends:** adding the concavity bound on $E_1$ and the convexity bound on $E_2$ gives $E(\boldsymbol\xi^{t+1}) - E(\boldsymbol\xi^t) \le [\nabla E_1(\boldsymbol\xi^t) + \nabla E_2(\boldsymbol\xi^{t+1})]^\top(\boldsymbol\xi^{t+1}-\boldsymbol\xi^t) = 0$ by the CCCP condition. Monotone + bounded below + iterates confined to the convex hull ⇒ global convergence. **No asynchronous requirement.**

### 5.3 The causal chain, stated once

$$\text{want capacity} \to F=\exp \to \text{continuous form needs lse} \to \frac{\partial\operatorname{lse}}{\partial z_i}=\operatorname{softmax} \to \text{update rule}$$

The only real decision is at step two. **Softmax is a by-product of a derivative.** Saying "they chose softmax" is wrong at the level of grammar.

### 5.4 Three theorems

**(A) Exponential storage capacity.** For random patterns on a sphere of radius $M$: $C \gtrsim \sqrt{p},c^{(d-1)/4}$, $c>1$.

**(B) One-step retrieval.** With **separation** $\Delta_i = \min_{j\neq i}(\mathbf{x}_i^\top\mathbf{x}_i - \mathbf{x}_i^\top\mathbf{x}_j)$, a single update achieves $|\boldsymbol\xi^{\text{new}} - \mathbf{x}_i| \le 2\varepsilon M$ with $\varepsilon \sim Ne^{-\beta\Delta_i}$. **In practice retrieval completes in one step** — which is why a Transformer applies the rule once and moves on.

**(C) Three regimes.** $\beta$ selects the regime:

| Regime                 | Occurs when                          | $\boldsymbol\xi^*$ becomes  | Used for              |
| ---------------------- | ------------------------------------ | --------------------------- | --------------------- |
| **Global fixed point** | small $\beta$, patterns similar      | average of **all** patterns | full-set pooling      |
| **Metastable state**   | moderate $\beta$, patterns clustered | average of a **subgroup**   | averaging, clustering |
| **Single-pattern**     | large $\beta$, high separation       | **one** pattern             | associative recall    |

⚠️ $\beta$ does **not** blend multiple landscapes. There is only ever one landscape. $\beta$ blends **the wells inside it.**

---

## §6. It is literally attention

$$\boldsymbol\Xi^{\text{new}} = \operatorname{softmax}\big(\beta,\boldsymbol\Xi\mathbf{X}^\top\big)\mathbf{X} \qquad\Longleftrightarrow\qquad \operatorname{softmax}\left(\tfrac{1}{\sqrt{d_k}}\mathbf{Q}\mathbf{K}^\top\right)\mathbf{V}$$

| Hopfield                                | Transformer    |
| --------------------------------------- | -------------- |
| state patterns $\boldsymbol\Xi$         | $\mathbf{Q}$   |
| stored patterns $\mathbf{X}$            | $\mathbf{K}$   |
| $\mathbf{X}$ multiplied back at the end | $\mathbf{V}$   |
| $\beta$                                 | $1/\sqrt{d_k}$ |

$$\boxed{\text{Transformer attention} ;=; \text{one step of Hopfield retrieval} ;+; \text{projection matrices}}$$

Three details that matter and are usually skipped:

**Only one step.** Not a roll to the bottom of the well. Theorem (B) says one step suffices when separation is large enough.

**$\mathbf{K}$ and $\mathbf{V}$ are different matrices.** In pure Hopfield, $\mathbf{X}$ appears in both roles. In a Transformer, $W_K$ and $W_V$ are separate — that's the "+ projection matrices". It lets the model separate _"what makes me findable"_ from _"what I hand back when found."_

**$\beta$ is not trained in a Transformer.** It's pinned at $1/\sqrt{d_k}$. But the model still controls sharpness indirectly, **by scaling the norms of $Q$ and $K$** — longer vectors ⇒ more spread-out scores ⇒ peakier softmax. That's how BERT heads land in different regimes while sharing the same $1/\sqrt{d_k}$.

**Empirical payoff:** the paper finds BERT's heads spread across all three regimes — lower layers skew global/metastable (averaging), upper layers skew single-pattern (specific recall). The theory becomes a diagnostic for _what each head is doing._

### What "getting smarter" actually means

Gradient descent doesn't make the landscape clever. It increases **separation**:

$$\Delta_i = \min_{j\neq i}\big(\mathbf{x}_i^\top\mathbf{x}_i - \mathbf{x}_i^\top\mathbf{x}_j\big)$$

$W_K$ is pushed to place things that should be distinguishable far apart, and to leave things that should be averaged together close. The rolling-downhill mechanism never changes. Only the terrain gets rearranged so that it works.

---

## §6.5 The fixed point: goal on one side, failure mode on the other

**The single most important asymmetry in this document.** Skipping it makes everything above look like the two systems want the same thing. They want opposite things.

### What slope = 0 actually means

$$\nabla E(\boldsymbol\xi) = \boldsymbol\xi - \mathbf{X}\operatorname{softmax}(\beta\mathbf{X}^\top\boldsymbol\xi) ;=; \mathbf{0} \qquad\Longleftrightarrow\qquad \boldsymbol\xi^* = \sum_i a_i(\boldsymbol\xi^*),\mathbf{x}_i$$

Read the right-hand side out loud: **"the question I asked equals the answer I got back."** That's not a statement about flatness. It's a **self-consistency condition** — the point where the system stops changing its mind. Query with the answer, get the answer.

So Hopfield is not chasing _flatness_. Flatness is the **signature**; self-consistency is the thing. And it wants that because its job is retrieval: it needs one definite item out, and every guarantee it has (halts, lands on a stored pattern, error $\sim Ne^{-\beta\Delta_i}$) lives at the fixed point.

### A Transformer is not "deliberately stopping short"

Two corrections, and the second is the structural one.

**One step is often already convergence.** Theorem (B): with enough separation, a single update lands essentially _at_ the fixed point. So when attention output is a blend, that is **not** because descent was halted early. At low $\beta$ **the minimum itself sits at a mixture** — that's the global / metastable regime of theorem (C). How much blending happens is set by $\beta$ and separation, **not by step count.**

**Each layer is a different landscape.** Layer 2 builds its landscape from layer 1's output. These are not successive steps toward one minimum — they're independent one-shot retrievals from unrelated terrain.

> **There is no energy function for the stack.** Only per-head, per-layer ones. "Descent" is not defined across depth, and layer 5 is not mathematically "closer to the answer" than layer 2. What makes depth useful is the **training loss**, not downhill motion at inference.

### But collapse is real, and $\beta$ is not what causes it

Pure attention _does_ fail: Dong et al. (2021) show rank collapse **doubly exponential in depth** — every token converging to the same vector.

⚠️ **The direction is the opposite of the intuitive guess.** Collapse comes from $\beta$ being **too low**, not too high:

|                  | fixed point is                | effect on the token matrix                                       |
| ---------------- | ----------------------------- | ---------------------------------------------------------------- |
| **low $\beta$**  | the average of _all_ patterns | every token → the same vector — **this is rank collapse**        |
| **high $\beta$** | one pattern per query         | attention → near-permutation matrix — **full rank, no collapse** |

Attention output is a convex combination of the other tokens. Iterating weighted averages is **consensus dynamics**: everyone keeps talking, everyone ends up agreeing. Flatter ⇒ collapses faster. Sharper ⇒ resists.

### What actually prevents it

Not early stopping — the surrounding parts that are usually mistaken for plumbing:

| Guard                                     | Mechanism                                                                                               |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **residual** $x + \operatorname{Attn}(x)$ | the blended value is **added**, not substituted — a token's identity never disappears. **The big one.** |
| **FFN**                                   | operates per token, mixes nothing across positions → re-differentiates them                             |
| **multi-head**                            | several landscapes at once, so tokens don't share one basin                                             |
| **LayerNorm**                             | keeps scale from letting anything swallow anything                                                      |

Clean division of labour: **$\beta$ controls how much blending happens per step; the architecture controls whether blending accumulates into collapse.** Two different levels.

### The line to remember

$$\text{Hopfield: the fixed point is } \textbf{the answer.} \qquad\qquad \text{Transformer: the fixed point is } \textbf{the failure mode.}$$

Hopfield wants the bottom of the valley, because the bottom _is_ the retrieved item. A Transformer must **never** reach a shared bottom, because that means every token is identical.

Identical mechanism, identical equation, opposite goals. This is probably the strongest reason to be careful with "Transformers operate on associative-memory principles" — they borrowed the move, not the destination.

---

## §6.6 The dead end: "so just raise $\beta$"

Recorded because it's the natural next thought, it's wrong twice over, and _why_ it's wrong is the real lesson of this whole document.

### Attempt 1 — "push $\beta$ up so layers get closer to slope = 0"

Four walls, and the fourth is the interesting one.

**1. It's already tunable, and the model chooses not to.** $\beta$ is pinned at $1/\sqrt{d_k}$, but effective sharpness is learnable through the norms of $Q$ and $K$ (§6). BERT's answer: many heads sit in global / metastable on purpose. **Softness is a chosen, useful behaviour** — lower layers genuinely need to aggregate — not a defect awaiting a fix.

**2. $1/\sqrt{d_k}$ exists to prevent exactly this.** Large $d_k$ ⇒ large dot products ⇒ softmax pushed into its saturated zone where gradients vanish:

$$\nabla\operatorname{softmax} = \beta\big(\operatorname{diag}(a) - aa^\top\big) ;\xrightarrow{;a \to \text{one-hot};}; \mathbf{0}$$

Raising $\beta$ un-fixes the thing that scaling was invented to fix. **Sharper ⇒ harder to train.**

**3. Entropy collapse.** Over-peaked attention early in training destabilises it — loss spikes, a studied failure. The modern fix is **QK-norm**, which normalises $Q$ and $K$ specifically to _stop effective $\beta$ from drifting upward._ Note the direction: the field installs a **brake**, not an accelerator.

**4. Narrow basins.** The §4 trade-off never goes away. High $\beta$ ⇒ narrow wells ⇒ needs a closer cue ⇒ a small input perturbation retrieves something different. Brittle.

**And the deepest error:** _"getting closer to slope = 0"_ **is not a valuable objective for a Transformer at all.** The fixed point isn't an achievement the layer is failing to reach — it's meaningless here. A layer isn't finishing a retrieval problem; it's producing features the next layer can use. **Convergence has no intrinsic value.** Optimising toward it is optimising toward something that doesn't exist.

### Attempt 2 — "then use high $\beta$ only in the last layers"

Better — and **directionally, this is already true.** BERT: lower layers global/metastable, upper layers single-pattern, all sharing one $1/\sqrt{d_k}$. The model **learns that schedule by itself.** The reasoning behind the guess is sound: you have to gather broad context before there's anything specific to retrieve.

Which is precisely the problem with imposing it. If the model already builds that schedule, hard-coding high $\beta$ in late layers doesn't add capability — it **removes freedom** and replaces a learned solution with a coarser guess. Coarser in two ways:

- Sharpness is set **per head**, not per layer. The same layer holds both averaging and retrieving heads; a layer-wide override crushes the ones that should stay soft.
- Some soft upper-layer heads are soft **for real reasons** — positional heads, and no-op heads that effectively pass the residual through. Both are documented and useful.

**If you still want to try it, do it as a prior rather than a constraint:**

- make $\beta$ a per-head parameter (`hflayers` already allows this), initialised higher in upper layers — a **starting point** gradient can overrule
- anneal: start soft everywhere to avoid entropy collapse, let sharpness grow
- measure before and after with attention entropy $H(\mathbf{a})$ — otherwise you can't tell which regime anything is in

Realistic expectation: the technique that actually works in this area is **QK-norm**, which does the opposite of the proposal.

### The meta-lesson

Both attempts fail for the same reason, and it's the sharpest version of §14's _"explains ≠ equals"_:

> **The Hopfield frame tells you what the mechanism is doing. It does not tell you what the system is for.**

It gives you real vocabulary — $\beta$ as a dial, three regimes, entropy as a diagnostic, why BERT's heads differ. Every one of those is genuine. But a layer doesn't want to converge and doesn't want to be sharp; it wants to hand good features upward, and that goal is **outside the language of the energy function entirely.** The moment you use the frame as a compass for _improvements_ rather than a lens for _understanding_, it points the wrong way.

---

## §6.7 N queries, one landscape — where "sequence" is, and where it isn't

Triggered by the question _"so how does Hopfield do **sequence** attention?"_ — which has a blunt answer worth stating first:

> **It doesn't. $\mathbf{X}$ is a set, not a sequence.** "$N$ = sequence length" in Appendix B means _number of stored patterns_ and nothing more.

Everything below is what fills that hole.

### 6.7.1 It is not one marble

$\boldsymbol\xi$ lowercase = one query = one marble. But §6 uses $\boldsymbol\Xi$ **uppercase** — $N$ marbles stacked into a matrix:

$$\boldsymbol\Xi^{\text{new}} = \operatorname{softmax}(\beta,\boldsymbol\Xi\mathbf{X}^\top)\mathbf{X}$$

| object                   | count | comes from                    |
| ------------------------ | ----- | ----------------------------- |
| landscape                | **1** | the whole sequence, via $W_K$ |
| wells in it              | $N$   | one per key token             |
| marbles dropped into it  | $N$   | one per query token           |
| marble $i$'s start point | $q_i$ | that token's query vector     |

One sequence, one head, one layer ⇒ **one landscape**, with $N$ marbles released into it **simultaneously**. Within a single step the marbles **never interact** — they merely happen to be rolling on the same terrain. $N$ marbles × $N$ wells is the $N\times N$ attention matrix, and the source of the $O(N^2)$ cost.

**"Self"** in self-attention = marbles and wells are minted from the same tokens. Each token queries a landscape it is itself part of.

> Correction to the natural first picture: **one input = one whole sequence = one landscape.** Not one landscape per token.

### 6.7.2 Where order actually lives

Permute the columns of $\mathbf{X}$ and the landscape is **unchanged** — the wells just swap labels. That's permutation equivariance, and it is not an oversight:

$$\text{order cannot be represented by the mechanism} ;\Longrightarrow; \text{positional encoding must exist}$$

Order is baked into the token vectors as **numbers**, before $W_Q,W_K$ ever run. By the time a landscape exists, position is already just another direction in the space. Positional encoding isn't a bolt-on convenience — it is the repair for exactly this gap.

**Causal masking** is the closest thing to "sequence" inside the frame: query $t$ sees only $\mathbf{X}[:,{:}t]$, so a decoder really holds **$N$ nested landscapes**, each one well larger than the last. That structure comes from the mask, not from Hopfield.

⚠️ **Do not conflate the Hopfield iteration index $t$ with position in the sequence.** Hopfield's loop re-asks the _same_ landscape (and by Theorem B it is usually finished in one step). Autoregressive generation is a loop **outside** the layer entirely. Different level.

_(Sequence-storing Hopfield does exist — **asymmetric** $\mathbf{W}$, each pattern pushing toward its successor, limit cycles instead of fixed points: temporal associative memory, Amari / Sompolinsky–Kanter. A different lineage. The 2020 line requires symmetry ⇒ requires convergence ⇒ **forbids cycles by construction.**)_

### 6.7.3 Why any $N$ works — check the shapes

The whole answer to _"how do you train it to accept sets of any size?"_ is: look for $N$ in the trained parameters. It isn't there.

| trained thing                                  | shape                       | contains $N$? |
| ---------------------------------------------- | --------------------------- | ------------- |
| $W_Q, W_K, W_V$                                | $d\times d_k$               | ❌            |
| $\boldsymbol\xi$ in `HopfieldPooling`          | $1\times d_k$               | ❌            |
| $\mathbf{X}_K,\mathbf{X}_V$ in `HopfieldLayer` | $M\times d$ ($M$ is yours)  | ❌            |
| the shared per-instance encoder                | runs one instance at a time | ❌            |

$N$ appears **only at runtime**, as the number of columns that happened to arrive; softmax normalises over whatever is present. Contrast `nn.Linear(N*d, ...)`, which welds $N$ into the weights — that is the layer that _can't_ do this, and the reason §8.1 says pooling isn't optional.

### 6.7.4 The dataset is not what it looks like

The DeepRC setup reads like it demands impossible data. It doesn't:

> **1 patient = 1 bag of ~300,000 receptors = ONE training example carrying ONE bit.**

Two things make that workable:

- **No instance labels exist.** Nobody points at the ~10 relevant receptors. That's what makes it MIL, and it's why the pooling layer has to be learnable at all.
- **The loss reaches all 300,000 anyway.** It flows back through the softmax into a **shared** encoder, so the encoder's real training signal is _patients × instances_, not _patients_. Meanwhile $\boldsymbol\xi$ is a single vector — one question asked of every bag. **The parameters sit on the question, not on the archive.**

And real bags differ in size — 200k for one patient, 400k for another. **Variable $N$ is the default condition here, not a special case.** Batching is pad + mask, exactly as attention already does for uneven sentence lengths.

### 6.7.5 What length extrapolation actually costs

Three items; only the first is a real knob.

**1. $\beta \sim \log N$.** From Theorem (B), $\varepsilon \sim Ne^{-\beta\Delta_i}$: more distractors raise error **linearly** in $N$, while $\beta$ pushes back **exponentially**. So 300k → 400k asks for $\log(4/3)\approx0.29$ against $\log(3\times10^5)\approx12.6$ — roughly **2%**, i.e. nothing. What genuinely needs a move is a jump of _orders of magnitude_ (300 → 300,000). And $\beta$ is one scalar — or the model shifts effective sharpness itself through the norms of $Q,K$ (§6).

**2. Memory.** `HopfieldPooling` has a single query ⇒ $O(Nd)$, which is the only reason $N=300{,}000$ is thinkable. Full self-attention at that $N$ would want $9\times10^{10}$ entries — dead before it starts. **Set problems are affordable precisely because they don't need every-token-queries-every-token.**

**3. Noise floor.** Sample more junk, get more junk that coincidentally resembles the target. This is item 1 restated, not a separate failure.

### 6.7.6 The context-length intuition does not transfer

The reflex _"trained at 300k ⇒ breaks at 400k"_ is borrowed from LLM context limits. It's correct about LLMs and **wrong about the mechanism**:

> Attention has no intrinsic length limit. What breaks when you extend an LLM is the **positional encoding** — RoPE that only ever saw angles in $[0,4096]$ is out of distribution at 8192.

And per §6.7.2, **a set has no positional encoding at all.** There is no "item number 300,001" to encode, because there is no order. The usual culprit is simply absent.

$$\boxed{\text{What locks input length is (a) weights whose shape depends on } N, \text{ and (b) positional encoding.}}$$

$$\text{MIL Hopfield has neither. Only } \beta\sim\log N \text{ and VRAM remain.}$$

### 6.7.7 Two sharpenings that belong here

**$\beta$ does not steepen a well — it melts wells together.** The shape and depth of an individual well come from the pattern itself (direction and norm of $\mathbf{x}_i$); $\beta$ can't touch that. What $\beta$ sets is how `lse` **combines** them: high $\beta$ ⇒ `lse` ≈ `max` ⇒ wells stay distinct and narrow; low $\beta$ ⇒ neighbouring wells fuse into one broad basin whose floor sits at their average. Theorem (C)'s three regimes are therefore _"how many wells got fused"_, not _"how steep the floor is"_ — and sharper always costs basin width (§4). There is no setting that buys both.

**Nobody places the wells; gradient does.** The intuitive summary _"Hopfield is just the surface, the intelligence is in where we put the wells"_ is right in spirit and wrong about the agent. No one positions anything: $W_K$ and the encoder get pushed until **separation** $\Delta_i$ is right — distinguishable things far apart, averageable things close (§6). And per §6.6, that's the whole division of labour:

$$\text{Hopfield supplies the geometry; learning supplies the coordinates.}$$

Which is also why the layer is worth having even though it "knows" nothing: `max()` would kill the gradient, `mean()` would drown a witness rate of $10^{-4}$, and softmax is the one in between that turns _relevance_ into a continuous quantity gradient descent can push on (§8.6).

---

# Part V — The four variants

## §7. One axis explains all of them

Every variant is the same equation. They differ **only** in which slot is data and which is a parameter:

$$y = \mathbf{X}_V\operatorname{softmax}(\beta,\mathbf{X}_K^\top q)$$

|                       | $\boldsymbol\xi$ / Q from | $\mathbf{X}$ / K,V from      | Trained                             | Landscape                  |
| --------------------- | ------------------------- | ---------------------------- | ----------------------------------- | -------------------------- |
| **Hopfield 1982**     | data (query)              | data, frozen after one write | **nothing**                         | persistent, never learned  |
| **`Hopfield`**        | data                      | data                         | $W_Q,W_K,W_V$                       | rebuilt every pass         |
| **`HopfieldPooling`** | **parameter**             | data                         | $\boldsymbol\xi$ + upstream encoder | rebuilt every pass         |
| **`HopfieldLayer`**   | data                      | **parameter**                | $\mathbf{X}_K,\mathbf{X}_V$         | persistent **and** learned |

The useful diagnostic question: **what survives across forward passes?**

- 1982 — the wells survive but never learn
- `Hopfield` — nothing survives except the digging procedure
- `HopfieldPooling` — the _question_ survives
- `HopfieldLayer` — the _wells_ survive and learn

> You are not choosing a mechanism. You are choosing **who digs the wells.**

Where each one physically goes in code:

- **`Hopfield`** → wherever attention already is (seq2seq, cross-attention)
- **`HopfieldPooling`** → wherever you would write `.mean(dim=1)`
- **`HopfieldLayer`** → wherever you would write `nn.Linear`

---

## §8. `HopfieldPooling`

### 8.1 Pooling is not optional

Take DeepRC, because every constraint bites:

> One patient = **~300,000 T-cell receptor sequences.** About **10** of them indicate CMV infection. Available label: **one bit per patient.**

Feeding it all to a linear layer fails three separate ways:

1. $300{,}000\times32 = 9{,}600{,}000$ input dims — **9.6M parameters against 1 bit of label.** Instant overfit.
2. The next patient has 250,000 sequences. **A linear layer breaks on variable size.**
3. Swapping sequence #5 with #5000 must not change the answer. **A linear layer gives a different answer.**

You have no choice but to pool. The only question is _how_.

### 8.2 Why mean pooling is structurally dead

$$\frac{\partial L}{\partial\mathbf{z}_i} = \frac{1}{N}\frac{\partial L}{\partial\bar{\mathbf{z}}} \qquad\text{— identical for every } i$$

A disease-marker sequence and a junk sequence receive **exactly the same gradient.** The CNN can never learn to distinguish them. Not a tuning problem — no amount of data fixes it.

### 8.3 The pipeline

```
bag of 300,000 sequences (N varies)
        ↓
one shared CNN, applied to every sequence     →  N × 32
        ↓
HopfieldPooling   (ξ is nn.Parameter)         →  1 × 32
        ↓
linear + sigmoid                              →  1 bit
```

Trained end-to-end with a single loss. Parameters don't explode because **one CNN is shared across all 300,000 instances.**

Inside the pooling layer there is nothing but:

$$\text{out} = \sum_i a_i\mathbf{z}_i, \qquad a = \operatorname{softmax}(\beta,\boldsymbol\xi^\top\mathbf{Z})$$

$\boldsymbol\xi$ is one 32-dim vector declared as `nn.Parameter`. It is a **fixed question asked of every bag.** The bag changes, so the wells change, so the answer changes — but the question is the same every time. Multiple $\boldsymbol\xi$ = multiple heads = several different questions, concatenated.

### 8.4 Variable length is free

$$\sum_i a_i = 1 \quad\text{always, regardless of } N$$

The output always lies in the convex hull of ${\mathbf{z}_i}$ ⇒ scale never explodes, order doesn't matter, $N$ can change between examples. **All of it falls out of the shape of softmax** — no special machinery.

$N$ affects exactly one thing: larger $N$ ⇒ flatter initial softmax ⇒ you need higher $\beta$ for the loop to ignite. Rule of thumb $\beta \sim \log N$.

### 8.5 What it actually does: routes gradient

$$\text{out} = \sum_i a_i\mathbf{z}_i \quad\Longrightarrow\quad \frac{\partial L}{\partial\mathbf{z}_i} ;\propto; a_i\cdot\frac{\partial L}{\partial\text{out}}$$

Selected instances get large gradients; the rest get roughly zero. The loop:

1. $\boldsymbol\xi$ selects some instances (initially near-arbitrary)
2. Those get gradient; the rest get ~zero
3. The CNN updates only with respect to those
4. If the selection was useful, loss drops → the CNN encodes those instances more distinctly, and $\boldsymbol\xi$ is pushed toward them
5. Repeat — **the signal amplifies itself**

> **Hopfield doesn't learn what to store. It makes the thing above it learn.** It is a gradient router.

### 8.6 The sharper reason: expressivity, not gradient size

"Signal accumulates across examples, noise cancels" is true, but it's how _every_ SGD model works — stated alone it explains nothing.

The real problem with mean pooling is that the classifier only ever sees $\bar{\mathbf{z}}$. The hypothesis _"attend only to instances resembling $\mathbf{s}$"_ has **no parameter that can represent it.** Same category of failure as a linear model on XOR: not a sample-size problem, a **representability** problem.

$$\boldsymbol\xi \text{ is precisely the parameter that makes "selection" expressible}$$

Only then does the ordinary accumulate-and-cancel dynamic have something to act on.

### 8.7 Known failure mode

If the signal is too faint or the dataset too small, the loop **never ignites** — $\boldsymbol\xi$ parks at the mean forever. Real, documented limitation of attention-based MIL. Mitigations: multiple heads, $\beta$ tuning/annealing, more data.

---

## §9. `HopfieldLayer`

### 9.1 It is a learned lookup table

$\mathbf{X}_K$ and $\mathbf{X}_V$ are both `nn.Parameter`, randomly initialised and trained with everything else:

|                | shape                    | is                            | does                                 |
| -------------- | ------------------------ | ----------------------------- | ------------------------------------ |
| $\mathbf{X}_K$ | $d_{\text{in}}\times M$  | $M$ prototypes in input space | **"does the input look like this?"** |
| $\mathbf{X}_V$ | $d_{\text{out}}\times M$ | $M$ answers                   | **"if so, return this"**             |
|                |                          |                               |                                      |

Column $m$ of each is paired. Forward pass is three lines:

$$s = \beta,\mathbf{X}_K^\top q ;; (M\text{ scores}) ;\longrightarrow; a = \operatorname{softmax}(s) ;\longrightarrow; y = \mathbf{X}_V,a$$

Compare the input against $M$ prototypes, soft-select, return a mixture of the stored answers.

> ⚠️ Similarity here is an **inner product, not a distance.** The regions of input space that map to a given prototype are angular wedges radiating from the origin, not balls around the prototype. Direction _and_ magnitude both matter. True of all attention, not just this layer.

### 9.2 Why it can replace `nn.Linear`

**Mechanically:** identical signature, $\mathbb{R}^{d_{\text{in}}} \to \mathbb{R}^{d_{\text{out}}}$. Drop-in.

**Interestingly:**

$$\text{Linear: } y_k = \sum_j W_{kj},q_j \qquad\qquad \text{Hopfield: } y = \sum_m a_m(q)\cdot v_m$$

In Linear, $W$ **does not depend on $q$ at all** — every input gets the same transformation, forever. In `HopfieldLayer`, the mixing weights **are a function of the input.** Different inputs get processed differently.

Linear says _"mix all dimensions by this fixed recipe."_ `HopfieldLayer` says _"this input resembles prototype 37 — send back its answer."_

It is a **soft k-NN whose exemplars are parameters.** Instead of storing training data, it learns $M$ synthetic prototypes.

|                  | Linear                        | `HopfieldLayer`                                      |
| ---------------- | ----------------------------- | ---------------------------------------------------- |
| possible outputs | anywhere in the space         | **only the convex hull of $\mathbf{X}_V$'s columns** |
| how it selects   | fixed mixing of dimensions    | **content similarity** against $\mathbf{X}_K$        |
| interpretable    | no                            | yes — you can read off which prototype fired         |
| parameters       | $d_{\text{in}}d_{\text{out}}$ | $M(d_{\text{in}}+d_{\text{out}})$                    |

With $d_{\text{in}}=d_{\text{out}}=128$: Linear = 16,384; $M=32$ ⇒ 8,192. Half.

### 9.3 Where you put it

- **Final classifier head**, replacing the last `nn.Linear` — the model ends by comparing against prototypes rather than cutting the space with hyperplanes
- **A memory module mid-network** — e.g. tabular / drug design, where $\mathbf{X}_K$ can even be initialised from real known molecules and then fine-tuned
- **Replacing an LSTM's recurrent state** — query = current hidden state, memory bank = what the model learned; no state that gradually forgets

### 9.4 Caveats that bite

**Output can never leave the convex hull of $\mathbf{X}_V$**, because $\sum a_m = 1$. Regression that must extrapolate beyond seen values is impossible here, where Linear does it trivially. **This is simultaneously the strength and the weakness** — that constraint _is_ the inductive bias that stops small-data overfitting.

**$\beta$ must be right.** Too low ⇒ every input gets nearly the same output, the layer degenerates to a constant. Too high ⇒ hard nearest-neighbour, unselected prototypes get ~zero gradient and never train.

**$M$ is yours to choose**, like a cluster count. Too few underfits; too many overfits and eats the parameter advantage.

---

## §10. `HopfieldLayer` vs attention

The mechanism is **identical, not similar.** Both are $y = \mathbf{X}_V\operatorname{softmax}(\beta\mathbf{X}_K^\top q)$. One thing differs: **where $\mathbf{X}_K,\mathbf{X}_V$ come from.** That one thing cascades:

|                   | self-attention                    | `HopfieldLayer`                                   |
| ----------------- | --------------------------------- | ------------------------------------------------- |
| memory lifetime   | dies with the sequence            | persists for the model's life                     |
| what it can know  | only what's inside the same input | **can inject knowledge not present in the input** |
| number of wells   | $N$ = sequence length, varies     | $M$ = a hyperparameter                            |
| number of queries | $N$ at once (every token queries) | usually one                                       |
| cost              | $O(N^2 d)$ — **quadratic**        | $O(Md)$ — constant in input length                |

Row two is the important one. **Attention can only relate things sitting in the same box.** If the answer isn't in the sequence, attention cannot help. `HopfieldLayer` is a channel through which the whole training set enters.

> Attention = **searching the document in front of you.** `HopfieldLayer` = **searching the notebook you've been keeping for years.** Same act of looking things up; completely different source.

### The punchline

A Transformer block already contains both:

```
tokens of this sequence          (N × d)
        ↓
self-attention                   ← searches what's in front of it, dies each pass
        ↓
FFN  ≈  HopfieldLayer            ← searches trained weights, persists
```

$\text{FFN} = W_2,\sigma(W_1 q)$: the rows of $W_1$ act as keys, the columns of $W_2$ as values. It differs from `HopfieldLayer` only in using ReLU/GELU instead of softmax, hence no normalisation. (Geva et al. 2021. This is an _interpretation_, not a mathematical identity — don't overclaim it.)

Read that way, **a Transformer block is two lookups stacked**: first in the input, then in what the model remembers. Which explains rather well why models answer things that were never in the prompt — the knowledge lives in the second stage.

> "It works the same" is **right at the level of mechanism, wrong at the level of function.**

---

# Part VI — softmax vs ReLU

## §11. Budget vs no budget

This is the cleanest way to see it. Given raw scores $s$:

$$\textstyle\sum_m \operatorname{softmax}(s)_m = 1 \text{ always} \qquad\qquad \operatorname{ReLU}(s) \text{ has no constraint at all}$$

Two consequences you can feel immediately:

**Add another matching key.** Under softmax every existing weight **shrinks** — the newcomer steals from a fixed budget. Under ReLU nothing moves; each unit has its own budget.

**Shift every score by a constant.** Softmax output is **completely unchanged** — it only sees _differences_. ReLU output rises or dies entirely — it has an **absolute threshold** that softmax simply does not possess.

### The sparsity inversion

$$e^{\beta s}>0 ;\text{ always} \quad\Longrightarrow\quad \textbf{softmax can never output an exact zero}$$

ReLU can, and does, for most units. **ReLU is sparser in the exact-zero sense; softmax is merely _concentrated_.** Softmax _crowds_; ReLU _cuts_.

### Why FFN needs ReLU — with numbers

BERT-base has $d_{\text{ff}} = 3072$. Suppose 50 units genuinely should fire for a given input.

- **Softmax:** those 50 split a budget of 1 ⇒ ~0.02 each, and the other 3022 still consume some. The output ends up barely distinguishable from the mean of all of $\mathbf{X}_V$. **The more knowledge you store, the fainter each fact becomes.**
- **ReLU:** those 50 fire at whatever magnitude they want; the other 3022 are **exactly zero** and cost nothing.

This is superposition. FFN's job is to answer _"this input has features A + D + K + …"_ simultaneously. Softmax is built to answer _"take item 37."_ **Different jobs.**

### The mechanical root: the Jacobian

$$\nabla\operatorname{softmax} = \beta\big(\operatorname{diag}(a) - aa^\top\big) \qquad\qquad \nabla\operatorname{ReLU} = \operatorname{diag}(\mathbb{1}[s>0])$$

Softmax has off-diagonal terms $-a_ia_j$ ⇒ **every unit's gradient is entangled with every other's.** Pushing $a_3$ up automatically pushes the others down. (Same matrix as the `lse` convexity proof in §5.2.)

ReLU's Jacobian is **purely diagonal** ⇒ each unit learns in complete isolation. 3072 units can each grab a different feature without ever negotiating.

### Choosing

Ask one question: **is the right answer "one thing" or "several things at once"?**

|                      | softmax                                                         | ReLU                                                  |
| -------------------- | --------------------------------------------------------------- | ----------------------------------------------------- |
| nature               | **zero-sum** — one rises, others must fall                      | **independent** — all can fire together               |
| suited to            | retrieving one item; controlled weighted averages; variable $N$ | accumulating many features; storing lots of knowledge |
| information per pass | capped by $\sum a = 1$                                          | uncapped                                              |

The budget is a **feature** on the softmax side — it's exactly why `HopfieldPooling` handles any sequence length for free.

### A correction worth keeping

It's tempting to say "ReLU survives because modern models are robust to data volume, whereas Hopfield dies there, so it needs softmax." That mixes two different things.

- **Hopfield capacity** = how many patterns you can store before retrieval gets confused. A crosstalk problem at retrieval time.
- **FFN + ReLU** = no such notion, because **FFN is not trying to retrieve anything cleanly.** It doesn't need to keep prototypes distinguishable; it just emits features that add up.

ReLU doesn't _survive_ the capacity problem. **It's playing a different game** — one with no clean single-item retrieval, so there's nothing for crosstalk to ruin.

---

## §12. So why does `HopfieldLayer` use softmax?

**It didn't choose.** Re-read §5.3: softmax fell out of $\partial\operatorname{lse}/\partial z_i$. The layer is _an energy function's update rule wrapped in a module._ Strip out the softmax and you haven't swapped a hyperparameter — you've discarded:

- the energy function itself
- the convergence proof (iterating more than once becomes meaningless)
- the three-regime theory
- the capacity bounds
- $\beta$ entirely (ReLU has no temperature)

> **The "Hopfield" in `HopfieldLayer` _is_ the softmax.**

And note what you'd actually be proposing:

$$\text{HopfieldLayer} + \text{ReLU} ;=; W_2,\operatorname{ReLU}(W_1 q) ;=; \textbf{FFN}$$

$\mathbf{X}_K \to W_1$, $\mathbf{X}_V \to W_2$. So the empirical answer to "can we use ReLU instead?" is: **yes, and it's called the FFN**, and every Transformer on earth already runs it in every block.

### Accidentally well-matched

Softmax's constraints happen to align with the layer's target audience:

- **convex-hull confinement** → a very strong regulariser → exactly what **small datasets** need
- **$\sum a = 1$** → readable as "this sample is 62% prototype 3" → exactly what **tabular / drug design** needs, where results must be explained
- **theory attached** → capacity bounds, regimes, a $\beta$ dial → things FFN has none of

FFN suits **lots of data, lots of knowledge, many features at once.** `HopfieldLayer` suits **little data, few prototypes, one answer at a time.** Same slot in the network, opposite inductive biases. That's why both exist.

### What is it designed for — three honest layers

**1. What the paper says.** A trainable associative memory; a replacement for `nn.Linear` or LSTM state on small datasets, tabular data, drug design.

**2. Its actual role in the paper.** The **third corner of a completeness argument.** The thesis is "attention = Hopfield retrieval." To claim the framework is general, every data/parameter assignment must yield a usable layer:

- data in Q **and** K/V → `Hopfield` = attention
- parameter in Q → `HopfieldPooling` = set / MIL
- parameter in K/V → `HopfieldLayer` = persistent memory

`HopfieldLayer` closes the last cell. It exists because **the table had to be complete**, not because a problem was demanding it.

**3. Empirically.** It's the least-used of the three. `HopfieldPooling` is the one that wins real problems (pathology, immunology, point clouds). `HopfieldLayer` remains niche, and nobody changed how large Transformers are built because of it.

---

## §13. GELU and SwiGLU

### GELU

$$\operatorname{ReLU}(x) = x\cdot\mathbb{1}[x>0] \qquad\qquad \operatorname{GELU}(x) = x\cdot\Phi(x)$$

Identical shape; the **hard switch** $\mathbb{1}[x>0]$ becomes a **soft switch** $\Phi(x)$ (the Gaussian CDF). Instead of "is it above zero?", it asks "what fraction of the time would it be?"

Two consequences:

**Small negatives don't die.** GELU dips slightly negative around $x\in(-2,0)$ before approaching zero. This is where **gradient survives**. The _dying ReLU_ problem is that once a unit lands on the negative side its gradient is exactly zero and it can never come back. GELU leaves a thin slope to climb.

**Continuous derivative.** ReLU has a kink at 0 where the derivative jumps 0 → 1. GELU is smooth everywhere, which suits curvature-estimating optimisers (Adam) better.

### SwiGLU — a structural change, not just a function swap

$$\text{standard FFN: } W_2,\sigma(W_1 x) \qquad\qquad \text{SwiGLU: } W_3\big(\underbrace{\operatorname{Swish}(W_1 x)}_{\text{gate}} \odot \underbrace{W_2 x}_{\text{content}}\big)$$

It splits into two paths and multiplies them elementwise. One path is a **gate**, the other is **content**; the gate decides how much content passes.

The point: in ReLU/GELU, the thing deciding _whether to fire_ and the thing being _sent onward_ are **the same number**. SwiGLU **separates those two jobs**, so the model can learn _when to open_ independently of _what to let through._

Cost: three matrices instead of two. In practice $d_{\text{ff}}$ shrinks to about $\tfrac23$ to keep the parameter count level.

### Why does it win? — honest answer

**Nobody really knows.** The author of the SwiGLU paper says so explicitly, attributing the improvement to divine benevolence — a well-known line in the field.

The evidence is **purely empirical**: consistently slightly lower loss, holds up as you scale, so it became the default (LLaMA, PaLM, and later models). The gains are typically a few percent, not a paradigm shift. And results here are sensitive to learning rate, initialisation, and model scale — plain ReLU is often competitive when tuned properly. Don't assume newer = better.

### The meta-point worth keeping

|                              | how the knowledge was produced                                                                                                                                           |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Hopfield side**            | energy function → prove monotone descent → prove convergence → derive capacity bounds → softmax falls out of a derivative. **Everything deduced from first principles.** |
| **Activation-function side** | try ten of them, keep whichever has the lowest loss. **Not a single theorem.**                                                                                           |

Both work. They deliver different things — one delivers understanding, the other delivers benchmark numbers. And in fairness: **what actually drives Transformers today is mostly the second column.**

---

## §14. Honest assessment

**On the theory:**

1. **"Explains" vs "equals."** Attention having the same _form_ as a Hopfield update doesn't prove Transformers _operate on_ associative-memory principles. A reasonable reading is: a beautiful mathematical coincidence with limited predictive power.
2. **The capacity bound is loose.** Worst-case, random patterns on a sphere — nothing like real embeddings, which are structured and clustered.
3. **In real code, $E$ is never computed.** A Transformer only ever evaluates `softmax(QKᵀ/√d)V`. The energy landscape is a **lens for seeing**, not an object that gets processed.

**By domain:**

| Framing                         | Verdict                                                                                        |
| ------------------------------- | ---------------------------------------------------------------------------------------------- |
| As an NLP architecture          | **Unimportant.** Nobody uses it. It explains 2017 retroactively rather than inventing anything |
| As a layer for set/MIL problems | **Genuinely useful.** Real deployment in pathology, immunology, point clouds                   |
| As theory                       | **Important.** Gives vocabulary for what attention does, beyond "it just works"                |

$$\text{Hopfield 1982} ;\gg; \text{Hopfield 2020}$$

The 2024 Nobel Prize in Physics honoured the **1982** work — associative memory and spin glasses — not this paper. Which is fitting: 1982 is the weakest model in this entire document, and it's the one that first showed content-addressable memory _emerging_ from an ordinary dynamical system. Everything since is tuning.

> Hopfield isn't a model you take to a leaderboard. It's a **framework that explains why the model you already use works.**

**Next:** Universal Hopfield Networks (Millidge et al., 2022) generalises all associative-memory models as `separation ∘ similarity ∘ projection` — one abstraction level above this document.

---

# Appendix A — Traps

Wrong pictures that felt right along the way. Kept because they'll feel right again on re-reading.

**❌ "The landscape is a store you write into."** There is no Hopfield object. $E$ contains only $\mathbf{X}$, $\boldsymbol\xi$, $\beta$. You never write to the landscape — you change what it's computed from. A landscape lives one forward pass.

**❌ "Each $\mathbf{X}$ has its own learned equation."** Every landscape uses the **same** equation, hardcoded in 2020, unchanged by training. Two landscapes differ only in the numbers inside $\mathbf{X}$. Like gravity: one formula, many mountains.

**❌ "$\beta$ blends landscapes together."** There is only ever one landscape. $\beta$ blends **wells within it.** The system never holds two landscapes at once.

**❌ "A1…C are a spectrum of $\mathbf{X}$."** They **are** $\mathbf{X}$ — its columns. The spectrum is $a = \operatorname{softmax}(\beta\mathbf{X}^\top\boldsymbol\xi)$, which changes with every query. $\mathbf{X}$ = the vertices; the reachable outputs = their convex hull.

**❌ "It's a giant hashmap."** It's exactly as big as $\mathbf{X}$ — $N$ patterns, $N$ wells. And unlike a hashmap it returns **blends**: at low $\beta$ it returns things that were never stored. A bug for recall, a **feature** for MIL.

**❌ "Hopfield does sequences."** $\mathbf{X}$ is a **set** of wells with no order at all — permute its columns and the landscape is identical. Attention just drops $N$ marbles into one shared landscape at once. Order arrives via **positional encoding** (already numeric inside $\mathbf{X}$) and via **causal masking** ($N$ nested landscapes), never from the Hopfield mechanism. §6.7

**❌ "Trained at $N=300$k, so it breaks at $N=400$k."** No trained parameter has $N$ in its shape — $N$ is only a runtime column count, and softmax normalises over whatever showed up. What locks length elsewhere is (a) weights shaped by $N$ and (b) positional encoding; a set has neither. All that remains is $\beta\sim\log N$ (logarithmic — 300k→400k is ~2%) and VRAM. §6.7.3, §6.7.5

**❌ "LLMs can't extrapolate context, so attention has a length limit."** Attention doesn't. **RoPE** does — angles outside the trained range are out of distribution. Wrong component. §6.7.6

**❌ "The Hopfield loop index $t$ is the position in the sequence."** The loop re-asks the **same** landscape and is usually done in one step (Thm B). Autoregressive generation is a loop **outside** the layer. §6.7.2

**❌ "$\beta$ controls how steep each well is."** $\beta$ can't touch an individual well — that's fixed by the pattern's direction and norm. It controls how much neighbouring wells **melt together** inside `lse`. The three regimes are "how many wells fused," not "how steep the floor." And sharper always costs basin width. §6.7.7

**❌ "We choose where to put the wells to make it smart."** Right spirit, wrong agent — nobody places them. $W_K$ / the encoder get pushed by gradient until **separation** $\Delta_i$ is right. Hopfield supplies the geometry; learning supplies the coordinates. §6.7.7

**❌ "Hopfield 1982 has a $\beta$."** It doesn't. Its dial is welded to the exponent 2. $\beta$ arrives with $F=\exp$ in 2017.

**❌ "1982 has no sharp nonlinearity."** $\operatorname{sgn}$ is harder than any softmax. It's just in the wrong place — per-neuron, after crosstalk has already been summed in.

**❌ "It knows which instances are relevant."** It knows nothing. It supplies a parameter that makes relevance **expressible**, so gradient descent can find it.

**❌ "softmax just does ranking before multiplying by V."** Softmax **is** the nonlinearity, and it's the whole reason the layer is called Hopfield.

**❌ "ReLU is better than softmax."** Better _for that job._ And modern FFNs use GELU/SwiGLU anyway, while a research line is busy removing softmax from attention to kill the $O(N^2)$ cost.

**❌ "Rank collapse happens because $\beta$ is too high — the fixed point returns a single item."** Backwards. **Low $\beta$** puts the fixed point at the average of _everything_, which is exactly all-tokens-identical. High $\beta$ gives a near-permutation attention matrix, which is full rank. The _picture_ of collapse (every position becomes one thing) is right; the cause label is on the wrong side.

**❌ "A Transformer descends toward a final answer across its layers."** Each layer builds a **new** landscape from the previous layer's output. There is no energy function for the stack, so "descent" isn't defined across depth. Layer 5 is not closer to any minimum than layer 2. Depth pays off because of the **training loss**, not downhill motion at inference.

**❌ "So getting layers closer to slope = 0 would be an improvement."** The fixed point is not an achievement a layer is failing to reach — it's meaningless here. Layers produce features for the next layer; convergence has no intrinsic value. **Optimising toward it is optimising toward nothing.**

**❌ "High $\beta$ makes variance explode."** Softmax still normalises, so values stay in the convex hull — nothing numerically blows up. What degrades is **gradient behaviour**: a near-one-hot $a$ drives $\operatorname{diag}(a)-aa^\top \to 0$, and training gets unstable and spiky. Sensitivity, not magnitude.

**❌ "Then hard-code high $\beta$ in the late layers."** The model already learns that schedule (BERT: soft below, sharp above, all on one $1/\sqrt{d_k}$). Hard-coding it removes freedom and is coarser — sharpness is per **head**, not per layer, and some soft upper-layer heads are soft for good reasons. What the field actually deploys here is **QK-norm**, which does the opposite.

---

# Appendix B — Quick reference

$$E(\boldsymbol\xi) = -\operatorname{lse}(\beta,\mathbf{X}^\top\boldsymbol\xi) + \tfrac12|\boldsymbol\xi|^2 + \text{const}$$

$$\boldsymbol\xi^{\text{new}} = \mathbf{X}\operatorname{softmax}(\beta\mathbf{X}^\top\boldsymbol\xi) ;=; \operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k})\mathbf{V}$$

| Symbol           | Meaning                                                   | In a Transformer               |
| ---------------- | --------------------------------------------------------- | ------------------------------ |
| $\boldsymbol\xi$ | state / query                                             | Q                              |
| $\mathbf{X}$     | stored patterns                                           | K (and V)                      |
| $\beta$          | inverse temperature — the regime dial                     | $1/\sqrt{d_k}$                 |
| $N$              | number of stored patterns                                 | sequence length                |
| $M$              | number of prototypes (`HopfieldLayer`) / max pattern norm | —                              |
| $\Delta_i$       | separation of pattern $i$                                 | how distinguishable a token is |
| $a_i$            | retrieval weight                                          | attention weight               |

**Rules of thumb**

- $\beta\uparrow$ → peaked → single-pattern retrieval; $\beta\downarrow$ → flat → averaging
- Need $\beta \sim \log N$ to hold error fixed as the bag grows
- $d\uparrow$ → separation ↑ → retrieval _easier_ (blessing of dimensionality)
- Entropy $H(\mathbf{a})$ diagnoses the regime: high = drowning, near-zero = over-peaked
- Compute is $O(Nd)$ per query — same as attention, same quadratic cost across $N$ queries

**Mantras**

- It's a loop, not a pipeline
- Wells are _written_, not _dug_
- A landscape lives exactly one forward pass
- One equation for every landscape — only $\mathbf{X}$ changes
- Nobody chose softmax; it fell out of a derivative
- Hopfield doesn't learn what to store — it **routes gradient** so the encoder learns
- It doesn't know what's relevant; it makes relevance **findable by gradient descent**
- Attention searches the page in front of you; `HopfieldLayer` searches the notebook
- softmax = pick one; ReLU = accumulate many
- Merged wells are a bug for recall and a feature for MIL
- slope = 0 means **self-consistency**, not flatness: asking with the answer returns the answer
- Hopfield's fixed point is the **answer**; a Transformer's is the **failure mode**
- Collapse comes from $\beta$ too **low**, not too high — averaging is consensus dynamics
- $\beta$ sets blending per step; the **architecture** decides whether blending accumulates
- Hopfield has no order — a set of wells, $N$ marbles at once; order is smuggled in as numbers
- No trained parameter has $N$ in its shape — that alone is why any bag size works
- $\beta$ fights $N$ logarithmically: $\varepsilon \sim Ne^{-\beta\Delta_i}$
- $\beta$ doesn't steepen wells, it melts them together
- Hopfield supplies the geometry; learning supplies the coordinates
- No energy function exists for the stack — "descent" is undefined across depth
- The frame explains the **mechanism**, never the **purpose**. Use it as a lens, not a compass

---

## References

- Hopfield, J.J. (1982) — _Neural networks and physical systems with emergent collective computational abilities_
- Amari (1972) · Sompolinsky & Kanter (1986) — _asymmetric/temporal associative memory_ — the lineage that actually stores sequences (§6.7.2)
- Krotov & Hopfield (2016) — _Dense Associative Memory for Pattern Recognition_
- Demircigil et al. (2017) — _On a model of associative memory with huge storage capacity_
- Ramsauer et al. (2020) — _Hopfield Networks is All You Need_ · [arXiv:2008.02217](https://arxiv.org/abs/2008.02217)
- Widrich et al. (2020) — _Modern Hopfield Networks and Attention for Immune Repertoire Classification_ (DeepRC)
- Millidge et al. (2022) — _Universal Hopfield Networks_
- Yuille & Rangarajan (2003) — _The Concave-Convex Procedure_
- Dong et al. (2021) — _Attention is not all you need: pure attention loses rank doubly exponentially with depth_
- Geva et al. (2021) — _Transformer Feed-Forward Layers Are Key-Value Memories_
- Hendrycks & Gimpel (2016) — _Gaussian Error Linear Units (GELUs)_
- Shazeer (2020) — _GLU Variants Improve Transformer_

---

## Companion files

- `2-hopfield.md` — the same material organised around the paper's structure
- `3-transformer-internals.md` — $W_Q,W_K,W_V,W_O$, head specialisation, phases of capability formation, grokking / induction bump / double descent
