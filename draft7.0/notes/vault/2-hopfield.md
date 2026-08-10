# Hopfield Networks: What The Hell Are They, Actually?

> Notes from an argument fought to conclusion — on _"how is it smart, what is its structure, and what is it actually doing?"_
>
> Primary reference: Ramsauer et al., **"Hopfield Networks is All You Need"** (2020)

---

## How to read this

Most explanations of Hopfield networks say "data becomes wells in an energy landscape" and stop there. That sentence answers nothing. This document is built around the questions that sentence leaves open:

- **What kind of object is a Hopfield network?** (Not a pipeline. §0)
- **Where is the learning?** (In classical Hopfield: nowhere. §2–4)
- **So how is it smart?** (It isn't. It makes the thing above it smart. §9–10)
- **How is this different from CNN + linear layer?** (This is the real question. §9)
- **Does it matter?** (Less than the hype suggests. §17)

Sections 0–5 build machinery. §7 is the punchline. **§7, §9, and §10 are the sections that actually answer "how is it smart."** If you're re-reading this later and short on time, read those three.

---

# Part I — Foundations

## §0. What kind of object is this?

Before any math, fix the category. This is where the confusion starts.

A **Transformer is a pipeline.** Data enters one end, flows through block 1 → block 2 → ... → exits the other. It has direction, depth, and weights inside the pipe. When you make it smarter, you adjust things inside the pipe.

A **Hopfield network is not a pipeline.** It has:

- no layers
- no depth
- no forward pass

It has exactly two things:

1. **One vector** (the state)
2. **One rule**, applied to that vector repeatedly until it stops moving

It is a **loop**, not a pipeline. Formally: a _dynamical system_, not an _architecture_.

So what are you doing when you use one? You aren't pushing data through layers. You are:

> **Shaping a landscape so it has valleys where you want them, then dropping a marble in.**

Hold that image. Everything below refines it.

### The problem it solves: content-addressable memory

|        | Address-addressable (RAM) | Content-addressable (associative memory)  |
| ------ | ------------------------- | ----------------------------------------- |
| Input  | an address                | a _partial or corrupted_ piece of data    |
| Output | the data at that address  | the _complete, clean_ nearest stored item |

Store 100 face images. Feed in a face with half occluded. Get back the complete matching face.

The mechanism: build an energy function $E$ whose local minima sit exactly at the stored patterns, then let the query roll downhill.

$$\text{query } \boldsymbol\xi \;\longrightarrow\; \text{descend } E \;\longrightarrow\; \text{stored pattern}$$

Every version below is a different design of $E$. That's the whole subject.

---

## §1. Classical Hopfield (1982)

### 1.1 Definition

- State: $\mathbf{x} \in \{-1,+1\}^d$ — $d$ binary neurons
- Weights: $\mathbf{W} \in \mathbb{R}^{d\times d}$, required **symmetric** ($w_{ij}=w_{ji}$) with **zero diagonal** ($w_{ii}=0$)
- Bias: $\mathbf{b}\in\mathbb{R}^d$

**Energy:**

$$E(\mathbf{x}) = -\frac{1}{2}\mathbf{x}^\top\mathbf{W}\mathbf{x} + \mathbf{b}^\top\mathbf{x}$$

**Update (asynchronous — one neuron at a time):**

$$x_i \leftarrow \operatorname{sgn}\Big(\sum_j w_{ij}x_j - b_i\Big)$$

### 1.2 Proof: energy never increases

This proof is the entire reason the model works. Three steps.

**Step 1 — isolate the terms containing $x_i$.**

The terms involving $x_i$ come from row $i$ and column $i$:

$$-\frac{1}{2}\Big(\underbrace{\sum_l w_{il}x_ix_l}_{\text{row }i} + \underbrace{\sum_k w_{ki}x_kx_i}_{\text{col }i}\Big) = -x_i\sum_{l\neq i}w_{il}x_l$$

The two halves are equal by **symmetry**; the $l=i$ term vanishes by **zero diagonal**. (This is why both constraints exist — without them this step fails and the whole model loses its guarantee.)

Adding the bias:

$$E = -x_i\underbrace{\Big(\sum_{l\neq i}w_{il}x_l - b_i\Big)}_{\displaystyle h_i \;=\;\text{local field}} + \;C$$

where $C$ **does not depend on $x_i$ at all.** That's the crux.

**Step 2 — the update can only help.**

$$E^{\text{new}} - E^{\text{old}} = -(x_i^{\text{new}} - x_i^{\text{old}})\,h_i$$

Since $x_i^{\text{new}} = \operatorname{sgn}(h_i)$, we get $x_i^{\text{new}}h_i = |h_i|$ — the maximum possible value of $x_i h_i$ over $x_i\in\{-1,+1\}$. Therefore

$$x_i^{\text{new}}h_i \ge x_i^{\text{old}}h_i \quad\Longrightarrow\quad \Delta E \le 0 \qquad\blacksquare$$

**Step 3 — convergence.**

The state space has $2^d$ points, so $E$ takes finitely many values. A non-increasing sequence over a finite set must terminate. It halts at a local minimum in **finitely many steps.** $\blacksquare$

> ⚠️ **This proof requires asynchronous updates.** Update all neurons simultaneously and it breaks — you can land in a period-2 limit cycle. Remember this; it returns in §13.

### 1.3 Storing patterns: the Hebbian rule

Given patterns $\{\mathbf{x}^1,\dots,\mathbf{x}^N\}$:

$$\mathbf{W} = \frac{1}{d}\sum_{\mu=1}^N \mathbf{x}^\mu(\mathbf{x}^\mu)^\top \qquad(\text{then zero the diagonal})$$

**Does it work?** Feed in $\mathbf{x}^\nu$ and compute the local field:

$$h_i = \frac{1}{d}\sum_\mu\sum_j x_i^\mu x_j^\mu x_j^\nu = \underbrace{x_i^\nu}_{\text{signal}} + \underbrace{\frac{1}{d}\sum_{\mu\neq\nu}x_i^\mu\langle\mathbf{x}^\mu,\mathbf{x}^\nu\rangle}_{\displaystyle C_i^\nu \;=\;\text{crosstalk}}$$

- **Orthogonal patterns** → crosstalk $=0$ → perfect recall
- **Random patterns** → $C_i^\nu$ is a sum of $(N-1)d$ terms of size $\pm 1/d$ → mean 0, variance $\approx N/d$

Recall fails when crosstalk overwhelms signal:

$$P_{\text{error}} = \Phi\\left(-\sqrt{d/N}\right)$$

### 1.4 Capacity — the fatal weakness

$$\boxed{N_{\max}\approx 0.14\,d}$$

(Demanding _zero_ errors across all patterns tightens this to $N\approx d/(2\ln d)$.)

**Capacity is linear in $d$.** 1000 neurons → ~138 patterns. Push past that and the network enters a spin-glass phase producing **spurious states**: fake wells that are blends of real patterns.

**Root cause: the energy is quadratic.** Quadratic wells are wide and shallow, so they overlap. Everything in Part III is a response to this one fact.

---

# Part II — The critical realization

_This part is the answer to "I can't see what we're touching."_

## §2. Nothing here is trained

**Classical Hopfield has no training procedure.** No gradient descent. No loss function. No backpropagation. No epochs.

Storage is one line of arithmetic:

$$\mathbf{W} = \sum_\mu \mathbf{x}^\mu(\mathbf{x}^\mu)^\top$$

Add the patterns together. Done. It's a million times faster than training a neural network because **it is not training.**

> If you've been hunting for "the learning" inside a classical Hopfield network, you've been hunting for something that isn't there. That confusion is the correct response to the material, not a failure to understand it.

## §3. The wells are not dug — they are written down

Substitute the Hebbian $\mathbf{W}$ back into the energy:

$$E(\boldsymbol\xi) = -\frac{1}{2d}\sum_\mu \langle\mathbf{x}^\mu,\boldsymbol\xi\rangle^2$$

Read this carefully. Each stored pattern contributes **exactly one term** to the sum. That term is most negative when $\boldsymbol\xi$ aligns with $\mathbf{x}^\mu$.

$$\text{1 pattern} \;=\; \text{1 term} \;=\; \text{1 well}$$

**No intermediate step. No optimization. No selection.** The landscape is the sum of independent contributions.

### Concrete numbers: $d=4$, two patterns

$\mathbf{x}^1 = (1,1,1,1)$, $\mathbf{x}^2 = (1,-1,1,-1)$ (orthogonal)

| $\boldsymbol\xi$ | $\langle\mathbf{x}^1,\boldsymbol\xi\rangle$ | $\langle\mathbf{x}^2,\boldsymbol\xi\rangle$ | $E$    |             |
| ---------------- | ------------------------------------------- | ------------------------------------------- | ------ | ----------- |
| $(1,1,1,1)$      | 4                                           | 0                                           | **−2** | ← well      |
| $(1,-1,1,-1)$    | 0                                           | 4                                           | **−2** | ← well      |
| $(1,1,1,-1)$     | 2                                           | 2                                           | −1     | one bit off |
| $(1,1,-1,-1)$    | 0                                           | 0                                           | 0      | ← ridge     |

Nothing was done except plugging values into a formula. **The wells appear on their own.**

## §4. "How do two data points get sorted into different wells?"

**They don't get sorted.** There is no sorting step.

- **No labels.** Nobody says "this is class A, this is class B."
- **No decision boundary** discovered by an optimizer.
- Each pattern **digs its own well independently**, unaware the others exist. The wells simply add.

Separation is pure geometry: $\langle\mathbf{x}^1,\boldsymbol\xi\rangle^2$ is large near $\mathbf{x}^1$, small near $\mathbf{x}^2$.

**When does it break?** When patterns are too similar or too numerous. Wells overlap and merge into a single well between them — a **spurious state**. Feed in $\mathbf{x}^1$, get a blend back.

> The entire program of "modern Hopfield" is: **make the wells narrow and deep enough that they stop merging.**

---

# Part III — Making the wells sharper

## §5. Dense Associative Memory (2016–2017)

Krotov & Hopfield's move is disarmingly simple: **generalize the energy.**

$$E(\mathbf{x}) = -\sum_{\mu=1}^N F\big(\langle\mathbf{x}^\mu,\mathbf{x}\rangle\big)$$

$F$ is the **interaction / separation function.**

| $F(z)$    | Capacity     | Source                 |
| --------- | ------------ | ---------------------- |
| $z^2$     | $O(d)$       | = classical Hopfield   |
| $z^n$     | $O(d^{n-1})$ | Krotov & Hopfield 2016 |
| $\exp(z)$ | $O(2^{d/2})$ | Demircigil et al. 2017 |

### Why sharper $F$ buys capacity

Look at the overlaps. The matching pattern gives $\langle\mathbf{x}^\mu,\mathbf{x}\rangle \approx d$; the rest give $\approx\sqrt{d}$.

| $F$   | signal : noise after $F$ | margin           |
| ----- | ------------------------ | ---------------- |
| $z^2$ | $d^2 : d$                | factor $d$       |
| $z^n$ | $d^n : d^{n/2}$          | factor $d^{n/2}$ |
| $e^z$ | $e^d : e^{\sqrt d}$      | **exponential**  |

**The steeper $F$ is, the more completely the largest term drowns out every other term.** Crosstalk stops mattering, so you can store vastly more.

Demircigil's energy:

$$E = -\exp\big(\operatorname{lse}(1,\mathbf{X}^\top\mathbf{x})\big), \qquad \operatorname{lse}(\beta,\mathbf{z}) = \beta^{-1}\log\sum_i e^{\beta z_i}$$

**The price:** narrower wells → smaller basins of attraction → you need a closer cue to retrieve. This trade-off is fundamental and never goes away.

**Remaining limitation:** still binary. Useless for deep learning, where everything is continuous.

---

# Part IV — The 2020 paper

## §6. Continuous Modern Hopfield

### 6.1 Setup

- Stored patterns: $\mathbf{X} = (\mathbf{x}_1,\dots,\mathbf{x}_N)\in\mathbb{R}^{d\times N}$
- State / query: $\boldsymbol\xi\in\mathbb{R}^d$ — **now continuous**
- $M = \max_i\|\mathbf{x}_i\|$

### 6.2 The new energy

$$\boxed{\;E(\boldsymbol\xi) = -\operatorname{lse}(\beta,\mathbf{X}^\top\boldsymbol\xi) + \tfrac12\boldsymbol\xi^\top\boldsymbol\xi + \beta^{-1}\log N + \tfrac12 M^2\;}$$

Term by term:

| Term                                                       | Role                                                                                                                             |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| $-\operatorname{lse}(\beta,\mathbf{X}^\top\boldsymbol\xi)$ | continuous version of Demircigil's exponential — pulls $\boldsymbol\xi$ toward patterns                                          |
| $\tfrac12\|\boldsymbol\xi\|^2$                             | **confinement** — stops $\boldsymbol\xi$ escaping to infinity. Newly necessary: the hypercube $\{-1,1\}^d$ no longer boxes it in |
| $\beta^{-1}\log N + \tfrac12M^2$                           | constants chosen so $E\ge0$                                                                                                      |

**Proof that $E \ge 0$:**

$$\operatorname{lse}(\beta,\mathbf{z}) \le \max_i z_i + \beta^{-1}\log N \le M\|\boldsymbol\xi\| + \beta^{-1}\log N$$

$$\Longrightarrow\; E \ge -M\|\boldsymbol\xi\| + \tfrac12\|\boldsymbol\xi\|^2 + \tfrac12M^2 = \tfrac12\big(\|\boldsymbol\xi\|-M\big)^2 \ge 0 \qquad\blacksquare$$

### 6.3 Deriving the update rule via CCCP

The most elegant part of the paper.

**Step 1 — split into convex + concave.**

$$E = \underbrace{-\operatorname{lse}(\beta,\mathbf{X}^\top\boldsymbol\xi)}_{E_1:\ \text{concave}} \;+\; \underbrace{\tfrac12\boldsymbol\xi^\top\boldsymbol\xi + \text{const}}_{E_2:\ \text{convex}}$$

_Sub-proof that lse is convex._ Let $p = \operatorname{softmax}(\beta\mathbf{z})$. Then

$$\frac{\partial\operatorname{lse}}{\partial z_i} = p_i, \qquad \nabla^2\operatorname{lse} = \beta\big(\operatorname{diag}(p) - pp^\top\big)$$

For any $\mathbf{v}$:

$$\mathbf{v}^\top(\operatorname{diag}(p)-pp^\top)\mathbf{v} = \sum_i p_iv_i^2 - \Big(\sum_i p_iv_i\Big)^2 = \operatorname{Var}_p[v] \ge 0$$

PSD Hessian ⇒ lse convex ⇒ $-$lse concave. $\blacksquare$

> Note $\partial\operatorname{lse}/\partial z_i = p_i$ — **the softmax appears right here, before we've done anything else.** It was hiding inside log-sum-exp all along. Attention was never designed; it was implied by the choice of $F = \exp$.

**Step 2 — apply CCCP.**

The Concave-Convex Procedure (Yuille & Rangarajan): linearize the concave part, keep the convex part, solve

$$\nabla E_2(\boldsymbol\xi^{t+1}) = -\nabla E_1(\boldsymbol\xi^t)$$

With $\nabla E_2(\boldsymbol\xi) = \boldsymbol\xi$ and $\nabla E_1(\boldsymbol\xi) = -\mathbf{X}\operatorname{softmax}(\beta\mathbf{X}^\top\boldsymbol\xi)$:

$$\boxed{\;\boldsymbol\xi^{t+1} = \mathbf{X}\operatorname{softmax}\big(\beta\mathbf{X}^\top\boldsymbol\xi^t\big)\;}$$

**Step 3 — prove it descends.**

From concavity of $E_1$ (lies below its tangent):

$$E_1(\boldsymbol\xi^{t+1}) \le E_1(\boldsymbol\xi^t) + \nabla E_1(\boldsymbol\xi^t)^\top(\boldsymbol\xi^{t+1}-\boldsymbol\xi^t)$$

From convexity of $E_2$ (lies above its tangent at $\boldsymbol\xi^{t+1}$):

$$E_2(\boldsymbol\xi^t) \ge E_2(\boldsymbol\xi^{t+1}) + \nabla E_2(\boldsymbol\xi^{t+1})^\top(\boldsymbol\xi^t - \boldsymbol\xi^{t+1})$$

Add them:

$$E(\boldsymbol\xi^{t+1}) - E(\boldsymbol\xi^t) \;\le\; \underbrace{\big[\nabla E_1(\boldsymbol\xi^t) + \nabla E_2(\boldsymbol\xi^{t+1})\big]^\top}_{=\;\mathbf{0}\ \text{by the CCCP condition}}(\boldsymbol\xi^{t+1}-\boldsymbol\xi^t) \;=\; 0 \qquad\blacksquare$$

**Step 4 — convergence.** Monotone decreasing + bounded below ($E\ge0$) + iterates confined to the convex hull of the patterns (softmax weights sum to 1, so $\|\boldsymbol\xi^t\|\le M$) ⇒ global convergence to a stationary point. $\blacksquare$

> **Crucially, this proof does not require asynchronous updates.** The whole vector updates at once and the guarantee survives — because it comes from the convex/concave split, not from freezing coordinates. This is what makes §13's $O(1)$ claim possible.

### 6.4 The three theorems

**(A) Exponential storage capacity.** For random patterns on a sphere of radius $M$ in dimension $d$:

$$C \;\gtrsim\; \sqrt{p}\;c^{\frac{d-1}{4}}, \qquad c > 1$$

Exponential in $d$, versus $0.14d$ for classical. A jump across complexity classes.

**(B) One-step retrieval with exponentially small error.** Define the **separation**

$$\Delta_i = \min_{j\neq i}\big(\mathbf{x}_i^\top\mathbf{x}_i - \mathbf{x}_i^\top\mathbf{x}_j\big)$$

If $\Delta_i$ is large enough, a fixed point $\mathbf{x}_i^*$ exists near $\mathbf{x}_i$, and **a single update** achieves

$$\|\boldsymbol\xi^{\text{new}} - \mathbf{x}_i\| \le 2\varepsilon M, \qquad \varepsilon \sim Ne^{-\beta\Delta_i}$$

Error decays exponentially in $\beta\Delta_i$. In practice: retrieval completes in one step.

**(C) Three regimes of fixed points.**

| Regime                 | Occurs when                          | $\boldsymbol\xi^*$ becomes  | Used for              |
| ---------------------- | ------------------------------------ | --------------------------- | --------------------- |
| **Global fixed point** | small $\beta$, patterns similar      | average of **all** patterns | full-set pooling      |
| **Metastable state**   | moderate $\beta$, patterns clustered | average of a **subgroup**   | averaging, clustering |
| **Single-pattern**     | large $\beta$, high separation       | **one** pattern             | associative recall    |

**$\beta$ is the dial that selects the regime.** Low $\beta$ = flat softmax = average everything. High $\beta$ = peaked softmax = pick one.

**Memorize this table.** §11 turns it into the central design decision, and it's the reason the theory has practical value at all.

---

## §7. It is literally attention

Rewrite the update for many queries at once (row-vector convention):

$$\boldsymbol\Xi^{\text{new}} = \operatorname{softmax}\big(\beta\,\boldsymbol\Xi\mathbf{X}^\top\big)\mathbf{X}$$

| Hopfield                                | Transformer    |
| --------------------------------------- | -------------- |
| state patterns $\boldsymbol\Xi$         | $\mathbf{Q}$   |
| stored patterns $\mathbf{X}$            | $\mathbf{K}$   |
| $\mathbf{X}$ multiplied back at the end | $\mathbf{V}$   |
| $\beta$                                 | $1/\sqrt{d_k}$ |

$$\operatorname{softmax}\\left(\tfrac{1}{\sqrt{d_k}}\mathbf{Q}\mathbf{K}^\top\right)\mathbf{V}$$

$$\boxed{\text{Transformer attention} \;=\; \text{one step of Hopfield retrieval} \;+\; \text{projection matrices}}$$

**Empirical payoff:** the paper analyzes BERT's attention heads and finds them spread across all three regimes — lower layers skew global/metastable (averaging), upper layers skew single-pattern (specific recall). The theory becomes a diagnostic for _what each head is doing_.

### The consequence people miss

In attention, $\mathbf{K}$ and $\mathbf{V}$ are **not persistent memory.** They're computed on the fly from the current input.

> **Every new input sequence builds a brand-new energy landscape. The wells sit on that sequence's tokens. When the sequence ends, the landscape is gone.**

What gets trained is $\mathbf{W}_Q,\mathbf{W}_K,\mathbf{W}_V$ — not the wells, but **the well-digging procedure.**

$$\text{A Transformer doesn't learn } \textit{what to remember.}\text{ It learns } \textbf{how to build a landscape when new data arrives.}$$

---

# Part V — Where the intelligence actually is

_This is the part the paper's abstract doesn't tell you and most summaries skip._

## §8. The three Hopfield layers

The paper ships a PyTorch package (`hflayers`) with three layer types, distinguished by **which of Q/K/V come from data and which are learned parameters.**

### 1. `Hopfield` — the general case

- Q from one input, K/V from another
- = ordinary self- or cross-attention (with tunable $\beta$ and iteration count)
- Use: seq2seq, associating two sets

### 2. `HopfieldPooling` — the query is a parameter

- $\mathbf{Q}$ = learned prototype vector(s), static; $\mathbf{K},\mathbf{V}$ from data
- Collapses a variable-size set into one vector
- Use: **Multiple Instance Learning**, set representation
- **This is the one that wins on real problems**

### 3. `HopfieldLayer` — the stored patterns are parameters

- $\mathbf{K},\mathbf{V}$ = learned weights (a memory bank); $\mathbf{Q}$ from data
- A trainable memory that can replace an FC layer or LSTM state
- Use: small datasets, tabular data, drug design

## §9. Why you can't just use a linear layer — the real argument

Take the DeepRC problem, because it makes every constraint bite:

> One patient = **~300,000 T-cell receptor sequences.**
> About **10** of them indicate CMV infection.
> Available label: **one bit per patient** ("infected / not").

### Option 1: CNN + linear layer on everything

$300{,}000 \times 32 = 9{,}600{,}000$ input dimensions per patient.

1. **Variable size** — the next patient has 250,000 sequences. The linear layer breaks.
2. **No order** — swapping sequence #5 with #5000 must not change the answer. A linear layer gives a different answer.
3. **9.6M parameters against 1 bit of label** — instant overfit.

**You have no choice but to pool first.** The real question is: pool _how_?

### Option 2: mean pooling

$$\bar{\mathbf{z}} = \frac{1}{N}\sum_i\mathbf{z}_i$$

The obvious problem is signal dilution (10 / 300,000 = 0.003%). **The deeper problem is the gradient:**

$$\frac{\partial L}{\partial\mathbf{z}_i} = \frac{1}{N}\frac{\partial L}{\partial\bar{\mathbf{z}}} \qquad\text{— identical for every } i$$

A disease-marker sequence and a junk sequence receive **exactly the same gradient.**

⇒ The CNN receives no signal about which instances mattered.
⇒ **It can never learn to distinguish them. Ever.**

This is not a tuning problem. It is structural. No amount of training fixes it.

### Option 3: Hopfield pooling

$$\text{out} = \sum_i a_i\mathbf{z}_i \quad\Longrightarrow\quad \frac{\partial L}{\partial\mathbf{z}_i} \;\propto\; a_i\cdot\frac{\partial L}{\partial\text{out}} \;+\;(\text{term via } a_i)$$

**The gradient is weighted by attention.** Selected instances get large gradients; the rest get roughly zero.

|                                                     | mean pooling   | Hopfield pooling |
| --------------------------------------------------- | -------------- | ---------------- |
| gradient per instance                               | $1/N$, uniform | $\propto a_i$    |
| **can the upstream encoder learn to discriminate?** | **no**         | **yes**          |
| differentiable                                      | yes            | yes              |
| permutation invariant                               | yes            | yes              |
| content-based                                       | **no**         | **yes**          |
| handles variable $N$                                | yes            | yes              |

### The mechanism, stated plainly

**Hopfield doesn't learn what to store. It makes the CNN learn instead.** It is a _gradient router._

The feedback loop:

1. $\boldsymbol\xi$ selects some instances (initially near-arbitrary)
2. Those get gradient; the rest get ~zero
3. **The CNN updates only with respect to those**
4. If the selection was useful, loss drops → the CNN is pushed to encode those instances more distinctly, and $\boldsymbol\xi$ is pushed toward them
5. Repeat — **the signal amplifies itself**

$$\text{The wells don't learn what to hold} \;\longrightarrow\; \textbf{the encoder learns to dig wells where } \boldsymbol\xi \textbf{ will find them}$$

> A Hopfield layer is a **differentiable hash table.** The table itself isn't smart. But because gradients flow through it, it teaches the thing above it how to construct keys that make lookup succeed.

## §10. But how does ξ know which instances actually matter?

The sharpest version of the question. The answer is one derivative.

Let $s_i = \beta\boldsymbol\xi^\top\mathbf{z}_i$, $a = \operatorname{softmax}(s)$, $\text{out} = \sum_i a_i\mathbf{z}_i$, $\mathbf{g} = \partial L/\partial\text{out}$.

From the softmax Jacobian, $\dfrac{\partial\,\text{out}}{\partial s_j} = a_j(\mathbf{z}_j - \text{out})$, so

$$\boxed{\;\frac{\partial L}{\partial\boldsymbol\xi} = \beta\sum_j a_j\,\underbrace{\langle\mathbf{g},\,\mathbf{z}_j-\text{out}\rangle}_{\displaystyle u_j}\,\mathbf{z}_j\;}$$

**Read $u_j$ as a question:** _"if we had weighted $\mathbf{z}_j$ above average, would the loss have improved?"_

|                                  | effect                                                                            |
| -------------------------------- | --------------------------------------------------------------------------------- |
| $u_j < 0$ (this instance helped) | update pushes $\boldsymbol\xi$ **toward** $\mathbf{z}_j$ → next round $a_j$ rises |
| $u_j > 0$ (this instance hurt)   | pushes $\boldsymbol\xi$ **away** → next round $a_j$ falls                         |

This is a **"move toward what demonstrably helped"** rule that _emerges from backprop._ Nobody designed it. It is a consequence of the softmax Jacobian.

### What happens at initialization?

At init all $a_j \approx 1/N$, giving

$$\frac{\partial L}{\partial\boldsymbol\xi} \approx \frac{\beta}{N}\sum_j\langle\mathbf{g},\mathbf{z}_j-\bar{\mathbf{z}}\rangle\,\mathbf{z}_j \;=\; \frac{\beta}{N}\mathbf{C}\,\mathbf{g}, \qquad \mathbf{C}\approx N\cdot\operatorname{Cov}(\mathbf{Z})$$

**It does not move randomly.** It moves along the principal variance directions of the bag, projected onto the loss-reducing direction.

Hopfield pooling _starts out_ behaving like mean pooling — but it has a **second-order handle** mean pooling structurally lacks: a parameter that perceives _differences between instances._

### ⭐ Where the information actually comes from

**Not from one patient. From comparison across patients.**

Accumulate $\partial L/\partial\boldsymbol\xi$ over a minibatch:

| Instance type        | In CMV+ bags | In CMV− bags | Sign of $u_j$             |
| -------------------- | ------------ | ------------ | ------------------------- |
| The ~10 real markers | present      | absent       | **consistent every time** |
| The 299,990 junk     | present      | present      | random ±                  |

$$\sum_{\text{patients}} u_j \;\longrightarrow\; \begin{cases}\text{accumulates} & \text{if correlated with the label}\\[4pt]\text{cancels to zero} & \text{if it's noise}\end{cases}$$

**Signal accumulates; noise cancels.** That is the entire information source. No magic.

Within a single patient it genuinely cannot tell — that intuition is correct. Across 500 patients with differing labels, directions that co-occur with the label rise above the noise floor.

### Variable length comes free

$$\sum_i a_i = 1 \quad\text{always, regardless of } N$$

The output always lies in the convex hull of $\{\mathbf{z}_i\}$ ⇒ scale never explodes, order doesn't matter, $N$ can change between examples. All of it falls out of the shape of softmax — no special machinery.

$N$ affects exactly one thing: larger $N$ ⇒ flatter initial softmax ⇒ you need higher $\beta$ for the feedback loop to ignite.

### Contrast with CNN pooling

$$\text{CNN pooling: prior} = \textbf{spatial proximity} \;\;(2\times2\text{ window})$$
$$\text{Hopfield pooling: prior} = \textbf{similarity to a learnable template}$$

Convolution asserts _"what matters is located near what matters"_ — true for images, meaningless for an unordered bag. Hopfield asserts _"what matters is what resembles $\boldsymbol\xi$"_ and lets gradient decide what $\boldsymbol\xi$ should be.

**Both are inductive biases, not intelligence.** Just different ones, suited to different data shapes.

### The failure mode is real

If the signal is too faint or the dataset too small, the loop **never ignites** — $\boldsymbol\xi$ stays parked at the mean forever. This is a known limitation of attention-based MIL, not a hypothetical. Mitigations: multiple heads, $\beta$ tuning/annealing, enough data.

$$\text{Hopfield doesn't } \textit{know}\text{ which instances are relevant — it makes relevance }\textbf{findable by gradient descent.}$$

## §10.5. Why sparse signal actually survives — and the amplification trap

> **This section corrects §10.** The claim there — _"signal accumulates across examples, noise cancels"_ — is true but **not specific to Hopfield.** It is the operating principle of every SGD-trained model. Stated alone, it explains nothing. Here is what's actually different.

### The real problem with mean pooling is expressivity, not gradient magnitude

"Signal accumulates" only works if there is a **parameter capable of encoding the thing being learned.**

With mean pooling, the classifier sees only $\bar{\mathbf{z}}$. The hypothesis _"attend only to instances resembling $\mathbf{s}$"_ has **no parameter that can represent it.** The issue isn't a weak gradient — it's that the function class cannot express the hypothesis at all.

$$\bar{\mathbf{z}} \text{ is a sufficient statistic that has already destroyed instance-level information}$$

More data does not help. This is the same category of failure as a linear model on XOR: not a sample-size problem, a representability problem.

**$\boldsymbol\xi$ is precisely the parameter that makes "selection" expressible.** Only then does the ordinary accumulate-and-cancel dynamic have something to operate on.

$$\text{The difference isn't the principle — it's whether the principle has anything to act on}$$

### How large the gap is

Let a bag hold $N$ instances, $k$ of them carrying signal $\mathbf{s}$, the rest noise $\sim\mathcal{N}(0,\sigma^2)$.

**Mean pooling:**
$$\bar{\mathbf{z}} = \frac{k}{N}\mathbf{s} + \frac{1}{N}\sum\mathbf{n}_j \quad\Longrightarrow\quad \text{SNR} = \frac{(k/N)|\mathbf{s}|}{\sigma/\sqrt N} = \frac{k\,|\mathbf{s}|}{\sigma\sqrt N}$$

**Attention peaked on the right instances:**
$$\text{out} \approx \mathbf{s} + \frac{1}{k}\sum_{\text{signal}}\mathbf{n} \quad\Longrightarrow\quad \text{SNR} = \frac{\sqrt k\,|\mathbf{s}|}{\sigma}$$

$$\boxed{\;\frac{\text{SNR}_{\text{Hopfield}}}{\text{SNR}_{\text{mean}}} = \sqrt{N/k}\;}$$

At $N=300{,}000$, $k=10$: a factor of **173**.

Since required sample size scales as $1/\text{SNR}^2$:

$$\text{mean pooling would need } \frac{N}{k} = \textbf{30,000}\times \text{ more patients}$$

500 patients becomes 15 million. **Not theoretically impossible — practically impossible.** That distinction is the whole argument.

### ⭐ The genuinely non-generic part: its SNR is not constant

An ordinary layer has a gradient SNR **fixed by the data.** You average over batches and converge.

Hopfield does not:

$$\boldsymbol\xi \text{ improves} \;\Longrightarrow\; \mathbf{a} \text{ sharpens on signal} \;\Longrightarrow\; \text{next round's gradient SNR} \uparrow \;\Longrightarrow\; \boldsymbol\xi \text{ improves more}$$

$$\text{SNR}_{t+1} > \text{SNR}_t$$

**It amplifies its own signal-to-noise ratio.** As $\beta\boldsymbol\xi^\top\mathbf{z}$ climbs from flat toward peaked, SNR compounds rather than merely accumulating linearly.

The consequence is that training is **bistable**, not gradual:

$$\text{SNR}_0 > \text{threshold} \;\Rightarrow\; \text{ignites, then takes off}$$
$$\text{SNR}_0 < \text{threshold} \;\Rightarrow\; \text{stuck at the mean forever}$$

This is why attention-based MIL typically either sits flat for many epochs and then jumps, or never moves at all — it's a **phase transition**, not standard convergence. If your MIL model is flat, adding epochs is usually the wrong response; raising $\beta$, adding heads, or improving encoder initialization is the right one.

### The amplification trap

The feedback loop **cannot tell whether the signal it is amplifying is real.**

$$\text{spurious correlation in the dataset} \;\Longrightarrow\; \text{the loop amplifies it with equal enthusiasm}$$

Documented failure in computational pathology: attention-MIL latching onto **scanner model** or **staining batch** rather than tumor morphology — because hospitals sending more cancer cases happened to share equipment.

> **It is confirmation bias written into the architecture:** attend to what has seemed useful, and thereby see more of it.

This means Hopfield pooling is **more** sensitive to violations of the iid assumption than a plain averaging layer, not less. Correlated or batch-confounded data doesn't merely slow it down — it gets actively magnified.

### Summary of what Hopfield actually contributes

| Claim                                            | Verdict                                                     |
| ------------------------------------------------ | ----------------------------------------------------------- |
| "Signal accumulates, noise cancels"              | Generic to all SGD. Explains nothing on its own             |
| **A parameter that makes selection expressible** | ✅ The real reason it works                                 |
| **A loop that amplifies its own SNR**            | ✅ Real, and the source of both its power and its fragility |

**The first is why it works. The second is why it sometimes doesn't work at all.**

> **The self-amplifying loop is a recurring shape, not a Hopfield quirk.** The same structure — _no partial credit → flat gradient → sudden jump_ — shows up in induction-head formation and in grokking. See the companion file `3-transformer-internals.md` §3 for the full family.

---

## §11. Full worked example: DeepRC end to end

```
300,000 receptor sequences        ← variable-size, unordered set
        │
        ▼
1D CNN encoder                    ← handles within-sequence structure
        │                            "CASSLGQAYEQYF" → z_i ∈ R^32
        ▼
X = {z_1, ..., z_300000}          ← now just a bag of vectors
        │                            HOPFIELD NEVER SEES A SEQUENCE
        ▼
HopfieldPooling                   ← Q = learned ξ, K = V = X
        │                            a = softmax(β oᵀX);  out = Σ aᵢzᵢ
        ├──────────────► attention weights aᵢ   ← interpretable output
        ▼
Linear head
        │
        ▼
P(CMV+)  ──► BCE loss ──► backprop reaches BOTH the CNN and ξ
```

**Three things to take from this diagram:**

### 1. The sequence problem is solved _before_ Hopfield

The CNN converts each sequence to a vector. By the time Hopfield sees the data, it's an orderless bag of 32-dim vectors. Clean division of labor:

$$\text{CNN: } \textit{within-item structure} \qquad\qquad \text{Hopfield: } \textit{which items matter}$$

Hopfield is not bad at sequences. It simply never touches them.

### 2. The output is readable, exactly like attention weights

$$\mathbf{a} = \operatorname{softmax}(\beta\boldsymbol\xi^\top\mathbf{X}) \in \mathbb{R}^{300000}, \qquad \sum_i a_i = 1$$

Sort by $a_i$, take the top 10. DeepRC did this and found the top-attention sequences match **CMV-associated motifs already known to immunologists** — evidence the model found real biology rather than a shortcut.

This directly answers "how would we ever use the output": the same way you use attention weights in a Transformer, because it is the same quantity.

### 3. The metastable regime is the point

Recall §6.4's table:

| $\beta$      | Behavior                | For MIL                      |
| ------------ | ----------------------- | ---------------------------- |
| low          | averages all 300,000    | ❌ signal drowns             |
| **moderate** | **averages a subgroup** | ✅ **exactly what you want** |
| high         | picks a single instance | ❌ discards the other 9      |

If the 10 target sequences share a motif, the encoder maps them to a **tight cluster** — their wells merge into one large well, and the query lands in that metastable state, returning their average.

> **What is a bug for associative memory (merged wells) is the feature for MIL.**

And it's measurable: the entropy $H(\mathbf{a})$ tells you which regime you're in.

| $H(\mathbf{a})$        | diagnosis                                 |
| ---------------------- | ----------------------------------------- |
| high (near $\log N$)   | drowning — spread across everything       |
| effective support ≈ 10 | **correctly tuned**                       |
| near zero              | over-peaked — collapsed onto one instance |

**The theory tells you which way to turn the dial and how to know you got it right**, instead of guessing. This is the concrete cash value of §6.4.

---

# Part VI — Follow-on confusions, resolved

## §12. "How does this work for NLP if it doesn't know about sequences?"

**Surprise: attention doesn't know about sequences either.**

Let $\mathbf{P}$ be a permutation matrix. Then $\mathbf{X}\to\mathbf{X}\mathbf{P}$ gives $\mathbf{X}^\top\boldsymbol\xi \to \mathbf{P}^\top\mathbf{X}^\top\boldsymbol\xi$. Softmax is elementwise, so $\operatorname{softmax}(\mathbf{P}^\top\mathbf{z}) = \mathbf{P}^\top\operatorname{softmax}(\mathbf{z})$, and

$$\text{out} \to \mathbf{X}\mathbf{P}\mathbf{P}^\top\operatorname{softmax}(\beta\mathbf{X}^\top\boldsymbol\xi) = \mathbf{X}\operatorname{softmax}(\beta\mathbf{X}^\top\boldsymbol\xi) = \text{out}$$

since $\mathbf{P}\mathbf{P}^\top = \mathbf{I}$. **Unchanged.**

Without help, a Transformer sees _"cat eats fish"_ = _"fish eats cat."_ Self-attention is **permutation-equivariant by construction.** This isn't a Hopfield weakness — it's a property of the attention you already use every day.

### How Transformers fix it

$$\mathbf{z}_i = \underbrace{\text{embed}(\text{token}_i)}_{\text{content}} + \underbrace{\mathbf{p}_i}_{\text{position}}$$

**Order is injected into the pattern content, not into the attention mechanism.**

In Hopfield language:

$$\text{The well isn't at "cat" — it's at }\textbf{"cat-at-position-3."}$$

Same word at a different position = different pattern = different well. Retrieval separates them naturally.

**This is structurally identical to the DeepRC CNN encoder.** Positional encoding and the 1D CNN play the same architectural role: _fold structure into the vector before the set-machinery touches it._ Once you see this, both examples become one pattern.

## §13. "Doesn't more data / higher dimension mean more steps? Why is retrieval $O(1)$?"

Two things are being conflated:

$$\underbrace{\text{number of iterations}}_{O(1)} \quad\neq\quad \underbrace{\text{cost per iteration}}_{O(N\cdot d)}$$

$O(1)$ refers to the fixed-point loop count, not to compute being free. And pooling was never iterative — $\sum_i a_i\mathbf{z}_i$ is one sum regardless of $N$.

### Why classical needed many steps

Asynchronous updates touch **one neuron at a time**, so covering all $d$ neurons takes at least $d$ steps, and settling takes several sweeps. This was **mandatory**, not a choice — §1.2's descent proof fails under simultaneous updates.

Continuous modern Hopfield updates the whole vector in one matmul, and **CCCP still guarantees descent** (§6.3), because the guarantee comes from the convex/concave split rather than from freezing coordinates.

$$O(d)\text{ steps} \;\longrightarrow\; 1\text{ step, with the guarantee intact}$$

### Why one step suffices mathematically

$$\|\boldsymbol\xi^{\text{new}}-\mathbf{x}_i\| \le 2\varepsilon M, \qquad \varepsilon \approx Ne^{-\beta\Delta_i}$$

The key fact: **$\Delta_i$ grows with $d$.** For random patterns on a sphere,

$$\langle\mathbf{x}_i,\mathbf{x}_i\rangle = M^2\sim d, \qquad \langle\mathbf{x}_i,\mathbf{x}_j\rangle \sim \frac{M^2}{\sqrt d} \;\Longrightarrow\; \Delta_i \approx M^2\Big(1-\tfrac{1}{\sqrt d}\Big) \sim d$$

$$\Longrightarrow \varepsilon \sim Ne^{-\beta d}$$

**$d$ sits in the exponent; $N$ sits out front linearly.** The exponential always wins.

### ⭐ The counterintuitive part: higher dimension makes it _easier_

$$d\uparrow \;\Longrightarrow\; \Delta\uparrow \;\Longrightarrow\; \varepsilon = Ne^{-\beta\Delta}\downarrow\ \text{exponentially}$$

In high dimensions random vectors are nearly orthogonal (concentration of measure) → patterns self-separate → softmax sharpens on its own → retrieval gets easier.

$$\textbf{Blessing of dimensionality, not curse.}$$

Contrast with classical, where crosstalk variance is $N/d$ and you must roll downhill repeatedly to grind noise away. Modern Hopfield uses the exponential to annihilate noise in a single step.

### What you actually pay

Not steps — **$\beta$.** Larger $N$ ⇒ flatter softmax ⇒ push $\beta$ up:

$$\varepsilon \le \delta \;\Longrightarrow\; \beta \;\gtrsim\; \frac{\log N + \log(1/\delta)}{\Delta} \qquad\Longrightarrow\qquad \boxed{\beta \sim \log N}$$

**A bigger bag costs you logarithmically in $\beta$, not linearly in iterations.** Compute stays $O(Nd)$ per query — precisely the source of $O(L^2)$ cost in Transformers.

### Then why is GPT 96 layers deep?

Because **each layer does not retrieve from the same landscape:**

$$\text{layer }\ell:\quad \mathbf{X}^{(\ell)} \xrightarrow{\ \text{attention} + \text{FFN}\ } \mathbf{X}^{(\ell+1)}$$

|                                 | what it means                                             |
| ------------------------------- | --------------------------------------------------------- |
| **iterating a Hopfield update** | rolling further down the _same_ well — one step is enough |
| **the next Transformer layer**  | **building a brand-new landscape**, then retrieving again |

$$\text{Depth} \;=\; \text{repeated retrieval from a landscape re-sculpted at every layer}$$

This is why "one step suffices" doesn't contradict deep models working better. They aren't doing the same thing.

### Caveat on the $O(1)$ claim

One-step retrieval holds **when separation is large.** If patterns cluster (metastable regime), one step returns the _cluster average_, and iterating **does not** separate them — it converges to that cluster's centroid.

For MIL that's the desired outcome. But if you genuinely want single-pattern retrieval and hit this case: **more iterations won't fix it. Fix $\beta$, or fix the encoder.**

---

# Part VII — Summing up

## §13.5. What happens when retrieval reaches a fixed point?

Natural question: if a Transformer's state lands at the bottom of a well where the gradient is zero, is it stuck? Doesn't the next layer just build a new landscape from it?

**Two cases, opposite answers.** The difference between them is the whole story.

### Case A: one query converges → normal, and often desirable

$$\boldsymbol\xi^* = \mathbf{X}\operatorname{softmax}(\beta\mathbf{X}^\top\boldsymbol\xi^*)$$

The intuition is correct here: the next layer has **different** $\mathbf{W}_Q,\mathbf{W}_K,\mathbf{W}_V$, so it implies a completely different landscape. What was a valley floor may be a hillside one layer up.

This isn't a problem — it's **successful retrieval**, and it has a name: an **induction head** driving $\mathbf{a}$ toward one-hot in order to _copy_ a token. That's the high-$\beta$ single-pattern regime of §6.4 working exactly as intended.

### Case B: every token converges to the same point → permanent death

Here the "next layer rebuilds it" intuition **fails**.

Suppose $\mathbf{z}_i = \mathbf{z}$ for all $i$. Ask what the next layer can possibly do:

$$\mathbf{k}_i = \mathbf{W}_K\mathbf{z} \;\;\text{identical} \;\Longrightarrow\; s_{ji} \;\text{identical} \;\Longrightarrow\; \operatorname{softmax}(s_j) = \tfrac1N\mathbf{1}$$
$$\Longrightarrow \text{out}_j = \tfrac1N\sum_i\mathbf{W}_V\mathbf{z} = \mathbf{W}_V\mathbf{z} \qquad\text{— still identical}$$

Check the rest of the block:

| Component | Escapes?                                                                                        |
| --------- | ----------------------------------------------------------------------------------------------- |
| FFN       | ❌ acts per-token; identical in → identical out                                                 |
| LayerNorm | ❌ identical                                                                                    |
| Residual  | ❌ $\mathbf{z} + \mathbf{W}_V\mathbf{z} = (\mathbf{I}+\mathbf{W}_V)\mathbf{z}$, still identical |

$$\boxed{\text{Token uniformity is an } \textbf{absorbing state} \text{ — once entered, unreachable in reverse}}$$

**The algebraic reason:**

$$\text{Attn}(\mathbf{X}) = \mathbf{W}_V\mathbf{X}\mathbf{A} \quad\Longrightarrow\quad \operatorname{rank}(\text{Attn}(\mathbf{X})) \le \operatorname{rank}(\mathbf{X})$$

**Attention can never increase rank.** Both the projection and the convex combination are linear/affine operations.

⇒ The next layer _does_ build a new landscape — **but it builds it out of $\mathbf{X}$.** If $\mathbf{X}$ has collapsed to rank 1, every landscape derivable from it is degenerate too.

$$\text{A new landscape} \neq \text{a good landscape — it is built from the wreckage of the old one}$$

Attention operates on **differences between tokens.** Remove the differences and it has nothing to work with.

### What actually prevents Case B — and why residual isn't what you think

Residual does **not** restore rank (proved above — it can't). It does something else:

$$\mathbf{x}^{(\ell+1)} = \mathbf{x}^{(\ell)} + \text{Attn}(\mathbf{x}^{(\ell)})$$

Compare with a true Hopfield step:

$$\boldsymbol\xi^{\text{new}} = \mathbf{X}\operatorname{softmax}(\beta\mathbf{X}^\top\boldsymbol\xi) \qquad\text{— discards the old } \boldsymbol\xi \text{ entirely}$$

$$\boxed{\text{A Transformer never actually } \textit{completes} \text{ a Hopfield step}}$$

It _adds_ the retrieval result to the current position rather than _replacing_ it.

$$\text{Hopfield: } \textbf{jump into the convex hull} \qquad \text{Transformer: } \textbf{move partway and keep the original}$$

So it isn't fixed-point iteration at all — it's **direction accumulation**, closer to explicit Euler integration on $-\nabla E$.

$$\text{Residual is a } \textbf{rank floor}, \text{ not a rank restorer — it prevents collapse, it cannot cure it}$$

This is why removing residuals doesn't merely slow deep Transformers down — they become untrainable.

### Documented manifestations

| Phenomenon                           | What it is                                                  |
| ------------------------------------ | ----------------------------------------------------------- |
| **Rank collapse** (Dong et al. 2021) | pure attention → rank → 1, _doubly exponentially_ in depth  |
| **Oversmoothing**                    | deep Transformers whose token representations converge      |
| **Attention entropy collapse**       | $H(\mathbf{a})$ collapsing mid-training → training diverges |
| **Attention sink**                   | LLMs dumping attention mass on the BOS token                |

The last one is the most revealing. A head with nothing useful to retrieve learns to dump its weight onto a semantically empty token so its output is ≈ 0 and the residual carries the input through nearly untouched.

$$\text{Attention sink} = \textbf{a "do nothing" button the model invents for itself}$$

Direct evidence that models _need_ an escape from collapse, and find one unprompted.

---

## §13.6. The convergence inversion — the deepest difference

|                      | Hopfield      | Transformer               |
| -------------------- | ------------- | ------------------------- |
| Number of landscapes | **one**       | **one per layer**         |
| Steps per landscape  | until settled | **exactly one**           |
| A fixed point is     | **the goal**  | **death (rank collapse)** |
| Success means        | converging    | **not converging**        |

$$\text{A Transformer is a Hopfield network deliberately prevented from converging, at every layer}$$

Residual, FFN, and LayerNorm all serve one shared function: **kick the state out of the convex hull before it can collapse.**

### But the FFN is also a Hopfield network

Geva et al. (2021) showed the FFN is a key-value memory:

$$\text{FFN}(\mathbf{x}) = \mathbf{W}_2\,\sigma(\mathbf{W}_1\mathbf{x}) \quad\longleftrightarrow\quad \mathbf{W}_1 = \mathbf{K},\; \mathbf{W}_2 = \mathbf{V}$$

Compare `HopfieldLayer` from §8 — **K and V as learned parameters.** Same definition.

⇒ **One Transformer block = two retrievals:**

$$\underbrace{\text{Attention}}_{\text{Hopfield, patterns from context}} \;\to\; \underbrace{\text{FFN}}_{\text{Hopfield, patterns are permanent weights}}$$

$$\boxed{\text{"Retrieve from what you just read, then retrieve from what you have always known"}}$$

This is the sharpest available answer to _what a Transformer block does_ — and it's an answer the Hopfield framing gives you that the pipeline framing does not.

### What lies outside the correspondence

| Component           | Hopfield analogue?                       |
| ------------------- | ---------------------------------------- |
| Attention           | ✅ **is** Hopfield                       |
| FFN                 | ✅ **is** static Hopfield                |
| Residual            | ❌ none — and it's the primary safeguard |
| LayerNorm           | ❌ none                                  |
| Causal mask         | ❌ none                                  |
| Multi-head          | ❌ none — Hopfield has a single query    |
| Positional encoding | ❌ folded into patterns (§12)            |

**Hopfield explains roughly half a block — and the half it cannot explain is the half that makes depth usable.**

### Consequence for long context

$N$ = context length = number of stored patterns. From §13:

$$\beta \sim \log N$$

Longer context ⇒ flatter softmax ⇒ you need higher $\beta$ to stay sharp. But $1/\sqrt{d_k}$ is **fixed** and does not scale with $N$.

⇒ **attention dilution.** This is a theoretical account of "lost in the middle" behaviour in long-context models, and of why RoPE scaling and attention-temperature tuning are structural necessities rather than tricks.

---

## §14. The whole evolution in one table

|                       | Classical (1982)                  | Dense AM (2016) | Demircigil (2017) | **Ramsauer (2020)**                                                    |
| --------------------- | --------------------------------- | --------------- | ----------------- | ---------------------------------------------------------------------- |
| **State**             | binary                            | binary          | binary            | **continuous**                                                         |
| **$F(z)$**            | $z^2$                             | $z^n$           | $e^z$             | $e^z$ via lse                                                          |
| **Capacity**          | $0.14d$                           | $O(d^{n-1})$    | $O(2^{d/2})$      | **exponential in $d$**                                                 |
| **Steps to retrieve** | many, async                       | many            | 1                 | **1, synchronous**                                                     |
| **Update rule**       | $\operatorname{sgn}(\mathbf{Wx})$ | energy diff     | energy diff       | $\mathbf{X}\operatorname{softmax}(\beta\mathbf{X}^\top\boldsymbol\xi)$ |
| **Trainable?**        | no                                | no              | no                | **yes — via $\mathbf{X}$, $\boldsymbol\xi$, $\beta$**                  |
| **Equals**            | —                                 | —               | —                 | **Transformer attention**                                              |

**The storyline in one sentence:** make the interaction function steeper → crosstalk vanishes → capacity explodes; make the state continuous and add $\tfrac12\|\boldsymbol\xi\|^2$ to prevent blow-up → derive via CCCP → softmax attention falls out for free.

## §15. What are you actually touching?

|                        | What you hold      | What you tune                     |
| ---------------------- | ------------------ | --------------------------------- |
| **Transformer**        | a pipeline         | weights inside the pipe           |
| **Classical Hopfield** | an energy function | nothing — you just add data in    |
| **Hopfield layer**     | an energy function | **the thing that digs the wells** |

Hopfield raises the level of abstraction: from _"tune a function"_ to _"tune the objective the system flows toward."_

## §16. In one line

$$\text{Hopfield pooling} \;=\; \textbf{a soft, content-based, learnable max-pool}$$

- Not `max` — that discards the other 9 and has a broken gradient
- Not `mean` — that drowns everything and kills the gradient
- **The tunable middle**, controlled by $\beta$ and learned through $\boldsymbol\xi$

What Hopfield adds beyond just saying "attention pooling" is **theory**: it tells you when you'll get the full-set average, when a subgroup, when a single item, and how many patterns you can hold before things start blending.

## §16.5 Mental model check — three tempting but wrong pictures

### ❌ "The landscape is a store you write into"

There is no Hopfield _object_ to write to. Look at the energy:

$$E(\boldsymbol\xi) = -\operatorname{lse}(\beta,\mathbf{X}^\top\boldsymbol\xi) + \tfrac12\|\boldsymbol\xi\|^2$$

It contains only $\mathbf{X}$, $\boldsymbol\xi$, $\beta$. **Nothing else exists.** The landscape is a _function_ of those three, not a separate structure that persists.

$$\text{You never write to the landscape — you change what it is computed from}$$

The correct loop:

$$N\text{ inputs} \to \text{encoder} \to \mathbf{X} \;\Longrightarrow\; \text{a landscape is implied, right now}$$
$$\to \text{one retrieval step} \to \text{loss} \to \text{gradient adjusts } \boldsymbol\xi \text{ and the encoder}$$
$$\to \text{landscape dissolves} \to \text{next pass implies a slightly different one}$$

**A landscape lives exactly one forward pass.**

### ❌ "Hopfield is a model you train"

It's a _layer_, and by itself it has no parameters worth speaking of. Compare it to an attention block, not to a Transformer. (§8)

### ❌ "It knows which instances are relevant"

It knows nothing. It provides a parameter that makes relevance **expressible**, so that gradient descent can find it. (§10.5)

### ✅ The one-sentence version that is correct

> **Hopfield's real function is differentiable signal amplification** — it magnifies a sparse subset of the input in a way that backprop can still flow through cleanly, so the encoder upstream learns which instances to make findable.

---

## §17. Honest assessment

Because this deserves not to be oversold.

**On the theory:**

1. **"Explains" vs. "equals."** Attention having the same _form_ as a Hopfield update doesn't prove Transformers _operate on_ associative-memory principles. Some read it as a beautiful mathematical coincidence with limited predictive power.
2. **The capacity bound is loose.** It's a worst-case result for random patterns on a sphere — nothing like real embeddings, which are structured and clustered.
3. **Limited practical impact.** Hopfield layers do well on MIL and small datasets, but nobody changed how large Transformers are built.

**On importance, by domain:**

| Framing                         | Verdict                                                                                                                                 |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| As an NLP architecture          | **Unimportant.** Nobody uses it. Transformers existed since 2017; this paper explains them retroactively rather than inventing anything |
| As a layer for set/MIL problems | **Genuinely useful.** Real deployment in pathology, immunology, point clouds                                                            |
| As theory                       | **Important.** Gives us vocabulary for what attention does, beyond "it just works"                                                      |

$$\text{Hopfield 1982} \;\gg\; \text{Hopfield 2020}$$

The 2024 Nobel Prize in Physics honored the **1982** work (associative memory, spin glasses) — not this paper. The 2020 paper is good, well-cited _connective_ work, not field-changing work.

**If your instinct was that this doesn't revolutionize anything — that instinct is correct.**

> Hopfield isn't a model you take to a benchmark leaderboard.
> It's a **framework** that explains why the model you already use works.

**Related work worth knowing:** Universal Hopfield Networks (Millidge et al., 2022) generalizes all associative memory models as `separation ∘ similarity ∘ projection`.

---

## §18. Quick reference

$$E(\boldsymbol\xi) = -\operatorname{lse}(\beta,\mathbf{X}^\top\boldsymbol\xi) + \tfrac12\|\boldsymbol\xi\|^2 + \text{const}$$

$$\boldsymbol\xi^{\text{new}} = \mathbf{X}\operatorname{softmax}(\beta\mathbf{X}^\top\boldsymbol\xi) \;=\; \operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k})\mathbf{V}$$

| Symbol           | Meaning                               | In a Transformer               |
| ---------------- | ------------------------------------- | ------------------------------ |
| $\boldsymbol\xi$ | state / query                         | Q                              |
| $\mathbf{X}$     | stored patterns                       | K (and V)                      |
| $\beta$          | inverse temperature — the regime dial | $1/\sqrt{d_k}$                 |
| $N$              | number of stored patterns             | sequence length                |
| $d$              | dimension                             | $d_{\text{model}}$             |
| $\Delta_i$       | separation of pattern $i$             | how distinguishable a token is |
| $a_i$            | retrieval weight                      | attention weight               |
| $M$              | max pattern norm                      | —                              |

**Rules of thumb**

- $\beta\uparrow$ → peaked → single-pattern retrieval
- $\beta\downarrow$ → flat → averaging
- Need $\beta \sim \log N$ to hold error fixed as the bag grows
- $d\uparrow$ → separation ↑ → retrieval _easier_ (blessing of dimensionality)
- $H(\mathbf{a})$ diagnoses your regime: high = drowning, near-zero = over-peaked
- Compute is $O(Nd)$ per query — same as attention, same quadratic cost

**Mantras**

- Wells are _written_, not _dug_
- Hopfield doesn't learn what to store — it **routes gradient** so the encoder learns
- Order is folded into the _pattern_, never into the _mechanism_
- $O(1)$ means iterations, not compute
- Signal accumulates across examples; noise cancels
- Merged wells are a bug for recall and a feature for MIL

---

## References

- Hopfield, J.J. (1982) — _Neural networks and physical systems with emergent collective computational abilities_
- Krotov & Hopfield (2016) — _Dense Associative Memory for Pattern Recognition_
- Demircigil et al. (2017) — _On a model of associative memory with huge storage capacity_
- Ramsauer et al. (2020) — _Hopfield Networks is All You Need_ · [arXiv:2008.02217](https://arxiv.org/abs/2008.02217)
- Widrich et al. (2020) — _Modern Hopfield Networks and Attention for Immune Repertoire Classification_ (DeepRC)
- Millidge et al. (2022) — _Universal Hopfield Networks_
- Yuille & Rangarajan (2003) — _The Concave-Convex Procedure_
- Dong et al. (2021) — _Attention is not all you need: pure attention loses rank doubly exponentially with depth_
- Geva et al. (2021) — _Transformer Feed-Forward Layers Are Key-Value Memories_

---

## Companion file

`3-transformer-internals.md` picks up where this leaves off. Everything here needs the energy landscape to make sense; everything there can be explained without mentioning Hopfield at all:

- what $W_Q, W_K, W_V, W_O$ actually are (spoiler: individually they don't exist)
- how a randomly initialized model differentiates into specialized heads
- the phase-by-phase formation of capability during training
- grokking vs. induction bump vs. double descent
- why alternative factorizations (K1→K2→V) collapse, and what real variants do instead
