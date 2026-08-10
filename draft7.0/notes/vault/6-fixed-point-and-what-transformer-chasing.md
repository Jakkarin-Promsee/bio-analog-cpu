# Fixed Points, and What a Transformer Is Actually Chasing

> Notes reconstructed from a conversation that started at _"what if I store more patterns than capacity?"_ and ended at _"the wells are finite, but the input isn't."_
> 
> This file is not organised around a paper. It is organised around **one question that kept regenerating itself** — every answer exposed a deeper misunderstanding sitting underneath it. The chain is the content.
> 
> Builds on `4-hopfield-internal.md` (mechanism) and `5-interp-hull-and-superposition.md` (hull + superposition). Read those first — this one assumes $E$, $\operatorname{lse}$, $\beta$, separation $\Delta_i$, and the four variants are already familiar.

---

## How to read this

Each step answers the question the previous step makes unavoidable. Read in order the first time.

|§|The question|The wrong picture it kills|
|---|---|---|
|1|What happens past capacity?|"capacity is a container that fills up"|
|2|What counts as **one well**?|"32 dims means 32 wells"|
|3|How does attention retrieve?|"one landscape per token"|
|4|What does "retrieval" even mean?|"retrieval is one well-defined thing"|
|5|What **is** stopping, mechanically?|"stopping is a property of the update rule"|
|6|Is a fixed point the best answer?|"the well is the most complete memory"|
|7|How do you train a Hopfield?|"training shapes the energy function"|
|8|So what is attention actually doing?|"attention is retrieval"|
|9|Why not let attention converge?|"FFN can rescue it afterwards"|
|10|The closing asymmetry|"depth is the compute budget"|

**If re-reading and short on time: §5, §6, §10.**

---

# Part I — Capacity is not a container

## §1. Writing always succeeds. Reading is what breaks.

### 1.1 The framing that has to go first

The word "capacity" imports a container image: a box that fills, then rejects. That image is wrong in **every** Hopfield variant.

$$\mathbf{W} = \frac{1}{d}\sum_\mu \mathbf{x}^\mu(\mathbf{x}^\mu)^\top \qquad\text{or}\qquad \mathbf{X} = [\mathbf{x}^1,\ldots,\mathbf{x}^N]$$

Adding pattern $N{+}1$ is just adding a term or a column. **No error. No rejection. Nothing is overwritten.** Every pattern you ever wrote is still fully present in the parameters.

> **Capacity is never a statement about writing. It is a statement about whether reading still lands where you wanted.**

This kills the "overwrite" hypothesis outright. It also means the failure is _statistical_, not a threshold event — nothing announces itself at $N = 0.14d$.

### 1.2 What actually degrades — 1982

$0.14d$ is the classical number, and past it three things happen **at once**:

|Failure|What it looks like|
|---|---|
|**Wells drift**|the local minimum moves off $\mathbf{x}^\mu$ — you retrieve it with some bits flipped|
|**Wells vanish**|some stored patterns stop being local minima at all|
|**Wells sprout**|spurious states appear: $\operatorname{sgn}(\mathbf{x}^1 + \mathbf{x}^2 + \mathbf{x}^3)$ becomes stable, though it was never stored|

The third is the counter-intuitive one and it answers the original question directly:

> **The number of wells does not stay the same past capacity. It grows — exponentially — with junk.**

This is the spin-glass phase. It is only possible because 1982 crushes everything into one fixed-size $d\times d$ matrix, from which the individual patterns cannot be recovered.

### 1.3 What actually degrades — modern (2020)

Completely different failure mode, because there is **no cramming**. $\mathbf{X} \in \mathbb{R}^{d\times N}$ — one pattern, one column, one term. Adding patterns widens the matrix; it never competes for fixed space.

So the term count is always exactly $N$. But:

> **$N$ terms ≠ $N$ local minima.**

`lse` merges wells _softly_. When wells sit close together relative to $\beta$, they **melt into a single broader well whose floor is the group average**. That is the Metastable regime of Theorem (C).

So of the three original hypotheses — merge into a bigger well / overwrite / stay the same — **"merge" is right for modern, "sprout junk" is right for 1982, and "overwrite" is never right.**

### 1.4 The real criterion is not a count

$$\varepsilon \sim N,e^{-\beta\Delta_i}$$

Read the shape: $N$ is **linear on the outside**, separation $\Delta_i$ is **inside the exponent**. Consequences:

- Adding more patterns barely hurts — compensate with $\beta \sim \log N$. (300k → 400k stored ≈ 2% more $\beta$.)
- Adding patterns **similar to existing ones** is the killer, because it crushes $\Delta_i$.

> **"Over capacity" in the modern version almost never means "too many things." It means "things too alike for the $\beta$ you have."**

### 1.5 The bug/feature inversion

Merged wells are a **bug for recall** and a **feature for MIL**. A broad basin whose floor is the group average is exactly what `HopfieldPooling` wants. Same phenomenon, opposite verdict, depending on what you asked for. Hold onto this — it comes back in §4.

---

# Part II — Counting things correctly

## §2. $d$ and $N$ are different axes

### 2.1 The confusion, stated plainly

Given DeepRC's shapes:

```
bag of ~300,000 sequences
        ↓
one shared CNN, applied to every sequence     →  N × 32
        ↓
HopfieldPooling   (ξ is nn.Parameter)         →  1 × 32
        ↓
linear + sigmoid                              →  1 bit
```

Which number is "the wells"?

- **32 = the dimension of the space** the landscape lives in ($d$)
- **$N$ = the number of wells** (columns of $\mathbf{X}$)

One instance → CNN → a 32-vector → that vector is the **coordinates of one point** → that point is one well floor.

> **1 instance = 1 vector of 32 numbers = 1 point = 1 well.** The 32 numbers are _where the well is_, not _how many wells there are_.

One instance can never produce multiple wells. The only direction the count can go wrong is **down** (merging, §1.3) — or, in 1982 only, up via spurious states.

### 2.2 Set-at-once, not a loop

The two stages run on different rhythms:

|Stage|Rhythm|Why|
|---|---|---|
|CNN|per instance (parallel, shared weights)|this is what keeps parameter count sane|
|Hopfield|**the entire set, in one shot**|forced by softmax|

The forcing is worth internalising: $a = \operatorname{softmax}(\beta\boldsymbol\xi^\top\mathbf{Z})$ needs a denominator summed over **all $N$**. You cannot accumulate incrementally without already knowing the total. The property $\sum_i a_i = 1$ — which is what makes variable bag size free — comes from exactly this.

And `HopfieldPooling` has **one** $\boldsymbol\xi$: a single marble dropped into a landscape of 300,000 wells. No loop, no accumulator.

### 2.3 Multi-head is more marbles, not more landscapes

Several $\boldsymbol\xi$ = several questions asked of **the same** landscape, answers concatenated. The wells are unchanged. This is the first hint that "landscape" is a per-input object, not a per-query one.

---

# Part III — Attention, mechanically

## §3. $N$ marbles on one landscape

### 3.1 The count, per sequence per head per layer

|Object|Count|Source|
|---|---|---|
|landscape|**1**|the whole sequence through $W_K$|
|wells|$N$|one per key token|
|marbles|$N$|one per query token|

$N$ marbles × $N$ wells = the $N\times N$ attention matrix, and the $O(N^2)$ bill.

**Within one step the marbles do not interact.** They merely happen to roll on the same terrain. There is one landscape per _sequence_, not per token — and "self" means marbles and wells are minted from the same tokens.

Contrast with §2: `HopfieldPooling` survives $N = 300{,}000$ precisely because it has **one** query — $O(Nd)$ instead of $O(N^2)$, which would be $9\times10^{10}$ entries here.

### 3.2 K/V decoupling — the one real structural difference

In pure Hopfield, $\mathbf{X}$ does double duty: it sets where wells are _and_ is what comes back. Transformers split it:

$$\underbrace{\mathbf{K}}_{\text{where the well sits}} \qquad \underbrace{\mathbf{V}}_{\text{what's at the bottom}}$$

This lets a token separate _"what makes me findable"_ from _"what I hand over when found."_ Pure Hopfield cannot express that. This is what the "+ projection matrices" in the equivalence is actually buying.

### 3.3 One step, and why that's fine

Theorem (B): with enough separation, **one step already lands on the fixed point.** Attention applies the rule once and moves on — not as a shortcut, but because the theorem says one step is typically enough.

**Trap:** when the output looks like a blend, that is **not** "stopped too early." At that $\beta$, the well floor _is_ the blend (metastable). Mixture is governed by $\beta$ and $\Delta_i$, never by step count.

### 3.4 $\beta$ is fixed but effective $\beta$ is not

$\beta = 1/\sqrt{d_k}$ is hard-coded and untrained. But the model stretches $|q|,|k|$ freely, which spreads the logits and sharpens softmax. This is why BERT heads land in **all three regimes** while sharing one $1/\sqrt{d_k}$.

### 3.5 Retrieval here **adds**, it does not replace

$$x + \operatorname{Attn}(x)$$

Hopfield wants the query to _become_ the retrieved pattern. Attention only _augments_ the token with a summary of relevant context. The token's identity is never discarded. This single difference is load-bearing for everything in §9.

### 3.6 The line

> **Hopfield: the fixed point is the answer. Transformer: the fixed point is the failure mode.**

If a Transformer's tokens genuinely reached a common fixed point, they'd all be the same vector — rank collapse. Same equation, opposite desiderata.

---

# Part IV — What the words mean

## §4. "Retrieval" bundles two things

### 4.1 Unbundling

**Layer 1 — the operation.** The only thing the math defines:

$$\text{out} = \sum_i a_i\mathbf{x}_i,\qquad a = \operatorname{softmax}(\beta\mathbf{X}^\top\boldsymbol\xi)$$

Feed in $\boldsymbol\xi$, get a point in the convex hull of $\mathbf{X}$. **This always succeeds.** There is no notion of right or wrong here; the equation does not know those words.

**Layer 2 — the intent.** Something we bolt on: _"I want one stored pattern back, cleanly."_

Only once that sentence exists do capacity, $\Delta_i$, spurious states, and $\varepsilon \sim Ne^{-\beta\Delta_i}$ mean anything. Every one of them measures **the gap between what came out and what you wanted**.

> **"Retrieval failed" is never a statement about the equation. It is a statement about the gap.**

This is why "fixed point = answer / fixed point = failure mode" can both be true. Identical operation, different intent.

### 4.2 What makes it _retrieval_ rather than _computation_

Content-addressability, which requires one structural property:

> **The question and the answer live in the same space.** $\boldsymbol\xi$ and $\mathbf{x}_i$ are both $d$-dimensional, comparable by inner product.

Because $E$ is built from $\langle\mathbf{x}^\mu,\boldsymbol\xi\rangle$, the property _"more similar ⇒ lower energy"_ comes for free. A hashmap can't be a landscape: it has matching, not similarity.

### 4.3 The output is never literally a pattern

It is **always** a weighted average. $\text{out} = \mathbf{x}_3$ exactly requires genuinely one-hot $a$, i.e. $\beta\to\infty$. In practice you get _close enough that the first four decimals agree_.

> **Single-pattern retrieval is not the base behaviour. It is one regime among three**, appearing when $\beta$ is high and separation is good.

### 4.4 How much intent each variant carries

||Operation|Bolted-on intent|How much "retrieval" applies|
|---|---|---|---|
|Hopfield 1982 / 2020|identical|return one stored item, correctly|fully|
|`HopfieldLayer`|identical|look up a learned, persistent archive|closest to the IR sense|
|`HopfieldPooling`|identical|summarise; mixing is _desired_|borrowed word|
|attention|identical|**none** — just wants good features for the next layer|borrowed mechanism, not the goal|

The bottom row is the dangerous one. Attention isn't searching memory at all: its $\mathbf{X}$ is **this input's own tokens**, and the landscape is born and dies inside one forward pass. There is no archive.

---

# Part V — Stopping

## §5. What stopping actually is

### 5.1 The unromantic definition

The update rule is a function:

$$f(\boldsymbol\xi) = \mathbf{X}\operatorname{softmax}(\beta\mathbf{X}^\top\boldsymbol\xi)$$

"Stopping" means $f(\boldsymbol\xi) = \boldsymbol\xi$. Ask with the answer, get the answer back. **That is the whole definition.**

### 5.2 Why iterating must reach one

Three ingredients, none optional:

1. $E$ **decreases at every step** (CCCP guarantees monotonicity)
2. $E$ **is bounded below** (proved $E \ge 0$)
3. monotone + bounded ⇒ must settle

> **Termination is not a property of the update rule alone. It is a property of the update rule _plus the existence of $E$_.**

### 5.3 Re-anchoring the original equation

$$E(\boldsymbol\xi) = \underbrace{-\operatorname{lse}(\beta,\mathbf{X}^\top\boldsymbol\xi)}_{\text{pull outward toward patterns}} + \underbrace{\tfrac12|\boldsymbol\xi|^2}_{\text{pull back to origin}}$$

Term 1 wants $\boldsymbol\xi$ to run along a pattern direction — left alone it dives to $-\infty$. Term 2 charges for distance. **A well floor is where the two forces balance.**

Term 2 is the _only_ reason $E$ has a floor. Without it the entire theory collapses.

**Why 1982 didn't need it:** states were caged in ${-1,+1}^d$. Going continuous removed the cage, so the cage had to be written back into the equation.

### 5.4 The three structural conditions for $E$ to exist

1. **Same space in, same space out** — otherwise "apply again" is undefined, so there is no iteration and no fixed point question.
2. **The rule does not change between steps** — same $\mathbf{X}$, same $\beta$, every time. Step 5 uses the identical function as step 1.
3. **Symmetry** ($\mathbf{W} = \mathbf{W}^\top$) — asymmetric weights have no energy function and admit limit cycles. (That's the Amari/Sompolinsky lineage that stores _sequences_. Symmetry ⇒ convergence ⇒ **cycles structurally forbidden**.)

### 5.5 Transformers violate #2, fatally

Every layer has its own $W_Q, W_K, W_V$. So the stack is not iteration at all:

$$f\circ f\circ f\circ f \qquad\text{vs}\qquad f_L\circ\cdots\circ f_2\circ f_1$$

> **A fixed point is a property of _one_ function. There is no "the function" here to have one.**

"No energy function for the stack" is a _consequence_ of this, not an independent fact.

### 5.6 It is possible — people have done it

|Approach|What it restores|
|---|---|
|**Weight tying** (Universal Transformer, ALBERT)|condition #2 — now genuinely $f\circ f\circ f$. UT + ACT has a real per-token halting rule.|
|**DEQ** (Deep Equilibrium Models)|runs one layer to its fixed point, differentiates via the implicit function theorem, stores no intermediate activations|

DEQ's survival trick matters: $z^* = f(z^*, \mathbf{x})$ — **input injection**. The original input is re-fed every step and anchors the dynamics. Remove it and the fixed point of a convex-averaging map is exactly consensus collapse.

### 5.7 Why the field mostly declines

- **Expressivity.** $L$ different functions strictly beat one function applied $L$ times. Tying weights is giving something up.
- **Meaning.** Even a working halt condition would carry no semantic content — see §6.5.

> **Making it stop is cheap. Making the stop _mean something_ requires that the quantity being minimised is the quantity you want.**

Hopfield gets this free because retrieval is directly expressible as an energy. "Predict the next token correctly" is not — if it were, you wouldn't need to train.

---

## §6. A fixed point is not a best answer

### 6.1 Self-consistency ≠ optimality

$$\boldsymbol\xi^* = \sum_i a_i(\boldsymbol\xi^*),\mathbf{x}_i$$

Read aloud: _"asking with the answer returns the answer."_ Nothing in that sentence claims the answer is good.

**Proof by counterexample, already in the theory: spurious states are fixed points too.** Perfectly self-consistent, perfectly still, completely wrong. The system cannot tell from the inside which kind of well it is in.

### 6.2 It doesn't even find the global minimum

Hopfield descends to the **nearest** local minimum, never searches for the deepest one.

> If it did find the global minimum, every query would return the same thing. **The fact that it answers differently per query is precisely because it isn't optimising globally.**

So "trapped, can't get out" is the right description — but that trapping **is** the basin of attraction, i.e. the mechanism working as designed. The only error is believing the basin you fell into is the best one.

### 6.3 What $E$ actually measures

Only: _"how much does $\boldsymbol\xi$ resemble the contents of $\mathbf{X}$."_ Not correctness, not completeness, not usefulness. Store garbage and it descends happily into garbage.

And the answer space is closed: **output always lies in the convex hull of $\mathbf{X}$.** It can never produce anything new. Perfect for retrieval, disqualifying for generation.

### 6.4 The decider

$$E \text{ is computable at inference} \qquad\text{but}\qquad \text{loss is not}$$

||Hopfield|Transformer|
|---|---|---|
|minimised quantity|$E$|loss|
|needs|$\mathbf{X},\boldsymbol\xi,\beta$|**the ground truth**|
|available at inference?|**yes**|**no**|

$E$ needs only things already in hand, so Hopfield can genuinely check itself. To know "no better answer exists," a Transformer would need the correct answer — the one thing it doesn't have. Loss exists only during training, because only training has labels.

### 6.5 Confidence is a proxy, not the objective

You _can_ use output entropy as a "done" signal — early-exit work does exactly this. But a proxy fails where it's miscalibrated: **a confidently wrong model halts confidently.** $E$ is not a proxy for anything; it is the actual objective.

> **A well is not "the most complete memory." It is the point where the dynamics stop changing their mind** — which coincides with a stored item only while you're under capacity.

---

# Part VI — Training

## §7. Where the learning actually lands

### 7.1 Hopfield 1982: there is no training

$$\mathbf{W} = \frac{1}{d}\sum_\mu \mathbf{x}^\mu(\mathbf{x}^\mu)^\top$$

Closed form. One pass. No loop, no gradient, no loss, nothing an optimiser touches.

> **Wells are written, not dug.** The verb is _store_, not _learn_.

Making one useful is a **design** task: choose near-orthogonal patterns, keep $N < 0.14d$. Better closed-form rules exist (Storkey, pseudo-inverse — pushing capacity toward $\sim d$), but they are still formulas, not training.

### 7.2 Modern: the loss exists, just not inside the Hopfield

DeepRC trains end-to-end with ordinary BCE on a 1-bit patient label. Fully standard.

What "doesn't depend on loss" is the **equation**: $E$ has been hard-coded since 2020 and training never touches it.

```
CNN (shared) ──► Hopfield ──► sigmoid ──► loss
   ▲                ▲                       │
   └────────────────┴───────────────────────┘
              gradient flows back

gradient passes *through* the equation,
but there is nothing in the equation to adjust
```

> **Hopfield supplies the geometry. Learning supplies the coordinates.**

### 7.3 What gradient does: raises separation

$$\Delta_i = \min_{j\neq i}\big(\mathbf{x}_i^\top\mathbf{x}_i - \mathbf{x}_i^\top\mathbf{x}_j\big)$$

The encoder is pushed to place _things that must be distinguished_ far apart, and _things that should be averaged together_ close. The rolling-downhill mechanism never changes — only the terrain it runs on.

### 7.4 Why gradient can flow at all

$$\frac{\partial L}{\partial\mathbf{z}_i} \propto a_i \cdot \frac{\partial L}{\partial\text{out}}$$

Selected instances get a large share; the rest get ≈0. Compare the alternatives:

|Op|Failure|
|---|---|
|`max()`|gradient severed — cannot learn|
|`mean()`|gradient split equally; a disease-marker sequence and junk receive **identically**. Not a tuning problem — **structurally dead**, no amount of data helps|
|`softmax`|the only middle ground: turns _relevance_ into a continuous quantity gradient can push|

### 7.5 The self-amplifying loop — this **is** "training a Hopfield"

1. $\boldsymbol\xi$ selects some instances (near-random at init)
2. selected instances get gradient; the rest get ≈0
3. the CNN adapts only to the selected ones
4. if the selection was useful, loss drops → CNN encodes those more distinctly, and $\boldsymbol\xi$ is pulled toward them
5. repeat — the signal amplifies itself

Bootstrapping out of a random guess, made possible because softmax makes the hypothesis _"attend only to things resembling $\mathbf{s}$"_ **expressible** in the first place.

### 7.6 The one exception: `HopfieldLayer`

$\mathbf{X}_K, \mathbf{X}_V$ are `nn.Parameter` of shape $M\times d$, $M$ chosen by you. **The wells persist across forward passes and are learned directly by gradient descent.** This is the only variant matching the intuitive picture of "a trained memory." The other three build and discard their landscape every pass.

---

# Part VII — What attention is for

## §8. Contextualisation, not recall

### 8.1 Correction worth keeping

Hopfield does not extract features — **the encoder does.** Hopfield makes the separation _worth learning_. It is a **gradient router**, not a feature extractor.

### 8.2 The one difference from pooling

`HopfieldPooling`: $\boldsymbol\xi$ is a parameter — one question asked of every bag. Attention: **the query comes from data** — every token asks its own question and adds the answer to itself.

### 8.3 The functional definition

> **Attention is the only operation in the block that moves information between positions.**

Everything else — FFN, LayerNorm — is strictly per-token.

```
attention:  tokens ──╳──╳──╳──►   (lines cross positions)
FFN:        tokens ──│──│──│──►   (lines never cross)
```

So: attention decides, per token per head, **which positions to pull from and how much, based on content rather than position.** FFN supplies per-token capacity — and is itself a key-value memory (Geva et al.), but over a _learned persistent_ store rather than this input's tokens.

### 8.4 The purpose-level answer

There is no archive to search — $\mathbf{X}$ is this input's own tokens. What attention does is turn context-free embeddings into context-aware ones:

> "bank" near "river" must come out a different vector from "bank" near "money."

That requires one token to see others, which only attention can do. And the residual makes it _"me, plus a content-selected summary of my neighbours"_ — never a replacement.

### 8.5 What heads empirically do — mapped onto the regimes

|Head type|Behaviour|Regime|
|---|---|---|
|**induction**|sees $[A][B]\ldots[A]$, points at $B$ — the mechanism behind in-context learning; near-permutation attention matrix|single-pattern (high effective $\beta$)|
|**previous-token**|attends $t{-}1$ regardless of content; pairs with induction heads|sharp, positional|
|**no-op**|dumps weight on BOS/first token, effectively passing the residual through — how a layer "chooses to do nothing"|degenerate|
|**averaging (lower layers)**|broad context gathering, giving upper layers something to select from|global / metastable|

Note the induction head **copies**, it does not average. "Attention blends everything" is false as a general claim; blending is a _mechanism_, not a goal.

### 8.6 What depth buys

Each layer builds its landscape from the previous layer's output, so **the things being attended over get progressively more abstract** — words, then phrases, then facts and relations. Nobody specifies the schedule; BERT's lower-layers-gather / upper-layers-select split is learned.

This is not descent toward anything. Depth pays off because of the **training loss**, not because of motion downhill.

### 8.7 What the Hopfield lens gives, and doesn't

**Gives:** output always in the convex hull (no explosion; any $N$ accepted) · $\beta$ as the regime dial · entropy $H(\mathbf{a})$ as a per-head diagnostic · "learning = raising separation" · rank collapse traced to low $\beta$.

**Does not give:** any reason why these produce good next-token prediction.

> **It explains the mechanism. It never explains the purpose.** A lens, not a compass.

---

## §9. Why you can't just let attention converge

### 9.1 The proposal

_"Iterate attention to a fixed point in each layer; FFN will kick it back out; the next layer starts fresh."_

Self-consistent, and strictly better than "just raise $\beta$." It still fails — for sharper reasons.

### 9.2 Mostly, iterating changes nothing

Theorem (B): with adequate separation you're **already** at the fixed point after one step. Ten more iterations return the same vector. Pure compute burn.

### 9.3 FFN cannot recover collapse — structurally, not merely with difficulty

FFN is a function, applied identically to every token:

$$\mathbf{z}_i = \mathbf{z}_j ;\Longrightarrow; \text{FFN}(\mathbf{z}_i) = \text{FFN}(\mathbf{z}_j)$$

Once two tokens have fused into the same vector, **nothing downstream can separate them.** Same for LayerNorm.

> **FFN amplifies differences that already exist. It cannot create them.** Destroyed information stays destroyed.

And iterating a convex-averaging map to convergence is running the difference-destroying process **to completion in every layer**, instead of taking one step and leaving.

### 9.4 Raising $\beta$ to avoid collapse hits the opposite wall

High enough $\beta$ and each query lands in its own well — no fusion. But then $a$ is near one-hot and:

$$\nabla\operatorname{softmax} = \beta\big(\operatorname{diag}(a) - aa^\top\big) \longrightarrow \mathbf{0}$$

**Gradient dies; the model can't train.** You also throw away the metastable regime that lower-layer heads legitimately use.

> Soft and iterated → collapse. Sharp and iterated → untrainable. **No $\beta$ buys both.**

### 9.5 And it would mean nothing anyway

A fixed point is meaningful because $E$ encodes similarity-to-stored-items. A Transformer layer's job is to hand better features upward. Reaching a fixed point isn't an achievement it fails at — it's a goal with no content.

### 9.6 DEQ does it, and how it survives is the lesson

$$\mathbf{z}^* = f(\mathbf{z}^*, \mathbf{x})$$

**Input injection** — the original input is re-fed every step, anchoring identity so everything doesn't drift into consensus. Remove it and DEQ collapses.

Which reframes everything: converging alone is **not enough**; something must continuously hold the original identity in place.

> **In a standard Transformer, that something is the residual.** $x + \operatorname{Attn}(x)$ _is_ input injection, performed exactly once.

---

# Part VIII — The closing asymmetry

## §10. Finite wells, unbounded input

### 10.1 The sentence the whole conversation was walking toward

> **A Hopfield's answer space is the convex hull of $\mathbf{X}$ — a closed vocabulary. Input, and language, are open-ended.**

Retrieval terminates because the set of possible answers is _given in advance_. That is the precondition for a well-defined energy, a provable halt, and a meaningful fixed point.

Generation has no such set. The answer isn't among the input tokens; it has to be constructed. **The very property that makes Hopfield's stopping meaningful is the property that disqualifies it from the Transformer's job.**

### 10.2 Fixed depth = fixed compute per token

$$\text{easiest token} \quad\text{and}\quad \text{hardest token} \quad\longrightarrow\quad \text{exactly } L \text{ layers}$$

Hopfield adapts naturally: a well-separated query halts in one step, an ambiguous one iterates. Input difficulty sets the compute. A Transformer structurally cannot do this — $L$ is chosen by a human before training and welded into the architecture.

So $L$ is simultaneously **too much** for easy tokens and **too little** for hard ones, always.

### 10.3 The escape: borrow the loop from outside

Autoregressive generation is a loop that lives **outside** the layers. (Do not confuse it with Hopfield's iteration index $t$ — different loop entirely.)

> **When depth runs out, the model writes a token and reads it back in — buying another $L$ layers.**

**This is what chain-of-thought is, mechanically.** Not "the model thinks in steps and gets smarter," but **trading depth for length**: the loop Hopfield owns internally and can halt on its own, a Transformer has to borrow from the outermost shell of the system.

It changes what is computable, not just what is convenient — a fixed-depth Transformer sits in a limited complexity class, and problems needing more sequential steps cannot be done in a single forward pass however well trained. Emitting intermediate tokens relaxes that bound.

### 10.4 Why "just answer as well as it can within $L$" is right — with one correction

Compensation isn't happening at inference. It is **baked in during training.**

Every weight is co-adapted to that specific $L$. Lower layers learn to do work useful to upper layers _knowing how many remain_. Truncating a trained 12-layer model to 8 at inference doesn't degrade gracefully — it **shatters**, because layer 8 was never trained to be last.

Compare: Hopfield can be stopped at any step and degrades smoothly, because $E$ genuinely decreases the whole way. The stack has no such quantity to cut against.

---

# Appendix A — Traps from this conversation

|Wrong picture|Correction|
|---|---|
|capacity is a container that fills up|writing always succeeds; **reading** is what degrades|
|past capacity, patterns overwrite each other|nothing is ever overwritten — it's superposition (1982) or more columns (2020)|
|past capacity, the well count stays the same|1982: it **grows** with spurious junk. Modern: it **shrinks** via merging|
|32 dims means 32 wells|32 is the _dimension_; $N$ is the well count|
|Hopfield processes the bag one instance at a time|the CNN does; Hopfield takes the **whole set at once** (softmax denominator forces it)|
|multi-head = multiple landscapes|multiple **marbles**, one landscape|
|attention has one landscape per token|one per **sequence**; $N$ marbles roll on it|
|blended output = "stopped too early"|at that $\beta$ the well floor **is** the blend|
|a fixed point is the best answer|it's self-consistency; spurious states qualify too|
|Hopfield finds the global minimum|nearest local minimum — global search would make every query identical|
|$E$ measures correctness|it measures **resemblance to $\mathbf{X}$**|
|$E = 0$ is the stopping condition|$\nabla E = 0$ is. $E \ge 0$ is just where the constant puts the floor|
|training reshapes the energy function|$E$ is hard-coded; training moves the **coordinates fed into it**|
|Hopfield "doesn't use a loss"|modern variants train with ordinary losses — just outside the equation|
|"attention is retrieval"|mechanism borrowed; **there is no archive** — $\mathbf{X}$ is this input's own tokens|
|FFN can undo rank collapse|$\mathbf{z}_i = \mathbf{z}_j \Rightarrow$ identical outputs. It amplifies difference, never creates it|
|just iterate attention to convergence|soft → collapse; sharp → gradient dies. And it would mean nothing|
|CoT = the model reasoning|mechanically: **trading depth for length**, borrowing the loop from outside|

---

# Appendix B — Quick reference

**The chain in one pass**

1. Capacity isn't a container — writing always works, reading degrades
2. $N$ terms ≠ $N$ minima; $d$ is the dimension, $N$ is the well count
3. Attention = $N$ marbles, one landscape, K/V decoupled, one step, result **added**
4. "Retrieval" = a always-succeeding operation + a bolted-on intent
5. Stopping = $f(\boldsymbol\xi)=\boldsymbol\xi$, possible only because a decreasing, bounded-below $E$ exists
6. A fixed point is self-consistency, not optimality — and $E$ is checkable at inference, loss is not
7. Training never touches the equation; it raises separation, routed by softmax
8. Attention = the only cross-position mixer; its purpose is contextualisation, not recall
9. Converging attention is impossible-in-principle, not merely hard — unless you add input injection
10. Wells are finite, input isn't. Fixed $L$ → CoT borrows the loop from outside

**Formulas to keep in the hand**

$$E(\boldsymbol\xi) = -\operatorname{lse}(\beta,\mathbf{X}^\top\boldsymbol\xi) + \tfrac12|\boldsymbol\xi|^2 \qquad f(\boldsymbol\xi) = \mathbf{X}\operatorname{softmax}(\beta\mathbf{X}^\top\boldsymbol\xi)$$

$$\varepsilon \sim N e^{-\beta\Delta_i} \qquad \frac{\partial L}{\partial\mathbf{z}_i} \propto a_i\frac{\partial L}{\partial\text{out}} \qquad \nabla\operatorname{softmax} = \beta(\operatorname{diag}(a) - aa^\top)$$

**Mantras**

> Capacity is never about writing. It is about whether reading still lands where you wanted.

> $N$ terms is not $N$ local minima.

> Retrieval failed is a statement about the gap, never about the equation.

> Termination is the update rule **plus** the existence of $E$.

> A fixed point is a property of one function. A stack has no such function.

> Making it stop is cheap. Making the stop mean something is not.

> A well is where the dynamics stop changing their mind.

> Wells are written, not dug. Hopfield supplies the geometry; learning supplies the coordinates.

> FFN amplifies differences. It cannot create them.

> The residual is input injection, performed once.

> The wells are finite. The input is not.

---

## Open threads

Things this conversation opened but did not close — carry into future notes.

- **Jacobian / spectral conditions** at the fixed point — the remaining gap flagged in note 4. Ties directly to §5.2 and to why one step suffices.
- **Rank collapse, formally** — Dong et al. 2021, "Attention is not all you need." The proof that residual + FFN are what prevent doubly-exponential convergence to rank 1.
- **Induction heads** — Olsson et al., in-context learning. The cleanest empirical case of §8.5's single-pattern regime.
- **DEQ and implicit differentiation** — Bai et al. The concrete version of §9.6.
- **ACT / Universal Transformer / Mixture-of-Depths** — every attempt to bolt a halting rule onto §10.2.
- **CoT and complexity classes** — formal results on what intermediate tokens buy over fixed depth. The rigorous form of §10.3.
- **Storkey / pseudo-inverse rules** — closed-form storage beating Hebbian capacity, still without training (§7.1).

---

## Companion files

- `1-transformer.md` — the architecture end to end
- `2-hopfield.md` — organised around the 2020 paper
- `3-transformer-internal.md` — what a Transformer really is
- `4-hopfield-internal.md` — the four variants, mechanism by mechanism
- `5-interp-hull-and-superposition.md` — convex hull and superposition
- **this file** — where the two frames meet, and where the analogy breaks