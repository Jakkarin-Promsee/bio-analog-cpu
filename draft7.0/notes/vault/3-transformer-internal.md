# Transformer Internals: How the Thing Assembles Itself

> Companion to `2-hopfield.md`. That file needs the energy landscape to explain anything. This one doesn't mention it once.
>
> The question here: **you understand multi-head, blocks, residual, FFN, CoT — but you can't picture the formation.** What is each layer running toward at random init? How does a uniform soup of random matrices become specialized machinery? And what are $W_Q,W_K,W_V,W_O$ _actually_?

---

## Contents

1. **What the parameters are** — gauge freedom, and why $W_Q$ alone is meaningless
2. **How structure appears from randomness** — symmetry breaking and winner-take-all
3. **The formation timeline** — phase by phase, with the mechanism for each
4. **Phase transitions** — grokking vs. induction bump vs. double descent
5. **Alternative factorizations** — why K1→K2→V collapses, and what actually works

---

# §1. What are $W_Q$, $W_K$, $W_V$, $W_O$ really?

## §1.1 The uncomfortable answer: individually, they don't exist

Not a figure of speech. Here's the proof.

$$s_{ji} = (W_Q\mathbf{x}_j)^\top(W_K\mathbf{x}_i) = \mathbf{x}_j^\top \underbrace{W_Q^\top W_K}_{\textstyle W_{QK}} \mathbf{x}_i$$

Insert any invertible $R$:

$$W_Q \to RW_Q, \qquad W_K \to R^{-\top}W_K$$
$$\Longrightarrow (RW_Q)^\top(R^{-\top}W_K) = W_Q^\top R^\top R^{-\top} W_K = W_Q^\top W_K \qquad \textbf{unchanged}$$

**There is a full $GL(d_k)$ gauge freedom.** Distort $W_Q$ however you like; as long as $W_K$ is distorted inversely, the model computes exactly the same function.

Same on the output side:

$$W_V \to RW_V, \quad W_O \to W_O R^{-1} \;\Longrightarrow\; W_OW_V \text{ unchanged}$$

$$\boxed{\text{The "true value" of } W_Q \text{ alone does not exist. Only } W_Q^\top W_K \text{ does.}}$$

So: _"how does $W_Q$ know what its job is?"_ — **it doesn't, and it doesn't have one.** The pair has a job.

Gradient descent optimizes the **product**. How credit gets divided between the two factors is **completely arbitrary** — an accident of initialization and optimizer path, carrying no information.

## §1.2 What actually exists: two circuits

From Elhage et al. (2021), _A Mathematical Framework for Transformer Circuits_:

$$\boxed{W_{QK} = W_Q^\top W_K} \qquad\qquad \boxed{W_{OV} = W_O W_V}$$

| Circuit        | Shape                                     | Question it answers                             |
| -------------- | ----------------------------------------- | ----------------------------------------------- |
| **QK circuit** | $d_{\text{model}}\times d_{\text{model}}$ | **where to look**                               |
| **OV circuit** | $d_{\text{model}}\times d_{\text{model}}$ | **what to write back into the residual stream** |

$$\text{head} = \underbrace{\text{softmax}(\mathbf{x}^\top W_{QK}\mathbf{x})}_{\text{read from where}} \cdot \underbrace{W_{OV}\mathbf{x}}_{\text{write what}}$$

**Four matrices is an illusion. There are two objects.**

### Why factorize at all, then?

Pure economics:

$$768\times768 = 590\text{k parameters} \qquad\text{vs}\qquad 2\times(768\times64) = 98\text{k}$$

Two benefits: 6× fewer parameters, and a forced constraint $\operatorname{rank}(W_{QK}) \le 64$, which is a useful inductive bias. Neither has anything to do with Q and K "meaning" different things.

## §1.3 So what makes Q different from K?

$$W_{QK} \neq W_{QK}^\top$$

**The asymmetry of $W_{QK}$** is the only thing distinguishing _"A attends to B"_ from _"B attends to A."_

Position in the computation graph defines the role: row index = query position, column index = key position.

$$\textbf{The structure of the computation assigns the role. The matrix does not carry one.}$$

The labels Q/K/V are **names we attach afterward** for explanatory convenience.

## §1.4 Practical consequence for interpretability

$$\text{To interpret a head, examine } W_{QK} \text{ and } W_{OV}. \text{ Examining } W_Q \text{ alone is examining a gauge.}$$

This is a general lesson in mechanistic interpretability: **the unit that is interpretable does not coincide with the unit that is stored.** The same lesson appears again with superposition, where individual neurons don't hold individual features.

---

# §2. How structure appears from randomness

## §2.1 At initialization, nothing is "wrong" — nothing is _different_

$$\mathbf{q}, \mathbf{k} \sim \mathcal{N}(0,\sigma^2) \;\Longrightarrow\; \frac{\mathbf{q}^\top\mathbf{k}}{\sqrt{d_k}} \approx 0 \;\Longrightarrow\; \mathbf{a} \approx \Big[\tfrac1N,\dots,\tfrac1N\Big]$$

**Every head performs mean pooling.** This isn't "attention that's still bad" — it's _no attention at all_. The output is the average of $V$ across the whole window:

$$\text{out}_j = \frac{1}{N}\sum_i\mathbf{v}_i \qquad \text{— identical for every position } j$$

Logits are near-uniform:

$$L_0 = \ln|V| = \ln 50257 \approx \mathbf{10.8}$$

And crucially: **every head is doing exactly the same thing.** No head is specialized in anything.

> The right mental image of initialization is not _chaos_. It is _undifferentiated sameness_.

## §2.2 Symmetry breaking

If all heads start identical, why does head 3 become the previous-token head rather than head 5?

**Pure chance.** Whichever head happens to initialize slightly closer to a useful function receives a slightly larger gradient toward it. Then winner-take-all kicks in:

$$\text{head } h \text{ improves} \;\Rightarrow\; \text{that job is now done} \;\Rightarrow\; \text{gradient pushing other heads to duplicate it} \to 0 \;\Rightarrow\; \text{they diverge to other jobs}$$

$$\boxed{\text{Symmetry breaking: a microscopic accident, amplified into a division of labor}}$$

**Nobody assigns roles.** It's a competition that resolves into specialization, because duplicated work earns no gradient.

This is the same self-amplifying loop described in `2-hopfield.md` §10.5 — and it's the reason head assignments differ between training runs even with identical data and hyperparameters.

---

# §3. The formation timeline

The governing rule, stated first:

$$\boxed{\text{Gradient descent follows } \textbf{consistency across examples}, \text{ not simplicity}}$$

$$\text{Easy things come first not because they are easy — but because they } \textit{agree across more examples}$$

## Phase 0 — initialization

$L \approx 10.8$. Uniform attention, undifferentiated heads, near-uniform output distribution.

## Phase 1 — unigram statistics ($L: 10.8 \to \approx 6.5$)

Look at the output gradient:

$$\frac{\partial L}{\partial \text{logits}} = \mathbf{p} - \text{onehot}(y)$$

Every example whose target is `the` pushes up the logit for `the` **identically, regardless of context.**

$$\Longrightarrow \text{the most consistent direction in the entire dataset is token frequency}$$

The model learns unigram statistics first, via biases and a near-constant residual signal. **Attention plays no role.** This is "signal accumulates, noise cancels" in its purest form.

## Phase 2 — bigrams, and the first real head ($L: 6.5 \to \approx 5.0$)

The next-most-consistent direction: _"`New` is usually followed by `York`."_

To use this, the model must know **what the previous token was** ⇒ it needs a head that looks back one position.

$$\textbf{The previous-token head forms here — because it is useful on its own terms}$$

**Remember this sentence. It's the key to everything in Phase 3.**

## Phase 3 — the induction bump ($L: 5.0 \to \approx 4.2$)

### The problem bigrams can't touch

$$\texttt{... Q7 K2 ... ... Q7 } \to \; ?$$

`Q7` and `K2` have **no statistical relationship in the corpus.** They co-occur only in this context. (Think: an unusual name introduced earlier in the same article.)

Getting this right requires three steps:

1. At the second `Q7`, find the earlier occurrence of `Q7`
2. See what token followed it
3. Copy that

### This needs two heads across two layers

$$\underbrace{\text{Prev-token head}}_{\text{early layer}}: \text{writes into every position "the token before me was } X\text{"}$$
$$\underbrace{\text{Induction head}}_{\text{later layer}}: \text{query} = \texttt{Q7},\;\; \text{key} = \text{"before me was } \texttt{Q7}\text{"} \;\to\; \text{locate } \texttt{K2} \;\to\; \text{copy}$$

### ⭐ Why it cannot be learned incrementally

| Situation                     | What happens                                                                              | Gradient    |
| ----------------------------- | ----------------------------------------------------------------------------------------- | ----------- |
| Head 2 exists, head 1 doesn't | queries for "before me was Q7" but nothing wrote that → keys are noise → attention random | $\approx 0$ |
| Head 1 exists, head 2 doesn't | the information is written but nothing reads it usefully                                  | $\approx 0$ |

$$\textbf{Half a circuit earns nothing ⇒ the gradient is flat in both directions ⇒ no gradual ascent}$$

### How it happens anyway: scaffolding

**Head 1 does not form for induction's sake.** It formed back in Phase 2 **because bigram prediction needed it**, for entirely unrelated reasons.

$$\text{prev-token head already exists "for free"} \;\Longrightarrow\; \text{head 2's gradient becomes nonzero}$$
$$\Longrightarrow \text{head 2 forms} \;\Longrightarrow\; \text{circuit completes} \;\Longrightarrow\; \textbf{loss drops sharply}$$

$$\boxed{\text{Not simultaneous discovery — a prerequisite built for another purpose becomes a ladder}}$$

**This is the answer to "what is actually happening during formation."** The model never plans to build an induction head. It builds a previous-token head for bigrams, and that component happens to open a door to something more complex.

Once the circuit closes, the self-amplifying loop takes over:

$$\text{it works} \to \mathbf{q}^\top\mathbf{k} \uparrow \to \mathbf{a} \text{ sharpens} \to \text{copying gets precise} \to \text{gradient strengthens} \to \text{sharper still}$$

In-context learning appears in the same window. This is the best-documented case of _"a capability emerged"_ being identical to _"a specific circuit finished assembling"_ (Olsson et al. 2022, verified by ablation).

## Phase 4 — long refinement ($L: 4.2 \to \approx 3.3$)

Syntax, semantics, factual recall in the FFN, and much else.

**This phase remains largely a black box.** We know FFNs store facts and that features live in superposition, but nobody has traced its formation order the way induction heads were traced.

## Summary

| Phase | Most consistent direction   | Requires                   | $L$  |
| ----- | --------------------------- | -------------------------- | ---- |
| 0     | —                           | —                          | 10.8 |
| 1     | token frequency             | biases only                | 6.5  |
| 2     | previous token → next token | **prev-token head**        | 5.0  |
| 3     | in-context patterns         | **prev-token + induction** | 4.2  |
| 4     | syntax, semantics, facts    | whole model                | 3.3  |

> **Caveat on the numbers.** These are approximate values for a small model. Real training does not have crisp phase boundaries — phases overlap, and large models learn many things in parallel. Treat this as a useful frame, not a clean staircase.

## The three-line version

1. **At init nothing is chaotic — everything is identical.** All heads mean-pool.
2. **Differentiation comes from amplified accidents.** Tiny random asymmetry + winner-take-all = division of labor.
3. **Complex capabilities are never discovered directly.** They ride on components built earlier for unrelated reasons.

$$\text{The model doesn't search for answers — it accumulates incidentally useful parts until some of them compose}$$

---

# §4. Phase transitions: three different things that look alike

|                    | Cause                                      | What jumps                              | Visible en route?                |
| ------------------ | ------------------------------------------ | --------------------------------------- | -------------------------------- |
| **Induction bump** | a two-part circuit completes               | train **and** val together              | ❌ not from loss                 |
| **Grokking**       | weight decay strips the memorizing circuit | **val only** (train saturated long ago) | ❌ but progress measures show it |
| **Double descent** | crossing the interpolation threshold       | val, after first getting worse          | ✅ visible as a curve            |

**Instant discriminator:** look at train accuracy.

$$\text{train not yet saturated} \Rightarrow \text{induction bump} \qquad\qquad \text{train saturated long ago} \Rightarrow \text{grokking}$$

## §4.1 Grokking in detail

Power et al. (2022): small models on modular arithmetic reach 100% train accuracy immediately, then sit at chance validation accuracy for **tens of thousands of steps**, then jump to 100%.

The accepted explanation (Nanda et al. 2023, _Progress measures for grokking via mechanistic interpretability_):

| Stage                    | What's happening                                                                     |
| ------------------------ | ------------------------------------------------------------------------------------ |
| 1. Memorization          | training set memorized; no general circuit yet                                       |
| 2. **Circuit formation** | a Fourier/trig circuit grows steadily — **while val accuracy shows nothing**         |
| 3. Cleanup               | weight decay eats the memorizing circuit; the general one is exposed → **val jumps** |

$$\text{Grokking is not a } \textit{jump} \textbf{ — it is hidden progress, suddenly revealed}$$

Stage 2 is continuous the whole time. It's simply invisible to accuracy, because the memorizing circuit masks it.

## §4.2 The shared shape

Three instances of one pattern across both files:

$$\text{HopfieldPooling ignition} \quad|\quad \text{Induction head} \quad|\quad \text{Grokking}$$

$$\text{no partial credit} \;\Longrightarrow\; \text{gradient cannot guide the intermediate states} \;\Longrightarrow\; \text{a jump}$$

## §4.3 A necessary caution

Many claimed "emergent abilities" in large models are **not real phase transitions.** Schaeffer et al. (2023) showed they are artifacts of discontinuous metrics (exact-match accuracy). Measure log-probability instead and the curves are smooth.

**Induction heads and grokking are different** — both are measured at the circuit level (progress measures, ablations), not by accuracy alone. Those two are real.

---

# §5. Alternative factorizations

## §5.1 Why K1 → K2 → V collapses

Natural proposal: why not chain the lookup — a key pointing to another key, which then points to the value?

If the composition is linear, it collapses immediately:

$$W_{Q}^\top W_{K_1} W_{K_2} = \text{just another matrix}$$

And it's strictly worse:

$$\operatorname{rank}(ABC) \le \min\big(\operatorname{rank}A, \operatorname{rank}B, \operatorname{rank}C\big)$$

$$\Longrightarrow \textbf{Q, K1, K2, V is } \textit{less} \textbf{ expressive than Q, K, V}$$

Every extra factor squeezes the rank without buying anything. General rule: **linear depth adds no expressive power.**

## §5.2 But with a nonlinearity between them — that's exactly right, and it already exists

$$\text{K1} \xrightarrow{\ \text{softmax}\ } \text{K2} \xrightarrow{\ \text{softmax}\ } \text{V}$$

This is **two-hop attention**, and it is:

$$\textbf{the induction head}$$

$$\underbrace{\text{prev-token head (layer } \ell)}_{\text{K1: what came before me}} \;\to\; \underbrace{\text{induction head (layer } \ell+1)}_{\text{K2: use that to search further}} \;\to\; \text{copy V}$$

$$\boxed{\text{Chained key composition } = \textbf{depth}}$$

Transformers don't put K2 inside a single head because that gains nothing (it collapses). They let you **stack heads across layers** instead, which works — because softmax, FFN, and LayerNorm sit in between.

**The intuition is correct; the answer lives on the depth axis rather than the matrices-per-head axis.**

## §5.3 What real variants do

| Method               | Change                                                                      | Gain                  |
| -------------------- | --------------------------------------------------------------------------- | --------------------- |
| **MQA / GQA**        | many heads share one $W_K, W_V$                                             | much smaller KV cache |
| **MLA** (DeepSeek)   | compress KV through a latent, then decompress — **genuinely adds a factor** | ~93% smaller cache    |
| **Linear attention** | drop softmax, use associativity $(QK^\top)V \to Q(K^\top V)$                | $O(N^2) \to O(N)$     |

**MLA is the working version of the K1→K2 idea** — but note its goal is **memory compression, not added capability**, exactly as the theory predicts.

**Linear attention is the instructive one.** Remove the softmax and the structure changes completely: you can regroup the matrix products and save enormously. But capability genuinely degrades — linear-attention models are markedly worse at induction-style tasks.

$$\text{Softmax is not decoration. It is the only thing preventing everything from collapsing into one matrix.}$$

---

# §6. Quick reference

| Question                               | Answer                                                       |
| -------------------------------------- | ------------------------------------------------------------ |
| What is $W_Q$?                         | A gauge. Individually meaningless                            |
| What's real?                           | $W_{QK} = W_Q^\top W_K$ and $W_{OV} = W_OW_V$                |
| What distinguishes Q from K?           | Asymmetry of $W_{QK}$ plus position in the computation graph |
| What does the model look like at init? | All heads mean-pool; all heads identical                     |
| How do heads specialize?               | Random asymmetry, amplified by winner-take-all               |
| What does gradient descent follow?     | Consistency across examples, not simplicity                  |
| How do complex circuits form?          | Scaffolding — prerequisites built for other reasons          |
| Why do jumps happen?                   | No partial credit ⇒ flat gradient ⇒ bistable                 |
| Why not chain more matrices?           | Linear composition collapses and lowers rank                 |
| Where does chaining actually work?     | Across layers, with softmax between — i.e. depth             |

**Mantras**

- Interpretable units ≠ stored units
- Init is _undifferentiated_, not chaotic
- Consistency beats simplicity in the race to be learned first
- Nothing plans; things accumulate until they compose
- Softmax is the only thing standing between depth and collapse

---

## References

- Elhage et al. (2021) — _A Mathematical Framework for Transformer Circuits_
- Olsson et al. (2022) — _In-context Learning and Induction Heads_
- Power et al. (2022) — _Grokking: Generalization Beyond Overfitting_
- Nanda et al. (2023) — _Progress measures for grokking via mechanistic interpretability_
- Schaeffer et al. (2023) — _Are Emergent Abilities of Large Language Models a Mirage?_
- Geva et al. (2021) — _Transformer Feed-Forward Layers Are Key-Value Memories_
- Dong et al. (2021) — _Attention is not all you need_
- DeepSeek-AI (2024) — _DeepSeek-V2_ (MLA)

---

## Companion file

`2-hopfield.md` — everything that needs the energy-landscape framing: associative memory from 1982 onward, why attention _is_ Hopfield retrieval, gradient routing in MIL, and why a Transformer is a Hopfield network deliberately prevented from converging.
