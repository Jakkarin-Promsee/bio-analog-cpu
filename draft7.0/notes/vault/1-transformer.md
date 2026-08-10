# Transformer — Complete Notes

> Personal reference. Built from zero, in flow order. Every symbol is defined at the point it first appears. Read top to bottom the first time — later sections assume earlier ones.

**Reading map**

| Part                  | Sections                               | What it answers                            |
| --------------------- | -------------------------------------- | ------------------------------------------ |
| I. Input              | Embedding                              | How text becomes numbers                   |
| II. Motivation        | Why RNN isn't enough                   | What problem this architecture solves      |
| III. Core             | What is Transformer / Attention        | The one mechanism everything else supports |
| IV. Patches           | Multi-Head, Masking, Position Encoding | Fixing three holes in raw attention        |
| V. Rest of the block  | FFN, Residual, Normalization           | Why attention alone collapses              |
| VI. Output            | Prediction                             | How vectors become words                   |
| VII. Depth            | Multi Block                            | What stacking actually buys                |
| VIII. Learning        | Training                               | How random numbers become a model          |
| IX. Test-time compute | CoT, TC⁰, Latent CoT                   | Why "think step by step" works             |
| X. More               | Inference, Post-training, Theory       | Everything that didn't fit above           |

---

## Notation used everywhere

| Symbol                          | Meaning                                                |
| ------------------------------- | ------------------------------------------------------ |
| $n$                             | number of tokens in the sequence (sequence length)     |
| $d$                             | model dimension — how many numbers represent one token |
| $\lvert V \rvert$               | vocabulary size (~50k–200k)                            |
| $L$                             | number of blocks (layers)                              |
| $h$                             | number of attention heads                              |
| $d_k, d_v$                      | key/value dimension **per head**, usually $d/h$        |
| $d_{ff}$                        | FFN inner dimension, usually $4d$                      |
| $\theta$                        | all learnable parameters of the model                  |
| $x \in \mathbb{R}^d$            | "x is a vector of $d$ real numbers"                    |
| $W \in \mathbb{R}^{a \times b}$ | matrix with $a$ rows, $b$ columns                      |
| $O(\cdot)$                      | growth rate, ignoring constants                        |

**Linear projection** $y = xW$ means: mix the numbers in $x$ into a new set of numbers. $y_j = \sum_i x_i W_{ij}$. Think: $x$ = ingredients, $W$ = recipes, $y$ = dishes.

---

# Embedding

## Tokenization (upstream of everything)

The model never sees characters or words. It sees **tokens** — subword chunks.

```
"I want ice cream"  ->  ["I", " want", " ice", " cream"]  ->  [40, 765, 4771, 8566]
```

Why subword and not word or character:

- **Word-level**: vocabulary explodes, and any unseen word is OOV (out of vocabulary).
- **Character-level**: sequences get very long — and cost is $O(n^2)$, so length is expensive.

BPE (Byte Pair Encoding): start from characters, repeatedly merge the most frequent adjacent pair until you hit the target vocab size. WordPiece and SentencePiece/Unigram are variants of the same idea.

## Embedding table

Translate token id into a vector.

$$E \in \mathbb{R}^{|V| \times d}$$

- $|V|$ = number of rows (1 token per row)
- $d$ = model dimension ($d$ numbers per token vector)

Token id 77 → grab row 77. Output for the whole sequence: $X \in \mathbb{R}^{n \times d}$.

The numbers in $E$ are **not designed** — they start random and are shaped by training. The end result is that words with similar meaning end up with similar vectors.

---

# Why RNN/LSTM isn't enough?

RNN/LSTM use seq-to-seq, gradually reading token by token. Each token read updates the note (hidden state):

$$h_t = f(h_{t-1}, x_t)$$

- $h_t$ = the memory note at time $t$ (a fixed-size vector)
- $x_t$ = token at $t$
- $f$ = "old note + new token → new note"

This forces a fixed order: read token 1 → update $h_1$ → predict token 2 → read token 2 → update $h_2$ → predict → etc. Two problems follow:

**1. Sequential bottleneck.** To compute $h_5$ you need $h_4$, which needs $h_3$… **nothing can be parallelized along time.** GPUs are good at doing many things at once; forcing one-at-a-time throws that away. A 1000-token sentence = 1000 strictly serial rounds.

**2. Path length.** Information from position $i$ reaches position $j$ only after $|i-j|$ matrix multiplications, and each step can overwrite the meaning. Multiply by numbers slightly below 1 fifteen times → the signal **vanishes**; slightly above 1 → it **explodes**. Gradients travelling backwards hit the same wall.

Concretely: _"The **scientist** who worked in the lab located in the suburbs since 1990 **received** the award."_ — "received" must connect to "scientist" 15 tokens away.

LSTM's gating helps but does not remove either problem.

**Result: RNN/LSTM are bad at long-range dependencies and bad at throughput.**

## The comparison table from the paper

| Layer type               | Complexity/layer | Sequential ops | Max path length |
| ------------------------ | ---------------- | -------------- | --------------- |
| Self-attention           | $O(n^2 d)$       | $O(1)$         | $O(1)$          |
| Recurrent                | $O(n d^2)$       | $O(n)$         | $O(n)$          |
| Convolution (kernel $k$) | $O(k n d^2)$     | $O(1)$         | $O(\log_k n)$   |

The trade is explicit: **buy $O(1)$ path length and full parallelism, pay $O(n^2)$ compute.**

---

# What is Transformer?

Transformer solves it with **content-based addressing**:

$$\alpha_{ij} = \text{softmax}_j\left(\frac{q_i^\top k_j}{\sqrt{d_k}}\right), \qquad y_i = \sum_j \alpha_{ij} v_j$$

$y_i$ does not depend on $y_{i-1}$. Path length becomes $O(1)$ and the sequential bottleneck is gone.

## What is Attention?

### 1. Dictionary lookup

Three parts:

- **Query** = what we're looking for
- **Key** = the label on each entry
- **Value** = the actual content stored at that entry

Take the query, compare against every key, return the value of the matching one.

```python
d = {"cat": "four-legged furry animal", "fish": "water animal"}
d["cat"]   # -> "four-legged furry animal"
```

**Why this exact match fails for us:** it's brittle (`"meow"` finds nothing), and more importantly **it is not differentiable**. Match/no-match is a jump. Nudge the input slightly and either nothing changes or everything changes at once. No gradient → no learning.

### 2. Soft

Instead of picking one entry, score similarity against _all_ keys and average.

```
query = "meow"

1. key = "cat"  0.8  -> value = "description of cat..."  * 0.8
2. key = "fish" 0.2  -> value = "description of fish..." * 0.2

result = 0.8 * value(cat) + 0.2 * value(fish)
```

Now everything is continuous: nudge the query → scores shift slightly → output shifts slightly → **gradient exists → trainable.**

**This is the whole idea.** Everything below is detail about how "similar" is measured and how scores become proportions.

### 3. Measure similarity (Dot product)

$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

- $q$ = what we're looking for
- $k$ = the label of an entry
- $q \cdot k$ = the score for that key

Example: $[1,2,3] \cdot [4,0,5] = 4 + 0 + 15 = 19$.

Why it measures similarity — it's large when two vectors point the same way:

$$[1,0]\cdot[1,0] = 1 \quad(\text{same direction})$$ $$[1,0]\cdot[0,1] = 0 \quad(\text{orthogonal})$$ $$[1,0]\cdot[-1,0] = -1 \quad(\text{opposite})$$

The single number that comes out is called a **score** or **logit**.

### 4. Softmax — scores into proportions

Scores can be any real number, including negative. We need proportions that sum to 1.

$$\text{softmax}(s)_i = \frac{e^{s_i}}{\sum_j e^{s_j}}$$

Two jobs: (a) $e^x$ is always positive, killing negatives; (b) dividing by the sum forces them to total 1.

Example, $s = [2, 1, -1]$:

$$e^2 = 7.39,\quad e^1 = 2.72,\quad e^{-1} = 0.37 \quad \text{sum} = 10.48$$ $$\text{softmax} = [0.71,\ 0.26,\ 0.03]$$

**Important property: $e^x$ amplifies differences.** A score gap of 3 becomes a proportion ratio of ~20×. This is exactly why scaling matters next.

### 5. Why divide by $\sqrt{d_k}$?

Because $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ sums $d_k$ terms. If each $q_i, k_i$ has mean 0 and variance 1:

$$\mathbb{E}[q \cdot k] = 0, \qquad \text{Var}[q \cdot k] = d_k, \qquad \text{so } \text{std} = \sqrt{d_k}$$

With $d_k = 64$ the logits swing over roughly $\pm 8$. Feed that to softmax:

$$e^{8} = 2981,\quad e^{0} = 1,\quad e^{-8} = 0.000335$$ $$\text{softmax}([8, 0, -8]) = [0.9997,\ 0.00034,\ \approx 0]$$

That is **effectively one-hot** — the softness is gone, and we're back to hard lookup.

**Why saturation is fatal.** The Jacobian of softmax is

$$\frac{\partial p_i}{\partial s_j} = p_i(\delta_{ij} - p_j), \qquad \delta_{ij} = 1 \text{ if } i=j \text{ else } 0$$

- $p_i \to 1$: term becomes $1 \cdot (1-1) = 0$
- $p_i \to 0$: term becomes $0 \cdot (\dots) = 0$

Either way **the gradient dies**. Fix:

$$\text{Var}\left[\frac{q \cdot k}{\sqrt{d_k}}\right] = \frac{d_k}{(\sqrt{d_k})^2} = 1$$

Variance is now 1 **regardless of $d_k$** → softmax stays in the zone where gradients flow.

### 6. Where do Q, K, V come from?

In self-attention everything comes from the _same_ sequence — each token is both asker and asked. So we build three roles from one input using three matrices:

$$Q = XW^Q, \qquad K = XW^K, \qquad V = XW^V$$

- $n$ = number of tokens
- $d$ = model dimension (numbers per token vector)
- $d_k$ = key dimension **per head**
- $X$ = all token vectors from the input ($n \times d$)
- $W^Q, W^K \in \mathbb{R}^{d \times d_k}$, $W^V \in \mathbb{R}^{d \times d_v}$ — learnable
- $Q$ = what we're looking for ($n \times d_k$)
- $K$ = what we have on offer ($n \times d_k$)
- $V$ = what each entry actually contains ($n \times d_v$)

> **Correction to an earlier intuition of mine.** $d_k$ is _not_ "the number of distinct key meanings" versus $d$ being "the number of word types". $d_k$ is just the width of the per-head subspace, typically $d/h$, chosen so that $h$ heads together cost the same as one full-width head. It's a capacity/efficiency knob, not a semantic one. The structural meaning is: **the QK matrix is forced to be low-rank**, which is a deliberate inductive bias (see Multi-Head).

**Why three separate projections?** Because a token plays different roles in different moments.

Sentence: _"Cat eats fish"_

- "eats" as a **query**: "I'm a verb, I'm looking for who does the action."
- "cat" as a **key**: "I'm a noun, I'm animate."
- "cat" as a **value**: "actual content about cats — animal, four legs, …"

One shared vector could not distinguish "what I'm looking for" from "what I am".

### 7. Full formula

$$\boxed{\ \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V\ }$$

Read piece by piece:

1. $QK^\top$ — every query dotted with every key → an $n \times n$ table. ($\top$ = transpose, so the shapes line up.)
2. $\div \sqrt{d_k}$ — rescale so softmax doesn't saturate.
3. $\text{softmax}$ — each **row** becomes proportions summing to 1.
4. $\times V$ — use those proportions to average the values.

Per-token form:

$$z_i = \sum_{j=1}^{n} \alpha_{ij}, v_j, \qquad \alpha_{ij} = \frac{\exp\left(q_i \cdot k_j / \sqrt{d_k}\right)}{\sum_{j'} \exp\left(q_i \cdot k_{j'} / \sqrt{d_k}\right)}$$

- $i$ = position of the querying token
- $j$ = position of the attended token
- $\alpha_{ij}$ = attention weight from $i$ to $j$ (what fraction of its attention $i$ spends on $j$)
- $\alpha_{ij}v_j$ = the contribution $j$ makes to $i$
- $z_i$ = output for token $i$ (sum of all contributions)

**Reading the $n \times n$ matrix:** rows = who is asking, columns = who is being asked. Cell $(2,3)$ = how much token 2 attends to token 3.

### 8. Worked example

Sentence "Cat Eats Fish" ($n=3$). For simplicity $d_k = d_v = 2$. Computing the output of token 2, "Eats".

**Step 1** — query of "Eats" and keys of all tokens:

$$q_2 = [1,\ 0] \qquad k_1 = [1,\ 0],\quad k_2 = [0,\ 1],\quad k_3 = [0.7,\ 0.7]$$

**Step 2** — dot products:

$$q_2 \cdot k_1 = 1, \qquad q_2 \cdot k_2 = 0, \qquad q_2 \cdot k_3 = 0.7$$

**Step 3** — divide by $\sqrt{d_k} = \sqrt{2} \approx 1.414$:

$$[0.707,\quad 0,\quad 0.495]$$

**Step 4** — softmax:

$$e^{0.707} = 2.028,\quad e^{0} = 1,\quad e^{0.495} = 1.640 \qquad \text{total} = 4.668$$ $$\alpha_2 = [0.43,\quad 0.21,\quad 0.35]$$

"Eats" pays 43% attention to "Cat", 21% to itself, 35% to "Fish".

**Step 5** — weighted sum of values ($v_1 = [10,0],\ v_2 = [0,10],\ v_3 = [5,5]$):

$$z_2 = 0.43[10,0] + 0.21[0,10] + 0.35[5,5] = [6.1,\ 3.9]$$

**Done.** This is the new vector for "Eats", now carrying contextual information from "Cat" and "Fish".

### 9. Two consequences that matter later

**(a) The output is a convex combination.** $\alpha_{ij} \ge 0$ and $\sum_j \alpha_{ij} = 1$, so $z_i$ always lies inside the convex hull of ${v_j}$. **Attention can only move and blend information — it cannot create anything outside what already exists.** (This is exactly why the FFN is mandatory. See Rank Collapse.)

**(b) Everything is parallel.** $z_1, z_2, z_3$ don't wait on each other. This is the whole point of the architecture.

### 10. Theoretical views of attention (optional, but they pay off)

**Attention is a kernel smoother.** Compare the Nadaraya–Watson estimator:

$$\hat f(x) = \frac{\sum_j K(x, x_j), y_j}{\sum_j K(x, x_j)}$$

Attention _is_ this, with $K(q,k) = \exp(q\cdot k/\sqrt{d_k})$ — an asymmetric exponential kernel. So attention is **non-parametric regression that learns its own metric.**

**QK circuit / OV circuit** (Elhage et al., _A Mathematical Framework for Transformer Circuits_). Note that

$$q_i \cdot k_j = x_i W^Q (W^K)^\top x_j^\top = x_i, \underbrace{W^{QK}}_{\text{rank} \le d_k}, x_j^\top$$

$W^Q$ and $W^K$ **never appear separately** — only their product. Same for $W^V W^O = W^{OV}$.

- **QK circuit** decides **where to read from**
- **OV circuit** decides **what to move**

Splitting each into two thin matrices is an efficiency choice ($2dd_k$ parameters instead of $d^2$), not an expressivity one.

**Attention moves information; FFN processes it.** In the residual-stream view, an attention layer reads from other positions and writes back at its own position; the FFN works entirely within one position. This division of labour is the foundation of all interpretability work.

---

# Multi-Head Attention

## The problem with one head

One softmax = **one distribution per query** = the token can focus one way only.

Real language needs several at once. _"The **student** the teacher praised yesterday, **he** felt happy."_ — "he" must simultaneously resolve:

1. who "he" is → look at "student" (coreference)
2. what its verb is → look at "felt" (syntax)
3. what came just before → look at "yesterday" (position)

## Why "just attend to both" doesn't work

**Averaging destroys information.** If one head must attend equally to two places:

$$v_{\text{cat}} = [10,0], \quad v_{\text{fish}} = [0,10] \implies z = 0.5[10,0] + 0.5[0,10] = [5,5]$$

$[5,5]$ is **neither cat nor fish**. The fact that there were two distinct things is gone.

With two heads: head 1 → $[10,0]$, head 2 → $[0,10]$, concatenated → $[10,0,0,10]$. **Both survive.**

## Mechanism

Don't add width — **split** the existing width. With $d = 512$, $h = 8$:

$$d_k = d_v = d/h = 64$$

Each head has its **own** $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{512 \times 64}$, so each can learn a different relation.

$$\text{head}_i = \text{Attention}(XW_i^Q,\ XW_i^K,\ XW_i^V) \in \mathbb{R}^{n \times 64}$$ $$\text{MHA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h), W^O$$

Concat gives $n \times 512$ again ($8 \times 64 = 512$), then $W^O \in \mathbb{R}^{512 \times 512}$.

## What $W^O$ is actually for

Without $W^O$, head 1's output lives only in dimensions 1–64, head 2's in 65–128 — **they can never mix**. $W^O$ lets information from any head land in any dimension. It turns "8 separate results" into "1 result combining 8 viewpoints."

## Why it's free

|                 | one head at full width | 8 heads of 64                         |
| --------------- | ---------------------- | ------------------------------------- |
| $W^Q$           | $512\times512$         | $8\times(512\times64) = 512\times512$ |
| $W^K, W^V, W^O$ | same                   | same                                  |
| **total**       | $4d^2$                 | $4d^2$                                |

Identical parameter count and identical FLOPs. Multiple viewpoints cost nothing.

## Theory notes

**Rank as inductive bias.** Forcing $d_k = 64$ means each head's QK matrix has **rank ≤ 64**. (Rank = how many dimensions a matrix can actually reach.) This is deliberately limiting capacity so each head must specialize instead of trying to catch everything — and empirically it helps more than it hurts.

**Head pruning** (Michel et al., _Are Sixteen Heads Really Better than One?_): after training, many heads can be deleted with almost no loss. But training _with_ few heads is much worse. **Many heads are necessary for learning, not for running.**

---

# Masking

## Causal mask — the point

The model's job is to predict the next token. If, during training, position "Eats" can see "Fish", it just copies the answer — loss ≈ 0, learned nothing. At inference there is no future to copy from, so it collapses. This is **information leakage**.

Fix: add a mask to the scores **before** softmax.

$$M_{ij} = \begin{cases} 0 & j \le i \quad \text{(past and self — allow)} \ -\infty & j > i \quad \text{(future — block)} \end{cases}$$

$$A = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)$$

**Why $-\infty$ works:** $e^{-\infty} = 0$, so those weights become exactly 0. And softmax still normalizes the survivors to sum to 1 automatically — nothing extra to do.

> In code use $-10^9$ or the dtype's min, not literal `-inf`, because $-\infty \times 0 = \text{NaN}$ will destroy training.

## Concrete

$$\frac{QK^\top}{\sqrt{d_k}} = \begin{bmatrix} 1.2 & 0.5 & 0.8 \ 0.3 & 1.5 & 0.9 \ 0.7 & 0.4 & 1.1 \end{bmatrix} + \begin{bmatrix} 0 & -\infty & -\infty \ 0 & 0 & -\infty \ 0 & 0 & 0 \end{bmatrix}$$

After row-wise softmax:

$$A = \begin{bmatrix} 1.00 & 0 & 0 \ 0.23 & 0.77 & 0 \ 0.34 & 0.25 & 0.41 \end{bmatrix}$$

Lower-triangular, each row still sums to 1.

## Why this makes training fast

Feed "Cat eats fish big fat" (5 tokens) **once** and get:

- position 1 predicts "eats" (sees only "Cat")
- position 2 predicts "fish"
- position 3 predicts "big"
- position 4 predicts "fat"

**4 training signals from one forward pass, all computed simultaneously.** RNNs need 5 serial rounds for the same thing.

Feeding ground truth as input (rather than the model's own output) is called **teacher forcing**.

> **Side effect — exposure bias.** During training the model only ever eats correct human text. At inference it eats its own output. One bad token puts it in a distribution it never saw, and errors compound.

## Padding mask (different thing)

GPUs want rectangles, but sentences in a batch differ in length. Pad short ones with `<pad>` and **mask those positions**, or real tokens will attend to garbage.

---

# Position Encoding

## The problem is severe

Look at the attention formula again:

$$z_i = \sum_j \alpha_{ij} v_j, \qquad \alpha_{ij} \propto \exp(q_i \cdot k_j/\sqrt{d_k})$$

**Nowhere does position appear.** The indices $i, j$ are just names. So

$$\text{"Cat eats fish"} \quad \text{and} \quad \text{"Fish eats cat"}$$

produce **identical outputs** (just reordered). The model cannot tell who eats whom.

**Permutation-equivariant**: $f(\text{Permute}(X)) = \text{Permute}(f(X))$. Self-attention is this by construction — it sees a _bag_ of tokens, not a sequence.

## Why naive fixes fail

- **Add the raw index** (0,1,2,…): unbounded. Position 5000 swamps embeddings of scale ~1. And a model trained to 512 has never seen 513.
- **Normalize to [0,1]**: "distance 0.1" means 1 token in a short sentence and 100 in a long one. **Not consistent.**

## Sinusoidal (original paper)

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \qquad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

- $pos$ = position in the sequence (0, 1, 2, …)
- $i$ = index of the **dimension pair** ($i = 0 \dots d/2 - 1$)
- even dims use sin, odd dims use cos
- 10000 = chosen constant controlling the frequency range

The denominator varies with $i$:

- $i=0$: denominator $=1$ → $\sin(pos)$, **fast**, wavelength $2\pi \approx 6.3$
- $i = d/2-1$: denominator $\approx 10000$ → **very slow**, wavelength $\approx 62{,}800$

**Mental model: a car odometer.**

```
ones digit   fast   0 1 2 3 4 5 6 7 8 9 0 1 2 ...
tens digit   slower 0 0 0 0 0 0 0 0 0 0 1 1 1 ...
hundreds     slow   0 0 0 0 0 0 0 0 0 0 0 0 0 ...
```

Fast digits say "where exactly nearby", slow digits say "which region of the sequence". Together they identify position uniquely, and every value stays in $[-1,1]$.

**Key property:** for any offset $k$,

$$PE_{pos+k} = M_k \cdot PE_{pos}$$

where $M_k$ is a rotation matrix depending on $k$ **only, not on $pos$**. So "5 tokens apart" looks the same everywhere → relative position is learnable in principle.

## Learned absolute (BERT, GPT-2)

Drop the formula, use a trained table $P \in \mathbb{R}^{n_{\max} \times d}$.

- Simpler, often slightly better inside the trained range.
- **Hard wall**: one token past $n_{\max}$ and there is no row to fetch.

## RoPE (what everything uses now)

Don't add — **rotate**. Pair up dimensions, treat each pair as a point in 2D, rotate by an angle proportional to position.

$$R_m = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \ \sin(m\theta) & \cos(m\theta) \end{bmatrix}$$

Applied to $q$ and $k$ only — **not** to $v$, not to the input. Then:

$$\langle R_m q,\ R_n k\rangle = q^\top R_m^\top R_n k = q^\top R_{n-m}, k$$

(using $R_m^\top R_n = R_{n-m}$: rotate back by $m$, forward by $n$, net $n-m$)

**The inner product depends only on $n - m$ — the distance.**

Advantages:

1. Relative position for free, structurally, not hoped-for
2. **Zero extra parameters**
3. Injected at **every layer**, unlike additive PE which fades with depth
4. Rotation preserves norm, so it doesn't disturb score scales

## ALiBi

Even simpler — don't touch $q,k$, just penalize distance directly:

$$\text{score}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}} - m,|i-j|$$

with $m$ a fixed per-head constant. "Further = penalized more." Extrapolates very well because the rule holds at any distance never seen in training.

> **Insight I got wrong at first.** The fixed context window is _not_ caused by "distance being a fixed value". RoPE is already relative. The real limit is **unseen rotation angles** — train to 8k, meet 100k, and the angles land in a regime never practiced. Hence the "dynamic distance" family of fixes: **Position Interpolation** (divide positions by $k$ to squeeze 32k into an 8k-looking range), **NTK-aware / YaRN** (squeeze low frequencies hard, keep high frequencies intact), **ALiBi** (no angles at all). The _other_ two real limits are KV-cache memory and $O(n^2 d)$ compute — neither has anything to do with position encoding.

---

# Feed-Forward Network

## Why attention alone is not enough

**Reason 1 — the value path is linear.** $v_j = x_j W^V$, and $z_i = \sum_j \alpha_{ij} v_j$ is a weighted sum. Stacking linear on linear collapses: $(xA)B = x(AB)$. Ninety-six attention layers still reduce to something with limited expressive power along the content path. (The $\alpha$ are nonlinear in $x$ via softmax, but they only decide _how to mix_, not how to transform content.)

**Reason 2 — rank collapse.** Because output is a convex combination, repeated attention is a **contraction**:

```
layer 1:  cat=[10,0]     eats=[0,10]     fish=[5,5]      (distinct)
layer 2:  cat=[6,4]      eats=[4,6]      fish=[5,5]      (converging)
layer 3:  cat=[5.2,4.8]  eats=[4.8,5.2]  fish=[5,5]      (nearly identical)
...
layer N:  everything = [5,5]                             (dead)
```

Like stirring paint — everything ends up the same brown. Dong et al. (2021) proved pure self-attention converges to **rank-1 doubly exponentially** in depth. Details under Theory.

## The FFN

$$\text{FFN}(x) = W_2, \sigma(W_1 x + b_1) + b_2$$

| symbol     | meaning                       | shape             |
| ---------- | ----------------------------- | ----------------- |
| $x$        | vector of **one token**       | $d$               |
| $W_1$      | expand                        | $d \times d_{ff}$ |
| $b_1, b_2$ | biases                        | $d_{ff}$, $d$     |
| $\sigma$   | activation — the nonlinearity | —                 |
| $W_2$      | contract                      | $d_{ff} \times d$ |
| $d_{ff}$   | inner dim, usually $4d$       | —                 |

**ReLU** is the simplest $\sigma$: $\text{ReLU}(x) = \max(0,x)$, e.g. $[-2,3,-0.5,7] \to [0,3,0,7]$.

Cutting negatives looks trivial but it is a **decision** — which channels open, which close. Linear maps cannot do that. This is what makes depth actually buy expressivity.

## "Position-wise" means

Same $W_1, W_2$ applied to **every position independently**, with no communication between positions. (Equivalent to a $1\times1$ convolution.)

**This is the core division of labour in a Transformer:**

|               | job                                         | direction  |
| ------------- | ------------------------------------------- | ---------- |
| **Attention** | move/blend information **across positions** | horizontal |
| **FFN**       | process information **within one position** | vertical   |

Attention fetches; FFN thinks.

## Why expand then contract ($512 \to 2048 \to 512$)

More room to separate patterns, and 2048 independent on/off decisions instead of 512. Then compress back to $d$ to pass along.

## FFN is where knowledge lives

Geva et al. showed the FFN behaves like a key-value memory:

- rows of $W_1$ = **pattern detectors** ("this input is about a country's capital")
- ReLU = **filter** for which detectors actually fired
- columns of $W_2$ = **content written back** ("→ Paris")

Facts like "Paris is the capital of France" are **not in attention** — they are in the FFN.

Parameter counts agree:

$$\text{FFN} = \underbrace{d \cdot 4d}_{W_1} + \underbrace{4d \cdot d}_{W_2} = 8d^2 \qquad\text{vs}\qquad \text{Attention} = 4d^2$$

**~2/3 of the model is FFN** — reasonable, if it is the knowledge store.

## Gated variants — SwiGLU

$$\text{FFN}_{\text{SwiGLU}}(x) = W_2\big(\underbrace{\text{Swish}(W_1 x)}_{\text{gate}} \odot \underbrace{W_3 x}_{\text{content}}\big)$$

- $\odot$ = element-wise product
- $\text{Swish}(x) = x \cdot \text{sigmoid}(x)$ — like ReLU but smooth, so gradients flow better

**Gating idea:** compute two streams, one is content, one is a valve saying how much of it passes (0 = closed, 1 = open). More flexible than a hard cut at 0.

Three matrices, so $d_{ff}$ drops to $\tfrac{8}{3}d$ to keep the parameter count equal ($3 \times \tfrac{8}{3}d^2 = 8d^2$). Consistently better than ReLU/GELU; Shazeer's paper openly admits nobody knows why and credits "divine benevolence".

---

# Residual Connection

$$x \leftarrow x + \text{Sublayer}(x)$$

Wrapped around **every** sublayer (attention and FFN). Looks trivial; it's what makes depth possible.

## Reason 1 — gradient highway

$$\frac{\partial}{\partial x}\big(x + F(x)\big) = I + \frac{\partial F}{\partial x}$$

$I$ = identity matrix (the matrix version of 1). Even if $\partial F/\partial x \approx 0$, the total is $\approx I$, not 0. Chain 96 of them:

$$(I + J_{96})(I + J_{95}) \cdots (I + J_1)$$

An identity path always survives, so gradients reach the bottom layers directly.

## Reason 2 — it prevents rank collapse

$$x_{\text{new}} = \underbrace{x_{\text{old}}}_{\text{my identity}} + \underbrace{\text{Attention}(x)}_{\text{what I fetched}}$$

Even if attention returns the same average for everyone, $x_{\text{old}}$ still differs → no collapse to rank-1.

## Residual stream — the most useful mental model

Don't picture data _flowing through_ layers. Picture:

> A **bus of width $d$** running straight from input to output. Each layer **reads** part of it, computes, and **adds** its result back.

Because everything is addition (not replacement), information **accumulates**.

Consequence: **layer 5 and layer 40 can communicate directly** — layer 5 writes into some subspace, layer 40 reads that subspace. They negotiate "channels" among themselves during training. This is the foundation of mechanistic interpretability.

---

# Normalization

## The problem

Adding and adding along the residual stream makes magnitudes grow, unevenly across dimensions. Then softmax saturates, activations jam, gradients die.

## LayerNorm

$$\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

| symbol                                     | meaning                                      |
| ------------------------------------------ | -------------------------------------------- |
| $\mu = \frac{1}{d}\sum_i x_i$              | mean over all channels **of this one token** |
| $\sigma^2 = \frac{1}{d}\sum_i (x_i-\mu)^2$ | variance                                     |
| $\epsilon \approx 10^{-5}$                 | prevents division by zero                    |
| $\gamma, \beta$                            | **learnable** vectors of size $d$            |

Subtract the mean → divide by std → let the model scale it back however it wants. $\gamma,\beta$ exist because forcing mean 0 / var 1 always is too rigid.

**Critical: statistics are computed across the feature dimension of a single token, not across the batch.**

```
token "cat"  = [2.1, -0.5, 3.8, 0.2, ...]  <- normalize these numbers
token "eats" = [1.0,  4.2, -1.1, 0.9, ...] <- separately, these
```

## Why not BatchNorm

1. **Variable lengths** — position 500 may exist in only 2 of 32 sequences; statistics from 2 samples are garbage.
2. **Inference is often batch = 1** — nothing to average over; you'd need stored training statistics that may not match.
3. LayerNorm doesn't depend on the batch at all, so **train and inference behave identically**.

## RMSNorm (LLaMA and later)

$$\text{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}}$$

Drops mean subtraction (and $\beta$). Faster, same quality — which tells us **the benefit of normalization is re-scaling, not re-centering.**

## Post-LN vs Pre-LN — bigger deal than it looks

**Post-LN (original 2017):** $x \leftarrow \text{LN}(x + \text{Sublayer}(x))$

**Pre-LN (everything since ~2020):** $x \leftarrow x + \text{Sublayer}(\text{LN}(x))$

Only the parentheses move. Look at the residual path:

- **Pre-LN**: $x \to x \to x \to \dots$ is a **pure identity path from input to output, nothing in the way.** LN sits on the side branch. Perfect gradient highway.
- **Post-LN**: every layer inserts an LN into the highway → gradients pass through 96 LNs → highway broken.

Xiong et al. (2020): Post-LN has gradients distributed unevenly across depth → **diverges without warmup**. Pre-LN has uniform $O(1/\sqrt{L})$ scale → trains without warmup, tolerates higher LR, goes deeper.

---

# Prediction

## First, clear up a misconception

**The output of a block is not "relationships between tokens."** The relationships are the $\alpha_{ij}$, and they are **used and discarded inside the block**:

$$z_i = \sum_j \underbrace{\alpha_{ij}}_{\text{intermediate, thrown away}} v_j$$

What comes out is $z_i$, living in the same $d$-dimensional space as the input:

$$z_i = \text{"token } i\text{, rewritten now that it knows its context"}$$

```
into block:  "bank" = a generic vector meaning bank (money? river? unknown)
out of block: "bank" = a vector meaning "river bank"
              (because it pulled information from "river")
```

$n$ tokens in, $n$ tokens out — each one smarter.

## Which vector predicts what

**The vector at position $t$ predicts token $t+1$.**

```
input:   [Cat]  [eats] [fish] [big]
           v      v      v      v
output:   h1     h2     h3     h4
           v      v      v      v
predict: "eats" "fish" "big"  "fat"
```

The causal mask guarantees $h_3$ only saw tokens 1–3, so asking it for token 4 is a fair question.

- **Training**: use **all** positions at once → $n$ lessons per forward pass.
- **Inference**: only the **last** one matters; the rest are computed anyway because the last one attends to them.

## Unembedding

$$\text{logits} = W_{\text{out}}, h_t, \qquad W_{\text{out}} \in \mathbb{R}^{|V| \times d}$$ $$p = \text{softmax}(\text{logits})$$

One row per vocabulary word; dot each row with $h_t$ to get that word's score; softmax over all $|V|$ scores.

**Logits** = raw pre-softmax scores; can be any real number.

## The nicest intuition — it's attention again

With **weight tying**, $W_{\text{out}} = E^\top$: reuse the input embedding table for output. Then

$$\text{logit}_w = h_t \cdot E_w$$

> **Next-token prediction = asking which of the 50,000 word embeddings points most in the same direction as my final vector.**

The whole 96-layer stack has exactly one job: **rotate the last position's vector until it points at the right word.** Reading the answer is then just a dot product.

(Structurally identical to attention: $h_t$ is the query, all word embeddings are keys, softmax is the same.)

Weight tying also saves $|V| \cdot d$ parameters (~400M for $|V|=100k$, $d=4096$) and improves quality by forcing input and output spaces to agree.

## Why does $h_t$ contain the answer?

**Nobody designed it to — the loss forced it.**

$$\mathcal{L} = -\frac{1}{n}\sum_{t=1}^{n} \log p_\theta(y_t \mid y_{<t})$$

Gradients flow back from the loss through $W_{\text{out}}$, through all 96 blocks, into the embeddings, adjusting everything so $h_t$ points more toward the correct word. Repeat a few trillion times.

**Everything discussed — attention, FFN, multi-head, depth — is just a structure flexible enough for gradient descent to sculpt into this.** Nobody told the model what a subject or a verb is.

## Turning one word into text

$$\text{"I want to eat"} \to \text{"ice"} \to \text{"I want to eat ice"} \to \text{"cream"} \to \dots$$

**Autoregressive decoding.** This is why generation cannot be parallelized (unlike training). It is the latency bottleneck for everything.

| strategy               | what it does                                                                   |
| ---------------------- | ------------------------------------------------------------------------------ |
| **Greedy**             | always take the argmax — safe, but bland and repetitive                        |
| **Temperature** $\tau$ | divide logits by $\tau$ before softmax; $\tau<1$ sharper, $\tau>1$ more random |
| **Top-k**              | sample from the best $k$                                                       |
| **Top-p (nucleus)**    | sample from the smallest set whose cumulative $p$ reaches e.g. 0.9             |
| **Beam search**        | keep several partial hypotheses; good when there's one right answer (MT)       |

Temperature limits: $\tau \to 0$ = greedy; $\tau \to \infty$ = uniform.

---

# Multi Block

## Terminology — this is where confusion comes from

| word              | what it is                                   | GPT-3         |
| ----------------- | -------------------------------------------- | ------------- |
| **head**          | one head inside one attention module         | 96            |
| **sublayer**      | the whole attention module, or the whole FFN | 2 per block   |
| **block / layer** | attention + FFN together                     | **96 blocks** |
| **model**         | $L$ blocks stacked                           | —             |

"GPT-3 has 96 layers" means 96 **blocks**.

## Two axes that behave completely differently

**Horizontal — heads (parallel)**

```
        input X (the same X)
       /   |   |   \
    head1 head2 head3 head4     <- simultaneous, blind to each other
       \   |   |   /
        concat, then W^O
```

Every head gets the **identical input**; head 3 has no idea what head 1 computed; they meet exactly once, at the end. **All of this is one layer.**

**Vertical — blocks (serial)**

```
block 1 -> block 2 -> block 3 -> ... -> block 96
```

The output of the lower block is the input of the upper one. Block 2 works on _already-processed_ representations, not raw text.

## What flows between blocks: $z$, never $\alpha$

$\alpha$ never leaves the block. It is born and dies inside.

```
block l receives:  x   (n x d)
    |
    compute alpha fresh from x        <- born here
    z = sum(alpha * v)                <- used here
    x = x + z                         <- alpha dies here, nobody stores it
    x = x + FFN(LN(x))
    |
block l+1 receives: x   (n x d)
```

$\alpha$ is $n \times n$ — a different shape from $x$ entirely. It couldn't be passed even if you wanted to.

## $\alpha$ is recomputed every block and looks nothing like the previous one

$$\alpha^{(2)} = \text{softmax}\left(\frac{(x^{(1)}W_2^Q)(x^{(1)}W_2^K)^\top}{\sqrt{d_k}}\right)$$

Both the weights and the input are new. So each block **asks a different question**:

```
block 1  alpha:  "what token is immediately before me?"      (previous-token head)
block 4  alpha:  "which noun is the subject of my verb?"
block 20 alpha:  "who does 'he' refer to in this paragraph?"
block 60 alpha:  "has this topic been mentioned before?"
```

Not the same question refined — **different questions**, askable only because earlier blocks prepared the information.

## What actually stacks: the residual stream

$$x^{(1)} = x^{(0)} + \text{Attn}^{(1)} + \text{FFN}^{(1)}$$ $$x^{(2)} = x^{(1)} + \text{Attn}^{(2)} + \text{FFN}^{(2)}$$

Unrolled:

$$\boxed{\ x^{(L)} = x^{(0)} + \sum_{l=1}^{L}\left(\text{Attn}^{(l)} + \text{FFN}^{(l)}\right)\ }$$

**The final vector is the initial embedding plus everything every block wrote.** No replacement, no deletion — 192 additive contributions (96 attention + 96 FFN).

|                         | analogy                                                       |
| ----------------------- | ------------------------------------------------------------- |
| $\alpha$                | **opening a book to the page you need** — done, then closed   |
| $z$ added to the stream | **what you write in your notebook** — permanent, accumulating |
| $x$ between blocks      | **the whole notebook**, passed to the next round              |

## Shapes make stacking trivial

$$\underbrace{n \times d}_{\text{in}} ;\longrightarrow; \text{block} ;\longrightarrow; \underbrace{n \times d}_{\text{out}}$$

Like Lego bricks with matching connectors. **The shape never changes — the content gets denser.**

```
before block 1:  "cat" = "the word cat"
after block 1:   "cat" = "the word cat + I am the subject here"
after block 10:  "cat" = "subject cat + animate + currently eating"
after block 96:  "cat" = "...everything needed to predict the next token"
```

## What depth actually buys: reasoning hops

**One attention layer = one hop of information movement.**

> \_"The capital of the country where the Eiffel Tower is located is _\_\_"_

Requires two hops: Eiffel Tower → France → Paris.

**One layer cannot do this.** When the last token builds its query $q = xW^Q$, it builds it from what it knows _right now_ — "I'm the word 'is', awaiting an answer." It doesn't know about France yet. **How can it search for France-related information when it doesn't yet know the topic is France?**

**Two layers can:**

- **Block 1**: attention pulls from "Eiffel Tower"; FFN maps Eiffel Tower → France. Now "France" is written in the residual stream at that position.
- **Block 2**: $W^Q$ builds a query from a stream that _now contains France_ → can search France-related information. FFN maps France + capital → Paris.

> **Depth = number of reasoning steps available in one forward pass.** More heads in one layer cannot substitute, because all heads see the same input and none sees another's output.

## Real evidence: induction heads need exactly two layers

Behaviour: seeing `[A][B] ... [A]`, predict `[B]`.

- **Shallow layer — previous-token head**: every token copies information about the token before it into its own stream. So the first "kind" records "the token before me was John".
- **Deep layer — induction head**: the second "John" queries _"who has a note saying the token before me was John?"_ → finds "kind" → copies it.

Impossible in one layer: the note must already be written before it can be searched.

The loss curve shows a **visible phase change** exactly when this circuit forms.

## Depth specialization (empirical trend, not a law)

| depth           | typically does                                                       |
| --------------- | -------------------------------------------------------------------- |
| early (1–20%)   | assemble subword pieces into words, local syntax, position           |
| middle (20–70%) | subject–verb links, coreference, fact retrieval, multi-hop reasoning |
| late (70–100%)  | consolidate everything into the next-token prediction                |

## Logit lens — watching the answer form

Apply $W_{\text{out}}$ to the residual stream at an _intermediate_ layer to see what the model would say if it stopped there.

Prompt: \_"The Eiffel Tower is located in the country of _\_\_"_

| layer | top predictions                                       |
| ----- | ----------------------------------------------------- |
| 10    | `the`, `a`, `this` — noise                            |
| 30    | `Europe`, `city`, `various` — knows the answer _type_ |
| 60    | `France` 0.31, `Italy` 0.18, `Germany` 0.09           |
| 96    | `France` 0.94                                         |

**The answer doesn't appear at the end — it accumulates.** $x^{(0)}$ points at "country"; 192 added contributions rotate it until it points at "France"; $W_{\text{out}}$ just reads the direction.

## Are the weights shared across blocks?

**No — every block has its own full parameter set.** $W_1^Q \ne W_2^Q \ne \dots \ne W_{96}^Q$. That is where 175B parameters come from; sharing would leave ~1.8B.

But sharing has been tried and **works**:

| model                                  | what it does                                             |
| -------------------------------------- | -------------------------------------------------------- |
| **ALBERT** (2019)                      | shares all layers in BERT — much smaller, usable, weaker |
| **Universal Transformer** (2018)       | one block looped, with adaptive halting                  |
| **Looped / recurrent-depth** (2024–25) | one core looped at inference to "think longer"           |

**Why not sharing wins:**

1. **Layers genuinely do different jobs** — induction heads appear at specific depths; logit lens shows layer 10 and layer 90 are in different modes; only early layers need to assemble subwords.
2. **FFN is the knowledge store** — sharing means one storage cabinet instead of 96. Scaling laws say loss depends on $N$; cutting $N$ by 96× lands you far up the curve.
3. **The killer argument: sharing saves memory but saves zero compute.**

```
not shared:  run 96 blocks  ->  FLOPs = X,  params = 175B
shared:      run 96 loops   ->  FLOPs = X,  params = 1.8B
```

You still run all 96 rounds. **Same compute bill, dumber model.** (ALBERT's motivation was memory constraints specifically.)

4. **Residual-stream channel allocation requires different weights.** If all layers share weights, they all read and write the same subspaces, and cross-layer communication breaks.

**But the truth is in between.** "Layers as painters"-style work finds middle layers share a representation space — you can reorder, skip, or repeat them with graceful degradation. Layer pruning removes several deep layers cheaply. And **Universal Transformer beats vanilla on algorithmic tasks** (copy, sort, expression evaluation), where repeating the same step _is_ the answer.

> Sharing suits genuinely iterative problems; not sharing suits problems where each step does something different. Natural language is the latter. This idea is now returning as **looped / latent recurrent depth** — see Latent Chain of Thought.

## Full block structures

**Encoder block** (BERT — no causal mask, every token sees everything):

```
x = x + MHA(LN(x))
x = x + FFN(LN(x))
```

**Decoder block, encoder-decoder (translation):**

```
x = x + MaskedMHA(LN(x))              # look at what I'm writing
x = x + CrossAttn(LN(x), enc_out)     # look at the source
x = x + FFN(LN(x))
```

**Cross-attention** differs from self-attention only in where Q and K,V come from:

$$Q = x_{\text{decoder}} W^Q, \qquad K = \text{enc_out}, W^K, \qquad V = \text{enc_out}, W^V$$

"What do I (writing the translation) need? Go find it in the source." This is where information crosses languages, and it generalizes Bahdanau-style attention.

**Decoder-only (GPT family)** — drop cross-attention entirely and concatenate everything into one sequence:

```
"Translate to English: cat eats fish -> The cat eats fish"
```

The causal mask handles who sees what. Simpler, and adapts to any task.

## Parameter counting

Per block: $\underbrace{4d^2}_{\text{attention}} + \underbrace{8d^2}_{\text{FFN}} = 12d^2$

Whole model: $\approx 12Ld^2 + |V|d$

Check against GPT-3 ($d = 12288$, $L = 96$):

$$12 \times 96 \times 12288^2 \approx 1.74\times10^{11} = 174\text{B} \quad\checkmark$$

---

# Training

## Where the data comes from — self-supervision

The task is "predict the next token", so **the label is already inside the text**.

```
raw:     "cat eats fish big fat"

input:   [cat]  [eats] [fish] [big]
label:   [eats] [fish] [big]  [fat]      <- shifted by one
```

**Self-supervised learning.** Every piece of text on the internet becomes training data with no human labelling. This is the only reason trillion-token scale is possible.

## The loop

```
1. forward pass      -> predict at every position
2. compute loss      -> compare with the true next token
3. backpropagation   -> gradient for every parameter
4. optimizer step    -> nudge parameters
   (repeat 10^5 - 10^6 times)
```

## Loss

$$\mathcal{L} = -\frac{1}{n}\sum_{t=1}^{n} \log p_\theta(y_t \mid y_{<t})$$

| symbol                      | meaning                                                 |
| --------------------------- | ------------------------------------------------------- |
| $\theta$                    | all parameters                                          |
| $y_t$                       | the correct token at position $t$                       |
| $p_\theta(y_t \mid y_{<t})$ | probability the model assigned to the **correct** token |
| $\frac{1}{n}\sum$           | average over positions                                  |

**Why $-\log$:**

$$p = 1.0 \to 0 \qquad p = 0.5 \to 0.69 \qquad p = 0.1 \to 2.30 \qquad p = 0.001 \to 6.91 \qquad p \to 0 \Rightarrow \infty$$

**It punishes confident wrongness brutally.** Huge loss → huge gradient → large correction. A plain right/wrong signal couldn't distinguish "nearly right" from "absurd".

**Perplexity** $= e^{\mathcal{L}}$ — "how many options is the model effectively torn between". Lower is better.

Equivalent formulations (useful in exams):

- **= Maximum Likelihood Estimation** — find $\theta$ making the observed data most probable
- **= minimizing KL divergence** between the data distribution and the model distribution

## Backpropagation

**Gradient** $\frac{\partial \mathcal{L}}{\partial\theta_i}$ = "if I increase $\theta_i$ slightly, how does loss change?"

- positive → increasing hurts → **decrease** it
- negative → **increase** it

Backprop gets all of them at once via the **chain rule**, walking backwards from the loss:

$$\frac{\partial \mathcal{L}}{\partial x^{(l)}} = \frac{\partial \mathcal{L}}{\partial x^{(l+1)}} \cdot \frac{\partial x^{(l+1)}}{\partial x^{(l)}}$$

"Take the gradient at the layer above, multiply by this layer's local derivative, get the gradient below."

**Cost:** you must **keep every layer's activations** from the forward pass to use during the backward pass. That's why training uses far more VRAM than inference.

> This is the situation residual connections were designed for. $\partial(x+F(x))/\partial x = I + \partial F/\partial x$ — that $I$ is what lets gradients survive 96 layers.

## Optimizer

**Plain SGD:**

$$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$$

$\eta$ = **learning rate** = step size. Too large → overshoot, oscillate, explode. Too small → never finishes. The "stochastic" part: we estimate the loss from a random **batch**, not the whole dataset.

**Transformers barely train with SGD at all.** Two reasons:

**(a) Terrible condition number** — some directions of the loss landscape are very steep, others very flat. One step size cannot suit both.

**(b) Rare-token imbalance** — "the" appears millions of times, "mitochondria" rarely. Gradients for rare-token parameters are tiny, so SGD barely moves them.

**Adam** keeps two running statistics of the gradient:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \qquad \text{(momentum — average direction)}$$ $$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \qquad \text{(average squared magnitude)}$$ $$\theta \leftarrow \theta - \eta,\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}$$

- $m$ = a rolling ball; ignores per-batch noise
- $v$ = remembers how big this parameter's gradients usually are
- **dividing by $\sqrt{v}$ normalizes per parameter** → every parameter takes a comparably sized step

**This fixes (a) and (b) simultaneously.** Rare-token parameters have small gradients but also small $v$; the ratio comes out normal.

Typical: $\beta_1 = 0.9$, $\beta_2 = 0.95$–$0.999$. **AdamW** adds decoupled weight decay and is the current standard.

**Cost:** one $m$ and one $v$ per parameter → **3× the model size in memory** just for the optimizer.

## Learning rate schedule

Original:

$$\eta = d^{-0.5} \cdot \min\left(step^{-0.5},\ step \cdot warmup^{-1.5}\right)$$

Modern: linear warmup, then cosine decay.

```
lr
 |      /\
 |     /  \___
 |    /       \____
 |   /             \____
 +--/-------------------------> steps
    ^ warmup ends
```

**Why warmup:**

1. Adam's $v$ estimate is unreliable in the first steps; dividing by a bad $\sqrt{v}$ can produce a giant step.
2. Parameters are still random; gradients point in noisy directions; a big early step can do permanent damage.
3. Post-LN literally requires it (see Normalization). Pre-LN doesn't need it but still uses it for stability.

**Why decay:** near the minimum you need small steps to settle in rather than bouncing around.

## Regularization

**Overfitting** = memorizing the training set instead of learning.

| technique                              | what it does                                                                     |
| -------------------------------------- | -------------------------------------------------------------------------------- |
| **Dropout** ($p = 0.1$)                | randomly zero 10% of units during training → forces redundancy; off at inference |
| **Label smoothing** ($\epsilon = 0.1$) | target 0.9 instead of 1.0, spread 0.1 over others → less overconfidence          |
| **Weight decay**                       | gently pull parameters toward 0                                                  |
| **Gradient clipping**                  | rescale the gradient if its norm exceeds e.g. 1.0 → survives weird batches       |

**Label smoothing curiosity:** it makes **perplexity worse** but **BLEU better**. Less certainty at each step leaves room for better overall sequences during decoding.

> Modern large models often use dropout = 0 — with enough data there's essentially nothing to overfit.

## How much data — Chinchilla

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

$N$ = parameters, $D$ = training tokens, $E$ = irreducible floor.

**Compute-optimal:** $D \approx 20N$. A 70B model wants ~1.4T tokens.

**GPT-3 (175B params, 300B tokens) was badly off** — it should have had ~3.5T. Chinchilla (70B, 1.4T) beat Gopher (280B, 300B) on almost everything. See Theory for details and caveats.

**Scale reference:** training GPT-3 ≈ $3.14 \times 10^{23}$ FLOPs (rule of thumb: FLOPs $\approx 6ND$) — thousands of A100s for months.

## This is only stage one

Everything above is **pre-training**. The result is "very good at predicting the next token" — **not an assistant**.

Ask a base model _"What is the capital of Thailand?"_ and it may reply _"What is the capital of Japan? What is the capital of Laos?"_ — because on the internet, questions are often followed by more questions. It is behaving exactly as trained.

See **Post-Training** under More.

---

# Chain of Thought (CoT) and context window

## The phenomenon

Wei et al. (2022):

```
Direct:
  Q: A shop had 23 apples, used 20, bought 6 more. How many now?
  A: 27                                     <- wrong

CoT:
  A: Start with 23. Used 20, so 23-20 = 3.
     Bought 6, so 3+6 = 9. The answer is 9. <- right
```

On GSM8K, PaLM 540B went **17.9% → 56.9%** just by writing steps. Kojima et al. showed that appending _"Let's think step by step"_ gets similar gains with zero examples.

**The question: not one weight changed, and no information was added. Why is it better?**

Three layers of answer.

## Answer 1 — CoT buys compute (the main one)

**One forward pass has fixed depth.** Note the strange property:

$$\text{compute per token} = \text{always } 12Ld^2 \text{ FLOPs}$$

```
"2 + 2 = ?"                       -> 96 layers
"Prove sqrt(2) is irrational"     -> 96 layers (identical)
```

**There is no mechanism for "think longer when the problem is harder."** Unlike an RNN (loops with length) or a human.

**CoT turns the Transformer into a recurrent machine.** When a token is emitted, it is fed back as input — the intermediate result is stored **outside the model, in the context window** — and re-read on the next pass.

```
[96 layers] -> step 1 -+
                       v
[96 layers] -> step 2 -+
                       v
[96 layers] -> step 3 -+
                       v
[96 layers] -> answer

effective depth = 96 x (number of tokens)
```

| Turing machine  | LLM + CoT                |
| --------------- | ------------------------ |
| read/write head | forward pass             |
| memory tape     | **context window**       |
| number of steps | number of tokens written |

## Answer 2 — residual stream bandwidth

Everything a position carries must fit in $d$ dimensions and pass through 96 layers.

$$\text{direct answer:} \quad 1 \text{ position} \times d$$ $$\text{200-token CoT:} \quad 200 \text{ positions} \times d$$

**CoT is asking for more workspace** — scratch paper instead of mental arithmetic. And the intermediates are **pinned as discrete tokens**, not fuzzy values that might get overwritten deeper in the stack.

## Answer 3 — factorizing a hard distribution

$$p(a \mid q) = \sum_{r} \underbrace{p(r \mid q)}_{\text{write the steps}} \cdot \underbrace{p(a \mid r, q)}_{\text{read off the answer}}$$

Each factor is far easier. And it matches the training data: humans almost never write _"the answer is 9"_ with no work shown, so $p(\text{correct} \mid \text{problem} + \text{worked solution})$ is well trained while $p(\text{correct} \mid \text{problem})$ alone barely appears.

**CoT puts the model back in a region of distribution it has seen a lot of.**

## The experiment that separates the hypotheses

Competing explanations: (A) it's the _content_ of the text; (B) it's the extra _compute_.

**Pfau et al. (2024)** had models emit **meaningless filler** like `"........."` before answering — **and it helped on some tasks.** Those tokens carry no information.

**Direct evidence that compute alone has value**, though content clearly matters too (and training a model to use filler tokens is much harder than using real reasoning text).

## Why it's emergent at scale

Small models produce wrong steps and compound the error.

$$P(\text{whole chain correct}) = p^k$$

| $p$ per step | 10 steps |
| ------------ | -------- |
| 0.80         | 11%      |
| 0.90         | 35%      |
| 0.95         | 60%      |
| 0.99         | **90%**  |

**Below a reliability threshold, longer chains just add failure modes.** (This is where **self-consistency** helps — sample several chains and majority-vote; correct answers converge, wrong ones scatter.)

## The context window: same space, no separation

**There is no separate "CoT memory".** Tokens the model writes while thinking are appended to the same sequence as the prompt.

```
[system][prompt][ reasoning steps ][answer][ unused ]
|________________ one context window _______________|
```

|                          | user's prompt | model's own thinking tokens |
| ------------------------ | ------------- | --------------------------- |
| stored in KV cache       | yes           | yes                         |
| gets positional encoding | yes           | yes                         |
| same causal mask         | yes           | yes                         |
| visible to later tokens  | yes           | yes                         |

**Mechanically indistinguishable.** The model can only tell them apart via special tokens (`<|im_start|>assistant`, `<|thinking|>`) — which are themselves just learned tokens.

**Consequences:**

- CoT **consumes context budget**: 128k context, 20k prompt + 90k thinking → 18k left for the answer.
- CoT **inflates the KV cache** proportionally. This is why reasoning models are slow and expensive.

## What actually limits the context window

The architecture doesn't. Nothing in the equations fixes $n$ — $QK^\top$ works at any size.

| limit                   | detail                                                                               |
| ----------------------- | ------------------------------------------------------------------------------------ |
| **Positional encoding** | learned PE = hard wall at $n_{\max}$; RoPE/ALiBi = soft degradation on unseen angles |
| **Trained length**      | never practiced attending across 100k                                                |
| **KV cache memory**     | linear in $n$                                                                        |
| **FLOPs**               | $O(n^2 d)$ dominates once $n > d$                                                    |

**Advertised context ≠ usable context.** _"Lost in the Middle"_ (Liu et al. 2023) found a **U-shaped curve** — retrieval is good at the start and end of the context, and degrades badly in the middle.

## Two memory systems — keep them separate

|            | **weights**                            | **context**                          |
| ---------- | -------------------------------------- | ------------------------------------ |
| analogy    | long-term memory                       | working memory / scratch paper       |
| holds      | knowledge, skills, habits              | this conversation, current reasoning |
| written by | pre-training / fine-tuning (expensive) | just typing (free, instant)          |
| lifetime   | permanent                              | **gone when the session ends**       |
| size       | 175B parameters                        | 128k tokens                          |

**CoT writes to working memory only.** No matter how deeply it reasons this session, a new chat starts blank.

## Two flavours of CoT — don't conflate them

|                                             | touches weights?                 | touches architecture? |
| ------------------------------------------- | -------------------------------- | --------------------- |
| **CoT by prompting** ("think step by step") | **no**                           | no                    |
| **Reasoning model** (o1-style)              | **yes** — RL on existing weights | no                    |

The first is a _discovery_ that existing models can already do this if asked correctly. The second trains with RL on **verifiable final answers** and lets the model discover its own reasoning style — producing self-checking, backtracking, exploring alternatives.

## Limitations

**Unfaithfulness (Turpin et al. 2023) — the most worrying one.** Inject a bias into few-shot examples (e.g. the answer is always A). The model picks A, then **writes a plausible-sounding justification that never mentions the bias.**

> **The written chain may not be the computation that happened. It can be post-hoc rationalization.**

This is a serious problem for interpretability and safety: we cannot read CoT as a trustworthy log of thought.

**It doesn't help everything.** Sprague et al. (2024), meta-analysis over 100+ papers:

| task type                                             | CoT effect         |
| ----------------------------------------------------- | ------------------ |
| math, logic, symbolic                                 | **large gains**    |
| commonsense, factual knowledge, reading comprehension | little or negative |

Matches theory exactly: **CoT helps problems that need sequential computation.** Fact retrieval from the FFN needs no extra hops.

**Cost.** Generation is serial; a 500-token chain is 500× the latency and cost.

## The CoT family

| technique                       | what it does                                        |
| ------------------------------- | --------------------------------------------------- |
| **Scratchpad** (Nye 2021)       | ancestor — fine-tune to emit intermediate steps     |
| **Zero-shot CoT**               | just add "Let's think step by step"                 |
| **Self-consistency**            | sample many chains, majority vote                   |
| **Tree of Thoughts**            | branch and search with backtracking                 |
| **Filler tokens**               | meaningless tokens — proof that compute alone helps |
| **Looped transformer**          | loop blocks in latent space (see below)             |
| **Reasoning models** (o1-style) | RL-trained long chains + test-time compute scaling  |

## The third scaling axis

$$\text{performance} = f(\underbrace{N}_{\text{params}},\ \underbrace{D}_{\text{data}},\ \underbrace{T}_{\text{thinking tokens}})$$

All three trade off. A small model that thinks long can beat a big model that answers instantly. This is the basis of modern reasoning models.

---

# What is $\text{TC}^0 \subseteq \text{NC}^1 \subseteq \text{L} \subseteq \text{NL} \subseteq \text{P}$ ?

These are **complexity classes** — sets of problems grouped by how much of some resource they need. The chain reads "everything in TC⁰ is also in NC¹, which is also in L, …". Each inclusion is believed strict but mostly unproven.

| class   | definition                                                                                        | intuition                                                                         |
| ------- | ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **TC⁰** | constant-depth, polynomial-size circuits with unbounded fan-in AND/OR/NOT **and threshold gates** | "how much can you do in a fixed number of parallel rounds, if you can also count" |
| **NC¹** | $O(\log n)$-depth circuits with bounded fan-in                                                    | logarithmically many sequential rounds                                            |
| **L**   | solvable with $O(\log n)$ working memory                                                          | tiny scratch space, any time                                                      |
| **NL**  | same but nondeterministic                                                                         | includes graph reachability                                                       |
| **P**   | solvable in polynomial time                                                                       | "efficiently solvable" in the classical sense                                     |

A **threshold gate** answers "are more than $k$ of my inputs 1?" — it can count, which is why TC⁰ is more powerful than AC⁰ (same thing without threshold gates).

## The result about Transformers

**Merrill & Sabharwal (2023):** a constant-depth Transformer with $O(\log n)$ precision can be simulated by a uniform TC⁰ circuit.

**Why:** constant depth in, constant depth out. Every layer is a fixed number of parallel operations, and the number of layers doesn't grow with input size. The architecture literally cannot perform a computation whose serial length depends on $n$.

**Believed outside TC⁰:**

- evaluating deeply nested Boolean formulas (NC¹-complete)
- graph connectivity (NL-complete)
- Circuit Value Problem (P-complete)
- simulating certain finite automata for $n$ steps

**Common thread: they need a chain of steps that grows with the input.**

> ### Two corrections worth keeping
>
> **1. Parity.** The claim "Transformers can't do parity" is loosely stated. **Parity IS in TC⁰** — threshold gates count bits. What _cannot_ do parity is **AC⁰** (no threshold gates).
>
> **Hahn (2020)** is a different result: soft attention with bounded Lipschitz constant cannot compute parity or Dyck-2 **robustly** — it can in principle, but requires weights growing with $n$, so it isn't learnable in practice. That's a statement about **learnability and robustness**, not about the complexity class. Keep the two separate.
>
> **2. Turing completeness vs TC⁰ — not a contradiction.**
>
> |            | Pérez et al.            | Merrill & Sabharwal   |
> | ---------- | ----------------------- | --------------------- |
> | assumption | **unbounded** precision | $O(\log n)$ precision |
> | conclusion | Turing-complete         | **TC⁰**               |
>
> One real number with infinite precision _is_ infinite memory — you can store a whole Turing tape in its decimals. That assumption is false in practice (fp16 has 10 mantissa bits), so **TC⁰ is the result that describes real models.**
>
> **General lesson: a theoretical result is worth exactly as much as its assumptions. Read the assumptions before the conclusion.**

## How CoT lifts the ceiling

**Merrill & Sabharwal (2024), _The Expressive Power of Transformers with Chain of Thought_:**

| CoT length $t(n)$ | reachable class                |
| ----------------- | ------------------------------ |
| $O(1)$ constant   | still **TC⁰** — nothing gained |
| $O(\log n)$       | roughly **L**                  |
| $\text{poly}(n)$  | **exactly P**                  |

**Polynomially long CoT gives you all of P**, including P-complete problems that constant-depth circuits provably cannot touch.

**Feng et al. (2023)** agrees empirically: constant-depth Transformers cannot do multi-digit addition or solve linear systems in one pass, **but can with CoT** — those problems are "repeat the same step, input-many times" (carrying, elimination).

**This is the theoretical explanation of why Chain-of-Thought works.**

---

# Latent Chain of Thought

Reasoning without emitting text tokens. Two families.

## Family A — horizontal: Coconut

_Chain of Continuous Thought_ (Hao et al., 2024). Skip sampling entirely:

$$\text{normal CoT:} \quad h_t \xrightarrow{W_{\text{out}}} \text{logits} \xrightarrow{\text{sample}} \text{token} \xrightarrow{E} \text{embedding} \to \text{next position}$$ $$\text{Coconut:} \quad h_t \longrightarrow \text{next position directly}$$

- Still occupies a new position; still writes to the KV cache
- But **no compression of $d$ dimensions down to one choice out of 50,000**
- Can therefore **hold several possibilities at once** — like breadth-first search rather than committing to one path

Trained with a curriculum that progressively replaces written reasoning steps with latent ones.

## Family B — vertical: recurrent depth

Loop at the **same position**, consuming no sequence budget. Example: **Huginn** (Geiping et al., 2025), a 3.5B model split into three parts.

```
      input embedding
             |
        prelude (2 layers, runs once)
             |
     +---> recurrent core (4 layers, one weight set)  --+   loop r times
     |                                                  |
     +--------------------------------------------------+
             |
        coda (2 layers, runs once)
             |
        sample token

effective depth with r=4:  2 + (4 x 4) + 2 = 20 layers per token
```

**Mechanism:**

$$e = \text{Prelude}(x) \qquad \text{(once)}$$ $$s_0 = \text{random noise}$$ $$s_i = \text{Core}(s_{i-1} + e), \qquad i = 1 \dots r$$ $$\text{logits} = W_{\text{out}}\cdot\text{Coda}(s_r)$$

The thing being looped is $s$ — the same $n \times d$ state we've dealt with all along, overwritten in place.

**Three non-obvious design details:**

**(a) Why inject $e$ every iteration.** Without it, $s$ **drifts** and the loop forgets the question. (Same underlying reason as rank collapse: repeating one operator converges to something.) Re-adding $e$ re-pins it to the problem each round.

**(b) Why start from random noise.** It forces the core to work **independently of its starting point**, so it must learn to _converge toward_ an answer rather than memorize a path. Consequence: it stays usable at iteration counts it wasn't trained on.

**(c) Why $r$ must be randomized during training.** Train always at 4 and it breaks at 8. Sampling $r$ per batch teaches the model to produce a usable answer **at any depth**.

**The payoff: adaptive compute at inference with no retraining.**

```
"the" (easy function word)   ->  2 iterations
"therefore the answer is..." -> 32 iterations
```

**Problems to solve:**

- **Memory explosion.** Backprop stores activations for every layer; 32 loops × 4 layers = 128 layers of activations **per token**. Fix: **truncated backpropagation** through only the last $k$ iterations, which works because the architecture is designed to converge.
- **When to stop.** **Adaptive halting** — measure how much the output changed from the previous iteration (e.g. KL of the logits); if it has converged, stop. (Ancestor: **ACT**, Adaptive Computation Time, in Universal Transformer 2018, with a learned halting unit — harder to train.)
- **KV cache.** Each iteration writes its own KV entries, so the cache grows by a factor of $r$. Still a disadvantage vs normal CoT.

## Comparison

|                             | **normal CoT**                | **Coconut** (horizontal) | **Recurrent depth** (vertical) |
| --------------------------- | ----------------------------- | ------------------------ | ------------------------------ |
| consumes sequence positions | yes                           | yes                      | **no**                         |
| converts to tokens          | yes                           | no                       | no                             |
| depth per position          | $L$                           | $L$                      | **$L \times r$**               |
| adaptive compute per token  | no                            | no                       | **yes**                        |
| human-readable              | yes (but possibly unfaithful) | no                       | no                             |
| eats context budget         | a lot                         | yes                      | **no**                         |
| status                      | used everywhere               | research                 | research                       |

## Why latent reasoning hasn't won — the real reason

This is the most important insight in this section, and it has **two layers with one root cause.**

**(1) Supervision density — for training.**

Normal CoT slots into the existing loss perfectly:

$$\mathcal{L} = -\sum_t \log p_\theta(y_t \mid y_{<t})$$

**Every token of the chain is one crisp teaching signal**: "at this position, this word."

Coconut and recurrent depth have **no target at all** for intermediates. Nobody knows what $s_2$ _should_ be. The only signal is the final token's loss, travelling back through every iteration.

Worse for recurrent depth: **credit assignment is blurred.** 32 iterations, wrong answer — which iteration failed? Normal CoT shows you: _"step 3 computed 23-20 incorrectly."_ And since all iterations share one weight set, fixing iteration 17 perturbs iterations 1–32 simultaneously — unlike 96 independent blocks.

**(2) Inspectability — for safety.**

Normal CoT is readable, so you can check arithmetic, spot bad reasoning, and sometimes catch stated intent. **CoT monitorability** is a real (if fragile) oversight channel, and researchers across several organizations warned in 2025 that it may disappear if we push toward latent reasoning.

Note the caveat: Turpin et al. means readable ≠ faithful. But _something_ to read still beats nothing.

**(3) And the root cause of both: no training data exists.**

The internet is full of humans writing out their reasoning — worked math, proofs, commented code — trillions of tokens. **Models learn step-by-step thinking during pre-training, for free.**

**Nobody has ever recorded a "sequence of thought vectors."** Latent reasoning must be discovered from scratch, from a very thin signal.

> **Latent reasoning isn't losing because it's a worse idea. It's losing because it has neither a teacher nor an inspector.**

## But the principled advantages remain

1. **Language is a bottleneck.** Compressing a $d$-dimensional vector into one of 50,000 words discards enormous information at every step. Some thoughts may have no words.
2. **Per-token adaptive compute** — normal CoT cannot do this at all.
3. **No context budget consumed** — currently the most expensive resource.

---

# More...

## Inference and serving

Training and inference have **opposite bottlenecks**: training is compute-bound, inference is memory-bound.

### KV cache

Generating token $t+1$ recomputes $K,V$ for tokens $1..t$ — wastefully, because **causal masking means past $k_j, v_j$ never change**. So cache them.

Cost per step: $O(n^2) \to O(n)$.

**Memory price:**

$$\text{cache size} = 2 \times L \times n \times d \times \text{bytes}$$

For a 70B model ($L=80$, $d=8192$), 8192 context, fp16:

$$2 \times 80 \times 8192 \times 8192 \times 2 = \textbf{21.5 GB per conversation}$$

The model itself is 140GB. Ten concurrent users = 215GB of cache. **This, not FLOPs, is the real reason long context is expensive.**

### Prefill vs decode

|          | **prefill**                | **decode**                   |
| -------- | -------------------------- | ---------------------------- |
| does     | process the whole prompt   | emit one token               |
| parallel | **fully**                  | **not at all**               |
| rounds   | 1                          | length of the answer         |
| bound by | **compute**                | **memory bandwidth**         |
| metric   | TTFT (time to first token) | TPOT (time per output token) |

**Why decode is memory-bound:** to produce **one** token the GPU must read all 140GB of weights from VRAM, then do almost no arithmetic with them.

```
prefill 1000 tokens: read 140GB once -> 1000 tokens of work
decode 1 token:      read 140GB once -> 1 token of work    <- terrible ratio
```

Two consequences:

- **Batching is enormously effective.** Read the weights once, serve 64 users. Per-user cost drops ~64× with barely any slowdown. (This is why APIs are cheaper than self-hosting.)
- **Speculative decoding.** A small model guesses 5 tokens; the big model **verifies all 5 in one round** (verification is parallel, generation isn't). If the guesses are right you got 5 tokens for the price of 1. 2–3× speedup, identical output distribution.

### MQA / GQA

Cache size scales with the number of KV heads, so share them.

|                            | Q heads | KV heads | cache     |
| -------------------------- | ------- | -------- | --------- |
| **MHA**                    | 64      | 64       | 100%      |
| **GQA** (current standard) | 64      | 8        | **12.5%** |
| **MQA**                    | 64      | 1        | 1.6%      |

**Why quality barely drops:** $K$ is the "name tag" — fairly universal. $Q$ is "what I'm looking for" — needs diversity. So share K,V, keep many Q. MQA (share everything) loses a bit; GQA is the sweet spot used by LLaMA-2/3, Mistral, and most others.

### FlashAttention

**The real problem is GPU memory hierarchy:**

|                             | size   | speed    |
| --------------------------- | ------ | -------- |
| **HBM** (what we call VRAM) | 80 GB  | ~2 TB/s  |
| **SRAM** (on-chip)          | ~20 MB | ~19 TB/s |

Naive attention writes the $n\times n$ matrix to HBM, reads it back for softmax, writes again, reads again for the $V$ multiply. At $n = 8192$ that's 67M numbers **per head per layer**, shuttled repeatedly.

**FlashAttention never materializes that matrix.** It tiles $Q,K,V$ into blocks that fit in SRAM and finishes the computation there.

**The trick — online softmax.** Keep two running statistics:

- $m$ = max seen so far
- $\ell$ = running sum of $e^{s-m}$

When a new tile has a larger max, rescale the old accumulation:

$$\ell_{\text{new}} = \ell_{\text{old}}, e^{m_{\text{old}} - m_{\text{new}}} + \sum_{\text{new tile}} e^{s - m_{\text{new}}}$$

The accumulated output is rescaled by the same factor. Result is **numerically identical to normal softmax — not an approximation.**

(Subtracting $m$ before $\exp$ also prevents overflow; this "numerically stable softmax" is standard regardless.)

**Outcome:** memory $O(n^2) \to O(n)$, 2–4× faster, exact. No downside — universally adopted, and the reason long context became practical.

### Where the FLOPs actually go

| component                       | cost       |
| ------------------------------- | ---------- |
| QKV + O projections             | $O(nd^2)$  |
| attention scores + weighted sum | $O(n^2 d)$ |
| FFN                             | $O(nd^2)$  |

$n^2d$ overtakes $nd^2$ only when $n > d$. **For $d = 4096$, any context under 4096 tokens is dominated by the FFN, not attention.**

Common misconception corrected: attention is expensive in **memory** from the start; it only wins on **compute** at long context.

### Other serving techniques

| technique                             | what it does                                                                                                                  |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **Quantization**                      | 8-bit or 4-bit weights instead of 16 → 2–4× smaller, faster to read (hits the memory bottleneck exactly), slight quality loss |
| **PagedAttention (vLLM)**             | manage the KV cache like OS virtual memory — no need for large contiguous allocations                                         |
| **Continuous batching**               | when one request in a batch finishes, immediately slot in a new one                                                           |
| **Sliding window / sparse attention** | each token attends to only the last $w$ → constant cache regardless of context                                                |

---

## Post-Training

Pre-training gives **capability**. Post-training gives **behaviour**.

| stage               | teaches                          | data                            | compute |
| ------------------- | -------------------------------- | ------------------------------- | ------- |
| **1. Pre-training** | language + knowledge + reasoning | trillions of raw tokens         | ~99%    |
| **2. SFT**          | the question-answer format       | 1k–100k human-written dialogues | ~0.5%   |
| **3. RLHF / DPO**   | what counts as a _good_ answer   | 10k–1M preference pairs         | ~0.5%   |

### Pre-training gives more than "language"

It gives essentially everything: all knowledge (in the FFN), reasoning circuits, coding, translation — **and even question-answering ability**, if you prompt correctly:

```
Bad prompt:  "What is the capital of Thailand?"
             -> "What is the capital of Laos? What is the capital of Vietnam?"

Good prompt: "Q: What is the capital of France?
              A: Paris
              Q: What is the capital of Thailand?
              A:"
             -> "Bangkok"     <- correct
```

**The knowledge was already there.** The model just didn't know which mode it was supposed to be in. Having read the whole internet, it can imitate anyone — academics, forum posters, spammers, novelists — all superimposed.

### SFT uses the _exact same_ loss

$$\mathcal{L} = -\sum_t \log p_\theta(y_t \mid y_{<t})$$

Identical. Still next-token prediction, still cross-entropy, still AdamW. **Only two things change:**

**(a) Data** — human-written conversations:

```
User: Explain photosynthesis.
Assistant: Photosynthesis is the process by which plants...
```

**(b) Loss masking** — only the assistant's tokens count. We don't want it good at inventing questions.

**That's it.** SFT is pre-training on one specific kind of high-quality data.

**Evidence that SFT teaches no new knowledge — LIMA (2023)** turned a base model into a usable assistant with **1,000 examples**. Against trillions of pre-training tokens, that cannot possibly install knowledge.

> **Superficial Alignment Hypothesis:** all knowledge and ability come from pre-training; alignment only teaches **which mode to select**.

Analogy: someone who read the whole library but was never told "when asked a question, answer it." SFT doesn't make them smarter — it teaches conversational convention.

### RLHF genuinely changes the objective

**(1) Collect comparisons** — two answers to the same prompt, a human picks the better.

**(2) Train a reward model** $r_\phi(x,y)$ giving a single score:

$$\mathcal{L}_{RM} = -\log \sigma\big(r_\phi(x, y_{\text{win}}) - r_\phi(x, y_{\text{lose}})\big)$$

**(3) Optimize the policy:**

$$\max_\theta\ \mathbb{E}\big[r_\phi(x,y)\big] - \beta,\text{KL}\big(\pi_\theta ,|, \pi_{\text{SFT}}\big)$$

- first term: get high reward
- second term (**KL penalty**): don't drift far from the SFT model

**Why the KL term is essential:** the reward model is imperfect. Chasing reward alone leads the policy to **exploit its flaws** — producing weird text that scores highly and reads like nonsense. This is **reward hacking**.

> **The real difference: the loss no longer says what the next token should be. It says whether the whole response was good.** Coarser signal — but it conveys something imitation cannot.

**DPO (Direct Preference Optimization)** proves you can skip the reward model and RL loop entirely, rewriting the objective to train directly from preference pairs with a supervised-style loss. Simpler, more stable, comparable results.

### Why SFT can't replace RLHF

1. **Humans judge better than they demonstrate.** I can tell which of two quantum mechanics explanations is better without being able to write either. RLHF exploits exactly this gap.
2. **SFT has no negative signal.** It can say "do this" but never "don't do that." How would you write a demonstration of _not_ hallucinating? Preferences say it directly.
3. **SFT is capped at demonstrator level.** Imitation cannot exceed the humans writing the examples. RLHF can, because the model proposes and is scored.

### LoRA

Instead of updating all of $W$, add a low-rank correction:

$$W + BA, \qquad B \in \mathbb{R}^{d\times r},\ A \in \mathbb{R}^{r \times d},\ r \ll d$$

Trains ~0.1% as many parameters with near-equivalent results. Standard for fine-tuning on limited hardware.

### Summary analogy

| stage            | like                                                                               |
| ---------------- | ---------------------------------------------------------------------------------- |
| **Pre-training** | reading the entire library — knows everything, doesn't know what to do with it     |
| **SFT**          | learning conversational manners — "when asked, answer; don't ask back"             |
| **RLHF**         | an internship with feedback — "this was good, this was too long, this was made up" |

> **Pre-training builds capability. Post-training builds habits.**

---

## Theoretical results

### Universal approximation (Yun et al., 2020)

Transformers **with positional encoding** are universal approximators of continuous sequence-to-sequence functions on a compact domain. **Without PE**, they can only represent permutation-equivariant functions — exactly matching the Position Encoding section.

Proof sketch: (1) quantize the input space into a fine grid; (2) use attention + PE to give every input sequence a **unique identifier** (a "contextual mapping"); (3) let the FFN act as a lookup table from identifier to output.

> Don't over-read it: the construction needs exponentially many parameters and says **nothing about learnability**. It proves a weight setting exists, not that gradient descent finds it.

### Turing completeness (Pérez et al.)

With **unbounded precision** and hard attention, Transformers are Turing-complete. See the boxed correction in the TC⁰ section for why this doesn't conflict with the TC⁰ result — it's entirely about the precision assumption.

### Rank collapse (Dong et al., 2021)

For a network of **pure self-attention** (no skip, no FFN):

$$\left|X^{(L)} - \mathbf{1}x^\top\right| \le \left(\frac{c\gamma}{\sqrt{d}}\right)^{\frac{3^L-1}{2}}$$

The exponent $\frac{3^L-1}{2}$ grows **doubly exponentially**. With base 0.9:

| depth $L$ | exponent | residual         |
| --------- | -------- | ---------------- |
| 1         | 1        | 0.9              |
| 2         | 4        | 0.66             |
| 3         | 13       | 0.25             |
| 4         | 40       | 0.015            |
| 5         | 121      | $3\times10^{-6}$ |

**Dead by layer 5.** Not a slow problem that appears at depth 100.

**Why:** the attention matrix $A$ is **row-stochastic** (non-negative, rows sum to 1) — a Markov chain transition matrix.

$$X^{(L)} = A^{(L)}A^{(L-1)}\cdots A^{(1)}X^{(0)}W$$

Basic Markov theorem: repeated multiplication converges to the stationary distribution, independent of the start.

$$\lim_{L\to\infty} A^L = \mathbf{1}\pi^\top \quad(\text{rank-1})$$

**That is rank collapse** — every token forgets who it is. (The formal tool is the **Birkhoff contraction coefficient**.)

**What saves it:**

- **Residual connections** — via **path decomposition**, the network is a sum over all paths: $$\text{output} = \underbrace{X}_{\text{skips everything}} + \underbrace{\sum \text{short paths}}_{\text{1–2 attentions}} + \underbrace{\sum \text{long paths}}_{\text{these collapse}}$$ Long paths collapse; short paths don't, and they dominate.
- **FFN** — attention contracts, but the FFN can have Lipschitz constant > 1, so it **expands** and re-separates tokens.
- **LayerNorm** — helps least (fixes scale, not direction).

> **Same math as over-smoothing in Graph Neural Networks.** Message passing = averaging neighbours = a contraction. A Transformer _is_ a GNN on a fully connected graph with learned edge weights, so the result should be unsurprising.

### Induction heads and in-context learning

**The circuit** (Olsson et al., Anthropic), at the matrix level:

- **Shallow — previous-token head.** Its QK circuit implements "attend one position back" (pure positional pattern). Its OV circuit copies that token's information into the current position's residual stream. Now "kind" carries _"the token before me was John."_
- **Deep — induction head.** Its **QK circuit reads from two different sources**: the query comes from the current token (the second "John"), while the key reads the **subspace the shallow layer wrote to** — so it matches "who has John in front of them." Its **OV circuit is essentially a copy operation** ($W^{OV} \approx$ identity), moving "kind" to the current position.

**A perfect illustration of QK = "where to read from" and OV = "what to move."** And a clean proof that one layer cannot do it: the key must read what a previous layer wrote.

**Phase change:** the loss curve shows a visible bump exactly when induction heads form. Before it, in-context learning is essentially absent; after, it works. **Ablation** of induction heads destroys most ICL — causal evidence, not just correlation.

### ICL as gradient descent (von Oswald et al.)

A linear self-attention layer can **implement one step of gradient descent**.

Given in-context examples $(x_1,y_1),\dots,(x_k,y_k)$ and query $x_q$, one GD step on linear regression from weights $W$:

$$\Delta W = -\eta\sum_i (Wx_i - y_i)x_i^\top$$

Its effect on the prediction for $x_q$:

$$\Delta \hat y_q = -\eta \sum_i (Wx_i - y_i),(x_i^\top x_q)$$

**Look at the shape:** $\sum_i (\text{something}_i)\times(x_i \cdot x_q)$ — **exactly the form of attention**, with $x_i\cdot x_q$ as the score and the bracket as the value. Specific settings of $W^Q,W^K,W^V,W^O$ compute this, so **$k$ layers ≈ $k$ GD steps.**

> **Interpretation: in-context learning may be a meta-learned optimizer embedded in the forward pass** — the model isn't just pattern-matching, it's training a small model in its head from the prompt examples.

**Caveats:** proven for **linear attention** on **linear regression**; how far it extends to real LLMs is debated. A competing view is **Xie et al.** — ICL as **implicit Bayesian inference**: the model infers "which task is this?" from the examples and switches to that mode. Both are probably true at different levels.

### Scaling laws

**Kaplan et al. (2020):** loss follows a power law, smooth over 7+ orders of magnitude:

$$L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}$$

Straight line on a log-log plot — **predictable**, which is very unusual in deep learning. Kaplan's conclusion was to scale parameters mainly, data only a little. This drove the 2020–21 race to build ever-larger models.

**Chinchilla (Hoffmann et al., 2022)** refit with $N$ and $D$ separated:

$$L(N,D) = \underbrace{E}_{\approx 1.69} + \underbrace{\frac{A}{N^{0.34}}}_{\text{model too small}} + \underbrace{\frac{B}{D^{0.28}}}_{\text{data too little}}$$

- $E$ = **irreducible entropy of language**. Infinite model, infinite data — you still cannot go below it, because language is genuinely uncertain.
- Exponents 0.34 and 0.28 are **close**, so $N$ and $D$ should grow at similar rates.

Optimizing under $C \approx 6ND$:

$$\boxed{D_{\text{opt}} \approx 20N}$$

**Verified:** Chinchilla (70B, 1.4T) beat Gopher (280B, 300B) on nearly every benchmark while being 4× smaller.

**Why Kaplan was wrong:** his LR schedule wasn't adjusted to training length, which systematically penalized longer-trained models.

**Caveats:**

**(a) Chinchilla-optimal ≠ practically optimal.** Chinchilla only counts **training** cost, not **inference** — which is paid on every request for the model's whole life. **LLaMA deliberately overtrains** (7B on 1T tokens = 140×, not 20×) because a smaller model that's good enough is **cheaper to serve**.

**(b) Data wall.** High-quality text on the internet is finite; estimates put exhaustion in the mid-2020s. Hence synthetic data and the shift from quantity to quality.

**(c) The emergence debate.** **Schaeffer et al.** argue "sudden emergent abilities" may be a **metric artifact** — switch from 0/1 accuracy to a continuous metric and the curve becomes smooth.

---

## What changed since the 2017 paper

| component      | original (2017)     | modern                        |
| -------------- | ------------------- | ----------------------------- |
| Norm placement | Post-LN             | **Pre-LN**                    |
| Norm type      | LayerNorm           | **RMSNorm**                   |
| Position       | sinusoidal absolute | **RoPE / ALiBi**              |
| Activation     | ReLU                | **SwiGLU / GeGLU**            |
| Attention      | MHA                 | **GQA + FlashAttention**      |
| Bias terms     | present             | mostly removed                |
| Architecture   | encoder-decoder     | **decoder-only**              |
| Sparsity       | dense               | **Mixture-of-Experts** (some) |

**MoE:** replace one FFN with $N$ experts plus a router that activates only the top-$k$ (usually 1–2) per token. Total parameters grow a lot while **FLOPs per token stay constant**. Main problems: **load balancing** (needs an auxiliary loss so a few experts don't hog everything) and memory (all experts must still be resident).

---

## The insight list — what I'd want to remember first

1. **Attention is a differentiable dictionary lookup.** Everything else follows from making lookup soft enough to have gradients.
2. **$\sqrt{d_k}$ exists to stop softmax saturating**, because a saturated softmax has zero gradient.
3. **Attention only blends; it cannot create.** Output is a convex combination of values. This single fact forces the existence of the FFN.
4. **Attention moves information horizontally; FFN processes it vertically.** All interpretability rests on this split.
5. **Residual + FFN + LayerNorm are not decorations** — without them the architecture collapses to rank-1 within about five layers.
6. **The residual stream is a bus, not a pipe.** $x^{(L)} = x^{(0)} + \sum(\text{Attn} + \text{FFN})$ — 192 additive writes, no overwrites.
7. **$\alpha$ never leaves its block.** Only $z$ (folded into $x$) flows onward. Each block computes entirely new attention patterns and asks entirely different questions.
8. **Depth = number of reasoning hops per forward pass.** More heads cannot substitute, because heads can't see each other's output.
9. **The final vector isn't "relationships" — it's "everything needed to predict the next token,"** because that's what the loss sculpted it into. With weight tying, prediction is just "which word embedding do I point at."
10. **The answer accumulates gradually across depth** (logit lens), it doesn't appear at the end.
11. **Weights aren't shared across blocks, and that's correct** — sharing saves memory but zero compute, so it's a pure capacity loss.
12. **Training is compute-bound; inference is memory-bound.** Nearly every serving trick reduces bytes moved, not operations performed.
13. **CoT changes nothing architectural.** It uses the context window as a memory tape, converting a fixed-depth circuit into a recurrent machine — lifting TC⁰ to P.
14. **The context window and the "CoT space" are the same thing.** Thinking tokens are mechanically indistinguishable from prompt tokens, and consume the same budget.
15. **A theory result is worth exactly its assumptions** (Turing-complete vs TC⁰ = infinite vs log precision).
16. **Latent reasoning isn't losing on merit — it's losing because it has no teacher (no per-step target) and no inspector (nothing to read).**
17. **Pre-training builds capability; post-training builds habits.** SFT uses the identical loss to pre-training; only RLHF changes the objective.
18. **Weights = permanent memory, context = scratch paper that's erased every session.**

---

## Open questions I want to come back to

- Full backprop derivation through attention, by hand
- Mechanistic interpretability: superposition, sparse autoencoders, feature discovery
- MoE routing, load balancing, and expert specialization in depth
- Whether recurrent-depth models can find a supervision signal for intermediate states
- What replaces the internet as a data source once the data wall is hit
- Whether CoT faithfulness can be _trained for_, rather than hoped for

---

## Reading list

**Foundations**

- _Attention Is All You Need_ (Vaswani et al., 2017) — the original
- _The Illustrated Transformer_ (Jay Alammar) — visual walkthrough
- _The Annotated Transformer_ (Harvard NLP) — line-by-line code

**Architecture**

- _On Layer Normalization in the Transformer Architecture_ (Xiong et al.) — Pre-LN vs Post-LN
- _RoFormer_ (Su et al.) — RoPE
- _GLU Variants Improve Transformer_ (Shazeer) — SwiGLU
- _GQA_ (Ainslie et al.)

**Theory**

- _Attention is not all you need_ (Dong et al.) — rank collapse
- _A Mathematical Framework for Transformer Circuits_ (Elhage et al., Anthropic) — QK/OV, residual stream
- _In-context Learning and Induction Heads_ (Olsson et al., Anthropic)
- _The Expressive Power of Transformers with Chain of Thought_ (Merrill & Sabharwal, 2024)
- _Are Transformers universal approximators of sequence-to-sequence functions?_ (Yun et al.)

**Systems**

- _FlashAttention_ (Dao et al.)
- _Efficient Memory Management for LLM Serving with PagedAttention_ (Kwon et al., vLLM)

**Scaling & training**

- _Scaling Laws for Neural Language Models_ (Kaplan et al.)
- _Training Compute-Optimal Large Language Models_ (Hoffmann et al., Chinchilla)
- _LIMA: Less Is More for Alignment_ (Zhou et al.)
- _Direct Preference Optimization_ (Rafailov et al.)

**Reasoning**

- _Chain-of-Thought Prompting Elicits Reasoning_ (Wei et al.)
- _Language Models Don't Always Say What They Think_ (Turpin et al.) — unfaithfulness
- _Let's Think Dot by Dot_ (Pfau et al.) — filler tokens
- _Training LLMs to Reason in a Continuous Latent Space_ (Hao et al.) — Coconut
- _Scaling up Test-Time Compute with Latent Reasoning_ (Geiping et al.) — recurrent depth
