# 5 — Hull & Superposition

> Geometry of what a transformer can reach, and information theory of what it can hold. Built from zero. Read Part 0 first, then either half standalone.

---

## 0. The two axes

Everything here is one of two questions. Keep them separate — they get conflated constantly.

$$\underbrace{\text{convex hull}}_{\textit{where can it go}} \qquad\qquad \underbrace{\text{superposition}}_{\textit{how much can it hold}}$$

|               | Question                           | Governing quantity                      |
| ------------- | ---------------------------------- | --------------------------------------- |
| Hull          | Which states are reachable at all? | span and convexity of the reachable set |
| Superposition | How many features fit in $d$ dims? | $\sqrt{S/d}$                            |

The one line both halves converge on:

$$\textbf{attention reweights what is already there} ;\longrightarrow; \textbf{FFN injects what is not}$$

---

# Part I — Where can it go

## 1. What a convex hull actually is

Pin points on a board, stretch a rubber band around the outside. Everything inside is the hull.

$$\text{hull}(V) = \Big\{ \sum_i a_i v_i \ : \ a_i \ge 0,; \sum_i a_i = 1 \Big \}$$

Those two conditions — non-negative, sums to one — **are exactly softmax**. A single attention step doesn't "happen to land inside the hull"; it is the definition of the hull in different notation. It cannot escape, ever.

Sharper: softmax outputs are strictly positive, never exactly zero. So attention output sits in the **relative interior** — it can't even touch a vertex, no matter how large $\beta$ gets.

**A hull is not a ball. It has no radius.** It's a polytope whose shape is determined entirely by where the vertices sit.

## 2. Attention contracts — it does not expand

The most counterintuitive result here. The naive picture ("stack layers, hull grows, more abstraction") is backwards.

### 2.1 Why re-attending cannot move the hull

New values are built from old states by a **linear** map:

$$v_i^{(2)} = W_V x_i^{(1)} = W_V\Big(\sum_k a_{ik} v_k^{(1)}\Big) = \sum_k a_{ik}\big(W_V v_k^{(1)}\big)$$

$W_V$ slides through the summation. Linear maps commute with convex combination. So every new vertex is still a weighted average of old vertices:

$$\text{hull}(V^{(1)}) \supseteq \text{hull}(V^{(2)}) \supseteq \text{hull}(V^{(3)}) \supseteq \cdots$$

Monotonically nested and shrinking. $W_OW_V$ does rotate and stretch the hull, but that's a fixed change of coordinates applied identically every layer — it moves the hull, it doesn't take it anywhere new.

### 2.2 The contraction proof

Let $A$ be row-stochastic ($a_{ij}\ge 0$, $\sum_j a_{ij}=1$), $V' = AV$.

Pick any unit direction $u$ and project everything onto it:

$$M = \max_i \langle u,v_i\rangle,\quad m = \min_i \langle u,v_i\rangle,\quad W = M-m$$

$W$ is the hull's **width in direction $u$** (support function). Show it shrinks for every $u$ and you're done.

**Step 1 — can't leave.** $$\langle u, v'_i\rangle = \sum_j a_{ij}\langle u, v_j\rangle \in [m, M]$$ A weighted average of numbers in $[m,M]$ stays in $[m,M]$.

**Step 2 — softmax's positivity is what kills it.** Let $\delta = \min_{ij} a_{ij} > 0$ (exists precisely because softmax never returns an exact zero). Let $j^\star$ be the argmin projection:

$$\langle u, v'_i\rangle = \underbrace{a_{ij^\star}}_{\ge \delta} m + \sum_{j\ne j^\star} a_{ij}\langle u,v_j\rangle ;\le; \delta m + (1-\delta)M$$

Even the rightmost token is forced to carry $\delta$ worth of the leftmost. Symmetrically $\langle u,v'_i\rangle \ge \delta M + (1-\delta)m$.

**Step 3 — subtract.** $$W' \le \big[\delta m + (1-\delta)M\big] - \big[\delta M + (1-\delta)m\big] = (1-2\delta)W$$

Strictly less than 1, independent of $u$. Over $L$ layers:

$$W_L \le (1-2\delta)^L W_0 \xrightarrow[L\to\infty]{} 0$$

Hull collapses to a point. Every token becomes the same vector. **Rank collapse.**

### 2.3 The matrix property behind it

$(1-2\delta)$ is a special case of the **Dobrushin ergodicity coefficient**:

$$\tau(A) = \tfrac12 \max_{i,k}\sum_j |a_{ij}-a_{kj}| \in [0,1], \qquad \tau(AB)\le\tau(A)\tau(B)$$

Submultiplicative — that's the engine. Products of row-stochastic matrices with $\tau<1$ converge to **rank 1** (all rows identical). Same theorem as Markov chains converging to a stationary distribution.

Note $\tau(A)=1$ requires two rows with disjoint support — **softmax cannot do that**. Every entry is positive, supports always overlap. Attention is structurally incapable of avoiding this.

### 2.4 Why doubly exponential

The proof above gives exponential (constant $\delta$). The real rate is worse because of feedback:

$$\text{tokens converge} \Rightarrow \text{logits } \langle q_i,k_j\rangle \text{ converge} \Rightarrow \text{softmax flattens} \Rightarrow \delta \text{ rises} \Rightarrow \text{stronger contraction}$$

Self-reinforcing. Dong et al. 2021 (_Attention is not all you need_) bound it at roughly $|\cdot|^{3^L}$ — doubly exponential in depth.

### 2.5 The framing that makes it stick

A row-stochastic $A$ is a **random walk on the token graph**. Repeated $AV$ is the **heat equation**.

| Quantity              | Fate                                                       |
| --------------------- | ---------------------------------------------------------- |
| mean (1st moment)     | approximately conserved — this is what $\sum a=1$ protects |
| variance (2nd moment) | dies exponentially                                         |

$\sum a = 1$ conserves the first moment only. Nothing protects the second. Pure attention is diffusion to thermal equilibrium, and equilibrium means "everywhere identical" means "no information left."

> **Meaning collapse is caused by attention, not by FFN.** FFN is what prevents it.

_(Thread: this is the same math as oversmoothing in GNNs. Same contraction, same fixes.)_

## 3. What residual actually contributes

Residual is **addition**, not averaging. $x + o$ is a Minkowski sum of two sets, so diameters add. Stack $L$ layers, diameter adds $L$ times.

But this is nearly worthless as "abstraction":

| A growing hull means |                                        |
| -------------------- | -------------------------------------- |
| naive reading        | seeing a bigger picture, more abstract |
| actual reading       | the constraint became vacuous          |

A huge hull says nothing about capability. It says the theorem stopped being useful — like bounding an answer to $[-10^9, 10^9]$.

**Abstraction comes from the vertices moving, not the hull inflating.** Layer 1's vertices are raw token embeddings. Layer 8's vertices are layer 7's outputs, which already contain FFN-injected directions. Not one balloon inflating — a different polytope every layer.

## 4. Where the constraint is actually meaningful: direction, not distance

Any stack of attention, however deep, stays inside

$$\operatorname{span}{W_OW_V v_1,\dots,W_OW_V v_n}\cup{x}$$

a subspace determined entirely by the input. The hull can swallow the universe and this is still true.

> "Boundary of $x$" does not mean _how far can it go_. It means **which axes can it move along** — and the answer is: only axes that arrived with the input.

## 5. How FFN escapes

Expand $W_2\sigma(W_1x+b_1)$ with $k_j$ = row $j$ of $W_1$, $u_j$ = **column** $j$ of $W_2$:

$$\text{FFN}(x) = \sum_{j=1}^{4d} \underbrace{\operatorname{ReLU}(\langle k_j,x\rangle + b_j)}_{\text{coefficient}}\cdot \underbrace{u_j}_{\text{direction}}$$

Structurally identical to attention — coefficient × direction — but both slots differ:

|           | Coefficients                  | Directions                   |
| --------- | ----------------------------- | ---------------------------- |
| attention | softmax: $\ge0$, **$\sum=1$** | $v_i$ — from input           |
| FFN       | ReLU: $\ge0$, **unbounded**   | $u_j$ — from trained weights |

**Two independent escapes:**

- **Directions** → leaves the span. $u_j$ was never in the input.
- **Coefficients** → leaves the hull even using the same directions. Without $\sum=1$ it's a **cone**, not a hull. Unbounded reach.

**The $x=0$ test.** Feed in zero. Attention returns $0$ (nothing to average). FFN returns $W_2\operatorname{ReLU}(b_1)\ne 0$. FFN manufactures content from nothing. That's what "injection" means.

### 5.1 What the nonlinearity does geometrically

ReLU is **piecewise linear**. Each $\langle k_j,x\rangle + b_j = 0$ is a hyperplane carving space into cells. Inside one cell the map is affine — **convexity survives intact**. Different cells get different affine maps.

So when a convex set **straddles** a hyperplane, the two halves get different maps → **it folds** → the image is non-convex.

$$\text{straight segment crossing a boundary} ;\xrightarrow{;\text{ReLU};}; \text{bent polyline}$$

The midpoint no longer lies on the chord. That's where the theorem dies.

**Precision worth keeping:** if the set lies entirely inside one cell, FFN is just affine and the image is still convex. Escape isn't free — it requires actual straddling. The number of cells is exponential in neuron count; that's the _folding budget_, and it's why FFN is $4d$ wide.

### 5.2 FFN as memory

$$\underbrace{\langle k_j,x\rangle + b_j}_{\text{key: does this match what I stored}} \xrightarrow{;\operatorname{ReLU};} \underbrace{u_j}_{\text{value: the stored direction}}$$

$k_j$ = key, $u_j$ = value, $b_j$ = match threshold. FFN is a **key–value memory** (Geva et al. 2021). Maps onto the earlier framing exactly: attention searches the document in front of you, FFN opens the notebook you've kept for years.

## 6. Where convexity breaks, layer by layer

| Stage              | Operation                               | Convex?                      |
| ------------------ | --------------------------------------- | ---------------------------- |
| single head        | $\sum a_i v_i$                          | yes — strictly interior      |
| multi-head + $W_O$ | concat, linear map                      | yes — but a much larger hull |
| residual $x+o$     | Minkowski sum                           | yes — grows with depth       |
| **FFN**            | $W_2\sigma(W_1x)$                       | **no — breaks here**         |
| LayerNorm          | project onto sphere of radius $\sqrt d$ | no — sphere isn't convex     |

**Claim discipline:**

> valid: "a single retrieval step is trapped in the hull" invalid: "an entire latent thought chain is trapped in the hull"

This also settles the `HopfieldLayer` question: it can't do out-of-range regression, transformers can — **because of FFN**, not because of attention.

---

# Part II — How much can it hold

## 7. Start from zero: there is no storage

At one token position the model has exactly this:

$$x = [0.31,; -1.24,; 0.08,; \dots] \in \mathbb{R}^{768}$$

No slots. No labels. No table. Nothing in those numbers says "index 5 means French."

**Only two operations ever touch this vector:**

$$\textbf{write:}; x \leftarrow x + v \qquad\qquad \textbf{read:}; \langle w, x\rangle$$

Write is the residual add. Read is any weight row dotted against the stream. Everything called "memory" must be built from these two alone.

### 7.1 So where is a feature stored?

$\mathbf{f}_i$ is **not in $x$**. It lives in weights, as an agreement between two matrices:

|        | What it actually is                                                    |
| ------ | ---------------------------------------------------------------------- |
| writer | a column of $W_2$ (or attention OV output) that adds $c_i\mathbf{f}_i$ |
| reader | a row of $W_1$ that dots the stream to detect $\mathbf{f}_i$           |

**Memory is a contract between a writer and a reader.** Nothing is stored anywhere.

### 7.2 The residual stream is a shared bus

| Writers (add in) | Readers (dot out)       |
| ---------------- | ----------------------- |
| embedding        | attention $W_Q/W_K/W_V$ |
| attention $W_O$  | FFN $W_1$               |
| FFN $W_2$        | unembedding             |

Layers never talk to each other directly. They all read and write the same 768 numbers.

## 8. What "orthogonal" actually means

**There are two different dot products. Never conflate them.**

$$\underbrace{\langle \mathbf{f}_j, x\rangle}_{\text{reader vs. }\textbf{state}} \qquad\text{vs}\qquad \underbrace{\langle \mathbf{f}_j,\mathbf{f}_i\rangle}_{\text{reader vs. }\textbf{other dictionary entries}}$$

|                                                      | Should be             | Called       |
| ---------------------------------------------------- | --------------------- | ------------ |
| $\langle \mathbf{f}_j, x\rangle$ when $j$ fires      | large (near-parallel) | signal       |
| $\langle \mathbf{f}_j,\mathbf{f}_i\rangle$, $i\ne j$ | zero (orthogonal)     | interference |

Orthogonality **only ever refers to the second**. So both intuitions hold at once: the state should be near-parallel to whichever feature is active, and the features should be mutually orthogonal. No contradiction.

Why orthogonality is the right condition: I want a reader $w$ with $\langle w,\mathbf{f}_1\rangle = 1$ and $\langle w,\mathbf{f}_2\rangle = 0$. The simplest choice $w=\mathbf{f}_1$ turns the second condition into $\langle \mathbf{f}_1,\mathbf{f}_2\rangle = 0$.

> **Orthogonal is not about angles. It means "mutually invisible."** The angle is just what the dot product happens to measure ($\langle u,v\rangle = \cos\theta$ for unit vectors).

### 8.1 The routing view

Treat each row of $W_1$ as a path that can light up.

$$\langle \mathbf{f}_2,\mathbf{f}_1\rangle = 0 \iff \text{firing } \mathbf{f}_1 \text{ does } \textbf{not} \text{ light path 2}$$ $$\langle \mathbf{f}_2,\mathbf{f}_1\rangle = 0.9 \iff \text{firing } \mathbf{f}_1 \text{ drags path 2 on with it}$$

**Orthogonality = independent routing.** Non-orthogonality = crossed wiring.

### 8.2 The better name

Compressed sensing and dictionary learning call this **mutual coherence**:

$$\mu = \max_{i\ne j}|\langle \mathbf{f}_i,\mathbf{f}_j\rangle|$$

Same $\varepsilon$. "Low coherence" is the clearer name — no need to picture right angles at all.

### 8.3 Reader is not necessarily writer

Nothing forces the $W_1$ row to equal the $W_2$ column. The **optimal** reader is the dual vector:

$$\langle w_j,\mathbf{f}_j\rangle = 1, \qquad \langle w_j,\mathbf{f}_i\rangle = 0 ;;\forall i\ne j$$

i.e. the reader tilts away from its own feature in order to be orthogonal to competitors. When features are near-orthogonal, $w_j\approx\mathbf{f}_j$ (so the "reader = writer" shorthand is fine). **When $n > d$ the dual does not exist** — $n$ constraints in $d$ dimensions is overdetermined. That's the algebraic ceiling of superposition.

## 9. Concrete example: 3 features in 2 dimensions

$$\mathbf{f}_1 = (1,0),\quad \mathbf{f}_2 = (-0.5,, 0.866),\quad \mathbf{f}_3 = (-0.5,,-0.866)$$

Every pair has $\langle \mathbf{f}_i,\mathbf{f}_j\rangle = -0.5$. Terrible coherence. Watch it work anyway.

**One feature active,** $c_1 = 1 \Rightarrow x = (1,0)$:

| reader         | $\langle \mathbf{f}_j,x\rangle$ | after ReLU |
| -------------- | ------------------------------- | ---------- |
| $\mathbf{f}_1$ | $+1$                            | **1**      |
| $\mathbf{f}_2$ | $-0.5$                          | **0**      |
| $\mathbf{f}_3$ | $-0.5$                          | **0**      |

Perfect readout with zero orthogonal pairs, because interference here is **negative** and ReLU kills negatives for free. This is why toy models converge on antipodal pairs, triangles, pentagons — negative interference is the cheap arrangement.

**Two active,** $c_1=c_2=1 \Rightarrow x = (0.5,, 0.866)$: reading $\mathbf{f}_1$ gives $0.5$ instead of $1$. Both corrupted.

> The two-active failure is a $d=2$ artifact, **not** a general law. See §13.1.

## 10. Capacity: how $\varepsilon$ works

**Exact orthogonality:** at most $d$ directions. Hard ceiling.

**Relaxed to $|\cos\theta|\le\varepsilon$:** the count goes exponential. Derivation is a union bound over random unit vectors:

$$\Pr[|\langle u,v\rangle| > \varepsilon] \le 2e^{-d\varepsilon^2/2} ;\Longrightarrow; \binom{N}{2}2e^{-d\varepsilon^2/2} < 1 ;\Longrightarrow; N \lesssim e^{d\varepsilon^2/4}$$

### 10.1 The trap in the numbers (correcting the earlier table)

At $d=768$ the formula gives $\approx 7$ for $\varepsilon = 0.1$ — **lower than the 768 you get for free at $\varepsilon = 0$.** Impossible as a capacity, and the resolution matters:

$$N(\varepsilon) \text{ is monotone non-decreasing in } \varepsilon \text{ — always.}$$

(The $\varepsilon=0$ solution set is a _subset_ of the $\varepsilon=0.1$ solution set. Relaxing a constraint cannot shrink the answer.)

$e^{d\varepsilon^2/4}$ is a **lower bound from random construction**, not a ceiling. It's weak at small $\varepsilon$ because (a) random darts are bad there while deliberate placement gets $d$ for free, and (b) the union bound assumes worst-case simultaneous failure across all pairs.

$$N(d,\varepsilon) ;\gtrsim; \max\big(d,; e^{d\varepsilon^2/4}\big)$$

At $d=768$ the exponential overtakes the floor around $\varepsilon\approx 0.19$.

| $\varepsilon$ | $e^{d\varepsilon^2/4}$ | actual capacity     |
| ------------- | ---------------------- | ------------------- |
| 0             | 1                      | 768                 |
| 0.1           | ~7                     | 768 (bound vacuous) |
| 0.2           | ~2,200                 | ~2,200              |
| 0.3           | $3\times10^7$          | $3\times10^7$       |
| 0.4           | $2\times10^{13}$       | $2\times10^{13}$    |

### 10.2 The threshold at $1/\sqrt d$

$$\frac{1}{\sqrt{768}} \approx 0.036$$

| regime                      | behaviour                                                                     |
| --------------------------- | ----------------------------------------------------------------------------- |
| $\varepsilon \ll 1/\sqrt d$ | **rigid** — tighter than the ambient noise floor; capacity stuck at order $d$ |
| $\varepsilon \gg 1/\sqrt d$ | **explosive** — exponential in $d\varepsilon^2$                               |

Asking for $\varepsilon < 0.036$ is asking for something finer than the background hum of the space itself.

### 10.3 Why high dimensions have room

Two random unit vectors in $\mathbb{R}^d$: $\mathbb{E}[\langle u,v\rangle]=0$, $\text{sd} = 1/\sqrt d$.

> In high dimensions, **"random" and "nearly orthogonal" are the same word.**

Measure concentration: the spherical cap within angle $\theta$ of any fixed pole shrinks exponentially in $d$. Small caps mean you can pack exponentially many of them.

## 11. The Gram matrix

$$G = W^\top W, \qquad G_{ij} = \langle \mathbf{f}_i,\mathbf{f}_j\rangle$$

Diagonal = squared lengths. Off-diagonal = pairwise angles. $G$ determines the vector set **completely, up to rotation** — a coordinate-free description of the geometry, which is exactly right for a residual stream with no privileged basis.

Named after Jørgen Gram, 1880s. It is old.

### 11.1 One-line proof that interference is unavoidable

$$\operatorname{rank}(G) = \operatorname{rank}(W^\top W) = \operatorname{rank}(W) \le d$$

Perfect mutual orthogonality means $G = I_n$, which has rank $n$. Therefore:

$$n > d ;\Longrightarrow; G \ne I ;\Longrightarrow; \text{some pair must be non-orthogonal}$$

No probability, no softmax, no training. **Pure algebra.** At least $n-d$ eigenvalues of $G$ are zero.

### 11.2 The Welch bound — $0.036$ is a hard floor

Unit norms give $\operatorname{tr}(G) = n$. Cauchy–Schwarz on eigenvalues with $\operatorname{rank}\le d$:

$$\operatorname{tr}(G^2) = \sum_i \lambda_i^2 \ge \frac{(\operatorname{tr}G)^2}{d} = \frac{n^2}{d}$$

And $\operatorname{tr}(G^2) = \sum_{i,j}G_{ij}^2 = n + \sum_{i\ne j}G_{ij}^2$, so:

$$\sum_{i\ne j}G_{ij}^2 \ge \frac{n^2}{d} - n \qquad\Longrightarrow\qquad \mu ;\ge; \sqrt{\frac{n-d}{d(n-1)}} ;\xrightarrow[n\gg d]{}; \frac{1}{\sqrt d}$$

**So $1/\sqrt d = 0.036$ isn't just "what random vectors happen to give."** It's the theoretical minimum. No arrangement, however clever, goes below it. The random-vector statistic and the optimal packing coincide — which is why random init is already near-optimal for this.

### 11.3 Where you have already met it

| Seen as                                   | Actually                    |
| ----------------------------------------- | --------------------------- |
| $X^\top X$ in normal equations            | Gram matrix                 |
| kernel matrix in SVM                      | Gram matrix                 |
| covariance in PCA                         | Gram matrix of centred data |
| $QK^\top$ before softmax                  | cross-Gram                  |
| similarity matrix in contrastive learning | Gram matrix                 |
| "Gram matrix" in neural style transfer    | called by name, for once    |

## 12. Why gradient descent enforces near-orthogonality

**There is no orthogonality regularizer anywhere.** It falls out of plain reconstruction loss on sparse data.

Toy autoencoder: $h = Wx$, $\hat x = W^\top h = Gx$, so

$$\hat x_j = \underbrace{G_{jj}x_j}_{=x_j} + \sum_{i\ne j}G_{ji}x_i$$

With independent features ($\mathbb{E}[x_ix_k]=0$ for $i\ne k$):

$$\mathcal{L} = \mathbb{E}|x-\hat x|^2 = \sum_j \mathbb{E}\Big[\Big(\sum_{i\ne j}G_{ji}x_i\Big)^2\Big] = m_2\sum_j\sum_{i\ne j}G_{ji}^2$$

$$\mathcal{L} = m_2 \sum_{i\ne j} G_{ij}^2$$

**The sum of squared off-diagonal Gram entries appears on its own.** That _is_ "be orthogonal," written algebraically. Nobody put it there.

$$\frac{\partial\mathcal{L}}{\partial G_{ji}} = 2m_2 G_{ji} ;\Longrightarrow; \text{always pushes } G_{ji}\to 0$$

**And here's the sparsity link:** $m_2 = \mathbb{E}[x_i^2] \propto p$ (activation probability). Sparser data → smaller $m_2$ → weaker orthogonalization pressure → **more overlap tolerated → more features packed.** That's the algebraic form of the capacity/density trade-off.

**What ReLU adds:** if interference falls below the negative bias it's clipped to exactly zero → error zero → **gradient exactly zero**. That overlap is genuinely free, not merely cheap. This creates the free-overlap zone.

**The arrangement is data-driven, not geometric:**

| Feature pair                             | Pressure                                     |
| ---------------------------------------- | -------------------------------------------- |
| overlapping **and** frequently co-active | high loss → pushed apart                     |
| overlapping but **never** co-active      | no loss → no gradient → **overlap for free** |

Consequence: **interference is systematic, not random.** The same pair collides the same way every time — a fixed bias, not fresh noise each pass. Gradient can learn around it.

> **Caveat.** This derivation is for a toy autoencoder. Real transformers have no reconstruction term — only next-token cross-entropy. The pressure is _indirect_: misread feature → wrong prediction → loss → gradient traced back through many layers to $G$. Same algebra, very different constants and sharpness. **Toy models are a microscope for the mechanism, not a photograph of the real thing.**

## 13. Signal vs. noise: the model cannot tell

$$\langle \mathbf{f}_j,x\rangle = \underbrace{c_j}_{?} + \underbrace{\sum_{i\ne j}c_i\langle \mathbf{f}_j,\mathbf{f}_i\rangle}_{?}$$

One scalar. No metadata. No way to know whether $0.8$ is signal $0.8$ or signal $0.6$ plus noise $0.2$.

ReLU + bias is not a separator — it's a **statistical decision**:

$$\text{signal}\sim 1, \qquad \text{noise}\sim\mathcal{N}(0,;\varepsilon^2 S)$$

$b$ is a point on the ROC curve. Too high → false negatives. Too low → false positives. **No setting is right on both.** Only the one minimizing total loss. The model doesn't need correct readouts — it needs to be wrong only where it doesn't affect prediction.

_(We can separate them, with SAEs or ablation. That's outside information the model doesn't have.)_

### 13.1 Co-activation is the point, not the problem

|                  | Cause                                | Verdict                        |
| ---------------- | ------------------------------------ | ------------------------------ |
| **true firing**  | those features are genuinely present | **good — this is the purpose** |
| **false firing** | interference from other features     | bad — misread                  |

At $d=768$, $\varepsilon\approx 0.036$, $S=4$:

$$\text{noise} \approx \varepsilon\sqrt S = 0.036\cdot 2 \approx 0.07 \quad\text{vs signal}\approx 1$$

Completely fine.

$$\text{The condition is } S \ll d, \textbf{ not } S = 1$$

Why you _need_ co-activation — that's where the combinatorics live:

$$\binom{n}{S}, \quad \text{e.g. } \binom{10{,}000}{5} \approx 8\times 10^{17} \text{ distinguishable states}$$

And AND-gates only work with co-activation. With $k = \mathbf{f}_1+\mathbf{f}_2$, $b=-1.5$:

| state               | $\langle k,x\rangle$ | $\operatorname{ReLU}(\cdot - 1.5)$ |
| ------------------- | -------------------- | ---------------------------------- |
| only $\mathbf{f}_1$ | 1                    | 0 — silent                         |
| both                | 2                    | 0.5 — **fires**                    |

**That is computation, not readout.** A logical condition evaluated on superposed features without ever unpacking them. The same negative bias also does cleanup. One operation, two jobs.

### 13.2 Two earlier claims that need refining

**"Latent CoT has no cleanup"** — wrong as stated. **FFN does cleanup at every layer** via ReLU + negative bias: small interference falls below threshold and is zeroed exactly.

Correct version: latent CoT **has** soft cleanup (continuous, bounded) but **lacks** hard cleanup (collapse to one of 50,000 tokens, absolutely). `argmax` discards error far more cleanly — a different class of operation.

**"5 parallel lines of thought = not sparse"** — too crude. 5 against 768 is nothing. The real issue is that **one line of thought is not one feature** — it's a bundle of tens or hundreds. $k$ lines at $S$ features each gives $\sqrt{kS/d}$. It's a **multiplier on an already-tight budget**, not a problem with the number 5.

## 14. Where superposition lives

Decisive variable: **is there a privileged basis?** = does an elementwise nonlinearity act on these coordinates?

| Location                | Dim  | Privileged basis           | Character                                  |
| ----------------------- | ---- | -------------------------- | ------------------------------------------ |
| residual stream         | 768  | **no**                     | heaviest — directions fully arbitrary      |
| FFN hidden              | 3072 | **yes** (ReLU per channel) | aligns to neurons; overflow → polysemantic |
| attention head interior | 64   | no                         | very dense, tiny dimension                 |
| the head itself         | —    | —                          | multiple behaviours per head               |
| embedding               | 768  | no                         | 50,000 tokens in 768 dims                  |

**Answer: yes, everywhere** — because everywhere in the model has more things to represent than dimensions to represent them in.

The residual stream has **no privileged basis** (no elementwise nonlinearity acts directly on it), so features have no reason to align with axes → superposition is the natural state there.

### 14.1 Polysemantic neurons

A neuron is one row $k_j$ of $W_1$. With more features to compute than neurons available, that direction can't belong to one feature:

$$k_j \approx \alpha\mathbf{f}_A + \beta\mathbf{f}_B + \gamma\mathbf{f}_C$$

so it fires on unrelated things (the vision example: cat faces, car fronts, cat legs).

$$\textbf{polysemantic neuron} = \textbf{superposition viewed through a privileged basis}$$

Not a new phenomenon. The residual stream has it _worse_ — nobody calls it polysemantic there only because there's no axis to look along.

**Why it still works:** sparsity. Cat-face and Chinese-character don't co-occur, so surrounding neurons disambiguate. Information isn't lost, it's distributed. A downstream reader combining several polysemantic neurons recovers a clean feature — **which is exactly what a sparse autoencoder does**: find a dictionary larger than the neuron count and unfold the entangled basis.

### 14.2 Superposition one level up: subspaces

One attention head reads a 768-dim bus through a 64-dim $W_Q$. It doesn't see the bus — it sees **the subspace it claimed**. Other heads claim theirs.

$$\text{heads claim } \textbf{nearly orthogonal subspaces} \text{ of the same bus}$$

Same mechanism, higher level: from "features nearly orthogonal" to "**channels** nearly orthogonal," so layer 3's writes don't clobber what layer 7 is about to read.

The 768 dims aren't just feature storage — they're **bandwidth allocated across every component in the model**, using the same trick: accept a little noise, buy a lot of capacity.

$$\text{feature} \subset \text{subspace} \subset \text{bus} \qquad \text{one mechanism at every scale}$$

---

# Part III — Bridge back to Hopfield

## 15. Crosstalk (1982) = interference (superposition)

Hopfield, from [[2-hopfield]] §3.3:

$$C_i^\nu = \frac1d\sum_{\mu\ne\nu} x_i^\mu\langle \mathbf{x}^\mu,\mathbf{x}^\nu\rangle, \qquad \operatorname{Var}\approx N/d$$

Superposition readout:

$$\langle \mathbf{f}_j, v\rangle = \underbrace{c_j}_{\text{signal}} + \underbrace{\sum_{i\in A,, i\ne j}c_i\langle \mathbf{f}_j,\mathbf{f}_i\rangle}_{\text{noise};\sim;\sqrt{S/d}}$$

**Identical statistic.** Both are $\sqrt{(\text{number overlapping})/d}$. Only the name changed.

## 16. Same computation, different readout

|               | Readout                       | Capacity           |
| ------------- | ----------------------------- | ------------------ |
| Hopfield 1982 | `sgn` per neuron              | $0.14d$            |
| Hopfield 2020 | `softmax` over whole patterns | exponential        |
| Superposition | `ReLU` + negative bias        | sparsity-dependent |

Superposition capacity and Hopfield capacity are **the same calculation with a different decoder attached.**

Critical distinction: noise depends on $S$ = **number simultaneously active**, not $N$ = number stored. Store a million; just don't fire many at once. Condition: $S \ll d$.

## 17. Hopfield 2020 is attention

One retrieval step = one softmax-weighted average = hull-bound. That's why `HopfieldLayer` can't extrapolate outside the stored range while a transformer can — **the difference is FFN**, not attention.

_(Still open: energy function and Jacobian derivation — see [[4-hopfield-internal]].)_

---

# Part IV — The big picture

## 18. What attention and FFN are actually doing

$$\text{attention} = \text{averaging} \Rightarrow \text{contracts} \qquad \text{residual} = \text{addition} \Rightarrow \text{translates} \qquad \text{FFN} = \text{nonlinear} \Rightarrow \text{escapes}$$

|                         | Attention                    | FFN                         |
| ----------------------- | ---------------------------- | --------------------------- |
| operation               | weighted average             | keyed lookup                |
| coefficients            | softmax, $\sum=1$            | ReLU, unbounded             |
| directions              | from input                   | from trained weights        |
| nonlinearity lives in   | the coefficients             | the basis                   |
| information source      | the document in front of you | the notebook kept for years |
| effect on reachable set | shrinks it                   | breaks it open              |
| effect on interference  | none                         | cleans it up                |
| left alone, causes      | rank collapse                | unbounded activation growth |

**Per-layer cycle:**

$$\text{read superposition} \to \text{compute (AND gates)} \to \text{clean up} \to \text{write superposition back}$$

Repeat 12–96 times, then unembedding reads once more into logits.

## 19. The analogy to keep: CDMA

| CDMA                            | Transformer                     |
| ------------------------------- | ------------------------------- |
| one shared frequency band       | 768-dim residual stream         |
| per-user spreading code         | $\mathbf{f}_i$                  |
| correlate against your own code | $\langle \mathbf{f}_j,x\rangle$ |
| interference from other users   | $\sqrt{S/d}$                    |
| limit on simultaneous talkers   | sparsity                        |

The model doesn't store features anywhere. **It mixes everything onto one wire and trusts the receivers to separate it** — and that trust is justified because high dimensions make everyone nearly orthogonal for free.

## 20. The one number

$$\text{crosstalk (1982)} ;=; \text{interference (superposition)} ;=; \sqrt{S/d}$$

Same number, three appearances, from [[2-hopfield]] §3.6 through latent CoT. If only one thing survives from this note, this is it.

---

# Part V — What is still black box

Not gaps in my understanding — gaps in the field. Worth knowing which is which.

- **How much superposition is in real models?** Unmeasured. All quantitative results come from toy models.
- **Do SAEs find real features, or features they were forced to find?** Actively contested. Dictionary size, sparsity penalty, and feature splitting all bias the answer.
- **Linear representation hypothesis** — is everything really a direction? Increasingly questioned (circular and multi-dimensional features).
- **What does FFN actually inject?** Mechanism understood, content barely.
- **Attention-head superposition** — how many behaviours share a head, and how do they avoid each other?
- **LayerNorm** — it projects onto a sphere, also breaking convexity, but its representational role is far less understood than FFN's.
- **Does any of this change with SwiGLU / gated FFN / MoE?** The key-value framing was derived for plain ReLU MLPs.
- Interpretability is roughly six years old. Nobody has the full picture.

---

# Part VI — Threads worth pulling next

**Directly extending this note**

- Welch bound and frame theory; equiangular tight frames (the optimal packings)
- Compressed sensing: mutual coherence, RIP, sparse recovery guarantees — superposition _is_ compressed sensing with a learned dictionary
- Dobrushin coefficient and Markov ergodicity — the general theory behind §2.3
- Neural collapse — a related geometry of learned representations

**Papers**

- Elhage et al. 2022, _Toy Models of Superposition_ — the algebra of §12 in full, plus phase diagrams of which geometry appears at which sparsity
- Dong et al. 2021, _Attention is not all you need_ — rank collapse, doubly exponential
- Geva et al. 2021, _Transformer Feed-Forward Layers Are Key-Value Memories_
- Elhage et al. 2021, _A Mathematical Framework for Transformer Circuits_ — QK/OV decomposition; the natural next step after this note
- Bricken et al. 2023 onward — sparse autoencoders, dictionary learning

**Prehistory — the same idea under other names**

| Year  | Name                                 | Field               |
| ----- | ------------------------------------ | ------------------- |
| 1880s | Gram matrix                          | linear algebra      |
| 1982  | crosstalk                            | statistical physics |
| 1986  | distributed representations (Hinton) | connectionism       |
| 1995  | HRR / VSA (Plate)                    | cognitive science   |
| 2000s | mutual coherence                     | applied math        |
| 2020  | polysemantic neurons (Olah)          | interp              |
| 2022  | superposition (Elhage)               | interp              |

Hinton's 1986 _Distributed Representations_ essentially states the idea 36 years early — without the capacity math.

**Cross-field**

- Oversmoothing in GNNs — literally the same contraction proof as §2.2
- CDMA / spread spectrum — the engineering version of superposition
