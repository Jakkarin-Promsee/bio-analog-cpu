# 11 — Perceiver, Perceiver IO, and the Economics of Reading

> **What this note is.** A from-zero derivation of the Perceiver family, written so that every formula, every variable, and every design decision is explained rather than asserted. The through-line is not "here is an architecture" but **"attention is a priced resource, and every architecture in this family is a different budget."**
>
> **Prerequisites.** [[1-transformer]], [[2-hopfield]]. Strongly complementary: [[6-fixed-point-and-what-transformer-chasing]], [[10-attention-collapse-and-field-equilibrium]], [[7-vit-foundation]], [[8-position-transformer]].
>
> **Primary papers.** Jaegle et al. 2021, _Perceiver_ (arXiv 2103.03206); Jaegle et al. 2021, _Perceiver IO_ (arXiv 2107.14795).

---

## Table of contents

0. [Notation](https://claude.ai/chat/18b8e8c6-25dc-4b4a-8462-9c038a531ca3#0-notation)
1. [Attention from zero](https://claude.ai/chat/18b8e8c6-25dc-4b4a-8462-9c038a531ca3#1-attention-from-zero)
2. [The asymmetry insight](https://claude.ai/chat/18b8e8c6-25dc-4b4a-8462-9c038a531ca3#2-the-asymmetry-insight)
3. [Cross-attention in full](https://claude.ai/chat/18b8e8c6-25dc-4b4a-8462-9c038a531ca3#3-cross-attention-in-full)
4. [The latent array Z0](https://claude.ai/chat/18b8e8c6-25dc-4b4a-8462-9c038a531ca3#4-the-latent-array-z0)
5. [The latent transformer](https://claude.ai/chat/18b8e8c6-25dc-4b4a-8462-9c038a531ca3#5-the-latent-transformer)
6. [Position encoding](https://claude.ai/chat/18b8e8c6-25dc-4b4a-8462-9c038a531ca3#6-position-encoding)
7. [Iteration](https://claude.ai/chat/18b8e8c6-25dc-4b4a-8462-9c038a531ca3#7-iteration)
8. [Weight sharing — the deep version](https://claude.ai/chat/18b8e8c6-25dc-4b4a-8462-9c038a531ca3#8-weight-sharing--the-deep-version)
9. [The Hopfield connection](https://claude.ai/chat/18b8e8c6-25dc-4b4a-8462-9c038a531ca3#9-the-hopfield-connection)
10. [Perceiver IO](https://claude.ai/chat/18b8e8c6-25dc-4b4a-8462-9c038a531ca3#10-perceiver-io)
11. [Why one read is mostly enough](https://claude.ai/chat/18b8e8c6-25dc-4b4a-8462-9c038a531ca3#11-why-one-read-is-mostly-enough)
12. [Making reads cheaper](https://claude.ai/chat/18b8e8c6-25dc-4b4a-8462-9c038a531ca3#12-making-reads-cheaper)
13. [CNN and ViT front-ends](https://claude.ai/chat/18b8e8c6-25dc-4b4a-8462-9c038a531ca3#13-cnn-and-vit-front-ends)
14. [Two-tier memory: a design sketch](https://claude.ai/chat/18b8e8c6-25dc-4b4a-8462-9c038a531ca3#14-two-tier-memory-a-design-sketch)
15. [Landscape: closed and open problems](https://claude.ai/chat/18b8e8c6-25dc-4b4a-8462-9c038a531ca3#15-landscape-closed-and-open-problems)
16. [Takeaways](https://claude.ai/chat/18b8e8c6-25dc-4b4a-8462-9c038a531ca3#16-takeaways)
17. [Reading list](https://claude.ai/chat/18b8e8c6-25dc-4b4a-8462-9c038a531ca3#17-reading-list)

---

## 0. Notation

Fixed for the whole note. The right column gives the ImageNet configuration from the Perceiver paper, used as a running numerical example throughout.

| Symbol | Meaning                                               | Example value                      |
| ------ | ----------------------------------------------------- | ---------------------------------- |
| `N`    | number of elements in the input (byte array)          | 50,176 (224 x 224 pixels)          |
| `C`    | channels per input element                            | 261 (3 RGB + 258 Fourier features) |
| `M`    | number of latent vectors — **you choose this**        | 512                                |
| `D`    | channels per latent — **you choose this**             | 1024                               |
| `O`    | number of output elements (Perceiver IO)              | task-dependent                     |
| `E`    | channels of an output query                           | task-dependent                     |
| `E'`   | channels of the final output                          | task-dependent                     |
| `d`    | internal attention width (query/key/value)            | 1024                               |
| `h`    | number of heads                                       | 1 cross-attn, 8 self-attn          |
| `L`    | self-attention blocks per latent transformer          | 6                                  |
| `T`    | number of cross-attends (iterations)                  | 8                                  |
| `B`    | batch size                                            | —                                  |
| `X`    | byte array, `X ∈ ℝ^(N×C)` — the frozen external world | —                                  |
| `Z`    | latent array, `Z ∈ ℝ^(M×D)` — the internal state      | —                                  |

**The single sentence that generates the architecture:**

> `N` is dictated by the data. `M` is dictated by you. The entire Perceiver family is the project of moving cost off the `N` axis and onto the `M` axis.

**Three widths that are easy to conflate — keep them apart:**

```
C  = input channel count        X    ∈ ℝ^(N×C)
D  = latent channel count       Z    ∈ ℝ^(M×D)     ← the residual stream width
d  = attention inner width      QKV  ∈ ℝ^(·×d)     ← transient, lives inside one block
```

`Z` is `M×D` at every point in the network. `d` exists only between the QKV projections and the output projection `W_O ∈ ℝ^(d×D)`, which maps back to `D` before the residual add.

---

## 1. Attention from zero

### 1.1 Attention is a differentiable database lookup

Forget the formula; start from the operation you want.

You have a database of `N` records. Record `i` has a **key** `kᵢ` (its label, used for matching) and a **value** `vᵢ` (its content, what you want back). You have a **query** `q` describing what you are looking for.

A hard lookup returns `vᵢ` where `kᵢ == q`. That is an `argmax`: zero gradient almost everywhere, untrainable. The fix is to return a **weighted average of all values**, weighted by match quality:

```
similarity :  sᵢ = q · kᵢ                       ∈ ℝ
weights    :  aᵢ = exp(sᵢ) / Σⱼ exp(sⱼ)         Σᵢ aᵢ = 1,  aᵢ > 0
output     :  out = Σᵢ aᵢ vᵢ
```

**Term by term:**

- `q · kᵢ` — dot product measures alignment in the projected space. This is the only place "relevance" is defined; everything else is bookkeeping.
- `exp(·)` — makes scores positive and **amplifies differences exponentially**. If `s₁ = 5, s₂ = 3` the raw gap is 2 but `e⁵/e³ = 7.4x`. This is why softmax behaves as a _soft argmax_, approaching hard selection as the score gap grows.
- dividing by `Σⱼ exp(sⱼ)` — normalises to a distribution over the `N` records.
- `Σᵢ aᵢ vᵢ` — the retrieval itself.

**A second reading that matters later.** This is exactly **Nadaraya–Watson kernel regression** with kernel `κ(q,k) = exp(q·k)`. Attention is non-parametric regression with a learnable kernel. Not magic — smoothing.

### 1.2 Why `1/√d`

Let `q, k ∈ ℝ^d` with i.i.d. entries, mean 0, variance 1.

```
q · k = Σ_{j=1}^{d} qⱼ kⱼ

E[q·k]   = 0
Var(q·k) = Σⱼ Var(qⱼ)·Var(kⱼ) = d
std(q·k) = √d
```

With `d = 1024`, logits have std ≈ 32. Softmax over logits spread across ±32 is numerically one-hot, and the Jacobian of softmax at a one-hot point is ≈ 0 — the model is dead at initialisation. Hence:

```
sᵢ = (q · kᵢ) / √d
```

which restores unit variance.

> **Link to [[2-hopfield]]:** `1/√d` is the inverse temperature `β` in the modern Hopfield update `ξ ← Xᵀ softmax(β X ξ)`. `β` controls whether retrieval is _sharp_ (converges onto a single stored pattern) or _blurry_ (settles into a metastable mixture). Attention temperature and Hopfield temperature are the same knob.

### 1.3 Matrix form and cost

Self-attention: every element supplies query, key, and value.

```
Q = X W_Q      W_Q ∈ ℝ^(C×d)      Q ∈ ℝ^(N×d)
K = X W_K      W_K ∈ ℝ^(C×d)      K ∈ ℝ^(N×d)
V = X W_V      W_V ∈ ℝ^(C×d)      V ∈ ℝ^(N×d)

A   = softmax_rows(Q Kᵀ / √d)     A   ∈ ℝ^(N×N)     ← the problem
out = A V                         out ∈ ℝ^(N×d)
```

- Since `Q Kᵀ = X (W_Q W_Kᵀ) Xᵀ`, the model is learning a **bilinear form** `W_Q W_Kᵀ ∈ ℝ^(C×C)` — a learned metric on input space. Similarity is not raw cosine similarity; it is similarity _in a space the model chose_. See [[5-interp-hull-and-superposition]] for what that metric can and cannot express.
- `softmax_rows`: each row sums to 1. Query `i` distributes 100% of its attention budget over the `N` keys.

Cost:

```
Q Kᵀ    :  N · N · d
softmax :  N · N
A V     :  N · N · d
memory  :  N²  per head per layer  (A is kept for backward)
```

At ImageNet resolution:

```
N   = 50,176
N²  = 2,517,630,976 ≈ 2.5e9
with d=1024, h=8, L=12   →  ~1e17 FLOPs per image
attention matrices alone: 2.5e9 × 4 bytes × 8 heads ≈ 80 GB per image
```

Infeasible. ViT sidesteps it by patchifying to `N = 196`, but **patchifying reintroduces a grid prior** — exactly what Perceiver refuses, so that one architecture can serve point clouds, audio, and multimodal input.

### 1.4 Where the quadratic really comes from

Not "nested loops". Semantically:

> **The quadratic term is the price of content-based all-pairs routing.** Every element decides from content alone which other elements it reads. Nobody specified the wiring in advance, so the wiring costs `N²`.

Breaking `O(N²)` therefore **requires giving something up**. Three families:

| Family                | What is sacrificed                                     | Examples                                    |
| --------------------- | ------------------------------------------------------ | ------------------------------------------- |
| **Sparsity**          | the read pattern is fixed in advance                   | Longformer, BigBird, Sparse Transformer     |
| **Low-rank / kernel** | softmax is approximated so matmuls can be reassociated | Linformer, Performer, Linear Attention      |
| **Bottleneck**        | information must pass a fixed-size channel you choose  | **Perceiver**, Set Transformer (ISAB), DETR |

Perceiver takes the third route and **approximates nothing**. Softmax stays exact. Only the identity of the querier changes.

---

## 2. The asymmetry insight

This is the whole idea. Everything else is engineering.

Return to `out = Σᵢ aᵢ vᵢ` and observe:

- the number of **queries** and the number of **keys/values** are unrelated
- `Q ∈ ℝ^(n_q × d)`, `K,V ∈ ℝ^(n_kv × d)` ⟹ `A ∈ ℝ^(n_q × n_kv)` ⟹ `out ∈ ℝ^(n_q × d)`
- **the output has exactly as many rows as there are queries.** `n_kv` never appears in the output shape.

```
cost = O(n_q · n_kv · d)
```

Self-attention _chooses_ `n_q = n_kv = N`. That choice — not any law — produces `N²`.

> Set `n_q = M` (yours to choose) and `n_kv = N` (the data's):
>
> - cost becomes `O(M · N · d)` — **linear in N**
> - output becomes `M × d` — **a shape you chose, not one the data imposed**

With `M = 512`, `N = 50,176`: 25.7M operations instead of 2.5B. Factor `N/M = 98`.

Two questions follow immediately, and they organise the rest of the architecture:

1. **Where do `M` queries come from,** when the input contains no such thing? → §4
2. **Does a single `N → M` read lose too much?** (13.1M numbers → 0.52M numbers) → §7, §11

---

## 3. Cross-attention in full

```
Inputs:  X ∈ ℝ^(N×C)     byte array   (frozen, external)
         Z ∈ ℝ^(M×D)     latent array (current state)

1. Pre-normalisation
   Z̃ = LN(Z)                          ∈ ℝ^(M×D)
   X̃ = LN(X)                          ∈ ℝ^(N×C)

2. Projections
   Q = Z̃ W_Q     W_Q ∈ ℝ^(D×d)        Q ∈ ℝ^(M×d)    ← query from the LATENT
   K = X̃ W_K     W_K ∈ ℝ^(C×d)        K ∈ ℝ^(N×d)    ← key   from the INPUT
   V = X̃ W_V     W_V ∈ ℝ^(C×d)        V ∈ ℝ^(N×d)    ← value from the INPUT

3. Attention
   S = Q Kᵀ / √d                       ∈ ℝ^(M×N)
   A = softmax_rows(S)                 ∈ ℝ^(M×N)     each row sums to 1 over N
   H = A V                             ∈ ℝ^(M×d)

4. Output projection + residual
   Z ← Z + H W_O          W_O ∈ ℝ^(d×D)

5. MLP + residual
   Z ← Z + MLP(LN(Z))
   MLP(u) = W₂ · GELU(W₁ u + b₁) + b₂
```

**Reading each line properly:**

- **`W_Q ∈ ℝ^(D×d)` vs `W_K ∈ ℝ^(C×d)` — different input widths.** This is the structural difference from self-attention. Latents carry `D = 1024`, pixels carry `C = 261`. Separate projections meeting at a shared `d`.
- **`A ∈ ℝ^(M×N)` is the object to understand.** Row `m` reads as: _"how much does latent `m` weight input element `n`."_ Reshaped to 224x224, each row is a heat map over the whole image. Latent 1 may attach to edges, latent 47 to central texture. The paper visualises these; they are worth staring at.
- **Softmax is over `N`, not over `M`.** Each latent spends its full budget across all inputs, but **no input element is guaranteed a reader**. A pixel that every latent ignores contributes nothing. The compression is genuinely lossy, and lossy in a _selected_ rather than uniform way. _(Flipping this softmax axis is what turns Perceiver into Slot Attention — see §15.)_
- **The residual `Z ← Z + ...` is semantic, not cosmetic.** The latent **accumulates** rather than being **replaced**. `Z` is a residual stream that gains content each round. §8.7 shows this is also the only reason the iterated system is trainable at all.
- **Pre-LN, not post-LN.** Perceiver uses `Z + f(LN(Z))`, not `LN(Z + f(Z))`. The residual path is then a clean identity, so `∂Z_out/∂Z_in = I + (something)`. With post-LN a LayerNorm sits on the identity path and distorts it. Load-bearing under weight sharing.

**Multi-head.** Split into `h` heads of width `d/h`, attend independently, concatenate, project:

```
headᵢ     = softmax( Z̃W_Q^i (X̃W_K^i)ᵀ / √(d/h) ) · X̃W_V^i     ∈ ℝ^(M × d/h)
MultiHead = [head₁ ; … ; head_h] W_O
```

The ImageNet Perceiver uses **one head** for cross-attention — the `M×N` attention matrix has 25.7M entries and each head multiplies that memory — and 8 heads inside the latent transformer.

---

## 4. The latent array Z0

`Z₀ ∈ ℝ^(M×D)` is **an ordinary learnable parameter**, like any weight matrix.

From the paper: the initial latents are learned per-element weights with the same shape as the latent array (512 x 1024 for ImageNet). They function like learned position encodings in the Transformer literature, or like a learned initial state in the RNN literature. They are initialised from a truncated normal (mean 0, std 0.02, bounds [-2, 2]), and performance is fairly robust to the scale of this initialisation.

**"Random" applies only to initialisation before training.** Afterwards `Z₀` is a learned tensor, identical for every input — a dog image and a cat image both start from the same `Z₀`. So it encodes something input-independent:

> **`Z₀` is a set of `M` learned questions: what this model has discovered is worth asking about the world.**

Latent 1 might have learned to ask _"where are the vertical edges"_, latent 47 _"what is in the central region"_. Nobody supervised this.

**Why this matters for a goal/fact split:**

```
Z₀  =  goal / query prior   — learned from data, independent of any particular input
X   =  fact / evidence      — changes with every input
A   =  the matching of goals against facts
```

Perceiver already _has_ this split structurally, but implicitly: `Z₀` is just numbers with no enforced semantics. Making it explicit — splitting `Z` into a structured `Z_goal` and a free `Z_work` — is the natural place to intervene.

**One caveat the paper raises against itself.** Performance is robust to the scale of `Z₀`'s initialisation. If `Z₀` carried as much load as the framing suggests, initialisation scale should matter more. The honest reading is that **the operator matters more than the starting point** — worth remembering if you are designing something where the initial state is supposed to carry structure.

---

## 5. The latent transformer

After a cross-attend, `Z ∈ ℝ^(M×D)` passes through `L` ordinary self-attention blocks:

```
for l = 1..L:
    Z ← Z + MHSA(LN(Z))       Q, K, V all from Z
    Z ← Z + MLP(LN(Z))
```

The attention matrix here is `M×M` = 512 x 512 = 262,144 entries versus `N²` = 2.5e9 — a factor of **9,600** cheaper.

**Why it is needed.** Cross-attention never lets latents talk to each other. Each latent reads the image alone, unaware of what its neighbours found. Self-attention is where the `M` latents exchange information and assemble a higher-level representation.

**The division of labour — the reason this architecture is a loop container:**

```
cross-attention  =  READ    expensive, O(M·N), touches the outside world
self-attention   =  THINK   cheap,     O(M²),  purely internal
```

The paper allocates **6 self-attention blocks per cross-attend**: read rarely, think a lot. A standard Transformer cannot make this trade — for it, reading and thinking are the same operation and cannot be separately budgeted.

> **Terminology.** The literature calls this **read–process–write** (inherited from Neural Turing Machines and Memory Networks). "Read / think" is the same thing with `process` renamed to stress compute allocation. The Perceiver papers do not use "read/think".
>
> The clean test for which is which: **does `X` appear in the equation?** If yes it is a read and its cost scales with `N`. If no it is thinking and its cost is independent of `N`. Not the presence of an MLP (cross-attention blocks have MLPs too), and not the relative size of `Q` versus `K`.

---

## 6. Position encoding

### 6.1 The problem

`H = A V = Σₙ aₘₙ vₙ` is a weighted **sum**, hence commutative. Permute the rows of `X` and the output is bit-identical.

> Cross-attention is **permutation invariant with respect to the byte array.** To the model, an image is a _bag of pixels_, not a grid.

CNNs do not have this problem (convolution has grid structure built in). ViT does not (patches are cut on a grid). Perceiver deliberately refuses both priors — that is the generality claim — so **position must be injected as data**.

### 6.2 Fourier features

For a point with coordinates `x = (x₁, …, x_δ)`, where `δ` is the number of spatial dimensions (2 for images, 3 for video or point clouds), normalised to `[-1, 1]`:

```
γ(x) = [ sin(f₁ π x₁), cos(f₁ π x₁), …, sin(f_K π x₁), cos(f_K π x₁),
         sin(f₁ π x₂), cos(f₁ π x₂), …, sin(f_K π x₂), cos(f_K π x₂),
         …,
         x₁, x₂, …, x_δ ]
```

- `K` — number of frequency bands. **64** for ImageNet.
- `f_k` — frequencies spaced **linearly** from 1 to `μ/2`, where `μ` is the target resolution (**224**). `μ/2 = 112` is the **Nyquist frequency**: the highest frequency this sampling rate can represent. Above it you get aliasing, i.e. garbage.
- `sin` **and** `cos` are both required: `(sin θ, cos θ)` determines `θ` uniquely, whereas `sin θ` alone is ambiguous since `sin θ = sin(π − θ)`.
- `π` makes the phase complete exactly at `x = ±1`.
- Raw coordinates are appended so a direct low-frequency signal is available.

Dimension: `2·K·δ + δ = 2·64·2 + 2 = 258`, plus 3 RGB → **C = 261**.

### 6.3 Concatenate, do not add

Standard Transformers **add** position embeddings because the widths match. Perceiver **concatenates**:

```
X[n] = [ RGB(n) ; γ(pos(n)) ]        3 + 258 = 261 channels
```

Adding 258 dimensions of position onto 3 dimensions of colour would obliterate the colour. Concatenation keeps them in separate subspaces and lets `W_K` learn the mixing.

### 6.4 Why Fourier rather than raw coordinates

The dot product between Fourier features of two positions depends **only on their separation**:

```
γ(x) · γ(x′) = Σₖ [ sin(fₖπx)sin(fₖπx′) + cos(fₖπx)cos(fₖπx′) ]
             = Σₖ cos( fₖ π (x − x′) )
```

A shift-invariant kernel, for free, before any training. Attention can compare positions as _"how far apart"_ immediately. With raw coordinates the dot product `x·x′` depends on **absolute** position — a harder function to learn and a worse prior.

This is Random Fourier Features (Rahimi & Recht 2007), ultimately Bochner's theorem. See [[8-position-transformer]] for the wider treatment.

---

## 7. Iteration

### 7.1 Counting the bits

```
input :  N × C = 50,176 × 261 ≈ 13.1M numbers
latent:  M × D = 512 × 1024   ≈ 0.52M numbers
compression ratio ≈ 25 : 1
```

**This ratio is important and frequently misstated.** The latent is _not_ a single 1024-dimensional vector; it is a `512 x 1024` array. That distinction separates Perceiver from a conventional autoencoder:

```
autoencoder :  compress to one vector       →  a genuine hard bottleneck
Perceiver   :  compress to a 512-row array  →  a bottleneck in the number of SLOTS,
                                               not in raw capacity
```

512 rows means 512 mutually distinct things can be held **separately** rather than averaged together. This is working memory with 512 registers, not a summary vector.

### 7.2 Read, think, read again

```
Z ← Z₀
repeat T times:
    Z ← CrossAttentionBlock(Z, X)     read   O(M·N·D)
    Z ← LatentTransformer(Z)          think  O(L·M²·D)
```

**The conceptual point.** On round 2, `Z` is no longer `Z₀`; it contains content from the image. Therefore `Q = LN(Z)W_Q` is a **new set of questions conditioned on what was just read**.

```
round 1: "what is in this image?"            → finds fur
round 2: "given fur, where are the ears?"    → reads different pixels
round 3: "the ears are pointed; whiskers?"   → …
```

This is **content-based iterative retrieval** — several purposeful glances rather than one stare. `X` is constant (the world), `Z` evolves (the state), the operator is constant (shared weights). The structure is a genuine loop, not a deep network shaped like one.

The best ImageNet configuration attends to the image 8 times, each time processing the full 50,176-element input with a cross-attend module plus a 6-block latent Transformer, using a single head per cross-attend.

---

## 8. Weight sharing — the deep version

Most explanations of weight sharing stop at "fewer parameters, less overfitting". That is true and almost useless. The interesting content is what sharing does to the **gradient** and to the **class of functions the model can be**.

### 8.1 Parameter inventory

You cannot reason about sharing until you can name every tensor.

**One cross-attention block:**

```
LN_q      : γ, β ∈ ℝ^D                    normalise latent before Q
LN_kv     : γ, β ∈ ℝ^C                    normalise input before K, V
W_Q       : ℝ^(D×d)    1024 × 1024        latent → query
W_K       : ℝ^(C×d)     261 × 1024        input  → key
W_V       : ℝ^(C×d)     261 × 1024        input  → value
W_O       : ℝ^(d×D)    1024 × 1024        back into the residual stream
LN_mlp    : γ, β ∈ ℝ^D
W_1, W_2  : ℝ^(D×D) each                  MLP
```

**One self-attention block:**

```
LN_1        : γ, β ∈ ℝ^D
W_Q,W_K,W_V : ℝ^(D×D) each                all three come from Z, so all are D-wide
W_O         : ℝ^(D×D)
LN_2        : γ, β ∈ ℝ^D
W_1, W_2    : ℝ^(D×D) each
```

The only structural difference: cross-attention's `W_K, W_V` take `C`-wide input, self-attention's take `D`-wide input.

### 8.2 What is actually shared

The ImageNet configuration runs 8 rounds, each = 1 cross-attend + 6 self-attention blocks:

```
round 1:  C₁  S¹ S² S³ S⁴ S⁵ S⁶
round 2:  C₂  S¹ S² S³ S⁴ S⁵ S⁶
...
round 8:  C₈  S¹ S² S³ S⁴ S⁵ S⁶
```

That is **8 cross-attends + 48 self-attends = 56 blocks executed**. The paper shares weights between cross-attention modules 2–8, and between _corresponding_ blocks of all latent Transformers. The first cross-attention module uses its own unshared weights.

So 56 executed blocks draw on **8 distinct weight sets**: `C₁`, `C*`, and `S¹ … S⁶`.

**The rule, stated precisely:**

```
sharing is ACROSS ROUNDS (horizontal), never ACROSS POSITIONS (vertical).
S¹ and S² are different tensors. S¹ in round 3 and S¹ in round 7 are the same tensor.
```

In code:

```python
cross_first = CrossAttnBlock()                 # C₁
cross_rest  = CrossAttnBlock()                 # C*
selfs       = [SelfAttnBlock() for _ in range(6)]   # S¹..S⁶

Z = self.latent_init                           # Z₀ is also a parameter, M×D
for t in range(8):
    Z = (cross_first if t == 0 else cross_rest)(Z, X)
    for s in selfs:                            # the SAME six objects every round
        Z = s(Z)
```

The `for` loop does not create layers. It calls the same objects again. That is the whole of weight sharing.

### 8.3 Parameter counting: 326M → 45M

With `D = 1024`, `C = 261`, and MLPs that do not widen (the paper notes the dense subblock uses no bottleneck):

**Self-attention block:**

```
W_Q, W_K, W_V, W_O :  4 × (1024×1024) = 4.19M
MLP W₁, W₂         :  2 × (1024×1024) = 2.10M
LN + biases        :  ~0.01M
                                        ≈ 6.3M
```

**Cross-attention block:**

```
W_Q  : 1024×1024      = 1.05M
W_K  :  261×1024      = 0.27M      ← smaller: input has only 261 channels
W_V  :  261×1024      = 0.27M
W_O  : 1024×1024      = 1.05M
MLP  : 2×(1024×1024)  = 2.10M
                        ≈ 4.7M
```

**Assembled:**

|            | formula                         | total      |
| ---------- | ------------------------------- | ---------- |
| no sharing | 8 cross × 4.7M + 48 self × 6.3M | ≈ **340M** |
| sharing    | 2 cross × 4.7M + 6 self × 6.3M  | ≈ **47M**  |

The paper reports **326.2M** and **44.9M**. The estimate is within ~5%; residual discrepancies are biases and the exact `qk_channels` setting. Close enough to confirm the structure.

### 8.4 The number that reframes everything: FLOPs are identical

From the paper's ablation table:

|                     | Valid | Train | Params | FLOPs  |
| ------------------- | ----- | ----- | ------ | ------ |
| no weight sharing   | 72.9  | 87.7  | 326.2M | 707.2B |
| with weight sharing | 78.0  | 79.5  | 44.9M  | 707.2B |

**FLOPs are the same to the decimal.** Sharing runs the same 56 blocks and performs the same matrix multiplications. It saves _storage_, not _time_.

> **Therefore weight sharing is not a compute optimisation. It is a change of hypothesis space.** This sentence is the key to the rest of the section.

Also read the accuracy columns carefully: **train accuracy drops 87.7 → 79.5**. The model memorises the training set _worse_. Capacity was genuinely removed, not merely redistributed. Validation rises 72.9 → 78.0 and the generalisation gap collapses from 14.8 points to 1.5. Textbook bias–variance.

### 8.5 What happens in the backward pass

Forward is trivial: call the same object again. The interesting behaviour is backward.

**The one rule:** if a parameter `θ` is used at several points in the graph, its gradient is the **sum over all uses**.

```
∂L/∂θ_C*  =  Σ_{t=2}^{8}  [ ∂L/∂(output of C* at round t) · ∂C*/∂θ |_{Z_{t-1}} ]
             └──────────────────── 7 terms ────────────────────┘
```

Each term is evaluated at a **different `Z`**. The `t=2` term is evaluated at `Z₁` (one glance taken), the `t=8` term at `Z₇` (seven glances taken). Those live in very different regions of latent space.

**The gradient becomes a vote:**

```
θ ← θ − η · ( g₂ + g₃ + g₄ + g₅ + g₆ + g₇ + g₈ )
```

Each `gₜ` says _"this operator should move in direction X — in the context of round t."_

| case                            | consequence                                                                      |
| ------------------------------- | -------------------------------------------------------------------------------- |
| all rounds agree (`gₜ` aligned) | terms reinforce; strong signal; and it is a signal that is true in every context |
| rounds disagree (`gₜ` opposed)  | terms cancel; the operator settles on a compromise, or one term dominates        |

> **This is the real mechanism by which sharing regularises.** Any update that helps one round while hurting others is cancelled automatically. What survives is only what is **true in every context** — which is the operational definition of generalisation.

### 8.6 Function-space view

```
no sharing :  model = f₈ ∘ f₇ ∘ … ∘ f₁         each fᵢ chosen independently
              hypothesis space = 𝓕⁸

sharing    :  model = f ∘ f ∘ … ∘ f            one f
              hypothesis space = { (f,f,…,f) : f ∈ 𝓕 }   ← the diagonal of 𝓕⁸
```

The second set is vastly smaller — **but smallness is not the point.** If you only wanted smaller you could shrink `D` from 1024 to 400 and match the parameter count.

The point is that the constraint is **physically meaningful**:

> _"The rule for reading an image should not depend on which glance this is."_

Humans do not use a different pair of eyes on the fifth look.

**The consequence that matters most:**

```
no sharing  →  an 8-station assembly line; depth means "many layers"
sharing     →  one operator applied 8 times:  Z_{t+1} = f(Z_t ; X)
               a genuine dynamical system / unrolled RNN
```

> **Weight sharing does not make a model smaller. It changes what kind of object the model is: from a pipeline into an iteration.**

An unshared model has no right to be called a loop — each round is a different function. Only the shared model has a fixed point to talk about, stability to analyse, and (in principle) an adjustable number of iterations at inference.

### 8.7 Why `C₁` must be excluded — and which axis the problem lives on

The paper reports the empirical fact: sharing the initial cross-attention with subsequent cross-attends led to instability in training, so they share all cross-attends after the first. The analysis below is mine, not the paper's, but it accounts for the observation.

**Look at the gradient of `W_Q` in round 1:**

```
round t=1:   Q = LN(Z₀) W_Q

∂L/∂W_Q |_{t=1}  =  LN(Z₀)ᵀ · G₁          where G₁ = ∂L/∂Q
                    └──────┘
                    a CONSTANT matrix — no dependence on the input at all
```

Two consequences:

**(a) The gradient direction is rank-limited and frozen.** Every round-1 gradient lies in the column space of `LN(Z₀)ᵀ`, rank ≤ M, and that subspace never changes during training.

**(b) It does not average out across the batch.** Gradients from rounds `t ≥ 2` depend on the individual image; averaging over a batch cancels sample-specific noise and leaves the signal, so the mean grows like `√B` (a random walk). The round-1 gradient shares the factor `LN(Z₀)ᵀ` across the entire batch, so it adds **coherently** and grows like `B`.

```
rounds 2-8:  ‖mean gradient‖ ~ √B
round 1   :  ‖mean gradient‖ ~ B          →  with B = 1024, a 32x advantage
```

So `W_Q` receives one persistent, low-rank, batch-coherent push and one diverse, sample-dependent push. The first can dominate, dragging `W_Q` toward serving a context-free query. Rounds 2–8 then read worse, loss rises, gradients grow, and it diverges.

DeepMind's fix is blunt and effective: give round 1 its own weights.

**The generalisable lesson, which recurs in any loop you build:**

> **A step that begins with "nothing in mind" and a step that begins with "something in mind" are different operators by nature. Forcing them to be one operator is the failure mode.**

Three ways out (Perceiver takes the first):

1. **Separate the bootstrap step.** Simple; costs you the purity of the loop — there is no longer a single fixed point.
2. **Give `Z₀` context from the start,** e.g. initialise it from a pooling of `X`, so round 1 resembles later rounds.
3. **Keep one weight set but add a per-round gate** (a few scalars), letting behaviour vary without duplicating parameters.

### 8.8 Which round's gradient is largest?

A natural worry: `g₈` sits closest to the loss, so surely it dominates.

Write the backward chain. Let `Z_t = F(Z_{t-1}; θ)` where `F` is the whole (cross + 6 self) block, and let `δ = ∂L/∂Z₈`, `J_s = ∂F/∂Z |_{Z_{s-1}}`:

```
∂L/∂Z₈ = δ
∂L/∂Z₇ = δ J₈
∂L/∂Z₆ = δ J₈ J₇
…
∂L/∂Z₂ = δ J₈ J₇ J₆ J₅ J₄ J₃

‖gₜ‖ ∝ ‖δ‖ · ∏_{s=t+1}^{8} ‖J_s‖ · ‖∂F/∂θ|_{Z_{t-1}}‖
```

So everything reduces to the spectral radius `ρ(J)`:

| regime                   | outcome                                                                         | name               |
| ------------------------ | ------------------------------------------------------------------------------- | ------------------ |
| `ρ(J) < 1` (contractive) | `g₈ ≫ g₂` — the operator is tuned for late rounds; early rounds barely learn    | vanishing gradient |
| `ρ(J) > 1` (expansive)   | **`g₂ ≫ g₈`** — the _furthest_ round dominates, because it is amplified 6 times | exploding gradient |
| `ρ(J) ≈ 1`               | all rounds have comparable voice                                                | what we want       |

**The middle row is the counter-intuitive one: distance from the loss does not decide the winner. Gain does.**

**How sensitive is this?** One round contains 14 residual sublayers (cross-attn: attn + MLP = 2; six self blocks: 6 × 2 = 12). From `g₂` to the loss is 6 rounds = **84 residual sublayers**. If each sublayer has gain `(1+ε)`:

```
ε = +0.05  →  1.05^84 ≈ 60x      g₂ is 60x larger than g₈
ε =  0     →  1                  equal
ε = −0.05  →  0.95^84 ≈ 0.013    g₂ is 77x smaller than g₈
```

A 5% per-layer deviation produces a 60–77x imbalance at the far end.

**This is why residual + pre-LN are load-bearing rather than decorative:**

- **residual** makes `J = I + (∂h/∂Z)`, pinning `ρ` near 1 by construction
- **pre-LN** keeps the identity path clean, so `J` genuinely has an identity term. With post-LN a LayerNorm sits on that path and destroys it — this is exactly why modern Transformers moved to pre-LN
- **small-initialised `h`** plus LayerNorm keeps `‖∂h/∂Z‖ ≪ 1` at the start

> Your intuition that `g₈` dominates is correct **for a network without residuals**. Perceiver survives because the residual structure gives every round a comparable voice. That property is the precondition for weight sharing being viable at all.

### 8.9 Magnitude is the wrong thing to worry about

```
case A:  g₂ = 0.1·u,  g₈ = 10·u        aligned
         sum = 10.1·u  →  merely a larger effective learning rate. Harmless.

case B:  g₂ = 1.0·u,  g₈ = 1.0·(−u)    equal magnitude, opposite direction
         sum = 0      →  the operator does not move at all,
                         even though both rounds want it to move.
```

**Case B is the dangerous one**, and it looks perfectly balanced by any norm-based metric. It is the signature of one operator being asked to serve two conflicting roles.

**The diagnostic to log is therefore cosine similarity between per-round gradients, not their norms:**

```
cos(g₂, g₈) ≈ +1   →  all rounds agree; weight sharing is the right prior     ✓
cos(g₂, g₈) ≈  0   →  early and late rounds do unrelated work, but do not fight
cos(g₂, g₈) ≈ −1   →  direct conflict; split the weights or add a gate        ✗
```

There is also a mild self-correcting pressure: if `g₈` dominates and early rounds degrade, `Z₂` gets worse, all later rounds receive worse input, loss rises, `δ` grows, and every gradient including `g₂` grows. But it is only a pressure, not a guarantee — the equilibrium may be "early rounds are mediocre" rather than "all rounds are equally good".

### 8.10 The practical consequence people forget

Gradients **sum**; they do not average.

```
unshared parameter        :  1 gradient contribution
parameter shared 7 ways   :  ≈ 7x  (if aligned)  or  ≈ √7 ≈ 2.6x  (if random directions)
```

> **A shared parameter has an effective learning rate 2.6–7x higher than an unshared one, automatically.**

Convert a model from unshared to shared, keep the same LR, and it will diverge — and you will misdiagnose it as a convergence problem in the loop when it is simply too large a step.

```python
# option 1: normalise at backward time
param.grad /= n_shared_uses

# option 2: scale LR down by ~1/T and retune upward
```

Relevant aside: the paper found these models easier to optimise with **LAMB** than with SGD. LAMB performs layer-wise normalisation of the update, i.e. it divides out the gradient norm — which incidentally hides the sharing-induced scale problem.

### 8.11 Measuring it

```python
grads = {}

def make_hook(t):
    def hook(module, grad_in, grad_out):
        grads[t] = grad_out[0].detach()
    return hook

# register a hook on the shared block's output for each call index t
# then log both:
#   norm ratio      : grads[8].norm() / grads[2].norm()
#   directional fit : F.cosine_similarity(grads[8].flatten(), grads[2].flatten(), dim=0)
```

Interpretation:

```
norm ratio  > 10   →  vanishing regime (ρ < 1), late rounds dominate
norm ratio  < 0.1  →  exploding regime (ρ > 1), early rounds dominate
cosine      < 0    →  the shared operator is being asked to do two conflicting jobs
```

Log both from day one of any loop you build. Together they diagnose almost everything.

### 8.12 What DEQ does differently

Every issue above stems from **unrolling**: `T` terms weighted unevenly by distance.

Deep Equilibrium Models do not unroll. They solve `Z* = f(Z*; X)` with a root-finder and differentiate through the implicit function theorem:

```
∂L/∂θ  =  (∂L/∂Z*) (I − J*)⁻¹ (∂f/∂θ)|_{Z*}
                    └────────┘
          the effect of infinitely many iterations, in one term
```

- **no `Σₜ`** — the question "which round gets more gradient" dissolves; there are no rounds
- **no repeated Jacobian products** — no depth-driven explosion or vanishing
- `(I − J*)⁻¹ = I + J* + J*² + …` (Neumann series), which converges **iff `ρ(J*) < 1`** — so the stability condition becomes explicit rather than implicit
- **memory is O(1)** in the number of iterations, since no intermediate activations are stored

The price: `ρ(J*) < 1` must actually hold, and a root-finder runs on every forward pass.

See [[6-fixed-point-and-what-transformer-chasing]] for the fixed-point framing in general.

---

## 9. The Hopfield connection

### 9.1 Where the correspondence holds

Modern continuous Hopfield (Ramsauer et al. 2020) update rule:

```
ξ_new = Xᵀ softmax(β X ξ)

X ∈ ℝ^(N×d)  stored patterns
ξ ∈ ℝ^d      state / query
β            inverse temperature
```

Perceiver cross-attention:

```
Z_new = softmax( (ZW_Q)(XW_K)ᵀ / √d ) · (XW_V)
```

Mapping:

| Hopfield              | Perceiver                                       |
| --------------------- | ----------------------------------------------- |
| `ξ` (state / query)   | `Z` (latent, `M` of them in parallel)           |
| `X` (stored patterns) | `X` (byte array)                                |
| `β`                   | `1/√d`                                          |
| `X ξ`                 | `(ZW_Q)(XW_K)ᵀ` — same, plus a learnable metric |
| `Xᵀ softmax(·)`       | `softmax(·)(XW_V)`                              |
| one update step       | one cross-attention                             |

**Perceiver is a Hopfield network laid out as an architecture.** In particular `Z₀` is the _partial cue_ that triggers retrieval — the fragment you present in order to recall the whole pattern — except made **learnable** and instantiated `M` times in parallel. Iteration is iterative retrieval.

### 9.2 Where it breaks — which matters more

**(a) The energy function is gone.** The original result relies on an energy

```
E(ξ) = −lse(β, Xξ) + ½ ξᵀξ + const          (lse = log-sum-exp)
```

which the update provably decreases, giving guaranteed convergence. That proof needs **symmetry** between reading and writing: `Xξ` out, `Xᵀ(·)` back, with the same `X`.

Perceiver uses **different** `W_K` and `W_V`, so `XW_K ≠ XW_V`. The operator is asymmetric, **no energy function exists, and there is no convergence guarantee of any kind.**

**(b) Self-attention sits between retrievals.** Hopfield is `ξ → retrieve → ξ`. Perceiver is `Z → retrieve → think → retrieve → …`. Self-attention among latents is not a retrieval; it is interaction among state components, which makes the dynamical system strictly richer than Hopfield.

**(c) `M` coupled states, not one.** Classical Hopfield has a single state. Perceiver has `M` states that talk to each other — a coupled dynamical system.

> **If you want a convergence guarantee back, you must pay for it:**
>
> - tie `W_K = W_V` to restore symmetry (an energy function returns; expressiveness drops)
> - add an explicit gate that bounds the spectral radius below 1
> - or abandon the guarantee and use a DEQ solver with implicit differentiation instead of unrolling

See [[4-hopfield-internal]] and [[6-fixed-point-and-what-transformer-chasing]].

---

## 10. Perceiver IO

### 10.1 What was fatally wrong with the original decoder

Perceiver's decoder is:

```
z̄      = (1/M) Σ_{m=1..M} Z[m]        ∈ ℝ^D          global average pool
logits = z̄ W_cls                       ∈ ℝ^1000
```

Count what that discards:

```
Z before pooling :  512 × 1024 = 524,288 numbers
z̄ after pooling  :             1024 numbers
```

**99.8% is thrown away — and not randomly.** It discards precisely _which latent said what_, keeping only the average. If latent 17 specialised in "what is in the top-left" and latent 204 in "which direction is motion", pooling averages them with 510 others and the operation is not invertible.

The deeper problem is the output shape:

```
segmentation  needs   50,176 values
optical flow  needs   50,176 × 2 values
language      needs    2,048 × vocab values
average pool  gives         1 value
```

You cannot patch this with a large MLP: `W ∈ ℝ^(1024 × 100,352)` is 100M parameters per task, and every output element would be computed from the _same_ vector `z̄` — no per-element query, no specificity.

> **The original Perceiver solves "large inputs" elegantly and does not touch "large outputs" at all.** The paper's own framing: much of the complexity of real tasks comes from the variety, size, and structure of their outputs, and in that respect the original Perceiver cannot be considered general purpose.

### 10.2 The decoder, in full

The paper's stated key insight: produce each output by attending to the latent array using a specific output query associated with that particular output.

```
Inputs:  Z     ∈ ℝ^(M×D)     the processed latent
         Q_out ∈ ℝ^(O×E)     the output query array — YOU construct this

Q = LN(Q_out) W_Q'    W_Q' ∈ ℝ^(E×d)     Q ∈ ℝ^(O×d)
K = LN(Z)     W_K'    W_K' ∈ ℝ^(D×d)     K ∈ ℝ^(M×d)
V = LN(Z)     W_V'    W_V' ∈ ℝ^(D×d)     V ∈ ℝ^(M×d)

A_out = softmax_rows(Q Kᵀ / √d)          A_out ∈ ℝ^(O×M)
Y     = A_out V W_O'                     Y     ∈ ℝ^(O×E')
```

**Line by line:**

- `Q_out[o]` — the _question card_ for output element `o`. E.g. _"at pixel (100, 42), what is the optical flow?"_
- `A_out[o, :]` — a distribution over the `M` latents: _"where is my answer stored?"_ Softmax is over `M` here.
- `Y[o]` — the answer: a weighted average of latent content.
- `O` is **independent of both `N` and `M`.** Fully your choice.

Cost `O(O · M · d)` — **linear in `O`**.

### 10.3 The whole architecture is one operation

Compare the encoder's `A_in ∈ ℝ^(M×N)` with the decoder's `A_out ∈ ℝ^(O×M)`.

**They are the same operation with the query holder swapped:**

```
encode  :  Q from Z (M rows),      K,V from X (N rows)   →   N → M   compress
process :  Q,K,V   from Z (M rows)                       →   M → M   compute
decode  :  Q from Q_out (O rows),  K,V from Z (M rows)   →   M → O   expand
```

There is exactly one primitive in this architecture: QKV attention. Everything else is a choice of who holds `Q`. This symmetry is the aesthetic core of Perceiver IO.

### 10.4 The mental model to use

> **`Z` is not a feature vector. `Z` is a queryable database.**

```
encode  =  WRITE     transcribe the world into M rows
process =  COMPUTE   reorganise and infer within the database
decode  =  READ      issue any queries you like, as many as you like
```

And because `Z` is a database, you can fire **any query set, any number of times, against the same `Z`**. One latent array can simultaneously produce a dense flow field, a token sequence, and a class label.

### 10.5 Constructing output queries — where the craft is

`Q_out` is **not necessarily a learnable parameter.** It is assembled from three ingredients:

```
Q_out[o] = concat[ position_encoding(o) , task_embedding , input_feature(o) ]
                   └─ which one? ─┘       └─ what job? ─┘   └─ raw evidence ─┘
                     (mandatory)          (if multi-task)   (if dense task)
```

**The one hard requirement — identifiability.** If `Q_out[i] = Q_out[j]` then `Y[i] = Y[j]` necessarily: same function, same input. Omit position encoding on a dense task and every pixel gets the same answer.

#### Classification — `O = 1`

```
Q_out = w_cls ∈ ℝ^(1×E)      a single learned vector
Y     = A_out V ∈ ℝ^(1×1000)
```

`A_out ∈ ℝ^(1×512)` asks _"to decide what this image is, which latents should I listen to?"_

Compare to average pooling, which hard-codes `A_out[0,m] = 1/M` for all `m`. Perceiver IO turns pooling into **learned, content-dependent pooling** — which is why IO improves even on classification, a task the original could already do.

#### Optical flow — the showcase

Target: per-pixel motion vectors, `O = H·W`, `E' = 2`.

The paper's illustration: to predict flow at one pixel, compose a query from that pixel's xy coordinates plus an optical-flow task embedding. In practice they go further and include the pixel's own input features:

```
Q_out[o] = concat[ γ(x_o, y_o) , X[o] ]
                                 └──┬──┘
                        the raw feature of this very pixel
```

**Why this is the most important line in the paper.**

```
query = position only:
    the model must recover "what is at (100,42)" from Z alone
    Z does not hold that at pixel resolution  →  blurry output

query carries the feature:
    pixel detail arrives at the destination directly
    Z only has to supply high-level context (what is moving, and where)
```

> **This is a U-Net skip connection disguised as a query.** Unlike a U-Net skip it imposes no resolution-matching constraint at all — it is a vector concatenation.

And it works: Perceiver IO reaches state of the art on Sintel optical flow, beating PWCNet and RAFT, **without any architectural machinery for multiscale correspondence** — the component every specialist flow model is built around.

#### Language — no tokenizer

`O = 2048` (sequence length), `E' = 256` (byte vocabulary):

```
Q_out[o] = learned_position_embedding(o)     ∈ ℝ^E
```

Perceiver IO matches a Transformer-based BERT baseline on GLUE **with no input tokenization**, consuming raw UTF-8 bytes.

**Why that is possible.** Tokenizers exist to fight `O(N²)`: sequences are too long, so merge characters into words. Once complexity is `O(MN)`, the reason for the tokenizer evaporates.

> A clean instance of a general pattern: _a tool built to work around a constraint becomes unnecessary when the constraint is removed._ Tokenization is not something language needs; it is something Transformers needed.

#### Multimodal autoencoding — the extreme case

Input: video + audio. Output: video + audio + class label, simultaneously.

```
Q_out = vertical concat of [
    one query per video pixel   : [ γ(t,x,y) ; emb_video ]
    one query per audio sample  : [ γ(t)     ; emb_audio ]
    one query for the class     : [           emb_label ]
]
```

For outputs with multi-task or multimodal structure, the paper learns a single query per task or per modality. All query types are padded to a common width `E`, stacked into one array, and passed through **one** cross-attention. The modality embedding is what tells the model _"this row is asking about vision, not audio."_

#### StarCraft II — symbolic set output

One query per game unit. The output is an **unordered set** — neither a sequence nor a grid. Perceiver IO can replace the Transformer used in AlphaStar.

### 10.6 The property nobody discusses: outputs are conditionally independent

The decoder is a **single** cross-attention. There is **no self-attention among outputs**.

```
Y[o]  depends on  Q_out[o]  and  Z
Y[o]  does NOT depend on  Y[o']  for any o' ≠ o
```

In probabilistic language: **outputs are conditionally independent given `Z`.**

**Three benefits:**

1. **Linear, not quadratic.** Self-attention among outputs would cost `O(O²)`; at `O = 50k` that is fatal. Its absence is why IO scales.
2. **Full parallelism.** All `O` outputs compute simultaneously; no dependencies.
3. **Training-time subsampling — the elegant one.** The paper notes StarCraft has up to 800,000 output points, prohibitively expensive to decode at once even with linear scaling; so they **subsample the output array at training time** and compute the loss on an affordable subset.

```
train     :  sample 1,000 of 800,000 queries    →  cheap
inference :  issue all 800,000                  →  full output
```

**This works only because outputs are independent.** Skipping `Y[5]` does not perturb `Y[9]`. Contrast a CNN, where you cannot subsample outputs because neighbouring pixels are computed by shared kernels. IO gets this property for free from having a single-layer decoder.

**The cost: no autoregressive generation.**

```
Y[o] cannot see Y[o-1]  →  "what was the previous token" is not expressible in the query
```

Hence IO's language work is **masked LM (BERT-style)**, not generation (GPT-style), and hence Perceiver never became an LLM backbone. All coherence among outputs must be mediated by `Z`; if `Z` does not encode a constraint, outputs can contradict each other (e.g. discontinuous segmentation boundaries).

> **The sharpest trade in Perceiver IO: it exchanges inter-output coherence for scalability.**

### 10.7 Cost and the three-way decoupling

```
encode  :  O(M · N · D)
process :  O(L · M² · D)
decode  :  O(O · M · D)
─────────────────────────
total   :  O( M·D·(N + O) + L·M²·D )
```

Perceiver IO still decouples model depth from data size and still scales linearly — but now with respect to **both** input and output size.

**What that actually unlocks.** In every prior architecture three things were welded together:

```
CNN          :  input resolution ↔ depth ↔ output resolution
Transformer  :  N ↔ compute ↔ N
ViT          :  #patches ↔ compute ↔ #patches

Perceiver IO :  N   ⊥   (M, L)   ⊥   O          three independent knobs
```

Configurations that become expressible:

- input 800k points, latent 512, output 10 values
- input 100 values, latent 2048, output 1M points (generative)
- input video+audio, latent a few hundred, output video+audio+label

And most importantly: **you can buy more thinking without touching input or output cost.** Raise `L` from 6 to 26 and nothing on the `N` or `O` axis changes.

### 10.8 Perceiver vs Perceiver IO, side by side

|            | Perceiver                                        | Perceiver IO                                |
| ---------- | ------------------------------------------------ | ------------------------------------------- |
| encoder    | cross-attn × 8 (iterated)                        | cross-attn × 1 (typically)                  |
| processor  | 6 self-attn per round (48 total)                 | deep self-attn stack, once                  |
| decoder    | mean pool + linear                               | **cross-attn with a query array**           |
| output     | one vector                                       | `O × E'`, arbitrary                         |
| cost       | `O(T·M·N + T·L·M²)`                              | `O(M·N + L·M² + O·M)`                       |
| philosophy | read repeatedly; cram everything into the latent | read once, think deeply, then ask precisely |

> **Perceiver fights the bottleneck with iteration. Perceiver IO fights it with better queries.** Both address the same problem, and IO demonstrates the second route is cheaper and works better.

### 10.9 One-paragraph summary

Perceiver IO splits a network into three mutually ignorant parts — **a writer (encoder), a thinker (processor), and a reader (decoder)** — mediated by a fixed-size scratchpad `Z`, with a single communication primitive (QKV attention). Input size, compute, and output size become three independently adjustable knobs, which no prior architecture allowed. And "the task" is reduced to "how do I compose the query". In other architectures changing the task means changing the network; in Perceiver IO changing the task means changing data.

---

## 11. Why one read is mostly enough

If the encoder collapses `N = 50,176` down to `M = 512` in a single pass, is that enough? The honest answer: **not in theory — but the question is aimed at the wrong target.**

### 11.1 What a single read cannot do (the convex-hull argument)

```
Z_out = A · V           A ∈ ℝ^(M×N), rows sum to 1, all entries > 0
                        V = X W_V ∈ ℝ^(N×d)

Z_out[m] = Σₙ A[m,n] · V[n]     with Σₙ A[m,n] = 1, A[m,n] ≥ 0
         ∈ conv{ V[1], …, V[N] }
```

> **One cross-attention can only produce `M` points inside the convex hull of the value vectors. It cannot leave that hull, no matter how well trained.**

What iteration buys:

```
Z₁  = Z₀ + convex-combination₁       inside a hull (translated by Z₀)
Z₁' = Z₁ + MLP(LN(Z₁))               nonlinear — now outside the hull
Z₂  = Z₁' + convex-combination₂      another hull added, queried from a NEW position
…
Z_T = Z₀ + Σₜ (convex-combination)ₜ + Σₜ (MLP terms)ₜ
```

The reachable set grows from _a hull_ to _a Minkowski sum of `T` hulls_, each selected by a query conditioned on everything read so far. See [[5-interp-hull-and-superposition]] for the hull framing in general.

**In plain terms, iteration buys exactly two things:**

1. **Conditional querying.** Round 1 asks blind; round 5 asks knowing. **No amount of extra `M` can buy this.**
2. **Separating "wide" from "sharp".** Softmax is a low-pass filter: spread attention and you average away detail; peak it and you get detail but no context. With one read the `M` latents must split these duties. With `T` reads, early rounds go wide and later rounds go sharp.

So the intuition that one read is insufficient is **correct**. The question is whether it is **worth the price**.

### 11.2 The price: a cross-attend is absurdly expensive

FLOPs for one cross-attention (`D = d = 1024`, `M = 512`, `N = 50,176`, `C = 261`):

```
K, V projections from X  :  2 · N · C · d   ≈ 2.7e10     33%
Q Kᵀ                     :      M · N · d   ≈ 2.6e10     33%
A V                      :      M · N · d   ≈ 2.6e10     33%
Q proj, W_O, MLP         :   ~ M · D²       ≈ 2.2e9       3%
                                              ────────
                                              ≈ 8.0e10
```

One self-attention block:

```
Q,K,V,W_O  :  4 · M · D²   ≈ 2.1e9
Q Kᵀ, A V  :  2 · M² · d   ≈ 5.4e8
MLP        :  2 · M · D²   ≈ 1.1e9
                             ≈ 3.7e9
```

> **1 cross-attend ≈ 22 self-attention blocks.**

Sanity check against the paper: `8 × 8e10 + 48 × 3.7e9 ≈ 8.2e11` versus the reported 707.2B = 7.07e11. The estimate runs slightly high but the structure is right.

**The number that should change your view of the architecture:**

```
~80% of the model's total compute goes into READING
~20% goes into THINKING
```

The model spends four fifths of its budget reopening the book.

Removing 7 of the 8 reads frees `7 × 8e10 = 5.6e11` FLOPs, which buys `5.6e11 / 3.7e9 ≈ 150` self-attention blocks.

> **General principle.** Reading costs scale with `N`; thinking costs scale with `M`. When `N/M ≈ 100`, one more unit of reading costs 20–100x one more unit of thinking. Re-reading must produce an enormous return to break even, and empirically it does not.

### 11.3 Buying the same thing more cheaply: raise `M`

If the complaint is "512 slots is not enough", the direct fix is more slots, not more passes:

```
option A:  M = 512,  cross × 8   →   8 × 8.0e10 = 6.4e11
option B:  M = 4096, cross × 1   →   1 × 3.4e11 = 3.4e11      ~half the cost
```

(B is cheaper partly because the `K,V` projections from `X` happen once instead of 8 times.)

**But the two are not equivalent, and this must be stated honestly:**

```
more M      →  more questions, all fixed in advance by Z₀    (unconditional)
more rounds →  fewer questions, but adaptable to what was read (conditional)
```

Conditionality is more powerful per parameter — adaptive search beats blanket coverage. But conditionality carries every cost discussed in §8: deep Jacobian chains, shared-weight compromise, the separate `C₁`, training instability.

```
more M      :  fully parallel, shallow gradients, easy to train
more rounds :  sequential, deep gradients, hard to train
```

DeepMind's empirical verdict: for most tasks, the extra power of conditional querying does not justify the whole bill.

### 11.4 The real reason: the decoder redefines what "reading enough" means

Everything above is budgeting. The structural reason is deeper.

**Ask what `Z` must actually contain.**

In the original Perceiver, `Z` is the **only** path from input to output. Therefore `Z` must hold everything the output will need. The bottleneck is a genuine bottleneck, and re-reading is the only remedy.

Perceiver IO adds a second path:

```
Perceiver:      X ──read──▶ Z ──pool──▶ Y
                            └─ single path; everything must fit through

Perceiver IO:   X ──read──▶ Z ──┐
                │               ├──▶ Y
                └───query───────┘
                  a shortcut: detail reaches the destination directly
```

With `Q_out[o] = [ γ(x_o, y_o) ; X[o] ]`, the detail of pixel `o` travels to the output **without passing through `Z` at all**.

**So the job description of `Z` changes completely:**

```
Perceiver    :  Z must be "a compressed copy of the image"     →  25:1 is a real bottleneck
Perceiver IO :  Z must only be "high-level understanding"      →  25:1 may be generous
                (what is where, what is happening)
```

High-level understanding is genuinely small. For optical flow it is "which groups of things are moving which way" — perhaps a few dozen facts. "Which group does _this_ pixel belong to" is recoverable by the decoder from `X[o]` carried in the query.

> **The direct answer to the question.**
>
> One read is not enough **if `Z` has to carry everything.** Perceiver IO arranges for `Z` not to carry everything. **So one read is enough.**
>
> IO does not fix the bottleneck. **It makes the bottleneck irrelevant.**

### 11.5 When one read genuinely is not enough

For balance, the cases where iteration remains necessary:

| situation                                                                 | why one read fails                                                                              |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **the decoder has no access to `X`**                                      | the shortcut is gone; `Z` must carry everything again                                           |
| **search tasks** — _"find the object similar to the one in the top-left"_ | you must read the top-left before you know what to search for; logically impossible in one pass |
| **multi-hop reasoning** — _"the capital of the country where X was born"_ | step 2's query depends on step 1's answer                                                       |
| **extreme `N/M`** — e.g. `N = 1M`, `M = 256` (4000:1)                     | one pass captures only the coarsest sketch                                                      |

The first three share a structure: **the next query must depend on the previous query's result.** That is the capability iteration monopolises, and the capability Perceiver IO gives up.

> This is precisely why Perceiver IO, elegant as it is, did not become the architecture of reasoning systems: **reasoning is multi-hop by nature.**

---

## 12. Making reads cheaper

If reading is where the money goes, the design question becomes: _can we keep conditional querying while paying far less for it?_ Yes — and the moves below are ordered by effort-to-payoff.

### 12.1 Free money first: cache `K` and `V`

Rounds 2–8 share weights, so `W_K` and `W_V` are literally the same tensors. And `X` never changes. Therefore:

```
K = LN(X) W_K
V = LN(X) W_V
```

are **bit-identical across all seven rounds** — yet Perceiver recomputes them each time. (This is a known observation: the Perceiver family re-projects `K,V` from the input at every cross-attention call; the projection matrices are weight-tied but the resulting tensors are not cached.)

```python
K = X @ W_K           # once
V = X @ W_V           # once
for t in range(2, 9):
    Z = cross_attn_with_cached_kv(Z, K, V)
```

Saving: `2.68e10 × 7 ≈ 1.9e11` FLOPs = **27% of the entire model**, for zero loss in quality. Do this before anything else.

After caching, the per-round cost is `≈ 5.4e10`, all of it in `2·M·N·d`.

### 12.2 The four axes you can compress

Remaining cost is `2 · M · N · d`. All three factors are adjustable.

| axis                    | method                               | new cost  | reduction | differentiable?        |
| ----------------------- | ------------------------------------ | --------- | --------- | ---------------------- |
| **lower `d`**           | narrow heads on re-reads (`d = 128`) | 6.6e9     | 8x        | yes                    |
| **lower `M`**           | only 64 latents may re-read          | 6.6e9     | 8x        | yes                    |
| **lower `N`, random**   | subsample `X` to 6,272 points        | 6.6e9     | 8x        | yes (with variance)    |
| **lower `N`, selected** | top-k retrieval                      | **1.1e9** | **50x**   | **no** — needs a trick |

Against the natural unit (`1 self-attention block = 3.7e9`):

```
full cross-attend        =  5.4e10  =  14.6 self-blocks     brutal
compressed (d/M/N)       =   6.6e9  =   1.8 self-blocks     tolerable
sparse top-k             =   1.1e9  =   0.3 self-blocks     cheaper than a self-block
```

The last row is the significant one: **with sparse re-reads, seven extra reads cost about as much as two self-attention blocks.** Conditional querying becomes nearly free.

### 12.3 Two-stage coarse-to-fine selection

The obstacle with top-k is circular: you need `QKᵀ` to know which elements are worth reading, and `QKᵀ` is the expensive part. Break the circle by **scoring cheaply, then reading expensively only where it matters.**

```
Stage 1 — coarse scan
    partition X into blocks of B = 64  →  N/B = 784 blocks
    k̄_b = mean(K[block b]) ∈ ℝ^d        (computed once, cached)
    score with a narrow width d_s = 64:
        S = Q[:, :64] · K̄[:, :64]ᵀ  ∈ ℝ^(M × 784)
    cost:  M · (N/B) · d_s = 512 × 784 × 64  ≈ 2.6e7      negligible

Stage 2 — sharp read
    each latent selects top-k blocks (k = 16) → 1,024 elements out of 50,176 (2%)
    full attention over the selected set:
        2 · M · 1024 · d ≈ 1.07e9

total ≈ 1.1e9   →  ~50x cheaper than a full cross-attend
```

> **Why the cheap stage suffices.** It does not need to find the exact answer. It only needs to **discard the irrelevant 98% correctly**, which is a far easier problem. A 64-dimensional score can separate "relevant" from "irrelevant" even if it cannot separate "very relevant" from "moderately relevant" — and stage 2 makes that finer distinction among survivors.

This is the same principle as modern sparse attention (block scoring, then selection) and the same principle as the **fovea**: a low-resolution wide field that says _"look there"_, followed by a high-resolution narrow field that actually looks.

### 12.4 The lazy version that already exists

If you would rather not write sparse kernels: Perceiver-VL applies dropout to cross-attention layers with probability 0.5 during vision-and-language pretraining, and explicitly **does not** apply it to the first cross-attention, so that input signal always reaches the latent array.

```
round 1   :  always read
rounds 2-8:  coin flip, p = 0.5    →  read compute halved
```

Note this is the same design instinct: **the first read is protected, the rest are made cheap or skipped.** It also acts as a regulariser, forcing `Z` after round 1 to be useful on its own.

### 12.5 Pitfalls

**(1) Top-k is not differentiable — gradients never reach unselected elements.**

```
∂L/∂X[n] = 0  for every n outside the top-k
```

This creates a rich-get-richer trap: an element never selected never gets the chance to become worth selecting. Mitigations used in practice:

- mix in randomness (top-k plus `m` random blocks) so every region occasionally gets gradient
- Gumbel-softmax or straight-through estimators
- give stage 1 its **own auxiliary loss**: train the block scores to predict the dense attention mass that block would have received — i.e. distil the selector from a dense teacher

**(2) Making round 1 different costs you the loop.**

```
round 1   :  operator A (dense,  d = 1024)
rounds 2-8:  operator B (sparse, d = 128)
```

The system is no longer `Z_{t+1} = f(Z_t)` but `Z₁ = g(Z₀)` then `Z_{t+1} = f(Z_t)`. There is no single fixed point to analyse. The consolation: Perceiver already separates `C₁` anyway. You are turning an accident into a design decision — and it has a name in the literature: **prelude + recurrent core**.

**(3) The subtle one: do not compress the axis that carries the value.**

Re-reading is valuable _because it is conditional_ — because the query is sharp enough to ask a precise question. Crush `d` and queries stop discriminating, so re-reads stop being conditional and become blurry averaging.

```
read 2% of the image at full sharpness    ✓  still genuine conditional querying
read 100% of the image at 12% sharpness   ✗  a worse repeat of what round 1 already did
```

Both cost the same. The first is far better.

> **Rule: compress `N` as hard as you can; protect `d`.**

### 12.6 The reordering worth considering

The natural instinct is **expensive first, then cheap**. But consider:

```
your ordering (Perceiver-like)  :  read wide and detailed  →  then top up
fovea ordering (biological)     :  glance wide and coarse  →  then peer narrowly
```

```
wide + coarse    :  all of N, small d or downsampled   →  cheap
narrow + sharp   :  tiny N, full d                     →  cheap
wide + sharp     :  all of N × full d                  →  the ONLY expensive quadrant
```

> **The deepest point in this section: the expensive read is the one that is "wide AND detailed" simultaneously — and it is not obvious anyone ever needs that.**

Rough totals:

```
Perceiver original      :  8 × 8.0e10           = 6.4e11
Perceiver IO            :  1 × 8.0e10           = 8.0e10
1 full + 7 sparse       :  8.0e10 + 7 × 1.1e9   = 8.8e10     (+10% over IO)
fovea-style             :  1.3e10 + 7 × 1.1e9   = 2.1e10     (4x cheaper than IO)
```

The third row recovers the full conditional-query loop for a 10% premium over Perceiver IO.

---

## 13. CNN and ViT front-ends

If the expensive thing is `N`, the obvious move is to shrink `N` with something cheap before any attention happens. This is what every deployed system does.

### 13.1 The numbers

```
ViT patch embedding, 16×16 patches:
    N: 50,176 → 196                      (256x reduction)
    FLOPs: 196 × 768 × 768 ≈ 1.2e8
```

Compared with one cross-attend:

```
Perceiver cross-attend (full)  :  8.0e10
ViT patch embedding            :  1.2e8      670x cheaper
ResNet-50, entire network      :  4.1e9      20x cheaper
```

> **An entire 50-layer ResNet-50 costs 20x less than a single Perceiver cross-attention.**

And the comparison that matters most:

```
Perceiver (ImageNet, 8 rounds)  :  707 GFLOPs   →  78.0% top-1
ViT-B/16, all 12 layers         :  ~17.6 GFLOPs →  ~78% top-1
```

**Perceiver pays ~40x more for the same accuracy — and all of ViT-B costs less than one Perceiver read.** The paper does not dispute this; Perceiver never beat a specialist model in any domain. Its selling point is generality, not efficiency.

### 13.2 Why convolution is so cheap

```
conv 3×3      :  O(N · k² · C_in · C_out)   =  N × 9 × C²
cross-attn    :  O(M · N · d)               =  N × M × d
```

Multiplier per input element:

```
conv:        9 × C   ≈ 9 × 256    =     2,304
cross-attn:  M × d   ≈ 512 × 1024 =   524,288      →  227x
```

The cause: convolution lets each element talk to **9 neighbours**; cross-attention lets each element be read by **all `M` latents**.

> **Principle. Aggregating locally is nearly free. Content-based routing over long distances is expensive. Reducing `N` is precisely the act of converting "long distance" into "local" before you have to pay for routing.**

### 13.3 The trap

**If you patchify down to `N = 196`, why are you using a Perceiver at all?**

```
N = 196  →  N² = 38,416
full self-attention: 2 × 38,416 × 768 ≈ 5.9e7      →  a rounding error
```

The `O(N²)` problem that justified Perceiver's existence is gone. What remains is ViT.

```
front-end reduces N sufficiently   →  you do not need a Perceiver
front-end cannot reduce N enough   →  you still do
```

Know which regime you are in before designing for it.

### 13.4 When you still need it

**Case A — `N` is still large after patchification.**

```
32-frame video, 16×16 patches   :  32 × 196 = 6,272     N² = 3.9e7   getting costly
300-frame video                 :  58,800               N² = 3.5e9   dead
100 images in one context       :  19,600               dead
30s audio + video               :  hundreds of thousands  very dead
```

Patchifying gains a factor of 256. If the data is 1000x larger, that is not enough.

**Case B — `N` varies but the downstream consumer needs a fixed size.** This is the reason real systems use it.

```
1 image    →  196 tokens
8 images   →  1,568 tokens
a video    →  unbounded
```

An LLM downstream cannot absorb a variable, unbounded number of tokens. **Perceiver solves this with no competition: any `N`, always `M` out.**

**Case C — multimodality.** Convolution cannot mix pixels with audio samples with text tokens. Cross-attention does not care where a key came from; concatenate and fire.

### 13.5 Real systems that do exactly this

| system       | pipeline                                                                                                                |
| ------------ | ----------------------------------------------------------------------------------------------------------------------- |
| **Flamingo** | NFNet (CNN) encodes video/images → **Perceiver Resampler** → 64 fixed latent tokens → cross-attention into a frozen LLM |
| **BLIP-2**   | ViT-g → **Q-Former** with 32 learned queries → 32 tokens → LLM                                                          |
| **DETR**     | CNN backbone → transformer → 100 **object queries** cross-attend → 100 boxes                                            |

All three are "learned-query cross-attention", and **none of them reads raw pixels**. Every one has a CNN or ViT in front.

Even the Perceiver IO paper concedes the point: it states the architecture can be used standalone, in conjunction with pre/post-processing steps, or as part of larger pipelines — and its best ImageNet result uses convolutional preprocessing.

### 13.6 But conv and cross-attention do not compress the same way

This is the caveat that keeps the two from being interchangeable.

```
conv/pool compresses  :  unconditionally, uniformly, with a fixed rule
                         every region is reduced by the same factor
                         it does not know the task, nor what is important

cross-attn compresses :  conditionally, adaptively, based on content
                         A[m,n] depends on what is there, so budget can be
                         concentrated where it matters
```

The example that separates them: a 4K image with small text in one corner and sky elsewhere.

```
conv stride 16 :  sky reduced 256x        ✓ sensible
                  text reduced 256x       ✗ now unreadable
                  → information lost precisely where it mattered, unrecoverably

cross-attn     :  latents can concentrate on that corner if they learned it matters
                  → non-uniform compression preserves what is needed
```

```
conv       =  lossy compression with a fixed prior (locality + fixed scale)
cross-attn =  lossy compression with a learned, content-adaptive prior
```

**Convolution is cheap because it is dumber, not because it is smarter** — and it is dumb in a way that happens to be correct for natural images, which really do have locality and scale hierarchy. It is wrong for data without those priors.

### 13.7 The cost hierarchy (worth memorising)

| operator                   | cost per element | can reduce `N`? | content-adaptive? | works on         |
| -------------------------- | ---------------- | --------------- | ----------------- | ---------------- |
| strided conv / patchify    | `k²·C` ≈ 2e3     | heavily (256x)  | no                | grids only       |
| pooling                    | ~0               | heavily         | no                | grids only       |
| local / windowed attention | `w·d` ≈ 5e4      | moderately      | within the window | grids, sequences |
| cross-attn (learned query) | `M·d` ≈ 5e5      | any ratio       | globally          | **anything**     |
| full self-attention        | `N·d` ≈ 5e7      | not at all      | globally          | anything         |

> **Rule: work down the table. Use the cheapest operator that still does the job, and move down only when forced. Do not pay attention prices for work convolution can do — and do not expect convolution to do work that requires content-based selection.**

The architecture this implies:

```
X (50,176) ──patchify/CNN──▶ 6,272 ──cross-attn──▶ Z (512) ──self-attn ×L──▶ Z
              very cheap              costly but worth it     cheap; think a lot
              non-selective           selective                    │
                                                                   └──query──▶ output
```

**Two reductions of `N`, by two different mechanisms:**

- first, a **non-selective** reduction with a cheap operator — removing spatial redundancy, which requires no intelligence to detect
- second, a **selective** reduction with an expensive operator — deciding what matters, which does

And performing the first makes the second **8x cheaper**. That is the entire gain.

### 13.8 What you give up by adding a front-end

| lost                                                                          | how much it matters                                 |
| ----------------------------------------------------------------------------- | --------------------------------------------------- |
| **non-grid data** (point clouds, sets, graphs, irregular sampling)            | a lot, if you care about multiple modalities        |
| **scale-agnosticism** — conv has already decided what counts as "nearby"      | a lot, for data with unusual scale structure        |
| **raw-level modality mixing** — each modality now needs its own encoder first | moderate; Flamingo does fine with separate encoders |
| **philosophical purity**                                                      | none                                                |

Decision rule:

```
goal = "work well on one data type"      →  add a CNN/ViT, no deliberation needed
goal = "one architecture for everything" →  a CNN betrays the thesis
goal = "study the loop mechanism"        →  add one; experiments get 40x cheaper
```

---

## 14. Two-tier memory: a design sketch

A natural synthesis of §12 and §13: **do not use the CNN to replace reading — use it to build a second, cheap memory that the loop can consult repeatedly.**

```
X   = raw input, 50,176 × 261        expensive, high fidelity
X̃   = CNN/ViT features, 196 × 768    cheap, coarse, semantically dense

round 1     :  full read from X        →  build Z with maximum degrees of freedom
rounds 2–8  :  poor read from X̃        →  conditional re-querying for reasoning
decode      :  output query carries X  →  fine detail delivered directly to the output
```

### 14.1 Cost

Per poor read (`N' = 196`, `C' = 768`, `d = 1024`):

```
Q projection    :  M · D · d       = 5.4e8      22%
K,V from X̃      :  2 · N' · C' · d = 3.1e8      (cacheable)
Q K̃ᵀ            :  M · N' · d      = 1.0e8       4%
A Ṽ             :  M · N' · d      = 1.0e8       4%
W_O             :  M · D · d       = 5.4e8      22%
MLP             :  2 · M · D²      = 1.1e9      45%
                                     ────────
                                     ≈ 2.4e9
```

**Notice where the cost has gone:**

```
terms containing N'  :   8%
terms not containing N' : 92%     ← Q, W_O, MLP all operate on M alone
```

> Once `N` is small, **shrinking `N` further buys nothing.** The cost has migrated from "reading" to "preparing to read". To go cheaper you must now shrink `d` or delete the MLP.

With `d = 256` and no MLP (the latent transformer already supplies six):

```
Q:   512 × 1024 × 256      = 1.3e8
attn: 2 × 512 × 196 × 256  = 5.1e7
W_O: 1.3e8
                             ≈ 3.2e8 per round
```

System totals:

```
Perceiver original                     :  6.4e11
Perceiver IO                           :  8.0e10
this design (1 full + patchify + 7 poor):  8.0e10 + 1.2e8 + 2.2e9 ≈ 8.2e10
```

**Full conditional querying restored for a ~3% premium over Perceiver IO, and 7.8x cheaper than the original Perceiver.**

### 14.2 Why coarse re-reads are sufficient

Split what re-reading is for:

```
Job A — search / localisation
    "the thing I just found — is there more of it, and where?"
    "near that fur, are there ears?"
    needs: spatial resolution + mid-level semantics
    does NOT need: high-frequency pixel detail

Job B — fine retrieval
    "at exactly (100,42), what is the colour, how sharp is the edge?"
    needs: pixel-level high frequency
```

What survives patchification:

```
X̃ retains  :  14×14 spatial resolution      ✓ enough for "roughly where"
              dense per-patch semantics      ✓ enough for "what is it"
              pixel-level high frequency     ✗ gone
```

**`X̃` preserves Job A entirely and discards Job B.** And Job B is already handled elsewhere — by the output query carrying `X[o]` (§10.5). It never needed to go through the loop.

The resulting three tiers of access:

```
X  (50k, full detail)  ──full read × 1───▶  Z    build initial understanding   expensive, once
X̃  (196, coarse)       ──poor read × 7───▶  Z    search / reasoning            cheap, often
X  (50k, full detail)  ──output query────▶  Y    final detail                  cheap, once
```

> **The insight worth keeping: "detailed" and "repeated" never actually need to co-occur.**
>
> - detailed reads are needed **once** (at the start, and at the end)
> - repeated reads only need to be **coarse**, because they are searching, not retrieving
>
> **Perceiver is expensive because it assumes the two must come together. They do not.**

### 14.3 Does the first full read still earn its place?

If `X̃` supports search and the decoder supplies detail, what does the expensive first read buy?

**Not capacity.** A common misconception to clear up:

```
Z ∈ ℝ^(512×1024) = 524,288 numbers      ← fixed, regardless of what it reads from
```

Reading from `X` (13.1M) or from `X̃` (150k) yields the same-sized `Z`.

What the full read buys is **a wider set of things to choose from**:

```
reading from X̃  :  Z is assembled from 196 vectors the CNN already selected
                   → bounded by the CNN's prior (locality, fixed scale)

reading from X   :  Z is assembled from 50,176 raw vectors
                   → can recover what the CNN discarded, if it matters
```

> **The first full read is insurance against the CNN's prior, not a wider data pipe.**

Consequence: for natural images where the CNN's prior is correct, **the full read may be removable entirely** (`X̃ → cross × 8`, total ≈ 5e9). For data containing things CNNs tend to destroy — small text, single critical points, non-local structure — the full read is the only thing preventing loss at layer one. This is empirical; measure it, do not assume.

### 14.4 The failure mode to guard against

Calling the coarse tier a "fact check" is the right ambition, but there is a trap.

```
gradient flows back from the loss  →  through the poor read  →  into the CNN  →  CNN adapts

The CNN learns to emit X̃ that MINIMISES LOSS,
which is not the same as X̃ that is FAITHFUL TO THE IMAGE.
```

If `Z` holds a wrong belief and the CNN is rewarded for producing features consistent with that belief, you have built an echo chamber, not a fact checker.

Three fixes, hardest to softest:

```
1. Freeze the CNN         :  use pretrained (CLIP / DINO) and freeze
                             → maximally faithful, but cannot adapt to the task

2. Reconstruction loss    :  require X̃ to reconstruct X
                             L = L_task + λ · ‖decode(X̃) − X‖²
                             → forces X̃ to retain real information, not just convenient
                               information; also forces coverage across the whole image
                               rather than clustering where the loop likes to look

3. Stop-gradient          :  X̃ = stopgrad(CNN(X)) at the poor-read call sites
                             → the CNN still learns, but not from the loop's pressure
```

Option 2 is the best trade: it keeps both faithfulness and adaptability.

### 14.5 Use a pyramid, not a single level

A CNN produces multi-scale feature maps for free. Do not discard them.

```
X    :  50,176 elements  (stride 1)     full read, once
C₃   :   3,136 elements  (stride 4)     fine poor-read
C₄   :     784 elements  (stride 8)     mid  poor-read
C₅   :     196 elements  (stride 16)    coarse poor-read
```

Either schedule the rounds by hand:

```
rounds 2-3  :  read C₅ (196)     "what is here, roughly, and in which zone"
rounds 4-6  :  read C₄ (784)     "what is the detail in that zone"
rounds 7-8  :  read C₃ (3,136)   "confirm the boundaries"
```

Cost `2·M·(196 + 784 + 3136)·d`, still in the 1e9 range — **an order of magnitude cheaper than one full read**.

Or concatenate all levels into a single key/value array (`N' = 4,116`) and let attention choose the scale itself. Then **latents learn which questions belong at which scale**, which Perceiver cannot do at all since it has exactly one scale.

> **Unintended bonus.** Perceiver is criticised for lacking a multi-scale inductive bias (unlike CNNs or Swin). This pyramid restores multi-scale **without hard-coding it** — the model selects, rather than the architecture dictating one scale per layer.

### 14.6 Engineering details that must be right

**(1) The two key spaces are not the same — use separate projections.**

```
X  :  C  = 261   (RGB + Fourier features)
X̃  :  C' = 768   (CNN semantic features)
```

The geometries are unrelated. Do **not** share `W_K, W_V` between full and poor reads, and consider separate `W_Q` as well, since `Z` must form questions that "key into" two different spaces. An extra `W_Q` costs `D·d ≈ 1e6` parameters — noise against 45M.

**(2) Cache `K̃, Ṽ`.** `X̃` is constant and the poor-read `W_K, W_V` are shared, so compute them once.

**(3) The new weight-sharing structure:**

```
full read (round 1)   :  its own weight set
poor read (rounds 2-8):  one shared set
self-attn S¹..S⁶      :  shared across rounds as before
```

Note that separating round 1 is **no longer a hack to fix instability — it is now an architectural fact**, because round 1 genuinely reads a different source. The bug has been promoted to a feature.

**(4) Delete the MLP from the poor read.** It is 45% of that block's cost, and six MLPs follow immediately in the latent transformer.

### 14.7 The 30-minute experiment that validates the idea

The whole design rests on one hypothesis: **rounds 2–8 do not need pixel-level detail.** This is testable on an already-trained Perceiver, with no retraining.

```python
# inference only
X_full = build_input(image)                      # 50,176 × 261
X_down = build_input(avg_pool(image, k))         # downsampled by factor k

Z = Z0
for t in range(8):
    src = X_full if t == 0 else X_down           # full first, coarse thereafter
    Z = cross[t](Z, src)
    Z = latent_transformer(Z)
```

Sweep `k = 1, 2, 4, 8, 16` and record top-1 accuracy.

```
accuracy roughly flat up to k = 8 or 16
    →  later rounds were never using high-frequency detail
    →  the design is sound and the savings are real                   ✓

accuracy collapses at k = 2
    →  later rounds genuinely need detail
    →  switch strategy: sparse-but-sharp instead of dense-but-coarse  ✗
```

**Companion measurement — attention entropy per round:**

```
H_t = − Σₙ A_t[m,n] log A_t[m,n]        averaged over m

H₁ high (diffuse) and H₂…H₈ low (peaked)
    →  later rounds already attend to only a few locations
    →  sparse re-reads will cost almost nothing in quality           ✓

H roughly equal across all rounds
    →  later rounds still read broadly; sparsification will hurt      ✗
```

Run both before writing any training code. Together they determine the design.

### 14.8 Prior art to compare against

| work                            | what it does                                                         | overlap                                                                |
| ------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **Compressive Transformer**     | fine-grained memory for recent items, compressed memory for old ones | two-tier memory at different fidelities — the closest structural match |
| **FPN / multi-scale detection** | detection heads read from a feature pyramid                          | §14.5                                                                  |
| **Native Sparse Attention**     | three parallel branches: compressed, selected, local                 | the poor read is the compressed branch                                 |
| **RETRO**                       | a main model cross-attending into an external database               | "go back and check the facts", conceptually                            |

None does exactly this: _read raw and detailed once to construct state, then loop reasoning over a compressed memory._ The nearest is Compressive Transformer, but its split is by **time** (old vs recent) rather than by **role** (construct vs verify).

---

## 15. Landscape: closed and open problems

Status as of mid-2026. Papers dated 2026 below were located by search rather than recalled; verify against the originals.

### 15.1 Closed — do not spend time here

| question                                               | status                                                                                                                        |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| can a cross-attention bottleneck actually beat `O(N²)` | closed since Set Transformer (2019)                                                                                           |
| can we decode arbitrary output shapes                  | closed — output queries settle it                                                                                             |
| does weight sharing help here                          | closed — clear regularisation; the first round must be excluded                                                               |
| how much of an image fits in a few latents             | closed — TiTok reconstructs from 32 tokens; FlexTok reports FID < 2 across 8–128 tokens                                       |
| does Perceiver beat specialist models                  | closed — it does not, and nobody is still trying. It survives as a **module** (Perceiver Resampler, Q-Former), not a backbone |
| is a CNN/ViT front-end needed                          | closed — every deployed system has one                                                                                        |
| is 1D latent tokenization better than a 2D grid        | effectively closed — 1D removes grid redundancy and reaches high generation quality                                           |

> **Summary: "Perceiver as a backbone" is dead. "Learned latent bottleneck as a component" won completely.**

### 15.2 Half-open — answers exist but they are unsatisfying

**(a) Dynamic `M`.**

Existing work:

- **Dynamic Latent Perceivers** — because the input is mapped into a latent space of size `n`, complexity in input length is only linear; they exploit this to vary the latent count at inference after a single training run.
- **FlexTok** — projects a 256×256 image into anywhere from 1 to 256 ordered 1D tokens, compressing hierarchically and semantically; a rectified-flow decoder plus **nested dropout** makes reconstructions plausible at any chosen length.
- **ElasticTok** — conditions on prior frames to encode a frame into a variable number of tokens, using a mask that randomly drops a suffix of each frame's token encoding. Reports 3.5x efficiency on images and 5x on video at matched reconstruction thresholds.

The gap, stated explicitly by AdaTok: these methods show variable-length visual representation is possible, but they leave a gap between **flexible reconstruction** and **autonomous budgeting**.

```
exists  :  the model can reconstruct at any length   →  but a HUMAN picks the length
missing :  the model looks at an image and DECIDES   →  "this one needs 12 tokens,
                                                         that one needs 180"
```

Why it is hard: `M` is an integer (not differentiable), batching is awkward when every sample wants a different length, and there is no obvious supervision signal for "enough".

The motivation is real: a fixed-length tokenizer pays a near worst-case price on every input, and in VLMs visual tokens compete with text for context.

**(b) Latent as persistent state across time.**

A great deal of engineering, very little principle. Streaming video models must build a bounded working context from an unbounded stream; existing approaches use memory banks or event memories, KV-cache plus retrieval, visual token pruning or compression, and recurrent or latent states.

The dominant training-free line compresses the KV cache using cosine-similarity metrics to remove redundancy, then uses a **generic proxy query** (often a generation template) to compute attention scores as an importance signal.

> **"Generic proxy query" is an admission of defeat.** The system does not know what it will be asked, so it guesses at importance — **while Perceiver already has learned queries by construction.** This is an unusually clear gap.

**(c) Fixed points and convergence.** DEQ has existed since 2019 and is almost never combined with Perceiver, which still unrolls a fixed 8 rounds. Everything in §8 — Jacobian products, spectral radius, gradient summation, the `C₁` anomaly — **has not been analysed for Perceiver specifically in the literature.**

### 15.3 Open

**1. Who decides `M`, and how.**

```
sub-problems:
├─ what should the predictor look at?  raw image / coarse features / result of round 1
├─ how to make it differentiable?      Gumbel / ACT-style halting / RL
├─ how to batch when M varies?         pad + mask, or bucketing
└─ what supervises "enough"?           reconstruction / task loss / uncertainty
```

An untried combination that looks promising: apply **ACT (Adaptive Computation Time)** to the number of **latents** rather than the number of layers — give latent `m` a halting probability `h_m` and stop once the cumulative mass reaches 1. Soft, differentiable, and compatible with FlexTok's nested-dropout ordering.

**2. Write / evict / consolidate rules for the latent — the widest gap.**

Perceiver has only `Z ← Z + cross(Z, X)`. Addition, nothing else.

```
Perceiver has     :  write (additive)
Perceiver lacks   :  evict       (remove what is no longer used)
                     consolidate (merge duplicates)
                     allocate    (reserve slots for new content)
                     protect     (prevent important content being overwritten)
```

LLM-side context compaction has all of these. Translating them to vision is open.

There are system-level attempts — e.g. organising history as a dynamic latent evidence graph: writing projected embeddings into memory, consolidating with priority-aware penalties, retrieving query-conditioned subgraphs, and injecting them back as calibrated latent evidence tokens — but these are agent-level designs, not **learned, end-to-end update rules**.

> **The sharpest formulation: what is the optimal latent update rule given bounded slots and unbounded data?** This is cache eviction policy, in a differentiable setting. Unanswered.

**3. What should each latent mean — the binding problem.**

Nothing in Perceiver forces latents to specialise. They may duplicate each other or smear.

**And here is the technical detail that changes everything with a one-line edit:**

```
Perceiver      :  A = softmax over N (QKᵀ)    each latent spreads its budget over pixels
                  → latents do not compete → all can attach to the same salient region

Slot Attention :  A = softmax over M (QKᵀ)    each PIXEL spreads its budget over slots
                  → slots compete for pixels → object binding emerges
```

**Changing the softmax axis introduces competition, and competition produces object-centric representations.** Perceiver never tries this, and no one has seriously combined the two.

> This is a prerequisite, not a side quest: **if latents are not bound to anything identifiable, you cannot evict or compact them, because you do not know what you would be deleting.**

**4. Multi-scale inside the latent.** Perceiver has one scale, no hierarchy, no coarse-to-fine. FlexTok obtains hierarchy from nested dropout, but along the **ordering** axis, not the **spatial scale** axis. A latent array that is genuinely pyramidal — some latents viewing the whole scene, some a region, some a point, with the allocation learned — is open.

**5. Causal / autoregressive latents for images.** Perceiver AR solved this for language. For vision, a latent updated causally along a scan order, a scale order, or an object order has no good answer.

**6. Generative rather than merely discriminative latents.** Perceiver IO's multimodal autoencoding is only adequate (20.7 PSNR). A latent bottleneck that both _understands_ and _reconstructs_ is still open — and it is the prerequisite for a world model.

### 15.4 On "images have nothing to grab onto"

A common framing: language is symbolic so compaction has natural seams, whereas images do not. **Half true.**

```
language :  units are separated already (words, sentences, paragraphs, turns)
            → compaction has natural cut points

images   :  no intrinsic boundaries; pixels are continuous
            → no natural cut points                          ← the true half
```

But images have properties language does not:

| images / video have                                                                   | language does not                |
| ------------------------------------------------------------------------------------- | -------------------------------- |
| enormous **spatial redundancy** — neighbouring pixels are predictable from each other | every token carries information  |
| **temporal redundancy** — 30 fps is the same picture 30 times                         | none                             |
| **physical continuity** — objects do not vanish and reappear                          | none                             |
| **object boundaries** — real separable units, though they must be discovered          | words are units but not _things_ |

**And the decisive point: LLM compaction works because text has cuttable seams. Still images have none — but video does.**

```
still image :  no natural boundary  →  the unit must be the OBJECT   (spatial axis)
video       :  natural boundaries  →  the unit can be SHOT / SCENE / EVENT  (temporal axis)
```

Existing work already leans this way. Temporal Perceiver divides latent queries into two kinds — **boundary** and **context** — to handle the temporal redundancy of video, which is a goal/fact split implemented on video. And recent streaming work uses **objects as memory anchors**: persistent object histories, transient object-level changes, and a recent visual window, all under a bounded visual-token budget.

> **Restated: images are not structureless. The units simply are not handed to you the way a tokenizer hands you tokens.**
>
> ```
> language :  the tokenizer supplies units for free   →  compaction can start immediately
> images   :  units must be DISCOVERED first          →  an extra step language skips
> ```
>
> **So the real problem is unit discovery, not compaction.** Once units exist, compaction looks the same on both sides.

### 15.5 The dependency order

```
1. who decides the size          →  autonomous budgeting
2. what each latent holds        →  binding
3. when to evict / merge         →  eviction and consolidation
```

**These must be solved in the order 2 → 1 → 3**, because (3) requires (2), and (2) gives (1) its meaning. Jumping to (1) or (3) without (2) leads to "I do not know what to delete" — which is the wall the entire streaming-video literature is currently up against.

---

## 16. Takeaways

**On attention**

1. Attention is differentiable database lookup, equivalently kernel regression with a learned kernel. `1/√d` is an inverse temperature, the same knob as Hopfield's `β`.
2. `O(N²)` is the price of content-based all-pairs routing. Breaking it requires sacrificing something: sparsity, low-rank approximation, or a bottleneck.
3. Attention never required `n_q = n_kv`. Self-attention _chose_ that. Decoupling them is the entire Perceiver idea; cost becomes `O(n_q · n_kv · d)` and the output shape is yours.

**On the architecture**

4. `Z₀` is a learned parameter — a set of `M` questions the model decided are worth asking. Not noise.
5. The bottleneck is `512 × 1024`, a 25:1 ratio — working memory with 512 slots, not a summary vector.
6. Cross-attention is a read (cost scales with `N`), self-attention is thinking (cost scales with `M`). The test is whether `X` appears in the equation.
7. Position must be injected because cross-attention is permutation invariant over the input. Fourier features are concatenated (not added) and give a shift-invariant kernel for free.

**On weight sharing**

8. 56 executed blocks, 8 distinct weight sets. Sharing is across rounds, never across positions.
9. **FLOPs are identical with and without sharing.** It is not a compute optimisation; it changes the model from a pipeline into an iteration.
10. Backward, a shared parameter's gradient is a **sum over uses** — a vote. Updates that help one round and hurt others cancel; what survives is what is true in every context.
11. `C₁` must be separate because its gradient is batch-coherent (`~B`, not `~√B`) and directionally frozen, so it swamps the others.
12. Which round's gradient dominates is set by `ρ(J)`, not by distance from the loss. Residual + pre-LN pin `ρ ≈ 1`, which is what makes sharing viable.
13. Log `‖gₜ‖` **and** `cos(gₜ, gₜ')`. The cosine matters more: negative cosine means one operator is being asked to do two conflicting jobs.
14. Shared parameters have a 2.6–7x higher effective learning rate automatically. Retune.

**On Perceiver IO**

15. Encode, process, and decode are one operation with the query holder swapped. `Z` is a queryable database.
16. Output queries must be identifiable; they may also carry task identity and **raw input features** — the latter being a U-Net skip connection in disguise.
17. Outputs are conditionally independent given `Z`. This buys linear cost, parallelism, and training-time subsampling. It costs autoregressive generation and inter-output coherence.
18. `N ⊥ (M, L) ⊥ O` — three independent knobs, which no prior architecture allowed.

**On economics**

19. One cross-attend ≈ 22 self-attention blocks ≈ 80% of the whole model's compute.
20. Perceiver fights the bottleneck with iteration; IO fights it with better queries. **IO does not widen the bottleneck — it makes it irrelevant.**
21. Iteration's monopoly is **multi-hop**: when query `t+1` must depend on the answer to query `t`. That is what IO gives up, and why it is not a reasoning architecture.
22. Caching `K,V` across shared rounds is 27% of the model, free.
23. **Reading wide AND detailed is the only expensive quadrant.** Wide-and-coarse is cheap. Narrow-and-sharp is cheap. Perceiver stands in the one expensive square.
24. "Detailed" and "repeated" never need to co-occur: detail is needed once at the start and once at the end; repetition only needs to be coarse, because repetition is search.
25. Reduce `N` twice with two different mechanisms — cheap and non-selective first (removing redundancy), expensive and selective second (deciding importance). The first makes the second 8x cheaper.
26. **Compress `N` hard; protect `d`.** Crushing `d` destroys the conditional querying that was the reason to re-read at all.

---

## 17. Reading list

**Core**

```
Jaegle et al. 2021, Perceiver        (2103.03206)   §2 Method in full; Appendix Table 7
Jaegle et al. 2021, Perceiver IO     (2107.14795)   §2.2 decoder; §4.2 optical flow
```

What to actually stare at in the Perceiver paper:

1. the visualisation of `A ∈ ℝ^(M×N)` — evidence for or against `Z₀` learning "questions"
2. Table 7 (weight sharing): 72.9 → 78.0 valid, 87.7 → 79.5 train, **identical FLOPs**
3. the sentence on instability when sharing the first cross-attend
4. `Z₀` initialisation, and the claim that performance is robust to its scale
5. the ablation on number of cross-attends — where do diminishing returns start
6. in IO: input features concatenated into the flow query — the hidden skip connection

**Adjacent, in priority order**

```
Slot Attention                      softmax over M instead of N → binding. Read this first.
Set Transformer (ISAB)              the true ancestor of the latent bottleneck
FlexTok            (2502.13967)     nested dropout, variable length, coarse-to-fine ordering
ElasticTok                          adaptive tokenization for image and video
TiTok / One-D-Piece                 1D tokenizer foundations
AdaTok             (2606.07185)     names the "autonomous budgeting" gap most precisely
Temporal Perceiver (2203.00307)     boundary vs context latent split on video
Perceiver AR                        causal latents
DEQ (Bai et al. 2019)               fixed-point iteration done properly
Perceiver Resampler (Flamingo)      Perceiver as deployed
ObjectStream       (2607.28312)     objects as memory anchors for streaming
Compressive Transformer             two-tier memory at different fidelities
```

---

## 18. Related notes

- [[1-transformer]] — the base architecture
- [[2-hopfield]], [[4-hopfield-internal]] — retrieval, energy, the `β` knob
- [[3-transformer-internal]] — what the weights are doing
- [[5-interp-hull-and-superposition]] — the convex-hull argument in §11.1 lives here
- [[6-fixed-point-and-what-transformer-chasing]] — §8.12 and §9.2 belong to this thread
- [[7-vit-foundation]] — patchification, the front-end of §13
- [[8-position-transformer]] — Fourier features in depth
- [[9-vision-transformer-landscape]] — where Perceiver sits among ViT variants
- [[10-attention-collapse-and-field-equilibrium]] — `N×d` collapse vs Hopfield pooling's `1×d`; the dynamic-latent-space thread continues directly into §15 here
