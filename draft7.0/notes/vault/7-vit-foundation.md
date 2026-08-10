# 7 — Vision Transformers: Foundations

> **What this document covers.** How a Transformer, an architecture built for sequences of word vectors, was made to eat images. Everything from why patches exist at all, through the exact equations of ViT, to why the original model needed 300 million images to beat a ResNet — and what the "hybrid" variant changes.
>
> **Prerequisites.** Linear algebra (matrix multiplication, rank, subspaces, orthogonality). Basic calculus. Familiarity with what "training a neural network with SGD" means. You do _not_ need prior knowledge of CNNs or ViT; both are built up here.
>
> **Primary source.** Dosovitskiy et al., _An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale_, ICLR 2021 (arXiv Oct 2020). Sections 3 and 4.5 in particular.
>
> **Related notes.** `8-position-in-transformers.md` goes deep on positional encoding, which is only sketched here. `9-vision-transformer-landscape.md` covers everything that happened _after_ 2020.

---

## Table of contents

0. [Notation and conventions](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#0-notation-and-conventions)
1. [The starting problem: sequences vs grids](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#1-the-starting-problem-sequences-vs-grids)
2. [The Transformer block, in full](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#2-the-transformer-block-in-full)
3. [Patch embedding: Equation (1)](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#3-patch-embedding-equation-1)
4. [The encoder: Equations (2)–(4)](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#4-the-encoder-equations-24)
5. [Model configurations](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#5-model-configurations)
6. [Inductive bias: the central concept](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#6-inductive-bias-the-central-concept)
7. [The hybrid architecture](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#7-the-hybrid-architecture)
8. [What §4.5 actually measured](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#8-what-45-actually-measured)
9. [Common misconceptions, corrected](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#9-common-misconceptions-corrected)
10. [Summary](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#10-summary)

---

## 0. Notation and conventions

| Symbol         | Meaning                                                     |
| -------------- | ----------------------------------------------------------- |
| $H, W$         | image height and width, in pixels                           |
| $C$            | number of input channels (3 for RGB)                        |
| $P$            | patch size (the `16` in `ViT-B/16`)                         |
| $N$            | number of tokens in the sequence                            |
| $D$            | model width / embedding dimension (768 for ViT-Base)        |
| $L$            | number of Transformer blocks (depth)                        |
| $h$            | number of attention heads                                   |
| $d_k$          | dimension per head, $= D/h$                                 |
| $x$            | the raw input image, $x \in \mathbb{R}^{H\times W\times C}$ |
| $x_p$          | the image reshaped into patches                             |
| $E$            | patch embedding matrix                                      |
| $E_\text{pos}$ | position embedding matrix                                   |
| $z_\ell$       | the hidden state (the "residual stream") after block $\ell$ |

Convention: vectors are **row vectors**, so a linear map is written $xW$, not $Wx$. A matrix $X \in \mathbb{R}^{N\times D}$ holds $N$ tokens stacked as rows. Superscripts index tokens ($x_p^i$ = patch number $i$); subscripts index components within a vector.

---

## 1. The starting problem: sequences vs grids

### 1.1 What a Transformer consumes

A Transformer is a function on a **set of vectors equipped with an ordering**:

$$X \in \mathbb{R}^{N \times D}$$

- $N$ is the sequence length — how many things there are.
- $D$ is the width — how many numbers describe each thing.
- Row $i$, written $x_i \in \mathbb{R}^D$, is the vector representing token $i$.

In language modelling, one token is roughly one word or sub-word, and $D$ is the size of the word embedding.

### 1.2 What an image is

An image is not a sequence. It is a **2-dimensional grid**:

$$x \in \mathbb{R}^{H\times W\times C}$$

Every pixel has a location $(i, j)$, and locations have structure: pixel $(10, 10)$ is adjacent to $(10, 11)$, is far from $(200, 200)$, and the notion of "distance" between two pixels is meaningful and Euclidean.

Converting the grid into a sequence is therefore a design decision, and the component that does it is called the **stem**. Almost every architectural question in this whole field turns out to be a question about the stem.

### 1.3 Why not just use pixels as tokens?

The obvious first idea: one pixel = one token. Then a $224\times224$ image gives

$$N = 224 \times 224 = 50{,}176 \text{ tokens}$$

Self-attention (defined properly in §2) has time and memory complexity

$$O(N^2 D)$$

because it must compute a **compatibility score for every ordered pair of tokens**, producing an $N\times N$ matrix. At $N = 50{,}176$:

$$N^2 = 50{,}176^2 \approx 2.52\times 10^9 \text{ entries}$$

In fp32 that is roughly **10 GB for a single attention matrix**, for one head, in one layer. ViT-Base has 12 layers × 12 heads. This is not a matter of optimising the implementation; it is off by three orders of magnitude.

### 1.4 The patch solution and its exact price

Group pixels into non-overlapping $P\times P$ blocks and call each block one token. At $P=16$:

$$N = \frac{224 \times 224}{16 \times 16} = \frac{50{,}176}{256} = 196 = 14\times14$$

The attention cost ratio relative to pixel-level tokens is

$$\left(\frac{50{,}176}{196}\right)^2 = 256^2 = 65{,}536$$

**Patching makes attention roughly 65,000× cheaper.** That is the entire original motivation. It is a compute concession, not a claim that $16\times16$ blocks are a natural unit of visual meaning. Keep this framing: the patch is a compromise, and most of the ViT literature after 2020 consists of paying that compromise back in different currencies.

> **Aside on modern long-context models.** People reasonably ask: LLMs handle 1M-token contexts now, so why is 50k a problem? Two answers. (a) Historically: in 2020, FlashAttention did not exist, so the $N\times N$ matrix had to be materialised in HBM. (b) Structurally: FlashAttention reduces attention's _memory_ from $O(N^2)$ to $O(N)$ via tiling and recomputation, but the _compute_ remains $O(N^2 D)$ — it computes exact attention, not an approximation. Million-token contexts come from combining FlashAttention with sparse or sliding-window patterns, sequence parallelism (Ring Attention), and a great deal of hardware. And separately: pixels are enormously redundant, so spending pairwise compute on two adjacent near-identical pixels buys almost nothing.

---

## 2. The Transformer block, in full

This section defines every operation used by ViT. If you already know Transformers, skim it for §2.5, which is the property that drives everything later.

### 2.1 Scaled dot-product attention

Start from $X \in \mathbb{R}^{N\times D}$. Produce three projections:

$$Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V$$

- $W_Q, W_K \in \mathbb{R}^{D\times d_k}$ and $W_V \in \mathbb{R}^{D\times d_v}$ are learned parameters.
- $Q \in \mathbb{R}^{N\times d_k}$ — the **query**. Row $i$ encodes "what token $i$ is looking for."
- $K \in \mathbb{R}^{N\times d_k}$ — the **key**. Row $j$ encodes "what token $j$ offers."
- $V \in \mathbb{R}^{N\times d_v}$ — the **value**. Row $j$ encodes "what token $j$ will contribute if selected."

Compute raw compatibility scores for all pairs:

$$S = \frac{QK^\top}{\sqrt{d_k}} \in \mathbb{R}^{N\times N}, \qquad S_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}}$$

**Why divide by $\sqrt{d_k}$.** Suppose the entries of $q_i$ and $k_j$ are i.i.d. with mean 0 and variance 1. Then

$$q_i \cdot k_j = \sum_{m=1}^{d_k} q_{im}k_{jm} ;\Longrightarrow; \mathbb{E}[q_i\cdot k_j] = 0, \quad \mathrm{Var}[q_i \cdot k_j] = d_k$$

so the standard deviation is $\sqrt{d_k}$. Without the correction, logits grow with model width; feeding large-magnitude logits into a softmax saturates it (outputs near 0 or 1), and a saturated softmax has vanishing gradient. Dividing by $\sqrt{d_k}$ restores unit variance regardless of $d_k$.

Normalise each row into a probability distribution:

$$A_{ij} = \frac{\exp(S_{ij})}{\sum_{j'=1}^{N}\exp(S_{ij'})}, \qquad \sum_{j} A_{ij} = 1 ;; \forall i$$

$A \in \mathbb{R}^{N\times N}$ is the **attention matrix**. Row $i$ is how token $i$ distributes its (fixed, unit) budget of attention across all tokens.

> **Remember the constraint $\sum_j A_{ij} = 1$.** There is no way for a token to attend to _nothing_. This becomes the mechanistic explanation for attention sinks and register tokens — see note 9.

Finally:

$$\mathrm{Attn}(X) = AV \in \mathbb{R}^{N\times d_v}, \qquad [\mathrm{Attn}(X)]_i = \sum_j A_{ij}v_j$$

Each output row is a weighted average of value vectors.

### 2.2 Multi-head attention

Instead of one attention computation at full width, run $h$ of them at $d_k = d_v = D/h$:

$$\mathrm{head}_m = \mathrm{Attn}\left(XW_Q^{(m)},; XW_K^{(m)},; XW_V^{(m)}\right)$$

$$\mathrm{MSA}(X) = \left[\mathrm{head}_1 ; \mathrm{head}_2 ; \dots ; \mathrm{head}_h\right] W_O$$

where $[,\cdot,;,\cdot,]$ concatenates along the feature axis (giving $\mathbb{R}^{N\times D}$ again, since $h \cdot D/h = D$) and $W_O \in \mathbb{R}^{D\times D}$ mixes the heads.

**A precision point that is very often stated wrongly.** Multi-head attention does _not_ slice the input vector into chunks and give one chunk to each head. Look at the shape:

$$W_Q^{(m)} \in \mathbb{R}^{D\times d_k}$$

The **input** dimension is $D$. Every head reads the **entire** 768-dimensional token. What differs is the **output**: each head projects into its own $d_k$-dimensional subspace. So:

$$\text{every head reads all of } D ;\longrightarrow; \text{writes into its own subspace } d_k$$

**Why not give each head the full $D$ on output too?** Cost. With $d_k = D/h$, the total parameters across all $h$ heads are $h \times (D \times D/h) = D^2$, identical to a single full-width head, and the compute for $QK^\top$ summed over heads is also identical. So multi-head buys $h$ different "views" for the price of one. If each head had $d_k = D$, cost would be $h\cdot D^2$ — an $h$-fold increase.

### 2.3 LayerNorm

$$\mathrm{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

- $x \in \mathbb{R}^D$ is **one token**.
- $\mu = \frac{1}{D}\sum_{d}x_d$ and $\sigma^2 = \frac{1}{D}\sum_d (x_d-\mu)^2$ are computed **across the features of that single token**. Batch statistics are never used.
- $\gamma, \beta \in \mathbb{R}^D$ are learned scale and shift.
- $\epsilon \approx 10^{-6}$ prevents division by zero.

Because it does not depend on the batch, LayerNorm behaves identically at batch size 1 and batch size 4096. This matters when pretraining at scale with small per-device batches, and is why the ResNets used inside ViT hybrids replace BatchNorm with GroupNorm (§7.2).

### 2.4 The MLP (also called FFN)

$$\mathrm{MLP}(z) = \mathrm{GELU}(zW_1 + b_1)W_2 + b_2$$

- $W_1 \in \mathbb{R}^{D\times 4D}$ expands the width 4×.
- $W_2 \in \mathbb{R}^{4D\times D}$ projects back down.
- $\mathrm{GELU}(x) \approx x,\Phi(x)$ where $\Phi$ is the standard normal CDF. It is smooth and _non-monotonic_: it has a minimum near $x \approx -0.75$ where its value is about $-0.17$, not 0. It is often described as a "gate", which is a useful metaphor, but unlike ReLU it is not a hard on/off switch.
- The MLP acts **independently on each token** (position-wise). There is no interaction across the $N$ axis anywhere inside it.

So the block alternates two kinds of mixing:

$$\text{MSA} = \text{mixing across tokens (spatial)} \qquad \text{MLP} = \text{mixing across features (channel)}$$

A useful interpretation of the MLP, supported by _Transformer Feed-Forward Layers Are Key-Value Memories_ (Geva et al. 2021): $zW_1$ compares the token against $4D$ learned "keys", GELU selects which fire, and $W_2$ reads out the corresponding "values". Because which neurons fire depends on the input token, the MLP is a form of **input-conditional computation**, not a fixed statistical transform.

There is also a hard theoretical result on why the MLP must be there: _Attention is Not All You Need_ (Dong et al. 2021) proves that a stack of pure self-attention layers, with MLPs and skip connections removed, converges to a rank-1 output **doubly exponentially fast in depth** — every token collapses to the same vector. MLPs and residual connections are what prevent this.

### 2.5 Permutation equivariance — the property that causes everything

Let $\Pi$ be a permutation matrix (it reorders rows). Then:

$$Q' = (\Pi X)W_Q = \Pi Q, \qquad K' = \Pi K, \qquad V' = \Pi V$$

$$S' = \frac{\Pi Q (\Pi K)^\top}{\sqrt{d_k}} = \frac{\Pi Q K^\top \Pi^\top}{\sqrt{d_k}} = \Pi S \Pi^\top$$

Softmax is applied row-wise, and right-multiplying by $\Pi^\top$ only permutes columns, so $A' = \Pi A \Pi^\top$. Therefore, using $\Pi^\top\Pi = I$:

$$\mathrm{Attn}(\Pi X) = A'V' = \Pi A \Pi^\top \Pi V = \Pi A V = \Pi,\mathrm{Attn}(X)$$

$$\boxed{;\mathrm{Attn}(\Pi X) = \Pi,\mathrm{Attn}(X);}$$

**Self-attention has no concept of order or position whatsoever.** Shuffle the tokens and the output is shuffled identically — no value changes. To a bare Transformer, a photograph of a dog and the same photograph cut into 196 squares and shaken in a bag are literally the same input.

This single fact is the root of:

- why position embeddings must exist at all,
- what "ViT has weak inductive bias" concretely means,
- why patch-level design decisions matter so much.

---

## 3. Patch embedding: Equation (1)

### 3.1 Cutting the image

$$x \in \mathbb{R}^{H\times W\times C} ;\longrightarrow; x_p \in \mathbb{R}^{N\times (P^2 C)}, \qquad N = \frac{HW}{P^2}$$

Row $i$, written $x_p^i$, is the $P\times P\times C$ block number $i$ flattened into a vector of length $P^2C$.

Note carefully what the indices mean:

- **$p$** is a label meaning "this is patched, not raw" — it is not an index.
- **$i$** is the **patch index**, $i \in {1, \dots, N}$: which square of the board this is. It is _not_ an index into the vector's components.

For ViT-B/16 at 224 resolution:

$$N = 196, \qquad P^2C = 16\times16\times3 = 768$$

Adjacent patches share **zero** pixels. The partition is a hard tiling.

### 3.2 The flatten is a fixed bijection, and $E$ is a set of conv filters

Flattening is not arbitrary scrambling. It is a fixed, invertible index map

$$j ;\longleftrightarrow; (a, b, c), \qquad a, b \in {1..P}, ; c\in{1..C}$$

where $a,b$ are the row and column _inside_ the patch and $c$ is the channel. Crucially, **this map is the same for every patch**: component 5 of patch 1 and component 5 of patch 196 refer to the same in-patch location and the same colour channel.

Now project:

$$e_i = x_p^i E, \qquad E \in \mathbb{R}^{(P^2C)\times D}$$

Expand one output dimension $d$:

$$e_i[d] = \sum_{j=1}^{P^2C} x_p^i[j],E[j,d] = \sum_{a=1}^{P}\sum_{b=1}^{P}\sum_{c=1}^{C} x^i(a,b,c); W_d(a,b,c)$$

where $W_d(a,b,c) := E[,j(a,b,c),, d,]$.

**This is exactly the definition of a convolution filter.** $E$ is not "like" a convolution — it _is_ a bank of $D$ filters of size $P\times P\times C$, written in matrix form. Correspondingly, every real implementation writes it as:

```python
# timm / HuggingFace: the patch embedding IS a conv layer
self.proj = nn.Conv2d(in_channels=3, out_channels=768,
                      kernel_size=16, stride=16)
```

Parameter count for ViT-B/16: $768 \times 768 + 768 = 590{,}592$.

So the frequently repeated claim "ViT contains no convolution" is false as stated. The accurate version is: **ViT contains exactly one convolution, and it is the weakest possible one** — a single layer, with kernel size equal to stride (no overlap), and no nonlinearity after it.

### 3.3 $E$ has no 2D prior — a short proof

It is tempting to conclude from §3.2 that the flatten is harmless because $E$ can learn 2D structure. It can. But it has no _preference_ for it, and that is different.

Let $\Pi$ be any fixed permutation of the $P^2C$ in-patch indices, applied identically to every patch. Define $E' = \Pi^{-1}E$. Then:

$$(x_p^i \Pi),E' = x_p^i \Pi \Pi^{-1} E = x_p^i E$$

**The model realises exactly the same function.** Scrambling the pixels inside every patch (in the same way each time) changes nothing about what ViT can learn or how easily it learns it.

Compare with a $3\times3$ convolution, where such a scramble would be catastrophic, because conv _hardcodes_ which positions are neighbours.

$$\boxed{;E \text{ \emph{can} learn 2D structure; conv is \emph{forced} into it};}$$

This is the precise sense in which the patch embedding "destroys spatial structure": no information is lost (see §3.4), but no spatial prior is supplied either.

### 3.4 Is information lost? Depends on the variant

$E \in \mathbb{R}^{(P^2C)\times D}$.

- **ViT-B/16**: $P^2C = 768 = D$. Square matrix; if full rank, invertible; $x_p^i = e_i E^{-1}$. Information is fully preserved.
- **ViT-B/32**: $P^2C = 32\cdot32\cdot3 = 3072$, but $D = 768$. This is a **4× compression**. Rank is at most 768, so the kernel has dimension 2304 — information is genuinely and irreversibly lost. And ViT-B/32 works fine.

So "the projection is lossless" is a property of one specific configuration, not a general principle, and the fact that the lossy configuration also works shows that losslessness was never the reason ViT functions.

**The important distinction, stated once so it can be reused:**

$$\text{information preserved} ;\neq; \text{structure usable}$$

A fixed permutation of an image's pixels is perfectly invertible and destroys a CNN completely. Inductive bias lives on the right-hand side of that inequality.

### 3.5 Class token and position embedding

$$z_0 = \left[,x_\text{class},;; x_p^1 E,;; x_p^2 E,;; \dots,;; x_p^N E,\right] + E_\text{pos} \tag{1}$$

- $x_\text{class} \in \mathbb{R}^D$ is a **learned vector that does not depend on the image**, prepended at index 0. Borrowed from BERT. It is a dedicated "read slot": a place for the network to accumulate a summary without having to overwrite any patch's content.
- $E_\text{pos} \in \mathbb{R}^{(N+1)\times D}$ is a **learned** position embedding, added elementwise. Row $i$ says "this slot is location $i$".
- $z_0 \in \mathbb{R}^{(N+1)\times D}$; for ViT-B/16 this is $197 \times 768$.

$E$ is shared across all patches — one matrix for the whole model, not one per position. This gives parameter economy ($0.59$M instead of $196 \times 0.59$M $\approx 115$M), lets a pattern learned at one location transfer to any other, and allows the model to run at a different input resolution without new weights.

The flip side of that sharing is stated exactly: because $e_i = x_p^i E$ takes only $x_p^i$ as input and $i$ appears nowhere on the right-hand side,

$$x_p^i = x_p^k ;\Longrightarrow; e_i = e_k \qquad \forall i,k$$

**Identical content at two different locations produces identical embeddings.** Combined with §2.5, this means that without $E_\text{pos}$ the entire model is permutation-invariant at the output: a bag of patches. $E_\text{pos}$ is the only thing standing between ViT and that degeneracy.

Note also that $E_\text{pos}$ is a _learned_ absolute encoding — nobody tells the model that slot 1 is adjacent to slot 2. It has to discover the 2D topology of the grid from data. (It does; see §8.2.) Everything about why this is a strange design, and what replaced it, is in note 8.

---

## 4. The encoder: Equations (2)–(4)

For $\ell = 1, \dots, L$:

$$z'_\ell = \mathrm{MSA}\left(\mathrm{LN}(z_{\ell-1})\right) + z_{\ell-1} \tag{2}$$

$$z_\ell = \mathrm{MLP}\left(\mathrm{LN}(z'_\ell)\right) + z'_\ell \tag{3}$$

$$y = \mathrm{LN}\left(z_L^0\right) \tag{4}$$

Read line by line:

- **(2)** Normalise, let tokens exchange information via attention, add the result back to the input. The `+ z_{\ell-1}` is a **residual (skip) connection**.
- **(3)** Normalise again, let each token process its own updated content through the MLP, add back.
- **(4)** Take **row 0** of the final layer — the class token position — apply a last LayerNorm, and hand it to the classification head.

Both sub-layers share the shape `output = f(LN(input)) + input`. This is **pre-LN** (normalisation before the sub-layer), which differs from the original 2017 Transformer's post-LN. Pre-LN keeps the residual path free of normalisation, so gradients flow cleanly through it, which is what makes very deep stacks trainable.

### 4.1 The residual stream

Unrolling the recursion:

$$z_L = z_0 + \sum_{\ell=1}^{L}\left(\mathrm{MSA}_\ell + \mathrm{MLP}_\ell\right)$$

This has a consequence worth internalising: **$z_0$ is still present at layer $L$**. Whatever was written at the start — including $E_\text{pos}$ — remains in the stream unless the network actively learns to subtract it.

A productive mental model is that the residual stream is a **fixed-width communication bus of $D$ dimensions**, shared by every layer, which each sub-layer reads from and writes to additively. It is a scarce resource. Many of the phenomena in note 9 (register tokens, stochastic depth) are best understood as managing pressure on this bus.

### 4.2 Class token versus global average pooling

The ViT paper's appendix reports that replacing the class token with global average pooling over patch tokens works about as well, but requires a different learning rate. Mechanistically, the class token is a slot whose local content is free — it is not obliged to represent any patch — so it can be used purely for aggregation. This idea (a slot with no local duties) returns in a much more interesting form as register tokens.

---

## 5. Model configurations

| Model     | $L$ | $D$  | MLP width | heads $h$ | $d_k$ | params |
| --------- | --- | ---- | --------- | --------- | ----- | ------ |
| ViT-Base  | 12  | 768  | 3072      | 12        | 64    | 86M    |
| ViT-Large | 24  | 1024 | 4096      | 16        | 64    | 307M   |
| ViT-Huge  | 32  | 1280 | 5120      | 16        | 80    | 632M   |

Naming: `ViT-B/16` = Base with patch size 16. Smaller patch = more tokens = more compute and better accuracy; `ViT-L/16` beats `ViT-L/32` clearly, and the Huge model uses `/14`.

Where the parameters go, for ViT-B/16:

| Component                          | Params               |
| ---------------------------------- | -------------------- |
| Patch embedding $E$                | 0.59M                |
| Position embedding $E_\text{pos}$  | 0.15M                |
| Per block: MSA ($W_Q,W_K,W_V,W_O$) | $4D^2 \approx 2.36$M |
| Per block: MLP ($W_1, W_2$)        | $8D^2 \approx 4.72$M |
| 12 blocks                          | $\approx 85$M        |

Note the ratio: **the MLP holds twice as many parameters as attention.** The stem holds under 1% of the model. This is worth remembering when reading claims about which component "does the work."

### 5.1 Fine-tuning at higher resolution

Pretrain at 224, fine-tune at 384, keeping $P = 16$:

- $N$ changes from $14^2 = 196$ to $24^2 = 576$.
- $E$ is reusable unchanged — its input dimension $P^2C$ did not change.
- $E_\text{pos}$ has only 197 rows and now needs 577. The fix is **2D bicubic interpolation** of the position embeddings according to their true grid locations.

That interpolation step is the one place in the entire architecture where a human manually injects knowledge of 2D layout. Everywhere else the model is left to discover it. This is a design smell, and note 8 covers the encodings that removed the need for it.

---

## 6. Inductive bias: the central concept

**Inductive bias** = the set of assumptions built into a model that determine which hypotheses it prefers, before any data is seen. Less bias means a larger space of representable functions, which means more data is needed to pin down the right one.

### 6.1 What a CNN has for free

**(a) Translation equivariance.** Let $T_\delta$ shift a signal: $(T_\delta x)[i] = x[i - \delta]$. Then for convolution with kernel $w$:

$$(w * T_\delta x)[i] = \sum_k w[k], x[i - k - \delta] = (w_x)[i-\delta] = \big(T_\delta(w_x)\big)[i]$$

$$\boxed{;w * T_\delta x = T_\delta (w * x);}$$

Shift the image, the feature map shifts identically. Exactly, for free, with no training and no data.

> Note the vocabulary: this is **equivariance** (shift in → shift out), not **invariance** (shift in → no change out). Invariance comes from a later global pooling step, not from convolution itself. These are routinely confused.

**(b) Locality.** Output at position $i$ depends only on inputs within a $k\times k$ window. The assumption "nearby pixels are more related than distant ones" is hardcoded.

**(c) Weight sharing.** One kernel is applied at every location; a dog in the top-left is processed by the same filter as a dog in the bottom-right.

### 6.2 What ViT has and lacks

| Property                          | ViT        | Notes                                           |
| --------------------------------- | ---------- | ----------------------------------------------- |
| Patch-level locality              | ✅ partial | one layer only, hard boundaries                 |
| Weight sharing in $E$ and MLP     | ✅         | but see the caveat below                        |
| Translation equivariance          | ❌         | holds only for shifts that are multiples of $P$ |
| Locality across patches           | ❌         | layer 1 is already fully global                 |
| Knowledge of which patches adjoin | ❌         | must be learned via $E_\text{pos}$              |

On the third row: patchify commutes with $T_\delta$ **only when $\delta \equiv 0 \pmod P$**. Shift an image by 8 pixels with $P = 16$ and the tiling grid falls differently, so every token takes a new value. This is the same phenomenon as the "orange straddling two patches" intuition: an object crossing a patch boundary is split, and the two halves cannot be related until attention runs in layer 1.

### 6.3 Why less bias means more data

Cordonnier et al. (2020), _On the Relationship between Self-Attention and Convolutional Layers_, prove that multi-head self-attention **with relative positional encoding** can express any convolutional layer, provided the number of heads satisfies $h \ge k^2$ for a $k\times k$ kernel.

(The phrase "with relative positional encoding" is a load-bearing condition, not a technicality — see note 8.)

So in terms of expressiveness:

$$\mathcal{H}_\text{conv} \subset \mathcal{H}_\text{attn}$$

where $\mathcal{H}$ denotes the hypothesis space, the set of functions the architecture can represent. Attention can do everything convolution can, and more.

**But representable is not the same as findable.** A CNN is a Transformer that has already been _constrained_ to a region of function space where good solutions live. ViT has to walk to that region itself, using SGD, guided only by data.

- Little data → SGD does not find it → it settles on a solution that fits the training set but generalises poorly.
- Enough data → SGD finds it, **and** can go beyond what the convolutional constraint would have allowed.

### 6.4 The empirical scaling result

| Pre-training set | Size | Outcome on ImageNet                                      |
| ---------------- | ---- | -------------------------------------------------------- |
| ImageNet-1k      | 1.3M | ViT clearly **loses** to BiT-ResNet; bigger ViT is worse |
| ImageNet-21k     | 14M  | Roughly comparable; ViT-L begins to pull ahead           |
| JFT-300M         | 303M | ViT **wins** at every size; ViT-H/14 sets SOTA           |

The most diagnostic row is the first: **on ImageNet-1k, ViT-Large performs worse than ViT-Base.** A bigger model doing worse is the signature of a hypothesis space that is too large for the available constraint.

> **Important caveat, and it matters more than the table.** This result is specific to the 2020 training recipe. Subsequent work (DeiT, AugReg, MAE, DeiT-III) showed that strong augmentation, regularisation, and self-supervised objectives substitute for raw data quantity to a very large degree — ViT trained on ImageNet-1k alone reaches 83–85%. The statement "ViT requires JFT-300M" is a fact about 2020 tooling, not a law of nature. See note 9.

---

## 7. The hybrid architecture

The ViT paper itself proposes a variant in §3.1, and this is where the "CNN front half → Transformer back half" design lives.

### 7.1 The equations

Let $g_\theta$ be a CNN backbone:

$$f = g_\theta(x) \in \mathbb{R}^{H'\times W'\times C'}$$

Then use **patch size 1** on this feature map — each spatial location becomes one token:

$$N = H'W', \qquad e_i = f_i E, \qquad E \in \mathbb{R}^{C'\times D}$$

Compared with pure ViT, exactly one thing changes: $f_i \in \mathbb{R}^{C'}$ replaces $x_p^i \in \mathbb{R}^{P^2C}$. Equation (1)'s class token and position embedding, and all of Equations (2)–(4), are **character-for-character identical**.

### 7.2 The concrete configuration: R50 + ViT-B/16

The paper chose the setup so the comparison is clean:

- A **ResNet-50** in BiT style: BatchNorm replaced by **GroupNorm**, and **weight-standardised convolutions**. (Rationale: at large-scale pretraining with small per-device batches, BN statistics are unreliable; GN, like LN, does not depend on the batch.)
- **Stage 4 is removed** and its blocks are moved into stage 3, so the block counts go from $[3,4,6,3]$ to $[3,4,9]$, preserving total depth.
- The output of that extended stage 3 is downsampled **16×** overall.

At 224 input:

$$H' = W' = 14 ;\Longrightarrow; N = 196, \qquad C' = 1024, \qquad E \in \mathbb{R}^{1024\times768}$$

**$N = 196$ matches ViT-B/16 exactly**, so the Transformer half has identical compute. All the difference is concentrated in the stem. This is a clean controlled experiment and the comparison methodology is worth copying.

### 7.3 Three separate mechanisms by which it helps

**(a) Overlapping receptive fields.** In pure ViT, token $i$ sees a $16\times16$ pixel set completely disjoint from every other token's; cross- boundary information waits for layer 1. In the hybrid, each token is the output of dozens of convolutions and its theoretical receptive field spans hundreds of pixels (the _effective_ receptive field, per Luo et al. 2016, is smaller and roughly Gaussian, but still heavily overlapping with neighbours). Boundary information is already blended before the Transformer sees anything.

**(b) Nonlinearity before tokenisation.** The ViT stem is a single affine map: $x_p^i E + b$, no activation. Its output is a linear combination of raw pixels. The hybrid stem is conv → GN → ReLU repeated many times, so its tokens are already semantic features (edges, textures, parts) rather than pixel mixtures.

**(c) It supplies what ViT learns least efficiently.** See §8.3 — the measurement that early ViT layers spend capacity re-deriving local processing, and that hybrids do not need to.

### 7.4 The result

The paper plots accuracy against pre-training compute for BiT-ResNets, ViTs, and hybrids. Three conclusions:

1. ViT dominates ResNet on the performance/compute tradeoff — 2–4× less compute for the same accuracy.
2. **Hybrids slightly outperform pure ViT at small compute budgets.**
3. The difference **vanishes at large model/compute scale** — and the paper explicitly calls this surprising, since one would naively expect convolutional local processing to help at any size.

Interpretation: a CNN stem is borrowed prior. It is valuable when you cannot afford to learn the prior from data. Once you can, the borrowed prior starts to constrain rather than help.

---

## 8. What §4.5 actually measured

This section of the paper is more useful than its benchmark tables, because it supplies three probes that later work reuses.

### 8.1 Probe on $E$ — inspecting the stem

Take $E \in \mathbb{R}^{768\times768}$, run PCA on it, reshape the top principal components back to $16\times16\times3$ and view them as images. The result: **Gabor-like and low-frequency 2D basis functions**, closely resembling the first layer of a trained CNN.

Confirms §3.2: a single linear map does learn intra-patch structure. But it learns exactly one level of it, with no nonlinearity to build on.

### 8.2 Probe on $E_\text{pos}$ — does the model recover 2D?

Compute $\cos\big(E_\text{pos}[i],, E_\text{pos}[j]\big)$ for all pairs and plot, per patch, its similarity to the whole $14\times14$ grid.

Result: patches close together have high similarity, and clear row/column structure appears. **The model reconstructs the 2D topology of the image on its own**, from freely-parameterised 1D vectors. Note 8 examines what these vectors are doing in much more depth.

### 8.3 Mean attention distance — inspecting the encoder

For head $m$ in layer $\ell$:

$$\bar d^{(\ell,m)} = \frac{1}{N}\sum_{i}\sum_{j} A^{(\ell,m)}_{ij},\lVert p_i - p_j\rVert_2$$

where $p_i \in \mathbb{R}^2$ is the centre of patch $i$ in **pixel** units. This is the average distance a head looks, and it is directly comparable to a CNN's receptive field size.

Findings:

|                        | early layers                               | deep layers |
| ---------------------- | ------------------------------------------ | ----------- |
| **Pure ViT**           | some heads with small $\bar d$, some large | all global  |
| **R50 + ViT (hybrid)** | **no small-$\bar d$ heads**                | all global  |

Two things follow. First, ViT spontaneously develops local, convolution-like behaviour in early layers — but has to spend capacity and data doing so. Second, **the hybrid does not need to**, because the conv stem already did that work.

This is the one mechanistic piece of evidence in the paper for _why_ hybrids help, as opposed to _that_ they help.

> **Precision.** These low-$\bar d$ heads are not convolutions. They have no translation equivariance, and their weights depend on token content rather than on spatial offset. The correct statement is "some heads learn locally scoped behaviour," not "ViT becomes a CNN."

### 8.4 Attention rollout

Composing attention matrices across layers (accounting for residual connections) and tracing back from the output token shows attention concentrating on semantically relevant image regions. Useful as a qualitative sanity check; note that it is a heuristic, not a faithful attribution method.

---

## 9. Common misconceptions, corrected

Collected here because each one is repeated widely, including by otherwise careful sources.

**"ViT has no convolution."** False. The patch embedding _is_ `Conv2d(3, D, kernel=P, stride=P)`. The accurate claim is that it is a single-layer, non-overlapping convolution with no activation — the weakest form a convolution can take.

**"Multi-head attention splits the vector between heads."** False. Every head reads the full $D$-dimensional token; the projection matrices have input dimension $D$. What is per-head is the _output_ subspace of size $d_k = D/h$. The reason for $d_k = D/h$ rather than $D$ is cost parity, not information partitioning.

**"The linear projection $E$ preserves all information, so nothing is lost."** Two errors. First, this only holds when $D \ge P^2C$; ViT-B/32 compresses 4× and loses information irreversibly, and works fine. Second, invertibility is irrelevant to the actual question: a fixed permutation of pixels is perfectly invertible and destroys a CNN. What is lost is _prior_, not _information_.

**"Weight sharing in $E$ gives translation invariance."** Two errors. It is equivariance, not invariance. And it holds only for shifts that are exact multiples of $P$; sub-patch shifts change every token.

**"$E$ can't encode position, so patches only work by luck."** $E$ encodes position **within** a patch fully — $W_d(a,b,c)$ depends on $(a,b)$, which is how it learns edge detectors. It cannot encode position **between** patches, because $i$ never enters the formula. Two different levels; both matter.

**"The model has to wait several blocks to learn intra-patch structure."** No. All intra-patch information sits inside a single token vector from layer 0. The MLP in block 1 can use it immediately. Attention is not involved.

**"ViT requires JFT-300M."** True of the 2020 recipe only. DeiT, AugReg, MAE and DeiT-III all reach 83–85% on ImageNet-1k alone. The binding constraint was training methodology, not data volume per se.

**"CNNs are obsolete / ViT is strictly better."** ConvNeXt (Liu et al. 2022) took a plain ResNet, applied ViT-era design choices (large kernels, LN, GELU, inverted bottleneck, modern recipe) and matched Swin. The variable that determines outcomes is scale, data, and objective — not the choice of operator.

---

## 10. Summary

The through-line of this note in seven statements:

1. Transformers consume sequences; images are grids; the **stem** is the converter, and it is where nearly every design question lives.
2. Patches exist because pixel-level attention costs 65,536× more. They are a compute concession.
3. The patch embedding $E$ is literally a convolution — one layer, no overlap, no nonlinearity.
4. Self-attention is **permutation equivariant**, so without position embeddings the model is a bag of patches. This is provable in three lines and drives everything downstream.
5. ViT's information is intact but its **prior** is not. Convolution's translation equivariance and locality are given away; the model must rediscover them from data. That is why 2020-era ViT needed 300M images.
6. The **hybrid** (CNN stem → Transformer) restores that prior, and helps most exactly where compute and data are scarce. The paper measured why: hybrids don't need to grow local attention heads.
7. **Everything in point 5 is about the training recipe as much as the architecture.** Do not read the 2020 data-scaling table as a law.

### Where to go next

- **Note 8** — position encoding in depth: what "position" actually means, why adding it to content does not destroy either, and the encodings (relative bias, RoPE, conditional PE) that superseded ViT's approach.
- **Note 9** — the post-2020 landscape: DeiT, Early Convolutions, Registers, Swin, TNT, T2T, plus which research questions are closed and which are open.

### Reading list for this note

| Paper                                                                                         | Why                                                                        |
| --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Dosovitskiy et al. 2020, _An Image is Worth 16×16 Words_                                      | The source. Read §3 and §4.5; skip the benchmark tables.                   |
| Cordonnier et al. 2020, _On the Relationship between Self-Attention and Convolutional Layers_ | The formal $\mathcal{H}_\text{conv}\subset\mathcal{H}_\text{attn}$ result. |
| Dong et al. 2021, _Attention is Not All You Need_                                             | Rank collapse; why the MLP is structurally necessary.                      |
| Geva et al. 2021, _Transformer Feed-Forward Layers Are Key-Value Memories_                    | What the MLP is doing.                                                     |
| Luo et al. 2016, _Understanding the Effective Receptive Field_                                | Why theoretical RF overstates real RF.                                     |
