# 8 — Position in Transformers

> **What this document covers.** What "position" actually _is_ inside a Transformer, why adding a position vector to a content vector does not destroy both, what breaks when it does, and the full family of alternatives — relative bias, RoPE, ALiBi, conditional PE — with the mechanism of each. Ends with the specifically 2D question: how much spatial resolution does a vision Transformer actually need, and at what level does position stop being represented at all.
>
> **Why this deserves its own note.** Position encoding looks like a small implementation detail bolted onto the input. It is not. It is the **only** thing preventing a Transformer from being a set function, it determines which symmetry group the model respects, and the entire evolution from ViT (2020) to Swin and RoPE is a story about moving position out of the data and into the comparison operation.
>
> **Prerequisites.** Note 7, especially §2.5 (permutation equivariance) and §3.5 (Equation 1).

---

## Table of contents

1. [What "position" means: slot versus value](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#1-what-position-means-slot-versus-value)
2. [Absolute learned PE and the addition problem](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#2-absolute-learned-pe-and-the-addition-problem)
3. [The four-term decomposition](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#3-the-four-term-decomposition)
4. [Failure modes of additive absolute PE](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#4-failure-modes-of-additive-absolute-pe)
5. [Relative position bias](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#5-relative-position-bias)
6. [RoPE](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#6-rope)
7. [ALiBi, sinusoidal, and conditional PE](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#7-alibi-sinusoidal-and-conditional-pe)
8. [Comparison](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#8-comparison)
9. [Position below the patch: the pixel level](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#9-position-below-the-patch-the-pixel-level)
10. [How much spatial does a Transformer need?](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#10-how-much-spatial-does-a-transformer-need)
11. [The unifying frame: choosing a symmetry group](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#11-the-unifying-frame-choosing-a-symmetry-group)
12. [NoPE and why it does not apply to ViT](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#12-nope-and-why-it-does-not-apply-to-vit)
13. [Practical consequences for experiments](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#13-practical-consequences-for-experiments)

---

## 1. What "position" means: slot versus value

Before any equations, get this distinction right, because most confusion about position encoding dissolves once it is clear.

There are two fundamentally different ways a system can represent "where something is":

**(A) By slot.** The position is the _address at which the data is stored_. In a CNN feature map, the activation for location $(5,7)$ is literally stored at index $(5,7)$ of a tensor. Nothing needs to encode the position — the position _is_ the index. Distance between two features is read off directly by subtracting indices, and it is exact, metric, and available for free.

**(B) By value.** The position is encoded _inside the numbers_, and must be decoded by some learned operation. A vector $v$ that happens to mean "there is a vertical edge in the upper-left region of this patch" carries positional information, but only in a form that has to be read out.

The whole picture for ViT:

| Level                  | How position is carried                                   | Metric available? |
| ---------------------- | --------------------------------------------------------- | ----------------- |
| CNN feature map        | **slot** — cell $(5,7)$ _is_ position $(5,7)$             | yes, exact        |
| ViT, _between_ patches | **slot** (token index) + $E_\text{pos}$ giving it meaning | yes, learned      |
| ViT, _within_ a patch  | **value** only — no slot exists below patch granularity   | **no**            |

$$\boxed{;\text{usable position} = \text{position carried by a \textbf{slot}, not by a \textbf{value}};}$$

### 1.1 Working through the convolution case, because it is easy to get backwards

Consider a filter $\begin{bmatrix}-1 & 0.5\ -0.5 & -1\end{bmatrix}$. Does it "know" position?

- **Within its window: yes, absolutely.** The weight at index $(a,b)$ is permanently bound to the pixel at $(a,b)$. That binding is what makes edge detection possible at all.
- **In its output: no.** The output is a single scalar. All the positional structure has been collapsed into one number. Two different arrangements of pixels can produce the same response.

What recovers position is not the value of the response but **which cell of the feature map responded**. The filter provides selectivity; the grid provides position.

This is why "the patch embedding destroys spatial structure" and "the patch embedding is a convolution" are both true and not in conflict. $E$ has the same within-window positional binding as any convolution. What it lacks is a _grid below the patch_ — everything inside $16\times16$ pixels gets collapsed into one $D$-dimensional value with no addressable sub-structure.

### 1.2 Consequently

$$\text{patch size} = \text{the boundary between position-as-slot and position-as-value}$$

Above $16$ pixels: token indices give you a grid, and $E_\text{pos}$ tells the model what that grid means. Below $16$ pixels: no slots exist, only values.

This immediately explains a fact from note 7 that otherwise looks arbitrary: smaller patches consistently work better (ViT-L/16 > ViT-L/32, ViT-H uses /14). Reducing patch size does not just add capacity — it moves the slot/value boundary downward, converting position-that-must-be-decoded into position-that-is-free.

---

## 2. Absolute learned PE and the addition problem

### 2.1 The construction

From Equation (1) of note 7:

$$z_0 = \left[x_\text{class}; x_p^1E; \dots; x_p^NE\right] + E_\text{pos}, \qquad E_\text{pos}\in\mathbb{R}^{(N+1)\times D}$$

Write $e_i$ for the content part and $p_i$ for the position part of token $i$:

$$z_i = e_i + p_i$$

- $e_i = x_p^i E$ — depends on the image, changes every sample.
- $p_i = E_\text{pos}[i]$ — a free learned parameter, **the same for every image in the dataset**, depending only on the slot index.

For ViT-B/16, $E_\text{pos}$ is $197\times768 = 151{,}296$ parameters.

### 2.2 The obvious objection

If you add 2 and 3 you get 5, and you cannot recover the operands. So how can the model possibly tell content from position after adding them? And if they are entangled, surely there are cases where they interfere destructively?

The short honest answers: **they do mix, there is no tag marking which is which, the model has no built-in knowledge of the split, and yes, interference is real and measurable.** What makes it work anyway is high-dimensional geometry.

### 2.3 Near-orthogonality in high dimensions

For two independent random vectors $u, v$ uniform on the unit sphere in $\mathbb{R}^D$:

$$\mathbb{E}[\cos(u,v)] = 0, \qquad \mathrm{Std}[\cos(u,v)] = \frac{1}{\sqrt{D}}$$

At $D = 768$: $1/\sqrt{768} \approx 0.036$. Two random directions are almost always between about 88° and 92° apart. **High-dimensional space is overwhelmingly composed of near-orthogonal directions.**

This changes what addition means. In one dimension, addition is destructive mixing. In many dimensions, addition of near-orthogonal components is **superposition** — closer to two audio signals occupying different frequency bands than to mixing two colours of paint. Superposed signals can be filtered apart; mixed paint cannot.

### 2.4 Linear recovery — the proof

Suppose content vectors live (approximately) in a subspace $U$ and position vectors in a subspace $V$, with $U \perp V$. Let $P_V$ be the orthogonal projection onto $V$. Then:

$$P_V(e_i + p_i) = \underbrace{P_V e_i}_{=,0} + P_V p_i = p_i$$

**Exact recovery, no loss.** And symmetrically for $P_U$ recovering $e_i$.

Now the key observation: **a projection is a matrix multiplication**, and $W_Q, W_K, W_V$ are matrix multiplications. The architecture already contains, in every attention head of every layer, exactly the operation required to separate the two. It gets the capability for free; it only needs to learn which subspace to project onto.

### 2.5 Is there room? A capacity count

$E_\text{pos} \in \mathbb{R}^{197\times 768}$ has **at most rank 197** — there are only 197 rows. So position can occupy at most 197 of the 768 available dimensions, leaving at least 571 for content.

In practice it uses far fewer. Trained $E_\text{pos}$ matrices are smooth and low-rank: PCA reveals a small number of sinusoid-like 2D basis patterns, so effective rank is more like 20–40. There is plenty of room.

### 2.6 How does the model "know" which subspace is which?

It does not know, and nobody tells it. But there is a strong statistical signature to exploit:

$$p_i \text{ is constant across every image in the dataset} \qquad\text{vs}\qquad e_i \text{ varies with every image}$$

Over millions of samples, the optimiser is looking at each slot and seeing a component that never moves superimposed on a component that always moves. If the two occupy the same directions, the model confuses them, the loss goes up, and gradients push them apart.

**The subspace separation is an emergent property of training, not a design feature.** No line of code enforces it. This is worth stating plainly because it means the separation is only ever approximate, and can fail — which is §4.

---

## 3. The four-term decomposition

This is the single most useful equation in the whole topic. Everything about why additive PE is problematic, and how each alternative fixes it, is visible here.

Let $A := W_Q W_K^\top \in \mathbb{R}^{D\times D}$ (folding both projections into one matrix). Then the pre-softmax attention logit between tokens $i$ and $j$ is:

$$S_{ij} = \frac{z_i A z_j^\top}{\sqrt{d_k}} = \frac{1}{\sqrt{d_k}}\Big[ \underbrace{e_i A e_j^\top}_{(1)} + \underbrace{e_i A p_j^\top}_{(2)} + \underbrace{p_i A e_j^\top}_{(3)} + \underbrace{p_i A p_j^\top}_{(4)} \Big]$$

Reading each term:

| Term    | Form              | Meaning                                                                                |
| ------- | ----------------- | -------------------------------------------------------------------------------------- |
| **(1)** | content–content   | "Do these two contents go together?" Ignores location entirely.                        |
| **(2)** | content–position  | "Given what token $i$ contains, which locations should it look at?"                    |
| **(3)** | position–content  | "Given where token $i$ is, what content should it seek?"                               |
| **(4)** | position–position | **Pure spatial prior.** Which location attends to which, independent of image content. |

### 3.1 Where local heads come from

Term (4) is the mechanistic answer to a question left open in note 7 §8.3: how does a head become "local"?

If a head shapes $A$ so that $p_i A p_j^\top$ is large when $\lVert p_i - p_j \rVert$ is small, it produces distance-decaying attention **without reference to the image at all**. That is a convolution-like head, and it is precisely what the mean-attention-distance probe detects in early layers.

Additive absolute PE therefore _can_ produce local heads. It just has to learn the whole distance function from scratch, encoded implicitly in a $D\times D$ matrix, from data.

### 3.2 Terms (2) and (3) are largely noise

Term (2) says "the raw content of patch $i$ interacts with the raw absolute coordinate of patch $j$." There is rarely a good reason for this to carry signal, and it consumes representational capacity.

This is not speculation. **TUPE** (Ke, He, Liu 2020, _Rethinking Positional Encoding in Language Pre-training_) removes the cross terms, computing content–content and position–position with separate projection matrices, and reports improvements. That is direct evidence that additive coupling generates interference rather than useful structure.

---

## 4. Failure modes of additive absolute PE

Collected evidence that the mixing is genuinely a problem, not just theoretically inelegant.

**(a) Cross-term interference.** §3.2 — TUPE's result.

**(b) Resolution changes require manual surgery.** Fine-tuning from 224 to 384 requires 2D bicubic interpolation of $E_\text{pos}$ (note 7 §5.1). A robust positional scheme would not need a human to reinterpret its parameters when the grid changes shape.

**(c) At initialisation the subspaces are not separated.** The separation in §2.6 is learned. At step 0, $e_i$ and $p_i$ are random and heavily overlapping. This is part of why ViT is fragile in early training and requires long linear warmup — the model is simultaneously trying to learn the task and to disentangle its own input representation.

**(d) Position permanently occupies the residual stream.** From note 7 §4.1:

$$z_L = z_0 + \sum_\ell (\cdots)$$

$p_i$ was written into $z_0$, so it is still sitting in the stream at layer 12, consuming bus width, unless the network learns to subtract it. Position is probably most useful in early layers; the additive scheme makes it pay rent throughout.

**(e) Absolute encoding fixes no symmetry.** Slots $(3,5)$ and $(10,12)$ have no built-in relationship, even though the offset between each pair is identical. Every relative relationship must be learned separately for every absolute pair.

---

## 5. Relative position bias

Used by T5 (1D) and Swin Transformer (2D). The cleanest fix conceptually.

### 5.1 The equation

$$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + B\right)V$$

Position is added **to the attention logits**, not to the token vectors. It never enters the residual stream at all.

In Swin, attention is computed inside $M\times M$ windows ($M = 7$). Offsets between two tokens in a window satisfy

$$\Delta x, \Delta y \in {-(M-1), \dots, M-1}$$

so there are $(2M-1)^2 = 169$ distinct offsets. Store a table $\hat B \in \mathbb{R}^{(2M-1)\times(2M-1)}$ and set

$$B_{ij} = \hat B[\Delta x_{ij},, \Delta y_{ij}]$$

Parameter cost: **169 values per head per layer** — trivially small compared with $E_\text{pos}$'s 151K.

### 5.2 The key insight: this is a convolution inside the softmax

Take the limiting case where the content term carries no information, i.e. $q_i k_j^\top$ is approximately constant in $j$. Then:

$$\alpha_{ij} = \mathrm{softmax}_j\left(\text{const} + B_{i-j}\right) = \underbrace{\mathrm{softmax}_j\left(B_{i-j}\right)}_{=:; w_{i-j}}$$

$$y_i = \sum_j \alpha_{ij}v_j = \sum_j w_{i-j},v_j = (w * v)_i$$

**That is a depthwise convolution**, with kernel $w = \mathrm{softmax}(B)$. Not an analogy — the same operation.

So attention with relative position bias is a **continuous spectrum**:

$$\text{convolution (fixed kernel)} ;\longleftrightarrow; \text{attention (content-driven)}$$

Each head picks its own point on that spectrum by the relative magnitude of $B$ versus $q k^\top$. Empirically, trained Swin heads spread across it: some learn distance-decaying kernels, some directional patterns, some near-flat (global).

The local behaviour that pure ViT required JFT-300M to discover is obtained here from 169 parameters.

### 5.3 Connecting back to Cordonnier

Note 7 §6.3 stated that self-attention **with relative positional encoding** can express any convolution, with the "relative" clause flagged as load-bearing. §5.2 shows why: the bias table is exactly the mechanism that lets attention _be_ a convolution. With only absolute additive encoding, expressing a convolution requires the model to reconstruct a distance function inside a $D\times D$ matrix — possible, but not natural.

### 5.4 The cost

- Complexity drops to $O(N\cdot M^2)$ instead of $O(N^2)$ — linear in $N$.
- **But window partitioning creates new hard boundaries** — the same problem as patchify, moved to a coarser scale. Swin mitigates this with **shifted windows**: alternate layers displace the window grid by $\lfloor M/2 \rfloor$ so tokens that were separated can communicate. That is a patch on the problem, not an elimination of it.
- The bias table is tied to a specific window size; changing $M$ requires interpolating the table.

---

## 6. RoPE

Rotary Position Embedding (Su et al. 2021). Widely adopted in LLMs, and increasingly in vision.

### 6.1 The construction

Pair up the dimensions of $q$ and $k$. Rotate pair $m$ by an angle proportional to the position $p$:

$$R_{p} = \bigoplus_{m=1}^{d_k/2} \begin{pmatrix} \cos p\theta_m & -\sin p\theta_m \ \sin p\theta_m & \cos p\theta_m \end{pmatrix}, \qquad \theta_m = \text{base}^{-2m/d_k}$$

with base typically $10{,}000$.

- $\bigoplus$ is a block-diagonal direct sum: each 2D subspace gets its own rotation.
- $\theta_m$ decreases with $m$, giving a range of angular frequencies. The effect is a **clock with many hands**: fast hands resolve fine position, slow hands resolve coarse position. This is closely analogous to place-value notation.
- It is applied to $q$ and $k$ **only**, inside each attention layer. The residual stream is untouched.

### 6.2 The relative property

$R$ is orthogonal and rotations compose: $R_a^\top R_b = R_{b-a}$. Hence:

$$(R_p q_i)^\top (R_j k_j) = q_i^\top R_p^\top R_j k_j = q_i^\top R_{,j-p}, k_j$$

$$\boxed{;\text{the logit depends only on } (j - p);}$$

Relative position falls out of the algebra. Nothing is learned, nothing is stored, no table is needed. Rotating the entire configuration (translating the image) leaves every pairwise angle unchanged — translation equivariance from geometry rather than from training.

### 6.3 Two-dimensional RoPE

Patch positions are $(x,y)$, not scalars. The standard approach is **axial RoPE**: split the head dimension in half, rotate one half by $x$ and the other by $y$:

$$q ;\longrightarrow; \left(R_{\Delta x} \oplus R_{\Delta y}\right)\text{-compatible form}$$

giving logits that depend on $(\Delta x, \Delta y)$. Heo et al. 2024, _Rotary Position Embedding for Vision Transformer_, study this along with mixed-frequency variants and find clear gains, especially when fine-tuning at a resolution different from pretraining — exactly where additive absolute PE is weakest.

### 6.4 Properties and limits

Advantages:

- **Zero parameters.**
- **Norm-preserving** ($\lVert Rq\rVert = \lVert q \rVert$), so it neither amplifies nor attenuates anything.
- **Consumes no residual-stream capacity at all.**
- Extrapolates to unseen positions better than learned absolute PE.

Limits:

- It can only affect the $q^\top k$ term. **It cannot make $V$ position-dependent**, whereas additive absolute PE can, because $p_i$ flows into the value projection too.
- Distance decay is not hardcoded. RoPE has a "long-term decay" property that emerges from summing many frequencies, but it is weaker and less directly controllable than an explicit bias table.

---

## 7. ALiBi, sinusoidal, and conditional PE

### 7.1 Sinusoidal absolute (Vaswani et al. 2017)

Fixed, not learned:

$$PE_{(p, 2m)} = \sin\left(\frac{p}{10000^{2m/D}}\right),\qquad PE_{(p, 2m+1)} = \cos\left(\frac{p}{10000^{2m/D}}\right)$$

Same multi-frequency clock idea as RoPE, but **added to the token** rather than used to rotate $q,k$. Zero parameters and defined for arbitrary $p$, so it extends to longer sequences — although quality degrades. Note that ViT tried this and found learned absolute performed comparably; the paper also tried 2D-aware variants and found little difference.

### 7.2 ALiBi (Press et al. 2021)

Radical simplicity — a linear penalty on distance added to the logits:

$$S_{ij} ;\mathrel{+}=; -,m_h \cdot |i - j|$$

where $m_h$ is a fixed, head-specific slope (a geometric sequence across heads, not learned). **Zero parameters**, no table, and very strong length extrapolation. It is a relative bias with the functional form fixed a priori rather than learned.

Conceptually: ALiBi hardcodes "attend less to things that are further away" with a different decay rate per head, which is a hand-designed version of what Swin's $B$ learns.

### 7.3 Conditional PE / CPVT

**CPVT** (Chu et al. 2021, _Conditional Positional Encodings for Vision Transformers_) removes $E_\text{pos}$ entirely and generates position from the tokens themselves.

The Positional Encoding Generator (PEG): reshape the token sequence back to an $H'\times W'$ grid, run a depthwise convolution, flatten, add back:

$$z ;\leftarrow; z + \mathrm{Flatten}\left(\mathrm{DWConv}_{3\times3}\left(\mathrm{Reshape}(z)\right)\right)$$

**Where does the position come from?** Zero padding. A token at the image border sees padding zeros in its convolution window; an interior token does not. The network can therefore infer distance-from-border, and from that, absolute position — from a single convolution.

Results: no $E_\text{pos}$ at all, **no interpolation needed when the resolution changes**, and better accuracy than learned absolute.

This has a direct implication for hybrid-architecture experiments, taken up in §13.

### 7.4 Supporting evidence: CNNs already leak position

Islam et al. 2020, _How Much Position Information Do Convolutional Neural Networks Encode?_, probed trained CNNs and found they encode **absolute position** substantially — and identified zero padding as the source. Remove padding, and positional probing accuracy falls toward chance.

CPVT is essentially the deliberate exploitation of that leak.

---

## 8. Comparison

| Scheme                   | Injected where               | Abs/Rel                 | Params              | Enters residual stream?   | Resolution change     |
| ------------------------ | ---------------------------- | ----------------------- | ------------------- | ------------------------- | --------------------- |
| Learned absolute (ViT)   | added to $z_0$               | abs                     | $N\timesD$ (151K)   | **yes, permanently**      | needs interpolation   |
| Sinusoidal               | added to $z_0$               | abs                     | 0                   | yes                       | extends, degrades     |
| Relative bias (Swin, T5) | added to logits              | rel                     | $(2M{-}1)^2$/head   | no                        | interpolate table     |
| RoPE                     | rotates $q,k$                | rel                     | **0**               | no                        | best in class         |
| ALiBi                    | added to logits              | rel                     | **0**               | no                        | very good             |
| CPE / PEG (CPVT)         | depthwise conv on token grid | rel (+ abs via padding) | one $3\times3$ conv | yes (as a residual write) | **nothing to change** |

### 8.1 The evolutionary arc

Line the three main approaches up by where position appears in the logit:

$$\text{ViT:}\quad S_{ij} = \frac{(e_i + p_i),A,(e_j+p_j)^\top}{\sqrt{d_k}} \qquad\text{— four entangled terms}$$

$$\text{Swin:}\quad S_{ij} = \frac{e_i A e_j^\top}{\sqrt{d_k}} + B_\Delta \qquad\text{— two clean, separate terms}$$

$$\text{RoPE:}\quad S_{ij} = \frac{q_i^\top R_\Delta k_j}{\sqrt{d_k}} \qquad\text{— one term, position inside the operator}$$

$$\boxed{;\text{The trend is moving position out of the \emph{data} and into the \emph{comparison operation}.};}$$

ViT puts position into the token values, then requires the model to disentangle it. Swin keeps it as a separate additive term, so nothing needs disentangling. RoPE embeds it in the bilinear form itself, so there is nothing separable in the first place.

And all three post-ViT schemes share one property: they depend on **$\Delta$, not on $i$ and $j$ separately.** That is a collective admission that spatial structure is a prior worth supplying, not a fact worth making the model derive from 300 million images.

---

## 9. Position below the patch: the pixel level

Now the question that motivated a lot of this note: **is there anything encoding position at the pixel level, or does that information simply not exist?**

### 9.1 The information is there; the metric is not

At ViT-B/16, if $E$ is full rank, then $x_p^i = e_i E^{-1}$ — the complete $16\times16$ pixel layout is recoverable from the token. Nothing is lost.

Also worth clearing up: **there is no waiting period.** All intra-patch information sits in one token vector from layer 0. The MLP in block 1 can use it immediately; attention is not involved in accessing it. There is no sense in which the model "has to wait several blocks to figure out what is inside a patch."

What is missing is what §1 identified: **no slot, therefore no free metric.** The model is not told that component $j$ and component $j+1$ of the flattened patch correspond to adjacent pixels, and the proof in note 7 §3.3 shows it has no preference for that being true.

### 9.2 So would adding pixel coordinates help?

This has been tested directly. **CoordConv** (Liu et al. 2018) appends two extra channels, $x$ and $y$ coordinate maps, to the input of a convolution. The results split sharply:

| Task                                                                  | Effect                                                                    |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Render a dot at a given coordinate; regress coordinates from an image | Plain CNN **fails almost completely**; CoordConv solves it near-perfectly |
| ImageNet classification                                               | **Essentially no change** — within noise                                  |

That is the direct answer. **Explicit absolute coordinates are transformative when the output is a coordinate, and worthless when the output is a category label.**

### 9.3 Why coordinate channels are not the right tool anyway

Suppose you concatenate $x,y$ to each pixel and project. Write the augmented projection as $E' = \begin{bmatrix} E \ w_x \ w_y \end{bmatrix}$:

$$[x_p,; x,; y],E' = x_p E + x,w_x + y,w_y$$

Compare with ViT's actual scheme, $x_p E + E_\text{pos}[i]$. **These have the same functional form.** The difference is the constraint:

- Coordinate channels: the positional contribution is forced to lie in a **rank-2 affine family** — the span of $w_x$ and $w_y$, varying linearly with $x$ and $y$.
- $E_\text{pos}$: an unconstrained $D$-dimensional vector per slot.

So the coordinate-channel idea is not wrong, it is a heavily restricted special case of what ViT already does. A rank-2 linear function of coordinates cannot express distance decay, periodicity, or any non-monotonic spatial structure. (This is also why the frequently given explanation "appending coordinates distorts the colour values" is not the real reason it is not used.)

### 9.4 And in practice, padding supplies it anyway

Per §7.4, any convolution with zero padding already leaks absolute position. Architectures with convolutional stems get this without asking for it. So the question "should we inject pixel coordinates?" is often moot — the answer depends on whether your stem has padding.

---

## 10. How much spatial does a Transformer need?

The honest answer: **it depends on the resolution of the output, not the resolution of the input.**

| Task                  | Spatial precision needed                  | Typical architectural response            |
| --------------------- | ----------------------------------------- | ----------------------------------------- |
| Classification        | Very low — arguably you want _invariance_ | ViT/16, even /32, is adequate             |
| Detection             | Medium — boxes accurate to a few pixels   | ViTDet, FPN, smaller patches              |
| Semantic segmentation | High — per-pixel                          | Hierarchical backbones, patch 8, decoders |
| Keypoint / pose       | Highest                                   | High-resolution heatmaps                  |

Evidence within the ViT paper itself: ViT-L/16 clearly beats ViT-L/32, and the Huge model uses /14. Spatial resolution has real value.

But note _how_ that value is best purchased. Smaller patches give you **actual slots** — addressable positions in the sense of §1 — whereas coordinate channels give only decodable values. Slots are what the architecture can use directly.

Caveat: comparing /16 against /32 confounds spatial resolution with compute, since token count scales as $1/P^2$. It is not a clean measurement of the value of resolution alone.

---

## 11. The unifying frame: choosing a symmetry group

The frame that makes the whole zoo legible. Consider what the attention score is a function of:

$$S_{ij} = f\left(\text{content}_i,\ \text{content}_j,\ i,\ j\right)$$

- **Absolute encoding**: $f$ depends on $i$ and $j$ **separately**. No symmetry is imposed. The relationship between slots 3 and 5 tells the model nothing about slots 10 and 12, even though the offset is the same.
- **Relative encoding**: $f$ depends only on $(i - j)$. This forces

$$S_{i+\delta,; j+\delta} = S_{ij} \qquad \forall \delta$$

which is **translation equivariance in the attention pattern**.

Recall from note 7 §6.2 that the first thing ViT gave up relative to a CNN was translation equivariance. Relative position encoding is exactly the mechanism that returns it — **without introducing a single convolution.**

Stated as a slogan: choosing a positional encoding is choosing which transformations of the input the model treats as "the same situation." That is a statement about inductive bias, not about implementation.

---

## 12. NoPE and why it does not apply to ViT

Kazemnejad et al. 2023 showed that **decoder-only** language models can be trained with _no positional encoding at all_ and still learn position — because the causal mask already breaks permutation symmetry. A token that can see 5 predecessors is structurally distinguishable from one that can see 50, so the model can count.

**This does not transfer to ViT.** ViT is a bidirectional encoder with no mask, so the permutation-equivariance proof (note 7 §2.5) holds exactly. Remove $E_\text{pos}$ and the model becomes a genuine bag of patches.

The contrast is instructive: positional information can come from an explicit encoding, from the _connectivity structure_ of attention (causal masking), or from architectural side effects (conv padding). ViT 2020 has only the first of these, which is why it depends on it so completely.

This asymmetry also explains a difference in how attention sinks manifest. In an LLM, the causal mask means token 0 is visible to everyone, so it becomes the canonical sink. ViT has no such privileged token, so the sink lands on arbitrary low-information background patches, different in every image — which is much messier. See note 9.

---

## 13. Practical consequences for experiments

### 13.1 The padding confound in hybrid-vs-plain comparisons

This one is easy to miss and can invert conclusions.

$$224 / 16 = 14 \text{ exactly} ;\Longrightarrow; \text{patchify uses \textbf{no padding}}$$ $$\text{A ResNet stem uses padding at every layer}$$

By §7.4, **the hybrid arm receives absolute positional information for free through padding, and the pure-ViT arm does not.** This has nothing to do with feature extraction. If a hybrid outperforms a plain ViT in your experiment, some unknown fraction of that margin may be coming purely through this channel.

### 13.2 An experiment that isolates it

**Remove $E_\text{pos}$ from both arms and retrain.**

- Pure ViT should collapse dramatically (bag of patches — this is a prediction with a proof behind it, so it doubles as a sanity check that the pipeline is correct).
- The hybrid should degrade far less, because padding supplies position.

Then:

$$\Delta = \mathrm{acc}_{\text{hybrid, no-PE}} - \mathrm{acc}_{\text{ViT, no-PE}}$$

$\Delta$ is a direct numerical estimate of **how much positional prior the convolutional stem is leaking in**, cleanly separated from any feature-extraction benefit. Two extra runs, and it decomposes a variable that headline accuracy numbers cannot.

### 13.3 Three separable axes

A conv stem changes at least three things at once, and papers routinely report their sum as a single number:

1. **Feature-extraction prior** — nonlinearity and overlapping receptive fields.
2. **Positional prior** — via padding; also obtainable from RPB, RoPE, or CPE with no convolution at all.
3. **Optimisation stability** — a smoother loss landscape (see note 9).

A $2\times2$ design over `{plain, hybrid} × {absolute PE, RoPE}` separates axis 1 from axis 2 immediately. Adding a learning-rate sweep separates axis 3.

---

## 14. Summary

1. **Position is only usable when carried by a slot**, not by a value. Patch size is exactly the boundary between the two regimes in ViT.
2. Adding $p_i$ to $e_i$ is not destructive because high-dimensional spaces are full of near-orthogonal directions, and a **linear projection recovers either component exactly** if the subspaces are orthogonal. $W_Q,W_K,W_V$ are already such projections.
3. The separation is **learned, not designed**, driven by the signature "position is constant across images, content varies."
4. The **four-term decomposition** of $S_{ij}$ explains everything: term (4) is how local heads emerge; terms (2) and (3) are largely interference, and TUPE showed removing them helps.
5. Additive absolute PE genuinely fails in identifiable ways: cross-term noise, interpolation on resolution change, entanglement at initialisation, permanent residual-stream occupancy.
6. **Relative position bias is a convolution living inside the softmax** — this is exact, not an analogy, and it explains why Cordonnier's theorem requires relative encoding.
7. **RoPE** obtains relative position from the algebra of rotations: zero parameters, norm-preserving, no residual-stream cost. It cannot make $V$ position-dependent.
8. **CPE/CPVT** shows that a single padded convolution supplies position, confirming that CNN stems leak positional prior — a confound in any hybrid-vs-plain comparison.
9. At the pixel level, information exists but no slot does. **Explicit coordinates help enormously for coordinate-valued outputs and negligibly for classification** (CoordConv). Coordinate channels are a rank-2 special case of what $E_\text{pos}$ already does.
10. The unifying frame: **positional encoding is the choice of symmetry group.** Absolute imposes none; relative imposes translation equivariance — returning the exact property ViT gave up, with no convolution required.

### Reading list

| Paper                                                                                | Why                                                 |
| ------------------------------------------------------------------------------------ | --------------------------------------------------- |
| Su et al. 2021, _RoFormer_                                                           | RoPE, original.                                     |
| Heo et al. 2024, _Rotary Position Embedding for Vision Transformer_                  | 2D RoPE; the resolution-transfer result.            |
| Liu et al. 2021, _Swin Transformer_                                                  | Relative position bias in 2D; windowing.            |
| Ke et al. 2020, _Rethinking Positional Encoding in Language Pre-training_ (TUPE)     | Evidence that cross terms are harmful.              |
| Press et al. 2021, _Train Short, Test Long_ (ALiBi)                                  | Zero-parameter relative bias.                       |
| Chu et al. 2021, _Conditional Positional Encodings for Vision Transformers_          | CPE/PEG; position from padded convolution.          |
| Islam et al. 2020, _How Much Position Information Do CNNs Encode?_                   | Padding is the source of absolute position in CNNs. |
| Liu et al. 2018, _An Intriguing Failing of CNNs and the CoordConv Solution_          | The definitive test of explicit coordinates.        |
| Kazemnejad et al. 2023, _The Impact of Positional Encoding on Length Generalization_ | NoPE; why causal masking substitutes for PE.        |
