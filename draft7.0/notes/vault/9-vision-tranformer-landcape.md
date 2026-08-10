# 9 — The Vision Transformer Landscape

> **What this document covers.** Everything that happened after the 2020 ViT paper. Organised not by chronology but by **where in the machine each paper intervenes**, because that framing shows that most of them are buying the same thing with different currency. Covers DeiT, Early Convolutions, Registers, Swin, TNT, T2T and the surrounding families, then the empirical "~2% ceiling" observation, then which research questions are settled and which are open, then how to design an experiment in this area that produces a conclusion you can actually interpret.
>
> **Prerequisites.** Notes 7 and 8.
>
> **Epistemic note.** Sections 1–7 are reported results. Section 8 (the ceiling) is an empirical pattern with no accepted theory behind it — treat it as a working hypothesis. Section 9 (open vs closed) is a judgement call about the state of a fast-moving field as of mid-2026; verify anything load-bearing.

---

## Table of contents

0. [The organising frame: four intervention points](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#0-the-organising-frame-four-intervention-points)
1. [What ViT §4.5 gave everyone: the probe toolkit](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#1-what-vit-45-gave-everyone-the-probe-toolkit)
2. [DeiT — intervening at the loss](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#2-deit--intervening-at-the-loss)
3. [Early Convolutions — intervening at the stem](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#3-early-convolutions--intervening-at-the-stem)
4. [Registers — intervening at the sequence](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#4-registers--intervening-at-the-sequence)
5. [Swin — intervening at the attention structure](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#5-swin--intervening-at-the-attention-structure)
6. [TNT — nesting a transformer inside each patch](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#6-tnt--nesting-a-transformer-inside-each-patch)
7. [T2T — overlapping tokenisation](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#7-t2t--overlapping-tokenisation)
8. [The ~2% ceiling](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#8-the-2-ceiling)
9. [Open versus closed problems](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#9-open-versus-closed-problems)
10. [Designing an experiment in this space](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#10-designing-an-experiment-in-this-space)
11. [Summary](https://claude.ai/chat/093ea72c-3649-4034-b0ed-d7beef5d6c31#11-summary)

---

## 0. The organising frame: four intervention points

The ViT pipeline:

```
image  →  STEM  →  SEQUENCE z₀  →  ENCODER × L  →  HEAD
          (E)      (CLS+patches)   (MSA,MLP,res)   (read CLS)
```

Each major paper modifies exactly one stage:

| Stage           | Paper                          | What it changes                                   |
| --------------- | ------------------------------ | ------------------------------------------------- |
| **Stem**        | Early Convolutions (Xiao 2021) | replaces patchify with a conv stack               |
| **Sequence**    | Registers (Darcet 2023)        | appends blank tokens                              |
| **Encoder**     | Swin (Liu 2021)                | windows attention, adds relative bias             |
| **Head / loss** | DeiT (Touvron 2021)            | adds a distillation token and teacher supervision |

Two structural observations that emerge immediately from this table:

**Nobody touches the MLP.** Not one of these papers modifies the feed-forward network, despite it holding twice the parameters of attention. Every diagnosed pathology — training instability, data hunger, attention artefacts — lives in the stem, the sequence, or the attention.

**ViT has no mask.** Unlike a language model, there is no causal structure. This turns out to be the direct cause of the artefact that Registers fixes (§4.4).

---

## 1. What ViT §4.5 gave everyone: the probe toolkit

Before the modifications, the measurement tools. All three are from note 7 §8, restated because every paper below uses at least one.

**Probe 1 — visualise $E$.** PCA the patch embedding matrix, reshape components to $16\times16\times3$. Trained ViTs produce Gabor-like, low-frequency basis functions. _Tells you whether the stem learned sensible filters._

**Probe 2 — $E_\text{pos}$ similarity map.** Cosine similarity of position embeddings, plotted per patch over the grid. Trained ViTs recover clean 2D topology. _Tells you whether the model found the spatial structure._

**Probe 3 — mean attention distance.**

$$\bar d^{(\ell,m)} = \frac{1}{N}\sum_{i}\sum_{j} A^{(\ell,m)}_{ij},\lVert p_i - p_j\rVert_2$$

_Tells you how far each head looks, in pixels, comparable to a CNN receptive field._ This is the probe that showed pure ViT grows local heads in early layers while hybrids do not — the single most mechanistically informative result in the original paper.

**Probe 4 (added later, by Registers) — token norm histograms per layer.** _Tells you whether the model is hijacking tokens as scratch space._

---

## 2. DeiT — intervening at the loss

Touvron et al. 2021, _Training data-efficient image transformers & distillation through attention_.

### 2.1 The architectural change is almost nothing

$$z_0 = \left[,x_\text{class},;; x_p^1E,;\dots;, x_p^NE,;; \underbrace{x_\text{dist}}_{\text{new}},\right] + E_\text{pos}$$

One extra learned token appended at the end, giving $N+2$ tokens. MSA, MLP, residual connections, and the stem are all untouched.

### 2.2 How the distillation token works

- It participates in full self-attention with everything, including the class token.
- It has its own output head, supervised by a **teacher network** — a RegNetY, i.e. **a ConvNet**.
- The paper measures $\cos\big(z_L^\text{class}, z_L^\text{dist}\big)$ across depth: approximately **0.06 in the first layer, rising to about 0.93 in the last**. They converge but never coincide.

That measurement is the interesting part. If the two read slots were redundant, similarity would saturate at 1.0 in the middle layers. It does not, so **the residual stream is carrying two genuinely different summaries in parallel through all 12 blocks.**

### 2.3 The central result: hard distillation beats soft

Using $\arg\max$ of the teacher as a hard label outperforms KL divergence against the teacher's soft distribution. (DeiT-B is roughly 81.8% without distillation and roughly 83.4% with hard distillation at 224 resolution; treat these as approximate.)

Combined with the fact that the teacher is a ConvNet, the conclusion is:

$$\boxed{;\text{convolutional inductive bias can be transferred through \textbf{labels}, without touching the \textbf{architecture}};}$$

This is a third independent route for supplying the prior ViT lacks. The first is architectural (conv stem, note 7 §7). The second is positional (relative bias / RoPE, note 8 §5–6). This is the third.

### 2.4 What each augmentation actually does mechanically

| Technique                       | Mechanism                                                                                                                                                                                                                                                                                |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Mixup / CutMix**              | Produces inputs whose labels are mixtures, which forbids winner-take-all attention — the class token must aggregate from several regions. This is a regulariser acting **directly on the attention distribution**, not on weights.                                                       |
| **Stochastic depth (DropPath)** | $z_\ell = z_{\ell-1} + b_\ell,f(\mathrm{LN}(z_{\ell-1}))$ with $b_\ell \sim \mathrm{Bernoulli}(1-p)$ — drops entire residual branches. Keeps the identity path dominant in the residual stream and makes the network an implicit ensemble of shallower networks. Critical for deep ViTs. |
| **Repeated augmentation**       | Multiple augmented views of the same image within one batch; reduces gradient variance with respect to the augmentation distribution.                                                                                                                                                    |
| **RandAugment, random erasing** | Broadens the support of the data distribution, substituting for data volume.                                                                                                                                                                                                             |

### 2.5 DeiT-III, and why it matters more than DeiT

Touvron et al. 2022 revisited this and showed **distillation is not necessary**: LayerScale, a simplified augmentation policy (3-Augment), and binary cross-entropy loss reach higher accuracy than the original DeiT recipe.

The implication is larger than any single number. **Most of the 2020 gap between ViT and ResNet was a training-recipe artefact, not an architectural fact.** Any comparison you run where the recipe differs between arms is measuring the recipe.

---

## 3. Early Convolutions — intervening at the stem

Xiao et al. 2021, _Early Convolutions Help Transformers See Better_. Short paper, directly relevant to any hybrid-stem design.

### 3.1 What changes

Replace `Conv2d(3, 768, k=16, s=16)` with a stack of $3\times3$ stride-2 convolutions:

$$3 \to 64 \to 128 \to 256 \to 512 \xrightarrow{;1\times1;} 768$$

Four stride-2 layers give the same 16× downsampling and the same 196 tokens, but with BN and ReLU between each. **They then remove one Transformer block so that total FLOPs match.** That compute-matching is the methodological point worth copying — it makes the comparison interpretable.

Naming: ViT-P (patchify) vs ViT-C (convolutional stem).

### 3.2 What they measured

1. **ViT-P fails or degrades badly under SGD** and effectively requires AdamW. **ViT-C trains fine with SGD.**
2. Across a grid of learning rate × weight decay, ViT-P's accuracy varies substantially; **ViT-C's is nearly flat.**
3. ViT-C **converges considerably faster** at equal epochs.
4. Top-1 on ImageNet-1k improves by roughly **1–2%**.

### 3.3 Mechanism: measured versus proposed

Be careful to separate these.

**Proposed by the authors (not proven):** the patchify stem is a "non-VGG-style" convolution — huge kernel, huge stride, one layer, no activation — which is unlike every other layer in the network, and this mismatch produces the poor optimisation behaviour.

**A supporting argument consistent with the rest of these notes:** consider the gradient path and the output statistics.

- _Patchify_: one matrix, 590K parameters, receiving gradient from the entire loss through a single layer, producing outputs that are linear combinations of raw pixels — heavy-tailed, wide dynamic range — fed straight into the first block's LayerNorm.
- _Conv stem_: gradient distributed over four layers with BN between each, producing an output distribution that is already well-conditioned.

Stated differently: **the problem may be less "no inductive bias" and more "the first layer has statistical properties unlike the rest of the network."** That is a different failure mode from the one the ViT paper describes, and it is a third axis (alongside feature prior and positional prior) that gets conflated in headline accuracy numbers.

---

## 4. Registers — intervening at the sequence

Darcet et al. 2023, _Vision Transformers Need Registers_. The most mechanistically interesting of the four.

### 4.1 The observed pathology

In trained large ViTs (DINOv2, OpenCLIP, DeiT-III), roughly **2% of patch tokens** have norms an order of magnitude or more above typical. The pattern is highly systematic:

- They appear in **low-information patches** — background regions redundant with their neighbours.
- They appear in **middle layers**, not early ones.
- They appear only in models that are **large enough and trained long enough**. Small or briefly-trained models show nothing.

### 4.2 The probes that establish the mechanism

Linear probes on the high-norm tokens versus normal patch tokens, with opposite results:

| Probe                                         | High-norm tokens | Normal patch tokens |
| --------------------------------------------- | ---------------- | ------------------- |
| Predict own position / reconstruct own pixels | **poor**         | good                |
| Predict the image-level class                 | **better**       | worse               |

$$\Longrightarrow \text{the model \textbf{discards the local content} of those tokens and reuses the space for global information}$$

**The network builds itself a scratchpad, unprompted**, and it does so in the cheapest available locations — patches whose local content was redundant anyway.

### 4.3 Why it must do this: softmax sums to one

From note 7 §2.1:

$$\sum_j A_{ij} = 1 \qquad \forall i$$

**There is no "attend to nothing" option.** A token with nothing to retrieve still has to distribute a full unit of attention mass. So the network appoints a dump target — and having appointed one, that target is also the natural place to broadcast global information from, since everyone is already attending to it.

This is the same **attention sink** phenomenon documented in language models (Xiao et al. 2023, StreamingLLM). It is a structural consequence of softmax normalisation, not a quirk of one modality.

### 4.4 The vision/language asymmetry

In an LLM the causal mask means token 0 is visible to every subsequent token, so the sink reliably lands on position 0 — predictable and harmless.

**ViT has no mask** (note 8 §12). No token is structurally privileged, so the sink lands on arbitrary background patches, in different places in every image. That is why the artefact is messy: attention maps look noisy, and dense prediction tasks that consume patch tokens directly are damaged.

### 4.5 The fix

Append $K$ register tokens (typically 4) to $z_0$, treated exactly like the class token, and **discard them at the output**.

Results: high-norm artefacts disappear, attention maps become clean and interpretable, dense-prediction performance improves, and unsupervised object discovery (LOST) improves substantially.

This is the same architectural idea as "give the model scratch tokens so it can compute before answering" — related to Pause Tokens (Goyal et al. 2023) in NLP, though discovered from the opposite direction: the symptom was found first, then the explanation.

---

## 5. Swin — intervening at the attention structure

Liu et al. 2021, _Swin Transformer_. Covered mainly in note 8 §5; the architectural parts are here.

### 5.1 Three changes

**(a) Windowed attention.** Attention is computed only within non-overlapping $M\times M$ windows ($M=7$, i.e. 49 tokens). Complexity drops from $O(N^2)$ to $O(N \cdot M^2)$ — linear in $N$.

**(b) Shifted windows.** Windowing creates hard boundaries, exactly the patchify problem at a coarser scale. Alternate layers displace the window grid by $\lfloor M/2 \rfloor$, so tokens separated in one layer can communicate in the next. This is mitigation rather than elimination.

**(c) Hierarchical patch merging.** Every stage concatenates $2\times2$ neighbouring tokens and projects, halving spatial resolution and doubling channels — the ViT analogue of a CNN's stride-2 stages. Produces a multi-scale feature pyramid, which is what dense-prediction heads expect.

Plus relative position bias, which is the subject of note 8 §5 and is where the "attention with RPB literally contains convolution" derivation lives.

### 5.2 The important later correction

The initial reading was "hierarchical structure is _required_ for dense prediction." **ViTDet** (Li et al. 2022) refuted this: a plain, non-hierarchical ViT plus a simple feature pyramid built from a single scale competes with Swin on detection, provided pretraining is good enough (they used MAE).

Revised understanding: hierarchy is an **efficiency choice**, not a **requirement**. It still wins at very high input resolutions where the compute argument dominates, but the boundary is not well characterised.

Vocabulary, since it recurs: **"plain ViT"** means the original non-hierarchical column — constant patch size throughout, no downsampling between stages, global attention in every layer.

---

## 6. TNT — nesting a transformer inside each patch

Han et al. 2021, _Transformer in Transformer_. The natural "what if the stem were itself attention?" idea.

### 6.1 Construction

- **Outer**: 16×16 patches → 196 outer tokens $Z_i \in \mathbb{R}^d$, $d = 384$ for TNT-S.
- **Inner**: each 16×16 patch is subdivided into 4×4 sub-patches → $m = 16$ inner tokens $Y_i \in \mathbb{R}^{m\times c}$ with $c = 24$ (deliberately small).
- The inner transformer's weights are **shared across all outer patches**, exactly as $E$ is.
- There is an **inner position embedding**, also shared — necessary because the 16 sub-patches form a sequence and permutation equivariance applies at every scale.

Inner block, run independently within each patch:

$$Y_i \leftarrow Y_i + \mathrm{MSA}(\mathrm{LN}(Y_i)), \qquad Y_i \leftarrow Y_i + \mathrm{MLP}(\mathrm{LN}(Y_i))$$

Then inject into the outer stream:

$$Z_i \leftarrow Z_i + \mathrm{Vec}(Y_i),W + b, \qquad W\in\mathbb{R}^{mc\times d}$$

and the outer block runs normally over all 196 tokens. **Both are interleaved at every layer**, not just once at the beginning.

### 6.2 Cost: why nesting is not quadratic blowup

Flat 4×4 patching would give:

$$N = \frac{224^2}{4^2} = 3136 ;\Longrightarrow; N^2 \approx 9.8\times10^6$$

versus $196^2 = 38{,}416$ — a **256× increase**.

Nested:

$$\underbrace{196^2}_{\text{outer}} + \underbrace{196 \times 16^2}_{\text{inner, 196 groups}} = 38{,}416 + 50{,}176 \approx 8.9\times10^4$$

Roughly **110× cheaper than flat 4×4**, before accounting for the inner width being 24 instead of 384.

### 6.3 The structural insight

$$\text{nesting} = \text{imposing a \textbf{block-diagonal mask} on the fine-grained attention matrix}$$

Full attention over 3136 tokens is a $3136\times3136$ matrix. Nesting says: allow communication only within $16\times16$ diagonal blocks, and delegate cross-block communication to a second transformer operating at coarse resolution.

**This is the same factorisation Swin performs.** The two differ only in how they handle cross-block flow and in which direction they build:

|                        | Swin                    | TNT                          |
| ---------------------- | ----------------------- | ---------------------------- |
| Cross-block mechanism  | shift the window grid   | a separate outer transformer |
| Direction of hierarchy | merge upward to coarser | nest downward to finer       |

### 6.4 Results and why it is not used

| Model     | Params | ImageNet-1k top-1 |
| --------- | ------ | ----------------- |
| DeiT-S    | ~22M   | 79.8%             |
| **TNT-S** | ~24M   | **81.5%**         |
| **TNT-B** | ~66M   | **~82.9%**        |

The idea works. But TNT is essentially unused in practice, for two reasons.

**(a) Poor hardware utilisation.** The inner block runs sequences of length 16 at width 24, in 196 independent groups. These are tiny matrix multiplications with very low arithmetic intensity; the GPU spends its time on overhead. FLOPs on paper are modest but **wall-clock cost is much worse than the FLOP count suggests**. A 4×4 convolution does comparable work with one kernel launch that maps cleanly onto tensor cores.

**(b) Attention buys little at the texture scale.** Attention's advantages over convolution are content-adaptive routing and long-range interaction. Within a 16×16 pixel window:

- long-range: nonexistent by construction,
- content-adaptive: marginal, because edge and texture detection is nearly optimal with fixed filters.

$$\boxed{;\text{attention pays off at the semantic scale, not at the texture scale};}$$

That is the mechanistic reason a conv stem beats a nested transformer in practice despite being "less expressive."

### 6.5 What nesting does not fix

Nesting increases resolution **inside** a patch. It does nothing about the hard 16×16 boundary. An object straddling two patches is still cut in half — the block-diagonal mask explicitly prevents patch 1's inner tokens from ever seeing patch 2's.

$$\text{nesting fixes: intra-patch resolution} \qquad \text{does not fix: inter-patch continuity}$$

A conv stem fixes the second (overlapping receptive fields) without much improving the first. **These are separate problems**, which makes them separately testable — see §10.

---

## 7. T2T — overlapping tokenisation

Yuan et al. 2021, _Tokens-to-Token ViT_. The paper that attacks the boundary problem directly.

### 7.1 Overlap does not increase the token count

A very common misconception is that overlapping patches blow up $N$. They do not:

$$N = \left(\left\lfloor \frac{H + 2p - k}{s}\right\rfloor + 1\right)^2 \approx \left(\frac{H}{s}\right)^2$$

**$k$ appears only in the boundary term. Token count is governed by stride alone.** Compare at equal stride:

|                          | $k$ | $s$ | $N$      | dims per token    |
| ------------------------ | --- | --- | -------- | ----------------- |
| Non-overlapping          | 4   | 4   | **3136** | $3\cdot4^2 = 48$  |
| Overlapping (soft split) | 7   | 4   | **3136** | $3\cdot7^2 = 147$ |

Identical $N$. What grows is the **dimensionality of each token**, by a factor

$$\frac{k^2}{s^2} = \frac{49}{16} \approx 3.06$$

So the cost of overlap is:

- an **im2col buffer inflated by $k^2/s^2$** — pixels are duplicated; this is memory bandwidth,
- a **larger projection matrix**, $\mathbb{R}^{147\times d}$ instead of $\mathbb{R}^{48\times d}$.

$$\boxed{;\text{overlap costs \textbf{channels and memory}, not } N^2;}$$

What makes T2T expensive is that it chooses **stride 4 rather than 16** — a separate decision from overlapping.

### 7.2 The architecture

Three iterations of: reshape tokens back to 2D → soft split (unfold with overlap) → transformer → repeat.

$$224 \xrightarrow[k=7,,s=4]{} 56^2 = 3136 \xrightarrow[k=3,,s=2]{} 28^2 = 784 \xrightarrow[k=3,,s=2]{} 14^2 = 196 ;\to; \text{standard backbone}$$

Tokens overlap at every stage, so boundary pixels are continuously shared and there is no hard boundary anywhere — the thing TNT cannot do.

The bottleneck is the first step: a transformer over 3136 tokens. The paper acknowledges this and mitigates it by shrinking the channel dimension to 64 inside the T2T module and experimenting with **Performer** (linear attention) there.

### 7.3 Results, with a confound

| Model          | Params   | GFLOPs | IN-1k from scratch |
| -------------- | -------- | ------ | ------------------ |
| DeiT-S         | 22M      | 4.6    | 79.8%              |
| **T2T-ViT-14** | 21.5M    | 4.8    | **81.5%**          |
| T2T-ViT-24     | 64M      | 13.8   | 82.3%              |
| T2T-ViT-7      | **4.3M** | 1.1    | 71.7%              |
| ResNet-18      | 11.7M    | 1.8    | 69.8%              |

The small model beating ResNet-18 at a third of the parameters is the most telling row — the benefit is largest exactly in the low-data, low-parameter regime, consistent with everything in note 7 §6.

**⚠️ Confound:** T2T also changes the backbone to a **deep-narrow** design (more layers, smaller hidden dimension, imitating CNN proportions). Their own ablation shows deep-narrow contributes part of the gain. The headline +1.7% is not attributable to overlap alone.

### 7.4 Other members of the family

| Work                | Idea                                                                      | Relation to the above             |
| ------------------- | ------------------------------------------------------------------------- | --------------------------------- |
| **NesT** (2022)     | local attention in blocks, aggregated up a tree                           | nests upward rather than downward |
| **CrossViT** (2021) | two parallel branches at different patch sizes, joined by cross-attention | parallel, not nested              |
| **PVT** (2021)      | pyramid ViT with spatial-reduction attention                              | hierarchical, like Swin           |
| **MViT** (2021)     | multiscale via pooling attention                                          | hierarchical                      |
| **CoAtNet** (2021)  | conv stages then attention stages, explicitly staged                      | the hybrid taken seriously        |
| **LeViT** (2021)    | conv stem + hybrid, optimised for inference speed                         | hybrid, engineering-focused       |

---

## 8. The ~2% ceiling

> **This section is an observation, not an established result.** No paper states it as a claim and there is no theory explaining it. Treat it as a hypothesis that happens to organise the data well.

Line up the reported ImageNet-1k gains from every method that supplies ViT with spatial prior:

| Method                      | Mechanism                           | Gain               |
| --------------------------- | ----------------------------------- | ------------------ |
| Conv stem (Early Convs)     | overlap + nonlinearity in the stem  | +1–2%              |
| TNT                         | intra-patch depth                   | +1.7%              |
| T2T                         | overlap + iterative re-tokenisation | +1.7% (confounded) |
| DeiT hard distillation      | conv prior via labels               | +1–2%              |
| Swin relative position bias | conv prior via attention logits     | comparable range   |

**Every route delivers roughly the same 1.5–2%.**

The natural reading:

$$\text{these are all purchasing the \textbf{same commodity} in different currencies}$$

The commodity is the spatial prior that plain ViT lacks, and it appears to be **bounded**. Stacking the methods does not yield 8%.

If that reading is right, the practical ordering is determined by **price, not cleverness**:

| Method                 | Cost                                                         |
| ---------------------- | ------------------------------------------------------------ |
| **Conv stem**          | FLOP-neutral, maps cleanly to tensor cores, ~5 lines of code |
| Relative position bias | 169 params/head, requires restructuring attention            |
| RoPE                   | zero params, but requires touching every attention layer     |
| TNT                    | poor GPU utilisation, wall-clock expensive                   |
| T2T                    | 3× im2col inflation, 3136 tokens at stage 1                  |
| Distillation           | requires training a teacher first                            |

**Conv stems dominate practice because they are the cheapest, not because they are the most expressive.** That is the honest summary of why, in 2026, everyone uses a conv stem and nobody uses TNT.

---

## 9. Open versus closed problems

State of the field as of mid-2026. Confidence varies by item and is flagged.

### 9.1 Closed — broad consensus exists

**ViT requires JFT-300M.** Dead. MAE, DeiT-III and AugReg established that recipe plus self-supervised objectives substitute for labelled data volume. AugReg found augmentation and regularisation worth roughly a 10× increase in data.

**Conv versus attention: which is better?** The question was malformed. ConvNeXt showed a modernised ResNet matches Swin under the same recipe. Outcomes are determined by scale, data and objective, not by the choice of operator.

**How should positional encoding be done?** Relative schemes (RoPE, relative bias) beat learned absolute. Details remain, the direction does not.

**What should the stem be?** Conv stem or overlapping patchify beat plain patchify. Settled since 2021; uncontested.

**Attention sinks and registers.** Confirmed in both vision and language, with an accepted mechanism (softmax normalisation) and a working fix.

**Is local/hierarchical attention necessary for dense prediction?** No. ViTDet showed plain ViT plus a simple FPN competes with Swin given strong pretraining.

### 9.2 Half-closed

**Hierarchical versus plain.** Plain wins with good pretraining; hierarchical still wins at very high resolution on compute grounds. **Where the crossover sits is not characterised.**

**How necessary is attention at all?** MLP-Mixer, ConvNeXt and Mamba all come close. The capability with no known substitute is in-context learning, which is a sequence-modelling property rather than a vision one. For pure vision, attention may not be essential.

**State-space models versus attention.** Mamba and Vision Mamba achieve $O(N)$ and are competitive, but have not decisively won at large scale. Unresolved.

### 9.3 Open

**(a) Content-adaptive tokenisation.** Everything above uses a **fixed grid**, which is indefensible on information grounds — empty sky receives as many tokens as a face. Attempts exist (TokenLearner, DynamicViT, Perceiver) but none has become standard. The core obstacle is as much engineering as science: variable token counts per image make batching awkward.

**(b) Test-time compute for vision.** In language this has become central (reasoning models). In vision there is **no real equivalent** — a ViT spends identical compute on a trivial image and a hard one. The Registers result says the model _wants_ scratch space; the natural extension is letting it compute iteratively. This is, in my assessment, the widest open space right now.

**(c) Video, 3D, and non-grid data.** Video tokenisation is still crude (cut into tubelets and hope); temporal redundancy is unsolved. Point clouds, meshes and volumetric medical data are more open still.

**(d) Mechanistic interpretability for vision.** Language has circuit analysis, sparse autoencoders and feature-level understanding. Vision lags badly — we know "this head is local" but not what the circuit computing "ears + snout → dog" looks like. The Registers paper is one of very few genuinely mechanistic vision results. **This is also the area most accessible without a GPU cluster**, since it operates on pretrained models.

**(e) Efficient attention that actually wins at scale.** Hundreds of linear- and sparse-attention papers exist; almost none are used in production, because at scale they lose to full attention plus FlashAttention. Still open.

**(f) Why the ~2% ceiling exists.** §8. No theory. (Lowest confidence item here — this is my framing, not a recognised open problem.)

### 9.4 Where the room is, for a single person with limited compute

| Area                         | Compute needed                  | Room available              |
| ---------------------------- | ------------------------------- | --------------------------- |
| Mechanistic interpretability | **low** (use pretrained models) | large                       |
| Adaptive tokenisation        | medium                          | large                       |
| Test-time compute for vision | medium                          | **very large**              |
| New architectures            | **very high**                   | small — competing with labs |
| Efficient attention          | high                            | small                       |

---

## 10. Designing an experiment in this space

### 10.1 Frame the question as diagnosis, not architecture search

"Does a hybrid stem help?" is **closed** — yes, by roughly 2%, known since 2021. "**Why** does it help?" is open. Frame accordingly; the second question is answerable with modest compute and the first is not worth answering again.

### 10.2 The three axes a conv stem changes simultaneously

Papers routinely report the sum of these as one number:

1. **Feature prior** — nonlinearity and overlapping receptive fields.
2. **Positional prior** — leaked through zero padding (note 8 §7.4, §13.1); obtainable without convolution via RPB, RoPE or CPE.
3. **Optimisation stability** — smoother loss landscape (§3).

### 10.3 Designs that separate them

**Separating axis 1 from axis 2 — a $2\times2$:**

`{plain ViT, conv stem} × {absolute PE, RoPE}`

If RoPE closes most of the gap for plain ViT, the conv stem's benefit was largely positional.

**Measuring axis 2 directly — remove PE from both arms:**

- Plain ViT should collapse (bag of patches; provable, so it doubles as a pipeline sanity check).
- The hybrid should survive substantially, because padding supplies position.

$$\Delta = \mathrm{acc}_{\text{hybrid, no-PE}} - \mathrm{acc}_{\text{plain, no-PE}}$$

is a direct numerical estimate of leaked positional prior. Two extra runs.

**Isolating axis 3 — a learning-rate × weight-decay sweep.** Accuracy at a single tuned LR conflates "generalises better" with "was easier to optimise." Following Early Convs, sweep and report the spread, not just the maximum.

**Separating intra-patch depth from inter-patch overlap** — a design not, to my knowledge, run compute-matched anywhere:

| Arm                   | Intra-patch depth        | Inter-patch overlap | Stem nonlinearity |
| --------------------- | ------------------------ | ------------------- | ----------------- |
| Plain ViT             | shallow (one linear map) | none                | none              |
| Conv stem             | deep                     | **yes**             | yes               |
| **Nested (TNT-like)** | **deep**                 | none                | yes               |

- If nested ≈ conv stem → the benefit is mostly **depth of in-window processing**.
- If nested < conv stem clearly → the benefit is mostly **overlapping receptive fields**.

### 10.4 Instrumentation: record probes, not just accuracy

For every run:

1. **Mean attention distance per layer per head.** If the hybrid's advantage is really a local prior, plain ViT's early local heads should be absent in the hybrid (reproducing note 7 §8.3).
2. **LR × WD sweep spread.** Separates generalisation from optimisation.
3. **Token norm histogram per layer.** Checks whether attention-sink artefacts appear at your scale. If they do in one arm and not another, the comparison is not fair until registers are added to both.
4. **Cosine similarity between read slots**, if using more than one.

### 10.5 Noise floor — check this before running anything

If the ceiling is ~2%, differences between hybrid variants will be **0.3–0.8%**.

- Seed variance on full ImageNet-1k: roughly ±0.1–0.2%. Resolvable with ≥3 seeds.
- Seed variance on small datasets (CIFAR, IN-100, subsets): roughly ±0.5–1.5%. **The effect drowns.**

$$\text{One seed on a small dataset produces a number that means essentially nothing.}$$

**Run the same baseline 3 times first and measure your own $\sigma$.** Then decide whether the effect you are hunting sits above your noise floor. If it does not, add seeds or move to a larger dataset before running the real comparison — otherwise conclusions can invert on a reseed.

### 10.6 Where a novel result is most likely

The ~2% ceiling was measured on ImageNet-1k with natural images. It has not been established that it holds:

- at **very small data scales** (< 50k images),
- on **non-natural domains** — medical imaging, satellite, spectrograms, microscopy, where the relevant spatial statistics are entirely different.

Running the same arena in a regime nobody has measured is a more promising route to something new than optimising in the regime everyone has.

---

## 11. Summary

1. The four major post-ViT papers each intervene at a **different stage** of the same pipeline: stem (Early Convs), sequence (Registers), attention (Swin), loss (DeiT).
2. **Nobody modifies the MLP**, despite it holding most of the parameters. Every diagnosed pathology lives elsewhere.
3. **DeiT** shows convolutional prior can be transferred through **labels**; DeiT-III then shows distillation was not even necessary — most of the 2020 gap was recipe.
4. **Early Convolutions** shows part of the stem's benefit is **optimisation stability**, not generalisation: SGD becomes viable and hyperparameter sensitivity collapses.
5. **Registers** shows ViT builds itself a scratchpad by hijacking low-information tokens, because **softmax has no "attend to nothing" option**. ViT's lack of a causal mask is why the sink lands in messy places.
6. **Swin**'s relative bias is a convolution inside the softmax; its hierarchy is an efficiency choice, not a requirement (ViTDet).
7. **TNT** = block-diagonal masking of fine attention; works (+1.7%) but loses on hardware utilisation and because attention adds little at texture scale.
8. **T2T** = overlapping tokenisation; overlap costs **channels and memory, not $N^2$**. Fixes the boundary problem TNT cannot.
9. All spatial-prior methods deliver **roughly the same ~2%**, so the practical winner is decided by **cost**. Conv stems win on price.
10. Most architectural questions are **closed**. What is open: adaptive tokenisation, test-time compute for vision, non-grid data, and mechanistic interpretability — the last two being the most accessible without a cluster.

### Reading list

| Paper                                                                                        | Why                                                              |
| -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Touvron et al. 2021, _DeiT_                                                                  | Distillation token; the augmentation recipe.                     |
| Touvron et al. 2022, _DeiT III_                                                              | Shows distillation was not the essential ingredient.             |
| Xiao et al. 2021, _Early Convolutions Help Transformers See Better_                          | Short, and the compute-matched methodology is worth copying.     |
| Darcet et al. 2023, _Vision Transformers Need Registers_                                     | The best mechanistic result in vision ViT work.                  |
| Xiao et al. 2023, _StreamingLLM_                                                             | The same attention-sink phenomenon in language.                  |
| Liu et al. 2021, _Swin Transformer_                                                          | Windows, shifts, hierarchy, relative bias.                       |
| Li et al. 2022, _Exploring Plain Vision Transformer Backbones for Object Detection_ (ViTDet) | Hierarchy is optional.                                           |
| Han et al. 2021, _Transformer in Transformer_                                                | Nested tokenisation.                                             |
| Yuan et al. 2021, _Tokens-to-Token ViT_                                                      | Overlapping tokenisation.                                        |
| He et al. 2021, _Masked Autoencoders Are Scalable Vision Learners_                           | The real answer to "not enough data".                            |
| Liu et al. 2022, _A ConvNet for the 2020s_ (ConvNeXt)                                        | The strongest argument that the operator was never the variable. |
| Steiner et al. 2021, _How to train your ViT_ (AugReg)                                        | Quantifies augmentation as ~10× data.                            |
