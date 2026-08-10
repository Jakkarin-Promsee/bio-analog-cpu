# INDEX — the vault

> **What this file is.** The sync layer over the 15 study notes in this folder. It answers one question for
> anyone (human or agent) picking up this work: **how far did he actually get on each topic, what does he now
> own, and where does his knowledge stop.**
>
> **What it is not.** Not a summary of the subject matter — the notes do that far better, and they are argument-
> shaped, not encyclopaedic. This is a *depth map*. Read it to know what can be assumed in conversation and what
> still has to be built.
>
> **Corpus:** 15 files, ~770 KB, written over the three days to 2026-08-10. Two of the 15
> ([[s1-opened-topic-ideas]], [[s2-opened-topic-ideas]]) are not study notes at all — they are **positions**.
>
> **⚑ On 2026-08-10 this vault became draft 7.0's direction.** The live position —
> [`../../docs/direction.md`](../../docs/direction.md) — is built out of `s1` idea 1 and `s2` idea 1, and §8 below
> maps every part of it back to the note it came from. The previous direction is archived.
>
> **Companion:** [`README.md`](README.md) holds the author's own one-line self-assessment of each file. Where his
> read and this index's read differ, both are recorded below. His is the one that counts.

---

## 0 — The depth scale

Every rating below is assigned from evidence *inside the file*, not from topic difficulty. A note can be L4 on one
axis and L2 on another; where that happens it is split.

| | Level | What it means | What proves it in a note |
| --- | --- | --- | --- |
| **L1** | Vocabulary | Names the thing, quotes results | Correct terms, no mechanism |
| **L2** | Mechanism | Can explain how it works in plain words | Formulas quoted and applied, not built |
| **L3** | Derivation | Could re-derive it on a blank page | Symbols defined at first use, every step justified |
| **L4** | Critique | Knows the failure modes and what the field has *not* settled | Kills wrong pictures, honest-assessment sections, flags confounds |
| **L5** | Original position | Generates new mechanisms/experiments from the material | Proposes, costs, and risk-analyses something nobody ran |

**The vault's centre of mass is L3–L4.** Two files reach L5 in sections ([[9-vision-tranformer-landcape]] §10,
[[11-perciever-and-more]] §14–15), one is L5 throughout ([[13-latent-space-and-shortcut]]), and both idea files are
L5 by construction.

---

## 1 — The map in one screen

| # | File | What it settles | Depth | Status of the topic, in his hands |
| --- | --- | --- | --- | --- |
| 1 | [[1-transformer]] | The whole architecture as a chain of fixes, ending at CoT/latent-CoT complexity | **L3** core · L2 breadth · L4 critique | Owned. Breadth items (Adam, MoE, RLHF, FlashAttention) are read-not-derived |
| 2 | [[2-hopfield]] | 1982 → Dense AM → Ramsauer 2020, and *where the intelligence actually sits* | **L3** derivation · **L4** critique | Owned, including the honest verdict that the 2020 paper is connective, not field-changing |
| 3 | [[3-transformer-internal]] | Gauge freedom of W_Q/W_K, and the causal order in which capability assembles | **L3–L4** | Owned. Phase 4 of training explicitly conceded as a black box |
| 4 | [[4-hopfield-internal]] | The four-variant axis, the fixed-point reframe, softmax-vs-ReLU as budget-vs-no-budget | **L4** (+L3 retained, L5-adjacent §6.6) | Owned. The strongest single file on retrieval mechanics |
| 5 | [[5-interp-hull-and-superposition]] | Where the model can go (hull) × how much it can hold (superposition) | **L3** derivation · **L4** | Owned to the Welch bound. Interp-field limits explicitly listed as the *field's* gaps |
| 6 | [[6-fixed-point-and-what-transformer-chasing]] | Why a Transformer structurally cannot use energy descent to stop | **L4** · L3 native · L2 imported | Owned as an argument chain. Seven named threads left unpulled |
| 7 | [[7-vit-foundation]] | Patch embedding *is* a convolution; inductive bias = findability, not representability | **L3** proofs · L4 · L2 config | Owned |
| 8 | [[8-position-transformer]] | Position as slot-vs-value; the four-term decomposition; ViT→Swin→RoPE as one arc | **L3** core proofs · **L4** | Owned |
| 9 | [[9-vision-tranformer-landcape]] | Post-2020 ViT as four intervention points; the ~2% ceiling; open-vs-closed map | **L4** · L2 per-paper · **L5** §10 | Owned as a *field map*, self-flagged as a mid-2026 judgement call |
| 10 | [[10-attention-collapse-and-field-equilibrium]] | What a block does to a token; why equilibrium-halting is meaningless and what works instead | **L4** throughout · L3 places · L2 §14–16 | Owned. Issues a correction to note 6 (see §3.10) |
| 11 | [[11-perciever-and-more]] | Attention as a priced resource; the full latent-bottleneck economics | **L3** mech+cost · L3→**L4** own derivation · **L5** §14–15 | Owned, with original unpublished analysis in §8 |
| 12 | [[12-recurrent-visual-attention]] | Glimpse attention end to end: POMDP → REINFORCE → STN/DRAW/Gumbel → why it died | **L3** derivation · **L4** critique | Owned. The autopsy is the contribution |
| 13 | [[13-latent-space-and-shortcut]] | What latent space mechanically is, and why it collapses to a placeholder in vision | **L5** · L4 · L2 formalism | Owned *and extended* — proposes a fix, a protocol, and a threat model |
| s1 | [[s1-opened-topic-ideas]] | Position after note 11: object-bound dynamic latent, query-conditioned compaction | **L5** | Live |
| s2 | [[s2-opened-topic-ideas]] | Position after note 13: **bounded-bandwidth visual reasoning** | **L5** | **Live — this is where the vault points** |

---

## 2 — The arc: how the 15 files actually connect

Six lineages, two convergence points. Nothing here is a reading list in file-number order — the number is
chronology, not dependency.

```
  A. TRANSFORMER            1 ──────────────▶ 3
                            │                 │
  B. HOPFIELD               2 ──────────────▶ 4
                            │                 │
                            ▼                 ▼
  C. GEOMETRY & DYNAMICS    5 ──▶ 6 ──────▶ 10        (10 corrects 6, §4.3)
                                  │           │
  D. VISION                 7 ──▶ 8 ──▶ 9     │
                            │                 │
                            ▼                 ▼
  E. ECONOMICS OF READING   11 ─────────────────────▶ s1   ⟵ CONVERGENCE 1
                            │                          │
  F. BOUNDED BANDWIDTH      12 ──▶ 13 ──────────────▶ s2   ⟵ CONVERGENCE 2
                                                       ▲
                            s1 ideas 1 & 2 merge into s2 ideas 1 & 2
```

**The through-line, in one sentence:** *what does a system do when it cannot afford to look at everything* —
approached first as memory geometry (2/4/5/6/10), then as read cost (11), then as gaze control (12), then as the
question of whether an internal state can carry anything at all (13).

**Convergence 1 (s1, after note 11):** the image-side problem is **unit discovery, not compaction** — and the
solve order is forced: **binding → budgeting → eviction**.

**Convergence 2 (s2, after note 13):** the problem itself gets rewritten — from *"how well does a model understand
a whole image"* to ***"how well does it understand when it must choose what to look at"***. Everyone else's work
becomes the special case `attention size = N`.

---

## 3 — Per-file dossiers

Each entry: what the note settles · depth evidence · the claims he now holds · the misconceptions it destroys (the
best available proof of depth in this vault, because every note is *built* around killing them) · what the note
itself says is still missing.

---

### 3.1 · [[1-transformer]] — 98 KB

**Settles:** the whole architecture, treated as a chain where every component is the specific fix for the previous
component's specific failure — ending in a complexity-theoretic account of why CoT and latent CoT work.

**Depth: L3 core / L2 breadth / L4 critique.** L3 because `√d_k` is derived from a variance calculation, softmax
saturation is proven dead-gradient via the Jacobian `∂p_i/∂s_j = p_i(δ_ij − p_j)`, RoPE's relative-position
invariance is derived, and rank collapse is walked as a Markov contraction. L2 on the breadth material — Adam,
LoRA, FlashAttention's online softmax, MoE routing, RLHF/DPO, quantization, PagedAttention are correct but quoted.

**Author's own read (README):** *"I already understand all part deeply. Now lacking only deep insight how weight
activation inside work."* — consistent with the evidence.

**Holds:**
- Attention buys `O(1)` path length and parallelism by paying `O(n²)`; that trade is the entire reason the
  architecture exists.
- Attention output is strictly a convex combination — it blends, never creates. Repeated pure attention provably
  contracts to rank-1 within ~5 layers: `‖X^(L) − 1xᵀ‖ ≤ (cγ/√d)^((3^L−1)/2)` (Dong et al. 2021). This one fact is
  why FFN, residual and normalization are structurally mandatory.
- The residual stream is an additive bus, so **depth = number of sequential reasoning hops in one forward pass**.
  Induction heads provably need two layers minimum.
- CoT changes nothing architecturally — it converts a fixed-depth circuit into a recurrent machine using the
  context window as an external tape. Formally: `O(1)` chain → TC⁰, `O(log n)` → ~L, `poly(n)` → exactly P
  (Merrill & Sabharwal 2024).
- Parameter arithmetic: per-block `12d²`, model `≈ 12Ld² + |V|d`, checked against GPT-3 = 174B.
  KV cache `2·L·n·d·bytes` → 21.5 GB for 70B at 8192 — argued as the real cost driver, not FLOPs.
- **Latent CoT loses on supervision, not mechanism:** ordinary CoT gets per-token teacher signal for free (humans
  write reasoning) and is inspectable; latent state has neither.

**Kills:** ❌ attention is exact-match lookup → ✅ soft differentiable weighted average · ❌ `d_k` = number of
distinct key meanings → ✅ per-head subspace width, a rank knob (explicit self-correction) · ❌ long-context failure
is "distance is a fixed value" → ✅ unseen rotation angles, KV memory, `O(n²d)` · ❌ α passes between blocks → ✅ only
`z` flows; α is born and dies inside the block · ❌ more heads substitute for depth → ✅ heads in a layer are blind
to each other · ❌ weight sharing is a free win → ✅ same compute bill, dumber model · ❌ "Transformers can't do
parity" → ✅ parity *is* in TC⁰; AC⁰ can't — the casual claim conflates this with Hahn's narrower result ·
❌ CoT works because of the semantic content → ✅ Pfau et al.: filler tokens also help · ❌ written CoT is a
faithful log → ✅ Turpin et al.: post-hoc rationalization · ❌ emergent abilities are real discontinuities →
✅ often metric artifacts (Schaeffer et al.) — flagged as still debated.

**Own gaps (explicit list):** backprop through attention by hand · mechanistic interpretability (superposition,
SAEs) · MoE routing depth · whether recurrent-depth models can find a supervision signal for intermediate states ·
what replaces internet-scale data · whether CoT faithfulness can be trained for. Also quotes Shazeer conceding
nobody knows why SwiGLU works.

**Read first:** the **Latent Chain of Thought** section — the closest prior art in this file to a loop with a
self-generated halt, with an honest diagnosis of why it loses. Runner-up: **Multi Block** (depth-as-hops).

---

### 3.2 · [[2-hopfield]] — 60 KB

**Settles:** 1982 → Dense AM (2016) → Demircigil (2017) → Ramsauer (2020), the CCCP derivation of the softmax
update, the proof that it *is* attention — and then the real payload: **where the intelligence actually sits.**

**Depth: L3 derivation + L4 critique.** CCCP split, lse convexity via the softmax Jacobian, the `E ≥ 0` bound and
the descent inequality are all built step-by-step (§6.2–6.3). §17 "Honest assessment" separates *explains* from
*equals*, calls the capacity bound loose, and rates the 2020 paper "connective work, not field-changing work."

**Author's own read (README):** *"I know about its concept now, didn't deep drive into each formular yet."* —
**he under-rates this file.** The derivations are present and complete; what he means is he has not re-proved them
unaided. Treat the material as owned.

**Holds:**
- A Hopfield network is a dynamical system, not a pipeline (§0) — the category error the whole file exists to
  prevent. Classical Hopfield has **zero training**: wells are *written* (Hebbian outer products), not dug.
- Capacity: `N_max ≈ 0.14d` classical → exponential `C ≳ √p · c^((d−1)/4)` in 2020. Every variant in between is a
  response to sharpening the interaction function `F` to kill crosstalk.
- `E(ξ) = −lse(β, Xᵀξ) + ½‖ξ‖² + β⁻¹log N + ½M²`, CCCP-derived update `ξ^{t+1} = X softmax(β Xᵀ ξ^t)` — literally
  attention with `β ↔ 1/√d_k`. **Softmax was hiding inside lse all along; nobody chose it.**
- Three fixed-point regimes gated by β: global average / metastable subgroup / single pattern (Theorem C) — the
  note flags this table as memorize-this.
- One-step error `ε ~ N e^(−βΔᵢ)`, and the **blessing of dimensionality**: higher `d` makes retrieval *easier*
  because `Δᵢ ~ d` while `N` only enters linearly.
- **§9–10, the payload:** Hopfield pooling does not know what is relevant — it makes relevance *expressible* as a
  gradient-bearing parameter ξ, so it acts as a **gradient router** that teaches the upstream encoder to
  discriminate. Mean pooling is *structurally* incapable, not merely weaker: it has no parameter that can represent
  "attend to instance j". SNR gap `√(N/k)` = **173×** at DeepRC's N=300,000, k=10.

**Kills:** ❌ the landscape is a store you write into → ✅ `E` is a function of `(X, ξ, β)` only; a landscape lives
exactly one forward pass · ❌ Hopfield is a model you train → ✅ a layer with almost no parameters of its own ·
❌ signal accumulates / noise cancels is why it works → ✅ that's generic to all SGD; the real reason is
expressivity · ❌ attention knows about sequences → ✅ provably permutation-equivariant; order is injected into
pattern *content* · ❌ `O(1)` retrieval means free → ✅ iteration count only; per-iteration is still `O(Nd)` ·
❌ deep Transformers should converge to an answer across layers → ✅ **a fixed point in a Transformer is rank
collapse — death, not success** · ❌ residual connections restore rank → ✅ proven they cannot; residual prevents
collapse from starting, cannot cure it.

**Own gaps (§17, near-verbatim):** *explains vs equals* — sharing the form of a Hopfield update does not prove
Transformers operate on associative-memory principles; could be "a beautiful mathematical coincidence with limited
predictive power." · the capacity bound is worst-case on random spherical patterns, nothing like real embeddings ·
"nobody changed how large Transformers are built."

**Read first:** **§10.5** — why sparse signal survives, and the amplification trap. It establishes that a
gradient-bearing query parameter is what makes *selection* representable at all, that the resulting loop
self-amplifies its own SNR (bistable, phase-transition-like), and that **the same loop indiscriminately amplifies
spurious correlations** — a built-in confirmation bias any candidate-rejection loop would inherit.

---

### 3.3 · [[3-transformer-internal]] — 20 KB

**Settles:** that `W_Q, W_K, W_V, W_O` individually do not exist (full `GL(d_k)` gauge freedom), and the causal,
non-simultaneous order in which capability assembles during training.

**Depth: L3–L4, tight and single-threaded.** The gauge proof is built from scratch: insert invertible `R`, show
`(RW_Q)ᵀ(R⁻ᵀW_K) = W_Qᵀ W_K` unchanged. The K1→K2→V collapse is derived from `rank(ABC) ≤ min(...)`, not asserted.

**Holds:**
- Only `W_QK = W_Qᵀ W_K` and `W_OV = W_O W_V` are invariant. *"Four matrices is an illusion. There are two objects."*
  Interpreting `W_Q` alone is examining a gauge.
- At init there is no chaos, only **undifferentiated sameness** — every head mean-pools identically. Specialization
  is symmetry breaking: a microscopic accident amplified by winner-take-all. Head roles are **not** predetermined.
- **Gradient descent learns what is most *consistent across examples* first, not what is simplest.** This produces
  a strict causal order: unigrams (`L: 10.8→6.5`) → bigrams (`→5.0`) → induction bump (`→4.2`) → long refinement
  (`→3.3`).
- **The scaffolding principle:** complex circuits cannot be learned incrementally because *half a circuit earns
  zero gradient in both directions*. Induction heads form only because a previous-token head already existed —
  built for an unrelated reason (bigrams) — and becomes a ladder.
- Grokking = continuous hidden circuit growth suddenly revealed when weight decay strips the memorizing shortcut.
  Operational discriminator: train accuracy already saturated ⇒ grokking; not yet ⇒ induction bump.
- K1→K2→V inside one head collapses under linear composition; the place that idea works is *across layers with a
  nonlinearity between* — which is exactly what an induction head is. MLA (DeepSeek) validates it as memory
  compression (~93% smaller KV cache), not a capability win.

**Own gaps (direct quotes):** Phase 4 *"remains largely a black box. We know FFNs store facts and that features
live in superposition, but nobody has traced its formation order the way induction heads were traced."* · the phase
numbers are for a small model — *"treat this as a useful frame, not a clean staircase."*

**Read first:** **§3, Phase 3** — the scaffolding argument. Generalized, it is the most transferable principle in
the vault for anything that has to bootstrap a signal from nothing: such signals do not get learned directly, they
ride on some other individually-useful component built first.

---

### 3.4 · [[4-hopfield-internal]] — 73 KB

**Settles:** the same derivation re-entered from where the confusion actually clears, plus everything note 2 left:
the four-variant taxonomy, the fixed-point reframe, "N queries one landscape", and softmax-vs-ReLU as
budget-vs-no-budget.

**Depth: L4 across the board, L3 retained, L5-adjacent in §6.6** — where he runs a full risk analysis on a
candidate intervention ("raise β in late layers"), rejects it with four independent counterarguments, and still
specifies a falsifiable protocol (per-head β, annealing, measure via entropy) for anyone who wants to try it. That
is generating-and-stress-testing, not recall. Appendix A holds 20 explicit trap-kills — the densest L4 concentration
in the vault.

**Author's own read (README):** *"Still need more time to understand energy or jacobain formular."* The **Jacobian
at the fixed point is the single most-repeated unresolved item in this vault** — flagged again in notes 6 and 10.

**Holds:**
- 1982's real limitation is not "no sharp nonlinearity" (sgn is sharper than any softmax) — it is that the sharp
  decision fires **per-neuron, after crosstalk is already summed in**, and everything is superimposed into one
  `d×d` matrix that cannot be pulled apart.
- **The four-variant axis (§7): every variant is the same equation `y = X_V softmax(β X_Kᵀ q)`; the only design
  choice is which slot is data and which is a parameter** — and that determines what survives across forward passes:

  | Variant | Q | K,V | What survives |
  | --- | --- | --- | --- |
  | `Hopfield` | data | data | nothing — landscape rebuilt every pass |
  | `HopfieldPooling` | **parameter** | data | **the question** |
  | `HopfieldLayer` | data | **parameter** | **the wells** (and they learn) |
  | classical 1982 | data | fixed | the wells, but they never learn |

- **The fixed point is a self-consistency condition** `ξ* = Σᵢ aᵢ(ξ*) xᵢ` — *"the question I asked equals the answer
  I got back"* — not flatness. **And it is Hopfield's goal but the Transformer's failure mode.** Identical
  mechanism, opposite destination (the "convergence inversion", §13.6).
- **Rank collapse is caused by β too LOW**, not too high — the intuitive direction is backwards. Low β → average-of-
  everything fixed point; high β → near-permutation matrix, full rank.
- **§6.7, "N queries, one landscape":** X is a set. There is exactly *one* landscape per sequence/head/layer, with
  N marbles dropped in simultaneously — not N landscapes. **No trained parameter has N in its shape**, which is the
  entire reason variable-length input works for free. What locks input length is (a) weights whose shape depends on
  N and (b) positional encoding — a set has neither. β ~ log N, so 300k→400k patterns costs ~2%; only
  order-of-magnitude jumps cost anything.
- Softmax vs ReLU is **budget vs no budget** (`Σa = 1` vs unconstrained). Jacobians: `∇softmax = β(diag(a) − aaᵀ)`
  (entangled, off-diagonal) vs `∇ReLU = diag(1[s>0])` (fully diagonal). This is the mechanical root of "softmax
  picks one, ReLU accumulates many". **`HopfieldLayer` + ReLU = `W₂ ReLU(W₁q)` = the FFN, exactly** — an identity,
  not an analogy. Softmax is the whole reason the layer is called Hopfield.

**Kills (selection from 20):** ❌ each X has its own learned equation → ✅ one hardcoded equation, many mountains,
like gravity · ❌ β blends landscapes → ✅ there is only one; β blends the wells inside it · ❌ it's a giant hashmap
→ ✅ exactly as big as X, and it returns *blends* at low β · ❌ trained at N=300k so it breaks at 400k → ✅ no trained
parameter has N in its shape; what breaks LLM context (RoPE out-of-distribution) does not exist for a set ·
❌ the loop index t is position in the sequence → ✅ the loop re-asks the same landscape · ❌ we choose where to put
the wells → ✅ nobody places wells; gradient pushes `W_K`/encoder until separation `Δᵢ` is right — *"Hopfield
supplies the geometry; learning supplies the coordinates"* · ❌ a Transformer descends toward an answer across
layers → ✅ no energy function exists for the stack; layer 5 is not "closer" to a minimum than layer 2 ·
❌ getting layers closer to slope 0 would be an improvement → ✅ *"optimising toward it is optimising toward nothing."*

**Own gaps (§14):** same *explains vs equals* caveat · capacity bound loose · **`E` is never computed in real code**
— a Transformer only runs `softmax(QKᵀ/√d)V`; the energy landscape is a lens for seeing, not an object that gets
processed · §13 concedes that outside the Hopfield frame nobody knows why SwiGLU wins ("purely empirical"), flagged
as a meta-point about two modes of producing knowledge in this field.

**Read first:** **§6.7** — the precise object model (one landscape per forward pass, N non-interacting marbles, no
parameter carrying N) that any loop reusing attention as its "call candidates from memory" step must get right.

---

### 3.5 · [[5-interp-hull-and-superposition]] — 36 KB

**Settles:** two separate axes that get conflated constantly — **where can it go** (convex hull) and **how much can
it hold** (superposition) — and shows both converge on the same crosstalk statistic `√(S/d)` that Hopfield 1982
already had.

**Depth: L3 derivation + L4 critique.** The Welch bound is built from Cauchy–Schwarz on Gram eigenvalues (§11.2),
contraction is proven via the Dobrushin coefficient (§2.2–2.3), and orthogonality-from-gradient-descent is derived
from a toy autoencoder loss (§12).

**Holds:**
- Attention output is provably inside `hull(V)` — **softmax *is* the convexity condition** — and hulls only shrink,
  doubly exponentially. FFN is the only escape, by ReLU-folding a convex set across a hyperplane.
- **Nothing is "stored" in the residual stream.** A feature is a *contract between a writer* (a `W₂` column) *and a
  reader* (a `W₁` row). The stream is a shared bus with no privileged basis.
- Orthogonality means **mutual invisibility between dictionary entries** — independent routing — not "angle to the
  current state." Two different dot products, never to be conflated.
- Capacity `N(d,ε) ≳ max(d, e^(dε²/4))`, floored below by the **Welch bound** `μ ≥ √((n−d)/(d(n−1))) → 1/√d`. At
  d=768 the crossover is `ε ≈ 0.036`. `n > d ⇒ G ≠ I` is pure algebra — no probability needed.
- **Gradient descent enforces near-orthogonality with no explicit regularizer** — it falls straight out of
  reconstruction loss on sparse data: `L = m₂ Σ_{i≠j} G_ij²`.
- Crosstalk (1982) and interference (superposition) are the literal same quantity: **`√(S/d)` — the one number.**

**Kills:** ❌ stacking attention grows the hull → ✅ monotonically shrinking; abstraction is vertices moving ·
❌ meaning collapse is caused by FFN → ✅ caused by attention; FFN prevents it · ❌ latent CoT has no cleanup →
✅ FFN does soft cleanup every layer; only *hard* argmax-style cleanup is missing · ❌ orthogonality is about angles
→ ✅ mutual invisibility · ❌ reader vector = writer vector → ✅ the optimal reader is the *dual*; equality only
holds near-orthogonally, and the dual doesn't exist when `n > d` · ❌ `1/√d` is just what random vectors happen to
give → ✅ it is the provable Welch minimum, which random init happens to achieve · ❌ interference is a problem to
fix → ✅ it is the computational point (AND-gates), fine while `S ≪ d` · ❌ polysemantic neurons are novel →
✅ superposition seen through a privileged basis; the residual stream has it worse and unnamed ·
❌ `HopfieldLayer` can't extrapolate because attention can't → ✅ specifically because it lacks an FFN.

**Own gaps — framed as the *field's* gaps, not his:** how much superposition is in real models (unmeasured) · do
SAEs find real features or features they were forced to find (actively contested) · the linear representation
hypothesis (increasingly questioned) · what FFN actually injects (mechanism understood, content barely) ·
attention-head superposition (unknown) · LayerNorm's representational role · whether any of it survives
SwiGLU/MoE (derived for plain ReLU MLPs) · *"Interpretability is roughly six years old. Nobody has the full
picture."* Plus his own caveat: real transformers have no reconstruction loss, so toy models are *"a microscope for
the mechanism, not a photograph of the real thing."*

**Read first:** §11–12 (Welch bound + why GD enforces orthogonality) — the only hard, provable capacity ceiling in
the vault. The note itself nominates **§20, `√(S/d)`**, if only one line survives.

---

### 3.6 · [[6-fixed-point-and-what-transformer-chasing]] — 35 KB

**Settles:** in one continuous chain, that a Transformer **structurally cannot** use Hopfield-style energy descent
as its stopping mechanism — and that autoregression/CoT is exactly the trick of borrowing the halting loop from
*outside* the stack.

**Depth: L4 critique · L3 native derivation · L2 imported formalism.** The critique is structural, not empirical:
§5 gives three necessary conditions for a fixed point to exist and shows Transformers violate one; §9 gives four
independent reasons iterating attention cannot work.

**Holds:**
- **Capacity is never about writing** (always succeeds) — only about whether reading lands where intended.
  `ε ~ N e^(−βΔᵢ)`: N linear outside, `Δᵢ` exponential inside, so over-capacity mostly means *too similar*, not
  *too many*. Nothing ever overwrites.
- Self-attention = N Hopfield-poolings in parallel: **N marbles on one shared per-sequence landscape.** K/V
  decoupling separates *findable* from *returned*.
- **Termination = the update rule *plus* the existence of `E`.** `E` decreases every step (CCCP) and is bounded
  below (the `½‖ξ‖²` term is the only reason it has a floor) ⟹ it must settle. Three conditions are required:
  same-space I/O, **unchanging rule**, symmetry. **Transformers violate condition 2** — every layer has different
  `W_Q,W_K,W_V`, so there is no single `f` to have a fixed point of at all.
- **A fixed point is self-consistency, not correctness.** Spurious states are fixed points too. `E` is checkable at
  inference; loss is not.
- Hopfield training never touches `E` — gradient only raises separation `Δᵢ`. Geometry fixed, coordinates learned.
- Iterating attention to convergence fails four independent ways (soft → collapse; sharp → gradient dies via
  `∇softmax = β(diag(a) − aaᵀ) → 0`; FFN can't undo it since `zᵢ = zⱼ ⇒ FFN(zᵢ) = FFN(zⱼ)`; and it would mean
  nothing even if reached) **unless input injection is added — and the residual is exactly one dose of that** (DEQ).
- **The closing asymmetry:** Hopfield's answer space (hull of X) is closed and given in advance; language is
  open-ended. So a Transformer borrows the halting loop from outside via autoregression = CoT = **trading depth for
  length.**

**Kills (selection):** ❌ capacity is a container that fills → ✅ writing always succeeds, reading degrades ·
❌ 32 dims means 32 wells → ✅ different axes · ❌ multi-head = multiple landscapes → ✅ multiple marbles, one
landscape · ❌ blended output means it stopped too early → ✅ at that β the well floor *is* the blend ·
❌ `E` measures correctness → ✅ resemblance to X · ❌ `E = 0` is the stopping condition → ✅ `∇E = 0` is ·
❌ training reshapes the energy function → ✅ `E` is hardcoded; training moves the coordinates fed into it ·
❌ "attention is retrieval" in the full sense → ✅ mechanism borrowed, no archive exists · ❌ CoT = the model
reasoning → ✅ mechanically, trading depth for length · ❌ depth is just the compute budget → ✅ a fixed L is
simultaneously too much for easy tokens and too little for hard ones.

**Own gaps ("Open threads", 7 items):** Jacobian/spectral conditions at the fixed point *(again)* · Dong et al.
2021's contraction proof not internalized · induction heads (Olsson et al.) not read · DEQ/implicit differentiation
gestured at, not derived · ACT / Universal Transformer / Mixture-of-Depths not surveyed · CoT complexity results
claimed, not proven · Storkey / pseudo-inverse rules mentioned only.

**Read first:** the note's own pointer is §5, §6, §10. **§6 is the one** — *self-consistency ≠ correctness* is
precisely the trap any self-generated halting signal must avoid: a confidently-wrong loop halts confidently.

---

### 3.7 · [[7-vit-foundation]] — 40 KB

**Settles:** how a sequence architecture was made to eat images, and why the 2020 recipe needed 300M images.

**Depth: L3 on the proofs actually carried out** — permutation equivariance (`Attn(ΠX) = Π Attn(X)`, full proof via
`S' = ΠSΠᵀ`), patch-embedding-as-convolution, and the `√d_k` variance derivation. **L4** on critique. **L2** on
model-configuration bookkeeping.

**Holds:**
- The **stem** is where nearly every design question in this field concentrates. Patches exist because pixel-level
  attention costs **65,536×** more — a compute concession, not a claim that 16×16 is a natural unit of meaning.
- `E` **is** a convolution — literally `Conv2d(3, 768, k=16, s=16)`, 590,592 params — just the weakest possible
  kind: one layer, no overlap, no nonlinearity. *"ViT has no convolution" is false as commonly stated.*
- Self-attention is provably permutation-equivariant, so without `E_pos` the model is a bag of patches. This single
  fact is the root of why position embeddings must exist.
- **Information preserved ≠ structure usable** — the load-bearing inequality. A fixed permutation is invertible yet
  destroys a CNN; ViT-B/32 is *lossy* (4× compression, 3072→768) and works fine. Losslessness was never why ViT
  works.
- **Inductive bias = findability, not representability.** Attention can express any conv layer (Cordonnier: MSA
  with relative PE, `h ≥ k²`) but SGD must *locate* that region of function space unaided — hence JFT-300M.
- The hybrid stem is a **borrowed prior**: helps most when compute/data are scarce, and the advantage vanishes at
  scale — a result the paper itself calls surprising.
- Probe toolkit (§8, reused by every later paper): E-PCA, `E_pos` similarity, and mean attention distance
  `d̄ = (1/N) Σᵢ Σⱼ A_ij ‖pᵢ − pⱼ‖`. Pure ViT grows local heads early; the hybrid has **none**, because the conv
  stem already did that work.
- MLP holds **2× the parameters** of attention per block.

**Kills (§9, in full):** ❌ ViT has no convolution → ✅ the patch embedding is one · ❌ multi-head splits the vector
between heads → ✅ every head reads the full D-dim token; only the output subspace is per-head · ❌ `E` preserves all
information so nothing is lost → ✅ only when `D ≥ P²C`; and invertibility was never the relevant property ·
❌ weight sharing in `E` gives translation invariance → ✅ equivariance, and only for shifts that are exact multiples
of P · ❌ `E` can't encode position → ✅ it fully encodes position *within* a patch, never *between* ·
❌ the model waits several blocks to learn intra-patch structure → ✅ it's in the token from layer 0 ·
❌ ViT requires JFT-300M → ✅ true of the 2020 recipe only (DeiT/AugReg/MAE/DeiT-III reach 83–85% on IN-1k alone) ·
❌ CNNs are obsolete → ✅ ConvNeXt matches Swin; outcome is scale/data/objective, not operator.

**Own gaps:** §6.4's caveat box calls the JFT-300M table *"a fact about 2020 tooling, not a law of nature"* and
says the caveat matters more than the table. §7.4 leaves the vanishing hybrid advantage unexplained (picked up in
note 9).

**Read first:** **§6, inductive bias** — the theoretical spine everything else explains. Close second: **§8**, the
probe toolkit, which is the measurement instrument every paper in note 9 reuses.

---

### 3.8 · [[8-position-transformer]] — 37 KB

**Settles:** what position *is* inside a Transformer, why adding it to content doesn't destroy both, and the entire
ViT→Swin→RoPE arc as one movement.

**Depth: L3 core proofs · L4 critique.** The four-term decomposition, linear-recovery-by-projection, and RoPE's
relative property `R_pᵀ R_j = R_{j−p}` are built from scratch.

**Holds:**
- **Position is only usable when carried by a *slot* (index), not a *value* (content requiring decode).** This one
  distinction dissolves most of the confusion. And **patch size is literally the boundary between the two regimes**
  — which explains why smaller patches help beyond raw capacity.
- Additive PE isn't destructive because high-D space is near-orthogonal (`Std[cos] = 1/√D ≈ 0.036` at D=768) and
  `W_Q,W_K,W_V` are linear projections that can separate the subspaces — **but that separation is learned, not
  designed, and can fail.** `E_pos` has rank ≤197 of 768; effective rank in trained models is ~20–40.
- **The four-term decomposition** (`S_ij ∝ eᵢAeⱼᵀ + eᵢApⱼᵀ + pᵢAeⱼᵀ + pᵢApⱼᵀ`) — the note calls it *the single
  most useful equation in the whole topic*. Term 4 (position–position) is exactly how local/conv-like heads emerge;
  terms 2 and 3 are largely interference (TUPE removes them and improves).
- **Relative position bias is literally a convolution inside the softmax** — proved, not analogy: `yᵢ = Σⱼ w_{i−j} vⱼ`.
  This is why Cordonnier's theorem needs the "relative" qualifier. Swin: **169 params/head** vs `E_pos`'s 151,296.
- RoPE gets relative position free from rotation algebra, zero parameters, but **cannot make `V` position-dependent** —
  a real, stated limit.
- **The unifying trend is moving position out of the data and into the comparison operation**, and choosing a
  positional scheme is choosing which symmetry group the model respects.
- **NoPE does not transfer to ViT**: it works for causal decoders because the mask already breaks permutation
  symmetry; ViT is unmasked, so note 7 §2.5's proof holds exactly.
- CoordConv: plain CNN fails coordinate regression, near-perfect with coord channels, **"essentially no change"** on
  ImageNet — because coordinate channels are a **rank-2 affine special case** of what `E_pos` already does.

**Kills:** ❌ adding position to content mixes them destructively → ✅ in high-D it is superposition, exactly
recoverable by linear projection · ❌ the patch embedding destroys spatial structure entirely → ✅ only *inter*-patch ·
❌ coordinate channels are avoided because they distort colour values → ✅ they're a restricted rank-2 case, useless
for classification because classification doesn't need absolute coordinates · ❌ NoPE generalizes to ViT → ✅ it does
not · ❌ a hybrid beating plain ViT proves better feature extraction → ✅ **confounded by the padding-supplied
positional prior** — `224/16 = 14` exactly, so patchify pads nothing while a ResNet stem pads every layer.

**Own gaps:** §2.6 states plainly that subspace separation is *"an emergent property of training, not a design
feature… only ever approximate, and can fail."* §6.4 flags RoPE's distance decay as weaker and less controllable
than an explicit bias table. §10 flags that /16-vs/32 confounds spatial resolution with compute.

**Read first:** **§3, the four-term decomposition.** Close second: §9–10 (position below the patch / how much
spatial resolution is actually needed).

---

### 3.9 · [[9-vision-tranformer-landcape]] — 40 KB

**Settles:** everything after 2020, organised by **where in the machine each paper intervenes** — which reveals
most of them are buying the same thing in different currencies.

**Depth: L4 dominant · L2 per-paper · L5 in §10.** §10 proposes a genuinely unrun, compute-matched 3-arm design and
says so explicitly.

**Holds:**
- **Four intervention points:** stem (Early Convolutions), sequence (Registers), attention structure (Swin), loss
  (DeiT). **Nobody touches the MLP** — despite it holding 2× the parameters of attention.
- **Attention sinks / registers are a structural consequence of softmax normalization** (`Σⱼ A_ij = 1`, no
  "attend to nothing" option) — not a training quirk. ViT's *lack* of a causal mask is why the sink lands in messy,
  image-dependent locations instead of a fixed slot. ~2% of patch tokens develop order-of-magnitude-higher norms.
- **Hierarchy is an efficiency choice, not a requirement** — ViTDet refutes "hierarchy required for dense prediction."
- DeiT: `cos(z_class, z_dist)` rises 0.06 → 0.93 but never converges — two genuinely different summaries riding the
  residual stream in parallel. Hard distillation (argmax) beats KL-to-soft: 81.8% → 83.4%. **DeiT-III then beat the
  distillation recipe with LayerScale + 3-Augment + BCE — most of the 2020 ViT-vs-ResNet gap was recipe artifact.**
- TNT nested cost `≈ 8.9×10⁴` vs flat 4×4's `≈ 9.8×10⁶` — **~110× cheaper**, +1.7% — and still failed in practice
  for **hardware utilization** reasons, not conceptual ones.
- T2T: token count is governed by **stride alone** (`N ≈ (H/s)²`); overlap costs channels (~3.06×), not `N²`. Its
  headline gain is confounded with a simultaneous deep-narrow backbone change, per the authors' own ablation.
- **The ~2% ceiling:** conv stem +1–2%, TNT +1.7%, T2T +1.7%, DeiT hard distillation +1–2%, Swin RPB comparable.
  Every method supplying spatial prior converges on the same gain → **the winner is decided by cost, not cleverness.**
- **Noise floor (§10.5):** seed variance ±0.1–0.2% on full IN-1k, ±0.5–1.5% on small datasets — *"one seed on a
  small dataset produces a number that means essentially nothing."*

**Own gaps — the note states its own epistemic status up front:** *"Sections 1–7 are reported results. Section 8
(the ceiling) is an empirical pattern with no accepted theory behind it — treat it as a working hypothesis.
Section 9 is a judgement call about the state of a fast-moving field as of mid-2026; verify anything load-bearing."*
§9.3(f) on the ceiling: *"Lowest confidence item here — this is my framing, not a recognised open problem."*

**§9.4, the room-vs-compute table he now navigates by:**

| Area | Compute needed | Room left |
| --- | --- | --- |
| Mechanistic interpretability | low | large — *"most accessible without a GPU cluster"* |
| Adaptive tokenization | medium | large |
| **Test-time compute for vision** | medium | **very large — "the widest open space right now"** |
| New architectures | very high | small (competing with labs) |
| Efficient attention | high | small |

**Read first:** **§10, designing an experiment in this space** — diagnosis not architecture search, the 3-axis
decomposition (feature prior / positional prior / optimization stability), the PE-removal isolation design, the
required per-run instrumentation, and the noise-floor gate (run the baseline 3× before trusting any comparison).

---

### 3.10 · [[10-attention-collapse-and-field-equilibrium]] — 50 KB

**Settles:** the physical ledger of what a block does to a token — what is added, what is destroyed, what is left —
and whether "run it to equilibrium" can ever be a halting rule.

**Depth: L4 throughout** (the most critique-dense file here: four independent structural proofs in §10, a
wrong-picture table at the top *and* Appendix A), **L3** in places, **L2** in §14–16 where it reframes existing
literature (progressive stacking, LiGO, layer dropout) rather than deriving it.

**⚠ This file issues the vault's one explicit correction.** Note 6 can be read as claiming a Transformer must
*avoid* reaching `∇E = 0`. **§4.3 says that reading is wrong:** attention reaches its head-level fixed point every
single time, in one step, and that is normal, required behaviour governed by a real `E`. What must be prevented is
every token converging to the *same* fixed point across depth — **stack-level rank collapse, which has no governing
`E` at all**; it is a geometric side effect of repeated convex averaging, not energy descent. *"Rank collapse is
metastable merging, propagated across depth until nothing is distinguishable."*

**Holds:**
- `N` is a **contracted** index, never stored. Shape invariance comes from contraction, scale invariance from
  `Σaᵢ = 1` — two separate properties, both required.
- **A context window is not a compression mechanism.** It is simply the largest N accepted. Out-of-window tokens
  are absent from X entirely, not down-weighted.
- **Two categorically different fixed points** (see the correction above) — conflating them was note 6's own risk.
- **The residual stream is an accumulator** whose norm grows with depth: `z^(ℓ) = z^(0) + Σ writes`. **Late-layer
  quiet is chosen non-writing (no-op heads), not exhaustion.**
- Not one shrinking hull across depth but a **chain of freshly-built hulls**; the residual re-expands past each
  hull's boundary every layer. Usable depth is that balance.
- **The five-place destruction ledger:** window boundary (total, largest loss) · the `aᵢ ≈ 0` tail (locally lossy,
  globally recoverable) · rank collapse (irreversible) · the exit/pooling (deliberate) · compression (deliberate,
  separate). **Nothing is destroyed "in the middle" by attention averaging.**
- Attention *is* genuine content-addressable retrieval — whose archive is the present input, not a persistent one.
- **§10–11, the halting result.** You structurally cannot run the field to equilibrium (four independent reasons;
  equilibrium is input-independent, the same failure as collapse in a different shape). **What does work is
  `‖Δ^(ℓ)‖ = ‖Attn^(ℓ) + FFN^(ℓ)‖ < τ` — "stop when the block stops writing"** — real, deployed, 20–50% compute
  savings (early exit, CALM, MoD). **And it is explicitly a savings criterion, never a correctness criterion:** it
  tells you the model stopped changing its mind, not that it is right. PABEE's variant (unchanged prediction for p
  consecutive layers) is more robust than a confidence threshold.
- Reading the stream is free: the shared residual basis already makes it readable (logit lens
  `logits^(ℓ) = W_U LN(z^(ℓ))`). Cost asymmetry: classification head ~1,500 params vs a block's ~7M — *"saving on
  the head is ~0.02%, skipping blocks is ~40%."*
- **§14–16, growing depth: never freeze earlier layers.** A layer trained alone learns to be a *final* layer — the
  wrong contract. Grow without freezing, attach exit gates last. The real risk is not layer 2 stealing layer 1's
  job (relocation is fine if loss drops) — it is nobody doing the job.

**Kills (selection from ~22):** ❌ variable set size is still a problem downstream → ✅ N is contracted ·
❌ `W_Q,W_K,W_V` expand dimension to absorb N → ✅ they are `d × d/h`, shrink, act per-token · ❌ rank collapse =
tokens stop moving → ✅ tokens keep moving; `‖zᵢ − zⱼ‖ → 0` is what happens · ❌ the residual signal is used up by
the last layers → ✅ it accumulates · ❌ equilibrium means done thinking → ✅ equilibrium is input-independent ·
❌ FFN "thinks" if iterated → ✅ memoryless; the 4d expansion collapses back on the same line every time ·
❌ pooling squeezes data to separate signal from noise → ✅ noise isn't removed, it gets `aᵢ ≈ 0`; it is selection
via gradient routing · ❌ prune from the bottom or the end → ✅ **middle layers are empirically the redundant ones** ·
❌ you must build a mechanism to preserve information across depth → ✅ the residual makes preservation the default;
you would have to work to destroy it.

**Own gaps ("Open threads", 8 items):** Dong et al. 2021 and precisely which of residual/FFN/high-β heads does how
much work · attention sinks — why token 0 becomes the dumping ground · logit lens vs tuned lens — what the learned
affine corrects for · MoD routers — how a router predicts which tokens need a block when it cannot measure `‖Δ‖`
first · COCONUT / looped transformers · layer-pruning studies · emergent-capability degradation under compression ·
**and still outstanding from notes 4 and 6: the Jacobian / spectral conditions at the fixed point.**

**Read first:** the note says §4, §6, §8, §11, §16. **§11 is the one** — the entire equilibrium investigation
converts into one measurable, label-free, already-deployed criterion, paired with the honesty about exactly what it
cannot tell you.

---

### 3.11 · [[11-perciever-and-more]] — 106 KB, the largest file here

**Settles:** the Perceiver family derived from zero under one frame — **attention is a priced resource, and every
architecture in this family is a different budget.**

**Depth, per axis:** **L3** on mechanism, cross-attention, position encoding, decoder, and the full FLOP algebra
(sanity-checked against the paper's 707.2B). **L3→L4 and original** on weight sharing (§8): he derives the
gradient-as-sum-over-uses rule, the spectral-radius argument for which round dominates, and the batch-coherence
explanation for why `C₁` must be excluded — and states plainly *"the analysis below is mine, not the paper's."*
**L4** on the literature map. **L5** in §14–15.

**Holds:**
- **`O(N²)` is not a law of nature — it is the cost of choosing `n_q = n_kv = N`.** Decoupling query count (M,
  yours) from key/value count (N, the data's) is the entire idea. At ImageNet `N/M = 98`.
- **Weight sharing is not a size optimization.** Parameters 340M → 47M but **FLOPs are bit-identical at 707.2B**.
  What it changes is the hypothesis space: pipeline `f₈∘…∘f₁` → dynamical system `Z_{t+1} = f(Z_t; X)`. *That is
  the precondition for calling it a loop at all.*
- The `C₁` exclusion is *mechanically* explained: round-1 gradient is rank-limited (≤M) and **batch-coherent (~B vs
  ~√B for t≥2)** — a 32× advantage at B=1024. Generalized: *a step that begins with "nothing in mind" and a step
  that begins with "something in mind" are different operators by nature.*
- **Spectral-radius rule:** `ρ(J) < 1` → late rounds dominate; `ρ(J) > 1` → **early rounds dominate, counter-
  intuitively**; `ρ(J) ≈ 1` → balanced. A ±0.05 per-sublayer gain error over 84 sublayers gives 60–77× imbalance.
  **Diagnostic: log `cos(g₂, g₈)`, not just norms — negative cosine means one operator is serving conflicting roles.**
- **~80% of compute is reading, ~20% is thinking.** One cross-attend ≈ 8.0×10¹⁰ ≈ **22 self-attention blocks**.
  Caching K,V saves 27% of total model compute for free; two-stage coarse-to-fine top-k gets a re-read 50× cheaper.
- **Perceiver IO doesn't widen the bottleneck — it makes it irrelevant**, by giving the decoder an output-query path
  that lets fine detail skip `Z` entirely. That is why one read becomes sufficient.
- **§14, his own design:** full detailed read once, cheap coarse re-reads for search, output query carries detail
  directly. Costed at ≈8.2×10¹⁰ — **~3% over Perceiver IO, 7.8× cheaper than the original Perceiver.** The
  principle: ***"detailed" and "repeated" never had to come together*** — repetition is *search*, not
  detail-extraction. Includes a falsifiable 30-minute validation with two pass/fail signatures.
- Perceiver-as-backbone is dead; **Perceiver-as-component won completely** (Resampler, Q-Former, DETR).

**Kills:** ❌ quadratic cost is inherent to attention → ✅ it's the cost of a choice · ❌ weight sharing is about
fewer parameters → ✅ identical FLOPs; it changes the hypothesis space · ❌ the round closest to the loss has the
largest gradient → ✅ set by `ρ(J)`, not distance · ❌ 25:1 compression means a hard bottleneck → ✅ 512 *slots*,
working memory with distinct registers, not a summary · ❌ one read must be insufficient → ✅ only if Z must carry
everything · ❌ convolution is cheap because it is a smarter prior → ✅ **cheap because it is dumber** (fixed,
non-adaptive) — correct for natural images, wrong for data without locality · ❌ you still need Perceiver machinery
after patchifying → ✅ once `N = 196`, `N²` is a rounding error and you just need ViT · ❌ generic proxy queries in
KV-cache compression are a reasonable design → ✅ **an admission of defeat.**

**Own gaps:** flags §8.7's mechanism as his own unpublished analysis · states DEQ+Perceiver and his Jacobian
analysis *"has not been analysed for Perceiver specifically in the literature"* · on his own two-tier design:
*"This is empirical; measure it, do not assume"* · names the echo-chamber risk (loop gradients making the CNN
faithful to the loss rather than to the image) as real and unfixed · **§15 header: papers dated 2026 were located
by search, not read — verify against originals.**

**Read first:** **§15** for direction-setting (the closed/half-open/open map, and the **binding → budgeting →
eviction** dependency constraint). **§14** for the original design. **§8** if anything shared-weight and iterative
is ever built — it is directly load-bearing, not Perceiver trivia.

---

### 3.12 · [[12-recurrent-visual-attention]] — 55 KB

**Settles:** glimpse attention end to end — RAM (2014) as a POMDP with a bandwidth-limited sensor, the full
REINFORCE derivation, the four families of differentiable fixes, and **an autopsy: why the mechanism died and what
survived.**

**Depth: L3 derivation · L4 critique.** The log-derivative trick is built step by step, the Gaussian policy
gradient is chain-ruled to plain language, baseline unbiasedness is proven by integral manipulation, and a
two-choice Bernoulli toy problem is solved by hand *and checked against the closed form*. Not L5 — it catalogues
fixes rather than proposing new ones; that move happens in note 13 and s2.

**Holds:**
- RAM is a **control problem, not a supervised one** — the structural break from every other note here. The
  192-dim sensor is a deliberate **read-bandwidth limit**; compute is `O(1)` in image size, and that — not accuracy
  — is the paper's real claim. `h_t` is the only memory, which is why an RNN is structurally necessary.
- **Gradient death is two independent layers, not one:** Layer 1 = pixel-grid discretization of `crop`
  (`∂ρ/∂l = 0` almost everywhere — **the real killer**); Layer 2 = sampling from a distribution (a genuine barrier
  only for truly discrete choices, since a Gaussian was always reparameterizable). Conflating them misreads STN.
- **REINFORCE variance is linear in T and action-dimension d, not exponential** — `E‖ĝ‖² ≈ (Td/σ²)E[R²]`. That is
  exactly why it worked for RAM (d=2) and cannot work at LLM scale (d=10¹¹). And because reward exists only at T,
  `R₁ = … = R_T = R`: **every glimpse, useful or not, gets an identical signal** — the true source of difficulty.
  Baselines are provably unbiased and cut variance 10× at p=0.9.
- **RL is never a whole-system choice — it is the smallest possible patch over the one hole backprop cannot cross
  (~1% of parameters).** The same pattern RLHF later reused.
- **Kernel choice = gradient choice:** box (naive crop, zero gradient a.e.) < hat (bilinear/STN, local) < Gaussian
  (DRAW, smooth, wide support). But **the wall's reach cost is quadratic** `(S+2d)²` — image-spanning gradient
  reach costs more than reading the whole image. RL alone decouples search radius from read bandwidth.
- **Why RAM lost, ranked honestly (§14):** (1) **sequential reads kill GPU parallelism** — 3× fewer FLOPs and still
  slower wall-clock, *the real cause of death*; (2) reads too little (~1,150 dims for a whole image); (3) `h_t` is
  an unfixed bottleneck — memory, not input, becomes the constraint; (4) hardware lottery; (5) **cluttered MNIST
  never demanded sequential reasoning** — RAM proved *selecting* where to look helps, never that *looking in
  sequence* helps.
- **The pointer survived everything.** Predict an address, read only there, `O(1)` regardless of memory size —
  deformable attention, RAG, tool use, KV-cache eviction, an agent running `grep`.

**Kills:** ❌ choosing where to look is discrete therefore non-differentiable → ✅ `l_t` is continuous; the blocker
is pixel-grid discretization · ❌ `location_net` outputs a location → ✅ it outputs a policy **mean** · ❌ REINFORCE
variance is exponential in problem size → ✅ linear in T and d · ❌ RAM lost because of REINFORCE → ✅ it lost
because of parallelism · ❌ differentiable implies optimizable → ✅ the location gradient only knows local texture;
the aperture problem survives any amount of smoothing · ❌ just widen the kernel to reach across the image →
✅ reach cost is quadratic, a closed trade-off · ❌ RAM proved sequential looking is useful → ✅ it proved location
selection helps · ❌ hard attention wins because it is cheaper in FLOPs → ✅ irrelevant; wall-clock is what matters ·
❌ RAM is "attention" in the self-attention sense → ✅ cross-attention only; no pixel ever talks to another pixel.

**The three sentences (§20, verbatim):**
1. *"RAM's real contribution is the framing: what to read is a decision that must be learned, not an architecture
   that is fixed."*
2. *"It lost because it optimised the wrong resource for its era — it saved FLOPs (getting cheaper every year) and
   spent parallelism (the thing that mattered)."*
3. *"The pointer survived everything. Whenever a system predicts an address and reads only there — deformable
   attention, RAG, tool use, agents running `grep` — that is this idea, still running."*

**Read first:** **§15, where the idea actually survived.** It draws the pointer (address-based, `O(1)`) vs
Hopfield/attention (content-based, `O(N)`) distinction and lands on ***"RAM is Hopfield where the agent chooses its
own next probe."***

---

### 3.13 · [[13-latent-space-and-shortcut]] — 39 KB, the newest and the sharpest

**Settles:** what latent space mechanically **is**, and why latent reasoning works in text and collapses in images.

**Depth: L5 · L4 · L2 formalism.** L5 because §10 builds a complete 8-experiment evaluation protocol explicitly
marked as *"the transferable contribution… regardless of whether any particular architecture wins"*, §11 proposes a
new mechanism, and §12 runs an explicit threat model with concessions. L2 only on the imported causal-mediation
formalism, which is used rather than built.

**Holds:**
- **Latent space is a negative definition:** `h_t → h_{t+1}` replaces `h_t → logits → token → embed → h_{t+1}`.
  **The same space with the vocabulary constraint removed** — not a new medium. You gain bandwidth and lose
  anchoring, supervision, and measurability.
- **A latent that is a deterministic function of the prefix carries `I(Z_t; Y | prefix) = 0`.** It buys
  computational depth and *never new information.* **Placeholder behaviour is the mathematical default, not a
  training failure.**
- In MLLMs, visual/latent tokens are **ordinary sequence positions**, merged at the projector before layer 1;
  information leaves a token only through attention. "Reading in more detail" means *a longer sentence*, not a
  deeper model.
- **The shortcut:** path B `v → ans` is **1 hop, pretrained, and live at all 32 layers**; path A `v → z → ans` is
  **2 hops starting from random.** B wins, and **the optimizer is right to prefer it.** `v_i` is not one shortcut
  but 32 simultaneous ones, which is why regularization cannot realistically close it. Confirmed empirically: Li
  et al. ICML 2026 causal mediation finds `α ≈ 0`, `β ≈ 0`, latents highly homogeneous; CapImagine (explicit *text*
  imagination) beats latent baselines by +3.44% HR-Bench, +2.6% V*.
- **The general law (§6): *anything avoidable will be avoided*** — latents, glimpses, CoT, layers, auxiliary heads.
  Same shape as RAM's location collapse, unfaithful CoT, dead residual layers. Only three fixes exist (supervise it
  / cheapen A and penalize B / **delete B**) and **only deleting the alternative path is robust.**
- **Two independent layers of shortcut (§7).** Layer 1 = bypass-latent-as-carrier, fixed by architecture. Layer 2 =
  bypass-latent-as-computation — collapses to a single-shot saliency reflex, and **is fixed only by task design.**
  Open-loop `μ_{1..k} = g(glance, q)` needs no reasoning and is fully parallelizable; closed-loop
  `μ_t = g(glance, q, ρ_{1..t−1})` is the only one that requires a loop.
- **Wide-hard vs deep-hard (§9).** GSM8K needs ~5 sequential steps with no shortcut; most VQA is 1-hop. **Vision has
  essentially no GSM8K.** Checklist for a deep-hard task: referential chain · resolution gap · low blind baseline ·
  area ratio ≫ 1. **Pointer chase** (a linked list rendered as an image) is flagged ⭐ as the best instrument.
- **The fix (§11):** make the latent **act**, not merely think. `read = ρ(x, μ(z)) ⇒ I(Z; Y | glance) > 0`. This
  deletes the architectural shortcut *and* — as a side effect — turns an opaque 4096-dim mediator into **2 plottable
  coordinates.**

**Own gaps — §12, "the threat model, what must be conceded":**
- **Threat 1** — CapImagine already beats latent methods with plain text. Concession: *"CapImagine wins because it
  is verifiable and grounded — not because it is text."* Counter: a coordinate is the **minimal symbolic
  interface** — 2 numbers vs ~200 tokens, equally verifiable and grounded.
- **Threat 2** — the shortcut can silently return: *"any of the four checklist violations in §9.4 restores it… the
  architecture does not save you from a bad benchmark."*
- **Threat 3** — the space is crowded (V*/SEAL, DyFo, ZoomEye, CVSearch, BVS, FOVEA). Claimed distinction: *"Theirs:
  discrete tool calls mediated by language. Ours: latent emits coordinates, differentiable end to end."*

**The five sentences (§15, verbatim):**
1. *"Latent space is the same space with the vocabulary constraint removed — a negative definition. You gain
   bandwidth and lose anchoring, supervision, and measurability."*
2. *"A latent that is a deterministic function of the prefix adds compute, never information. Placeholder behaviour
   is the mathematical default, not a bug."*
3. *"Visual tokens are sequence positions, not architectural layers. They merge at the projector, before layer 1,
   and information leaves them only through attention."*
4. *"The shortcut v → ans is one hop, pretrained, and available at every layer, while v → z → ans is two hops
   starting from random. B wins, and the optimiser is right to prefer it."*
5. *"Anything avoidable will be avoided — latents, glimpses, CoT, layers, auxiliary heads. The only robust fix is to
   make the mechanism unavoidable: delete the alternative path, and choose a task that cannot be answered in one
   hop."*

**Read first:** **§11** — it converts the whole diagnostic apparatus into one constructive move, and it is the
section that turns an opaque metric problem into something plottable.

---

## 4 — The two idea files: positions, not study

These are not notes. They are **stated positions with risk analysis**, written in Thai in his own working voice
(*"โน้ตงานของกูเอง เขียนลบได้ตามใจ"*). Treat their contents as **[CLAIM] with zero evidence** in draft-7 terms — but
treat their *status tables* as his current operating map of the field.

### 4.1 · [[s1-opened-topic-ideas]] — after note 11

**Theme:** a fixed-size context window that still handles big data — porting LLM-side context compaction to images.

**The reframe that survived:** ***the real problem is unit discovery, not compaction.*** Once you have a unit,
compaction is the same on both sides. Still images have no natural boundary → the unit must be **the object**
(spatial axis); video does → the unit is **shot / scene / event** (temporal axis).

**The forced solve order — the single most reusable line in the file:** **binding → budgeting → eviction.** Jump
ahead and you hit *"I don't know what to delete"* — which he identifies as the exact wall the whole streaming-video
field is currently stuck on.

| | Idea | Value | Core move |
| --- | --- | --- | --- |
| 1 | **Object-bound dynamic latent** | ⭐⭐⭐ | **Flip the softmax axis.** Perceiver normalizes over N (latents don't compete); Slot Attention normalizes over M (pixels distribute their budget, **slots compete**) → object binding emerges unsupervised. Plus ACT halting over `M_max` with soft mask `α_m = clamp(1 − relu(c_m − 1), 0, 1)`, so used-slot count `Σα_m` is real and differentiable. Pareto (`Σα` vs quality) is the paper's main figure. |
| 2 | **Query-conditioned compaction** | ⭐⭐ | Both current keep-criteria guess: redundancy scores a never-repeating sky highly; proxy-query uses a template that isn't the real question. **Perceiver already has a learned query.** `keep_score(x) = max_m ⟨Z₀[m]W_Q, xW_K⟩/√d` — a shift from *compression criterion* to *relevance criterion*. Plus invented evict rules: `utility − γ·age`, consolidate on `cos > τ`, protect by gate. |
| 3 | **Two-tier read** | ⭐ | = note 11 §14 carried over verbatim. |

**Best instrumentation idea in the file: the slot drop test.** Delete slot *m*; if you lose *one object*, binding
is real and you have the building block of compaction. If you lose *global quality*, you don't. He names it the
decider, and it costs 2 hours.

**Named biggest risk, in his own judgement:** Slot Attention is famous for working beautifully on CLEVR and dying on
natural images. His fix: **start from DINO/DINOv2 features, not a self-trained CNN.**

**Notes to his future self (all five still load-bearing):** binding always first · don't shrink `d` during a poor
read, it kills the conditional query which is the only reason to re-read · measure `cos(g_t, g_t′)` before blaming
the loop · shared params have 2.6–7× higher effective LR · **the 2026 papers here came from search, originals not
read.**

### 4.2 · [[s2-opened-topic-ideas]] — after note 13, **the live position**

**§0 is stated as more important than any of the ideas:**

> old: *"how do we optimize a model that looks at a still image and understands it"*
> **new: *"how do we optimize when attention size is small but the model chooses what to look at and what to write
> down"***
>
> *"This isn't changing the solution, it's changing the problem"* — the right move against a crowded field.

**The name for humans: `Bounded-bandwidth visual reasoning`.** Pitch: *"today we measure understanding when the
model sees the whole image; I care how much it understands when it has to choose what to see."* Side effect:
everyone else's work becomes the special case `attention size = N` instead of a competitor.

| | Idea | Value | Core move |
| --- | --- | --- | --- |
| **1** | **Latent computes both `l` (position) and `s` (scale)** | ⭐⭐⭐ | The table that carries it: RAM learns position, scale **fixed** · STN could learn it, nobody studied it · **DRAW learned σ in 2015 — the only one in history** · Deformable DETR fixed multi-scale · ViT fixed patch size. `l_t = tanh(W_l z_t)`, `log s_t = W_s z_t` (predict **log** s so scale is multiplicative), sampling grid `p_ij = l_t + s_t(u_i, v_j)` on a fixed g×g grid = STN affine locked to isotropic scale + translation. `∂ρ/∂s = Σ (∂ρ/∂p_ij)(u_i,v_j)` = image gradient weighted by distance from centre = literally *"does the outer rim add information? if not, zoom in."* |
| 2 | **Hopfield-style address retrieval for gaze** | ⭐⭐ | `l = P softmax(β Mᵀ q)` — modern Hopfield retrieval exactly, but **retrieving an address instead of content**, then reading the real thing at full resolution. No walking → no wall, no local minimum, no RL. `O(N_low)` on a thumbnail (8× downscale = 64× cheaper). Interpretable: the attention map on the thumbnail *is* "what it is thinking about." |
| 3 | Latent chain decomposing an image into parts | ⚠ **demote** | = the part-whole hierarchy problem. Hinton fought it 10 years (Capsules → GLOM), nobody succeeded: no supervision, no metric, and by note 13 §6 it becomes a placeholder because nothing forces it. **Keep as an analysis section, never a method section.** |
| 4 | **Pointer-chase benchmark** | ⭐⭐ | The most de-risked contribution. Model fails → the benchmark still exists for others. Model works → the benchmark makes the result mean something. ~2 weeks, not 6 months. |

**⚠ The implementation detail he flags as deciding idea 1: aliasing.** A large `s` on a fixed g×g grid samples below
Nyquist → garbage reads *and* garbage gradients. Fix: read from an **image pyramid and interpolate across levels**,
`ℓ* = log₂(s_t·W/g)`, trilinear — what deformable attention does across feature scales and what graphics has done
for 30 years. **The by-product is the point:** the pyramid makes the trade-off *physically real* (large `s` = wide
but genuinely blurry), so the model cannot cheat by setting `s = 1` forever. Without it, it learns to zoom all the
way out and stop. He names this *"the shortcut specific to this idea"* and cites note 13 §6 by name.

**Why `s` is not just one more parameter:** it is a **budget decision.** Zoom out = **explore**, zoom in =
**exploit**. The model has to learn *"am I searching or am I verifying."* Plot `s_t` vs `t`: a downward slope is
**coarse-to-fine that emerged rather than being imposed** — and coarse-to-fine = graduated non-convexity = the
wall-crossing method from note 12 §12.4, meaning **the model discovers the fix to the wall by itself.** *"This one
figure sells the paper."*

**The metric nobody has plotted:** accuracy vs read-fraction `Σ(s_t W)²/HW` → ***"90% while reading 2%."***

**The merge he already identified, and it is the thread to pull:**
> **s1 idea 1 + s2 idea 1** — *if each latent holds one object and each has its own `(l, s)`, that IS s2 idea 1 in
> multi-latent form = deformable attention where the latent picks its own scale.* **These two can merge.**
> **s1 idea 2 + s2 idea 2** — associative retrieval is compaction where the query picks *coordinates* rather than
> tokens → **a stronger special case, because the compaction happens before the read, not after.**

**Work plan (framed as an outside project — optimize knowledge-per-time, not certainty of result):**
- **Phase 0 (1–2 wk)** — prove the problem is real. Build the pointer-chase generator, fire at Qwen-VL / LLaVA with
  **no training**, measure accuracy vs `k` plus a **blind baseline** (delete the image). *Kill criterion: if SOTA
  handles k=4 easily, the problem isn't real → stop.*
- **Phase 1 (1 mo)** — smallest possible model: glance 64×64 → Hopfield retrieval → `(l_t, s_t)` → trilinear
  pyramid read → loop. Single latent vector, T glimpses, **no LLM, no pretrain.** Baselines: full-image ViT,
  random glimpse, **parallel-k (the important one)**. *Kill criterion: if sequential ≯ parallel, no reasoning is
  happening →* ***go fix the dataset, not the model.***
- **Phase 2** — only if phase 1 survives. *"Absolutely do not start here"* — you cannot debug what broke.

**§9, the danger he names himself, boxed:** ***doing all three ideas at once.*** Pick 1 + 2, demote 3 to analysis,
and do 4 **first** because it is the insurance.

---

## 5 — What he now believes: the cross-cutting laws

These recur across three or more files and are the things that can be assumed in any conversation from here.

1. **Attention blends, never creates.** Output is confined to `hull(V)`; softmax *is* the convexity condition. FFN
   is the only escape. *(1, 5, 10, 11)*
2. **Softmax = a budget** (`Σa = 1`). That constraint is what makes it Hopfield, what makes competition possible,
   and — when the axis is flipped — what produces binding. *(4, 5, s1)*
3. **A fixed point is self-consistency, not correctness.** Spurious states are fixed points too. *(4 §6.5, 6 §6)*
4. **The convergence inversion:** Hopfield's goal is the Transformer's failure mode. Same mechanism, opposite
   destination. *(2 §13.6, 4, 6)*
5. **Head-level fixed points are normal and reached every step; stack-level fixed points are rank collapse and
   fatal.** Conflating them is the vault's one recorded self-correction. *(10 §4.3, correcting 6)*
6. **The only halting rule that works is `‖Δ^(ℓ)‖ < τ` — "stop when the block stops writing" — and it is a savings
   criterion, never a correctness criterion.** *(10 §11)*
7. **The residual stream is an accumulator.** Information is destroyed in exactly two places: the window boundary
   and the exit. Never "in the middle." *(10 §7–8, 5 §7)*
8. **Anything avoidable will be avoided.** The only robust fix is to *delete the alternative path* — supervising or
   penalizing it is gameable. *(13 §6, 12 §14, 1 on CoT faithfulness)*
9. **A deterministic function of the prefix adds compute, never information.** Placeholder behaviour is the
   mathematical default. *(13 §1)*
10. **Inductive bias is about findability, not representability.** Attention *can* express convolution; SGD has to
    locate it. *(7 §6)*
11. **Position must move out of the data and into the comparison operation.** That is the whole ViT→Swin→RoPE arc.
    *(8 §8.1)*
12. **Attention is a priced resource.** `O(N²)` is a choice, not a law; decoupling `n_q` from `n_kv` is the entire
    Perceiver idea. ~80% of the bill is reading. *(11 §2, §11.2)*
13. **"Detailed" and "repeated" never had to come together.** Repetition is *search*; search can be coarse.
    *(11 §14)*
14. **Half a circuit earns zero gradient.** Capabilities ride on prerequisites built for unrelated reasons —
    scaffolding, not gradual ascent. *(3 §3)*
15. **The pointer survived; the mechanism didn't.** Predict an address, read only there. *(12 §15)*
16. **Deep-hard ≠ wide-hard, and vision has no GSM8K.** Without a task that forces sequential dependency, nothing
    about a loop can be proven. *(13 §9, 12 §14 R5)*
17. **Unit discovery, not compaction, is the image-side problem — and the order is binding → budgeting → eviction.**
    *(11 §15.5, s1)*
18. **`√(S/d)` is the one number:** crosstalk (Hopfield 1982) and interference (superposition) are literally the
    same quantity. *(5 §20)*

---

## 6 — The consolidated status map

Merged from [[9-vision-tranformer-landcape]] §9, [[11-perciever-and-more]] §15, [[12-recurrent-visual-attention]] §18,
[[13-latent-space-and-shortcut]] §13, [[s1-opened-topic-ideas]], [[s2-opened-topic-ideas]] §6.

### 🔴 Closed — do not spend time here

| Topic | Closed by |
| --- | --- |
| Cross-attention bottleneck beats `O(N²)` | Set Transformer (2019) |
| Arbitrary output shapes are decodable | output queries (Perceiver IO) |
| How much image fits in few latents | TiTok (32 tokens), FlexTok (FID < 2 at 8–128) |
| Perceiver as a backbone | dead — survives only as a module (Resampler, Q-Former) |
| Is a CNN/ViT front-end needed | yes; every deployed system has one |
| "ViT requires JFT-300M" | dead — MAE / DeiT-III / AugReg substitute recipe for data |
| "Conv vs attention, which is better" | malformed question — ConvNeXt |
| What the stem should be | conv/overlap beats plain patchify, settled since 2021 |
| Attention sinks / registers | confirmed mechanism *and* working fix |
| Is hierarchy required for dense prediction | no — ViTDet |
| Hard glimpse + REINFORCE for classification | STN, deformable attention — strictly dominated |
| Patch selection on gigapixel images | IPS (16 gigapixel / 5 GB), Attention Sampling |
| Sparse/deformable attention as an efficiency mechanism | Deformable DETR — standard equipment now |
| Zoom-in visual search on high-res images | V*/SEAL, DyFo (training-free), ZoomEye, CVSearch |
| Does latent CoT work in text | yes — Coconut, CODI, Huginn + survey |
| Bolt latent visual tokens onto a VLM and collect VQA points | saturated — Mirage, LVR, ILVR, Monet, LanteRn, DeepLatent, OneLatent, UniVLR |
| **Do current latent visual tokens carry visual semantics** | **answered NO — ICML 2026, causal mediation** |
| Is hard attention cheaper in FLOPs | yes, and irrelevant — wall-clock is what matters |

### 🟡 Pending — active, results not settled

Making latent visual tokens **load-bearing** *(named the main crack)* · RL on latents via Gumbel/superposition
(Latent-SFT, Latent-GRPO, SofT-GRPO) · active perception as Bayesian experimental design (FOVEA, perceptual
bandwidth bottleneck) · interpretability of latent reasoning (no accepted protocol) · self-supervised
where-to-look (LTRP, LookWhere) · dynamic `M` — variable-length reconstruction exists, **autonomous budgeting does
not** (AdaTok names the gap) · latent as persistent cross-time state (lots of engineering, little principle) ·
DEQ + Perceiver / the Jacobian analysis (**unpublished for Perceiver specifically**) · hierarchical vs plain ViT
(crossover uncharacterized) · state-space models vs attention · is depth recurrence "reasoning" or iterative
refinement.

### 🟢 Open — the actual space, in his ranking

1. **A latent that picks its own scale/resolution** — DRAW did it in 2015 and **nobody followed for 10 years.**
2. **A verifiable evaluation protocol for latent visual reasoning** — ICML just opened the problem and proposed no
   solution. [[13-latent-space-and-shortcut]] §10 is a candidate, and *"the most durable contribution available here."*
3. **A deep-hard vision benchmark** — vision has no GSM8K.
4. **A latent that issues coordinates itself, differentiable end to end** — the visual-search line all uses text,
   tool calls, or MCTS.
5. **What the ground truth of a "visual CoT" even is** — nobody has defined it. Gaze traces? Eye tracking?
6. **Test-time compute for vision** — from note 9, *"the widest open space right now."*
7. Write/evict/consolidate/allocate/protect rules for a latent — *"the widest gap"* on the memory side.
8. Binding: what each latent should *mean*. Adaptive/content-adaptive tokenization.
9. Pointer reads outside a 2D grid (video, 3D, long documents) — genuinely empty, large scope.
10. Generative (not just discriminative) latents — the prerequisite for a world model.
11. Quantifying the shortcut — nobody publishes `β` as a function of task depth, glance resolution, or model scale.

---

## 7 — Where his knowledge actually stops

The honest boundary, consolidated. Everything here is something he has flagged himself.

**The one recurring item — flag it if it ever becomes load-bearing:**
> **Jacobian / spectral conditions at the fixed point.** Named as unresolved in [[4-hopfield-internal]] (his own
> README line), again in [[6-fixed-point-and-what-transformer-chasing]] §"Open threads", and again in
> [[10-attention-collapse-and-field-equilibrium]]. Three files, one hole. It is also exactly what any DEQ-style or
> convergence-based loop design would need.

**Read-but-not-derived:** Adam, LoRA, FlashAttention's online softmax, MoE routing, RLHF/DPO, quantization,
PagedAttention *(note 1)* · the causal-mediation formalism *(note 13)* · progressive stacking / LiGO / layer
dropout *(note 10 §14–16)* · per-paper mechanisms in the ViT landscape *(note 9)*.

**Named but not read:** Dong et al. 2021's contraction proof · Olsson et al. on induction heads · Bai et al. on DEQ
· ACT / Universal Transformer / Mixture-of-Depths · the CoT complexity-class papers · Storkey / pseudo-inverse
rules · Slot Attention (Locatello 2020) — *first in his own reading queue* · DINOSAUR · AdaTok · CapImagine
(arXiv 2602.22766, which he calls **the most important paper of this line**) · Attention Sampling (arXiv 1905.03711).

**Field-level unknowns he correctly identifies as nobody's answer, not his gap:** how much superposition is in real
models · whether SAEs find real features · the linear representation hypothesis · what FFN actually injects ·
attention-head superposition · LayerNorm's representational role · why SwiGLU works · why the ~2% ceiling exists ·
why token 0 becomes the attention dumping ground · what the tuned lens's learned affine corrects for.

**⚠ The citation caveat, stated in three separate files:** *papers dated 2026 in this vault were located by search,
not read in the original.* [[11-perciever-and-more]] §15, [[s1-opened-topic-ideas]] note 5, and
[[9-vision-tranformer-landcape]]'s epistemic header all say so independently. **Verify before anything is cited
outward.**

---

## 8 — Where this vault touches draft 7.0

**On 2026-08-10 this vault became draft 7.0's direction.** The live position is
[`../../docs/direction.md`](../../docs/direction.md); the previous one (the loop of thought) is in
[`../../.archive/`](../../.archive/README.md). Two tables — what backs the current direction, and what happened to
the old picture.

### 8.1 What backs the live direction

| Element of [`docs/direction.md`](../../docs/direction.md) | Where it comes from |
| --- | --- |
| The diagnosis — latent fails on images because of a 1-hop shortcut | [[13-latent-space-and-shortcut]] §4–§6, and the ICML 2026 causal-mediation result in §5 |
| "Change the objective, not the architecture" | [[13-latent-space-and-shortcut]] §6 (*anything avoidable will be avoided*; only deleting the alternative path is robust) + §9 (deep-hard vs wide-hard) |
| **§3.1 the hand** — latent predicts `(l, s)` | [[s2-opened-topic-ideas]] §1 · the differentiable-sampling machinery from [[12-recurrent-visual-attention]] §11–§13 · the DRAW-learned-σ gap from §18 |
| **§3.2 the brain** — associative retrieval of an address | [[s2-opened-topic-ideas]] §2 · object model from [[4-hopfield-internal]] §6.7 · the bridge sentence from [[12-recurrent-visual-attention]] §15 |
| **§3.3 later, not now** — object-bound dynamic latent | [[s1-opened-topic-ideas]] IDEA 1 · mechanism from [[11-perciever-and-more]] §15.3 (the softmax-axis flip) · ordering constraint from §15.5. **Deliberately deferred** — one acting latent tests the core claim first |
| The pyramid / aliasing fix, and why it kills the zoom-out shortcut | [[s2-opened-topic-ideas]] §1.3, applying [[13-latent-space-and-shortcut]] §6 to this design specifically |
| The pointer-chase benchmark and the phase-0 kill criterion | [[13-latent-space-and-shortcut]] §9.3–§9.4, §10 · plan in [[s2-opened-topic-ideas]] §7 |
| The six kill criteria | assembled from the risk tables in [[s1-opened-topic-ideas]] §1.7, [[s2-opened-topic-ideas]] §1.7, and [[13-latent-space-and-shortcut]] §12 |
| "Is any of this new?" | §6 above is the closest thing to an answer this project has. **Check it before proposing.** |

### 8.2 What the vault did to the archived picture

Kept for the record — this is *why* the direction moved, not a live to-do list.

| Archived draft-7 element | What the vault said about it |
| --- | --- |
| **The read-bandwidth limit** — *"an information bound, not a hoped-for difficulty"* | **Survived, and was promoted to the centre.** [[12-recurrent-visual-attention]] §3 is the same argument, built. [[13-latent-space-and-shortcut]] §9 supplied the missing half: **the bound is worthless without a deep-hard task**, or the loop degenerates to open-loop |
| **Binding — named as thin** | Correct, and it became **§3.3** — deferred on purpose. [[11-perciever-and-more]] §15.5: binding is *first in a forced order* on the memory side; the **slot drop test** is a 2-hour decider when it comes |
| **The ladder rule / coarse-to-fine** | Became a **measurable prediction** rather than a thesis: [[s2-opened-topic-ideas]] §1.4, `s_t` vs `t` sloping down, connected to graduated non-convexity |
| **"The backbone must be self-supervised"** | Reached independently from the other side — [[s1-opened-topic-ideas]]'s biggest named risk is Slot Attention dying on natural images, fix = **start from DINO/DINOv2** |
| **"The right feeling"** | **Closed as a topic.** The vault hardened it rather than solving it: [[6-fixed-point-and-what-transformer-chasing]] §6 — a fixed point is self-consistency, not correctness; [[10-attention-collapse-and-field-equilibrium]] §11 — the only working stop rule is savings-not-correctness. The live question became *how associative memory should be used on data that was never symbolically dense* |
| **The arena** — CNN halves with a loop between | Superseded by [[s2-opened-topic-ideas]] §7 phase 1, which is the same instinct **specified and costed** |
| **The two-motion loop**, `f(goal, fact)` as the object | Not carried. Nearest prior art recorded: [[12-recurrent-visual-attention]] §15 — ***"RAM is Hopfield where the agent chooses its own next probe."*** Object model if it ever returns: [[4-hopfield-internal]] §6.7 and §7 |

---

## 9 — Maintenance

**Broken wikilinks (14 total).** Obsidian will show these as unresolved:

| Link used | Times | Actual file |
| --- | --- | --- |
| `[[13-latent-space-and-shortcuts]]` | 10 | `13-latent-space-and-shortcut.md` *(singular)* |
| `[[11-perceiver-and-more]]` | 3 | `11-perciever-and-more.md` *(the file name is misspelled)* |
| `[[9-vision-transformer-landscape]]` | 1 | `9-vision-tranformer-landcape.md` *(the file name is misspelled twice)* |

The cleanest fix is to **rename the two misspelled files** (`11-perciever…` → `11-perceiver…`,
`9-vision-tranformer-landcape` → `9-vision-transformer-landscape`) and repoint every link, since the correct
spellings are what the notes keep reaching for. Not done here — it renames files across the vault.

**Caveats on this index itself.** Depth ratings are judgements made from evidence inside each file (derivations
carried out, wrong pictures killed, honest-assessment sections), not from the topics' difficulty — the rubric in §0
is the standard used. Where the author's own README line disagrees with the rating, both are recorded and **his
is the one that counts**. The 2026-dated citations inherited from the notes are unverified, per §7.

**Keeping it current.** Each new note needs: a row in §1, a dossier in §3, and its verdicts merged into §6. If a
note revises an earlier one, record it the way [[10-attention-collapse-and-field-equilibrium]] §4.3 did — as an
explicit dated correction, not a silent edit.
