# reading checklist — 4 days, then the long lane

> **Commissioned 2026-08-06.** The author's own words: *"ตอนนี้กุแม่งตัวเปล่าเลย ทั้งหมดคือ pure intuition without anything
> ref"* and *"กุอยากไปอ่านก่อนว่า move ที่กุเลือก มันทำได้จริงๆมั้ย และมีวิธีที่ดีกว่านี้มั้ย"*
>
> **Working notes** — this folder is live and editable; record verdicts straight into it as you read. The picture
> being attacked is in [`the-model.md`](the-model.md) (working) / [`../docs/the-thought.md`](../docs/the-thought.md) (frozen record).
>
> **This is not a syllabus.** Its job is to attack the standing candidate ([`the-thought.md`](../docs/the-thought.md)) with
> the best rivals the world already has. Every item answers one of two questions:
> **(a) does my move already exist and does it work?** **(b) is there a better move?**
>
> **Budget: 4 days × 16 h = 64 h**, then a presentation. After that, the long lane.
>
> ⚠ **Depth beats breadth.** 64 h of a new field buys **5–8 papers understood properly**, not 25 skimmed. Tier 0 is
> deliberately small. If you are behind, cut Tier 2 entirely — never cut Tier 0's depth to reach Tier 1.
>
> This file **replaces nothing**: the deleted `learning-roadmap.md` was build-first and pointed the wrong way. This
> one is read-first and aimed at the standing candidate. Do not restore the old one.

---

## §0 — Read this section first: what of your design already exists

**The single most useful thing 64 hours can buy you.** Almost every *component* of the standing candidate has
already been built by someone. That is **not bad news** — it means you are not wrong, and the parts are known to
work. It means the contribution has to be **the composition**, and that is exactly what draft 6's red-team already
concluded about this project (*novelty = composition*). Much better to learn this now than from a professor.

| Your move | Already exists as | Verdict to check | Tier |
| --- | --- | --- | --- |
| A latent that **repeatedly attends into a large set of conv features** | **Perceiver / Perceiver IO** (Jaegle et al. 2021) | Nearly your architecture, minus the goal/fact split and the ladder. **Read first.** | **T0** |
| **Limited read per step** on an image (your bandwidth limit) | **Recurrent Models of Visual Attention** (Mnih et al. 2014); hard/glimpse attention | Built in 2014. Works, but is hard to train (REINFORCE). Find out how they trained it. | **T0** |
| **Memory retrieval ≡ attention** | **Hopfield Networks is All You Need** (Ramsauer et al. 2020) | Proves modern Hopfield retrieval *is* transformer attention. May remove your separate "memory module" entirely. | **T0** |
| **Attributes → class**, compositional recognition | **Attribute-based zero-shot learning** (Lampert et al. 2009); **Concept Bottleneck Models** (Koh et al. 2020) | Your "mandarin = round + orange + rough peel" is textbook attribute-based ZSL. It has benchmarks and a metric that is *not* the continual-learning ruler. | **T0** |
| Learning visual concepts and **executing a step-by-step question over them**, no concept labels | **Neuro-Symbolic Concept Learner** (Mao et al. 2019); Neural Module Networks (Andreas et al. 2016) | The closest thing to your *ladder* that exists. Treat as the primary rival. | **T0** |
| **Variable step count + learned halt** | **ACT** (Graves 2016), **PonderNet** (Banino et al. 2021), **Universal Transformer** (Dehghani et al. 2018) | Your criteria 2–4 already have machinery. Do not reinvent. | **T0** |
| **Vectors that add up into a concept** | **Bag of visual words → VLAD → NetVLAD** (Arandjelović et al. 2016) | Literally `Σ_c a_c · e_c`. Classical CV solved the "channels → one addable vector" step. | T1 |
| **Vectors that add *and* bind** | **VSA / hyperdimensional computing** (Kanerva 2009); Kleyko et al. surveys | Gives you `bind` — the operation co-occurrence never provides. The algebra your §2 needs. | **T0** |
| **Coarse → fine search structure** | **HNSW** (Malkov & Yashunin) — the layered graph inside every vector DB | An engineering answer to your ladder that already ships. Either it validates the ladder or replaces it. | T1 |
| **Conv channels → real separable attributes** | Sparse autoencoders / dictionary learning ("Towards Monosemanticity", Anthropic 2023) | Channels are polysemantic. This is the method that pulls actual features out. | T1 |
| **Some images genuinely need recurrence** | **Kar, Kubilius, Schmidt, Issa & DiCarlo 2019** (Nat. Neuroscience) | Empirical neuroscience evidence that hard images need extra recurrent time. **This supports you** — cite it. | T1 |
| **Ordering questions by information gain** | Version spaces & candidate elimination; decision-tree induction (Mitchell 1997, ch. 2–3); active learning (Settles 2009) | The formal form of "check the most discriminative property next." | T1 |

**How to use this table:** for each row, after reading, write **one line** in your own words: *does mine still make
sense, and if so, what exactly is different?* If you cannot state the difference, that component is not yours and
should be **used**, not rebuilt.

---

## §1 — The 4-day lane

Tiers: **T0 = must** (the design cannot be defended without it) · **T1 = should** (answers "doable / better?") ·
**T2 = if time** · **T3 = after the presentation**.

### Day 1 (16 h) — the closest prior art, so you know where you stand

| h | What | Why |
| --- | --- | --- |
| 5 | **Transformers, properly.** *The Illustrated Transformer* (Alammar) → Karpathy's *Let's build GPT* / nanoGPT, typed out by hand | You are committing to transformer-first. Know QKV as *retrieval*, not as a formula. This reframes everything below. |
| 2 | **Hopfield Networks is All You Need** (Ramsauer et al. 2020) — the idea and figures; skip the proofs | Attention **is** associative retrieval. Possibly deletes your separate memory module. |
| 4 | **Perceiver** + **Perceiver IO** (Jaegle et al. 2021) | The closest existing architecture to yours. Latent array ← cross-attention ← huge input set, iterated, weights shared. |
| 2 | **Recurrent Models of Visual Attention** (Mnih et al. 2014) | Your read-bandwidth limit, built. Note *how they trained it* — that is the hard part. |
| 3 | **Write-up:** the §0 table, filled in with your own one-line verdicts | This is the day's deliverable. Not notes — verdicts. |

### Day 2 (16 h) — memory: how it is stored and got back

| h | What | Why |
| --- | --- | --- |
| 4 | **VSA / HDC**: Kanerva 2009 *Hyperdimensional Computing: An Introduction…* → skim a Kleyko et al. survey | `bundle` vs `bind` vs `permute`. Bundling is generality; **binding is what co-occurrence never gives you.** The missing algebra. |
| 2 | **Sparse Distributed Memory** (Kanerva) — overview level | Associative memory addressed by similarity, designed as a model of *human recall*. Your memory formula's ancestor. |
| 3 | **Attribute-based zero-shot learning**: Lampert et al. 2009 → skim Xian et al.'s ZSL evaluation survey | Recognising an unseen class from known attributes **is your loop's payoff**, and it comes with benchmarks that are not AA/BWT. |
| 2 | **Concept Bottleneck Models** (Koh et al. 2020) + one critique | "Predict attributes, then class." A professor *will* ask how you differ. Have the answer. |
| 2 | **HNSW** (Malkov & Yashunin) | What a real vector DB does — a layered coarse-to-fine graph. Compare to your ladder directly. |
| 3 | **Write-up:** your memory formula, restated using their vocabulary | If you cannot restate it in their words, you do not yet understand either. |

### Day 3 (16 h) — the loop, and the ladder's ordering

| h | What | Why |
| --- | --- | --- |
| 3 | **ACT** (Graves 2016) + **PonderNet** (Banino et al. 2021) | Learned halting with variable steps — criteria 2–4 already solved. Note *what they halt on*, and that it is **not** a feeling. |
| 2 | **Universal Transformer** (Dehghani et al. 2018) | Transformer + recurrence in depth + ACT. The "looped transformer" that already exists. |
| 4 | **Neuro-Symbolic Concept Learner** (Mao et al. 2019); skim **Neural Module Networks** (Andreas et al. 2016) | The strongest rival to the ladder: learns concepts from images without concept labels and executes a stepwise program over them. **Know exactly how yours differs.** |
| 3 | **Mitchell 1997, *Machine Learning* ch. 2 (version spaces / candidate elimination) + ch. 3 (information gain)** | The formal shape of "rejection must be constructive" and of ordering the ladder conditionally. Old, short, and exactly on target. |
| 2 | **Kar et al. 2019** (recurrence in the ventral stream) | Evidence that *hard images need more time*. Your arena's justification, from neuroscience. |
| 2 | **Write-up:** which loop family you are actually in, and what the halt is grounded on | Axis B is still the months-eater. Say what you now believe. |

### Day 4 (16 h) — playground, then consolidate

| h | What | Why |
| --- | --- | --- |
| 3 | **dSprites / 3D Shapes / MPI3D** (disentanglement benchmarks) and **CLEVR** (Johnson et al. 2017) | dSprites/3D Shapes have **exact ground-truth factors** — you can check whether your components recover real attributes. CLEVR is built for multi-step relational questions = your axis F. |
| 2 | **Self-supervised backbones**: DINO / DINOv2, MAE, SimCLR — enough to pick one | **Load-bearing**, not optional. See §5. |
| 2 | **NTM / DNC** (Graves 2014, 2016) — skim | The canonical "network + external addressable memory", and *why it did not catch on*. Know the failure mode before you rebuild it. |
| 6 | **Consolidate:** rewrite the standing candidate with references attached, section by section | The real deliverable of the four days. |
| 3 | **List what is still open** after all this | Honest holes, named. This is what makes the presentation credible rather than salesy. |

**If you fall behind:** protect Day 1 and Day 3's NSCL block. Those two decide whether the design survives contact
with the literature. Everything else can move to the long lane.

---

## §2 — Section 1 of yours: *the thought* (what holds goal + fact + memory output)

**Your position:** transformer-first, because the picture shares a root with it. Vectors grounded in conv channels,
so `Layer[x][y][0]` can be added to `Layer[x][y][1]`.

**T0**
- **Vaswani et al. 2017, "Attention Is All You Need"** — plus Alammar's *Illustrated Transformer* and Karpathy's
  *Let's build GPT*. Read attention as **retrieval by similarity**, which is the frame that makes the rest click.
- **Jaegle et al. 2021, "Perceiver" / "Perceiver IO"** — a small latent array repeatedly cross-attending into a
  large, unstructured input array, weights shared across iterations. **This is your architecture minus the
  goal/fact split.** Read it as: *someone already built the container; what I add is what runs inside it.*

**T1 — the rivals for "what is a thought made of"** (this is [`open-questions.md`](../docs/open-questions.md) axis A, which
has **three** candidates and no committed answer)
- **Ba, Hinton et al. 2016, "Using Fast Weights to Attend to the Recent Past"** — your candidate 1 (thought = a
  fast-changing weight matrix). Short.
- **Bai, Kolter & Koltun 2019, "Deep Equilibrium Models"** — thought = **the fixed point** of a dynamical system,
  not a set of tokens. A genuinely different answer, constant memory, loop is native.
- **Hochreiter & Schmidhuber 1997, LSTM** — enough to state *why not*, since you already rejected the RNN framing
  and will be asked.

**T2**
- **Locatello et al. 2020, "Object-Centric Learning with Slot Attention"** — iterative attention that **binds**
  features into slots. ⚠ It factors *entities*, you want *properties*. Read it to know the difference cold.

**What would kill your move:** if Perceiver-style cross-attention already collapses the whole "components →
thought" step and nothing about it is loop-shaped, then your contribution lives entirely in the goal/fact/ladder,
not in the representation. **That is fine — but you must know it before you present.**

---

## §3 — Section 2 of yours: *the memory output formula*

**Your position:** something like a vector database; concepts connected through shared conv components; ranking by
connection count, recomputed against the fact cache.

**T0**
- **Ramsauer et al. 2020, "Hopfield Networks is All You Need"** — modern Hopfield retrieval **equals** attention.
  If your memory is a matrix of stored patterns and retrieval is a softmax-weighted read, **you have already built
  attention** and should say so deliberately rather than discover it later.
- **Kanerva 2009, "Hyperdimensional Computing: An Introduction…"** — the algebra you are missing:
  - **bundle** (superposition) — *many things → one thing that keeps what they share* = **generality**, which is
    your ladder's rungs
  - **bind** (role–filler) — *this colour belongs to this object* — **co-occurrence never gives you this**
  - **unbind** — the query operation that pulls a filler back out
  This is the strongest candidate for [`open-questions.md`](../docs/open-questions.md) axes D and H simultaneously.

**T1**
- **Kanerva, Sparse Distributed Memory** (overview) — similarity-addressed memory, built as a model of human recall.
- **Malkov & Yashunin, HNSW** — the hierarchical navigable small-world graph inside FAISS and every vector DB. It
  searches **coarse layer → fine layer**. Compare to your ladder honestly: is your ladder a semantic version of
  this, or something HNSW cannot do?
- **Arandjelović et al. 2016, NetVLAD** (and classical VLAD / bag-of-visual-words) — the accepted way to turn a set
  of local features into **one addable vector**, i.e. exactly `Σ_c a_c · e_c`.
- **Sparse autoencoders / dictionary learning** — "Towards Monosemanticity" (2023) and "Toy Models of
  Superposition" (2022). Conv channels are **polysemantic**; a channel is not an attribute. This is the method that
  extracts real, separable features — directly load-bearing for your decomposition claim.

**T2**
- **kNN-LM** (Khandelwal et al. 2020) and **RETRO** (Borgeaud et al. 2022) — retrieval-augmented models. Read as
  the **contrast case**: retrieval is single-shot there, never iterative. It sharpens what is yours.

**What would kill your move:** if the ranking by connection-count turns out to be **globally static** in practice
(the same order every time regardless of fact cache), the ladder is a fixed pipeline with early exit and not a
loop. Mitchell ch. 3 (information gain) is the repair — check it early.

---

## §4 — Section 3 of yours: *the loop*

**Your position:** cycle `f(goal, fact) = memory output`; unsure about dynamic vs fixed weights; interested in a
**context window that does not take all input at once**, because that is what makes a thought a thought rather than
hidden memorisation of everything.

> That instinct is right and it now has a name in this repo: the **read-bandwidth limit**
> ([`the-thought.md`](../docs/the-thought.md) §9.5). It is what makes single-pass failure **provable** instead of hoped-for.

**T0**
- **Mnih et al. 2014, "Recurrent Models of Visual Attention"** — a glimpse network that sees only a small patch per
  step and must decide where to look next. **This is your bandwidth-limited loop, on images, already built.** Pay
  attention to *how it is trained* (REINFORCE) — that difficulty is the reason the line went quiet, and knowing it
  saves you a month.
- **Graves 2016, "Adaptive Computation Time"** + **Banino et al. 2021, "PonderNet"** — learned halting, variable
  steps per input. Note carefully: their halt is a **trained head**, which
  [`open-questions.md`](../docs/open-questions.md) axis B specifically excludes as *the feeling*. That contrast is one of
  your clearest differentiators — be able to state it.
- **Mao et al. 2019, "The Neuro-Symbolic Concept Learner"** — learns visual concepts from images and questions
  **without concept-level supervision**, then executes a stepwise program over them, on CLEVR. This is the closest
  existing system to your ladder. Either you can state the difference precisely, or the ladder is NSCL with
  different words.

**T1**
- **Dehghani et al. 2018, "Universal Transformers"** — recurrence in depth + ACT. The looped transformer.
- **Mitchell 1997 ch. 2–3** — version spaces / candidate elimination (constructive rejection, formally) and
  information gain (conditional ladder ordering). Short, classical, exactly on target.
- **Kar et al. 2019** — recurrent computation is *needed* for certain "challenge" images in primate vision.
  Empirical support for the arena, and a good slide later.

**T2**
- **Rao & Ballard 1999, predictive coding** — the loop as error minimisation; a rival where the loop is native and
  no explicit goal object exists.
- **Andreas et al. 2016, Neural Module Networks** — the earlier, more explicit version of program-over-concepts.

**What would kill your move:** if NSCL already does *concept learning + stepwise question execution without concept
labels*, then your novelty must be somewhere else — most likely the **coarse-to-fine ladder ordering** and the
**self-generated halt**, neither of which NSCL has (its programs come from parsed language). Confirm this yourself;
do not take this file's word for it.

---

## §5 — Section 4 of yours: *the dataset playground*

**T0 — the backbone decision, and it is load-bearing**
- **DINO / DINOv2, MAE, SimCLR** — enough to choose one. ⚠ **This is not cheap insurance any more.** Your taxonomy
  *is* the conv co-occurrence structure. If the backbone was trained on ImageNet labels, its features are organised
  by **WordNet — a hierarchy built by hand** — and your claim *"the taxonomy came from images, not language"*
  fails through the back door. A self-supervised backbone lets you say **no human ever categorised anything for
  this system**. That is your strongest claim; protect it.

**T0 — the staging, in order**
1. **dSprites / 3D Shapes / MPI3D** — synthetic, with **exact ground-truth factors** (shape, colour, scale,
   position). The only place you can *verify* that your components recover real attributes rather than hoping. Tiny,
   CPU-sized, notebook-scale. **Start here.**
2. **Single-object natural images** — **CUB-200-2011** carries **312 human attribute annotations**. Use them as a
   **diagnostic only** (does my emergent decomposition line up with human attributes?), **never** as a training
   signal — that would re-import the hand-built taxonomy.
3. **CLEVR** (Johnson et al. 2017) — compositional scenes with multi-step relational questions. This is where
   **serial dependency of lookups** (axis F) becomes natural, and where **binding becomes mandatory**: one image
   containing round + orange + green + leafy does **not** mean the orange is leafy. Co-occurrence alone breaks
   exactly here.
4. **CLEVRER** — the video version. Your own §5 example (*watch a moving object, connect it to a law of physics,
   predict it*) lives here. **Destination, not start** — expensive, and there is no GPU.

**T1**
- **Xian et al., ZSL evaluation survey** — how attribute-based recognition is measured. A ruler that is *not* the
  continual-learning ruler that caused the turn.

---

## §6 — What to have written by the end of day 4

Not notes. Five artefacts:

1. **The §0 table, filled in** — one line per row: *does mine still make sense, and what exactly is different?*
2. **The standing candidate, rewritten with references** — each of the four sections, in the field's vocabulary,
   with the prior art named. This is what makes it a research position instead of an intuition.
3. **The differentiator paragraph** — after Perceiver, RAM, ACT/PonderNet, NSCL, HNSW and ZSL, **what is left that
   is yours?** Current best answer, to be tested: the **two-motion loop** (goal tightens on match) and the
   **coarse-to-fine ladder** — nobody above has a goal that rewrites itself on success.
4. **The honest holes list** — what is still open (axes B, D, H, I are all live). A credible research talk names
   its holes; a salesy one does not.
5. **One decision:** backbone, and first playground. Nothing else needs deciding yet.

---

## §7 — The long lane (after the presentation)

Deliberately not scheduled. Roughly, and only when the four days are done:

- **Binding, seriously** — VSA implementations, and what binding costs on real conv features
- **Halting without labels** — axis B's unresolved half: what grounds the feeling for internally generated thought
  (*q = mc∆t*), where no sensory input pins either side. Human metacognition (Koriat's accessibility account,
  Nelson–Narens, tip-of-the-tongue) is a hole in this repo's reading and is the right place to look.
- **The generality axis** — hyperbolic / Poincaré embeddings, order embeddings, box embeddings: the other ways a
  space can encode *is-a* rather than *looks-like*
- **Training bandwidth-limited loops** — REINFORCE's successors; why hard attention stalled and what replaced it
- **Then, and only then:** re-imposing the project's constraints one at a time (online, local, backward-free) —
  the cost column from [`CLAUDE.md`](../CLAUDE.md) discipline rule 2

---

## Standing note

Everything in this file is `[LIT]` — pointers, not conclusions. Nothing here has been read yet by anyone in this
repo, and titles can be misremembered: **verify each reference exists before citing it in public.** When a reading
kills or confirms part of the standing candidate, record it in [`open-questions.md`](../docs/open-questions.md) with a date
— that is how this draft accumulates evidence instead of opinions.
