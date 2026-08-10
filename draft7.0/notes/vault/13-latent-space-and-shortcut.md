# 13 — Latent Space and the Shortcut Problem

> **Prerequisites**: [[1-transformer]], [[3-transformer-internal]], [[10-attention-collapse-and-field-equilibrium]], [[12-recurrent-visual-attention]] **Feeds into**: [[s2-opened-topic-ideas]]

---

## 0. The two questions this note answers

1. **What is "latent space" actually?** Not the vague sense of _"where the thinking happens"_ — the precise mechanical definition.
2. **Why does latent reasoning work in text and fail in images?** The answer is a single structural phenomenon that recurs everywhere in deep learning.

---

## 1. What latent space is — a negative definition

### 1.1 The mechanism

A normal Transformer decodes like this:

$$h_t \ \longrightarrow\ \text{logits} \ \longrightarrow\ \underset{\text{argmax / sample}}{\text{token}} \ \longrightarrow\ \text{embed} \ \longrightarrow\ h_{t+1}$$

**Latent reasoning deletes the middle:**

$$h_t \ \longrightarrow\ h_{t+1} \qquad \text{(feed the hidden state back in as the next input embedding)}$$

$$\boxed{\ \text{latent space} = \text{the same space, with the \textbf{vocabulary constraint removed}}\ }$$

This is a **negative definition**. Latent space is not a substance or a place. It is _the absence of a constraint_. People discuss it as if it were a new medium; mechanically, it is a bottleneck that was deleted.

### 1.2 What you gain and lose

|                                 | Token                                          | Latent vector                                                     |
| ------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------- |
| Bandwidth                       | $\log_2 \lvert V\rvert \approx 17$ bits        | thousands of bits                                                 |
| Hold several hypotheses at once | no                                             | **yes** (superposition — see [[5-interp-hull-and-superposition]]) |
| Must be linguistically coherent | yes                                            | no                                                                |
| **Has an anchor**               | **yes** — embeddings are pinned by pretraining | **none**                                                          |
| **Has a supervision target**    | **yes** — the next token                       | **none**                                                          |
| **Is measurable**               | **yes**                                        | **no**                                                            |

The last three rows are the price, and it is higher than people expect.

$$\text{more degrees of freedom} + \text{no additional constraints} \Longrightarrow \text{the optimiser will do the \textbf{easiest} thing, which is often \textbf{nothing}}$$

### 1.3 The information-theoretic fact that settles the matter

In Coconut/CODI, the latent is defined as the previous step's hidden state, so

$$Z_t = f(\text{prefix}) \qquad\Longrightarrow\qquad I(Z_t,;,Y \mid \text{prefix}) = 0$$

**The latent is a deterministic function of what the model already knows. It carries zero new information.**

$$\Longrightarrow \text{the only thing it can provide is \textbf{computational depth}, never \textbf{knowledge}}$$

And _"a token that adds compute but no information"_ already has a name: **filler token / placeholder**.

> Therefore the empirical finding that latent tokens behave like placeholders is **not a surprising failure — it is the mathematical default.** Absent something forcing the latent to do more, a placeholder is exactly what it is.

---

## 2. Where latents live: the taxonomy

Transformers have **two axes**, and latent mechanisms are classified by which axis they extend.

$$\underbrace{\text{sequence axis}}_{\text{left} \to \text{right},\ N \text{ positions}} \qquad\qquad \underbrace{\text{depth axis}}_{\text{bottom} \to \text{top},\ L \text{ layers}}$$

| Family                       | Axis extended              | Mechanism                                                  | Examples                                                         |
| ---------------------------- | -------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------- |
| **Depth recurrence**         | depth                      | rerun the same block stack $n$ times                       | Universal Transformer, **Huginn-3.5B**, CoTFormer                |
| **Hidden-state injection**   | **sequence**               | previous hidden state becomes the next input embedding     | **Coconut**, **CODI**                                            |
| **Vocabulary superposition** | sequence                   | latent = a mixture over the vocabulary, sampled via Gumbel | **Latent-SFT**, **Latent-GRPO**, **SofT-GRPO**                   |
| **Bottleneck latents**       | neither — a separate array | fixed-size latent array cross-attends to the input         | **Perceiver / Perceiver IO** ([[11-perciever-and-more]])         |
| **Latent visual tokens**     | sequence                   | continuous visual states inserted into MLLM decoding       | Mirage, LVR, ILVR, Monet, LanteRn, DeepLatent, OneLatent, UniVLR |

### Notes per family

**Depth recurrence** genuinely adds computation depth without lengthening the sequence. Saunshi et al. argue reasoning performance is driven mainly by _computational depth_ rather than parameter count. Huginn showed latent reasoning can emerge zero-shot from recurrent depth. But probing studies question whether it is _structured CoT-like reasoning_ or merely _iterative refinement_.

**Hidden-state injection** is the canonical "latent CoT". Coconut feeds the last-layer hidden state back directly; CODI passes it through a trained 2-layer MLP and replaces Coconut's curriculum with single-stage self-distillation (a teacher path with explicit CoT and a student path with continuous thoughts, aligned with an $L_1$ feature loss). Both inherit the same latent geometry.

**Vocabulary superposition** is the interesting recent turn — and it is [[12-recurrent-visual-attention]] §13.3 returning in a new costume. Latent-SFT restricts to a top-$k$ mixture trained with stochastic Gumbel-Softmax targets; Latent-GRPO uses vocabulary superposition with one-sided Gumbel margins; SofT-GRPO adds Gumbel noise to logits. **The shared property is that the latent becomes _samplable_ with a tractable density — which is precisely what makes policy-gradient RL applicable.** Everything from note 12 about hard choices, relaxation, and needing a density reappears here verbatim.

**Bottleneck latents** are structurally different: they are not "thoughts," they are a _compressed workspace_. Worth keeping separate in your head.

---

## 3. The MLLM token layout — where visual information actually sits

This section exists because the mental picture is usually wrong.

### 3.1 The pipeline

```
image 336×336
    │
    ▼
┌────────────────────────────────────┐
│ ViT (a separate model, often frozen)│   ← entirely outside the LLM
│ 336/14 = 24×24 = 576 patches       │
└────────────────────────────────────┘
    │  576 × 1024      (ViT width)
    ▼
┌────────────────────────────────────┐
│ Projector (2-layer MLP)            │   ← THE junction point
└────────────────────────────────────┘
    │  576 × 4096      (LLM width)
    ▼
═════ this is the LLM's input-embedding stage ═════
    │
    ▼
[<s>] [v₁ ... v₅₇₆] [question tokens] [answer]
    │
    ▼  layer 1 → 2 → ... → 32          ← all positions travel together
```

### 3.2 The junction

$$\text{Projector plays the role of } \texttt{Embed[token_id]}$$

|                         | Text token                   | Visual token            |
| ----------------------- | ---------------------------- | ----------------------- |
| How it becomes a vector | `Embed[id]` — a lookup table | **`MLP(ViT(patch))`**   |
| Result                  | $\mathbb{R}^{4096}$          | $\mathbb{R}^{4096}$     |
| After that              | enters layer 1               | **identical treatment** |

$$\Longrightarrow \textbf{they merge only before layer 1. Nothing is special afterwards.}$$

The LLM does not even know positions 1–576 came from an image. It sees 576 vectors of width 4096.

> **A visual token is not an architectural component. It is a word that did not come from the dictionary.**

### 3.3 Inside a layer

```
layer ℓ:
   x                          [N × 4096]  all positions
   │
   ├─→ Attention ────────→ ⊕     ← the ONLY place positions exchange information
   │                       │
   └───────────────────────┘
   ├─→ FFN ──────────────→ ⊕     ← each position independently
   │                       │
   └───────────────────────┘
                           ▼  to layer ℓ+1
```

$$\text{Attention} = \text{exchange \textbf{across positions}} \qquad \text{FFN} = \text{process \textbf{each position alone}}$$

$$\Longrightarrow\ \boxed{\ \text{information from } v_i \text{ can reach the answer \textbf{only through attention}}\ }$$

### 3.4 Consequences that matter

**Context blows up because visual tokens are sequence positions.**

$$336 \times 336 \to 576 \text{ positions} \qquad\qquad 4\text{K image} \to \sim 60{,}000 \text{ positions}$$

$$\text{"read in more detail"} = \text{"a longer sentence"}, \text{ not } \text{"a deeper model"}$$

This is the mechanical reason pointer reads help: they add information **without lengthening the sequence**, because the read collapses into an existing latent instead of being appended.

**Latent positions are just positions.** $z_i$ is not in a different dimension from $v_i$; they are neighbours in the same sentence, differing only in where their vector came from.

$$\Longrightarrow\ \text{"latent thought"} = \text{\textbf{reserving a slot in the sentence} and filling it with a vector that need not be a word}$$

Structurally, that _is_ a soft prompt. The only difference is that the vector can vary with the input — and if that dependence collapses, it becomes a soft prompt outright.

---

## 4. The shortcut — path A and path B

### 4.1 Definition on the token grid

```
        ┌────────────── path B (shortcut) ──────┐
        │                                       ↓
[<s>] [v₁ ... v₅₇₆] [question] [z₁ ... z_k] [answer]
        │                        ↑   │          ↑
        └──── path A, hop 1 ─────┘   └──────────┘
                                     path A, hop 2
```

|                       | Route                    | Attention edges                                       |
| --------------------- | ------------------------ | ----------------------------------------------------- |
| **path A** (intended) | $v \to z \to \text{ans}$ | $z_i$ attends to $v_j$, **then** ans attends to $z_i$ |
| **path B** (shortcut) | $v \to \text{ans}$       | ans attends to $v_j$ **directly**                     |

**Both edges exist in the attention graph simultaneously. Nothing forbids either.**

### 4.2 Why B wins — three independent reasons

|                         | path A                         | path B                                   |
| ----------------------- | ------------------------------ | ---------------------------------------- |
| Hops required           | **2**, across different layers | **1**                                    |
| State at initialisation | **random** — never trained     | **fully trained** — it _is_ the base VLM |
| Availability            | requires two learned edges     | **available at every one of 32 layers**  |

$$\text{B reduces loss from step 1} \Rightarrow \text{gradient reinforces B} \Rightarrow \text{loss is explained} \Rightarrow \text{A has nothing left to learn}$$

**Path A never gets off the starting line.** Not because it is worse, but because **B arrived first**. This is rich-get-richer dynamics, and _the optimiser is behaving perfectly correctly_ — there is no reason to construct a more elaborate mechanism for something already solved.

> Note also that $v_i$ **sits in the sequence forever**. Every subsequent position at every layer can attend to it. It is not "a shortcut" — it is **32 simultaneous shortcuts**. This is why regularisation cannot realistically close it.

---

## 5. The empirical evidence

### 5.1 _Imagination Helps Visual Reasoning, But Not Yet in Latent Space_ (Li et al., ICML 2026)

**Method — causal mediation analysis.** The process is modelled as a causal chain with the input as treatment, the latent tokens as mediator, and the final answer as outcome:

$$X \ \xrightarrow{\ \alpha\ }\ Z_{\text{latent}} \ \xrightarrow{\ \beta\ }\ Y$$

If latents are genuinely "imagination," both $\alpha$ and $\beta$ must be strong.

**Findings — both links are broken:**

- **(a) Input-Latent Disconnect**: dramatic perturbations to the input produce negligible changes in the latent tokens, indicating latents do not effectively attend to the input sequence.
- **(b) Latent-Answer Disconnect**: perturbing the latent tokens has minimal impact on the final answer, indicating limited causal effect on the outcome.

$$\alpha \approx 0 \qquad\text{and}\qquad \beta \approx 0$$

**Probing result:** latent tokens encode limited visual semantics, are insufficient for deriving the answer, and are **highly homogeneous**. The conclusion drawn is that the model adopts an **implicit shortcut that circumvents the latent visual reasoning pathway**, and that latent tokens behave **like soft prompts or placeholders rather than active carriers of visual imagination**.

**Their proposed fix — CapImagine**: imagine _explicitly, in text_. Reported gains: **+3.44% on HR-Bench, +2.6% on V\***. A simpler text-space method beating complex latent-space baselines.

### 5.2 Corroboration

_What's Holding Back Latent Visual Reasoning?_ finds **latent predictions collapse into a narrow region of space** rather than capturing the variety of an oracle — mode collapse in latent space. (Monet is reported as an exception.)

_Beyond Visual Memory: Mechanistic Diagnostics of Latent Visual Reasoning_ similarly reports that recent analyses challenge the "visual-memory account," finding latent tokens only loosely tied to the image and only weakly causal for the answer.

### 5.3 One cause explains all three symptoms

This is the strongest argument for the shortcut hypothesis: it predicts every finding without additional assumptions.

| Observed symptom                                 | Shortcut explanation                                                          |
| ------------------------------------------------ | ----------------------------------------------------------------------------- |
| **Input-Latent Disconnect** ($\alpha \approx 0$) | $z$ never needs to know about the image → **never learns to attend to it**    |
| **Latent-Answer Disconnect** ($\beta \approx 0$) | the answer never needs $z$ → **perturbing $z$ changes nothing**               |
| **Latents are homogeneous**                      | $z$ does not depend on the input → **nearly the same vector for every image** |

The third is the smoking gun. A working latent must vary with the image; near-identical latents mean it encodes nothing input-related — exactly what the shortcut hypothesis predicts.

---

## 6. The general law

$$\boxed{\ \textbf{An optional mechanism will not be learned, no matter how useful it could be.}\ }$$

This is not specific to latent reasoning. It is a recurring structural phenomenon:

| System                                      | Mechanism that _should_ be learned     | Easier alternative                | Result                                    |
| ------------------------------------------- | -------------------------------------- | --------------------------------- | ----------------------------------------- |
| **RAM** ([[12-recurrent-visual-attention]]) | choose glimpse locations intelligently | always look at the centre         | $\mu_t \to (0,0)$ — **location collapse** |
| **Latent visual reasoning**                 | imagine in latent space                | read visual tokens directly       | $z$ becomes a **placeholder**             |
| **Chain-of-thought**                        | reason then answer                     | answer, then rationalise          | **unfaithful CoT**                        |
| **Deep residual nets**                      | use every layer                        | route through the skip connection | **redundant layers**                      |
| **Auxiliary heads**                         | learn the auxiliary task               | ignore it, main loss suffices     | head never learns                         |

### Only three fixes exist

| Fix                                           | What it does                             | Examples                                                                                                                  | Strength            |
| --------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| **1. Give the mechanism its own supervision** | manufacture gradient for path A directly | Mirage (image-embedding supervision), Monet (teacher-state distillation), CODI ($L_1$ feature alignment to a CoT teacher) | expensive, brittle  |
| **2. Make path A cheaper / penalise B**       | regularisation                           | various                                                                                                                   | **weak — gameable** |
| **3. Delete path B**                          | the shortcut does not physically exist   | **← the only robust option**                                                                                              | strong              |

**RAM chose fix 3 by accident** — it never had the full image available, so "look at the centre" genuinely fails on cluttered images. That is precisely why RAM works on cluttered MNIST at all.

---

## 7. Two layers of shortcut — and why fixing one is not enough

This is the subtlety that most work misses.

|                        | **Layer 1 shortcut**                     | **Layer 2 shortcut**                                     |
| ---------------------- | ---------------------------------------- | -------------------------------------------------------- |
| Bypasses               | the latent **as an information carrier** | the latent **as computation**                            |
| Symptom                | $z$ holds no visual content              | $z$ holds content **but does not think**                 |
| Mechanism              | answer reads $v$ directly                | **$z_1$ collapses into a single-shot saliency detector** |
| Fixed by architecture? | **yes** — remove path B                  | **no**                                                   |
| Fixed by task design?  | no                                       | **yes** — require sequential dependency                  |

### 7.1 Layer 2 in detail

"Move to the most salient thing" is a **single-shot reflex**, not reasoning:

$$\mu = \mathrm{Saliency}(x) \quad\text{— one function, no recurrence, no memory}$$

If the task is _"find the digit"_ or _"is there a dog"_, saliency suffices → latent steps $z_2 \ldots z_k$ have no work → they revert to placeholders.

$$\text{Same shortcut, relocated: from "ans bypasses } z\text{" to "} z_1 \text{ bypasses } z_{2..k}\text{"}$$

### 7.2 RAM itself fell into layer 2

Cluttered MNIST is a pure saliency task. No step's target depends on what a previous step observed.

$$\text{RAM proved \textbf{selecting where to look} helps; it never proved \textbf{looking in sequence} helps.}$$

### 7.3 Open-loop vs closed-loop — the decisive distinction

|                 | Definition                                                              | Requires thinking?                   |
| --------------- | ----------------------------------------------------------------------- | ------------------------------------ |
| **Open-loop**   | $\mu_1, \ldots, \mu_k = g(\text{glance},\ q)$ — all computable up front | **no** — it is a plan, not reasoning |
| **Closed-loop** | $\mu_t = g(\text{glance},\ q,\ \rho_1, \ldots, \rho_{t-1})$             | **yes**                              |

**Open-loop is fully parallelisable.** If the model is operating open-loop, iterating $k$ times is pure waste.

---

## 8. Why latent works in text but not in images

Three reasons, and every one of them is a deficiency on the vision side.

|                              | Text                                                                                                                          | Image                                                                          |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **Task depth**               | GSM8K genuinely requires ~5 sequential steps — no shortcut suffices                                                           | most VQA is **1 hop** — the shortcut suffices                                  |
| **A teacher for path A**     | **explicit CoT is ground truth for the latent** — Coconut progressively replaces CoT tokens; CODI distills from a CoT teacher | **no ground-truth "visual CoT" exists** — Mirage/Monet must invent supervision |
| **Benchmarks measure depth** | harder = more steps = **deeper**                                                                                              | harder = more objects / bigger images = **wider**                              |

$$\boxed{\ \text{Text latents have both a task that forces depth and a teacher showing what intermediate states look like. Vision has neither.}\ }$$

**The historical cause**: vision benchmarks descend from ImageNet, which defined _understanding an image_ as _emitting a label in one shot_. VQA wrapped that label in a question. **The structure of the task never changed.** Models therefore have no incentive to think sequentially — and dutifully do not.

---

## 9. Task design: what actually forces depth

### 9.1 Two kinds of "harder"

| Kind                                                      | Adds                          | Helps?                                                         |
| --------------------------------------------------------- | ----------------------------- | -------------------------------------------------------------- |
| **Wide-hard** (2 objects, 10 objects, larger images)      | more **independent** subtasks | ❌ parallelisable — multi-head attention handles it in one hop |
| **Deep-hard** (step $t$ needs the result of step $t{-}1$) | **dependency chain length**   | ✅ not parallelisable                                          |

> Concretely: _"is there a cat and a dog"_ fails — head 3 finds the cat, head 7 finds the dog, done in one layer. **Multi-head attention was built for exactly this kind of parallel work.**

### 9.2 The quick test

**Can you say in advance where the model will have to look?**

- _"Is there a cat and a dog"_ → yes, you can → **fails**
- _"What colour is the animal to the left of the cat"_ → **you must find the cat first before "to the left" means anything** → **passes**

### 9.3 Task types that pass, in increasing strength

| Level                           | Example                                                                   | Chain                                                       |
| ------------------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------- |
| **1. Relational reference**     | _"What breed is the dog next to the cat?"_                                | find cat → its position determines where to look → read dog |
| **2. Two-hop attribute lookup** | _"What colour are the shoes of the person holding the red umbrella?"_     | umbrella → person → shoes                                   |
| **3. Pointer chase** ⭐         | sign $A$ reads "go to $B$"; sign $B$ reads "go to $C$"; $C$ is the answer | **a linked list rendered as an image**                      |
| **4. Stateful counting**        | _"How many people wear hats?"_ in a 30-person image                       | must remember where it already counted                      |

**Pointer chase is the best instrument** because:

- $k$ is tunable → gives an accuracy-vs-chain-length curve
- parallel execution is _provably_ impossible
- decoy signs kill the saliency shortcut
- synthetic → complete ground truth for every step
- no confound from world knowledge or language priors

Level 4 stresses a **different** axis (memory, not dependency). Both are needed.

### 9.4 The four-point checklist

Every candidate task must satisfy **all four**:

| #   | Requirement                                                                    | Fails if                                       |
| --- | ------------------------------------------------------------------------------ | ---------------------------------------------- |
| 1   | **Referential chain** — cannot pre-compute where to look                       | open-loop suffices                             |
| 2   | **Resolution gap** — a low-res glance cannot answer                            | one glance suffices → layer 1 shortcut returns |
| 3   | **Low blind baseline** — remove the image, accuracy collapses                  | the LLM prior answers it                       |
| 4   | **Area ratio** $\dfrac{\text{image area}}{k \times \text{glimpse area}} \gg 1$ | random raster scanning wins                    |

**Miss one and the latent reverts to a placeholder.**

> Requirement 3 deserves emphasis: **the LLM prior is the strongest shortcut of all, and no architecture can close it.** _"What shoes is the umbrella person wearing?"_ → "sneakers" is a decent statistical guess. Always report the **blind baseline**; without it the numbers are meaningless.

---

## 10. Evaluation protocol

Replicating the ICML causal mediation analysis, but on a **measurable** mediator. This is the transferable contribution: a protocol others can reuse regardless of whether any particular architecture wins.

| ID     | Test                                                                                                                   | Measures                       | Pass condition                                                                                       |
| ------ | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------ | ---------------------------------------------------------------------------------------------------- |
| **E1** | **Counterfactual relocation** — move the target from $A$ to $B$, measure $\Delta$gaze                                  | $\alpha$ (Input→Latent)        | gaze displacement correlates with object displacement                                                |
| **E2** | **Gaze intervention** — $do(\text{gaze} \leftarrow \text{random})$ and $do(\text{gaze} \leftarrow \text{oracle bbox})$ | $\beta$ (Latent→Answer)        | random collapses accuracy; oracle raises it. **Oracle − learned = headroom, reportable as a number** |
| **E3** | **Gaze dispersion** — $\mathrm{Var}_t[\mu_t]$                                                                          | homogeneity                    | non-zero; the model looks at different places over time                                              |
| **E4** | **Faithfulness** — feed patches from elsewhere without changing the latent                                             | is gaze causally used          | answer changes                                                                                       |
| **E5** | **Parallel-$k$ vs Sequential-$k$** ⭐                                                                                  | **closed-loop or not**         | sequential > parallel at matched reads and FLOPs                                                     |
| **E6** | **Content intervention on recurrence** — $do(\rho_{t-1} \leftarrow \text{other patch})$, does $\mu_t$ change?          | closed-loop                    | $\mu_t$ changes                                                                                      |
| **E7** | **Chain-length scaling** ⭐                                                                                            | compositional depth            | the gap over baselines **widens** as $k$ grows                                                       |
| **E8** | **Glance-resolution sweep**                                                                                            | the shortcut hypothesis itself | as glance resolution drops, $\beta$ **rises**                                                        |

**E5 is the cleanest single experiment in the set.** Same number of reads, same FLOPs, same information budget — the _only_ difference is whether the model may condition on what it saw. Any performance difference is reasoning, uncontaminated.

**E7 is the strongest possible evidence**, because it is not merely _"we win"_ but _"we win more as the task gets deeper"_ — which the shortcut hypothesis cannot explain away.

**E8 is the experiment that tests the theory rather than the model.** If $\beta$ rises exactly as the glance degrades, the shortcut account is confirmed as _predictive_, not post-hoc.

---

## 11. Why routing perception through the latent fixes both layers

### Layer 1 — path B is deleted

$$\text{full-resolution information is \textbf{not in the context at all} until } z \text{ requests it}$$

Not a penalty, not a regulariser — **the shortcut does not exist**.

### Layer 2 — the latent gains real information

Return to §1.3: the problem was $I(Z;Y \mid \text{prefix}) = 0$. But if $z$ decides where to read:

$$\text{read} = \rho\big(x,\ \mu(z)\big) \qquad\Longrightarrow\qquad I(Z,;,Y \mid \text{glance})\ >\ 0$$

Because what gets read depends on where it looks, and where it looks depends on $z$, **$z$ now determines what information enters the model.**

$$\text{a latent that only thinks} = \text{extra compute} \qquad\qquad \text{a latent that \textbf{acts}} = \text{genuinely new information}$$

> **The one-line summary of this note**: latent thought becomes meaningful only when it can _act_. Thinking that changes nothing is, information-theoretically, sitting still.

### And the mediator becomes measurable

$$4096 \text{ opaque dimensions} \ \longrightarrow\ 2 \text{ dimensions you can plot on the image}$$

Every previously unanswerable diagnostic question becomes answerable:

| Question                               | In latent space | With gaze                      |
| -------------------------------------- | --------------- | ------------------------------ |
| How do I perturb it meaningfully?      | unknown         | **move the fixation point**    |
| What does "correct" look like?         | unknown         | **matches the bounding box**   |
| What does collapse mean geometrically? | unknown         | **always looks at the centre** |

Ground truth also becomes free: bounding boxes already exist in most datasets, giving a dense auxiliary loss with **no teacher model and no distillation**.

---

## 12. The threat model — what must be conceded

### Threat 1: CapImagine already beats latent methods with plain text

The honest counter is **not** "text is worse." It is:

$$\text{CapImagine wins because it is \textbf{verifiable and grounded} — not because it is \textbf{text}}$$

|                        | Verifiable | Grounded | Cost            |
| ---------------------- | ---------- | -------- | --------------- |
| latent vector          | ✗          | ✗        | cheap           |
| **coordinate $(x,y)$** | **✓**      | **✓**    | **2 numbers**   |
| text imagination       | ✓          | ✓        | **~200 tokens** |

**A coordinate is the minimal symbolic interface**: it delivers both properties at ~1% of the token cost. And spatial position is precisely what language describes badly (_"upper right, a bit toward the middle"_).

### Threat 2: the shortcut can silently return

Any of the four checklist violations in §9.4 restores it. In particular, if the low-res glance can answer, $\beta \to 0$ again — **the architecture does not save you from a bad benchmark**.

### Threat 3: this space is crowded

V\*/SEAL, DyFo, ZoomEye, CVSearch, BVS, FOVEA all do visual search. **The distinction to state clearly**:

$$\text{Theirs: discrete tool calls mediated by language} \quad|\quad \text{Ours: latent emits coordinates, differentiable end to end}$$

Existing work depends on hand-designed tools and does not learn internal visual abstractions — that is the gap.

---

## 13. Status: Open / Pending / Closed

### 🔴 Closed

| Topic                                                      | Status                                                                           |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------- |
| "does latent CoT work in text"                             | answered yes; surveys exist (arXiv 2507.06203)                                   |
| "add latent visual tokens to a VLM and collect VQA points" | **saturated** — Mirage, LVR, ILVR, Monet, LanteRn, DeepLatent, OneLatent, UniVLR |
| "do current latent visual tokens carry visual semantics"   | **answered NO** — ICML 2026, causal mediation                                    |
| latent as pure efficiency (fewer tokens than explicit CoT) | done — CODI, 3.1× fewer tokens at matched GSM8K                                  |

### 🟡 Pending

| Topic                                                             | State                                                              |
| ----------------------------------------------------------------- | ------------------------------------------------------------------ |
| **how to make latent visual tokens actually load-bearing**        | **wide open — this is the crack**                                  |
| RL on latent thought via Gumbel/superposition                     | active — Latent-SFT, Latent-GRPO, SofT-GRPO                        |
| interpretability of latent reasoning                              | early — probing, logit lens, causal analyses; no accepted protocol |
| whether depth recurrence is "reasoning" or "iterative refinement" | contested — Huginn probing studies                                 |
| safety/faithfulness of continuous thought                         | barely started                                                     |

### 🟢 Open

1. **Architectures where the latent is load-bearing by construction**, not by supervision. Everyone currently uses fix 1 (teacher supervision); almost nobody uses fix 3 (delete the shortcut).
2. **A verifiable evaluation protocol for latent visual reasoning.** §10 is a candidate. This survives even if a given architecture fails — the most durable contribution available here.
3. **Deep-hard vision benchmarks.** Vision has essentially no analogue of GSM8K. Building one is itself a contribution.
4. **A "visual CoT" ground truth.** Text latents work partly because explicit CoT supervises them. What is the vision equivalent? Gaze traces? Human eye-tracking? Programmatic sub-goal traces?
5. **Quantifying the shortcut.** Nobody publishes $\beta$ as a function of task depth, glance resolution, or model scale. E7/E8 are unclaimed.

---

## 14. Reading order

1. **Hao et al. 2024, Coconut** — the origin of hidden-state injection
2. **Shen et al. 2025, CODI** — self-distillation variant; shows what supervising a latent actually requires
3. **Geiping et al. 2025, Huginn** — depth recurrence; latent reasoning emerging zero-shot
4. **Yang et al. 2025, Mirage** (arXiv 2506.17218) — origin of latent _visual_ reasoning
5. ⭐ **Li et al. 2026, _Imagination Helps Visual Reasoning, But Not Yet in Latent Space_** (arXiv 2602.22766) — **the most important paper for this line of work**; code at github.com/AI9Stars/CapImagine
6. _What's Holding Back Latent Visual Reasoning?_ — latent collapse
7. _Beyond Visual Memory: Mechanistic Diagnostics of Latent Visual Reasoning_ (arXiv 2606.01287)
8. **Zhu et al. 2025, _A Survey on Latent Reasoning_** (arXiv 2507.06203) — the map

---

## 15. The five sentences to remember

1. **Latent space is the same space with the vocabulary constraint removed** — a negative definition. You gain bandwidth and lose anchoring, supervision, and measurability.
2. **A latent that is a deterministic function of the prefix adds compute, never information.** Placeholder behaviour is the mathematical default, not a bug.
3. **Visual tokens are sequence positions, not architectural layers.** They merge at the projector, before layer 1, and information leaves them only through attention.
4. **The shortcut $v \to \text{ans}$ is one hop, pretrained, and available at every layer**, while $v \to z \to \text{ans}$ is two hops starting from random. B wins, and the optimiser is right to prefer it.
5. **Anything avoidable will be avoided** — latents, glimpses, CoT, layers, auxiliary heads. The only robust fix is to make the mechanism unavoidable: **delete the alternative path, and choose a task that cannot be answered in one hop.**
