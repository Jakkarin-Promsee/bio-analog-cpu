# ⚑ the thought — the standing candidate (primary source)

> **This file is the draft's position.** Every other document in `draft7.0/` points here. Adopted 2026-08-06 as the
> **FIRST decision, not the final one** — held so that research has something to attack:
>
> > เหตุผลคือ มันจะเถียงกันไม่ได้ ถ้าไม่มี candidate ที่เป็นภาพใหญ่จริงๆ · หัวข้อที่ใช้ตอนนี้ **ไม่ใช่ final decision แต่มันคือ first
> > decision** ก่อนกุจะไปทำ research เพื่อหา candidate มาสู้มัน
>
> `[STANDING]` — **evidence zero.** Replaced by a **better candidate**, never by an argument. Bring a rival big
> picture; an objection to a part is a note, not a replacement.

> Written 2026-08-05, the day after the turn, because re-reading [`context.md`](context.md) and
> [`handoff.md`](handoff.md) showed the author had **only told half of his picture**. The half that was missing is
> the one that says *what a thought is made of* and *what shape the loop actually has* — without it a future agent
> reads "loop until the feeling says stop" and invents its own loop.
>
> **This file is `[AUTHOR]`.** That tag is new and it means: *this is the target definition, in the author's own
> words.* It is **not evidence** — nothing here is measured — but it is also **not the agent's to overturn by
> argument.** An agent may say "this is hard to build" or "the literature calls this X"; it may not say "actually
> thought is probably Y" and proceed. Only the author revises this file.
>
> Thai is kept verbatim where it is load-bearing. Paraphrase is how a thesis gets quietly rewritten.

---

## 1. The root: the "right feeling" and the transformer vector are the same idea

The author's opening move, and the reason this file exists:

> right feeling ที่กุตามหา มันคือความสิ่งที่คล้ายคลึงกับ transformer vector มากๆเลยอ่ะ (เพราะทั้งคู่มันมีรากฐานชี้ไปทางเดียวกัน)
>
> สิ่งที่เกิดขึ้นจริงๆก็คือ เเต่ละ word หรือ image จะมี vector เฉพาะของมันอยู่ ที่สามารถบวกหรือลบกันได้ เพื่อเกิดความหมายใหม่ๆที่ใกล้เคียงกัน

`[AUTHOR]` The thing being searched for is **not** a mysterious new quantity. It is a **composable representation**:
meanings that add and subtract, where arithmetic on the representation lands you on nearby meanings. That is the
same root transformers stand on — which is why the feeling is reachable at all, rather than being a mystical
primitive the project has to invent from nothing.

**Why this matters more than it reads.** Everything in [`open-questions.md`](open-questions.md) axes B–D was
written as though "the feeling" were an unknown *scalar*. Under this framing it is a quantity over a
**composable space**, and composability is what makes the rest of the loop mechanically possible (see §6).

---

## 2. What "an orange" actually is: a bundle, and a label that compresses it

> สิ่งที่เกิดขึ้นจริงๆ ตอนกุคิดถึงผลส้ม จริงๆมันคือ ผลไม้ + ทรงกลม + สีส้ม + รสเปรี้ยว + ด้านในมีเมล็ด + etc. โดยที่คำว่าผลส้ม เป็นสิ่งที่ใช้ระบุตัวตนของมัน
> เเบบโดยรวม เป็นเหมือนภาษากลางของโลกอ่ะ เราเรียกส้มทุกผลว่าส้ม ทั้งๆที่จริงๆมันอาจจะเป็นแมนดาริน หรือเลมอนสีส้มก็ได้ เเค่คำว่าส้มมันตรงที่สุดใน
> context ตอนนี้ ที่กุไม่ได้อยากระบุมันละเอียดขนาดนั้น

`[AUTHOR]` Three separate statements are packed in here, and each one constrains the design:

1. **A concept is a bundle of attributes, not an atom.** *orange = fruit + round + orange-coloured + sour + seeds
   inside + …* The bundle is the real object; it is what gets compared.
2. **The word is an identifier, not the content.** "Orange" is a **shared world-language compression** over the
   bundle — it exists so two people can point at the same region without enumerating it. The loop operates on the
   bundle; the word is a handle.
3. **Resolution is on demand.** Every orange we call "orange" might be a mandarin, or an orange-skinned lemon.
   "Orange" is simply *the most accurate label at the resolution the current context asks for*. You match something
   orange-like, you call it an orange, and you **drill in only if the context demands it.**

Point 3 is not a caveat — it is the mechanism of §4. Under-specifying on purpose is how the loop stays cheap.

---

## 3. The definition of *thought*: a thesis with three parts

`[AUTHOR]` — the author's own words, restructured only for layout:

> thought ทำงานคล้าย thesis มี 3 ส่วน

| Part | Thai | What it is |
| --- | --- | --- |
| **Goal** | เป้าหมาย | what we are currently trying to prove — *"is the thing in front of me an orange?"* |
| **Fact** | fact | everything established so far: **perceived** (it is round, it is orange, it is a fruit) **and derived by deliberation through memory** (definitely not an apple → cut every non-fruit and everything red) |
| **Memory output** | สิ่งที่อยู่ในความทรงจำ | the output of **f(goal, fact)** — what memory returns when queried with the current goal under the current facts |

Three things follow immediately, and all three are load-bearing:

- **The fact register is a first-class object.** It is neither the goal nor the memory. It **accumulates across
  iterations**, and it holds *both* positive evidence and eliminations in the same place. Nothing in draft 6, and
  nothing in [`ideas.md`](ideas.md), has this component.
- **Retrieval is conditioned on both.** Memory is queried as `f(goal, fact)` — **not** `f(input)`. A retrieval
  system keyed only on the input cannot implement this, because the same input must return different candidates on
  iteration 1 and iteration 5.
- **A "thought" is the whole triple**, not the retrieved item. When the author says *thought is weight*
  ([`open-questions.md`](open-questions.md) axis A), the object he is describing is this triple in flight — not the
  candidate that came back.

---

## 4. The loop of thought: two motions, not one

`[AUTHOR]`

> loop of thought ก็คือการวนกันระหว่าง เป้าหมาย, fact, เเละสิ่งที่อยู่ในความทรงจำ

**Motion 1 — horizontal (same goal): accumulate and prune.**

> เป้าหมายเดิม จะ loop เพิ่ม fact เเละตัด choice ไปเรื่อยๆ จนกว่า output จากความทรงจำจะ match กับ fact

The goal is held fixed. Each iteration adds facts and eliminates candidates. It runs **until the memory output
matches the facts.**

**Motion 2 — vertical (on match): narrow the goal.**

> เมื่อมัน match กันเเล้ว ต่อไปก็คือการบีบ scope ของเป้าหมายให้เเม่นขึ้นไปเรื่อยๆ

A match does **not** end the process — it ends *this rung*. The goal's scope then tightens: *is it an orange?* →
*what kind of orange?* The loop restarts at higher resolution with the fact register carried forward.

> **This is the structural correction to the existing draft-7 documents.** Every one of them describes only motion 1
> (retrieve → compare → reject → re-query). Motion 2 — *the goal is itself rewritten on success* — was never
> written down, and it is what makes the thing a loop of **thought** rather than a search with a stopping rule.

---

## 5. The ladder rule: **you never ask the final question**

`[AUTHOR]` — the sharpest constraint in this file, and the easiest one for a future agent to throw away.

Asked *"is that a mandarin?"*, the model must **not** take "mandarin" as its goal and go hunting.

| The wrong shape | The right shape |
| --- | --- |
| goal := *mandarin?* → search memory from the definition of "mandarin", with no ground truth to prune against | goal := *is it a fruit?* → **match** → goal := *is it an orange?* → **match** → goal := *is it a mandarin?* |
| the right feeling is nearly unreachable — there is nothing to compare against early | each rung's match hands the next rung a pruned space and a loaded fact register |
| **this is brute-force search over big data.** It is not thinking, even when it returns the right answer | this is a loop of thought |

> ถ้าค่อยๆถามไปว่า มันเป็นผลไม้มั้ย? มันเป็นส้มมั้ย? มันเป็นส้มเเมนดารินใช่มั้ย? มันจะเป็น loop of thought ของจริงเลย

**The author pre-empts the obvious objection himself:**

> กุสามารถบอกได้เลยอ่ะว่าในงานง่ายๆ มันจะดู inefficient มากๆ เพราะเราสามารถหาเเมนดาริน ตรงๆจาก 1 loop of thought ได้เลย เพราะมันไม่ซับซ้อนขนาดนั้น
> เเต่ต้องคิดภาพว่าตอนนี้เรากำลังตั้งรากฐานของโมเดลอยู่ เป้าหมายจริงๆของมันไม่ได้เอามาเเข่ง transformer ใน field vector operation

**Yes, it is inefficient on easy tasks — on purpose.** The ladder is not an optimisation, it is the foundation. The
target is not to beat transformers at vector operations; it is to build a thing that actually *thinks*, so that it
later carries load a single-shot answer cannot: the author's own example — *watch a moving object, connect it to a
law of physics, predict where it goes.*

**⚠ Anti-fraud consequence.** [`ideas.md`](ideas.md) §6 criterion 3 currently reads *"step count correlates with
input difficulty."* Under the ladder that is **wrong as stated**: an easy input asked a fine-grained question still
costs several rungs. The corrected form: **step count tracks the distance from the opening coarse goal to the
resolution the question demands.** An agent that benchmarks against the old wording will conclude a correct ladder
is broken. `[ARG]`

---

## 6. What this means SCFF actually was

`[AUTHOR]` — the deeper version of [`context.md`](context.md) §4. That section established goodness is *unary,
therefore not the feeling*. This one says what the machine was doing instead.

> สิ่งที่ SCFF ทำจริงๆเเม่งคือการ compare ระหว่าง x_เป้าหมาย กับ x_noise ผ่านสมการ goodness เลยเว้ย … มันคือตัวขยายสัญญาณความเเตกต่างระหว่างสิ่งของ
>
> สิ่งที่ SCFF ทำคือ compare(สิ่งของ, noise) เเต่สิ่งที่เราตั้งไว้ตอนเเรกคือ (สิ่งของ, สิ่งของ) 💀💀

**This reading checks out against the committed design.** `[CARRIED]` The frozen cell is `SCFFContrastOverlap`
(InfoNCE, **two masked views of the same sample**, temp 0.2, w=2) plus Phase 6's `NoiseAugContrast` (iid σ=1.0). So
the positive pair is **a sample against a corrupted copy of itself**, and the negatives are **whatever else was in
the batch / LUT pool, class-blind**. The objective is literally *"stay yourself under corruption, differ from the
background"* — a **difference amplifier**, exactly as stated. It never compared two meanings.

**And the project already tested the fix, and it did nothing.** `[CARRIED]` — the result the author remembered but
could not place: **P2.2 / phase2 exp2, the `hard_oracle` arm** — the negative is *a different **true-class**
sample*, chosen with perfect labels. That is `compare(thing, thing-of-another-class)`, handed over for free.

| arm | depth-slope (CIFAR-flat, n=5) |
| --- | --- |
| random negative | −0.020 |
| **hard-oracle (different true class)** | **−0.022** |
| prototype | −0.015 |
| *synth control, where classes really are the clusters* | *+0.027 (real lift)* |

Making the comparison class-aware — perfectly, with an oracle — bought **nothing**. The synth control proves the
machinery was fine; the comparison was simply over the wrong thing.

*Scope, stated honestly so nobody overstates this later:* P2.2 ran inside the **energy-goodness** family, and Phase
3 later found that the **objective family** was the lever (energy → InfoNCE contrast). That does not weaken the
point — it strengthens it. **What finally made SCFF work was a sharper self-versus-corrupted-self discriminator,
never a better comparison between two real things.** SCFF's competence has never once come from comparing meanings.

**The author's verdict:**

> เหมือนเราได้ระบบสายไฟที่โคตรโหดมาอ่ะ เเต่เเม่งรถมึงล้อกับเครื่องยนต์ยังไม่มีเลย 😂 … เราทำมันผิดขั้นตอน เราต้องไปหา loop of thought ให้ได้ก่อน
> ที่จะมาร่าง network properties

An excellent wiring harness on a car with no wheels and no engine. The order was wrong: **find the loop of thought
first, then draft the network properties.** `[AUTHOR]` — this is the same conclusion as the turn, arrived at from
the mechanism side instead of the metric side, which is why it is worth having both.

---

## 7. What this file changes in the rest of draft 7.0

| Document | Change |
| --- | --- |
| [`open-questions.md`](open-questions.md) **axis D** (query update under rejection) | Gains a concrete candidate — see §8.1 below. Still `[OPEN]`. |
| [`open-questions.md`](open-questions.md) **axis A** (state vs weight) | Sharpened, not closed: the object in dispute is the **triple** (goal + fact + memory output), and the fact register is the part that accumulates. |
| [`open-questions.md`](open-questions.md) **new axis H** | Where does the coarse→fine ladder come from? The load-bearing hole — see §8.2. |
| [`ideas.md`](ideas.md) §6 criterion 3 | Restated — step count tracks goal-resolution distance, not raw input difficulty (§5). |
| [`ideas.md`](ideas.md) §4 "the named gap" | Still the gap, but the loop now has **two** update rules, not one: the query update under rejection *and* the goal update on match. |
| [`handoff.md`](handoff.md) | New tripwire #10 — "just ask the target question directly." |
| [`context.md`](context.md) §4 | Unchanged and still correct; §6 here is the mechanism-side version of the same finding. |
| **[`open-questions.md`](open-questions.md) axis A — superseded** | §9.4: it is **not** a two-way state-vs-weight contest. Three-plus candidates, and the author holds none firmly. Do not "resolve" it by picking a side. |
| **[`open-questions.md`](open-questions.md) axis D — widened** | §9.3: the author did **not** confirm attribute-subtraction. Two families must both be surveyed — a subtraction on the query, *or* an operation that lets `f(goal, fact)` sharpen its own scope. |
| **[`open-questions.md`](open-questions.md) new axis J** | §9.2: what input substrate does the first model get built on? Frozen-CNN is an idea, not taken — and it carries the generality-vs-similarity trap. |

---

## 8. What this file opens

### 8.1 Rejection is probably attribute subtraction — `[CLAIM]`, agent inference, **needs the author's confirmation**

Not the author's words. It is what §1 + §2 imply, and it is the largest payoff if it holds.

If a concept is a **bundle of attributes** (§2) in a space where meanings **add and subtract** (§1), then *"plus
it's not a tone of red apple"* is not bookkeeping — it is **arithmetic on the query**: subtract the *attribute* that
made the candidate fail (red-ness), not the *instance* (apple).

That is exactly the "constructive rejection" that [`open-questions.md`](open-questions.md) axis D demands: removing
an attribute removes **a whole region** of the space, while removing an instance removes one item and degrades the
loop into a linear scan. The author's framing supplies the mechanism the axis was missing — *if* the attribute
that caused the failure can be identified, which is itself unsolved.

### 8.2 Where does the ladder come from? — `[OPEN]`, the load-bearing hole

The ladder *fruit → orange → mandarin* has to come from somewhere, and the two possible answers are not equally good:

- **Stored as a taxonomy** → the loop is a decision-tree walk, and the intelligence lives in the tree, not the loop.
  **A tree also passes the k-vs-N sub-linearity test** ([`ideas.md`](ideas.md) §4.2) while being nothing new. This
  is a live way to fake a result.
- **Emergent from retrieval** → coarse concepts win early because they have the most support in the fact register;
  finer ones only win once enough facts accumulate. The ladder is then a *consequence* of the dynamics, not a
  stored structure. Motion 2 (§4) would generate the next goal from the **residual**: the attributes of the input
  that the matched concept fails to account for.

The second is the only version that is a research contribution — and **draft 6 gives no help here.** `[CARRIED]` C1:
SCFF clusters by **density, not class**; density is not generality either, so nothing measured in eleven phases
produces a "fruit-level" attractor. Whatever supports the coarse rung has to be built, and it is not in the repo.

### 8.3 Does the goal's scope set the acceptance threshold? — `[OPEN]`

In the founding story the feeling comes back as *"similar, not sure"* — enough to answer *orange* casually, not
enough under pressure. Under §4 that is not one absolute threshold at all: **the goal's current scope sets the
resolution the match must meet.** The same match strength passes at rung "is it an orange?" and fails at rung
"which orange?".

If so, [`ideas.md`](ideas.md) §6 criterion 6 ("the halt is calibrated in absolute value") is misframed — calibration
would be **relative to goal scope**, and an absolute-threshold halt is the wrong object.

---

## 9. The open ends — the author's own answers (2026-08-06)

The three questions in §8 went back to the author. **Two of them he answered "I don't know", and that is recorded
as the finding it is** — it marks what is genuinely open, instead of letting an agent's guess harden into doctrine.

### 9.0 SCFF, final statement — and the one job it keeps

`[AUTHOR]`

> สิ่งที่ SCFF ทำไม่ใช่การเปรียบเทียบ เพื่อดูว่าของสองสิ่งมันคล้ายจริงๆมั้ย มันแค่เป็นสมการที่บอกว่า weight มันจะ encourage กันยังไงให้มันแยก noise
> กับของในแต่ละชิ้นได้ เพราะแบบนั้น thesis ตั้งต้นเราเลยผิดหมดเลยบน illusion ตัวนี้

SCFF is **an equation describing how weights encourage each other so that an item separates from noise.** It is not
a comparison of two things in any sense the project needs. **The founding thesis was built on that illusion.**

**But it is not useless — it has a demoted, later job.** `[AUTHOR]`, explicitly *not now*:

> เราไม่สามารถใช้ SCFF มาทำสิ่งที่เราต้องการได้ในตอนนี้ แต่ว่าเราใช้มันเป็นเหมือน training model ที่เป็นส่วนประกอบแทนได้ (เช่นเอามาใส่แทน
> backprop แค่เราต้องมีตัว SLDA เป็น translator ไว้ (ตัวอย่างเฉยๆ ไม่ทำตอนนี้ รอได้ final ของ loop of thought แบบไม่มี constrain หน่อย))

SCFF's future role is **as a component training method** — a drop-in where backprop would otherwise sit, with SLDA
as the translator. ⚠ An example, **not a plan**, and gated: **only after** an unconstrained loop-of-thought final
exists. Proposing it before then is tripwire #1.

### 9.1 Where the ladder comes from — `[OPEN]`, no research done yet

`[AUTHOR]` — the honest starting position: *"กุยังไม่ได้ไป research จนเห็นทุกทางเลย เอาจริงๆ ยังไม่ได้เริ่มอ่านอะไรเลย"*

What he knows he must produce: **an abstract ladder, or the vector of one.** Model undecided. His current idea:

> กุคิดว่าเราทำ softmax ให้แต่ละ ladder ได้อ่ะ ความเป็นจริง คำว่าผลส้ม มันก็เชื่อมอยู่กับทุกคำอยู่แล้ว เราสามารถ ranking ได้ว่าแบบ simple ที่สุด
> mandarin มันชี้ไปไหน

**A softmax over the rungs** — since "orange" is already connected to every other word, you can rank, in the
simplest form, where "mandarin" points.

**The problem he names himself:** *"กุไม่ได้ทำโมเดลทางภาษาด้วยอ่ะ"* — the ranking story is native to language models,
and this is not one. Which opens the substrate question below.

### 9.2 The input substrate: **images, and the CNN as a decomposer** — `[OPEN]`, idea, not taken

`[AUTHOR]`, tagged by him *"ยังไม่ take แค่เป็น ideas"*.

**The role, stated precisely — this is not the foundation of the model.** An agent first read it as "build the model
on CNN features" and objected on those grounds; the author corrected it:

> freeze CNN มันทำงานเป็น abstract extractor ได้ เราไม่ได้ใช้มันเพื่อเป็นรากฐานโมเดลหลัก แต่เราใช้มันเป็นตัว generate data ให้ thought
> เราสามารถทำงานได้ … แบบที่กุจะใช้จริงๆเลยคือ CNN แยก 1 image ให้กลายเป็นหลายๆ component เฉพาะ ที่มันมีการเชื่อมโยงถึงกันได้

The CNN is a **sensor / data generator**: one image → **many specific components that can link to each other.** The
loop is built on the components, not on the CNN. His stated analogy is how an LM uses a mountain of book data to
discover how each word behaves as a vector alone and in composition.

**Why images and not text** `[AUTHOR]`:

> คำว่าแอปเปิล มันตายตัวมากๆทางภาษา … ที่มันยากจริงๆของโมเดลทางภาษาคือการต่อคำทุกคำเข้าด้วยกัน ให้เป็นประโยคที่มีความหมาย
>
> ในขณะที่ข้อมูลรูปภาพแม่งโคตร flexible เลย เอาแค่ CNN หรือ resNet ง่ายๆ มันก็คือการ stacking abstract ผ่าน convolutional
> len + pooling แล้วอ่ะ … แถมเราสามารถพัฒนาเป็น video ในอนาคตได้ด้วย เพื่อทำให้ thought มันมีข้อมูลที่ complex ขึ้น

Text fixes "apple" rigidly; the genuine difficulty in language modelling is composing words into meaningful
sentences. An image is flexible, a conv stack **is already** abstract-stacking (conv lens + pooling), and it extends
to **video** later — which is where thought gets complex enough to matter.

**Agent assessment — the objection above is withdrawn, and the choice is stronger than he argued it.** `[CLAIM]`

The sharper version of his own reason, and it falls straight out of §2: **text hands you the handle and hides the
bundle.** "Apple" is precisely the shared-world *label* that §2 says is an identifier, not the content — language
already performed the compression and threw the attributes away. To recover *round, red, sweet, has seeds* from the
token you must reconstruct, from co-occurrence over billions of tokens, the very bundle language deleted. **An image
hands you the bundle before it was ever named.** Roundness, colour and texture are physically present in the
pixels. For a loop that operates on bundles and treats the word as an optional handle, that is not a preference —
**it is close to the only substrate where the object of study exists pre-compression.**

And it defends against this draft's own tripwire: in text the generality hierarchy is **given** (hypernymy, WordNet,
co-occurrence), so a ladder would be handed to him and #11 fires. In images nothing hands it over — which is the
whole reason a ladder that emerges there would be a **result**. His instinct to avoid text and the agent's
anti-fraud tripwire point the same direction for different reasons.

**What survives of the objection, relocated.** `[CLAIM]` Two costs, both real, neither fatal:

1. **"Many components" is not free from a frozen CNN.** Conv channels are **entangled and polysemantic** — a channel
   is not an attribute. Getting *one image → several separable components* is itself an open research area, and it
   is where the reading should go: disentangled representation learning (β-VAE, FactorVAE), sparse
   autoencoders / dictionary learning for feature decomposition, and object-centric methods (Slot Attention, MONet).
   ⚠ **Object-centric ≠ attribute-centric.** Slot attention factors a scene into *entities* (this apple, that
   table); §2 needs factoring into *properties* (round, orange, sour). Reaching for slots gives the wrong axis.
2. **Use a self-supervised backbone, not a label-supervised one.** An ImageNet-classification CNN's features are
   shaped by ImageNet's label tree — **WordNet, a hierarchy built by hand.** Any ladder found on top of it may have
   been inherited rather than generated. A self-supervised backbone (DINO / MAE / SimCLR) removes that channel of
   contamination at no cost. Cheap insurance; take it.

**⚠ The one that matters most — do not let "linked" quietly become "fire together, wire together" again.**
`[CLAIM]` The phrase *"component เฉพาะ ที่มันมีการเชื่อมโยงถึงกันได้"* leaves the **link relation** undefined, and it is
carrying enormous weight — the link structure *is* the memory. If "linked" ends up meaning *co-occurred in the same
image*, the mechanism is **correlation**, and §6 is the whole lesson about where that leads: correlation-based
association amplifies difference, it does not compare meanings. That would reproduce SCFF's illusion one level up,
on a new substrate, after all of this.

The distinction that avoids it, and it is exactly VSA's: **bundling** (superposition — what co-occurrence gives you)
is *not* **binding** (role–filler attachment). Binding is what makes *orange-coloured* attach **to this fruit**
rather than merely occur alongside it. Co-occurrence gives bundling for free and **never gives binding.** If the
components are to compose, the binding operation has to be chosen deliberately.

**And the candidate that ties §9.2 back to the ladder** `[CLAIM]`: in VSA / hyperdimensional algebra **superposition
is the generality operation** — bundle many mandarins and only what they share survives; bundle wider and you climb
to *orange*, then *fruit*. **Generality = bundling depth**, over any vector source, including a frozen CNN's
components. It gives axis H's emergent answer for free: a heavily-bundled vector matches more inputs, so **the
coarse rung wins early because it is coarse** — no stored tree required. Unverified; VSA/HDC is a known hole in this
repo's reading and belongs at the top of the commissioned checklist.

**The known boundary of this substrate, stated now so it is not discovered late** `[CLAIM]`: images ground
*perceptual* thought — the orange-car story runs. They ground **nothing** for the *q = mc∆t* case, where the feeling
comes from consistency among memories with no sensory input pinning either side (axis B's unresolved half). The
first substrate covers the first of the author's two founding examples and not the second. That is an acceptable
scope for a first build, not a defect — but it must be said out loud.

**Video** is the right destination and the wrong start: the author's own §5 example (*watch a moving object, connect
it to a law of physics, predict it*) **is** video, which is what justifies choosing images at all. It is also
expensive, and there is no GPU. Static images carrying several attributes is the notebook-sized version of the same
thing.

### 9.3 Rejection — `[OPEN]`, the operation is believed to exist, its form is unknown

`[AUTHOR]`:

> กุไม่ได้เก่ง transformer ขนาดนั้น แต่กุคิดว่ามันมีoperation ในการลบแน่ๆ ไม่ก็ operation ในการทำให้ f(เป้าหมาย, fact) มันสามารถ scope
> ตัวเองให้แม่นขึ้นได้

He did **not** confirm §8.1's attribute-subtraction reading. What he commits to is weaker and broader, and the
breadth is the point — **two distinct families are on the table:**

1. **a subtraction operation** on the query, or
2. **an operation that lets `f(goal, fact)` sharpen its own scope**

Family 2 is not a variant of family 1. Subtracting from the query and tightening the function that reads memory are
different machines with different costs. §8.1 stays `[CLAIM]`, unconfirmed, and **axis D must survey both.**

### 9.4 What a thought is made of — `[OPEN]`, three live candidates

`[AUTHOR]` — *"กุยังไม่แน่ใจเหมือนกัน ว่ามันควรทำงานเป็น pure weight หรือใช้อะไร แบบในความคิดกุตอนนี้มีหลาย candidate มากๆอ่ะ"*

| # | Candidate | Note |
| --- | --- | --- |
| 1 | **thought = pure weight**, self-driving, continuously changing | the position recorded in axis A; nearest literature home is **fast weights** |
| 2 | **thought = fixed weight** running over a **context window of facts** that flows past | this is the §3 fact register as the moving part, with the machinery held still — closest to how a transformer treats context |
| 3 | **thought = an instruction** | least developed; a thought as a *program to run* rather than a thing to hold |

**This supersedes the framing of axis A.** That axis was written as a two-way contest (fast state vs fast weight)
between the agent and the author. It is **three-plus ways, and the author holds none of them firmly.** Anyone who
"resolves axis A" by picking a side has resolved a question that was never that shape. → survey material, not
argument material.

### 9.5 The pooling-slot architecture — `[AUTHOR]`, the first design with a shape

The substrate idea of §9.2, made concrete. **Do not use the whole CNN.** `[AUTHOR]`:

> เหมือนเราจะตัด CNN ส่วนครึ่งหลังออกไป ที่มีการ pooling โหดๆ จน component หลายตัวจากแต่ละ convolutional network มันเชื่อมกันได้อ่ะ
> กุเลยมองว่าแต่ละ convolutional layer แต่ละตัว มันทำหน้าที่เหมือน len เฉพาะจริงๆ ที่ใช้บอกตัวตนของสิ่งของสิ่งนั้นจริงๆเลย
>
> สิ่งที่ loop of thought model มันจะทำหน้าที่คล้ายๆตัว pooling เลย มันจะรวม abstract จาก component ต่างๆเข้าด้วยกันได้ และสุดท้าย
> ฝั่ง output มันก็สามารถเป็น convolution network ครึ่งหลังที่เราตัดทิ้งได้อ่ะ

```
image → [conv front half]  →  many components (multi-layer, multi-scale, spatially resolved)
                                        ↓
                            ⟨ LOOP OF THOUGHT ⟩          ← occupies the slot pooling used to hold
                                        ↓
                            [conv back half / head]  →  output
```

Each conv layer is **a specific lens that states the identity of the thing**, and the data is **denser and easier to
digest than language**. The heavy pooling of the back half is what destroys the linkage between components, so it is
cut; **the loop is what aggregates instead.**

**⚠ What this is FOR — the author's correction of an agent misreading, and it changes what counts as success.**
`[AUTHOR]` An agent first judged this design as *a task to beat* (does the loop match the CNN?) and warned it would
amount to an expensive pooling layer. That is judging the wrong thing:

> งานของเราไม่ได้ทำโมเดลแบบใหม่ ที่ดีที่สุดในโลก เราหา **"สนาม dataset ที่มีของมากพอที่เราจะจำลอง loop of thought ได้"** โดยเฉพาะ
> loop of thought ที่ไม่ได้มาจาก nlp
>
> loop of thought ของเราไม่ได้ทำหน้าที่เป็น pooling layer แต่เราให้มันได้ฝึกคิด **โดยที่เราไม่ใบ้อะไรให้มันเลย ไม่มี high abstract thing
> เหมือนภาษา แต่เป็นการป้อนตัวเลขทุกอย่างไปล้วนๆ เลย**
>
> เราไม่ได้หา pooling layer ที่ทำงานได้ดีกว่าเดิม เราหา model ไปอยู่ระหว่าง CNN ที่มันสามารถคิดเองได้ ที่เลือก CNN เพราะว่าเราสามารถดึงโมเดลให้
> complex มาก จนเกิด complex input/output ได้เฉยๆ

**This is an arena, not a benchmark.** The two CNN halves supply a rich **input space of numbers that already carry
meaning** and a rich **output space of numbers that already carry meaning**; the thing between them is given
practice at thinking. Beating pooling was never the goal, so "it is only as good as pooling" is not a refutation of
anything. **The arena is the deliverable of this stage.**

Two properties of the choice that an agent will under-weight:

- **It hints at nothing.** Language hands a model a pre-digested abstraction ladder for free — hypernyms, word
  senses, a whole human taxonomy baked into the tokens. Raw conv activations hand over **nothing**. If a loop of
  thought forms in there, nothing was given to it. This is the strongest form of the not-NLP argument in this
  document, and it generalises the WordNet warning of §9.2 from *the ladder* to *everything*.
- **Complexity is a dial.** CNN was chosen partly because the rig **scales**: a two-layer net on tiny images and a
  deep ResNet are the same rig at different settings, so the input/output spaces can be made as complex as needed.
  For an evening-pace solo researcher with no GPU, starting absurdly small and scaling **without changing rigs** is
  worth a great deal.

**Agent assessment: this has real potential, and it is the strongest structural idea in the draft.** `[CLAIM]`

1. **The analogy is not decorative — pooling and the loop are the same *kind* of operator.** Both take many local
   evidences and produce one summary. Pooling is fixed, input-independent, single-pass and lossy; the loop is
   query-conditioned, iterative, variable-length. **That is a real slot in a real architecture where this mechanism
   is the natural upgrade** — most novel mechanisms have nowhere to plug in, and this one does.
2. **The components genuinely have the §2 shape.** A conv feature map is `[C channels × H × W]`: channels at one
   location behave like *attributes of the same thing*, locations behave like *parts / where*. That is a bundle
   with a natural place for binding, handed over by the sensor.
3. **It supplies a baseline that does not import benchmark gravity.** Cut a pretrained net at layer *k*, put the
   loop in, keep the original head: the control is **the unmodified network itself**. That is an ablation, not a
   continual-learning benchmark — tripwire #3 never fires.
4. **The cut depth is a clean experimental axis** — cut early = more, rawer components and more work for the loop;
   cut late = fewer, more abstract ones. *How deep can the cut go before the loop can no longer recover the head's
   performance?* is a real curve.

**⚠ The concern that survives the correction above — stated properly this time.** `[ARG]` — the most important line
in this section.

It is **not** "the loop must beat the CNN"; that was the misreading. It is this: **the arena's default training
signal actively selects against looping.**

The only signal the rig hands you for free is *"produce the tensor the back half expects."* That is a supervised
regression target, and under it gradient descent will find the **cheapest** function that satisfies it — which is a
feedforward map. **A loop does not emerge from that objective; nothing in it rewards a second iteration.** The rig
is right; its default objective is not. Put another way: [`ideas.md`](ideas.md) §6 criterion 1 does not apply to the
*arena* — it applies to **the task run inside it**, and the arena's default task fails it.

**The fix lives inside the rig and costs nothing: limit the read bandwidth.** `[CLAIM]`

Let the loop see only **k components per step**, out of the `C × H × W` available. Then:

- **iteration becomes structurally necessary** — when the evidence an input needs exceeds *k*, a single pass
  **provably** cannot succeed; it is an information bound on reads, not a hoped-for difficulty
- **step count varies per input by construction** — an unambiguous image resolves in few reads, a cluttered or
  occluded one takes many (criteria 1, 2 and 3 satisfied at once, with no new dataset and no new task)
- and the three differentiators from attention stop being decorative and become **load-bearing**: you must
  **remember what you already read** (the fact register earns its existence), and **where to look next depends on
  what you just found** (the query update rule earns its existence)

It also happens to be the author's own story in hardware terms: you do not see everything at once — you look,
recall, look again.

**With that, the rig is both things at once:** an arena in which a loop can be watched running on dense
non-linguistic numbers *and* a place where single-pass failure is provable. Without it, it is an arena in which the
most likely outcome is a feedforward map wearing a loop's name.

**⚠ Second warning — the shape it collapses into if the loop is weak.** `[ARG]` Replacing pooling with an adaptive,
query-driven aggregation over spatial positions is, to first order, **attention** — which is what a Vision
Transformer already is. If the design cannot be distinguished from that, the honest outside verdict is *"you
reinvented attention with more steps."* The differentiators must be **load-bearing, not decorative**: variable step
count per input, an **accumulating fact register** across steps, and **rejection** changing the next query.
Attention has none of the three. If any of them can be removed without hurting the result, what was built is
attention.

**⚠ Third — the ladder still has no home here.** This design solves component extraction and gives the loop a slot;
it produces nothing resembling *fruit → orange → mandarin*. Classification hands you leaves. **Axis H is untouched
by this architecture**, and a nice-looking scaffold is exactly the kind of momentum that hides an open hole.

**One design decision to take deliberately rather than discover:** the cut-off back half was trained on a specific
layer's tensor shape and statistics. Either it is retrained (backprop in the loop — permitted during set zero, but
attribution gets muddy), or it is kept **frozen as a decoder/probe** and the loop must learn to emit into its input
distribution. The frozen option is the better one — it forces the loop's output to be *meaningful* rather than
merely convenient — but it is a real constraint on the loop's design, not a free choice.

**Two ways to make single-pass failure real inside this arena**, cheapest first `[CLAIM]`:

1. **Read-bandwidth limit** (above) — *k* components per step. Free, requires no new dataset or task, and forces
   the fact register and the query update to do real work.
2. **Serial dependency of lookups** — each retrieval's *address* is computed from the previous retrieval's
   *content* (*find the region whose attribute matches what was just found elsewhere*; multi-hop, hop count varying
   per input). Not flattenable into one bounded pass. This is [`open-questions.md`](open-questions.md) **axis F's**
   multi-hop candidate, now grounded in §9.2's substrate — and it is the harder, more convincing version.

Build the arena, then (1), then (2).

### 9.6 The base spec — `[AUTHOR]`, 2026-08-06, *"first ideas เอาไว้ตี, ยังไม่มี math proof"*

The standing candidate's concrete form. **Four parts.**

**1 · Input** — the **front half of a CNN**, broken into many components that may repeat.

**2 · Output** — the **back half of the CNN**, left with **several outputs to choose among** rather than collapsing
to one class (*"แทนที่จะปล่อยให้โมเดล shrink เองจาก n ไป 1 … ตอนนี้อาจจะ 32 ไป 20 แทน"*).

> **Why the back half is kept at all — the author's reason, and it is a separation-of-concerns argument:**
> *"loop of thought model เราไม่ได้มีหน้าที่ classification มันมีหน้าที่คิด เทียบกับความทรงจำ"* — if input were the front half and
> output were the class, the model would have to be **both a loop and a fine-grained classifier**, and it would
> break. Sending the output on to the back half leaves the loop **one job: make vectors and think on vectors**,
> while the back half acts as **the translator from abstract numbers into classes a human can see.**
>
> **⚠ Note the echo, because it is not a coincidence:** `loop (works in vector space) + CNN back half (names it)`
> is structurally **the same division as draft 6's** `SCFF bulk (vector space) + SLDA namer (names it)`. The same
> insight — *never make the representation learner also be the classifier* — carried across a complete change of
> substrate. Draft 6's architecture experience transfers even though SCFF itself is folded.

**3 · Thought = a transformer**, with vectors grounded in conv components rather than words. If
`Conv[x][y][0]` holds line features and `Conv[x][y][1]` holds curvature, the transformer builds its vectors from
**there**, so that `[x][y][0]` and `[x][y][1]` become addable. *(Agent reading, author-confirmed in substance: each
channel carries a learned **embedding** `e_c` and the activation is its **weight**, so what is added is
`Σ_c a_c(x,y) · e_c` — activations are scalars and cannot be added as meaning. Prior art for exactly this step:
bag-of-visual-words → VLAD → NetVLAD.)*

- **goal** = the pattern *"is this an x, or not?"* — a **comparison-shaped question**, not a search command
- **fact** = data vectors that **scope which region of memory to search**

**4 · Memory output ≈ a vector database**, with concepts linked through shared conv components.

- **The taxonomy is the co-occurrence structure itself, not language.** If an orange image carries
  `Conv[x][y][0]` and `Conv[x][y][1]`, and a sphere image carries `Conv[x][y][0]`, then *orange* and *sphere* are
  **connected through `[x][y][0]`** — *"โดยข้อมูลภาพตรงๆ ไม่ใช่จากภาษา"*. This is the author's answer to axis H: no stored
  tree, no WordNet, no human categorisation. **→ which is exactly why the backbone must be self-supervised
  (§9.2): a label-trained CNN would smuggle WordNet in through the features and void the claim.**
- **The ladder = ranked property checks.** Asked *"is it a mandarin?"* (≈ round + orange + rough peel, from image
  sensory alone), check the properties **one at a time, ordered by how heavily connected each is** — round (very
  connected) > orange (a colour) > rough peel (a texture) — until a fact produces the right feeling. Top-ranked
  components may be **summed**, since the sphere vector already adds to the orange vector.
- **The point, stated by the author:** *"เราไม่ได้ทำ brute force search ที่เอาทุกอย่างมาบวกกัน at once แล้วหาเลย แต่ว่าเรากำลังทำ
  **abstract stacking search** ที่ค่อยๆ ไล่เช็คทีละ fact"*.
- **⚠ The ordering is conditional, not global.** An agent objected that connection-count is a property of memory,
  not of the question, so the ladder would be the same for every input. The author's answer: *"ก็ต่อเมื่อ input เรามัน
  fix ไง … ตอนนี้ input เราคือ previous goal + fact มันทำให้ ladder เรียงกันได้แบบ dynamic เพราะว่า output จะขึ้นอยู่กับ fact cache"* —
  i.e. **the ranking is recomputed on the sub-graph that survives the current fact cache.** The objection is
  withdrawn. *(Nearest formal relative to check: information gain / candidate elimination — see
  [`reading-checklist.md`](../notes/reading-checklist.md) §4.)*
- **Two question types:**
  1. ***"is it x?"*** — the **training** mode: teaches the model how things connect
  2. ***"what is it?"*** — the **use** mode: reads the trained vector knowledge out directly
- **How memory is called.** The goal holds a **comparison** question and never touches the search directly. Only at
  the very start is the question **extracted into a ladder**, each rung proving one component that already has a
  vector — and those rungs live in **fact**:
  1. *is a mandarin an orange?* → prove: round > orange > peel texture
  2. `goal` → *is it round?* · `fact` → conv layer `[·][·][·]`
  3. `f(goal, fact) = compare(goal, read_memory(fact))`
  4. update `fact`; if the goal reaches the right feeling → advance to the next goal, *is it orange?*
- **Why goal was split out at all** — honestly, uncertainty: *"กุไม่แน่ใจเหมือนกันว่าเราจะสร้าง vector คำถามบน image dataset ยังไง
  เลยใช้ไอเดียนี้ไปก่อน เพราะมันดูจะเสถียรกว่า"*. `[OPEN]`

**⚠ Where this spec is thinnest** `[CLAIM]`: co-occurrence of conv channels gives **bundling** (*these appeared
together*) and never **binding** (*this property belongs to that object*). On **single-object images that gap does
not bite**; on **multi-object scenes it is fatal** — round + orange + green + leafy in one image does not make the
orange leafy. That line is therefore also the natural staging of the playground: **object-centric first, scenes
exactly when binding becomes mandatory** (see [`reading-checklist.md`](../notes/reading-checklist.md) §5).

---

## 10. The one-paragraph version

`[STANDING]` A thought is a **thesis in three parts** — a goal, an accumulating fact register, and what memory
returns given both. Concepts are **bundles of attributes** in a space where meaning **adds and subtracts** (the
same root the transformer vector stands on); a word is just the shared-world handle for a bundle, used at whatever
resolution the context asks for. The loop runs in **two motions**: hold the goal and accumulate facts until memory
matches, then **tighten the goal's scope and go again**. And it **never opens with the final question** — always
*fruit? → orange? → mandarin?*, because attacking the specific question directly is brute-force search over big
data, not thought. This looks inefficient on easy tasks, and that is accepted: the object being built is the
foundation, not a competitor to transformers at vector arithmetic. SCFF, measured against this, was
`compare(x, corrupted-x)` — a difference amplifier — when the picture always called for `compare(thing, thing)`;
a superb wiring harness on a car with no wheels or engine.

It gets built in an **arena, not a benchmark**: the front half of a CNN supplies input numbers that already carry
meaning, the back half supplies output numbers that already carry meaning, and the loop lives between them —
practising thinking on dense non-linguistic data **with nothing hinted to it**, with a **read-bandwidth limit** so
that iterating is structurally necessary rather than optional. And all of it is the **first decision, not the
final one**: evidence zero, held so the research about to start has something to attack, and replaceable only by a
better big picture.
