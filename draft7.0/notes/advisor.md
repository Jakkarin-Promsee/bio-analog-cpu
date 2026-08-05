# working with an advisor — a manual

> Written 2026-08-06, for a specific situation: **three years of self-teaching, a position that exists
> ([`the-model.md`](the-model.md)), and no idea how to use a person.** The author's own words: *"กุ self learning มา 3 ปี
> แล้วมั้ง กุเลยไม่แน่ใจว่าควร collaborate กับเขายังไง เพราะสุดท้ายกุไม่พึ่งเขาก็ได้"*
>
> Live file. Edit it as you learn what actually works with these particular people.

---

## §0 — First, delete the anxiety

**You are not begging.** A professor wants: interesting problems, students who actually produce, ideas they don't
have time to chase themselves, and eventually papers and students worth writing letters for. A motivated
undergraduate walking in with **a stated position** is *valuable inventory*, not a burden.

The evidence is already in: seven people walked to another building for an unprepared conversation. That does not
happen out of politeness.

**Their scarce resource is minutes, not knowledge.** Every rule below is a way to spend fewer of their minutes for
more of your value.

---

## §1 — The one reframe that fixes "there are too many holes to ask about"

**Never ask about a hole. Ask about a decision.**

A hole is unbounded — answering it means designing your project for you, which nobody will do, and asking it makes
you look like you want to be carried. A decision is bounded — it takes ten seconds to shoot at, and shooting at
things is *fun* for them.

You already have decisions everywhere. You just filed them as holes.

| Filed as a hole | ❌ Unbounded ask | ✅ Decision-shaped ask |
| --- | --- | --- |
| the right feeling has no form | "How do I build a correctness signal?" | "I think the signal is a **margin between the top-1 and top-2 candidate**, not an absolute distance — because *'similar but not sure'* describes two candidates scoring close together, not one scoring low. Is there a reason that's wrong?" |
| how to store memory | "How should memory work?" | "I'm treating memory as **a vector store where concepts link through shared conv channels**. Is there a reason to prefer an energy-based associative memory instead?" |
| binding vs bundling | "How do I do binding?" | "Co-occurrence gives me bundling but never binding. I think that's **survivable on single-object images and fatal on scenes**. Is that the right place to draw the line?" |
| where the ladder comes from | "How do I get a hierarchy?" | "I rank property checks by **how connected each component is, recomputed against what I've already established**. Is that just information gain with extra steps?" |
| which dataset | "What dataset should I use?" | "I plan to start on **dSprites because it has ground-truth factors I can verify against**. Is that too toy to learn anything real from?" |

**The pattern:** *"I chose X, for reason Y. What am I not seeing?"*

That sentence is the whole manual compressed. It shows you thought, gives them a target, and costs them ten
seconds. It is also — exactly — your own standing-candidate rule, applied to a person instead of a draft.

---

## §2 — The four questions that always work

**1. "Has this been done already?"** — the highest value per minute you will ever get from a human. A senior
researcher's pattern-match over the literature beats your 64 hours of reading and costs them thirty seconds.

**2. "Is this question actually open?"** — taste. Which of your unknowns are genuinely unsolved in the field vs.
already closed. You cannot get this from papers; papers don't tell you what's boring.

**3. "What would you attack first?"** — invites them to be smart, costs a minute, and hands you their **priority
ordering**, which is worth more than any single answer.

**4. "What am I not seeing?"** — the closer. Open, but bounded by the statement you just handed them, so it isn't
lazy. Ask it at the end of every meeting.

---

## §3 — The one-page rule

**You have 174 KB of documents. Never send more than one page. Ever.**

Sending the repo says *"do my reading for me."* Sending one page says *"I compressed this for you."* The second one
gets read.

The page:

```
1. The question           — one sentence
2. Why standard metrics can't answer it — one sentence
3. What I'm building      — three bullets
4. Where I think it's weak — three bullets      ← the part that makes it credible
5. What I'm asking you    — one specific question
```

**Section 4 is not modesty, it is the engine.** People engage with *problems*, not with *pitches*. A student who
names their own holes reads as a researcher; one who has an answer for everything reads as a salesman, and gets
politely nodded at.

You already have section 4 written — it is *"what I know is thin"* in [`the-model.md`](the-model.md).

---

## §4 — Meeting mechanics

- **30 minutes, one question.** Not five. Five means you haven't decided what matters.
- **Open with what changed** since last time — two minutes, no slides. *"Last time you said X; I read Y; it changed Z."*
- **Close by saying what you'll do and when.** This single habit is what makes people keep meeting you: it turns
  their advice into visible motion.
- **Never leave without the next contact point.** "Same time in two weeks" or "I'll email when the reading's done."
- **Send three lines afterwards**: *what I heard · what I'll do · when.* Costs you two minutes. Makes you the
  student they remember as reliable.
- **A recurring 30-minute slot beats an occasional two-hour one** — and it's a metronome against your own
  two-week disappearances.

---

## §5 — Nerd etiquette

- **Say "I don't know" immediately and plainly.** Bluffing is *instantly* visible to an expert, and it is the
  fastest way to lose one. *"I haven't read that yet"* is a complete, respectable sentence.
- **Never imply you've read something you haven't.** They will ask a follow-up question. They always do.
- **Credit precisely, and out loud, next time.** *"I read the paper you mentioned — it changed this part."* This is
  the highest-return behaviour in the entire manual: it proves their minutes converted into something, which is the
  only evidence that you're worth more of them.
- **Push back when you disagree — with a reason, not a feeling.** Advisors respect students who argue from
  evidence and lose interest in ones who nod. *(You're already good at this. You pushed back on me six times in
  two days and you were right at least three of them.)*
- **But argue to find out, not to win.** The difference is visible within one exchange, and only one of them makes
  people want to talk to you again.
- **Let small wrongness go.** If they misremember something in your domain and it doesn't change the decision,
  drop it. Correcting a professor in front of other people buys nothing and costs a little.
- **Answer messages within a day**, even if it's just *"got it, reading this week."* Silence reads as disinterest,
  never as depth.
- **Don't apologise for taking their time.** Do the compressing that makes the apology unnecessary.

---

## §6 — The traps specific to a self-learner

Your self-teaching is *why you have something to bring* — it is not the problem. It has exactly five failure modes
with advisors, and you are currently exposed to all five.

**1. Waiting until it's perfect before showing it.**
Self-learners polish in private. But an advisor's value is **highest when the work is unfinished** — a pointer in
week 1 saves six weeks; the same pointer in week 20 just hurts. **Show it early and rough on purpose.**

**2. Treating advice as either an order or as noise.**
Neither. Advice is **data with a strong prior** — you weigh it. You don't obey it, and you don't discard it because
you could have figured it out yourself. Write it down, then decide.

**3. Disappearing while incubating.**
Your two-week withdrawals are a working mode that *demonstrably produces results* — do **not** stop doing it. Just
send two lines when it starts: *"I'm sitting with X for a couple of weeks; I'll bring it back."* Zero cost, and it
stops your best mode from reading as loss of interest.

**4. Confusing "I can figure this out myself" with "I should."**
You can. That is precisely how you spent a year measuring something that could never answer your question.

> **The one thing a self-learner cannot self-learn is what he doesn't know exists.**

That is the entire job description of an advisor, and it is exactly the failure that produced draft 7.0. You have
the proof in your own repo.

**5. Not asking for the boring institutional things.**
Credits for the work. Lab or cluster access. A letter, eventually. Funding, eventually. These feel awkward to ask
for and are usually **cheap for them to give** — and they compound. Ask once you've shown you produce.

---

## §7 — Scripts

Sentences you can actually say. Adapt the register, keep the shape.

**The first real ask** *(this week — not "please advise me", which is unanswerable)*

> "อาจารย์ครับ ผมเขียนสิ่งที่ผมกำลังทำมา 1 หน้า — **มีอะไรที่มีคนทำไปแล้ว ที่ผมควรรู้ก่อนจะลงลึกมั้ยครับ**"

Flattering to their expertise, ten minutes of their time, and it de-risks your next 64 hours. If they answer well,
the advising relationship starts by itself and you never have to ask for it.

**Opening a follow-up meeting**

> "ครั้งที่แล้วอาจารย์บอกให้ดู X ผมไปอ่านมาแล้ว มันเปลี่ยนตรง Y ครับ — วันนี้ผมติดอยู่เรื่องเดียวคือ Z"

**Presenting a decision instead of a hole**

> "ตอนนี้ผมเลือก A เพราะ B แต่ผมไม่แน่ใจว่า C — **อาจารย์เห็นอะไรที่ผมไม่เห็นมั้ยครับ**"

**When you don't know**

> "อันนี้ผมยังไม่ได้อ่านครับ ผมจะไปอ่านแล้วมาเล่าให้ฟัง"

**When you disagree**

> "ผมเข้าใจที่อาจารย์พูดครับ แต่ตอนนี้ผมคิดต่างเพราะ [เหตุผล] — อาจารย์คิดว่าเหตุผลนี้มันพลาดตรงไหนครับ"

**Before you disappear**

> "ช่วงนี้ผมขอไปนั่งคิดเรื่อง X สักสองอาทิตย์ครับ แล้วจะกลับมาเล่าให้ฟัง"

**Closing every meeting**

> "แล้วมีอะไรที่ผมมองไม่เห็นมั้ยครับ"

---

## §8 — Who to ask what

You met more than one person. They are not interchangeable.

| Who | Ask them | Don't ask them |
| --- | --- | --- |
| **The neuro / quantum person** | *"Is this question real?"* · whether the loop framing corresponds to anything in actual brains · what "settled" means when nobody hands you an energy function | implementation, benchmarks, ML tooling |
| **Anyone ML-adjacent** | *"Has this been done?"* · which of Perceiver / NSCL / ACT / VSA is closest · what to read deeply vs. skim | whether the biology is right |
| **The quantum professor who walked you over** | keep them warm — a two-line update occasionally. They are the **bridge**, and bridges are worth more than any single answer | a detailed technical review of something outside their field |

**And the framing that lands with a physicist**, since that's the room you're in: don't lead with architecture,
lead with *"a system iterates until it settles — but what defines **settled** when nobody gives you the energy
function?"* That is a physics-shaped question wearing your problem's clothes, and they already have intuition for
it.

---

## §9 — The single sentence

If you remember nothing else:

> **Bring decisions to shoot at, not questions to answer — and always name your own holes first.**
