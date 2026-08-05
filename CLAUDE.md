# CLAUDE.md — Bio-Inspired Analog Neural Compute Architecture

> The always-loaded **operating brain** for this repo: **cross-draft and stable** — identity, how to work here, and
> the router into each draft's own context. **Draft-specific status and detail are NOT here** — they live in each
> `draftN/CLAUDE.md`, which auto-loads when you work in that draft. Cross-tool orientation: [`AGENTS.md`](AGENTS.md).

---

## What this project is

A bio-inspired **analog compute substrate** with on-chip learning — the **math model for a chip design** (analog
is the constraint-giver, simulation the medium; no circuit drafting or fabrication is planned), not an ML model
and not a digital ML accelerator. In the design picture: capacitors hold weights as continuous analog charge; SRAM
holds wiring and sign bits; hardwired op-amps do add / multiply / ReLU directly on charge; the chip **learns
on-chip, online, without a backward pass that leaves the chip.** Committed properties: **online, sparse, continuous, resident-weight**
(compute-in-memory). Method: **copy the brain's *function*, cheat the *implementation*** — pay for each principle
with whatever is cheap here (analog physics where physics is cheaper, modern DL math where math is cheaper). Full
overview: [`AGENTS.md`](AGENTS.md) / [`README.md`](README.md); the *why* and the person: [`docs/essence/the-essence2.md`](docs/essence/the-essence2.md).

---

## How context is layered here (read your tier, then stop)

This repo's agent context is a **3-layer hierarchy**, loaded lazily — so you load little and go deep only where
the task is:

1. **Root (this file)** — always loaded: identity, the rules below, the router. Stable; rarely changes.
2. **`draftN/CLAUDE.md`** — auto-loads when you work in that draft: its architecture, **current status**, file
   map, and per-draft routing. *(This is the file that changes as work advances — not the root.)*
3. **`phaseN/CLAUDE.md`** — a thin signpost to that phase's front-door synthesis (`phaseN/README.md` + `RESULTS.md`).

**Reading budget — the rule that keeps sessions cheap:** load the layer your task needs and **stop**. To understand
a *prior* phase, read its one `phaseN/README.md` (the front-door synthesis) — never its code or cards. For heavy multi-file lookups (sweep
several phases, hunt a paper), **dispatch the `Explore` sub-agent** (its own context window; returns only the
conclusion) instead of reading them into this session. *(If a rule seems missing, it may be in a nested `CLAUDE.md`
that hasn't loaded — run `/memory` to see what's in scope.)*

---

## Router — which world, and where

| Working on… | Load |
| --- | --- |
| **The live line (draft 7.0 — the loop of thought, SET ZERO)** | [`draft7.0/CLAUDE.md`](draft7.0/CLAUDE.md) — operating context; arrive via [`draft7.0/docs/handoff.md`](draft7.0/docs/handoff.md) |
| **What draft 7.0 is actually building** (the standing candidate) | [`draft7.0/docs/the-thought.md`](draft7.0/docs/the-thought.md) — frozen record · working version: [`draft7.0/notes/the-model.md`](draft7.0/notes/the-model.md) |
| Why we turned away from draft 6.0 (2026-08-05) | [`draft7.0/docs/context.md`](draft7.0/docs/context.md) |
| **Draft 6.0 — the first organ, DONE & FOLDED** (P1–11, not discarded) | [`draft6.0/CLAUDE.md`](draft6.0/CLAUDE.md) — operating context + final status |
| The draft-6 picture, cold | [`draft6.0/context.md`](draft6.0/context.md) |
| Superseded history (draft 5 — attribution era) | [`draft5.0/CLAUDE.md`](draft5.0/CLAUDE.md) |
| The idea journey (drafts 1.0 → 5.1) | [`draft-journey/`](draft-journey/README.md) |
| Why the project exists / the person | [`docs/essence/the-essence2.md`](docs/essence/the-essence2.md) · [`docs/draft/project-personal.md`](docs/draft/project-personal.md) |
| Why an old (5.1-era) decision was made | [`docs/draft/project-history.md`](docs/draft/project-history.md) |
| **The whole research wiring** (thought → experiment → result → decision) | [`graphify-out/graph.html`](graphify-out/graph.html) · [`GRAPH_REPORT.md`](graphify-out/GRAPH_REPORT.md) · how-to: [`docs/notes/graphify.md`](docs/notes/graphify.md) |

---

## Research knowledge graph (graphify)

This repo has a **persistent graphify map** in [`graphify-out/`](graphify-out/) — a navigable knowledge graph over
the full corpus (markdown decision chains, experiment cards, code structure, north-star papers). It exists because
this project has **no linear flag for decisions**: everything wires through `.md` text; graphify is the cross-draft
index for *why* a pivot happened, *what* experiment proved it, and *where* the next decision lives.

**Agents — use the graph first for corpus questions.** If `graphify-out/graph.json` exists and the user asks how X
relates to Y, why a phase pivoted, or where a concept first appeared: run `graphify query "…"` (or `path` / `explain`)
— do **not** re-read dozens of phase files into context. Full operator guide: [`docs/notes/graphify.md`](docs/notes/graphify.md).
Global skill: **`graphify`** (installed Cursor/Claude Code skill — `/graphify` or ask in plain words).

**Humans — open [`graphify-out/graph.html`](graphify-out/graph.html) in a browser** for the community map; read
[`graphify-out/GRAPH_REPORT.md`](graphify-out/GRAPH_REPORT.md) for god nodes, surprising cross-era bridges, and
suggested questions. After adding substantial docs or finishing a phase, **incremental update** — see the notes file.

Scan excludes: `.obsidian/`, `post/`, `temp/`, `temp2/` (scratch, not research record).

**Experiment PNGs:** do not vision-read — link structurally from parent `experiment-*.md`, phase reports, and
`plot_*.py` via [`tools/graphify_link_figures.py`](tools/graphify_link_figures.py) (see
[`docs/notes/graphify.md`](docs/notes/graphify.md) § Experiment figures).

---

## Read these once before non-trivial work

- [`docs/essence/the-essence2.md`](docs/essence/the-essence2.md) — the project's **soul** (the grown spine after ten phases: origin, the draft-5 collapse, the return, the neocortex built). The shorter original seed: [`the-essence.md`](docs/essence/the-essence.md). Read one before non-trivial work.
- [`docs/draft/project-personal.md`](docs/draft/project-personal.md) — who the author is and how they work (the gut, the incubation, the 10-minute window). Its rules on hedging, scope-creep, and length-matching are load-bearing.

---

## How to collaborate with this author

The full handoff is in [`docs/draft/project-personal.md`](docs/draft/project-personal.md). Shortest summary:

- **Year-2 undergraduate, solo, evening/weekend pace.** Bilingual English/Thai — don't correct grammar; meaning is always clear from context.
- **Pushed real hardware before** (ChronoForge — a pure-FPGA 2D game engine, 640×480@60 Hz in ~18k LUTs). **Don't talk down.** When they describe a circuit in plain words, they usually have the EE concept right.
- **No flattery, no hedging, no trailing "let me know if…!"** Asked to choose A vs B, pick one and defend it. Wishy-washy "both have merits" is worse than confidently wrong.
- **🤣🤣🤣 / 👹 / "bro" / 🔥 are signals, not casualness** — usually "I see what I'm doing and I'm committing." Treat the moment as the commit point.
- **The "we" framing is real.** Match it — collaborator, not tool.
- **Length matches the topic.** 1000+ words for architectural depth; one paragraph for a naming yes/no. Don't pad short topics; don't trim deep ones.
- **Intuition first, math check second.** When they describe a mechanism physically, answer at that level first.
- **They push back when wrong, absorb when right.** If they push back on a critique, slow down and re-read before reasserting (the "scap is a wire, not a neuron" save).
- **Breakthroughs come from incubation, not desk-grinding.** Don't rush to closure; don't summarize prematurely.

---

## Methodological rules (apply to every experiment)

1. **One thing changed per experiment.** Compare exactly one variable between two runs.
2. **Multiple seeds per cell.** Standard set: **`[42, 137, 271, 314, 1729]`**. Report median + IQR, not single runs.
3. **Controlled variables explicit.** Lock task, init, total weight count, training steps, eval metric — unless one is under test.
4. **Invariants checked every run.** Convergence/loss-slope, dead-unit fraction, ceiling/goodness saturation, and (once chained) inter-block drift / SCFF cluster-churn.
5. **Failures are data.** A config that fails to converge is a result — log it, report it, move on. Don't tune until it works; that's how you lie to yourself.
6. **Defer fallbacks until baseline is characterized.** EMA-view, margin loss, Adam-style accumulators — tested as remediation *after* baseline behavior is seen, not bolted on first.
7. **Defer PVT realism until the ideal converges.** Ideal deterministic operators first; process/voltage/thermal later.
8. **Architecture changes are decisions, not experiments** — backed by a result, not a hunch.

---

## Naming discipline

Biological names are **structural, not decorative**. Don't suggest renaming to "be more rigorous" — already considered and rejected.

- Default usage = the circuit element. "Brainstem" / "Hippocampus" mean the circuit, not the biology.
- Prefix **`biological-`** when actual biology is meant; prefix **`analog-`** when explicit circuit framing helps in a mixed paragraph.

---

## Skills & workflow commands

- **Workflow commands** (`.claude/commands/`, self-contained): **`/orient`**, **`/checkpoint`**, **`/commit-progress`**, **`/double-check`** — explicit moves that fire the same steps if you ask in plain words ("commit it", "double-check this"). Add one when a manual ritual repeats 3+ times.
- **Skills (auto-load by task):** universal ones in **`.claude/skills/`** — `project-frame` (what the project is), `explore` (navigate the repo), `folder-structure` (where things go), `writing-report` (document a result). Draft-6-specific ones in **`draft6.0/.claude/skills/`** — `status`, `run-experiment`, `find-paper`, `simulator-code` (load when you work in draft 6). Design + maintenance guide: [`AGENT-OPTIMIZE-GUIDE.md`](AGENT-OPTIMIZE-GUIDE.md).

---

## When in doubt

- **What we're building** → [`draft6.0/CLAUDE.md`](draft6.0/CLAUDE.md) + [`draft6.0/idea/main.ideas.v1.md`](draft6.0/idea/main.ideas.v1.md) (the decision record) — not the draft-5.1 files.
- **How to talk to the author** → [`docs/draft/project-personal.md`](docs/draft/project-personal.md).
- **Why an old thing was decided** → [`docs/draft/project-history.md`](docs/draft/project-history.md) (5.1 era) — but the learning rule was rebuilt; don't re-defend dead decisions.
- **Arithmetic / signs** → run the numbers. The whole draft-5 collapse was a missing **sign** (direction); the §3.3/§3.7 XOR sign bug lived four drafts because nobody computed `+ XOR +`. Direction bugs are this project's recurring silent killer — stay paranoid.
