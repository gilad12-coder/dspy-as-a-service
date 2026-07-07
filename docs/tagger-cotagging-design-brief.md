# Design Brief — Co-Tagging: Human + AI Interactive Tagging

> Produced by `/shape` (discovery interview + codebase exploration + HCI literature review).
> Hand off to `/implement`, `/impeccable craft`, or any implementation flow.
> Status: **awaiting founder confirmation**.

---

## 1. Feature Summary

Co-Tagging upgrades the existing manual tagger (`frontend/src/features/tagger/`) with an AI partner. After uploading a dataset, the model **interviews** the user about it and drafts a labeling rubric. In **Co-pilot** mode the user then labels a small recommended calibration set while the AI learns alongside them; the tagger compiles those labels into an optimized tagging prompt, auto-tags random review batches that the user audits, and — once human–AI agreement crosses a gate — tags the rest of the dataset in a background job. In **Autopilot** mode the AI tags everything straight from the interview. Manual mode remains exactly today's tagger.

Success looks like: a user with a 5,000-row dataset gets a fully-tagged, provenance-annotated dataset in ~15 minutes of human effort instead of hours, and *knows how much to trust it* because they watched the agreement meter earn the unlock.

## 2. Primary User Action

**Calibrated delegation.** The single thing the user must understand at every moment is *how good the AI currently is at their specific task* — and the single thing they must do is delegate exactly when that's earned (and not before). Everything — the interview, the live agreement meter, the review gate, provenance badges, cost estimates — serves that one comprehension.

## 3. Design Direction

Extends Skynet's **warm, precise, premium** voice (`.impeccable.md`). The AI is a **quiet apprentice, not a co-star**: it observes, predicts silently, speaks up briefly when it disagrees, and never takes over the keyboard. Calm, factual microcopy; no hype ("agreement 87% — one more round" not "Almost there! 🎉").

Gold `#C8A882` is spent on exactly one thing per view: the agreement meter milestone during calibration/review, and the "Tag the rest" unlock CTA once earned. Cream/espresso/taupe neutrals carry everything else. Cost is always shown before commitment (credits estimate) and after (actual) — this feature spends the user's money at scale.

### Modes (mirrors the agent trust-mode vocabulary)

| Mode | What it is | Analogy |
|---|---|---|
| **Manual** | Today's tagger, untouched | — |
| **Co-pilot** *(recommended)* | Interview → calibration co-tagging → agreement-gated review rounds → bulk auto-tag | `ask`/`auto_safe` |
| **Autopilot** | Interview → rubric confirm → tag everything, flag low-confidence | `yolo` |

`TaggerAssistMode = "manual" | "copilot" | "autopilot"`. Session phases extend the existing enum: `setup → interview → calibration → review → autotagging → complete` (manual keeps `setup → annotating`).

### HCI research → three binding interaction rules

The founder asked for a literature-grounded decision on "who labels first" during calibration:

- Pre-filled label suggestions bias annotators — they follow a different decision process and tend to accept suggestions; errors become **systematic instead of random** (exactly what poisons a training set). Non-experts are most susceptible, and suggestions amplify model errors ([Beck et al. 2021](https://arxiv.org/pdf/2105.12980), [Blind Spots and Biases](https://arxiv.org/html/2404.19071v1), [Anchoring in Syntactic Annotation](https://arxiv.org/pdf/1605.04481)).
- **Deciding before seeing the AI's recommendation** is a proven cognitive forcing function that significantly reduces overreliance — with the caveat that users *prefer* the lazier flow, so the blind flow must feel fast and rewarding ([Buçinca et al. 2021](https://arxiv.org/abs/2102.09692)).
- In review loops, people undercorrect when correcting costs more effort than accepting; **make correction cost exactly one keystroke, same as confirmation** ([Bias in the Loop, 2025](https://arxiv.org/html/2509.08514v1)).
- Interactive round-based retraining beats static suggestions ([Beck et al. 2021](https://arxiv.org/pdf/2105.12980)) — validates the review-round refinement loop.

**Rules:**
1. **Calibration is human-first.** User labels with the existing keyboard flow; AI predicts silently in parallel and reveals only *after* the human commits (agree → subtle tick; disagree → a brief "I'd have tagged this **X** — because…" card with *keep mine* (default) / *switch* / *note it in the rubric*). Configurable via a pref (`taggerCalibrationStyle: "blind" | "assisted"`, default `blind`; the assisted option exists for speed-hungry power users, with a one-line note that blind calibration trains a better tagger).
2. **Review is AI-first by design** (auditing AI output *is* the task), with equal-cost correction: `Enter` confirms the AI's label, pressing the correct key (Y/N/digit) in one stroke overrides it. Confidence is always visible.
3. **The AI never takes keyboard focus.** The companion rail is display-only; all interaction stays on the existing shortcut system.

## 4. Layout Strategy

All surfaces live **inside the tagger** — no drawer, no new top-level route. RTL-first: companion rail sits on the **inline-end** side, all positioning logical-CSS.

- **Setup — mode picker.** When the feature is enabled, `TaggerSetup` gains a final step: three flat option rows (not icon-topped cards) — Manual / Co-pilot / Autopilot — with one-line descriptions and a quiet "recommended" tag on Co-pilot. Gold focus ring on selection. Choosing Manual proceeds exactly as today.
- **Interview.** A centered single conversation column (~65ch) reusing `shared/ui/agent` primitives (`agent-thread`, `composer`, quick-reply chips). The AI has already read a dataset profile (columns + sampled rows + the user's question/categories/prompt) and asks **3–5 focused questions** — ambiguous boundaries, edge cases, dirty-row policy — never generic ones. Output is a **rubric card**: a short editable labeling guide the user confirms. The rubric is a living artifact for the rest of the session.
- **Calibration.** The existing annotation surface is untouched in the center — same typography, same progress bar, same shortcuts. The companion rail (≈300px, collapsible; container-query collapses to a bottom strip on narrow viewports) holds: AI status line, post-commit reveal, **live agreement meter** (the one gold element), calibration progress ("14 of 30"), and the rubric (updated entries pulse once, subtly).
- **Review rounds.** Same surface, inverted: AI's label arrives pre-committed with a confidence marker; the row's visual weight shifts to the label being audited. Between rounds, an interstitial summary: round agreement, gate state, what the rubric learned, and either "next round" or the unlocked **Tag the rest — ~N credits** CTA.
- **Auto-tagging.** A calm progress view: tagged count, simple confidence-distribution bars (plain, informative, not decorative), elapsed/ETA, cancel. Runs server-side; leaving the page is safe, the sidebar session card shows live progress.
- **Completion.** Provenance summary (n human · n AI-confirmed · n auto), credits actual vs estimate, a flagged-rows entry point ("Review 43 rows the AI wasn't sure about" — same keyboard flow over just that subset), then the existing export / save-to-library affordances.

### Calibration set size (recommended, no user override)

| Mode | Target |
|---|---|
| Binary | 24 |
| Multiclass | max(30, 6 × #categories), capped at 60 |
| Freetext | 30 |

Review batches: 20 rows each, sampled uniformly at random from untagged rows. Agreement gate: ≥ 92% for binary/multiclass; freetext counts normalized fuzzy match ≥ 0.85 as agreement, gate ≥ 85%. If two consecutive rounds miss the gate, proactively offer **Deep optimize** — a real DSPy optimization job over the calibration set (reuses existing optimization job infra; time + credit cost stated up front). Deep optimize is also always available as an explicit option, per founder.

If the dataset is small (rows ≤ target + one review batch, roughly ≤ 50), say so honestly at mode selection: "At this size you'd review nearly every row anyway — Manual or Autopilot will serve you better," and de-emphasize Co-pilot.

## 5. Key States

| State | User needs to see / feel |
|---|---|
| Feature off (settings) | Tagger identical to today; zero residue |
| Interview loading | "Reading your dataset…" status over a skeleton; profile sampling for large files |
| Interview error / model unavailable | Calm fallback: "Continue manually, or retry" — never a dead end |
| Rubric confirm | The editable rubric card; user feels ownership before anything runs |
| Calibration agree | Sub-second tick, no interruption to flow |
| Calibration disagree | The apprentice speaks once, briefly; user resolves with one action |
| Gate locked | Meter shows distance to gate factually; never shame ("agreement 84% — gate opens at 92%") |
| Gate stalled (2 rounds) | Deep-optimize offer with explicit cost; declining is frictionless |
| Gate unlocked | The gold moment: "Tag the rest — est. N credits" |
| Bulk job running | Live progress; safe to leave; cancelable; partial results kept |
| Bulk job failed / canceled | What was tagged stays; resume from where it stopped |
| Complete | Provenance breakdown, flagged-rows pass, cost actuals, export |
| Autopilot path | Interview → rubric → cost estimate → confirm → progress → complete (with flagged rows) |
| Resume mid-any-phase | Sessions persist per phase; reload lands exactly where you were |
| Tiny dataset | Honest de-recommendation of Co-pilot (above) |
| Huge dataset (→200k rows) | Profile built from a sample; batched job with ETA; estimates stay accurate |

## 6. Interaction Model

- **Keyboard is sacred.** Y/N, 1–9, arrows, U, E, Ctrl+H all work unchanged in every phase. New: `Enter` = confirm AI label (review), single label-key = override (equal cost). Post-commit reveal appears ~150ms after commit and never blocks the next keystroke.
- **The rubric is the shared memory.** Interview seeds it; calibration disagreements and review corrections append to it (visible, editable). Users can open it any time from the companion rail. It compiles — along with the labeled examples — into the tagging prompt (few-shot compile, instant; deep optimize as the heavyweight option).
- **Trust is earned monotonically.** The agreement meter never resets mysteriously; corrections visibly feed the next round ("3 corrections folded in").
- **Cost before commitment, always.** Credit estimate on the unlock CTA and Autopilot confirm; actuals on completion. Uses the user's configured task model — no new model picker in v1.
- **Provenance is honest.** Every annotation carries `human` / `ai_confirmed` / `ai_auto` (+ low-confidence flag). Shown in completion summary; available as an export column.

## 7. Content Requirements

All copy through the i18n layer (`msg()` / `TERMS`), translatable across 24 locales, RTL-correct. The AI's own interview/disagreement messages follow the existing "agent replies follow the chosen UI language" behavior.

Needed microcopy (calm, factual, no exclamation marks): mode-picker names + one-liners; interview intro ("I'll ask a few questions to learn how you think about this data"); rubric card title + edit affordance; agreement meter + gate labels; disagreement card ("I'd have tagged this X — {one-sentence reason}") and its three actions; round interstitial lines; unlock CTA with credit estimate; deep-optimize offer; bulk progress + cancel + failure recovery; provenance labels; flagged-rows CTA; small-dataset honesty line; feature-off settings row ("AI-assisted tagging"); calibration-style setting line.

Dynamic ranges to design for: category names (user-typed, any length, any script), 1–20+ categories, rubric 3–15 entries, datasets 10 → 200,000 rows, credit estimates from "<1" to thousands.

## 8. Recommended References

- `interaction-design.md` — keyboard flows, forms, loading, optimistic UI (the heart of this feature)
- `motion-design.md` — post-commit reveal, meter transitions, staggered interstitials, reduced-motion
- `ux-writing.md` — interview questions, gate copy, error states

## 9. Open Questions (for the implementer)

1. **Confidence estimation** — model self-rating vs logprob-derived; pick whichever the existing inference path exposes cheaply.
2. **Deep-optimize preset** — which optimizer configuration (GEPA/MIPRO tier) fits a ~30-example trainset; surface as one button, not a config panel.
3. **Calibration sampling** — uniform random for v1; stratified/embedding-diverse sampling is a later refinement.
4. **Export provenance column** — include by default or behind a toggle in the export popover.
5. **Blind seed rows in review batches** (known-answer QC items) — deliberately deferred; revisit if agreement gaming appears.

### Implementation seams (from codebase exploration — constraints, not design)

- **Reuse:** `shared/ui/agent/*` chat primitives; SSE streaming patterns from `generalist_agent.py`; trust-mode vocabulary; optimization job infra as the template for the bulk-tagging job; `parse-dataset.ts` + dataset profiler for the interview's dataset reading.
- **Extend:** `TaggerConfig`/`lib/types.ts` with `assistMode`; `TaggingSessionModel.phase` enum; `annotations` values from `string | string[]` to optionally `{value, source, confidence}` (back-compat: bare value ⇒ human) — update `_count_tagged` (`tagging_sessions.py:105`).
- **New:** LLM tagging engine under `service_gateway/` (few-shot prompt compiler + batch executor + interview conductor); interview SSE endpoint; bulk-tag job endpoints.
- **Settings:** `UserPrefs.taggerAssist` (default **true**), `UserPrefs.taggerCalibrationStyle` (default `"blind"`); build-time gate `NEXT_PUBLIC_FEATURE_TAGGER_ASSIST` following the generalist-agent flag pattern.
- **Gap to close en route:** tagger can save to the dataset library but can't load from it; the interview flow is the natural place to add "pick from library."
