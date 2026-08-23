# 05-revision-loop: within-paper revision dynamics (Harit's E1/E2)

- **Cluster:** research-loop
- **Question:** (H1) does K rounds of review→revise improve the SAME paper near-monotonically? (H2) do distilled lessons carry across episodes — better starting points, steeper trajectories, less variance?
- **Owner:** Harit (design) · Kaleb <kaleb.klara97@gmail.com> (run)
- **Status:** running (started 2026-07-22)
- **Headline:** —

## Design (from Harit's sprint writeup, 2026-07-22)

**E1 (arm `e1`, H1):** author writes an empirical paper; then K=4 rounds of
review → revise on the same paper, reviewer rating recorded each round plus a
final review of the last version. S=6 episodes (phase-1 topics, round-robin).
No lessons. Expectation: near-monotone rating improvement.

**E2 (arm `e2`, H2):** E1 + the lessons channel — each round's review is
distilled (stock 3-lesson prompt) into `lessons.md`, injected into subsequent
authoring and revise prompts. M=6 episodes share the file. Expectation: later
episodes start higher / improve faster, possibly with less variance
("M lines, each a run of E1-with-lessons").

## Relation to existing results

- Phase-1 ON (01-lessons-on-off) already shows H2's *across-episode* half:
  cycle-0 papers 2.33 → later cycles ~4.67 on the frozen rubric. What 05 adds
  is the **within-paper trajectory** (the loop has only ever done one revise,
  v1→v2) and per-episode starting points/variance.
- 03-review-source varies WHO reviews for the lessons channel; 05 varies the
  revision depth. Together they factorize the two feedback channels.

## Deviations from the writeup (deliberate)

1. **Dual metric.** Harit's design plots reviewer scores. Our reviewer-probe
   (verification/01) says LLM ratings are unreliable, so every round records
   BOTH the reviewer rating (his metric) and the frozen mechanical rubrics
   scored per version. Divergence — rating climbs while rubric doesn't — is
   itself a headline result (reviewer rewards its feedback being addressed,
   i.e. score inflation).
2. **Tool-based revise, not thread-resumed.** Stock revise resumes the
   author thread; we first tried re-sending paper + review as prompt text
   each round (needed for external orchestration) — this reliably degraded
   after 1-2 rounds into refusal or a change-log summary instead of an
   actual revision (the model didn't trust/use the embedded paper text,
   attempted blocked tool calls, then refused to fabricate further stats on
   request). Fixed by `revise_with_tools()`: the agent gets real Read/Write
   access, scoped to just that paper's own directory, and reads/writes
   `v{k}.md` directly instead of round-tripping the paper through the
   prompt. Same for both arms, so E1-vs-E2 contrasts are unaffected.
3. **Reviewer sees revision history.** From round 2 on, the reviewer prompt
   includes every prior version and its review in full (see
   `history_context()` in `driver.py`), so it can judge whether the revision
   addressed the last review rather than re-reviewing blind each round. This
   trades off cost — context grows each round — against a more realistic
   "how do editors actually reread revisions" model.
4. **No PAPER_CHARS truncation on review/revise.** Stock `PAPER_CHARS`
   (12,000 chars) is a review-excerpt length, but our papers routinely exceed
   it (14k-29k chars observed), and truncating both the reviewer's and the
   reviser's view of the SAME paper mid-document caused the reviewer to
   (correctly) flag the paper as incomplete every round, and the reviser to
   compound it — each round it saw less of its own growing paper, until by
   round 4-5 it refused to keep fabricating a "complete" revision. Fixed by
   reviewing/revising the full paper (now `review_with_tools()`, see
   deviation 9) and adding an explicit length-parity instruction to the
   revise prompt (see deviation 2)
   so revisions don't balloon (14.8k -> 24k -> 27k -> 28.8k chars was
   observed pre-fix) instead of tightening in place. The soft "~10%" version
   of that instruction still wasn't enough on its own (a later run still
   grew 19k -> 42k over 4 rounds), so length is now split into an intended
   target (`TARGET_PAPER_CHARS = 10,000`, stated in the author/revise
   prompts as what to aim for) and a hard cap (`MAX_PAPER_CHARS = 20,000`,
   "must not exceed... under any circumstance") enforced post-write:
   `enforce_char_limit()` checks the actual file length after every
   author/revise call and, if over the hard cap, issues a follow-up trim
   request (real Edit access, same paper-dir scoping, trims toward the
   10K target) rather than just trusting the prompt. A trim failure
   (timeout, non-zero exit) is logged and swallowed, not raised — it
   shouldn't cost an already-valid paper or the rest of a multi-episode run.
5. **Author surveys literature AND executes real experiments.** Stock
   `autoresearch` authoring (`_build_complete()`) has no tools at all — the
   model free-associates a fabricated-results paper straight from the topic
   string. `author_with_tools()` gives it real WebSearch + Read/Write +
   scoped Bash (`Bash(python *)`, `Bash(pip install *)`, not unrestricted
   shell): it surveys existing work, weighs candidate angles, then
   implements and RUNS its method (CPU-only, told to design something that
   finishes in a few minutes — synthetic data, a toy/small model, a small
   real dataset) and reports what it actually observed, saving code + logs
   in the paper directory. `revise_with_tools()` gets the same Bash access,
   so revisions can also re-run/adjust code rather than only editing prose.
   This was the `execution` knob `autoresearch.py`'s own docstring left as
   a future follow-up. Not a guarantee of correctness — a small CPU
   experiment can still be a weak test of the paper's claims, and if
   execution fails the author may fall back to a worked estimate (which
   `review_with_tools()`, deviation 9, is specifically able to catch by
   re-running the code itself).
6. **Review guidelines expanded from the stock multi-stage reviewer.**
   `ml.REVIEW_SPEC` (monkeylab.py) is a bare JSON schema + scoring rubric.
   `driver.py`'s `REVIEW_GUIDELINES` folds in the substantive dimensions from
   the stock `scientist_reviewer`'s stage prompts (section coverage, novelty
   & placement, rigor/reproducibility/threats-to-validity, calibration
   self-check) as guidance for our single-shot call, rather than replicating
   its separate read/section/novelty/rigor/draft/critique/finalize model
   calls. (A prior version added a "check if past review questions were
   addressed" line directly to the shared `ml.REVIEW_SPEC`; moved out since
   it duplicated `history_context()`, doesn't apply to round 1, and shouldn't
   leak into other experiments that reuse `monkeylab.py`.)
7. **Execution-leniency reviewer variant (opt-in, `--lenient-on-execution`).**
   Originally written for when the author had no compute at all and reviews
   made "no real experiments were run" the central, unresolvable objection.
   Now that the author actually executes code (deviation 5), this is mostly
   stale by default — `EXECUTION_LENIENCY_NOTE` in `driver.py` now only
   covers the case where the paper is honest that a *specific* number is a
   worked estimate rather than an executed result; it explicitly does NOT
   cover numbers presented as real results the reviewer can't verify from
   the code/logs in the directory (that's still a legitimate weakness).
   Kept as a separate opt-in constant/flag, writes to a separate `-lenient`
   agent dir, so default runs are untouched and both stay directly
   comparable.
8. **Cross-model, same-family, via `--author-model`/`--reviewer-model`.**
   Full cross-family replication (Claude vs. GPT vs. Gemini) is still a
   follow-up — the claude-p proxy and `author_with_tools()`/
   `revise_with_tools()`/`review_with_tools()` all shell out to the `claude`
   CLI specifically, so only Claude-family models (sonnet/opus/haiku) are
   reachable this way; other families need OpenRouter keys and would lose
   the WebSearch/Bash tool access unless reimplemented for that path. Card
   TODO for that part. Within the Claude family, `--author-model`/
   `--reviewer-model` (default: sonnet for both) already let you mix, e.g.
   `--author-model haiku --reviewer-model sonnet` — runs with non-default
   models get a distinguishing agent-dir suffix
   (`local-rev-e1-ahaiku-rsonnet`) so they never collide with same-model
   runs.
9. **Reviewer verifies by executing the author's code.** Stock review was a
   single text-completion call (`ml.chat`) with the paper embedded as a
   string — it could only evaluate claims, never check them. `driver.py`'s
   `review_with_tools()` gives the reviewer real Read + scoped Bash access
   to the paper's own directory: it's instructed to actually run any
   code/scripts the author left behind and treat a mismatch between claimed
   and actual output as a real weakness. It writes its review straight to
   `round{k}_review.json` via the Write tool (structured JSON-in-chat-reply
   got unreliable once a model does tool calls throughout — same lesson
   that moved revise off a text-only response). `VERIFICATION_GUIDELINES`
   in `driver.py` folds in the correction/retraction-worthy bar and
   "flagging should be rare" discipline from Paperena's
   `verify-text`/`verify-citation`/`verify-triage` skills
   (`paperena_agent/harnesses/claude/plugin/skills/`), adapted for a
   reviewer that can check claims directly rather than only reading them.
10. **Self-monitored time budget + timeout/cleanup/retry.** `claude -p` is a
    single headless call — there's no live channel to inject a warning
    mid-turn once it's started, so "warn before timeout" is implemented as
    self-monitoring instead: author/revise/review prompts state a hard
    `TOOL_TIMEOUT_S` (1800s), an `EXPERIMENT_BUDGET_S` ceiling (half of
    that) for implementing/running code, and a `CHECKPOINT_S` (300s before
    the hard cutoff) past which the agent is told to stop experimenting and
    write up immediately, tracked via its own `date +%s` calls. If the hard
    cutoff is hit anyway, `run_claude_tool()` kills the whole process tree
    (not just the immediate shell child — `--add-dir` doesn't sandbox Bash
    the way it does Read/Write/Edit, so a spawned experiment could outlive
    the top-level process on Windows if only that were killed), deletes any
    files created in the paper directory during that specific attempt (so a
    half-finished experiment/paper doesn't linger or corrupt a later
    resume), and retries up to twice before raising — which `main()`'s
    per-episode `try/except` then catches as an ordinary episode failure.

## How to run

```
python driver.py e1 --gpu 5     # resumable; rerun to continue
python driver.py e2 --gpu 5
python driver.py e1 --lenient-on-execution   # -> local-rev-e1-lenient, comparable to local-rev-e1
python driver.py e1 --author-model haiku --reviewer-model sonnet   # -> local-rev-e1-ahaiku-rsonnet
```

Ratings ledger: `<agent-dir>/data/autoresearch/rounds.jsonl`; versions
`v1..v{K+1}.md` per paper dir. Rubric-score versions offline via
`analysis/rubric10.py` / `rubric_lit.py` (custom per-version pass).

## Artifacts

- agent dirs: `local-rev-e1`, `local-rev-e2`
- driver: `driver.py` (via `infra/monkeylab.py`)
