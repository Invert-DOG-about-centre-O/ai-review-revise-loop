#!/usr/bin/env python3
"""05-revision-loop: shared engine for E1's author/review/revise mechanics.

Not a script to run directly — this is the common core imported by the
per-experiment entry points:
    run_standard.py       history-aware reviewer, single reviewer/round
    run_nohistory.py      reviewer sees only the current version, no history
    run_multireviewer.py  N independent no-history reviewers per round
    driver.py             flexible/advanced CLI for one-off variants
                           (cross-model, --lenient-on-execution, --tag,
                           seeding from another run) not worth a dedicated
                           script

Deviations from the stock research-loop, documented in EXPERIMENT.md:
- author gets real WebSearch + Read/Write + scoped Bash access
  (author_with_tools()) to survey existing work, weigh candidate angles,
  and actually implement + run its method (CPU-only, small-scale) instead
  of free-associating a paper or fabricating Results from the topic string
  alone (stock autoresearch._build_complete() has no tools at all);
- revise gets the same real Read/Write/Bash access to its own paper
  directory (revise_with_tools()) rather than resuming the author thread OR
  embedding the paper as text in a stateless prompt — the latter was tried
  first and degraded into refusal/change-log summaries instead of real
  revisions;
- reviewer also gets real Read/Bash access (review_with_tools()) to
  actually run the author's code and check reported Results against what
  it produces, instead of taking claims on faith from an embedded-text
  prompt; writes its review straight to round{k}_review.json;
- Bash access for all three is scoped to `Bash(python *)` and
  `Bash(pip install *)` only, not unrestricted shell — enough for real
  experiments without opening up arbitrary commands;
- from round 2 on, the history-aware reviewer is also given every prior
  version + its review (history_context()), so it judges whether the
  revision actually addressed the last review rather than re-reviewing
  blind each round.

Resumable: every artifact (v{k}.md, round{k}_review*.json, .distilled{k})
gates its step. Ratings ledger: data/autoresearch/rounds.jsonl.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "infra"))
import monkeylab as ml

# Shared time budget for tool-enabled claude -p calls (author/revise/review).
# EXPERIMENT_BUDGET_S: told to the agent as its ceiling for implementing +
# running experiments (leaves the other half for survey/write-up/verify).
# CHECKPOINT_S: told to the agent as "stop experimenting and wrap up by
# here" — self-imposed, checked via its own Bash access (elapsed wall-clock
# time), since claude -p is a single headless call with no live channel to
# inject a real external warning mid-turn once it's started.
TOOL_TIMEOUT_S = 1800
EXPERIMENT_BUDGET_S = TOOL_TIMEOUT_S // 2
CHECKPOINT_S = TOOL_TIMEOUT_S - 300


def run_claude_tool(cmd, prompt, pdir, expect_file, timeout=TOOL_TIMEOUT_S, retries=2):
    """Run a claude -p tool-call subprocess with failure -> terminate ->
    cleanup -> retry. Two things count as failure, handled identically:
    (a) a hard timeout, and (b) the process exiting normally without
    producing expect_file — observed in practice: the agent tries to run
    its experiment as a BACKGROUND Bash job, then ends its turn saying
    it'll "pause here until the background experiment finishes" — but
    claude -p is a single one-shot call with no later turn for that to be
    checked on, so nothing is ever produced. (Prompts now say not to do
    this; this is the backstop.) Either way: kill the whole process tree if
    still running (not just the immediate shell child — a spawned
    experiment could outlive the top-level process on Windows if only that
    were killed), delete every file created in pdir during that attempt (so
    a half-finished experiment/paper doesn't linger or corrupt a later
    resume), and retry up to `retries` times before raising."""
    env = {k_: v_ for k_, v_ in os.environ.items() if k_ != "CLAUDECODE"}
    for attempt in range(1, retries + 1):
        before = {p.name for p in pdir.iterdir()} if pdir.exists() else set()
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                errors="replace", cwd=str(pdir), env=env,
                                shell=(sys.platform == "win32"))
        timed_out = False
        try:
            stdout, stderr = proc.communicate(input=prompt, timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True)
            else:
                proc.kill()
            stdout, stderr = proc.communicate()  # reap, discard partial output
        result = SimpleNamespace(returncode=proc.returncode, stdout=stdout or "",
                                 stderr=stderr or "")
        if not timed_out and result.returncode == 0 and expect_file.exists():
            return result
        after = {p.name for p in pdir.iterdir()} if pdir.exists() else set()
        new_files = sorted(after - before)
        for name in new_files:
            path = pdir / name
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        reason = (f"timed out after {timeout}s" if timed_out else
                 f"exit {result.returncode}" if result.returncode != 0 else
                 f"{expect_file.name} not written")
        if attempt < retries:
            ml.log(f"{pdir.name}: attempt {attempt}/{retries} {reason}, "
                   f"cleaned up {len(new_files)} file(s) {new_files}, retrying")
            continue
        raise RuntimeError(
            f"{pdir.name}: {reason} on final attempt {attempt}/{retries}, "
            f"cleaned up {len(new_files)} file(s) {new_files} "
            f"(stdout tail: {result.stdout[-500:]!r})")
    raise RuntimeError("unreachable")  # loop always returns or raises

# --------------------------------------------------------------------------
# Local setup/author (bypass monkeylab.setup()/author_cycle(), which assume
# the /homes/kaleb/... remote layout and a `paperena autoresearch` CLI shape
# this checkout doesn't have). monkeylab.chat()/review_paper()/lessons_add()
# are path-independent (only PROXY, always 127.0.0.1:8899) and reused as-is.
# --------------------------------------------------------------------------
# Defaults to a "paperena-agent" checkout sitting next to this repo
# (monkey-experiments/../paperena-agent) — true on any machine that cloned
# both repos side by side. Override with PAPERENA_AGENT_DIR if yours lives
# elsewhere.
_MONKEY_EXPERIMENTS_ROOT = Path(__file__).resolve().parents[3]
PAPERENA_REPO_LOCAL = Path(os.environ.get(
    "PAPERENA_AGENT_DIR",
    str(_MONKEY_EXPERIMENTS_ROOT.parent / "paperena-agent")))
TEMPLATE_LOCAL = PAPERENA_REPO_LOCAL / "agents/local-rev-template"


def setup_local(name):
    d = PAPERENA_REPO_LOCAL / "agents" / name
    (d / "data/autoresearch").mkdir(parents=True, exist_ok=True)
    for f in ("config.json", "goal.md"):
        if not (d / f).exists():
            (d / f).write_text((TEMPLATE_LOCAL / f).read_text(encoding="utf-8"),
                               encoding="utf-8")
    return d


def author_with_tools(d, s, topics, lessons="", model="sonnet", skills=False):
    """Write v1.md for episode s: real WebSearch + Read/Write access, so the
    author surveys existing work on the topic and weighs candidate angles
    before drafting, instead of free-associating a paper straight from the
    topic string (the prior autoresearch._build_complete()-based behavior,
    which had no tools at all).

    skills=True (the skills-loop arm, see reflect_and_update_skills()) seeds
    this paper's directory with a real copy of the agent-level skills.md —
    written by the author itself at the end of each prior episode — and
    tells it to Read that file before drafting. Unlike `lessons` (embedded
    as prompt text, e2's mechanism), this tests whether the author agent
    consulting a self-maintained skills file via its own Read tool changes
    v1 quality over episodes. Mutually exclusive with `lessons` in practice
    (no experiment uses both), but not enforced — nothing stops combining
    them."""
    pdir = d / f"data/autoresearch/paper-{s}"
    v1 = pdir / "v1.md"
    if v1.exists():
        return v1.read_text(encoding="utf-8")
    pdir.mkdir(parents=True, exist_ok=True)
    topic = topics[s % len(topics)]
    lessons_block = (f"\n===== LESSONS FROM PRIOR REVIEWS (apply these) =====\n"
                     f"{lessons}\n\n") if lessons else ""
    skills_block = ""
    if skills:
        master_skills = d / "data/autoresearch/skills.md"
        skills_local = pdir / "skills.md"
        if master_skills.exists():
            shutil.copy(master_skills, skills_local)
        else:
            skills_local.write_text("", encoding="utf-8")
        skills_block = (
            "Before anything else, read skills.md in this directory using "
            "the Read tool — it contains lessons YOU distilled from the "
            "review process of earlier papers in this series (empty if "
            "this is the first paper). Apply anything relevant when "
            "choosing your direction and drafting.\n\n")
    prompt = (
        f"{lessons_block}{skills_block}"
        f"You are an AI research scientist. Your topic: {topic}\n\n"
        f"First, use WebSearch to survey existing work in this space — what "
        f"has been tried, and what the open problems or gaps are. Based on "
        f"that survey, consider and evaluate a few candidate directions for "
        f"a short empirical methods paper, then pick the most promising one.\n\n"
        f"Then implement your method (and any baselines you need) in Python "
        f"and RUN it with Bash to get real Results — you have Bash access "
        f"scoped to `python` and `pip install` in this directory. Compute "
        f"is CPU-only with a limited time budget: design something that "
        f"actually finishes in a few minutes (synthetic data, a toy/small "
        f"model, a small real dataset, or a lightweight simulation), not a "
        f"large-scale training run. Run everything IN THE FOREGROUND and "
        f"wait for it to finish — do NOT background a script (no `&`, no "
        f"nohup, no launching something and moving on), do not start a "
        f"monitor/watcher process and wait for it to notify you, and do not "
        f"end your turn saying you'll wait or pause for something to finish "
        f"or notify you later: this is a single one-shot session with no "
        f"later turn, so if you stop before a background job completes, "
        f"nothing from it is ever used — including monitor/watch output. "
        f"Save your code and its raw output/logs in this directory alongside the "
        f"paper, so the reviewer can inspect or re-run it. If something "
        f"errors or doesn't work as hoped, debug it or scale it down "
        f"further — report what you actually observed running it, not a "
        f"projected/hoped-for number.\n\n"
        f"TIME BUDGET: you have {TOOL_TIMEOUT_S}s total for this whole task "
        f"(survey + implement + run + write the paper), enforced as a hard "
        f"cutoff — if you're still running when it hits, your work is lost. "
        f"Budget at most {EXPERIMENT_BUDGET_S}s of that for implementing and "
        f"running experiments. Track this yourself: run `date +%s` in Bash "
        f"to record a start timestamp before you begin experimenting, and "
        f"check elapsed time (another `date +%s`) before starting any new "
        f"experimental step. The moment your TOTAL elapsed time (from the "
        f"very start of this task) passes {CHECKPOINT_S}s, STOP "
        f"experimenting immediately regardless of where you are, and write "
        f"up the paper NOW with whatever results you have — note honestly "
        f"what you didn't get to, rather than risk losing everything to the "
        f"hard cutoff.\n\n"
        f"Then write a short, self-contained methods paper on that direction "
        f"in Markdown: Title, Abstract, Introduction (including a Related "
        f"Work discussion informed by your search), Method, Results, "
        f"Conclusion, References. Be concrete about baselines, ablations, "
        f"and evaluation. Aim for about {TARGET_PAPER_CHARS} characters "
        f"total — be concise rather than exhaustive — and it MUST NOT exceed "
        f"{MAX_PAPER_CHARS} characters under any circumstance.\n\n"
        f"Write the complete paper to v1.md in this directory using the "
        f"Write tool.")
    cmd = ["claude", "-p", "--model", model,
           "--allowedTools", "Read Write Edit WebSearch Bash(python *) Bash(pip install *)",
           "--add-dir", str(pdir)]
    run_claude_tool(cmd, prompt, pdir, expect_file=v1)
    enforce_char_limit(pdir, "v1.md", model=model)
    tag = " + skills.md" if skills else ""
    ml.log(f"ep{s}: authored w/ search{tag} ({topic[:40]}...)")
    return v1.read_text(encoding="utf-8")


TOPICS = [  # phase-1 topics, for comparability with 01/03
    "Socio-technical aspects of AI",
    "Probabilistic methods in large language models",
]
S_EPISODES = 6
K_ROUNDS = 4
TARGET_PAPER_CHARS = 10_000  # intended/aimed-for length
MAX_PAPER_CHARS = 20_000     # hard cap, enforced post-write; the "~10%" soft
                              # instruction alone let papers reach 42K chars
                              # (see EXPERIMENT.md)


def enforce_char_limit(pdir, vname, limit=MAX_PAPER_CHARS, tries=2, model="sonnet"):
    """If vname is over the hard limit, ask the agent to trim it back down
    (real Edit access, same paper-dir scoping as author/revise_with_tools).
    A soft "keep it about the same length" instruction wasn't enough on its
    own — papers still grew round over round — so this backs it with an
    actual post-write check instead of just trusting the prompt.

    Best-effort: vname is already a valid, written paper by the time this
    runs, so a trim failure (timeout, non-zero exit) is logged and swallowed
    rather than raised — the run continues with the untrimmed version rather
    than losing an entire multi-episode run over a length-tidying step."""
    vpath = pdir / vname
    for attempt in range(tries):
        size = len(vpath.read_text(encoding="utf-8"))
        if size <= limit:
            return
        ml.log(f"{vname}: {size} chars over {limit} limit, trimming "
               f"(attempt {attempt + 1}/{tries})")
        prompt = (
            f"{vname} in this directory is {size} characters, over the "
            f"{limit}-character hard limit for this venue. Read it, then "
            f"edit it down to around {TARGET_PAPER_CHARS} characters (must "
            f"not exceed {limit}): tighten prose and cut redundant content, "
            f"but keep the same sections and don't drop substantive "
            f"claims/results — write the same content more concisely, don't "
            f"just delete sections. Save the trimmed version back to "
            f"{vname} using the Edit or Write tool.")
        cmd = ["claude", "-p", "--model", model,
               "--allowedTools", "Read Write Edit",
               "--disallowedTools", "Bash", "--add-dir", str(pdir)]
        env = {k_: v_ for k_, v_ in os.environ.items() if k_ != "CLAUDECODE"}
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace",
                                  cwd=str(pdir), env=env, timeout=580,
                                  shell=(sys.platform == "win32"))
        except subprocess.TimeoutExpired:
            ml.log(f"{vname}: trim attempt {attempt + 1} timed out, leaving "
                   f"as-is ({size} chars)")
            return
        if proc.returncode != 0:
            ml.log(f"{vname}: trim attempt {attempt + 1} exit "
                   f"{proc.returncode}, leaving as-is ({size} chars): "
                   f"{proc.stderr[-500:]}")
            return
    final_size = len(vpath.read_text(encoding="utf-8"))
    if final_size > limit:
        ml.log(f"{vname}: still {final_size} chars after {tries} trim "
               f"attempt(s) (limit {limit}) — leaving as-is")


def record(path, **row):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# Condensed from the stock multi-stage scientist_reviewer's stage prompts
# (paperena_agent/workers/scientist_reviewer/prompts.py, stages 02-04 + 06)
# into guidance for our single-shot review call, rather than replicating its
# separate read/section/novelty/rigor/draft/critique/finalize model calls.
REVIEW_GUIDELINES = """\
Before writing your JSON review, work through these dimensions (you don't
need to show this analysis — just let it inform your rating and the
strengths/weaknesses/questions you write):

- Section coverage: for each section present (Abstract, Introduction,
  Method, Results, Conclusion, References), note what it says, what's done
  well, and what's unclear, missing, or weak.
- Novelty & placement: is the contribution new, an incremental delta, or a
  restatement? Are the obvious prior works cited and contrasted? Name any
  specific missing comparisons (by author/topic if you can).
- Rigor: are claims supported by the evidence presented? Are baselines,
  datasets, metrics, and ablations appropriate? Are results reported with
  variance/seeds and significance testing? Is there enough detail
  (hyperparameters, data, code) to reproduce? Flag confounders, leakage,
  cherry-picking, or overclaiming with a quote or paraphrase.
- Calibration: your final rating must be consistent with the
  strengths/weaknesses you list — several substantive weaknesses and only
  mild strengths should not score 7+. Don't be overly harsh or overly
  lenient, and don't penalize the paper for not being work outside its own
  stated scope.
"""

# Opt-in addendum (--lenient-on-execution), kept separate from
# REVIEW_GUIDELINES rather than edited in place. Originally written for when
# the author had no compute at all (Results were necessarily projected).
# Now that author_with_tools/revise_with_tools actually execute real code,
# this is mostly stale — the author usually CAN produce real numbers, so
# leniency about "no real execution" no longer reflects reality by default.
# Left in place (still opt-in via --lenient-on-execution) for cases where
# execution genuinely wasn't feasible in the time/compute budget and the
# author said so honestly, rather than for judging fabricated numbers.
EXECUTION_LENIENCY_NOTE = """\
Note on experimental execution: the author may not always have been able to
run a real experiment to completion in the CPU/time budget available. If —
and only if — the paper is explicit that a specific number is a worked
estimate rather than an executed result, do not penalize that honesty on its
own; judge the reasoning behind the estimate instead. This does NOT apply to
numbers presented as real results — if the paper claims something was run
and you can't verify that from the code/logs in this directory, that's a
legitimate weakness to raise, not something to excuse.
"""

# Condensed from Paperena's verify-text / verify-citation / verify-triage
# skills (paperena_agent/harnesses/claude/plugin/skills/) — the same
# correction/retraction-worthy bar and "flagging should be rare" discipline
# those verifiers use, adapted for a reviewer that can now actually execute
# the author's code rather than only reading claims.
VERIFICATION_GUIDELINES = """\
You have Bash access to this directory — use it. If the paper references
code, scripts, data, or experiment logs here, read them and, where feasible
in a few minutes on CPU, RUN them to check whether the reported Results
actually match what the code produces. A mismatch between a claimed number
and what the code actually outputs when you run it is a real weakness —
quote the discrepancy and the command you ran to find it.

When deciding what counts as a reportable error (adapted from Paperena's
verify-text/verify-citation skills):
- Reserve real criticism for correction-worthy issues: a provably wrong
  result, a claim that contradicts the paper's own data/code, an
  invalidating methodology flaw (e.g. the wrong statistical test applied to
  the headline result), or numbers that are internally impossible or don't
  match what the code produces when you run it.
- Don't flag typos, style, missing citations, or "could be clearer" — those
  aren't correction-worthy and shouldn't move the rating.
- For novelty/attribution: only flag if the paper claims a contribution is
  new while ALSO stating (or citing) that it was already done, or invokes a
  citation to support a claim it plainly doesn't support — not just "could
  cite more work."
- A typical paper has zero or one real error of this kind. Flagging should
  be rare and specific — don't manufacture criticism to seem thorough, but
  don't go easy on something you actually verified is wrong either.
"""


def review_with_tools(pdir, k, extra_context="", lenient=False, model="sonnet",
                      out_name=None):
    """Tool-enabled review: real Read/Bash access, scoped to this paper's own
    directory (same isolation as author/revise_with_tools), so the reviewer
    can actually run any code/experiments the author left behind and check
    whether reported Results match what the code produces, instead of taking
    numbers on faith. Writes its review directly to round{k}_review.json via
    the Write tool rather than returning JSON in the chat reply — structured
    JSON-in-final-message got unreliable once a model is doing tool calls
    throughout (same lesson that moved revise off a text-only response).

    out_name overrides the output filename (default round{k}_review.json) —
    used by the multi-reviewer arm so N independent reviewers of the same
    round don't overwrite each other's file."""
    vname = "v1.md" if k == 1 else f"v{k}.md"
    out_name = out_name or f"round{k}_review.json"
    out_path = pdir / out_name
    leniency = EXECUTION_LENIENCY_NOTE if lenient else ""
    prompt = (
        f"{extra_context}"
        f"Read {vname} (the paper to review) in this directory using the "
        f"Read tool.\n\n{REVIEW_GUIDELINES}\n{VERIFICATION_GUIDELINES}\n"
        f"Run any verification code IN THE FOREGROUND and wait for it to "
        f"finish — do NOT background a script, do not start a "
        f"monitor/watcher process and wait for it to notify you, and do not "
        f"end your turn saying you'll wait or pause for something to finish "
        f"or notify you later: this is a single one-shot session with no "
        f"later turn, so if you stop before a background job completes, "
        f"nothing from it is ever used — including monitor/watch output.\n\n"
        f"TIME BUDGET: you have {TOOL_TIMEOUT_S}s total, enforced as a hard "
        f"cutoff — if you're still running when it hits, this review is "
        f"lost entirely. Verification should be quick (re-running existing "
        f"code, not open-ended exploration): track elapsed time yourself "
        f"with `date +%s`, and if you're not done verifying by "
        f"{CHECKPOINT_S}s elapsed, stop trying to verify further and write "
        f"your review NOW with whatever you were able to check, noting "
        f"what you didn't have time to verify.\n\n"
        f"{leniency}\n{ml.REVIEW_SPEC}\n\n"
        f"Write your review as that exact JSON object to {out_name} "
        f"in this directory using the Write tool — the file's contents "
        f"should be the JSON only, no fences, no prose around it.")
    cmd = ["claude", "-p", "--model", model,
           "--allowedTools", "Read Write Edit Bash(python *) Bash(pip install *)",
           "--add-dir", str(pdir)]
    run_claude_tool(cmd, prompt, pdir, expect_file=out_path)
    try:
        return ml.parse_review(out_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, AttributeError) as exc:
        out_path.unlink()  # don't leave a corrupt file blocking resume
        raise RuntimeError(f"review_with_tools: {out_path.name} wasn't "
                           f"valid JSON: {exc}") from exc


def revise_with_tools(pdir, k, lessons="", model="sonnet", review_names=None):
    """Default revise path: real Read/Write access to v{k}.md and
    round{k}_review.json, scoped to this paper's own directory only, instead
    of embedding the paper as text in a chat prompt. The text-embedded
    REVISE_PROMPT approach was found to degrade into refusal or a change-log
    summary instead of an actual full revision (observed on paper-1: the
    model didn't trust/use the embedded text, tried blocked tool calls, then
    refused to fabricate on request); giving it real file access fixed it.

    lessons (e2 only) is short curated text, passed inline in the prompt
    rather than as a file read — unlike full papers, there's no evidence the
    model distrusts short embedded text, and --add-dir is deliberately kept
    scoped to just this paper's own directory (see history_context()'s
    per-paper isolation), not the agent-level lessons.md location.

    review_names overrides which review file(s) to read (default
    [round{k}_review.json]) — the multi-reviewer arm passes all N
    independent reviewers' files so the author sees every review, not just
    one, before revising."""
    vname = "v1.md" if k == 1 else f"v{k}.md"
    v_next = pdir / f"v{k + 1}.md"
    review_names = review_names or [f"round{k}_review.json"]
    if len(review_names) == 1:
        review_instruction = (f"Read {vname} (your paper) and "
                              f"{review_names[0]} (the peer review of it) "
                              f"in this directory using the Read tool.")
    else:
        names_list = ", ".join(review_names)
        review_instruction = (
            f"Read {vname} (your paper) and all {len(review_names)} "
            f"independent peer reviews of it in this directory using the "
            f"Read tool: {names_list}. These are {len(review_names)} "
            f"separate reviewers who each read the paper independently "
            f"with no knowledge of each other's review — they may "
            f"disagree. Address weaknesses/questions raised by ANY of "
            f"them; where multiple reviewers independently raise the same "
            f"or a similar point, prioritize it, since independent "
            f"agreement is a stronger signal than a single reviewer's "
            f"opinion.")
    lessons_block = (f"\n===== LESSONS FROM PRIOR REVIEWS (apply these) =====\n"
                     f"{lessons}\n\n") if lessons else ""
    prompt = (
        f"{lessons_block}"
        f"{review_instruction} If the "
        f"review flags a result, or you're adding one, and there's code in "
        f"this directory (or you need to write/adjust some), you have Bash "
        f"access scoped to `python` and `pip install` — actually run it "
        f"and use the real output, don't just tweak the prose. Run "
        f"everything IN THE FOREGROUND and wait for it to finish — do NOT "
        f"background a script (no `&`, no nohup), do not start a "
        f"monitor/watcher process and wait for it to notify you, and do not "
        f"end your turn saying you'll wait or pause for something to finish "
        f"or notify you later: this is a single one-shot session with no "
        f"later turn, so if you stop before a background job completes, "
        f"nothing from it is ever used — including monitor/watch output.\n\n"
        f"TIME BUDGET: you have {TOOL_TIMEOUT_S}s total, enforced as a hard "
        f"cutoff — if you're still running when it hits, your work is lost. "
        f"Budget at most {EXPERIMENT_BUDGET_S}s for running/adjusting code; "
        f"track this yourself with `date +%s` before you start and before "
        f"any new experimental step. Once your TOTAL elapsed time passes "
        f"{CHECKPOINT_S}s, stop experimenting and write up the revision NOW "
        f"with whatever you have.\n\n"
        f"Revise the paper to directly address the review's weaknesses and "
        f"questions, then write the COMPLETE revised paper — not a diff, "
        f"not a change-log, not a summary of changes — to v{k + 1}.md using "
        f"the Write tool. Aim to stay around {TARGET_PAPER_CHARS} characters "
        f"total — if you're adding content to address the review, cut or "
        f"tighten something else to make room, don't just grow the paper — "
        f"and it MUST NOT exceed {MAX_PAPER_CHARS} characters under any "
        f"circumstance.")
    cmd = ["claude", "-p", "--model", model,
           "--allowedTools", "Read Write Edit Bash(python *) Bash(pip install *)",
           "--add-dir", str(pdir)]
    run_claude_tool(cmd, prompt, pdir, expect_file=v_next)
    enforce_char_limit(pdir, v_next.name, model=model)
    return v_next.read_text(encoding="utf-8")


def history_context(pdir, k):
    """Prior versions + their reviews, for reviewing round k (k>1). Lets the
    reviewer judge the revision itself, not just re-litigate v1 from scratch."""
    if k == 1:
        return ""
    parts = ["===== REVISION HISTORY (earlier versions of this paper and the "
             "reviews that drove each revision — judge whether THIS version "
             "actually addresses them, not just whether it reads well) ====="]
    for j in range(1, k):
        vname = "v1.md" if j == 1 else f"v{j}.md"
        vpath, rpath = pdir / vname, pdir / f"round{j}_review.json"
        if not (vpath.exists() and rpath.exists()):
            continue
        review = json.loads(rpath.read_text(encoding="utf-8"))
        parts.append(f"--- {vname} ---\n{vpath.read_text(encoding='utf-8')}\n\n"
                      f"--- review of {vname} (rating "
                      f"{review.get('rating')}) ---\n{ml.format_review(review)}")
    parts.append(
        "===== END REVISION HISTORY =====\n"
        "When rating the CURRENT version below, explicitly weigh how it "
        "responds to this history: give credit for weaknesses/questions from "
        "prior reviews that are now resolved, and penalize ones that remain "
        "unaddressed or issues newly introduced by the revision itself.")
    return "\n\n".join(parts) + "\n\n"


def reviewer_history_context(pdir, k, reviewer_i):
    """Like history_context(), but scoped to ONE reviewer's own past reviews
    only (round{j}_review_{reviewer_i}.json), not the other N-1 reviewers'.
    Used by the multi-reviewer-with-history arm: each of the N reviewers
    tracks whether ITS OWN prior feedback was addressed, but never sees what
    the other reviewers wrote — independence across reviewers is preserved,
    only the within-reviewer history is added back in."""
    if k == 1:
        return ""
    parts = [f"===== YOUR OWN REVIEW HISTORY (earlier versions of this paper "
             f"and YOUR OWN reviews of them — other reviewers' opinions are "
             f"not shown to you; judge whether THIS version addresses YOUR "
             f"feedback, not just whether it reads well) ====="]
    for j in range(1, k):
        vname = "v1.md" if j == 1 else f"v{j}.md"
        rpath = pdir / f"round{j}_review_{reviewer_i}.json"
        vpath = pdir / vname
        if not (vpath.exists() and rpath.exists()):
            continue
        review = json.loads(rpath.read_text(encoding="utf-8"))
        parts.append(f"--- {vname} ---\n{vpath.read_text(encoding='utf-8')}\n\n"
                      f"--- your review of {vname} (rating "
                      f"{review.get('rating')}) ---\n{ml.format_review(review)}")
    parts.append(
        "===== END YOUR OWN REVIEW HISTORY =====\n"
        "When rating the CURRENT version below, explicitly weigh how it "
        "responds to YOUR OWN prior feedback above: give credit for "
        "weaknesses/questions you raised that are now resolved, and "
        "penalize ones that remain unaddressed or issues newly introduced "
        "by the revision itself.")
    return "\n\n".join(parts) + "\n\n"


def reflect_and_update_skills(pdir, d, model="sonnet"):
    """End-of-episode step for the skills-loop arm: the author agent itself
    (real Read/Write access, not a separate ml.chat distill call like e2's
    lessons_on) reads every version and review of the paper it just
    finished, then rewrites skills.md with general, reusable lessons for
    drafting FUTURE papers — not paper-specific facts. Writes the new
    content to skills_updated.md (a file that doesn't exist yet) rather
    than editing skills.md in place, so run_claude_tool's expect_file
    existence check is a meaningful completion signal (skills.md itself may
    already exist and be non-empty going in, so its existence alone
    wouldn't prove anything happened). The result is then promoted to the
    agent-level skills.md, which author_with_tools(skills=True) seeds into
    each subsequent paper's directory.

    Idempotent/resumable: skipped if skills_updated.md already exists."""
    skills_updated = pdir / "skills_updated.md"
    if skills_updated.exists():
        return
    skills_local = pdir / "skills.md"
    if not skills_local.exists():
        skills_local.write_text("", encoding="utf-8")
    review_files = sorted(pdir.glob("round*_review.json"))
    names = ", ".join(f.name for f in review_files)
    prompt = (
        f"You just finished a full review-revise cycle for this paper. "
        f"Read every version of it (v1.md, v2.md, ...) and every review in "
        f"this directory ({names}) using the Read tool, to see what issues "
        f"came up across rounds and how well (or poorly) they were "
        f"addressed by each revision. Also read the experiment code, logs, "
        f"and result files (.py, .csv, .json, .txt, etc.) in this "
        f"directory that the reviews reference or verified against — "
        f"lessons grounded in what the code/results actually show are more "
        f"useful than lessons inferred only from the reviews' prose.\n\n"
        f"Then read skills.md in this directory — lessons distilled from "
        f"earlier papers in this series, may be empty if this is the first "
        f"one. Write an UPDATED, COMPLETE version of it to "
        f"skills_updated.md using the Write tool: add any new, general, "
        f"reusable lessons from THIS paper's review process that would "
        f"help you write a BETTER FIRST DRAFT of a future paper on a "
        f"different topic — recurring rigor gaps, framing mistakes, things "
        f"the reviewer consistently flagged. Do not include paper- or "
        f"topic-specific facts or results, only general lessons about what "
        f"makes these papers score well. Keep it concise (a bullet list is "
        f"fine); revise or remove an existing bullet if this paper's "
        f"experience refines or contradicts it rather than just appending "
        f"under it. If skills.md already covers a lesson well, carry it "
        f"forward unchanged rather than rewording it for its own sake.")
    cmd = ["claude", "-p", "--model", model,
           "--allowedTools", "Read Write Edit",
           "--add-dir", str(pdir)]
    run_claude_tool(cmd, prompt, pdir, expect_file=skills_updated)
    master_skills = d / "data/autoresearch/skills.md"
    shutil.copy(skills_updated, master_skills)
    ml.log(f"{pdir.name}: skills.md updated")


def run_episode(d, s, gpu, lessons_on, ledger, lenient=False,
                author_model="sonnet", reviewer_model="sonnet",
                reviewer_history=True, skills_on=False):
    lessons_path = d / "data/autoresearch/lessons.md"
    start_lessons = (lessons_path.read_text(encoding="utf-8").strip()
                     if lessons_on and lessons_path.exists() else "")
    author_with_tools(d, s, TOPICS, lessons=start_lessons, model=author_model,
                      skills=skills_on)
    pdir = d / f"data/autoresearch/paper-{s}"
    for k in range(1, K_ROUNDS + 2):        # K revise rounds + 1 final review
        vk = pdir / (f"v{k}.md" if k > 1 else "v1.md")
        rev_file = pdir / f"round{k}_review.json"
        if rev_file.exists():
            review = json.loads(rev_file.read_text(encoding="utf-8"))
        else:
            ctx = history_context(pdir, k) if reviewer_history else ""
            review = review_with_tools(pdir, k, extra_context=ctx,
                                       lenient=lenient, model=reviewer_model)
            rev_file.write_text(json.dumps(review, indent=2), encoding="utf-8")
            record(ledger, episode=s, round=k, version=vk.name,
                   rating=review.get("rating"),
                   confidence=review.get("confidence"))
            ml.log(f"ep{s} round{k}: rating {review.get('rating')}")
        if k > K_ROUNDS:                    # final review only — no revise
            break
        if lessons_on and not (pdir / f".distilled{k}").exists():
            lesson = ml.chat(ml.DISTILL_PROMPT.format(
                review=ml.format_review(review)), model=author_model)
            n = ml.lessons_add(d, lesson)
            (pdir / f".distilled{k}").write_text("", encoding="utf-8")
            ml.log(f"ep{s} round{k}: distilled -> {n} lessons")
        v_next = pdir / f"v{k + 1}.md"
        if not v_next.exists():
            lessons_text = (lessons_path.read_text(encoding="utf-8").strip()
                            if lessons_on and lessons_path.exists() else "")
            revise_with_tools(pdir, k, lessons=lessons_text, model=author_model)
    if skills_on:
        reflect_and_update_skills(pdir, d, model=author_model)


def run_episode_multi_reviewer(d, s, gpu, ledger, n_reviewers,
                               author_model="sonnet", reviewer_model="sonnet"):
    """N independent reviewers per round instead of one: each reads only
    the current version (no history_context — reviewers don't see each
    other's review or prior rounds, matching the no-history condition) and
    writes its own round{k}_review_{i}.json. The author then revises against
    all N reviews at once (revise_with_tools' review_names param), so
    agreement across independent reviewers becomes a signal the author can
    weigh, rather than a single reviewer's opinion being the whole story."""
    author_with_tools(d, s, TOPICS, model=author_model)
    pdir = d / f"data/autoresearch/paper-{s}"
    for k in range(1, K_ROUNDS + 2):        # K revise rounds + 1 final review
        vk = pdir / (f"v{k}.md" if k > 1 else "v1.md")
        review_names = [f"round{k}_review_{i}.json" for i in range(1, n_reviewers + 1)]
        ratings = []
        for i, name in enumerate(review_names, start=1):
            rev_file = pdir / name
            if rev_file.exists():
                review = json.loads(rev_file.read_text(encoding="utf-8"))
            else:
                review = review_with_tools(pdir, k, model=reviewer_model,
                                           out_name=name)
                rev_file.write_text(json.dumps(review, indent=2), encoding="utf-8")
                record(ledger, episode=s, round=k, reviewer=i,
                       version=vk.name, rating=review.get("rating"),
                       confidence=review.get("confidence"))
            ratings.append(review.get("rating"))
        mean_rating = sum(ratings) / len(ratings)
        ml.log(f"ep{s} round{k}: ratings {ratings} (mean {mean_rating:.1f})")
        if k > K_ROUNDS:                    # final review only — no revise
            break
        v_next = pdir / f"v{k + 1}.md"
        if not v_next.exists():
            revise_with_tools(pdir, k, model=author_model,
                              review_names=review_names)


def run_episode_multi_reviewer_history(d, s, gpu, ledger, n_reviewers,
                                       author_model="sonnet", reviewer_model="sonnet"):
    """Same as run_episode_multi_reviewer, except each of the N reviewers is
    given ITS OWN review history (reviewer_history_context()) — every prior
    version of the paper and that same reviewer's own past reviews of it —
    while still never seeing the other N-1 reviewers' opinions, past or
    present. Isolates "does per-reviewer memory help" from "does averaging
    N independent judges help" (run_episode_multi_reviewer)."""
    author_with_tools(d, s, TOPICS, model=author_model)
    pdir = d / f"data/autoresearch/paper-{s}"
    for k in range(1, K_ROUNDS + 2):        # K revise rounds + 1 final review
        vk = pdir / (f"v{k}.md" if k > 1 else "v1.md")
        review_names = [f"round{k}_review_{i}.json" for i in range(1, n_reviewers + 1)]
        ratings = []
        for i, name in enumerate(review_names, start=1):
            rev_file = pdir / name
            if rev_file.exists():
                review = json.loads(rev_file.read_text(encoding="utf-8"))
            else:
                ctx = reviewer_history_context(pdir, k, i)
                review = review_with_tools(pdir, k, extra_context=ctx,
                                           model=reviewer_model, out_name=name)
                rev_file.write_text(json.dumps(review, indent=2), encoding="utf-8")
                record(ledger, episode=s, round=k, reviewer=i,
                       version=vk.name, rating=review.get("rating"),
                       confidence=review.get("confidence"))
            ratings.append(review.get("rating"))
        mean_rating = sum(ratings) / len(ratings)
        ml.log(f"ep{s} round{k}: ratings {ratings} (mean {mean_rating:.1f})")
        if k > K_ROUNDS:                    # final review only — no revise
            break
        v_next = pdir / f"v{k + 1}.md"
        if not v_next.exists():
            revise_with_tools(pdir, k, model=author_model,
                              review_names=review_names)
