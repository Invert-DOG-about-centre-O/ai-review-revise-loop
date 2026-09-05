# Lessons for writing a stronger first draft

## Statistical rigor (most frequently flagged issue, every round)
- Never report a "trend" or "effect" from n=2 (or any tiny sample) per condition without variance/CI. Reviewers immediately treat means-of-2 as unsupported even when the sign of the effect is consistent.
- From the first draft, run enough repetitions (aim for ~10+ per condition, not 2-5) to compute standard deviations and confidence intervals — don't wait for reviewers to ask for this in a later round.
- Prefer a formal significance test (e.g., a t-test between conditions) over an informal heuristic like "confidence intervals don't overlap." Reviewers explicitly downgrade heuristic-overlap arguments and reward proper tests.
- When a result hinges on a specific comparison (e.g., "condition A differs from condition B"), test that exact comparison directly rather than eyeballing it from a table.
- If a "sharp jump" or "discontinuity" between two adjacent conditions is central to the claim, proactively check it isn't a small-sample/seed artifact (e.g., rerun with a fresh, disjoint batch of seeds and pool) — reviewers ask this every time a jump is reported, so do it before submission.
- Match the rigor of secondary/robustness experiments to the rigor of the primary experiment. Running the main sweep with 10+ seeds and formal tests but a "robustness check" with only ~5 seeds and no test is a recurring, easily-avoided asymmetry that reviewers flag.
- Distinguish statistical significance from practical/effect-size significance. A statistically clean result built on an already-small absolute effect should say so explicitly rather than letting the significance machinery imply the effect is large or consequential.

## Ruling out confounds
- Identify the single most obvious alternative explanation for your headline finding (e.g., "is this really effect X or just an artifact of insufficient training/tuning/sample size?") and test it directly and explicitly, don't just assert it away in prose.
- One check at one setting is "evidence, not proof" — say so plainly, and consider a second, complementary check (e.g., a different diagnostic, not just a scaled-up version of the same one) to strengthen a confound-elimination claim.
- Don't let a confound-elimination claim in the conclusion be broader ("X is not explained by Y in general") than what was actually tested (X was checked against Y only at one setting/scale) — keep the conclusion's phrasing exactly as narrow as the evidence.

## Ablations and coverage
- If an ablation or secondary analysis is only run at the extremes of a parameter range, reviewers will assume (correctly) that the interesting middle/interaction region is unexamined — either cover the full range from the start, or explicitly flag the gap as untested rather than implying full coverage.
- When later filling a coverage gap identified by reviewers, check whether the new data fits the previously-stated summary (e.g., "roughly X–Y range") — recompute the summary rather than leaving a stale characterization that a reviewer can falsify by rereading your own table.

## Reproducibility and internal consistency
- Ship the exact code, seeds, and raw result files used to produce every reported number; reviewers routinely rerun the pipeline end-to-end and compare bit-for-bit or number-for-number — any mismatch is immediately caught and costs credibility, so re-verify each prose claim against the underlying data table before submission.
- Double-check every summary statistic quoted in prose (fold-changes, ratios, ranges like "roughly 6–8x") against the actual per-condition numbers in your own tables; misstatements of your own reported data are an easy, embarrassing, and entirely avoidable error that reviewers catch by simple arithmetic.
- Avoid stating hardware/environment-dependent numbers (e.g., wall-clock runtime) as precise claims; either drop them or qualify them as approximate and environment-dependent, since reviewers who rerun on different hardware will flag any mismatch.
- Use two independent metrics/methods to corroborate a central quantitative finding when feasible (e.g., an analytically-grounded metric alongside a standard empirical one); agreement between them is treated as a genuine internal robustness check and a real strength.

## Framing, scope, and overclaiming
- Never let the title or abstract state a finding in more general terms than the limitations section later admits. Reviewers explicitly note when hedges are "buried" in the discussion while the abstract/title reads as an unconditional, general claim — scope the headline claim itself, not just the caveats.
- When a result is only demonstrated on one narrow synthetic/toy setup (single dataset family, single architecture, single hyperparameter configuration), say explicitly what is and isn't shown to generalize, and do not extrapolate to "real-world" or "practical" settings that were never tested.
- If robustness checks on a second independent instance/dataset show that a precise headline number (e.g., an exact threshold, crossover point, or magnitude) does NOT replicate, don't quietly keep emphasizing that number — explicitly narrow the generalizable claim to only what did replicate (e.g., the qualitative direction of an effect) and say so in the abstract and conclusion, not just in a discussion paragraph. This kind of self-correction is explicitly rewarded by reviewers ("commendable," "rare," "the right response to the data").
- Watch for numeric claims like "clear, monotonic" or "consistent" that are contradicted by the underlying per-run data (e.g., non-monotonicity between two adjacent conditions in individual runs, masked by an average) — describe noisy data honestly rather than smoothing it into a cleaner-sounding narrative.
- Label any proposed mechanism/explanation for a pattern as explicitly untested/speculative if it was not itself the target of an experiment — don't let a plausible-sounding "just-so story" read as an established finding.
- Practical/actionable recommendations should be scoped exactly to the tested conditions; don't generalize a recommendation (e.g., "practitioners should do X") beyond the narrow setting where X was actually verified.

## What separated well-received drafts from weak ones
- Drafts that treated each review round as a checklist to mechanically satisfy (add seeds, add a test, add a data point) scored fine but kept accumulating new critiques of the same shape (still small n, still narrow scope) — anticipating these issues in the first draft (adequate sample size, formal tests, full-range ablations, explicit confound checks, and appropriately hedged framing) avoids multiple rounds of nearly-identical feedback.
- The single biggest driver of improving reviewer sentiment across rounds was honest self-correction: explicitly walking back an earlier overclaim once new evidence complicated it, rather than defending the original framing or quietly minimizing the contradiction.
- Even in the best-received draft, reviewers still had legitimate complaints about narrow scope (single task family/domain, single method/architecture, only two robustness instances) — full breadth of validation across domains/methods is rarely achievable in one paper, so proactively and precisely stating that scope boundary is the realistic bar, not eliminating all scope limitations.
