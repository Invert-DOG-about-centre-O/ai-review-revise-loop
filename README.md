# Author-reviewer feedback loops for better AI-generated research

What happens to scientific paper quality when AI agents receive feedback from other AI agents? This GitHub repository consists of scripts for and results of experiments run to evaluate this question. Agents used are primarily Claude Sonnet 5, though some generated papers and reviews are also by Opus 5 and Haiku 4.5.

## Variations

Reviewer modes: history-aware (HA) or history-blind (HB) across reviews of the same paper. Either the reviewer agents can see previous reviews and paper versions and compare changes, or not.

Author modes: lessons on (L) or off (NL) across paper cycles. Either the author agent is allowed to transfer lessons across papers or not.

Number of independent reviewers (N): 1 or 4.

## File guide

This directory consists of scripts to run experiments, a script to generate plots and a results folder, which consists of papers and reviews generated in our experiments.

Scripts to run experiments all use `driver_lib.py`, and the naming convention uses HA/HB, L/NL and N1/N4 to refer to the variations above. So, `run_N4_HA_L.py` means 4 independent history-aware reviewers, and the author can transfer lessons across papers. All scripts are 3 rounds of revision i.e. 4 drafts and 4 reviews, with NL being 1 paper to be generated and L being 5 papers to be generated.

`plot_scores.py` is a plotting script.

The `results` folder consists of generated papers and reviews.
