import subprocess, sys
# just re-run experiment.py logic but save to a different file
src = open("experiment.py").read()
src = src.replace('with open("raw_results.json", "w") as f:', 'with open("raw_results_rerun.json", "w") as f:')
exec(compile(src, "experiment.py", "exec"))
