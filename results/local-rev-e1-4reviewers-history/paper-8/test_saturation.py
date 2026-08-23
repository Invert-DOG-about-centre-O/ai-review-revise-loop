import numpy as np
from sim import run_one, N_SEEDS

def time_to_saturation(cond, thresh=0.95, **kw):
    times = []
    for s in range(N_SEEDS):
        r = run_one(s, cond, return_trace=True, **kw)
        trace = r["alpha_trace"]
        hit = np.where(trace >= thresh)[0]
        times.append(int(hit[0]) if len(hit) else -1)
    return times

for name, cond, kw in [
    ("Transparency 0.8", "TRANSPARENCY", dict(transparency=0.8)),
    ("REG lam=0.5 q=1.0", "REG", dict(lam=0.5, q=1.0)),
    ("REG-fair lam=0.8 q=1.0", "REG", dict(lam=0.8, q=1.0)),
]:
    times = time_to_saturation(cond, 0.95, **kw)
    never = sum(1 for t in times if t == -1)
    hit_times = [t for t in times if t != -1]
    mean_t = np.mean(hit_times) if hit_times else float("nan")
    print("%s: %d/%d seeds ever reach alpha>=0.95; mean round-to-saturation among those = %.0f"
          % (name, N_SEEDS - never, N_SEEDS, mean_t))
