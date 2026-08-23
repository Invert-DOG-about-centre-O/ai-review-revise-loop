import numpy as np, json, sys
from sycophancy_sim import train, evaluate

def sweep_lambda():
    lambdas = [0.0, 0.05, 0.1, 0.15, 0.3, 0.6]
    out = {}
    for lam in lambdas:
        rows = []
        for seed in range(8):
            w_c, g, h = train(seed, l2_lambda=lam)
            rows.append(evaluate(w_c, g, h, seed))
        syco_a = np.array([r["syco_a"] for r in rows])
        g_arr = np.array([r["g"] for r in rows])
        out[str(lam)] = dict(syco_a_mean=float(syco_a.mean()), syco_a_std=float(syco_a.std()),
                              g_mean=float(g_arr.mean()))
    return out

def sweep_precommit():
    qs = [0.0, 0.25, 0.5, 0.75, 1.0]
    out = {}
    for q in qs:
        rows = []
        for seed in range(8):
            w_c, g, h = train(seed, precommit_q=q)
            rows.append(evaluate(w_c, g, h, seed))
        syco_a = np.array([r["syco_a"] for r in rows])
        g_arr = np.array([r["g"] for r in rows])
        out[str(q)] = dict(syco_a_mean=float(syco_a.mean()), syco_a_std=float(syco_a.std()),
                            g_mean=float(g_arr.mean()))
    return out

if __name__ == "__main__":
    result = dict(lambda_sweep=sweep_lambda(), precommit_q_sweep=sweep_precommit())
    print(json.dumps(result, indent=2))
    with open("sensitivity_results.json", "w") as f:
        json.dump(result, f, indent=2)
