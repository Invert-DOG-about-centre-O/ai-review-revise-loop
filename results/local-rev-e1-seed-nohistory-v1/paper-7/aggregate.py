import numpy as np, json
from experiment import full_run, auroc_for

def pooled(seeds, epochs, noise_p=0.0):
    all_rows = []
    accs = []
    per_seed_auc = {k: [] for k in ("token_entropy", "lexical_entropy", "semantic_entropy")}
    for seed in seeds:
        kb_rows = []
        r = full_run(seed, epochs=epochs, restrict_to_seen=True, noise_p=noise_p)
        accs.append(r["acc"])
        for k in per_seed_auc:
            per_seed_auc[k].append(r["auroc"][k])
    out = {}
    for k, v in per_seed_auc.items():
        v = [x for x in v if not np.isnan(x)]
        out[k] = dict(mean=float(np.mean(v)), std=float(np.std(v)))
    return dict(acc_mean=float(np.mean(accs)), auroc=out)

if __name__ == "__main__":
    res = pooled(range(100, 105), epochs=19)
    print(json.dumps(res, indent=2))
    with open("results_ablation_summary.json", "w") as f:
        json.dump(res, f, indent=2)
