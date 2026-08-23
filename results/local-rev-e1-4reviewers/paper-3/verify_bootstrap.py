import json, numpy as np
with open("raw_results.json") as f:
    data = json.load(f)
records = data["records"]
correct = np.array([r["correct"] for r in records], dtype=bool)
wrong = ~correct
mean_logp = np.array([r["mean_logp"] for r in records])
sem_ent = np.array([r["semantic_entropy"] for r in records])
sc_conf = np.array([r["self_consistency_conf"] for r in records])

def auroc(scores, is_positive):
    scores = np.asarray(scores, dtype=float)
    pos = scores[is_positive]; neg = scores[~is_positive]
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores)+1)
    sorted_scores = scores[order]
    i=0
    while i < len(sorted_scores):
        j=i
        while j+1 < len(sorted_scores) and sorted_scores[j+1]==sorted_scores[i]:
            j+=1
        if j>i:
            avg = ranks[order[i:j+1]].mean(); ranks[order[i:j+1]] = avg
        i=j+1
    rank_sum_pos = ranks[is_positive].sum()
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum_pos - n_pos*(n_pos+1)/2)/(n_pos*n_neg)

rng = np.random.default_rng(123)
n = len(records)
boot_logp, boot_sem, boot_sc = [], [], []
for _ in range(2000):
    idx = rng.integers(0, n, n)
    w = wrong[idx]
    if w.sum()==0 or w.sum()==n: continue
    boot_logp.append(auroc(-mean_logp[idx], w))
    boot_sem.append(auroc(sem_ent[idx], w))
    boot_sc.append(auroc(-sc_conf[idx], w))
boot_logp=np.array(boot_logp); boot_sem=np.array(boot_sem); boot_sc=np.array(boot_sc)
print("logp AUROC 95% CI:", np.percentile(boot_logp,[2.5,97.5]), "mean", boot_logp.mean())
print("sem_ent AUROC 95% CI:", np.percentile(boot_sem,[2.5,97.5]), "mean", boot_sem.mean())
print("self_cons AUROC 95% CI:", np.percentile(boot_sc,[2.5,97.5]), "mean", boot_sc.mean())
diff = boot_logp - boot_sem
print("logp - sem_ent diff 95% CI:", np.percentile(diff,[2.5,97.5]), "P(diff<=0)=", (diff<=0).mean())
diff2 = boot_logp - boot_sc
print("logp - self_cons diff 95% CI:", np.percentile(diff2,[2.5,97.5]), "P(diff<=0)=", (diff2<=0).mean())
