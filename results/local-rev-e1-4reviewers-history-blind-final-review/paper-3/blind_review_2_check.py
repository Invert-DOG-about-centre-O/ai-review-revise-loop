import json, numpy as np
d = json.load(open("analysis_summary.json"))
per = d["per_seed"]
for key in ["auc_mlp","auc_se","auc_sc","auc_ent1"]:
    vals = np.array([r[key] for r in per])
    print(key, vals.mean(), vals.std())

mm = json.load(open("multiseed_results.json"))
se_means = [r["diff_mlp_se"]["mean"] for r in mm]
se_lo = [r["diff_mlp_se"]["lo"] for r in mm]
se_hi = [r["diff_mlp_se"]["hi"] for r in mm]
sc_means = [r["diff_mlp_sc"]["mean"] for r in mm]
sc_lo = [r["diff_mlp_sc"]["lo"] for r in mm]
sc_hi = [r["diff_mlp_sc"]["hi"] for r in mm]

def meta(means, los, his):
    means=np.array(means); los=np.array(los); his=np.array(his)
    se = (his-los)/(2*1.96)
    w = 1/se**2
    pooled_mean = np.sum(w*means)/np.sum(w)
    pooled_se = np.sqrt(1/np.sum(w))
    z = pooled_mean/pooled_se
    from scipy.stats import norm
    p = 2*(1-norm.cdf(abs(z)))
    return pooled_mean, pooled_se, z, p

print("meta se", meta(se_means, se_lo, se_hi))
print("meta sc", meta(sc_means, sc_lo, sc_hi))
