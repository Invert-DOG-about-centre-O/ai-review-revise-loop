import json, numpy as np
d = json.load(open("round2_review_1_results_rerun.json"))
main = d["main"]["results"]
rob = d["robustness"]["results"]

widths = [2,4,16,64,256]

def agg(results, ls, width):
    rows = [r for r in results if r["label_smoothing"]==ls and r["width"]==width]
    Ts = np.array([r["T_star"] for r in rows])
    accs = np.array([r["acc"] for r in rows])
    eces = np.array([r["ece"] for r in rows])
    beces = np.array([r["bayes_ece"] for r in rows])
    ece_ts = np.array([r["ece_after_ts"] for r in rows])
    return dict(n=len(rows), acc=accs.mean(), ece=eces.mean(), bece=beces.mean(),
                T_mean=Ts.mean(), T_std=Ts.std(ddof=1), T_sem=Ts.std(ddof=1)/np.sqrt(len(Ts)),
                ece_ts=ece_ts.mean())

print("=== Main sweep (data_seed=0), no LS, n=10 seeds ===")
for w in widths:
    a = agg(main, 0.0, w)
    ci95 = 1.96*a['T_sem']
    print(f"w={w:4d} n={a['n']} acc={a['acc']:.3f} ece={a['ece']:.4f} bece={a['bece']:.4f} T*={a['T_mean']:.3f}+/-{a['T_std']:.3f} (95%CI +/-{ci95:.3f}) ece_ts={a['ece_ts']:.4f}")

print("\n=== Label smoothing sweep (data_seed=0), full width range, n=10 seeds ===")
for w in widths:
    a0 = agg(main, 0.0, w)
    a1 = agg(main, 0.1, w)
    print(f"w={w:4d}  T*(noLS)={a0['T_mean']:.3f}  T*(LS)={a1['T_mean']:.3f}+/-{a1['T_std']:.3f}  ece(noLS)={a0['ece']:.4f} ece(LS)={a1['ece']:.4f} ece_ts(LS)={a1['ece_ts']:.4f}")

print("\n=== Robustness check: second data-generating draw (data_seed=1), n=5 seeds ===")
for w in widths:
    rows = [r for r in rob if r["width"]==w]
    Ts = np.array([r["T_star"] for r in rows])
    accs = np.array([r["acc"] for r in rows])
    print(f"w={w:4d} n={len(rows)} acc={accs.mean():.3f} T*={Ts.mean():.3f}+/-{Ts.std(ddof=1):.3f}")
