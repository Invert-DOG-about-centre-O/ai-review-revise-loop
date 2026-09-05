from sim import run_condition
for lam in [0.50,0.60,0.65,0.70,0.75,0.80,1.00]:
    r = run_condition('x','REG', lam=lam, q=1.0)
    print(lam, r['final_alpha_mean'])
