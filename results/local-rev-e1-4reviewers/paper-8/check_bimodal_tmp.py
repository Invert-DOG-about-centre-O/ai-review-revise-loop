from sim_followup import run_condition, RNG_SEED0
for s in range(30):
    a, accs, trusts, alog = run_condition('REG_FAIR', RNG_SEED0+s, 0.65, 0.75, lam=0.8, n_rounds=20000, log_every=100)
    r10000 = alog[100]
    r20000 = alog[-1]
    print(s, round(r10000,3), round(r20000,3))
