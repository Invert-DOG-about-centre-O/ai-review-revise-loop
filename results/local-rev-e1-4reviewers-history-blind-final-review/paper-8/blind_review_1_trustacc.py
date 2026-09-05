from sim import run_condition
r = run_condition('regfair','REG', lam=0.8, q=1.0)
print('REG-fair 0.8,1.0: gap=', r['calibration_gap_mean'], 'gap_acctrust=', r['calibration_gap_acctrust_mean'])
r2 = run_condition('reg','REG', lam=0.5, q=0.3)
print('REG lam=0.5,q=0.3: gap=', r2['calibration_gap_mean'], 'gap_acctrust=', r2['calibration_gap_acctrust_mean'])
