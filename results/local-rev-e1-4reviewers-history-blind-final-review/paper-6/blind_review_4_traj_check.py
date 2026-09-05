from sim import run_condition
for s in [0.0, 0.4, 1.0]:
    r = run_condition(s, False, seed=0)
    traj = r['trust_traj']
    print(s, [round(traj[i], 2) for i in [0, 9, 19, 29, 39, 49, 59]])
