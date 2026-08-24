import json
r = json.load(open("results.json"))
print("main_gaussian rows for rho in 0,0.2,0.5,0.8,1.0:")
for row in r["main_gaussian"]:
    if row["rho"] in (0.0,0.2,0.5,0.8,1.0):
        print(row["rho"], "H=",round(row["H_mean"],3), "S=",round(row["S_mean"],3), "Acc=",round(row["Acc_mean"],3))
print()
print("main_student_t:")
for row in r["main_student_t"]:
    if row["rho"] in (0.0,0.2,0.5,0.8,1.0):
        print(row["rho"], "H=",round(row["H_mean"],3))
print()
print("ablation_K:")
for K,row in r["ablation_K"].items():
    print(K, "H=",round(row["H_mean"],3), "S=",round(row["S_mean"],3))
print()
print("ablation_p:")
for p,row in r["ablation_p"].items():
    print(p, "S=",round(row["S_mean"],3), "Acc=",round(row["Acc_mean"],3))
print()
print("mitigation_eta gaussian:")
for row in r["mitigation_eta"]:
    print(row["eta"], "H=",round(row["H_mean"],3), "Acc=",round(row["Acc_mean"],3))
print()
print("mitigation_eta student_t:")
for row in r["mitigation_eta_student_t"]:
    print(row["eta"], "H=",round(row["H_mean"],3), "Acc=",round(row["Acc_mean"],3))
print()
print("anchor K4 rho=0 and rho=1:")
for row in r["anchor_K4"]:
    if row["rho"] in (0.0, 1.0):
        print(row["rho"], "S=", round(row["S_mean"],3))
