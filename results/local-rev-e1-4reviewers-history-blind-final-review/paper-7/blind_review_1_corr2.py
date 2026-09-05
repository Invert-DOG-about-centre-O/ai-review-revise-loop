errors=[83,74,14,10,19]
auroc=[0.8517416829745598,0.8125200900032145,0.6626275510204082,0.6015228426395939,0.6737709576584257]

def rank(lst):
    sorted_idx = sorted(range(len(lst)), key=lambda i: lst[i])
    ranks = [0]*len(lst)
    for r,i in enumerate(sorted_idx):
        ranks[i] = r+1
    return ranks

re = rank(errors)
ra = rank(auroc)
n = len(errors)
d2 = sum((re[i]-ra[i])**2 for i in range(n))
spearman = 1 - 6*d2/(n*(n**2-1))
print("ranks errors", re)
print("ranks auroc", ra)
print("spearman", spearman)

mean_e = sum(errors)/n
mean_a = sum(auroc)/n
cov = sum((errors[i]-mean_e)*(auroc[i]-mean_a) for i in range(n))
var_e = sum((errors[i]-mean_e)**2 for i in range(n))
var_a = sum((auroc[i]-mean_a)**2 for i in range(n))
pearson = cov/((var_e*var_a)**0.5)
print("pearson", pearson)
