from scipy import stats
errors=[83,74,14,10,19]
auroc=[0.8517416829745598,0.8125200900032145,0.6626275510204082,0.6015228426395939,0.6737709576584257]
print(stats.spearmanr(errors,auroc))
print(stats.pearsonr(errors,auroc))
