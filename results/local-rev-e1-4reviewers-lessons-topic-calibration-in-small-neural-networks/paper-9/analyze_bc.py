import json
import numpy as np
from scipy.stats import ttest_ind

results = json.load(open('results_bc.json'))
widths = sorted(set(r['width'] for r in results))
print('width | acc mean+-sd | ece mean+-sd | nll mean+-sd')
byw = {}
for w in widths:
    rs = [r for r in results if r['width'] == w]
    acc = np.array([r['test_acc'] for r in rs])
    ece = np.array([r['test_ece'] for r in rs])
    nll = np.array([r['test_nll'] for r in rs])
    byw[w] = ece
    print(f'{w:4d} | {acc.mean():.3f}+-{acc.std():.3f} | {ece.mean():.3f}+-{ece.std():.3f} | {nll.mean():.3f}+-{nll.std():.3f}')

low = np.concatenate([byw[4], byw[8]])
plateau = np.concatenate([byw[w] for w in [16, 32, 64, 128]])
t, p = ttest_ind(low, plateau, equal_var=False)
print('4+8 vs plateau(16-128):', t, p)

rs2 = [r for r in results if r['width'] == 2]
accs2 = np.array([r['test_acc'] for r in rs2])
print('width2 accs:', accs2)
