import json
d = json.load(open('multiseed_results.json'))['results']
widths = [2, 4, 8, 16, 32, 64, 128, 256, 512]
seeds = list(range(10))

print('=== alpha=0 direction table (rows=seed, cols=width) ===')
print('seed', *[f'{w:>5}' for w in widths])
for s in seeds:
    row = sorted([r for r in d if r['seed'] == s and r['smoothing'] == 0.0], key=lambda r: r['hidden'])
    print(f'{s:>4}', *[('  U  ' if r['direction'] == 'underconfident' else '  O  ') for r in row])

print()
print('=== crossover width (first width where overconfident) per seed ===')
crossovers = []
for s in seeds:
    row = sorted([r for r in d if r['seed'] == s and r['smoothing'] == 0.0], key=lambda r: r['hidden'])
    first_over = next((r['hidden'] for r in row if r['direction'] == 'overconfident'), None)
    crossovers.append(first_over)
    print(f'seed {s}: first overconfident at hidden={first_over}')
print('crossover values:', crossovers)

print()
print('=== alpha=0.1: fraction underconfident per width across seeds ===')
for w in widths:
    rows = [r for r in d if r['smoothing'] == 0.1 and r['hidden'] == w]
    n_under = sum(1 for r in rows if r['direction'] == 'underconfident')
    eces = [r['test_ece_pre'] for r in rows]
    print(f'width={w:4d}: {n_under}/10 underconfident, mean ECE_pre={sum(eces)/len(eces):.4f}')

print()
print('=== alpha=0 mean ECE_pre/post by width ===')
for w in widths:
    rows = [r for r in d if r['smoothing'] == 0.0 and r['hidden'] == w]
    pre = [r['test_ece_pre'] for r in rows]
    post = [r['test_ece_post'] for r in rows]
    print(f'width={w:4d}: mean ECE_pre={sum(pre)/len(pre):.4f}, mean ECE_post={sum(post)/len(post):.4f}')
