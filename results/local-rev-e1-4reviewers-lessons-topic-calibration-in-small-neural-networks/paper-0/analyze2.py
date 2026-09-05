import json
import statistics as st
d = json.load(open('multiseed_results.json'))['results']
widths = [2, 4, 8, 16, 32, 64, 128, 256, 512]
seeds = list(range(10))

# monotonicity: does direction go U...U O...O with no reversal after first O?
non_monotonic = 0
for s in seeds:
    row = sorted([r for r in d if r['seed'] == s and r['smoothing'] == 0.0], key=lambda r: r['hidden'])
    dirs = [r['direction'] for r in row]
    first_o = next((i for i, x in enumerate(dirs) if x == 'overconfident'), None)
    if first_o is not None and any(x == 'underconfident' for x in dirs[first_o+1:]):
        non_monotonic += 1
print(f'{non_monotonic}/10 seeds show a reversal (U after first O)')

crossovers = [16, 8, 8, 8, 4, 4, 8, 8, 8, 8]
print('crossover median:', st.median(crossovers), 'mode: 8 (7/10 seeds)')
print('crossover range:', min(crossovers), '-', max(crossovers))

# ECE std at width=4 alpha=0 pre/post
for w in [2, 4, 8, 16]:
    rows = [r for r in d if r['smoothing'] == 0.0 and r['hidden'] == w]
    pre = [r['test_ece_pre'] for r in rows]
    post = [r['test_ece_post'] for r in rows]
    n_worse = sum(1 for r in rows if r['test_ece_post'] > r['test_ece_pre'])
    print(f'width={w}: ECE_pre {st.mean(pre):.4f}+-{st.pstdev(pre):.4f}, ECE_post {st.mean(post):.4f}+-{st.pstdev(post):.4f}, T-scaling hurt in {n_worse}/10 seeds')

# label smoothing ECE improvement magnitude vs unsmoothed, mean across seeds, per width
print()
print('alpha=0 vs alpha=0.1 mean ECE_pre ratio per width:')
for w in widths:
    r0 = [r['test_ece_pre'] for r in d if r['smoothing'] == 0.0 and r['hidden'] == w]
    r1 = [r['test_ece_pre'] for r in d if r['smoothing'] == 0.1 and r['hidden'] == w]
    print(f'width={w:4d}: alpha0={st.mean(r0):.4f} alpha0.1={st.mean(r1):.4f} ratio={st.mean(r1)/st.mean(r0):.1f}x')

# temp scaling recovery under label smoothing, mean post ECE
print()
print('alpha=0.1 mean ECE_post by width (recovery):')
for w in widths:
    r1 = [r['test_ece_post'] for r in d if r['smoothing'] == 0.1 and r['hidden'] == w]
    print(f'width={w:4d}: mean ECE_post={st.mean(r1):.4f}')
