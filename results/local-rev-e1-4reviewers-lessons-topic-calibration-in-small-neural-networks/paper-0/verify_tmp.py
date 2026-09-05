import json, statistics as st
d = json.load(open('multiseed_results.json'))['results']

widths=[2,4,8,16,32,64,128,256,512]

print("Table 2 crossover check:")
for seed in range(10):
    rows = sorted([r for r in d if r['seed']==seed and r['smoothing']==0.0], key=lambda r: r['hidden'])
    first_over = None
    for r in rows:
        if r['direction']=='overconfident':
            first_over = r['hidden']
            break
    print(seed, first_over)

print("\nTable 3 check:")
for w in widths:
    rows_s = [r for r in d if r['smoothing']==0.1 and r['hidden']==w]
    rows_0 = [r for r in d if r['smoothing']==0.0 and r['hidden']==w]
    mean_pre = st.mean(r['test_ece_pre'] for r in rows_s)
    mean_post = st.mean(r['test_ece_post'] for r in rows_s)
    mean_pre_0 = st.mean(r['test_ece_pre'] for r in rows_0)
    ratio = mean_pre/mean_pre_0
    print(w, round(mean_pre,3), round(ratio,1), round(mean_post,3))

print("\nTemp scaling by width band (smoothing=0.0):")
for w in [2,4,8,16]:
    rows = [r for r in d if r['smoothing']==0.0 and r['hidden']==w]
    mean_pre = st.mean(r['test_ece_pre'] for r in rows)
    mean_post = st.mean(r['test_ece_post'] for r in rows)
    hurts = sum(1 for r in rows if r['test_ece_post'] > r['test_ece_pre'])
    print(w, round(mean_pre,3), round(mean_post,3), f"hurts {hurts}/10")
