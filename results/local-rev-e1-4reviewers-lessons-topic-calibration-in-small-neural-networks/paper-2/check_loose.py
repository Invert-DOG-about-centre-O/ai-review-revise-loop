import json
d = json.load(open("results.json"))
results = d['raw_results']
widths = d['widths']
seeds = d['seeds']
for dataset_name in ['digits','synthetic']:
    loose_count = 0
    for seed in seeds:
        rows = sorted([r for r in results if r['dataset']==dataset_name and r['seed']==seed], key=lambda r:r['width'])
        biases = [r['bias'] for r in rows]
        touched = any(b>=0 for b in biases)
        if touched:
            loose_count += 1
    print(dataset_name, 'loose defined count:', loose_count, 'of', len(seeds))
