import json, statistics
d = json.load(open('joint_results.json'))['joint']
for k in ['calibrated_pool30_wide','sycophantic_pool30_wide','calibrated_pool30_bimodal','sycophantic_pool30_bimodal','calibrated_pool5_wide','sycophantic_pool5_wide','calibrated_pool5_bimodal','sycophantic_pool5_bimodal']:
    v = d[k]['var']
    bc = d[k]['bc']
    print(k, round(statistics.mean(v),3), '+-', round(statistics.stdev(v),3), '| bc', round(statistics.mean(bc),3), '+-', round(statistics.stdev(bc),3))
