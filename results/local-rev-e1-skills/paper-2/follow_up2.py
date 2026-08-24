"""
Round-2 follow-up: extend Q2 (score-distribution shift) from 1 seed to 5 seeds,
and Q3 (matched-set-size shift-robustness) from 3 seeds to 5 seeds, to directly
answer round-2 review questions:
 - Q2 (review Q1): is the LAC=0.680 vs APS=0.847 normalized-shift ordering
   seed-dependent, or does it hold across seeds?
 - Q3 (review Q3): is the residual matched-size gap (APS drops slightly more
   than LAC in all 3 original seeds) a real small effect or noise?
Reuses identical generator/model/calibration code from follow_up.py.
"""
import numpy as np
import importlib.util
import sys

_spec = importlib.util.spec_from_file_location("follow_up_lib", "follow_up.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["follow_up_lib"] = _mod
_orig_argv = sys.argv
try:
    # follow_up.py runs its Q1-Q3 experiments at import time (module-level code);
    # temporarily silence stdout so we only get the function/constant definitions
    # without re-printing or re-running its own experiment log.
    import io
    import contextlib
    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf):
        _spec.loader.exec_module(_mod)
finally:
    sys.argv = _orig_argv

VOCAB = _mod.VOCAB
ORDER = _mod.ORDER
SEQ_LEN = _mod.SEQ_LEN
N_TEST_SEQ = _mod.N_TEST_SEQ
ALPHA = _mod.ALPHA
train_model = _mod.train_model
get_probs = _mod.get_probs
lac_calibrate = _mod.lac_calibrate
aps_calibrate = _mod.aps_calibrate
aps_scores = _mod.aps_scores
aps_sets_deterministic = _mod.aps_sets_deterministic
lac_sets = _mod.lac_sets
evaluate = _mod.evaluate
shift_kernel = _mod.shift_kernel
sample_sequences = _mod.sample_sequences
find_alpha_for_target_size = _mod.find_alpha_for_target_size

print("=" * 70)
print("Q2-extended: score-distribution shift, LAC vs APS, 5 seeds")
q2_lac_moves, q2_aps_moves = [], []
for seed in range(5):
    rng, kernel, calib_seqs, test_seqs, model = train_model(seed)
    calib_probs, calib_y = get_probs(model, calib_seqs, VOCAB)
    lac_score_calib = 1 - calib_probs[np.arange(len(calib_y)), calib_y]
    cum_before_c, p_true_c = aps_scores(calib_probs, calib_y)
    aps_score_calib = cum_before_c + 0.5 * p_true_c

    sk0 = shift_kernel(kernel, 0.0, VOCAB)
    sk1 = shift_kernel(kernel, 1.0, VOCAB)
    seqs0 = sample_sequences(sk0, 500, SEQ_LEN, VOCAB, ORDER, rng)
    seqs1 = sample_sequences(sk1, 500, SEQ_LEN, VOCAB, ORDER, rng)
    probs1, y1 = get_probs(model, seqs1, VOCAB)
    lac_score1 = 1 - probs1[np.arange(len(y1)), y1]
    cum_before1, p_true1 = aps_scores(probs1, y1)
    aps_score1 = cum_before1 + 0.5 * p_true1

    lac_move = (lac_score1.mean() - lac_score_calib.mean()) / lac_score_calib.std()
    aps_move = (aps_score1.mean() - aps_score_calib.mean()) / aps_score_calib.std()
    q2_lac_moves.append(lac_move)
    q2_aps_moves.append(aps_move)
    print(f"  seed {seed}: LAC_move={lac_move:.3f}  APS_move={aps_move:.3f}  "
          f"APS>LAC: {aps_move > lac_move}")

q2_lac_moves = np.array(q2_lac_moves)
q2_aps_moves = np.array(q2_aps_moves)
print(f"  Mean over 5 seeds: LAC={q2_lac_moves.mean():.3f}+/-{q2_lac_moves.std():.3f}  "
      f"APS={q2_aps_moves.mean():.3f}+/-{q2_aps_moves.std():.3f}  "
      f"APS>LAC in {int((q2_aps_moves > q2_lac_moves).sum())}/5 seeds")

print("=" * 70)
print("Q3-extended: matched-set-size shift-robustness, 5 seeds (add seeds 3,4)")
q3_rows = []
for seed in [0, 1, 2, 3, 4]:
    rng_s, kernel_s, calib_s, test_s, model_s = train_model(seed)
    cprobs, cy = get_probs(model_s, calib_s, VOCAB)
    qhat_aps_s = aps_calibrate(cprobs, cy, ALPHA, np.random.default_rng(seed + 100))
    probs0, y0 = get_probs(model_s, test_s, VOCAB)
    order_a0, k_a0 = aps_sets_deterministic(probs0, qhat_aps_s)
    target_size = k_a0.mean()
    alpha_matched, qhat_lac_matched = find_alpha_for_target_size(cprobs, cy, target_size, probs0)

    out = {}
    for shift_frac in [0.0, 1.0]:
        sk = shift_kernel(kernel_s, shift_frac, VOCAB)
        seqs_shift = sample_sequences(sk, N_TEST_SEQ, SEQ_LEN, VOCAB, ORDER, rng_s)
        probs, y = get_probs(model_s, seqs_shift, VOCAB)
        o_l, k_l = lac_sets(probs, qhat_lac_matched)
        cov_l, size_l = evaluate(o_l, k_l, y)
        o_a, k_a = aps_sets_deterministic(probs, qhat_aps_s)
        cov_a, size_a = evaluate(o_a, k_a, y)
        out[shift_frac] = dict(lac_cov=cov_l, lac_size=size_l, aps_cov=cov_a, aps_size=size_a)
    drop_l = out[0.0]['lac_cov'] - out[1.0]['lac_cov']
    drop_a = out[0.0]['aps_cov'] - out[1.0]['aps_cov']
    q3_rows.append(dict(seed=seed, lac_drop=drop_l, aps_drop=drop_a))
    print(f"  seed {seed}: coverage drop: LAC={drop_l:.4f} APS={drop_a:.4f}  "
          f"APS>LAC: {drop_a > drop_l}")

lac_drops = np.array([r['lac_drop'] for r in q3_rows])
aps_drops = np.array([r['aps_drop'] for r in q3_rows])
diffs = aps_drops - lac_drops
print(f"  Matched-size mean drop over 5 seeds: LAC={lac_drops.mean():.4f}+/-{lac_drops.std():.4f}  "
      f"APS={aps_drops.mean():.4f}+/-{aps_drops.std():.4f}")
print(f"  Per-seed (APS-LAC) diff: {[round(d,4) for d in diffs]}  "
      f"mean={diffs.mean():.4f} std={diffs.std():.4f}  positive in {int((diffs>0).sum())}/5 seeds")

print("=" * 70)
print("DONE")
