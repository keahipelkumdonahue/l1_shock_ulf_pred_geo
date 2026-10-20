import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import confusion_matrix

try:
    from aeon.transformations.collection.convolution_based import MiniRocketMultivariate
except ImportError:
    from sktime.transformations.panel.rocket import MiniRocketMultivariate

from amp_config import CASE, AMP_PERCENTILE, TS_WINDOW_MARGIN, build_amp_label, ts_scores_path

###################################################################################################
############################################ LOAD + MERGE #########################################
###################################################################################################

ts_window_margin = TS_WINDOW_MARGIN

ts_data_b = joblib.load(f'ts_window_scan/{ts_window_margin}/refined_cases_ts_bfield.pkl')
ts_data_v = joblib.load(f'ts_window_scan/{ts_window_margin}/refined_cases_ts_vfield.pkl')
ts_data_fp = joblib.load(f'ts_window_scan/{ts_window_margin}/refined_cases_ts_fp.pkl')
all_events = joblib.load('ULF_shocks/all_ulf_shock_events.pkl')

print(all_events.keys())

### classify events from amplitude (CASE / AMP_PERCENTILE set in amp_config.py) ###

amp_label, cutoff = build_amp_label(all_events, CASE, AMP_PERCENTILE)
all_events['amp_label'] = amp_label
print(f"CASE='{CASE}', AMP_PERCENTILE={AMP_PERCENTILE} -> cutoff={cutoff:.4f}")

MERGE_KEY = 'l1_departure_time'

# Merge on the B-field time index first (as before)
ts_df_b = pd.DataFrame({'time': pd.to_datetime(ts_data_b['time']), 'row_b': np.arange(len(ts_data_b['time']))})
all_events[MERGE_KEY] = pd.to_datetime(all_events[MERGE_KEY])
merged = all_events.merge(ts_df_b, left_on=MERGE_KEY, right_on='time', how='inner')

# Merge on the V-field time index too, and confirm rows line up 1:1 with the B merge
ts_df_v = pd.DataFrame({'time': pd.to_datetime(ts_data_v['time']), 'row_v': np.arange(len(ts_data_v['time']))})
merged = merged.merge(ts_df_v, on='time', how='inner')

# Merge on the fp time index
ts_df_fp = pd.DataFrame({'time': pd.to_datetime(ts_data_fp['time']), 'row_fp': np.arange(len(ts_data_fp['time']))})
merged = merged.merge(ts_df_fp, on='time', how='inner')

print(f'Matched {len(merged)} / {len(all_events)} events to B, V, and fp time-series windows '
      f'({len(all_events) - len(merged)} events dropped: missing from one or more series).')

merged = merged.loc[merged['amp_label'] >= 0]  # removes events outside the selected CASE

X_ts_b = ts_data_b['znorm'][merged['row_b'].to_numpy()]  # (n, 3, TRIM_LEN)
X_ts_v = ts_data_v['znorm'][merged['row_v'].to_numpy()]  # (n, 3, TRIM_LEN)
X_ts_fp = ts_data_fp['znorm'][merged['row_fp'].to_numpy()]  # (n, 1, TRIM_LEN)

y = (merged['amp_label'] > 0).to_numpy().astype(int)

scale_pos_weight = (y == 0).sum() / (y == 1).sum()
print(f'n={len(y)}, positives={y.sum()}, negatives={len(y)-y.sum()}')

###################################################################################################
###################################### TSS SCORER (same def as 6a/6b) #############################
###################################################################################################

def tss_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sens + spec - 1.0

def best_threshold_tss(y_true, y_score):
    thresholds = np.linspace(y_score.min(), y_score.max(), 91)
    best_t, best_tss = thresholds[0], -1
    for t in thresholds:
        score = tss_score(y_true, (y_score >= t).astype(int))
        if score > best_tss:
            best_tss, best_t = score, t
    return best_t, best_tss

###################################################################################################
################################ MINIROCKET -> RIDGE, OUT-OF-FOLD (reusable) #######################
###################################################################################################

def run_minirocket_oof(X_ts, y, label, n_kernels=10000, n_splits=5, n_repeats=10):
    oof_scores = np.zeros((n_repeats, len(y)))

    for seed in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        for train_idx, test_idx in cv.split(X_ts, y):
            mr = MiniRocketMultivariate(num_kernels=n_kernels, random_state=seed)
            feat_train = mr.fit_transform(X_ts[train_idx])
            feat_test = mr.transform(X_ts[test_idx])

            scaler = StandardScaler()
            feat_train = scaler.fit_transform(feat_train)
            feat_test = scaler.transform(feat_test)

            clf = RidgeClassifierCV(alphas=np.logspace(-3, 4, 15), class_weight='balanced')
            clf.fit(feat_train, y[train_idx])

            oof_scores[seed, test_idx] = clf.decision_function(feat_test)

        fold_thresh, fold_tss = best_threshold_tss(y, oof_scores[seed])
        print(f'  [{label}] seed {seed}: standalone ts_score TSS (in-seed OOF) = {fold_tss:.4f}')

    ts_score = oof_scores.mean(axis=0)
    thresh, tss = best_threshold_tss(y, ts_score)
    print(f'[{label}] Averaged ts_score across {n_repeats} seeds: standalone TSS = {tss:.4f} '
          f'(threshold={thresh:.3f})\n')
    return ts_score

###################################################################################################
############################################# RUN: B, V, AND COMBINED ###############################
###################################################################################################

N_KERNELS = 10000
N_SPLITS = 5
N_REPEATS = 10

print("Running MiniRocket on B-field (3 channels)...")
ts_score_b = run_minirocket_oof(X_ts_b, y, label='B-field',
                                 n_kernels=N_KERNELS, n_splits=N_SPLITS, n_repeats=N_REPEATS)

print("Running MiniRocket on V-field (3 channels)...")
ts_score_v = run_minirocket_oof(X_ts_v, y, label='V-field',
                                 n_kernels=N_KERNELS, n_splits=N_SPLITS, n_repeats=N_REPEATS)

print("Running MiniRocket on flow pressure (1 channel)...")
ts_score_fp = run_minirocket_oof(X_ts_fp, y, label='FP',
                                  n_kernels=N_KERNELS, n_splits=N_SPLITS, n_repeats=N_REPEATS)

# Stack B and V into one 6-channel input and run MiniRocket once on the
# combined series, rather than treating B/V as two independent scores
X_ts_combined = np.concatenate([X_ts_b, X_ts_v], axis=1)  # (n, 6, TRIM_LEN)
print("Running MiniRocket on combined B+V (6 channels)...")
ts_score_combined = run_minirocket_oof(X_ts_combined, y, label='B+V combined',
                                        n_kernels=N_KERNELS, n_splits=N_SPLITS, n_repeats=N_REPEATS)

# Also add fp to the stack (7 channels total)
X_ts_all = np.concatenate([X_ts_b, X_ts_v, X_ts_fp], axis=1)  # (n, 7, TRIM_LEN)
print("Running MiniRocket on combined B+V+fp (7 channels)...")
ts_score_all = run_minirocket_oof(X_ts_all, y, label='B+V+fp combined',
                                   n_kernels=N_KERNELS, n_splits=N_SPLITS, n_repeats=N_REPEATS)

print(f"Compare all standalone/combined TSS values above to your current 3-feature "
      f"tabular TSS and to each other -- this tells you whether B, V, and fp "
      f"carry independent signal (all scores useful, add to 10c/10d) or largely "
      f"redundant signal (combined ~= max of the individual scores, and fewer may be enough). "
      f"[CASE='{CASE}', AMP_PERCENTILE={AMP_PERCENTILE}, window={ts_window_margin}min]")

###################################################################################################
################################################ SAVE ##############################################
###################################################################################################

out = merged[['time']].copy()
out['ts_score_b'] = ts_score_b
out['ts_score_v'] = ts_score_v
out['ts_score_fp'] = ts_score_fp
out['ts_score_combined'] = ts_score_combined
out['ts_score_all'] = ts_score_all

out_path = ts_scores_path(CASE, AMP_PERCENTILE, ts_window_margin)
joblib.dump(out, out_path)
print(f"\nSaved {len(out)} rows to {out_path} -- merge into all_events on "
      f"'{MERGE_KEY}' == 'time', then add 'ts_score_b'/'ts_score_v'/'ts_score_fp'/"
      f"'ts_score_combined'/'ts_score_all' (as appropriate) to rfecv_candidates in 10c "
      f"and core_features in 10d.")
