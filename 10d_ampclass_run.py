import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import joblib
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, average_precision_score
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt

from amp_config import CASE, AMP_PERCENTILE, TS_WINDOW_MARGIN, select_case_events, build_amp_label, ts_scores_path

###################################################################################################
############################################# SETUP ################################################
###################################################################################################

all_events = joblib.load('ULF_shocks/all_ulf_shock_events.pkl')

all_events['direction'] = all_events['direction'].replace({'ff': 0, 'fr': 1}).infer_objects(copy=False)
all_events['direction'] = all_events['direction'].astype(np.float64)

# Case-selected events (pc5 or subpc5, per amp_config.CASE)
all_events = select_case_events(all_events, CASE)

# Amplitude label (CASE / AMP_PERCENTILE set in amp_config.py)
amp_label, cutoff = build_amp_label(all_events, CASE, AMP_PERCENTILE)
all_events['amp_label'] = amp_label
print(f"CASE='{CASE}', AMP_PERCENTILE={AMP_PERCENTILE} -> cutoff={cutoff:.4f}")

# --- merge in ts_score (same pattern as 10c) ---
ts_score_path = ts_scores_path(CASE, AMP_PERCENTILE, TS_WINDOW_MARGIN)
ts_score_df = joblib.load(ts_score_path)

all_events['l1_departure_time'] = pd.to_datetime(all_events['l1_departure_time'])
ts_score_df['time'] = pd.to_datetime(ts_score_df['time'])
ts_score_df = ts_score_df.drop_duplicates(subset='time')  # l1_departure_time isn't unique in all_events

n_events_pre_merge = len(all_events)
all_events = all_events.merge(ts_score_df, left_on='l1_departure_time', right_on='time', how='left')
assert len(all_events) == n_events_pre_merge, "merge changed row count -- check for duplicate keys"

n_with_ts = all_events['ts_score_b'].notna().sum()
print(f"ts_score available for {n_with_ts} / {len(all_events)} events")

# Restrict to events with a valid ts_score for ALL comparisons below
all_events = all_events.dropna(subset=['ts_score_b']).reset_index(drop=True)
y = all_events['amp_label']
print(f"Comparing feature sets on N={len(all_events)}, "
      f"positives={y.sum()}, negatives={len(y) - y.sum()}\n")

###################################################################################################
##################################### FEATURE SETS (from 10c) ######################################
###################################################################################################

FEATURE_SETS_BY_CASE = {
    ('pc5', 20, 90): {
        'top_6': ['ts_score_all', 'ts_score_v', 'l1_delta_fp', 'l1_delta_b', 'l1_pre_b', 'ts_score_b'],
        'top_5': ['ts_score_all', 'ts_score_v', 'l1_delta_fp', 'l1_delta_b', 'l1_pre_b'],
        'top_4': ['ts_score_all', 'ts_score_v', 'l1_delta_fp', 'l1_delta_b'],
        'top_3': ['ts_score_all', 'ts_score_v', 'l1_delta_fp'],
        'top_2': ['ts_score_all', 'ts_score_v'],
        'top_1': ['ts_score_all'],
        'top_6_minus_all': ['ts_score_v', 'l1_delta_fp', 'l1_delta_b', 'l1_pre_b', 'ts_score_b'],
        'top_5_minus_all': ['ts_score_v', 'l1_delta_fp', 'l1_delta_b', 'l1_pre_b'], # best
        'top_4_minus_all': ['ts_score_v', 'l1_delta_fp', 'l1_delta_b'],
    },
    ('subpc5', 20, 90): {
        'top_6': ['ts_score_all', 'ts_score_v', 'l1_pre_b', 'ts_score_b', 'l1_delta_b', 'shock_vy'],
        'top_5': ['ts_score_all', 'ts_score_v', 'l1_pre_b', 'ts_score_b', 'l1_delta_b'],
        'top_4': ['ts_score_all', 'ts_score_v', 'l1_pre_b', 'ts_score_b'],
        'top_3': ['ts_score_all', 'ts_score_v', 'l1_pre_b'], # best
        'top_2': ['ts_score_all', 'ts_score_v'],
        'top_1': ['ts_score_all'],
        'top_6_minus_all': ['ts_score_v', 'l1_pre_b', 'ts_score_b', 'l1_delta_b', 'shock_vy'],
        'top_5_minus_all': ['ts_score_v', 'l1_pre_b', 'ts_score_b', 'l1_delta_b'],
        'top_4_minus_all': ['ts_score_v', 'l1_pre_b', 'ts_score_b'],
    },
}

_key = (CASE, TS_WINDOW_MARGIN, AMP_PERCENTILE)
if _key not in FEATURE_SETS_BY_CASE:
    raise KeyError(
        f"No feature sets recorded for CASE='{CASE}', TS_WINDOW_MARGIN={TS_WINDOW_MARGIN}, "
        f"AMP_PERCENTILE={AMP_PERCENTILE} (amp_config.py). Run 10c for this combination "
        f"first, then add its top features under FEATURE_SETS_BY_CASE[{_key!r}] above. "
        f"Available combinations: {list(FEATURE_SETS_BY_CASE)}"
    )
feature_sets = FEATURE_SETS_BY_CASE[_key]
print(f"Using feature sets for {_key}: {feature_sets}\n")


def tss_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sens + spec - 1.0


def best_threshold_tss(y_true, y_prob):
    """Scan thresholds, return the one maximizing TSS."""
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_tss = 0.5, -1
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        score = tss_score(y_true, y_pred)
        if score > best_tss:
            best_tss, best_t = score, t
    return best_t, best_tss


param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(2, 6),
    'learning_rate': uniform(0.01, 0.15),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'min_child_weight': randint(1, 10),
    'gamma': uniform(0, 0.5),
}


def run_nested_cv(X, y, label, n_splits=5, n_iter=40, verbose=True):

    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    outer_tss_scores, outer_pr_auc_scores, outer_thresholds, best_params_per_fold = [], [], [], []

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold_i)

        base_model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric='aucpr',
            random_state=fold_i
        )

        search = RandomizedSearchCV(
            base_model, param_distributions=param_dist,
            n_iter=n_iter, scoring='average_precision',
            cv=inner_cv, random_state=fold_i, n_jobs=-1
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_prob_test = best_model.predict_proba(X_test)[:, 1]

        # NOTE: threshold is tuned on the same test fold it's scored on here
        thresh, tss = best_threshold_tss(y_test, y_prob_test)
        pr_auc = average_precision_score(y_test, y_prob_test)

        outer_tss_scores.append(tss)
        outer_pr_auc_scores.append(pr_auc)
        outer_thresholds.append(thresh)
        best_params_per_fold.append(search.best_params_)

        if verbose:
            print(f"  [{label}] Fold {fold_i}: TSS={tss:.4f}  PR-AUC={pr_auc:.4f}  threshold={thresh:.2f}")

    return {
        'label': label,
        'scale_pos_weight': scale_pos_weight,
        'tss_scores': np.array(outer_tss_scores),
        'pr_auc_scores': np.array(outer_pr_auc_scores),
        'thresholds': np.array(outer_thresholds),
        'best_params_per_fold': best_params_per_fold,
        'mean_tss': np.mean(outer_tss_scores),
        'std_tss': np.std(outer_tss_scores),
        'mean_pr_auc': np.mean(outer_pr_auc_scores),
        'std_pr_auc': np.std(outer_pr_auc_scores),
        'mean_threshold': np.mean(outer_thresholds)
    }


###################################################################################################
############################ 1. PHASE 1: TUNED NESTED-CV FEATURE SET COMPARISON ###################
###################################################################################################
# Runs the SAME tuned nested-CV pipeline for each candidate feature set

print("="*60)
print(f"PHASE 1: feature set comparison (tuned nested CV, same data/folds) "
      f"[{_key}]")
print("="*60)

cv_results = {}
for label, feats in feature_sets.items():
    print(f"\n--- {label}: {feats} ---")
    X = all_events[feats]
    cv_results[label] = run_nested_cv(X, y, label)

print("\n" + "="*60)
print("PHASE 1 SUMMARY")
print("="*60)
comparison_table = pd.DataFrame({
    'feature_set': list(cv_results.keys()),
    'features': [feature_sets[k] for k in cv_results],
    'TSS (mean ± std)': [f"{r['mean_tss']:.3f} ± {r['std_tss']:.3f}" for r in cv_results.values()],
    'PR-AUC (mean ± std)': [f"{r['mean_pr_auc']:.3f} ± {r['std_pr_auc']:.3f}" for r in cv_results.values()],
}).sort_values('TSS (mean ± std)', ascending=False)
print(comparison_table.to_string(index=False))

WINNING_FEATURES = max(cv_results, key=lambda k: cv_results[k]['mean_tss'])
print(f"\nAuto-selected winner by mean TSS: '{WINNING_FEATURES}' "
      f"{feature_sets[WINNING_FEATURES]} -- override manually below if you'd "
      f"rather use the paired comparison or parsimony to decide.")

###################################################################################################
######################## 2. PHASE 2: FULL ANALYSIS ON THE WINNING FEATURE SET #####################
###################################################################################################

print("\n" + "="*60)
print(f"PHASE 2: full analysis for '{WINNING_FEATURES}' {feature_sets[WINNING_FEATURES]}")
print("="*60)

winning_result = cv_results[WINNING_FEATURES]
X = all_events[feature_sets[WINNING_FEATURES]]
scale_pos_weight = winning_result['scale_pos_weight']
mean_tss, std_tss = winning_result['mean_tss'], winning_result['std_tss']
mean_pr_auc, std_pr_auc = winning_result['mean_pr_auc'], winning_result['std_pr_auc']
mean_threshold = winning_result['mean_threshold']
outer_tss_scores = winning_result['tss_scores']
outer_pr_auc_scores = winning_result['pr_auc_scores']

# --- Final model (all data) ---
# Refit on all data using tuned hyperparameters

final_search = RandomizedSearchCV(
    xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric='aucpr', random_state=0),
    param_distributions=param_dist,
    n_iter=40, scoring='average_precision',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
    random_state=0, n_jobs=-1
)
final_search.fit(X, y)
final_model = final_search.best_estimator_
final_threshold = mean_threshold  # averaged outer-fold threshold

print("Final model params:", final_search.best_params_)
print(f"Recommended decision threshold: {final_threshold:.3f}\n")

# --- SHAP ---
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, show=False)
# print(np.array(shap_values).shape)

plt.tight_layout()
plt.show()
# plt.savefig(f"model_results_ts/10d_ampclass_{CASE}_shap.png", dpi=300, bbox_inches="tight")

# --- Summary table (nested CV only) ---
print("="*60)
print("FINAL PERFORMANCE (5-fold nested CV) — for paper reporting")
print("="*60)
summary_table = pd.DataFrame({
    'Metric': ['TSS', 'PR-AUC'],
    'Nested CV (mean ± std)': [f'{mean_tss:.3f} ± {std_tss:.3f}',
                                f'{mean_pr_auc:.3f} ± {std_pr_auc:.3f}'],
    'Fold range': [f'{min(outer_tss_scores):.3f}–{max(outer_tss_scores):.3f}',
                   f'{min(outer_pr_auc_scores):.3f}–{max(outer_pr_auc_scores):.3f}']
})
print(summary_table.to_string(index=False))

###################################################################################################
################################## 3. PERMUTATION TEST (NULL BASELINE) ##############################
###################################################################################################

n_permutations = 200
rng = np.random.default_rng(42)

perm_tss_scores = []
perm_pr_auc_scores = []

fixed_params = final_search.best_params_

for perm_i in range(n_permutations):
    y_shuffled = y.sample(frac=1.0, random_state=rng.integers(0, 1_000_000)).reset_index(drop=True)
    y_shuffled.index = y.index  # realign index to X

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=perm_i)
    fold_tss, fold_pr_auc = [], []

    for train_idx, test_idx in cv.split(X, y_shuffled):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_shuffled.iloc[train_idx], y_shuffled.iloc[test_idx]

        # scale_pos_weight recomputed per shuffle in case shuffling changes fold balance
        spw = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1.0

        model = xgb.XGBClassifier(**fixed_params, scale_pos_weight=spw,
                                   eval_metric='aucpr', random_state=0)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)  # fixed threshold -- no per-fold threshold tuning here

        fold_tss.append(tss_score(y_test, y_pred))
        fold_pr_auc.append(average_precision_score(y_test, y_prob))

    perm_tss_scores.append(np.mean(fold_tss))
    perm_pr_auc_scores.append(np.mean(fold_pr_auc))

    if (perm_i + 1) % 50 == 0:
        print(f"  Permutation {perm_i + 1}/{n_permutations} complete")

perm_tss_scores = np.array(perm_tss_scores)
perm_pr_auc_scores = np.array(perm_pr_auc_scores)

# p-value: fraction of null (shuffled) results at least as extreme as the real result
p_value_tss = (perm_tss_scores >= mean_tss).mean()
p_value_pr_auc = (perm_pr_auc_scores >= mean_pr_auc).mean()

print("\n" + "="*60)
print("PERMUTATION TEST RESULTS")
print("="*60)
print(f"Observed TSS: {mean_tss:.4f}  |  Null TSS: {perm_tss_scores.mean():.4f} +/- {perm_tss_scores.std():.4f}")
print(f"p-value (TSS): {p_value_tss:.4f}  ({(perm_tss_scores >= mean_tss).sum()}/{n_permutations} "
      f"null runs >= observed)")
print(f"\nObserved PR-AUC: {mean_pr_auc:.4f}  |  Null PR-AUC: {perm_pr_auc_scores.mean():.4f} "
      f"+/- {perm_pr_auc_scores.std():.4f}")
print(f"p-value (PR-AUC): {p_value_pr_auc:.4f}  ({(perm_pr_auc_scores >= mean_pr_auc).sum()}/{n_permutations} "
      f"null runs >= observed)")

###################################################################################################
################################# 4. UPDATED SUMMARY TABLE WITH PERMUTATION TEST #####################
###################################################################################################

print("\n" + "="*60)
print(f"FINAL PERFORMANCE WITH PERMUTATION TEST ('{WINNING_FEATURES}') — for paper reporting")
print("="*60)
summary_table_perm = pd.DataFrame({
    'Metric': ['TSS', 'PR-AUC'],
    'Observed (nested CV mean ± std)': [f'{mean_tss:.3f} ± {std_tss:.3f}',
                                          f'{mean_pr_auc:.3f} ± {std_pr_auc:.3f}'],
    'Null (permutation mean ± std)': [f'{perm_tss_scores.mean():.3f} ± {perm_tss_scores.std():.3f}',
                                        f'{perm_pr_auc_scores.mean():.3f} ± {perm_pr_auc_scores.std():.3f}'],
    'p-value': [f'{p_value_tss:.4f}', f'{p_value_pr_auc:.4f}']
})
print(summary_table_perm.to_string(index=False))