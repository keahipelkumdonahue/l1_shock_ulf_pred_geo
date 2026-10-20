import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import joblib
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, average_precision_score
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt

###################################################################################################
############################################# SETUP ################################################
###################################################################################################

all_events = joblib.load('ULF_shocks/all_ulf_shock_events.pkl')

all_events['direction'] = all_events['direction'].replace({'ff': 0, 'fr': 1}).infer_objects(copy=False)
all_events['direction'] = all_events['direction'].astype(np.float64)
all_events['oscillation'] = all_events['n_detections'] > 0

# --- merge in ts_score (same pattern as 6a) ---
ts_window_margin = 20
ts_score_df = joblib.load(f'ts_window_scan/{ts_window_margin}/ts_scores_for_merge.pkl')

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
y = all_events['oscillation']
print(f"Comparing feature sets on N={len(all_events)}, "
      f"positives={y.sum()}, negatives={len(y) - y.sum()}\n")

feature_sets = {
    'top_6': ['direction', 'ts_score_v', 'ts_score_fp', 'ts_score_all', 'ts_score_b', 'imf_clock'],
    'top_5': ['direction', 'ts_score_v', 'ts_score_fp', 'ts_score_all', 'ts_score_b'],
    'top_4': ['direction', 'ts_score_v', 'ts_score_fp', 'ts_score_all'],
    'top_3': ['direction', 'ts_score_v', 'ts_score_fp'],
    'top_2': ['direction', 'ts_score_v'],
    'top_1': ['direction'],
}

VARIANTS_TO_RUN = list(feature_sets.keys())


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


def run_permutation_test(X, y, fixed_params, mean_tss, mean_pr_auc, label,
                          n_permutations=200, seed=42, verbose=True):

    rng = np.random.default_rng(seed)
    perm_tss_scores, perm_pr_auc_scores = [], []

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

        if verbose and (perm_i + 1) % 50 == 0:
            print(f"  [{label}] permutation {perm_i + 1}/{n_permutations} complete")

    perm_tss_scores = np.array(perm_tss_scores)
    perm_pr_auc_scores = np.array(perm_pr_auc_scores)

    return {
        'perm_tss_scores': perm_tss_scores,
        'perm_pr_auc_scores': perm_pr_auc_scores,
        'p_value_tss': (perm_tss_scores >= mean_tss).mean(),
        'p_value_pr_auc': (perm_pr_auc_scores >= mean_pr_auc).mean(),
    }


###################################################################################################
##################################### 1. RAW MLT-SECTOR CHECK #####################################
###################################################################################################

def mlt_sector(mlt):
    if 3 <= mlt < 9:
        return 'dawn/morning (3-9)'
    elif 9 <= mlt < 15:
        return 'noon (9-15)'
    elif 15 <= mlt < 21:
        return 'dusk/evening (15-21)'
    else:
        return 'midnight (21-3)'


all_events['mlt_sector'] = all_events['arrival_goes_mlt'].apply(mlt_sector)

sector_summary = all_events.groupby('mlt_sector')['oscillation'].agg(
    n_events='count',
    n_oscillation='sum',
    oscillation_rate='mean'
).sort_values('oscillation_rate', ascending=False)

print("Raw oscillation rate by MLT sector (no model, just counts):")
print(sector_summary, "\n")

###################################################################################################
################### 2. FULL TRAINING SWEEP: EVERY VARIANT, SAME COMPLETE PIPELINE #################
###################################################################################################

print("="*60)
print(f"FULL TRAINING SWEEP -- {len(VARIANTS_TO_RUN)} variants")
print("="*60)
print("(No permutation test per variant here -- it's expensive, and every variant")
print(" tested so far has come back p<0.005, so it isn't doing useful screening")
print(" work here. It's run once below, on the winning variant only.)\n")

full_results = {}

for label in VARIANTS_TO_RUN:
    feats = feature_sets[label]
    print(f"\n--- {label}: {feats} ---")
    X = all_events[feats]

    # (a) tuned nested CV
    cv_result = run_nested_cv(X, y, label)

    # (b) full-data refit with its own tuned hyperparameters -- this is the
    # model that would actually be deployed/interpreted for this variant, so
    # it's built now rather than only for whichever variant looks best on the
    # nested-CV numbers alone.
    scale_pos_weight = cv_result['scale_pos_weight']
    final_search = RandomizedSearchCV(
        xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric='aucpr', random_state=0),
        param_distributions=param_dist,
        n_iter=40, scoring='average_precision',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        random_state=0, n_jobs=-1
    )
    final_search.fit(X, y)

    full_results[label] = {
        'features': feats,
        'cv_result': cv_result,
        'final_search': final_search,
        'final_model': final_search.best_estimator_,
        'final_threshold': cv_result['mean_threshold'],
    }

    print(f"[{label}] nested CV: TSS={cv_result['mean_tss']:.3f} ± {cv_result['std_tss']:.3f}  "
          f"PR-AUC={cv_result['mean_pr_auc']:.3f} ± {cv_result['std_pr_auc']:.3f}")

###################################################################################################
######################## 3. FULL-TRAINING COMPARISON ACROSS ALL VARIANTS ##########################
###################################################################################################

print("\n" + "="*60)
print("FULL-TRAINING COMPARISON ACROSS ALL VARIANTS")
print("="*60)

summary_rows = []
for label, res in full_results.items():
    cvr = res['cv_result']
    summary_rows.append({
        'variant': label,
        'features': res['features'],
        'nested_tss_mean': cvr['mean_tss'],
        'nested_tss_std': cvr['std_tss'],
        'nested_pr_auc_mean': cvr['mean_pr_auc'],
        'nested_pr_auc_std': cvr['std_pr_auc'],
    })

full_comparison_df = pd.DataFrame(summary_rows).sort_values(
    'nested_tss_mean', ascending=False
).reset_index(drop=True)

display_table = pd.DataFrame({
    'variant': full_comparison_df['variant'],
    'features': full_comparison_df['features'],
    'TSS (nested CV)': full_comparison_df.apply(
        lambda r: f"{r['nested_tss_mean']:.3f} ± {r['nested_tss_std']:.3f}", axis=1),
    'PR-AUC (nested CV)': full_comparison_df.apply(
        lambda r: f"{r['nested_pr_auc_mean']:.3f} ± {r['nested_pr_auc_std']:.3f}", axis=1),
})
print(display_table.to_string(index=False))

WINNING_FEATURES = full_comparison_df.iloc[0]['variant']
print(f"\nWinner by full-training nested-CV TSS: '{WINNING_FEATURES}' "
      f"{feature_sets[WINNING_FEATURES]} -- override manually below if the paired "
      f"comparison or parsimony says otherwise (e.g. if two variants are "
      f"statistically indistinguishable, prefer the simpler one).")

###################################################################################################
################### 4. DEEP DIVE ON THE WINNING VARIANT (reuses its sweep results) ################
###################################################################################################

print("\n" + "="*60)
print(f"DEEP DIVE: '{WINNING_FEATURES}' {feature_sets[WINNING_FEATURES]}")
print("="*60)

winning = full_results[WINNING_FEATURES]
cv_result = winning['cv_result']
X = all_events[feature_sets[WINNING_FEATURES]]
final_search = winning['final_search']
final_model = winning['final_model']
final_threshold = winning['final_threshold']
mean_tss, std_tss = cv_result['mean_tss'], cv_result['std_tss']
mean_pr_auc, std_pr_auc = cv_result['mean_pr_auc'], cv_result['std_pr_auc']
outer_tss_scores = cv_result['tss_scores']
outer_pr_auc_scores = cv_result['pr_auc_scores']

print("Final model params:", final_search.best_params_)
print(f"Recommended decision threshold: {final_threshold:.3f}\n")

# --- Permutation test (winning variant only -- see note in the sweep above) ---
n_permutations = 200
perm_result = run_permutation_test(
    X, y, final_search.best_params_, mean_tss, mean_pr_auc,
    label=WINNING_FEATURES, n_permutations=n_permutations
)
perm_tss_scores = perm_result['perm_tss_scores']
perm_pr_auc_scores = perm_result['perm_pr_auc_scores']
p_value_tss = perm_result['p_value_tss']
p_value_pr_auc = perm_result['p_value_pr_auc']

# --- SHAP ---
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, show=False)

plt.tight_layout()
plt.savefig("model_results_ts/8c_ocurrence_shap.png", dpi=300, bbox_inches="tight")

# --- Final model confusion matrix (in-sample sanity check only) ---
y_prob_final = final_model.predict_proba(X)[:, 1]
y_pred_final = (y_prob_final >= final_threshold).astype(int)

cm = confusion_matrix(y, y_pred_final)
cm_df = pd.DataFrame(cm,
                      index=['Actual: No Osc', 'Actual: Osc'],
                      columns=['Pred: No Osc', 'Pred: Osc'])

print("Confusion matrix (final model, in-sample, at recommended threshold):")
print(cm_df)
print(f"\nIn-sample TSS: {tss_score(y, y_pred_final):.4f}\n")

###################################################################################################
######################## 5. FINAL SUMMARY WITH PERMUTATION TEST, FOR PAPER ########################
###################################################################################################
# perm_tss_scores/perm_pr_auc_scores/p_value_tss/p_value_pr_auc were just
# computed for this variant in section 4 above.

print("\n" + "="*60)
print(f"FINAL PERFORMANCE WITH PERMUTATION TEST ('{WINNING_FEATURES}') — for paper reporting")
print("="*60)
print(f"Observed TSS: {mean_tss:.4f}  |  Null TSS: {perm_tss_scores.mean():.4f} +/- {perm_tss_scores.std():.4f}")
print(f"p-value (TSS): {p_value_tss:.4f}  ({(perm_tss_scores >= mean_tss).sum()}/{n_permutations} "
      f"null runs >= observed)")
print(f"\nObserved PR-AUC: {mean_pr_auc:.4f}  |  Null PR-AUC: {perm_pr_auc_scores.mean():.4f} "
      f"+/- {perm_pr_auc_scores.std():.4f}")
print(f"p-value (PR-AUC): {p_value_pr_auc:.4f}  ({(perm_pr_auc_scores >= mean_pr_auc).sum()}/{n_permutations} "
      f"null runs >= observed)")

summary_table_perm = pd.DataFrame({
    'Metric': ['TSS', 'PR-AUC'],
    'Observed (nested CV mean ± std)': [f'{mean_tss:.3f} ± {std_tss:.3f}',
                                          f'{mean_pr_auc:.3f} ± {std_pr_auc:.3f}'],
    'Null (permutation mean ± std)': [f'{perm_tss_scores.mean():.3f} ± {perm_tss_scores.std():.3f}',
                                        f'{perm_pr_auc_scores.mean():.3f} ± {perm_pr_auc_scores.std():.3f}'],
    'p-value': [f'{p_value_tss:.4f}', f'{p_value_pr_auc:.4f}']
})
print(summary_table_perm.to_string(index=False))