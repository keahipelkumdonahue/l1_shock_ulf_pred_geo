import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import re
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import average_precision_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, make_scorer

###################################################################################################
############################################ LOAD DATA ############################################
###################################################################################################

all_events = joblib.load('ULF_shocks/all_ulf_shock_events.pkl')
print(all_events.keys())

###################################################################################################
##################################### MERGE IN TIME-SERIES SCORE ##################################
###################################################################################################

ts_window_margin = 20
ts_score_df = joblib.load(f'ts_window_scan/{ts_window_margin}/ts_scores_for_merge.pkl')
all_events['l1_departure_time'] = pd.to_datetime(all_events['l1_departure_time'])
ts_score_df['time'] = pd.to_datetime(ts_score_df['time'])

# l1_departure_time is not a unique key in all_events (multiple goes_num rows
# can share the same L1 shock time)

n_before_dedup = len(ts_score_df)
ts_score_df = ts_score_df.drop_duplicates(subset='time')
print(f"ts_score_df deduplicated on time: {n_before_dedup} -> {len(ts_score_df)} rows")

n_events_pre_merge = len(all_events)
all_events = all_events.merge(ts_score_df, left_on='l1_departure_time', right_on='time', how='left')
assert len(all_events) == n_events_pre_merge, (
    f"Merge changed row count ({n_events_pre_merge} -> {len(all_events)}) -- "
    f"left join should never do this; check for duplicate/near-duplicate keys."
)

n_with_ts = all_events['ts_score_b'].notna().sum()
print(f"\nts_score available for {n_with_ts} / {len(all_events)} events "
      f"({len(all_events) - n_with_ts} missing -- dropped in 5c before a time series was saved)")

# Convert direction to binary: FF=0, FR=1
all_events['direction'] = all_events['direction'].replace({'ff': 0, 'fr': 1}).infer_objects(copy=False)
all_events['direction'] = all_events['direction'].astype(np.float64)

input_params = ['direction', 'avg_r2', 'fit_amp', 'fit_vert', 'fit_rate',
                'shock_vx', 'shock_vy', 'shock_vz', 'imf_clock', 'shock_clock',
                'l1_pre_b', 'l1_delta_b', 'l1_delta_b_cv', 'l1_pre_std_b',
                'l1_pre_v', 'l1_delta_v', 'l1_delta_v_cv',
                'l1_pre_np','l1_delta_np',
                'l1_pre_fp', 'l1_delta_fp',
                'arrival_goes_mlt',
                'ts_score_b', 'ts_score_v', 'ts_score_combined', 'ts_score_fp', 'ts_score_all']
# NOTE: VIF/correlation below use X.dropna(), which will additionally drop the
# events missing ts_score for this diagnostic section only -- it does not affect
# the RFECV/permutation or variant-comparison sections further down.

osc_exist = []
for i in range(len(all_events)):
    event = all_events.iloc[i]
    if event['n_detections'] == 0:
        osc_exist.append(False)
    else:
        osc_exist.append(True)
all_events['oscillation'] = osc_exist

X = all_events[input_params]
y = all_events['oscillation']

###################################################################################################
###################################### SPEARMAN CORRELATIONS ######################################
###################################################################################################

print("\n.............................Spearman correlations .............................")

corr = X.corr(method='spearman')

# Flag high-correlation pairs
threshold = 0.85
high_corr_pairs = []
cols = corr.columns
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        r = corr.iloc[i, j]
        if abs(r) > threshold:
            high_corr_pairs.append((cols[i], cols[j], r))

high_corr_pairs = sorted(high_corr_pairs, key=lambda x: -abs(x[2]))

print(f"\n{len(high_corr_pairs)} pairs above |rho| = {threshold}:")
for f1, f2, r in high_corr_pairs:
    print(f"  {f1:25s} <-> {f2:25s}  rho = {r:.3f}")

###################################################################################################
############################################### VIF ###############################################
###################################################################################################

X_vif = X.dropna()
print(f"\nRows used for VIF after dropna: {len(X_vif)} (out of {len(X)})")

def compute_vif(df_in):
    vif_data = pd.DataFrame()
    vif_data['feature'] = df_in.columns
    vif_data['VIF'] = [variance_inflation_factor(df_in.values, i)
                        for i in range(df_in.shape[1])]
    return vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)

print("\n.............................Initial VIF:.............................")

vif_result = compute_vif(X_vif)
print(vif_result)

# --- capture VIF at each selection stage for plotting (stages = kept path, rejected = discarded branches)
stages = {}      # starts after manual drop of uninformative features
rejected = {}

print('\n.............................Dropping physically uninformative features:.............................')

manual_drop = ['fit_amp', 'fit_vert', 'avg_r2']
X_reduced = X_vif.drop(columns=manual_drop)

vif_result = compute_vif(X_reduced)
print(vif_result)
stages["1. All physical features"] = vif_result

print('\n.............................Dropping either scalar FP or V+Np?:.............................')

# Option A: pressure (Fp) only, no raw V/Np features
drop_A = ['l1_pre_np', 'l1_delta_np', 'l1_pre_v', 'l1_delta_v']

# Option B: density/velocity (V+Np) only, no raw Fp features
drop_B = ['l1_pre_fp', 'l1_delta_fp']

X_optA = X_reduced.drop(columns=drop_A)
X_optB = X_reduced.drop(columns=drop_B)

vif_fp = compute_vif(X_optA)
vif_npv = compute_vif(X_optB)
print("Option A VIF (FP features only):")
print(vif_fp)
print("\nOption B VIF (Raw Np/v features only):")
print(vif_npv)
stages["2. Keep FP, drop Np/v"] = vif_fp
rejected["2'. (rejected) Keep Np/v, drop Fp"] = vif_npv

print('RESULT: KEEP SCALAR FP FEATURES, DROP NP/V:')
X_reduced = X_reduced.drop(columns=drop_A)

print('\n.............................Dropping either ts combined or all:.............................')

# Option A: combined (B+v) only
drop_A = ['ts_score_all']

# Option B: all (B+v+FP) only
drop_B = ['ts_score_combined']

X_optA = X_reduced.drop(columns=drop_A)
X_optB = X_reduced.drop(columns=drop_B)

vif_ts_comb = compute_vif(X_optA)
vif_ts_all = compute_vif(X_optB)
print("Option A VIF (combined B+v only):")
print(vif_ts_comb)
print("\nOption B VIF (all B+v+NP only):")
print(vif_ts_all)
stages["3. Keep ts_all, drop ts_combined"] = vif_ts_all
rejected["3'. (rejected) Keep ts_combined, drop ts_all"] = vif_ts_comb

print('RESULT: KEEP All, DROP COMBINED:')
X_reduced = X_reduced.drop(columns=drop_B)

###################################################################################################
############################################ VIF PLOTS ############################################
###################################################################################################
import os

def plot_vif_rank_curves(stages, rejected=None, title="Oscillation Occurrence",
                         fname='paper figures/vif_occurrence.png'):
    """Sorted-VIF curve per stage: x = feature rank (1 = highest VIF), y = VIF (log)."""
    fig, ax = plt.subplots(figsize=(4, 6))
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(stages)))

    for (name, d), c in zip(stages.items(), colors):
        v = d.sort_values("VIF", ascending=False).reset_index(drop=True)
        ax.plot(np.arange(1, len(v) + 1), v["VIF"], "-o", ms=4, lw=2, color=c,
                label=f"{name}, max={v.VIF.max():.0f})")
        # ax.annotate(v.loc[0, "feature"], (1, v.loc[0, "VIF"]),
        #             xytext=(6, 0), textcoords="offset points",
        #             fontsize=7, color=c, va="center")

    # rejected branches as faint dotted lines
    stage_color = {int(re.match(r"\d+", n).group()): c for (n, _), c in zip(stages.items(), colors)}
    for name, d in (rejected or {}).items():
        c = stage_color[int(re.match(r"\d+", name).group())]
        v = d.sort_values("VIF", ascending=False).reset_index(drop=True)
        ax.plot(np.arange(1, len(v) + 1), v["VIF"], "--", lw=1.2, color=c, alpha=0.7,
                label=f"{name} (n={len(v)}, max={v.VIF.max():.0f})")

    ax.set_yscale("log")
    ax.set_xlabel("Feature rank (1 = highest VIF)")
    ax.set_ylabel("VIF (log)")
    ax.set_title(title)
    ax.set_ylim(1,50)
    ax.grid(alpha=.3, which="both")
    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=300, format='png', bbox_inches='tight')
    plt.show()


plot_vif_rank_curves(stages, rejected)

###################################################################################################
############################################## TSS SCORER #########################################
###################################################################################################

def tss_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity + specificity - 1.0

tss_scorer = make_scorer(tss_score)

# Derived directly from X_optA/X_optB above (not re-typed) so the RFECV
# candidates can't silently drift out of sync with the VIF-driven drops.
rfecv_candidates = list(X_reduced.columns)

print('\n\nRFECV candidates:', rfecv_candidates)

events_ts = all_events.dropna(subset=['ts_score_b']).reset_index(drop=True)
print(f"Using {len(events_ts)} / {len(all_events)} events with a valid ts_score "
      f"for RFECV/permutation/variant comparisons below.")

y = events_ts['oscillation']
scale_pos_weight = (y == 0).sum() / (y == 1).sum()

X = events_ts[rfecv_candidates]

###################################################################################################
################################## REPEATED PERMUTATIONS + RFECV (TSS) ############################
###################################################################################################

n_repeats_cv = 30

def run_rfecv_permutation(X, y, candidate_features, scale_pos_weight, label, n_repeats_cv=30):
    """
    Repeated (seed x 5-fold) permutation importance + RFECV, TSS-scored.
    Returns (imp_summary, rfecv_summary, rfecv_selected_counts) so the two
    variants can be compared side by side afterward.
    """
    all_importances = []
    rfecv_selected_counts = []
    rfecv_selection_flags = []

    for seed in range(n_repeats_cv):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='aucpr',
            random_state=seed
        )

        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)
            result = permutation_importance(
                model, X_val, y_val,
                scoring=tss_scorer,
                n_repeats=50,
                random_state=seed,
                n_jobs=-1
            )
            all_importances.append(result.importances_mean)

        rfecv = RFECV(
            estimator=model,
            step=1,
            cv=cv,
            scoring=tss_scorer,
            min_features_to_select=3,
            n_jobs=-1
        )
        rfecv.fit(X, y)
        rfecv_selected_counts.append(rfecv.n_features_)
        rfecv_selection_flags.append(rfecv.support_)

        print(f"[{label}] seed {seed}: RFECV optimal n_features = {rfecv.n_features_}, "
              f"selected = {list(X.columns[rfecv.support_])}")

    all_importances = np.array(all_importances)
    imp_summary = pd.DataFrame({
        'feature': candidate_features,
        'mean_importance': all_importances.mean(axis=0),
        'std_importance': all_importances.std(axis=0),
        'pct_runs_positive': (all_importances > 0).mean(axis=0) * 100
    }).sort_values('mean_importance', ascending=False)

    print(f"\n[{label}] Permutation importance summary (TSS drop) across all seeds/folds:")
    print(imp_summary.to_string(index=False))

    rfecv_selection_flags = np.array(rfecv_selection_flags)
    rfecv_summary = pd.DataFrame({
        'feature': candidate_features,
        'pct_seeds_selected': rfecv_selection_flags.mean(axis=0) * 100
    }).sort_values('pct_seeds_selected', ascending=False)

    print(f"\n[{label}] RFECV selection frequency across seeds (TSS-scored):")
    print(rfecv_summary.to_string(index=False))
    print(f"\n[{label}] Optimal feature count across seeds: mean={np.mean(rfecv_selected_counts):.1f}, "
          f"values={rfecv_selected_counts}")

    return imp_summary, rfecv_summary, rfecv_selected_counts

print("\n=====================  RUN FULL FEATURE SELECTION =====================")
imp_summary, rfecv_summary, rfecv_counts = run_rfecv_permutation(
    X, y, rfecv_candidates, scale_pos_weight, label='A: Fp, ts_score_all', n_repeats_cv=n_repeats_cv
)
