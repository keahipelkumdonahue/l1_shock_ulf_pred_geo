"""
Shared configuration for the 10a-10d amplitude-classification pipeline.
"""

import numpy as np

# =====================================================================
# TOGGLE -- everything else in the pipeline follows from these two lines
# =====================================================================
CASE = 'pc5'          # 'pc5' or 'subpc5'
AMP_PERCENTILE = 90   # e.g. 50 for median, 75 for upper-quartile cutoff

# Time-series window margin (minutes)
TS_WINDOW_MARGIN = 20

# ---------------------------------------------------------------------
# Per-case column definitions -- add a new case here if needed later
# ---------------------------------------------------------------------
_CASE_CONFIG = {
    'pc5': {
        'filter_col': 'best_is_pc5',
        'amp_col': 'best_amp_nT',
    },
    'subpc5': {
        'filter_col': 'comp',
        'amp_col': 'amp',
    },
}


def get_case_config(case=None):
    """Returns (case, cfg_dict). Defaults to the module-level CASE."""
    case = case or CASE
    if case not in _CASE_CONFIG:
        raise ValueError(f"Unknown CASE '{case}' -- expected one of {list(_CASE_CONFIG)}")
    return case, _CASE_CONFIG[case]


def in_case_mask(all_events, case=None):
    """Boolean mask selecting rows belonging to `case` within all_events."""
    case, cfg = get_case_config(case)
    if case == 'pc5':
        return all_events[cfg['filter_col']] == True
    else:  # subpc5
        return all_events[cfg['filter_col']].notna()


def select_case_events(all_events, case=None):
    """Subset of all_events belonging to `case` (pc5 or subpc5)."""
    return all_events.loc[in_case_mask(all_events, case)]


def compute_amp_cutoff(all_events, case=None, percentile=None):
    """Amplitude cutoff (percentile-th percentile of amp_col, within case)."""
    case, cfg = get_case_config(case)
    percentile = AMP_PERCENTILE if percentile is None else percentile
    case_events = select_case_events(all_events, case)
    return np.percentile(case_events[cfg['amp_col']], percentile)


def build_amp_label(all_events, case=None, percentile=None, cutoff=None):

    case, cfg = get_case_config(case)
    if cutoff is None:
        cutoff = compute_amp_cutoff(all_events, case, percentile)

    mask = in_case_mask(all_events, case).to_numpy()
    amp = all_events[cfg['amp_col']].to_numpy()

    labels = np.full(len(all_events), -1, dtype=int)
    labels[mask] = (amp[mask] > cutoff).astype(int)
    return labels, cutoff


# ---------------------------------------------------------------------
# Filenames -- tagged with case + percentile (+ window margin, where
# relevant) so different toggle settings never collide on disk.
# ---------------------------------------------------------------------

def amp_tag(percentile=None):
    percentile = AMP_PERCENTILE if percentile is None else percentile
    return f"p{int(percentile)}"


def classified_events_path(case=None, percentile=None):
    """Output of 10a."""
    case, _ = get_case_config(case)
    return f'ULF_shocks/{case}_classified_events_{amp_tag(percentile)}.pkl'


def ts_scores_path(case=None, percentile=None, window_margin=None):
    """Output of 10b / input to 10c & 10d."""
    case, _ = get_case_config(case)
    window_margin = TS_WINDOW_MARGIN if window_margin is None else window_margin
    return f'ampclass_{case}_ts_scores_for_merge_{amp_tag(percentile)}_{window_margin}min.pkl'
