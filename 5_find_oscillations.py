"""
==================================================================================
Detect shock-triggered wave activity in GOES magnetometer data at ANY MLT.

Search for the WAVE ACTIVITY a shock/discontinuity triggers at GEO,
at any MLT. For each shock arrival time tA we ask:

    Is there a statistically significant magnetic oscillation at GEO in the window
    (tA - PRE_HOURS, tA + POST_HOURS), above the pre-shock baseline, and what is
    its period / band (is it Pc5)?

For each short sliding window we take the single
highest spectral peak, test it against an AR(1) red-noise null, and keep the most
significant of the three MFA components (b_radial, b_azimuthal, b_parallel).

Pipeline
--------
    L1 shock catalogue (tA)                 <-- YOU fill this in (load_shock_catalogue)
      -> assemble GOES MAG window [tA-PRE, tA+POST]   (crosses midnight; handled below)
      -> detect_shock_triggered_ulf()                 (fdl_ulf_shock_detect.py)
      -> per-shock summary + per-window detections     -> CSV

==================================================================================
"""

import os
import datetime
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from fdl_ulf_shock_detect import detect_shock_triggered_ulf, DEFAULT_BAND_HZ
from helper import MFAC_convert, extract_drift_echo_candidates_dyn_echo
import spacepy

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# =============================================================================
# CONFIG
# =============================================================================
SATELLITE   = 'GOES13'
OUTPUT_DIR  = 'ULF_shocks'

# --- search window around each arrival time tA ---
PRE_HOURS   = 1.0     # baseline window before tA  (widen to 2-3 h for a steadier baseline)
POST_HOURS  = 12.0    # search window after tA  (generous upper bound for any MLT)

# --- sliding analysis window (a transient Pc5 wavetrain is only a few cycles) ---
WIN_MIN     = 60      # analysis window length (minutes)
STEP_MIN    = 30      # stride between windows (minutes)

# --- detection thresholds (these are your catalogue-cut knobs) ---
FAP_THRESHOLD = 0.05  # peak is "significant" if its global false-alarm prob < this
ENHANCEMENT   = 1.0   # post-shock peak must exceed ENHANCEMENT x pre-shock baseline
N_SIM         = 2000  # AR(1) surrogates per window (more = smoother FAP, slower)

# --- band: periods 120-600 s by default (Pc5 + resolvable upper Pc4). ---
STRICT_PC5  = False   # True -> restrict the peak search to strict Pc5 (150-600 s)
BAND_HZ     = (1.0 / 600.0, 1.0 / 150.0) if STRICT_PC5 else DEFAULT_BAND_HZ

# --- components tested per window; the most significant is kept ---
COMPONENTS  = ('b_radial', 'b_azimuthal', 'b_parallel')

N_JOBS      = 7       # parallel shocks

# Optional: validate on ONE known shock first. Set to e.g. '2017-09-07 23:00:00'
# to run a single event with verbose output and skip the full catalogue.
# TEST_SINGLE_TA = '2017-07-09 00:11:30'
TEST_SINGLE_TA = None

def mfac_for_date(d):
    """
    Load one day of GOES magnetic field in MFA coordinates.

    Returns a DataFrame with columns ['time', 'mlt', 'b_parallel', 'b_azimuthal',
    'b_radial'] at 1-min cadence, or None if the day is unavailable/unreadable.

    Looks for 'goes_annual_combined/{yr}_g{satnum}_calc.pkl'; returns None
    silently if the file doesn't exist (e.g. satellite not operational that year).
    """
    import glob

    satnum = SATELLITE.split('GOES')[-1]
    yr = d.year

    pattern = f'goes_annual_combined/{yr}_g{satnum}_calc.pkl'
    if not glob.glob(pattern):
        return None

    try:
        goes_data = joblib.load(pattern)
        goes_data_d = goes_data.loc[goes_data.index.date == d]
        if goes_data_d.empty:
            return None

        df_mfac = MFAC_convert(goes_data_d[['He', 'Hn', 'Hp']])
        df_mfac.insert(1, 'mlt', goes_data_d['mlt'].values)
        return df_mfac

    except Exception as exc:
        print(f"[mfac_for_date] Could not load {pattern}: {exc}")
        return None


def load_window(tA, pre_hours=PRE_HOURS, post_hours=POST_HOURS):
    """
    Assemble a continuous df_mfac spanning [tA - pre_hours, tA + post_hours].

    The window routinely crosses midnight, so we load every calendar day it
    touches and concatenate. Returns None if no data is available.
    """
    tA = pd.Timestamp(tA)
    days = sorted({
        (tA - pd.Timedelta(hours=pre_hours)).date(),
        tA.date(),
        (tA + pd.Timedelta(hours=post_hours)).date(),
    })
    frames = [f for f in (mfac_for_date(d) for d in days) if f is not None]
    if not frames:
        return None
    df = (pd.concat(frames, ignore_index=True)
            .drop_duplicates('time')
            .sort_values('time')
            .reset_index(drop=True))
    return df

goes_arrivals = joblib.load('goes_impacts_xcorr.pkl')
def load_shock_catalogue():
    """
    Return the L1 IP-shock catalogue as a DataFrame.

    REQUIRED: a column 'tA' of timezone-naive UTC datetimes = the shock arrival
    time at Earth/GEO (already propagated from L1; if not, propagate here).

    Optional but handy: keep any shock parameters (e.g. 'v_sw', 'mach', 'theta_bn',
    'shock_id') as extra columns -- they are carried through into the output so you
    can correlate them with the detected wave activity afterwards.

    Example (edit to your file):
        cat = pd.read_csv('shock_catalogue.csv', parse_dates=['arrival_time'])
        cat = cat.rename(columns={'arrival_time': 'tA'})
        return cat[['tA', 'v_sw', 'mach', 'shock_id']]
    """

    cat = goes_arrivals.rename(columns={'xcorr_arrival': 'tA'})

    satnum = SATELLITE.split('GOES')[-1]
    cat = cat.loc[cat['goes_num'] == satnum]

    cat['shock_vx'] = cat['v_corrected'].apply(lambda x: x[0])
    cat['shock_vy'] = cat['v_corrected'].apply(lambda x: x[1])
    cat['shock_vz'] = cat['v_corrected'].apply(lambda x: x[2])
    cat.drop(columns=['v_corrected'], inplace=True)
    return cat


# =============================================================================
# Per-shock processing
# =============================================================================
def process_shock(shock_row, verbose=False):
    """
    Run the detector for a single shock. `shock_row` is a dict-like with 'tA'
    (and optionally extra catalogue parameters, which are passed through).

    Returns (summary_dict, detection_windows_df) -- both tagged with the shock's
    tA and any extra catalogue columns. Either may be empty if no data/detection.
    """
    tA = pd.Timestamp(shock_row['tA'])
    extra = {k: v for k, v in dict(shock_row).items() if k != 'tA'}

    df_mfac = load_window(tA)
    if df_mfac is None or df_mfac.empty:
        if verbose:
            print(f"  {tA}: no MAG data in window -- skipped")
        summary = {'tA': tA, 'status': 'no_data', **extra}
        # keep the shock in the output with NaN GEO-derived fields, but retain
        # its L1 parameters -- build this by name, not position, same reasoning
        # as the earlier fixes.
        placeholder = pd.DataFrame(index=[0])
        placeholder['tA'] = tA
        placeholder['status'] = 'no_data'
        for k, v in extra.items():
            placeholder[k] = v
        return (summary, placeholder)

    print('-------------------------------- Branch 1: Pc5 oscillations --------------------------------')

    detections, summary, all_windows = detect_shock_triggered_ulf(
        df_mfac, tA,
        pre_hours=PRE_HOURS, post_hours=POST_HOURS,
        win_min=WIN_MIN, step_min=STEP_MIN,
        band=BAND_HZ, N_sim=N_SIM,
        fap_threshold=FAP_THRESHOLD, enhancement=ENHANCEMENT,
    )
    summary = {**summary, 'status': 'ok', **extra}

    # tag the detection windows with this shock so the combined table is traceable
    if not detections.empty:
        detections = detections.copy()
        detections['tA'] = tA
        for k, v in extra.items():
            detections[k] = v
    else:
        # detections may come back with 0 rows AND 0 columns in some edge cases —
        # start from a bare 1-row index and add columns by name, which works
        # regardless of what (if any) columns detections already had.
        print(detections.shape, extra.get('shock_id', tA))
        detections = pd.DataFrame(index=[0])
        for c in detections.columns:  # preserves any columns it did have, as NaN
            detections[c] = None
        detections['tA'] = tA
        for k, v in extra.items():
            detections[k] = v

    print('-------------------------------- Branch 2: Lower-freq oscillations --------------------------------')

    freq_cols = ['freq', 'amp', 'phase', 'period', 't1_mlt', 't2_mlt', 'date1_utc', 'date2_utc']
    events_filtered = pd.DataFrame(columns=freq_cols + ['tA', 'comp'])

    if summary['n_detections'] > 0:
        selected_component = summary['best_component']

        df_selected = df_mfac[['time', 'mlt', selected_component]]
        df_selected = df_selected.rename(columns={selected_component: 'b_component'})
        df_selected = df_selected.loc[(df_selected['time'] + datetime.timedelta(hours=PRE_HOURS)) >= tA]
        df_selected = df_selected.loc[df_selected['time'] <= (tA + datetime.timedelta(hours=POST_HOURS))]

        all_candidates = extract_drift_echo_candidates_dyn_echo(df_selected, 0, delta_t=4, verbose_midnight=False)
        for candidate in all_candidates:
            if len(candidate) > 0:
                mask = (candidate['period'] >= 10) & (candidate['period'] <= 40)
                events_filtered = candidate.loc[mask].reset_index(drop=True)

        # Assign tA/comp explicitly BY NAME, not positionally, and do it
        # regardless of how many rows survived the mask (0, 1, or many).
        events_filtered['tA'] = tA
        events_filtered['comp'] = selected_component

        if len(events_filtered) > 1:
            events_post = events_filtered.loc[events_filtered['tA'] >= tA]
            events_filtered = events_post if len(events_post) > 0 else events_filtered.iloc[[-1]]
        if len(events_filtered) > 1:
            events_filtered = events_filtered.iloc[[0]]

    if len(events_filtered) == 0:
        # Build the fallback row BY NAME so column order can never matter.
        row = {c: None for c in freq_cols}
        row['tA'] = tA
        row['comp'] = None
        events_filtered = pd.DataFrame([row])

    # push the branch-2 fields into summary too, so the summary CSV carries them
    first = events_filtered.iloc[0]
    for c in freq_cols + ['comp']:
        summary[c] = first[c]

    detections = pd.merge(detections, events_filtered, on='tA', how='left')


    if verbose:
        print(f"  {tA}: {summary['n_post_significant']} significant / "
              f"{summary['n_detections']} detections"
              + (f" | onset +{summary.get('onset_minutes', np.nan):.0f} min @ MLT "
                 f"{summary.get('onset_mlt', np.nan):.1f}, "
                 f"{summary.get('best_period_s', np.nan):.0f} s "
                 f"({summary.get('best_band', '-')}), comp "
                 f"{summary.get('best_component', '-')}"
                 if summary['n_detections'] else ""))

    return summary, detections

# =============================================================================
# Main
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- single-event validation mode ---
    if TEST_SINGLE_TA is not None:
        print(f"\n[TEST MODE] single shock at {TEST_SINGLE_TA}\n")
        summary, detections = process_shock({'tA': TEST_SINGLE_TA}, verbose=True)
        print("\nSummary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        if not detections.empty:
            cols = ['win_center', 'mlt', 'component', 'peak_period_s',
                    'band', 'is_pc5', 'fap', 'peak_amp_nT', 'tA', 'freq', 'amp', 'period', 'date1_utc', 'comp']
            print("\nDetection windows:")
            print(detections[cols].to_string(index=False))
        return

    # --- full catalogue ---
    shocks = load_shock_catalogue()
    shocks = shocks.dropna(subset=['tA']).reset_index(drop=True)
    print(f"\nProcessing {len(shocks)} shocks "
          f"[window (tA-{PRE_HOURS:g}h, tA+{POST_HOURS:g}h), "
          f"win {WIN_MIN} min / step {STEP_MIN} min, "
          f"band {1/BAND_HZ[1]:.0f}-{1/BAND_HZ[0]:.0f} s]\n")

    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(process_shock)(row) for _, row in shocks.iterrows()
    )

    summaries = pd.DataFrame(
        [s for s, _ in results])  # no status filter — no_data shocks now keep their L1 fields, NaN GEO fields
    detections = [d for _, d in results if d is not None and not d.empty]
    detections = (pd.concat(detections, ignore_index=True)
                  if detections else pd.DataFrame())

    summary_path = os.path.join(OUTPUT_DIR, f'ulf_shock_summary_{SATELLITE}.csv')
    detect_path  = os.path.join(OUTPUT_DIR, f'ulf_shock_detections_{SATELLITE}.csv')
    summaries.to_csv(summary_path, index=False)
    detections.to_csv(detect_path, index=False)

    n_det = int((summaries.get('n_detections', pd.Series(dtype=int)) > 0).sum())
    print(f"\nDone. {n_det}/{len(summaries)} shocks with a triggered detection.")
    print(f"  per-shock summary  -> {summary_path}")
    print(f"  detection windows  -> {detect_path}")


if __name__ == '__main__':
    main()
