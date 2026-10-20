"""
Shock-triggered ULF / wave-activity detector for GOES magnetometer data.

Goal: given an IP-shock arrival time tA (propagated to Earth), detect whether the
GOES magnetic field at GEO shows significant wave activity at *any* MLT in the
window (tA - pre_hours, tA + post_hours), and characterise the dominant peak
(period, band, is_Pc5). This is the simplified replacement for the CLEAN method:
we do NOT deconvolve the spectrum, we just take the single highest peak per window
and test it against an AR(1) red-noise null.

Reuses the proven AR(1) machinery from the drift-echo code (helpers_padded).
"""

import numpy as np
import pandas as pd

from fdl_helpers_padded import _fit_ar1_yw, _simulate_ar1

# Continuous-pulsation Pc bands (period in seconds). 1-min data resolves periods
# >= 120 s, so only Pc5 (and the long edge of Pc4) is accessible.
PC_BANDS = {'Pc3': (10, 45), 'Pc4': (45, 150), 'Pc5': (150, 600)}

# Default search band, in Hz: periods 120-600 s (all of Pc5 + resolvable upper Pc4).
# Use (1/600, 1/150) for strict Pc5 only.
DEFAULT_BAND_HZ = (1.0 / 600.0, 1.0 / 120.0)


def classify_ulf_band(period_s):
    """Label a period (seconds) by its Pc band."""
    if period_s is None or not np.isfinite(period_s):
        return 'undefined'
    for name, (lo, hi) in PC_BANDS.items():
        if lo <= period_s < hi:
            return name
    if period_s >= 600:
        return 'sub-Pc5'          # longer than Pc5
    return 'sub-resolution'        # shorter than 1-min data can resolve


def _parabolic_peak(freqs, amp, k):
    """3-point parabolic refinement of the peak at bin k -> (freq, amp)."""
    if 0 < k < len(amp) - 1:
        y0, y1, y2 = amp[k - 1], amp[k], amp[k + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom != 0:
            delta = 0.5 * (y0 - y2) / denom                  # sub-bin shift in (-0.5, 0.5)
            d_f = freqs[1] - freqs[0]
            return freqs[k] + delta * d_f, y1 - 0.25 * (y0 - y2) * delta
    return freqs[k], amp[k]


def dominant_peak_significance(signal, dt_sec=60.0, N_sim=2000,
                               band=DEFAULT_BAND_HZ, detrend='linear'):
    """
    Single largest spectral peak + AR(1) red-noise significance. No CLEAN.

    Parameters
    ----------
    signal   : 1-D array (one component, one window), must be finite.
    dt_sec   : sample spacing in seconds (60 for avg1m).
    N_sim    : number of AR(1) surrogates.
    band     : (f_lo, f_hi) Hz search band; peak is taken as the band maximum.
    detrend  : 'linear' (removes the SI step + trend) or 'mean'.

    Returns dict (or None if window unusable):
      peak_freq_hz, peak_period_s, peak_amp_nT, fap, snr, band, is_pc5
    `fap` is a GLOBAL false-alarm probability: the observed band-max compared to
    the distribution of surrogate band-maxima, so it is already corrected for the
    number of frequencies searched (the look-elsewhere effect).
    """
    x = np.asarray(signal, dtype=float)
    n = x.size
    if n < 20 or not np.all(np.isfinite(x)):
        return None

    t = np.arange(n)
    if detrend == 'linear':
        x = x - np.polyval(np.polyfit(t, x, 1), t)
    else:
        x = x - np.nanmean(x)

    w = np.hanning(n)
    CG = w.mean()                      # coherent gain (0.5 for Hanning)
    norm = 2.0 / (n * CG)
    freqs = np.fft.rfftfreq(n, dt_sec)
    amp = norm * np.abs(np.fft.rfft(x * w))

    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if not in_band.any():
        return None

    k = int(np.argmax(np.where(in_band, amp, -np.inf)))
    peak_f, peak_amp = _parabolic_peak(freqs, amp, k)

    # AR(1) red-noise null; global significance via surrogate band-maxima
    try:
        phi, sigma = _fit_ar1_yw(x)
    except RuntimeError:
        phi, sigma = 0.0, float(np.nanstd(x))
    sur = _simulate_ar1(phi, sigma, n, N_sim)
    sur_amp = norm * np.abs(np.fft.rfft(sur * w[None, :], axis=1))
    sur_max = sur_amp[:, in_band].max(axis=1)
    fap = (np.sum(sur_max >= peak_amp) + 1.0) / (N_sim + 1.0)   # conservative, never 0

    period_s = 1.0 / peak_f if peak_f > 0 else np.nan
    return {
        'peak_freq_hz': peak_f,
        'peak_period_s': period_s,
        'peak_amp_nT': peak_amp,
        'fap': fap,
        'snr': peak_amp / np.median(sur_max),
        'band': classify_ulf_band(period_s),
        'is_pc5': bool(150.0 <= period_s <= 600.0),
    }


def scan_windows_for_ulf(df, t0, t1,
                         components=('b_radial', 'b_azimuthal', 'b_parallel'),
                         win_min=45, step_min=15, dt_sec=60.0,
                         band=DEFAULT_BAND_HZ, N_sim=2000):
    """
    Slide a win_min window (step_min stride) over [t0, t1]. For each window run the
    detector on every component and keep the most significant (lowest FAP).

    `df` must have 'time', 'mlt', and the component columns at 1-min cadence.
    Returns one row per window (best component), or an empty DataFrame.
    """
    df = df.sort_values('time').reset_index(drop=True)
    win = pd.Timedelta(minutes=win_min)
    step = pd.Timedelta(minutes=step_min)
    min_pts = int(0.8 * win_min)

    rows = []
    ws = pd.Timestamp(t0)
    t1 = pd.Timestamp(t1)
    while ws + win <= t1:
        we = ws + win
        sl = df[(df['time'] >= ws) & (df['time'] < we)]
        if len(sl) >= min_pts:
            best = None
            for comp in components:
                if comp not in sl.columns:
                    continue
                sig = sl[comp].to_numpy(dtype=float)
                if not np.all(np.isfinite(sig)):                       # patch tiny gaps
                    sig = pd.Series(sig).interpolate(limit=3, limit_direction='both').to_numpy()
                if not np.all(np.isfinite(sig)):
                    continue
                res = dominant_peak_significance(sig, dt_sec=dt_sec, band=band, N_sim=N_sim)
                if res is None:
                    continue
                res['component'] = comp
                if best is None or res['fap'] < best['fap']:
                    best = res
            if best is not None:
                best['win_start'] = ws
                best['win_end'] = we
                best['win_center'] = ws + win / 2
                best['mlt'] = float(sl['mlt'].iloc[len(sl) // 2])
                rows.append(best)
        ws += step
    return pd.DataFrame(rows)


def detect_shock_triggered_ulf(df_mfac, tA, pre_hours=1.0, post_hours=12.0,
                               win_min=45, step_min=15, dt_sec=60.0,
                               band=DEFAULT_BAND_HZ, N_sim=2000,
                               fap_threshold=0.05, enhancement=1.0):
    """
    Detect shock-triggered wave activity around an arrival time tA.

    Scans a pre-shock baseline (tA - pre_hours, tA) and a post-shock search window
    (tA, tA + post_hours). A post-shock window counts as a detection if it is
    statistically significant (fap < fap_threshold) AND its peak amplitude exceeds
    `enhancement` x the pre-shock baseline amplitude.

    `df_mfac` must already span [tA - pre_hours, tA + post_hours] (load and
    concatenate day N / N+1 upstream so the window can cross midnight).

    Returns (detections_df, summary_dict, all_windows_df).
    """
    tA = pd.Timestamp(tA)

    base = scan_windows_for_ulf(df_mfac, tA - pd.Timedelta(hours=pre_hours), tA,
                                win_min=win_min, step_min=step_min, dt_sec=dt_sec,
                                band=band, N_sim=N_sim)
    post = scan_windows_for_ulf(df_mfac, tA, tA + pd.Timedelta(hours=post_hours),
                                win_min=win_min, step_min=step_min, dt_sec=dt_sec,
                                band=band, N_sim=N_sim)
    if not base.empty:
        base['segment'] = 'pre'
    if not post.empty:
        post['segment'] = 'post'

    # Baseline = ambient wave level just before arrival. If the pre-shock window has
    # a significant peak, use its amplitude; otherwise use the 95th-pct noise level.
    if not base.empty:
        sig_base = base[base['fap'] < fap_threshold]
        base_amp = (sig_base['peak_amp_nT'].max() if not sig_base.empty
                    else base['peak_amp_nT'].quantile(0.95))
    else:
        base_amp = 0.0

    if not post.empty:
        post['exceeds_baseline'] = post['peak_amp_nT'] > enhancement * base_amp
        post['is_detection'] = (post['fap'] < fap_threshold) & post['exceeds_baseline']
        detections = post[post['is_detection']].copy()
    else:
        detections = pd.DataFrame()

    summary = {
        'tA': tA,
        'baseline_amp_nT': float(base_amp),
        'n_post_significant': int((post['fap'] < fap_threshold).sum()) if not post.empty else 0,
        'n_detections': int(detections.shape[0]),
    }
    if not detections.empty:
        first = detections.sort_values('win_center').iloc[0]
        strongest = detections.sort_values('fap').iloc[0]
        summary.update({
            'onset_minutes': (first['win_center'] - tA).total_seconds() / 60.0,
            'onset_mlt': first['mlt'],
            'best_period_s': strongest['peak_period_s'],
            'best_band': strongest['band'],
            'best_is_pc5': strongest['is_pc5'],
            'best_component': strongest['component'],
            'best_fap': strongest['fap'],
            'best_amp_nT': strongest['peak_amp_nT'],
            'best_mlt': strongest['mlt'],
        })

    all_windows = pd.concat([d for d in (base, post) if not d.empty], ignore_index=True)
    return detections, summary, all_windows
