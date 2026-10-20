import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.signal import detrend

def interpolate(df, resolution, verbose):

    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'time'})
    df = df.drop_duplicates(subset=['time'])  # drop duplicate rows (start or end)
    df.set_index('time', inplace=True)  # set index to time

    # if verbose:
        # print('before interpolation:')
        # print(df)

    df_resampled = df.resample(resolution)
    df_interp = df_resampled.interpolate(method='linear')

    if verbose:
        # print('after interpolation:')
        # print(df_interp)

        plt.plot(df[:100], label='original')
        plt.plot(df_interp[:100], label='interpolated')
        plt.legend()
        plt.show() # plots should be identical

    return df_interp

def cut_time_series(data, start, end):
    data = data.loc[data.index <= end]
    data = data.loc[data.index >= start]

    return data

def LP_filter_data(data, cutoff1, order):
    '''this is a fix filter for this data set only. also only in pandas'''
    fs = (1 / 60)
    nyq = 0.5 * fs

    cutoff_freq1 = (1 / (cutoff1 * 60)) / nyq
    b_high, a_high = signal.butter(order, cutoff_freq1, 'low')
    LOW = signal.filtfilt(b_high, a_high, data)

    return LOW

def MFAC_convert(abgsm):
    """Convert magnetic field from GSM to MFA coordinates."""
    abgsm.index.name = 'time'
    dt = abgsm.index
    n_points = dt.size

    abgsmav = abgsm.resample('30min').mean()
    full_abgsmav = \
        abgsmav.reindex(abgsmav.index.union(abgsm.index)).interpolate(method='index', limit_direction='both').loc[
            abgsm.index]

    output = np.full((n_points, 3), np.nan, dtype=np.double)
    for i in range(n_points):
        bf = full_abgsmav.values[i]
        bxyz = np.sqrt(bf[0] ** 2 + bf[1] ** 2 + bf[2] ** 2)
        bxz = np.sqrt(bf[0] ** 2 + bf[2] ** 2)

        u = bf / bxyz  # Unit vector
        Rg = np.array([bf[0] / bxz, 0.0, bf[2] / bxz])  # Radial unit vector

        H = u
        tmp_vec1 = np.cross(H, Rg)
        Q = np.linalg.norm(tmp_vec1)
        D = tmp_vec1 / Q
        V = np.cross(D, H)

        mat = np.array([V, D, H])
        output[i] = np.dot(mat, abgsm.values[i])

    df_mfac = pd.DataFrame(output, columns=['b_radial', 'b_azimuthal', 'b_parallel'], index=dt)
    df_mfac.reset_index(inplace=True)
    return df_mfac

def hanning_peak_model(x, a, mu, w):
    """Hanning-window-aware sinc^2 model for peak fitting."""
    return a * (np.sinc((x - mu) / w)) ** 2

def _fit_ar1_yw(x):
    """Fit AR(1) model using Yule-Walker for noise simulation."""
    x = np.asarray(x)
    x = x - np.nanmean(x)
    valid = np.isfinite(x)
    x = x[valid]
    if x.size < 20:
        raise RuntimeError("Too short for reliable AR(1)")
    r0 = np.dot(x, x) / x.size
    r1 = np.dot(x[1:], x[:-1]) / (x.size - 1)
    phi = np.clip(r1 / (r0 + 1e-12), -0.99, 0.99)
    sigma2 = (1 - phi ** 2) * r0
    return phi, np.sqrt(max(sigma2, 1e-12))


def _simulate_ar1(phi, sigma, n, N_sim):
    """Simulate AR(1) noise for N_sim realizations."""
    e = np.random.normal(0, sigma, (N_sim, n))
    y = np.zeros_like(e)
    for t in range(1, n):
        y[:, t] = phi * y[:, t - 1] + e[:, t]
    return y

def extract_drift_echo_candidates_dyn_echo(df_input, t1, delta_t,
                                           df_pitch_tel=None, energy_keV_mass_eq=None,
                                           C_d=None, Lshell=None, verbose_midnight=False):
    """
    Extract drift echo or ULF wave candidates from data within time_hours windows.

    Parameters:
    - df_input: DataFrame with 'time_hours' and relevant data columns
    - t1: Starting time_hours hour
    - delta_t: Window size in hours
    - df_pitch_tel: Pitch angle DataFrame for the telescope
    - energy_keV_mass_eq: Energy normalized by rest mass (E/m0c2)
    - C_d: Drift constant (seconds)
    - Lshell: L-shell value
    - verbose_midnight: If True, print when windows cross midnight (hour 24)

    Returns:
    - List of dictionaries or DataFrames with results
    """
    df_input = df_input.sort_values(by='time').reset_index(drop=True)
    reference_time = df_input['time'].min().floor('h')
    df_input['time_hours'] = (df_input['time'] - reference_time) / pd.Timedelta(hours=1)
    overlap_t = 0.25 * delta_t

    # if df_pitch_tel is not None:
    #     df_pitch_tel = df_pitch_tel.copy()  # Create a deep copy
    #     reference_time = df_pitch_tel['time'].min().floor('h')
    #     df_pitch_tel['time_hours'] = (df_pitch_tel['time'] - reference_time) / pd.Timedelta(hours=1)

    all_candidates = []
    t2 = t1 + delta_t

    # Derive max hours from actual data instead of hardcoding 24
    # This allows windows to extend into Day N+1 padding when available
    max_time_hours = df_input['time_hours'].max()

    # IMPORTANT: Only process windows that START on Day N (t1 < 24)
    # Windows can END beyond 24h (into Day N+1 padding), but windows that
    # START on Day N+1 will be processed when Day N+1 is the primary day.
    # This prevents duplicate processing of the same UTC time windows.
    #
    # max_t1 is constrained by TWO factors:
    # 1. Window must START on Day N: t1 < 24 → effectively t1 <= 24 - overlap_t (last step before crossing)
    # 2. Window must have complete data: t1 + delta_t <= max_time_hours → t1 <= max_time_hours - delta_t
    max_t1 = min(24 - overlap_t, max_time_hours - delta_t)

    if verbose_midnight:
        print(f"    [DEBUG] Data spans {max_time_hours:.2f} hours (24h + {max(0, max_time_hours - 24):.2f}h padding)")
        print(f"    [DEBUG] Window will slide until t1 <= {max_t1:.2f}h (last window ends at {max_t1 + delta_t:.2f}h)")

    while t1 <= max_t1:
        df_slice = df_input.loc[(df_input.time_hours >= t1) & (df_input.time_hours < t2)]

        # Print ALL midnight crossing windows for verification
        if verbose_midnight and t2 > 24:
            # Get actual UTC times for the window
            window_start_utc = df_slice['time'].min() if not df_slice.empty else "N/A"
            window_end_utc = df_slice['time'].max() if not df_slice.empty else "N/A"
            status = "EMPTY!" if df_slice.empty else f"{len(df_slice)} pts"
            print(f"    >>> MIDNIGHT CROSSING: Window [{t1:.2f}h - {t2:.2f}h] | "
                  f"UTC: {window_start_utc} to {window_end_utc} | {status}")

        # Skip if df_slice is empty
        if df_slice.empty:
            print(f"Warning: Empty data slice for MLT window {t1}-{t2}. Skipping this window.")
            t1 += overlap_t
            t2 += overlap_t
            continue

        # if df_pitch_tel is not None:
        #     pitch_slice = df_pitch_tel.loc[
        #         (df_pitch_tel['time_hours'] >= df_slice['time_hours'].min()) &
        #         (df_pitch_tel['time_hours'] <= df_slice['time_hours'].max())
        #         ]
        #
        #     if pitch_slice.iloc[:, 1].isna().sum() != pitch_slice.iloc[:, 1].shape[0]:
        #         alpha_eq_median_rad = np.deg2rad(pitch_slice.iloc[:, 1].median())  # safe for unnamed telescope column
        #
        #         drift_period = C_d / Lshell * (
        #                     (energy_keV_mass_eq + 1) / (energy_keV_mass_eq * (energy_keV_mass_eq + 2))) * \
        #                        (1 - 0.3333 * np.sin(alpha_eq_median_rad) ** 0.62)
        #     else:
        #         # print(f"Warning: All pitch angles are NaN for time window {t1}-{t2}. Setting drift_period to NaN.")
        #         alpha_eq_median_rad = np.nan
        #         drift_period = np.nan

        # Linear interpolation for gaps <=3 minutes
        col_to_check = 'b_component'
        df_slice.loc[:, col_to_check] = df_slice[col_to_check].interpolate(method='linear', limit=3,
                                                                           limit_direction='both')

        # sigma_col = 'av_diff_flux_sigma_hp' if 'av_diff_flux_sigma_hp' in df_slice.columns else 'av_diff_flux_sigma'
        # if sigma_col in df_slice.columns:
        #     df_slice.loc[:, sigma_col] = df_slice[sigma_col].interpolate(method='linear', limit=3,
        #                                                                  limit_direction='both')
        # Check for NaNs in relevant columns based on analysis type
        skip_window = False
        # Check for NaNs in av_diff_flux (or av_diff_flux_hp if filtered)
        col_to_check = 'b_component'
        if df_slice[col_to_check].isna().any():
            skip_window = True
            print(f"Warning: NaN values found in {col_to_check} for UTC window {t1}-{t2}. Skipping this window.")

        if skip_window:
            t1 += overlap_t
            t2 += overlap_t
            continue

        # Proceed with processing if no issues
        # pitch angle alpha_eq_median_rad

        df_slice = df_slice.copy()
        df_slice['b_component'] = detrend(df_slice['b_component'].values)  # removes mean and linear trend

        df_echo = echo_CLEAN_function_new(df_slice)
        df_echo['t1_mlt'] = df_slice.mlt.iloc[0]
        df_echo['t2_mlt'] = df_slice.mlt.iloc[-1]
        df_echo['date1_utc'] = df_slice.time.iloc[0]
        df_echo['date2_utc'] = df_slice.time.iloc[-1]
        # df_echo['median_pitch_angle_deg'] = np.rad2deg(alpha_eq_median_rad)
        # df_echo['expected_drift_per'] = drift_period

        # df_echo["energy_keV_from_period"] = df_echo["period"].apply(
        #     lambda p: energy_from_drift_period_walt(
        #         drift_period_sec=p * 60.0,
        #         Lshell=Lshell,
        #         alpha_deg=np.rad2deg(alpha_eq_median_rad),
        #         C_d=C_d
        #     )
        #     if pd.notna(p) and p > 0 and not np.isnan(alpha_eq_median_rad)
        #     else np.nan
        # )
        all_candidates.append(df_echo)

        t1 += overlap_t
        t2 += overlap_t

    if verbose_midnight:
        n_total = len(all_candidates)
        # Filter out empty DataFrames before counting midnight crossings
        non_empty = [c for c in all_candidates if not c.empty]
        n_midnight = sum(1 for c in non_empty if
                         c['date2_utc'].iloc[0].date() > c['date1_utc'].iloc[0].date())
        print(
            f"    [DEBUG] Total windows processed: {n_total} ({len(non_empty)} non-empty), Midnight crossings: {n_midnight}")
        if non_empty:
            first_window = non_empty[0]
            last_window = non_empty[-1]
            print(
                f"    [DEBUG] First window: {first_window['date1_utc'].iloc[0]} to {first_window['date2_utc'].iloc[0]}")
            print(f"    [DEBUG] Last window:  {last_window['date1_utc'].iloc[0]} to {last_window['date2_utc'].iloc[0]}")

    return all_candidates

def echo_CLEAN_function_new(data_frame, full_length=1440, N_sim=5000, max_iterations=50, sig_N_threshold=0.1):
    """
    Optimized CLEAN algorithm for drift echo detection with time-domain subtraction.

    Parameters:
    - data_frame: DataFrame with the 120-minute window data, containing 'av_diff_flux' or 'av_diff_flux_hp' and 'time_hours'.
    - full_length: Length of the padded dataset in minutes (default 1440 minutes).
    - N_sim: Number of noise simulations for significance testing (default 5000).
    - max_iterations: Maximum number of CLEAN iterations (default 50).
    - sig_N_threshold: Significance threshold for peak acceptance (default 0.1).
    - plot: If True, plot the power spectrum and detected peaks.

    Returns:
    - DataFrame with columns ['freq', 'amp', 'phase', 'period'] for detected peaks (period in minutes).
    """
    df = pd.DataFrame(columns=['freq', 'amp', 'phase'])
    CG = 0.5  # Hanning window coherent gain
    rejected_peaks = []  # Track rejected frequencies to avoid loops

    # Original signal size (e.g., 120 minutes)
    signal_size = data_frame.shape[0]
    data_key = 'b_component'
    signal_data = data_frame[data_key].values.copy()  # residual
    time_hours = data_frame['time_hours'].values
    time_seconds = time_hours * 3600
    time_step = np.mean(np.diff(time_seconds))
    initial_noise_std = np.nanstd(signal_data)
    freq_resolution = 1 / (signal_size * time_step)

    i = 0
    while i < max_iterations:
        # Compute FFT
        hanning_window = np.hanning(signal_size)
        windowed_signal = hanning_window * signal_data
        padded_size = full_length
        pad_left = (padded_size - signal_size) // 2
        pad_right = padded_size - signal_size - pad_left
        padded_signal = np.pad(windowed_signal, (pad_left, pad_right), mode='constant', constant_values=0)
        freq_padded = np.fft.rfftfreq(padded_size, time_step)
        power = np.fft.rfft(padded_signal)
        power_abs = np.abs(power)
        power_nT = (2.0 / (signal_size * CG)) * power_abs
        phase = np.angle(power)

        # Simulate noise
        noise_mean = np.nanmean(signal_data)
        if i == 0:
            try:
                phi, sigma = _fit_ar1_yw(signal_data)
                noise_samples = _simulate_ar1(phi, sigma, signal_size, N_sim)
            except:
                noise_std = min(np.nanstd(signal_data), initial_noise_std * 2)
                noise_samples = np.random.normal(0, noise_std, (N_sim, signal_size))
        else:
            try:
                phi, sigma = _fit_ar1_yw(signal_data)
                noise_samples = _simulate_ar1(phi, sigma, signal_size, N_sim)
            except RuntimeError:
                noise_std = np.nanstd(signal_data)
                noise_samples = np.random.normal(0, noise_std, (N_sim, signal_size))

        windowed_noise = noise_samples * hanning_window[None, :]
        padded_noise = np.pad(windowed_noise, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
        noise_power = np.fft.rfft(padded_noise, axis=1)
        noise_abs = np.abs(noise_power)
        av_power_noise = ((2.0 / (signal_size * CG)) * noise_abs).T

        # Find peaks
        noise_med = np.median(av_power_noise, axis=1)
        noise_std = np.std(av_power_noise, axis=1)
        peaks, _ = find_peaks(power_nT, height=noise_med + 4 * noise_std)
        if peaks.size == 0:
            # print(f"No peaks above threshold at iteration {i}")
            break

        max_ind = peaks[np.argmax(power_nT[peaks])]
        max_amp = power_nT[max_ind]
        freq_max_amp = freq_padded[max_ind]
        phase_max = phase[max_ind]

        # print(f"Iteration {i}: Initial peak at {freq_max_amp:.6f} Hz ({(1 / freq_max_amp) / 60 if freq_max_amp > 0 else 0:.1f} min), amp={max_amp:.3f}")

        # Fit peak frequency
        fit_successful = False
        popt = None
        max_window = 50
        min_window = 5
        for window in range(max_window, min_window - 1, -1):
            left = max(0, max_ind - window)
            right = min(len(freq_padded), max_ind + window + 1)
            x = freq_padded[left:right]
            y = power_nT[left:right]
            if len(x) < 5:
                continue
            amp_guess = np.max(y)
            center_guess = freq_padded[max_ind]
            width_guess = 2 / (signal_size * time_step)
            p0 = [amp_guess, center_guess, width_guess]
            try:
                popt, _ = curve_fit(hanning_peak_model, x, y, p0=p0,
                                    bounds=([0, freq_padded[0], 0], [np.inf, freq_padded[-1], np.inf]))
                freq_max_amp = popt[1]
                max_amp = popt[0]
                fit_successful = True
                break
            except (RuntimeError, ValueError):
                continue

        # Theoretical Hanning fallback with interpolation
        if not fit_successful:
            center = freq_padded[max_ind]
            main_lobe_width = 2 / (signal_size * time_step)
            amp = power_nT[max_ind]
            if max_ind > 0 and max_ind < len(power_nT) - 1:
                x = freq_padded[max_ind - 1:max_ind + 2]
                y = power_nT[max_ind - 1:max_ind + 2]
                p = np.polyfit(x, y, 2)
                center = -p[1] / (2 * p[0])
            freq_max_amp = center
            max_amp = amp

        # Significance test
        closest_ind = np.argmin(np.abs(freq_padded - freq_max_amp))
        sig_num = np.sum(av_power_noise[closest_ind] > max_amp)

        sig_N = sig_num / N_sim

        if sig_N > sig_N_threshold:
            # print(f"Peak at {freq_max_amp:.6f} Hz rejected (sig_N={sig_N:.3f} > {sig_N_threshold})")
            rejected_peaks.append(freq_max_amp)
            i += 1
            continue

        # Check for near-duplicate peaks
        if not df.empty:
            freq_diff = np.abs(df['freq'] - freq_max_amp)
            if (freq_diff < freq_resolution).any() or any(
                    abs(f - freq_max_amp) < freq_resolution for f in rejected_peaks):
                # print(f"Peak at {freq_max_amp:.6f} Hz too close to existing or rejected peak")
                rejected_peaks.append(freq_max_amp)
                i += 1
                continue

        # Fit amplitude and phase in time domain
        freq = freq_max_amp
        cos_part = np.cos(2 * np.pi * freq * time_seconds)
        sin_part = np.sin(2 * np.pi * freq * time_seconds)
        A = np.vstack([cos_part, sin_part]).T
        params, _, _, _ = np.linalg.lstsq(A, signal_data, rcond=None)
        c, d = params
        amp_fitted = np.sqrt(c ** 2 + d ** 2)
        phase_fitted = np.arctan2(-d, c)

        # Update df
        df.loc[i, 'freq'] = freq
        df.loc[i, 'amp'] = amp_fitted
        df.loc[i, 'phase'] = phase_fitted

        # Subtract from residual
        component = amp_fitted * np.cos(2 * np.pi * freq * time_seconds + phase_fitted)
        signal_data -= component
        i += 1

    # Filter periods and add period column
    df = df.loc[df['freq'] > 1 / (60 * signal_size / 2)].reset_index(drop=True)
    if not df.empty:
        df['period'] = (1 / df['freq']) / 60
    else:
        df = pd.DataFrame(columns=['freq', 'amp', 'phase', 'period'])

    return df
