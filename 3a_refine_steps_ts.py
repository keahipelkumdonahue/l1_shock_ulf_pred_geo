import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import curve_fit, OptimizeWarning
import sklearn
from geopack import geopack as gp

def cut_time_series(data, start, end):
    left_idx = data.index.searchsorted(start)
    right_idx = data.index.searchsorted(end)
    return data.iloc[left_idx:right_idx]

def logistic(x, a, b, c, d):
    # a is amplitude
    # b is vertical shift
    # c is growth rate
    # d is x value of midpoint
    return (a / (1+np.exp(c*(x-d))))+b

def log_reg_fit(data, plot):
    x = np.arange(0, len(data))
    y = data.to_numpy()

    a0 = abs(max(y) - min(y))
    b0 = min(y)
    c0 = 1
    d0 = len(x) / 2
    params, cov = curve_fit(f=logistic, xdata=x, ydata=y, p0=[a0, b0, c0, d0])

    ypred = logistic(x, *params)
    r2 = sklearn.metrics.r2_score(y, ypred)

    amp = params[0]
    vert = params[1]
    rate = params[2]
    midpoint = params[3]

    if plot:
        plt.title('R2: {}, midpoint: {}'.format(round(r2, 2), round(midpoint,2)))
        plt.plot(x, y)
        plt.plot(x, ypred)
        plt.show()

    return r2, amp, vert, rate, midpoint

def circular_median_deg(angles_deg):
    """
    Robust circular median of a set of angles (degrees, any wrap).
    """
    angles_rad = np.deg2rad(angles_deg)
    x = np.median(np.cos(angles_rad))
    y = np.median(np.sin(angles_rad))
    return np.degrees(np.arctan2(y, x)) % 360

def imf_clock_angle_gsm(bx_gse, by_gse, bz_gse, epoch):
    """
    Median IMF clock angle (deg, [0, 360)) over a B-field window.
    """
    bx = np.asarray(bx_gse, dtype=float)
    by = np.asarray(by_gse, dtype=float)
    bz = np.asarray(bz_gse, dtype=float)

    good = np.isfinite(bx) & np.isfinite(by) & np.isfinite(bz)
    if not np.any(good):
        return np.nan

    ut = (epoch - datetime(1970, 1, 1)).total_seconds()
    gp.recalc(ut)
    _, by_gsm, bz_gsm = gp.gsmgse(bx[good], by[good], bz[good], -1)  # j=-1: GSE -> GSM

    clock_deg = np.degrees(np.arctan2(by_gsm, bz_gsm)) % 360
    return circular_median_deg(clock_deg)

all_cases = joblib.load('vlim_all_cases.pkl')
all_data = joblib.load('all_data_np_fp.pkl')

ace_b = np.sqrt(all_data['ace_bx'] ** 2 + all_data['ace_by'] ** 2 + all_data['ace_bz'] ** 2)
wind_b = np.sqrt(all_data['wind_bx'] ** 2 + all_data['wind_by'] ** 2 + all_data['wind_bz'] ** 2)
dsc_b = np.sqrt(all_data['dsc_bx'] ** 2 + all_data['dsc_by'] ** 2 + all_data['dsc_bz'] ** 2)
ace_v = np.sqrt(all_data['ace_vx'] ** 2 + all_data['ace_vy'] ** 2 + all_data['ace_vz'] ** 2)
wind_v = np.sqrt(all_data['wind_vx'] ** 2 + all_data['wind_vy'] ** 2 + all_data['wind_vz'] ** 2)
dsc_v = np.sqrt(all_data['dsc_vx'] ** 2 + all_data['dsc_vy'] ** 2 + all_data['dsc_vz'] ** 2)

refined_cases = pd.DataFrame(columns = ['time', 'window', 'direction',
                                        'sat_a', 'sat_b', 'sat_c', 'avg_r2',
                                        't_a', 'lag_ab', 'lag_ac',
                                        'pre_b', 'delta_b', 'delta_b_cv', 'pre_std_b',
                                        'pre_v', 'delta_v', 'delta_v_cv',
                                        'pre_np', 'delta_np',
                                        'pre_fp', 'delta_fp',
                                        'fit_amp', 'fit_vert', 'fit_rate',
                                        'imf_clock'])

trim_margin = 20        # shock-detection window (+/- min) -- do not change lightly, feeds the fit/lag/delta features
TRIM_LEN = 2 * trim_margin

ts_margin = 20           # <-- independent knob: minutes of ts data saved on each side of arrival (try 5-40)
TS_LEN = 2 * ts_margin

ts_times = []
ts_raw = []
ts_znorm = []

ts_times_v = []
ts_raw_v = []
ts_znorm_v = []

ts_times_fp = []
ts_raw_fp = []
ts_znorm_fp = []

for i in range(len(all_cases)):
    l1_shock = all_cases.iloc[i]

    t_a = l1_shock['t_a']
    t_b = l1_shock['t_a'] + l1_shock['lag_ab']
    t_c = l1_shock['t_a'] + l1_shock['lag_ac']

    start = l1_shock['time']
    start_exp = start - timedelta(minutes=120) # pull extra leading data for pre-shock stats
    end = start + timedelta(minutes=int(120 + l1_shock['window'])) # end padding to ensure data lengths consistent

    ace_b_window = cut_time_series(ace_b, start_exp, end)
    wind_b_window = cut_time_series(wind_b, start_exp, end)
    dsc_b_window = cut_time_series(dsc_b, start_exp, end)
    ace_v_window = cut_time_series(ace_v, start_exp, end)
    wind_v_window = cut_time_series(wind_v, start_exp, end)
    dsc_v_window = cut_time_series(dsc_v, start_exp, end)
    wind_np_window = cut_time_series(all_data['wind_np'], start_exp, end)
    wind_fp_window = cut_time_series(all_data['wind_fp'], start_exp, end)

    # GSE B-field components, cut the same way, for the IMF clock angle calc
    bx_windows = {
        'ace': cut_time_series(all_data['ace_bx'], start_exp, end),
        'wind': cut_time_series(all_data['wind_bx'], start_exp, end),
        'dsc': cut_time_series(all_data['dsc_bx'], start_exp, end),
    }
    by_windows = {
        'ace': cut_time_series(all_data['ace_by'], start_exp, end),
        'wind': cut_time_series(all_data['wind_by'], start_exp, end),
        'dsc': cut_time_series(all_data['dsc_by'], start_exp, end),
    }
    bz_windows = {
        'ace': cut_time_series(all_data['ace_bz'], start_exp, end),
        'wind': cut_time_series(all_data['wind_bz'], start_exp, end),
        'dsc': cut_time_series(all_data['dsc_bz'], start_exp, end),
    }
    sat_c_bx_window = bx_windows[l1_shock['sat_c']]
    sat_c_by_window = by_windows[l1_shock['sat_c']]
    sat_c_bz_window = bz_windows[l1_shock['sat_c']]

    wind_loc = 'a'

    if l1_shock['sat_a'] == 'ace':
        sat_a_window = ace_b_window
        sat_a_v_window = ace_v_window
        if l1_shock['sat_b'] == 'wind':
            sat_b_window = wind_b_window
            sat_b_v_window = wind_v_window
            t_wind = t_b
            wind_loc = 'b'

            sat_c_window = dsc_b_window
            sat_c_v_window = dsc_v_window
        elif l1_shock['sat_b'] == 'dsc':
            sat_b_window = dsc_b_window
            sat_b_v_window = dsc_v_window

            sat_c_window = wind_b_window
            sat_c_v_window = wind_v_window
            t_wind = t_c
            wind_loc = 'c'

    elif l1_shock['sat_a'] == 'wind':
        sat_a_window = wind_b_window
        sat_a_v_window = wind_v_window
        t_wind = t_a
        wind_loc = 'a'

        if l1_shock['sat_b'] == 'ace':
            sat_b_window = ace_b_window
            sat_b_v_window = ace_v_window

            sat_c_window = dsc_b_window
            sat_c_v_window = dsc_v_window
        elif l1_shock['sat_b'] == 'dsc':
            sat_b_window = dsc_b_window
            sat_b_v_window = dsc_v_window

            sat_c_window = ace_b_window
            sat_c_v_window = ace_v_window
    else:
        sat_a_window = dsc_b_window
        sat_a_v_window = dsc_v_window

        if l1_shock['sat_b'] == 'ace':
            sat_b_window = ace_b_window
            sat_b_v_window = ace_v_window

            sat_c_window = wind_b_window
            sat_c_v_window = wind_v_window
            t_wind = t_c
            wind_loc = 'c'
        else:
            sat_b_window = wind_b_window
            sat_b_v_window = wind_v_window
            t_wind = t_b
            wind_loc = 'b'

            sat_c_window = ace_b_window
            sat_c_v_window = ace_v_window


    sat_a_trimmed = sat_a_window.iloc[t_a + 120 - trim_margin: t_a + 120 + trim_margin] # all t_a have 120 added to account for longer pre
    sat_b_trimmed = sat_b_window.iloc[t_b + 120 - trim_margin: t_b + 120 + trim_margin]
    sat_c_trimmed = sat_c_window.iloc[t_c + 120 - trim_margin: t_c + 120 + trim_margin]

    # same window, GSE B-field components -- feeds the IMF clock angle calc below
    sat_c_bx_trimmed = sat_c_bx_window.iloc[t_c + 120 - trim_margin: t_c + 120 + trim_margin]
    sat_c_by_trimmed = sat_c_by_window.iloc[t_c + 120 - trim_margin: t_c + 120 + trim_margin]
    sat_c_bz_trimmed = sat_c_bz_window.iloc[t_c + 120 - trim_margin: t_c + 120 + trim_margin]

    sat_a_v_trimmed = sat_a_v_window.iloc[t_a + 120 - trim_margin: t_a + 120 + trim_margin]
    sat_b_v_trimmed = sat_b_v_window.iloc[t_b + 120 - trim_margin: t_b + 120 + trim_margin]
    sat_c_v_trimmed = sat_c_v_window.iloc[t_c  + 120- trim_margin: t_c + 120 + trim_margin]

    wind_np_trimmed = wind_np_window.iloc[t_wind + 120 - trim_margin: t_wind + 120 + trim_margin]
    wind_fp_trimmed = wind_fp_window.iloc[t_wind + 120 - trim_margin: t_wind + 120 + trim_margin]

    try:
        c_r2, c_amp, c_vert, c_rate, c_t = log_reg_fit(sat_c_trimmed, False)
        c_t_dt = sat_c_trimmed.index.to_list()[int(c_t)]
    except:
        continue

    # Median IMF clock angle at sat_c over the 40 min (2*trim_margin) B-field
    # window, GSM per the clock-angle convention (see imf_clock_angle_gsm doc).
    imf_clock_angle = imf_clock_angle_gsm(
        sat_c_bx_trimmed.to_numpy(),
        sat_c_by_trimmed.to_numpy(),
        sat_c_bz_trimmed.to_numpy(),
        c_t_dt,
    )

    def fixed_v_logistic(x, d):
        return (c_amp / (1 + np.exp(c_rate * (x - d)))) + c_vert

    def fit_fixed_logistic(data, plot):
        x = np.arange(0, len(data))
        y = data.to_numpy()

        d0 = len(x) / 2

        params, cov = curve_fit(f=fixed_v_logistic, xdata=x, ydata=y, p0=[d0])

        ypred = fixed_v_logistic(x, *params)
        r2 = sklearn.metrics.r2_score(y, ypred)

        midpoint = params[0]

        if plot:
            plt.title('R2: {}, midpoint: {}'.format(round(r2, 2), round(midpoint, 2)))
            plt.plot(x, y)
            plt.plot(x, ypred)
            plt.show()

        return r2, midpoint

    try:
        a_r2, a_t = fit_fixed_logistic(data=sat_a_trimmed, plot=False)
        a_t_dt = sat_a_trimmed.index.to_list()[int(a_t)]

        b_r2, b_t = fit_fixed_logistic(data=sat_b_trimmed, plot=False)
        b_t_dt = sat_b_trimmed.index.to_list()[int(b_t)]
    except:
        continue

    t_a = int((a_t_dt - start).total_seconds() / 60)
    lag_ab = int((b_t_dt - a_t_dt).total_seconds() / 60)
    lag_ac = int((c_t_dt - a_t_dt).total_seconds() / 60)

    # re-check R^2 values
    avg_r2 = (a_r2 + b_r2 + c_r2) / 3
    if avg_r2 < 0.6:
        print('Failed R2 criteria')
        continue

    ############################## Save parameters and refined times ##############################

    pre_b_a = sat_a_window.loc[a_t_dt - timedelta(minutes=10)]
    pre_b_b = sat_b_window.loc[b_t_dt - timedelta(minutes=10)]
    pre_b_c = sat_c_window.loc[c_t_dt - timedelta(minutes=10)]
    pre_b = np.mean([pre_b_a, pre_b_b, pre_b_c])

    pre_std_b_a = np.std(sat_a_window.loc[a_t_dt - timedelta(minutes=30):a_t_dt])
    pre_std_b_b = np.std(sat_b_window.loc[b_t_dt - timedelta(minutes=30):b_t_dt])
    pre_std_b_c = np.std(sat_c_window.loc[c_t_dt - timedelta(minutes=30):c_t_dt])
    pre_std_b = np.mean([pre_std_b_a, pre_std_b_b, pre_std_b_c])

    delta_b_a = np.max(sat_a_trimmed) - np.min(sat_a_trimmed)
    delta_b_b = np.max(sat_b_trimmed) - np.min(sat_b_trimmed)
    delta_b_c = np.max(sat_c_trimmed) - np.min(sat_c_trimmed)
    delta_b = np.mean([delta_b_a, delta_b_b, delta_b_c])
    delta_b_cv = np.std([delta_b_a, delta_b_b, delta_b_c]) / delta_b

    pre_v_a = sat_a_v_window.loc[a_t_dt - timedelta(minutes=10)]
    pre_v_b = sat_b_v_window.loc[b_t_dt - timedelta(minutes=10)]
    pre_v_c = sat_c_v_window.loc[c_t_dt - timedelta(minutes=10)]
    pre_v = np.mean([pre_v_a, pre_v_b, pre_v_c])

    pre_std_v_a = np.std(sat_a_v_window.loc[a_t_dt - timedelta(minutes=30):a_t_dt])
    pre_std_v_b = np.std(sat_b_v_window.loc[b_t_dt - timedelta(minutes=30):b_t_dt])
    pre_std_v_c = np.std(sat_c_v_window.loc[c_t_dt - timedelta(minutes=30):c_t_dt])

    delta_v_a = np.max(sat_a_v_trimmed) - np.min(sat_a_v_trimmed)
    delta_v_b = np.max(sat_b_v_trimmed) - np.min(sat_b_v_trimmed)
    delta_v_c = np.max(sat_c_v_trimmed) - np.min(sat_c_v_trimmed)
    delta_v = np.mean([delta_v_a, delta_v_b, delta_v_c])
    delta_v_cv = np.std([delta_v_a, delta_v_b, delta_v_c]) / delta_v

    if wind_loc == 'a':
        wind_dt = a_t_dt
    elif wind_loc == 'b':
        wind_dt = b_t_dt
    else:
        wind_dt = c_t_dt

    pre_np = wind_np_window.loc[wind_dt - timedelta(minutes=10)]
    delta_np = np.max(wind_np_trimmed) - np.min(wind_np_trimmed)

    pre_fp = wind_fp_window.loc[wind_dt - timedelta(minutes=10)]
    delta_fp = np.max(wind_fp_trimmed) - np.min(wind_fp_trimmed)

    pre_fp_window = wind_fp_window.loc[wind_dt - timedelta(minutes=30): wind_dt]
    pre_std_fp = np.std(pre_fp_window)

    # -----------------------------------------------------------------
    # Cut the ts-save window, independently of the shock-detection window
    # above. This is centered on the *refined* per-satellite arrival times
    # -----------------------------------------------------------------
    sat_a_ts = cut_time_series(sat_a_window, a_t_dt - timedelta(minutes=ts_margin), a_t_dt + timedelta(minutes=ts_margin))
    sat_b_ts = cut_time_series(sat_b_window, b_t_dt - timedelta(minutes=ts_margin), b_t_dt + timedelta(minutes=ts_margin))
    sat_c_ts = cut_time_series(sat_c_window, c_t_dt - timedelta(minutes=ts_margin), c_t_dt + timedelta(minutes=ts_margin))

    sat_a_v_ts = cut_time_series(sat_a_v_window, a_t_dt - timedelta(minutes=ts_margin), a_t_dt + timedelta(minutes=ts_margin))
    sat_b_v_ts = cut_time_series(sat_b_v_window, b_t_dt - timedelta(minutes=ts_margin), b_t_dt + timedelta(minutes=ts_margin))
    sat_c_v_ts = cut_time_series(sat_c_v_window, c_t_dt - timedelta(minutes=ts_margin), c_t_dt + timedelta(minutes=ts_margin))

    wind_fp_ts = cut_time_series(wind_fp_window, wind_dt - timedelta(minutes=ts_margin), wind_dt + timedelta(minutes=ts_margin))

    # -----------------------------------------------------------------
    # Build the 3-channel time-series array for this event.
    # -----------------------------------------------------------------
    raw_stack = np.stack([
        sat_a_ts.to_numpy(),
        sat_b_ts.to_numpy(),
        sat_c_ts.to_numpy(),
    ])  # shape (3, TS_LEN)

    raw_stack_v = np.stack([
        sat_a_v_ts.to_numpy(),
        sat_b_v_ts.to_numpy(),
        sat_c_v_ts.to_numpy(),
    ])

    if raw_stack.shape[1] == TS_LEN and not np.isnan(raw_stack).any():
        # per-channel, per-event z-score using that channel's own
        # pre-shock stats already computed above (pre_b_x / pre_std_b_x)
        chan_means = np.array([pre_b_a, pre_b_b, pre_b_c]).reshape(3, 1)
        chan_stds = np.array([pre_std_b_a, pre_std_b_b, pre_std_b_c]).reshape(3, 1)
        chan_stds[chan_stds == 0] = 1e-6  # guard against div-by-zero on flat pre-shock data

        znorm_stack = (raw_stack - chan_means) / chan_stds

        ts_times.append(c_t_dt)
        ts_raw.append(raw_stack)
        ts_znorm.append(znorm_stack)

    if raw_stack_v.shape[1] == TS_LEN and not np.isnan(raw_stack_v).any():
        # per-channel, per-event z-score using that channel's own
        # pre-shock stats already computed above (pre_b_x / pre_std_b_x)
        chan_means = np.array([pre_v_a, pre_v_b, pre_v_c]).reshape(3, 1)
        chan_stds = np.array([pre_std_v_a, pre_std_v_b, pre_std_v_c]).reshape(3, 1)
        chan_stds[chan_stds == 0] = 1e-6  # guard against div-by-zero on flat pre-shock data

        znorm_stack_v = (raw_stack_v - chan_means) / chan_stds

        ts_times_v.append(c_t_dt)
        ts_raw_v.append(raw_stack_v)
        ts_znorm_v.append(znorm_stack_v)

    raw_stack_fp = np.stack([
        wind_fp_ts.to_numpy(),
    ])  # shape (1, TS_LEN)

    if raw_stack_fp.shape[1] == TS_LEN and not np.isnan(raw_stack_fp).any():
        chan_means_fp = np.array([pre_fp]).reshape(1, 1)
        chan_stds_fp = np.array([pre_std_fp]).reshape(1, 1)
        chan_stds_fp[chan_stds_fp == 0] = 1e-6  # guard against div-by-zero on flat pre-shock data

        znorm_stack_fp = (raw_stack_fp - chan_means_fp) / chan_stds_fp

        ts_times_fp.append(c_t_dt)
        ts_raw_fp.append(raw_stack_fp)
        ts_znorm_fp.append(znorm_stack_fp)

    refined_cases.loc[len(refined_cases)] = [start, l1_shock['window'], l1_shock['direction'],
                                             l1_shock['sat_a'], l1_shock['sat_b'], l1_shock['sat_c'],
                                             avg_r2,
                                             t_a, lag_ab, lag_ac,
                                             pre_b, delta_b, delta_b_cv, pre_std_b,
                                             pre_v, delta_v, delta_v_cv,
                                             pre_np, delta_np,
                                             pre_fp, delta_fp,
                                             c_amp, c_vert, c_rate,
                                             imf_clock_angle]

print(refined_cases)
print(len(refined_cases))

out_dir = f'ts_window_scan/{ts_margin}'

joblib.dump(refined_cases, f'{out_dir}/refined_cases_np_fp.pkl')

ts_raw = np.stack(ts_raw)      # shape (n_events, 3, TS_LEN)
ts_znorm = np.stack(ts_znorm)  # shape (n_events, 3, TS_LEN)

ts_raw_v = np.stack(ts_raw_v)
ts_znorm_v = np.stack(ts_znorm_v)

print(f'\nSaved time-series for {len(ts_times)} / {len(refined_cases)} refined events '
      f'(dropped: incomplete window or NaNs)')
print('ts_raw shape:', ts_raw.shape, ' ts_znorm shape:', ts_znorm.shape)

joblib.dump({
    'time': np.array(ts_times),      # (n_events,) -- merge key, matches refined_cases['time']
    'raw': ts_raw,                   # (n_events, 3, TS_LEN) nT, channel order = (sat_a, sat_b, sat_c)
    'znorm': ts_znorm,               # same shape, per-channel/per-event z-scored
    'channel_order': ['sat_a', 'sat_b', 'sat_c'],
    'trim_margin': trim_margin,      # shock-detection window (+/- min) used to locate arrival times
    'ts_margin': ts_margin,          # ts-save window (+/- min) used to cut the saved series
    'trim_len': TS_LEN,
}, f'{out_dir}/refined_cases_ts_bfield.pkl')

joblib.dump({
    'time': np.array(ts_times_v),      # (n_events,) -- merge key, matches refined_cases['time']
    'raw': ts_raw_v,                   # (n_events, 3, TS_LEN) nT, channel order = (sat_a, sat_b, sat_c)
    'znorm': ts_znorm_v,               # same shape, per-channel/per-event z-scored
    'channel_order': ['sat_a', 'sat_b', 'sat_c'],
    'trim_margin': trim_margin,
    'ts_margin': ts_margin,
    'trim_len': TS_LEN,
}, f'{out_dir}/refined_cases_ts_vfield.pkl')

ts_raw_fp = np.stack(ts_raw_fp)      # shape (n_events, 1, TS_LEN)
ts_znorm_fp = np.stack(ts_znorm_fp)  # shape (n_events, 1, TS_LEN)

print(f'\nSaved fp time-series for {len(ts_times_fp)} / {len(refined_cases)} refined events '
      f'(dropped: incomplete window or NaNs)')
print('ts_raw_fp shape:', ts_raw_fp.shape, ' ts_znorm_fp shape:', ts_znorm_fp.shape)

joblib.dump({
    'time': np.array(ts_times_fp),   # (n_events,) -- merge key, matches refined_cases['time']
    'raw': ts_raw_fp,                # (n_events, 1, TS_LEN), channel order = (wind_fp,)
    'znorm': ts_znorm_fp,            # same shape, per-event z-scored using pre_fp/pre_std_fp
    'channel_order': ['wind_fp'],
    'trim_margin': trim_margin,
    'ts_margin': ts_margin,
    'trim_len': TS_LEN,
}, f'{out_dir}/refined_cases_ts_fp.pkl')
