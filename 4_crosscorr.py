import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
from scipy.signal import detrend
from helper import cut_time_series
from scipy.signal import correlate, correlation_lags

def find_lag_xcorr(l1_b, goes_b, est_time, search_before=30, search_after=30):
    l1_b = l1_b.dropna()
    goes_b = goes_b.dropna()

    if len(l1_b) < 10 or len(goes_b) < 10:
        return None, None, None, None

    time_offset_min = (goes_b.index[0] - l1_b.index[0]).total_seconds() / 60

    l1_proc   = detrend(l1_b.dropna().values)
    goes_proc = detrend(goes_b.dropna().values)
    l1_proc   = (l1_proc - l1_proc.mean()) / (l1_proc.std() + 1e-10)
    goes_proc = (goes_proc - goes_proc.mean()) / (goes_proc.std() + 1e-10)

    corr = correlate(goes_proc, l1_proc, mode='full')
    sample_lags = correlation_lags(len(goes_proc), len(l1_proc), mode='full')
    physical_lags = sample_lags + time_offset_min

    mask = (physical_lags >= est_time - search_before) & \
           (physical_lags <= est_time + search_after)

    if not mask.any():
        return None, physical_lags, corr, mask

    corr_norm = corr / np.sqrt(np.sum(l1_proc**2) * np.sum(goes_proc**2))
    best_lag = int(physical_lags[mask][corr[mask].argmax()])

    return best_lag, physical_lags, corr_norm, mask

###################################### Prep data ######################################

goes_impacts = joblib.load('goes_impacts.pkl')
all_l1_data = joblib.load('all_data_np_fp.pkl')
all_l1_data['ace_b']  = np.sqrt(all_l1_data['ace_bx']**2  + all_l1_data['ace_by']**2  + all_l1_data['ace_bz']**2)
all_l1_data['wind_b'] = np.sqrt(all_l1_data['wind_bx']**2 + all_l1_data['wind_by']**2 + all_l1_data['wind_bz']**2)
all_l1_data['dsc_b']  = np.sqrt(all_l1_data['dsc_bx']**2  + all_l1_data['dsc_by']**2  + all_l1_data['dsc_bz']**2)

###################################### Iterate ######################################

plane_arrival_times = []
xcorr_arrival_times = []
xcorr_corrs = []
xcorr_arrivals = []

for i in range(len(goes_impacts)):
    goes_impact = goes_impacts.iloc[i]

    # # for paper plotting - remove for data processing
    # if goes_impact['l1_departure_time'].month != 7:
    #     continue

    # L1 data
    l1_data = cut_time_series(all_l1_data,
        goes_impact['l1_ta'] - timedelta(minutes=30),
        goes_impact['l1_tc'] + timedelta(minutes=10))

    # GOES data
    year     = goes_impact['l1_departure_time'].year

    goes_sat = goes_impact['goes_num']
    all_goes_data = joblib.load('goes_annual_combined/{}_g{}_calc.pkl'.format(year, goes_sat))
    all_goes_data['goes_b'] = np.sqrt(all_goes_data['Bx']**2 + all_goes_data['By']**2 + all_goes_data['Bz']**2)
    goes_data = cut_time_series(all_goes_data,
        goes_impact['l1_departure_time'] - timedelta(minutes=len(l1_data)),
        goes_impact['pred_goes_arrival'] + timedelta(minutes=60))

    # print('L1 range:', min(l1_data.index), max(l1_data.index))
    # print('GOES range:', min(goes_data.index), max(goes_data.index))

    est_time = int((goes_impact['pred_goes_arrival'] - goes_impact['l1_departure_time']).total_seconds() / 60)

    if est_time < 0:
        plane_arrival_times.append(np.nan)
        xcorr_arrival_times.append(np.nan)
        xcorr_corrs.append(np.nan)
        xcorr_arrivals.append(np.nan)
        continue

    plane_arrival_times.append(est_time)

    # Plot for FIGURE S1 of manuscript
    # fig, ax = plt.subplots(2, 1, figsize=(10,7))
    # ax[0].plot(l1_data[f'{goes_impact['sat_a']}_b'],  label='Sat A', color='blue')
    # ax[0].plot(l1_data[f'{goes_impact['sat_b']}_b'],  label='Sat B', color='green')
    # ax[0].plot(l1_data[f'{goes_impact['sat_c']}_b'],  label='Sat C', color='orange')
    # ax[0].axvline(x=goes_impact['l1_ta'], ls='--', color='blue',   label='Arrival A')
    # ax[0].axvline(x=goes_impact['l1_tb'], ls='--', color='green',  label='Arrival B')
    # ax[0].axvline(x=goes_impact['l1_tc'], ls='--', color='orange', label='Arrival C')
    # ax[0].set_ylabel('L1 B-field (nT)')
    # ax[0].set_title('Raw B-field measurements')
    # ax[0].set_xlabel('Time (UTC)')
    # ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # ax[0].legend(loc='upper left')
    # ax2 = ax[0].twinx()
    # ax2.plot(goes_data['goes_b'][len(l1_data):], label='GOES', color='red')
    # ax2.set_ylabel('GOES B-field (nT)', color='tab:red')
    # ax2.tick_params('y', colors='tab:red')
    # ax2.axvline(x=goes_impact['pred_goes_arrival'], ls='--', color='pink', label='Plane wave predicted arrival')
    # ax2.legend(loc='upper right')

    l1_b = l1_data[f'{goes_impact["sat_c"]}_b'].dropna()
    goes_b = goes_data['goes_b'].dropna()

    best_lag, lags, corr_norm, mask = find_lag_xcorr(
        l1_b, goes_b, est_time,
        search_before=min(30, est_time),  # don't search negative lags
        search_after=30
    )
    xcorr_arrival_times.append(best_lag)
    xcorr_corrs.append(max(corr_norm[mask]) if corr_norm is not None else np.nan)
    xcorr_arrivals.append(goes_impact['l1_tc'] + timedelta(minutes=int(best_lag)) if corr_norm is not None else np.nan)

    # Plot for FIGURE S1 of manuscript... continued
    # Convert lags (minutes) to real timestamps, anchored the same way est_time was
    # computed (relative to l1_departure_time), so the bottom plot's time axis lines
    # up exactly with the top plot's time axis.
    # lag_times = pd.to_datetime([goes_impact['l1_departure_time'] + timedelta(minutes=float(l)) for l in lags])
    # mask_times = lag_times[mask]
    # est_time_dt = goes_impact['l1_departure_time'] + timedelta(minutes=est_time)
    # best_lag_dt = goes_impact['l1_departure_time'] + timedelta(minutes=int(best_lag))
    #
    # ax[1].plot(lag_times, corr_norm, alpha=0.4, color='steelblue', label='Raw xcorr')
    # ax[1].plot(mask_times, corr_norm[mask], color='orange', label='Search window')
    # ax[1].axvline(x=est_time_dt, ls='--', color='pink', label=f'Predicted: {est_time} min')
    # ax[1].axvline(x=best_lag_dt, ls='--', lw=2.5, color='red', label=f'Xcorr peak: {best_lag} min')
    # ax[0].axvline(x=best_lag_dt, ls='--', lw=2.5, label='XCorr lag', color='red')
    # ax[1].set_xlabel('Lag (minutes)')
    # ax[1].set_ylabel('Cross-correlation norm')
    # ax[1].set_title(f'XCorr - Last L1 satellite & GOES')
    #
    # # Keep pixel-for-pixel alignment with the top (datetime) axis, but relabel the
    # # ticks as lag in minutes relative to l1_departure_time instead of clock time.
    # ax[1].set_xlim(ax[0].get_xlim())
    # ax[1].set_xticks(ax[0].get_xticks())
    # l1_dep_num = mdates.date2num(goes_impact['l1_departure_time'])
    # ax[1].xaxis.set_major_formatter(
    #     FuncFormatter(lambda x, pos: f'{(x - l1_dep_num) * 24 * 60:.0f}')
    # )
    # fig.suptitle(f'Magnetosphere Shock Impact: {best_lag_dt.date()}', weight="bold")
    # ax[1].legend()
    # plt.tight_layout()
    #
    # # plt.savefig('paper figures/xcorr.png', dpi=300, format='png', bbox_inches='tight')
    #
    # plt.show()

    # break # for paper plotting - remove for data processing

goes_impacts['plane_travel_time'] = plane_arrival_times
goes_impacts['xcorr_travel_time'] = xcorr_arrival_times
goes_impacts['xcorr_corr'] = xcorr_corrs
goes_impacts['xcorr_arrival'] = xcorr_arrivals
joblib.dump(goes_impacts, 'goes_impacts_xcorr_sector.pkl')