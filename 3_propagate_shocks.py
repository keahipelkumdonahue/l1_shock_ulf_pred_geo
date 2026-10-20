import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import glob
import math

# import shock list from Pelkum Donahue & Inceoglu, 2026 at doi.org/10.5281/zenodo.20315800
l1_shocks = joblib.load('refined_corrected_cases_np_fp.pkl')

###################################### Set up L1 list for departure times ######################################

ta = []
tb = []
tc = []

for i in range(len(l1_shocks)):
    shock = l1_shocks.iloc[i]

    t_a = shock['start'] + timedelta(minutes=int(shock['t_a']))
    t_b = t_a + timedelta(minutes=int(shock['lag_ab']))
    t_c = t_a + timedelta(minutes=int(shock['lag_ac']))

    ta.append(t_a)
    tb.append(t_b)
    tc.append(t_c)

l1_shocks['l1_a'] = ta
l1_shocks['l1_b'] = tb
l1_shocks['l1_c'] = tc

print(l1_shocks.keys())
# ['start', 'tot_window', 'direction', 'sat_a', 'sat_b', 'sat_c', 'avg_r2', 't_a',
#        'lag_ab', 'lag_ac', 'pos_a_x', 'pos_a_y', 'pos_a_z', 'pos_b_x',
#        'pos_b_y', 'pos_b_z', 'pos_c_x', 'pos_c_y', 'pos_c_z', 'orb_norm_x',
#        'orb_norm_y', 'orb_norm_z', 'v_calc_x', 'v_calc_y', 'v_calc_z', 'pre_b',
#        'delta_b', 'pre_v', 'delta_v', 'pre_np', 'delta_np', 'pre_fp',
#        'delta_fp', 'v_corrected', 'l1_a', 'l1_b', 'l1_c']

##################################### Iterative plane wave propagation ######################################

goes_data_files = ['2017_g13_calc.pkl', '2017_g15_calc.pkl', '2018_g16_calc.pkl', '2018_g17_calc.pkl',
                   '2019_g16_calc.pkl', '2019_g17_calc.pkl', '2020_g16_calc.pkl', '2020_g17_calc.pkl',
                   '2021_g16_calc.pkl', '2021_g17_calc.pkl',
                   '2022_g16_calc.pkl', '2022_g17_calc.pkl', '2022_g18_calc.pkl',
                   '2023_g16_calc.pkl', '2023_g18_calc.pkl']

goes_impacts = pd.DataFrame(columns=['l1_departure_time', 'direction',
                                     'sat_a', 'sat_b', 'sat_c',
                                     'avg_r2', 'fit_amp', 'fit_vert', 'fit_rate',
                                     'l1_ta', 'l1_tb', 'l1_tc', 'v_corrected', 'imf_clock', 'shock_clock',
                                     'l1_pre_b', 'l1_delta_b', 'l1_delta_b_cv', 'l1_pre_std_b',
                                     'l1_pre_v', 'l1_delta_v', 'l1_delta_v_cv',
                                     'l1_pre_np', 'l1_delta_np',
                                     'l1_pre_fp', 'l1_delta_fp',
                                     'goes_num', 'pred_goes_arrival', 'arrival_goes_mlt', 'mlt_sector'])

for i in range(len(l1_shocks)):

    ############ Shock params ############
    shock = l1_shocks.iloc[i]

    l1_departure_time = shock['l1_c']  # datetime
    l1_v = shock['v_corrected']  # km/s, xyz array of shock velocity vector
    l1_pos = [shock['pos_c_x'], shock['pos_c_y'], shock['pos_c_z']]  # Re
    l1_pos = np.array(l1_pos) * 6378  # km

    ############ GOES params ############

    yr = l1_departure_time.year
    print(yr)

    goes_files = glob.glob(f'goes_annual_combined/{yr}_g*_calc.pkl')
    # print('Accessing GOES files:', goes_files)

    for name in goes_files:

        goes_sat_num = name[27:29]

        goes_data = joblib.load(name)
        goes_pos = goes_data[['x', 'y', 'z']] * 6378

        ############ Plane wave intersection with MLT ############
        dt = (goes_pos.index - l1_departure_time).total_seconds().values
        n = l1_v / np.linalg.norm(l1_v)
        goes_pos['signed_distances'] = (goes_pos[['x', 'y', 'z']].values - l1_pos) @ n - np.linalg.norm(l1_v) * dt

        signs = np.sign(goes_pos['signed_distances'])
        flip = (signs != signs.shift(1)) & signs.shift(1).notna()

        try:
            arrival_time = goes_pos['signed_distances'].index[flip][0]
            # print(f'Shock arrival at GOES: {arrival_time}')
        except:
            # print('No shock arrival at GOES')
            continue

        mlt_at_arrival = goes_data['mlt'].loc[arrival_time]
        # print(f'MLT at shock arrival: {mlt_at_arrival}')

        goes_impacts.loc[len(goes_impacts)] = [l1_departure_time, shock['direction'],
                                               shock['sat_a'], shock['sat_b'], shock['sat_c'],
                                               shock['avg_r2'], shock['fit_amp'], shock['fit_vert'], shock['fit_rate'],
                                               shock['l1_a'], shock['l1_b'], shock['l1_c'], shock['v_corrected'],
                                               shock['imf_clock'], shock['shock_clock'],
                                                shock['pre_b'], shock['delta_b'], shock['delta_b_cv'], shock['pre_std_b'],
                                                shock['pre_v'], shock['delta_v'], shock['delta_v_cv'],
                                                shock['pre_np'], shock['delta_np'],
                                                shock['pre_fp'], shock['delta_fp'],
                                                goes_sat_num, arrival_time, mlt_at_arrival]

joblib.dump(goes_impacts, 'goes_impacts_sector.pkl')
