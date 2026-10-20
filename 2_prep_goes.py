import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from helper import interpolate

sats = [13, 15, 16, 17, 18]
years = np.arange(2017, 2024)

proj_directory = ''

for year in years:

    print('...........................................', year, '...........................................')

    for sat in sats:
        print('................................. GOES', sat, '.................................')

        try:
            pos = joblib.load('{}/mag_prop_1/raw_data/goes_annual/{}_g{}_pos.pkl'.format(proj_directory, year, sat))
            b_epn = joblib.load('{}/mag_prop_1/raw_data/goes_annual/{}_g{}_b_epn.pkl'.format(proj_directory, year, sat))
            b_xyz = joblib.load('{}/mag_prop_1/raw_data/goes_annual/{}_g{}_b_xyz.pkl'.format(proj_directory, year, sat))

        except:
            print('Sat not operational - skipping')

        else:

            print('............... Position ...............')

            pos.index = pd.to_datetime(pos.index).round('1s')  # round off nanoseconds

            print('Before:', len(pos))

            pos = interpolate(pos, '1s', False)
            pos = pos.loc[pos.index.second == 30]

            print('After:', len(pos))

            print('............... B EPN ...............')

            b_epn.index = pd.to_datetime(b_epn.index).round('1s')  # round off nanoseconds

            print('Before:', len(b_epn))

            # b_epn = interpolate(b_epn, '1s', False)  # interpolate missing values
            b_epn = b_epn.resample('min').mean()
            b_epn = b_epn.shift(0.5, freq='min')

            print('After:', len(b_epn))

            print('............... B XYZ ...............')

            b_xyz.index = pd.to_datetime(b_xyz.index).round('1s')  # round off nanoseconds

            print('Before:', len(b_xyz))

            # b_xyz = interpolate(b_xyz, '1s', False)  # interpolate missing values
            b_xyz = b_xyz.resample('min').mean()
            b_xyz = b_xyz.shift(0.5, freq='min')

            print('After:', len(b_xyz))

            print('.................... Cut off extraneous ends ....................')

            print('Pos:', pos.index[0], 'to', pos.index[-1])
            print('B EPN:', b_epn.index[0], 'to', b_epn.index[-1])
            print('B XYZ:', b_xyz.index[0], 'to', b_xyz.index[-1])

            starts = [pos.index[0], b_epn.index[0], b_xyz.index[0]]
            latest_start = max(starts)

            print('Latest start:', latest_start)
            ends = [pos.index[-1], b_epn.index[-1], b_xyz.index[-1]]
            earliest_end = min(ends)

            print('Earliest end:', earliest_end)


            ############ Trimming ############

            def trim(data):
                trimmed = data.loc[data.index >= latest_start]
                trimmed = trimmed.loc[trimmed.index <= earliest_end]
                return trimmed

            pos, b_epn, b_xyz = trim(pos), trim(b_epn), trim(b_xyz)

            print('\nAdjusted lengths:')
            print('Pos:', len(pos))
            print('EPN:', len(b_epn))
            print('XYZ:', len(b_xyz))

            print(False in pos.index == b_epn.index)  # check that all datetime indices are identical (Should print False)
            print(False in b_epn.index == b_xyz.index)

            print('..................... Combining into 1 df ....................')

            x = pos['x']
            y = pos['y']
            z = pos['z']

            be = b_epn['x']
            bp = b_epn['y']
            bn = b_epn['z']

            bx = b_xyz['x']
            by = b_xyz['y']
            bz = b_xyz['z']

            all_data = pd.concat([x, y, z,
                                  be, bp, bn,
                                  bx, by, bz], axis=1, ignore_index=True)
            all_data.rename(columns={0:'x', 1:'y', 2:'z',
                                     3:'be', 4:'bp', 5:'bn',
                                     6:'bx', 7:'by', 8:'bz'}, inplace=True)

            # print('length before dropna:', len(all_data))
            # all_data = all_data.dropna()
            # print('length after dropna:', len(all_data))

            # print(all_data)

            print('.................... Saving Data ....................')

            joblib.dump(all_data, '{}/mag_prop_1/raw_data/goes_annual_combined/{}_g{}.pkl'.format(proj_directory, year, sat))


