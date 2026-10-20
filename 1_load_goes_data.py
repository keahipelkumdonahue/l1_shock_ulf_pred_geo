import numpy as np
import cftime
import netCDF4 as nc
import os
import wget
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from cdasws.datarepresentation import DataRepresentation as dr
from cdasws import CdasWs
from datetime import datetime, timedelta
from helper import interpolate
from spacepy import coordinates as coord
from spacepy.time import Ticktock
import joblib
from joblib import Parallel, delayed

cdas = CdasWs()

# ─────────────────────────────────────────────
# URL / filename
# ─────────────────────────────────────────────

import urllib.request

VERSIONS_NEW = ['v1-0-1', 'v1-0-2', 'v2-0-3']  # extend this list if new versions appear

def url_exists(url):
    """HEAD request to check if a URL is reachable without downloading."""
    try:
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status == 200
    except Exception:
        return False

def get_url_and_filename(sat, day):
    if sat <= 15:
        base = (f'https://www.ncei.noaa.gov/data/goes-space-environment-monitor/'
                f'access/science/mag/goes{sat}/magn-l2-hires/{day.year}/{day.month:02d}/')
        filename = f'dn_magn-l2-hires_g{sat}_d{day.year}{day.month:02d}{day.day:02d}_v0_0_2.nc'
        return base + filename, filename
    else:
        base = (f'https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/'
                f'goes/goes{sat}/l2/data/magn-l2-avg1m/{day.year}/{day.month:02d}/')
        stem = f'dn_magn-l2-hires_g{sat}_d{day.year}{day.month:02d}{day.day:02d}'
        for version in VERSIONS_NEW:
            filename = f'{stem}_{version}.nc'
            url = base + filename
            if url_exists(url):
                return url, filename
        # Fall back to v1-0-1 if none found (wget will fail gracefully)
        filename = f'{stem}_{VERSIONS_NEW[0]}.nc'
        return base + filename, filename

# ─────────────────────────────────────────────
# Download (parallel, python wget)
# ─────────────────────────────────────────────

def wget_all(sat, start_day, n_days, n_workers=4):
    tasks = []
    for i in range(n_days):
        day = start_day + timedelta(days=i)
        url_path, filename = get_url_and_filename(sat, day)
        nc_path = os.path.join(NC_DIR, filename)
        out_path = os.path.join(OUT_DIR, f'g{sat}_{day.date()}.pkl')
        # skip if pkl already exists OR .nc already downloaded
        if os.path.exists(out_path) or os.path.exists(nc_path):
            continue
        tasks.append((url_path, NC_DIR))

    if not tasks:
        print('All .nc files already downloaded or processed.')
        return

    print(f'Downloading {len(tasks)} files ({n_workers} parallel)...')
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(wget.download, url, dst): url for url, dst in tasks}
        for fut in as_completed(futures):
            url = futures[fut]
            try:
                fut.result()
                # print(f'\n  Done: {os.path.basename(url)}')
            except Exception as e:
                print(f'\n  FAILED {os.path.basename(url)}: {e}')

    print('All downloads complete.')

# ─────────────────────────────────────────────
# CDAWeb position
# ─────────────────────────────────────────────

def framebuilder(label, array, idx):
    return pd.DataFrame({str(label): pd.Series(array[:], index=idx[:])}).sort_index()


def get_and_process_data(starttime, endtime, quant, dataset, variable):
    start_str = starttime.strftime('%Y%m%dT%H:%M:00Z')
    end_str   = endtime.strftime('%Y%m%dT%H:%M:00Z')
    data = cdas.get_data(dataset, [variable], start_str, end_str,
                         dataRepresentation=dr.XARRAY)[1]
    time = data.coords['Epoch']
    a = getattr(data, variable)[:, 0].to_numpy()
    b = getattr(data, variable)[:, 1].to_numpy()
    c = getattr(data, variable)[:, 2].to_numpy()
    return framebuilder(quant, a, time), framebuilder(quant, b, time), framebuilder(quant, c, time)


def load_cdaweb_pos(sat, day):
    x, y, z = get_and_process_data(
        day, day + timedelta(days=1), 'pos',
        f'GOES{sat}_EPHEMERIS_SSC', 'XYZ_GSE'
    )
    pos = pd.concat([x, y, z], axis=1, ignore_index=True)
    pos.rename(columns={0: 'x', 1: 'y', 2: 'z'}, inplace=True)
    return pos

# ─────────────────────────────────────────────
# MLT
# ─────────────────────────────────────────────

def add_mlt_spacepy(df, x_col='x', y_col='y', z_col='z'):
    """Full GSE -> SM transform via SpacePy (IGRF dipole tilt corrected)."""
    pos_gse = coord.Coords(
        list(zip(df[x_col], df[y_col], df[z_col])), 'GSE', 'car'
    )
    pos_gse.ticks = Ticktock(df.index.to_pydatetime(), 'UTC')
    pos_sm = pos_gse.convert('SM', 'car')
    df['mlt'] = (np.arctan2(pos_sm.data[:, 1], pos_sm.data[:, 0]) * (12 / np.pi) + 12) % 24
    return df

# ─────────────────────────────────────────────
# Process one day
# ─────────────────────────────────────────────
def find_local_nc(sat, day):
    """Find whichever version of the .nc file exists locally."""
    if sat <= 15:
        _, filename = get_url_and_filename(sat, day)
        return os.path.join(NC_DIR, filename)

    stem = f'dn_magn-l2-hires_g{sat}_d{day.year}{day.month:02d}{day.day:02d}'
    for version in VERSIONS_NEW:
        filename = f'{stem}_{version}.nc'
        nc_path = os.path.join(NC_DIR, filename)
        if os.path.exists(nc_path):
            return nc_path
    return None  # not found


def process_day(sat, day):
    out_path = os.path.join(OUT_DIR, f'g{sat}_{day.date()}.pkl')

    if os.path.exists(out_path):
        print(f'[{day.date()}] Already processed, skipping.')
        return

    # Use local scan instead of reconstructing filename
    nc_path = find_local_nc(sat, day)

    if nc_path is None:
        print(f'[{day.date()}] MISSING .nc, skipping.')
        return

    print(f'[{day.date()}] Processing...')
    try:
        # Load B-field and clip to this day only
        day_start = pd.Timestamp(day)
        day_end = pd.Timestamp(day_start + timedelta(days=1))

        daily_b = pd.DataFrame(columns=['time', 'be', 'bp', 'bn', 'bx', 'by', 'bz'])
        with nc.Dataset(nc_path) as ncf:
            time_b = pd.DatetimeIndex(
                cftime.num2date(ncf.variables['time'][:],
                                ncf.variables['time'].units,
                                only_use_cftime_datetimes=False,
                                only_use_python_datetimes=True)
            )
            b_epn = ncf['b_epn'][:]
            b_gsm = ncf['b_gsm'][:]
            daily_b['time'] = time_b
            daily_b['be'] = b_epn[:, 0]
            daily_b['bp'] = b_epn[:, 1]
            daily_b['bn'] = b_epn[:, 2]
            daily_b['bx'] = b_gsm[:, 0]
            daily_b['by'] = b_gsm[:, 1]
            daily_b['bz'] = b_gsm[:, 2]
            daily_b.set_index('time', inplace=True)

        # Clip to exact day in case .nc contains extra data
        daily_b = daily_b.loc[(daily_b.index <= day_end)]

        # Load position from CDAWeb
        daily_pos = load_cdaweb_pos(sat, day)

        # Resample B to 1-min means, shift timestamps to bin centres
        b_min = daily_b.resample('min').mean()
        b_min = b_min.shift(0.5, freq='min')

        # MLT + interpolate position to 1-min
        pos_mlt = add_mlt_spacepy(daily_pos)
        pos_min = interpolate(pos_mlt, '1s', False)
        pos_min = pos_min.loc[pos_min.index.second == 30]

        if len(pos_min) != len(b_min):
            print('------------------------------------------------')
            print(f'  WARNING: Lengths do not match: {len(pos_min)} vs {len(b_min)}')
            print('Date:', day)
            print('pos_min:', pos_min.index[0], 'to', pos_min.index[-1])
            print('b_min:', b_min.index[0], 'to', b_min.index[-1])
            print('------------------------------------------------')

        print(f'  Lengths match: {len(pos_min) == len(b_min)}')
        print(f'  Index discrepancies: {(pos_min.index != b_min.index).sum()}')

        goes_daily = pd.concat([b_min, pos_min], axis=1)
        joblib.dump(goes_daily, out_path)
        # print(f'  Saved {out_path}')

    except Exception as e:
        print(f'  FAILED: {e}')

    finally:
        # Delete .nc when done regardless of success/failure
        if os.path.exists(nc_path):
            os.remove(nc_path)
            # print(f'  Deleted {nc_path}')

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run(sat, start_day, n_days, n_jobs=4):
    # Step 1: download all .nc files in parallel
    wget_all(sat, start_day, n_days, n_workers=n_jobs)

    # Step 2: process all days in parallel
    days = [start_day + timedelta(days=i) for i in range(n_days)]
    Parallel(n_jobs=n_jobs)(delayed(process_day)(sat, day) for day in days)

NC_DIR    = 'raw_data/nc'
OUT_DIR   = 'raw_data/goes_daily'

os.makedirs(NC_DIR,  exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

year = 2017
sat = 13

days = (datetime(year+1, 1, 1) - datetime(year, 1, 1)).days
print(days)

print('{}: GOES {}'.format(year, sat))
run(sat, datetime(year, 1, 1), days)
