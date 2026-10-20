import joblib
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas as pd
from spacepy import coordinates as coord
from spacepy.time import Ticktock

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def convert_gsm_to_mfa(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Convert GOES magnetic field data from GSM to MFA coordinates.

    Replaces load_goes_mfa_day() for use with an already-loaded DataFrame.

    MFA coordinate system (right-handed):
        b_parallel  : along local B field vector (field-aligned)
        b_azimuthal : B × r̂  (roughly eastward, perpendicular to B and radial)
        b_radial    : parallel × azimuthal (completes RH system)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
            'time'      - datetime-like index or column
            'x','y','z' - spacecraft position in GSE, Earth radii (Re)
            'Bx','By','Bz' - magnetic field in GSM, any consistent unit (e.g. nT)
            'mlt'       - magnetic local time

    Returns
    -------
    pd.DataFrame with columns ['time', 'mlt', 'b_parallel', 'b_azimuthal', 'b_radial']
    at whatever cadence the input has, or None if conversion fails.
    """

    required = {"x", "y", "z", "Bx", "By", "Bz", "mlt"}
    missing = required - set(df.columns)
    if missing:
        print(f"[convert_gse_to_mfa] Missing required columns: {missing}")
        return None

    try:
        # ------------------------------------------------------------------ #
        # 1.  Extract arrays
        # ------------------------------------------------------------------ #
        pos = df[["x", "y", "z"]].to_numpy(dtype=float)  # (N, 3)  GSE Re
        B = df[["Bx", "By", "Bz"]].to_numpy(dtype=float)  # (N, 3)  GSE nT

        B_mag = np.linalg.norm(B, axis=1, keepdims=True)  # (N, 1)

        # Drop rows where |B| == 0 to avoid divide-by-zero; will NaN-fill later
        valid = (B_mag[:, 0] > 0) & np.all(np.isfinite(B), axis=1) \
                & np.all(np.isfinite(pos), axis=1)

        # ------------------------------------------------------------------ #
        # 2.  Build MFA unit vectors
        # ------------------------------------------------------------------ #
        # â_parallel : unit vector along B
        e_par = np.where(valid[:, None], B / np.where(B_mag > 0, B_mag, 1), np.nan)

        # â_azimuthal : B × r̂  (normalised)
        r_mag = np.linalg.norm(pos, axis=1, keepdims=True)
        e_r = np.where(r_mag > 0, pos / np.where(r_mag > 0, r_mag, 1), np.nan)

        cross = np.cross(e_par, e_r)  # B × r̂
        cross_mag = np.linalg.norm(cross, axis=1, keepdims=True)
        e_az = np.where(cross_mag > 0, cross / np.where(cross_mag > 0, cross_mag, 1), np.nan)

        # â_radial : parallel × azimuthal  (completes RH triad)
        e_rad = np.cross(e_par, e_az)

        # ------------------------------------------------------------------ #
        # 3.  Project B onto MFA axes
        # ------------------------------------------------------------------ #
        b_parallel = np.einsum("ij,ij->i", B, e_par)
        b_azimuthal = np.einsum("ij,ij->i", B, e_az)
        b_radial = np.einsum("ij,ij->i", B, e_rad)

        # NaN out invalid rows
        b_parallel[~valid] = np.nan
        b_azimuthal[~valid] = np.nan
        b_radial[~valid] = np.nan

        # ------------------------------------------------------------------ #
        # 4.  Assemble output DataFrame — original columns + 3 MFA components
        # ------------------------------------------------------------------ #
        out = df.copy()
        out["b_parallel"] = b_parallel
        out["b_azimuthal"] = b_azimuthal
        out["b_radial"] = b_radial

        return out

    except Exception as exc:
        print(f"[convert_gse_to_mfa] Conversion failed: {exc}")
        return None

sats = [13, 15, 16, 17, 18]
years = np.arange(2017, 2024)

for year in years:

    print('...........................................', year, '...........................................')

    for sat in sats:

        try:
            data = joblib.load('goes_annual_combined/{}_g{}.pkl'.format(year, sat))

        except:
            print('Sat not operational - skipping')

        else:
            print(data.keys())
            data.columns = ['x', 'y', 'z', 'He', 'Hp', 'Hn', 'Bx', 'By', 'Bz']

            data['dHp_dt'] = np.gradient(data['Hp'])

            data['B_total'] = np.sqrt(data['Bx']**2 + data['By']**2 + data['Bz']**2)
            data['dB_total_dt'] = np.gradient(data['B_total'])

            angle = np.degrees(np.arctan2(data['y'], data['x']))
            data['mlt'] = (12 + angle / 15.0) % 24

            data = convert_gsm_to_mfa(data)

            joblib.dump(data, 'raw_data/goes_annual_combined/{}_g{}_calc.pkl'.format(year, sat))
