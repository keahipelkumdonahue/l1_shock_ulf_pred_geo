# Predicting ULF Oscillation Behavior at Geostationary Orbit from Multi-Satellite L1 Shock Observations

Zenodo Release: to be added upon acceptance

Paper: to be added upon acceptance

Authors: K. Pelkum Donahue (writing), F. Inceoglu (editing and review)

-----

Python files displayed in boldface

Data files displayed in italics

-----

**helper**: General methods used in project

**1_load_goes_data.py**: Load GOES 13, 15, 16, 17, 18 MAG data from NCEI servers

**2_prep_goes.py**: Unify GOES data into 1-min cadence dataset

**2a_preprocess_goes.py**: Calculate derived time series quantities from GOES data, including MFA field and satellite MLT

**3a_refine_steps_ts.py**: Refines shock candidate list by Pelkum Donahue & Inceoglu, 2026b: https://github.com/keahipelkumdonahue/xgboost_3_sat_shocks

_all_cases.pkl_: Raw catalogue of all candidate shock events observed a L1 by ACE, Wind, and DSCOVR, prepared by Pelkum Donahue & Inceoglu.

**3_propagate_shock.pys**: Propagates all shocks observed at L1 to GEO via plane wave propagation

**4_crosscorr.py**: Refines shock arrival time at GEO via magnetic field magnitude cross-correlation

**5_find_oscillation.py**: Search around predicted shock arrivals for ULF waves via spectral peak extraction

  **ulf_shock_detect.py**: Helper methods for ULF oscillation detection

**5a_analyze_ULF.py**: Compile all detected ULF oscillations and plot MLT distributions

## Model 1: Binary oscillation occurrence prediction

**8a_process_ts.py**: Encode time-series data via MiniRocket

**8b_ts_binary_feature_select.py**: Run feature selection: Pearson scores, VIF, RFECV

**8c_ts_binary_xgb_run**: Fully trains XGBoost models on permutations of optimal hyperparameter combinations from 8b

## Models 2-3: Large amplitude-oscillation prediction

**amp_config.py**: shared configuration file for amplitude prediction cases: toggles frequency case (Pc5, subPc5) and cutoff percentile (90 used here)

**10b_amplitude_process_ts.py**: Encode time-series data via MiniRocket

**10c_ampclass_feature_select.py**: Run feature selection: Pearson scores, VIF, RFECV

**10d_ampclass_run**: Fully trains XGBoost models on permutations of optimal hyperparameter combinations from 10c
