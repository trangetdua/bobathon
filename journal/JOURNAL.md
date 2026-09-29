# JOURNAL: Parkinson MDS-UPDRS true-OFF motor score prediction

## Status

- **Goal:** Regression — predict the continuous "true OFF" MDS-UPDRS motor score per patient visit.
- **Dataset:** Synthetic multi-cohort PD dataset; 4 CSV files in `data/`; 44 590 train rows / 11 013 test rows; 5 576 / 1 395 patients; holdout is by patient (0 overlap).

## Data understanding (EDA)

- **Status:** done — 2026-09-29
- **Summary:** 44 590 train visits (5 576 patients), 11 013 test (1 395 patients); no patient overlap. Target is near-symmetric (mean 37.5, std 16.5, skewness 0.05). `off` (r=0.886) and `on` (r=0.688) are the dominant predictors; `time_since_diagnosis` (derived) r=0.542. Heavy MNAR missingness: `ledd` 36.6%, `on` 29.6%, `off` 42.4%, `time_since_intake_off` 78.8%. Observed OFF systematically underestimates target by ~6 pts; bias worsens with shorter `time_since_intake_off`. Grouped CV on `patient_id` is mandatory (between-patient variance 14.9 pts >> within-patient 6.5 pts).
- **Report:** [data/eda.md](../data/eda.md)
- **Rich reports:** [eda_X_train.html](../data/eda_X_train.html) · [eda_X_test.html](../data/eda_X_test.html)
