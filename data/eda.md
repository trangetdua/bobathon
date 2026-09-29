# EDA: Parkinson MDS-UPDRS true-OFF motor score prediction

_Generated from `data/eda.py` on 2026-09-29._

## Dataset at a glance

| Table | Rows | Columns | Role |
|---|---|---|---|
| X_train | 44 590 | 11 features | Training features |
| y_train | 44 590 | 1 (target) | True OFF MDS-UPDRS motor score |
| X_test | 11 013 | 11 features | Inference features |
| sample_submission | 11 013 | 1 (target) | Submission template |

- **Task:** Regression — predict the continuous "true OFF" MDS-UPDRS motor score per visit.
- **Patient split:** 5 576 patients in train, 1 395 in test; **zero overlap** (holdout is by patient, not by row).
- **Rich reports:** [eda_X_train.html](eda_X_train.html) · [eda_y_train.html](eda_y_train.html) · [eda_X_test.html](eda_X_test.html) · [eda_sample_submission.html](eda_sample_submission.html)

---

## 1 — Structure and schema integrity

**Columns (11 features, identical in train and test):** `patient_id`, `cohort`, `sexM`, `gene`, `age_at_diagnosis`, `age`, `ledd`, `time_since_intake_on`, `time_since_intake_off`, `on`, `off`.

- No column appears in only one split; dtypes are identical across train and test. ✔
- Index is unique in X_train, y_train, and sample_submission; inner-join of X_train ↔ y_train yields exactly 44 590 rows (0 rows lost). ✔
- No duplicate rows in X_train, X_test, or the joined frame. ✔

**Column meanings:**

| Column | Type | Meaning |
|---|---|---|
| `patient_id` | string | Patient identifier (8-char anonymous code) |
| `cohort` | string | Study cohort: A (88.9%) or B (11.1%) |
| `sexM` | binary int | 1 = male (60%), 0 = female (40%) |
| `gene` | string | Genetic mutation: No Mutation / LRRK2+ / GBA+ / OTHER+ / NaN |
| `age_at_diagnosis` | float | Age at PD diagnosis (years) |
| `age` | float | Age at visit (years) |
| `ledd` | float | Levodopa equivalent daily dose (mg) |
| `time_since_intake_on` | float | Hours since last levodopa intake for the ON measurement |
| `time_since_intake_off` | float | Hours since last levodopa intake for the OFF measurement |
| `on` | float | Observed MDS-UPDRS motor score in the ON state |
| `off` | float | Observed MDS-UPDRS motor score in the OFF state |

---

## 2 — Missing values

| Column | Train missing | Train % | Test missing | Test % |
|---|---|---|---|---|
| `patient_id` | 0 | 0 | 0 | 0 |
| `cohort` | 0 | 0 | 0 | 0 |
| `sexM` | 0 | 0 | 0 | 0 |
| `gene` | 14 432 | **32.4** | 3 526 | 32.0 |
| `age_at_diagnosis` | 2 319 | 5.2 | 511 | 4.6 |
| `age` | 0 | 0 | 0 | 0 |
| `ledd` | 16 338 | **36.6** | 4 255 | 38.6 |
| `time_since_intake_on` | 20 678 | **46.4** | 5 262 | 47.8 |
| `time_since_intake_off` | 35 116 | **78.8** | 8 712 | 79.1 |
| `on` | 13 220 | 29.6 | 3 439 | 31.2 |
| `off` | 18 913 | **42.4** | 4 494 | 40.8 |

**Key co-missingness patterns:**

- `on` and `off` are **never both missing** in the same row (train): both_present = 12 457, only_on = 18 913, only_off = 13 220, both_missing = 0. Every visit has at least one of ON/OFF, confirming the dataset description.
- `time_since_intake_off` is present in only 21.2% of rows (train); it is defined only when an OFF measure was taken.
- `ledd` and `gene` are co-missing in 4 964 rows — partial overlap suggesting that some visits lack both dosage and genetic information, likely an earlier-era cohort data gap.
- `age_at_diagnosis` is missing in 5.2% of rows. The sanity check `age ≥ age_at_diagnosis` fails for exactly these rows (the 2 319 train violations are all due to NaN comparisons in pandas, not genuine violations). No row with both fields present violates the constraint.

---

## 3 — Variable-by-variable analysis

### Categorical and binary

| Variable | Values | Notes |
|---|---|---|
| `cohort` | A: 88.9%, B: 11.1% | Heavily imbalanced; B is a minority cohort |
| `sexM` | 1: 60.0%, 0: 40.0% | Slight male majority |
| `gene` | No Mutation: 32.0%, NaN: 32.4%, LRRK2+: 17.0%, GBA+: 14.3%, OTHER+: 4.4% | No rare category except OTHER+ at 4.4%; NaN is essentially a fifth category |

### Numerical statistics (train, non-missing)

| Column | n | Mean | Median | Std | Min | Q25 | Q75 | Max | IQR×3 outliers |
|---|---|---|---|---|---|---|---|---|---|
| `age_at_diagnosis` | 42 271 | 56.8 | 56.8 | 10.9 | 16.5 | 49.6 | 64.4 | 89.9 | 0 |
| `age` | 44 590 | 62.6 | 62.5 | 11.5 | 16.5 | 54.9 | 70.5 | 103.3 | 0 |
| `ledd` | 28 252 | 637.9 | 611.0 | 218.5 | 50.0 | 482.0 | 764.0 | 1 796.0 | 15 |
| `time_since_intake_on` | 23 912 | 1.96 | 1.6 | 1.17 | 0.0 | 1.1 | 2.6 | 6.3 | 0 |
| `time_since_intake_off` | 9 474 | 14.2 | 13.8 | 2.84 | 6.9 | 12.2 | 15.7 | 24.7 | 0 |
| `on` | 31 370 | 22.0 | 21.0 | 10.4 | 0.0 | 14.0 | 28.0 | 95.0 | 10 |
| `off` | 25 677 | 26.4 | 24.0 | 16.6 | 0.0 | 13.0 | 38.0 | 106.0 | 0 |

- **`time_since_intake_on`**: short window (0–6.3 h, median 1.6 h), consistent with typical ON-phase assessments shortly after levodopa intake.
- **`time_since_intake_off`**: longer window (7–25 h, median ~14 h), consistent with overnight washout or morning pre-dose OFF assessments.
- **`ledd`**: 15 extreme high-dose outliers (>1 300 mg LEDD equivalent). Range is plausible for advanced PD.
- **`on` and `off`**: discrete integer values (0–95 and 0–106). These are clinician-scored MDS-UPDRS totals. Small numbers of outliers in `on` (10 values above ~50 at 3×IQR fence) are clinically possible.
- **`age_at_diagnosis`** range 16.5–89.9 and `age` range 16.5–103.3 span early-onset to very elderly PD, consistent with multi-centre registry data.

---

## 4 — Patient / visit structure

- **5 576 patients** in train; **1 395** in test; **0 patients in both** (true patient-level split).
- Visits per patient (train): mean = 8.0, median = 7, min = 4, Q25 = 4, Q75 = 12, max = 12.
- Visits per patient (test): mean = 7.89, median = 7, min = 4, max = 12.
- Every patient belongs to exactly **one cohort** (no cross-cohort patients in either split).

**Why grouped validation is mandatory:** Each patient appears in 4–12 rows. Their visits share the same demographics, genetics, disease trajectory, and pharmacokinetics. A row-wise random split would place some visits of a patient in the validation fold and others in training, letting the model effectively "see" the patient during training and predict on a known entity — inflating cross-validation scores versus true out-of-sample generalisation. Because the Kaggle holdout is by patient, any internal validation must also split by `patient_id` (e.g., `GroupKFold` or `GroupShuffleSplit`).

---

## 5 — Target distribution

| Stat | Value |
|---|---|
| n | 44 590 |
| min | 0.0 |
| Q25 | 25.6 |
| median | 37.3 |
| mean | 37.5 |
| Q75 | 49.3 |
| max | 109.5 |
| std | 16.5 |
| skewness | 0.051 |

The target is **nearly symmetric** (skewness ≈ 0.05), spanning the full MDS-UPDRS range [0, 109.5]. No heavy tail requiring a target transform at baseline.

**By cohort:**

| Cohort | n | Mean | Median | Std |
|---|---|---|---|---|
| A | 39 636 | 37.95 | 37.9 | 16.62 |
| B | 4 954 | 33.68 | 33.3 | 14.97 |

Cohort B patients have on average **~4.3 points lower** true OFF scores and less variance, consistent with their shorter mean disease duration (4.1 vs 6.1 years).

**Within- vs between-patient variability:**

- Between-patient SD of individual means: **14.89 points** — patients differ substantially in baseline severity.
- Within-patient SD (mean across patients): **6.53 points** — individual patients also progress over their visits.
- This large between-patient variation (>>within-patient variation) reinforces the need for patient-grouped CV: a model can trivially memorise patient averages.

---

## 6 — Feature–target relationships

### 6a. Target mean by missingness

| Variable | Target mean when present | Target mean when missing | Interpretation |
|---|---|---|---|
| `off` | 32.4 | 44.4 | OFF is more often measured when severity is lower (less impaired patients?) |
| `on` | 44.3 | 21.3 | ON is more often measured for higher-severity patients — likely because sicker patients need ON monitoring |
| `ledd` | 44.3 | 25.6 | LEDD absent in milder patients (possibly earlier disease, no/low treatment) |
| `gene` | 37.0 | 38.6 | Gene missingness barely shifts target mean; likely administrative |

These patterns indicate **missingness is not at random (MNAR)**. The missingness of `off`, `on`, and `ledd` correlates with disease severity. Imputing with global means would introduce bias; missingness indicators should be considered as features.

### 6b. Measurement bias: (off − target) by time since OFF intake

Overall mean bias: **−5.95 ± 8.0** (off systematically **underestimates** true OFF, i.e., observed OFF is lower than true OFF).

| Time since intake (OFF) | Mean bias | Median bias | n |
|---|---|---|---|
| 4–8 h | −13.9 | −13.1 | 24 |
| 8–12 h | −9.0 | −7.9 | 2 105 |
| 12–24 h | −7.9 | −6.7 | 7 326 |
| >24 h | −4.7 | −2.2 | 19 |
| missing | −4.7 | −2.7 | 16 203 |

**Interpretation (observed fact):** The shorter the time since last levodopa intake when the OFF measurement is taken, the greater the negative bias (observed OFF < true OFF). This is consistent with the pharmacodynamic model in `CONTEXT.md`: if measured too early after the last dose, residual levodopa still suppresses symptoms, so the OFF score looks better than the "true" fully-washed-out state. The model must use `time_since_intake_off` to correct this bias. The large `missing` group (16 203 rows where OFF is present but time is unknown) has intermediate bias.

### 6c. Target progression with disease duration

| Disease duration (years) | Mean target | n |
|---|---|---|
| 0–2.5 | 26.1 | 9 331 |
| 2.5–4.9 | 32.0 | 10 491 |
| 4.9–7.4 | 38.9 | 8 961 |
| 7.4–9.9 | 45.4 | 6 539 |
| 9.9–12.4 | 50.1 | 4 333 |
| 12.4–14.8 | 52.8 | 1 727 |
| 14.8+ | ~55 | <1 000 |

Clear monotonic increase in mean target over the first 12–15 years post-diagnosis, consistent with progressive neurodegeneration. `time_since_diagnosis = age − age_at_diagnosis` is a strong derived feature.

### 6d. Individual patient trajectories (random_state=0)

Four sampled patients confirm: target increases monotonically with age for most; observed `on` and `off` values fluctuate around target; some patients have mostly ON measurements, others mostly OFF.

### 6e. Pearson correlations with target

| Feature | Pearson r | n |
|---|---|---|
| `off` | **+0.886** | 25 677 |
| `on` | **+0.688** | 31 370 |
| `time_since_diagnosis` (derived) | **+0.542** | 42 271 |
| `age` | +0.310 | 44 590 |
| `ledd` | +0.298 | 28 252 |
| `age_at_diagnosis` | +0.133 | 42 271 |
| `time_since_intake_on` | 0.000 | 23 912 |
| `time_since_intake_off` | +0.008 | 9 474 |

`off` is the single strongest predictor (r = 0.886), followed by `on` (r = 0.688). `time_since_diagnosis` is the strongest purely-derived signal (r = 0.542). The intake-time features have near-zero linear correlation with target but are mechanistically important for bias correction (non-linear effect seen in 6b).

### 6f. Cohort comparison

| | Cohort A | Cohort B |
|---|---|---|
| Visits | 39 636 | 4 954 |
| Target mean | 37.95 | 33.68 |
| Target std | 16.62 | 14.97 |
| Mean time-since-diagnosis | 6.1 yr | 4.1 yr |
| Mean LEDD | 647 mg | 555 mg |
| Mean (off − target) bias | −6.1 | −4.7 |

Cohort B is earlier-stage (less time since diagnosis, lower LEDD, lower target), and its OFF measurements are slightly less biased. A `cohort` indicator or separate cohort-level intercept may be useful.

---

## 7 — Column associations (skrub)

Top pairwise associations in the joined train set (Cramér's V + Pearson):

| Left | Right | Cramér's V | Pearson |
|---|---|---|---|
| `age_at_diagnosis` | `age` | 0.562 | **0.942** | ← strongest structural link |
| `off` | `target` | 0.495 | 0.886 | ← strongest predictive signal |
| `on` | `target` | 0.403 | 0.688 | |
| `on` | `off` | 0.345 | 0.867 | ← on and off are highly correlated |
| `ledd` | `on` | 0.292 | 0.210 | |
| `time_since_intake_on` | `on` | 0.254 | −0.220 | ← earlier ON measures → higher apparent score |
| `ledd` | `target` | 0.236 | 0.298 | |

The `age_at_diagnosis` / `age` near-perfect Pearson (0.942) is expected — most patients are diagnosed and followed within a similar age window. No implausible 1.0 associations suggesting leakage.

---

## 8 — Data leakage assessment

| Finding | Classification |
|---|---|
| `on` and `off` present in X_test | ✅ **Normal** — legitimate clinical inputs, not leaks. They are scores measured at each visit, defined in X_test by the competition. |
| `off` r = 0.886 with target | ✅ **Normal** — observed OFF is the biased version of true OFF; the task is exactly to un-bias it. |
| `patient_id` shared across rows | ⚠️ **Risk** — should be excluded as a raw feature (high cardinality identifier, 5 576 unique values ≈ unique ratio 0.125); group-fold on it for CV. |
| Row-wise random train/val split | ⚠️ **Confirmed patient-leakage risk** — same patient visits in both folds. Must use GroupKFold on `patient_id`. |
| Duplicate rows | ✅ None found (0 in train, test, and joined frame). |
| Temporal column | No explicit date; `age` serves as a proxy for visit time within a patient's trajectory. No `TimeSeriesSplit` is needed unless we want within-patient forecasting, but `age` ordering matters for within-patient features. |

**Distinguishing normal from problematic:** `off` and `on` are biased measurements of the same construct as `target`; their presence in X_test makes them valid predictors. The leakage concern is cross-patient (grouping), not feature-level.

---

## 9 — Train / test distribution comparison

All distributions are well-aligned:

- **Cohort proportions:** A=88.9%/89.5%, B=11.1%/10.5% — essentially the same.
- **Sex:** train 60%M / test 58.1%M — minor shift.
- **Gene:** proportions within ±1.5 pp across all categories.
- **Numeric features:** means differ by ≤1 unit, stds are nearly identical.
- **Missingness rates:** all columns within ±2 pp between train and test.
- **No new categories** in test that are absent from train.

No significant covariate shift detected. The split appears to be an i.i.d. random draw by patient.

---

## Key findings and modelling implications

### Observed facts

1. **`off` (r=0.886) and `on` (r=0.688)** are the strongest predictors — they are clinically measured proxies for target. Both are present in X_test and are legitimate inputs.
2. **`time_since_diagnosis`** (derived: `age − age_at_diagnosis`) explains 29% of target variance (r=0.542) and captures progressive neurodegeneration.
3. **Observed `off` systematically underestimates target by ~6 points** (mean bias −5.95). This bias grows with shorter `time_since_intake_off`, confirming the pharmacodynamic mechanism.
4. **ON/OFF are never simultaneously missing.** Every row has at least one; 27.9% of rows have both.
5. **Missingness is not at random:** `ledd` and `on` are more often missing for lower-severity patients; `off` is more often missing for higher-severity patients.
6. **Patient-level holdout confirmed:** 0 patients in common between train and test.
7. **5 576 patients, 4–12 visits each.** Between-patient variability (SD 14.9 pts) exceeds within-patient variability (SD 6.5 pts).
8. **Target is near-normally distributed** (skewness 0.05); no log-transform needed at baseline.
9. **Cohort B** is earlier-stage (~2 years shorter disease duration, ~4 pts lower mean target, less LEDD, less bias).

### Hypotheses (to validate with the model)

- The bias in `off` relative to `target` is primarily driven by `time_since_intake_off`; a model that uses both `off` and `time_since_intake_off` should correct the bulk of this bias.
- Missingness indicators for `ledd`, `on`, `off`, and `gene` may carry signal correlated with disease stage.
- Within-patient temporal features (e.g., previous visit's target, rate of progression) could reduce within-patient residuals, but require careful engineering to avoid leakage.

### Preparation, validation, and model selection guidance

| Decision | Recommendation | Evidence |
|---|---|---|
| **Cross-validation splitter** | `GroupKFold(n_splits=5)` on `patient_id` | 0 patient overlap in Kaggle holdout; between-patient variance >> within-patient |
| **Metric** | RMSE (competition default); check MAE for outlier sensitivity | Symmetric, continuous target |
| **Derived feature** | `time_since_diagnosis = age − age_at_diagnosis` | r=0.542 with target; encodes progression |
| **Missingness** | Do NOT global-mean impute `on`/`off`/`ledd` without indicators; MNAR pattern | Target mean shifts by 10–20 pts between present/absent rows |
| **`patient_id`** | Exclude as raw feature; use only as group key for CV | Near-unique identifier; would overfit |
| **`gene` NaN** | Treat NaN as its own category (`"missing"`) | 32.4% NaN; it may encode a cohort subtype |
| **`cohort`** | Include as feature; A and B differ in target mean, LEDD, and bias pattern | 4.3-pt difference in target means |
| **Target transform** | Not needed at baseline | Skewness ≈ 0 |
| **Gradient-boosted trees** | Good default (handles missingness natively, non-linear interactions) | MNAR, non-linear bias correction with `time_since_intake_off` |

### Open questions

1. **Meaning of `age_at_diagnosis` when missing:** Is it truly unknown or a data entry gap? (5.2% missing; violations in `age ≥ age_at_diagnosis` are all NaN-comparisons, not genuine errors.)
2. **`time_since_intake_off` distribution (missing vs present):** Large majority (78.8%) missing — does missing mean "no OFF measurement taken" or "time not recorded"? Currently modelled as "either no OFF was measured or time not recorded". The co-missingness check shows `time_since_intake_off` is only populated when `off` is present (by construction), but not always even then.
3. **Is `ledd` the dose at the time of visit or an average?** The CONTEXT.md suggests it is the daily dose; confirmed by the plausible unit range (50–1 796 mg LEDD).
4. **`OTHER+` gene variant (4.4%):** Small category; may need to be merged with `No Mutation` or left as-is depending on model sensitivity.
5. **Whether within-patient lag features** (target at previous visit, delta since last visit) are permissible under competition rules and available in the test set structure.
