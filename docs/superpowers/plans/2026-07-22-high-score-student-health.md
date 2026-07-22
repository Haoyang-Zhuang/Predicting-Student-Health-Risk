# High Score Student Health Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible high-score Kaggle pipeline for Predicting Student Health Risk.

**Architecture:** Keep reusable logic in a small `predict_health` package and keep the training entry point in one CLI script. The pipeline reads Kaggle CSVs, adds compact tabular features, trains stratified-fold models, tunes class probability multipliers on OOF predictions for balanced accuracy, and writes `submission.csv`.

**Tech Stack:** Python 3.8, Pandas, NumPy, scikit-learn, CatBoost, LightGBM, pytest.

---

### Task 1: Project Structure and Utility Tests

**Files:**
- Create: `src/predict_health/__init__.py`
- Create: `src/predict_health/features.py`
- Create: `src/predict_health/metrics.py`
- Create: `src/predict_health/submission.py`
- Create: `tests/test_features.py`
- Create: `tests/test_metrics.py`
- Create: `tests/test_submission.py`

- [ ] Write tests for feature creation preserving rows, adding missing indicators, and preserving categorical columns as strings.
- [ ] Write tests for balanced-accuracy multiplier search improving or matching baseline predictions.
- [ ] Write tests for submission validation enforcing `id` and `health_condition` columns.
- [ ] Run tests and confirm they fail because implementation files do not exist.

### Task 2: Utility Implementation

**Files:**
- Modify: `src/predict_health/features.py`
- Modify: `src/predict_health/metrics.py`
- Modify: `src/predict_health/submission.py`

- [ ] Implement `add_features(df)` with missing indicators, compact ratio/difference features, and string-filled categorical columns.
- [ ] Implement `optimize_class_multipliers(y_true, proba, classes)` using deterministic random/grid search around per-class multipliers.
- [ ] Implement `make_submission(ids, labels)` and `validate_submission(submission, sample)`.
- [ ] Run the utility tests and confirm they pass.

### Task 3: High-Score Training CLI

**Files:**
- Create: `train_high_score.py`
- Create: `requirements.txt`
- Create: `README.md`

- [ ] Implement CLI arguments for data path, output path, fold count, model list, random seed, and quick mode.
- [ ] Train `StratifiedKFold` models and store OOF predictions.
- [ ] Support CatBoost and LightGBM when installed, with scikit-learn HistGradientBoosting fallback.
- [ ] Tune class multipliers against OOF predictions using balanced accuracy.
- [ ] Predict test probabilities, apply tuned multipliers, and write `submission.csv`.
- [ ] Save run metadata with CV score, multipliers, classes, models, and feature list.

### Task 4: Verification and Submission

**Files:**
- Output: `outputs/<run-name>/submission.csv`
- Output: `outputs/<run-name>/metadata.json`

- [ ] Install missing dependencies or use fallback model if dependency installation is unavailable.
- [ ] Run unit tests.
- [ ] Run a quick validation training job to confirm the pipeline works end-to-end.
- [ ] Run a stronger full-data job when dependencies are available.
- [ ] Confirm submission row count and ID alignment with `sample_submission.csv`.
