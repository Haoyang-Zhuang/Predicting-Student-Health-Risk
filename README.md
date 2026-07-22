# Predicting Student Health Risk

High-score Kaggle pipeline for Playground Series S6E7.

## Quick smoke test

```powershell
D:\Anaconda3\envs\py38\python.exe -m unittest discover -s tests -v
D:\Anaconda3\envs\py38\python.exe train_high_score.py --quick --models hgb --run-name quick-hgb
```

## High-score run

Install `catboost` and `lightgbm` in the Python environment, then run:

```powershell
D:\Anaconda3\envs\py38\python.exe train_high_score.py --models catboost,lgbm --folds 5 --run-name full-cb-lgbm
```

The script writes `submission.csv` and `metadata.json` under `outputs/<run-name>/`.
