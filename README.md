# Predicting Student Health Risk

High-score Kaggle pipeline for Playground Series S6E7.

## Quick smoke test

```powershell
D:\Anaconda3\envs\py38\python.exe -m unittest discover -s tests -v
D:\Anaconda3\envs\py38\python.exe train_high_score.py --quick --models hgb --run-name quick-hgb
```

## Full training

Default full ensemble: CatBoost GPU + LightGBM CPU:

```powershell
D:\Anaconda3\envs\py38\python.exe train_high_score.py --models catboost,lgbm --folds 5 --run-name full-cb-gpu-lgbm
```

CatBoost on GPU:

```powershell
D:\Anaconda3\envs\py38\python.exe train_high_score.py --models catboost --folds 5 --run-name full-catboost-gpu --catboost-devices 0
```

LightGBM on GPU, only if the installed LightGBM build supports GPU:

```powershell
D:\Anaconda3\envs\py38\python.exe train_high_score.py --models lgbm --folds 5 --run-name full-lgbm-gpu --lgbm-device-type gpu
```

Mixed run with CatBoost GPU and LightGBM CPU:

```powershell
D:\Anaconda3\envs\py38\python.exe train_high_score.py --models catboost,lgbm --folds 5 --run-name full-cb-gpu-lgbm --catboost-devices 0
```

Each training run writes these files under `outputs/<run-name>/`:

```text
submission.csv
metadata.json
oof_proba.npy
test_proba.npy
y_true.npy
classes.npy
test_ids.npy
```

## Blend previous runs

To blend separately trained runs, both source runs must have the `.npy` artifact files listed above. Older runs created before artifact support need to be rerun.

```powershell
D:\Anaconda3\envs\py38\python.exe blend_runs.py --runs outputs\full-lgbm-artifacts outputs\full-catboost-gpu --weights 0.5,0.5 --run-name blend-lgbm-catboost
```

If `--weights` is omitted, all runs use equal weights. The blend script searches class multipliers on the blended OOF probabilities, then writes a new `submission.csv` and `metadata.json`.

