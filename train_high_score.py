from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from src.predict_health.artifacts import save_probability_artifacts
from src.predict_health.features import add_features, feature_columns
from src.predict_health.metrics import optimize_class_multipliers, predict_with_multipliers
from src.predict_health.modeling import align_probabilities, average_probabilities, class_weight_map
from src.predict_health.submission import make_submission, validate_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a high-score student health risk pipeline.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", default="catboost,lgbm", help="Comma-separated: catboost,lgbm,hgb")
    parser.add_argument("--quick", action="store_true", help="Use fewer rows and fewer trees for smoke tests.")
    parser.add_argument("--quick-rows", type=int, default=120000)
    parser.add_argument("--multiplier-rounds", type=int, default=800)
    parser.add_argument("--catboost-iterations", type=int, default=1200)
    parser.add_argument("--catboost-task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--catboost-devices", default="0", help="CatBoost GPU device string, for example 0 or 0:1.")
    parser.add_argument("--lgbm-estimators", type=int, default=1600)
    parser.add_argument("--lgbm-device-type", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--lgbm-gpu-platform-id", type=int, default=None)
    parser.add_argument("--lgbm-gpu-device-id", type=int, default=None)
    return parser.parse_args()


def catboost_training_options(args: argparse.Namespace) -> dict[str, Any]:
    options = {"task_type": args.catboost_task_type}
    if args.catboost_task_type == "GPU":
        options["devices"] = args.catboost_devices
    return options


def lgbm_training_options(args: argparse.Namespace) -> dict[str, Any]:
    options = {"device_type": args.lgbm_device_type}
    if args.lgbm_gpu_platform_id is not None:
        options["gpu_platform_id"] = args.lgbm_gpu_platform_id
    if args.lgbm_gpu_device_id is not None:
        options["gpu_device_id"] = args.lgbm_gpu_device_id
    return options


def load_competition_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    return train, test, sample


def build_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    y = train["health_condition"].copy()
    test_ids = test["id"].copy()
    train_x = add_features(train.drop(columns=["id", "health_condition"]))
    test_x = add_features(test.drop(columns=["id"]))
    train_x, test_x = train_x.align(test_x, join="left", axis=1, fill_value=np.nan)
    return train_x, y, test_x, test_ids


def maybe_sample_quick(
    x: pd.DataFrame,
    y: pd.Series,
    rows: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if rows <= 0 or rows >= len(x):
        return x, y
    x_sample, _, y_sample, _ = train_test_split(x, y, train_size=rows, stratify=y, random_state=seed)
    return x_sample.reset_index(drop=True), y_sample.reset_index(drop=True)


def _catboost_available() -> bool:
    try:
        import catboost  # noqa: F401
    except ImportError:
        return False
    return True


def _lgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        return False
    return True


def train_catboost_fold(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    x_test: pd.DataFrame,
    categorical: list[str],
    classes: np.ndarray,
    seed: int,
    iterations: int,
    training_options: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from catboost import CatBoostClassifier, Pool

    weights = class_weight_map(y_train)
    class_weights = [weights.get(int(cls), 1.0) for cls in classes]
    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=iterations,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=6.0,
        random_strength=0.4,
        bagging_temperature=0.4,
        class_weights=class_weights,
        random_seed=seed,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=80,
        verbose=100,
        **training_options,
    )
    train_pool = Pool(x_train, y_train, cat_features=categorical)
    valid_pool = Pool(x_valid, y_valid, cat_features=categorical)
    test_pool = Pool(x_test, cat_features=categorical)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    valid_proba = align_probabilities(model.predict_proba(valid_pool), model.classes_, classes)
    test_proba = align_probabilities(model.predict_proba(test_pool), model.classes_, classes)
    return valid_proba, test_proba, {
        "best_iteration": int(model.get_best_iteration() or iterations),
        "training_options": training_options,
    }


def train_lgbm_fold(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    x_test: pd.DataFrame,
    categorical: list[str],
    classes: np.ndarray,
    seed: int,
    estimators: int,
    training_options: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import lightgbm as lgb

    x_train = x_train.copy()
    x_valid = x_valid.copy()
    x_test = x_test.copy()
    for column in categorical:
        x_train[column] = x_train[column].astype("category")
        x_valid[column] = x_valid[column].astype("category")
        x_test[column] = x_test[column].astype("category")

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(classes),
        n_estimators=estimators,
        learning_rate=0.035,
        num_leaves=48,
        min_child_samples=80,
        subsample=0.86,
        subsample_freq=1,
        colsample_bytree=0.86,
        reg_alpha=0.08,
        reg_lambda=1.2,
        class_weight=class_weight_map(y_train),
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        **training_options,
    )
    callbacks = [lgb.early_stopping(100), lgb.log_evaluation(100)]
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="multi_logloss",
        categorical_feature=categorical,
        callbacks=callbacks,
    )
    valid_proba = align_probabilities(model.predict_proba(x_valid), model.classes_, classes)
    test_proba = align_probabilities(model.predict_proba(x_test), model.classes_, classes)
    return valid_proba, test_proba, {
        "best_iteration": int(getattr(model, "best_iteration_", estimators) or estimators),
        "training_options": training_options,
    }


def train_hgb_fold(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    x_test: pd.DataFrame,
    categorical: list[str],
    classes: np.ndarray,
    seed: int,
    quick: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    numeric = [column for column in x_train.columns if column not in categorical]
    encoder_kwargs = {"handle_unknown": "ignore"}
    try:
        OneHotEncoder(sparse_output=False)
        encoder_kwargs["sparse_output"] = False
    except TypeError:
        encoder_kwargs["sparse"] = False
    preprocessor = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="__missing__"),
                    OneHotEncoder(**encoder_kwargs),
                ),
                categorical,
            ),
        ],
        sparse_threshold=0,
    )
    model = make_pipeline(
        preprocessor,
        HistGradientBoostingClassifier(
            max_iter=180 if quick else 420,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            class_weight="balanced",
            early_stopping=True,
            random_state=seed,
        ),
    )
    model.fit(x_train, y_train)
    valid_proba = align_probabilities(model.predict_proba(x_valid), model[-1].classes_, classes)
    test_proba = align_probabilities(model.predict_proba(x_test), model[-1].classes_, classes)
    return valid_proba, test_proba, {"best_iteration": None, "training_options": {}}


def train_model_family(
    name: str,
    x: pd.DataFrame,
    y_encoded: np.ndarray,
    x_test: pd.DataFrame,
    categorical: list[str],
    classes_encoded: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros((len(x), len(classes_encoded)), dtype=float)
    test_folds = []
    fold_scores = []
    fold_meta = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x, y_encoded), start=1):
        print(f"\n[{name}] fold {fold}/{args.folds}", flush=True)
        x_train, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
        y_train, y_valid = y_encoded[train_idx], y_encoded[valid_idx]
        fold_seed = args.seed + 1000 * fold
        if name == "catboost":
            valid_proba, test_proba, meta = train_catboost_fold(
                x_train,
                y_train,
                x_valid,
                y_valid,
                x_test,
                categorical,
                classes_encoded,
                fold_seed,
                250 if args.quick else args.catboost_iterations,
                catboost_training_options(args),
            )
        elif name == "lgbm":
            valid_proba, test_proba, meta = train_lgbm_fold(
                x_train,
                y_train,
                x_valid,
                y_valid,
                x_test,
                categorical,
                classes_encoded,
                fold_seed,
                400 if args.quick else args.lgbm_estimators,
                lgbm_training_options(args),
            )
        elif name == "hgb":
            valid_proba, test_proba, meta = train_hgb_fold(
                x_train,
                y_train,
                x_valid,
                y_valid,
                x_test,
                categorical,
                classes_encoded,
                fold_seed,
                args.quick,
            )
        else:
            raise ValueError(f"unknown model family: {name}")

        oof[valid_idx] = valid_proba
        test_folds.append(test_proba)
        fold_pred = classes_encoded[np.argmax(valid_proba, axis=1)]
        fold_score = float(balanced_accuracy_score(y_valid, fold_pred))
        fold_scores.append(fold_score)
        meta["fold"] = fold
        meta["balanced_accuracy"] = fold_score
        fold_meta.append(meta)
        print(f"[{name}] fold {fold} balanced_accuracy={fold_score:.6f}", flush=True)

    return oof, average_probabilities(test_folds), {
        "model": name,
        "fold_scores": fold_scores,
        "mean_fold_score": float(np.mean(fold_scores)),
        "folds": fold_meta,
    }


def available_models(requested: list[str]) -> list[str]:
    chosen = []
    for name in requested:
        if name == "catboost" and not _catboost_available():
            print("Skipping catboost because it is not installed.", flush=True)
            continue
        if name == "lgbm" and not _lgbm_available():
            print("Skipping lgbm because lightgbm is not installed.", flush=True)
            continue
        chosen.append(name)
    if not chosen:
        print("No requested high-score libraries are installed; falling back to hgb.", flush=True)
        chosen = ["hgb"]
    return chosen


def main() -> None:
    args = parse_args()
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train, test, sample = load_competition_data(args.data_dir)
    x, y, x_test, test_ids = build_features(train, test)
    if args.quick:
        x, y = maybe_sample_quick(x, y, args.quick_rows, args.seed)
        args.folds = min(args.folds, 3)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    classes = label_encoder.classes_
    classes_encoded = np.arange(len(classes))

    _, categorical = feature_columns(x)
    requested = [name.strip().lower() for name in args.models.split(",") if name.strip()]
    model_names = available_models(requested)

    model_oofs = []
    model_tests = []
    model_metadata = []
    for name in model_names:
        oof, test_proba, meta = train_model_family(
            name,
            x,
            y_encoded,
            x_test,
            categorical,
            classes_encoded,
            args,
        )
        model_oofs.append(oof)
        model_tests.append(test_proba)
        model_metadata.append(meta)

    oof_proba = average_probabilities(model_oofs)
    test_proba = average_probabilities(model_tests)
    raw_oof_pred = classes[np.argmax(oof_proba, axis=1)]
    raw_score = float(balanced_accuracy_score(y, raw_oof_pred))
    multipliers, tuned_score = optimize_class_multipliers(
        y.to_numpy(),
        oof_proba,
        classes,
        rounds=args.multiplier_rounds,
        random_state=args.seed,
    )
    tuned_oof_pred = predict_with_multipliers(oof_proba, classes, multipliers)
    tuned_score = float(balanced_accuracy_score(y, tuned_oof_pred))
    test_pred = predict_with_multipliers(test_proba, classes, multipliers)

    submission = make_submission(test_ids, test_pred)
    validate_submission(submission, sample)
    submission_path = run_dir / "submission.csv"
    metadata_path = run_dir / "metadata.json"
    submission.to_csv(submission_path, index=False)
    artifact_paths = save_probability_artifacts(run_dir, oof_proba, test_proba, y.to_numpy(), classes, test_ids.to_numpy())

    metadata = {
        "run_name": run_name,
        "rows_train": int(len(x)),
        "rows_test": int(len(x_test)),
        "quick": bool(args.quick),
        "folds": int(args.folds),
        "models": model_names,
        "classes": classes.tolist(),
        "raw_oof_balanced_accuracy": raw_score,
        "tuned_oof_balanced_accuracy": tuned_score,
        "multipliers": multipliers,
        "classification_report": classification_report(y, tuned_oof_pred, output_dict=True),
        "features": x.columns.tolist(),
        "categorical_features": categorical,
        "model_metadata": model_metadata,
        "catboost_options": catboost_training_options(args),
        "lgbm_options": lgbm_training_options(args),
        "artifacts": artifact_paths,
        "submission_path": str(submission_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nRun complete", flush=True)
    print(f"raw_oof_balanced_accuracy={raw_score:.6f}", flush=True)
    print(f"tuned_oof_balanced_accuracy={tuned_score:.6f}", flush=True)
    print(f"multipliers={multipliers}", flush=True)
    print(f"submission={submission_path}", flush=True)
    print(f"metadata={metadata_path}", flush=True)
    print(f"artifacts={artifact_paths}", flush=True)


if __name__ == "__main__":
    main()

