"""Model validation utilities for March Madness Predictor.

Provides cross-validation, calibration evaluation, historical backtesting,
and multi-model comparison helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

from config import RANDOM_SEED, CV_FOLDS


# ------------------------------------------------------------------
# Cross-validation
# ------------------------------------------------------------------

def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = CV_FOLDS,
) -> dict:
    """Run stratified k-fold cross-validation and return per-fold metrics.

    Parameters
    ----------
    model : BaseMarchMadnessModel
        A model instance that exposes ``train``, ``predict``, and
        ``predict_proba`` methods.  A fresh clone is created for each fold
        by importing the model's class and re-instantiating it.
    X : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    y : np.ndarray
        Binary label array of shape ``(n_samples,)``.
    n_folds : int
        Number of stratified folds (default from ``config.CV_FOLDS``).

    Returns
    -------
    dict
        ``accuracy``, ``log_loss``, ``auc`` — each a list of per-fold
        values — plus ``mean_accuracy``, ``mean_log_loss``, and
        ``mean_auc`` summaries.
    """
    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED
    )

    fold_accuracy: list[float] = []
    fold_log_loss: list[float] = []
    fold_auc: list[float] = []

    model_cls = model.__class__

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Fresh model per fold to avoid data leakage
        fold_model = model_cls(random_seed=RANDOM_SEED)
        fold_model.train(X_train, y_train)

        preds = fold_model.predict(X_val)
        probas = fold_model.predict_proba(X_val)[:, 1]

        fold_accuracy.append(accuracy_score(y_val, preds))
        fold_log_loss.append(log_loss(y_val, probas))
        fold_auc.append(roc_auc_score(y_val, probas))

    return {
        "accuracy": fold_accuracy,
        "log_loss": fold_log_loss,
        "auc": fold_auc,
        "mean_accuracy": float(np.mean(fold_accuracy)),
        "mean_log_loss": float(np.mean(fold_log_loss)),
        "mean_auc": float(np.mean(fold_auc)),
        "std_accuracy": float(np.std(fold_accuracy)),
        "std_log_loss": float(np.std(fold_log_loss)),
        "std_auc": float(np.std(fold_auc)),
        "n_folds": n_folds,
    }


# ------------------------------------------------------------------
# Calibration
# ------------------------------------------------------------------

def evaluate_calibration(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Check whether predicted probabilities match actual outcome rates.

    Parameters
    ----------
    model : BaseMarchMadnessModel
        A fitted model with ``predict_proba``.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        True binary labels.
    n_bins : int
        Number of probability bins for the calibration curve.

    Returns
    -------
    dict
        ``brier_score`` — Brier score (lower is better).
        ``fraction_of_positives`` — actual positive rate per bin.
        ``mean_predicted_value`` — mean predicted probability per bin.
        ``bin_counts`` — number of samples in each bin.
        ``calibration_error`` — expected calibration error (ECE).
    """
    probas = model.predict_proba(X)[:, 1]

    brier = brier_score_loss(y, probas)
    fraction_pos, mean_pred = calibration_curve(
        y, probas, n_bins=n_bins, strategy="uniform"
    )

    # Bin counts for weighting
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_counts = np.histogram(probas, bins=bin_edges)[0]
    # Only keep bins that have samples (matching calibration_curve output)
    non_empty = bin_counts > 0
    bin_counts_filtered = bin_counts[non_empty]

    # Expected calibration error (weighted by bin size)
    total_samples = len(y)
    ece = float(
        np.sum(
            bin_counts_filtered * np.abs(fraction_pos - mean_pred)
        ) / total_samples
    )

    return {
        "brier_score": float(brier),
        "fraction_of_positives": fraction_pos.tolist(),
        "mean_predicted_value": mean_pred.tolist(),
        "bin_counts": bin_counts_filtered.tolist(),
        "calibration_error": ece,
    }


# ------------------------------------------------------------------
# Historical back-testing
# ------------------------------------------------------------------

def historical_backtest(
    model,
    seasons: list[int],
) -> dict:
    """Back-test a model on past tournament seasons.

    For each season in *seasons* the function loads historical game data
    via :pymod:`data`, engineers features with
    :class:`features.FeatureEngineeringPipeline`, trains on all other
    seasons, and evaluates on the held-out season.

    Parameters
    ----------
    model : BaseMarchMadnessModel
        Model class instance (will be re-instantiated per fold).
    seasons : list[int]
        Tournament years to iterate over (e.g. ``[2019, 2021, 2022, 2023]``).

    Returns
    -------
    dict
        ``season_results`` — per-season accuracy/log_loss/auc dict.
        ``overall_accuracy``, ``overall_log_loss``, ``overall_auc`` —
        averages across all back-tested seasons.
    """
    from data import get_all_teams
    from features import FeatureEngineeringPipeline

    model_cls = model.__class__
    season_results: dict[int, dict] = {}
    all_accuracy: list[float] = []
    all_log_loss_vals: list[float] = []
    all_auc: list[float] = []

    for hold_out_season in seasons:
        try:
            # Build training set from all other seasons
            train_seasons = [s for s in seasons if s != hold_out_season]
            X_train_parts, y_train_parts = [], []
            for s in train_seasons:
                pipeline = FeatureEngineeringPipeline(season=s)
                X_s, y_s = pipeline.build_matchup_dataset()
                X_train_parts.append(X_s)
                y_train_parts.append(y_s)

            if not X_train_parts:
                continue

            X_train = np.vstack(X_train_parts)
            y_train = np.concatenate(y_train_parts)

            # Build test set from held-out season
            test_pipeline = FeatureEngineeringPipeline(season=hold_out_season)
            X_test, y_test = test_pipeline.build_matchup_dataset()

            # Train and evaluate
            fold_model = model_cls(random_seed=RANDOM_SEED)
            fold_model.train(X_train, y_train)

            preds = fold_model.predict(X_test)
            probas = fold_model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, preds)
            ll = log_loss(y_test, probas)
            auc = roc_auc_score(y_test, probas)

            season_results[hold_out_season] = {
                "accuracy": acc,
                "log_loss": ll,
                "auc": auc,
                "n_games": len(y_test),
            }
            all_accuracy.append(acc)
            all_log_loss_vals.append(ll)
            all_auc.append(auc)

        except Exception as exc:
            season_results[hold_out_season] = {"error": str(exc)}

    return {
        "season_results": season_results,
        "overall_accuracy": float(np.mean(all_accuracy)) if all_accuracy else None,
        "overall_log_loss": float(np.mean(all_log_loss_vals)) if all_log_loss_vals else None,
        "overall_auc": float(np.mean(all_auc)) if all_auc else None,
        "seasons_evaluated": len(all_accuracy),
    }


# ------------------------------------------------------------------
# Model comparison
# ------------------------------------------------------------------

def compare_models(
    models: list,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = CV_FOLDS,
) -> pd.DataFrame:
    """Compare multiple models via cross-validation and return a summary table.

    Parameters
    ----------
    models : list[BaseMarchMadnessModel]
        Fitted or unfitted model instances to compare.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Label array.
    n_folds : int
        Number of CV folds.

    Returns
    -------
    pd.DataFrame
        One row per model with columns: ``model``, ``mean_accuracy``,
        ``std_accuracy``, ``mean_log_loss``, ``std_log_loss``,
        ``mean_auc``, ``std_auc``.
    """
    rows: list[dict] = []

    for mdl in models:
        cv_results = cross_validate_model(mdl, X, y, n_folds=n_folds)
        rows.append({
            "model": mdl.name,
            "mean_accuracy": cv_results["mean_accuracy"],
            "std_accuracy": cv_results["std_accuracy"],
            "mean_log_loss": cv_results["mean_log_loss"],
            "std_log_loss": cv_results["std_log_loss"],
            "mean_auc": cv_results["mean_auc"],
            "std_auc": cv_results["std_auc"],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("mean_accuracy", ascending=False).reset_index(drop=True)
    return df
