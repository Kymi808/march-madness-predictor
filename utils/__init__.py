"""Utility functions for March Madness Predictor.

Exports validation and visualization helpers.
"""

from utils.validation import (
    cross_validate_model,
    evaluate_calibration,
    historical_backtest,
    compare_models,
)
from utils.visualization import (
    plot_bracket,
    plot_win_probabilities,
    plot_feature_importance,
    plot_model_comparison,
    plot_upset_probability,
)

__all__ = [
    # Validation
    "cross_validate_model",
    "evaluate_calibration",
    "historical_backtest",
    "compare_models",
    # Visualization
    "plot_bracket",
    "plot_win_probabilities",
    "plot_feature_importance",
    "plot_model_comparison",
    "plot_upset_probability",
]
