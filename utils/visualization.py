"""Visualization tools for March Madness Predictor.

All plots use matplotlib/seaborn and accept an optional ``save_path``
argument.  When ``save_path`` is provided the figure is saved to disk;
otherwise it is displayed interactively.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from config import ROUNDS


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _save_or_show(fig: plt.Figure, save_path: Optional[str] = None) -> None:
    """Save the figure to *save_path* if provided, else call ``plt.show``."""
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ------------------------------------------------------------------
# Bracket visualisation
# ------------------------------------------------------------------

def plot_bracket(
    results: dict,
    save_path: Optional[str] = None,
) -> None:
    """Draw a simplified tournament bracket from simulation *results*.

    Parameters
    ----------
    results : dict
        Expected keys per round (matching ``config.ROUNDS``), each mapping
        to a list of winner team-name strings.  Example::

            {
                "Round of 64": ["Duke", "UConn", ...],
                "Round of 32": ["Duke", ...],
                ...
                "Championship": ["Duke"],
            }
    save_path : str, optional
        File path to save the figure.  If ``None`` the plot is displayed
        interactively.
    """
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title("March Madness Predicted Bracket", fontsize=18, fontweight="bold", pad=20)

    # Layout: columns for each round
    round_keys = [r for r in ROUNDS if r in results and results[r]]
    n_rounds = len(round_keys)
    if n_rounds == 0:
        ax.text(50, 50, "No results to display", ha="center", va="center", fontsize=14)
        _save_or_show(fig, save_path)
        return

    col_width = 90 / n_rounds

    for col_idx, round_name in enumerate(round_keys):
        winners = results[round_name]
        n_teams = len(winners)
        x = 5 + col_idx * col_width
        spacing = 85 / max(n_teams, 1)

        # Round header
        ax.text(
            x + col_width / 2, 97, round_name,
            ha="center", va="top", fontsize=8, fontweight="bold",
            color="#333333",
        )

        for i, team in enumerate(winners):
            y = 90 - i * spacing
            box = mpatches.FancyBboxPatch(
                (x, y - 1.5), col_width * 0.9, 3,
                boxstyle="round,pad=0.3",
                facecolor="#e8f4fd" if col_idx % 2 == 0 else "#fff3e0",
                edgecolor="#666666",
                linewidth=0.5,
            )
            ax.add_patch(box)
            ax.text(
                x + col_width * 0.45, y,
                team, ha="center", va="center",
                fontsize=max(5, 9 - n_teams // 16),
            )

    fig.tight_layout()
    _save_or_show(fig, save_path)


# ------------------------------------------------------------------
# Championship win probabilities
# ------------------------------------------------------------------

def plot_win_probabilities(
    teams: list[str],
    probs: list[float],
    save_path: Optional[str] = None,
) -> None:
    """Horizontal bar chart of championship (or round-win) probabilities.

    Parameters
    ----------
    teams : list[str]
        Team names.
    probs : list[float]
        Corresponding probabilities (0-1 scale).
    save_path : str, optional
        File path to save the figure.
    """
    # Sort by probability descending
    order = np.argsort(probs)[::-1]
    sorted_teams = [teams[i] for i in order]
    sorted_probs = [probs[i] for i in order]

    n_teams = len(sorted_teams)
    fig_height = max(6, n_teams * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    palette = sns.color_palette("viridis", n_colors=n_teams)
    bars = ax.barh(range(n_teams), sorted_probs, color=palette)

    ax.set_yticks(range(n_teams))
    ax.set_yticklabels(sorted_teams)
    ax.invert_yaxis()
    ax.set_xlabel("Win Probability")
    ax.set_title("Championship Win Probabilities", fontsize=14, fontweight="bold")

    # Annotate bars with percentage text
    for bar, prob in zip(bars, sorted_probs):
        width = bar.get_width()
        ax.text(
            width + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{prob:.1%}", va="center", fontsize=8,
        )

    ax.set_xlim(0, max(sorted_probs) * 1.15 if sorted_probs else 1)
    sns.despine(left=True, bottom=False)
    fig.tight_layout()
    _save_or_show(fig, save_path)


# ------------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------------

def plot_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 25,
    save_path: Optional[str] = None,
) -> None:
    """Plot the most important features from a trained model.

    Parameters
    ----------
    model : BaseMarchMadnessModel
        A fitted model that implements ``get_feature_importance()``.
    feature_names : list[str]
        Feature names aligned with the importance array.
    top_n : int
        Number of top features to display (default 25).
    save_path : str, optional
        File path to save the figure.
    """
    importances = model.get_feature_importance()
    if importances is None:
        print(f"[visualization] {model.name} does not provide feature importances.")
        return

    importances = np.asarray(importances)
    if len(importances) != len(feature_names):
        print(
            f"[visualization] Mismatch: {len(importances)} importances vs "
            f"{len(feature_names)} feature names."
        )
        return

    # Top-N by absolute value
    top_idx = np.argsort(np.abs(importances))[::-1][:top_n]
    top_names = [feature_names[i] for i in top_idx]
    top_values = importances[top_idx]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in top_values]
    ax.barh(range(len(top_names)), top_values, color=colors)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(
        f"Top {top_n} Feature Importances — {model.name}",
        fontsize=14, fontweight="bold",
    )
    sns.despine(left=True, bottom=False)
    fig.tight_layout()
    _save_or_show(fig, save_path)


# ------------------------------------------------------------------
# Model comparison (radar chart)
# ------------------------------------------------------------------

def plot_model_comparison(
    comparison_df,
    save_path: Optional[str] = None,
) -> None:
    """Radar chart comparing models across key metrics.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame produced by :func:`utils.validation.compare_models`
        with columns ``model``, ``mean_accuracy``, ``mean_log_loss``,
        ``mean_auc``.
    save_path : str, optional
        File path to save the figure.
    """
    metrics = ["mean_accuracy", "mean_auc"]
    # Invert log_loss so that higher is better on the radar
    if "mean_log_loss" in comparison_df.columns:
        comparison_df = comparison_df.copy()
        comparison_df["inv_log_loss"] = 1 - comparison_df["mean_log_loss"]
        metrics.append("inv_log_loss")

    labels = [m.replace("_", " ").title() for m in metrics]
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    palette = sns.color_palette("Set2", n_colors=len(comparison_df))

    for idx, (_, row) in enumerate(comparison_df.iterrows()):
        values = [row[m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row["model"], color=palette[idx])
        ax.fill(angles, values, alpha=0.15, color=palette[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.tight_layout()
    _save_or_show(fig, save_path)


# ------------------------------------------------------------------
# Upset probability heatmap
# ------------------------------------------------------------------

def plot_upset_probability(
    matchups: list[dict],
    save_path: Optional[str] = None,
) -> None:
    """Heatmap of upset probabilities for a set of matchups.

    Parameters
    ----------
    matchups : list[dict]
        Each dict should have keys ``higher_seed`` (str), ``lower_seed``
        (str), and ``upset_prob`` (float, probability the lower-seeded
        team wins).
    save_path : str, optional
        File path to save the figure.
    """
    if not matchups:
        print("[visualization] No matchups to display.")
        return

    labels = [
        f"{m['higher_seed']} vs {m['lower_seed']}" for m in matchups
    ]
    probs = [m["upset_prob"] for m in matchups]

    # Sort by upset probability descending
    order = np.argsort(probs)[::-1]
    labels = [labels[i] for i in order]
    probs_sorted = np.array([probs[i] for i in order])

    # Build a 2-D array for the heatmap (single column)
    data = probs_sorted.reshape(-1, 1)

    fig_height = max(5, len(labels) * 0.45)
    fig, ax = plt.subplots(figsize=(6, fig_height))

    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    sns.heatmap(
        data,
        annot=True,
        fmt=".1%",
        cmap=cmap,
        yticklabels=labels,
        xticklabels=["Upset Probability"],
        cbar_kws={"label": "Probability"},
        vmin=0,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title("Upset Probabilities", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_or_show(fig, save_path)
