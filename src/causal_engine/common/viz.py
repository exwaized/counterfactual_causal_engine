"""Shared plotting. Every module saves figures the same way so results/
directories look consistent regardless of which method produced them."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — pipelines save PNGs, they don't pop windows
import matplotlib.pyplot as plt
import numpy as np


def plot_qini_curve(qini_x: np.ndarray, qini_y: np.ndarray, random_y: np.ndarray, auuc: float, save_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(qini_x, qini_y, label="uplift model", color="#2A6F97")
    ax.plot(qini_x, random_y, label="random targeting", color="#999999", linestyle="--")
    ax.set_xlabel("Fraction of population targeted")
    ax.set_ylabel("Cumulative incremental outcome")
    ax.set_title(f"Qini curve (AUUC={auuc:.3f})")
    ax.legend()
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_ope_comparison(comparison_df, true_value: float, save_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    y_pos = range(len(comparison_df))
    points = comparison_df["point_estimate"].values
    err_low = points - comparison_df["ci_lower"].values
    err_high = comparison_df["ci_upper"].values - points
    ax.errorbar(points, y_pos, xerr=[err_low, err_high], fmt="o", color="#2A6F97", capsize=4, label="estimate (95% CI)")
    ax.axvline(true_value, color="#B23A48", linestyle="--", label="true value")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(comparison_df["estimator"].values)
    ax.set_xlabel("Estimated policy value")
    ax.set_title("Off-policy value estimates vs. ground truth")
    ax.legend()
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_geo_trajectory(periods: np.ndarray, actual: np.ndarray, synthetic: np.ndarray, n_pre: int, save_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(periods, actual, label="treated geo (actual)", color="#2A6F97")
    ax.plot(periods, synthetic, label="synthetic control", color="#999999", linestyle="--")
    ax.axvline(periods[n_pre], color="black", linestyle=":", linewidth=1, label="campaign start")
    ax.set_xlabel("Period")
    ax.set_ylabel("Outcome")
    ax.set_title("Treated geo vs. synthetic control")
    ax.legend()
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
