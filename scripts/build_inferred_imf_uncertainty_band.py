from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz


def schechter_dndlogm(log_mass: np.ndarray, alpha: float, log10_m_c_msun: float) -> np.ndarray:
    mass = np.power(10.0, log_mass)
    m_c = np.power(10.0, log10_m_c_msun)
    return np.power(mass, alpha + 1.0) * np.exp(-mass / m_c)


def normalize_above_threshold(
    log_mass: np.ndarray,
    shape: np.ndarray,
    total_count_above_threshold: float,
    threshold_log_mass: float,
) -> np.ndarray:
    mask = log_mass >= threshold_log_mass
    integral = float(np.trapezoid(shape[mask], log_mass[mask]))
    if not np.isfinite(integral) or integral <= 0.0:
        return np.zeros_like(shape)
    return shape * (total_count_above_threshold / integral)


def cumulative_from_differential(log_mass: np.ndarray, dndlogm: np.ndarray) -> np.ndarray:
    cumulative = np.zeros_like(dndlogm)
    for idx in range(len(log_mass)):
        cumulative[idx] = np.trapezoid(dndlogm[idx:], log_mass[idx:])
    return cumulative


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an inferred Schechter IMF plot with posterior uncertainty bands."
    )
    parser.add_argument(
        "--posterior-samples",
        type=Path,
        required=True,
        help="CSV containing posterior samples with alpha, log10 Mc, and N0(>1e4).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="PNG output path.",
    )
    parser.add_argument(
        "--threshold-log-mass",
        type=float,
        default=4.0,
        help="Lower mass threshold used for the reported total count normalization.",
    )
    parser.add_argument(
        "--log-mass-min",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--log-mass-max",
        type=float,
        default=7.3,
    )
    parser.add_argument(
        "--n-grid",
        type=int,
        default=500,
    )
    args = parser.parse_args()

    samples = pd.read_csv(args.posterior_samples)
    required_columns = [
        "input_alpha_dndm",
        "input_log10_m_c_msun",
        "final_total_initial_count_above_log10_4",
    ]
    missing = [column for column in required_columns if column not in samples.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    samples = samples.loc[:, required_columns].dropna()
    log_mass = np.linspace(args.log_mass_min, args.log_mass_max, args.n_grid)

    differential_curves = []
    cumulative_curves = []
    for row in samples.itertuples(index=False):
        shape = schechter_dndlogm(log_mass, row.input_alpha_dndm, row.input_log10_m_c_msun)
        dndlogm = normalize_above_threshold(
            log_mass,
            shape,
            float(row.final_total_initial_count_above_log10_4),
            args.threshold_log_mass,
        )
        cumulative = cumulative_from_differential(log_mass, dndlogm)
        differential_curves.append(dndlogm)
        cumulative_curves.append(cumulative)

    differential_array = np.asarray(differential_curves, dtype=float)
    cumulative_array = np.asarray(cumulative_curves, dtype=float)

    diff_q02, diff_q16, diff_q50, diff_q84, diff_q98 = np.quantile(
        differential_array, [0.025, 0.16, 0.5, 0.84, 0.975], axis=0
    )
    cum_q02, cum_q16, cum_q50, cum_q84, cum_q98 = np.quantile(
        cumulative_array, [0.025, 0.16, 0.5, 0.84, 0.975], axis=0
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)

    axes[0].fill_between(log_mass, diff_q02, diff_q98, color="#fdd0a2", alpha=0.35, linewidth=0.0)
    axes[0].fill_between(log_mass, diff_q16, diff_q84, color="#f16913", alpha=0.25, linewidth=0.0)
    axes[0].plot(log_mass, diff_q50, color="#b30000", lw=2.0)
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$\log_{10}(M_{\rm ini}/{\rm M}_\odot)$")
    axes[0].set_ylabel(r"$dN/d\log_{10} M_{\rm ini}$")
    axes[0].set_title("Differential IMF")

    axes[1].fill_between(log_mass, cum_q02, cum_q98, color="#c6dbef", alpha=0.45, linewidth=0.0)
    axes[1].fill_between(log_mass, cum_q16, cum_q84, color="#6baed6", alpha=0.35, linewidth=0.0)
    axes[1].plot(log_mass, cum_q50, color="#08519c", lw=2.0)
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$\log_{10}(M_{\rm ini}/{\rm M}_\odot)$")
    axes[1].set_ylabel(r"$N(>M_{\rm ini})$")
    axes[1].set_title("Cumulative IMF")

    for axis in axes:
        axis.set_xlim(args.log_mass_min, args.log_mass_max)
        axis.grid(alpha=0.15, linewidth=0.6)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
