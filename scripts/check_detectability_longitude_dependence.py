from __future__ import annotations

import json
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
from scipy import stats


def centers_to_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or len(centers) < 2:
        raise ValueError("Need at least two centers to reconstruct edges.")
    steps = np.diff(centers)
    left_edge = centers[0] - 0.5 * steps[0]
    right_edge = centers[-1] + 0.5 * steps[-1]
    interior = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate(([left_edge], interior, [right_edge]))


def assign_bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(edges, values, side="right") - 1
    return np.clip(indices, 0, len(edges) - 2)


def fit_longitude_harmonic_model(residual: np.ndarray, l_deg: np.ndarray) -> dict[str, float]:
    y = np.asarray(residual, dtype=float)
    l_rad = np.deg2rad(np.asarray(l_deg, dtype=float))
    x_null = np.ones((len(y), 1), dtype=float)
    x_full = np.column_stack([np.ones(len(y), dtype=float), np.sin(l_rad), np.cos(l_rad)])

    beta_null, _, _, _ = np.linalg.lstsq(x_null, y, rcond=None)
    beta_full, _, _, _ = np.linalg.lstsq(x_full, y, rcond=None)

    fitted_null = x_null @ beta_null
    fitted_full = x_full @ beta_full

    rss_null = float(np.sum((y - fitted_null) ** 2))
    rss_full = float(np.sum((y - fitted_full) ** 2))
    n_obs = len(y)
    k_null = x_null.shape[1]
    k_full = x_full.shape[1]

    if rss_full <= 0.0 or n_obs <= k_full:
        raise ValueError("Degenerate residual fit encountered.")

    df_num = k_full - k_null
    df_den = n_obs - k_full
    f_value = ((rss_null - rss_full) / df_num) / (rss_full / df_den)
    joint_pvalue = float(stats.f.sf(f_value, df_num, df_den))

    sigma2 = rss_full / df_den
    covariance = sigma2 * np.linalg.inv(x_full.T @ x_full)
    stderr = np.sqrt(np.diag(covariance))
    t_values = beta_full / stderr
    coefficient_pvalues = 2.0 * stats.t.sf(np.abs(t_values), df_den)

    amplitude = float(np.hypot(beta_full[1], beta_full[2]))
    phase_peak_deg = float((np.degrees(np.arctan2(beta_full[1], beta_full[2])) + 360.0) % 360.0)
    r_squared = float(1.0 - rss_full / rss_null) if rss_null > 0.0 else 0.0

    bic_null = float(n_obs * np.log(rss_null / n_obs) + k_null * np.log(n_obs))
    bic_full = float(n_obs * np.log(rss_full / n_obs) + k_full * np.log(n_obs))

    pearson_sin = stats.pearsonr(y, np.sin(l_rad))
    pearson_cos = stats.pearsonr(y, np.cos(l_rad))

    return {
        "n_clusters": int(n_obs),
        "intercept": float(beta_full[0]),
        "sin_coefficient": float(beta_full[1]),
        "cos_coefficient": float(beta_full[2]),
        "sin_stderr": float(stderr[1]),
        "cos_stderr": float(stderr[2]),
        "sin_pvalue": float(coefficient_pvalues[1]),
        "cos_pvalue": float(coefficient_pvalues[2]),
        "joint_f_value": float(f_value),
        "joint_pvalue": joint_pvalue,
        "rss_null": rss_null,
        "rss_full": rss_full,
        "r_squared": r_squared,
        "bic_null": bic_null,
        "bic_full": bic_full,
        "delta_bic_full_minus_null": float(bic_full - bic_null),
        "harmonic_amplitude": amplitude,
        "harmonic_peak_longitude_deg": phase_peak_deg,
        "pearson_r_sin_l": float(pearson_sin.statistic),
        "pearson_p_sin_l": float(pearson_sin.pvalue),
        "pearson_r_cos_l": float(pearson_cos.statistic),
        "pearson_p_cos_l": float(pearson_cos.pvalue),
    }


def build_binned_longitude_summary(per_gc_table: pd.DataFrame, n_longitude_bins: int = 12) -> pd.DataFrame:
    longitude_edges = np.linspace(0.0, 360.0, n_longitude_bins + 1)
    working = per_gc_table.copy()
    wrapped_l = np.mod(working["galactic_l_deg"].to_numpy(dtype=float), 360.0)
    longitude_index = assign_bin_index(wrapped_l, longitude_edges)
    working["longitude_bin_index"] = longitude_index

    rows: list[dict[str, float]] = []
    for index in range(n_longitude_bins):
        subset = working.loc[working["longitude_bin_index"] == index]
        left = float(longitude_edges[index])
        right = float(longitude_edges[index + 1])
        residual = subset["detectability_bin_residual_sigma"].to_numpy(dtype=float)
        rows.append(
            {
                "longitude_bin_index": index,
                "longitude_left_deg": left,
                "longitude_right_deg": right,
                "longitude_center_deg": 0.5 * (left + right),
                "n_clusters": int(len(subset)),
                "mean_residual_sigma": float(np.mean(residual)) if len(residual) else np.nan,
                "sem_residual_sigma": float(np.std(residual, ddof=1) / np.sqrt(len(residual)))
                if len(residual) >= 2
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_longitude_residual_diagnostic(
    per_gc_table: pd.DataFrame,
    fit_summary: dict[str, float],
    binned_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    l_deg = np.mod(per_gc_table["galactic_l_deg"].to_numpy(dtype=float), 360.0)
    residual = per_gc_table["detectability_bin_residual_sigma"].to_numpy(dtype=float)
    l_model = np.linspace(0.0, 360.0, 500)
    l_model_rad = np.deg2rad(l_model)
    harmonic_curve = (
        fit_summary["intercept"]
        + fit_summary["sin_coefficient"] * np.sin(l_model_rad)
        + fit_summary["cos_coefficient"] * np.cos(l_model_rad)
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.scatter(
        l_deg,
        residual,
        s=24,
        alpha=0.45,
        color="#4c78a8",
        edgecolors="none",
        label="Per-GC assigned bin residual",
    )
    ax.errorbar(
        binned_summary["longitude_center_deg"],
        binned_summary["mean_residual_sigma"],
        yerr=binned_summary["sem_residual_sigma"],
        fmt="o",
        color="#d95f02",
        ms=5.0,
        capsize=2.5,
        label="Longitude-binned mean",
    )
    ax.plot(
        l_model,
        harmonic_curve,
        color="#1b9e77",
        linewidth=2.0,
        label="Best sin/cos longitude fit",
    )
    ax.set_xlim(0.0, 360.0)
    ax.set_xlabel(r"Galactic longitude $l$ [deg]")
    ax.set_ylabel(r"Detectability residual $\sigma$")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.text(
        0.02,
        0.03,
        (
            rf"$\Delta \mathrm{{BIC}}={fit_summary['delta_bic_full_minus_null']:.2f}$" "\n"
            rf"$p_{{\sin,\cos}}={fit_summary['joint_pvalue']:.3f}$" "\n"
            rf"$R^2={fit_summary['r_squared']:.3f}$"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2.0),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    project_root = PROJECT_ROOT
    outputs_tables = project_root / "outputs" / "tables"
    outputs_figures = project_root / "outputs" / "figures"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    outputs_figures.mkdir(parents=True, exist_ok=True)

    catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog.csv"
    histogram_path = outputs_tables / "joint_fixed_survival_detectability_em_observable_histogram.csv"
    completeness_path = outputs_tables / "joint_fixed_survival_detectability_em_catalog_completeness.csv"

    catalog = pd.read_csv(catalog_path)
    histogram = pd.read_csv(histogram_path)
    completeness = pd.read_csv(completeness_path)

    log_mass_centers = np.sort(histogram["log10_present_mass_center_msun"].unique())
    log_distance_centers = np.sort(np.log10(histogram["distance_center_kpc"].unique()))
    abs_latitude_centers = np.sort(histogram["abs_latitude_center_deg"].unique())

    log_mass_edges = centers_to_edges(log_mass_centers)
    log_distance_edges = centers_to_edges(log_distance_centers)
    abs_latitude_edges = centers_to_edges(abs_latitude_centers)

    bin_table = histogram.copy()
    expected = np.clip(bin_table["predicted_observed_count"].to_numpy(dtype=float), 1.0e-6, None)
    observed = bin_table["observed_count"].to_numpy(dtype=float)
    bin_table["detectability_bin_residual_sigma"] = (observed - expected) / np.sqrt(expected)
    bin_table["detectability_bin_residual_count"] = observed - expected

    completeness_with_l = completeness.merge(
        catalog[["cluster_name", "galactic_l_deg", "galactic_b_deg"]],
        on="cluster_name",
        how="left",
        validate="one_to_one",
    )
    if completeness_with_l["galactic_l_deg"].isna().any():
        missing = completeness_with_l.loc[completeness_with_l["galactic_l_deg"].isna(), "cluster_name"].tolist()
        missing_names = ", ".join(map(str, missing[:5]))
        raise ValueError(f"Failed to match Galactic longitude for some clusters, e.g. {missing_names}")

    log_distance = np.log10(completeness_with_l["r_sun_kpc"].to_numpy(dtype=float))
    completeness_with_l["present_mass_bin_index"] = assign_bin_index(
        completeness_with_l["log10_present_mass_msun"].to_numpy(dtype=float),
        log_mass_edges,
    )
    completeness_with_l["distance_bin_index"] = assign_bin_index(log_distance, log_distance_edges)
    completeness_with_l["latitude_bin_index"] = assign_bin_index(
        completeness_with_l["abs_galactic_b_deg"].to_numpy(dtype=float),
        abs_latitude_edges,
    )

    bin_lookup = bin_table[
        [
            "present_mass_bin_index",
            "distance_bin_index",
            "latitude_bin_index",
            "observed_count",
            "predicted_observed_count",
            "predicted_complete_count",
            "completeness",
            "detectability_bin_residual_sigma",
            "detectability_bin_residual_count",
        ]
    ]
    per_gc = completeness_with_l.merge(
        bin_lookup,
        on=["present_mass_bin_index", "distance_bin_index", "latitude_bin_index"],
        how="left",
        validate="many_to_one",
    )

    fit_summary = fit_longitude_harmonic_model(
        residual=per_gc["detectability_bin_residual_sigma"].to_numpy(dtype=float),
        l_deg=per_gc["galactic_l_deg"].to_numpy(dtype=float),
    )
    binned_summary = build_binned_longitude_summary(per_gc)

    per_gc_output = outputs_tables / "detectability_longitude_residuals_per_gc.csv"
    bin_output = outputs_tables / "detectability_longitude_residuals_by_l_bin.csv"
    summary_output = outputs_tables / "detectability_longitude_residuals_summary.json"
    figure_output = outputs_figures / "detectability_longitude_residuals_vs_l.png"

    per_gc.to_csv(per_gc_output, index=False)
    binned_summary.to_csv(bin_output, index=False)
    summary_output.write_text(json.dumps(fit_summary, indent=2))
    plot_longitude_residual_diagnostic(per_gc, fit_summary, binned_summary, figure_output)

    print(f"Saved {per_gc_output}")
    print(f"Saved {bin_output}")
    print(f"Saved {summary_output}")
    print(f"Saved {figure_output}")
    print(json.dumps(fit_summary, indent=2))


if __name__ == "__main__":
    main()
