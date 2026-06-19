from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_detection_corrected_mdf import (  # noqa: E402
    DEFAULT_VARIANT,
    _interpolate_selection_for_catalog,
    _kde_density,
    _load_surface_archives,
    _normalized_curve,
    _normalized_density,
    _weighted_histogram_density,
)


DEFAULT_CATALOG = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog_with_origin_and_chemistry.csv"
DEFAULT_AGES = PROJECT_ROOT / "data" / "processed" / "vandenberg2013_gc_ages.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "tables"
DEFAULT_FIGURE_PDF = PROJECT_ROOT / "paper" / "figures" / "detection_corrected_age_distribution.pdf"
DEFAULT_FIGURE_PNG = PROJECT_ROOT / "paper" / "figures" / "detection_corrected_age_distribution.png"
DEFAULT_KDE_FIGURE_PDF = PROJECT_ROOT / "paper" / "figures" / "detection_corrected_age_distribution_kde.pdf"
DEFAULT_KDE_FIGURE_PNG = PROJECT_ROOT / "paper" / "figures" / "detection_corrected_age_distribution_kde.png"


def _merge_age_catalog(catalog: pd.DataFrame, ages: pd.DataFrame) -> pd.DataFrame:
    required_age_columns = ["catalog_match_key", "age_gyr", "age_error_gyr"]
    missing = [column for column in required_age_columns if column not in ages.columns]
    if missing:
        raise ValueError(f"Age table is missing required columns: {missing}")
    required_catalog_columns = ["cluster_name_key", "initial_mass_msun", "semi_major_axis_kpc"]
    missing = [column for column in required_catalog_columns if column not in catalog.columns]
    if missing:
        raise ValueError(f"Catalogue is missing required columns: {missing}")
    merged = ages.merge(catalog, left_on="catalog_match_key", right_on="cluster_name_key", how="left", indicator=True)
    unmatched = merged.loc[merged["_merge"] != "both", ["ngc", "name", "catalog_match_key"]]
    if not unmatched.empty:
        raise ValueError("Some VandenBerg age rows did not match the catalogue:\n" + unmatched.to_string(index=False))
    finite = (
        np.isfinite(merged["age_gyr"])
        & np.isfinite(merged["initial_mass_msun"])
        & np.isfinite(merged["semi_major_axis_kpc"])
        & (merged["initial_mass_msun"] > 0.0)
        & (merged["semi_major_axis_kpc"] > 0.0)
    )
    return merged.loc[finite].copy().reset_index(drop=True)


def _age_weight_table(catalog: pd.DataFrame, selection_samples: np.ndarray, clipped_log_mass: np.ndarray, clipped_log_a: np.ndarray) -> pd.DataFrame:
    birth_weights = 1.0 / selection_samples
    missing_weights = birth_weights - 1.0
    output = catalog.copy()
    output["log10_initial_mass_for_selection"] = clipped_log_mass
    output["log10_semimajor_axis_for_selection"] = clipped_log_a
    output["selection_coordinate_was_clipped"] = (
        ~np.isclose(np.log10(output["initial_mass_msun"].to_numpy(dtype=float)), clipped_log_mass)
        | ~np.isclose(np.log10(output["semi_major_axis_kpc"].to_numpy(dtype=float)), clipped_log_a)
    )
    for prefix, values in (
        ("selection_probability", selection_samples),
        ("birth_weight", birth_weights),
        ("missing_weight", missing_weights),
    ):
        output[f"{prefix}_q16"] = np.nanquantile(values, 0.16, axis=0)
        output[f"{prefix}_q50"] = np.nanquantile(values, 0.50, axis=0)
        output[f"{prefix}_q84"] = np.nanquantile(values, 0.84, axis=0)
    keep_columns = [
        "cluster_label",
        "cluster_name",
        "origin_label",
        "progenitor_group",
        "catalog_match_key",
        "age_gyr",
        "age_error_gyr",
        "age_method",
        "vandenberg_feh",
        "initial_mass_msun",
        "present_mass_msun",
        "semi_major_axis_kpc",
        "log10_initial_mass_for_selection",
        "log10_semimajor_axis_for_selection",
        "selection_coordinate_was_clipped",
        "selection_probability_q16",
        "selection_probability_q50",
        "selection_probability_q84",
        "birth_weight_q16",
        "birth_weight_q50",
        "birth_weight_q84",
        "missing_weight_q16",
        "missing_weight_q50",
        "missing_weight_q84",
    ]
    return output.loc[:, [column for column in keep_columns if column in output.columns]]


def _build_histogram_table(values: np.ndarray, selection_samples: np.ndarray, edges: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    observed_density = _weighted_histogram_density(values, np.ones_like(values), edges)
    observed_normalized = _normalized_density(observed_density, edges)

    birth_densities = []
    missing_densities = []
    for selection in selection_samples:
        birth_weight = 1.0 / selection
        missing_weight = birth_weight - 1.0
        birth_densities.append(_weighted_histogram_density(values, birth_weight, edges))
        missing_densities.append(_weighted_histogram_density(values, missing_weight, edges))
    birth_densities = np.asarray(birth_densities, dtype=float)
    missing_densities = np.asarray(missing_densities, dtype=float)
    birth_norm = np.asarray([_normalized_density(row, edges) for row in birth_densities])
    missing_norm = np.asarray([_normalized_density(row, edges) for row in missing_densities])

    rows = []
    centers = 0.5 * (edges[:-1] + edges[1:])
    for index, center in enumerate(centers):
        row = {
            "age_bin_index": int(index),
            "age_left_gyr": float(edges[index]),
            "age_right_gyr": float(edges[index + 1]),
            "age_center_gyr": float(center),
            "observed_density": float(observed_density[index]),
            "observed_normalized_density": float(observed_normalized[index]),
        }
        for prefix, table in (
            ("birth_corrected_density", birth_densities),
            ("missing_density", missing_densities),
            ("birth_corrected_normalized_density", birth_norm),
            ("missing_normalized_density", missing_norm),
        ):
            row[f"{prefix}_q16"] = float(np.nanquantile(table[:, index], 0.16))
            row[f"{prefix}_q50"] = float(np.nanquantile(table[:, index], 0.50))
            row[f"{prefix}_q84"] = float(np.nanquantile(table[:, index], 0.84))
        rows.append(row)

    totals = {
        "n_observed_with_vandenberg_age": float(len(values)),
        "birth_corrected_total_q16": float(np.nanquantile(np.sum(1.0 / selection_samples, axis=1), 0.16)),
        "birth_corrected_total_q50": float(np.nanquantile(np.sum(1.0 / selection_samples, axis=1), 0.50)),
        "birth_corrected_total_q84": float(np.nanquantile(np.sum(1.0 / selection_samples, axis=1), 0.84)),
        "missing_total_q16": float(np.nanquantile(np.sum(1.0 / selection_samples - 1.0, axis=1), 0.16)),
        "missing_total_q50": float(np.nanquantile(np.sum(1.0 / selection_samples - 1.0, axis=1), 0.50)),
        "missing_total_q84": float(np.nanquantile(np.sum(1.0 / selection_samples - 1.0, axis=1), 0.84)),
    }
    return pd.DataFrame(rows), totals


def _build_kde_table(values: np.ndarray, selection_samples: np.ndarray, x_grid: np.ndarray, bandwidth_scale: float) -> pd.DataFrame:
    observed_density = _kde_density(values, np.ones_like(values), x_grid, bandwidth_scale)
    observed_normalized = _normalized_curve(observed_density, x_grid)

    birth_densities = []
    missing_densities = []
    for selection in selection_samples:
        birth_weight = 1.0 / selection
        missing_weight = birth_weight - 1.0
        birth_densities.append(_kde_density(values, birth_weight, x_grid, bandwidth_scale))
        missing_densities.append(_kde_density(values, missing_weight, x_grid, bandwidth_scale))
    birth_densities = np.asarray(birth_densities, dtype=float)
    missing_densities = np.asarray(missing_densities, dtype=float)
    birth_norm = np.asarray([_normalized_curve(row, x_grid) for row in birth_densities])
    missing_norm = np.asarray([_normalized_curve(row, x_grid) for row in missing_densities])

    rows = []
    for index, age in enumerate(x_grid):
        row = {
            "age_gyr": float(age),
            "observed_density": float(observed_density[index]),
            "observed_normalized_density": float(observed_normalized[index]),
        }
        for prefix, table in (
            ("birth_corrected_density", birth_densities),
            ("missing_density", missing_densities),
            ("birth_corrected_normalized_density", birth_norm),
            ("missing_normalized_density", missing_norm),
        ):
            row[f"{prefix}_q16"] = float(np.nanquantile(table[:, index], 0.16))
            row[f"{prefix}_q50"] = float(np.nanquantile(table[:, index], 0.50))
            row[f"{prefix}_q84"] = float(np.nanquantile(table[:, index], 0.84))
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_distribution(
    table: pd.DataFrame,
    totals: dict[str, float],
    output_pdf: Path,
    output_png: Path,
    *,
    is_kde: bool,
    bandwidth_scale: float,
) -> None:
    observed_color = "#222222"
    birth_color = "#2f6fbb"
    missing_color = "#c44e52"
    if is_kde:
        x = table["age_gyr"].to_numpy(dtype=float)
        observed = table["observed_density"].to_numpy(dtype=float)
        observed_norm = table["observed_normalized_density"].to_numpy(dtype=float)
        draw_observed = lambda axis, y, label: axis.plot(x, y, color=observed_color, linewidth=1.8, label=label)
        title_prefix = "KDE"
    else:
        x = table["age_center_gyr"].to_numpy(dtype=float)
        edges = np.concatenate(
            [table["age_left_gyr"].to_numpy(dtype=float), [float(table["age_right_gyr"].iloc[-1])]]
        )
        observed = table["observed_density"].to_numpy(dtype=float)
        observed_norm = table["observed_normalized_density"].to_numpy(dtype=float)
        draw_observed = lambda axis, y, label: axis.stairs(y, edges, color=observed_color, linewidth=1.8, label=label)
        title_prefix = "Binned"

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), constrained_layout=True)
    draw_observed(axes[0], observed, f"Observed survivors ({int(totals['n_observed_with_vandenberg_age'])})")
    axes[0].plot(x, table["birth_corrected_density_q50"], color=birth_color, linewidth=2.1, label="Birth-corrected")
    axes[0].fill_between(
        x,
        table["birth_corrected_density_q16"],
        table["birth_corrected_density_q84"],
        color=birth_color,
        alpha=0.18,
        linewidth=0,
    )
    axes[0].plot(x, table["missing_density_q50"], color=missing_color, linewidth=1.8, ls="--", label="Destroyed only")
    axes[0].fill_between(
        x,
        table["missing_density_q16"],
        table["missing_density_q84"],
        color=missing_color,
        alpha=0.13,
        linewidth=0,
    )
    axes[0].set_yscale("log")
    axes[0].set_ylim(bottom=0.8)
    axes[0].set_xlabel("Age [Gyr]")
    axes[0].set_ylabel("dN/dAge [Gyr$^{-1}$]")
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[0].set_title("Absolute counts")

    draw_observed(axes[1], observed_norm, "Observed survivors")
    axes[1].plot(
        x,
        table["birth_corrected_normalized_density_q50"],
        color=birth_color,
        linewidth=2.1,
        label="Birth-corrected",
    )
    axes[1].fill_between(
        x,
        table["birth_corrected_normalized_density_q16"],
        table["birth_corrected_normalized_density_q84"],
        color=birth_color,
        alpha=0.18,
        linewidth=0,
    )
    axes[1].plot(
        x,
        table["missing_normalized_density_q50"],
        color=missing_color,
        linewidth=1.8,
        ls="--",
        label="Destroyed only",
    )
    axes[1].fill_between(
        x,
        table["missing_normalized_density_q16"],
        table["missing_normalized_density_q84"],
        color=missing_color,
        alpha=0.13,
        linewidth=0,
    )
    axes[1].set_xlabel("Age [Gyr]")
    axes[1].set_ylabel("Probability density")
    axes[1].set_title("Normalized age distribution")
    axes[1].legend(frameon=False, fontsize=8.5)

    for axis in axes:
        axis.set_xlim(float(x.min()), float(x.max()))
        axis.grid(alpha=0.18, linewidth=0.7)

    if is_kde:
        subtitle = (
            rf"{title_prefix}, VandenBerg et al. ages, "
            rf"$w_{{\rm birth}}=1/[S Q]$, bandwidth scale $={bandwidth_scale:.2f}$"
        )
    else:
        subtitle = rf"{title_prefix}, VandenBerg et al. ages, $w_{{\rm birth}}=1/[S Q]$"
    fig.suptitle(subtitle, fontsize=10)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--ages", type=Path, default=DEFAULT_AGES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_FIGURE_PDF)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_FIGURE_PNG)
    parser.add_argument("--kde-output-pdf", type=Path, default=DEFAULT_KDE_FIGURE_PDF)
    parser.add_argument("--kde-output-png", type=Path, default=DEFAULT_KDE_FIGURE_PNG)
    parser.add_argument("--age-min", type=float, default=8.5)
    parser.add_argument("--age-max", type=float, default=13.5)
    parser.add_argument("--age-bin-width", type=float, default=0.5)
    parser.add_argument("--kde-n-grid", type=int, default=500)
    parser.add_argument("--kde-bandwidth-scale", type=float, default=0.85)
    parser.add_argument("--max-surface-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--selection-floor", type=float, default=1.0e-4)
    args = parser.parse_args()

    catalog = pd.read_csv(args.catalog)
    ages = pd.read_csv(args.ages)
    working = _merge_age_catalog(catalog, ages)

    variant_root = PROJECT_ROOT / "variants" / args.variant
    log_mass_grid, log_a_grid, surface_refs, surface_metadata = _load_surface_archives(
        variant_root=variant_root,
        max_surface_samples=None if int(args.max_surface_samples) <= 0 else int(args.max_surface_samples),
        seed=int(args.seed),
    )
    points = np.column_stack(
        [
            np.log10(working["initial_mass_msun"].to_numpy(dtype=float)),
            np.log10(working["semi_major_axis_kpc"].to_numpy(dtype=float)),
        ]
    )
    clipped_points = points.copy()
    clipped_points[:, 0] = np.clip(clipped_points[:, 0], float(log_mass_grid.min()), float(log_mass_grid.max()))
    clipped_points[:, 1] = np.clip(clipped_points[:, 1], float(log_a_grid.min()), float(log_a_grid.max()))
    selection_samples = _interpolate_selection_for_catalog(
        log_mass_grid=log_mass_grid,
        log_a_grid=log_a_grid,
        surface_refs=surface_refs,
        points=clipped_points,
        floor=float(args.selection_floor),
    )
    if not np.isfinite(selection_samples).all():
        bad = np.argwhere(~np.isfinite(selection_samples))
        raise RuntimeError(f"Non-finite interpolated selection probabilities, first bad index={bad[0].tolist()}")

    values = working["age_gyr"].to_numpy(dtype=float)
    edges = np.arange(float(args.age_min), float(args.age_max) + 0.5 * float(args.age_bin_width), float(args.age_bin_width))
    histogram_table, totals = _build_histogram_table(values, selection_samples, edges)
    kde_grid = np.linspace(float(args.age_min), float(args.age_max), int(args.kde_n_grid))
    kde_table = _build_kde_table(values, selection_samples, kde_grid, bandwidth_scale=float(args.kde_bandwidth_scale))
    cluster_weights = _age_weight_table(
        working,
        selection_samples,
        clipped_log_mass=clipped_points[:, 0],
        clipped_log_a=clipped_points[:, 1],
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    histogram_path = args.output_root / "detection_corrected_age_distribution.csv"
    kde_path = args.output_root / "detection_corrected_age_distribution_kde.csv"
    weights_path = args.output_root / "detection_corrected_age_distribution_cluster_weights.csv"
    surface_meta_path = args.output_root / "detection_corrected_age_distribution_surface_samples.csv"
    summary_path = args.output_root / "detection_corrected_age_distribution_summary.json"
    histogram_table.to_csv(histogram_path, index=False)
    kde_table.to_csv(kde_path, index=False)
    cluster_weights.to_csv(weights_path, index=False)
    surface_metadata.to_csv(surface_meta_path, index=False)

    summary = {
        "variant": str(args.variant),
        "catalog": str(args.catalog),
        "ages": str(args.ages),
        "n_catalog_rows": int(len(catalog)),
        "n_vandenberg_age_rows": int(len(ages)),
        "n_matched_age_rows": int(len(working)),
        "n_surface_samples": int(len(surface_refs)),
        "selection_floor": float(args.selection_floor),
        "n_selection_coordinates_clipped": int(np.any(~np.isclose(points, clipped_points), axis=1).sum()),
        "age_edges_gyr": [float(value) for value in edges],
        "kde_n_grid": int(args.kde_n_grid),
        "kde_bandwidth_scale": float(args.kde_bandwidth_scale),
        "totals": totals,
        "outputs": {
            "histogram_table": str(histogram_path),
            "kde_table": str(kde_path),
            "cluster_weights": str(weights_path),
            "surface_samples": str(surface_meta_path),
            "figure_pdf": str(args.output_pdf),
            "figure_png": str(args.output_png),
            "kde_figure_pdf": str(args.kde_output_pdf),
            "kde_figure_png": str(args.kde_output_png),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    _plot_distribution(
        histogram_table,
        totals,
        args.output_pdf,
        args.output_png,
        is_kde=False,
        bandwidth_scale=float(args.kde_bandwidth_scale),
    )
    _plot_distribution(
        kde_table,
        totals,
        args.kde_output_pdf,
        args.kde_output_png,
        is_kde=True,
        bandwidth_scale=float(args.kde_bandwidth_scale),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
