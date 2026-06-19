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
from scipy import stats

from build_detection_corrected_age_distribution import DEFAULT_AGES, DEFAULT_CATALOG, _merge_age_catalog  # noqa: E402
from build_detection_corrected_mdf import DEFAULT_VARIANT, _interpolate_selection_for_catalog, _load_surface_archives  # noqa: E402


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "tables"
DEFAULT_FIGURE_PDF = PROJECT_ROOT / "paper" / "figures" / "detection_corrected_age_metallicity.pdf"
DEFAULT_FIGURE_PNG = PROJECT_ROOT / "paper" / "figures" / "detection_corrected_age_metallicity.png"
DEFAULT_SPLIT_FIGURE_PDF = PROJECT_ROOT / "paper" / "figures" / "detection_corrected_age_metallicity_by_origin.pdf"
DEFAULT_SPLIT_FIGURE_PNG = PROJECT_ROOT / "paper" / "figures" / "detection_corrected_age_metallicity_by_origin.png"


def _weighted_kde_2d(
    feh: np.ndarray,
    age: np.ndarray,
    weights: np.ndarray,
    feh_grid: np.ndarray,
    age_grid: np.ndarray,
    bandwidth_scale: float,
) -> np.ndarray:
    values = np.vstack([np.asarray(feh, dtype=float), np.asarray(age, dtype=float)])
    weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(values).all(axis=0) & np.isfinite(weights) & (weights > 0.0)
    if finite.sum() < 3:
        return np.full((len(age_grid), len(feh_grid)), np.nan, dtype=float)

    scale = float(bandwidth_scale)

    def bw_method(kde: stats.gaussian_kde) -> float:
        return kde.scotts_factor() * scale

    kde = stats.gaussian_kde(values[:, finite], weights=weights[finite], bw_method=bw_method)
    xx, yy = np.meshgrid(feh_grid, age_grid)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    integral = float(np.trapezoid(np.trapezoid(density, feh_grid, axis=1), age_grid))
    if np.isfinite(integral) and integral > 0.0:
        density = density / integral
    return density


def _contour_levels(densities: list[np.ndarray], n_levels: int = 9) -> np.ndarray:
    finite = np.concatenate([density[np.isfinite(density) & (density > 0.0)] for density in densities])
    if finite.size == 0:
        return np.linspace(0.0, 1.0, n_levels)
    high = float(np.nanquantile(finite, 0.995))
    low = max(float(np.nanquantile(finite, 0.20)), high * 0.04)
    return np.linspace(low, high, n_levels)


def _enclosed_fraction_level(density: np.ndarray, fraction: float) -> float:
    values = np.asarray(density, dtype=float)
    finite = values[np.isfinite(values) & (values > 0.0)]
    if finite.size == 0:
        return np.nan
    sorted_values = np.sort(finite)[::-1]
    cumulative = np.cumsum(sorted_values)
    total = float(cumulative[-1])
    if not np.isfinite(total) or total <= 0.0:
        return np.nan
    target = np.clip(float(fraction), 0.0, 1.0)
    index = int(np.searchsorted(cumulative / total, target, side="left"))
    return float(sorted_values[min(index, sorted_values.size - 1)])


def _plot_age_metallicity(
    *,
    table: pd.DataFrame,
    observed_density: np.ndarray,
    corrected_density: np.ndarray,
    age_grid: np.ndarray,
    feh_grid: np.ndarray,
    corrected_total: float,
    output_pdf: Path,
    output_png: Path,
    bandwidth_scale: float,
    metallicity_column: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True, sharex=True, sharey=True)
    levels = _contour_levels([observed_density, corrected_density])
    cmap = "Blues"
    origin_colors = {"in_situ": "#2f6fbb", "accreted": "#c44e52"}
    point_edge = "white"

    panels = [
        (axes[0], observed_density, "Observed survivors", np.ones(len(table), dtype=float), 55.0),
        (axes[1], corrected_density, "Birth-corrected", table["birth_weight_q50"].to_numpy(dtype=float), corrected_total),
    ]
    for axis, density, title, point_weights, total in panels:
        image = axis.contourf(feh_grid, age_grid, density, levels=levels, cmap=cmap, alpha=0.92, extend="max")
        axis.contour(feh_grid, age_grid, density, levels=levels[2::2], colors="0.25", linewidths=0.45, alpha=0.55)
        for origin_label, subset in table.groupby("origin_label"):
            color = origin_colors.get(str(origin_label), "0.35")
            indices = subset.index.to_numpy()
            sizes = 24.0 if title == "Observed survivors" else 15.0 + 10.0 * np.sqrt(point_weights[indices])
            axis.scatter(
                subset[metallicity_column],
                subset["age_gyr"],
                s=sizes,
                color=color,
                edgecolor=point_edge,
                linewidth=0.45,
                alpha=0.88,
                label=str(origin_label).replace("_", " "),
                zorder=3,
            )
        axis.set_title(f"{title} ({total:.0f})")
        axis.set_xlabel("[Fe/H]")
        axis.grid(alpha=0.16, linewidth=0.7)

    axes[0].set_ylabel("Age [Gyr]")
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    axes[1].legend(frameon=False, fontsize=8.5, loc="lower left")
    axes[0].set_xlim(float(feh_grid.min()), float(feh_grid.max()))
    axes[0].set_ylim(float(age_grid.min()), float(age_grid.max()))
    cbar = fig.colorbar(image, ax=axes, shrink=0.90, pad=0.015)
    cbar.set_label("Normalized 2D density")
    fig.suptitle(
        rf"Age--metallicity density, VandenBerg ages, $w_{{\rm birth}}=1/[S Q]$, "
        rf"KDE bandwidth scale $={bandwidth_scale:.2f}$",
        fontsize=10,
    )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def _origin_densities(
    *,
    table: pd.DataFrame,
    weights: np.ndarray,
    feh_grid: np.ndarray,
    age_grid: np.ndarray,
    bandwidth_scale: float,
    metallicity_column: str,
) -> dict[str, np.ndarray]:
    weights = np.asarray(weights, dtype=float)
    total_weight = float(np.sum(weights))
    densities: dict[str, np.ndarray] = {}
    for origin_label, subset in table.groupby("origin_label", sort=False):
        indices = subset.index.to_numpy()
        density = _weighted_kde_2d(
            subset[metallicity_column].to_numpy(dtype=float),
            subset["age_gyr"].to_numpy(dtype=float),
            weights[indices],
            feh_grid,
            age_grid,
            bandwidth_scale=bandwidth_scale,
        )
        origin_weight = float(np.sum(weights[indices]))
        if np.isfinite(total_weight) and total_weight > 0.0:
            density = density * origin_weight / total_weight
        densities[str(origin_label)] = density
    return densities


def _plot_age_metallicity_by_origin(
    *,
    table: pd.DataFrame,
    age_grid: np.ndarray,
    feh_grid: np.ndarray,
    corrected_total: float,
    output_pdf: Path,
    output_png: Path,
    bandwidth_scale: float,
    metallicity_column: str,
    contour_fraction: float,
) -> dict[str, dict[str, np.ndarray]]:
    observed_weights = np.ones(len(table), dtype=float)
    corrected_weights = table["birth_weight_q50"].to_numpy(dtype=float)
    observed_densities = _origin_densities(
        table=table,
        weights=observed_weights,
        feh_grid=feh_grid,
        age_grid=age_grid,
        bandwidth_scale=bandwidth_scale,
        metallicity_column=metallicity_column,
    )
    corrected_densities = _origin_densities(
        table=table,
        weights=corrected_weights,
        feh_grid=feh_grid,
        age_grid=age_grid,
        bandwidth_scale=bandwidth_scale,
        metallicity_column=metallicity_column,
    )

    ordered_origins = [
        ("in_situ", "in situ", "#2f6fbb"),
        ("accreted", "accreted", "#e68a2e"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True, sharex=True, sharey=True)
    panels = [
        (axes[0], observed_densities, "Observed survivors", observed_weights, float(len(table))),
        (axes[1], corrected_densities, "Birth-corrected", corrected_weights, corrected_total),
    ]

    for axis, densities, title, point_weights, total in panels:
        for origin_label, label, color in ordered_origins:
            if origin_label not in densities:
                continue
            density = densities[origin_label]
            level = _enclosed_fraction_level(density, contour_fraction)
            if np.isfinite(level):
                axis.contour(
                    feh_grid,
                    age_grid,
                    density,
                    levels=[level],
                    colors=color,
                    linewidths=1.8,
                    alpha=0.98,
                )

            subset = table.loc[table["origin_label"] == origin_label]
            indices = subset.index.to_numpy()
            origin_total = float(np.sum(point_weights[indices]))
            sizes = 24.0 if title == "Observed survivors" else 15.0 + 10.0 * np.sqrt(point_weights[indices])
            axis.scatter(
                subset[metallicity_column],
                subset["age_gyr"],
                s=sizes,
                color=color,
                edgecolor="white",
                linewidth=0.45,
                alpha=0.90,
                label=f"{label} ({origin_total:.0f})",
                zorder=3,
            )
        axis.set_title(f"{title} ({total:.0f})")
        axis.set_xlabel("[Fe/H]")
        axis.grid(alpha=0.16, linewidth=0.7)
        axis.legend(frameon=False, fontsize=8.5, loc="lower left")

    axes[0].set_ylabel("Age [Gyr]")
    axes[0].set_xlim(float(feh_grid.min()), float(feh_grid.max()))
    axes[0].set_ylim(float(age_grid.min()), float(age_grid.max()))
    fig.suptitle(
        rf"Origin-split age--metallicity density, VandenBerg ages, "
        rf"{100.0 * contour_fraction:.0f}\% enclosed KDE contours, "
        rf"$w_{{\rm birth}}=1/[S Q]$, bandwidth scale $={bandwidth_scale:.2f}$",
        fontsize=10,
    )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)
    return {"observed": observed_densities, "birth_corrected": corrected_densities}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--ages", type=Path, default=DEFAULT_AGES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_FIGURE_PDF)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_FIGURE_PNG)
    parser.add_argument("--split-output-pdf", type=Path, default=DEFAULT_SPLIT_FIGURE_PDF)
    parser.add_argument("--split-output-png", type=Path, default=DEFAULT_SPLIT_FIGURE_PNG)
    parser.add_argument("--metallicity-column", choices=["vandenberg_feh", "local_feh"], default="vandenberg_feh")
    parser.add_argument("--age-min", type=float, default=8.5)
    parser.add_argument("--age-max", type=float, default=13.5)
    parser.add_argument("--feh-min", type=float, default=-2.6)
    parser.add_argument("--feh-max", type=float, default=0.0)
    parser.add_argument("--n-age-grid", type=int, default=180)
    parser.add_argument("--n-feh-grid", type=int, default=180)
    parser.add_argument("--kde-bandwidth-scale", type=float, default=0.90)
    parser.add_argument("--split-contour-fraction", type=float, default=0.50)
    parser.add_argument("--max-surface-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--selection-floor", type=float, default=1.0e-4)
    args = parser.parse_args()

    catalog = pd.read_csv(args.catalog)
    ages = pd.read_csv(args.ages)
    working = _merge_age_catalog(catalog, ages)
    if args.metallicity_column == "local_feh":
        finite_feh = np.isfinite(working["local_feh"]) & (working["local_feh"] > -10.0)
        working = working.loc[finite_feh].copy().reset_index(drop=True)
    working = working.loc[np.isfinite(working[str(args.metallicity_column)])].copy().reset_index(drop=True)
    if working.empty:
        raise ValueError("No age--metallicity rows remain after filtering.")

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
    birth_weights = 1.0 / selection_samples
    working["selection_probability_q16"] = np.nanquantile(selection_samples, 0.16, axis=0)
    working["selection_probability_q50"] = np.nanquantile(selection_samples, 0.50, axis=0)
    working["selection_probability_q84"] = np.nanquantile(selection_samples, 0.84, axis=0)
    working["birth_weight_q16"] = np.nanquantile(birth_weights, 0.16, axis=0)
    working["birth_weight_q50"] = np.nanquantile(birth_weights, 0.50, axis=0)
    working["birth_weight_q84"] = np.nanquantile(birth_weights, 0.84, axis=0)

    age_grid = np.linspace(float(args.age_min), float(args.age_max), int(args.n_age_grid))
    feh_grid = np.linspace(float(args.feh_min), float(args.feh_max), int(args.n_feh_grid))
    age = working["age_gyr"].to_numpy(dtype=float)
    feh = working[str(args.metallicity_column)].to_numpy(dtype=float)
    observed_density = _weighted_kde_2d(
        feh,
        age,
        np.ones_like(age),
        feh_grid,
        age_grid,
        bandwidth_scale=float(args.kde_bandwidth_scale),
    )
    corrected_density = _weighted_kde_2d(
        feh,
        age,
        working["birth_weight_q50"].to_numpy(dtype=float),
        feh_grid,
        age_grid,
        bandwidth_scale=float(args.kde_bandwidth_scale),
    )
    corrected_total = float(np.nanquantile(np.sum(birth_weights, axis=1), 0.50))

    args.output_root.mkdir(parents=True, exist_ok=True)
    cluster_path = args.output_root / "detection_corrected_age_metallicity_cluster_weights.csv"
    grid_path = args.output_root / "detection_corrected_age_metallicity_kde_grid.csv"
    split_grid_path = args.output_root / "detection_corrected_age_metallicity_by_origin_kde_grid.csv"
    surface_meta_path = args.output_root / "detection_corrected_age_metallicity_surface_samples.csv"
    summary_path = args.output_root / "detection_corrected_age_metallicity_summary.json"
    cluster_columns = [
        "cluster_label",
        "cluster_name",
        "origin_label",
        "progenitor_group",
        "age_gyr",
        "age_error_gyr",
        "vandenberg_feh",
        "local_feh",
        "initial_mass_msun",
        "semi_major_axis_kpc",
        "selection_probability_q16",
        "selection_probability_q50",
        "selection_probability_q84",
        "birth_weight_q16",
        "birth_weight_q50",
        "birth_weight_q84",
    ]
    working.loc[:, [column for column in cluster_columns if column in working.columns]].to_csv(cluster_path, index=False)
    surface_metadata.to_csv(surface_meta_path, index=False)
    xx, yy = np.meshgrid(feh_grid, age_grid)
    pd.DataFrame(
        {
            "feh": xx.ravel(),
            "age_gyr": yy.ravel(),
            "observed_density": observed_density.ravel(),
            "birth_corrected_density": corrected_density.ravel(),
        }
    ).to_csv(grid_path, index=False)
    split_densities = _plot_age_metallicity_by_origin(
        table=working,
        age_grid=age_grid,
        feh_grid=feh_grid,
        corrected_total=corrected_total,
        output_pdf=args.split_output_pdf,
        output_png=args.split_output_png,
        bandwidth_scale=float(args.kde_bandwidth_scale),
        metallicity_column=str(args.metallicity_column),
        contour_fraction=float(args.split_contour_fraction),
    )
    split_grid = {
        "feh": xx.ravel(),
        "age_gyr": yy.ravel(),
    }
    for panel_name, panel_densities in split_densities.items():
        for origin_label, density in panel_densities.items():
            split_grid[f"{panel_name}_{origin_label}_density"] = density.ravel()
    pd.DataFrame(split_grid).to_csv(split_grid_path, index=False)

    summary = {
        "variant": str(args.variant),
        "catalog": str(args.catalog),
        "ages": str(args.ages),
        "metallicity_column": str(args.metallicity_column),
        "n_age_metallicity_rows": int(len(working)),
        "n_surface_samples": int(len(surface_refs)),
        "selection_floor": float(args.selection_floor),
        "birth_corrected_total_q16": float(np.nanquantile(np.sum(birth_weights, axis=1), 0.16)),
        "birth_corrected_total_q50": corrected_total,
        "birth_corrected_total_q84": float(np.nanquantile(np.sum(birth_weights, axis=1), 0.84)),
        "kde_bandwidth_scale": float(args.kde_bandwidth_scale),
        "split_contour_fraction": float(args.split_contour_fraction),
        "outputs": {
            "cluster_weights": str(cluster_path),
            "kde_grid": str(grid_path),
            "split_kde_grid": str(split_grid_path),
            "surface_samples": str(surface_meta_path),
            "figure_pdf": str(args.output_pdf),
            "figure_png": str(args.output_png),
            "split_figure_pdf": str(args.split_output_pdf),
            "split_figure_png": str(args.split_output_png),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    _plot_age_metallicity(
        table=working,
        observed_density=observed_density,
        corrected_density=corrected_density,
        age_grid=age_grid,
        feh_grid=feh_grid,
        corrected_total=corrected_total,
        output_pdf=args.output_pdf,
        output_png=args.output_png,
        bandwidth_scale=float(args.kde_bandwidth_scale),
        metallicity_column=str(args.metallicity_column),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
