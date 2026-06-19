from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


DEFAULT_VARIANT = "profile_map_and_exact_mcmc_schechter_logpoly3_logistic_global_monotonic_q"
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog_with_origin_and_chemistry.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "tables"
DEFAULT_FIGURE_PDF = PROJECT_ROOT / "paper" / "figures" / "detection_corrected_metallicity_distribution.pdf"
DEFAULT_FIGURE_PNG = PROJECT_ROOT / "paper" / "figures" / "detection_corrected_metallicity_distribution.png"


def _weighted_histogram_density(
    values: np.ndarray,
    weights: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges, weights=weights)
    widths = np.diff(edges)
    return counts / widths


def _normalized_density(density: np.ndarray, edges: np.ndarray) -> np.ndarray:
    integral = float(np.sum(np.asarray(density, dtype=float) * np.diff(edges)))
    if not np.isfinite(integral) or integral <= 0.0:
        return np.full_like(density, np.nan, dtype=float)
    return density / integral


def _load_surface_archives(
    variant_root: Path,
    max_surface_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], pd.DataFrame]:
    worker_dir = variant_root / "outputs" / "parallel_exact_mcmc_workers"
    npz_paths = sorted(worker_dir.glob("chain_*_selection_surfaces.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No selection-surface archives found in {worker_dir}")

    surface_refs: list[tuple[Path, int]] = []
    metadata_frames = []
    log_mass_grid: np.ndarray | None = None
    log_a_grid: np.ndarray | None = None

    for npz_path in npz_paths:
        csv_path = npz_path.with_suffix(".csv")
        with np.load(npz_path) as archive:
            if log_mass_grid is None:
                log_mass_grid = np.asarray(archive["log_mass_grid"], dtype=float)
                log_a_grid = np.asarray(archive["log_a_grid"], dtype=float)
            n_surface = int(np.asarray(archive["survival_probability"]).shape[0])
        if csv_path.exists():
            metadata = pd.read_csv(csv_path)
            metadata = metadata.iloc[:n_surface].copy()
        else:
            metadata = pd.DataFrame(index=np.arange(n_surface))
        metadata["surface_npz_path"] = str(npz_path)
        metadata["surface_index_in_npz"] = np.arange(n_surface, dtype=int)
        metadata_frames.append(metadata)
        surface_refs.extend((npz_path, index) for index in range(n_surface))

    all_metadata = pd.concat(metadata_frames, ignore_index=True)
    if len(surface_refs) != len(all_metadata):
        raise RuntimeError("Selection-surface metadata and archive counts do not match.")

    if max_surface_samples is not None and int(max_surface_samples) > 0 and len(surface_refs) > int(max_surface_samples):
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(len(surface_refs), size=int(max_surface_samples), replace=False))
        surface_refs = [surface_refs[int(index)] for index in chosen]
        all_metadata = all_metadata.iloc[chosen].reset_index(drop=True)
    else:
        all_metadata = all_metadata.reset_index(drop=True)

    if log_mass_grid is None or log_a_grid is None:
        raise RuntimeError("No selection surfaces could be loaded.")
    return log_mass_grid, log_a_grid, surface_refs, all_metadata


def _interpolate_selection_for_catalog(
    *,
    log_mass_grid: np.ndarray,
    log_a_grid: np.ndarray,
    surface_refs: list[tuple[Path, int]],
    points: np.ndarray,
    floor: float,
) -> np.ndarray:
    selections = np.empty((len(surface_refs), len(points)), dtype=np.float64)
    archive_cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    for surface_number, (npz_path, surface_index) in enumerate(surface_refs):
        if npz_path not in archive_cache:
            archive = np.load(npz_path)
            archive_cache[npz_path] = (
                np.asarray(archive["survival_probability"], dtype=float),
                np.asarray(archive["effective_detectability"], dtype=float),
            )
        survival_stack, detectability_stack = archive_cache[npz_path]
        selection_grid = np.clip(
            survival_stack[int(surface_index)] * detectability_stack[int(surface_index)],
            floor,
            1.0,
        )
        interpolator = RegularGridInterpolator(
            (log_mass_grid, log_a_grid),
            selection_grid,
            bounds_error=False,
            fill_value=np.nan,
        )
        selection = np.asarray(interpolator(points), dtype=float)
        selections[surface_number] = np.clip(selection, floor, 1.0)
    return selections


def _cluster_weight_table(
    catalog: pd.DataFrame,
    selection_samples: np.ndarray,
    clipped_log_mass: np.ndarray,
    clipped_log_a: np.ndarray,
) -> pd.DataFrame:
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
        "local_feh",
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


def _build_mdf_tables(
    *,
    feh: np.ndarray,
    selection_samples: np.ndarray,
    edges: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float]]:
    observed_density = _weighted_histogram_density(feh, np.ones_like(feh), edges)
    observed_normalized = _normalized_density(observed_density, edges)

    birth_densities = []
    missing_densities = []
    for selection in selection_samples:
        birth_weight = 1.0 / selection
        missing_weight = birth_weight - 1.0
        birth_densities.append(_weighted_histogram_density(feh, birth_weight, edges))
        missing_densities.append(_weighted_histogram_density(feh, missing_weight, edges))

    birth_densities = np.asarray(birth_densities, dtype=float)
    missing_densities = np.asarray(missing_densities, dtype=float)
    birth_norm = np.asarray([_normalized_density(row, edges) for row in birth_densities])
    missing_norm = np.asarray([_normalized_density(row, edges) for row in missing_densities])

    rows = []
    centers = 0.5 * (edges[:-1] + edges[1:])
    for index, center in enumerate(centers):
        row = {
            "feh_bin_index": int(index),
            "feh_left": float(edges[index]),
            "feh_right": float(edges[index + 1]),
            "feh_center": float(center),
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
        "n_observed_with_valid_feh": float(len(feh)),
        "birth_corrected_total_q16": float(np.nanquantile(np.sum(1.0 / selection_samples, axis=1), 0.16)),
        "birth_corrected_total_q50": float(np.nanquantile(np.sum(1.0 / selection_samples, axis=1), 0.50)),
        "birth_corrected_total_q84": float(np.nanquantile(np.sum(1.0 / selection_samples, axis=1), 0.84)),
        "missing_total_q16": float(np.nanquantile(np.sum(1.0 / selection_samples - 1.0, axis=1), 0.16)),
        "missing_total_q50": float(np.nanquantile(np.sum(1.0 / selection_samples - 1.0, axis=1), 0.50)),
        "missing_total_q84": float(np.nanquantile(np.sum(1.0 / selection_samples - 1.0, axis=1), 0.84)),
    }
    return pd.DataFrame(rows), totals


def _plot_mdf(table: pd.DataFrame, totals: dict[str, float], output_pdf: Path, output_png: Path) -> None:
    x = table["feh_center"].to_numpy(dtype=float)
    edges = np.concatenate(
        [
            table["feh_left"].to_numpy(dtype=float),
            [float(table["feh_right"].iloc[-1])],
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), constrained_layout=True)
    observed_color = "#222222"
    birth_color = "#2f6fbb"
    missing_color = "#c44e52"

    axes[0].stairs(
        table["observed_density"].to_numpy(dtype=float),
        edges,
        color=observed_color,
        linewidth=1.8,
        label=f"Observed survivors ({int(totals['n_observed_with_valid_feh'])})",
    )
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
    axes[0].set_xlabel(r"$[{\rm Fe/H}]$")
    axes[0].set_ylabel(r"$dN/d[{\rm Fe/H}]$")
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[0].set_title("Absolute counts")

    axes[1].stairs(
        table["observed_normalized_density"].to_numpy(dtype=float),
        edges,
        color=observed_color,
        linewidth=1.8,
        label="Observed survivors",
    )
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
    axes[1].set_xlabel(r"$[{\rm Fe/H}]$")
    axes[1].set_ylabel(r"Probability density")
    axes[1].set_title("Normalized MDF shape")
    axes[1].legend(frameon=False, fontsize=8.5)

    for axis in axes:
        axis.set_xlim(edges[0], edges[-1])
        axis.grid(alpha=0.18, linewidth=0.7)

    subtitle = (
        r"$w_{\rm birth}=1/[S(M_{\rm ini},a)Q(M_{\rm ini},a)]$, "
        r"$w_{\rm dest}=w_{\rm birth}-1$"
    )
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
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_FIGURE_PDF)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_FIGURE_PNG)
    parser.add_argument("--feh-min", type=float, default=-2.6)
    parser.add_argument("--feh-max", type=float, default=0.0)
    parser.add_argument("--feh-bin-width", type=float, default=0.2)
    parser.add_argument("--max-surface-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--selection-floor", type=float, default=1.0e-4)
    args = parser.parse_args()

    catalog = pd.read_csv(args.catalog)
    required_columns = ["local_feh", "initial_mass_msun", "semi_major_axis_kpc"]
    missing = [column for column in required_columns if column not in catalog.columns]
    if missing:
        raise ValueError(f"{args.catalog} is missing required columns: {missing}")
    finite = (
        np.isfinite(catalog["local_feh"])
        & (catalog["local_feh"] > -10.0)
        & np.isfinite(catalog["initial_mass_msun"])
        & np.isfinite(catalog["semi_major_axis_kpc"])
        & (catalog["initial_mass_msun"] > 0.0)
        & (catalog["semi_major_axis_kpc"] > 0.0)
    )
    working = catalog.loc[finite].copy().reset_index(drop=True)
    if working.empty:
        raise ValueError("No finite metallicity entries are available.")

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

    edges = np.arange(float(args.feh_min), float(args.feh_max) + 0.5 * float(args.feh_bin_width), float(args.feh_bin_width))
    mdf_table, totals = _build_mdf_tables(
        feh=working["local_feh"].to_numpy(dtype=float),
        selection_samples=selection_samples,
        edges=edges,
    )
    cluster_weights = _cluster_weight_table(
        working,
        selection_samples,
        clipped_log_mass=clipped_points[:, 0],
        clipped_log_a=clipped_points[:, 1],
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    mdf_path = args.output_root / "detection_corrected_mdf.csv"
    weights_path = args.output_root / "detection_corrected_mdf_cluster_weights.csv"
    surface_meta_path = args.output_root / "detection_corrected_mdf_surface_samples.csv"
    summary_path = args.output_root / "detection_corrected_mdf_summary.json"
    mdf_table.to_csv(mdf_path, index=False)
    cluster_weights.to_csv(weights_path, index=False)
    surface_metadata.to_csv(surface_meta_path, index=False)
    summary = {
        "variant": str(args.variant),
        "catalog": str(args.catalog),
        "n_catalog_rows": int(len(catalog)),
        "n_with_valid_feh_and_coordinates": int(len(working)),
        "n_surface_samples": int(len(surface_refs)),
        "selection_floor": float(args.selection_floor),
        "n_selection_coordinates_clipped": int(np.any(~np.isclose(points, clipped_points), axis=1).sum()),
        "feh_edges": [float(value) for value in edges],
        "totals": totals,
        "outputs": {
            "mdf_table": str(mdf_path),
            "cluster_weights": str(weights_path),
            "surface_samples": str(surface_meta_path),
            "figure_pdf": str(args.output_pdf),
            "figure_png": str(args.output_png),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    _plot_mdf(mdf_table, totals, args.output_pdf, args.output_png)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
