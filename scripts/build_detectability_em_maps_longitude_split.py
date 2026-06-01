from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build longitude-split versions of the detectability completeness-map figure "
            "while holding the intrinsic single-component model fixed."
        )
    )
    parser.add_argument(
        "--threshold-deg",
        type=float,
        default=None,
        help="Split on |l| at this threshold in degrees. If omitted, use the requested quantile.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=None,
        help="Quantile of |l| used to define a balanced split, e.g. 0.5 for the median.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="detectability_em_maps_longitude_split",
        help="Stem for the output PDF/PNG and summary CSV files.",
    )
    parser.add_argument(
        "--smooth-survivability",
        action="store_true",
        help="Use the smooth eta_t-based survivability surface instead of the default hard-threshold grid.",
    )
    parser.add_argument(
        "--eta-t",
        type=float,
        default=1.0,
        help="Global lifetime renormalization used when --smooth-survivability is enabled.",
    )
    return parser.parse_args()


def histogram_observed_counts(
    catalog: pd.DataFrame,
    log_present_mass_edges: np.ndarray,
    distance_edges_kpc: np.ndarray,
    abs_latitude_edges_deg: np.ndarray,
) -> np.ndarray:
    return np.histogramdd(
        np.column_stack(
            [
                np.log10(catalog["present_mass_msun"].to_numpy(dtype=float)),
                catalog["r_sun_kpc"].to_numpy(dtype=float),
                np.abs(catalog["galactic_b_deg"].to_numpy(dtype=float)),
            ]
        ),
        bins=[log_present_mass_edges, distance_edges_kpc, abs_latitude_edges_deg],
    )[0].astype(float)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(project_root / ".cache"))
    (project_root / ".mplconfig").mkdir(parents=True, exist_ok=True)
    (project_root / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)

    from globular_clusters_imf.detectability_model import (
        build_completeness_grid_table,
        build_observable_histogram_table,
        evaluate_completeness_bin_grid,
        fit_detectability_corrected_single_component_models,
        fit_logistic_completeness_model,
    )
    from globular_clusters_imf.model import fit_catalog_models
    from globular_clusters_imf.paper_assets import plot_detectability_em_maps_by_longitude_split_for_paper
    from globular_clusters_imf.smooth_survivability import build_smooth_survivability_grid

    catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    prepared_catalog = fit_catalog_models(catalog, project_root)["catalog"]
    survival_grid_override = None
    if args.smooth_survivability:
        smooth_survival = build_smooth_survivability_grid(prepared_catalog, eta_t=float(args.eta_t))
        survival_grid_override = {
            "log_mass_grid": smooth_survival["log_mass_grid"],
            "log_a_grid": smooth_survival["log_a_grid"],
            "semi_major_axis_grid_kpc": smooth_survival["semi_major_axis_grid_kpc"],
            "survival_probability": smooth_survival["survival_probability"],
            "selection_offset_dex": 0.0,
            "bandwidth_log10_a_dex": smooth_survival["bandwidth_log10_a_dex"],
            "smooth_survivability_summary": smooth_survival["summary"],
        }

    full_comparison = fit_detectability_corrected_single_component_models(
        prepared_catalog,
        project_root=project_root,
        survival_grid_override=survival_grid_override,
    )
    full_best_result = full_comparison["best_result"]
    full_observable_context = full_best_result["observable_context"]
    full_predicted_complete_counts = full_best_result["final_predicted_complete_counts"]
    full_best_row = full_comparison["summary_table"].iloc[0]

    signed_longitude = ((prepared_catalog["galactic_l_deg"].to_numpy(dtype=float) + 180.0) % 360.0) - 180.0
    abs_longitude = np.abs(signed_longitude)
    if args.threshold_deg is not None:
        threshold_deg = float(args.threshold_deg)
        threshold_label = f"{threshold_deg:.1f}"
    elif args.quantile is not None:
        threshold_deg = float(np.quantile(abs_longitude, args.quantile))
        threshold_label = f"{threshold_deg:.1f}"
    else:
        threshold_deg = 90.0
        threshold_label = f"{threshold_deg:.1f}"

    inner_mask = abs_longitude < threshold_deg
    subset_definitions = [
        (rf"$|l| < {threshold_label}^\circ$", "lower_abs_longitude", inner_mask),
        (rf"$|l| \geq {threshold_label}^\circ$", "higher_abs_longitude", ~inner_mask),
    ]

    longitude_results: list[tuple[str, str, dict[str, object]]] = []
    summary_rows: list[dict[str, object]] = []
    output_tables_dir = project_root / "outputs" / "tables"
    output_figures_dir = project_root / "outputs" / "figures"
    output_tables_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)

    for display_label, subset_key, subset_mask in subset_definitions:
        subset_catalog = prepared_catalog.loc[subset_mask].copy().reset_index(drop=True)
        subset_observed_counts = histogram_observed_counts(
            subset_catalog,
            log_present_mass_edges=full_observable_context.log_present_mass_edges,
            distance_edges_kpc=full_observable_context.distance_edges_kpc,
            abs_latitude_edges_deg=full_observable_context.abs_latitude_edges_deg,
        )
        subset_observable_context = replace(
            full_observable_context,
            observed_counts=subset_observed_counts,
        )
        completeness_fit = fit_logistic_completeness_model(
            observable_context=subset_observable_context,
            predicted_complete_counts=full_predicted_complete_counts,
            start_params=full_best_result["final_completeness_raw_parameters"],
        )
        completeness_grid = evaluate_completeness_bin_grid(
            completeness_fit["raw_parameters"],
            subset_observable_context,
        )
        predicted_observed_counts = full_predicted_complete_counts * completeness_grid
        completeness_grid_table = build_completeness_grid_table(
            subset_observable_context,
            completeness_grid,
        )
        completeness_grid_table.to_csv(
            output_tables_dir / f"{args.output_stem}_{subset_key}_completeness_grid.csv",
            index=False,
        )
        observable_histogram_table = build_observable_histogram_table(
            observable_context=subset_observable_context,
            predicted_complete_counts=full_predicted_complete_counts,
            predicted_observed_counts=predicted_observed_counts,
            completeness_bin_grid=completeness_grid,
        )
        observable_histogram_table.to_csv(
            output_tables_dir / f"{args.output_stem}_{subset_key}_observable_histogram.csv",
            index=False,
        )

        longitude_results.append(
            (
                display_label,
                subset_key,
                {"completeness_grid_table": completeness_grid_table},
            )
        )
        summary_rows.append(
            {
                "longitude_subset": subset_key,
                "longitude_subset_label": display_label,
                "abs_longitude_threshold_deg": threshold_deg,
                "n_clusters": int(len(subset_catalog)),
                "shared_intrinsic_imf_family": str(full_best_row["imf_family"]),
                "shared_intrinsic_radial_model": str(full_best_row["radial_model"]),
                "shared_total_initial_count": float(full_best_row["total_initial_count"]),
                "negative_log_likelihood": float(completeness_fit["negative_log_likelihood"]),
                "intercept": float(completeness_fit["raw_parameters"][0]),
                "mass_slope": float(np.exp(completeness_fit["raw_parameters"][1])),
                "distance_slope": float(np.exp(completeness_fit["raw_parameters"][2])),
                "latitude_slope": float(np.exp(completeness_fit["raw_parameters"][3])),
                "mean_detectability_against_shared_intrinsic": float(
                    np.sum(full_predicted_complete_counts * completeness_grid)
                    / max(np.sum(full_predicted_complete_counts), 1.0e-12)
                ),
                "predicted_complete_count_shared_intrinsic": float(np.sum(full_predicted_complete_counts)),
                "predicted_observed_count_subset_fit": float(np.sum(predicted_observed_counts)),
                "observed_count_subset": float(np.sum(subset_observed_counts)),
                "smooth_survivability": bool(args.smooth_survivability),
                "eta_t": float(args.eta_t),
            }
        )

    figure_pdf_path = output_figures_dir / f"{args.output_stem}.pdf"
    figure_png_path = output_figures_dir / f"{args.output_stem}.png"
    plot_detectability_em_maps_by_longitude_split_for_paper(longitude_results, figure_pdf_path)
    plot_detectability_em_maps_by_longitude_split_for_paper(longitude_results, figure_png_path)

    summary_path = output_tables_dir / f"{args.output_stem}_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Wrote {figure_pdf_path}")
    print(f"Wrote {figure_png_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
