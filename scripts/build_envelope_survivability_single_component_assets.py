from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


def histogram_observed_counts_with_abs_longitude(
    catalog: pd.DataFrame,
    *,
    log_present_mass_edges: np.ndarray,
    distance_edges_kpc: np.ndarray,
    abs_latitude_edges_deg: np.ndarray,
    abs_longitude_edges_deg: np.ndarray,
) -> np.ndarray:
    return np.histogramdd(
        np.column_stack(
            [
                np.log10(catalog["present_mass_msun"].to_numpy(dtype=float)),
                catalog["r_sun_kpc"].to_numpy(dtype=float),
                np.abs(catalog["galactic_b_deg"].to_numpy(dtype=float)),
                np.abs(((catalog["galactic_l_deg"].to_numpy(dtype=float) + 180.0) % 360.0) - 180.0),
            ]
        ),
        bins=[
            log_present_mass_edges,
            distance_edges_kpc,
            abs_latitude_edges_deg,
            abs_longitude_edges_deg,
        ],
    )[0].astype(float)


def load_family_scan_results(variant_root: Path) -> dict[str, object]:
    outputs_root = variant_root / "outputs"
    tables_root = outputs_root / "tables"
    summary_path = tables_root / "single_component_family_profile_scan_summary.json"
    powerlaw_path = tables_root / "powerlaw_profile_scan.csv"
    lognormal_path = tables_root / "lognormal_profile_scan.csv"
    schechter_path = tables_root / "schechter_profile_scan.csv"
    best_result_path = tables_root / "schechter_best_result.pkl"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing family scan summary: {summary_path}")
    if not best_result_path.exists():
        raise FileNotFoundError(f"Missing saved best Schechter result: {best_result_path}")
    return {
        "variant_output_root": outputs_root,
        "summary_payload": json.loads(summary_path.read_text()),
        "powerlaw_table": pd.read_csv(powerlaw_path),
        "lognormal_table": pd.read_csv(lognormal_path),
        "schechter_table": pd.read_csv(schechter_path),
        "best_result_path": best_result_path,
    }


def load_flexible_overlay_from_variant_root(
    variant_root: Path,
    log_mass_grid: np.ndarray,
) -> dict[str, np.ndarray] | None:
    from scipy import interpolate

    tables_root = variant_root / "outputs" / "tables"
    summary_path = tables_root / "imf_profile_vs_logspline_summary.json"
    band_path = tables_root / "logspline6_bootstrap_imf_band.csv"
    if not summary_path.exists() or not band_path.exists():
        return None
    summary_payload = json.loads(summary_path.read_text())
    model_payload = summary_payload.get("logspline6_bootstrap_model", {})
    imf_parameters_json = model_payload.get("imf_parameters_json")
    if not imf_parameters_json:
        return None
    imf_parameters = json.loads(imf_parameters_json)
    knot_positions = np.asarray(imf_parameters["knot_log10_msun"], dtype=float)
    node_log_amplitudes = np.asarray(imf_parameters["node_log_amplitudes_relative"], dtype=float)
    spline = interpolate.PchipInterpolator(knot_positions, node_log_amplitudes, extrapolate=True)
    density_grid = np.exp(np.clip(spline(log_mass_grid), -700.0, 700.0))
    density_grid /= max(float(np.trapezoid(density_grid, log_mass_grid)), 1.0e-12)
    band_table = pd.read_csv(band_path)
    return {
        "log_mass_grid": np.asarray(log_mass_grid, dtype=float),
        "imf_density_grid": np.asarray(density_grid, dtype=float),
        "imf_band_low": np.interp(
            log_mass_grid,
            np.asarray(band_table["log_initial_mass_msun"], dtype=float),
            np.asarray(band_table["imf_band_low"], dtype=float),
        ),
        "imf_band_high": np.interp(
            log_mass_grid,
            np.asarray(band_table["log_initial_mass_msun"], dtype=float),
            np.asarray(band_table["imf_band_high"], dtype=float),
        ),
    }


def build_survival_grid_override(survivability_map: dict[str, object]) -> dict[str, object]:
    summary = survivability_map.get("summary")
    payload = {
        "log_mass_grid": np.asarray(survivability_map["log_mass_grid"], dtype=float),
        "log_a_grid": np.asarray(survivability_map["log_a_grid"], dtype=float),
        "semi_major_axis_grid_kpc": np.asarray(survivability_map["semi_major_axis_grid_kpc"], dtype=float),
        "survival_probability": np.asarray(survivability_map["survival_probability"], dtype=float),
        "selection_offset_dex": 0.0,
        "bandwidth_log10_a_dex": float(survivability_map.get("bandwidth_log10_a_dex", np.nan)),
    }
    if summary is not None:
        payload["envelope_survivability_summary"] = summary
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root-name",
        type=str,
        default="single_component_envelope_survivability_sanity",
    )
    parser.add_argument(
        "--family-scan-root-name",
        type=str,
        default="single_component_family_profile_scan_envelope_survival",
    )
    parser.add_argument(
        "--flexible-root-name",
        type=str,
        default="flexible_imf_bootstrap_comparison_abs_longitude_envelope_survival",
    )
    parser.add_argument(
        "--skip-flexible-imf",
        action="store_true",
        help="Skip the auxiliary flexible-IMF bootstrap overlay and build only the core single-component figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(project_root / ".cache"))
    (project_root / ".mplconfig").mkdir(parents=True, exist_ok=True)
    (project_root / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)

    from globular_clusters_imf.detectability_longitude_model import (
        build_completeness_grid_table_with_abs_longitude,
        build_observable_histogram_table_with_abs_longitude,
        evaluate_completeness_bin_grid_with_abs_longitude,
        fit_detectability_corrected_single_component_models_with_abs_longitude,
        fit_logistic_completeness_model_with_abs_longitude,
    )
    from globular_clusters_imf.envelope_survivability import build_envelope_survivability_grid
    from globular_clusters_imf.flexible_imf import build_profile_vs_flexible_imf_comparison
    from globular_clusters_imf.joint_model import (
        estimate_best_model_uncertainty,
        fit_fixed_survival_joint_models,
    )
    from globular_clusters_imf.model import fit_catalog_models
    from globular_clusters_imf.paper_assets import (
        PAPER_LOG_MASS_MIN,
        best_single_model_total_initial_count,
        best_single_model_total_initial_stellar_mass,
        detectability_corrected_single_total_initial_count,
        detectability_corrected_single_total_initial_stellar_mass,
        plot_best_single_component_summary_for_paper,
        plot_catalog_mass_semimajor_axis_overview_for_paper,
        plot_detectability_em_convergence_for_paper,
        plot_detectability_em_maps_by_longitude_split_for_paper,
        plot_single_component_family_profile_scan_for_paper,
        plot_single_component_profiles_for_paper,
        plot_single_component_radial_profile_for_paper,
    )

    family_scan_root = project_root / "variants" / args.family_scan_root_name
    output_root = project_root / "variants" / args.output_root_name
    flexible_root = project_root / "variants" / args.flexible_root_name
    outputs_dir = output_root / "outputs"
    figures_dir = outputs_dir / "figures"
    tables_dir = outputs_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    fit_catalog = fit_catalog_models(catalog, output_root)["catalog"]

    survivability_map = build_envelope_survivability_grid(fit_catalog)
    survival_grid_override = build_survival_grid_override(survivability_map)

    plot_catalog_mass_semimajor_axis_overview_for_paper(
        fit_catalog,
        survivability_map,
        figures_dir / "catalog_mass_semimajor_axis_overview.png",
    )
    survivability_map["hull_table"].to_csv(tables_dir / "envelope_survivability_hull_vertices.csv", index=False)

    baseline_joint_results = fit_fixed_survival_joint_models(
        fit_catalog,
        output_root,
        survival_grid_override=survival_grid_override,
    )
    unconstrained_comparison = fit_detectability_corrected_single_component_models_with_abs_longitude(
        fit_catalog,
        output_root,
        survival_grid_override=survival_grid_override,
    )
    figure4_convergence_result = next(
        result
        for result in unconstrained_comparison["all_results"]
        if result["spec"].imf_family == "schechter" and result["spec"].radial_model == "logpoly3"
    )

    family_scan_results = load_family_scan_results(family_scan_root)
    with family_scan_results["best_result_path"].open("rb") as handle:
        detectability_result = pickle.load(handle)
    adopted_solution_source = "family_scan_best_result"
    unconstrained_best_result = unconstrained_comparison["best_result"]
    unconstrained_best_logl = float(unconstrained_best_result["final_payload"]["summary"].log_likelihood)
    scanned_best_logl = float(detectability_result["final_payload"]["summary"].log_likelihood)
    if unconstrained_best_logl > scanned_best_logl:
        detectability_result = unconstrained_best_result
        adopted_solution_source = "unconstrained_best_result"
    detectability_uncertainty = estimate_best_model_uncertainty(
        best_payload=detectability_result["final_payload"],
        context=detectability_result["final_context"],
    )

    flexible_comparison = None
    flexible_imf_overlay = None
    if not args.skip_flexible_imf:
        flexible_comparison = build_profile_vs_flexible_imf_comparison(
            catalog=catalog,
            output_root=flexible_root,
            detectability_variant="abs_longitude",
            survival_grid_override=survival_grid_override,
            n_bootstrap=24,
            random_seed=12345,
        )
        flexible_imf_overlay = load_flexible_overlay_from_variant_root(
            flexible_root,
            np.asarray(detectability_result["final_context"].log_mass_grid, dtype=float),
        )

    full_observable_context = detectability_result["observable_context"]
    full_predicted_complete_counts = detectability_result["final_predicted_complete_counts"]
    abs_longitude = np.abs(((fit_catalog["galactic_l_deg"].to_numpy(dtype=float) + 180.0) % 360.0) - 180.0)
    threshold_deg = 30.0
    subset_definitions = [
        (r"$|l| < 30^\circ$", "lower_abs_longitude", abs_longitude < threshold_deg),
        (r"$|l| \geq 30^\circ$", "higher_abs_longitude", abs_longitude >= threshold_deg),
    ]
    longitude_split_results: list[tuple[str, str, dict[str, object]]] = []
    longitude_split_summary_rows: list[dict[str, object]] = []
    for display_label, subset_key, subset_mask in subset_definitions:
        subset_catalog = fit_catalog.loc[subset_mask].copy().reset_index(drop=True)
        subset_observed_counts = histogram_observed_counts_with_abs_longitude(
            subset_catalog,
            log_present_mass_edges=full_observable_context.log_present_mass_edges,
            distance_edges_kpc=full_observable_context.distance_edges_kpc,
            abs_latitude_edges_deg=full_observable_context.abs_latitude_edges_deg,
            abs_longitude_edges_deg=full_observable_context.abs_longitude_edges_deg,
        )
        subset_observable_context = replace(full_observable_context, observed_counts=subset_observed_counts)
        completeness_fit = fit_logistic_completeness_model_with_abs_longitude(
            observable_context=subset_observable_context,
            predicted_complete_counts=full_predicted_complete_counts,
            start_params=detectability_result["final_completeness_raw_parameters"],
        )
        completeness_grid = evaluate_completeness_bin_grid_with_abs_longitude(
            completeness_fit["raw_parameters"],
            subset_observable_context,
        )
        predicted_observed_counts = full_predicted_complete_counts * completeness_grid
        observable_histogram_table = build_observable_histogram_table_with_abs_longitude(
            observable_context=subset_observable_context,
            predicted_complete_counts=full_predicted_complete_counts,
            predicted_observed_counts=predicted_observed_counts,
            completeness_bin_grid=completeness_grid,
        )
        plot_completeness_grid_table = (
            observable_histogram_table.groupby(
                [
                    "present_mass_bin_index",
                    "distance_bin_index",
                    "latitude_bin_index",
                    "log10_present_mass_center_msun",
                    "distance_center_kpc",
                    "abs_latitude_center_deg",
                ],
                as_index=False,
            )[["predicted_complete_count", "predicted_observed_count"]]
            .sum()
        )
        plot_completeness_grid_table["completeness"] = (
            plot_completeness_grid_table["predicted_observed_count"]
            / np.clip(plot_completeness_grid_table["predicted_complete_count"], 1.0e-12, None)
        )
        plot_completeness_grid_table.to_csv(
            tables_dir / f"detectability_em_maps_abs_longitude_30deg_split_envelope_{subset_key}_completeness_grid.csv",
            index=False,
        )
        longitude_split_results.append(
            (display_label, subset_key, {"completeness_grid_table": plot_completeness_grid_table})
        )
        longitude_split_summary_rows.append(
            {
                "longitude_subset": subset_key,
                "longitude_subset_label": display_label,
                "abs_longitude_threshold_deg": threshold_deg,
                "n_clusters": int(len(subset_catalog)),
                "shared_intrinsic_imf_family": str(detectability_result["final_payload"]["summary"].imf_family),
                "shared_intrinsic_radial_model": str(detectability_result["final_payload"]["summary"].radial_model),
                "shared_total_initial_count": float(detectability_result["final_payload"]["model"]["total_initial_count"]),
                "negative_log_likelihood": float(completeness_fit["negative_log_likelihood"]),
                "intercept": float(completeness_fit["raw_parameters"][0]),
                "mass_slope": float(np.exp(completeness_fit["raw_parameters"][1])),
                "distance_slope": float(np.exp(completeness_fit["raw_parameters"][2])),
                "latitude_slope": float(np.exp(completeness_fit["raw_parameters"][3])),
                "longitude_slope": float(np.exp(completeness_fit["raw_parameters"][4])),
                "mean_detectability_against_shared_intrinsic": float(
                    np.sum(full_predicted_complete_counts * completeness_grid)
                    / max(np.sum(full_predicted_complete_counts), 1.0e-12)
                ),
                "predicted_complete_count_shared_intrinsic": float(np.sum(full_predicted_complete_counts)),
                "predicted_observed_count_subset_fit": float(np.sum(predicted_observed_counts)),
                "observed_count_subset": float(np.sum(subset_observed_counts)),
            }
        )
    pd.DataFrame(longitude_split_summary_rows).to_csv(
        tables_dir / "detectability_em_maps_abs_longitude_30deg_split_envelope_summary.csv",
        index=False,
    )

    plot_detectability_em_maps_by_longitude_split_for_paper(
        longitude_split_results,
        figures_dir / "detectability_em_maps.png",
    )
    plot_detectability_em_convergence_for_paper(
        figure4_convergence_result,
        figures_dir / "detectability_em_convergence.png",
    )
    plot_single_component_family_profile_scan_for_paper(
        family_scan_results,
        figures_dir / "single_component_model_performance.png",
    )
    plot_best_single_component_summary_for_paper(
        fit_catalog,
        context=detectability_result["final_context"],
        best_payload=detectability_result["final_payload"],
        uncertainty_payload=detectability_uncertainty,
        output_path=figures_dir / "best_single_component_summary.png",
    )
    plot_single_component_profiles_for_paper(
        baseline_joint_results=baseline_joint_results,
        detectability_result=detectability_result,
        uncertainty_payload=detectability_uncertainty,
        flexible_imf_overlay=flexible_imf_overlay,
        output_path=figures_dir / "single_component_profiles.png",
    )
    plot_single_component_radial_profile_for_paper(
        baseline_joint_results=baseline_joint_results,
        detectability_result=detectability_result,
        uncertainty_payload=detectability_uncertainty,
        flexible_imf_overlay=flexible_imf_overlay,
        output_path=figures_dir / "single_component_radial_profile.png",
    )

    baseline_n0 = float(best_single_model_total_initial_count(baseline_joint_results))
    baseline_mass0 = float(best_single_model_total_initial_stellar_mass(baseline_joint_results))
    corrected_n0 = float(detectability_corrected_single_total_initial_count(detectability_result))
    corrected_mass0 = float(detectability_corrected_single_total_initial_stellar_mass(detectability_result))
    family_summary = family_scan_results["summary_payload"]
    powerlaw_best = family_summary["powerlaw_scan"]["best_scan_point"]
    lognormal_best = family_summary["lognormal_scan"]["best_scan_point"]
    schechter_best = family_summary["schechter_scan"]["best_scan_point"]
    summary_payload = {
        "reported_log10_initial_mass_min": float(PAPER_LOG_MASS_MIN),
        "survivability_model": "hard_envelope_below_all_gcs",
        "survivability_summary": survivability_map["summary"].__dict__,
        "baseline_single_component": {
            "total_initial_count_above_1e4_msun": baseline_n0,
            "total_initial_stellar_mass_above_1e4_msun": baseline_mass0,
        },
        "detectability_corrected_single_component": {
            "adopted_solution_source": adopted_solution_source,
            "imf_family": str(detectability_result["final_payload"]["summary"].imf_family),
            "radial_model": str(detectability_result["final_payload"]["summary"].radial_model),
            "log_likelihood": float(detectability_result["final_payload"]["summary"].log_likelihood),
            "total_initial_count_above_1e4_msun": corrected_n0,
            "total_initial_stellar_mass_above_1e4_msun": corrected_mass0,
            "selection_fraction": float(detectability_result["final_payload"]["model"]["selection_fraction"]),
            "raw_survival_fraction": float(detectability_result["final_payload"]["model"]["raw_survival_fraction"]),
            "mean_detectability": float(
                detectability_result["final_payload"]["model"]["selection_fraction"]
                / max(detectability_result["final_payload"]["model"]["raw_survival_fraction"], 1.0e-12)
            ),
            "count_ratio_vs_baseline": corrected_n0 / max(baseline_n0, 1.0e-12),
            "imf_parameters": json.loads(detectability_result["final_payload"]["summary"].imf_parameters_json),
        },
        "family_scan": {
            "schechter_best": schechter_best,
            "lognormal_best": lognormal_best,
            "powerlaw_best": powerlaw_best,
            "powerlaw_delta_log_likelihood_vs_schechter": float(
                schechter_best["log_likelihood"] - powerlaw_best["log_likelihood"]
            ),
            "powerlaw_delta_bic_vs_schechter": float(
                powerlaw_best["bic"] - schechter_best["bic"]
            ),
            "lognormal_delta_log_likelihood_vs_schechter": float(
                schechter_best["log_likelihood"] - lognormal_best["log_likelihood"]
            ),
            "lognormal_delta_bic_vs_schechter": float(
                lognormal_best["bic"] - schechter_best["bic"]
            ),
        },
        "flexible_imf_comparison": None if flexible_comparison is None else flexible_comparison["summary_payload"],
        "figure_paths": {
            "figure1_overview": str(figures_dir / "catalog_mass_semimajor_axis_overview.png"),
            "figure3_detectability_maps": str(figures_dir / "detectability_em_maps.png"),
            "figure4_convergence": str(figures_dir / "detectability_em_convergence.png"),
            "figure5_family_scan": str(figures_dir / "single_component_model_performance.png"),
            "figure6_best_summary": str(figures_dir / "best_single_component_summary.png"),
            "figure7_profiles": str(figures_dir / "single_component_profiles.png"),
            "figure8_radial_profile": str(figures_dir / "single_component_radial_profile.png"),
        },
        "supporting_variant_roots": {
            "family_scan_root": str(family_scan_root),
            "flexible_root": None if args.skip_flexible_imf else str(flexible_root),
        },
    }
    (tables_dir / "single_component_envelope_survivability_summary.json").write_text(
        json.dumps(summary_payload, indent=2, default=float)
    )
    print(f"Wrote {figures_dir / 'catalog_mass_semimajor_axis_overview.png'}")
    print(f"Wrote {figures_dir / 'detectability_em_maps.png'}")
    print(f"Wrote {figures_dir / 'detectability_em_convergence.png'}")
    print(f"Wrote {figures_dir / 'single_component_model_performance.png'}")
    print(f"Wrote {figures_dir / 'best_single_component_summary.png'}")
    print(f"Wrote {figures_dir / 'single_component_profiles.png'}")
    print(f"Wrote {figures_dir / 'single_component_radial_profile.png'}")
    print(f"Wrote {tables_dir / 'single_component_envelope_survivability_summary.json'}")


if __name__ == "__main__":
    main()
