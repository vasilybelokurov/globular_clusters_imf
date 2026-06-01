from __future__ import annotations

import json
import os
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


def fit_adopted_profile_schechter_result(
    fit_catalog: pd.DataFrame,
    project_root: Path,
    *,
    survival_grid_override: dict[str, object],
    family_profile_scan_results: dict[str, object],
    reference_result: dict[str, object],
) -> dict[str, object]:
    from globular_clusters_imf.detectability_longitude_model import (
        fit_single_component_detectability_em_with_abs_longitude,
    )
    from globular_clusters_imf.joint_model import JointModelSpec

    variant_output_root = Path(family_profile_scan_results["variant_output_root"])
    saved_result_path = variant_output_root / "tables" / "schechter_best_result.pkl"
    if saved_result_path.exists():
        with saved_result_path.open("rb") as handle:
            return pickle.load(handle)

    scan_configuration = family_profile_scan_results["summary_payload"]["scan_configuration"]
    schechter_best_point = family_profile_scan_results["summary_payload"]["schechter_scan"]["best_scan_point"]
    n_iterations = int(scan_configuration["n_iterations_per_grid_point"])
    schechter_alpha_grid = np.asarray(scan_configuration["schechter_alpha_grid"], dtype=float)
    schechter_logmc_grid = np.asarray(scan_configuration["schechter_log10_mc_grid"], dtype=float)
    target_alpha = float(schechter_best_point["alpha_dndm"])
    target_logmc = float(schechter_best_point["log10_m_c_msun"])
    target_alpha_index = int(np.argmin(np.abs(schechter_alpha_grid - target_alpha)))
    target_logmc_index = int(np.argmin(np.abs(schechter_logmc_grid - target_logmc)))
    spec = JointModelSpec(imf_family="schechter", radial_model="logpoly3")
    reference_start_completeness = np.asarray(reference_result["final_completeness_raw_parameters"], dtype=float)
    reference_start_radial = np.asarray(reference_result["final_payload"]["raw_parameters"], dtype=float)[2:]

    previous_row_results: list[dict[str, object]] | None = None
    best_result = None
    for logmc_index, log_mc in enumerate(schechter_logmc_grid):
        current_row_results: list[dict[str, object]] = []
        max_alpha_index = len(schechter_alpha_grid) - 1
        if logmc_index == target_logmc_index:
            max_alpha_index = target_alpha_index
        for alpha_index, alpha in enumerate(schechter_alpha_grid[: max_alpha_index + 1]):
            left_neighbor = current_row_results[-1] if current_row_results else None
            upper_neighbor = None if previous_row_results is None else previous_row_results[alpha_index]
            start_source = left_neighbor if left_neighbor is not None else upper_neighbor
            start_completeness = (
                None
                if start_source is None
                else np.asarray(start_source["final_completeness_raw_parameters"], dtype=float)
            )
            start_radial = (
                None
                if start_source is None
                else np.asarray(start_source["final_payload"]["radial_parameters_raw"], dtype=float)
            )
            result = fit_single_component_detectability_em_with_abs_longitude(
                fit_catalog,
                project_root=project_root,
                spec=spec,
                n_iterations=n_iterations,
                fixed_imf_params=np.array([float(alpha), float(log_mc)], dtype=float),
                start_completeness_raw_parameters=start_completeness,
                start_radial_params=start_radial,
                survival_grid_override=survival_grid_override,
            )
            node_best_result = result
            node_best_log_likelihood = float(result["final_payload"]["summary"].log_likelihood)

            # Also probe the unconstrained-reference start at each node, matching the scan logic.
            reference_started_result = fit_single_component_detectability_em_with_abs_longitude(
                fit_catalog,
                project_root=project_root,
                spec=spec,
                n_iterations=n_iterations,
                fixed_imf_params=np.array([float(alpha), float(log_mc)], dtype=float),
                start_completeness_raw_parameters=reference_start_completeness,
                start_radial_params=reference_start_radial,
                survival_grid_override=survival_grid_override,
            )
            reference_log_likelihood = float(reference_started_result["final_payload"]["summary"].log_likelihood)
            if reference_log_likelihood > node_best_log_likelihood:
                node_best_result = reference_started_result
                node_best_log_likelihood = reference_log_likelihood
            current_row_results.append(node_best_result)
            if logmc_index == target_logmc_index and alpha_index == target_alpha_index:
                best_result = node_best_result
        previous_row_results = current_row_results
        if logmc_index == target_logmc_index:
            break

    if best_result is None:
        raise RuntimeError("Failed to fit the adopted profiled Schechter solution.")

    saved_result_path.parent.mkdir(parents=True, exist_ok=True)
    with saved_result_path.open("wb") as handle:
        pickle.dump(best_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return best_result


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


def main() -> None:
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
    from globular_clusters_imf.joint_model import (
        estimate_best_model_uncertainty,
        fit_fixed_survival_joint_models,
    )
    from globular_clusters_imf.model import fit_catalog_models
    from globular_clusters_imf.paper_assets import (
        PAPER_LOG_MASS_MIN,
        best_single_model_total_initial_count,
        best_single_model_total_initial_stellar_mass,
        build_single_component_family_profile_scan_table,
        detectability_corrected_single_total_initial_count,
        detectability_corrected_single_total_initial_stellar_mass,
        load_precomputed_flexible_imf_overlay,
        load_precomputed_single_component_family_profile_scan_results,
        plot_best_single_component_summary_for_paper,
        plot_detectability_counts_for_paper,
        plot_detectability_em_convergence_for_paper,
        plot_detectability_em_maps_by_longitude_split_for_paper,
        plot_single_component_family_profile_scan_for_paper,
        plot_single_component_profiles_for_paper,
        plot_single_component_radial_profile_for_paper,
        write_key_results_table_tex,
        write_single_component_table_tex,
        write_summary_macros_tex,
    )
    from globular_clusters_imf.smooth_survivability import build_smooth_survivability_grid

    catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog.csv"

    catalog = pd.read_csv(catalog_path)
    catalog_results = fit_catalog_models(catalog, project_root)
    fit_catalog = catalog_results["catalog"]
    smooth_survivability = build_smooth_survivability_grid(fit_catalog, eta_t=1.0)
    survival_grid_override = {
        "log_mass_grid": np.asarray(smooth_survivability["log_mass_grid"], dtype=float),
        "log_a_grid": np.asarray(smooth_survivability["log_a_grid"], dtype=float),
        "semi_major_axis_grid_kpc": np.asarray(smooth_survivability["semi_major_axis_grid_kpc"], dtype=float),
        "survival_probability": np.asarray(smooth_survivability["survival_probability"], dtype=float),
        "selection_offset_dex": 0.0,
        "bandwidth_log10_a_dex": float(smooth_survivability["bandwidth_log10_a_dex"]),
        "smooth_survivability_summary": smooth_survivability["summary"],
    }
    joint_results = fit_fixed_survival_joint_models(
        fit_catalog,
        project_root,
        survival_grid_override=survival_grid_override,
    )
    detectability_comparison = fit_detectability_corrected_single_component_models_with_abs_longitude(
        fit_catalog,
        project_root,
        survival_grid_override=survival_grid_override,
    )
    figure4_convergence_result = next(
        result
        for result in detectability_comparison["all_results"]
        if result["spec"].imf_family == "schechter" and result["spec"].radial_model == "logpoly3"
    )
    family_profile_scan_results = load_precomputed_single_component_family_profile_scan_results(
        project_root=project_root,
    )
    detectability_result = fit_adopted_profile_schechter_result(
        fit_catalog,
        project_root,
        survival_grid_override=survival_grid_override,
        family_profile_scan_results=family_profile_scan_results,
        reference_result=detectability_comparison["best_result"],
    )
    detectability_uncertainty = estimate_best_model_uncertainty(
        best_payload=detectability_result["final_payload"],
        context=detectability_result["final_context"],
    )
    flexible_imf_overlay = load_precomputed_flexible_imf_overlay(
        project_root=project_root,
        log_mass_grid=detectability_result["final_context"].log_mass_grid,
    )
    output_stem = "detectability_em_maps_abs_longitude_30deg_split_smooth_survival_eta1"
    output_tables_dir = project_root / "outputs" / "tables"
    output_tables_dir.mkdir(parents=True, exist_ok=True)
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
        subset_observable_context = replace(
            full_observable_context,
            observed_counts=subset_observed_counts,
        )
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
        completeness_grid_table = build_completeness_grid_table_with_abs_longitude(
            subset_observable_context,
            completeness_grid,
        )
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
            output_tables_dir / f"{output_stem}_{subset_key}_completeness_grid.csv",
            index=False,
        )
        observable_histogram_table.to_csv(
            output_tables_dir / f"{output_stem}_{subset_key}_observable_histogram.csv",
            index=False,
        )
        longitude_split_results.append(
            (
                display_label,
                subset_key,
                {"completeness_grid_table": plot_completeness_grid_table},
            )
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
                "smooth_survivability": True,
                "eta_t": 1.0,
            }
        )
    pd.DataFrame(longitude_split_summary_rows).to_csv(
        output_tables_dir / f"{output_stem}_summary.csv",
        index=False,
    )

    paper_dir = project_root / "paper"
    figures_dir = paper_dir / "figures"
    tables_dir = paper_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    plot_detectability_counts_for_paper(
        fit_catalog,
        figures_dir / "detectability_counts.pdf",
        longitude_limit_deg=30.0,
    )
    plot_detectability_em_maps_by_longitude_split_for_paper(
        longitude_split_results,
        figures_dir / "detectability_em_maps.pdf",
    )
    plot_detectability_em_convergence_for_paper(
        figure4_convergence_result,
        figures_dir / "detectability_em_convergence.pdf",
    )
    plot_single_component_family_profile_scan_for_paper(
        family_profile_scan_results,
        figures_dir / "single_component_model_performance.pdf",
    )
    plot_best_single_component_summary_for_paper(
        fit_catalog,
        context=detectability_result["final_context"],
        best_payload=detectability_result["final_payload"],
        uncertainty_payload=detectability_uncertainty,
        output_path=figures_dir / "best_single_component_summary.pdf",
    )
    plot_single_component_profiles_for_paper(
        baseline_joint_results=joint_results,
        detectability_result=detectability_result,
        uncertainty_payload=detectability_uncertainty,
        flexible_imf_overlay=flexible_imf_overlay,
        output_path=figures_dir / "single_component_profiles.pdf",
    )
    plot_single_component_radial_profile_for_paper(
        baseline_joint_results=joint_results,
        detectability_result=detectability_result,
        uncertainty_payload=detectability_uncertainty,
        flexible_imf_overlay=flexible_imf_overlay,
        output_path=figures_dir / "single_component_radial_profile.pdf",
    )

    single_component_table = build_single_component_family_profile_scan_table(
        family_profile_scan_results,
    )
    single_component_table.to_csv(tables_dir / "single_component_model_comparison.csv", index=False)
    write_single_component_table_tex(single_component_table, tables_dir / "single_component_model_comparison.tex")

    key_results_path = tables_dir / "key_results_summary.csv"
    key_results_table = pd.read_csv(key_results_path)
    single_row_mask = (
        key_results_table["model"] == "Detectability-corrected single component"
    ) & (key_results_table["component"] == "all")
    baseline_row_mask = (
        key_results_table["model"] == "Single component"
    ) & (key_results_table["component"] == "all")
    detectability_summary = detectability_result["final_payload"]["summary"]
    detectability_imf_parameters = json.loads(detectability_summary.imf_parameters_json)
    key_results_table.loc[baseline_row_mask, "total_initial_count"] = float(
        best_single_model_total_initial_count(joint_results)
    )
    key_results_table.loc[baseline_row_mask, "total_initial_stellar_mass_msun"] = float(
        best_single_model_total_initial_stellar_mass(joint_results)
    )
    key_results_table.loc[single_row_mask, "imf_family"] = detectability_summary.imf_family
    key_results_table.loc[single_row_mask, "radial_model"] = detectability_summary.radial_model
    key_results_table.loc[single_row_mask, "alpha_dndm"] = detectability_imf_parameters.get("alpha_dndm")
    key_results_table.loc[single_row_mask, "log10_m_c_msun"] = detectability_imf_parameters.get("log10_m_c_msun")
    key_results_table.loc[single_row_mask, "total_initial_count"] = float(
        detectability_corrected_single_total_initial_count(detectability_result)
    )
    key_results_table.loc[single_row_mask, "survival_fraction"] = float(
        detectability_result["final_payload"]["model"]["selection_fraction"]
    )
    key_results_table.loc[single_row_mask, "total_initial_stellar_mass_msun"] = float(
        detectability_corrected_single_total_initial_stellar_mass(detectability_result)
    )
    key_results_table.to_csv(key_results_path, index=False)
    write_key_results_table_tex(key_results_table, tables_dir / "key_results_summary.tex")

    paper_summary_path = tables_dir / "paper_results_summary.json"
    paper_summary = json.loads(paper_summary_path.read_text())
    paper_summary["single_component_best_model"]["total_initial_count"] = float(
        best_single_model_total_initial_count(joint_results)
    )
    paper_summary["single_component_best_model"]["total_initial_stellar_mass_msun"] = float(
        best_single_model_total_initial_stellar_mass(joint_results)
    )
    paper_summary["single_component_best_model"]["reported_log10_initial_mass_min"] = float(PAPER_LOG_MASS_MIN)
    paper_summary["detectability_corrected_single_component_model"] = {
        "imf_family": str(detectability_summary.imf_family),
        "radial_model": str(detectability_summary.radial_model),
        "log_likelihood": float(detectability_summary.log_likelihood),
        "total_initial_count": float(detectability_corrected_single_total_initial_count(detectability_result)),
        "total_initial_stellar_mass_msun": float(detectability_corrected_single_total_initial_stellar_mass(detectability_result)),
        "selection_fraction": float(detectability_result["final_payload"]["model"]["selection_fraction"]),
        "raw_survival_fraction": float(detectability_result["final_payload"]["model"]["raw_survival_fraction"]),
        "mean_detectability": float(
            detectability_result["final_payload"]["model"]["selection_fraction"]
            / max(detectability_result["final_payload"]["model"]["raw_survival_fraction"], 1.0e-12)
        ),
        "reported_log10_initial_mass_min": float(PAPER_LOG_MASS_MIN),
        "imf_parameters": detectability_imf_parameters,
        "count_ratio_vs_baseline": float(
            detectability_corrected_single_total_initial_count(detectability_result)
            / max(float(paper_summary["single_component_best_model"]["total_initial_count"]), 1.0e-12)
        ),
    }
    paper_summary_path.write_text(json.dumps(paper_summary, indent=2))
    write_summary_macros_tex(paper_summary, tables_dir / "paper_numbers.tex")

    print("Updated paper single-component assets from the longitude-aware detectability inference.")
    print(f"Wrote {figures_dir / 'detectability_counts.pdf'}")
    print(f"Wrote {figures_dir / 'detectability_em_maps.pdf'}")
    print(f"Wrote {figures_dir / 'detectability_em_convergence.pdf'}")
    print(f"Wrote {figures_dir / 'single_component_model_performance.pdf'}")
    print(f"Wrote {figures_dir / 'best_single_component_summary.pdf'}")
    print(f"Wrote {figures_dir / 'single_component_profiles.pdf'}")
    print(f"Wrote {figures_dir / 'single_component_radial_profile.pdf'}")
    print(f"Wrote {tables_dir / 'single_component_model_comparison.tex'}")
    print(f"Wrote {tables_dir / 'key_results_summary.tex'}")
    print(f"Wrote {tables_dir / 'paper_numbers.tex'}")


if __name__ == "__main__":
    main()
