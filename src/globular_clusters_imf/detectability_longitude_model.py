from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special

from .detectability_model import (
    PresentMassProxyModel,
    aggregate_two_component_selection_stats,
    apply_effective_completeness_to_component_contexts,
    best_pair_component_row,
    build_best_component_catalog_prediction_table,
    build_best_component_imf_grid_table,
    build_best_component_radial_grid_table,
    build_component_summary_table,
    build_detectability_corrected_performance_row,
    build_separate_two_component_detectability_summary_row,
    build_shared_best_component_catalog_prediction_table,
    build_shared_best_component_imf_grid_table,
    build_shared_best_component_radial_grid_table,
    build_shared_best_component_summary_table,
    build_shared_two_component_detectability_summary_row,
    build_two_component_pair_payloads,
    build_mass_bin_probabilities_grid,
    compute_complete_survivor_intensity_grid,
    compute_separate_complete_survivor_intensity_grid,
    compute_shared_complete_survivor_intensity_grid,
    fit_present_mass_proxy_model,
    predict_log_present_mass_grid,
)
from .joint_model import (
    JointLikelihoodContext,
    JointModelSpec,
    build_fixed_survival_grid,
    calibrate_fixed_selection_offset_dex,
    centers_to_edges_local,
    fit_single_joint_model_with_fixed_imf_params,
    fit_single_joint_model,
)
from .two_component_model import (
    SharedImfTwoComponentSpec,
    SplitAlphaTwoComponentSpec,
    build_split_alpha_best_component_catalog_prediction_table,
    build_split_alpha_best_component_imf_grid_table,
    build_split_alpha_best_component_radial_grid_table,
    build_split_alpha_best_component_summary_table,
    build_split_alpha_two_component_specs,
    build_shared_imf_two_component_specs,
    fit_split_alpha_two_component_single_model,
    fit_shared_imf_two_component_single_model,
    prepare_two_component_contexts,
)


@dataclass
class ObservablePredictionContextWithLongitude:
    present_mass_proxy: PresentMassProxyModel
    log_present_mass_edges: np.ndarray
    distance_edges_kpc: np.ndarray
    abs_latitude_edges_deg: np.ndarray
    abs_longitude_edges_deg: np.ndarray
    log_present_mass_centers: np.ndarray
    log_distance_centers: np.ndarray
    abs_latitude_centers_deg: np.ndarray
    abs_longitude_centers_deg: np.ndarray
    observed_counts: np.ndarray
    mass_bin_probabilities_grid: np.ndarray
    sky_bin_probabilities_by_a: np.ndarray
    log_present_mass_mean_grid: np.ndarray
    log_present_mass_feature_mean: float
    log_present_mass_feature_std: float
    log_distance_feature_mean: float
    log_distance_feature_std: float
    abs_latitude_feature_mean: float
    abs_latitude_feature_std: float
    abs_longitude_feature_mean: float
    abs_longitude_feature_std: float
    sun_galactocentric_radius_kpc: float
    n_geometry_samples: int


@dataclass
class DetectabilityAbsLongitudeIterationSummary:
    iteration: int
    log_likelihood: float
    rms_residual_sigma_2d: float
    mean_abs_residual_sigma_2d: float
    total_initial_count: float
    total_initial_count_above_log10_4: float
    selection_fraction: float
    raw_survival_fraction: float
    completeness_mean: float
    selection_fraction_above_log10_4: float
    raw_survival_fraction_above_log10_4: float
    completeness_mean_above_log10_4: float
    completeness_intercept: float
    completeness_mass_slope: float
    completeness_distance_slope: float
    completeness_latitude_slope: float
    completeness_longitude_slope: float
    predicted_complete_survivor_count: float
    predicted_observed_count: float


@dataclass
class DetectabilityAbsLongitudeTwoComponentIterationSummary:
    iteration: int
    log_likelihood: float
    total_initial_count: float
    selection_fraction: float
    raw_survival_fraction: float
    completeness_mean: float
    completeness_intercept: float
    completeness_mass_slope: float
    completeness_distance_slope: float
    completeness_latitude_slope: float
    completeness_longitude_slope: float
    predicted_complete_survivor_count: float
    predicted_observed_count: float


def _restrict_log_mass_support_local(
    log_mass_grid: np.ndarray,
    values: np.ndarray,
    log_mass_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(log_mass_grid, dtype=float)
    y = np.asarray(values, dtype=float)
    if log_mass_min <= float(x[0]):
        return x.copy(), y.copy()
    if log_mass_min >= float(x[-1]):
        return np.asarray([float(x[-1])], dtype=float), np.asarray([float(y[-1])], dtype=float)
    mask = x > log_mass_min
    y0 = float(np.interp(log_mass_min, x, y))
    return (
        np.concatenate(([float(log_mass_min)], x[mask])),
        np.concatenate(([y0], y[mask])),
    )


def _single_component_reporting_stats_above_log_mass(
    *,
    context,
    model: dict[str, object],
    log_mass_min: float,
) -> dict[str, float]:
    log_mass_grid = np.asarray(context.log_mass_grid, dtype=float)
    imf_density_grid = np.asarray(model["imf_density_grid"], dtype=float)
    radial_density_grid = np.asarray(model["radial_density_grid"], dtype=float)
    mass_support, imf_support = _restrict_log_mass_support_local(log_mass_grid, imf_density_grid, log_mass_min)
    imf_fraction_above = float(np.trapezoid(imf_support, mass_support))
    total_initial_count_above = float(model["total_initial_count"]) * imf_fraction_above

    radial_support = np.asarray(context.log_a_grid, dtype=float)
    survival_grid = np.asarray(context.survival_probability_grid, dtype=float)
    selection_grid = np.asarray(context.selection_probability_grid, dtype=float)
    survival_support = np.vstack(
        [np.interp(mass_support, log_mass_grid, survival_grid[:, j]) for j in range(survival_grid.shape[1])]
    ).T
    selection_support = np.vstack(
        [np.interp(mass_support, log_mass_grid, selection_grid[:, j]) for j in range(selection_grid.shape[1])]
    ).T
    integrand_base = imf_support[:, None] * radial_density_grid[None, :]
    raw_survival_fraction_above = float(np.trapezoid(np.trapezoid(integrand_base * survival_support, radial_support, axis=1), mass_support))
    selection_fraction_above = float(np.trapezoid(np.trapezoid(integrand_base * selection_support, radial_support, axis=1), mass_support))
    completeness_mean_above = float(selection_fraction_above / max(raw_survival_fraction_above, 1.0e-12))
    return {
        "total_initial_count_above": total_initial_count_above,
        "selection_fraction_above": selection_fraction_above,
        "raw_survival_fraction_above": raw_survival_fraction_above,
        "completeness_mean_above": completeness_mean_above,
    }


def fit_detectability_corrected_single_component_models_with_abs_longitude(
    catalog: pd.DataFrame,
    project_root: Path,
    model_specs: list[JointModelSpec] | None = None,
    survival_grid_override: dict[str, object] | None = None,
    **kwargs,
) -> dict[str, object]:
    if model_specs is None:
        model_specs = [
            JointModelSpec(imf_family="lognormal", radial_model="step5"),
            JointModelSpec(imf_family="powerlaw", radial_model="step5"),
            JointModelSpec(imf_family="schechter", radial_model="step5"),
            JointModelSpec(imf_family="lognormal", radial_model="logpoly3"),
            JointModelSpec(imf_family="powerlaw", radial_model="logpoly3"),
            JointModelSpec(imf_family="schechter", radial_model="logpoly3"),
        ]

    all_results = []
    summary_rows: list[dict[str, object]] = []
    for spec in model_specs:
        result = fit_single_component_detectability_em_with_abs_longitude(
            catalog=catalog,
            project_root=project_root,
            spec=spec,
            survival_grid_override=survival_grid_override,
            **kwargs,
        )
        all_results.append(result)
        summary_rows.append(build_detectability_corrected_performance_row(result))

    summary_table = pd.DataFrame(summary_rows).sort_values(
        ["log_likelihood", "rms_residual_sigma_2d"],
        ascending=[False, True],
    ).reset_index(drop=True)
    best_spec = JointModelSpec(
        imf_family=str(summary_table.iloc[0]["imf_family"]),
        radial_model=str(summary_table.iloc[0]["radial_model"]),
    )
    best_result = next(
        result
        for result in all_results
        if result["spec"].imf_family == best_spec.imf_family and result["spec"].radial_model == best_spec.radial_model
    )
    best_log_likelihood = float(summary_table["log_likelihood"].max())
    summary_table["delta_log_likelihood"] = best_log_likelihood - summary_table["log_likelihood"]
    write_best_abs_longitude_detectability_outputs(best_result, summary_table, project_root)
    return {
        "all_results": all_results,
        "summary_table": summary_table,
        "best_result": best_result,
    }


def fit_single_component_detectability_em_with_abs_longitude(
    catalog: pd.DataFrame,
    project_root: Path,
    spec: JointModelSpec | None = None,
    n_iterations: int = 12,
    relaxation: float = 0.7,
    n_present_mass_bins: int = 6,
    n_distance_bins: int = 6,
    n_latitude_bins: int = 6,
    n_longitude_bins: int = 6,
    n_geometry_samples: int = 5000,
    sun_galactocentric_radius_kpc: float = 8.2,
    start_completeness_raw_parameters: np.ndarray | None = None,
    fixed_imf_params: np.ndarray | None = None,
    start_radial_params: np.ndarray | None = None,
    survival_grid_override: dict[str, object] | None = None,
) -> dict[str, object]:
    if spec is None:
        spec = JointModelSpec(imf_family="schechter", radial_model="logpoly3")

    working = catalog.copy()
    required_columns = {
        "log_initial_mass_msun",
        "semi_major_axis_kpc",
        "log_survival_mass_cut_msun",
        "present_mass_msun",
        "r_sun_kpc",
        "galactic_b_deg",
        "galactic_l_deg",
    }
    missing = required_columns.difference(working.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Catalog is missing required columns for longitude-aware detectability fitting: {missing_list}")

    if survival_grid_override is None:
        selection_offset_dex = calibrate_fixed_selection_offset_dex(working)
        survival_grid = build_fixed_survival_grid(
            working,
            selection_offset_dex=selection_offset_dex,
        )
    else:
        selection_offset_dex = float(survival_grid_override.get("selection_offset_dex", np.nan))
        survival_grid = survival_grid_override
    base_context = JointLikelihoodContext.from_catalog_and_survival_grid(working, survival_grid)
    observable_context = build_observable_prediction_context_with_abs_longitude(
        catalog=working,
        base_context=base_context,
        n_present_mass_bins=n_present_mass_bins,
        n_distance_bins=n_distance_bins,
        n_latitude_bins=n_latitude_bins,
        n_longitude_bins=n_longitude_bins,
        n_geometry_samples=n_geometry_samples,
        sun_galactocentric_radius_kpc=sun_galactocentric_radius_kpc,
    )

    baseline_context = base_context.with_selection_probability_grid(base_context.survival_probability_grid)
    if fixed_imf_params is None:
        baseline_payload = fit_single_joint_model(context=baseline_context, spec=spec)
        current_radial_params = None
    else:
        baseline_payload = fit_single_joint_model_with_fixed_imf_params(
            context=baseline_context,
            spec=spec,
            fixed_imf_params=np.asarray(fixed_imf_params, dtype=float),
            start_radial_params=start_radial_params,
        )
        current_radial_params = np.asarray(baseline_payload["radial_parameters_raw"], dtype=float)
    if start_completeness_raw_parameters is None:
        current_raw_params = fit_logistic_completeness_model_with_abs_longitude(
            observable_context=observable_context,
            predicted_complete_counts=predict_complete_observable_histogram_with_abs_longitude(
                complete_survivor_intensity_grid=compute_complete_survivor_intensity_grid(
                    baseline_payload["model"],
                    base_context=base_context,
                ),
                base_context=base_context,
                observable_context=observable_context,
            ),
            start_params=None,
        )["raw_parameters"]
    else:
        current_raw_params = np.asarray(start_completeness_raw_parameters, dtype=float).copy()

    iteration_rows: list[DetectabilityAbsLongitudeIterationSummary] = []
    current_context = baseline_context
    current_payload = baseline_payload
    current_effective_completeness_grid = np.ones_like(base_context.survival_probability_grid)
    report_log_mass_min = 4.0

    for iteration in range(1, n_iterations + 1):
        completeness_bin_grid = evaluate_completeness_bin_grid_with_abs_longitude(
            current_raw_params,
            observable_context,
        )
        current_effective_completeness_grid = compute_effective_completeness_grid_with_abs_longitude(
            observable_context=observable_context,
            completeness_bin_grid=completeness_bin_grid,
        )
        current_context = base_context.with_selection_probability_grid(
            np.clip(
                base_context.survival_probability_grid * current_effective_completeness_grid,
                1.0e-12,
                1.0,
            )
        )
        if fixed_imf_params is None:
            current_payload = fit_single_joint_model(context=current_context, spec=spec)
        else:
            current_payload = fit_single_joint_model_with_fixed_imf_params(
                context=current_context,
                spec=spec,
                fixed_imf_params=np.asarray(fixed_imf_params, dtype=float),
                start_radial_params=current_radial_params,
            )
            current_radial_params = np.asarray(current_payload["radial_parameters_raw"], dtype=float)
        performance_row = build_detectability_corrected_performance_row(
            {
                "final_context": current_context,
                "final_payload": current_payload,
            }
        )
        complete_survivor_intensity_grid = compute_complete_survivor_intensity_grid(
            current_payload["model"],
            base_context=base_context,
        )
        predicted_complete_counts = predict_complete_observable_histogram_with_abs_longitude(
            complete_survivor_intensity_grid=complete_survivor_intensity_grid,
            base_context=base_context,
            observable_context=observable_context,
        )
        completeness_fit = fit_logistic_completeness_model_with_abs_longitude(
            observable_context=observable_context,
            predicted_complete_counts=predicted_complete_counts,
            start_params=current_raw_params,
        )
        target_raw_params = completeness_fit["raw_parameters"]
        current_raw_params = (1.0 - relaxation) * current_raw_params + relaxation * target_raw_params
        updated_completeness_bin_grid = evaluate_completeness_bin_grid_with_abs_longitude(
            current_raw_params,
            observable_context,
        )
        predicted_observed_counts = predicted_complete_counts * updated_completeness_bin_grid
        report_stats = _single_component_reporting_stats_above_log_mass(
            context=current_context,
            model=current_payload["model"],
            log_mass_min=report_log_mass_min,
        )
        iteration_rows.append(
            DetectabilityAbsLongitudeIterationSummary(
                iteration=iteration,
                log_likelihood=float(current_payload["summary"].log_likelihood),
                rms_residual_sigma_2d=float(performance_row["rms_residual_sigma_2d"]),
                mean_abs_residual_sigma_2d=float(performance_row["mean_abs_residual_sigma_2d"]),
                total_initial_count=float(current_payload["model"]["total_initial_count"]),
                total_initial_count_above_log10_4=float(report_stats["total_initial_count_above"]),
                selection_fraction=float(current_payload["model"]["selection_fraction"]),
                raw_survival_fraction=float(current_payload["model"]["raw_survival_fraction"]),
                completeness_mean=float(
                    np.sum(predicted_complete_counts * updated_completeness_bin_grid)
                    / max(np.sum(predicted_complete_counts), 1.0e-12)
                ),
                selection_fraction_above_log10_4=float(report_stats["selection_fraction_above"]),
                raw_survival_fraction_above_log10_4=float(report_stats["raw_survival_fraction_above"]),
                completeness_mean_above_log10_4=float(report_stats["completeness_mean_above"]),
                completeness_intercept=float(current_raw_params[0]),
                completeness_mass_slope=float(np.exp(current_raw_params[1])),
                completeness_distance_slope=float(np.exp(current_raw_params[2])),
                completeness_latitude_slope=float(np.exp(current_raw_params[3])),
                completeness_longitude_slope=float(np.exp(current_raw_params[4])),
                predicted_complete_survivor_count=float(np.sum(predicted_complete_counts)),
                predicted_observed_count=float(np.sum(predicted_observed_counts)),
            )
        )

    final_completeness_bin_grid = evaluate_completeness_bin_grid_with_abs_longitude(
        current_raw_params,
        observable_context,
    )
    final_effective_completeness_grid = compute_effective_completeness_grid_with_abs_longitude(
        observable_context=observable_context,
        completeness_bin_grid=final_completeness_bin_grid,
    )
    final_context = base_context.with_selection_probability_grid(
        np.clip(
            base_context.survival_probability_grid * final_effective_completeness_grid,
            1.0e-12,
            1.0,
        )
    )
    if fixed_imf_params is None:
        final_payload = fit_single_joint_model(context=final_context, spec=spec)
    else:
        final_payload = fit_single_joint_model_with_fixed_imf_params(
            context=final_context,
            spec=spec,
            fixed_imf_params=np.asarray(fixed_imf_params, dtype=float),
            start_radial_params=current_radial_params,
        )
    final_complete_survivor_intensity_grid = compute_complete_survivor_intensity_grid(
        final_payload["model"],
        base_context=base_context,
    )
    final_predicted_complete_counts = predict_complete_observable_histogram_with_abs_longitude(
        complete_survivor_intensity_grid=final_complete_survivor_intensity_grid,
        base_context=base_context,
        observable_context=observable_context,
    )
    final_predicted_observed_counts = final_predicted_complete_counts * final_completeness_bin_grid
    baseline_report_stats = _single_component_reporting_stats_above_log_mass(
        context=baseline_context,
        model=baseline_payload["model"],
        log_mass_min=report_log_mass_min,
    )
    final_report_stats = _single_component_reporting_stats_above_log_mass(
        context=final_context,
        model=final_payload["model"],
        log_mass_min=report_log_mass_min,
    )

    iteration_history_table = pd.DataFrame([asdict(row) for row in iteration_rows])
    completeness_grid_table = build_completeness_grid_table_with_abs_longitude(
        observable_context,
        final_completeness_bin_grid,
    )
    observable_histogram_table = build_observable_histogram_table_with_abs_longitude(
        observable_context=observable_context,
        predicted_complete_counts=final_predicted_complete_counts,
        predicted_observed_counts=final_predicted_observed_counts,
        completeness_bin_grid=final_completeness_bin_grid,
    )
    catalog_completeness_table = build_catalog_completeness_table_with_abs_longitude(
        catalog=working,
        context=final_context,
        observable_context=observable_context,
        completeness_raw_params=current_raw_params,
    )

    summary_payload = {
        "spec": asdict(spec),
        "selection_offset_dex": selection_offset_dex,
        "sun_galactocentric_radius_kpc": sun_galactocentric_radius_kpc,
        "n_iterations": n_iterations,
        "relaxation": relaxation,
        "n_longitude_bins": n_longitude_bins,
        "baseline_total_initial_count": float(baseline_payload["model"]["total_initial_count"]),
        "baseline_total_initial_count_above_log10_4": float(baseline_report_stats["total_initial_count_above"]),
        "baseline_raw_survival_fraction": float(baseline_payload["model"]["raw_survival_fraction"]),
        "final_total_initial_count": float(final_payload["model"]["total_initial_count"]),
        "final_total_initial_count_above_log10_4": float(final_report_stats["total_initial_count_above"]),
        "final_selection_fraction": float(final_payload["model"]["selection_fraction"]),
        "final_raw_survival_fraction": float(final_payload["model"]["raw_survival_fraction"]),
        "final_mean_detectability": float(
            final_payload["model"]["selection_fraction"] / max(final_payload["model"]["raw_survival_fraction"], 1.0e-12)
        ),
        "final_mean_detectability_above_log10_4": float(final_report_stats["completeness_mean_above"]),
        "total_initial_count_ratio_vs_baseline": float(
            final_payload["model"]["total_initial_count"] / max(baseline_payload["model"]["total_initial_count"], 1.0e-12)
        ),
        "total_initial_count_ratio_vs_baseline_above_log10_4": float(
            final_report_stats["total_initial_count_above"]
            / max(baseline_report_stats["total_initial_count_above"], 1.0e-12)
        ),
        "present_mass_proxy": {
            "model_kind": getattr(observable_context.present_mass_proxy, "model_kind", "polynomial_log_mass_ratio"),
            "coefficients": observable_context.present_mass_proxy.coefficients.tolist(),
            "log_mass_mean": observable_context.present_mass_proxy.log_mass_mean,
            "log_a_mean": observable_context.present_mass_proxy.log_a_mean,
            "log_mass_std": getattr(observable_context.present_mass_proxy, "log_mass_std", 1.0),
            "log_a_std": getattr(observable_context.present_mass_proxy, "log_a_std", 1.0),
            "residual_sigma_dex": observable_context.present_mass_proxy.residual_sigma_dex,
        },
        "final_completeness_parameters": {
            "intercept": float(current_raw_params[0]),
            "mass_slope": float(np.exp(current_raw_params[1])),
            "distance_slope": float(np.exp(current_raw_params[2])),
            "latitude_slope": float(np.exp(current_raw_params[3])),
            "longitude_slope": float(np.exp(current_raw_params[4])),
        },
        "baseline_model": asdict(baseline_payload["summary"]),
        "final_model": {
            **asdict(final_payload["summary"]),
            "raw_survival_fraction": float(final_payload["model"]["raw_survival_fraction"]),
            "selection_fraction": float(final_payload["model"]["selection_fraction"]),
        },
        "iteration_history": iteration_history_table.to_dict(orient="records"),
    }

    return {
        "spec": spec,
        "selection_offset_dex": selection_offset_dex,
        "base_context": base_context,
        "final_context": final_context,
        "observable_context": observable_context,
        "baseline_payload": baseline_payload,
        "final_payload": final_payload,
        "iteration_history_table": iteration_history_table,
        "completeness_grid_table": completeness_grid_table,
        "observable_histogram_table": observable_histogram_table,
        "catalog_completeness_table": catalog_completeness_table,
        "final_completeness_raw_parameters": current_raw_params,
        "final_completeness_bin_grid": final_completeness_bin_grid,
        "final_effective_completeness_grid": final_effective_completeness_grid,
        "final_predicted_complete_counts": final_predicted_complete_counts,
        "final_predicted_observed_counts": final_predicted_observed_counts,
        "final_complete_survivor_intensity_grid": final_complete_survivor_intensity_grid,
        "summary_payload": summary_payload,
    }


def write_best_abs_longitude_detectability_outputs(
    best_result: dict[str, object],
    summary_table: pd.DataFrame,
    project_root: Path,
    prefix: str = "joint_fixed_survival_detectability_abs_longitude_em",
) -> None:
    outputs_tables = project_root / "outputs" / "tables"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    best_result["iteration_history_table"].to_csv(
        outputs_tables / f"{prefix}_iteration_history.csv",
        index=False,
    )
    best_result["completeness_grid_table"].to_csv(
        outputs_tables / f"{prefix}_completeness_grid.csv",
        index=False,
    )
    best_result["observable_histogram_table"].to_csv(
        outputs_tables / f"{prefix}_observable_histogram.csv",
        index=False,
    )
    best_result["catalog_completeness_table"].to_csv(
        outputs_tables / f"{prefix}_catalog_completeness.csv",
        index=False,
    )
    summary_table.to_csv(
        outputs_tables / f"{prefix}_model_summary.csv",
        index=False,
    )
    (outputs_tables / f"{prefix}_summary.json").write_text(
        json.dumps(
            {
                "best_joint_model": asdict(best_result["final_payload"]["summary"]),
                "best_model_detectability_summary": best_result["summary_payload"],
                "model_comparison": summary_table.to_dict(orient="records"),
            },
            indent=2,
        )
    )


def prepare_two_component_detectability_environment_with_abs_longitude(
    catalog: pd.DataFrame,
    n_present_mass_bins: int = 6,
    n_distance_bins: int = 6,
    n_latitude_bins: int = 6,
    n_longitude_bins: int = 6,
    n_geometry_samples: int = 5000,
    sun_galactocentric_radius_kpc: float = 8.2,
) -> dict[str, object]:
    working, subsets, selection_offset_dex, survival_grid, component_base_contexts = prepare_two_component_contexts(catalog)
    base_context = JointLikelihoodContext.from_catalog_and_survival_grid(working, survival_grid)
    observable_context = build_observable_prediction_context_with_abs_longitude(
        catalog=working,
        base_context=base_context,
        n_present_mass_bins=n_present_mass_bins,
        n_distance_bins=n_distance_bins,
        n_latitude_bins=n_latitude_bins,
        n_longitude_bins=n_longitude_bins,
        n_geometry_samples=n_geometry_samples,
        sun_galactocentric_radius_kpc=sun_galactocentric_radius_kpc,
    )
    return {
        "working": working,
        "subsets": subsets,
        "selection_offset_dex": selection_offset_dex,
        "survival_grid": survival_grid,
        "base_context": base_context,
        "observable_context": observable_context,
        "component_base_contexts": component_base_contexts,
    }


def fit_shared_imf_two_component_detectability_em_models_with_abs_longitude(
    catalog: pd.DataFrame,
    project_root: Path,
    imf_families: list[str] | None = None,
    radial_models: list[str] | None = None,
    fixed_effective_completeness_grid: np.ndarray | None = None,
    fixed_completeness_bin_grid: np.ndarray | None = None,
    fixed_completeness_raw_parameters: np.ndarray | None = None,
    **kwargs,
) -> dict[str, object]:
    if imf_families is None:
        imf_families = ["lognormal", "powerlaw", "schechter"]
    if radial_models is None:
        radial_models = ["step5", "logpoly3"]

    env = prepare_two_component_detectability_environment_with_abs_longitude(catalog=catalog, **kwargs)
    specs = build_shared_imf_two_component_specs(imf_families=imf_families, radial_models=radial_models)
    all_results = []
    summary_rows: list[dict[str, object]] = []
    for spec in specs:
        result = fit_shared_imf_two_component_detectability_em_single_model_with_abs_longitude(
            spec=spec,
            fixed_effective_completeness_grid=fixed_effective_completeness_grid,
            fixed_completeness_bin_grid=fixed_completeness_bin_grid,
            fixed_completeness_raw_parameters=fixed_completeness_raw_parameters,
            **env,
        )
        all_results.append(result)
        summary_rows.append(build_shared_two_component_detectability_summary_row(result))

    summary_table = pd.DataFrame(summary_rows).sort_values("bic", ascending=True).reset_index(drop=True)
    best_bic = float(summary_table["bic"].min())
    summary_table["delta_bic"] = summary_table["bic"] - best_bic
    best_key = (
        str(summary_table.iloc[0]["imf_family"]),
        str(summary_table.iloc[0]["in_situ_radial_model"]),
        str(summary_table.iloc[0]["accreted_radial_model"]),
    )
    best_result = next(
        result
        for result in all_results
        if (
            result["spec"].imf_family,
            result["spec"].in_situ_radial_model,
            result["spec"].accreted_radial_model,
        )
        == best_key
    )
    for result in all_results:
        result["final_payload"]["summary"].delta_bic = float(result["final_payload"]["summary"].bic - best_bic)

    best_component_summary_table = build_shared_best_component_summary_table(
        best_payload=best_result["final_payload"],
        n_clusters_by_component={label: len(subset) for label, subset in env["subsets"].items()},
    )
    best_imf_grid_table = build_shared_best_component_imf_grid_table(
        best_payload=best_result["final_payload"],
        contexts=best_result["final_contexts"],
    )
    best_radial_grid_table = build_shared_best_component_radial_grid_table(
        best_payload=best_result["final_payload"],
        contexts=best_result["final_contexts"],
    )
    catalog_prediction_table = build_shared_best_component_catalog_prediction_table(
        subsets=env["subsets"],
        contexts=best_result["final_contexts"],
        best_payload=best_result["final_payload"],
    )

    outputs_tables = project_root / "outputs" / "tables"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    summary_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_abs_longitude_shared_imf_two_component_model_summary.csv",
        index=False,
    )
    best_component_summary_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_abs_longitude_shared_imf_two_component_best_component_summary.csv",
        index=False,
    )
    best_imf_grid_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_abs_longitude_shared_imf_two_component_best_imf_grids.csv",
        index=False,
    )
    best_radial_grid_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_abs_longitude_shared_imf_two_component_best_radial_grids.csv",
        index=False,
    )
    catalog_prediction_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_abs_longitude_shared_imf_two_component_catalog_predictions.csv",
        index=False,
    )
    best_result["iteration_history_table"].to_csv(
        outputs_tables / "joint_fixed_survival_detectability_abs_longitude_shared_imf_two_component_best_em_iteration_history.csv",
        index=False,
    )
    best_result["completeness_grid_table"].to_csv(
        outputs_tables / "joint_fixed_survival_detectability_abs_longitude_shared_imf_two_component_best_completeness_grid.csv",
        index=False,
    )
    best_result["observable_histogram_table"].to_csv(
        outputs_tables / "joint_fixed_survival_detectability_abs_longitude_shared_imf_two_component_best_observable_histogram.csv",
        index=False,
    )
    best_result["catalog_completeness_table"].to_csv(
        outputs_tables / "joint_fixed_survival_detectability_abs_longitude_shared_imf_two_component_best_catalog_completeness.csv",
        index=False,
    )

    detailed_summary = {
        "selection_offset_dex": env["selection_offset_dex"],
        "survival_grid_bandwidth_log10_a_dex": env["survival_grid"]["bandwidth_log10_a_dex"],
        "n_clusters_total": int(len(env["working"])),
        "n_clusters_in_situ": int(len(env["subsets"]["in_situ"])),
        "n_clusters_accreted": int(len(env["subsets"]["accreted"])),
        "best_joint_model": asdict(best_result["final_payload"]["summary"]),
        "best_component_models": best_component_summary_table.to_dict(orient="records"),
        "all_joint_models_ranked": summary_table.to_dict(orient="records"),
        "best_model_detectability_summary": best_result["summary_payload"],
    }
    (
        outputs_tables / "joint_fixed_survival_detectability_abs_longitude_shared_imf_two_component_model_summary.json"
    ).write_text(json.dumps(detailed_summary, indent=2))

    return {
        "summary_table": summary_table,
        "best_component_summary_table": best_component_summary_table,
        "best_imf_grid_table": best_imf_grid_table,
        "best_radial_grid_table": best_radial_grid_table,
        "catalog_prediction_table": catalog_prediction_table,
        "best_result": best_result,
        "all_results": all_results,
        "contexts": best_result["final_contexts"],
        "subsets": env["subsets"],
        "base_context": env["base_context"],
        "observable_context": env["observable_context"],
        "survival_grid": env["survival_grid"],
    }


def compute_split_alpha_complete_survivor_intensity_grid(
    model: dict[str, object],
    base_context: JointLikelihoodContext,
) -> np.ndarray:
    total = np.zeros_like(base_context.survival_probability_grid)
    for component_label in ("in_situ", "accreted"):
        total += compute_complete_survivor_intensity_grid(
            model={
                "imf_density_grid": model["imf_density_grid"][component_label],
                "radial_density_grid": model["radial_density_grid"][component_label],
                "total_initial_count": model["total_initial_count"][component_label],
            },
            base_context=base_context,
        )
    return total


def resolve_fixed_completeness_inputs_with_abs_longitude(
    observable_context: ObservablePredictionContextWithLongitude,
    fixed_completeness_bin_grid: np.ndarray | None,
    fixed_completeness_raw_parameters: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    raw_params = (
        np.asarray(fixed_completeness_raw_parameters, dtype=float)
        if fixed_completeness_raw_parameters is not None
        else None
    )
    if fixed_completeness_bin_grid is not None:
        bin_grid = np.asarray(fixed_completeness_bin_grid, dtype=float)
    elif raw_params is not None:
        bin_grid = evaluate_completeness_bin_grid_with_abs_longitude(raw_params, observable_context)
    else:
        raise ValueError(
            "A fixed two-component detectability fit requires fixed_completeness_bin_grid "
            "or fixed_completeness_raw_parameters from the single-component run."
        )
    if raw_params is None:
        raise ValueError(
            "A fixed two-component detectability fit requires fixed_completeness_raw_parameters "
            "so the per-cluster completeness table can be evaluated consistently."
        )
    return bin_grid, raw_params


def build_fixed_two_component_detectability_result_with_abs_longitude(
    working: pd.DataFrame,
    subsets: dict[str, pd.DataFrame],
    base_context: JointLikelihoodContext,
    observable_context: ObservablePredictionContextWithLongitude,
    baseline_payload: dict[str, object],
    final_payload: dict[str, object],
    final_contexts: dict[str, JointLikelihoodContext],
    final_effective_completeness_grid: np.ndarray,
    final_completeness_bin_grid: np.ndarray,
    final_completeness_raw_parameters: np.ndarray,
    complete_survivor_intensity_grid: np.ndarray,
    spec_payload: dict[str, object],
) -> dict[str, object]:
    final_predicted_complete_counts = predict_complete_observable_histogram_with_abs_longitude(
        complete_survivor_intensity_grid=complete_survivor_intensity_grid,
        base_context=base_context,
        observable_context=observable_context,
    )
    final_predicted_observed_counts = final_predicted_complete_counts * final_completeness_bin_grid
    selection_stats = aggregate_two_component_selection_stats(
        counts_by_component={label: len(subset) for label, subset in subsets.items()},
        total_initial_count_by_component=final_payload["model"]["total_initial_count"],
        raw_survival_fraction_by_component=final_payload["model"]["raw_survival_fraction"],
        selection_fraction_by_component=final_payload["model"]["selection_fraction"],
    )
    iteration_rows = [
        DetectabilityAbsLongitudeTwoComponentIterationSummary(
            iteration=0,
            log_likelihood=float(final_payload["summary"].log_likelihood),
            total_initial_count=float(selection_stats["total_initial_count"]),
            selection_fraction=float(selection_stats["selection_fraction"]),
            raw_survival_fraction=float(selection_stats["raw_survival_fraction"]),
            completeness_mean=float(selection_stats["mean_detectability"]),
            completeness_intercept=float(final_completeness_raw_parameters[0]),
            completeness_mass_slope=float(np.exp(final_completeness_raw_parameters[1])),
            completeness_distance_slope=float(np.exp(final_completeness_raw_parameters[2])),
            completeness_latitude_slope=float(np.exp(final_completeness_raw_parameters[3])),
            completeness_longitude_slope=float(np.exp(final_completeness_raw_parameters[4])),
            predicted_complete_survivor_count=float(np.sum(final_predicted_complete_counts)),
            predicted_observed_count=float(np.sum(final_predicted_observed_counts)),
        )
    ]
    iteration_history_table = pd.DataFrame([asdict(row) for row in iteration_rows])
    completeness_grid_table = build_completeness_grid_table_with_abs_longitude(
        observable_context,
        final_completeness_bin_grid,
    )
    observable_histogram_table = build_observable_histogram_table_with_abs_longitude(
        observable_context=observable_context,
        predicted_complete_counts=final_predicted_complete_counts,
        predicted_observed_counts=final_predicted_observed_counts,
        completeness_bin_grid=final_completeness_bin_grid,
    )
    catalog_completeness_table = build_catalog_completeness_table_with_abs_longitude(
        catalog=working,
        context=base_context.with_selection_probability_grid(
            np.clip(base_context.survival_probability_grid * final_effective_completeness_grid, 1.0e-12, 1.0)
        ),
        observable_context=observable_context,
        completeness_raw_params=final_completeness_raw_parameters,
    )
    summary_payload = {
        "spec": spec_payload,
        "detectability_mode": "fixed_from_single_component",
        "baseline_total_initial_count": float(
            baseline_payload["model"]["total_initial_count"]["in_situ"]
            + baseline_payload["model"]["total_initial_count"]["accreted"]
        ),
        "final_total_initial_count": float(selection_stats["total_initial_count"]),
        "final_selection_fraction": float(selection_stats["selection_fraction"]),
        "final_raw_survival_fraction": float(selection_stats["raw_survival_fraction"]),
        "final_mean_detectability": float(selection_stats["mean_detectability"]),
        "total_initial_count_ratio_vs_baseline": float(
            selection_stats["total_initial_count"]
            / max(
                baseline_payload["model"]["total_initial_count"]["in_situ"]
                + baseline_payload["model"]["total_initial_count"]["accreted"],
                1.0e-12,
            )
        ),
        "final_completeness_parameters": {
            "intercept": float(final_completeness_raw_parameters[0]),
            "mass_slope": float(np.exp(final_completeness_raw_parameters[1])),
            "distance_slope": float(np.exp(final_completeness_raw_parameters[2])),
            "latitude_slope": float(np.exp(final_completeness_raw_parameters[3])),
            "longitude_slope": float(np.exp(final_completeness_raw_parameters[4])),
        },
        "baseline_model": asdict(baseline_payload["summary"]),
        "final_model": asdict(final_payload["summary"]),
        "iteration_history": iteration_history_table.to_dict(orient="records"),
    }
    return {
        "baseline_payload": baseline_payload,
        "final_payload": final_payload,
        "final_contexts": final_contexts,
        "iteration_history_table": iteration_history_table,
        "completeness_grid_table": completeness_grid_table,
        "observable_histogram_table": observable_histogram_table,
        "catalog_completeness_table": catalog_completeness_table,
        "final_completeness_raw_parameters": final_completeness_raw_parameters,
        "final_completeness_bin_grid": final_completeness_bin_grid,
        "final_effective_completeness_grid": final_effective_completeness_grid,
        "final_predicted_complete_counts": final_predicted_complete_counts,
        "final_predicted_observed_counts": final_predicted_observed_counts,
        "summary_payload": summary_payload,
    }


def fit_shared_imf_two_component_detectability_em_single_model_with_abs_longitude(
    working: pd.DataFrame,
    subsets: dict[str, pd.DataFrame],
    base_context: JointLikelihoodContext,
    observable_context: ObservablePredictionContextWithLongitude,
    component_base_contexts: dict[str, JointLikelihoodContext],
    spec: SharedImfTwoComponentSpec,
    n_iterations: int = 12,
    relaxation: float = 0.7,
    fixed_effective_completeness_grid: np.ndarray | None = None,
    fixed_completeness_bin_grid: np.ndarray | None = None,
    fixed_completeness_raw_parameters: np.ndarray | None = None,
    **_: object,
) -> dict[str, object]:
    baseline_payload = fit_shared_imf_two_component_single_model(contexts=component_base_contexts, spec=spec)
    if fixed_effective_completeness_grid is not None:
        final_completeness_bin_grid, final_completeness_raw_parameters = resolve_fixed_completeness_inputs_with_abs_longitude(
            observable_context=observable_context,
            fixed_completeness_bin_grid=fixed_completeness_bin_grid,
            fixed_completeness_raw_parameters=fixed_completeness_raw_parameters,
        )
        final_contexts = apply_effective_completeness_to_component_contexts(
            component_base_contexts=component_base_contexts,
            effective_completeness_grid=fixed_effective_completeness_grid,
        )
        final_payload = fit_shared_imf_two_component_single_model(contexts=final_contexts, spec=spec)
        result = build_fixed_two_component_detectability_result_with_abs_longitude(
            working=working,
            subsets=subsets,
            base_context=base_context,
            observable_context=observable_context,
            baseline_payload=baseline_payload,
            final_payload=final_payload,
            final_contexts=final_contexts,
            final_effective_completeness_grid=fixed_effective_completeness_grid,
            final_completeness_bin_grid=final_completeness_bin_grid,
            final_completeness_raw_parameters=final_completeness_raw_parameters,
            complete_survivor_intensity_grid=compute_shared_complete_survivor_intensity_grid(
                final_payload["model"],
                base_context=base_context,
            ),
            spec_payload={
                "imf_family": spec.imf_family,
                "detectability_mode": "fixed_from_single_component",
                "in_situ_radial_model": spec.in_situ_radial_model,
                "accreted_radial_model": spec.accreted_radial_model,
            },
        )
        result["spec"] = spec
        return result

    current_raw_params = fit_logistic_completeness_model_with_abs_longitude(
        observable_context=observable_context,
        predicted_complete_counts=predict_complete_observable_histogram_with_abs_longitude(
            complete_survivor_intensity_grid=compute_shared_complete_survivor_intensity_grid(
                baseline_payload["model"],
                base_context=base_context,
            ),
            base_context=base_context,
            observable_context=observable_context,
        ),
        start_params=None,
    )["raw_parameters"]

    iteration_rows: list[DetectabilityAbsLongitudeTwoComponentIterationSummary] = []
    current_contexts = component_base_contexts
    current_payload = baseline_payload

    for iteration in range(1, n_iterations + 1):
        completeness_bin_grid = evaluate_completeness_bin_grid_with_abs_longitude(current_raw_params, observable_context)
        effective_completeness_grid = compute_effective_completeness_grid_with_abs_longitude(
            observable_context=observable_context,
            completeness_bin_grid=completeness_bin_grid,
        )
        current_contexts = apply_effective_completeness_to_component_contexts(
            component_base_contexts=component_base_contexts,
            effective_completeness_grid=effective_completeness_grid,
        )
        current_payload = fit_shared_imf_two_component_single_model(contexts=current_contexts, spec=spec)
        predicted_complete_counts = predict_complete_observable_histogram_with_abs_longitude(
            complete_survivor_intensity_grid=compute_shared_complete_survivor_intensity_grid(
                current_payload["model"],
                base_context=base_context,
            ),
            base_context=base_context,
            observable_context=observable_context,
        )
        completeness_fit = fit_logistic_completeness_model_with_abs_longitude(
            observable_context=observable_context,
            predicted_complete_counts=predicted_complete_counts,
            start_params=current_raw_params,
        )
        target_raw_params = completeness_fit["raw_parameters"]
        current_raw_params = (1.0 - relaxation) * current_raw_params + relaxation * target_raw_params
        updated_completeness_bin_grid = evaluate_completeness_bin_grid_with_abs_longitude(
            current_raw_params,
            observable_context,
        )
        predicted_observed_counts = predicted_complete_counts * updated_completeness_bin_grid
        selection_stats = aggregate_two_component_selection_stats(
            counts_by_component={label: len(context.log_mass_data) for label, context in current_contexts.items()},
            total_initial_count_by_component=current_payload["model"]["total_initial_count"],
            raw_survival_fraction_by_component=current_payload["model"]["raw_survival_fraction"],
            selection_fraction_by_component=current_payload["model"]["selection_fraction"],
        )
        iteration_rows.append(
            DetectabilityAbsLongitudeTwoComponentIterationSummary(
                iteration=iteration,
                log_likelihood=float(current_payload["summary"].log_likelihood),
                total_initial_count=float(selection_stats["total_initial_count"]),
                selection_fraction=float(selection_stats["selection_fraction"]),
                raw_survival_fraction=float(selection_stats["raw_survival_fraction"]),
                completeness_mean=float(selection_stats["mean_detectability"]),
                completeness_intercept=float(current_raw_params[0]),
                completeness_mass_slope=float(np.exp(current_raw_params[1])),
                completeness_distance_slope=float(np.exp(current_raw_params[2])),
                completeness_latitude_slope=float(np.exp(current_raw_params[3])),
                completeness_longitude_slope=float(np.exp(current_raw_params[4])),
                predicted_complete_survivor_count=float(np.sum(predicted_complete_counts)),
                predicted_observed_count=float(np.sum(predicted_observed_counts)),
            )
        )

    final_completeness_bin_grid = evaluate_completeness_bin_grid_with_abs_longitude(current_raw_params, observable_context)
    final_effective_completeness_grid = compute_effective_completeness_grid_with_abs_longitude(
        observable_context=observable_context,
        completeness_bin_grid=final_completeness_bin_grid,
    )
    final_contexts = apply_effective_completeness_to_component_contexts(
        component_base_contexts=component_base_contexts,
        effective_completeness_grid=final_effective_completeness_grid,
    )
    final_payload = fit_shared_imf_two_component_single_model(contexts=final_contexts, spec=spec)
    final_predicted_complete_counts = predict_complete_observable_histogram_with_abs_longitude(
        complete_survivor_intensity_grid=compute_shared_complete_survivor_intensity_grid(
            final_payload["model"],
            base_context=base_context,
        ),
        base_context=base_context,
        observable_context=observable_context,
    )
    final_predicted_observed_counts = final_predicted_complete_counts * final_completeness_bin_grid
    iteration_history_table = pd.DataFrame([asdict(row) for row in iteration_rows])
    completeness_grid_table = build_completeness_grid_table_with_abs_longitude(observable_context, final_completeness_bin_grid)
    observable_histogram_table = build_observable_histogram_table_with_abs_longitude(
        observable_context=observable_context,
        predicted_complete_counts=final_predicted_complete_counts,
        predicted_observed_counts=final_predicted_observed_counts,
        completeness_bin_grid=final_completeness_bin_grid,
    )
    catalog_completeness_table = build_catalog_completeness_table_with_abs_longitude(
        catalog=working,
        context=base_context.with_selection_probability_grid(
            np.clip(base_context.survival_probability_grid * final_effective_completeness_grid, 1.0e-12, 1.0)
        ),
        observable_context=observable_context,
        completeness_raw_params=current_raw_params,
    )
    selection_stats = aggregate_two_component_selection_stats(
        counts_by_component={label: len(subset) for label, subset in subsets.items()},
        total_initial_count_by_component=final_payload["model"]["total_initial_count"],
        raw_survival_fraction_by_component=final_payload["model"]["raw_survival_fraction"],
        selection_fraction_by_component=final_payload["model"]["selection_fraction"],
    )
    summary_payload = {
        "spec": {
            "imf_family": spec.imf_family,
            "in_situ_radial_model": spec.in_situ_radial_model,
            "accreted_radial_model": spec.accreted_radial_model,
        },
        "baseline_total_initial_count": float(
            baseline_payload["model"]["total_initial_count"]["in_situ"]
            + baseline_payload["model"]["total_initial_count"]["accreted"]
        ),
        "final_total_initial_count": float(selection_stats["total_initial_count"]),
        "final_selection_fraction": float(selection_stats["selection_fraction"]),
        "final_raw_survival_fraction": float(selection_stats["raw_survival_fraction"]),
        "final_mean_detectability": float(selection_stats["mean_detectability"]),
        "total_initial_count_ratio_vs_baseline": float(
            selection_stats["total_initial_count"]
            / max(
                baseline_payload["model"]["total_initial_count"]["in_situ"]
                + baseline_payload["model"]["total_initial_count"]["accreted"],
                1.0e-12,
            )
        ),
        "final_completeness_parameters": {
            "intercept": float(current_raw_params[0]),
            "mass_slope": float(np.exp(current_raw_params[1])),
            "distance_slope": float(np.exp(current_raw_params[2])),
            "latitude_slope": float(np.exp(current_raw_params[3])),
            "longitude_slope": float(np.exp(current_raw_params[4])),
        },
        "baseline_model": asdict(baseline_payload["summary"]),
        "final_model": asdict(final_payload["summary"]),
        "iteration_history": iteration_history_table.to_dict(orient="records"),
    }
    return {
        "spec": spec,
        "baseline_payload": baseline_payload,
        "final_payload": final_payload,
        "final_contexts": final_contexts,
        "iteration_history_table": iteration_history_table,
        "completeness_grid_table": completeness_grid_table,
        "observable_histogram_table": observable_histogram_table,
        "catalog_completeness_table": catalog_completeness_table,
        "final_completeness_raw_parameters": current_raw_params,
        "final_completeness_bin_grid": final_completeness_bin_grid,
        "final_effective_completeness_grid": final_effective_completeness_grid,
        "final_predicted_complete_counts": final_predicted_complete_counts,
        "final_predicted_observed_counts": final_predicted_observed_counts,
        "summary_payload": summary_payload,
    }


def fit_split_alpha_two_component_detectability_em_models_with_abs_longitude(
    catalog: pd.DataFrame,
    project_root: Path,
    radial_models: list[str] | None = None,
    fixed_effective_completeness_grid: np.ndarray | None = None,
    fixed_completeness_bin_grid: np.ndarray | None = None,
    fixed_completeness_raw_parameters: np.ndarray | None = None,
    **kwargs,
) -> dict[str, object]:
    if radial_models is None:
        radial_models = ["step5", "logpoly3"]

    env = prepare_two_component_detectability_environment_with_abs_longitude(catalog=catalog, **kwargs)
    model_specs = build_split_alpha_two_component_specs(radial_models=radial_models)
    all_results = [
        fit_split_alpha_two_component_detectability_em_single_model_with_abs_longitude(
            spec=spec,
            fixed_effective_completeness_grid=fixed_effective_completeness_grid,
            fixed_completeness_bin_grid=fixed_completeness_bin_grid,
            fixed_completeness_raw_parameters=fixed_completeness_raw_parameters,
            **env,
        )
        for spec in model_specs
    ]
    summary_table = pd.DataFrame(
        [build_shared_two_component_detectability_summary_row(result) for result in all_results]
    ).sort_values("bic", ascending=True).reset_index(drop=True)
    best_bic = float(summary_table["bic"].min())
    summary_table["delta_bic"] = summary_table["bic"] - best_bic
    best_key = (
        str(summary_table.iloc[0]["in_situ_radial_model"]),
        str(summary_table.iloc[0]["accreted_radial_model"]),
    )
    best_result = next(
        result
        for result in all_results
        if (
            result["spec"].in_situ_radial_model,
            result["spec"].accreted_radial_model,
        )
        == best_key
    )
    for result in all_results:
        result["final_payload"]["summary"].delta_bic = float(result["final_payload"]["summary"].bic - best_bic)

    best_component_summary_table = build_split_alpha_best_component_summary_table(
        best_payload=best_result["final_payload"],
        n_clusters_by_component={label: len(subset) for label, subset in env["subsets"].items()},
    )
    best_imf_grid_table = build_split_alpha_best_component_imf_grid_table(
        best_payload=best_result["final_payload"],
        contexts=best_result["final_contexts"],
    )
    best_radial_grid_table = build_split_alpha_best_component_radial_grid_table(
        best_payload=best_result["final_payload"],
        contexts=best_result["final_contexts"],
    )
    catalog_prediction_table = build_split_alpha_best_component_catalog_prediction_table(
        subsets=env["subsets"],
        contexts=best_result["final_contexts"],
        best_payload=best_result["final_payload"],
    )

    outputs_tables = project_root / "outputs" / "tables"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    summary_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_abs_longitude_split_alpha_two_component_model_summary.csv",
        index=False,
    )
    best_component_summary_table.to_csv(
        outputs_tables
        / "joint_fixed_survival_detectability_abs_longitude_split_alpha_two_component_best_component_summary.csv",
        index=False,
    )
    best_imf_grid_table.to_csv(
        outputs_tables
        / "joint_fixed_survival_detectability_abs_longitude_split_alpha_two_component_best_imf_grids.csv",
        index=False,
    )
    best_radial_grid_table.to_csv(
        outputs_tables
        / "joint_fixed_survival_detectability_abs_longitude_split_alpha_two_component_best_radial_grids.csv",
        index=False,
    )
    catalog_prediction_table.to_csv(
        outputs_tables
        / "joint_fixed_survival_detectability_abs_longitude_split_alpha_two_component_catalog_predictions.csv",
        index=False,
    )
    best_result["iteration_history_table"].to_csv(
        outputs_tables
        / "joint_fixed_survival_detectability_abs_longitude_split_alpha_two_component_best_em_iteration_history.csv",
        index=False,
    )
    best_result["completeness_grid_table"].to_csv(
        outputs_tables
        / "joint_fixed_survival_detectability_abs_longitude_split_alpha_two_component_best_completeness_grid.csv",
        index=False,
    )
    best_result["observable_histogram_table"].to_csv(
        outputs_tables
        / "joint_fixed_survival_detectability_abs_longitude_split_alpha_two_component_best_observable_histogram.csv",
        index=False,
    )
    best_result["catalog_completeness_table"].to_csv(
        outputs_tables
        / "joint_fixed_survival_detectability_abs_longitude_split_alpha_two_component_best_catalog_completeness.csv",
        index=False,
    )

    detailed_summary = {
        "selection_offset_dex": env["selection_offset_dex"],
        "survival_grid_bandwidth_log10_a_dex": env["survival_grid"]["bandwidth_log10_a_dex"],
        "n_clusters_total": int(len(env["working"])),
        "n_clusters_in_situ": int(len(env["subsets"]["in_situ"])),
        "n_clusters_accreted": int(len(env["subsets"]["accreted"])),
        "best_joint_model": asdict(best_result["final_payload"]["summary"]),
        "best_component_models": best_component_summary_table.to_dict(orient="records"),
        "all_joint_models_ranked": summary_table.to_dict(orient="records"),
        "best_model_detectability_summary": best_result["summary_payload"],
    }
    (
        outputs_tables / "joint_fixed_survival_detectability_abs_longitude_split_alpha_two_component_model_summary.json"
    ).write_text(json.dumps(detailed_summary, indent=2))

    return {
        "summary_table": summary_table,
        "best_component_summary_table": best_component_summary_table,
        "best_imf_grid_table": best_imf_grid_table,
        "best_radial_grid_table": best_radial_grid_table,
        "catalog_prediction_table": catalog_prediction_table,
        "best_result": best_result,
        "all_results": all_results,
        "contexts": best_result["final_contexts"],
        "subsets": env["subsets"],
        "base_context": env["base_context"],
        "observable_context": env["observable_context"],
        "survival_grid": env["survival_grid"],
    }


def fit_split_alpha_two_component_detectability_em_single_model_with_abs_longitude(
    working: pd.DataFrame,
    subsets: dict[str, pd.DataFrame],
    base_context: JointLikelihoodContext,
    observable_context: ObservablePredictionContextWithLongitude,
    component_base_contexts: dict[str, JointLikelihoodContext],
    spec: SplitAlphaTwoComponentSpec,
    n_iterations: int = 12,
    relaxation: float = 0.7,
    fixed_effective_completeness_grid: np.ndarray | None = None,
    fixed_completeness_bin_grid: np.ndarray | None = None,
    fixed_completeness_raw_parameters: np.ndarray | None = None,
    **_: object,
) -> dict[str, object]:
    baseline_payload = fit_split_alpha_two_component_single_model(contexts=component_base_contexts, spec=spec)
    if fixed_effective_completeness_grid is not None:
        final_completeness_bin_grid, final_completeness_raw_parameters = resolve_fixed_completeness_inputs_with_abs_longitude(
            observable_context=observable_context,
            fixed_completeness_bin_grid=fixed_completeness_bin_grid,
            fixed_completeness_raw_parameters=fixed_completeness_raw_parameters,
        )
        final_contexts = apply_effective_completeness_to_component_contexts(
            component_base_contexts=component_base_contexts,
            effective_completeness_grid=fixed_effective_completeness_grid,
        )
        final_payload = fit_split_alpha_two_component_single_model(contexts=final_contexts, spec=spec)
        result = build_fixed_two_component_detectability_result_with_abs_longitude(
            working=working,
            subsets=subsets,
            base_context=base_context,
            observable_context=observable_context,
            baseline_payload=baseline_payload,
            final_payload=final_payload,
            final_contexts=final_contexts,
            final_effective_completeness_grid=fixed_effective_completeness_grid,
            final_completeness_bin_grid=final_completeness_bin_grid,
            final_completeness_raw_parameters=final_completeness_raw_parameters,
            complete_survivor_intensity_grid=compute_split_alpha_complete_survivor_intensity_grid(
                final_payload["model"],
                base_context=base_context,
            ),
            spec_payload={
                "imf_family": "schechter",
                "difference_model": "split_alpha",
                "detectability_mode": "fixed_from_single_component",
                "in_situ_radial_model": spec.in_situ_radial_model,
                "accreted_radial_model": spec.accreted_radial_model,
            },
        )
        result["spec"] = spec
        return result

    current_raw_params = fit_logistic_completeness_model_with_abs_longitude(
        observable_context=observable_context,
        predicted_complete_counts=predict_complete_observable_histogram_with_abs_longitude(
            complete_survivor_intensity_grid=compute_split_alpha_complete_survivor_intensity_grid(
                baseline_payload["model"],
                base_context=base_context,
            ),
            base_context=base_context,
            observable_context=observable_context,
        ),
        start_params=None,
    )["raw_parameters"]

    iteration_rows: list[DetectabilityAbsLongitudeTwoComponentIterationSummary] = []
    current_contexts = component_base_contexts
    current_payload = baseline_payload

    for iteration in range(1, n_iterations + 1):
        completeness_bin_grid = evaluate_completeness_bin_grid_with_abs_longitude(current_raw_params, observable_context)
        effective_completeness_grid = compute_effective_completeness_grid_with_abs_longitude(
            observable_context=observable_context,
            completeness_bin_grid=completeness_bin_grid,
        )
        current_contexts = apply_effective_completeness_to_component_contexts(
            component_base_contexts=component_base_contexts,
            effective_completeness_grid=effective_completeness_grid,
        )
        current_payload = fit_split_alpha_two_component_single_model(contexts=current_contexts, spec=spec)
        predicted_complete_counts = predict_complete_observable_histogram_with_abs_longitude(
            complete_survivor_intensity_grid=compute_split_alpha_complete_survivor_intensity_grid(
                current_payload["model"],
                base_context=base_context,
            ),
            base_context=base_context,
            observable_context=observable_context,
        )
        completeness_fit = fit_logistic_completeness_model_with_abs_longitude(
            observable_context=observable_context,
            predicted_complete_counts=predicted_complete_counts,
            start_params=current_raw_params,
        )
        target_raw_params = completeness_fit["raw_parameters"]
        current_raw_params = (1.0 - relaxation) * current_raw_params + relaxation * target_raw_params
        updated_completeness_bin_grid = evaluate_completeness_bin_grid_with_abs_longitude(
            current_raw_params,
            observable_context,
        )
        predicted_observed_counts = predicted_complete_counts * updated_completeness_bin_grid
        selection_stats = aggregate_two_component_selection_stats(
            counts_by_component={label: len(context.log_mass_data) for label, context in current_contexts.items()},
            total_initial_count_by_component=current_payload["model"]["total_initial_count"],
            raw_survival_fraction_by_component=current_payload["model"]["raw_survival_fraction"],
            selection_fraction_by_component=current_payload["model"]["selection_fraction"],
        )
        iteration_rows.append(
            DetectabilityAbsLongitudeTwoComponentIterationSummary(
                iteration=iteration,
                log_likelihood=float(current_payload["summary"].log_likelihood),
                total_initial_count=float(selection_stats["total_initial_count"]),
                selection_fraction=float(selection_stats["selection_fraction"]),
                raw_survival_fraction=float(selection_stats["raw_survival_fraction"]),
                completeness_mean=float(selection_stats["mean_detectability"]),
                completeness_intercept=float(current_raw_params[0]),
                completeness_mass_slope=float(np.exp(current_raw_params[1])),
                completeness_distance_slope=float(np.exp(current_raw_params[2])),
                completeness_latitude_slope=float(np.exp(current_raw_params[3])),
                completeness_longitude_slope=float(np.exp(current_raw_params[4])),
                predicted_complete_survivor_count=float(np.sum(predicted_complete_counts)),
                predicted_observed_count=float(np.sum(predicted_observed_counts)),
            )
        )

    final_completeness_bin_grid = evaluate_completeness_bin_grid_with_abs_longitude(current_raw_params, observable_context)
    final_effective_completeness_grid = compute_effective_completeness_grid_with_abs_longitude(
        observable_context=observable_context,
        completeness_bin_grid=final_completeness_bin_grid,
    )
    final_contexts = apply_effective_completeness_to_component_contexts(
        component_base_contexts=component_base_contexts,
        effective_completeness_grid=final_effective_completeness_grid,
    )
    final_payload = fit_split_alpha_two_component_single_model(contexts=final_contexts, spec=spec)
    final_predicted_complete_counts = predict_complete_observable_histogram_with_abs_longitude(
        complete_survivor_intensity_grid=compute_split_alpha_complete_survivor_intensity_grid(
            final_payload["model"],
            base_context=base_context,
        ),
        base_context=base_context,
        observable_context=observable_context,
    )
    final_predicted_observed_counts = final_predicted_complete_counts * final_completeness_bin_grid
    iteration_history_table = pd.DataFrame([asdict(row) for row in iteration_rows])
    completeness_grid_table = build_completeness_grid_table_with_abs_longitude(observable_context, final_completeness_bin_grid)
    observable_histogram_table = build_observable_histogram_table_with_abs_longitude(
        observable_context=observable_context,
        predicted_complete_counts=final_predicted_complete_counts,
        predicted_observed_counts=final_predicted_observed_counts,
        completeness_bin_grid=final_completeness_bin_grid,
    )
    catalog_completeness_table = build_catalog_completeness_table_with_abs_longitude(
        catalog=working,
        context=base_context.with_selection_probability_grid(
            np.clip(base_context.survival_probability_grid * final_effective_completeness_grid, 1.0e-12, 1.0)
        ),
        observable_context=observable_context,
        completeness_raw_params=current_raw_params,
    )
    selection_stats = aggregate_two_component_selection_stats(
        counts_by_component={label: len(subset) for label, subset in subsets.items()},
        total_initial_count_by_component=final_payload["model"]["total_initial_count"],
        raw_survival_fraction_by_component=final_payload["model"]["raw_survival_fraction"],
        selection_fraction_by_component=final_payload["model"]["selection_fraction"],
    )
    summary_payload = {
        "spec": {
            "imf_family": "schechter",
            "difference_model": "split_alpha",
            "in_situ_radial_model": spec.in_situ_radial_model,
            "accreted_radial_model": spec.accreted_radial_model,
        },
        "baseline_total_initial_count": float(
            baseline_payload["model"]["total_initial_count"]["in_situ"]
            + baseline_payload["model"]["total_initial_count"]["accreted"]
        ),
        "final_total_initial_count": float(selection_stats["total_initial_count"]),
        "final_selection_fraction": float(selection_stats["selection_fraction"]),
        "final_raw_survival_fraction": float(selection_stats["raw_survival_fraction"]),
        "final_mean_detectability": float(selection_stats["mean_detectability"]),
        "total_initial_count_ratio_vs_baseline": float(
            selection_stats["total_initial_count"]
            / max(
                baseline_payload["model"]["total_initial_count"]["in_situ"]
                + baseline_payload["model"]["total_initial_count"]["accreted"],
                1.0e-12,
            )
        ),
        "final_completeness_parameters": {
            "intercept": float(current_raw_params[0]),
            "mass_slope": float(np.exp(current_raw_params[1])),
            "distance_slope": float(np.exp(current_raw_params[2])),
            "latitude_slope": float(np.exp(current_raw_params[3])),
            "longitude_slope": float(np.exp(current_raw_params[4])),
        },
        "baseline_model": asdict(baseline_payload["summary"]),
        "final_model": asdict(final_payload["summary"]),
        "iteration_history": iteration_history_table.to_dict(orient="records"),
    }
    return {
        "spec": spec,
        "baseline_payload": baseline_payload,
        "final_payload": final_payload,
        "final_contexts": final_contexts,
        "iteration_history_table": iteration_history_table,
        "completeness_grid_table": completeness_grid_table,
        "observable_histogram_table": observable_histogram_table,
        "catalog_completeness_table": catalog_completeness_table,
        "final_completeness_raw_parameters": current_raw_params,
        "final_completeness_bin_grid": final_completeness_bin_grid,
        "final_effective_completeness_grid": final_effective_completeness_grid,
        "final_predicted_complete_counts": final_predicted_complete_counts,
        "final_predicted_observed_counts": final_predicted_observed_counts,
        "summary_payload": summary_payload,
    }


def fit_separate_imf_two_component_detectability_em_models_with_abs_longitude(
    catalog: pd.DataFrame,
    project_root: Path,
    component_model_specs: list[JointModelSpec] | None = None,
    **kwargs,
) -> dict[str, object]:
    if component_model_specs is None:
        component_model_specs = [
            JointModelSpec(imf_family="lognormal", radial_model="step5"),
            JointModelSpec(imf_family="powerlaw", radial_model="step5"),
            JointModelSpec(imf_family="schechter", radial_model="step5"),
            JointModelSpec(imf_family="lognormal", radial_model="logpoly3"),
            JointModelSpec(imf_family="powerlaw", radial_model="logpoly3"),
            JointModelSpec(imf_family="schechter", radial_model="logpoly3"),
        ]

    env = prepare_two_component_detectability_environment_with_abs_longitude(catalog=catalog, **kwargs)
    all_results = []
    pair_summary_rows: list[dict[str, object]] = []
    for in_situ_spec in component_model_specs:
        for accreted_spec in component_model_specs:
            result = fit_separate_imf_two_component_detectability_em_single_model_with_abs_longitude(
                in_situ_spec=in_situ_spec,
                accreted_spec=accreted_spec,
                **env,
            )
            all_results.append(result)
            pair_summary_rows.append(build_separate_two_component_detectability_summary_row(result))

    pair_summary_table = pd.DataFrame(pair_summary_rows).sort_values("bic", ascending=True).reset_index(drop=True)
    best_bic = float(pair_summary_table["bic"].min())
    pair_summary_table["delta_bic"] = pair_summary_table["bic"] - best_bic
    best_key = (
        str(pair_summary_table.iloc[0]["in_situ_imf_family"]),
        str(pair_summary_table.iloc[0]["in_situ_radial_model"]),
        str(pair_summary_table.iloc[0]["accreted_imf_family"]),
        str(pair_summary_table.iloc[0]["accreted_radial_model"]),
    )
    best_result = next(
        result
        for result in all_results
        if (
            result["in_situ_spec"].imf_family,
            result["in_situ_spec"].radial_model,
            result["accreted_spec"].imf_family,
            result["accreted_spec"].radial_model,
        )
        == best_key
    )
    for result in all_results:
        result["pair_payload"]["summary"].delta_bic = float(result["pair_payload"]["summary"].bic - best_bic)

    component_payloads = best_result["pair_payload"]["component_payloads"]
    best_component_summary_table = pd.DataFrame(
        [
            best_pair_component_row(component_label, payload, len(env["subsets"][component_label]))
            for component_label, payload in component_payloads.items()
        ]
    ).sort_values("component_label").reset_index(drop=True)
    best_imf_grid_table = build_best_component_imf_grid_table(
        component_payloads=component_payloads,
        contexts=best_result["final_contexts"],
    )
    best_radial_grid_table = build_best_component_radial_grid_table(
        component_payloads=component_payloads,
        contexts=best_result["final_contexts"],
    )
    catalog_prediction_table = build_best_component_catalog_prediction_table(
        subsets=env["subsets"],
        contexts=best_result["final_contexts"],
        component_payloads=component_payloads,
    )

    component_summary_table = build_component_summary_table(
        component_payloads={
            "in_situ": [result["final_component_payloads"]["in_situ"] for result in all_results],
            "accreted": [result["final_component_payloads"]["accreted"] for result in all_results],
        },
        n_clusters_by_component={label: len(subset) for label, subset in env["subsets"].items()},
    )

    outputs_tables = project_root / "outputs" / "tables"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    component_summary_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_abs_longitude_two_component_component_model_summary.csv",
        index=False,
    )
    pair_summary_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_abs_longitude_two_component_model_summary.csv",
        index=False,
    )
    best_component_summary_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_abs_longitude_two_component_best_component_summary.csv",
        index=False,
    )
    best_imf_grid_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_abs_longitude_two_component_best_imf_grids.csv",
        index=False,
    )
    best_radial_grid_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_abs_longitude_two_component_best_radial_grids.csv",
        index=False,
    )
    catalog_prediction_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_abs_longitude_two_component_catalog_predictions.csv",
        index=False,
    )
    best_result["iteration_history_table"].to_csv(
        outputs_tables / "joint_fixed_survival_detectability_abs_longitude_two_component_best_em_iteration_history.csv",
        index=False,
    )
    best_result["completeness_grid_table"].to_csv(
        outputs_tables / "joint_fixed_survival_detectability_abs_longitude_two_component_best_completeness_grid.csv",
        index=False,
    )
    best_result["observable_histogram_table"].to_csv(
        outputs_tables / "joint_fixed_survival_detectability_abs_longitude_two_component_best_observable_histogram.csv",
        index=False,
    )
    best_result["catalog_completeness_table"].to_csv(
        outputs_tables / "joint_fixed_survival_detectability_abs_longitude_two_component_best_catalog_completeness.csv",
        index=False,
    )

    detailed_summary = {
        "selection_offset_dex": env["selection_offset_dex"],
        "survival_grid_bandwidth_log10_a_dex": env["survival_grid"]["bandwidth_log10_a_dex"],
        "n_clusters_total": int(len(env["working"])),
        "n_clusters_in_situ": int(len(env["subsets"]["in_situ"])),
        "n_clusters_accreted": int(len(env["subsets"]["accreted"])),
        "best_joint_model": asdict(best_result["pair_payload"]["summary"]),
        "best_component_models": best_component_summary_table.to_dict(orient="records"),
        "all_component_models_ranked": component_summary_table.to_dict(orient="records"),
        "all_joint_models_ranked": pair_summary_table.to_dict(orient="records"),
        "best_model_detectability_summary": best_result["summary_payload"],
    }
    (
        outputs_tables / "joint_fixed_survival_detectability_abs_longitude_two_component_model_summary.json"
    ).write_text(json.dumps(detailed_summary, indent=2))

    return {
        "component_summary_table": component_summary_table,
        "pair_summary_table": pair_summary_table,
        "best_component_summary_table": best_component_summary_table,
        "best_imf_grid_table": best_imf_grid_table,
        "best_radial_grid_table": best_radial_grid_table,
        "catalog_prediction_table": catalog_prediction_table,
        "best_result": best_result,
        "all_results": all_results,
        "contexts": best_result["final_contexts"],
        "subsets": env["subsets"],
        "base_context": env["base_context"],
        "observable_context": env["observable_context"],
        "survival_grid": env["survival_grid"],
    }


def fit_separate_imf_two_component_detectability_em_single_model_with_abs_longitude(
    working: pd.DataFrame,
    subsets: dict[str, pd.DataFrame],
    base_context: JointLikelihoodContext,
    observable_context: ObservablePredictionContextWithLongitude,
    component_base_contexts: dict[str, JointLikelihoodContext],
    in_situ_spec: JointModelSpec,
    accreted_spec: JointModelSpec,
    n_iterations: int = 12,
    relaxation: float = 0.7,
    **_: object,
) -> dict[str, object]:
    baseline_component_payloads = {
        "in_situ": fit_single_joint_model(context=component_base_contexts["in_situ"], spec=in_situ_spec),
        "accreted": fit_single_joint_model(context=component_base_contexts["accreted"], spec=accreted_spec),
    }
    baseline_pair_payload = build_two_component_pair_payloads(
        in_situ_payloads=[baseline_component_payloads["in_situ"]],
        accreted_payloads=[baseline_component_payloads["accreted"]],
        n_clusters_in_situ=len(subsets["in_situ"]),
        n_clusters_accreted=len(subsets["accreted"]),
    )[0]
    current_raw_params = fit_logistic_completeness_model_with_abs_longitude(
        observable_context=observable_context,
        predicted_complete_counts=predict_complete_observable_histogram_with_abs_longitude(
            complete_survivor_intensity_grid=compute_separate_complete_survivor_intensity_grid(
                component_payloads=baseline_component_payloads,
                base_context=base_context,
            ),
            base_context=base_context,
            observable_context=observable_context,
        ),
        start_params=None,
    )["raw_parameters"]

    iteration_rows: list[DetectabilityAbsLongitudeTwoComponentIterationSummary] = []
    current_contexts = component_base_contexts
    current_component_payloads = baseline_component_payloads
    current_pair_payload = baseline_pair_payload

    for iteration in range(1, n_iterations + 1):
        completeness_bin_grid = evaluate_completeness_bin_grid_with_abs_longitude(current_raw_params, observable_context)
        effective_completeness_grid = compute_effective_completeness_grid_with_abs_longitude(
            observable_context=observable_context,
            completeness_bin_grid=completeness_bin_grid,
        )
        current_contexts = apply_effective_completeness_to_component_contexts(
            component_base_contexts=component_base_contexts,
            effective_completeness_grid=effective_completeness_grid,
        )
        current_component_payloads = {
            "in_situ": fit_single_joint_model(context=current_contexts["in_situ"], spec=in_situ_spec),
            "accreted": fit_single_joint_model(context=current_contexts["accreted"], spec=accreted_spec),
        }
        current_pair_payload = build_two_component_pair_payloads(
            in_situ_payloads=[current_component_payloads["in_situ"]],
            accreted_payloads=[current_component_payloads["accreted"]],
            n_clusters_in_situ=len(subsets["in_situ"]),
            n_clusters_accreted=len(subsets["accreted"]),
        )[0]
        predicted_complete_counts = predict_complete_observable_histogram_with_abs_longitude(
            complete_survivor_intensity_grid=compute_separate_complete_survivor_intensity_grid(
                component_payloads=current_component_payloads,
                base_context=base_context,
            ),
            base_context=base_context,
            observable_context=observable_context,
        )
        completeness_fit = fit_logistic_completeness_model_with_abs_longitude(
            observable_context=observable_context,
            predicted_complete_counts=predicted_complete_counts,
            start_params=current_raw_params,
        )
        target_raw_params = completeness_fit["raw_parameters"]
        current_raw_params = (1.0 - relaxation) * current_raw_params + relaxation * target_raw_params
        updated_completeness_bin_grid = evaluate_completeness_bin_grid_with_abs_longitude(
            current_raw_params,
            observable_context,
        )
        predicted_observed_counts = predicted_complete_counts * updated_completeness_bin_grid
        selection_stats = aggregate_two_component_selection_stats(
            counts_by_component={"in_situ": len(subsets["in_situ"]), "accreted": len(subsets["accreted"])},
            total_initial_count_by_component={
                label: payload["model"]["total_initial_count"] for label, payload in current_component_payloads.items()
            },
            raw_survival_fraction_by_component={
                label: payload["model"]["raw_survival_fraction"] for label, payload in current_component_payloads.items()
            },
            selection_fraction_by_component={
                label: payload["model"]["selection_fraction"] for label, payload in current_component_payloads.items()
            },
        )
        iteration_rows.append(
            DetectabilityAbsLongitudeTwoComponentIterationSummary(
                iteration=iteration,
                log_likelihood=float(current_pair_payload["summary"].log_likelihood),
                total_initial_count=float(selection_stats["total_initial_count"]),
                selection_fraction=float(selection_stats["selection_fraction"]),
                raw_survival_fraction=float(selection_stats["raw_survival_fraction"]),
                completeness_mean=float(selection_stats["mean_detectability"]),
                completeness_intercept=float(current_raw_params[0]),
                completeness_mass_slope=float(np.exp(current_raw_params[1])),
                completeness_distance_slope=float(np.exp(current_raw_params[2])),
                completeness_latitude_slope=float(np.exp(current_raw_params[3])),
                completeness_longitude_slope=float(np.exp(current_raw_params[4])),
                predicted_complete_survivor_count=float(np.sum(predicted_complete_counts)),
                predicted_observed_count=float(np.sum(predicted_observed_counts)),
            )
        )

    final_completeness_bin_grid = evaluate_completeness_bin_grid_with_abs_longitude(current_raw_params, observable_context)
    final_effective_completeness_grid = compute_effective_completeness_grid_with_abs_longitude(
        observable_context=observable_context,
        completeness_bin_grid=final_completeness_bin_grid,
    )
    final_contexts = apply_effective_completeness_to_component_contexts(
        component_base_contexts=component_base_contexts,
        effective_completeness_grid=final_effective_completeness_grid,
    )
    final_component_payloads = {
        "in_situ": fit_single_joint_model(context=final_contexts["in_situ"], spec=in_situ_spec),
        "accreted": fit_single_joint_model(context=final_contexts["accreted"], spec=accreted_spec),
    }
    final_pair_payload = build_two_component_pair_payloads(
        in_situ_payloads=[final_component_payloads["in_situ"]],
        accreted_payloads=[final_component_payloads["accreted"]],
        n_clusters_in_situ=len(subsets["in_situ"]),
        n_clusters_accreted=len(subsets["accreted"]),
    )[0]
    final_predicted_complete_counts = predict_complete_observable_histogram_with_abs_longitude(
        complete_survivor_intensity_grid=compute_separate_complete_survivor_intensity_grid(
            component_payloads=final_component_payloads,
            base_context=base_context,
        ),
        base_context=base_context,
        observable_context=observable_context,
    )
    final_predicted_observed_counts = final_predicted_complete_counts * final_completeness_bin_grid
    iteration_history_table = pd.DataFrame([asdict(row) for row in iteration_rows])
    completeness_grid_table = build_completeness_grid_table_with_abs_longitude(observable_context, final_completeness_bin_grid)
    observable_histogram_table = build_observable_histogram_table_with_abs_longitude(
        observable_context=observable_context,
        predicted_complete_counts=final_predicted_complete_counts,
        predicted_observed_counts=final_predicted_observed_counts,
        completeness_bin_grid=final_completeness_bin_grid,
    )
    catalog_completeness_table = build_catalog_completeness_table_with_abs_longitude(
        catalog=working,
        context=base_context.with_selection_probability_grid(
            np.clip(base_context.survival_probability_grid * final_effective_completeness_grid, 1.0e-12, 1.0)
        ),
        observable_context=observable_context,
        completeness_raw_params=current_raw_params,
    )
    selection_stats = aggregate_two_component_selection_stats(
        counts_by_component={"in_situ": len(subsets["in_situ"]), "accreted": len(subsets["accreted"])},
        total_initial_count_by_component={
            label: payload["model"]["total_initial_count"] for label, payload in final_component_payloads.items()
        },
        raw_survival_fraction_by_component={
            label: payload["model"]["raw_survival_fraction"] for label, payload in final_component_payloads.items()
        },
        selection_fraction_by_component={
            label: payload["model"]["selection_fraction"] for label, payload in final_component_payloads.items()
        },
    )
    summary_payload = {
        "spec": {
            "in_situ_imf_family": in_situ_spec.imf_family,
            "in_situ_radial_model": in_situ_spec.radial_model,
            "accreted_imf_family": accreted_spec.imf_family,
            "accreted_radial_model": accreted_spec.radial_model,
        },
        "baseline_total_initial_count": float(
            baseline_component_payloads["in_situ"]["model"]["total_initial_count"]
            + baseline_component_payloads["accreted"]["model"]["total_initial_count"]
        ),
        "final_total_initial_count": float(selection_stats["total_initial_count"]),
        "final_selection_fraction": float(selection_stats["selection_fraction"]),
        "final_raw_survival_fraction": float(selection_stats["raw_survival_fraction"]),
        "final_mean_detectability": float(selection_stats["mean_detectability"]),
        "total_initial_count_ratio_vs_baseline": float(
            selection_stats["total_initial_count"]
            / max(
                baseline_component_payloads["in_situ"]["model"]["total_initial_count"]
                + baseline_component_payloads["accreted"]["model"]["total_initial_count"],
                1.0e-12,
            )
        ),
        "final_completeness_parameters": {
            "intercept": float(current_raw_params[0]),
            "mass_slope": float(np.exp(current_raw_params[1])),
            "distance_slope": float(np.exp(current_raw_params[2])),
            "latitude_slope": float(np.exp(current_raw_params[3])),
            "longitude_slope": float(np.exp(current_raw_params[4])),
        },
        "baseline_model": asdict(baseline_pair_payload["summary"]),
        "final_model": asdict(final_pair_payload["summary"]),
        "iteration_history": iteration_history_table.to_dict(orient="records"),
    }
    return {
        "in_situ_spec": in_situ_spec,
        "accreted_spec": accreted_spec,
        "baseline_component_payloads": baseline_component_payloads,
        "baseline_pair_payload": baseline_pair_payload,
        "final_component_payloads": final_component_payloads,
        "pair_payload": final_pair_payload,
        "final_contexts": final_contexts,
        "iteration_history_table": iteration_history_table,
        "completeness_grid_table": completeness_grid_table,
        "observable_histogram_table": observable_histogram_table,
        "catalog_completeness_table": catalog_completeness_table,
        "final_completeness_raw_parameters": current_raw_params,
        "final_completeness_bin_grid": final_completeness_bin_grid,
        "final_effective_completeness_grid": final_effective_completeness_grid,
        "final_predicted_complete_counts": final_predicted_complete_counts,
        "final_predicted_observed_counts": final_predicted_observed_counts,
        "summary_payload": summary_payload,
    }


def build_observable_prediction_context_with_abs_longitude(
    catalog: pd.DataFrame,
    base_context: JointLikelihoodContext,
    n_present_mass_bins: int,
    n_distance_bins: int,
    n_latitude_bins: int,
    n_longitude_bins: int,
    n_geometry_samples: int,
    sun_galactocentric_radius_kpc: float,
) -> ObservablePredictionContextWithLongitude:
    present_mass_proxy = fit_present_mass_proxy_model(catalog)
    log_present_mass_mean_grid = predict_log_present_mass_grid(
        log_mass_grid=base_context.log_mass_grid,
        log_a_grid=base_context.log_a_grid,
        proxy_model=present_mass_proxy,
    )
    observed_log_present_mass = np.log10(catalog["present_mass_msun"].to_numpy())
    observed_distance_kpc = catalog["r_sun_kpc"].to_numpy()
    observed_abs_latitude_deg = np.abs(catalog["galactic_b_deg"].to_numpy())
    observed_abs_longitude_deg = absolute_wrapped_longitude_degrees(catalog["galactic_l_deg"].to_numpy())

    log_present_mass_min = min(
        float(observed_log_present_mass.min()),
        float(np.nanmin(log_present_mass_mean_grid) - 0.5 * present_mass_proxy.residual_sigma_dex),
    )
    log_present_mass_max = max(
        float(observed_log_present_mass.max()),
        float(np.nanmax(log_present_mass_mean_grid) + 0.5 * present_mass_proxy.residual_sigma_dex),
    )
    log_present_mass_edges = np.linspace(
        np.floor(log_present_mass_min * 4.0) / 4.0,
        np.ceil(log_present_mass_max * 4.0) / 4.0,
        n_present_mass_bins + 1,
    )

    max_distance_kpc = max(
        float(observed_distance_kpc.max()),
        float(np.power(10.0, base_context.log_a_grid.max()) + sun_galactocentric_radius_kpc),
    )
    min_distance_kpc = max(0.5, float(observed_distance_kpc.min()) * 0.8)
    distance_edges_kpc = np.geomspace(min_distance_kpc, max_distance_kpc, n_distance_bins + 1)
    abs_latitude_edges_deg = np.linspace(0.0, 90.0, n_latitude_bins + 1)
    abs_longitude_edges_deg = np.linspace(0.0, 180.0, n_longitude_bins + 1)

    observed_counts, _ = np.histogramdd(
        np.column_stack(
            [
                observed_log_present_mass,
                observed_distance_kpc,
                observed_abs_latitude_deg,
                observed_abs_longitude_deg,
            ]
        ),
        bins=[log_present_mass_edges, distance_edges_kpc, abs_latitude_edges_deg, abs_longitude_edges_deg],
    )

    mass_bin_probabilities_grid = build_mass_bin_probabilities_grid(
        log_present_mass_mean_grid=log_present_mass_mean_grid,
        log_present_mass_edges=log_present_mass_edges,
        sigma_dex=present_mass_proxy.residual_sigma_dex,
    )
    sky_bin_probabilities_by_a = build_spherical_sky_bin_probabilities_with_abs_longitude(
        a_grid_kpc=np.power(10.0, base_context.log_a_grid),
        distance_edges_kpc=distance_edges_kpc,
        abs_latitude_edges_deg=abs_latitude_edges_deg,
        abs_longitude_edges_deg=abs_longitude_edges_deg,
        n_geometry_samples=n_geometry_samples,
        sun_galactocentric_radius_kpc=sun_galactocentric_radius_kpc,
    )

    log_present_mass_centers = 0.5 * (log_present_mass_edges[:-1] + log_present_mass_edges[1:])
    distance_centers_kpc = np.sqrt(distance_edges_kpc[:-1] * distance_edges_kpc[1:])
    log_distance_centers = np.log10(distance_centers_kpc)
    abs_latitude_centers_deg = 0.5 * (abs_latitude_edges_deg[:-1] + abs_latitude_edges_deg[1:])
    abs_longitude_centers_deg = 0.5 * (abs_longitude_edges_deg[:-1] + abs_longitude_edges_deg[1:])

    return ObservablePredictionContextWithLongitude(
        present_mass_proxy=present_mass_proxy,
        log_present_mass_edges=log_present_mass_edges,
        distance_edges_kpc=distance_edges_kpc,
        abs_latitude_edges_deg=abs_latitude_edges_deg,
        abs_longitude_edges_deg=abs_longitude_edges_deg,
        log_present_mass_centers=log_present_mass_centers,
        log_distance_centers=log_distance_centers,
        abs_latitude_centers_deg=abs_latitude_centers_deg,
        abs_longitude_centers_deg=abs_longitude_centers_deg,
        observed_counts=observed_counts.astype(float),
        mass_bin_probabilities_grid=mass_bin_probabilities_grid,
        sky_bin_probabilities_by_a=sky_bin_probabilities_by_a,
        log_present_mass_mean_grid=log_present_mass_mean_grid,
        log_present_mass_feature_mean=float(np.mean(observed_log_present_mass)),
        log_present_mass_feature_std=float(max(np.std(observed_log_present_mass), 1.0e-6)),
        log_distance_feature_mean=float(np.mean(np.log10(observed_distance_kpc))),
        log_distance_feature_std=float(max(np.std(np.log10(observed_distance_kpc)), 1.0e-6)),
        abs_latitude_feature_mean=float(np.mean(observed_abs_latitude_deg)),
        abs_latitude_feature_std=float(max(np.std(observed_abs_latitude_deg), 1.0e-6)),
        abs_longitude_feature_mean=float(np.mean(observed_abs_longitude_deg)),
        abs_longitude_feature_std=float(max(np.std(observed_abs_longitude_deg), 1.0e-6)),
        sun_galactocentric_radius_kpc=sun_galactocentric_radius_kpc,
        n_geometry_samples=n_geometry_samples,
    )


def absolute_wrapped_longitude_degrees(longitude_deg: np.ndarray) -> np.ndarray:
    return np.abs(((np.asarray(longitude_deg, dtype=float) + 180.0) % 360.0) - 180.0)


def build_spherical_sky_bin_probabilities_with_abs_longitude(
    a_grid_kpc: np.ndarray,
    distance_edges_kpc: np.ndarray,
    abs_latitude_edges_deg: np.ndarray,
    abs_longitude_edges_deg: np.ndarray,
    n_geometry_samples: int,
    sun_galactocentric_radius_kpc: float,
    random_seed: int = 12345,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    cos_theta = rng.uniform(-1.0, 1.0, size=n_geometry_samples)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n_geometry_samples)
    sin_theta = np.sqrt(np.clip(1.0 - np.square(cos_theta), 0.0, 1.0))
    x_unit = sin_theta * np.cos(phi)
    y_unit = sin_theta * np.sin(phi)
    z_unit = cos_theta

    probabilities = np.zeros(
        (
            len(a_grid_kpc),
            len(distance_edges_kpc) - 1,
            len(abs_latitude_edges_deg) - 1,
            len(abs_longitude_edges_deg) - 1,
        ),
        dtype=float,
    )
    for index, a_value in enumerate(a_grid_kpc):
        x = a_value * x_unit
        y = a_value * y_unit
        z = a_value * z_unit
        x_helio_toward_gc = sun_galactocentric_radius_kpc - x
        distance = np.sqrt(np.square(x_helio_toward_gc) + np.square(y) + np.square(z))
        abs_latitude = np.degrees(np.arcsin(np.clip(np.abs(z) / np.clip(distance, 1.0e-12, None), 0.0, 1.0)))
        abs_longitude = np.degrees(np.abs(np.arctan2(y, x_helio_toward_gc)))
        counts, _ = np.histogramdd(
            np.column_stack([distance, abs_latitude, abs_longitude]),
            bins=[distance_edges_kpc, abs_latitude_edges_deg, abs_longitude_edges_deg],
        )
        probabilities[index] = counts / max(np.sum(counts), 1.0)
    return probabilities


def predict_complete_observable_histogram_with_abs_longitude(
    complete_survivor_intensity_grid: np.ndarray,
    base_context: JointLikelihoodContext,
    observable_context: ObservablePredictionContextWithLongitude,
) -> np.ndarray:
    log_mass_edges = centers_to_edges_local(base_context.log_mass_grid)
    log_a_edges = centers_to_edges_local(base_context.log_a_grid)
    cell_counts = complete_survivor_intensity_grid * np.diff(log_mass_edges)[:, None] * np.diff(log_a_edges)[None, :]
    mass_counts_by_a = np.einsum(
        "ma,mak->ak",
        cell_counts,
        observable_context.mass_bin_probabilities_grid,
    )
    return np.einsum(
        "ak,adbl->kdbl",
        mass_counts_by_a,
        observable_context.sky_bin_probabilities_by_a,
    )


def fit_logistic_completeness_model_with_abs_longitude(
    observable_context: ObservablePredictionContextWithLongitude,
    predicted_complete_counts: np.ndarray,
    start_params: np.ndarray | None,
) -> dict[str, object]:
    total_predicted = float(np.sum(predicted_complete_counts))
    total_observed = float(np.sum(observable_context.observed_counts))
    ratio = np.clip(total_observed / max(total_predicted, 1.0e-12), 1.0e-3, 0.999)
    default_start = np.array(
        [
            float(np.log(ratio / (1.0 - ratio))),
            np.log(0.25),
            np.log(0.25),
            np.log(0.25),
            np.log(0.25),
        ]
    )
    starts = [default_start]
    if start_params is not None:
        starts.insert(0, np.asarray(start_params, dtype=float))
    starts.append(np.array([default_start[0], np.log(0.6), np.log(0.6), np.log(0.6), np.log(0.6)]))
    bounds = [(-8.0, 8.0), (-8.0, 4.0), (-8.0, 4.0), (-8.0, 4.0), (-8.0, 4.0)]

    best_result = None
    best_value = np.inf
    for start in starts:
        result = optimize.minimize(
            lambda params: negative_completeness_log_likelihood_with_abs_longitude(
                params=params,
                observable_context=observable_context,
                predicted_complete_counts=predicted_complete_counts,
            ),
            x0=np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if result.fun < best_value:
            best_value = float(result.fun)
            best_result = result

    if best_result is None:
        raise RuntimeError("Longitude-aware completeness optimization failed to start.")
    completeness_bin_grid = evaluate_completeness_bin_grid_with_abs_longitude(best_result.x, observable_context)
    return {
        "raw_parameters": np.asarray(best_result.x, dtype=float),
        "completeness_bin_grid": completeness_bin_grid,
        "negative_log_likelihood": float(best_result.fun),
        "success": bool(best_result.success),
        "message": str(best_result.message),
    }


def negative_completeness_log_likelihood_with_abs_longitude(
    params: np.ndarray,
    observable_context: ObservablePredictionContextWithLongitude,
    predicted_complete_counts: np.ndarray,
) -> float:
    completeness_bin_grid = evaluate_completeness_bin_grid_with_abs_longitude(params, observable_context)
    mu = np.clip(predicted_complete_counts * completeness_bin_grid, 1.0e-12, None)
    observed = observable_context.observed_counts
    return float(-(np.sum(observed * np.log(mu) - mu)))


def evaluate_completeness_bin_grid_with_abs_longitude(
    raw_params: np.ndarray,
    observable_context: ObservablePredictionContextWithLongitude,
) -> np.ndarray:
    intercept = float(raw_params[0])
    mass_slope = float(np.exp(raw_params[1]))
    distance_slope = float(np.exp(raw_params[2]))
    latitude_slope = float(np.exp(raw_params[3]))
    longitude_slope = float(np.exp(raw_params[4]))

    z_mass = (
        observable_context.log_present_mass_centers[:, None, None, None]
        - observable_context.log_present_mass_feature_mean
    ) / observable_context.log_present_mass_feature_std
    z_distance = (
        observable_context.log_distance_centers[None, :, None, None]
        - observable_context.log_distance_feature_mean
    ) / observable_context.log_distance_feature_std
    z_latitude = (
        observable_context.abs_latitude_centers_deg[None, None, :, None]
        - observable_context.abs_latitude_feature_mean
    ) / observable_context.abs_latitude_feature_std
    z_longitude = (
        observable_context.abs_longitude_centers_deg[None, None, None, :]
        - observable_context.abs_longitude_feature_mean
    ) / observable_context.abs_longitude_feature_std
    logits = (
        intercept
        + mass_slope * z_mass
        - distance_slope * z_distance
        + latitude_slope * z_latitude
        + longitude_slope * z_longitude
    )
    return np.clip(special.expit(logits), 1.0e-6, 1.0)


def compute_effective_completeness_grid_with_abs_longitude(
    observable_context: ObservablePredictionContextWithLongitude,
    completeness_bin_grid: np.ndarray,
) -> np.ndarray:
    sky_averaged_completeness = np.einsum(
        "adbl,kdbl->ak",
        observable_context.sky_bin_probabilities_by_a,
        completeness_bin_grid,
    )
    return np.clip(
        np.einsum(
            "mak,ak->ma",
            observable_context.mass_bin_probabilities_grid,
            sky_averaged_completeness,
        ),
        1.0e-4,
        1.0,
    )


def build_completeness_grid_table_with_abs_longitude(
    observable_context: ObservablePredictionContextWithLongitude,
    completeness_bin_grid: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for i_mass, log_mass_center in enumerate(observable_context.log_present_mass_centers):
        for i_distance, log_distance_center in enumerate(observable_context.log_distance_centers):
            for i_latitude, latitude_center in enumerate(observable_context.abs_latitude_centers_deg):
                for i_longitude, longitude_center in enumerate(observable_context.abs_longitude_centers_deg):
                    rows.append(
                        {
                            "present_mass_bin_index": i_mass,
                            "distance_bin_index": i_distance,
                            "latitude_bin_index": i_latitude,
                            "longitude_bin_index": i_longitude,
                            "log10_present_mass_center_msun": float(log_mass_center),
                            "present_mass_center_msun": float(np.power(10.0, log_mass_center)),
                            "distance_center_kpc": float(np.power(10.0, log_distance_center)),
                            "log10_distance_center_kpc": float(log_distance_center),
                            "abs_latitude_center_deg": float(latitude_center),
                            "abs_longitude_center_deg": float(longitude_center),
                            "completeness": float(completeness_bin_grid[i_mass, i_distance, i_latitude, i_longitude]),
                        }
                    )
    return pd.DataFrame(rows)


def build_observable_histogram_table_with_abs_longitude(
    observable_context: ObservablePredictionContextWithLongitude,
    predicted_complete_counts: np.ndarray,
    predicted_observed_counts: np.ndarray,
    completeness_bin_grid: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for i_mass, log_mass_center in enumerate(observable_context.log_present_mass_centers):
        for i_distance, log_distance_center in enumerate(observable_context.log_distance_centers):
            for i_latitude, latitude_center in enumerate(observable_context.abs_latitude_centers_deg):
                for i_longitude, longitude_center in enumerate(observable_context.abs_longitude_centers_deg):
                    rows.append(
                        {
                            "present_mass_bin_index": i_mass,
                            "distance_bin_index": i_distance,
                            "latitude_bin_index": i_latitude,
                            "longitude_bin_index": i_longitude,
                            "log10_present_mass_center_msun": float(log_mass_center),
                            "distance_center_kpc": float(np.power(10.0, log_distance_center)),
                            "abs_latitude_center_deg": float(latitude_center),
                            "abs_longitude_center_deg": float(longitude_center),
                            "observed_count": float(
                                observable_context.observed_counts[i_mass, i_distance, i_latitude, i_longitude]
                            ),
                            "predicted_complete_count": float(
                                predicted_complete_counts[i_mass, i_distance, i_latitude, i_longitude]
                            ),
                            "predicted_observed_count": float(
                                predicted_observed_counts[i_mass, i_distance, i_latitude, i_longitude]
                            ),
                            "completeness": float(
                                completeness_bin_grid[i_mass, i_distance, i_latitude, i_longitude]
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def build_catalog_completeness_table_with_abs_longitude(
    catalog: pd.DataFrame,
    context: JointLikelihoodContext,
    observable_context: ObservablePredictionContextWithLongitude,
    completeness_raw_params: np.ndarray,
) -> pd.DataFrame:
    completeness_bin_grid = evaluate_completeness_bin_grid_with_abs_longitude(
        completeness_raw_params,
        observable_context,
    )
    effective_completeness_grid = compute_effective_completeness_grid_with_abs_longitude(
        observable_context,
        completeness_bin_grid,
    )
    effective_interpolator = context.with_selection_probability_grid(
        np.clip(context.survival_probability_grid * effective_completeness_grid, 1.0e-12, 1.0)
    ).selection_interpolator

    log_present_mass = np.log10(catalog["present_mass_msun"].to_numpy())
    log_distance = np.log10(catalog["r_sun_kpc"].to_numpy())
    abs_latitude = np.abs(catalog["galactic_b_deg"].to_numpy())
    abs_longitude = absolute_wrapped_longitude_degrees(catalog["galactic_l_deg"].to_numpy())
    actual_completeness = evaluate_completeness_at_values_with_abs_longitude(
        raw_params=completeness_raw_params,
        observable_context=observable_context,
        log_present_mass=log_present_mass,
        log_distance=log_distance,
        abs_latitude_deg=abs_latitude,
        abs_longitude_deg=abs_longitude,
    )
    effective_selection = np.clip(
        effective_interpolator(
            np.column_stack(
                [
                    catalog["log_initial_mass_msun"].to_numpy(),
                    np.log10(catalog["semi_major_axis_kpc"].to_numpy()),
                ]
            )
        ),
        1.0e-12,
        1.0,
    )
    raw_survival = np.clip(
        context.survival_interpolator(
            np.column_stack(
                [
                    catalog["log_initial_mass_msun"].to_numpy(),
                    np.log10(catalog["semi_major_axis_kpc"].to_numpy()),
                ]
            )
        ),
        1.0e-12,
        1.0,
    )
    rows = []
    for index, row in catalog.reset_index(drop=True).iterrows():
        rows.append(
            {
                "cluster_name": row.get("cluster_name", row.get("cluster_label", index)),
                "log_initial_mass_msun": float(row["log_initial_mass_msun"]),
                "log10_present_mass_msun": float(log_present_mass[index]),
                "semi_major_axis_kpc": float(row["semi_major_axis_kpc"]),
                "r_sun_kpc": float(row["r_sun_kpc"]),
                "abs_galactic_b_deg": float(abs_latitude[index]),
                "abs_galactic_l_deg": float(abs_longitude[index]),
                "raw_survival_probability": float(raw_survival[index]),
                "detectability_probability_at_observed_values": float(actual_completeness[index]),
                "effective_selection_probability_intrinsic": float(effective_selection[index]),
                "effective_detectability_intrinsic": float(effective_selection[index] / raw_survival[index]),
            }
        )
    return pd.DataFrame(rows)


def evaluate_completeness_at_values_with_abs_longitude(
    raw_params: np.ndarray,
    observable_context: ObservablePredictionContextWithLongitude,
    log_present_mass: np.ndarray,
    log_distance: np.ndarray,
    abs_latitude_deg: np.ndarray,
    abs_longitude_deg: np.ndarray,
) -> np.ndarray:
    intercept = float(raw_params[0])
    mass_slope = float(np.exp(raw_params[1]))
    distance_slope = float(np.exp(raw_params[2]))
    latitude_slope = float(np.exp(raw_params[3]))
    longitude_slope = float(np.exp(raw_params[4]))
    z_mass = (log_present_mass - observable_context.log_present_mass_feature_mean) / (
        observable_context.log_present_mass_feature_std
    )
    z_distance = (log_distance - observable_context.log_distance_feature_mean) / (
        observable_context.log_distance_feature_std
    )
    z_latitude = (abs_latitude_deg - observable_context.abs_latitude_feature_mean) / (
        observable_context.abs_latitude_feature_std
    )
    z_longitude = (abs_longitude_deg - observable_context.abs_longitude_feature_mean) / (
        observable_context.abs_longitude_feature_std
    )
    logits = (
        intercept
        + mass_slope * z_mass
        - distance_slope * z_distance
        + latitude_slope * z_latitude
        + longitude_slope * z_longitude
    )
    return np.clip(special.expit(logits), 1.0e-6, 1.0)
