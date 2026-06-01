from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .joint_model import (
    JointLikelihoodContext,
    JointModelSpec,
    build_fixed_survival_grid,
    calibrate_fixed_selection_offset_dex,
    centers_to_edges_local,
    compute_observed_intensity_grid,
    fit_single_joint_model,
    rebin_expected_counts_2d,
)
from .two_component_model import (
    SharedImfTwoComponentSpec,
    build_best_component_catalog_prediction_table,
    build_best_component_imf_grid_table,
    build_best_component_radial_grid_table,
    build_component_summary_table,
    build_shared_best_component_catalog_prediction_table,
    build_shared_best_component_imf_grid_table,
    build_shared_best_component_radial_grid_table,
    build_shared_best_component_summary_table,
    build_shared_imf_two_component_specs,
    build_two_component_pair_payloads,
    best_pair_component_row,
    fit_shared_imf_two_component_single_model,
    prepare_two_component_contexts,
)

TINY = 1.0e-300


@dataclass
class PresentMassProxyModel:
    coefficients: np.ndarray
    log_mass_mean: float
    log_a_mean: float
    residual_sigma_dex: float
    log_mass_ratio_min: float
    log_present_mass_min: float
    log_present_mass_max: float
    log_mass_std: float = 1.0
    log_a_std: float = 1.0
    model_kind: str = "polynomial_log_mass_ratio"


@dataclass
class ObservablePredictionContext:
    present_mass_proxy: PresentMassProxyModel
    log_present_mass_edges: np.ndarray
    distance_edges_kpc: np.ndarray
    abs_latitude_edges_deg: np.ndarray
    log_present_mass_centers: np.ndarray
    log_distance_centers: np.ndarray
    abs_latitude_centers_deg: np.ndarray
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
    sun_galactocentric_radius_kpc: float
    n_geometry_samples: int


@dataclass
class DetectabilityIterationSummary:
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
    predicted_complete_survivor_count: float
    predicted_observed_count: float


def fit_detectability_corrected_single_component_models(
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
        result = fit_single_component_detectability_em(
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
    best_index = int(summary_table.index[0])
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
    return {
        "all_results": all_results,
        "summary_table": summary_table,
        "best_result": best_result,
    }


def fit_single_component_detectability_em(
    catalog: pd.DataFrame,
    project_root: Path,
    spec: JointModelSpec | None = None,
    n_iterations: int = 6,
    relaxation: float = 0.7,
    n_present_mass_bins: int = 6,
    n_distance_bins: int = 6,
    n_latitude_bins: int = 6,
    n_geometry_samples: int = 5000,
    sun_galactocentric_radius_kpc: float = 8.2,
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
    }
    missing = required_columns.difference(working.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Catalog is missing required columns for detectability fitting: {missing_list}")

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
    observable_context = build_observable_prediction_context(
        catalog=working,
        base_context=base_context,
        n_present_mass_bins=n_present_mass_bins,
        n_distance_bins=n_distance_bins,
        n_latitude_bins=n_latitude_bins,
        n_geometry_samples=n_geometry_samples,
        sun_galactocentric_radius_kpc=sun_galactocentric_radius_kpc,
    )

    baseline_context = base_context.with_selection_probability_grid(base_context.survival_probability_grid)
    baseline_payload = fit_single_joint_model(context=baseline_context, spec=spec)
    current_raw_params = fit_logistic_completeness_model(
        observable_context=observable_context,
        predicted_complete_counts=predict_complete_observable_histogram(
            complete_survivor_intensity_grid=compute_complete_survivor_intensity_grid(
                baseline_payload["model"],
                base_context=base_context,
            ),
            base_context=base_context,
            observable_context=observable_context,
        ),
        start_params=None,
    )["raw_parameters"]

    iteration_rows: list[DetectabilityIterationSummary] = []
    current_context = baseline_context
    current_payload = baseline_payload
    current_effective_completeness_grid = np.ones_like(base_context.survival_probability_grid)

    for iteration in range(1, n_iterations + 1):
        completeness_bin_grid = evaluate_completeness_bin_grid(current_raw_params, observable_context)
        current_effective_completeness_grid = compute_effective_completeness_grid(
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
        current_payload = fit_single_joint_model(context=current_context, spec=spec)
        complete_survivor_intensity_grid = compute_complete_survivor_intensity_grid(
            current_payload["model"],
            base_context=base_context,
        )
        predicted_complete_counts = predict_complete_observable_histogram(
            complete_survivor_intensity_grid=complete_survivor_intensity_grid,
            base_context=base_context,
            observable_context=observable_context,
        )
        completeness_fit = fit_logistic_completeness_model(
            observable_context=observable_context,
            predicted_complete_counts=predicted_complete_counts,
            start_params=current_raw_params,
        )
        target_raw_params = completeness_fit["raw_parameters"]
        current_raw_params = (1.0 - relaxation) * current_raw_params + relaxation * target_raw_params
        updated_completeness_bin_grid = evaluate_completeness_bin_grid(current_raw_params, observable_context)
        predicted_observed_counts = predicted_complete_counts * updated_completeness_bin_grid
        iteration_rows.append(
            DetectabilityIterationSummary(
                iteration=iteration,
                log_likelihood=float(current_payload["summary"].log_likelihood),
                total_initial_count=float(current_payload["model"]["total_initial_count"]),
                selection_fraction=float(current_payload["model"]["selection_fraction"]),
                raw_survival_fraction=float(current_payload["model"]["raw_survival_fraction"]),
                completeness_mean=float(
                    np.sum(predicted_complete_counts * updated_completeness_bin_grid)
                    / max(np.sum(predicted_complete_counts), 1.0e-12)
                ),
                completeness_intercept=float(current_raw_params[0]),
                completeness_mass_slope=float(np.exp(current_raw_params[1])),
                completeness_distance_slope=float(np.exp(current_raw_params[2])),
                completeness_latitude_slope=float(np.exp(current_raw_params[3])),
                predicted_complete_survivor_count=float(np.sum(predicted_complete_counts)),
                predicted_observed_count=float(np.sum(predicted_observed_counts)),
            )
        )

    final_completeness_bin_grid = evaluate_completeness_bin_grid(current_raw_params, observable_context)
    final_effective_completeness_grid = compute_effective_completeness_grid(
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
    final_payload = fit_single_joint_model(context=final_context, spec=spec)
    final_complete_survivor_intensity_grid = compute_complete_survivor_intensity_grid(
        final_payload["model"],
        base_context=base_context,
    )
    final_predicted_complete_counts = predict_complete_observable_histogram(
        complete_survivor_intensity_grid=final_complete_survivor_intensity_grid,
        base_context=base_context,
        observable_context=observable_context,
    )
    final_predicted_observed_counts = final_predicted_complete_counts * final_completeness_bin_grid

    iteration_history_table = pd.DataFrame([asdict(row) for row in iteration_rows])
    completeness_grid_table = build_completeness_grid_table(observable_context, final_completeness_bin_grid)
    observable_histogram_table = build_observable_histogram_table(
        observable_context=observable_context,
        predicted_complete_counts=final_predicted_complete_counts,
        predicted_observed_counts=final_predicted_observed_counts,
        completeness_bin_grid=final_completeness_bin_grid,
    )
    catalog_completeness_table = build_catalog_completeness_table(
        catalog=working,
        context=final_context,
        observable_context=observable_context,
        completeness_raw_params=current_raw_params,
    )

    outputs_tables = project_root / "outputs" / "tables"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    iteration_history_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_em_iteration_history.csv",
        index=False,
    )
    completeness_grid_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_em_completeness_grid.csv",
        index=False,
    )
    observable_histogram_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_em_observable_histogram.csv",
        index=False,
    )
    catalog_completeness_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_em_catalog_completeness.csv",
        index=False,
    )

    summary_payload = {
        "spec": asdict(spec),
        "selection_offset_dex": selection_offset_dex,
        "sun_galactocentric_radius_kpc": sun_galactocentric_radius_kpc,
        "n_iterations": n_iterations,
        "relaxation": relaxation,
        "baseline_total_initial_count": float(baseline_payload["model"]["total_initial_count"]),
        "baseline_raw_survival_fraction": float(baseline_payload["model"]["raw_survival_fraction"]),
        "final_total_initial_count": float(final_payload["model"]["total_initial_count"]),
        "final_selection_fraction": float(final_payload["model"]["selection_fraction"]),
        "final_raw_survival_fraction": float(final_payload["model"]["raw_survival_fraction"]),
        "final_mean_detectability": float(
            final_payload["model"]["selection_fraction"] / max(final_payload["model"]["raw_survival_fraction"], 1.0e-12)
        ),
        "total_initial_count_ratio_vs_baseline": float(
            final_payload["model"]["total_initial_count"] / max(baseline_payload["model"]["total_initial_count"], 1.0e-12)
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
        },
        "baseline_model": asdict(baseline_payload["summary"]),
        "final_model": {
            **asdict(final_payload["summary"]),
            "raw_survival_fraction": float(final_payload["model"]["raw_survival_fraction"]),
            "selection_fraction": float(final_payload["model"]["selection_fraction"]),
        },
        "iteration_history": iteration_history_table.to_dict(orient="records"),
    }
    (outputs_tables / "joint_fixed_survival_detectability_em_summary.json").write_text(
        json.dumps(summary_payload, indent=2)
    )

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


def fit_shared_imf_two_component_detectability_em_models(
    catalog: pd.DataFrame,
    project_root: Path,
    imf_families: list[str] | None = None,
    radial_models: list[str] | None = None,
    **kwargs,
) -> dict[str, object]:
    if imf_families is None:
        imf_families = ["lognormal", "powerlaw", "schechter"]
    if radial_models is None:
        radial_models = ["step5", "logpoly3"]

    env = prepare_two_component_detectability_environment(catalog=catalog, **kwargs)
    specs = build_shared_imf_two_component_specs(imf_families=imf_families, radial_models=radial_models)
    all_results = []
    summary_rows: list[dict[str, object]] = []
    for spec in specs:
        result = fit_shared_imf_two_component_detectability_em_single_model(
            spec=spec,
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
        outputs_tables / "joint_fixed_survival_detectability_shared_imf_two_component_model_summary.csv",
        index=False,
    )
    best_component_summary_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_shared_imf_two_component_best_component_summary.csv",
        index=False,
    )
    best_imf_grid_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_shared_imf_two_component_best_imf_grids.csv",
        index=False,
    )
    best_radial_grid_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_shared_imf_two_component_best_radial_grids.csv",
        index=False,
    )
    catalog_prediction_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_shared_imf_two_component_catalog_predictions.csv",
        index=False,
    )
    best_result["iteration_history_table"].to_csv(
        outputs_tables / "joint_fixed_survival_detectability_shared_imf_two_component_best_em_iteration_history.csv",
        index=False,
    )
    best_result["completeness_grid_table"].to_csv(
        outputs_tables / "joint_fixed_survival_detectability_shared_imf_two_component_best_completeness_grid.csv",
        index=False,
    )
    best_result["observable_histogram_table"].to_csv(
        outputs_tables / "joint_fixed_survival_detectability_shared_imf_two_component_best_observable_histogram.csv",
        index=False,
    )
    best_result["catalog_completeness_table"].to_csv(
        outputs_tables / "joint_fixed_survival_detectability_shared_imf_two_component_best_catalog_completeness.csv",
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
    (outputs_tables / "joint_fixed_survival_detectability_shared_imf_two_component_model_summary.json").write_text(
        json.dumps(detailed_summary, indent=2)
    )

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


def fit_separate_imf_two_component_detectability_em_models(
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

    env = prepare_two_component_detectability_environment(catalog=catalog, **kwargs)
    all_results = []
    pair_summary_rows: list[dict[str, object]] = []
    for in_situ_spec in component_model_specs:
        for accreted_spec in component_model_specs:
            result = fit_separate_imf_two_component_detectability_em_single_model(
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

    component_summary_rows = []
    for component_label in ("in_situ", "accreted"):
        table = pd.DataFrame(
            [row for row in pair_summary_rows if row[f"{component_label}_imf_family"] is not None]
        )
        if not table.empty:
            component_summary_rows.extend(
                table[
                    [
                        f"{component_label}_imf_family",
                        f"{component_label}_radial_model",
                    ]
                ]
                .drop_duplicates()
                .assign(component_label=component_label)
                .to_dict(orient="records")
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
        outputs_tables / "joint_fixed_survival_detectability_two_component_component_model_summary.csv",
        index=False,
    )
    pair_summary_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_two_component_model_summary.csv",
        index=False,
    )
    best_component_summary_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_two_component_best_component_summary.csv",
        index=False,
    )
    best_imf_grid_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_two_component_best_imf_grids.csv",
        index=False,
    )
    best_radial_grid_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_two_component_best_radial_grids.csv",
        index=False,
    )
    catalog_prediction_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_two_component_catalog_predictions.csv",
        index=False,
    )
    best_result["iteration_history_table"].to_csv(
        outputs_tables / "joint_fixed_survival_detectability_two_component_best_em_iteration_history.csv",
        index=False,
    )
    best_result["completeness_grid_table"].to_csv(
        outputs_tables / "joint_fixed_survival_detectability_two_component_best_completeness_grid.csv",
        index=False,
    )
    best_result["observable_histogram_table"].to_csv(
        outputs_tables / "joint_fixed_survival_detectability_two_component_best_observable_histogram.csv",
        index=False,
    )
    best_result["catalog_completeness_table"].to_csv(
        outputs_tables / "joint_fixed_survival_detectability_two_component_best_catalog_completeness.csv",
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
    (outputs_tables / "joint_fixed_survival_detectability_two_component_model_summary.json").write_text(
        json.dumps(detailed_summary, indent=2)
    )

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


def prepare_two_component_detectability_environment(
    catalog: pd.DataFrame,
    n_present_mass_bins: int = 6,
    n_distance_bins: int = 6,
    n_latitude_bins: int = 6,
    n_geometry_samples: int = 5000,
    sun_galactocentric_radius_kpc: float = 8.2,
) -> dict[str, object]:
    working, subsets, selection_offset_dex, survival_grid, component_base_contexts = prepare_two_component_contexts(catalog)
    base_context = JointLikelihoodContext.from_catalog_and_survival_grid(working, survival_grid)
    observable_context = build_observable_prediction_context(
        catalog=working,
        base_context=base_context,
        n_present_mass_bins=n_present_mass_bins,
        n_distance_bins=n_distance_bins,
        n_latitude_bins=n_latitude_bins,
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


def fit_shared_imf_two_component_detectability_em_single_model(
    working: pd.DataFrame,
    subsets: dict[str, pd.DataFrame],
    base_context: JointLikelihoodContext,
    observable_context: ObservablePredictionContext,
    component_base_contexts: dict[str, JointLikelihoodContext],
    spec: SharedImfTwoComponentSpec,
    n_iterations: int = 6,
    relaxation: float = 0.7,
    **_: object,
) -> dict[str, object]:
    baseline_payload = fit_shared_imf_two_component_single_model(contexts=component_base_contexts, spec=spec)
    current_raw_params = fit_logistic_completeness_model(
        observable_context=observable_context,
        predicted_complete_counts=predict_complete_observable_histogram(
            complete_survivor_intensity_grid=compute_shared_complete_survivor_intensity_grid(
                baseline_payload["model"],
                base_context=base_context,
            ),
            base_context=base_context,
            observable_context=observable_context,
        ),
        start_params=None,
    )["raw_parameters"]

    iteration_rows: list[DetectabilityIterationSummary] = []
    current_contexts = component_base_contexts
    current_payload = baseline_payload

    for iteration in range(1, n_iterations + 1):
        completeness_bin_grid = evaluate_completeness_bin_grid(current_raw_params, observable_context)
        effective_completeness_grid = compute_effective_completeness_grid(
            observable_context=observable_context,
            completeness_bin_grid=completeness_bin_grid,
        )
        current_contexts = apply_effective_completeness_to_component_contexts(
            component_base_contexts=component_base_contexts,
            effective_completeness_grid=effective_completeness_grid,
        )
        current_payload = fit_shared_imf_two_component_single_model(contexts=current_contexts, spec=spec)
        predicted_complete_counts = predict_complete_observable_histogram(
            complete_survivor_intensity_grid=compute_shared_complete_survivor_intensity_grid(
                current_payload["model"],
                base_context=base_context,
            ),
            base_context=base_context,
            observable_context=observable_context,
        )
        completeness_fit = fit_logistic_completeness_model(
            observable_context=observable_context,
            predicted_complete_counts=predicted_complete_counts,
            start_params=current_raw_params,
        )
        target_raw_params = completeness_fit["raw_parameters"]
        current_raw_params = (1.0 - relaxation) * current_raw_params + relaxation * target_raw_params
        updated_completeness_bin_grid = evaluate_completeness_bin_grid(current_raw_params, observable_context)
        predicted_observed_counts = predicted_complete_counts * updated_completeness_bin_grid
        selection_stats = aggregate_two_component_selection_stats(
            counts_by_component={
                label: len(context.log_mass_data)
                for label, context in current_contexts.items()
            },
            total_initial_count_by_component=current_payload["model"]["total_initial_count"],
            raw_survival_fraction_by_component=current_payload["model"]["raw_survival_fraction"],
            selection_fraction_by_component=current_payload["model"]["selection_fraction"],
        )
        iteration_rows.append(
            DetectabilityIterationSummary(
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
                predicted_complete_survivor_count=float(np.sum(predicted_complete_counts)),
                predicted_observed_count=float(np.sum(predicted_observed_counts)),
            )
        )

    final_completeness_bin_grid = evaluate_completeness_bin_grid(current_raw_params, observable_context)
    final_effective_completeness_grid = compute_effective_completeness_grid(
        observable_context=observable_context,
        completeness_bin_grid=final_completeness_bin_grid,
    )
    final_contexts = apply_effective_completeness_to_component_contexts(
        component_base_contexts=component_base_contexts,
        effective_completeness_grid=final_effective_completeness_grid,
    )
    final_payload = fit_shared_imf_two_component_single_model(contexts=final_contexts, spec=spec)
    final_predicted_complete_counts = predict_complete_observable_histogram(
        complete_survivor_intensity_grid=compute_shared_complete_survivor_intensity_grid(
            final_payload["model"],
            base_context=base_context,
        ),
        base_context=base_context,
        observable_context=observable_context,
    )
    final_predicted_observed_counts = final_predicted_complete_counts * final_completeness_bin_grid
    iteration_history_table = pd.DataFrame([asdict(row) for row in iteration_rows])
    completeness_grid_table = build_completeness_grid_table(observable_context, final_completeness_bin_grid)
    observable_histogram_table = build_observable_histogram_table(
        observable_context=observable_context,
        predicted_complete_counts=final_predicted_complete_counts,
        predicted_observed_counts=final_predicted_observed_counts,
        completeness_bin_grid=final_completeness_bin_grid,
    )
    catalog_completeness_table = build_catalog_completeness_table(
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


def fit_separate_imf_two_component_detectability_em_single_model(
    working: pd.DataFrame,
    subsets: dict[str, pd.DataFrame],
    base_context: JointLikelihoodContext,
    observable_context: ObservablePredictionContext,
    component_base_contexts: dict[str, JointLikelihoodContext],
    in_situ_spec: JointModelSpec,
    accreted_spec: JointModelSpec,
    n_iterations: int = 6,
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
    current_raw_params = fit_logistic_completeness_model(
        observable_context=observable_context,
        predicted_complete_counts=predict_complete_observable_histogram(
            complete_survivor_intensity_grid=compute_separate_complete_survivor_intensity_grid(
                component_payloads=baseline_component_payloads,
                base_context=base_context,
            ),
            base_context=base_context,
            observable_context=observable_context,
        ),
        start_params=None,
    )["raw_parameters"]

    iteration_rows: list[DetectabilityIterationSummary] = []
    current_contexts = component_base_contexts
    current_component_payloads = baseline_component_payloads
    current_pair_payload = baseline_pair_payload

    for iteration in range(1, n_iterations + 1):
        completeness_bin_grid = evaluate_completeness_bin_grid(current_raw_params, observable_context)
        effective_completeness_grid = compute_effective_completeness_grid(
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
        predicted_complete_counts = predict_complete_observable_histogram(
            complete_survivor_intensity_grid=compute_separate_complete_survivor_intensity_grid(
                component_payloads=current_component_payloads,
                base_context=base_context,
            ),
            base_context=base_context,
            observable_context=observable_context,
        )
        completeness_fit = fit_logistic_completeness_model(
            observable_context=observable_context,
            predicted_complete_counts=predicted_complete_counts,
            start_params=current_raw_params,
        )
        target_raw_params = completeness_fit["raw_parameters"]
        current_raw_params = (1.0 - relaxation) * current_raw_params + relaxation * target_raw_params
        updated_completeness_bin_grid = evaluate_completeness_bin_grid(current_raw_params, observable_context)
        predicted_observed_counts = predicted_complete_counts * updated_completeness_bin_grid
        selection_stats = aggregate_two_component_selection_stats(
            counts_by_component={
                "in_situ": len(subsets["in_situ"]),
                "accreted": len(subsets["accreted"]),
            },
            total_initial_count_by_component={
                label: payload["model"]["total_initial_count"]
                for label, payload in current_component_payloads.items()
            },
            raw_survival_fraction_by_component={
                label: payload["model"]["raw_survival_fraction"]
                for label, payload in current_component_payloads.items()
            },
            selection_fraction_by_component={
                label: payload["model"]["selection_fraction"]
                for label, payload in current_component_payloads.items()
            },
        )
        iteration_rows.append(
            DetectabilityIterationSummary(
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
                predicted_complete_survivor_count=float(np.sum(predicted_complete_counts)),
                predicted_observed_count=float(np.sum(predicted_observed_counts)),
            )
        )

    final_completeness_bin_grid = evaluate_completeness_bin_grid(current_raw_params, observable_context)
    final_effective_completeness_grid = compute_effective_completeness_grid(
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
    final_predicted_complete_counts = predict_complete_observable_histogram(
        complete_survivor_intensity_grid=compute_separate_complete_survivor_intensity_grid(
            component_payloads=final_component_payloads,
            base_context=base_context,
        ),
        base_context=base_context,
        observable_context=observable_context,
    )
    final_predicted_observed_counts = final_predicted_complete_counts * final_completeness_bin_grid
    iteration_history_table = pd.DataFrame([asdict(row) for row in iteration_rows])
    completeness_grid_table = build_completeness_grid_table(observable_context, final_completeness_bin_grid)
    observable_histogram_table = build_observable_histogram_table(
        observable_context=observable_context,
        predicted_complete_counts=final_predicted_complete_counts,
        predicted_observed_counts=final_predicted_observed_counts,
        completeness_bin_grid=final_completeness_bin_grid,
    )
    catalog_completeness_table = build_catalog_completeness_table(
        catalog=working,
        context=base_context.with_selection_probability_grid(
            np.clip(base_context.survival_probability_grid * final_effective_completeness_grid, 1.0e-12, 1.0)
        ),
        observable_context=observable_context,
        completeness_raw_params=current_raw_params,
    )
    selection_stats = aggregate_two_component_selection_stats(
        counts_by_component={
            "in_situ": len(subsets["in_situ"]),
            "accreted": len(subsets["accreted"]),
        },
        total_initial_count_by_component={
            label: payload["model"]["total_initial_count"]
            for label, payload in final_component_payloads.items()
        },
        raw_survival_fraction_by_component={
            label: payload["model"]["raw_survival_fraction"]
            for label, payload in final_component_payloads.items()
        },
        selection_fraction_by_component={
            label: payload["model"]["selection_fraction"]
            for label, payload in final_component_payloads.items()
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


def apply_effective_completeness_to_component_contexts(
    component_base_contexts: dict[str, JointLikelihoodContext],
    effective_completeness_grid: np.ndarray,
) -> dict[str, JointLikelihoodContext]:
    return {
        component_label: context.with_selection_probability_grid(
            np.clip(context.survival_probability_grid * effective_completeness_grid, 1.0e-12, 1.0)
        )
        for component_label, context in component_base_contexts.items()
    }


def compute_shared_complete_survivor_intensity_grid(
    model: dict[str, object],
    base_context: JointLikelihoodContext,
) -> np.ndarray:
    total = np.zeros_like(base_context.survival_probability_grid)
    for component_label in ("in_situ", "accreted"):
        total += compute_observed_intensity_grid(
            model["imf_density_grid"],
            model["radial_density_grid"][component_label],
            base_context.survival_probability_grid,
            model["total_initial_count"][component_label],
        )
    return total


def compute_separate_complete_survivor_intensity_grid(
    component_payloads: dict[str, dict[str, object]],
    base_context: JointLikelihoodContext,
) -> np.ndarray:
    total = np.zeros_like(base_context.survival_probability_grid)
    for payload in component_payloads.values():
        total += compute_complete_survivor_intensity_grid(
            model=payload["model"],
            base_context=base_context,
        )
    return total


def aggregate_two_component_selection_stats(
    counts_by_component: dict[str, int],
    total_initial_count_by_component: dict[str, float],
    raw_survival_fraction_by_component: dict[str, float],
    selection_fraction_by_component: dict[str, float],
) -> dict[str, float]:
    total_initial_count = float(sum(total_initial_count_by_component.values()))
    observed_total = float(sum(counts_by_component.values()))
    weighted_raw_survival = float(
        sum(
            float(total_initial_count_by_component[label]) * float(raw_survival_fraction_by_component[label])
            for label in total_initial_count_by_component
        )
        / max(total_initial_count, 1.0e-12)
    )
    weighted_selection = float(observed_total / max(total_initial_count, 1.0e-12))
    _ = selection_fraction_by_component
    mean_detectability = float(weighted_selection / max(weighted_raw_survival, 1.0e-12))
    return {
        "total_initial_count": total_initial_count,
        "selection_fraction": weighted_selection,
        "raw_survival_fraction": weighted_raw_survival,
        "mean_detectability": mean_detectability,
    }


def build_shared_two_component_detectability_summary_row(
    detectability_result: dict[str, object],
) -> dict[str, object]:
    summary = detectability_result["final_payload"]["summary"]
    model = detectability_result["final_payload"]["model"]
    selection_stats = aggregate_two_component_selection_stats(
        counts_by_component={
            "in_situ": int(summary.n_clusters_in_situ),
            "accreted": int(summary.n_clusters_accreted),
        },
        total_initial_count_by_component=model["total_initial_count"],
        raw_survival_fraction_by_component=model["raw_survival_fraction"],
        selection_fraction_by_component=model["selection_fraction"],
    )
    row = asdict(summary)
    row["mean_detectability"] = float(selection_stats["mean_detectability"])
    row["selection_fraction_total"] = float(selection_stats["selection_fraction"])
    row["raw_survival_fraction_total"] = float(selection_stats["raw_survival_fraction"])
    return row


def build_separate_two_component_detectability_summary_row(
    detectability_result: dict[str, object],
) -> dict[str, object]:
    summary = detectability_result["pair_payload"]["summary"]
    selection_stats = aggregate_two_component_selection_stats(
        counts_by_component={
            "in_situ": int(summary.n_clusters_in_situ),
            "accreted": int(summary.n_clusters_accreted),
        },
        total_initial_count_by_component={
            label: payload["model"]["total_initial_count"]
            for label, payload in detectability_result["final_component_payloads"].items()
        },
        raw_survival_fraction_by_component={
            label: payload["model"]["raw_survival_fraction"]
            for label, payload in detectability_result["final_component_payloads"].items()
        },
        selection_fraction_by_component={
            label: payload["model"]["selection_fraction"]
            for label, payload in detectability_result["final_component_payloads"].items()
        },
    )
    row = asdict(summary)
    row["mean_detectability"] = float(selection_stats["mean_detectability"])
    row["selection_fraction_total"] = float(selection_stats["selection_fraction"])
    row["raw_survival_fraction_total"] = float(selection_stats["raw_survival_fraction"])
    return row


def build_detectability_corrected_performance_row(
    detectability_result: dict[str, object],
    n_mass_bins: int = 12,
    n_a_bins: int = 9,
) -> dict[str, object]:
    context = detectability_result["final_context"]
    model = detectability_result["final_payload"]["model"]
    summary = detectability_result["final_payload"]["summary"]
    mass_bin_edges = np.linspace(context.log_mass_grid[0], context.log_mass_grid[-1], n_mass_bins + 1)
    log_a_bin_edges = np.linspace(context.log_a_grid[0], context.log_a_grid[-1], n_a_bins + 1)
    observed_2d, _, _ = np.histogram2d(
        context.log_mass_data,
        context.log_a_data,
        bins=[mass_bin_edges, log_a_bin_edges],
    )
    point_intensity_grid = compute_observed_intensity_grid(
        model["imf_density_grid"],
        model["radial_density_grid"],
        context.selection_probability_grid,
        model["total_initial_count"],
    )
    expected_2d = rebin_expected_counts_2d(
        point_intensity_grid,
        log_mass_grid=context.log_mass_grid,
        log_a_grid=context.log_a_grid,
        mass_bin_edges=mass_bin_edges,
        log_a_bin_edges=log_a_bin_edges,
    )
    residual_sigma = (observed_2d - expected_2d) / np.sqrt(np.clip(expected_2d, 1.0, None))
    valid = expected_2d > 0.2
    rms_residual_sigma = float(np.sqrt(np.mean(np.square(residual_sigma[valid]))))
    mean_abs_residual_sigma = float(np.mean(np.abs(residual_sigma[valid])))
    return {
        "imf_family": summary.imf_family,
        "radial_model": summary.radial_model,
        "log_likelihood": float(summary.log_likelihood),
        "aic": float(summary.aic),
        "bic": float(summary.bic),
        "n_parameters": int(summary.n_parameters),
        "imf_parameters_json": str(summary.imf_parameters_json),
        "radial_parameters_json": str(summary.radial_parameters_json),
        "total_initial_count": float(model["total_initial_count"]),
        "selection_fraction": float(model["selection_fraction"]),
        "raw_survival_fraction": float(model["raw_survival_fraction"]),
        "mean_detectability": float(
            model["selection_fraction"] / max(model["raw_survival_fraction"], 1.0e-12)
        ),
        "rms_residual_sigma_2d": rms_residual_sigma,
        "mean_abs_residual_sigma_2d": mean_abs_residual_sigma,
    }


def build_observable_prediction_context(
    catalog: pd.DataFrame,
    base_context: JointLikelihoodContext,
    n_present_mass_bins: int,
    n_distance_bins: int,
    n_latitude_bins: int,
    n_geometry_samples: int,
    sun_galactocentric_radius_kpc: float,
) -> ObservablePredictionContext:
    present_mass_proxy = fit_present_mass_proxy_model(catalog)
    log_present_mass_mean_grid = predict_log_present_mass_grid(
        log_mass_grid=base_context.log_mass_grid,
        log_a_grid=base_context.log_a_grid,
        proxy_model=present_mass_proxy,
    )
    observed_log_present_mass = np.log10(catalog["present_mass_msun"].to_numpy())
    observed_distance_kpc = catalog["r_sun_kpc"].to_numpy()
    observed_abs_latitude_deg = np.abs(catalog["galactic_b_deg"].to_numpy())

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

    observed_counts, _ = np.histogramdd(
        np.column_stack([observed_log_present_mass, observed_distance_kpc, observed_abs_latitude_deg]),
        bins=[log_present_mass_edges, distance_edges_kpc, abs_latitude_edges_deg],
    )

    mass_bin_probabilities_grid = build_mass_bin_probabilities_grid(
        log_present_mass_mean_grid=log_present_mass_mean_grid,
        log_present_mass_edges=log_present_mass_edges,
        sigma_dex=present_mass_proxy.residual_sigma_dex,
    )
    sky_bin_probabilities_by_a = build_spherical_sky_bin_probabilities(
        a_grid_kpc=np.power(10.0, base_context.log_a_grid),
        distance_edges_kpc=distance_edges_kpc,
        abs_latitude_edges_deg=abs_latitude_edges_deg,
        n_geometry_samples=n_geometry_samples,
        sun_galactocentric_radius_kpc=sun_galactocentric_radius_kpc,
    )

    log_present_mass_centers = 0.5 * (log_present_mass_edges[:-1] + log_present_mass_edges[1:])
    distance_centers_kpc = np.sqrt(distance_edges_kpc[:-1] * distance_edges_kpc[1:])
    log_distance_centers = np.log10(distance_centers_kpc)
    abs_latitude_centers_deg = 0.5 * (abs_latitude_edges_deg[:-1] + abs_latitude_edges_deg[1:])

    return ObservablePredictionContext(
        present_mass_proxy=present_mass_proxy,
        log_present_mass_edges=log_present_mass_edges,
        distance_edges_kpc=distance_edges_kpc,
        abs_latitude_edges_deg=abs_latitude_edges_deg,
        log_present_mass_centers=log_present_mass_centers,
        log_distance_centers=log_distance_centers,
        abs_latitude_centers_deg=abs_latitude_centers_deg,
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
        sun_galactocentric_radius_kpc=sun_galactocentric_radius_kpc,
        n_geometry_samples=n_geometry_samples,
    )


def fit_present_mass_proxy_model(catalog: pd.DataFrame) -> PresentMassProxyModel:
    log_initial_mass = catalog["log_initial_mass_msun"].to_numpy()
    log_a = np.log10(catalog["semi_major_axis_kpc"].to_numpy())
    log_present_mass = np.log10(catalog["present_mass_msun"].to_numpy())
    log_mass_ratio = log_present_mass - log_initial_mass
    log_mass_loss = log_initial_mass - log_present_mass

    log_mass_mean = float(np.mean(log_initial_mass))
    log_a_mean = float(np.mean(log_a))
    log_mass_std = float(max(np.std(log_initial_mass), 1.0e-6))
    log_a_std = float(max(np.std(log_a), 1.0e-6))
    z_mass = (log_initial_mass - log_mass_mean) / log_mass_std
    z_a = (log_a - log_a_mean) / log_a_std

    def predict_loss(params: np.ndarray, eval_z_mass: np.ndarray, eval_z_a: np.ndarray) -> np.ndarray:
        b0, b1, b2, s0, s1 = np.asarray(params, dtype=float)
        radial = b0 + b1 * eval_z_a + b2 * np.square(eval_z_a)
        mass_slope = special.softplus(s0 + s1 * eval_z_a)
        return special.softplus(radial - mass_slope * eval_z_mass)

    def residual(params: np.ndarray) -> np.ndarray:
        return predict_loss(params, z_mass, z_a) - log_mass_loss

    starts = [
        np.array([0.0, -0.8, 0.15, -1.0, 0.0]),
        np.array([0.2, -0.8, 0.2, -1.0, 0.5]),
        np.array([0.2, -0.8, 0.2, -1.0, -0.5]),
        np.array([0.0, -0.4, 0.0, -2.0, 0.0]),
    ]
    best_result = None
    best_value = np.inf
    for start in starts:
        result = optimize.least_squares(residual, start, max_nfev=5000)
        value = float(np.sum(np.square(result.fun)))
        if value < best_value:
            best_value = value
            best_result = result
    if best_result is None:
        raise RuntimeError("Monotonic present-day mass proxy failed to fit.")
    coefficients = np.asarray(best_result.x, dtype=float)
    fitted_log_present_mass = log_initial_mass - predict_loss(coefficients, z_mass, z_a)
    residual_sigma_dex = float(max(np.std(log_present_mass - fitted_log_present_mass, ddof=len(coefficients)), 0.08))
    return PresentMassProxyModel(
        coefficients=np.asarray(coefficients, dtype=float),
        log_mass_mean=log_mass_mean,
        log_a_mean=log_a_mean,
        residual_sigma_dex=residual_sigma_dex,
        log_mass_ratio_min=float(np.min(log_mass_ratio)),
        log_present_mass_min=float(np.min(log_present_mass)),
        log_present_mass_max=float(np.max(log_present_mass)),
        log_mass_std=log_mass_std,
        log_a_std=log_a_std,
        model_kind="monotonic_mass_loss",
    )


def predict_log_present_mass_grid(
    log_mass_grid: np.ndarray,
    log_a_grid: np.ndarray,
    proxy_model: PresentMassProxyModel,
) -> np.ndarray:
    model_kind = getattr(proxy_model, "model_kind", "polynomial_log_mass_ratio")
    if model_kind == "monotonic_mass_loss":
        z_mass = (log_mass_grid[:, None] - proxy_model.log_mass_mean) / getattr(proxy_model, "log_mass_std", 1.0)
        z_a = (log_a_grid[None, :] - proxy_model.log_a_mean) / getattr(proxy_model, "log_a_std", 1.0)
        b0, b1, b2, s0, s1 = proxy_model.coefficients
        radial = b0 + b1 * z_a + b2 * np.square(z_a)
        mass_slope = special.softplus(s0 + s1 * z_a)
        log_mass_loss = special.softplus(radial - mass_slope * z_mass)
        log_present_mass = log_mass_grid[:, None] - log_mass_loss
    else:
        z_mass = log_mass_grid[:, None] - proxy_model.log_mass_mean
        z_a = log_a_grid[None, :] - proxy_model.log_a_mean
        c0, c1, c2, c3, c4 = proxy_model.coefficients
        log_mass_ratio = c0 + c1 * z_mass + c2 * z_a + c3 * z_mass * z_a + c4 * np.square(z_a)
        log_mass_ratio = np.clip(log_mass_ratio, proxy_model.log_mass_ratio_min, 0.0)
        log_present_mass = log_mass_grid[:, None] + log_mass_ratio
    return np.clip(
        log_present_mass,
        proxy_model.log_present_mass_min - 2.0 * proxy_model.residual_sigma_dex,
        proxy_model.log_present_mass_max + 2.0 * proxy_model.residual_sigma_dex,
    )


def build_mass_bin_probabilities_grid(
    log_present_mass_mean_grid: np.ndarray,
    log_present_mass_edges: np.ndarray,
    sigma_dex: float,
) -> np.ndarray:
    cdf_values = stats.norm.cdf(
        log_present_mass_edges[None, None, :],
        loc=log_present_mass_mean_grid[:, :, None],
        scale=sigma_dex,
    )
    probabilities = np.diff(cdf_values, axis=2)
    probability_sum = np.clip(np.sum(probabilities, axis=2, keepdims=True), 1.0e-12, None)
    return np.clip(probabilities / probability_sum, 1.0e-12, 1.0)


def build_spherical_sky_bin_probabilities(
    a_grid_kpc: np.ndarray,
    distance_edges_kpc: np.ndarray,
    abs_latitude_edges_deg: np.ndarray,
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
        (len(a_grid_kpc), len(distance_edges_kpc) - 1, len(abs_latitude_edges_deg) - 1),
        dtype=float,
    )
    for index, a_value in enumerate(a_grid_kpc):
        x = a_value * x_unit
        y = a_value * y_unit
        z = a_value * z_unit
        dx = x - sun_galactocentric_radius_kpc
        distance = np.sqrt(np.square(dx) + np.square(y) + np.square(z))
        abs_latitude = np.degrees(np.arcsin(np.clip(np.abs(z) / np.clip(distance, 1.0e-12, None), 0.0, 1.0)))
        counts, _, _ = np.histogram2d(
            distance,
            abs_latitude,
            bins=[distance_edges_kpc, abs_latitude_edges_deg],
        )
        probabilities[index] = counts / max(np.sum(counts), 1.0)
    return probabilities


def compute_complete_survivor_intensity_grid(
    model: dict[str, object],
    base_context: JointLikelihoodContext,
) -> np.ndarray:
    return compute_observed_intensity_grid(
        model["imf_density_grid"],
        model["radial_density_grid"],
        base_context.survival_probability_grid,
        model["total_initial_count"],
    )


def predict_complete_observable_histogram(
    complete_survivor_intensity_grid: np.ndarray,
    base_context: JointLikelihoodContext,
    observable_context: ObservablePredictionContext,
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
        "ak,adb->kdb",
        mass_counts_by_a,
        observable_context.sky_bin_probabilities_by_a,
    )


def fit_logistic_completeness_model(
    observable_context: ObservablePredictionContext,
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
        ]
    )
    starts = [default_start]
    if start_params is not None:
        starts.insert(0, np.asarray(start_params, dtype=float))
    starts.append(np.array([default_start[0], np.log(0.6), np.log(0.6), np.log(0.6)]))
    bounds = [(-8.0, 8.0), (-8.0, 4.0), (-8.0, 4.0), (-8.0, 4.0)]

    best_result = None
    best_value = np.inf
    for start in starts:
        result = optimize.minimize(
            lambda params: negative_completeness_log_likelihood(
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
        raise RuntimeError("Completeness optimization failed to start.")
    completeness_bin_grid = evaluate_completeness_bin_grid(best_result.x, observable_context)
    return {
        "raw_parameters": np.asarray(best_result.x, dtype=float),
        "completeness_bin_grid": completeness_bin_grid,
        "negative_log_likelihood": float(best_result.fun),
        "success": bool(best_result.success),
        "message": str(best_result.message),
    }


def negative_completeness_log_likelihood(
    params: np.ndarray,
    observable_context: ObservablePredictionContext,
    predicted_complete_counts: np.ndarray,
) -> float:
    completeness_bin_grid = evaluate_completeness_bin_grid(params, observable_context)
    mu = np.clip(predicted_complete_counts * completeness_bin_grid, 1.0e-12, None)
    observed = observable_context.observed_counts
    return float(-(np.sum(observed * np.log(mu) - mu)))


def evaluate_completeness_bin_grid(
    raw_params: np.ndarray,
    observable_context: ObservablePredictionContext,
) -> np.ndarray:
    intercept = float(raw_params[0])
    mass_slope = float(np.exp(raw_params[1]))
    distance_slope = float(np.exp(raw_params[2]))
    latitude_slope = float(np.exp(raw_params[3]))

    z_mass = (
        observable_context.log_present_mass_centers[:, None, None] - observable_context.log_present_mass_feature_mean
    ) / observable_context.log_present_mass_feature_std
    z_distance = (
        observable_context.log_distance_centers[None, :, None] - observable_context.log_distance_feature_mean
    ) / observable_context.log_distance_feature_std
    z_latitude = (
        observable_context.abs_latitude_centers_deg[None, None, :] - observable_context.abs_latitude_feature_mean
    ) / observable_context.abs_latitude_feature_std
    logits = intercept + mass_slope * z_mass - distance_slope * z_distance + latitude_slope * z_latitude
    return np.clip(special.expit(logits), 1.0e-6, 1.0)


def compute_effective_completeness_grid(
    observable_context: ObservablePredictionContext,
    completeness_bin_grid: np.ndarray,
) -> np.ndarray:
    sky_averaged_completeness = np.einsum(
        "adb,kdb->ak",
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


def build_completeness_grid_table(
    observable_context: ObservablePredictionContext,
    completeness_bin_grid: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for i_mass, log_mass_center in enumerate(observable_context.log_present_mass_centers):
        for i_distance, log_distance_center in enumerate(observable_context.log_distance_centers):
            for i_latitude, latitude_center in enumerate(observable_context.abs_latitude_centers_deg):
                rows.append(
                    {
                        "present_mass_bin_index": i_mass,
                        "distance_bin_index": i_distance,
                        "latitude_bin_index": i_latitude,
                        "log10_present_mass_center_msun": float(log_mass_center),
                        "present_mass_center_msun": float(np.power(10.0, log_mass_center)),
                        "distance_center_kpc": float(np.power(10.0, log_distance_center)),
                        "log10_distance_center_kpc": float(log_distance_center),
                        "abs_latitude_center_deg": float(latitude_center),
                        "completeness": float(completeness_bin_grid[i_mass, i_distance, i_latitude]),
                    }
                )
    return pd.DataFrame(rows)


def build_observable_histogram_table(
    observable_context: ObservablePredictionContext,
    predicted_complete_counts: np.ndarray,
    predicted_observed_counts: np.ndarray,
    completeness_bin_grid: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for i_mass, log_mass_center in enumerate(observable_context.log_present_mass_centers):
        for i_distance, log_distance_center in enumerate(observable_context.log_distance_centers):
            for i_latitude, latitude_center in enumerate(observable_context.abs_latitude_centers_deg):
                rows.append(
                    {
                        "present_mass_bin_index": i_mass,
                        "distance_bin_index": i_distance,
                        "latitude_bin_index": i_latitude,
                        "log10_present_mass_center_msun": float(log_mass_center),
                        "distance_center_kpc": float(np.power(10.0, log_distance_center)),
                        "abs_latitude_center_deg": float(latitude_center),
                        "observed_count": float(observable_context.observed_counts[i_mass, i_distance, i_latitude]),
                        "predicted_complete_count": float(predicted_complete_counts[i_mass, i_distance, i_latitude]),
                        "predicted_observed_count": float(predicted_observed_counts[i_mass, i_distance, i_latitude]),
                        "completeness": float(completeness_bin_grid[i_mass, i_distance, i_latitude]),
                    }
                )
    return pd.DataFrame(rows)


def build_catalog_completeness_table(
    catalog: pd.DataFrame,
    context: JointLikelihoodContext,
    observable_context: ObservablePredictionContext,
    completeness_raw_params: np.ndarray,
) -> pd.DataFrame:
    completeness_bin_grid = evaluate_completeness_bin_grid(completeness_raw_params, observable_context)
    effective_completeness_grid = compute_effective_completeness_grid(observable_context, completeness_bin_grid)
    effective_interpolator = context.with_selection_probability_grid(
        np.clip(context.survival_probability_grid * effective_completeness_grid, 1.0e-12, 1.0)
    ).selection_interpolator

    log_present_mass = np.log10(catalog["present_mass_msun"].to_numpy())
    log_distance = np.log10(catalog["r_sun_kpc"].to_numpy())
    abs_latitude = np.abs(catalog["galactic_b_deg"].to_numpy())
    actual_completeness = evaluate_completeness_at_values(
        raw_params=completeness_raw_params,
        observable_context=observable_context,
        log_present_mass=log_present_mass,
        log_distance=log_distance,
        abs_latitude_deg=abs_latitude,
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
                "raw_survival_probability": float(raw_survival[index]),
                "detectability_probability_at_observed_values": float(actual_completeness[index]),
                "effective_selection_probability_intrinsic": float(effective_selection[index]),
                "effective_detectability_intrinsic": float(effective_selection[index] / raw_survival[index]),
            }
        )
    return pd.DataFrame(rows)


def evaluate_completeness_at_values(
    raw_params: np.ndarray,
    observable_context: ObservablePredictionContext,
    log_present_mass: np.ndarray,
    log_distance: np.ndarray,
    abs_latitude_deg: np.ndarray,
) -> np.ndarray:
    intercept = float(raw_params[0])
    mass_slope = float(np.exp(raw_params[1]))
    distance_slope = float(np.exp(raw_params[2]))
    latitude_slope = float(np.exp(raw_params[3]))
    z_mass = (log_present_mass - observable_context.log_present_mass_feature_mean) / (
        observable_context.log_present_mass_feature_std
    )
    z_distance = (log_distance - observable_context.log_distance_feature_mean) / (
        observable_context.log_distance_feature_std
    )
    z_latitude = (abs_latitude_deg - observable_context.abs_latitude_feature_mean) / (
        observable_context.abs_latitude_feature_std
    )
    logits = intercept + mass_slope * z_mass - distance_slope * z_distance + latitude_slope * z_latitude
    return np.clip(special.expit(logits), 1.0e-6, 1.0)
