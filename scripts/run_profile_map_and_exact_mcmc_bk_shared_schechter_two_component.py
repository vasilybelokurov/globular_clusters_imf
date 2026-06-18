from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import warnings
from dataclasses import asdict
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

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"globular_clusters_imf\.model")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"globular_clusters_imf\.smooth_survivability")

from scan_schechter_survival_time_multipliers import _plot_logl_vs_multiplier
from run_profile_map_and_exact_mcmc_schechter_powerlaw_a import (
    GridSpec,
    _best_entry_on_edge,
    _build_anchor_library,
    _build_refined_spec_from_coarse_region,
    _catalog_and_survival_grid_for_theta,
    _compute_rhat,
    _corner_plot,
    _entry_stage_copy,
    _expand_refined_spec,
    _lightweight_entry,
    _round_key,
    _save_best_payload,
    _select_anchor_start_state,
    _select_diverse_entries,
    _select_high_likelihood_coarse_rows,
    _trace_plot,
)

LOG_MASS_MIN = 4.0
SURFACE_MODEL = "logistic"


def _survival_grid_override_from_smooth_survival(smooth_survival: dict[str, object]) -> dict[str, object]:
    return {
        "log_mass_grid": np.asarray(smooth_survival["log_mass_grid"], dtype=float),
        "log_a_grid": np.asarray(smooth_survival["log_a_grid"], dtype=float),
        "semi_major_axis_grid_kpc": np.asarray(smooth_survival["semi_major_axis_grid_kpc"], dtype=float),
        "survival_probability": np.asarray(smooth_survival["survival_probability"], dtype=float),
        "selection_offset_dex": 0.0,
        "bandwidth_log10_a_dex": float(smooth_survival["bandwidth_log10_a_dex"]),
        "smooth_survivability_summary": smooth_survival["summary"],
    }


def _load_catalog() -> pd.DataFrame:
    catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"BK-labeled two-component run requires {catalog_path} with origin_flag labels."
        )
    return pd.read_csv(catalog_path)


def _restrict_log_mass_support(
    log_mass_grid: np.ndarray,
    density_grid: np.ndarray,
    log_mass_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    log_mass_grid = np.asarray(log_mass_grid, dtype=float)
    density_grid = np.asarray(density_grid, dtype=float)
    mask = log_mass_grid >= float(log_mass_min)
    if not np.any(mask):
        raise ValueError("Requested log-mass threshold lies above the model support.")
    return log_mass_grid[mask], density_grid[mask]


def _count_and_mass_above_log_mass(
    *,
    total_initial_count: float,
    log_mass_grid: np.ndarray,
    imf_density_grid: np.ndarray,
    log_mass_min: float,
) -> tuple[float, float]:
    x, y = _restrict_log_mass_support(log_mass_grid, imf_density_grid, log_mass_min)
    number_fraction = float(np.trapezoid(y, x))
    mean_mass_above = float(np.trapezoid(np.power(10.0, x) * y, x) / max(number_fraction, 1.0e-12))
    return float(total_initial_count * number_fraction), float(total_initial_count * number_fraction * mean_mass_above)


def _selection_stats_above_log_mass(
    *,
    context,
    imf_density_grid: np.ndarray,
    radial_density_grid: np.ndarray,
    log_mass_min: float,
) -> dict[str, float]:
    log_mass_grid = np.asarray(context.log_mass_grid, dtype=float)
    radial_support = np.asarray(context.log_a_grid, dtype=float)
    survival_grid = np.asarray(context.survival_probability_grid, dtype=float)
    selection_grid = np.asarray(context.selection_probability_grid, dtype=float)

    mass_support, imf_support = _restrict_log_mass_support(log_mass_grid, imf_density_grid, log_mass_min)
    survival_support = np.vstack(
        [np.interp(mass_support, log_mass_grid, survival_grid[:, j]) for j in range(survival_grid.shape[1])]
    ).T
    selection_support = np.vstack(
        [np.interp(mass_support, log_mass_grid, selection_grid[:, j]) for j in range(selection_grid.shape[1])]
    ).T
    integrand_base = imf_support[:, None] * np.asarray(radial_density_grid, dtype=float)[None, :]
    raw_survival_fraction_above = float(
        np.trapezoid(np.trapezoid(integrand_base * survival_support, radial_support, axis=1), mass_support)
    )
    selection_fraction_above = float(
        np.trapezoid(np.trapezoid(integrand_base * selection_support, radial_support, axis=1), mass_support)
    )
    return {
        "raw_survival_fraction_above_log10_4": raw_survival_fraction_above,
        "selection_fraction_above_log10_4": selection_fraction_above,
        "mean_detectability_above_log10_4": float(
            selection_fraction_above / max(raw_survival_fraction_above, 1.0e-12)
        ),
    }


def _prepare_two_component_environment_with_smooth_survival(
    *,
    prepared_catalog: pd.DataFrame,
    survival_grid_override: dict[str, object],
    n_present_mass_bins: int = 6,
    n_distance_bins: int = 6,
    n_latitude_bins: int = 6,
    n_longitude_bins: int = 6,
    n_geometry_samples: int = 5000,
    sun_galactocentric_radius_kpc: float = 8.2,
) -> dict[str, object]:
    from globular_clusters_imf.detectability_longitude_model import build_observable_prediction_context_with_abs_longitude
    from globular_clusters_imf.joint_model import JointLikelihoodContext

    working = prepared_catalog.copy()
    if "origin_flag" not in working.columns:
        raise ValueError("Prepared catalog is missing origin_flag required for BK-labeled two-component fitting.")
    working["origin_flag"] = pd.to_numeric(working["origin_flag"], errors="coerce").astype("Int64")
    subsets = {
        "in_situ": working.loc[working["origin_flag"] == 1].copy(),
        "accreted": working.loc[working["origin_flag"] == 0].copy(),
    }
    for component_label, subset in subsets.items():
        if subset.empty:
            raise ValueError(f"No clusters found for component {component_label!r}.")
        subset["origin_label"] = component_label

    base_context = JointLikelihoodContext.from_catalog_and_survival_grid(working, survival_grid_override)
    component_base_contexts = {
        component_label: JointLikelihoodContext.from_catalog_and_survival_grid(subset, survival_grid_override)
        for component_label, subset in subsets.items()
    }
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
        "base_context": base_context,
        "observable_context": observable_context,
        "component_base_contexts": component_base_contexts,
    }


def _fit_shared_schechter_two_component_single_model_fixed_imf(
    *,
    contexts: dict[str, object],
    spec,
    fixed_imf_params: np.ndarray,
    start_radial_params_by_component: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    from scipy import optimize

    from globular_clusters_imf.two_component_model import (
        SharedImfTwoComponentJointFitResult,
        radial_parameter_bounds,
        shared_full_log_likelihood_from_model,
        shared_negative_profile_log_likelihood,
        unique_radial_starts,
        unpack_shared_imf_two_component_model,
    )

    fixed_imf_params = np.asarray(fixed_imf_params, dtype=float)
    in_situ_bounds = radial_parameter_bounds(spec.in_situ_radial_model)
    accreted_bounds = radial_parameter_bounds(spec.accreted_radial_model)
    radial_bounds = list(in_situ_bounds) + list(accreted_bounds)

    starts: list[np.ndarray] = []
    if start_radial_params_by_component is not None:
        starts.append(
            np.concatenate(
                [
                    np.asarray(start_radial_params_by_component["in_situ"], dtype=float),
                    np.asarray(start_radial_params_by_component["accreted"], dtype=float),
                ]
            )
        )
    for in_situ_start in unique_radial_starts(spec.in_situ_radial_model):
        for accreted_start in unique_radial_starts(spec.accreted_radial_model):
            candidate = np.concatenate([np.asarray(in_situ_start, dtype=float), np.asarray(accreted_start, dtype=float)])
            if not any(np.allclose(candidate, existing) for existing in starts):
                starts.append(candidate)

    best_result = None
    best_value = np.inf

    def objective(radial_params: np.ndarray) -> float:
        full_params = np.concatenate([fixed_imf_params, np.asarray(radial_params, dtype=float)])
        return float(shared_negative_profile_log_likelihood(full_params, contexts=contexts, spec=spec))

    for start in starts:
        result = optimize.minimize(
            objective,
            x0=np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=radial_bounds,
        )
        if float(result.fun) < best_value:
            best_value = float(result.fun)
            best_result = result

    if best_result is None:
        raise RuntimeError("Fixed-IMF shared two-component optimization failed to initialize.")

    radial_params = np.asarray(best_result.x, dtype=float)
    n_radial_in_situ = len(in_situ_bounds)
    full_params = np.concatenate([fixed_imf_params, radial_params])
    model = unpack_shared_imf_two_component_model(full_params, contexts=contexts, spec=spec)
    log_likelihood = shared_full_log_likelihood_from_model(model, contexts)
    n_clusters_in_situ = len(contexts["in_situ"].log_mass_data)
    n_clusters_accreted = len(contexts["accreted"].log_mass_data)
    n_clusters_total = n_clusters_in_situ + n_clusters_accreted
    n_parameters = len(full_params)
    summary = SharedImfTwoComponentJointFitResult(
        imf_family=spec.imf_family,
        in_situ_radial_model=spec.in_situ_radial_model,
        accreted_radial_model=spec.accreted_radial_model,
        log_likelihood=float(log_likelihood),
        aic=float(2 * n_parameters - 2 * log_likelihood),
        bic=float(np.log(n_clusters_total) * n_parameters - 2 * log_likelihood),
        delta_bic=np.nan,
        n_parameters=n_parameters,
        n_clusters_total=n_clusters_total,
        n_clusters_in_situ=n_clusters_in_situ,
        n_clusters_accreted=n_clusters_accreted,
        total_initial_count_in_situ=float(model["total_initial_count"]["in_situ"]),
        total_initial_count_accreted=float(model["total_initial_count"]["accreted"]),
        total_initial_count=float(model["total_initial_count"]["in_situ"] + model["total_initial_count"]["accreted"]),
        survival_fraction_in_situ=float(model["survival_fraction"]["in_situ"]),
        survival_fraction_accreted=float(model["survival_fraction"]["accreted"]),
        shared_imf_parameters_json=json.dumps(model["imf_parameters"]),
        in_situ_radial_parameters_json=json.dumps(model["radial_parameters"]["in_situ"]),
        accreted_radial_parameters_json=json.dumps(model["radial_parameters"]["accreted"]),
    )
    return {
        "summary": summary,
        "model": model,
        "spec": spec,
        "raw_parameters": full_params,
        "radial_parameters_raw": {
            "in_situ": radial_params[:n_radial_in_situ].copy(),
            "accreted": radial_params[n_radial_in_situ:].copy(),
        },
    }


def _fit_shared_schechter_two_component_detectability_em_fixed_imf_with_abs_longitude(
    *,
    working: pd.DataFrame,
    subsets: dict[str, pd.DataFrame],
    base_context,
    observable_context,
    component_base_contexts,
    spec,
    fixed_imf_params: np.ndarray,
    n_iterations: int,
    relaxation: float,
    start_completeness_raw_parameters: np.ndarray | None,
    start_radial_params_by_component: dict[str, np.ndarray] | None,
) -> dict[str, object]:
    from globular_clusters_imf.detectability_longitude_model import (
        DetectabilityAbsLongitudeTwoComponentIterationSummary,
        build_catalog_completeness_table_with_abs_longitude,
        build_completeness_grid_table_with_abs_longitude,
        build_observable_histogram_table_with_abs_longitude,
        compute_effective_completeness_grid_with_abs_longitude,
        compute_shared_complete_survivor_intensity_grid,
        evaluate_completeness_bin_grid_with_abs_longitude,
        fit_logistic_completeness_model_with_abs_longitude,
        predict_complete_observable_histogram_with_abs_longitude,
    )
    from globular_clusters_imf.detectability_model import aggregate_two_component_selection_stats, apply_effective_completeness_to_component_contexts

    baseline_payload = _fit_shared_schechter_two_component_single_model_fixed_imf(
        contexts=component_base_contexts,
        spec=spec,
        fixed_imf_params=fixed_imf_params,
        start_radial_params_by_component=start_radial_params_by_component,
    )
    if start_completeness_raw_parameters is None:
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
    else:
        current_raw_params = np.asarray(start_completeness_raw_parameters, dtype=float).copy()

    iteration_rows = []
    current_payload = baseline_payload
    current_contexts = component_base_contexts
    current_radial_state = current_payload["radial_parameters_raw"]

    for iteration in range(1, int(n_iterations) + 1):
        completeness_bin_grid = evaluate_completeness_bin_grid_with_abs_longitude(current_raw_params, observable_context)
        effective_completeness_grid = compute_effective_completeness_grid_with_abs_longitude(
            observable_context=observable_context,
            completeness_bin_grid=completeness_bin_grid,
        )
        current_contexts = apply_effective_completeness_to_component_contexts(
            component_base_contexts=component_base_contexts,
            effective_completeness_grid=effective_completeness_grid,
        )
        current_payload = _fit_shared_schechter_two_component_single_model_fixed_imf(
            contexts=current_contexts,
            spec=spec,
            fixed_imf_params=fixed_imf_params,
            start_radial_params_by_component=current_radial_state,
        )
        current_radial_state = current_payload["radial_parameters_raw"]
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
        current_raw_params = (1.0 - float(relaxation)) * current_raw_params + float(relaxation) * target_raw_params
        updated_completeness_bin_grid = evaluate_completeness_bin_grid_with_abs_longitude(current_raw_params, observable_context)
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
    final_payload = _fit_shared_schechter_two_component_single_model_fixed_imf(
        contexts=final_contexts,
        spec=spec,
        fixed_imf_params=fixed_imf_params,
        start_radial_params_by_component=current_radial_state,
    )
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
            "input_alpha_dndm": float(fixed_imf_params[0]),
            "input_log10_m_c_msun": float(fixed_imf_params[1]),
        },
        "detectability_mode": "bk_shared_schechter_fixed_imf_outer_profile",
        "n_iterations": int(n_iterations),
        "relaxation": float(relaxation),
        "baseline_total_initial_count": float(
            baseline_payload["model"]["total_initial_count"]["in_situ"]
            + baseline_payload["model"]["total_initial_count"]["accreted"]
        ),
        "final_total_initial_count": float(selection_stats["total_initial_count"]),
        "final_selection_fraction": float(selection_stats["selection_fraction"]),
        "final_raw_survival_fraction": float(selection_stats["raw_survival_fraction"]),
        "final_mean_detectability": float(selection_stats["mean_detectability"]),
        "total_initial_count_ratio_vs_baseline": float(
            selection_stats["total_initial_count"] / max(
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
        "final_model": {
            **asdict(final_payload["summary"]),
            "raw_survival_fraction_total": float(selection_stats["raw_survival_fraction"]),
            "selection_fraction_total": float(selection_stats["selection_fraction"]),
        },
    }
    return {
        "spec": spec,
        "working": working,
        "subsets": subsets,
        "base_context": base_context,
        "observable_context": observable_context,
        "baseline_payload": baseline_payload,
        "final_payload": final_payload,
        "final_contexts": final_contexts,
        "iteration_history_table": iteration_history_table,
        "completeness_grid_table": completeness_grid_table,
        "observable_histogram_table": observable_histogram_table,
        "catalog_completeness_table": catalog_completeness_table,
        "final_completeness_raw_parameters": np.asarray(current_raw_params, dtype=float),
        "final_completeness_bin_grid": final_completeness_bin_grid,
        "final_effective_completeness_grid": final_effective_completeness_grid,
        "final_predicted_complete_counts": final_predicted_complete_counts,
        "final_predicted_observed_counts": final_predicted_observed_counts,
        "summary_payload": summary_payload,
    }


def _component_above_statistics(*, contexts: dict[str, object], model: dict[str, object], log_mass_min: float) -> dict[str, dict[str, float]]:
    component_stats: dict[str, dict[str, float]] = {}
    for component_label in ("in_situ", "accreted"):
        context = contexts[component_label]
        total_initial_count = float(model["total_initial_count"][component_label])
        imf_density_grid = np.asarray(model["imf_density_grid"], dtype=float)
        radial_density_grid = np.asarray(model["radial_density_grid"][component_label], dtype=float)
        count_above, mass_above = _count_and_mass_above_log_mass(
            total_initial_count=total_initial_count,
            log_mass_grid=np.asarray(context.log_mass_grid, dtype=float),
            imf_density_grid=imf_density_grid,
            log_mass_min=log_mass_min,
        )
        selection_stats = _selection_stats_above_log_mass(
            context=context,
            imf_density_grid=imf_density_grid,
            radial_density_grid=radial_density_grid,
            log_mass_min=log_mass_min,
        )
        component_stats[component_label] = {
            "count_above": float(count_above),
            "mass_above": float(mass_above),
            **selection_stats,
        }
    return component_stats


def _aggregate_component_above_statistics(component_stats: dict[str, dict[str, float]]) -> dict[str, float]:
    total_count_above = sum(values["count_above"] for values in component_stats.values())
    total_mass_above = sum(values["mass_above"] for values in component_stats.values())
    total_raw_surviving_above = sum(values["raw_survival_fraction_above_log10_4"] * values["count_above"] for values in component_stats.values())
    total_selected_above = sum(values["selection_fraction_above_log10_4"] * values["count_above"] for values in component_stats.values())
    return {
        "count_above": float(total_count_above),
        "mass_above": float(total_mass_above),
        "raw_survival_fraction_above_log10_4": float(total_raw_surviving_above / max(total_count_above, 1.0e-12)),
        "selection_fraction_above_log10_4": float(total_selected_above / max(total_count_above, 1.0e-12)),
        "mean_detectability_above_log10_4": float(total_selected_above / max(total_raw_surviving_above, 1.0e-12)),
    }


def _row_from_two_component_result(
    *,
    eta_t: float,
    spec,
    survival_summary,
    result: dict[str, object],
    log_mass_min: float,
) -> dict[str, object]:
    from globular_clusters_imf.detectability_model import build_shared_two_component_detectability_summary_row

    row = build_shared_two_component_detectability_summary_row(result)
    baseline_component_stats = _component_above_statistics(
        contexts=result["component_base_contexts"],
        model=result["baseline_payload"]["model"],
        log_mass_min=log_mass_min,
    )
    final_component_stats = _component_above_statistics(
        contexts=result["final_contexts"],
        model=result["final_payload"]["model"],
        log_mass_min=log_mass_min,
    )
    baseline_total_above = _aggregate_component_above_statistics(baseline_component_stats)
    final_total_above = _aggregate_component_above_statistics(final_component_stats)
    row.update(
        {
            "eta_t": float(eta_t),
            "radial_model": f"{spec.in_situ_radial_model}+{spec.accreted_radial_model}",
            "in_situ_radial_model": spec.in_situ_radial_model,
            "accreted_radial_model": spec.accreted_radial_model,
            "alpha_dndm": float(result["final_payload"]["model"]["imf_parameters"]["alpha_dndm"]),
            "log10_m_c_msun": float(result["final_payload"]["model"]["imf_parameters"]["log10_m_c_msun"]),
            "input_alpha_dndm": float(result["final_payload"]["model"]["imf_parameters"]["alpha_dndm"]),
            "input_log10_m_c_msun": float(result["final_payload"]["model"]["imf_parameters"]["log10_m_c_msun"]),
            "baseline_total_initial_count": float(
                result["baseline_payload"]["model"]["total_initial_count"]["in_situ"]
                + result["baseline_payload"]["model"]["total_initial_count"]["accreted"]
            ),
            "baseline_total_initial_count_above_log10_4": float(baseline_total_above["count_above"]),
            "baseline_total_initial_stellar_mass_above_log10_4_msun": float(baseline_total_above["mass_above"]),
            "final_total_initial_count": float(row["total_initial_count"]),
            "final_total_initial_count_above_log10_4": float(final_total_above["count_above"]),
            "final_total_initial_stellar_mass_above_log10_4_msun": float(final_total_above["mass_above"]),
            "final_total_initial_count_above_log10_4_in_situ": float(final_component_stats["in_situ"]["count_above"]),
            "final_total_initial_count_above_log10_4_accreted": float(final_component_stats["accreted"]["count_above"]),
            "final_total_initial_stellar_mass_above_log10_4_msun_in_situ": float(final_component_stats["in_situ"]["mass_above"]),
            "final_total_initial_stellar_mass_above_log10_4_msun_accreted": float(final_component_stats["accreted"]["mass_above"]),
            "raw_survival_fraction_above_log10_4": float(final_total_above["raw_survival_fraction_above_log10_4"]),
            "selection_fraction_above_log10_4": float(final_total_above["selection_fraction_above_log10_4"]),
            "mean_detectability_above_log10_4": float(final_total_above["mean_detectability_above_log10_4"]),
            "count_ratio_vs_baseline_above_log10_4": float(final_total_above["count_above"] / max(baseline_total_above["count_above"], 1.0e-12)),
            "mass_ratio_vs_baseline_above_log10_4": float(final_total_above["mass_above"] / max(baseline_total_above["mass_above"], 1.0e-12)),
            "survival_outer_level_90_log10_msun": float(getattr(survival_summary, "outer_level_90_log10_msun")),
            "survival_outer_level_50_log10_msun": float(getattr(survival_summary, "outer_level_50_log10_msun")),
            "survival_outer_level_10_log10_msun": float(getattr(survival_summary, "outer_level_10_log10_msun")),
            "survival_inner_level_90_log10_msun": float(getattr(survival_summary, "inner_level_90_log10_msun")),
            "survival_inner_level_50_log10_msun": float(getattr(survival_summary, "inner_level_50_log10_msun")),
            "survival_inner_level_10_log10_msun": float(getattr(survival_summary, "inner_level_10_log10_msun")),
            "survival_transition_a_kpc": float(getattr(survival_summary, "transition_a_kpc")),
            "survival_width_log10_a_dex": float(getattr(survival_summary, "width_log10_a_dex")),
            "survival_transition_band_width_dex": float(getattr(survival_summary, "transition_band_width_dex")),
            "surface_model": SURFACE_MODEL,
            "status": "ok",
            "failure_message": "",
        }
    )
    return row


def _start_state_from_result(result: dict[str, object]) -> dict[str, object]:
    return {
        "completeness": np.asarray(result["final_completeness_raw_parameters"], dtype=float),
        "radial": {
            "in_situ": np.asarray(result["final_payload"]["radial_parameters_raw"]["in_situ"], dtype=float),
            "accreted": np.asarray(result["final_payload"]["radial_parameters_raw"]["accreted"], dtype=float),
        },
    }


def _failure_row(theta: np.ndarray, stage: str, message: str, spec) -> dict[str, object]:
    eta_t, alpha, log_mc = [float(value) for value in theta]
    return {
        "eta_t": eta_t,
        "radial_model": f"{spec.in_situ_radial_model}+{spec.accreted_radial_model}",
        "in_situ_radial_model": spec.in_situ_radial_model,
        "accreted_radial_model": spec.accreted_radial_model,
        "log_likelihood": -np.inf,
        "aic": np.inf,
        "bic": np.inf,
        "alpha_dndm": alpha,
        "log10_m_c_msun": log_mc,
        "input_alpha_dndm": alpha,
        "input_log10_m_c_msun": log_mc,
        "baseline_total_initial_count": np.nan,
        "baseline_total_initial_count_above_log10_4": np.nan,
        "baseline_total_initial_stellar_mass_above_log10_4_msun": np.nan,
        "final_total_initial_count": np.nan,
        "final_total_initial_count_above_log10_4": np.nan,
        "final_total_initial_stellar_mass_above_log10_4_msun": np.nan,
        "raw_survival_fraction_total": np.nan,
        "selection_fraction_total": np.nan,
        "mean_detectability": np.nan,
        "raw_survival_fraction_above_log10_4": np.nan,
        "selection_fraction_above_log10_4": np.nan,
        "mean_detectability_above_log10_4": np.nan,
        "count_ratio_vs_baseline_above_log10_4": np.nan,
        "mass_ratio_vs_baseline_above_log10_4": np.nan,
        "surface_model": SURFACE_MODEL,
        "survivability_backend": "baumgardt",
        "gg23_model_name": "",
        "gg23_model_label": "",
        "gg23_mini_eta_t_dependent": False,
        "max_abs_present_mass_residual_fraction": np.nan,
        "stage": stage,
        "status": "failed",
        "failure_message": message,
    }


def _evaluate_theta_single_start(
    *,
    prepared_catalog: pd.DataFrame,
    spec,
    theta: np.ndarray,
    start_state: dict[str, object] | None,
    project_root: Path,
    n_detectability_iterations: int,
    relaxation: float,
    survivability_backend: str,
    gg23_model_name: str | None,
) -> dict[str, object]:
    eta_t, alpha, log_mc = [float(value) for value in theta]
    working_catalog, smooth_survival, metadata = _catalog_and_survival_grid_for_theta(
        prepared_catalog=prepared_catalog,
        eta_t=eta_t,
        survivability_backend=survivability_backend,
        gg23_model_name=gg23_model_name,
    )
    survival_override = _survival_grid_override_from_smooth_survival(smooth_survival)
    env = _prepare_two_component_environment_with_smooth_survival(
        prepared_catalog=working_catalog,
        survival_grid_override=survival_override,
    )
    result = _fit_shared_schechter_two_component_detectability_em_fixed_imf_with_abs_longitude(
        working=env["working"],
        subsets=env["subsets"],
        base_context=env["base_context"],
        observable_context=env["observable_context"],
        component_base_contexts=env["component_base_contexts"],
        spec=spec,
        fixed_imf_params=np.array([alpha, log_mc], dtype=float),
        n_iterations=n_detectability_iterations,
        relaxation=relaxation,
        start_completeness_raw_parameters=None if start_state is None else start_state["completeness"],
        start_radial_params_by_component=None if start_state is None else start_state["radial"],
    )
    result["component_base_contexts"] = env["component_base_contexts"]
    row = _row_from_two_component_result(
        eta_t=eta_t,
        spec=spec,
        survival_summary=smooth_survival["summary"],
        result=result,
        log_mass_min=LOG_MASS_MIN,
    )
    row.update(metadata)
    return {
        "theta": np.asarray(theta, dtype=float),
        "log_posterior": float(row["log_likelihood"]),
        "row": row,
        "result": result,
        "start_state": _start_state_from_result(result),
    }


def _evaluate_theta_multistart(
    *,
    prepared_catalog: pd.DataFrame,
    spec,
    theta: np.ndarray,
    stage: str,
    project_root: Path,
    anchor_start_state: dict[str, object] | None,
    n_detectability_iterations: int,
    relaxation: float,
    survivability_backend: str,
    gg23_model_name: str | None,
) -> dict[str, object]:
    start_candidates = [None]
    if anchor_start_state is not None:
        start_candidates.append(anchor_start_state)

    best_entry = None
    best_logp = -np.inf
    failure_messages: list[str] = []
    for start_state in start_candidates:
        try:
            entry = _evaluate_theta_single_start(
                prepared_catalog=prepared_catalog,
                spec=spec,
                theta=theta,
                start_state=start_state,
                project_root=project_root,
                n_detectability_iterations=n_detectability_iterations,
                relaxation=relaxation,
                survivability_backend=survivability_backend,
                gg23_model_name=gg23_model_name,
            )
        except Exception as exc:
            failure_messages.append(type(exc).__name__ + ": " + str(exc))
            continue
        logp = float(entry["log_posterior"])
        if logp > best_logp:
            best_logp = logp
            best_entry = entry

    if best_entry is None:
        return {
            "theta": np.asarray(theta, dtype=float),
            "log_posterior": -np.inf,
            "row": _failure_row(theta, stage=stage, message=" | ".join(failure_messages), spec=spec),
            "result": None,
            "start_state": None,
        }
    best_entry["row"]["stage"] = stage
    return best_entry


def _within_bounds(theta: np.ndarray, bounds: np.ndarray) -> bool:
    theta = np.asarray(theta, dtype=float)
    return bool(np.all(theta >= bounds[:, 0]) and np.all(theta <= bounds[:, 1]))


def _lightweight_entry(entry: dict[str, object]) -> dict[str, object]:
    start_state = entry.get("start_state")
    if start_state is None:
        start_state_copy = None
    else:
        radial_state = start_state.get("radial")
        if radial_state is None:
            radial_state_copy = None
        elif isinstance(radial_state, dict):
            radial_state_copy = {
                str(component): np.asarray(values, dtype=float).copy()
                for component, values in radial_state.items()
            }
        else:
            radial_state_copy = np.asarray(radial_state, dtype=float).copy()
        start_state_copy = {
            "completeness": np.asarray(start_state["completeness"], dtype=float).copy(),
            "radial": radial_state_copy,
        }
    return {
        "theta": np.asarray(entry["theta"], dtype=float).copy(),
        "log_posterior": float(entry["log_posterior"]),
        "row": dict(entry["row"]),
        "start_state": start_state_copy,
    }


def _run_exact_mcmc_chain_worker(
    *,
    chain_id: int,
    n_steps: int,
    adapt_until: int,
    adapt_every: int,
    seed: int,
    prepared_catalog: pd.DataFrame,
    spec,
    project_root: Path,
    bounds: np.ndarray,
    widths: np.ndarray,
    initial_entry: dict[str, object],
    fixed_anchor_library: list[dict[str, object]],
    n_detectability_iterations: int,
    relaxation: float,
    survivability_backend: str,
    gg23_model_name: str | None,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    proposal_scales = 0.08 * np.asarray(widths, dtype=float)
    current = _lightweight_entry(initial_entry)
    local_cache: dict[tuple[float, float, float], dict[str, object]] = {}
    local_cache[_round_key(current["theta"])] = current
    for anchor in fixed_anchor_library:
        local_cache[_round_key(np.asarray(anchor["theta"], dtype=float))] = _lightweight_entry(anchor)

    accepts = np.zeros(n_steps, dtype=bool)
    records: list[dict[str, object]] = []

    for step in range(n_steps):
        theta_prop = np.asarray(current["theta"], dtype=float) + rng.normal(scale=proposal_scales, size=3)
        accepted = False
        if _within_bounds(theta_prop, bounds):
            key = _round_key(theta_prop)
            if key in local_cache:
                proposal_entry = _entry_stage_copy(local_cache[key], stage="mcmc")
            else:
                anchor_state = _select_anchor_start_state(theta=theta_prop, anchors=fixed_anchor_library, bounds=bounds)
                proposal_exact = _evaluate_theta_multistart(
                    prepared_catalog=prepared_catalog,
                    spec=spec,
                    theta=theta_prop,
                    stage="mcmc",
                    project_root=project_root,
                    anchor_start_state=anchor_state,
                    n_detectability_iterations=n_detectability_iterations,
                    relaxation=relaxation,
                    survivability_backend=survivability_backend,
                    gg23_model_name=gg23_model_name,
                )
                proposal_entry = _lightweight_entry(proposal_exact)
                proposal_entry["row"]["stage"] = "mcmc"
                local_cache[key] = proposal_entry
            if np.isfinite(proposal_entry["log_posterior"]):
                delta = float(proposal_entry["log_posterior"]) - float(current["log_posterior"])
                if np.log(rng.uniform()) < min(0.0, delta):
                    current = proposal_entry
                    accepted = True

        accepts[step] = accepted
        row = dict(current["row"])
        row["chain"] = int(chain_id)
        row["step"] = int(step)
        row["accepted"] = bool(accepted)
        row["proposal_scale_eta_t"] = float(proposal_scales[0])
        row["proposal_scale_alpha"] = float(proposal_scales[1])
        row["proposal_scale_logmc"] = float(proposal_scales[2])
        records.append(row)

        if step + 1 <= adapt_until and (step + 1) % adapt_every == 0:
            window = float(accepts[step + 1 - adapt_every : step + 1].mean())
            if window < 0.15:
                proposal_scales *= 0.85
            elif window > 0.35:
                proposal_scales *= 1.15
            proposal_scales = np.clip(proposal_scales, 0.01 * widths, 0.30 * widths)

    best_row = max(records, key=lambda row: float(row["log_likelihood"]))
    return {
        "chain_id": int(chain_id),
        "records": records,
        "acceptance": float(accepts.mean()),
        "best_row": best_row,
        "final_row": records[-1],
        "cache_size": int(len(local_cache)),
    }


def _run_worker_mode(config_path: Path, output_path: Path) -> None:
    with config_path.open("rb") as handle:
        config = pickle.load(handle)
    result = _run_exact_mcmc_chain_worker(**config)
    with output_path.open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    best_row = result["best_row"]
    print(
        f"[worker chain={config['chain_id']}] accept={float(result['acceptance']):.3f} "
        f"best logL={float(best_row['log_likelihood']):.3f} eta_t={float(best_row['eta_t']):.3f} "
        f"alpha={float(best_row['input_alpha_dndm']):.3f} logMc={float(best_row['input_log10_m_c_msun']):.3f}"
    )


def _plot_properties_two_component(profiled_table: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.2), sharex=True)
    x = np.asarray(profiled_table["eta_t"], dtype=float)
    axes[0, 0].plot(x, np.asarray(profiled_table["input_alpha_dndm"], dtype=float), color="#1b9e77", lw=2.0)
    axes[0, 0].set_ylabel(r"$\alpha$")
    axes[0, 1].plot(x, np.asarray(profiled_table["input_log10_m_c_msun"], dtype=float), color="#d95f02", lw=2.0)
    axes[0, 1].set_ylabel(r"$\log_{10}(M_c/{\rm M}_\odot)$")
    axes[1, 0].plot(x, np.asarray(profiled_table["final_total_initial_count_above_log10_4"], dtype=float), color="#4c72b0", lw=2.0)
    axes[1, 0].set_ylabel(r"$N_0(M_{\rm ini}>10^4)$")
    axes[1, 0].set_yscale("log")
    axes[1, 1].plot(x, np.asarray(profiled_table["mean_detectability_above_log10_4"], dtype=float), color="#7570b3", lw=2.0)
    axes[1, 1].set_ylabel(r"$\langle Q \rangle_{>10^4}$")
    for ax in axes[1, :]:
        ax.set_xlabel(r"Lifetime Multiplier $\eta_t$")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root-name", default="profile_map_and_exact_mcmc_bk_shared_schechter_two_component_logistic_global")
    parser.add_argument("--in-situ-radial-model", default="logpoly3", choices=["logpoly3", "step5"])
    parser.add_argument("--accreted-radial-model", default="logpoly3", choices=["logpoly3", "step5"])
    parser.add_argument("--coarse-eta-min", type=float, default=0.6)
    parser.add_argument("--coarse-eta-max", type=float, default=2.2)
    parser.add_argument("--coarse-eta-n", type=int, default=7)
    parser.add_argument("--coarse-alpha-min", type=float, default=-1.8)
    parser.add_argument("--coarse-alpha-max", type=float, default=-0.6)
    parser.add_argument("--coarse-alpha-n", type=int, default=7)
    parser.add_argument("--coarse-logmc-min", type=float, default=6.0)
    parser.add_argument("--coarse-logmc-max", type=float, default=6.8)
    parser.add_argument("--coarse-logmc-n", type=int, default=6)
    parser.add_argument("--refine-delta-logl", type=float, default=5.0)
    parser.add_argument("--refine-min-points", type=int, default=16)
    parser.add_argument("--refine-padding-steps", type=float, default=1.0)
    parser.add_argument("--local-eta-n", type=int, default=11)
    parser.add_argument("--local-alpha-n", type=int, default=11)
    parser.add_argument("--local-logmc-n", type=int, default=9)
    parser.add_argument("--local-max-passes", type=int, default=4)
    parser.add_argument("--local-expand-steps", type=float, default=1.0)
    parser.add_argument("--anchor-k", type=int, default=18)
    parser.add_argument("--skip-mcmc", action="store_true")
    parser.add_argument("--mcmc-chains", type=int, default=6)
    parser.add_argument("--mcmc-steps", type=int, default=900)
    parser.add_argument("--mcmc-burn", type=int, default=300)
    parser.add_argument("--mcmc-thin", type=int, default=2)
    parser.add_argument("--mcmc-adapt-until", type=int, default=240)
    parser.add_argument("--mcmc-adapt-every", type=int, default=20)
    parser.add_argument("--mcmc-seed", type=int, default=20260529)
    parser.add_argument("--n-detectability-iterations", type=int, default=12)
    parser.add_argument("--detectability-relaxation", type=float, default=0.7)
    parser.add_argument("--survivability-backend", choices=["baumgardt", "gg23"], default="baumgardt")
    parser.add_argument("--gg23-model", default="")
    parser.add_argument("--chain-worker-config")
    parser.add_argument("--chain-worker-output")
    args = parser.parse_args()

    if args.chain_worker_config and args.chain_worker_output:
        _run_worker_mode(Path(args.chain_worker_config), Path(args.chain_worker_output))
        return

    output_root = PROJECT_ROOT / "variants" / args.output_root_name
    figures_dir = output_root / "outputs" / "figures"
    tables_dir = output_root / "outputs" / "tables"
    worker_dir = output_root / "outputs" / "parallel_exact_mcmc_workers"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    worker_dir.mkdir(parents=True, exist_ok=True)

    from globular_clusters_imf.model import fit_catalog_models
    from globular_clusters_imf.two_component_model import SharedImfTwoComponentSpec

    catalog = _load_catalog()
    prepared_catalog = fit_catalog_models(catalog, output_root)["catalog"]
    spec = SharedImfTwoComponentSpec(
        imf_family="schechter",
        in_situ_radial_model=str(args.in_situ_radial_model),
        accreted_radial_model=str(args.accreted_radial_model),
    )

    coarse_spec = GridSpec(
        eta_min=float(args.coarse_eta_min),
        eta_max=float(args.coarse_eta_max),
        eta_n=int(args.coarse_eta_n),
        alpha_min=float(args.coarse_alpha_min),
        alpha_max=float(args.coarse_alpha_max),
        alpha_n=int(args.coarse_alpha_n),
        logmc_min=float(args.coarse_logmc_min),
        logmc_max=float(args.coarse_logmc_max),
        logmc_n=int(args.coarse_logmc_n),
    )
    coarse_bounds = np.array(
        [
            [coarse_spec.eta_min, coarse_spec.eta_max],
            [coarse_spec.alpha_min, coarse_spec.alpha_max],
            [coarse_spec.logmc_min, coarse_spec.logmc_max],
        ],
        dtype=float,
    )

    evaluation_cache: dict[tuple[float, float, float], dict[str, object]] = {}
    coarse_entries: list[dict[str, object]] = []
    for eta_t in coarse_spec.eta_grid():
        for log_mc in coarse_spec.logmc_grid():
            for alpha in coarse_spec.alpha_grid():
                theta = np.array([eta_t, alpha, log_mc], dtype=float)
                key = _round_key(theta)
                if key in evaluation_cache:
                    entry = _entry_stage_copy(evaluation_cache[key], stage="coarse")
                else:
                    entry = _evaluate_theta_multistart(
                        prepared_catalog=prepared_catalog,
                        spec=spec,
                        theta=theta,
                        stage="coarse",
                        project_root=output_root,
                        anchor_start_state=None,
                        n_detectability_iterations=int(args.n_detectability_iterations),
                        relaxation=float(args.detectability_relaxation),
                        survivability_backend=str(args.survivability_backend),
                        gg23_model_name=str(args.gg23_model) or None,
                    )
                    evaluation_cache[key] = entry
                coarse_entries.append(entry)
                print(
                    f"[coarse] eta_t={eta_t:.3f} alpha={alpha:.3f} logMc={log_mc:.3f} "
                    f"logL={float(entry['row']['log_likelihood']):.3f} "
                    f"N0>1e4={float(entry['row']['final_total_initial_count_above_log10_4']):.1f}"
                )

    coarse_table = pd.DataFrame([entry["row"] for entry in coarse_entries]).sort_values(
        ["eta_t", "input_alpha_dndm", "input_log10_m_c_msun"]
    ).reset_index(drop=True)
    coarse_table.to_csv(tables_dir / "coarse_grid_results.csv", index=False)
    coarse_good = coarse_table.loc[np.isfinite(coarse_table["log_likelihood"])].copy()
    coarse_best_by_eta = coarse_good.loc[coarse_good.groupby("eta_t")["log_likelihood"].idxmax()].sort_values("eta_t").reset_index(drop=True)
    coarse_best_by_eta["best_radial_model"] = f"{spec.in_situ_radial_model}+{spec.accreted_radial_model}"
    coarse_best_by_eta.to_csv(tables_dir / "coarse_profiled_eta_results.csv", index=False)
    _plot_logl_vs_multiplier(coarse_best_by_eta, figures_dir / "coarse_profiled_logl_vs_eta_t.png")
    _plot_properties_two_component(coarse_best_by_eta, figures_dir / "coarse_profiled_properties_vs_eta_t.png")

    coarse_successes = [entry for entry in coarse_entries if np.isfinite(entry["log_posterior"])]
    coarse_best_entry = max(coarse_successes, key=lambda entry: float(entry["log_posterior"]))
    _save_best_payload(coarse_best_entry, tables_dir, prefix="coarse")

    selected_coarse = _select_high_likelihood_coarse_rows(
        coarse_table,
        delta_logl=float(args.refine_delta_logl),
        min_points=int(args.refine_min_points),
    )
    selected_coarse.to_csv(tables_dir / "coarse_high_likelihood_region.csv", index=False)
    refined_spec = _build_refined_spec_from_coarse_region(
        coarse_spec,
        selected_coarse,
        local_eta_n=int(args.local_eta_n),
        local_alpha_n=int(args.local_alpha_n),
        local_logmc_n=int(args.local_logmc_n),
        padding_steps=float(args.refine_padding_steps),
    )

    refined_entries: list[dict[str, object]] = []
    refined_table = pd.DataFrame()
    refined_successes: list[dict[str, object]] = []
    refined_best_entry = coarse_best_entry
    refined_pass_summaries: list[dict[str, object]] = []

    for pass_index in range(int(args.local_max_passes)):
        refined_bounds = np.array(
            [
                [refined_spec.eta_min, refined_spec.eta_max],
                [refined_spec.alpha_min, refined_spec.alpha_max],
                [refined_spec.logmc_min, refined_spec.logmc_max],
            ],
            dtype=float,
        )
        anchor_entries = _build_anchor_library(
            coarse_successes,
            refined_successes,
            k=max(int(args.anchor_k), int(args.mcmc_chains)),
        )
        current_entries: list[dict[str, object]] = []
        for eta_t in refined_spec.eta_grid():
            for log_mc in refined_spec.logmc_grid():
                for alpha in refined_spec.alpha_grid():
                    theta = np.array([eta_t, alpha, log_mc], dtype=float)
                    key = _round_key(theta)
                    if key in evaluation_cache:
                        entry = _entry_stage_copy(evaluation_cache[key], stage=f"refined_pass_{pass_index + 1}")
                    else:
                        anchor_state = _select_anchor_start_state(theta=theta, anchors=anchor_entries, bounds=refined_bounds)
                        entry = _evaluate_theta_multistart(
                            prepared_catalog=prepared_catalog,
                            spec=spec,
                            theta=theta,
                            stage=f"refined_pass_{pass_index + 1}",
                            project_root=output_root,
                            anchor_start_state=anchor_state,
                            n_detectability_iterations=int(args.n_detectability_iterations),
                            relaxation=float(args.detectability_relaxation),
                            survivability_backend=str(args.survivability_backend),
                            gg23_model_name=str(args.gg23_model) or None,
                        )
                        evaluation_cache[key] = entry
                    current_entries.append(entry)
                    print(
                        f"[refined {pass_index + 1}] eta_t={eta_t:.3f} alpha={alpha:.3f} logMc={log_mc:.3f} "
                        f"logL={float(entry['row']['log_likelihood']):.3f} "
                        f"N0>1e4={float(entry['row']['final_total_initial_count_above_log10_4']):.1f}"
                    )

        current_table = pd.DataFrame([entry["row"] for entry in current_entries]).sort_values(
            ["eta_t", "input_alpha_dndm", "input_log10_m_c_msun"]
        ).reset_index(drop=True)
        current_successes = [entry for entry in current_entries if np.isfinite(entry["log_posterior"])]
        current_best_entry = max(current_successes, key=lambda entry: float(entry["log_posterior"]))
        edge_flags = _best_entry_on_edge(current_best_entry, refined_spec)
        refined_pass_summaries.append(
            {
                "pass_index": pass_index + 1,
                "spec": refined_spec.__dict__,
                "best_row": json.loads(pd.Series(current_best_entry["row"]).to_json()),
                "best_on_edge": edge_flags,
            }
        )
        refined_entries = current_entries
        refined_table = current_table
        refined_successes = current_successes
        refined_best_entry = current_best_entry
        if any(edge_flags.values()) and pass_index + 1 < int(args.local_max_passes):
            expanded_spec, changed = _expand_refined_spec(
                refined_spec,
                edge_flags,
                coarse_bounds,
                expand_steps=float(args.local_expand_steps),
            )
            if changed:
                refined_spec = expanded_spec
                continue
        break

    refined_table.to_csv(tables_dir / "refined_grid_results.csv", index=False)
    refined_good = refined_table.loc[np.isfinite(refined_table["log_likelihood"])].copy()
    refined_best_by_eta = refined_good.loc[refined_good.groupby("eta_t")["log_likelihood"].idxmax()].sort_values("eta_t").reset_index(drop=True)
    refined_best_by_eta["best_radial_model"] = f"{spec.in_situ_radial_model}+{spec.accreted_radial_model}"
    refined_best_by_eta.to_csv(tables_dir / "refined_profiled_eta_results.csv", index=False)
    _plot_logl_vs_multiplier(refined_best_by_eta, figures_dir / "refined_profiled_logl_vs_eta_t.png")
    _plot_properties_two_component(refined_best_by_eta, figures_dir / "refined_profiled_properties_vs_eta_t.png")
    _save_best_payload(refined_best_entry, tables_dir, prefix="refined")
    _save_best_payload(refined_best_entry, tables_dir, prefix="best")

    if bool(args.skip_mcmc):
        summary_payload = {
            "surface_model": SURFACE_MODEL,
            "model_spec": {
                "model_class": "bk_shared_schechter_two_component",
                "imf_family": spec.imf_family,
                "in_situ_radial_model": spec.in_situ_radial_model,
                "accreted_radial_model": spec.accreted_radial_model,
            },
            "survivability_backend": str(args.survivability_backend),
            "gg23_model_name": str(args.gg23_model),
            "gg23_mini_eta_t_dependent": bool(str(args.survivability_backend) == "gg23"),
            "n_detectability_iterations": int(args.n_detectability_iterations),
            "coarse_grid_spec": coarse_spec.__dict__,
            "coarse_best": json.loads(pd.Series(coarse_best_entry["row"]).to_json()),
            "refine_delta_logl": float(args.refine_delta_logl),
            "refine_min_points": int(args.refine_min_points),
            "refine_padding_steps": float(args.refine_padding_steps),
            "coarse_high_likelihood_point_count": int(len(selected_coarse)),
            "refined_grid_spec": refined_spec.__dict__,
            "refined_passes": refined_pass_summaries,
            "refined_best": json.loads(pd.Series(refined_best_entry["row"]).to_json()),
            "mcmc": {"sampler": "skipped"},
            "n_unique_profile_evaluations_parent_process": int(len(evaluation_cache)),
        }
        (tables_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2))
        print(figures_dir / "coarse_profiled_logl_vs_eta_t.png")
        print(figures_dir / "refined_profiled_logl_vs_eta_t.png")
        print(tables_dir / "summary.json")
        return

    refined_bounds = np.array(
        [
            [refined_spec.eta_min, refined_spec.eta_max],
            [refined_spec.alpha_min, refined_spec.alpha_max],
            [refined_spec.logmc_min, refined_spec.logmc_max],
        ],
        dtype=float,
    )
    fixed_anchor_library = _build_anchor_library(
        coarse_successes,
        refined_successes,
        k=max(int(args.anchor_k), int(args.mcmc_chains) * 2),
    )
    current_states = _select_diverse_entries(
        refined_successes,
        n_select=int(args.mcmc_chains),
        bounds=refined_bounds,
    )
    if len(current_states) == 0:
        raise RuntimeError("No successful refined-grid evaluations available for exact MCMC starts.")
    while len(current_states) < int(args.mcmc_chains):
        current_states.append(current_states[-1])

    widths = refined_bounds[:, 1] - refined_bounds[:, 0]
    procs = []
    for chain_id in range(int(args.mcmc_chains)):
        config = {
            "chain_id": chain_id,
            "n_steps": int(args.mcmc_steps),
            "adapt_until": int(args.mcmc_adapt_until),
            "adapt_every": int(args.mcmc_adapt_every),
            "seed": int(args.mcmc_seed) + chain_id,
            "prepared_catalog": prepared_catalog,
            "spec": spec,
            "project_root": output_root,
            "bounds": refined_bounds,
            "widths": widths,
            "initial_entry": _lightweight_entry(current_states[chain_id]),
            "fixed_anchor_library": [_lightweight_entry(entry) for entry in fixed_anchor_library],
            "n_detectability_iterations": int(args.n_detectability_iterations),
            "relaxation": float(args.detectability_relaxation),
            "survivability_backend": str(args.survivability_backend),
            "gg23_model_name": str(args.gg23_model) or None,
        }
        config_path = worker_dir / f"chain_{chain_id}_config.pkl"
        result_path = worker_dir / f"chain_{chain_id}_result.pkl"
        log_path = worker_dir / f"chain_{chain_id}.log"
        with config_path.open("wb") as handle:
            pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)
        log_handle = log_path.open("w")
        proc = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--chain-worker-config", str(config_path),
                "--chain-worker-output", str(result_path),
            ],
            cwd=str(PROJECT_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        procs.append((chain_id, proc, log_handle, result_path, log_path))

    chain_results = []
    for chain_id, proc, log_handle, result_path, log_path in procs:
        return_code = proc.wait()
        log_handle.close()
        if return_code != 0:
            log_text = log_path.read_text()
            raise RuntimeError(f"Chain worker {chain_id} failed with code {return_code}\n{log_text}")
        with result_path.open("rb") as handle:
            chain_results.append(pickle.load(handle))
        best_row = chain_results[-1]["best_row"]
        print(
            f"[parallel exact mcmc] chain={chain_id} done accept={float(chain_results[-1]['acceptance']):.3f} "
            f"best logL={float(best_row['log_likelihood']):.3f} eta_t={float(best_row['eta_t']):.3f} "
            f"alpha={float(best_row['input_alpha_dndm']):.3f} logMc={float(best_row['input_log10_m_c_msun']):.3f}"
        )

    chain_results.sort(key=lambda item: int(item["chain_id"]))
    records = [row for result in chain_results for row in result["records"]]
    chain_table = pd.DataFrame(records).sort_values(["chain", "step"]).reset_index(drop=True)
    chain_table.to_csv(tables_dir / "exact_parallel_mcmc_chain.csv", index=False)

    posterior_parts = []
    for _, frame in chain_table.loc[chain_table["step"] >= int(args.mcmc_burn)].groupby("chain"):
        posterior_parts.append(frame.iloc[:: int(args.mcmc_thin)])
    posterior_table = pd.concat(posterior_parts, ignore_index=True)
    posterior_table.to_csv(tables_dir / "exact_parallel_mcmc_posterior_samples.csv", index=False)

    summary_candidate_columns = [
        "eta_t",
        "input_alpha_dndm",
        "input_log10_m_c_msun",
        "final_total_initial_count_above_log10_4",
        "final_total_initial_stellar_mass_above_log10_4_msun",
        "final_total_initial_count_above_log10_4_in_situ",
        "final_total_initial_count_above_log10_4_accreted",
        "mean_detectability_above_log10_4",
        "log_likelihood",
    ]
    summary_rows = []
    for column in summary_candidate_columns:
        values = np.asarray(posterior_table[column], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        q16, q50, q84 = np.quantile(finite, [0.16, 0.50, 0.84])
        summary_rows.append(
            {
                "parameter": column,
                "q16": float(q16),
                "q50": float(q50),
                "q84": float(q84),
                "minus": float(q50 - q16),
                "plus": float(q84 - q50),
            }
        )
    posterior_summary = pd.DataFrame(summary_rows)
    posterior_summary.to_csv(tables_dir / "exact_parallel_posterior_summary.csv", index=False)

    rhat = {}
    for column in ["eta_t", "input_alpha_dndm", "input_log10_m_c_msun"]:
        pivot = (
            posterior_table.pivot_table(index="step", columns="chain", values=column, aggfunc="last")
            .dropna()
            .to_numpy()
            .T
        )
        if pivot.size == 0:
            continue
        rhat[column] = _compute_rhat(pivot)

    acceptance_by_chain = {str(result["chain_id"]): float(result["acceptance"]) for result in chain_results}
    best_posterior_row = posterior_table.sort_values("log_likelihood", ascending=False).iloc[0].to_dict()
    _corner_plot(posterior_table, refined_best_entry["row"], figures_dir / "exact_parallel_profiled_posterior_corner.png")
    _trace_plot(chain_table, figures_dir / "exact_parallel_profiled_posterior_traces.png", burn_in=int(args.mcmc_burn))

    best_theta = np.array(
        [
            float(best_posterior_row["eta_t"]),
            float(best_posterior_row["input_alpha_dndm"]),
            float(best_posterior_row["input_log10_m_c_msun"]),
        ],
        dtype=float,
    )
    best_anchor = _select_anchor_start_state(theta=best_theta, anchors=fixed_anchor_library, bounds=refined_bounds)
    best_entry = _evaluate_theta_multistart(
        prepared_catalog=prepared_catalog,
        spec=spec,
        theta=best_theta,
        stage="exact_parallel_mcmc_best",
        project_root=output_root,
        anchor_start_state=best_anchor,
        n_detectability_iterations=int(args.n_detectability_iterations),
        relaxation=float(args.detectability_relaxation),
        survivability_backend=str(args.survivability_backend),
        gg23_model_name=str(args.gg23_model) or None,
    )
    _save_best_payload(best_entry, tables_dir, prefix="exact_parallel_mcmc")

    summary = {
        "source_output_root_name": args.output_root_name,
        "output_root_name": args.output_root_name,
        "surface_model": SURFACE_MODEL,
        "model_spec": {
            "model_class": "bk_shared_schechter_two_component",
            "imf_family": spec.imf_family,
            "in_situ_radial_model": spec.in_situ_radial_model,
            "accreted_radial_model": spec.accreted_radial_model,
        },
        "survivability_backend": str(args.survivability_backend),
        "gg23_model_name": str(args.gg23_model),
        "gg23_mini_eta_t_dependent": bool(str(args.survivability_backend) == "gg23"),
        "sampler": "exact_profiled_random_walk_metropolis_subprocess_parallel",
        "n_detectability_iterations": int(args.n_detectability_iterations),
        "n_chains": int(args.mcmc_chains),
        "n_steps": int(args.mcmc_steps),
        "burn_in": int(args.mcmc_burn),
        "thin": int(args.mcmc_thin),
        "acceptance_by_chain": acceptance_by_chain,
        "rhat": rhat,
        "best_posterior_sample": json.loads(pd.Series(best_posterior_row).to_json()),
        "posterior_summary": posterior_summary.to_dict(orient="records"),
        "worker_cache_sizes": {str(result["chain_id"]): int(result["cache_size"]) for result in chain_results},
        "anchor_count": int(len(fixed_anchor_library)),
        "refined_bounds": refined_bounds.tolist(),
    }
    (tables_dir / "exact_parallel_mcmc_summary.json").write_text(json.dumps(summary, indent=2))

    print(figures_dir / "exact_parallel_profiled_posterior_corner.png")
    print(figures_dir / "exact_parallel_profiled_posterior_traces.png")
    print(tables_dir / "exact_parallel_posterior_summary.csv")
    print(tables_dir / "exact_parallel_mcmc_summary.json")


if __name__ == "__main__":
    main()
