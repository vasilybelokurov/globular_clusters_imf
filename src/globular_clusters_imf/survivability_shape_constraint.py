from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import interpolate

from .detectability_longitude_model import fit_single_component_detectability_em_with_abs_longitude
from .joint_model import JointModelSpec, centers_to_edges_local, rebin_expected_counts_2d
from .smooth_survivability import fit_monotonic_soft_survivability_model


@dataclass(frozen=True)
class SurvivabilityShapeIterationSummary:
    outer_iteration: int
    log_likelihood: float
    total_initial_count: float
    total_initial_count_above_log10_4: float
    total_initial_stellar_mass_above_log10_4_msun: float
    alpha_dndm: float
    log10_m_c_msun: float
    raw_survival_fraction: float
    selection_fraction: float
    mean_detectability: float
    boundary_10_at_1kpc: float
    boundary_50_at_1kpc: float
    boundary_90_at_1kpc: float
    boundary_10_at_10kpc: float
    boundary_50_at_10kpc: float
    boundary_90_at_10kpc: float
    boundary_10_at_100kpc: float
    boundary_50_at_100kpc: float
    boundary_90_at_100kpc: float


def _expected_counts_without_survival(
    *,
    total_initial_count: float,
    imf_density_grid: np.ndarray,
    radial_density_grid: np.ndarray,
    effective_completeness_grid: np.ndarray,
) -> np.ndarray:
    return (
        float(total_initial_count)
        * np.asarray(imf_density_grid, dtype=float)[:, None]
        * np.asarray(radial_density_grid, dtype=float)[None, :]
        * np.asarray(effective_completeness_grid, dtype=float)
    )


def _build_shape_ratio_coarse_grid(
    *,
    catalog: pd.DataFrame,
    model: dict[str, object],
    context,
    effective_completeness_grid: np.ndarray,
    n_mass_bins: int = 12,
    n_a_bins: int = 8,
) -> dict[str, np.ndarray]:
    log_mass_data = np.asarray(catalog["log_initial_mass_msun"], dtype=float)
    log_a_data = np.log10(np.asarray(catalog["semi_major_axis_kpc"], dtype=float))

    log_mass_edges = np.linspace(float(log_mass_data.min()), float(log_mass_data.max()), n_mass_bins + 1)
    log_a_edges = np.quantile(log_a_data, np.linspace(0.0, 1.0, n_a_bins + 1))
    log_a_edges[0] -= 1.0e-6
    log_a_edges[-1] += 1.0e-6

    observed_counts, _, _ = np.histogram2d(
        log_mass_data,
        log_a_data,
        bins=(log_mass_edges, log_a_edges),
    )
    expected_counts = rebin_expected_counts_2d(
        _expected_counts_without_survival(
            total_initial_count=float(model["total_initial_count"]),
            imf_density_grid=np.asarray(model["imf_density_grid"], dtype=float),
            radial_density_grid=np.asarray(model["radial_density_grid"], dtype=float),
            effective_completeness_grid=np.asarray(effective_completeness_grid, dtype=float),
        ),
        np.asarray(context.log_mass_grid, dtype=float),
        np.asarray(context.log_a_grid, dtype=float),
        log_mass_edges,
        log_a_edges,
    )

    observed_shape = observed_counts / np.clip(observed_counts.sum(axis=0, keepdims=True), 1.0e-12, None)
    expected_shape = expected_counts / np.clip(expected_counts.sum(axis=0, keepdims=True), 1.0e-12, None)
    ratio = observed_shape / np.clip(expected_shape, 1.0e-8, None)

    for col in range(ratio.shape[1]):
        valid = expected_shape[:, col] > 1.0e-4
        if np.count_nonzero(valid) == 0:
            ratio[:, col] = 1.0
            continue
        upper_start = max(int(0.6 * ratio.shape[0]), ratio.shape[0] - 3)
        anchor_mask = valid & (np.arange(ratio.shape[0]) >= upper_start)
        if np.count_nonzero(anchor_mask) == 0:
            anchor_mask = valid
        anchor = float(np.nanpercentile(ratio[anchor_mask, col], 90.0))
        if not np.isfinite(anchor) or anchor <= 0.0:
            anchor = float(np.nanmax(ratio[valid, col]))
        ratio[:, col] = ratio[:, col] / max(anchor, 1.0e-3)

    ratio = np.clip(ratio, 0.0, 1.0)
    ratio = np.maximum.accumulate(ratio, axis=0)
    return {
        "log_mass_edges": log_mass_edges,
        "log_a_edges": log_a_edges,
        "log_mass_centers": 0.5 * (log_mass_edges[:-1] + log_mass_edges[1:]),
        "log_a_centers": 0.5 * (log_a_edges[:-1] + log_a_edges[1:]),
        "observed_counts": observed_counts,
        "expected_counts_without_survival": expected_counts,
        "shape_ratio": ratio,
    }


def _smooth_axis(values: np.ndarray, axis: int, kernel: np.ndarray) -> np.ndarray:
    kernel = np.asarray(kernel, dtype=float)
    kernel = kernel / np.sum(kernel)
    padded = np.pad(
        np.asarray(values, dtype=float),
        [(1, 1) if dim == axis else (0, 0) for dim in range(values.ndim)],
        mode="edge",
    )
    moved = np.moveaxis(padded, axis, 0)
    smoothed = np.empty_like(np.moveaxis(values, axis, 0), dtype=float)
    for idx in range(smoothed.shape[0]):
        smoothed[idx] = (
            kernel[0] * moved[idx]
            + kernel[1] * moved[idx + 1]
            + kernel[2] * moved[idx + 2]
        )
    return np.moveaxis(smoothed, 0, axis)


def _interpolate_shape_ratio_to_full_grid(
    *,
    coarse_payload: dict[str, np.ndarray],
    full_log_mass_grid: np.ndarray,
    full_log_a_grid: np.ndarray,
) -> np.ndarray:
    interpolator = interpolate.RegularGridInterpolator(
        (
            np.asarray(coarse_payload["log_mass_centers"], dtype=float),
            np.asarray(coarse_payload["log_a_centers"], dtype=float),
        ),
        np.asarray(coarse_payload["shape_ratio"], dtype=float),
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    mesh_mass, mesh_a = np.meshgrid(full_log_mass_grid, full_log_a_grid, indexing="ij")
    full_ratio = interpolator(np.column_stack([mesh_mass.ravel(), mesh_a.ravel()])).reshape(mesh_mass.shape)
    full_ratio = np.clip(full_ratio, 0.0, 1.0)
    full_ratio = _smooth_axis(full_ratio, axis=1, kernel=np.array([0.25, 0.5, 0.25]))
    full_ratio = np.maximum.accumulate(full_ratio, axis=0)
    return np.clip(full_ratio, 0.0, 1.0)


def _blend_survival_probability(
    *,
    old_probability: np.ndarray,
    target_probability: np.ndarray,
    relaxation: float,
) -> np.ndarray:
    blended = (1.0 - float(relaxation)) * np.asarray(old_probability, dtype=float) + float(relaxation) * np.asarray(
        target_probability,
        dtype=float,
    )
    blended = np.maximum.accumulate(blended, axis=0)
    return np.clip(blended, 1.0e-12, 1.0)


def estimate_shape_constrained_survival_grid(
    *,
    catalog: pd.DataFrame,
    fit_result: dict[str, object],
    relaxation: float = 0.7,
    n_mass_bins: int = 12,
    n_a_bins: int = 8,
) -> dict[str, object]:
    base_context = fit_result["base_context"]
    model = fit_result["final_payload"]["model"]
    coarse_payload = _build_shape_ratio_coarse_grid(
        catalog=catalog,
        model=model,
        context=base_context,
        effective_completeness_grid=np.asarray(fit_result["final_effective_completeness_grid"], dtype=float),
        n_mass_bins=n_mass_bins,
        n_a_bins=n_a_bins,
    )
    full_ratio = _interpolate_shape_ratio_to_full_grid(
        coarse_payload=coarse_payload,
        full_log_mass_grid=np.asarray(base_context.log_mass_grid, dtype=float),
        full_log_a_grid=np.asarray(base_context.log_a_grid, dtype=float),
    )
    blended_raw = _blend_survival_probability(
        old_probability=np.asarray(base_context.survival_probability_grid, dtype=float),
        target_probability=full_ratio,
        relaxation=relaxation,
    )
    smooth_fit = fit_monotonic_soft_survivability_model(
        catalog=catalog,
        log_mass_grid=np.asarray(base_context.log_mass_grid, dtype=float),
        log_a_grid=np.asarray(base_context.log_a_grid, dtype=float),
        survival_probability_raw=blended_raw,
        bandwidth_log10_a_dex=0.18,
    )
    return {
        "log_mass_grid": np.asarray(base_context.log_mass_grid, dtype=float),
        "log_a_grid": np.asarray(base_context.log_a_grid, dtype=float),
        "semi_major_axis_grid_kpc": np.power(10.0, np.asarray(base_context.log_a_grid, dtype=float)),
        "survival_probability": np.asarray(smooth_fit["fitted_probability"], dtype=float),
        "raw_shape_ratio_full_grid": full_ratio,
        "blended_raw_probability": blended_raw,
        "coarse_shape_payload": coarse_payload,
        "fit_payload": smooth_fit,
        "bandwidth_log10_a_dex": 0.18,
    }


def _boundary_value_at_a(boundary: np.ndarray, log_a_grid: np.ndarray, a_kpc: float) -> float:
    return float(np.interp(np.log10(a_kpc), np.asarray(log_a_grid, dtype=float), np.asarray(boundary, dtype=float)))


def _total_initial_stellar_mass_above_log_mass(
    *,
    total_initial_count: float,
    log_mass_grid: np.ndarray,
    imf_density_grid: np.ndarray,
    log_mass_min: float,
) -> float:
    support = np.asarray(log_mass_grid, dtype=float)
    density = np.asarray(imf_density_grid, dtype=float)
    mask = support >= float(log_mass_min)
    if not np.any(mask):
        return 0.0
    x = support[mask]
    y = density[mask]
    number_fraction = float(np.trapezoid(y, x))
    mean_mass = float(np.trapezoid(np.power(10.0, x) * y, x) / max(number_fraction, 1.0e-12))
    return float(total_initial_count * number_fraction * mean_mass)


def _build_iteration_summary(
    *,
    outer_iteration: int,
    fit_result: dict[str, object],
    survival_grid: dict[str, object],
) -> SurvivabilityShapeIterationSummary:
    final_model = fit_result["final_payload"]["model"]
    imf_parameters = final_model["imf_parameters"]
    fit_summary = fit_result["summary_payload"]
    fitted_10 = np.asarray(survival_grid["fit_payload"]["fitted_boundary_10_log10_msun"], dtype=float)
    fitted_50 = np.asarray(survival_grid["fit_payload"]["fitted_boundary_50_log10_msun"], dtype=float)
    fitted_90 = np.asarray(survival_grid["fit_payload"]["fitted_boundary_90_log10_msun"], dtype=float)
    log_a_grid = np.asarray(survival_grid["log_a_grid"], dtype=float)
    return SurvivabilityShapeIterationSummary(
        outer_iteration=int(outer_iteration),
        log_likelihood=float(fit_result["final_payload"]["summary"].log_likelihood),
        total_initial_count=float(final_model["total_initial_count"]),
        total_initial_count_above_log10_4=float(fit_summary["final_total_initial_count_above_log10_4"]),
        total_initial_stellar_mass_above_log10_4_msun=float(
            _total_initial_stellar_mass_above_log_mass(
                total_initial_count=float(final_model["total_initial_count"]),
                log_mass_grid=np.asarray(fit_result["base_context"].log_mass_grid, dtype=float),
                imf_density_grid=np.asarray(final_model["imf_density_grid"], dtype=float),
                log_mass_min=4.0,
            )
        ),
        alpha_dndm=float(imf_parameters.get("alpha_dndm", np.nan)),
        log10_m_c_msun=float(imf_parameters.get("log10_m_c_msun", np.nan)),
        raw_survival_fraction=float(final_model["raw_survival_fraction"]),
        selection_fraction=float(final_model["selection_fraction"]),
        mean_detectability=float(fit_summary["final_mean_detectability"]),
        boundary_10_at_1kpc=_boundary_value_at_a(fitted_10, log_a_grid, 1.0),
        boundary_50_at_1kpc=_boundary_value_at_a(fitted_50, log_a_grid, 1.0),
        boundary_90_at_1kpc=_boundary_value_at_a(fitted_90, log_a_grid, 1.0),
        boundary_10_at_10kpc=_boundary_value_at_a(fitted_10, log_a_grid, 10.0),
        boundary_50_at_10kpc=_boundary_value_at_a(fitted_50, log_a_grid, 10.0),
        boundary_90_at_10kpc=_boundary_value_at_a(fitted_90, log_a_grid, 10.0),
        boundary_10_at_100kpc=_boundary_value_at_a(fitted_10, log_a_grid, 100.0),
        boundary_50_at_100kpc=_boundary_value_at_a(fitted_50, log_a_grid, 100.0),
        boundary_90_at_100kpc=_boundary_value_at_a(fitted_90, log_a_grid, 100.0),
    )


def run_survivability_shape_constraint_experiment(
    *,
    catalog: pd.DataFrame,
    project_root,
    initial_survival_grid: dict[str, object],
    spec: JointModelSpec | None = None,
    outer_iterations: int = 3,
    inner_iterations: int = 6,
    survival_relaxation: float = 0.7,
    detectability_relaxation: float = 0.7,
    n_mass_bins: int = 12,
    n_a_bins: int = 8,
) -> dict[str, object]:
    if spec is None:
        spec = JointModelSpec(imf_family="schechter", radial_model="logpoly3")

    current_survival_grid = {
        key: (value.copy() if isinstance(value, np.ndarray) else value)
        for key, value in initial_survival_grid.items()
    }
    fit_results: list[dict[str, object]] = []
    survival_grids: list[dict[str, object]] = [current_survival_grid]
    summaries: list[SurvivabilityShapeIterationSummary] = []

    for outer_iteration in range(1, outer_iterations + 1):
        fit_result = fit_single_component_detectability_em_with_abs_longitude(
            catalog=catalog,
            project_root=project_root,
            spec=spec,
            n_iterations=inner_iterations,
            relaxation=detectability_relaxation,
            survival_grid_override=current_survival_grid,
        )
        updated_survival_grid = estimate_shape_constrained_survival_grid(
            catalog=catalog,
            fit_result=fit_result,
            relaxation=survival_relaxation,
            n_mass_bins=n_mass_bins,
            n_a_bins=n_a_bins,
        )
        fit_results.append(fit_result)
        survival_grids.append(updated_survival_grid)
        summaries.append(
            _build_iteration_summary(
                outer_iteration=outer_iteration,
                fit_result=fit_result,
                survival_grid=updated_survival_grid,
            )
        )
        current_survival_grid = updated_survival_grid

    final_refit_result = fit_single_component_detectability_em_with_abs_longitude(
        catalog=catalog,
        project_root=project_root,
        spec=spec,
        n_iterations=inner_iterations,
        relaxation=detectability_relaxation,
        survival_grid_override=current_survival_grid,
    )

    return {
        "spec": spec,
        "initial_survival_grid": initial_survival_grid,
        "fit_results": fit_results,
        "survival_grids": survival_grids,
        "iteration_summary_table": pd.DataFrame([summary.__dict__ for summary in summaries]),
        "final_fit_result": final_refit_result,
        "final_survival_grid": survival_grids[-1],
    }
