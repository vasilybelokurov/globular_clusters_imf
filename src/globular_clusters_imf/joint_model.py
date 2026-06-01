from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate, optimize, stats

TINY = 1.0e-300


@dataclass(frozen=True)
class JointModelSpec:
    imf_family: str
    radial_model: str


@dataclass
class JointFitResult:
    imf_family: str
    radial_model: str
    success: bool
    log_likelihood: float
    aic: float
    bic: float
    n_parameters: int
    total_initial_count: float
    survival_fraction: float
    optimizer_message: str
    imf_parameters_json: str
    radial_parameters_json: str


@dataclass
class JointUncertaintySummary:
    n_display_parameters: int
    n_posterior_like_samples: int
    covariance_condition_number: float


def fit_fixed_survival_joint_models(
    catalog: pd.DataFrame,
    project_root: Path,
    model_specs: list[JointModelSpec] | None = None,
    survival_grid_override: dict[str, object] | None = None,
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

    working = catalog.copy()
    required_columns = {"log_initial_mass_msun", "semi_major_axis_kpc", "log_survival_mass_cut_msun"}
    missing = required_columns.difference(working.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Catalog is missing required columns for joint fitting: {missing_list}")

    if survival_grid_override is None:
        selection_offset_dex = calibrate_fixed_selection_offset_dex(working)
        survival_grid = build_fixed_survival_grid(
            working,
            selection_offset_dex=selection_offset_dex,
        )
    else:
        selection_offset_dex = float(survival_grid_override.get("selection_offset_dex", np.nan))
        survival_grid = survival_grid_override
    context = JointLikelihoodContext.from_catalog_and_survival_grid(working, survival_grid)

    fit_results: list[JointFitResult] = []
    imf_grid_rows: list[dict[str, object]] = []
    radial_grid_rows: list[dict[str, object]] = []
    catalog_rows: list[dict[str, object]] = []
    all_payloads: list[dict[str, object]] = []

    best_payload: dict[str, object] | None = None
    for spec in model_specs:
        payload = fit_single_joint_model(context=context, spec=spec)
        all_payloads.append(payload)
        fit_results.append(payload["summary"])
        imf_grid_rows.extend(payload["imf_grid_rows"])
        radial_grid_rows.extend(payload["radial_grid_rows"])
        if best_payload is None or payload["summary"].log_likelihood > best_payload["summary"].log_likelihood:
            best_payload = payload

    if best_payload is None:
        raise RuntimeError("No joint model fits were produced.")

    uncertainty_payload = estimate_best_model_uncertainty(
        best_payload=best_payload,
        context=context,
    )
    best_model = best_payload["model"]
    point_intensity = compute_observed_intensity_grid(
        best_model["imf_density_grid"],
        best_model["radial_density_grid"],
        context.survival_probability_grid,
        best_model["total_initial_count"],
    )
    best_data_survival = np.clip(
        context.survival_interpolator(np.column_stack([context.log_mass_data, context.log_a_data])),
        1.0e-12,
        1.0,
    )
    best_data_imf = np.clip(best_model["imf_density_data"], 1.0e-12, None)
    best_data_radial = np.clip(best_model["radial_density_data"], 1.0e-12, None)
    best_data_intensity = best_model["total_initial_count"] * best_data_imf * best_data_radial * best_data_survival

    for index in range(len(context.log_mass_data)):
        catalog_rows.append(
            {
                "cluster_label": working.iloc[index].get("cluster_label", index),
                "log_initial_mass_msun": float(context.log_mass_data[index]),
                "semi_major_axis_kpc": float(np.power(10.0, context.log_a_data[index])),
                "log10_semi_major_axis_kpc": float(context.log_a_data[index]),
                "survival_probability_fixed": float(best_data_survival[index]),
                "imf_density_best": float(best_data_imf[index]),
                "radial_density_best_per_dex_a": float(best_data_radial[index]),
                "observed_intensity_best": float(best_data_intensity[index]),
                "best_imf_family": best_model["spec"].imf_family,
                "best_radial_model": best_model["spec"].radial_model,
            }
        )

    outputs_tables = project_root / "outputs" / "tables"
    outputs_tables.mkdir(parents=True, exist_ok=True)

    summary_table = pd.DataFrame([asdict(result) for result in fit_results]).sort_values(
        "log_likelihood",
        ascending=False,
    )
    performance_summary_table, residual_map_table = build_model_performance_tables(
        all_payloads=all_payloads,
        context=context,
    )
    imf_grid_table = pd.DataFrame(imf_grid_rows)
    radial_grid_table = pd.DataFrame(radial_grid_rows)
    catalog_prediction_table = pd.DataFrame(catalog_rows)
    best_sample_table = uncertainty_payload["display_sample_table"]
    parameter_covariance_table = uncertainty_payload["parameter_covariance_table"]
    summary_table.to_csv(outputs_tables / "joint_fixed_survival_model_summary.csv", index=False)
    performance_summary_table.to_csv(
        outputs_tables / "joint_fixed_survival_model_performance_summary.csv",
        index=False,
    )
    residual_map_table.to_csv(
        outputs_tables / "joint_fixed_survival_model_performance_residual_maps.csv",
        index=False,
    )
    imf_grid_table.to_csv(outputs_tables / "joint_fixed_survival_imf_grids.csv", index=False)
    radial_grid_table.to_csv(outputs_tables / "joint_fixed_survival_radial_grids.csv", index=False)
    catalog_prediction_table.to_csv(outputs_tables / "joint_fixed_survival_catalog_predictions.csv", index=False)
    best_sample_table.to_csv(outputs_tables / "joint_fixed_survival_best_model_triangle_samples.csv", index=False)
    parameter_covariance_table.reset_index(names="parameter_name").to_csv(
        outputs_tables / "joint_fixed_survival_best_model_parameter_covariance.csv",
        index=False,
    )

    best_summary_row = asdict(best_payload["summary"])
    detailed_summary = {
        "selection_offset_dex": selection_offset_dex,
        "survival_grid_bandwidth_log10_a_dex": survival_grid["bandwidth_log10_a_dex"],
        "best_model": best_summary_row,
        "best_model_uncertainty": asdict(uncertainty_payload["summary"]),
        "all_models_ranked": summary_table.to_dict(orient="records"),
    }
    (outputs_tables / "joint_fixed_survival_model_summary.json").write_text(
        json.dumps(detailed_summary, indent=2)
    )

    return {
        "summary_table": summary_table,
        "performance_summary_table": performance_summary_table,
        "residual_map_table": residual_map_table,
        "imf_grid_table": imf_grid_table,
        "radial_grid_table": radial_grid_table,
        "catalog_prediction_table": catalog_prediction_table,
        "best_payload": best_payload,
        "all_payloads": all_payloads,
        "best_model_uncertainty": uncertainty_payload,
        "survival_grid": survival_grid,
        "point_intensity_grid": point_intensity,
        "context": context,
    }


def calibrate_fixed_selection_offset_dex(catalog: pd.DataFrame, margin_dex: float = 1.0e-3) -> float:
    differences = catalog["log_initial_mass_msun"].to_numpy() - catalog["log_survival_mass_cut_msun"].to_numpy()
    return float(np.nanmin(differences) - margin_dex)


def build_fixed_survival_grid(
    catalog: pd.DataFrame,
    selection_offset_dex: float,
    n_radius_grid: int = 160,
    n_mass_grid: int = 180,
    bandwidth_log10_a_dex: float = 0.18,
) -> dict[str, object]:
    log_a_data = np.log10(catalog["semi_major_axis_kpc"].to_numpy())
    effective_log_cut_data = catalog["log_survival_mass_cut_msun"].to_numpy() + selection_offset_dex

    log_a_grid = np.linspace(log_a_data.min(), log_a_data.max(), n_radius_grid)
    log_mass_min = min(3.5, float(np.floor(catalog["log_initial_mass_msun"].min() * 10.0) / 10.0))
    log_mass_max = max(7.3, float(np.ceil(catalog["log_initial_mass_msun"].max() * 10.0) / 10.0))
    log_mass_grid = np.linspace(log_mass_min, log_mass_max, n_mass_grid)

    weights = np.exp(
        -0.5 * np.square((log_a_grid[:, None] - log_a_data[None, :]) / bandwidth_log10_a_dex)
    )
    weights /= np.clip(weights.sum(axis=1, keepdims=True), 1.0e-12, None)

    survival_probability = (log_mass_grid[:, None] >= effective_log_cut_data[None, :]) @ weights.T
    return {
        "log_mass_grid": log_mass_grid,
        "log_a_grid": log_a_grid,
        "semi_major_axis_grid_kpc": np.power(10.0, log_a_grid),
        "survival_probability": survival_probability,
        "selection_offset_dex": selection_offset_dex,
        "bandwidth_log10_a_dex": bandwidth_log10_a_dex,
    }


@dataclass
class JointLikelihoodContext:
    log_mass_data: np.ndarray
    log_a_data: np.ndarray
    log_mass_grid: np.ndarray
    log_a_grid: np.ndarray
    survival_probability_grid: np.ndarray
    survival_interpolator: interpolate.RegularGridInterpolator
    selection_probability_grid: np.ndarray
    selection_interpolator: interpolate.RegularGridInterpolator
    radial_step_edges: np.ndarray
    log_a_mean: float
    log_a_std: float

    @classmethod
    def from_catalog_and_survival_grid(
        cls,
        catalog: pd.DataFrame,
        survival_grid: dict[str, object],
        n_radial_step_bins: int = 5,
    ) -> "JointLikelihoodContext":
        log_a_data = np.log10(catalog["semi_major_axis_kpc"].to_numpy())
        log_a_grid = np.asarray(survival_grid["log_a_grid"])
        log_mass_grid = np.asarray(survival_grid["log_mass_grid"])
        survival_probability_grid = np.asarray(survival_grid["survival_probability"])
        radial_step_edges = np.quantile(log_a_data, np.linspace(0.0, 1.0, n_radial_step_bins + 1))
        interpolator = interpolate.RegularGridInterpolator(
            (log_mass_grid, log_a_grid),
            survival_probability_grid,
            bounds_error=False,
            fill_value=None,
        )
        return cls(
            log_mass_data=catalog["log_initial_mass_msun"].to_numpy(),
            log_a_data=log_a_data,
            log_mass_grid=log_mass_grid,
            log_a_grid=log_a_grid,
            survival_probability_grid=survival_probability_grid,
            survival_interpolator=interpolator,
            selection_probability_grid=survival_probability_grid.copy(),
            selection_interpolator=interpolator,
            radial_step_edges=radial_step_edges,
            log_a_mean=float(log_a_data.mean()),
            log_a_std=float(log_a_data.std()),
        )

    def with_selection_probability_grid(self, selection_probability_grid: np.ndarray) -> "JointLikelihoodContext":
        selection_probability_grid = np.asarray(selection_probability_grid, dtype=float)
        if selection_probability_grid.shape != self.survival_probability_grid.shape:
            raise ValueError(
                "Selection probability grid shape does not match survival grid shape: "
                f"{selection_probability_grid.shape} versus {self.survival_probability_grid.shape}"
            )
        selection_probability_grid = np.clip(selection_probability_grid, 1.0e-12, 1.0)
        selection_interpolator = interpolate.RegularGridInterpolator(
            (self.log_mass_grid, self.log_a_grid),
            selection_probability_grid,
            bounds_error=False,
            fill_value=None,
        )
        return JointLikelihoodContext(
            log_mass_data=self.log_mass_data.copy(),
            log_a_data=self.log_a_data.copy(),
            log_mass_grid=self.log_mass_grid.copy(),
            log_a_grid=self.log_a_grid.copy(),
            survival_probability_grid=self.survival_probability_grid.copy(),
            survival_interpolator=self.survival_interpolator,
            selection_probability_grid=selection_probability_grid,
            selection_interpolator=selection_interpolator,
            radial_step_edges=self.radial_step_edges.copy(),
            log_a_mean=self.log_a_mean,
            log_a_std=self.log_a_std,
        )


def fit_single_joint_model(context: JointLikelihoodContext, spec: JointModelSpec) -> dict[str, object]:
    starts = initial_parameter_vectors(spec)
    bounds = parameter_bounds(spec)
    best_result = None
    best_value = np.inf
    for start in starts:
        result = optimize.minimize(
            lambda params: negative_profile_log_likelihood(params, context=context, spec=spec),
            x0=np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if result.fun < best_value:
            best_value = float(result.fun)
            best_result = result

    if best_result is None:
        raise RuntimeError(f"Optimization failed to start for {spec}")

    model = unpack_model(best_result.x, context=context, spec=spec)
    log_likelihood = full_log_likelihood_from_model(model, context)
    n_parameters = len(best_result.x)
    n_data = len(context.log_mass_data)
    summary = JointFitResult(
        imf_family=spec.imf_family,
        radial_model=spec.radial_model,
        success=bool(best_result.success),
        log_likelihood=float(log_likelihood),
        aic=float(2 * n_parameters - 2 * log_likelihood),
        bic=float(np.log(n_data) * n_parameters - 2 * log_likelihood),
        n_parameters=n_parameters,
        total_initial_count=float(model["total_initial_count"]),
        survival_fraction=float(model["survival_fraction"]),
        optimizer_message=str(best_result.message),
        imf_parameters_json=json.dumps(model["imf_parameters"]),
        radial_parameters_json=json.dumps(model["radial_parameters"]),
    )
    return {
        "summary": summary,
        "model": model,
        "spec": spec,
        "raw_parameters": np.asarray(best_result.x, dtype=float),
        "bounds": bounds,
        "imf_grid_rows": make_imf_grid_rows(model, context, spec),
        "radial_grid_rows": make_radial_grid_rows(model, context, spec),
    }


def fit_single_joint_model_with_fixed_imf_params(
    context: JointLikelihoodContext,
    spec: JointModelSpec,
    fixed_imf_params: np.ndarray,
    start_radial_params: np.ndarray | None = None,
) -> dict[str, object]:
    imf_param_count = imf_parameter_count(spec.imf_family)
    fixed_imf_params = np.asarray(fixed_imf_params, dtype=float)
    if len(fixed_imf_params) != imf_param_count:
        raise ValueError(
            f"Fixed IMF parameter length {len(fixed_imf_params)} does not match "
            f"family {spec.imf_family} requirement of {imf_param_count}."
        )

    full_bounds = parameter_bounds(spec)
    radial_bounds = full_bounds[imf_param_count:]
    radial_starts = [np.asarray(start[imf_param_count:], dtype=float) for start in initial_parameter_vectors(spec)]
    if start_radial_params is not None:
        radial_starts.insert(0, np.asarray(start_radial_params, dtype=float))

    best_result = None
    best_value = np.inf
    for radial_start in radial_starts:
        result = optimize.minimize(
            lambda radial_params: negative_profile_log_likelihood(
                np.concatenate([fixed_imf_params, np.asarray(radial_params, dtype=float)]),
                context=context,
                spec=spec,
            ),
            x0=np.asarray(radial_start, dtype=float),
            method="L-BFGS-B",
            bounds=radial_bounds,
        )
        if result.fun < best_value:
            best_value = float(result.fun)
            best_result = result

    if best_result is None:
        raise RuntimeError(f"Optimization failed to start for {spec} with fixed IMF parameters.")

    full_params = np.concatenate([fixed_imf_params, np.asarray(best_result.x, dtype=float)])
    model = unpack_model(full_params, context=context, spec=spec)
    log_likelihood = full_log_likelihood_from_model(model, context)
    n_parameters = imf_param_count + len(best_result.x)
    n_data = len(context.log_mass_data)
    summary = JointFitResult(
        imf_family=spec.imf_family,
        radial_model=spec.radial_model,
        success=bool(best_result.success),
        log_likelihood=float(log_likelihood),
        aic=float(2 * n_parameters - 2 * log_likelihood),
        bic=float(np.log(n_data) * n_parameters - 2 * log_likelihood),
        n_parameters=n_parameters,
        total_initial_count=float(model["total_initial_count"]),
        survival_fraction=float(model["survival_fraction"]),
        optimizer_message=str(best_result.message),
        imf_parameters_json=json.dumps(model["imf_parameters"]),
        radial_parameters_json=json.dumps(model["radial_parameters"]),
    )
    return {
        "summary": summary,
        "model": model,
        "spec": spec,
        "raw_parameters": full_params,
        "bounds": full_bounds,
        "fixed_imf_params": fixed_imf_params.copy(),
        "radial_parameters_raw": np.asarray(best_result.x, dtype=float),
        "imf_grid_rows": make_imf_grid_rows(model, context, spec),
        "radial_grid_rows": make_radial_grid_rows(model, context, spec),
    }


def negative_profile_log_likelihood(
    params: np.ndarray,
    context: JointLikelihoodContext,
    spec: JointModelSpec,
) -> float:
    model = unpack_model(params, context=context, spec=spec)
    if np.any(model["imf_density_data"] <= 0.0) or np.any(model["radial_density_data"] <= 0.0):
        return 1.0e30
    if model["survival_fraction"] <= 0.0:
        return 1.0e30
    profile_log_like = (
        np.sum(np.log(model["imf_density_data"]))
        + np.sum(np.log(model["radial_density_data"]))
        - len(context.log_mass_data) * np.log(model["survival_fraction"])
    )
    return float(-profile_log_like + imf_smoothness_penalty(params, spec=spec))


def imf_smoothness_penalty(params: np.ndarray, spec: JointModelSpec) -> float:
    if spec.imf_family != "logspline6":
        return 0.0
    node_log_amplitudes = np.concatenate([np.asarray(params[:5], dtype=float), np.array([0.0])])
    second_difference = np.diff(node_log_amplitudes, n=2)
    penalty_strength = 0.5
    return float(penalty_strength * np.sum(np.square(second_difference)))


def full_log_likelihood_from_model(model: dict[str, object], context: JointLikelihoodContext) -> float:
    selection_data = np.clip(
        context.selection_interpolator(np.column_stack([context.log_mass_data, context.log_a_data])),
        1.0e-12,
        1.0,
    )
    total_initial_count = float(model["total_initial_count"])
    return float(
        len(context.log_mass_data) * np.log(total_initial_count)
        - total_initial_count * model["survival_fraction"]
        + np.sum(np.log(np.clip(model["imf_density_data"], 1.0e-12, None)))
        + np.sum(np.log(np.clip(model["radial_density_data"], 1.0e-12, None)))
        + np.sum(np.log(selection_data))
    )


def unpack_model(params: np.ndarray, context: JointLikelihoodContext, spec: JointModelSpec) -> dict[str, object]:
    imf_param_count = imf_parameter_count(spec.imf_family)
    imf_params = np.asarray(params[:imf_param_count], dtype=float)
    radial_params = np.asarray(params[imf_param_count:], dtype=float)

    imf_density_grid, imf_density_data, imf_parameters = evaluate_imf_family(
        spec.imf_family,
        imf_params,
        context.log_mass_grid,
        context.log_mass_data,
    )
    radial_density_grid, radial_density_data, radial_parameters = evaluate_radial_model(
        spec.radial_model,
        radial_params,
        context,
    )
    raw_survival_fraction = integrate_survival_fraction(
        imf_density_grid,
        radial_density_grid,
        context.log_mass_grid,
        context.log_a_grid,
        context.survival_probability_grid,
    )
    selection_fraction = integrate_survival_fraction(
        imf_density_grid,
        radial_density_grid,
        context.log_mass_grid,
        context.log_a_grid,
        context.selection_probability_grid,
    )
    total_initial_count = len(context.log_mass_data) / selection_fraction
    return {
        "spec": spec,
        "imf_density_grid": imf_density_grid,
        "imf_density_data": imf_density_data,
        "radial_density_grid": radial_density_grid,
        "radial_density_data": radial_density_data,
        "survival_fraction": selection_fraction,
        "selection_fraction": selection_fraction,
        "raw_survival_fraction": raw_survival_fraction,
        "total_initial_count": total_initial_count,
        "imf_parameters": imf_parameters,
        "radial_parameters": radial_parameters,
    }


def imf_parameter_count(imf_family: str) -> int:
    if imf_family == "lognormal":
        return 2
    if imf_family == "powerlaw":
        return 1
    if imf_family == "powerlaw_m2":
        return 0
    if imf_family == "schechter":
        return 2
    if imf_family == "logspline6":
        return 5
    raise ValueError(f"Unknown IMF family: {imf_family}")


def evaluate_imf_family(
    imf_family: str,
    params: np.ndarray,
    log_mass_grid: np.ndarray,
    log_mass_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if imf_family == "lognormal":
        mu = float(params[0])
        sigma = float(np.exp(params[1]))
        cdf_low = stats.norm.cdf(log_mass_grid[0], loc=mu, scale=sigma)
        cdf_high = stats.norm.cdf(log_mass_grid[-1], loc=mu, scale=sigma)
        normalization = max(cdf_high - cdf_low, 1.0e-12)
        density_grid = stats.norm.pdf(log_mass_grid, loc=mu, scale=sigma) / normalization
        density_data = stats.norm.pdf(log_mass_data, loc=mu, scale=sigma) / normalization
        return density_grid, density_data, {"mu_log10_msun": mu, "sigma_log10_msun": sigma}

    if imf_family == "powerlaw":
        alpha = float(params[0])
        density_grid = np.power(10.0, (alpha + 1.0) * log_mass_grid)
        density_grid = normalize_density_on_grid(density_grid, log_mass_grid)
        density_data = interpolate_density(log_mass_grid, density_grid, log_mass_data)
        return density_grid, density_data, {"alpha_dndm": alpha}

    if imf_family == "powerlaw_m2":
        alpha = -2.0
        density_grid = np.power(10.0, (alpha + 1.0) * log_mass_grid)
        density_grid = normalize_density_on_grid(density_grid, log_mass_grid)
        density_data = interpolate_density(log_mass_grid, density_grid, log_mass_data)
        return density_grid, density_data, {"alpha_dndm": alpha}

    if imf_family == "schechter":
        alpha = float(params[0])
        log_m_c = float(params[1])
        m_c = np.power(10.0, log_m_c)
        mass_grid = np.power(10.0, log_mass_grid)
        mass_data = np.power(10.0, log_mass_data)
        density_grid = np.power(mass_grid, alpha + 1.0) * np.exp(-mass_grid / m_c)
        density_grid = normalize_density_on_grid(density_grid, log_mass_grid)
        density_data = interpolate_density(log_mass_grid, density_grid, log_mass_data)
        _ = mass_data
        return density_grid, density_data, {"alpha_dndm": alpha, "log10_m_c_msun": log_m_c}

    if imf_family == "logspline6":
        knot_positions = logspline_knot_positions(log_mass_grid, n_knots=6)
        node_log_amplitudes = np.concatenate([np.asarray(params, dtype=float), np.array([0.0])])
        node_log_amplitudes -= np.max(node_log_amplitudes)
        spline = interpolate.PchipInterpolator(knot_positions, node_log_amplitudes, extrapolate=True)
        density_grid = np.exp(np.clip(spline(log_mass_grid), -700.0, 700.0))
        density_grid = normalize_density_on_grid(density_grid, log_mass_grid)
        density_data = interpolate_density(log_mass_grid, density_grid, log_mass_data)
        return density_grid, density_data, {
            "knot_log10_msun": knot_positions.tolist(),
            "node_log_amplitudes_relative": node_log_amplitudes.tolist(),
        }

    raise ValueError(f"Unknown IMF family: {imf_family}")


def logspline_knot_positions(log_mass_grid: np.ndarray, n_knots: int = 6) -> np.ndarray:
    return np.linspace(float(log_mass_grid[0]), float(log_mass_grid[-1]), int(n_knots))


def evaluate_radial_model(
    radial_model: str,
    params: np.ndarray,
    context: JointLikelihoodContext,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if radial_model == "step5":
        weights = softmax_with_reference(params)
        bin_widths = np.diff(context.radial_step_edges)
        density_grid = piecewise_constant_density(
            context.log_a_grid,
            edges=context.radial_step_edges,
            weights=weights,
        )
        density_data = piecewise_constant_density(
            context.log_a_data,
            edges=context.radial_step_edges,
            weights=weights,
        )
        return density_grid, density_data, {
            "bin_edges_log10_a": context.radial_step_edges.tolist(),
            "bin_edges_a_kpc": np.power(10.0, context.radial_step_edges).tolist(),
            "bin_weights": weights.tolist(),
            "bin_widths_log10_a": bin_widths.tolist(),
        }

    if radial_model == "logpoly3":
        z_grid = (context.log_a_grid - context.log_a_mean) / context.log_a_std
        z_data = (context.log_a_data - context.log_a_mean) / context.log_a_std
        beta1, beta2, beta3 = params
        log_raw_grid = beta1 * z_grid + beta2 * np.square(z_grid) + beta3 * np.power(z_grid, 3.0)
        log_raw_grid -= np.max(log_raw_grid)
        raw_grid = np.exp(log_raw_grid)
        density_grid = normalize_density_on_grid(raw_grid, context.log_a_grid)
        density_data = interpolate_density(context.log_a_grid, density_grid, context.log_a_data)
        return density_grid, density_data, {
            "beta1": float(beta1),
            "beta2": float(beta2),
            "beta3": float(beta3),
            "log_a_mean": float(context.log_a_mean),
            "log_a_std": float(context.log_a_std),
        }

    if radial_model == "powerlaw_a":
        beta = float(params[0])
        raw_grid = np.power(10.0, beta * context.log_a_grid)
        density_grid = normalize_density_on_grid(raw_grid, context.log_a_grid)
        density_data = interpolate_density(context.log_a_grid, density_grid, context.log_a_data)
        return density_grid, density_data, {
            "beta_log10_a": beta,
            "gamma_linear_a": float(1.0 - beta),
        }

    if radial_model == "cored_powerlaw_a":
        beta = float(params[0])
        log_a_core = float(params[1])
        a_core = np.power(10.0, log_a_core)
        a_grid = np.power(10.0, context.log_a_grid)
        gamma = 1.0 - beta
        log_raw_grid = np.log(a_grid) - gamma * np.log(a_grid + a_core)
        log_raw_grid -= np.max(log_raw_grid)
        raw_grid = np.exp(log_raw_grid)
        density_grid = normalize_density_on_grid(raw_grid, context.log_a_grid)
        density_data = interpolate_density(context.log_a_grid, density_grid, context.log_a_data)
        return density_grid, density_data, {
            "beta_log10_a": beta,
            "gamma_linear_a": float(gamma),
            "log10_a_core_kpc": log_a_core,
            "a_core_kpc": float(a_core),
        }

    raise ValueError(f"Unknown radial model: {radial_model}")


def integrate_survival_fraction(
    imf_density_grid: np.ndarray,
    radial_density_grid: np.ndarray,
    log_mass_grid: np.ndarray,
    log_a_grid: np.ndarray,
    survival_probability_grid: np.ndarray,
) -> float:
    integrand = (
        imf_density_grid[:, None]
        * radial_density_grid[None, :]
        * survival_probability_grid
    )
    return float(np.trapezoid(np.trapezoid(integrand, log_a_grid, axis=1), log_mass_grid))


def compute_observed_intensity_grid(
    imf_density_grid: np.ndarray,
    radial_density_grid: np.ndarray,
    survival_probability_grid: np.ndarray,
    total_initial_count: float,
) -> np.ndarray:
    return total_initial_count * imf_density_grid[:, None] * radial_density_grid[None, :] * survival_probability_grid


def normalize_density_on_grid(raw_density: np.ndarray, grid: np.ndarray) -> np.ndarray:
    integral = float(np.trapezoid(raw_density, grid))
    return np.clip(raw_density / max(integral, 1.0e-12), TINY, None)


def interpolate_density(grid: np.ndarray, density: np.ndarray, values: np.ndarray) -> np.ndarray:
    interpolator = interpolate.interp1d(
        grid,
        density,
        kind="linear",
        bounds_error=False,
        fill_value=(density[0], density[-1]),
    )
    return np.clip(np.asarray(interpolator(values), dtype=float), TINY, None)


def softmax_with_reference(params: np.ndarray) -> np.ndarray:
    logits = np.concatenate([np.asarray(params, dtype=float), np.array([0.0])])
    logits -= np.max(logits)
    weights = np.exp(logits)
    return weights / np.sum(weights)


def piecewise_constant_density(values: np.ndarray, edges: np.ndarray, weights: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(edges[1:-1], values, side="right")
    widths = np.diff(edges)
    density = weights[indices] / widths[indices]
    return np.clip(density, TINY, None)


def parameter_bounds(spec: JointModelSpec) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    if spec.imf_family == "lognormal":
        bounds.extend([(4.0, 7.2), (-3.0, 1.0)])
    elif spec.imf_family == "powerlaw":
        bounds.extend([(-4.0, -0.2)])
    elif spec.imf_family == "powerlaw_m2":
        pass
    elif spec.imf_family == "schechter":
        bounds.extend([(-4.0, -0.2), (4.5, 7.5)])
    elif spec.imf_family == "logspline6":
        bounds.extend([(-8.0, 8.0)] * 5)
    else:
        raise ValueError(f"Unknown IMF family: {spec.imf_family}")

    if spec.radial_model == "step5":
        bounds.extend([(-6.0, 6.0)] * 4)
    elif spec.radial_model == "logpoly3":
        bounds.extend([(-5.0, 5.0)] * 3)
    elif spec.radial_model == "powerlaw_a":
        bounds.extend([(-5.0, 3.0)])
    elif spec.radial_model == "cored_powerlaw_a":
        bounds.extend([(-5.0, 3.0), (-1.0, 1.5)])
    else:
        raise ValueError(f"Unknown radial model: {spec.radial_model}")
    return bounds


def initial_parameter_vectors(spec: JointModelSpec) -> list[np.ndarray]:
    if spec.radial_model == "step5":
        radial_zero = np.zeros(4)
    elif spec.radial_model == "logpoly3":
        radial_zero = np.zeros(3)
    elif spec.radial_model == "powerlaw_a":
        radial_zero = np.zeros(1)
    elif spec.radial_model == "cored_powerlaw_a":
        radial_zero = np.array([0.0, 0.0])
    else:
        raise ValueError(f"Unknown radial model: {spec.radial_model}")

    if spec.imf_family == "lognormal":
        imf_starts = [
            np.array([5.85, np.log(0.60)]),
            np.array([5.60, np.log(0.75)]),
            np.array([6.05, np.log(0.50)]),
        ]
    elif spec.imf_family == "powerlaw":
        imf_starts = [np.array([-2.0]), np.array([-1.5]), np.array([-2.5])]
    elif spec.imf_family == "powerlaw_m2":
        imf_starts = [np.array([], dtype=float)]
    elif spec.imf_family == "schechter":
        imf_starts = [
            np.array([-2.0, 6.20]),
            np.array([-1.7, 6.00]),
            np.array([-2.3, 6.50]),
        ]
    elif spec.imf_family == "logspline6":
        imf_starts = [
            np.zeros(5),
            np.array([0.8, 0.4, 0.0, -0.4, -0.8]),
            np.array([1.2, 0.6, 0.0, -0.6, -1.2]),
            np.array([-0.6, -0.2, 0.0, 0.2, 0.6]),
        ]
    else:
        raise ValueError(f"Unknown IMF family: {spec.imf_family}")

    starts: list[np.ndarray] = []
    for imf_start in imf_starts:
        starts.append(np.concatenate([imf_start, radial_zero]))
        if spec.radial_model == "step5":
            starts.append(np.concatenate([imf_start, np.array([1.0, 0.5, 0.0, -0.5])]))
        elif spec.radial_model == "logpoly3":
            starts.append(np.concatenate([imf_start, np.array([-0.4, -0.8, 0.2])]))
        elif spec.radial_model == "powerlaw_a":
            starts.append(np.concatenate([imf_start, np.array([-1.0])]))
            starts.append(np.concatenate([imf_start, np.array([-2.0])]))
        elif spec.radial_model == "cored_powerlaw_a":
            starts.append(np.concatenate([imf_start, np.array([-1.0, -0.5])]))
            starts.append(np.concatenate([imf_start, np.array([-1.0, 0.5])]))
            starts.append(np.concatenate([imf_start, np.array([-2.0, 0.0])]))
        else:
            raise ValueError(f"Unknown radial model: {spec.radial_model}")
    return starts


def make_imf_grid_rows(
    model: dict[str, object],
    context: JointLikelihoodContext,
    spec: JointModelSpec,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for log_mass, density in zip(context.log_mass_grid, model["imf_density_grid"], strict=True):
        rows.append(
            {
                "imf_family": spec.imf_family,
                "radial_model": spec.radial_model,
                "log_initial_mass_msun": float(log_mass),
                "imf_density_per_dex": float(density),
            }
        )
    return rows


def make_radial_grid_rows(
    model: dict[str, object],
    context: JointLikelihoodContext,
    spec: JointModelSpec,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for log_a, density in zip(context.log_a_grid, model["radial_density_grid"], strict=True):
        rows.append(
            {
                "imf_family": spec.imf_family,
                "radial_model": spec.radial_model,
                "semi_major_axis_kpc": float(np.power(10.0, log_a)),
                "log10_semi_major_axis_kpc": float(log_a),
                "radial_density_per_dex_a": float(density),
                "birth_intensity_per_dex_a": float(model["total_initial_count"] * density),
            }
        )
    return rows


def estimate_best_model_uncertainty(
    best_payload: dict[str, object],
    context: JointLikelihoodContext,
    n_samples: int = 4000,
) -> dict[str, object]:
    spec = best_payload["spec"]
    best_params = np.asarray(best_payload["raw_parameters"], dtype=float)
    bounds = best_payload["bounds"]
    covariance = estimate_parameter_covariance(
        best_params,
        context=context,
        spec=spec,
        bounds=bounds,
    )
    raw_samples = draw_asymptotic_parameter_samples(
        center=best_params,
        covariance=covariance,
        context=context,
        spec=spec,
        bounds=bounds,
        n_samples=n_samples,
    )
    display_sample_table = build_display_parameter_sample_table(raw_samples, context=context, spec=spec)
    parameter_names = raw_parameter_names(spec)
    parameter_covariance_table = pd.DataFrame(covariance, index=parameter_names, columns=parameter_names)
    covariance_condition_number = float(np.linalg.cond(covariance))
    summary = JointUncertaintySummary(
        n_display_parameters=display_sample_table.shape[1],
        n_posterior_like_samples=len(display_sample_table),
        covariance_condition_number=covariance_condition_number,
    )
    return {
        "raw_samples": raw_samples,
        "display_sample_table": display_sample_table,
        "parameter_covariance": covariance,
        "parameter_covariance_table": parameter_covariance_table,
        "summary": summary,
    }


def estimate_parameter_covariance(
    best_params: np.ndarray,
    context: JointLikelihoodContext,
    spec: JointModelSpec,
    bounds: list[tuple[float, float]],
) -> np.ndarray:
    objective = lambda params: negative_profile_log_likelihood(params, context=context, spec=spec)
    hessian = finite_difference_hessian(objective, best_params, bounds=bounds)
    hessian = 0.5 * (hessian + hessian.T)
    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(hessian)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 1.0e-10, None)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def finite_difference_hessian(
    objective,
    center: np.ndarray,
    bounds: list[tuple[float, float]],
) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    n_dim = len(center)
    hessian = np.zeros((n_dim, n_dim), dtype=float)
    steps = 1.0e-4 * np.maximum(np.abs(center), 1.0)
    f0 = float(objective(center))

    for i in range(n_dim):
        step_i = basis_step(center, index=i, step=steps[i], bounds=bounds)
        fp = float(objective(center + step_i))
        fm = float(objective(center - step_i))
        hessian[i, i] = (fp - 2.0 * f0 + fm) / max(step_i[i] ** 2, 1.0e-12)

        for j in range(i + 1, n_dim):
            step_j = basis_step(center, index=j, step=steps[j], bounds=bounds)
            fpp = float(objective(center + step_i + step_j))
            fpm = float(objective(center + step_i - step_j))
            fmp = float(objective(center - step_i + step_j))
            fmm = float(objective(center - step_i - step_j))
            denominator = 4.0 * step_i[i] * step_j[j]
            hessian_ij = (fpp - fpm - fmp + fmm) / max(denominator, 1.0e-12)
            hessian[i, j] = hessian_ij
            hessian[j, i] = hessian_ij
    return hessian


def basis_step(center: np.ndarray, index: int, step: float, bounds: list[tuple[float, float]]) -> np.ndarray:
    lower, upper = bounds[index]
    usable_step = min(step, center[index] - lower - 1.0e-6, upper - center[index] - 1.0e-6)
    if usable_step <= 0.0:
        usable_step = min(step, max(upper - lower, 1.0e-3) * 0.1)
    vector = np.zeros_like(center, dtype=float)
    vector[index] = usable_step
    return vector


def draw_asymptotic_parameter_samples(
    center: np.ndarray,
    covariance: np.ndarray,
    context: JointLikelihoodContext,
    spec: JointModelSpec,
    bounds: list[tuple[float, float]],
    n_samples: int,
) -> np.ndarray:
    rng = np.random.default_rng(12345)
    accepted: list[np.ndarray] = []
    max_draws = n_samples * 50
    draws = 0
    while len(accepted) < n_samples and draws < max_draws:
        proposal = rng.multivariate_normal(center, covariance)
        draws += 1
        if not parameters_within_bounds(proposal, bounds):
            continue
        objective_value = negative_profile_log_likelihood(proposal, context=context, spec=spec)
        if not np.isfinite(objective_value) or objective_value > 1.0e12:
            continue
        accepted.append(proposal)
    if not accepted:
        accepted = [center.copy()]
    sample_array = np.vstack(accepted)
    if len(sample_array) < n_samples:
        repeats = np.repeat(sample_array[-1:, :], n_samples - len(sample_array), axis=0)
        sample_array = np.vstack([sample_array, repeats])
    return sample_array[:n_samples]


def parameters_within_bounds(params: np.ndarray, bounds: list[tuple[float, float]]) -> bool:
    return bool(
        np.all([lower <= value <= upper for value, (lower, upper) in zip(params, bounds, strict=True)])
    )


def raw_parameter_names(spec: JointModelSpec) -> list[str]:
    if spec.imf_family == "lognormal":
        names = ["mu_log10_msun", "log_sigma"]
    elif spec.imf_family == "powerlaw":
        names = ["alpha_dndm"]
    elif spec.imf_family == "powerlaw_m2":
        names = []
    elif spec.imf_family == "schechter":
        names = ["alpha_dndm", "log10_m_c_msun"]
    elif spec.imf_family == "logspline6":
        names = [f"spline_logit_{index}" for index in range(1, 6)]
    else:
        raise ValueError(f"Unknown IMF family: {spec.imf_family}")

    if spec.radial_model == "step5":
        names.extend(["step_logit_1", "step_logit_2", "step_logit_3", "step_logit_4"])
    elif spec.radial_model == "logpoly3":
        names.extend(["beta1", "beta2", "beta3"])
    elif spec.radial_model == "powerlaw_a":
        names.extend(["beta_log10_a"])
    elif spec.radial_model == "cored_powerlaw_a":
        names.extend(["beta_log10_a", "log10_a_core_kpc"])
    else:
        raise ValueError(f"Unknown radial model: {spec.radial_model}")
    return names


def build_display_parameter_sample_table(
    raw_samples: np.ndarray,
    context: JointLikelihoodContext,
    spec: JointModelSpec,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for raw_params in raw_samples:
        model = unpack_model(raw_params, context=context, spec=spec)
        row: dict[str, float] = {}
        if spec.imf_family == "lognormal":
            row["mu_log10_msun"] = float(raw_params[0])
            row["sigma_log10_msun"] = float(np.exp(raw_params[1]))
            offset = 2
        elif spec.imf_family == "powerlaw":
            row["alpha_dndm"] = float(raw_params[0])
            offset = 1
        elif spec.imf_family == "powerlaw_m2":
            row["alpha_dndm"] = -2.0
            offset = 0
        elif spec.imf_family == "schechter":
            row["alpha_dndm"] = float(raw_params[0])
            row["log10_m_c_msun"] = float(raw_params[1])
            offset = 2
        elif spec.imf_family == "logspline6":
            knot_positions = logspline_knot_positions(context.log_mass_grid, n_knots=6)
            node_log_amplitudes = np.concatenate([np.asarray(raw_params[:5], dtype=float), np.array([0.0])])
            node_log_amplitudes -= np.max(node_log_amplitudes)
            for index, (knot, amplitude) in enumerate(zip(knot_positions, node_log_amplitudes, strict=True), start=1):
                row[f"log10_m_knot_{index}"] = float(knot)
                row[f"node_log_amplitude_{index}"] = float(amplitude)
            offset = 5
        else:
            raise ValueError(f"Unknown IMF family: {spec.imf_family}")

        if spec.radial_model == "step5":
            weights = softmax_with_reference(raw_params[offset:])
            for index, weight in enumerate(weights, start=1):
                row[f"w_a_bin_{index}"] = float(weight)
        elif spec.radial_model == "logpoly3":
            row["beta1"] = float(raw_params[offset + 0])
            row["beta2"] = float(raw_params[offset + 1])
            row["beta3"] = float(raw_params[offset + 2])
        elif spec.radial_model == "powerlaw_a":
            row["beta_log10_a"] = float(raw_params[offset + 0])
            row["gamma_linear_a"] = float(1.0 - raw_params[offset + 0])
        elif spec.radial_model == "cored_powerlaw_a":
            row["beta_log10_a"] = float(raw_params[offset + 0])
            row["gamma_linear_a"] = float(1.0 - raw_params[offset + 0])
            row["log10_a_core_kpc"] = float(raw_params[offset + 1])
            row["a_core_kpc"] = float(np.power(10.0, raw_params[offset + 1]))
        else:
            raise ValueError(f"Unknown radial model: {spec.radial_model}")

        row["log10_N0"] = float(np.log10(model["total_initial_count"]))
        row["survival_fraction"] = float(model["survival_fraction"])
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_imf_density_at_log_mass(
    params: np.ndarray,
    context: JointLikelihoodContext,
    spec: JointModelSpec,
    log_mass_value: float,
) -> float:
    imf_param_count = imf_parameter_count(spec.imf_family)
    imf_params = np.asarray(params[:imf_param_count], dtype=float)
    _, density_at_value, _ = evaluate_imf_family(
        spec.imf_family,
        imf_params,
        context.log_mass_grid,
        np.asarray([log_mass_value], dtype=float),
    )
    return float(np.clip(density_at_value[0], TINY, None))


def compute_profile_likelihood_imf_band(
    best_payload: dict[str, object],
    context: JointLikelihoodContext,
    log_mass_support: np.ndarray,
    delta_nll: float = 0.5,
    initial_step_dex: float = 0.05,
    max_offset_dex: float = 1.5,
    n_bisection: int = 12,
) -> dict[str, np.ndarray]:
    spec = best_payload["spec"]
    bounds = best_payload["bounds"]
    best_params = np.asarray(best_payload["raw_parameters"], dtype=float)
    best_nll = negative_profile_log_likelihood(best_params, context=context, spec=spec)

    support = np.asarray(log_mass_support, dtype=float)
    best_density = np.asarray(
        [evaluate_imf_density_at_log_mass(best_params, context=context, spec=spec, log_mass_value=value) for value in support],
        dtype=float,
    )

    lower_density = np.full_like(best_density, np.nan)
    upper_density = np.full_like(best_density, np.nan)
    lower_delta_nll = np.full_like(best_density, np.nan)
    upper_delta_nll = np.full_like(best_density, np.nan)

    lower_start = best_params.copy()
    upper_start = best_params.copy()

    for index, log_mass_value in enumerate(support):
        lower_result = find_profile_likelihood_imf_bound(
            best_params=best_params,
            best_nll=best_nll,
            context=context,
            spec=spec,
            bounds=bounds,
            log_mass_value=float(log_mass_value),
            direction=-1.0,
            delta_nll=delta_nll,
            initial_step_dex=initial_step_dex,
            max_offset_dex=max_offset_dex,
            n_bisection=n_bisection,
            warm_start=lower_start,
        )
        lower_density[index] = lower_result["density"]
        lower_delta_nll[index] = lower_result["delta_nll"]
        lower_start = np.asarray(lower_result["params"], dtype=float)

        upper_result = find_profile_likelihood_imf_bound(
            best_params=best_params,
            best_nll=best_nll,
            context=context,
            spec=spec,
            bounds=bounds,
            log_mass_value=float(log_mass_value),
            direction=+1.0,
            delta_nll=delta_nll,
            initial_step_dex=initial_step_dex,
            max_offset_dex=max_offset_dex,
            n_bisection=n_bisection,
            warm_start=upper_start,
        )
        upper_density[index] = upper_result["density"]
        upper_delta_nll[index] = upper_result["delta_nll"]
        upper_start = np.asarray(upper_result["params"], dtype=float)

    return {
        "log_mass_support": support,
        "best_density": best_density,
        "lower_density": lower_density,
        "upper_density": upper_density,
        "lower_delta_nll": lower_delta_nll,
        "upper_delta_nll": upper_delta_nll,
    }


def evaluate_radial_birth_intensity_at_log_a(
    params: np.ndarray,
    context: JointLikelihoodContext,
    spec: JointModelSpec,
    log_a_value: float,
) -> float:
    model = unpack_model(np.asarray(params, dtype=float), context=context, spec=spec)
    radial_density = interpolate_density(
        context.log_a_grid,
        np.asarray(model["radial_density_grid"], dtype=float),
        np.asarray([log_a_value], dtype=float),
    )[0]
    return float(np.clip(model["total_initial_count"] * radial_density, TINY, None))


def compute_profile_likelihood_radial_birth_band(
    best_payload: dict[str, object],
    context: JointLikelihoodContext,
    log_a_support: np.ndarray,
    delta_nll: float = 0.5,
    initial_step_dex: float = 0.05,
    max_offset_dex: float = 1.5,
    n_bisection: int = 12,
) -> dict[str, np.ndarray]:
    spec = best_payload["spec"]
    bounds = best_payload["bounds"]
    best_params = np.asarray(best_payload["raw_parameters"], dtype=float)
    best_nll = negative_profile_log_likelihood(best_params, context=context, spec=spec)

    support = np.asarray(log_a_support, dtype=float)
    best_density = np.asarray(
        [
            evaluate_radial_birth_intensity_at_log_a(
                best_params,
                context=context,
                spec=spec,
                log_a_value=value,
            )
            for value in support
        ],
        dtype=float,
    )

    lower_density = np.full_like(best_density, np.nan)
    upper_density = np.full_like(best_density, np.nan)
    lower_delta_nll = np.full_like(best_density, np.nan)
    upper_delta_nll = np.full_like(best_density, np.nan)

    lower_start = best_params.copy()
    upper_start = best_params.copy()

    for index, log_a_value in enumerate(support):
        lower_result = find_profile_likelihood_radial_birth_bound(
            best_params=best_params,
            best_nll=best_nll,
            context=context,
            spec=spec,
            bounds=bounds,
            log_a_value=float(log_a_value),
            direction=-1.0,
            delta_nll=delta_nll,
            initial_step_dex=initial_step_dex,
            max_offset_dex=max_offset_dex,
            n_bisection=n_bisection,
            warm_start=lower_start,
        )
        lower_density[index] = lower_result["density"]
        lower_delta_nll[index] = lower_result["delta_nll"]
        lower_start = np.asarray(lower_result["params"], dtype=float)

        upper_result = find_profile_likelihood_radial_birth_bound(
            best_params=best_params,
            best_nll=best_nll,
            context=context,
            spec=spec,
            bounds=bounds,
            log_a_value=float(log_a_value),
            direction=+1.0,
            delta_nll=delta_nll,
            initial_step_dex=initial_step_dex,
            max_offset_dex=max_offset_dex,
            n_bisection=n_bisection,
            warm_start=upper_start,
        )
        upper_density[index] = upper_result["density"]
        upper_delta_nll[index] = upper_result["delta_nll"]
        upper_start = np.asarray(upper_result["params"], dtype=float)

    return {
        "log_a_support": support,
        "best_density": best_density,
        "lower_density": lower_density,
        "upper_density": upper_density,
        "lower_delta_nll": lower_delta_nll,
        "upper_delta_nll": upper_delta_nll,
    }


def find_profile_likelihood_imf_bound(
    best_params: np.ndarray,
    best_nll: float,
    context: JointLikelihoodContext,
    spec: JointModelSpec,
    bounds: list[tuple[float, float]],
    log_mass_value: float,
    direction: float,
    delta_nll: float,
    initial_step_dex: float,
    max_offset_dex: float,
    n_bisection: int,
    warm_start: np.ndarray | None = None,
) -> dict[str, object]:
    best_density = evaluate_imf_density_at_log_mass(
        best_params,
        context=context,
        spec=spec,
        log_mass_value=log_mass_value,
    )
    best_log10_density = float(np.log10(best_density))
    base_state = {
        "target_log10_density": best_log10_density,
        "delta_nll": 0.0,
        "nll": best_nll,
        "density": best_density,
        "params": np.asarray(best_params, dtype=float),
    }

    inside = base_state
    outside: dict[str, object] | None = None
    offset_dex = float(initial_step_dex)
    branch_start = np.asarray(warm_start if warm_start is not None else best_params, dtype=float)

    while offset_dex <= max_offset_dex:
        target_log10_density = best_log10_density + float(direction) * offset_dex
        candidate = profile_nll_at_fixed_imf_density(
            target_log10_density=target_log10_density,
            log_mass_value=log_mass_value,
            context=context,
            spec=spec,
            bounds=bounds,
            start_vectors=[
                branch_start,
                np.asarray(inside["params"], dtype=float),
                np.asarray(best_params, dtype=float),
                *initial_parameter_vectors(spec),
            ],
        )
        if not candidate["success"]:
            break
        candidate_delta_nll = float(candidate["nll"] - best_nll)
        candidate_state = {
            "target_log10_density": float(target_log10_density),
            "delta_nll": candidate_delta_nll,
            "nll": float(candidate["nll"]),
            "density": float(10.0 ** target_log10_density),
            "params": np.asarray(candidate["params"], dtype=float),
        }
        if candidate_delta_nll <= delta_nll:
            inside = candidate_state
            branch_start = np.asarray(candidate["params"], dtype=float)
            offset_dex *= 2.0
        else:
            outside = candidate_state
            break

    if outside is None:
        return inside

    for _ in range(n_bisection):
        target_log10_density = 0.5 * (
            float(inside["target_log10_density"]) + float(outside["target_log10_density"])
        )
        candidate = profile_nll_at_fixed_imf_density(
            target_log10_density=target_log10_density,
            log_mass_value=log_mass_value,
            context=context,
            spec=spec,
            bounds=bounds,
            start_vectors=[
                np.asarray(inside["params"], dtype=float),
                np.asarray(outside["params"], dtype=float),
                np.asarray(best_params, dtype=float),
            ],
        )
        if not candidate["success"]:
            break
        candidate_delta_nll = float(candidate["nll"] - best_nll)
        candidate_state = {
            "target_log10_density": float(target_log10_density),
            "delta_nll": candidate_delta_nll,
            "nll": float(candidate["nll"]),
            "density": float(10.0 ** target_log10_density),
            "params": np.asarray(candidate["params"], dtype=float),
        }
        if candidate_delta_nll <= delta_nll:
            inside = candidate_state
        else:
            outside = candidate_state

    if abs(float(outside["delta_nll"]) - delta_nll) < abs(float(inside["delta_nll"]) - delta_nll):
        return outside
    return inside


def find_profile_likelihood_radial_birth_bound(
    best_params: np.ndarray,
    best_nll: float,
    context: JointLikelihoodContext,
    spec: JointModelSpec,
    bounds: list[tuple[float, float]],
    log_a_value: float,
    direction: float,
    delta_nll: float,
    initial_step_dex: float,
    max_offset_dex: float,
    n_bisection: int,
    warm_start: np.ndarray | None = None,
) -> dict[str, object]:
    best_density = evaluate_radial_birth_intensity_at_log_a(
        best_params,
        context=context,
        spec=spec,
        log_a_value=log_a_value,
    )
    best_log10_density = float(np.log10(best_density))
    base_state = {
        "target_log10_density": best_log10_density,
        "delta_nll": 0.0,
        "nll": best_nll,
        "density": best_density,
        "params": np.asarray(best_params, dtype=float),
    }

    inside = base_state
    outside: dict[str, object] | None = None
    offset_dex = float(initial_step_dex)
    branch_start = np.asarray(warm_start if warm_start is not None else best_params, dtype=float)

    while offset_dex <= max_offset_dex:
        target_log10_density = best_log10_density + float(direction) * offset_dex
        candidate = profile_nll_at_fixed_radial_birth_intensity(
            target_log10_density=target_log10_density,
            log_a_value=log_a_value,
            context=context,
            spec=spec,
            bounds=bounds,
            start_vectors=[
                branch_start,
                np.asarray(inside["params"], dtype=float),
                np.asarray(best_params, dtype=float),
                *initial_parameter_vectors(spec),
            ],
        )
        if not candidate["success"]:
            break
        candidate_delta_nll = float(candidate["nll"] - best_nll)
        candidate_state = {
            "target_log10_density": float(target_log10_density),
            "delta_nll": candidate_delta_nll,
            "nll": float(candidate["nll"]),
            "density": float(10.0 ** target_log10_density),
            "params": np.asarray(candidate["params"], dtype=float),
        }
        if candidate_delta_nll <= delta_nll:
            inside = candidate_state
            branch_start = np.asarray(candidate["params"], dtype=float)
            offset_dex *= 2.0
        else:
            outside = candidate_state
            break

    if outside is None:
        return inside

    for _ in range(n_bisection):
        target_log10_density = 0.5 * (
            float(inside["target_log10_density"]) + float(outside["target_log10_density"])
        )
        candidate = profile_nll_at_fixed_radial_birth_intensity(
            target_log10_density=target_log10_density,
            log_a_value=log_a_value,
            context=context,
            spec=spec,
            bounds=bounds,
            start_vectors=[
                np.asarray(inside["params"], dtype=float),
                np.asarray(outside["params"], dtype=float),
                np.asarray(best_params, dtype=float),
            ],
        )
        if not candidate["success"]:
            break
        candidate_delta_nll = float(candidate["nll"] - best_nll)
        candidate_state = {
            "target_log10_density": float(target_log10_density),
            "delta_nll": candidate_delta_nll,
            "nll": float(candidate["nll"]),
            "density": float(10.0 ** target_log10_density),
            "params": np.asarray(candidate["params"], dtype=float),
        }
        if candidate_delta_nll <= delta_nll:
            inside = candidate_state
        else:
            outside = candidate_state

    if abs(float(outside["delta_nll"]) - delta_nll) < abs(float(inside["delta_nll"]) - delta_nll):
        return outside
    return inside


def profile_nll_at_fixed_imf_density(
    target_log10_density: float,
    log_mass_value: float,
    context: JointLikelihoodContext,
    spec: JointModelSpec,
    bounds: list[tuple[float, float]],
    start_vectors: list[np.ndarray],
) -> dict[str, object]:
    def constraint_value(params: np.ndarray) -> float:
        density = evaluate_imf_density_at_log_mass(
            params,
            context=context,
            spec=spec,
            log_mass_value=log_mass_value,
        )
        return float(np.log10(density) - target_log10_density)

    constraint = {"type": "eq", "fun": constraint_value}
    best_result = None
    best_value = np.inf
    best_violation = np.inf

    for start in start_vectors:
        start = np.asarray(start, dtype=float)
        if start.shape != np.asarray(bounds, dtype=float).shape[:1]:
            continue
        result = optimize.minimize(
            lambda params: negative_profile_log_likelihood(params, context=context, spec=spec),
            x0=start,
            method="SLSQP",
            bounds=bounds,
            constraints=[constraint],
            options={"ftol": 1.0e-9, "maxiter": 500, "disp": False},
        )
        if not np.all(np.isfinite(result.x)) or not np.isfinite(result.fun):
            continue
        violation = abs(constraint_value(result.x))
        if violation < 1.0e-4 and (float(result.fun) < best_value or violation < best_violation):
            best_result = result
            best_value = float(result.fun)
            best_violation = violation

    if best_result is None:
        return {
            "success": False,
            "nll": np.inf,
            "params": np.asarray(start_vectors[0], dtype=float),
            "constraint_violation": np.inf,
        }
    return {
        "success": True,
        "nll": float(best_result.fun),
        "params": np.asarray(best_result.x, dtype=float),
        "constraint_violation": float(best_violation),
    }


def profile_nll_at_fixed_radial_birth_intensity(
    target_log10_density: float,
    log_a_value: float,
    context: JointLikelihoodContext,
    spec: JointModelSpec,
    bounds: list[tuple[float, float]],
    start_vectors: list[np.ndarray],
) -> dict[str, object]:
    def constraint_value(params: np.ndarray) -> float:
        density = evaluate_radial_birth_intensity_at_log_a(
            params,
            context=context,
            spec=spec,
            log_a_value=log_a_value,
        )
        return float(np.log10(density) - target_log10_density)

    constraint = {"type": "eq", "fun": constraint_value}
    best_result = None
    best_value = np.inf
    best_violation = np.inf

    for start in start_vectors:
        start = np.asarray(start, dtype=float)
        if start.shape != np.asarray(bounds, dtype=float).shape[:1]:
            continue
        result = optimize.minimize(
            lambda params: negative_profile_log_likelihood(params, context=context, spec=spec),
            x0=start,
            method="SLSQP",
            bounds=bounds,
            constraints=[constraint],
            options={"ftol": 1.0e-9, "maxiter": 500, "disp": False},
        )
        if not np.all(np.isfinite(result.x)) or not np.isfinite(result.fun):
            continue
        violation = abs(constraint_value(result.x))
        if violation < 1.0e-4 and (float(result.fun) < best_value or violation < best_violation):
            best_result = result
            best_value = float(result.fun)
            best_violation = violation

    if best_result is None:
        return {
            "success": False,
            "nll": np.inf,
            "params": np.asarray(start_vectors[0], dtype=float),
            "constraint_violation": np.inf,
        }
    return {
        "success": True,
        "nll": float(best_result.fun),
        "params": np.asarray(best_result.x, dtype=float),
        "constraint_violation": float(best_violation),
    }


def build_model_performance_tables(
    all_payloads: list[dict[str, object]],
    context: JointLikelihoodContext,
    n_mass_bins: int = 12,
    n_a_bins: int = 9,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mass_bin_edges = np.linspace(context.log_mass_grid[0], context.log_mass_grid[-1], n_mass_bins + 1)
    log_a_bin_edges = np.linspace(context.log_a_grid[0], context.log_a_grid[-1], n_a_bins + 1)
    observed_2d, _, _ = np.histogram2d(
        context.log_mass_data,
        context.log_a_data,
        bins=[mass_bin_edges, log_a_bin_edges],
    )

    summary_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    for payload in all_payloads:
        model = payload["model"]
        summary = payload["summary"]
        point_intensity_grid = compute_observed_intensity_grid(
            model["imf_density_grid"],
            model["radial_density_grid"],
            context.survival_probability_grid,
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
        pearson_chi2 = float(np.sum(np.square(observed_2d[valid] - expected_2d[valid]) / expected_2d[valid]))
        rms_residual_sigma = float(np.sqrt(np.mean(np.square(residual_sigma[valid]))))
        mean_abs_residual_sigma = float(np.mean(np.abs(residual_sigma[valid])))
        poisson_deviance = float(np.sum(poisson_deviance_terms(observed_2d, expected_2d)))
        summary_rows.append(
            {
                "imf_family": summary.imf_family,
                "radial_model": summary.radial_model,
                "log_likelihood": summary.log_likelihood,
                "aic": summary.aic,
                "bic": summary.bic,
                "pearson_chi2_2d": pearson_chi2,
                "rms_residual_sigma_2d": rms_residual_sigma,
                "mean_abs_residual_sigma_2d": mean_abs_residual_sigma,
                "poisson_deviance_2d": poisson_deviance,
            }
        )

        mass_centers = 0.5 * (mass_bin_edges[:-1] + mass_bin_edges[1:])
        log_a_centers = 0.5 * (log_a_bin_edges[:-1] + log_a_bin_edges[1:])
        for i_mass, log_mass_center in enumerate(mass_centers):
            for i_a, log_a_center in enumerate(log_a_centers):
                residual_rows.append(
                    {
                        "imf_family": summary.imf_family,
                        "radial_model": summary.radial_model,
                        "log_initial_mass_center_msun": float(log_mass_center),
                        "semi_major_axis_center_kpc": float(np.power(10.0, log_a_center)),
                        "log10_semi_major_axis_center_kpc": float(log_a_center),
                        "observed_count": float(observed_2d[i_mass, i_a]),
                        "expected_count": float(expected_2d[i_mass, i_a]),
                        "residual_sigma": float(residual_sigma[i_mass, i_a]),
                    }
                )

    performance_summary_table = pd.DataFrame(summary_rows).sort_values("bic", ascending=True).reset_index(drop=True)
    best_bic = float(performance_summary_table["bic"].min())
    performance_summary_table["delta_bic"] = performance_summary_table["bic"] - best_bic
    return performance_summary_table, pd.DataFrame(residual_rows)


def rebin_expected_counts_2d(
    point_intensity_grid: np.ndarray,
    log_mass_grid: np.ndarray,
    log_a_grid: np.ndarray,
    mass_bin_edges: np.ndarray,
    log_a_bin_edges: np.ndarray,
) -> np.ndarray:
    log_mass_edges = centers_to_edges_local(log_mass_grid)
    log_a_edges = centers_to_edges_local(log_a_grid)
    cell_expected = point_intensity_grid * np.diff(log_mass_edges)[:, None] * np.diff(log_a_edges)[None, :]
    mass_index = np.digitize(log_mass_grid, bins=mass_bin_edges[1:-1], right=False)
    a_index = np.digitize(log_a_grid, bins=log_a_bin_edges[1:-1], right=False)
    rebinned = np.zeros((len(mass_bin_edges) - 1, len(log_a_bin_edges) - 1), dtype=float)
    for i_mass, mass_bin in enumerate(mass_index):
        for i_a, a_bin in enumerate(a_index):
            rebinned[mass_bin, a_bin] += cell_expected[i_mass, i_a]
    return rebinned


def centers_to_edges_local(centers: np.ndarray) -> np.ndarray:
    steps = np.diff(centers)
    left_edge = centers[0] - 0.5 * steps[0]
    right_edge = centers[-1] + 0.5 * steps[-1]
    interior = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate(([left_edge], interior, [right_edge]))


def poisson_deviance_terms(observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
    expected = np.clip(expected, 1.0e-12, None)
    terms = np.zeros_like(expected, dtype=float)
    positive = observed > 0.0
    terms[positive] = 2.0 * (
        observed[positive] * np.log(observed[positive] / expected[positive])
        - (observed[positive] - expected[positive])
    )
    terms[~positive] = 2.0 * expected[~positive]
    return terms
