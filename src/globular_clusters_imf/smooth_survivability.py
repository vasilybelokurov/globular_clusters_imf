from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, special

from .model import AGE_MYR, survival_mass_cut_msun


@dataclass(frozen=True)
class SmoothSurvivabilityFitSummary:
    eta_t: float
    bandwidth_log10_a_dex: float
    outer_plateau_90_log10_msun_from_raw_boundary: float
    outer_level_90_log10_msun: float
    outer_level_50_log10_msun: float
    outer_level_10_log10_msun: float
    inner_level_90_log10_msun: float
    inner_level_50_log10_msun: float
    inner_level_10_log10_msun: float
    transition_log10_a_kpc: float
    transition_a_kpc: float
    width_log10_a_dex: float
    transition_band_width_dex: float
    optimization_success: bool
    optimizer_message: str
    optimizer_value: float


def build_raw_survival_grid_from_eta_t(
    catalog: pd.DataFrame,
    *,
    eta_t: float,
    n_radius_grid: int = 160,
    n_mass_grid: int = 180,
    bandwidth_log10_a_dex: float = 0.18,
) -> dict[str, object]:
    working = catalog.copy()
    effective_age_myr = AGE_MYR / float(eta_t)
    working["renormalized_survival_mass_cut_msun"] = working.apply(
        lambda row: survival_mass_cut_msun(
            r_apo_kpc=row["r_apo_kpc"],
            eccentricity=row["eccentricity"],
            age_myr=effective_age_myr,
        ),
        axis=1,
    )
    working["log_renormalized_survival_mass_cut_msun"] = np.log10(
        working["renormalized_survival_mass_cut_msun"].to_numpy(dtype=float)
    )

    log_a_data = np.log10(working["semi_major_axis_kpc"].to_numpy(dtype=float))
    log_cut_data = working["log_renormalized_survival_mass_cut_msun"].to_numpy(dtype=float)

    log_a_grid = np.linspace(log_a_data.min(), log_a_data.max(), n_radius_grid)
    log_mass_min = min(3.5, float(np.floor(working["log_initial_mass_msun"].min() * 10.0) / 10.0))
    log_mass_max = max(7.3, float(np.ceil(working["log_initial_mass_msun"].max() * 10.0) / 10.0))
    log_mass_grid = np.linspace(log_mass_min, log_mass_max, n_mass_grid)

    weights = np.exp(
        -0.5 * np.square((log_a_grid[:, None] - log_a_data[None, :]) / bandwidth_log10_a_dex)
    )
    weights /= np.clip(weights.sum(axis=1, keepdims=True), 1.0e-12, None)

    indicators = log_mass_grid[:, None] >= log_cut_data[None, :]
    survival_probability = indicators @ weights.T
    return {
        "catalog": working,
        "log_mass_grid": log_mass_grid,
        "log_a_grid": log_a_grid,
        "semi_major_axis_grid_kpc": np.power(10.0, log_a_grid),
        "survival_probability": np.clip(survival_probability, 1.0e-12, 1.0),
        "bandwidth_log10_a_dex": bandwidth_log10_a_dex,
        "eta_t": float(eta_t),
    }


def extract_probability_boundary(
    log_mass_grid: np.ndarray,
    log_a_grid: np.ndarray,
    survival_probability: np.ndarray,
    *,
    level: float,
) -> np.ndarray:
    boundary = np.full_like(log_a_grid, np.nan, dtype=float)
    for idx, _ in enumerate(log_a_grid):
        column = survival_probability[:, idx]
        if np.all(column < level):
            boundary[idx] = float(log_mass_grid[-1])
            continue
        if np.all(column > level):
            boundary[idx] = float(log_mass_grid[0])
            continue
        above = np.flatnonzero(column >= level)
        upper = int(above[0])
        lower = max(upper - 1, 0)
        x0 = float(column[lower])
        x1 = float(column[upper])
        y0 = float(log_mass_grid[lower])
        y1 = float(log_mass_grid[upper])
        if np.isclose(x0, x1):
            boundary[idx] = 0.5 * (y0 + y1)
        else:
            frac = (level - x0) / (x1 - x0)
            boundary[idx] = y0 + frac * (y1 - y0)
    return boundary


def estimate_outer_plateau_from_raw_boundary_90(
    raw_boundary_90: np.ndarray,
    semi_major_axis_grid_kpc: np.ndarray,
) -> float:
    semi_major_axis_grid_kpc = np.asarray(semi_major_axis_grid_kpc, dtype=float)
    raw_boundary_90 = np.asarray(raw_boundary_90, dtype=float)
    mask = semi_major_axis_grid_kpc >= 100.0
    if np.count_nonzero(mask) >= 3:
        return float(np.nanmedian(raw_boundary_90[mask]))
    return float(np.nanmedian(raw_boundary_90[-10:]))


def boundary_curve_from_params(
    params: np.ndarray,
    log_a_grid: np.ndarray,
    *,
    outer_level_90: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    inner_excess = float(np.exp(params[0]))
    transition_log_a = float(params[1])
    width_dex = float(np.exp(params[2]))
    transition_band_width_dex = float(np.exp(params[3]))
    logistic_weight = special.expit(-(log_a_grid - transition_log_a) / width_dex)
    boundary_90 = outer_level_90 + inner_excess * logistic_weight
    boundary_50 = boundary_90 - 0.5 * transition_band_width_dex
    boundary_10 = boundary_90 - transition_band_width_dex
    metadata = {
        "outer_level_90_log10_msun": outer_level_90,
        "outer_level_50_log10_msun": outer_level_90 - 0.5 * transition_band_width_dex,
        "outer_level_10_log10_msun": outer_level_90 - transition_band_width_dex,
        "inner_level_90_log10_msun": outer_level_90 + inner_excess,
        "inner_level_50_log10_msun": outer_level_90 + inner_excess - 0.5 * transition_band_width_dex,
        "inner_level_10_log10_msun": outer_level_90 + inner_excess - transition_band_width_dex,
        "transition_log10_a_kpc": transition_log_a,
        "transition_a_kpc": float(np.power(10.0, transition_log_a)),
        "width_log10_a_dex": width_dex,
        "transition_band_width_dex": transition_band_width_dex,
    }
    return boundary_10, boundary_50, boundary_90, metadata


def compact_transition_surface(
    log_mass_grid: np.ndarray,
    boundary_10: np.ndarray,
    boundary_90: np.ndarray,
) -> np.ndarray:
    width = np.clip(boundary_90 - boundary_10, 1.0e-6, None)
    u = (log_mass_grid[:, None] - boundary_10[None, :]) / width[None, :]
    s = np.zeros_like(u, dtype=float)
    inside = (u > 0.0) & (u < 1.0)
    u_inside = u[inside]
    smooth = u_inside * u_inside * (3.0 - 2.0 * u_inside)
    s[inside] = 0.1 + 0.8 * smooth
    s[u >= 1.0] = 1.0
    return np.clip(s, 0.0, 1.0)


def logistic_tail_transition_surface(
    log_mass_grid: np.ndarray,
    boundary_10: np.ndarray,
    boundary_90: np.ndarray,
) -> np.ndarray:
    width = np.clip(boundary_90 - boundary_10, 1.0e-6, None)
    logit_90 = np.log(0.9 / 0.1)
    sigma = width / (2.0 * logit_90)
    boundary_50 = 0.5 * (boundary_10 + boundary_90)
    z = (log_mass_grid[:, None] - boundary_50[None, :]) / sigma[None, :]
    return special.expit(z)


def build_transition_surface(
    log_mass_grid: np.ndarray,
    boundary_10: np.ndarray,
    boundary_90: np.ndarray,
    *,
    surface_model: str,
) -> np.ndarray:
    if surface_model == "compact":
        return compact_transition_surface(log_mass_grid, boundary_10, boundary_90)
    if surface_model == "logistic":
        return logistic_tail_transition_surface(log_mass_grid, boundary_10, boundary_90)
    raise ValueError(f"Unknown surface_model: {surface_model}")


def fit_monotonic_soft_survivability_model(
    catalog: pd.DataFrame,
    log_mass_grid: np.ndarray,
    log_a_grid: np.ndarray,
    survival_probability_raw: np.ndarray,
    bandwidth_log10_a_dex: float,
    *,
    surface_model: str = "compact",
) -> dict[str, object]:
    raw_boundary_10 = extract_probability_boundary(log_mass_grid, log_a_grid, survival_probability_raw, level=0.1)
    raw_boundary_50 = extract_probability_boundary(log_mass_grid, log_a_grid, survival_probability_raw, level=0.5)
    raw_boundary_80 = extract_probability_boundary(log_mass_grid, log_a_grid, survival_probability_raw, level=0.8)
    raw_boundary_90 = extract_probability_boundary(log_mass_grid, log_a_grid, survival_probability_raw, level=0.9)
    outer_plateau_90 = estimate_outer_plateau_from_raw_boundary_90(
        raw_boundary_90,
        np.power(10.0, log_a_grid),
    )
    log_a_mid = 0.5 * (float(log_a_grid.min()) + float(log_a_grid.max()))
    raw_transition_width = np.nanmedian(raw_boundary_90 - raw_boundary_10)
    base_start = np.array(
        [
            np.log(
                max(
                    float(np.nanpercentile(raw_boundary_90, 90.0) - outer_plateau_90),
                    0.2,
                )
            ),
            log_a_mid,
            np.log(0.45),
            np.log(max(float(raw_transition_width), 0.12)),
        ],
        dtype=float,
    )
    starts = [
        base_start,
        np.array([base_start[0], log_a_mid - 0.2, np.log(0.35), np.log(max(float(raw_transition_width), 0.18))]),
        np.array([base_start[0], log_a_mid + 0.2, np.log(0.60), np.log(max(float(raw_transition_width), 0.25))]),
        np.array([base_start[0], log_a_mid + 0.35, np.log(0.75), np.log(0.55)]),
    ]

    transition_weights = 0.08 + 4.0 * survival_probability_raw * (1.0 - survival_probability_raw)
    log_mass_data = catalog["log_initial_mass_msun"].to_numpy(dtype=float)
    log_a_data = np.log10(catalog["semi_major_axis_kpc"].to_numpy(dtype=float))
    local_density_weights = np.exp(
        -0.5 * np.square((log_a_grid[:, None] - log_a_data[None, :]) / bandwidth_log10_a_dex)
    ).sum(axis=1)
    local_density_weights /= np.clip(np.max(local_density_weights), 1.0e-12, None)
    boundary_weight_profile = 0.3 + 2.7 * local_density_weights
    boundary_weight_50 = 2.5
    boundary_weight_edges = 1.2
    occupancy_weight = 0.6
    a_bin_edges = np.quantile(log_a_data, np.linspace(0.0, 1.0, 9))
    a_bin_edges[0] -= 1.0e-6
    a_bin_edges[-1] += 1.0e-6

    def transition_band_fraction_variance(boundary_10_data: np.ndarray, boundary_90_data: np.ndarray) -> float:
        inside_band = (log_mass_data >= boundary_10_data) & (log_mass_data <= boundary_90_data)
        fractions = []
        for lower, upper in zip(a_bin_edges[:-1], a_bin_edges[1:], strict=True):
            mask = (log_a_data >= lower) & (log_a_data < upper)
            if np.count_nonzero(mask) == 0:
                continue
            fractions.append(float(np.mean(inside_band[mask])))
        if len(fractions) <= 1:
            return 0.0
        return float(np.var(np.asarray(fractions, dtype=float)))

    def objective(params: np.ndarray) -> float:
        boundary_10, boundary_50, boundary_90, _ = boundary_curve_from_params(
            np.asarray(params, dtype=float),
            log_a_grid,
            outer_level_90=outer_plateau_90,
        )
        model_probability = build_transition_surface(
            log_mass_grid,
            boundary_10,
            boundary_90,
            surface_model=surface_model,
        )
        map_residual = model_probability - survival_probability_raw
        boundary_residual_50 = boundary_50 - raw_boundary_50
        boundary_residual_10 = boundary_10 - raw_boundary_10
        boundary_residual_90 = boundary_90 - raw_boundary_90
        boundary_10_data, _, boundary_90_data, _ = boundary_curve_from_params(
            np.asarray(params, dtype=float),
            log_a_data,
            outer_level_90=outer_plateau_90,
        )
        occupancy_penalty = transition_band_fraction_variance(boundary_10_data, boundary_90_data)
        return float(
            np.mean(transition_weights * np.square(map_residual))
            + boundary_weight_50 * np.mean(boundary_weight_profile * np.square(boundary_residual_50))
            + boundary_weight_edges * (
                np.mean(boundary_weight_profile * np.square(boundary_residual_10))
                + np.mean(boundary_weight_profile * np.square(boundary_residual_90))
            )
            + occupancy_weight * occupancy_penalty
        )

    bounds = [
        (np.log(0.08), np.log(4.0)),
        (float(log_a_grid.min()), float(log_a_grid.max())),
        (np.log(0.08), np.log(1.2)),
        (np.log(0.03), np.log(1.2)),
    ]
    best_result = None
    best_value = np.inf
    for start in starts:
        result = optimize.minimize(objective, start, method="L-BFGS-B", bounds=bounds)
        if float(result.fun) < best_value:
            best_value = float(result.fun)
            best_result = result
    if best_result is None:
        raise RuntimeError("Survivability surface fit did not start.")
    result = best_result
    best_params = np.asarray(result.x, dtype=float)
    boundary_10, boundary_50, boundary_90, meta = boundary_curve_from_params(
        best_params,
        log_a_grid,
        outer_level_90=outer_plateau_90,
    )
    fitted_probability = build_transition_surface(
        log_mass_grid,
        boundary_10,
        boundary_90,
        surface_model=surface_model,
    )
    occupancy_rows = []
    boundary_10_data, _, boundary_90_data, _ = boundary_curve_from_params(
        best_params,
        log_a_data,
        outer_level_90=outer_plateau_90,
    )
    inside_band = (log_mass_data >= boundary_10_data) & (log_mass_data <= boundary_90_data)
    for lower, upper in zip(a_bin_edges[:-1], a_bin_edges[1:], strict=True):
        mask = (log_a_data >= lower) & (log_a_data < upper)
        if np.count_nonzero(mask) == 0:
            continue
        occupancy_rows.append(
            {
                "log10_a_lower": float(lower),
                "log10_a_upper": float(upper),
                "a_lower_kpc": float(np.power(10.0, lower)),
                "a_upper_kpc": float(np.power(10.0, upper)),
                "n_clusters": int(np.count_nonzero(mask)),
                "transition_band_fraction": float(np.mean(inside_band[mask])),
            }
        )
    summary = SmoothSurvivabilityFitSummary(
        eta_t=np.nan,
        bandwidth_log10_a_dex=float(bandwidth_log10_a_dex),
        outer_plateau_90_log10_msun_from_raw_boundary=float(outer_plateau_90),
        outer_level_90_log10_msun=float(meta["outer_level_90_log10_msun"]),
        outer_level_50_log10_msun=float(meta["outer_level_50_log10_msun"]),
        outer_level_10_log10_msun=float(meta["outer_level_10_log10_msun"]),
        inner_level_90_log10_msun=float(meta["inner_level_90_log10_msun"]),
        inner_level_50_log10_msun=float(meta["inner_level_50_log10_msun"]),
        inner_level_10_log10_msun=float(meta["inner_level_10_log10_msun"]),
        transition_log10_a_kpc=float(meta["transition_log10_a_kpc"]),
        transition_a_kpc=float(meta["transition_a_kpc"]),
        width_log10_a_dex=float(meta["width_log10_a_dex"]),
        transition_band_width_dex=float(meta["transition_band_width_dex"]),
        optimization_success=bool(result.success),
        optimizer_message=str(result.message),
        optimizer_value=float(result.fun),
    )
    return {
        "summary": summary,
        "raw_boundary_10_log10_msun": raw_boundary_10,
        "raw_boundary_50_log10_msun": raw_boundary_50,
        "raw_boundary_80_log10_msun": raw_boundary_80,
        "raw_boundary_90_log10_msun": raw_boundary_90,
        "fitted_boundary_10_log10_msun": boundary_10,
        "fitted_boundary_50_log10_msun": boundary_50,
        "fitted_boundary_90_log10_msun": boundary_90,
        "fitted_probability": fitted_probability,
        "occupancy_table": pd.DataFrame(occupancy_rows),
    }


def build_smooth_survivability_grid(
    catalog: pd.DataFrame,
    *,
    eta_t: float = 1.0,
    n_radius_grid: int = 160,
    n_mass_grid: int = 180,
    bandwidth_log10_a_dex: float = 0.18,
    surface_model: str = "compact",
) -> dict[str, object]:
    raw_grid = build_raw_survival_grid_from_eta_t(
        catalog,
        eta_t=eta_t,
        n_radius_grid=n_radius_grid,
        n_mass_grid=n_mass_grid,
        bandwidth_log10_a_dex=bandwidth_log10_a_dex,
    )
    fit_payload = fit_monotonic_soft_survivability_model(
        raw_grid["catalog"],
        np.asarray(raw_grid["log_mass_grid"], dtype=float),
        np.asarray(raw_grid["log_a_grid"], dtype=float),
        np.asarray(raw_grid["survival_probability"], dtype=float),
        bandwidth_log10_a_dex=float(raw_grid["bandwidth_log10_a_dex"]),
        surface_model=surface_model,
    )
    summary = fit_payload["summary"]
    summary = SmoothSurvivabilityFitSummary(
        eta_t=float(eta_t),
        bandwidth_log10_a_dex=summary.bandwidth_log10_a_dex,
        outer_plateau_90_log10_msun_from_raw_boundary=summary.outer_plateau_90_log10_msun_from_raw_boundary,
        outer_level_90_log10_msun=summary.outer_level_90_log10_msun,
        outer_level_50_log10_msun=summary.outer_level_50_log10_msun,
        outer_level_10_log10_msun=summary.outer_level_10_log10_msun,
        inner_level_90_log10_msun=summary.inner_level_90_log10_msun,
        inner_level_50_log10_msun=summary.inner_level_50_log10_msun,
        inner_level_10_log10_msun=summary.inner_level_10_log10_msun,
        transition_log10_a_kpc=summary.transition_log10_a_kpc,
        transition_a_kpc=summary.transition_a_kpc,
        width_log10_a_dex=summary.width_log10_a_dex,
        transition_band_width_dex=summary.transition_band_width_dex,
        optimization_success=summary.optimization_success,
        optimizer_message=summary.optimizer_message,
        optimizer_value=summary.optimizer_value,
    )
    return {
        "log_mass_grid": np.asarray(raw_grid["log_mass_grid"], dtype=float),
        "log_a_grid": np.asarray(raw_grid["log_a_grid"], dtype=float),
        "semi_major_axis_grid_kpc": np.asarray(raw_grid["semi_major_axis_grid_kpc"], dtype=float),
        "survival_probability": np.asarray(fit_payload["fitted_probability"], dtype=float),
        "eta_t": float(eta_t),
        "surface_model": str(surface_model),
        "bandwidth_log10_a_dex": float(bandwidth_log10_a_dex),
        "raw_survival_probability": np.asarray(raw_grid["survival_probability"], dtype=float),
        "raw_boundary_10_log10_msun": np.asarray(fit_payload["raw_boundary_10_log10_msun"], dtype=float),
        "raw_boundary_50_log10_msun": np.asarray(fit_payload["raw_boundary_50_log10_msun"], dtype=float),
        "raw_boundary_80_log10_msun": np.asarray(fit_payload["raw_boundary_80_log10_msun"], dtype=float),
        "raw_boundary_90_log10_msun": np.asarray(fit_payload["raw_boundary_90_log10_msun"], dtype=float),
        "fitted_boundary_10_log10_msun": np.asarray(fit_payload["fitted_boundary_10_log10_msun"], dtype=float),
        "fitted_boundary_50_log10_msun": np.asarray(fit_payload["fitted_boundary_50_log10_msun"], dtype=float),
        "fitted_boundary_90_log10_msun": np.asarray(fit_payload["fitted_boundary_90_log10_msun"], dtype=float),
        "occupancy_table": fit_payload["occupancy_table"].copy(),
        "summary": summary,
    }
