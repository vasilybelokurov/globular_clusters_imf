from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import pickle
import sys
import warnings
from dataclasses import dataclass
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

from scan_schechter_survival_time_multipliers import _plot_logl_vs_multiplier, _plot_properties, _row_from_result


LOG_MASS_MIN = 4.0
N_DETECTABILITY_ITERATIONS = 12
SURFACE_MODEL = "logistic"


@dataclass(frozen=True)
class GridSpec:
    eta_min: float
    eta_max: float
    eta_n: int
    alpha_min: float
    alpha_max: float
    alpha_n: int
    logmc_min: float
    logmc_max: float
    logmc_n: int

    def eta_grid(self) -> np.ndarray:
        return np.linspace(self.eta_min, self.eta_max, self.eta_n)

    def alpha_grid(self) -> np.ndarray:
        return np.linspace(self.alpha_min, self.alpha_max, self.alpha_n)

    def logmc_grid(self) -> np.ndarray:
        return np.linspace(self.logmc_min, self.logmc_max, self.logmc_n)


def _round_key(theta: np.ndarray) -> tuple[float, float, float]:
    return tuple(round(float(value), 6) for value in theta)


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


def _start_state_from_result(result: dict[str, object]) -> dict[str, np.ndarray]:
    return {
        "completeness": np.asarray(result["final_completeness_raw_parameters"], dtype=float),
        "radial": np.asarray(result["final_payload"]["radial_parameters_raw"], dtype=float),
    }


def _failure_row(theta: np.ndarray, stage: str, message: str, radial_model: str) -> dict[str, object]:
    eta_t, alpha, log_mc = [float(value) for value in theta]
    return {
        "eta_t": eta_t,
        "radial_model": radial_model,
        "log_likelihood": -np.inf,
        "aic": np.inf,
        "bic": np.inf,
        "alpha_dndm": alpha,
        "log10_m_c_msun": log_mc,
        "baseline_total_initial_count": np.nan,
        "baseline_total_initial_count_above_log10_4": np.nan,
        "baseline_total_initial_stellar_mass_above_log10_4_msun": np.nan,
        "final_total_initial_count": np.nan,
        "final_total_initial_count_above_log10_4": np.nan,
        "final_total_initial_stellar_mass_above_log10_4_msun": np.nan,
        "raw_survival_fraction": np.nan,
        "selection_fraction": np.nan,
        "mean_detectability": np.nan,
        "raw_survival_fraction_above_log10_4": np.nan,
        "selection_fraction_above_log10_4": np.nan,
        "mean_detectability_above_log10_4": np.nan,
        "count_ratio_vs_baseline_above_log10_4": np.nan,
        "mass_ratio_vs_baseline_above_log10_4": np.nan,
        "survival_outer_level_90_log10_msun": np.nan,
        "survival_outer_level_50_log10_msun": np.nan,
        "survival_outer_level_10_log10_msun": np.nan,
        "survival_inner_level_90_log10_msun": np.nan,
        "survival_inner_level_50_log10_msun": np.nan,
        "survival_inner_level_10_log10_msun": np.nan,
        "survival_transition_a_kpc": np.nan,
        "survival_width_log10_a_dex": np.nan,
        "survival_transition_band_width_dex": np.nan,
        "beta_log10_a": np.nan,
        "gamma_linear_a": np.nan,
        "log10_a_core_kpc": np.nan,
        "a_core_kpc": np.nan,
        "input_alpha_dndm": alpha,
        "input_log10_m_c_msun": log_mc,
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


def _catalog_and_survival_grid_for_theta(
    *,
    prepared_catalog: pd.DataFrame,
    eta_t: float,
    survivability_backend: str,
    gg23_model_name: str | None,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    if survivability_backend == "baumgardt":
        from globular_clusters_imf.smooth_survivability import build_smooth_survivability_grid

        smooth_survival = build_smooth_survivability_grid(
            prepared_catalog,
            eta_t=eta_t,
            surface_model=SURFACE_MODEL,
        )
        metadata = {
            "survivability_backend": "baumgardt",
            "gg23_model_name": "",
            "gg23_model_label": "",
            "gg23_mini_eta_t_dependent": False,
            "max_abs_present_mass_residual_fraction": np.nan,
        }
        return prepared_catalog, smooth_survival, metadata

    if survivability_backend != "gg23":
        raise ValueError(f"Unknown survivability backend: {survivability_backend!r}")
    if not gg23_model_name:
        raise ValueError("--gg23-model is required when --survivability-backend=gg23")

    from globular_clusters_imf.gg23_survivability import (
        GG23_MODELS,
        build_gg23_survivability_grid,
        effective_radius_kpc_from_semimajor_axis,
        gg23_initial_mass_from_present_msun,
        gg23_present_mass_msun,
        gg23_survival_mass_cut_msun,
    )
    from globular_clusters_imf.model import AGE_GYR

    if gg23_model_name not in GG23_MODELS:
        raise ValueError(f"Unknown GG23 model {gg23_model_name!r}. Available: {sorted(GG23_MODELS)}")

    model = GG23_MODELS[gg23_model_name]
    working = prepared_catalog.copy()
    semi_major_axis = working["semi_major_axis_kpc"].to_numpy(dtype=float)
    eccentricity = working["eccentricity"].to_numpy(dtype=float)
    present_mass = working["present_mass_msun"].to_numpy(dtype=float)
    effective_radius = effective_radius_kpc_from_semimajor_axis(semi_major_axis, eccentricity)
    gg23_initial_mass = gg23_initial_mass_from_present_msun(
        present_mass,
        effective_radius,
        model,
        gradient_radius_kpc=semi_major_axis,
        age_gyr=AGE_GYR,
        eta_t=float(eta_t),
    )
    reconstructed_present_mass = gg23_present_mass_msun(
        gg23_initial_mass,
        effective_radius,
        model,
        gradient_radius_kpc=semi_major_axis,
        age_gyr=AGE_GYR,
        eta_t=float(eta_t),
    )
    survival_cut = gg23_survival_mass_cut_msun(
        effective_radius,
        model,
        gradient_radius_kpc=semi_major_axis,
        age_gyr=AGE_GYR,
        eta_t=float(eta_t),
    )
    valid = (
        np.isfinite(gg23_initial_mass)
        & np.isfinite(survival_cut)
        & (gg23_initial_mass > 0.0)
        & (survival_cut > 0.0)
        & np.isfinite(semi_major_axis)
        & (semi_major_axis > 0.0)
    )
    if not np.all(valid):
        working = working.loc[valid].copy()
        semi_major_axis = semi_major_axis[valid]
        present_mass = present_mass[valid]
        effective_radius = effective_radius[valid]
        gg23_initial_mass = gg23_initial_mass[valid]
        reconstructed_present_mass = reconstructed_present_mass[valid]
        survival_cut = survival_cut[valid]

    working["baumgardt_initial_mass_msun"] = working["initial_mass_msun"].to_numpy(dtype=float)
    working["baumgardt_log_initial_mass_msun"] = working["log_initial_mass_msun"].to_numpy(dtype=float)
    working["initial_mass_msun"] = gg23_initial_mass
    working["log_initial_mass_msun"] = np.log10(gg23_initial_mass)
    working["gg23_effective_radius_kpc"] = effective_radius
    working["gg23_model_name"] = gg23_model_name
    working["gg23_model_label"] = model.label
    working["gg23_survival_mass_cut_msun"] = survival_cut
    working["log_gg23_survival_mass_cut_msun"] = np.log10(survival_cut)
    working["log_survival_mass_cut_msun"] = np.log10(survival_cut)
    working["gg23_mass_loss_fraction"] = 1.0 - present_mass / gg23_initial_mass
    working["gg23_reconstructed_present_mass_msun"] = reconstructed_present_mass
    working["gg23_present_mass_residual_fraction"] = (reconstructed_present_mass - present_mass) / present_mass

    smooth_survival = build_gg23_survivability_grid(
        working,
        model,
        eta_t=float(eta_t),
        surface_model=SURFACE_MODEL,
    )
    metadata = {
        "survivability_backend": "gg23",
        "gg23_model_name": gg23_model_name,
        "gg23_model_label": model.label,
        "gg23_mini_eta_t_dependent": True,
        "max_abs_present_mass_residual_fraction": float(
            np.nanmax(np.abs(working["gg23_present_mass_residual_fraction"].to_numpy(dtype=float)))
        ),
    }
    return working, smooth_survival, metadata


def _evaluate_theta_single_start(
    *,
    prepared_catalog: pd.DataFrame,
    spec,
    theta: np.ndarray,
    start_state: dict[str, np.ndarray] | None,
    project_root: Path,
    survivability_backend: str,
    gg23_model_name: str | None,
) -> dict[str, object]:
    from globular_clusters_imf.detectability_longitude_model import fit_single_component_detectability_em_with_abs_longitude

    eta_t, alpha, log_mc = [float(value) for value in theta]
    working_catalog, smooth_survival, metadata = _catalog_and_survival_grid_for_theta(
        prepared_catalog=prepared_catalog,
        eta_t=eta_t,
        survivability_backend=survivability_backend,
        gg23_model_name=gg23_model_name,
    )
    result = fit_single_component_detectability_em_with_abs_longitude(
        working_catalog,
        project_root=project_root,
        spec=spec,
        n_iterations=N_DETECTABILITY_ITERATIONS,
        fixed_imf_params=np.array([alpha, log_mc], dtype=float),
        start_completeness_raw_parameters=None if start_state is None else start_state["completeness"],
        start_radial_params=None if start_state is None else start_state["radial"],
        survival_grid_override=_survival_grid_override_from_smooth_survival(smooth_survival),
    )
    row = _row_from_result(
        eta_t=eta_t,
        radial_model=spec.radial_model,
        survival_summary=smooth_survival["summary"],
        result=result,
        log_mass_min=LOG_MASS_MIN,
    )
    radial_params = result["final_payload"]["model"]["radial_parameters"]
    row["beta_log10_a"] = float(radial_params.get("beta_log10_a", np.nan))
    row["gamma_linear_a"] = float(radial_params.get("gamma_linear_a", np.nan))
    row["log10_a_core_kpc"] = float(radial_params.get("log10_a_core_kpc", np.nan))
    row["a_core_kpc"] = float(radial_params.get("a_core_kpc", np.nan))
    row["input_alpha_dndm"] = alpha
    row["input_log10_m_c_msun"] = log_mc
    row["surface_model"] = SURFACE_MODEL
    row.update(metadata)
    row["status"] = "ok"
    row["failure_message"] = ""
    return {
        "theta": np.asarray(theta, dtype=float),
        "log_posterior": float(row["log_likelihood"]),
        "row": row,
        "result": result,
        "start_state": _start_state_from_result(result),
    }


def _select_anchor_start_state(
    *,
    theta: np.ndarray,
    anchors: list[dict[str, object]],
    bounds: np.ndarray,
) -> dict[str, np.ndarray] | None:
    if len(anchors) == 0:
        return None
    theta = np.asarray(theta, dtype=float)
    widths = np.maximum(bounds[:, 1] - bounds[:, 0], 1.0e-12)
    best_anchor = min(
        anchors,
        key=lambda anchor: float(
            np.sum(np.square((theta - np.asarray(anchor["theta"], dtype=float)) / widths))
        ),
    )
    return best_anchor["start_state"]


def _evaluate_theta_multistart(
    *,
    prepared_catalog: pd.DataFrame,
    spec,
    theta: np.ndarray,
    stage: str,
    project_root: Path,
    anchor_start_state: dict[str, np.ndarray] | None,
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
            "row": {
                **_failure_row(theta, stage=stage, message=" | ".join(failure_messages), radial_model=spec.radial_model),
                "survivability_backend": survivability_backend,
                "gg23_model_name": "" if gg23_model_name is None else str(gg23_model_name),
            },
            "result": None,
            "start_state": None,
        }

    best_entry["row"]["stage"] = stage
    return best_entry


def _entry_stage_copy(entry: dict[str, object], stage: str) -> dict[str, object]:
    row = dict(entry["row"])
    row["stage"] = stage
    return {
        "theta": np.asarray(entry["theta"], dtype=float).copy(),
        "log_posterior": float(entry["log_posterior"]),
        "row": row,
        "result": entry["result"],
        "start_state": entry["start_state"],
    }


def _best_by_eta(table: pd.DataFrame, radial_model: str) -> pd.DataFrame:
    good = table.loc[np.isfinite(table["log_likelihood"])].copy()
    best_idx = good.groupby("eta_t")["log_likelihood"].idxmax()
    best_table = good.loc[best_idx].sort_values("eta_t").reset_index(drop=True)
    best_table["best_radial_model"] = radial_model
    return best_table


def _save_best_payload(entry: dict[str, object], output_tables: Path, prefix: str) -> None:
    if entry["result"] is None:
        return
    with (output_tables / f"{prefix}_best_result.pkl").open("wb") as handle:
        pickle.dump(entry["result"], handle, protocol=pickle.HIGHEST_PROTOCOL)
    (output_tables / f"{prefix}_best_result_summary.json").write_text(json.dumps(entry["row"], indent=2))


def _grid_step(grid: np.ndarray) -> float:
    grid = np.asarray(grid, dtype=float)
    if len(grid) < 2:
        return 1.0
    return float(np.median(np.diff(grid)))


def _select_high_likelihood_coarse_rows(
    table: pd.DataFrame,
    *,
    delta_logl: float,
    min_points: int,
) -> pd.DataFrame:
    good = table.loc[np.isfinite(table["log_likelihood"])].copy()
    best_logl = float(good["log_likelihood"].max())
    selected = good.loc[good["log_likelihood"] >= best_logl - float(delta_logl)].copy()
    if len(selected) < int(min_points):
        selected = good.sort_values("log_likelihood", ascending=False).head(int(min_points)).copy()
    return selected.sort_values("log_likelihood", ascending=False).reset_index(drop=True)


def _expanded_axis_bounds(
    selected_values: np.ndarray,
    coarse_grid: np.ndarray,
    *,
    global_lower: float,
    global_upper: float,
    padding_steps: float,
) -> tuple[float, float]:
    step = _grid_step(coarse_grid)
    lower = max(float(global_lower), float(np.min(selected_values) - padding_steps * step))
    upper = min(float(global_upper), float(np.max(selected_values) + padding_steps * step))
    if not lower < upper:
        lower = max(float(global_lower), float(np.min(selected_values) - step))
        upper = min(float(global_upper), float(np.max(selected_values) + step))
    return float(lower), float(upper)


def _build_refined_spec_from_coarse_region(
    coarse_spec: GridSpec,
    selected_rows: pd.DataFrame,
    *,
    local_eta_n: int,
    local_alpha_n: int,
    local_logmc_n: int,
    padding_steps: float,
) -> GridSpec:
    coarse_eta = coarse_spec.eta_grid()
    coarse_alpha = coarse_spec.alpha_grid()
    coarse_logmc = coarse_spec.logmc_grid()
    eta_min, eta_max = _expanded_axis_bounds(
        np.asarray(selected_rows["eta_t"], dtype=float),
        coarse_eta,
        global_lower=coarse_spec.eta_min,
        global_upper=coarse_spec.eta_max,
        padding_steps=padding_steps,
    )
    alpha_min, alpha_max = _expanded_axis_bounds(
        np.asarray(selected_rows["input_alpha_dndm"], dtype=float),
        coarse_alpha,
        global_lower=coarse_spec.alpha_min,
        global_upper=coarse_spec.alpha_max,
        padding_steps=padding_steps,
    )
    logmc_min, logmc_max = _expanded_axis_bounds(
        np.asarray(selected_rows["input_log10_m_c_msun"], dtype=float),
        coarse_logmc,
        global_lower=coarse_spec.logmc_min,
        global_upper=coarse_spec.logmc_max,
        padding_steps=padding_steps,
    )
    return GridSpec(
        eta_min=eta_min,
        eta_max=eta_max,
        eta_n=local_eta_n,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_n=local_alpha_n,
        logmc_min=logmc_min,
        logmc_max=logmc_max,
        logmc_n=local_logmc_n,
    )


def _best_entry_on_edge(entry: dict[str, object], spec: GridSpec) -> dict[str, bool]:
    theta = np.asarray(entry["theta"], dtype=float)
    eta_grid = spec.eta_grid()
    alpha_grid = spec.alpha_grid()
    logmc_grid = spec.logmc_grid()
    return {
        "eta_low": bool(np.isclose(theta[0], eta_grid[0])),
        "eta_high": bool(np.isclose(theta[0], eta_grid[-1])),
        "alpha_low": bool(np.isclose(theta[1], alpha_grid[0])),
        "alpha_high": bool(np.isclose(theta[1], alpha_grid[-1])),
        "logmc_low": bool(np.isclose(theta[2], logmc_grid[0])),
        "logmc_high": bool(np.isclose(theta[2], logmc_grid[-1])),
    }


def _expand_refined_spec(
    spec: GridSpec,
    edge_flags: dict[str, bool],
    global_bounds: np.ndarray,
    *,
    expand_steps: float,
) -> tuple[GridSpec, bool]:
    eta_step = _grid_step(spec.eta_grid())
    alpha_step = _grid_step(spec.alpha_grid())
    logmc_step = _grid_step(spec.logmc_grid())

    eta_min, eta_max = spec.eta_min, spec.eta_max
    alpha_min, alpha_max = spec.alpha_min, spec.alpha_max
    logmc_min, logmc_max = spec.logmc_min, spec.logmc_max

    if edge_flags["eta_low"]:
        eta_min = max(float(global_bounds[0, 0]), float(spec.eta_min - expand_steps * eta_step))
    if edge_flags["eta_high"]:
        eta_max = min(float(global_bounds[0, 1]), float(spec.eta_max + expand_steps * eta_step))
    if edge_flags["alpha_low"]:
        alpha_min = max(float(global_bounds[1, 0]), float(spec.alpha_min - expand_steps * alpha_step))
    if edge_flags["alpha_high"]:
        alpha_max = min(float(global_bounds[1, 1]), float(spec.alpha_max + expand_steps * alpha_step))
    if edge_flags["logmc_low"]:
        logmc_min = max(float(global_bounds[2, 0]), float(spec.logmc_min - expand_steps * logmc_step))
    if edge_flags["logmc_high"]:
        logmc_max = min(float(global_bounds[2, 1]), float(spec.logmc_max + expand_steps * logmc_step))

    changed = not (
        np.isclose(eta_min, spec.eta_min)
        and np.isclose(eta_max, spec.eta_max)
        and np.isclose(alpha_min, spec.alpha_min)
        and np.isclose(alpha_max, spec.alpha_max)
        and np.isclose(logmc_min, spec.logmc_min)
        and np.isclose(logmc_max, spec.logmc_max)
    )
    return (
        GridSpec(
            eta_min=float(eta_min),
            eta_max=float(eta_max),
            eta_n=spec.eta_n,
            alpha_min=float(alpha_min),
            alpha_max=float(alpha_max),
            alpha_n=spec.alpha_n,
            logmc_min=float(logmc_min),
            logmc_max=float(logmc_max),
            logmc_n=spec.logmc_n,
        ),
        changed,
    )


def _build_anchor_library(
    coarse_entries: list[dict[str, object]],
    refined_entries: list[dict[str, object]],
    *,
    k: int,
) -> list[dict[str, object]]:
    merged: dict[tuple[float, float, float], dict[str, object]] = {}
    for entry in coarse_entries + refined_entries:
        if not np.isfinite(entry["log_posterior"]):
            continue
        merged[_round_key(np.asarray(entry["theta"], dtype=float))] = entry
    ordered = sorted(merged.values(), key=lambda entry: float(entry["log_posterior"]), reverse=True)
    return ordered[: max(1, int(k))]


def _normalized_distance(theta_a: np.ndarray, theta_b: np.ndarray, bounds: np.ndarray) -> float:
    widths = np.maximum(bounds[:, 1] - bounds[:, 0], 1.0e-12)
    return float(np.sqrt(np.sum(np.square((theta_a - theta_b) / widths))))


def _select_diverse_entries(
    entries: list[dict[str, object]],
    *,
    n_select: int,
    bounds: np.ndarray,
    candidate_pool: int = 80,
) -> list[dict[str, object]]:
    ordered = sorted(entries, key=lambda entry: float(entry["log_posterior"]), reverse=True)
    pool = ordered[: min(len(ordered), max(int(candidate_pool), int(n_select)))]
    if not pool:
        return []
    selected = [pool[0]]
    remaining = pool[1:]
    while remaining and len(selected) < int(n_select):
        best_candidate = None
        best_score = -np.inf
        for candidate in remaining:
            theta = np.asarray(candidate["theta"], dtype=float)
            score = min(
                _normalized_distance(theta, np.asarray(chosen["theta"], dtype=float), bounds)
                for chosen in selected
            )
            if score > best_score:
                best_score = score
                best_candidate = candidate
        if best_candidate is None:
            break
        selected.append(best_candidate)
        remaining = [entry for entry in remaining if entry is not best_candidate]
    for entry in ordered:
        if len(selected) >= int(n_select):
            break
        if all(entry is not chosen for chosen in selected):
            selected.append(entry)
    return selected[: int(n_select)]


def _within_bounds(theta: np.ndarray, bounds: np.ndarray) -> bool:
    theta = np.asarray(theta, dtype=float)
    return bool(np.all(theta >= bounds[:, 0]) and np.all(theta <= bounds[:, 1]))


def _compute_rhat(chains: np.ndarray) -> float:
    m, n = chains.shape
    if m < 2 or n < 2:
        return float("nan")
    chain_means = np.mean(chains, axis=1)
    chain_vars = np.var(chains, axis=1, ddof=1)
    within = np.mean(chain_vars)
    between = n * np.var(chain_means, ddof=1)
    if within <= 0.0:
        return float("nan")
    var_hat = ((n - 1) / n) * within + between / n
    return float(np.sqrt(var_hat / within))


def _surface_payload_from_result(result: dict[str, object]) -> dict[str, np.ndarray]:
    context = result["final_context"]
    return {
        "log_mass_grid": np.asarray(context.log_mass_grid, dtype=np.float32),
        "log_a_grid": np.asarray(context.log_a_grid, dtype=np.float32),
        "survival_probability": np.asarray(context.survival_probability_grid, dtype=np.float32),
        "effective_detectability": np.asarray(result["final_effective_completeness_grid"], dtype=np.float32),
    }


def _lightweight_entry(entry: dict[str, object], *, include_surfaces: bool = False) -> dict[str, object]:
    lightweight = {
        "theta": np.asarray(entry["theta"], dtype=float).copy(),
        "log_posterior": float(entry["log_posterior"]),
        "row": dict(entry["row"]),
        "start_state": None
        if entry.get("start_state") is None
        else {
            "completeness": np.asarray(entry["start_state"]["completeness"], dtype=float).copy(),
                "radial": np.asarray(entry["start_state"]["radial"], dtype=float).copy(),
            },
    }
    if include_surfaces:
        if "surfaces" in entry:
            lightweight["surfaces"] = {
                key: np.asarray(value, dtype=np.float32).copy()
                for key, value in entry["surfaces"].items()
            }
        elif entry.get("result") is not None:
            lightweight["surfaces"] = _surface_payload_from_result(entry["result"])
    return lightweight


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
    survivability_backend: str,
    gg23_model_name: str | None,
    surface_output_path: str | None = None,
    surface_burn_in: int = 0,
    surface_thin: int = 1,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    proposal_scales = 0.08 * np.asarray(widths, dtype=float)
    current = _lightweight_entry(initial_entry, include_surfaces=True)
    current_surface = current.get("surfaces")
    local_cache: dict[tuple[float, float, float], dict[str, object]] = {}
    local_cache[_round_key(current["theta"])] = current
    for anchor in fixed_anchor_library:
        local_cache[_round_key(np.asarray(anchor["theta"], dtype=float))] = _lightweight_entry(anchor)

    accepts = np.zeros(n_steps, dtype=bool)
    records: list[dict[str, object]] = []
    surface_rows: list[dict[str, object]] = []
    survival_surfaces: list[np.ndarray] = []
    detectability_surfaces: list[np.ndarray] = []
    log_mass_grid: np.ndarray | None = None
    log_a_grid: np.ndarray | None = None

    for step in range(n_steps):
        theta_prop = np.asarray(current["theta"], dtype=float) + rng.normal(scale=proposal_scales, size=3)
        accepted = False
        proposal_surface = None
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
                    survivability_backend=survivability_backend,
                    gg23_model_name=gg23_model_name,
                )
                proposal_surface = _surface_payload_from_result(proposal_exact["result"]) if proposal_exact.get("result") is not None else None
                proposal_entry = _lightweight_entry(proposal_exact)
                proposal_entry["row"]["stage"] = "mcmc"
                local_cache[key] = proposal_entry
            if np.isfinite(proposal_entry["log_posterior"]):
                delta = float(proposal_entry["log_posterior"]) - float(current["log_posterior"])
                if np.log(rng.uniform()) < min(0.0, delta):
                    current = proposal_entry
                    current_surface = proposal_surface
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

        if (
            surface_output_path is not None
            and step >= int(surface_burn_in)
            and (step - int(surface_burn_in)) % max(int(surface_thin), 1) == 0
        ):
            if current_surface is None:
                surface_entry = _evaluate_theta_multistart(
                    prepared_catalog=prepared_catalog,
                    spec=spec,
                    theta=np.asarray(current["theta"], dtype=float),
                    stage="surface_archive",
                    project_root=project_root,
                    anchor_start_state=current.get("start_state"),
                    survivability_backend=survivability_backend,
                    gg23_model_name=gg23_model_name,
                )
                current_surface = _surface_payload_from_result(surface_entry["result"]) if surface_entry.get("result") is not None else None
            if current_surface is not None:
                if log_mass_grid is None:
                    log_mass_grid = np.asarray(current_surface["log_mass_grid"], dtype=np.float32)
                    log_a_grid = np.asarray(current_surface["log_a_grid"], dtype=np.float32)
                survival_surfaces.append(np.asarray(current_surface["survival_probability"], dtype=np.float32))
                detectability_surfaces.append(np.asarray(current_surface["effective_detectability"], dtype=np.float32))
                surface_rows.append(row)

        if step + 1 <= adapt_until and (step + 1) % adapt_every == 0:
            window = float(accepts[step + 1 - adapt_every : step + 1].mean())
            if window < 0.15:
                proposal_scales *= 0.85
            elif window > 0.35:
                proposal_scales *= 1.15
            proposal_scales = np.clip(
                proposal_scales,
                0.01 * widths,
                0.30 * widths,
            )

    best_row = max(records, key=lambda row: float(row["log_likelihood"]))
    surface_path = None
    if surface_output_path is not None and len(surface_rows) > 0:
        surface_path_obj = Path(surface_output_path)
        surface_path_obj.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            surface_path_obj,
            log_mass_grid=np.asarray(log_mass_grid, dtype=np.float32),
            log_a_grid=np.asarray(log_a_grid, dtype=np.float32),
            survival_probability=np.asarray(survival_surfaces, dtype=np.float32),
            effective_detectability=np.asarray(detectability_surfaces, dtype=np.float32),
        )
        pd.DataFrame(surface_rows).to_csv(surface_path_obj.with_suffix(".csv"), index=False)
        surface_path = str(surface_path_obj)
    return {
        "chain_id": int(chain_id),
        "records": records,
        "acceptance": float(accepts.mean()),
        "best_row": best_row,
        "final_row": records[-1],
        "cache_size": int(len(local_cache)),
        "surface_path": surface_path,
        "n_surface_records": int(len(surface_rows)),
    }


def _corner_plot(samples: pd.DataFrame, reference_row: dict[str, object], output_path: Path) -> None:
    candidate_columns = [
        ("eta_t", r"$\eta_t$"),
        ("input_alpha_dndm", r"$\alpha$"),
        ("input_log10_m_c_msun", r"$\log_{10}(M_c/{\rm M}_\odot)$"),
        ("gamma_linear_a", r"$\gamma_a$"),
        ("log10_a_core_kpc", r"$\log_{10}(a_c/{\rm kpc})$"),
    ]
    columns = []
    for name, label in candidate_columns:
        if name not in samples.columns:
            continue
        values = np.asarray(samples[name], dtype=float)
        if not np.isfinite(values).any():
            continue
        columns.append((name, label))
    n_dim = len(columns)
    if n_dim == 0:
        return
    fig, axes = plt.subplots(n_dim, n_dim, figsize=(10.5, 10.5))
    if n_dim == 1:
        axes = np.asarray([[axes]])
    for i, (y_name, y_label) in enumerate(columns):
        for j, (x_name, x_label) in enumerate(columns):
            ax = axes[i, j]
            if i < j:
                ax.axis("off")
                continue
            if i == j:
                values = np.asarray(samples[x_name], dtype=float)
                ax.hist(values, bins=30, color="#4c72b0", alpha=0.85, histtype="stepfilled")
                q16, q50, q84 = np.quantile(values, [0.16, 0.50, 0.84])
                ax.axvline(q16, color="0.5", linestyle="--", linewidth=0.8)
                ax.axvline(q50, color="black", linewidth=1.0)
                ax.axvline(q84, color="0.5", linestyle="--", linewidth=0.8)
                ax.axvline(float(reference_row[x_name]), color="#d62728", linewidth=1.0)
            else:
                ax.hist2d(
                    np.asarray(samples[x_name], dtype=float),
                    np.asarray(samples[y_name], dtype=float),
                    bins=30,
                    cmap="Blues",
                )
                ax.plot(
                    float(reference_row[x_name]),
                    float(reference_row[y_name]),
                    marker="x",
                    color="#d62728",
                    markersize=6,
                )
            if i == n_dim - 1:
                ax.set_xlabel(x_label)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel("Count" if i == j == 0 else y_label)
            else:
                ax.set_yticklabels([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _trace_plot(chain_table: pd.DataFrame, output_path: Path, burn_in: int) -> None:
    candidate_columns = [
        ("eta_t", r"$\eta_t$"),
        ("input_alpha_dndm", r"$\alpha$"),
        ("input_log10_m_c_msun", r"$\log_{10} M_c$"),
        ("gamma_linear_a", r"$\gamma_a$"),
        ("log10_a_core_kpc", r"$\log_{10} a_c$"),
    ]
    columns = []
    for name, label in candidate_columns:
        if name not in chain_table.columns:
            continue
        values = np.asarray(chain_table[name], dtype=float)
        if not np.isfinite(values).any():
            continue
        columns.append((name, label))
    if not columns:
        return
    fig, axes = plt.subplots(len(columns), 1, figsize=(9.0, 8.5), sharex=True)
    if len(columns) == 1:
        axes = np.asarray([axes])
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2", "#937860"]
    for ax, (column, label) in zip(axes, columns):
        for color, chain_id in zip(colors, sorted(chain_table["chain"].unique())):
            subset = chain_table.loc[chain_table["chain"] == chain_id]
            ax.plot(subset["step"], subset[column], color=color, linewidth=0.9, alpha=0.8)
        ax.axvline(burn_in, color="0.6", linestyle="--", linewidth=1.0)
        ax.set_ylabel(label)
    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root-name", default="profile_map_and_exact_mcmc_schechter_powerlaw_a_logistic")
    parser.add_argument(
        "--survivability-backend",
        default="baumgardt",
        choices=["baumgardt", "gg23"],
        help="Dynamical survivability backend. GG23 also recomputes catalogue M_ini at each eta_t.",
    )
    parser.add_argument(
        "--gg23-model",
        default="",
        choices=[
            "",
            "gg23_no_bh",
            "gg23_bh",
            "gg23_bh_feh_gradient",
            "gg23_bh_past_tidal",
            "gg23_bh_feh_gradient_past_tidal",
        ],
        help="GG23 disruption variant used when --survivability-backend=gg23.",
    )
    parser.add_argument(
        "--radial-model",
        default="powerlaw_a",
        choices=["powerlaw_a", "cored_powerlaw_a", "logpoly3", "step5"],
    )
    parser.add_argument("--coarse-eta-min", type=float, default=0.6)
    parser.add_argument("--coarse-eta-max", type=float, default=2.4)
    parser.add_argument("--coarse-eta-n", type=int, default=7)
    parser.add_argument("--coarse-alpha-min", type=float, default=-1.8)
    parser.add_argument("--coarse-alpha-max", type=float, default=-0.4)
    parser.add_argument("--coarse-alpha-n", type=int, default=8)
    parser.add_argument("--coarse-logmc-min", type=float, default=6.0)
    parser.add_argument("--coarse-logmc-max", type=float, default=6.6)
    parser.add_argument("--coarse-logmc-n", type=int, default=5)
    parser.add_argument("--refine-delta-logl", type=float, default=3.0)
    parser.add_argument("--refine-min-points", type=int, default=10)
    parser.add_argument("--refine-padding-steps", type=float, default=1.0)
    parser.add_argument("--local-eta-n", type=int, default=9)
    parser.add_argument("--local-alpha-n", type=int, default=9)
    parser.add_argument("--local-logmc-n", type=int, default=7)
    parser.add_argument("--local-max-passes", type=int, default=3)
    parser.add_argument("--local-expand-steps", type=float, default=1.0)
    parser.add_argument("--anchor-k", type=int, default=12)
    parser.add_argument("--skip-mcmc", action="store_true")
    parser.add_argument("--mcmc-chains", type=int, default=6)
    parser.add_argument("--mcmc-workers", type=int, default=0)
    parser.add_argument("--mcmc-steps", type=int, default=260)
    parser.add_argument("--mcmc-burn", type=int, default=80)
    parser.add_argument("--mcmc-thin", type=int, default=2)
    parser.add_argument("--mcmc-adapt-until", type=int, default=120)
    parser.add_argument("--mcmc-adapt-every", type=int, default=20)
    parser.add_argument("--mcmc-seed", type=int, default=20260527)
    args = parser.parse_args()
    if str(args.survivability_backend) == "gg23" and not str(args.gg23_model):
        parser.error("--gg23-model is required when --survivability-backend=gg23")
    if str(args.survivability_backend) != "gg23" and str(args.gg23_model):
        parser.error("--gg23-model can only be used with --survivability-backend=gg23")

    output_root = PROJECT_ROOT / "variants" / args.output_root_name
    figures_dir = output_root / "outputs" / "figures"
    tables_dir = output_root / "outputs" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    from globular_clusters_imf.joint_model import JointModelSpec
    from globular_clusters_imf.model import fit_catalog_models

    rng = np.random.default_rng(20260527)
    spec = JointModelSpec(imf_family="schechter", radial_model=str(args.radial_model))

    catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    prepared_catalog = fit_catalog_models(catalog, output_root)["catalog"]

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
    coarse_eta = coarse_spec.eta_grid()
    coarse_alpha = coarse_spec.alpha_grid()
    coarse_logmc = coarse_spec.logmc_grid()
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
    for eta_t in coarse_eta:
        for log_mc in coarse_logmc:
            for alpha in coarse_alpha:
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
                        survivability_backend=str(args.survivability_backend),
                        gg23_model_name=str(args.gg23_model) or None,
                    )
                    evaluation_cache[key] = entry
                coarse_entries.append(entry)
                print(
                    f"[coarse] eta_t={eta_t:.3f} alpha={alpha:.3f} logMc={log_mc:.3f} "
                    f"logL={float(entry['row']['log_likelihood']):.3f} gamma_a={float(entry['row']['gamma_linear_a']):.3f} "
                    f"N0>1e4={float(entry['row']['final_total_initial_count_above_log10_4']):.1f}"
                )

    coarse_table = pd.DataFrame([entry["row"] for entry in coarse_entries]).sort_values(
        ["eta_t", "input_alpha_dndm", "input_log10_m_c_msun"]
    ).reset_index(drop=True)
    coarse_table.to_csv(tables_dir / "coarse_grid_results.csv", index=False)
    coarse_best_by_eta = _best_by_eta(coarse_table, spec.radial_model)
    coarse_best_by_eta.to_csv(tables_dir / "coarse_profiled_eta_results.csv", index=False)
    _plot_logl_vs_multiplier(coarse_best_by_eta, figures_dir / "coarse_profiled_logl_vs_eta_t.png")
    _plot_properties(coarse_best_by_eta, figures_dir / "coarse_profiled_properties_vs_eta_t.png")

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
                            survivability_backend=str(args.survivability_backend),
                            gg23_model_name=str(args.gg23_model) or None,
                        )
                        evaluation_cache[key] = entry
                    current_entries.append(entry)
                    print(
                        f"[refined {pass_index + 1}] eta_t={eta_t:.3f} alpha={alpha:.3f} logMc={log_mc:.3f} "
                        f"logL={float(entry['row']['log_likelihood']):.3f} gamma_a={float(entry['row']['gamma_linear_a']):.3f} "
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
    refined_best_by_eta = _best_by_eta(refined_table, spec.radial_model)
    refined_best_by_eta.to_csv(tables_dir / "refined_profiled_eta_results.csv", index=False)
    _plot_logl_vs_multiplier(refined_best_by_eta, figures_dir / "refined_profiled_logl_vs_eta_t.png")
    _plot_properties(refined_best_by_eta, figures_dir / "refined_profiled_properties_vs_eta_t.png")
    _save_best_payload(refined_best_entry, tables_dir, prefix="refined")
    _save_best_payload(refined_best_entry, tables_dir, prefix="best")

    if bool(args.skip_mcmc):
        summary_payload = {
            "surface_model": SURFACE_MODEL,
            "survivability_backend": str(args.survivability_backend),
            "gg23_model_name": str(args.gg23_model),
            "gg23_mini_eta_t_dependent": bool(str(args.survivability_backend) == "gg23"),
            "model_spec": {"imf_family": spec.imf_family, "radial_model": spec.radial_model},
            "n_detectability_iterations": N_DETECTABILITY_ITERATIONS,
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

    n_chains = int(args.mcmc_chains)
    n_steps = int(args.mcmc_steps)
    burn_in = int(args.mcmc_burn)
    thin = int(args.mcmc_thin)
    adapt_until = int(args.mcmc_adapt_until)
    adapt_every = int(args.mcmc_adapt_every)
    mcmc_seed = int(args.mcmc_seed)
    n_workers = int(args.mcmc_workers) if int(args.mcmc_workers) > 0 else n_chains
    widths = refined_bounds[:, 1] - refined_bounds[:, 0]

    parallel_inputs = [
        _lightweight_entry(entry)
        for entry in current_states[:n_chains]
    ]
    fixed_anchor_library_light = [
        _lightweight_entry(entry)
        for entry in fixed_anchor_library
    ]

    chain_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_chain = {
            executor.submit(
                _run_exact_mcmc_chain_worker,
                chain_id=chain_id,
                n_steps=n_steps,
                adapt_until=adapt_until,
                adapt_every=adapt_every,
                seed=mcmc_seed + chain_id,
                prepared_catalog=prepared_catalog,
                spec=spec,
                project_root=output_root,
                bounds=refined_bounds,
                widths=widths,
                initial_entry=parallel_inputs[chain_id],
                fixed_anchor_library=fixed_anchor_library_light,
                survivability_backend=str(args.survivability_backend),
                gg23_model_name=str(args.gg23_model) or None,
            ): chain_id
            for chain_id in range(n_chains)
        }
        for future in as_completed(future_to_chain):
            chain_id = future_to_chain[future]
            result = future.result()
            chain_results.append(result)
            best_row = result["best_row"]
            print(
                f"[exact-mcmc parallel] chain={chain_id} done "
                f"accept={float(result['acceptance']):.3f} "
                f"best logL={float(best_row['log_likelihood']):.3f} "
                f"eta_t={float(best_row['eta_t']):.3f} "
                f"alpha={float(best_row['input_alpha_dndm']):.3f} "
                f"logMc={float(best_row['input_log10_m_c_msun']):.3f}"
            )

    chain_results.sort(key=lambda item: int(item["chain_id"]))
    records = [row for result in chain_results for row in result["records"]]
    chain_table = pd.DataFrame(records).sort_values(["chain", "step"]).reset_index(drop=True)
    chain_table.to_csv(tables_dir / "exact_mcmc_chain.csv", index=False)
    posterior_parts = []
    for _, frame in chain_table.loc[chain_table["step"] >= burn_in].groupby("chain"):
        posterior_parts.append(frame.iloc[::thin])
    posterior_table = pd.concat(posterior_parts, ignore_index=True)
    posterior_table.to_csv(tables_dir / "exact_mcmc_posterior_samples.csv", index=False)

    posterior_summary_rows = []
    for column in [
        "eta_t",
        "input_alpha_dndm",
        "input_log10_m_c_msun",
        "gamma_linear_a",
        "log10_a_core_kpc",
        "final_total_initial_count_above_log10_4",
        "final_total_initial_stellar_mass_above_log10_4_msun",
        "mean_detectability_above_log10_4",
        "log_likelihood",
    ]:
        if column not in posterior_table.columns:
            continue
        values = np.asarray(posterior_table[column], dtype=float)
        if not np.isfinite(values).any():
            continue
        q16, q50, q84 = np.quantile(values, [0.16, 0.50, 0.84])
        posterior_summary_rows.append(
            {
                "parameter": column,
                "q16": float(q16),
                "q50": float(q50),
                "q84": float(q84),
                "minus": float(q50 - q16),
                "plus": float(q84 - q50),
            }
        )
    posterior_summary_table = pd.DataFrame(posterior_summary_rows)
    posterior_summary_table.to_csv(tables_dir / "exact_posterior_summary.csv", index=False)

    rhat = {}
    for column in ["eta_t", "input_alpha_dndm", "input_log10_m_c_msun", "gamma_linear_a", "log10_a_core_kpc"]:
        if column not in posterior_table.columns:
            continue
        if not np.isfinite(np.asarray(posterior_table[column], dtype=float)).any():
            continue
        pivot = (
            posterior_table.pivot_table(index="step", columns="chain", values=column, aggfunc="last")
            .dropna()
            .to_numpy()
            .T
        )
        rhat[column] = _compute_rhat(pivot)
    acceptance_by_chain = {str(result["chain_id"]): float(result["acceptance"]) for result in chain_results}

    best_posterior_row = posterior_table.sort_values("log_likelihood", ascending=False).iloc[0].to_dict()
    _corner_plot(posterior_table, refined_best_entry["row"], figures_dir / "exact_profiled_posterior_corner.png")
    _trace_plot(chain_table, figures_dir / "exact_profiled_posterior_traces.png", burn_in=burn_in)

    best_mcmc_theta = np.array(
        [
            float(best_posterior_row["eta_t"]),
            float(best_posterior_row["input_alpha_dndm"]),
            float(best_posterior_row["input_log10_m_c_msun"]),
        ],
        dtype=float,
    )
    best_mcmc_anchor = _select_anchor_start_state(theta=best_mcmc_theta, anchors=fixed_anchor_library_light, bounds=refined_bounds)
    best_mcmc_entry = _evaluate_theta_multistart(
        prepared_catalog=prepared_catalog,
        spec=spec,
        theta=best_mcmc_theta,
        stage="mcmc_best",
        project_root=output_root,
        anchor_start_state=best_mcmc_anchor,
        survivability_backend=str(args.survivability_backend),
        gg23_model_name=str(args.gg23_model) or None,
    )
    _save_best_payload(best_mcmc_entry, tables_dir, prefix="mcmc")

    summary_payload = {
        "surface_model": SURFACE_MODEL,
        "survivability_backend": str(args.survivability_backend),
        "gg23_model_name": str(args.gg23_model),
        "gg23_mini_eta_t_dependent": bool(str(args.survivability_backend) == "gg23"),
        "model_spec": {"imf_family": spec.imf_family, "radial_model": spec.radial_model},
        "n_detectability_iterations": N_DETECTABILITY_ITERATIONS,
        "coarse_grid_spec": coarse_spec.__dict__,
        "coarse_best": json.loads(pd.Series(coarse_best_entry["row"]).to_json()),
        "refine_delta_logl": float(args.refine_delta_logl),
        "refine_min_points": int(args.refine_min_points),
        "refine_padding_steps": float(args.refine_padding_steps),
        "coarse_high_likelihood_point_count": int(len(selected_coarse)),
        "refined_grid_spec": refined_spec.__dict__,
        "refined_passes": refined_pass_summaries,
        "refined_best": json.loads(pd.Series(refined_best_entry["row"]).to_json()),
        "mcmc": {
            "sampler": "exact_profiled_random_walk_metropolis_parallel",
            "n_chains": n_chains,
            "n_workers": n_workers,
            "n_steps": n_steps,
            "burn_in": burn_in,
            "thin": thin,
            "acceptance_by_chain": acceptance_by_chain,
            "rhat": rhat,
            "best_posterior_sample": json.loads(pd.Series(best_posterior_row).to_json()),
            "posterior_summary": posterior_summary_table.to_dict(orient="records"),
        },
        "n_unique_profile_evaluations_parent_process": int(len(evaluation_cache)),
        "mcmc_worker_cache_sizes": {str(result["chain_id"]): int(result["cache_size"]) for result in chain_results},
    }
    (tables_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2))

    print(figures_dir / "coarse_profiled_logl_vs_eta_t.png")
    print(figures_dir / "refined_profiled_logl_vs_eta_t.png")
    print(figures_dir / "exact_profiled_posterior_corner.png")
    print(figures_dir / "exact_profiled_posterior_traces.png")
    print(tables_dir / "exact_posterior_summary.csv")
    print(tables_dir / "summary.json")


if __name__ == "__main__":
    main()
