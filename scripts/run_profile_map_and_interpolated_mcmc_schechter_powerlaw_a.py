from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

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


def _failure_row(theta: np.ndarray, stage: str, message: str) -> dict[str, object]:
    eta_t, alpha, log_mc = [float(value) for value in theta]
    return {
        "eta_t": eta_t,
        "radial_model": "powerlaw_a",
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
        "input_alpha_dndm": alpha,
        "input_log10_m_c_msun": log_mc,
        "surface_model": SURFACE_MODEL,
        "stage": stage,
        "status": "failed",
        "failure_message": message,
    }


def _evaluate_theta_single_start(
    *,
    prepared_catalog: pd.DataFrame,
    spec,
    theta: np.ndarray,
    start_state: dict[str, np.ndarray] | None,
    project_root: Path,
) -> dict[str, object]:
    from globular_clusters_imf.detectability_longitude_model import fit_single_component_detectability_em_with_abs_longitude
    from globular_clusters_imf.smooth_survivability import build_smooth_survivability_grid

    eta_t, alpha, log_mc = [float(value) for value in theta]
    smooth_survival = build_smooth_survivability_grid(
        prepared_catalog,
        eta_t=eta_t,
        surface_model=SURFACE_MODEL,
    )
    result = fit_single_component_detectability_em_with_abs_longitude(
        prepared_catalog,
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
    row["input_alpha_dndm"] = alpha
    row["input_log10_m_c_msun"] = log_mc
    row["surface_model"] = SURFACE_MODEL
    row["status"] = "ok"
    row["failure_message"] = ""
    return {
        "theta": np.asarray(theta, dtype=float),
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
) -> dict[str, object]:
    start_candidates = [None]
    if anchor_start_state is not None:
        start_candidates.append(anchor_start_state)

    best_entry = None
    best_logl = -np.inf
    failure_messages: list[str] = []
    for start_state in start_candidates:
        try:
            entry = _evaluate_theta_single_start(
                prepared_catalog=prepared_catalog,
                spec=spec,
                theta=theta,
                start_state=start_state,
                project_root=project_root,
            )
        except Exception as exc:
            failure_messages.append(type(exc).__name__ + ": " + str(exc))
            continue
        logl = float(entry["row"]["log_likelihood"])
        if logl > best_logl:
            best_logl = logl
            best_entry = entry

    if best_entry is None:
        return {
            "theta": np.asarray(theta, dtype=float),
            "row": _failure_row(theta, stage=stage, message=" | ".join(failure_messages)),
            "result": None,
            "start_state": None,
        }

    best_entry["row"]["stage"] = stage
    return best_entry


def _best_by_eta(table: pd.DataFrame) -> pd.DataFrame:
    good = table.loc[np.isfinite(table["log_likelihood"])].copy()
    best_idx = good.groupby("eta_t")["log_likelihood"].idxmax()
    best_table = good.loc[best_idx].sort_values("eta_t").reset_index(drop=True)
    best_table["best_radial_model"] = "powerlaw_a"
    return best_table


def _save_best_payload(entry: dict[str, object], output_tables: Path, prefix: str) -> None:
    if entry["result"] is None:
        return
    with (output_tables / f"{prefix}_best_result.pkl").open("wb") as handle:
        pickle.dump(entry["result"], handle, protocol=pickle.HIGHEST_PROTOCOL)
    (output_tables / f"{prefix}_best_result_summary.json").write_text(json.dumps(entry["row"], indent=2))


def _coarse_cell_bounds(grid: np.ndarray, best_value: float, lower_bound: float, upper_bound: float) -> tuple[float, float, bool]:
    grid = np.asarray(grid, dtype=float)
    idx = int(np.argmin(np.abs(grid - float(best_value))))
    on_edge = idx == 0 or idx == len(grid) - 1
    if idx == 0:
        lower = lower_bound
        upper = 0.5 * (grid[0] + grid[1])
    elif idx == len(grid) - 1:
        lower = 0.5 * (grid[-2] + grid[-1])
        upper = upper_bound
    else:
        lower = 0.5 * (grid[idx - 1] + grid[idx])
        upper = 0.5 * (grid[idx] + grid[idx + 1])
    return float(lower), float(upper), on_edge


def _build_local_spec_from_coarse(
    coarse_spec: GridSpec,
    coarse_best_row: pd.Series,
    *,
    local_eta_n: int,
    local_alpha_n: int,
    local_logmc_n: int,
) -> tuple[GridSpec, dict[str, bool]]:
    eta_grid = coarse_spec.eta_grid()
    alpha_grid = coarse_spec.alpha_grid()
    logmc_grid = coarse_spec.logmc_grid()
    eta_min, eta_max, eta_edge = _coarse_cell_bounds(
        eta_grid,
        float(coarse_best_row["eta_t"]),
        coarse_spec.eta_min,
        coarse_spec.eta_max,
    )
    alpha_min, alpha_max, alpha_edge = _coarse_cell_bounds(
        alpha_grid,
        float(coarse_best_row["input_alpha_dndm"]),
        coarse_spec.alpha_min,
        coarse_spec.alpha_max,
    )
    logmc_min, logmc_max, logmc_edge = _coarse_cell_bounds(
        logmc_grid,
        float(coarse_best_row["input_log10_m_c_msun"]),
        coarse_spec.logmc_min,
        coarse_spec.logmc_max,
    )
    return (
        GridSpec(
            eta_min=eta_min,
            eta_max=eta_max,
            eta_n=local_eta_n,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            alpha_n=local_alpha_n,
            logmc_min=logmc_min,
            logmc_max=logmc_max,
            logmc_n=local_logmc_n,
        ),
        {"eta_t": eta_edge, "alpha": alpha_edge, "logmc": logmc_edge},
    )


def _build_grid_array(
    table: pd.DataFrame,
    *,
    eta_grid: np.ndarray,
    alpha_grid: np.ndarray,
    logmc_grid: np.ndarray,
    value_column: str,
) -> np.ndarray:
    array = np.full((len(eta_grid), len(alpha_grid), len(logmc_grid)), np.nan, dtype=float)
    eta_lookup = {round(float(value), 10): index for index, value in enumerate(eta_grid)}
    alpha_lookup = {round(float(value), 10): index for index, value in enumerate(alpha_grid)}
    logmc_lookup = {round(float(value), 10): index for index, value in enumerate(logmc_grid)}
    for _, row in table.iterrows():
        i = eta_lookup[round(float(row["eta_t"]), 10)]
        j = alpha_lookup[round(float(row["input_alpha_dndm"]), 10)]
        k = logmc_lookup[round(float(row["input_log10_m_c_msun"]), 10)]
        array[i, j, k] = float(row[value_column])
    return array


def _interpolator_from_grid(
    eta_grid: np.ndarray,
    alpha_grid: np.ndarray,
    logmc_grid: np.ndarray,
    values: np.ndarray,
) -> RegularGridInterpolator:
    return RegularGridInterpolator(
        (eta_grid, alpha_grid, logmc_grid),
        values,
        bounds_error=False,
        fill_value=np.nan,
    )


def _evaluate_interpolated(interpolator: RegularGridInterpolator, theta: np.ndarray) -> float:
    value = interpolator(np.asarray(theta, dtype=float))
    if np.isscalar(value):
        scalar = float(value)
    else:
        scalar = float(np.asarray(value).reshape(-1)[0])
    return scalar


def _logposterior_interpolated(interpolator: RegularGridInterpolator, theta: np.ndarray, bounds: np.ndarray) -> float:
    theta = np.asarray(theta, dtype=float)
    if np.any(theta < bounds[:, 0]) or np.any(theta > bounds[:, 1]):
        return -np.inf
    value = _evaluate_interpolated(interpolator, theta)
    if not np.isfinite(value):
        return -np.inf
    return value


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


def _draw_initial_mcmc_positions(
    *,
    rng: np.random.Generator,
    center: np.ndarray,
    bounds: np.ndarray,
    n_chains: int,
) -> list[np.ndarray]:
    widths = bounds[:, 1] - bounds[:, 0]
    positions = []
    for chain in range(n_chains):
        if chain == 0:
            theta = center.copy()
        else:
            theta = center + rng.normal(scale=0.20 * widths, size=3)
            theta = np.minimum(np.maximum(theta, bounds[:, 0] + 1.0e-8), bounds[:, 1] - 1.0e-8)
        positions.append(theta)
    return positions


def _corner_plot(samples: pd.DataFrame, reference_row: dict[str, object], output_path: Path) -> None:
    columns = [
        ("eta_t", r"$\eta_t$"),
        ("input_alpha_dndm", r"$\alpha$"),
        ("input_log10_m_c_msun", r"$\log_{10}(M_c/{\rm M}_\odot)$"),
        ("gamma_linear_a", r"$\gamma_a$"),
    ]
    n_dim = len(columns)
    fig, axes = plt.subplots(n_dim, n_dim, figsize=(10.5, 10.5))
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
    columns = [
        ("eta_t", r"$\eta_t$"),
        ("input_alpha_dndm", r"$\alpha$"),
        ("input_log10_m_c_msun", r"$\log_{10} M_c$"),
        ("gamma_linear_a", r"$\gamma_a$"),
    ]
    fig, axes = plt.subplots(len(columns), 1, figsize=(9.0, 8.5), sharex=True)
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
    parser.add_argument("--output-root-name", default="profile_map_and_interpolated_mcmc_schechter_powerlaw_a_logistic")
    parser.add_argument("--coarse-eta-min", type=float, default=0.6)
    parser.add_argument("--coarse-eta-max", type=float, default=2.1)
    parser.add_argument("--coarse-eta-n", type=int, default=6)
    parser.add_argument("--coarse-alpha-min", type=float, default=-1.8)
    parser.add_argument("--coarse-alpha-max", type=float, default=-0.6)
    parser.add_argument("--coarse-alpha-n", type=int, default=5)
    parser.add_argument("--coarse-logmc-min", type=float, default=6.10)
    parser.add_argument("--coarse-logmc-max", type=float, default=6.55)
    parser.add_argument("--coarse-logmc-n", type=int, default=4)
    parser.add_argument("--local-eta-n", type=int, default=7)
    parser.add_argument("--local-alpha-n", type=int, default=7)
    parser.add_argument("--local-logmc-n", type=int, default=7)
    parser.add_argument("--anchor-k", type=int, default=4)
    parser.add_argument("--mcmc-chains", type=int, default=8)
    parser.add_argument("--mcmc-steps", type=int, default=5000)
    parser.add_argument("--mcmc-burn", type=int, default=1000)
    parser.add_argument("--mcmc-thin", type=int, default=5)
    parser.add_argument("--mcmc-adapt-until", type=int, default=800)
    parser.add_argument("--mcmc-adapt-every", type=int, default=50)
    args = parser.parse_args()

    output_root = PROJECT_ROOT / "variants" / args.output_root_name
    figures_dir = output_root / "outputs" / "figures"
    tables_dir = output_root / "outputs" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    from globular_clusters_imf.joint_model import JointModelSpec
    from globular_clusters_imf.model import fit_catalog_models

    rng = np.random.default_rng(20260527)
    spec = JointModelSpec(imf_family="schechter", radial_model="powerlaw_a")

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

    coarse_entries: list[dict[str, object]] = []
    coarse_cache: dict[tuple[float, float, float], dict[str, object]] = {}
    for eta_t in coarse_eta:
        for log_mc in coarse_logmc:
            for alpha in coarse_alpha:
                theta = np.array([eta_t, alpha, log_mc], dtype=float)
                entry = _evaluate_theta_multistart(
                    prepared_catalog=prepared_catalog,
                    spec=spec,
                    theta=theta,
                    stage="coarse",
                    project_root=output_root,
                    anchor_start_state=None,
                )
                coarse_entries.append(entry)
                coarse_cache[_round_key(theta)] = entry
                print(
                    f"[coarse] eta_t={eta_t:.3f} alpha={alpha:.3f} logMc={log_mc:.3f} "
                    f"logL={float(entry['row']['log_likelihood']):.3f} beta_a={float(entry['row']['beta_log10_a']):.3f} "
                    f"N0>1e4={float(entry['row']['final_total_initial_count_above_log10_4']):.1f}"
                )

    coarse_table = pd.DataFrame([entry["row"] for entry in coarse_entries]).sort_values(
        ["eta_t", "input_alpha_dndm", "input_log10_m_c_msun"]
    ).reset_index(drop=True)
    coarse_table.to_csv(tables_dir / "coarse_grid_results.csv", index=False)
    coarse_best_by_eta = _best_by_eta(coarse_table)
    coarse_best_by_eta.to_csv(tables_dir / "coarse_profiled_eta_results.csv", index=False)
    _plot_logl_vs_multiplier(coarse_best_by_eta, figures_dir / "coarse_profiled_logl_vs_eta_t.png")
    _plot_properties(coarse_best_by_eta, figures_dir / "coarse_profiled_properties_vs_eta_t.png")

    coarse_successes = [entry for entry in coarse_entries if np.isfinite(entry["row"]["log_likelihood"])]
    coarse_best_entry = max(coarse_successes, key=lambda entry: float(entry["row"]["log_likelihood"]))
    coarse_best_row = pd.Series(coarse_best_entry["row"])
    _save_best_payload(coarse_best_entry, tables_dir, prefix="coarse")

    local_spec, edge_flags = _build_local_spec_from_coarse(
        coarse_spec,
        coarse_best_row,
        local_eta_n=int(args.local_eta_n),
        local_alpha_n=int(args.local_alpha_n),
        local_logmc_n=int(args.local_logmc_n),
    )
    local_eta = local_spec.eta_grid()
    local_alpha = local_spec.alpha_grid()
    local_logmc = local_spec.logmc_grid()
    local_bounds = np.array(
        [
            [local_spec.eta_min, local_spec.eta_max],
            [local_spec.alpha_min, local_spec.alpha_max],
            [local_spec.logmc_min, local_spec.logmc_max],
        ],
        dtype=float,
    )

    anchor_entries = sorted(
        coarse_successes,
        key=lambda entry: float(entry["row"]["log_likelihood"]),
        reverse=True,
    )[: max(1, int(args.anchor_k))]

    local_entries: list[dict[str, object]] = []
    local_cache: dict[tuple[float, float, float], dict[str, object]] = {}
    for eta_t in local_eta:
        for log_mc in local_logmc:
            for alpha in local_alpha:
                theta = np.array([eta_t, alpha, log_mc], dtype=float)
                anchor_state = _select_anchor_start_state(theta=theta, anchors=anchor_entries, bounds=coarse_bounds)
                entry = _evaluate_theta_multistart(
                    prepared_catalog=prepared_catalog,
                    spec=spec,
                    theta=theta,
                    stage="local",
                    project_root=output_root,
                    anchor_start_state=anchor_state,
                )
                local_entries.append(entry)
                local_cache[_round_key(theta)] = entry
                print(
                    f"[local] eta_t={eta_t:.3f} alpha={alpha:.3f} logMc={log_mc:.3f} "
                    f"logL={float(entry['row']['log_likelihood']):.3f} beta_a={float(entry['row']['beta_log10_a']):.3f} "
                    f"N0>1e4={float(entry['row']['final_total_initial_count_above_log10_4']):.1f}"
                )

    local_table = pd.DataFrame([entry["row"] for entry in local_entries]).sort_values(
        ["eta_t", "input_alpha_dndm", "input_log10_m_c_msun"]
    ).reset_index(drop=True)
    local_table.to_csv(tables_dir / "local_grid_results.csv", index=False)
    local_best_by_eta = _best_by_eta(local_table)
    local_best_by_eta.to_csv(tables_dir / "local_profiled_eta_results.csv", index=False)
    _plot_logl_vs_multiplier(local_best_by_eta, figures_dir / "local_profiled_logl_vs_eta_t.png")
    _plot_properties(local_best_by_eta, figures_dir / "local_profiled_properties_vs_eta_t.png")

    local_successes = [entry for entry in local_entries if np.isfinite(entry["row"]["log_likelihood"])]
    local_best_entry = max(local_successes, key=lambda entry: float(entry["row"]["log_likelihood"]))
    _save_best_payload(local_best_entry, tables_dir, prefix="local")

    logl_grid = _build_grid_array(
        local_table,
        eta_grid=local_eta,
        alpha_grid=local_alpha,
        logmc_grid=local_logmc,
        value_column="log_likelihood",
    )
    gamma_grid = _build_grid_array(
        local_table,
        eta_grid=local_eta,
        alpha_grid=local_alpha,
        logmc_grid=local_logmc,
        value_column="gamma_linear_a",
    )
    count_grid = _build_grid_array(
        local_table,
        eta_grid=local_eta,
        alpha_grid=local_alpha,
        logmc_grid=local_logmc,
        value_column="final_total_initial_count_above_log10_4",
    )
    mass_grid = _build_grid_array(
        local_table,
        eta_grid=local_eta,
        alpha_grid=local_alpha,
        logmc_grid=local_logmc,
        value_column="final_total_initial_stellar_mass_above_log10_4_msun",
    )
    completeness_grid = _build_grid_array(
        local_table,
        eta_grid=local_eta,
        alpha_grid=local_alpha,
        logmc_grid=local_logmc,
        value_column="mean_detectability_above_log10_4",
    )
    if not np.all(np.isfinite(logl_grid)):
        raise RuntimeError("Local profile cube is incomplete or contains failed points; cannot build interpolated posterior.")

    logl_interpolator = _interpolator_from_grid(local_eta, local_alpha, local_logmc, logl_grid)
    gamma_interpolator = _interpolator_from_grid(local_eta, local_alpha, local_logmc, gamma_grid)
    count_interpolator = _interpolator_from_grid(local_eta, local_alpha, local_logmc, count_grid)
    mass_interpolator = _interpolator_from_grid(local_eta, local_alpha, local_logmc, mass_grid)
    completeness_interpolator = _interpolator_from_grid(local_eta, local_alpha, local_logmc, completeness_grid)

    n_chains = int(args.mcmc_chains)
    n_steps = int(args.mcmc_steps)
    burn_in = int(args.mcmc_burn)
    thin = int(args.mcmc_thin)
    adapt_until = int(args.mcmc_adapt_until)
    adapt_every = int(args.mcmc_adapt_every)
    proposal_scales = np.tile(0.10 * (local_bounds[:, 1] - local_bounds[:, 0]), (n_chains, 1))
    current_thetas = _draw_initial_mcmc_positions(
        rng=rng,
        center=np.array(
            [
                float(local_best_entry["row"]["eta_t"]),
                float(local_best_entry["row"]["input_alpha_dndm"]),
                float(local_best_entry["row"]["input_log10_m_c_msun"]),
            ],
            dtype=float,
        ),
        bounds=local_bounds,
        n_chains=n_chains,
    )
    current_logps = np.array(
        [_logposterior_interpolated(logl_interpolator, theta, local_bounds) for theta in current_thetas],
        dtype=float,
    )
    accepts = np.zeros((n_chains, n_steps), dtype=bool)
    chain_records: list[dict[str, object]] = []

    for step in range(n_steps):
        for chain in range(n_chains):
            proposal = current_thetas[chain] + rng.normal(scale=proposal_scales[chain], size=3)
            proposal_logp = _logposterior_interpolated(logl_interpolator, proposal, local_bounds)
            accepted = False
            if np.isfinite(proposal_logp):
                if np.log(rng.uniform()) < min(0.0, proposal_logp - current_logps[chain]):
                    current_thetas[chain] = proposal
                    current_logps[chain] = proposal_logp
                    accepted = True
            accepts[chain, step] = accepted
            theta = current_thetas[chain]
            chain_records.append(
                {
                    "chain": int(chain),
                    "step": int(step),
                    "accepted": bool(accepted),
                    "eta_t": float(theta[0]),
                    "input_alpha_dndm": float(theta[1]),
                    "input_log10_m_c_msun": float(theta[2]),
                    "gamma_linear_a": _evaluate_interpolated(gamma_interpolator, theta),
                    "final_total_initial_count_above_log10_4": _evaluate_interpolated(count_interpolator, theta),
                    "final_total_initial_stellar_mass_above_log10_4_msun": _evaluate_interpolated(mass_interpolator, theta),
                    "mean_detectability_above_log10_4": _evaluate_interpolated(completeness_interpolator, theta),
                    "log_likelihood": float(current_logps[chain]),
                    "proposal_scale_eta_t": float(proposal_scales[chain, 0]),
                    "proposal_scale_alpha": float(proposal_scales[chain, 1]),
                    "proposal_scale_logmc": float(proposal_scales[chain, 2]),
                }
            )
        if step + 1 <= adapt_until and (step + 1) % adapt_every == 0:
            window = accepts[:, step + 1 - adapt_every : step + 1].mean(axis=1)
            for chain in range(n_chains):
                if window[chain] < 0.18:
                    proposal_scales[chain] *= 0.85
                elif window[chain] > 0.35:
                    proposal_scales[chain] *= 1.15
                proposal_scales[chain] = np.clip(
                    proposal_scales[chain],
                    0.01 * (local_bounds[:, 1] - local_bounds[:, 0]),
                    0.35 * (local_bounds[:, 1] - local_bounds[:, 0]),
                )
        if (step + 1) % 500 == 0 or step == 0:
            best_chain = int(np.argmax(current_logps))
            print(
                f"[interp-mcmc] step={step+1:04d} "
                f"best logL={float(current_logps[best_chain]):.3f} "
                f"eta_t={float(current_thetas[best_chain][0]):.3f} "
                f"alpha={float(current_thetas[best_chain][1]):.3f} "
                f"logMc={float(current_thetas[best_chain][2]):.3f} "
                f"accept={float(accepts[:, max(0, step-499):step+1].mean()):.3f}"
            )

    chain_table = pd.DataFrame(chain_records)
    chain_table.to_csv(tables_dir / "interpolated_mcmc_chain.csv", index=False)
    posterior_parts = []
    for _, frame in chain_table.loc[chain_table["step"] >= burn_in].groupby("chain"):
        posterior_parts.append(frame.iloc[::thin])
    posterior_table = pd.concat(posterior_parts, ignore_index=True)
    posterior_table.to_csv(tables_dir / "interpolated_mcmc_posterior_samples.csv", index=False)

    posterior_summary_rows = []
    for column in [
        "eta_t",
        "input_alpha_dndm",
        "input_log10_m_c_msun",
        "gamma_linear_a",
        "final_total_initial_count_above_log10_4",
        "final_total_initial_stellar_mass_above_log10_4_msun",
        "mean_detectability_above_log10_4",
        "log_likelihood",
    ]:
        values = np.asarray(posterior_table[column], dtype=float)
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
    posterior_summary_table.to_csv(tables_dir / "interpolated_posterior_summary.csv", index=False)

    rhat = {}
    for column in ["eta_t", "input_alpha_dndm", "input_log10_m_c_msun", "gamma_linear_a"]:
        pivot = (
            posterior_table.pivot_table(index="step", columns="chain", values=column, aggfunc="last")
            .dropna()
            .to_numpy()
            .T
        )
        rhat[column] = _compute_rhat(pivot)
    acceptance_by_chain = {
        str(chain): float(accepts[chain].mean())
        for chain in range(n_chains)
    }

    best_posterior_row = posterior_table.sort_values("log_likelihood", ascending=False).iloc[0].to_dict()
    reference_row = dict(local_best_entry["row"])
    _corner_plot(posterior_table, reference_row, figures_dir / "interpolated_posterior_corner.png")
    _trace_plot(chain_table, figures_dir / "interpolated_posterior_traces.png", burn_in=burn_in)

    summary_payload = {
        "surface_model": SURFACE_MODEL,
        "model_spec": {"imf_family": spec.imf_family, "radial_model": spec.radial_model},
        "n_detectability_iterations": N_DETECTABILITY_ITERATIONS,
        "coarse_grid_spec": coarse_spec.__dict__,
        "coarse_best": json.loads(pd.Series(coarse_best_entry["row"]).to_json()),
        "local_grid_spec": local_spec.__dict__,
        "local_bounds_from_coarse_cell": {
            "eta_t": [local_spec.eta_min, local_spec.eta_max],
            "alpha_dndm": [local_spec.alpha_min, local_spec.alpha_max],
            "log10_m_c_msun": [local_spec.logmc_min, local_spec.logmc_max],
        },
        "coarse_best_on_edge": edge_flags,
        "anchor_library": [json.loads(pd.Series(entry["row"]).to_json()) for entry in anchor_entries],
        "local_best": json.loads(pd.Series(local_best_entry["row"]).to_json()),
        "mcmc": {
            "n_chains": n_chains,
            "n_steps": n_steps,
            "burn_in": burn_in,
            "thin": thin,
            "acceptance_by_chain": acceptance_by_chain,
            "rhat": rhat,
            "best_posterior_sample": json.loads(pd.Series(best_posterior_row).to_json()),
            "posterior_summary": posterior_summary_table.to_dict(orient="records"),
        },
    }
    (tables_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2))

    print(figures_dir / "coarse_profiled_logl_vs_eta_t.png")
    print(figures_dir / "local_profiled_logl_vs_eta_t.png")
    print(figures_dir / "interpolated_posterior_corner.png")
    print(figures_dir / "interpolated_posterior_traces.png")
    print(tables_dir / "interpolated_posterior_summary.csv")
    print(tables_dir / "summary.json")


if __name__ == "__main__":
    main()
