from __future__ import annotations

import argparse
import json
import os
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
from matplotlib.colors import LinearSegmentedColormap
from scipy import optimize, special


def compute_eta_t_equivalent_to_current_shift(catalog: pd.DataFrame, selection_offset_dex: float) -> float:
    from globular_clusters_imf.model import AGE_GYR, total_dissolution_time_myr

    shifted_cut_mass = catalog["survival_mass_cut_msun"].to_numpy(dtype=float) * np.power(10.0, selection_offset_dex)
    lifetimes_gyr = np.array(
        [
            total_dissolution_time_myr(mass, row.r_apo_kpc, row.eccentricity) / 1000.0
            for mass, row in zip(shifted_cut_mass, catalog.itertuples(index=False), strict=True)
        ],
        dtype=float,
    )
    lifetime_ratio = lifetimes_gyr / AGE_GYR
    return float(1.0 / np.nanmedian(lifetime_ratio))


def build_raw_survival_grid_from_eta_t(
    catalog: pd.DataFrame,
    *,
    eta_t: float,
    n_radius_grid: int = 160,
    n_mass_grid: int = 180,
    bandwidth_log10_a_dex: float = 0.18,
) -> dict[str, object]:
    from globular_clusters_imf.model import AGE_MYR, survival_mass_cut_msun

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


def extract_half_probability_boundary(
    log_mass_grid: np.ndarray,
    log_a_grid: np.ndarray,
    survival_probability: np.ndarray,
) -> np.ndarray:
    return extract_probability_boundary(log_mass_grid, log_a_grid, survival_probability, level=0.5)


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


def fit_monotonic_soft_survivability_model(
    catalog: pd.DataFrame,
    log_mass_grid: np.ndarray,
    log_a_grid: np.ndarray,
    survival_probability_raw: np.ndarray,
    bandwidth_log10_a_dex: float,
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
        boundary_10, boundary_50, boundary_90, meta = boundary_curve_from_params(
            np.asarray(params, dtype=float),
            log_a_grid,
            outer_level_90=outer_plateau_90,
        )
        model_probability = compact_transition_surface(log_mass_grid, boundary_10, boundary_90)
        map_residual = model_probability - survival_probability_raw
        boundary_residual_50 = boundary_50 - raw_boundary_50
        boundary_residual_10 = boundary_10 - raw_boundary_10
        boundary_residual_90 = boundary_90 - raw_boundary_90
        boundary_10_data, boundary_50_data, boundary_90_data, _ = boundary_curve_from_params(
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
        (float(log_a_grid.min()), float(log_a_grid.max())),
        (np.log(0.08), np.log(1.2)),
        (np.log(0.03), np.log(0.5)),
        (np.log(0.08), np.log(4.0)),
    ]
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
    fitted_probability = compact_transition_surface(log_mass_grid, boundary_10, boundary_90)
    boundary_10_data, boundary_50_data, boundary_90_data, _ = boundary_curve_from_params(
        best_params,
        log_a_data,
        outer_level_90=outer_plateau_90,
    )
    inside_band = (log_mass_data >= boundary_10_data) & (log_mass_data <= boundary_90_data)
    occupancy_rows = []
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
    return {
        "optimization_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_value": float(result.fun),
        "raw_boundary_10_log10_msun": raw_boundary_10,
        "raw_boundary_50_log10_msun": raw_boundary_50,
        "raw_boundary_80_log10_msun": raw_boundary_80,
        "raw_boundary_90_log10_msun": raw_boundary_90,
        "fitted_boundary_10_log10_msun": boundary_10,
        "fitted_boundary_50_log10_msun": boundary_50,
        "fitted_boundary_90_log10_msun": boundary_90,
        "fitted_probability": fitted_probability,
        "outer_plateau_90_log10_msun_from_raw_boundary": outer_plateau_90,
        "occupancy_table": pd.DataFrame(occupancy_rows),
        "parameters": meta,
    }


def plot_survivability_panel(
    catalog: pd.DataFrame,
    *,
    log_mass_grid: np.ndarray,
    semi_major_axis_grid_kpc: np.ndarray,
    raw_probability: np.ndarray,
    fitted_probability: np.ndarray,
    fitted_boundary_10_log10_msun: np.ndarray,
    fitted_boundary_50_log10_msun: np.ndarray,
    fitted_boundary_90_log10_msun: np.ndarray,
    output_path: Path,
) -> None:
    semi_major_axis = catalog["semi_major_axis_kpc"].to_numpy(dtype=float)
    initial_mass = catalog["initial_mass_msun"].to_numpy(dtype=float)
    mass_grid = np.power(10.0, log_mass_grid)
    survivability_cmap = LinearSegmentedColormap.from_list(
        "survivability_grey_to_white",
        ["#8a8a8a", "#ffffff"],
    )

    x_limits = (
        float(semi_major_axis.min() / 1.15),
        float(semi_major_axis.max() * 1.15),
    )
    y_limits = (
        1.0e3,
        float(initial_mass.max() * 1.08),
    )

    radius_edges_core = 10.0 ** centers_to_edges(np.log10(semi_major_axis_grid_kpc))
    radius_edges = np.concatenate(([x_limits[0]], radius_edges_core, [x_limits[1]]))
    log_mass_edges = centers_to_edges(log_mass_grid)
    mass_edges_core = np.power(10.0, log_mass_edges)
    mass_edges = np.concatenate(([y_limits[0]], mass_edges_core, [y_limits[1]]))
    fitted_probability = np.asarray(fitted_probability, dtype=float)
    survival_probability_plot = np.pad(fitted_probability, ((1, 1), (1, 1)), mode="edge")
    survival_probability_plot[0, :] = 0.0

    fig, ax = plt.subplots(figsize=(3.6, 3.35))
    ax.pcolormesh(
        radius_edges,
        mass_edges,
        survival_probability_plot,
        cmap=survivability_cmap,
        vmin=0.0,
        vmax=1.0,
        shading="auto",
        rasterized=True,
    )
    y_min_log10 = np.log10(y_limits[0])
    ax.contour(
        semi_major_axis_grid_kpc,
        mass_grid,
        np.asarray(raw_probability, dtype=float),
        levels=[0.1, 0.5, 0.9],
        colors="#4d4d4d",
        linewidths=1.0,
        linestyles="dashed",
    )
    model_line_specs = [
        ("#74a9cf", np.asarray(fitted_boundary_10_log10_msun, dtype=float)),
        ("#2b8cbe", np.asarray(fitted_boundary_50_log10_msun, dtype=float)),
        ("#045a8d", np.asarray(fitted_boundary_90_log10_msun, dtype=float)),
    ]
    for color, line_log_mass in model_line_specs:
        line_log_mass = np.maximum(line_log_mass, y_min_log10)
        line_mass = np.power(10.0, line_log_mass)
        ax.plot(
            semi_major_axis_grid_kpc,
            line_mass,
            color=color,
            linewidth=1.35,
        )
    ax.scatter(
        semi_major_axis,
        initial_mass,
        s=11,
        color="black",
        alpha=0.35,
        linewidths=0.0,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_xlabel(r"Orbital semimajor axis $a$ [kpc]")
    ax.set_ylabel(r"Mass [$\mathrm{M_\odot}$]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def centers_to_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    inner = 0.5 * (centers[1:] + centers[:-1])
    first = centers[0] - 0.5 * (centers[1] - centers[0])
    last = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return np.concatenate(([first], inner, [last]))


def main() -> None:
    from globular_clusters_imf.joint_model import calibrate_fixed_selection_offset_dex
    from globular_clusters_imf.model import fit_catalog_models

    parser = argparse.ArgumentParser()
    parser.add_argument("--eta-t", type=float, default=None)
    parser.add_argument("--output-stem", type=str, default="survivability_eta_t_monotonic_panel")
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    outputs_figures = project_root / "outputs" / "figures"
    outputs_tables = project_root / "outputs" / "tables"
    outputs_figures.mkdir(parents=True, exist_ok=True)
    outputs_tables.mkdir(parents=True, exist_ok=True)

    catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    catalog_results = fit_catalog_models(catalog, project_root / "tmp_survivability_eta_t_monotonic")
    fit_catalog = catalog_results["catalog"]

    selection_offset_dex = calibrate_fixed_selection_offset_dex(fit_catalog)
    eta_t = (
        float(args.eta_t)
        if args.eta_t is not None
        else compute_eta_t_equivalent_to_current_shift(fit_catalog, selection_offset_dex)
    )
    raw_grid = build_raw_survival_grid_from_eta_t(fit_catalog, eta_t=eta_t)
    fitted_model = fit_monotonic_soft_survivability_model(
        raw_grid["catalog"],
        np.asarray(raw_grid["log_mass_grid"], dtype=float),
        np.asarray(raw_grid["log_a_grid"], dtype=float),
        np.asarray(raw_grid["survival_probability"], dtype=float),
        bandwidth_log10_a_dex=float(raw_grid["bandwidth_log10_a_dex"]),
    )

    output_path = outputs_figures / f"{args.output_stem}.png"
    plot_survivability_panel(
        raw_grid["catalog"],
        log_mass_grid=np.asarray(raw_grid["log_mass_grid"], dtype=float),
        semi_major_axis_grid_kpc=np.asarray(raw_grid["semi_major_axis_grid_kpc"], dtype=float),
        raw_probability=np.asarray(raw_grid["survival_probability"], dtype=float),
        fitted_probability=np.asarray(fitted_model["fitted_probability"], dtype=float),
        fitted_boundary_10_log10_msun=np.asarray(fitted_model["fitted_boundary_10_log10_msun"], dtype=float),
        fitted_boundary_50_log10_msun=np.asarray(fitted_model["fitted_boundary_50_log10_msun"], dtype=float),
        fitted_boundary_90_log10_msun=np.asarray(fitted_model["fitted_boundary_90_log10_msun"], dtype=float),
        output_path=output_path,
    )

    summary = {
        "eta_t": float(eta_t),
        "selection_offset_dex_equivalent": float(selection_offset_dex),
        "bandwidth_log10_a_dex": float(raw_grid["bandwidth_log10_a_dex"]),
        "outer_plateau_90_log10_msun_from_raw_boundary": float(
            fitted_model["outer_plateau_90_log10_msun_from_raw_boundary"]
        ),
        **fitted_model["parameters"],
        "optimization_success": bool(fitted_model["optimization_success"]),
        "optimizer_message": str(fitted_model["optimizer_message"]),
        "optimizer_value": float(fitted_model["optimizer_value"]),
    }
    (outputs_tables / f"{args.output_stem}_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    boundary_table = pd.DataFrame(
        {
            "log10_semi_major_axis_kpc": np.asarray(raw_grid["log_a_grid"], dtype=float),
            "semi_major_axis_kpc": np.asarray(raw_grid["semi_major_axis_grid_kpc"], dtype=float),
            "raw_boundary_10_log10_msun": np.asarray(fitted_model["raw_boundary_10_log10_msun"], dtype=float),
            "raw_boundary_50_log10_msun": np.asarray(fitted_model["raw_boundary_50_log10_msun"], dtype=float),
            "raw_boundary_80_log10_msun": np.asarray(fitted_model["raw_boundary_80_log10_msun"], dtype=float),
            "raw_boundary_90_log10_msun": np.asarray(fitted_model["raw_boundary_90_log10_msun"], dtype=float),
            "fitted_boundary_10_log10_msun": np.asarray(fitted_model["fitted_boundary_10_log10_msun"], dtype=float),
            "fitted_boundary_50_log10_msun": np.asarray(fitted_model["fitted_boundary_50_log10_msun"], dtype=float),
            "fitted_boundary_90_log10_msun": np.asarray(fitted_model["fitted_boundary_90_log10_msun"], dtype=float),
        }
    )
    boundary_table.to_csv(outputs_tables / f"{args.output_stem}_boundary.csv", index=False)
    fitted_model["occupancy_table"].to_csv(
        outputs_tables / f"{args.output_stem}_transition_band_occupancy.csv",
        index=False,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
