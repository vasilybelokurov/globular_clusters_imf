from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

AGE_GYR = 12.0
AGE_MYR = AGE_GYR * 1_000.0
MEAN_INITIAL_STELLAR_MASS_MSUN = 0.65
MILKY_WAY_CIRCULAR_SPEED_KMS = 240.0
LARGE_PENALTY = 1.0e30


@dataclass
class LognormalFitResult:
    model_name: str
    mu_log10_msun: float
    sigma_log10_msun: float
    selection_offset_dex: float
    log_likelihood: float
    n_clusters: int


@dataclass
class PowerLawFitResult:
    model_name: str
    beta: float
    mass_min_msun: float
    mass_max_msun: float
    selection_offset_dex: float
    log_likelihood: float
    n_clusters: int


def fit_catalog_models(
    catalog: pd.DataFrame,
    project_root: Path,
    mass_min_msun: float = 1.0e3,
    mass_max_msun: float = 1.0e8,
) -> dict[str, object]:
    working = catalog.copy()
    working["survival_mass_cut_msun"] = working.apply(
        lambda row: survival_mass_cut_msun(
            r_apo_kpc=row["r_apo_kpc"],
            eccentricity=row["eccentricity"],
            age_myr=AGE_MYR,
        ),
        axis=1,
    )
    working["log_survival_mass_cut_msun"] = np.log10(working["survival_mass_cut_msun"])

    fit_sample = working.loc[
        np.isfinite(working["log_initial_mass_msun"])
        & np.isfinite(working["log_survival_mass_cut_msun"])
        & np.isfinite(working["semi_major_axis_kpc"])
    ].copy()

    lognormal_result = fit_truncated_lognormal(fit_sample)
    powerlaw_result = fit_truncated_powerlaw(
        fit_sample,
        mass_min_msun=mass_min_msun,
        mass_max_msun=mass_max_msun,
    )

    fit_sample["survival_probability_lognormal"] = survival_probability_lognormal(
        fit_sample["log_survival_mass_cut_msun"].to_numpy() + lognormal_result.selection_offset_dex,
        lognormal_result.mu_log10_msun,
        lognormal_result.sigma_log10_msun,
    )
    fit_sample["survival_probability_powerlaw"] = survival_probability_powerlaw(
        np.power(
            10.0,
            fit_sample["log_survival_mass_cut_msun"].to_numpy() + powerlaw_result.selection_offset_dex,
        ),
        powerlaw_result.beta,
        powerlaw_result.mass_min_msun,
        powerlaw_result.mass_max_msun,
    )

    lognormal_profile = estimate_radial_profile(
        fit_sample,
        survival_probability_column="survival_probability_lognormal",
    )
    radial_patch_summary, radial_patch_table = estimate_lognormal_imf_patches(
        fit_sample,
        lognormal_result=lognormal_result,
    )
    survivability_map = estimate_survivability_map(
        fit_sample,
        selection_offset_dex=lognormal_result.selection_offset_dex,
    )
    powerlaw_profile = estimate_radial_profile(
        fit_sample,
        survival_probability_column="survival_probability_powerlaw",
    )

    outputs_tables = project_root / "outputs" / "tables"
    outputs_tables.mkdir(parents=True, exist_ok=True)

    fit_sample.to_csv(outputs_tables / "catalog_with_survival_thresholds.csv", index=False)
    lognormal_profile.to_csv(outputs_tables / "radial_profile_lognormal.csv", index=False)
    radial_patch_summary.to_csv(outputs_tables / "radial_imf_patch_summary_5bins.csv", index=False)
    radial_patch_table.to_csv(outputs_tables / "radial_imf_patch_table_5bins.csv", index=False)
    survivability_map["table"].to_csv(outputs_tables / "survivability_map_lognormal.csv", index=False)
    powerlaw_profile.to_csv(outputs_tables / "radial_profile_powerlaw.csv", index=False)

    summary = {
        "age_gyr": AGE_GYR,
        "mean_initial_stellar_mass_msun": MEAN_INITIAL_STELLAR_MASS_MSUN,
        "milky_way_circular_speed_kms": MILKY_WAY_CIRCULAR_SPEED_KMS,
        "lognormal": asdict(lognormal_result),
        "powerlaw": asdict(powerlaw_result),
        "lognormal_total_estimated_initial_count": float(
            lognormal_profile["estimated_initial_count"].sum()
        ),
        "powerlaw_total_estimated_initial_count": float(
            powerlaw_profile["estimated_initial_count"].sum()
        ),
        "patch_plot_radius_binning": "5 equal-count bins in log10(semimajor axis)",
        "survivability_map_loga_bandwidth_dex": float(survivability_map["bandwidth_log10_a_dex"]),
    }
    (outputs_tables / "model_summary.json").write_text(json.dumps(summary, indent=2))
    return {
        "catalog": fit_sample,
        "lognormal": lognormal_result,
        "powerlaw": powerlaw_result,
        "lognormal_profile": lognormal_profile,
        "radial_patch_summary": radial_patch_summary,
        "radial_patch_table": radial_patch_table,
        "survivability_map": survivability_map,
        "powerlaw_profile": powerlaw_profile,
        "summary": summary,
    }


def total_dissolution_time_myr(
    initial_mass_msun: float,
    r_apo_kpc: float,
    eccentricity: float,
    mean_initial_stellar_mass_msun: float = MEAN_INITIAL_STELLAR_MASS_MSUN,
    circular_speed_kms: float = MILKY_WAY_CIRCULAR_SPEED_KMS,
) -> float:
    n_initial = initial_mass_msun / mean_initial_stellar_mass_msun
    coulomb_argument = np.log(0.02 * n_initial)
    if coulomb_argument <= 0.0:
        return np.nan
    return (
        1.35
        * np.power(initial_mass_msun / coulomb_argument, 0.75)
        * r_apo_kpc
        * np.power(circular_speed_kms / 240.0, -1.0)
        * (1.0 - eccentricity)
    )


def survival_mass_cut_msun(r_apo_kpc: float, eccentricity: float, age_myr: float = AGE_MYR) -> float:
    def objective(log10_mass: float) -> float:
        mass = np.power(10.0, log10_mass)
        return total_dissolution_time_myr(mass, r_apo_kpc, eccentricity) - age_myr

    low = 2.0
    high = 8.5
    low_value = objective(low)
    high_value = objective(high)

    if np.isnan(low_value) or np.isnan(high_value):
        return np.nan
    if low_value >= 0.0:
        return np.power(10.0, low)
    if high_value <= 0.0:
        return np.power(10.0, high)
    root = optimize.brentq(objective, low, high)
    return np.power(10.0, root)


def fit_truncated_lognormal(catalog: pd.DataFrame) -> LognormalFitResult:
    log_masses = catalog["log_initial_mass_msun"].to_numpy()
    log_cuts = catalog["log_survival_mass_cut_msun"].to_numpy()

    def negative_log_likelihood(params: np.ndarray) -> float:
        mu, log_sigma, selection_offset_dex = params
        sigma = np.exp(log_sigma)
        effective_log_cuts = log_cuts + selection_offset_dex
        if np.any(log_masses < effective_log_cuts):
            return LARGE_PENALTY
        survival = 1.0 - stats.norm.cdf(effective_log_cuts, loc=mu, scale=sigma)
        if np.any(survival <= 0.0):
            return LARGE_PENALTY
        log_pdf = stats.norm.logpdf(log_masses, loc=mu, scale=sigma)
        return -np.sum(log_pdf - np.log(survival))

    minimum_offset = float(np.nanmin(log_masses - log_cuts)) - 1.0e-3
    initial_guess = np.array(
        [
            float(np.nanmedian(log_masses)),
            np.log(max(float(np.nanstd(log_masses)), 0.2)),
            min(minimum_offset, -0.2),
        ]
    )
    result = optimize.minimize(
        negative_log_likelihood,
        initial_guess,
        method="L-BFGS-B",
        bounds=[(3.0, 7.5), (-3.0, 1.5), (-1.5, 0.0)],
    )
    mu_fit = float(result.x[0])
    sigma_fit = float(np.exp(result.x[1]))
    selection_offset_dex = float(result.x[2])
    return LognormalFitResult(
        model_name="truncated_lognormal",
        mu_log10_msun=mu_fit,
        sigma_log10_msun=sigma_fit,
        selection_offset_dex=selection_offset_dex,
        log_likelihood=float(-result.fun),
        n_clusters=len(catalog),
    )


def survival_probability_lognormal(log_cuts: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.clip(1.0 - stats.norm.cdf(log_cuts, loc=mu, scale=sigma), 1.0e-9, 1.0)


def fit_truncated_powerlaw(
    catalog: pd.DataFrame,
    mass_min_msun: float,
    mass_max_msun: float,
) -> PowerLawFitResult:
    masses = catalog["initial_mass_msun"].to_numpy()
    cuts = catalog["survival_mass_cut_msun"].to_numpy()

    def negative_log_likelihood(beta: float) -> float:
        effective_cuts = cuts * np.power(10.0, selection_offset_dex)
        if np.any(masses < np.maximum(effective_cuts, mass_min_msun)) or np.any(masses > mass_max_msun):
            return LARGE_PENALTY
        truncated_norm = np.array(
            [powerlaw_integral(max(cut, mass_min_msun), mass_max_msun, beta) for cut in effective_cuts]
        )
        if np.any(truncated_norm <= 0.0):
            return LARGE_PENALTY
        return -np.sum(beta * np.log(masses) - np.log(truncated_norm))

    minimum_offset = float(np.nanmin(np.log10(masses) - np.log10(cuts))) - 1.0e-3
    selection_offset_dex = min(minimum_offset, -0.2)
    result = optimize.minimize_scalar(negative_log_likelihood, bounds=(-4.5, 0.5), method="bounded")
    return PowerLawFitResult(
        model_name="truncated_powerlaw",
        beta=float(result.x),
        mass_min_msun=mass_min_msun,
        mass_max_msun=mass_max_msun,
        selection_offset_dex=selection_offset_dex,
        log_likelihood=float(-result.fun),
        n_clusters=len(catalog),
    )


def survival_probability_powerlaw(
    cuts_msun: np.ndarray,
    beta: float,
    mass_min_msun: float,
    mass_max_msun: float,
) -> np.ndarray:
    total = powerlaw_integral(mass_min_msun, mass_max_msun, beta)
    survival = np.array(
        [powerlaw_integral(max(cut, mass_min_msun), mass_max_msun, beta) / total for cut in cuts_msun]
    )
    return np.clip(survival, 1.0e-9, 1.0)


def powerlaw_integral(lower: float, upper: float, beta: float) -> float:
    if lower >= upper:
        return 0.0
    exponent = beta + 1.0
    if np.isclose(exponent, 0.0):
        return float(np.log(upper / lower))
    return float((upper**exponent - lower**exponent) / exponent)


def estimate_radial_profile(
    catalog: pd.DataFrame,
    survival_probability_column: str,
    bins: tuple[float, ...] = (0.0, 3.0, 15.0, np.inf),
) -> pd.DataFrame:
    working = catalog.copy()
    working["inverse_probability_weight"] = 1.0 / working[survival_probability_column]
    working["radial_bin"] = pd.cut(
        working["semi_major_axis_kpc"],
        bins=bins,
        labels=["inner_<3kpc", "mid_3_15kpc", "outer_>15kpc"],
        right=False,
    )
    grouped_rows = []
    for radial_bin, group in working.groupby("radial_bin", observed=False):
        if len(group) == 0:
            continue
        estimated_initial_count = group["inverse_probability_weight"].sum()
        grouped_rows.append(
            {
                "radial_bin": str(radial_bin),
                "n_observed_survivors": int(len(group)),
                "estimated_initial_count": float(estimated_initial_count),
                "estimated_destroyed_count": float(estimated_initial_count - len(group)),
                "mean_survival_probability": float(group[survival_probability_column].mean()),
                "median_survival_mass_cut_msun": float(group["survival_mass_cut_msun"].median()),
                "median_initial_mass_msun": float(group["initial_mass_msun"].median()),
                "median_semi_major_axis_kpc": float(group["semi_major_axis_kpc"].median()),
                "sum_initial_mass_weighted_msun": float(
                    (group["initial_mass_msun"] * group["inverse_probability_weight"]).sum()
                ),
            }
        )
    return pd.DataFrame(grouped_rows)


def estimate_lognormal_imf_patches(
    catalog: pd.DataFrame,
    lognormal_result: LognormalFitResult,
    n_radius_bins: int = 5,
    mass_bin_width_dex: float = 0.25,
    completeness_threshold: float = 0.5,
    min_expected_initial_count_per_mass_bin: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = catalog.copy()
    working["effective_log_survival_cut_msun"] = (
        working["log_survival_mass_cut_msun"] + lognormal_result.selection_offset_dex
    )

    radius_edges_log10 = np.quantile(
        np.log10(working["semi_major_axis_kpc"].to_numpy()),
        np.linspace(0.0, 1.0, n_radius_bins + 1),
    )
    working["radius_bin_index_5"] = np.digitize(
        np.log10(working["semi_major_axis_kpc"].to_numpy()),
        bins=radius_edges_log10[1:-1],
        right=False,
    )

    radius_bin_labels = [
        format_radius_bin_label(
            lower_kpc=float(np.power(10.0, radius_edges_log10[index])),
            upper_kpc=float(np.power(10.0, radius_edges_log10[index + 1])),
        )
        for index in range(n_radius_bins)
    ]
    working["radius_bin_label_5"] = [
        radius_bin_labels[index] for index in working["radius_bin_index_5"].to_numpy()
    ]

    mass_bin_factor = 1.0 / mass_bin_width_dex
    mass_min = max(
        3.0,
        float(np.floor(working["log_initial_mass_msun"].min() * mass_bin_factor) / mass_bin_factor),
    )
    mass_max = min(
        7.5,
        float(
            np.ceil(working["log_initial_mass_msun"].max() * mass_bin_factor) / mass_bin_factor
            + mass_bin_width_dex
        ),
    )
    mass_edges = np.arange(mass_min, mass_max + 0.5 * mass_bin_width_dex, mass_bin_width_dex)
    mass_centers = 0.5 * (mass_edges[:-1] + mass_edges[1:])
    integration_grid = np.linspace(mass_edges[0], mass_edges[-1], 4000)
    global_phi_grid = lognormal_pdf_per_dex(
        integration_grid,
        lognormal_result.mu_log10_msun,
        lognormal_result.sigma_log10_msun,
    )
    global_phi_centers = lognormal_pdf_per_dex(
        mass_centers,
        lognormal_result.mu_log10_msun,
        lognormal_result.sigma_log10_msun,
    )

    summary_rows: list[dict[str, object]] = []
    patch_rows: list[dict[str, object]] = []

    for radius_bin_index, radius_bin_label in enumerate(radius_bin_labels):
        group = working.loc[working["radius_bin_index_5"] == radius_bin_index].copy()
        effective_cuts = group["effective_log_survival_cut_msun"].to_numpy()
        completeness_grid = np.mean(
            integration_grid[:, None] >= effective_cuts[None, :],
            axis=1,
        )
        survivor_fraction = float(np.trapezoid(global_phi_grid * completeness_grid, integration_grid))
        estimated_initial_count = float(len(group) / survivor_fraction)

        completeness_centers = np.mean(mass_centers[:, None] >= effective_cuts[None, :], axis=1)
        expected_initial_count_per_mass_bin = (
            estimated_initial_count * global_phi_centers * mass_bin_width_dex
        )
        expected_survivor_count_per_mass_bin = (
            expected_initial_count_per_mass_bin * completeness_centers
        )
        observed_counts, _ = np.histogram(group["log_initial_mass_msun"], bins=mass_edges)

        corrected_density = np.full_like(mass_centers, np.nan, dtype=float)
        corrected_density_error = np.full_like(mass_centers, np.nan, dtype=float)
        valid = completeness_centers > 0.0
        corrected_density[valid] = (
            observed_counts[valid]
            / (estimated_initial_count * completeness_centers[valid] * mass_bin_width_dex)
        )
        corrected_density_error[valid] = (
            np.sqrt(observed_counts[valid])
            / (estimated_initial_count * completeness_centers[valid] * mass_bin_width_dex)
        )

        leverage_mask = (
            (completeness_centers >= completeness_threshold)
            & (expected_initial_count_per_mass_bin >= min_expected_initial_count_per_mass_bin)
        )
        patch_mask = leverage_mask & (observed_counts > 0)

        probed_low = np.nan
        probed_high = np.nan
        if np.any(leverage_mask):
            probed_low = float(mass_edges[np.where(leverage_mask)[0][0]])
            probed_high = float(mass_edges[np.where(leverage_mask)[0][-1] + 1])

        summary_rows.append(
            {
                "radius_bin_index_5": radius_bin_index,
                "radius_bin_label_5": radius_bin_label,
                "radius_min_kpc": float(np.power(10.0, radius_edges_log10[radius_bin_index])),
                "radius_max_kpc": float(np.power(10.0, radius_edges_log10[radius_bin_index + 1])),
                "n_survivors": int(len(group)),
                "survivor_fraction_lognormal": survivor_fraction,
                "estimated_initial_count_lognormal": estimated_initial_count,
                "probed_log_mass_min": probed_low,
                "probed_log_mass_max": probed_high,
            }
        )

        for mass_bin_index, mass_center in enumerate(mass_centers):
            patch_rows.append(
                {
                    "radius_bin_index_5": radius_bin_index,
                    "radius_bin_label_5": radius_bin_label,
                    "mass_bin_index": mass_bin_index,
                    "log_mass_left_edge": float(mass_edges[mass_bin_index]),
                    "log_mass_right_edge": float(mass_edges[mass_bin_index + 1]),
                    "log_mass_center": float(mass_center),
                    "observed_survivor_count": int(observed_counts[mass_bin_index]),
                    "completeness": float(completeness_centers[mass_bin_index]),
                    "expected_initial_count_per_mass_bin": float(
                        expected_initial_count_per_mass_bin[mass_bin_index]
                    ),
                    "expected_survivor_count_per_mass_bin": float(
                        expected_survivor_count_per_mass_bin[mass_bin_index]
                    ),
                    "corrected_density_per_dex": float(corrected_density[mass_bin_index]),
                    "corrected_density_error_per_dex": float(corrected_density_error[mass_bin_index]),
                    "global_lognormal_density_per_dex": float(global_phi_centers[mass_bin_index]),
                    "is_in_leverage_range": bool(leverage_mask[mass_bin_index]),
                    "is_patch_bin": bool(patch_mask[mass_bin_index]),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(patch_rows)


def lognormal_pdf_per_dex(log_masses: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return stats.norm.pdf(log_masses, loc=mu, scale=sigma)


def format_radius_bin_label(lower_kpc: float, upper_kpc: float) -> str:
    return f"{lower_kpc:.2f} <= a < {upper_kpc:.2f} kpc"


def estimate_survivability_map(
    catalog: pd.DataFrame,
    selection_offset_dex: float,
    n_radius_grid: int = 160,
    n_mass_grid: int = 180,
    bandwidth_log10_a_dex: float = 0.18,
) -> dict[str, object]:
    working = catalog.copy()
    log_a_data = np.log10(working["semi_major_axis_kpc"].to_numpy())
    log_cut_data = working["log_survival_mass_cut_msun"].to_numpy() + selection_offset_dex

    log_a_grid = np.linspace(log_a_data.min(), log_a_data.max(), n_radius_grid)
    log_mass_min = min(3.5, float(np.floor(working["log_initial_mass_msun"].min() * 10.0) / 10.0))
    log_mass_max = max(7.3, float(np.ceil(working["log_initial_mass_msun"].max() * 10.0) / 10.0))
    log_mass_grid = np.linspace(log_mass_min, log_mass_max, n_mass_grid)

    weights = np.exp(
        -0.5 * np.square((log_a_grid[:, None] - log_a_data[None, :]) / bandwidth_log10_a_dex)
    )
    weights_sum = np.clip(weights.sum(axis=1, keepdims=True), 1.0e-12, None)
    weights = weights / weights_sum

    indicators = log_mass_grid[:, None] >= log_cut_data[None, :]
    survival_probability = indicators @ weights.T

    rows = []
    for mass_index, log_mass in enumerate(log_mass_grid):
        for radius_index, log_a in enumerate(log_a_grid):
            rows.append(
                {
                    "log_initial_mass_msun": float(log_mass),
                    "semi_major_axis_kpc": float(np.power(10.0, log_a)),
                    "log10_semi_major_axis_kpc": float(log_a),
                    "survival_probability": float(survival_probability[mass_index, radius_index]),
                }
            )

    return {
        "table": pd.DataFrame(rows),
        "log_mass_grid": log_mass_grid,
        "semi_major_axis_grid_kpc": np.power(10.0, log_a_grid),
        "log10_semi_major_axis_grid_kpc": log_a_grid,
        "survival_probability": survival_probability,
        "bandwidth_log10_a_dex": bandwidth_log10_a_dex,
    }
