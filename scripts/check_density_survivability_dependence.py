from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class OLSResult:
    coefficients: np.ndarray
    stderr: np.ndarray
    t_value: np.ndarray
    p_value: np.ndarray
    rss: float
    bic: float
    aic: float
    r_squared: float
    adjusted_r_squared: float
    sigma: float
    fitted: np.ndarray


def fit_ols(design: np.ndarray, response: np.ndarray) -> OLSResult:
    coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
    fitted = design @ coefficients
    residuals = response - fitted
    n_samples, n_parameters = design.shape
    rss = float(np.sum(residuals**2))
    tss = float(np.sum((response - np.mean(response)) ** 2))
    sigma2 = rss / max(n_samples - n_parameters, 1)
    covariance = sigma2 * np.linalg.inv(design.T @ design)
    stderr = np.sqrt(np.diag(covariance))
    t_value = coefficients / np.clip(stderr, 1.0e-12, None)
    dof = max(n_samples - n_parameters, 1)
    p_value = 2.0 * stats.t.sf(np.abs(t_value), df=dof)
    log_like = -0.5 * n_samples * (np.log(2.0 * np.pi) + 1.0 + np.log(rss / n_samples))
    bic = float(n_parameters * np.log(n_samples) - 2.0 * log_like)
    aic = float(2.0 * n_parameters - 2.0 * log_like)
    r_squared = 1.0 - rss / max(tss, 1.0e-12)
    adjusted_r_squared = 1.0 - (1.0 - r_squared) * (n_samples - 1.0) / max(n_samples - n_parameters, 1)
    return OLSResult(
        coefficients=coefficients,
        stderr=stderr,
        t_value=t_value,
        p_value=p_value,
        rss=rss,
        bic=bic,
        aic=aic,
        r_squared=float(r_squared),
        adjusted_r_squared=float(adjusted_r_squared),
        sigma=float(np.sqrt(sigma2)),
        fitted=fitted,
    )


def residualize(target: np.ndarray, controls: np.ndarray) -> np.ndarray:
    coefficients, _, _, _ = np.linalg.lstsq(controls, target, rcond=None)
    return target - controls @ coefficients


def summarize_predictor_dependence(
    catalog: pd.DataFrame,
    predictor_name: str,
    predictor_values: np.ndarray,
) -> dict[str, object]:
    finite = np.isfinite(np.asarray(predictor_values, dtype=float))
    finite &= np.isfinite(catalog["log_initial_mass_msun"].to_numpy(dtype=float))
    finite &= np.isfinite(catalog["semi_major_axis_kpc"].to_numpy(dtype=float))
    finite &= np.isfinite(catalog["remaining_dissolution_time_gyr"].to_numpy(dtype=float))
    finite &= catalog["remaining_dissolution_time_gyr"].to_numpy(dtype=float) > 0.0

    working = catalog.loc[finite, [
        "cluster_label",
        "log_initial_mass_msun",
        "semi_major_axis_kpc",
        "remaining_dissolution_time_gyr",
    ]].copy()
    working["predictor"] = np.asarray(predictor_values, dtype=float)[finite]

    log_mass = working["log_initial_mass_msun"].to_numpy(dtype=float)
    log_a = np.log10(working["semi_major_axis_kpc"].to_numpy(dtype=float))
    predictor = working["predictor"].to_numpy(dtype=float)
    log_remaining = np.log10(working["remaining_dissolution_time_gyr"].to_numpy(dtype=float))

    base_design = np.column_stack([np.ones(len(working)), log_mass, log_a])
    extended_design = np.column_stack([np.ones(len(working)), log_mass, log_a, predictor])

    base_fit = fit_ols(base_design, log_remaining)
    extended_fit = fit_ols(extended_design, log_remaining)

    residual_remaining = residualize(log_remaining, base_design)
    residual_predictor = residualize(predictor, base_design)
    partial_pearson = stats.pearsonr(residual_predictor, residual_remaining)
    partial_spearman = stats.spearmanr(residual_predictor, residual_remaining)

    density_index = 3
    return {
        "predictor_name": predictor_name,
        "n_clusters": int(len(working)),
        "base_model": {
            "bic": base_fit.bic,
            "aic": base_fit.aic,
            "r_squared": base_fit.r_squared,
            "adjusted_r_squared": base_fit.adjusted_r_squared,
            "sigma_dex": base_fit.sigma,
        },
        "extended_model": {
            "bic": extended_fit.bic,
            "aic": extended_fit.aic,
            "r_squared": extended_fit.r_squared,
            "adjusted_r_squared": extended_fit.adjusted_r_squared,
            "sigma_dex": extended_fit.sigma,
            "predictor_coefficient": float(extended_fit.coefficients[density_index]),
            "predictor_stderr": float(extended_fit.stderr[density_index]),
            "predictor_t_value": float(extended_fit.t_value[density_index]),
            "predictor_p_value": float(extended_fit.p_value[density_index]),
        },
        "delta_bic_extended_minus_base": float(extended_fit.bic - base_fit.bic),
        "delta_aic_extended_minus_base": float(extended_fit.aic - base_fit.aic),
        "delta_r_squared": float(extended_fit.r_squared - base_fit.r_squared),
        "partial_correlation": {
            "pearson_r": float(partial_pearson.statistic),
            "pearson_p_value": float(partial_pearson.pvalue),
            "spearman_rho": float(partial_spearman.statistic),
            "spearman_p_value": float(partial_spearman.pvalue),
        },
    }


def summarize_density_dependence(catalog: pd.DataFrame, density_column: str) -> dict[str, object]:
    result = summarize_predictor_dependence(
        catalog=catalog,
        predictor_name=density_column,
        predictor_values=catalog[density_column].to_numpy(dtype=float),
    )
    result["density_column"] = density_column
    result["predictor_kind"] = "catalog_density"
    return result


def build_proxy_initial_half_mass_density(catalog: pd.DataFrame) -> np.ndarray:
    log_initial_mass = catalog["log_initial_mass_msun"].to_numpy(dtype=float)
    half_mass_radius = catalog["half_mass_radius_pc"].to_numpy(dtype=float)
    volume_factor = np.log10((4.0 / 3.0) * np.pi)
    return log_initial_mass - volume_factor - 3.0 * np.log10(half_mass_radius)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog.csv"

    catalog = pd.read_csv(catalog_path)
    outputs_dir = project_root / "outputs" / "tables"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    summaries = [
        summarize_density_dependence(catalog, "log_half_mass_density"),
        summarize_density_dependence(catalog, "log_core_density"),
        summarize_predictor_dependence(
            catalog=catalog,
            predictor_name="log_core_radius_pc",
            predictor_values=np.log10(catalog["core_radius_pc"].to_numpy(dtype=float)),
        ),
        summarize_predictor_dependence(
            catalog=catalog,
            predictor_name="log_half_mass_radius_pc",
            predictor_values=np.log10(catalog["half_mass_radius_pc"].to_numpy(dtype=float)),
        ),
        summarize_predictor_dependence(
            catalog=catalog,
            predictor_name="log_tidal_radius_pc",
            predictor_values=np.log10(catalog["tidal_radius_pc"].to_numpy(dtype=float)),
        ),
        summarize_predictor_dependence(
            catalog=catalog,
            predictor_name="log_proxy_initial_half_mass_density_current_rh",
            predictor_values=build_proxy_initial_half_mass_density(catalog),
        ),
    ]

    summary_rows = []
    for payload in summaries:
        row = {
            "predictor_name": payload["predictor_name"],
            "n_clusters": payload["n_clusters"],
            "delta_bic_extended_minus_base": payload["delta_bic_extended_minus_base"],
            "delta_aic_extended_minus_base": payload["delta_aic_extended_minus_base"],
            "delta_r_squared": payload["delta_r_squared"],
            "predictor_coefficient": payload["extended_model"]["predictor_coefficient"],
            "predictor_p_value": payload["extended_model"]["predictor_p_value"],
            "partial_pearson_r": payload["partial_correlation"]["pearson_r"],
            "partial_pearson_p_value": payload["partial_correlation"]["pearson_p_value"],
            "partial_spearman_rho": payload["partial_correlation"]["spearman_rho"],
            "partial_spearman_p_value": payload["partial_correlation"]["spearman_p_value"],
        }
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(
        outputs_dir / "density_dependence_survivability_summary.csv",
        index=False,
    )
    (outputs_dir / "density_dependence_survivability_summary.json").write_text(json.dumps(summaries, indent=2))

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
