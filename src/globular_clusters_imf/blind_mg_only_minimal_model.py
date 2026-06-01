from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .blind_mixture_model import (
    BlindBkComparisonSummary,
    build_blind_vs_bk_group_table,
    build_blind_vs_bk_summary,
    build_fixed_abs_longitude_selection_payload,
)
from .joint_model import JointModelSpec, JointLikelihoodContext, fit_single_joint_model

TINY = 1.0e-300
MIX_FRACTION_FLOOR = 0.05
DEFAULT_OUTPUT_PREFIX = "blind_mg_only_minimal_mixture"
DEFAULT_DETECTABILITY_SUMMARY = (
    "outputs/tables/joint_fixed_survival_detectability_abs_longitude_em_summary.json"
)


@dataclass(frozen=True)
class MgObservationContext:
    mgfe: np.ndarray
    mgfe_err: np.ndarray
    has_mgfe: np.ndarray
    log_a_standardized_all: np.ndarray
    log_a_standardized_mg: np.ndarray


@dataclass
class BlindMinimalMgFitResult:
    model_class: str
    success: bool
    chemistry_log_likelihood: float
    aic: float
    bic: float
    delta_bic: float
    n_parameters: int
    n_clusters_with_mg: int
    total_initial_count: float
    total_initial_stellar_mass_msun: float
    selection_fraction: float
    raw_survival_fraction: float
    mean_detectability: float
    imf_family: str
    radial_model: str
    component_mix_fraction_concentrated: float
    component_mix_fraction_extended: float
    component_initial_count_concentrated: float
    component_initial_count_extended: float
    expected_observed_count_concentrated: float
    expected_observed_count_extended: float
    optimizer_message: str
    chemistry_parameters_json: str


def fit_minimal_mg_only_models(
    catalog: pd.DataFrame,
    project_root: Path,
    output_root: Path,
    detectability_summary_path: Path | None = None,
    output_prefix: str = DEFAULT_OUTPUT_PREFIX,
) -> dict[str, object]:
    selection_payload = build_fixed_abs_longitude_selection_payload(
        catalog=catalog,
        project_root=project_root,
        detectability_summary_path=detectability_summary_path,
    )
    summary_path = (
        detectability_summary_path
        if detectability_summary_path is not None
        else project_root / DEFAULT_DETECTABILITY_SUMMARY
    )
    detectability_summary = json.loads(summary_path.read_text())
    best_joint_model = detectability_summary["best_joint_model"]
    reference_spec = JointModelSpec(
        imf_family=str(best_joint_model["imf_family"]),
        radial_model=str(best_joint_model["radial_model"]),
    )
    base_payload = fit_single_joint_model(
        context=selection_payload["selection_context"],
        spec=reference_spec,
    )
    base_payload["context"] = selection_payload["selection_context"]
    mg_context = build_mg_observation_context(catalog, selection_payload["selection_context"])

    payloads = [
        fit_single_mg_gaussian_model(base_payload, mg_context),
        fit_two_component_mg_mixture_model(base_payload, mg_context),
    ]

    summary_table = pd.DataFrame([asdict(payload["summary"]) for payload in payloads]).sort_values(
        ["bic", "aic"],
        ascending=[True, True],
    ).reset_index(drop=True)
    best_bic = float(summary_table["bic"].min())
    summary_table["delta_bic"] = summary_table["bic"] - best_bic
    for payload in payloads:
        payload["summary"].delta_bic = float(payload["summary"].bic - best_bic)

    best_payload = min(payloads, key=lambda item: item["summary"].bic)
    posterior_table = build_mg_posterior_probability_table(best_payload, catalog, mg_context)
    bk_comparison = build_blind_vs_bk_summary(posterior_table)
    bk_group_table = build_blind_vs_bk_group_table(posterior_table)
    radial_grid_table = build_component_radial_grid_table(best_payload, base_payload["model"], selection_payload["selection_context"])
    mg_grid_table = build_mg_density_grid_table(best_payload, mg_context)
    summary_json = {
        "selection_payload": selection_payload["summary_payload"],
        "base_model": {
            "imf_family": base_payload["summary"].imf_family,
            "radial_model": base_payload["summary"].radial_model,
            "log_likelihood": float(base_payload["summary"].log_likelihood),
            "bic": float(base_payload["summary"].bic),
            "total_initial_count": float(base_payload["summary"].total_initial_count),
            "survival_fraction": float(base_payload["summary"].survival_fraction),
            "imf_parameters": json.loads(base_payload["summary"].imf_parameters_json),
            "radial_parameters": json.loads(base_payload["summary"].radial_parameters_json),
        },
        "mg_summary": {
            "n_clusters_total": int(len(catalog)),
            "n_clusters_with_mg": int(mg_context.has_mgfe.sum()),
        },
        "all_models_ranked": summary_table.to_dict(orient="records"),
        "best_model": asdict(best_payload["summary"]),
        "best_model_bk_comparison": asdict(bk_comparison),
        "delta_log_likelihood_two_component_minus_single": float(
            payloads[1]["summary"].chemistry_log_likelihood - payloads[0]["summary"].chemistry_log_likelihood
        ),
        "delta_bic_two_component_minus_single": float(
            payloads[1]["summary"].bic - payloads[0]["summary"].bic
        ),
    }

    outputs_tables = output_root / "outputs" / "tables"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    summary_table.to_csv(outputs_tables / f"{output_prefix}_model_summary.csv", index=False)
    posterior_table.to_csv(outputs_tables / f"{output_prefix}_best_model_posterior_probabilities.csv", index=False)
    bk_group_table.to_csv(outputs_tables / f"{output_prefix}_best_model_vs_bk_groups.csv", index=False)
    radial_grid_table.to_csv(outputs_tables / f"{output_prefix}_best_model_component_radial_grid.csv", index=False)
    mg_grid_table.to_csv(outputs_tables / f"{output_prefix}_best_model_mg_density_grid.csv", index=False)
    (outputs_tables / f"{output_prefix}_model_summary.json").write_text(json.dumps(summary_json, indent=2))

    return {
        "selection_payload": selection_payload,
        "base_payload": base_payload,
        "mg_context": mg_context,
        "summary_table": summary_table,
        "best_payload": best_payload,
        "posterior_table": posterior_table,
        "bk_comparison": bk_comparison,
        "bk_group_table": bk_group_table,
        "radial_grid_table": radial_grid_table,
        "mg_grid_table": mg_grid_table,
        "all_payloads": payloads,
    }


def build_mg_observation_context(
    catalog: pd.DataFrame,
    selection_context: JointLikelihoodContext,
) -> MgObservationContext:
    mgfe = pd.to_numeric(catalog.get("mgfe_combined"), errors="coerce").to_numpy(dtype=float)
    mgfe_err = pd.to_numeric(catalog.get("mgfe_combined_err"), errors="coerce").to_numpy(dtype=float)
    has_mgfe = np.isfinite(mgfe) & np.isfinite(mgfe_err) & (mgfe_err > 0.0)
    log_a_standardized_all = (
        selection_context.log_a_data - selection_context.log_a_mean
    ) / selection_context.log_a_std
    return MgObservationContext(
        mgfe=mgfe,
        mgfe_err=mgfe_err,
        has_mgfe=has_mgfe,
        log_a_standardized_all=log_a_standardized_all,
        log_a_standardized_mg=log_a_standardized_all[has_mgfe],
    )


def fit_single_mg_gaussian_model(
    base_payload: dict[str, object],
    mg_context: MgObservationContext,
) -> dict[str, object]:
    measured_mg = mg_context.mgfe[mg_context.has_mgfe]
    measured_err = mg_context.mgfe_err[mg_context.has_mgfe]
    start = np.array(
        [
            float(np.nanmedian(measured_mg)),
            float(np.log(max(np.nanstd(measured_mg), 0.05))),
        ],
        dtype=float,
    )
    result = optimize.minimize(
        lambda params: single_mg_negative_log_likelihood(params, measured_mg, measured_err),
        x0=start,
        method="L-BFGS-B",
        bounds=[(-1.0, 1.0), (np.log(0.01), np.log(0.6))],
    )
    mu, log_sigma = np.asarray(result.x, dtype=float)
    sigma = float(np.exp(log_sigma))
    chemistry_log_likelihood = -float(result.fun)
    model = {
        "model_class": "single_mg_gaussian",
        "mu_mgfe": float(mu),
        "sigma_mgfe": sigma,
        "chemistry_log_likelihood": chemistry_log_likelihood,
        "w_all": np.ones_like(mg_context.log_a_standardized_all),
        "w_grid": np.ones_like(base_payload["model"]["radial_density_grid"]),
        "mix_fraction_concentrated": 1.0,
        "mix_fraction_extended": 0.0,
    }
    summary = build_mg_model_summary(
        model=model,
        n_parameters=2,
        n_clusters_with_mg=int(mg_context.has_mgfe.sum()),
        success=bool(result.success),
        optimizer_message=str(result.message),
        base_payload=base_payload,
    )
    return {"summary": summary, "raw_parameters": np.asarray(result.x, dtype=float), "model": model}


def fit_two_component_mg_mixture_model(
    base_payload: dict[str, object],
    mg_context: MgObservationContext,
) -> dict[str, object]:
    measured_mg = mg_context.mgfe[mg_context.has_mgfe]
    measured_err = mg_context.mgfe_err[mg_context.has_mgfe]
    measured_x = mg_context.log_a_standardized_mg

    low_q, high_q = np.quantile(measured_mg, [0.25, 0.75])
    starts = [
        np.array([0.0, -2.0, high_q, low_q, np.log(0.06)], dtype=float),
        np.array([0.5, -1.0, high_q, low_q, np.log(0.08)], dtype=float),
        np.array([-0.5, -3.0, np.nanmedian(measured_mg) + 0.05, np.nanmedian(measured_mg) - 0.05, np.log(0.10)], dtype=float),
        np.array([0.0, -0.2, high_q, low_q, np.log(0.12)], dtype=float),
    ]
    bounds = [
        (-8.0, 8.0),
        (-10.0, 0.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
        (np.log(0.01), np.log(0.6)),
    ]

    best_result = None
    best_value = np.inf
    for start in starts:
        result = optimize.minimize(
            lambda params: two_component_mg_negative_log_likelihood(
                params,
                measured_mg=measured_mg,
                measured_err=measured_err,
                measured_x=measured_x,
            ),
            x0=start,
            method="L-BFGS-B",
            bounds=bounds,
        )
        if result.fun < best_value:
            best_value = float(result.fun)
            best_result = result
    if best_result is None:
        raise RuntimeError("Two-component Mg-only mixture optimization did not start.")

    c0, c1, mu_concentrated, mu_extended, log_sigma = np.asarray(best_result.x, dtype=float)
    sigma = float(np.exp(log_sigma))
    w_all = bounded_logistic_mix(c0 + c1 * mg_context.log_a_standardized_all)
    radial_grid = np.asarray(base_payload["model"]["radial_density_grid"], dtype=float)
    context = base_payload["context"]
    w_grid = bounded_logistic_mix(
        c0 + c1 * ((context.log_a_grid - context.log_a_mean) / context.log_a_std)
    )
    mix_fraction_concentrated = float(np.trapezoid(radial_grid * w_grid, context.log_a_grid))
    mix_fraction_concentrated = float(np.clip(mix_fraction_concentrated, TINY, 1.0 - TINY))
    chemistry_log_likelihood = -float(best_result.fun)
    model = {
        "model_class": "two_component_mg_mixture",
        "c0": float(c0),
        "c1": float(c1),
        "mu_mgfe_concentrated": float(mu_concentrated),
        "mu_mgfe_extended": float(mu_extended),
        "sigma_mgfe_shared": sigma,
        "chemistry_log_likelihood": chemistry_log_likelihood,
        "w_all": np.clip(w_all, TINY, 1.0 - TINY),
        "w_grid": np.clip(w_grid, TINY, 1.0 - TINY),
        "mix_fraction_concentrated": mix_fraction_concentrated,
        "mix_fraction_extended": float(1.0 - mix_fraction_concentrated),
    }
    summary = build_mg_model_summary(
        model=model,
        n_parameters=5,
        n_clusters_with_mg=int(mg_context.has_mgfe.sum()),
        success=bool(best_result.success),
        optimizer_message=str(best_result.message),
        base_payload=base_payload,
    )
    return {"summary": summary, "raw_parameters": np.asarray(best_result.x, dtype=float), "model": model}


def single_mg_negative_log_likelihood(
    params: np.ndarray,
    measured_mg: np.ndarray,
    measured_err: np.ndarray,
) -> float:
    mu, log_sigma = np.asarray(params, dtype=float)
    sigma = float(np.exp(log_sigma))
    density = mg_gaussian_density(measured_mg, measured_err, mu=mu, sigma=sigma)
    return float(-np.sum(np.log(np.clip(density, TINY, None))))


def two_component_mg_negative_log_likelihood(
    params: np.ndarray,
    measured_mg: np.ndarray,
    measured_err: np.ndarray,
    measured_x: np.ndarray,
) -> float:
    c0, c1, mu_concentrated, mu_extended, log_sigma = np.asarray(params, dtype=float)
    sigma = float(np.exp(log_sigma))
    w = bounded_logistic_mix(c0 + c1 * measured_x)
    concentrated = mg_gaussian_density(measured_mg, measured_err, mu=mu_concentrated, sigma=sigma)
    extended = mg_gaussian_density(measured_mg, measured_err, mu=mu_extended, sigma=sigma)
    density = w * concentrated + (1.0 - w) * extended
    return float(-np.sum(np.log(np.clip(density, TINY, None))))


def mg_gaussian_density(
    mg_values: np.ndarray,
    mg_err: np.ndarray,
    mu: float,
    sigma: float,
) -> np.ndarray:
    total_sigma = np.sqrt(np.square(float(sigma)) + np.square(np.asarray(mg_err, dtype=float)))
    return stats.norm.pdf(np.asarray(mg_values, dtype=float), loc=float(mu), scale=total_sigma)


def bounded_logistic_mix(raw: np.ndarray | float, floor: float = MIX_FRACTION_FLOOR) -> np.ndarray:
    raw_array = np.asarray(raw, dtype=float)
    return floor + (1.0 - 2.0 * floor) * special.expit(raw_array)


def build_mg_model_summary(
    model: dict[str, object],
    n_parameters: int,
    n_clusters_with_mg: int,
    success: bool,
    optimizer_message: str,
    base_payload: dict[str, object],
) -> BlindMinimalMgFitResult:
    base_model = base_payload["model"]
    context: JointLikelihoodContext = base_payload["context"]
    radial_grid = np.asarray(base_model["radial_density_grid"], dtype=float)
    imf_grid = np.asarray(base_model["imf_density_grid"], dtype=float)
    selection_grid = np.asarray(context.selection_probability_grid, dtype=float)
    total_initial_count = float(base_model["total_initial_count"])
    w_grid = np.asarray(model["w_grid"], dtype=float)

    concentrated_birth_intensity = total_initial_count * radial_grid * w_grid
    extended_birth_intensity = total_initial_count * radial_grid * (1.0 - w_grid)
    component_initial_count_concentrated = float(np.trapezoid(concentrated_birth_intensity, context.log_a_grid))
    component_initial_count_extended = float(np.trapezoid(extended_birth_intensity, context.log_a_grid))

    observed_intensity_grid = (
        total_initial_count
        * imf_grid[:, None]
        * radial_grid[None, :]
        * selection_grid
    )
    expected_observed_count_concentrated = float(
        np.trapezoid(
            np.trapezoid(observed_intensity_grid * w_grid[None, :], context.log_a_grid, axis=1),
            context.log_mass_grid,
        )
    )
    expected_observed_count_extended = float(
        np.trapezoid(
            np.trapezoid(observed_intensity_grid * (1.0 - w_grid)[None, :], context.log_a_grid, axis=1),
            context.log_mass_grid,
        )
    )
    total_initial_stellar_mass = compute_total_initial_stellar_mass(base_model, context)

    chemistry_parameters = {
        key: value
        for key, value in model.items()
        if key
        in {
            "mu_mgfe",
            "sigma_mgfe",
            "c0",
            "c1",
            "mu_mgfe_concentrated",
            "mu_mgfe_extended",
            "sigma_mgfe_shared",
        }
    }
    chemistry_parameters["mix_fraction_floor"] = MIX_FRACTION_FLOOR

    chemistry_log_likelihood = float(model["chemistry_log_likelihood"])
    return BlindMinimalMgFitResult(
        model_class=str(model["model_class"]),
        success=bool(success),
        chemistry_log_likelihood=chemistry_log_likelihood,
        aic=float(2 * n_parameters - 2 * chemistry_log_likelihood),
        bic=float(np.log(max(n_clusters_with_mg, 1)) * n_parameters - 2 * chemistry_log_likelihood),
        delta_bic=np.nan,
        n_parameters=int(n_parameters),
        n_clusters_with_mg=int(n_clusters_with_mg),
        total_initial_count=total_initial_count,
        total_initial_stellar_mass_msun=float(total_initial_stellar_mass),
        selection_fraction=float(base_model["selection_fraction"]),
        raw_survival_fraction=float(base_model["raw_survival_fraction"]),
        mean_detectability=float(base_model["selection_fraction"] / max(base_model["raw_survival_fraction"], 1.0e-12)),
        imf_family=str(base_model["spec"].imf_family),
        radial_model=str(base_model["spec"].radial_model),
        component_mix_fraction_concentrated=float(model["mix_fraction_concentrated"]),
        component_mix_fraction_extended=float(model["mix_fraction_extended"]),
        component_initial_count_concentrated=component_initial_count_concentrated,
        component_initial_count_extended=component_initial_count_extended,
        expected_observed_count_concentrated=expected_observed_count_concentrated,
        expected_observed_count_extended=expected_observed_count_extended,
        optimizer_message=str(optimizer_message),
        chemistry_parameters_json=json.dumps(chemistry_parameters),
    )


def compute_total_initial_stellar_mass(
    base_model: dict[str, object],
    context: JointLikelihoodContext,
) -> float:
    mass_grid = np.power(10.0, context.log_mass_grid)
    mean_initial_mass = float(np.trapezoid(mass_grid * base_model["imf_density_grid"], context.log_mass_grid))
    return float(base_model["total_initial_count"] * mean_initial_mass)


def build_mg_posterior_probability_table(
    best_payload: dict[str, object],
    catalog: pd.DataFrame,
    mg_context: MgObservationContext,
) -> pd.DataFrame:
    model = best_payload["model"]
    output = catalog.copy()
    if model["model_class"] == "single_mg_gaussian":
        p_concentrated = np.ones(len(output), dtype=float)
    else:
        w = np.asarray(model["w_all"], dtype=float)
        sigma = float(model["sigma_mgfe_shared"])
        concentrated = mg_gaussian_density(
            mg_context.mgfe,
            np.where(mg_context.has_mgfe, mg_context.mgfe_err, 0.0),
            mu=float(model["mu_mgfe_concentrated"]),
            sigma=sigma,
        )
        extended = mg_gaussian_density(
            mg_context.mgfe,
            np.where(mg_context.has_mgfe, mg_context.mgfe_err, 0.0),
            mu=float(model["mu_mgfe_extended"]),
            sigma=sigma,
        )
        numerator = w.copy()
        denominator = np.ones_like(w)
        measured = mg_context.has_mgfe
        numerator[measured] = w[measured] * concentrated[measured]
        denominator[measured] = numerator[measured] + (1.0 - w[measured]) * extended[measured]
        p_concentrated = np.clip(numerator / np.clip(denominator, TINY, None), TINY, 1.0)
    output["has_mgfe"] = mg_context.has_mgfe
    output["p_concentrated"] = p_concentrated
    output["p_extended"] = 1.0 - p_concentrated
    output["blind_component_label"] = np.where(output["p_concentrated"] >= 0.5, "concentrated", "extended")
    return output


def build_component_radial_grid_table(
    best_payload: dict[str, object],
    base_model: dict[str, object],
    context: JointLikelihoodContext,
) -> pd.DataFrame:
    radial_grid = np.asarray(base_model["radial_density_grid"], dtype=float)
    w_grid = np.asarray(best_payload["model"]["w_grid"], dtype=float)
    total_initial_count = float(base_model["total_initial_count"])
    return pd.DataFrame(
        {
            "semi_major_axis_kpc": np.power(10.0, context.log_a_grid),
            "log10_semi_major_axis_kpc": context.log_a_grid,
            "w_concentrated": w_grid,
            "total_radial_density_per_dex_a": radial_grid,
            "concentrated_radial_density_per_dex_a": radial_grid * w_grid,
            "extended_radial_density_per_dex_a": radial_grid * (1.0 - w_grid),
            "total_birth_intensity_per_dex_a": total_initial_count * radial_grid,
            "concentrated_birth_intensity_per_dex_a": total_initial_count * radial_grid * w_grid,
            "extended_birth_intensity_per_dex_a": total_initial_count * radial_grid * (1.0 - w_grid),
        }
    )


def build_mg_density_grid_table(
    best_payload: dict[str, object],
    mg_context: MgObservationContext,
) -> pd.DataFrame:
    measured = mg_context.mgfe[mg_context.has_mgfe]
    mg_grid = np.linspace(
        float(np.nanmin(measured)) - 0.1,
        float(np.nanmax(measured)) + 0.1,
        300,
    )
    rows = {"mgfe": mg_grid}
    model = best_payload["model"]
    if model["model_class"] == "single_mg_gaussian":
        rows["single_density"] = stats.norm.pdf(
            mg_grid,
            loc=float(model["mu_mgfe"]),
            scale=float(model["sigma_mgfe"]),
        )
    else:
        sigma = float(model["sigma_mgfe_shared"])
        mean_w = float(np.mean(model["w_all"][mg_context.has_mgfe]))
        concentrated = stats.norm.pdf(
            mg_grid,
            loc=float(model["mu_mgfe_concentrated"]),
            scale=sigma,
        )
        extended = stats.norm.pdf(
            mg_grid,
            loc=float(model["mu_mgfe_extended"]),
            scale=sigma,
        )
        rows["mean_weighted_mixture_density"] = mean_w * concentrated + (1.0 - mean_w) * extended
        rows["concentrated_density"] = concentrated
        rows["extended_density"] = extended
    return pd.DataFrame(rows)
