from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .blind_mixture_model import (
    build_blind_vs_bk_group_table,
    build_blind_vs_bk_summary,
    build_fixed_abs_longitude_selection_payload,
)
from .blind_mg_only_minimal_model import (
    BlindMinimalMgFitResult,
    MIX_FRACTION_FLOOR,
    TINY,
    bounded_logistic_mix,
    build_component_radial_grid_table,
    compute_total_initial_stellar_mass,
)
from .joint_model import JointLikelihoodContext, JointModelSpec, fit_single_joint_model

DEFAULT_OUTPUT_PREFIX = "blind_mg_feh_minimal_mixture"
DEFAULT_DETECTABILITY_SUMMARY = (
    "outputs/tables/joint_fixed_survival_detectability_abs_longitude_em_summary.json"
)


@dataclass(frozen=True)
class MgFeHObservationContext:
    mgfe: np.ndarray
    mgfe_err: np.ndarray
    feh_centered: np.ndarray
    feh_center: float
    has_mgfe: np.ndarray
    log_a_standardized_all: np.ndarray
    log_a_standardized_mg: np.ndarray


def fit_minimal_mg_feh_models(
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
    obs_context = build_mg_feh_observation_context(catalog, selection_payload["selection_context"])

    payloads = [
        fit_single_mg_feh_gaussian_model(base_payload, obs_context),
        fit_two_component_mg_feh_mixture_model(base_payload, obs_context),
    ]

    summary_table = pd.DataFrame([asdict(payload["summary"]) for payload in payloads]).sort_values(
        ["bic", "aic"], ascending=[True, True]
    ).reset_index(drop=True)
    best_bic = float(summary_table["bic"].min())
    summary_table["delta_bic"] = summary_table["bic"] - best_bic
    for payload in payloads:
        payload["summary"].delta_bic = float(payload["summary"].bic - best_bic)

    best_payload = min(payloads, key=lambda item: item["summary"].bic)
    posterior_table = build_mg_feh_posterior_probability_table(best_payload, catalog, obs_context)
    bk_comparison = build_blind_vs_bk_summary(posterior_table)
    bk_group_table = build_blind_vs_bk_group_table(posterior_table)
    radial_grid_table = build_component_radial_grid_table(
        best_payload,
        base_payload["model"],
        selection_payload["selection_context"],
    )
    mg_grid_table = build_mg_feh_density_grid_table(best_payload, obs_context)
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
        "mg_feh_summary": {
            "n_clusters_total": int(len(catalog)),
            "n_clusters_with_mg": int(obs_context.has_mgfe.sum()),
            "feh_center": float(obs_context.feh_center),
        },
        "all_models_ranked": summary_table.to_dict(orient="records"),
        "best_model": asdict(best_payload["summary"]),
        "best_model_bk_comparison": asdict(bk_comparison),
        "delta_log_likelihood_two_component_minus_single": float(
            payloads[1]["summary"].chemistry_log_likelihood - payloads[0]["summary"].chemistry_log_likelihood
        ),
        "delta_bic_two_component_minus_single": float(payloads[1]["summary"].bic - payloads[0]["summary"].bic),
    }

    outputs_tables = output_root / "outputs" / "tables"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    summary_table.to_csv(outputs_tables / f"{output_prefix}_model_summary.csv", index=False)
    posterior_table.to_csv(outputs_tables / f"{output_prefix}_best_model_posterior_probabilities.csv", index=False)
    bk_group_table.to_csv(outputs_tables / f"{output_prefix}_best_model_vs_bk_groups.csv", index=False)
    radial_grid_table.to_csv(outputs_tables / f"{output_prefix}_best_model_component_radial_grid.csv", index=False)
    mg_grid_table.to_csv(outputs_tables / f"{output_prefix}_best_model_mg_feh_density_grid.csv", index=False)
    (outputs_tables / f"{output_prefix}_model_summary.json").write_text(json.dumps(summary_json, indent=2))

    return {
        "selection_payload": selection_payload,
        "base_payload": base_payload,
        "obs_context": obs_context,
        "summary_table": summary_table,
        "best_payload": best_payload,
        "posterior_table": posterior_table,
        "bk_comparison": bk_comparison,
        "bk_group_table": bk_group_table,
        "radial_grid_table": radial_grid_table,
        "mg_grid_table": mg_grid_table,
        "all_payloads": payloads,
    }


def build_mg_feh_observation_context(
    catalog: pd.DataFrame,
    selection_context: JointLikelihoodContext,
) -> MgFeHObservationContext:
    mgfe = pd.to_numeric(catalog.get("mgfe_combined"), errors="coerce").to_numpy(dtype=float)
    mgfe_err = pd.to_numeric(catalog.get("mgfe_combined_err"), errors="coerce").to_numpy(dtype=float)
    feh = pd.to_numeric(catalog.get("local_feh"), errors="coerce").to_numpy(dtype=float)
    has_mgfe = np.isfinite(mgfe) & np.isfinite(mgfe_err) & (mgfe_err > 0.0) & np.isfinite(feh)
    feh_center = float(np.nanmean(feh[has_mgfe]))
    feh_centered = feh - feh_center
    log_a_standardized_all = (selection_context.log_a_data - selection_context.log_a_mean) / selection_context.log_a_std
    return MgFeHObservationContext(
        mgfe=mgfe,
        mgfe_err=mgfe_err,
        feh_centered=feh_centered,
        feh_center=feh_center,
        has_mgfe=has_mgfe,
        log_a_standardized_all=log_a_standardized_all,
        log_a_standardized_mg=log_a_standardized_all[has_mgfe],
    )


def fit_single_mg_feh_gaussian_model(
    base_payload: dict[str, object],
    obs_context: MgFeHObservationContext,
) -> dict[str, object]:
    measured_mg = obs_context.mgfe[obs_context.has_mgfe]
    measured_err = obs_context.mgfe_err[obs_context.has_mgfe]
    measured_feh = obs_context.feh_centered[obs_context.has_mgfe]
    start = np.array(
        [
            float(np.nanmedian(measured_mg)),
            0.0,
            float(np.log(max(np.nanstd(measured_mg), 0.05))),
        ],
        dtype=float,
    )
    result = optimize.minimize(
        lambda params: single_mg_feh_negative_log_likelihood(params, measured_mg, measured_err, measured_feh),
        x0=start,
        method="L-BFGS-B",
        bounds=[(-1.0, 1.0), (-0.5, 0.5), (np.log(0.01), np.log(0.6))],
    )
    mu, slope_feh, log_sigma = np.asarray(result.x, dtype=float)
    sigma = float(np.exp(log_sigma))
    chemistry_log_likelihood = -float(result.fun)
    model = {
        "model_class": "single_mg_feh_gaussian",
        "mu_mgfe": float(mu),
        "slope_feh": float(slope_feh),
        "sigma_mgfe": sigma,
        "chemistry_log_likelihood": chemistry_log_likelihood,
        "w_all": np.ones_like(obs_context.log_a_standardized_all),
        "w_grid": np.ones_like(base_payload["model"]["radial_density_grid"]),
        "mix_fraction_concentrated": 1.0,
        "mix_fraction_extended": 0.0,
    }
    summary = build_mg_feh_model_summary(
        model=model,
        n_parameters=3,
        n_clusters_with_mg=int(obs_context.has_mgfe.sum()),
        success=bool(result.success),
        optimizer_message=str(result.message),
        base_payload=base_payload,
    )
    return {"summary": summary, "raw_parameters": np.asarray(result.x, dtype=float), "model": model}


def fit_two_component_mg_feh_mixture_model(
    base_payload: dict[str, object],
    obs_context: MgFeHObservationContext,
) -> dict[str, object]:
    measured_mg = obs_context.mgfe[obs_context.has_mgfe]
    measured_err = obs_context.mgfe_err[obs_context.has_mgfe]
    measured_feh = obs_context.feh_centered[obs_context.has_mgfe]
    measured_x = obs_context.log_a_standardized_mg

    low_q, high_q = np.quantile(measured_mg, [0.25, 0.75])
    starts = [
        np.array([0.0, -2.0, high_q, low_q, 0.0, np.log(0.05), np.log(0.07)], dtype=float),
        np.array([0.5, -1.0, high_q, low_q, 0.05, np.log(0.06), np.log(0.09)], dtype=float),
        np.array([-0.5, -3.0, np.nanmedian(measured_mg) + 0.05, np.nanmedian(measured_mg) - 0.05, -0.05, np.log(0.08), np.log(0.10)], dtype=float),
        np.array([0.0, -0.2, high_q, low_q, 0.0, np.log(0.10), np.log(0.10)], dtype=float),
    ]
    bounds = [
        (-8.0, 8.0),
        (-10.0, 0.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
        (-0.5, 0.5),
        (np.log(0.01), np.log(0.6)),
        (np.log(0.01), np.log(0.6)),
    ]
    best_result = None
    best_value = np.inf
    for start in starts:
        result = optimize.minimize(
            lambda params: two_component_mg_feh_negative_log_likelihood(
                params,
                measured_mg=measured_mg,
                measured_err=measured_err,
                measured_feh=measured_feh,
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
        raise RuntimeError("Two-component Mg+Fe/H mixture optimization did not start.")

    c0, c1, mu_conc, mu_ext, slope_feh, log_sigma_conc, log_sigma_ext = np.asarray(best_result.x, dtype=float)
    sigma_conc = float(np.exp(log_sigma_conc))
    sigma_ext = float(np.exp(log_sigma_ext))
    w_all = bounded_logistic_mix(c0 + c1 * obs_context.log_a_standardized_all)
    radial_grid = np.asarray(base_payload["model"]["radial_density_grid"], dtype=float)
    context = base_payload["context"]
    w_grid = bounded_logistic_mix(c0 + c1 * ((context.log_a_grid - context.log_a_mean) / context.log_a_std))
    mix_fraction_concentrated = float(np.trapezoid(radial_grid * w_grid, context.log_a_grid))
    mix_fraction_concentrated = float(np.clip(mix_fraction_concentrated, TINY, 1.0 - TINY))
    chemistry_log_likelihood = -float(best_result.fun)
    model = {
        "model_class": "two_component_mg_feh_mixture",
        "c0": float(c0),
        "c1": float(c1),
        "mu_mgfe_concentrated": float(mu_conc),
        "mu_mgfe_extended": float(mu_ext),
        "slope_feh": float(slope_feh),
        "sigma_mgfe_concentrated": sigma_conc,
        "sigma_mgfe_extended": sigma_ext,
        "chemistry_log_likelihood": chemistry_log_likelihood,
        "w_all": np.clip(w_all, TINY, 1.0 - TINY),
        "w_grid": np.clip(w_grid, TINY, 1.0 - TINY),
        "mix_fraction_concentrated": mix_fraction_concentrated,
        "mix_fraction_extended": float(1.0 - mix_fraction_concentrated),
    }
    summary = build_mg_feh_model_summary(
        model=model,
        n_parameters=7,
        n_clusters_with_mg=int(obs_context.has_mgfe.sum()),
        success=bool(best_result.success),
        optimizer_message=str(best_result.message),
        base_payload=base_payload,
    )
    return {"summary": summary, "raw_parameters": np.asarray(best_result.x, dtype=float), "model": model}


def single_mg_feh_negative_log_likelihood(
    params: np.ndarray,
    measured_mg: np.ndarray,
    measured_err: np.ndarray,
    measured_feh: np.ndarray,
) -> float:
    mu, slope_feh, log_sigma = np.asarray(params, dtype=float)
    sigma = float(np.exp(log_sigma))
    density = mg_feh_gaussian_density(measured_mg, measured_err, measured_feh, mu=mu, slope_feh=slope_feh, sigma=sigma)
    return float(-np.sum(np.log(np.clip(density, TINY, None))))


def two_component_mg_feh_negative_log_likelihood(
    params: np.ndarray,
    measured_mg: np.ndarray,
    measured_err: np.ndarray,
    measured_feh: np.ndarray,
    measured_x: np.ndarray,
) -> float:
    c0, c1, mu_conc, mu_ext, slope_feh, log_sigma_conc, log_sigma_ext = np.asarray(params, dtype=float)
    sigma_conc = float(np.exp(log_sigma_conc))
    sigma_ext = float(np.exp(log_sigma_ext))
    w = bounded_logistic_mix(c0 + c1 * measured_x)
    conc = mg_feh_gaussian_density(measured_mg, measured_err, measured_feh, mu=mu_conc, slope_feh=slope_feh, sigma=sigma_conc)
    ext = mg_feh_gaussian_density(measured_mg, measured_err, measured_feh, mu=mu_ext, slope_feh=slope_feh, sigma=sigma_ext)
    density = w * conc + (1.0 - w) * ext
    return float(-np.sum(np.log(np.clip(density, TINY, None))))


def mg_feh_gaussian_density(
    mg_values: np.ndarray,
    mg_err: np.ndarray,
    feh_centered: np.ndarray,
    mu: float,
    slope_feh: float,
    sigma: float,
) -> np.ndarray:
    total_sigma = np.sqrt(np.square(float(sigma)) + np.square(np.asarray(mg_err, dtype=float)))
    mean = float(mu) + float(slope_feh) * np.asarray(feh_centered, dtype=float)
    return stats.norm.pdf(np.asarray(mg_values, dtype=float), loc=mean, scale=total_sigma)


def build_mg_feh_model_summary(
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
    observed_intensity_grid = total_initial_count * imf_grid[:, None] * radial_grid[None, :] * selection_grid
    expected_observed_count_concentrated = float(
        np.trapezoid(np.trapezoid(observed_intensity_grid * w_grid[None, :], context.log_a_grid, axis=1), context.log_mass_grid)
    )
    expected_observed_count_extended = float(
        np.trapezoid(np.trapezoid(observed_intensity_grid * (1.0 - w_grid)[None, :], context.log_a_grid, axis=1), context.log_mass_grid)
    )
    total_initial_stellar_mass = compute_total_initial_stellar_mass(base_model, context)
    chemistry_parameters = {
        key: value
        for key, value in model.items()
        if key in {
            "mu_mgfe",
            "slope_feh",
            "sigma_mgfe",
            "c0",
            "c1",
            "mu_mgfe_concentrated",
            "mu_mgfe_extended",
            "sigma_mgfe_concentrated",
            "sigma_mgfe_extended",
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


def build_mg_feh_posterior_probability_table(
    best_payload: dict[str, object],
    catalog: pd.DataFrame,
    obs_context: MgFeHObservationContext,
) -> pd.DataFrame:
    model = best_payload["model"]
    output = catalog.copy()
    if model["model_class"] == "single_mg_feh_gaussian":
        p_concentrated = np.ones(len(output), dtype=float)
    else:
        w = np.asarray(model["w_all"], dtype=float)
        conc = mg_feh_gaussian_density(
            obs_context.mgfe,
            np.where(obs_context.has_mgfe, obs_context.mgfe_err, 0.0),
            obs_context.feh_centered,
            mu=float(model["mu_mgfe_concentrated"]),
            slope_feh=float(model["slope_feh"]),
            sigma=float(model["sigma_mgfe_concentrated"]),
        )
        ext = mg_feh_gaussian_density(
            obs_context.mgfe,
            np.where(obs_context.has_mgfe, obs_context.mgfe_err, 0.0),
            obs_context.feh_centered,
            mu=float(model["mu_mgfe_extended"]),
            slope_feh=float(model["slope_feh"]),
            sigma=float(model["sigma_mgfe_extended"]),
        )
        numerator = w.copy()
        denominator = np.ones_like(w)
        measured = obs_context.has_mgfe
        numerator[measured] = w[measured] * conc[measured]
        denominator[measured] = numerator[measured] + (1.0 - w[measured]) * ext[measured]
        p_concentrated = np.clip(numerator / np.clip(denominator, TINY, None), TINY, 1.0)
    output["has_mgfe"] = obs_context.has_mgfe
    output["p_concentrated"] = p_concentrated
    output["p_extended"] = 1.0 - p_concentrated
    output["blind_component_label"] = np.where(output["p_concentrated"] >= 0.5, "concentrated", "extended")
    return output


def build_mg_feh_density_grid_table(
    best_payload: dict[str, object],
    obs_context: MgFeHObservationContext,
) -> pd.DataFrame:
    measured = obs_context.mgfe[obs_context.has_mgfe]
    mg_grid = np.linspace(float(np.nanmin(measured)) - 0.1, float(np.nanmax(measured)) + 0.1, 300)
    feh_grid = np.array([np.nanpercentile(obs_context.feh_centered[obs_context.has_mgfe], q) for q in (16, 50, 84)], dtype=float)
    rows: list[dict[str, float | str]] = []
    model = best_payload["model"]
    if model["model_class"] == "single_mg_feh_gaussian":
        for feh_value, label in zip(feh_grid, ("feh_p16", "feh_p50", "feh_p84"), strict=True):
            density = stats.norm.pdf(mg_grid, loc=float(model["mu_mgfe"]) + float(model["slope_feh"]) * feh_value, scale=float(model["sigma_mgfe"]))
            for mg_value, density_value in zip(mg_grid, density, strict=True):
                rows.append({"slice_label": label, "mgfe": mg_value, "density": density_value, "component_label": "single"})
    else:
        for feh_value, label in zip(feh_grid, ("feh_p16", "feh_p50", "feh_p84"), strict=True):
            conc = stats.norm.pdf(
                mg_grid,
                loc=float(model["mu_mgfe_concentrated"]) + float(model["slope_feh"]) * feh_value,
                scale=float(model["sigma_mgfe_concentrated"]),
            )
            ext = stats.norm.pdf(
                mg_grid,
                loc=float(model["mu_mgfe_extended"]) + float(model["slope_feh"]) * feh_value,
                scale=float(model["sigma_mgfe_extended"]),
            )
            for mg_value, density_value in zip(mg_grid, conc, strict=True):
                rows.append({"slice_label": label, "mgfe": mg_value, "density": density_value, "component_label": "concentrated"})
            for mg_value, density_value in zip(mg_grid, ext, strict=True):
                rows.append({"slice_label": label, "mgfe": mg_value, "density": density_value, "component_label": "extended"})
    return pd.DataFrame(rows)
