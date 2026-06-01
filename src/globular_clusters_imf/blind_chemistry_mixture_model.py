from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

from .blind_mixture_model import (
    BlindBkComparisonSummary,
    BlindRadialModelSpec as BaseBlindRadialModelSpec,
    blind_model_initial_parameter_vectors,
    blind_model_parameter_bounds,
    blind_full_log_likelihood_from_model as blind_full_log_likelihood_without_chemistry,
    build_blind_vs_bk_group_table,
    build_blind_vs_bk_summary,
    build_fixed_abs_longitude_selection_payload,
    build_imf_grid_table,
    build_radial_grid_table,
    compute_total_initial_stellar_mass,
    unpack_blind_model as unpack_blind_model_without_chemistry,
)
from .joint_model import JointLikelihoodContext

TINY = 1.0e-300
DEFAULT_OUTPUT_PREFIX = "blind_mg_al_mixture"
COMPONENT_LABELS = ("concentrated", "extended")


@dataclass(frozen=True)
class BlindChemistryModelSpec:
    model_class: str
    imf_family: str


@dataclass(frozen=True)
class ChemistryObservationContext:
    mgfe: np.ndarray
    mgfe_err: np.ndarray
    alfe: np.ndarray
    alfe_err: np.ndarray
    has_mgfe: np.ndarray
    has_alfe: np.ndarray
    has_any_chemistry: np.ndarray
    has_mgfe_and_alfe: np.ndarray
    both_indices: np.ndarray
    mg_only_indices: np.ndarray
    al_only_indices: np.ndarray


@dataclass
class BlindChemistryModelFitResult:
    model_class: str
    imf_family: str
    success: bool
    log_likelihood: float
    aic: float
    bic: float
    delta_bic: float
    n_parameters: int
    total_initial_count: float
    total_initial_stellar_mass_msun: float
    selection_fraction: float
    raw_survival_fraction: float
    mean_detectability: float
    component_mix_fraction_concentrated: float
    component_mix_fraction_extended: float
    component_initial_count_concentrated: float
    component_initial_count_extended: float
    expected_observed_count_concentrated: float
    expected_observed_count_extended: float
    optimizer_message: str
    imf_parameters_json: str
    radial_parameters_json: str
    chemistry_parameters_json: str


def fit_blind_powerlaw_a_chemistry_models(
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
    chemistry_context = build_chemistry_observation_context(catalog)
    model_specs = [
        BlindChemistryModelSpec(model_class="single_powerlaw_radial", imf_family="schechter"),
        BlindChemistryModelSpec(model_class="two_component_powerlaw_mixture", imf_family="schechter"),
        BlindChemistryModelSpec(model_class="two_component_powerlaw_mixture_split_alpha", imf_family="schechter"),
    ]
    payload_by_class: dict[str, dict[str, object]] = {}
    for spec in model_specs:
        nested_core_starts = None
        if spec.model_class == "two_component_powerlaw_mixture_split_alpha":
            nested_core_starts = nested_split_alpha_core_starts(
                payload_by_class["two_component_powerlaw_mixture"]
            )
        payload = fit_blind_powerlaw_a_chemistry_single_model(
            selection_context=selection_payload["selection_context"],
            chemistry_context=chemistry_context,
            spec=spec,
            core_start_override=nested_core_starts,
        )
        payload_by_class[spec.model_class] = payload
    payloads = [payload_by_class[spec.model_class] for spec in model_specs]

    summary_table = pd.DataFrame([asdict(payload["summary"]) for payload in payloads]).sort_values(
        ["bic", "aic"],
        ascending=[True, True],
    ).reset_index(drop=True)
    best_bic = float(summary_table["bic"].min())
    summary_table["delta_bic"] = summary_table["bic"] - best_bic
    for payload in payloads:
        payload["summary"].delta_bic = float(payload["summary"].bic - best_bic)

    best_payload = min(payloads, key=lambda item: item["summary"].bic)
    posterior_table = build_posterior_probability_table(best_payload, catalog)
    bk_comparison = build_blind_vs_bk_summary(posterior_table)
    bk_group_table = build_blind_vs_bk_group_table(posterior_table)
    imf_grid_table = build_imf_grid_table(best_payload, selection_payload["selection_context"])
    radial_grid_table = build_radial_grid_table(best_payload, selection_payload["selection_context"])
    summary_json = {
        "selection_payload": selection_payload["summary_payload"],
        "chemistry_summary": {
            "n_clusters": int(len(catalog)),
            "n_with_any_chemistry": int(chemistry_context.has_any_chemistry.sum()),
            "n_with_mgfe": int(chemistry_context.has_mgfe.sum()),
            "n_with_alfe": int(chemistry_context.has_alfe.sum()),
            "n_with_mgfe_and_alfe": int(chemistry_context.has_mgfe_and_alfe.sum()),
        },
        "all_models_ranked": summary_table.to_dict(orient="records"),
        "best_model": asdict(best_payload["summary"]),
        "best_model_bk_comparison": asdict(bk_comparison),
    }

    outputs_tables = output_root / "outputs" / "tables"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    summary_table.to_csv(outputs_tables / f"{output_prefix}_model_summary.csv", index=False)
    posterior_table.to_csv(outputs_tables / f"{output_prefix}_best_model_posterior_probabilities.csv", index=False)
    bk_group_table.to_csv(outputs_tables / f"{output_prefix}_best_model_vs_bk_groups.csv", index=False)
    imf_grid_table.to_csv(outputs_tables / f"{output_prefix}_best_model_imf_grid.csv", index=False)
    radial_grid_table.to_csv(outputs_tables / f"{output_prefix}_best_model_radial_grid.csv", index=False)
    (outputs_tables / f"{output_prefix}_model_summary.json").write_text(json.dumps(summary_json, indent=2))

    return {
        "selection_payload": selection_payload,
        "chemistry_context": chemistry_context,
        "summary_table": summary_table,
        "best_payload": best_payload,
        "posterior_table": posterior_table,
        "bk_comparison": bk_comparison,
        "bk_group_table": bk_group_table,
        "imf_grid_table": imf_grid_table,
        "radial_grid_table": radial_grid_table,
        "all_payloads": payloads,
    }


def build_chemistry_observation_context(catalog: pd.DataFrame) -> ChemistryObservationContext:
    mgfe = pd.to_numeric(catalog.get("mgfe_combined"), errors="coerce").to_numpy(dtype=float)
    mgfe_err = pd.to_numeric(catalog.get("mgfe_combined_err"), errors="coerce").to_numpy(dtype=float)
    alfe = pd.to_numeric(catalog.get("alfe_combined"), errors="coerce").to_numpy(dtype=float)
    alfe_err = pd.to_numeric(catalog.get("alfe_combined_err"), errors="coerce").to_numpy(dtype=float)

    has_mgfe = np.isfinite(mgfe) & np.isfinite(mgfe_err) & (mgfe_err > 0.0)
    has_alfe = np.isfinite(alfe) & np.isfinite(alfe_err) & (alfe_err > 0.0)
    has_any = has_mgfe | has_alfe
    has_both = has_mgfe & has_alfe
    both_indices = np.flatnonzero(has_both)
    mg_only_indices = np.flatnonzero(has_mgfe & ~has_alfe)
    al_only_indices = np.flatnonzero(has_alfe & ~has_mgfe)
    return ChemistryObservationContext(
        mgfe=mgfe,
        mgfe_err=mgfe_err,
        alfe=alfe,
        alfe_err=alfe_err,
        has_mgfe=has_mgfe,
        has_alfe=has_alfe,
        has_any_chemistry=has_any,
        has_mgfe_and_alfe=has_both,
        both_indices=both_indices,
        mg_only_indices=mg_only_indices,
        al_only_indices=al_only_indices,
    )


def fit_blind_powerlaw_a_chemistry_single_model(
    selection_context: JointLikelihoodContext,
    chemistry_context: ChemistryObservationContext,
    spec: BlindChemistryModelSpec,
    core_start_override: list[np.ndarray] | None = None,
) -> dict[str, object]:
    base_spec = BaseBlindRadialModelSpec(model_class=spec.model_class, imf_family=spec.imf_family)
    core_starts = (
        [np.asarray(start, dtype=float) for start in core_start_override]
        if core_start_override is not None
        else blind_model_initial_parameter_vectors(base_spec)
    )
    core_bounds = blind_model_parameter_bounds(base_spec)
    chemistry_starts = chemistry_initial_parameter_vectors(chemistry_context, spec)
    chemistry_bounds = chemistry_parameter_bounds(spec)

    best_result = None
    best_value = np.inf
    for core_start in core_starts:
        for chemistry_start in chemistry_starts:
            start = np.concatenate([np.asarray(core_start, dtype=float), np.asarray(chemistry_start, dtype=float)])
            result = optimize.minimize(
                lambda params: blind_chemistry_negative_profile_log_likelihood(
                    params,
                    selection_context=selection_context,
                    chemistry_context=chemistry_context,
                    spec=spec,
                ),
                x0=start,
                method="L-BFGS-B",
                bounds=list(core_bounds) + list(chemistry_bounds),
            )
            if result.fun < best_value:
                best_value = float(result.fun)
                best_result = result

    if best_result is None:
        raise RuntimeError(f"Blind chemistry model optimization failed to start for {spec}")

    model = unpack_blind_chemistry_model(
        best_result.x,
        selection_context=selection_context,
        chemistry_context=chemistry_context,
        spec=spec,
    )
    log_likelihood = blind_chemistry_full_log_likelihood_from_model(model, selection_context)
    n_parameters = len(best_result.x)
    n_clusters = len(selection_context.log_mass_data)
    total_initial_stellar_mass = compute_total_initial_stellar_mass(model, selection_context)
    summary = BlindChemistryModelFitResult(
        model_class=spec.model_class,
        imf_family=spec.imf_family,
        success=bool(best_result.success),
        log_likelihood=float(log_likelihood),
        aic=float(2 * n_parameters - 2 * log_likelihood),
        bic=float(np.log(n_clusters) * n_parameters - 2 * log_likelihood),
        delta_bic=np.nan,
        n_parameters=n_parameters,
        total_initial_count=float(model["total_initial_count"]),
        total_initial_stellar_mass_msun=float(total_initial_stellar_mass),
        selection_fraction=float(model["selection_fraction_total"]),
        raw_survival_fraction=float(model["raw_survival_fraction_total"]),
        mean_detectability=float(
            model["selection_fraction_total"] / max(model["raw_survival_fraction_total"], 1.0e-12)
        ),
        component_mix_fraction_concentrated=float(model["mix_fraction_concentrated"]),
        component_mix_fraction_extended=float(1.0 - model["mix_fraction_concentrated"]),
        component_initial_count_concentrated=float(model["component_initial_count"]["concentrated"]),
        component_initial_count_extended=float(model["component_initial_count"]["extended"]),
        expected_observed_count_concentrated=float(model["expected_observed_count"]["concentrated"]),
        expected_observed_count_extended=float(model["expected_observed_count"]["extended"]),
        optimizer_message=str(best_result.message),
        imf_parameters_json=json.dumps(model["imf_parameters"]),
        radial_parameters_json=json.dumps(model["radial_parameters"]),
        chemistry_parameters_json=json.dumps(model["chemistry_parameters"]),
    )
    return {
        "spec": spec,
        "summary": summary,
        "model": model,
        "raw_parameters": np.asarray(best_result.x, dtype=float),
        "bounds": list(core_bounds) + list(chemistry_bounds),
    }


def nested_split_alpha_core_starts(shared_payload: dict[str, object]) -> list[np.ndarray]:
    shared_spec = BaseBlindRadialModelSpec(model_class="two_component_powerlaw_mixture", imf_family="schechter")
    n_core = len(blind_model_parameter_bounds(shared_spec))
    shared_core_params = np.asarray(shared_payload["raw_parameters"], dtype=float)[:n_core]
    shared_alpha = float(shared_core_params[0])
    shared_log_mc = float(shared_core_params[1])
    radial_tail = shared_core_params[2:]
    starts = []
    for delta in (0.0, 0.15, 0.3, -0.15):
        alpha_concentrated = shared_alpha + max(delta, 0.0)
        alpha_extended = shared_alpha - max(delta, 0.0)
        if delta < 0.0:
            alpha_concentrated = shared_alpha + delta
            alpha_extended = shared_alpha - delta
        starts.append(
            np.concatenate(
                [
                    np.array([alpha_concentrated, alpha_extended, shared_log_mc], dtype=float),
                    radial_tail,
                ]
            )
        )
    return deduplicate_starts(starts)


def chemistry_parameter_bounds(spec: BlindChemistryModelSpec) -> list[tuple[float, float]]:
    component_bounds = [
        (-0.6, 1.5),   # mu_al
        (-0.4, 0.8),   # mu_mg
        (-4.0, 0.0),   # log sigma_al
        (-4.0, 0.0),   # log sigma_mg
    ]
    shared_slope_bound = [(-4.0, 1.0)]
    if spec.model_class == "single_powerlaw_radial":
        return shared_slope_bound + component_bounds
    return shared_slope_bound + component_bounds + component_bounds


def chemistry_initial_parameter_vectors(
    chemistry_context: ChemistryObservationContext,
    spec: BlindChemistryModelSpec,
) -> list[np.ndarray]:
    al = chemistry_context.alfe[chemistry_context.has_alfe]
    mg = chemistry_context.mgfe[chemistry_context.has_mgfe]
    both_al = chemistry_context.alfe[chemistry_context.has_mgfe_and_alfe]
    both_mg = chemistry_context.mgfe[chemistry_context.has_mgfe_and_alfe]
    if len(both_al) < 3:
        raise ValueError("Chemistry-aware blind mixture requires at least three clusters with both Mg and Al.")

    slope_fit = float(np.polyfit(both_al, both_mg, deg=1)[0]) if len(both_al) >= 2 else -0.75
    slope_starts = sorted({-1.0, -0.6, float(np.clip(slope_fit, -3.5, 0.5))})
    al_med = float(np.nanmedian(al))
    mg_med = float(np.nanmedian(mg))
    al_q25, al_q75 = np.nanpercentile(al, [25, 75])
    mg_q25, mg_q75 = np.nanpercentile(mg, [25, 75])
    sigma_al = float(np.clip(np.nanstd(al, ddof=1), 0.08, 0.35))
    sigma_mg = float(np.clip(np.nanstd(mg, ddof=1), 0.05, 0.25))
    log_sigma_al = float(np.log(sigma_al))
    log_sigma_mg = float(np.log(sigma_mg))
    compact_log_sigma_al = float(np.log(np.clip(0.7 * sigma_al, 0.06, 0.25)))
    compact_log_sigma_mg = float(np.log(np.clip(0.7 * sigma_mg, 0.04, 0.18)))

    starts: list[np.ndarray] = []
    if spec.model_class == "single_powerlaw_radial":
        for slope in slope_starts:
            starts.append(np.array([slope, al_med, mg_med, log_sigma_al, log_sigma_mg], dtype=float))
            starts.append(
                np.array(
                    [slope, 0.5 * (al_q25 + al_q75), 0.5 * (mg_q25 + mg_q75), compact_log_sigma_al, compact_log_sigma_mg],
                    dtype=float,
                )
            )
        return deduplicate_starts(starts)

    for slope in slope_starts:
        starts.append(
            np.array(
                [
                    slope,
                    float(al_q75),
                    float(mg_q75),
                    compact_log_sigma_al,
                    compact_log_sigma_mg,
                    float(al_q25),
                    float(mg_q25),
                    compact_log_sigma_al,
                    compact_log_sigma_mg,
                ],
                dtype=float,
            )
        )
        starts.append(
            np.array(
                [
                    slope,
                    float(al_med + 0.15),
                    float(mg_med + 0.10),
                    log_sigma_al,
                    log_sigma_mg,
                    float(al_med - 0.10),
                    float(mg_med - 0.08),
                    log_sigma_al,
                    log_sigma_mg,
                ],
                dtype=float,
            )
        )
    return deduplicate_starts(starts)


def deduplicate_starts(starts: list[np.ndarray]) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for candidate in starts:
        if not any(np.allclose(candidate, existing) for existing in unique):
            unique.append(candidate)
    return unique


def blind_chemistry_negative_profile_log_likelihood(
    params: np.ndarray,
    selection_context: JointLikelihoodContext,
    chemistry_context: ChemistryObservationContext,
    spec: BlindChemistryModelSpec,
) -> float:
    model = unpack_blind_chemistry_model(
        params,
        selection_context=selection_context,
        chemistry_context=chemistry_context,
        spec=spec,
    )
    if np.any(model["mixture_observed_density_data"] <= 0.0):
        return 1.0e30
    if model["selection_fraction_total"] <= 0.0:
        return 1.0e30
    profile_log_like = (
        np.sum(np.log(model["mixture_observed_density_data"]))
        - len(selection_context.log_mass_data) * np.log(model["selection_fraction_total"])
    )
    return float(-profile_log_like)


def unpack_blind_chemistry_model(
    params: np.ndarray,
    selection_context: JointLikelihoodContext,
    chemistry_context: ChemistryObservationContext,
    spec: BlindChemistryModelSpec,
) -> dict[str, object]:
    base_spec = BaseBlindRadialModelSpec(model_class=spec.model_class, imf_family=spec.imf_family)
    n_core = len(blind_model_parameter_bounds(base_spec))
    core_params = np.asarray(params[:n_core], dtype=float)
    chemistry_params = np.asarray(params[n_core:], dtype=float)
    base_model = unpack_blind_model_without_chemistry(core_params, context=selection_context, spec=base_spec)

    chemistry_shared_slope = float(chemistry_params[0])
    concentrated_chemistry = unpack_component_chemistry_parameters(chemistry_params[1:5])
    if spec.model_class == "single_powerlaw_radial":
        extended_chemistry = {
            "mu_alfe": np.nan,
            "mu_mgfe": np.nan,
            "sigma_alfe": np.nan,
            "sigma_mgfe": np.nan,
        }
    else:
        extended_chemistry = unpack_component_chemistry_parameters(chemistry_params[5:9])

    chemistry_density_concentrated = component_chemistry_density(
        chemistry_context=chemistry_context,
        shared_slope=chemistry_shared_slope,
        component_parameters=concentrated_chemistry,
    )
    if spec.model_class == "single_powerlaw_radial":
        chemistry_density_extended = np.ones_like(chemistry_density_concentrated)
    else:
        chemistry_density_extended = component_chemistry_density(
            chemistry_context=chemistry_context,
            shared_slope=chemistry_shared_slope,
            component_parameters=extended_chemistry,
        )

    concentrated_weight = (
        base_model["mix_fraction_concentrated"]
        * base_model["component_imf_density_data"]["concentrated"]
        * base_model["component_radial_density_data"]["concentrated"]
        * chemistry_density_concentrated
    )
    extended_weight = (
        (1.0 - base_model["mix_fraction_concentrated"])
        * base_model["component_imf_density_data"]["extended"]
        * base_model["component_radial_density_data"]["extended"]
        * chemistry_density_extended
    )
    mixture_observed_density = np.clip(concentrated_weight + extended_weight, TINY, None)

    model = dict(base_model)
    model["component_chemistry_density_data"] = {
        "concentrated": chemistry_density_concentrated,
        "extended": chemistry_density_extended,
    }
    model["mixture_observed_density_data"] = mixture_observed_density
    model["chemistry_parameters"] = {
        "shared_mg_vs_al_slope": chemistry_shared_slope,
        "concentrated": concentrated_chemistry,
        "extended": extended_chemistry,
    }
    return model


def unpack_component_chemistry_parameters(raw_component_params: np.ndarray) -> dict[str, float]:
    mu_alfe = float(raw_component_params[0])
    mu_mgfe = float(raw_component_params[1])
    sigma_alfe = float(np.exp(raw_component_params[2]))
    sigma_mgfe = float(np.exp(raw_component_params[3]))
    return {
        "mu_alfe": mu_alfe,
        "mu_mgfe": mu_mgfe,
        "sigma_alfe": sigma_alfe,
        "sigma_mgfe": sigma_mgfe,
    }


def component_chemistry_density(
    chemistry_context: ChemistryObservationContext,
    shared_slope: float,
    component_parameters: dict[str, float],
) -> np.ndarray:
    density = np.ones_like(chemistry_context.mgfe, dtype=float)
    mu_al = float(component_parameters["mu_alfe"])
    mu_mg = float(component_parameters["mu_mgfe"])
    sigma_al = max(float(component_parameters["sigma_alfe"]), 1.0e-3)
    sigma_mg = max(float(component_parameters["sigma_mgfe"]), 1.0e-3)
    var_al = sigma_al**2
    cov_al_mg = shared_slope * var_al
    var_mg = shared_slope**2 * var_al + sigma_mg**2

    if chemistry_context.both_indices.size > 0:
        for index in chemistry_context.both_indices:
            total_var_al = var_al + chemistry_context.alfe_err[index] ** 2
            total_var_mg = var_mg + chemistry_context.mgfe_err[index] ** 2
            total_cov = cov_al_mg
            density[index] = bivariate_gaussian_pdf(
                x=float(chemistry_context.alfe[index]),
                y=float(chemistry_context.mgfe[index]),
                mean_x=mu_al,
                mean_y=mu_mg,
                var_x=total_var_al,
                var_y=total_var_mg,
                cov_xy=total_cov,
            )

    if chemistry_context.mg_only_indices.size > 0:
        variance = var_mg + chemistry_context.mgfe_err[chemistry_context.mg_only_indices] ** 2
        density[chemistry_context.mg_only_indices] = gaussian_pdf(
            chemistry_context.mgfe[chemistry_context.mg_only_indices],
            mu_mg,
            variance,
        )

    if chemistry_context.al_only_indices.size > 0:
        variance = var_al + chemistry_context.alfe_err[chemistry_context.al_only_indices] ** 2
        density[chemistry_context.al_only_indices] = gaussian_pdf(
            chemistry_context.alfe[chemistry_context.al_only_indices],
            mu_al,
            variance,
        )

    return np.clip(density, TINY, None)


def gaussian_pdf(values: np.ndarray, mean: float, variance: np.ndarray | float) -> np.ndarray:
    variance = np.asarray(variance, dtype=float)
    variance = np.clip(variance, 1.0e-8, None)
    return np.exp(-0.5 * np.square(np.asarray(values, dtype=float) - mean) / variance) / np.sqrt(
        2.0 * np.pi * variance
    )


def bivariate_gaussian_pdf(
    x: float,
    y: float,
    mean_x: float,
    mean_y: float,
    var_x: float,
    var_y: float,
    cov_xy: float,
) -> float:
    determinant = max(var_x * var_y - cov_xy**2, 1.0e-12)
    dx = x - mean_x
    dy = y - mean_y
    quadratic = (var_y * dx**2 + var_x * dy**2 - 2.0 * cov_xy * dx * dy) / determinant
    normalizer = 2.0 * np.pi * np.sqrt(determinant)
    return float(np.exp(-0.5 * quadratic) / max(normalizer, 1.0e-12))


def blind_chemistry_full_log_likelihood_from_model(
    model: dict[str, object],
    selection_context: JointLikelihoodContext,
) -> float:
    base_log_likelihood = blind_full_log_likelihood_without_chemistry(model, context=selection_context)
    chemistry_free_density = np.clip(model["mixture_component_density_data"], TINY, None)
    chemistry_full_density = np.clip(model["mixture_observed_density_data"], TINY, None)
    chemistry_contribution = np.sum(np.log(chemistry_full_density) - np.log(chemistry_free_density))
    return float(base_log_likelihood + chemistry_contribution)


def component_membership_probabilities_with_chemistry(model: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    if model["spec"].model_class == "single_powerlaw_radial":
        concentrated = np.ones_like(model["mixture_observed_density_data"])
        extended = np.zeros_like(concentrated)
        return concentrated, extended

    concentrated_weight = (
        model["mix_fraction_concentrated"]
        * model["component_imf_density_data"]["concentrated"]
        * model["component_radial_density_data"]["concentrated"]
        * model["component_chemistry_density_data"]["concentrated"]
    )
    extended_weight = (
        (1.0 - model["mix_fraction_concentrated"])
        * model["component_imf_density_data"]["extended"]
        * model["component_radial_density_data"]["extended"]
        * model["component_chemistry_density_data"]["extended"]
    )
    denominator = np.clip(concentrated_weight + extended_weight, TINY, None)
    p_concentrated = np.clip(concentrated_weight / denominator, 1.0e-12, 1.0)
    return p_concentrated, 1.0 - p_concentrated


def build_posterior_probability_table(best_payload: dict[str, object], catalog: pd.DataFrame) -> pd.DataFrame:
    p_concentrated, p_extended = component_membership_probabilities_with_chemistry(best_payload["model"])
    output = catalog.copy()
    output["p_concentrated"] = p_concentrated
    output["p_extended"] = p_extended
    output["blind_component_label"] = np.where(output["p_concentrated"] >= 0.5, "concentrated", "extended")
    return output
