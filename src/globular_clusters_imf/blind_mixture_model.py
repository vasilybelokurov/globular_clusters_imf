from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .detectability_longitude_model import (
    build_observable_prediction_context_with_abs_longitude,
    compute_effective_completeness_grid_with_abs_longitude,
    evaluate_completeness_bin_grid_with_abs_longitude,
)
from .joint_model import (
    JointLikelihoodContext,
    JointModelSpec,
    build_fixed_survival_grid,
    calibrate_fixed_selection_offset_dex,
    evaluate_imf_family,
    imf_parameter_count,
    initial_parameter_vectors,
    integrate_survival_fraction,
    interpolate_density,
    parameter_bounds,
)
from .model import survival_mass_cut_msun

TINY = 1.0e-300
DEFAULT_DETECTABILITY_SUMMARY = (
    "outputs/tables/joint_fixed_survival_detectability_abs_longitude_em_summary.json"
)
DEFAULT_OUTPUT_PREFIX = "blind_powerlaw_a_mixture"
DEFAULT_SPLIT_ALPHA_OUTPUT_PREFIX = "blind_powerlaw_a_split_alpha_mixture"
MIX_FRACTION_FLOOR = 0.05
COMPONENT_LABELS = ("concentrated", "extended")


@dataclass(frozen=True)
class BlindRadialModelSpec:
    model_class: str
    imf_family: str


@dataclass
class BlindRadialModelFitResult:
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


@dataclass
class BlindBkComparisonSummary:
    n_clusters: int
    n_with_origin_flag: int
    auc_in_situ_vs_p_concentrated: float
    hard_assignment_accuracy: float
    mean_p_concentrated_in_situ: float
    mean_p_concentrated_accreted: float


def fit_blind_powerlaw_a_models(
    catalog: pd.DataFrame,
    project_root: Path,
    output_root: Path,
    detectability_summary_path: Path | None = None,
    imf_families: tuple[str, ...] = ("lognormal", "powerlaw", "schechter"),
    include_split_alpha_schechter: bool = False,
    output_prefix: str = DEFAULT_OUTPUT_PREFIX,
) -> dict[str, object]:
    selection_payload = build_fixed_abs_longitude_selection_payload(
        catalog=catalog,
        project_root=project_root,
        detectability_summary_path=detectability_summary_path,
    )
    context = selection_payload["selection_context"]

    model_specs = build_blind_model_specs(
        imf_families=imf_families,
        include_split_alpha_schechter=include_split_alpha_schechter,
    )
    payloads = [fit_blind_powerlaw_a_single_model(context=context, spec=spec) for spec in model_specs]

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
    imf_grid_table = build_imf_grid_table(best_payload, context)
    radial_grid_table = build_radial_grid_table(best_payload, context)
    summary_json = {
        "selection_payload": selection_payload["summary_payload"],
        "all_models_ranked": summary_table.to_dict(orient="records"),
        "best_model": asdict(best_payload["summary"]),
        "best_model_bk_comparison": asdict(bk_comparison),
    }

    outputs_tables = output_root / "outputs" / "tables"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    summary_table.to_csv(outputs_tables / f"{output_prefix}_model_summary.csv", index=False)
    posterior_table.to_csv(
        outputs_tables / f"{output_prefix}_best_model_posterior_probabilities.csv",
        index=False,
    )
    bk_group_table.to_csv(outputs_tables / f"{output_prefix}_best_model_vs_bk_groups.csv", index=False)
    imf_grid_table.to_csv(outputs_tables / f"{output_prefix}_best_model_imf_grid.csv", index=False)
    radial_grid_table.to_csv(outputs_tables / f"{output_prefix}_best_model_radial_grid.csv", index=False)
    (outputs_tables / f"{output_prefix}_model_summary.json").write_text(
        json.dumps(summary_json, indent=2)
    )

    return {
        "selection_payload": selection_payload,
        "summary_table": summary_table,
        "best_payload": best_payload,
        "posterior_table": posterior_table,
        "bk_comparison": bk_comparison,
        "bk_group_table": bk_group_table,
        "imf_grid_table": imf_grid_table,
        "radial_grid_table": radial_grid_table,
        "all_payloads": payloads,
    }


def build_fixed_abs_longitude_selection_payload(
    catalog: pd.DataFrame,
    project_root: Path,
    detectability_summary_path: Path | None = None,
) -> dict[str, object]:
    summary_path = (
        detectability_summary_path
        if detectability_summary_path is not None
        else project_root / DEFAULT_DETECTABILITY_SUMMARY
    )
    summary_payload = json.loads(summary_path.read_text())
    detectability = summary_payload["best_model_detectability_summary"]

    working = catalog.copy()
    if "log_survival_mass_cut_msun" not in working.columns:
        if {"r_apo_kpc", "eccentricity"}.difference(working.columns):
            raise ValueError(
                "Blind mixture analysis requires either log_survival_mass_cut_msun or "
                "the orbit columns r_apo_kpc and eccentricity."
            )
        working["survival_mass_cut_msun"] = working.apply(
            lambda row: survival_mass_cut_msun(
                r_apo_kpc=float(row["r_apo_kpc"]),
                eccentricity=float(row["eccentricity"]),
            ),
            axis=1,
        )
        working["log_survival_mass_cut_msun"] = np.log10(working["survival_mass_cut_msun"])
    selection_offset_dex = calibrate_fixed_selection_offset_dex(working)
    survival_grid = build_fixed_survival_grid(
        working,
        selection_offset_dex=selection_offset_dex,
    )
    base_context = JointLikelihoodContext.from_catalog_and_survival_grid(working, survival_grid)
    observable_context = build_observable_prediction_context_with_abs_longitude(
        catalog=working,
        base_context=base_context,
        n_present_mass_bins=6,
        n_distance_bins=6,
        n_latitude_bins=6,
        n_longitude_bins=int(detectability.get("n_longitude_bins", 6)),
        n_geometry_samples=5000,
        sun_galactocentric_radius_kpc=float(detectability.get("sun_galactocentric_radius_kpc", 8.2)),
    )
    raw_parameters = raw_detectability_parameters_from_summary(detectability["final_completeness_parameters"])
    completeness_bin_grid = evaluate_completeness_bin_grid_with_abs_longitude(
        raw_parameters,
        observable_context,
    )
    effective_completeness_grid = compute_effective_completeness_grid_with_abs_longitude(
        observable_context=observable_context,
        completeness_bin_grid=completeness_bin_grid,
    )
    selection_context = base_context.with_selection_probability_grid(
        np.clip(
            base_context.survival_probability_grid * effective_completeness_grid,
            1.0e-12,
            1.0,
        )
    )
    return {
        "summary_payload": detectability,
        "selection_offset_dex": selection_offset_dex,
        "base_context": base_context,
        "selection_context": selection_context,
        "observable_context": observable_context,
        "raw_detectability_parameters": raw_parameters,
        "completeness_bin_grid": completeness_bin_grid,
        "effective_completeness_grid": effective_completeness_grid,
    }


def raw_detectability_parameters_from_summary(parameters: dict[str, float]) -> np.ndarray:
    return np.array(
        [
            float(parameters["intercept"]),
            np.log(float(parameters["mass_slope"])),
            np.log(float(parameters["distance_slope"])),
            np.log(float(parameters["latitude_slope"])),
            np.log(float(parameters["longitude_slope"])),
        ],
        dtype=float,
    )


def build_blind_model_specs(
    imf_families: tuple[str, ...],
    include_split_alpha_schechter: bool = False,
) -> list[BlindRadialModelSpec]:
    specs: list[BlindRadialModelSpec] = []
    for imf_family in imf_families:
        specs.append(BlindRadialModelSpec(model_class="single_powerlaw_radial", imf_family=imf_family))
        specs.append(BlindRadialModelSpec(model_class="two_component_powerlaw_mixture", imf_family=imf_family))
        if include_split_alpha_schechter and imf_family == "schechter":
            specs.append(
                BlindRadialModelSpec(
                    model_class="two_component_powerlaw_mixture_split_alpha",
                    imf_family="schechter",
                )
            )
    return specs


def fit_blind_powerlaw_a_single_model(
    context: JointLikelihoodContext,
    spec: BlindRadialModelSpec,
) -> dict[str, object]:
    starts = blind_model_initial_parameter_vectors(spec)
    bounds = blind_model_parameter_bounds(spec)
    best_result = None
    best_value = np.inf
    for start in starts:
        result = optimize.minimize(
            lambda params: blind_negative_profile_log_likelihood(params, context=context, spec=spec),
            x0=np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if result.fun < best_value:
            best_value = float(result.fun)
            best_result = result

    if best_result is None:
        raise RuntimeError(f"Blind model optimization failed to start for {spec}")

    model = unpack_blind_model(best_result.x, context=context, spec=spec)
    log_likelihood = blind_full_log_likelihood_from_model(model, context=context)
    n_parameters = len(best_result.x)
    n_clusters = len(context.log_mass_data)
    total_initial_stellar_mass = compute_total_initial_stellar_mass(model, context)
    summary = BlindRadialModelFitResult(
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
    )
    return {
        "spec": spec,
        "summary": summary,
        "model": model,
        "raw_parameters": np.asarray(best_result.x, dtype=float),
        "bounds": bounds,
    }


def blind_model_parameter_bounds(spec: BlindRadialModelSpec) -> list[tuple[float, float]]:
    n_imf = blind_model_imf_parameter_count(spec)
    if spec.model_class == "two_component_powerlaw_mixture_split_alpha":
        schechter_bounds = parameter_bounds(JointModelSpec("schechter", "step5"))[:2]
        alpha_bounds = schechter_bounds[0]
        log_mc_bounds = schechter_bounds[1]
        return [alpha_bounds, alpha_bounds, log_mc_bounds, (-6.0, 3.0), (-5.0, 1.5), (-6.0, 6.0)]

    imf_bounds = parameter_bounds(JointModelSpec(spec.imf_family, "step5"))[:n_imf]
    if spec.model_class == "single_powerlaw_radial":
        return list(imf_bounds) + [(-6.0, 3.0)]
    if spec.model_class == "two_component_powerlaw_mixture":
        return list(imf_bounds) + [(-6.0, 3.0), (-5.0, 1.5), (-6.0, 6.0)]
    raise ValueError(f"Unknown blind radial model class: {spec.model_class}")


def blind_model_initial_parameter_vectors(spec: BlindRadialModelSpec) -> list[np.ndarray]:
    starts: list[np.ndarray] = []
    if spec.model_class == "two_component_powerlaw_mixture_split_alpha":
        alpha_starts = unique_schechter_alpha_starts()
        log_mc_starts = unique_schechter_log_mc_starts()
        radial_starts = [
            np.array([-1.0, -0.7, 0.0]),
            np.array([-1.5, -1.2, -1.0]),
            np.array([-0.5, -1.0, 1.0]),
            np.array([0.0, -0.7, 0.0]),
        ]
        for alpha_concentrated in alpha_starts:
            for alpha_extended in alpha_starts:
                for log_mc in log_mc_starts:
                    for radial_start in radial_starts:
                        starts.append(
                            np.concatenate(
                                [
                                    np.array([alpha_concentrated, alpha_extended, log_mc], dtype=float),
                                    radial_start,
                                ]
                            )
                        )
        return starts

    imf_starts = unique_imf_starts(spec.imf_family)
    if spec.model_class == "single_powerlaw_radial":
        radial_starts = [np.array([-1.5]), np.array([-0.8]), np.array([0.0])]
    elif spec.model_class == "two_component_powerlaw_mixture":
        radial_starts = [
            np.array([-1.0, -0.7, 0.0]),
            np.array([-1.5, -1.2, -1.0]),
            np.array([-0.5, -1.0, 1.0]),
            np.array([0.0, -0.7, 0.0]),
        ]
    else:
        raise ValueError(f"Unknown blind radial model class: {spec.model_class}")

    for imf_start in imf_starts:
        for radial_start in radial_starts:
            starts.append(np.concatenate([imf_start, radial_start]))
    return starts


def blind_model_imf_parameter_count(spec: BlindRadialModelSpec) -> int:
    if spec.model_class == "two_component_powerlaw_mixture_split_alpha":
        return 3
    return imf_parameter_count(spec.imf_family)


def unique_imf_starts(imf_family: str) -> list[np.ndarray]:
    helper_spec = JointModelSpec(imf_family=imf_family, radial_model="step5")
    n_imf = imf_parameter_count(imf_family)
    starts: list[np.ndarray] = []
    for start in initial_parameter_vectors(helper_spec):
        candidate = np.asarray(start[:n_imf], dtype=float)
        if not any(np.allclose(candidate, existing) for existing in starts):
            starts.append(candidate)
    return starts


def unique_schechter_alpha_starts() -> list[float]:
    values: list[float] = []
    for start in unique_imf_starts("schechter"):
        alpha = float(start[0])
        if not any(np.isclose(alpha, existing) for existing in values):
            values.append(alpha)
    return values


def unique_schechter_log_mc_starts() -> list[float]:
    values: list[float] = []
    for start in unique_imf_starts("schechter"):
        log_mc = float(start[1])
        if not any(np.isclose(log_mc, existing) for existing in values):
            values.append(log_mc)
    return values


def blind_negative_profile_log_likelihood(
    params: np.ndarray,
    context: JointLikelihoodContext,
    spec: BlindRadialModelSpec,
) -> float:
    model = unpack_blind_model(params, context=context, spec=spec)
    if np.any(model["mixture_component_density_data"] <= 0.0):
        return 1.0e30
    if model["selection_fraction_total"] <= 0.0:
        return 1.0e30
    profile_log_like = (
        np.sum(np.log(model["mixture_component_density_data"]))
        - len(context.log_mass_data) * np.log(model["selection_fraction_total"])
    )
    return float(-profile_log_like)


def unpack_blind_model(
    params: np.ndarray,
    context: JointLikelihoodContext,
    spec: BlindRadialModelSpec,
) -> dict[str, object]:
    n_imf = blind_model_imf_parameter_count(spec)
    imf_params = np.asarray(params[:n_imf], dtype=float)
    radial_params = np.asarray(params[n_imf:], dtype=float)

    if spec.model_class == "two_component_powerlaw_mixture_split_alpha":
        alpha_concentrated = float(imf_params[0])
        alpha_extended = float(imf_params[1])
        shared_log_mc = float(imf_params[2])
        concentrated_imf_grid, concentrated_imf_data, concentrated_imf_parameters = evaluate_imf_family(
            "schechter",
            np.array([alpha_concentrated, shared_log_mc], dtype=float),
            context.log_mass_grid,
            context.log_mass_data,
        )
        extended_imf_grid, extended_imf_data, extended_imf_parameters = evaluate_imf_family(
            "schechter",
            np.array([alpha_extended, shared_log_mc], dtype=float),
            context.log_mass_grid,
            context.log_mass_data,
        )
        beta_mid = float(radial_params[0])
        slope_gap = float(np.exp(radial_params[1]))
        mix_fraction = bounded_mix_fraction(float(radial_params[2]))
        beta_concentrated = beta_mid - 0.5 * slope_gap
        beta_extended = beta_mid + 0.5 * slope_gap
        concentrated_grid = radial_powerlaw_density_grid(context.log_a_grid, beta_concentrated, pivot=context.log_a_mean)
        extended_grid = radial_powerlaw_density_grid(context.log_a_grid, beta_extended, pivot=context.log_a_mean)
        concentrated_data = interpolate_density(context.log_a_grid, concentrated_grid, context.log_a_data)
        extended_data = interpolate_density(context.log_a_grid, extended_grid, context.log_a_data)
        total_imf_grid = np.clip(
            mix_fraction * concentrated_imf_grid + (1.0 - mix_fraction) * extended_imf_grid,
            TINY,
            None,
        )
        total_imf_data = np.clip(
            mix_fraction * concentrated_imf_data + (1.0 - mix_fraction) * extended_imf_data,
            TINY,
            None,
        )
        mixture_grid = np.clip(mix_fraction * concentrated_grid + (1.0 - mix_fraction) * extended_grid, TINY, None)
        selection_fraction_concentrated = integrate_survival_fraction(
            concentrated_imf_grid,
            concentrated_grid,
            context.log_mass_grid,
            context.log_a_grid,
            context.selection_probability_grid,
        )
        selection_fraction_extended = integrate_survival_fraction(
            extended_imf_grid,
            extended_grid,
            context.log_mass_grid,
            context.log_a_grid,
            context.selection_probability_grid,
        )
        raw_survival_fraction_concentrated = integrate_survival_fraction(
            concentrated_imf_grid,
            concentrated_grid,
            context.log_mass_grid,
            context.log_a_grid,
            context.survival_probability_grid,
        )
        raw_survival_fraction_extended = integrate_survival_fraction(
            extended_imf_grid,
            extended_grid,
            context.log_mass_grid,
            context.log_a_grid,
            context.survival_probability_grid,
        )
        selection_fraction_total = (
            mix_fraction * selection_fraction_concentrated
            + (1.0 - mix_fraction) * selection_fraction_extended
        )
        raw_survival_fraction_total = (
            mix_fraction * raw_survival_fraction_concentrated
            + (1.0 - mix_fraction) * raw_survival_fraction_extended
        )
        total_initial_count = len(context.log_mass_data) / max(selection_fraction_total, 1.0e-12)
        component_initial_count = {
            "concentrated": float(total_initial_count * mix_fraction),
            "extended": float(total_initial_count * (1.0 - mix_fraction)),
        }
        component_expected_observed_count = {
            "concentrated": float(component_initial_count["concentrated"] * selection_fraction_concentrated),
            "extended": float(component_initial_count["extended"] * selection_fraction_extended),
        }
        mixture_component_density_data = np.clip(
            mix_fraction * concentrated_imf_data * concentrated_data
            + (1.0 - mix_fraction) * extended_imf_data * extended_data,
            TINY,
            None,
        )
        return {
            "spec": spec,
            "imf_density_grid": total_imf_grid,
            "imf_density_data": total_imf_data,
            "component_imf_density_grid": {
                "concentrated": concentrated_imf_grid,
                "extended": extended_imf_grid,
            },
            "component_imf_density_data": {
                "concentrated": concentrated_imf_data,
                "extended": extended_imf_data,
            },
            "mixture_component_density_data": mixture_component_density_data,
            "mixture_radial_density_grid": mixture_grid,
            "mixture_radial_density_data": np.clip(
                mix_fraction * concentrated_data + (1.0 - mix_fraction) * extended_data,
                TINY,
                None,
            ),
            "component_radial_density_grid": {
                "concentrated": concentrated_grid,
                "extended": extended_grid,
            },
            "component_radial_density_data": {
                "concentrated": concentrated_data,
                "extended": extended_data,
            },
            "selection_fraction_total": float(selection_fraction_total),
            "raw_survival_fraction_total": float(raw_survival_fraction_total),
            "component_selection_fraction": {
                "concentrated": float(selection_fraction_concentrated),
                "extended": float(selection_fraction_extended),
            },
            "component_raw_survival_fraction": {
                "concentrated": float(raw_survival_fraction_concentrated),
                "extended": float(raw_survival_fraction_extended),
            },
            "total_initial_count": float(total_initial_count),
            "component_initial_count": component_initial_count,
            "expected_observed_count": component_expected_observed_count,
            "mix_fraction_concentrated": float(mix_fraction),
            "imf_parameters": {
                "shared_log10_m_c_msun": float(shared_log_mc),
                "concentrated": concentrated_imf_parameters,
                "extended": extended_imf_parameters,
            },
            "radial_parameters": {
                "radial_model": "two_component_powerlaw_mixture_split_alpha",
                "beta_concentrated_logdensity_per_log10a": float(beta_concentrated),
                "beta_extended_logdensity_per_log10a": float(beta_extended),
                "gamma_concentrated_linear_a_density": float(1.0 - beta_concentrated),
                "gamma_extended_linear_a_density": float(1.0 - beta_extended),
                "mix_fraction_concentrated": float(mix_fraction),
                "pivot_log10_a_kpc": float(context.log_a_mean),
            },
        }

    imf_density_grid, imf_density_data, imf_parameters = evaluate_imf_family(
        spec.imf_family,
        imf_params,
        context.log_mass_grid,
        context.log_mass_data,
    )

    if spec.model_class == "single_powerlaw_radial":
        beta = float(radial_params[0])
        component_density_grid = radial_powerlaw_density_grid(context.log_a_grid, beta, pivot=context.log_a_mean)
        component_density_data = interpolate_density(context.log_a_grid, component_density_grid, context.log_a_data)
        selection_fraction = integrate_survival_fraction(
            imf_density_grid,
            component_density_grid,
            context.log_mass_grid,
            context.log_a_grid,
            context.selection_probability_grid,
        )
        raw_survival_fraction = integrate_survival_fraction(
            imf_density_grid,
            component_density_grid,
            context.log_mass_grid,
            context.log_a_grid,
            context.survival_probability_grid,
        )
        total_initial_count = len(context.log_mass_data) / max(selection_fraction, 1.0e-12)
        return {
            "spec": spec,
            "imf_density_grid": imf_density_grid,
            "imf_density_data": imf_density_data,
            "component_imf_density_grid": {
                "concentrated": imf_density_grid,
                "extended": np.zeros_like(imf_density_grid),
            },
            "component_imf_density_data": {
                "concentrated": imf_density_data,
                "extended": np.zeros_like(imf_density_data),
            },
            "mixture_component_density_data": np.clip(imf_density_data * component_density_data, TINY, None),
            "mixture_radial_density_grid": component_density_grid,
            "mixture_radial_density_data": component_density_data,
            "component_radial_density_grid": {
                "concentrated": component_density_grid,
                "extended": np.zeros_like(component_density_grid),
            },
            "component_radial_density_data": {
                "concentrated": component_density_data,
                "extended": np.zeros_like(component_density_data),
            },
            "selection_fraction_total": float(selection_fraction),
            "raw_survival_fraction_total": float(raw_survival_fraction),
            "component_selection_fraction": {
                "concentrated": float(selection_fraction),
                "extended": 0.0,
            },
            "component_raw_survival_fraction": {
                "concentrated": float(raw_survival_fraction),
                "extended": 0.0,
            },
            "total_initial_count": float(total_initial_count),
            "component_initial_count": {
                "concentrated": float(total_initial_count),
                "extended": 0.0,
            },
            "expected_observed_count": {
                "concentrated": float(len(context.log_mass_data)),
                "extended": 0.0,
            },
            "mix_fraction_concentrated": 1.0,
            "imf_parameters": imf_parameters,
            "radial_parameters": {
                "radial_model": "single_powerlaw",
                "beta_logdensity_per_log10a": beta,
                "gamma_linear_a_density": 1.0 - beta,
                "pivot_log10_a_kpc": float(context.log_a_mean),
            },
        }

    if spec.model_class != "two_component_powerlaw_mixture":
        raise ValueError(f"Unknown blind radial model class: {spec.model_class}")

    beta_mid = float(radial_params[0])
    slope_gap = float(np.exp(radial_params[1]))
    mix_fraction = bounded_mix_fraction(float(radial_params[2]))
    beta_concentrated = beta_mid - 0.5 * slope_gap
    beta_extended = beta_mid + 0.5 * slope_gap
    concentrated_grid = radial_powerlaw_density_grid(context.log_a_grid, beta_concentrated, pivot=context.log_a_mean)
    extended_grid = radial_powerlaw_density_grid(context.log_a_grid, beta_extended, pivot=context.log_a_mean)
    concentrated_data = interpolate_density(context.log_a_grid, concentrated_grid, context.log_a_data)
    extended_data = interpolate_density(context.log_a_grid, extended_grid, context.log_a_data)
    mixture_grid = mix_fraction * concentrated_grid + (1.0 - mix_fraction) * extended_grid
    mixture_grid = np.clip(mixture_grid, TINY, None)
    mixture_data = mix_fraction * concentrated_data + (1.0 - mix_fraction) * extended_data
    mixture_data = np.clip(mixture_data, TINY, None)

    selection_fraction_concentrated = integrate_survival_fraction(
        imf_density_grid,
        concentrated_grid,
        context.log_mass_grid,
        context.log_a_grid,
        context.selection_probability_grid,
    )
    selection_fraction_extended = integrate_survival_fraction(
        imf_density_grid,
        extended_grid,
        context.log_mass_grid,
        context.log_a_grid,
        context.selection_probability_grid,
    )
    raw_survival_fraction_concentrated = integrate_survival_fraction(
        imf_density_grid,
        concentrated_grid,
        context.log_mass_grid,
        context.log_a_grid,
        context.survival_probability_grid,
    )
    raw_survival_fraction_extended = integrate_survival_fraction(
        imf_density_grid,
        extended_grid,
        context.log_mass_grid,
        context.log_a_grid,
        context.survival_probability_grid,
    )
    selection_fraction_total = (
        mix_fraction * selection_fraction_concentrated
        + (1.0 - mix_fraction) * selection_fraction_extended
    )
    raw_survival_fraction_total = (
        mix_fraction * raw_survival_fraction_concentrated
        + (1.0 - mix_fraction) * raw_survival_fraction_extended
    )
    total_initial_count = len(context.log_mass_data) / max(selection_fraction_total, 1.0e-12)
    component_initial_count = {
        "concentrated": float(total_initial_count * mix_fraction),
        "extended": float(total_initial_count * (1.0 - mix_fraction)),
    }
    expected_observed_count = {
        "concentrated": float(component_initial_count["concentrated"] * selection_fraction_concentrated),
        "extended": float(component_initial_count["extended"] * selection_fraction_extended),
    }
    return {
        "spec": spec,
        "imf_density_grid": imf_density_grid,
        "imf_density_data": imf_density_data,
        "component_imf_density_grid": {
            "concentrated": imf_density_grid,
            "extended": imf_density_grid,
        },
        "component_imf_density_data": {
            "concentrated": imf_density_data,
            "extended": imf_density_data,
        },
        "mixture_component_density_data": np.clip(
            mix_fraction * imf_density_data * concentrated_data
            + (1.0 - mix_fraction) * imf_density_data * extended_data,
            TINY,
            None,
        ),
        "mixture_radial_density_grid": mixture_grid,
        "mixture_radial_density_data": mixture_data,
        "component_radial_density_grid": {
            "concentrated": concentrated_grid,
            "extended": extended_grid,
        },
        "component_radial_density_data": {
            "concentrated": concentrated_data,
            "extended": extended_data,
        },
        "selection_fraction_total": float(selection_fraction_total),
        "raw_survival_fraction_total": float(raw_survival_fraction_total),
        "component_selection_fraction": {
            "concentrated": float(selection_fraction_concentrated),
            "extended": float(selection_fraction_extended),
        },
        "component_raw_survival_fraction": {
            "concentrated": float(raw_survival_fraction_concentrated),
            "extended": float(raw_survival_fraction_extended),
        },
        "total_initial_count": float(total_initial_count),
        "component_initial_count": component_initial_count,
        "expected_observed_count": expected_observed_count,
        "mix_fraction_concentrated": float(mix_fraction),
        "imf_parameters": imf_parameters,
        "radial_parameters": {
            "radial_model": "two_component_powerlaw_mixture",
            "beta_concentrated_logdensity_per_log10a": float(beta_concentrated),
            "beta_extended_logdensity_per_log10a": float(beta_extended),
            "gamma_concentrated_linear_a_density": float(1.0 - beta_concentrated),
            "gamma_extended_linear_a_density": float(1.0 - beta_extended),
            "mix_fraction_concentrated": float(mix_fraction),
            "pivot_log10_a_kpc": float(context.log_a_mean),
        },
    }


def bounded_mix_fraction(raw_mix: float, floor: float = MIX_FRACTION_FLOOR) -> float:
    return float(floor + (1.0 - 2.0 * floor) * special.expit(raw_mix))


def radial_powerlaw_density_grid(log_a_values: np.ndarray, beta: float, pivot: float) -> np.ndarray:
    raw_density = np.exp(np.clip(beta * (log_a_values - pivot), -700.0, 700.0))
    integral = float(np.trapezoid(raw_density, log_a_values))
    return np.clip(raw_density / max(integral, 1.0e-12), TINY, None)


def blind_full_log_likelihood_from_model(
    model: dict[str, object],
    context: JointLikelihoodContext,
) -> float:
    selection_data = np.clip(
        context.selection_interpolator(np.column_stack([context.log_mass_data, context.log_a_data])),
        1.0e-12,
        1.0,
    )
    total_initial_count = float(model["total_initial_count"])
    return float(
        len(context.log_mass_data) * np.log(total_initial_count)
        - total_initial_count * model["selection_fraction_total"]
        + np.sum(np.log(selection_data))
        + np.sum(np.log(np.clip(model["mixture_component_density_data"], 1.0e-12, None)))
    )


def compute_total_initial_stellar_mass(model: dict[str, object], context: JointLikelihoodContext) -> float:
    mass_grid = np.power(10.0, context.log_mass_grid)
    total_mass = 0.0
    for component_label in COMPONENT_LABELS:
        component_count = float(model["component_initial_count"][component_label])
        if component_count <= 0.0:
            continue
        component_density = np.asarray(model["component_imf_density_grid"][component_label], dtype=float)
        mean_initial_mass = float(np.trapezoid(mass_grid * component_density, context.log_mass_grid))
        total_mass += component_count * mean_initial_mass
    return float(total_mass)


def component_membership_probabilities(model: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    if model["spec"].model_class == "single_powerlaw_radial":
        concentrated = np.ones_like(model["mixture_radial_density_data"])
        extended = np.zeros_like(concentrated)
        return concentrated, extended

    concentrated_weight = (
        model["mix_fraction_concentrated"]
        * model["component_imf_density_data"]["concentrated"]
        * model["component_radial_density_data"]["concentrated"]
    )
    extended_weight = (
        (1.0 - model["mix_fraction_concentrated"])
        * model["component_imf_density_data"]["extended"]
        * model["component_radial_density_data"]["extended"]
    )
    denominator = np.clip(concentrated_weight + extended_weight, 1.0e-12, None)
    p_concentrated = np.clip(concentrated_weight / denominator, 1.0e-12, 1.0)
    return p_concentrated, 1.0 - p_concentrated


def build_posterior_probability_table(best_payload: dict[str, object], catalog: pd.DataFrame) -> pd.DataFrame:
    p_concentrated, p_extended = component_membership_probabilities(best_payload["model"])
    output = catalog.copy()
    output["p_concentrated"] = p_concentrated
    output["p_extended"] = p_extended
    output["blind_component_label"] = np.where(output["p_concentrated"] >= 0.5, "concentrated", "extended")
    return output


def build_blind_vs_bk_group_table(posterior_probability_table: pd.DataFrame) -> pd.DataFrame:
    if "origin_flag" not in posterior_probability_table.columns:
        return pd.DataFrame([{"summary": "origin_flag column unavailable; BK comparison not computed"}])

    rows: list[dict[str, object]] = []
    for origin_flag, group in posterior_probability_table.dropna(subset=["origin_flag"]).groupby("origin_flag"):
        rows.append(
            {
                "origin_flag": int(origin_flag),
                "origin_label": "in_situ" if int(origin_flag) == 1 else "accreted",
                "n_clusters": int(len(group)),
                "mean_p_concentrated": float(group["p_concentrated"].mean()),
                "median_p_concentrated": float(group["p_concentrated"].median()),
                "hard_concentrated_count": int((group["blind_component_label"] == "concentrated").sum()),
                "hard_extended_count": int((group["blind_component_label"] == "extended").sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("origin_flag").reset_index(drop=True)


def build_blind_vs_bk_summary(posterior_probability_table: pd.DataFrame) -> BlindBkComparisonSummary:
    valid = posterior_probability_table.dropna(subset=["origin_flag"]).copy()
    if valid.empty:
        return BlindBkComparisonSummary(
            n_clusters=len(posterior_probability_table),
            n_with_origin_flag=0,
            auc_in_situ_vs_p_concentrated=np.nan,
            hard_assignment_accuracy=np.nan,
            mean_p_concentrated_in_situ=np.nan,
            mean_p_concentrated_accreted=np.nan,
        )

    labels = valid["origin_flag"].astype(int).to_numpy()
    scores = valid["p_concentrated"].to_numpy()
    predicted = (scores >= 0.5).astype(int)
    auc = binary_auc(scores, labels)
    mean_in_situ = float(valid.loc[valid["origin_flag"] == 1, "p_concentrated"].mean())
    mean_accreted = float(valid.loc[valid["origin_flag"] == 0, "p_concentrated"].mean())
    return BlindBkComparisonSummary(
        n_clusters=len(posterior_probability_table),
        n_with_origin_flag=len(valid),
        auc_in_situ_vs_p_concentrated=float(auc),
        hard_assignment_accuracy=float(np.mean(predicted == labels)),
        mean_p_concentrated_in_situ=mean_in_situ,
        mean_p_concentrated_accreted=mean_accreted,
    )


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_positive = int(np.sum(labels == 1))
    n_negative = int(np.sum(labels == 0))
    if n_positive == 0 or n_negative == 0:
        return np.nan
    ranks = stats.rankdata(scores, method="average")
    positive_rank_sum = float(np.sum(ranks[labels == 1]))
    return float(
        (positive_rank_sum - n_positive * (n_positive + 1) / 2.0) / max(n_positive * n_negative, 1.0)
    )


def build_imf_grid_table(best_payload: dict[str, object], context: JointLikelihoodContext) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for log_mass, density in zip(context.log_mass_grid, best_payload["model"]["imf_density_grid"], strict=True):
        rows.append(
            {
                "imf_family": best_payload["summary"].imf_family,
                "model_class": best_payload["summary"].model_class,
                "log_initial_mass_msun": float(log_mass),
                "initial_mass_msun": float(np.power(10.0, log_mass)),
                "imf_density_per_dex": float(density),
                "birth_imf_per_dex": float(best_payload["model"]["total_initial_count"] * density),
            }
        )
    return pd.DataFrame(rows)


def build_radial_grid_table(best_payload: dict[str, object], context: JointLikelihoodContext) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    model = best_payload["model"]
    total_initial_count = float(model["total_initial_count"])
    mix_fraction = float(model["mix_fraction_concentrated"])
    for index, log_a in enumerate(context.log_a_grid):
        rows.append(
            {
                "component_label": "total",
                "imf_family": best_payload["summary"].imf_family,
                "model_class": best_payload["summary"].model_class,
                "log10_semi_major_axis_kpc": float(log_a),
                "semi_major_axis_kpc": float(np.power(10.0, log_a)),
                "radial_density_per_dex_a": float(model["mixture_radial_density_grid"][index]),
                "birth_intensity_per_dex_a": float(total_initial_count * model["mixture_radial_density_grid"][index]),
            }
        )
        rows.append(
            {
                "component_label": "concentrated",
                "imf_family": best_payload["summary"].imf_family,
                "model_class": best_payload["summary"].model_class,
                "log10_semi_major_axis_kpc": float(log_a),
                "semi_major_axis_kpc": float(np.power(10.0, log_a)),
                "radial_density_per_dex_a": float(model["component_radial_density_grid"]["concentrated"][index]),
                "birth_intensity_per_dex_a": float(
                    total_initial_count * mix_fraction * model["component_radial_density_grid"]["concentrated"][index]
                ),
            }
        )
        rows.append(
            {
                "component_label": "extended",
                "imf_family": best_payload["summary"].imf_family,
                "model_class": best_payload["summary"].model_class,
                "log10_semi_major_axis_kpc": float(log_a),
                "semi_major_axis_kpc": float(np.power(10.0, log_a)),
                "radial_density_per_dex_a": float(model["component_radial_density_grid"]["extended"][index]),
                "birth_intensity_per_dex_a": float(
                    total_initial_count
                    * (1.0 - mix_fraction)
                    * model["component_radial_density_grid"]["extended"][index]
                ),
            }
        )
    return pd.DataFrame(rows)
