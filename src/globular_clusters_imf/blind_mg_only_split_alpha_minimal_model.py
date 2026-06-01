from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

from .blind_mg_only_minimal_model import (
    DEFAULT_DETECTABILITY_SUMMARY,
    BlindMinimalMgFitResult,
    MgObservationContext,
    TINY,
    bounded_logistic_mix,
    build_component_radial_grid_table,
    build_mg_observation_context,
    build_mg_posterior_probability_table,
    build_mg_density_grid_table,
    compute_total_initial_stellar_mass,
    fit_single_mg_gaussian_model,
    fit_two_component_mg_mixture_model,
    mg_gaussian_density,
)
from .blind_mixture_model import (
    build_blind_vs_bk_group_table,
    build_blind_vs_bk_summary,
    build_fixed_abs_longitude_selection_payload,
)
from .joint_model import (
    JointLikelihoodContext,
    JointModelSpec,
    evaluate_imf_family,
    fit_single_joint_model,
)

DEFAULT_OUTPUT_PREFIX = "blind_mg_only_split_alpha_minimal_mixture"


@dataclass
class BlindMgSplitAlphaFitResult:
    model_class: str
    success: bool
    joint_log_likelihood: float
    aic: float
    bic: float
    delta_bic: float
    n_parameters: int
    n_clusters_total: int
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
    imf_parameters_json: str


def fit_mg_only_split_alpha_models(
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

    baseline_imf_parameters = dict(base_payload["model"]["imf_parameters"])
    baseline_alpha = float(baseline_imf_parameters["alpha_dndm"])
    baseline_log_m_c = float(baseline_imf_parameters["log10_m_c_msun"])
    baseline_mass_density_all = np.asarray(base_payload["model"]["imf_density_data"], dtype=float)
    baseline_mass_log_likelihood = float(np.sum(np.log(np.clip(baseline_mass_density_all, TINY, None))))

    single_shared = fit_single_mg_gaussian_model(base_payload, mg_context)
    two_shared = fit_two_component_mg_mixture_model(base_payload, mg_context)
    split_alpha = fit_two_component_mg_split_alpha_mixture_model(
        base_payload=base_payload,
        mg_context=mg_context,
        baseline_alpha=baseline_alpha,
        baseline_log_m_c=baseline_log_m_c,
    )

    payloads = [
        finalize_shared_alpha_payload(
            payload=single_shared,
            base_payload=base_payload,
            baseline_alpha=baseline_alpha,
            baseline_log_m_c=baseline_log_m_c,
            baseline_mass_log_likelihood=baseline_mass_log_likelihood,
            n_total=len(catalog),
            n_with_mg=int(mg_context.has_mgfe.sum()),
        ),
        finalize_shared_alpha_payload(
            payload=two_shared,
            base_payload=base_payload,
            baseline_alpha=baseline_alpha,
            baseline_log_m_c=baseline_log_m_c,
            baseline_mass_log_likelihood=baseline_mass_log_likelihood,
            n_total=len(catalog),
            n_with_mg=int(mg_context.has_mgfe.sum()),
        ),
        split_alpha,
    ]

    summary_table = pd.DataFrame([asdict(payload["summary"]) for payload in payloads]).sort_values(
        ["bic", "aic"], ascending=[True, True]
    ).reset_index(drop=True)
    best_bic = float(summary_table["bic"].min())
    summary_table["delta_bic"] = summary_table["bic"] - best_bic
    for payload in payloads:
        payload["summary"].delta_bic = float(payload["summary"].bic - best_bic)

    shared_payload = find_payload(payloads, "two_component_mg_mixture")
    split_payload = find_payload(payloads, "two_component_mg_split_alpha_mixture")

    shared_posterior = build_mg_posterior_probability_table(shared_payload, catalog, mg_context)
    split_posterior = build_mg_split_alpha_posterior_probability_table(split_payload, catalog, mg_context)
    bk_comparison = build_blind_vs_bk_summary(split_posterior)
    bk_group_table = build_blind_vs_bk_group_table(split_posterior)
    radial_grid_table = build_component_radial_grid_table(
        split_payload,
        {"total_initial_count": split_payload["summary"].total_initial_count, "radial_density_grid": base_payload["model"]["radial_density_grid"]},
        selection_payload["selection_context"],
    )
    mg_grid_table = build_mg_density_grid_table_for_split_alpha(split_payload, mg_context)
    imf_grid_table = build_imf_grid_table_for_split_alpha(split_payload, selection_payload["selection_context"], baseline_alpha, baseline_log_m_c)

    summary_json = {
        "selection_payload": selection_payload["summary_payload"],
        "base_model": {
            "imf_family": base_payload["summary"].imf_family,
            "radial_model": base_payload["summary"].radial_model,
            "log_likelihood": float(base_payload["summary"].log_likelihood),
            "bic": float(base_payload["summary"].bic),
            "total_initial_count": float(base_payload["summary"].total_initial_count),
            "survival_fraction": float(base_payload["summary"].survival_fraction),
            "imf_parameters": baseline_imf_parameters,
            "radial_parameters": json.loads(base_payload["summary"].radial_parameters_json),
        },
        "mg_summary": {
            "n_clusters_total": int(len(catalog)),
            "n_clusters_with_mg": int(mg_context.has_mgfe.sum()),
        },
        "all_models_ranked": summary_table.to_dict(orient="records"),
        "shared_alpha_model": asdict(shared_payload["summary"]),
        "split_alpha_model": asdict(split_payload["summary"]),
        "split_alpha_bk_comparison": asdict(bk_comparison),
        "delta_log_likelihood_split_minus_shared": float(
            split_payload["summary"].joint_log_likelihood - shared_payload["summary"].joint_log_likelihood
        ),
        "delta_bic_split_minus_shared": float(
            split_payload["summary"].bic - shared_payload["summary"].bic
        ),
        "delta_aic_split_minus_shared": float(
            split_payload["summary"].aic - shared_payload["summary"].aic
        ),
    }

    outputs_tables = output_root / "outputs" / "tables"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    summary_table.to_csv(outputs_tables / f"{output_prefix}_model_summary.csv", index=False)
    shared_posterior.to_csv(outputs_tables / f"{output_prefix}_shared_alpha_posterior_probabilities.csv", index=False)
    split_posterior.to_csv(outputs_tables / f"{output_prefix}_split_alpha_posterior_probabilities.csv", index=False)
    bk_group_table.to_csv(outputs_tables / f"{output_prefix}_split_alpha_vs_bk_groups.csv", index=False)
    radial_grid_table.to_csv(outputs_tables / f"{output_prefix}_split_alpha_component_radial_grid.csv", index=False)
    mg_grid_table.to_csv(outputs_tables / f"{output_prefix}_split_alpha_mg_density_grid.csv", index=False)
    imf_grid_table.to_csv(outputs_tables / f"{output_prefix}_split_alpha_imf_grid.csv", index=False)
    (outputs_tables / f"{output_prefix}_comparison_summary.json").write_text(json.dumps(summary_json, indent=2))

    return {
        "selection_payload": selection_payload,
        "base_payload": base_payload,
        "mg_context": mg_context,
        "summary_table": summary_table,
        "shared_payload": shared_payload,
        "split_payload": split_payload,
        "shared_posterior": shared_posterior,
        "split_posterior": split_posterior,
        "bk_comparison": bk_comparison,
        "bk_group_table": bk_group_table,
        "radial_grid_table": radial_grid_table,
        "mg_grid_table": mg_grid_table,
        "imf_grid_table": imf_grid_table,
        "all_payloads": payloads,
    }


def find_payload(payloads: list[dict[str, object]], model_class: str) -> dict[str, object]:
    for payload in payloads:
        if payload["summary"].model_class == model_class:
            return payload
    raise KeyError(model_class)


def finalize_shared_alpha_payload(
    payload: dict[str, object],
    base_payload: dict[str, object],
    baseline_alpha: float,
    baseline_log_m_c: float,
    baseline_mass_log_likelihood: float,
    n_total: int,
    n_with_mg: int,
) -> dict[str, object]:
    old_summary: BlindMinimalMgFitResult = payload["summary"]
    base_model = base_payload["model"]
    summary = BlindMgSplitAlphaFitResult(
        model_class=old_summary.model_class,
        success=old_summary.success,
        joint_log_likelihood=float(old_summary.chemistry_log_likelihood + baseline_mass_log_likelihood),
        aic=float(2 * old_summary.n_parameters - 2 * (old_summary.chemistry_log_likelihood + baseline_mass_log_likelihood)),
        bic=float(np.log(max(n_total, 1)) * old_summary.n_parameters - 2 * (old_summary.chemistry_log_likelihood + baseline_mass_log_likelihood)),
        delta_bic=np.nan,
        n_parameters=int(old_summary.n_parameters),
        n_clusters_total=int(n_total),
        n_clusters_with_mg=int(n_with_mg),
        total_initial_count=float(old_summary.total_initial_count),
        total_initial_stellar_mass_msun=float(old_summary.total_initial_stellar_mass_msun),
        selection_fraction=float(old_summary.selection_fraction),
        raw_survival_fraction=float(old_summary.raw_survival_fraction),
        mean_detectability=float(old_summary.mean_detectability),
        imf_family=str(old_summary.imf_family),
        radial_model=str(old_summary.radial_model),
        component_mix_fraction_concentrated=float(old_summary.component_mix_fraction_concentrated),
        component_mix_fraction_extended=float(old_summary.component_mix_fraction_extended),
        component_initial_count_concentrated=float(old_summary.component_initial_count_concentrated),
        component_initial_count_extended=float(old_summary.component_initial_count_extended),
        expected_observed_count_concentrated=float(old_summary.expected_observed_count_concentrated),
        expected_observed_count_extended=float(old_summary.expected_observed_count_extended),
        optimizer_message=str(old_summary.optimizer_message),
        chemistry_parameters_json=str(old_summary.chemistry_parameters_json),
        imf_parameters_json=json.dumps(
            {
                "alpha_dndm_concentrated": baseline_alpha,
                "alpha_dndm_extended": baseline_alpha,
                "shared_log10_m_c_msun": baseline_log_m_c,
            }
        ),
    )
    payload["summary"] = summary
    payload["model"]["alpha_dndm_concentrated"] = baseline_alpha
    payload["model"]["alpha_dndm_extended"] = baseline_alpha
    payload["model"]["shared_log10_m_c_msun"] = baseline_log_m_c
    payload["model"]["component_imf_density_grid"] = {
        "concentrated": np.asarray(base_model["imf_density_grid"], dtype=float),
        "extended": np.asarray(base_model["imf_density_grid"], dtype=float),
    }
    payload["model"]["component_imf_density_data"] = {
        "concentrated": np.asarray(base_model["imf_density_data"], dtype=float),
        "extended": np.asarray(base_model["imf_density_data"], dtype=float),
    }
    return payload


def fit_two_component_mg_split_alpha_mixture_model(
    base_payload: dict[str, object],
    mg_context: MgObservationContext,
    baseline_alpha: float,
    baseline_log_m_c: float,
) -> dict[str, object]:
    shared_payload = fit_two_component_mg_mixture_model(base_payload, mg_context)
    shared_model = shared_payload["model"]
    measured_mg = mg_context.mgfe[mg_context.has_mgfe]
    measured_err = mg_context.mgfe_err[mg_context.has_mgfe]
    measured_x = mg_context.log_a_standardized_mg
    context: JointLikelihoodContext = base_payload["context"]
    baseline_radial_density_grid = np.asarray(base_payload["model"]["radial_density_grid"], dtype=float)

    starts = [
        np.array(
            [
                float(shared_model["c0"]),
                float(shared_model["c1"]),
                float(shared_model["mu_mgfe_concentrated"]),
                float(shared_model["mu_mgfe_extended"]),
                float(np.log(shared_model["sigma_mgfe_shared"])),
                baseline_alpha,
                baseline_alpha,
            ],
            dtype=float,
        ),
        np.array(
            [
                float(shared_model["c0"]),
                float(shared_model["c1"]),
                float(shared_model["mu_mgfe_concentrated"]),
                float(shared_model["mu_mgfe_extended"]),
                float(np.log(shared_model["sigma_mgfe_shared"])),
                baseline_alpha + 0.35,
                baseline_alpha - 0.35,
            ],
            dtype=float,
        ),
        np.array(
            [
                float(shared_model["c0"]),
                float(shared_model["c1"]),
                float(shared_model["mu_mgfe_concentrated"]),
                float(shared_model["mu_mgfe_extended"]),
                float(np.log(shared_model["sigma_mgfe_shared"])),
                baseline_alpha - 0.35,
                baseline_alpha + 0.35,
            ],
            dtype=float,
        ),
    ]
    bounds = [
        (-8.0, 8.0),
        (-10.0, 0.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
        (np.log(0.01), np.log(0.6)),
        (-4.0, -0.2),
        (-4.0, -0.2),
    ]
    best_result = None
    best_value = np.inf
    for start in starts:
        result = optimize.minimize(
            lambda params: split_alpha_negative_log_likelihood(
                params,
                context=context,
                mg_context=mg_context,
                baseline_log_m_c=baseline_log_m_c,
                baseline_radial_density_grid=baseline_radial_density_grid,
            ),
            x0=start,
            method="L-BFGS-B",
            bounds=bounds,
        )
        if result.fun < best_value:
            best_value = float(result.fun)
            best_result = result
    if best_result is None:
        raise RuntimeError("Split-alpha Mg-only mixture optimization did not start.")

    model = unpack_split_alpha_model(
        best_result.x,
        context=context,
        mg_context=mg_context,
        baseline_log_m_c=baseline_log_m_c,
        baseline_radial_density_grid=baseline_radial_density_grid,
    )
    summary = build_split_alpha_summary(
        model=model,
        base_payload=base_payload,
        context=context,
        baseline_radial_density_grid=baseline_radial_density_grid,
        n_total=len(context.log_mass_data),
        n_with_mg=int(mg_context.has_mgfe.sum()),
        success=bool(best_result.success),
        optimizer_message=str(best_result.message),
    )
    return {"summary": summary, "raw_parameters": np.asarray(best_result.x, dtype=float), "model": model}


def split_alpha_negative_log_likelihood(
    params: np.ndarray,
    context: JointLikelihoodContext,
    mg_context: MgObservationContext,
    baseline_log_m_c: float,
    baseline_radial_density_grid: np.ndarray,
) -> float:
    model = unpack_split_alpha_model(
        params,
        context=context,
        mg_context=mg_context,
        baseline_log_m_c=baseline_log_m_c,
        baseline_radial_density_grid=baseline_radial_density_grid,
    )
    return float(-model["joint_log_likelihood"])


def unpack_split_alpha_model(
    params: np.ndarray,
    context: JointLikelihoodContext,
    mg_context: MgObservationContext,
    baseline_log_m_c: float,
    baseline_radial_density_grid: np.ndarray,
) -> dict[str, object]:
    c0, c1, mu_conc, mu_ext, log_sigma, alpha_conc, alpha_ext = np.asarray(params, dtype=float)
    sigma = float(np.exp(log_sigma))
    w_all = bounded_logistic_mix(c0 + c1 * mg_context.log_a_standardized_all)
    w_grid = bounded_logistic_mix(c0 + c1 * ((context.log_a_grid - context.log_a_mean) / context.log_a_std))

    imf_grid_conc, imf_data_conc, _ = evaluate_imf_family(
        "schechter",
        np.array([alpha_conc, baseline_log_m_c], dtype=float),
        context.log_mass_grid,
        context.log_mass_data,
    )
    imf_grid_ext, imf_data_ext, _ = evaluate_imf_family(
        "schechter",
        np.array([alpha_ext, baseline_log_m_c], dtype=float),
        context.log_mass_grid,
        context.log_mass_data,
    )

    chem_conc = np.ones_like(context.log_mass_data, dtype=float)
    chem_ext = np.ones_like(context.log_mass_data, dtype=float)
    measured = mg_context.has_mgfe
    chem_conc[measured] = mg_gaussian_density(
        mg_context.mgfe[measured],
        mg_context.mgfe_err[measured],
        mu=float(mu_conc),
        sigma=sigma,
    )
    chem_ext[measured] = mg_gaussian_density(
        mg_context.mgfe[measured],
        mg_context.mgfe_err[measured],
        mu=float(mu_ext),
        sigma=sigma,
    )
    component_conc = np.clip(w_all * imf_data_conc * chem_conc, TINY, None)
    component_ext = np.clip((1.0 - w_all) * imf_data_ext * chem_ext, TINY, None)
    mixture_density_data = np.clip(component_conc + component_ext, TINY, None)
    joint_log_likelihood = float(np.sum(np.log(mixture_density_data)))

    radial_grid = np.asarray(baseline_radial_density_grid, dtype=float)
    radial_grid_conc = radial_grid * w_grid
    radial_grid_ext = radial_grid * (1.0 - w_grid)
    mixture_grid = imf_grid_conc[:, None] * radial_grid_conc[None, :] + imf_grid_ext[:, None] * radial_grid_ext[None, :]
    selection_fraction = float(
        np.trapezoid(
            np.trapezoid(mixture_grid * context.selection_probability_grid, context.log_a_grid, axis=1),
            context.log_mass_grid,
        )
    )
    raw_survival_fraction = float(
        np.trapezoid(
            np.trapezoid(mixture_grid * context.survival_probability_grid, context.log_a_grid, axis=1),
            context.log_mass_grid,
        )
    )
    total_initial_count = float(len(context.log_mass_data) / max(selection_fraction, 1.0e-12))
    mix_fraction_concentrated = float(np.trapezoid(radial_grid_conc, context.log_a_grid))
    mix_fraction_concentrated = float(np.clip(mix_fraction_concentrated, TINY, 1.0 - TINY))

    return {
        "model_class": "two_component_mg_split_alpha_mixture",
        "c0": float(c0),
        "c1": float(c1),
        "mu_mgfe_concentrated": float(mu_conc),
        "mu_mgfe_extended": float(mu_ext),
        "sigma_mgfe_shared": sigma,
        "alpha_dndm_concentrated": float(alpha_conc),
        "alpha_dndm_extended": float(alpha_ext),
        "shared_log10_m_c_msun": float(baseline_log_m_c),
        "joint_log_likelihood": joint_log_likelihood,
        "w_all": np.clip(w_all, TINY, 1.0 - TINY),
        "w_grid": np.clip(w_grid, TINY, 1.0 - TINY),
        "mix_fraction_concentrated": mix_fraction_concentrated,
        "mix_fraction_extended": float(1.0 - mix_fraction_concentrated),
        "component_imf_density_grid": {
            "concentrated": np.asarray(imf_grid_conc, dtype=float),
            "extended": np.asarray(imf_grid_ext, dtype=float),
        },
        "component_imf_density_data": {
            "concentrated": np.asarray(imf_data_conc, dtype=float),
            "extended": np.asarray(imf_data_ext, dtype=float),
        },
        "component_chem_density_data": {
            "concentrated": np.asarray(chem_conc, dtype=float),
            "extended": np.asarray(chem_ext, dtype=float),
        },
        "mixture_density_data": mixture_density_data,
        "selection_fraction": selection_fraction,
        "raw_survival_fraction": raw_survival_fraction,
        "total_initial_count": total_initial_count,
    }


def build_split_alpha_summary(
    model: dict[str, object],
    base_payload: dict[str, object],
    context: JointLikelihoodContext,
    baseline_radial_density_grid: np.ndarray,
    n_total: int,
    n_with_mg: int,
    success: bool,
    optimizer_message: str,
) -> BlindMgSplitAlphaFitResult:
    base_model = base_payload["model"]
    radial_grid = np.asarray(baseline_radial_density_grid, dtype=float)
    w_grid = np.asarray(model["w_grid"], dtype=float)
    total_initial_count = float(base_model["total_initial_count"])
    radial_grid_conc = radial_grid * w_grid
    radial_grid_ext = radial_grid * (1.0 - w_grid)
    component_initial_count_concentrated = float(total_initial_count * np.trapezoid(radial_grid_conc, context.log_a_grid))
    component_initial_count_extended = float(total_initial_count * np.trapezoid(radial_grid_ext, context.log_a_grid))
    expected_observed_count_concentrated = float(
        total_initial_count
        * np.trapezoid(
            np.trapezoid(
                model["component_imf_density_grid"]["concentrated"][:, None]
                * radial_grid_conc[None, :]
                * context.selection_probability_grid,
                context.log_a_grid,
                axis=1,
            ),
            context.log_mass_grid,
        )
    )
    expected_observed_count_extended = float(
        total_initial_count
        * np.trapezoid(
            np.trapezoid(
                model["component_imf_density_grid"]["extended"][:, None]
                * radial_grid_ext[None, :]
                * context.selection_probability_grid,
                context.log_a_grid,
                axis=1,
            ),
            context.log_mass_grid,
        )
    )
    mean_mass_conc = float(
        np.trapezoid(
            np.power(10.0, context.log_mass_grid) * model["component_imf_density_grid"]["concentrated"],
            context.log_mass_grid,
        )
    )
    mean_mass_ext = float(
        np.trapezoid(
            np.power(10.0, context.log_mass_grid) * model["component_imf_density_grid"]["extended"],
            context.log_mass_grid,
        )
    )
    total_initial_stellar_mass = float(
        component_initial_count_concentrated * mean_mass_conc
        + component_initial_count_extended * mean_mass_ext
    )
    chemistry_parameters = {
        "c0": model["c0"],
        "c1": model["c1"],
        "mu_mgfe_concentrated": model["mu_mgfe_concentrated"],
        "mu_mgfe_extended": model["mu_mgfe_extended"],
        "sigma_mgfe_shared": model["sigma_mgfe_shared"],
        "mix_fraction_floor": 0.05,
    }
    imf_parameters = {
        "alpha_dndm_concentrated": model["alpha_dndm_concentrated"],
        "alpha_dndm_extended": model["alpha_dndm_extended"],
        "shared_log10_m_c_msun": model["shared_log10_m_c_msun"],
    }
    return BlindMgSplitAlphaFitResult(
        model_class=str(model["model_class"]),
        success=bool(success),
        joint_log_likelihood=float(model["joint_log_likelihood"]),
        aic=float(2 * 7 - 2 * model["joint_log_likelihood"]),
        bic=float(np.log(max(n_total, 1)) * 7 - 2 * model["joint_log_likelihood"]),
        delta_bic=np.nan,
        n_parameters=7,
        n_clusters_total=int(n_total),
        n_clusters_with_mg=int(n_with_mg),
        total_initial_count=total_initial_count,
        total_initial_stellar_mass_msun=total_initial_stellar_mass,
        selection_fraction=float(base_model["selection_fraction"]),
        raw_survival_fraction=float(base_model["raw_survival_fraction"]),
        mean_detectability=float(
            base_model["selection_fraction"] / max(base_model["raw_survival_fraction"], 1.0e-12)
        ),
        imf_family="schechter",
        radial_model="logpoly3",
        component_mix_fraction_concentrated=float(model["mix_fraction_concentrated"]),
        component_mix_fraction_extended=float(model["mix_fraction_extended"]),
        component_initial_count_concentrated=component_initial_count_concentrated,
        component_initial_count_extended=component_initial_count_extended,
        expected_observed_count_concentrated=expected_observed_count_concentrated,
        expected_observed_count_extended=expected_observed_count_extended,
        optimizer_message=str(optimizer_message),
        chemistry_parameters_json=json.dumps(chemistry_parameters),
        imf_parameters_json=json.dumps(imf_parameters),
    )


def build_mg_split_alpha_posterior_probability_table(
    split_payload: dict[str, object],
    catalog: pd.DataFrame,
    mg_context: MgObservationContext,
) -> pd.DataFrame:
    model = split_payload["model"]
    output = catalog.copy()
    w = np.asarray(model["w_all"], dtype=float)
    chem_conc = np.asarray(model["component_chem_density_data"]["concentrated"], dtype=float)
    chem_ext = np.asarray(model["component_chem_density_data"]["extended"], dtype=float)
    mass_conc = np.asarray(model["component_imf_density_data"]["concentrated"], dtype=float)
    mass_ext = np.asarray(model["component_imf_density_data"]["extended"], dtype=float)
    numerator = w * chem_conc * mass_conc
    denominator = np.clip(numerator + (1.0 - w) * chem_ext * mass_ext, TINY, None)
    p_concentrated = np.clip(numerator / denominator, TINY, 1.0)
    output["has_mgfe"] = mg_context.has_mgfe
    output["p_concentrated"] = p_concentrated
    output["p_extended"] = 1.0 - p_concentrated
    output["blind_component_label"] = np.where(output["p_concentrated"] >= 0.5, "concentrated", "extended")
    return output


def build_mg_density_grid_table_for_split_alpha(
    split_payload: dict[str, object],
    mg_context: MgObservationContext,
) -> pd.DataFrame:
    measured = mg_context.mgfe[mg_context.has_mgfe]
    mg_grid = np.linspace(float(np.nanmin(measured)) - 0.1, float(np.nanmax(measured)) + 0.1, 300)
    model = split_payload["model"]
    sigma = float(model["sigma_mgfe_shared"])
    return pd.DataFrame(
        {
            "mgfe": mg_grid,
            "concentrated_density": mg_gaussian_density(
                mg_grid,
                np.zeros_like(mg_grid),
                mu=float(model["mu_mgfe_concentrated"]),
                sigma=sigma,
            ),
            "extended_density": mg_gaussian_density(
                mg_grid,
                np.zeros_like(mg_grid),
                mu=float(model["mu_mgfe_extended"]),
                sigma=sigma,
            ),
        }
    )


def build_imf_grid_table_for_split_alpha(
    split_payload: dict[str, object],
    context: JointLikelihoodContext,
    baseline_alpha: float,
    baseline_log_m_c: float,
) -> pd.DataFrame:
    baseline_grid, _, _ = evaluate_imf_family(
        "schechter",
        np.array([baseline_alpha, baseline_log_m_c], dtype=float),
        context.log_mass_grid,
        context.log_mass_data,
    )
    model = split_payload["model"]
    return pd.DataFrame(
        {
            "log_initial_mass_msun": context.log_mass_grid,
            "baseline_imf_density": baseline_grid,
            "concentrated_imf_density": model["component_imf_density_grid"]["concentrated"],
            "extended_imf_density": model["component_imf_density_grid"]["extended"],
        }
    )
