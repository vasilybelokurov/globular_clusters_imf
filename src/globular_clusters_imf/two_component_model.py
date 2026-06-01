from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .joint_model import (
    JointLikelihoodContext,
    JointModelSpec,
    build_fixed_survival_grid,
    calibrate_fixed_selection_offset_dex,
    compute_observed_intensity_grid,
    evaluate_imf_family,
    evaluate_radial_model,
    fit_single_joint_model,
    imf_parameter_count,
    initial_parameter_vectors,
    integrate_survival_fraction,
    make_imf_grid_rows,
    make_radial_grid_rows,
    parameter_bounds,
)


@dataclass
class TwoComponentJointFitResult:
    in_situ_imf_family: str
    in_situ_radial_model: str
    accreted_imf_family: str
    accreted_radial_model: str
    log_likelihood: float
    aic: float
    bic: float
    delta_bic: float
    n_parameters: int
    n_clusters_total: int
    n_clusters_in_situ: int
    n_clusters_accreted: int
    total_initial_count_in_situ: float
    total_initial_count_accreted: float
    total_initial_count: float
    survival_fraction_in_situ: float
    survival_fraction_accreted: float


@dataclass(frozen=True)
class SharedImfTwoComponentSpec:
    imf_family: str
    in_situ_radial_model: str
    accreted_radial_model: str


@dataclass(frozen=True)
class SplitAlphaTwoComponentSpec:
    in_situ_radial_model: str
    accreted_radial_model: str


@dataclass
class SharedImfTwoComponentJointFitResult:
    imf_family: str
    in_situ_radial_model: str
    accreted_radial_model: str
    log_likelihood: float
    aic: float
    bic: float
    delta_bic: float
    n_parameters: int
    n_clusters_total: int
    n_clusters_in_situ: int
    n_clusters_accreted: int
    total_initial_count_in_situ: float
    total_initial_count_accreted: float
    total_initial_count: float
    survival_fraction_in_situ: float
    survival_fraction_accreted: float
    shared_imf_parameters_json: str
    in_situ_radial_parameters_json: str
    accreted_radial_parameters_json: str


@dataclass
class SplitAlphaTwoComponentJointFitResult:
    imf_family: str
    in_situ_radial_model: str
    accreted_radial_model: str
    log_likelihood: float
    aic: float
    bic: float
    delta_bic: float
    n_parameters: int
    n_clusters_total: int
    n_clusters_in_situ: int
    n_clusters_accreted: int
    total_initial_count_in_situ: float
    total_initial_count_accreted: float
    total_initial_count: float
    survival_fraction_in_situ: float
    survival_fraction_accreted: float
    shared_log10_m_c_msun: float
    in_situ_alpha_dndm: float
    accreted_alpha_dndm: float
    in_situ_imf_parameters_json: str
    accreted_imf_parameters_json: str
    in_situ_radial_parameters_json: str
    accreted_radial_parameters_json: str


def fit_two_component_fixed_survival_joint_models(
    catalog: pd.DataFrame,
    project_root: Path,
    component_model_specs: list[JointModelSpec] | None = None,
) -> dict[str, object]:
    if component_model_specs is None:
        component_model_specs = [
            JointModelSpec(imf_family="lognormal", radial_model="step5"),
            JointModelSpec(imf_family="powerlaw", radial_model="step5"),
            JointModelSpec(imf_family="schechter", radial_model="step5"),
            JointModelSpec(imf_family="lognormal", radial_model="logpoly3"),
            JointModelSpec(imf_family="powerlaw", radial_model="logpoly3"),
            JointModelSpec(imf_family="schechter", radial_model="logpoly3"),
        ]

    working, subsets, selection_offset_dex, survival_grid, contexts = prepare_two_component_contexts(catalog)

    component_payloads = {
        component_label: [
            fit_single_joint_model(context=contexts[component_label], spec=spec)
            for spec in component_model_specs
        ]
        for component_label in subsets
    }

    component_summary_table = build_component_summary_table(
        component_payloads=component_payloads,
        n_clusters_by_component={label: len(subset) for label, subset in subsets.items()},
    )
    pair_payloads = build_two_component_pair_payloads(
        in_situ_payloads=component_payloads["in_situ"],
        accreted_payloads=component_payloads["accreted"],
        n_clusters_in_situ=len(subsets["in_situ"]),
        n_clusters_accreted=len(subsets["accreted"]),
    )
    pair_summary_table = pd.DataFrame([asdict(payload["summary"]) for payload in pair_payloads]).sort_values(
        "bic",
        ascending=True,
    ).reset_index(drop=True)
    if pair_summary_table.empty:
        raise RuntimeError("No two-component joint fits were produced.")
    best_pair_payload = min(pair_payloads, key=lambda payload: payload["summary"].bic)

    best_component_summary_table = pd.DataFrame(
        [
            best_pair_component_row(component_label, payload, len(subsets[component_label]))
            for component_label, payload in best_pair_payload["component_payloads"].items()
        ]
    ).sort_values("component_label")
    best_imf_grid_table = build_best_component_imf_grid_table(
        best_pair_payload["component_payloads"],
        contexts,
    )
    best_radial_grid_table = build_best_component_radial_grid_table(
        best_pair_payload["component_payloads"],
        contexts,
    )
    catalog_prediction_table = build_best_component_catalog_prediction_table(
        subsets=subsets,
        contexts=contexts,
        component_payloads=best_pair_payload["component_payloads"],
    )

    outputs_tables = project_root / "outputs" / "tables"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    component_summary_table.to_csv(
        outputs_tables / "joint_fixed_survival_two_component_component_model_summary.csv",
        index=False,
    )
    pair_summary_table.to_csv(
        outputs_tables / "joint_fixed_survival_two_component_model_summary.csv",
        index=False,
    )
    best_component_summary_table.to_csv(
        outputs_tables / "joint_fixed_survival_two_component_best_component_summary.csv",
        index=False,
    )
    best_imf_grid_table.to_csv(
        outputs_tables / "joint_fixed_survival_two_component_best_imf_grids.csv",
        index=False,
    )
    best_radial_grid_table.to_csv(
        outputs_tables / "joint_fixed_survival_two_component_best_radial_grids.csv",
        index=False,
    )
    catalog_prediction_table.to_csv(
        outputs_tables / "joint_fixed_survival_two_component_catalog_predictions.csv",
        index=False,
    )

    detailed_summary = {
        "selection_offset_dex": selection_offset_dex,
        "survival_grid_bandwidth_log10_a_dex": survival_grid["bandwidth_log10_a_dex"],
        "n_clusters_total": int(len(working)),
        "n_clusters_in_situ": int(len(subsets["in_situ"])),
        "n_clusters_accreted": int(len(subsets["accreted"])),
        "best_joint_model": asdict(best_pair_payload["summary"]),
        "best_component_models": best_component_summary_table.to_dict(orient="records"),
        "all_component_models_ranked": component_summary_table.to_dict(orient="records"),
        "all_joint_models_ranked": pair_summary_table.to_dict(orient="records"),
    }
    (outputs_tables / "joint_fixed_survival_two_component_model_summary.json").write_text(
        json.dumps(detailed_summary, indent=2)
    )

    return {
        "component_summary_table": component_summary_table,
        "pair_summary_table": pair_summary_table,
        "best_component_summary_table": best_component_summary_table,
        "best_imf_grid_table": best_imf_grid_table,
        "best_radial_grid_table": best_radial_grid_table,
        "catalog_prediction_table": catalog_prediction_table,
        "best_pair_payload": best_pair_payload,
        "contexts": contexts,
        "subsets": subsets,
        "survival_grid": survival_grid,
    }


def fit_shared_imf_two_component_fixed_survival_joint_models(
    catalog: pd.DataFrame,
    project_root: Path,
    imf_families: list[str] | None = None,
    radial_models: list[str] | None = None,
) -> dict[str, object]:
    if imf_families is None:
        imf_families = ["lognormal", "powerlaw", "schechter"]
    if radial_models is None:
        radial_models = ["step5", "logpoly3"]

    working, subsets, selection_offset_dex, survival_grid, contexts = prepare_two_component_contexts(catalog)
    model_specs = build_shared_imf_two_component_specs(imf_families=imf_families, radial_models=radial_models)
    payloads = [
        fit_shared_imf_two_component_single_model(contexts=contexts, spec=spec)
        for spec in model_specs
    ]
    summary_table = pd.DataFrame([asdict(payload["summary"]) for payload in payloads]).sort_values(
        "bic",
        ascending=True,
    ).reset_index(drop=True)
    if summary_table.empty:
        raise RuntimeError("No shared-IMF two-component joint fits were produced.")
    best_bic = float(summary_table["bic"].min())
    summary_table["delta_bic"] = summary_table["bic"] - best_bic
    for payload in payloads:
        payload["summary"].delta_bic = float(payload["summary"].bic - best_bic)
    best_payload = min(payloads, key=lambda payload: payload["summary"].bic)
    best_component_summary_table = build_shared_best_component_summary_table(
        best_payload=best_payload,
        n_clusters_by_component={label: len(subset) for label, subset in subsets.items()},
    )
    best_imf_grid_table = build_shared_best_component_imf_grid_table(
        best_payload=best_payload,
        contexts=contexts,
    )
    best_radial_grid_table = build_shared_best_component_radial_grid_table(
        best_payload=best_payload,
        contexts=contexts,
    )
    catalog_prediction_table = build_shared_best_component_catalog_prediction_table(
        subsets=subsets,
        contexts=contexts,
        best_payload=best_payload,
    )

    outputs_tables = project_root / "outputs" / "tables"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    summary_table.to_csv(
        outputs_tables / "joint_fixed_survival_shared_imf_two_component_model_summary.csv",
        index=False,
    )
    best_component_summary_table.to_csv(
        outputs_tables / "joint_fixed_survival_shared_imf_two_component_best_component_summary.csv",
        index=False,
    )
    best_imf_grid_table.to_csv(
        outputs_tables / "joint_fixed_survival_shared_imf_two_component_best_imf_grids.csv",
        index=False,
    )
    best_radial_grid_table.to_csv(
        outputs_tables / "joint_fixed_survival_shared_imf_two_component_best_radial_grids.csv",
        index=False,
    )
    catalog_prediction_table.to_csv(
        outputs_tables / "joint_fixed_survival_shared_imf_two_component_catalog_predictions.csv",
        index=False,
    )

    detailed_summary = {
        "selection_offset_dex": selection_offset_dex,
        "survival_grid_bandwidth_log10_a_dex": survival_grid["bandwidth_log10_a_dex"],
        "n_clusters_total": int(len(working)),
        "n_clusters_in_situ": int(len(subsets["in_situ"])),
        "n_clusters_accreted": int(len(subsets["accreted"])),
        "best_joint_model": asdict(best_payload["summary"]),
        "best_component_models": best_component_summary_table.to_dict(orient="records"),
        "all_joint_models_ranked": summary_table.to_dict(orient="records"),
    }
    (outputs_tables / "joint_fixed_survival_shared_imf_two_component_model_summary.json").write_text(
        json.dumps(detailed_summary, indent=2)
    )
    return {
        "summary_table": summary_table,
        "best_component_summary_table": best_component_summary_table,
        "best_imf_grid_table": best_imf_grid_table,
        "best_radial_grid_table": best_radial_grid_table,
        "catalog_prediction_table": catalog_prediction_table,
        "best_payload": best_payload,
        "contexts": contexts,
        "subsets": subsets,
        "survival_grid": survival_grid,
    }


def build_population_model_class_comparison(
    joint_results: dict[str, object],
    shared_two_component_results: dict[str, object],
    separate_two_component_results: dict[str, object],
    project_root: Path,
) -> pd.DataFrame:
    single_best = joint_results["summary_table"].iloc[0]
    shared_best = shared_two_component_results["summary_table"].iloc[0]
    separate_best = separate_two_component_results["pair_summary_table"].iloc[0]

    rows = [
        {
            "model_class": "single_population",
            "description": f"{single_best['imf_family']} + {single_best['radial_model']}",
            "log_likelihood": float(single_best["log_likelihood"]),
            "aic": float(single_best["aic"]),
            "bic": float(single_best["bic"]),
            "n_parameters": int(single_best["n_parameters"]),
        },
        {
            "model_class": "two_component_shared_imf",
            "description": (
                f"shared {shared_best['imf_family']}; "
                f"in-situ {shared_best['in_situ_radial_model']}; "
                f"accreted {shared_best['accreted_radial_model']}"
            ),
            "log_likelihood": float(shared_best["log_likelihood"]),
            "aic": float(shared_best["aic"]),
            "bic": float(shared_best["bic"]),
            "n_parameters": int(shared_best["n_parameters"]),
        },
        {
            "model_class": "two_component_separate_imf",
            "description": (
                f"in-situ {separate_best['in_situ_imf_family']} + {separate_best['in_situ_radial_model']}; "
                f"accreted {separate_best['accreted_imf_family']} + {separate_best['accreted_radial_model']}"
            ),
            "log_likelihood": float(separate_best["log_likelihood"]),
            "aic": float(separate_best["aic"]),
            "bic": float(separate_best["bic"]),
            "n_parameters": int(separate_best["n_parameters"]),
        },
    ]
    comparison = pd.DataFrame(rows).sort_values("bic", ascending=True).reset_index(drop=True)
    comparison["delta_bic"] = comparison["bic"] - float(comparison["bic"].min())
    comparison["delta_aic"] = comparison["aic"] - float(comparison["aic"].min())

    outputs_tables = project_root / "outputs" / "tables"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(
        outputs_tables / "joint_fixed_survival_population_model_class_comparison.csv",
        index=False,
    )
    (outputs_tables / "joint_fixed_survival_population_model_class_comparison.json").write_text(
        json.dumps(comparison.to_dict(orient="records"), indent=2)
    )
    return comparison


def prepare_two_component_contexts(
    catalog: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
    float,
    dict[str, object],
    dict[str, JointLikelihoodContext],
]:
    working = catalog.copy()
    required_columns = {
        "log_initial_mass_msun",
        "semi_major_axis_kpc",
        "log_survival_mass_cut_msun",
        "origin_flag",
    }
    missing = required_columns.difference(working.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(
            f"Catalog is missing required columns for two-component fitting: {missing_list}"
        )

    working["origin_flag"] = pd.to_numeric(working["origin_flag"], errors="coerce").astype("Int64")
    working["origin_label"] = working.get("origin_label", pd.Series(pd.NA, index=working.index, dtype="string"))
    subsets = {
        "in_situ": working.loc[working["origin_flag"] == 1].copy(),
        "accreted": working.loc[working["origin_flag"] == 0].copy(),
    }
    for component_label, subset in subsets.items():
        if subset.empty:
            raise ValueError(f"No clusters found for component {component_label!r}.")
        subset["origin_label"] = component_label

    selection_offset_dex = calibrate_fixed_selection_offset_dex(working)
    survival_grid = build_fixed_survival_grid(working, selection_offset_dex=selection_offset_dex)
    contexts = {
        component_label: JointLikelihoodContext.from_catalog_and_survival_grid(subset, survival_grid)
        for component_label, subset in subsets.items()
    }
    return working, subsets, selection_offset_dex, survival_grid, contexts


def build_shared_imf_two_component_specs(
    imf_families: list[str],
    radial_models: list[str],
) -> list[SharedImfTwoComponentSpec]:
    return [
        SharedImfTwoComponentSpec(
            imf_family=imf_family,
            in_situ_radial_model=in_situ_radial_model,
            accreted_radial_model=accreted_radial_model,
        )
        for imf_family, in_situ_radial_model, accreted_radial_model in product(
            imf_families,
            radial_models,
            radial_models,
        )
    ]


def build_split_alpha_two_component_specs(
    radial_models: list[str],
) -> list[SplitAlphaTwoComponentSpec]:
    return [
        SplitAlphaTwoComponentSpec(
            in_situ_radial_model=in_situ_radial_model,
            accreted_radial_model=accreted_radial_model,
        )
        for in_situ_radial_model, accreted_radial_model in product(radial_models, radial_models)
    ]


def fit_shared_imf_two_component_single_model(
    contexts: dict[str, JointLikelihoodContext],
    spec: SharedImfTwoComponentSpec,
) -> dict[str, object]:
    starts = shared_imf_initial_parameter_vectors(spec)
    bounds = shared_imf_parameter_bounds(spec)
    best_result = None
    best_value = np.inf
    for start in starts:
        result = optimize_shared_model(
            start=start,
            bounds=bounds,
            contexts=contexts,
            spec=spec,
        )
        if result["fun"] < best_value:
            best_value = float(result["fun"])
            best_result = result
    if best_result is None:
        raise RuntimeError(f"Optimization failed to start for shared IMF spec {spec}")

    model = unpack_shared_imf_two_component_model(best_result["x"], contexts=contexts, spec=spec)
    log_likelihood = shared_full_log_likelihood_from_model(model, contexts)
    n_clusters_in_situ = len(contexts["in_situ"].log_mass_data)
    n_clusters_accreted = len(contexts["accreted"].log_mass_data)
    n_clusters_total = n_clusters_in_situ + n_clusters_accreted
    n_parameters = len(best_result["x"])
    summary = SharedImfTwoComponentJointFitResult(
        imf_family=spec.imf_family,
        in_situ_radial_model=spec.in_situ_radial_model,
        accreted_radial_model=spec.accreted_radial_model,
        log_likelihood=float(log_likelihood),
        aic=float(2 * n_parameters - 2 * log_likelihood),
        bic=float(np.log(n_clusters_total) * n_parameters - 2 * log_likelihood),
        delta_bic=np.nan,
        n_parameters=n_parameters,
        n_clusters_total=n_clusters_total,
        n_clusters_in_situ=n_clusters_in_situ,
        n_clusters_accreted=n_clusters_accreted,
        total_initial_count_in_situ=float(model["total_initial_count"]["in_situ"]),
        total_initial_count_accreted=float(model["total_initial_count"]["accreted"]),
        total_initial_count=float(
            model["total_initial_count"]["in_situ"] + model["total_initial_count"]["accreted"]
        ),
        survival_fraction_in_situ=float(model["survival_fraction"]["in_situ"]),
        survival_fraction_accreted=float(model["survival_fraction"]["accreted"]),
        shared_imf_parameters_json=json.dumps(model["imf_parameters"]),
        in_situ_radial_parameters_json=json.dumps(model["radial_parameters"]["in_situ"]),
        accreted_radial_parameters_json=json.dumps(model["radial_parameters"]["accreted"]),
    )
    return {
        "summary": summary,
        "model": model,
        "spec": spec,
        "raw_parameters": np.asarray(best_result["x"], dtype=float),
        "bounds": bounds,
    }


def fit_split_alpha_two_component_single_model(
    contexts: dict[str, JointLikelihoodContext],
    spec: SplitAlphaTwoComponentSpec,
) -> dict[str, object]:
    starts = split_alpha_initial_parameter_vectors(spec)
    bounds = split_alpha_parameter_bounds(spec)
    best_result = None
    best_value = np.inf
    for start in starts:
        result = optimize_split_alpha_model(
            start=start,
            bounds=bounds,
            contexts=contexts,
            spec=spec,
        )
        if result["fun"] < best_value:
            best_value = float(result["fun"])
            best_result = result
    if best_result is None:
        raise RuntimeError(f"Optimization failed to start for split-alpha spec {spec}")

    model = unpack_split_alpha_two_component_model(best_result["x"], contexts=contexts, spec=spec)
    log_likelihood = split_alpha_full_log_likelihood_from_model(model, contexts)
    n_clusters_in_situ = len(contexts["in_situ"].log_mass_data)
    n_clusters_accreted = len(contexts["accreted"].log_mass_data)
    n_clusters_total = n_clusters_in_situ + n_clusters_accreted
    n_parameters = len(best_result["x"])
    in_situ_imf_parameters = model["imf_parameters"]["in_situ"]
    accreted_imf_parameters = model["imf_parameters"]["accreted"]
    summary = SplitAlphaTwoComponentJointFitResult(
        imf_family="schechter",
        in_situ_radial_model=spec.in_situ_radial_model,
        accreted_radial_model=spec.accreted_radial_model,
        log_likelihood=float(log_likelihood),
        aic=float(2 * n_parameters - 2 * log_likelihood),
        bic=float(np.log(n_clusters_total) * n_parameters - 2 * log_likelihood),
        delta_bic=np.nan,
        n_parameters=n_parameters,
        n_clusters_total=n_clusters_total,
        n_clusters_in_situ=n_clusters_in_situ,
        n_clusters_accreted=n_clusters_accreted,
        total_initial_count_in_situ=float(model["total_initial_count"]["in_situ"]),
        total_initial_count_accreted=float(model["total_initial_count"]["accreted"]),
        total_initial_count=float(
            model["total_initial_count"]["in_situ"] + model["total_initial_count"]["accreted"]
        ),
        survival_fraction_in_situ=float(model["survival_fraction"]["in_situ"]),
        survival_fraction_accreted=float(model["survival_fraction"]["accreted"]),
        shared_log10_m_c_msun=float(model["shared_log10_m_c_msun"]),
        in_situ_alpha_dndm=float(in_situ_imf_parameters["alpha_dndm"]),
        accreted_alpha_dndm=float(accreted_imf_parameters["alpha_dndm"]),
        in_situ_imf_parameters_json=json.dumps(in_situ_imf_parameters),
        accreted_imf_parameters_json=json.dumps(accreted_imf_parameters),
        in_situ_radial_parameters_json=json.dumps(model["radial_parameters"]["in_situ"]),
        accreted_radial_parameters_json=json.dumps(model["radial_parameters"]["accreted"]),
    )
    return {
        "summary": summary,
        "model": model,
        "spec": spec,
        "raw_parameters": np.asarray(best_result["x"], dtype=float),
        "bounds": bounds,
    }


def optimize_shared_model(
    start: np.ndarray,
    bounds: list[tuple[float, float]],
    contexts: dict[str, JointLikelihoodContext],
    spec: SharedImfTwoComponentSpec,
) -> dict[str, object]:
    from scipy import optimize

    result = optimize.minimize(
        lambda params: shared_negative_profile_log_likelihood(params, contexts=contexts, spec=spec),
        x0=np.asarray(start, dtype=float),
        method="L-BFGS-B",
        bounds=bounds,
    )
    return {"x": np.asarray(result.x, dtype=float), "fun": float(result.fun), "success": bool(result.success)}


def optimize_split_alpha_model(
    start: np.ndarray,
    bounds: list[tuple[float, float]],
    contexts: dict[str, JointLikelihoodContext],
    spec: SplitAlphaTwoComponentSpec,
) -> dict[str, object]:
    from scipy import optimize

    result = optimize.minimize(
        lambda params: split_alpha_negative_profile_log_likelihood(params, contexts=contexts, spec=spec),
        x0=np.asarray(start, dtype=float),
        method="L-BFGS-B",
        bounds=bounds,
    )
    return {"x": np.asarray(result.x, dtype=float), "fun": float(result.fun), "success": bool(result.success)}


def shared_imf_parameter_bounds(spec: SharedImfTwoComponentSpec) -> list[tuple[float, float]]:
    imf_prefix = imf_parameter_count(spec.imf_family)
    imf_bounds = parameter_bounds(JointModelSpec(spec.imf_family, "step5"))[:imf_prefix]
    in_situ_bounds = radial_parameter_bounds(spec.in_situ_radial_model)
    accreted_bounds = radial_parameter_bounds(spec.accreted_radial_model)
    return list(imf_bounds) + list(in_situ_bounds) + list(accreted_bounds)


def split_alpha_parameter_bounds(spec: SplitAlphaTwoComponentSpec) -> list[tuple[float, float]]:
    schechter_bounds = parameter_bounds(JointModelSpec("schechter", "step5"))[:2]
    alpha_bounds = schechter_bounds[0]
    log_mc_bounds = schechter_bounds[1]
    in_situ_bounds = radial_parameter_bounds(spec.in_situ_radial_model)
    accreted_bounds = radial_parameter_bounds(spec.accreted_radial_model)
    return [alpha_bounds, alpha_bounds, log_mc_bounds] + list(in_situ_bounds) + list(accreted_bounds)


def radial_parameter_bounds(radial_model: str) -> list[tuple[float, float]]:
    helper_spec = JointModelSpec(imf_family="lognormal", radial_model=radial_model)
    helper_bounds = parameter_bounds(helper_spec)
    return helper_bounds[imf_parameter_count("lognormal") :]


def shared_imf_initial_parameter_vectors(spec: SharedImfTwoComponentSpec) -> list[np.ndarray]:
    starts: list[np.ndarray] = []
    for imf_start in unique_imf_starts(spec.imf_family):
        for in_situ_radial_start in unique_radial_starts(spec.in_situ_radial_model):
            for accreted_radial_start in unique_radial_starts(spec.accreted_radial_model):
                starts.append(np.concatenate([imf_start, in_situ_radial_start, accreted_radial_start]))
    return starts


def split_alpha_initial_parameter_vectors(spec: SplitAlphaTwoComponentSpec) -> list[np.ndarray]:
    starts: list[np.ndarray] = []
    alpha_starts = unique_schechter_alpha_starts()
    log_mc_starts = unique_schechter_log_mc_starts()
    for in_situ_alpha in alpha_starts:
        for accreted_alpha in alpha_starts:
            for log_mc in log_mc_starts:
                for in_situ_radial_start in unique_radial_starts(spec.in_situ_radial_model):
                    for accreted_radial_start in unique_radial_starts(spec.accreted_radial_model):
                        starts.append(
                            np.concatenate(
                                [
                                    np.array([in_situ_alpha, accreted_alpha, log_mc], dtype=float),
                                    in_situ_radial_start,
                                    accreted_radial_start,
                                ]
                            )
                        )
    return starts


def unique_imf_starts(imf_family: str) -> list[np.ndarray]:
    helper_spec = JointModelSpec(imf_family=imf_family, radial_model="step5")
    n_imf = imf_parameter_count(imf_family)
    starts = []
    for start in initial_parameter_vectors(helper_spec):
        candidate = np.asarray(start[:n_imf], dtype=float)
        if not any(np.allclose(candidate, existing) for existing in starts):
            starts.append(candidate)
    return starts


def unique_schechter_alpha_starts() -> list[float]:
    alpha_starts: list[float] = []
    for start in unique_imf_starts("schechter"):
        alpha_value = float(start[0])
        if not any(np.isclose(alpha_value, existing) for existing in alpha_starts):
            alpha_starts.append(alpha_value)
    return alpha_starts


def unique_schechter_log_mc_starts() -> list[float]:
    log_mc_starts: list[float] = []
    for start in unique_imf_starts("schechter"):
        log_mc_value = float(start[1])
        if not any(np.isclose(log_mc_value, existing) for existing in log_mc_starts):
            log_mc_starts.append(log_mc_value)
    return log_mc_starts


def unique_radial_starts(radial_model: str) -> list[np.ndarray]:
    helper_spec = JointModelSpec(imf_family="lognormal", radial_model=radial_model)
    offset = imf_parameter_count("lognormal")
    starts = []
    for start in initial_parameter_vectors(helper_spec):
        candidate = np.asarray(start[offset:], dtype=float)
        if not any(np.allclose(candidate, existing) for existing in starts):
            starts.append(candidate)
    return starts


def shared_negative_profile_log_likelihood(
    params: np.ndarray,
    contexts: dict[str, JointLikelihoodContext],
    spec: SharedImfTwoComponentSpec,
) -> float:
    model = unpack_shared_imf_two_component_model(params, contexts=contexts, spec=spec)
    if any(np.any(values <= 0.0) for values in model["imf_density_data"].values()):
        return 1.0e30
    if any(np.any(values <= 0.0) for values in model["radial_density_data"].values()):
        return 1.0e30
    if any(value <= 0.0 for value in model["survival_fraction"].values()):
        return 1.0e30

    profile_log_like = 0.0
    for component_label, context in contexts.items():
        profile_log_like += np.sum(np.log(model["imf_density_data"][component_label]))
        profile_log_like += np.sum(np.log(model["radial_density_data"][component_label]))
        profile_log_like -= len(context.log_mass_data) * np.log(model["survival_fraction"][component_label])
    return float(-profile_log_like)


def split_alpha_negative_profile_log_likelihood(
    params: np.ndarray,
    contexts: dict[str, JointLikelihoodContext],
    spec: SplitAlphaTwoComponentSpec,
) -> float:
    model = unpack_split_alpha_two_component_model(params, contexts=contexts, spec=spec)
    if any(np.any(values <= 0.0) for values in model["imf_density_data"].values()):
        return 1.0e30
    if any(np.any(values <= 0.0) for values in model["radial_density_data"].values()):
        return 1.0e30
    if any(value <= 0.0 for value in model["survival_fraction"].values()):
        return 1.0e30

    profile_log_like = 0.0
    for component_label, context in contexts.items():
        profile_log_like += np.sum(np.log(model["imf_density_data"][component_label]))
        profile_log_like += np.sum(np.log(model["radial_density_data"][component_label]))
        profile_log_like -= len(context.log_mass_data) * np.log(model["survival_fraction"][component_label])
    return float(-profile_log_like)


def unpack_shared_imf_two_component_model(
    params: np.ndarray,
    contexts: dict[str, JointLikelihoodContext],
    spec: SharedImfTwoComponentSpec,
) -> dict[str, object]:
    n_imf = imf_parameter_count(spec.imf_family)
    n_radial_in_situ = len(radial_parameter_bounds(spec.in_situ_radial_model))
    imf_params = np.asarray(params[:n_imf], dtype=float)
    in_situ_radial_params = np.asarray(params[n_imf : n_imf + n_radial_in_situ], dtype=float)
    accreted_radial_params = np.asarray(params[n_imf + n_radial_in_situ :], dtype=float)

    reference_context = contexts["in_situ"]
    imf_density_grid, _, imf_parameters = evaluate_imf_family(
        spec.imf_family,
        imf_params,
        reference_context.log_mass_grid,
        reference_context.log_mass_data,
    )
    imf_density_data = {
        component_label: evaluate_imf_family(
            spec.imf_family,
            imf_params,
            context.log_mass_grid,
            context.log_mass_data,
        )[1]
        for component_label, context in contexts.items()
    }
    radial_density_grid_in_situ, radial_density_data_in_situ, radial_parameters_in_situ = evaluate_radial_model(
        spec.in_situ_radial_model,
        in_situ_radial_params,
        contexts["in_situ"],
    )
    radial_density_grid_accreted, radial_density_data_accreted, radial_parameters_accreted = evaluate_radial_model(
        spec.accreted_radial_model,
        accreted_radial_params,
        contexts["accreted"],
    )
    radial_density_grid = {
        "in_situ": radial_density_grid_in_situ,
        "accreted": radial_density_grid_accreted,
    }
    radial_density_data = {
        "in_situ": radial_density_data_in_situ,
        "accreted": radial_density_data_accreted,
    }
    radial_parameters = {
        "in_situ": radial_parameters_in_situ,
        "accreted": radial_parameters_accreted,
    }
    raw_survival_fraction = {
        component_label: integrate_survival_fraction(
            imf_density_grid,
            radial_density_grid[component_label],
            contexts[component_label].log_mass_grid,
            contexts[component_label].log_a_grid,
            contexts[component_label].survival_probability_grid,
        )
        for component_label in contexts
    }
    selection_fraction = {
        component_label: integrate_survival_fraction(
            imf_density_grid,
            radial_density_grid[component_label],
            contexts[component_label].log_mass_grid,
            contexts[component_label].log_a_grid,
            contexts[component_label].selection_probability_grid,
        )
        for component_label in contexts
    }
    total_initial_count = {
        component_label: len(contexts[component_label].log_mass_data) / selection_fraction[component_label]
        for component_label in contexts
    }
    return {
        "spec": spec,
        "imf_density_grid": imf_density_grid,
        "imf_density_data": imf_density_data,
        "radial_density_grid": radial_density_grid,
        "radial_density_data": radial_density_data,
        "survival_fraction": selection_fraction,
        "selection_fraction": selection_fraction,
        "raw_survival_fraction": raw_survival_fraction,
        "total_initial_count": total_initial_count,
        "imf_parameters": imf_parameters,
        "radial_parameters": radial_parameters,
    }


def unpack_split_alpha_two_component_model(
    params: np.ndarray,
    contexts: dict[str, JointLikelihoodContext],
    spec: SplitAlphaTwoComponentSpec,
) -> dict[str, object]:
    alpha_in_situ = float(params[0])
    alpha_accreted = float(params[1])
    shared_log_m_c = float(params[2])
    n_radial_in_situ = len(radial_parameter_bounds(spec.in_situ_radial_model))
    in_situ_radial_params = np.asarray(params[3 : 3 + n_radial_in_situ], dtype=float)
    accreted_radial_params = np.asarray(params[3 + n_radial_in_situ :], dtype=float)

    in_situ_imf_density_grid, in_situ_imf_density_data, in_situ_imf_parameters = evaluate_imf_family(
        "schechter",
        np.array([alpha_in_situ, shared_log_m_c], dtype=float),
        contexts["in_situ"].log_mass_grid,
        contexts["in_situ"].log_mass_data,
    )
    accreted_imf_density_grid, accreted_imf_density_data, accreted_imf_parameters = evaluate_imf_family(
        "schechter",
        np.array([alpha_accreted, shared_log_m_c], dtype=float),
        contexts["accreted"].log_mass_grid,
        contexts["accreted"].log_mass_data,
    )
    imf_density_grid = {
        "in_situ": in_situ_imf_density_grid,
        "accreted": accreted_imf_density_grid,
    }
    imf_density_data = {
        "in_situ": in_situ_imf_density_data,
        "accreted": accreted_imf_density_data,
    }

    radial_density_grid_in_situ, radial_density_data_in_situ, radial_parameters_in_situ = evaluate_radial_model(
        spec.in_situ_radial_model,
        in_situ_radial_params,
        contexts["in_situ"],
    )
    radial_density_grid_accreted, radial_density_data_accreted, radial_parameters_accreted = evaluate_radial_model(
        spec.accreted_radial_model,
        accreted_radial_params,
        contexts["accreted"],
    )
    radial_density_grid = {
        "in_situ": radial_density_grid_in_situ,
        "accreted": radial_density_grid_accreted,
    }
    radial_density_data = {
        "in_situ": radial_density_data_in_situ,
        "accreted": radial_density_data_accreted,
    }
    radial_parameters = {
        "in_situ": radial_parameters_in_situ,
        "accreted": radial_parameters_accreted,
    }
    raw_survival_fraction = {
        component_label: integrate_survival_fraction(
            imf_density_grid[component_label],
            radial_density_grid[component_label],
            contexts[component_label].log_mass_grid,
            contexts[component_label].log_a_grid,
            contexts[component_label].survival_probability_grid,
        )
        for component_label in contexts
    }
    selection_fraction = {
        component_label: integrate_survival_fraction(
            imf_density_grid[component_label],
            radial_density_grid[component_label],
            contexts[component_label].log_mass_grid,
            contexts[component_label].log_a_grid,
            contexts[component_label].selection_probability_grid,
        )
        for component_label in contexts
    }
    total_initial_count = {
        component_label: len(contexts[component_label].log_mass_data) / selection_fraction[component_label]
        for component_label in contexts
    }
    return {
        "spec": spec,
        "shared_log10_m_c_msun": shared_log_m_c,
        "imf_density_grid": imf_density_grid,
        "imf_density_data": imf_density_data,
        "radial_density_grid": radial_density_grid,
        "radial_density_data": radial_density_data,
        "survival_fraction": selection_fraction,
        "selection_fraction": selection_fraction,
        "raw_survival_fraction": raw_survival_fraction,
        "total_initial_count": total_initial_count,
        "imf_parameters": {
            "in_situ": in_situ_imf_parameters,
            "accreted": accreted_imf_parameters,
        },
        "radial_parameters": radial_parameters,
    }


def shared_full_log_likelihood_from_model(
    model: dict[str, object],
    contexts: dict[str, JointLikelihoodContext],
) -> float:
    total = 0.0
    for component_label, context in contexts.items():
        selection_data = np.clip(
            context.selection_interpolator(np.column_stack([context.log_mass_data, context.log_a_data])),
            1.0e-12,
            1.0,
        )
        total_initial_count = float(model["total_initial_count"][component_label])
        total += (
            len(context.log_mass_data) * np.log(total_initial_count)
            - total_initial_count * model["survival_fraction"][component_label]
            + np.sum(np.log(np.clip(model["imf_density_data"][component_label], 1.0e-12, None)))
            + np.sum(np.log(np.clip(model["radial_density_data"][component_label], 1.0e-12, None)))
            + np.sum(np.log(selection_data))
        )
    return float(total)


def split_alpha_full_log_likelihood_from_model(
    model: dict[str, object],
    contexts: dict[str, JointLikelihoodContext],
) -> float:
    total = 0.0
    for component_label, context in contexts.items():
        selection_data = np.clip(
            context.selection_interpolator(np.column_stack([context.log_mass_data, context.log_a_data])),
            1.0e-12,
            1.0,
        )
        total_initial_count = float(model["total_initial_count"][component_label])
        total += (
            len(context.log_mass_data) * np.log(total_initial_count)
            - total_initial_count * model["survival_fraction"][component_label]
            + np.sum(np.log(np.clip(model["imf_density_data"][component_label], 1.0e-12, None)))
            + np.sum(np.log(np.clip(model["radial_density_data"][component_label], 1.0e-12, None)))
            + np.sum(np.log(selection_data))
        )
    return float(total)


def build_shared_best_component_summary_table(
    best_payload: dict[str, object],
    n_clusters_by_component: dict[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    summary = best_payload["summary"]
    for component_label in ("accreted", "in_situ"):
        radial_model = (
            summary.accreted_radial_model if component_label == "accreted" else summary.in_situ_radial_model
        )
        rows.append(
            {
                "component_label": component_label,
                "n_clusters_component": int(n_clusters_by_component[component_label]),
                "imf_family": summary.imf_family,
                "radial_model": radial_model,
                "log_likelihood": float(summary.log_likelihood),
                "aic": float(summary.aic),
                "bic": float(summary.bic),
                "n_parameters": int(summary.n_parameters),
                "total_initial_count": float(best_payload["model"]["total_initial_count"][component_label]),
                "survival_fraction": float(best_payload["model"]["survival_fraction"][component_label]),
                "shared_imf_parameters_json": summary.shared_imf_parameters_json,
                "radial_parameters_json": json.dumps(best_payload["model"]["radial_parameters"][component_label]),
            }
        )
    return pd.DataFrame(rows).sort_values("component_label").reset_index(drop=True)


def split_alpha_component_payloads_from_best_payload(
    best_payload: dict[str, object],
) -> dict[str, dict[str, object]]:
    component_payloads: dict[str, dict[str, object]] = {}
    for component_label in ("in_situ", "accreted"):
        radial_model = (
            best_payload["spec"].in_situ_radial_model
            if component_label == "in_situ"
            else best_payload["spec"].accreted_radial_model
        )
        component_payloads[component_label] = {
            "spec": JointModelSpec(imf_family="schechter", radial_model=radial_model),
            "model": {
                "imf_density_grid": best_payload["model"]["imf_density_grid"][component_label],
                "imf_density_data": best_payload["model"]["imf_density_data"][component_label],
                "radial_density_grid": best_payload["model"]["radial_density_grid"][component_label],
                "radial_density_data": best_payload["model"]["radial_density_data"][component_label],
                "selection_fraction": best_payload["model"]["selection_fraction"][component_label],
                "raw_survival_fraction": best_payload["model"]["raw_survival_fraction"][component_label],
                "survival_fraction": best_payload["model"]["survival_fraction"][component_label],
                "total_initial_count": best_payload["model"]["total_initial_count"][component_label],
            },
        }
    return component_payloads


def build_split_alpha_best_component_summary_table(
    best_payload: dict[str, object],
    n_clusters_by_component: dict[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    summary = best_payload["summary"]
    for component_label in ("accreted", "in_situ"):
        radial_model = (
            summary.accreted_radial_model if component_label == "accreted" else summary.in_situ_radial_model
        )
        rows.append(
            {
                "component_label": component_label,
                "n_clusters_component": int(n_clusters_by_component[component_label]),
                "imf_family": summary.imf_family,
                "radial_model": radial_model,
                "log_likelihood": float(summary.log_likelihood),
                "aic": float(summary.aic),
                "bic": float(summary.bic),
                "n_parameters": int(summary.n_parameters),
                "total_initial_count": float(best_payload["model"]["total_initial_count"][component_label]),
                "survival_fraction": float(best_payload["model"]["survival_fraction"][component_label]),
                "shared_log10_m_c_msun": float(summary.shared_log10_m_c_msun),
                "alpha_dndm": float(best_payload["model"]["imf_parameters"][component_label]["alpha_dndm"]),
                "imf_parameters_json": json.dumps(best_payload["model"]["imf_parameters"][component_label]),
                "radial_parameters_json": json.dumps(best_payload["model"]["radial_parameters"][component_label]),
            }
        )
    return pd.DataFrame(rows).sort_values("component_label").reset_index(drop=True)


def build_split_alpha_best_component_imf_grid_table(
    best_payload: dict[str, object],
    contexts: dict[str, JointLikelihoodContext],
) -> pd.DataFrame:
    component_payloads = split_alpha_component_payloads_from_best_payload(best_payload)
    return build_best_component_imf_grid_table(component_payloads=component_payloads, contexts=contexts)


def build_split_alpha_best_component_radial_grid_table(
    best_payload: dict[str, object],
    contexts: dict[str, JointLikelihoodContext],
) -> pd.DataFrame:
    component_payloads = split_alpha_component_payloads_from_best_payload(best_payload)
    return build_best_component_radial_grid_table(component_payloads=component_payloads, contexts=contexts)


def build_split_alpha_best_component_catalog_prediction_table(
    subsets: dict[str, pd.DataFrame],
    contexts: dict[str, JointLikelihoodContext],
    best_payload: dict[str, object],
) -> pd.DataFrame:
    component_payloads = split_alpha_component_payloads_from_best_payload(best_payload)
    return build_best_component_catalog_prediction_table(
        subsets=subsets,
        contexts=contexts,
        component_payloads=component_payloads,
    )


def build_shared_best_component_imf_grid_table(
    best_payload: dict[str, object],
    contexts: dict[str, JointLikelihoodContext],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    spec = JointModelSpec(
        imf_family=best_payload["spec"].imf_family,
        radial_model=best_payload["spec"].in_situ_radial_model,
    )
    base_rows = make_imf_grid_rows(
        {
            "imf_density_grid": best_payload["model"]["imf_density_grid"],
            "total_initial_count": 1.0,
        },
        contexts["in_situ"],
        spec,
    )
    for component_label in ("in_situ", "accreted"):
        for row in base_rows:
            copy_row = dict(row)
            copy_row["component_label"] = component_label
            rows.append(copy_row)
    return pd.DataFrame(rows)


def build_shared_best_component_radial_grid_table(
    best_payload: dict[str, object],
    contexts: dict[str, JointLikelihoodContext],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for component_label in ("in_situ", "accreted"):
        radial_model = (
            best_payload["spec"].in_situ_radial_model
            if component_label == "in_situ"
            else best_payload["spec"].accreted_radial_model
        )
        spec = JointModelSpec(
            imf_family=best_payload["spec"].imf_family,
            radial_model=radial_model,
        )
        component_rows = make_radial_grid_rows(
            {
                "radial_density_grid": best_payload["model"]["radial_density_grid"][component_label],
                "total_initial_count": best_payload["model"]["total_initial_count"][component_label],
            },
            contexts[component_label],
            spec,
        )
        for row in component_rows:
            row["component_label"] = component_label
        rows.extend(component_rows)
    return pd.DataFrame(rows)


def build_shared_best_component_catalog_prediction_table(
    subsets: dict[str, pd.DataFrame],
    contexts: dict[str, JointLikelihoodContext],
    best_payload: dict[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for component_label, subset in subsets.items():
        context = contexts[component_label]
        selection_data = np.clip(
            context.selection_interpolator(np.column_stack([context.log_mass_data, context.log_a_data])),
            1.0e-12,
            1.0,
        )
        imf_density_data = np.clip(best_payload["model"]["imf_density_data"][component_label], 1.0e-12, None)
        radial_density_data = np.clip(best_payload["model"]["radial_density_data"][component_label], 1.0e-12, None)
        total_initial_count = float(best_payload["model"]["total_initial_count"][component_label])
        data_intensity = total_initial_count * imf_density_data * radial_density_data * selection_data
        for index in range(len(context.log_mass_data)):
            rows.append(
                {
                    "cluster_label": subset.iloc[index].get("cluster_label", index),
                    "component_label": component_label,
                    "origin_flag": int(subset.iloc[index]["origin_flag"]),
                    "log_initial_mass_msun": float(context.log_mass_data[index]),
                    "semi_major_axis_kpc": float(np.power(10.0, context.log_a_data[index])),
                    "log10_semi_major_axis_kpc": float(context.log_a_data[index]),
                    "survival_probability_fixed": float(selection_data[index]),
                    "imf_density_best": float(imf_density_data[index]),
                    "radial_density_best_per_dex_a": float(radial_density_data[index]),
                    "observed_intensity_best": float(data_intensity[index]),
                    "best_imf_family": best_payload["spec"].imf_family,
                    "best_radial_model": (
                        best_payload["spec"].in_situ_radial_model
                        if component_label == "in_situ"
                        else best_payload["spec"].accreted_radial_model
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_component_summary_table(
    component_payloads: dict[str, list[dict[str, object]]],
    n_clusters_by_component: dict[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for component_label, payloads in component_payloads.items():
        component_table = pd.DataFrame([asdict(payload["summary"]) for payload in payloads]).sort_values(
            "bic",
            ascending=True,
        )
        best_bic = float(component_table["bic"].min())
        for payload in payloads:
            row = asdict(payload["summary"])
            row["component_label"] = component_label
            row["n_clusters_component"] = int(n_clusters_by_component[component_label])
            row["delta_bic_component"] = float(row["bic"] - best_bic)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["component_label", "bic", "log_likelihood"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def build_two_component_pair_payloads(
    in_situ_payloads: list[dict[str, object]],
    accreted_payloads: list[dict[str, object]],
    n_clusters_in_situ: int,
    n_clusters_accreted: int,
) -> list[dict[str, object]]:
    n_clusters_total = n_clusters_in_situ + n_clusters_accreted
    pair_payloads: list[dict[str, object]] = []

    for in_situ_payload, accreted_payload in product(in_situ_payloads, accreted_payloads):
        in_summary = in_situ_payload["summary"]
        acc_summary = accreted_payload["summary"]
        n_parameters = int(in_summary.n_parameters + acc_summary.n_parameters)
        log_likelihood = float(in_summary.log_likelihood + acc_summary.log_likelihood)
        aic = float(2 * n_parameters - 2 * log_likelihood)
        bic = float(np.log(n_clusters_total) * n_parameters - 2 * log_likelihood)
        total_initial_count_in_situ = float(in_summary.total_initial_count)
        total_initial_count_accreted = float(acc_summary.total_initial_count)
        summary = TwoComponentJointFitResult(
            in_situ_imf_family=in_summary.imf_family,
            in_situ_radial_model=in_summary.radial_model,
            accreted_imf_family=acc_summary.imf_family,
            accreted_radial_model=acc_summary.radial_model,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            delta_bic=np.nan,
            n_parameters=n_parameters,
            n_clusters_total=n_clusters_total,
            n_clusters_in_situ=n_clusters_in_situ,
            n_clusters_accreted=n_clusters_accreted,
            total_initial_count_in_situ=total_initial_count_in_situ,
            total_initial_count_accreted=total_initial_count_accreted,
            total_initial_count=total_initial_count_in_situ + total_initial_count_accreted,
            survival_fraction_in_situ=float(in_summary.survival_fraction),
            survival_fraction_accreted=float(acc_summary.survival_fraction),
        )
        pair_payloads.append(
            {
                "summary": summary,
                "component_payloads": {
                    "in_situ": in_situ_payload,
                    "accreted": accreted_payload,
                },
            }
        )

    if pair_payloads:
        best_bic = min(payload["summary"].bic for payload in pair_payloads)
        for payload in pair_payloads:
            payload["summary"].delta_bic = float(payload["summary"].bic - best_bic)
    return pair_payloads


def best_pair_component_row(
    component_label: str,
    payload: dict[str, object],
    n_clusters_component: int,
) -> dict[str, object]:
    summary = payload["summary"]
    return {
        "component_label": component_label,
        "n_clusters_component": int(n_clusters_component),
        "imf_family": summary.imf_family,
        "radial_model": summary.radial_model,
        "log_likelihood": float(summary.log_likelihood),
        "aic": float(summary.aic),
        "bic": float(summary.bic),
        "n_parameters": int(summary.n_parameters),
        "total_initial_count": float(summary.total_initial_count),
        "survival_fraction": float(summary.survival_fraction),
        "imf_parameters_json": summary.imf_parameters_json,
        "radial_parameters_json": summary.radial_parameters_json,
    }


def build_best_component_imf_grid_table(
    component_payloads: dict[str, dict[str, object]],
    contexts: dict[str, JointLikelihoodContext],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for component_label, payload in component_payloads.items():
        spec = payload["spec"]
        component_rows = make_imf_grid_rows(payload["model"], contexts[component_label], spec)
        for row in component_rows:
            row["component_label"] = component_label
        rows.extend(component_rows)
    return pd.DataFrame(rows)


def build_best_component_radial_grid_table(
    component_payloads: dict[str, dict[str, object]],
    contexts: dict[str, JointLikelihoodContext],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for component_label, payload in component_payloads.items():
        spec = payload["spec"]
        component_rows = make_radial_grid_rows(payload["model"], contexts[component_label], spec)
        for row in component_rows:
            row["component_label"] = component_label
        rows.extend(component_rows)
    return pd.DataFrame(rows)


def build_best_component_catalog_prediction_table(
    subsets: dict[str, pd.DataFrame],
    contexts: dict[str, JointLikelihoodContext],
    component_payloads: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for component_label, subset in subsets.items():
        context = contexts[component_label]
        payload = component_payloads[component_label]
        model = payload["model"]
        selection_data = np.clip(
            context.selection_interpolator(np.column_stack([context.log_mass_data, context.log_a_data])),
            1.0e-12,
            1.0,
        )
        data_intensity = (
            model["total_initial_count"]
            * np.clip(model["imf_density_data"], 1.0e-12, None)
            * np.clip(model["radial_density_data"], 1.0e-12, None)
            * selection_data
        )
        for index in range(len(context.log_mass_data)):
            rows.append(
                {
                    "cluster_label": subset.iloc[index].get("cluster_label", index),
                    "component_label": component_label,
                    "origin_flag": int(subset.iloc[index]["origin_flag"]),
                    "log_initial_mass_msun": float(context.log_mass_data[index]),
                    "semi_major_axis_kpc": float(np.power(10.0, context.log_a_data[index])),
                    "log10_semi_major_axis_kpc": float(context.log_a_data[index]),
                    "survival_probability_fixed": float(selection_data[index]),
                    "imf_density_best": float(model["imf_density_data"][index]),
                    "radial_density_best_per_dex_a": float(model["radial_density_data"][index]),
                    "observed_intensity_best": float(data_intensity[index]),
                    "best_imf_family": payload["spec"].imf_family,
                    "best_radial_model": payload["spec"].radial_model,
                }
            )
    return pd.DataFrame(rows)
