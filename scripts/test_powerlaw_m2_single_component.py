from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def find_result_by_spec(
    all_results: list[dict[str, object]],
    imf_family: str,
    radial_model: str,
) -> dict[str, object]:
    for result in all_results:
        spec = result["spec"]
        if spec.imf_family == imf_family and spec.radial_model == radial_model:
            return result
    raise KeyError(f"Could not find result for spec ({imf_family}, {radial_model}).")


def summarize_row(
    row: pd.Series,
    best_log_likelihood: float,
    best_bic: float,
) -> dict[str, object]:
    imf_parameters = json.loads(str(row["imf_parameters_json"]))
    radial_parameters = json.loads(str(row["radial_parameters_json"]))
    return {
        "imf_family": str(row["imf_family"]),
        "radial_model": str(row["radial_model"]),
        "log_likelihood": float(row["log_likelihood"]),
        "delta_log_likelihood_vs_best": float(best_log_likelihood - float(row["log_likelihood"])),
        "aic": float(row["aic"]),
        "bic": float(row["bic"]),
        "delta_bic_vs_best": float(float(row["bic"]) - best_bic),
        "n_parameters": int(row["n_parameters"]),
        "total_initial_count": float(row["total_initial_count"]),
        "selection_fraction": float(row["selection_fraction"]),
        "raw_survival_fraction": float(row["raw_survival_fraction"]),
        "mean_detectability": float(row["mean_detectability"]),
        "rms_residual_sigma_2d": float(row["rms_residual_sigma_2d"]),
        "mean_abs_residual_sigma_2d": float(row["mean_abs_residual_sigma_2d"]),
        "imf_parameters": imf_parameters,
        "radial_parameters": radial_parameters,
    }


def plot_imf_comparison(
    output_path: Path,
    schechter_result: dict[str, object],
    free_powerlaw_result: dict[str, object],
    fixed_powerlaw_m2_result: dict[str, object],
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 3.8))

    line_specs = [
        (
            schechter_result,
            "Best overall: Schechter",
            "#d95f02",
            "-",
            2.2,
        ),
        (
            free_powerlaw_result,
            "Best free power law",
            "#1b9e77",
            "--",
            2.0,
        ),
        (
            fixed_powerlaw_m2_result,
            r"Fixed power law: $\alpha=-2$",
            "#7570b3",
            ":",
            2.4,
        ),
    ]

    for result, label, color, linestyle, linewidth in line_specs:
        model = result["final_payload"]["model"]
        context = result["final_context"]
        birth_imf = np.asarray(model["total_initial_count"], dtype=float) * np.asarray(
            model["imf_density_grid"],
            dtype=float,
        )
        ax.plot(
            np.power(10.0, np.asarray(context.log_mass_grid, dtype=float)),
            birth_imf,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Initial mass $M_{\rm ini}\,[{\rm M}_\odot]$")
    ax.set_ylabel(r"$dN_0 / d\log M_{\rm ini}$")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Single-component IMF comparison")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_root = project_root / "variants" / "single_component_powerlaw_m2_test"
    outputs_tables = output_root / "outputs" / "tables"
    outputs_figures = output_root / "outputs" / "figures"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    outputs_figures.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(project_root / ".cache"))
    (project_root / ".mplconfig").mkdir(parents=True, exist_ok=True)
    (project_root / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)

    from globular_clusters_imf.detectability_longitude_model import (
        fit_detectability_corrected_single_component_models_with_abs_longitude,
        fit_single_component_detectability_em_with_abs_longitude,
    )
    from globular_clusters_imf.detectability_model import build_detectability_corrected_performance_row
    from globular_clusters_imf.joint_model import JointModelSpec, fit_fixed_survival_joint_models, fit_single_joint_model
    from globular_clusters_imf.model import fit_catalog_models

    catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog.csv"

    catalog = pd.read_csv(catalog_path)
    prepared_catalog = fit_catalog_models(catalog, output_root)["catalog"]

    model_specs = [
        JointModelSpec(imf_family="lognormal", radial_model="step5"),
        JointModelSpec(imf_family="powerlaw", radial_model="step5"),
        JointModelSpec(imf_family="powerlaw_m2", radial_model="step5"),
        JointModelSpec(imf_family="schechter", radial_model="step5"),
        JointModelSpec(imf_family="lognormal", radial_model="logpoly3"),
        JointModelSpec(imf_family="powerlaw", radial_model="logpoly3"),
        JointModelSpec(imf_family="powerlaw_m2", radial_model="logpoly3"),
        JointModelSpec(imf_family="schechter", radial_model="logpoly3"),
    ]

    baseline_results = fit_fixed_survival_joint_models(
        prepared_catalog,
        output_root,
        model_specs=model_specs,
    )
    detectability_comparison = fit_detectability_corrected_single_component_models_with_abs_longitude(
        prepared_catalog,
        project_root=output_root,
        model_specs=model_specs,
    )

    detectability_summary = detectability_comparison["summary_table"].copy()
    detectability_summary.to_csv(
        outputs_tables / "single_component_powerlaw_m2_detectability_comparison.csv",
        index=False,
    )
    baseline_results["summary_table"].to_csv(
        outputs_tables / "single_component_powerlaw_m2_fixed_survival_comparison.csv",
        index=False,
    )

    best_row = detectability_summary.iloc[0]
    best_log_likelihood = float(best_row["log_likelihood"])
    best_bic = float(detectability_summary["bic"].min())

    best_free_powerlaw_row = (
        detectability_summary.loc[detectability_summary["imf_family"] == "powerlaw"]
        .sort_values("bic", ascending=True)
        .iloc[0]
    )
    best_fixed_powerlaw_row = (
        detectability_summary.loc[detectability_summary["imf_family"] == "powerlaw_m2"]
        .sort_values("bic", ascending=True)
        .iloc[0]
    )

    best_overall_result = find_result_by_spec(
        detectability_comparison["all_results"],
        imf_family=str(best_row["imf_family"]),
        radial_model=str(best_row["radial_model"]),
    )
    best_fixed_powerlaw_result = find_result_by_spec(
        detectability_comparison["all_results"],
        imf_family=str(best_fixed_powerlaw_row["imf_family"]),
        radial_model=str(best_fixed_powerlaw_row["radial_model"]),
    )

    refined_powerlaw_candidates: list[dict[str, object]] = []
    for radial_model in ("step5", "logpoly3"):
        default_result = find_result_by_spec(
            detectability_comparison["all_results"],
            imf_family="powerlaw",
            radial_model=radial_model,
        )
        fixed_m2_result = find_result_by_spec(
            detectability_comparison["all_results"],
            imf_family="powerlaw_m2",
            radial_model=radial_model,
        )
        refined_powerlaw_candidates.append(
            {
                "result": default_result,
                "start_completeness_source": "default_em_run",
            }
        )
        refined_powerlaw_candidates.append(
            {
                "result": fit_single_component_detectability_em_with_abs_longitude(
                    prepared_catalog,
                    project_root=output_root,
                    spec=JointModelSpec(imf_family="powerlaw", radial_model=radial_model),
                    start_completeness_raw_parameters=fixed_m2_result["final_completeness_raw_parameters"],
                ),
                "start_completeness_source": f"powerlaw_m2_{radial_model}",
            }
        )
        refined_powerlaw_candidates.append(
            {
                "result": fit_single_component_detectability_em_with_abs_longitude(
                    prepared_catalog,
                    project_root=output_root,
                    spec=JointModelSpec(imf_family="powerlaw", radial_model=radial_model),
                    start_completeness_raw_parameters=best_overall_result["final_completeness_raw_parameters"],
                ),
                "start_completeness_source": "best_overall_schechter",
            }
        )

    refined_powerlaw_rows = pd.DataFrame(
        [
            {
                **build_detectability_corrected_performance_row(candidate["result"]),
                "start_completeness_source": candidate["start_completeness_source"],
            }
            for candidate in refined_powerlaw_candidates
        ]
    ).sort_values(["log_likelihood", "bic"], ascending=[False, True]).reset_index(drop=True)
    refined_powerlaw_rows.to_csv(
        outputs_tables / "single_component_powerlaw_refinement_candidates.csv",
        index=False,
    )
    best_refined_powerlaw_row = refined_powerlaw_rows.iloc[0]
    best_free_powerlaw_candidate = max(
        refined_powerlaw_candidates,
        key=lambda candidate: float(candidate["result"]["summary_payload"]["final_model"]["log_likelihood"]),
    )
    best_free_powerlaw_result = best_free_powerlaw_candidate["result"]
    best_free_powerlaw_row = pd.Series(build_detectability_corrected_performance_row(best_free_powerlaw_result))

    free_powerlaw_with_fixed_m2_context_payload = fit_single_joint_model(
        context=best_fixed_powerlaw_result["final_context"],
        spec=JointModelSpec(imf_family="powerlaw", radial_model=str(best_fixed_powerlaw_row["radial_model"])),
    )

    plot_imf_comparison(
        outputs_figures / "single_component_powerlaw_m2_imf_comparison.pdf",
        schechter_result=best_overall_result,
        free_powerlaw_result=best_free_powerlaw_result,
        fixed_powerlaw_m2_result=best_fixed_powerlaw_result,
    )

    summary_payload = {
        "comparison_variant": "single_component_abs_longitude_detectability",
        "best_overall_model": summarize_row(best_row, best_log_likelihood=best_log_likelihood, best_bic=best_bic),
        "best_free_powerlaw_model": summarize_row(
            best_free_powerlaw_row,
            best_log_likelihood=best_log_likelihood,
            best_bic=best_bic,
        ),
        "best_fixed_powerlaw_m2_model": summarize_row(
            best_fixed_powerlaw_row,
            best_log_likelihood=best_log_likelihood,
            best_bic=best_bic,
        ),
        "fixed_powerlaw_m2_vs_free_powerlaw": {
            "delta_log_likelihood": float(
                float(best_fixed_powerlaw_row["log_likelihood"]) - float(best_free_powerlaw_row["log_likelihood"])
            ),
            "delta_bic": float(float(best_fixed_powerlaw_row["bic"]) - float(best_free_powerlaw_row["bic"])),
        },
        "fixed_powerlaw_m2_vs_best_overall": {
            "delta_log_likelihood": float(float(best_fixed_powerlaw_row["log_likelihood"]) - best_log_likelihood),
            "delta_bic": float(float(best_fixed_powerlaw_row["bic"]) - best_bic),
        },
        "nested_sanity_check_free_powerlaw_in_fixed_m2_context": {
            "log_likelihood": float(free_powerlaw_with_fixed_m2_context_payload["summary"].log_likelihood),
            "delta_log_likelihood_vs_fixed_powerlaw_m2": float(
                float(free_powerlaw_with_fixed_m2_context_payload["summary"].log_likelihood)
                - float(best_fixed_powerlaw_row["log_likelihood"])
            ),
            "imf_parameters": free_powerlaw_with_fixed_m2_context_payload["model"]["imf_parameters"],
            "interpretation": (
                "With the powerlaw_m2 detectability solution held fixed, the free power-law improves as expected; "
                "the lower EM-run free-powerlaw result is therefore a detectability-iteration path issue rather "
                "than evidence that alpha=-2 strictly beats all free power laws."
            ),
        },
        "best_refined_free_powerlaw_source": {
            "radial_model": str(best_free_powerlaw_result["spec"].radial_model),
            "started_from_abs_longitude_completeness_of": str(best_free_powerlaw_candidate["start_completeness_source"]),
        },
    }
    (outputs_tables / "single_component_powerlaw_m2_summary.json").write_text(
        json.dumps(summary_payload, indent=2)
    )

    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
