from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(project_root / ".cache"))
    (project_root / ".mplconfig").mkdir(parents=True, exist_ok=True)
    (project_root / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)

    from globular_clusters_imf.detectability_longitude_model import (
        fit_detectability_corrected_single_component_models_with_abs_longitude,
    )
    from globular_clusters_imf.joint_model import estimate_best_model_uncertainty, fit_fixed_survival_joint_models
    from globular_clusters_imf.model import fit_catalog_models
    from globular_clusters_imf.paper_assets import (
        detectability_corrected_single_total_initial_stellar_mass,
        plot_single_component_profiles_for_paper,
    )

    catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog.csv"

    catalog = pd.read_csv(catalog_path)
    prepared_catalog = fit_catalog_models(catalog, project_root)["catalog"]

    baseline_joint_results = fit_fixed_survival_joint_models(prepared_catalog, project_root)
    comparison = fit_detectability_corrected_single_component_models_with_abs_longitude(
        prepared_catalog,
        project_root=project_root,
    )
    best_result = comparison["best_result"]
    uncertainty_payload = estimate_best_model_uncertainty(
        best_payload=best_result["final_payload"],
        context=best_result["final_context"],
    )

    figures_dir = project_root / "outputs" / "figures"
    tables_dir = project_root / "outputs" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    profile_pdf_path = figures_dir / "single_component_profiles_detectability_abs_longitude.pdf"
    profile_png_path = figures_dir / "single_component_profiles_detectability_abs_longitude.png"
    plot_single_component_profiles_for_paper(
        baseline_joint_results=baseline_joint_results,
        detectability_result=best_result,
        uncertainty_payload=uncertainty_payload,
        output_path=profile_pdf_path,
        flexible_imf_overlay=None,
    )
    subprocess.run(
        [
            "pdftoppm",
            "-singlefile",
            "-png",
            str(profile_pdf_path),
            str(profile_png_path.with_suffix("")),
        ],
        check=True,
    )

    baseline_detectability_summary_path = tables_dir / "joint_fixed_survival_detectability_em_summary.json"
    baseline_detectability_summary = json.loads(baseline_detectability_summary_path.read_text())
    key_results_table = pd.read_csv(project_root / "paper" / "tables" / "key_results_summary.csv")
    baseline_key_row = key_results_table.loc[
        key_results_table["model"] == "Detectability-corrected single component"
    ].iloc[0]
    longitude_summary = best_result["summary_payload"]
    comparison_rows = [
        {
            "model_variant": "baseline_detectability",
            "imf_family": baseline_detectability_summary["best_joint_model"]["imf_family"],
            "radial_model": baseline_detectability_summary["best_joint_model"]["radial_model"],
            "alpha_dndm": json.loads(baseline_detectability_summary["best_joint_model"]["imf_parameters_json"]).get(
                "alpha_dndm"
            ),
            "log10_m_c_msun": json.loads(
                baseline_detectability_summary["best_joint_model"]["imf_parameters_json"]
            ).get("log10_m_c_msun"),
            "total_initial_count": float(baseline_detectability_summary["best_joint_model"]["total_initial_count"]),
            "selection_fraction": float(
                baseline_detectability_summary["best_joint_model"]["selection_fraction"]
            ),
            "raw_survival_fraction": float(
                baseline_detectability_summary["best_joint_model"]["raw_survival_fraction"]
            ),
            "mean_detectability": float(
                baseline_detectability_summary["best_model_detectability_summary"]["final_mean_detectability"]
            ),
            "total_initial_stellar_mass_msun": float(baseline_key_row["total_initial_stellar_mass_msun"]),
        },
        {
            "model_variant": "abs_longitude_detectability",
            "imf_family": longitude_summary["final_model"]["imf_family"],
            "radial_model": longitude_summary["final_model"]["radial_model"],
            "alpha_dndm": json.loads(longitude_summary["final_model"]["imf_parameters_json"]).get("alpha_dndm"),
            "log10_m_c_msun": json.loads(longitude_summary["final_model"]["imf_parameters_json"]).get(
                "log10_m_c_msun"
            ),
            "total_initial_count": float(longitude_summary["final_model"]["total_initial_count"]),
            "selection_fraction": float(longitude_summary["final_model"]["selection_fraction"]),
            "raw_survival_fraction": float(longitude_summary["final_model"]["raw_survival_fraction"]),
            "mean_detectability": float(longitude_summary["final_mean_detectability"]),
            "total_initial_stellar_mass_msun": float(
                detectability_corrected_single_total_initial_stellar_mass(best_result)
            ),
        },
    ]
    comparison_table = pd.DataFrame(comparison_rows)
    comparison_table["delta_vs_baseline_total_initial_count"] = (
        comparison_table["total_initial_count"] - float(comparison_table.iloc[0]["total_initial_count"])
    )
    comparison_table["ratio_vs_baseline_total_initial_count"] = (
        comparison_table["total_initial_count"] / float(comparison_table.iloc[0]["total_initial_count"])
    )
    comparison_table.to_csv(
        tables_dir / "joint_fixed_survival_detectability_abs_longitude_em_vs_baseline.csv",
        index=False,
    )

    print(f"Wrote {tables_dir / 'joint_fixed_survival_detectability_abs_longitude_em_model_summary.csv'}")
    print(f"Wrote {tables_dir / 'joint_fixed_survival_detectability_abs_longitude_em_vs_baseline.csv'}")
    print(f"Wrote {profile_pdf_path}")
    print(f"Wrote {profile_png_path}")


if __name__ == "__main__":
    main()
