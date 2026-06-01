from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    variant_root = project_root / "variants" / "a_lt_100kpc"
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(project_root / ".cache"))
    (project_root / ".mplconfig").mkdir(parents=True, exist_ok=True)
    (project_root / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)

    from globular_clusters_imf.detectability_model import fit_detectability_corrected_single_component_models
    from globular_clusters_imf.joint_model import estimate_best_model_uncertainty, fit_fixed_survival_joint_models
    from globular_clusters_imf.model import fit_catalog_models
    from globular_clusters_imf.paper_assets import plot_single_component_profiles_for_paper

    catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog.csv"

    catalog = pd.read_csv(catalog_path)
    subset = catalog.loc[catalog["semi_major_axis_kpc"] < 100.0].copy()
    if subset.empty:
        raise RuntimeError("No clusters satisfy semi_major_axis_kpc < 100.")

    (variant_root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (variant_root / "outputs" / "figures").mkdir(parents=True, exist_ok=True)
    subset.to_csv(variant_root / "data" / "processed" / "baumgardt_gc_catalog_a_lt_100kpc.csv", index=False)

    catalog_results = fit_catalog_models(subset, variant_root)
    fit_catalog = catalog_results["catalog"]
    joint_results = fit_fixed_survival_joint_models(fit_catalog, variant_root)
    detectability_comparison = fit_detectability_corrected_single_component_models(fit_catalog, variant_root)
    detectability_result = detectability_comparison["best_result"]
    detectability_uncertainty = estimate_best_model_uncertainty(
        best_payload=detectability_result["final_payload"],
        context=detectability_result["final_context"],
    )

    figure_pdf = variant_root / "outputs" / "figures" / "figure8_single_component_profiles_a_lt_100kpc.pdf"
    figure_png = variant_root / "outputs" / "figures" / "figure8_single_component_profiles_a_lt_100kpc.png"
    plot_single_component_profiles_for_paper(
        baseline_joint_results=joint_results,
        detectability_result=detectability_result,
        uncertainty_payload=detectability_uncertainty,
        output_path=figure_pdf,
    )
    plot_single_component_profiles_for_paper(
        baseline_joint_results=joint_results,
        detectability_result=detectability_result,
        uncertainty_payload=detectability_uncertainty,
        output_path=figure_png,
    )

    best_summary = detectability_comparison["summary_table"].iloc[0].to_dict()
    summary = {
        "selection": "semi_major_axis_kpc < 100",
        "n_input_clusters": int(len(subset)),
        "n_fit_clusters": int(len(fit_catalog)),
        "best_imf_family": str(best_summary["imf_family"]),
        "best_radial_model": str(best_summary["radial_model"]),
        "best_log_likelihood": float(best_summary["log_likelihood"]),
        "best_total_initial_count": float(best_summary["total_initial_count"]),
        "best_selection_fraction": float(best_summary["selection_fraction"]),
        "best_mean_detectability": float(best_summary["mean_detectability"]),
        "figure_pdf": str(figure_pdf),
        "figure_png": str(figure_png),
    }
    (variant_root / "outputs" / "tables").mkdir(parents=True, exist_ok=True)
    (variant_root / "outputs" / "tables" / "subset_a_lt_100kpc_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
