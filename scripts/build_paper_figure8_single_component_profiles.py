from __future__ import annotations

import os
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
        load_precomputed_flexible_imf_overlay,
        plot_single_component_profiles_for_paper,
    )

    catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog.csv"

    catalog = pd.read_csv(catalog_path)
    catalog_results = fit_catalog_models(catalog, project_root)
    fit_catalog = catalog_results["catalog"]
    joint_results = fit_fixed_survival_joint_models(fit_catalog, project_root)
    detectability_comparison = fit_detectability_corrected_single_component_models_with_abs_longitude(
        fit_catalog,
        project_root,
    )
    detectability_result = detectability_comparison["best_result"]
    detectability_uncertainty = estimate_best_model_uncertainty(
        best_payload=detectability_result["final_payload"],
        context=detectability_result["final_context"],
    )
    flexible_imf_overlay = load_precomputed_flexible_imf_overlay(
        project_root=project_root,
        log_mass_grid=detectability_result["final_context"].log_mass_grid,
    )

    output_path = project_root / "paper" / "figures" / "single_component_profiles.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_single_component_profiles_for_paper(
        baseline_joint_results=joint_results,
        detectability_result=detectability_result,
        uncertainty_payload=detectability_uncertainty,
        flexible_imf_overlay=flexible_imf_overlay,
        output_path=output_path,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
