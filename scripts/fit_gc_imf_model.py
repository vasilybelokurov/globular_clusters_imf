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

    from globular_clusters_imf.detectability_model import fit_single_component_detectability_em
    from globular_clusters_imf.joint_model import fit_fixed_survival_joint_models
    from globular_clusters_imf.model import fit_catalog_models
    from globular_clusters_imf.plotting import make_all_figures
    from globular_clusters_imf.two_component_model import (
        build_population_model_class_comparison,
        fit_shared_imf_two_component_fixed_survival_joint_models,
        fit_two_component_fixed_survival_joint_models,
    )

    catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    results = fit_catalog_models(catalog, project_root)
    joint_results = fit_fixed_survival_joint_models(results["catalog"], project_root)
    detectability_results = fit_single_component_detectability_em(
        results["catalog"],
        project_root,
        spec=joint_results["best_payload"]["spec"],
    )
    two_component_results = None
    shared_two_component_results = None
    class_comparison_table = None
    if "origin_flag" in results["catalog"].columns:
        shared_two_component_results = fit_shared_imf_two_component_fixed_survival_joint_models(
            results["catalog"],
            project_root,
        )
        two_component_results = fit_two_component_fixed_survival_joint_models(
            results["catalog"],
            project_root,
        )
        class_comparison_table = build_population_model_class_comparison(
            joint_results=joint_results,
            shared_two_component_results=shared_two_component_results,
            separate_two_component_results=two_component_results,
            project_root=project_root,
        )
    make_all_figures(
        results["catalog"],
        results["lognormal"],
        results["powerlaw"],
        results["radial_patch_summary"],
        results["radial_patch_table"],
        results["survivability_map"],
        joint_results,
        detectability_results,
        two_component_results,
        project_root,
    )
    print("Model fitting complete.")
    print(results["summary"])
    print(joint_results["summary_table"].head().to_string(index=False))
    print(pd.DataFrame([detectability_results["summary_payload"]]).to_string(index=False))
    if shared_two_component_results is not None:
        print(shared_two_component_results["best_component_summary_table"].to_string(index=False))
    if two_component_results is not None:
        print(two_component_results["best_component_summary_table"].to_string(index=False))
        print(two_component_results["pair_summary_table"].head().to_string(index=False))
    if class_comparison_table is not None:
        print(class_comparison_table.to_string(index=False))


if __name__ == "__main__":
    main()
