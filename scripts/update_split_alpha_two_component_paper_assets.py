from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from globular_clusters_imf.detectability_longitude_model import (
    fit_detectability_corrected_single_component_models_with_abs_longitude,
    fit_shared_imf_two_component_detectability_em_models_with_abs_longitude,
    fit_split_alpha_two_component_detectability_em_models_with_abs_longitude,
)
from globular_clusters_imf.joint_model import fit_fixed_survival_joint_models
from globular_clusters_imf.model import fit_catalog_models
from globular_clusters_imf.paper_assets import (
    build_conditional_population_model_table,
    build_key_results_table,
    build_paper_summary_payload,
    plot_two_component_results_for_paper,
    write_key_results_table_tex,
    write_population_class_table_tex,
    write_summary_macros_tex,
)


def load_catalog(project_root: Path) -> pd.DataFrame:
    catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog.csv"
    return pd.read_csv(catalog_path)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    catalog = load_catalog(project_root)
    catalog_results = fit_catalog_models(catalog, project_root)
    fit_catalog = catalog_results["catalog"]
    joint_results = fit_fixed_survival_joint_models(fit_catalog, project_root)
    detectability_comparison = fit_detectability_corrected_single_component_models_with_abs_longitude(
        fit_catalog,
        project_root,
    )
    detectability_results = detectability_comparison["best_result"]
    fixed_detectability_kwargs = {
        "fixed_effective_completeness_grid": detectability_results["final_effective_completeness_grid"],
        "fixed_completeness_bin_grid": detectability_results["final_completeness_bin_grid"],
        "fixed_completeness_raw_parameters": detectability_results["final_completeness_raw_parameters"],
    }
    shared_results = fit_shared_imf_two_component_detectability_em_models_with_abs_longitude(
        fit_catalog,
        project_root,
        **fixed_detectability_kwargs,
    )
    split_alpha_results = fit_split_alpha_two_component_detectability_em_models_with_abs_longitude(
        fit_catalog,
        project_root,
        **fixed_detectability_kwargs,
    )

    paper_dir = project_root / "paper"
    figures_dir = paper_dir / "figures"
    tables_dir = paper_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    plot_two_component_results_for_paper(
        detectability_results=detectability_results,
        shared_results=shared_results,
        split_alpha_results=split_alpha_results,
        output_path=figures_dir / "two_component_results.pdf",
    )
    conditional_class_table = build_conditional_population_model_table(
        detectability_comparison=detectability_comparison,
        shared_results=shared_results,
        split_alpha_results=split_alpha_results,
    )
    key_results_table = build_key_results_table(
        joint_results=joint_results,
        detectability_results=detectability_results,
        shared_results=shared_results,
        split_alpha_results=split_alpha_results,
    )
    conditional_class_table.to_csv(tables_dir / "population_model_class_comparison.csv", index=False)
    key_results_table.to_csv(tables_dir / "key_results_summary.csv", index=False)
    write_population_class_table_tex(conditional_class_table, tables_dir / "population_model_class_comparison.tex")
    write_key_results_table_tex(key_results_table, tables_dir / "key_results_summary.tex")

    summary_payload = build_paper_summary_payload(
        fit_catalog=fit_catalog,
        joint_results=joint_results,
        detectability_results=detectability_results,
        shared_results=shared_results,
        split_alpha_results=split_alpha_results,
        conditional_class_table=conditional_class_table,
    )
    (tables_dir / "paper_results_summary.json").write_text(json.dumps(summary_payload, indent=2))
    write_summary_macros_tex(summary_payload, tables_dir / "paper_numbers.tex")

    shared_best = shared_results["summary_table"].iloc[0]
    split_alpha_best = split_alpha_results["summary_table"].iloc[0]
    print("Updated two-component paper assets.")
    print(
        "Shared IMF best:",
        f"logL={shared_best['log_likelihood']:.6f}",
        f"N0={shared_best['total_initial_count']:.1f}",
    )
    print(
        "Split-alpha best:",
        f"logL={split_alpha_best['log_likelihood']:.6f}",
        f"N0={split_alpha_best['total_initial_count']:.1f}",
        f"alpha_in={split_alpha_best['in_situ_alpha_dndm']:.3f}",
        f"alpha_acc={split_alpha_best['accreted_alpha_dndm']:.3f}",
    )


if __name__ == "__main__":
    main()
