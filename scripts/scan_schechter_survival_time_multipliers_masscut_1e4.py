from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd

from scan_schechter_survival_time_multipliers import (
    _plot_logl_vs_multiplier,
    _plot_properties,
    _row_from_result,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-mass-min", type=float, default=4.0)
    parser.add_argument("--exclude-cluster-label", action="append", default=[])
    parser.add_argument("--output-tag", type=str, default="")
    args = parser.parse_args()
    log_mass_min = float(args.log_mass_min)
    excluded_labels = [str(x) for x in args.exclude_cluster_label]
    threshold_tag = f"log10_{log_mass_min:.2f}".replace(".", "p")
    extra_tag = args.output_tag.strip()
    if not extra_tag and excluded_labels:
        cleaned = []
        for label in excluded_labels:
            tag = ''.join(ch.lower() if ch.isalnum() else '_' for ch in label).strip('_')
            cleaned.append(tag)
        extra_tag = 'exclude_' + '_'.join(cleaned)
    from globular_clusters_imf.detectability_longitude_model import fit_single_component_detectability_em_with_abs_longitude
    from globular_clusters_imf.joint_model import JointModelSpec, imf_parameter_count
    from globular_clusters_imf.model import fit_catalog_models
    from globular_clusters_imf.smooth_survivability import build_smooth_survivability_grid

    variant_name = f"schechter_survival_time_multiplier_scan_masscut_{threshold_tag}"
    if extra_tag:
        variant_name = f"{variant_name}_{extra_tag}"
    output_root = PROJECT_ROOT / "variants" / variant_name
    figures_dir = output_root / "outputs" / "figures"
    tables_dir = output_root / "outputs" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    prepared_catalog_full = fit_catalog_models(catalog, output_root)["catalog"]
    prepared_catalog = prepared_catalog_full.loc[
        prepared_catalog_full["log_initial_mass_msun"] >= log_mass_min
    ].copy()
    if excluded_labels:
        prepared_catalog = prepared_catalog.loc[~prepared_catalog["cluster_label"].astype(str).isin(excluded_labels)].copy()
    prepared_catalog = prepared_catalog.reset_index(drop=True)

    eta_grid = np.linspace(0.1, 3.0, 30)
    specs = [
        JointModelSpec(imf_family="schechter", radial_model="step5"),
        JointModelSpec(imf_family="schechter", radial_model="logpoly3"),
    ]
    prior_state = {
        spec.radial_model: {"completeness": None, "radial": None}
        for spec in specs
    }
    all_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []

    for eta_t in eta_grid:
        smooth_survival = build_smooth_survivability_grid(prepared_catalog_full, eta_t=float(eta_t))
        survival_grid_override = {
            "log_mass_grid": np.asarray(smooth_survival["log_mass_grid"], dtype=float),
            "log_a_grid": np.asarray(smooth_survival["log_a_grid"], dtype=float),
            "semi_major_axis_grid_kpc": np.asarray(smooth_survival["semi_major_axis_grid_kpc"], dtype=float),
            "survival_probability": np.asarray(smooth_survival["survival_probability"], dtype=float),
            "selection_offset_dex": 0.0,
            "bandwidth_log10_a_dex": float(smooth_survival["bandwidth_log10_a_dex"]),
            "smooth_survivability_summary": smooth_survival["summary"],
        }
        eta_rows: list[dict[str, object]] = []
        for spec in specs:
            state = prior_state[spec.radial_model]
            result = fit_single_component_detectability_em_with_abs_longitude(
                prepared_catalog,
                project_root=output_root,
                spec=spec,
                n_iterations=12,
                start_completeness_raw_parameters=state["completeness"],
                start_radial_params=state["radial"],
                survival_grid_override=survival_grid_override,
            )
            n_imf = imf_parameter_count(spec.imf_family)
            prior_state[spec.radial_model] = {
                "completeness": np.asarray(result["final_completeness_raw_parameters"], dtype=float),
                "radial": np.asarray(result["final_payload"]["raw_parameters"][n_imf:], dtype=float),
            }
            row = _row_from_result(
                eta_t=float(eta_t),
                radial_model=spec.radial_model,
                survival_summary=smooth_survival["summary"],
                result=result,
                log_mass_min=log_mass_min,
            )
            row["n_clusters_fitted"] = int(len(prepared_catalog))
            row["n_clusters_total_catalog"] = int(len(prepared_catalog_full))
            row["n_clusters_excluded_below_threshold"] = int((prepared_catalog_full["log_initial_mass_msun"] < log_mass_min).sum())
            row["n_clusters_excluded_by_label"] = int(len(excluded_labels))
            eta_rows.append(row)
            print(
                f"eta_t={eta_t:.3f} radial={spec.radial_model} "
                f"logL={row['log_likelihood']:.3f} alpha={row['alpha_dndm']:.3f} "
                f"logMc={row['log10_m_c_msun']:.3f} N0>1e4={row['final_total_initial_count_above_log10_4']:.1f}"
            )

        eta_table = pd.DataFrame(eta_rows).sort_values("log_likelihood", ascending=False).reset_index(drop=True)
        best_row = dict(eta_table.iloc[0])
        best_row["best_radial_model"] = str(best_row["radial_model"])
        best_rows.append(best_row)
        all_rows.extend(eta_rows)

    all_table = pd.DataFrame(all_rows).sort_values(["eta_t", "log_likelihood"], ascending=[True, False]).reset_index(drop=True)
    best_table = pd.DataFrame(best_rows).sort_values("eta_t").reset_index(drop=True)
    all_table.to_csv(tables_dir / "schechter_all_models_vs_eta_t.csv", index=False)
    best_table.to_csv(tables_dir / "schechter_best_models_vs_eta_t.csv", index=False)

    _plot_logl_vs_multiplier(best_table, figures_dir / "schechter_logl_vs_eta_t.png")
    _plot_properties(best_table, figures_dir / "schechter_properties_vs_eta_t.png")

    excluded = prepared_catalog_full.loc[prepared_catalog_full["log_initial_mass_msun"] < log_mass_min, ["cluster_label", "log_initial_mass_msun"]].copy()
    if len(excluded_labels) > 0:
        excluded_by_label = prepared_catalog_full.loc[prepared_catalog_full["cluster_label"].astype(str).isin(excluded_labels), ["cluster_label", "log_initial_mass_msun"]].copy()
        excluded = pd.concat([excluded, excluded_by_label], ignore_index=True).drop_duplicates().reset_index(drop=True)
    if len(excluded) > 0:
        excluded.to_csv(tables_dir / "excluded_clusters.csv", index=False)

    summary_payload = {
        "eta_grid": eta_grid.tolist(),
        "n_detectability_iterations": 12,
        "model_specs": [{"imf_family": spec.imf_family, "radial_model": spec.radial_model} for spec in specs],
        "log_mass_threshold_log10_msun": log_mass_min,
        "log_mass_threshold_msun": float(10.0 ** log_mass_min),
        "n_clusters_fitted": int(len(prepared_catalog)),
        "n_clusters_total_catalog": int(len(prepared_catalog_full)),
        "n_clusters_excluded_below_threshold": int((prepared_catalog_full["log_initial_mass_msun"] < log_mass_min).sum()),
        "n_clusters_excluded_by_label": int(len(excluded_labels)),
        "excluded_cluster_labels": excluded_labels,
        "best_rows": best_table.to_dict(orient="records"),
        "global_best": json.loads(best_table.sort_values("log_likelihood", ascending=False).iloc[0].to_json()),
    }
    (tables_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2))
    print(figures_dir / "schechter_logl_vs_eta_t.png")
    print(figures_dir / "schechter_properties_vs_eta_t.png")
    print(tables_dir / "summary.json")


if __name__ == "__main__":
    main()
