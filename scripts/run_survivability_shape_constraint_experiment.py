from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _plot_boundary_comparison(
    *,
    catalog: pd.DataFrame,
    initial_survival_grid: dict[str, object],
    final_survival_grid: dict[str, object],
    output_path: Path,
) -> None:
    semi_major_axis = np.asarray(catalog["semi_major_axis_kpc"], dtype=float)
    initial_mass = np.asarray(catalog["initial_mass_msun"], dtype=float)
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.scatter(semi_major_axis, initial_mass, s=13, color="black", alpha=0.35, linewidths=0.0)

    log_a_initial = np.asarray(initial_survival_grid["log_a_grid"], dtype=float)
    a_initial = np.power(10.0, log_a_initial)
    for key, color in (
        ("fitted_boundary_10_log10_msun", "#9e9e9e"),
        ("fitted_boundary_50_log10_msun", "#6f6f6f"),
        ("fitted_boundary_90_log10_msun", "#3d3d3d"),
    ):
        ax.plot(
            a_initial,
            np.power(10.0, np.asarray(initial_survival_grid[key], dtype=float)),
            color=color,
            linestyle="dashed",
            linewidth=1.2,
        )

    log_a_final = np.asarray(final_survival_grid["log_a_grid"], dtype=float)
    a_final = np.power(10.0, log_a_final)
    fit_payload = final_survival_grid["fit_payload"]
    for key, color in (
        ("fitted_boundary_10_log10_msun", "#74a9cf"),
        ("fitted_boundary_50_log10_msun", "#2b8cbe"),
        ("fitted_boundary_90_log10_msun", "#045a8d"),
    ):
        ax.plot(
            a_final,
            np.power(10.0, np.asarray(fit_payload[key], dtype=float)),
            color=color,
            linewidth=1.45,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(float(semi_major_axis.min() / 1.15), float(semi_major_axis.max() * 1.15))
    ax.set_ylim(1.0e3, max(3.0e7, float(initial_mass.max() * 1.08)))
    ax.set_xlabel(r"Orbital semimajor axis $a$ [kpc]")
    ax.set_ylabel(r"Mass [$\mathrm{M_\odot}$]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    from globular_clusters_imf.joint_model import JointModelSpec
    from globular_clusters_imf.model import fit_catalog_models
    from globular_clusters_imf.smooth_survivability import build_smooth_survivability_grid
    from globular_clusters_imf.survivability_shape_constraint import (
        _build_iteration_summary,
        run_survivability_shape_constraint_experiment,
    )

    variant_root = PROJECT_ROOT / "variants" / "single_component_survivability_shape_constraint"
    figures_dir = variant_root / "outputs" / "figures"
    tables_dir = variant_root / "outputs" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    fit_catalog = fit_catalog_models(catalog, variant_root)["catalog"]

    initial_survival_grid = build_smooth_survivability_grid(fit_catalog, eta_t=1.0)
    experiment = run_survivability_shape_constraint_experiment(
        catalog=fit_catalog,
        project_root=variant_root,
        initial_survival_grid=initial_survival_grid,
        spec=JointModelSpec(imf_family="schechter", radial_model="logpoly3"),
        outer_iterations=3,
        inner_iterations=6,
        survival_relaxation=0.7,
        detectability_relaxation=0.7,
        n_mass_bins=12,
        n_a_bins=8,
    )

    final_summary = _build_iteration_summary(
        outer_iteration=len(experiment["fit_results"]) + 1,
        fit_result=experiment["final_fit_result"],
        survival_grid=experiment["final_survival_grid"],
    )
    iteration_summary_table = experiment["iteration_summary_table"].copy()
    iteration_summary_table = pd.concat(
        [
            iteration_summary_table,
            pd.DataFrame([{**final_summary.__dict__, "outer_iteration": int(final_summary.outer_iteration)}]),
        ],
        ignore_index=True,
    )
    iteration_summary_table.to_csv(tables_dir / "iteration_summary.csv", index=False)

    _plot_boundary_comparison(
        catalog=fit_catalog,
        initial_survival_grid=initial_survival_grid,
        final_survival_grid=experiment["final_survival_grid"],
        output_path=figures_dir / "survivability_boundary_comparison.png",
    )

    boundary_table = pd.DataFrame(
        {
            "log10_semi_major_axis_kpc": np.asarray(experiment["final_survival_grid"]["log_a_grid"], dtype=float),
            "semi_major_axis_kpc": np.asarray(experiment["final_survival_grid"]["semi_major_axis_grid_kpc"], dtype=float),
            "initial_boundary_10_log10_msun": np.asarray(initial_survival_grid["fitted_boundary_10_log10_msun"], dtype=float),
            "initial_boundary_50_log10_msun": np.asarray(initial_survival_grid["fitted_boundary_50_log10_msun"], dtype=float),
            "initial_boundary_90_log10_msun": np.asarray(initial_survival_grid["fitted_boundary_90_log10_msun"], dtype=float),
            "final_boundary_10_log10_msun": np.asarray(experiment["final_survival_grid"]["fit_payload"]["fitted_boundary_10_log10_msun"], dtype=float),
            "final_boundary_50_log10_msun": np.asarray(experiment["final_survival_grid"]["fit_payload"]["fitted_boundary_50_log10_msun"], dtype=float),
            "final_boundary_90_log10_msun": np.asarray(experiment["final_survival_grid"]["fit_payload"]["fitted_boundary_90_log10_msun"], dtype=float),
        }
    )
    boundary_table.to_csv(tables_dir / "boundary_comparison.csv", index=False)

    initial_fit = experiment["fit_results"][0]
    final_fit = experiment["final_fit_result"]
    summary_payload = {
        "variant": "single_component_survivability_shape_constraint",
        "spec": {"imf_family": "schechter", "radial_model": "logpoly3"},
        "algorithm": {
            "outer_iterations": 3,
            "inner_iterations": 6,
            "survival_relaxation": 0.7,
            "detectability_relaxation": 0.7,
            "n_mass_bins": 12,
            "n_a_bins": 8,
        },
        "initial_fit_summary": initial_fit["summary_payload"],
        "final_fit_summary": final_fit["summary_payload"],
        "initial_model": {
            "log_likelihood": float(initial_fit["final_payload"]["summary"].log_likelihood),
            "imf_parameters": initial_fit["final_payload"]["model"]["imf_parameters"],
            "total_initial_count_above_log10_4": float(initial_fit["summary_payload"]["final_total_initial_count_above_log10_4"]),
            "mean_detectability_above_log10_4": float(initial_fit["summary_payload"]["final_mean_detectability_above_log10_4"]),
        },
        "final_model": {
            "log_likelihood": float(final_fit["final_payload"]["summary"].log_likelihood),
            "imf_parameters": final_fit["final_payload"]["model"]["imf_parameters"],
            "total_initial_count_above_log10_4": float(final_fit["summary_payload"]["final_total_initial_count_above_log10_4"]),
            "mean_detectability_above_log10_4": float(final_fit["summary_payload"]["final_mean_detectability_above_log10_4"]),
            "total_initial_stellar_mass_above_log10_4_msun": float(final_summary.total_initial_stellar_mass_above_log10_4_msun),
        },
        "initial_smooth_survivability_summary": {
            field: getattr(initial_survival_grid["summary"], field)
            for field in initial_survival_grid["summary"].__dataclass_fields__
        },
        "final_shape_constrained_survivability_summary": {
            field: getattr(experiment["final_survival_grid"]["fit_payload"]["summary"], field)
            for field in experiment["final_survival_grid"]["fit_payload"]["summary"].__dataclass_fields__
        },
        "iteration_summaries": iteration_summary_table.to_dict(orient="records"),
    }
    (tables_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2))

    print(figures_dir / "survivability_boundary_comparison.png")
    print(tables_dir / "summary.json")


if __name__ == "__main__":
    main()
