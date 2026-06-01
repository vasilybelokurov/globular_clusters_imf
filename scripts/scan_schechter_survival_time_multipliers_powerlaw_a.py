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

from scan_schechter_survival_time_multipliers import _plot_logl_vs_multiplier, _plot_properties, _row_from_result


def _plot_step5_vs_powerlaw_a(*, step5_table: pd.DataFrame, powerlaw_table: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.2))

    ax = axes[0, 0]
    ax.plot(step5_table["eta_t"], step5_table["log_likelihood"], color="black", marker="o", label="step5")
    ax.plot(powerlaw_table["eta_t"], powerlaw_table["log_likelihood"], color="#1f78b4", marker="o", label="powerlaw_a")
    ax.set_xlabel(r"$\eta_t$")
    ax.set_ylabel(r"Best Schechter $\log L$")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(step5_table["eta_t"], step5_table["alpha_dndm"], color="black", marker="o")
    ax.plot(powerlaw_table["eta_t"], powerlaw_table["alpha_dndm"], color="#1f78b4", marker="o")
    ax.set_xlabel(r"$\eta_t$")
    ax.set_ylabel(r"$\alpha$")

    ax = axes[1, 0]
    ax.plot(step5_table["eta_t"], step5_table["final_total_initial_count_above_log10_4"], color="black", marker="o")
    ax.plot(powerlaw_table["eta_t"], powerlaw_table["final_total_initial_count_above_log10_4"], color="#1f78b4", marker="o")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\eta_t$")
    ax.set_ylabel(r"$N_0(M_{\rm ini}\geq 10^4 M_\odot)$")

    ax = axes[1, 1]
    ax.plot(step5_table["eta_t"], step5_table["mean_detectability_above_log10_4"], color="black", marker="o")
    ax.plot(powerlaw_table["eta_t"], powerlaw_table["mean_detectability_above_log10_4"], color="#1f78b4", marker="o")
    ax.set_xlabel(r"$\eta_t$")
    ax.set_ylabel(r"$\langle C\rangle$ above $10^4 M_\odot$")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    from globular_clusters_imf.detectability_longitude_model import fit_single_component_detectability_em_with_abs_longitude
    from globular_clusters_imf.joint_model import JointModelSpec, imf_parameter_count
    from globular_clusters_imf.model import fit_catalog_models
    from globular_clusters_imf.smooth_survivability import build_smooth_survivability_grid

    output_root = PROJECT_ROOT / "variants" / "schechter_survival_time_multiplier_scan_powerlaw_a"
    figures_dir = output_root / "outputs" / "figures"
    tables_dir = output_root / "outputs" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    prepared_catalog = fit_catalog_models(catalog, output_root)["catalog"]

    eta_grid = np.linspace(0.1, 3.0, 30)
    spec = JointModelSpec(imf_family="schechter", radial_model="powerlaw_a")
    prior_state = {"completeness": None, "radial": None}
    rows: list[dict[str, object]] = []

    for eta_t in eta_grid:
        smooth_survival = build_smooth_survivability_grid(prepared_catalog, eta_t=float(eta_t))
        survival_grid_override = {
            "log_mass_grid": np.asarray(smooth_survival["log_mass_grid"], dtype=float),
            "log_a_grid": np.asarray(smooth_survival["log_a_grid"], dtype=float),
            "semi_major_axis_grid_kpc": np.asarray(smooth_survival["semi_major_axis_grid_kpc"], dtype=float),
            "survival_probability": np.asarray(smooth_survival["survival_probability"], dtype=float),
            "selection_offset_dex": 0.0,
            "bandwidth_log10_a_dex": float(smooth_survival["bandwidth_log10_a_dex"]),
            "smooth_survivability_summary": smooth_survival["summary"],
        }
        result = fit_single_component_detectability_em_with_abs_longitude(
            prepared_catalog,
            project_root=output_root,
            spec=spec,
            n_iterations=12,
            start_completeness_raw_parameters=prior_state["completeness"],
            start_radial_params=prior_state["radial"],
            survival_grid_override=survival_grid_override,
        )
        n_imf = imf_parameter_count(spec.imf_family)
        prior_state = {
            "completeness": np.asarray(result["final_completeness_raw_parameters"], dtype=float),
            "radial": np.asarray(result["final_payload"]["raw_parameters"][n_imf:], dtype=float),
        }
        row = _row_from_result(
            eta_t=float(eta_t),
            radial_model=spec.radial_model,
            survival_summary=smooth_survival["summary"],
            result=result,
            log_mass_min=4.0,
        )
        radial_params = result["final_payload"]["model"]["radial_parameters"]
        row["beta_log10_a"] = float(radial_params.get("beta_log10_a", np.nan))
        row["gamma_linear_a"] = float(radial_params.get("gamma_linear_a", np.nan))
        rows.append(row)
        print(
            f"eta_t={eta_t:.3f} radial=powerlaw_a "
            f"logL={row['log_likelihood']:.3f} alpha={row['alpha_dndm']:.3f} "
            f"logMc={row['log10_m_c_msun']:.3f} beta_a={row['beta_log10_a']:.3f} "
            f"N0>1e4={row['final_total_initial_count_above_log10_4']:.1f}"
        )

    table = pd.DataFrame(rows).sort_values("eta_t").reset_index(drop=True)
    table.to_csv(tables_dir / "schechter_powerlaw_a_vs_eta_t.csv", index=False)

    _plot_logl_vs_multiplier(table.assign(best_radial_model="powerlaw_a"), figures_dir / "schechter_powerlaw_a_logl_vs_eta_t.png")
    _plot_properties(table, figures_dir / "schechter_powerlaw_a_properties_vs_eta_t.png")

    step5_table = pd.read_csv(
        PROJECT_ROOT / "variants" / "schechter_survival_time_multiplier_scan" / "outputs" / "tables" / "schechter_best_models_vs_eta_t.csv"
    ).sort_values("eta_t").reset_index(drop=True)
    _plot_step5_vs_powerlaw_a(
        step5_table=step5_table,
        powerlaw_table=table,
        output_path=figures_dir / "schechter_step5_vs_powerlaw_a_comparison.png",
    )

    best_row = table.sort_values("log_likelihood", ascending=False).iloc[0]
    summary_payload = {
        "eta_grid": eta_grid.tolist(),
        "n_detectability_iterations": 12,
        "model_spec": {"imf_family": spec.imf_family, "radial_model": spec.radial_model},
        "best_rows": table.to_dict(orient="records"),
        "global_best": json.loads(best_row.to_json()),
    }
    (tables_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2))
    print(figures_dir / "schechter_powerlaw_a_logl_vs_eta_t.png")
    print(figures_dir / "schechter_powerlaw_a_properties_vs_eta_t.png")
    print(figures_dir / "schechter_step5_vs_powerlaw_a_comparison.png")
    print(tables_dir / "summary.json")


if __name__ == "__main__":
    main()
