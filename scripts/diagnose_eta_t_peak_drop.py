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


def _cluster_contribution_table(*, result: dict[str, object], catalog: pd.DataFrame) -> pd.DataFrame:
    context = result["final_context"]
    model = result["final_payload"]["model"]
    selection_data = np.clip(
        context.selection_interpolator(np.column_stack([context.log_mass_data, context.log_a_data])),
        1.0e-300,
        1.0,
    )
    total_initial_count = float(model["total_initial_count"])
    log_n0 = float(np.log(total_initial_count))
    imf_density = np.clip(np.asarray(model["imf_density_data"], dtype=float), 1.0e-300, None)
    radial_density = np.clip(np.asarray(model["radial_density_data"], dtype=float), 1.0e-300, None)
    cluster_labels = catalog.get("cluster_label", pd.Series(np.arange(len(catalog)), index=catalog.index)).astype(str).to_numpy()
    table = pd.DataFrame(
        {
            "cluster_label": cluster_labels,
            "log_initial_mass_msun": np.asarray(context.log_mass_data, dtype=float),
            "semi_major_axis_kpc": np.power(10.0, np.asarray(context.log_a_data, dtype=float)),
            "log10_semi_major_axis_kpc": np.asarray(context.log_a_data, dtype=float),
            "log_imf_term": np.log(imf_density),
            "log_radial_term": np.log(radial_density),
            "log_selection_term": np.log(selection_data),
        }
    )
    table["log_n0_term"] = log_n0
    table["log_contribution_total"] = (
        table["log_n0_term"] + table["log_imf_term"] + table["log_radial_term"] + table["log_selection_term"]
    )
    table["trusted_broad"] = (
        (table["log_initial_mass_msun"] >= 4.0) & (table["semi_major_axis_kpc"] <= 50.0)
    )
    table["trusted_strict"] = (
        (table["log_initial_mass_msun"] >= 5.0) & (table["semi_major_axis_kpc"] <= 30.0)
    )
    return table


def _adjacent_pair_summary(low_eta: float, high_eta: float, low_table: pd.DataFrame, high_table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    merged = low_table.merge(
        high_table[["cluster_label", "log_contribution_total", "log_imf_term", "log_radial_term", "log_selection_term"]],
        on="cluster_label",
        suffixes=("_low", "_high"),
    )
    merged["delta_log_contribution"] = merged["log_contribution_total_high"] - merged["log_contribution_total_low"]
    merged["delta_log_imf_term"] = merged["log_imf_term_high"] - merged["log_imf_term_low"]
    merged["delta_log_radial_term"] = merged["log_radial_term_high"] - merged["log_radial_term_low"]
    merged["delta_log_selection_term"] = merged["log_selection_term_high"] - merged["log_selection_term_low"]

    summary = {
        "eta_low": low_eta,
        "eta_high": high_eta,
        "delta_data_term_total": float(merged["delta_log_contribution"].sum()),
        "delta_data_term_trusted_broad": float(merged.loc[merged["trusted_broad"], "delta_log_contribution"].sum()),
        "delta_data_term_outside_trusted_broad": float(merged.loc[~merged["trusted_broad"], "delta_log_contribution"].sum()),
        "delta_data_term_trusted_strict": float(merged.loc[merged["trusted_strict"], "delta_log_contribution"].sum()),
        "delta_data_term_outside_trusted_strict": float(merged.loc[~merged["trusted_strict"], "delta_log_contribution"].sum()),
    }
    return merged.sort_values("delta_log_contribution", ascending=False).reset_index(drop=True), summary


def _plot_transition_diagnostics(summary_table: pd.DataFrame, adjacent_table: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.2), sharex=True)

    ax = axes[0]
    ax.plot(summary_table["eta_t"], summary_table["log_likelihood"], color="black", marker="o")
    ax.set_ylabel(r"$\log L$")

    ax = axes[1]
    for col, color, label in [
        ("step_weight_1", "#1b9e77", "w1"),
        ("step_weight_2", "#d95f02", "w2"),
        ("step_weight_3", "#7570b3", "w3"),
        ("step_weight_4", "#e7298a", "w4"),
        ("step_weight_5", "#66a61e", "w5"),
    ]:
        ax.plot(summary_table["eta_t"], summary_table[col], marker="o", label=label, color=color)
    ax.set_ylabel("Step5 bin weight")
    ax.legend(frameon=False, ncol=5, fontsize=8, loc="upper center")

    ax = axes[2]
    ax.plot(adjacent_table["eta_high"], adjacent_table["delta_data_term_total"], color="black", marker="o", label="total")
    ax.plot(adjacent_table["eta_high"], adjacent_table["delta_data_term_trusted_broad"], color="#1b9e77", marker="o", label="trusted broad")
    ax.plot(adjacent_table["eta_high"], adjacent_table["delta_data_term_outside_trusted_broad"], color="#d95f02", marker="o", label="outside broad")
    ax.axhline(0.0, color="0.8", linewidth=1.0)
    ax.set_ylabel(r"Adjacent $\Delta$ data term")
    ax.set_xlabel(r"$\eta_t$ at upper point")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    from globular_clusters_imf.detectability_longitude_model import fit_single_component_detectability_em_with_abs_longitude
    from globular_clusters_imf.joint_model import JointModelSpec, imf_parameter_count
    from globular_clusters_imf.model import fit_catalog_models
    from globular_clusters_imf.smooth_survivability import build_smooth_survivability_grid

    output_root = PROJECT_ROOT / "variants" / "schechter_survival_peak_drop_diagnosis"
    figures_dir = output_root / "outputs" / "figures"
    tables_dir = output_root / "outputs" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    prepared_catalog = fit_catalog_models(catalog, output_root)["catalog"]

    eta_grid = np.arange(0.1, 2.0, 0.1)
    spec = JointModelSpec(imf_family="schechter", radial_model="step5")
    prior_state = {"completeness": None, "radial": None}

    result_tables: dict[float, pd.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []

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
        result_tables[round(float(eta_t), 1)] = _cluster_contribution_table(result=result, catalog=prepared_catalog)

        radial_params = result["final_payload"]["model"]["radial_parameters"]
        comp_params = result["summary_payload"]["final_completeness_parameters"]
        row = {
            "eta_t": round(float(eta_t), 1),
            "log_likelihood": float(result["final_payload"]["summary"].log_likelihood),
            "alpha_dndm": float(result["final_payload"]["model"]["imf_parameters"]["alpha_dndm"]),
            "log10_m_c_msun": float(result["final_payload"]["model"]["imf_parameters"]["log10_m_c_msun"]),
            "total_initial_count": float(result["final_payload"]["model"]["total_initial_count"]),
            "selection_fraction": float(result["final_payload"]["model"]["selection_fraction"]),
            "raw_survival_fraction": float(result["final_payload"]["model"]["raw_survival_fraction"]),
            "mean_detectability": float(result["summary_payload"]["final_mean_detectability"]),
            "completeness_intercept": float(comp_params["intercept"]),
            "completeness_mass_slope": float(comp_params["mass_slope"]),
            "completeness_distance_slope": float(comp_params["distance_slope"]),
            "completeness_latitude_slope": float(comp_params["latitude_slope"]),
            "completeness_longitude_slope": float(comp_params["longitude_slope"]),
        }
        for idx, weight in enumerate(radial_params["bin_weights"], start=1):
            row[f"step_weight_{idx}"] = float(weight)
        summary_rows.append(row)
        print(f"eta_t={eta_t:.1f} logL={row['log_likelihood']:.3f}")

    summary_table = pd.DataFrame(summary_rows).sort_values("eta_t").reset_index(drop=True)
    summary_table.to_csv(tables_dir / "eta_t_transition_summary.csv", index=False)

    adjacent_rows = []
    focus_pairs = [(1.0, 1.1), (1.4, 1.5), (1.5, 1.6), (1.6, 1.7), (1.7, 1.8)]
    for low_eta, high_eta in zip(summary_table["eta_t"].iloc[:-1], summary_table["eta_t"].iloc[1:], strict=True):
        merged, summary = _adjacent_pair_summary(low_eta, high_eta, result_tables[round(float(low_eta),1)], result_tables[round(float(high_eta),1)])
        adjacent_rows.append(summary)
        if (round(float(low_eta),1), round(float(high_eta),1)) in focus_pairs:
            stem = f"eta_{low_eta:.1f}_to_{high_eta:.1f}".replace('.', 'p')
            merged.to_csv(tables_dir / f"cluster_contribution_{stem}.csv", index=False)
            merged.head(30).to_csv(tables_dir / f"top_positive_{stem}.csv", index=False)
            merged.tail(30).to_csv(tables_dir / f"top_negative_{stem}.csv", index=False)

    adjacent_table = pd.DataFrame(adjacent_rows)
    adjacent_table.to_csv(tables_dir / "adjacent_eta_transition_breakdown.csv", index=False)
    _plot_transition_diagnostics(summary_table, adjacent_table, figures_dir / "eta_t_peak_drop_diagnostics.png")

    summary = {
        "focus_pairs": focus_pairs,
        "peak_eta_t": float(summary_table.loc[summary_table['log_likelihood'].idxmax(), 'eta_t']),
    }
    (tables_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(figures_dir / "eta_t_peak_drop_diagnostics.png")


if __name__ == "__main__":
    main()
