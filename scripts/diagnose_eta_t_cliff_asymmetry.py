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
            "selection_probability": selection_data,
            "log_selection_term": np.log(selection_data),
            "log_imf_term": np.log(imf_density),
            "log_radial_term": np.log(radial_density),
        }
    )
    table["log_n0_term"] = log_n0
    table["log_contribution_total"] = (
        table["log_n0_term"] + table["log_imf_term"] + table["log_radial_term"] + table["log_selection_term"]
    )
    return table


def _adjacent_pair_table(low_eta: float, high_eta: float, low_table: pd.DataFrame, high_table: pd.DataFrame) -> pd.DataFrame:
    merged = low_table.merge(
        high_table[
            [
                "cluster_label",
                "log_contribution_total",
                "log_imf_term",
                "log_radial_term",
                "log_selection_term",
                "selection_probability",
            ]
        ],
        on="cluster_label",
        suffixes=("_low", "_high"),
    )
    merged["eta_low"] = low_eta
    merged["eta_high"] = high_eta
    merged["delta_log_contribution"] = merged["log_contribution_total_high"] - merged["log_contribution_total_low"]
    merged["delta_log_imf_term"] = merged["log_imf_term_high"] - merged["log_imf_term_low"]
    merged["delta_log_radial_term"] = merged["log_radial_term_high"] - merged["log_radial_term_low"]
    merged["delta_log_selection_term"] = merged["log_selection_term_high"] - merged["log_selection_term_low"]
    merged["selection_ratio_high_over_low"] = merged["selection_probability_high"] / np.maximum(
        merged["selection_probability_low"], 1.0e-300
    )
    return merged.sort_values("delta_log_contribution", ascending=False).reset_index(drop=True)


def _tiny_selection_row(eta_t: float, radial_model: str, table: pd.DataFrame) -> dict[str, float]:
    logsel = np.asarray(table["log_selection_term"], dtype=float)
    sel = np.asarray(table["selection_probability"], dtype=float)
    row = {
        "eta_t": eta_t,
        "radial_model": radial_model,
        "min_log_selection": float(np.min(logsel)),
        "p01_log_selection": float(np.quantile(logsel, 0.01)),
        "p05_log_selection": float(np.quantile(logsel, 0.05)),
        "median_log_selection": float(np.median(logsel)),
        "n_sel_lt_1e_20": int(np.sum(sel < 1.0e-20)),
        "n_sel_lt_1e_15": int(np.sum(sel < 1.0e-15)),
        "n_sel_lt_1e_10": int(np.sum(sel < 1.0e-10)),
        "n_sel_lt_1e_8": int(np.sum(sel < 1.0e-8)),
        "n_sel_lt_1e_6": int(np.sum(sel < 1.0e-6)),
        "n_sel_lt_1e_4": int(np.sum(sel < 1.0e-4)),
    }
    worst = table.nsmallest(5, "selection_probability")[
        ["cluster_label", "selection_probability", "log_selection_term", "log_initial_mass_msun", "semi_major_axis_kpc"]
    ]
    for idx, rec in enumerate(worst.to_dict(orient="records"), start=1):
        row[f"worst_{idx}_cluster"] = rec["cluster_label"]
        row[f"worst_{idx}_selection"] = float(rec["selection_probability"])
        row[f"worst_{idx}_log_selection"] = float(rec["log_selection_term"])
    return row


def _plot(summary_df: pd.DataFrame, tiny_df: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 9.5), sharex="col")
    colors = {"step5": "#d95f02", "logpoly3": "#1b9e77"}
    for radial_model, group in summary_df.groupby("radial_model"):
        color = colors.get(radial_model, None)
        axes[0, 0].plot(group["eta_t"], group["log_likelihood"], marker="o", color=color, label=radial_model)
        axes[0, 1].plot(group["eta_t"], group["dlogL_prev"], marker="o", color=color, label=radial_model)
    axes[0, 0].set_ylabel(r"$\log L$")
    axes[0, 1].set_ylabel(r"Adjacent $\Delta \log L$")
    axes[0, 1].axhline(0.0, color="0.8", linewidth=1.0)
    axes[0, 0].legend(frameon=False)

    for radial_model, group in tiny_df.groupby("radial_model"):
        color = colors.get(radial_model, None)
        axes[1, 0].plot(group["eta_t"], group["min_log_selection"], marker="o", color=color)
        axes[1, 1].plot(group["eta_t"], group["n_sel_lt_1e_10"], marker="o", color=color, label="<1e-10")
        axes[1, 1].plot(group["eta_t"], group["n_sel_lt_1e_8"], marker="s", color=color, linestyle="--", alpha=0.7, label=f"{radial_model} <1e-8")
        axes[2, 0].plot(group["eta_t"], group["median_log_selection"], marker="o", color=color)
        axes[2, 1].plot(group["eta_t"], group["p05_log_selection"], marker="o", color=color)
    axes[1, 0].set_ylabel("Min log selection")
    axes[1, 1].set_ylabel("Count with tiny selection")
    axes[2, 0].set_ylabel("Median log selection")
    axes[2, 1].set_ylabel("5th pct log selection")
    axes[2, 0].set_xlabel(r"$\eta_t$")
    axes[2, 1].set_xlabel(r"$\eta_t$")
    axes[1, 1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-mass-min", type=float, default=4.1)
    parser.add_argument("--exclude-cluster-label", action="append", default=[])
    parser.add_argument("--eta-min", type=float, default=0.6)
    parser.add_argument("--eta-max", type=float, default=1.2)
    args = parser.parse_args()

    from globular_clusters_imf.detectability_longitude_model import fit_single_component_detectability_em_with_abs_longitude
    from globular_clusters_imf.joint_model import JointModelSpec, imf_parameter_count
    from globular_clusters_imf.model import fit_catalog_models
    from globular_clusters_imf.smooth_survivability import build_smooth_survivability_grid

    exclude_labels = [str(x) for x in args.exclude_cluster_label]
    tag = f"log10_{args.log_mass_min:.2f}".replace('.', 'p')
    if exclude_labels:
        extra = '_'.join(''.join(ch.lower() if ch.isalnum() else '_' for ch in x).strip('_') for x in exclude_labels)
        tag = f"{tag}_exclude_{extra}"
    output_root = PROJECT_ROOT / "variants" / f"eta_t_cliff_diagnosis_{tag}"
    figures_dir = output_root / "outputs" / "figures"
    tables_dir = output_root / "outputs" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    prepared_full = fit_catalog_models(catalog, output_root)["catalog"]
    prepared = prepared_full.loc[prepared_full["log_initial_mass_msun"] >= float(args.log_mass_min)].copy()
    if exclude_labels:
        prepared = prepared.loc[~prepared["cluster_label"].astype(str).isin(exclude_labels)].copy()
    prepared = prepared.reset_index(drop=True)

    eta_grid = np.round(np.arange(args.eta_min, args.eta_max + 0.0001, 0.1), 1)
    specs = [
        JointModelSpec(imf_family="schechter", radial_model="step5"),
        JointModelSpec(imf_family="schechter", radial_model="logpoly3"),
    ]
    state = {spec.radial_model: {"completeness": None, "radial": None} for spec in specs}
    per_spec_tables: dict[str, dict[float, pd.DataFrame]] = {spec.radial_model: {} for spec in specs}
    summary_rows = []
    tiny_rows = []

    for eta_t in eta_grid:
        smooth = build_smooth_survivability_grid(prepared_full, eta_t=float(eta_t))
        override = {
            "log_mass_grid": np.asarray(smooth["log_mass_grid"], dtype=float),
            "log_a_grid": np.asarray(smooth["log_a_grid"], dtype=float),
            "semi_major_axis_grid_kpc": np.asarray(smooth["semi_major_axis_grid_kpc"], dtype=float),
            "survival_probability": np.asarray(smooth["survival_probability"], dtype=float),
            "selection_offset_dex": 0.0,
            "bandwidth_log10_a_dex": float(smooth["bandwidth_log10_a_dex"]),
            "smooth_survivability_summary": smooth["summary"],
        }
        for spec in specs:
            st = state[spec.radial_model]
            result = fit_single_component_detectability_em_with_abs_longitude(
                prepared,
                project_root=output_root,
                spec=spec,
                n_iterations=12,
                start_completeness_raw_parameters=st["completeness"],
                start_radial_params=st["radial"],
                survival_grid_override=override,
            )
            n_imf = imf_parameter_count(spec.imf_family)
            state[spec.radial_model] = {
                "completeness": np.asarray(result["final_completeness_raw_parameters"], dtype=float),
                "radial": np.asarray(result["final_payload"]["raw_parameters"][n_imf:], dtype=float),
            }
            table = _cluster_contribution_table(result=result, catalog=prepared)
            per_spec_tables[spec.radial_model][float(eta_t)] = table
            tiny_rows.append(_tiny_selection_row(float(eta_t), spec.radial_model, table))
            summary_rows.append({
                "eta_t": float(eta_t),
                "radial_model": spec.radial_model,
                "log_likelihood": float(result["final_payload"]["summary"].log_likelihood),
                "alpha_dndm": float(result["final_payload"]["model"]["imf_parameters"]["alpha_dndm"]),
                "log10_m_c_msun": float(result["final_payload"]["model"]["imf_parameters"]["log10_m_c_msun"]),
                "total_initial_count": float(result["final_payload"]["model"]["total_initial_count"]),
                "selection_fraction": float(result["final_payload"]["model"]["selection_fraction"]),
                "raw_survival_fraction": float(result["final_payload"]["model"]["raw_survival_fraction"]),
            })
            print(f"eta_t={eta_t:.1f} radial={spec.radial_model} logL={result['final_payload']['summary'].log_likelihood:.3f}")

    summary_df = pd.DataFrame(summary_rows).sort_values(["radial_model", "eta_t"]).reset_index(drop=True)
    summary_df["dlogL_prev"] = summary_df.groupby("radial_model")["log_likelihood"].diff()
    tiny_df = pd.DataFrame(tiny_rows).sort_values(["radial_model", "eta_t"]).reset_index(drop=True)

    summary_df.to_csv(tables_dir / "summary.csv", index=False)
    tiny_df.to_csv(tables_dir / "tiny_selection_counts.csv", index=False)

    for radial_model, tables in per_spec_tables.items():
        etas = sorted(tables)
        for low_eta, high_eta in zip(etas[:-1], etas[1:]):
            trans = _adjacent_pair_table(low_eta, high_eta, tables[low_eta], tables[high_eta])
            stem = f"{radial_model}_eta_{low_eta:.1f}_to_{high_eta:.1f}".replace('.', 'p')
            trans.to_csv(tables_dir / f"{stem}_transition.csv", index=False)
            trans.head(20).to_csv(tables_dir / f"{stem}_top_positive.csv", index=False)
            trans.tail(20).to_csv(tables_dir / f"{stem}_top_negative.csv", index=False)

    _plot(summary_df, tiny_df, figures_dir / "eta_t_cliff_asymmetry.png")
    (tables_dir / "summary.json").write_text(json.dumps({
        "log_mass_min": float(args.log_mass_min),
        "exclude_cluster_labels": exclude_labels,
        "eta_grid": eta_grid.tolist(),
        "n_clusters_fitted": int(len(prepared)),
        "n_clusters_total": int(len(prepared_full)),
    }, indent=2))
    print(figures_dir / "eta_t_cliff_asymmetry.png")


if __name__ == "__main__":
    main()
