from __future__ import annotations

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

from globular_clusters_imf.gg23_survivability import (  # noqa: E402
    GG23_MODELS,
    effective_radius_kpc_from_semimajor_axis,
    gg23_initial_mass_from_present_msun,
    gg23_present_mass_msun,
    gg23_survival_mass_cut_msun,
)
from globular_clusters_imf.model import AGE_GYR  # noqa: E402


DEFAULT_MODEL_NAMES = [
    "gg23_no_bh",
    "gg23_bh",
    "gg23_bh_feh_gradient",
    "gg23_bh_past_tidal",
    "gg23_bh_feh_gradient_past_tidal",
]


def main() -> None:
    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog_with_origin_and_chemistry.csv"
    if not catalog_path.exists():
        catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog.csv"
    catalog = pd.read_csv(catalog_path)

    long_table = build_long_table(catalog)
    wide_table = build_wide_table(long_table)
    summary_table = build_summary_table(long_table)

    long_path = tables_dir / "gg23_initial_mass_estimates_long.csv"
    wide_path = tables_dir / "gg23_initial_mass_estimates_wide.csv"
    summary_path = tables_dir / "gg23_initial_mass_estimates_summary.csv"
    long_table.to_csv(long_path, index=False)
    wide_table.to_csv(wide_path, index=False)
    summary_table.to_csv(summary_path, index=False)

    plot_comparison(long_table, figures_dir / "gg23_initial_mass_estimates_comparison.pdf")
    plot_comparison(long_table, figures_dir / "gg23_initial_mass_estimates_comparison.png")

    print(f"Wrote {long_path}")
    print(f"Wrote {wide_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {figures_dir / 'gg23_initial_mass_estimates_comparison.pdf'}")


def build_long_table(catalog: pd.DataFrame) -> pd.DataFrame:
    working = catalog.copy()
    working["log10_present_mass_msun"] = np.log10(working["present_mass_msun"].to_numpy(dtype=float))

    semi_major_axis = working["semi_major_axis_kpc"].to_numpy(dtype=float)
    eccentricity = working["eccentricity"].to_numpy(dtype=float)
    effective_radius = effective_radius_kpc_from_semimajor_axis(semi_major_axis, eccentricity)
    present_mass = working["present_mass_msun"].to_numpy(dtype=float)
    baumgardt_initial_mass = working["initial_mass_msun"].to_numpy(dtype=float)

    base_columns = [
        "cluster_label",
        "cluster_name",
        "present_mass_msun",
        "log10_present_mass_msun",
        "initial_mass_msun",
        "log_initial_mass_msun",
        "semi_major_axis_kpc",
        "eccentricity",
    ]
    optional_columns = [
        "origin_flag",
        "origin_label",
        "progenitor_group",
    ]
    columns = [column for column in [*base_columns, *optional_columns] if column in working.columns]
    base = working[columns].copy()
    base["gg23_effective_radius_kpc"] = effective_radius

    rows = []
    for model_name in DEFAULT_MODEL_NAMES:
        model = GG23_MODELS[model_name]
        gg23_initial_mass = gg23_initial_mass_from_present_msun(
            present_mass,
            effective_radius,
            model,
            gradient_radius_kpc=semi_major_axis,
            age_gyr=AGE_GYR,
            eta_t=1.0,
        )
        reconstructed_present_mass = gg23_present_mass_msun(
            gg23_initial_mass,
            effective_radius,
            model,
            gradient_radius_kpc=semi_major_axis,
            age_gyr=AGE_GYR,
            eta_t=1.0,
        )
        survival_cut = gg23_survival_mass_cut_msun(
            effective_radius,
            model,
            gradient_radius_kpc=semi_major_axis,
            age_gyr=AGE_GYR,
            eta_t=1.0,
        )

        model_table = base.copy()
        model_table["gg23_model_name"] = model_name
        model_table["gg23_model_label"] = model.label
        model_table["gg23_initial_mass_msun"] = gg23_initial_mass
        model_table["log10_gg23_initial_mass_msun"] = np.log10(gg23_initial_mass)
        model_table["gg23_mass_loss_msun"] = gg23_initial_mass - present_mass
        model_table["gg23_mass_loss_fraction"] = 1.0 - present_mass / gg23_initial_mass
        model_table["gg23_survival_mass_cut_msun"] = survival_cut
        model_table["log10_gg23_survival_mass_cut_msun"] = np.log10(survival_cut)
        model_table["gg23_minus_baumgardt_log10_initial_mass_dex"] = (
            model_table["log10_gg23_initial_mass_msun"] - np.log10(baumgardt_initial_mass)
        )
        model_table["gg23_to_baumgardt_initial_mass_ratio"] = gg23_initial_mass / baumgardt_initial_mass
        model_table["gg23_reconstructed_present_mass_msun"] = reconstructed_present_mass
        model_table["gg23_present_mass_residual_fraction"] = (
            reconstructed_present_mass - present_mass
        ) / present_mass
        rows.append(model_table)

    return pd.concat(rows, ignore_index=True)


def build_wide_table(long_table: pd.DataFrame) -> pd.DataFrame:
    index_columns = [
        "cluster_label",
        "cluster_name",
        "present_mass_msun",
        "log10_present_mass_msun",
        "initial_mass_msun",
        "log_initial_mass_msun",
        "semi_major_axis_kpc",
        "eccentricity",
        "gg23_effective_radius_kpc",
    ]
    index_columns = [column for column in index_columns if column in long_table.columns]
    value_columns = [
        "gg23_initial_mass_msun",
        "log10_gg23_initial_mass_msun",
        "gg23_mass_loss_fraction",
        "gg23_minus_baumgardt_log10_initial_mass_dex",
        "gg23_to_baumgardt_initial_mass_ratio",
    ]
    wide = long_table.pivot(index=index_columns, columns="gg23_model_name", values=value_columns)
    wide.columns = [
        f"{model_name}_{wide_value_name(value_name)}" for value_name, model_name in wide.columns
    ]
    return wide.reset_index()


def wide_value_name(value_name: str) -> str:
    replacements = {
        "gg23_initial_mass_msun": "initial_mass_msun",
        "log10_gg23_initial_mass_msun": "log10_initial_mass_msun",
        "gg23_mass_loss_fraction": "mass_loss_fraction",
        "gg23_minus_baumgardt_log10_initial_mass_dex": "minus_baumgardt_log10_initial_mass_dex",
        "gg23_to_baumgardt_initial_mass_ratio": "to_baumgardt_initial_mass_ratio",
    }
    return replacements.get(value_name, value_name)


def build_summary_table(long_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in long_table.groupby("gg23_model_name", sort=False):
        rows.append(
            {
                "gg23_model_name": model_name,
                "gg23_model_label": group["gg23_model_label"].iloc[0],
                "n_clusters": int(len(group)),
                "median_log10_gg23_initial_mass_msun": float(
                    group["log10_gg23_initial_mass_msun"].median()
                ),
                "median_gg23_minus_baumgardt_dex": float(
                    group["gg23_minus_baumgardt_log10_initial_mass_dex"].median()
                ),
                "p16_gg23_minus_baumgardt_dex": float(
                    group["gg23_minus_baumgardt_log10_initial_mass_dex"].quantile(0.16)
                ),
                "p84_gg23_minus_baumgardt_dex": float(
                    group["gg23_minus_baumgardt_log10_initial_mass_dex"].quantile(0.84)
                ),
                "median_gg23_mass_loss_fraction": float(group["gg23_mass_loss_fraction"].median()),
                "max_abs_present_mass_residual_fraction": float(
                    group["gg23_present_mass_residual_fraction"].abs().max()
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_comparison(long_table: pd.DataFrame, output_path: Path) -> None:
    labels = {name: GG23_MODELS[name].label for name in DEFAULT_MODEL_NAMES}
    colors = {
        "gg23_no_bh": "#1b9e77",
        "gg23_bh": "#d95f02",
        "gg23_bh_feh_gradient": "#7570b3",
        "gg23_bh_past_tidal": "#e7298a",
        "gg23_bh_feh_gradient_past_tidal": "#66a61e",
    }

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), constrained_layout=True)
    for model_name, group in long_table.groupby("gg23_model_name", sort=False):
        axes[0].scatter(
            group["log_initial_mass_msun"],
            group["log10_gg23_initial_mass_msun"],
            s=12,
            alpha=0.55,
            linewidths=0.0,
            color=colors[model_name],
            label=labels[model_name],
        )
        order = np.argsort(group["semi_major_axis_kpc"].to_numpy(dtype=float))
        axes[1].plot(
            np.log10(group["semi_major_axis_kpc"].to_numpy(dtype=float)[order]),
            group["gg23_minus_baumgardt_log10_initial_mass_dex"].to_numpy(dtype=float)[order],
            color=colors[model_name],
            linewidth=1.1,
            alpha=0.85,
        )
    mass_limits = [
        min(long_table["log_initial_mass_msun"].min(), long_table["log10_gg23_initial_mass_msun"].min()) - 0.1,
        max(long_table["log_initial_mass_msun"].max(), long_table["log10_gg23_initial_mass_msun"].max()) + 0.1,
    ]
    axes[0].plot(mass_limits, mass_limits, color="black", linewidth=1.0, linestyle="--")
    axes[0].set_xlim(mass_limits)
    axes[0].set_ylim(mass_limits)
    axes[0].set_xlabel(r"Baumgardt $\log_{10}(M_{\rm ini}/{\rm M}_\odot)$")
    axes[0].set_ylabel(r"GG23 $\log_{10}(M_{\rm ini}/{\rm M}_\odot)$")
    axes[0].legend(loc="upper left", fontsize=7, frameon=False)

    axes[1].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_xlabel(r"$\log_{10}(a/{\rm kpc})$")
    axes[1].set_ylabel(r"GG23 - Baumgardt $\Delta\log_{10}M_{\rm ini}$")
    fig.savefig(output_path, dpi=200 if output_path.suffix.lower() == ".png" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()
