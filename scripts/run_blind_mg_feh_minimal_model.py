from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from globular_clusters_imf.blind_mg_feh_minimal_model import (
    DEFAULT_OUTPUT_PREFIX,
    build_mg_feh_posterior_probability_table,
    fit_minimal_mg_feh_models,
)
from globular_clusters_imf.catalog import (
    attach_local_gc_chemistry_to_baumgardt_catalog,
    export_local_gc_chemistry_markers,
)


def main() -> None:
    project_root = PROJECT_ROOT
    output_root = project_root / "variants" / DEFAULT_OUTPUT_PREFIX
    output_root.mkdir(parents=True, exist_ok=True)

    chemistry_markers = export_local_gc_chemistry_markers(project_root)
    catalog = attach_local_gc_chemistry_to_baumgardt_catalog(
        project_root,
        chemistry_markers=chemistry_markers,
    )

    results = fit_minimal_mg_feh_models(
        catalog=catalog,
        project_root=project_root,
        output_root=output_root,
        output_prefix=DEFAULT_OUTPUT_PREFIX,
    )

    best_payload = results["best_payload"]
    posterior = build_mg_feh_posterior_probability_table(best_payload, catalog, results["obs_context"])
    outputs_tables = output_root / "outputs" / "tables"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    posterior.to_csv(
        outputs_tables / f"{DEFAULT_OUTPUT_PREFIX}_best_model_posterior_probabilities.csv",
        index=False,
    )

    build_diagnostic_figure(output_root=output_root, results=results, posterior=posterior)
    write_summary(output_root=output_root, results=results, chemistry_markers=chemistry_markers)


def build_diagnostic_figure(
    output_root: Path,
    results: dict[str, object],
    posterior: pd.DataFrame,
) -> None:
    base_payload = results["base_payload"]
    obs_context = results["obs_context"]
    model = results["best_payload"]["model"]
    summary_table = results["summary_table"]
    mg_grid_table = results["mg_grid_table"]
    context = base_payload["context"]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8))
    ax_scatter, ax_mix, ax_resid, ax_bk = axes.flat

    with_mg = posterior.loc[posterior["has_mgfe"]].copy()
    marker_map = {1: ("o", "in-situ"), 0: ("s", "accreted")}
    scatter = None
    for origin_flag, (marker, label) in marker_map.items():
        subset = with_mg.loc[with_mg["origin_flag"] == origin_flag]
        if subset.empty:
            continue
        scatter = ax_scatter.scatter(
            subset["local_feh"],
            subset["mgfe_combined"],
            c=subset["p_concentrated"],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            s=46,
            marker=marker,
            edgecolor="k",
            linewidth=0.35,
            alpha=0.9,
            label=label,
        )
    if model["model_class"] == "two_component_mg_feh_mixture":
        feh_center = float(obs_context.feh_center)
        feh_plot = np.linspace(with_mg["local_feh"].min(), with_mg["local_feh"].max(), 200)
        feh_centered = feh_plot - feh_center
        slope = float(model["slope_feh"])
        ax_scatter.plot(
            feh_plot,
            float(model["mu_mgfe_concentrated"]) + slope * feh_centered,
            color="#d95f02",
            lw=2.0,
            ls="--",
            label="Concentrated mean",
        )
        ax_scatter.plot(
            feh_plot,
            float(model["mu_mgfe_extended"]) + slope * feh_centered,
            color="#7570b3",
            lw=2.0,
            ls="--",
            label="Extended mean",
        )
    elif model["model_class"] == "single_mg_feh_gaussian":
        feh_center = float(obs_context.feh_center)
        feh_plot = np.linspace(with_mg["local_feh"].min(), with_mg["local_feh"].max(), 200)
        ax_scatter.plot(
            feh_plot,
            float(model["mu_mgfe"]) + float(model["slope_feh"]) * (feh_plot - feh_center),
            color="#d95f02",
            lw=2.2,
            label="Single-component mean",
        )
    ax_scatter.set_xlabel(r"[Fe/H]")
    ax_scatter.set_ylabel(r"[Mg/Fe]")
    ax_scatter.legend(frameon=False, fontsize=8.4, loc="best")
    if scatter is not None:
        fig.colorbar(scatter, ax=ax_scatter, label=r"$P({\rm concentrated}\mid a, {\rm Mg}, {\rm [Fe/H]})$")

    a_grid = np.power(10.0, context.log_a_grid)
    ax_mix.plot(a_grid, np.asarray(model["w_grid"], dtype=float), color="#1b9e77", lw=2.4, label=r"$w(a)$")
    order = np.argsort(posterior["semi_major_axis_kpc"].to_numpy())
    ax_mix.scatter(
        posterior["semi_major_axis_kpc"].to_numpy()[order],
        posterior["p_concentrated"].to_numpy()[order],
        s=14,
        color="0.15",
        alpha=0.55,
        label=r"$P({\rm conc}\mid a, {\rm Mg}, {\rm [Fe/H]})$",
    )
    ax_mix.set_xscale("log")
    ax_mix.set_ylim(0.0, 1.0)
    ax_mix.set_xlabel(r"$a\ [{\rm kpc}]$")
    ax_mix.set_ylabel("Concentrated fraction")
    ax_mix.legend(frameon=False, fontsize=8.4, loc="best")

    for slice_label, color in zip(("feh_p16", "feh_p50", "feh_p84"), ("#1b9e77", "#d95f02", "#7570b3"), strict=True):
        subset = mg_grid_table.loc[mg_grid_table["slice_label"] == slice_label]
        if subset.empty:
            continue
        if model["model_class"] == "single_mg_feh_gaussian":
            density = subset.loc[subset["component_label"] == "single"]
            ax_resid.plot(density["mgfe"], density["density"], color=color, lw=2.0, label=slice_label)
        else:
            for component_label, ls in (("concentrated", "--"), ("extended", ":")):
                density = subset.loc[subset["component_label"] == component_label]
                ax_resid.plot(
                    density["mgfe"],
                    density["density"],
                    color=color,
                    lw=1.8,
                    ls=ls,
                    label=f"{slice_label} {component_label}" if slice_label == "feh_p16" else None,
                )
    ax_resid.hist(
        with_mg["mgfe_combined"].to_numpy(dtype=float),
        bins=12,
        density=True,
        color="0.88",
        edgecolor="0.35",
        linewidth=0.8,
        alpha=0.7,
        label="Measured Mg sample",
    )
    ax_resid.set_xlabel(r"[Mg/Fe]")
    ax_resid.set_ylabel("Density")
    ax_resid.legend(frameon=False, fontsize=8.0, loc="best")

    group_positions = {"accreted": 0, "in-situ": 1}
    for origin_flag, label in ((0, "accreted"), (1, "in-situ")):
        subset = posterior.loc[posterior["origin_flag"] == origin_flag, "p_concentrated"]
        if subset.empty:
            continue
        ax_bk.scatter(
            np.full(len(subset), group_positions[label]),
            subset,
            color="#1f78b4" if origin_flag == 1 else "#e31a1c",
            s=26,
            alpha=0.65,
            edgecolor="none",
        )
        ax_bk.hlines(float(subset.median()), group_positions[label] - 0.22, group_positions[label] + 0.22, color="0.1", lw=2.1)
    ax_bk.set_xlim(-0.5, 1.5)
    ax_bk.set_ylim(0.0, 1.0)
    ax_bk.set_xticks([0, 1], ["accreted", "in-situ"])
    ax_bk.set_ylabel(r"$P({\rm concentrated})$")

    delta_logl = (
        float(summary_table.loc[summary_table["model_class"] == "two_component_mg_feh_mixture", "chemistry_log_likelihood"].iloc[0])
        - float(summary_table.loc[summary_table["model_class"] == "single_mg_feh_gaussian", "chemistry_log_likelihood"].iloc[0])
    )
    delta_bic = (
        float(summary_table.loc[summary_table["model_class"] == "two_component_mg_feh_mixture", "bic"].iloc[0])
        - float(summary_table.loc[summary_table["model_class"] == "single_mg_feh_gaussian", "bic"].iloc[0])
    )
    fig.suptitle(
        (
            "Minimal blind 2-component model with Mg + [Fe/H]\n"
            f"Fixed baseline: {base_payload['summary'].imf_family} + {base_payload['summary'].radial_model}; "
            f"ΔlogL={delta_logl:+.2f}, ΔBIC={delta_bic:+.2f}"
        ),
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    outputs_figures = output_root / "outputs" / "figures"
    outputs_figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs_figures / f"{DEFAULT_OUTPUT_PREFIX}_diagnostic.pdf")
    fig.savefig(outputs_figures / f"{DEFAULT_OUTPUT_PREFIX}_diagnostic.png", dpi=220)
    plt.close(fig)


def write_summary(
    output_root: Path,
    results: dict[str, object],
    chemistry_markers: pd.DataFrame,
) -> None:
    summary_table = results["summary_table"]
    best_row = summary_table.iloc[0]
    base_payload = results["base_payload"]
    mg_source_counts = {
        str(key): int(value)
        for key, value in chemistry_markers["mgfe_combined_source"].value_counts(dropna=False).items()
    }
    payload = {
        "n_clusters_total": int(len(results["posterior_table"])),
        "n_clusters_with_mg": int(results["obs_context"].has_mgfe.sum()),
        "best_model": json.loads(best_row.to_json()),
        "base_model": {
            "imf_family": base_payload["summary"].imf_family,
            "radial_model": base_payload["summary"].radial_model,
            "total_initial_count": float(base_payload["summary"].total_initial_count),
            "log_likelihood": float(base_payload["summary"].log_likelihood),
        },
        "bk_comparison": {
            "auc_in_situ_vs_p_concentrated": float(results["bk_comparison"].auc_in_situ_vs_p_concentrated),
            "hard_assignment_accuracy": float(results["bk_comparison"].hard_assignment_accuracy),
            "mean_p_concentrated_in_situ": float(results["bk_comparison"].mean_p_concentrated_in_situ),
            "mean_p_concentrated_accreted": float(results["bk_comparison"].mean_p_concentrated_accreted),
        },
        "mg_marker_sources": mg_source_counts,
        "feh_center": float(results["obs_context"].feh_center),
    }
    mg_only_summary_path = (
        PROJECT_ROOT
        / "variants"
        / "blind_mg_only_minimal_mixture"
        / "outputs"
        / "tables"
        / "blind_mg_only_minimal_mixture_model_summary.csv"
    )
    if mg_only_summary_path.exists():
        mg_only_summary = pd.read_csv(mg_only_summary_path).set_index("model_class")
        mg_feh_summary = summary_table.set_index("model_class")
        payload["comparison_vs_mg_only"] = {
            "two_component_delta_log_likelihood": float(
                mg_feh_summary.loc["two_component_mg_feh_mixture", "chemistry_log_likelihood"]
                - mg_only_summary.loc["two_component_mg_mixture", "chemistry_log_likelihood"]
            ),
            "two_component_delta_aic": float(
                mg_feh_summary.loc["two_component_mg_feh_mixture", "aic"]
                - mg_only_summary.loc["two_component_mg_mixture", "aic"]
            ),
            "two_component_delta_bic": float(
                mg_feh_summary.loc["two_component_mg_feh_mixture", "bic"]
                - mg_only_summary.loc["two_component_mg_mixture", "bic"]
            ),
            "single_component_delta_log_likelihood": float(
                mg_feh_summary.loc["single_mg_feh_gaussian", "chemistry_log_likelihood"]
                - mg_only_summary.loc["single_mg_gaussian", "chemistry_log_likelihood"]
            ),
            "single_component_delta_aic": float(
                mg_feh_summary.loc["single_mg_feh_gaussian", "aic"]
                - mg_only_summary.loc["single_mg_gaussian", "aic"]
            ),
            "single_component_delta_bic": float(
                mg_feh_summary.loc["single_mg_feh_gaussian", "bic"]
                - mg_only_summary.loc["single_mg_gaussian", "bic"]
            ),
        }
    outputs_tables = output_root / "outputs" / "tables"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    (outputs_tables / f"{DEFAULT_OUTPUT_PREFIX}_comparison_summary.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
