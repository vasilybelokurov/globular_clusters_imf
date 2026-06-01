from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

from globular_clusters_imf.blind_mixture_model import (
    build_blind_vs_bk_summary,
    build_posterior_probability_table,
    fit_blind_powerlaw_a_models,
)
from globular_clusters_imf.joint_model import JointModelSpec, fit_single_joint_model


def load_catalog(project_root: Path) -> pd.DataFrame:
    catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    return pd.read_csv(catalog_path)


def main() -> None:
    project_root = PROJECT_ROOT
    output_root = project_root / "variants" / "blind_powerlaw_a_mixture"
    output_root.mkdir(parents=True, exist_ok=True)
    skip_plot = "--skip-plot" in sys.argv

    catalog = load_catalog(project_root)
    results = fit_blind_powerlaw_a_models(
        catalog=catalog,
        project_root=project_root,
        output_root=output_root,
    )
    best_two_component_payload = min(
        [
            payload
            for payload in results["all_payloads"]
            if payload["summary"].model_class == "two_component_powerlaw_mixture"
        ],
        key=lambda payload: payload["summary"].bic,
    )
    best_two_component_posterior = build_posterior_probability_table(best_two_component_payload, catalog)
    best_two_component_bk = build_blind_vs_bk_summary(best_two_component_posterior)
    best_two_component_posterior.to_csv(
        output_root
        / "outputs"
        / "tables"
        / "blind_powerlaw_a_mixture_best_two_component_posterior_probabilities.csv",
        index=False,
    )

    selection_payload = results["selection_payload"]
    reference_summary = json.loads(
        (project_root / "outputs" / "tables" / "joint_fixed_survival_detectability_abs_longitude_em_summary.json").read_text()
    )
    reference_model = reference_summary["best_joint_model"]
    reference_spec = JointModelSpec(
        imf_family=str(reference_model["imf_family"]),
        radial_model=str(reference_model["radial_model"]),
    )
    reference_payload = fit_single_joint_model(
        context=selection_payload["selection_context"],
        spec=reference_spec,
    )

    if not skip_plot:
        build_diagnostic_figure(
            output_root=output_root,
            blind_results=results,
            two_component_payload=best_two_component_payload,
            two_component_posterior=best_two_component_posterior,
            reference_payload=reference_payload,
        )
    write_comparison_summary(
        output_root=output_root,
        blind_results=results,
        two_component_payload=best_two_component_payload,
        two_component_bk=best_two_component_bk,
        reference_payload=reference_payload,
    )


def build_diagnostic_figure(
    output_root: Path,
    blind_results: dict[str, object],
    two_component_payload: dict[str, object],
    two_component_posterior: pd.DataFrame,
    reference_payload: dict[str, object],
) -> None:
    import matplotlib.pyplot as plt

    radial_grid = build_two_component_radial_grid(two_component_payload, blind_results["selection_payload"]["selection_context"])
    context = blind_results["selection_payload"]["selection_context"]

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))

    blind_imf = build_two_component_imf_grid(two_component_payload, context)
    axes[0].plot(
        blind_imf["initial_mass_msun"],
        blind_imf["birth_imf_per_dex"],
        color="#1b9e77",
        lw=2.4,
        label="Blind 2-component best",
    )
    axes[0].plot(
        np.power(10.0, context.log_mass_grid),
        reference_payload["model"]["total_initial_count"] * reference_payload["model"]["imf_density_grid"],
        color="0.45",
        lw=2.4,
        label="Current 1-component",
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$M_{\rm ini}\ [{\rm M}_\odot]$")
    axes[0].set_ylabel(r"$dN/d\log M_{\rm ini}$")
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")

    for component_label, color, label in (
        ("total", "0.45", "Total blind mixture"),
        ("concentrated", "#d95f02", "Concentrated component"),
        ("extended", "#7570b3", "Extended component"),
    ):
        subset = radial_grid.loc[radial_grid["component_label"] == component_label]
        axes[1].plot(
            subset["semi_major_axis_kpc"],
            subset["birth_intensity_per_dex_a"],
            color=color,
            lw=2.4 if component_label == "total" else 2.0,
            ls="-" if component_label == "total" else "--",
            label=label,
        )
    axes[1].plot(
        np.power(10.0, context.log_a_grid),
        reference_payload["model"]["total_initial_count"] * reference_payload["model"]["radial_density_grid"],
        color="0.75",
        lw=3.0,
        zorder=0,
        label="Current 1-component",
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$a\ [{\rm kpc}]$")
    axes[1].set_ylabel(r"$dN/d\log a$")
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")

    marker_map = {
        1: ("#d95f02", "in-situ"),
        0: ("#1f77b4", "accreted"),
    }
    if "origin_flag" in two_component_posterior.columns:
        for origin_flag, (color, label) in marker_map.items():
            subset = two_component_posterior.loc[two_component_posterior["origin_flag"] == origin_flag]
            if subset.empty:
                continue
            axes[2].scatter(
                subset["semi_major_axis_kpc"],
                subset["p_concentrated"],
                s=18,
                alpha=0.75,
                color=color,
                label=label,
            )
    else:
        axes[2].scatter(
            two_component_posterior["semi_major_axis_kpc"],
            two_component_posterior["p_concentrated"],
            s=18,
            alpha=0.75,
            color="0.3",
        )
    order = np.argsort(two_component_posterior["semi_major_axis_kpc"].to_numpy())
    axes[2].plot(
        two_component_posterior["semi_major_axis_kpc"].to_numpy()[order],
        two_component_posterior["p_concentrated"].to_numpy()[order],
        color="0.5",
        lw=1.0,
        alpha=0.5,
    )
    axes[2].axhline(0.5, color="0.7", lw=1.0, ls=":")
    axes[2].set_xscale("log")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_xlabel(r"$a\ [{\rm kpc}]$")
    axes[2].set_ylabel(r"$P({\rm concentrated}\mid M_{\rm ini},a)$")
    if "origin_flag" in two_component_posterior.columns:
        axes[2].legend(frameon=False, fontsize=9, loc="lower left")

    for axis, panel_label in zip(axes, ("(a)", "(b)", "(c)"), strict=True):
        axis.text(0.03, 0.96, panel_label, transform=axis.transAxes, ha="left", va="top", fontsize=11)

    fig.tight_layout()
    outputs_figures = output_root / "outputs" / "figures"
    outputs_figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs_figures / "blind_powerlaw_a_mixture_diagnostic.pdf", bbox_inches="tight")
    fig.savefig(outputs_figures / "blind_powerlaw_a_mixture_diagnostic.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_comparison_summary(
    output_root: Path,
    blind_results: dict[str, object],
    two_component_payload: dict[str, object],
    two_component_bk,
    reference_payload: dict[str, object],
) -> None:
    best_row = blind_results["summary_table"].iloc[0].to_dict()
    best_two_component_row = as_summary_dict(two_component_payload["summary"])
    comparison = {
        "blind_best_model": best_row,
        "best_two_component_model": best_two_component_row,
        "best_two_component_bk_comparison": {
            "n_clusters": int(two_component_bk.n_clusters),
            "n_with_origin_flag": int(two_component_bk.n_with_origin_flag),
            "auc_in_situ_vs_p_concentrated": float(two_component_bk.auc_in_situ_vs_p_concentrated),
            "hard_assignment_accuracy": float(two_component_bk.hard_assignment_accuracy),
            "mean_p_concentrated_in_situ": float(two_component_bk.mean_p_concentrated_in_situ),
            "mean_p_concentrated_accreted": float(two_component_bk.mean_p_concentrated_accreted),
        },
        "reference_single_component_fixed_q": {
            "imf_family": reference_payload["summary"].imf_family,
            "radial_model": reference_payload["summary"].radial_model,
            "log_likelihood": float(reference_payload["summary"].log_likelihood),
            "bic": float(reference_payload["summary"].bic),
            "total_initial_count": float(reference_payload["summary"].total_initial_count),
            "total_initial_stellar_mass_msun": float(
                reference_payload["model"]["total_initial_count"]
                * np.trapezoid(
                    np.power(10.0, blind_results["selection_payload"]["selection_context"].log_mass_grid)
                    * reference_payload["model"]["imf_density_grid"],
                    blind_results["selection_payload"]["selection_context"].log_mass_grid,
                )
            ),
        },
        "delta_log_likelihood_two_component_minus_single_powerlaw_null": float(
            best_two_component_row["log_likelihood"] - best_row["log_likelihood"]
        ),
        "delta_bic_two_component_minus_single_powerlaw_null": float(
            best_two_component_row["bic"] - best_row["bic"]
        ),
    }
    outputs_tables = output_root / "outputs" / "tables"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    (outputs_tables / "blind_powerlaw_a_mixture_comparison_summary.json").write_text(
        json.dumps(comparison, indent=2)
    )


def build_two_component_imf_grid(two_component_payload: dict[str, object], context) -> pd.DataFrame:
    rows = []
    total_initial_count = float(two_component_payload["model"]["total_initial_count"])
    for log_mass, density in zip(context.log_mass_grid, two_component_payload["model"]["imf_density_grid"], strict=True):
        rows.append(
            {
                "log_initial_mass_msun": float(log_mass),
                "initial_mass_msun": float(np.power(10.0, log_mass)),
                "imf_density_per_dex": float(density),
                "birth_imf_per_dex": float(total_initial_count * density),
            }
        )
    return pd.DataFrame(rows)


def build_two_component_radial_grid(two_component_payload: dict[str, object], context) -> pd.DataFrame:
    rows = []
    model = two_component_payload["model"]
    total_initial_count = float(model["total_initial_count"])
    mix_fraction = float(model["mix_fraction_concentrated"])
    for index, log_a in enumerate(context.log_a_grid):
        rows.append(
            {
                "component_label": "total",
                "semi_major_axis_kpc": float(np.power(10.0, log_a)),
                "birth_intensity_per_dex_a": float(total_initial_count * model["mixture_radial_density_grid"][index]),
            }
        )
        rows.append(
            {
                "component_label": "concentrated",
                "semi_major_axis_kpc": float(np.power(10.0, log_a)),
                "birth_intensity_per_dex_a": float(
                    total_initial_count
                    * mix_fraction
                    * model["component_radial_density_grid"]["concentrated"][index]
                ),
            }
        )
        rows.append(
            {
                "component_label": "extended",
                "semi_major_axis_kpc": float(np.power(10.0, log_a)),
                "birth_intensity_per_dex_a": float(
                    total_initial_count
                    * (1.0 - mix_fraction)
                    * model["component_radial_density_grid"]["extended"][index]
                ),
            }
        )
    return pd.DataFrame(rows)


def as_summary_dict(summary) -> dict[str, object]:
    return {
        "model_class": summary.model_class,
        "imf_family": summary.imf_family,
        "success": bool(summary.success),
        "log_likelihood": float(summary.log_likelihood),
        "aic": float(summary.aic),
        "bic": float(summary.bic),
        "delta_bic": float(summary.delta_bic),
        "n_parameters": int(summary.n_parameters),
        "total_initial_count": float(summary.total_initial_count),
        "total_initial_stellar_mass_msun": float(summary.total_initial_stellar_mass_msun),
        "selection_fraction": float(summary.selection_fraction),
        "raw_survival_fraction": float(summary.raw_survival_fraction),
        "mean_detectability": float(summary.mean_detectability),
        "component_mix_fraction_concentrated": float(summary.component_mix_fraction_concentrated),
        "component_mix_fraction_extended": float(summary.component_mix_fraction_extended),
        "component_initial_count_concentrated": float(summary.component_initial_count_concentrated),
        "component_initial_count_extended": float(summary.component_initial_count_extended),
        "expected_observed_count_concentrated": float(summary.expected_observed_count_concentrated),
        "expected_observed_count_extended": float(summary.expected_observed_count_extended),
        "optimizer_message": str(summary.optimizer_message),
        "imf_parameters_json": str(summary.imf_parameters_json),
        "radial_parameters_json": str(summary.radial_parameters_json),
    }


if __name__ == "__main__":
    main()
