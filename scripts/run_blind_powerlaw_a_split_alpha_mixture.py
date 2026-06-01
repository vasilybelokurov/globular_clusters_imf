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

from globular_clusters_imf.blind_mixture_model import (
    DEFAULT_SPLIT_ALPHA_OUTPUT_PREFIX,
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
    output_root = project_root / "variants" / "blind_powerlaw_a_split_alpha_mixture"
    output_root.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog(project_root)
    results = fit_blind_powerlaw_a_models(
        catalog=catalog,
        project_root=project_root,
        output_root=output_root,
        imf_families=("schechter",),
        include_split_alpha_schechter=True,
        output_prefix=DEFAULT_SPLIT_ALPHA_OUTPUT_PREFIX,
    )

    shared_payload = find_payload(results["all_payloads"], "two_component_powerlaw_mixture")
    split_alpha_payload = find_payload(results["all_payloads"], "two_component_powerlaw_mixture_split_alpha")
    single_payload = find_payload(results["all_payloads"], "single_powerlaw_radial")
    split_alpha_posterior = build_posterior_probability_table(split_alpha_payload, catalog)
    split_alpha_bk = build_blind_vs_bk_summary(split_alpha_posterior)
    outputs_tables = output_root / "outputs" / "tables"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    split_alpha_posterior.to_csv(
        outputs_tables / f"{DEFAULT_SPLIT_ALPHA_OUTPUT_PREFIX}_best_split_alpha_posterior_probabilities.csv",
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

    build_diagnostic_figure(
        output_root=output_root,
        selection_context=selection_payload["selection_context"],
        reference_payload=reference_payload,
        shared_payload=shared_payload,
        split_alpha_payload=split_alpha_payload,
        split_alpha_posterior=split_alpha_posterior,
    )
    write_summary(
        output_root=output_root,
        results=results,
        single_payload=single_payload,
        shared_payload=shared_payload,
        split_alpha_payload=split_alpha_payload,
        split_alpha_bk=split_alpha_bk,
        reference_payload=reference_payload,
    )


def find_payload(payloads: list[dict[str, object]], model_class: str) -> dict[str, object]:
    for payload in payloads:
        if payload["summary"].model_class == model_class:
            return payload
    raise KeyError(model_class)


def build_diagnostic_figure(
    output_root: Path,
    selection_context,
    reference_payload: dict[str, object],
    shared_payload: dict[str, object],
    split_alpha_payload: dict[str, object],
    split_alpha_posterior: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))

    mass_grid = np.power(10.0, selection_context.log_mass_grid)
    axes[0].plot(
        mass_grid,
        reference_payload["model"]["total_initial_count"] * reference_payload["model"]["imf_density_grid"],
        color="0.75",
        lw=3.2,
        zorder=0,
        label="Current 1-component",
    )
    axes[0].plot(
        mass_grid,
        shared_payload["model"]["total_initial_count"] * shared_payload["model"]["imf_density_grid"],
        color="#1b9e77",
        lw=2.2,
        ls=":",
        label="Blind shared IMF",
    )
    total_count_split = float(split_alpha_payload["model"]["total_initial_count"])
    mix_fraction = float(split_alpha_payload["model"]["mix_fraction_concentrated"])
    axes[0].plot(
        mass_grid,
        total_count_split * split_alpha_payload["model"]["imf_density_grid"],
        color="0.15",
        lw=2.2,
        label="Blind split-$\\alpha$ total",
    )
    axes[0].plot(
        mass_grid,
        total_count_split
        * mix_fraction
        * split_alpha_payload["model"]["component_imf_density_grid"]["concentrated"],
        color="#d95f02",
        lw=2.0,
        ls="--",
        label="Concentrated IMF",
    )
    axes[0].plot(
        mass_grid,
        total_count_split
        * (1.0 - mix_fraction)
        * split_alpha_payload["model"]["component_imf_density_grid"]["extended"],
        color="#7570b3",
        lw=2.0,
        ls="--",
        label="Extended IMF",
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$M_{\rm ini}\ [{\rm M}_\odot]$")
    axes[0].set_ylabel(r"$dN/d\log M_{\rm ini}$")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")

    a_grid = np.power(10.0, selection_context.log_a_grid)
    axes[1].plot(
        a_grid,
        reference_payload["model"]["total_initial_count"] * reference_payload["model"]["radial_density_grid"],
        color="0.75",
        lw=3.2,
        zorder=0,
        label="Current 1-component",
    )
    axes[1].plot(
        a_grid,
        shared_payload["model"]["total_initial_count"] * shared_payload["model"]["mixture_radial_density_grid"],
        color="#1b9e77",
        lw=2.2,
        ls=":",
        label="Blind shared IMF",
    )
    axes[1].plot(
        a_grid,
        total_count_split * split_alpha_payload["model"]["mixture_radial_density_grid"],
        color="0.15",
        lw=2.2,
        label="Blind split-$\\alpha$ total",
    )
    axes[1].plot(
        a_grid,
        total_count_split
        * mix_fraction
        * split_alpha_payload["model"]["component_radial_density_grid"]["concentrated"],
        color="#d95f02",
        lw=2.0,
        ls="--",
        label="Concentrated radial",
    )
    axes[1].plot(
        a_grid,
        total_count_split
        * (1.0 - mix_fraction)
        * split_alpha_payload["model"]["component_radial_density_grid"]["extended"],
        color="#7570b3",
        lw=2.0,
        ls="--",
        label="Extended radial",
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$a\ [{\rm kpc}]$")
    axes[1].set_ylabel(r"$dN/d\log a$")
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper right")

    marker_map = {
        1: ("#d95f02", "in-situ"),
        0: ("#1f77b4", "accreted"),
    }
    for origin_flag, (color, label) in marker_map.items():
        subset = split_alpha_posterior.loc[split_alpha_posterior["origin_flag"] == origin_flag]
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
    order = np.argsort(split_alpha_posterior["semi_major_axis_kpc"].to_numpy())
    axes[2].plot(
        split_alpha_posterior["semi_major_axis_kpc"].to_numpy()[order],
        split_alpha_posterior["p_concentrated"].to_numpy()[order],
        color="0.45",
        lw=1.1,
        alpha=0.65,
    )
    axes[2].axhline(0.5, color="0.7", lw=1.0, ls=":")
    axes[2].set_xscale("log")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_xlabel(r"$a\ [{\rm kpc}]$")
    axes[2].set_ylabel(r"$P({\rm concentrated}\mid M_{\rm ini},a)$")
    axes[2].legend(frameon=False, fontsize=9, loc="lower left")

    for axis, panel_label in zip(axes, ("(a)", "(b)", "(c)"), strict=True):
        axis.text(0.03, 0.96, panel_label, transform=axis.transAxes, ha="left", va="top", fontsize=11)

    fig.tight_layout()
    outputs_figures = output_root / "outputs" / "figures"
    outputs_figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs_figures / f"{DEFAULT_SPLIT_ALPHA_OUTPUT_PREFIX}_diagnostic.pdf", bbox_inches="tight")
    fig.savefig(outputs_figures / f"{DEFAULT_SPLIT_ALPHA_OUTPUT_PREFIX}_diagnostic.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    output_root: Path,
    results: dict[str, object],
    single_payload: dict[str, object],
    shared_payload: dict[str, object],
    split_alpha_payload: dict[str, object],
    split_alpha_bk,
    reference_payload: dict[str, object],
) -> None:
    summary_table = results["summary_table"]
    outputs_tables = output_root / "outputs" / "tables"
    comparison = {
        "blind_best_model": summary_table.iloc[0].to_dict(),
        "blind_single_powerlaw_null": summary_row_dict(single_payload["summary"]),
        "blind_shared_imf_two_component": summary_row_dict(shared_payload["summary"]),
        "blind_split_alpha_two_component": summary_row_dict(split_alpha_payload["summary"]),
        "blind_split_alpha_bk_comparison": {
            "n_clusters": int(split_alpha_bk.n_clusters),
            "n_with_origin_flag": int(split_alpha_bk.n_with_origin_flag),
            "auc_in_situ_vs_p_concentrated": float(split_alpha_bk.auc_in_situ_vs_p_concentrated),
            "hard_assignment_accuracy": float(split_alpha_bk.hard_assignment_accuracy),
            "mean_p_concentrated_in_situ": float(split_alpha_bk.mean_p_concentrated_in_situ),
            "mean_p_concentrated_accreted": float(split_alpha_bk.mean_p_concentrated_accreted),
        },
        "reference_current_single_component": {
            "imf_family": reference_payload["summary"].imf_family,
            "radial_model": reference_payload["summary"].radial_model,
            "log_likelihood": float(reference_payload["summary"].log_likelihood),
            "bic": float(reference_payload["summary"].bic),
            "total_initial_count": float(reference_payload["summary"].total_initial_count),
        },
        "delta_log_likelihood_split_alpha_minus_shared": float(
            split_alpha_payload["summary"].log_likelihood - shared_payload["summary"].log_likelihood
        ),
        "delta_bic_split_alpha_minus_shared": float(
            split_alpha_payload["summary"].bic - shared_payload["summary"].bic
        ),
        "delta_log_likelihood_split_alpha_minus_single_null": float(
            split_alpha_payload["summary"].log_likelihood - single_payload["summary"].log_likelihood
        ),
        "delta_bic_split_alpha_minus_single_null": float(
            split_alpha_payload["summary"].bic - single_payload["summary"].bic
        ),
        "delta_bic_split_alpha_minus_current_logpoly3_single_component": float(
            split_alpha_payload["summary"].bic - reference_payload["summary"].bic
        ),
    }
    (outputs_tables / f"{DEFAULT_SPLIT_ALPHA_OUTPUT_PREFIX}_comparison_summary.json").write_text(
        json.dumps(comparison, indent=2)
    )


def summary_row_dict(summary) -> dict[str, object]:
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
