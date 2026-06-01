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

from globular_clusters_imf.blind_chemistry_mixture_model import (
    DEFAULT_OUTPUT_PREFIX,
    build_posterior_probability_table,
    fit_blind_powerlaw_a_chemistry_models,
)
from globular_clusters_imf.catalog import (
    attach_local_gc_chemistry_to_baumgardt_catalog,
    export_local_gc_chemistry_markers,
)
from globular_clusters_imf.joint_model import JointModelSpec, fit_single_joint_model


def main() -> None:
    project_root = PROJECT_ROOT
    output_root = project_root / "variants" / DEFAULT_OUTPUT_PREFIX
    output_root.mkdir(parents=True, exist_ok=True)

    chemistry_markers = export_local_gc_chemistry_markers(project_root)
    catalog = attach_local_gc_chemistry_to_baumgardt_catalog(
        project_root,
        chemistry_markers=chemistry_markers,
    )

    results = fit_blind_powerlaw_a_chemistry_models(
        catalog=catalog,
        project_root=project_root,
        output_root=output_root,
        output_prefix=DEFAULT_OUTPUT_PREFIX,
    )

    single_payload = find_payload(results["all_payloads"], "single_powerlaw_radial")
    shared_payload = find_payload(results["all_payloads"], "two_component_powerlaw_mixture")
    split_alpha_payload = find_payload(results["all_payloads"], "two_component_powerlaw_mixture_split_alpha")
    posterior = build_posterior_probability_table(split_alpha_payload, catalog)
    if "has_any_chemistry" not in posterior.columns:
        posterior["has_any_chemistry"] = posterior["has_mgfe"] | posterior["has_alfe"]
    outputs_tables = output_root / "outputs" / "tables"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    posterior.to_csv(
        outputs_tables / f"{DEFAULT_OUTPUT_PREFIX}_best_split_alpha_posterior_probabilities.csv",
        index=False,
    )

    selection_payload = results["selection_payload"]
    reference_summary = json.loads(
        (
            project_root / "outputs" / "tables" / "joint_fixed_survival_detectability_abs_longitude_em_summary.json"
        ).read_text()
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
        posterior=posterior,
    )
    write_summary(
        output_root=output_root,
        results=results,
        single_payload=single_payload,
        shared_payload=shared_payload,
        split_alpha_payload=split_alpha_payload,
        chemistry_markers=chemistry_markers,
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
    posterior: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    ax_chem, ax_imf, ax_radial, ax_post = axes.flat

    posterior_with_both = posterior.loc[posterior["has_mgfe_and_alfe"]].copy()
    marker_map = {
        1: ("o", "in-situ"),
        0: ("s", "accreted"),
    }
    for origin_flag, (marker, label) in marker_map.items():
        subset = posterior_with_both.loc[posterior_with_both["origin_flag"] == origin_flag]
        if subset.empty:
            continue
        scatter = ax_chem.scatter(
            subset["alfe_combined"],
            subset["mgfe_combined"],
            c=subset["p_concentrated"],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            s=42,
            marker=marker,
            edgecolor="k",
            linewidth=0.35,
            alpha=0.9,
            label=label,
        )
    al_grid = np.linspace(
        float(np.nanmin(posterior_with_both["alfe_combined"])) - 0.05,
        float(np.nanmax(posterior_with_both["alfe_combined"])) + 0.05,
        200,
    )
    chemistry = split_alpha_payload["model"]["chemistry_parameters"]
    slope = float(chemistry["shared_mg_vs_al_slope"])
    for label, color in (("concentrated", "#d95f02"), ("extended", "#7570b3")):
        params = chemistry[label]
        if not np.isfinite(params["mu_alfe"]):
            continue
        mg_line = params["mu_mgfe"] + slope * (al_grid - params["mu_alfe"])
        ax_chem.plot(al_grid, mg_line, color=color, lw=1.8, ls="--")
        ax_chem.scatter(
            [params["mu_alfe"]],
            [params["mu_mgfe"]],
            s=80,
            color=color,
            edgecolor="k",
            linewidth=0.4,
            zorder=5,
        )
    ax_chem.set_xlabel(r"[Al/Fe]")
    ax_chem.set_ylabel(r"[Mg/Fe]")
    ax_chem.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.colorbar(scatter, ax=ax_chem, label=r"$P({\rm concentrated}\mid M_{\rm ini}, a, {\rm Mg, Al})$")

    mass_grid = np.power(10.0, selection_context.log_mass_grid)
    ax_imf.plot(
        mass_grid,
        reference_payload["model"]["total_initial_count"] * reference_payload["model"]["imf_density_grid"],
        color="0.75",
        lw=3.2,
        zorder=0,
        label="Current 1-component",
    )
    ax_imf.plot(
        mass_grid,
        shared_payload["model"]["total_initial_count"] * shared_payload["model"]["imf_density_grid"],
        color="#1b9e77",
        lw=2.2,
        ls=":",
        label="Blind shared IMF + chem",
    )
    total_count_split = float(split_alpha_payload["model"]["total_initial_count"])
    mix_fraction = float(split_alpha_payload["model"]["mix_fraction_concentrated"])
    ax_imf.plot(
        mass_grid,
        total_count_split * split_alpha_payload["model"]["imf_density_grid"],
        color="0.15",
        lw=2.2,
        label="Blind split-$\\alpha$ + chem",
    )
    ax_imf.plot(
        mass_grid,
        total_count_split
        * mix_fraction
        * split_alpha_payload["model"]["component_imf_density_grid"]["concentrated"],
        color="#d95f02",
        lw=2.0,
        ls="--",
        label="Concentrated IMF",
    )
    ax_imf.plot(
        mass_grid,
        total_count_split
        * (1.0 - mix_fraction)
        * split_alpha_payload["model"]["component_imf_density_grid"]["extended"],
        color="#7570b3",
        lw=2.0,
        ls="--",
        label="Extended IMF",
    )
    ax_imf.set_xscale("log")
    ax_imf.set_yscale("log")
    ax_imf.set_xlabel(r"$M_{\rm ini}\ [{\rm M}_\odot]$")
    ax_imf.set_ylabel(r"$dN/d\log M_{\rm ini}$")
    ax_imf.legend(frameon=False, fontsize=8.3, loc="upper right")

    a_grid = np.power(10.0, selection_context.log_a_grid)
    ax_radial.plot(
        a_grid,
        reference_payload["model"]["total_initial_count"] * reference_payload["model"]["radial_density_grid"],
        color="0.75",
        lw=3.2,
        zorder=0,
        label="Current 1-component",
    )
    ax_radial.plot(
        a_grid,
        shared_payload["model"]["total_initial_count"] * shared_payload["model"]["mixture_radial_density_grid"],
        color="#1b9e77",
        lw=2.2,
        ls=":",
        label="Blind shared IMF + chem",
    )
    ax_radial.plot(
        a_grid,
        total_count_split * split_alpha_payload["model"]["mixture_radial_density_grid"],
        color="0.15",
        lw=2.2,
        label="Blind split-$\\alpha$ + chem",
    )
    ax_radial.plot(
        a_grid,
        total_count_split
        * mix_fraction
        * split_alpha_payload["model"]["component_radial_density_grid"]["concentrated"],
        color="#d95f02",
        lw=2.0,
        ls="--",
        label="Concentrated radial",
    )
    ax_radial.plot(
        a_grid,
        total_count_split
        * (1.0 - mix_fraction)
        * split_alpha_payload["model"]["component_radial_density_grid"]["extended"],
        color="#7570b3",
        lw=2.0,
        ls="--",
        label="Extended radial",
    )
    ax_radial.set_xscale("log")
    ax_radial.set_yscale("log")
    ax_radial.set_xlabel(r"$a\ [{\rm kpc}]$")
    ax_radial.set_ylabel(r"$dN/d\log a$")
    ax_radial.legend(frameon=False, fontsize=8.3, loc="upper right")

    for origin_flag, (marker, label) in marker_map.items():
        subset = posterior.loc[posterior["origin_flag"] == origin_flag]
        if subset.empty:
            continue
        ax_post.scatter(
            subset["semi_major_axis_kpc"],
            subset["p_concentrated"],
            s=np.where(subset["has_any_chemistry"], 40, 20),
            alpha=0.8,
            marker=marker,
            edgecolor="none",
            color="#d95f02" if origin_flag == 1 else "#1f77b4",
            label=label,
        )
    order = np.argsort(posterior["semi_major_axis_kpc"].to_numpy())
    ax_post.plot(
        posterior["semi_major_axis_kpc"].to_numpy()[order],
        posterior["p_concentrated"].to_numpy()[order],
        color="0.45",
        lw=1.0,
        alpha=0.6,
    )
    ax_post.axhline(0.5, color="0.7", lw=1.0, ls=":")
    ax_post.set_xscale("log")
    ax_post.set_ylim(-0.02, 1.02)
    ax_post.set_xlabel(r"$a\ [{\rm kpc}]$")
    ax_post.set_ylabel(r"$P({\rm concentrated}\mid M_{\rm ini}, a, {\rm Mg, Al})$")
    ax_post.legend(frameon=False, fontsize=8.5, loc="lower left")

    for axis, panel_label in zip(axes.flat, ("(a)", "(b)", "(c)", "(d)"), strict=True):
        axis.text(0.03, 0.97, panel_label, transform=axis.transAxes, ha="left", va="top", fontsize=11)

    fig.tight_layout()
    outputs_figures = output_root / "outputs" / "figures"
    outputs_figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs_figures / f"{DEFAULT_OUTPUT_PREFIX}_diagnostic.pdf", bbox_inches="tight")
    fig.savefig(outputs_figures / f"{DEFAULT_OUTPUT_PREFIX}_diagnostic.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    output_root: Path,
    results: dict[str, object],
    single_payload: dict[str, object],
    shared_payload: dict[str, object],
    split_alpha_payload: dict[str, object],
    chemistry_markers: pd.DataFrame,
) -> None:
    summary_table = results["summary_table"]
    outputs_tables = output_root / "outputs" / "tables"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    comparison = {
        "n_clusters_total": int(len(results["posterior_table"])),
        "n_with_mgfe": int(results["posterior_table"]["has_mgfe"].sum()),
        "n_with_alfe": int(results["posterior_table"]["has_alfe"].sum()),
        "n_with_mgfe_and_alfe": int(results["posterior_table"]["has_mgfe_and_alfe"].sum()),
        "mgfe_offset_dex": float(chemistry_markers["mgfe_offset_dex"].iloc[0]),
        "mgfe_offset_scatter_dex": float(chemistry_markers["mgfe_offset_scatter_dex"].iloc[0]),
        "alfe_offset_dex": float(chemistry_markers["alfe_offset_dex"].iloc[0]),
        "alfe_offset_scatter_dex": float(chemistry_markers["alfe_offset_scatter_dex"].iloc[0]),
        "single_model": json.loads(summary_table.loc[summary_table["model_class"] == "single_powerlaw_radial"].iloc[0].to_json()),
        "shared_imf_model": json.loads(summary_table.loc[summary_table["model_class"] == "two_component_powerlaw_mixture"].iloc[0].to_json()),
        "split_alpha_model": json.loads(summary_table.loc[summary_table["model_class"] == "two_component_powerlaw_mixture_split_alpha"].iloc[0].to_json()),
        "delta_log_likelihood_shared_minus_single": float(
            shared_payload["summary"].log_likelihood - single_payload["summary"].log_likelihood
        ),
        "delta_bic_shared_minus_single": float(
            shared_payload["summary"].bic - single_payload["summary"].bic
        ),
        "delta_log_likelihood_split_alpha_minus_shared": float(
            split_alpha_payload["summary"].log_likelihood - shared_payload["summary"].log_likelihood
        ),
        "delta_bic_split_alpha_minus_shared": float(
            split_alpha_payload["summary"].bic - shared_payload["summary"].bic
        ),
        "delta_log_likelihood_split_alpha_minus_single": float(
            split_alpha_payload["summary"].log_likelihood - single_payload["summary"].log_likelihood
        ),
        "delta_bic_split_alpha_minus_single": float(
            split_alpha_payload["summary"].bic - single_payload["summary"].bic
        ),
        "bk_comparison_best_model": json.loads(json.dumps(results["bk_comparison"], default=lambda value: value.__dict__)),
    }
    (outputs_tables / f"{DEFAULT_OUTPUT_PREFIX}_comparison_summary.json").write_text(
        json.dumps(comparison, indent=2)
    )


if __name__ == "__main__":
    main()
