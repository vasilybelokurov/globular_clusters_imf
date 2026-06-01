from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from globular_clusters_imf.blind_mg_only_split_alpha_minimal_model import (
    DEFAULT_OUTPUT_PREFIX,
    fit_mg_only_split_alpha_models,
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

    results = fit_mg_only_split_alpha_models(
        catalog=catalog,
        project_root=project_root,
        output_root=output_root,
        output_prefix=DEFAULT_OUTPUT_PREFIX,
    )
    build_diagnostic_figure(output_root=output_root, results=results)


def build_diagnostic_figure(
    output_root: Path,
    results: dict[str, object],
) -> None:
    base_payload = results["base_payload"]
    mg_context = results["mg_context"]
    shared_posterior = results["shared_posterior"]
    split_posterior = results["split_posterior"]
    summary_table = results["summary_table"].set_index("model_class")
    split_payload = results["split_payload"]
    imf_grid_table = results["imf_grid_table"]
    mg_grid_table = results["mg_grid_table"]
    context = base_payload["context"]
    split_model = split_payload["model"]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8))
    ax_scatter, ax_mix, ax_imf, ax_bk = axes.flat

    with_mg = split_posterior.loc[split_posterior["has_mgfe"]].copy()
    marker_map = {1: ("o", "in-situ"), 0: ("s", "accreted")}
    scatter = None
    for origin_flag, (marker, label) in marker_map.items():
        subset = with_mg.loc[with_mg["origin_flag"] == origin_flag]
        if subset.empty:
            continue
        scatter = ax_scatter.scatter(
            subset["semi_major_axis_kpc"],
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
    ax_scatter.set_xscale("log")
    ax_scatter.set_xlabel(r"$a\ [{\rm kpc}]$")
    ax_scatter.set_ylabel(r"[Mg/Fe]")
    ax_scatter.legend(frameon=False, fontsize=8.3, loc="lower left")
    if scatter is not None:
        fig.colorbar(scatter, ax=ax_scatter, label=r"$P({\rm concentrated}\mid a, M_{\rm ini}, {\rm Mg})$")

    a_grid = np.power(10.0, context.log_a_grid)
    ax_mix.plot(
        a_grid,
        np.asarray(split_model["w_grid"], dtype=float),
        color="#1b9e77",
        lw=2.4,
        label=r"$w(a)$",
    )
    order = np.argsort(split_posterior["semi_major_axis_kpc"].to_numpy())
    ax_mix.scatter(
        split_posterior["semi_major_axis_kpc"].to_numpy()[order],
        split_posterior["p_concentrated"].to_numpy()[order],
        s=14,
        color="0.15",
        alpha=0.55,
        label=r"split-$\alpha$ posterior",
    )
    ax_mix.scatter(
        shared_posterior["semi_major_axis_kpc"].to_numpy()[order],
        shared_posterior["p_concentrated"].to_numpy()[order],
        s=12,
        color="#d95f02",
        alpha=0.45,
        label="shared-$\\alpha$ posterior",
    )
    ax_mix.set_xscale("log")
    ax_mix.set_ylim(0.0, 1.0)
    ax_mix.set_xlabel(r"$a\ [{\rm kpc}]$")
    ax_mix.set_ylabel("Concentrated fraction")
    ax_mix.legend(frameon=False, fontsize=8.1, loc="best")

    mass_grid = np.power(10.0, imf_grid_table["log_initial_mass_msun"].to_numpy())
    ax_imf.plot(
        mass_grid,
        imf_grid_table["baseline_imf_density"],
        color="0.2",
        lw=2.4,
        label="Baseline shared IMF",
    )
    ax_imf.plot(
        mass_grid,
        imf_grid_table["concentrated_imf_density"],
        color="#d95f02",
        lw=2.1,
        ls="--",
        label="Concentrated IMF",
    )
    ax_imf.plot(
        mass_grid,
        imf_grid_table["extended_imf_density"],
        color="#7570b3",
        lw=2.1,
        ls="--",
        label="Extended IMF",
    )
    ax_imf.set_xscale("log")
    ax_imf.set_yscale("log")
    ax_imf.set_xlabel(r"$M_{\rm ini}\ [{\rm M}_\odot]$")
    ax_imf.set_ylabel(r"$\phi(M_{\rm ini})$")
    ax_imf.legend(frameon=False, fontsize=8.1, loc="best")

    group_positions = {"accreted": 0, "in-situ": 1}
    for origin_flag, label in ((0, "accreted"), (1, "in-situ")):
        subset = split_posterior.loc[split_posterior["origin_flag"] == origin_flag, "p_concentrated"]
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
        ax_bk.hlines(
            float(subset.median()),
            group_positions[label] - 0.22,
            group_positions[label] + 0.22,
            color="0.1",
            lw=2.1,
        )
    ax_bk.set_xlim(-0.5, 1.5)
    ax_bk.set_ylim(0.0, 1.0)
    ax_bk.set_xticks([0, 1], ["accreted", "in-situ"])
    ax_bk.set_ylabel(r"$P({\rm concentrated})$")

    delta_logl = float(
        summary_table.loc["two_component_mg_split_alpha_mixture", "joint_log_likelihood"]
        - summary_table.loc["two_component_mg_mixture", "joint_log_likelihood"]
    )
    delta_bic = float(
        summary_table.loc["two_component_mg_split_alpha_mixture", "bic"]
        - summary_table.loc["two_component_mg_mixture", "bic"]
    )
    fig.suptitle(
        (
            "Mg-only blind 2-component model with split low-mass slopes\n"
            f"Fixed radial structure and fixed $M_c$; ΔlogL={delta_logl:+.2f}, ΔBIC={delta_bic:+.2f}"
        ),
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    outputs_figures = output_root / "outputs" / "figures"
    outputs_figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs_figures / f"{DEFAULT_OUTPUT_PREFIX}_diagnostic.pdf")
    fig.savefig(outputs_figures / f"{DEFAULT_OUTPUT_PREFIX}_diagnostic.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
