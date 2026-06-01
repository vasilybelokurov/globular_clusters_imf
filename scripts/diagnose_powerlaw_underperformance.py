from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def expected_projection_counts(
    point_intensity_grid: np.ndarray,
    log_mass_grid: np.ndarray,
    log_a_grid: np.ndarray,
    mass_bin_edges: np.ndarray,
    log_a_bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from globular_clusters_imf.plotting import rebin_expected_counts_2d
    from globular_clusters_imf.paper_assets import centers_to_edges

    log_mass_edges = centers_to_edges(np.asarray(log_mass_grid, dtype=float))
    log_a_edges = centers_to_edges(np.asarray(log_a_grid, dtype=float))
    cell_weights = (
        point_intensity_grid
        * np.diff(log_mass_edges)[:, None]
        * np.diff(log_a_edges)[None, :]
    )
    expected_2d = rebin_expected_counts_2d(
        cell_weights=cell_weights,
        log_mass_grid=log_mass_grid,
        log_a_grid=log_a_grid,
        mass_bin_edges=mass_bin_edges,
        log_a_bin_edges=log_a_bin_edges,
    )
    return expected_2d.sum(axis=1), expected_2d.sum(axis=0)


def compute_cluster_contribution_table(
    catalog: pd.DataFrame,
    *,
    model_name: str,
    result: dict[str, object],
) -> pd.DataFrame:
    context = result["final_context"]
    model = result["final_payload"]["model"]
    selection_data = np.clip(
        context.selection_interpolator(np.column_stack([context.log_mass_data, context.log_a_data])),
        1.0e-12,
        1.0,
    )
    table = pd.DataFrame(
        {
            "cluster_label": catalog["cluster_label"].to_numpy(),
            "log_initial_mass_msun": context.log_mass_data,
            "semi_major_axis_kpc": np.power(10.0, context.log_a_data),
            "log10_semi_major_axis_kpc": context.log_a_data,
            f"log_imf_density_{model_name}": np.log(np.clip(model["imf_density_data"], 1.0e-12, None)),
            f"log_radial_density_{model_name}": np.log(np.clip(model["radial_density_data"], 1.0e-12, None)),
            f"log_selection_data_{model_name}": np.log(selection_data),
            f"log_selection_fraction_{model_name}": np.full(
                len(context.log_mass_data),
                np.log(float(model["selection_fraction"])),
            ),
        }
    )
    table[f"log_profile_density_{model_name}"] = (
        table[f"log_imf_density_{model_name}"]
        + table[f"log_radial_density_{model_name}"]
        + table[f"log_selection_data_{model_name}"]
        - table[f"log_selection_fraction_{model_name}"]
    )
    return table


def build_mass_and_radius_projection_plot(
    axes,
    *,
    context,
    observed_mass_counts: np.ndarray,
    observed_radius_counts: np.ndarray,
    mass_bin_edges: np.ndarray,
    log_a_bin_edges: np.ndarray,
    schechter_mass_counts: np.ndarray,
    powerlaw_mass_counts: np.ndarray,
    schechter_radius_counts: np.ndarray,
    powerlaw_radius_counts: np.ndarray,
) -> None:
    mass_centers = 0.5 * (mass_bin_edges[:-1] + mass_bin_edges[1:])
    axes[0].stairs(observed_mass_counts, mass_bin_edges, color="black", linewidth=1.8, label="Observed")
    axes[0].plot(mass_centers, schechter_mass_counts, color="#d95f02", linewidth=2.0, label="Schechter")
    axes[0].plot(mass_centers, powerlaw_mass_counts, color="#1b9e77", linewidth=2.0, label=r"Power law, $\alpha=-2.225$")
    axes[0].set_xlabel(r"$\log_{10}(M_{\rm ini}/{\rm M}_\odot)$")
    axes[0].set_ylabel("Counts per bin")
    axes[0].set_title("Mass projection")
    axes[0].text(0.03, 0.96, "(a)", transform=axes[0].transAxes, ha="left", va="top")
    axes[0].legend(frameon=False, fontsize=8.5)

    a_edges_kpc = np.power(10.0, log_a_bin_edges)
    a_centers_kpc = np.power(10.0, 0.5 * (log_a_bin_edges[:-1] + log_a_bin_edges[1:]))
    axes[1].stairs(observed_radius_counts, a_edges_kpc, color="black", linewidth=1.8, label="Observed")
    axes[1].plot(a_centers_kpc, schechter_radius_counts, color="#d95f02", linewidth=2.0, label="Schechter")
    axes[1].plot(a_centers_kpc, powerlaw_radius_counts, color="#1b9e77", linewidth=2.0, label=r"Power law, $\alpha=-2.225$")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"Orbital semimajor axis $a$ [kpc]")
    axes[1].set_ylabel("Counts per bin")
    axes[1].set_title("Radius projection")
    axes[1].text(0.03, 0.96, "(b)", transform=axes[1].transAxes, ha="left", va="top")


def build_contribution_barplots(
    axes,
    *,
    cluster_table: pd.DataFrame,
) -> None:
    mass_edges = np.array([cluster_table["log_initial_mass_msun"].min(), 4.8, 5.4, 6.0, cluster_table["log_initial_mass_msun"].max()])
    radius_edges = np.array([cluster_table["semi_major_axis_kpc"].min(), 3.0, 15.0, 60.0, cluster_table["semi_major_axis_kpc"].max()])

    mass_labels = [r"$<4.8$", r"$4.8{-}5.4$", r"$5.4{-}6.0$", r"$>6.0$"]
    radius_labels = [r"$<3$", r"$3{-}15$", r"$15{-}60$", r"$>60$"]

    mass_groups = pd.cut(
        cluster_table["log_initial_mass_msun"],
        bins=mass_edges,
        labels=mass_labels,
        include_lowest=True,
        duplicates="drop",
    )
    radius_groups = pd.cut(
        cluster_table["semi_major_axis_kpc"],
        bins=radius_edges,
        labels=radius_labels,
        include_lowest=True,
        duplicates="drop",
    )

    mass_contrib = cluster_table.groupby(mass_groups, observed=False)["delta_log_profile_density"].sum().reindex(mass_labels)
    radius_contrib = cluster_table.groupby(radius_groups, observed=False)["delta_log_profile_density"].sum().reindex(radius_labels)

    axes[0].bar(mass_labels, mass_contrib.to_numpy(dtype=float), color="#7570b3")
    axes[0].axhline(0.0, color="0.7", linewidth=1.0)
    axes[0].set_ylabel(r"Total $\Delta\log\mathcal{L}$")
    axes[0].set_xlabel(r"$\log_{10}(M_{\rm ini}/{\rm M}_\odot)$ bin")
    axes[0].set_title("Where Schechter gains in mass")
    axes[0].text(0.03, 0.96, "(c)", transform=axes[0].transAxes, ha="left", va="top")

    axes[1].bar(radius_labels, radius_contrib.to_numpy(dtype=float), color="#7570b3")
    axes[1].axhline(0.0, color="0.7", linewidth=1.0)
    axes[1].set_ylabel(r"Total $\Delta\log\mathcal{L}$")
    axes[1].set_xlabel(r"$a$ [kpc] bin")
    axes[1].set_title("Where Schechter gains in radius")
    axes[1].text(0.03, 0.96, "(d)", transform=axes[1].transAxes, ha="left", va="top")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_root = project_root / "variants" / "powerlaw_underperformance_diagnostic"
    outputs_tables = output_root / "outputs" / "tables"
    outputs_figures = output_root / "outputs" / "figures"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    outputs_figures.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(project_root / ".cache"))
    (project_root / ".mplconfig").mkdir(parents=True, exist_ok=True)
    (project_root / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)

    from globular_clusters_imf.detectability_longitude_model import fit_single_component_detectability_em_with_abs_longitude
    from globular_clusters_imf.joint_model import JointModelSpec, compute_observed_intensity_grid
    from globular_clusters_imf.model import fit_catalog_models

    catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    prepared_catalog = fit_catalog_models(catalog, output_root)["catalog"]

    schechter_spec = JointModelSpec(imf_family="schechter", radial_model="logpoly3")
    powerlaw_spec = JointModelSpec(imf_family="powerlaw", radial_model="logpoly3")

    schechter_result = fit_single_component_detectability_em_with_abs_longitude(
        prepared_catalog,
        project_root=output_root,
        spec=schechter_spec,
        n_iterations=8,
    )
    powerlaw_reference_result = fit_single_component_detectability_em_with_abs_longitude(
        prepared_catalog,
        project_root=output_root,
        spec=powerlaw_spec,
        n_iterations=8,
    )

    best_alpha = -2.225
    alpha_scan = np.array([-2.6, -2.525, -2.45, -2.375, -2.3, -2.225], dtype=float)
    warm_result: dict[str, object] | None = None
    best_powerlaw_result = None
    best_powerlaw_label = "profile_scan_warm_start"
    for alpha in alpha_scan:
        start_completeness = (
            powerlaw_reference_result["final_completeness_raw_parameters"]
            if warm_result is None
            else warm_result["final_completeness_raw_parameters"]
        )
        start_radial = (
            powerlaw_reference_result["final_payload"]["raw_parameters"][1:]
            if warm_result is None
            else warm_result["final_payload"]["raw_parameters"][1:]
        )
        result = fit_single_component_detectability_em_with_abs_longitude(
            prepared_catalog,
            project_root=output_root,
            spec=powerlaw_spec,
            n_iterations=6,
            fixed_imf_params=np.array([alpha], dtype=float),
            start_completeness_raw_parameters=np.asarray(start_completeness, dtype=float),
            start_radial_params=np.asarray(start_radial, dtype=float),
        )
        warm_result = result
        if np.isclose(alpha, best_alpha):
            best_powerlaw_result = result
    if best_powerlaw_result is None:
        raise RuntimeError("Failed to recover the profiled power-law scan point at alpha=-2.225.")

    schechter_context = schechter_result["final_context"]
    powerlaw_context = best_powerlaw_result["final_context"]
    schechter_model = schechter_result["final_payload"]["model"]
    powerlaw_model = best_powerlaw_result["final_payload"]["model"]

    schechter_cluster_table = compute_cluster_contribution_table(
        prepared_catalog,
        model_name="schechter",
        result=schechter_result,
    )
    powerlaw_cluster_table = compute_cluster_contribution_table(
        prepared_catalog,
        model_name="powerlaw",
        result=best_powerlaw_result,
    )
    cluster_table = schechter_cluster_table.merge(
        powerlaw_cluster_table.drop(columns=["cluster_label", "log_initial_mass_msun", "semi_major_axis_kpc", "log10_semi_major_axis_kpc"]),
        left_index=True,
        right_index=True,
    )
    cluster_table["delta_log_imf_density"] = (
        cluster_table["log_imf_density_schechter"] - cluster_table["log_imf_density_powerlaw"]
    )
    cluster_table["delta_log_radial_density"] = (
        cluster_table["log_radial_density_schechter"] - cluster_table["log_radial_density_powerlaw"]
    )
    cluster_table["delta_log_selection_shape"] = (
        cluster_table["log_selection_data_schechter"]
        - cluster_table["log_selection_data_powerlaw"]
        - cluster_table["log_selection_fraction_schechter"]
        + cluster_table["log_selection_fraction_powerlaw"]
    )
    cluster_table["delta_log_profile_density"] = (
        cluster_table["delta_log_imf_density"]
        + cluster_table["delta_log_radial_density"]
        + cluster_table["delta_log_selection_shape"]
    )

    delta_logl_exact = float(cluster_table["delta_log_profile_density"].sum())
    delta_logl_reported = float(
        schechter_result["final_payload"]["summary"].log_likelihood
        - best_powerlaw_result["final_payload"]["summary"].log_likelihood
    )

    mass_bin_edges = np.linspace(schechter_context.log_mass_grid[0], schechter_context.log_mass_grid[-1], 13)
    log_a_bin_edges = np.linspace(schechter_context.log_a_grid[0], schechter_context.log_a_grid[-1], 10)
    observed_2d, _, _ = np.histogram2d(
        schechter_context.log_mass_data,
        schechter_context.log_a_data,
        bins=[mass_bin_edges, log_a_bin_edges],
    )
    observed_mass_counts = observed_2d.sum(axis=1)
    observed_radius_counts = observed_2d.sum(axis=0)

    schechter_point_intensity_grid = compute_observed_intensity_grid(
        schechter_model["imf_density_grid"],
        schechter_model["radial_density_grid"],
        schechter_context.selection_probability_grid,
        schechter_model["total_initial_count"],
    )
    powerlaw_point_intensity_grid = compute_observed_intensity_grid(
        powerlaw_model["imf_density_grid"],
        powerlaw_model["radial_density_grid"],
        powerlaw_context.selection_probability_grid,
        powerlaw_model["total_initial_count"],
    )
    schechter_mass_counts, schechter_radius_counts = expected_projection_counts(
        schechter_point_intensity_grid,
        schechter_context.log_mass_grid,
        schechter_context.log_a_grid,
        mass_bin_edges,
        log_a_bin_edges,
    )
    powerlaw_mass_counts, powerlaw_radius_counts = expected_projection_counts(
        powerlaw_point_intensity_grid,
        powerlaw_context.log_mass_grid,
        powerlaw_context.log_a_grid,
        mass_bin_edges,
        log_a_bin_edges,
    )

    # Coarse 2D Poisson-cell diagnostic.
    from globular_clusters_imf.paper_assets import centers_to_edges
    from globular_clusters_imf.plotting import rebin_expected_counts_2d

    schechter_log_mass_edges = centers_to_edges(np.asarray(schechter_context.log_mass_grid, dtype=float))
    schechter_log_a_edges = centers_to_edges(np.asarray(schechter_context.log_a_grid, dtype=float))
    schechter_cell_weights = (
        schechter_point_intensity_grid
        * np.diff(schechter_log_mass_edges)[:, None]
        * np.diff(schechter_log_a_edges)[None, :]
    )
    schechter_expected_2d = rebin_expected_counts_2d(
        cell_weights=schechter_cell_weights,
        log_mass_grid=schechter_context.log_mass_grid,
        log_a_grid=schechter_context.log_a_grid,
        mass_bin_edges=mass_bin_edges,
        log_a_bin_edges=log_a_bin_edges,
    )
    powerlaw_log_mass_edges = centers_to_edges(np.asarray(powerlaw_context.log_mass_grid, dtype=float))
    powerlaw_log_a_edges = centers_to_edges(np.asarray(powerlaw_context.log_a_grid, dtype=float))
    powerlaw_cell_weights = (
        powerlaw_point_intensity_grid
        * np.diff(powerlaw_log_mass_edges)[:, None]
        * np.diff(powerlaw_log_a_edges)[None, :]
    )
    powerlaw_expected_2d = rebin_expected_counts_2d(
        cell_weights=powerlaw_cell_weights,
        log_mass_grid=powerlaw_context.log_mass_grid,
        log_a_grid=powerlaw_context.log_a_grid,
        mass_bin_edges=mass_bin_edges,
        log_a_bin_edges=log_a_bin_edges,
    )
    observed_int = observed_2d.astype(float)
    schechter_cell_logl = observed_int * np.log(np.clip(schechter_expected_2d, 1.0e-12, None)) - schechter_expected_2d
    powerlaw_cell_logl = observed_int * np.log(np.clip(powerlaw_expected_2d, 1.0e-12, None)) - powerlaw_expected_2d
    cell_delta = schechter_cell_logl - powerlaw_cell_logl

    dominant_bins = []
    for i_mass in range(cell_delta.shape[0]):
        for i_a in range(cell_delta.shape[1]):
            dominant_bins.append(
                {
                    "mass_left_edge": float(mass_bin_edges[i_mass]),
                    "mass_right_edge": float(mass_bin_edges[i_mass + 1]),
                    "a_left_edge_kpc": float(np.power(10.0, log_a_bin_edges[i_a])),
                    "a_right_edge_kpc": float(np.power(10.0, log_a_bin_edges[i_a + 1])),
                    "observed_count": float(observed_int[i_mass, i_a]),
                    "schechter_expected_count": float(schechter_expected_2d[i_mass, i_a]),
                    "powerlaw_expected_count": float(powerlaw_expected_2d[i_mass, i_a]),
                    "delta_logl_cell": float(cell_delta[i_mass, i_a]),
                }
            )
    dominant_bins_table = pd.DataFrame(dominant_bins).sort_values("delta_logl_cell", ascending=False)

    top_clusters = cluster_table.sort_values("delta_log_profile_density", ascending=False).head(20)
    bottom_clusters = cluster_table.sort_values("delta_log_profile_density", ascending=True).head(20)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    build_mass_and_radius_projection_plot(
        axes[0],
        context=schechter_context,
        observed_mass_counts=observed_mass_counts,
        observed_radius_counts=observed_radius_counts,
        mass_bin_edges=mass_bin_edges,
        log_a_bin_edges=log_a_bin_edges,
        schechter_mass_counts=schechter_mass_counts,
        powerlaw_mass_counts=powerlaw_mass_counts,
        schechter_radius_counts=schechter_radius_counts,
        powerlaw_radius_counts=powerlaw_radius_counts,
    )
    build_contribution_barplots(axes[1], cluster_table=cluster_table)
    fig.tight_layout()
    fig.savefig(outputs_figures / "powerlaw_underperformance_diagnostic.png", dpi=220)
    plt.close(fig)

    cluster_table.to_csv(outputs_tables / "powerlaw_underperformance_cluster_contributions.csv", index=False)
    dominant_bins_table.to_csv(outputs_tables / "powerlaw_underperformance_2d_cell_contributions.csv", index=False)
    top_clusters.to_csv(outputs_tables / "powerlaw_underperformance_top_clusters.csv", index=False)
    bottom_clusters.to_csv(outputs_tables / "powerlaw_underperformance_bottom_clusters.csv", index=False)

    mass_projection_table = pd.DataFrame(
        {
            "mass_left_edge": mass_bin_edges[:-1],
            "mass_right_edge": mass_bin_edges[1:],
            "observed_count": observed_mass_counts,
            "schechter_expected_count": schechter_mass_counts,
            "powerlaw_expected_count": powerlaw_mass_counts,
            "delta_expected_schechter_minus_powerlaw": schechter_mass_counts - powerlaw_mass_counts,
        }
    )
    radius_projection_table = pd.DataFrame(
        {
            "log_a_left_edge": log_a_bin_edges[:-1],
            "log_a_right_edge": log_a_bin_edges[1:],
            "a_left_edge_kpc": np.power(10.0, log_a_bin_edges[:-1]),
            "a_right_edge_kpc": np.power(10.0, log_a_bin_edges[1:]),
            "observed_count": observed_radius_counts,
            "schechter_expected_count": schechter_radius_counts,
            "powerlaw_expected_count": powerlaw_radius_counts,
            "delta_expected_schechter_minus_powerlaw": schechter_radius_counts - powerlaw_radius_counts,
        }
    )
    mass_projection_table.to_csv(outputs_tables / "powerlaw_underperformance_mass_projection.csv", index=False)
    radius_projection_table.to_csv(outputs_tables / "powerlaw_underperformance_radius_projection.csv", index=False)

    summary = {
        "schechter_log_likelihood": float(schechter_result["final_payload"]["summary"].log_likelihood),
        "powerlaw_log_likelihood": float(best_powerlaw_result["final_payload"]["summary"].log_likelihood),
        "delta_log_likelihood_schechter_minus_powerlaw": delta_logl_reported,
        "delta_log_likelihood_exact_cluster_sum": delta_logl_exact,
        "powerlaw_start_choice": best_powerlaw_label,
        "powerlaw_alpha_fixed": best_alpha,
        "schechter_total_initial_count": float(schechter_model["total_initial_count"]),
        "powerlaw_total_initial_count": float(powerlaw_model["total_initial_count"]),
        "schechter_selection_fraction": float(schechter_model["selection_fraction"]),
        "powerlaw_selection_fraction": float(powerlaw_model["selection_fraction"]),
        "schechter_mean_detectability": float(
            schechter_result["iteration_history_table"].iloc[-1]["completeness_mean"]
        ),
        "powerlaw_mean_detectability": float(
            best_powerlaw_result["iteration_history_table"].iloc[-1]["completeness_mean"]
        ),
        "delta_imf_term": float(cluster_table["delta_log_imf_density"].sum()),
        "delta_radial_term": float(cluster_table["delta_log_radial_density"].sum()),
        "delta_selection_shape_term": float(cluster_table["delta_log_selection_shape"].sum()),
        "top_2d_bins": dominant_bins_table.head(10).to_dict(orient="records"),
    }
    (outputs_tables / "powerlaw_underperformance_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
