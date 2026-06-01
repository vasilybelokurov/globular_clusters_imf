from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import interpolate, stats

from .detectability_longitude_model import (
    fit_detectability_corrected_single_component_models_with_abs_longitude,
    fit_split_alpha_two_component_detectability_em_models_with_abs_longitude,
    fit_shared_imf_two_component_detectability_em_models_with_abs_longitude,
)
from .joint_model import (
    compute_profile_likelihood_radial_birth_band,
    compute_profile_likelihood_imf_band,
    compute_observed_intensity_grid,
    estimate_best_model_uncertainty,
    fit_fixed_survival_joint_models,
    unpack_model,
)
from .model import fit_catalog_models
from .plotting import (
    build_gc_detectability_histogram_tables,
    centers_to_edges,
    plot_survivability_map,
    rebin_expected_counts_2d,
)
from .smooth_survivability import build_smooth_survivability_grid

PAPER_LOG_MASS_MIN = 4.0


def _restrict_log_mass_support(
    log_mass_grid: np.ndarray,
    values: np.ndarray,
    log_mass_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(log_mass_grid, dtype=float)
    y = np.asarray(values, dtype=float)
    if log_mass_min <= float(x[0]):
        return x.copy(), y.copy()
    if log_mass_min >= float(x[-1]):
        return np.asarray([float(x[-1])], dtype=float), np.asarray([float(y[-1])], dtype=float)
    mask = x > log_mass_min
    y0 = float(np.interp(log_mass_min, x, y))
    return (
        np.concatenate(([float(log_mass_min)], x[mask])),
        np.concatenate(([y0], y[mask])),
    )


def integrate_density_above_log_mass(
    log_mass_grid: np.ndarray,
    density_per_dex: np.ndarray,
    log_mass_min: float,
) -> float:
    x, y = _restrict_log_mass_support(log_mass_grid, density_per_dex, log_mass_min)
    return float(np.trapezoid(y, x))


def mean_cluster_initial_mass_above_log_mass(
    log_mass_grid: np.ndarray,
    imf_density_grid: np.ndarray,
    log_mass_min: float,
) -> float:
    x, y = _restrict_log_mass_support(log_mass_grid, imf_density_grid, log_mass_min)
    normalization = max(float(np.trapezoid(y, x)), 1.0e-12)
    return float(np.trapezoid(np.power(10.0, x) * y, x) / normalization)


def total_initial_count_above_log_mass(
    total_initial_count: float,
    log_mass_grid: np.ndarray,
    imf_density_grid: np.ndarray,
    log_mass_min: float,
) -> float:
    fraction = integrate_density_above_log_mass(log_mass_grid, imf_density_grid, log_mass_min)
    return float(total_initial_count * fraction)


def total_initial_stellar_mass_above_log_mass(
    total_initial_count: float,
    log_mass_grid: np.ndarray,
    imf_density_grid: np.ndarray,
    log_mass_min: float,
) -> float:
    fraction = integrate_density_above_log_mass(log_mass_grid, imf_density_grid, log_mass_min)
    mean_mass_above = mean_cluster_initial_mass_above_log_mass(log_mass_grid, imf_density_grid, log_mass_min)
    return float(total_initial_count * fraction * mean_mass_above)

def build_paper_assets(project_root: Path) -> dict[str, object]:
    catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog.csv"

    catalog = pd.read_csv(catalog_path)
    catalog_results = fit_catalog_models(catalog, project_root)
    fit_catalog = catalog_results["catalog"]
    smooth_survivability_map = build_smooth_survivability_grid(fit_catalog, eta_t=1.0, surface_model="logistic")
    eta_boundary_maps = {
        0.5: build_smooth_survivability_grid(fit_catalog, eta_t=0.5, surface_model="logistic"),
        2.0: build_smooth_survivability_grid(fit_catalog, eta_t=2.0, surface_model="logistic"),
    }
    joint_results = fit_fixed_survival_joint_models(fit_catalog, project_root)
    detectability_comparison = fit_detectability_corrected_single_component_models_with_abs_longitude(
        fit_catalog,
        project_root,
    )
    detectability_results = detectability_comparison["best_result"]
    detectability_uncertainty = estimate_best_model_uncertainty(
        best_payload=detectability_results["final_payload"],
        context=detectability_results["final_context"],
    )
    flexible_imf_overlay = load_precomputed_flexible_imf_overlay(
        project_root=project_root,
        log_mass_grid=np.asarray(detectability_results["final_context"].log_mass_grid, dtype=float),
    )
    family_profile_scan_results = load_precomputed_single_component_family_profile_scan_results(
        project_root=project_root,
    )
    shared_results = fit_shared_imf_two_component_detectability_em_models_with_abs_longitude(
        fit_catalog,
        project_root,
        fixed_effective_completeness_grid=detectability_results["final_effective_completeness_grid"],
        fixed_completeness_bin_grid=detectability_results["final_completeness_bin_grid"],
        fixed_completeness_raw_parameters=detectability_results["final_completeness_raw_parameters"],
    )
    split_alpha_results = fit_split_alpha_two_component_detectability_em_models_with_abs_longitude(
        fit_catalog,
        project_root,
        fixed_effective_completeness_grid=detectability_results["final_effective_completeness_grid"],
        fixed_completeness_bin_grid=detectability_results["final_completeness_bin_grid"],
        fixed_completeness_raw_parameters=detectability_results["final_completeness_raw_parameters"],
    )

    paper_dir = project_root / "paper"
    figures_dir = paper_dir / "figures"
    tables_dir = paper_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    plot_catalog_mass_semimajor_axis_overview_for_paper(
        fit_catalog,
        smooth_survivability_map,
        figures_dir / "catalog_mass_semimajor_axis_overview.pdf",
        eta_boundary_maps=eta_boundary_maps,
    )
    plot_survivability_map(
        fit_catalog,
        catalog_results["survivability_map"],
        figures_dir / "survivability_plane.pdf",
    )
    plot_detectability_counts_for_paper(
        fit_catalog,
        figures_dir / "detectability_counts.pdf",
        longitude_limit_deg=30.0,
    )
    plot_single_component_family_profile_scan_for_paper(
        family_profile_scan_results,
        figures_dir / "single_component_model_performance.pdf",
    )
    plot_best_single_component_summary_for_paper(
        fit_catalog,
        context=detectability_results["final_context"],
        best_payload=detectability_results["final_payload"],
        uncertainty_payload=detectability_uncertainty,
        output_path=figures_dir / "best_single_component_summary.pdf",
    )
    plot_single_component_profiles_for_paper(
        baseline_joint_results=joint_results,
        detectability_result=detectability_results,
        uncertainty_payload=detectability_uncertainty,
        flexible_imf_overlay=flexible_imf_overlay,
        output_path=figures_dir / "single_component_profiles.pdf",
    )
    plot_single_component_radial_profile_for_paper(
        baseline_joint_results=joint_results,
        detectability_result=detectability_results,
        uncertainty_payload=detectability_uncertainty,
        flexible_imf_overlay=flexible_imf_overlay,
        output_path=figures_dir / "single_component_radial_profile.pdf",
    )
    longitude_split_results = load_precomputed_detectability_longitude_split_results(
        project_root=project_root,
        output_stem="detectability_em_maps_abs_longitude_30deg_split",
    )
    plot_detectability_em_maps_by_longitude_split_for_paper(
        longitude_split_results,
        figures_dir / "detectability_em_maps.pdf",
    )
    plot_detectability_em_convergence_for_paper(
        detectability_results,
        figures_dir / "detectability_em_convergence.pdf",
    )
    plot_two_component_results_for_paper(
        detectability_results,
        shared_results,
        split_alpha_results,
        figures_dir / "two_component_results.pdf",
    )

    single_component_table = build_single_component_family_profile_scan_table(
        family_profile_scan_results,
    )
    conditional_class_table = build_conditional_population_model_table(
        detectability_comparison=detectability_comparison,
        shared_results=shared_results,
        split_alpha_results=split_alpha_results,
    )
    key_results_table = build_key_results_table(
        joint_results=joint_results,
        detectability_results=detectability_results,
        shared_results=shared_results,
        split_alpha_results=split_alpha_results,
    )

    single_component_table.to_csv(tables_dir / "single_component_model_comparison.csv", index=False)
    conditional_class_table.to_csv(tables_dir / "population_model_class_comparison.csv", index=False)
    key_results_table.to_csv(tables_dir / "key_results_summary.csv", index=False)

    write_single_component_table_tex(single_component_table, tables_dir / "single_component_model_comparison.tex")
    write_population_class_table_tex(conditional_class_table, tables_dir / "population_model_class_comparison.tex")
    write_key_results_table_tex(key_results_table, tables_dir / "key_results_summary.tex")

    summary_payload = build_paper_summary_payload(
        fit_catalog=fit_catalog,
        joint_results=joint_results,
        detectability_results=detectability_results,
        shared_results=shared_results,
        split_alpha_results=split_alpha_results,
        conditional_class_table=conditional_class_table,
    )
    (tables_dir / "paper_results_summary.json").write_text(json.dumps(summary_payload, indent=2))
    write_summary_macros_tex(summary_payload, tables_dir / "paper_numbers.tex")

    return {
        "catalog_results": catalog_results,
        "joint_results": joint_results,
        "detectability_comparison": detectability_comparison,
        "detectability_results": detectability_results,
        "shared_results": shared_results,
        "split_alpha_results": split_alpha_results,
        "single_component_table": single_component_table,
        "conditional_class_table": conditional_class_table,
        "key_results_table": key_results_table,
        "summary_payload": summary_payload,
    }


def plot_catalog_mass_semimajor_axis_overview_for_paper(
    catalog: pd.DataFrame,
    survivability_map: dict[str, object],
    output_path: Path,
    eta_boundary_maps: dict[float, dict[str, object]] | None = None,
) -> None:
    semi_major_axis = catalog["semi_major_axis_kpc"].to_numpy(dtype=float)
    present_mass = catalog["present_mass_msun"].to_numpy(dtype=float)
    initial_mass = catalog["initial_mass_msun"].to_numpy(dtype=float)
    radius_grid = np.asarray(survivability_map["semi_major_axis_grid_kpc"], dtype=float)
    log_mass_grid = np.asarray(survivability_map["log_mass_grid"], dtype=float)
    mass_grid = np.power(10.0, log_mass_grid)
    survival_probability = np.asarray(survivability_map["survival_probability"], dtype=float)
    survivability_cmap = LinearSegmentedColormap.from_list(
        "survivability_grey_to_white",
        ["#8a8a8a", "#ffffff"],
    )

    x_limits = (
        float(semi_major_axis.min() / 1.15),
        float(semi_major_axis.max() * 1.15),
    )
    y_limits = (1.0e3, 3.0e7)
    radius_edges_core = 10.0 ** centers_to_edges(np.log10(radius_grid))
    radius_edges = np.concatenate(([x_limits[0]], radius_edges_core, [x_limits[1]]))
    log_mass_edges = centers_to_edges(log_mass_grid)
    mass_edges_core = np.power(10.0, log_mass_edges)
    mass_edges = np.concatenate(([y_limits[0]], mass_edges_core, [y_limits[1]]))
    survival_probability_plot = np.pad(survival_probability, ((1, 1), (1, 1)), mode="edge")
    survival_probability_plot[0, :] = 0.0

    if eta_boundary_maps is None:
        eta_boundary_maps = {}
    fitted_boundary_50 = survivability_map.get("fitted_boundary_50_log10_msun")

    present_color = "#1f77b4"
    initial_color = "#ff7f0e"

    fig, axes = plt.subplots(ncols=2, figsize=(7.2, 3.6))

    axes[0].scatter(
        semi_major_axis,
        present_mass,
        s=26,
        alpha=0.70,
        color=present_color,
        label="Present mass",
    )
    axes[0].scatter(
        semi_major_axis,
        initial_mass,
        s=26,
        alpha=0.70,
        color=initial_color,
        label="Initial mass",
    )
    axes[0].text(0.985, 0.975, "(a)", transform=axes[0].transAxes, ha="right", va="top")
    axes[0].legend(
        frameon=False,
        fontsize=8.5,
        loc="lower left",
        bbox_to_anchor=(0.03, 0.03),
        borderaxespad=0.0,
        handletextpad=0.4,
    )

    axes[1].pcolormesh(
        radius_edges,
        mass_edges,
        survival_probability_plot,
        cmap=survivability_cmap,
        vmin=0.0,
        vmax=1.0,
        shading="auto",
        rasterized=True,
    )
    boundary_handles = []
    if fitted_boundary_50 is not None:
        line = axes[1].plot(
            radius_grid,
            np.power(10.0, np.maximum(np.asarray(fitted_boundary_50, dtype=float), np.log10(y_limits[0]))),
            color="#045a8d",
            linewidth=1.55,
            linestyle="-",
            label=r"$S=0.5$, $\eta_t=1$",
        )[0]
        boundary_handles.append(line)

    eta_line_styles = {
        0.5: ("#d95f0e", "--", r"$S=0.5$, $\eta_t=0.5$"),
        2.0: ("#238b45", ":", r"$S=0.5$, $\eta_t=2$"),
    }
    for eta_t, comparison_map in sorted(eta_boundary_maps.items()):
        boundary = comparison_map.get("fitted_boundary_50_log10_msun")
        comparison_radius_grid = np.asarray(comparison_map["semi_major_axis_grid_kpc"], dtype=float)
        color, linestyle, label = eta_line_styles.get(
            float(eta_t),
            ("#4d4d4d", "--", rf"$S=0.5$, $\eta_t={eta_t:g}$"),
        )
        if boundary is not None:
            line_log_mass = np.asarray(boundary, dtype=float)
            axes[1].plot(
                comparison_radius_grid,
                np.power(10.0, np.maximum(line_log_mass, np.log10(y_limits[0]))),
                color=color,
                linewidth=1.45,
                linestyle=linestyle,
                label=label,
            )
            boundary_handles.append(axes[1].lines[-1])
    axes[1].scatter(
        catalog["semi_major_axis_kpc"],
        catalog["initial_mass_msun"],
        s=10,
        color="black",
        alpha=0.35,
        linewidths=0.0,
    )
    axes[1].text(0.985, 0.975, "(b)", transform=axes[1].transAxes, ha="right", va="top")

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlim(*x_limits)
    axes[0].set_ylim(*y_limits)
    axes[0].set_xlabel(r"Orbital semimajor axis $a$ [kpc]")
    axes[0].set_ylabel(r"Mass [$\mathrm{M_\odot}$]")

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlim(*x_limits)
    axes[1].set_ylim(*y_limits)
    axes[1].set_xlabel(r"Orbital semimajor axis $a$ [kpc]")
    axes[1].set_ylabel(r"Mass [$\mathrm{M_\odot}$]")
    if boundary_handles:
        axes[1].legend(
            handles=boundary_handles,
            frameon=False,
            fontsize=7.4,
            loc="lower left",
            bbox_to_anchor=(0.03, 0.03),
            borderaxespad=0.0,
            handlelength=2.6,
        )

    fig.tight_layout(w_pad=1.0)
    fig.savefig(output_path)
    plt.close(fig)


def plot_single_component_model_performance_for_paper(
    comparison_table: pd.DataFrame,
    output_path: Path,
) -> None:
    merged = comparison_table.sort_values(
        ["log_likelihood", "rms_residual_sigma_2d"],
        ascending=[False, True],
    ).reset_index(drop=True)
    colors = [imf_family_color(imf_family) for imf_family in merged["imf_family"]]
    markers = {"logpoly3": "o", "step5": "s"}

    fig, ax = plt.subplots(figsize=(3.35, 3.35))
    for row, color in zip(merged.itertuples(index=False), colors, strict=True):
        ax.scatter(
            row.log_likelihood,
            row.rms_residual_sigma_2d,
            s=46 if row.delta_log_likelihood == 0.0 else 38,
            color=color,
            marker=markers.get(row.radial_model, "o"),
            edgecolor="black" if row.delta_log_likelihood == 0.0 else "none",
            linewidth=0.7,
            zorder=3,
        )

    ax.set_xlabel(r"$\log \mathcal{L}$")
    ax.set_ylabel(r"RMS residual in $(\log M_{\rm ini}, \log a)$")
    ax.grid(alpha=0.25, zorder=0)

    imf_handles = [
        plt.Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            color=imf_family_color(name),
            markerfacecolor=imf_family_color(name),
            markersize=5.5,
            label=name,
        )
        for name in ("schechter", "lognormal", "powerlaw")
    ]
    radial_handles = [
        plt.Line2D(
            [],
            [],
            linestyle="none",
            marker=marker,
            color="#555555",
            markerfacecolor="#555555",
            markersize=5.5,
            label=model,
        )
        for model, marker in markers.items()
    ]
    legend_handles = imf_handles + radial_handles
    legend_labels = [handle.get_label() for handle in legend_handles]
    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        fontsize=6.5,
        handletextpad=0.4,
        columnspacing=0.9,
        borderaxespad=0.2,
    )

    ax.set_xlim(float(merged["log_likelihood"].min()) - 3.0, float(merged["log_likelihood"].max()) + 2.0)
    ax.set_ylim(0.95, max(1.55, float(merged["rms_residual_sigma_2d"].max()) + 0.08))
    fig.tight_layout(pad=0.25, rect=(0.0, 0.0, 1.0, 0.88))
    fig.savefig(output_path)
    plt.close(fig)


def plot_single_component_family_profile_scan_for_paper(
    family_scan_results: dict[str, object],
    output_path: Path,
) -> None:
    summary_payload = family_scan_results["summary_payload"]
    powerlaw_table = family_scan_results["powerlaw_table"]
    lognormal_table = family_scan_results["lognormal_table"]
    schechter_table = family_scan_results["schechter_table"]
    global_best_logl = max(
        float(powerlaw_table["log_likelihood"].max()),
        float(lognormal_table["log_likelihood"].max()),
        float(schechter_table["log_likelihood"].max()),
    )
    contour_floor = -50.0
    contour_levels = np.array(
        [
            -50.0,
            -40.0,
            -30.0,
            -22.0,
            -16.0,
            -11.0,
            -7.5,
            -5.0,
            -3.5,
            -2.5,
            -1.8,
            -1.2,
            -0.8,
            -0.5,
            -0.3,
            -0.2,
            -0.1,
            -0.05,
            0.0,
        ],
        dtype=float,
    )
    powerlaw_delta_logl = np.maximum(
        powerlaw_table["log_likelihood"].to_numpy(dtype=float) - global_best_logl,
        contour_floor,
    )

    fig, axes = plt.subplots(ncols=3, figsize=(11.0, 3.5))

    axes[0].plot(
        powerlaw_table["alpha_dndm"],
        powerlaw_delta_logl,
        color=imf_family_color("powerlaw"),
        linewidth=2.0,
    )
    axes[0].axhline(0.0, color="0.75", linewidth=1.0)
    axes[0].set_xlabel(r"$\alpha$")
    axes[0].set_ylabel(r"$\Delta \log \mathcal{L}$ from global best")
    axes[0].set_title("Power law")
    axes[0].set_ylim(contour_floor, 5.0)
    powerlaw_best = powerlaw_table.sort_values("log_likelihood", ascending=False).iloc[0]
    axes[0].plot(
        float(powerlaw_best["alpha_dndm"]),
        max(float(powerlaw_best["log_likelihood"]) - global_best_logl, contour_floor),
        marker="x",
        color="white",
        markersize=7,
        markeredgewidth=1.6,
        linestyle="none",
    )
    axes[0].text(0.03, 0.95, "(a)", transform=axes[0].transAxes, ha="left", va="top")

    lognormal_mu = np.sort(lognormal_table["mu_log10_msun"].unique())
    lognormal_sigma = np.sort(lognormal_table["sigma_log10_msun"].unique())
    lognormal_grid = (
        lognormal_table.pivot(index="sigma_log10_msun", columns="mu_log10_msun", values="log_likelihood")
        .sort_index()
        .sort_index(axis=1)
        .to_numpy(dtype=float)
        - global_best_logl
    )
    lognormal_grid = np.maximum(lognormal_grid, contour_floor)
    axes[1].contourf(lognormal_mu, lognormal_sigma, lognormal_grid, levels=contour_levels, cmap="magma")
    lognormal_best = lognormal_table.sort_values("log_likelihood", ascending=False).iloc[0]
    axes[1].plot(
        float(lognormal_best["mu_log10_msun"]),
        float(lognormal_best["sigma_log10_msun"]),
        marker="x",
        color="white",
        markersize=7,
        markeredgewidth=1.6,
    )
    axes[1].set_xlabel(r"$\mu_{\log M}$")
    axes[1].set_ylabel(r"$\sigma_{\log M}$")
    axes[1].set_title("Lognormal")
    axes[1].text(0.03, 0.95, "(b)", transform=axes[1].transAxes, ha="left", va="top", color="white")

    schechter_alpha = np.sort(schechter_table["alpha_dndm"].unique())
    schechter_logmc = np.sort(schechter_table["log10_m_c_msun"].unique())
    schechter_grid = (
        schechter_table.pivot(index="log10_m_c_msun", columns="alpha_dndm", values="log_likelihood")
        .sort_index()
        .sort_index(axis=1)
        .to_numpy(dtype=float)
        - global_best_logl
    )
    schechter_grid = np.maximum(schechter_grid, contour_floor)
    contour = axes[2].contourf(
        schechter_alpha,
        schechter_logmc,
        schechter_grid,
        levels=contour_levels,
        cmap="magma",
    )
    schechter_best = schechter_table.sort_values("log_likelihood", ascending=False).iloc[0]
    axes[2].plot(
        float(schechter_best["alpha_dndm"]),
        float(schechter_best["log10_m_c_msun"]),
        marker="x",
        color="white",
        markersize=7,
        markeredgewidth=1.6,
    )
    axes[2].set_xlabel(r"$\alpha$")
    axes[2].set_ylabel(r"$\log_{10}(M_c/{\rm M}_\odot)$")
    axes[2].set_title("Schechter")
    axes[2].text(0.03, 0.95, "(c)", transform=axes[2].transAxes, ha="left", va="top", color="white")

    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.18, top=0.88, wspace=0.32)
    cax = fig.add_axes([0.90, 0.18, 0.018, 0.64])
    colorbar = fig.colorbar(contour, cax=cax)
    colorbar.set_label(r"$\Delta \log \mathcal{L}$ from global best")

    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_detectability_counts_for_paper(
    catalog: pd.DataFrame,
    output_path: Path,
    longitude_limit_deg: float | None = None,
) -> None:
    if longitude_limit_deg is not None:
        plot_detectability_counts_by_longitude_split_for_paper(
            catalog=catalog,
            output_path=output_path,
            longitude_limit_deg=longitude_limit_deg,
            combine_mass_bins=True,
        )
        return

    mass_summary_table, histogram_table = build_gc_detectability_histogram_tables(
        catalog,
        n_mass_bins=3,
        n_distance_bins=10,
        n_latitude_bins=6,
    )

    distance_edges = np.unique(
        np.concatenate(
            [
                histogram_table["distance_left_edge_kpc"].to_numpy(),
                histogram_table["distance_right_edge_kpc"].to_numpy(),
            ]
        )
    )
    latitude_edges = np.unique(
        np.concatenate(
            [
                histogram_table["abs_latitude_left_edge_deg"].to_numpy(),
                histogram_table["abs_latitude_right_edge_deg"].to_numpy(),
            ]
        )
    )

    count_cmap = LinearSegmentedColormap.from_list(
        "white_to_black",
        ["#ffffff", "#000000"],
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(mass_summary_table),
        figsize=(11.0, 3.7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    mesh = None
    vmax = max(float(histogram_table["gc_count"].max()), 1.0)
    for axis, summary_row in zip(axes, mass_summary_table.itertuples(index=False), strict=True):
        subset = histogram_table.loc[histogram_table["mass_bin_index"] == summary_row.mass_bin_index]
        count_grid = (
            subset.pivot(index="distance_bin_index", columns="latitude_bin_index", values="gc_count")
            .sort_index(axis=0)
            .sort_index(axis=1)
            .to_numpy(dtype=float)
        )
        mesh = axis.pcolormesh(
            distance_edges,
            latitude_edges,
            count_grid.T,
            shading="auto",
            cmap=count_cmap,
            vmin=0.0,
            vmax=vmax,
            rasterized=True,
        )
        axis.set_xscale("log")
        axis.set_xlabel(r"$D_{\odot}$ [kpc]")
        axis.set_title(
            rf"$\log_{{10}}(M_{{\rm now}}/M_\odot)$"
            rf"$\in[{summary_row.log10_present_mass_left_edge:.2f},"
            rf"{summary_row.log10_present_mass_right_edge:.2f}]$"
            + "\n"
            + rf"$N={summary_row.n_clusters}$",
            fontsize=9,
        )
    axes[0].set_ylabel(r"$|b|$ [deg]")
    if mesh is not None:
        colorbar = fig.colorbar(mesh, ax=axes, shrink=0.92)
        colorbar.set_label("Observed GC counts")
    fig.savefig(output_path)
    plt.close(fig)


def plot_detectability_counts_by_longitude_split_for_paper(
    catalog: pd.DataFrame,
    output_path: Path,
    longitude_limit_deg: float = 90.0,
    combine_mass_bins: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_mass_bins = 1 if combine_mass_bins else 3
    full_mass_summary_table, _ = build_gc_detectability_histogram_tables(
        catalog,
        n_mass_bins=n_mass_bins,
        n_distance_bins=5,
        n_latitude_bins=4,
    )
    mass_edges = np.concatenate(
        [
            full_mass_summary_table["log10_present_mass_left_edge"].to_numpy(dtype=float),
            [float(full_mass_summary_table["log10_present_mass_right_edge"].iloc[-1])],
        ]
    )
    finite_distance = catalog.loc[np.isfinite(catalog["r_sun_kpc"]) & (catalog["r_sun_kpc"] > 0.0), "r_sun_kpc"]
    distance_edges = np.geomspace(float(finite_distance.min()), float(finite_distance.max()), 6)
    latitude_edges = np.linspace(0.0, 90.0, 5)

    signed_longitude = ((catalog["galactic_l_deg"].to_numpy(dtype=float) + 180.0) % 360.0) - 180.0
    central_mask = np.abs(signed_longitude) < longitude_limit_deg
    threshold_label = f"{float(longitude_limit_deg):.1f}"
    subset_definitions = [
        (rf"$|l| < {threshold_label}^\circ$", "lower_abs_longitude", central_mask),
        (rf"$|l| \geq {threshold_label}^\circ$", "higher_abs_longitude", ~central_mask),
    ]

    subset_tables: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for display_label, subset_key, subset_mask in subset_definitions:
        subset_catalog = catalog.loc[subset_mask].copy()
        mass_summary_table, histogram_table = build_gc_detectability_histogram_tables(
            subset_catalog,
            mass_edges=mass_edges,
            distance_edges=distance_edges,
            latitude_edges=latitude_edges,
        )
        mass_summary_table["longitude_subset"] = subset_key
        mass_summary_table["longitude_subset_label"] = display_label
        histogram_table["longitude_subset"] = subset_key
        histogram_table["longitude_subset_label"] = display_label
        subset_tables.append((display_label, mass_summary_table, histogram_table))

    vmax = max(
        max(float(histogram_table["gc_count"].max()), 1.0)
        for _, _, histogram_table in subset_tables
    )

    count_cmap = LinearSegmentedColormap.from_list(
        "white_to_black",
        ["#ffffff", "#000000"],
    )

    if combine_mass_bins:
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(subset_tables),
            figsize=(7.6, 3.7),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
    else:
        fig, axes = plt.subplots(
            nrows=len(subset_tables),
            ncols=len(full_mass_summary_table),
            figsize=(11.0, 6.9),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
    mesh = None
    if combine_mass_bins:
        for axis, (display_label, mass_summary_table, histogram_table) in zip(axes, subset_tables, strict=True):
            summary_row = mass_summary_table.iloc[0]
            subset = histogram_table.loc[histogram_table["mass_bin_index"] == 0]
            count_grid = (
                subset.pivot(index="distance_bin_index", columns="latitude_bin_index", values="gc_count")
                .sort_index(axis=0)
                .sort_index(axis=1)
                .to_numpy(dtype=float)
            )
            mesh = axis.pcolormesh(
                distance_edges,
                latitude_edges,
                count_grid.T,
                shading="auto",
                cmap=count_cmap,
                vmin=0.0,
                vmax=vmax,
                rasterized=True,
            )
            axis.set_xscale("log")
            axis.set_xlabel(r"$D_{\odot}$ [kpc]")
            axis.set_title(display_label, fontsize=10)
            axis.text(
                0.97,
                0.05,
                rf"$N={summary_row.n_clusters}$",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="white",
                bbox=dict(boxstyle="round,pad=0.18", facecolor=(0.0, 0.0, 0.0, 0.25), edgecolor="none"),
            )
        axes[0].set_ylabel(r"$|b|$ [deg]")
    else:
        for row_index, (display_label, mass_summary_table, histogram_table) in enumerate(subset_tables):
            row_axes = axes[row_index]
            for axis, summary_row in zip(row_axes, mass_summary_table.itertuples(index=False), strict=True):
                subset = histogram_table.loc[histogram_table["mass_bin_index"] == summary_row.mass_bin_index]
                count_grid = (
                    subset.pivot(index="distance_bin_index", columns="latitude_bin_index", values="gc_count")
                    .sort_index(axis=0)
                    .sort_index(axis=1)
                    .to_numpy(dtype=float)
                )
                mesh = axis.pcolormesh(
                    distance_edges,
                    latitude_edges,
                    count_grid.T,
                    shading="auto",
                    cmap=count_cmap,
                    vmin=0.0,
                    vmax=vmax,
                    rasterized=True,
                )
                axis.set_xscale("log")
                axis.set_xlabel(r"$D_{\odot}$ [kpc]")
                if row_index == 0:
                    axis.set_title(
                        rf"$\log_{{10}}(M_{{\rm now}}/M_\odot)$"
                        rf"$\in[{summary_row.log10_present_mass_left_edge:.2f},"
                        rf"{summary_row.log10_present_mass_right_edge:.2f}]$",
                        fontsize=9,
                    )
                axis.text(
                    0.97,
                    0.05,
                    rf"$N={summary_row.n_clusters}$",
                    transform=axis.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor=(0.0, 0.0, 0.0, 0.25), edgecolor="none"),
                )
            row_axes[0].set_ylabel(r"$|b|$ [deg]")
            row_axes[0].text(
                0.02,
                0.95,
                display_label,
                transform=row_axes[0].transAxes,
                ha="left",
                va="top",
                fontsize=10,
                color="white",
                bbox=dict(boxstyle="round,pad=0.22", facecolor=(0.0, 0.0, 0.0, 0.28), edgecolor="none"),
            )

    if mesh is not None:
        colorbar = fig.colorbar(mesh, ax=axes, shrink=0.94)
        colorbar.set_label("Observed GC counts")
    fig.savefig(output_path)
    plt.close(fig)

    combined_mass_summary_table = pd.concat(
        [mass_summary_table for _, mass_summary_table, _ in subset_tables],
        ignore_index=True,
    )
    combined_histogram_table = pd.concat(
        [histogram_table for _, _, histogram_table in subset_tables],
        ignore_index=True,
    )
    return combined_mass_summary_table, combined_histogram_table


def load_precomputed_detectability_longitude_split_results(
    project_root: Path,
    output_stem: str,
) -> list[tuple[str, str, dict[str, object]]]:
    summary_path = project_root / "outputs" / "tables" / f"{output_stem}_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing precomputed longitude-split summary table: {summary_path}")

    summary_table = pd.read_csv(summary_path)
    longitude_results: list[tuple[str, str, dict[str, object]]] = []
    for row in summary_table.itertuples(index=False):
        subset_key = str(row.longitude_subset)
        display_label = str(row.longitude_subset_label)
        completeness_grid_path = (
            project_root / "outputs" / "tables" / f"{output_stem}_{subset_key}_completeness_grid.csv"
        )
        if not completeness_grid_path.exists():
            raise FileNotFoundError(f"Missing precomputed longitude-split completeness grid: {completeness_grid_path}")
        longitude_results.append(
            (
                display_label,
                subset_key,
                {"completeness_grid_table": pd.read_csv(completeness_grid_path)},
            )
        )
    return longitude_results


def load_precomputed_single_component_family_profile_scan_results(
    project_root: Path,
) -> dict[str, object]:
    candidate_roots = [
        project_root / "variants" / "single_component_family_profile_scan_smooth_survival_eta1" / "outputs",
        project_root / "variants" / "single_component_family_profile_scan_v4" / "outputs",
        project_root / "variants" / "single_component_family_profile_scan_v3" / "outputs",
    ]
    variant_root = None
    for candidate_root in candidate_roots:
        if (candidate_root / "tables" / "single_component_family_profile_scan_summary.json").exists():
            variant_root = candidate_root
            break
    if variant_root is None:
        variant_root = candidate_roots[0]
    summary_path = variant_root / "tables" / "single_component_family_profile_scan_summary.json"
    powerlaw_path = variant_root / "tables" / "powerlaw_profile_scan.csv"
    lognormal_path = variant_root / "tables" / "lognormal_profile_scan.csv"
    schechter_path = variant_root / "tables" / "schechter_profile_scan.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing precomputed family-profile summary: {summary_path}")
    if not powerlaw_path.exists():
        raise FileNotFoundError(f"Missing precomputed power-law profile scan: {powerlaw_path}")
    if not lognormal_path.exists():
        raise FileNotFoundError(f"Missing precomputed lognormal profile scan: {lognormal_path}")
    if not schechter_path.exists():
        raise FileNotFoundError(f"Missing precomputed Schechter profile scan: {schechter_path}")
    return {
        "variant_output_root": variant_root,
        "summary_payload": json.loads(summary_path.read_text()),
        "powerlaw_table": pd.read_csv(powerlaw_path),
        "lognormal_table": pd.read_csv(lognormal_path),
        "schechter_table": pd.read_csv(schechter_path),
        "summary_figure_pdf_path": variant_root / "figures" / "single_component_family_profile_scan_summary.pdf",
        "summary_figure_png_path": variant_root / "figures" / "single_component_family_profile_scan_summary.png",
    }


def plot_detectability_em_maps_for_paper(
    detectability_results: dict[str, object],
    output_path: Path,
) -> None:
    completeness_grid_table = detectability_results["completeness_grid_table"]

    mass_bin_indices = sorted(completeness_grid_table["present_mass_bin_index"].unique())
    if len(mass_bin_indices) <= 3:
        selected_indices = mass_bin_indices
    else:
        selected_indices = [
            mass_bin_indices[len(mass_bin_indices) // 4],
            mass_bin_indices[len(mass_bin_indices) // 2],
            mass_bin_indices[(3 * len(mass_bin_indices)) // 4],
        ]

    fig, axes_maps = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(11.5, 3.9),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    mesh = None
    for panel_index, (axis, mass_bin_index) in enumerate(zip(axes_maps, selected_indices, strict=True), start=1):
        subset = completeness_grid_table.loc[
            completeness_grid_table["present_mass_bin_index"] == mass_bin_index
        ].copy()
        distance_centers = np.sort(subset["distance_center_kpc"].unique())
        latitude_centers = np.sort(subset["abs_latitude_center_deg"].unique())
        distance_edges = np.power(10.0, centers_to_edges(np.log10(distance_centers)))
        latitude_edges = centers_to_edges(latitude_centers)
        grid_values = (
            subset.pivot(index="distance_center_kpc", columns="abs_latitude_center_deg", values="completeness")
            .reindex(index=distance_centers, columns=latitude_centers)
            .to_numpy(dtype=float)
        )
        mesh = axis.pcolormesh(
            distance_edges,
            latitude_edges,
            grid_values.T,
            shading="auto",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            rasterized=True,
        )
        mass_center = subset["log10_present_mass_center_msun"].iloc[0]
        axis.set_xscale("log")
        axis.set_xlabel(r"$D_{\odot}$ [kpc]")
        if panel_index == 1:
            axis.set_ylabel(r"$|b|$ [deg]")
        axis.set_title(rf"$\log_{{10}}(M_{{\rm now}}/M_\odot)\approx{mass_center:.2f}$", fontsize=10)
        axis.text(0.03, 0.95, f"({chr(96 + panel_index)})", transform=axis.transAxes, ha="left", va="top", color="white")
    if mesh is not None:
        colorbar = fig.colorbar(mesh, ax=axes_maps, shrink=0.96)
        colorbar.set_label("Detectability completeness")
    fig.savefig(output_path)
    plt.close(fig)


def plot_detectability_em_maps_by_longitude_split_for_paper(
    longitude_results: list[tuple[str, str, dict[str, object]]],
    output_path: Path,
) -> None:
    if not longitude_results:
        raise ValueError("longitude_results must contain at least one subset.")

    sample_grid_table = longitude_results[0][2]["completeness_grid_table"]
    mass_bin_indices = sorted(sample_grid_table["present_mass_bin_index"].unique())
    if len(mass_bin_indices) <= 3:
        selected_indices = mass_bin_indices
    else:
        selected_indices = [
            mass_bin_indices[len(mass_bin_indices) // 4],
            mass_bin_indices[len(mass_bin_indices) // 2],
            mass_bin_indices[(3 * len(mass_bin_indices)) // 4],
        ]

    fig, axes = plt.subplots(
        nrows=len(longitude_results),
        ncols=len(selected_indices),
        figsize=(12.6, 3.9 * len(longitude_results)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(longitude_results) == 1:
        axes = np.array([axes])

    for row_index, (display_label, subset_key, detectability_results) in enumerate(longitude_results):
        completeness_grid_table = detectability_results["completeness_grid_table"]
        row_axes = axes[row_index]
        for column_index, (axis, mass_bin_index) in enumerate(zip(row_axes, selected_indices, strict=True), start=1):
            subset = completeness_grid_table.loc[
                completeness_grid_table["present_mass_bin_index"] == mass_bin_index
            ].copy()
            distance_centers = np.sort(subset["distance_center_kpc"].unique())
            latitude_centers = np.sort(subset["abs_latitude_center_deg"].unique())
            distance_edges = np.power(10.0, centers_to_edges(np.log10(distance_centers)))
            latitude_edges = centers_to_edges(latitude_centers)
            grid_values = (
                subset.pivot(index="distance_center_kpc", columns="abs_latitude_center_deg", values="completeness")
                .reindex(index=distance_centers, columns=latitude_centers)
                .to_numpy(dtype=float)
            )
            finite_values = grid_values[np.isfinite(grid_values)]
            panel_vmin = 0.0
            panel_vmax = 1.0
            if finite_values.size > 0:
                panel_vmin = float(np.nanpercentile(finite_values, 5.0))
                panel_vmax = float(np.nanpercentile(finite_values, 95.0))
                panel_vmin = min(max(panel_vmin, 0.0), 1.0)
                panel_vmax = min(max(panel_vmax, 1.0e-6), 1.0)
            if panel_vmax <= panel_vmin:
                delta = max(1.0e-3, 0.05 * max(abs(panel_vmax), 1.0e-3))
                panel_vmin = max(0.0, panel_vmin - delta)
                panel_vmax = min(1.0, panel_vmax + delta)
            mesh = axis.pcolormesh(
                distance_edges,
                latitude_edges,
                grid_values.T,
                shading="auto",
                cmap="magma",
                vmin=panel_vmin,
                vmax=panel_vmax,
                rasterized=True,
            )
            axis.set_xscale("log")
            axis.set_xlabel(r"$D_{\odot}$ [kpc]")
            if column_index == 1:
                axis.set_ylabel(r"$|b|$ [deg]")
            mass_center = subset["log10_present_mass_center_msun"].iloc[0]
            axis.set_title(
                rf"$\log_{{10}}(M_{{\rm now}}/M_\odot)\approx{mass_center:.2f}$"
                + "\n"
                + rf"$C_{{5}}={panel_vmin:.3f},\ C_{{95}}={panel_vmax:.3f}$",
                fontsize=10,
            )
            axis.text(
                0.03,
                0.95,
                f"({chr(96 + row_index * len(selected_indices) + column_index)})",
                transform=axis.transAxes,
                ha="left",
                va="top",
                color="white",
            )
            colorbar = fig.colorbar(mesh, ax=axis, pad=0.01, fraction=0.045)
            colorbar.set_label("Completeness", fontsize=8.5)
            colorbar.ax.tick_params(labelsize=8)
        row_axes[0].text(
            0.03,
            0.08,
            display_label,
            transform=row_axes[0].transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=(0.0, 0.0, 0.0, 0.25), edgecolor="none"),
        )
    fig.savefig(output_path)
    plt.close(fig)


def plot_detectability_em_convergence_for_paper(
    detectability_results: dict[str, object] | list[dict[str, object]],
    output_path: Path,
) -> None:
    result_list = detectability_results if isinstance(detectability_results, list) else [detectability_results]
    observed_count = len(np.asarray(result_list[0]["base_context"].log_mass_data))

    fig, axes = plt.subplots(4, 1, figsize=(3.35, 5.45), sharex=True)
    ax_count, ax_det, ax_obs, ax_logl = axes
    final_log_likelihoods = np.array(
        [
            float(result["iteration_history_table"]["log_likelihood"].to_numpy(dtype=float)[-1])
            for result in result_list
        ],
        dtype=float,
    )
    if np.allclose(final_log_likelihoods.max(), final_log_likelihoods.min()):
        curve_norm = plt.Normalize(
            vmin=float(final_log_likelihoods.min()) - 0.5,
            vmax=float(final_log_likelihoods.max()) + 0.5,
        )
    else:
        curve_norm = plt.Normalize(
            vmin=float(final_log_likelihoods.min()),
            vmax=float(final_log_likelihoods.max()),
        )
    curve_cmap = plt.get_cmap("viridis")
    max_iteration = 0.0
    predicted_observed_values = []

    draw_order = np.argsort(final_log_likelihoods)
    for result_index in draw_order:
        result = result_list[int(result_index)]
        iteration_history_table = result["iteration_history_table"]
        baseline_payload = result["baseline_payload"]

        iterations = iteration_history_table["iteration"].to_numpy(dtype=float)
        n0_series = np.concatenate(([0.0], iterations))
        n0_column = (
            "total_initial_count_above_log10_4"
            if "total_initial_count_above_log10_4" in iteration_history_table.columns
            else "total_initial_count"
        )
        completeness_column = (
            "completeness_mean_above_log10_4"
            if "completeness_mean_above_log10_4" in iteration_history_table.columns
            else "completeness_mean"
        )
        baseline_n0 = total_initial_count_above_log_mass(
            float(baseline_payload["model"]["total_initial_count"]),
            np.asarray(result["base_context"].log_mass_grid, dtype=float),
            np.asarray(baseline_payload["model"]["imf_density_grid"], dtype=float),
            PAPER_LOG_MASS_MIN,
        )
        n0_values = np.concatenate(([baseline_n0], iteration_history_table[n0_column].to_numpy(dtype=float)))
        mean_detectability = np.concatenate(([1.0], iteration_history_table[completeness_column].to_numpy(dtype=float)))
        predicted_observed = iteration_history_table["predicted_observed_count"].to_numpy(dtype=float)
        log_likelihood = iteration_history_table["log_likelihood"].to_numpy(dtype=float)
        delta_log_likelihood = log_likelihood - float(log_likelihood[0])
        color = curve_cmap(curve_norm(final_log_likelihoods[int(result_index)]))
        max_iteration = max(max_iteration, float(np.max(iterations)))
        predicted_observed_values.append(predicted_observed)

        ax_count.plot(
            n0_series,
            n0_values,
            color=color,
            linewidth=1.25,
            alpha=0.55,
        )
        ax_det.plot(
            n0_series,
            mean_detectability,
            color=color,
            linewidth=1.25,
            alpha=0.55,
        )
        ax_obs.plot(
            iterations,
            predicted_observed,
            color=color,
            linewidth=1.25,
            alpha=0.55,
        )
        ax_logl.plot(
            iterations,
            delta_log_likelihood,
            color=color,
            linewidth=1.25,
            alpha=0.55,
        )

    ax_count.set_yscale("log")
    ax_count.set_ylabel(r"$N_0(>10^4\,{\rm M}_\odot)$")
    ax_count.grid(alpha=0.22)
    ax_det.set_ylabel(r"$\langle Q\rangle_{>10^4}$")
    ax_det.grid(alpha=0.22)

    ax_obs.axhline(
        observed_count,
        color="#333333",
        linewidth=1.0,
        linestyle=":",
        label=rf"$N={observed_count}$",
    )
    ax_obs.set_ylabel(r"Predicted observed")
    ax_obs.grid(alpha=0.22)
    predicted_observed_values_array = np.concatenate(predicted_observed_values)
    predicted_observed_limits = np.percentile(predicted_observed_values_array, [0.5, 99.5])
    if predicted_observed_limits[1] > predicted_observed_limits[0]:
        ax_obs.set_ylim(float(predicted_observed_limits[0]), float(predicted_observed_limits[1]))
    ax_obs.legend(
        frameon=False,
        fontsize=7.0,
        loc="lower right",
        handlelength=1.8,
    )

    ax_logl.axhline(0.0, color="#333333", linewidth=0.8, linestyle=":")
    ax_logl.set_ylabel(r"$\Delta\ln\mathcal{L}$")
    ax_logl.set_xlabel("Iteration")
    ax_logl.grid(alpha=0.22)
    adopted_iteration_count = 12
    for ax in axes:
        ax.axvline(
            adopted_iteration_count,
            color="#777777",
            linewidth=0.9,
            linestyle="--",
            alpha=0.75,
            zorder=0,
        )
    for label, ax in zip(("a", "b", "c", "d"), axes, strict=True):
        ax.text(
            0.03,
            0.88,
            f"({label})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
        )
    ax_logl.set_xlim(0.0, max_iteration)

    fig.subplots_adjust(left=0.27, right=0.98, bottom=0.085, top=0.94, hspace=0.16)
    fig.savefig(output_path)
    plt.close(fig)


def plot_best_single_component_summary_for_paper(
    catalog: pd.DataFrame,
    context,
    best_payload: dict[str, object],
    uncertainty_payload: dict[str, object],
    output_path: Path,
    n_projection_samples: int = 250,
) -> None:
    point_intensity_grid = compute_observed_intensity_grid(
        np.asarray(best_payload["model"]["imf_density_grid"]),
        np.asarray(best_payload["model"]["radial_density_grid"]),
        np.asarray(context.selection_probability_grid),
        float(best_payload["model"]["total_initial_count"]),
    )
    log_mass_grid = np.asarray(context.log_mass_grid)
    log_a_grid = np.asarray(context.log_a_grid)
    radius_grid_kpc = np.power(10.0, log_a_grid)

    log_mass_edges = centers_to_edges(log_mass_grid)
    log_a_edges = centers_to_edges(log_a_grid)
    radius_edges_kpc = 10.0 ** log_a_edges

    mass_bin_edges = np.linspace(log_mass_edges[0], log_mass_edges[-1], 13)
    log_a_bin_edges = np.linspace(log_a_edges[0], log_a_edges[-1], 10)

    observed_mass_counts, _ = np.histogram(context.log_mass_data, bins=mass_bin_edges)
    observed_a_counts, _ = np.histogram(context.log_a_data, bins=log_a_bin_edges)

    mass_yerr = poisson_count_errors(observed_mass_counts)
    a_yerr = poisson_count_errors(observed_a_counts)

    cell_weights = point_intensity_grid * np.diff(log_mass_edges)[:, None] * np.diff(log_a_edges)[None, :]
    expected_2d = rebin_expected_counts_2d(
        cell_weights,
        log_mass_grid=log_mass_grid,
        log_a_grid=log_a_grid,
        mass_bin_edges=mass_bin_edges,
        log_a_bin_edges=log_a_bin_edges,
    )
    residual_significance = (
        np.histogram2d(
            context.log_mass_data,
            context.log_a_data,
            bins=[mass_bin_edges, log_a_bin_edges],
        )[0]
        - expected_2d
    ) / np.sqrt(np.clip(expected_2d, 1.0, None))

    sample_projection = sample_joint_projection_bands(
        context=context,
        best_payload=best_payload,
        raw_samples=np.asarray(uncertainty_payload["raw_samples"], dtype=float),
        n_samples=n_projection_samples,
        mass_bin_width=mass_bin_edges[1] - mass_bin_edges[0],
        log_a_bin_width=log_a_bin_edges[1] - log_a_bin_edges[0],
    )

    fig = plt.figure(figsize=(11.5, 9.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.0, 1.0])

    ax_intensity = fig.add_subplot(grid[0, 0])
    ax_mass = fig.add_subplot(grid[0, 1])
    ax_radius = fig.add_subplot(grid[1, 0])
    ax_residual = fig.add_subplot(grid[1, 1])

    mesh = ax_intensity.pcolormesh(
        radius_edges_kpc,
        log_mass_edges,
        point_intensity_grid,
        cmap="magma",
        shading="auto",
        rasterized=True,
    )
    ax_intensity.scatter(
        catalog["semi_major_axis_kpc"],
        catalog["log_initial_mass_msun"],
        s=10,
        color="white",
        alpha=0.5,
        linewidths=0.0,
    )
    ax_intensity.set_xscale("log")
    ax_intensity.set_xlabel(r"Orbital semimajor axis $a$ [kpc]")
    ax_intensity.set_ylabel(r"$\log_{10}(M_{\rm ini}/{\rm M}_\odot)$")
    ax_intensity.set_title("Observed intensity")
    ax_intensity.text(0.03, 0.96, "(a)", transform=ax_intensity.transAxes, ha="left", va="top", color="white")
    colorbar = fig.colorbar(mesh, ax=ax_intensity, pad=0.01)
    colorbar.set_label("Observed point-process intensity")

    mass_centers = 0.5 * (mass_bin_edges[:-1] + mass_bin_edges[1:])
    ax_mass.errorbar(
        mass_centers,
        observed_mass_counts,
        yerr=mass_yerr,
        fmt="o",
        color="black",
        ms=4.0,
        capsize=2.5,
        label="Observed",
    )
    ax_mass.fill_between(
        sample_projection["dense_log_mass"],
        sample_projection["mass_band_low"],
        sample_projection["mass_band_high"],
        color="#d95f02",
        alpha=0.22,
        linewidth=0.0,
        label=r"Model $1\sigma$",
    )
    ax_mass.plot(
        sample_projection["dense_log_mass"],
        sample_projection["mass_median"],
        color="#d95f02",
        linewidth=2.0,
        label="Model median",
    )
    ax_mass.set_xlabel(r"$\log_{10}(M_{\rm ini}/{\rm M}_\odot)$")
    ax_mass.set_ylabel("Counts per bin")
    ax_mass.set_title("Mass projection")
    ax_mass.text(0.03, 0.96, "(b)", transform=ax_mass.transAxes, ha="left", va="top")
    ax_mass.legend(frameon=False, fontsize=9)

    a_centers_kpc = 10.0 ** (0.5 * (log_a_bin_edges[:-1] + log_a_bin_edges[1:]))
    ax_radius.errorbar(
        a_centers_kpc,
        observed_a_counts,
        yerr=a_yerr,
        fmt="o",
        color="black",
        ms=4.0,
        capsize=2.5,
        label="Observed",
    )
    ax_radius.fill_between(
        sample_projection["dense_a_kpc"],
        sample_projection["a_band_low"],
        sample_projection["a_band_high"],
        color="#1b9e77",
        alpha=0.22,
        linewidth=0.0,
        label=r"Model $1\sigma$",
    )
    ax_radius.plot(
        sample_projection["dense_a_kpc"],
        sample_projection["a_median"],
        color="#1b9e77",
        linewidth=2.0,
        label="Model median",
    )
    ax_radius.set_xscale("log")
    ax_radius.set_xlabel(r"Orbital semimajor axis $a$ [kpc]")
    ax_radius.set_ylabel("Counts per bin")
    ax_radius.set_title("Radius projection")
    ax_radius.text(0.03, 0.96, "(c)", transform=ax_radius.transAxes, ha="left", va="top")
    ax_radius.legend(frameon=False, fontsize=9)

    image = ax_residual.pcolormesh(
        10.0 ** log_a_bin_edges,
        mass_bin_edges,
        residual_significance,
        cmap="coolwarm",
        vmin=-3.0,
        vmax=3.0,
        shading="auto",
        rasterized=True,
    )
    ax_residual.set_xscale("log")
    ax_residual.set_xlabel(r"Orbital semimajor axis $a$ [kpc]")
    ax_residual.set_ylabel(r"$\log_{10}(M_{\rm ini}/{\rm M}_\odot)$")
    ax_residual.set_title("Residual significance")
    ax_residual.text(0.03, 0.96, "(d)", transform=ax_residual.transAxes, ha="left", va="top")
    residual_cbar = fig.colorbar(image, ax=ax_residual, pad=0.01)
    residual_cbar.set_label(r"$(N_{\rm obs}-N_{\rm exp})/\sqrt{N_{\rm exp}}$")

    fig.savefig(output_path)
    plt.close(fig)


def _cumulative_from_right(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cumulative = np.zeros_like(y, dtype=float)
    if len(x) < 2:
        return cumulative
    trapezoids = 0.5 * (y[:-1] + y[1:]) * np.diff(x)
    cumulative[:-1] = np.cumsum(trapezoids[::-1])[::-1]
    return cumulative


def _prepare_single_component_profile_plot_inputs(
    baseline_joint_results: dict[str, object],
    detectability_result: dict[str, object],
    uncertainty_payload: dict[str, object],
    n_profile_samples: int = 250,
    n_profile_likelihood_nodes: int = 31,
    n_radial_profile_likelihood_nodes: int = 17,
    flexible_imf_overlay: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    baseline_summary = baseline_joint_results["summary_table"].iloc[0]
    baseline_imf = baseline_joint_results["imf_grid_table"].loc[
        (baseline_joint_results["imf_grid_table"]["imf_family"] == baseline_summary["imf_family"])
        & (baseline_joint_results["imf_grid_table"]["radial_model"] == baseline_summary["radial_model"])
    ].sort_values("log_initial_mass_msun")
    baseline_radial = baseline_joint_results["radial_grid_table"].loc[
        (baseline_joint_results["radial_grid_table"]["imf_family"] == baseline_summary["imf_family"])
        & (baseline_joint_results["radial_grid_table"]["radial_model"] == baseline_summary["radial_model"])
    ].sort_values("log10_semi_major_axis_kpc")

    corrected_context = detectability_result["final_context"]
    corrected_model = detectability_result["final_payload"]["model"]
    corrected_radial_birth = corrected_model["total_initial_count"] * corrected_model["radial_density_grid"]
    profile_bands = sample_intrinsic_profile_bands(
        context=corrected_context,
        best_payload=detectability_result["final_payload"],
        raw_samples=np.asarray(uncertainty_payload["raw_samples"], dtype=float),
        n_samples=n_profile_samples,
    )
    profile_node_indices = np.unique(
        np.linspace(0, len(corrected_context.log_mass_grid) - 1, n_profile_likelihood_nodes, dtype=int)
    )
    radial_profile_node_indices = np.unique(
        np.linspace(0, len(corrected_context.log_a_grid) - 1, n_radial_profile_likelihood_nodes, dtype=int)
    )
    profile_band = compute_profile_likelihood_imf_band(
        best_payload=detectability_result["final_payload"],
        context=corrected_context,
        log_mass_support=np.asarray(corrected_context.log_mass_grid, dtype=float)[profile_node_indices],
    )
    valid_profile_nodes = (
        np.isfinite(profile_band["lower_density"])
        & np.isfinite(profile_band["upper_density"])
        & (profile_band["lower_density"] > 0.0)
        & (profile_band["upper_density"] > 0.0)
    )
    if np.count_nonzero(valid_profile_nodes) >= 2:
        log_mass_support = np.asarray(profile_band["log_mass_support"], dtype=float)[valid_profile_nodes]
        imf_band_low = np.power(
            10.0,
            np.interp(
                corrected_context.log_mass_grid,
                log_mass_support,
                np.log10(np.asarray(profile_band["lower_density"], dtype=float)[valid_profile_nodes]),
            ),
        )
        imf_band_high = np.power(
            10.0,
            np.interp(
                corrected_context.log_mass_grid,
                log_mass_support,
                np.log10(np.asarray(profile_band["upper_density"], dtype=float)[valid_profile_nodes]),
            ),
        )
    else:
        imf_band_low = profile_bands["imf_band_low"]
        imf_band_high = profile_bands["imf_band_high"]
    radial_profile_band = compute_profile_likelihood_radial_birth_band(
        best_payload=detectability_result["final_payload"],
        context=corrected_context,
        log_a_support=np.asarray(corrected_context.log_a_grid, dtype=float)[radial_profile_node_indices],
    )
    valid_radial_profile_nodes = (
        np.isfinite(radial_profile_band["lower_density"])
        & np.isfinite(radial_profile_band["upper_density"])
        & (radial_profile_band["lower_density"] > 0.0)
        & (radial_profile_band["upper_density"] > 0.0)
    )
    if np.count_nonzero(valid_radial_profile_nodes) >= 2:
        log_a_support = np.asarray(radial_profile_band["log_a_support"], dtype=float)[valid_radial_profile_nodes]
        radial_band_low = np.power(
            10.0,
            np.interp(
                corrected_context.log_a_grid,
                log_a_support,
                np.log10(np.asarray(radial_profile_band["lower_density"], dtype=float)[valid_radial_profile_nodes]),
            ),
        )
        radial_band_high = np.power(
            10.0,
            np.interp(
                corrected_context.log_a_grid,
                log_a_support,
                np.log10(np.asarray(radial_profile_band["upper_density"], dtype=float)[valid_radial_profile_nodes]),
            ),
        )
    else:
        radial_band_low = profile_bands["radial_birth_band_low"]
        radial_band_high = profile_bands["radial_birth_band_high"]

    payload: dict[str, object] = {
        "baseline_imf": baseline_imf,
        "baseline_radial": baseline_radial,
        "corrected_context": corrected_context,
        "corrected_model": corrected_model,
        "corrected_radial_birth": corrected_radial_birth,
        "imf_band_low": imf_band_low,
        "imf_band_high": imf_band_high,
        "radial_band_low": radial_band_low,
        "radial_band_high": radial_band_high,
        "flexible_imf_overlay": flexible_imf_overlay,
    }
    return payload


def plot_single_component_profiles_for_paper(
    baseline_joint_results: dict[str, object],
    detectability_result: dict[str, object],
    uncertainty_payload: dict[str, object],
    output_path: Path,
    n_profile_samples: int = 250,
    n_profile_likelihood_nodes: int = 31,
    n_radial_profile_likelihood_nodes: int = 17,
    flexible_imf_overlay: dict[str, np.ndarray] | None = None,
) -> None:
    payload = _prepare_single_component_profile_plot_inputs(
        baseline_joint_results=baseline_joint_results,
        detectability_result=detectability_result,
        uncertainty_payload=uncertainty_payload,
        n_profile_samples=n_profile_samples,
        n_profile_likelihood_nodes=n_profile_likelihood_nodes,
        n_radial_profile_likelihood_nodes=n_radial_profile_likelihood_nodes,
        flexible_imf_overlay=flexible_imf_overlay,
    )
    baseline_imf = payload["baseline_imf"]
    corrected_context = payload["corrected_context"]
    corrected_model = payload["corrected_model"]
    imf_band_low = payload["imf_band_low"]
    imf_band_high = payload["imf_band_high"]
    flexible_imf_overlay = payload["flexible_imf_overlay"]

    baseline_x, baseline_y = _restrict_log_mass_support(
        np.asarray(baseline_imf["log_initial_mass_msun"], dtype=float),
        np.asarray(baseline_imf["imf_density_per_dex"], dtype=float),
        PAPER_LOG_MASS_MIN,
    )
    corrected_x, corrected_y = _restrict_log_mass_support(
        np.asarray(corrected_context.log_mass_grid, dtype=float),
        np.asarray(corrected_model["imf_density_grid"], dtype=float),
        PAPER_LOG_MASS_MIN,
    )
    _, corrected_band_low = _restrict_log_mass_support(
        np.asarray(corrected_context.log_mass_grid, dtype=float),
        np.asarray(imf_band_low, dtype=float),
        PAPER_LOG_MASS_MIN,
    )
    _, corrected_band_high = _restrict_log_mass_support(
        np.asarray(corrected_context.log_mass_grid, dtype=float),
        np.asarray(imf_band_high, dtype=float),
        PAPER_LOG_MASS_MIN,
    )

    baseline_imf_cumulative = _cumulative_from_right(baseline_x, baseline_y)
    baseline_imf_cumulative /= max(float(baseline_imf_cumulative[0]), 1.0e-12)
    corrected_imf_cumulative = _cumulative_from_right(corrected_x, corrected_y)
    corrected_imf_cumulative /= max(float(corrected_imf_cumulative[0]), 1.0e-12)
    imf_band_low_cumulative = _cumulative_from_right(corrected_x, corrected_band_low)
    imf_band_low_cumulative /= max(float(imf_band_low_cumulative[0]), 1.0e-12)
    imf_band_high_cumulative = _cumulative_from_right(corrected_x, corrected_band_high)
    imf_band_high_cumulative /= max(float(imf_band_high_cumulative[0]), 1.0e-12)
    if flexible_imf_overlay is not None:
        flexible_x, flexible_y = _restrict_log_mass_support(
            np.asarray(flexible_imf_overlay["log_mass_grid"], dtype=float),
            np.asarray(flexible_imf_overlay["imf_density_grid"], dtype=float),
            PAPER_LOG_MASS_MIN,
        )
        _, flexible_low = _restrict_log_mass_support(
            np.asarray(flexible_imf_overlay["log_mass_grid"], dtype=float),
            np.asarray(flexible_imf_overlay["imf_band_low"], dtype=float),
            PAPER_LOG_MASS_MIN,
        )
        _, flexible_high = _restrict_log_mass_support(
            np.asarray(flexible_imf_overlay["log_mass_grid"], dtype=float),
            np.asarray(flexible_imf_overlay["imf_band_high"], dtype=float),
            PAPER_LOG_MASS_MIN,
        )
        flexible_cumulative = _cumulative_from_right(flexible_x, flexible_y)
        flexible_cumulative /= max(float(flexible_cumulative[0]), 1.0e-12)
        flexible_low_cumulative = _cumulative_from_right(flexible_x, flexible_low)
        flexible_low_cumulative /= max(float(flexible_low_cumulative[0]), 1.0e-12)
        flexible_high_cumulative = _cumulative_from_right(flexible_x, flexible_high)
        flexible_high_cumulative /= max(float(flexible_high_cumulative[0]), 1.0e-12)
        flexible_imf_overlay = {
            **flexible_imf_overlay,
            "log_mass_grid_display": flexible_x,
            "imf_density_grid_display": flexible_y,
            "imf_band_low_display": flexible_low,
            "imf_band_high_display": flexible_high,
            "imf_cumulative": flexible_cumulative,
            "imf_band_low_cumulative": flexible_low_cumulative,
            "imf_band_high_cumulative": flexible_high_cumulative,
        }

    fig, axes = plt.subplots(ncols=2, figsize=(11.5, 4.4))

    axes[0].plot(
        baseline_x,
        baseline_y,
        color="#b3b3b3",
        linewidth=4.5,
        alpha=0.9,
        solid_capstyle="round",
        label="Perfect-detectability baseline",
    )
    axes[0].fill_between(
        corrected_x,
        corrected_band_low,
        corrected_band_high,
        color="#d95f02",
        alpha=0.20,
        linewidth=0.0,
        label=r"Profile-likelihood $1\sigma$",
    )
    axes[0].plot(
        corrected_x,
        corrected_y,
        color="#111111",
        linewidth=2.2,
        label="Detectability-corrected fit",
    )
    if flexible_imf_overlay is not None:
        axes[0].fill_between(
            flexible_imf_overlay["log_mass_grid_display"],
            flexible_imf_overlay["imf_band_low_display"],
            flexible_imf_overlay["imf_band_high_display"],
            color="#1b9e77",
            alpha=0.14,
            linewidth=0.0,
            label=r"Flexible IMF bootstrap $1\sigma$",
        )
        axes[0].plot(
            flexible_imf_overlay["log_mass_grid_display"],
            flexible_imf_overlay["imf_density_grid_display"],
            color="#1b9e77",
            linewidth=2.0,
            linestyle="--",
            label="Best flexible IMF cross-check",
        )
    axes[0].set_xlabel(r"$\log_{10}(M_{\rm ini}/{\rm M}_\odot)$")
    axes[0].set_ylabel("Intrinsic IMF density per dex")
    axes[0].set_yscale("log")
    axes[0].set_xlim(PAPER_LOG_MASS_MIN, float(corrected_x[-1]))
    axes[0].set_title("Single-component IMF")
    axes[0].text(0.03, 0.96, "(a)", transform=axes[0].transAxes, ha="left", va="top")
    axes[0].legend(frameon=False, fontsize=8.5)

    axes[1].plot(
        baseline_x,
        baseline_imf_cumulative,
        color="#b3b3b3",
        linewidth=4.5,
        alpha=0.9,
        solid_capstyle="round",
        label="Perfect-detectability baseline",
    )
    axes[1].fill_between(
        corrected_x,
        imf_band_low_cumulative,
        imf_band_high_cumulative,
        color="#d95f02",
        alpha=0.20,
        linewidth=0.0,
        label=r"Profile-likelihood $1\sigma$",
    )
    axes[1].plot(
        corrected_x,
        corrected_imf_cumulative,
        color="#111111",
        linewidth=2.2,
        label="Detectability-corrected fit",
    )
    if flexible_imf_overlay is not None:
        axes[1].fill_between(
            flexible_imf_overlay["log_mass_grid_display"],
            flexible_imf_overlay["imf_band_low_cumulative"],
            flexible_imf_overlay["imf_band_high_cumulative"],
            color="#1b9e77",
            alpha=0.14,
            linewidth=0.0,
        )
        axes[1].plot(
            flexible_imf_overlay["log_mass_grid_display"],
            flexible_imf_overlay["imf_cumulative"],
            color="#1b9e77",
            linewidth=2.0,
            linestyle="--",
        )
    axes[1].set_xlabel(r"$\log_{10}(M_{\rm ini}/{\rm M}_\odot)$")
    axes[1].set_ylabel(r"Cumulative IMF fraction $> M_{\rm ini}$")
    axes[1].set_yscale("log")
    axes[1].set_xlim(PAPER_LOG_MASS_MIN, float(corrected_x[-1]))
    axes[1].set_title("Cumulative single-component IMF")
    axes[1].text(0.03, 0.96, "(b)", transform=axes[1].transAxes, ha="left", va="top")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_single_component_radial_profile_for_paper(
    baseline_joint_results: dict[str, object],
    detectability_result: dict[str, object],
    uncertainty_payload: dict[str, object],
    output_path: Path,
    n_profile_samples: int = 250,
    n_profile_likelihood_nodes: int = 31,
    n_radial_profile_likelihood_nodes: int = 17,
    flexible_imf_overlay: dict[str, np.ndarray] | None = None,
) -> None:
    del uncertainty_payload, n_profile_samples, n_profile_likelihood_nodes, flexible_imf_overlay
    baseline_summary = baseline_joint_results["summary_table"].iloc[0]
    baseline_radial = baseline_joint_results["radial_grid_table"].loc[
        (baseline_joint_results["radial_grid_table"]["imf_family"] == baseline_summary["imf_family"])
        & (baseline_joint_results["radial_grid_table"]["radial_model"] == baseline_summary["radial_model"])
    ].sort_values("log10_semi_major_axis_kpc")
    corrected_context = detectability_result["final_context"]
    corrected_model = detectability_result["final_payload"]["model"]
    baseline_imf_fraction_above = integrate_density_above_log_mass(
        np.asarray(baseline_joint_results["context"].log_mass_grid, dtype=float),
        np.asarray(baseline_joint_results["best_payload"]["model"]["imf_density_grid"], dtype=float),
        PAPER_LOG_MASS_MIN,
    )
    corrected_imf_fraction_above = integrate_density_above_log_mass(
        np.asarray(corrected_context.log_mass_grid, dtype=float),
        np.asarray(corrected_model["imf_density_grid"], dtype=float),
        PAPER_LOG_MASS_MIN,
    )
    corrected_radial_birth = (
        corrected_model["total_initial_count"] * corrected_imf_fraction_above * corrected_model["radial_density_grid"]
    )

    radial_profile_node_indices = np.unique(
        np.linspace(0, len(corrected_context.log_a_grid) - 1, n_radial_profile_likelihood_nodes, dtype=int)
    )
    radial_profile_band = compute_profile_likelihood_radial_birth_band(
        best_payload=detectability_result["final_payload"],
        context=corrected_context,
        log_a_support=np.asarray(corrected_context.log_a_grid, dtype=float)[radial_profile_node_indices],
    )
    valid_radial_profile_nodes = (
        np.isfinite(radial_profile_band["lower_density"])
        & np.isfinite(radial_profile_band["upper_density"])
        & (radial_profile_band["lower_density"] > 0.0)
        & (radial_profile_band["upper_density"] > 0.0)
    )
    if np.count_nonzero(valid_radial_profile_nodes) >= 2:
        log_a_support = np.asarray(radial_profile_band["log_a_support"], dtype=float)[valid_radial_profile_nodes]
        radial_band_low = np.power(
            10.0,
            np.interp(
                corrected_context.log_a_grid,
                log_a_support,
                np.log10(np.asarray(radial_profile_band["lower_density"], dtype=float)[valid_radial_profile_nodes]),
            ),
        ) * corrected_imf_fraction_above
        radial_band_high = np.power(
            10.0,
            np.interp(
                corrected_context.log_a_grid,
                log_a_support,
                np.log10(np.asarray(radial_profile_band["upper_density"], dtype=float)[valid_radial_profile_nodes]),
            ),
        ) * corrected_imf_fraction_above
    else:
        radial_band_low = np.full_like(corrected_radial_birth, np.nan, dtype=float)
        radial_band_high = np.full_like(corrected_radial_birth, np.nan, dtype=float)

    fig, ax = plt.subplots(figsize=(3.35, 2.8))
    ax.plot(
        baseline_radial["semi_major_axis_kpc"],
        baseline_radial["birth_intensity_per_dex_a"] * baseline_imf_fraction_above,
        color="#b3b3b3",
        linewidth=4.5,
        alpha=0.9,
        solid_capstyle="round",
        label="Perfect-detectability baseline",
    )
    ax.fill_between(
        np.power(10.0, corrected_context.log_a_grid),
        radial_band_low,
        radial_band_high,
        color="#d95f02",
        alpha=0.20,
        linewidth=0.0,
        label=r"Profile-likelihood $1\sigma$",
    )
    ax.plot(
        np.power(10.0, corrected_context.log_a_grid),
        corrected_radial_birth,
        color="#111111",
        linewidth=2.2,
        label="Detectability-corrected fit",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Orbital semimajor axis $a$ [kpc]")
    ax.set_ylabel(r"Birth intensity per dex in $a$")
    ax.text(0.03, 0.96, "(a)", transform=ax.transAxes, ha="left", va="top")
    ax.legend(frameon=False, fontsize=7.2, loc="lower left")
    fig.tight_layout(pad=0.35)
    fig.savefig(output_path)
    plt.close(fig)


def load_precomputed_flexible_imf_overlay(
    project_root: Path,
    log_mass_grid: np.ndarray,
) -> dict[str, np.ndarray] | None:
    variant_roots = [
        project_root / "variants" / "flexible_imf_bootstrap_comparison_abs_longitude_smooth_survival_eta1",
        project_root / "variants" / "flexible_imf_bootstrap_comparison_abs_longitude",
        project_root / "variants" / "flexible_imf_bootstrap_comparison",
    ]
    summary_path = None
    band_path = None
    for variant_root in variant_roots:
        trial_summary = variant_root / "outputs" / "tables" / "imf_profile_vs_logspline_summary.json"
        trial_band = variant_root / "outputs" / "tables" / "logspline6_bootstrap_imf_band.csv"
        if trial_summary.exists() and trial_band.exists():
            summary_path = trial_summary
            band_path = trial_band
            break
    if summary_path is None or band_path is None:
        return None

    summary_payload = json.loads(summary_path.read_text())
    model_payload = summary_payload.get("logspline6_bootstrap_model", {})
    imf_parameters_json = model_payload.get("imf_parameters_json")
    if not imf_parameters_json:
        return None

    imf_parameters = json.loads(imf_parameters_json)
    knot_positions = np.asarray(imf_parameters["knot_log10_msun"], dtype=float)
    node_log_amplitudes = np.asarray(imf_parameters["node_log_amplitudes_relative"], dtype=float)
    spline = interpolate.PchipInterpolator(knot_positions, node_log_amplitudes, extrapolate=True)
    density_grid = np.exp(np.clip(spline(log_mass_grid), -700.0, 700.0))
    density_grid /= max(float(np.trapezoid(density_grid, log_mass_grid)), 1.0e-12)
    band_table = pd.read_csv(band_path)
    return {
        "log_mass_grid": np.asarray(log_mass_grid, dtype=float),
        "imf_density_grid": np.asarray(density_grid, dtype=float),
        "imf_band_low": np.interp(
            log_mass_grid,
            np.asarray(band_table["log_initial_mass_msun"], dtype=float),
            np.asarray(band_table["imf_band_low"], dtype=float),
        ),
        "imf_band_high": np.interp(
            log_mass_grid,
            np.asarray(band_table["log_initial_mass_msun"], dtype=float),
            np.asarray(band_table["imf_band_high"], dtype=float),
        ),
    }


def sample_intrinsic_profile_bands(
    context,
    best_payload: dict[str, object],
    raw_samples: np.ndarray,
    n_samples: int,
) -> dict[str, np.ndarray]:
    raw_samples = np.asarray(raw_samples, dtype=float)
    if len(raw_samples) > n_samples:
        selection = np.linspace(0, len(raw_samples) - 1, n_samples, dtype=int)
        raw_samples = raw_samples[selection]

    imf_curves = []
    radial_birth_curves = []
    for params in raw_samples:
        model = unpack_model(params, context=context, spec=best_payload["spec"])
        imf_curves.append(np.asarray(model["imf_density_grid"], dtype=float))
        radial_birth_curves.append(
            float(model["total_initial_count"]) * np.asarray(model["radial_density_grid"], dtype=float)
        )

    imf_curves_array = np.asarray(imf_curves, dtype=float)
    radial_birth_curves_array = np.asarray(radial_birth_curves, dtype=float)

    imf_low, imf_median, imf_high = np.quantile(imf_curves_array, [0.16, 0.5, 0.84], axis=0)
    radial_low, radial_median, radial_high = np.quantile(radial_birth_curves_array, [0.16, 0.5, 0.84], axis=0)

    return {
        "log_mass_grid": np.asarray(context.log_mass_grid, dtype=float),
        "imf_band_low": imf_low,
        "imf_median": imf_median,
        "imf_band_high": imf_high,
        "log_a_grid": np.asarray(context.log_a_grid, dtype=float),
        "radial_birth_band_low": radial_low,
        "radial_birth_median": radial_median,
        "radial_birth_band_high": radial_high,
    }


def plot_two_component_results_for_paper(
    detectability_results: dict[str, object],
    shared_results: dict[str, object],
    split_alpha_results: dict[str, object],
    output_path: Path,
) -> None:
    single_context = detectability_results["final_context"]
    single_model = detectability_results["final_payload"]["model"]
    single_imf_x = single_context.log_mass_grid
    single_imf_y = single_model["imf_density_grid"]
    single_radial_x = np.power(10.0, single_context.log_a_grid)
    single_radial_y = single_model["total_initial_count"] * single_model["radial_density_grid"]

    shared_imf = shared_results["best_imf_grid_table"].loc[
        shared_results["best_imf_grid_table"]["component_label"] == "in_situ"
    ].sort_values("log_initial_mass_msun")
    shared_radial = shared_results["best_radial_grid_table"].copy()
    split_alpha_imf = split_alpha_results["best_imf_grid_table"].copy()

    fig, axes = plt.subplots(ncols=2, figsize=(11.5, 4.8))

    axes[0].plot(
        single_imf_x,
        single_imf_y,
        color="#b3b3b3",
        linewidth=8.0,
        alpha=0.9,
        solid_capstyle="round",
        zorder=1,
        label="Best detectability-corrected single-component IMF",
    )
    axes[0].plot(
        shared_imf["log_initial_mass_msun"],
        shared_imf["imf_density_per_dex"],
        color="black",
        linewidth=2.2,
        zorder=3,
        label="Shared IMF",
    )
    for component_label, color, label in (
        ("in_situ", "#d95f02", r"Best split-$\alpha$ in-situ IMF"),
        ("accreted", "#1b9e77", r"Best split-$\alpha$ accreted IMF"),
    ):
        subset = split_alpha_imf.loc[split_alpha_imf["component_label"] == component_label].sort_values(
            "log_initial_mass_msun"
        )
        axes[0].plot(
            subset["log_initial_mass_msun"],
            subset["imf_density_per_dex"],
            color=color,
            linewidth=1.8,
            linestyle="--",
            zorder=4,
            label=label,
        )
    axes[0].set_xlabel(r"$\log_{10}(M_{\rm ini}/{\rm M}_\odot)$")
    axes[0].set_ylabel("Intrinsic IMF density per dex")
    axes[0].set_title(r"Shared versus split-$\alpha$ IMF fits")
    axes[0].text(0.03, 0.96, "(a)", transform=axes[0].transAxes, ha="left", va="top")
    axes[0].legend(frameon=False, fontsize=8.5)

    shared_summary = shared_results["best_component_summary_table"].set_index("component_label").sort_index()
    axes[1].plot(
        single_radial_x,
        single_radial_y,
        color="#b3b3b3",
        linewidth=8.0,
        alpha=0.9,
        solid_capstyle="round",
        zorder=1,
        label="Best detectability-corrected single-component profile",
    )
    for component_label, color, label in (
        ("in_situ", "#d95f02", "In-situ"),
        ("accreted", "#1b9e77", "Accreted"),
    ):
        subset = shared_radial.loc[shared_radial["component_label"] == component_label].sort_values(
            "log10_semi_major_axis_kpc"
        )
        summary_row = shared_summary.loc[component_label]
        axes[1].plot(
            subset["semi_major_axis_kpc"],
            subset["birth_intensity_per_dex_a"],
            color=color,
            linewidth=2.2,
            zorder=3,
            label=(
                f"{label}: $N_0={summary_row['total_initial_count']:.0f}$, "
                f"$f_{{\\rm sel}}={summary_row['survival_fraction']:.3f}$"
            ),
        )
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"Orbital semimajor axis $a$ [kpc]")
    axes[1].set_ylabel(r"Birth intensity per dex in $a$")
    axes[1].set_title("Best shared-IMF two-component model")
    axes[1].text(0.03, 0.96, "(b)", transform=axes[1].transAxes, ha="left", va="top")
    axes[1].legend(frameon=False, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def sample_joint_projection_bands(
    context,
    best_payload: dict[str, object],
    raw_samples: np.ndarray,
    n_samples: int,
    mass_bin_width: float,
    log_a_bin_width: float,
) -> dict[str, np.ndarray]:
    raw_samples = np.asarray(raw_samples, dtype=float)
    if len(raw_samples) > n_samples:
        selection = np.linspace(0, len(raw_samples) - 1, n_samples, dtype=int)
        raw_samples = raw_samples[selection]

    mass_curves = []
    a_curves = []
    for params in raw_samples:
        model = unpack_model(params, context=context, spec=best_payload["spec"])
        point_intensity_grid = compute_observed_intensity_grid(
            model["imf_density_grid"],
            model["radial_density_grid"],
            context.selection_probability_grid,
            model["total_initial_count"],
        )
        mass_density = np.trapezoid(point_intensity_grid, context.log_a_grid, axis=1)
        a_density = np.trapezoid(point_intensity_grid, context.log_mass_grid, axis=0)
        mass_curves.append(mass_density * mass_bin_width)
        a_curves.append(a_density * log_a_bin_width)

    mass_curves_array = np.asarray(mass_curves, dtype=float)
    a_curves_array = np.asarray(a_curves, dtype=float)

    mass_low, mass_median, mass_high = np.quantile(mass_curves_array, [0.16, 0.5, 0.84], axis=0)
    a_low, a_median, a_high = np.quantile(a_curves_array, [0.16, 0.5, 0.84], axis=0)

    dense_log_mass = np.linspace(context.log_mass_grid[0], context.log_mass_grid[-1], 500)
    dense_log_a = np.linspace(context.log_a_grid[0], context.log_a_grid[-1], 500)
    dense_a_kpc = 10.0 ** dense_log_a

    return {
        "dense_log_mass": dense_log_mass,
        "mass_band_low": interpolate.interp1d(context.log_mass_grid, mass_low, kind="linear")(dense_log_mass),
        "mass_median": interpolate.interp1d(context.log_mass_grid, mass_median, kind="linear")(dense_log_mass),
        "mass_band_high": interpolate.interp1d(context.log_mass_grid, mass_high, kind="linear")(dense_log_mass),
        "dense_a_kpc": dense_a_kpc,
        "a_band_low": interpolate.interp1d(context.log_a_grid, a_low, kind="linear")(dense_log_a),
        "a_median": interpolate.interp1d(context.log_a_grid, a_median, kind="linear")(dense_log_a),
        "a_band_high": interpolate.interp1d(context.log_a_grid, a_high, kind="linear")(dense_log_a),
    }


def poisson_count_errors(counts: np.ndarray, confidence_level: float = 0.6826894921370859) -> np.ndarray:
    alpha = 1.0 - confidence_level
    counts = np.asarray(counts, dtype=float)
    lower = np.zeros_like(counts)
    positive = counts > 0
    lower[positive] = 0.5 * stats.chi2.ppf(alpha / 2.0, 2.0 * counts[positive])
    upper = 0.5 * stats.chi2.ppf(1.0 - alpha / 2.0, 2.0 * (counts + 1.0))
    return np.vstack([counts - lower, upper - counts])


def build_single_component_model_table(comparison_table: pd.DataFrame) -> pd.DataFrame:
    merged = comparison_table.copy().sort_values(
        ["log_likelihood", "rms_residual_sigma_2d"],
        ascending=[False, True],
    )
    merged["model_label"] = merged["imf_family"] + " + " + merged["radial_model"]
    return merged[
        [
            "model_label",
            "log_likelihood",
            "rms_residual_sigma_2d",
            "mean_detectability",
            "total_initial_count",
        ]
    ].reset_index(drop=True)


def select_best_family_profile_summary_row(
    summary_payload: dict[str, object],
    family_name: str,
) -> dict[str, object]:
    family_scan_key = f"{family_name}_scan"
    unconstrained_row = dict(summary_payload["unconstrained_family_best_fits"][family_name])
    scan_row = dict(summary_payload[family_scan_key]["best_scan_point"])
    if float(scan_row["log_likelihood"]) > float(unconstrained_row["log_likelihood"]):
        return scan_row
    return unconstrained_row


def format_profile_scan_parameter_label(family_name: str, row: dict[str, object]) -> str:
    if family_name == "powerlaw":
        return rf"$\alpha={float(row['alpha_dndm']):.3f}$"
    if family_name == "lognormal":
        return (
            rf"$\mu={float(row['mu_log10_msun']):.2f},\ "
            rf"\sigma={float(row['sigma_log10_msun']):.2f}$"
        )
    if family_name == "schechter":
        return (
            rf"$\alpha={float(row['alpha_dndm']):.3f},\ "
            rf"\log_{{10}}(M_c/{{\rm M}}_\odot)={float(row['log10_m_c_msun']):.3f}$"
        )
    return family_name


def _reported_count_from_profile_row(
    family_name: str,
    row: dict[str, object],
    *,
    support_log_mass_min: float = 3.5,
    support_log_mass_max: float = 7.3,
    report_log_mass_min: float = PAPER_LOG_MASS_MIN,
) -> float:
    total_initial_count = float(row["total_initial_count"])
    dense_log_mass = np.linspace(support_log_mass_min, support_log_mass_max, 2000)
    if family_name == "powerlaw":
        alpha = float(row["alpha_dndm"])
        density = np.power(10.0, (alpha + 1.0) * dense_log_mass)
    elif family_name == "lognormal":
        mu = float(row["mu_log10_msun"])
        sigma = float(row["sigma_log10_msun"])
        density = stats.norm.pdf(dense_log_mass, loc=mu, scale=sigma)
    elif family_name == "schechter":
        alpha = float(row["alpha_dndm"])
        m_c = np.power(10.0, float(row["log10_m_c_msun"]))
        mass_grid = np.power(10.0, dense_log_mass)
        density = np.power(mass_grid, alpha + 1.0) * np.exp(-mass_grid / m_c)
    else:
        return total_initial_count
    total = max(float(np.trapezoid(density, dense_log_mass)), 1.0e-12)
    mask = dense_log_mass >= report_log_mass_min
    reported = float(np.trapezoid(density[mask], dense_log_mass[mask]))
    return total_initial_count * reported / total


def build_single_component_family_profile_scan_table(
    family_scan_results: dict[str, object],
) -> pd.DataFrame:
    summary_payload = family_scan_results["summary_payload"]
    rows: list[dict[str, object]] = []
    for family_name in ("schechter", "lognormal", "powerlaw"):
        best_row = select_best_family_profile_summary_row(summary_payload, family_name)
        total_initial_count = _reported_count_from_profile_row(family_name, best_row)
        rows.append(
            {
                "model_family": family_name,
                "radial_model": "logpoly3",
                "best_parameters_label": format_profile_scan_parameter_label(family_name, best_row),
                "log_likelihood": float(best_row["log_likelihood"]),
                "bic": float(best_row["bic"]),
                "total_initial_count": total_initial_count,
            }
        )

    table = pd.DataFrame(rows)
    best_bic = float(table["bic"].min())
    table["delta_bic_from_best"] = table["bic"] - best_bic
    return table.sort_values("bic").reset_index(drop=True)


def build_conditional_population_model_table(
    detectability_comparison: dict[str, object],
    shared_results: dict[str, object],
    split_alpha_results: dict[str, object],
) -> pd.DataFrame:
    single_best = detectability_comparison["summary_table"].iloc[0]
    shared_best = shared_results["summary_table"].iloc[0]
    split_alpha_best = split_alpha_results["summary_table"].iloc[0]

    n_total = int(shared_best["n_clusters_total"])
    component_counts = np.array(
        [int(shared_best["n_clusters_in_situ"]), int(shared_best["n_clusters_accreted"])],
        dtype=float,
    )
    partition_constant = float(n_total * np.log(n_total) - np.sum(component_counts * np.log(component_counts)))

    rows = [
        {
            "model_class": "single_population",
            "description": f"{single_best['imf_family']} + {single_best['radial_model']}",
            "n_parameters": int(single_best["n_parameters"]),
            "raw_log_likelihood": float(single_best["log_likelihood"]),
            "conditional_log_likelihood": float(single_best["log_likelihood"]),
            "raw_bic": float(single_best["bic"]),
            "conditional_bic": float(single_best["bic"]),
        },
        {
            "model_class": "two_component_shared_imf",
            "description": (
                f"shared {shared_best['imf_family']}; "
                f"in-situ {shared_best['in_situ_radial_model']}; "
                f"accreted {shared_best['accreted_radial_model']}"
            ),
            "n_parameters": int(shared_best["n_parameters"]),
            "raw_log_likelihood": float(shared_best["log_likelihood"]),
            "conditional_log_likelihood": float(shared_best["log_likelihood"] + partition_constant),
            "raw_bic": float(shared_best["bic"]),
            "conditional_bic": float(shared_best["bic"] - 2.0 * partition_constant),
        },
        {
            "model_class": "two_component_split_alpha",
            "description": (
                r"shared $M_{\rm c}$, split $\alpha$; "
                f"in-situ {split_alpha_best['in_situ_radial_model']}; "
                f"accreted {split_alpha_best['accreted_radial_model']}"
            ),
            "n_parameters": int(split_alpha_best["n_parameters"]),
            "raw_log_likelihood": float(split_alpha_best["log_likelihood"]),
            "conditional_log_likelihood": float(split_alpha_best["log_likelihood"] + partition_constant),
            "raw_bic": float(split_alpha_best["bic"]),
            "conditional_bic": float(split_alpha_best["bic"] - 2.0 * partition_constant),
        },
    ]
    table = pd.DataFrame(rows).sort_values("conditional_bic", ascending=True).reset_index(drop=True)
    table["delta_conditional_bic"] = table["conditional_bic"] - float(table["conditional_bic"].min())
    table["partition_constant"] = partition_constant
    return table


def build_key_results_table(
    joint_results: dict[str, object],
    detectability_results: dict[str, object],
    shared_results: dict[str, object],
    split_alpha_results: dict[str, object],
) -> pd.DataFrame:
    best_single = joint_results["summary_table"].iloc[0]
    single_total_mass = best_single_model_total_initial_stellar_mass(joint_results)
    detectability_total_mass = detectability_corrected_single_total_initial_stellar_mass(detectability_results)
    detectability_model = detectability_results["final_payload"]["model"]
    detectability_summary = detectability_results["final_payload"]["summary"]
    shared_components = shared_results["best_component_summary_table"].set_index("component_label")
    shared_component_masses = shared_imf_component_total_initial_stellar_masses(shared_results)
    split_alpha_components = split_alpha_results["best_component_summary_table"].set_index("component_label")
    split_alpha_component_masses = split_alpha_component_total_initial_stellar_masses(split_alpha_results)

    shared_imf = json.loads(shared_components.loc["in_situ", "shared_imf_parameters_json"])
    split_alpha_in_situ_imf = json.loads(split_alpha_components.loc["in_situ", "imf_parameters_json"])
    split_alpha_accreted_imf = json.loads(split_alpha_components.loc["accreted", "imf_parameters_json"])

    rows = [
        {
            "model": "Best single component",
            "component": "all",
            "imf_family": best_single["imf_family"],
            "radial_model": best_single["radial_model"],
            "alpha_dndm": json.loads(best_single["imf_parameters_json"]).get("alpha_dndm"),
            "log10_m_c_msun": json.loads(best_single["imf_parameters_json"]).get("log10_m_c_msun"),
            "total_initial_count": float(best_single["total_initial_count"]),
            "survival_fraction": float(best_single["survival_fraction"]),
            "total_initial_stellar_mass_msun": float(single_total_mass),
        },
        {
            "model": "Detectability-corrected single component",
            "component": "all",
            "imf_family": detectability_summary.imf_family,
            "radial_model": detectability_summary.radial_model,
            "alpha_dndm": json.loads(detectability_summary.imf_parameters_json).get("alpha_dndm"),
            "log10_m_c_msun": json.loads(detectability_summary.imf_parameters_json).get("log10_m_c_msun"),
            "total_initial_count": float(detectability_model["total_initial_count"]),
            "survival_fraction": float(detectability_model["selection_fraction"]),
            "total_initial_stellar_mass_msun": float(detectability_total_mass),
        },
        {
            "model": "Best detectability-corrected shared-IMF two component",
            "component": "in_situ",
            "imf_family": shared_components.loc["in_situ", "imf_family"],
            "radial_model": shared_components.loc["in_situ", "radial_model"],
            "alpha_dndm": shared_imf.get("alpha_dndm"),
            "log10_m_c_msun": shared_imf.get("log10_m_c_msun"),
            "total_initial_count": float(shared_components.loc["in_situ", "total_initial_count"]),
            "survival_fraction": float(shared_components.loc["in_situ", "survival_fraction"]),
            "total_initial_stellar_mass_msun": float(shared_component_masses["in_situ"]),
        },
        {
            "model": "Best detectability-corrected shared-IMF two component",
            "component": "accreted",
            "imf_family": shared_components.loc["accreted", "imf_family"],
            "radial_model": shared_components.loc["accreted", "radial_model"],
            "alpha_dndm": shared_imf.get("alpha_dndm"),
            "log10_m_c_msun": shared_imf.get("log10_m_c_msun"),
            "total_initial_count": float(shared_components.loc["accreted", "total_initial_count"]),
            "survival_fraction": float(shared_components.loc["accreted", "survival_fraction"]),
            "total_initial_stellar_mass_msun": float(shared_component_masses["accreted"]),
        },
        {
            "model": r"Best detectability-corrected split-$\alpha$ two component",
            "component": "in_situ",
            "imf_family": split_alpha_components.loc["in_situ", "imf_family"],
            "radial_model": split_alpha_components.loc["in_situ", "radial_model"],
            "alpha_dndm": split_alpha_in_situ_imf.get("alpha_dndm"),
            "log10_m_c_msun": split_alpha_in_situ_imf.get("log10_m_c_msun"),
            "total_initial_count": float(split_alpha_components.loc["in_situ", "total_initial_count"]),
            "survival_fraction": float(split_alpha_components.loc["in_situ", "survival_fraction"]),
            "total_initial_stellar_mass_msun": float(split_alpha_component_masses["in_situ"]),
        },
        {
            "model": r"Best detectability-corrected split-$\alpha$ two component",
            "component": "accreted",
            "imf_family": split_alpha_components.loc["accreted", "imf_family"],
            "radial_model": split_alpha_components.loc["accreted", "radial_model"],
            "alpha_dndm": split_alpha_accreted_imf.get("alpha_dndm"),
            "log10_m_c_msun": split_alpha_accreted_imf.get("log10_m_c_msun"),
            "total_initial_count": float(split_alpha_components.loc["accreted", "total_initial_count"]),
            "survival_fraction": float(split_alpha_components.loc["accreted", "survival_fraction"]),
            "total_initial_stellar_mass_msun": float(split_alpha_component_masses["accreted"]),
        },
    ]
    return pd.DataFrame(rows)


def build_paper_summary_payload(
    fit_catalog: pd.DataFrame,
    joint_results: dict[str, object],
    detectability_results: dict[str, object],
    shared_results: dict[str, object],
    split_alpha_results: dict[str, object],
    conditional_class_table: pd.DataFrame,
) -> dict[str, object]:
    best_single = joint_results["summary_table"].iloc[0]
    single_total_mass = best_single_model_total_initial_stellar_mass(joint_results)
    detectability_total_mass = detectability_corrected_single_total_initial_stellar_mass(detectability_results)
    detectability_summary = detectability_results["final_payload"]["summary"]
    detectability_model = detectability_results["final_payload"]["model"]
    best_shared = shared_results["summary_table"].iloc[0]
    shared_component_masses = shared_imf_component_total_initial_stellar_masses(shared_results)
    best_split_alpha = split_alpha_results["summary_table"].iloc[0]
    split_alpha_component_masses = split_alpha_component_total_initial_stellar_masses(split_alpha_results)
    best_shared_components = shared_results["best_component_summary_table"].set_index("component_label")
    comparison = conditional_class_table.set_index("model_class")
    return {
        "n_clusters_total": int(len(fit_catalog)),
        "n_clusters_in_situ": int((fit_catalog["origin_flag"] == 1).sum()),
        "n_clusters_accreted": int((fit_catalog["origin_flag"] == 0).sum()),
        "single_component_best_model": {
            "imf_family": str(best_single["imf_family"]),
            "radial_model": str(best_single["radial_model"]),
            "log_likelihood": float(best_single["log_likelihood"]),
            "total_initial_count": float(best_single["total_initial_count"]),
            "total_initial_stellar_mass_msun": float(single_total_mass),
            "survival_fraction": float(best_single["survival_fraction"]),
            "imf_parameters": json.loads(best_single["imf_parameters_json"]),
        },
        "detectability_corrected_single_component_model": {
            "imf_family": str(detectability_summary.imf_family),
            "radial_model": str(detectability_summary.radial_model),
            "log_likelihood": float(detectability_summary.log_likelihood),
            "total_initial_count": float(detectability_model["total_initial_count"]),
            "total_initial_stellar_mass_msun": float(detectability_total_mass),
            "selection_fraction": float(detectability_model["selection_fraction"]),
            "raw_survival_fraction": float(detectability_model["raw_survival_fraction"]),
            "mean_detectability": float(
                detectability_model["selection_fraction"] / max(detectability_model["raw_survival_fraction"], 1.0e-12)
            ),
            "imf_parameters": json.loads(detectability_summary.imf_parameters_json),
            "count_ratio_vs_baseline": float(
                detectability_model["total_initial_count"] / max(float(best_single["total_initial_count"]), 1.0e-12)
            ),
        },
        "shared_imf_two_component_best_model": {
            "imf_family": str(best_shared["imf_family"]),
            "in_situ_radial_model": str(best_shared["in_situ_radial_model"]),
            "accreted_radial_model": str(best_shared["accreted_radial_model"]),
            "raw_log_likelihood": float(best_shared["log_likelihood"]),
            "conditional_log_likelihood": float(
                comparison.loc["two_component_shared_imf", "conditional_log_likelihood"]
            ),
            "conditional_bic": float(comparison.loc["two_component_shared_imf", "conditional_bic"]),
            "delta_conditional_bic": float(comparison.loc["two_component_shared_imf", "delta_conditional_bic"]),
            "total_initial_count": float(best_shared["total_initial_count"]),
            "total_initial_stellar_mass_msun": float(sum(shared_component_masses.values())),
            "in_situ_total_initial_count": float(best_shared_components.loc["in_situ", "total_initial_count"]),
            "in_situ_total_initial_stellar_mass_msun": float(shared_component_masses["in_situ"]),
            "accreted_total_initial_count": float(best_shared_components.loc["accreted", "total_initial_count"]),
            "accreted_total_initial_stellar_mass_msun": float(shared_component_masses["accreted"]),
            "in_situ_survival_fraction": float(best_shared_components.loc["in_situ", "survival_fraction"]),
            "accreted_survival_fraction": float(best_shared_components.loc["accreted", "survival_fraction"]),
            "mean_detectability": float(best_shared["mean_detectability"]),
            "shared_imf_parameters": json.loads(best_shared["shared_imf_parameters_json"]),
        },
        "split_alpha_two_component_best_model": {
            "raw_log_likelihood": float(best_split_alpha["log_likelihood"]),
            "conditional_log_likelihood": float(
                comparison.loc["two_component_split_alpha", "conditional_log_likelihood"]
            ),
            "conditional_bic": float(comparison.loc["two_component_split_alpha", "conditional_bic"]),
            "delta_conditional_bic": float(comparison.loc["two_component_split_alpha", "delta_conditional_bic"]),
            "total_initial_count": float(best_split_alpha["total_initial_count"]),
            "total_initial_stellar_mass_msun": float(sum(split_alpha_component_masses.values())),
            "mean_detectability": float(best_split_alpha["mean_detectability"]),
            "shared_log10_m_c_msun": float(best_split_alpha["shared_log10_m_c_msun"]),
            "in_situ_alpha_dndm": float(best_split_alpha["in_situ_alpha_dndm"]),
            "accreted_alpha_dndm": float(best_split_alpha["accreted_alpha_dndm"]),
        },
        "population_model_class_comparison": conditional_class_table.to_dict(orient="records"),
        "lower_bound_total_initial_gcs_preferred_shared_imf": float(best_shared["total_initial_count"]),
    }


def write_single_component_table_tex(table: pd.DataFrame, output_path: Path) -> None:
    if "best_parameters_label" in table.columns:
        lines = [
            r"\begin{table*}",
            r"\caption{Profile-likelihood maxima of the longitude-aware detectability-corrected single-component IMF families within the preferred smooth \texttt{logpoly3} radial model. At each IMF grid point the nuisance radial parameters and the iterative detectability correction were re-optimized. Lower $\Delta$BIC is preferred, and $\Delta$BIC$=0$ marks the best family maximum.}",
            r"\label{tab:single_component_models}",
            r"\small",
            r"\centering",
            r"\begin{tabular}{lcccc}",
            r"\hline",
            r"IMF family & Best IMF parameters & $\log \mathcal{L}$ & $\Delta$BIC & $N_0$ \\",
            r"\hline",
        ]
        for row in table.itertuples(index=False):
            if abs(float(row.total_initial_count)) >= 1.0e4:
                exponent = int(np.floor(np.log10(abs(float(row.total_initial_count)))))
                mantissa = float(row.total_initial_count) / (10.0 ** exponent)
                count_label = rf"${mantissa:.2f}\times10^{{{exponent}}}$"
            else:
                count_label = f"{float(row.total_initial_count):.1f}"
            lines.append(
                f"{row.model_family} & {row.best_parameters_label} & {row.log_likelihood:.2f} & "
                f"{row.delta_bic_from_best:.2f} & {count_label} \\\\"
            )
        lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
        output_path.write_text("\n".join(lines) + "\n")
        return

    lines = [
        r"\begin{table}",
        r"\caption{Performance of the longitude-aware detectability-corrected single-component models.}",
        r"\label{tab:single_component_models}",
        r"\small",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Model & $\log \mathcal{L}$ & RMS residual & $\langle C \rangle$ & $N_0$ \\",
        r"\hline",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"{row.model_label} & {row.log_likelihood:.2f} & {row.rms_residual_sigma_2d:.3f} & "
            f"{row.mean_detectability:.3f} & {row.total_initial_count:.1f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}}", r"\end{table}"])
    output_path.write_text("\n".join(lines) + "\n")


def write_population_class_table_tex(table: pd.DataFrame, output_path: Path) -> None:
    lines = [
        r"\begin{table*}",
        r"\caption{Comparison of the one- and two-component model classes. "
        r"The conditional BIC restores the common count-partition constant for the fixed in-situ/accreted split. "
        r"Lower conditional BIC is preferred; the reported $\Delta$BIC is measured relative to the best model, so $\Delta$BIC$=0$ marks the preferred model and larger positive values are worse.}",
        r"\label{tab:population_model_classes}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Model class & $k$ & raw $\log \mathcal{L}$ & conditional BIC & $\Delta$BIC$_{\rm cond}$ from best \\",
        r"\hline",
    ]
    for row in table.itertuples(index=False):
        model_label = pretty_population_model_class(row.model_class)
        lines.append(
            f"{model_label} & {row.n_parameters:d} & {row.raw_log_likelihood:.2f} & "
            f"{row.conditional_bic:.2f} & {row.delta_conditional_bic:.2f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
    output_path.write_text("\n".join(lines) + "\n")


def write_key_results_table_tex(table: pd.DataFrame, output_path: Path) -> None:
    lines = [
        r"\begin{table*}",
        r"\caption{Best-fit parameters for the preferred single-component and two-component models.}",
        r"\label{tab:key_results}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccccc}",
        r"\hline",
        r"Model & Component & IMF & $A(a)$ & $\alpha$ & $\log_{10}(M_{\rm c}/{\rm M}_\odot)$ & $N_0$ & $f_{\rm sel}$ & $M_{\star,0}$ [$10^8\,{\rm M}_\odot$] \\",
        r"\hline",
    ]
    for row in table.itertuples(index=False):
        alpha_value = "..." if pd.isna(row.alpha_dndm) else f"{row.alpha_dndm:.3f}"
        mc_value = "..." if pd.isna(row.log10_m_c_msun) else f"{row.log10_m_c_msun:.3f}"
        lines.append(
            f"{row.model} & {pretty_component_label(row.component)} & {row.imf_family} & {row.radial_model} & "
            f"{alpha_value} & {mc_value} & {row.total_initial_count:.1f} & {row.survival_fraction:.3f} & "
            f"{row.total_initial_stellar_mass_msun / 1.0e8:.3f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}}", r"\end{table*}"])
    output_path.write_text("\n".join(lines) + "\n")


def write_summary_macros_tex(summary_payload: dict[str, object], output_path: Path) -> None:
    single = summary_payload["single_component_best_model"]
    detectability = summary_payload["detectability_corrected_single_component_model"]
    shared = summary_payload["shared_imf_two_component_best_model"]
    comparison_rows = summary_payload["population_model_class_comparison"]
    comparison = {row["model_class"]: row for row in comparison_rows}

    lines = [
        rf"\providecommand{{\SingleComponentNzero}}{{{single['total_initial_count']:.1f}}}",
        rf"\providecommand{{\SingleComponentMassZeroEight}}{{{single['total_initial_stellar_mass_msun'] / 1.0e8:.2f}}}",
        rf"\providecommand{{\DetectabilityCorrectedNzero}}{{{detectability['total_initial_count']:.1f}}}",
        rf"\providecommand{{\DetectabilityCorrectedMassZeroEight}}{{{detectability['total_initial_stellar_mass_msun'] / 1.0e8:.2f}}}",
        rf"\providecommand{{\DetectabilityMeanCompleteness}}{{{detectability['mean_detectability']:.3f}}}",
        rf"\providecommand{{\DetectabilityCountRatio}}{{{detectability['count_ratio_vs_baseline']:.2f}}}",
        rf"\providecommand{{\SharedTwoComponentNzero}}{{{shared['total_initial_count']:.1f}}}",
        rf"\providecommand{{\SharedTwoComponentMassZeroEight}}{{{shared['total_initial_stellar_mass_msun'] / 1.0e8:.2f}}}",
        rf"\providecommand{{\SharedInSituNzero}}{{{shared['in_situ_total_initial_count']:.1f}}}",
        rf"\providecommand{{\SharedInSituMassZeroEight}}{{{shared['in_situ_total_initial_stellar_mass_msun'] / 1.0e8:.2f}}}",
        rf"\providecommand{{\SharedAccretedNzero}}{{{shared['accreted_total_initial_count']:.1f}}}",
        rf"\providecommand{{\SharedAccretedMassZeroEight}}{{{shared['accreted_total_initial_stellar_mass_msun'] / 1.0e8:.2f}}}",
        rf"\providecommand{{\SharedMeanDetectability}}{{{shared['mean_detectability']:.3f}}}",
        rf"\providecommand{{\SplitAlphaTwoComponentNzero}}{{{summary_payload['split_alpha_two_component_best_model']['total_initial_count']:.1f}}}",
        rf"\providecommand{{\SplitAlphaTwoComponentMassZeroEight}}{{{summary_payload['split_alpha_two_component_best_model']['total_initial_stellar_mass_msun'] / 1.0e8:.2f}}}",
        rf"\providecommand{{\SharedVsSingleDeltaBICCond}}{{{comparison['single_population']['delta_conditional_bic']:.1f}}}",
        rf"\providecommand{{\SplitAlphaVsSharedDeltaBICCond}}{{{comparison['two_component_split_alpha']['delta_conditional_bic']:.1f}}}",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def mean_cluster_initial_mass_from_grid(log_mass_grid: np.ndarray, imf_density_grid: np.ndarray) -> float:
    return float(
        np.trapezoid(
            np.power(10.0, np.asarray(log_mass_grid, dtype=float)) * np.asarray(imf_density_grid, dtype=float),
            np.asarray(log_mass_grid, dtype=float),
        )
    )


def best_single_model_total_initial_stellar_mass(joint_results: dict[str, object]) -> float:
    context = joint_results["context"]
    best_model = joint_results["best_payload"]["model"]
    return total_initial_stellar_mass_above_log_mass(
        float(best_model["total_initial_count"]),
        np.asarray(context.log_mass_grid, dtype=float),
        np.asarray(best_model["imf_density_grid"], dtype=float),
        PAPER_LOG_MASS_MIN,
    )


def best_single_model_total_initial_count(joint_results: dict[str, object]) -> float:
    context = joint_results["context"]
    best_model = joint_results["best_payload"]["model"]
    return total_initial_count_above_log_mass(
        float(best_model["total_initial_count"]),
        log_mass_grid=np.asarray(context.log_mass_grid, dtype=float),
        imf_density_grid=np.asarray(best_model["imf_density_grid"], dtype=float),
        log_mass_min=PAPER_LOG_MASS_MIN,
    )


def detectability_corrected_single_total_initial_stellar_mass(detectability_results: dict[str, object]) -> float:
    context = detectability_results["base_context"]
    best_model = detectability_results["final_payload"]["model"]
    return total_initial_stellar_mass_above_log_mass(
        float(best_model["total_initial_count"]),
        np.asarray(context.log_mass_grid, dtype=float),
        np.asarray(best_model["imf_density_grid"], dtype=float),
        PAPER_LOG_MASS_MIN,
    )


def detectability_corrected_single_total_initial_count(detectability_results: dict[str, object]) -> float:
    context = detectability_results["base_context"]
    best_model = detectability_results["final_payload"]["model"]
    return total_initial_count_above_log_mass(
        float(best_model["total_initial_count"]),
        log_mass_grid=np.asarray(context.log_mass_grid, dtype=float),
        imf_density_grid=np.asarray(best_model["imf_density_grid"], dtype=float),
        log_mass_min=PAPER_LOG_MASS_MIN,
    )


def shared_imf_component_total_initial_stellar_masses(shared_results: dict[str, object]) -> dict[str, float]:
    best_model = shared_results["best_result"]["final_payload"]["model"]
    contexts = shared_results["contexts"]
    reference_context = contexts["in_situ"]
    mean_initial_mass = mean_cluster_initial_mass_from_grid(
        log_mass_grid=np.asarray(reference_context.log_mass_grid, dtype=float),
        imf_density_grid=np.asarray(best_model["imf_density_grid"], dtype=float),
    )
    return {
        component_label: float(mean_initial_mass * float(best_model["total_initial_count"][component_label]))
        for component_label in ("in_situ", "accreted")
    }


def split_alpha_component_total_initial_stellar_masses(split_alpha_results: dict[str, object]) -> dict[str, float]:
    best_model = split_alpha_results["best_result"]["final_payload"]["model"]
    contexts = split_alpha_results["contexts"]
    component_masses: dict[str, float] = {}
    for component_label in ("in_situ", "accreted"):
        mean_initial_mass = mean_cluster_initial_mass_from_grid(
            log_mass_grid=np.asarray(contexts[component_label].log_mass_grid, dtype=float),
            imf_density_grid=np.asarray(best_model["imf_density_grid"][component_label], dtype=float),
        )
        component_masses[component_label] = float(
            mean_initial_mass * float(best_model["total_initial_count"][component_label])
        )
    return component_masses


def imf_family_color(imf_family: str) -> str:
    colors = {
        "schechter": "#d95f02",
        "lognormal": "#1f78b4",
        "powerlaw": "#1b9e77",
    }
    return colors.get(imf_family, "#4c4c4c")


def pretty_component_label(component_label: str) -> str:
    mapping = {
        "all": "all",
        "in_situ": "in-situ",
        "accreted": "accreted",
    }
    return mapping.get(component_label, component_label)


def pretty_population_model_class(model_class: str) -> str:
    mapping = {
        "single_population": "single population",
        "two_component_shared_imf": "two component, shared IMF",
        "two_component_split_alpha": r"two component, split $\alpha$",
    }
    return mapping.get(model_class, model_class)


def short_model_label(imf_family: str, radial_model: str) -> str:
    imf_map = {"schechter": "Sch", "lognormal": "Logn", "powerlaw": "PL"}
    radial_map = {"logpoly3": "poly", "step5": "step"}
    return f"{imf_map.get(imf_family, imf_family)}+{radial_map.get(radial_model, radial_model)}"
