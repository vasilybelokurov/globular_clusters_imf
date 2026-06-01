from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .joint_model import centers_to_edges_local
from .model import powerlaw_integral


def make_all_figures(
    catalog: pd.DataFrame,
    lognormal_result,
    powerlaw_result,
    radial_patch_summary: pd.DataFrame,
    radial_patch_table: pd.DataFrame,
    survivability_map: dict[str, object],
    joint_results: dict[str, object] | None,
    detectability_results: dict[str, object] | None,
    two_component_results: dict[str, object] | None,
    project_root: Path,
) -> None:
    figures_dir = project_root / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_mass_vs_radius(catalog, figures_dir / "figure7_like_mass_vs_radius.png")
    plot_imf_by_radius(
        catalog,
        lognormal_result,
        powerlaw_result,
        figures_dir / "figure8_like_imf_by_radius.png",
    )
    plot_global_imf_with_radial_patches(
        radial_patch_summary,
        radial_patch_table,
        lognormal_result,
        figures_dir / "global_imf_patches_5_radius_bins.png",
    )
    plot_survivability_map(
        catalog,
        survivability_map,
        figures_dir / "survivability_map_initial_mass_vs_radius.png",
    )
    if joint_results is not None:
        plot_joint_model_comparison(
            joint_results["performance_summary_table"],
            joint_results["residual_map_table"],
            figures_dir / "joint_fixed_survival_model_comparison.png",
        )
        plot_joint_model_profiles(
            joint_results["summary_table"],
            joint_results["imf_grid_table"],
            joint_results["radial_grid_table"],
            figures_dir / "joint_fixed_survival_best_model_profiles.png",
        )
        plot_joint_observed_intensity_map(
            catalog,
            joint_results["best_payload"],
            joint_results["point_intensity_grid"],
            joint_results["survival_grid"],
            figures_dir / "joint_fixed_survival_best_observed_intensity.png",
        )
        plot_joint_model_performance(
            catalog,
            joint_results,
            figures_dir / "joint_fixed_survival_best_model_performance.png",
        )
        plot_best_model_triangle(
            joint_results["best_model_uncertainty"]["display_sample_table"],
            joint_results["best_payload"],
            figures_dir / "joint_fixed_survival_best_model_triangle.png",
        )
    if detectability_results is not None:
        plot_detectability_em_completeness_maps(
            detectability_results["completeness_grid_table"],
            figures_dir / "joint_fixed_survival_detectability_em_completeness_by_mass.png",
        )
        plot_detectability_em_iteration_history(
            detectability_results["iteration_history_table"],
            detectability_results["baseline_payload"],
            detectability_results["final_payload"],
            figures_dir / "joint_fixed_survival_detectability_em_convergence.png",
        )
    if two_component_results is not None:
        plot_two_component_best_profiles(
            two_component_results["best_component_summary_table"],
            two_component_results["best_imf_grid_table"],
            two_component_results["best_radial_grid_table"],
            figures_dir / "joint_fixed_survival_two_component_best_profiles.png",
        )


def build_gc_detectability_histogram_tables(
    catalog: pd.DataFrame,
    n_mass_bins: int = 3,
    n_distance_bins: int = 12,
    n_latitude_bins: int = 12,
    mass_edges: np.ndarray | None = None,
    distance_edges: np.ndarray | None = None,
    latitude_edges: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_columns = {"present_mass_msun", "r_sun_kpc", "galactic_b_deg"}
    missing = required_columns.difference(catalog.columns)
    if missing:
        raise KeyError(f"Catalog is missing required columns: {sorted(missing)}")

    working = catalog.loc[
        np.isfinite(catalog["present_mass_msun"])
        & np.isfinite(catalog["r_sun_kpc"])
        & np.isfinite(catalog["galactic_b_deg"])
        & (catalog["present_mass_msun"] > 0.0)
        & (catalog["r_sun_kpc"] > 0.0)
    ].copy()
    if working.empty:
        raise ValueError("No finite GC entries are available for the detectability plot.")

    working["log10_present_mass_msun"] = np.log10(working["present_mass_msun"].to_numpy())
    working["abs_galactic_b_deg"] = np.abs(working["galactic_b_deg"].to_numpy())

    if mass_edges is None:
        mass_quantiles = np.linspace(0.0, 1.0, n_mass_bins + 1)
        mass_edges = np.quantile(working["log10_present_mass_msun"], mass_quantiles)
        mass_edges = np.maximum.accumulate(mass_edges)
        for index in range(1, len(mass_edges)):
            if mass_edges[index] <= mass_edges[index - 1]:
                mass_edges[index] = mass_edges[index - 1] + 1.0e-6
    else:
        mass_edges = np.asarray(mass_edges, dtype=float)
        n_mass_bins = len(mass_edges) - 1

    if distance_edges is None:
        distance_edges = np.geomspace(working["r_sun_kpc"].min(), working["r_sun_kpc"].max(), n_distance_bins + 1)
    else:
        distance_edges = np.asarray(distance_edges, dtype=float)
        n_distance_bins = len(distance_edges) - 1

    if latitude_edges is None:
        latitude_edges = np.linspace(0.0, 90.0, n_latitude_bins + 1)
    else:
        latitude_edges = np.asarray(latitude_edges, dtype=float)
        n_latitude_bins = len(latitude_edges) - 1

    mass_summary_rows: list[dict[str, float | int | str]] = []
    histogram_rows: list[dict[str, float | int | str]] = []

    for mass_bin_index in range(n_mass_bins):
        mass_left = float(mass_edges[mass_bin_index])
        mass_right = float(mass_edges[mass_bin_index + 1])
        if mass_bin_index == n_mass_bins - 1:
            mask = (
                (working["log10_present_mass_msun"] >= mass_left)
                & (working["log10_present_mass_msun"] <= mass_right)
            )
        else:
            mask = (
                (working["log10_present_mass_msun"] >= mass_left)
                & (working["log10_present_mass_msun"] < mass_right)
            )

        subset = working.loc[mask]
        counts_2d, _, _ = np.histogram2d(
            subset["r_sun_kpc"].to_numpy(),
            subset["abs_galactic_b_deg"].to_numpy(),
            bins=[distance_edges, latitude_edges],
        )
        label = (
            rf"${mass_left:.2f} \leq \log_{{10}}(M_{{\rm now}}/{'{'}\rm M_\odot{'}'})"
            + (rf" < {mass_right:.2f}$" if mass_bin_index < n_mass_bins - 1 else rf" \leq {mass_right:.2f}$")
        )

        mass_summary_rows.append(
            {
                "mass_bin_index": mass_bin_index,
                "panel_label": label,
                "n_clusters": int(len(subset)),
                "log10_present_mass_left_edge": mass_left,
                "log10_present_mass_right_edge": mass_right,
                "present_mass_left_edge_msun": float(np.power(10.0, mass_left)),
                "present_mass_right_edge_msun": float(np.power(10.0, mass_right)),
            }
        )

        for i_distance in range(n_distance_bins):
            for i_latitude in range(n_latitude_bins):
                histogram_rows.append(
                    {
                        "mass_bin_index": mass_bin_index,
                        "distance_bin_index": i_distance,
                        "latitude_bin_index": i_latitude,
                        "distance_left_edge_kpc": float(distance_edges[i_distance]),
                        "distance_right_edge_kpc": float(distance_edges[i_distance + 1]),
                        "abs_latitude_left_edge_deg": float(latitude_edges[i_latitude]),
                        "abs_latitude_right_edge_deg": float(latitude_edges[i_latitude + 1]),
                        "gc_count": int(counts_2d[i_distance, i_latitude]),
                    }
                )

    return pd.DataFrame(mass_summary_rows), pd.DataFrame(histogram_rows)


def plot_gc_detectability_histograms_by_present_mass(
    catalog: pd.DataFrame,
    output_path: Path,
    n_mass_bins: int = 3,
    n_distance_bins: int = 12,
    n_latitude_bins: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mass_summary_table, histogram_table = build_gc_detectability_histogram_tables(
        catalog=catalog,
        n_mass_bins=n_mass_bins,
        n_distance_bins=n_distance_bins,
        n_latitude_bins=n_latitude_bins,
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

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(mass_summary_table),
        figsize=(4.3 * len(mass_summary_table), 4.3),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(mass_summary_table) == 1:
        axes = np.array([axes])

    vmax = float(histogram_table["gc_count"].max())
    mesh = None
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
            cmap="cividis",
            vmin=0.0,
            vmax=max(vmax, 1.0),
            rasterized=True,
        )
        axis.set_xscale("log")
        axis.set_title(f"{summary_row.panel_label}\n$N={summary_row.n_clusters}$", fontsize=10)
        axis.set_xlabel(r"$D_{\odot}$ [kpc]")
        axis.set_xlim(distance_edges[0], distance_edges[-1])
        axis.set_ylim(latitude_edges[0], latitude_edges[-1])

    axes[0].set_ylabel(r"$|b|$ [deg]")
    colorbar = fig.colorbar(mesh, ax=axes, pad=0.02)
    colorbar.set_label("GC count per 2D bin")

    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return mass_summary_table, histogram_table


def plot_mass_vs_radius(catalog: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        catalog["semi_major_axis_kpc"],
        catalog["present_mass_msun"],
        s=20,
        alpha=0.7,
        label="Present mass",
    )
    ax.scatter(
        catalog["semi_major_axis_kpc"],
        catalog["initial_mass_msun"],
        s=20,
        alpha=0.7,
        label="Initial mass",
    )
    order = np.argsort(catalog["semi_major_axis_kpc"].to_numpy())
    ax.plot(
        catalog["semi_major_axis_kpc"].to_numpy()[order],
        catalog["survival_mass_cut_msun"].to_numpy()[order],
        color="black",
        linewidth=1.5,
        label="Approx. survival threshold",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Orbital semimajor axis [kpc]")
    ax.set_ylabel("Mass [Msun]")
    ax.set_title("Baumgardt GC Masses Versus Semimajor Axis")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_imf_by_radius(catalog: pd.DataFrame, lognormal_result, powerlaw_result, output_path: Path) -> None:
    bins = [
        ("inner_<3kpc", "Inner: a < 3 kpc"),
        ("mid_3_15kpc", "Middle: 3 <= a < 15 kpc"),
        ("outer_>15kpc", "Outer: a >= 15 kpc"),
    ]
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(9, 10), sharex=True)
    log_grid = np.linspace(3.0, 7.5, 300)
    mass_grid = np.power(10.0, log_grid)
    bin_width = log_grid[1] - log_grid[0]

    for axis, (bin_name, title) in zip(axes, bins, strict=True):
        subset = catalog.loc[catalog["radius_bin_paper"] == bin_name]
        axis.hist(
            subset["log_initial_mass_msun"],
            bins=np.linspace(3.0, 7.5, 19),
            alpha=0.6,
            color="#4c78a8",
            label="Observed surviving clusters",
        )

        if len(subset) > 0:
            weights_lognormal = 1.0 / subset["survival_probability_lognormal"]
            estimated_initial_count = weights_lognormal.sum()
            pdf_lognormal = stats.norm.pdf(
                log_grid,
                loc=lognormal_result.mu_log10_msun,
                scale=lognormal_result.sigma_log10_msun,
            )
            axis.plot(
                log_grid,
                estimated_initial_count * pdf_lognormal * bin_width,
                color="#f58518",
                linewidth=2,
                label="Universal lognormal x inferred normalization",
            )

            total_powerlaw = powerlaw_integral(
                powerlaw_result.mass_min_msun,
                powerlaw_result.mass_max_msun,
                powerlaw_result.beta,
            )
            pdf_powerlaw = (mass_grid**powerlaw_result.beta) * mass_grid * np.log(10.0) / total_powerlaw
            weights_powerlaw = 1.0 / subset["survival_probability_powerlaw"]
            estimated_initial_count_powerlaw = weights_powerlaw.sum()
            axis.plot(
                log_grid,
                estimated_initial_count_powerlaw * pdf_powerlaw * bin_width,
                color="#54a24b",
                linewidth=2,
                linestyle="--",
                label="Universal power law x inferred normalization",
            )

        axis.set_ylabel("Clusters")
        axis.set_title(title)
        axis.legend(frameon=False, fontsize=9)

    axes[-1].set_xlabel("log10(initial mass / Msun)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_global_imf_with_radial_patches(
    radial_patch_summary: pd.DataFrame,
    radial_patch_table: pd.DataFrame,
    lognormal_result,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.5))
    colors = ["#b22222", "#d17c00", "#2c7c31", "#1f78b4", "#6a3d9a"]

    log_grid = np.linspace(
        radial_patch_table["log_mass_left_edge"].min(),
        radial_patch_table["log_mass_right_edge"].max(),
        500,
    )
    global_density = stats.norm.pdf(
        log_grid,
        loc=lognormal_result.mu_log10_msun,
        scale=lognormal_result.sigma_log10_msun,
    )
    ax.plot(
        log_grid,
        global_density,
        color="black",
        linewidth=2.5,
        label="Global lognormal IMF",
        zorder=5,
    )

    bar_levels = np.linspace(1.22, 0.94, len(radial_patch_summary))

    for color, (_, summary_row), bar_level in zip(
        colors,
        radial_patch_summary.sort_values("radius_bin_index_5").iterrows(),
        bar_levels,
        strict=True,
    ):
        bin_table = radial_patch_table.loc[
            radial_patch_table["radius_bin_index_5"] == summary_row["radius_bin_index_5"]
        ].sort_values("mass_bin_index")
        patch_mask = bin_table["is_patch_bin"].to_numpy(dtype=bool)
        leverage_mask = bin_table["is_in_leverage_range"].to_numpy(dtype=bool)

        for start_index, end_index in contiguous_true_segments(patch_mask):
            segment = bin_table.iloc[start_index:end_index]
            ax.stairs(
                values=segment["corrected_density_per_dex"].to_numpy(),
                edges=np.append(
                    segment["log_mass_left_edge"].to_numpy(),
                    segment["log_mass_right_edge"].to_numpy()[-1],
                ),
                baseline=0.0,
                fill=True,
                alpha=0.28,
                color=color,
                linewidth=1.8,
                label=summary_row["radius_bin_label_5"],
            )
            break
        for start_index, end_index in contiguous_true_segments(patch_mask)[1:]:
            segment = bin_table.iloc[start_index:end_index]
            ax.stairs(
                values=segment["corrected_density_per_dex"].to_numpy(),
                edges=np.append(
                    segment["log_mass_left_edge"].to_numpy(),
                    segment["log_mass_right_edge"].to_numpy()[-1],
                ),
                baseline=0.0,
                fill=True,
                alpha=0.28,
                color=color,
                linewidth=1.8,
            )

        if np.any(leverage_mask):
            leverage_left = float(bin_table.loc[leverage_mask, "log_mass_left_edge"].iloc[0])
            leverage_right = float(bin_table.loc[leverage_mask, "log_mass_right_edge"].iloc[-1])
            ax.hlines(
                y=bar_level,
                xmin=leverage_left,
                xmax=leverage_right,
                color=color,
                linewidth=5,
                alpha=0.9,
            )

    ax.set_xlim(
        radial_patch_table["log_mass_left_edge"].min(),
        radial_patch_table["log_mass_right_edge"].max(),
    )
    ax.set_ylim(0.0, 1.32)
    ax.set_xlabel("log10(initial mass / Msun)")
    ax.set_ylabel("IMF density per dex")
    ax.set_title("Global GC IMF With Radius-Dependent Patches")
    ax.text(
        0.02,
        0.96,
        "Colored bars: leverage range after correcting for survival and radial normalization",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def contiguous_true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start_index: int | None = None
    for index, is_true in enumerate(mask):
        if is_true and start_index is None:
            start_index = index
        if not is_true and start_index is not None:
            segments.append((start_index, index))
            start_index = None
    if start_index is not None:
        segments.append((start_index, len(mask)))
    return segments


def plot_survivability_map(
    catalog: pd.DataFrame,
    survivability_map: dict[str, object],
    output_path: Path,
) -> None:
    radius_grid = np.asarray(survivability_map["semi_major_axis_grid_kpc"])
    log_mass_grid = np.asarray(survivability_map["log_mass_grid"])
    survival_probability = np.asarray(survivability_map["survival_probability"])

    radius_edges = 10.0 ** centers_to_edges(np.log10(radius_grid))
    log_mass_edges = centers_to_edges(log_mass_grid)

    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    mesh = ax.pcolormesh(
        radius_edges,
        log_mass_edges,
        survival_probability,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        shading="auto",
    )
    contour = ax.contour(
        radius_grid,
        log_mass_grid,
        survival_probability,
        levels=[0.1, 0.5, 0.9],
        colors="white",
        linewidths=1.0,
    )
    ax.clabel(contour, fmt={0.1: "0.1", 0.5: "0.5", 0.9: "0.9"}, fontsize=8)
    ax.scatter(
        catalog["semi_major_axis_kpc"],
        catalog["log_initial_mass_msun"],
        s=9,
        color="black",
        alpha=0.35,
        linewidths=0.0,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Orbital semimajor axis a [kpc]")
    ax.set_ylabel("log10(initial mass / Msun)")
    ax.set_title("Estimated GC Survivability S(Mini, a)")
    ax.text(
        0.02,
        0.02,
        "Current model: survival averaged over the observed orbit distribution at fixed a",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="white",
        fontsize=8,
    )
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label("Survival probability")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def centers_to_edges(centers: np.ndarray) -> np.ndarray:
    steps = np.diff(centers)
    left_edge = centers[0] - 0.5 * steps[0]
    right_edge = centers[-1] + 0.5 * steps[-1]
    interior = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate(([left_edge], interior, [right_edge]))


def plot_joint_model_profiles(
    summary_table: pd.DataFrame,
    imf_grid_table: pd.DataFrame,
    radial_grid_table: pd.DataFrame,
    output_path: Path,
) -> None:
    best_row = summary_table.iloc[0]
    best_mask = (
        (imf_grid_table["imf_family"] == best_row["imf_family"])
        & (imf_grid_table["radial_model"] == best_row["radial_model"])
    )
    best_imf = imf_grid_table.loc[best_mask].sort_values("log_initial_mass_msun")
    best_radial = radial_grid_table.loc[
        (radial_grid_table["imf_family"] == best_row["imf_family"])
        & (radial_grid_table["radial_model"] == best_row["radial_model"])
    ].sort_values("log10_semi_major_axis_kpc")

    fig, axes = plt.subplots(ncols=2, figsize=(11, 4.8))
    axes[0].plot(
        best_imf["log_initial_mass_msun"],
        best_imf["imf_density_per_dex"],
        color="black",
        linewidth=2.2,
    )
    axes[0].set_xlabel("log10(initial mass / Msun)")
    axes[0].set_ylabel("Intrinsic IMF density per dex")
    axes[0].set_title(
        f"Best IMF: {best_row['imf_family']} + {best_row['radial_model']}"
    )

    axes[1].plot(
        best_radial["semi_major_axis_kpc"],
        best_radial["birth_intensity_per_dex_a"],
        color="#1f78b4",
        linewidth=2.2,
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Orbital semimajor axis a [kpc]")
    axes[1].set_ylabel("Birth intensity per dex in a")
    axes[1].set_title("Best-Fit Radial Birth Profile")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_joint_observed_intensity_map(
    catalog: pd.DataFrame,
    best_payload: dict[str, object],
    point_intensity_grid: np.ndarray,
    survival_grid: dict[str, object],
    output_path: Path,
) -> None:
    radius_grid = np.asarray(survival_grid["semi_major_axis_grid_kpc"])
    log_mass_grid = np.asarray(survival_grid["log_mass_grid"])
    radius_edges = 10.0 ** centers_to_edges(np.log10(radius_grid))
    log_mass_edges = centers_to_edges(log_mass_grid)

    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    mesh = ax.pcolormesh(
        radius_edges,
        log_mass_edges,
        point_intensity_grid,
        cmap="magma",
        shading="auto",
    )
    ax.scatter(
        catalog["semi_major_axis_kpc"],
        catalog["log_initial_mass_msun"],
        s=10,
        color="white",
        alpha=0.45,
        linewidths=0.0,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Orbital semimajor axis a [kpc]")
    ax.set_ylabel("log10(initial mass / Msun)")
    ax.set_title(
        "Best Joint Model: Observed Intensity in (log Mini, log a)"
    )
    spec = best_payload["model"]["spec"]
    ax.text(
        0.02,
        0.98,
        f"{spec.imf_family} IMF + {spec.radial_model} radial model",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=9,
    )
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label("Observed point-process intensity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_joint_model_performance(
    catalog: pd.DataFrame,
    joint_results: dict[str, object],
    output_path: Path,
) -> None:
    context = joint_results["context"]
    point_intensity_grid = np.asarray(joint_results["point_intensity_grid"])
    log_mass_grid = np.asarray(context.log_mass_grid)
    log_a_grid = np.asarray(context.log_a_grid)
    log_mass_edges = centers_to_edges(log_mass_grid)
    log_a_edges = centers_to_edges(log_a_grid)
    cell_weights = point_intensity_grid * np.diff(log_mass_edges)[:, None] * np.diff(log_a_edges)[None, :]

    mass_bin_edges = np.linspace(log_mass_edges[0], log_mass_edges[-1], 13)
    log_a_bin_edges = np.linspace(log_a_edges[0], log_a_edges[-1], 10)

    observed_mass_counts, _ = np.histogram(context.log_mass_data, bins=mass_bin_edges)
    observed_a_counts, _ = np.histogram(context.log_a_data, bins=log_a_bin_edges)
    observed_2d, _, _ = np.histogram2d(
        context.log_mass_data,
        context.log_a_data,
        bins=[mass_bin_edges, log_a_bin_edges],
    )
    expected_2d = rebin_expected_counts_2d(
        cell_weights,
        log_mass_grid=log_mass_grid,
        log_a_grid=log_a_grid,
        mass_bin_edges=mass_bin_edges,
        log_a_bin_edges=log_a_bin_edges,
    )
    expected_mass_counts = expected_2d.sum(axis=1)
    expected_a_counts = expected_2d.sum(axis=0)
    residual_significance = (observed_2d - expected_2d) / np.sqrt(np.clip(expected_2d, 1.0, None))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    mass_centers = 0.5 * (mass_bin_edges[:-1] + mass_bin_edges[1:])
    axes[0, 0].stairs(observed_mass_counts, mass_bin_edges, color="black", linewidth=1.8, label="Observed")
    axes[0, 0].plot(mass_centers, expected_mass_counts, color="#d95f02", linewidth=2.0, label="Expected")
    axes[0, 0].set_xlabel("log10(initial mass / Msun)")
    axes[0, 0].set_ylabel("Counts")
    axes[0, 0].set_title("Mass Projection")
    axes[0, 0].legend(frameon=False)

    a_edges_kpc = 10.0 ** log_a_bin_edges
    a_centers_kpc = 10.0 ** (0.5 * (log_a_bin_edges[:-1] + log_a_bin_edges[1:]))
    axes[0, 1].stairs(observed_a_counts, a_edges_kpc, color="black", linewidth=1.8, label="Observed")
    axes[0, 1].plot(a_centers_kpc, expected_a_counts, color="#1b9e77", linewidth=2.0, label="Expected")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("Orbital semimajor axis a [kpc]")
    axes[0, 1].set_ylabel("Counts")
    axes[0, 1].set_title("Radius Projection")
    axes[0, 1].legend(frameon=False)

    image = axes[1, 0].pcolormesh(
        10.0 ** log_a_bin_edges,
        mass_bin_edges,
        residual_significance,
        cmap="coolwarm",
        vmin=-3.0,
        vmax=3.0,
        shading="auto",
    )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("Orbital semimajor axis a [kpc]")
    axes[1, 0].set_ylabel("log10(initial mass / Msun)")
    axes[1, 0].set_title("Binned Residual Significance")
    colorbar = fig.colorbar(image, ax=axes[1, 0], pad=0.02)
    colorbar.set_label("(Observed - Expected) / sqrt(Expected)")

    observed_flat = observed_2d.ravel()
    expected_flat = expected_2d.ravel()
    mask = expected_flat > 0.2
    axes[1, 1].scatter(
        expected_flat[mask],
        observed_flat[mask],
        s=24,
        alpha=0.75,
        color="#7570b3",
    )
    max_count = max(float(expected_flat[mask].max(initial=1.0)), float(observed_flat[mask].max(initial=1.0)))
    axes[1, 1].plot([0, max_count], [0, max_count], color="black", linewidth=1.2)
    axes[1, 1].set_xlabel("Expected counts per 2D bin")
    axes[1, 1].set_ylabel("Observed counts per 2D bin")
    axes[1, 1].set_title("Observed vs Expected in 2D Bins")

    best_row = joint_results["summary_table"].iloc[0]
    fig.suptitle(
        f"Best Joint Model Performance: {best_row['imf_family']} + {best_row['radial_model']}",
        y=0.99,
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_joint_model_comparison(
    performance_summary_table: pd.DataFrame,
    residual_map_table: pd.DataFrame,
    output_path: Path,
) -> None:
    ordered = performance_summary_table.sort_values("bic", ascending=True).reset_index(drop=True)
    model_labels = [
        f"{row.imf_family}\n+ {row.radial_model}"
        for row in ordered.itertuples(index=False)
    ]

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    grid = fig.add_gridspec(nrows=3, ncols=3, height_ratios=[0.95, 1.0, 1.0])

    bar_ax = fig.add_subplot(grid[0, :])
    bar_colors = ["#d95f02" if delta == 0 else "#7570b3" for delta in ordered["delta_bic"]]
    bar_ax.barh(np.arange(len(ordered)), ordered["delta_bic"], color=bar_colors, alpha=0.85)
    bar_ax.set_yticks(np.arange(len(ordered)))
    bar_ax.set_yticklabels(model_labels)
    bar_ax.invert_yaxis()
    bar_ax.set_xlabel("Delta BIC relative to best model")
    bar_ax.set_title("Joint Model Comparison")
    for index, row in enumerate(ordered.itertuples(index=False)):
        bar_ax.text(
            row.delta_bic + 0.6,
            index,
            f"RMS sigma={row.rms_residual_sigma_2d:.2f}",
            va="center",
            fontsize=9,
        )

    mesh_axes = []
    image = None
    for panel_index, row in enumerate(ordered.itertuples(index=False)):
        axis = fig.add_subplot(grid[1 + panel_index // 3, panel_index % 3])
        subset = residual_map_table.loc[
            (residual_map_table["imf_family"] == row.imf_family)
            & (residual_map_table["radial_model"] == row.radial_model)
        ].copy()
        pivot = subset.pivot(
            index="log_initial_mass_center_msun",
            columns="semi_major_axis_center_kpc",
            values="residual_sigma",
        )
        mass_centers = pivot.index.to_numpy(dtype=float)
        radius_centers = pivot.columns.to_numpy(dtype=float)
        image = axis.pcolormesh(
            10.0 ** centers_to_edges(np.log10(radius_centers)),
            centers_to_edges(mass_centers),
            pivot.to_numpy(dtype=float),
            cmap="coolwarm",
            vmin=-3.0,
            vmax=3.0,
            shading="auto",
        )
        axis.set_xscale("log")
        axis.set_title(f"{row.imf_family} + {row.radial_model}\nDelta BIC={row.delta_bic:.1f}", fontsize=10)
        if panel_index % 3 == 0:
            axis.set_ylabel("log10(initial mass / Msun)")
        else:
            axis.set_yticklabels([])
        if panel_index // 3 == 1:
            axis.set_xlabel("Orbital semimajor axis a [kpc]")
        else:
            axis.set_xticklabels([])
        mesh_axes.append(axis)

    if image is not None:
        colorbar = fig.colorbar(image, ax=mesh_axes, pad=0.02, shrink=0.78)
        colorbar.set_label("Residual significance")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def rebin_expected_counts_2d(
    cell_weights: np.ndarray,
    log_mass_grid: np.ndarray,
    log_a_grid: np.ndarray,
    mass_bin_edges: np.ndarray,
    log_a_bin_edges: np.ndarray,
) -> np.ndarray:
    mass_index = np.digitize(log_mass_grid, bins=mass_bin_edges[1:-1], right=False)
    a_index = np.digitize(log_a_grid, bins=log_a_bin_edges[1:-1], right=False)
    rebinned = np.zeros((len(mass_bin_edges) - 1, len(log_a_bin_edges) - 1), dtype=float)
    for i_mass, mass_bin in enumerate(mass_index):
        for i_a, a_bin in enumerate(a_index):
            rebinned[mass_bin, a_bin] += cell_weights[i_mass, i_a]
    return rebinned


def plot_best_model_triangle(
    sample_table: pd.DataFrame,
    best_payload: dict[str, object],
    output_path: Path,
) -> None:
    columns = [column for column in sample_table.columns if column != "survival_fraction"]
    samples = sample_table[columns]
    n_dim = len(columns)
    fig, axes = plt.subplots(n_dim, n_dim, figsize=(2.25 * n_dim, 2.25 * n_dim))

    for row in range(n_dim):
        for col in range(n_dim):
            axis = axes[row, col]
            if row < col:
                axis.axis("off")
                continue

            x = samples.iloc[:, col].to_numpy()
            if row == col:
                axis.hist(x, bins=30, color="#4c78a8", alpha=0.8, density=True)
                q16, q50, q84 = np.quantile(x, [0.16, 0.5, 0.84])
                axis.axvline(q50, color="black", linewidth=1.0)
                axis.axvline(q16, color="black", linewidth=0.8, linestyle="--")
                axis.axvline(q84, color="black", linewidth=0.8, linestyle="--")
                axis.set_yticks([])
            else:
                y = samples.iloc[:, row].to_numpy()
                axis.scatter(x, y, s=4, alpha=0.12, color="#1f78b4", linewidths=0.0)

            if row == n_dim - 1:
                axis.set_xlabel(pretty_parameter_label(columns[col]), fontsize=9)
            else:
                axis.set_xticklabels([])
            if col == 0 and row > 0:
                axis.set_ylabel(pretty_parameter_label(columns[row]), fontsize=9)
            elif col != 0:
                axis.set_yticklabels([])

    spec = best_payload["model"]["spec"]
    fig.suptitle(
        f"Best-Model Triangle Plot: {spec.imf_family} + {spec.radial_model}",
        y=0.995,
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_two_component_best_profiles(
    best_component_summary_table: pd.DataFrame,
    best_imf_grid_table: pd.DataFrame,
    best_radial_grid_table: pd.DataFrame,
    output_path: Path,
) -> None:
    component_colors = {"in_situ": "#d95f02", "accreted": "#1b9e77"}
    label_map = {"in_situ": "In-situ", "accreted": "Accreted"}

    fig, axes = plt.subplots(ncols=2, figsize=(11.5, 4.8))

    for component_label in ("in_situ", "accreted"):
        color = component_colors[component_label]
        component_imf = best_imf_grid_table.loc[
            best_imf_grid_table["component_label"] == component_label
        ].sort_values("log_initial_mass_msun")
        component_radial = best_radial_grid_table.loc[
            best_radial_grid_table["component_label"] == component_label
        ].sort_values("log10_semi_major_axis_kpc")
        summary_row = best_component_summary_table.loc[
            best_component_summary_table["component_label"] == component_label
        ].iloc[0]

        axes[0].plot(
            component_imf["log_initial_mass_msun"],
            component_imf["imf_density_per_dex"],
            color=color,
            linewidth=2.2,
            label=(
                f"{label_map[component_label]}: "
                f"{summary_row['imf_family']} + {summary_row['radial_model']}"
            ),
        )
        axes[1].plot(
            component_radial["semi_major_axis_kpc"],
            component_radial["birth_intensity_per_dex_a"],
            color=color,
            linewidth=2.2,
            label=(
                f"{label_map[component_label]}: "
                f"log10 N0={np.log10(summary_row['total_initial_count']):.2f}"
            ),
        )

    axes[0].set_xlabel("log10(initial mass / Msun)")
    axes[0].set_ylabel("Intrinsic IMF density per dex")
    axes[0].set_title("Best Two-Component IMFs")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].set_xscale("log")
    axes[1].set_xlabel("Orbital semimajor axis a [kpc]")
    axes[1].set_ylabel("Birth intensity per dex in a")
    axes[1].set_title("Best Two-Component Radial Birth Profiles")
    axes[1].legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def pretty_parameter_label(column_name: str) -> str:
    replacements = {
        "mu_log10_msun": "mu",
        "sigma_log10_msun": "sigma",
        "alpha_dndm": "alpha",
        "log10_m_c_msun": "log10 Mc",
        "beta1": "beta1",
        "beta2": "beta2",
        "beta3": "beta3",
        "log10_N0": "log10 N0",
        "w_a_bin_1": "w1",
        "w_a_bin_2": "w2",
        "w_a_bin_3": "w3",
        "w_a_bin_4": "w4",
        "w_a_bin_5": "w5",
    }
    return replacements.get(column_name, column_name)


def plot_detectability_em_completeness_maps(
    completeness_grid_table: pd.DataFrame,
    output_path: Path,
) -> None:
    mass_bin_indices = sorted(completeness_grid_table["present_mass_bin_index"].unique())
    if len(mass_bin_indices) <= 3:
        selected_indices = mass_bin_indices
    else:
        selected_indices = [
            mass_bin_indices[len(mass_bin_indices) // 4],
            mass_bin_indices[len(mass_bin_indices) // 2],
            mass_bin_indices[(3 * len(mass_bin_indices)) // 4],
        ]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(selected_indices),
        figsize=(4.0 * len(selected_indices), 4.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(selected_indices) == 1:
        axes = np.array([axes])

    mesh = None
    for axis, mass_bin_index in zip(axes, selected_indices, strict=True):
        subset = completeness_grid_table.loc[
            completeness_grid_table["present_mass_bin_index"] == mass_bin_index
        ].copy()
        subset["distance_left_edge_kpc"] = subset["distance_center_kpc"]
        subset["abs_latitude_left_edge_deg"] = subset["abs_latitude_center_deg"]
        distance_centers = np.sort(subset["distance_center_kpc"].unique())
        latitude_centers = np.sort(subset["abs_latitude_center_deg"].unique())
        distance_edges = centers_to_edges_local(np.log10(distance_centers))
        distance_edges = np.power(10.0, distance_edges)
        latitude_edges = centers_to_edges_local(latitude_centers)
        grid = (
            subset.pivot(index="distance_center_kpc", columns="abs_latitude_center_deg", values="completeness")
            .reindex(index=distance_centers, columns=latitude_centers)
            .to_numpy(dtype=float)
        )
        mesh = axis.pcolormesh(
            distance_edges,
            latitude_edges,
            grid.T,
            shading="auto",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            rasterized=True,
        )
        mass_center = subset["log10_present_mass_center_msun"].iloc[0]
        axis.set_xscale("log")
        axis.set_title(rf"$\log_{{10}}(M_{{\rm now}}/M_\odot) \approx {mass_center:.2f}$", fontsize=10)
        axis.set_xlabel(r"$D_{\odot}$ [kpc]")

    axes[0].set_ylabel(r"$|b|$ [deg]")
    if mesh is not None:
        colorbar = fig.colorbar(mesh, ax=axes, shrink=0.9)
        colorbar.set_label("Detectability completeness")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_detectability_em_iteration_history(
    iteration_history_table: pd.DataFrame,
    baseline_payload: dict[str, object],
    final_payload: dict[str, object],
    output_path: Path,
) -> None:
    if iteration_history_table.empty:
        return

    fig, axes = plt.subplots(ncols=2, figsize=(10.0, 4.2), constrained_layout=True)
    iterations = iteration_history_table["iteration"].to_numpy()

    axes[0].plot(
        np.concatenate(([0], iterations)),
        np.concatenate(([baseline_payload["model"]["total_initial_count"]], iteration_history_table["total_initial_count"])),
        color="#dd8452",
        marker="o",
        linewidth=2.0,
        label=r"$N_0$",
    )
    axes[0].axhline(
        final_payload["model"]["total_initial_count"],
        color="black",
        linewidth=1.0,
        linestyle="--",
        label="Final fit",
    )
    axes[0].set_xlabel("EM iteration")
    axes[0].set_ylabel("Total initial GC count")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(
        iterations,
        iteration_history_table["completeness_mean"],
        color="#4c78a8",
        marker="o",
        linewidth=2.0,
        label="Mean detectability",
    )
    axes[1].plot(
        iterations,
        iteration_history_table["completeness_mass_slope"],
        color="#54a24b",
        marker="s",
        linewidth=1.6,
        label="Mass slope",
    )
    axes[1].plot(
        iterations,
        iteration_history_table["completeness_distance_slope"],
        color="#e45756",
        marker="^",
        linewidth=1.6,
        label="Distance slope",
    )
    axes[1].plot(
        iterations,
        iteration_history_table["completeness_latitude_slope"],
        color="#72b7b2",
        marker="D",
        linewidth=1.6,
        label="Latitude slope",
    )
    axes[1].set_xlabel("EM iteration")
    axes[1].set_ylabel("Completeness parameters")
    axes[1].legend(frameon=False, fontsize=8)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
