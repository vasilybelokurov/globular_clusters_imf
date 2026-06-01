from __future__ import annotations

import json
import os
from dataclasses import asdict
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
from matplotlib.colors import LinearSegmentedColormap


def centers_to_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    inner = 0.5 * (centers[1:] + centers[:-1])
    first = centers[0] - 0.5 * (centers[1] - centers[0])
    last = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return np.concatenate(([first], inner, [last]))


def _plot_panel(ax, *, catalog: pd.DataFrame, smooth_map: dict[str, object], x_limits: tuple[float, float], y_limits: tuple[float, float], title: str) -> None:
    semi_major_axis = catalog["semi_major_axis_kpc"].to_numpy(dtype=float)
    initial_mass = catalog["initial_mass_msun"].to_numpy(dtype=float)
    log_mass_grid = np.asarray(smooth_map["log_mass_grid"], dtype=float)
    semi_major_axis_grid_kpc = np.asarray(smooth_map["semi_major_axis_grid_kpc"], dtype=float)
    fitted_probability = np.asarray(smooth_map["survival_probability"], dtype=float)
    raw_probability = np.asarray(smooth_map["raw_survival_probability"], dtype=float)
    fitted_boundary_10 = np.asarray(smooth_map["fitted_boundary_10_log10_msun"], dtype=float)
    fitted_boundary_50 = np.asarray(smooth_map["fitted_boundary_50_log10_msun"], dtype=float)
    fitted_boundary_90 = np.asarray(smooth_map["fitted_boundary_90_log10_msun"], dtype=float)

    mass_grid = np.power(10.0, log_mass_grid)
    survivability_cmap = LinearSegmentedColormap.from_list("survivability_grey_to_white", ["#8a8a8a", "#ffffff"])

    radius_edges_core = 10.0 ** centers_to_edges(np.log10(semi_major_axis_grid_kpc))
    radius_edges = np.concatenate(([x_limits[0]], radius_edges_core, [x_limits[1]]))
    log_mass_edges = centers_to_edges(log_mass_grid)
    mass_edges_core = np.power(10.0, log_mass_edges)
    mass_edges = np.concatenate(([y_limits[0]], mass_edges_core, [y_limits[1]]))
    survival_probability_plot = np.pad(fitted_probability, ((1, 1), (1, 1)), mode="edge")
    survival_probability_plot[0, :] = 0.0

    ax.pcolormesh(
        radius_edges,
        mass_edges,
        survival_probability_plot,
        cmap=survivability_cmap,
        vmin=0.0,
        vmax=1.0,
        shading="auto",
        rasterized=True,
    )
    ax.contour(
        semi_major_axis_grid_kpc,
        mass_grid,
        raw_probability,
        levels=[0.1, 0.5, 0.9],
        colors="#4d4d4d",
        linewidths=1.0,
        linestyles="dashed",
    )
    y_min_log10 = np.log10(y_limits[0])
    for color, line_log_mass in [
        ("#74a9cf", fitted_boundary_10),
        ("#2b8cbe", fitted_boundary_50),
        ("#045a8d", fitted_boundary_90),
    ]:
        line_mass = np.power(10.0, np.maximum(line_log_mass, y_min_log10))
        ax.plot(semi_major_axis_grid_kpc, line_mass, color=color, linewidth=1.35)
    ax.scatter(semi_major_axis, initial_mass, s=11, color="black", alpha=0.35, linewidths=0.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_title(title, fontsize=11)


def main() -> None:
    from globular_clusters_imf.model import fit_catalog_models
    from globular_clusters_imf.smooth_survivability import build_smooth_survivability_grid

    outputs_figures = PROJECT_ROOT / "outputs" / "figures"
    outputs_tables = PROJECT_ROOT / "outputs" / "tables"
    outputs_figures.mkdir(parents=True, exist_ok=True)
    outputs_tables.mkdir(parents=True, exist_ok=True)

    catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    catalog_results = fit_catalog_models(catalog, PROJECT_ROOT / "tmp_survivability_eta_t_triptych")
    fit_catalog = catalog_results["catalog"]

    eta_values = [0.2, 1.0, 2.0]
    smooth_maps = [build_smooth_survivability_grid(fit_catalog, eta_t=eta) for eta in eta_values]

    x_limits = (float(fit_catalog["semi_major_axis_kpc"].min() / 1.15), float(fit_catalog["semi_major_axis_kpc"].max() * 1.15))
    y_limits = (1.0e3, 3.0e7)

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), sharex=True, sharey=True)
    for ax, eta_t, smooth_map in zip(axes, eta_values, smooth_maps, strict=True):
        _plot_panel(
            ax,
            catalog=fit_catalog,
            smooth_map=smooth_map,
            x_limits=x_limits,
            y_limits=y_limits,
            title=fr"$\eta_t={eta_t:g}$",
        )
    axes[0].set_ylabel(r"Initial mass $M_{\rm ini}\ [\mathrm{M_\odot}]$")
    for ax in axes:
        ax.set_xlabel(r"Orbital semimajor axis $a$ [kpc]")
    fig.tight_layout()

    output_path = outputs_figures / "survivability_eta_t_triptych.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    summary_rows = []
    for smooth_map in smooth_maps:
        summary = smooth_map["summary"]
        row = asdict(summary)
        summary_rows.append(row)
    summary_table = pd.DataFrame(summary_rows)
    summary_table.to_csv(outputs_tables / "survivability_eta_t_triptych_summary.csv", index=False)
    (outputs_tables / "survivability_eta_t_triptych_summary.json").write_text(json.dumps(summary_rows, indent=2))

    print(output_path)


if __name__ == "__main__":
    main()
