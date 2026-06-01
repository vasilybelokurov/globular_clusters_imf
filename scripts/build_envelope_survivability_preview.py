from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(project_root / ".cache"))
    (project_root / ".mplconfig").mkdir(parents=True, exist_ok=True)
    (project_root / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    from globular_clusters_imf.envelope_survivability import build_envelope_survivability_grid
    from globular_clusters_imf.model import fit_catalog_models
    from globular_clusters_imf.paper_assets import centers_to_edges

    catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    fit_catalog = fit_catalog_models(catalog, project_root)["catalog"]
    survivability_map = build_envelope_survivability_grid(fit_catalog)

    figures_dir = project_root / "outputs" / "figures"
    tables_dir = project_root / "outputs" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "envelope_survivability_preview.png"

    radius_grid = np.asarray(survivability_map["semi_major_axis_grid_kpc"], dtype=float)
    log_mass_grid = np.asarray(survivability_map["log_mass_grid"], dtype=float)
    survival_probability = np.asarray(survivability_map["survival_probability"], dtype=float)
    boundary_log_mass = np.asarray(survivability_map["boundary_log10_msun"], dtype=float)

    semi_major_axis = fit_catalog["semi_major_axis_kpc"].to_numpy(dtype=float)
    initial_mass = fit_catalog["initial_mass_msun"].to_numpy(dtype=float)

    x_limits = (
        float(semi_major_axis.min() / 1.15),
        float(semi_major_axis.max() * 1.15),
    )
    y_limits = (1.0e3, 3.0e7)
    radius_edges_core = 10.0 ** centers_to_edges(np.log10(radius_grid))
    radius_edges = np.concatenate(([x_limits[0]], radius_edges_core, [x_limits[1]]))
    mass_edges_core = np.power(10.0, centers_to_edges(log_mass_grid))
    mass_edges = np.concatenate(([y_limits[0]], mass_edges_core, [y_limits[1]]))
    survival_probability_plot = np.pad(survival_probability, ((1, 1), (1, 1)), mode="edge")
    survival_probability_plot[0, :] = 0.0
    survival_probability_plot[-1, :] = 1.0

    cmap = LinearSegmentedColormap.from_list(
        "survivability_grey_to_white",
        ["#8a8a8a", "#ffffff"],
    )

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.pcolormesh(
        radius_edges,
        mass_edges,
        survival_probability_plot,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        shading="auto",
        rasterized=True,
    )
    ax.plot(
        radius_grid,
        np.power(10.0, boundary_log_mass),
        color="#08519c",
        linewidth=1.8,
    )
    ax.scatter(
        semi_major_axis,
        initial_mass,
        s=28,
        color="black",
        alpha=0.35,
        linewidths=0.0,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_xlabel(r"Orbital semimajor axis $a$ [kpc]")
    ax.set_ylabel(r"Initial mass [$\mathrm{M_\odot}$]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    hull_table = survivability_map["hull_table"].copy()
    hull_table.to_csv(tables_dir / "envelope_survivability_hull_vertices.csv", index=False)
    (tables_dir / "envelope_survivability_preview_summary.json").write_text(
        json.dumps(
            {
                "summary": survivability_map["summary"].__dict__,
                "preview_png": str(output_path),
                "n_catalog_clusters": int(len(fit_catalog)),
            },
            indent=2,
            default=float,
        )
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
