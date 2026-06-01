from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(project_root / ".cache"))
    (project_root / ".mplconfig").mkdir(parents=True, exist_ok=True)
    (project_root / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)

    from globular_clusters_imf.model import fit_catalog_models
    from globular_clusters_imf.paper_assets import plot_catalog_mass_semimajor_axis_overview_for_paper
    from globular_clusters_imf.smooth_survivability import build_smooth_survivability_grid

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
    output_path = project_root / "paper" / "figures" / "catalog_mass_semimajor_axis_overview.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_catalog_mass_semimajor_axis_overview_for_paper(
        fit_catalog,
        smooth_survivability_map,
        output_path,
        eta_boundary_maps=eta_boundary_maps,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
