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

    from globular_clusters_imf.plotting import plot_gc_detectability_histograms_by_present_mass

    catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog.csv"
    catalog = pd.read_csv(catalog_path)

    figures_dir = project_root / "outputs" / "figures"
    tables_dir = project_root / "outputs" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    output_figure = figures_dir / "gc_detectability_dsun_absb_by_present_mass.png"
    mass_summary_table, histogram_table = plot_gc_detectability_histograms_by_present_mass(
        catalog=catalog,
        output_path=output_figure,
        n_mass_bins=3,
        n_distance_bins=12,
        n_latitude_bins=12,
    )
    mass_summary_table.to_csv(tables_dir / "gc_detectability_present_mass_bins.csv", index=False)
    histogram_table.to_csv(tables_dir / "gc_detectability_histogram_table.csv", index=False)

    print(f"Wrote figure: {output_figure}")
    print("Present-day mass bins:")
    print(
        mass_summary_table[
            [
                "mass_bin_index",
                "n_clusters",
                "log10_present_mass_left_edge",
                "log10_present_mass_right_edge",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
