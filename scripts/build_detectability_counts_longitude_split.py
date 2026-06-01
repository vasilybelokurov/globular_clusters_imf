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

    from globular_clusters_imf.paper_assets import plot_detectability_counts_by_longitude_split_for_paper

    catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog.csv"

    catalog = pd.read_csv(catalog_path)

    figures_dir = project_root / "outputs" / "figures"
    tables_dir = project_root / "outputs" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    figure_pdf_path = figures_dir / "detectability_counts_longitude_split.pdf"
    mass_summary_table, histogram_table = plot_detectability_counts_by_longitude_split_for_paper(
        catalog,
        figure_pdf_path,
    )
    figure_png_path = figures_dir / "detectability_counts_longitude_split.png"
    plot_detectability_counts_by_longitude_split_for_paper(
        catalog,
        figure_png_path,
    )
    mass_summary_table.to_csv(tables_dir / "detectability_counts_longitude_split_mass_summary.csv", index=False)
    histogram_table.to_csv(tables_dir / "detectability_counts_longitude_split_histogram_table.csv", index=False)

    subset_summary = (
        mass_summary_table.groupby("longitude_subset", as_index=False)
        .agg(
            longitude_subset_label=("longitude_subset_label", "first"),
            n_clusters_total=("n_clusters", "sum"),
        )
        .sort_values("longitude_subset")
    )
    subset_summary.to_csv(tables_dir / "detectability_counts_longitude_split_subset_summary.csv", index=False)

    print(f"Wrote {figure_pdf_path}")
    print(f"Wrote {figure_png_path}")
    print(f"Wrote {tables_dir / 'detectability_counts_longitude_split_mass_summary.csv'}")
    print(f"Wrote {tables_dir / 'detectability_counts_longitude_split_histogram_table.csv'}")
    print(f"Wrote {tables_dir / 'detectability_counts_longitude_split_subset_summary.csv'}")


if __name__ == "__main__":
    main()
