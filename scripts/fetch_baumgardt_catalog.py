from __future__ import annotations

from pathlib import Path

from globular_clusters_imf.catalog import fetch_and_prepare_catalog


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    catalog = fetch_and_prepare_catalog(project_root)
    output_path = project_root / "data" / "processed" / "baumgardt_gc_catalog.csv"
    print(f"Wrote {len(catalog)} joined clusters to {output_path}")


if __name__ == "__main__":
    main()
