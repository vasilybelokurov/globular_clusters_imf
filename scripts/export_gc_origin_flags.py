from __future__ import annotations

from pathlib import Path

from globular_clusters_imf.catalog import (
    attach_local_origin_flags_to_baumgardt_catalog,
    export_local_gc_origin_flags,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    origin_flags = export_local_gc_origin_flags(project_root)
    augmented_catalog = attach_local_origin_flags_to_baumgardt_catalog(
        project_root,
        origin_flags=origin_flags,
    )

    origin_output = project_root / "data" / "processed" / "gc_origin_flags.csv"
    augmented_output = project_root / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    n_matches = int(augmented_catalog["origin_flag"].notna().sum())

    print(f"Wrote {len(origin_flags)} GC origin rows to {origin_output}")
    print(f"Wrote {len(augmented_catalog)} Baumgardt rows with {n_matches} origin matches to {augmented_output}")


if __name__ == "__main__":
    main()
