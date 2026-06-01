from __future__ import annotations

from pathlib import Path

from globular_clusters_imf.catalog import (
    attach_local_gc_chemistry_to_baumgardt_catalog,
    export_local_gc_chemistry_markers,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    chemistry = export_local_gc_chemistry_markers(PROJECT_ROOT)
    attach_local_gc_chemistry_to_baumgardt_catalog(
        PROJECT_ROOT,
        chemistry_markers=chemistry,
    )


if __name__ == "__main__":
    main()
