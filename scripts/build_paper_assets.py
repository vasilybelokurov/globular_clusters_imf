from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(project_root / ".cache"))
    (project_root / ".mplconfig").mkdir(parents=True, exist_ok=True)
    (project_root / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)

    from globular_clusters_imf.paper_assets import build_paper_assets

    results = build_paper_assets(project_root)
    summary = results["summary_payload"]
    print("Paper assets built.")
    print(
        "Single-component lower bound: "
        f"{summary['single_component_best_model']['total_initial_count']:.1f} clusters, "
        f"{summary['single_component_best_model']['total_initial_stellar_mass_msun'] / 1.0e8:.2f}e8 Msun"
    )
    print(
        "Detectability-corrected single-component lower bound: "
        f"{summary['detectability_corrected_single_component_model']['total_initial_count']:.1f} clusters, "
        f"{summary['detectability_corrected_single_component_model']['total_initial_stellar_mass_msun'] / 1.0e8:.2f}e8 Msun"
    )
    print(
        "Preferred shared-IMF two-component lower bound: "
        f"{summary['shared_imf_two_component_best_model']['total_initial_count']:.1f} clusters, "
        f"{summary['shared_imf_two_component_best_model']['total_initial_stellar_mass_msun'] / 1.0e8:.2f}e8 Msun"
    )


if __name__ == "__main__":
    main()
