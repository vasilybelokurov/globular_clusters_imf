from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from build_paper_assets_exact_single_component import PROJECT_ROOT
from globular_clusters_imf.detectability_longitude_model import (
    fit_single_component_detectability_em_with_abs_longitude,
)
from globular_clusters_imf.joint_model import JointModelSpec
from globular_clusters_imf.model import fit_catalog_models
from globular_clusters_imf.paper_assets import plot_detectability_em_convergence_for_paper
from globular_clusters_imf.smooth_survivability import build_smooth_survivability_grid


ILLUSTRATIVE_TRIALS = [
    (0.8, -1.3, 6.3),
    (0.8, -1.0, 6.3),
    (0.8, -0.7, 6.3),
    (1.0, -1.3, 6.3),
    (1.0, -1.0, 6.3),
    (1.0, -0.7, 6.3),
    (1.13, -1.3, 6.3),
    (1.13, -1.0, 6.3),
    (1.13, -0.7, 6.3),
    (1.3, -1.3, 6.3),
    (1.3, -1.0, 6.3),
    (1.3, -0.7, 6.3),
]


def _load_fit_catalog() -> pd.DataFrame:
    catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    return fit_catalog_models(catalog, PROJECT_ROOT)["catalog"]


def _build_illustrative_results() -> list[dict[str, object]]:
    fit_catalog = _load_fit_catalog()
    results = []
    for eta_t, alpha, log_mc in ILLUSTRATIVE_TRIALS:
        survivability_map = build_smooth_survivability_grid(
            fit_catalog,
            eta_t=eta_t,
            surface_model="logistic",
        )
        result = fit_single_component_detectability_em_with_abs_longitude(
            fit_catalog,
            project_root=PROJECT_ROOT,
            spec=JointModelSpec(imf_family="schechter", radial_model="logpoly3"),
            n_iterations=30,
            fixed_imf_params=np.array([alpha, log_mc], dtype=float),
            survival_grid_override=survivability_map,
        )
        results.append(result)
    return results


def _slim_result(result: dict[str, object]) -> dict[str, object]:
    base_context = result["base_context"]
    baseline_model = result["baseline_payload"]["model"]
    return {
        "iteration_history_table": result["iteration_history_table"],
        "baseline_payload": {
            "model": {
                "total_initial_count": float(baseline_model["total_initial_count"]),
                "imf_density_grid": np.asarray(baseline_model["imf_density_grid"], dtype=float),
            }
        },
        "base_context": SimpleNamespace(
            log_mass_data=np.asarray(base_context.log_mass_data, dtype=float),
            log_mass_grid=np.asarray(base_context.log_mass_grid, dtype=float),
        ),
    }


def _slim_results(results: list[dict[str, object]]) -> list[dict[str, object]]:
    return [_slim_result(result) for result in results]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore the cached iteration histories and recompute the illustrative solves.",
    )
    args = parser.parse_args()

    tables_dir = PROJECT_ROOT / "paper" / "tables"
    figures_dir = PROJECT_ROOT / "paper" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    cache_path = tables_dir / "detectability_em_convergence_illustrative_results.pkl"

    if cache_path.exists() and not args.recompute:
        with cache_path.open("rb") as handle:
            illustrative_results = pickle.load(handle)
        illustrative_results = _slim_results(illustrative_results)
    else:
        illustrative_results = _slim_results(_build_illustrative_results())

    with cache_path.open("wb") as handle:
        pickle.dump(illustrative_results, handle)

    plot_detectability_em_convergence_for_paper(
        illustrative_results,
        figures_dir / "detectability_em_convergence.pdf",
    )
    print(f"Wrote {figures_dir / 'detectability_em_convergence.pdf'}")


if __name__ == "__main__":
    main()
