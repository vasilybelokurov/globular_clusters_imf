from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

from globular_clusters_imf.joint_model import JointModelSpec  # noqa: E402
from globular_clusters_imf.model import fit_catalog_models  # noqa: E402
from run_profile_map_and_exact_mcmc_schechter_powerlaw_a import (  # noqa: E402
    _evaluate_theta_multistart,
    _select_anchor_start_state,
)


def _load_catalog() -> pd.DataFrame:
    catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog.csv"
    return pd.read_csv(catalog_path)


def _entry_from_result_payload(result: dict[str, object], theta: np.ndarray) -> dict[str, object]:
    raw_parameters = np.asarray(result["final_payload"]["raw_parameters"], dtype=float)
    return {
        "theta": np.asarray(theta, dtype=float),
        "log_posterior": float(result["final_payload"]["summary"].log_likelihood),
        "row": {},
        "result": result,
        "start_state": {
            "completeness": np.asarray(result["final_completeness_raw_parameters"], dtype=float),
            "radial": raw_parameters[2:].copy(),
        },
    }


def _mass_curve_average(
    *,
    grid: np.ndarray,
    weights: np.ndarray,
    log_a_grid: np.ndarray,
) -> np.ndarray:
    numerator = np.trapezoid(np.asarray(grid, dtype=float) * np.asarray(weights, dtype=float), log_a_grid, axis=1)
    denominator = np.trapezoid(np.asarray(weights, dtype=float), log_a_grid, axis=1)
    return numerator / np.clip(denominator, 1.0e-12, None)


def _select_surface_rows(posterior_table: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    columns = ["eta_t", "input_alpha_dndm", "input_log10_m_c_msun", "log_likelihood"]
    rows = posterior_table.loc[:, columns].dropna().drop_duplicates(columns[:3]).reset_index(drop=True)
    if len(rows) <= max_samples:
        return rows
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(rows), size=max_samples, replace=False))
    return rows.iloc[indices].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-name", required=True)
    parser.add_argument(
        "--radial-model",
        default="logpoly3",
        choices=["logpoly3", "step5", "powerlaw_a", "cored_powerlaw_a"],
    )
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260530)
    args = parser.parse_args()

    variant_root = PROJECT_ROOT / "variants" / args.variant_name
    tables_dir = variant_root / "outputs" / "tables"
    posterior_path = tables_dir / "exact_parallel_mcmc_posterior_samples.csv"
    best_path = tables_dir / "exact_parallel_mcmc_best_result.pkl"
    if not posterior_path.exists():
        raise FileNotFoundError(f"Missing posterior samples: {posterior_path}")
    if not best_path.exists():
        raise FileNotFoundError(f"Missing best result payload: {best_path}")

    posterior = pd.read_csv(posterior_path)
    sample_rows = _select_surface_rows(posterior, max_samples=int(args.max_samples), seed=int(args.seed))
    catalog = _load_catalog()
    prepared_catalog = fit_catalog_models(catalog, variant_root / "outputs" / "posterior_surface_prepare")["catalog"]
    spec = JointModelSpec(imf_family="schechter", radial_model=str(args.radial_model))
    refined_grid = pd.read_csv(tables_dir / "refined_grid_results.csv")
    bounds = np.array(
        [
            [float(refined_grid["eta_t"].min()), float(refined_grid["eta_t"].max())],
            [float(refined_grid["input_alpha_dndm"].min()), float(refined_grid["input_alpha_dndm"].max())],
            [float(refined_grid["input_log10_m_c_msun"].min()), float(refined_grid["input_log10_m_c_msun"].max())],
        ],
        dtype=float,
    )

    with best_path.open("rb") as handle:
        best_result = pickle.load(handle)
    best_summary = json.loads((tables_dir / "exact_parallel_mcmc_best_result_summary.json").read_text())
    best_theta = np.array(
        [
            float(best_summary["eta_t"]),
            float(best_summary["input_alpha_dndm"]),
            float(best_summary["input_log10_m_c_msun"]),
        ],
        dtype=float,
    )
    anchors = [_entry_from_result_payload(best_result, best_theta)]

    surface_dir = tables_dir / "posterior_selection_surfaces"
    surface_dir.mkdir(parents=True, exist_ok=True)
    meta_rows: list[dict[str, object]] = []
    survival_grids = []
    detectability_grids = []
    survival_mass_curves = []
    detectability_mass_curves = []
    log_mass_grid = None
    log_a_grid = None

    for index, row in sample_rows.iterrows():
        theta = np.array([float(row.eta_t), float(row.input_alpha_dndm), float(row.input_log10_m_c_msun)], dtype=float)
        anchor_state = _select_anchor_start_state(theta=theta, anchors=anchors, bounds=bounds)
        entry = _evaluate_theta_multistart(
            prepared_catalog=prepared_catalog,
            spec=spec,
            theta=theta,
            stage="posterior_surface",
            project_root=variant_root,
            anchor_start_state=anchor_state,
        )
        result = entry["result"]
        if result is None:
            continue
        anchors.append(_entry_from_result_payload(result, theta))

        context = result["final_context"]
        model = result["final_payload"]["model"]
        log_mass_grid = np.asarray(context.log_mass_grid, dtype=float)
        log_a_grid = np.asarray(context.log_a_grid, dtype=float)
        radial_density = np.asarray(model["radial_density_grid"], dtype=float)
        survival_grid = np.asarray(context.survival_probability_grid, dtype=np.float32)
        detectability_grid = np.asarray(result["final_effective_completeness_grid"], dtype=np.float32)
        survival_weights = radial_density[None, :]
        detectability_weights = np.asarray(context.survival_probability_grid, dtype=float) * radial_density[None, :]

        survival_grids.append(survival_grid)
        detectability_grids.append(detectability_grid)
        survival_mass_curves.append(
            _mass_curve_average(grid=survival_grid, weights=survival_weights, log_a_grid=log_a_grid).astype(np.float32)
        )
        detectability_mass_curves.append(
            _mass_curve_average(grid=detectability_grid, weights=detectability_weights, log_a_grid=log_a_grid).astype(np.float32)
        )
        meta = dict(entry["row"])
        meta["posterior_source_index"] = int(index)
        meta_rows.append(meta)
        print(
            f"[surface {len(meta_rows)}/{len(sample_rows)}] eta_t={theta[0]:.3f} "
            f"alpha={theta[1]:.3f} logMc={theta[2]:.3f} logL={float(entry['row']['log_likelihood']):.3f}"
        )

    if log_mass_grid is None or log_a_grid is None:
        raise RuntimeError("No posterior surface samples were produced.")

    np.savez_compressed(
        surface_dir / "posterior_selection_surface_samples.npz",
        log_mass_grid=np.asarray(log_mass_grid, dtype=np.float32),
        log_a_grid=np.asarray(log_a_grid, dtype=np.float32),
        survival_probability=np.asarray(survival_grids, dtype=np.float32),
        effective_detectability=np.asarray(detectability_grids, dtype=np.float32),
        survival_mass_curve=np.asarray(survival_mass_curves, dtype=np.float32),
        detectability_mass_curve=np.asarray(detectability_mass_curves, dtype=np.float32),
    )
    pd.DataFrame(meta_rows).to_csv(surface_dir / "posterior_selection_surface_samples.csv", index=False)
    summary = {
        "variant_name": args.variant_name,
        "radial_model": args.radial_model,
        "requested_max_samples": int(args.max_samples),
        "n_surface_samples": int(len(meta_rows)),
        "outputs": {
            "npz": str(surface_dir / "posterior_selection_surface_samples.npz"),
            "metadata": str(surface_dir / "posterior_selection_surface_samples.csv"),
        },
    }
    (surface_dir / "posterior_selection_surface_samples_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
