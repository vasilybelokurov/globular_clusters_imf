from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd

from scan_schechter_survival_time_multipliers import _plot_logl_vs_multiplier, _plot_properties, _row_from_result


OUTPUT_ROOT = PROJECT_ROOT / "variants" / "global_profiled_schechter_powerlaw_a_logistic"
FIGURES_DIR = OUTPUT_ROOT / "outputs" / "figures"
TABLES_DIR = OUTPUT_ROOT / "outputs" / "tables"
LOG_MASS_MIN = 4.0
N_ITERATIONS = 12

ETA_BOUNDS = (0.1, 3.0)
ALPHA_BOUNDS = (-4.0, -0.2)
LOGMC_BOUNDS = (4.5, 7.5)


def _round_key(eta_t: float, alpha: float, log_mc: float) -> tuple[float, float, float]:
    return (round(float(eta_t), 6), round(float(alpha), 6), round(float(log_mc), 6))


def _clip_and_deduplicate(values: list[float] | np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    low, high = bounds
    clipped = [min(max(float(value), low), high) for value in values]
    unique = sorted({round(value, 6) for value in clipped})
    return np.asarray(unique, dtype=float)


def _survival_grid_override_from_smooth_survival(smooth_survival: dict[str, object]) -> dict[str, object]:
    return {
        "log_mass_grid": np.asarray(smooth_survival["log_mass_grid"], dtype=float),
        "log_a_grid": np.asarray(smooth_survival["log_a_grid"], dtype=float),
        "semi_major_axis_grid_kpc": np.asarray(smooth_survival["semi_major_axis_grid_kpc"], dtype=float),
        "survival_probability": np.asarray(smooth_survival["survival_probability"], dtype=float),
        "selection_offset_dex": 0.0,
        "bandwidth_log10_a_dex": float(smooth_survival["bandwidth_log10_a_dex"]),
        "smooth_survivability_summary": smooth_survival["summary"],
    }


def _start_state_from_result(result: dict[str, object]) -> dict[str, np.ndarray]:
    return {
        "completeness": np.asarray(result["final_completeness_raw_parameters"], dtype=float),
        "radial": np.asarray(result["final_payload"]["radial_parameters_raw"], dtype=float),
    }


def _evaluate_point(
    *,
    prepared_catalog: pd.DataFrame,
    spec,
    eta_t: float,
    alpha: float,
    log_mc: float,
    start_state: dict[str, np.ndarray] | None,
) -> dict[str, object]:
    from globular_clusters_imf.detectability_longitude_model import fit_single_component_detectability_em_with_abs_longitude
    from globular_clusters_imf.smooth_survivability import build_smooth_survivability_grid

    smooth_survival = build_smooth_survivability_grid(
        prepared_catalog,
        eta_t=float(eta_t),
        surface_model="logistic",
    )
    result = fit_single_component_detectability_em_with_abs_longitude(
        prepared_catalog,
        project_root=OUTPUT_ROOT,
        spec=spec,
        n_iterations=N_ITERATIONS,
        fixed_imf_params=np.array([float(alpha), float(log_mc)], dtype=float),
        start_completeness_raw_parameters=None if start_state is None else start_state["completeness"],
        start_radial_params=None if start_state is None else start_state["radial"],
        survival_grid_override=_survival_grid_override_from_smooth_survival(smooth_survival),
    )
    row = _row_from_result(
        eta_t=float(eta_t),
        radial_model=spec.radial_model,
        survival_summary=smooth_survival["summary"],
        result=result,
        log_mass_min=LOG_MASS_MIN,
    )
    radial_params = result["final_payload"]["model"]["radial_parameters"]
    row["beta_log10_a"] = float(radial_params.get("beta_log10_a", np.nan))
    row["gamma_linear_a"] = float(radial_params.get("gamma_linear_a", np.nan))
    row["input_alpha_dndm"] = float(alpha)
    row["input_log10_m_c_msun"] = float(log_mc)
    row["surface_model"] = "logistic"
    return {
        "key": _round_key(eta_t, alpha, log_mc),
        "row": row,
        "result": result,
        "start_state": _start_state_from_result(result),
    }


def _run_grid(
    *,
    prepared_catalog: pd.DataFrame,
    spec,
    eta_grid: np.ndarray,
    alpha_grid: np.ndarray,
    logmc_grid: np.ndarray,
    cache: dict[tuple[float, float, float], dict[str, object]],
    stage: str,
    initial_start_state: dict[str, np.ndarray] | None = None,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    eta_reference_state = initial_start_state

    for eta_index, eta_t in enumerate(eta_grid):
        previous_row_entries: list[dict[str, object] | None] | None = None
        eta_best_entry: dict[str, object] | None = None
        for logmc_index, log_mc in enumerate(logmc_grid):
            alpha_scan = alpha_grid if (eta_index + logmc_index) % 2 == 0 else alpha_grid[::-1]
            current_row_entries_by_alpha: dict[float, dict[str, object]] = {}
            last_entry_in_row: dict[str, object] | None = None
            for alpha in alpha_scan:
                key = _round_key(eta_t, alpha, log_mc)
                cached_entry = cache.get(key)
                if cached_entry is not None:
                    entry = cached_entry
                else:
                    start_state = None
                    if last_entry_in_row is not None:
                        start_state = last_entry_in_row["start_state"]
                    elif previous_row_entries is not None:
                        alpha_matches = [
                            candidate
                            for candidate in previous_row_entries
                            if candidate is not None and round(float(candidate["row"]["input_alpha_dndm"]), 6) == round(float(alpha), 6)
                        ]
                        if alpha_matches:
                            start_state = alpha_matches[0]["start_state"]
                    elif eta_reference_state is not None:
                        start_state = eta_reference_state
                    entry = _evaluate_point(
                        prepared_catalog=prepared_catalog,
                        spec=spec,
                        eta_t=float(eta_t),
                        alpha=float(alpha),
                        log_mc=float(log_mc),
                        start_state=start_state,
                    )
                    cache[key] = entry
                entry["row"]["stage"] = stage
                entries.append(entry)
                current_row_entries_by_alpha[round(float(alpha), 6)] = entry
                last_entry_in_row = entry
                if eta_best_entry is None or float(entry["row"]["log_likelihood"]) > float(eta_best_entry["row"]["log_likelihood"]):
                    eta_best_entry = entry
                print(
                    f"[{stage}] eta_t={eta_t:.3f} alpha={alpha:.3f} logMc={log_mc:.3f} "
                    f"logL={float(entry['row']['log_likelihood']):.3f} beta_a={float(entry['row']['beta_log10_a']):.3f} "
                    f"N0>1e4={float(entry['row']['final_total_initial_count_above_log10_4']):.1f}"
                )
            previous_row_entries = [current_row_entries_by_alpha.get(round(float(alpha), 6)) for alpha in alpha_grid]
        if eta_best_entry is not None:
            eta_reference_state = eta_best_entry["start_state"]
    return entries


def _best_by_eta(table: pd.DataFrame) -> pd.DataFrame:
    best_idx = table.groupby("eta_t")["log_likelihood"].idxmax()
    best_table = table.loc[best_idx].sort_values("eta_t").reset_index(drop=True)
    best_table["best_radial_model"] = "powerlaw_a"
    return best_table


def _save_best_payload(best_entry: dict[str, object]) -> None:
    with (TABLES_DIR / "best_result.pkl").open("wb") as handle:
        pickle.dump(best_entry["result"], handle, protocol=pickle.HIGHEST_PROTOCOL)
    (TABLES_DIR / "best_result_summary.json").write_text(json.dumps(best_entry["row"], indent=2))


def main() -> None:
    from globular_clusters_imf.joint_model import JointModelSpec
    from globular_clusters_imf.model import fit_catalog_models

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    prepared_catalog = fit_catalog_models(catalog, OUTPUT_ROOT)["catalog"]

    spec = JointModelSpec(imf_family="schechter", radial_model="powerlaw_a")

    coarse_eta = np.asarray([0.6, 0.9, 1.2, 1.5, 1.8, 2.1], dtype=float)
    coarse_alpha = np.asarray([-1.8, -1.4, -1.0, -0.6], dtype=float)
    coarse_logmc = np.asarray([6.10, 6.25, 6.40, 6.55], dtype=float)

    cache: dict[tuple[float, float, float], dict[str, object]] = {}
    coarse_entries = _run_grid(
        prepared_catalog=prepared_catalog,
        spec=spec,
        eta_grid=coarse_eta,
        alpha_grid=coarse_alpha,
        logmc_grid=coarse_logmc,
        cache=cache,
        stage="coarse",
    )
    coarse_table = pd.DataFrame([entry["row"] for entry in coarse_entries]).sort_values(
        ["eta_t", "input_alpha_dndm", "input_log10_m_c_msun"]
    ).reset_index(drop=True)
    coarse_table.to_csv(TABLES_DIR / "coarse_grid_results.csv", index=False)

    coarse_best_entry = max(coarse_entries, key=lambda entry: float(entry["row"]["log_likelihood"]))
    coarse_best_row = coarse_best_entry["row"]

    refined_eta = _clip_and_deduplicate(
        [
            float(coarse_best_row["eta_t"]) - 0.15,
            float(coarse_best_row["eta_t"]),
            float(coarse_best_row["eta_t"]) + 0.15,
        ],
        ETA_BOUNDS,
    )
    refined_alpha = _clip_and_deduplicate(
        [
            float(coarse_best_row["input_alpha_dndm"]) - 0.15,
            float(coarse_best_row["input_alpha_dndm"]),
            float(coarse_best_row["input_alpha_dndm"]) + 0.15,
        ],
        ALPHA_BOUNDS,
    )
    refined_logmc = _clip_and_deduplicate(
        [
            float(coarse_best_row["input_log10_m_c_msun"]) - 0.075,
            float(coarse_best_row["input_log10_m_c_msun"]),
            float(coarse_best_row["input_log10_m_c_msun"]) + 0.075,
        ],
        LOGMC_BOUNDS,
    )

    refined_entries = _run_grid(
        prepared_catalog=prepared_catalog,
        spec=spec,
        eta_grid=refined_eta,
        alpha_grid=refined_alpha,
        logmc_grid=refined_logmc,
        cache=cache,
        stage="refined",
        initial_start_state=coarse_best_entry["start_state"],
    )
    refined_table = pd.DataFrame([entry["row"] for entry in refined_entries]).sort_values(
        ["eta_t", "input_alpha_dndm", "input_log10_m_c_msun"]
    ).reset_index(drop=True)
    refined_table.to_csv(TABLES_DIR / "refined_grid_results.csv", index=False)

    unique_entries = list(cache.values())
    all_table = pd.DataFrame([entry["row"] for entry in unique_entries]).sort_values(
        ["eta_t", "input_alpha_dndm", "input_log10_m_c_msun"]
    ).reset_index(drop=True)
    all_table.to_csv(TABLES_DIR / "all_grid_results.csv", index=False)

    profiled_eta_table = _best_by_eta(all_table)
    profiled_eta_table.to_csv(TABLES_DIR / "profiled_eta_results.csv", index=False)
    _plot_logl_vs_multiplier(profiled_eta_table, FIGURES_DIR / "profiled_logl_vs_eta_t.png")
    _plot_properties(profiled_eta_table, FIGURES_DIR / "profiled_properties_vs_eta_t.png")

    best_entry = max(unique_entries, key=lambda entry: float(entry["row"]["log_likelihood"]))
    _save_best_payload(best_entry)

    summary_payload = {
        "surface_model": "logistic",
        "n_detectability_iterations": N_ITERATIONS,
        "model_spec": {"imf_family": spec.imf_family, "radial_model": spec.radial_model},
        "coarse_grid": {
            "eta_t": coarse_eta.tolist(),
            "alpha_dndm": coarse_alpha.tolist(),
            "log10_m_c_msun": coarse_logmc.tolist(),
        },
        "refined_grid": {
            "eta_t": refined_eta.tolist(),
            "alpha_dndm": refined_alpha.tolist(),
            "log10_m_c_msun": refined_logmc.tolist(),
        },
        "n_unique_evaluations": int(len(unique_entries)),
        "global_best": json.loads(pd.Series(best_entry["row"]).to_json()),
        "profiled_eta_best_rows": profiled_eta_table.to_dict(orient="records"),
    }
    (TABLES_DIR / "summary.json").write_text(json.dumps(summary_payload, indent=2))

    print(FIGURES_DIR / "profiled_logl_vs_eta_t.png")
    print(FIGURES_DIR / "profiled_properties_vs_eta_t.png")
    print(TABLES_DIR / "summary.json")


if __name__ == "__main__":
    main()
