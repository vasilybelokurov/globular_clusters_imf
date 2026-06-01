from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import numpy as np
import pandas as pd

from globular_clusters_imf.model import fit_catalog_models
from globular_clusters_imf.two_component_model import SharedImfTwoComponentSpec
from run_profile_map_and_exact_mcmc_bk_shared_schechter_two_component import (
    LOG_MASS_MIN,
    SURFACE_MODEL,
    _compute_rhat,
    _corner_plot,
    _evaluate_theta_multistart,
    _load_catalog,
    _select_anchor_start_state,
    _trace_plot,
)
from run_profile_map_and_exact_mcmc_schechter_powerlaw_a import (
    _build_anchor_library,
    _save_best_payload,
    _select_diverse_entries,
)


def _lightweight_entry(entry: dict[str, object]) -> dict[str, object]:
    return {
        "theta": np.asarray(entry["theta"], dtype=float).copy(),
        "log_posterior": float(entry["log_posterior"]),
        "row": dict(entry["row"]),
        "start_state": None,
    }


def _entry_from_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "theta": np.array(
            [float(row["eta_t"]), float(row["input_alpha_dndm"]), float(row["input_log10_m_c_msun"])],
            dtype=float,
        ),
        "log_posterior": float(row["log_likelihood"]),
        "row": dict(row),
        "start_state": None,
    }


def _grid_spec_bounds_from_table(table: pd.DataFrame) -> np.ndarray:
    eta_vals = np.sort(table["eta_t"].unique().astype(float))
    alpha_vals = np.sort(table["input_alpha_dndm"].unique().astype(float))
    logmc_vals = np.sort(table["input_log10_m_c_msun"].unique().astype(float))
    return np.array(
        [
            [float(eta_vals[0]), float(eta_vals[-1])],
            [float(alpha_vals[0]), float(alpha_vals[-1])],
            [float(logmc_vals[0]), float(logmc_vals[-1])],
        ],
        dtype=float,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root-name", default="profile_map_and_exact_mcmc_bk_shared_schechter_two_component_logistic_global")
    parser.add_argument("--in-situ-radial-model", default="logpoly3", choices=["logpoly3", "step5"])
    parser.add_argument("--accreted-radial-model", default="logpoly3", choices=["logpoly3", "step5"])
    parser.add_argument("--mcmc-chains", type=int, default=6)
    parser.add_argument("--mcmc-steps", type=int, default=900)
    parser.add_argument("--mcmc-burn", type=int, default=300)
    parser.add_argument("--mcmc-thin", type=int, default=2)
    parser.add_argument("--mcmc-adapt-until", type=int, default=240)
    parser.add_argument("--mcmc-adapt-every", type=int, default=20)
    parser.add_argument("--mcmc-seed", type=int, default=20260529)
    parser.add_argument("--anchor-k", type=int, default=18)
    parser.add_argument("--n-detectability-iterations", type=int, default=12)
    parser.add_argument("--detectability-relaxation", type=float, default=0.7)
    args = parser.parse_args()

    output_root = PROJECT_ROOT / "variants" / args.output_root_name
    tables_dir = output_root / "outputs" / "tables"
    figures_dir = output_root / "outputs" / "figures"
    worker_dir = output_root / "outputs" / "parallel_exact_mcmc_workers"
    figures_dir.mkdir(parents=True, exist_ok=True)
    worker_dir.mkdir(parents=True, exist_ok=True)

    coarse_table = pd.read_csv(tables_dir / "coarse_grid_results.csv")
    refined_table = pd.read_csv(tables_dir / "refined_grid_results.csv")
    coarse_good = coarse_table.loc[np.isfinite(coarse_table["log_likelihood"])].copy()
    refined_good = refined_table.loc[np.isfinite(refined_table["log_likelihood"])].copy()
    if coarse_good.empty or refined_good.empty:
        raise RuntimeError("Saved coarse/refined tables do not contain successful likelihood evaluations.")

    coarse_entries = [_entry_from_row(row) for row in coarse_good.to_dict(orient="records")]
    refined_entries = [_entry_from_row(row) for row in refined_good.to_dict(orient="records")]
    coarse_best_entry = max(coarse_entries, key=lambda entry: float(entry["log_posterior"]))
    refined_best_entry = max(refined_entries, key=lambda entry: float(entry["log_posterior"]))
    refined_bounds = _grid_spec_bounds_from_table(refined_good)

    catalog = _load_catalog()
    prepared_catalog = fit_catalog_models(catalog, output_root)["catalog"]
    spec = SharedImfTwoComponentSpec(
        imf_family="schechter",
        in_situ_radial_model=str(args.in_situ_radial_model),
        accreted_radial_model=str(args.accreted_radial_model),
    )

    fixed_anchor_library = _build_anchor_library(
        coarse_entries,
        refined_entries,
        k=max(int(args.anchor_k), int(args.mcmc_chains) * 2),
    )
    current_states = _select_diverse_entries(refined_entries, n_select=int(args.mcmc_chains), bounds=refined_bounds)
    if len(current_states) == 0:
        raise RuntimeError("No successful refined-grid evaluations available for exact MCMC starts.")
    while len(current_states) < int(args.mcmc_chains):
        current_states.append(current_states[-1])

    widths = refined_bounds[:, 1] - refined_bounds[:, 0]
    for stale in worker_dir.glob("chain_*_config.pkl"):
        stale.unlink()
    for stale in worker_dir.glob("chain_*_result.pkl"):
        stale.unlink()
    for stale in worker_dir.glob("chain_*.log"):
        stale.unlink()

    procs = []
    worker_script = PROJECT_ROOT / "scripts" / "run_profile_map_and_exact_mcmc_bk_shared_schechter_two_component.py"
    for chain_id in range(int(args.mcmc_chains)):
        config = {
            "chain_id": chain_id,
            "n_steps": int(args.mcmc_steps),
            "adapt_until": int(args.mcmc_adapt_until),
            "adapt_every": int(args.mcmc_adapt_every),
            "seed": int(args.mcmc_seed) + chain_id,
            "prepared_catalog": prepared_catalog,
            "spec": spec,
            "project_root": output_root,
            "bounds": refined_bounds,
            "widths": widths,
            "initial_entry": _lightweight_entry(current_states[chain_id]),
            "fixed_anchor_library": [_lightweight_entry(entry) for entry in fixed_anchor_library],
            "n_detectability_iterations": int(args.n_detectability_iterations),
            "relaxation": float(args.detectability_relaxation),
        }
        config_path = worker_dir / f"chain_{chain_id}_config.pkl"
        result_path = worker_dir / f"chain_{chain_id}_result.pkl"
        log_path = worker_dir / f"chain_{chain_id}.log"
        with config_path.open("wb") as handle:
            pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)
        log_handle = log_path.open("w")
        proc = subprocess.Popen(
            [
                sys.executable,
                str(worker_script),
                "--chain-worker-config",
                str(config_path),
                "--chain-worker-output",
                str(result_path),
            ],
            cwd=str(PROJECT_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        procs.append((chain_id, proc, log_handle, result_path, log_path))

    chain_results = []
    for chain_id, proc, log_handle, result_path, log_path in procs:
        return_code = proc.wait()
        log_handle.close()
        if return_code != 0:
            log_text = log_path.read_text()
            raise RuntimeError(f"Chain worker {chain_id} failed with code {return_code}\n{log_text}")
        with result_path.open("rb") as handle:
            chain_results.append(pickle.load(handle))
        best_row = chain_results[-1]["best_row"]
        print(
            f"[parallel exact mcmc] chain={chain_id} done accept={float(chain_results[-1]['acceptance']):.3f} "
            f"best logL={float(best_row['log_likelihood']):.3f} eta_t={float(best_row['eta_t']):.3f} "
            f"alpha={float(best_row['input_alpha_dndm']):.3f} logMc={float(best_row['input_log10_m_c_msun']):.3f}"
        )

    chain_results.sort(key=lambda item: int(item["chain_id"]))
    records = [row for result in chain_results for row in result["records"]]
    chain_table = pd.DataFrame(records).sort_values(["chain", "step"]).reset_index(drop=True)
    chain_table.to_csv(tables_dir / "exact_parallel_mcmc_chain.csv", index=False)

    posterior_parts = []
    for _, frame in chain_table.loc[chain_table["step"] >= int(args.mcmc_burn)].groupby("chain"):
        posterior_parts.append(frame.iloc[:: int(args.mcmc_thin)])
    posterior_table = pd.concat(posterior_parts, ignore_index=True)
    posterior_table.to_csv(tables_dir / "exact_parallel_mcmc_posterior_samples.csv", index=False)

    summary_candidate_columns = [
        "eta_t",
        "input_alpha_dndm",
        "input_log10_m_c_msun",
        "final_total_initial_count_above_log10_4",
        "final_total_initial_stellar_mass_above_log10_4_msun",
        "final_total_initial_count_above_log10_4_in_situ",
        "final_total_initial_count_above_log10_4_accreted",
        "mean_detectability_above_log10_4",
        "log_likelihood",
    ]
    summary_rows = []
    for column in summary_candidate_columns:
        values = np.asarray(posterior_table[column], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        q16, q50, q84 = np.quantile(finite, [0.16, 0.50, 0.84])
        summary_rows.append(
            {
                "parameter": column,
                "q16": float(q16),
                "q50": float(q50),
                "q84": float(q84),
                "minus": float(q50 - q16),
                "plus": float(q84 - q50),
            }
        )
    posterior_summary = pd.DataFrame(summary_rows)
    posterior_summary.to_csv(tables_dir / "exact_parallel_posterior_summary.csv", index=False)

    rhat = {}
    for column in ["eta_t", "input_alpha_dndm", "input_log10_m_c_msun"]:
        pivot = posterior_table.pivot_table(index="step", columns="chain", values=column, aggfunc="last").dropna().to_numpy().T
        if pivot.size == 0:
            continue
        rhat[column] = _compute_rhat(pivot)

    acceptance_by_chain = {str(result["chain_id"]): float(result["acceptance"]) for result in chain_results}
    best_posterior_row = posterior_table.sort_values("log_likelihood", ascending=False).iloc[0].to_dict()
    _corner_plot(posterior_table, refined_best_entry["row"], figures_dir / "exact_parallel_profiled_posterior_corner.png")
    _trace_plot(chain_table, figures_dir / "exact_parallel_profiled_posterior_traces.png", burn_in=int(args.mcmc_burn))

    best_theta = np.array(
        [
            float(best_posterior_row["eta_t"]),
            float(best_posterior_row["input_alpha_dndm"]),
            float(best_posterior_row["input_log10_m_c_msun"]),
        ],
        dtype=float,
    )
    best_anchor = _select_anchor_start_state(theta=best_theta, anchors=fixed_anchor_library, bounds=refined_bounds)
    best_entry = _evaluate_theta_multistart(
        prepared_catalog=prepared_catalog,
        spec=spec,
        theta=best_theta,
        stage="exact_parallel_mcmc_best",
        project_root=output_root,
        anchor_start_state=best_anchor,
        n_detectability_iterations=int(args.n_detectability_iterations),
        relaxation=float(args.detectability_relaxation),
    )
    _save_best_payload(best_entry, tables_dir, prefix="exact_parallel_mcmc")

    summary = {
        "source_output_root_name": args.output_root_name,
        "output_root_name": args.output_root_name,
        "surface_model": SURFACE_MODEL,
        "model_spec": {
            "model_class": "bk_shared_schechter_two_component",
            "imf_family": spec.imf_family,
            "in_situ_radial_model": spec.in_situ_radial_model,
            "accreted_radial_model": spec.accreted_radial_model,
        },
        "sampler": "exact_profiled_random_walk_metropolis_subprocess_parallel_resumed",
        "n_detectability_iterations": int(args.n_detectability_iterations),
        "n_chains": int(args.mcmc_chains),
        "n_steps": int(args.mcmc_steps),
        "burn_in": int(args.mcmc_burn),
        "thin": int(args.mcmc_thin),
        "acceptance_by_chain": acceptance_by_chain,
        "rhat": rhat,
        "best_posterior_sample": json.loads(pd.Series(best_posterior_row).to_json()),
        "posterior_summary": posterior_summary.to_dict(orient="records"),
        "worker_cache_sizes": {str(result['chain_id']): int(result['cache_size']) for result in chain_results},
        "anchor_count": int(len(fixed_anchor_library)),
        "refined_bounds": refined_bounds.tolist(),
        "coarse_best": json.loads(pd.Series(coarse_best_entry['row']).to_json()),
        "refined_best": json.loads(pd.Series(refined_best_entry['row']).to_json()),
    }
    (tables_dir / "exact_parallel_mcmc_summary.json").write_text(json.dumps(summary, indent=2))

    print(figures_dir / "exact_parallel_profiled_posterior_corner.png")
    print(figures_dir / "exact_parallel_profiled_posterior_traces.png")
    print(tables_dir / "exact_parallel_posterior_summary.csv")
    print(tables_dir / "exact_parallel_mcmc_summary.json")


if __name__ == "__main__":
    main()
