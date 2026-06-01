from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import pickle
import sys

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from globular_clusters_imf.joint_model import JointModelSpec
from globular_clusters_imf.model import fit_catalog_models
from globular_clusters_imf.paper_assets import (
    plot_best_single_component_summary_for_paper,
    plot_single_component_profiles_for_paper,
    plot_single_component_radial_profile_for_paper,
)
from run_parallel_exact_mcmc_from_existing_refined_grid import (
    _build_exact_anchor_entries,
    _compute_rhat,
    _corner_plot,
    _lightweight_entry,
    _load_catalog,
    _save_best_payload,
    _select_anchor_start_state,
    _trace_plot,
)
from run_profile_map_and_exact_mcmc_schechter_powerlaw_a import _evaluate_theta_multistart
from build_inferred_imf_uncertainty_band import main as build_imf_band_main


def _write_exact_summary(
    *,
    posterior_table: pd.DataFrame,
    chain_table: pd.DataFrame,
    chain_results: list[dict[str, object]],
    refined_bounds: np.ndarray,
    tables_dir: Path,
    best_result_entry: dict[str, object],
    source_output_root_name: str,
) -> None:
    summary_candidate_columns = [
        "eta_t",
        "input_alpha_dndm",
        "input_log10_m_c_msun",
        "gamma_linear_a",
        "final_total_initial_count_above_log10_4",
        "final_total_initial_stellar_mass_above_log10_4_msun",
        "mean_detectability_above_log10_4",
        "log_likelihood",
    ]
    summary_rows = []
    for column in summary_candidate_columns:
        if column not in posterior_table.columns:
            continue
        values = np.asarray(posterior_table[column], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        q16, q50, q84 = np.quantile(finite, [0.16, 0.50, 0.84])
        summary_rows.append({
            "parameter": column,
            "q16": float(q16),
            "q50": float(q50),
            "q84": float(q84),
            "minus": float(q50 - q16),
            "plus": float(q84 - q50),
        })
    posterior_summary = pd.DataFrame(summary_rows)
    posterior_summary.to_csv(tables_dir / "exact_parallel_posterior_summary.csv", index=False)

    rhat = {}
    for column in ["eta_t", "input_alpha_dndm", "input_log10_m_c_msun", "gamma_linear_a"]:
        if column not in posterior_table.columns:
            continue
        pivot = (
            posterior_table.pivot_table(index="step", columns="chain", values=column, aggfunc="last")
            .dropna()
            .to_numpy()
            .T
        )
        if pivot.size == 0:
            continue
        rhat[column] = _compute_rhat(pivot)

    acceptance_by_chain = {str(result["chain_id"]): float(result["acceptance"]) for result in chain_results}
    best_posterior_row = posterior_table.sort_values("log_likelihood", ascending=False).iloc[0].to_dict()
    summary = {
        "source_output_root_name": source_output_root_name,
        "output_root_name": source_output_root_name,
        "sampler": "exact_profiled_random_walk_metropolis_subprocess_parallel",
        "n_chains": int(chain_table["chain"].nunique()),
        "n_steps": int(chain_table["step"].max() + 1),
        "burn_in": 300,
        "thin": 2,
        "acceptance_by_chain": acceptance_by_chain,
        "rhat": rhat,
        "best_posterior_sample": json.loads(pd.Series(best_posterior_row).to_json()),
        "posterior_summary": posterior_summary.to_dict(orient="records"),
        "worker_cache_sizes": {str(result["chain_id"]): int(result["cache_size"]) for result in chain_results},
        "anchor_count": 18,
        "refined_bounds": refined_bounds.tolist(),
        "exact_best_result_summary": asdict(best_result_entry["result"]["final_payload"]["summary"]),
    }
    (tables_dir / "exact_parallel_mcmc_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root-name",
        default="profile_map_and_exact_mcmc_schechter_logpoly3_logistic_global",
    )
    parser.add_argument("--n-raw-sample-reprofiles", type=int, default=60)
    args = parser.parse_args()

    output_root = PROJECT_ROOT / "variants" / args.output_root_name
    tables_dir = output_root / "outputs" / "tables"
    figures_dir = output_root / "outputs" / "figures"
    worker_dir = output_root / "outputs" / "parallel_exact_mcmc_workers"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    with (tables_dir / "exact_parallel_mcmc_best_result.pkl").open("rb") as handle:
        best_result = pickle.load(handle)

    chain_table = pd.read_csv(tables_dir / "exact_parallel_mcmc_chain.csv")
    posterior_table = pd.read_csv(tables_dir / "exact_parallel_mcmc_posterior_samples.csv")
    refined_table = pd.read_csv(tables_dir / "refined_grid_results.csv")
    refined_best_row = refined_table.loc[refined_table["log_likelihood"].idxmax()].to_dict()
    refined_bounds = np.array([
        [float(refined_table["eta_t"].min()), float(refined_table["eta_t"].max())],
        [float(refined_table["input_alpha_dndm"].min()), float(refined_table["input_alpha_dndm"].max())],
        [float(refined_table["input_log10_m_c_msun"].min()), float(refined_table["input_log10_m_c_msun"].max())],
    ], dtype=float)

    chain_results = []
    for path in sorted(worker_dir.glob("chain_*_result.pkl")):
        with path.open("rb") as handle:
            chain_results.append(pickle.load(handle))
    chain_results.sort(key=lambda item: int(item["chain_id"]))

    _corner_plot(posterior_table, refined_best_row, figures_dir / "exact_parallel_profiled_posterior_corner.png")
    _trace_plot(chain_table, figures_dir / "exact_parallel_profiled_posterior_traces.png", burn_in=300)

    prepared_catalog = _load_catalog()
    spec = JointModelSpec(imf_family="schechter", radial_model="logpoly3")
    exact_anchor_entries = _build_exact_anchor_entries(
        prepared_catalog=prepared_catalog,
        spec=spec,
        project_root=output_root,
        refined_table=refined_table,
        refined_bounds=refined_bounds,
        anchor_k=18,
        anchor_pool=36,
    )
    lightweight_anchors = [_lightweight_entry(entry) for entry in exact_anchor_entries]

    outer_unique = posterior_table.loc[:, ["eta_t", "input_alpha_dndm", "input_log10_m_c_msun"]].drop_duplicates()
    if len(outer_unique) > args.n_raw_sample_reprofiles:
        idx = np.linspace(0, len(outer_unique) - 1, args.n_raw_sample_reprofiles, dtype=int)
        outer_unique = outer_unique.iloc[idx].reset_index(drop=True)

    raw_param_rows = []
    for idx, row in outer_unique.iterrows():
        theta = np.array([row["eta_t"], row["input_alpha_dndm"], row["input_log10_m_c_msun"]], dtype=float)
        anchor = _select_anchor_start_state(theta=theta, anchors=lightweight_anchors, bounds=refined_bounds)
        entry = _evaluate_theta_multistart(
            prepared_catalog=prepared_catalog,
            spec=spec,
            theta=theta,
            stage=f"plot_reprofile_{idx}",
            project_root=output_root,
            anchor_start_state=anchor,
        )
        raw = np.asarray(entry["result"]["final_payload"]["raw_parameters"], dtype=float)
        raw_param_rows.append({
            "eta_t": float(theta[0]),
            "input_alpha_dndm": float(theta[1]),
            "input_log10_m_c_msun": float(theta[2]),
            "raw_param_0": float(raw[0]),
            "raw_param_1": float(raw[1]),
            "raw_param_2": float(raw[2]),
            "raw_param_3": float(raw[3]),
            "raw_param_4": float(raw[4]),
            "log_likelihood": float(entry["row"]["log_likelihood"]),
        })
        if (idx + 1) % 10 == 0 or (idx + 1) == len(outer_unique):
            print(f"reprofiled {idx + 1}/{len(outer_unique)} posterior support points")

    raw_param_table = pd.DataFrame(raw_param_rows)
    raw_param_table.to_csv(tables_dir / "exact_parallel_raw_parameter_samples.csv", index=False)
    raw_samples = raw_param_table.loc[:, [f"raw_param_{i}" for i in range(5)]].to_numpy(dtype=float)
    uncertainty_payload = {"raw_samples": raw_samples}

    fit_catalog = _load_catalog()
    baseline_payload = best_result["baseline_payload"]
    baseline_joint_results = {
        "summary_table": pd.DataFrame([{
            "imf_family": baseline_payload["spec"].imf_family,
            "radial_model": baseline_payload["spec"].radial_model,
        }]),
        "imf_grid_table": pd.DataFrame(baseline_payload["imf_grid_rows"]),
        "radial_grid_table": pd.DataFrame(baseline_payload["radial_grid_rows"]),
        "best_payload": baseline_payload,
        "context": best_result["base_context"],
    }

    for ext in ["png", "pdf"]:
        plot_best_single_component_summary_for_paper(
            catalog=fit_catalog,
            context=best_result["final_context"],
            best_payload=best_result["final_payload"],
            uncertainty_payload=uncertainty_payload,
            output_path=figures_dir / f"best_single_component_summary.{ext}",
            n_projection_samples=min(len(raw_samples), 250),
        )
        plot_single_component_profiles_for_paper(
            baseline_joint_results=baseline_joint_results,
            detectability_result=best_result,
            uncertainty_payload=uncertainty_payload,
            output_path=figures_dir / f"single_component_profiles.{ext}",
        )
        plot_single_component_radial_profile_for_paper(
            baseline_joint_results=baseline_joint_results,
            detectability_result=best_result,
            uncertainty_payload=uncertainty_payload,
            output_path=figures_dir / f"single_component_radial_profile.{ext}",
        )

    # Reuse the existing IMF-band builder on the exact posterior table.
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "build_inferred_imf_uncertainty_band.py",
            "--posterior-samples", str(tables_dir / "exact_parallel_mcmc_posterior_samples.csv"),
            "--output", str(figures_dir / "inferred_imf_truthful_uncertainty_band.png"),
        ]
        build_imf_band_main()
    finally:
        sys.argv = old_argv

    best_theta = np.array([
        float(posterior_table.sort_values("log_likelihood", ascending=False).iloc[0]["eta_t"]),
        float(posterior_table.sort_values("log_likelihood", ascending=False).iloc[0]["input_alpha_dndm"]),
        float(posterior_table.sort_values("log_likelihood", ascending=False).iloc[0]["input_log10_m_c_msun"]),
    ], dtype=float)
    best_anchor = _select_anchor_start_state(theta=best_theta, anchors=lightweight_anchors, bounds=refined_bounds)
    best_result_entry = _evaluate_theta_multistart(
        prepared_catalog=prepared_catalog,
        spec=spec,
        theta=best_theta,
        stage="exact_parallel_mcmc_best_refresh",
        project_root=output_root,
        anchor_start_state=best_anchor,
    )
    _save_best_payload(best_result_entry, tables_dir, prefix="exact_parallel_mcmc")
    _write_exact_summary(
        posterior_table=posterior_table,
        chain_table=chain_table,
        chain_results=chain_results,
        refined_bounds=refined_bounds,
        tables_dir=tables_dir,
        best_result_entry=best_result_entry,
        source_output_root_name=args.output_root_name,
    )

    manifest = {
        "corner": str(figures_dir / "exact_parallel_profiled_posterior_corner.png"),
        "traces": str(figures_dir / "exact_parallel_profiled_posterior_traces.png"),
        "best_summary_png": str(figures_dir / "best_single_component_summary.png"),
        "profiles_png": str(figures_dir / "single_component_profiles.png"),
        "radial_profile_png": str(figures_dir / "single_component_radial_profile.png"),
        "imf_band_png": str(figures_dir / "inferred_imf_truthful_uncertainty_band.png"),
        "raw_parameter_samples": str(tables_dir / "exact_parallel_raw_parameter_samples.csv"),
    }
    (tables_dir / "usual_plots_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
