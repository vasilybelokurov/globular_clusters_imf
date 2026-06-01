from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd

from globular_clusters_imf.model import fit_catalog_models
from globular_clusters_imf.paper_assets import plot_best_single_component_summary_for_paper

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz



def _load_catalog() -> pd.DataFrame:
    catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    prepared = fit_catalog_models(
        catalog,
        PROJECT_ROOT / "variants" / "tmp_best_single_component_summary_exact_preview_prepare",
    )["catalog"]
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a best_single_component_summary-style preview figure using the "
            "exact parallel MCMC best-fit payload and posterior parameter samples."
        )
    )
    parser.add_argument(
        "--best-result",
        type=Path,
        default=PROJECT_ROOT
        / "variants"
        / "profile_map_and_exact_mcmc_schechter_powerlaw_a_logistic_parallel_long"
        / "outputs"
        / "tables"
        / "exact_parallel_mcmc_best_result.pkl",
    )
    parser.add_argument(
        "--posterior-samples",
        type=Path,
        default=PROJECT_ROOT
        / "variants"
        / "profile_map_and_exact_mcmc_schechter_powerlaw_a_logistic_parallel_long"
        / "outputs"
        / "tables"
        / "exact_parallel_mcmc_posterior_samples.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "variants"
        / "profile_map_and_exact_mcmc_schechter_powerlaw_a_logistic_parallel_long"
        / "outputs"
        / "figures"
        / "best_single_component_summary_exact_parallel_preview.png",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=250,
        help="Maximum number of posterior parameter draws used for projection bands.",
    )
    args = parser.parse_args()

    catalog = _load_catalog()
    with args.best_result.open("rb") as handle:
        best_result = pickle.load(handle)
    posterior_samples = pd.read_csv(args.posterior_samples)
    raw_samples = (
        posterior_samples.loc[:, ["input_alpha_dndm", "input_log10_m_c_msun", "beta_log10_a"]]
        .dropna()
        .to_numpy(dtype=float)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_best_single_component_summary_for_paper(
        catalog=catalog,
        context=best_result["final_context"],
        best_payload=best_result["final_payload"],
        uncertainty_payload={"raw_samples": raw_samples},
        output_path=args.output,
        n_projection_samples=args.n_samples,
    )


if __name__ == "__main__":
    main()
