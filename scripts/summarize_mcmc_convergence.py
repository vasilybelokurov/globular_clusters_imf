"""Convergence gate for any cached profiled-MCMC run.

Reads a chain table (one row per chain step) from variants/<variant>/outputs/tables/,
and writes <chain stem>_convergence.json next to it with split-R-hat, effective sample
size, the fraction of post-burn samples within 2 per cent of each prior edge, and the
fraction of rows whose detectability iteration reached its tolerance.

Usage
-----
    python scripts/summarize_mcmc_convergence.py --variant baseline_2026-10_bk_two_component \
        --chain-file exact_mcmc_chain.csv --burn 400 \
        --prior-eta 0.2,4 --prior-alpha=-2.5,-0.3 --prior-logmc 5.5,7.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from globular_clusters_imf.mcmc_diagnostics import compute_split_rhat, effective_sample_size

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = {
    "eta_t": "prior_eta",
    "input_alpha_dndm": "prior_alpha",
    "input_log10_m_c_msun": "prior_logmc",
}
DERIVED = ["final_total_initial_count_above_log10_4"]


def _parse_bounds(text: str) -> tuple[float, float] | None:
    if not text:
        return None
    lo, hi = (float(value) for value in text.split(","))
    return lo, hi


def summarize(chain: pd.DataFrame, burn: int, bounds: dict[str, tuple[float, float] | None],
              min_ess: float, max_split_rhat: float) -> dict[str, object]:
    post = chain.loc[chain["step"] >= burn]
    convergence: dict[str, dict[str, float]] = {}
    for column in [*PARAMETERS, *DERIVED]:
        if column not in post.columns:
            continue
        stacked = np.array([frame[column].to_numpy(dtype=float) for _, frame in post.groupby("chain")])
        if stacked.size == 0 or np.ptp(stacked) == 0.0:
            continue
        convergence[column] = {"split_rhat": compute_split_rhat(stacked), "ess": effective_sample_size(stacked)}
    edges: dict[str, dict[str, float]] = {}
    for column, prior in bounds.items():
        if prior is None or prior[1] == prior[0] or column not in post.columns:
            continue
        values = post[column].to_numpy(dtype=float)
        margin = 0.02 * (prior[1] - prior[0])
        edges[column] = {
            "below_lo_plus_2pc": float(np.mean(values < prior[0] + margin)),
            "above_hi_minus_2pc": float(np.mean(values > prior[1] - margin)),
        }
    passed = all(
        np.isfinite(item["split_rhat"]) and item["split_rhat"] <= max_split_rhat
        and np.isfinite(item["ess"]) and item["ess"] >= min_ess
        for item in convergence.values()
    )
    converged_fraction = (
        float(post["detectability_converged"].astype(bool).mean()) if "detectability_converged" in post else float("nan")
    )
    return {
        "n_chains": int(post["chain"].nunique()),
        "n_post_burn_rows": int(len(post)),
        "burn": int(burn),
        "convergence": convergence,
        "gate": {"passed": bool(passed), "min_ess": min_ess, "max_split_rhat": max_split_rhat},
        "prior_edge_fraction": edges,
        "posterior_reaches_prior_box": bool(any(max(item.values()) >= 0.01 for item in edges.values())),
        "detectability_converged_fraction": converged_fraction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--chain-file", default="exact_parallel_mcmc_chain.csv")
    parser.add_argument("--burn", type=int, required=True)
    parser.add_argument("--prior-eta", default="")
    parser.add_argument("--prior-alpha", default="")
    parser.add_argument("--prior-logmc", default="")
    parser.add_argument("--min-ess", type=float, default=400.0)
    parser.add_argument("--max-split-rhat", type=float, default=1.02)
    args = parser.parse_args()

    tables = PROJECT_ROOT / "variants" / args.variant / "outputs" / "tables"
    chain = pd.read_csv(tables / args.chain_file)
    bounds = {column: _parse_bounds(getattr(args, attribute)) for column, attribute in PARAMETERS.items()}
    result = summarize(chain, args.burn, bounds, args.min_ess, args.max_split_rhat)
    output = tables / f"{Path(args.chain_file).stem}_convergence.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(output)


if __name__ == "__main__":
    main()
