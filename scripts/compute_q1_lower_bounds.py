"""Lower bounds on the birth population with detectability set to 1 (Q = 1).

For each disruption law at its published normalisation (eta_t = 1), fit the Schechter IMF
and logpoly3 radial profile correcting for survival only. Since the true detectability is
<= 1, the resulting N0 is a lower bound conditional on the survival law and population
model (confirmed on simulated catalogues: scripts/injection_recovery_detectability.py).

Writes one row per law: best-fit alpha, log10 Mc, N0 above 1e4 and 10^5.5 Msun, and the
number and stellar mass of clusters born above 1e4 Msun that were destroyed.

Usage
-----
    python scripts/compute_q1_lower_bounds.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
OUTPUT = PROJECT_ROOT / "variants" / "baseline_2026-10_q1_lower_bounds.csv"
LAWS = [
    ("baumgardt", None),
    ("gg23", "gg23_no_bh"),
    ("gg23", "gg23_bh"),
    ("gg23", "gg23_bh_feh_gradient"),
    ("gg23", "gg23_bh_past_tidal"),
    ("gg23", "gg23_bh_feh_gradient_past_tidal"),
]


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    warnings.simplefilter("ignore")
    import run_profile_map_and_exact_mcmc_schechter_powerlaw_a as runner
    from globular_clusters_imf.joint_model import JointLikelihoodContext, JointModelSpec, fit_single_joint_model
    from globular_clusters_imf.model import fit_catalog_models

    catalog = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv")
    with tempfile.TemporaryDirectory() as scratch:
        prepared = fit_catalog_models(catalog, Path(scratch))["catalog"]
    spec = JointModelSpec(imf_family="schechter", radial_model="logpoly3")

    rows = []
    for backend, law in LAWS:
        working, survival, metadata = runner._catalog_and_survival_grid_for_theta(
            prepared_catalog=prepared, eta_t=1.0, survivability_backend=backend, gg23_model_name=law
        )
        context = JointLikelihoodContext.from_catalog_and_survival_grid(
            working, runner._survival_grid_override_from_smooth_survival(survival)
        )
        fit = fit_single_joint_model(context, spec)  # selection = survival only, i.e. Q = 1
        model = fit["model"]
        lm, la = context.log_mass_grid, context.log_a_grid
        intensity = (
            model["total_initial_count"]
            * np.asarray(model["imf_density_grid"])[:, None]
            * np.asarray(model["radial_density_grid"])[None, :]
        )
        destroyed = intensity * (1.0 - context.survival_probability_grid)

        def integral(grid: np.ndarray, mass_mask: np.ndarray) -> float:
            return float(np.trapezoid(np.trapezoid(np.where(mass_mask[:, None], grid, 0.0), la, axis=1), lm))

        low = (lm >= 4.0) & (lm < 5.0)
        above4 = lm >= 4.0
        rows.append(
            {
                "law": law or "baumgardt",
                "eta_t": 1.0,
                "alpha_dndm": model["imf_parameters"]["alpha_dndm"],
                "log10_m_c_msun": model["imf_parameters"]["log10_m_c_msun"],
                "n0_above_log10_4_lower_bound": integral(intensity, above4),
                "n0_above_log10_5p5_lower_bound": integral(intensity, lm >= 5.5),
                "born_log10_4_to_5": integral(intensity, low),
                "survivors_log10_4_to_5": int(((working.log_initial_mass_msun >= 4) & (working.log_initial_mass_msun < 5)).sum()),
                "destroyed_log10_4_to_5": integral(destroyed, low),
                "destroyed_mass_log10_4_to_5_msun": integral(destroyed * np.power(10.0, lm)[:, None], low),
                "destroyed_mass_above_log10_4_msun": integral(destroyed * np.power(10.0, lm)[:, None], above4),
                "a_range_kpc": f"{10**la[0]:.2f}-{10**la[-1]:.0f}",
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.round(3).to_string(index=False))
    print(OUTPUT)


if __name__ == "__main__":
    main()
