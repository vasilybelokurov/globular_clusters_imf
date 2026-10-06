"""Injection-recovery test of the iterative detectability correction.

Question: in the most favourable case -- catalogues simulated from the model itself, with
known birth population, survival and completeness -- does the alternating detectability
iteration recover the completeness and N0?

Truth is taken from a fitted single-component result (default: the Baumgardt baseline best
fit). Each simulated catalogue is generated as follows:

* births: Poisson(N0) draws from phi(log M) A(log a) on the model grid;
* survival: Bernoulli with the same S(log M, a) that the fitter is given;
* present mass: the fitted present-mass proxy mean plus Gaussian scatter of its width;
* position: isotropic on a shell of radius a around the Galactic centre, Sun at R0;
* detection: the binned logistic completeness (bin centres, same standardisation as the
  fitter's form) with the true parameters, optionally with the intercept shifted to
  lower the overall completeness.

The fitter is then run with the true (alpha, log10 Mc) fixed, so only the radial profile,
completeness and N0 are inferred -- exactly the inner problem solved at each MCMC step.

Usage
-----
    python scripts/injection_recovery_detectability.py --n-realisations 8 --workers 7
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate, special

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRUTH = (
    PROJECT_ROOT / "variants" / "baseline_2026-10_baumgardt_logpoly3" / "outputs" / "tables"
    / "exact_parallel_mcmc_best_result.pkl"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "variants" / "baseline_2026-10_injection_recovery"


def load_truth(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        result = pickle.load(handle)
    model = result["final_payload"]["model"]
    base = result["base_context"]
    oc = result["observable_context"]
    return {
        "log_mass_grid": np.asarray(base.log_mass_grid),
        "log_a_grid": np.asarray(base.log_a_grid),
        "survival": np.asarray(base.survival_probability_grid),
        "imf_density": np.asarray(model["imf_density_grid"]),
        "radial_density": np.asarray(model["radial_density_grid"]),
        "n0_total": float(model["total_initial_count"]),
        "imf_params": np.array(
            [model["imf_parameters"]["alpha_dndm"], model["imf_parameters"]["log10_m_c_msun"]], dtype=float
        ),
        "completeness_raw": np.asarray(result["final_completeness_raw_parameters"], dtype=float),
        "observable_context": oc,
    }


def binned_completeness(raw, oc, log_mnow, log_d, abs_b, abs_l):
    """True detection probability: the fitter's logistic form evaluated at bin centres."""

    def centre(values, edges, centres):
        index = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(centres) - 1)
        return centres[index]

    z_m = (centre(log_mnow, oc.log_present_mass_edges, oc.log_present_mass_centers) - oc.log_present_mass_feature_mean) / oc.log_present_mass_feature_std
    z_d = (centre(log_d, np.log10(oc.distance_edges_kpc), oc.log_distance_centers) - oc.log_distance_feature_mean) / oc.log_distance_feature_std
    z_b = (centre(abs_b, oc.abs_latitude_edges_deg, oc.abs_latitude_centers_deg) - oc.abs_latitude_feature_mean) / oc.abs_latitude_feature_std
    z_l = (centre(abs_l, oc.abs_longitude_edges_deg, oc.abs_longitude_centers_deg) - oc.abs_longitude_feature_mean) / oc.abs_longitude_feature_std
    logits = raw[0] + np.exp(raw[1]) * z_m - np.exp(raw[2]) * z_d + np.exp(raw[3]) * z_b + np.exp(raw[4]) * z_l
    return special.expit(logits)


def simulate(truth: dict[str, object], n0: float, completeness_raw: np.ndarray, rng: np.random.Generator) -> tuple[pd.DataFrame, dict[str, float]]:
    lm, la = truth["log_mass_grid"], truth["log_a_grid"]
    dm = np.gradient(lm)
    da = np.gradient(la)
    cell = truth["imf_density"][:, None] * truth["radial_density"][None, :] * dm[:, None] * da[None, :]
    probabilities = (cell / cell.sum()).ravel()
    n_birth = rng.poisson(n0)
    index = rng.choice(probabilities.size, size=n_birth, p=probabilities)
    im, ia = np.unravel_index(index, cell.shape)
    log_m = lm[im] + (rng.uniform(size=n_birth) - 0.5) * dm[im]
    log_a = la[ia] + (rng.uniform(size=n_birth) - 0.5) * da[ia]
    log_m = np.clip(log_m, lm[0], lm[-1])
    log_a = np.clip(log_a, la[0], la[-1])

    survival = interpolate.RegularGridInterpolator((lm, la), truth["survival"], bounds_error=False, fill_value=None)
    survives = rng.uniform(size=n_birth) < np.clip(survival(np.column_stack([log_m, log_a])), 0.0, 1.0)

    oc = truth["observable_context"]
    proxy_mean = interpolate.RegularGridInterpolator((lm, la), oc.log_present_mass_mean_grid, bounds_error=False, fill_value=None)
    log_mnow = proxy_mean(np.column_stack([log_m, log_a])) + rng.normal(scale=oc.present_mass_proxy.residual_sigma_dex, size=n_birth)
    # The model folds probability outside the fixed present-mass bin range into the edge
    # bins; clamp the simulated masses the same way so the catalogue stays inside it.
    edges = oc.log_present_mass_edges
    log_mnow = np.clip(log_mnow, edges[0] + 1.0e-6, edges[-1] - 1.0e-6)

    r = np.power(10.0, log_a)
    u = rng.normal(size=(n_birth, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    x, y, z = (r[:, None] * u).T
    r0 = float(oc.sun_galactocentric_radius_kpc)
    distance = np.sqrt((x - r0) ** 2 + y**2 + z**2)
    b_deg = np.degrees(np.arcsin(z / distance))
    l_deg = np.degrees(np.arctan2(y, r0 - x))

    detect_prob = binned_completeness(completeness_raw, oc, log_mnow, np.log10(distance), np.abs(b_deg), np.abs(l_deg))
    detected = survives & (rng.uniform(size=n_birth) < detect_prob)

    catalog = pd.DataFrame(
        {
            "cluster_name": [f"sim{i}" for i in range(int(detected.sum()))],
            "log_initial_mass_msun": log_m[detected],
            "initial_mass_msun": np.power(10.0, log_m[detected]),
            "semi_major_axis_kpc": r[detected],
            "log_survival_mass_cut_msun": log_m[detected] - 1.0,  # unused with a survival override
            "present_mass_msun": np.power(10.0, log_mnow[detected]),
            "r_sun_kpc": distance[detected],
            "galactic_b_deg": b_deg[detected],
            "galactic_l_deg": l_deg[detected],
        }
    )
    above = log_m >= 4.0
    truth_row = {
        "n_birth_above_log10_5p5": int(np.sum(log_m >= 5.5)),
        "n_birth": int(n_birth),
        "n_birth_above_log10_4": int(np.sum(above)),
        "n_survivors": int(survives.sum()),
        "n_detected": int(detected.sum()),
        "true_mean_detectability_survivors": float(detected.sum() / max(survives.sum(), 1)),
        "true_mean_detectability_survivors_above_log10_4": float(
            np.sum(detected & above) / max(np.sum(survives & above), 1)
        ),
    }
    return catalog, truth_row


def run_one(job: dict[str, object]) -> list[dict[str, object]]:
    warnings.simplefilter("ignore")
    from globular_clusters_imf.detectability_longitude_model import fit_single_component_detectability_em_with_abs_longitude
    from globular_clusters_imf.joint_model import JointModelSpec

    truth = load_truth(Path(job["truth_path"]))
    rng = np.random.default_rng(int(job["seed"]))
    catalog, truth_row = simulate(truth, float(job["n0"]), np.asarray(job["completeness_raw"]), rng)
    override = {
        "log_mass_grid": truth["log_mass_grid"],
        "log_a_grid": truth["log_a_grid"],
        "semi_major_axis_grid_kpc": np.power(10.0, truth["log_a_grid"]),
        "survival_probability": truth["survival"],
        "selection_offset_dex": 0.0,
    }
    rows = []
    for n_iterations in job["iteration_caps"]:
        fit = fit_single_component_detectability_em_with_abs_longitude(
            catalog,
            project_root=Path(job["scratch"]),
            spec=JointModelSpec(imf_family="schechter", radial_model="logpoly3"),
            n_iterations=int(n_iterations),
            fixed_imf_params=truth["imf_params"],
            survival_grid_override=override,
        )
        model = fit["final_payload"]["model"]
        lm = np.asarray(fit["base_context"].log_mass_grid)
        imf = np.asarray(model["imf_density_grid"])
        frac_above = float(np.trapezoid(np.where(lm >= 4.0, imf, 0.0), lm) / np.trapezoid(imf, lm))
        frac_above_5p5 = float(np.trapezoid(np.where(lm >= 5.5, imf, 0.0), lm) / np.trapezoid(imf, lm))
        base_model = fit["baseline_payload"]["model"]  # same fit with Q = 1 (perfect detectability)
        base_imf = np.asarray(base_model["imf_density_grid"])
        base_frac = float(np.trapezoid(np.where(lm >= 4.0, base_imf, 0.0), lm) / np.trapezoid(base_imf, lm))
        base_frac_5p5 = float(np.trapezoid(np.where(lm >= 5.5, base_imf, 0.0), lm) / np.trapezoid(base_imf, lm))
        sel = np.asarray(fit["final_context"].selection_probability_grid)
        surv = np.asarray(fit["final_context"].survival_probability_grid)
        weight = imf[:, None] * np.asarray(model["radial_density_grid"])[None, :] * surv * (lm >= 4.0)[:, None]
        rows.append(
            {
                "regime": job["regime"],
                "seed": int(job["seed"]),
                "iteration_cap": int(n_iterations),
                **truth_row,
                "fit_n0_total": float(model["total_initial_count"]),
                "fit_n0_above_log10_4": float(model["total_initial_count"]) * frac_above,
                "fit_n0_above_log10_5p5": float(model["total_initial_count"]) * frac_above_5p5,
                "q1_n0_above_log10_4": float(base_model["total_initial_count"]) * base_frac,
                "q1_n0_above_log10_5p5": float(base_model["total_initial_count"]) * base_frac_5p5,
                "fit_mean_detectability_above_log10_4": float(np.sum(sel / np.clip(surv, 1e-12, None) * weight) / np.sum(weight)),
                "fit_log_likelihood": float(fit["final_payload"]["summary"].log_likelihood),
                "converged": bool(fit["em_converged"]),
                "iterations_run": int(fit["em_iterations_run"]),
                "residual": float(fit["em_parameter_residual"]),
                "fit_completeness_intercept": float(fit["final_completeness_raw_parameters"][0]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-realisations", type=int, default=8)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--intercept-shift-low", type=float, default=-3.0,
                        help="intercept shift for the low-completeness regime")
    parser.add_argument("--target-detected", type=float, default=165.0)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--iteration-caps", type=int, nargs="+", default=[12, 60])
    args = parser.parse_args()

    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(variable, "1")
    args.output.mkdir(parents=True, exist_ok=True)
    scratch = args.output / "fit_scratch"
    scratch.mkdir(exist_ok=True)

    truth = load_truth(args.truth)
    regimes = {
        "realistic": truth["completeness_raw"],
        "low": truth["completeness_raw"] + np.array([args.intercept_shift_low, 0, 0, 0, 0]),
    }
    # Scale N0 per regime so that the expected detected count is ~target (sample size held fixed).
    rng = np.random.default_rng(args.seed)
    n0_by_regime = {}
    for name, raw in regimes.items():
        _, pilot = simulate(truth, 20.0 * truth["n0_total"], raw, rng)
        n0_by_regime[name] = args.target_detected / (pilot["n_detected"] / pilot["n_birth"])
    jobs = [
        {
            "regime": name,
            "seed": args.seed + 1000 * k + i,
            "n0": n0_by_regime[name],
            "completeness_raw": regimes[name].tolist(),
            "truth_path": str(args.truth),
            "scratch": str(scratch),
            "iteration_caps": list(args.iteration_caps),
        }
        for k, name in enumerate(regimes)
        for i in range(args.n_realisations)
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, job) for job in jobs]
        for job, future in zip(jobs, futures):
            try:
                rows.extend(future.result())
            except Exception as error:  # record and continue: one bad catalogue must not hide the rest
                rows.append({"regime": job["regime"], "seed": job["seed"], "failure": repr(error)})
    table = pd.DataFrame(rows)
    if "failure" in table:
        print("FAILED jobs:", table.loc[table["failure"].notna(), ["regime", "seed", "failure"]].to_string())
        table = table.loc[table["failure"].isna()].drop(columns="failure")
    table.to_csv(args.output / "injection_recovery.csv", index=False)
    (args.output / "injection_recovery_config.json").write_text(
        json.dumps({"truth": str(args.truth), "n0_by_regime": n0_by_regime, "regimes": {k: v.tolist() for k, v in regimes.items()}}, indent=2)
    )
    table["n0_ratio"] = table["fit_n0_above_log10_4"] / table["n_birth_above_log10_4"]
    table["q_error"] = table["fit_mean_detectability_above_log10_4"] - table["true_mean_detectability_survivors_above_log10_4"]
    summary = table.groupby(["regime", "iteration_cap"]).agg(
        n_detected=("n_detected", "median"),
        true_q=("true_mean_detectability_survivors_above_log10_4", "median"),
        fit_q=("fit_mean_detectability_above_log10_4", "median"),
        n0_ratio_median=("n0_ratio", "median"),
        n0_ratio_p16=("n0_ratio", lambda v: np.percentile(v, 16)),
        n0_ratio_p84=("n0_ratio", lambda v: np.percentile(v, 84)),
        converged_fraction=("converged", "mean"),
    )
    print(summary.round(3).to_string())
    print(args.output / "injection_recovery.csv")


if __name__ == "__main__":
    main()
