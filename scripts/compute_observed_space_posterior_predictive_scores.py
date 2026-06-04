"""Posterior predictive scores in common observed spaces.

This is a diagnostic/ranking script.  It does not update model outputs used by
the fitting pipeline or by paper-number generation.  It reads saved MCMC worker
surface files, reconstructs each sampled detectability-corrected model, refits
the observable completeness law C for that draw, and scores the observed GC
catalog in binned observed spaces.

The score reported as ``posterior_predictive_log_likelihood`` is

    log mean_theta p(observed binned counts | theta),

estimated by Monte Carlo over posterior draws.  This differs from the faster
best-fit plug-in score in ``compute_observed_space_predictive_scores.py``.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate, special

from globular_clusters_imf.detectability_model import compute_complete_survivor_intensity_grid
from globular_clusters_imf.detectability_longitude_model import (
    fit_logistic_completeness_model_with_abs_longitude,
    predict_complete_observable_histogram_with_abs_longitude,
)
from globular_clusters_imf.joint_model import (
    JointLikelihoodContext,
    JointModelSpec,
    centers_to_edges_local,
    fit_single_joint_model_with_fixed_imf_params,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_OUTPUT = PROJECT_ROOT / "outputs" / "tables" / "observed_space_posterior_predictive_summary.csv"
DEFAULT_DRAWS_OUTPUT = PROJECT_ROOT / "outputs" / "tables" / "observed_space_posterior_predictive_draw_scores.csv"


@dataclass(frozen=True)
class ModelRun:
    label: str
    variant: str


RUNS = [
    ModelRun(
        "Baumgardt 2019",
        "profile_map_and_exact_mcmc_schechter_logpoly3_logistic_global_monotonic_q",
    ),
    ModelRun("GG23 no BHs", "gg23_schechter_no_bh_logpoly3_eta01_105"),
    ModelRun("GG23 BHs", "gg23_schechter_bh_logpoly3"),
    ModelRun("GG23 BHs + [Fe/H]", "gg23_schechter_bh_feh_gradient_logpoly3"),
    ModelRun("GG23 BHs + past tides", "gg23_schechter_bh_past_tidal_logpoly3"),
    ModelRun(
        "GG23 BHs + [Fe/H] + past tides",
        "gg23_schechter_bh_feh_gradient_past_tidal_logpoly3",
    ),
]


def poisson_log_likelihood(observed: np.ndarray, predicted: np.ndarray) -> float:
    y = np.asarray(observed, dtype=float)
    mu = np.clip(np.asarray(predicted, dtype=float), 1.0e-300, None)
    return float(np.sum(y * np.log(mu) - mu - special.gammaln(y + 1.0)))


def logmeanexp(values: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    x_max = float(np.max(x))
    weights = np.exp(x - x_max)
    mean_weight = float(np.mean(weights))
    log_mean = x_max + float(np.log(mean_weight))
    if len(weights) > 1 and mean_weight > 0.0:
        log_mc_se = float(np.std(weights, ddof=1) / np.sqrt(len(weights)) / mean_weight)
    else:
        log_mc_se = 0.0
    effective_draws = float(np.square(np.sum(weights)) / np.sum(np.square(weights)))
    return log_mean, log_mc_se, effective_draws


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def load_best_result(run: ModelRun) -> dict[str, object]:
    path = PROJECT_ROOT / "variants" / run.variant / "outputs" / "tables" / "exact_parallel_mcmc_best_result.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_surface_table(run: ModelRun) -> pd.DataFrame:
    worker_dir = PROJECT_ROOT / "variants" / run.variant / "outputs" / "parallel_exact_mcmc_workers"
    frames = []
    for csv_path in sorted(worker_dir.glob("chain_*_selection_surfaces.csv")):
        chain = int(csv_path.name.split("_")[1])
        frame = pd.read_csv(csv_path).reset_index(names="surface_index")
        frame["chain"] = chain
        frame["surface_npz"] = str(worker_dir / f"chain_{chain}_selection_surfaces.npz")
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No posterior surface CSV files found under {worker_dir}")
    table = pd.concat(frames, ignore_index=True)
    ok = table.loc[table["status"].astype(str) == "ok"].copy()
    if ok.empty:
        raise RuntimeError(f"No successful posterior surface rows for {run.variant}")
    return ok.reset_index(drop=True)


def sample_surface_rows(table: pd.DataFrame, n_draws: int, seed: int) -> pd.DataFrame:
    if n_draws <= 0 or n_draws >= len(table):
        return table.reset_index(drop=True)
    return table.sample(n=n_draws, random_state=seed).reset_index(drop=True)


def context_with_surface_draws(
    base_context: JointLikelihoodContext,
    survival_probability: np.ndarray,
    effective_detectability: np.ndarray,
) -> JointLikelihoodContext:
    survival_probability = np.clip(np.asarray(survival_probability, dtype=float), 1.0e-12, 1.0)
    selection_probability = np.clip(
        survival_probability * np.asarray(effective_detectability, dtype=float),
        1.0e-12,
        1.0,
    )
    survival_interpolator = interpolate.RegularGridInterpolator(
        (base_context.log_mass_grid, base_context.log_a_grid),
        survival_probability,
        bounds_error=False,
        fill_value=None,
    )
    selection_interpolator = interpolate.RegularGridInterpolator(
        (base_context.log_mass_grid, base_context.log_a_grid),
        selection_probability,
        bounds_error=False,
        fill_value=None,
    )
    return JointLikelihoodContext(
        log_mass_data=base_context.log_mass_data.copy(),
        log_a_data=base_context.log_a_data.copy(),
        log_mass_grid=base_context.log_mass_grid.copy(),
        log_a_grid=base_context.log_a_grid.copy(),
        survival_probability_grid=survival_probability,
        survival_interpolator=survival_interpolator,
        selection_probability_grid=selection_probability,
        selection_interpolator=selection_interpolator,
        radial_step_edges=base_context.radial_step_edges.copy(),
        log_a_mean=base_context.log_a_mean,
        log_a_std=base_context.log_a_std,
    )


def observed_mnow_a_counts(result: dict[str, object], a_edges: np.ndarray) -> np.ndarray:
    observable_context = result["observable_context"]
    table = result["catalog_completeness_table"].copy()
    log_a = np.log10(table["semi_major_axis_kpc"].to_numpy(dtype=float))
    counts, _ = np.histogramdd(
        np.column_stack([table["log10_present_mass_msun"].to_numpy(dtype=float), log_a]),
        bins=[observable_context.log_present_mass_edges, a_edges],
    )
    return counts.astype(float)


def observed_mnow_a_sky_counts(result: dict[str, object], a_edges: np.ndarray) -> np.ndarray:
    observable_context = result["observable_context"]
    table = result["catalog_completeness_table"].copy()
    log_a = np.log10(table["semi_major_axis_kpc"].to_numpy(dtype=float))
    counts, _ = np.histogramdd(
        np.column_stack(
            [
                table["log10_present_mass_msun"].to_numpy(dtype=float),
                log_a,
                table["r_sun_kpc"].to_numpy(dtype=float),
                table["abs_galactic_b_deg"].to_numpy(dtype=float),
                table["abs_galactic_l_deg"].to_numpy(dtype=float),
            ]
        ),
        bins=[
            observable_context.log_present_mass_edges,
            a_edges,
            observable_context.distance_edges_kpc,
            observable_context.abs_latitude_edges_deg,
            observable_context.abs_longitude_edges_deg,
        ],
    )
    return counts.astype(float)


def predict_mnow_a_counts(
    result: dict[str, object],
    context: JointLikelihoodContext,
    model: dict[str, object],
    a_edges: np.ndarray,
) -> np.ndarray:
    observable_context = result["observable_context"]
    log_mass_edges = centers_to_edges_local(context.log_mass_grid)
    log_a_edges = centers_to_edges_local(context.log_a_grid)
    selected_intensity = (
        float(model["total_initial_count"])
        * np.asarray(model["imf_density_grid"], dtype=float)[:, None]
        * np.asarray(model["radial_density_grid"], dtype=float)[None, :]
        * np.asarray(context.selection_probability_grid, dtype=float)
    )
    selected_cell_counts = selected_intensity * np.diff(log_mass_edges)[:, None] * np.diff(log_a_edges)[None, :]
    mass_counts_by_a = np.einsum(
        "ma,mak->ak",
        selected_cell_counts,
        np.asarray(observable_context.mass_bin_probabilities_grid, dtype=float),
    )
    a_bin_index = np.searchsorted(a_edges, context.log_a_grid, side="right") - 1
    a_bin_index = np.clip(a_bin_index, 0, len(a_edges) - 2)
    predicted = np.zeros((len(observable_context.log_present_mass_edges) - 1, len(a_edges) - 1), dtype=float)
    for fine_a_index, coarse_a_index in enumerate(a_bin_index):
        predicted[:, coarse_a_index] += mass_counts_by_a[fine_a_index, :]
    return predicted


def predict_mnow_a_sky_counts(
    result: dict[str, object],
    context: JointLikelihoodContext,
    complete_survivor_intensity_grid: np.ndarray,
    completeness_bin_grid: np.ndarray,
    a_edges: np.ndarray,
) -> np.ndarray:
    observable_context = result["observable_context"]
    log_mass_edges = centers_to_edges_local(context.log_mass_grid)
    log_a_edges = centers_to_edges_local(context.log_a_grid)
    complete_cell_counts = (
        np.asarray(complete_survivor_intensity_grid, dtype=float)
        * np.diff(log_mass_edges)[:, None]
        * np.diff(log_a_edges)[None, :]
    )
    mass_counts_by_a = np.einsum(
        "ma,mak->ak",
        complete_cell_counts,
        np.asarray(observable_context.mass_bin_probabilities_grid, dtype=float),
    )
    a_bin_index = np.searchsorted(a_edges, context.log_a_grid, side="right") - 1
    a_bin_index = np.clip(a_bin_index, 0, len(a_edges) - 2)
    a_membership = np.zeros((len(context.log_a_grid), len(a_edges) - 1), dtype=float)
    a_membership[np.arange(len(context.log_a_grid)), a_bin_index] = 1.0
    predicted_complete = np.einsum(
        "ak,ai,adbl->kidbl",
        mass_counts_by_a,
        a_membership,
        np.asarray(observable_context.sky_bin_probabilities_by_a, dtype=float),
    )
    return predicted_complete * np.asarray(completeness_bin_grid, dtype=float)[:, None, :, :, :]


def score_run(run: ModelRun, n_draws: int, seed: int, n_a_bins: int) -> list[dict[str, object]]:
    result = load_best_result(run)
    base_context = result["final_context"]
    observable_context = result["observable_context"]
    spec = JointModelSpec(imf_family="schechter", radial_model="logpoly3")
    surface_table = sample_surface_rows(load_surface_table(run), n_draws=n_draws, seed=seed)
    a_edges = np.linspace(
        float(centers_to_edges_local(base_context.log_a_grid)[0]),
        float(centers_to_edges_local(base_context.log_a_grid)[-1]),
        n_a_bins + 1,
    )
    observed_by_space = {
        "logMnow_loga": observed_mnow_a_counts(result, a_edges),
        "logMnow_D_absb_absl": np.asarray(observable_context.observed_counts, dtype=float),
        "logMnow_loga_D_absb_absl": observed_mnow_a_sky_counts(result, a_edges),
    }

    npz_cache: dict[str, np.lib.npyio.NpzFile] = {}
    draw_rows = []
    radial_start = np.asarray(result["final_payload"]["radial_parameters_raw"], dtype=float)
    completeness_start = np.asarray(result["final_completeness_raw_parameters"], dtype=float)
    for draw_index, row in enumerate(surface_table.itertuples(index=False)):
        npz_path = str(row.surface_npz)
        if npz_path not in npz_cache:
            npz_cache[npz_path] = np.load(npz_path)
        surfaces = npz_cache[npz_path]
        surface_index = int(row.surface_index)
        context = context_with_surface_draws(
            base_context,
            surfaces["survival_probability"][surface_index],
            surfaces["effective_detectability"][surface_index],
        )
        payload = fit_single_joint_model_with_fixed_imf_params(
            context=context,
            spec=spec,
            fixed_imf_params=np.array(
                [float(row.input_alpha_dndm), float(row.input_log10_m_c_msun)],
                dtype=float,
            ),
            start_radial_params=radial_start,
        )
        radial_start = np.asarray(payload["radial_parameters_raw"], dtype=float)
        complete_survivor_intensity_grid = compute_complete_survivor_intensity_grid(
            payload["model"],
            base_context=context,
        )
        predicted_complete_4d = predict_complete_observable_histogram_with_abs_longitude(
            complete_survivor_intensity_grid=complete_survivor_intensity_grid,
            base_context=context,
            observable_context=observable_context,
        )
        completeness_fit = fit_logistic_completeness_model_with_abs_longitude(
            observable_context=observable_context,
            predicted_complete_counts=predicted_complete_4d,
            start_params=completeness_start,
        )
        completeness_start = np.asarray(completeness_fit["raw_parameters"], dtype=float)
        completeness_grid = np.asarray(completeness_fit["completeness_bin_grid"], dtype=float)
        predicted_by_space = {
            "logMnow_loga": predict_mnow_a_counts(result, context, payload["model"], a_edges),
            "logMnow_D_absb_absl": predicted_complete_4d * completeness_grid,
            "logMnow_loga_D_absb_absl": predict_mnow_a_sky_counts(
                result,
                context,
                complete_survivor_intensity_grid,
                completeness_grid,
                a_edges,
            ),
        }
        for score_space, observed in observed_by_space.items():
            predicted = predicted_by_space[score_space]
            draw_rows.append(
                {
                    "model_label": run.label,
                    "variant": run.variant,
                    "score_space": score_space,
                    "draw_index": draw_index,
                    "chain": int(row.chain),
                    "surface_index": surface_index,
                    "step": int(row.step),
                    "eta_t": float(row.eta_t),
                    "input_alpha_dndm": float(row.input_alpha_dndm),
                    "input_log10_m_c_msun": float(row.input_log10_m_c_msun),
                    "fit_log_likelihood_intrinsic": float(payload["summary"].log_likelihood),
                    "poisson_log_likelihood": poisson_log_likelihood(observed, predicted),
                    "observed_total_count": float(np.sum(observed)),
                    "predicted_total_count": float(np.sum(predicted)),
                    "n_prediction_bins": int(np.size(predicted)),
                }
            )

    for npz_file in npz_cache.values():
        npz_file.close()
    return draw_rows


def summarize(draw_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_label, variant, score_space), group in draw_table.groupby(
        ["model_label", "variant", "score_space"], sort=False
    ):
        values = group["poisson_log_likelihood"].to_numpy(dtype=float)
        log_mean, log_mc_se, effective_draws = logmeanexp(values)
        rows.append(
            {
                "model_label": model_label,
                "variant": variant,
                "score_space": score_space,
                "n_draws": int(len(group)),
                "posterior_predictive_log_likelihood": log_mean,
                "posterior_predictive_log_likelihood_mc_se": log_mc_se,
                "effective_draws_logmeanexp": effective_draws,
                "median_draw_log_likelihood": float(np.median(values)),
                "p16_draw_log_likelihood": float(np.quantile(values, 0.16)),
                "p84_draw_log_likelihood": float(np.quantile(values, 0.84)),
                "mean_predicted_total_count": float(group["predicted_total_count"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    summary["delta_posterior_predictive_log_likelihood"] = summary.groupby("score_space")[
        "posterior_predictive_log_likelihood"
    ].transform(lambda values: values - values.max())
    return summary.sort_values(
        ["score_space", "posterior_predictive_log_likelihood"],
        ascending=[True, False],
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-draws", type=int, default=200, help="Posterior surface draws per model; <=0 uses all saved draws.")
    parser.add_argument("--seed", type=int, default=20260604, help="Random seed used to subsample posterior draws.")
    parser.add_argument("--n-a-bins", type=int, default=6, help="Number of coarse log-a bins for scores that retain a.")
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--draw-output", type=Path, default=DEFAULT_DRAWS_OUTPUT)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset by exact model label.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_runs = RUNS
    if args.models:
        allowed = set(args.models)
        selected_runs = [run for run in RUNS if run.label in allowed]
        missing = allowed.difference(run.label for run in selected_runs)
        if missing:
            raise ValueError(f"Unknown model labels: {sorted(missing)}")

    draw_rows = []
    for run in selected_runs:
        print(f"Scoring {run.label} with n_draws={args.n_draws}...")
        draw_rows.extend(score_run(run, n_draws=args.n_draws, seed=args.seed, n_a_bins=args.n_a_bins))

    draw_table = pd.DataFrame(draw_rows)
    summary = summarize(draw_table)
    args.draw_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    draw_table.to_csv(args.draw_output, index=False)
    summary.to_csv(args.summary_output, index=False)

    with pd.option_context("display.max_columns", 20, "display.width", 180):
        print(summary.to_string(index=False))
    print(f"\nWrote {display_path(args.summary_output)}")
    print(f"Wrote {display_path(args.draw_output)}")


if __name__ == "__main__":
    main()
