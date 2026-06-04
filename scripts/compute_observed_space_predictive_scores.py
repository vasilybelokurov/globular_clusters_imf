from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special

from globular_clusters_imf.detectability_longitude_model import absolute_wrapped_longitude_degrees
from globular_clusters_imf.joint_model import centers_to_edges_local


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "tables" / "observed_space_predictive_scores.csv"


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


def multinomial_log_likelihood(observed: np.ndarray, predicted: np.ndarray) -> float:
    y = np.asarray(observed, dtype=float)
    mu = np.clip(np.asarray(predicted, dtype=float), 1.0e-300, None)
    probability = mu / np.sum(mu)
    n_observed = float(np.sum(y))
    return float(
        special.gammaln(n_observed + 1.0)
        + np.sum(y * np.log(probability) - special.gammaln(y + 1.0))
    )


def deviance(observed: np.ndarray, predicted: np.ndarray) -> float:
    y = np.asarray(observed, dtype=float)
    mu = np.clip(np.asarray(predicted, dtype=float), 1.0e-300, None)
    term = np.where(y > 0.0, y * np.log(np.clip(y, 1.0e-300, None) / mu), 0.0)
    return float(2.0 * np.sum(term - (y - mu)))


def model_path(run: ModelRun) -> Path:
    path = PROJECT_ROOT / "variants" / run.variant / "outputs" / "tables" / "exact_parallel_mcmc_best_result.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_best_result(run: ModelRun) -> dict[str, object]:
    with model_path(run).open("rb") as handle:
        return pickle.load(handle)


def selected_cell_counts(result: dict[str, object]) -> np.ndarray:
    context = result["final_context"]
    model = result["final_payload"]["model"]
    log_mass_edges = centers_to_edges_local(context.log_mass_grid)
    log_a_edges = centers_to_edges_local(context.log_a_grid)
    selected_intensity = (
        float(model["total_initial_count"])
        * np.asarray(model["imf_density_grid"], dtype=float)[:, None]
        * np.asarray(model["radial_density_grid"], dtype=float)[None, :]
        * np.asarray(context.selection_probability_grid, dtype=float)
    )
    return selected_intensity * np.diff(log_mass_edges)[:, None] * np.diff(log_a_edges)[None, :]


def complete_survivor_cell_counts(result: dict[str, object]) -> np.ndarray:
    context = result["final_context"]
    log_mass_edges = centers_to_edges_local(context.log_mass_grid)
    log_a_edges = centers_to_edges_local(context.log_a_grid)
    return (
        np.asarray(result["final_complete_survivor_intensity_grid"], dtype=float)
        * np.diff(log_mass_edges)[:, None]
        * np.diff(log_a_edges)[None, :]
    )


def log_a_bin_edges(result: dict[str, object], n_a_bins: int) -> np.ndarray:
    context = result["final_context"]
    fine_edges = centers_to_edges_local(context.log_a_grid)
    return np.linspace(float(fine_edges[0]), float(fine_edges[-1]), n_a_bins + 1)


def observed_catalog_table(result: dict[str, object]) -> pd.DataFrame:
    table = result["catalog_completeness_table"].copy()
    if "log10_semi_major_axis_kpc" not in table.columns:
        table["log10_semi_major_axis_kpc"] = np.log10(table["semi_major_axis_kpc"].to_numpy(dtype=float))
    if "abs_galactic_l_deg" not in table.columns and "galactic_l_deg" in table.columns:
        table["abs_galactic_l_deg"] = absolute_wrapped_longitude_degrees(table["galactic_l_deg"].to_numpy(dtype=float))
    return table


def predict_mnow_a_counts(result: dict[str, object], a_edges: np.ndarray) -> np.ndarray:
    observable_context = result["observable_context"]
    context = result["final_context"]
    mass_counts_by_a = np.einsum(
        "ma,mak->ak",
        selected_cell_counts(result),
        np.asarray(observable_context.mass_bin_probabilities_grid, dtype=float),
    )
    a_bin_index = np.searchsorted(a_edges, context.log_a_grid, side="right") - 1
    a_bin_index = np.clip(a_bin_index, 0, len(a_edges) - 2)
    predicted = np.zeros((len(observable_context.log_present_mass_edges) - 1, len(a_edges) - 1), dtype=float)
    for fine_a_index, coarse_a_index in enumerate(a_bin_index):
        predicted[:, coarse_a_index] += mass_counts_by_a[fine_a_index, :]
    return predicted


def observed_mnow_a_counts(result: dict[str, object], a_edges: np.ndarray) -> np.ndarray:
    observable_context = result["observable_context"]
    table = observed_catalog_table(result)
    counts, _ = np.histogramdd(
        np.column_stack(
            [
                table["log10_present_mass_msun"].to_numpy(dtype=float),
                table["log10_semi_major_axis_kpc"].to_numpy(dtype=float),
            ]
        ),
        bins=[observable_context.log_present_mass_edges, a_edges],
    )
    return counts


def predict_mnow_a_sky_counts(result: dict[str, object], a_edges: np.ndarray) -> np.ndarray:
    observable_context = result["observable_context"]
    context = result["final_context"]
    mass_counts_by_a = np.einsum(
        "ma,mak->ak",
        complete_survivor_cell_counts(result),
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
    return predicted_complete * np.asarray(result["final_completeness_bin_grid"], dtype=float)[:, None, :, :, :]


def observed_mnow_a_sky_counts(result: dict[str, object], a_edges: np.ndarray) -> np.ndarray:
    observable_context = result["observable_context"]
    table = observed_catalog_table(result)
    counts, _ = np.histogramdd(
        np.column_stack(
            [
                table["log10_present_mass_msun"].to_numpy(dtype=float),
                table["log10_semi_major_axis_kpc"].to_numpy(dtype=float),
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
    return counts


def score_one(run: ModelRun, n_a_bins: int) -> list[dict[str, object]]:
    result = load_best_result(run)
    observable_context = result["observable_context"]
    a_edges = log_a_bin_edges(result, n_a_bins=n_a_bins)
    score_inputs = [
        (
            "logMnow_loga",
            observed_mnow_a_counts(result, a_edges),
            predict_mnow_a_counts(result, a_edges),
            "selected intrinsic S*Q projected through p(Mnow|Mini,a)",
        ),
        (
            "logMnow_D_absb_absl",
            np.asarray(observable_context.observed_counts, dtype=float),
            np.asarray(result["final_predicted_observed_counts"], dtype=float),
            "saved observable C(logMnow,D,|b|,|l|)",
        ),
        (
            "logMnow_loga_D_absb_absl",
            observed_mnow_a_sky_counts(result, a_edges),
            predict_mnow_a_sky_counts(result, a_edges),
            "saved observable C(logMnow,D,|b|,|l|), with a retained before projection",
        ),
    ]

    rows = []
    for score_name, observed, predicted, selection_treatment in score_inputs:
        rows.append(
            {
                "model_label": run.label,
                "variant": run.variant,
                "score_space": score_name,
                "score_kind": "best_fit_plugin",
                "n_a_bins": n_a_bins if "loga" in score_name else 0,
                "poisson_log_likelihood": poisson_log_likelihood(observed, predicted),
                "multinomial_log_likelihood": multinomial_log_likelihood(observed, predicted),
                "poisson_deviance": deviance(observed, predicted),
                "observed_total_count": float(np.sum(observed)),
                "predicted_total_count": float(np.sum(predicted)),
                "n_observed_bins_nonzero": int(np.count_nonzero(observed)),
                "n_prediction_bins": int(np.size(predicted)),
                "selection_treatment": selection_treatment,
            }
        )
    return rows


def main() -> None:
    all_rows = []
    for run in RUNS:
        all_rows.extend(score_one(run, n_a_bins=6))
    table = pd.DataFrame(all_rows)
    table["delta_poisson_log_likelihood"] = table.groupby("score_space")["poisson_log_likelihood"].transform(
        lambda values: values - values.max()
    )
    table["delta_multinomial_log_likelihood"] = table.groupby("score_space")[
        "multinomial_log_likelihood"
    ].transform(lambda values: values - values.max())
    table = table.sort_values(["score_space", "poisson_log_likelihood"], ascending=[True, False])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_PATH, index=False)
    with pd.option_context("display.max_columns", 20, "display.width", 180):
        print(table.to_string(index=False))
    print(f"\nWrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
