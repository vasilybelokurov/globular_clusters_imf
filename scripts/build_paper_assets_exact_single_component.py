from __future__ import annotations

import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

LIGHT_GREYS = LinearSegmentedColormap.from_list(
    "light_greys",
    plt.get_cmap("Greys")(np.linspace(0.0, 0.78, 256)),
)

from globular_clusters_imf.detectability_model import (  # noqa: E402
    ObservablePredictionContext,
    build_completeness_grid_table,
    fit_logistic_completeness_model,
)
from globular_clusters_imf.detectability_longitude_model import (  # noqa: E402
    fit_single_component_detectability_em_with_abs_longitude,
)
from globular_clusters_imf.joint_model import compute_observed_intensity_grid  # noqa: E402
from globular_clusters_imf.joint_model import JointModelSpec  # noqa: E402
from globular_clusters_imf.model import fit_catalog_models  # noqa: E402
from globular_clusters_imf.paper_assets import (  # noqa: E402
    PAPER_LOG_MASS_MIN,
    plot_catalog_mass_semimajor_axis_overview_for_paper,
    plot_detectability_em_convergence_for_paper,
    plot_detectability_em_maps_by_longitude_split_for_paper,
    poisson_count_errors,
    sample_joint_projection_bands,
)
from globular_clusters_imf.plotting import centers_to_edges, rebin_expected_counts_2d  # noqa: E402
from globular_clusters_imf.smooth_survivability import build_smooth_survivability_grid  # noqa: E402
from run_profile_map_and_exact_mcmc_schechter_powerlaw_a import _corner_plot  # noqa: E402


LOGPOLY3_VARIANT = "profile_map_and_exact_mcmc_schechter_logpoly3_logistic_global_monotonic_q"
STEP5_VARIANT = "profile_map_and_exact_mcmc_schechter_step5_logistic_global_monotonic_q"
POWERLAW_A_VARIANT = "profile_map_and_exact_mcmc_schechter_powerlaw_a_logistic_global_monotonic_q"
CORED_POWERLAW_A_VARIANT = "profile_map_and_exact_mcmc_schechter_cored_powerlaw_a_logistic_global_monotonic_q"


@dataclass
class ExactRunBundle:
    variant_name: str
    best_result: dict[str, object]
    best_summary: dict[str, object]
    posterior_summary: pd.DataFrame
    posterior_samples: pd.DataFrame
    refined_grid: pd.DataFrame
    raw_parameter_samples: pd.DataFrame | None


def _load_exact_bundle(variant_name: str) -> ExactRunBundle:
    tables_dir = PROJECT_ROOT / "variants" / variant_name / "outputs" / "tables"
    with (tables_dir / "exact_parallel_mcmc_best_result.pkl").open("rb") as handle:
        best_result = pickle.load(handle)
    best_summary = json.loads((tables_dir / "exact_parallel_mcmc_best_result_summary.json").read_text())
    posterior_summary = pd.read_csv(tables_dir / "exact_parallel_posterior_summary.csv")
    posterior_samples = pd.read_csv(tables_dir / "exact_parallel_mcmc_posterior_samples.csv")
    refined_grid = pd.read_csv(tables_dir / "refined_grid_results.csv")
    raw_parameter_samples_path = tables_dir / "exact_parallel_raw_parameter_samples.csv"
    raw_parameter_samples = None
    if raw_parameter_samples_path.exists():
        raw_parameter_samples = pd.read_csv(raw_parameter_samples_path)
    return ExactRunBundle(
        variant_name=variant_name,
        best_result=best_result,
        best_summary=best_summary,
        posterior_summary=posterior_summary,
        posterior_samples=posterior_samples,
        refined_grid=refined_grid,
        raw_parameter_samples=raw_parameter_samples,
    )


def _posterior_row(summary_table: pd.DataFrame, parameter: str) -> pd.Series:
    match = summary_table.loc[summary_table["parameter"] == parameter]
    if match.empty:
        raise KeyError(f"Missing posterior summary row for {parameter}")
    return match.iloc[0]


def _build_uncertainty_payload(bundle: ExactRunBundle) -> dict[str, np.ndarray]:
    if bundle.raw_parameter_samples is None:
        raw = np.asarray(bundle.best_result["final_payload"]["raw_parameters"], dtype=float)
        print(
            f"WARNING: missing posterior raw-parameter support samples for {bundle.variant_name}; "
            "using the best profiled model for projection bands."
        )
        return {"raw_samples": raw[None, :]}
    raw_columns = [column for column in bundle.raw_parameter_samples.columns if column.startswith("raw_param_")]
    raw_samples = bundle.raw_parameter_samples.loc[:, raw_columns].to_numpy(dtype=float)
    return {"raw_samples": raw_samples}


def _build_projection_bundle(
    *,
    catalog: pd.DataFrame,
    context,
    best_payload: dict[str, object],
    uncertainty_payload: dict[str, np.ndarray],
    n_projection_samples: int = 250,
) -> dict[str, object]:
    point_intensity_grid = compute_observed_intensity_grid(
        np.asarray(best_payload["model"]["imf_density_grid"]),
        np.asarray(best_payload["model"]["radial_density_grid"]),
        np.asarray(context.selection_probability_grid),
        float(best_payload["model"]["total_initial_count"]),
    )
    log_mass_grid = np.asarray(context.log_mass_grid)
    log_a_grid = np.asarray(context.log_a_grid)
    radius_grid_kpc = np.power(10.0, log_a_grid)

    log_mass_edges = centers_to_edges(log_mass_grid)
    log_a_edges = centers_to_edges(log_a_grid)
    radius_edges_kpc = 10.0 ** log_a_edges

    mass_bin_edges = np.linspace(log_mass_edges[0], log_mass_edges[-1], 13)
    log_a_bin_edges = np.linspace(log_a_edges[0], log_a_edges[-1], 10)

    observed_mass_counts, _ = np.histogram(context.log_mass_data, bins=mass_bin_edges)
    observed_a_counts, _ = np.histogram(context.log_a_data, bins=log_a_bin_edges)

    mass_yerr = poisson_count_errors(observed_mass_counts)
    a_yerr = poisson_count_errors(observed_a_counts)

    cell_weights = point_intensity_grid * np.diff(log_mass_edges)[:, None] * np.diff(log_a_edges)[None, :]
    expected_2d = rebin_expected_counts_2d(
        cell_weights,
        log_mass_grid=log_mass_grid,
        log_a_grid=log_a_grid,
        mass_bin_edges=mass_bin_edges,
        log_a_bin_edges=log_a_bin_edges,
    )
    observed_2d = np.histogram2d(
        context.log_mass_data,
        context.log_a_data,
        bins=[mass_bin_edges, log_a_bin_edges],
    )[0]
    residual_significance = (observed_2d - expected_2d) / np.sqrt(np.clip(expected_2d, 1.0, None))

    sample_projection = sample_joint_projection_bands(
        context=context,
        best_payload=best_payload,
        raw_samples=np.asarray(uncertainty_payload["raw_samples"], dtype=float),
        n_samples=n_projection_samples,
        mass_bin_width=mass_bin_edges[1] - mass_bin_edges[0],
        log_a_bin_width=log_a_bin_edges[1] - log_a_bin_edges[0],
    )

    return {
        "catalog": catalog,
        "point_intensity_grid": point_intensity_grid,
        "log_mass_grid": log_mass_grid,
        "log_a_grid": log_a_grid,
        "radius_grid_kpc": radius_grid_kpc,
        "log_mass_edges": log_mass_edges,
        "log_a_edges": log_a_edges,
        "radius_edges_kpc": radius_edges_kpc,
        "mass_bin_edges": mass_bin_edges,
        "log_a_bin_edges": log_a_bin_edges,
        "observed_mass_counts": observed_mass_counts,
        "observed_a_counts": observed_a_counts,
        "mass_yerr": mass_yerr,
        "a_yerr": a_yerr,
        "residual_significance": residual_significance,
        "sample_projection": sample_projection,
    }


def plot_single_component_intensity_plane_for_paper(
    projection_bundle: dict[str, object],
    output_path: Path,
) -> None:
    display_probability = np.asarray(projection_bundle["point_intensity_grid"], dtype=float)
    display_probability = display_probability / np.nanmax(display_probability)
    positive = display_probability[np.isfinite(display_probability) & (display_probability > 0.0)]
    vmin = max(float(np.nanmin(positive)), 1.0e-3) if positive.size else 1.0e-3
    fig, ax = plt.subplots(figsize=(5.6, 4.4), constrained_layout=True)
    mesh = ax.pcolormesh(
        projection_bundle["radius_edges_kpc"],
        projection_bundle["log_mass_edges"],
        display_probability,
        cmap=LIGHT_GREYS,
        norm=mcolors.LogNorm(vmin=vmin, vmax=1.0),
        shading="auto",
        rasterized=True,
    )
    ax.contour(
        projection_bundle["radius_grid_kpc"],
        projection_bundle["log_mass_grid"],
        display_probability,
        levels=[0.01, 0.1, 0.5],
        colors=["#6f6f6f", "#404040", "#111111"],
        linewidths=[0.8, 1.0, 1.2],
    )
    ax.scatter(
        projection_bundle["catalog"]["semi_major_axis_kpc"],
        projection_bundle["catalog"]["log_initial_mass_msun"],
        s=13,
        color="black",
        alpha=0.9,
        edgecolors="none",
        linewidths=0.0,
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"Orbital semimajor axis $a$ [kpc]")
    ax.set_ylabel(r"$\log_{10}(M_{\rm ini}/{\rm M}_\odot)$")
    ax.set_title(r"Detected probability density in the $(M_{\rm ini}, a)$ plane")
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.01)
    colorbar.set_label("Relative detected probability density")
    fig.savefig(output_path)
    plt.close(fig)


def plot_three_panel_summary_for_paper(
    projection_bundle: dict[str, object],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(ncols=3, figsize=(11.5, 3.75), constrained_layout=True)
    ax_mass, ax_radius, ax_residual = axes
    sample_projection = projection_bundle["sample_projection"]

    mass_centers = 0.5 * (
        projection_bundle["mass_bin_edges"][:-1] + projection_bundle["mass_bin_edges"][1:]
    )
    ax_mass.errorbar(
        mass_centers,
        projection_bundle["observed_mass_counts"],
        yerr=projection_bundle["mass_yerr"],
        fmt="o",
        color="black",
        ms=4.0,
        capsize=2.5,
        label="Observed",
    )
    ax_mass.fill_between(
        sample_projection["dense_log_mass"],
        sample_projection["mass_band_low"],
        sample_projection["mass_band_high"],
        color="#d95f02",
        alpha=0.22,
        linewidth=0.0,
        label=r"Model $1\sigma$",
    )
    ax_mass.plot(
        sample_projection["dense_log_mass"],
        sample_projection["mass_median"],
        color="#d95f02",
        linewidth=2.0,
        label="Model median",
    )
    ax_mass.set_xlabel(r"$\log_{10}(M_{\rm ini}/{\rm M}_\odot)$")
    ax_mass.set_ylabel("Counts per bin")
    ax_mass.set_title("Mass projection")
    ax_mass.text(0.03, 0.95, "(a)", transform=ax_mass.transAxes, ha="left", va="top")
    ax_mass.legend(frameon=False, fontsize=8.5)

    ax_centers_kpc = 10.0 ** (
        0.5 * (projection_bundle["log_a_bin_edges"][:-1] + projection_bundle["log_a_bin_edges"][1:])
    )
    ax_radius.errorbar(
        ax_centers_kpc,
        projection_bundle["observed_a_counts"],
        yerr=projection_bundle["a_yerr"],
        fmt="o",
        color="black",
        ms=4.0,
        capsize=2.5,
        label="Observed",
    )
    ax_radius.fill_between(
        sample_projection["dense_a_kpc"],
        sample_projection["a_band_low"],
        sample_projection["a_band_high"],
        color="#1b9e77",
        alpha=0.22,
        linewidth=0.0,
        label=r"Model $1\sigma$",
    )
    ax_radius.plot(
        sample_projection["dense_a_kpc"],
        sample_projection["a_median"],
        color="#1b9e77",
        linewidth=2.0,
        label="Model median",
    )
    ax_radius.set_xscale("log")
    ax_radius.set_xlabel(r"Orbital semimajor axis $a$ [kpc]")
    ax_radius.set_ylabel("Counts per bin")
    ax_radius.set_title("Radius projection")
    ax_radius.text(0.03, 0.95, "(b)", transform=ax_radius.transAxes, ha="left", va="top")
    ax_radius.legend(frameon=False, fontsize=8.5)

    image = ax_residual.pcolormesh(
        10.0 ** projection_bundle["log_a_bin_edges"],
        projection_bundle["mass_bin_edges"],
        projection_bundle["residual_significance"],
        cmap="coolwarm",
        vmin=-3.0,
        vmax=3.0,
        shading="auto",
        rasterized=True,
    )
    ax_residual.set_xscale("log")
    ax_residual.set_xlabel(r"Orbital semimajor axis $a$ [kpc]")
    ax_residual.set_ylabel(r"$\log_{10}(M_{\rm ini}/{\rm M}_\odot)$")
    ax_residual.set_title("Residual significance")
    ax_residual.text(0.03, 0.95, "(c)", transform=ax_residual.transAxes, ha="left", va="top")
    colorbar = fig.colorbar(image, ax=ax_residual, pad=0.01)
    colorbar.set_label(r"$(N_{\rm obs}-N_{\rm exp})/\sqrt{N_{\rm exp}}$")

    fig.savefig(output_path)
    plt.close(fig)


def _build_longitude_split_results_from_exact_best(
    best_result: dict[str, object],
    threshold_deg: float = 30.0,
) -> list[tuple[str, str, dict[str, object]]]:
    full_observable_context = best_result["observable_context"]
    full_predicted_complete_counts = np.asarray(best_result["final_predicted_complete_counts"], dtype=float)
    full_raw_params = np.asarray(best_result["final_completeness_raw_parameters"], dtype=float)
    full_start_params = full_raw_params[:4]
    longitude_centers = np.asarray(full_observable_context.abs_longitude_centers_deg, dtype=float)

    subset_definitions = [
        (rf"$|l| < {threshold_deg:.0f}^\circ$", "lower_abs_longitude", longitude_centers < threshold_deg),
        (rf"$|l| \geq {threshold_deg:.0f}^\circ$", "higher_abs_longitude", longitude_centers >= threshold_deg),
    ]
    longitude_results: list[tuple[str, str, dict[str, object]]] = []

    for display_label, subset_key, subset_mask in subset_definitions:
        observed_counts = np.asarray(full_observable_context.observed_counts[..., subset_mask], dtype=float).sum(axis=3)
        predicted_complete_counts = np.asarray(full_predicted_complete_counts[..., subset_mask], dtype=float).sum(axis=3)
        sky_probabilities = np.asarray(
            full_observable_context.sky_bin_probabilities_by_a[..., subset_mask],
            dtype=float,
        ).sum(axis=3)
        collapsed_context = ObservablePredictionContext(
            present_mass_proxy=full_observable_context.present_mass_proxy,
            log_present_mass_edges=np.asarray(full_observable_context.log_present_mass_edges, dtype=float),
            distance_edges_kpc=np.asarray(full_observable_context.distance_edges_kpc, dtype=float),
            abs_latitude_edges_deg=np.asarray(full_observable_context.abs_latitude_edges_deg, dtype=float),
            log_present_mass_centers=np.asarray(full_observable_context.log_present_mass_centers, dtype=float),
            log_distance_centers=np.asarray(full_observable_context.log_distance_centers, dtype=float),
            abs_latitude_centers_deg=np.asarray(full_observable_context.abs_latitude_centers_deg, dtype=float),
            observed_counts=observed_counts,
            mass_bin_probabilities_grid=np.asarray(full_observable_context.mass_bin_probabilities_grid, dtype=float),
            sky_bin_probabilities_by_a=sky_probabilities,
            log_present_mass_mean_grid=np.asarray(full_observable_context.log_present_mass_mean_grid, dtype=float),
            log_present_mass_feature_mean=float(full_observable_context.log_present_mass_feature_mean),
            log_present_mass_feature_std=float(full_observable_context.log_present_mass_feature_std),
            log_distance_feature_mean=float(full_observable_context.log_distance_feature_mean),
            log_distance_feature_std=float(full_observable_context.log_distance_feature_std),
            abs_latitude_feature_mean=float(full_observable_context.abs_latitude_feature_mean),
            abs_latitude_feature_std=float(full_observable_context.abs_latitude_feature_std),
            sun_galactocentric_radius_kpc=float(full_observable_context.sun_galactocentric_radius_kpc),
            n_geometry_samples=int(full_observable_context.n_geometry_samples),
        )
        completeness_fit = fit_logistic_completeness_model(
            observable_context=collapsed_context,
            predicted_complete_counts=predicted_complete_counts,
            start_params=full_start_params,
        )
        completeness_grid_table = build_completeness_grid_table(
            collapsed_context,
            completeness_fit["completeness_bin_grid"],
        )
        longitude_results.append(
            (
                display_label,
                subset_key,
                {"completeness_grid_table": completeness_grid_table},
            )
        )
    return longitude_results


def plot_schechter_profile_only_for_paper(
    refined_grid: pd.DataFrame,
    output_path: Path,
) -> None:
    profiled = (
        refined_grid.groupby(["input_alpha_dndm", "input_log10_m_c_msun"], as_index=False)["log_likelihood"]
        .max()
        .sort_values(["input_log10_m_c_msun", "input_alpha_dndm"])
    )
    alpha_grid = np.sort(profiled["input_alpha_dndm"].unique())
    logmc_grid = np.sort(profiled["input_log10_m_c_msun"].unique())
    surface = (
        profiled.pivot(index="input_log10_m_c_msun", columns="input_alpha_dndm", values="log_likelihood")
        .sort_index()
        .sort_index(axis=1)
        .to_numpy(dtype=float)
    )
    delta_logl = surface - float(np.nanmax(surface))
    floor = -6.0
    delta_logl = np.maximum(delta_logl, floor)
    levels = np.array(
        [-6.0, -5.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.7, -0.5, -0.3, -0.2, -0.1, -0.05, 0.0],
        dtype=float,
    )
    best_row = profiled.sort_values("log_likelihood", ascending=False).iloc[0]

    fig, ax = plt.subplots(figsize=(5.5, 4.1), constrained_layout=True)
    contour = ax.contourf(
        alpha_grid,
        logmc_grid,
        delta_logl,
        levels=levels,
        cmap="magma",
    )
    ax.plot(
        float(best_row["input_alpha_dndm"]),
        float(best_row["input_log10_m_c_msun"]),
        marker="x",
        color="white",
        markersize=8,
        markeredgewidth=1.7,
    )
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\log_{10}(M_c/{\rm M}_\odot)$")
    ax.set_title(r"Profile likelihood for the Schechter IMF")
    colorbar = fig.colorbar(contour, ax=ax, pad=0.01)
    colorbar.set_label(r"$\Delta \log \mathcal{L}$ from best fit")
    fig.savefig(output_path)
    plt.close(fig)


def plot_schechter_imf_pdf_only(
    posterior_samples: pd.DataFrame,
    output_path: Path,
    threshold_log_mass: float = 4.0,
    log_mass_min: float = 4.0,
    log_mass_max: float = 7.3,
    n_grid: int = 500,
) -> None:
    samples = posterior_samples.loc[
        :,
        ["input_alpha_dndm", "input_log10_m_c_msun", "final_total_initial_count_above_log10_4"],
    ].dropna()
    log_mass = np.linspace(log_mass_min, log_mass_max, n_grid)

    def schechter_dndlogm(alpha: float, log10_m_c_msun: float) -> np.ndarray:
        mass = np.power(10.0, log_mass)
        m_c = np.power(10.0, log10_m_c_msun)
        return np.power(mass, alpha + 1.0) * np.exp(-mass / m_c)

    curves = []
    mask = log_mass >= threshold_log_mass
    for row in samples.itertuples(index=False):
        shape = schechter_dndlogm(float(row.input_alpha_dndm), float(row.input_log10_m_c_msun))
        integral = float(np.trapezoid(shape[mask], log_mass[mask]))
        if not np.isfinite(integral) or integral <= 0.0:
            continue
        curves.append(shape * (float(row.final_total_initial_count_above_log10_4) / integral))
    curve_array = np.asarray(curves, dtype=float)
    q02, q16, q50, q84, q98 = np.quantile(curve_array, [0.025, 0.16, 0.5, 0.84, 0.975], axis=0)

    fig, ax = plt.subplots(figsize=(5.6, 4.0), constrained_layout=True)
    ax.fill_between(log_mass, q02, q98, color="#fdd0a2", alpha=0.35, linewidth=0.0, label=r"95\% posterior band")
    ax.fill_between(log_mass, q16, q84, color="#f16913", alpha=0.25, linewidth=0.0, label=r"68\% posterior band")
    ax.plot(log_mass, q50, color="#b30000", lw=2.0, label="Posterior median")
    ax.set_yscale("log")
    ax.set_xlim(log_mass_min, log_mass_max)
    ax.set_xlabel(r"$\log_{10}(M_{\rm ini}/{\rm M}_\odot)$")
    ax.set_ylabel(r"$dN/d\log_{10} M_{\rm ini}$")
    ax.set_title("Inferred Schechter IMF")
    ax.grid(alpha=0.15, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8.5)
    fig.savefig(output_path)
    plt.close(fig)


def _compute_radial_projection_counts(best_result: dict[str, object], log_a_bin_edges: np.ndarray) -> np.ndarray:
    context = best_result["final_context"]
    point_intensity_grid = compute_observed_intensity_grid(
        np.asarray(best_result["final_payload"]["model"]["imf_density_grid"]),
        np.asarray(best_result["final_payload"]["model"]["radial_density_grid"]),
        np.asarray(context.selection_probability_grid),
        float(best_result["final_payload"]["model"]["total_initial_count"]),
    )
    log_mass_edges = centers_to_edges(np.asarray(context.log_mass_grid, dtype=float))
    log_a_grid = np.asarray(context.log_a_grid, dtype=float)
    cell_weights = point_intensity_grid * np.diff(log_mass_edges)[:, None] * np.diff(centers_to_edges(log_a_grid))[None, :]
    expected_2d = rebin_expected_counts_2d(
        cell_weights,
        log_mass_grid=np.asarray(context.log_mass_grid, dtype=float),
        log_a_grid=log_a_grid,
        mass_bin_edges=np.linspace(log_mass_edges[0], log_mass_edges[-1], 13),
        log_a_bin_edges=log_a_bin_edges,
    )
    return np.sum(expected_2d, axis=0)


def _compute_intrinsic_radial_profile(
    bundle: ExactRunBundle,
) -> tuple[np.ndarray, np.ndarray]:
    context = bundle.best_result["final_context"]
    radius_grid = np.power(10.0, np.asarray(context.log_a_grid, dtype=float))
    radial_density = np.asarray(
        bundle.best_result["final_payload"]["model"]["radial_density_grid"],
        dtype=float,
    )
    n_above = float(bundle.best_summary["final_total_initial_count_above_log10_4"])
    return radius_grid, n_above * radial_density


def plot_radial_model_comparison_for_paper(
    *,
    catalog: pd.DataFrame,
    logpoly3_bundle: ExactRunBundle,
    step5_bundle: ExactRunBundle,
    powerlaw_bundle: ExactRunBundle,
    cored_powerlaw_bundle: ExactRunBundle,
    output_path: Path,
) -> None:
    context = logpoly3_bundle.best_result["final_context"]
    log_a_grid = np.asarray(context.log_a_grid, dtype=float)
    log_a_edges = centers_to_edges(log_a_grid)
    log_a_bin_edges = np.linspace(log_a_edges[0], log_a_edges[-1], 10)
    observed_counts, _ = np.histogram(context.log_a_data, bins=log_a_bin_edges)
    observed_yerr = poisson_count_errors(observed_counts)
    a_centers_kpc = np.power(10.0, 0.5 * (log_a_bin_edges[:-1] + log_a_bin_edges[1:]))

    predictions = {
        "logpoly3": _compute_radial_projection_counts(logpoly3_bundle.best_result, log_a_bin_edges),
        "step5": _compute_radial_projection_counts(step5_bundle.best_result, log_a_bin_edges),
        "powerlaw_a": _compute_radial_projection_counts(powerlaw_bundle.best_result, log_a_bin_edges),
        "cored_powerlaw_a": _compute_radial_projection_counts(cored_powerlaw_bundle.best_result, log_a_bin_edges),
    }
    legend_meta = {
        "logpoly3": (
            "logpoly3",
            "#111111",
            float(logpoly3_bundle.best_summary["log_likelihood"]),
        ),
        "step5": (
            "step5",
            "#d95f02",
            float(step5_bundle.best_summary["log_likelihood"]),
        ),
        "powerlaw_a": (
            "power-law $A(a)$",
            "#1f78b4",
            float(powerlaw_bundle.best_summary["log_likelihood"]),
        ),
        "cored_powerlaw_a": (
            "cored power-law $A(a)$",
            "#7570b3",
            float(cored_powerlaw_bundle.best_summary["log_likelihood"]),
        ),
    }
    intrinsic_profiles = {
        "logpoly3": _compute_intrinsic_radial_profile(logpoly3_bundle),
        "step5": _compute_intrinsic_radial_profile(step5_bundle),
        "powerlaw_a": _compute_intrinsic_radial_profile(powerlaw_bundle),
        "cored_powerlaw_a": _compute_intrinsic_radial_profile(cored_powerlaw_bundle),
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), constrained_layout=True)
    profile_ax, counts_ax = axes
    for key in ["logpoly3", "step5", "powerlaw_a", "cored_powerlaw_a"]:
        label, color, _ = legend_meta[key]
        radius_grid, profile = intrinsic_profiles[key]
        profile_ax.plot(
            radius_grid,
            profile,
            color=color,
            linewidth=2.0,
            drawstyle="steps-mid" if key == "step5" else "default",
            label=label,
        )
    profile_ax.set_xscale("log")
    profile_ax.set_yscale("log")
    profile_ax.set_xlabel(r"Orbital semimajor axis $a$ [kpc]")
    profile_ax.set_ylabel(r"$dN_0(>10^4\,{\rm M}_\odot)/d\log_{10}a$")
    profile_ax.set_title("Intrinsic radial profiles")
    profile_ax.legend(frameon=False, fontsize=8.0, loc="lower left")
    profile_ax.grid(alpha=0.18, linewidth=0.6)

    counts_ax.errorbar(
        a_centers_kpc,
        observed_counts,
        yerr=observed_yerr,
        fmt="o",
        color="black",
        ms=4.0,
        capsize=2.5,
        label="Observed",
    )
    for key in ["logpoly3", "step5", "powerlaw_a", "cored_powerlaw_a"]:
        _, color, _ = legend_meta[key]
        counts_ax.plot(
            a_centers_kpc,
            predictions[key],
            color=color,
            linewidth=2.0,
            marker="o",
            ms=3.0,
        )
    counts_ax.set_xscale("log")
    counts_ax.set_xlabel(r"Orbital semimajor axis $a$ [kpc]")
    counts_ax.set_ylabel("Detected counts per bin")
    counts_ax.set_title("Observed-space comparison")
    counts_ax.legend(frameon=False, fontsize=7.6)
    counts_ax.grid(alpha=0.18, linewidth=0.6)
    fig.savefig(output_path)
    plt.close(fig)


def _radial_weighted_mass_curve(
    *,
    surface_grid: np.ndarray,
    radial_density_grid: np.ndarray,
    log_a_grid: np.ndarray,
) -> np.ndarray:
    numerator = np.trapezoid(
        np.asarray(surface_grid, dtype=float) * np.asarray(radial_density_grid, dtype=float)[None, :],
        np.asarray(log_a_grid, dtype=float),
        axis=1,
    )
    denominator = np.trapezoid(np.asarray(radial_density_grid, dtype=float), np.asarray(log_a_grid, dtype=float))
    return numerator / max(float(denominator), 1.0e-12)


def _survivor_weighted_q_mass_curve(
    *,
    q_grid: np.ndarray,
    survival_grid: np.ndarray,
    radial_density_grid: np.ndarray,
    log_a_grid: np.ndarray,
) -> np.ndarray:
    weights = np.asarray(survival_grid, dtype=float) * np.asarray(radial_density_grid, dtype=float)[None, :]
    numerator = np.trapezoid(weights * np.asarray(q_grid, dtype=float), np.asarray(log_a_grid, dtype=float), axis=1)
    denominator = np.trapezoid(weights, np.asarray(log_a_grid, dtype=float), axis=1)
    return numerator / np.clip(denominator, 1.0e-12, None)


def _selection_surface_curve_bands(
    *,
    bundle: ExactRunBundle,
    radial_density_grid: np.ndarray,
    reference_log_mass_grid: np.ndarray,
    reference_log_a_grid: np.ndarray,
) -> dict[str, np.ndarray] | None:
    worker_dir = PROJECT_ROOT / "variants" / bundle.variant_name / "outputs" / "parallel_exact_mcmc_workers"
    archive_paths = sorted(worker_dir.glob("chain_*_selection_surfaces.npz"))
    if not archive_paths:
        return None

    radial_density_grid = np.asarray(radial_density_grid, dtype=float)
    reference_log_mass_grid = np.asarray(reference_log_mass_grid, dtype=float)
    reference_log_a_grid = np.asarray(reference_log_a_grid, dtype=float)
    survival_curves = []
    q_curves = []
    radial_denominator = np.trapezoid(radial_density_grid, reference_log_a_grid)
    radial_denominator = max(float(radial_denominator), 1.0e-12)

    for path in archive_paths:
        with np.load(path) as archive:
            log_mass_grid = np.asarray(archive["log_mass_grid"], dtype=float)
            log_a_grid = np.asarray(archive["log_a_grid"], dtype=float)
            if not (np.allclose(log_mass_grid, reference_log_mass_grid) and np.allclose(log_a_grid, reference_log_a_grid)):
                continue
            survival = np.asarray(archive["survival_probability"], dtype=float)
            q_grid = np.asarray(archive["effective_detectability"], dtype=float)
        radial_weights = radial_density_grid[None, None, :]
        survival_weighted = survival * radial_weights
        survival_curve = np.trapezoid(survival_weighted, reference_log_a_grid, axis=2) / radial_denominator
        q_numerator = np.trapezoid(survival_weighted * q_grid, reference_log_a_grid, axis=2)
        q_denominator = np.trapezoid(survival_weighted, reference_log_a_grid, axis=2)
        q_curve = q_numerator / np.clip(q_denominator, 1.0e-12, None)
        survival_curves.append(survival_curve)
        q_curves.append(q_curve)

    if not survival_curves:
        return None
    survival_array = np.vstack(survival_curves)
    q_array = np.vstack(q_curves)
    s16, s50, s84 = np.quantile(survival_array, [0.16, 0.50, 0.84], axis=0)
    q16, q50, q84 = np.quantile(q_array, [0.16, 0.50, 0.84], axis=0)
    return {
        "survival_q16": s16,
        "survival_q50": s50,
        "survival_q84": s84,
        "detectability_q16": q16,
        "detectability_q50": q50,
        "detectability_q84": q84,
    }


def plot_survivability_detectability_mass_profiles_for_paper(
    *,
    catalog: pd.DataFrame,
    bundle: ExactRunBundle,
    output_path: Path,
) -> None:
    best_result = bundle.best_result
    context = best_result["final_context"]
    log_mass_grid = np.asarray(context.log_mass_grid, dtype=float)
    log_a_grid = np.asarray(context.log_a_grid, dtype=float)
    radial_density_grid = np.asarray(best_result["final_payload"]["model"]["radial_density_grid"], dtype=float)

    best_survival_curve = _radial_weighted_mass_curve(
        surface_grid=np.asarray(context.survival_probability_grid, dtype=float),
        radial_density_grid=radial_density_grid,
        log_a_grid=log_a_grid,
    )
    best_q_curve = _survivor_weighted_q_mass_curve(
        q_grid=np.asarray(best_result["final_effective_completeness_grid"], dtype=float),
        survival_grid=np.asarray(context.survival_probability_grid, dtype=float),
        radial_density_grid=radial_density_grid,
        log_a_grid=log_a_grid,
    )
    bands = _selection_surface_curve_bands(
        bundle=bundle,
        radial_density_grid=radial_density_grid,
        reference_log_mass_grid=log_mass_grid,
        reference_log_a_grid=log_a_grid,
    )

    observed_log_mass = np.asarray(context.log_mass_data, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True, constrained_layout=True)

    ax = axes[0]
    if bands is not None:
        ax.fill_between(
            log_mass_grid,
            bands["survival_q16"],
            bands["survival_q84"],
            color="#9ecae1",
            alpha=0.35,
            linewidth=0.0,
        )
    ax.plot(log_mass_grid, best_survival_curve, color="#08519c", linewidth=2.0)
    ax.set_ylabel(r"$\langle S\rangle_{\rho(a)}$")
    ax.set_title("Survivability")

    ax = axes[1]
    if bands is not None:
        ax.fill_between(
            log_mass_grid,
            bands["detectability_q16"],
            bands["detectability_q84"],
            color="#fdd0a2",
            alpha=0.45,
            linewidth=0.0,
        )
    ax.plot(log_mass_grid, best_q_curve, color="#7f2704", linewidth=2.0)
    ax.axhline(
        float(_posterior_row(bundle.posterior_summary, "mean_detectability_above_log10_4")["q50"]),
        color="#d94801",
        linewidth=1.0,
        linestyle=":",
    )
    ax.set_ylabel(r"$\langle Q\rangle_{\rho(a)S}$")
    ax.set_title("Detectability of survivors")

    for ax in axes:
        ax.set_xlim(4.0, 7.25)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(r"$\log_{10}(M_{\rm ini}/{\rm M}_\odot)$")
        ax.grid(alpha=0.18, linewidth=0.5)
        rug_y = np.full_like(observed_log_mass, 0.015)
        ax.plot(observed_log_mass, rug_y, "|", color="0.25", markersize=3.5, alpha=0.35)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_conditional_observable_approximation_for_paper(
    *,
    catalog: pd.DataFrame,
    base_context,
    observable_context,
    output_path: Path,
) -> None:
    log_initial_mass_data = catalog["log_initial_mass_msun"].to_numpy(dtype=float)
    radius_data = catalog["semi_major_axis_kpc"].to_numpy(dtype=float)
    log_present_mass_data = np.log10(catalog["present_mass_msun"].to_numpy(dtype=float))
    observed_log_loss = log_initial_mass_data - log_present_mass_data

    log_mass_grid = np.asarray(base_context.log_mass_grid, dtype=float)
    log_a_grid = np.asarray(base_context.log_a_grid, dtype=float)
    radius_grid = np.power(10.0, log_a_grid)
    present_mass_proxy = getattr(observable_context, "present_mass_proxy", None)
    if getattr(present_mass_proxy, "model_kind", "") == "monotonic_mass_loss":
        z_mass = (log_mass_grid[:, None] - float(present_mass_proxy.log_mass_mean)) / float(present_mass_proxy.log_mass_std)
        z_a = (log_a_grid[None, :] - float(present_mass_proxy.log_a_mean)) / float(present_mass_proxy.log_a_std)
        b0, b1, b2, s0, s1 = np.asarray(present_mass_proxy.coefficients, dtype=float)
        radial = b0 + b1 * z_a + b2 * np.square(z_a)
        mass_slope = np.logaddexp(0.0, s0 + s1 * z_a)
        model_log_loss = np.logaddexp(0.0, radial - mass_slope * z_mass)
    else:
        model_log_present_mass = np.asarray(observable_context.log_present_mass_mean_grid, dtype=float)
        model_log_loss = log_mass_grid[:, None] - model_log_present_mass

    model_log_loss_at_data = np.empty_like(observed_log_loss)
    for index, (log_mass, radius) in enumerate(zip(log_initial_mass_data, radius_data, strict=True)):
        i_mass = int(np.argmin(np.abs(log_mass_grid - log_mass)))
        i_a = int(np.argmin(np.abs(radius_grid - radius)))
        model_log_loss_at_data[index] = model_log_loss[i_mass, i_a]
    residual = observed_log_loss - model_log_loss_at_data

    radius_edges = np.power(10.0, centers_to_edges(log_a_grid))
    log_mass_edges = centers_to_edges(log_mass_grid)
    radius_limits = (float(np.nanmin(radius_data) / 1.15), float(np.nanmax(radius_data) * 1.15))
    log_mass_limits = (float(np.nanmin(log_initial_mass_data) - 0.12), float(np.nanmax(log_initial_mass_data) + 0.12))

    fig, axes = plt.subplots(ncols=3, figsize=(10.2, 3.4), constrained_layout=True, sharex=True, sharey=True)

    observed_image = axes[0].scatter(
        radius_data,
        log_initial_mass_data,
        c=observed_log_loss,
        s=18,
        cmap="viridis",
        vmin=0.0,
        vmax=1.8,
        alpha=0.78,
        linewidths=0.0,
    )
    axes[0].set_title("Observed mass loss", fontsize=9.5)
    axes[0].text(0.04, 0.95, "(a)", transform=axes[0].transAxes, ha="left", va="top")
    cbar = fig.colorbar(observed_image, ax=axes[0], pad=0.01)
    cbar.set_label(r"$\log_{10}(M_{\rm ini}/M_{\rm now})$")

    model_image = axes[1].pcolormesh(
        radius_edges,
        log_mass_edges,
        model_log_loss,
        cmap="viridis",
        vmin=0.0,
        vmax=1.8,
        shading="auto",
    )
    axes[1].scatter(radius_data, log_initial_mass_data, s=5, color="black", alpha=0.22, linewidths=0.0)
    axes[1].set_title("Proxy model", fontsize=9.5)
    axes[1].text(0.04, 0.95, "(b)", transform=axes[1].transAxes, ha="left", va="top")
    cbar = fig.colorbar(model_image, ax=axes[1], pad=0.01)
    cbar.set_label(r"$\log_{10}(M_{\rm ini}/M_{\rm now})$")

    residual_image = axes[2].scatter(
        radius_data,
        log_initial_mass_data,
        c=residual,
        s=18,
        cmap="coolwarm",
        vmin=-0.75,
        vmax=0.75,
        alpha=0.78,
        linewidths=0.0,
    )
    axes[2].set_title("Observed - model", fontsize=9.5)
    axes[2].text(0.04, 0.95, "(c)", transform=axes[2].transAxes, ha="left", va="top")
    cbar = fig.colorbar(residual_image, ax=axes[2], pad=0.01)
    cbar.set_label(r"$\Delta\log_{10}M_{\rm now}$ residual")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(*radius_limits)
        ax.set_ylim(*log_mass_limits)
        ax.set_xlabel(r"Orbital semimajor axis $a$ [kpc]")
        ax.grid(alpha=0.14, linewidth=0.5)
    axes[0].set_ylabel(r"$\log_{10}(M_{\rm ini}/\mathrm{M_\odot})$")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _evaluate_abs_longitude_completeness(
    *,
    raw_params: np.ndarray,
    observable_context,
    log_present_mass: np.ndarray,
    log_distance: np.ndarray,
    abs_latitude_deg: np.ndarray,
    abs_longitude_deg: np.ndarray,
) -> np.ndarray:
    intercept = float(raw_params[0])
    mass_slope = float(np.exp(raw_params[1]))
    distance_slope = float(np.exp(raw_params[2]))
    latitude_slope = float(np.exp(raw_params[3]))
    longitude_slope = float(np.exp(raw_params[4]))
    z_mass = (np.asarray(log_present_mass, dtype=float) - float(observable_context.log_present_mass_feature_mean)) / float(
        observable_context.log_present_mass_feature_std
    )
    z_distance = (np.asarray(log_distance, dtype=float) - float(observable_context.log_distance_feature_mean)) / float(
        observable_context.log_distance_feature_std
    )
    z_latitude = (
        np.asarray(abs_latitude_deg, dtype=float) - float(observable_context.abs_latitude_feature_mean)
    ) / float(observable_context.abs_latitude_feature_std)
    z_longitude = (
        np.asarray(abs_longitude_deg, dtype=float) - float(observable_context.abs_longitude_feature_mean)
    ) / float(observable_context.abs_longitude_feature_std)
    logits = intercept + mass_slope * z_mass - distance_slope * z_distance + latitude_slope * z_latitude + longitude_slope * z_longitude
    return 1.0 / (1.0 + np.exp(-logits))


def plot_detectability_c_and_q_summary_for_paper(
    *,
    catalog: pd.DataFrame,
    base_context,
    observable_context,
    raw_params: np.ndarray,
    effective_completeness_grid: np.ndarray,
    output_path: Path,
) -> None:
    observed_log_present_mass = np.log10(catalog["present_mass_msun"].to_numpy(dtype=float))
    observed_log_distance = np.log10(catalog["r_sun_kpc"].to_numpy(dtype=float))
    observed_abs_latitude = np.abs(catalog["galactic_b_deg"].to_numpy(dtype=float))
    observed_abs_longitude = np.abs(((catalog["galactic_l_deg"].to_numpy(dtype=float) + 180.0) % 360.0) - 180.0)

    fixed_values = {
        "log_present_mass": float(np.median(observed_log_present_mass)),
        "log_distance": float(np.median(observed_log_distance)),
        "abs_latitude": float(np.median(observed_abs_latitude)),
        "abs_longitude": float(np.median(observed_abs_longitude)),
    }

    fig = plt.figure(figsize=(7.2, 5.2))
    grid = fig.add_gridspec(
        nrows=2,
        ncols=5,
        height_ratios=[0.95, 1.35],
        width_ratios=[1.0, 1.0, 1.0, 1.0, 0.06],
        hspace=0.42,
        wspace=0.45,
    )

    curve_specs = [
        (
            "log_present_mass",
            np.linspace(
                float(np.min(observed_log_present_mass)) - 0.08,
                float(np.max(observed_log_present_mass)) + 0.08,
                220,
            ),
            observed_log_present_mass,
            r"$\log_{10}(M_{\rm now}/\mathrm{M_\odot})$",
            False,
        ),
        (
            "log_distance",
            np.linspace(
                float(np.min(observed_log_distance)) - 0.05,
                float(np.max(observed_log_distance)) + 0.05,
                220,
            ),
            observed_log_distance,
            r"$D_\odot$ [kpc]",
            True,
        ),
        (
            "abs_latitude",
            np.linspace(0.0, 30.0, 220),
            observed_abs_latitude,
            r"$|b|$ [deg]",
            False,
        ),
        (
            "abs_longitude",
            np.linspace(0.0, 60.0, 220),
            observed_abs_longitude,
            r"$|l|$ [deg]",
            False,
        ),
    ]

    for index, (variable, x_values, observed_values, xlabel, use_log_x) in enumerate(curve_specs):
        ax = fig.add_subplot(grid[0, index])
        values = {
            "log_present_mass": np.full_like(x_values, fixed_values["log_present_mass"], dtype=float),
            "log_distance": np.full_like(x_values, fixed_values["log_distance"], dtype=float),
            "abs_latitude": np.full_like(x_values, fixed_values["abs_latitude"], dtype=float),
            "abs_longitude": np.full_like(x_values, fixed_values["abs_longitude"], dtype=float),
        }
        values[variable] = x_values
        completeness = _evaluate_abs_longitude_completeness(
            raw_params=np.asarray(raw_params, dtype=float),
            observable_context=observable_context,
            log_present_mass=values["log_present_mass"],
            log_distance=values["log_distance"],
            abs_latitude_deg=values["abs_latitude"],
            abs_longitude_deg=values["abs_longitude"],
        )
        plot_x = np.power(10.0, x_values) if use_log_x else x_values
        observed_plot_x = np.power(10.0, observed_values) if use_log_x else observed_values
        ax.plot(plot_x, completeness, color="#111111", linewidth=1.8)
        if use_log_x:
            ax.set_xscale("log")
        if variable == "abs_latitude":
            ax.set_xlim(0.0, 30.0)
        elif variable == "abs_longitude":
            ax.set_xlim(0.0, 60.0)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xlabel(xlabel)
        if index == 0:
            ax.set_ylabel(r"$C$")
        ax.grid(alpha=0.18, linewidth=0.5)
        rug_y = np.full_like(observed_plot_x, 0.02, dtype=float)
        ax.plot(observed_plot_x, rug_y, "|", color="#555555", markersize=4.0, alpha=0.38)
        ax.text(0.04, 0.92, f"({chr(ord('a') + index)})", transform=ax.transAxes, ha="left", va="top")

    ax_q = fig.add_subplot(grid[1, :4])
    log_mass_grid = np.asarray(base_context.log_mass_grid, dtype=float)
    log_a_grid = np.asarray(base_context.log_a_grid, dtype=float)
    radius_grid = np.power(10.0, log_a_grid)
    radius_edges = np.power(10.0, centers_to_edges(log_a_grid))
    log_mass_edges = centers_to_edges(log_mass_grid)
    q_grid = np.asarray(effective_completeness_grid, dtype=float)
    mesh = ax_q.pcolormesh(
        radius_edges,
        log_mass_edges,
        q_grid,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        shading="auto",
        rasterized=True,
    )
    ax_q.scatter(
        catalog["semi_major_axis_kpc"].to_numpy(dtype=float),
        catalog["log_initial_mass_msun"].to_numpy(dtype=float),
        s=8,
        color="white",
        edgecolor="black",
        linewidth=0.25,
        alpha=0.75,
    )
    ax_q.set_xscale("log")
    ax_q.set_xlim(float(np.min(radius_grid)), float(np.max(radius_grid)))
    ax_q.set_ylim(float(np.min(log_mass_grid)), float(np.max(log_mass_grid)))
    ax_q.set_xlabel(r"Orbital semimajor axis $a$ [kpc]")
    ax_q.set_ylabel(r"$\log_{10}(M_{\rm ini}/\mathrm{M_\odot})$")
    ax_q.text(0.015, 0.95, "(e)", transform=ax_q.transAxes, ha="left", va="top", color="white")
    cax = fig.add_subplot(grid[1, 4])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label(r"$Q(\log M_{\rm ini},a)$")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _write_paper_numbers_tex(
    *,
    logpoly3_bundle: ExactRunBundle,
    step5_bundle: ExactRunBundle,
    powerlaw_bundle: ExactRunBundle,
    cored_powerlaw_bundle: ExactRunBundle,
    output_path: Path,
) -> None:
    best = logpoly3_bundle.best_summary
    posterior = logpoly3_bundle.posterior_summary
    baseline_n0 = float(best["baseline_total_initial_count_above_log10_4"])
    baseline_mass_e8 = float(best["baseline_total_initial_stellar_mass_above_log10_4_msun"]) / 1.0e8
    final_n0 = float(best["final_total_initial_count_above_log10_4"])
    final_mass_e8 = float(best["final_total_initial_stellar_mass_above_log10_4_msun"]) / 1.0e8
    eta_row = _posterior_row(posterior, "eta_t")
    alpha_row = _posterior_row(posterior, "input_alpha_dndm")
    mc_row = _posterior_row(posterior, "input_log10_m_c_msun")
    n0_row = _posterior_row(posterior, "final_total_initial_count_above_log10_4")
    mass_row = _posterior_row(posterior, "final_total_initial_stellar_mass_above_log10_4_msun")
    det_row = _posterior_row(posterior, "mean_detectability_above_log10_4")

    delta_aic_step5 = float(step5_bundle.best_summary["aic"]) - float(best["aic"])
    delta_bic_step5 = float(step5_bundle.best_summary["bic"]) - float(best["bic"])
    delta_aic_power = float(powerlaw_bundle.best_summary["aic"]) - float(best["aic"])
    delta_bic_power = float(powerlaw_bundle.best_summary["bic"]) - float(best["bic"])
    delta_aic_cored_power = float(cored_powerlaw_bundle.best_summary["aic"]) - float(best["aic"])
    delta_bic_cored_power = float(cored_powerlaw_bundle.best_summary["bic"]) - float(best["bic"])
    cored_gamma_row = _posterior_row(cored_powerlaw_bundle.posterior_summary, "gamma_linear_a")
    cored_log_core_row = _posterior_row(cored_powerlaw_bundle.posterior_summary, "log10_a_core_kpc")
    cored_core_q16 = 10.0 ** float(cored_log_core_row["q16"])
    cored_core_q50 = 10.0 ** float(cored_log_core_row["q50"])
    cored_core_q84 = 10.0 ** float(cored_log_core_row["q84"])

    lines = [
        "% Generated by scripts/build_paper_assets_exact_single_component.py.",
        "% Do not edit numerical values here by hand.",
        rf"\providecommand{{\ExactBestEta}}{{{best['eta_t']:.3f}}}",
        rf"\providecommand{{\ExactBestAlpha}}{{{best['alpha_dndm']:.3f}}}",
        rf"\providecommand{{\ExactBestMc}}{{{best['log10_m_c_msun']:.3f}}}",
        rf"\providecommand{{\ExactBestLogL}}{{{best['log_likelihood']:.2f}}}",
        rf"\providecommand{{\ExactBestNzero}}{{{final_n0:.1f}}}",
        rf"\providecommand{{\ExactBestMassZeroEight}}{{{final_mass_e8:.2f}}}",
        rf"\providecommand{{\ExactBestMeanDetectability}}{{{best['mean_detectability_above_log10_4']:.3f}}}",
        rf"\providecommand{{\ExactBaselineNzero}}{{{baseline_n0:.1f}}}",
        rf"\providecommand{{\ExactBaselineMassZeroEight}}{{{baseline_mass_e8:.2f}}}",
        rf"\providecommand{{\ExactCountRatio}}{{{final_n0 / baseline_n0:.2f}}}",
        rf"\providecommand{{\ExactMassRatio}}{{{final_mass_e8 / baseline_mass_e8:.2f}}}",
        rf"\providecommand{{\PosteriorEtaMed}}{{{eta_row['q50']:.3f}}}",
        rf"\providecommand{{\PosteriorEtaMinus}}{{{eta_row['minus']:.3f}}}",
        rf"\providecommand{{\PosteriorEtaPlus}}{{{eta_row['plus']:.3f}}}",
        rf"\providecommand{{\PosteriorAlphaMed}}{{{alpha_row['q50']:.3f}}}",
        rf"\providecommand{{\PosteriorAlphaMinus}}{{{alpha_row['minus']:.3f}}}",
        rf"\providecommand{{\PosteriorAlphaPlus}}{{{alpha_row['plus']:.3f}}}",
        rf"\providecommand{{\PosteriorMcMed}}{{{mc_row['q50']:.3f}}}",
        rf"\providecommand{{\PosteriorMcMinus}}{{{mc_row['minus']:.3f}}}",
        rf"\providecommand{{\PosteriorMcPlus}}{{{mc_row['plus']:.3f}}}",
        rf"\providecommand{{\PosteriorNzeroMed}}{{{n0_row['q50']:.0f}}}",
        rf"\providecommand{{\PosteriorNzeroMinus}}{{{n0_row['minus']:.0f}}}",
        rf"\providecommand{{\PosteriorNzeroPlus}}{{{n0_row['plus']:.0f}}}",
        rf"\providecommand{{\PosteriorMassZeroEightMed}}{{{mass_row['q50'] / 1.0e8:.2f}}}",
        rf"\providecommand{{\PosteriorMassZeroEightMinus}}{{{mass_row['minus'] / 1.0e8:.2f}}}",
        rf"\providecommand{{\PosteriorMassZeroEightPlus}}{{{mass_row['plus'] / 1.0e8:.2f}}}",
        rf"\providecommand{{\PosteriorMeanDetectabilityMed}}{{{det_row['q50']:.3f}}}",
        rf"\providecommand{{\PosteriorMeanDetectabilityMinus}}{{{det_row['minus']:.3f}}}",
        rf"\providecommand{{\PosteriorMeanDetectabilityPlus}}{{{det_row['plus']:.3f}}}",
        rf"\providecommand{{\StepFiveBestLogL}}{{{step5_bundle.best_summary['log_likelihood']:.2f}}}",
        rf"\providecommand{{\StepFiveDeltaAIC}}{{{delta_aic_step5:.2f}}}",
        rf"\providecommand{{\StepFiveDeltaBIC}}{{{delta_bic_step5:.2f}}}",
        rf"\providecommand{{\PowerLawABestLogL}}{{{powerlaw_bundle.best_summary['log_likelihood']:.2f}}}",
        rf"\providecommand{{\PowerLawADeltaAIC}}{{{delta_aic_power:.2f}}}",
        rf"\providecommand{{\PowerLawADeltaBIC}}{{{delta_bic_power:.2f}}}",
        rf"\providecommand{{\CoredPowerLawABestLogL}}{{{cored_powerlaw_bundle.best_summary['log_likelihood']:.2f}}}",
        rf"\providecommand{{\CoredPowerLawADeltaAIC}}{{{delta_aic_cored_power:.2f}}}",
        rf"\providecommand{{\CoredPowerLawADeltaBIC}}{{{delta_bic_cored_power:.2f}}}",
        rf"\providecommand{{\CoredPowerLawABestGamma}}{{{cored_powerlaw_bundle.best_summary['gamma_linear_a']:.3f}}}",
        rf"\providecommand{{\CoredPowerLawABestLogCore}}{{{cored_powerlaw_bundle.best_summary['log10_a_core_kpc']:.3f}}}",
        rf"\providecommand{{\CoredPowerLawABestCoreKpc}}{{{cored_powerlaw_bundle.best_summary['a_core_kpc']:.2f}}}",
        rf"\providecommand{{\CoredPowerLawAGammaMed}}{{{cored_gamma_row['q50']:.3f}}}",
        rf"\providecommand{{\CoredPowerLawAGammaMinus}}{{{cored_gamma_row['minus']:.3f}}}",
        rf"\providecommand{{\CoredPowerLawAGammaPlus}}{{{cored_gamma_row['plus']:.3f}}}",
        rf"\providecommand{{\CoredPowerLawACoreKpcMed}}{{{cored_core_q50:.2f}}}",
        rf"\providecommand{{\CoredPowerLawACoreKpcMinus}}{{{cored_core_q50 - cored_core_q16:.2f}}}",
        rf"\providecommand{{\CoredPowerLawACoreKpcPlus}}{{{cored_core_q84 - cored_core_q50:.2f}}}",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    paper_dir = PROJECT_ROOT / "paper"
    figures_dir = paper_dir / "figures"
    tables_dir = paper_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    logpoly3_bundle = _load_exact_bundle(LOGPOLY3_VARIANT)
    step5_bundle = _load_exact_bundle(STEP5_VARIANT)
    powerlaw_bundle = _load_exact_bundle(POWERLAW_A_VARIANT)
    cored_powerlaw_bundle = _load_exact_bundle(CORED_POWERLAW_A_VARIANT)

    exact_best = logpoly3_bundle.best_result
    uncertainty_payload = _build_uncertainty_payload(logpoly3_bundle)

    catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    fit_catalog = fit_catalog_models(catalog, PROJECT_ROOT)["catalog"]

    smooth_survivability_map = build_smooth_survivability_grid(
        fit_catalog,
        eta_t=1.0,
        surface_model="logistic",
    )
    eta_boundary_maps = {
        0.5: build_smooth_survivability_grid(fit_catalog, eta_t=0.5, surface_model="logistic"),
        2.0: build_smooth_survivability_grid(fit_catalog, eta_t=2.0, surface_model="logistic"),
    }
    plot_catalog_mass_semimajor_axis_overview_for_paper(
        fit_catalog,
        smooth_survivability_map,
        figures_dir / "catalog_mass_semimajor_axis_overview.pdf",
        eta_boundary_maps=eta_boundary_maps,
    )

    plot_detectability_c_and_q_summary_for_paper(
        catalog=fit_catalog,
        base_context=exact_best["base_context"],
        observable_context=exact_best["observable_context"],
        raw_params=np.asarray(exact_best["final_completeness_raw_parameters"], dtype=float),
        effective_completeness_grid=np.asarray(exact_best["final_effective_completeness_grid"], dtype=float),
        output_path=figures_dir / "detectability_em_maps.pdf",
    )
    convergence_cache_path = tables_dir / "detectability_em_convergence_illustrative_results.pkl"
    if convergence_cache_path.exists():
        with convergence_cache_path.open("rb") as handle:
            illustrative_iteration_results = pickle.load(handle)
    else:
        illustrative_iteration_results = []
        for eta_t, alpha, log_mc in [
            (0.8, -1.3, 6.3),
            (0.8, -1.0, 6.3),
            (0.8, -0.7, 6.3),
            (1.0, -1.3, 6.3),
            (1.0, -1.0, 6.3),
            (1.0, -0.7, 6.3),
            (1.13, -1.3, 6.3),
            (1.13, -1.0, 6.3),
            (1.13, -0.7, 6.3),
            (1.3, -1.3, 6.3),
            (1.3, -1.0, 6.3),
            (1.3, -0.7, 6.3),
        ]:
            illustrative_iteration_survivability_map = build_smooth_survivability_grid(
                fit_catalog,
                eta_t=eta_t,
                surface_model="logistic",
            )
            illustrative_iteration_result = fit_single_component_detectability_em_with_abs_longitude(
                fit_catalog,
                project_root=PROJECT_ROOT,
                spec=JointModelSpec(imf_family="schechter", radial_model="logpoly3"),
                n_iterations=30,
                fixed_imf_params=np.array([alpha, log_mc], dtype=float),
                survival_grid_override=illustrative_iteration_survivability_map,
            )
            illustrative_iteration_results.append(illustrative_iteration_result)
        with convergence_cache_path.open("wb") as handle:
            pickle.dump(illustrative_iteration_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    plot_detectability_em_convergence_for_paper(
        illustrative_iteration_results,
        figures_dir / "detectability_em_convergence.pdf",
    )
    plot_conditional_observable_approximation_for_paper(
        catalog=fit_catalog,
        base_context=exact_best["final_context"],
        observable_context=exact_best["observable_context"],
        output_path=figures_dir / "conditional_observable_approximation.pdf",
    )
    reference_row = logpoly3_bundle.refined_grid.loc[logpoly3_bundle.refined_grid["log_likelihood"].idxmax()].to_dict()
    _corner_plot(
        logpoly3_bundle.posterior_samples,
        reference_row,
        figures_dir / "single_component_posterior_corner.png",
    )
    _corner_plot(
        logpoly3_bundle.posterior_samples,
        reference_row,
        figures_dir / "single_component_posterior_corner.pdf",
    )

    projection_bundle = _build_projection_bundle(
        catalog=fit_catalog,
        context=exact_best["final_context"],
        best_payload=exact_best["final_payload"],
        uncertainty_payload=uncertainty_payload,
        n_projection_samples=250,
    )
    plot_single_component_intensity_plane_for_paper(
        projection_bundle,
        figures_dir / "single_component_intensity_plane.pdf",
    )
    plot_three_panel_summary_for_paper(
        projection_bundle,
        figures_dir / "best_single_component_summary.pdf",
    )
    plot_schechter_imf_pdf_only(
        logpoly3_bundle.posterior_samples,
        figures_dir / "single_component_profiles.pdf",
    )
    plot_survivability_detectability_mass_profiles_for_paper(
        catalog=fit_catalog,
        bundle=logpoly3_bundle,
        output_path=figures_dir / "survivability_detectability_mass_profiles.pdf",
    )
    plot_radial_model_comparison_for_paper(
        catalog=fit_catalog,
        logpoly3_bundle=logpoly3_bundle,
        step5_bundle=step5_bundle,
        powerlaw_bundle=powerlaw_bundle,
        cored_powerlaw_bundle=cored_powerlaw_bundle,
        output_path=figures_dir / "single_component_radial_profile.pdf",
    )

    _write_paper_numbers_tex(
        logpoly3_bundle=logpoly3_bundle,
        step5_bundle=step5_bundle,
        powerlaw_bundle=powerlaw_bundle,
        cored_powerlaw_bundle=cored_powerlaw_bundle,
        output_path=tables_dir / "paper_numbers.tex",
    )

    summary_payload = {
        "single_component_exact_global_logpoly3": logpoly3_bundle.best_summary,
        "single_component_exact_global_step5": step5_bundle.best_summary,
        "single_component_exact_powerlaw_a": powerlaw_bundle.best_summary,
        "single_component_exact_cored_powerlaw_a": cored_powerlaw_bundle.best_summary,
        "single_component_posterior_summary": logpoly3_bundle.posterior_summary.to_dict(orient="records"),
    }
    (tables_dir / "paper_results_summary.json").write_text(json.dumps(summary_payload, indent=2))

    manifest = {
        "figure1_overview": str(figures_dir / "catalog_mass_semimajor_axis_overview.pdf"),
        "figure_detectability_maps": str(figures_dir / "detectability_em_maps.pdf"),
        "figure_detectability_convergence": str(figures_dir / "detectability_em_convergence.pdf"),
        "figure_conditional_observables": str(figures_dir / "conditional_observable_approximation.pdf"),
        "figure_corner": str(figures_dir / "single_component_posterior_corner.pdf"),
        "figure_intensity_plane": str(figures_dir / "single_component_intensity_plane.pdf"),
        "figure_summary": str(figures_dir / "best_single_component_summary.pdf"),
        "figure_imf": str(figures_dir / "single_component_profiles.pdf"),
        "figure_survivability_detectability_mass_profiles": str(figures_dir / "survivability_detectability_mass_profiles.pdf"),
        "figure_radial_models": str(figures_dir / "single_component_radial_profile.pdf"),
        "paper_numbers": str(tables_dir / "paper_numbers.tex"),
        "paper_summary_json": str(tables_dir / "paper_results_summary.json"),
    }
    (tables_dir / "exact_single_component_paper_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
