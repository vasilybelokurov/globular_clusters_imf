from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib.pyplot as plt
import numpy as np

from compare_present_mass_proxy_models import (  # noqa: E402
    _fit_monotonic_loss_model,
    _load_fit_catalog,
    _predict_monotonic_loss_model,
    _standardized_features,
)
from globular_clusters_imf.plotting import centers_to_edges  # noqa: E402


def main() -> None:
    catalog = _load_fit_catalog()
    log_initial_mass_data = catalog["log_initial_mass_msun"].to_numpy(dtype=float)
    radius_data = catalog["semi_major_axis_kpc"].to_numpy(dtype=float)
    log_present_mass_data = np.log10(catalog["present_mass_msun"].to_numpy(dtype=float))
    observed_log_loss = log_initial_mass_data - log_present_mass_data

    variant_name = "profile_map_and_exact_mcmc_schechter_logpoly3_logistic_global"
    best_path = (
        PROJECT_ROOT
        / "variants"
        / variant_name
        / "outputs"
        / "tables"
        / "exact_parallel_mcmc_best_result.pkl"
    )
    with best_path.open("rb") as handle:
        exact_best = pickle.load(handle)
    base_context = exact_best["final_context"]

    features, _ = _standardized_features(catalog)
    z_mass_data = features[:, 0]
    z_a_data = features[:, 1]
    monotonic_params = _fit_monotonic_loss_model(
        z_mass_data,
        z_a_data,
        observed_log_loss,
    )

    log_mass_grid = np.asarray(base_context.log_mass_grid, dtype=float)
    log_a_grid = np.asarray(base_context.log_a_grid, dtype=float)
    radius_grid = np.power(10.0, log_a_grid)

    log_mass_mean = float(np.mean(log_initial_mass_data))
    log_mass_std = float(np.std(log_initial_mass_data))
    log_a_data = np.log10(radius_data)
    log_a_mean = float(np.mean(log_a_data))
    log_a_std = float(np.std(log_a_data))
    log_mass_std = log_mass_std if log_mass_std > 0.0 else 1.0
    log_a_std = log_a_std if log_a_std > 0.0 else 1.0

    z_mass_grid = (log_mass_grid[:, None] - log_mass_mean) / log_mass_std
    z_a_grid = (log_a_grid[None, :] - log_a_mean) / log_a_std
    model_log_loss = _predict_monotonic_loss_model(
        monotonic_params,
        z_mass_grid + np.zeros_like(z_a_grid),
        z_a_grid + np.zeros_like(z_mass_grid),
    )

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
    axes[1].set_title("Monotonic proxy model", fontsize=9.5)
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

    output_stem = PROJECT_ROOT / "paper" / "figures" / "conditional_observable_approximation_monotonic_oldfit"
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".png"), dpi=200)
    plt.close(fig)

    rms = float(np.sqrt(np.mean(np.square(residual))))
    print(output_stem.with_suffix(".pdf"))
    print(output_stem.with_suffix(".png"))
    print(f"monotonic residual RMS: {rms:.3f} dex")


if __name__ == "__main__":
    main()
