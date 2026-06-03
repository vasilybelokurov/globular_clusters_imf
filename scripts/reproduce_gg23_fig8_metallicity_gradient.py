from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from globular_clusters_imf.gg23_survivability import (  # noqa: E402
    GG23_MODELS,
    gg23_radius_dependent_mass_loss_parameters,
)


def main() -> None:
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog_with_origin_and_chemistry.csv"
    catalog = pd.read_csv(catalog_path)
    working = catalog.loc[
        np.isfinite(catalog["r_gc_kpc"]) & np.isfinite(catalog["local_feh"]) & (catalog["local_feh"] > -10.0)
    ].copy()
    working["log10_r_gc_kpc"] = np.log10(working["r_gc_kpc"].to_numpy(dtype=float))
    working["gg23_feh_approx"] = gg23_feh_approximation(working["r_gc_kpc"].to_numpy(dtype=float))

    bin_table = build_binned_metallicity_table(working)
    gradient_table = build_gradient_parameter_table()
    working.to_csv(tables_dir / "gg23_fig8_metallicity_gradient_points.csv", index=False)
    bin_table.to_csv(tables_dir / "gg23_fig8_metallicity_gradient_bins.csv", index=False)
    gradient_table.to_csv(tables_dir / "gg23_fig8_metallicity_gradient_parameters.csv", index=False)

    plot_path_pdf = figures_dir / "gg23_fig8_metallicity_gradient_reproduction.pdf"
    plot_path_png = figures_dir / "gg23_fig8_metallicity_gradient_reproduction.png"
    plot_metallicity_gradient(working, bin_table, gradient_table, plot_path_pdf, plot_path_png)
    print(f"Wrote {plot_path_pdf}")
    print(f"Wrote {plot_path_png}")
    print(f"Wrote {tables_dir / 'gg23_fig8_metallicity_gradient_parameters.csv'}")


def gg23_feh_approximation(radius_kpc: np.ndarray) -> np.ndarray:
    radius = np.asarray(radius_kpc, dtype=float)
    feh = np.full_like(radius, -1.5, dtype=float)
    mask = radius < 10.0
    feh[mask] = -0.5 - np.log10(radius[mask])
    return feh


def build_binned_metallicity_table(catalog: pd.DataFrame) -> pd.DataFrame:
    edges = np.array([-0.1, 0.25, 0.5, 0.75, 1.0, 1.35, 2.15], dtype=float)
    rows = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        subset = catalog.loc[(catalog["log10_r_gc_kpc"] >= left) & (catalog["log10_r_gc_kpc"] < right)]
        if len(subset) == 0:
            continue
        rows.append(
            {
                "log10_r_left": float(left),
                "log10_r_right": float(right),
                "n_clusters": int(len(subset)),
                "median_log10_r_gc_kpc": float(subset["log10_r_gc_kpc"].median()),
                "p16_log10_r_gc_kpc": float(subset["log10_r_gc_kpc"].quantile(0.16)),
                "p84_log10_r_gc_kpc": float(subset["log10_r_gc_kpc"].quantile(0.84)),
                "median_feh": float(subset["local_feh"].median()),
                "p16_feh": float(subset["local_feh"].quantile(0.16)),
                "p84_feh": float(subset["local_feh"].quantile(0.84)),
            }
        )
    return pd.DataFrame(rows)


def build_gradient_parameter_table() -> pd.DataFrame:
    radius = np.logspace(-0.2, 2.2, 240)
    model = GG23_MODELS["gg23_bh_feh_gradient"]
    mdot_ref, y = gg23_radius_dependent_mass_loss_parameters(radius, model)
    return pd.DataFrame(
        {
            "r_gc_kpc": radius,
            "log10_r_gc_kpc": np.log10(radius),
            "gg23_feh_approx": gg23_feh_approximation(radius),
            "gg23_y": y,
            "gg23_mdot_ref_msun_per_myr": mdot_ref,
            "gg23_abs_mdot_ref_msun_per_myr": np.abs(mdot_ref),
        }
    )


def plot_metallicity_gradient(
    catalog: pd.DataFrame,
    bin_table: pd.DataFrame,
    gradient_table: pd.DataFrame,
    pdf_path: Path,
    png_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), constrained_layout=True)

    ax = axes[0]
    ax.scatter(
        catalog["log10_r_gc_kpc"],
        catalog["local_feh"],
        s=10,
        color="0.45",
        alpha=0.6,
        linewidths=0.0,
    )
    xerr = np.vstack(
        [
            bin_table["median_log10_r_gc_kpc"] - bin_table["p16_log10_r_gc_kpc"],
            bin_table["p84_log10_r_gc_kpc"] - bin_table["median_log10_r_gc_kpc"],
        ]
    )
    yerr = np.vstack(
        [
            bin_table["median_feh"] - bin_table["p16_feh"],
            bin_table["p84_feh"] - bin_table["median_feh"],
        ]
    )
    ax.errorbar(
        bin_table["median_log10_r_gc_kpc"],
        bin_table["median_feh"],
        xerr=xerr,
        yerr=yerr,
        fmt="o",
        ms=4,
        color="black",
        ecolor="black",
        elinewidth=0.8,
        capsize=0,
    )
    ax.plot(
        gradient_table["log10_r_gc_kpc"],
        gradient_table["gg23_feh_approx"],
        color="black",
        linestyle="--",
        linewidth=1.2,
    )
    ax.set_xlim(-0.2, 2.2)
    ax.set_ylim(-2.5, 0.0)
    ax.set_xlabel(r"$\log_{10} R\ [{\rm kpc}]$")
    ax.set_ylabel("[Fe/H]")

    ax = axes[1]
    ax.plot(
        gradient_table["log10_r_gc_kpc"],
        gradient_table["gg23_y"],
        color="#1b9e77",
        linewidth=1.6,
        label=r"$y(R)$",
    )
    twin = ax.twinx()
    twin.plot(
        gradient_table["log10_r_gc_kpc"],
        gradient_table["gg23_abs_mdot_ref_msun_per_myr"],
        color="#d95f02",
        linewidth=1.6,
        label=r"$|\dot{M}_{\rm ref}(R)|$",
    )
    ax.set_xlim(-0.2, 2.2)
    ax.set_ylim(0.45, 1.45)
    twin.set_ylim(20.0, 50.0)
    ax.set_xlabel(r"$\log_{10} R\ [{\rm kpc}]$")
    ax.set_ylabel(r"$y(R)$", color="#1b9e77")
    twin.set_ylabel(r"$|\dot{M}_{\rm ref}(R)|\ [{\rm M}_\odot\,{\rm Myr}^{-1}]$", color="#d95f02")
    ax.tick_params(axis="y", labelcolor="#1b9e77")
    twin.tick_params(axis="y", labelcolor="#d95f02")
    ax.axvline(1.0, color="0.3", linestyle=":", linewidth=0.9)

    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
