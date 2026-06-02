from __future__ import annotations

import os
from dataclasses import replace
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
    GG23DisruptionModel,
    GG23_REFERENCE_MASS_MSUN,
    gg23_initial_mass_from_present_msun,
    gg23_present_mass_msun,
    gg23_survival_mass_cut_msun,
)
from globular_clusters_imf.model import AGE_GYR  # noqa: E402


MDOT_REF_ABS_MSUN_PER_MYR = 30.0
REFERENCE_EFFECTIVE_RADIUS_KPC = 1.0
MASS_GRID_MIN = 1.0e2
MASS_GRID_MAX = 1.0e7
PAPER_TABLE2 = {
    (2.0 / 3.0, 2.0 / 3.0): (0.55, 5.16, 4.99, 0.93),
    (2.0 / 3.0, 1.0): (0.70, 5.53, 5.48, 0.73),
    (2.0 / 3.0, 4.0 / 3.0): (0.80, 5.78, 5.78, 0.60),
    (2.0 / 3.0, 1.75): (0.88, 5.99, 6.03, 0.49),
    (2.0 / 3.0, 2.0): (0.92, 6.10, 6.14, 0.45),
    (1.0, 2.0 / 3.0): (0.91, 5.34, 5.13, 0.89),
    (1.0, 1.0): (1.00, 5.56, 5.49, 0.71),
    (1.0, 4.0 / 3.0): (1.05, 5.70, 5.71, 0.59),
    (1.0, 1.75): (1.08, 5.83, 5.89, 0.51),
    (1.0, 2.0): (1.10, 5.90, 5.97, 0.47),
}


def main() -> None:
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    table = build_table2_check()
    table_path = tables_dir / "gg23_fig6_table2_mass_loss_check.csv"
    table.to_csv(table_path, index=False)

    pdf_path = figures_dir / "gg23_fig6_mass_loss_reproduction.pdf"
    png_path = figures_dir / "gg23_fig6_mass_loss_reproduction.png"
    plot_fig6_style(table, pdf_path, png_path)

    print(f"Wrote {table_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


def model_for_xy(x: float, y: float) -> GG23DisruptionModel:
    return GG23DisruptionModel(
        name=f"gg23_fig6_x{x:.3f}_y{y:.3f}",
        label=fr"$x={x:.2f}, y={y:.2f}$",
        x=x,
        y=y,
        mdot_ref_msun_per_myr=-MDOT_REF_ABS_MSUN_PER_MYR,
    )


def build_table2_check() -> pd.DataFrame:
    rows = []
    for x, y in PAPER_TABLE2:
        model = model_for_xy(x, y)
        moments = evolved_powerlaw_mass_function_moments(model)
        paper_mto_ratio, paper_log_mto, paper_mu, paper_sigma = PAPER_TABLE2[(x, y)]
        row = {
            "x": x,
            "y": y,
            **moments,
            "paper_mto_over_mimin": paper_mto_ratio,
            "paper_log10_mto_msun": paper_log_mto,
            "paper_mu_log10_m": paper_mu,
            "paper_sigma_log10_m": paper_sigma,
        }
        row["delta_mto_over_mimin"] = row["mto_over_mimin"] - paper_mto_ratio
        row["delta_log10_mto_msun"] = row["log10_mto_msun"] - paper_log_mto
        row["delta_mu_log10_m"] = row["mu_log10_m"] - paper_mu
        row["delta_sigma_log10_m"] = row["sigma_log10_m"] - paper_sigma
        rows.append(row)
    return pd.DataFrame(rows)


def evolved_powerlaw_mass_function_moments(model: GG23DisruptionModel) -> dict[str, float]:
    mimin = gg23_survival_mass_cut_msun(
        REFERENCE_EFFECTIVE_RADIUS_KPC,
        model,
        age_gyr=AGE_GYR,
    ).item()
    initial_mass = np.logspace(np.log10(mimin * (1.0 + 1.0e-9)), np.log10(MASS_GRID_MAX), 200_000)
    present_mass = gg23_present_mass_msun(
        initial_mass,
        REFERENCE_EFFECTIVE_RADIUS_KPC,
        model,
        age_gyr=AGE_GYR,
    )
    positive = present_mass > 0.0
    initial_mass = initial_mass[positive]
    present_mass = present_mass[positive]

    dlnm_dlnmi = (
        1.0
        - float(model.x) / float(model.y)
        + float(model.x) / float(model.y) * np.power(present_mass / initial_mass, -float(model.y))
    )
    dnd_log_present_mass = np.power(initial_mass, -1.0) / dlnm_dlnmi
    mto_index = int(np.nanargmax(dnd_log_present_mass))

    above = present_mass >= 1.0e3
    log_present_mass = np.log10(present_mass[above])
    weights = dnd_log_present_mass[above]
    area = np.trapezoid(weights, x=log_present_mass)
    mu = np.trapezoid(log_present_mass * weights, x=log_present_mass) / area
    sigma = np.sqrt(np.trapezoid(np.square(log_present_mass - mu) * weights, x=log_present_mass) / area)

    return {
        "mimin_msun": float(mimin),
        "log10_mimin_msun": float(np.log10(mimin)),
        "mto_msun": float(present_mass[mto_index]),
        "mto_over_mimin": float(present_mass[mto_index] / mimin),
        "log10_mto_msun": float(np.log10(present_mass[mto_index])),
        "mu_log10_m": float(mu),
        "sigma_log10_m": float(sigma),
    }


def plot_fig6_style(table: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), constrained_layout=True)
    colors = {
        (2.0 / 3.0, 2.0 / 3.0): "#1b9e77",
        (2.0 / 3.0, 1.0): "#7570b3",
        (2.0 / 3.0, 4.0 / 3.0): "#d95f02",
        (1.0, 2.0 / 3.0): "#66a61e",
        (1.0, 1.0): "#e7298a",
        (1.0, 4.0 / 3.0): "#a6761d",
    }

    time_gyr = np.linspace(0.0, AGE_GYR, 500)
    initial_masses = [2.0e5, 5.0e5, 1.0e6]
    line_styles = {2.0e5: "-", 5.0e5: "--", 1.0e6: ":"}
    left_models = [
        (2.0 / 3.0, 2.0 / 3.0),
        (2.0 / 3.0, 4.0 / 3.0),
        (1.0, 1.0),
    ]
    for x, y in left_models:
        model = model_for_xy(x, y)
        for initial_mass in initial_masses:
            present = np.asarray(
                [
                    gg23_present_mass_msun(
                        initial_mass,
                        REFERENCE_EFFECTIVE_RADIUS_KPC,
                        model,
                        age_gyr=float(time_value),
                    ).item()
                    for time_value in time_gyr
                ],
                dtype=float,
            )
            axes[0].plot(
                time_gyr,
                present,
                color=colors[(x, y)],
                linestyle=line_styles[initial_mass],
                linewidth=1.4,
                label=fr"$x={x:.2f}, y={y:.2f}$" if initial_mass == initial_masses[0] else None,
            )
    axes[0].set_yscale("log")
    axes[0].set_xlim(0.0, AGE_GYR)
    axes[0].set_ylim(8.0e2, 1.2e6)
    axes[0].set_xlabel("time [Gyr]")
    axes[0].set_ylabel(r"$M(t)\ [{\rm M}_\odot]$")
    axes[0].legend(loc="lower left", fontsize=7, frameon=False)

    present_mass_grid = np.logspace(3.0, 7.4, 800)
    right_models = [
        (2.0 / 3.0, 2.0 / 3.0),
        (2.0 / 3.0, 1.0),
        (2.0 / 3.0, 4.0 / 3.0),
        (1.0, 2.0 / 3.0),
        (1.0, 1.0),
        (1.0, 4.0 / 3.0),
    ]
    for x, y in right_models:
        model = model_for_xy(x, y)
        initial_mass = gg23_initial_mass_from_present_msun(
            present_mass_grid,
            REFERENCE_EFFECTIVE_RADIUS_KPC,
            model,
            age_gyr=AGE_GYR,
        )
        forward_present = gg23_present_mass_msun(
            initial_mass,
            REFERENCE_EFFECTIVE_RADIUS_KPC,
            model,
            age_gyr=AGE_GYR,
        )
        dlnm_dlnmi = (
            1.0
            - float(model.x) / float(model.y)
            + float(model.x) / float(model.y) * np.power(forward_present / initial_mass, -float(model.y))
        )
        dnd_logm = np.power(initial_mass, -1.0) / dlnm_dlnmi
        dnd_logm /= np.nanmax(dnd_logm)
        axes[1].plot(
            present_mass_grid,
            dnd_logm,
            color=colors[(x, y)],
            linewidth=1.5,
            label=fr"$x={x:.2f}, y={y:.2f}$",
        )
    axes[1].set_xscale("log")
    axes[1].set_xlim(1.0e3, 2.5e7)
    axes[1].set_ylim(0.0, 1.08)
    axes[1].set_xlabel(r"$M\ [{\rm M}_\odot]$")
    axes[1].set_ylabel(r"normalized $dN/d\log M$")
    axes[1].legend(loc="upper right", fontsize=7, frameon=False)

    fig.suptitle("GG23 Fig. 6-style mass-loss check", fontsize=11)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
