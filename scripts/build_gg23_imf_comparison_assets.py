from __future__ import annotations

import os
from dataclasses import dataclass
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

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz


@dataclass(frozen=True)
class ImfRun:
    label: str
    tex_label: str
    variant: str
    color: str


RUNS = [
    ImfRun(
        label="Baumgardt 2019",
        tex_label=r"Baumgardt 2019",
        variant="profile_map_and_exact_mcmc_schechter_logpoly3_logistic_global_monotonic_q",
        color="#222222",
    ),
    ImfRun(
        label="GG23 no BHs",
        tex_label=r"GG23 no BHs",
        variant="gg23_schechter_no_bh_logpoly3_eta01_105",
        color="#1b9e77",
    ),
    ImfRun(
        label="GG23 BHs",
        tex_label=r"GG23 BHs",
        variant="gg23_schechter_bh_logpoly3",
        color="#d95f02",
    ),
    ImfRun(
        label="GG23 BHs + [Fe/H]",
        tex_label=r"GG23 BHs + [Fe/H]",
        variant="gg23_schechter_bh_feh_gradient_logpoly3",
        color="#7570b3",
    ),
    ImfRun(
        label="GG23 BHs + past tides",
        tex_label=r"GG23 BHs + past tides",
        variant="gg23_schechter_bh_past_tidal_logpoly3",
        color="#e7298a",
    ),
    ImfRun(
        label="GG23 BHs + [Fe/H] + past tides",
        tex_label=r"GG23 BHs + [Fe/H] + past tides",
        variant="gg23_schechter_bh_feh_gradient_past_tidal_logpoly3",
        color="#66a61e",
    ),
]


def schechter_dndlogm(log_mass: np.ndarray, alpha: float, log10_m_c_msun: float) -> np.ndarray:
    mass = np.power(10.0, log_mass)
    m_c = np.power(10.0, log10_m_c_msun)
    return np.power(mass, alpha + 1.0) * np.exp(-mass / m_c)


def normalize_above_threshold(
    log_mass: np.ndarray,
    shape: np.ndarray,
    total_count_above_threshold: float,
    threshold_log_mass: float,
) -> np.ndarray:
    mask = log_mass >= threshold_log_mass
    integral = float(np.trapezoid(shape[mask], log_mass[mask]))
    if not np.isfinite(integral) or integral <= 0.0:
        return np.zeros_like(shape)
    return shape * (total_count_above_threshold / integral)


def cumulative_from_differential(log_mass: np.ndarray, dndlogm: np.ndarray) -> np.ndarray:
    cumulative = np.zeros_like(dndlogm)
    for idx in range(len(log_mass)):
        cumulative[idx] = np.trapezoid(dndlogm[idx:], log_mass[idx:])
    return cumulative


def load_samples(run: ImfRun) -> pd.DataFrame:
    path = PROJECT_ROOT / "variants" / run.variant / "outputs" / "tables" / "exact_parallel_mcmc_posterior_samples.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    samples = pd.read_csv(path)
    required = [
        "eta_t",
        "input_alpha_dndm",
        "input_log10_m_c_msun",
        "final_total_initial_count_above_log10_4",
        "final_total_initial_stellar_mass_above_log10_4_msun",
    ]
    missing = [column for column in required if column not in samples.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    return samples.loc[:, required].dropna().reset_index(drop=True)


def quantile_curves(samples: pd.DataFrame, log_mass: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    differential_curves = []
    cumulative_curves = []
    for row in samples.itertuples(index=False):
        shape = schechter_dndlogm(
            log_mass,
            float(row.input_alpha_dndm),
            float(row.input_log10_m_c_msun),
        )
        dndlogm = normalize_above_threshold(
            log_mass,
            shape,
            float(row.final_total_initial_count_above_log10_4),
            threshold_log_mass=4.0,
        )
        differential_curves.append(dndlogm)
        cumulative_curves.append(cumulative_from_differential(log_mass, dndlogm))
    return (
        np.quantile(np.asarray(differential_curves), [0.16, 0.5, 0.84], axis=0),
        np.quantile(np.asarray(cumulative_curves), [0.16, 0.5, 0.84], axis=0),
    )


def summary_from_samples(samples: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    return {
        column: tuple(float(value) for value in samples[column].quantile([0.16, 0.5, 0.84]))
        for column in samples.columns
    }


def format_symmetric_interval(q16: float, q50: float, q84: float, precision: int) -> str:
    lower = q50 - q16
    upper = q84 - q50
    return (
        f"{q50:.{precision}f}"
        f"$^{{+{upper:.{precision}f}}}_{{-{lower:.{precision}f}}}$"
    )


def format_count(q16: float, q50: float, q84: float) -> str:
    lower = q50 - q16
    upper = q84 - q50
    return f"{q50:.0f}$^{{+{upper:.0f}}}_{{-{lower:.0f}}}$"


def format_mass_1e8(q16: float, q50: float, q84: float) -> str:
    return format_symmetric_interval(q16 / 1.0e8, q50 / 1.0e8, q84 / 1.0e8, 2)


def build_figure(curves: list[tuple[ImfRun, np.ndarray, np.ndarray]], output_pdf: Path, output_png: Path) -> None:
    log_mass = np.linspace(4.0, 7.35, 550)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)

    for run, differential_quantiles, cumulative_quantiles in curves:
        diff_q16, diff_q50, diff_q84 = differential_quantiles
        cum_q16, cum_q50, cum_q84 = cumulative_quantiles

        axes[0].fill_between(log_mass, diff_q16, diff_q84, color=run.color, alpha=0.13, linewidth=0.0)
        axes[0].plot(log_mass, diff_q50, color=run.color, lw=2.0, label=run.label)

        axes[1].fill_between(log_mass, cum_q16, cum_q84, color=run.color, alpha=0.13, linewidth=0.0)
        axes[1].plot(log_mass, cum_q50, color=run.color, lw=2.0, label=run.label)

    axes[0].set_title("Differential IMF")
    axes[0].set_ylabel(r"$dN/d\log_{10}M_{\rm ini}$")
    axes[0].set_ylim(1.0e-2, 8.0e3)
    axes[1].set_title("Cumulative IMF")
    axes[1].set_ylabel(r"$N(>M_{\rm ini})$")
    axes[1].set_ylim(2.0e-1, 1.5e4)

    for axis in axes:
        axis.set_xlim(4.0, 7.35)
        axis.set_yscale("log")
        axis.set_xlabel(r"$\log_{10}(M_{\rm ini}/{\rm M}_\odot)$")
        axis.grid(alpha=0.15, linewidth=0.6)

    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def write_tables(rows: list[dict[str, object]], table_tex: Path, table_csv: Path) -> None:
    table_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(table_csv, index=False)

    lines = []
    for row in rows:
        lines.append(
            "        "
            + " & ".join(
                [
                    str(row["tex_label"]),
                    str(row["eta_t_tex"]),
                    str(row["alpha_tex"]),
                    str(row["logmc_tex"]),
                    str(row["n0_tex"]),
                    str(row["m0_1e8_tex"]),
                ]
            )
            + r" \\"
        )
    table_tex.write_text("\n".join(lines) + "\n")


def main() -> None:
    log_mass = np.linspace(4.0, 7.35, 550)
    curves = []
    rows: list[dict[str, object]] = []

    for run in RUNS:
        samples = load_samples(run)
        differential_quantiles, cumulative_quantiles = quantile_curves(samples, log_mass)
        curves.append((run, differential_quantiles, cumulative_quantiles))

        summary = summary_from_samples(samples)
        eta = summary["eta_t"]
        alpha = summary["input_alpha_dndm"]
        logmc = summary["input_log10_m_c_msun"]
        n0 = summary["final_total_initial_count_above_log10_4"]
        m0 = summary["final_total_initial_stellar_mass_above_log10_4_msun"]
        rows.append(
            {
                "model": run.label,
                "tex_label": run.tex_label,
                "variant": run.variant,
                "eta_t_q16": eta[0],
                "eta_t_q50": eta[1],
                "eta_t_q84": eta[2],
                "alpha_q16": alpha[0],
                "alpha_q50": alpha[1],
                "alpha_q84": alpha[2],
                "logmc_q16": logmc[0],
                "logmc_q50": logmc[1],
                "logmc_q84": logmc[2],
                "n0_q16": n0[0],
                "n0_q50": n0[1],
                "n0_q84": n0[2],
                "m0_q16": m0[0],
                "m0_q50": m0[1],
                "m0_q84": m0[2],
                "eta_t_tex": format_symmetric_interval(*eta, precision=2),
                "alpha_tex": format_symmetric_interval(*alpha, precision=2),
                "logmc_tex": format_symmetric_interval(*logmc, precision=2),
                "n0_tex": format_count(*n0),
                "m0_1e8_tex": format_mass_1e8(*m0),
            }
        )

    build_figure(
        curves,
        PROJECT_ROOT / "paper" / "figures" / "gg23_imf_comparison.pdf",
        PROJECT_ROOT / "paper" / "figures" / "gg23_imf_comparison.png",
    )
    write_tables(
        rows,
        PROJECT_ROOT / "paper" / "tables" / "gg23_imf_comparison.tex",
        PROJECT_ROOT / "paper" / "tables" / "gg23_imf_comparison.csv",
    )
    print(PROJECT_ROOT / "paper" / "figures" / "gg23_imf_comparison.pdf")
    print(PROJECT_ROOT / "paper" / "tables" / "gg23_imf_comparison.tex")


if __name__ == "__main__":
    main()
