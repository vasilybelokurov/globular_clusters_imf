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
from scipy import stats

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

POSTERIOR_PREDICTIVE_SCORE_SPACE = "logMnow_loga_D_absb_absl"


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


def quantile_imf_curves(samples: pd.DataFrame, log_mass: np.ndarray) -> np.ndarray:
    differential_curves = []
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
    return np.quantile(np.asarray(differential_curves), [0.16, 0.5, 0.84], axis=0)


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


def format_delta_log_predictive(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.2f}"


def load_posterior_predictive_deltas() -> dict[str, float]:
    path = PROJECT_ROOT / "outputs" / "tables" / "observed_space_posterior_predictive_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is required for the Table 2 predictive-score column. "
            "Run scripts/compute_observed_space_posterior_predictive_scores.py first."
        )
    table = pd.read_csv(path)
    subset = table.loc[table["score_space"].astype(str) == POSTERIOR_PREDICTIVE_SCORE_SPACE]
    if subset.empty:
        raise ValueError(
            f"{path} does not contain score_space={POSTERIOR_PREDICTIVE_SCORE_SPACE!r}."
        )
    return {
        str(row.model_label): float(row.delta_posterior_predictive_log_likelihood)
        for row in subset.itertuples(index=False)
    }


def padded_limits(values: np.ndarray, quantiles: tuple[float, float] = (0.005, 0.995), pad_fraction: float = 0.08) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    low, high = np.quantile(finite, quantiles)
    width = max(float(high - low), 1.0e-6)
    return float(low - pad_fraction * width), float(high + pad_fraction * width)


def hpd_contour_grid(
    x: np.ndarray,
    y: np.ndarray,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    n_grid: int = 90,
    probabilities: tuple[float, float] = (0.95, 0.68),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    kde = stats.gaussian_kde(np.vstack([x, y]))
    x_grid = np.linspace(x_limits[0], x_limits[1], n_grid)
    y_grid = np.linspace(y_limits[0], y_limits[1], n_grid)
    xx, yy = np.meshgrid(x_grid, y_grid)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    sorted_density = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(sorted_density)
    cumulative /= cumulative[-1]
    levels = []
    for probability in probabilities:
        index = int(np.searchsorted(cumulative, probability, side="left"))
        index = min(index, len(sorted_density) - 1)
        levels.append(float(sorted_density[index]))
    return xx, yy, density, levels


def draw_hpd_contours(
    axis,
    x: np.ndarray,
    y: np.ndarray,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    color: str,
) -> None:
    xx, yy, density, levels = hpd_contour_grid(x, y, x_limits, y_limits)
    for level, linewidth, alpha in zip(levels, [0.9, 1.8], [0.45, 0.95], strict=True):
        if float(np.nanmin(density)) < level < float(np.nanmax(density)):
            axis.contour(xx, yy, density, levels=[level], colors=[color], linewidths=linewidth, alpha=alpha)
    axis.scatter(
        np.median(x),
        np.median(y),
        s=18,
        color=color,
        edgecolor="white",
        linewidth=0.45,
        zorder=4,
    )


def build_figure(curves: list[tuple[ImfRun, np.ndarray, pd.DataFrame]], output_pdf: Path, output_png: Path) -> None:
    log_mass = np.linspace(4.0, 7.35, 550)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)

    all_alpha = np.concatenate([samples["input_alpha_dndm"].to_numpy(dtype=float) for _, _, samples in curves])
    all_logmc = np.concatenate([samples["input_log10_m_c_msun"].to_numpy(dtype=float) for _, _, samples in curves])
    all_n0 = np.concatenate([samples["final_total_initial_count_above_log10_4"].to_numpy(dtype=float) for _, _, samples in curves])
    all_mstar_1e8 = np.concatenate(
        [samples["final_total_initial_stellar_mass_above_log10_4_msun"].to_numpy(dtype=float) / 1.0e8 for _, _, samples in curves]
    )
    alpha_limits = padded_limits(all_alpha)
    logmc_limits = padded_limits(all_logmc)
    n0_limits = padded_limits(all_n0)
    mstar_limits = padded_limits(all_mstar_1e8)

    for run, differential_quantiles, samples in curves:
        diff_q16, diff_q50, diff_q84 = differential_quantiles

        axes[0].fill_between(log_mass, diff_q16, diff_q84, color=run.color, alpha=0.13, linewidth=0.0)
        axes[0].plot(log_mass, diff_q50, color=run.color, lw=2.0, label=run.label)

        draw_hpd_contours(
            axes[1],
            samples["input_alpha_dndm"].to_numpy(dtype=float),
            samples["input_log10_m_c_msun"].to_numpy(dtype=float),
            alpha_limits,
            logmc_limits,
            run.color,
        )
        draw_hpd_contours(
            axes[2],
            samples["final_total_initial_count_above_log10_4"].to_numpy(dtype=float),
            samples["final_total_initial_stellar_mass_above_log10_4_msun"].to_numpy(dtype=float) / 1.0e8,
            n0_limits,
            mstar_limits,
            run.color,
        )

    axes[0].set_title("Differential IMF")
    axes[0].set_ylabel(r"$dN/d\log_{10}M_{\rm ini}$")
    axes[0].set_ylim(1.0e-2, 8.0e3)
    axes[0].set_xlim(4.0, 7.35)
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$\log_{10}(M_{\rm ini}/{\rm M}_\odot)$")

    axes[1].set_title("Schechter parameters")
    axes[1].set_xlim(*alpha_limits)
    axes[1].set_ylim(*logmc_limits)
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel(r"$\log_{10}(M_c/{\rm M}_\odot)$")

    axes[2].set_title("Birth population")
    axes[2].set_xlim(*n0_limits)
    axes[2].set_ylim(*mstar_limits)
    axes[2].set_xlabel(r"$N_0(>10^4{\rm M}_\odot)$")
    axes[2].set_ylabel(r"$M_{\star,0}(>10^4{\rm M}_\odot)\ [10^8{\rm M}_\odot]$")

    for axis in axes:
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
                    str(row["delta_log_pp_tex"]),
                ]
            )
            + r" \\"
        )
    table_tex.write_text("\n".join(lines) + "\n")


def main() -> None:
    log_mass = np.linspace(4.0, 7.35, 550)
    curves = []
    rows: list[dict[str, object]] = []
    delta_log_pp_by_model = load_posterior_predictive_deltas()

    for run in RUNS:
        samples = load_samples(run)
        differential_quantiles = quantile_imf_curves(samples, log_mass)
        curves.append((run, differential_quantiles, samples))

        summary = summary_from_samples(samples)
        eta = summary["eta_t"]
        alpha = summary["input_alpha_dndm"]
        logmc = summary["input_log10_m_c_msun"]
        n0 = summary["final_total_initial_count_above_log10_4"]
        m0 = summary["final_total_initial_stellar_mass_above_log10_4_msun"]
        delta_log_pp = delta_log_pp_by_model[run.label]
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
                "delta_log_pp": delta_log_pp,
                "eta_t_tex": format_symmetric_interval(*eta, precision=2),
                "alpha_tex": format_symmetric_interval(*alpha, precision=2),
                "logmc_tex": format_symmetric_interval(*logmc, precision=2),
                "n0_tex": format_count(*n0),
                "m0_1e8_tex": format_mass_1e8(*m0),
                "delta_log_pp_tex": format_delta_log_predictive(delta_log_pp),
            }
        )

    rows = sorted(rows, key=lambda row: float(row["delta_log_pp"]), reverse=True)
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
