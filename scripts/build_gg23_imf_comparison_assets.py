from __future__ import annotations

import pickle
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
from scipy import interpolate

from globular_clusters_imf.joint_model import (
    JointLikelihoodContext,
    JointModelSpec,
    fit_single_joint_model_with_fixed_imf_params,
)

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


def load_best_result(run: ImfRun) -> dict[str, object]:
    path = PROJECT_ROOT / "variants" / run.variant / "outputs" / "tables" / "exact_parallel_mcmc_best_result.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        return pickle.load(handle)


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


def load_surface_index_table(run: ImfRun) -> pd.DataFrame:
    worker_dir = PROJECT_ROOT / "variants" / run.variant / "outputs" / "parallel_exact_mcmc_workers"
    frames = []
    for csv_path in sorted(worker_dir.glob("chain_*_selection_surfaces.csv")):
        chain = int(csv_path.name.split("_")[1])
        frame = pd.read_csv(csv_path).reset_index(names="surface_index")
        frame["chain"] = chain
        frame["surface_npz"] = str(worker_dir / f"chain_{chain}_selection_surfaces.npz")
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No posterior surface files found under {worker_dir}")
    return pd.concat(frames, ignore_index=True)


def quantile_radial_curves(
    run: ImfRun,
    n_samples: int = 120,
    seed: int = 20260604,
) -> tuple[np.ndarray, np.ndarray]:
    best_result = load_best_result(run)
    base_context = best_result["final_context"]
    radius_grid = np.power(10.0, np.asarray(base_context.log_a_grid, dtype=float))
    spec = JointModelSpec(imf_family="schechter", radial_model="logpoly3")
    best_radial_start = np.asarray(best_result["final_payload"]["radial_parameters_raw"], dtype=float)

    surface_table = load_surface_index_table(run)
    ok = surface_table.loc[surface_table["status"].astype(str) == "ok"].reset_index(drop=True)
    if ok.empty:
        raise ValueError(f"No successful posterior surface rows for {run.variant}")
    selected = ok.sample(n=min(n_samples, len(ok)), random_state=seed).reset_index(drop=True)

    npz_cache: dict[str, np.lib.npyio.NpzFile] = {}
    radial_curves = []
    current_start = best_radial_start
    for row in selected.itertuples(index=False):
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
            context,
            spec=spec,
            fixed_imf_params=np.array(
                [float(row.input_alpha_dndm), float(row.input_log10_m_c_msun)],
                dtype=float,
            ),
            start_radial_params=current_start,
        )
        current_start = np.asarray(payload["radial_parameters_raw"], dtype=float)
        radial_density = np.asarray(payload["model"]["radial_density_grid"], dtype=float)
        radial_curves.append(float(row.final_total_initial_count_above_log10_4) * radial_density)

    for item in npz_cache.values():
        item.close()
    return radius_grid, np.quantile(np.asarray(radial_curves), [0.16, 0.5, 0.84], axis=0)


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


def build_figure(curves: list[tuple[ImfRun, np.ndarray, np.ndarray, np.ndarray]], output_pdf: Path, output_png: Path) -> None:
    log_mass = np.linspace(4.0, 7.35, 550)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)

    for run, differential_quantiles, radius_grid, radial_quantiles in curves:
        diff_q16, diff_q50, diff_q84 = differential_quantiles
        radial_q16, radial_q50, radial_q84 = radial_quantiles

        axes[0].fill_between(log_mass, diff_q16, diff_q84, color=run.color, alpha=0.13, linewidth=0.0)
        axes[0].plot(log_mass, diff_q50, color=run.color, lw=2.0, label=run.label)

        axes[1].fill_between(radius_grid, radial_q16, radial_q84, color=run.color, alpha=0.13, linewidth=0.0)
        axes[1].plot(radius_grid, radial_q50, color=run.color, lw=2.0, label=run.label)

    axes[0].set_title("Differential IMF")
    axes[0].set_ylabel(r"$dN/d\log_{10}M_{\rm ini}$")
    axes[0].set_ylim(1.0e-2, 8.0e3)
    axes[0].set_xlim(4.0, 7.35)
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$\log_{10}(M_{\rm ini}/{\rm M}_\odot)$")

    axes[1].set_title("Intrinsic radial law")
    axes[1].set_ylabel(r"$dN_0(>10^4{\rm M}_\odot)/d\log_{10}a$")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlim(0.45, 300.0)
    axes[1].set_ylim(2.0e1, 7.0e3)
    axes[1].set_xlabel(r"$a\ [{\rm kpc}]$")

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
        differential_quantiles = quantile_imf_curves(samples, log_mass)
        radius_grid, radial_quantiles = quantile_radial_curves(run)
        curves.append((run, differential_quantiles, radius_grid, radial_quantiles))

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
