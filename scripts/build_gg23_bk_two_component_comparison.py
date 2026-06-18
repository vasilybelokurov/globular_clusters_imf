from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd

from globular_clusters_imf.model import fit_catalog_models
from globular_clusters_imf.two_component_model import SharedImfTwoComponentSpec
from run_profile_map_and_exact_mcmc_bk_shared_schechter_two_component import (
    LOG_MASS_MIN,
    _evaluate_theta_single_start,
    _load_catalog,
)


@dataclass(frozen=True)
class Gg23Run:
    label: str
    tex_label: str
    variant: str
    gg23_model: str


RUNS = [
    Gg23Run(
        label="GG23 no BHs",
        tex_label=r"GG23 no BHs",
        variant="gg23_schechter_no_bh_logpoly3_eta01_105",
        gg23_model="gg23_no_bh",
    ),
    Gg23Run(
        label="GG23 BHs",
        tex_label=r"GG23 BHs",
        variant="gg23_schechter_bh_logpoly3",
        gg23_model="gg23_bh",
    ),
    Gg23Run(
        label="GG23 BHs + [Fe/H]",
        tex_label=r"GG23 BHs + [Fe/H]",
        variant="gg23_schechter_bh_feh_gradient_logpoly3",
        gg23_model="gg23_bh_feh_gradient",
    ),
    Gg23Run(
        label="GG23 BHs + past tides",
        tex_label=r"GG23 BHs + past tides",
        variant="gg23_schechter_bh_past_tidal_logpoly3",
        gg23_model="gg23_bh_past_tidal",
    ),
    Gg23Run(
        label="GG23 BHs + [Fe/H] + past tides",
        tex_label=r"GG23 BHs + [Fe/H] + past tides",
        variant="gg23_schechter_bh_feh_gradient_past_tidal_logpoly3",
        gg23_model="gg23_bh_feh_gradient_past_tidal",
    ),
]


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: tuple[float, ...]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        return np.full(len(quantiles), np.nan)
    values = values[finite]
    weights = weights[finite]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return np.interp(np.asarray(quantiles, dtype=float), cdf, values)


def _normalized_importance_weights(log_weight: np.ndarray) -> np.ndarray:
    log_weight = np.asarray(log_weight, dtype=float)
    finite = np.isfinite(log_weight)
    weights = np.zeros_like(log_weight, dtype=float)
    if not np.any(finite):
        return weights
    shifted = log_weight[finite] - float(np.nanmax(log_weight[finite]))
    weights[finite] = np.exp(shifted)
    total = float(np.sum(weights))
    if total > 0.0:
        weights /= total
    return weights


def _interval_tex(q16: float, q50: float, q84: float, ndigits: int) -> str:
    return f"{q50:.{ndigits}f}$^{{+{q84 - q50:.{ndigits}f}}}_{{-{q50 - q16:.{ndigits}f}}}$"


def _count_tex(q16: float, q50: float, q84: float) -> str:
    return _interval_tex(q16, q50, q84, 0)


def _mass_tex(q16: float, q50: float, q84: float) -> str:
    return _interval_tex(q16 / 1.0e8, q50 / 1.0e8, q84 / 1.0e8, 2)


def _sample_rows(samples: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
    required = ["eta_t", "input_alpha_dndm", "input_log10_m_c_msun", "log_likelihood"]
    missing = [column for column in required if column not in samples.columns]
    if missing:
        raise ValueError(f"Posterior sample table is missing columns {missing}")
    clean = samples.loc[:, required].dropna().reset_index(drop=True)
    if len(clean) <= n_samples:
        return clean.copy()
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(clean), size=n_samples, replace=False)
    best = int(clean["log_likelihood"].idxmax())
    if best not in selected:
        selected[0] = best
    return clean.iloc[np.sort(selected)].reset_index(drop=True)


def _summarize_weighted(table: pd.DataFrame, column: str, weights: np.ndarray) -> tuple[float, float, float]:
    q16, q50, q84 = _weighted_quantile(table[column].to_numpy(dtype=float), weights, (0.16, 0.50, 0.84))
    return float(q16), float(q50), float(q84)


def _single_summary(samples: pd.DataFrame, column: str) -> tuple[float, float, float]:
    q16, q50, q84 = samples[column].quantile([0.16, 0.50, 0.84])
    return float(q16), float(q50), float(q84)


def _partition_constant(n_total: int, n_in_situ: int, n_accreted: int) -> float:
    return math.lgamma(n_total + 1) - math.lgamma(n_in_situ + 1) - math.lgamma(n_accreted + 1)


def evaluate_run(
    *,
    run: Gg23Run,
    prepared_catalog: pd.DataFrame,
    spec: SharedImfTwoComponentSpec,
    output_root: Path,
    n_samples: int,
    seed: int,
    n_detectability_iterations: int,
    relaxation: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    single_tables = PROJECT_ROOT / "variants" / run.variant / "outputs" / "tables"
    samples = pd.read_csv(single_tables / "exact_parallel_mcmc_posterior_samples.csv")
    single_best = json.loads((single_tables / "exact_parallel_mcmc_best_result_summary.json").read_text())
    sampled = _sample_rows(samples, n_samples=n_samples, seed=seed)

    records: list[dict[str, object]] = []
    start_state = None
    for index, sample in sampled.iterrows():
        theta = np.array(
            [
                float(sample["eta_t"]),
                float(sample["input_alpha_dndm"]),
                float(sample["input_log10_m_c_msun"]),
            ],
            dtype=float,
        )
        entry = _evaluate_theta_single_start(
            prepared_catalog=prepared_catalog,
            spec=spec,
            theta=theta,
            start_state=start_state,
            project_root=output_root,
            n_detectability_iterations=n_detectability_iterations,
            relaxation=relaxation,
            survivability_backend="gg23",
            gg23_model_name=run.gg23_model,
        )
        start_state = entry["start_state"]
        row = dict(entry["row"])
        row["sample_index"] = int(index)
        row["model_label"] = run.label
        row["tex_label"] = run.tex_label
        row["variant"] = run.variant
        row["single_log_likelihood"] = float(sample["log_likelihood"])
        row["importance_log_weight"] = float(row["log_likelihood"] - sample["log_likelihood"])
        records.append(row)
        print(
            f"[{run.label}] {len(records):03d}/{len(sampled):03d} "
            f"eta={theta[0]:.3f} alpha={theta[1]:.3f} logMc={theta[2]:.3f} "
            f"logL2={float(row['log_likelihood']):.3f}"
        )

    table = pd.DataFrame(records)
    weights = _normalized_importance_weights(table["importance_log_weight"].to_numpy(dtype=float))
    ess = float(1.0 / np.sum(weights**2)) if np.sum(weights**2) > 0.0 else 0.0

    n_total = int(table["n_clusters_total"].dropna().iloc[0])
    n_in = int(table["n_clusters_in_situ"].dropna().iloc[0])
    n_acc = int(table["n_clusters_accreted"].dropna().iloc[0])
    partition = _partition_constant(n_total, n_in, n_acc)
    best_two = table.loc[table["log_likelihood"].idxmax()]
    two_raw_bic = float(np.log(n_total) * 8 - 2.0 * float(best_two["log_likelihood"]))
    two_cond_bic = two_raw_bic - 2.0 * partition
    single_bic = float(single_best["bic"])

    single_n0 = _single_summary(samples, "final_total_initial_count_above_log10_4")
    single_mass = _single_summary(samples, "final_total_initial_stellar_mass_above_log10_4_msun")
    total_n0 = _summarize_weighted(table, "final_total_initial_count_above_log10_4", weights)
    in_n0 = _summarize_weighted(table, "final_total_initial_count_above_log10_4_in_situ", weights)
    acc_n0 = _summarize_weighted(table, "final_total_initial_count_above_log10_4_accreted", weights)
    total_mass = _summarize_weighted(table, "final_total_initial_stellar_mass_above_log10_4_msun", weights)
    in_mass = _summarize_weighted(table, "final_total_initial_stellar_mass_above_log10_4_msun_in_situ", weights)
    acc_mass = _summarize_weighted(table, "final_total_initial_stellar_mass_above_log10_4_msun_accreted", weights)
    facc_values = (
        table["final_total_initial_count_above_log10_4_accreted"].to_numpy(dtype=float)
        / table["final_total_initial_count_above_log10_4"].to_numpy(dtype=float)
    )
    facc = tuple(float(value) for value in _weighted_quantile(facc_values, weights, (0.16, 0.50, 0.84)))

    summary = {
        "model": run.label,
        "tex_label": run.tex_label,
        "variant": run.variant,
        "gg23_model": run.gg23_model,
        "n_evaluated": int(len(table)),
        "importance_ess": ess,
        "single_bic": single_bic,
        "two_raw_bic_sampled_best": two_raw_bic,
        "two_conditional_bic_sampled_best": two_cond_bic,
        "delta_bic_raw_sampled_best": two_raw_bic - single_bic,
        "delta_bic_conditional_sampled_best": two_cond_bic - single_bic,
        "partition_constant": partition,
        "best_two_log_likelihood": float(best_two["log_likelihood"]),
        "single_n0_q16": single_n0[0],
        "single_n0_q50": single_n0[1],
        "single_n0_q84": single_n0[2],
        "single_mass_q16": single_mass[0],
        "single_mass_q50": single_mass[1],
        "single_mass_q84": single_mass[2],
        "two_n0_q16": total_n0[0],
        "two_n0_q50": total_n0[1],
        "two_n0_q84": total_n0[2],
        "in_situ_n0_q16": in_n0[0],
        "in_situ_n0_q50": in_n0[1],
        "in_situ_n0_q84": in_n0[2],
        "accreted_n0_q16": acc_n0[0],
        "accreted_n0_q50": acc_n0[1],
        "accreted_n0_q84": acc_n0[2],
        "two_mass_q16": total_mass[0],
        "two_mass_q50": total_mass[1],
        "two_mass_q84": total_mass[2],
        "in_situ_mass_q16": in_mass[0],
        "in_situ_mass_q50": in_mass[1],
        "in_situ_mass_q84": in_mass[2],
        "accreted_mass_q16": acc_mass[0],
        "accreted_mass_q50": acc_mass[1],
        "accreted_mass_q84": acc_mass[2],
        "f_acc_q16": facc[0],
        "f_acc_q50": facc[1],
        "f_acc_q84": facc[2],
    }
    summary.update(
        {
            "single_n0_tex": _count_tex(*single_n0),
            "two_n0_tex": _count_tex(*total_n0),
            "in_situ_n0_tex": _count_tex(*in_n0),
            "accreted_n0_tex": _count_tex(*acc_n0),
            "two_mass_tex": _mass_tex(*total_mass),
            "in_situ_mass_tex": _mass_tex(*in_mass),
            "accreted_mass_tex": _mass_tex(*acc_mass),
            "f_acc_tex": _interval_tex(*facc, 3),
            "delta_bic_conditional_tex": f"{summary['delta_bic_conditional_sampled_best']:+.1f}",
            "ess_tex": f"{ess:.0f}",
        }
    )
    return table, summary


def write_tex_table(summary_table: pd.DataFrame, output_path: Path) -> None:
    lines = []
    for row in summary_table.itertuples(index=False):
        lines.append(
            "        "
            + " & ".join(
                [
                    str(row.tex_label),
                    str(row.single_n0_tex),
                    str(row.two_n0_tex),
                    str(row.in_situ_n0_tex),
                    str(row.accreted_n0_tex),
                    str(row.f_acc_tex),
                    str(row.delta_bic_conditional_tex),
                    str(row.ess_tex),
                ]
            )
            + r" \\"
        )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--n-detectability-iterations", type=int, default=12)
    parser.add_argument("--detectability-relaxation", type=float, default=0.7)
    parser.add_argument("--output-root-name", default="gg23_bk_shared_schechter_two_component_reweighted")
    args = parser.parse_args()

    output_root = PROJECT_ROOT / "variants" / args.output_root_name
    tables_dir = output_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    paper_tables_dir = PROJECT_ROOT / "paper" / "tables"
    paper_tables_dir.mkdir(parents=True, exist_ok=True)

    catalog = _load_catalog()
    prepared_catalog = fit_catalog_models(catalog, output_root)["catalog"]
    spec = SharedImfTwoComponentSpec(
        imf_family="schechter",
        in_situ_radial_model="logpoly3",
        accreted_radial_model="logpoly3",
    )

    summary_rows: list[dict[str, object]] = []
    for run_index, run in enumerate(RUNS):
        table, summary = evaluate_run(
            run=run,
            prepared_catalog=prepared_catalog,
            spec=spec,
            output_root=output_root,
            n_samples=int(args.n_samples),
            seed=int(args.seed) + run_index,
            n_detectability_iterations=int(args.n_detectability_iterations),
            relaxation=float(args.detectability_relaxation),
        )
        table.to_csv(tables_dir / f"{run.gg23_model}_two_component_reweighted_samples.csv", index=False)
        summary_rows.append(summary)

    summary_table = pd.DataFrame(summary_rows)
    summary_table.to_csv(tables_dir / "gg23_bk_two_component_summary.csv", index=False)
    summary_table.to_csv(paper_tables_dir / "gg23_bk_two_component_summary.csv", index=False)
    write_tex_table(summary_table, paper_tables_dir / "gg23_bk_two_component_summary.tex")
    print(tables_dir / "gg23_bk_two_component_summary.csv")
    print(paper_tables_dir / "gg23_bk_two_component_summary.tex")


if __name__ == "__main__":
    main()
