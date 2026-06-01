from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SINGLE_VARIANT = "profile_map_and_exact_mcmc_schechter_logpoly3_logistic_global_monotonic_q"
DEFAULT_TWO_COMPONENT_VARIANT = "profile_map_and_exact_mcmc_bk_shared_schechter_two_component_logistic_global_monotonic_q"


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _variant_tables_dir(variant_name: str) -> Path:
    return PROJECT_ROOT / "variants" / variant_name / "outputs" / "tables"


def _posterior_quantile(samples: pd.DataFrame, column: str) -> tuple[float, float, float]:
    if column not in samples.columns:
        raise KeyError(f"Missing posterior column {column!r}")
    values = np.asarray(samples[column], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"Posterior column {column!r} has no finite samples")
    q16, q50, q84 = np.quantile(values, [0.16, 0.50, 0.84])
    return float(q16), float(q50), float(q84)


def _posterior_summary_quantile(summary: pd.DataFrame, parameter: str) -> tuple[float, float, float]:
    match = summary.loc[summary["parameter"] == parameter]
    if match.empty:
        raise KeyError(f"Missing posterior summary row {parameter!r}")
    row = match.iloc[0]
    return float(row["q16"]), float(row["q50"]), float(row["q84"])


def _interval_tex(q16: float, q50: float, q84: float, *, ndigits: int) -> tuple[str, str, str]:
    return (
        f"{q50:.{ndigits}f}",
        f"{q50 - q16:.{ndigits}f}",
        f"{q84 - q50:.{ndigits}f}",
    )


def _signed(value: float, *, ndigits: int) -> str:
    return f"{value:+.{ndigits}f}"


def _provide(name: str, value: str) -> str:
    return rf"\providecommand{{\{name}}}{{{value}}}"


def _write_two_component_numbers(
    *,
    single_best: dict[str, object],
    two_best: dict[str, object],
    two_summary: pd.DataFrame,
    two_samples: pd.DataFrame,
    output_path: Path,
) -> None:
    n_total = int(two_best["n_clusters_total"])
    n_in = int(two_best["n_clusters_in_situ"])
    n_acc = int(two_best["n_clusters_accreted"])
    partition_constant = math.lgamma(n_total + 1) - math.lgamma(n_in + 1) - math.lgamma(n_acc + 1)

    single_bic = float(single_best["bic"])
    two_raw_bic = float(two_best["bic"])
    two_cond_bic = two_raw_bic - 2.0 * partition_constant
    raw_delta_bic = two_raw_bic - single_bic
    cond_delta_bic = two_cond_bic - single_bic

    eta = _interval_tex(*_posterior_summary_quantile(two_summary, "eta_t"), ndigits=3)
    alpha = _interval_tex(*_posterior_summary_quantile(two_summary, "input_alpha_dndm"), ndigits=3)
    logmc = _interval_tex(*_posterior_summary_quantile(two_summary, "input_log10_m_c_msun"), ndigits=3)
    total_count = _interval_tex(
        *_posterior_summary_quantile(two_summary, "final_total_initial_count_above_log10_4"),
        ndigits=0,
    )
    total_mass = _interval_tex(
        *tuple(value / 1.0e8 for value in _posterior_summary_quantile(two_summary, "final_total_initial_stellar_mass_above_log10_4_msun")),
        ndigits=2,
    )
    mean_detectability = _interval_tex(
        *_posterior_summary_quantile(two_summary, "mean_detectability_above_log10_4"),
        ndigits=3,
    )
    in_situ_count = _interval_tex(
        *_posterior_summary_quantile(two_summary, "final_total_initial_count_above_log10_4_in_situ"),
        ndigits=0,
    )
    accreted_count = _interval_tex(
        *_posterior_summary_quantile(two_summary, "final_total_initial_count_above_log10_4_accreted"),
        ndigits=0,
    )
    in_situ_mass = _interval_tex(
        *tuple(
            value / 1.0e8
            for value in _posterior_quantile(two_samples, "final_total_initial_stellar_mass_above_log10_4_msun_in_situ")
        ),
        ndigits=2,
    )
    accreted_mass = _interval_tex(
        *tuple(
            value / 1.0e8
            for value in _posterior_quantile(two_samples, "final_total_initial_stellar_mass_above_log10_4_msun_accreted")
        ),
        ndigits=2,
    )
    accreted_fraction_values = (
        np.asarray(two_samples["final_total_initial_count_above_log10_4_accreted"], dtype=float)
        / np.asarray(two_samples["final_total_initial_count_above_log10_4"], dtype=float)
    )
    accreted_fraction = _interval_tex(
        *tuple(np.quantile(accreted_fraction_values[np.isfinite(accreted_fraction_values)], [0.16, 0.50, 0.84])),
        ndigits=3,
    )

    macro_values = {
        "TwoCompNTotal": f"{n_total:d}",
        "TwoCompNInSitu": f"{n_in:d}",
        "TwoCompNAccreted": f"{n_acc:d}",
        "TwoCompPartitionConstant": f"{partition_constant:.2f}",
        "TwoCompPartitionBICCorrection": f"{2.0 * partition_constant:.2f}",
        "TwoCompSingleBIC": f"{single_bic:.2f}",
        "TwoCompRawBIC": f"{two_raw_bic:.2f}",
        "TwoCompConditionalBIC": f"{two_cond_bic:.2f}",
        "TwoCompRawDeltaBIC": _signed(raw_delta_bic, ndigits=1),
        "TwoCompConditionalDeltaBIC": _signed(cond_delta_bic, ndigits=1),
        "TwoCompBestLogL": f"{float(two_best['log_likelihood']):.2f}",
        "TwoCompEtaMed": eta[0],
        "TwoCompEtaMinus": eta[1],
        "TwoCompEtaPlus": eta[2],
        "TwoCompAlphaMed": alpha[0],
        "TwoCompAlphaMinus": alpha[1],
        "TwoCompAlphaPlus": alpha[2],
        "TwoCompMcMed": logmc[0],
        "TwoCompMcMinus": logmc[1],
        "TwoCompMcPlus": logmc[2],
        "TwoCompNzeroMed": total_count[0],
        "TwoCompNzeroMinus": total_count[1],
        "TwoCompNzeroPlus": total_count[2],
        "TwoCompMassZeroEightMed": total_mass[0],
        "TwoCompMassZeroEightMinus": total_mass[1],
        "TwoCompMassZeroEightPlus": total_mass[2],
        "TwoCompMeanDetectabilityMed": mean_detectability[0],
        "TwoCompMeanDetectabilityMinus": mean_detectability[1],
        "TwoCompMeanDetectabilityPlus": mean_detectability[2],
        "TwoCompInSituNzeroMed": in_situ_count[0],
        "TwoCompInSituNzeroMinus": in_situ_count[1],
        "TwoCompInSituNzeroPlus": in_situ_count[2],
        "TwoCompAccretedNzeroMed": accreted_count[0],
        "TwoCompAccretedNzeroMinus": accreted_count[1],
        "TwoCompAccretedNzeroPlus": accreted_count[2],
        "TwoCompInSituMassZeroEightMed": in_situ_mass[0],
        "TwoCompInSituMassZeroEightMinus": in_situ_mass[1],
        "TwoCompInSituMassZeroEightPlus": in_situ_mass[2],
        "TwoCompAccretedMassZeroEightMed": accreted_mass[0],
        "TwoCompAccretedMassZeroEightMinus": accreted_mass[1],
        "TwoCompAccretedMassZeroEightPlus": accreted_mass[2],
        "TwoCompAccretedFractionMed": accreted_fraction[0],
        "TwoCompAccretedFractionMinus": accreted_fraction[1],
        "TwoCompAccretedFractionPlus": accreted_fraction[2],
    }
    lines = [
        "% Generated by scripts/build_two_component_paper_tables.py.",
        "% Do not edit numerical values here by hand.",
    ]
    lines.extend(_provide(name, value) for name, value in macro_values.items())
    output_path.write_text("\n".join(lines) + "\n")


def _write_two_component_table_body(output_path: Path) -> None:
    lines = [
        "% Generated by scripts/build_two_component_paper_tables.py.",
        "% This file is the body of Table~\\ref{tab:two_component_check}.",
        r"Quantity & Single component & Two-component total & In situ & Accreted \\",
        r"\hline",
        (
            r"$\eta_t$ & "
            r"$\PosteriorEtaMed_{-\PosteriorEtaMinus}^{+\PosteriorEtaPlus}$ & "
            r"$\TwoCompEtaMed_{-\TwoCompEtaMinus}^{+\TwoCompEtaPlus}$ & -- & -- \\"
        ),
        (
            r"$\alpha$ & "
            r"$\PosteriorAlphaMed_{-\PosteriorAlphaMinus}^{+\PosteriorAlphaPlus}$ & "
            r"$\TwoCompAlphaMed_{-\TwoCompAlphaMinus}^{+\TwoCompAlphaPlus}$ & -- & -- \\"
        ),
        (
            r"$\log_{10}(M_c/{\rm M}_\odot)$ & "
            r"$\PosteriorMcMed_{-\PosteriorMcMinus}^{+\PosteriorMcPlus}$ & "
            r"$\TwoCompMcMed_{-\TwoCompMcMinus}^{+\TwoCompMcPlus}$ & -- & -- \\"
        ),
        (
            r"$N_0(>10^4\,{\rm M}_\odot)$ & "
            r"$\PosteriorNzeroMed_{-\PosteriorNzeroMinus}^{+\PosteriorNzeroPlus}$ & "
            r"$\TwoCompNzeroMed_{-\TwoCompNzeroMinus}^{+\TwoCompNzeroPlus}$ & "
            r"$\TwoCompInSituNzeroMed_{-\TwoCompInSituNzeroMinus}^{+\TwoCompInSituNzeroPlus}$ & "
            r"$\TwoCompAccretedNzeroMed_{-\TwoCompAccretedNzeroMinus}^{+\TwoCompAccretedNzeroPlus}$ \\"
        ),
        (
            r"$M_{\star,0}(>10^4\,{\rm M}_\odot)\,[10^8\,{\rm M}_\odot]$ & "
            r"$\PosteriorMassZeroEightMed_{-\PosteriorMassZeroEightMinus}^{+\PosteriorMassZeroEightPlus}$ & "
            r"$\TwoCompMassZeroEightMed_{-\TwoCompMassZeroEightMinus}^{+\TwoCompMassZeroEightPlus}$ & "
            r"$\TwoCompInSituMassZeroEightMed_{-\TwoCompInSituMassZeroEightMinus}^{+\TwoCompInSituMassZeroEightPlus}$ & "
            r"$\TwoCompAccretedMassZeroEightMed_{-\TwoCompAccretedMassZeroEightMinus}^{+\TwoCompAccretedMassZeroEightPlus}$ \\"
        ),
        (
            r"$\langle Q\rangle_{>10^4}$ & "
            r"$\PosteriorMeanDetectabilityMed_{-\PosteriorMeanDetectabilityMinus}^{+\PosteriorMeanDetectabilityPlus}$ & "
            r"$\TwoCompMeanDetectabilityMed_{-\TwoCompMeanDetectabilityMinus}^{+\TwoCompMeanDetectabilityPlus}$ & -- & -- \\"
        ),
        (
            r"$f_{\rm acc}$ & -- & "
            r"$\TwoCompAccretedFractionMed_{-\TwoCompAccretedFractionMinus}^{+\TwoCompAccretedFractionPlus}$ & -- & -- \\"
        ),
        r"\hline",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-variant", default=DEFAULT_SINGLE_VARIANT)
    parser.add_argument("--two-component-variant", default=DEFAULT_TWO_COMPONENT_VARIANT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "paper" / "tables")
    args = parser.parse_args()

    single_tables = _variant_tables_dir(args.single_variant)
    two_tables = _variant_tables_dir(args.two_component_variant)
    single_best = _load_json(single_tables / "exact_parallel_mcmc_best_result_summary.json")
    two_best = _load_json(two_tables / "exact_parallel_mcmc_best_result_summary.json")
    two_summary = pd.read_csv(two_tables / "exact_parallel_posterior_summary.csv")
    two_samples = pd.read_csv(two_tables / "exact_parallel_mcmc_posterior_samples.csv")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    numbers_path = args.output_dir / "two_component_numbers.tex"
    table_path = args.output_dir / "two_component_results.tex"
    _write_two_component_numbers(
        single_best=single_best,
        two_best=two_best,
        two_summary=two_summary,
        two_samples=two_samples,
        output_path=numbers_path,
    )
    _write_two_component_table_body(table_path)
    print(numbers_path)
    print(table_path)


if __name__ == "__main__":
    main()
