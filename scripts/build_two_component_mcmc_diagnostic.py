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


SINGLE_ROOT = PROJECT_ROOT / "variants" / "profile_map_and_exact_mcmc_schechter_logpoly3_logistic_global_monotonic_q"
TWO_ROOT = PROJECT_ROOT / "variants" / "profile_map_and_exact_mcmc_bk_shared_schechter_two_component_logistic_global_monotonic_q"


def _load_samples(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "outputs" / "tables" / "exact_parallel_mcmc_posterior_samples.csv")


def _summary(values: np.ndarray) -> tuple[float, float, float]:
    q16, q50, q84 = np.percentile(np.asarray(values, dtype=float), [16.0, 50.0, 84.0])
    return float(q16), float(q50), float(q84)


def _step_hist(ax, values: np.ndarray, *, bins: np.ndarray, label: str, color: str, linestyle: str = "-") -> None:
    ax.hist(
        np.asarray(values, dtype=float),
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.7,
        color=color,
        linestyle=linestyle,
        label=label,
    )


def main() -> None:
    single = _load_samples(SINGLE_ROOT)
    two = _load_samples(TWO_ROOT)
    two["accreted_fraction_N_gt4"] = (
        two["final_total_initial_count_above_log10_4_accreted"]
        / two["final_total_initial_count_above_log10_4"]
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    ax_eta, ax_alpha, ax_mc, ax_n0 = axes.ravel()

    hist_specs = [
        (ax_eta, "eta_t", r"$\eta_t$", np.linspace(0.65, 2.15, 30)),
        (ax_alpha, "input_alpha_dndm", r"$\alpha$", np.linspace(-1.75, -0.75, 30)),
        (ax_mc, "input_log10_m_c_msun", r"$\log_{10}(M_c/{\rm M}_\odot)$", np.linspace(6.05, 6.65, 30)),
    ]
    for ax, column, xlabel, bins in hist_specs:
        _step_hist(ax, single[column], bins=bins, label="single component", color="#333333")
        _step_hist(ax, two[column], bins=bins, label="B--K two component", color="#1b9e77")
        single_q16, single_q50, single_q84 = _summary(single[column].to_numpy(dtype=float))
        two_q16, two_q50, two_q84 = _summary(two[column].to_numpy(dtype=float))
        ax.axvline(single_q50, color="#333333", linewidth=1.0, alpha=0.7)
        ax.axvline(two_q50, color="#1b9e77", linewidth=1.0, alpha=0.9)
        ax.axvspan(single_q16, single_q84, color="#333333", alpha=0.08, linewidth=0)
        ax.axvspan(two_q16, two_q84, color="#1b9e77", alpha=0.10, linewidth=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Posterior density")
        ax.grid(alpha=0.16, linewidth=0.5)

    ax_eta.legend(frameon=False, fontsize=8.0, loc="upper right")

    labels = [
        "single total",
        "two total",
        "in situ",
        "accreted",
    ]
    columns = [
        ("single", "final_total_initial_count_above_log10_4"),
        ("two", "final_total_initial_count_above_log10_4"),
        ("two", "final_total_initial_count_above_log10_4_in_situ"),
        ("two", "final_total_initial_count_above_log10_4_accreted"),
    ]
    colors = ["#333333", "#1b9e77", "#d95f02", "#7570b3"]
    y_positions = np.arange(len(labels))[::-1]
    for y, label, (which, column), color in zip(y_positions, labels, columns, colors, strict=True):
        sample = single if which == "single" else two
        q16, q50, q84 = _summary(sample[column].to_numpy(dtype=float))
        ax_n0.errorbar(
            q50,
            y,
            xerr=[[q50 - q16], [q84 - q50]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.7,
            capsize=3.0,
            markersize=4.5,
        )
    frac_q16, frac_q50, frac_q84 = _summary(two["accreted_fraction_N_gt4"].to_numpy(dtype=float))
    ax_n0.text(
        0.98,
        0.06,
        rf"$f_{{\rm acc}}={frac_q50:.2f}_{{-{frac_q50 - frac_q16:.2f}}}^{{+{frac_q84 - frac_q50:.2f}}}$",
        transform=ax_n0.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
    )
    ax_n0.set_xscale("log")
    ax_n0.set_yticks(y_positions)
    ax_n0.set_yticklabels(labels)
    ax_n0.set_xlabel(r"$N_0(M_{\rm ini}>10^4\,{\rm M}_\odot)$")
    ax_n0.set_title("Normalization split", fontsize=9.5)
    ax_n0.grid(alpha=0.16, linewidth=0.5, axis="x")

    for label, ax in zip(("a", "b", "c", "d"), axes.ravel(), strict=True):
        ax.text(0.04, 0.94, f"({label})", transform=ax.transAxes, ha="left", va="top", fontsize=9.0)

    output_stem = PROJECT_ROOT / "paper" / "figures" / "two_component_mcmc_diagnostic"
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".png"), dpi=220)
    plt.close(fig)

    print(output_stem.with_suffix(".pdf"))
    print(output_stem.with_suffix(".png"))
    print(
        "two-component accreted N0 fraction: "
        f"{frac_q50:.3f} -{frac_q50 - frac_q16:.3f} +{frac_q84 - frac_q50:.3f}"
    )


if __name__ == "__main__":
    main()
