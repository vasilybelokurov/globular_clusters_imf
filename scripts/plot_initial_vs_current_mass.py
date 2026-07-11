from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt

DEFAULT_CATALOG = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog_with_origin_and_chemistry.csv"
DEFAULT_OUTPUT_PDF = PROJECT_ROOT / "paper" / "figures" / "initial_vs_current_mass.pdf"
DEFAULT_OUTPUT_PNG = PROJECT_ROOT / "paper" / "figures" / "initial_vs_current_mass.png"
DEFAULT_SUMMARY = PROJECT_ROOT / "outputs" / "tables" / "initial_vs_current_mass_summary.json"


def _setup_matplotlib() -> None:
    (PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)


def _finite_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    required = ["cluster_label", "present_mass_msun", "initial_mass_msun"]
    missing = [column for column in required if column not in catalog.columns]
    if missing:
        raise ValueError(f"Catalogue is missing required columns: {missing}")
    finite = (
        np.isfinite(catalog["present_mass_msun"].to_numpy(dtype=float))
        & np.isfinite(catalog["initial_mass_msun"].to_numpy(dtype=float))
        & (catalog["present_mass_msun"].to_numpy(dtype=float) > 0.0)
        & (catalog["initial_mass_msun"].to_numpy(dtype=float) > 0.0)
    )
    return catalog.loc[finite].copy().reset_index(drop=True)


def _plot(
    catalog: pd.DataFrame,
    output_pdf: Path,
    output_png: Path,
    initial_mass_min: float,
) -> dict[str, object]:
    _setup_matplotlib()
    fit_mask = catalog["initial_mass_msun"].to_numpy(dtype=float) >= float(initial_mass_min)
    below = catalog.loc[~fit_mask].copy()
    fitted = catalog.loc[fit_mask].copy()

    origin_colors = {
        "in_situ": "#2f6fbb",
        "accreted": "#e68a2e",
    }
    origin_order = ["in_situ", "accreted"]

    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    if not below.empty:
        ax.scatter(
            below["present_mass_msun"],
            below["initial_mass_msun"],
            s=48,
            facecolors="none",
            edgecolors="0.45",
            linewidth=1.1,
            label=rf"$M_{{\rm ini}}<10^4\,{{\rm M}}_\odot$ ({len(below)})",
            zorder=3,
        )

    if "origin_label" in fitted.columns:
        for origin_label in origin_order:
            subset = fitted.loc[fitted["origin_label"] == origin_label]
            if subset.empty:
                continue
            ax.scatter(
                subset["present_mass_msun"],
                subset["initial_mass_msun"],
                s=33,
                color=origin_colors[origin_label],
                edgecolor="white",
                linewidth=0.45,
                alpha=0.88,
                label=f"{origin_label.replace('_', ' ')} ({len(subset)})",
                zorder=4,
            )
        other = fitted.loc[~fitted["origin_label"].isin(origin_order)]
        if not other.empty:
            ax.scatter(
                other["present_mass_msun"],
                other["initial_mass_msun"],
                s=33,
                color="0.35",
                edgecolor="white",
                linewidth=0.45,
                alpha=0.88,
                label=f"other ({len(other)})",
                zorder=4,
            )
    else:
        ax.scatter(
            fitted["present_mass_msun"],
            fitted["initial_mass_msun"],
            s=33,
            color="#2f6fbb",
            edgecolor="white",
            linewidth=0.45,
            alpha=0.88,
            label=rf"$M_{{\rm ini}}\geq10^4\,{{\rm M}}_\odot$ ({len(fitted)})",
            zorder=4,
        )

    all_masses = np.concatenate(
        [
            catalog["present_mass_msun"].to_numpy(dtype=float),
            catalog["initial_mass_msun"].to_numpy(dtype=float),
        ]
    )
    lower = 10.0 ** np.floor(np.log10(np.nanmin(all_masses) * 0.7))
    upper = 10.0 ** np.ceil(np.log10(np.nanmax(all_masses) * 1.3))
    ax.plot([lower, upper], [lower, upper], color="0.25", linestyle="--", linewidth=1.0, label=r"$M_{\rm ini}=M_{\rm now}$")
    ax.axhline(initial_mass_min, color="0.35", linestyle=":", linewidth=1.0)
    ax.text(
        lower * 1.12,
        initial_mass_min * 1.08,
        r"fitted support: $M_{\rm ini}\geq10^4\,{\rm M}_\odot$",
        fontsize=8.2,
        color="0.25",
        va="bottom",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel(r"Current mass, $M_{\rm now}\ [{\rm M}_\odot]$")
    ax.set_ylabel(r"Inferred initial mass, $M_{\rm ini}\ [{\rm M}_\odot]$")
    ax.set_title("Baumgardt GC masses used in the IMF reconstruction")
    ax.grid(alpha=0.18, linewidth=0.7, which="both")
    ax.legend(frameon=False, fontsize=8.0, loc="upper left")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)

    summary: dict[str, object] = {
        "n_catalog": int(len(catalog)),
        "initial_mass_min_msun": float(initial_mass_min),
        "n_in_fitted_initial_mass_support": int(fit_mask.sum()),
        "n_below_fitted_initial_mass_support": int((~fit_mask).sum()),
        "present_mass_min_msun": float(catalog["present_mass_msun"].min()),
        "present_mass_max_msun": float(catalog["present_mass_msun"].max()),
        "initial_mass_min_catalog_msun": float(catalog["initial_mass_msun"].min()),
        "initial_mass_max_catalog_msun": float(catalog["initial_mass_msun"].max()),
        "outputs": {
            "figure_pdf": str(output_pdf),
            "figure_png": str(output_png),
        },
    }
    if "origin_label" in catalog.columns:
        summary["origin_counts_in_fitted_support"] = {
            str(key): int(value)
            for key, value in fitted["origin_label"].value_counts(dropna=False).sort_index().items()
        }
    if not below.empty:
        summary["clusters_below_fitted_initial_mass_support"] = below[
            ["cluster_label", "present_mass_msun", "initial_mass_msun"]
        ].to_dict(orient="records")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--initial-mass-min", type=float, default=1.0e4)
    args = parser.parse_args()

    catalog = _finite_catalog(pd.read_csv(args.catalog))
    summary = _plot(
        catalog=catalog,
        output_pdf=args.output_pdf,
        output_png=args.output_png,
        initial_mass_min=float(args.initial_mass_min),
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
