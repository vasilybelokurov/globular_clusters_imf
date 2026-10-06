from __future__ import annotations

import argparse
import json
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


DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "tables" / "gg23_initial_mass_estimates_long.csv"
DEFAULT_OUTPUT_PDF = PROJECT_ROOT / "outputs" / "figures" / "initial_vs_current_mass_all_models.pdf"
DEFAULT_OUTPUT_PNG = PROJECT_ROOT / "outputs" / "figures" / "initial_vs_current_mass_all_models.png"
DEFAULT_SUMMARY = PROJECT_ROOT / "outputs" / "tables" / "initial_vs_current_mass_all_models_summary.json"

MODEL_ORDER = [
    "baumgardt",
    "gg23_no_bh",
    "gg23_bh",
    "gg23_bh_feh_gradient",
    "gg23_bh_past_tidal",
    "gg23_bh_feh_gradient_past_tidal",
]

MODEL_LABELS = {
    "baumgardt": "Baumgardt orbit model",
    "gg23_no_bh": "GG23 no BHs",
    "gg23_bh": "GG23 BHs",
    "gg23_bh_feh_gradient": "GG23 BHs + [Fe/H]",
    "gg23_bh_past_tidal": "GG23 BHs + past tides",
    "gg23_bh_feh_gradient_past_tidal": "GG23 BHs + [Fe/H] + past tides",
}

MODEL_COLORS = {
    "baumgardt": "#222222",
    "gg23_no_bh": "#1b9e77",
    "gg23_bh": "#d95f02",
    "gg23_bh_feh_gradient": "#7570b3",
    "gg23_bh_past_tidal": "#e7298a",
    "gg23_bh_feh_gradient_past_tidal": "#66a61e",
}


def build_plot_table(input_table: pd.DataFrame) -> pd.DataFrame:
    required = {
        "cluster_label",
        "present_mass_msun",
        "initial_mass_msun",
        "gg23_model_name",
        "gg23_initial_mass_msun",
    }
    missing = sorted(required.difference(input_table.columns))
    if missing:
        raise ValueError(f"{DEFAULT_INPUT} is missing required columns: {missing}")

    base_columns = ["cluster_label", "present_mass_msun", "initial_mass_msun"]
    if "origin_label" in input_table.columns:
        base_columns.append("origin_label")

    baumgardt = (
        input_table[base_columns]
        .drop_duplicates(subset=["cluster_label"])
        .rename(columns={"initial_mass_msun": "model_initial_mass_msun"})
    )
    baumgardt["model_name"] = "baumgardt"

    gg23 = input_table[
        [
            column
            for column in [
                "cluster_label",
                "present_mass_msun",
                "origin_label",
                "gg23_model_name",
                "gg23_initial_mass_msun",
            ]
            if column in input_table.columns
        ]
    ].rename(
        columns={
            "gg23_model_name": "model_name",
            "gg23_initial_mass_msun": "model_initial_mass_msun",
        }
    )

    table = pd.concat([baumgardt, gg23], ignore_index=True, sort=False)
    finite = (
        np.isfinite(table["present_mass_msun"].to_numpy(dtype=float))
        & np.isfinite(table["model_initial_mass_msun"].to_numpy(dtype=float))
        & (table["present_mass_msun"].to_numpy(dtype=float) > 0.0)
        & (table["model_initial_mass_msun"].to_numpy(dtype=float) > 0.0)
    )
    table = table.loc[finite].copy()
    table["model_label"] = table["model_name"].map(MODEL_LABELS).fillna(table["model_name"])
    table["mass_ratio_to_current"] = table["model_initial_mass_msun"] / table["present_mass_msun"]
    return table


def plot_initial_vs_current(table: pd.DataFrame, output_pdf: Path, output_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 5.7), constrained_layout=True)

    all_masses = np.concatenate(
        [
            table["present_mass_msun"].to_numpy(dtype=float),
            table["model_initial_mass_msun"].to_numpy(dtype=float),
        ]
    )
    lower = np.nanmin(all_masses) / 1.35
    upper = np.nanmax(all_masses) * 1.35

    for model_name in MODEL_ORDER:
        group = table.loc[table["model_name"] == model_name]
        if group.empty:
            continue
        is_baumgardt = model_name == "baumgardt"
        ax.scatter(
            group["present_mass_msun"],
            group["model_initial_mass_msun"],
            s=28 if is_baumgardt else 17,
            facecolors="none" if is_baumgardt else MODEL_COLORS.get(model_name, "0.4"),
            edgecolors=MODEL_COLORS.get(model_name, "0.4") if is_baumgardt else "none",
            alpha=0.85 if is_baumgardt else 0.52,
            linewidths=0.75 if is_baumgardt else 0.0,
            label=f"{MODEL_LABELS.get(model_name, model_name)} ({len(group)})",
            zorder=4 if is_baumgardt else 3,
        )

    ax.plot(
        [lower, upper],
        [lower, upper],
        color="0.25",
        linestyle="--",
        linewidth=1.0,
        label=r"$M_{\rm ini}=M_{\rm now}$",
        zorder=2,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel(r"Current mass, $M_{\rm now}\ [{\rm M}_\odot]$")
    ax.set_ylabel(r"Initial mass estimate, $M_{\rm ini}\ [{\rm M}_\odot]$")
    ax.set_title("Initial-mass prescriptions used for the GCIMF reconstruction")
    ax.grid(alpha=0.16, linewidth=0.7, which="both")
    ax.legend(frameon=False, fontsize=7.4, loc="lower right", ncol=1)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def build_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_ORDER:
        group = table.loc[table["model_name"] == model_name]
        if group.empty:
            continue
        rows.append(
            {
                "model_name": model_name,
                "model_label": MODEL_LABELS.get(model_name, model_name),
                "n_clusters": int(group["cluster_label"].nunique()),
                "min_initial_mass_msun": float(group["model_initial_mass_msun"].min()),
                "median_initial_mass_msun": float(group["model_initial_mass_msun"].median()),
                "max_initial_mass_msun": float(group["model_initial_mass_msun"].max()),
                "median_initial_to_current_mass_ratio": float(group["mass_ratio_to_current"].median()),
                "n_with_initial_mass_ge_1e4": int((group["model_initial_mass_msun"] >= 1.0e4).sum()),
                "n_with_initial_mass_ge_1e6": int((group["model_initial_mass_msun"] >= 1.0e6).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    long_table = pd.read_csv(args.input)
    plot_table = build_plot_table(long_table)
    plot_initial_vs_current(plot_table, args.output_pdf, args.output_png)

    summary_table = build_summary(plot_table)
    summary = {
        "input": str(args.input),
        "n_clusters": int(plot_table["cluster_label"].nunique()),
        "n_mass_estimates": int(len(plot_table)),
        "outputs": {
            "figure_pdf": str(args.output_pdf),
            "figure_png": str(args.output_png),
        },
        "by_model": summary_table.to_dict(orient="records"),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
