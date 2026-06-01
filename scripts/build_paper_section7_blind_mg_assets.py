from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    project_root = PROJECT_ROOT
    paper_dir = project_root / "paper"
    figures_dir = paper_dir / "figures"
    tables_dir = paper_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    shared_root = project_root / "variants" / "blind_mg_only_minimal_mixture" / "outputs" / "tables"
    split_root = project_root / "variants" / "blind_mg_only_split_alpha_minimal_mixture" / "outputs" / "tables"

    shared_summary = pd.read_csv(shared_root / "blind_mg_only_minimal_mixture_model_summary.csv").set_index("model_class")
    shared_compare = json.loads((shared_root / "blind_mg_only_minimal_mixture_comparison_summary.json").read_text())
    shared_radial = pd.read_csv(shared_root / "blind_mg_only_minimal_mixture_best_model_component_radial_grid.csv")
    shared_mg_density = pd.read_csv(shared_root / "blind_mg_only_minimal_mixture_best_model_mg_density_grid.csv")
    shared_posterior = pd.read_csv(shared_root / "blind_mg_only_minimal_mixture_best_model_posterior_probabilities.csv")

    split_summary = pd.read_csv(split_root / "blind_mg_only_split_alpha_minimal_mixture_model_summary.csv").set_index("model_class")
    split_compare = json.loads((split_root / "blind_mg_only_split_alpha_minimal_mixture_comparison_summary.json").read_text())
    split_imf = pd.read_csv(split_root / "blind_mg_only_split_alpha_minimal_mixture_split_alpha_imf_grid.csv")
    split_posterior = pd.read_csv(split_root / "blind_mg_only_split_alpha_minimal_mixture_split_alpha_posterior_probabilities.csv")

    plot_blind_mg_figure(
        shared_summary=shared_summary,
        shared_compare=shared_compare,
        shared_radial=shared_radial,
        shared_mg_density=shared_mg_density,
        shared_posterior=shared_posterior,
        split_imf=split_imf,
        split_posterior=split_posterior,
        output_path=figures_dir / "blind_mg_two_component_results.pdf",
    )

    comparison_table = build_blind_mg_comparison_table(split_summary, shared_compare, split_compare)
    comparison_table.to_csv(tables_dir / "blind_mg_model_comparison.csv", index=False)
    write_blind_mg_table_tex(comparison_table, tables_dir / "blind_mg_model_comparison.tex")
    write_blind_mg_macros_tex(
        shared_summary=shared_summary,
        split_summary=split_summary,
        shared_compare=shared_compare,
        split_compare=split_compare,
        output_path=tables_dir / "blind_mg_numbers.tex",
    )


def plot_blind_mg_figure(
    shared_summary: pd.DataFrame,
    shared_compare: dict[str, object],
    shared_radial: pd.DataFrame,
    shared_mg_density: pd.DataFrame,
    shared_posterior: pd.DataFrame,
    split_imf: pd.DataFrame,
    split_posterior: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    ax_mg, ax_radius, ax_imf, ax_mass = axes.flat

    mg_subset = shared_posterior.loc[shared_posterior["has_mgfe"]].copy()
    measured_mg = mg_subset["mgfe_combined"].to_numpy(dtype=float)

    # (a) Mg distribution
    ax_mg.hist(
        measured_mg,
        bins=12,
        density=True,
        color="0.88",
        edgecolor="0.45",
        linewidth=0.8,
        label="Measured Mg sample",
    )
    single_params = json.loads(shared_summary.loc["single_mg_gaussian", "chemistry_parameters_json"])
    mg_grid = shared_mg_density["mgfe"].to_numpy(dtype=float)
    single_density = (
        1.0
        / (single_params["sigma_mgfe"] * np.sqrt(2.0 * np.pi))
        * np.exp(-0.5 * ((mg_grid - single_params["mu_mgfe"]) / single_params["sigma_mgfe"]) ** 2)
    )
    ax_mg.plot(mg_grid, single_density, color="0.35", lw=1.8, ls=":", label="Single-Gaussian null")
    ax_mg.plot(
        mg_grid,
        shared_mg_density["mean_weighted_mixture_density"].to_numpy(dtype=float),
        color="black",
        lw=2.2,
        label="Shared-IMF 2-component mixture",
    )
    ax_mg.plot(
        mg_grid,
        shared_mg_density["concentrated_density"].to_numpy(dtype=float),
        color="#d95f02",
        lw=1.9,
        ls="--",
        label="Concentrated Mg component",
    )
    ax_mg.plot(
        mg_grid,
        shared_mg_density["extended_density"].to_numpy(dtype=float),
        color="#1b9e77",
        lw=1.9,
        ls="--",
        label="Extended Mg component",
    )
    ax_mg.set_xlabel(r"[Mg/Fe]")
    ax_mg.set_ylabel("Density")
    ax_mg.set_title("Mg-only blind split with shared IMF")
    ax_mg.text(0.03, 0.96, "(a)", transform=ax_mg.transAxes, ha="left", va="top")
    ax_mg.text(
        0.97,
        0.06,
        r"$\Delta {\rm BIC}=-5.8$ vs. 1-component Mg",
        transform=ax_mg.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    ax_mg.legend(frameon=False, fontsize=8.0, loc="upper left")

    # (b) Radius-dependent mixing and posterior
    radial_subset = mg_subset.sort_values("semi_major_axis_kpc")
    marker_map = {
        1: ("o", "#d95f02", "BK in-situ"),
        0: ("s", "#1b9e77", "BK accreted"),
    }
    for origin_flag, (marker, color, label) in marker_map.items():
        subset = radial_subset.loc[radial_subset["origin_flag"] == origin_flag]
        if subset.empty:
            continue
        ax_radius.scatter(
            subset["semi_major_axis_kpc"],
            subset["p_concentrated"],
            s=24,
            marker=marker,
            color=color,
            alpha=0.70,
            edgecolor="none",
            label=label,
        )
    ax_radius.plot(
        shared_radial["semi_major_axis_kpc"],
        shared_radial["w_concentrated"],
        color="black",
        lw=2.1,
        label=r"Mixing law $w(a)$",
    )
    ax_radius.set_xscale("log")
    ax_radius.set_ylim(0.0, 1.0)
    ax_radius.set_xlabel(r"Orbital semimajor axis $a$ [kpc]")
    ax_radius.set_ylabel(r"$P({\rm concentrated})$")
    ax_radius.set_title("Radial mixing implied by Mg")
    ax_radius.text(0.03, 0.96, "(b)", transform=ax_radius.transAxes, ha="left", va="top")
    ax_radius.legend(frameon=False, fontsize=8.0, loc="lower left")

    # (c) Split-alpha IMF comparison
    ax_imf.plot(
        split_imf["log_initial_mass_msun"],
        split_imf["baseline_imf_density"],
        color="0.35",
        lw=2.1,
        label="Shared-IMF baseline",
    )
    ax_imf.plot(
        split_imf["log_initial_mass_msun"],
        split_imf["concentrated_imf_density"],
        color="#d95f02",
        lw=2.0,
        ls="--",
        label=r"Concentrated, $\alpha_{\rm c}$",
    )
    ax_imf.plot(
        split_imf["log_initial_mass_msun"],
        split_imf["extended_imf_density"],
        color="#1b9e77",
        lw=2.0,
        ls="--",
        label=r"Extended, $\alpha_{\rm e}$",
    )
    ax_imf.set_yscale("log")
    ax_imf.set_xlabel(r"$\log_{10}(M_{\rm ini}/{\rm M}_\odot)$")
    ax_imf.set_ylabel("Intrinsic IMF density per dex")
    ax_imf.set_title(r"Conditional split-$\alpha$ extension")
    ax_imf.text(0.03, 0.96, "(c)", transform=ax_imf.transAxes, ha="left", va="top")
    ax_imf.text(
        0.97,
        0.06,
        r"$\Delta {\rm BIC}=-200.4$ vs. shared IMF",
        transform=ax_imf.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    ax_imf.legend(frameon=False, fontsize=8.0, loc="lower left")

    # (d) Mass dependence of split-alpha posterior
    split_plot = split_posterior.copy()
    for origin_flag, (marker, color, label) in marker_map.items():
        subset = split_plot.loc[split_plot["origin_flag"] == origin_flag]
        if subset.empty:
            continue
        ax_mass.scatter(
            subset["log_initial_mass_msun"],
            subset["p_concentrated"],
            s=18,
            marker=marker,
            color=color,
            alpha=0.60,
            edgecolor="none",
            label=label,
        )
    ax_mass.set_xlabel(r"$\log_{10}(M_{\rm ini}/{\rm M}_\odot)$")
    ax_mass.set_ylabel(r"$P({\rm concentrated})$")
    ax_mass.set_ylim(0.0, 1.0)
    ax_mass.set_title(r"Mass sharpening in the split-$\alpha$ model")
    ax_mass.text(0.03, 0.96, "(d)", transform=ax_mass.transAxes, ha="left", va="top")
    ax_mass.legend(frameon=False, fontsize=8.0, loc="best")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def build_blind_mg_comparison_table(
    split_summary: pd.DataFrame,
    shared_compare: dict[str, object],
    split_compare: dict[str, object],
) -> pd.DataFrame:
    shared_best = json.loads(split_summary.loc["two_component_mg_mixture"].to_json())
    split_best = json.loads(split_summary.loc["two_component_mg_split_alpha_mixture"].to_json())
    single_best = json.loads(split_summary.loc["single_mg_gaussian"].to_json())
    return pd.DataFrame(
        [
            {
                "model": "single Mg Gaussian",
                "n_parameters": int(single_best["n_parameters"]),
                "log_likelihood": float(single_best["joint_log_likelihood"]),
                "bic": float(single_best["bic"]),
                "concentrated_fraction": 1.0,
                "bk_auc": np.nan,
                "main_parameters": r"$\mu_{\rm Mg}=0.228$, $\sigma_{\rm Mg}=0.078$",
            },
            {
                "model": "2-component Mg, shared IMF",
                "n_parameters": int(shared_best["n_parameters"]),
                "log_likelihood": float(shared_best["joint_log_likelihood"]),
                "bic": float(shared_best["bic"]),
                "concentrated_fraction": float(shared_best["component_mix_fraction_concentrated"]),
                "bk_auc": float(shared_compare["bk_comparison"]["auc_in_situ_vs_p_concentrated"]),
                "main_parameters": (
                    r"$\mu_{\rm c}=0.272$, $\mu_{\rm e}=0.131$, "
                    r"$\sigma_{\rm Mg}=0.044$"
                ),
            },
            {
                "model": r"2-component Mg, split $\alpha$",
                "n_parameters": int(split_best["n_parameters"]),
                "log_likelihood": float(split_best["joint_log_likelihood"]),
                "bic": float(split_best["bic"]),
                "concentrated_fraction": float(split_best["component_mix_fraction_concentrated"]),
                "bk_auc": float(split_compare["split_alpha_bk_comparison"]["auc_in_situ_vs_p_concentrated"]),
                "main_parameters": (
                    r"$\alpha_{\rm c}=-0.20$, $\alpha_{\rm e}=-0.64$, "
                    r"$\log_{10}(M_{\rm c}/{\rm M}_\odot)=6.300$"
                ),
            },
        ]
    )


def write_blind_mg_table_tex(table: pd.DataFrame, output_path: Path) -> None:
    lines = [
        r"\begin{table*}",
        r"\caption{Blind Mg-informed latent two-component models built on the fixed detectability-corrected single-component baseline. "
        r"The reported $\ln \mathcal{L}_{\rm mix}$ and ${\rm BIC}_{\rm mix}$ belong to this conditional mixture layer rather than to a new full point-process fit. "
        r"Lower BIC is preferred.}",
        r"\label{tab:blind_mg_models}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"Model & $k$ & $\ln \mathcal{L}_{\rm mix}$ & ${\rm BIC}_{\rm mix}$ & $f_{\rm conc}$ & Main parameters \\",
        r"\hline",
    ]
    for row in table.itertuples(index=False):
        auc = "..." if pd.isna(row.bk_auc) else f"{row.bk_auc:.3f}"
        lines.append(
            f"{row.model} & {row.n_parameters:d} & {row.log_likelihood:.2f} & {row.bic:.2f} & "
            f"{row.concentrated_fraction:.3f} & {row.main_parameters} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
    output_path.write_text("\n".join(lines) + "\n")


def write_blind_mg_macros_tex(
    shared_summary: pd.DataFrame,
    split_summary: pd.DataFrame,
    shared_compare: dict[str, object],
    split_compare: dict[str, object],
    output_path: Path,
) -> None:
    shared_row = split_summary.loc["two_component_mg_mixture"]
    split_row = split_summary.loc["two_component_mg_split_alpha_mixture"]
    single_row = split_summary.loc["single_mg_gaussian"]
    shared_params = json.loads(shared_row["chemistry_parameters_json"])
    split_params = json.loads(split_row["chemistry_parameters_json"])
    split_imf_params = json.loads(split_row["imf_parameters_json"])

    log_a_mean = float(split_compare["selection_payload"]["final_model"]["radial_parameters_json"] and json.loads(split_compare["selection_payload"]["final_model"]["radial_parameters_json"])["log_a_mean"])
    log_a_std = float(json.loads(split_compare["selection_payload"]["final_model"]["radial_parameters_json"])["log_a_std"])

    c0 = float(shared_params["c0"])
    c1 = float(shared_params["c1"])
    z_cross = -c0 / c1
    log_a_cross = log_a_mean + log_a_std * z_cross
    a_cross = 10.0 ** log_a_cross

    lines = [
        rf"\providecommand{{\BlindMgNWithMg}}{{{int(shared_row['n_clusters_with_mg'])}}}",
        rf"\providecommand{{\BlindMgSharedDeltaLogL}}{{{shared_row['joint_log_likelihood'] - single_row['joint_log_likelihood']:.2f}}}",
        rf"\providecommand{{\BlindMgSharedDeltaBIC}}{{{shared_row['bic'] - single_row['bic']:.2f}}}",
        rf"\providecommand{{\BlindMgSharedMuConc}}{{{shared_params['mu_mgfe_concentrated']:.3f}}}",
        rf"\providecommand{{\BlindMgSharedMuExt}}{{{shared_params['mu_mgfe_extended']:.3f}}}",
        rf"\providecommand{{\BlindMgSharedSigma}}{{{shared_params['sigma_mgfe_shared']:.3f}}}",
        rf"\providecommand{{\BlindMgSharedFConc}}{{{shared_row['component_mix_fraction_concentrated']:.3f}}}",
        rf"\providecommand{{\BlindMgSharedNConc}}{{{shared_row['component_initial_count_concentrated']:.1f}}}",
        rf"\providecommand{{\BlindMgSharedNExt}}{{{shared_row['component_initial_count_extended']:.1f}}}",
        rf"\providecommand{{\BlindMgSharedAUC}}{{{shared_compare['bk_comparison']['auc_in_situ_vs_p_concentrated']:.3f}}}",
        rf"\providecommand{{\BlindMgSharedAccuracy}}{{{shared_compare['bk_comparison']['hard_assignment_accuracy']:.3f}}}",
        rf"\providecommand{{\BlindMgSharedACross}}{{{a_cross:.1f}}}",
        rf"\providecommand{{\BlindMgSplitDeltaLogL}}{{{split_compare['delta_log_likelihood_split_minus_shared']:.2f}}}",
        rf"\providecommand{{\BlindMgSplitDeltaBIC}}{{{split_compare['delta_bic_split_minus_shared']:.2f}}}",
        rf"\providecommand{{\BlindMgSplitAlphaConc}}{{{split_imf_params['alpha_dndm_concentrated']:.3f}}}",
        rf"\providecommand{{\BlindMgSplitAlphaExt}}{{{split_imf_params['alpha_dndm_extended']:.3f}}}",
        rf"\providecommand{{\BlindMgSplitMc}}{{{split_imf_params['shared_log10_m_c_msun']:.3f}}}",
        rf"\providecommand{{\BlindMgSplitFConc}}{{{split_row['component_mix_fraction_concentrated']:.3f}}}",
        rf"\providecommand{{\BlindMgSplitNConc}}{{{split_row['component_initial_count_concentrated']:.1f}}}",
        rf"\providecommand{{\BlindMgSplitNExt}}{{{split_row['component_initial_count_extended']:.1f}}}",
        rf"\providecommand{{\BlindMgSplitAUC}}{{{split_compare['split_alpha_bk_comparison']['auc_in_situ_vs_p_concentrated']:.3f}}}",
        rf"\providecommand{{\BlindMgSplitAccuracy}}{{{split_compare['split_alpha_bk_comparison']['hard_assignment_accuracy']:.3f}}}",
        rf"\providecommand{{\BlindMgSplitMuConc}}{{{split_params['mu_mgfe_concentrated']:.3f}}}",
        rf"\providecommand{{\BlindMgSplitMuExt}}{{{split_params['mu_mgfe_extended']:.3f}}}",
        rf"\providecommand{{\BlindMgSplitSigma}}{{{split_params['sigma_mgfe_shared']:.3f}}}",
    ]
    output_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
