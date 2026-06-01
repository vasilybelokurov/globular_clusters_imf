from __future__ import annotations

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


def build_soft_survival_grid(
    catalog: pd.DataFrame,
    *,
    selection_offset_dex: float,
    n_radius_grid: int = 160,
    n_mass_grid: int = 180,
    bandwidth_log10_a_dex: float = 0.18,
    softness_dex: float = 0.15,
    global_shift_dex: float = 0.0,
    outer_shift_per_dex: float = 0.0,
    outer_break_log10_a: float = 1.0,
) -> dict[str, object]:
    from globular_clusters_imf.joint_model import build_fixed_survival_grid

    log_a_data = np.log10(catalog["semi_major_axis_kpc"].to_numpy(dtype=float))
    effective_log_cut_data = catalog["log_survival_mass_cut_msun"].to_numpy(dtype=float) + selection_offset_dex
    effective_log_cut_data = effective_log_cut_data + global_shift_dex
    effective_log_cut_data = effective_log_cut_data + outer_shift_per_dex * np.clip(
        log_a_data - outer_break_log10_a,
        0.0,
        None,
    )

    base_grid = build_fixed_survival_grid(
        catalog,
        selection_offset_dex=selection_offset_dex,
        n_radius_grid=n_radius_grid,
        n_mass_grid=n_mass_grid,
        bandwidth_log10_a_dex=bandwidth_log10_a_dex,
    )
    log_a_grid = np.asarray(base_grid["log_a_grid"], dtype=float)
    log_mass_grid = np.asarray(base_grid["log_mass_grid"], dtype=float)

    weights = np.exp(
        -0.5 * np.square((log_a_grid[:, None] - log_a_data[None, :]) / bandwidth_log10_a_dex)
    )
    weights /= np.clip(weights.sum(axis=1, keepdims=True), 1.0e-12, None)

    logits = (log_mass_grid[:, None] - effective_log_cut_data[None, :]) / softness_dex
    indicators = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    survival_probability = indicators @ weights.T
    return {
        "log_mass_grid": log_mass_grid,
        "log_a_grid": log_a_grid,
        "semi_major_axis_grid_kpc": np.power(10.0, log_a_grid),
        "survival_probability": np.clip(survival_probability, 1.0e-12, 1.0),
        "selection_offset_dex": selection_offset_dex,
        "bandwidth_log10_a_dex": bandwidth_log10_a_dex,
        "softness_dex": softness_dex,
        "global_shift_dex": global_shift_dex,
        "outer_shift_per_dex": outer_shift_per_dex,
        "outer_break_log10_a": outer_break_log10_a,
    }


def fit_powerlaw_profile_scan(
    context,
    alpha_grid: np.ndarray,
) -> tuple[dict[str, object], pd.DataFrame]:
    from globular_clusters_imf.joint_model import JointModelSpec, fit_single_joint_model_with_fixed_imf_params

    spec = JointModelSpec(imf_family="powerlaw", radial_model="logpoly3")
    rows: list[dict[str, float]] = []
    best_payload = None
    best_logl = -np.inf
    start_radial = None
    for alpha in alpha_grid:
        payload = fit_single_joint_model_with_fixed_imf_params(
            context=context,
            spec=spec,
            fixed_imf_params=np.array([alpha], dtype=float),
            start_radial_params=start_radial,
        )
        logl = float(payload["summary"].log_likelihood)
        rows.append(
            {
                "alpha_dndm": float(alpha),
                "log_likelihood": logl,
                "bic": float(payload["summary"].bic),
                "total_initial_count": float(payload["model"]["total_initial_count"]),
                "selection_fraction": float(payload["model"]["selection_fraction"]),
            }
        )
        start_radial = np.asarray(payload["radial_parameters_raw"], dtype=float)
        if logl > best_logl:
            best_logl = logl
            best_payload = payload
    if best_payload is None:
        raise RuntimeError("Power-law profile scan returned no fit.")
    return best_payload, pd.DataFrame(rows)


def fit_schechter_local(context) -> dict[str, object]:
    from globular_clusters_imf.joint_model import JointModelSpec, fit_single_joint_model

    spec = JointModelSpec(imf_family="schechter", radial_model="logpoly3")
    return fit_single_joint_model(context=context, spec=spec)


def main() -> None:
    project_root = PROJECT_ROOT
    output_root = project_root / "variants" / "powerlaw_survivability_tweak_scan"
    outputs_tables = output_root / "outputs" / "tables"
    outputs_figures = output_root / "outputs" / "figures"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    outputs_figures.mkdir(parents=True, exist_ok=True)

    from globular_clusters_imf.detectability_longitude_model import fit_single_component_detectability_em_with_abs_longitude
    from globular_clusters_imf.joint_model import JointLikelihoodContext, calibrate_fixed_selection_offset_dex
    from globular_clusters_imf.model import fit_catalog_models

    catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    prepared_catalog = fit_catalog_models(catalog, output_root)["catalog"]

    # Fixed detectability from the best current single-component run.
    best_result = fit_single_component_detectability_em_with_abs_longitude(
        prepared_catalog,
        project_root=output_root,
        n_iterations=8,
    )
    fixed_q_grid = (
        best_result["final_effective_completeness_grid"]
        / np.clip(np.max(best_result["final_effective_completeness_grid"]), 1.0e-12, None)
    )
    # Actually use the full effective completeness, not renormalized.
    fixed_q_grid = np.asarray(best_result["final_effective_completeness_grid"], dtype=float)

    selection_offset_dex = calibrate_fixed_selection_offset_dex(prepared_catalog)
    alpha_grid = np.linspace(-2.6, -1.4, 17)
    candidate_rows: list[dict[str, float | str]] = []

    tweak_specs = [
        ("hard_bandwidth", value) for value in (0.12, 0.18, 0.24, 0.30, 0.36)
    ] + [
        ("soft_global_shift", value) for value in (0.0, 0.1, 0.2, 0.3, 0.4)
    ] + [
        ("soft_outer_shift", value) for value in (0.0, 0.2, 0.4, 0.6, 0.8)
    ]

    best_entry = None
    for tweak_name, tweak_value in tweak_specs:
        if tweak_name == "hard_bandwidth":
            from globular_clusters_imf.joint_model import build_fixed_survival_grid

            survival_grid = build_fixed_survival_grid(
                prepared_catalog,
                selection_offset_dex=selection_offset_dex,
                bandwidth_log10_a_dex=float(tweak_value),
            )
        elif tweak_name == "soft_global_shift":
            survival_grid = build_soft_survival_grid(
                prepared_catalog,
                selection_offset_dex=selection_offset_dex,
                bandwidth_log10_a_dex=0.18,
                softness_dex=0.15,
                global_shift_dex=float(tweak_value),
            )
        elif tweak_name == "soft_outer_shift":
            survival_grid = build_soft_survival_grid(
                prepared_catalog,
                selection_offset_dex=selection_offset_dex,
                bandwidth_log10_a_dex=0.18,
                softness_dex=0.15,
                outer_shift_per_dex=float(tweak_value),
                outer_break_log10_a=1.0,
            )
        else:
            raise ValueError(tweak_name)

        base_context = JointLikelihoodContext.from_catalog_and_survival_grid(prepared_catalog, survival_grid)
        selection_grid = np.clip(
            np.asarray(survival_grid["survival_probability"], dtype=float) * fixed_q_grid,
            1.0e-12,
            1.0,
        )
        context = base_context.with_selection_probability_grid(selection_grid)

        schechter_payload = fit_schechter_local(context)
        powerlaw_payload, powerlaw_scan_table = fit_powerlaw_profile_scan(context, alpha_grid)
        best_alpha = float(powerlaw_scan_table.sort_values("log_likelihood", ascending=False).iloc[0]["alpha_dndm"])
        row = {
            "tweak_name": tweak_name,
            "tweak_value": float(tweak_value),
            "schechter_log_likelihood": float(schechter_payload["summary"].log_likelihood),
            "powerlaw_log_likelihood": float(powerlaw_payload["summary"].log_likelihood),
            "delta_log_likelihood_schechter_minus_powerlaw": float(
                schechter_payload["summary"].log_likelihood - powerlaw_payload["summary"].log_likelihood
            ),
            "powerlaw_best_alpha": best_alpha,
            "powerlaw_total_initial_count": float(powerlaw_payload["model"]["total_initial_count"]),
            "powerlaw_selection_fraction": float(powerlaw_payload["model"]["selection_fraction"]),
            "schechter_total_initial_count": float(schechter_payload["model"]["total_initial_count"]),
            "schechter_selection_fraction": float(schechter_payload["model"]["selection_fraction"]),
        }
        candidate_rows.append(row)
        if best_entry is None or row["delta_log_likelihood_schechter_minus_powerlaw"] < best_entry["delta_log_likelihood_schechter_minus_powerlaw"]:
            best_entry = row
        powerlaw_scan_table.to_csv(
            outputs_tables / f"powerlaw_scan_{tweak_name}_{tweak_value:.2f}.csv",
            index=False,
        )
        print(row)

    summary_table = pd.DataFrame(candidate_rows).sort_values(
        ["delta_log_likelihood_schechter_minus_powerlaw", "tweak_name", "tweak_value"]
    )
    summary_table.to_csv(outputs_tables / "powerlaw_survivability_tweak_scan_summary.csv", index=False)
    (outputs_tables / "powerlaw_survivability_tweak_scan_best.json").write_text(json.dumps(best_entry, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for tweak_name, color, label in (
        ("hard_bandwidth", "#1b9e77", "Hard threshold, bandwidth"),
        ("soft_global_shift", "#d95f02", "Soft threshold + global shift"),
        ("soft_outer_shift", "#7570b3", "Soft threshold + outer shift"),
    ):
        subset = summary_table.loc[summary_table["tweak_name"] == tweak_name].sort_values("tweak_value")
        axes[0].plot(
            subset["tweak_value"],
            subset["delta_log_likelihood_schechter_minus_powerlaw"],
            marker="o",
            linewidth=2.0,
            color=color,
            label=label,
        )
        axes[1].plot(
            subset["tweak_value"],
            subset["powerlaw_best_alpha"],
            marker="o",
            linewidth=2.0,
            color=color,
            label=label,
        )
    axes[0].axhline(0.0, color="0.75", linewidth=1.0)
    axes[0].set_xlabel("Tweak value")
    axes[0].set_ylabel(r"$\Delta \log \mathcal{L}$ (Schechter $-$ power law)")
    axes[0].set_title("Power-law competitiveness")
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[1].set_xlabel("Tweak value")
    axes[1].set_ylabel(r"Best power-law $\alpha$")
    axes[1].set_title("Best profiled power-law slope")
    fig.tight_layout()
    fig.savefig(outputs_figures / "powerlaw_survivability_tweak_scan.png", dpi=220)
    plt.close(fig)

    print(json.dumps(best_entry, indent=2))


if __name__ == "__main__":
    main()
