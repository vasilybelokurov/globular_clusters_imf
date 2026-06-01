from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .detectability_longitude_model import fit_single_component_detectability_em_with_abs_longitude
from .detectability_model import build_detectability_corrected_performance_row, fit_single_component_detectability_em
from .joint_model import JointModelSpec, compute_profile_likelihood_imf_band
from .model import fit_catalog_models


def fit_detectability_model_for_spec(
    catalog: pd.DataFrame,
    project_root: Path,
    spec: JointModelSpec,
    detectability_variant: str = "baseline",
    survival_grid_override: dict[str, object] | None = None,
) -> dict[str, object]:
    catalog_results = fit_catalog_models(catalog, project_root)
    if detectability_variant == "baseline":
        detectability_result = fit_single_component_detectability_em(
            catalog_results["catalog"],
            project_root,
            spec=spec,
        )
    elif detectability_variant == "abs_longitude":
        detectability_result = fit_single_component_detectability_em_with_abs_longitude(
            catalog_results["catalog"],
            project_root,
            spec=spec,
            survival_grid_override=survival_grid_override,
        )
    else:
        raise ValueError(f"Unsupported detectability_variant={detectability_variant!r}")
    return {
        "catalog_results": catalog_results,
        "detectability_result": detectability_result,
        "performance_row": build_detectability_corrected_performance_row(detectability_result),
    }


def bootstrap_detectability_imf_band(
    catalog: pd.DataFrame,
    output_root: Path,
    spec: JointModelSpec,
    reference_log_mass_grid: np.ndarray,
    detectability_variant: str = "baseline",
    survival_grid_override: dict[str, object] | None = None,
    n_bootstrap: int = 32,
    random_seed: int = 12345,
) -> dict[str, object]:
    rng = np.random.default_rng(random_seed)
    bootstrap_rows: list[dict[str, object]] = []
    imf_curves: list[np.ndarray] = []

    bootstrap_root = output_root / "bootstrap_runs"
    bootstrap_root.mkdir(parents=True, exist_ok=True)

    for bootstrap_index in range(n_bootstrap):
        sampled_indices = rng.integers(0, len(catalog), size=len(catalog))
        sampled_catalog = catalog.iloc[sampled_indices].reset_index(drop=True)
        run_root = bootstrap_root / f"iter_{bootstrap_index:03d}"
        run_root.mkdir(parents=True, exist_ok=True)
        try:
            fit_result = fit_detectability_model_for_spec(
                sampled_catalog,
                run_root,
                spec=spec,
                detectability_variant=detectability_variant,
                survival_grid_override=survival_grid_override,
            )
        except Exception as exc:  # pragma: no cover - defensive logging path
            bootstrap_rows.append(
                {
                    "bootstrap_index": bootstrap_index,
                    "success": False,
                    "error_message": str(exc),
                }
            )
            continue

        detectability_result = fit_result["detectability_result"]
        context = detectability_result["final_context"]
        model = detectability_result["final_payload"]["model"]
        curve = np.interp(
            reference_log_mass_grid,
            np.asarray(context.log_mass_grid, dtype=float),
            np.asarray(model["imf_density_grid"], dtype=float),
        )
        imf_curves.append(curve)
        bootstrap_rows.append(
            {
                "bootstrap_index": bootstrap_index,
                "success": True,
                "log_likelihood": float(detectability_result["final_payload"]["summary"].log_likelihood),
                "aic": float(detectability_result["final_payload"]["summary"].aic),
                "bic": float(detectability_result["final_payload"]["summary"].bic),
                "total_initial_count": float(model["total_initial_count"]),
                "selection_fraction": float(model["selection_fraction"]),
                "raw_survival_fraction": float(model["raw_survival_fraction"]),
            }
        )

    bootstrap_table = pd.DataFrame(bootstrap_rows)
    successful_curves = np.asarray(imf_curves, dtype=float)
    if len(successful_curves) == 0:
        raise RuntimeError("No bootstrap fits completed successfully for the flexible IMF model.")

    band_low, band_median, band_high = np.quantile(successful_curves, [0.16, 0.5, 0.84], axis=0)
    band_table = pd.DataFrame(
        {
            "log_initial_mass_msun": np.asarray(reference_log_mass_grid, dtype=float),
            "imf_band_low": band_low,
            "imf_band_median": band_median,
            "imf_band_high": band_high,
        }
    )
    return {
        "bootstrap_table": bootstrap_table,
        "band_table": band_table,
        "n_successful_bootstraps": int(len(successful_curves)),
    }


def build_profile_vs_flexible_imf_comparison(
    catalog: pd.DataFrame,
    output_root: Path,
    detectability_variant: str = "baseline",
    survival_grid_override: dict[str, object] | None = None,
    n_bootstrap: int = 32,
    random_seed: int = 12345,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    schechter_spec = JointModelSpec(imf_family="schechter", radial_model="logpoly3")
    flexible_spec = JointModelSpec(imf_family="logspline6", radial_model="logpoly3")

    schechter_fit = fit_detectability_model_for_spec(
        catalog,
        output_root / "schechter_profile",
        schechter_spec,
        detectability_variant=detectability_variant,
        survival_grid_override=survival_grid_override,
    )
    flexible_fit = fit_detectability_model_for_spec(
        catalog,
        output_root / "logspline6_bootstrap",
        flexible_spec,
        detectability_variant=detectability_variant,
        survival_grid_override=survival_grid_override,
    )

    schechter_result = schechter_fit["detectability_result"]
    flexible_result = flexible_fit["detectability_result"]
    reference_log_mass_grid = np.asarray(schechter_result["final_context"].log_mass_grid, dtype=float)

    schechter_profile_band = compute_profile_likelihood_imf_band(
        best_payload=schechter_result["final_payload"],
        context=schechter_result["final_context"],
        log_mass_support=reference_log_mass_grid[np.unique(np.linspace(0, len(reference_log_mass_grid) - 1, 31, dtype=int))],
    )
    profile_band_table = interpolate_profile_band_to_grid(
        reference_log_mass_grid=reference_log_mass_grid,
        profile_band=schechter_profile_band,
    )

    bootstrap_result = bootstrap_detectability_imf_band(
        catalog=catalog,
        output_root=output_root / "logspline6_bootstrap",
        spec=flexible_spec,
        reference_log_mass_grid=reference_log_mass_grid,
        detectability_variant=detectability_variant,
        survival_grid_override=survival_grid_override,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )

    comparison_summary = pd.DataFrame(
        [
            {
                "model_label": "schechter_profile",
                **schechter_fit["performance_row"],
            },
            {
                "model_label": "logspline6_bootstrap",
                **flexible_fit["performance_row"],
            },
        ]
    )

    plot_profile_vs_bootstrap_imf_comparison(
        reference_log_mass_grid=reference_log_mass_grid,
        schechter_best_curve=np.asarray(schechter_result["final_payload"]["model"]["imf_density_grid"], dtype=float),
        schechter_profile_table=profile_band_table,
        flexible_best_curve=np.asarray(flexible_result["final_payload"]["model"]["imf_density_grid"], dtype=float),
        flexible_bootstrap_table=bootstrap_result["band_table"],
        output_path=output_root / "outputs" / "figures" / "imf_profile_vs_logspline_bootstrap_comparison.pdf",
    )
    plot_profile_vs_bootstrap_imf_comparison(
        reference_log_mass_grid=reference_log_mass_grid,
        schechter_best_curve=np.asarray(schechter_result["final_payload"]["model"]["imf_density_grid"], dtype=float),
        schechter_profile_table=profile_band_table,
        flexible_best_curve=np.asarray(flexible_result["final_payload"]["model"]["imf_density_grid"], dtype=float),
        flexible_bootstrap_table=bootstrap_result["band_table"],
        output_path=output_root / "outputs" / "figures" / "imf_profile_vs_logspline_bootstrap_comparison.png",
    )

    outputs_tables = output_root / "outputs" / "tables"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    comparison_summary.to_csv(outputs_tables / "imf_profile_vs_logspline_model_comparison.csv", index=False)
    profile_band_table.to_csv(outputs_tables / "schechter_profile_likelihood_imf_band.csv", index=False)
    bootstrap_result["band_table"].to_csv(outputs_tables / "logspline6_bootstrap_imf_band.csv", index=False)
    bootstrap_result["bootstrap_table"].to_csv(outputs_tables / "logspline6_bootstrap_summary.csv", index=False)

    summary_payload = {
        "detectability_variant": detectability_variant,
        "n_bootstrap": int(n_bootstrap),
        "n_successful_bootstraps": int(bootstrap_result["n_successful_bootstraps"]),
        "schechter_profile_model": comparison_summary.iloc[0].to_dict(),
        "logspline6_bootstrap_model": comparison_summary.iloc[1].to_dict(),
    }
    (outputs_tables / "imf_profile_vs_logspline_summary.json").write_text(
        json.dumps(summary_payload, indent=2, default=float)
    )

    return {
        "comparison_summary": comparison_summary,
        "schechter_fit": schechter_fit,
        "flexible_fit": flexible_fit,
        "schechter_profile_band_table": profile_band_table,
        "flexible_bootstrap_band_table": bootstrap_result["band_table"],
        "bootstrap_summary_table": bootstrap_result["bootstrap_table"],
        "summary_payload": summary_payload,
    }


def interpolate_profile_band_to_grid(
    reference_log_mass_grid: np.ndarray,
    profile_band: dict[str, np.ndarray],
) -> pd.DataFrame:
    valid = (
        np.isfinite(profile_band["lower_density"])
        & np.isfinite(profile_band["upper_density"])
        & (profile_band["lower_density"] > 0.0)
        & (profile_band["upper_density"] > 0.0)
    )
    support = np.asarray(profile_band["log_mass_support"], dtype=float)[valid]
    lower = np.asarray(profile_band["lower_density"], dtype=float)[valid]
    best = np.asarray(profile_band["best_density"], dtype=float)[valid]
    upper = np.asarray(profile_band["upper_density"], dtype=float)[valid]
    return pd.DataFrame(
        {
            "log_initial_mass_msun": np.asarray(reference_log_mass_grid, dtype=float),
            "imf_band_low": np.power(
                10.0,
                np.interp(reference_log_mass_grid, support, np.log10(lower)),
            ),
            "imf_band_best": np.power(
                10.0,
                np.interp(reference_log_mass_grid, support, np.log10(best)),
            ),
            "imf_band_high": np.power(
                10.0,
                np.interp(reference_log_mass_grid, support, np.log10(upper)),
            ),
        }
    )


def plot_profile_vs_bootstrap_imf_comparison(
    reference_log_mass_grid: np.ndarray,
    schechter_best_curve: np.ndarray,
    schechter_profile_table: pd.DataFrame,
    flexible_best_curve: np.ndarray,
    flexible_bootstrap_table: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        nrows=2,
        figsize=(8.4, 7.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.4]},
    )

    ax = axes[0]
    ax.fill_between(
        schechter_profile_table["log_initial_mass_msun"],
        schechter_profile_table["imf_band_low"],
        schechter_profile_table["imf_band_high"],
        color="#d95f02",
        alpha=0.22,
        linewidth=0.0,
        label=r"Schechter profile $1\sigma$",
    )
    ax.plot(
        reference_log_mass_grid,
        schechter_best_curve,
        color="black",
        linewidth=2.2,
        label="Best Schechter fit",
    )
    ax.fill_between(
        flexible_bootstrap_table["log_initial_mass_msun"],
        flexible_bootstrap_table["imf_band_low"],
        flexible_bootstrap_table["imf_band_high"],
        color="#1b9e77",
        alpha=0.20,
        linewidth=0.0,
        label=r"Flexible IMF bootstrap $1\sigma$",
    )
    ax.plot(
        reference_log_mass_grid,
        flexible_best_curve,
        color="#1b9e77",
        linewidth=2.0,
        linestyle="--",
        label="Best flexible IMF fit",
    )
    ax.set_ylabel("Intrinsic IMF density per dex")
    ax.set_title("Schechter profile band versus flexible-IMF bootstrap band")
    ax.legend(frameon=False, fontsize=9)

    schechter_frac = (
        schechter_profile_table["imf_band_high"] - schechter_profile_table["imf_band_low"]
    ) / (2.0 * np.clip(schechter_profile_table["imf_band_best"], 1.0e-12, None))
    flexible_frac = (
        flexible_bootstrap_table["imf_band_high"] - flexible_bootstrap_table["imf_band_low"]
    ) / (2.0 * np.clip(flexible_bootstrap_table["imf_band_median"], 1.0e-12, None))
    axes[1].plot(
        reference_log_mass_grid,
        schechter_frac,
        color="#d95f02",
        linewidth=2.0,
        label="Schechter profile band",
    )
    axes[1].plot(
        reference_log_mass_grid,
        flexible_frac,
        color="#1b9e77",
        linewidth=2.0,
        linestyle="--",
        label="Flexible bootstrap band",
    )
    axes[1].set_xlabel(r"$\log_{10}(M_{\rm ini}/{\rm M}_\odot)$")
    axes[1].set_ylabel(r"Half-width / $\phi$")
    axes[1].set_ylim(bottom=0.0)
    axes[1].legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220 if output_path.suffix.lower() == ".png" else None)
    plt.close(fig)
