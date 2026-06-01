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


def _restrict_log_mass_support(
    log_mass_grid: np.ndarray,
    values: np.ndarray,
    log_mass_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(log_mass_grid, dtype=float)
    y = np.asarray(values, dtype=float)
    if log_mass_min <= float(x[0]):
        return x.copy(), y.copy()
    if log_mass_min >= float(x[-1]):
        return np.asarray([float(x[-1])]), np.asarray([float(y[-1])])
    mask = x > log_mass_min
    y0 = float(np.interp(log_mass_min, x, y))
    return np.concatenate(([float(log_mass_min)], x[mask])), np.concatenate(([y0], y[mask]))


def _count_and_mass_above_log_mass(
    *,
    total_initial_count: float,
    log_mass_grid: np.ndarray,
    imf_density_grid: np.ndarray,
    log_mass_min: float,
) -> tuple[float, float]:
    x, y = _restrict_log_mass_support(log_mass_grid, imf_density_grid, log_mass_min)
    number_fraction = float(np.trapezoid(y, x))
    mean_mass_above = float(np.trapezoid(np.power(10.0, x) * y, x) / max(number_fraction, 1.0e-12))
    return float(total_initial_count * number_fraction), float(total_initial_count * number_fraction * mean_mass_above)


def _selection_stats_above_log_mass(
    *,
    context,
    model: dict[str, object],
    log_mass_min: float,
) -> dict[str, float]:
    log_mass_grid = np.asarray(context.log_mass_grid, dtype=float)
    radial_support = np.asarray(context.log_a_grid, dtype=float)
    imf_density_grid = np.asarray(model["imf_density_grid"], dtype=float)
    radial_density_grid = np.asarray(model["radial_density_grid"], dtype=float)
    survival_grid = np.asarray(context.survival_probability_grid, dtype=float)
    selection_grid = np.asarray(context.selection_probability_grid, dtype=float)

    mass_support, imf_support = _restrict_log_mass_support(log_mass_grid, imf_density_grid, log_mass_min)
    survival_support = np.vstack(
        [np.interp(mass_support, log_mass_grid, survival_grid[:, j]) for j in range(survival_grid.shape[1])]
    ).T
    selection_support = np.vstack(
        [np.interp(mass_support, log_mass_grid, selection_grid[:, j]) for j in range(selection_grid.shape[1])]
    ).T
    integrand_base = imf_support[:, None] * radial_density_grid[None, :]
    raw_survival_fraction_above = float(
        np.trapezoid(np.trapezoid(integrand_base * survival_support, radial_support, axis=1), mass_support)
    )
    selection_fraction_above = float(
        np.trapezoid(np.trapezoid(integrand_base * selection_support, radial_support, axis=1), mass_support)
    )
    return {
        "raw_survival_fraction_above_log10_4": raw_survival_fraction_above,
        "selection_fraction_above_log10_4": selection_fraction_above,
        "mean_detectability_above_log10_4": float(
            selection_fraction_above / max(raw_survival_fraction_above, 1.0e-12)
        ),
    }


def _row_from_result(
    *,
    eta_t: float,
    radial_model: str,
    survival_summary,
    result: dict[str, object],
    log_mass_min: float,
) -> dict[str, object]:
    final_payload = result["final_payload"]
    baseline_payload = result["baseline_payload"]
    final_model = final_payload["model"]
    baseline_model = baseline_payload["model"]
    imf_params = final_model["imf_parameters"]
    baseline_count_above, baseline_mass_above = _count_and_mass_above_log_mass(
        total_initial_count=float(baseline_model["total_initial_count"]),
        log_mass_grid=np.asarray(result["base_context"].log_mass_grid, dtype=float),
        imf_density_grid=np.asarray(baseline_model["imf_density_grid"], dtype=float),
        log_mass_min=log_mass_min,
    )
    final_count_above, final_mass_above = _count_and_mass_above_log_mass(
        total_initial_count=float(final_model["total_initial_count"]),
        log_mass_grid=np.asarray(result["base_context"].log_mass_grid, dtype=float),
        imf_density_grid=np.asarray(final_model["imf_density_grid"], dtype=float),
        log_mass_min=log_mass_min,
    )
    above_stats = _selection_stats_above_log_mass(
        context=result["final_context"],
        model=final_model,
        log_mass_min=log_mass_min,
    )
    return {
        "eta_t": float(eta_t),
        "radial_model": radial_model,
        "log_likelihood": float(final_payload["summary"].log_likelihood),
        "aic": float(final_payload["summary"].aic),
        "bic": float(final_payload["summary"].bic),
        "alpha_dndm": float(imf_params.get("alpha_dndm", np.nan)),
        "log10_m_c_msun": float(imf_params.get("log10_m_c_msun", np.nan)),
        "baseline_total_initial_count": float(baseline_model["total_initial_count"]),
        "baseline_total_initial_count_above_log10_4": float(baseline_count_above),
        "baseline_total_initial_stellar_mass_above_log10_4_msun": float(baseline_mass_above),
        "final_total_initial_count": float(final_model["total_initial_count"]),
        "final_total_initial_count_above_log10_4": float(final_count_above),
        "final_total_initial_stellar_mass_above_log10_4_msun": float(final_mass_above),
        "raw_survival_fraction": float(final_model["raw_survival_fraction"]),
        "selection_fraction": float(final_model["selection_fraction"]),
        "mean_detectability": float(result["summary_payload"]["final_mean_detectability"]),
        "raw_survival_fraction_above_log10_4": float(above_stats["raw_survival_fraction_above_log10_4"]),
        "selection_fraction_above_log10_4": float(above_stats["selection_fraction_above_log10_4"]),
        "mean_detectability_above_log10_4": float(above_stats["mean_detectability_above_log10_4"]),
        "count_ratio_vs_baseline_above_log10_4": float(final_count_above / max(baseline_count_above, 1.0e-12)),
        "mass_ratio_vs_baseline_above_log10_4": float(final_mass_above / max(baseline_mass_above, 1.0e-12)),
        "survival_outer_level_90_log10_msun": float(getattr(survival_summary, "outer_level_90_log10_msun")),
        "survival_outer_level_50_log10_msun": float(getattr(survival_summary, "outer_level_50_log10_msun")),
        "survival_outer_level_10_log10_msun": float(getattr(survival_summary, "outer_level_10_log10_msun")),
        "survival_inner_level_90_log10_msun": float(getattr(survival_summary, "inner_level_90_log10_msun")),
        "survival_inner_level_50_log10_msun": float(getattr(survival_summary, "inner_level_50_log10_msun")),
        "survival_inner_level_10_log10_msun": float(getattr(survival_summary, "inner_level_10_log10_msun")),
        "survival_transition_a_kpc": float(getattr(survival_summary, "transition_a_kpc")),
        "survival_width_log10_a_dex": float(getattr(survival_summary, "width_log10_a_dex")),
        "survival_transition_band_width_dex": float(getattr(survival_summary, "transition_band_width_dex")),
    }


def _plot_logl_vs_multiplier(best_table: pd.DataFrame, output_path: Path) -> None:
    eta = np.asarray(best_table["eta_t"], dtype=float)
    logl = np.asarray(best_table["log_likelihood"], dtype=float)
    delta_logl = logl - float(np.max(logl))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(5.6, 5.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.08},
    )
    ax_full, ax_zoom = axes

    handles = []
    labels = []
    for radial_model, color in (("step5", "#d95f02"), ("logpoly3", "#1b9e77"), ("powerlaw_a", "#1f78b4")):
        subset = best_table.loc[best_table["best_radial_model"] == radial_model]
        if len(subset) == 0:
            continue
        handle = ax_full.scatter(
            subset["eta_t"],
            subset["log_likelihood"],
            s=34,
            color=color,
            label=radial_model,
            zorder=3,
        )
        ax_zoom.scatter(
            subset["eta_t"],
            subset["log_likelihood"] - float(np.max(logl)),
            s=34,
            color=color,
            zorder=3,
        )
        handles.append(handle)
        labels.append(radial_model)

    ax_full.plot(eta, logl, color="0.2", linewidth=1.4, zorder=2)
    ax_zoom.plot(eta, delta_logl, color="0.2", linewidth=1.4, zorder=2)

    best_idx = int(np.argmax(logl))
    best_eta = float(eta[best_idx])
    ax_full.axvline(best_eta, color="0.7", linestyle="--", linewidth=1.0, zorder=1)
    ax_zoom.axvline(best_eta, color="0.7", linestyle="--", linewidth=1.0, zorder=1)
    for level in (-0.5, -1.0, -2.0, -3.0, -5.0):
        ax_zoom.axhline(level, color="0.88", linewidth=0.8, zorder=1)

    finite_zoom = delta_logl[np.isfinite(delta_logl)]
    zoom_floor = min(-5.5, float(np.floor(np.min(finite_zoom[finite_zoom > -20.0]) - 0.5)) if np.any(finite_zoom > -20.0) else -5.5)

    ax_full.set_ylabel(r"Best Schechter $\log L$")
    ax_zoom.set_ylabel(r"$\Delta \log L$ from max")
    ax_zoom.set_xlabel(r"Lifetime Multiplier $\eta_t$")
    ax_zoom.set_ylim(zoom_floor, 0.25)

    if handles:
        ax_full.legend(handles, labels, frameon=False, title="Best radial model", loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_properties(best_table: pd.DataFrame, output_path: Path) -> None:
    eta = np.asarray(best_table["eta_t"], dtype=float)
    fig, axes = plt.subplots(3, 2, figsize=(8.8, 9.2))
    ax = axes[0, 0]
    ax.plot(eta, best_table["alpha_dndm"], color="#1f78b4", marker="o")
    ax.set_ylabel(r"$\alpha$")
    ax.set_xlabel(r"$\eta_t$")

    ax = axes[0, 1]
    ax.plot(eta, best_table["log10_m_c_msun"], color="#1f78b4", marker="o")
    ax.set_ylabel(r"$\log_{10}(M_c/{\rm M}_\odot)$")
    ax.set_xlabel(r"$\eta_t$")

    ax = axes[1, 0]
    ax.plot(eta, best_table["baseline_total_initial_count_above_log10_4"], color="0.6", marker="o", label="Baseline")
    ax.plot(eta, best_table["final_total_initial_count_above_log10_4"], color="black", marker="o", label="Detectability-corrected")
    ax.set_yscale("log")
    ax.set_ylabel(r"$N_0(M_{\rm ini}\geq 10^4\,M_\odot)$")
    ax.set_xlabel(r"$\eta_t$")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(
        eta,
        best_table["baseline_total_initial_stellar_mass_above_log10_4_msun"],
        color="0.6",
        marker="o",
        label="Baseline",
    )
    ax.plot(
        eta,
        best_table["final_total_initial_stellar_mass_above_log10_4_msun"],
        color="black",
        marker="o",
        label="Detectability-corrected",
    )
    ax.set_yscale("log")
    ax.set_ylabel(r"$M_{\star,0}(M_{\rm ini}\geq 10^4\,M_\odot)\ [{\rm M}_\odot]$")
    ax.set_xlabel(r"$\eta_t$")

    ax = axes[2, 0]
    ax.plot(
        eta,
        best_table["raw_survival_fraction_above_log10_4"],
        color="#7570b3",
        marker="o",
        label="Raw survival fraction",
    )
    ax.plot(
        eta,
        best_table["selection_fraction_above_log10_4"],
        color="#e7298a",
        marker="o",
        label="Selection fraction",
    )
    ax.set_yscale("log")
    ax.set_ylabel("Fraction above $10^4 M_\\odot$")
    ax.set_xlabel(r"$\eta_t$")
    ax.legend(frameon=False)

    ax = axes[2, 1]
    ax.plot(
        eta,
        best_table["mean_detectability_above_log10_4"],
        color="#66a61e",
        marker="o",
        label=r"$\langle C \rangle$",
    )
    ax2 = ax.twinx()
    ax2.plot(
        eta,
        best_table["count_ratio_vs_baseline_above_log10_4"],
        color="#a6761d",
        marker="s",
        label=r"$N_{\rm corr}/N_{\rm base}$",
    )
    ax.set_ylabel(r"Mean detectability above $10^4 M_\odot$")
    ax2.set_ylabel(r"$N_{\rm corr}/N_{\rm base}$")
    ax.set_xlabel(r"$\eta_t$")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface-model", choices=["compact", "logistic"], default="compact")
    parser.add_argument("--output-tag", type=str, default="")
    args = parser.parse_args()
    surface_model = str(args.surface_model)
    extra_tag = args.output_tag.strip()
    from globular_clusters_imf.detectability_longitude_model import fit_single_component_detectability_em_with_abs_longitude
    from globular_clusters_imf.joint_model import JointModelSpec, imf_parameter_count
    from globular_clusters_imf.model import fit_catalog_models
    from globular_clusters_imf.smooth_survivability import build_smooth_survivability_grid

    variant_name = "schechter_survival_time_multiplier_scan"
    if surface_model != "compact":
        variant_name = f"{variant_name}_{surface_model}"
    if extra_tag:
        variant_name = f"{variant_name}_{extra_tag}"
    output_root = PROJECT_ROOT / "variants" / variant_name
    figures_dir = output_root / "outputs" / "figures"
    tables_dir = output_root / "outputs" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    prepared_catalog = fit_catalog_models(catalog, output_root)["catalog"]

    eta_grid = np.linspace(0.1, 3.0, 30)
    specs = [
        JointModelSpec(imf_family="schechter", radial_model="step5"),
        JointModelSpec(imf_family="schechter", radial_model="logpoly3"),
    ]
    prior_state = {
        spec.radial_model: {"completeness": None, "radial": None}
        for spec in specs
    }
    all_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    best_results: dict[float, dict[str, object]] = {}

    for eta_t in eta_grid:
        smooth_survival = build_smooth_survivability_grid(prepared_catalog, eta_t=float(eta_t), surface_model=surface_model)
        survival_grid_override = {
            "log_mass_grid": np.asarray(smooth_survival["log_mass_grid"], dtype=float),
            "log_a_grid": np.asarray(smooth_survival["log_a_grid"], dtype=float),
            "semi_major_axis_grid_kpc": np.asarray(smooth_survival["semi_major_axis_grid_kpc"], dtype=float),
            "survival_probability": np.asarray(smooth_survival["survival_probability"], dtype=float),
            "selection_offset_dex": 0.0,
            "bandwidth_log10_a_dex": float(smooth_survival["bandwidth_log10_a_dex"]),
            "smooth_survivability_summary": smooth_survival["summary"],
        }
        eta_rows: list[dict[str, object]] = []
        eta_results: dict[str, dict[str, object]] = {}
        for spec in specs:
            state = prior_state[spec.radial_model]
            result = fit_single_component_detectability_em_with_abs_longitude(
                prepared_catalog,
                project_root=output_root,
                spec=spec,
                n_iterations=12,
                start_completeness_raw_parameters=state["completeness"],
                start_radial_params=state["radial"],
                survival_grid_override=survival_grid_override,
            )
            n_imf = imf_parameter_count(spec.imf_family)
            prior_state[spec.radial_model] = {
                "completeness": np.asarray(result["final_completeness_raw_parameters"], dtype=float),
                "radial": np.asarray(result["final_payload"]["raw_parameters"][n_imf:], dtype=float),
            }
            row = _row_from_result(
                eta_t=float(eta_t),
                radial_model=spec.radial_model,
                survival_summary=smooth_survival["summary"],
                result=result,
                log_mass_min=4.0,
            )
            eta_rows.append(row)
            eta_results[spec.radial_model] = result
            print(
                f"eta_t={eta_t:.3f} radial={spec.radial_model} "
                f"logL={row['log_likelihood']:.3f} alpha={row['alpha_dndm']:.3f} "
                f"logMc={row['log10_m_c_msun']:.3f} N0>1e4={row['final_total_initial_count_above_log10_4']:.1f}"
            )

        eta_table = pd.DataFrame(eta_rows).sort_values("log_likelihood", ascending=False).reset_index(drop=True)
        best_row = dict(eta_table.iloc[0])
        best_row["best_radial_model"] = str(best_row["radial_model"])
        best_rows.append(best_row)
        best_results[float(eta_t)] = eta_results[str(best_row["best_radial_model"])]
        all_rows.extend(eta_rows)

    all_table = pd.DataFrame(all_rows).sort_values(["eta_t", "log_likelihood"], ascending=[True, False]).reset_index(drop=True)
    best_table = pd.DataFrame(best_rows).sort_values("eta_t").reset_index(drop=True)
    all_table.to_csv(tables_dir / "schechter_all_models_vs_eta_t.csv", index=False)
    best_table.to_csv(tables_dir / "schechter_best_models_vs_eta_t.csv", index=False)

    _plot_logl_vs_multiplier(best_table, figures_dir / "schechter_logl_vs_eta_t.png")
    _plot_properties(best_table, figures_dir / "schechter_properties_vs_eta_t.png")

    summary_payload = {
        "eta_grid": eta_grid.tolist(),
        "surface_model": surface_model,
        "n_detectability_iterations": 12,
        "model_specs": [{"imf_family": spec.imf_family, "radial_model": spec.radial_model} for spec in specs],
        "best_rows": best_table.to_dict(orient="records"),
        "global_best": json.loads(best_table.sort_values("log_likelihood", ascending=False).iloc[0].to_json()),
    }
    (tables_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2))
    print(figures_dir / "schechter_logl_vs_eta_t.png")
    print(figures_dir / "schechter_properties_vs_eta_t.png")
    print(tables_dir / "summary.json")


if __name__ == "__main__":
    main()
