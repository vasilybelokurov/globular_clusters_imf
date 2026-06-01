from __future__ import annotations

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
from scipy.interpolate import RegularGridInterpolator


def _selection_integrand_grid(*, context, model: dict[str, object]) -> np.ndarray:
    imf_density_grid = np.asarray(model["imf_density_grid"], dtype=float)
    radial_density_grid = np.asarray(model["radial_density_grid"], dtype=float)
    selection_grid = np.asarray(context.selection_probability_grid, dtype=float)
    return (
        float(model["total_initial_count"])
        * imf_density_grid[:, None]
        * radial_density_grid[None, :]
        * selection_grid
    )


def _clamp_bounds(
    *,
    grid: np.ndarray,
    lower: float | None,
    upper: float | None,
) -> tuple[float, float] | None:
    lo = float(grid[0]) if lower is None else max(float(lower), float(grid[0]))
    hi = float(grid[-1]) if upper is None else min(float(upper), float(grid[-1]))
    if hi <= lo:
        return None
    return lo, hi


def _integrate_rectangular_region(
    *,
    log_mass_grid: np.ndarray,
    log_a_grid: np.ndarray,
    integrand_grid: np.ndarray,
    log_mass_min: float | None = None,
    log_mass_max: float | None = None,
    log_a_min: float | None = None,
    log_a_max: float | None = None,
) -> float:
    mass_bounds = _clamp_bounds(grid=log_mass_grid, lower=log_mass_min, upper=log_mass_max)
    a_bounds = _clamp_bounds(grid=log_a_grid, lower=log_a_min, upper=log_a_max)
    if mass_bounds is None or a_bounds is None:
        return 0.0
    m_lo, m_hi = mass_bounds
    a_lo, a_hi = a_bounds
    n_mass = max(64, 2 * int(np.ceil((m_hi - m_lo) / max(float(np.diff(log_mass_grid).mean()), 1.0e-6))) + 1)
    n_a = max(64, 2 * int(np.ceil((a_hi - a_lo) / max(float(np.diff(log_a_grid).mean()), 1.0e-6))) + 1)
    mass_support = np.linspace(m_lo, m_hi, n_mass)
    a_support = np.linspace(a_lo, a_hi, n_a)
    mass_mesh, a_mesh = np.meshgrid(mass_support, a_support, indexing="ij")
    interpolator = RegularGridInterpolator(
        (np.asarray(log_mass_grid, dtype=float), np.asarray(log_a_grid, dtype=float)),
        np.asarray(integrand_grid, dtype=float),
        bounds_error=False,
        fill_value=0.0,
    )
    values = interpolator(np.column_stack([mass_mesh.ravel(), a_mesh.ravel()])).reshape(mass_mesh.shape)
    return float(np.trapezoid(np.trapezoid(values, a_support, axis=1), mass_support))


def _mask_data_in_region(
    *,
    log_mass_data: np.ndarray,
    log_a_data: np.ndarray,
    log_mass_min: float | None = None,
    log_mass_max: float | None = None,
    log_a_min: float | None = None,
    log_a_max: float | None = None,
) -> np.ndarray:
    mask = np.ones(len(log_mass_data), dtype=bool)
    if log_mass_min is not None:
        mask &= log_mass_data >= float(log_mass_min)
    if log_mass_max is not None:
        mask &= log_mass_data <= float(log_mass_max)
    if log_a_min is not None:
        mask &= log_a_data >= float(log_a_min)
    if log_a_max is not None:
        mask &= log_a_data <= float(log_a_max)
    return mask


def _local_log_likelihood_terms(
    *,
    context,
    model: dict[str, object],
    log_mass_min: float | None = None,
    log_mass_max: float | None = None,
    log_a_min: float | None = None,
    log_a_max: float | None = None,
) -> dict[str, float]:
    log_mass_data = np.asarray(context.log_mass_data, dtype=float)
    log_a_data = np.asarray(context.log_a_data, dtype=float)
    mask = _mask_data_in_region(
        log_mass_data=log_mass_data,
        log_a_data=log_a_data,
        log_mass_min=log_mass_min,
        log_mass_max=log_mass_max,
        log_a_min=log_a_min,
        log_a_max=log_a_max,
    )
    selection_data = np.clip(
        context.selection_interpolator(np.column_stack([log_mass_data, log_a_data])),
        1.0e-300,
        1.0,
    )
    imf_density_data = np.clip(np.asarray(model["imf_density_data"], dtype=float), 1.0e-300, None)
    radial_density_data = np.clip(np.asarray(model["radial_density_data"], dtype=float), 1.0e-300, None)
    total_initial_count = float(model["total_initial_count"])

    shape_term = float(
        int(mask.sum()) * np.log(total_initial_count)
        + np.sum(np.log(imf_density_data[mask]))
        + np.sum(np.log(radial_density_data[mask]))
    )
    selection_term = float(np.sum(np.log(selection_data[mask])))
    data_term = shape_term + selection_term
    selection_fraction_region = _integrate_rectangular_region(
        log_mass_grid=np.asarray(context.log_mass_grid, dtype=float),
        log_a_grid=np.asarray(context.log_a_grid, dtype=float),
        integrand_grid=_selection_integrand_grid(context=context, model=model) / max(total_initial_count, 1.0e-300),
        log_mass_min=log_mass_min,
        log_mass_max=log_mass_max,
        log_a_min=log_a_min,
        log_a_max=log_a_max,
    )
    normalization_term = float(-total_initial_count * selection_fraction_region)
    return {
        "n_obs": int(mask.sum()),
        "shape_term": shape_term,
        "selection_logsum_term": selection_term,
        "data_term": data_term,
        "normalization_term": normalization_term,
        "log_likelihood": data_term + normalization_term,
        "selection_fraction_region": float(selection_fraction_region),
    }


def _full_log_likelihood_terms(*, context, model: dict[str, object], log_likelihood: float) -> dict[str, float]:
    log_mass_data = np.asarray(context.log_mass_data, dtype=float)
    log_a_data = np.asarray(context.log_a_data, dtype=float)
    selection_data = np.clip(
        context.selection_interpolator(np.column_stack([log_mass_data, log_a_data])),
        1.0e-300,
        1.0,
    )
    imf_density_data = np.clip(np.asarray(model["imf_density_data"], dtype=float), 1.0e-300, None)
    radial_density_data = np.clip(np.asarray(model["radial_density_data"], dtype=float), 1.0e-300, None)
    total_initial_count = float(model["total_initial_count"])
    n_obs = len(log_mass_data)
    shape_term = float(
        n_obs * np.log(total_initial_count)
        + np.sum(np.log(imf_density_data))
        + np.sum(np.log(radial_density_data))
    )
    selection_term = float(np.sum(np.log(selection_data)))
    data_term = shape_term + selection_term
    normalization_term = float(-total_initial_count * float(model["selection_fraction"]))
    return {
        "n_obs": int(n_obs),
        "shape_term": shape_term,
        "selection_logsum_term": selection_term,
        "data_term": data_term,
        "normalization_term": normalization_term,
        "log_likelihood": float(log_likelihood),
        "selection_fraction_region": float(model["selection_fraction"]),
    }


def _region_label(region_key: str) -> str:
    labels = {
        "full": "Full range",
        "trusted_broad": r"$M_{\rm ini}\geq10^4\,M_\odot,\ a\leq50\,{\rm kpc}$",
        "trusted_strict": r"$M_{\rm ini}\geq10^5\,M_\odot,\ a\leq30\,{\rm kpc}$",
        "outside_trusted_broad": "Outside broad trusted region",
        "outside_trusted_strict": "Outside strict trusted region",
    }
    return labels.get(region_key, region_key)


def _plot_decomposition(table: pd.DataFrame, output_path: Path) -> None:
    eta = np.asarray(table["eta_t"], dtype=float)
    best_idx = int(np.argmax(np.asarray(table["full_log_likelihood"], dtype=float)))
    delta_total = np.asarray(table["full_log_likelihood"], dtype=float) - float(table["full_log_likelihood"].iloc[best_idx])
    delta_data = np.asarray(table["full_data_term"], dtype=float) - float(table["full_data_term"].iloc[best_idx])
    delta_norm = np.asarray(table["full_normalization_term"], dtype=float) - float(
        table["full_normalization_term"].iloc[best_idx]
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(eta, delta_total, color="black", marker="o", linewidth=2.0, label=r"$\Delta\log L$")
    ax.plot(eta, delta_data, color="#1b9e77", marker="o", linewidth=1.6, label=r"$\Delta$ data term")
    ax.plot(eta, delta_norm, color="#d95f02", marker="o", linewidth=1.6, label=r"$\Delta$ normalization term")
    ax.axhline(0.0, color="0.75", linewidth=1.0)
    ax.axvline(float(table["eta_t"].iloc[best_idx]), color="0.6", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"Lifetime multiplier $\eta_t$")
    ax.set_ylabel(r"Change relative to best-fit $\eta_t$")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_restricted_curves(table: pd.DataFrame, output_path: Path) -> None:
    eta = np.asarray(table["eta_t"], dtype=float)
    regions = [
        ("full", "black", "-"),
        ("trusted_broad", "#1b9e77", "-"),
        ("trusted_strict", "#7570b3", "-"),
        ("outside_trusted_broad", "#d95f02", "--"),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for region_key, color, linestyle in regions:
        values = np.asarray(table[f"{region_key}_log_likelihood"], dtype=float)
        delta = values - float(np.max(values))
        ax.plot(
            eta,
            delta,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            marker="o",
            label=_region_label(region_key),
        )
    ax.axhline(0.0, color="0.75", linewidth=1.0)
    ax.set_xlabel(r"Lifetime multiplier $\eta_t$")
    ax.set_ylabel(r"Regional $\Delta\log L$")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    from globular_clusters_imf.detectability_longitude_model import fit_single_component_detectability_em_with_abs_longitude
    from globular_clusters_imf.joint_model import JointModelSpec, imf_parameter_count
    from globular_clusters_imf.model import fit_catalog_models
    from globular_clusters_imf.smooth_survivability import build_smooth_survivability_grid

    output_root = PROJECT_ROOT / "variants" / "schechter_survival_time_multiplier_scan_diagnostics"
    figures_dir = output_root / "outputs" / "figures"
    tables_dir = output_root / "outputs" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    scan_table_path = (
        PROJECT_ROOT
        / "variants"
        / "schechter_survival_time_multiplier_scan"
        / "outputs"
        / "tables"
        / "schechter_best_models_vs_eta_t.csv"
    )
    eta_grid = pd.read_csv(scan_table_path)["eta_t"].to_numpy(dtype=float)

    catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    prepared_catalog = fit_catalog_models(catalog, output_root)["catalog"]

    spec = JointModelSpec(imf_family="schechter", radial_model="step5")
    prior_completeness = None
    prior_radial = None

    region_specs = {
        "full": {},
        "trusted_broad": {"log_mass_min": 4.0, "log_a_max": float(np.log10(50.0))},
        "trusted_strict": {"log_mass_min": 5.0, "log_a_max": float(np.log10(30.0))},
    }

    rows: list[dict[str, object]] = []
    for eta_t in eta_grid:
        smooth_survival = build_smooth_survivability_grid(prepared_catalog, eta_t=float(eta_t))
        survival_grid_override = {
            "log_mass_grid": np.asarray(smooth_survival["log_mass_grid"], dtype=float),
            "log_a_grid": np.asarray(smooth_survival["log_a_grid"], dtype=float),
            "semi_major_axis_grid_kpc": np.asarray(smooth_survival["semi_major_axis_grid_kpc"], dtype=float),
            "survival_probability": np.asarray(smooth_survival["survival_probability"], dtype=float),
            "selection_offset_dex": 0.0,
            "bandwidth_log10_a_dex": float(smooth_survival["bandwidth_log10_a_dex"]),
            "smooth_survivability_summary": smooth_survival["summary"],
        }
        result = fit_single_component_detectability_em_with_abs_longitude(
            prepared_catalog,
            project_root=output_root,
            spec=spec,
            n_iterations=12,
            start_completeness_raw_parameters=prior_completeness,
            start_radial_params=prior_radial,
            survival_grid_override=survival_grid_override,
        )
        n_imf = imf_parameter_count(spec.imf_family)
        prior_completeness = np.asarray(result["final_completeness_raw_parameters"], dtype=float)
        prior_radial = np.asarray(result["final_payload"]["raw_parameters"][n_imf:], dtype=float)

        model = result["final_payload"]["model"]
        context = result["final_context"]
        row: dict[str, object] = {
            "eta_t": float(eta_t),
            "log_likelihood": float(result["final_payload"]["summary"].log_likelihood),
            "alpha_dndm": float(model["imf_parameters"]["alpha_dndm"]),
            "log10_m_c_msun": float(model["imf_parameters"]["log10_m_c_msun"]),
            "total_initial_count": float(model["total_initial_count"]),
            "selection_fraction": float(model["selection_fraction"]),
            "raw_survival_fraction": float(model["raw_survival_fraction"]),
            "mean_detectability": float(
                model["selection_fraction"] / max(model["raw_survival_fraction"], 1.0e-12)
            ),
        }
        full_terms = _full_log_likelihood_terms(
            context=context,
            model=model,
            log_likelihood=float(result["final_payload"]["summary"].log_likelihood),
        )
        row["full_n_obs"] = full_terms["n_obs"]
        row["full_shape_term"] = full_terms["shape_term"]
        row["full_selection_logsum_term"] = full_terms["selection_logsum_term"]
        row["full_data_term"] = full_terms["data_term"]
        row["full_normalization_term"] = full_terms["normalization_term"]
        row["full_log_likelihood"] = full_terms["log_likelihood"]
        row["full_selection_fraction_region"] = full_terms["selection_fraction_region"]

        for region_key, bounds in region_specs.items():
            if region_key == "full":
                continue
            region_terms = _local_log_likelihood_terms(context=context, model=model, **bounds)
            for key, value in region_terms.items():
                row[f"{region_key}_{key}"] = value

            row[f"outside_{region_key}_n_obs"] = full_terms["n_obs"] - region_terms["n_obs"]
            row[f"outside_{region_key}_shape_term"] = full_terms["shape_term"] - region_terms["shape_term"]
            row[f"outside_{region_key}_selection_logsum_term"] = (
                full_terms["selection_logsum_term"] - region_terms["selection_logsum_term"]
            )
            row[f"outside_{region_key}_data_term"] = full_terms["data_term"] - region_terms["data_term"]
            row[f"outside_{region_key}_normalization_term"] = (
                full_terms["normalization_term"] - region_terms["normalization_term"]
            )
            row[f"outside_{region_key}_log_likelihood"] = (
                full_terms["log_likelihood"] - region_terms["log_likelihood"]
            )
            row[f"outside_{region_key}_selection_fraction_region"] = (
                full_terms["selection_fraction_region"] - region_terms["selection_fraction_region"]
            )

        rows.append(row)
        print(
            f"eta_t={eta_t:.3f} logL={row['log_likelihood']:.3f} "
            f"full_data={row['full_data_term']:.3f} full_norm={row['full_normalization_term']:.3f} "
            f"trusted_broad_logL={row['trusted_broad_log_likelihood']:.3f}"
        )

    table = pd.DataFrame(rows).sort_values("eta_t").reset_index(drop=True)
    best_idx = int(np.argmax(np.asarray(table["full_log_likelihood"], dtype=float)))
    best_eta = float(table["eta_t"].iloc[best_idx])
    for prefix in [
        "full",
        "trusted_broad",
        "trusted_strict",
        "outside_trusted_broad",
        "outside_trusted_strict",
    ]:
        values = np.asarray(table[f"{prefix}_log_likelihood"], dtype=float)
        table[f"{prefix}_delta_log_likelihood_from_region_best"] = float(np.max(values)) - values
    table["delta_log_likelihood_from_global_best"] = float(table["log_likelihood"].max()) - np.asarray(
        table["log_likelihood"], dtype=float
    )
    table.to_csv(tables_dir / "eta_t_plateau_diagnostics.csv", index=False)

    _plot_decomposition(table, figures_dir / "eta_t_logl_decomposition.png")
    _plot_restricted_curves(table, figures_dir / "eta_t_restricted_logl.png")

    summary = {
        "best_eta_t_full_log_likelihood": best_eta,
        "eta_grid": eta_grid.tolist(),
        "region_definitions": {
            "trusted_broad": {"log10_m_ini_min_msun": 4.0, "a_max_kpc": 50.0},
            "trusted_strict": {"log10_m_ini_min_msun": 5.0, "a_max_kpc": 30.0},
        },
        "best_row": json.loads(table.iloc[best_idx].to_json()),
    }
    (tables_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(figures_dir / "eta_t_logl_decomposition.png")
    print(figures_dir / "eta_t_restricted_logl.png")
    print(tables_dir / "eta_t_plateau_diagnostics.csv")


if __name__ == "__main__":
    main()
