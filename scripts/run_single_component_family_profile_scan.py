from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def result_to_row(
    family_name: str,
    radial_model: str,
    fixed_imf_params: dict[str, float],
    result: dict[str, object],
) -> dict[str, object]:
    from globular_clusters_imf.detectability_model import build_detectability_corrected_performance_row

    row = build_detectability_corrected_performance_row(result)
    row["scan_family"] = family_name
    row["scan_radial_model"] = radial_model
    for key, value in fixed_imf_params.items():
        row[key] = float(value)
    return row


def fit_fixed_family_point(
    prepared_catalog: pd.DataFrame,
    output_root: Path,
    *,
    spec,
    fixed_imf_params: np.ndarray,
    fixed_imf_param_dict: dict[str, float],
    n_iterations: int,
    start_candidates: list[tuple[np.ndarray | None, np.ndarray | None]],
    survival_grid_override: dict[str, object] | None,
) -> dict[str, object]:
    from globular_clusters_imf.detectability_longitude_model import fit_single_component_detectability_em_with_abs_longitude

    best_result = None
    best_log_likelihood = -np.inf
    for start_completeness_raw_parameters, start_radial_params in start_candidates:
        result = fit_single_component_detectability_em_with_abs_longitude(
            prepared_catalog,
            project_root=output_root,
            spec=spec,
            n_iterations=n_iterations,
            fixed_imf_params=np.asarray(fixed_imf_params, dtype=float),
            start_completeness_raw_parameters=start_completeness_raw_parameters,
            start_radial_params=start_radial_params,
            survival_grid_override=survival_grid_override,
        )
        log_likelihood = float(result["final_payload"]["summary"].log_likelihood)
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_result = result

    if best_result is None:
        raise RuntimeError(f"No successful fit returned for {spec} at {fixed_imf_param_dict}.")

    return {
        "result": best_result,
        "row": result_to_row(
            family_name=spec.imf_family,
            radial_model=spec.radial_model,
            fixed_imf_params=fixed_imf_param_dict,
            result=best_result,
        ),
    }


def run_powerlaw_scan(
    prepared_catalog: pd.DataFrame,
    output_root: Path,
    alpha_grid: np.ndarray,
    n_iterations: int,
    reference_start_completeness: np.ndarray,
    reference_start_radial: np.ndarray,
    survival_grid_override: dict[str, object] | None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    from globular_clusters_imf.joint_model import JointModelSpec

    spec = JointModelSpec(imf_family="powerlaw", radial_model="logpoly3")
    entries: list[dict[str, object]] = []
    warm_result: dict[str, object] | None = None
    for alpha in alpha_grid:
        entry = fit_fixed_family_point(
            prepared_catalog,
            output_root,
            spec=spec,
            fixed_imf_params=np.array([alpha], dtype=float),
            fixed_imf_param_dict={"alpha_dndm": float(alpha)},
            n_iterations=n_iterations,
            start_candidates=[
                (
                    None if warm_result is None else warm_result["result"]["final_completeness_raw_parameters"],
                    None if warm_result is None else warm_result["result"]["final_payload"]["radial_parameters_raw"],
                ),
                (
                    reference_start_completeness,
                    reference_start_radial,
                ),
            ],
            survival_grid_override=survival_grid_override,
        )
        entries.append(entry)
        warm_result = entry
        print(f"powerlaw alpha={alpha:.3f} logL={entry['row']['log_likelihood']:.3f}")
    table = pd.DataFrame([entry["row"] for entry in entries]).sort_values("alpha_dndm").reset_index(drop=True)
    return table, entries


def run_2d_family_scan(
    prepared_catalog: pd.DataFrame,
    output_root: Path,
    *,
    spec,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    param_builder,
    n_iterations: int,
    reference_start_completeness: np.ndarray,
    reference_start_radial: np.ndarray,
    survival_grid_override: dict[str, object] | None,
) -> tuple[pd.DataFrame, list[list[dict[str, object]]]]:
    scan_rows: list[dict[str, object]] = []
    scan_grid: list[list[dict[str, object]]] = []
    previous_row: list[dict[str, object]] | None = None

    for y_value in grid_y:
        current_row: list[dict[str, object]] = []
        for x_index, x_value in enumerate(grid_x):
            left_neighbor = current_row[-1] if current_row else None
            upper_neighbor = None if previous_row is None else previous_row[x_index]
            start_source = left_neighbor if left_neighbor is not None else upper_neighbor
            fixed_imf_params, fixed_imf_param_dict = param_builder(float(x_value), float(y_value))
            entry = fit_fixed_family_point(
                prepared_catalog,
                output_root,
                spec=spec,
                fixed_imf_params=fixed_imf_params,
                fixed_imf_param_dict=fixed_imf_param_dict,
                n_iterations=n_iterations,
                start_candidates=[
                    (
                        None if start_source is None else start_source["result"]["final_completeness_raw_parameters"],
                        None if start_source is None else start_source["result"]["final_payload"]["radial_parameters_raw"],
                    ),
                    (
                        reference_start_completeness,
                        reference_start_radial,
                ),
            ],
                survival_grid_override=survival_grid_override,
            )
            current_row.append(entry)
            scan_rows.append(entry["row"])
            print(
                f"{spec.imf_family} "
                + " ".join(f"{key}={value:.3f}" for key, value in fixed_imf_param_dict.items())
                + f" logL={entry['row']['log_likelihood']:.3f}"
            )
        scan_grid.append(current_row)
        previous_row = current_row

    table = pd.DataFrame(scan_rows)
    return table, scan_grid


def summarize_family_scan(
    family_table: pd.DataFrame,
    unconstrained_row: pd.Series,
) -> dict[str, object]:
    best_row = family_table.sort_values("log_likelihood", ascending=False).iloc[0]
    return {
        "best_scan_point": json.loads(pd.Series(best_row).to_json()),
        "unconstrained_reference": json.loads(pd.Series(unconstrained_row).to_json()),
        "delta_log_likelihood_scan_minus_unconstrained": float(
            float(best_row["log_likelihood"]) - float(unconstrained_row["log_likelihood"])
        ),
        "delta_bic_scan_minus_unconstrained": float(float(best_row["bic"]) - float(unconstrained_row["bic"])),
    }


def save_best_family_result(
    output_tables: Path,
    family_name: str,
    entries: list[dict[str, object]],
) -> dict[str, object]:
    best_entry = max(
        entries,
        key=lambda entry: float(entry["result"]["final_payload"]["summary"].log_likelihood),
    )
    result_path = output_tables / f"{family_name}_best_result.pkl"
    metadata_path = output_tables / f"{family_name}_best_result_summary.json"
    with result_path.open("wb") as handle:
        pickle.dump(best_entry["result"], handle, protocol=pickle.HIGHEST_PROTOCOL)
    metadata_path.write_text(json.dumps(best_entry["row"], indent=2))
    return best_entry


def plot_family_scan_summary(
    output_path: Path,
    powerlaw_table: pd.DataFrame,
    lognormal_table: pd.DataFrame,
    schechter_table: pd.DataFrame,
    unconstrained_lookup: dict[str, pd.Series],
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(ncols=3, figsize=(11.0, 3.5))

    best_powerlaw_logl = float(powerlaw_table["log_likelihood"].max())
    axes[0].plot(
        powerlaw_table["alpha_dndm"],
        powerlaw_table["log_likelihood"] - best_powerlaw_logl,
        color="#1b9e77",
        linewidth=2.0,
    )
    axes[0].axhline(0.0, color="0.75", linewidth=1.0)
    axes[0].set_xlabel(r"$\alpha$")
    axes[0].set_ylabel(r"$\Delta \log L$")
    axes[0].set_title("Power law")

    lognormal_pivot = lognormal_table["log_likelihood"].max()
    x_unique = np.sort(lognormal_table["mu_log10_msun"].unique())
    y_unique = np.sort(lognormal_table["sigma_log10_msun"].unique())
    z_grid = (
        lognormal_table.pivot(index="sigma_log10_msun", columns="mu_log10_msun", values="log_likelihood")
        .sort_index()
        .sort_index(axis=1)
        .to_numpy()
        - lognormal_pivot
    )
    im1 = axes[1].contourf(x_unique, y_unique, z_grid, levels=18, cmap="magma")
    axes[1].plot(
        float(unconstrained_lookup["lognormal"]["mu_log10_msun"]),
        float(unconstrained_lookup["lognormal"]["sigma_log10_msun"]),
        marker="x",
        color="white",
        markersize=8,
        markeredgewidth=1.8,
    )
    axes[1].set_xlabel(r"$\mu_{\log M}$")
    axes[1].set_ylabel(r"$\sigma_{\log M}$")
    axes[1].set_title("Lognormal")

    schechter_pivot = schechter_table["log_likelihood"].max()
    x_unique = np.sort(schechter_table["alpha_dndm"].unique())
    y_unique = np.sort(schechter_table["log10_m_c_msun"].unique())
    z_grid = (
        schechter_table.pivot(index="log10_m_c_msun", columns="alpha_dndm", values="log_likelihood")
        .sort_index()
        .sort_index(axis=1)
        .to_numpy()
        - schechter_pivot
    )
    im2 = axes[2].contourf(x_unique, y_unique, z_grid, levels=18, cmap="magma")
    axes[2].plot(
        float(unconstrained_lookup["schechter"]["alpha_dndm"]),
        float(unconstrained_lookup["schechter"]["log10_m_c_msun"]),
        marker="x",
        color="white",
        markersize=8,
        markeredgewidth=1.8,
    )
    axes[2].set_xlabel(r"$\alpha$")
    axes[2].set_ylabel(r"$\log_{10}(M_c/{\rm M}_\odot)$")
    axes[2].set_title("Schechter")

    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.18, top=0.88, wspace=0.32)
    cax = fig.add_axes([0.90, 0.18, 0.018, 0.64])
    cbar = fig.colorbar(im2, cax=cax)
    cbar.set_label(r"$\Delta \log L$")

    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root-name",
        type=str,
        default="single_component_family_profile_scan_v4",
    )
    parser.add_argument(
        "--smooth-survivability",
        action="store_true",
        help="Use the smooth decreasing-plus-plateau survivability surface instead of the default hard-threshold grid.",
    )
    parser.add_argument(
        "--eta-t",
        type=float,
        default=1.0,
        help="Global lifetime renormalization to use when --smooth-survivability is enabled.",
    )
    parser.add_argument(
        "--envelope-survivability",
        action="store_true",
        help="Use the hard lower-envelope survivability surface instead of the default or smooth survivability grid.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_root = project_root / "variants" / args.output_root_name
    outputs_tables = output_root / "outputs" / "tables"
    outputs_figures = output_root / "outputs" / "figures"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    outputs_figures.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(project_root / ".cache"))
    (project_root / ".mplconfig").mkdir(parents=True, exist_ok=True)
    (project_root / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)

    from globular_clusters_imf.detectability_longitude_model import (
        fit_detectability_corrected_single_component_models_with_abs_longitude,
    )
    from globular_clusters_imf.joint_model import JointModelSpec, imf_parameter_count
    from globular_clusters_imf.model import fit_catalog_models
    from globular_clusters_imf.envelope_survivability import build_envelope_survivability_grid
    from globular_clusters_imf.smooth_survivability import build_smooth_survivability_grid

    catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog.csv"

    catalog = pd.read_csv(catalog_path)
    prepared_catalog = fit_catalog_models(catalog, output_root)["catalog"]
    if args.smooth_survivability and args.envelope_survivability:
        raise ValueError("Use at most one of --smooth-survivability and --envelope-survivability.")
    survival_grid_override = None
    if args.smooth_survivability:
        smooth_survival = build_smooth_survivability_grid(
            prepared_catalog,
            eta_t=float(args.eta_t),
        )
        survival_grid_override = {
            "log_mass_grid": np.asarray(smooth_survival["log_mass_grid"], dtype=float),
            "log_a_grid": np.asarray(smooth_survival["log_a_grid"], dtype=float),
            "semi_major_axis_grid_kpc": np.asarray(smooth_survival["semi_major_axis_grid_kpc"], dtype=float),
            "survival_probability": np.asarray(smooth_survival["survival_probability"], dtype=float),
            "selection_offset_dex": 0.0,
            "bandwidth_log10_a_dex": float(smooth_survival["bandwidth_log10_a_dex"]),
            "smooth_survivability_summary": smooth_survival["summary"],
        }
    elif args.envelope_survivability:
        envelope_survival = build_envelope_survivability_grid(prepared_catalog)
        survival_grid_override = {
            "log_mass_grid": np.asarray(envelope_survival["log_mass_grid"], dtype=float),
            "log_a_grid": np.asarray(envelope_survival["log_a_grid"], dtype=float),
            "semi_major_axis_grid_kpc": np.asarray(envelope_survival["semi_major_axis_grid_kpc"], dtype=float),
            "survival_probability": np.asarray(envelope_survival["survival_probability"], dtype=float),
            "selection_offset_dex": 0.0,
            "bandwidth_log10_a_dex": float(envelope_survival["bandwidth_log10_a_dex"]),
            "envelope_survivability_summary": envelope_survival["summary"],
        }

    comparison_specs = [
        JointModelSpec(imf_family="lognormal", radial_model="step5"),
        JointModelSpec(imf_family="powerlaw", radial_model="step5"),
        JointModelSpec(imf_family="schechter", radial_model="step5"),
        JointModelSpec(imf_family="lognormal", radial_model="logpoly3"),
        JointModelSpec(imf_family="powerlaw", radial_model="logpoly3"),
        JointModelSpec(imf_family="schechter", radial_model="logpoly3"),
    ]
    unconstrained_comparison = fit_detectability_corrected_single_component_models_with_abs_longitude(
        prepared_catalog,
        project_root=output_root,
        model_specs=comparison_specs,
        n_iterations=12,
        survival_grid_override=survival_grid_override,
    )
    unconstrained_table = unconstrained_comparison["summary_table"].copy()
    unconstrained_table.to_csv(outputs_tables / "unconstrained_family_comparison.csv", index=False)

    unconstrained_lookup: dict[str, pd.Series] = {}
    unconstrained_reference_results: dict[str, dict[str, object]] = {}
    for family_name in ("powerlaw", "lognormal", "schechter"):
        family_best = (
            unconstrained_table.loc[unconstrained_table["imf_family"] == family_name]
            .sort_values("log_likelihood", ascending=False)
            .iloc[0]
            .copy()
        )
        imf_params = json.loads(str(family_best["imf_parameters_json"]))
        for key, value in imf_params.items():
            family_best[key] = value
        unconstrained_lookup[family_name] = family_best
        reference_result = next(
            result
            for result in unconstrained_comparison["all_results"]
            if result["spec"].imf_family == family_name
            and result["spec"].radial_model == str(family_best["radial_model"])
        )
        imf_param_count = imf_parameter_count(family_name)
        unconstrained_reference_results[family_name] = {
            "completeness": np.asarray(reference_result["final_completeness_raw_parameters"], dtype=float),
            "radial": np.asarray(reference_result["final_payload"]["raw_parameters"][imf_param_count:], dtype=float),
        }

    n_iterations = 12
    powerlaw_alpha_grid = np.linspace(-3.0, -0.8, 29)
    lognormal_mu_grid = np.linspace(3.5, 5.85, 17)
    lognormal_sigma_grid = np.linspace(0.35, 1.05, 9)
    schechter_alpha_grid = np.linspace(-3.0, -0.35, 15)
    schechter_logmc_grid = np.linspace(5.75, 6.95, 9)

    powerlaw_table, powerlaw_entries = run_powerlaw_scan(
        prepared_catalog,
        output_root,
        alpha_grid=powerlaw_alpha_grid,
        n_iterations=n_iterations,
        reference_start_completeness=unconstrained_reference_results["powerlaw"]["completeness"],
        reference_start_radial=unconstrained_reference_results["powerlaw"]["radial"],
        survival_grid_override=survival_grid_override,
    )
    powerlaw_table.to_csv(outputs_tables / "powerlaw_profile_scan.csv", index=False)
    save_best_family_result(outputs_tables, "powerlaw", powerlaw_entries)

    lognormal_table, lognormal_grid = run_2d_family_scan(
        prepared_catalog,
        output_root,
        spec=JointModelSpec(imf_family="lognormal", radial_model="logpoly3"),
        grid_x=lognormal_mu_grid,
        grid_y=lognormal_sigma_grid,
        param_builder=lambda mu, sigma: (
            np.array([mu, np.log(sigma)], dtype=float),
            {"mu_log10_msun": mu, "sigma_log10_msun": sigma},
        ),
        n_iterations=n_iterations,
        reference_start_completeness=unconstrained_reference_results["lognormal"]["completeness"],
        reference_start_radial=unconstrained_reference_results["lognormal"]["radial"],
        survival_grid_override=survival_grid_override,
    )
    lognormal_table.to_csv(outputs_tables / "lognormal_profile_scan.csv", index=False)
    save_best_family_result(
        outputs_tables,
        "lognormal",
        [entry for row in lognormal_grid for entry in row],
    )

    schechter_table, schechter_grid = run_2d_family_scan(
        prepared_catalog,
        output_root,
        spec=JointModelSpec(imf_family="schechter", radial_model="logpoly3"),
        grid_x=schechter_alpha_grid,
        grid_y=schechter_logmc_grid,
        param_builder=lambda alpha, log_mc: (
            np.array([alpha, log_mc], dtype=float),
            {"alpha_dndm": alpha, "log10_m_c_msun": log_mc},
        ),
        n_iterations=n_iterations,
        reference_start_completeness=unconstrained_reference_results["schechter"]["completeness"],
        reference_start_radial=unconstrained_reference_results["schechter"]["radial"],
        survival_grid_override=survival_grid_override,
    )
    schechter_table.to_csv(outputs_tables / "schechter_profile_scan.csv", index=False)
    save_best_family_result(
        outputs_tables,
        "schechter",
        [entry for row in schechter_grid for entry in row],
    )

    plot_family_scan_summary(
        outputs_figures / "single_component_family_profile_scan_summary.pdf",
        powerlaw_table=powerlaw_table,
        lognormal_table=lognormal_table,
        schechter_table=schechter_table,
        unconstrained_lookup=unconstrained_lookup,
    )

    summary_payload = {
        "scan_configuration": {
            "radial_model": "logpoly3",
            "n_iterations_per_grid_point": n_iterations,
            "smooth_survivability": bool(args.smooth_survivability),
            "envelope_survivability": bool(args.envelope_survivability),
            "eta_t": float(args.eta_t),
            "powerlaw_alpha_grid": powerlaw_alpha_grid.tolist(),
            "lognormal_mu_grid": lognormal_mu_grid.tolist(),
            "lognormal_sigma_grid": lognormal_sigma_grid.tolist(),
            "schechter_alpha_grid": schechter_alpha_grid.tolist(),
            "schechter_log10_mc_grid": schechter_logmc_grid.tolist(),
        },
        "unconstrained_family_best_fits": {
            family_name: json.loads(pd.Series(row).to_json())
            for family_name, row in unconstrained_lookup.items()
        },
        "powerlaw_scan": summarize_family_scan(powerlaw_table, unconstrained_lookup["powerlaw"]),
        "lognormal_scan": summarize_family_scan(lognormal_table, unconstrained_lookup["lognormal"]),
        "schechter_scan": summarize_family_scan(schechter_table, unconstrained_lookup["schechter"]),
    }
    if survival_grid_override is not None:
        if "smooth_survivability_summary" in survival_grid_override:
            summary_payload["smooth_survivability_summary"] = asdict_survival_summary(
                survival_grid_override["smooth_survivability_summary"]
            )
        if "envelope_survivability_summary" in survival_grid_override:
            summary_payload["envelope_survivability_summary"] = asdict_survival_summary(
                survival_grid_override["envelope_survivability_summary"]
            )
    (outputs_tables / "single_component_family_profile_scan_summary.json").write_text(
        json.dumps(summary_payload, indent=2)
    )
    print(json.dumps(summary_payload, indent=2))


def asdict_survival_summary(summary_obj) -> dict[str, object]:
    if hasattr(summary_obj, "__dict__"):
        return {
            key: (float(value) if isinstance(value, (np.floating, float)) else value)
            for key, value in summary_obj.__dict__.items()
        }
    return dict(summary_obj)


if __name__ == "__main__":
    main()
