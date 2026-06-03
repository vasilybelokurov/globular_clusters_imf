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
from matplotlib.colors import LinearSegmentedColormap

from globular_clusters_imf.gg23_survivability import (  # noqa: E402
    GG23_MODELS,
    build_gg23_survivability_grid,
    effective_radius_kpc_from_semimajor_axis,
    gg23_initial_mass_from_present_msun,
)
from globular_clusters_imf.joint_model import centers_to_edges_local  # noqa: E402
from globular_clusters_imf.smooth_survivability import build_smooth_survivability_grid  # noqa: E402


DEFAULT_MODEL_NAMES = [
    "gg23_no_bh",
    "gg23_bh",
    "gg23_bh_feh_gradient",
    "gg23_bh_past_tidal",
    "gg23_bh_feh_gradient_past_tidal",
]


def main() -> None:
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog_with_origin_and_chemistry.csv"
    if not catalog_path.exists():
        catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog.csv"
    catalog = pd.read_csv(catalog_path)

    baumgardt_grid = build_smooth_survivability_grid(
        catalog,
        eta_t=1.0,
        surface_model="logistic",
    )
    gg23_grids = {
        name: build_gg23_survivability_grid(
            catalog,
            GG23_MODELS[name],
            eta_t=1.0,
            surface_model="logistic",
        )
        for name in DEFAULT_MODEL_NAMES
    }

    all_grids = {"baumgardt2019": baumgardt_grid, **gg23_grids}
    labels = {
        "baumgardt2019": "Baumgardt 2019 fiducial",
        **{name: GG23_MODELS[name].label for name in DEFAULT_MODEL_NAMES},
    }

    write_summary_tables(all_grids, labels, tables_dir)
    write_grid_archive(all_grids, tables_dir / "gg23_survivability_surface_grids.npz")
    model_mass_points = build_model_specific_point_masses(catalog, tables_dir)
    plot_survivability_maps(
        catalog,
        all_grids,
        labels,
        figures_dir / "gg23_survivability_maps.pdf",
        figures_dir / "gg23_survivability_maps.png",
    )
    plot_survivability_maps(
        catalog,
        all_grids,
        labels,
        figures_dir / "gg23_survivability_maps_model_mini.pdf",
        figures_dir / "gg23_survivability_maps_model_mini.png",
        log_mass_points_by_model=model_mass_points,
    )
    plot_boundary_comparison(
        catalog,
        all_grids,
        labels,
        figures_dir / "gg23_survivability_boundary_comparison.pdf",
        figures_dir / "gg23_survivability_boundary_comparison.png",
    )

    print(f"Wrote {figures_dir / 'gg23_survivability_maps.pdf'}")
    print(f"Wrote {figures_dir / 'gg23_survivability_maps_model_mini.pdf'}")
    print(f"Wrote {figures_dir / 'gg23_survivability_boundary_comparison.pdf'}")
    print(f"Wrote {tables_dir / 'gg23_survivability_model_summary.csv'}")


def write_summary_tables(
    grids: dict[str, dict[str, object]],
    labels: dict[str, str],
    tables_dir: Path,
) -> None:
    summary_rows = []
    boundary_rows = []
    for name, grid in grids.items():
        summary = grid["summary"]
        model_payload = grid.get("gg23_model", {})
        summary_rows.append(
            {
                "model_name": name,
                "label": labels[name],
                "eta_t": float(grid["eta_t"]),
                "surface_model": str(grid["surface_model"]),
                "bandwidth_log10_a_dex": float(grid["bandwidth_log10_a_dex"]),
                "outer_level_50_log10_msun": float(summary.outer_level_50_log10_msun),
                "inner_level_50_log10_msun": float(summary.inner_level_50_log10_msun),
                "transition_a_kpc": float(summary.transition_a_kpc),
                "transition_band_width_dex": float(summary.transition_band_width_dex),
                "optimizer_value": float(summary.optimizer_value),
                "gg23_x": model_payload.get("x", np.nan),
                "gg23_y": model_payload.get("y", np.nan),
                "gg23_mdot_ref_msun_per_myr": model_payload.get("mdot_ref_msun_per_myr", np.nan),
                "gg23_metallicity_gradient": model_payload.get("metallicity_gradient", False),
                "gg23_past_tidal_evolution": model_payload.get("past_tidal_evolution", False),
            }
        )
        for log_a, a, boundary in zip(
            grid["log_a_grid"],
            grid["semi_major_axis_grid_kpc"],
            grid["fitted_boundary_50_log10_msun"],
            strict=True,
        ):
            boundary_rows.append(
                {
                    "model_name": name,
                    "label": labels[name],
                    "log10_a_kpc": float(log_a),
                    "a_kpc": float(a),
                    "fitted_boundary_50_log10_msun": float(boundary),
                }
            )
    pd.DataFrame(summary_rows).to_csv(tables_dir / "gg23_survivability_model_summary.csv", index=False)
    pd.DataFrame(boundary_rows).to_csv(tables_dir / "gg23_survivability_boundary_50.csv", index=False)


def write_grid_archive(grids: dict[str, dict[str, object]], output_path: Path) -> None:
    payload: dict[str, np.ndarray | str] = {}
    metadata = {}
    for name, grid in grids.items():
        payload[f"{name}_log_mass_grid"] = np.asarray(grid["log_mass_grid"], dtype=float)
        payload[f"{name}_log_a_grid"] = np.asarray(grid["log_a_grid"], dtype=float)
        payload[f"{name}_survival_probability"] = np.asarray(grid["survival_probability"], dtype=float)
        payload[f"{name}_boundary_50_log10_msun"] = np.asarray(
            grid["fitted_boundary_50_log10_msun"],
            dtype=float,
        )
        metadata[name] = {
            "eta_t": float(grid["eta_t"]),
            "surface_model": str(grid["surface_model"]),
            "gg23_model": grid.get("gg23_model"),
        }
    payload["metadata_json"] = np.asarray(json.dumps(metadata, indent=2))
    np.savez_compressed(output_path, **payload)


def build_model_specific_point_masses(
    catalog: pd.DataFrame,
    tables_dir: Path,
) -> dict[str, np.ndarray]:
    point_masses: dict[str, np.ndarray] = {
        "baumgardt2019": catalog["log_initial_mass_msun"].to_numpy(dtype=float),
    }
    semi_major_axis = catalog["semi_major_axis_kpc"].to_numpy(dtype=float)
    effective_radius = effective_radius_kpc_from_semimajor_axis(
        semi_major_axis,
        catalog["eccentricity"].to_numpy(dtype=float),
    )
    present_mass = catalog["present_mass_msun"].to_numpy(dtype=float)
    rows = []
    for name in DEFAULT_MODEL_NAMES:
        initial_mass = gg23_initial_mass_from_present_msun(
            present_mass,
            effective_radius,
            GG23_MODELS[name],
            gradient_radius_kpc=semi_major_axis,
        )
        log_initial_mass = np.log10(initial_mass)
        point_masses[name] = log_initial_mass
        rows.append(
            pd.DataFrame(
                {
                    "cluster_label": catalog["cluster_label"].to_numpy(),
                    "model_name": name,
                    "model_label": GG23_MODELS[name].label,
                    "semi_major_axis_kpc": semi_major_axis,
                    "effective_radius_kpc": effective_radius,
                    "present_mass_msun": present_mass,
                    "log10_panel_initial_mass_msun": log_initial_mass,
                    "panel_initial_mass_msun": initial_mass,
                    "baumgardt_log10_initial_mass_msun": catalog["log_initial_mass_msun"].to_numpy(dtype=float),
                    "delta_log10_initial_mass_vs_baumgardt": log_initial_mass
                    - catalog["log_initial_mass_msun"].to_numpy(dtype=float),
                }
            )
        )
    pd.concat(rows, ignore_index=True).to_csv(
        tables_dir / "gg23_survivability_model_mini_panel_points.csv",
        index=False,
    )
    return point_masses


def plot_survivability_maps(
    catalog: pd.DataFrame,
    grids: dict[str, dict[str, object]],
    labels: dict[str, str],
    pdf_path: Path,
    png_path: Path,
    *,
    log_mass_points_by_model: dict[str, np.ndarray] | None = None,
) -> None:
    ordered_names = list(grids)
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.4), sharex=True, sharey=True, constrained_layout=True)
    axes = axes.ravel()
    cmap = LinearSegmentedColormap.from_list(
        "survivability_gg23",
        [(0.98, 0.98, 0.98), (0.76, 0.84, 0.92), (0.14, 0.35, 0.63)],
    )
    log_a_points = np.log10(catalog["semi_major_axis_kpc"].to_numpy(dtype=float))
    mesh = None
    for ax, name in zip(axes, ordered_names, strict=False):
        log_m_points = (
            log_mass_points_by_model[name]
            if log_mass_points_by_model is not None and name in log_mass_points_by_model
            else catalog["log_initial_mass_msun"].to_numpy(dtype=float)
        )
        grid = grids[name]
        log_a = np.asarray(grid["log_a_grid"], dtype=float)
        log_m = np.asarray(grid["log_mass_grid"], dtype=float)
        survival = np.asarray(grid["survival_probability"], dtype=float)
        mesh = ax.pcolormesh(
            centers_to_edges_local(log_a),
            centers_to_edges_local(log_m),
            survival,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            shading="auto",
        )
        ax.contour(log_a, log_m, survival, levels=[0.1, 0.5, 0.9], colors="black", linewidths=[0.7, 1.1, 0.7])
        ax.plot(
            log_a,
            grid["fitted_boundary_50_log10_msun"],
            color="black",
            linewidth=1.6,
            label=r"$S=0.5$",
        )
        ax.scatter(log_a_points, log_m_points, s=8, color="white", edgecolors="black", linewidths=0.25, alpha=0.75)
        ax.set_title(labels[name], fontsize=10)
        ax.set_xlim(np.log10(0.7), np.log10(220.0))
        ax.set_ylim(3.55, 7.35)
    for ax in axes[len(ordered_names) :]:
        ax.axis("off")
    for ax in axes[::3]:
        ax.set_ylabel(r"$\log_{10}(M_{\rm ini}/{\rm M}_\odot)$")
    for ax in axes[-3:]:
        ax.set_xlabel(r"$\log_{10}(a/{\rm kpc})$")
    if mesh is not None:
        fig.colorbar(mesh, ax=axes.tolist(), label=r"Survivability $S(M_{\rm ini},a)$", shrink=0.88)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


def plot_boundary_comparison(
    catalog: pd.DataFrame,
    grids: dict[str, dict[str, object]],
    labels: dict[str, str],
    pdf_path: Path,
    png_path: Path,
) -> None:
    colors = {
        "baumgardt2019": "#000000",
        "gg23_no_bh": "#1b9e77",
        "gg23_bh": "#d95f02",
        "gg23_bh_feh_gradient": "#7570b3",
        "gg23_bh_past_tidal": "#e7298a",
        "gg23_bh_feh_gradient_past_tidal": "#66a61e",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.scatter(
        np.log10(catalog["semi_major_axis_kpc"].to_numpy(dtype=float)),
        catalog["log_initial_mass_msun"].to_numpy(dtype=float),
        s=13,
        color="0.25",
        alpha=0.55,
        label="Observed GCs",
    )
    for name, grid in grids.items():
        ax.plot(
            grid["log_a_grid"],
            grid["fitted_boundary_50_log10_msun"],
            color=colors[name],
            linewidth=2.0,
            label=labels[name],
        )
    ax.set_xlim(np.log10(0.7), np.log10(220.0))
    ax.set_ylim(3.55, 7.35)
    ax.set_xlabel(r"$\log_{10}(a/{\rm kpc})$")
    ax.set_ylabel(r"$\log_{10}(M_{\rm ini}/{\rm M}_\odot)$")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
