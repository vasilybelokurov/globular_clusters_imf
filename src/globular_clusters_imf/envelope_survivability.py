from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EnvelopeSurvivabilitySummary:
    boundary_kind: str
    margin_dex: float
    n_hull_vertices: int
    log10_a_min_kpc: float
    log10_a_max_kpc: float
    log10_mass_min_msun: float
    log10_mass_max_msun: float


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def compute_lower_convex_hull(log_a: np.ndarray, log_mass: np.ndarray) -> np.ndarray:
    points = np.column_stack([np.asarray(log_a, dtype=float), np.asarray(log_mass, dtype=float)])
    if len(points) == 0:
        raise ValueError("Need at least one point to build a survivability envelope.")
    order = np.lexsort((points[:, 1], points[:, 0]))
    points = points[order]
    lower: list[np.ndarray] = []
    for point in points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point.copy())
    hull = np.asarray(lower, dtype=float)
    if len(hull) == 1:
        hull = np.vstack([hull[0], hull[0]])
    return hull


def evaluate_piecewise_linear_boundary(
    log_a_grid: np.ndarray,
    hull_vertices: np.ndarray,
    *,
    margin_dex: float = 1.0e-3,
) -> np.ndarray:
    log_a_grid = np.asarray(log_a_grid, dtype=float)
    hull_vertices = np.asarray(hull_vertices, dtype=float)
    boundary = np.interp(
        log_a_grid,
        hull_vertices[:, 0],
        hull_vertices[:, 1],
        left=float(hull_vertices[0, 1]),
        right=float(hull_vertices[-1, 1]),
    )
    return np.asarray(boundary - float(margin_dex), dtype=float)


def build_envelope_survivability_grid(
    catalog: pd.DataFrame,
    *,
    n_radius_grid: int = 160,
    n_mass_grid: int = 180,
    margin_dex: float = 1.0e-3,
) -> dict[str, object]:
    if "semi_major_axis_kpc" not in catalog or "log_initial_mass_msun" not in catalog:
        raise ValueError("Catalog must contain semi_major_axis_kpc and log_initial_mass_msun.")

    working = catalog.copy()
    log_a_data = np.log10(working["semi_major_axis_kpc"].to_numpy(dtype=float))
    log_mass_data = working["log_initial_mass_msun"].to_numpy(dtype=float)
    hull_vertices = compute_lower_convex_hull(log_a_data, log_mass_data)

    log_a_grid = np.linspace(float(log_a_data.min()), float(log_a_data.max()), n_radius_grid)
    log_mass_min = min(3.5, float(np.floor(log_mass_data.min() * 10.0) / 10.0))
    log_mass_max = max(7.3, float(np.ceil(log_mass_data.max() * 10.0) / 10.0))
    log_mass_grid = np.linspace(log_mass_min, log_mass_max, n_mass_grid)

    boundary_log_mass = evaluate_piecewise_linear_boundary(
        log_a_grid,
        hull_vertices,
        margin_dex=margin_dex,
    )
    survival_probability = (log_mass_grid[:, None] >= boundary_log_mass[None, :]).astype(float)

    summary = EnvelopeSurvivabilitySummary(
        boundary_kind="lower_convex_hull_piecewise_linear",
        margin_dex=float(margin_dex),
        n_hull_vertices=int(len(hull_vertices)),
        log10_a_min_kpc=float(log_a_data.min()),
        log10_a_max_kpc=float(log_a_data.max()),
        log10_mass_min_msun=float(log_mass_data.min()),
        log10_mass_max_msun=float(log_mass_data.max()),
    )
    hull_table = pd.DataFrame(
        {
            "vertex_index": np.arange(len(hull_vertices), dtype=int),
            "log10_a_kpc": hull_vertices[:, 0],
            "a_kpc": np.power(10.0, hull_vertices[:, 0]),
            "log10_initial_mass_msun": hull_vertices[:, 1],
            "initial_mass_msun": np.power(10.0, hull_vertices[:, 1]),
        }
    )
    return {
        "log_mass_grid": log_mass_grid,
        "log_a_grid": log_a_grid,
        "semi_major_axis_grid_kpc": np.power(10.0, log_a_grid),
        "survival_probability": survival_probability,
        "raw_survival_probability": survival_probability.copy(),
        "boundary_log10_msun": boundary_log_mass,
        "hull_vertices": hull_vertices,
        "hull_table": hull_table,
        "margin_dex": float(margin_dex),
        "bandwidth_log10_a_dex": float("nan"),
        "summary": summary,
    }
