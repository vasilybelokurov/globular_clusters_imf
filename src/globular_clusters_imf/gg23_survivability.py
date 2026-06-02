from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from .model import AGE_GYR
from .smooth_survivability import (
    fit_monotonic_soft_survivability_model,
)


GG23_REFERENCE_MASS_MSUN = 2.0e5
GG23_REFERENCE_MDOT_ABS_MSUN_PER_MYR = 30.0
GG23_REFERENCE_OMEGA_TID_MYR_INV = 0.32
GG23_REFERENCE_VC_KMS = 220.0


@dataclass(frozen=True)
class GG23DisruptionModel:
    """Effective GG23 analytic mass-loss prescription.

    This implements equations (4)-(6) of Gieles & Gnedin (2023) as a
    survival-threshold model. It uses each catalogue orbit through
    R_eff = R_p(1+e) = a(1-e^2), the effective radius used by GG23 for an
    isothermal-sphere tidal field. Full anisotropic phase-space averaging is
    not part of this 2D backend; it is an orbit-distribution model rather than
    a per-orbit survivability law.
    """

    name: str
    label: str
    x: float = 2.0 / 3.0
    y: float = 2.0 / 3.0
    mdot_ref_msun_per_myr: float = -30.0
    circular_speed_kms: float = GG23_REFERENCE_VC_KMS
    metallicity_gradient: bool = False
    past_tidal_evolution: bool = False


GG23_MODELS: dict[str, GG23DisruptionModel] = {
    "gg23_no_bh": GG23DisruptionModel(
        name="gg23_no_bh",
        label="GG23 no BHs",
        y=2.0 / 3.0,
        mdot_ref_msun_per_myr=-30.0,
    ),
    "gg23_bh": GG23DisruptionModel(
        name="gg23_bh",
        label="GG23 BHs",
        y=4.0 / 3.0,
        mdot_ref_msun_per_myr=-45.0,
    ),
    "gg23_bh_feh_gradient": GG23DisruptionModel(
        name="gg23_bh_feh_gradient",
        label="GG23 BHs + [Fe/H] gradient",
        y=4.0 / 3.0,
        mdot_ref_msun_per_myr=-45.0,
        metallicity_gradient=True,
    ),
    "gg23_bh_past_tidal": GG23DisruptionModel(
        name="gg23_bh_past_tidal",
        label="GG23 BHs + past tides",
        y=4.0 / 3.0,
        mdot_ref_msun_per_myr=-45.0,
        past_tidal_evolution=True,
    ),
    "gg23_bh_feh_gradient_past_tidal": GG23DisruptionModel(
        name="gg23_bh_feh_gradient_past_tidal",
        label="GG23 BHs + [Fe/H] + past tides",
        y=4.0 / 3.0,
        mdot_ref_msun_per_myr=-45.0,
        metallicity_gradient=True,
        past_tidal_evolution=True,
    ),
}


def gg23_model(name: str) -> GG23DisruptionModel:
    try:
        return GG23_MODELS[name]
    except KeyError as error:
        raise KeyError(f"Unknown GG23 model {name!r}. Available: {sorted(GG23_MODELS)}") from error


def effective_radius_kpc_from_semimajor_axis(
    semi_major_axis_kpc: np.ndarray | float,
    eccentricity: np.ndarray | float,
) -> np.ndarray:
    a = np.asarray(semi_major_axis_kpc, dtype=float)
    e = np.clip(np.asarray(eccentricity, dtype=float), 0.0, 0.999999)
    return np.clip(a * (1.0 - np.square(e)), 1.0e-4, None)


def gg23_omega_tid_myr_inverse(
    effective_radius_kpc: np.ndarray | float,
    *,
    circular_speed_kms: float = GG23_REFERENCE_VC_KMS,
) -> np.ndarray:
    radius = np.clip(np.asarray(effective_radius_kpc, dtype=float), 1.0e-4, None)
    return GG23_REFERENCE_OMEGA_TID_MYR_INV * (float(circular_speed_kms) / GG23_REFERENCE_VC_KMS) / radius


def gg23_radius_dependent_mass_loss_parameters(
    effective_radius_kpc: np.ndarray | float,
    model: GG23DisruptionModel,
) -> tuple[np.ndarray, np.ndarray]:
    radius = np.asarray(effective_radius_kpc, dtype=float)
    mdot_ref = np.full_like(radius, float(model.mdot_ref_msun_per_myr), dtype=float)
    y = np.full_like(radius, float(model.y), dtype=float)

    if model.metallicity_gradient:
        radius_for_gradient = np.clip(radius, 1.0, 10.0)
        log_radius = np.log10(radius_for_gradient)
        mdot_ref = -30.0 * (1.0 + 0.5 * log_radius)
        y = (2.0 / 3.0) + (2.0 / 3.0) * log_radius

    if model.past_tidal_evolution:
        past_factor = np.ones_like(radius, dtype=float)
        mask = radius > 4.0
        past_factor[mask] = np.sqrt(radius[mask] / 4.0)
        mdot_ref = mdot_ref * past_factor

    return mdot_ref, y


def gg23_total_disruption_time_gyr(
    initial_mass_msun: np.ndarray | float,
    effective_radius_kpc: np.ndarray | float,
    model: GG23DisruptionModel,
) -> np.ndarray:
    mass = np.asarray(initial_mass_msun, dtype=float)
    radius = np.asarray(effective_radius_kpc, dtype=float)
    omega_tid = gg23_omega_tid_myr_inverse(radius, circular_speed_kms=model.circular_speed_kms)
    mdot_ref, y = gg23_radius_dependent_mass_loss_parameters(radius, model)
    prefactor_gyr = (
        10.0
        * ((2.0 / 3.0) / y)
        * (GG23_REFERENCE_MDOT_ABS_MSUN_PER_MYR / np.abs(mdot_ref))
        * (GG23_REFERENCE_OMEGA_TID_MYR_INV / omega_tid)
    )
    return prefactor_gyr * np.power(mass / GG23_REFERENCE_MASS_MSUN, float(model.x))


def gg23_survival_mass_cut_msun(
    effective_radius_kpc: np.ndarray | float,
    model: GG23DisruptionModel,
    *,
    age_gyr: float = AGE_GYR,
    eta_t: float = 1.0,
) -> np.ndarray:
    radius = np.asarray(effective_radius_kpc, dtype=float)
    omega_tid = gg23_omega_tid_myr_inverse(radius, circular_speed_kms=model.circular_speed_kms)
    mdot_ref, y = gg23_radius_dependent_mass_loss_parameters(radius, model)
    target_age_gyr = float(age_gyr) / float(eta_t)
    prefactor_gyr = (
        10.0
        * ((2.0 / 3.0) / y)
        * (GG23_REFERENCE_MDOT_ABS_MSUN_PER_MYR / np.abs(mdot_ref))
        * (GG23_REFERENCE_OMEGA_TID_MYR_INV / omega_tid)
    )
    ratio = np.clip(target_age_gyr / np.clip(prefactor_gyr, 1.0e-12, None), 0.0, None)
    return GG23_REFERENCE_MASS_MSUN * np.power(ratio, 1.0 / float(model.x))


def gg23_present_mass_msun(
    initial_mass_msun: np.ndarray | float,
    effective_radius_kpc: np.ndarray | float,
    model: GG23DisruptionModel,
    *,
    age_gyr: float = AGE_GYR,
    eta_t: float = 1.0,
) -> np.ndarray:
    mass = np.asarray(initial_mass_msun, dtype=float)
    target_age_gyr = float(age_gyr) / float(eta_t)
    t_dis = gg23_total_disruption_time_gyr(mass, effective_radius_kpc, model)
    remaining = 1.0 - target_age_gyr / np.clip(t_dis, 1.0e-12, None)
    present = mass * np.power(np.clip(remaining, 0.0, None), 1.0 / float(model.y))
    return np.where(remaining > 0.0, present, 0.0)


def build_raw_gg23_survival_grid_from_catalog(
    catalog: pd.DataFrame,
    model: GG23DisruptionModel,
    *,
    eta_t: float = 1.0,
    n_radius_grid: int = 160,
    n_mass_grid: int = 180,
    bandwidth_log10_a_dex: float = 0.18,
) -> dict[str, object]:
    working = catalog.copy()
    effective_radius = effective_radius_kpc_from_semimajor_axis(
        working["semi_major_axis_kpc"].to_numpy(dtype=float),
        working["eccentricity"].to_numpy(dtype=float),
    )
    cuts = gg23_survival_mass_cut_msun(
        effective_radius,
        model,
        age_gyr=AGE_GYR,
        eta_t=eta_t,
    )
    working["gg23_effective_radius_kpc"] = effective_radius
    working["gg23_survival_mass_cut_msun"] = cuts
    working["log_gg23_survival_mass_cut_msun"] = np.log10(cuts)

    log_a_data = np.log10(working["semi_major_axis_kpc"].to_numpy(dtype=float))
    log_cut_data = working["log_gg23_survival_mass_cut_msun"].to_numpy(dtype=float)
    log_a_grid = np.linspace(log_a_data.min(), log_a_data.max(), n_radius_grid)
    log_mass_min = min(3.5, float(np.floor(working["log_initial_mass_msun"].min() * 10.0) / 10.0))
    log_mass_max = max(7.3, float(np.ceil(working["log_initial_mass_msun"].max() * 10.0) / 10.0))
    log_mass_grid = np.linspace(log_mass_min, log_mass_max, n_mass_grid)

    weights = np.exp(
        -0.5 * np.square((log_a_grid[:, None] - log_a_data[None, :]) / bandwidth_log10_a_dex)
    )
    weights /= np.clip(weights.sum(axis=1, keepdims=True), 1.0e-12, None)

    indicators = log_mass_grid[:, None] >= log_cut_data[None, :]
    survival_probability = indicators @ weights.T
    return {
        "catalog": working,
        "log_mass_grid": log_mass_grid,
        "log_a_grid": log_a_grid,
        "semi_major_axis_grid_kpc": np.power(10.0, log_a_grid),
        "survival_probability": np.clip(survival_probability, 1.0e-12, 1.0),
        "bandwidth_log10_a_dex": bandwidth_log10_a_dex,
        "eta_t": float(eta_t),
        "gg23_model": asdict(model),
    }


def build_gg23_survivability_grid(
    catalog: pd.DataFrame,
    model: GG23DisruptionModel | str,
    *,
    eta_t: float = 1.0,
    n_radius_grid: int = 160,
    n_mass_grid: int = 180,
    bandwidth_log10_a_dex: float = 0.18,
    surface_model: str = "logistic",
) -> dict[str, object]:
    model = gg23_model(model) if isinstance(model, str) else model
    raw_grid = build_raw_gg23_survival_grid_from_catalog(
        catalog,
        model,
        eta_t=eta_t,
        n_radius_grid=n_radius_grid,
        n_mass_grid=n_mass_grid,
        bandwidth_log10_a_dex=bandwidth_log10_a_dex,
    )
    fit_payload = fit_monotonic_soft_survivability_model(
        raw_grid["catalog"],
        np.asarray(raw_grid["log_mass_grid"], dtype=float),
        np.asarray(raw_grid["log_a_grid"], dtype=float),
        np.asarray(raw_grid["survival_probability"], dtype=float),
        bandwidth_log10_a_dex=float(raw_grid["bandwidth_log10_a_dex"]),
        surface_model=surface_model,
    )
    summary = replace(fit_payload["summary"], eta_t=float(eta_t))
    return {
        "log_mass_grid": np.asarray(raw_grid["log_mass_grid"], dtype=float),
        "log_a_grid": np.asarray(raw_grid["log_a_grid"], dtype=float),
        "semi_major_axis_grid_kpc": np.asarray(raw_grid["semi_major_axis_grid_kpc"], dtype=float),
        "survival_probability": np.asarray(fit_payload["fitted_probability"], dtype=float),
        "eta_t": float(eta_t),
        "surface_model": str(surface_model),
        "bandwidth_log10_a_dex": float(bandwidth_log10_a_dex),
        "raw_survival_probability": np.asarray(raw_grid["survival_probability"], dtype=float),
        "raw_boundary_10_log10_msun": np.asarray(fit_payload["raw_boundary_10_log10_msun"], dtype=float),
        "raw_boundary_50_log10_msun": np.asarray(fit_payload["raw_boundary_50_log10_msun"], dtype=float),
        "raw_boundary_80_log10_msun": np.asarray(fit_payload["raw_boundary_80_log10_msun"], dtype=float),
        "raw_boundary_90_log10_msun": np.asarray(fit_payload["raw_boundary_90_log10_msun"], dtype=float),
        "fitted_boundary_10_log10_msun": np.asarray(fit_payload["fitted_boundary_10_log10_msun"], dtype=float),
        "fitted_boundary_50_log10_msun": np.asarray(fit_payload["fitted_boundary_50_log10_msun"], dtype=float),
        "fitted_boundary_90_log10_msun": np.asarray(fit_payload["fitted_boundary_90_log10_msun"], dtype=float),
        "occupancy_table": fit_payload["occupancy_table"].copy(),
        "summary": summary,
        "gg23_model": asdict(model),
        "raw_catalog": raw_grid["catalog"].copy(),
    }
