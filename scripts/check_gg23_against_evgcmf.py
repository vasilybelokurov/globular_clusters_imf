from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd

from globular_clusters_imf.gg23_survivability import (  # noqa: E402
    GG23_MODELS,
    GG23_REFERENCE_MASS_MSUN,
    gg23_present_mass_msun,
    gg23_radius_dependent_mass_loss_parameters,
    gg23_survival_mass_cut_msun,
    gg23_total_disruption_time_gyr,
)
from globular_clusters_imf.model import AGE_GYR  # noqa: E402


TEST_INITIAL_MASS_MSUN = np.array([1.5e5, 3.0e5, 7.0e5, 1.5e6, 4.0e6, 1.0e7])
TEST_GRADIENT_RADIUS_KPC = np.array([0.7, 1.0, 2.5, 6.0, 10.0, 35.0])
TEST_EFFECTIVE_RADIUS_KPC = np.array([0.45, 1.0, 3.2, 5.0, 9.0, 25.0])


def main() -> None:
    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    details = []
    summary = []
    for model_name, model in GG23_MODELS.items():
        ours_mmin = gg23_survival_mass_cut_msun(
            TEST_EFFECTIVE_RADIUS_KPC,
            model,
            gradient_radius_kpc=TEST_GRADIENT_RADIUS_KPC,
            age_gyr=AGE_GYR,
        )
        ours_tdis = gg23_total_disruption_time_gyr(
            TEST_INITIAL_MASS_MSUN,
            TEST_EFFECTIVE_RADIUS_KPC,
            model,
            gradient_radius_kpc=TEST_GRADIENT_RADIUS_KPC,
        )
        ours_present = gg23_present_mass_msun(
            TEST_INITIAL_MASS_MSUN,
            TEST_EFFECTIVE_RADIUS_KPC,
            model,
            gradient_radius_kpc=TEST_GRADIENT_RADIUS_KPC,
            age_gyr=AGE_GYR,
        )

        ref_mmin, ref_tdis, ref_present, ref_y, ref_mdot_ref = literal_evgcmf_mass_loss(
            TEST_INITIAL_MASS_MSUN,
            TEST_EFFECTIVE_RADIUS_KPC,
            TEST_GRADIENT_RADIUS_KPC,
            model_name,
        )
        _, ours_y = gg23_radius_dependent_mass_loss_parameters(TEST_GRADIENT_RADIUS_KPC, model)

        model_table = pd.DataFrame(
            {
                "model_name": model_name,
                "initial_mass_msun": TEST_INITIAL_MASS_MSUN,
                "gradient_radius_kpc": TEST_GRADIENT_RADIUS_KPC,
                "effective_radius_kpc": TEST_EFFECTIVE_RADIUS_KPC,
                "reference_y": ref_y,
                "ours_y": ours_y,
                "reference_mdot_ref_abs_msun_per_myr": ref_mdot_ref,
                "reference_survival_mass_cut_msun": ref_mmin,
                "ours_survival_mass_cut_msun": ours_mmin,
                "relative_survival_mass_cut_error": relative_error(ours_mmin, ref_mmin),
                "reference_tdis_gyr": ref_tdis,
                "ours_tdis_gyr": ours_tdis,
                "relative_tdis_error": relative_error(ours_tdis, ref_tdis),
                "reference_present_mass_msun": ref_present,
                "ours_present_mass_msun": ours_present,
                "relative_present_mass_error": relative_error(ours_present, ref_present),
            }
        )
        details.append(model_table)
        summary.append(
            {
                "model_name": model_name,
                "max_abs_relative_survival_mass_cut_error": float(
                    np.nanmax(np.abs(model_table["relative_survival_mass_cut_error"]))
                ),
                "max_abs_relative_tdis_error": float(np.nanmax(np.abs(model_table["relative_tdis_error"]))),
                "max_abs_relative_present_mass_error": float(
                    np.nanmax(np.abs(model_table["relative_present_mass_error"]))
                ),
                "max_abs_y_error": float(np.nanmax(np.abs(model_table["ours_y"] - model_table["reference_y"]))),
            }
        )

    details_table = pd.concat(details, ignore_index=True)
    summary_table = pd.DataFrame(summary)
    details_path = tables_dir / "gg23_evgcmf_formula_check_details.csv"
    summary_path = tables_dir / "gg23_evgcmf_formula_check_summary.csv"
    details_table.to_csv(details_path, index=False)
    summary_table.to_csv(summary_path, index=False)
    print(f"Wrote {details_path}")
    print(f"Wrote {summary_path}")
    print(summary_table.to_string(index=False))


def literal_evgcmf_mass_loss(
    initial_mass_msun: np.ndarray,
    effective_radius_kpc: np.ndarray,
    gradient_radius_kpc: np.ndarray,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = 2.0 / 3.0
    if model_name == "gg23_no_bh":
        base_y = 2.0 / 3.0
        mdot_ref_abs = np.full_like(initial_mass_msun, 30.0, dtype=float)
    else:
        base_y = 4.0 / 3.0
        mdot_ref_abs = np.full_like(initial_mass_msun, 45.0, dtype=float)
    y = np.full_like(initial_mass_msun, base_y, dtype=float)

    if "feh_gradient" in model_name:
        radius = np.asarray(gradient_radius_kpc, dtype=float)
        log_radius = np.log10(radius)
        feh_mask = radius < 10.0
        y[feh_mask] = 2.0 / 3.0 + 2.0 / 3.0 * log_radius[feh_mask]
        mdot_ref_abs[feh_mask] *= 2.0 / 3.0 + 1.0 / 3.0 * log_radius[feh_mask]

    mdot_abs = mdot_ref_abs / np.asarray(effective_radius_kpc, dtype=float)
    if "past_tidal" in model_name:
        past_mask = effective_radius_kpc > 4.0
        mdot_abs[past_mask] *= np.sqrt(effective_radius_kpc[past_mask] / 4.0)

    age_myr = AGE_GYR * 1.0e3
    mref = GG23_REFERENCE_MASS_MSUN
    mmin = mref * np.power(y * age_myr * mdot_abs / mref, 1.0 / x)
    tdis_myr = (1.0 / y) * mref / mdot_abs * np.power(initial_mass_msun / mref, x)
    remaining = 1.0 - age_myr / tdis_myr
    present = np.where(remaining > 0.0, initial_mass_msun * np.power(np.clip(remaining, 0.0, None), 1.0 / y), 0.0)
    return mmin, tdis_myr / 1.0e3, present, y, mdot_ref_abs


def relative_error(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return (value - reference) / np.maximum(np.abs(reference), 1.0e-300)


if __name__ == "__main__":
    main()
