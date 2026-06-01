from __future__ import annotations

import io
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ORBITS_URL = "https://people.smp.uq.edu.au/HolgerBaumgardt/globular/orbits.html"
PARAMETERS_URL = "https://people.smp.uq.edu.au/HolgerBaumgardt/globular/parameter.html"
LOCAL_GC_UPDATED_ENVVAR = "GC_IMF_UPDATED_CATALOG_PATH"
LOCAL_GC_PINSITU_ENVVAR = "GC_IMF_PINSITU_CATALOG_PATH"
DEFAULT_LOCAL_GC_UPDATED_FITS = Path.home() / "data" / "catalogues" / "gc_catalog_updated.fits"
DEFAULT_LOCAL_GC_PINSITU_FITS = Path.home() / "Documents" / "Work" / "lists" / "gc_catalog_pinsitu.fits"

DEFAULT_TIMEOUT_SECONDS = 60
SCIENTIFIC_VALUE_RE = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?)\s*(?:±\s*([+-]?\d+(?:\.\d+)?))?\s*·\s*(\d+)\s*$"
)
PLAIN_VALUE_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*(?:±\s*([+-]?\d+(?:\.\d+)?))?\s*$")


def fetch_and_prepare_catalog(project_root: Path) -> pd.DataFrame:
    data_raw = project_root / "data" / "raw"
    data_processed = project_root / "data" / "processed"
    data_raw.mkdir(parents=True, exist_ok=True)
    data_processed.mkdir(parents=True, exist_ok=True)

    orbits_html = download_text(ORBITS_URL, data_raw / "baumgardt_orbits.html")
    parameters_html = download_text(PARAMETERS_URL, data_raw / "baumgardt_parameters.html")

    orbits = parse_orbits_table(orbits_html)
    parameters = parse_parameters_table(parameters_html)

    merged = parameters.merge(
        orbits,
        on="cluster_label",
        how="inner",
        validate="one_to_one",
        suffixes=("_param", "_orbit"),
    )
    merged["cluster_name"] = merged["cluster_label"].map(canonical_cluster_name)
    merged["cluster_alias"] = merged["cluster_label"].map(cluster_alias)
    merged["semi_major_axis_kpc"] = 0.5 * (merged["r_peri_kpc"] + merged["r_apo_kpc"])
    merged["eccentricity"] = (
        (merged["r_apo_kpc"] - merged["r_peri_kpc"]) / (merged["r_apo_kpc"] + merged["r_peri_kpc"])
    ).clip(lower=0.0, upper=0.999)
    merged["initial_mass_msun"] = np.power(10.0, merged["log_initial_mass_msun"])
    merged["mass_loss_fraction"] = 1.0 - merged["present_mass_msun"] / merged["initial_mass_msun"]
    merged["radius_bin_paper"] = pd.cut(
        merged["semi_major_axis_kpc"],
        bins=[0.0, 3.0, 15.0, np.inf],
        labels=["inner_<3kpc", "mid_3_15kpc", "outer_>15kpc"],
        right=False,
    ).astype("string")

    cleaned = merged.sort_values("semi_major_axis_kpc").reset_index(drop=True)
    cleaned.to_csv(data_processed / "baumgardt_gc_catalog.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "n_clusters_joined": len(cleaned),
                "n_missing_joined": len(parameters) + len(orbits) - 2 * len(cleaned),
                "median_log_initial_mass_msun": cleaned["log_initial_mass_msun"].median(),
                "median_semi_major_axis_kpc": cleaned["semi_major_axis_kpc"].median(),
            }
        ]
    )
    summary.to_csv(data_processed / "baumgardt_gc_catalog_summary.csv", index=False)
    return cleaned


def export_local_gc_origin_flags(
    project_root: Path,
    updated_catalog_path: Path | None = None,
    pinsitu_catalog_path: Path | None = None,
) -> pd.DataFrame:
    data_processed = project_root / "data" / "processed"
    data_processed.mkdir(parents=True, exist_ok=True)

    updated_path, pinsitu_path = resolve_local_gc_origin_catalog_paths(
        updated_catalog_path=updated_catalog_path,
        pinsitu_catalog_path=pinsitu_catalog_path,
    )

    updated = read_fits_table(
        updated_path,
        columns=("NAME", "FLAG", "PROG_MA", "RA", "DEC"),
    ).rename(
        columns={
            "NAME": "source_name_updated",
            "FLAG": "origin_flag",
            "PROG_MA": "progenitor_group",
            "RA": "ra_deg",
            "DEC": "dec_deg",
        }
    )
    updated["cluster_name"] = updated["source_name_updated"].map(normalize_gc_name)
    updated["cluster_name_key"] = updated["cluster_name"].map(normalize_gc_name_key)
    updated["origin_flag"] = pd.to_numeric(updated["origin_flag"], errors="coerce").astype("Int64")
    updated["origin_label"] = updated["origin_flag"].map({1: "in_situ", 0: "accreted"}).astype("string")
    updated["is_in_situ"] = updated["origin_flag"] == 1
    updated["is_accreted"] = updated["origin_flag"] == 0
    updated["progenitor_group"] = updated["progenitor_group"].map(clean_string_or_na)

    if pinsitu_path is not None:
        legacy = read_fits_table(
            pinsitu_path,
            columns=("NAME", "FLAG", "INSITU_PROB"),
        ).rename(
            columns={
                "NAME": "source_name_pinsitu",
                "FLAG": "legacy_origin_flag",
                "INSITU_PROB": "legacy_insitu_prob",
            }
        )
        legacy["cluster_name"] = legacy["source_name_pinsitu"].map(normalize_gc_name)
        legacy["cluster_name_key"] = legacy["cluster_name"].map(normalize_gc_name_key)
        legacy["legacy_origin_flag"] = pd.to_numeric(legacy["legacy_origin_flag"], errors="coerce").astype("Int64")
        legacy["legacy_insitu_prob"] = pd.to_numeric(legacy["legacy_insitu_prob"], errors="coerce")
        legacy = legacy[
            ["cluster_name_key", "source_name_pinsitu", "legacy_origin_flag", "legacy_insitu_prob"]
        ].drop_duplicates(subset="cluster_name_key")
        updated = updated.merge(legacy, on="cluster_name_key", how="left", validate="one_to_one")
    else:
        updated["source_name_pinsitu"] = pd.Series(pd.NA, index=updated.index, dtype="string")
        updated["legacy_origin_flag"] = pd.Series(pd.NA, index=updated.index, dtype="Int64")
        updated["legacy_insitu_prob"] = np.nan

    origin_flags = updated.sort_values("cluster_name").reset_index(drop=True)
    origin_flags.to_csv(data_processed / "gc_origin_flags.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "updated_catalog_path": str(updated_path),
                "pinsitu_catalog_path": str(pinsitu_path) if pinsitu_path is not None else "",
                "n_clusters": len(origin_flags),
                "n_in_situ": int(origin_flags["is_in_situ"].sum()),
                "n_accreted": int(origin_flags["is_accreted"].sum()),
                "n_with_legacy_insitu_prob": int(origin_flags["legacy_insitu_prob"].notna().sum()),
                "n_with_progenitor_group": int(origin_flags["progenitor_group"].notna().sum()),
            }
        ]
    )
    summary.to_csv(data_processed / "gc_origin_flags_summary.csv", index=False)
    return origin_flags


def attach_local_origin_flags_to_baumgardt_catalog(
    project_root: Path,
    baumgardt_catalog: pd.DataFrame | None = None,
    origin_flags: pd.DataFrame | None = None,
) -> pd.DataFrame:
    data_processed = project_root / "data" / "processed"
    data_processed.mkdir(parents=True, exist_ok=True)

    if baumgardt_catalog is None:
        baumgardt_catalog = pd.read_csv(data_processed / "baumgardt_gc_catalog.csv")
    if origin_flags is None:
        origin_flags_path = data_processed / "gc_origin_flags.csv"
        if origin_flags_path.exists():
            origin_flags = pd.read_csv(origin_flags_path)
        else:
            origin_flags = export_local_gc_origin_flags(project_root)

    origin_columns = [
        "cluster_name_key",
        "origin_flag",
        "origin_label",
        "is_in_situ",
        "is_accreted",
        "progenitor_group",
        "legacy_origin_flag",
        "legacy_insitu_prob",
        "source_name_updated",
        "source_name_pinsitu",
    ]
    join_table = origin_flags[origin_columns].drop_duplicates(subset="cluster_name_key")

    augmented = baumgardt_catalog.copy()
    augmented["cluster_name_key"] = augmented["cluster_name"].map(normalize_gc_name_key)
    augmented = augmented.merge(join_table, on="cluster_name_key", how="left", validate="many_to_one")
    augmented["origin_flag_missing"] = augmented["origin_flag"].isna()
    augmented = augmented.sort_values("semi_major_axis_kpc").reset_index(drop=True)
    augmented.to_csv(data_processed / "baumgardt_gc_catalog_with_origin_flags.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "n_baumgardt_clusters": len(augmented),
                "n_origin_matches": int(augmented["origin_flag"].notna().sum()),
                "n_origin_missing": int(augmented["origin_flag_missing"].sum()),
                "n_with_legacy_insitu_prob": int(augmented["legacy_insitu_prob"].notna().sum()),
            }
        ]
    )
    summary.to_csv(data_processed / "baumgardt_gc_catalog_with_origin_flags_summary.csv", index=False)
    return augmented


def export_local_gc_chemistry_markers(
    project_root: Path,
    updated_catalog_path: Path | None = None,
) -> pd.DataFrame:
    data_processed = project_root / "data" / "processed"
    data_processed.mkdir(parents=True, exist_ok=True)

    updated_path, _ = resolve_local_gc_origin_catalog_paths(
        updated_catalog_path=updated_catalog_path,
        pinsitu_catalog_path=None,
    )
    chemistry = read_fits_table(
        updated_path,
        columns=(
            "NAME",
            "FEH",
            "MG_APO",
            "MG_ERR_APO",
            "MG_OTHER",
            "MG_ERR_OTHER",
            "AL_APO",
            "AL_ERR_APO",
            "AL_OTHER",
            "AL_ERR_OTHER",
        ),
    ).rename(
        columns={
            "NAME": "source_name_updated",
            "FEH": "local_feh",
            "MG_APO": "mgfe_apo",
            "MG_ERR_APO": "mgfe_err_apo",
            "MG_OTHER": "mgfe_other",
            "MG_ERR_OTHER": "mgfe_err_other",
            "AL_APO": "alfe_apo",
            "AL_ERR_APO": "alfe_err_apo",
            "AL_OTHER": "alfe_other",
            "AL_ERR_OTHER": "alfe_err_other",
        }
    )
    chemistry["cluster_name"] = chemistry["source_name_updated"].map(normalize_gc_name)
    chemistry["cluster_name_key"] = chemistry["cluster_name"].map(normalize_gc_name_key)
    chemistry["local_feh"] = pd.to_numeric(chemistry["local_feh"], errors="coerce")

    chemistry = combine_local_abundance_measurements(
        chemistry,
        value_prefix="mgfe",
        apo_source_label="APOGEE",
        other_source_label="literature",
    )
    chemistry = combine_local_abundance_measurements(
        chemistry,
        value_prefix="alfe",
        apo_source_label="APOGEE",
        other_source_label="literature",
    )

    chemistry = chemistry.sort_values("cluster_name").reset_index(drop=True)
    chemistry.to_csv(data_processed / "gc_chemistry_markers.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "updated_catalog_path": str(updated_path),
                "n_clusters": len(chemistry),
                "n_with_mgfe": int(chemistry["mgfe_combined"].notna().sum()),
                "n_with_alfe": int(chemistry["alfe_combined"].notna().sum()),
                "n_with_mgfe_and_alfe": int(
                    chemistry["mgfe_combined"].notna().mul(chemistry["alfe_combined"].notna()).sum()
                ),
                "mgfe_offset_dex": float(chemistry["mgfe_offset_dex"].iloc[0]),
                "mgfe_offset_scatter_dex": float(chemistry["mgfe_offset_scatter_dex"].iloc[0]),
                "alfe_offset_dex": float(chemistry["alfe_offset_dex"].iloc[0]),
                "alfe_offset_scatter_dex": float(chemistry["alfe_offset_scatter_dex"].iloc[0]),
            }
        ]
    )
    summary.to_csv(data_processed / "gc_chemistry_markers_summary.csv", index=False)
    return chemistry


def attach_local_gc_chemistry_to_baumgardt_catalog(
    project_root: Path,
    baumgardt_catalog: pd.DataFrame | None = None,
    chemistry_markers: pd.DataFrame | None = None,
) -> pd.DataFrame:
    data_processed = project_root / "data" / "processed"
    data_processed.mkdir(parents=True, exist_ok=True)

    if baumgardt_catalog is None:
        origin_augmented_path = data_processed / "baumgardt_gc_catalog_with_origin_flags.csv"
        if origin_augmented_path.exists():
            baumgardt_catalog = pd.read_csv(origin_augmented_path)
        else:
            baumgardt_catalog = attach_local_origin_flags_to_baumgardt_catalog(project_root)
    if chemistry_markers is None:
        chemistry_path = data_processed / "gc_chemistry_markers.csv"
        if chemistry_path.exists():
            chemistry_markers = pd.read_csv(chemistry_path)
        else:
            chemistry_markers = export_local_gc_chemistry_markers(project_root)

    chemistry_columns = [
        "cluster_name_key",
        "local_feh",
        "mgfe_combined",
        "mgfe_combined_err",
        "mgfe_combined_source",
        "mgfe_offset_dex",
        "mgfe_offset_scatter_dex",
        "alfe_combined",
        "alfe_combined_err",
        "alfe_combined_source",
        "alfe_offset_dex",
        "alfe_offset_scatter_dex",
    ]
    join_table = chemistry_markers[chemistry_columns].drop_duplicates(subset="cluster_name_key")

    augmented = baumgardt_catalog.copy()
    augmented["cluster_name_key"] = augmented["cluster_name"].map(normalize_gc_name_key)
    augmented = augmented.merge(join_table, on="cluster_name_key", how="left", validate="many_to_one")
    augmented["has_mgfe"] = augmented["mgfe_combined"].notna()
    augmented["has_alfe"] = augmented["alfe_combined"].notna()
    augmented["has_any_chemistry"] = augmented["has_mgfe"] | augmented["has_alfe"]
    augmented["has_mgfe_and_alfe"] = augmented["has_mgfe"] & augmented["has_alfe"]
    augmented = augmented.sort_values("semi_major_axis_kpc").reset_index(drop=True)
    augmented.to_csv(data_processed / "baumgardt_gc_catalog_with_origin_and_chemistry.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "n_baumgardt_clusters": len(augmented),
                "n_with_mgfe": int(augmented["has_mgfe"].sum()),
                "n_with_alfe": int(augmented["has_alfe"].sum()),
                "n_with_mgfe_and_alfe": int(augmented["has_mgfe_and_alfe"].sum()),
            }
        ]
    )
    summary.to_csv(
        data_processed / "baumgardt_gc_catalog_with_origin_and_chemistry_summary.csv",
        index=False,
    )
    return augmented


def combine_local_abundance_measurements(
    chemistry: pd.DataFrame,
    value_prefix: str,
    apo_source_label: str,
    other_source_label: str,
) -> pd.DataFrame:
    apo_value = pd.to_numeric(chemistry[f"{value_prefix}_apo"], errors="coerce")
    apo_err = pd.to_numeric(chemistry[f"{value_prefix}_err_apo"], errors="coerce")
    other_value = pd.to_numeric(chemistry[f"{value_prefix}_other"], errors="coerce")
    other_err = pd.to_numeric(chemistry[f"{value_prefix}_err_other"], errors="coerce")

    has_apo = apo_value.notna() & apo_err.notna() & (apo_err > 0.0)
    has_other = other_value.notna() & other_err.notna() & (other_err > 0.0) & (other_value > -90.0)
    has_both = has_apo & has_other

    if has_both.any():
        differences = (other_value[has_both] - apo_value[has_both]).to_numpy(dtype=float)
        offset = float(np.nanmedian(differences))
        scatter = robust_scatter(differences - offset)
    else:
        offset = 0.0
        scatter = 0.0

    combined_value = pd.Series(np.nan, index=chemistry.index, dtype=float)
    combined_error = pd.Series(np.nan, index=chemistry.index, dtype=float)
    combined_source = pd.Series(pd.NA, index=chemistry.index, dtype="string")

    combined_value.loc[has_apo] = apo_value.loc[has_apo]
    combined_error.loc[has_apo] = apo_err.loc[has_apo]
    combined_source.loc[has_apo] = apo_source_label

    other_only = ~has_apo & has_other
    combined_value.loc[other_only] = other_value.loc[other_only] - offset
    combined_error.loc[other_only] = np.sqrt(np.square(other_err.loc[other_only]) + scatter**2)
    combined_source.loc[other_only] = other_source_label

    chemistry[f"{value_prefix}_combined"] = combined_value
    chemistry[f"{value_prefix}_combined_err"] = combined_error
    chemistry[f"{value_prefix}_combined_source"] = combined_source
    chemistry[f"{value_prefix}_offset_dex"] = offset
    chemistry[f"{value_prefix}_offset_scatter_dex"] = scatter
    return chemistry


def robust_scatter(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    mad = float(np.nanmedian(np.abs(finite - np.nanmedian(finite))))
    if mad > 0.0:
        return 1.4826 * mad
    if finite.size > 1:
        return float(np.nanstd(finite, ddof=1))
    return 0.0


def download_text(url: str, destination: Path) -> str:
    response = requests.get(url, timeout=DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()
    text = response.text
    destination.write_text(text)
    return text


def parse_orbits_table(html_text: str) -> pd.DataFrame:
    table = pd.read_html(io.StringIO(html_text))[1].copy()
    table = table.rename(
        columns={
            "ClusterName": "cluster_label",
            "RA[°]": "ra_deg",
            "DEC[°]": "dec_deg",
            "l[°]": "galactic_l_deg",
            "b[°]": "galactic_b_deg",
            "R☉[kpc]": "r_sun_display",
            "RGC[kpc]": "r_gc_kpc_display",
            "RV[km/sec]": "rv_kms_display",
            "μαcosδ[mas/yr]": "pmra_masyr_display",
            "μδ[mas/yr]": "pmdec_masyr_display",
            "ρμαμδ": "pm_correlation",
            "X[kpc]": "x_kpc_display",
            "Y[kpc]": "y_kpc_display",
            "Z[kpc]": "z_kpc_display",
            "U[km/sec]": "u_kms_display",
            "V[km/sec]": "v_kms_display",
            "W[km/sec]": "w_kms_display",
            "RPeri[kpc]": "r_peri_kpc_display",
            "RApo[kpc]": "r_apo_kpc_display",
        }
    )
    table["cluster_label"] = table["cluster_label"].map(normalize_whitespace)
    for source_column, target_column in (
        ("r_sun_display", "r_sun_kpc"),
        ("r_gc_kpc_display", "r_gc_kpc"),
        ("rv_kms_display", "rv_kms"),
        ("pmra_masyr_display", "pmra_masyr"),
        ("pmdec_masyr_display", "pmdec_masyr"),
        ("x_kpc_display", "x_kpc"),
        ("y_kpc_display", "y_kpc"),
        ("z_kpc_display", "z_kpc"),
        ("u_kms_display", "u_kms"),
        ("v_kms_display", "v_kms"),
        ("w_kms_display", "w_kms"),
        ("r_peri_kpc_display", "r_peri_kpc"),
        ("r_apo_kpc_display", "r_apo_kpc"),
    ):
        table[target_column] = table[source_column].map(parse_value)
    return table


def parse_parameters_table(html_text: str) -> pd.DataFrame:
    table = pd.read_html(io.StringIO(html_text))[1].copy()
    table = table.rename(
        columns={
            "ClusterName": "cluster_label",
            "RA": "ra_deg_param",
            "DEC": "dec_deg_param",
            "R☉[kpc]": "r_sun_display_param",
            "RGC[kpc]": "r_gc_kpc_display_param",
            "NRV": "n_rv",
            "NPM": "n_pm",
            "Mass[M☉]": "present_mass_display",
            "V[mag]": "v_mag_display",
            "M/LV[M☉/L☉]": "mass_to_light_display",
            "rc[pc]": "core_radius_pc",
            "rh,l[pc]": "half_light_radius_pc",
            "rh,m[pc]": "half_mass_radius_pc",
            "rt[pc]": "tidal_radius_pc",
            "log ρc[M☉/pc3]": "log_core_density",
            "log ρh,m[M☉/pc3]": "log_half_mass_density",
            "log Trh[yr]": "log_relaxation_time_yr",
            "log Mini[M☉]": "log_initial_mass_msun",
            "TDiss[Gyr]": "remaining_dissolution_time_gyr",
            "Mass range[M☉]": "mass_range_display",
            "MFSlope": "mf_slope_display",
            "σ0[km/sec]": "sigma0_kms",
            "vesc[km/sec]": "vesc_kms",
            "ηc": "eta_core",
            "ηh": "eta_half_mass",
            "ARot[km/sec]": "rotation_amplitude_display",
            "PRot[%]": "rotation_probability_percent",
        }
    )
    table["cluster_label"] = table["cluster_label"].map(normalize_whitespace)
    for source_column, target_column in (
        ("r_sun_display_param", "r_sun_kpc_param"),
        ("r_gc_kpc_display_param", "r_gc_kpc_param"),
        ("present_mass_display", "present_mass_msun"),
        ("v_mag_display", "v_mag"),
        ("mass_to_light_display", "mass_to_light_v"),
        ("mf_slope_display", "mf_slope"),
        ("rotation_amplitude_display", "rotation_amplitude_kms"),
    ):
        table[target_column] = table[source_column].map(parse_value)
    for column in (
        "ra_deg_param",
        "dec_deg_param",
        "core_radius_pc",
        "half_light_radius_pc",
        "half_mass_radius_pc",
        "tidal_radius_pc",
        "log_core_density",
        "log_half_mass_density",
        "log_relaxation_time_yr",
        "log_initial_mass_msun",
        "remaining_dissolution_time_gyr",
        "sigma0_kms",
        "vesc_kms",
        "eta_core",
        "eta_half_mass",
        "rotation_probability_percent",
    ):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    return table


def normalize_whitespace(value: object) -> str:
    return " ".join(str(value).split())


def normalize_gc_name(value: object) -> str:
    return normalize_whitespace(value).strip()


def normalize_gc_name_key(value: object) -> str:
    text = normalize_gc_name(value).upper()
    patterns = (
        (r"^NGC\s+(\d+)\b.*$", r"NGC \1"),
        (r"^PAL\s+(\d+)\b.*$", r"PAL \1"),
        (r"^TER\s+(\d+)\b.*$", r"TERZAN \1"),
        (r"^TERZAN\s+(\d+)\b.*$", r"TERZAN \1"),
        (r"^DJOR\s+(\d+)\b.*$", r"DJORG \1"),
        (r"^DJORG\s+(\d+)\b.*$", r"DJORG \1"),
        (r"^BH\s+(\d+)\b.*$", r"BH \1"),
        (r"^GRAN\s+(\d+)\b.*$", r"GRAN \1"),
        (r"^AM\s+(\d+)\b.*$", r"AM \1"),
        (r"^ESO\s+(\S+)\b.*$", r"ESO \1"),
        (r"^FSR\s+(\d+)\b.*$", r"FSR \1"),
        (r"^LILLER\s+(\d+)\b.*$", r"LILLER \1"),
        (r"^WHITING\s+(\d+)\b.*$", r"WHITING \1"),
        (r"^IC\s+(\d+)\b.*$", r"IC \1"),
        (r"^RUP\s+(\d+)\b.*$", r"RUP \1"),
        (r"^LAEVENS\s+(\d+)\b.*$", r"LAEVENS \1"),
    )
    for pattern, replacement in patterns:
        if re.match(pattern, text):
            return re.sub(pattern, replacement, text)
    return text


def clean_string_or_na(value: object) -> str | pd.NA:
    text = normalize_whitespace(value)
    return text if text else pd.NA


def resolve_local_gc_origin_catalog_paths(
    updated_catalog_path: Path | None = None,
    pinsitu_catalog_path: Path | None = None,
) -> tuple[Path, Path | None]:
    updated_path = resolve_catalog_path(
        explicit_path=updated_catalog_path,
        envvar_name=LOCAL_GC_UPDATED_ENVVAR,
        default_path=DEFAULT_LOCAL_GC_UPDATED_FITS,
        required=True,
    )
    legacy_path = resolve_catalog_path(
        explicit_path=pinsitu_catalog_path,
        envvar_name=LOCAL_GC_PINSITU_ENVVAR,
        default_path=DEFAULT_LOCAL_GC_PINSITU_FITS,
        required=False,
    )
    return updated_path, legacy_path


def resolve_catalog_path(
    explicit_path: Path | None,
    envvar_name: str,
    default_path: Path,
    required: bool,
) -> Path | None:
    if explicit_path is not None:
        candidate = explicit_path
    else:
        envvar_value = os.environ.get(envvar_name)
        candidate = Path(envvar_value).expanduser() if envvar_value else default_path

    candidate = candidate.expanduser()
    if candidate.exists():
        return candidate
    if required:
        raise FileNotFoundError(
            f"Required catalog not found at {candidate}. "
            f"Override with the {envvar_name} environment variable if needed."
        )
    return None


def read_fits_table(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    try:
        from astropy.table import Table
    except ImportError as error:
        raise ImportError(
            "Reading local GC FITS catalogs requires astropy. "
            "Install project dependencies with `pip install -e .`."
        ) from error

    table = Table.read(path)
    missing_columns = [column for column in columns if column not in table.colnames]
    if missing_columns:
        raise KeyError(f"Missing columns {missing_columns} in catalog {path}")
    frame = table[list(columns)].to_pandas()
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(lambda value: value.decode() if isinstance(value, bytes) else value)
    return frame


def parse_value(value: object) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        return float(value)
    if value is None or pd.isna(value):
        return np.nan

    text = normalize_numeric_text(str(value))
    scientific_match = SCIENTIFIC_VALUE_RE.match(text)
    if scientific_match:
        scale = parse_power_of_ten_suffix(scientific_match.group(3))
        return float(scientific_match.group(1)) * scale

    plain_match = PLAIN_VALUE_RE.match(text)
    if plain_match:
        return float(plain_match.group(1))

    first_number = re.search(r"[+-]?\d+(?:\.\d+)?", text)
    if first_number:
        return float(first_number.group(0))
    return np.nan


def normalize_numeric_text(text: str) -> str:
    return (
        text.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("\u2212", "-")
        .strip()
    )


def parse_power_of_ten_suffix(text: str) -> float:
    digits = re.sub(r"\D", "", text)
    if digits.startswith("10") and len(digits) > 2:
        exponent = int(digits[2:])
        return float(10**exponent)
    return float(digits)


def canonical_cluster_name(label: str) -> str:
    tokens = label.split()
    if len(tokens) >= 2 and tokens[0] == "NGC":
        return " ".join(tokens[:2])
    return label


def cluster_alias(label: str) -> str | None:
    tokens = label.split()
    if len(tokens) >= 3 and tokens[0] == "NGC":
        return " ".join(tokens[2:])
    return None
