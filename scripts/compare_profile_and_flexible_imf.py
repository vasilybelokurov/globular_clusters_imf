from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the Schechter profile-likelihood IMF band to a flexible-IMF bootstrap band."
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=24,
        help="Number of non-parametric bootstrap resamples for the flexible IMF fit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for the bootstrap resampling.",
    )
    parser.add_argument(
        "--detectability-variant",
        choices=("baseline", "abs_longitude"),
        default="baseline",
        help="Which detectability model to use in the Schechter and flexible-IMF fits.",
    )
    parser.add_argument(
        "--smooth-survivability",
        action="store_true",
        help="Use the smooth eta_t-based survivability surface instead of the default hard-threshold grid.",
    )
    parser.add_argument(
        "--eta-t",
        type=float,
        default=1.0,
        help="Global lifetime renormalization used when --smooth-survivability is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(project_root / ".cache"))
    (project_root / ".mplconfig").mkdir(parents=True, exist_ok=True)
    (project_root / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)

    from globular_clusters_imf.flexible_imf import build_profile_vs_flexible_imf_comparison
    from globular_clusters_imf.model import fit_catalog_models
    from globular_clusters_imf.smooth_survivability import build_smooth_survivability_grid

    catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = project_root / "data" / "processed" / "baumgardt_gc_catalog.csv"

    catalog = pd.read_csv(catalog_path)
    survival_grid_override = None
    if args.smooth_survivability:
        prepared_catalog = fit_catalog_models(catalog, project_root)["catalog"]
        smooth_survival = build_smooth_survivability_grid(prepared_catalog, eta_t=float(args.eta_t))
        survival_grid_override = {
            "log_mass_grid": smooth_survival["log_mass_grid"],
            "log_a_grid": smooth_survival["log_a_grid"],
            "semi_major_axis_grid_kpc": smooth_survival["semi_major_axis_grid_kpc"],
            "survival_probability": smooth_survival["survival_probability"],
            "selection_offset_dex": 0.0,
            "bandwidth_log10_a_dex": smooth_survival["bandwidth_log10_a_dex"],
            "smooth_survivability_summary": smooth_survival["summary"],
        }
    variant_name = (
        "flexible_imf_bootstrap_comparison"
        if args.detectability_variant == "baseline"
        else "flexible_imf_bootstrap_comparison_abs_longitude"
    )
    if args.smooth_survivability:
        variant_name += "_smooth_survival_eta1"
    variant_root = project_root / "variants" / variant_name
    result = build_profile_vs_flexible_imf_comparison(
        catalog=catalog,
        output_root=variant_root,
        detectability_variant=args.detectability_variant,
        survival_grid_override=survival_grid_override,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.seed,
    )
    print(f"Wrote outputs to {variant_root}")
    print(result["comparison_summary"].to_string(index=False))
    print(f"Successful bootstrap fits: {result['summary_payload']['n_successful_bootstraps']} / {args.n_bootstrap}")


if __name__ == "__main__":
    main()
