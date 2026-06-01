from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    outputs_tables = project_root / "outputs" / "tables"

    rows: list[dict[str, object]] = []

    baseline_summary = json.loads((outputs_tables / "model_summary.json").read_text())
    rows.append(
        {
            "model_family_group": "baseline_truncated_imf",
            "model_class": "single_population",
            "model_name": "truncated_lognormal",
            "description": "baseline truncated lognormal",
            "bic": None,
            "aic": None,
            "log_likelihood": baseline_summary["lognormal"]["log_likelihood"],
            "total_initial_count": baseline_summary["lognormal_total_estimated_initial_count"],
        }
    )
    rows.append(
        {
            "model_family_group": "baseline_truncated_imf",
            "model_class": "single_population",
            "model_name": "truncated_powerlaw",
            "description": "baseline truncated power law",
            "bic": None,
            "aic": None,
            "log_likelihood": baseline_summary["powerlaw"]["log_likelihood"],
            "total_initial_count": baseline_summary["powerlaw_total_estimated_initial_count"],
        }
    )

    single_joint = pd.read_csv(outputs_tables / "joint_fixed_survival_model_summary.csv")
    for row in single_joint.itertuples(index=False):
        rows.append(
            {
                "model_family_group": "joint_fixed_survival_single_population",
                "model_class": "single_population",
                "model_name": f"{row.imf_family}+{row.radial_model}",
                "description": f"{row.imf_family} IMF + {row.radial_model} A(a)",
                "bic": float(row.bic),
                "aic": float(row.aic),
                "log_likelihood": float(row.log_likelihood),
                "total_initial_count": float(row.total_initial_count),
            }
        )

    shared_two = pd.read_csv(outputs_tables / "joint_fixed_survival_shared_imf_two_component_model_summary.csv")
    for row in shared_two.itertuples(index=False):
        rows.append(
            {
                "model_family_group": "joint_fixed_survival_two_component_shared_imf",
                "model_class": "two_component_shared_imf",
                "model_name": f"shared {row.imf_family}; in_situ {row.in_situ_radial_model}; accreted {row.accreted_radial_model}",
                "description": (
                    f"shared {row.imf_family} IMF; "
                    f"in-situ {row.in_situ_radial_model} A(a); "
                    f"accreted {row.accreted_radial_model} A(a)"
                ),
                "bic": float(row.bic),
                "aic": float(row.aic),
                "log_likelihood": float(row.log_likelihood),
                "total_initial_count": float(row.total_initial_count),
            }
        )

    separate_two = pd.read_csv(outputs_tables / "joint_fixed_survival_two_component_model_summary.csv")
    for row in separate_two.itertuples(index=False):
        rows.append(
            {
                "model_family_group": "joint_fixed_survival_two_component_separate_imf",
                "model_class": "two_component_separate_imf",
                "model_name": (
                    f"in_situ {row.in_situ_imf_family}+{row.in_situ_radial_model}; "
                    f"accreted {row.accreted_imf_family}+{row.accreted_radial_model}"
                ),
                "description": (
                    f"in-situ {row.in_situ_imf_family} IMF + {row.in_situ_radial_model} A(a); "
                    f"accreted {row.accreted_imf_family} IMF + {row.accreted_radial_model} A(a)"
                ),
                "bic": float(row.bic),
                "aic": float(row.aic),
                "log_likelihood": float(row.log_likelihood),
                "total_initial_count": float(row.total_initial_count),
            }
        )

    summary_table = pd.DataFrame(rows)
    summary_table = summary_table.sort_values(
        by=["model_family_group", "bic", "total_initial_count"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    output_path = outputs_tables / "total_initial_cluster_counts_by_model.csv"
    summary_table.to_csv(output_path, index=False)
    print(f"Wrote {len(summary_table)} model total-count rows to {output_path}")


if __name__ == "__main__":
    main()
