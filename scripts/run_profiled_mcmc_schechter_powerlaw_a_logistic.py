from __future__ import annotations

import json
import math
import os
import pickle
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

from scan_schechter_survival_time_multipliers import _row_from_result


BEST_FIT_ROOT = PROJECT_ROOT / "variants" / "global_profiled_schechter_powerlaw_a_logistic"
OUTPUT_ROOT = PROJECT_ROOT / "variants" / "global_profiled_schechter_powerlaw_a_logistic_mcmc"
FIGURES_DIR = OUTPUT_ROOT / "outputs" / "figures"
TABLES_DIR = OUTPUT_ROOT / "outputs" / "tables"

N_ITERATIONS = 12
LOG_MASS_MIN = 4.0
ETA_BOUNDS = (0.1, 3.0)
ALPHA_BOUNDS = (-4.0, -0.2)
LOGMC_BOUNDS = (4.5, 7.5)
PRIOR_BOUNDS = np.array([ETA_BOUNDS, ALPHA_BOUNDS, LOGMC_BOUNDS], dtype=float)

N_CHAINS = 4
N_STEPS = 220
BURN_IN = 60
THIN = 2
ADAPT_UNTIL = 60
ADAPT_EVERY = 15


def _round_key(theta: np.ndarray) -> tuple[float, float, float]:
    return tuple(round(float(value), 6) for value in theta)


def _within_bounds(theta: np.ndarray) -> bool:
    theta = np.asarray(theta, dtype=float)
    return bool(np.all(theta >= PRIOR_BOUNDS[:, 0]) and np.all(theta <= PRIOR_BOUNDS[:, 1]))


def _survival_grid_override_from_smooth_survival(smooth_survival: dict[str, object]) -> dict[str, object]:
    return {
        "log_mass_grid": np.asarray(smooth_survival["log_mass_grid"], dtype=float),
        "log_a_grid": np.asarray(smooth_survival["log_a_grid"], dtype=float),
        "semi_major_axis_grid_kpc": np.asarray(smooth_survival["semi_major_axis_grid_kpc"], dtype=float),
        "survival_probability": np.asarray(smooth_survival["survival_probability"], dtype=float),
        "selection_offset_dex": 0.0,
        "bandwidth_log10_a_dex": float(smooth_survival["bandwidth_log10_a_dex"]),
        "smooth_survivability_summary": smooth_survival["summary"],
    }


def _start_state_from_result(result: dict[str, object]) -> dict[str, np.ndarray]:
    return {
        "completeness": np.asarray(result["final_completeness_raw_parameters"], dtype=float),
        "radial": np.asarray(result["final_payload"]["radial_parameters_raw"], dtype=float),
    }


def _evaluate_theta(
    *,
    prepared_catalog: pd.DataFrame,
    spec,
    theta: np.ndarray,
    start_state: dict[str, np.ndarray] | None,
) -> dict[str, object]:
    from globular_clusters_imf.detectability_longitude_model import fit_single_component_detectability_em_with_abs_longitude
    from globular_clusters_imf.smooth_survivability import build_smooth_survivability_grid

    eta_t, alpha, log_mc = [float(value) for value in theta]
    smooth_survival = build_smooth_survivability_grid(
        prepared_catalog,
        eta_t=eta_t,
        surface_model="logistic",
    )
    result = fit_single_component_detectability_em_with_abs_longitude(
        prepared_catalog,
        project_root=OUTPUT_ROOT,
        spec=spec,
        n_iterations=N_ITERATIONS,
        fixed_imf_params=np.array([alpha, log_mc], dtype=float),
        start_completeness_raw_parameters=None if start_state is None else start_state["completeness"],
        start_radial_params=None if start_state is None else start_state["radial"],
        survival_grid_override=_survival_grid_override_from_smooth_survival(smooth_survival),
    )
    row = _row_from_result(
        eta_t=eta_t,
        radial_model=spec.radial_model,
        survival_summary=smooth_survival["summary"],
        result=result,
        log_mass_min=LOG_MASS_MIN,
    )
    radial_params = result["final_payload"]["model"]["radial_parameters"]
    row["beta_log10_a"] = float(radial_params.get("beta_log10_a", np.nan))
    row["gamma_linear_a"] = float(radial_params.get("gamma_linear_a", np.nan))
    row["input_alpha_dndm"] = alpha
    row["input_log10_m_c_msun"] = log_mc
    row["surface_model"] = "logistic"
    return {
        "theta": np.asarray(theta, dtype=float),
        "log_posterior": float(row["log_likelihood"]),
        "row": row,
        "result": result,
        "start_state": _start_state_from_result(result),
    }


def _draw_initial_positions(center: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    offsets = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.05, 0.07, -0.03],
            [-0.06, -0.06, 0.03],
            [0.09, -0.08, 0.05],
        ],
        dtype=float,
    )
    positions = []
    for offset in offsets[:N_CHAINS]:
        theta = center + offset
        theta = np.minimum(np.maximum(theta, PRIOR_BOUNDS[:, 0] + 1.0e-6), PRIOR_BOUNDS[:, 1] - 1.0e-6)
        theta += rng.normal(scale=np.array([0.01, 0.015, 0.008]), size=3)
        theta = np.minimum(np.maximum(theta, PRIOR_BOUNDS[:, 0] + 1.0e-6), PRIOR_BOUNDS[:, 1] - 1.0e-6)
        positions.append(theta)
    return positions


def _compute_rhat(chains: np.ndarray) -> float:
    # chains shape: (n_chains, n_samples)
    m, n = chains.shape
    if m < 2 or n < 2:
        return float("nan")
    chain_means = np.mean(chains, axis=1)
    chain_vars = np.var(chains, axis=1, ddof=1)
    b = n * np.var(chain_means, ddof=1)
    w = np.mean(chain_vars)
    if w <= 0.0:
        return float("nan")
    var_hat = ((n - 1) / n) * w + (1 / n) * b
    return float(np.sqrt(var_hat / w))


def _corner_plot(samples: pd.DataFrame, mle_row: dict[str, object], output_path: Path) -> None:
    columns = [
        ("eta_t", r"$\eta_t$"),
        ("input_alpha_dndm", r"$\alpha$"),
        ("input_log10_m_c_msun", r"$\log_{10}(M_c/{\rm M}_\odot)$"),
        ("gamma_linear_a", r"$\gamma_a$"),
    ]
    n_dim = len(columns)
    fig, axes = plt.subplots(n_dim, n_dim, figsize=(10.5, 10.5))
    mle_values = [float(mle_row[name]) for name, _ in columns]

    for i, (x_name, x_label) in enumerate(columns):
        for j, (y_name, y_label) in enumerate(columns):
            ax = axes[i, j]
            if i < j:
                ax.axis("off")
                continue
            if i == j:
                values = np.asarray(samples[x_name], dtype=float)
                ax.hist(values, bins=28, color="#4c72b0", alpha=0.80, histtype="stepfilled")
                q16, q50, q84 = np.quantile(values, [0.16, 0.50, 0.84])
                ax.axvline(q50, color="black", linewidth=1.0)
                ax.axvline(q16, color="0.5", linewidth=0.8, linestyle="--")
                ax.axvline(q84, color="0.5", linewidth=0.8, linestyle="--")
                ax.axvline(mle_values[i], color="#d62728", linewidth=1.0, linestyle="-")
            else:
                x = np.asarray(samples[y_name], dtype=float)
                y = np.asarray(samples[x_name], dtype=float)
                ax.hist2d(x, y, bins=28, cmap="Blues")
                ax.plot(float(mle_row[y_name]), float(mle_row[x_name]), marker="x", color="#d62728", markersize=6)
            if i == n_dim - 1:
                ax.set_xlabel(y_label if i != j else x_label)
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(x_label)
            elif i == j and j == 0:
                ax.set_ylabel("Count")
            else:
                if j != 0:
                    ax.set_yticklabels([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _trace_plot(chain_table: pd.DataFrame, output_path: Path) -> None:
    columns = [
        ("eta_t", r"$\eta_t$"),
        ("input_alpha_dndm", r"$\alpha$"),
        ("input_log10_m_c_msun", r"$\log_{10} M_c$"),
        ("gamma_linear_a", r"$\gamma_a$"),
    ]
    fig, axes = plt.subplots(len(columns), 1, figsize=(9.0, 8.5), sharex=True)
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    for ax, (column, label) in zip(axes, columns):
        for chain_id, color in zip(sorted(chain_table["chain"].unique()), colors):
            subset = chain_table.loc[chain_table["chain"] == chain_id]
            ax.plot(subset["step"], subset[column], color=color, alpha=0.8, linewidth=1.0)
        ax.set_ylabel(label)
        ax.axvline(BURN_IN, color="0.6", linestyle="--", linewidth=1.0)
    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    from globular_clusters_imf.joint_model import JointModelSpec
    from globular_clusters_imf.model import fit_catalog_models

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260527)
    spec = JointModelSpec(imf_family="schechter", radial_model="powerlaw_a")

    catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    prepared_catalog = fit_catalog_models(catalog, OUTPUT_ROOT)["catalog"]

    with (BEST_FIT_ROOT / "outputs" / "tables" / "best_result.pkl").open("rb") as handle:
        best_result = pickle.load(handle)

    best_summary = json.loads((BEST_FIT_ROOT / "outputs" / "tables" / "summary.json").read_text())
    mle_theta = np.array(
        [
            float(best_summary["global_best"]["eta_t"]),
            float(best_result["final_payload"]["model"]["imf_parameters"]["alpha_dndm"]),
            float(best_result["final_payload"]["model"]["imf_parameters"]["log10_m_c_msun"]),
        ],
        dtype=float,
    )
    mle_start_state = _start_state_from_result(best_result)

    cache: dict[tuple[float, float, float], dict[str, object]] = {}
    mle_entry = _evaluate_theta(
        prepared_catalog=prepared_catalog,
        spec=spec,
        theta=mle_theta,
        start_state=mle_start_state,
    )
    cache[_round_key(mle_theta)] = mle_entry

    initial_positions = _draw_initial_positions(mle_theta, rng)
    current_states: list[dict[str, object]] = []
    for theta0 in initial_positions:
        key = _round_key(theta0)
        if key in cache:
            entry = cache[key]
        else:
            entry = _evaluate_theta(
                prepared_catalog=prepared_catalog,
                spec=spec,
                theta=np.asarray(theta0, dtype=float),
                start_state=mle_start_state,
            )
            cache[key] = entry
        current_states.append(entry)

    proposal_scales = np.tile(np.array([0.08, 0.10, 0.05], dtype=float), (N_CHAINS, 1))
    chain_accepts = np.zeros((N_CHAINS, N_STEPS), dtype=bool)
    records: list[dict[str, object]] = []

    for step in range(N_STEPS):
        for chain in range(N_CHAINS):
            current = current_states[chain]
            theta_prop = np.asarray(current["theta"], dtype=float) + rng.normal(scale=proposal_scales[chain], size=3)
            accepted = False
            proposal_entry = None
            if _within_bounds(theta_prop):
                key = _round_key(theta_prop)
                if key in cache:
                    proposal_entry = cache[key]
                else:
                    try:
                        proposal_entry = _evaluate_theta(
                            prepared_catalog=prepared_catalog,
                            spec=spec,
                            theta=theta_prop,
                            start_state=current["start_state"],
                        )
                    except Exception:
                        proposal_entry = None
                    if proposal_entry is not None:
                        cache[key] = proposal_entry
                if proposal_entry is not None:
                    delta = float(proposal_entry["log_posterior"]) - float(current["log_posterior"])
                    if math.log(rng.uniform()) < min(0.0, delta):
                        current_states[chain] = proposal_entry
                        current = proposal_entry
                        accepted = True
            chain_accepts[chain, step] = accepted
            row = dict(current["row"])
            row["chain"] = int(chain)
            row["step"] = int(step)
            row["accepted"] = bool(accepted)
            row["proposal_scale_eta_t"] = float(proposal_scales[chain, 0])
            row["proposal_scale_alpha"] = float(proposal_scales[chain, 1])
            row["proposal_scale_logmc"] = float(proposal_scales[chain, 2])
            records.append(row)

        if step + 1 <= ADAPT_UNTIL and (step + 1) % ADAPT_EVERY == 0:
            window = chain_accepts[:, step + 1 - ADAPT_EVERY : step + 1].mean(axis=1)
            for chain in range(N_CHAINS):
                if window[chain] < 0.15:
                    proposal_scales[chain] *= 0.80
                elif window[chain] > 0.40:
                    proposal_scales[chain] *= 1.20
                proposal_scales[chain] = np.clip(proposal_scales[chain], [0.01, 0.02, 0.01], [0.25, 0.30, 0.15])

        if (step + 1) % 20 == 0 or step == 0:
            current_best = max(current_states, key=lambda entry: float(entry["log_posterior"]))
            print(
                f"step={step+1:03d} "
                f"best logL={float(current_best['log_posterior']):.3f} "
                f"eta_t={float(current_best['theta'][0]):.3f} alpha={float(current_best['theta'][1]):.3f} "
                f"logMc={float(current_best['theta'][2]):.3f} "
                f"accept={float(chain_accepts[:, max(0, step-19):step+1].mean()):.3f}"
            )

    chain_table = pd.DataFrame(records)
    chain_table.to_csv(TABLES_DIR / "mcmc_chain.csv", index=False)

    posterior_parts = []
    for _, frame in chain_table.loc[chain_table["step"] >= BURN_IN].groupby("chain"):
        posterior_parts.append(frame.iloc[::THIN])
    posterior_table = pd.concat(posterior_parts, ignore_index=True)
    posterior_table.to_csv(TABLES_DIR / "mcmc_posterior_samples.csv", index=False)

    summary_rows = []
    summary_columns = [
        "eta_t",
        "input_alpha_dndm",
        "input_log10_m_c_msun",
        "gamma_linear_a",
        "final_total_initial_count_above_log10_4",
        "final_total_initial_stellar_mass_above_log10_4_msun",
        "mean_detectability_above_log10_4",
        "log_likelihood",
    ]
    for column in summary_columns:
        values = np.asarray(posterior_table[column], dtype=float)
        q16, q50, q84 = np.quantile(values, [0.16, 0.50, 0.84])
        summary_rows.append(
            {
                "parameter": column,
                "q16": float(q16),
                "q50": float(q50),
                "q84": float(q84),
                "minus": float(q50 - q16),
                "plus": float(q84 - q50),
            }
        )
    summary_table = pd.DataFrame(summary_rows)
    summary_table.to_csv(TABLES_DIR / "posterior_summary.csv", index=False)

    rhat_payload = {}
    for column in ["eta_t", "input_alpha_dndm", "input_log10_m_c_msun", "gamma_linear_a"]:
        pivot = (
            posterior_table.pivot_table(index="step", columns="chain", values=column, aggfunc="last")
            .dropna()
            .to_numpy()
            .T
        )
        rhat_payload[column] = _compute_rhat(pivot)

    acceptance_by_chain = {
        str(chain): float(chain_accepts[chain].mean())
        for chain in range(N_CHAINS)
    }

    best_posterior_row = posterior_table.sort_values("log_likelihood", ascending=False).iloc[0].to_dict()
    _corner_plot(posterior_table, mle_entry["row"], FIGURES_DIR / "profiled_posterior_corner.png")
    _trace_plot(chain_table, FIGURES_DIR / "profiled_posterior_traces.png")

    summary_payload = {
        "sampler": "random_walk_metropolis_profiled",
        "surface_model": "logistic",
        "model_spec": {"imf_family": spec.imf_family, "radial_model": spec.radial_model},
        "n_chains": N_CHAINS,
        "n_steps": N_STEPS,
        "burn_in": BURN_IN,
        "thin": THIN,
        "acceptance_by_chain": acceptance_by_chain,
        "rhat": rhat_payload,
        "mle_reference": {
            "eta_t": float(mle_theta[0]),
            "alpha_dndm": float(mle_theta[1]),
            "log10_m_c_msun": float(mle_theta[2]),
            "gamma_linear_a": float(mle_entry["row"]["gamma_linear_a"]),
            "log_likelihood": float(mle_entry["log_posterior"]),
        },
        "best_posterior_sample": json.loads(pd.Series(best_posterior_row).to_json()),
        "posterior_summary": summary_table.to_dict(orient="records"),
        "n_unique_profile_evaluations": int(len(cache)),
    }
    (TABLES_DIR / "summary.json").write_text(json.dumps(summary_payload, indent=2))

    print(FIGURES_DIR / "profiled_posterior_corner.png")
    print(FIGURES_DIR / "profiled_posterior_traces.png")
    print(TABLES_DIR / "posterior_summary.csv")
    print(TABLES_DIR / "summary.json")


if __name__ == "__main__":
    main()
