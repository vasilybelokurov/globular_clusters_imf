from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402
from scipy import optimize, special  # noqa: E402

from globular_clusters_imf.detectability_model import fit_present_mass_proxy_model  # noqa: E402
from globular_clusters_imf.model import fit_catalog_models  # noqa: E402


def _load_fit_catalog() -> pd.DataFrame:
    catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"
    if not catalog_path.exists():
        catalog_path = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    return fit_catalog_models(catalog, PROJECT_ROOT)["catalog"]


def _polynomial_predictions(catalog: pd.DataFrame) -> np.ndarray:
    proxy = fit_present_mass_proxy_model(catalog)
    log_initial_mass = catalog["log_initial_mass_msun"].to_numpy(dtype=float)
    log_a = np.log10(catalog["semi_major_axis_kpc"].to_numpy(dtype=float))
    z_mass = log_initial_mass - float(proxy.log_mass_mean)
    z_a = log_a - float(proxy.log_a_mean)
    c0, c1, c2, c3, c4 = np.asarray(proxy.coefficients, dtype=float)
    log_mass_ratio = c0 + c1 * z_mass + c2 * z_a + c3 * z_mass * z_a + c4 * np.square(z_a)
    log_mass_ratio = np.clip(log_mass_ratio, float(proxy.log_mass_ratio_min), 0.0)
    return log_initial_mass + log_mass_ratio


def _fit_polynomial_delta(
    train_log_initial_mass: np.ndarray,
    train_log_a: np.ndarray,
    train_delta: np.ndarray,
    eval_log_initial_mass: np.ndarray,
    eval_log_a: np.ndarray,
) -> np.ndarray:
    log_mass_mean = float(np.mean(train_log_initial_mass))
    log_a_mean = float(np.mean(train_log_a))
    z_mass = np.asarray(train_log_initial_mass, dtype=float) - log_mass_mean
    z_a = np.asarray(train_log_a, dtype=float) - log_a_mean
    design_matrix = np.column_stack(
        [
            np.ones_like(z_mass),
            z_mass,
            z_a,
            z_mass * z_a,
            np.square(z_a),
        ]
    )
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, train_delta, rcond=None)

    eval_z_mass = np.asarray(eval_log_initial_mass, dtype=float) - log_mass_mean
    eval_z_a = np.asarray(eval_log_a, dtype=float) - log_a_mean
    eval_design = np.column_stack(
        [
            np.ones_like(eval_z_mass),
            eval_z_mass,
            eval_z_a,
            eval_z_mass * eval_z_a,
            np.square(eval_z_a),
        ]
    )
    delta_min = float(np.min(train_delta))
    return np.clip(eval_design @ coefficients, delta_min, 0.0)


def _polynomial_leave_one_out_predictions(catalog: pd.DataFrame) -> np.ndarray:
    log_initial_mass = catalog["log_initial_mass_msun"].to_numpy(dtype=float)
    log_a = np.log10(catalog["semi_major_axis_kpc"].to_numpy(dtype=float))
    log_present_mass = np.log10(catalog["present_mass_msun"].to_numpy(dtype=float))
    delta = log_present_mass - log_initial_mass
    predictions = np.empty_like(log_present_mass)
    for index in range(len(catalog)):
        train = np.ones(len(catalog), dtype=bool)
        train[index] = False
        delta_prediction = _fit_polynomial_delta(
            log_initial_mass[train],
            log_a[train],
            delta[train],
            log_initial_mass[index : index + 1],
            log_a[index : index + 1],
        )[0]
        predictions[index] = log_initial_mass[index] + delta_prediction
    return predictions


def _standardized_features(catalog: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    log_initial_mass = catalog["log_initial_mass_msun"].to_numpy(dtype=float)
    log_a = np.log10(catalog["semi_major_axis_kpc"].to_numpy(dtype=float))
    means = np.array([np.mean(log_initial_mass), np.mean(log_a)])
    scales = np.array([np.std(log_initial_mass), np.std(log_a)])
    scales = np.where(scales > 0.0, scales, 1.0)
    features = np.column_stack([log_initial_mass, log_a])
    metadata = {
        "log_initial_mass_mean": float(means[0]),
        "log_a_mean": float(means[1]),
        "log_initial_mass_std": float(scales[0]),
        "log_a_std": float(scales[1]),
    }
    return (features - means) / scales, metadata


def _kernel_predict(
    train_features: np.ndarray,
    train_delta: np.ndarray,
    eval_features: np.ndarray,
    *,
    bandwidth: float,
) -> np.ndarray:
    diff = eval_features[:, None, :] - train_features[None, :, :]
    distance2 = np.sum(np.square(diff), axis=2)
    weights = np.exp(-0.5 * distance2 / np.square(bandwidth))
    weight_sum = np.sum(weights, axis=1)
    fallback = float(np.mean(train_delta))
    predictions = np.full(len(eval_features), fallback, dtype=float)
    valid = weight_sum > 1.0e-12
    predictions[valid] = (weights[valid] @ train_delta) / weight_sum[valid]
    return predictions


def _kernel_leave_one_out_predictions(
    features: np.ndarray,
    delta: np.ndarray,
    *,
    bandwidth: float,
) -> np.ndarray:
    diff = features[:, None, :] - features[None, :, :]
    distance2 = np.sum(np.square(diff), axis=2)
    weights = np.exp(-0.5 * distance2 / np.square(bandwidth))
    np.fill_diagonal(weights, 0.0)
    weight_sum = np.sum(weights, axis=1)
    fallback = float(np.mean(delta))
    predictions = np.full(len(delta), fallback, dtype=float)
    valid = weight_sum > 1.0e-12
    predictions[valid] = (weights[valid] @ delta) / weight_sum[valid]
    return predictions


def _fit_monotonic_loss_model(
    train_z_mass: np.ndarray,
    train_z_a: np.ndarray,
    train_loss: np.ndarray,
    *,
    start_params: np.ndarray | None = None,
) -> np.ndarray:
    def predict(params: np.ndarray, z_mass: np.ndarray, z_a: np.ndarray) -> np.ndarray:
        b0, b1, b2, s0, s1 = np.asarray(params, dtype=float)
        radial = b0 + b1 * z_a + b2 * np.square(z_a)
        mass_slope = special.softplus(s0 + s1 * z_a)
        return special.softplus(radial - mass_slope * z_mass)

    def residual(params: np.ndarray) -> np.ndarray:
        return predict(params, train_z_mass, train_z_a) - train_loss

    starts = [
        np.array([0.0, -0.8, 0.15, -1.0, 0.0]),
        np.array([0.2, -0.8, 0.2, -1.0, 0.5]),
        np.array([0.2, -0.8, 0.2, -1.0, -0.5]),
        np.array([0.0, -0.4, 0.0, -2.0, 0.0]),
    ]
    if start_params is not None:
        starts.insert(0, np.asarray(start_params, dtype=float))

    best_result = None
    best_value = np.inf
    for start in starts:
        result = optimize.least_squares(residual, start, max_nfev=5000)
        value = float(np.sum(np.square(result.fun)))
        if value < best_value:
            best_value = value
            best_result = result
    if best_result is None:
        raise RuntimeError("Monotonic loss model failed to start.")
    return np.asarray(best_result.x, dtype=float)


def _predict_monotonic_loss_model(params: np.ndarray, z_mass: np.ndarray, z_a: np.ndarray) -> np.ndarray:
    b0, b1, b2, s0, s1 = np.asarray(params, dtype=float)
    radial = b0 + b1 * z_a + b2 * np.square(z_a)
    mass_slope = special.softplus(s0 + s1 * z_a)
    return special.softplus(radial - mass_slope * z_mass)


def _monotonic_leave_one_out_predictions(
    z_mass: np.ndarray,
    z_a: np.ndarray,
    log_initial_mass: np.ndarray,
    loss: np.ndarray,
    *,
    start_params: np.ndarray,
) -> np.ndarray:
    predictions = np.empty_like(log_initial_mass)
    for index in range(len(loss)):
        train = np.ones(len(loss), dtype=bool)
        train[index] = False
        params = _fit_monotonic_loss_model(
            z_mass[train],
            z_a[train],
            loss[train],
            start_params=start_params,
        )
        predicted_loss = _predict_monotonic_loss_model(
            params,
            z_mass[index : index + 1],
            z_a[index : index + 1],
        )[0]
        predictions[index] = log_initial_mass[index] - predicted_loss
    return predictions


def _metrics(residual: np.ndarray) -> dict[str, float]:
    residual = np.asarray(residual, dtype=float)
    return {
        "bias_dex": float(np.mean(residual)),
        "rms_dex": float(np.sqrt(np.mean(np.square(residual)))),
        "median_abs_dex": float(np.median(np.abs(residual))),
        "p90_abs_dex": float(np.percentile(np.abs(residual), 90.0)),
    }


def _plot_comparison(
    catalog: pd.DataFrame,
    *,
    polynomial_prediction: np.ndarray,
    kernel_prediction: np.ndarray,
    output_path: Path,
) -> None:
    log_initial_mass = catalog["log_initial_mass_msun"].to_numpy(dtype=float)
    log_present_mass = np.log10(catalog["present_mass_msun"].to_numpy(dtype=float))
    log_a = np.log10(catalog["semi_major_axis_kpc"].to_numpy(dtype=float))

    fig, axes = plt.subplots(ncols=2, figsize=(7.2, 3.45), sharex=True, sharey=True, constrained_layout=True)
    for ax, prediction, title in [
        (axes[0], polynomial_prediction, "Current polynomial proxy"),
        (axes[1], kernel_prediction, "Kernel-smoothed residual proxy"),
    ]:
        scatter = ax.scatter(
            log_present_mass,
            prediction,
            c=log_a,
            s=18,
            cmap="viridis",
            alpha=0.75,
            linewidths=0.0,
        )
        limits = [
            min(float(np.min(log_present_mass)), float(np.min(prediction))) - 0.08,
            max(float(np.max(log_present_mass)), float(np.max(prediction))) + 0.08,
        ]
        ax.plot(limits, limits, color="black", linewidth=1.0, linestyle=":")
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel(r"Observed $\log_{10}(M_{\rm now}/\mathrm{M_\odot})$")
    axes[0].set_ylabel(r"Predicted $\log_{10}(M_{\rm now}/\mathrm{M_\odot})$")
    colorbar = fig.colorbar(scatter, ax=axes, pad=0.012, fraction=0.035)
    colorbar.set_label(r"$\log_{10}(a/\mathrm{kpc})$")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    catalog = _load_fit_catalog()
    log_initial_mass = catalog["log_initial_mass_msun"].to_numpy(dtype=float)
    log_present_mass = np.log10(catalog["present_mass_msun"].to_numpy(dtype=float))
    delta = log_present_mass - log_initial_mass
    features, feature_metadata = _standardized_features(catalog)
    z_mass = features[:, 0]
    z_a = features[:, 1]

    polynomial_prediction = _polynomial_predictions(catalog)
    polynomial_metrics = _metrics(log_present_mass - polynomial_prediction)
    polynomial_prediction_loo = _polynomial_leave_one_out_predictions(catalog)
    polynomial_metrics_loo = _metrics(log_present_mass - polynomial_prediction_loo)

    bandwidth_grid = np.round(np.linspace(0.15, 1.60, 30), 3)
    loo_rows: list[dict[str, float]] = []
    for bandwidth in bandwidth_grid:
        loo_delta_prediction = _kernel_leave_one_out_predictions(features, delta, bandwidth=float(bandwidth))
        loo_log_present_prediction = log_initial_mass + loo_delta_prediction
        metrics = _metrics(log_present_mass - loo_log_present_prediction)
        loo_rows.append({"bandwidth": float(bandwidth), **metrics})

    loo_table = pd.DataFrame(loo_rows).sort_values("rms_dex").reset_index(drop=True)
    best_bandwidth = float(loo_table.iloc[0]["bandwidth"])
    kernel_delta_prediction_in_sample = _kernel_predict(features, delta, features, bandwidth=best_bandwidth)
    kernel_prediction_in_sample = log_initial_mass + kernel_delta_prediction_in_sample
    kernel_metrics_in_sample = _metrics(log_present_mass - kernel_prediction_in_sample)

    kernel_delta_prediction_loo = _kernel_leave_one_out_predictions(features, delta, bandwidth=best_bandwidth)
    kernel_prediction_loo = log_initial_mass + kernel_delta_prediction_loo
    kernel_metrics_loo = _metrics(log_present_mass - kernel_prediction_loo)

    monotonic_params = _fit_monotonic_loss_model(z_mass, z_a, log_initial_mass - log_present_mass)
    monotonic_loss = _predict_monotonic_loss_model(monotonic_params, z_mass, z_a)
    monotonic_prediction_in_sample = log_initial_mass - monotonic_loss
    monotonic_metrics_in_sample = _metrics(log_present_mass - monotonic_prediction_in_sample)
    monotonic_prediction_loo = _monotonic_leave_one_out_predictions(
        z_mass,
        z_a,
        log_initial_mass,
        log_initial_mass - log_present_mass,
        start_params=monotonic_params,
    )
    monotonic_metrics_loo = _metrics(log_present_mass - monotonic_prediction_loo)

    outputs_tables = PROJECT_ROOT / "outputs" / "tables"
    outputs_figures = PROJECT_ROOT / "outputs" / "figures"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    outputs_figures.mkdir(parents=True, exist_ok=True)

    comparison = {
        "feature_metadata": feature_metadata,
        "current_polynomial_in_sample": polynomial_metrics,
        "current_polynomial_leave_one_out": polynomial_metrics_loo,
        "kernel_smoothed_residual_best_bandwidth": best_bandwidth,
        "kernel_smoothed_residual_in_sample": kernel_metrics_in_sample,
        "kernel_smoothed_residual_leave_one_out": kernel_metrics_loo,
        "monotonic_loss_model_parameters": monotonic_params.tolist(),
        "monotonic_loss_model_in_sample": monotonic_metrics_in_sample,
        "monotonic_loss_model_leave_one_out": monotonic_metrics_loo,
    }
    (outputs_tables / "present_mass_proxy_model_comparison.json").write_text(json.dumps(comparison, indent=2))
    loo_table.to_csv(outputs_tables / "present_mass_proxy_kernel_bandwidth_scan.csv", index=False)

    name_column = "cluster_label" if "cluster_label" in catalog.columns else "cluster_name"
    per_cluster = catalog.loc[
        :,
        [name_column, "log_initial_mass_msun", "semi_major_axis_kpc", "present_mass_msun"],
    ].copy()
    per_cluster = per_cluster.rename(columns={name_column: "cluster"})
    per_cluster["log10_present_mass_observed"] = log_present_mass
    per_cluster["log10_present_mass_polynomial"] = polynomial_prediction
    per_cluster["log10_present_mass_polynomial_loo"] = polynomial_prediction_loo
    per_cluster["log10_present_mass_kernel_in_sample"] = kernel_prediction_in_sample
    per_cluster["log10_present_mass_kernel_loo"] = kernel_prediction_loo
    per_cluster["log10_present_mass_monotonic"] = monotonic_prediction_in_sample
    per_cluster["log10_present_mass_monotonic_loo"] = monotonic_prediction_loo
    per_cluster["residual_polynomial_dex"] = log_present_mass - polynomial_prediction
    per_cluster["residual_polynomial_loo_dex"] = log_present_mass - polynomial_prediction_loo
    per_cluster["residual_kernel_in_sample_dex"] = log_present_mass - kernel_prediction_in_sample
    per_cluster["residual_kernel_loo_dex"] = log_present_mass - kernel_prediction_loo
    per_cluster["residual_monotonic_dex"] = log_present_mass - monotonic_prediction_in_sample
    per_cluster["residual_monotonic_loo_dex"] = log_present_mass - monotonic_prediction_loo
    per_cluster.to_csv(outputs_tables / "present_mass_proxy_per_cluster_predictions.csv", index=False)

    _plot_comparison(
        catalog,
        polynomial_prediction=polynomial_prediction,
        kernel_prediction=kernel_prediction_loo,
        output_path=outputs_figures / "present_mass_proxy_model_comparison.pdf",
    )

    print(json.dumps(comparison, indent=2))
    print(outputs_tables / "present_mass_proxy_kernel_bandwidth_scan.csv")
    print(outputs_figures / "present_mass_proxy_model_comparison.pdf")


if __name__ == "__main__":
    main()
