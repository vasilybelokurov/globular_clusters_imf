from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def main() -> None:
    catalog = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog_with_origin_and_chemistry.csv")
    subset = catalog.loc[catalog["has_mgfe_and_alfe"]].copy()
    if subset.empty:
        raise RuntimeError("No clusters with both Mg and Al abundances were found.")

    diagnostics = compute_mg_al_diagnostics(subset)
    summary_table = pd.DataFrame([diagnostics["summary"]])
    score_table = diagnostics["score_table"]

    outputs_tables = PROJECT_ROOT / "outputs" / "tables"
    outputs_figures = PROJECT_ROOT / "outputs" / "figures"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    outputs_figures.mkdir(parents=True, exist_ok=True)

    summary_table.to_csv(outputs_tables / "mg_al_gc_chemistry_diagnostic_summary.csv", index=False)
    (outputs_tables / "mg_al_gc_chemistry_diagnostic_summary.json").write_text(
        json.dumps(diagnostics["summary"], indent=2)
    )
    score_table.to_csv(outputs_tables / "mg_al_gc_chemistry_diagnostic_scores.csv", index=False)

    build_diagnostic_figure(
        subset=subset,
        diagnostics=diagnostics,
        output_pdf=outputs_figures / "mg_al_gc_chemistry_diagnostic.pdf",
        output_png=outputs_figures / "mg_al_gc_chemistry_diagnostic.png",
    )


def compute_mg_al_diagnostics(subset: pd.DataFrame) -> dict[str, object]:
    x = subset["alfe_combined"].to_numpy(dtype=float)
    y = subset["mgfe_combined"].to_numpy(dtype=float)
    labels = subset["origin_flag"].astype(int).to_numpy()

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    covariance = np.cov(np.vstack([x - x_mean, y - y_mean]))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal_vector = eigenvectors[:, np.argmax(eigenvalues)]
    if principal_vector[0] < 0.0:
        principal_vector = -principal_vector
    slope = float(principal_vector[1] / principal_vector[0])
    intercept = y_mean - slope * x_mean
    orth_scatter_std, orth_scatter_mad = orthogonal_scatter(x, y, x_mean, y_mean, principal_vector)
    variance_fraction_pc1 = float(np.max(eigenvalues) / np.sum(eigenvalues))

    z = y - slope * x

    mg_auc_raw = binary_auc(y, labels)
    z_auc_raw = binary_auc(z, labels)
    mg_loo_scores = gaussian_1d_leave_one_out_scores(y, labels)
    z_loo_scores = gaussian_1d_leave_one_out_scores(z, labels)
    lda_loo_scores = fisher_lda_leave_one_out_scores(np.column_stack([x, y]), labels)

    diagnostics = {
        "summary": {
            "n_clusters_with_mg_and_al": int(len(subset)),
            "n_in_situ": int(np.sum(labels == 1)),
            "n_accreted": int(np.sum(labels == 0)),
            "common_sequence_slope_dmg_dal": slope,
            "common_sequence_intercept_mg_at_al0": intercept,
            "orthogonal_scatter_std_dex": orth_scatter_std,
            "orthogonal_scatter_mad_dex": orth_scatter_mad,
            "variance_fraction_pc1": variance_fraction_pc1,
            "auc_raw_mg_only": mg_auc_raw,
            "auc_raw_z_residual": z_auc_raw,
            "auc_loo_mg_only": binary_auc(mg_loo_scores, labels),
            "auc_loo_z_residual": binary_auc(z_loo_scores, labels),
            "auc_loo_2d_fisher_lda": binary_auc(lda_loo_scores, labels),
        },
        "score_table": subset.assign(
            chemistry_score_mg_only=y,
            chemistry_score_z_residual=z,
            chemistry_score_loo_mg_only=mg_loo_scores,
            chemistry_score_loo_z_residual=z_loo_scores,
            chemistry_score_loo_2d_lda=lda_loo_scores,
        ),
        "roc_curves": {
            "mg_only": roc_curve_from_scores(mg_loo_scores, labels),
            "z_residual": roc_curve_from_scores(z_loo_scores, labels),
            "lda_2d": roc_curve_from_scores(lda_loo_scores, labels),
        },
        "fit_line": {
            "x_mean": x_mean,
            "y_mean": y_mean,
            "slope": slope,
            "intercept": intercept,
        },
    }
    return diagnostics


def orthogonal_scatter(
    x: np.ndarray,
    y: np.ndarray,
    x_mean: float,
    y_mean: float,
    principal_vector: np.ndarray,
) -> tuple[float, float]:
    signed_distances = (principal_vector[0] * (y - y_mean) - principal_vector[1] * (x - x_mean)) / np.hypot(
        principal_vector[0], principal_vector[1]
    )
    std = float(np.std(signed_distances, ddof=1))
    mad = float(1.4826 * np.median(np.abs(signed_distances - np.median(signed_distances))))
    return std, mad


def gaussian_1d_leave_one_out_scores(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    scores = np.zeros(len(values), dtype=float)
    for index in range(len(values)):
        mask = np.ones(len(values), dtype=bool)
        mask[index] = False
        train_values = values[mask]
        train_labels = labels[mask]
        mu_in = float(np.mean(train_values[train_labels == 1]))
        mu_acc = float(np.mean(train_values[train_labels == 0]))
        var_in = float(np.var(train_values[train_labels == 1], ddof=1))
        var_acc = float(np.var(train_values[train_labels == 0], ddof=1))
        var_in = max(var_in, 1.0e-6)
        var_acc = max(var_acc, 1.0e-6)
        value = float(values[index])
        scores[index] = (
            -0.5 * np.log(var_in)
            - 0.5 * (value - mu_in) ** 2 / var_in
            + 0.5 * np.log(var_acc)
            + 0.5 * (value - mu_acc) ** 2 / var_acc
        )
    return scores


def fisher_lda_leave_one_out_scores(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    scores = np.zeros(len(features), dtype=float)
    for index in range(len(features)):
        mask = np.ones(len(features), dtype=bool)
        mask[index] = False
        train_features = features[mask]
        train_labels = labels[mask]
        mu_in = train_features[train_labels == 1].mean(axis=0)
        mu_acc = train_features[train_labels == 0].mean(axis=0)
        cov_in = np.cov(train_features[train_labels == 1].T, ddof=1)
        cov_acc = np.cov(train_features[train_labels == 0].T, ddof=1)
        within = ((train_labels == 1).sum() - 1) * cov_in + ((train_labels == 0).sum() - 1) * cov_acc
        weights = np.linalg.solve(within + 1.0e-6 * np.eye(features.shape[1]), mu_in - mu_acc)
        scores[index] = float(features[index] @ weights)
    return scores


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive = labels == 1
    n_positive = int(np.sum(positive))
    n_negative = int(np.sum(~positive))
    ranks = stats.rankdata(scores, method="average")
    return float((np.sum(ranks[positive]) - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative))


def roc_curve_from_scores(scores: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1], -np.inf]
    tpr: list[float] = []
    fpr: list[float] = []
    positive = labels == 1
    negative = ~positive
    n_positive = max(int(np.sum(positive)), 1)
    n_negative = max(int(np.sum(negative)), 1)
    for threshold in thresholds:
        predicted = scores >= threshold
        tpr.append(float(np.sum(predicted & positive) / n_positive))
        fpr.append(float(np.sum(predicted & negative) / n_negative))
    return {
        "fpr": np.asarray(fpr, dtype=float),
        "tpr": np.asarray(tpr, dtype=float),
    }


def build_diagnostic_figure(
    subset: pd.DataFrame,
    diagnostics: dict[str, object],
    output_pdf: Path,
    output_png: Path,
) -> None:
    summary = diagnostics["summary"]
    score_table = diagnostics["score_table"]
    roc_curves = diagnostics["roc_curves"]
    fit_line = diagnostics["fit_line"]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    ax_scatter, ax_roc = axes

    marker_map = {
        1: ("o", "#d95f02", "in-situ"),
        0: ("s", "#1f77b4", "accreted"),
    }
    for origin_flag, (marker, color, label) in marker_map.items():
        group = subset.loc[subset["origin_flag"] == origin_flag]
        ax_scatter.scatter(
            group["alfe_combined"],
            group["mgfe_combined"],
            s=44,
            marker=marker,
            color=color,
            edgecolor="k",
            linewidth=0.35,
            alpha=0.9,
            label=label,
        )

    x_grid = np.linspace(float(subset["alfe_combined"].min()) - 0.05, float(subset["alfe_combined"].max()) + 0.05, 200)
    y_grid = fit_line["intercept"] + fit_line["slope"] * x_grid
    ax_scatter.plot(x_grid, y_grid, color="0.25", lw=1.8, ls="--")
    ax_scatter.set_xlabel(r"[Al/Fe]")
    ax_scatter.set_ylabel(r"[Mg/Fe]")
    ax_scatter.legend(frameon=False, loc="lower left", fontsize=9)
    ax_scatter.text(
        0.04,
        0.97,
        (
            rf"$s = {summary['common_sequence_slope_dmg_dal']:.3f}$" "\n"
            rf"$\sigma_\perp = {summary['orthogonal_scatter_std_dex']:.3f}$ dex" "\n"
            rf"$f_{{\rm PC1}} = {summary['variance_fraction_pc1']:.2f}$"
        ),
        transform=ax_scatter.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.85"},
    )

    curve_styles = [
        ("mg_only", "#1b9e77", rf"[Mg/Fe] only, AUC$_{{\rm LOO}}={summary['auc_loo_mg_only']:.3f}$"),
        ("z_residual", "#7570b3", rf"$z = {{\rm Mg}} - s\,{{\rm Al}}$, AUC$_{{\rm LOO}}={summary['auc_loo_z_residual']:.3f}$"),
        ("lda_2d", "#d95f02", rf"2D LDA(Mg,Al), AUC$_{{\rm LOO}}={summary['auc_loo_2d_fisher_lda']:.3f}$"),
    ]
    for key, color, label in curve_styles:
        ax_roc.plot(roc_curves[key]["fpr"], roc_curves[key]["tpr"], color=color, lw=2.0, label=label)
    ax_roc.plot([0, 1], [0, 1], color="0.75", lw=1.0, ls=":")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_xlim(0.0, 1.0)
    ax_roc.set_ylim(0.0, 1.0)
    ax_roc.legend(frameon=False, loc="lower right", fontsize=8.8)

    for axis, label in zip(axes, ("(a)", "(b)"), strict=True):
        axis.text(0.03, 0.97, label, transform=axis.transAxes, ha="left", va="top", fontsize=11)

    fig.tight_layout()
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
