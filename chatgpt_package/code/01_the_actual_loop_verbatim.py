"""Verbatim source of every function in the alternating loop.
Extracted from the project so the logic can be read in one place.
Order: the population fit, then the completeness fit, then the loop that alternates them.
"""

# ============================================================================
# joint_model.unpack_model  -- defines N0 = N_obs / selection_fraction
# ============================================================================
def unpack_model(params: np.ndarray, context: JointLikelihoodContext, spec: JointModelSpec) -> dict[str, object]:
    imf_param_count = imf_parameter_count(spec.imf_family)
    imf_params = np.asarray(params[:imf_param_count], dtype=float)
    radial_params = np.asarray(params[imf_param_count:], dtype=float)

    imf_density_grid, imf_density_data, imf_parameters = evaluate_imf_family(
        spec.imf_family,
        imf_params,
        context.log_mass_grid,
        context.log_mass_data,
    )
    radial_density_grid, radial_density_data, radial_parameters = evaluate_radial_model(
        spec.radial_model,
        radial_params,
        context,
    )
    raw_survival_fraction = integrate_survival_fraction(
        imf_density_grid,
        radial_density_grid,
        context.log_mass_grid,
        context.log_a_grid,
        context.survival_probability_grid,
    )
    selection_fraction = integrate_survival_fraction(
        imf_density_grid,
        radial_density_grid,
        context.log_mass_grid,
        context.log_a_grid,
        context.selection_probability_grid,
    )
    total_initial_count = len(context.log_mass_data) / selection_fraction
    return {
        "spec": spec,
        "imf_density_grid": imf_density_grid,
        "imf_density_data": imf_density_data,
        "radial_density_grid": radial_density_grid,
        "radial_density_data": radial_density_data,
        # `selection_fraction` = survival x detectability. There used to be a
        # `survival_fraction` alias here pointing at the same value, which caused
        # detectability-corrected selection fractions to be published under a
        # `survival_fraction` heading. Use the explicit names.
        "selection_fraction": selection_fraction,
        "raw_survival_fraction": raw_survival_fraction,
        "total_initial_count": total_initial_count,
        "n0_log_mass_min": float(context.log_mass_grid[0]),
        "imf_parameters": imf_parameters,
        "radial_parameters": radial_parameters,
    }


# ============================================================================
# joint_model.integrate_survival_fraction
# ============================================================================
def integrate_survival_fraction(
    imf_density_grid: np.ndarray,
    radial_density_grid: np.ndarray,
    log_mass_grid: np.ndarray,
    log_a_grid: np.ndarray,
    survival_probability_grid: np.ndarray,
) -> float:
    integrand = (
        imf_density_grid[:, None]
        * radial_density_grid[None, :]
        * survival_probability_grid
    )
    return float(np.trapezoid(np.trapezoid(integrand, log_a_grid, axis=1), log_mass_grid))


# ============================================================================
# joint_model.negative_profile_log_likelihood
# ============================================================================
def negative_profile_log_likelihood(
    params: np.ndarray,
    context: JointLikelihoodContext,
    spec: JointModelSpec,
) -> float:
    model = unpack_model(params, context=context, spec=spec)
    if np.any(model["imf_density_data"] <= 0.0) or np.any(model["radial_density_data"] <= 0.0):
        return 1.0e30
    if model["selection_fraction"] <= 0.0:
        return 1.0e30
    profile_log_like = (
        np.sum(np.log(model["imf_density_data"]))
        + np.sum(np.log(model["radial_density_data"]))
        - len(context.log_mass_data) * np.log(model["selection_fraction"])
    )
    return float(-profile_log_like + imf_smoothness_penalty(params, spec=spec))


# ============================================================================
# joint_model.full_log_likelihood_from_model
# ============================================================================
def full_log_likelihood_from_model(model: dict[str, object], context: JointLikelihoodContext) -> float:
    selection_data = np.clip(
        context.selection_interpolator(np.column_stack([context.log_mass_data, context.log_a_data])),
        1.0e-12,
        1.0,
    )
    total_initial_count = float(model["total_initial_count"])
    return float(
        len(context.log_mass_data) * np.log(total_initial_count)
        - total_initial_count * model["selection_fraction"]
        + np.sum(np.log(np.clip(model["imf_density_data"], 1.0e-12, None)))
        + np.sum(np.log(np.clip(model["radial_density_data"], 1.0e-12, None)))
        + np.sum(np.log(selection_data))
    )


# ============================================================================
# detectability_model.evaluate_completeness_bin_grid
# ============================================================================
def evaluate_completeness_bin_grid(
    raw_params: np.ndarray,
    observable_context: ObservablePredictionContext,
) -> np.ndarray:
    """Completeness on the (present mass, distance, |b|) bin grid.

    ``raw_params[0]`` is the intercept. When the caller is anchoring the mean
    completeness it will have already solved for that intercept with
    ``solve_intercept_for_mean_completeness`` and written it into slot 0, so this
    function stays a plain evaluator either way.
    """
    intercept = float(raw_params[0])
    shape_logits = completeness_shape_logits(raw_params, observable_context)
    return np.clip(special.expit(intercept + shape_logits), 1.0e-6, 1.0)


# ============================================================================
# detectability_model.completeness_shape_logits
# ============================================================================
def completeness_shape_logits(
    raw_params: np.ndarray,
    observable_context: ObservablePredictionContext,
) -> np.ndarray:
    """Intercept-free part of the completeness logit.

    Slopes are exponentiated, so completeness is forced to increase with present
    mass, decrease with distance and increase with |b|. That monotonicity is an
    assumption the data cannot overturn; it is deliberate, and it is what makes
    the *shape* identifiable once the amplitude is anchored.
    """
    mass_slope = float(np.exp(raw_params[1]))
    distance_slope = float(np.exp(raw_params[2]))
    latitude_slope = float(np.exp(raw_params[3]))

    z_mass = (
        observable_context.log_present_mass_centers[:, None, None] - observable_context.log_present_mass_feature_mean
    ) / observable_context.log_present_mass_feature_std
    z_distance = (
        observable_context.log_distance_centers[None, :, None] - observable_context.log_distance_feature_mean
    ) / observable_context.log_distance_feature_std
    z_latitude = (
        observable_context.abs_latitude_centers_deg[None, None, :] - observable_context.abs_latitude_feature_mean
    ) / observable_context.abs_latitude_feature_std
    return mass_slope * z_mass - distance_slope * z_distance + latitude_slope * z_latitude


# ============================================================================
# detectability_model.compute_effective_completeness_grid
# ============================================================================
def compute_effective_completeness_grid(
    observable_context: ObservablePredictionContext,
    completeness_bin_grid: np.ndarray,
) -> np.ndarray:
    sky_averaged_completeness = np.einsum(
        "adb,kdb->ak",
        observable_context.sky_bin_probabilities_by_a,
        completeness_bin_grid,
    )
    return np.clip(
        np.einsum(
            "mak,ak->ma",
            observable_context.mass_bin_probabilities_grid,
            sky_averaged_completeness,
        ),
        1.0e-4,
        1.0,
    )


# ============================================================================
# detectability_model.predict_complete_observable_histogram
# ============================================================================
def predict_complete_observable_histogram(
    complete_survivor_intensity_grid: np.ndarray,
    base_context: JointLikelihoodContext,
    observable_context: ObservablePredictionContext,
) -> np.ndarray:
    log_mass_edges = centers_to_edges_local(base_context.log_mass_grid)
    log_a_edges = centers_to_edges_local(base_context.log_a_grid)
    cell_counts = complete_survivor_intensity_grid * np.diff(log_mass_edges)[:, None] * np.diff(log_a_edges)[None, :]
    mass_counts_by_a = np.einsum(
        "ma,mak->ak",
        cell_counts,
        observable_context.mass_bin_probabilities_grid,
    )
    return np.einsum(
        "ak,adb->kdb",
        mass_counts_by_a,
        observable_context.sky_bin_probabilities_by_a,
    )


# ============================================================================
# detectability_model.compute_complete_survivor_intensity_grid
# ============================================================================
def compute_complete_survivor_intensity_grid(
    model: dict[str, object],
    base_context: JointLikelihoodContext,
) -> np.ndarray:
    return compute_observed_intensity_grid(
        model["imf_density_grid"],
        model["radial_density_grid"],
        base_context.survival_probability_grid,
        model["total_initial_count"],
    )


# ============================================================================
# detectability_model.negative_completeness_log_likelihood
# ============================================================================
def negative_completeness_log_likelihood(
    params: np.ndarray,
    observable_context: ObservablePredictionContext,
    predicted_complete_counts: np.ndarray,
    assumed_mean_completeness: float | None = DEFAULT_ASSUMED_MEAN_COMPLETENESS,
) -> float:
    params = anchored_completeness_params(
        params, observable_context, predicted_complete_counts, assumed_mean_completeness
    )
    completeness_bin_grid = evaluate_completeness_bin_grid(params, observable_context)
    mu = np.clip(predicted_complete_counts * completeness_bin_grid, 1.0e-12, None)
    observed = observable_context.observed_counts
    return float(-(np.sum(observed * np.log(mu) - mu)))


# ============================================================================
# detectability_model.fit_logistic_completeness_model
# ============================================================================
def fit_logistic_completeness_model(
    observable_context: ObservablePredictionContext,
    predicted_complete_counts: np.ndarray,
    start_params: np.ndarray | None,
    assumed_mean_completeness: float | None = DEFAULT_ASSUMED_MEAN_COMPLETENESS,
) -> dict[str, object]:
    total_predicted = float(np.sum(predicted_complete_counts))
    total_observed = float(np.sum(observable_context.observed_counts))
    ratio = np.clip(total_observed / max(total_predicted, 1.0e-12), 1.0e-3, 0.999)
    default_start = np.array(
        [
            float(np.log(ratio / (1.0 - ratio))),
            np.log(0.25),
            np.log(0.25),
            np.log(0.25),
        ]
    )
    starts = [default_start]
    if start_params is not None:
        starts.insert(0, np.asarray(start_params, dtype=float))
    starts.append(np.array([default_start[0], np.log(0.6), np.log(0.6), np.log(0.6)]))
    bounds = [(-8.0, 8.0), (-8.0, 4.0), (-8.0, 4.0), (-8.0, 4.0)]

    # When the mean completeness is anchored the intercept is a solved function of
    # the slopes, so leaving it free would add a redundant direction to the
    # optimiser. Freeze the slot; `anchored_completeness_params` fills it in.
    if assumed_mean_completeness is not None:
        bounds = [(0.0, 0.0), *bounds[1:]]
        starts = [np.array([0.0, *np.asarray(start, dtype=float)[1:]]) for start in starts]

    best_result = None
    best_value = np.inf
    for start in starts:
        result = optimize.minimize(
            lambda params: negative_completeness_log_likelihood(
                params=params,
                observable_context=observable_context,
                predicted_complete_counts=predicted_complete_counts,
                assumed_mean_completeness=assumed_mean_completeness,
            ),
            x0=np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if result.fun < best_value:
            best_value = float(result.fun)
            best_result = result

    if best_result is None:
        raise RuntimeError("Completeness optimization failed to start.")
    resolved_params = anchored_completeness_params(
        best_result.x, observable_context, predicted_complete_counts, assumed_mean_completeness
    )
    completeness_bin_grid = evaluate_completeness_bin_grid(resolved_params, observable_context)
    return {
        "raw_parameters": resolved_params,
        "completeness_bin_grid": completeness_bin_grid,
        "negative_log_likelihood": float(best_result.fun),
        "success": bool(best_result.success),
        "message": str(best_result.message),
    }


# ============================================================================
# detectability_model.fit_single_component_detectability_em  -- THE LOOP
# ============================================================================
def fit_single_component_detectability_em(
    catalog: pd.DataFrame,
    project_root: Path,
    spec: JointModelSpec | None = None,
    max_iterations: int = 200,
    tolerance: float = 1.0e-4,
    raise_on_non_convergence: bool = True,
    log_likelihood_monotonicity_atol: float = 1.0e-6,
    assumed_mean_completeness: float | None = DEFAULT_ASSUMED_MEAN_COMPLETENESS,
    relaxation: float = 0.7,
    n_present_mass_bins: int = 6,
    n_distance_bins: int = 6,
    n_latitude_bins: int = 6,
    n_geometry_samples: int = 5000,
    sun_galactocentric_radius_kpc: float = 8.2,
    survival_grid_override: dict[str, object] | None = None,
    n_iterations: int | None = None,
) -> dict[str, object]:
    """Iterate the joint population fit against the detectability model.

    This used to run a fixed `n_iterations=200` with no convergence test, which is
    how an unconverged (indeed diverging) iterate ended up being published. It now
    iterates to a fixed point and raises if it cannot reach one.

    `assumed_mean_completeness`, if set, declares the mean completeness instead of
    letting the logistic form imply it; see the note at the top of this module for
    why that identification is weak.
    """
    if n_iterations is not None:
        raise TypeError(
            "`n_iterations` is gone: running the detectability iteration for a "
            "fixed number of steps and reporting whatever came out is exactly the "
            "bug this replaced. Use `max_iterations` (a safety stop) together with "
            "`tolerance` (the actual convergence criterion)."
        )
    if spec is None:
        spec = JointModelSpec(imf_family="schechter", radial_model="logpoly3")

    working = catalog.copy()
    required_columns = {
        "log_initial_mass_msun",
        "semi_major_axis_kpc",
        "log_survival_mass_cut_msun",
        "present_mass_msun",
        "r_sun_kpc",
        "galactic_b_deg",
    }
    missing = required_columns.difference(working.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Catalog is missing required columns for detectability fitting: {missing_list}")

    if survival_grid_override is None:
        selection_offset_dex = calibrate_fixed_selection_offset_dex(working)
        survival_grid = build_fixed_survival_grid(
            working,
            selection_offset_dex=selection_offset_dex,
        )
    else:
        selection_offset_dex = float(survival_grid_override.get("selection_offset_dex", np.nan))
        survival_grid = survival_grid_override
    base_context = JointLikelihoodContext.from_catalog_and_survival_grid(working, survival_grid)
    observable_context = build_observable_prediction_context(
        catalog=working,
        base_context=base_context,
        n_present_mass_bins=n_present_mass_bins,
        n_distance_bins=n_distance_bins,
        n_latitude_bins=n_latitude_bins,
        n_geometry_samples=n_geometry_samples,
        sun_galactocentric_radius_kpc=sun_galactocentric_radius_kpc,
    )

    baseline_context = base_context.with_selection_probability_grid(base_context.survival_probability_grid)
    baseline_payload = fit_single_joint_model(context=baseline_context, spec=spec)
    current_raw_params = fit_logistic_completeness_model(
        observable_context=observable_context,
        predicted_complete_counts=predict_complete_observable_histogram(
            complete_survivor_intensity_grid=compute_complete_survivor_intensity_grid(
                baseline_payload["model"],
                base_context=base_context,
            ),
            base_context=base_context,
            observable_context=observable_context,
        ),
        start_params=None,
        assumed_mean_completeness=assumed_mean_completeness,
    )["raw_parameters"]

    iteration_rows: list[DetectabilityIterationSummary] = []
    current_context = baseline_context
    current_payload = baseline_payload
    current_effective_completeness_grid = np.ones_like(base_context.survival_probability_grid)

    parameter_residual = np.inf
    log_likelihood_residual = np.inf
    previous_log_likelihood: float | None = None
    log_likelihood_monotone = True
    converged = False
    stalled = False
    stall_detector = StallDetector()
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        completeness_bin_grid = evaluate_completeness_bin_grid(current_raw_params, observable_context)
        current_effective_completeness_grid = compute_effective_completeness_grid(
            observable_context=observable_context,
            completeness_bin_grid=completeness_bin_grid,
        )
        current_context = base_context.with_selection_probability_grid(
            np.clip(
                base_context.survival_probability_grid * current_effective_completeness_grid,
                1.0e-12,
                1.0,
            )
        )
        current_payload = fit_single_joint_model(context=current_context, spec=spec)
        complete_survivor_intensity_grid = compute_complete_survivor_intensity_grid(
            current_payload["model"],
            base_context=base_context,
        )
        predicted_complete_counts = predict_complete_observable_histogram(
            complete_survivor_intensity_grid=complete_survivor_intensity_grid,
            base_context=base_context,
            observable_context=observable_context,
        )
        completeness_fit = fit_logistic_completeness_model(
            observable_context=observable_context,
            predicted_complete_counts=predicted_complete_counts,
            start_params=current_raw_params,
            assumed_mean_completeness=assumed_mean_completeness,
        )
        target_raw_params = completeness_fit["raw_parameters"]

        previous_raw_params = np.asarray(current_raw_params, dtype=float)
        current_raw_params = (1.0 - relaxation) * previous_raw_params + relaxation * target_raw_params

        # Fixed-point residual: how far the *undamped* update still wants to move.
        # Damping shrinks the step, so measuring the damped step would make any
        # slow-moving runaway look converged. Measure the target instead.
        parameter_residual = float(np.max(np.abs(target_raw_params - previous_raw_params)))

        current_log_likelihood = float(current_payload["summary"].log_likelihood)
        if previous_log_likelihood is None:
            log_likelihood_residual = np.inf
        else:
            log_likelihood_residual = float(abs(current_log_likelihood - previous_log_likelihood))
            if current_log_likelihood < previous_log_likelihood - log_likelihood_monotonicity_atol:
                log_likelihood_monotone = False
        previous_log_likelihood = current_log_likelihood

        updated_completeness_bin_grid = evaluate_completeness_bin_grid(current_raw_params, observable_context)
        predicted_observed_counts = predicted_complete_counts * updated_completeness_bin_grid
        iteration_rows.append(
            DetectabilityIterationSummary(
                iteration=iteration,
                log_likelihood=current_log_likelihood,
                total_initial_count=float(current_payload["model"]["total_initial_count"]),
                selection_fraction=float(current_payload["model"]["selection_fraction"]),
                raw_survival_fraction=float(current_payload["model"]["raw_survival_fraction"]),
                completeness_mean=float(
                    np.sum(predicted_complete_counts * updated_completeness_bin_grid)
                    / max(np.sum(predicted_complete_counts), 1.0e-12)
                ),
                completeness_intercept=float(current_raw_params[0]),
                completeness_mass_slope=float(np.exp(current_raw_params[1])),
                completeness_distance_slope=float(np.exp(current_raw_params[2])),
                completeness_latitude_slope=float(np.exp(current_raw_params[3])),
                predicted_complete_survivor_count=float(np.sum(predicted_complete_counts)),
                predicted_observed_count=float(np.sum(predicted_observed_counts)),
                parameter_residual=parameter_residual,
                log_likelihood_residual=log_likelihood_residual,
            )
        )

        if parameter_residual < tolerance and log_likelihood_residual < tolerance:
            converged = True
            break

        # Bail out once the residual stops improving: a stalled iteration cannot reach
        # the tolerance however long it runs, and grinding to `max_iterations` only
        # burns time to reach the same non-convergent verdict.
        if stall_detector.update(parameter_residual):
            stalled = True
            break

    if not converged:
        how = (
            "stalled after {n} steps (residual stopped improving; it will not reach the "
            "tolerance however long it runs)".format(n=iteration)
            if stalled
            else f"did NOT converge in {iteration} steps"
        )
        message = (
            f"Detectability iteration {how}: "
            f"fixed-point residual {parameter_residual:.3e} > tol {tolerance:.1e}. "
            "The reported N0 is whatever the last iterate happened to be, not a "
            "solution. Do not publish it."
        )
        if not log_likelihood_monotone:
            message += (
                " The log-likelihood also decreased along the way, which indicates "
                "the iteration is sliding along an unidentified direction rather "
                "than maximising anything."
            )
        if raise_on_non_convergence:
            raise DetectabilityConvergenceError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    else:
        message = f"Converged in {iteration} iterations (residual {parameter_residual:.3e})."

    convergence = DetectabilityConvergence(
        converged=converged,
        n_iterations_run=iteration,
        parameter_residual=float(parameter_residual),
        log_likelihood_residual=float(log_likelihood_residual),
        log_likelihood_monotone=bool(log_likelihood_monotone),
        tolerance=float(tolerance),
        message=message,
    )

    final_completeness_bin_grid = evaluate_completeness_bin_grid(current_raw_params, observable_context)
    final_effective_completeness_grid = compute_effective_completeness_grid(
        observable_context=observable_context,
        completeness_bin_grid=final_completeness_bin_grid,
    )
    final_context = base_context.with_selection_probability_grid(
        np.clip(
            base_context.survival_probability_grid * final_effective_completeness_grid,
            1.0e-12,
            1.0,
        )
    )
    final_payload = fit_single_joint_model(context=final_context, spec=spec)
    final_complete_survivor_intensity_grid = compute_complete_survivor_intensity_grid(
        final_payload["model"],
        base_context=base_context,
    )
    final_predicted_complete_counts = predict_complete_observable_histogram(
        complete_survivor_intensity_grid=final_complete_survivor_intensity_grid,
        base_context=base_context,
        observable_context=observable_context,
    )
    final_predicted_observed_counts = final_predicted_complete_counts * final_completeness_bin_grid

    iteration_history_table = pd.DataFrame([asdict(row) for row in iteration_rows])
    completeness_grid_table = build_completeness_grid_table(observable_context, final_completeness_bin_grid)
    observable_histogram_table = build_observable_histogram_table(
        observable_context=observable_context,
        predicted_complete_counts=final_predicted_complete_counts,
        predicted_observed_counts=final_predicted_observed_counts,
        completeness_bin_grid=final_completeness_bin_grid,
    )
    catalog_completeness_table = build_catalog_completeness_table(
        catalog=working,
        context=final_context,
        observable_context=observable_context,
        completeness_raw_params=current_raw_params,
    )

    outputs_tables = project_root / "outputs" / "tables"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    iteration_history_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_em_iteration_history.csv",
        index=False,
    )
    completeness_grid_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_em_completeness_grid.csv",
        index=False,
    )
    observable_histogram_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_em_observable_histogram.csv",
        index=False,
    )
    catalog_completeness_table.to_csv(
        outputs_tables / "joint_fixed_survival_detectability_em_catalog_completeness.csv",
        index=False,
    )

    summary_payload = {
        "spec": asdict(spec),
        "selection_offset_dex": selection_offset_dex,
        "sun_galactocentric_radius_kpc": sun_galactocentric_radius_kpc,
        "max_iterations": max_iterations,
        "tolerance": tolerance,
        "assumed_mean_completeness": assumed_mean_completeness,
        "convergence": asdict(convergence),
        "relaxation": relaxation,
        "baseline_total_initial_count": float(baseline_payload["model"]["total_initial_count"]),
        "baseline_raw_survival_fraction": float(baseline_payload["model"]["raw_survival_fraction"]),
        "final_total_initial_count": float(final_payload["model"]["total_initial_count"]),
        "final_selection_fraction": float(final_payload["model"]["selection_fraction"]),
        "final_raw_survival_fraction": float(final_payload["model"]["raw_survival_fraction"]),
        "final_mean_detectability": float(
            final_payload["model"]["selection_fraction"] / max(final_payload["model"]["raw_survival_fraction"], 1.0e-12)
        ),
        "total_initial_count_ratio_vs_baseline": float(
            final_payload["model"]["total_initial_count"] / max(baseline_payload["model"]["total_initial_count"], 1.0e-12)
        ),
        "present_mass_proxy": {
            "model_kind": getattr(observable_context.present_mass_proxy, "model_kind", "polynomial_log_mass_ratio"),
            "coefficients": observable_context.present_mass_proxy.coefficients.tolist(),
            "log_mass_mean": observable_context.present_mass_proxy.log_mass_mean,
            "log_a_mean": observable_context.present_mass_proxy.log_a_mean,
            "log_mass_std": getattr(observable_context.present_mass_proxy, "log_mass_std", 1.0),
            "log_a_std": getattr(observable_context.present_mass_proxy, "log_a_std", 1.0),
            "residual_sigma_dex": observable_context.present_mass_proxy.residual_sigma_dex,
        },
        "final_completeness_parameters": {
            "intercept": float(current_raw_params[0]),
            "mass_slope": float(np.exp(current_raw_params[1])),
            "distance_slope": float(np.exp(current_raw_params[2])),
            "latitude_slope": float(np.exp(current_raw_params[3])),
        },
        # NOTE: this payload deliberately has no `best_joint_model` /
        # `best_model_detectability_summary`. Those keys belong to the *comparison*
        # writers, which fit several specs and pick a winner. This function fits ONE
        # spec, so there is no "best" to report -- the fitted model is `final_model`.
        # Duplicating the keys in here to satisfy a miswritten reader would be a lie
        # about what this payload is; the reader was fixed instead.
        "baseline_model": asdict(baseline_payload["summary"]),
        "final_model": {
            **asdict(final_payload["summary"]),
            "raw_survival_fraction": float(final_payload["model"]["raw_survival_fraction"]),
            "selection_fraction": float(final_payload["model"]["selection_fraction"]),
        },
        "iteration_history": iteration_history_table.to_dict(orient="records"),
    }
    (outputs_tables / "joint_fixed_survival_detectability_em_summary.json").write_text(
        json.dumps(summary_payload, indent=2)
    )

    return {
        "spec": spec,
        "convergence": convergence,
        "selection_offset_dex": selection_offset_dex,
        "base_context": base_context,
        "final_context": final_context,
        "observable_context": observable_context,
        "baseline_payload": baseline_payload,
        "final_payload": final_payload,
        "iteration_history_table": iteration_history_table,
        "completeness_grid_table": completeness_grid_table,
        "observable_histogram_table": observable_histogram_table,
        "catalog_completeness_table": catalog_completeness_table,
        "final_completeness_raw_parameters": current_raw_params,
        "final_completeness_bin_grid": final_completeness_bin_grid,
        "final_effective_completeness_grid": final_effective_completeness_grid,
        "final_predicted_complete_counts": final_predicted_complete_counts,
        "final_predicted_observed_counts": final_predicted_observed_counts,
        "final_complete_survivor_intensity_grid": final_complete_survivor_intensity_grid,
        "summary_payload": summary_payload,
    }

