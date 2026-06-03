# GG23 Schechter refit run plan

Date: 2026-06-03

Goal: refit the single-component Schechter IMF model using the five GG23
disruption prescriptions, with self-consistent eta-dependent initial masses and
survivability surfaces.

## Model grid

GG23 disruption variants:

- `gg23_no_bh`
- `gg23_bh`
- `gg23_bh_feh_gradient`
- `gg23_bh_past_tidal`
- `gg23_bh_feh_gradient_past_tidal`

Radial families:

- `logpoly3`
- `cored_powerlaw_a`

This gives 10 single-component Schechter fits.

## Implementation convention

For every proposed outer point `(eta_t, alpha, log10 M_c)`, the runner uses:

- `t_dis^GG23 = 12 Gyr / eta_t` for the survivability boundary.
- The same `eta_t` in the inversion from observed present-day mass to
  `M_ini^GG23(eta_t)`.

Thus the observed point coordinates and the survivability surface are
self-consistent at each profile-likelihood evaluation.

## Output roots

Use one output root per GG23/radial pair:

- `gg23_schechter_no_bh_logpoly3`
- `gg23_schechter_no_bh_cored_powerlaw_a`
- `gg23_schechter_bh_logpoly3`
- `gg23_schechter_bh_cored_powerlaw_a`
- `gg23_schechter_bh_feh_gradient_logpoly3`
- `gg23_schechter_bh_feh_gradient_cored_powerlaw_a`
- `gg23_schechter_bh_past_tidal_logpoly3`
- `gg23_schechter_bh_past_tidal_cored_powerlaw_a`
- `gg23_schechter_bh_feh_gradient_past_tidal_logpoly3`
- `gg23_schechter_bh_feh_gradient_past_tidal_cored_powerlaw_a`

## Grid phase

Run the 10 grid jobs with `--skip-mcmc`. Use up to 4 concurrent shell jobs.
Each grid job is internally single-process.

Command template:

```bash
.venv/bin/python scripts/run_profile_map_and_exact_mcmc_schechter_powerlaw_a.py \
  --output-root-name OUTPUT_ROOT \
  --survivability-backend gg23 \
  --gg23-model GG23_MODEL \
  --radial-model RADIAL_MODEL \
  --coarse-eta-min 0.4 \
  --coarse-eta-max 3.0 \
  --coarse-eta-n 9 \
  --coarse-alpha-min -1.8 \
  --coarse-alpha-max -0.4 \
  --coarse-alpha-n 8 \
  --coarse-logmc-min 5.8 \
  --coarse-logmc-max 6.9 \
  --coarse-logmc-n 7 \
  --refine-delta-logl 3.0 \
  --refine-min-points 10 \
  --refine-padding-steps 1.0 \
  --local-eta-n 9 \
  --local-alpha-n 9 \
  --local-logmc-n 7 \
  --local-max-passes 3 \
  --local-expand-steps 1.0 \
  --anchor-k 12 \
  --skip-mcmc
```

After the grid phase, inspect each `outputs/tables/summary.json` and
`outputs/tables/refined_grid_results.csv` for:

- best point on a coarse/refined boundary,
- failed evaluations near the preferred region,
- implausible `max_abs_present_mass_residual_fraction`,
- broad or multimodal likelihood structure.

If a best point is on a coarse-grid boundary, expand the relevant coarse range
and rerun that grid before MCMC.

## MCMC phase

Run one output root at a time, using six subprocess chain workers. Do not run
several MCMC roots in parallel.

Command template:

```bash
.venv/bin/python scripts/run_parallel_exact_mcmc_from_existing_refined_grid.py \
  --source-output-root-name OUTPUT_ROOT \
  --mcmc-chains 6 \
  --mcmc-steps 900 \
  --mcmc-burn 300 \
  --mcmc-thin 2 \
  --mcmc-adapt-until 240 \
  --mcmc-adapt-every 20 \
  --mcmc-seed MCMC_SEED \
  --anchor-k 18 \
  --anchor-pool 36
```

Use different seeds for the 10 fits, e.g. `2026060301` through `2026060310`.

The resume script reads `survivability_backend`, `gg23_model_name`, and
`radial_model` from the grid-phase `summary.json`, so the MCMC command should
not need model-specific flags.

## Readiness checks already performed

- `run_profile_map_and_exact_mcmc_schechter_powerlaw_a.py` compiles.
- `run_parallel_exact_mcmc_from_existing_refined_grid.py` compiles.
- GG23 no-BH `logpoly3` one-point grid/refined smoke passed.
- GG23 no-BH `cored_powerlaw_a` one-point grid/refined smoke passed.
- GG23 no-BH two-chain tiny MCMC smoke passed outside the sandbox.
- GG23 no-BH subprocess resume-MCMC smoke passed from an existing refined grid.
- GG23 BH + [Fe/H] + past tides `logpoly3` one-point grid/refined smoke passed.
