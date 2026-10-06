# Baseline re-audit and re-run, October 2026: findings and decisions

Branch `fix/baseline-rerun` (from `fix/identifiability-and-tests`). Not pushed.
Outputs are gitignored, under `variants/baseline_2026-10_*`.

## Summary

- The forward-model framework is sound in principle. As implemented, it does **not** yield data-driven constraints on the GC IMF slope or on N0. This verdict was reached independently by Claude and by Codex.
- **M_c** depends on the disruption law (log M_c 6.2-6.9 across published laws), but barely on the detectability treatment (<= 0.08 dex).
- **alpha** is constrained only conditionally. It shifts by 0.13-0.37 with the completeness treatment, and spans about -0.7 to -1.6 across laws.
- **N0** is not a measurement. The completeness amplitude is not identified by the catalogue, and the iterative correction is biased even when the model is exactly right.
- **What is defensible:** a lower bound on N0 and on the destroyed population, obtained by setting detectability Q = 1, for each disruption law at its published normalisation (eta_t = 1).

## Code fixes (commit ad4f9c7 and following)

- **GG23 Jacobian.** GG23 recomputes the catalogue M_ini from M_now at each eta_t, but the likelihood lacked sum ln|d ln M_ini / d ln M_now|. This term varies by about ±20 nats across the old eta_t posterior.
- **GG23 mass definition.** GG23 M_i is the mass after stellar evolution (mu_sev = 0.55, GG23 eq. 2). It is now converted to birth mass.
- **Baumgardt reference speed back to 240 km/s.** Baumgardt et al. 2019 eq. 5 uses 240 km/s, so eta_t = 1 is exactly the published law again. This only relabels eta_t.
- **Smooth survivability fit.** The band width was stuck on its 1.2 dex bound and the optimiser terminated ABNORMAL. The bounds are widened and each fit gets a Powell polish.
- **Fixed present-mass bins** for completeness, log M in [2, 7.5], identical for all backends and all eta_t.
- **MCMC.** Explicit prior box, split-R-hat/ESS gate, prior-edge fractions, and detectability convergence flags. Fixed parameters are supported (zero-width prior).
- **Two-component runner** brought up to date with the same fixes.
- **Provenance stamp** (commit, diff hash, input sha256) in every run summary. Grid scans can run in parallel.
- **Tests:** 108 fast tests pass.

## Key tests and results

### 1. GG23 with free eta_t has no finite maximum

With the Jacobian, the profile likelihood rises monotonically to the no-disruption limit:

| eta_t | logL |
|---|---|
| 6 | 395.6 |
| 10 | 398.8 |
| 20 | 400.9 |
| 50 | 402.4 |

The low-mass fall-off is then absorbed by Q instead of S. Under the Baumgardt backend eta_t is bounded, posterior 1.29 (-0.40/+0.64). That is probably helped by the fixed catalogue M_ini: 86/165 clusters lie below their own M_cut at eta_t = 1.

**Decision:** fix eta_t = 1 for GG23 (each law as published).

### 2. Detectability iteration: injection-recovery (`scripts/injection_recovery_detectability.py`)

Catalogues were simulated from the model itself. The fitter was given the true survival, alpha and M_c, so only the radial profile, completeness and N0 were free. 8-10 simulations per regime, each about 160 detected clusters.

| True survivor completeness | Fitted | Fitted / true N0(>1e4) | Q = 1 / true |
|---|---|---|---|
| 0.64 | 0.76 | 0.80 (0.65-0.98), converges | 0.56 |
| 0.24 | 0.38 (12 steps) / 0.24 (60 steps) | 0.52 / 0.42-2.07, no fixed point | 0.14 |

- The Q = 1 fit was below the truth in 20/20 simulations, so it is a valid lower bound.
- The rule "divide by completeness" under-corrects: 0.56 vs 0.64.
- Other evidence:
  - The outer likelihood is exactly invariant under Q -> kQ.
  - On real data the fitted completeness implies about 37% of massive survivors (10^5.5-10^6 Msun) are missing, which is implausible.
  - 43-79% of posterior samples in three runs are unconverged after 12 steps. In drifting draws, more iterations lower logL by 0.3-3.7 and raise N0 by 15-40%.

### 3. Baseline posteriors with the detectability iteration (`variants/baseline_2026-10_*`)

| Run | alpha | log10 Mc | N0(>1e4) | mean Q | min ESS |
|---|---|---|---|---|---|
| Baumgardt, eta free | -1.16 (-0.27/+0.18) | 6.29 | 1609 (-651/+2546) | 0.67 | 128 |
| GG23 no BH, eta = 1 | -1.03 | 6.21 | 904 | 0.54 | 376 |
| GG23 BH | -1.52 | 6.81 | 4043 | 0.65 | 341 |
| GG23 BH + [Fe/H] | -1.06 | 6.25 | 876 | 0.78 | 414 |
| GG23 BH + past tides | -1.59 | 6.86 | 5411 | 0.84 | 272 |

- The GG23 BH + [Fe/H] + past tides MCMC was cancelled on 2026-10-03; its stage-1 grid exists.
- The remaining Baumgardt radial families and the two-component run were not started.
- The cached paper chain (6 x 900) has split-R-hat 1.03-1.06 and ESS 144-219, so it fails an ESS >= 400 gate.

### 4. Q = 1 lower bounds, eta_t = 1 (`scripts/compute_q1_lower_bounds.py` -> `variants/baseline_2026-10_q1_lower_bounds.csv`)

| Law | alpha | log10 Mc | N0(>1e4) >= | N(>10^5.5) >= | destroyed 1e4-1e5 >= | destroyed mass >1e4 >= [Msun] |
|---|---|---|---|---|---|---|
| Baumgardt | -1.03 | 6.30 | 1027 | 291 | 494 | 2.0e8 |
| GG23 no BH | -0.69 | 6.20 | 292 | 134 | 71 | 2.5e7 |
| GG23 BH | -1.15 | 6.73 | 959 | 287 | 474 | 2.2e8 |
| GG23 BH + [Fe/H] | -0.84 | 6.23 | 477 | 179 | 175 | 6.0e7 |
| GG23 BH + past tides | -1.46 | 6.84 | 2703 | 386 | 1897 | 3.6e8 |
| GG23 BH + [Fe/H] + past tides | -1.08 | 6.29 | 753 | 192 | 393 | 7.3e7 |

- Values are best fits; uncertainties have not been computed yet.
- Coverage is clusters born at a >= 0.47 kpc only.
- The catalogue has only 10-17 survivors with M_ini 1e4-1e5, so the destroyed low-mass population is an extrapolation of alpha. It is a lower bound conditional on the law and the population model.

## Proposed pragmatic scope (not yet agreed)

- **Paper:** M_c and lower bounds on N0 and on the destroyed population for each disruption law (Q = 1, eta_t = 1). The detectability-corrected values are given only as an illustrative "if the logistic completeness holds" estimate.
- **Possible discriminator between laws:** the predicted destroyed mass (2.5e7-3.6e8 Msun) against independent estimates of dissolved-GC debris, such as N-rich field stars and stream counts. **UNVERIFIED:** the literature values still need an ADS search.
- **Deferred:** a likelihood in observables with positions along orbits, completeness anchoring, coverage tests.

## Open items

- Uncertainties for the Q = 1 fits (profile likelihood or MCMC; cheap without the iteration).
- The manuscript still quotes the old (stale) numbers. The downstream builders are hard-coded to the old variant names (16 scripts).
- README model-comparison numbers are stale (Delta BIC 7.81 vs 5.0).
- `msp_population_extension_notes.md` and `scripts/plot_initial_vs_current_mass_all_models.py` are untracked.
