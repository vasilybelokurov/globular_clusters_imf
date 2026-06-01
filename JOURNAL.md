# Journal

## 2026-05-31 20:00:00 BST

### Monotonic-Q Paper Refresh and BK Two-Component MCMC
- Switched the main single-component paper asset builder to the completed
  monotonic-Q runs:
  - `profile_map_and_exact_mcmc_schechter_logpoly3_logistic_global_monotonic_q`
  - `profile_map_and_exact_mcmc_schechter_step5_logistic_global_monotonic_q`
  - `profile_map_and_exact_mcmc_schechter_powerlaw_a_logistic_global_monotonic_q`
  - `profile_map_and_exact_mcmc_schechter_cored_powerlaw_a_logistic_global_monotonic_q`
- Added the cored power-law radial family to the radial-model comparison figure
  and to the generated single-component macros in `paper/tables/paper_numbers.tex`.
- Updated `paper/main.tex` so the single-component comparison discusses all four
  tested radial families and reports the cored power-law information-criterion
  differences through generated macros.
- Updated the survivability/detectability mass-profile diagnostic to draw
  68 per cent bands from the stored posterior `S` and `Q` surface archives.
- Updated the two-component table generator defaults to point at the monotonic-Q
  single-component and BK two-component variants. The generated table body now
  labels the scalar mean as `Q`, not `C`.
- Confirmed from files on disk:
  - the original BK-labelled two-component run has completed MCMC outputs but
    used the older non-monotonic-Q implementation;
  - the monotonic-Q BK-labelled two-component variant had completed coarse and
    refined grid outputs but no posterior files.
- Restarted the monotonic-Q BK MCMC from the saved refined grid using six
  parallel chain workers:
  - `.venv/bin/python scripts/resume_exact_mcmc_bk_shared_schechter_two_component.py --output-root-name profile_map_and_exact_mcmc_bk_shared_schechter_two_component_logistic_global_monotonic_q --mcmc-chains 6 --mcmc-steps 900 --mcmc-burn 300 --mcmc-thin 2 --mcmc-adapt-until 240 --mcmc-adapt-every 20 --mcmc-seed 20260529`
- Regenerated the single-component paper assets with:
  - `.venv/bin/python scripts/build_paper_assets_exact_single_component.py`
- Recompiled the manuscript once after the single-component edits with:
  - `make -C paper pdf`
- The monotonic-Q BK MCMC finished cleanly:
  - best posterior-sample log-likelihoods by chain were 445.43--445.54;
  - acceptance fractions were 0.251--0.336;
  - Rhat values were 1.009 for `eta_t`, 1.024 for `input_alpha_dndm`, and
    1.026 for `input_log10_m_c_msun`.
- Regenerated:
  - `paper/tables/two_component_numbers.tex`
  - `paper/tables/two_component_results.tex`
  - `paper/figures/two_component_mcmc_diagnostic.pdf`
- Updated two-component headline results:
  - `eta_t = 1.151 -0.293 +0.282`
  - `alpha = -1.207 -0.193 +0.167`
  - `log10 Mc = 6.310 -0.092 +0.083`
  - `N0(>10^4) = 1813 -772 +2358`
  - `f_acc = 0.083 -0.026 +0.027`
  - corrected fixed-label `Delta BIC = -152.2`
- Final compile:
  - `make -C paper pdf`
- Checked `paper/build/main.log` for unresolved references/citations, overfull
  boxes, fatal errors, package errors, and rerun requests; none were found.

## 2026-05-30 12:35:00 BST

### Monotonic-Q Refits: Launch Plan and Grid Stage
- Production detectability change:
  - `src/globular_clusters_imf/detectability_model.py` now uses the monotonic present-day mass-loss proxy in `fit_present_mass_proxy_model`.
  - The proxy is saved in summaries with `model_kind = monotonic_mass_loss`.
  - The longitude-aware detectability path imports the same proxy, so the single-component and BK-labelled two-component fits use the same updated `Q`.
- Smoke checks before launch:
  - compiled the modified detectability modules and run scripts with `.venv/bin/python -m compileall`;
  - fitted the monotonic proxy on the working catalogue;
  - verified the predicted present-day mass is monotonic non-decreasing with initial mass at fixed `a`;
  - verified the proxy does not predict `M_now > M_ini` on the fitted grid.
- Refit management:
  - all new outputs use `_monotonic_q` variant directories;
  - old variants are not overwritten;
  - coarse/refined grid searches are serial within each model script but launched concurrently across the four independent model classes;
  - each grid process therefore uses one Python process, giving four concurrent grid processes total;
  - BLAS/thread environment variables are set to one thread per process: `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`;
  - all grid runs use `--skip-mcmc`; exact MCMC will be launched only after the grids finish and pass sanity checks.
- Grid jobs launched:
  - single-component Schechter + `logpoly3`:
    `variants/profile_map_and_exact_mcmc_schechter_logpoly3_logistic_global_monotonic_q/outputs/grid.log`
  - single-component Schechter + `step5`:
    `variants/profile_map_and_exact_mcmc_schechter_step5_logistic_global_monotonic_q/outputs/grid.log`
  - single-component Schechter + `powerlaw_a`:
    `variants/profile_map_and_exact_mcmc_schechter_powerlaw_a_logistic_global_monotonic_q/outputs/grid.log`
  - BK-labelled two-component shared Schechter IMF with `logpoly3/logpoly3`:
    `variants/profile_map_and_exact_mcmc_bk_shared_schechter_two_component_logistic_global_monotonic_q/outputs/grid.log`
- Planned next stage after grids:
  - inspect each `summary.json`, `coarse_grid_results.csv`, and `refined_grid_results.csv`;
  - check failure counts and whether refined best points land on grid boundaries;
  - then run exact MCMC sequentially, one model class at a time, with six chain workers per model.

## 2026-05-30 12:05:00 BST

### Two-Component Paper Tables Made Regenerable
- Added `scripts/build_two_component_paper_tables.py`.
- The script reads saved exact MCMC outputs and writes:
  - `paper/tables/two_component_numbers.tex` for Section 5 prose macros,
  - `paper/tables/two_component_results.tex` for the body of Table 1.
- Updated `paper/main.tex` so Section 5 no longer hardwires the two-component posterior values, BK label counts, BIC correction, or table entries.
- Updated `paper/Makefile`:
  - `make two-component-tables` regenerates only the two-component paper macros/table,
  - `make assets` now also regenerates the two-component macros/table after the single-component assets.
- The generated BIC partition constant now follows the exact factorial expression in the manuscript equation:
  `ln[N!/(N_in! N_acc!)] = 104.25` for `N=165`, `N_in=107`, `N_acc=58`.
  The older `106.98` value was the `N ln N` approximation used in an older comparison table.
- Rebuilt `paper/build/main.pdf` with `make -C paper pdf`; the LaTeX log has no unresolved references, undefined citations, or overfull boxes.
- No model refits or MCMC runs were launched.

## 2026-05-30 10:45:00 BST

### Section 4/5 Restructure and BK Two-Component Text
- Moved the single-component result text out of a standalone `Results` section and made it the closing subsection of Section 4 (`Single-component implementation`).
- Added a new Section 5, `Two-component model with Belokurov--Kravtsov labels`, focused on the labelled in-situ/accreted extension.
- Wrote the two-component model explicitly as two fixed-label point-process intensities with:
  - shared Schechter IMF parameters,
  - shared survivability parameter `eta_t`,
  - separate `logpoly3` radial profiles and normalizations for the Belokurov--Kravtsov in-situ and accreted labels,
  - the same pooled detectability approximation `Q`.
- Added the labelled likelihood and the count-partition correction for BIC comparisons:
  - `N = 165`, `N_in = 107`, `N_acc = 58`,
  - `C_part = ln[N!/(N_in! N_acc!)] = 106.98`,
  - `BIC_2,cond = BIC_2,raw - 2 C_part`.
- Reported the current exact two-component MCMC result in text and retained the compact table, with BK labels only and no two-component figure included.
- Rebuilt `paper/build/main.pdf` with `make pdf`; the LaTeX log has no unresolved references, undefined citations, or overfull boxes after the edit.

## 2026-05-30 10:10:08 BST

### Paper-Writing Session: Pending Model Rerun Items
- Restructured the paper model section so Section 3 now introduces the full marked GC catalogue model first, then presents the current intrinsic-space likelihood using the effective detectability term `Q(x)`.
- Rewrote the survivability discussion so it presents the current logistic-tail model directly, without referring to older modelling attempts.
- Clarified the lifetime-scale parameter `eta_t`:
  - `eta_t = 1` corresponds to the nominal 12 Gyr survival condition
  - larger `eta_t` lowers the survival boundary in initial mass
  - smaller positive `eta_t` raises the survival boundary
- Updated Figure 1 right panel to show the reference survivability surface at `eta_t = 1` plus the `S = 0.5` boundaries for `eta_t = 0.5`, `1`, and `2`.
- Clarified Section 3.2 detectability:
  - observable-space completeness is binned in `log M_now`, `log D_sun`, `|b|`, and `|l|`
  - the slopes are constrained to the expected monotonic signs
  - `Q(log M_ini, a)` is the binned observable-space completeness averaged over an approximate `p(y|x)`
- Added a new paper diagnostic figure `paper/figures/conditional_observable_approximation.pdf` showing:
  - observed `log10(M_ini/M_now)` in the `(M_ini, a)` plane
  - the current proxy model for the same field
  - per-cluster residuals
- Redesigned the paper detectability diagnostic `paper/figures/detectability_em_maps.pdf`:
  - top row now shows one-dimensional response curves of the fitted observable-space completeness `C(log M_now, log D_sun, |b|, |l|)`, varying one observable at a time and holding the others at catalogue medians
  - bottom row now shows the effective intrinsic-space detectability `Q(log M_ini, a)`
  - this replaces the less compact longitude-split completeness-map presentation
- Replaced the convergence diagnostic `paper/figures/detectability_em_convergence.pdf`:
  - it now shows several fresh representative inner solves for the iterative detectability correction at nearby fixed trial outer parameters
  - the four panels show `N_0(>10^4 Msun)`, mean effective detectability above the same mass, the predicted observed catalogue size, and signed `Delta log L` relative to the first iteration of each solve
  - the text and caption now describe this as an example of the algorithmic inner iteration, not as the final best-fit history, and note that log-likelihood need not increase at every step because `Q` changes
- The new diagnostic figure revealed non-monotonic behaviour of the current present-day mass proxy as a function of `M_ini` at fixed `a`.
- Added `scripts/compare_present_mass_proxy_models.py` to compare present-day mass proxy models.
- Diagnostic results:
  - current polynomial proxy LOO RMS: `0.317 dex`
  - kernel-smoothed residual proxy LOO RMS: `0.330 dex`
  - monotonic mass-loss proxy LOO RMS: `0.313 dex`
  - the monotonic mass-loss proxy is slightly better and physically cleaner than the current polynomial proxy.
- Pending rerun item:
  - replace the current present-day mass proxy used in the detectability/Q construction with the monotonic mass-loss proxy
  - rerun the exact single-component modelling and regenerate paper numbers/figures only after the current round of paper edits is complete
- Important: current paper numbers still correspond to the old present-day mass proxy until that rerun is performed.

### Figure 4 Layout and Text Clarification
- Updated `paper/figures/detectability_em_convergence.pdf` so the horizontal trial-solve colorbar spans the plot-panel width without overlapping the panels or labels.
- Added the grey dashed vertical line at the adopted fixed iteration count (`12`) in all four Figure 4 panels.
- Rebuilt `paper/build/main.pdf` and visually checked page 6 for truncation and overlap.
- Clarified the Section 4.1 discussion: at each step the inner fit optimizes the likelihood for the current fixed `Q`, but the next step updates `Q`; therefore the plotted likelihood sequence need not be monotonic.
- Revised Figure 4 again: removed the colorbar title, moved the colorbar closer to the top panel, removed the in-panel `adopted` label, and changed the top-panel normalization axis to a logarithmic scale.
- Removed the Figure 4 colorbar entirely, expanded the four-panel stack into the freed vertical space, rebuilt `paper/build/main.pdf`, and visually checked page 6.
- Removed the standalone Schechter profile-likelihood figure from the manuscript because the same information is already represented in the posterior/corner figure. The paper asset script no longer regenerates or lists `single_component_model_performance.pdf` in the paper manifest. Rebuilt `paper/build/main.pdf` and checked the relevant pages.
- Fixed the paper build workflow: `make pdf` now only recompiles LaTeX using existing assets, while `make assets` regenerates figures/tables and `make full` runs both. This prevents slow diagnostic assets from being rerun for ordinary manuscript or layout tweaks.
- Built a diagnostic Figure 2 variant using the existing old optimized model grid/context but replacing only the present-day mass-loss proxy with the monotonic model. Outputs: `paper/figures/conditional_observable_approximation_monotonic_oldfit.pdf` and `.png`; in-sample residual RMS is `0.304 dex`. This was not inserted into the manuscript.
- Replaced the proposed two-component MCMC diagnostic figure in the manuscript with a compact table and text-only discussion. The table reports the single-component and labelled Belokurov--Kravtsov two-component posterior summaries, including the in-situ/accreted split of `N0` and `Mstar,0`. The diagnostic figure file remains on disk but is no longer referenced or included.

## 2026-05-22 12:40:00 BST

### Longitude-Split Detectability Figure
- Added a reusable paper-style plotting function `plot_detectability_counts_by_longitude_split_for_paper` in `src/globular_clusters_imf/paper_assets.py`.
- Extended `build_gc_detectability_histogram_tables` in `src/globular_clusters_imf/plotting.py` so the detectability histograms can be rebuilt on externally supplied mass, distance, and latitude edges. This allows fair like-for-like comparisons across subsamples.
- Added a new reusable script `scripts/build_detectability_counts_longitude_split.py`.
- The script:
  - loads the project Baumgardt catalog
  - wraps Galactic longitude onto `[-180^\circ, 180^\circ)`
  - splits the sample into `-90^\circ < l < 90^\circ` and the complementary `|l| \ge 90^\circ` range
  - uses the same present-day mass bins, heliocentric-distance bins, latitude bins, and colour scale in both halves
  - writes a two-row comparison figure plus supporting count tables
- New outputs:
  - `outputs/figures/detectability_counts_longitude_split.pdf`
  - `outputs/figures/detectability_counts_longitude_split.png`
  - `outputs/tables/detectability_counts_longitude_split_mass_summary.csv`
  - `outputs/tables/detectability_counts_longitude_split_histogram_table.csv`
  - `outputs/tables/detectability_counts_longitude_split_subset_summary.csv`
- The split is highly imbalanced:
  - `151` clusters in `-90^\circ < l < 90^\circ`
  - `14` clusters in `|l| \ge 90^\circ`
- Present-day mass-bin counts in the central-longitude half are `(46, 52, 53)`, whereas the complementary half has only `(9, 2, 3)`.
- Immediate qualitative takeaway from the figure:
  - the current Figure 3 signal is driven overwhelmingly by the `-90^\circ < l < 90^\circ` part of the sky
  - the complementary-longitude half is too sparse for a comparably detailed structural comparison on its own

### Longitude-Split Completeness-Map Figure
- The previous longitude split above was only for the descriptive count figure, not the detectability-completeness figure shown in the paper as Figure 3.
- Added a reusable builder `plot_detectability_em_maps_by_longitude_split_for_paper` in `src/globular_clusters_imf/paper_assets.py`.
- Added a reusable script `scripts/build_detectability_em_maps_longitude_split.py`.
- The script:
  - starts from the prepared modelling catalog produced by `fit_catalog_models(...)`
  - splits the sample into `-90^\circ < l < 90^\circ` and `|l| \ge 90^\circ`
  - reruns the full detectability-corrected single-component model comparison in each subset
  - takes the best model in each subset
  - renders a two-row comparison figure using the actual Figure-3-style detectability completeness maps
- New outputs:
  - `outputs/figures/detectability_em_maps_longitude_split.pdf`
  - `outputs/figures/detectability_em_maps_longitude_split.png`
  - `outputs/tables/detectability_em_longitude_split_summary.csv`
  - `outputs/tables/detectability_em_central_longitudes_model_summary.csv`
  - `outputs/tables/detectability_em_outer_longitudes_model_summary.csv`
- Best-fit subset summaries:
  - central longitudes (`151` clusters): best model remains `schechter + logpoly3`, with `N_0 \simeq 759.9`, mean detectability `0.802`, and selection fraction `0.199`
  - complementary longitudes (`14` clusters): best model is `schechter + step5`, with `N_0 \simeq 60.2`, mean detectability `0.584`, and selection fraction `0.233`
- Qualitative comparison of the actual completeness maps:
  - the `-90^\circ < l < 90^\circ` row broadly resembles the current paper Figure 3
  - the `|l| \ge 90^\circ` row shows a visibly harsher completeness drop toward large `D_\odot` even at moderate `|b|`
  - however, that second-row structure is based on only `14` clusters total and should therefore be treated as a noisy diagnostic rather than firm evidence for a necessary longitude term in the completeness law

### Balanced |l| Split for Figure 3
- Generalized `scripts/build_detectability_em_maps_longitude_split.py` so it can split either at a fixed `|l|` threshold or at a specified quantile of the `|l|` distribution, and write outputs under a caller-provided stem.
- Ran the actual Figure-3-style detectability-map diagnostic with a median split in `|l|`.
- The median of the current `|l|` distribution is `17.054^\circ`, giving a nearly balanced split of:
  - `82` clusters with `|l| < 17.1^\circ`
  - `83` clusters with `|l| \ge 17.1^\circ`
- New outputs:
  - `outputs/figures/detectability_em_maps_abs_longitude_median_split.pdf`
  - `outputs/figures/detectability_em_maps_abs_longitude_median_split.png`
  - `outputs/tables/detectability_em_maps_abs_longitude_median_split_summary.csv`
  - `outputs/tables/detectability_em_maps_abs_longitude_median_split_lower_abs_longitude_model_summary.csv`
  - `outputs/tables/detectability_em_maps_abs_longitude_median_split_higher_abs_longitude_model_summary.csv`
- Both balanced subsets prefer the same model family:
  - best IMF: `schechter`
  - best radial model: `logpoly3`
- Balanced-split summary:
  - `|l| < 17.1^\circ`: `N_0 \simeq 628.7`, mean detectability `0.904`, selection fraction `0.130`
  - `|l| \ge 17.1^\circ`: `N_0 \simeq 208.2`, mean detectability `0.790`, selection fraction `0.399`
- Qualitative comparison:
  - the low-`|l|` subset shows the expected strong incompleteness wedge near the plane, especially in the lowest-mass panel
  - the higher-`|l|` subset remains much more complete over the same `D_\odot` range
  - because the two subsets are now similar in size, this balanced split is a more meaningful visual argument that longitude-related sky structure is at least partly absorbed by the difference between lines of sight close to and far from the Galactic-centre direction

### Fixed |l| = 30 deg Split for Figure 3
- Ran the same actual Figure-3-style detectability-map diagnostic with a fixed split at `|l| = 30^\circ`.
- New outputs:
  - `outputs/figures/detectability_em_maps_abs_longitude_30deg_split.pdf`
  - `outputs/figures/detectability_em_maps_abs_longitude_30deg_split.png`
  - `outputs/tables/detectability_em_maps_abs_longitude_30deg_split_summary.csv`
  - `outputs/tables/detectability_em_maps_abs_longitude_30deg_split_lower_abs_longitude_model_summary.csv`
  - `outputs/tables/detectability_em_maps_abs_longitude_30deg_split_higher_abs_longitude_model_summary.csv`
- This gives a moderately imbalanced but still useful split:
  - `109` clusters with `|l| < 30^\circ`
  - `56` clusters with `|l| \ge 30^\circ`
- Both subsets again prefer the same best model family:
  - best IMF: `schechter`
  - best radial model: `logpoly3`
- Subset summaries:
  - `|l| < 30^\circ`: `N_0 \simeq 688.5`, mean detectability `0.872`, selection fraction `0.158`
  - `|l| \ge 30^\circ`: `N_0 \simeq 174.1`, mean detectability `0.565`, selection fraction `0.322`
- Qualitative comparison:
  - the `|l| < 30^\circ` row still shows the strongest incompleteness in the lowest-mass slice near the plane
  - the `|l| \ge 30^\circ` row shows a broader wedge-like decline extending through all three representative mass slices
  - the top-row second and third panels remain nearly saturated because the fitted completeness in those higher-mass slices is close to unity across the distance range populated by the low-`|l|` subset

### Correction: Longitude-Split Figure 3 Must Share the Intrinsic Model
- Corrected the methodology of `scripts/build_detectability_em_maps_longitude_split.py`.
- The earlier version wrongly refit the intrinsic single-component model independently in each longitude subset. This mixed intrinsic population differences with detectability and was not the intended diagnostic.
- The script now:
  - fits the full detectability-corrected single-component model once on the full catalog
  - keeps that intrinsic model fixed
  - keeps the same mass, distance, and latitude grid in both longitude subsets
  - refits only the logistic detectability law to the subset-specific observed counts
- This means the two rows in the longitude-split figure are now directly comparable as detectability corrections relative to the same underlying intrinsic GC population.
- Rebuilt the `|l| = 30^\circ` split outputs with the corrected logic:
  - `outputs/figures/detectability_em_maps_abs_longitude_30deg_split.pdf`
  - `outputs/figures/detectability_em_maps_abs_longitude_30deg_split.png`
  - `outputs/tables/detectability_em_maps_abs_longitude_30deg_split_summary.csv`
  - plus subset-specific completeness-grid and observable-histogram tables under the same stem
- Corrected `|l| = 30^\circ` summary:
  - `|l| < 30^\circ`: `109` clusters, mean detectability against the shared intrinsic model `0.536`
  - `|l| \ge 30^\circ`: `56` clusters, mean detectability against the shared intrinsic model `0.286`
- The corrected figure is much more strongly separated than the earlier subset-refit version, which confirms that the original interpretation was indeed contaminated by re-fitting the intrinsic model independently in each subset.

## 2026-05-22 15:10:00 BST

### Full Single-Component EM Inference with Longitude-Dependent Detectability
- Implemented a separate longitude-aware detectability pipeline in `src/globular_clusters_imf/detectability_longitude_model.py`.
- This new path keeps the existing detectability model intact and introduces a parallel EM-style single-component inference with
  - the same intrinsic variables `(log M_ini, log a)`
  - the same fixed survivability grid
  - a detectability law extended to `C(log M_now, log D_\odot, |b|, |l|)`
  - `|l|` implemented as wrapped absolute longitude in `[0^\circ, 180^\circ]`
- The observable prediction context is now four-dimensional in
  - present-day mass
  - heliocentric distance
  - absolute latitude
  - absolute longitude
- Added new reusable functions for:
  - building the 4D observable context
  - predicting the complete surviving population in 4D observable space
  - fitting the 5-parameter monotonic logistic completeness law
  - mapping the fitted completeness back into the intrinsic-space effective selection grid
  - writing completeness grids, observable histograms, and per-cluster completeness tables
- Added a reusable driver script `scripts/run_abs_longitude_detectability_inference.py`.
- The driver:
  - loads the prepared Baumgardt catalog
  - runs the six single-component model families with the longitude-aware EM iteration
  - saves the model-comparison table and best-fit summary
  - builds an updated Figure-8-style intrinsic profile plot for the best longitude-aware fit
  - writes a baseline-versus-longitude-aware comparison table

### Longitude-Aware Single-Component Results
- New summary products:
  - `outputs/tables/joint_fixed_survival_detectability_abs_longitude_em_model_summary.csv`
  - `outputs/tables/joint_fixed_survival_detectability_abs_longitude_em_summary.json`
  - `outputs/tables/joint_fixed_survival_detectability_abs_longitude_em_iteration_history.csv`
  - `outputs/tables/joint_fixed_survival_detectability_abs_longitude_em_completeness_grid.csv`
  - `outputs/tables/joint_fixed_survival_detectability_abs_longitude_em_observable_histogram.csv`
  - `outputs/tables/joint_fixed_survival_detectability_abs_longitude_em_catalog_completeness.csv`
  - `outputs/tables/joint_fixed_survival_detectability_abs_longitude_em_vs_baseline.csv`
- New figure products:
  - `outputs/figures/single_component_profiles_detectability_abs_longitude.pdf`
  - `outputs/figures/single_component_profiles_detectability_abs_longitude.png`
- Best longitude-aware model remains `schechter + logpoly3`.
- Best-fit parameters:
  - `alpha_dndm = -0.940`
  - `log10(M_c/Msun) = 6.301`
  - `N_0 = 792.27`
  - selection fraction `= 0.2083`
  - raw survival fraction `= 0.2509`
  - mean detectability `= 0.830`
  - total initial stellar mass `= 3.17e8 Msun`
- Relative to the current baseline detectability-corrected single-component result:
  - baseline: `alpha = -0.947`, `log10(M_c/Msun) = 6.309`, `N_0 = 784.62`, `M_{*,0} = 3.13e8 Msun`
  - longitude-aware: `alpha = -0.940`, `log10(M_c/Msun) = 6.301`, `N_0 = 792.27`, `M_{*,0} = 3.17e8 Msun`
  - change in total count: `+7.65` clusters, i.e. only `+0.98%`
- So the main scientific conclusion is that explicitly including `|l|` in the detectability model changes the single-component inference only mildly.

### IMF and Radial-Profile Comparison
- The updated intrinsic profile figure shows:
  - the IMF is slightly higher at low `M_ini` and very slightly lower around the peak than in the baseline detectability-corrected fit
  - the high-mass cutoff behaviour is essentially unchanged
  - the radial birth profile is shifted upward modestly in the inner Galaxy and is almost unchanged at larger semimajor axis
- Quantitative radial-profile diagnostics from the best longitude-aware fit:
  - peak birth intensity remains at `a ≈ 1.40 kpc`
  - peak birth intensity rises from `737.8` to `755.1` clusters per dex in `a`
  - the median birth radius changes only slightly, from `1.90 kpc` to `1.88 kpc`

### Verification
- Verified with:
  - `python -m compileall src/globular_clusters_imf/detectability_longitude_model.py scripts/run_abs_longitude_detectability_inference.py`
  - `python scripts/run_abs_longitude_detectability_inference.py`
  - manual inspection of `single_component_profiles_detectability_abs_longitude.pdf`

## 2026-05-21 16:35:00 BST

### Paper Figure Update
- Added a new paper-specific overview figure builder in `src/globular_clusters_imf/paper_assets.py` for the Baumgardt mass-versus-semimajor-axis catalogue view.
- The new figure has two panels:
  - scatter plot of present-day and initial GC masses versus semimajor axis
  - inferred initial masses versus semimajor axis coloured by present-day half-mass density
- Deliberately omitted the old black survival-threshold curve from this overview figure, per user request.
- Added a lightweight reusable script `scripts/build_paper_figure1_overview.py` so this figure can be regenerated without rerunning the full modelling pipeline.
- Updated `paper/main.tex` to insert the new catalogue overview as Figure 1 at the end of the `Data` section, leaving the survivability plane in place so it becomes Figure 2 automatically.
- Revised the right-hand panel after user feedback: instead of a smoothed density map, it now shows the spread in present-day half-mass density at fixed $(M_{\rm ini}, a)$ using point colour.
- Revised the right-hand panel again after user feedback: it now colour-codes the surviving clusters by current half-mass radius rather than density, to show the spread in present-day size at fixed $(M_{\rm ini}, a)$.

### Density Dependence Check
- Added `scripts/check_density_survivability_dependence.py` to test whether present-day density carries information about Baumgardt's estimated survivability beyond $(M_{\rm ini}, a)$.
- Used `remaining_dissolution_time_gyr` as the empirical survivability proxy and fitted simple regressions in `\log T_{\rm diss,rem}` with and without density terms.
- Result for half-mass density:
  - adding `log_half_mass_density` to a baseline model with `log_initial_mass_msun` and `log10(semi_major_axis_kpc)` improves the fit with `\Delta \mathrm{BIC} = -8.34`
  - coefficient on `log_half_mass_density` is positive and significant (`p \simeq 3\times10^{-4}`)
  - partial residual correlation remains modest but non-zero (`r \simeq 0.28`, `\rho_{\rm Spearman} \simeq 0.22`)
- Result for core density:
  - adding `log_core_density` does not materially improve the fit (`\Delta \mathrm{BIC} = +3.20`)
  - coefficient is not significant (`p \simeq 0.17`)
- Proxy-initial-density check using `M_{\rm ini}` and the current half-mass radius:
  - defined `\log \tilde{\rho}_{\rm ini,h} = \log M_{\rm ini} - \log[(4\pi/3) r_{h,\rm current}^3]`
  - adding this proxy to the baseline model improves the fit with `\Delta \mathrm{BIC} = -6.23`
  - equivalently, adding `\log r_{h,\rm current}` itself gives the same improvement and a positive coefficient (`p \simeq 9\times10^{-4}`)
  - interpretation: at fixed `(M_{\rm ini}, a)`, current half-mass size carries modest extra information about Baumgardt's remaining dissolution time
- Saved the direct `\log r_{h,\rm current}` test in the summary outputs alongside the density and proxy-density predictors for later reuse.
- Carried out a more controlled check including present-day mass in addition to `(M_{\rm ini}, a)`:
  - adding `\log r_{h,\rm current}` on top of `\log M_{\rm ini}`, `\log a`, and `\log M_{\rm now}` still gives a positive coefficient (`0.207 \pm 0.085`, `p \simeq 0.015`)
  - but the model improvement is only mild (`\Delta \mathrm{BIC} \simeq -0.96`)
  - equivalently, at fixed current mass, adding `\log \rho_h` gives a negative coefficient (`p \simeq 0.015`)
  - this indicates that the size effect is real but substantially weaker once current mass is accounted for
- Extended the controlled comparison to the other catalog radii:
  - `\log r_{c,\rm current}` adds no useful information once `\log M_{\rm ini}`, `\log a`, and `\log M_{\rm now}` are included (`\Delta \mathrm{BIC} \simeq +5.05`, `p \simeq 0.81`)
  - `\log r_{h,\rm current}` remains the only clearly non-zero radius term, but only mildly (`\Delta \mathrm{BIC} \simeq -0.96`, `p \simeq 0.015`)
  - `\log r_{t,\rm current}` is borderline (`\Delta \mathrm{BIC} \simeq +1.14`, `p \simeq 0.050`) and likely not an independently useful structural predictor in this controlled test
- Checked the likely explanation in the Baumgardt structural quantities:
  - `\log r_{h,\rm current}` correlates strongly with `\log t_{\rm relax}` (`r \simeq 0.83`)
  - `\log t_{\rm relax}` correlates strongly with `\log T_{\rm diss,rem}` (`r \simeq 0.79`)
  - so in the surviving sample, larger current half-mass radii are associated with longer relaxation times and hence longer remaining dissolution times
- Interpretation: in the observed Baumgardt sample, present-day half-mass density appears to contain some extra survivability information beyond $(M_{\rm ini}, a)$, but the effect is modest and is not included in the current survival model.

## 2026-05-21 17:25:00 BST

### Restricted-Radius Variant
- Added a reusable subset runner `scripts/run_subset_a_lt_100kpc_single_component_profiles.py`.
- This script:
  - loads the project Baumgardt catalog
  - restricts to clusters with `semi_major_axis_kpc < 100`
  - runs the baseline fixed-survival single-component analysis
  - runs the detectability-corrected single-component analysis
  - writes all outputs into a separate variant tree under `variants/a_lt_100kpc/`
  - generates the current-paper Figure 8 analogue for that restricted sample
- Confirmed that no existing main-project outputs were overwritten; all restricted-sample products live under:
  - `variants/a_lt_100kpc/outputs/tables`
  - `variants/a_lt_100kpc/outputs/figures`
- Restricted-sample summary:
  - input clusters: `159`
  - best detectability-corrected single-component family: `schechter + logpoly3`
  - best reconstructed total initial count: `798.1`
  - total selection fraction: `0.199`
  - mean detectability: `0.799`
- Generated restricted-sample Figure 8 equivalents:
  - `variants/a_lt_100kpc/outputs/figures/figure8_single_component_profiles_a_lt_100kpc.pdf`
  - `variants/a_lt_100kpc/outputs/figures/figure8_single_component_profiles_a_lt_100kpc.png`

## 2026-05-21 08:00:00 BST

### Operating Rules
- Maintain `JOURNAL.md` as a detailed running log of work performed in this workspace.
- Write reusable Python scripts to files instead of doing analysis only in ad hoc interactive commands.
- Keep the project focused on reconstructing the initial mass function of Milky Way globular clusters using Baumgardt data.

### Initial Workspace State
- Working directory: `/Users/vasilybelokurov/Work/Code/globular_clusters_imf`
- The directory was empty at session start.
- The directory is not a Git repository.
- No local Python scientific stack was installed in the default interpreter at session start.

### Project Definition
- Science goal: reconstruct the shape of the Milky Way globular-cluster initial mass function while allowing its normalization to vary with Galactocentric distance.
- Core working assumption from the user: the GC IMF shape is radius-independent, while the surviving sample is strongly radius-dependent because of dynamical destruction in the inner Galaxy and sparse sampling in the outskirts.
- Key data source: Baumgardt estimates of present-day and initial globular-cluster masses, combined with orbital information to use semimajor axis as the radial coordinate.

### Source-Backed Context
- Read Baumgardt et al. (2019), `MNRAS, 482, 5138`, especially the discussion around Figures 7 and 8.
- The paper shows that surviving inner-halo and bulge clusters are biased toward high initial masses, with nearly all clusters inside about `2 kpc` having inferred initial masses above `10^6 Msun`.
- The paper also shows that outer-halo clusters are fewer in number, so the high-mass tail is not well sampled at large semimajor axis even if the underlying IMF shape is shared.
- The paper fits outer clusters with both a lognormal and a `dN/dM ~ M^-2` form and argues that both can match the massive end once the radius-dependent destruction is accounted for.
- Identified the current Baumgardt online database (`March 2023` version) as the most convenient machine-readable source for present mass, initial mass, Galactocentric distance, and orbital parameters.

### Actions Taken
- Inspected the workspace and confirmed there was no existing project scaffold to extend.
- Inspected journal conventions in nearby projects to keep logging style consistent.
- Created the initial project directory structure:
  - `data/raw`
  - `data/processed`
  - `outputs/figures`
  - `outputs/tables`
  - `scripts`
  - `src/globular_clusters_imf`

### Immediate Next Steps
- Create the Python package scaffold and dependency metadata.
- Add a reusable ingestion script for Baumgardt catalog tables.
- Add a first-pass statistical model treating the observed surviving catalog as a radius-dependent truncation of a universal IMF.

## 2026-05-21 07:55:47 BST

### Project Scaffold
- Added root project files:
  - `README.md`
  - `pyproject.toml`
  - `.gitignore`
  - `src/globular_clusters_imf/__init__.py`
- Created a project-local virtual environment at `.venv`.
- Installed the initial scientific stack into the local environment with editable package install:
  - `numpy`
  - `pandas`
  - `scipy`
  - `matplotlib`
  - `requests`
  - `lxml`

### Ingestion Pipeline
- Added `src/globular_clusters_imf/catalog.py`.
- Added `scripts/fetch_baumgardt_catalog.py`.
- Implemented a reusable downloader/parser for:
  - `https://people.smp.uq.edu.au/HolgerBaumgardt/globular/orbits.html`
  - `https://people.smp.uq.edu.au/HolgerBaumgardt/globular/parameter.html`
- Saved the raw HTML snapshots to:
  - `data/raw/baumgardt_orbits.html`
  - `data/raw/baumgardt_parameters.html`
- Parsed the display-style numeric strings used on the Baumgardt site, including entries such as `8.53 ± 0.05 · 10^5`.
- Joined the orbital and structural tables on cluster label and wrote the clean catalog to:
  - `data/processed/baumgardt_gc_catalog.csv`
- Wrote a small catalog summary to:
  - `data/processed/baumgardt_gc_catalog_summary.csv`

### Current Catalog Contents
- Joined cluster count: `165`
- Derived columns now include:
  - `semi_major_axis_kpc`
  - `eccentricity`
  - `initial_mass_msun`
  - `mass_loss_fraction`
  - `radius_bin_paper`
- The first rows of the sorted catalog confirm the expected inner-Galaxy bias toward very large inferred initial masses.

### Statistical Model
- Added `src/globular_clusters_imf/model.py`.
- Added `src/globular_clusters_imf/plotting.py`.
- Added `scripts/fit_gc_imf_model.py`.
- Implemented a first-pass model in which:
  - the IMF shape is global
  - the survival selection is tied to the Baumgardt et al. (2019) dissolution-time prescription
  - the initial radial normalization is reconstructed after fitting by inverse-probability weighting
- Used the Baumgardt et al. (2019) prescription with:
  - age `= 12 Gyr`
  - mean initial stellar mass `= 0.65 Msun`
  - circular speed `= 240 km/s`

### Important Modeling Adjustment
- A strict hard survival threshold from the approximate dissolution-time formula is not exactly consistent with the Baumgardt catalog itself.
- Measured mismatch:
  - `80` of `165` clusters fall below the unshifted theoretical threshold
  - the maximum discrepancy is about `0.381 dex`
- Rather than hiding that mismatch, introduced an explicit fitted threshold calibration parameter:
  - `selection_offset_dex`
- In the current implementation the fitted lognormal model and the power-law baseline both use:
  - `selection_offset_dex = -0.38230922842646786`
- Interpretation:
  - the paper-motivated survival boundary is useful, but in this simple pipeline it must be shifted downward by about `0.38 dex` to remain compatible with the catalog.

### Fitted First-Pass Results
- Truncated lognormal fit to `log10(M_ini / Msun)`:
  - `mu_log10_msun = 5.88`
  - `sigma_log10_msun = 0.5926`
  - `selection_offset_dex = -0.3823`
- Truncated power-law baseline over `10^3` to `10^8 Msun`:
  - `beta = -1.726`
  - `selection_offset_dex = -0.3823`
- The power-law baseline reconstructs a vastly larger initial cluster population than the lognormal model, so the lognormal fit is currently the more conservative working baseline for this project.

### Reconstructed Initial Counts
- Under the current lognormal model:
  - total reconstructed initial cluster count: `344.1`
  - inner bin `a < 3 kpc`: `222.8`
  - middle bin `3 <= a < 15 kpc`: `87.1`
  - outer bin `a >= 15 kpc`: `34.1`
- These counts are in the same qualitative direction as the 2019 paper:
  - strong destruction in the inner Galaxy
  - much milder correction in the outskirts

### Outputs Written
- Tables:
  - `outputs/tables/catalog_with_survival_thresholds.csv`
  - `outputs/tables/model_summary.json`
  - `outputs/tables/radial_profile_lognormal.csv`
  - `outputs/tables/radial_profile_powerlaw.csv`
- Figures:
  - `outputs/figures/figure7_like_mass_vs_radius.png`
  - `outputs/figures/figure8_like_imf_by_radius.png`

### Execution Stability
- Configured `scripts/fit_gc_imf_model.py` to use repo-local Matplotlib and cache directories so figure generation works cleanly inside the sandboxed environment.
- Replaced infinite optimization penalties with large finite penalties to avoid SciPy finite-difference warnings from invalid parameter probes.

### Current Limitations
- The current pipeline uses the March 2023 Baumgardt web catalog, not a frozen 2019 catalog snapshot.
- The radial reconstruction uses the observed survivor orbits as the working orbit sample for inverse-probability weighting.
- The power-law and lognormal likelihood values are not directly comparable as model-selection metrics because one is written in `M` and the other in `log10 M`.
- The current model still approximates survival with a calibrated hard threshold; a soft selection function in mass and orbit is the next obvious refinement.

### Next Steps
- Replace the hard-threshold selection with a soft survival function centered on the paper-motivated threshold.
- Add a proper likelihood for the radial normalization, ideally as a smooth function of semimajor axis rather than only three bins.
- Freeze or recover the exact 2019 Baumgardt catalog used for Figures 7 and 8 so the project can distinguish paper-reproduction mode from updated-catalog inference mode.
- Add uncertainty propagation, likely via bootstrap or full Bayesian sampling once the selection model is stabilized.

## 2026-05-21 08:10:40 BST

### User Request
- The user asked for a clearer plot of the GC IMF as radius-dependent patches in `5` radial bins, with the global model overplotted.
- The main technical requirement was to handle the radial evolution of the IMF normalization correctly before comparing shapes across radius.

### Modeling Decision For Radial Normalization
- For a radial bin `j`, the model assumes:
  - local initial mass function `= N_init,j * phi(log M)`
  - `phi(log M)` is the global lognormal IMF shape
  - `N_init,j` is the radial-bin normalization
- I estimated the survival completeness curve in each radial bin directly from the orbit sample in that bin:
  - `C_j(log M) = mean_i [log M >= log M_cut,i]`
  - where `log M_cut,i` is the calibrated survival threshold for cluster orbit `i`
- I then inferred the bin normalization from the global lognormal and the completeness curve:
  - `f_surv,j = integral phi(log M) * C_j(log M) d log M`
  - `N_init,j = N_surv,j / f_surv,j`
- This is the key step that removes the radius dependence in the total number of clusters before comparing IMF shape.

### IMF Patch Construction
- Added `estimate_lognormal_imf_patches()` to `src/globular_clusters_imf/model.py`.
- Adopted `5` equal-count bins in `log10(semimajor axis)` so each radial patch has similar statistical weight.
- The resulting radial bins are:
  - `0.47 <= a < 1.90 kpc`
  - `1.90 <= a < 3.18 kpc`
  - `3.18 <= a < 6.04 kpc`
  - `6.04 <= a < 14.70 kpc`
  - `14.70 <= a < 284.91 kpc`
- In each radial bin and mass bin, estimated the local IMF density as:
  - `phi_hat_j = N_obs,j / (N_init,j * C_j * Delta log M)`
- This gives a completeness-corrected and normalization-corrected patch estimate of the common IMF.

### Definition Of The Probed Range
- A mass bin is counted as part of the radial bin's leverage range when:
  - completeness `C_j >= 0.5`
  - expected initial number in that mass bin from the global lognormal is at least `1`
- The plotted filled patches are the subset of those leverage bins that also contain at least one observed surviving cluster.
- This avoids pretending that a radial bin constrains masses that are dynamically destroyed at low `M_ini` or too sparsely populated at high `M_ini`.

### Code Added
- Extended `src/globular_clusters_imf/model.py` with:
  - `estimate_lognormal_imf_patches()`
  - `lognormal_pdf_per_dex()`
  - `format_radius_bin_label()`
- Extended `src/globular_clusters_imf/plotting.py` with:
  - `plot_global_imf_with_radial_patches()`
  - `contiguous_true_segments()`
- Updated `scripts/fit_gc_imf_model.py` so the new patch outputs and figure are produced automatically.

### Outputs Written
- New figure:
  - `outputs/figures/global_imf_patches_5_radius_bins.png`
- New tables:
  - `outputs/tables/radial_imf_patch_summary_5bins.csv`
  - `outputs/tables/radial_imf_patch_table_5bins.csv`

### Result
- The new figure shows the intended behavior:
  - the innermost bin contributes only the high-mass end
  - progressively larger semimajor-axis bins extend to lower `M_ini`
  - after correcting for radial normalization and survival completeness, the colored patches can be compared directly to the single global lognormal curve
- This is a better visualization of the user's working hypothesis than the earlier three-panel histogram plot.

### Verification
- Re-ran:
  - `.venv/bin/python -m compileall src scripts`
  - `.venv/bin/python scripts/fit_gc_imf_model.py`
- Confirmed the new figure and both new CSV outputs are written successfully.

## 2026-05-21 08:16:33 BST

### User Request
- The user asked for a plot of the survivability function `S(M_ini, a)` in the plane of initial mass and orbital semimajor axis.

### Modeling Choice
- In the current project, survivability is not defined by `a` alone at the object level.
- The underlying hard survival rule depends on orbital information through the calibrated dissolution threshold, which varies from cluster to cluster because of orbit differences such as eccentricity.
- Therefore, for plotting `S(M_ini, a)` I defined the displayed map as the current-model survival probability after averaging over the observed distribution of cluster orbits at fixed semimajor axis:
  - `S(log M, a) = < I[log M >= log M_cut,eff] >_{orbit | a}`
- Here:
  - `log M_cut,eff = log M_cut + selection_offset_dex`
  - `selection_offset_dex = -0.3823`
- This means the map is an empirical-orbit-marginalized survival function, not a fully first-principles dynamical calculation.

### Implementation
- Added `estimate_survivability_map()` to `src/globular_clusters_imf/model.py`.
- The map is constructed on a 2D grid in:
  - `log10(M_ini / Msun)`
  - `log10(a / kpc)`
- At each grid value of `a`, the code applies a Gaussian kernel in `log10(a)` to nearby observed clusters and averages their hard survival indicators.
- Adopted:
  - `n_radius_grid = 160`
  - `n_mass_grid = 180`
  - smoothing bandwidth `= 0.18 dex` in `log10(a)`
- Wrote the full map to:
  - `outputs/tables/survivability_map_lognormal.csv`

### Plot Added
- Added `plot_survivability_map()` to `src/globular_clusters_imf/plotting.py`.
- Added the new figure to the standard pipeline:
  - `outputs/figures/survivability_map_initial_mass_vs_radius.png`
- The plot shows:
  - color scale for survival probability from `0` to `1`
  - white contours at `S = 0.1`, `0.5`, and `0.9`

## 2026-05-21 16:15:00 BST

### User Request
- The user asked to implement an EM-like detectability correction for the single-component model.
- The working detectability function is:
  - `C(log M_now, log D_sun, |b|)`
- The intended iterative scheme is:
  - assume the current single-component intrinsic model is approximately correct
  - use spherical symmetry to predict the complete surviving GC distribution in observable space
  - infer a smooth completeness function from the mismatch between observed and predicted counts
  - fold that completeness back into the intrinsic `(log M_ini, log a)` likelihood
  - iterate until the model stabilizes

### Modeling Design
- Kept the existing single-component and two-component pipelines intact.
- Generalized `JointLikelihoodContext` in `src/globular_clusters_imf/joint_model.py` so it now carries:
  - the raw survival grid `S(log M_ini, a)`
  - a generic selection grid for the actual fitted likelihood
- In the original model these two are identical, so legacy behavior is unchanged.
- In the detectability-corrected model the fitted observation probability is:
  - `S(log M_ini, a) * Q(log M_ini, a)`
  - where `Q` is the detectability completeness averaged back into intrinsic space.

### New Detectability Module
- Added `src/globular_clusters_imf/detectability_model.py`.
- Implemented a new top-level reusable function:
  - `fit_single_component_detectability_em()`
- This function:
  - calibrates the same fixed Baumgardt survival grid as the baseline model
  - fits the chosen single-component IMF+radial model in intrinsic space
  - predicts the complete surviving catalog in observable space
  - fits a smooth logistic completeness model
  - averages that completeness back into `(log M_ini, log a)`
  - refits the intrinsic model and iterates

### Observable-Space Approximation
- Present-day mass mapping:
  - fitted an empirical smooth proxy for `log10(M_now / M_ini)` as a function of `log10 M_ini` and `log10 a`
  - design terms used:
    - constant
    - `z(log M_ini)`
    - `z(log a)`
    - `z(log M_ini) * z(log a)`
    - `z(log a)^2`
  - residual scatter is propagated as a Gaussian spread in `log10 M_now` when projecting into observable bins
- Sky geometry:
  - assumed spherical symmetry about the Galactic centre
  - approximated the current Galactocentric radius of a survivor by its orbital semimajor axis `a`
  - sampled isotropic sky positions at each `a`
  - converted those to heliocentric distance `D_sun` and Galactic latitude `b` with the Sun fixed at `8.2 kpc`

### Completeness Model
- Adopted a bounded logistic form:
  - `C = sigmoid(beta0 + sM * z(log M_now) - sD * z(log D_sun) + sb * z(|b|))`
- Enforced the expected monotonic trends by parameterizing:
  - `sM > 0`
  - `sD > 0`
  - `sb > 0`
  - and inserting the distance term with a minus sign
- Fitted the completeness model to binned observable counts with a Poisson likelihood:
  - observed counts per `(log M_now, D_sun, |b|)` bin
  - expected counts = complete predicted survivor counts times `C`

### EM-Like Iteration
- Started from the best completeness-free single-component model:
  - `schechter + logpoly3`
- Then alternated:
  1. fit intrinsic IMF+radial model given current detectability correction
  2. project the complete surviving model into observable space
  3. refit the logistic completeness surface
  4. average completeness back onto the intrinsic grid
- Used:
  - `6` iterations
  - relaxation factor `0.7`

### First-Pass Results
- The detectability correction converged smoothly rather than collapsing back to `C = 1`.
- Baseline best single-component model without detectability correction:
  - `N0 = 548.3`
  - raw survival fraction `= 0.30093`
- Detectability-corrected best single-component model after EM iteration:
  - `N0 = 784.6`
  - raw survival fraction `= 0.25199`
  - fitted selection fraction `= 0.21029`
  - implied mean detectability among surviving clusters `= 0.8345`
- Relative change:
  - `N0` increased by a factor of `1.431`
- The completeness-corrected fit also improves the single-component likelihood slightly:
  - baseline `logL = 469.02`
  - detectability-corrected `logL = 471.47`

### Completeness Behaviour
- The fitted completeness surface behaves as intended:
  - higher completeness at larger `|b|`
  - lower completeness at larger heliocentric distance
  - higher completeness at larger present-day mass
- The fitted slope parameters at convergence are:
  - mass slope `= 1.150`
  - distance slope `= 1.486`
  - latitude slope `= 6.731`
- The strongest effect in this first pass is the sharp loss of completeness close to the Galactic plane.

### Outputs Written
- Tables:
  - `outputs/tables/joint_fixed_survival_detectability_em_summary.json`
  - `outputs/tables/joint_fixed_survival_detectability_em_iteration_history.csv`
  - `outputs/tables/joint_fixed_survival_detectability_em_completeness_grid.csv`
  - `outputs/tables/joint_fixed_survival_detectability_em_observable_histogram.csv`
  - `outputs/tables/joint_fixed_survival_detectability_em_catalog_completeness.csv`
- Figures:
  - `outputs/figures/joint_fixed_survival_detectability_em_completeness_by_mass.png`
  - `outputs/figures/joint_fixed_survival_detectability_em_convergence.png`

### Pipeline Integration
- Updated `scripts/fit_gc_imf_model.py` so the detectability-corrected single-component fit runs automatically using the best baseline single-component model specification.
- Updated `src/globular_clusters_imf/plotting.py` to generate the new detectability figures.
- Updated `src/globular_clusters_imf/__init__.py` to export `fit_single_component_detectability_em`.

### Verification
- Re-ran:
  - `.venv/bin/python -m compileall src scripts`
  - `.venv/bin/python scripts/fit_gc_imf_model.py`
- Confirmed:
  - the new tables are written
  - the new figures are written
  - the standard existing outputs still build successfully

### Important Caveats
- This is still a first-pass approximation.
- The main assumptions that may need revision later are:
  - `r_current ~ a` for the spherical geometry projection
  - the empirical present-mass proxy fitted on the surviving sample
  - a single smooth logistic completeness law with only three observable covariates
- Even so, this is already useful because it provides a self-consistent lower-order correction that moves the inferred total initial GC count upward in the expected direction.

## 2026-05-21 17:35:00 BST

### User Request
- The user asked for the paper to be restructured around the new scientific logic:
  - introduce survivability and detectability early
  - present the perfect-detectability single-component model first
  - then describe and apply the EM-like detectability correction
  - then introduce the two-component model and compare its results
- The user specifically asked not to lose important technical detail while changing the narrative flow.

### Manuscript Restructure
- Reworked `paper/main.tex` to follow the new sequence:
  1. `Data`
  2. `Selection effects`
     - survivability
     - detectability
  3. `Single-component model with perfect detectability`
  4. `Single-component results assuming perfect detectability`
  5. `Detectability-corrected single-component model`
     - EM-like completeness correction
     - detectability-corrected results
  6. `Two-component model and results`
     - formulation
     - results
  7. `Discussion`
  8. `Conclusions`
- Preserved the explicit Poisson point-process likelihood for the single-component model.
- Moved the explicit two-component likelihood into its own later section, after the detectability-corrected single-component stage.
- Kept the full explanation of the conditional BIC correction for the labelled in-situ/accreted split.

### New Paper Figures
- Added a paper-quality detectability motivation figure:
  - `paper/figures/detectability_counts.pdf`
  - shows observed GC counts in the `(D_sun, |b|)` plane for three present-day mass bins
- Added a paper-quality detectability-correction figure:
  - `paper/figures/detectability_em_results.pdf`
  - top row: fitted completeness maps in `(D_sun, |b|)` for representative `M_now` slices
  - bottom panel: convergence of the EM-like iteration

### Paper Asset Pipeline Changes
- Extended `src/globular_clusters_imf/paper_assets.py` to:
  - run the detectability-corrected single-component fit
  - generate the two new paper figures
  - include the detectability-corrected single-component result in the paper summary payload
  - add detectability-corrected values to `paper_numbers.tex`
  - add a detectability-corrected row to the key results summary table
- The new paper macros are:
  - `\DetectabilityCorrectedNzero = 784.6`
  - `\DetectabilityCorrectedMassZeroEight = 3.13`
  - `\DetectabilityMeanCompleteness = 0.835`
  - `\DetectabilityCountRatio = 1.43`

### Updated Interpretation In The Paper
- The perfect-detectability single-component baseline remains:
  - `N0 = 548.3`
  - `M_star,0 = 2.69 x 10^8 Msun`
- The detectability-corrected single-component model is now stated explicitly as the stronger current lower bound:
  - `N0 = 784.6`
  - `M_star,0 = 3.13 x 10^8 Msun`
- The discussion now distinguishes carefully between:
  - the perfect-detectability baseline normalization
  - the detectability-corrected single-component lower bound
  - the perfect-detectability two-component split
- Added explicit caveat in the lower-bound discussion and conclusions that the full detectability correction has not yet been propagated through the two-component model.

### Build And Verification
- Rebuilt paper assets with:
  - `.venv/bin/python scripts/build_paper_assets.py`
- Recompiled the paper with:
  - `latexmk -pdf -bibtex -outdir=build main.tex`
- Confirmed the output PDF builds successfully:
  - `paper/build/main.pdf`

### Remaining Typesetting Notes
- Split the longest detectability equations across multiple lines to avoid obvious overfull boxes.
- Some non-blocking LaTeX warnings remain:
  - standard MNRAS font substitution warnings
  - duplicate destination warnings from float ordering
  - minor underfull paragraph warnings
- None of these prevent the PDF from building or affect the scientific content.
  - observed Baumgardt clusters overplotted as black points

### Interpretation
- The map shows the expected trend:
  - at small `a`, only very massive clusters survive
  - toward large `a`, the survival threshold moves to lower `M_ini`
  - the transition from destroyed to surviving is broad because the map averages over the orbit distribution at fixed `a`
- This plot is therefore a direct visualization of the selection function that connects the intrinsic IMF to the observed surviving GC population.

### Verification
- Re-ran:
  - `.venv/bin/python -m compileall src scripts`
  - `.venv/bin/python scripts/fit_gc_imf_model.py`
- Confirmed successful creation of:
  - `outputs/figures/survivability_map_initial_mass_vs_radius.png`
  - `outputs/tables/survivability_map_lognormal.csv`

## 2026-05-21 08:33:35 BST

### User Request
- The user asked to implement the cleaner model formulation in which:
  - the survival function `S(M_ini, a)` is fixed first
  - the intrinsic IMF family and its parameters are then fitted jointly with the radial birth normalization
- The intended target is the observed point-process intensity in the plane of initial mass and semimajor axis.

### Implemented Statistical Model
- Implemented a proper joint Poisson point-process model in the plane:
  - `m = log10(M_ini / Msun)`
  - `x = log10(a / kpc)`
- The fitted observed intensity is:
  - `lambda_obs(m, x) = N0 * phi(m | theta) * A(x | eta) * S(m, x)`
- Here:
  - `phi(m | theta)` is the intrinsic IMF density per dex in initial mass
  - `A(x | eta)` is the birth-profile density per dex in semimajor axis
  - `S(m, x)` is fixed input
  - `N0` is the total initial number of clusters
- `phi` and `A` are each normalized to integrate to `1` over the fitted domain, so `N0` carries the total count scale.

### Likelihood
- Used the full Poisson point-process likelihood:
  - `ln L = sum_i ln lambda_obs(m_i, x_i) - integral integral lambda_obs(m, x) dm dx`
- Profiled out `N0` analytically, leaving joint optimization over only:
  - IMF parameters `theta`
  - radial-profile parameters `eta`
- This is the correct implementation of the user's desired model structure.

### Fixed Survival Function
- Added a new fixed-survival calibration that does not depend on the new IMF fit family.
- Defined the hard-cut calibration offset directly from the catalog:
  - `selection_offset_dex = min(log M_ini - log M_cut) - 0.001`
- This gives:
  - `selection_offset_dex = -0.38230922842646786`
- Built the fixed survival map numerically on a grid in `(m, x)` by smoothing in `log a` and averaging hard survival indicators at fixed `a`.
- This keeps `S(m, x)` fixed while allowing the new optimizer to vary only `phi` and `A`.

### New Code Added
- Added `src/globular_clusters_imf/joint_model.py`.
- Exported the new API from `src/globular_clusters_imf/__init__.py`.
- Extended `scripts/fit_gc_imf_model.py` to run the new joint fitter automatically after the earlier exploratory fit.
- Extended `README.md` with a short description of the new joint fixed-survival model.

### Implemented Model Families
- IMF families:
  - `lognormal`
  - `powerlaw`
  - `schechter`
- Radial birth-profile families:
  - `step5`
    - piecewise constant in `5` bins of `log a`
  - `logpoly3`
    - smooth cubic polynomial in standardized `log a`

### Outputs Written
- Model ranking table:
  - `outputs/tables/joint_fixed_survival_model_summary.csv`
- JSON summary:
  - `outputs/tables/joint_fixed_survival_model_summary.json`
- IMF grids for each fitted model:
  - `outputs/tables/joint_fixed_survival_imf_grids.csv`
- Radial birth-profile grids for each fitted model:
  - `outputs/tables/joint_fixed_survival_radial_grids.csv`
- Per-cluster predictions under the best model:
  - `outputs/tables/joint_fixed_survival_catalog_predictions.csv`
- Best-model figures:
  - `outputs/figures/joint_fixed_survival_best_model_profiles.png`
  - `outputs/figures/joint_fixed_survival_best_observed_intensity.png`

### Best-Fit Model Ranking
- The current best joint model is:
  - IMF family: `schechter`
  - radial model: `logpoly3`
- Best-fit values:
  - `alpha_dndm = -0.8716`
  - `log10(M_c / Msun) = 6.3133`
  - total initial cluster count `N0 = 548.3`
  - mean survival fraction over the fitted domain `= 0.3009`
- The second-best model is:
  - `lognormal + logpoly3`
- Both smooth radial-profile models outperform the corresponding `step5` piecewise models in the current comparison.

### Interpretation
- The new code now does exactly what the user requested in structure:
  - fix `S`
  - fit `phi`
  - fit `A(a)`
  - compare model families directly within a single joint likelihood
- The resulting best-fit observed-intensity map is a direct prediction for where GCs should appear in the `(log M_ini, log a)` plane.

### Verification
- Re-ran:
  - `.venv/bin/python -m compileall src scripts`
  - `.venv/bin/python scripts/fit_gc_imf_model.py`
- Verified that all `6` joint model combinations converged successfully and wrote outputs.

## 2026-05-21 08:39:49 BST

### User Request
- The user asked for plots showing best-model performance and triangle plots for the best model.

### Uncertainty Approximation Added
- The joint fitter previously returned only maximum-likelihood solutions.
- Added a local asymptotic uncertainty approximation around the best fit by numerically differentiating the profiled negative log-likelihood and inverting the resulting Hessian.
- Important interpretation:
  - this is a Gaussian approximation around the MLE
  - it is not a full posterior sample from a Bayesian run
- Added helper functions in `src/globular_clusters_imf/joint_model.py` for:
  - numerical Hessian estimation
  - covariance estimation
  - drawing asymptotic parameter samples
  - transforming raw fitted parameters into human-readable triangle-plot parameters

### New Best-Model Diagnostics
- Added a best-model performance figure:
  - `outputs/figures/joint_fixed_survival_best_model_performance.png`
- This figure shows:
  - observed versus expected `log M_ini` projection
  - observed versus expected radial projection
  - 2D residual-significance map in `(log M_ini, log a)`
  - observed versus expected counts in coarse 2D bins
- The performance plot is intended to make clear where the best joint model fits well and where systematic residuals remain.

### Triangle Plot
- Added a best-model triangle plot:
  - `outputs/figures/joint_fixed_survival_best_model_triangle.png`
- Added the corresponding sample table:
  - `outputs/tables/joint_fixed_survival_best_model_triangle_samples.csv`
- Added the raw covariance matrix:
  - `outputs/tables/joint_fixed_survival_best_model_parameter_covariance.csv`

### Best-Model Triangle Parameters
- For the current best model `schechter + logpoly3`, the triangle plot shows:
  - `alpha_dndm`
  - `log10(M_c / Msun)`
  - `beta1`
  - `beta2`
  - `beta3`
  - `log10(N0)`
- The derived survival fraction is also stored in the sample table but is not plotted in the triangle figure to keep the figure readable.

### Implementation Details
- Extended `fit_fixed_survival_joint_models()` so it now returns:
  - asymptotic parameter samples for the best model
  - covariance information
  - extra tables used by the new plotting code
- Extended `src/globular_clusters_imf/plotting.py` with:
  - `plot_joint_model_performance()`
  - `plot_best_model_triangle()`
  - supporting helpers for rebinning expected counts and relabeling parameters

### Result
- The new diagnostics are scientifically useful:
  - the performance figure shows the best model captures the broad location of the observed GC population in the `(log M_ini, log a)` plane, while still leaving structured residuals in some bins
  - the triangle plot makes the main parameter degeneracies visible, especially between the Schechter slope and cutoff scale, and between the smooth radial-profile coefficients

### Verification
- Re-ran:
  - `.venv/bin/python -m compileall src scripts`
  - `.venv/bin/python scripts/fit_gc_imf_model.py`
- Confirmed successful creation of:
  - `outputs/figures/joint_fixed_survival_best_model_performance.png`
  - `outputs/figures/joint_fixed_survival_best_model_triangle.png`
  - `outputs/tables/joint_fixed_survival_best_model_triangle_samples.csv`
  - `outputs/tables/joint_fixed_survival_best_model_parameter_covariance.csv`

## 2026-05-21 08:45:46 BST

### User Follow-Up
- The user pointed out that the diagnostics still lacked a direct plot comparing the performance of the different fitted models.

### Cross-Model Performance Comparison Added
- Added an explicit comparison figure:
  - `outputs/figures/joint_fixed_survival_model_comparison.png`
- This figure contains:
  - a horizontal bar chart of `Delta BIC` relative to the best model
  - the RMS residual significance for each model
  - a `2 x 3` grid of residual-significance maps in `(log M_ini, log a)`, one panel per fitted model

### New Performance Tables
- Added:
  - `outputs/tables/joint_fixed_survival_model_performance_summary.csv`
  - `outputs/tables/joint_fixed_survival_model_performance_residual_maps.csv`
- The summary table now records for each model:
  - `log_likelihood`
  - `AIC`
  - `BIC`
  - `Delta BIC`
  - `Pearson chi2` in coarse 2D bins
  - RMS residual significance
  - mean absolute residual significance
  - coarse-binned Poisson deviance

### Interpretation
- The comparison figure now makes the model hierarchy visually obvious:
  - `schechter + logpoly3` remains the best model
  - `lognormal + logpoly3` is the nearest competitor
  - the power-law models perform substantially worse, both in `Delta BIC` and in the residual maps
- The residual panels are useful because they show not just which model wins numerically, but also where each model systematically underpredicts or overpredicts the observed GC distribution.

### Implementation
- Extended `src/globular_clusters_imf/joint_model.py` to compute and save per-model coarse-binned performance diagnostics.
- Extended `src/globular_clusters_imf/plotting.py` with `plot_joint_model_comparison()`.
- Used a common binning in `(log M_ini, log a)` for all models so the residual maps are directly comparable.

### Verification
- Re-ran:
  - `.venv/bin/python -m compileall src scripts`
  - `.venv/bin/python scripts/fit_gc_imf_model.py`
- Confirmed successful creation of the new comparison figure and both new performance tables.

## 2026-05-21: Local In-Situ/Accreted GC Flag Saved Into Project

### Goal
- Save the local Milky Way GC in-situ/accreted classification directly into this repository so the project no longer depends on ad hoc lookups elsewhere in `~/Work/Code/`.

### Implementation
- Extended `src/globular_clusters_imf/catalog.py` with reusable helpers to:
  - read the local FITS GC catalogs
  - export a compact local origin table
  - join those flags onto the project Baumgardt catalog
- Added `scripts/export_gc_origin_flags.py` to regenerate the local products.
- Added `astropy` to `pyproject.toml` because the local source catalogs are FITS tables.
- Updated `README.md` with the new export step and the source-path environment variables:
  - `GC_IMF_UPDATED_CATALOG_PATH`
  - `GC_IMF_PINSITU_CATALOG_PATH`

### Source Catalogs Used
- Binary origin flag source:
  - `/Users/vasilybelokurov/data/catalogues/gc_catalog_updated.fits`
- Legacy probability source:
  - `/Users/vasilybelokurov/Documents/Work/lists/gc_catalog_pinsitu.fits`

### Matching Logic
- Built a stable GC name key to handle Baumgardt alias variants such as:
  - `Ter 2` -> `Terzan 2`
  - `Djor 2 ESO 456-SC38` -> `Djorg 2`
  - `Gran 3 Patchick 125` -> `Gran 3`
  - `BH 261 AL 3` -> `BH 261`
- This removed the initial unmatched cases and gave complete coverage of the Baumgardt catalog.

### New Local Files
- Saved compact local origin table:
  - `data/processed/gc_origin_flags.csv`
- Saved its summary:
  - `data/processed/gc_origin_flags_summary.csv`
- Saved Baumgardt catalog augmented with local origin information:
  - `data/processed/baumgardt_gc_catalog_with_origin_flags.csv`
- Saved its summary:
  - `data/processed/baumgardt_gc_catalog_with_origin_flags_summary.csv`

### Contents
- `gc_origin_flags.csv` now includes:
  - `origin_flag`
  - `origin_label`
  - `is_in_situ`
  - `is_accreted`
  - `progenitor_group`
  - `legacy_origin_flag`
  - `legacy_insitu_prob`
- `baumgardt_gc_catalog_with_origin_flags.csv` appends those origin columns to the cleaned Baumgardt catalog already used in this project.

### Current Counts
- Local origin table:
  - `165` clusters total
  - `107` in-situ
  - `58` accreted
  - `141` with legacy `INSITU_PROB`
- Baumgardt join:
  - `165 / 165` origin matches
  - `0` unmatched clusters

### Verification
- Installed/updated project environment with:
  - `.venv/bin/pip install astropy`
  - `.venv/bin/pip install -e .`
- Re-ran:
  - `.venv/bin/python -m compileall src scripts`
  - `.venv/bin/python scripts/export_gc_origin_flags.py`
- Confirmed that the saved Baumgardt catalog now has complete local in-situ/accreted coverage.

## 2026-05-21: Local GC Catalog Provenance and In-Situ/Accreted Flag

### Goal
- Searched the wider `~/Work/Code/` tree to identify the local Milky Way GC catalog carrying the in-situ/accreted classification used in older and newer analyses.

### Findings
- Found an older explicit probability catalog:
  - `/Users/vasilybelokurov/Documents/Work/lists/gc_catalog_pinsitu.fits`
- This file is referenced in:
  - `~/Work/Code/nrich_apogee/idl_code/first_figures.pro`
  - `~/Work/Code/nrich_apogee/test_gc_removal.py`
- Its schema includes:
  - `NAME`, `RA`, `DEC`, `MASS`, `MINI`, `FEH`, `FLAG`, `INSITU_PROB`
- It contains `142` clusters.

- Found the current working GC catalog used by more recent notebooks:
  - `/Users/vasilybelokurov/data/catalogues/gc_catalog_full.fits`
  - `/Users/vasilybelokurov/data/catalogues/gc_catalog_updated.fits`
- These are used in:
  - `~/Work/Code/globular_clusters_mw/globular_mass_metallicity.ipynb`
  - `~/Work/Code/globular_clusters_mw/gc_concentration_metallicity.ipynb`
  - several `~/Work/Code/uma3/` notebooks/scripts
- `gc_catalog_updated.fits` is derived from `gc_catalog_full.fits` by adding:
  - `avg_helium_diff`
  - `max_helium_diff`
- This provenance is recorded in:
  - `~/Work/Code/milone2005_helium_table4_read.ipynb`

### Flag Meaning
- Confirmed by matching the old `INSITU_PROB` catalog onto `gc_catalog_updated.fits`:
  - `FLAG = 1` means in-situ
  - `FLAG = 0` means accreted/ex-situ
- Evidence:
  - matched `141` clusters by name
  - `FLAG=1` objects have mean `INSITU_PROB = 0.977`
  - `FLAG=0` objects have mean `INSITU_PROB = 0.120`
- Current catalog counts in `gc_catalog_updated.fits`:
  - `107` in-situ (`FLAG=1`)
  - `58` accreted (`FLAG=0`)

### Related Metadata
- The newer catalog also carries `PROG_MA`, which gives progenitor/group labels such as:
  - `M-B`, `M-D`, `L-E`, `G-E`, `Sag`, `H99`, `Seq`
- Accreted clusters (`FLAG=0`) are concentrated in progenitor-labelled groups like `G-E`, `H-E`, and `Sag`.

### Practical Conclusion
- For current work, the relevant local GC catalog with the in-situ/accreted flag is:
  - `/Users/vasilybelokurov/data/catalogues/gc_catalog_updated.fits`
- If the explicit probability rather than a binary label is needed, use:
  - `/Users/vasilybelokurov/Documents/Work/lists/gc_catalog_pinsitu.fits`

## 2026-05-21: Two-Component In-Situ/Accreted Joint Model

### Goal
- Implement a two-component fixed-survival model in which:
  - in-situ and accreted GCs are treated as separate labeled populations
  - each population has its own IMF shape
  - each population has its own radial birth profile `A(a)`
  - the existing single-component model remains unchanged

### Implementation Strategy
- Added a new module:
  - `src/globular_clusters_imf/two_component_model.py`
- The new fitter uses the saved local `origin_flag`:
  - `1` -> in-situ
  - `0` -> accreted
- The survival map `S(M_ini, a)` is still fixed globally from the full catalog.
- Because the class labels are fixed and the two components share no IMF or radial parameters, the joint likelihood factorizes into:
  - an in-situ fixed-survival fit
  - an accreted fixed-survival fit
- The code therefore reuses the existing single-component joint fitter on each subset and then combines the results into a two-component model-ranking table.

### Code Changes
- Added:
  - `src/globular_clusters_imf/two_component_model.py`
- Updated:
  - `scripts/fit_gc_imf_model.py`
  - `src/globular_clusters_imf/plotting.py`
  - `src/globular_clusters_imf/__init__.py`
  - `README.md`

### New Outputs
- Component model-ranking table:
  - `outputs/tables/joint_fixed_survival_two_component_component_model_summary.csv`
- Joint in-situ/accreted pair ranking:
  - `outputs/tables/joint_fixed_survival_two_component_model_summary.csv`
- Best component summaries:
  - `outputs/tables/joint_fixed_survival_two_component_best_component_summary.csv`
- Best component IMF grids:
  - `outputs/tables/joint_fixed_survival_two_component_best_imf_grids.csv`
- Best component radial grids:
  - `outputs/tables/joint_fixed_survival_two_component_best_radial_grids.csv`
- Best component catalog predictions:
  - `outputs/tables/joint_fixed_survival_two_component_catalog_predictions.csv`
- JSON summary:
  - `outputs/tables/joint_fixed_survival_two_component_model_summary.json`
- Figure:
  - `outputs/figures/joint_fixed_survival_two_component_best_profiles.png`

### Main Results
- Both the in-situ and accreted subsets prefer:
  - `schechter + logpoly3`
- Best joint two-component model by BIC:
  - in-situ: `schechter + logpoly3`
  - accreted: `schechter + logpoly3`

### Best-Fit In-Situ Component
- Observed clusters:
  - `107`
- Inferred initial count:
  - `503.5`
- Survival fraction:
  - `0.213`
- Best IMF parameters:
  - `alpha_dndm = -0.940`
  - `log10(M_c/Msun) = 6.381`
- Best radial profile:
  - strongly concentrated to small semimajor axis
  - radial birth-profile peak at `a ≈ 0.47 kpc`
  - cumulative median birth radius `a ≈ 1.64 kpc`
  - `90%` of the integrated birth profile lies inside `a ≈ 4.67 kpc`

### Best-Fit Accreted Component
- Observed clusters:
  - `58`
- Inferred initial count:
  - `90.3`
- Survival fraction:
  - `0.642`
- Best IMF parameters:
  - `alpha_dndm = -0.751`
  - `log10(M_c/Msun) = 6.099`
- Best radial profile:
  - much broader and shifted to larger semimajor axis than the in-situ population
  - radial birth-profile peak at `a ≈ 11.8 kpc`
  - cumulative median birth radius `a ≈ 14.4 kpc`
  - `90%` of the integrated birth profile lies inside `a ≈ 75.4 kpc`

### Interpretation
- The two-component model sharpens the physical picture:
  - the in-situ population is far more centrally concentrated and has a much lower net survival fraction
  - the accreted population is born and retained at substantially larger radii, so its survival fraction is much higher
- The in-situ IMF is somewhat more top-heavy at the massive end in the Schechter sense because its cutoff mass is higher:
  - `log10 M_c ≈ 6.38` vs `6.10` for accreted
- The total inferred initial population in the best two-component fit is:
  - `593.8` clusters
  - of which about `85%` are in-situ and `15%` accreted

### Verification
- Re-ran:
  - `.venv/bin/python -m compileall src scripts`
  - `.venv/bin/python scripts/fit_gc_imf_model.py`
- Confirmed successful creation of the new two-component tables and the profile figure.

## 2026-05-21: Single vs Shared-IMF Two-Component vs Separate-IMF Two-Component

### Goal
- Test the more relevant hierarchy of hypotheses:
  - single population
  - two labeled populations with a shared IMF but separate radial birth profiles
  - two labeled populations with separate IMFs and separate radial birth profiles

### Implementation
- Extended `src/globular_clusters_imf/two_component_model.py` with:
  - a constrained shared-IMF two-component fitter
  - a model-class comparison writer
- Updated:
  - `scripts/fit_gc_imf_model.py`
  - `src/globular_clusters_imf/__init__.py`
  - `README.md`

### New Outputs
- Shared-IMF two-component model summary:
  - `outputs/tables/joint_fixed_survival_shared_imf_two_component_model_summary.csv`
- Shared-IMF best-component summary:
  - `outputs/tables/joint_fixed_survival_shared_imf_two_component_best_component_summary.csv`
- Shared-IMF best grids:
  - `outputs/tables/joint_fixed_survival_shared_imf_two_component_best_imf_grids.csv`
  - `outputs/tables/joint_fixed_survival_shared_imf_two_component_best_radial_grids.csv`
- Shared-IMF best catalog predictions:
  - `outputs/tables/joint_fixed_survival_shared_imf_two_component_catalog_predictions.csv`
- Model-class comparison:
  - `outputs/tables/joint_fixed_survival_population_model_class_comparison.csv`
  - `outputs/tables/joint_fixed_survival_population_model_class_comparison.json`

### Results
- Best single-population model:
  - `schechter + logpoly3`
  - `logL = 469.021`
  - `AIC = -928.042`
  - `BIC = -912.512`

- Best shared-IMF two-component model:
  - shared `schechter` IMF
  - in-situ radial model: `logpoly3`
  - accreted radial model: `logpoly3`
  - `logL = 449.309`
  - `AIC = -882.618`
  - `BIC = -857.770`

- Best separate-IMF two-component model:
  - in-situ: `schechter + logpoly3`
  - accreted: `schechter + logpoly3`
  - `logL = 450.511`
  - `AIC = -881.023`
  - `BIC = -849.963`

### Interpretation
- The model-class ranking is:
  1. single population
  2. shared-IMF two-component
  3. separate-IMF two-component
- Quantitatively:
  - shared-IMF two-component is worse than single-population by:
    - `Delta BIC = 54.74`
    - `Delta AIC = 45.42`
  - separate-IMF two-component is worse than single-population by:
    - `Delta BIC = 62.55`
    - `Delta AIC = 47.02`
  - among the two-component models, the shared-IMF model is preferred over separate IMFs by:
    - `Delta BIC = 7.81`
    - `Delta AIC = 1.59`
- Therefore:
  - the current data do **not** require any two-component split at all in this likelihood framework
  - if one insists on a two-component in-situ/accreted description, the data prefer:
    - different radial birth profiles
    - but a **shared** IMF shape

### Shared-IMF Best-Fit Behavior
- Shared IMF parameters:
  - `alpha_dndm = -0.878`
  - `log10(M_c/Msun) = 6.314`
- Inferred initial counts:
  - in-situ: `473.6`
  - accreted: `94.5`
- Survival fractions:
  - in-situ: `0.226`
  - accreted: `0.613`
- The radial split remains the dominant effect:
  - in-situ stays centrally concentrated
  - accreted stays much more extended
- Allowing the IMF to differ between the two subsets only improves `logL` by about `1.20`, which is too small to justify the extra parameters.

### Verification
- Re-ran:
  - `.venv/bin/python -m compileall src scripts`
  - `.venv/bin/python scripts/fit_gc_imf_model.py`
- Confirmed successful creation of all shared-IMF and model-class comparison tables.

## 2026-05-21: Integrated Total Initial Counts Across All Models

### Goal
- Make the integrated initial number of clusters explicit for every fitted model, since this is the radius- and mass-integrated normalization of the birth distribution, not the integral of the normalized IMF shape alone.

### Implementation
- Added:
  - `scripts/summarize_total_initial_counts.py`
- The script reads:
  - `outputs/tables/model_summary.json`
  - `outputs/tables/joint_fixed_survival_model_summary.csv`
  - `outputs/tables/joint_fixed_survival_shared_imf_two_component_model_summary.csv`
  - `outputs/tables/joint_fixed_survival_two_component_model_summary.csv`
- It writes:
  - `outputs/tables/total_initial_cluster_counts_by_model.csv`

### Result
- Wrote a flat summary table with `56` rows covering:
  - baseline truncated-IMF models
  - single-population fixed-survival joint models
  - shared-IMF two-component models
  - separate-IMF two-component models

### Representative Totals
- Baseline truncated models:
  - truncated lognormal: `344.1`
  - truncated power law: `15036.1`
- Best single-population joint model:
  - `schechter + logpoly3`: `548.3`
- Best shared-IMF two-component model:
  - shared `schechter`, both `logpoly3`: `568.1`
- Best separate-IMF two-component model:
  - in-situ `schechter + logpoly3`, accreted `schechter + logpoly3`: `593.8`

### Verification
- Re-ran:
  - `.venv/bin/python -m compileall src scripts`
  - `.venv/bin/python scripts/summarize_total_initial_counts.py`
- Confirmed successful creation of `total_initial_cluster_counts_by_model.csv`.

## 2026-05-21: Local MNRAS Paper Draft And Corrected Two-Component Comparison

### User Request
- Move the project into paper-writing mode.
- Build a local MNRAS manuscript project under its own `paper/` directory.
- Leave the introduction as a placeholder for now.
- Use the current fitted results, figures, and the corrected interpretation of the one- versus two-component model comparison.

### Writing Workflow
- Explicitly switched into manuscript-drafting mode using the `writing-style` skill.
- Kept all manuscript assets isolated under:
  - `paper/`
  - `paper/figures/`
  - `paper/tables/`

### New Reusable Code
- Added:
  - `src/globular_clusters_imf/paper_assets.py`
  - `scripts/build_paper_assets.py`
- The new paper asset builder:
  - reloads the processed Baumgardt catalogue
  - reruns the fixed-survival single-component model
  - reruns the shared-IMF and separate-IMF two-component models
  - writes publication-style figures and table fragments directly into `paper/`

### Corrected Statistical Comparison
- Revisited the comparison between:
  - the single-population model
  - the shared-IMF two-component model
  - the separate-IMF two-component model
- The raw class-comparison table in `outputs/tables/` is not the right one for the paper because the two-component likelihood is evaluated on two fixed labelled subsets.
- Computed the count-partition constant:
  - `C = N ln N - N_in ln N_in - N_acc ln N_acc = 106.9826`
  - with `N = 165`, `N_in = 107`, `N_acc = 58`
- Built the corrected conditional comparison in:
  - `paper/tables/population_model_class_comparison.csv`
  - `paper/tables/population_model_class_comparison.tex`

### Corrected Interpretation
- Conditional on the adopted in-situ/accreted labels:
  - the shared-IMF two-component model is strongly preferred over the single-population model
  - the single-population model is worse by:
    - `Delta BIC_cond = 159.22`
- Within the two-component family:
  - the separate-IMF model is worse than the shared-IMF model by:
    - `Delta BIC_cond = 7.81`
- Therefore:
  - the data support a two-component split in radial structure
  - the data do not support distinct IMF shapes for the two populations

### New Paper Figures
- Wrote:
  - `paper/figures/survivability_plane.pdf`
  - `paper/figures/single_component_model_performance.pdf`
  - `paper/figures/best_single_component_summary.pdf`
  - `paper/figures/two_component_results.pdf`
- The new best-fit summary figure now includes:
  - Poisson error bars on the observed mass and radius projections
  - a smooth model median curve
  - a local Gaussian uncertainty band from the asymptotic parameter covariance

### New Paper Tables
- Wrote:
  - `paper/tables/single_component_model_comparison.tex`
  - `paper/tables/population_model_class_comparison.tex`
  - `paper/tables/key_results_summary.tex`
  - plus matching CSV files and:
  - `paper/tables/paper_results_summary.json`

### Manuscript Project
- Added:
  - `paper/main.tex`
  - `paper/references.bib`
  - `paper/Makefile`
- The manuscript currently contains:
  - abstract
  - placeholder introduction
  - data section
  - survivability-plane section
  - model description for the single- and two-component cases
  - single-component results
  - two-component results with the corrected `Delta BIC`
  - lower-bound discussion for the total initial GC population
  - short conclusions

### Current Science Summary In Draft Form
- Best single-component model:
  - `schechter + logpoly3`
  - `alpha = -0.872`
  - `log10(M_c/Msun) = 6.313`
  - `N0 = 548.3`
- Preferred shared-IMF two-component model:
  - shared `schechter` IMF
  - in-situ `logpoly3` radial profile
  - accreted `logpoly3` radial profile
  - `N0,in_situ = 473.6`
  - `N0,accreted = 94.5`
  - `N0,total = 568.1`
- Best separate-IMF model changes `logL` only slightly and is not preferred by BIC.
- The draft therefore interprets `N0 ~ 5.5e2 - 5.7e2` as a lower bound on the initial number of GCs tied to the surviving MW GC family.

### Verification
- Re-ran:
  - `.venv/bin/python -m compileall src scripts`
  - `.venv/bin/python scripts/build_paper_assets.py`
  - `cd paper && latexmk -pdf -bibtex -outdir=build main.tex`
- Confirmed successful compilation of:
  - `paper/build/main.pdf`

## 2026-05-21: Initial stellar-mass lower bounds added to manuscript

### Goal
- Compute the total initial stellar mass in GCs implied by the best single-component and two-component models and report it consistently in the paper.

### Implementation
- Added reusable stellar-mass integrals to `src/globular_clusters_imf/paper_assets.py`:
  - integrate `M * phi(log M)` over the fitted IMF grid
  - multiply by the inferred `N0` for the relevant model or component
- Extended the paper asset builder to write:
  - `paper/tables/key_results_summary.csv`
  - `paper/tables/key_results_summary.tex`
  - `paper/tables/paper_numbers.tex`
- Updated `paper/main.tex` to read the generated macros and report the stellar-mass lower bounds in:
  - the abstract
  - the single-component results section
  - the two-component results section
  - the lower-bound section
  - the conclusions
- Updated `scripts/build_paper_assets.py` to print both cluster counts and stellar masses for the preferred single-component and shared-IMF two-component solutions.

### Numbers now in the paper
- Best single-component model:
  - `N0 = 548.3`
  - `M_star,0 = 2.694e8 Msun`
- Preferred shared-IMF two-component model:
  - `N0 = 568.1`
  - `M_star,0 = 2.753e8 Msun`
  - split as:
    - in-situ: `2.295e8 Msun`
    - accreted: `4.582e7 Msun`

### Verification
- Re-ran:
  - `.venv/bin/python -m compileall src scripts`
  - `.venv/bin/python scripts/build_paper_assets.py`
  - `cd paper && latexmk -pdf -bibtex -outdir=build main.tex`
- Confirmed:
  - `paper/tables/paper_numbers.tex` contains the generated manuscript macros
  - `paper/tables/key_results_summary.tex` includes the new `$M_{\star,0}$` column
  - `paper/build/main.pdf` is up to date

## 2026-05-21: Literature review subsection added to Discussion

### Goal
- Carry out a literature review of previous constraints on the initial GC mass function and add a discussion subsection comparing the present results to earlier work.

### Literature strands included
- Dynamical-evolution arguments favouring bell-shaped or quasi-equilibrium GCIMFs:
  - `Vesperini (1998)`
  - `Parmentier & Gilmore (2005, 2007)`
- Evolved-Schechter / young-cluster-like starting points:
  - `Fall & Zhang (2001)`
  - `Jordán et al. (2007)`
  - `Kruijssen (2015)`
  - `Hughes et al. (2022)`
- Milky Way reconstructions and halo-budget arguments:
  - `Bonatto & Bica (2012)`
  - `Webb & Leigh (2015)`
  - `Baumgardt et al. (2019)`
  - `Schaerer & Charbonnel (2011)`

### Main comparison now in the manuscript
- Our preferred Milky Way fit is formally Schechter-like but effectively much closer, over the mass range actually constrained by the surviving sample, to a bell-shaped or low-mass-depleted GCIMF than to a pure `dN/dM ~ M^-2` young-cluster power law.
- The fitted cutoff `log10(M_c/Msun) ~ 6.31` is consistent with the few-`10^6 Msun` mass scales discussed in several earlier formation-based and evolved-Schechter interpretations.
- The preferred shared-IMF two-component normalization,
  - `N0 = 568.1`
  - `M_star,0 = 2.75e8 Msun`
  is very close to Baumgardt et al. (2019), and broadly consistent with Bonatto & Bica (2012) and Webb & Leigh (2015).
- Larger totals in multiple-population or halo-budget arguments are now explicitly interpreted as referring to a different quantity:
  - the total mass in all proto-GCs including a fully destroyed population,
  - which our present likelihood does not constrain.

### Files updated
- `paper/main.tex`
- `paper/references.bib`

### Verification
- Re-ran:
  - `cd paper && latexmk -pdf -bibtex -outdir=build main.tex`
- Confirmed:
  - new `Discussion` section present
  - literature comparison subsection compiles with citations
  - `paper/build/main.pdf` updated successfully

## 2026-05-21: Clearer conditional-BIC explanation in two-component section

### Goal
- Rewrite the explanation of the BIC correction in the two-component results section so the statistical logic is explicit and easier to follow.

### Changes made
- Replaced the compact original paragraph in `paper/main.tex` with a clearer step-by-step explanation:
  - the single-component model is an unlabeled pooled point process
  - the two-component model is conditional on the observed in-situ/accreted labels
  - this introduces a parameter-independent count-partition offset
  - the offset matters only for `1-component` versus `2-component` comparisons, not within the two-component family
  - the conditional comparison is defined through
    - `log L_cond = log L_raw + C`
    - `BIC_cond = BIC_raw - 2C`
- Added explicit language that the correction is not rewarding extra parameters; it only removes a bookkeeping offset tied to conditioning on the observed labels.

### Verification
- Re-ran:
  - `cd paper && latexmk -pdf -bibtex -outdir=build main.tex`
- Confirmed:
  - `paper/build/main.pdf` rebuilt successfully
  - the new explanation appears in the `Two-component results` section

## 2026-05-21: Expanded derivation of the conditional-likelihood offset

### Goal
- Make the derivation of equation 6 explicit and explain the meaning of `C` clearly in the manuscript text.

### Changes made
- Rewrote the beginning of the `Two-component results` section in `paper/main.tex` to show:
  - the pooled one-component Poisson count term, `Lambda^N exp(-Lambda) / N!`
  - the labelled two-component count term,
    `Lambda_in^{N_in} exp(-Lambda_in) / N_in!` times
    `Lambda_acc^{N_acc} exp(-Lambda_acc) / N_acc!`
  - the missing combinatorial factor `N! / (N_in! N_acc!)`
  - the definition of
    `C = ln[N!/(N_in! N_acc!)]`
    and its Stirling approximation used in the implementation
- Added explicit wording that `C`:
  - is not a fitted parameter
  - is not a physical property of the GC system
  - is the log of the number of distinct ways to realise the observed in-situ/accreted partition
- Added one sentence explaining equation 6 directly:
  - it simply adds back the missing label-partition constant before comparing 2-component and 1-component model classes

### Verification
- Re-ran:
  - `cd paper && latexmk -pdf -bibtex -outdir=build main.tex`
- Confirmed:
  - `paper/build/main.pdf` rebuilt successfully
  - the revised derivation and interpretation of `C` appear in the manuscript

## 2026-05-21: Explicit likelihood added to Section 4

### Goal
- Write down the likelihood explicitly in the model section so the later discussion of the partition constant `C` follows directly from the stated statistical formulation.

### Changes made
- Added to `paper/main.tex` at the end of Section 4:
  - the pooled single-component raw Poisson point-process log-likelihood
  - the labelled two-component raw Poisson point-process log-likelihood
  - explicit statements of the omitted additive constants:
    - `-ln N!` for the pooled model
    - `-ln N_in! - ln N_acc!` for the labelled model
- Added one sentence linking those omitted factorial terms directly to the partition constant introduced in the next section.
- Re-formatted the new equations as multi-line `aligned` displays so they fit the manuscript cleanly.

### Verification
- Re-ran:
  - `cd paper && latexmk -pdf -bibtex -outdir=build main.tex`
- Confirmed:
  - `paper/build/main.pdf` rebuilt successfully
  - the new likelihood equations appear at the end of Section 4
  - the added equations no longer produce overfull-box warnings

## 2026-05-21: Full Poisson probability/likelihood written into Section 4

### Goal
- Replace the earlier partial likelihood insert with a full statement of the Poisson point-process probability and likelihood in Section 4, so the correction factor in the two-component section follows immediately from the formalism.

### Changes made
- Rewrote the end of Section 4 in `paper/main.tex` to state explicitly:
  - the single-component catalogue probability density
    `P(D | theta, eta, N0) = exp[-∫ lambda dx] / N! * product lambda(x_i)`
  - the corresponding single-component log-likelihood
  - the labelled two-component catalogue probability density
    `P(D_in, D_acc | model)`
  - the corresponding labelled two-component log-likelihood
- Explicitly identified the omitted constant terms:
  - `-ln N!` for the pooled catalogue
  - `-ln N_in! - ln N_acc!` for the labelled catalogue
- Rewrote the start of the `Two-component results` section so the constant
  `C = ln[N! / (N_in! N_acc!)]`
  is introduced directly as the difference between those factorial terms.
- Added equation labels for the single- and two-component likelihoods and for the conditional correction formulas.
- Cleaned the displayed equations so the final manuscript build has no `Overfull \hbox` or `undefined`-reference messages in the log.

### Verification
- Re-ran:
  - `cd paper && latexmk -pdf -bibtex -outdir=build main.tex`
  - `grep -n "Overfull \\hbox\\|undefined" paper/build/main.log`
- Confirmed:
  - `paper/build/main.pdf` rebuilt successfully
  - the grep check returns no remaining overfull-box or undefined-reference matches

## 2026-05-21: Detectability illustration versus $D_\odot$, $|b|$, and present mass

### Goal
- Illustrate that Milky Way GC detectability is not plausibly 100 per cent by mapping the surviving GC counts in the plane of heliocentric distance and absolute Galactic latitude for different present-day GC mass bins.

### Implementation
- Added reusable functions to `src/globular_clusters_imf/plotting.py`:
  - `build_gc_detectability_histogram_tables`
  - `plot_gc_detectability_histograms_by_present_mass`
- Added a wrapper script:
  - `scripts/plot_gc_detectability_histograms.py`
- The plot uses:
  - `D_\odot = r_sun_kpc`
  - `|b| = abs(galactic_b_deg)`
  - present-day mass `M_now = present_mass_msun`
- For the first-pass illustration I used 3 equal-count bins in `log10(M_now/Msun)` and a 2D histogram of GC counts in each panel.

### Outputs
- Figure:
  - `outputs/figures/gc_detectability_dsun_absb_by_present_mass.png`
- Tables:
  - `outputs/tables/gc_detectability_present_mass_bins.csv`
  - `outputs/tables/gc_detectability_histogram_table.csv`

### Present-day mass bins used
- `2.87 <= log10(M_now/Msun) < 4.82`, `N = 55`
- `4.82 <= log10(M_now/Msun) < 5.28`, `N = 54`
- `5.28 <= log10(M_now/Msun) <= 6.60`, `N = 56`

### Verification
- Re-ran:
  - `.venv/bin/python -m compileall src scripts`
  - `.venv/bin/python scripts/plot_gc_detectability_histograms.py`
- Confirmed:
  - figure writes successfully
  - histogram tables are saved for later reuse in the paper

## 2026-05-21: Paper restructuring for detectability-corrected single-component figures

### Goal
- Move the main single-component diagnostic figures into the detectability-corrected results section, so the manuscript shows only the corrected versions of the model-comparison and best-fit diagnostic plots.
- Add an explicit single-component profile figure showing the IMF and radial birth distribution, and keep the baseline perfect-detectability single-component fit only as a short reference point.

### Implementation
- Extended the detectability-corrected fitting workflow in `src/globular_clusters_imf/detectability_model.py` to run the EM-style completeness correction for all six single-component model families:
  - lognormal, power law, and Schechter IMFs
  - step5 and logpoly3 radial models
- Added summary helpers for the corrected single-component family comparison, including:
  - corrected log-likelihood
  - 2D residual RMS in the $(\log M_{\rm ini}, \log a)$ plane
  - mean detectability
  - total initial GC count
- Updated `src/globular_clusters_imf/paper_assets.py` so the paper assets now use:
  - detectability-corrected model-comparison figure for the single-component family ranking
  - detectability-corrected best-fit summary figure
  - a new intrinsic single-component profile figure comparing the baseline and corrected IMF and radial birth profile
- Reworked `paper/main.tex` so the section order is now:
  - baseline single-component model and short baseline results
  - detectability-corrected single-component model
  - detectability-corrected results with:
    - EM convergence/completeness figure
    - corrected single-component model-comparison table and figure
    - corrected best-fit summary figure
    - intrinsic single-component IMF/radial-profile figure
- Left the two-component analysis in the perfect-detectability baseline, but made that limitation explicit in the manuscript.

### New or updated paper outputs
- `paper/figures/detectability_em_results.pdf`
- `paper/figures/single_component_model_performance.pdf`
- `paper/figures/best_single_component_summary.pdf`
- `paper/figures/single_component_profiles.pdf`
- `paper/tables/single_component_model_comparison.csv`
- `paper/tables/single_component_model_comparison.tex`
- `paper/build/main.pdf`

### Main scientific outcome retained in the revised presentation
- The detectability-corrected single-component lower bound remains:
  - `N_0 = 784.6`
  - `M_{\star,0} = 3.13 \times 10^8\,{\rm M}_\odot`
- The preferred corrected single-component model remains Schechter plus logpoly3.
- The detectability correction changes the radial birth normalization more strongly than the IMF shape.

### Verification
- Re-ran:
  - `.venv/bin/python -m compileall src scripts`
  - `.venv/bin/python scripts/build_paper_assets.py`
  - `cd paper && latexmk -pdf -bibtex -outdir=build main.tex`
- Confirmed:
  - the corrected single-component figures are generated successfully
  - the manuscript compiles to `paper/build/main.pdf`
  - the main single-component figures now appear only in the detectability-corrected section

## 2026-05-21: Detectability-corrected two-component model and paper update

### Goal
- Extend the EM-style detectability correction from the single-component model to the in-situ/accreted two-component models.
- Refit both the shared-IMF and separate-IMF two-component model families with the same common completeness law.
- Update the manuscript so the two-component section and Discussion reflect the detectability-corrected results rather than the old perfect-detectability baseline.

### Implementation
- Generalized the shared-IMF two-component likelihood in `src/globular_clusters_imf/two_component_model.py` so it now uses the current selection grid and interpolator, not just the raw survivability grid. This makes the two-component code compatible with detectability-corrected contexts.
- Added reusable detectability-corrected two-component fitting functions in `src/globular_clusters_imf/detectability_model.py`:
  - `prepare_two_component_detectability_environment`
  - `fit_shared_imf_two_component_detectability_em_models`
  - `fit_shared_imf_two_component_detectability_em_single_model`
  - `fit_separate_imf_two_component_detectability_em_models`
  - `fit_separate_imf_two_component_detectability_em_single_model`
  - helpers for common completeness propagation and pooled observable prediction
- The two-component detectability correction uses the same logic as the single-component EM scheme:
  - build a pooled prediction for the complete surviving population in observable space
  - fit one common logistic completeness law `C(log M_now, log D_sun, |b|)`
  - map it back to a common intrinsic `Q(log M_ini, a)`
  - refit the intrinsic in-situ/accreted components
  - iterate to convergence
- Updated `src/globular_clusters_imf/paper_assets.py` so the paper now uses:
  - detectability-corrected shared-IMF two-component model comparisons
  - detectability-corrected separate-IMF two-component comparisons
  - detectability-corrected conditional BIC table
  - detectability-corrected two-component figure
  - updated manuscript macros and key-results table
- Rewrote the two-component section in `paper/main.tex` to:
  - include the detectability factor in the 2-component intensity
  - explain the pooled completeness fit in the two-component EM loop
  - report detectability-corrected conditional-BIC comparisons
  - update the preferred lower bound in the Discussion and Conclusions

### Main detectability-corrected two-component results
- Preferred shared-IMF two-component model:
  - shared Schechter IMF
  - in-situ `logpoly3`
  - accreted `logpoly3`
  - total initial count `N_0 = 807.1`
  - total initial stellar mass `M_{\star,0} = 3.19 x 10^8 Msun`
  - in-situ: `685.4` clusters, `2.71 x 10^8 Msun`, `f_sel = 0.156`
  - accreted: `121.6` clusters, `4.80 x 10^7 Msun`, `f_sel = 0.477`
  - shared IMF parameters: `alpha = -0.952`, `log10(M_c/Msun) = 6.310`
  - mean detectability of surviving population: `0.838`
- Best separate-IMF two-component model:
  - in-situ Schechter + `logpoly3`
  - accreted Schechter + `logpoly3`
  - total initial count `N_0 = 962.7`
  - total initial stellar mass `M_{\star,0} = 3.21 x 10^8 Msun`
- Corrected conditional model comparison:
  - shared-IMF two-component is preferred
  - separate-IMF is worse by `Delta BIC_cond = 7.65`
  - single-component is worse by `Delta BIC_cond = 158.80`

### Outputs
- Corrected shared-IMF two-component tables:
  - `outputs/tables/joint_fixed_survival_detectability_shared_imf_two_component_model_summary.csv`
  - `outputs/tables/joint_fixed_survival_detectability_shared_imf_two_component_best_component_summary.csv`
- Corrected separate-IMF two-component tables:
  - `outputs/tables/joint_fixed_survival_detectability_two_component_model_summary.csv`
  - `outputs/tables/joint_fixed_survival_detectability_two_component_best_component_summary.csv`
- Updated paper tables/macros:
  - `paper/tables/population_model_class_comparison.csv`
  - `paper/tables/key_results_summary.csv`
  - `paper/tables/paper_numbers.tex`
- Updated paper figure:
  - `paper/figures/two_component_results.pdf`
- Updated manuscript PDF:
  - `paper/build/main.pdf`

### Verification
- Re-ran:
  - `.venv/bin/python -m compileall src scripts`
  - `.venv/bin/python scripts/build_paper_assets.py`
  - `cd paper && latexmk -pdf -bibtex -outdir=build main.tex`
- Confirmed:
  - the corrected two-component scans complete successfully
  - the paper tables and macros are regenerated from the corrected fits
  - the manuscript now reports detectability-corrected two-component results throughout the relevant sections

## 2026-05-21: Split the detectability EM composite figure in the paper

### Goal
- Separate the two rows of the detectability EM composite into two distinct paper figures:
  - a three-panel detectability-map figure spanning two columns
  - a square EM-convergence figure that fits in a single column

### Implementation
- Replaced the single composite plotting function in `src/globular_clusters_imf/paper_assets.py` with:
  - `plot_detectability_em_maps_for_paper`
  - `plot_detectability_em_convergence_for_paper`
- The detectability maps are now written to:
  - `paper/figures/detectability_em_maps.pdf`
- The convergence plot is now written to:
  - `paper/figures/detectability_em_convergence.pdf`
- Updated `paper/main.tex` so the detectability-corrected results section now cites and displays the two figures separately with revised captions.

### Verification
- Re-ran:
  - `.venv/bin/python scripts/build_paper_assets.py`
  - `cd paper && latexmk -pdf -bibtex -outdir=build main.tex`
- Confirmed:
  - both split figure PDFs are generated
  - the manuscript recompiles successfully to `paper/build/main.pdf`

## 2026-05-21: Add mass-dependent uncertainty bands to Figure 8

### Goal
- Show explicitly that the single-component IMF and radial-profile constraints are not equally precise across mass and radius.
- Propagate the existing asymptotic parameter covariance into the paper's Figure 8 so the low-mass IMF uncertainty is visible rather than inferred indirectly.

### Implementation
- Updated `src/globular_clusters_imf/paper_assets.py` so `plot_single_component_profiles_for_paper` now accepts the best-model uncertainty payload and draws local Gaussian `1sigma` bands by sampling the fitted parameter covariance.
- Added a helper to propagate the parameter samples into:
  - the intrinsic IMF profile `dN / dlogM`
  - the intrinsic radial birth profile `N_0 A(log a)`
- Wired the same uncertainty payload through the main paper asset build and the restricted-sample `a < 100 kpc` Figure 8 runner.
- Added a reusable one-off builder:
  - `scripts/build_paper_figure8_single_component_profiles.py`
- Updated the Figure 8 caption in `paper/main.tex` to explain that the band is strongly mass-dependent and widens toward the weakly constrained low-mass end.

### Outputs
- Updated paper figure:
  - `paper/figures/single_component_profiles.pdf`
- Updated restricted-sample analogue:
  - `variants/a_lt_100kpc/outputs/figures/figure8_single_component_profiles_a_lt_100kpc.pdf`
  - `variants/a_lt_100kpc/outputs/figures/figure8_single_component_profiles_a_lt_100kpc.png`
- Updated manuscript PDF:
  - `paper/build/main.pdf`

### Verification
- Re-ran:
  - `.venv/bin/python scripts/build_paper_figure8_single_component_profiles.py`
  - `.venv/bin/python scripts/run_subset_a_lt_100kpc_single_component_profiles.py`
  - `cd paper && latexmk -pdf -bibtex -outdir=build main.tex`
- Confirmed:
  - Figure 8 now shows visible uncertainty bands in both panels
  - the IMF band is narrow near the well-constrained peak and broadens toward low masses
  - the paper and the restricted-sample variant both compile successfully

## 2026-05-21: Replace the Figure 8 IMF band with a profile-likelihood band

### Goal
- Remove the artificial pinch in the Figure 8 IMF band caused by propagating the local Schechter-parameter covariance directly into function space.
- Replace it with a mass-local uncertainty estimate that reflects the actual likelihood leverage at each mass.

### Implementation
- Added a constrained profile-likelihood solver in `src/globular_clusters_imf/joint_model.py` that:
  - evaluates the IMF density at a chosen `log M_ini`
  - re-optimizes all remaining model parameters at fixed local IMF amplitude
  - finds the lower and upper `1sigma` bounds from `Delta nll = 0.5`
- Updated `src/globular_clusters_imf/paper_assets.py` so Figure 8 panel (a) now:
  - computes the profile band on a set of support masses
  - interpolates that pointwise band across the plotting grid
  - keeps the fitted Schechter curve itself unchanged
- Left the radial-profile band in panel (b) on the previous local Gaussian approximation for now, since the immediate problem was the IMF pivot artifact.
- Updated the Figure 8 discussion and caption in `paper/main.tex` to explain the new construction explicitly.

### Outputs
- Updated figure:
  - `paper/figures/single_component_profiles.pdf`
- Updated manuscript PDF:
  - `paper/build/main.pdf`

### Verification
- Re-ran:
  - `.venv/bin/python -m compileall src scripts`
  - `.venv/bin/python scripts/build_paper_figure8_single_component_profiles.py`
  - `cd paper && latexmk -pdf -bibtex -outdir=build main.tex`
- Confirmed:
  - the artificial IMF pinch is removed
  - the low-mass IMF uncertainty is now visibly broader
  - the manuscript compiles successfully with the updated caption and text

## 2026-05-21: Implement a flexible IMF with bootstrap uncertainty and compare it to the Schechter profile band

### Goal
- Go beyond the Schechter-family uncertainty geometry by fitting a more flexible IMF shape and deriving its uncertainty directly from bootstrap refits.
- Compare the resulting flexible-IMF band to the existing Schechter profile-likelihood band.

### Implementation
- Extended the core IMF machinery in `src/globular_clusters_imf/joint_model.py` with a new family:
  - `logspline6`
  - six fixed mass knots across the fitted mass range
  - positive IMF obtained by PCHIP interpolation in log-density and subsequent normalization
- Added a weak second-difference smoothness penalty to the `logspline6` objective so the flexible family behaves as a smooth alternative rather than an unconstrained oscillatory spline.
- Added a reusable comparison module:
  - `src/globular_clusters_imf/flexible_imf.py`
  - fits the detectability-corrected single-component model for:
    - the existing `schechter + logpoly3` case
    - the new `logspline6 + logpoly3` case
  - computes the Schechter pointwise profile-likelihood band
  - computes a bootstrap band for the flexible IMF from repeated non-parametric catalog resamples
- Added a reusable driver script:
  - `scripts/compare_profile_and_flexible_imf.py`
- Kept all outputs in a separate variant tree:
  - `variants/flexible_imf_bootstrap_comparison/`

### Main comparison result
- Using `12` bootstrap resamples, all `12/12` flexible-IMF refits converged.
- Detectability-corrected Schechter model:
  - `logL = 471.47`
  - `AIC = -932.93`
  - `BIC = -917.40`
  - `N0 = 784.6`
  - `rms residual sigma = 1.010`
- Detectability-corrected flexible `logspline6` model:
  - `logL = 475.17`
  - `AIC = -934.34`
  - `BIC = -909.50`
  - `N0 = 847.6`
  - `rms residual sigma = 0.933`

### Interpretation
- The flexible IMF gives a modestly better raw fit than the Schechter model:
  - `Delta logL = +3.71`
  - `Delta AIC = -1.41` in favour of the flexible model
- But the extra IMF freedom is still penalized by `BIC`:
  - `Delta BIC = +7.91` against the flexible model
- The flexible-IMF bootstrap band is broader than the Schechter profile band at the low-mass end and remains less explosive at the extreme high-mass end.
- The main scientific effect is therefore:
  - the low-mass IMF is less rigid than the Schechter curve suggests
  - but the present catalog still does not require the extra freedom strongly enough to overcome the BIC penalty

### Outputs
- Comparison figure:
  - `variants/flexible_imf_bootstrap_comparison/outputs/figures/imf_profile_vs_logspline_bootstrap_comparison.pdf`
  - `variants/flexible_imf_bootstrap_comparison/outputs/figures/imf_profile_vs_logspline_bootstrap_comparison.png`
- Comparison tables:
  - `variants/flexible_imf_bootstrap_comparison/outputs/tables/imf_profile_vs_logspline_model_comparison.csv`
  - `variants/flexible_imf_bootstrap_comparison/outputs/tables/schechter_profile_likelihood_imf_band.csv`
  - `variants/flexible_imf_bootstrap_comparison/outputs/tables/logspline6_bootstrap_imf_band.csv`
  - `variants/flexible_imf_bootstrap_comparison/outputs/tables/logspline6_bootstrap_summary.csv`
  - `variants/flexible_imf_bootstrap_comparison/outputs/tables/imf_profile_vs_logspline_summary.json`

### Verification
- Re-ran:
  - `.venv/bin/python -m compileall src scripts`
  - `.venv/bin/python scripts/compare_profile_and_flexible_imf.py --n-bootstrap 12`
- Confirmed:
  - the new `logspline6` family converges on the full catalog
  - the bootstrap comparison products are written successfully
  - the smoothed flexible IMF behaves sensibly and removes the need to interpret the Schechter profile band as the only allowed shape family

## 2026-05-21: Fold the flexible-IMF cross-check into the paper and Figure 8

### Goal
- Describe the flexible IMF model explicitly in the manuscript rather than leaving it as an external diagnostic.
- Add the flexible-IMF curve directly to Figure 8 so the paper shows the Schechter baseline and the flexible cross-check in the same panel.

### Implementation
- Updated `paper/main.tex` to add a dedicated subsection:
  - `A flexible IMF cross-check`
- The paper now states explicitly:
  - the six-knot log-spline IMF representation
  - the fixed-knot, five-parameter normalization convention
  - the weak second-difference smoothness penalty
  - the bootstrap uncertainty procedure
- Added the flexible-model comparison result to the detectability-corrected single-component results section, including:
  - `Delta logL = +3.71`
  - `Delta BIC = +7.9`
  - the shift in `N_0` from `784.6` to `847.6`
- Updated the Figure 8 caption to identify the dashed green flexible-IMF curve explicitly.

### Figure update
- Modified `src/globular_clusters_imf/paper_assets.py` so Figure 8 can read a precomputed flexible-IMF overlay from:
  - `variants/flexible_imf_bootstrap_comparison/outputs/tables/imf_profile_vs_logspline_summary.json`
- The overlay is reconstructed from the saved knot positions and node amplitudes, so the paper build does not need to rerun the flexible bootstrap comparison.
- Updated the reusable Figure 8 builder:
  - `scripts/build_paper_figure8_single_component_profiles.py`

### Outputs
- Updated paper figure:
  - `paper/figures/single_component_profiles.pdf`
- Updated manuscript PDF:
  - `paper/build/main.pdf`

### Verification
- Re-ran:
  - `.venv/bin/python -m compileall src scripts`
  - `.venv/bin/python scripts/build_paper_figure8_single_component_profiles.py`
  - `cd paper && latexmk -pdf -bibtex -outdir=build main.tex`
- Confirmed:
  - Figure 8 now shows the dashed flexible-IMF cross-check curve
  - the manuscript compiles successfully with the new subsection and updated caption

## 2026-05-21: Add the flexible-IMF bootstrap band to Figure 8

### Goal
- Show the flexible-IMF uncertainty visually in Figure 8, not just its best-fitting curve.

### Implementation
- Updated `src/globular_clusters_imf/paper_assets.py` so the Figure 8 builder now reads the precomputed flexible-IMF bootstrap band from:
  - `variants/flexible_imf_bootstrap_comparison/outputs/tables/logspline6_bootstrap_imf_band.csv`
- The same panel now plots:
  - the Schechter profile-likelihood band in orange
  - the flexible-IMF bootstrap band in translucent green
  - the best flexible-IMF curve as a dashed green line
- Updated the Figure 8 text and caption in `paper/main.tex` so the manuscript explicitly identifies the green shaded region as the flexible bootstrap `1sigma` band.

### Outputs
- Updated paper figure:
  - `paper/figures/single_component_profiles.pdf`
- Updated manuscript PDF:
  - `paper/build/main.pdf`

### Verification
- Re-ran:
  - `.venv/bin/python -m compileall src scripts`
  - `.venv/bin/python scripts/build_paper_figure8_single_component_profiles.py`
  - `cd paper && latexmk -pdf -bibtex -outdir=build main.tex`
- Confirmed:
  - Figure 8 panel (a) now shows both the flexible-IMF band and the flexible-IMF line
  - the updated figure remains legible and the paper recompiles successfully

## 2026-05-21: Add a profile-likelihood radial uncertainty band to Figure 8

### Goal
- Replace the right-panel Gaussian radial uncertainty summary in Figure 8 with the same pointwise profile-likelihood construction already used for the IMF in the left panel.
- Use the same orange/yellow shading in both panels so the uncertainty encoding is consistent.

### Implementation
- Added radial profile-likelihood utilities to `src/globular_clusters_imf/joint_model.py`:
  - `evaluate_radial_birth_intensity_at_log_a`
  - `compute_profile_likelihood_radial_birth_band`
  - `find_profile_likelihood_radial_birth_bound`
  - `profile_nll_at_fixed_radial_birth_intensity`
- These functions profile the intrinsic birth intensity `N_0 A(\log a)` at fixed `\log a` by re-optimizing all remaining model parameters subject to a local equality constraint, exactly analogous to the IMF profile-likelihood band in panel (a).
- Updated `src/globular_clusters_imf/paper_assets.py` so Figure 8 panel (b):
  - computes the radial pointwise profile band on a support grid in `\log a`
  - interpolates that band back to the plotting grid in log-space
  - plots it with the same orange shading used in panel (a)
- Added a small robustness fix so the radial support-node array is built from the radial grid length rather than reusing the mass-grid index array.
- Updated the Figure 8 discussion and caption in `paper/main.tex` so the manuscript now states explicitly that both panels use pointwise profile-likelihood `1\sigma` bands.

### Outputs
- Updated paper figure:
  - `paper/figures/single_component_profiles.pdf`

### Verification
- Re-ran:
  - `.venv/bin/python -m compileall src scripts`
  - `.venv/bin/python scripts/build_paper_figure8_single_component_profiles.py`
  - `cd paper && latexmk -pdf -bibtex -outdir=build main.tex`
- Confirmed:
  - Figure 8 panel (b) is now generated from a radial profile-likelihood band rather than the previous Gaussian propagation band
  - the same orange uncertainty shading is used in both panels
  - the manuscript PDF was rebuilt successfully

## 2026-05-21: Remove an unsupported interpretive claim from the Figure 8 caption

### Goal
- Avoid overstating what is visually evident in the new radial profile-likelihood band of Figure 8.

### Implementation
- Updated the Figure 8 caption in `paper/main.tex` to remove the sentence claiming that the radial-profile band broadens in the outer halo.
- Replaced it with a neutral description of the panel as the pointwise profile-likelihood uncertainty on the local birth intensity `N_0A(\log a)`.

### Verification
- Re-ran:
  - `cd paper && latexmk -pdf -bibtex -outdir=build main.tex`
- Confirmed:
  - the manuscript PDF was rebuilt successfully
  - the caption now matches the visual evidence more closely

## 2026-05-21: Rewrite the Section 3.1 survivability description for clarity

### Goal
- Unpack the construction of the fixed survivability function `S(log M_ini, a)` so the paper explains clearly:
  - which variables enter the underlying disruption law,
  - what the `-0.382 dex` offset means,
  - how the orbit-based hard threshold is converted into the effective survival plane shown in Figure 2.

### Implementation
- Rewrote the key paragraph in `paper/main.tex` into an explicit sequence:
  1. define the analytic dissolution time `t_dis(M_ini, r_apo, e)`,
  2. define the hard survival indicator for a `12 Gyr` system,
  3. explain the equivalent threshold mass `M_cut(r_apo, e)`,
  4. explain the empirical `-0.382 dex` shift in `log M_ini`,
  5. explain the averaging over the observed `(r_apo, e)` distribution at fixed semimajor axis `a`.
- Added two short equations:
  - the hard survival indicator `H(M_ini, r_apo, e)`,
  - the effective averaged survivability
    `S(log M_ini, a) = < H(M_ini, r_apo, e) >_{(r_apo,e)|a}`.
- Updated the Figure 2 caption so it now states the same construction in compact form.
- Also made the main approximation explicit: the model ignores any extra structural dependence and uses the observed orbit distribution at fixed `a` as a proxy for the latent orbit distribution entering the analytic law.

### Verification
- Re-ran:
  - `cd paper && latexmk -pdf -bibtex -outdir=build main.tex`
- Confirmed:
  - the manuscript PDF was rebuilt successfully
  - Section 3.1 now defines the quantities and assumptions much more explicitly

## 2026-05-22: Clarify that the survivability offset is not arbitrary

### Goal
- Fix the remaining ambiguity in Section 3.1 where the phrase
  `a constant offset of -0.382 dex in log M_ini`
  still sounded hand-chosen.

### Implementation
- Updated `paper/main.tex` so the offset is now defined explicitly as
  the minimum downward shift needed to place every observed survivor just above the hard disruption threshold:
  - `Delta_surv = min_i [log M_ini,i - log M_cut,i] - 10^{-3}`
- Added the corresponding effective-threshold equation
  - `log M_cut^eff = log M_cut + Delta_surv`
- Reworded the prose to say explicitly that for the present catalogue
  `Delta_surv = -0.382 dex`
  is not a free tuned parameter, but a uniquely defined catalogue-matching correction under the hard-threshold approximation.
- Updated the Figure 2 caption to use the same language.

### Verification
- Re-ran:
  - `cd paper && latexmk -pdf -bibtex -outdir=build main.tex`
- Confirmed:
  - the manuscript PDF was rebuilt successfully
  - the offset is now described as a uniquely defined empirical calibration rather than an arbitrary constant

## 2026-05-22: Merge the survivability plane into Figure 1 and restyle it

### Goal
- Replace the old right-hand panel of Figure 1 with the survivability plane that had previously appeared as standalone Figure 2.
- Restyle the survivability panel so that:
  - low survivability is dark grey,
  - higher survivability fades toward white,
  - contours are grey,
  - no colorbar is shown.

### Implementation
- Updated `src/globular_clusters_imf/paper_assets.py`:
  - `plot_catalog_mass_semimajor_axis_overview_for_paper` now takes the precomputed survivability map as input.
  - The left panel remains the present-day and initial mass versus semimajor axis plot.
  - The right panel is now the survivability plane `S(M_ini, a)` rendered with a custom grey-to-white colormap and grey contour levels at `0.1`, `0.5`, and `0.9`.
  - The surviving clusters remain overplotted as black points.
  - No colorbar is added.
- Updated `scripts/build_paper_figure1_overview.py` so it now loads the survivability map via `fit_catalog_models(...)` and passes it into the Figure 1 builder.
- Updated `paper/main.tex`:
  - Figure 1 text and caption now describe the right-hand panel as the survivability plane.
  - The standalone survivability figure block was removed from the manuscript.
  - The survivability discussion now points to the right-hand panel of Figure 1 instead.

### Outputs
- Updated paper figure:
  - `paper/figures/catalog_mass_semimajor_axis_overview.pdf`
- Updated manuscript PDF:
  - `paper/build/main.pdf`

### Verification
- Re-ran:
  - `.venv/bin/python -m compileall src scripts`
  - `.venv/bin/python scripts/build_paper_figure1_overview.py`
  - `cd paper && latexmk -pdf -bibtex -outdir=build main.tex`
- Confirmed:
  - Figure 1 now has the survivability plane as panel `(b)`
  - the survivability panel uses the requested grey-to-white palette and grey contours
  - no colorbar is shown
  - the manuscript PDF was rebuilt successfully

## 2026-05-22: Lighten the minimum-grey tone in Figure 1 panel (b)

### Goal
- Make the darkest part of the survivability panel slightly less dark.

### Implementation
- Updated the low-end anchor of the custom survivability colormap in `src/globular_clusters_imf/paper_assets.py`:
  - from `#6e6e6e`
  - to `#8a8a8a`
- Rebuilt the figure and forced a manuscript rebuild so the embedded PDF asset was refreshed.

### Verification
- Re-ran:
  - `.venv/bin/python scripts/build_paper_figure1_overview.py`
  - `cd paper && latexmk -g -pdf -bibtex -outdir=build main.tex`
- Confirmed:
  - the darkest grey in Figure 1 panel `(b)` is visibly lighter
  - the manuscript PDF was rebuilt successfully

## 2026-05-22: Match the axis ranges of both panels in Figure 1 exactly

### Goal
- Make the right-hand survivability panel use exactly the same axis ranges as the left-hand mass-versus-semimajor-axis panel.

### Implementation
- Updated `src/globular_clusters_imf/paper_assets.py` so Figure 1 panel `(b)` is now drawn in physical mass units rather than in `log10(M_ini)` coordinates:
  - converted the survivability map grid from `log M_ini` to `M_ini`
  - converted the contour grid likewise
  - overplotted the clusters using `initial_mass_msun`
  - set `yscale='log'`
  - applied the same `x_limits` and `y_limits` as panel `(a)`
- The right-hand panel now shares the same plotting range in both dimensions as the left panel, while preserving the grey-to-white survivability styling.

### Verification
- Re-ran:
  - `.venv/bin/python scripts/build_paper_figure1_overview.py`
  - `cd paper && latexmk -g -pdf -bibtex -outdir=build main.tex`
- Confirmed:
  - the right-hand panel now uses the same x and y ranges as the left-hand panel
  - the manuscript PDF was rebuilt successfully

## 2026-05-22: Fix the rendered extent of the survivability panel in Figure 1

### Goal
- Remove the remaining visual mismatch where the right-hand survivability image did not fill the full plotting range, even though the axis limits matched the left panel.

### Implementation
- Updated `src/globular_clusters_imf/paper_assets.py` to pad the plotted survivability grid itself, not just the axis limits:
  - expanded the x- and y-edge arrays to the full left-panel bounds
  - padded the survivability array with edge values using `np.pad(..., mode="edge")`
- This makes the pcolormesh occupy the full coordinate range of panel `(a)` rather than stopping at the original survivability-grid boundary.

### Verification
- Re-ran:
  - `.venv/bin/python scripts/build_paper_figure1_overview.py`
  - `pdftoppm -png -singlefile paper/figures/catalog_mass_semimajor_axis_overview.pdf /tmp/catalog_overview_check2`
  - `cd paper && latexmk -g -pdf -bibtex -outdir=build main.tex`
- Confirmed:
  - the right-hand panel now visibly fills the same plotting extent as the left-hand panel
  - the manuscript PDF was rebuilt successfully

## 2026-05-22: Reposition Figure 1 panel labels and left-panel legend

### Goal
- Move the `(a)` and `(b)` panel labels to the top-right corners.
- Move the `Present mass` / `Initial mass` legend to the lower-left corner of the left panel.

### Implementation
- Updated `src/globular_clusters_imf/paper_assets.py`:
  - panel `(a)` label moved from top-left to top-right
  - panel `(b)` label moved from top-left to top-right
  - left-panel legend moved from upper-right to lower-left

### Verification
- Re-ran:
  - `.venv/bin/python scripts/build_paper_figure1_overview.py`
  - `cd paper && latexmk -g -pdf -bibtex -outdir=build main.tex`
- Confirmed:
  - the panel labels are now in the top-right corners
  - the left-panel legend is now in the lower-left corner
  - the manuscript PDF was rebuilt successfully

## 2026-05-22: Tighten the Figure 1 corner placements after visual inspection

### Goal
- Make the Figure 1 layout changes visually unambiguous in the rendered output.

### Implementation
- Updated `src/globular_clusters_imf/paper_assets.py` again after checking the rendered PNG:
  - moved `(a)` and `(b)` to `(0.985, 0.975)` in axes coordinates
  - anchored the left-panel legend explicitly with
    `bbox_to_anchor=(0.03, 0.03)`
    and `borderaxespad=0.0`
- This pushes the legend tighter into the lower-left corner and the panel labels tighter into the upper-right corners.

### Verification
- Re-ran:
  - `.venv/bin/python scripts/build_paper_figure1_overview.py`
  - `cd paper && latexmk -g -pdf -bibtex -outdir=build main.tex`
- Confirmed:
  - the figure layout change is now more visually obvious
  - the manuscript PDF was rebuilt successfully

## 2026-05-22: Test for residual longitude dependence in the detectability model

### Goal
- Check whether the current detectability correction
  `C(log M_now, log D_sun, |b|)`
  leaves a measurable residual dependence on Galactic longitude `l`.

### Implementation
- Added the reusable script:
  - `scripts/check_detectability_longitude_dependence.py`
- The script:
  - reads the saved best-fit detectability outputs
    - `outputs/tables/joint_fixed_survival_detectability_em_observable_histogram.csv`
    - `outputs/tables/joint_fixed_survival_detectability_em_catalog_completeness.csv`
  - reconstructs the observable-space bins in
    `(log M_now, D_sun, |b|)`
  - assigns each observed GC the Poisson residual of its fitted detectability bin
  - merges in Galactic longitude from the project catalog
  - fits a circular first-harmonic model
    `residual ~ 1 + sin(l) + cos(l)`
    as a diagnostic for any leftover longitude trend
  - writes per-GC, binned, and summary outputs and a diagnostic figure
- Also updated the script to use project-local Matplotlib cache directories automatically so it runs cleanly without shell-specific cache workarounds.

### Outputs
- Per-GC diagnostic table:
  - `outputs/tables/detectability_longitude_residuals_per_gc.csv`
- Longitude-binned summary:
  - `outputs/tables/detectability_longitude_residuals_by_l_bin.csv`
- Summary statistics:
  - `outputs/tables/detectability_longitude_residuals_summary.json`
- Diagnostic figure:
  - `outputs/figures/detectability_longitude_residuals_vs_l.png`

### Result
- No convincing evidence for a residual longitude dependence was found in this first-pass test:
  - joint `sin(l), cos(l)` test: `p = 0.310`
  - `Delta BIC = +7.82` for the longitude model relative to the null, so BIC prefers no longitude term
  - `R^2 = 0.014`, i.e. the harmonic longitude term explains only about `1.4%` of the variance in the assigned detectability residuals
- The visual diagnostic is consistent with the formal test: the longitude-binned residual means fluctuate, but not in a statistically compelling way.

### Verification
- Re-ran:
  - `.venv/bin/python -m compileall scripts/check_detectability_longitude_dependence.py`
  - `.venv/bin/python scripts/check_detectability_longitude_dependence.py`
- Confirmed:
  - the script runs directly
  - the outputs are written successfully
  - the no-longitude-signal result is stable

## 2026-05-22: Update the paper to the longitude-aware single-component inference and the `|l|`-split detectability figures

### Goal
- Replace the paper detectability diagnostics with the `|l|`-split versions requested by the user.
- Update the single-component paper figures, macros, and text to the full longitude-aware detectability model
  `C(log M_now, log D_sun, |b|, |l|)`.
- Keep the previously computed two-component results intact, but make the manuscript explicit that they still use the simpler longitude-averaged detectability correction.

### Implementation
- Updated `src/globular_clusters_imf/paper_assets.py` so the paper builder now:
  - uses `fit_detectability_corrected_single_component_models_with_abs_longitude(...)` for the single-component paper products
  - builds Figure 2 as the `|l|=30 deg` split version of the observed count maps
  - builds Figure 3 from the precomputed corrected `|l|=30 deg` detectability split tables
  - prioritizes the new flexible-IMF overlay from
    `variants/flexible_imf_bootstrap_comparison_abs_longitude/`
  - refreshes the single-component comparison table caption to say explicitly that it is longitude-aware
- Added the reusable script:
  - `scripts/refresh_paper_single_component_abs_longitude.py`
- That script:
  - refits the full longitude-aware single-component model family
  - rebuilds the single-component paper figures
  - updates the detectability-related rows in the paper tables and macros
  - leaves the unchanged two-component paper assets alone
- Updated the helper script:
  - `scripts/build_paper_figure8_single_component_profiles.py`
  - so it also uses the longitude-aware detectability model
- Generalized the flexible-IMF infrastructure:
  - `src/globular_clusters_imf/flexible_imf.py`
  - `scripts/compare_profile_and_flexible_imf.py`
  - to support both the old baseline detectability model and the new `abs_longitude` variant

### New inference results
- Full longitude-aware single-component best model remains:
  - `schechter + logpoly3`
- Updated best-fit parameters:
  - `alpha_dndm = -0.9400`
  - `log10(M_c/Msun) = 6.3007`
  - `N0 = 792.3`
  - `M_star,0 = 3.17e8 Msun`
  - `selection_fraction = 0.2083`
  - `raw_survival_fraction = 0.2509`
  - `mean_detectability = 0.8301`
- Relative to the older longitude-averaged detectability correction:
  - `N0` increases only mildly, from `784.6` to `792.3`
  - the inferred IMF shape changes only slightly

### Flexible-IMF cross-check
- Ran:
  - `.venv/bin/python scripts/compare_profile_and_flexible_imf.py --detectability-variant abs_longitude --n-bootstrap 12 --seed 12345`
- New outputs are under:
  - `variants/flexible_imf_bootstrap_comparison_abs_longitude/`
- Result:
  - flexible `logspline6 + logpoly3` improves the raw likelihood relative to the Schechter model by
    `Delta logL = +3.77`
  - but remains worse by BIC by
    `Delta BIC = +7.78`
  - total initial count shifts from `792.3` to `851.4`
- The main scientific message remains unchanged:
  - the flexible band broadens the low-mass IMF uncertainty
  - but the present data do not require the extra shape freedom strongly enough to beat the Schechter baseline under BIC

### Paper outputs refreshed
- Updated figures:
  - `paper/figures/detectability_counts.pdf`
  - `paper/figures/detectability_em_maps.pdf`
  - `paper/figures/detectability_em_convergence.pdf`
  - `paper/figures/single_component_model_performance.pdf`
  - `paper/figures/best_single_component_summary.pdf`
  - `paper/figures/single_component_profiles.pdf`
- Updated tables/macros:
  - `paper/tables/single_component_model_comparison.csv`
  - `paper/tables/single_component_model_comparison.tex`
  - `paper/tables/key_results_summary.csv`
  - `paper/tables/key_results_summary.tex`
  - `paper/tables/paper_numbers.tex`
  - `paper/tables/paper_results_summary.json`
- Updated manuscript prose in `paper/main.tex`:
  - detectability section now defines
    `C(log M_now, log D_sun, |b|, |l|)`
  - Figure 2 and Figure 3 discussion now matches the `|l|`-split diagnostics
  - single-component results and Figure 8 discussion now use the longitude-aware numbers
  - the two-component section now states explicitly that those results still use the simpler longitude-averaged detectability correction

### Verification
- Re-ran:
  - `.venv/bin/python -m compileall src scripts`
  - `.venv/bin/python scripts/compare_profile_and_flexible_imf.py --detectability-variant abs_longitude --n-bootstrap 12 --seed 12345`
  - `.venv/bin/python scripts/refresh_paper_single_component_abs_longitude.py`
  - `cd paper && latexmk -pdf -bibtex -outdir=build main.tex`
- Confirmed:
  - the single-component paper products now match the longitude-aware detectability inference
  - Figures 2 and 3 are the requested `|l|`-split versions
  - Figure 8 now uses the updated longitude-aware flexible-IMF bootstrap band
  - `paper/build/main.pdf` rebuilds successfully

## 2026-05-22: Extended EM convergence for the longitude-aware detectability model

### Why
- Figure 4 previously stopped at iteration 6, where the total initial count `N_0`
  was still rising slightly.
- To show whether the EM-like detectability correction really converges, I
  extended the run and added an explicit fit-quality curve to the convergence
  figure.

### Code changes
- Updated `src/globular_clusters_imf/detectability_longitude_model.py`:
  - extended the default EM sequence from `6` to `12` iterations
  - recorded per-iteration fit quality:
    - `log_likelihood`
    - `rms_residual_sigma_2d`
    - `mean_abs_residual_sigma_2d`
- Updated `src/globular_clusters_imf/paper_assets.py`:
  - Figure 4 now shows:
    - `N_0` versus iteration
    - `Delta log L` versus iteration, relative to the perfect-detectability
      baseline

### Numerical convergence
- New iteration history:
  - `outputs/tables/joint_fixed_survival_detectability_abs_longitude_em_iteration_history.csv`
- Key values:
  - iteration `6`:
    - `N_0 = 790.31`
    - `log L = 471.67878`
    - `RMS = 1.011264`
  - iteration `12`:
    - `N_0 = 794.99`
    - `log L = 471.71594`
    - `RMS = 1.011237`
- Change from iteration `6` to `12`:
  - `Delta N_0 = +4.68` clusters
  - relative change `= +0.59%`
  - `Delta log L = +0.037`
  - `Delta RMS = -2.73e-5`
- Interpretation:
  - the rise after iteration 6 is real
  - but by iteration 12 the sequence is clearly flattening
  - the scientific inference is stable

### Updated best-fit single-component inference
- Best longitude-aware detectability-corrected model remains:
  - `schechter + logpoly3`
- Updated summary:
  - `alpha_dndm = -0.9400`
  - `log10(M_c/Msun) = 6.3003`
  - `N_0 = 795.1`
  - `M_star,0 = 3.18e8 Msun`
  - mean detectability `= 0.828`

### Paper updates
- Updated:
  - `paper/figures/detectability_em_convergence.pdf`
  - `paper/tables/paper_numbers.tex`
  - `paper/figures/single_component_profiles.pdf`
  - `paper/main.tex`
- The manuscript text now states explicitly that:
  - the EM-like run was extended to 12 iterations
  - the change relative to iteration 6 is only about `0.6%` in `N_0`
  - the gain in likelihood is only `Delta log L ~ 0.04`

### Verification
- Re-ran:
  - `.venv/bin/python -m compileall src scripts`
  - `python scripts/run_abs_longitude_detectability_inference.py`
  - `python scripts/refresh_paper_single_component_abs_longitude.py`
  - `cd paper && latexmk -pdf -bibtex -outdir=build main.tex`
- Confirmed:
  - the extended iteration history is written to disk
  - Figure 4 now shows both convergence in `N_0` and fit quality
  - `paper/build/main.pdf` rebuilds successfully

## 2026-05-22: Restored detectability curve in Figure 4 and propagated the longitude-aware EM update through the two-component analysis

### Figure 4 update
- Restored the mean detectability curve in the convergence figure, rather than
  replacing it with `Delta log L`.
- New layout:
  - top panel: `N_0` and mean detectability `⟨C⟩`
  - bottom panel: `Delta log L` relative to the perfect-detectability baseline
- This keeps the original detectability information visible while still showing
  whether the EM iteration is materially improving the fit.

### New longitude-aware two-component inference
- Added a full `|l|`-aware two-component EM path in
  `src/globular_clusters_imf/detectability_longitude_model.py`.
- Implemented:
  - shared-IMF two-component detectability model with
    `C(log M_now, log D_sun, |b|, |l|)`
  - separate-IMF two-component detectability model with the same completeness law
  - full output tables and JSON summaries parallel to the older
    longitude-averaged products
- New outputs:
  - `outputs/tables/joint_fixed_survival_detectability_abs_longitude_shared_imf_two_component_model_summary.csv`
  - `outputs/tables/joint_fixed_survival_detectability_abs_longitude_two_component_model_summary.csv`
  - plus the corresponding best-component, IMF-grid, radial-grid, iteration,
    completeness-grid, histogram, and catalog tables

### Updated two-component results
- Preferred shared-IMF longitude-aware model:
  - `schechter + logpoly3/logpoly3`
  - `log L = 451.7365`
  - `N_0 = 814.7`
  - `M_star,0 = 3.23e8 Msun`
  - shared IMF:
    - `alpha_dndm = -0.9444`
    - `log10(M_c/Msun) = 6.3016`
  - component split:
    - in-situ:
      - `N_0 = 696.3`
      - `f_sel = 0.1537`
      - `M_star,0 = 2.76e8 Msun`
    - accreted:
      - `N_0 = 118.4`
      - `f_sel = 0.4899`
      - `M_star,0 = 4.70e7 Msun`
  - mean detectability `= 0.831`

- Preferred separate-IMF longitude-aware model:
  - in-situ `schechter + logpoly3`
  - accreted `schechter + logpoly3`
  - `log L = 453.0263`
  - `N_0 = 1021.8`
  - `M_star,0 = 3.27e8 Msun`
  - component IMFs:
    - in-situ:
      - `alpha_dndm = -1.0891`
      - `log10(M_c/Msun) = 6.3990`
    - accreted:
      - `alpha_dndm = -0.8006`
      - `log10(M_c/Msun) = 6.1024`

### Updated model comparison
- Conditional BIC comparison with the longitude-aware detectability law:
  - shared-IMF two-component:
    - `BIC_cond = -1076.59`
  - separate-IMF two-component:
    - `BIC_cond = -1068.96`
  - single population:
    - `BIC_cond = -917.90`
- Therefore:
  - single population is worse than the preferred shared-IMF two-component model
    by `Delta BIC_cond = 158.69`
  - separate IMF is worse than shared IMF by
    `Delta BIC_cond = 7.63`
- So the qualitative conclusion is unchanged:
  - the data strongly support a two-component radial split
  - but still do not require separate IMF shapes

### Paper refresh
- Updated `src/globular_clusters_imf/paper_assets.py` so the paper now uses the
  longitude-aware two-component outputs rather than the older
  longitude-averaged ones.
- Rebuilt paper assets with:
  - `source .venv/bin/activate && python scripts/build_paper_assets.py`
- Refreshed paper products:
  - `paper/figures/detectability_em_convergence.pdf`
  - `paper/figures/two_component_results.pdf`
  - `paper/tables/paper_numbers.tex`
  - `paper/tables/key_results_summary.csv`
  - `paper/tables/population_model_class_comparison.csv`
  - `paper/tables/paper_results_summary.json`
- Updated manuscript prose in `paper/main.tex`:
  - Figure 4 caption now mentions both `⟨C⟩` and `Delta log L`
  - the two-component formulation now uses
    `C(log M_now, log D_sun, |b|, |l|)`
  - the two-component results section now reports the new
    longitude-aware shared/separate model numbers

### Verification
- Re-ran:
  - `.venv/bin/python -m compileall src scripts`
  - `source .venv/bin/activate && python scripts/build_paper_assets.py`
- Confirmed:
  - the new longitude-aware shared and separate two-component summaries are written
  - the paper tables/macros now reflect the updated two-component analysis

## 2026-05-22: Replace fully separate two-component IMF test with split-alpha test

### Motivation
- Replaced the previous two-component "different IMF" comparison with a tighter
  nested alternative:
  - shared Schechter cutoff mass `M_c`
  - separate low-mass slopes `alpha_in` and `alpha_acc`
  - separate radial profiles `A_in(a)` and `A_acc(a)`
- Also corrected the detectability treatment in the two-component stage:
  - do **not** re-estimate detectability from the labelled subsets
  - instead freeze the effective detectability correction `Q(log M_ini, a)` to
    the final single-component `|l|`-aware solution

### Code changes
- Added a reusable split-alpha two-component model in
  `src/globular_clusters_imf/two_component_model.py`:
  - `SplitAlphaTwoComponentSpec`
  - `SplitAlphaTwoComponentJointFitResult`
  - `fit_split_alpha_two_component_single_model`
  - parameter bounds / starts / unpack / likelihood helpers
  - component-table/grid builders
- Added the longitude-aware wrapper in
  `src/globular_clusters_imf/detectability_longitude_model.py`:
  - `fit_split_alpha_two_component_detectability_em_models_with_abs_longitude`
  - `fit_split_alpha_two_component_detectability_em_single_model_with_abs_longitude`
- Refactored the two-component detectability wrappers so shared-IMF and
  split-alpha fits can optionally use:
  - `fixed_effective_completeness_grid`
  - `fixed_completeness_bin_grid`
  - `fixed_completeness_raw_parameters`
  from the final single-component run instead of re-estimating detectability.
- Added a focused paper refresh script:
  - `scripts/update_split_alpha_two_component_paper_assets.py`
  which updates just:
  - `paper/figures/two_component_results.pdf`
  - `paper/tables/population_model_class_comparison.*`
  - `paper/tables/key_results_summary.*`
  - `paper/tables/paper_numbers.tex`
  - `paper/tables/paper_results_summary.json`

### Results
- Preferred shared-IMF two-component model with detectability fixed from the
  single-component run:
  - shared Schechter + `logpoly3/logpoly3`
  - `logL = 451.7219`
  - `N0 = 816.6`
  - `alpha = -0.9443`
  - `log10(M_c/Msun) = 6.3014`
- Best split-alpha two-component model:
  - shared `M_c`, separate `alpha`, `logpoly3/logpoly3`
  - `logL = 451.7374`
  - `N0 = 853.9`
  - `alpha_in = -0.9684`
  - `alpha_acc = -0.9403`
  - `log10(M_c/Msun) = 6.3092`
- Statistical comparison:
  - shared IMF:
    - `BIC_cond = -1076.56`
  - split-alpha:
    - `BIC_cond = -1071.49`
  - single population:
    - `BIC_cond = -917.90`
- Therefore:
  - the labelled two-component shared-IMF model is still strongly preferred
    over the single-population model by
    `Delta BIC_cond = 158.66`
  - allowing only `alpha` to split improves the raw likelihood by only
    `Delta logL = 0.016`
  - but is still worse than the shared-IMF model by
    `Delta BIC_cond = 5.07`

### Paper updates
- Updated `paper/main.tex` so the two-component section now states clearly that:
  - the detectability correction is frozen to the final single-component run
  - the IMF-difference test is the tighter split-alpha comparison
- Updated the abstract, Section 8, Discussion, Conclusions, and data
  availability note accordingly.
- Updated `paper/figures/two_component_results.pdf` so panel `(a)` now compares:
  - preferred shared IMF
  - best split-alpha in-situ IMF
  - best split-alpha accreted IMF

### Verification
- Re-ran:
  - `.venv/bin/python -m compileall src scripts`
  - `source .venv/bin/activate && python scripts/update_split_alpha_two_component_paper_assets.py`
- Confirmed:
  - the split-alpha output tables are written under
    `outputs/tables/joint_fixed_survival_detectability_abs_longitude_split_alpha_two_component_*`
  - the paper-side comparison tables and macros now reflect the shared versus
    split-alpha analysis with detectability fixed from the single-component run

## 2026-05-22: Blind latent 2-component radial-power-law mixture test

### Goal
- Test a blind unlabeled 2-component hypothesis without using the
  Baumgardt-Belokurov-Kruijssen in-situ/accreted labels in the fit.
- Keep the current shared survivability and detectability corrections, but let
  the intrinsic radial `a` distribution be a mixture of two overlapping power
  laws spanning the full `a` range.

### Model
- Replaced the abandoned unconstrained `inner/outer` EM split with a direct
  point-process fit in
  `src/globular_clusters_imf/blind_mixture_model.py`.
- The new blind model uses:
  - shared IMF `phi(log M_ini)`
  - fixed survivability `S(log M_ini, a)`
  - fixed detectability `Q(log M_ini, a)` reconstructed from the final
    single-component `|l|`-aware run
  - either:
    - a single power-law radial density in `d log a`, or
    - a 2-component overlapping mixture of two power-law radial densities
- The 2-component model is constrained by:
  - both components defined over the same finite `a` range
  - ordered slopes via a positive slope gap
  - a bounded mixture fraction away from exactly 0 or 1

### Implementation
- Added reusable runner:
  - `scripts/run_blind_powerlaw_a_mixture.py`
- The runner now supports `--skip-plot` so the numerical summary can complete
  even in headless environments where matplotlib/font cache setup is unstable.
- Outputs are written under the separate variant tree:
  - `variants/blind_powerlaw_a_mixture/outputs/tables/`

### Results
- Best model in this restricted blind family:
  - `single_powerlaw_radial + schechter`
  - `logL = 463.26011`
  - `BIC = -911.20`
  - `N0 = 899.34`
  - `M_star,0 = 3.77e8 Msun`
- Best blind 2-component model:
  - `two_component_powerlaw_mixture + schechter`
  - `logL = 463.26009`
  - `BIC = -900.99`
  - `Delta BIC = +10.21` relative to the 1-component power-law-radial null
- The 2-component fit collapses back toward the 1-component limit:
  - `beta_concentrated = -1.68158`
  - `beta_extended = -1.67484`
  - `mix_fraction_concentrated = 0.9445`
  - posterior `p_concentrated` spans only `0.94375` to `0.94474`
- Therefore the unlabeled latent 2-component hypothesis is **not supported** in
  this formulation.

### BK comparison
- The blind fit was not given the BK labels, but the saved posterior table can
  be compared afterward.
- Since the posterior is almost constant, the hard assignment is trivial:
  every cluster remains on the `concentrated` side of `p=0.5`.
- The score still decreases slightly toward the far outer halo, so:
  - `AUC(in-situ vs p_concentrated) = 0.988`
  but this should not be over-interpreted because the dynamic range is tiny and
  the classifier is effectively collapsed.

### Files
- New core model:
  - `src/globular_clusters_imf/blind_mixture_model.py`
- New runner:
  - `scripts/run_blind_powerlaw_a_mixture.py`
- Main outputs:
  - `variants/blind_powerlaw_a_mixture/outputs/tables/blind_powerlaw_a_mixture_model_summary.csv`
  - `variants/blind_powerlaw_a_mixture/outputs/tables/blind_powerlaw_a_mixture_model_summary.json`
  - `variants/blind_powerlaw_a_mixture/outputs/tables/blind_powerlaw_a_mixture_best_two_component_posterior_probabilities.csv`
  - `variants/blind_powerlaw_a_mixture/outputs/tables/blind_powerlaw_a_mixture_comparison_summary.json`

### Verification
- Re-ran:
  - `python -m compileall src/globular_clusters_imf/blind_mixture_model.py scripts/run_blind_powerlaw_a_mixture.py`
  - `source .venv/bin/activate && python scripts/run_blind_powerlaw_a_mixture.py --skip-plot`
- Confirmed:
  - the fixed `|l|`-aware detectability correction was reconstructed from the
    saved single-component summary
  - all blind-mixture tables were written to the variant output directory

### Plotting follow-up
- The earlier issue was not matplotlib itself; it was my runner setup.
- Fixed by forcing the non-interactive backend and writable local caches in:
  - `scripts/run_blind_powerlaw_a_mixture.py`
- Re-ran:
  - `source .venv/bin/activate && python scripts/run_blind_powerlaw_a_mixture.py`
- Confirmed figure outputs:
  - `variants/blind_powerlaw_a_mixture/outputs/figures/blind_powerlaw_a_mixture_diagnostic.pdf`
  - `variants/blind_powerlaw_a_mixture/outputs/figures/blind_powerlaw_a_mixture_diagnostic.png`

## 2026-05-22: Blind latent 2-component split-alpha radial-power-law mixture

### Goal
- Extend the blind unlabeled mixture test so class membership can depend on both
  mass and radius, not only on `a`.
- Keep the fixed single-component survivability and detectability corrections:
  - `S(log M_ini, a)`
  - `Q(log M_ini, a)`
- Compare three blind Schechter-based models:
  - `single_powerlaw_radial`
  - `two_component_powerlaw_mixture` with shared IMF
  - `two_component_powerlaw_mixture_split_alpha` with shared `M_c` and split
    `alpha`

### Implementation
- Extended `src/globular_clusters_imf/blind_mixture_model.py` with:
  - optional `include_split_alpha_schechter`
  - optional `output_prefix`
  - new model class:
    - `two_component_powerlaw_mixture_split_alpha`
- Generalized the blind point-process likelihood to work with the full
  mixture density in `(log M_ini, log a)` rather than factorized
  `phi(M) A(a)` terms only.
- Added component-level IMF grids and per-cluster posterior probabilities using
  both mass and radius:
  - `P(concentrated | M_ini, a)`
- Added new dedicated runner:
  - `scripts/run_blind_powerlaw_a_split_alpha_mixture.py`

### Results
- The extra IMF freedom does make the blind split non-trivial:
  - posterior `p_concentrated` now spans `0.393` to `0.999`
  - only `3` clusters are assigned to the `extended` side at `p<0.5`
  - `p_concentrated` correlates strongly with both:
    - `log M_ini`: `r = +0.79`
    - `log a`: `r = -0.76`
- Best split-alpha blind fit:
  - `logL = 463.44166`
  - `BIC = -896.25`
  - `N0 = 808.55`
  - `M_star,0 = 3.77e8 Msun`
  - `alpha_concentrated = -0.763`
  - `alpha_extended = -1.335`
  - shared `log10(M_c/Msun) = 6.186`
  - radial slopes:
    - concentrated `beta = -1.734`
    - extended `beta = -0.349`
- But the statistical support is still weak-to-negative:
  - versus blind shared-IMF 2-component model:
    - `Delta logL = +0.182`
    - `Delta BIC = +4.74`
  - versus blind 1-component power-law-radial null:
    - `Delta logL = +0.182`
    - `Delta BIC = +14.95`
- Therefore the blind split-alpha model is more interpretable than the blind
  shared-IMF mixture, but still not preferred by BIC.

### BK comparison
- The blind split-alpha model aligns with the external origin labels much more
  meaningfully than the radial-only blind model:
  - `AUC(in-situ vs p_concentrated) = 0.922`
  - hard-assignment accuracy `= 0.667`
  - mean `p_concentrated`:
    - in-situ `= 0.985`
    - accreted `= 0.875`
- Lowest `p_concentrated` objects are outer-halo, low-mass accreted clusters
  such as:
  - `Laevens 3`
  - `Pal 4`
  - `Whiting 1`

### Outputs
- Variant directory:
  - `variants/blind_powerlaw_a_split_alpha_mixture/`
- Main tables:
  - `outputs/tables/blind_powerlaw_a_split_alpha_mixture_model_summary.csv`
  - `outputs/tables/blind_powerlaw_a_split_alpha_mixture_comparison_summary.json`
  - `outputs/tables/blind_powerlaw_a_split_alpha_mixture_best_split_alpha_posterior_probabilities.csv`
- Diagnostic figure:
  - `outputs/figures/blind_powerlaw_a_split_alpha_mixture_diagnostic.pdf`
  - `outputs/figures/blind_powerlaw_a_split_alpha_mixture_diagnostic.png`

### Verification
- Re-ran:
  - `python -m compileall src/globular_clusters_imf/blind_mixture_model.py scripts/run_blind_powerlaw_a_split_alpha_mixture.py`
  - `source .venv/bin/activate && python scripts/run_blind_powerlaw_a_split_alpha_mixture.py`
- Confirmed:
  - the new split-alpha outputs were written under the dedicated variant tree
  - the diagnostic figure and comparison summary were both generated

## 2026-05-22: Mg-Al chemistry-augmented blind latent mixtures

### Goal
- Add the local GC `[Mg/Fe]` and `[Al/Fe]` measurements as a partially observed
  chemical marker to the blind latent mixture model.
- Keep the original blind radial-only and blind split-`alpha` runs intact.
- Use the chemistry in the spirit of Baumgardt & Kruijssen-style in-situ/accreted
  separation discussed around Figure 11 of Belokurov et al. (2024): the two GC
  populations occupy offset sequences in the `[Mg/Fe]-[Al/Fe]` plane.

### Catalog plumbing
- Added reusable chemistry export and join functions in
  `src/globular_clusters_imf/catalog.py`:
  - `export_local_gc_chemistry_markers(...)`
  - `attach_local_gc_chemistry_to_baumgardt_catalog(...)`
- Added regeneration script:
  - `scripts/export_gc_chemistry_markers.py`
- Combined abundances are constructed from the local updated FITS catalog
  `~/data/catalogues/gc_catalog_updated.fits` using:
  - APOGEE values when present (`*_ERR_APO > 0`)
  - otherwise literature values (`*_ERR_OTHER > 0` and value > `-90`)
    shifted onto the APOGEE scale via the median offset measured from clusters
    with both measurements.
- Adopted offsets:
  - `Mg`: `MG_OTHER - MG_APO = +0.14984 dex`
  - `Al`: `AL_OTHER - AL_APO = +0.24816 dex`
- Added the offset scatter in quadrature to the literature uncertainty when the
  literature value is used:
  - `Mg` offset scatter: `0.06443 dex`
  - `Al` offset scatter: `0.19696 dex`

### Chemistry coverage
- Total Milky Way GC catalog size after the Baumgardt join: `165`
- With combined `[Mg/Fe]`: `68`
- With combined `[Al/Fe]`: `63`
- With both combined `[Mg/Fe]` and `[Al/Fe]`: `63`
- New processed tables:
  - `data/processed/gc_chemistry_markers.csv`
  - `data/processed/gc_chemistry_markers_summary.csv`
  - `data/processed/baumgardt_gc_catalog_with_origin_and_chemistry.csv`
  - `data/processed/baumgardt_gc_catalog_with_origin_and_chemistry_summary.csv`

### Model
- Added a separate chemistry-aware blind-mixture module:
  - `src/globular_clusters_imf/blind_chemistry_mixture_model.py`
- Added runner:
  - `scripts/run_blind_mg_al_mixture.py`
- The intrinsic part of the model is unchanged from the blind radial-power-law
  family:
  - shared fixed survival `S(M_ini, a)`
  - shared fixed detectability `Q(M_ini, a)` from the final single-component
    `|l|`-aware EM run
  - radial densities as one or two power laws in `a`
  - either shared Schechter IMF or shared-`M_c` split-`alpha` Schechter IMFs
- The new chemical term multiplies the point-process intensity for the subset of
  clusters with chemistry:
  - each component has a chemistry likelihood in the `[Mg/Fe]-[Al/Fe]` plane
  - missing chemistry is marginalized naturally
- The chemistry distribution is parameterized as a shared-slope bivariate normal:
  - one common slope for the `Mg-Al` sequence
  - component-specific means and scatters
  - measurement errors are convolved in quadrature per cluster
- This follows the qualitative Figure 11 picture:
  - clusters evolve along approximately parallel `Mg-Al` sequences
  - the two populations are offset in the plane

### Results
- Outputs are in:
  - `variants/blind_mg_al_mixture/outputs/`
- Main comparison table:
  - `outputs/tables/blind_mg_al_mixture_model_summary.csv`
- Main comparison JSON:
  - `outputs/tables/blind_mg_al_mixture_comparison_summary.json`
- Diagnostic plot:
  - `outputs/figures/blind_mg_al_mixture_diagnostic.pdf`
  - `outputs/figures/blind_mg_al_mixture_diagnostic.png`

#### Model comparison
- Best chemistry-aware model by BIC is still the **single-component** blind
  power-law-radial model:
  - `logL = 537.29037`
  - `AIC = -1058.58`
  - `BIC = -1033.73`
  - `N0 = 899.34`
- Blind 2-component shared-IMF chemistry model:
  - `logL = 545.23217`
  - `AIC = -1062.46`
  - `BIC = -1018.98`
  - `N0 = 923.59`
- Blind 2-component split-`alpha` chemistry model:
  - `logL = 545.24149`
  - `AIC = -1060.48`
  - `BIC = -1013.89`
  - `N0 = 906.58`

#### Interpretation
- The chemistry marker *does* make the blind 2-component model non-trivial:
  - the raw likelihood improves substantially relative to the single-component
    chemistry model:
    - shared-IMF two-component: `Delta logL = +7.94`
    - split-`alpha` two-component: `Delta logL = +7.95`
  - AIC mildly prefers the shared-IMF 2-component chemistry model:
    - `Delta AIC(shared - single) = -3.88`
- However, BIC still prefers the simpler single-component model:
  - `Delta BIC(shared - single) = +14.75`
  - `Delta BIC(split-alpha - single) = +19.84`
- Allowing different low-mass slopes buys essentially nothing beyond the shared
  chemistry split:
  - `Delta logL(split-alpha - shared) = +0.0093`
  - `Delta BIC(split-alpha - shared) = +5.09`
- So the current chemistry subset is enough to make the blind latent split
  identifiable in practice, but not enough to make the 2-component chemistry
  model decisively preferred by BIC.

### Blind posterior behaviour
- Best split-`alpha` chemistry posterior probabilities span almost the full unit
  interval:
  - `p_concentrated min = 3.8e-05`
  - `p_concentrated max = 0.99999`
- This is a much sharper and more genuinely bimodal blind assignment than in the
  radial-only case.
- Mean `p_concentrated` by BK label:
  - in-situ: `0.776`
  - accreted: `0.424`
- Hard assignment accuracy versus BK:
  - `0.824`
- AUC(in-situ vs `p_concentrated`):
  - `0.873`
- Correlations with the blind posterior:
  - `corr(p_concentrated, log M_ini) = +0.346`
  - `corr(p_concentrated, log10 a) = -0.602`
- So chemistry reduces the near-degeneracy with radius alone:
  - radius still matters
  - but the posterior is no longer driven almost entirely by `a`

### Best-fit chemistry-aware split-alpha parameters
- Shared chemistry-sequence slope:
  - `d[Mg/Fe]/d[Al/Fe] = -0.004`
- Concentrated component chemistry mean:
  - `mu_[Al/Fe] = 0.272`
  - `mu_[Mg/Fe] = 0.275`
- Extended component chemistry mean:
  - `mu_[Al/Fe] = 0.038`
  - `mu_[Mg/Fe] = 0.136`
- Shared-`M_c` split-`alpha` IMF:
  - `alpha_concentrated = -0.838`
  - `alpha_extended = -0.871`
  - `log10(M_c/Msun) = 6.213`
- Radial slopes:
  - concentrated `beta = -2.050`
  - extended `beta = -1.078`

### Notable clusters
- Very low `p_concentrated` clusters in the chemistry-aware split include:
  - `NGC 6715`
  - `NGC 362`
  - `NGC 5272`
  - `NGC 3201`
  - `Pal 12`
- Most strongly concentrated clusters chemically are:
  - `NGC 6121`
  - `NGC 6380`
  - `NGC 104`
  - `NGC 6642`
  - `NGC 6171`
- One clear outlier is `NGC 6388`, which carries an in-situ BK flag but lands at
  extremely small `p_concentrated` in the blind chemistry model.

### Verification
- Re-ran:
  - `python -m compileall src scripts`
  - `python scripts/export_gc_chemistry_markers.py`
  - `python scripts/run_blind_mg_al_mixture.py`
- Confirmed all outputs were written under:
  - `variants/blind_mg_al_mixture/outputs/`

## 2026-05-22: Quick Mg-Al diagnostic for the simplest chemistry model

### Goal
- Test whether the chemistry can be compressed to a single useful coordinate.
- Measure from the `63` GCs with both combined `[Mg/Fe]` and `[Al/Fe]`:
  - the fitted common slope `s`
  - the orthogonal scatter
  - whether `[Mg/Fe]` alone is already almost sufficient

### Implementation
- Added reusable script:
  - `scripts/diagnose_mg_al_gc_chemistry.py`
- Inputs:
  - `data/processed/baumgardt_gc_catalog_with_origin_and_chemistry.csv`
- Outputs:
  - `outputs/figures/mg_al_gc_chemistry_diagnostic.pdf`
  - `outputs/figures/mg_al_gc_chemistry_diagnostic.png`
  - `outputs/tables/mg_al_gc_chemistry_diagnostic_summary.csv`
  - `outputs/tables/mg_al_gc_chemistry_diagnostic_summary.json`
  - `outputs/tables/mg_al_gc_chemistry_diagnostic_scores.csv`
- The common `Mg-Al` sequence is fit with an orthogonal-regression/PCA line in the
  plane:
  - `y = [Mg/Fe]`
  - `x = [Al/Fe]`
- For separation power, compared three chemistry scores against the BK
  in-situ/accreted labels:
  - `[Mg/Fe]` alone
  - `z = [Mg/Fe] - s[Al/Fe]`
  - 2D Fisher LDA in `([Al/Fe],[Mg/Fe])`
- Reported leave-one-out AUCs for the three scores.

### Results
- Number of GCs with both Mg and Al: `63`
  - in-situ: `40`
  - accreted: `23`
- Best-fit common slope:
  - `s = d[Mg/Fe]/d[Al/Fe] = +0.158`
- Intercept:
  - `[Mg/Fe] = 0.192 + 0.158 [Al/Fe]`
- Orthogonal scatter around the common line:
  - standard deviation: `0.0897 dex`
  - robust MAD scatter: `0.0855 dex`
- Variance fraction along the first principal axis:
  - `0.868`

### Separation power
- Raw AUCs:
  - `[Mg/Fe]` alone: `0.726`
  - `z = [Mg/Fe] - s[Al/Fe]`: `0.610`
- Leave-one-out AUCs:
  - `[Mg/Fe]` alone: `0.672`
  - `z = [Mg/Fe] - s[Al/Fe]`: `0.492`
  - 2D Fisher LDA in `(Al, Mg)`: `0.775`

### Interpretation
- In this **cluster-level mean abundance** sample, the common `Mg-Al` trend is
  shallow and positive, not a strong anti-correlation.
- Therefore the projected residual
  - `z = [Mg/Fe] - s [Al/Fe]`
  is **not** a good 1D discriminator here.
- `[Mg/Fe]` alone does carry useful information.
- But `[Mg/Fe]` alone is **not fully sufficient**:
  - the 2D `(Mg,Al)` score improves the leave-one-out AUC by `0.103`
  - so Al adds non-negligible information on top of Mg
- Therefore the simplest useful chemical model is probably **not**
  `z = [Mg/Fe] - s[Al/Fe]`, but rather:
  - `[Mg/Fe]` alone as a 1D first-pass marker, or
  - a very low-parameter 2D chemistry model if we want to keep Al explicitly

### Verification
- Re-ran:
  - `python -m compileall scripts/diagnose_mg_al_gc_chemistry.py`
  - `python scripts/diagnose_mg_al_gc_chemistry.py`

## 2026-05-22: Minimal blind 2-component Mg-only chemistry model

### Request
- Build the **minimal useful** chemistry-augmented 2-component latent model using
  only `[Mg/Fe]` as the chemical discriminator.
- Keep the single-component inference machinery fixed:
  - fixed IMF `phi(log M_ini)`
  - fixed total radial profile `A_tot(log a)`
  - fixed survivability `S(log M_ini, a)`
  - fixed detectability `Q(log M_ini, a)` from the final single-component
    `|l|`-aware EM run
- Add only the smallest extra structure needed to represent two populations.

### Model implemented
- Added:
  - `src/globular_clusters_imf/blind_mg_only_minimal_model.py`
  - `scripts/run_blind_mg_only_minimal_model.py`
- The baseline is the current best single-component model:
  - `schechter + logpoly3`
  - `N0 = 795.08`
  - `alpha = -0.940`
  - `log10(M_c/Msun) = 6.300`
- On top of that fixed baseline, fit only:
  - a bounded logistic mixing law in semimajor axis
    - `w(a) = eps + (1-2 eps) sigmoid(c0 + c1 z_a)`
    - `eps = 0.05`
    - `z_a = (log10 a - mean) / std`
    - `c1 <= 0` so the concentrated component is more important at small `a`
  - a 1D Mg chemistry model
    - concentrated component: `N(mu_c, sigma^2 + err_i^2)`
    - extended component: `N(mu_e, sigma^2 + err_i^2)`
    - shared intrinsic scatter `sigma`
- This gives:
  - single-chemistry null: `2` parameters
    - `mu`, `sigma`
  - two-component Mg model: `5` parameters
    - `c0`, `c1`, `mu_c`, `mu_e`, `sigma`

### Likelihood and comparison
- The comparison is **conditional on Mg being observed**, so only the `68`
  clusters with combined Mg measurements contribute non-constant chemistry
  likelihood.
- This avoids over-penalizing the chemistry model with the full catalog size.
- Saved outputs in:
  - `variants/blind_mg_only_minimal_mixture/outputs/tables/`
  - `variants/blind_mg_only_minimal_mixture/outputs/figures/`

### Results
- Best model:
  - `two_component_mg_mixture`
- Summary table:
  - `variants/blind_mg_only_minimal_mixture/outputs/tables/blind_mg_only_minimal_mixture_model_summary.csv`
- Key numbers:
  - 1-component Mg null:
    - `logL_chem = 65.919`
    - `BIC = -123.40`
  - 2-component Mg model:
    - `logL_chem = 75.143`
    - `BIC = -129.19`
- Therefore:
  - `Delta logL = +9.224`
  - `Delta BIC = -5.79`
- So this **minimal** Mg-only 2-component model is preferred over the Mg-only
  1-component null.

### Best-fit Mg-only two-component parameters
- Logistic mixing law:
  - `c0 = 1.162`
  - `c1 = -2.519`
  - bounded floor: `0.05`
- Chemistry:
  - concentrated mean:
    - `mu_[Mg/Fe],c = 0.272 dex`
  - extended mean:
    - `mu_[Mg/Fe],e = 0.131 dex`
  - shared intrinsic scatter:
    - `sigma_[Mg/Fe] = 0.0436 dex`
- Integrated mixture fractions over the fixed total radial profile:
  - concentrated:
    - `0.815`
  - extended:
    - `0.185`
- Corresponding initial counts:
  - concentrated:
    - `647.8`
  - extended:
    - `147.3`
- Expected observed counts:
  - concentrated:
    - `109.5`
  - extended:
    - `55.5`

### Radial interpretation
- The logistic crossover is at roughly:
  - `a ~ 10.0 kpc`
- Example concentrated fractions from the best-fit `w(a)`:
  - `a = 1 kpc`: `0.939`
  - `a = 3 kpc`: `0.867`
  - `a = 10 kpc`: `0.501`
  - `a = 30 kpc`: `0.150`
  - `a = 100 kpc`: `0.061`
- So both components span the full `a` range by construction, but the
  concentrated component dominates the inner Galaxy and the extended component
  becomes important beyond `~10 kpc`.

### Posterior behavior and BK comparison
- Posterior table:
  - `variants/blind_mg_only_minimal_mixture/outputs/tables/blind_mg_only_minimal_mixture_best_model_posterior_probabilities.csv`
- Posterior range:
  - `p_concentrated min = 3.07e-4`
  - `p_concentrated max = 0.999996`
- Correlations:
  - with `log10 a`: `-0.794`
  - with `log M_ini`: `+0.505`
- BK comparison:
  - `AUC = 0.892`
  - hard-assignment accuracy: `0.897`
  - mean `p_concentrated`
    - in-situ: `0.860`
    - accreted: `0.298`
- Hard assignments:
  - in-situ:
    - concentrated `102`
    - extended `5`
  - accreted:
    - concentrated `12`
    - extended `46`

### Interpretation
- This is the first blind chemistry model in the project that is both:
  - genuinely identifiable
  - and actually preferred by BIC
- The result suggests that a **small**, chemically informed 2-component model is
  much more effective than the earlier, more parameter-heavy chemistry models.
- Using only `[Mg/Fe]`, plus a simple monotonic radial mixing law, already gives
  a statistically useful latent split that aligns well with the BK labels.
- Because the total IMF and total radial profile are held fixed, this is best
  interpreted as a **population-splitting layer** on top of the established
  single-component inference, not yet as a new full refit of the IMF itself.

### Outputs
- Figure:
  - `variants/blind_mg_only_minimal_mixture/outputs/figures/blind_mg_only_minimal_mixture_diagnostic.pdf`
  - `variants/blind_mg_only_minimal_mixture/outputs/figures/blind_mg_only_minimal_mixture_diagnostic.png`
- Tables:
  - `variants/blind_mg_only_minimal_mixture/outputs/tables/blind_mg_only_minimal_mixture_model_summary.csv`
  - `variants/blind_mg_only_minimal_mixture/outputs/tables/blind_mg_only_minimal_mixture_model_summary.json`
  - `variants/blind_mg_only_minimal_mixture/outputs/tables/blind_mg_only_minimal_mixture_comparison_summary.json`
  - `variants/blind_mg_only_minimal_mixture/outputs/tables/blind_mg_only_minimal_mixture_best_model_posterior_probabilities.csv`
  - `variants/blind_mg_only_minimal_mixture/outputs/tables/blind_mg_only_minimal_mixture_best_model_component_radial_grid.csv`
  - `variants/blind_mg_only_minimal_mixture/outputs/tables/blind_mg_only_minimal_mixture_best_model_mg_density_grid.csv`

### Verification
- Re-ran:
  - `python -m compileall src/globular_clusters_imf/blind_mg_only_minimal_model.py scripts/run_blind_mg_only_minimal_model.py`
  - `.venv/bin/python scripts/run_blind_mg_only_minimal_model.py`

## 2026-05-23: Add [Fe/H] to the minimal Mg-only blind 2-component model

### Request
- Test whether adding metallicity `[Fe/H]` to the successful minimal Mg-only
  latent model improves the split **quantifiably**.
- Keep the same structure as before:
  - fixed single-component baseline:
    - shared IMF
    - shared total radial profile
    - fixed survivability `S`
    - fixed detectability `Q`
  - chemistry likelihood only for the `68` clusters with Mg measurements

### Model implemented
- Added:
  - `src/globular_clusters_imf/blind_mg_feh_minimal_model.py`
  - `scripts/run_blind_mg_feh_minimal_model.py`
- New chemistry model:
  - 1-component null:
    - `[Mg/Fe] ~ N(mu + b([Fe/H]-<Fe/H>), sigma^2 + err^2)`
    - parameters: `mu`, `b`, `sigma`
  - 2-component model:
    - same bounded logistic radial mixing law `w(a)`
    - concentrated mean `mu_c`
    - extended mean `mu_e`
    - shared metallicity slope `b`
    - separate intrinsic scatters `sigma_c`, `sigma_e`
    - parameters: `c0`, `c1`, `mu_c`, `mu_e`, `b`, `sigma_c`, `sigma_e`
- The local metallicity column used is:
  - `local_feh`
- Coverage:
  - all `68/68` Mg-bearing clusters also have `local_feh`

### Outputs
- Figure:
  - `variants/blind_mg_feh_minimal_mixture/outputs/figures/blind_mg_feh_minimal_mixture_diagnostic.pdf`
  - `variants/blind_mg_feh_minimal_mixture/outputs/figures/blind_mg_feh_minimal_mixture_diagnostic.png`
- Tables:
  - `variants/blind_mg_feh_minimal_mixture/outputs/tables/blind_mg_feh_minimal_mixture_model_summary.csv`
  - `variants/blind_mg_feh_minimal_mixture/outputs/tables/blind_mg_feh_minimal_mixture_model_summary.json`
  - `variants/blind_mg_feh_minimal_mixture/outputs/tables/blind_mg_feh_minimal_mixture_comparison_summary.json`
  - `variants/blind_mg_feh_minimal_mixture/outputs/tables/blind_mg_feh_minimal_mixture_best_model_posterior_probabilities.csv`

### Results within the Mg+[Fe/H] family
- Best model:
  - `two_component_mg_feh_mixture`
- 1-component Mg+[Fe/H] null:
  - `logL_chem = 65.9198`
  - `AIC = -125.84`
  - `BIC = -119.18`
- 2-component Mg+[Fe/H] model:
  - `logL_chem = 75.4528`
  - `AIC = -136.91`
  - `BIC = -121.37`
- Therefore within this family:
  - `Delta logL = +9.53`
  - `Delta AIC = -11.07`
  - `Delta BIC = -2.19`
- So the Mg+[Fe/H] 2-component model is still preferred over its own 1-component
  null.

### Direct comparison to the earlier Mg-only minimal model
- This is the key quantitative answer.
- From
  - `variants/blind_mg_feh_minimal_mixture/outputs/tables/blind_mg_feh_minimal_mixture_comparison_summary.json`
- Relative to the earlier Mg-only model:
  - 2-component model:
    - `Delta logL = +0.310`
    - `Delta AIC = +3.380`
    - `Delta BIC = +7.819`
  - 1-component null:
    - `Delta logL = +0.001`
    - `Delta AIC = +1.998`
    - `Delta BIC = +4.218`
- Therefore **adding [Fe/H] does not help enough to justify the extra
  parameters**.
- It improves the chemistry likelihood only marginally, while both AIC and BIC
  get worse relative to the simpler Mg-only model.

### Best-fit Mg+[Fe/H] parameters
- Shared metallicity slope:
  - `b = 0.0217 dex per dex`
- This is very small, i.e. the model finds only a weak Mg trend with `[Fe/H]`
  once the two populations are already allowed.
- Component means:
  - concentrated:
    - `mu_[Mg/Fe],c = 0.293`
  - extended:
    - `mu_[Mg/Fe],e = 0.158`
- Component scatters:
  - concentrated:
    - `sigma_[Mg/Fe],c = 0.0243`
  - extended:
    - `sigma_[Mg/Fe],e = 0.0579`
- Radial mixing:
  - `c0 = 0.0066`
  - `c1 = -0.9455`

### Effect on the inferred split
- The Mg+[Fe/H] model actually gives a **weaker** BK-aligned split than Mg-only:
  - Mg-only:
    - `AUC = 0.892`
    - hard-assignment accuracy `= 0.897`
  - Mg+[Fe/H]:
    - `AUC = 0.823`
    - hard-assignment accuracy `= 0.812`
- Mean posterior concentrated probability:
  - in-situ:
    - `0.626`
  - accreted:
    - `0.290`
- So the `[Fe/H]` extension softens the separation rather than sharpening it.

### Interpretation
- In this minimal latent model, `[Fe/H]` is **not** providing materially new
  information beyond `[Mg/Fe]`.
- The shared metallicity slope is close to zero, and the fit quality gain is too
  small to pay for the extra parameters.
- Therefore the current evidence is that:
  - the minimal Mg-only model is already capturing the useful chemistry signal
  - adding `[Fe/H]` is not worthwhile in this specific low-parameter framework

### Verification
- Re-ran:
  - `python -m compileall src/globular_clusters_imf/blind_mg_feh_minimal_model.py scripts/run_blind_mg_feh_minimal_model.py`
  - `.venv/bin/python scripts/run_blind_mg_feh_minimal_model.py`

## 2026-05-23: Mg-only blind 2-component model with split low-mass slopes

### Request
- Starting from the successful minimal Mg-only blind 2-component model, add a
  difference in the Schechter low-mass slope `alpha` between the concentrated
  and extended components.
- Keep:
  - fixed `S(M_ini,a)`
  - fixed `Q(M_ini,a)`
  - fixed total radial profile `A_tot(a)` from the best single-component model
  - fixed shared cutoff mass `M_c`
  - the same Mg means and shared Mg scatter structure

### Model implemented
- Added:
  - `src/globular_clusters_imf/blind_mg_only_split_alpha_minimal_model.py`
  - `scripts/run_blind_mg_only_split_alpha_minimal_model.py`
- Comparison family:
  - `single_mg_gaussian`
  - `two_component_mg_mixture`
  - `two_component_mg_split_alpha_mixture`
- Shared-`alpha` models use the baseline Schechter slope:
  - `alpha = -0.940`
  - `log10(M_c/Msun) = 6.300`
- Split-`alpha` model adds only:
  - `alpha_concentrated`
  - `alpha_extended`
- The likelihood for the split-`alpha` model is defined on **all 165 observed
  clusters**:
  - clusters with Mg use `(a, M_ini, Mg)`
  - clusters without Mg use `(a, M_ini)` only
- So unlike the earlier Mg-only minimal model, the split-`alpha` extension uses
  the full observed mass distribution to constrain the latent classes.

### Outputs
- Figure:
  - `variants/blind_mg_only_split_alpha_minimal_mixture/outputs/figures/blind_mg_only_split_alpha_minimal_mixture_diagnostic.pdf`
  - `variants/blind_mg_only_split_alpha_minimal_mixture/outputs/figures/blind_mg_only_split_alpha_minimal_mixture_diagnostic.png`
- Tables:
  - `variants/blind_mg_only_split_alpha_minimal_mixture/outputs/tables/blind_mg_only_split_alpha_minimal_mixture_model_summary.csv`
  - `variants/blind_mg_only_split_alpha_minimal_mixture/outputs/tables/blind_mg_only_split_alpha_minimal_mixture_comparison_summary.json`
  - `variants/blind_mg_only_split_alpha_minimal_mixture/outputs/tables/blind_mg_only_split_alpha_minimal_mixture_shared_alpha_posterior_probabilities.csv`
  - `variants/blind_mg_only_split_alpha_minimal_mixture/outputs/tables/blind_mg_only_split_alpha_minimal_mixture_split_alpha_posterior_probabilities.csv`
  - `variants/blind_mg_only_split_alpha_minimal_mixture/outputs/tables/blind_mg_only_split_alpha_minimal_mixture_split_alpha_imf_grid.csv`

### Results
- Best model in this comparison:
  - `two_component_mg_split_alpha_mixture`
- Shared-`alpha` two-component model:
  - `joint logL = -159.01`
  - `BIC = 343.55`
- Split-`alpha` two-component model:
  - `joint logL = -53.72`
  - `BIC = 143.19`
- Therefore:
  - `Delta logL = +105.29`
  - `Delta BIC = -200.36`

### Best-fit split-`alpha` parameters
- Chemistry and radial split:
  - `c0 = 1.522`
  - `c1 = -3.640`
  - `mu_[Mg/Fe],conc = 0.269`
  - `mu_[Mg/Fe],ext = 0.130`
  - shared `sigma_[Mg/Fe] = 0.0463`
- IMF:
  - concentrated:
    - `alpha_concentrated = -0.200`
      - this sits on the allowed upper bound
  - extended:
    - `alpha_extended = -0.640`
  - shared cutoff:
    - `log10(M_c/Msun) = 6.300`

### Posterior behavior
- Posterior range:
  - `p_concentrated min = 6.39e-5`
  - `p_concentrated max = 0.999995`
- BK comparison:
  - `AUC = 0.921`
  - hard-assignment accuracy `= 0.891`
  - mean `p_concentrated`
    - in-situ: `0.895`
    - accreted: `0.263`
- Hard assignments:
  - in-situ:
    - concentrated `102`
    - extended `5`
  - accreted:
    - concentrated `13`
    - extended `45`

### Interpretation
- Numerically, allowing `alpha` to differ makes the latent split much stronger.
- The split is informed not only by Mg, but also by the full observed mass
  distribution at fixed `a`.
- However, this result must be interpreted carefully:
  - this variant is still a **conditional latent-split layer** on top of the
    fixed single-component baseline
  - it is **not yet** a full two-component point-process refit of the catalog
  - therefore the huge `Delta BIC` should not be read as a final statement
    about the physical GC IMF without a more self-consistent generative model
- A warning sign is that:
  - `alpha_concentrated` hits the parameter bound `-0.2`
  which suggests the fit is trying to push toward an even shallower inner
  component.

### Practical takeaway
- As a latent classification model, adding split `alpha` is extremely effective.
- As a physical inference on the GCIMF, it is more tentative and probably calls
  for a proper next-step model where the two-component split is embedded directly
  in the full point-process likelihood rather than only in this conditional
  layer.

### Verification
- Re-ran:
  - `python -m compileall src/globular_clusters_imf/blind_mg_only_split_alpha_minimal_model.py scripts/run_blind_mg_only_split_alpha_minimal_model.py`
  - `.venv/bin/python scripts/run_blind_mg_only_split_alpha_minimal_model.py`

## 2026-05-23: Paper Section 7 updated with blind Mg-informed two-component models

### What changed
- Added a new blind Mg-informed subsection to Section 7 in
  `paper/main.tex` without removing the existing BK-labelled two-component
  analysis.
- Introduced a new paper figure:
  - `paper/figures/blind_mg_two_component_results.pdf`
- Introduced a new paper table:
  - `paper/tables/blind_mg_model_comparison.tex`
- Added Mg-model manuscript macros in:
  - `paper/tables/blind_mg_numbers.tex`
- Added a reusable asset builder:
  - `scripts/build_paper_section7_blind_mg_assets.py`

### Statistical footing
- The shared-IMF and split-`alpha` Mg models are now presented explicitly as
  **conditional latent decompositions** built on the fixed detectability-
  corrected single-component baseline.
- To make the two Mg variants internally consistent, the split-`alpha` summary
  was corrected in
  `src/globular_clusters_imf/blind_mg_only_split_alpha_minimal_model.py`
  so that it keeps the same fixed baseline:
  - total initial count `N0`
  - survivability fraction
  - detectability fraction
- This removes the misleading interpretation that the split-`alpha` conditional
  model had inferred a brand-new global `N0`.

### Numbers now reported in the paper
- Shared-IMF Mg model:
  - `Delta logL = +9.22`
  - conditional `Delta BIC = -3.13` relative to the 1-component Mg Gaussian
  - `mu_Mg,conc = 0.272`
  - `mu_Mg,ext = 0.131`
  - `sigma_Mg = 0.044`
  - `f_conc = 0.815`
  - implied counts at fixed total `N0 = 795.1`:
    - concentrated `647.8`
    - extended `147.3`
  - BK validation:
    - `AUC = 0.892`
    - hard-assignment accuracy `= 0.897`
- Split-`alpha` Mg model:
  - `Delta logL = +105.29`
  - conditional `Delta BIC = -200.36` relative to the shared-IMF Mg model
  - `alpha_conc = -0.200`
  - `alpha_ext = -0.640`
  - shared `log10(M_c/Msun) = 6.300`
  - `f_conc = 0.831`
  - implied counts at fixed total `N0 = 795.1`:
    - concentrated `660.4`
    - extended `134.6`
  - BK validation:
    - `AUC = 0.921`
    - hard-assignment accuracy `= 0.891`
- The paper text now states explicitly that the split-`alpha` result should be
  read as a conditional refinement of the latent split, not as a new
  self-consistent lower-bound reconstruction.

### Verification
- Re-ran:
  - `.venv/bin/python scripts/run_blind_mg_only_split_alpha_minimal_model.py`
  - `.venv/bin/python scripts/build_paper_section7_blind_mg_assets.py`
  - `cd paper && latexmk -pdf -bibtex -outdir=build main.tex`

## 2026-05-23: Tested a fixed `dN/dM ∝ M^-2` single-component IMF

### Motivation
- Added an explicit fixed-slope power-law IMF family to test the canonical
  extragalactic/simulation-inspired `dN/dM ∝ M^{-2}` hypothesis directly in
  the current single-component framework.
- Kept this test isolated from the main paper outputs by running it in a
  dedicated variant tree.

### Implementation
- Added `powerlaw_m2` support in:
  - `src/globular_clusters_imf/joint_model.py`
- This family:
  - has zero free IMF parameters
  - fixes `alpha_dndm = -2.0`
  - still allows the radial model to vary
- Added a dedicated runner:
  - `scripts/test_powerlaw_m2_single_component.py`
- The runner:
  - compares `lognormal`, free `powerlaw`, fixed `powerlaw_m2`, and
    `schechter`
  - runs both fixed-survival and longitude-aware detectability-corrected
    single-component fits
  - writes outputs under:
    - `variants/single_component_powerlaw_m2_test/outputs`

### Main detectability-corrected result
- Best overall model remains:
  - `schechter + logpoly3`
  - `logL = 471.72`
  - `BIC = -917.90`
  - `N0 = 795.1`
- Best fixed `M^-2` model is:
  - `powerlaw_m2 + logpoly3`
  - `logL = 453.45`
  - `BIC = -891.58`
  - `N0 = 3.13e4`
- Therefore, relative to the current best model:
  - `Delta logL = -18.27`
  - `Delta BIC = +26.33`
- So the canonical fixed `-2` power law is strongly disfavoured in the
  detectability-corrected single-component analysis.

### Comparison to the free power-law family
- In the full EM run, the best free power-law model came out as:
  - `powerlaw + logpoly3`
  - `alpha_dndm = -1.727`
  - `logL = 447.44`
  - `BIC = -874.45`
- This is formally worse than the fixed `-2` run.
- A nested sanity check showed that this is an EM-path issue rather than a
  true model-ordering statement:
  - holding the `powerlaw_m2` detectability solution fixed and refitting the
    free power-law within that same context gives
    - `alpha_dndm = -1.871`
    - `logL = 455.88`
  - so the free family does improve over the fixed `-2` model once evaluated
    in the same completeness context, as expected
- The main scientific conclusion is unchanged:
  - neither the fixed `-2` power law nor the free single power-law family
    matches the data as well as the Schechter single-component model

### Outputs
- Summary:
  - `variants/single_component_powerlaw_m2_test/outputs/tables/single_component_powerlaw_m2_summary.json`
- Detectability-corrected comparison table:
  - `variants/single_component_powerlaw_m2_test/outputs/tables/single_component_powerlaw_m2_detectability_comparison.csv`
- Free-power-law refinement candidates:
  - `variants/single_component_powerlaw_m2_test/outputs/tables/single_component_powerlaw_refinement_candidates.csv`
- IMF comparison figure:
  - `variants/single_component_powerlaw_m2_test/outputs/figures/single_component_powerlaw_m2_imf_comparison.pdf`

### Verification
- Re-ran:
  - `.venv/bin/python -m compileall src scripts/test_powerlaw_m2_single_component.py`
  - `.venv/bin/python scripts/test_powerlaw_m2_single_component.py`

## 2026-05-23: Profile-likelihood family scan for the single-component model

### Motivation
- The previous single-component family comparison relied on local multistart
  `L-BFGS-B` fits inside the detectability EM loop.
- The fixed `M^-2` test showed that the free power-law family could be
  under-optimized by that procedure.
- Added a profile-scan workflow to test the IMF families more thoroughly.

### Implementation
- Added fixed-IMF nuisance re-optimization helper:
  - `src/globular_clusters_imf/joint_model.py`
    - `fit_single_joint_model_with_fixed_imf_params(...)`
- Extended the longitude-aware detectability EM routine so it can:
  - hold IMF parameters fixed
  - re-optimize only the radial nuisance parameters
  - reuse radial and detectability warm starts between nearby grid points
- This was added in:
  - `src/globular_clusters_imf/detectability_longitude_model.py`
- Added a dedicated scan runner:
  - `scripts/run_single_component_family_profile_scan.py`
- The final successful scan is in:
  - `variants/single_component_family_profile_scan_v3`

### Scan design
- Used the empirically preferred radial family `logpoly3` for all three IMF
  families.
- Re-optimized the detectability-corrected single-component model at each grid
  point with fixed IMF parameters and warm starts from:
  - the best neighboring scan point
  - the unconstrained family-best EM solution
- Final scan grids:
  - power law: `alpha in [-2.6, -0.8]` with `25` points
  - lognormal: `mu in [4.7, 5.6]` and `sigma in [0.5, 0.9]` with `9 x 7`
  - Schechter: `alpha in [-2.1, -0.7]` and `log10(M_c/Msun) in [5.9, 6.7]`
    with `9 x 7`
- Used `6` EM iterations per fixed grid point.

### Main results
- Power-law family:
  - profile maximum at `alpha_dndm = -2.225`
  - `logL = 453.81`
  - `BIC = -887.20`
- Lognormal family:
  - profile maximum at `mu = 5.15`, `sigma = 0.70`
  - `logL = 466.61`
  - `BIC = -907.68`
- Schechter family:
  - coarse-grid maximum at `alpha = -1.05`, `log10(M_c/Msun) = 6.30`
  - `logL = 471.25`
  - `BIC = -916.97`
- Unconstrained best Schechter fit from the same `v3` reference comparison:
  - `alpha = -0.94`
  - `log10(M_c/Msun) = 6.3004`
  - `logL = 471.71`
  - `BIC = -917.89`

### Interpretation
- The family ordering is robust:
  - `Schechter` best
  - `lognormal` next
  - `single power law` much worse
- The important revision is for the power-law family:
  - the earlier free power-law EM fit near `alpha ~ -1.6` was not the true
    family maximum
  - the profile scan moves the best single-power-law slope to `alpha ~ -2.2`
  - this is close to the canonical extragalactic expectation
- But even after that correction, the power-law family remains strongly
  disfavoured relative to the Schechter family:
  - `Delta logL ~ -17.9`
  - `Delta BIC ~ +30.7` comparing the power-law profile maximum to the
    unconstrained Schechter best fit

### Outputs
- Summary JSON:
  - `variants/single_component_family_profile_scan_v3/outputs/tables/single_component_family_profile_scan_summary.json`
- Power-law profile:
  - `variants/single_component_family_profile_scan_v3/outputs/tables/powerlaw_profile_scan.csv`
- Lognormal surface:
  - `variants/single_component_family_profile_scan_v3/outputs/tables/lognormal_profile_scan.csv`
- Schechter surface:
  - `variants/single_component_family_profile_scan_v3/outputs/tables/schechter_profile_scan.csv`
- Summary figure:
  - `variants/single_component_family_profile_scan_v3/outputs/figures/single_component_family_profile_scan_summary.pdf`

### Verification
- Re-ran:
  - `.venv/bin/python -m compileall src scripts/run_single_component_family_profile_scan.py`
  - `.venv/bin/python scripts/run_single_component_family_profile_scan.py`

## 2026-05-30 - Figure 6 probability-density display update

### What changed
- Updated `scripts/build_paper_assets_exact_single_component.py` so
  `single_component_intensity_plane.pdf` now displays the fitted detected
  distribution as a relative probability-density map.
- Switched the image plane to greyscale with high values darker.
- Added three relative-density contours at 0.01, 0.1, and 0.5.
- Lightened the darkest greyscale values so the black contours remain visible.
- Changed the overplotted GC markers to black with no outlines.
- Updated `paper/main.tex` text and caption to describe the panel as relative
  detected probability density rather than observed point-process intensity,
  including the contour levels.

### Verification
- Regenerated only:
  - `paper/figures/single_component_intensity_plane.pdf`
- Rendered the figure to `/tmp/single_component_intensity_plane.png` for visual
  inspection.
- Rebuilt:
  - `make -C paper pdf`
- Checked `paper/build/main.log` for unresolved references/citations, overfull
  boxes, fatal errors, and rerun requests; none were found.

## 2026-05-30 - Figure 9 radial-profile panel update

### What changed
- Updated `scripts/build_paper_assets_exact_single_component.py` so
  `single_component_radial_profile.pdf` is now a two-panel radial-model
  comparison.
- Added a left panel showing the three fitted intrinsic radial profiles as
  `dN_0(>10^4 M_sun)/dlog10(a)`.
- Kept the observed-space semimajor-axis count comparison as the right panel,
  using the same fitted model colours and likelihood labels as before.
- Updated `paper/main.tex` to make Figure 9 a `figure*` and revised the
  caption to describe both panels.

### Verification
- Regenerated only:
  - `paper/figures/single_component_radial_profile.pdf`
- Rebuilt:
  - `make -C paper pdf`
- Checked `paper/build/main.log` for unresolved references/citations, overfull
  boxes, fatal errors, and rerun requests; none were found.
- Rendered the figure to `/tmp/single_component_radial_profile.png` for visual
  inspection.

## 2026-05-30 - Selection-factor mass-profile diagnostic and MCMC surface archive

### What changed
- Added a Section 4 diagnostic figure:
  - `paper/figures/survivability_detectability_mass_profiles.pdf`
- The figure uses the old exact single-component run currently used by the
  paper figures and shows best-fit mass projections of:
  - radial-profile-weighted survivability, `<S>_{rho(a)}`
  - survivor-weighted detectability, `<Q>_{rho(a)S}`
- Updated `paper/main.tex` to define the quoted mean detectability as
  `<Q>_{S,M>10^4}` and to state that it is weighted over the model-predicted
  surviving population, not over map area or only over observed clusters.
- Updated the single-component MCMC worker pipeline to archive retained
  posterior `S(logM, loga)` and `Q(logM, loga)` surfaces per chain into
  `chain_*_selection_surfaces.npz` files, aligned with the configured burn-in
  and thinning.
- Added `scripts/build_posterior_selection_surface_samples.py` as a fallback
  post-MCMC surface sampler for existing runs, but the preferred path is now
  native worker-time archiving.

### Verification
- Stopped the in-progress `logpoly3` single-component MCMC before restarting it
  with the new surface-archive code.
- Recompiled:
  - `.venv/bin/python -m py_compile scripts/run_profile_map_and_exact_mcmc_schechter_powerlaw_a.py scripts/run_parallel_exact_mcmc_from_existing_refined_grid.py scripts/build_paper_assets_exact_single_component.py scripts/build_posterior_selection_surface_samples.py`
- Regenerated:
  - `paper/figures/survivability_detectability_mass_profiles.pdf`
- Rendered the new figure to `/tmp/survivability_detectability_mass_profiles.png`
  for visual inspection.
- Rebuilt:
  - `make -C paper pdf`
- Checked `paper/build/main.log` for unresolved references/citations, overfull
  boxes, fatal errors, and rerun requests; none were found.

## 2026-05-30 - Figure 4 likelihood-coloured convergence update

### What changed
- Updated the Figure 4 plotting routine so curve colour is determined by the
  final profiled log-likelihood of each illustrative iterative-detectability
  correction solve.
- Set the predicted-observed-count panel limits to the 0.5th and 99.5th percentiles
  of the predicted observed counts in the plotted histories.
- Added `scripts/refresh_figure4_detectability_convergence.py`, which
  regenerates only Figure 4 and caches the illustrative iteration histories in:
  - `paper/tables/detectability_em_convergence_illustrative_results.pkl`
- Updated the Figure 4 caption to state that colours follow final profiled
  log-likelihood.

### Verification
- Recompiled:
  - `.venv/bin/python -m py_compile src/globular_clusters_imf/paper_assets.py scripts/refresh_figure4_detectability_convergence.py`
- Regenerated:
  - `paper/figures/detectability_em_convergence.pdf`
- Rendered the updated figure to `/tmp/detectability_em_convergence.png` for
  visual inspection.
- Rebuilt:
  - `make -C paper pdf`
- Checked `paper/build/main.log` for unresolved references/citations, overfull
  boxes, fatal errors, and rerun requests; none were found.

## 2026-05-23 - Paper update for single-component family profile scan

### What changed
- Updated the manuscript text in `paper/main.tex` so the single-component
  family comparison now explicitly uses the profile-likelihood scan rather than
  the older local-optimizer family summary.
- Replaced the old single-component comparison table with a profile-scan family
  summary in:
  - `paper/tables/single_component_model_comparison.tex`
  - `paper/tables/single_component_model_comparison.csv`
- Replaced the old single-component comparison figure with the precomputed
  family-scan summary figure in:
  - `paper/figures/single_component_model_performance.pdf`
- Updated the paper asset builder in
  `src/globular_clusters_imf/paper_assets.py` so future paper rebuilds pull the
  rigorous family-scan outputs from
  `variants/single_component_family_profile_scan_v3`.
- Updated `scripts/refresh_paper_single_component_abs_longitude.py` to use the
  same family-scan figure/table path.

### Scientific update recorded in the paper
- The best single-power-law slope is now stated as `alpha = -2.225`, close to
  the canonical `dN/dM ~ M^-2` expectation.
- Even at that profile maximum, the unbroken power-law family remains strongly
  disfavoured relative to Schechter:
  - `Delta logL = 17.9`
  - `Delta BIC = 30.7`
- The paper text now makes the correct distinction:
  - the data do not prefer a radically different single power-law slope
  - they disfavor the entire unbroken power-law family

### Verification
- Re-ran:
  - `.venv/bin/python -m compileall src scripts`
  - targeted regeneration of `paper/tables/single_component_model_comparison.*`
    and `paper/figures/single_component_model_performance.pdf`
  - `cd paper && latexmk -pdf -bibtex -outdir=build main.tex`

## 2026-06-01 - GitHub and Overleaf project setup

### What changed
- Initialized the local project as a Git repository on branch `main`.
- Created the private GitHub repository:
  - `https://github.com/vasilybelokurov/globular_clusters_imf`
- Pushed the curated project source to GitHub, including:
  - source package under `src/`
  - analysis and asset scripts under `scripts/`
  - processed catalogues under `data/processed/`
  - manuscript source, generated manuscript figures, and generated manuscript
    tables under `paper/`
- Tightened `.gitignore` to exclude local caches, virtual environments, build
  products, exploratory `tmp_*` directories, `outputs/`, and the bulky
  `variants/` directory.
- Cloned the newly created Overleaf project:
  - `https://www.overleaf.com/project/6a1d5a51e0122812de3774ce`
- Pushed a clean manuscript-only Overleaf layout containing:
  - `main.tex`
  - `references.bib`
  - referenced figure PDFs in `figures/`
  - generated table/macros inputs in `tables/`

### Verification
- Confirmed the GitHub repository was created as private and `main` was pushed.
- Locally compiled the exact Overleaf layout with:
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- Checked the Overleaf-layout compile log for fatal errors, undefined
  references/citations after reruns, rerun requests, and overfull boxes; none
  were found.
- Pushed the manuscript-only source to Overleaf `master`.

## 2026-06-01 - Track local GC FITS catalogues in repository

### What changed
- Added the local GC FITS snapshots to the repository under:
  - `data/raw/gc_catalog_updated.fits`
  - `data/raw/gc_catalog_pinsitu.fits`
- Changed the default catalogue paths in `src/globular_clusters_imf/catalog.py`
  from user-home locations to the tracked repo-local FITS files.
- Updated `README.md` so it no longer lists the old home-directory FITS paths
  as the default inputs.
- Regenerated the origin and chemistry summary CSV files so their source paths
  are recorded as repo-relative paths.

### Verification
- Re-ran:
  - `.venv/bin/python scripts/export_gc_origin_flags.py`
  - `.venv/bin/python scripts/export_gc_chemistry_markers.py`
  - `.venv/bin/python -m py_compile src/globular_clusters_imf/catalog.py scripts/export_gc_origin_flags.py scripts/export_gc_chemistry_markers.py`
- Confirmed no remaining live-code, README, script, or processed-data references
  to the old home-directory FITS paths.

## 2026-06-01 - Add model motivation paragraph

### What changed
- Added a short introductory paragraph at the start of Section 3 explaining the
  physical motivation for modelling the GC catalogue in $(M_{\rm ini},a)$.
- The paragraph connects the model to the Baumgardt initial-mass/orbit
  reconstruction and to the broader post-Gaia Galactic archaeology context.
- Added the Deason & Belokurov 2024 review to the bibliography.

### Verification
- Recompiled:
  - `make -C paper pdf`
- Checked `paper/build/main.log` for fatal errors, unresolved references or
  citations, rerun requests, and overfull boxes; none were found.
