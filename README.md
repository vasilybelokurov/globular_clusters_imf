# globular_clusters_imf

Reconstruct the shape of the Milky Way globular-cluster initial mass function using Baumgardt globular-cluster catalogs, while allowing the radial normalization of the population to vary with Galactocentric distance.

## Working idea

Baumgardt et al. (2019) show that the surviving Milky Way globular-cluster sample is strongly radius dependent:

- inner clusters survive only if they were initially massive
- outer clusters suffer less destruction but are few in number

This project treats the observed surviving catalog as a radius-dependent truncation of a universal initial-mass distribution:

- global IMF shape `phi(M | theta)`
- radial normalization `A(a)` as a function of orbital semimajor axis `a`
- survival threshold `M_cut(a)` that rises toward the Galactic centre

The first-pass model is intentionally simple and reproducible. It is meant to establish a clean inference baseline before moving to richer dynamical models.

## Current data source

The pipeline currently ingests the March 2023 Baumgardt online database:

- orbital parameters: `https://people.smp.uq.edu.au/HolgerBaumgardt/globular/orbits.html`
- structural parameters including `log M_ini`: `https://people.smp.uq.edu.au/HolgerBaumgardt/globular/parameter.html`

These are later than the 2019 paper catalog, but they expose the same key quantities needed for this project and are machine-readable. The 2019 paper is still the conceptual anchor for the modeling assumptions.

## Layout

- `scripts/fetch_baumgardt_catalog.py`: download, parse, and clean the Baumgardt tables
- `scripts/export_gc_origin_flags.py`: save the local in-situ/accreted GC classification into this project and join it onto the Baumgardt catalog
- `scripts/fit_gc_imf_model.py`: fit a first-pass truncated-IMF model and produce figures
- `src/globular_clusters_imf/joint_model.py`: fit fixed-survival joint point-process models in `(log M_ini, log a)`
- `src/globular_clusters_imf/`: reusable package code
- `data/processed/`: cleaned joined catalogs
- `outputs/figures/`: diagnostic plots
- `outputs/tables/`: fitted parameters and summaries
- `JOURNAL.md`: running project log

## Planned model

For a cluster with semimajor axis `a` and inferred initial mass `M_ini`, the observable surviving population is modeled as an inhomogeneous point process with intensity

`lambda(M, a) = A(a) * phi(M | theta) * I[M >= M_cut(a)]`

where:

- `phi(M | theta)` is a radius-independent IMF shape
- `A(a)` is the radial normalization of the initial cluster population
- `M_cut(a)` approximates the survival selection function

The first implementation fits either:

- a lognormal IMF in `log10(M)`
- a truncated power law in `M`

## Joint Fixed-Survival Model

The repository now also includes a proper joint likelihood fit in the plane of
`log10(M_ini)` and `log10(a)`, while treating the survival function `S(M_ini, a)`
as fixed input.

The fitted observed point-process intensity is

`lambda_obs(log M_ini, log a) = N0 * phi(log M_ini | theta) * A(log a | eta) * S(log M_ini, a)`

where:

- `phi(log M_ini | theta)` is the intrinsic IMF shape
- `A(log a | eta)` is the birth profile as a function of semimajor axis
- `S(log M_ini, a)` is the fixed survival map
- `N0` is the total initial number of clusters

Current implemented model families:

- IMF:
  - lognormal
  - power law
  - Schechter
- radial normalization:
  - `step5`: piecewise constant in five `log a` bins
  - `logpoly3`: smooth cubic polynomial in standardized `log a`

## Two-Component Fixed-Survival Model

The repository also now supports a two-component version of the joint model, using
the saved GC origin flag:

- `origin_flag = 1`: in-situ
- `origin_flag = 0`: accreted

In this model the class membership is fixed by the local catalog, while each
component gets its own:

- intrinsic IMF shape `phi_c(log M_ini | theta_c)`
- radial birth profile `A_c(log a | eta_c)`
- total initial normalization `N0_c`

The observed intensity for each component is

`lambda_obs,c(log M_ini, log a) = N0_c * phi_c(log M_ini | theta_c) * A_c(log a | eta_c) * S(log M_ini, a)`

The implementation fits the two labeled subsets independently with the same fixed
survival map, then combines them into a joint two-component score table. This keeps
the existing single-component model intact while allowing the in-situ and accreted
GC populations to have different IMF and radial-formation behavior.

The main fitting script now evaluates three model classes when `origin_flag` is
available:

- single population
- two components with a shared IMF and separate `A(a)`
- two components with separate IMFs and separate `A(a)`

and writes the class comparison table to:

- `outputs/tables/joint_fixed_survival_population_model_class_comparison.csv`

Important detail:

- the raw one- versus two-component likelihood comparison is not directly fair, because the two-component fit is evaluated on two fixed labelled subsets (`in-situ` and `accreted`)
- for paper-level model comparison, use the corrected conditional comparison written to:
  - `paper/tables/population_model_class_comparison.csv`
- under that corrected comparison:
  - the shared-IMF two-component model is strongly preferred over the single-population model, conditional on the adopted labels
  - the separate-IMF two-component model is worse than the shared-IMF model by `Delta BIC = 7.81`
- so the data support different radial birth profiles for the two populations, but do not support distinct IMF shapes

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run

```bash
source .venv/bin/activate
python scripts/fetch_baumgardt_catalog.py
python scripts/export_gc_origin_flags.py
python scripts/fit_gc_imf_model.py
```

## Paper Draft

The repository now also contains a local MNRAS manuscript project under `paper/`.
Paper-only figures and tables are regenerated from the fitted models with:

```bash
source .venv/bin/activate
python scripts/build_paper_assets.py
```

To compile the manuscript locally:

```bash
cd paper
make pdf
```

The current compiled draft is written to:

- `paper/build/main.pdf`

## Local GC Origin Flags

The project can snapshot the local Milky Way GC in-situ/accreted classification into
repo-managed CSV files. By default the export looks for:

- `~/data/catalogues/gc_catalog_updated.fits`
- `~/Documents/Work/lists/gc_catalog_pinsitu.fits`

The first file provides the current binary `FLAG`; the second, when present, adds the
older `INSITU_PROB` values. You can override the source paths with:

- `GC_IMF_UPDATED_CATALOG_PATH`
- `GC_IMF_PINSITU_CATALOG_PATH`

Running `python scripts/export_gc_origin_flags.py` writes:

- `data/processed/gc_origin_flags.csv`
- `data/processed/baumgardt_gc_catalog_with_origin_flags.csv`

The main fit script now prefers `data/processed/baumgardt_gc_catalog_with_origin_flags.csv`
when it exists, so the two-component model can run automatically.
