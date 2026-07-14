# Tables that are NOT used by the manuscript

`main.tex` cites macros from `paper_numbers.tex` and `two_component_numbers.tex`, and
`\input`s a handful of `.tex` tables. Some generated files here are **not** referenced by
the manuscript at all. They are diagnostics from alternative code paths.

Do not quote numbers from them.

## `key_results_summary.csv` / `key_results_summary.tex`

Contains a row "Detectability-corrected single component" with

    alpha = -1.675, log10 Mc = 6.35, N0 = 114361.7, selection_fraction = 6.4e-4

**This is not the paper's result and must not be quoted.** It comes from the
abs-longitude family profile scan, evaluated at a grid node in the low-likelihood region.
At those parameters the inner detectability iteration does not reach a fixed point: mean
completeness drifts 0.73 -> 0.43 (iteration 12) -> 0.11 (iteration 200) while N0 climbs
1.8e5 -> 5.7e5, and the log-likelihood is flat. The quoted N0 is simply the value at
whichever iteration the loop stopped.

The paper's actual single-component result is the profiled-MCMC posterior, quoted through
the `Exact*` and `Posterior*` macros in `paper_numbers.tex`:

    eta_t = 1.147, alpha = -1.169, log10 Mc = 6.284
    N0(>1e4 Msun) = 1722 (+1479 / -717)
    mean detectability = 0.811

At *those* parameters the detectability iteration converges in 17 iterations with 0.0 per
cent drift after iteration 12 -- which is precisely the behaviour the paper describes in
Sec. 4.1: drifting trials are low-likelihood and are rejected by the outer profile
likelihood, not removed by hand.

The two numbers differ by a factor of ~66. They are not alternative estimates of the same
thing; one is a converged fit and the other is an unconverged iterate at bad parameters.

## `paper_numbers.tex`: orphaned macros

`\DetectabilityCorrectedNzero`, `\DetectabilityMeanCompleteness` (= 0.037),
`\DetectabilityCountRatio` (= 32.81) and `\SingleComponentNzero` are defined but cited
**zero times** in `main.tex`. They are the same low-likelihood scan artefacts. Leave them
unused.
