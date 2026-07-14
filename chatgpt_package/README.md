# A fixed-point iteration with no fixed point

I have an alternating ("EM-like") iteration that estimates a selection/completeness
function jointly with a population model. It does not converge. I want a second opinion
on whether my diagnosis is right, and on what the correct fix is.

---

## 1. The setup

I observe `N_obs = 165` objects. I model the observed population as an inhomogeneous
Poisson point process over `(log M, log a)`:

```
lambda_obs(log M, log a) = N0 · phi(log M | theta) · A(log a | eta) · S_sel(log M, a)
```

* `phi` — intrinsic mass function (Schechter; its two parameters are **held fixed** in
  the run below — they are a grid node, not fitted here)
* `A` — radial profile (cubic polynomial in standardised `log a`, 3 free parameters)
* `S_sel = S_survival · C` — the selection map: a **known, fixed** survival probability
  `S_survival(log M, a)` multiplied by a **detection completeness** `C`
* `C` — logistic in three observables (present-day mass, heliocentric distance,
  |galactic latitude|): `C = expit(intercept + b1·z_mass − b2·z_dist + b3·z_lat)`.
  **4 free parameters.**
* `N0` — total number of objects ever formed. **This is what I actually want.**

I do not observe `C` independently. I am trying to infer it from the data at the same
time as the population.

## 2. The algorithm (what the code does)

```
repeat:
    1.  S_sel      = S_survival · C                      # build selection map
    2.  fit phi, A by maximum likelihood given S_sel
        N0         = N_obs / selection_fraction          # <<< see below
           where selection_fraction = ∫∫ phi · A · S_sel
    3.  P          = N0 · (population mapped into observable bins)   # "complete" counts
    4.  fit C's 4 parameters by Poisson likelihood on:
           mu      = P · C
           nll     = −Σ [ n_obs · log(mu) − mu ]
until done
```

It was run for a **fixed 12 iterations** with no convergence test, and the 12th iterate
was reported as the answer.

## 3. My diagnosis

Step 2 defines `N0` so that the model *always* reproduces the observed count, whatever
`C` happens to be. Now substitute `C → k·C`:

| quantity | scales as |
|---|---|
| `selection_fraction = ∫∫ phi·A·S_survival·C` | `k · selection_fraction` |
| `N0 = N_obs / selection_fraction` | `N0 / k` |
| `P = N0 · (…)` | `P / k` |
| **`mu = P · C`** | `(P/k)·(k·C)` = **unchanged** |

`mu` is invariant. The Poisson likelihood in step 4 is therefore **flat in the overall
scale of `C`**, so step 4 carries no information about it. The iteration has no fixed
point in that direction and simply drifts along it.

It is not *exactly* flat, because `C = expit(...)` saturates at 1 and so cannot be
rescaled with perfect freedom. That weak restoring force is (I think) why it drifts
slowly over hundreds of iterations rather than running away immediately — and why
stopping at 12 iterations produced a plausible-looking number.

**Consequence:** `N0` is a function of `n_iterations`, which is a tuning constant, not
of the data.

## 4. The evidence

Measured from the real code (`evidence/real_code_trajectory.csv`), Schechter IMF with
α = −1.675, log₁₀M_c = 6.35 held fixed:

| iteration | logL | N₀ | mean C | N₀ × meanC |
|---|---|---|---|---|
| 1 | 449.2 | 1.84e5 | 0.733 | 1.35e5 |
| **12** ← *the code stopped here* | 462.2 | **2.95e5** | 0.431 | 1.27e5 |
| 50 | 458.9 | 4.06e5 | 0.198 | 8.1e4 |
| 100 | 458.8 | 4.97e5 | 0.128 | 6.4e4 |
| 200 | 459.0 | **5.65e5** | 0.108 | 6.1e4 |

`N₀` climbs monotonically, mean completeness collapses monotonically, and **the
log-likelihood is flat after ~iteration 50** (458.78 → 458.95 over 150 iterations). It
is still moving at 200. It never converges.

`code/02_minimal_reproduction.py` is a standalone ~120-line numpy/scipy script (no
project dependencies) that reproduces the flat direction exactly:

```
     k   selection_fraction             N0      sum(mu)        mu[3]
  1.00             0.177480          929.7     165.0000    13.031654
  0.50             0.088740         1859.4     165.0000    13.031654
  0.25             0.044370         3718.7     165.0000    13.031654
  0.10             0.017748         9296.8     165.0000    13.031654
```

`N0` changes by 10×; `mu` does not change at all.

## 5. What I want to know

1. **Is the diagnosis right?** Is the completeness amplitude genuinely unidentified
   here, or have I missed a term that does constrain it?

2. **Is the weak identification through `expit` saturation real, or an artefact?** The
   only thing preventing an exact rescaling degeneracy is the logistic link's ceiling at
   1. If `N0` is identified *only* through the functional form of the link and not
   through information in the data, is any reported `N0` meaningful? Would a different
   link (probit, complementary log-log) give a different `N0` from the same data? (I
   suspect yes, which would settle it.)

3. **What is the right fix?** Candidates I see:
   - **(a)** Declare the amplitude. Fix the mean completeness (or the completeness in a
     reference bin) by fiat and report `N0` as explicitly conditional on it, with a
     sensitivity scan. Honest, but gives up on measuring `N0`.
   - **(b)** Supply `C` externally from a real survey selection function, and stop
     trying to infer it from the same counts.
   - **(c)** Add a prior / penalty on the completeness parameters. Feels arbitrary — it
     would be choosing the answer.
   - **(d)** Re-parameterise so the identified combination (`N0 × meanC`, which *is*
     pinned — see the last column of the table) is what gets reported, and admit `N0`
     itself is not estimable.
   - **(e)** Something I have not thought of.

4. **Is `N0 = N_obs / selection_fraction` itself the mistake?** It is what injects the
   degeneracy. Should `N0` instead be a free parameter in a proper joint likelihood
   (with the Poisson normalisation term `−N0·selection_fraction` retained rather than
   profiled out), and would that actually identify it, or would it just relocate the
   flat direction?

5. **Is there a diagnostic I should have run** that would have caught this immediately,
   short of noticing the iteration never settles? (Profile likelihood in the completeness
   intercept? Rank of the Fisher information / a near-zero eigenvalue?)

## 6. What is in this package

```
README.md                              this file
code/01_the_actual_loop_verbatim.py    the real source, all 12 functions in the loop,
                                       extracted verbatim and in reading order
code/02_minimal_reproduction.py        standalone, numpy+scipy only, runs in seconds,
                                       demonstrates the flat direction and the drift
evidence/real_code_trajectory.csv      the measured 200-iteration trajectory above
```

Start with `02_minimal_reproduction.py` — it is short and it is the whole problem.
`01_the_actual_loop_verbatim.py` is there if you want to check that the toy faithfully
represents the real thing.

## 7. Context that may matter

* `N_obs = 165`. Small.
* `S_survival` is treated as **known and fixed** (it comes from a separate model). It is
  not fitted, so it is not the source of the degeneracy.
* The completeness slopes are exponentiated (`exp(raw)`), forcing completeness to
  increase with mass, decrease with distance, increase with |b|. Monotonicity is imposed,
  not inferred.
* The population's spatial distribution given `a` is assumed **isotropic on a shell of
  radius `a`**. Since the completeness has a latitude slope, any real latitude structure
  in the population would be absorbed into `C` — a second worry, separate from the one
  above, and I would like a view on whether it makes the identification problem worse.
* The three fitted quantities in the loop are: 4 completeness parameters, 3 radial
  parameters, and `N0` (which is determined, not free).
