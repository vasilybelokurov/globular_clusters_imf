# Results of the diagnostics you prescribed

I ran all four. **Every one of your claims is confirmed.** Two of my original
statements were wrong and I withdraw them. One new quantitative result falls out that I
think sharpens the picture, and I would like your view on it (Test 4).

Scripts: `code/03_diagnostics.py`, `code/04_regime_dependence.py`.
All use the same toy as `02_minimal_reproduction.py` (12 bins, `N_obs = 165`, fixed
survival 0.25, Gaussian population shape, logistic completeness `expit(a + b·x)`,
`x ∈ [−2, 2]`).

---

## Test 0 — my first attempt, which was circular (withdrawn)

I first tried to demonstrate the ridge by making `log N0` a free parameter alongside the
*logistic* completeness and optimising from four widely separated starting intercepts
(−3, −1, +1, +3). Result:

```
 start intercept  fitted logN0    fitted N0  intercept          nll
            -3.0        6.8348        929.7     1.0000  -300.829163
            -1.0        6.8348        929.7     1.0000  -300.829163
             1.0        6.8348        929.7     1.0000  -300.829163
             3.0        6.8348        929.7     1.0000  -300.829163
```

All four converge to the *same* `N0` and recover the true intercept exactly. This looks
like clean identification — and it is worthless as evidence, for two reasons you had
already given:

1. the logistic family is not closed under multiplication, so it breaks the exact ridge;
2. the toy data were **generated** from that exact logistic, so the model is perfectly
   specified and the fit is circular.

I include it (`code/05_failed_first_attempt.py`) only to be explicit that this test does
*not* refute the degeneracy, in case the same reasoning tempts anyone else.

---

## Test 1 — explicit amplitude, profile over `a`

Your prescription: reparameterise `C(x) = a · C̃(x; γ)` with `C̃` normalised to
`max C̃ = 1`, and profile over the amplitude `a`.

For each fixed `a` I set `N0 = N_obs / q(a)` (the analytic MLE) and evaluated the full
unprofiled Poisson log-likelihood.

```
       a       N0_hat    profile nll   delta nll
    1.00        865.4    -300.829163    0.00e+00
    0.50       1730.8    -300.829163    0.00e+00
    0.20       4327.0    -300.829163    0.00e+00
    0.10       8654.1    -300.829163    0.00e+00
    0.05      17308.1    -300.829163    0.00e+00
    0.02      43270.3    -300.829163    0.00e+00
```

**`N0` spans 50× (865 → 43,270). `Δ(nll) = 0.00e+00` — flat to machine precision.**

The ridge is exact, as you said. My earlier test missed it purely because the logistic
link conceals it.

---

## Test 2 — Hessian in `(log N0, log a)`

```
  eigenvalues     : -7.105e-07    3.300e+02
  condition number:  4.644e+08
  null eigenvector: (-0.7071, +0.7071)
```

You predicted a null direction `∝ (1, −1)` in `(δ log N0, δ log a)`. Observed:
`(−0.7071, +0.7071)` = `(1, −1)/√2` up to an irrelevant overall sign.

The smallest eigenvalue is `−7×10⁻⁷` — numerically zero, and negative, which is the
expected finite-difference noise on an exactly singular direction. Condition number
`4.6×10⁸`.

Exactly the signature you described.

---

## Test 3 — link-function sensitivity

Same data, three links, `N0` free.

```
      link    fitted N0          nll  max|pred-obs|
     logit        929.7   -300.82916       2.67e-07
    probit        902.7   -300.82913       1.44e-02
   cloglog        853.5   -300.82439       1.60e-01
```

All three reproduce the observed counts essentially perfectly, yet give different `N0`.
Your diagnostic works — but the effect here is only ~9%, which is far weaker than the
argument implies. That bothered me, so:

---

## Test 4 — the regime dependence (new; this is what I want your view on)

Your point 3 says the ridge is broken *only near the logistic ceiling*, and that when
the probabilities are well below one, `expit(z) ≈ e^z`, which **is** closed under
rescaling, so the ridge becomes near-exact.

That implies the link sensitivity should be a strong function of the completeness
regime. My toy sits at mean `C ≈ 0.69` — high. The real code drifts down to
mean `C = 0.108`.

So I regenerated the data at a range of true completeness levels and refitted with each
link:

```
 true mean C |  N0(logit)  N0(probit)  N0(cloglog) |   spread
------------------------------------------------------------------
       0.931 |        701         696          692 |    1.01x
       0.693 |        930         903          853 |    1.09x
       0.400 |       1690        1841         1335 |    1.38x
       0.157 |       4720       18554         2976 |    6.23x
       0.044 |      17853   153740201         9610 | 15998.62x
```

At high completeness the link choice is nearly irrelevant. **As completeness falls the
identification collapses**: at mean `C = 0.044` the three links disagree by a factor of
16,000 (probit returns `1.5×10⁸`), while all of them fit the observed counts.

The part I find most striking, and want to check with you:

**The iteration walks itself into that regime.** In the real code, mean completeness goes
`0.73 → 0.43 (iter 12) → 0.20 (iter 50) → 0.108 (iter 200)`, monotonically. So the
algorithm drifts *from* the regime where `N0` is weakly identified *into* the regime
where it is essentially unidentified — and the further it goes, the flatter the ridge
becomes and the less anything opposes further drift.

That reads to me as self-reinforcing: the weak restoring force from the logistic ceiling
is strongest at high `C` and vanishes as `C → 0`, so the iteration has a one-way
direction to travel. It would explain why it never turns around and why it is still
moving at 200 iterations.

**Question: is that the right reading?** Specifically:

1. Is the drift direction (towards low `C`, high `N0`) actually *forced* by the geometry
   — i.e. is the restoring force strictly monotone in `C` — or is the direction merely a
   consequence of my starting point and the damping (`relaxation = 0.7`)?
2. If it is forced, then even a converged fit at a "reasonable" mean completeness would
   be untrustworthy, because the only thing holding it there is the link's ceiling. Does
   that change your recommendation, or is it just further support for (b) external
   calibration?
3. Is there a reparameterisation in which the pathology is *visible* rather than latent
   — e.g. always fitting `C = a · C̃` with `a` explicit, so that the flat direction is a
   named parameter and the singular Hessian shows up in any standard error?

---

## Corrections to my original statement

For the record, two things I said in the README are **wrong**:

1. **"`N0 = N_obs / selection_fraction` injects the degeneracy."** Wrong. It is the
   analytic profile-MLE of `N0` along a degeneracy that is already present in the model.
   Confirmed by Test 1: making `N0` free changes nothing; the ridge is in the
   observational model, not the estimator.

2. **"`N0 × meanC` is the identified combination."** Wrong. It is invariant only under an
   *exact* rescaling with everything else held fixed. In the real code it moved from
   `1.35×10⁵` to `6.11×10⁴` over the run, because the population shape and the weighting
   both change between iterations. You spotted this from my own table, which I had
   glossed over. The identified object is the detected intensity, and its integral is
   just `N_obs = 165`.

I also had not appreciated your point 4 — that the two steps do not ascend a common
objective, so this is not a valid EM and monotone improvement was never guaranteed. That
is the actual explanation for the log-likelihood *decreasing* (462.2 → 458.8), which I
had wrongly attributed to the flat direction alone.

## What I am changing in the code

- raise on a missing completeness anchor rather than returning a number off the ridge;
- rename outputs to `total_initial_count_conditional_on_mean_completeness`, and record
  `assumed_mean_completeness` plus the explicit weighting measure `w(x)` used to define
  it (your point that "mean completeness" is ambiguous without it is well taken);
- keep the convergence checks but demote them in the documentation to defensive
  programming, not the statistical fix;
- withdraw the 12-iteration output.

Not yet done, and I would value your view on priority: fitting `C = a · C̃` with an
explicit amplitude so the singular direction is a named parameter (Test 4, question 3).
