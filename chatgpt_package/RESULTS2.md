# Round 2 — your corrections tested

You were right on every point, and my Test 4 interpretation was wrong. I ran the
update-map diagnostic you specified. It refutes my claim cleanly. There is also one
result I **cannot** explain, which I think is now the real open question.

Scripts: `code/06_update_map.py`, `code/07_shape_refit.py`, `code/08_probit_plateau.py`.

---

## 1. My "the geometry forces the drift" claim is WRONG — refuted

Your prescription: one full outer iteration is a deterministic map `T(ψ)`. Take the
amplitude coordinate, compute the **undamped** update `Δu(u) = T_u(u) − u` over a grid,
re-optimising everything else as the real loop does. Sign constant ⇒ forced drift. Sign
change ⇒ a fixed point exists.

Result (toy, shape held fixed, relaxation = 1):

```
 intercept_in  meanC_in |  d(intercept)   sign
         4.00    0.9724 |       -0.7101   down
         3.00    0.9308 |       -0.1055   down
         2.00    0.8425 |       +0.0961   up
         1.00    0.6932 |       +0.0000   FIXED POINT
         0.00    0.5000 |       +0.0034   up
        -2.00    0.1575 |       +0.1647   up
        -4.00    0.0276 |       +0.2291   up
```

**The sign changes.** There is an attracting fixed point, exactly at the truth. The map
does not force downward motion — from low completeness it pushes *back up*.

So: flattening ≠ directional forcing, precisely as you said. I withdraw the claim. The
correct statement is yours: *"the alternating update has a systematic component toward
low C; once it enters the low-C regime, the likelihood supplies progressively less
curvature to oppose or correct that motion."* And even that I can no longer demonstrate
— see §3.

## 2. Test 3 wording — you are right, and I have fixed the script

My script printed "the fitted observed counts agree to ~1e-5". Its own output showed:

```
     logit   max|pred-obs| = 2.7e-07
    probit   max|pred-obs| = 1.4e-02
   cloglog   max|pred-obs| = 1.6e-01
```

Only logit agrees at that level. The conclusion line contradicted the numbers directly
above it. Corrected to your wording: *the links give nearly indistinguishable
likelihoods (ΔNLL ≈ 3.3e-5 probit, 4.8e-3 cloglog), although their binwise predictions
are not numerically identical.*

## 3. The result I cannot explain — and I think this is now the real question

You said the drift comes from the update map, not the geometry. Fine. **But my toy's
update map has a fixed point, and the real code drifts monotonically for 200+
iterations.** So the toy does not reproduce the pathology, and I should not pretend it
does.

I suspected the difference was that the toy holds the population **shape** fixed whereas
the real loop **refits** it (IMF + radial parameters) every iteration — one of the
transverse perturbations you listed. So I gave the toy a free shape parameter and refit
it each iteration exactly as the real loop does:

```
                 SHAPE FIXED          SHAPE REFIT EACH ITERATION
 intercept_in |  d(intercept)    |    d(intercept)
         4.00 |      -0.7101     |       +1.1862
         2.00 |      +0.0961     |       +0.0903
         1.00 |      +0.0000  FP |       +0.0000  FP
         0.00 |      +0.0034     |       -0.0049
        -1.00 |      +0.0892     |       -0.0004
        -2.00 |      +0.1647     |       +0.0006
        -3.00 |      +0.2087     |       +0.0004
        -4.00 |      +0.2291     |       +0.0002
```

Refitting the shape does **not** destroy the fixed point either. But note what it *does*
do: in the low-completeness region the update collapses to `~1e-4` — the map becomes
**essentially the identity**. No restoring force, in either direction. That is your third
bullet: *"Δu ≈ 0 but numerical trajectories drift → tolerances or numerical effects
dominate."*

So my honest position:

* the ridge is exact and structural (Tests 1–2, confirmed);
* the drift direction is **not** forced by the likelihood geometry (your correction, now
  verified);
* **and I cannot reproduce the real code's persistent one-way drift in a correctly
  specified toy at all.** Both toys have a fixed point at the truth.

That leaves **model misspecification** as the leading candidate: in the real problem the
completeness model cannot actually fit the observed structure, so `C` keeps absorbing
residual mismatch, and there is nothing to stop it because the ridge supplies no
curvature. Candidates for the misspecification, all present in the real code and absent
from the toy:

* the assumed **isotropic shell** geometry (population placed uniformly on a sphere of
  radius `a`) — you already flagged that intrinsic angular structure would be absorbed
  into the latitude slope;
* the **present-mass proxy**: observable mass is a fitted deterministic function of
  `(M_ini, a)` plus Gaussian scatter, itself fitted to the same 165 objects;
* **exponentiated slopes**, which impose monotonicity in mass/distance/|b|;
* the smearing of the population into observable bins via `mass_bin_probabilities_grid`.

**Question: is misspecification the right diagnosis for a persistent one-way drift on a
flat ridge, and is there a clean way to test it?** My instinct is a
posterior-predictive/residual check on the observable histogram at the converged-ish
iterate — if the model cannot reproduce the observed `(mass, distance, |b|)` structure,
that is the driver. But I would rather have your view than guess again.

## 4. The 1.5e8 probit number — you were right, it is a plateau

You cautioned that this is an optimizer running down an almost-flat tail rather than a
localised maximum. Profiling `N0` upward with the probit link, re-optimising the
completeness at each fixed `N0`:

```
            N0       best nll    delta nll      min C      max C
         1e+03    -301.890934    8.766e+00  4.090e-02  9.889e-01
         1e+05    -310.627381    2.986e-02  9.521e-04  2.249e-02
         1e+07    -310.656227    1.018e-03  1.040e-05  2.320e-04
         1e+08    -310.657244    0.000e+00  1.059e-06  2.334e-05
         1e+10    -310.656483    7.616e-04  1.081e-08  2.350e-07
```

`Δ(nll) < 0.03` across **five orders of magnitude** in `N0`. There is a nominal optimum
near 1e8, but the profile is effectively flat: this is an unbounded/weakly-bounded
plateau, not a fitted value. I have removed "N0(probit) = 1.5e8" as a quoted fit and
now describe it as you said — an effectively unbounded profile.

This strengthens rather than weakens the conclusion.

## 5. Where I have landed

Confirmed and not in dispute:

* exact structural non-identifiability of the completeness amplitude (Tests 1–2);
* `N0 = N_obs / q` is the profile MLE, **not** the source of the degeneracy (my error,
  withdrawn);
* `N0 × meanC` is **not** the identified combination (my error, withdrawn);
* the scheme is not a valid EM — the two steps do not ascend a common objective, which
  is why the log-likelihood *fell* 462.2 → 458.8 (your point 4, which I had missed);
* the 12th iterate has no privileged status and must be withdrawn.

Actions I am taking in the code:

* raise `IdentificationError` when no completeness anchor is supplied;
* parameterise `C(x) = a · R(x; γ)` with `R(x_ref) = 1`, so `a = C(x_ref)` has a stated
  scientific meaning and the singular direction is a **named parameter** — I take your
  point that normalising by `max` over the grid makes `a` depend on grid limits;
* enforce `a·R(x) ≤ 1` over the model domain;
* rename outputs to `total_initial_count_conditional_on_reference_completeness`, and
  record the reference point / weighting measure explicitly;
* keep the convergence checks, but document them as defensive programming, not the fix.

**The open question is §3**: what actually drives the real code's persistent drift, given
that a correctly specified toy has a fixed point? I would rather establish that than
patch around it.
