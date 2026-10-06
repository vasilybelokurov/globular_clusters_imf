# Extending the GCIMF Reconstruction to Millisecond Pulsars

Date: 2026-07-11

This note records the current idea for using the Milky Way globular-cluster
initial mass function (GCIMF) reconstruction to predict millisecond pulsars
(MSPs) in surviving clusters, missing surviving clusters, and the Galactic
field from disrupted clusters.

## Core Idea

The GCIMF inference can be coupled naturally to an MSP response model. The
important point is that the MSP yield should not be modelled as a simple
function of initial cluster mass alone. MSP production depends on the
time-dependent dynamical history of the cluster:

```latex
N_{\rm MSP}(t_0)
=
{\cal F}\!\left[
M(t), r_{\rm h}(t), r_{\rm c}(t), Z,
N_{\rm BH}(t), N_{\rm NS}(t),
f_{\rm b}(t), t_{\rm dis}
\right].
```

CMC is probably the best established framework for calibrating this function,
because it couples two-body relaxation, strong encounters, stellar and binary
evolution, neutron-star formation, accretion-driven recycling, and black-hole
heating. Existing CMC work predicts MSP numbers, binary properties, and spatial
distributions. A key physical result is that MSP production is enhanced by
dynamical interactions but can be suppressed while a substantial stellar-mass
black-hole subsystem remains.

## Link to the Current GCIMF Model

The current GCIMF reconstruction infers a latent birth population with variables
similar to

```latex
\Theta_j =
\{M_{{\rm ini},j}, a_j, Z_j, t_{{\rm form},j},
{\rm origin}_j\}.
```

For an MSP extension, add an MSP response kernel:

```latex
K_{\rm MSP}
\left(
N_{\rm MSP}, \mathbf{x}_{\rm MSP}
\mid
M_{\rm ini}, a, Z, t_{\rm form},
\Theta_{\rm dyn}
\right),
```

where the MSP observable vector could include

```latex
\mathbf{x}_{\rm MSP}
=
\{P, \dot P, P_{\rm orb}, e, M_{\rm comp},
{\rm binary/isolated}, \mathbf{x}_{\rm Gal}\}.
```

The total present-day MSP population from the GC channel should be divided into
three physically distinct components:

```latex
N_{\rm MSP}^{\rm MW}
=
N_{\rm MSP}^{\rm surv,obs}
+
N_{\rm MSP}^{\rm surv,miss}
+
N_{\rm MSP}^{\rm dissolved}.
```

This division matches the existing survival-selection framework: observed
survivors are in the Baumgardt catalogue, missing survivors are dynamically
surviving but observationally undetected clusters, and dissolved clusters are
the disrupted GC population inferred by the GCIMF model.

## 1. MSPs in Observed Surviving Clusters

For each observed Baumgardt cluster, infer its posterior over birth properties
and evolutionary prescription:

```latex
p(M_{\rm ini}, a, Z, \eta_t, \ldots \mid D_{\rm GC}).
```

Then compute

```latex
p(N_{\rm MSP} \mid D_{\rm GC})
=
\int
K_{\rm MSP}(N_{\rm MSP} \mid \Theta)
p(\Theta \mid D_{\rm GC})\,d\Theta.
```

These predictions can be checked against:

- known radio MSP counts;
- X-ray MSP candidates;
- gamma-ray luminosities;
- binary versus isolated fractions;
- projected radial distributions.

This gives a posterior predictive test of the GC evolution model. Existing CMC
calculations find of order 10 to 20 MSPs in representative present-day clusters
with masses around `2e5 Msun`, and nearly 100 in a massive model with
present-day mass around `1e6 Msun`. These numbers should not be used as a
universal mass scaling. They are outputs of particular dynamical histories.

## 2. MSPs in Surviving but Undetected Clusters

The detectability model predicts a latent population of clusters that survive
dynamically but are absent from the Baumgardt catalogue. For each posterior
draw, generate those clusters and assign MSP populations through the same MSP
kernel:

```latex
N_{\rm MSP}^{\rm surv,miss}
=
\sum_{j\in{\rm missing}}
N_{{\rm MSP},j}.
```

This component could predict:

- unidentified radio-MSP-rich clusters;
- unresolved Fermi sources;
- inner-Galaxy clusters hidden by extinction and crowding;
- clusters whose integrated gamma-ray emission is detectable even when their
  stellar counterpart is incomplete.

The cluster-discovery probability and MSP-detection probability should be
modelled separately:

```latex
P_{\rm obs}
=
P_{\rm GC\,det}
P_{\rm MSP\,det}.
```

MSPs are not automatically observable just because their host cluster is
observable, and a host cluster may remain hard to identify even if its MSP
population contributes to radio or gamma-ray catalogues.

## 3. MSPs Released by Dissolved Clusters

This is likely the most novel part of the project. A cluster that dissolves at
time `t_dis` releases the MSP population it contains at that time:

```latex
N_{\rm MSP,rel}
=
N_{\rm MSP}(t_{\rm dis}).
```

Each released MSP then evolves to the present day under:

- Galactic orbital dynamics;
- pulsar spin-down;
- binary evolution;
- possible binary disruption before dissolution;
- radio and gamma-ray luminosity evolution.

The field contribution from dissolved clusters can be written schematically as

```latex
f_{\rm MSP}^{\rm dis}(\mathbf{x}, t_0)
=
\int d\Theta_{\rm GC}\,
\frac{dN_{\rm GC}}{d\Theta_{\rm GC}}\,
\int dt_{\rm rel}\,
\frac{dN_{\rm MSP}}{dt_{\rm rel}}\,
{\cal P}_{\rm orb}\,
{\cal P}_{\rm evol},
```

where `P_orb` propagates the released MSP through the Galactic potential and
`P_evol` propagates its pulsar and binary properties.

Previous studies have explored disrupted-GC contributions to the Galactic
Centre MSP population, but their conclusions depend strongly on the assumed GC
disruption history, MSP luminosity evolution, and deposited stellar mass. The
GCIMF reconstruction can improve one of the largest uncertainties in those
studies: the normalization and orbital distribution of the disrupted GC
population.

## Why `N_MSP proportional to M_ini` Is Inadequate

The number of neutron stars initially formed is approximately proportional to
birth mass:

```latex
N_{\rm NS,form} \propto M_{\rm ini}.
```

But MSP formation requires a retained neutron star to acquire a suitable
companion and accrete enough material. A more useful decomposition is

```latex
N_{\rm MSP}(t)
\simeq
N_{\rm NS,form}
f_{\rm NS,ret}
f_{\rm recycle}
f_{\rm alive},
```

where every factor depends on cluster history. In particular,

```latex
f_{\rm recycle}
=
f_{\rm recycle}
\left[
\Gamma(t), f_{\rm b}(t), N_{\rm BH}(t),
\rho_{\rm c}(t), \sigma(t), Z
\right].
```

The CMC result that MSP number can be anticorrelated with retained BH number is
important. A massive cluster may form and retain more neutron stars, but a
long-lived BH subsystem can keep the luminous core expanded and delay neutron
star segregation and recycling.

Thus two clusters with the same `M_ini` and `a` can have different MSP yields
because of different:

- initial radii;
- BH natal kicks;
- NS formation channels and kicks;
- metallicities;
- binary fractions;
- tidal histories;
- dissolution times.

The initial half-mass radius, or an equivalent density parameter, should be an
additional latent variable:

```latex
\Theta_{\rm GC}
=
\{M_{\rm ini}, a, r_{\rm h,ini}, Z, t_{\rm form},
f_{\rm b,ini}, \ldots\}.
```

Without `r_h,ini`, the encounter and relaxation histories are not uniquely
specified.

## Three Modelling Levels

### A. Fast Empirical Prescription

The quickest implementation would model the expected present-day MSP number
using current cluster properties:

```latex
\lambda_{\rm MSP}
=
A
\left(\frac{\Gamma}{\Gamma_0}\right)^\beta
\left(\frac{M}{M_0}\right)^\gamma
10^{\delta [{\rm Fe/H}]}
g(N_{\rm BH}),
```

with an overdispersed count model such as

```latex
N_{\rm MSP} \sim {\rm NegBin}(\lambda_{\rm MSP}, k)
```

or a zero-inflated negative-binomial distribution.

A common encounter-rate proxy is

```latex
\Gamma \propto \rho_{\rm c}^{3/2} r_{\rm c}^{2},
```

or more generally

```latex
\Gamma \propto \int \frac{\rho^2}{\sigma}\,dV.
```

This level is useful for surviving clusters. It is inadequate for dissolved
clusters because the relevant quantity is the encounter history, especially
`\Gamma(t_dis)`, not only the present-day value.

### B. Semi-analytic Evolutionary Prescription

A better intermediate model would evolve a small set of state variables:

```latex
\{M, r_{\rm h}, N_{\rm BH}, N_{\rm NS}, N_{\rm bin},
N_{\rm LMXB}, N_{\rm MSP}\}.
```

For example,

```latex
\frac{dN_{\rm MSP}}{dt}
=
\epsilon_{\rm rec}(t)\,
\Gamma_{\rm NS-bin}(t)
-
\frac{N_{\rm MSP}}{\tau_{\rm MSP}}
-
\dot N_{\rm esc,MSP}.
```

One could write

```latex
\Gamma_{\rm NS-bin}
\propto
\frac{N_{\rm NS,c}N_{\rm bin,c}}{V_{\rm c}}
\frac{\Sigma_{\rm int}}{\sigma_{\rm c}},
```

with the core fractions calibrated against CMC. This model is inexpensive
enough to evaluate inside a hierarchical inference and preserves the
dependence on dissolution time. It still requires simulation calibration.

### C. CMC-trained Emulator

The most defensible long-term approach is to train an emulator on CMC models.
The training grid should span:

```latex
\begin{aligned}
M_{\rm ini} &= 10^4-2\times10^7\,M_\odot,\\
r_{\rm h,ini} &\ {\rm or}\ r_{\rm v,ini},\\
a &\ {\rm or\ a\ time-dependent\ tidal\ field},\\
Z, &\quad f_{\rm b,ini},\\
{\rm NS/BH\ kicks},&\quad {\rm BH\ retention\ prescriptions}.
\end{aligned}
```

The stored time-dependent outputs should include

```latex
Y(t)=
\{
M(t), r_{\rm h}(t), r_{\rm c}(t),
N_{\rm BH}(t), N_{\rm NS}(t),
N_{\rm LMXB}(t), N_{\rm MSP}(t)
\}.
```

The emulator would approximate

```latex
p\!\left[
Y(t)\mid
M_{\rm ini}, r_{\rm h,ini}, a, Z,
f_{\rm b}, \Theta_{\rm rem}
\right].
```

Possible emulator choices include:

- Gaussian processes for low-dimensional grids;
- neural conditional density estimators;
- normalizing flows;
- sparse-grid interpolation with a stochastic residual model.

Because `N_MSP` can be small and overdispersed, the emulator should predict a
distribution, not only a mean:

```latex
p(N_{\rm MSP}, N_{\rm bin}, N_{\rm iso}, \ldots \mid \Theta).
```

The existing CMC catalogue contains many useful models and covers much of the
present-day Milky Way GC structural range, but it was not designed to span the
high birth-mass range or all plausible dissolution histories needed here.

## Hybrid Strategy

The full GCIMF inference does not need to be rerun jointly with CMC at first.
A practical workflow is:

1. Infer the GC birth population using the current model.
2. Identify the posterior-supported region in `(M_ini, a, Z, r_h,ini)`.
3. Select simulation design points using active learning.
4. Run CMC only at those points.
5. Train an MSP-yield emulator.
6. Apply the emulator to every posterior realization of the GC birth
   population.
7. Compare the predicted surviving MSP population with radio, X-ray, and
   gamma-ray data.
8. Propagate dissolved systems into the field or Galactic bulge.

This avoids building an expensive rectangular simulation grid where many models
lie outside the posterior-supported part of parameter space.

## First-pass Use of Existing CMC Models

As a proof of concept, assign existing CMC model `k` to inferred cluster `j`
using a kernel in initial-parameter space:

```latex
w_{jk}
\propto
\exp\left[
-\frac{1}{2}
(\Theta_j-\Theta_k)^{\rm T}
\mathbf{C}^{-1}
(\Theta_j-\Theta_k)
\right].
```

Then approximate

```latex
p(Y_j)
\simeq
\sum_k w_{jk}p(Y_k).
```

This is a fast way to build a demonstration model, but it should not be used to
extrapolate far beyond the CMC grid, especially to
`M_ini ~ 10^7 Msun`. BH retention, relaxation time, and MSP recycling may
change nonlinearly in that regime.

## Survival Histories Needed for MSPs

For MSP predictions, the survival model needs more than a binary
survive/disrupt label. It needs an approximate dissolution-time posterior:

```latex
p(t_{\rm dis}\mid M_{\rm ini}, a, \eta_t, \ldots).
```

Two dissolved clusters with identical birth properties produce different
present-day MSP populations if one dissolved 10 Gyr ago and the other dissolved
1 Gyr ago.

The model also needs an approximate mass-loss trajectory:

```latex
M(t\mid M_{\rm ini}, a, \Theta_{\rm loss}),
```

because gradual tidal loss releases stars and binaries before final
dissolution. MSP release is therefore not necessarily an impulse at `t_dis`.
A more realistic source term is

```latex
\frac{dN_{\rm MSP,rel}}{dt}
=
f_{\rm esc,MSP}(t)
N_{\rm MSP}(t),
```

plus final release of the bound remainder at dissolution.

Because MSPs are more massive than the average surviving star and often
centrally concentrated, they should not be assumed to escape in proportion to
total stellar mass:

```latex
\frac{\dot N_{\rm MSP,esc}}{N_{\rm MSP}}
\neq
\frac{\dot M}{M}.
```

This segregation correction can be measured directly in CMC runs.

## Important Physics Parameters to Marginalize

The dominant MSP-specific uncertainties are likely to be:

- Neutron-star retention:

```latex
f_{\rm NS,ret}
=
f(v_{\rm esc}, p(v_{\rm kick}),
{\rm ECSN/AIC/MIC\ fractions}).
```

- BH retention and depletion:

```latex
N_{\rm BH}(t).
```

  This controls core heating and neutron-star segregation and may produce an
  MSP-BH anticorrelation.

- Binary evolution:

  Common-envelope efficiency, stable mass transfer, magnetic braking, and
  accretion efficiency determine whether an NS binary becomes an LMXB and how
  strongly it is recycled.

- Initial binary population:

  The primordial binary fraction and period/mass-ratio distributions affect
  both primordial and dynamically modified MSP channels.

- Initial radius:

```latex
t_{\rm rh}\propto
\frac{M^{1/2}r_{\rm h}^{3/2}}{\ln\Lambda},
\qquad
v_{\rm esc}\propto
\left(\frac{M}{r_{\rm h}}\right)^{1/2}.
```

  Mass alone does not determine the encounter history.

## Likelihood Structure

Let `n_i^obs` be the observed MSP count in surviving cluster `i`. A joint
likelihood can be written as

```latex
{\cal L}
=
{\cal L}_{\rm GC\,catalogue}
\prod_{i\in{\rm observed\ GCs}}
p(n_i^{\rm obs}\mid\lambda_i,q_i)
\times
{\cal L}_{\gamma}
\times
{\cal L}_{\rm field},
```

where

```latex
\lambda_i
=
\lambda_{\rm MSP}(\Theta_i)
```

is the intrinsic MSP population predicted by the response model or emulator,
and `q_i` represents survey completeness.

A thinning model gives

```latex
n_i^{\rm obs}\mid N_i,q_i
\sim
{\rm Binomial}(N_i,q_i).
```

For radio surveys, completeness can depend on

```latex
q_i =
q(d,l,b,{\rm DM},{\rm scattering},
P,P_{\rm orb},L_{\rm radio},
{\rm beaming},{\rm eclipsing}).
```

For gamma rays, the cluster likelihood can instead integrate over an MSP
luminosity function:

```latex
L_{\gamma,i}
=
\sum_{k=1}^{N_i}L_{\gamma,ik}.
```

Using both radio counts and integrated gamma-ray luminosities should help
separate intrinsic MSP abundance from radio-selection incompleteness.

## Recommended First Publishable Version

A manageable first implementation has four stages:

1. Use the GCIMF posterior to reconstruct the numbers and properties of
   observed survivors, missing survivors, and disrupted clusters.
2. Construct an empirical or CMC-informed MSP relation for surviving clusters:

```latex
N_{\rm MSP}
\sim
p(N\mid M_{\rm now}, \Gamma, Z, N_{\rm BH}^{\rm proxy}).
```

3. Construct a time-dependent prescription for disrupted clusters:

```latex
N_{\rm MSP,rel}
=
N_{\rm MSP}(t_{\rm dis}),
```

   calibrated to selected CMC histories.
4. Forward-model radio and gamma-ray observables and compare with the GC MSP
   census, unidentified inner-Galaxy sources, the field or bulge MSP spatial
   distribution, and integrated gamma-ray emission.

The most valuable output is not just a total MSP number. It is the joint
posterior

```latex
p\left(
N_{\rm MSP}^{\rm surv},
N_{\rm MSP}^{\rm miss},
N_{\rm MSP}^{\rm dis},
\mathbf{x}_{\rm MSP}
\mid D
\right),
```

with results split by in-situ and accreted GC components.

## Bottom Line

The GCIMF project is well structured for an MSP layer. The GCIMF inference
supplies the missing population normalization, initial masses, orbits,
metallicities, origins, and survival histories. CMC supplies the mapping from
those histories to neutron-star retention, LMXB production, MSP recycling, and
MSP escape.

The dissolved-cluster component is especially promising. The GCIMF and survival
model can replace externally assumed disruption normalizations used in previous
Galactic-centre MSP calculations. The right long-term model is a
CMC-calibrated, time-dependent response emulator, not a direct conversion from
birth mass to MSP count.
