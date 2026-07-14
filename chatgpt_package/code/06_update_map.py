"""ChatGPT's test: is the downward drift FORCED by the update map, or incidental?

Define one full outer iteration as a deterministic map T(psi). Look at the amplitude
coordinate u = log a (a = reference completeness). Compute the UNDAMPED update

    Delta_u(u) = T_u(u) - u

over a grid of u, re-optimising the other coordinates exactly as the real loop does.

  Delta_u < 0 everywhere      -> the map forces downward drift (my claim)
  Delta_u changes sign        -> a fixed point exists (my claim is wrong)
  Delta_u ~ 0 but it drifts   -> tolerances/numerics dominate
"""
import numpy as np
from scipy import optimize, special

N=12; X=np.linspace(-2,2,N); SURV=np.full(N,0.25)
SHAPE0=np.exp(-0.5*X**2); SHAPE0/=SHAPE0.sum(); N_OBS=165

# data generated at HIGH completeness so we start well away from the tail
TRUE=np.array([1.0,0.8])
OBS = N_OBS*SHAPE0*special.expit(TRUE[0]+TRUE[1]*X); OBS *= N_OBS/OBS.sum()

def C_of(intercept, slope):
    return np.clip(special.expit(intercept+slope*X),1e-12,1.0)

def one_undamped_iteration(intercept, slope):
    """Exactly the real loop's two steps, undamped (relaxation = 1)."""
    C = C_of(intercept, slope)
    # --- population step: N0 defined as the profile MLE
    q  = float(np.sum(SHAPE0*SURV*C))
    N0 = N_OBS/q
    predicted_complete = N0*SHAPE0*SURV        # "complete" counts, C divided out
    # --- completeness step: fit C to observed vs predicted_complete
    def nll(p):
        mu=np.clip(predicted_complete*C_of(p[0],p[1]),1e-300,None)
        return float(-(np.sum(OBS*np.log(mu)-mu)))
    r=optimize.minimize(nll,[intercept,slope],method="L-BFGS-B",
                        bounds=[(-12,12),(-8,4)])
    return float(r.x[0]), float(r.x[1])

# u = mean completeness is the interpretable amplitude coordinate here;
# sweep the intercept, which moves the amplitude.
print("Undamped update map, swept over the completeness amplitude:")
print(f"{'intercept_in':>13} {'meanC_in':>9} | {'intercept_out':>14} {'meanC_out':>10} | {'d(intercept)':>13} {'d(logit meanC)':>15}")
print("-"*90)
signs=[]
for a0 in (4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0):
    a1, b1 = one_undamped_iteration(a0, 0.8)
    mc0=float(np.mean(C_of(a0,0.8))); mc1=float(np.mean(C_of(a1,b1)))
    d_int = a1-a0
    d_log = np.log(mc1)-np.log(mc0)
    signs.append(np.sign(d_int))
    print(f"{a0:>13.2f} {mc0:>9.4f} | {a1:>14.4f} {mc1:>10.4f} | {d_int:>+13.4f} {d_log:>+15.4f}")
print()
uniq=set(s for s in signs if s!=0)
if uniq == {-1.0}:
    print("  VERDICT: d(intercept) < 0 everywhere -> the update map DOES force downward drift.")
elif len(uniq) > 1:
    print("  VERDICT: d(intercept) CHANGES SIGN -> a fixed point exists; the drift is NOT forced.")
    print("           My 'the geometry forces it' claim is WRONG. ChatGPT is right.")
else:
    print("  VERDICT: updates ~0; numerics dominate.")
