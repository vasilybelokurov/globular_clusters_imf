"""Why does the toy have a fixed point but the real code does not?

The toy above held the population SHAPE fixed. The real loop REFITS the population
(IMF + radial params) every iteration against the data, given the current selection map.

Hypothesis: refitting the shape is what destroys the fixed point, because the shape can
partially absorb the completeness -- they compete for the same observed structure.

Test: give the population a free shape parameter, refit it each iteration exactly as the
real loop does, and recompute the undamped update map.
"""
import numpy as np
from scipy import optimize, special

N=12; X=np.linspace(-2,2,N); SURV=np.full(N,0.25); N_OBS=165

def shape_of(tilt):
    """population shape with one free parameter (stands for the radial/IMF params)."""
    s = np.exp(-0.5*X**2 + tilt*X)      # a Gaussian with a fitted tilt
    return s/s.sum()

def C_of(intercept, slope):
    return np.clip(special.expit(intercept+slope*X),1e-12,1.0)

TRUE_TILT=0.0
TRUE=np.array([1.0,0.8])
OBS = N_OBS*shape_of(TRUE_TILT)*special.expit(TRUE[0]+TRUE[1]*X); OBS *= N_OBS/OBS.sum()

def iteration(intercept, slope, tilt, refit_shape: bool):
    C = C_of(intercept, slope)
    # --- population step: refit the SHAPE given the current selection (as the real loop does)
    if refit_shape:
        def pop_nll(t):
            sh = shape_of(float(t[0]))
            sel = SURV*C
            q = float(np.sum(sh*sel))
            # profiled point-process loglike: sum_i log(shape) - N log q  (weighted by data)
            return float(-(np.sum(OBS*np.log(np.clip(sh,1e-300,None))) - N_OBS*np.log(max(q,1e-300))))
        r = optimize.minimize(pop_nll, [tilt], method="L-BFGS-B", bounds=[(-3,3)])
        tilt = float(r.x[0])
    sh = shape_of(tilt)
    q  = float(np.sum(sh*SURV*C)); N0 = N_OBS/q
    predicted_complete = N0*sh*SURV
    # --- completeness step
    def nll(p):
        mu=np.clip(predicted_complete*C_of(p[0],p[1]),1e-300,None)
        return float(-(np.sum(OBS*np.log(mu)-mu)))
    r=optimize.minimize(nll,[intercept,slope],method="L-BFGS-B",bounds=[(-12,12),(-8,4)])
    return float(r.x[0]), float(r.x[1]), tilt

for refit in (False, True):
    label = "SHAPE REFIT EACH ITERATION (as the real code)" if refit else "SHAPE HELD FIXED (my earlier toy)"
    print("="*78); print(label); print("="*78)
    print(f"{'intercept_in':>13} {'meanC_in':>9} | {'d(intercept)':>13}   sign")
    signs=[]
    for a0 in (4.0,3.0,2.0,1.0,0.0,-1.0,-2.0,-3.0,-4.0):
        a1,b1,t1 = iteration(a0,0.8,TRUE_TILT,refit)
        d=a1-a0; signs.append(np.sign(round(d,6)))
        mc=float(np.mean(C_of(a0,0.8)))
        print(f"{a0:>13.2f} {mc:>9.4f} | {d:>+13.4f}   {'down' if d<0 else ('up' if d>0 else 'FIXED POINT')}")
    uniq={s for s in signs if s!=0}
    print(f"\n  -> {'sign CHANGES: a fixed point exists' if len(uniq)>1 else 'sign CONSTANT: drift is forced'}\n")
