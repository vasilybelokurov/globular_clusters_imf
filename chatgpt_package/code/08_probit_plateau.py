"""ChatGPT: the probit N0=1.5e8 is probably an optimizer on a flat plateau, not a
maximum. Check by profiling N0 upward and watching the likelihood."""
import numpy as np
from scipy import optimize, stats

N=12; X=np.linspace(-2,2,N); SURV=np.full(N,0.25)
SHAPE=np.exp(-0.5*X**2); SHAPE/=SHAPE.sum(); N_OBS=165
from scipy import special
TRUE=np.array([-3.5,0.8])
OBS=N_OBS*SHAPE*special.expit(TRUE[0]+TRUE[1]*X); OBS*=N_OBS/OBS.sum()
def probit(p): return np.clip(stats.norm.cdf(p[0]+p[1]*X),1e-300,1.0)
def nll(logN0,C):
    mu=np.exp(logN0)*SHAPE*SURV*C
    return float(-(np.sum(OBS*np.log(np.clip(mu,1e-300,None)))-mu.sum()))

print("PROBIT: profile the likelihood as N0 is forced upward")
print(f"{'N0':>14} {'best nll':>14} {'delta nll':>12} {'min C':>10} {'max C':>10}")
best=None; rows=[]
for logN0 in np.log([1e3,1e4,1e5,1e6,1e7,1e8,1e9,1e10]):
    r=optimize.minimize(lambda p: nll(logN0, probit(p)), [-3.5,0.8],
                        method="Nelder-Mead", options={"xatol":1e-10,"fatol":1e-12,"maxiter":40000})
    C=probit(r.x); rows.append((np.exp(logN0), r.fun, C.min(), C.max()))
    best = r.fun if best is None else min(best, r.fun)
for n0,v,cmin,cmax in rows:
    print(f"{n0:>14.3g} {v:>14.6f} {v-best:>12.3e} {cmin:>10.3e} {cmax:>10.3e}")
print()
print("  If delta nll stays ~0 as N0 -> 1e10, the profile is an unbounded PLATEAU:")
print("  there is no finite maximum, and '1.5e8' is just where the optimizer stopped.")
print("  ChatGPT's caution is then correct and the number must not be quoted as a fit.")
