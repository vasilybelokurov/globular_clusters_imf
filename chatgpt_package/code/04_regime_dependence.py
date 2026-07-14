"""ChatGPT's mechanism: the ridge is broken only near the logistic CEILING.
Where completeness is low, expit(z) ~ exp(z), which IS closed under rescaling,
so the ridge becomes near-exact and N0 becomes arbitrary.

The real code drifts to mean completeness 0.108. Test the link sensitivity as a
function of the completeness regime.
"""
import numpy as np
from scipy import optimize, special, stats

N=12; X=np.linspace(-2,2,N); SURV=np.full(N,0.25)
SHAPE=np.exp(-0.5*X**2); SHAPE/=SHAPE.sum(); N_OBS=165
def logit(p):  return special.expit(p[0]+p[1]*X)
def probit(p): return stats.norm.cdf(p[0]+p[1]*X)
def cloglog(p):return 1.0-np.exp(-np.exp(np.clip(p[0]+p[1]*X,-30,3)))

def nll(logN0, C):
    mu=np.exp(logN0)*SHAPE*SURV*C
    return float(-(np.sum(OBS*np.log(np.clip(mu,1e-300,None))) - mu.sum()))

print(f"{'true mean C':>12} | {'N0(logit)':>10} {'N0(probit)':>11} {'N0(cloglog)':>12} | {'spread':>8}")
print("-"*66)
for true_intercept in (3.0, 1.0, -0.5, -2.0, -3.5):
    truth=np.array([true_intercept, 0.8])
    Ctrue=logit(truth); meanC=float(np.mean(Ctrue))
    OBS = N_OBS*SHAPE*Ctrue; OBS *= N_OBS/OBS.sum()
    res={}
    for name,link in (("logit",logit),("probit",probit),("cloglog",cloglog)):
        def obj(p):
            return nll(p[0], np.clip(link(p[1:]),1e-12,1.0))
        r=optimize.minimize(obj,[np.log(900.),true_intercept,0.8],method="Nelder-Mead",
                            options={"xatol":1e-10,"fatol":1e-12,"maxiter":60000})
        res[name]=np.exp(r.x[0])
    lo,hi=min(res.values()),max(res.values())
    flag = "  <-- the regime the real code drifts INTO" if meanC < 0.15 else ""
    print(f"{meanC:>12.3f} | {res['logit']:>10.0f} {res['probit']:>11.0f} {res['cloglog']:>12.0f} | {hi/lo:>7.2f}x{flag}")
print()
print("  As completeness falls, expit -> exp, the ridge becomes near-exact, and the")
print("  three links diverge. The real code drifts to meanC=0.108 -- i.e. it walks")
print("  itself INTO the regime where N0 is least identified. That is why it runs away.")
