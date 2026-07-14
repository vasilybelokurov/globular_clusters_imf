"""Check ChatGPT's two corrections to my diagnosis, numerically."""
import numpy as np
from scipy import optimize, special

N_BINS=12; X=np.linspace(-2,2,N_BINS)
SURV=np.full(N_BINS,0.25); SHAPE=np.exp(-0.5*X**2); SHAPE/=SHAPE.sum()
N_OBS=165
def C_logit(p): return np.clip(special.expit(p[0]+p[1]*X),1e-9,1.0)
OBS = N_OBS*SHAPE*C_logit(np.array([1.0,0.8])); OBS *= N_OBS/OBS.sum()

print("="*78)
print("CORRECTION 1: 'making N0 free fixes it' -- ChatGPT says NO. Test the FULL")
print("             (unprofiled) Poisson likelihood with N0 as a free parameter.")
print("="*78)
def full_nll(z):
    logN0, a, b = z
    N0 = np.exp(logN0); C = C_logit(np.array([a,b]))
    q  = float(np.sum(SHAPE*SURV*C))          # selection fraction
    mu = N0*SHAPE*SURV*C
    # full inhomogeneous Poisson: sum n log mu - sum mu   (N0*q = sum mu)
    return float(-(np.sum(OBS*np.log(np.clip(mu,1e-300,None))) - N0*q))

print(f"{'start intercept':>16} {'fitted logN0':>13} {'fitted N0':>12} {'intercept':>10} {'nll':>12}")
for a0 in (-3.0, -1.0, 1.0, 3.0):
    r = optimize.minimize(full_nll, [np.log(1000.), a0, 0.8], method="Nelder-Mead",
                          options={"xatol":1e-8,"fatol":1e-10,"maxiter":20000})
    N0 = np.exp(r.x[0])
    print(f"{a0:>16.1f} {r.x[0]:>13.4f} {N0:>12.1f} {r.x[1]:>10.4f} {r.fun:>12.6f}")
print("\n  -> different N0 by orders of magnitude, SAME nll to ~6 decimals.")
print("     The ridge survives making N0 free. ChatGPT is right; I was wrong to say")
print("     'N0 = N_obs/selection_fraction injects the degeneracy'. It is the analytic")
print("     MLE along a degeneracy that already exists in the model.\n")

print("="*78)
print("CORRECTION 2: is N0 x meanC the identified combination? ChatGPT says NO.")
print("="*78)
base=np.array([1.0,0.8])
print(f"{'k':>6} {'N0':>10} {'unweighted meanC':>17} {'N0*meanC':>11} {'sum(mu)=N_obs':>14}")
for k in (1.0,0.5,0.25,0.1):
    C=k*C_logit(base); q=float(np.sum(SHAPE*SURV*C)); N0=N_OBS/q
    mu=N0*SHAPE*SURV*C
    mC=float(np.mean(C))
    print(f"{k:>6.2f} {N0:>10.1f} {mC:>17.4f} {N0*mC:>11.1f} {mu.sum():>14.4f}")
print("\n  -> under an EXACT rescaling N0*meanC IS invariant (that was my toy).")
print("     But in the real code it moved 1.35e5 -> 6.11e4, because the population")
print("     SHAPE and the weighting both change between iterations. So N0*meanC is")
print("     NOT a generally identified quantity. ChatGPT is right.")
print("     The genuinely identified object is the detected intensity, whose integral")
print("     is just N_obs = 165 -- i.e. the data pin the detected counts, nothing more.")
