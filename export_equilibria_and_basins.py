# export_equilibria_and_basins.py
import os, numpy as np, pandas as pd, numpy.linalg as npl
from scipy.optimize import root
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from mw_model_constants import FIT_PATH, N_HILL, KQ, D_OVERRIDE

OUT = "mw_eq_export"; os.makedirs(OUT, exist_ok=True)

# Load fitted globals
g = pd.read_csv(FIT_PATH, index_col=0, header=None).squeeze("columns")
# p = [r0P,rHP,r0C,K_M,gamma,c,d,g,u,K_u,p_low,p_high,H_on,H_off,tau_q,K_B]
p = np.array([float(g[k]) for k in g.index.values], float)
if D_OVERRIDE is not None:
    p[6] = float(D_OVERRIDE)
d_fit = float(p[6])

def q_inf(H,q,H_on,H_off):
    th = (1-q)*H_on + q*H_off
    return 1.0/(1.0 + np.exp(-KQ*(H - th)))

def rhs(y, pvec):
    P,C,H,B,q = y
    r0P,rHP,r0C,K_M,gamma,c,d,gH,u,K_u,pL,pH,H_on,H_off,tau,K_B = pvec
    pB = pL + (pH - pL)*np.clip(q,0,1)
    # ecology
    dP = P*( r0P + rHP*H - c*pB - (P + gamma*C)/K_M )
    dC = C*( r0C           -        (C + gamma*P)/K_M )
    # butyrate & host
    uptake = u*H*B/(K_u + B + 1e-9)
    dB = pB*P - uptake
    dH = gH*(B**N_HILL/(K_B**N_HILL + B**N_HILL))*(1 - H) - d*H
    dq = (q_inf(H,q,H_on,H_off) - q)/tau
    return np.array([dP,dC,dH,dB,dq], float)

def jac_fd(fun,y,args=(),eps=1e-7):
    f0=fun(y,*args); J=np.zeros((5,5))
    for i in range(5):
        y2=y.copy(); y2[i]+=eps
        J[:,i]=(fun(y2,*args)-f0)/eps
    return J

# Multi-start root-finding at baseline
seeds = [
    np.array([0.12,0.12,0.30,0.10,1.0]),
    np.array([0.05,0.20,0.90,0.12,0.0]),
    np.array([0.30,0.08,0.55,0.10,0.6]),
    np.array([0.15,0.15,0.65,0.12,0.4]),
    np.array([0.25,0.05,0.75,0.15,0.8]),
    np.array([0.08,0.25,0.85,0.10,0.2]),
]
rows=[]
for s in seeds:
    sol = root(lambda yy: rhs(yy, p), s, method="hybr")
    if not sol.success: continue
    y = sol.x
    y = np.array([max(0,y[0]), max(0,y[1]),
                  np.clip(y[2],0,1.2), max(0,y[3]), np.clip(y[4],0,1.2)], float)
    if not np.all(np.isfinite(y)): continue
    lam = np.real(npl.eigvals(jac_fd(lambda z: rhs(z,p), y)))
    rows.append({"P":y[0],"C":y[1],"H":y[2],"B":y[3],"q":y[4],
                 "lam_max":float(np.max(lam)),"stable":bool(np.max(lam)<0)})

if not rows:
    raise RuntimeError("No equilibria found from seeds. Try adjusting seeds slightly.")

eqs = pd.DataFrame(rows).sort_values("H").reset_index(drop=True)
# de-duplicate by H
dedup=[eqs.iloc[0]]
for i in range(1,len(eqs)):
    if abs(eqs.iloc[i]["H"] - dedup[-1]["H"]) > 1e-3:
        dedup.append(eqs.iloc[i])
eqs = pd.DataFrame(dedup)

# Require two stable equilibria
stables = eqs[eqs["stable"]]
if len(stables) < 2:
    eqs.to_csv(f"{OUT}/equilibria.csv", index=False)
    raise RuntimeError(
        "Monostable at current FIT/N_HILL/KQ/d. "
        "Re-run or adjust constants to the exact pocket that yielded YES."
    )

eqs.to_csv(f"{OUT}/equilibria.csv", index=False)

# Basin heatmap (H0,q0 grid)
def relax(y0,T=900):
    sol=solve_ivp(lambda t,z: rhs(z,p),(0,T),y0,t_eval=np.linspace(0,T,1200),
                  rtol=1e-6, atol=1e-8, max_step=0.5)
    return sol.y[:,-1]

Hs=np.linspace(0.15,0.95,25); qs=np.linspace(0.0,1.0,25)
Z=np.zeros((len(Hs),len(qs)))
for i,H0 in enumerate(Hs):
    for j,q0 in enumerate(qs):
        y0=np.array([0.12,0.12,H0,0.10,q0],float)
        Z[i,j]=relax(y0)[2]

plt.figure(figsize=(6.6,5.2))
plt.imshow(Z, origin="lower", extent=[qs[0],qs[-1],Hs[0],Hs[-1]], aspect="auto", cmap="viridis")
plt.colorbar(label="final H")
plt.xlabel("q0"); plt.ylabel("H0")
plt.title("Basins @ baseline")
plt.tight_layout(); plt.savefig(f"{OUT}/basins.png"), plt.close()

print("Saved ->", OUT, "| equilibria.csv + basins.png")
