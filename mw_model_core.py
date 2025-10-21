# mw_model_core.py
import numpy as np
import numpy.linalg as npl
from scipy.optimize import root
from scipy.integrate import solve_ivp
import pandas as pd

from mw_model_constants import FIT_PATH, N_HILL, KQ, D_OVERRIDE

# ---------- Load globals ----------
def load_params():
    g = pd.read_csv(FIT_PATH, index_col=0, header=None).squeeze("columns")
    # p = [r0P,rHP,r0C,K_M,gamma,c,d,g,u,K_u,p_low,p_high,H_on,H_off,tau_q,K_B]
    p = np.array([float(g[k]) for k in g.index.values], float)
    if D_OVERRIDE is not None:
        p[6] = float(D_OVERRIDE)
    return p

# ---------- Model ----------
def q_inf(H, q, H_on, H_off, KQ_local=None):
    KQ_use = KQ if KQ_local is None else KQ_local
    th = (1 - q) * H_on + q * H_off
    return 1.0 / (1.0 + np.exp(-KQ_use * (H - th)))

def rhs(y, pvec, KQ_local=None, N_HILL_local=None):
    """ 5D system: [P, C, H, B, q] """
    n = N_HILL if N_HILL_local is None else N_HILL_local
    P, C, H, B, q = y
    r0P,rHP,r0C,K_M,gamma,c,d,gH,u,K_u,pL,pH,H_on,H_off,tau,K_B = pvec
    pB = pL + (pH - pL) * np.clip(q, 0, 1)

    # ecology
    dP = P * ( r0P + rHP*H - c*pB - (P + gamma*C)/K_M )
    dC = C * ( r0C           -        (C + gamma*P)/K_M )

    # butyrate & host
    uptake = u * H * B / (K_u + B + 1e-9)
    dB = pB * P - uptake
    dH = gH * (B**n / (K_B**n + B**n)) * (1 - H) - d * H
    dq = (q_inf(H, q, H_on, H_off, KQ_local=KQ_local) - q) / tau
    return np.array([dP, dC, dH, dB, dq], float)

def jac_fd(fun, y, args=(), eps=1e-6):
    f0 = fun(y, *args); J = np.zeros((len(y), len(y)))
    for i in range(len(y)):
        y2 = y.copy(); y2[i] += eps
        J[:, i] = (fun(y2, *args) - f0) / eps
    return J

def clamp_state(y):
    return np.array([max(0,y[0]), max(0,y[1]),
                     np.clip(y[2],0,1.2), max(0,y[3]), np.clip(y[4],0,1.2)], float)

# ---------- Equilibria & Stability ----------
DEFAULT_SEEDS = [
    np.array([0.12,0.12,0.30,0.10,1.0]),
    np.array([0.05,0.20,0.90,0.12,0.0]),
    np.array([0.30,0.08,0.55,0.10,0.6]),
    np.array([0.15,0.15,0.65,0.12,0.4]),
    np.array([0.25,0.05,0.75,0.15,0.8]),
    np.array([0.08,0.25,0.85,0.10,0.2]),
]

def find_equilibria(pvec, seeds=None, KQ_local=None, N_HILL_local=None):
    if seeds is None: seeds = DEFAULT_SEEDS
    rows=[]
    for s in seeds:
        sol = root(lambda yy: rhs(yy, pvec, KQ_local, N_HILL_local), s, method="hybr")
        if not sol.success: continue
        y = clamp_state(sol.x)
        if not np.all(np.isfinite(y)): continue
        lam = np.real(npl.eigvals(jac_fd(lambda z: rhs(z,pvec,KQ_local,N_HILL_local), y)))
        rows.append({"P":y[0],"C":y[1],"H":y[2],"B":y[3],"q":y[4],
                     "lam_max":float(np.max(lam)),"stable":bool(np.max(lam)<0)})
    if not rows: 
        return pd.DataFrame(columns=["P","C","H","B","q","lam_max","stable"])
    eqs = pd.DataFrame(rows).sort_values("H").reset_index(drop=True)
    # de-dup by H
    dedup=[eqs.iloc[0]]
    for i in range(1,len(eqs)):
        if abs(eqs.iloc[i]["H"] - dedup[-1]["H"]) > 1e-3:
            dedup.append(eqs.iloc[i])
    return pd.DataFrame(dedup).reset_index(drop=True)

def relax_to_ss(pvec, y0, T=1200, KQ_local=None, N_HILL_local=None):
    sol = solve_ivp(lambda t,z: rhs(z, pvec, KQ_local, N_HILL_local),
                    (0, T), y0, t_eval=np.linspace(0, T, 1400),
                    rtol=1e-6, atol=1e-8, max_step=0.5)
    yss = clamp_state(sol.y[:,-1])
    return yss, sol

# ---------- Bistability test at a parameter point ----------
def bistability_by_multistart(pvec, n_inits=100, KQ_local=None, N_HILL_local=None,
                              init_box=((0.05,0.30),(0.05,0.30),(0.10,0.95),(0.05,0.20),(0.0,1.0))):
    """Return (#distinct steady states, endpoints dataframe) using random initial conditions."""
    rng = np.random.default_rng(42)
    def rand_in(box): 
        return np.array([rng.uniform(a,b) for (a,b) in box], float)
    endpoints=[]
    for _ in range(n_inits):
        y0 = rand_in(init_box)
        yss, _ = relax_to_ss(pvec, y0, T=1200, KQ_local=KQ_local, N_HILL_local=N_HILL_local)
        endpoints.append(yss)
    EP = np.array(endpoints)
    # cluster by H with a simple 1D gap heuristic
    H_sorted = np.sort(EP[:,2])
    gaps = np.diff(H_sorted)
    if gaps.size == 0: 
        n_states = 1
    else:
        # count distinct clusters separated by >= 0.03 in H
        n_states = 1
        for g in gaps:
            if g >= 0.03: n_states += 1
    df = pd.DataFrame(EP, columns=["P","C","H","B","q"])
    return n_states, df
