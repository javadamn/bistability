
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Tuple
import numpy as np
from scipy.integrate import solve_ivp

Array = np.ndarray

@dataclass
class Params:
    # Microbes
    r_M: float = 0.6      # d^-1
    r_C: float = 0.7      # d^-1
    K_M: float = 1.0      # carrying capacity (arbitrary units)
    K_C: float = 1.0
    gamma: float = 0.6    # M-C competition strength
    alpha: float = 0.6    # inflammation penalty on M (via low H)
    beta: float = 0.6     # inflammation boon on C
    
    # Butyrate production & fate
    p_min: float = 0.2    # baseline prod
    p_max: float = 1.0    # max prod
    u: float = 0.15       # host uptake coefficient
    k_B: float = 0.15     # washout/decay
    
    # Host health
    g: float = 0.8        # strength of B->H
    K_B: float = 0.5      # half-sat of S(B)
    n: float = 2.0        # Hill coef
    d0: float = 0.1       # natural decay
    chi: float = 0.6      # inflammation drag on H
    H_I: float = 0.5      # midpoint for I(H) sigmoid
    eta: float = 6.0      # steepness of I(H)
    
    # Slow expression state (E) hysteresis
    H_on: float = 0.3
    H_off: float = 0.7
    eps: float = 0.05
    k_on: float = 0.2
    k_off: float = 0.2
    
    # Interventions
    sigma_F: float = 0.2  # fiber effect on M
    sigma_A: float = 0.6  # antibiotic kill on M
    
    # Observation scaling (not part of dynamics)
    s_B: float = 1.0      # butyrate scale to instrument units
    s_H: float = 1.0      # health proxy scale

def S_B(B: float, K_B: float, n: float) -> float:
    return (B**n) / (B**n + K_B**n + 1e-12)

def I_of_H(H: float, H_I: float, eta: float) -> float:
    # Higher when H is low (inflammation)
    return 1.0 / (1.0 + np.exp(eta*(H - H_I)))

def q_inf(H: float, H_on: float, H_off: float, eps: float, q_current: float) -> float:
    # Smooth hysteresis target using sigmoids
    on = 1.0 / (1.0 + np.exp((H - H_on)/eps))  # near 1 if H < H_on
    off = 1.0 / (1.0 + np.exp((H_off - H)/eps)) # near 1 if H > H_off
    # When in middle zone, hold q by mixing with current state
    mid = 1.0 - (on + off - on*off)  # true when between thresholds
    return on*1.0 + off*0.0 + mid*q_current

def rhs(t: float, x: Array, p: Params, F_fun: Callable[[float], float], A_fun: Callable[[float], float]) -> Array:
    M, C, B, H, E = x
    F = F_fun(t)
    A = A_fun(t)
    
    I = I_of_H(H, p.H_I, p.eta)
    pB = p.p_min + (p.p_max - p.p_min) * E
    dM = (p.r_M*(1 - M/p.K_M) - p.alpha*I - p.gamma*C - p.sigma_A*A - 0.0*pB)*M + p.sigma_F*F*M
    dC = (p.r_C*(1 - C/p.K_C) + p.beta*I - p.gamma*M)*C
    dB = pB*M - p.u*H*B - p.k_B*B
    dH = p.g*S_B(B, p.K_B, p.n) - p.d0*H - p.chi*I*H
    qstar = q_inf(H, p.H_on, p.H_off, p.eps, E)
    dE = p.k_on*(1-E)*qstar - p.k_off*E*(1-qstar)
    return np.array([dM, dC, dB, dH, dE])

def piecewise_constant_from_samples(times: Array, values: Array) -> Callable[[float], float]:
    # Returns step function that holds last observation
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(times)
    times = times[order]
    values = values[order]
    def f(t: float) -> float:
        idx = np.searchsorted(times, t, side='right') - 1
        if idx < 0:
            return float(values[0]) if len(values)>0 else 0.0
        return float(values[min(idx, len(values)-1)])
    return f

def simulate(t0: float, t1: float, x0: Array, p: Params,
             F_fun: Callable[[float], float],
             A_fun: Callable[[float], float],
             dense: bool = True) -> Tuple[Array, Array]:
    sol = solve_ivp(lambda t, x: rhs(t, x, p, F_fun, A_fun),
                    (t0, t1), x0, method="RK45", rtol=1e-6, atol=1e-8, dense_output=dense, max_step=0.5)
    if not sol.success:
        raise RuntimeError(sol.message)
    t = sol.t
    X = sol.y.T
    return t, X

def obs_butyrate(B: Array, p: Params) -> Array:
    return p.s_B * B

def obs_health(H: Array, p: Params) -> Array:
    return p.s_H * H
