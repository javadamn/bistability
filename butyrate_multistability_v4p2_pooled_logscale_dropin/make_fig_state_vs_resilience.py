#!/usr/bin/env python3
# Python 3.10; deps: numpy, matplotlib
import numpy as np
import matplotlib.pyplot as plt

def main(out_png="fig_state_vs_resilience.png"):
    z = np.linspace(-2.5, 2.5, 800)

    # Left: illustrative bistable potential (double well)
    a = 0.15  # tilt
    V_bi = 0.55 * (z**2 - 1.0)**2 - a * z
    V_bi -= V_bi.min()

    # Right: single-well potentials with different curvature (resilience)
    z_star = 0.2
    kappa_lo = 0.4
    kappa_hi = 1.2
    V_lo = 0.5 * kappa_lo * (z - z_star)**2
    V_hi = 0.5 * kappa_hi * (z - z_star)**2
    V_lo -= V_lo.min()
    V_hi -= V_hi.min()

    fig = plt.figure(figsize=(10.2, 4.0))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(z, V_bi, linewidth=2)
    ax1.set_title("Discrete-state picture (illustrative)")
    ax1.set_xlabel("state z")
    ax1.set_ylabel("effective potential V(z)")
    ax1.text(0.02, 0.92, "two minima + barrier", transform=ax1.transAxes)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(z, V_lo, linewidth=2, label="weaker recovery")
    ax2.plot(z, V_hi, linewidth=2, label="stronger recovery")
    ax2.set_title("Continuous resilience picture")
    ax2.set_xlabel("state z")
    ax2.set_ylabel("effective potential V(z)")
    ax2.legend(loc="upper right", frameon=False)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"Wrote: {out_png}")

if __name__ == "__main__":
    main()
