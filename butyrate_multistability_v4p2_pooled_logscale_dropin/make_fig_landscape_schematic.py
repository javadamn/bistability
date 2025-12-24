#!/usr/bin/env python3
# Python 3.10; deps: numpy, matplotlib
import numpy as np
import matplotlib.pyplot as plt

def main(out_png="fig_landscape_schematic.png"):
    # Example stable linear drift around a fixed point z*
    z = np.linspace(-2.5, 2.5, 600)
    z_star = 0.8
    kappa = 1.2  # restoring strength
    f = -kappa * (z - z_star)  # stable: df/dz < 0 at equilibrium

    # Potential: V'(z) = -f(z) -> V(z) = ∫ kappa (z - z*) dz = 0.5*kappa*(z-z*)^2 + const
    V = 0.5 * kappa * (z - z_star) ** 2
    V = V - V.min()

    fig = plt.figure(figsize=(8.0, 6.5))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(z, f, linewidth=2)
    ax1.axhline(0.0, linewidth=1)
    ax1.axvline(z_star, linestyle="--", linewidth=1)
    ax1.set_ylabel("drift  f(z)")
    ax1.set_title("Effective drift and potential (schematic)")
    ax1.text(z_star + 0.05, 0.05, r"$z^\*$", transform=ax1.get_xaxis_transform())

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(z, V, linewidth=2)
    ax2.axvline(z_star, linestyle="--", linewidth=1)
    ax2.set_xlabel("state  z = log(1 + butyrate)")
    ax2.set_ylabel("potential  V(z)")
    # Mark curvature at the minimum
    ax2.scatter([z_star], [0.0], s=40)
    ax2.text(z_star + 0.05, 0.05, r"min; curvature $\propto \kappa$", transform=ax2.get_xaxis_transform())

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"Wrote: {out_png}")

if __name__ == "__main__":
    main()
