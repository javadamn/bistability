#!/usr/bin/env python3
# Python 3.10; deps: numpy, matplotlib
import numpy as np
import matplotlib.pyplot as plt

def euler_maruyama_doublewell(T=600.0, dt=0.01, sigma=0.6, a=0.10, seed=0):
    """
    dz = f(z) dt + sigma dW
    with potential V(z)=(z^2-1)^2/4 - a z, drift f(z) = -dV/dz = -z^3 + z + a
    """
    rng = np.random.default_rng(seed)
    n = int(T / dt)
    z = np.zeros(n + 1, dtype=float)
    z[0] = 0.0
    sqrt_dt = np.sqrt(dt)
    for i in range(n):
        zi = z[i]
        f = (-zi**3 + zi + a)
        z[i + 1] = zi + f * dt + sigma * sqrt_dt * rng.normal()
    t = np.linspace(0, T, n + 1)
    return t, z

def main(out_png="fig_sampling_schematic.png"):
    # Simulate a bistable system
    t, z = euler_maruyama_doublewell(T=600, dt=0.01, sigma=0.6, a=0.10, seed=0)

    dense_every = 1
    sparse_every = 225
    t_dense, z_dense = t[::dense_every], z[::dense_every]
    t_sparse, z_sparse = t[::sparse_every], z[::sparse_every]

    fig = plt.figure(figsize=(10.5, 6.0))

    # Dense
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(t_dense[:6000], z_dense[:6000], linewidth=1)  # show early segment
    ax1.set_title("Dense sampling")
    ax1.set_xlabel("time")
    ax1.set_ylabel("state z")

    # Sparse
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(t_sparse[:200], z_sparse[:200], marker="o", linewidth=1, markersize=3)
    ax2.set_title("Sparse sampling")
    ax2.set_xlabel("time")
    ax2.set_ylabel("state z")

    # Show observed points on the same underlying segment
    ax3 = fig.add_subplot(2, 1, 2)
    # underlying continuous-ish path
    ax3.plot(t_dense[:6000], z_dense[:6000], linewidth=1)
    # overlay sparse points
    mask = (t_sparse <= t_dense[5999])
    ax3.plot(t_sparse[mask], z_sparse[mask], marker="o", linewidth=0, markersize=4)
    ax3.set_title("Same trajectory segment: sparse observations may miss switches")
    ax3.set_xlabel("time")
    ax3.set_ylabel("state z")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"Wrote: {out_png}")

if __name__ == "__main__":
    main()
