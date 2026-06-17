import numpy as np
from md.integrator import step_nve

def compute_rdf(positions_traj, box, r_max, n_bins):
    """
    Compute radial distribution function g(r) from a trajectory.
    RDF tells likelihood of atom at distance r from center of reference particle

    positions_traj : array, shape (n_frames, N, 3)
        Atomic positions in Å.
    box : array-like, shape (3,)
        Box lengths in Å (orthorhombic).
    r_max : float
        Maximum radius for RDF in Å (usually <= box_min/2).
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    r : array, shape (n_bins,)
        Bin centers in Å.
    g_r : array, shape (n_bins,)
        Radial distribution function g(r).
    """
    n_frames, N, _ = positions_traj.shape
    box = np.asarray(box, dtype=float)
    rho = N / np.prod(box)

    dr = r_max / n_bins
    edges = np.linspace(0.0, r_max, n_bins + 1)
    hist = np.zeros(n_bins, dtype=float)

    for f in range(n_frames):
        pos = positions_traj[f]

        # double loop over i<j (bruteforce; you can optimize later)
        for i in range(N - 1):
            rij = pos[i+1:] - pos[i]
            # minimum image
            rij -= box * np.round(rij / box)
            r = np.linalg.norm(rij, axis=1)

            mask = (r > 0.0) & (r < r_max)
            r_valid = r[mask]
            bin_idx = (r_valid / dr).astype(int)
            np.add.at(hist, bin_idx, 2.0)   # factor 2 for i<->j

    # normalize
    r = 0.5 * (edges[:-1] + edges[1:])
    shell_vol = 4.0 * np.pi * r**2 * dr
    norm = n_frames * N * rho * shell_vol
    g_r = hist / norm

    return r, g_r


def compute_coordination_number(r, g_r, rho, r_max):
    """
    Coordination number from RDF:

        CN(r_max) = 4πρ ∫_0^{r_max} r^2 g(r) dr

    r : array
        Radii (Å).
    g_r : array
        RDF g(r).
    rho : float
        Number density N/V (1/Å^3).
    r_max : float
        Upper limit of integration (Å), e.g. up to first minimum of g(r).

    Returns
    -------
    CN : float
        Coordination number within r_max (dimensionless).
    """
    r = np.asarray(r, dtype=float)
    g_r = np.asarray(g_r, dtype=float)

    mask = r <= r_max
    r_sel = r[mask]
    g_sel = g_r[mask]

    integrand = r_sel**2 * g_sel
    CN = 4.0 * np.pi * rho * np.trapezoid(integrand, r_sel)

    return CN