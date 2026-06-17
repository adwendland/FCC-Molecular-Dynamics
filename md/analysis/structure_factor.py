import numpy as np
from md.integrator import step_nve

def compute_structure_factor(k_values, r, g_r, rho):
    """
    Static structure factor S(k) from RDF:

        S(k) = 1 + 4πρ ∫_0^∞ r^2 [g(r) - 1] sin(kr)/(kr) dr

    k_values : array
        Wavevectors k (1/Å).
    r : array
        Radii (Å).
    g_r : array
        RDF g(r).
    rho : float
        Number density N/V (1/Å^3).

    Returns
    -------
    k_values : array
        Same as input.
    S_k : array
        Structure factor S(k).
    """
    k_values = np.asarray(k_values, dtype=float)
    r = np.asarray(r, dtype=float)
    g_r = np.asarray(g_r, dtype=float)

    gr_minus_1 = g_r - 1.0
    S_k = np.zeros_like(k_values)

    for i, k in enumerate(k_values):
        if k == 0.0:
            # limit k→0: S(0) = 1 + 4πρ ∫ r^2 [g(r)-1] dr
            integrand = r**2 * gr_minus_1
        else:
            kr = k * r
            sinc = np.sin(kr) / kr
            integrand = r**2 * gr_minus_1 * sinc

        S_k[i] = 1.0 + 4.0 * np.pi * rho * np.trapezoid(integrand, r)

    return k_values, S_k