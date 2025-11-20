# md/analysis.py

import numpy as np

# Boltzmann constant in eV/K (same as in system.py)
kB = 8.617333262145e-5


# ------------------------------------------------------------
# 1. RDF, MSD, VACF 
# ------------------------------------------------------------

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


def compute_msd(positions_traj, box):
    """
    Compute mean squared displacement MSD(t).
    MSD tells average distance displaced from start.

    positions_traj : array, shape (n_frames, N, 3)
        Trajectory of positions (Å).
    box : array-like, shape (3,)
        Box lengths (Å) for unwrapping with minimum image.

    Returns
    -------
    times : array, shape (n_frames,)
        Time indices (in units of your MD timestep, if you like).
    msd : array, shape (n_frames,)
        MSD(t) in Å^2.
    """
    pos = np.asarray(positions_traj)
    box = np.asarray(box, dtype=float)
    n_frames, N, _ = pos.shape

    # use frame 0 as reference
    r0 = pos[0]
    msd = np.zeros(n_frames, dtype=float)

    for t in range(n_frames):
        dr = pos[t] - r0
        dr -= box * np.round(dr / box)  # minimum image
        msd[t] = np.mean(np.sum(dr**2, axis=1))

    times = np.arange(n_frames, dtype=float)  # multiply by dt outside if you want
    return times, msd


def compute_vacf(velocities_traj):
    """
    Compute (normalized) velocity autocorrelation function VACF(t).
    Tells how the velocity of a particle at one point in time relates to another.

    velocities_traj : array, shape (n_frames, N, 3)
        Velocities (in internal units consistent with your forces & dt).

    Returns
    -------
    times : array, shape (n_frames,)
        Time indices (0, 1, 2, ...).
    vacf : array, shape (n_frames,)
        VACF(t) = < v(0) · v(t) > averaged over atoms and time origins.
    """
    v = np.asarray(velocities_traj)
    n_frames, N, _ = v.shape

    # subtract COM drift frame-by-frame (just in case)
    v = v - v.mean(axis=1, keepdims=True)

    vacf = np.zeros(n_frames, dtype=float)
    n_origins = np.zeros(n_frames, dtype=float)

    # time–origin averaging
    for t0 in range(n_frames):
        v0 = v[t0]
        max_tau = n_frames - t0
        dots = np.sum(v0 * v[t0:], axis=2)  # shape (max_tau, N)
        vacf[:max_tau] += dots.mean(axis=1)
        n_origins[:max_tau] += 1.0

    vacf /= np.maximum(n_origins, 1.0)
    times = np.arange(n_frames, dtype=float)
    return times, vacf


# ------------------------------------------------------------
# 2. Thermodynamic quantities: Pressure, Heat capacity
# ------------------------------------------------------------

def compute_pressure(system):
    """
    Instantaneous pressure from virial:

    P = (2K + sum_i r_i · F_i) / (3V)

    where:
        K = kinetic energy (eV)
        r_i, F_i in Å and eV/Å
        V = volume in Å^3

    Units: eV/Å^3 
    """
    V = system.volume()                    # Å^3
    K = system.kinetic_energy()            # eV
    virial = np.sum(system.pos * system.force)  # sum_i r_i · F_i, units eV

    P = (2.0 * K + virial) / (3.0 * V)     # eV/Å^3
    return P


def compute_heat_capacity_from_energy(E_series, T):
    """
    Heat capacity C_V from energy fluctuations (canonical ensemble):

        C_V = ( <E^2> - <E>^2 ) / (k_B T^2)

    E_series : array-like
        Time series of total energies (eV) in NVT.
    T : float
        Target temperature in K (or average instantaneous T).

    Returns
    -------
    C_V : float
        Heat capacity, in units of eV/K.
    """
    E = np.asarray(E_series, dtype=float)
    E_mean = E.mean()
    E2_mean = (E**2).mean()

    var_E = E2_mean - E_mean**2
    C_V = var_E / (kB * T**2)

    return C_V


# ------------------------------------------------------------
# 3. Structural extras: coordination number, structure factor
# ------------------------------------------------------------

def compute_coordination_from_rdf(r, g_r, rho, r_max):
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
    CN = 4.0 * np.pi * rho * np.trapz(integrand, r_sel)

    return CN


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

        S_k[i] = 1.0 + 4.0 * np.pi * rho * np.trapz(integrand, r)

    return k_values, S_k


# ------------------------------------------------------------
# 4. Dynamical extras: diffusion constants
# ------------------------------------------------------------

def compute_diffusion_from_msd(times, msd, t_min=None, t_max=None):
    """
    Diffusion coefficient from long-time MSD:

        MSD(t) ≈ 6 D t  (for large t in 3D)

    times : array
        Time points (in fs).
    msd : array
        MSD(t) in Å^2.
    t_min, t_max : floats or None
        Time window to fit the slope. If None, use last 1/3 of data.

    Returns
    -------
    D : float
        Diffusion coefficient in Å^2 / (time-unit-of-times-array).
    slope : float
        Fitted slope d(MSD)/dt (should be ~6D).
    """
    t = np.asarray(times, dtype=float)
    m = np.asarray(msd, dtype=float)

    if t_min is None or t_max is None:
        # use last third of the data as default "diffusive" regime
        n = len(t)
        start = 2 * n // 3
        t_fit = t[start:]
        m_fit = m[start:]
    else:
        mask = (t >= t_min) & (t <= t_max)
        t_fit = t[mask]
        m_fit = m[mask]

    # linear fit MSD = a + b t
    coeffs = np.polyfit(t_fit, m_fit, 1)
    slope = coeffs[0]
    D = slope / 6.0
    return D, slope


def compute_diffusion_from_vacf(times, vacf, t_max=None):
    """
    Diffusion coefficient from VACF using Green–Kubo:

        D = (1/3) ∫_0^∞ < v(0) · v(t) > dt

    Assumes VACF is the total dot product <v(0)·v(t)> averaged over atoms,
    *not* divided by 3 already. If VACF is per-component averaged,
    drop the 1/3 factor.

    times : array
        Time points (same units as dt [fs]).
    vacf : array
        VACF(t) in (velocity-unit)^2.
    t_max : float or None
        Upper limit of integration. If None, integrate whole array.

    Returns
    -------
    D : float
        Diffusion coefficient in (length^2 / time).
    integral : float
        The raw integral ∫ VACF(t) dt (without 1/3).
    """
    t = np.asarray(times, dtype=float)
    c = np.asarray(vacf, dtype=float)

    if t_max is not None:
        mask = t <= t_max
        t = t[mask]
        c = c[mask]

    integral = np.trapz(c, t)
    D = integral / 3.0
    return D, integral


# ------------------------------------------------------------
# 5. Green–Kubo: viscosity, thermal conductivity
# ------------------------------------------------------------

def compute_viscosity_green_kubo(times, corr_Pxy, V, T):
    """
    Shear viscosity from stress autocorrelation (Green–Kubo):

        η = V / (k_B T) ∫_0^∞ < P_xy(0) P_xy(t) > dt

    times : array
        Time points.
    corr_Pxy : array
        Autocorrelation function <P_xy(0) P_xy(t)>,
        with P_xy in units of pressure (eV/Å^3).
    V : float
        System volume in Å^3.
    T : float
        Temperature in K.

    Returns
    -------
    eta : float
        Viscosity η in units of (eV·fs)/(Å^3) if times in fs, etc.
    integral : float
        Value of the time integral of the correlation function.
    """
    t = np.asarray(times, dtype=float)
    C = np.asarray(corr_Pxy, dtype=float)

    integral = np.trapz(C, t)            # (pressure^2 * time)
    eta = V * integral / (kB * T)        # check unit conversions as needed

    return eta, integral


def compute_thermal_conductivity_green_kubo(times, corr_Jq, V, T):
    """
    Thermal conductivity κ from heat flux autocorrelation:

        κ = 1 / (k_B T^2 V) ∫_0^∞ < J_q(0) · J_q(t) > dt

    times : array
        Time points.
    corr_Jq : array
        Autocorrelation of heat current <J_q(0)·J_q(t)>,
        with J_q in "energy per area per time" units.
    V : float
        Volume in Å^3.
    T : float
        Temperature in K.

    Returns
    -------
    kappa : float
        Thermal conductivity
    integral : float
        The raw time integral.
    """
    t = np.asarray(times, dtype=float)
    C = np.asarray(corr_Jq, dtype=float)

    integral = np.trapz(C, t)
    kappa = integral / (kB * T**2 * V)

    return kappa, integral
