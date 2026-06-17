import numpy as np
from md.integrator import step_nve

# Boltzmann constant in eV/K (same as in system.py)
kB = 8.617333262145e-5

def compute_viscosity(times, corr_Pxy, V, T):
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

    integral = np.trapezoid(C, t)            # (pressure^2 * time)
    eta = V * integral / (kB * T)        # check unit conversions as needed

    return eta, integral


def compute_thermal_conductivity(times, corr_Jq, V, T):
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

    integral = np.trapezoid(C, t)
    kappa = integral / (kB * T**2 * V)

    return kappa, integral

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

    integral = np.trapezoid(c, t)
    D = integral / 3.0
    return D, integral

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