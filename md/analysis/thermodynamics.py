import numpy as np
from md.integrator import step_nve

# Boltzmann constant in eV/K (same as in system.py)
kB = 8.617333262145e-5

def compute_pressure(system):
    """
    Instantaneous virial pressure in eV/Å^3.

    P = (2K + W) / (3V),

    where W = sum_{i<j} r_ij · F_ij.
    """
    V = system.volume()
    if V <= 0.0:
        return 0.0

    system.update_kinetic_energy()
    K = system.kinetic_energy
    virial = getattr(system, "virial", 0.0)

    return (2.0 * K + virial) / (3.0 * V)


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