import numpy as np
from md.integrator import step_nve

# Boltzmann constant in eV/K (same as in system.py)
kB = 8.617333262145e-5

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
    system.update_kinetic_energy()
    K = system.kinetic_energy          # eV
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