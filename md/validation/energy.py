import numpy as np
from md.integrator import step_nve

def test_relative_energy_drift(
        system, dt, n_steps, epsilon, sigma, rcut,sample_every=1):
    """
    Run NVE and compute relative total energy drift.
    """

    times = []
    energies = []

    system.remove_drift()

    # Make sure forces + PE initialized
    step_nve(system, 0.0, epsilon=epsilon, sigma=sigma, rcut=rcut)

    system.update_energies()
    E0 = system.total_energy

    for step in range(n_steps):
        step_nve(system, dt, epsilon=epsilon, sigma=sigma, rcut=rcut)

        if step % sample_every == 0:
            system.update_energies()
            E = system.total_energy

            times.append(step * dt)
            energies.append(E)

    times = np.array(times)
    energies = np.array(energies)

    rel_drift = (energies - E0) / abs(E0)

    # Linear drift slope
    if len(times) > 1:
        coeffs = np.polyfit(times, energies, 1)
        drift_slope = coeffs[0]
    else:
        drift_slope = 0.0

    max_abs_rel = np.max(np.abs(rel_drift))

    return {
        "times": times,
        "energies": energies,
        "rel_drift": rel_drift,
        "drift_slope": drift_slope,
        "max_abs_rel": max_abs_rel
    }