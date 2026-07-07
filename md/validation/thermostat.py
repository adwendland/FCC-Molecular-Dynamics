import numpy as np

from md.integrator import step_nvt_berendsen

kB = 8.617333262145e-5  # eV/K


def test_temperature_stability(
    system,
    dt,
    n_steps,
    T_target,
    tau_T,
    epsilon,
    sigma,
    rcut,
    sample_every=10,
):
    """
    Validate Berendsen NVT thermostat by checking that the mean temperature
    stays close to the target temperature.
    """

    times = []
    temperatures = []
    kinetic_energies = []
    velocities = []

    # initialize forces
    step_nvt_berendsen(
        system,
        dt=0.0,
        T_target=T_target,
        tau_T=tau_T,
        epsilon=epsilon,
        sigma=sigma,
        rcut=rcut,
    )

    for step in range(n_steps):
        step_nvt_berendsen(
            system,
            dt=dt,
            T_target=T_target,
            tau_T=tau_T,
            epsilon=epsilon,
            sigma=sigma,
            rcut=rcut,
        )

        if step % sample_every == 0:
            times.append(step * dt)
            temperatures.append(system.temperature())
            kinetic_energies.append(system.kinetic_energy)
            velocities.append(system.vel.copy())

    times = np.array(times)
    temperatures = np.array(temperatures)
    kinetic_energies = np.array(kinetic_energies)
    velocities = np.array(velocities)

    mean_T = np.mean(temperatures)
    std_T = np.std(temperatures)
    rel_error = abs(mean_T - T_target) / T_target

    return {
        "times": times,
        "temperatures": temperatures,
        "kinetic_energies": kinetic_energies,
        "velocities": velocities,
        "target_temperature": T_target,
        "mean_temperature": mean_T,
        "std_temperature": std_T,
        "relative_temperature_error": rel_error,
    }

def test_component_equipartition(
    system,
    dt,
    n_steps,
    T_target,
    tau_T,
    epsilon,
    sigma,
    rcut,
    sample_every=10,
):
    """
    Check that kinetic energy is equally distributed among the
    three translational degrees of freedom.

        <Kx> ≈ <Ky> ≈ <Kz> ≈ (1/2) N k_B T
    """

    result = test_temperature_stability(
        system=system,
        dt=dt,
        n_steps=n_steps,
        T_target=T_target,
        tau_T=tau_T,
        epsilon=epsilon,
        sigma=sigma,
        rcut=rcut,
        sample_every=sample_every,
    )

    velocities = result["velocities"]      # shape = (nsamples, N, 3)

    if np.isscalar(system.mass):
        masses = system.mass
    else:
        masses = system.mass[None, :]

    kinetic_x = 0.5 * np.sum(masses * velocities[:, :, 0] ** 2, axis=1)
    kinetic_y = 0.5 * np.sum(masses * velocities[:, :, 1] ** 2, axis=1)
    kinetic_z = 0.5 * np.sum(masses * velocities[:, :, 2] ** 2, axis=1)

    mean_kx = np.mean(kinetic_x)
    mean_ky = np.mean(kinetic_y)
    mean_kz = np.mean(kinetic_z)

    expected = 0.5 * system.N * kB * result["mean_temperature"]

    rel_error = max(
        abs(mean_kx - expected),
        abs(mean_ky - expected),
        abs(mean_kz - expected),
    ) / expected

    result.update(
        {
            "kinetic_x": kinetic_x,
            "kinetic_y": kinetic_y,
            "kinetic_z": kinetic_z,
            "mean_kinetic_x": mean_kx,
            "mean_kinetic_y": mean_ky,
            "mean_kinetic_z": mean_kz,
            "expected_component_energy": expected,
            "relative_component_equipartition_error": rel_error,
        }
    )

    return result

def test_total_kinetic_energy(
    system,
    dt,
    n_steps,
    T_target,
    tau_T,
    epsilon,
    sigma,
    rcut,
    sample_every=10,
):
    """
    Check whether average kinetic energy satisfies

        <K> ≈ (3/2) N kB T

    assuming center-of-mass motion has already been removed.
    """

    result = test_temperature_stability(
        system=system,
        dt=dt,
        n_steps=n_steps,
        T_target=T_target,
        tau_T=tau_T,
        epsilon=epsilon,
        sigma=sigma,
        rcut=rcut,
        sample_every=sample_every,
    )

    N = system.N
    K_measured = np.mean(result["kinetic_energies"])
    K_expected = 1.5 * N * kB * result["mean_temperature"]

    rel_error = abs(K_measured - K_expected) / K_expected

    result.update(
        {
            "mean_kinetic_energy": K_measured,
            "expected_kinetic_energy": K_expected,
            "relative_equipartition_error": rel_error,
        }
    )

    return result