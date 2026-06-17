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

    times = np.array(times)
    temperatures = np.array(temperatures)
    kinetic_energies = np.array(kinetic_energies)

    mean_T = np.mean(temperatures)
    std_T = np.std(temperatures)
    rel_error = abs(mean_T - T_target) / T_target

    return {
        "times": times,
        "temperatures": temperatures,
        "kinetic_energies": kinetic_energies,
        "target_temperature": T_target,
        "mean_temperature": mean_T,
        "std_temperature": std_T,
        "relative_temperature_error": rel_error,
    }


def test_equipartition(
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