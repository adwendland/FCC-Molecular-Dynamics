import numpy as np
from md.integrator import step_nve


def total_momentum(system):
    if np.isscalar(system.mass):
        return system.mass * np.sum(system.vel, axis=0)
    else:
        return np.sum(system.mass[:, None] * system.vel, axis=0)


def momentum_scale(system):
    """
    Natural momentum scale: sum_i m_i |v_i|.
    """
    if np.isscalar(system.mass):
        return system.mass * np.sum(np.linalg.norm(system.vel, axis=1))
    else:
        return np.sum(system.mass * np.linalg.norm(system.vel, axis=1))


def test_momentum_conservation(
    system, dt, n_steps, epsilon, sigma, rcut, sample_every=1
):
    times = []
    momenta = []
    momentum_norms = []

    P0 = total_momentum(system)
    P_scale = momentum_scale(system)

    step_nve(system, 0.0, epsilon=epsilon, sigma=sigma, rcut=rcut)

    for step in range(n_steps):
        step_nve(system, dt, epsilon=epsilon, sigma=sigma, rcut=rcut)

        if step % sample_every == 0:
            P = total_momentum(system)
            times.append(step * dt)
            momenta.append(P)
            momentum_norms.append(np.linalg.norm(P))

    times = np.array(times)
    momenta = np.array(momenta)
    momentum_norms = np.array(momentum_norms)

    delta_P = momenta - P0
    delta_P_norm = np.linalg.norm(delta_P, axis=1)

    normalized_momentum_drift = delta_P_norm / P_scale

    return {
        "times": times,
        "momenta": momenta,
        "momentum_norms": momentum_norms,
        "initial_momentum": P0,
        "final_momentum": momenta[-1],
        "delta_P": delta_P,
        "delta_P_norm": delta_P_norm,
        "normalized_momentum_drift": normalized_momentum_drift,
        "max_abs_drift": np.max(delta_P_norm),
        "max_rel_drift": np.max(normalized_momentum_drift),
        "momentum_scale": P_scale,
    }