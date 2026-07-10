import numpy as np
import pytest

import md.integrator as integrator
from md.system import System


def make_two_particle_system():
    positions = np.array([[4.5, 5.0, 5.0], [5.5, 5.0, 5.0]])
    system = System(positions, mass=10.0, box=[10.0] * 3, cutoff=2.5, skin=0.3)
    system.vel[:] = [[0.0, 0.01, 0.0], [0.0, -0.01, 0.0]]
    return system


def test_python_velocity_verlet_preserves_total_momentum(monkeypatch):
    monkeypatch.setattr(integrator, "_HAVE_CPP", False)
    system = make_two_particle_system()
    initial_momentum = system.mass * system.vel.sum(axis=0)

    for _ in range(20):
        integrator.velocity_verlet(system, dt=0.001)

    final_momentum = system.mass * system.vel.sum(axis=0)
    np.testing.assert_allclose(final_momentum, initial_momentum, atol=1e-11)
    assert np.all(np.isfinite(system.pos))
    assert np.all(np.isfinite(system.vel))


def test_simple_rescale_reaches_target_temperature():
    system = make_two_particle_system()
    target = 300.0
    integrator.simple_rescale_thermostat(system, target)
    assert system.temperature() == pytest.approx(target)


def test_berendsen_moves_temperature_toward_target():
    system = make_two_particle_system()
    initial = system.temperature()
    target = 2.0 * initial

    integrator.berendsen_thermostat(system, T_target=target, tau_T=1.0, dt=0.1)
    assert initial < system.temperature() < target
