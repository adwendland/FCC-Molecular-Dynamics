import numpy as np
import pytest

from md.system import System, kB


def make_system(n=2, mass=2.0):
    positions = np.column_stack((np.arange(n, dtype=float), np.zeros((n, 2))))
    return System(positions, mass=mass, box=[10.0, 10.0, 10.0], cutoff=2.5)


def test_volume_and_periodic_wrapping():
    system = make_system()
    system.box = np.array([2.0, 3.0, 4.0])
    system.pos[0] = [-0.5, 3.5, 8.5]
    system.apply_pbc()

    assert system.volume() == pytest.approx(24.0)
    np.testing.assert_allclose(system.pos[0], [1.5, 0.5, 0.5])


def test_scalar_mass_kinetic_energy():
    system = make_system(mass=2.0)
    system.vel[:] = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    assert system.update_kinetic_energy() == pytest.approx(5.0)


def test_particle_mass_array_kinetic_energy_and_total_energy():
    system = make_system(mass=np.array([2.0, 4.0]))
    system.vel[:] = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    system.potential_energy = -3.0
    system.update_energies()

    assert system.kinetic_energy == pytest.approx(9.0)
    assert system.total_energy == pytest.approx(6.0)


def test_temperature_matches_equipartition_definition():
    system = make_system(n=2, mass=2.0)
    system.vel[:] = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    expected = 2.0 * 2.0 / ((3 * system.N - 3) * kB)
    assert system.temperature() == pytest.approx(expected)


def test_remove_drift_zeroes_center_of_mass_velocity():
    system = make_system(n=3)
    system.vel[:] = [[2.0, 1.0, 0.0], [3.0, 1.0, 0.0], [4.0, 1.0, 0.0]]
    system.remove_drift()

    np.testing.assert_allclose(system.vel.mean(axis=0), 0.0, atol=1e-14)
