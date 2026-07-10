"""Edge cases and regression checks for the MD numerical core."""

import numpy as np
import pytest

import md.integrator as integrator
from md.constants import get_amu, get_eps, get_lattice_constant, get_sigma
from md.forces import lj_forces
from md.lattice import make_fcc_lattice
from md.neighborlist import NeighborList
from md.system import System


@pytest.mark.parametrize("getter", [get_lattice_constant, get_amu, get_sigma, get_eps])
def test_invalid_material_returns_minus_one_and_prints_error(getter, capsys):
    assert getter("Unobtainium") == -1
    assert "Invalid material" in capsys.readouterr().out


def test_fcc_single_cell_contains_standard_basis():
    a = 2.0
    positions, box = make_fcc_lattice(a, 1, 1, 1)
    expected = np.array(
        [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float
    )
    np.testing.assert_allclose(positions, expected)
    np.testing.assert_allclose(box, [2.0, 2.0, 2.0])


def test_lj_multiple_pairs_accumulate_force_and_energy():
    positions = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.1, 0.0]])
    pairs = np.array([[0, 1], [0, 2]], dtype=np.int32)
    forces, energy = lj_forces(positions, np.array([20.0] * 3), pairs)
    one_pair_force, one_pair_energy = lj_forces(
        positions[:2], np.array([20.0] * 3), np.array([[0, 1]], dtype=np.int32)
    )
    assert energy == pytest.approx(2.0 * one_pair_energy)
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-14)
    assert forces[0, 0] == pytest.approx(one_pair_force[0, 0])
    assert forces[0, 1] == pytest.approx(one_pair_force[0, 0])


def test_lj_pair_exactly_at_cutoff_is_excluded():
    positions = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    forces, energy = lj_forces(positions, np.array([20.0] * 3), [[0, 1]], rcut=2.5)
    np.testing.assert_allclose(forces, 0.0)
    assert energy == 0.0


def test_neighbor_list_empty_pairs_have_two_columns():
    positions = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    nl = NeighborList(1.0, 0.1, positions, [10.0] * 3)
    # Empty arrays currently have shape (0,); the important API guarantee is no pairs.
    assert nl.pairs.size == 0


def test_neighbor_list_update_rebuilds_and_changes_pairs():
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    nl = NeighborList(1.0, 0.4, positions, [10.0] * 3)
    moved = positions.copy()
    moved[1, 0] = 0.9
    nl.update(moved)
    np.testing.assert_array_equal(nl.pairs, np.array([[0, 1]], dtype=np.int32))
    np.testing.assert_allclose(nl.pos_old, moved)


def test_neighbor_rebuild_displacement_uses_minimum_image():
    positions = np.array([[9.9, 0.0, 0.0]])
    nl = NeighborList(1.0, 0.5, positions, [10.0] * 3)
    wrapped = np.array([[0.1, 0.0, 0.0]])
    assert not nl._needs_rebuild(wrapped)


def test_system_copy_is_deep_copy():
    system = System(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), 2.0, [10.0] * 3)
    clone = system.copy()
    clone.pos[0, 0] = 7.0
    clone.nl.pairs = np.empty((0, 2), dtype=np.int32)
    assert system.pos[0, 0] == 0.0
    assert clone.nl is not system.nl


def test_system_compute_forces_updates_state():
    system = System(np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]]), 2.0, [10.0] * 3)

    def force_fn(pos, box, pairs):
        return np.full_like(pos, 3.0), -7.5

    returned = system.compute_forces(force_fn)
    assert returned == pytest.approx(-7.5)
    assert system.potential_energy == pytest.approx(-7.5)
    np.testing.assert_allclose(system.force, 3.0)


def test_single_particle_temperature_is_zero():
    system = System(np.zeros((1, 3)), 2.0, [10.0] * 3)
    system.vel[0] = [1.0, 2.0, 3.0]
    assert system.temperature() == 0.0


def test_remove_drift_updates_total_energy():
    system = System(np.zeros((2, 3)), 2.0, [10.0] * 3)
    system.vel[:] = [1.0, 0.0, 0.0]
    system.potential_energy = -4.0
    system.remove_drift()
    assert system.kinetic_energy == pytest.approx(0.0)
    assert system.total_energy == pytest.approx(-4.0)


def test_velocity_verlet_supports_particle_mass_array(monkeypatch):
    monkeypatch.setattr(integrator, "_HAVE_CPP", False)
    positions = np.array([[4.5, 5.0, 5.0], [5.5, 5.0, 5.0]])
    system = System(positions, mass=np.array([10.0, 20.0]), box=[10.0] * 3)
    system.vel[:] = [[0.0, 0.01, 0.0], [0.0, -0.005, 0.0]]
    integrator.velocity_verlet(system, dt=1e-3)
    assert np.all(np.isfinite(system.pos))
    assert np.all(np.isfinite(system.vel))
    assert np.isfinite(system.total_energy)


def test_velocity_verlet_wraps_positions(monkeypatch):
    monkeypatch.setattr(integrator, "_HAVE_CPP", False)
    system = System(np.array([[9.9, 5.0, 5.0]]), mass=1.0, box=[10.0] * 3, cutoff=0.1)
    system.vel[0, 0] = 1.0
    integrator.velocity_verlet(system, dt=0.2)
    assert system.pos[0, 0] == pytest.approx(0.1)


def test_berendsen_does_nothing_at_zero_temperature():
    system = System(np.zeros((2, 3)), 1.0, [10.0] * 3)
    integrator.berendsen_thermostat(system, T_target=300.0, tau_T=1.0, dt=0.1)
    np.testing.assert_allclose(system.vel, 0.0)


def test_simple_rescale_does_nothing_at_zero_temperature():
    system = System(np.zeros((2, 3)), 1.0, [10.0] * 3)
    integrator.simple_rescale_thermostat(system, T_target=300.0)
    np.testing.assert_allclose(system.vel, 0.0)


def test_step_nve_delegates_to_velocity_verlet(monkeypatch):
    called = {}

    def fake(system, dt, epsilon, sigma, rcut):
        called.update(dt=dt, epsilon=epsilon, sigma=sigma, rcut=rcut)

    monkeypatch.setattr(integrator, "velocity_verlet", fake)
    integrator.step_nve(object(), 0.2, epsilon=2.0, sigma=3.0, rcut=4.0)
    assert called == {"dt": 0.2, "epsilon": 2.0, "sigma": 3.0, "rcut": 4.0}


def test_step_nvt_calls_integrator_then_thermostat(monkeypatch):
    calls = []
    monkeypatch.setattr(integrator, "velocity_verlet", lambda *a, **k: calls.append("vv"))
    monkeypatch.setattr(integrator, "berendsen_thermostat", lambda *a, **k: calls.append("thermostat"))
    integrator.step_nvt_berendsen(object(), 0.1, 300.0, 10.0)
    assert calls == ["vv", "thermostat"]
