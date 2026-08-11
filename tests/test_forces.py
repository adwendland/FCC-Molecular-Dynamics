import numpy as np
import pytest

from md.forces import lj_forces


def two_particle_result(distance, *, epsilon=1.0, sigma=1.0, rcut=2.5):
    positions = np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
    box = np.array([20.0, 20.0, 20.0])
    pairs = np.array([[0, 1]], dtype=np.int32)
    return lj_forces(positions, box, pairs, epsilon, sigma, rcut)


def test_lj_force_obeys_newtons_third_law():
    forces, _, _ = two_particle_result(1.1)
    np.testing.assert_allclose(forces[0], -forces[1])
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-14)


def test_lj_potential_is_zero_at_sigma():
    _, potential, _ = two_particle_result(1.0)
    assert potential == pytest.approx(0.0, abs=1e-14)


def test_lj_force_is_zero_at_equilibrium_distance():
    r_equilibrium = 2.0 ** (1.0 / 6.0)
    forces, potential, _ = two_particle_result(r_equilibrium)

    np.testing.assert_allclose(forces, 0.0, atol=1e-12)
    assert potential == pytest.approx(-1.0)


def test_lj_ignores_pairs_outside_cutoff():
    forces, potential, _ = two_particle_result(3.0, rcut=2.5)
    np.testing.assert_allclose(forces, 0.0)
    assert potential == 0.0


def test_lj_uses_minimum_image_periodic_distance():
    positions = np.array([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]])
    box = np.array([10.0, 10.0, 10.0])
    pairs = np.array([[0, 1]], dtype=np.int32)

    forces, _, _ = lj_forces(positions, box, pairs, rcut=1.0)
    assert np.linalg.norm(forces[0]) > 0.0
    np.testing.assert_allclose(forces[0], -forces[1])

def test_lj_returns_correct_pair_virial():
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.1, 0.0, 0.0],
    ])

    box = np.array([10.0, 10.0, 10.0])
    pairs = np.array([[0, 1]], dtype=np.int32)

    forces, _, virial = lj_forces(
        positions,
        box,
        pairs,
        epsilon=1.0,
        sigma=1.0,
        rcut=2.5,
    )

    rij = positions[0] - positions[1]
    expected = np.dot(rij, forces[0])

    assert virial == pytest.approx(expected)

def test_lj_virial_is_zero_outside_cutoff():
    _, _, virial = two_particle_result(3.0, rcut=2.5)

    assert virial == pytest.approx(0.0)