import numpy as np
import pytest

from md.lattice import make_fcc_lattice


def test_fcc_atom_count_and_box():
    a = 3.52
    positions, box = make_fcc_lattice(a, nx=2, ny=3, nz=4)

    assert positions.shape == (4 * 2 * 3 * 4, 3)
    np.testing.assert_allclose(box, [2 * a, 3 * a, 4 * a])


def test_fcc_positions_are_inside_box_and_unique():
    positions, box = make_fcc_lattice(3.52, 2, 2, 2)

    assert np.all(positions >= 0.0)
    assert np.all(positions < box)
    assert len(np.unique(positions, axis=0)) == len(positions)


def test_fcc_nearest_neighbor_distance():
    a = 3.52
    positions, _ = make_fcc_lattice(a, 1, 1, 1)
    distances = np.linalg.norm(positions[1:] - positions[0], axis=1)

    assert np.min(distances) == pytest.approx(a / np.sqrt(2.0))
