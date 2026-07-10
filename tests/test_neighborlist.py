import numpy as np

from md.neighborlist import NeighborList


def test_neighbor_list_contains_each_pair_once():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    nl = NeighborList(cutoff=1.2, skin=0.2, positions=positions, box=[10.0] * 3)

    np.testing.assert_array_equal(nl.pairs, np.array([[0, 1]], dtype=np.int32))


def test_neighbor_list_detects_periodic_neighbors():
    positions = np.array([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]])
    nl = NeighborList(cutoff=0.3, skin=0.1, positions=positions, box=[10.0] * 3)

    np.testing.assert_array_equal(nl.pairs, np.array([[0, 1]], dtype=np.int32))


def test_neighbor_list_rebuild_threshold():
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    nl = NeighborList(cutoff=1.0, skin=0.4, positions=positions, box=[10.0] * 3)

    small_move = positions.copy()
    small_move[0, 0] += 0.19
    assert not nl._needs_rebuild(small_move)

    large_move = positions.copy()
    large_move[0, 0] += 0.21
    assert nl._needs_rebuild(large_move)
