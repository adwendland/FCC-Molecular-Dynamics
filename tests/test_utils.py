from types import SimpleNamespace

import numpy as np

from md.utils import write_xyz


def test_write_xyz_creates_expected_frame(tmp_path):
    system = SimpleNamespace(
        N=2,
        box=np.array([5.0, 6.0, 7.0]),
        symbol="Ni",
        pos=np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
    )
    path = tmp_path / "trajectory.xyz"

    write_xyz(system, step=0, filename=path)
    lines = path.read_text().splitlines()

    assert lines[0] == "2"
    assert "Step=0" in lines[1]
    assert 'Lattice="5.0 0 0  0 6.0 0  0 0 7.0"' in lines[1]
    assert lines[2] == "Ni 0.00000000 1.00000000 2.00000000"
    assert lines[3] == "Ni 3.00000000 4.00000000 5.00000000"
