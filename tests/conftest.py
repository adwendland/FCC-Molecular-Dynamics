"""Shared pytest fixtures for the FCC molecular-dynamics test suite."""

import numpy as np
import pytest

from md.constants import get_eps, get_lattice_constant, get_mass_internal, get_sigma
from md.lattice import make_fcc_lattice
from md.system import System


@pytest.fixture
def ni_parameters():
    """Consistent Lennard--Jones parameters for nickel."""
    sigma = get_sigma("Ni")
    return {
        "a": get_lattice_constant("Ni"),
        "mass": get_mass_internal("Ni"),
        "epsilon": get_eps("Ni"),
        "sigma": sigma,
        "rcut": 2.5 * sigma,
    }


@pytest.fixture
def small_ni_system(ni_parameters):
    """A deterministic 2x2x2 FCC Ni system suitable for smoke tests."""
    p = ni_parameters
    positions, box = make_fcc_lattice(p["a"], 2, 2, 2)
    system = System(
        positions,
        mass=p["mass"],
        box=box,
        symbol="Ni",
        cutoff=p["rcut"],
        skin=0.3,
    )

    rng = np.random.default_rng(12345)
    system.vel[:] = rng.normal(0.0, 1.0e-3, size=system.vel.shape)
    system.remove_drift()
    return system
