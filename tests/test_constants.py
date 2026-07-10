import numpy as np
import pytest

from md.constants import (
    AMU_TO_INTERNAL,
    get_amu,
    get_eps,
    get_lattice_constant,
    get_mass_internal,
    get_sigma,
)

METALS = ["Ag", "Al", "Au", "Cu", "Ni", "Pb", "Pd", "Pt"]


@pytest.mark.parametrize("metal", METALS)
def test_material_parameters_are_positive(metal):
    """Every supported metal should have physically meaningful parameters."""
    assert get_lattice_constant(metal) > 0.0
    assert get_amu(metal) > 0.0
    assert get_sigma(metal) > 0.0
    assert get_eps(metal) > 0.0


def test_internal_mass_conversion():
    assert np.isclose(get_mass_internal("Ni"), get_amu("Ni") * AMU_TO_INTERNAL)


def test_known_nickel_lattice_constant():
    assert get_lattice_constant("Ni") == pytest.approx(3.52)
