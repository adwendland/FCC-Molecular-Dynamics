# Testing guide

This project uses **pytest** for automated tests and **GitHub Actions** for continuous integration.

## Run the suite

From the `MD_FCC` directory:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m pytest
```

Useful variants:

```bash
# Show every test name
python -m pytest -v

# Stop at the first failure
python -m pytest -x

# Run one file
python -m pytest tests/test_validation_helpers.py

# Run one test
python -m pytest tests/test_forces.py::test_lj_force_obeys_newtons_third_law

# Run with line coverage
python -m pytest --cov=md --cov-report=term-missing

# Skip the small end-to-end scientific smoke tests
python -m pytest -m "not scientific"
```

Use `python -m pytest`, rather than only `pytest`, so the active Python interpreter and the project package are resolved consistently.

## What is tested

The suite is divided into four layers:

1. **Unit tests** check small functions with exact or analytically known answers: material constants, FCC geometry, Lennard--Jones forces, neighbor lists, thermostats, and analysis formulas.
2. **Validation-helper tests** check the calculations inside energy, momentum, convergence, structure, and thermostat validation modules using deterministic stand-ins.
3. **Scientific smoke tests** run short, real molecular-dynamics calculations through the lattice, system, neighbor-list, force, integrator, and validation code.
4. **CI tests** run the entire suite on Linux under Python 3.11, 3.12, and 3.13 whenever code is pushed or a pull request is opened.

The short pytest suite complements, rather than replaces, long production validation runs and external comparisons against LAMMPS.

## Reading a test

Most tests follow the arrange--act--assert pattern:

```python
def test_lj_force_obeys_newtons_third_law():
    # Arrange a two-particle system.
    positions = ...

    # Act by evaluating the force.
    forces, _ = lj_forces(...)

    # Assert the physical invariant.
    np.testing.assert_allclose(forces[0], -forces[1])
```

A failed assertion identifies the property that changed. This is especially useful after optimizing the C++ extension, neighbor list, or integrator.

## Test markers

Tests marked `scientific` are still fast, but they exercise several real modules together:

```python
@pytest.mark.scientific
def test_short_nve_energy_drift_is_small(...):
    ...
```

Run only those tests with:

```bash
python -m pytest -m scientific
```

## GitHub Actions

The workflow is stored at `.github/workflows/tests.yml`. It:

- installs runtime and development dependencies;
- installs the package in editable mode;
- builds the optional pybind11 extension;
- runs pytest with coverage;
- uploads `coverage.xml` for each Python version.

A green check on a commit means the project built and passed the suite in a clean Linux environment.
