# FCC Molecular Dynamics Simulator

[![Tests](https://github.com/adwendland/FCC-Molecular-Dynamics/actions/workflows/tests.yml/badge.svg)](https://github.com/adwendland/FCC-Molecular-Dynamics/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Scientific Computing](https://img.shields.io/badge/Scientific-Computing-orange.svg)
![Materials Science](https://img.shields.io/badge/Materials-FCC%20Metals-red.svg)

A modular molecular dynamics code for face-centered cubic (FCC) metals, written in Python with an optional pybind11/C++ backend for the computational kernels.

The project was built as an end-to-end scientific computing study: construct an FCC crystal, evolve it with classical molecular dynamics, analyze the resulting trajectories, verify the numerical implementation, and benchmark the code. It includes both NVE and NVT simulations, structural and transport analysis, an automated validation suite, interactive visualization, and reproducible performance tests.

![FCC Molecular Dynamics Simulator Streamlit dashboard](screenshots/img_streamlit_dashboard.png)

## What the project demonstrates

The simulator uses a 12–6 Lennard–Jones model for eight FCC metals: **Ag, Al, Au, Cu, Ni, Pb, Pd, and Pt**. Particle trajectories are integrated with Velocity Verlet under periodic boundary conditions using Verlet neighbor lists, with a Berendsen thermostat available for NVT simulations.

The analysis and validation tools are part of the same package rather than separate post-processing notebooks. A simulation can be followed directly by thermodynamic, structural, and dynamical analysis, or by automated numerical verification.

Representative results include:

- Recovery of the expected FCC first-neighbor distance for all eight metals at 300 K, with the first RDF peak within approximately **0.56%** of $a/\sqrt{2}$.
- First-shell coordination number **CN = 12.000** for all eight metals at 300 K.
- Bounded low-temperature MSDs consistent with atoms vibrating about lattice sites.
- Second-order timestep convergence and stable long-time NVE energy behavior.
- Temperature sweeps for Cu and Ni showing the coupled loss of crystalline RDF structure and onset of diffusive MSD growth. In the present Lennard–Jones model, Cu changes from solid-like to liquid-like behavior between **4000–6000 K**, while Ni changes between **6000–8000 K**.

These high-temperature transition intervals are model-dependent and are not intended as experimental melting-point predictions; they are used to demonstrate that independent structural and dynamical observables identify the same change in physical regime.

## Documentation

The repository includes longer technical reports for readers who want more than the README overview.

| Document | Purpose |
|---|---|
| [`Validation_Report.md`](docs/Validation_Report.md) | Numerical and physical verification: conservation laws, convergence, FCC structure, and thermostat behavior |
| [`Analysis_Report.md`](docs/Analysis_Report.md) | Thermodynamic, structural, and dynamical analysis, including the Cu and Ni temperature sweeps |
| [`TESTING.md`](TESTING.md) | Test organization, markers, coverage, and instructions for extending the pytest suite |

The LAMMPS comparison is intentionally being developed as a separate study so that external-code comparison remains distinct from the simulator's internal validation.

---

## Molecular dynamics model

Particle trajectories satisfy Newton's equations of motion,

```math
m_i \frac{d^2\mathbf r_i}{dt^2} = \mathbf F_i,
```

with pair interactions described by the 12–6 Lennard–Jones potential,

```math
U(r) = 4\varepsilon
\left[
\left(\frac{\sigma}{r}\right)^{12}
-
\left(\frac{\sigma}{r}\right)^6
\right].
```

Metal-specific lattice constants, atomic masses, and Lennard–Jones parameters are taken from Heinz *et al.* [1]. The code uses periodic boundary conditions, the minimum-image convention, and a Verlet neighbor list to avoid evaluating every particle pair at every timestep.

Time integration is performed with Velocity Verlet [2–4]. NVE production runs evolve without thermostatting, while NVT simulations use Berendsen temperature coupling [5]. Velocities are initialized from a Maxwell–Boltzmann distribution with center-of-mass drift removed.

The numerical core is available in pure Python/NumPy and through an optional pybind11 C++ extension. Keeping both implementations makes it possible to use the Python code as a readable reference while benchmarking an accelerated backend on the same simulation problem.

---

## Analysis

A saved trajectory can be reduced to thermodynamic, structural, and dynamical observables using the analysis package. The current workflow computes temperature, pressure, kinetic/potential/total energy, radial distribution functions, coordination number, static structure factor, mean squared displacement, velocity autocorrelation functions, and diffusion estimates from both MSD and VACF.

![Streamlit analysis report for Ni, 5x5x5 FCC, NVT, 300 K](screenshots/img_analysis_report.png)

The structural quantities are intended to be interpreted together. At low temperature, sharp RDF coordination shells and a coordination number near 12 identify the FCC crystal. At high temperature, the Cu and Ni sweeps show broadening and eventual loss of higher-order RDF structure at the same temperatures where the MSD changes from a bounded plateau to sustained growth.

See [`Analysis_Report.md`](docs/Analysis_Report.md) for the full discussion of the 300 K metal survey and the Cu/Ni temperature sweeps.

---

## Verification and validation

The automated validation suite checks the numerical implementation against conservation laws, expected convergence behavior, and known properties of an FCC crystal. Tests include NVE energy conservation, total momentum conservation, component equipartition, timestep refinement, RDF peak positions, first-shell coordination number, and NVT temperature stability.

Each validation run produces quantitative pass/fail metrics together with figures suitable for inspection or inclusion in a technical report.

![Streamlit validation summary](screenshots/img_validation_pass.png)

<p align="center">
  <img src="screenshots/img_validation_energy.png" alt="NVE energy conservation" width="48%">
  <img src="screenshots/img_validation_timestep.png" alt="Timestep convergence" width="48%">
</p>

The representative Ni validation case gives bounded NVE energy error with no secular drift, momentum conservation at floating-point precision, the expected second-order convergence of Velocity Verlet, and the correct FCC structural signatures. The same validation framework can be run for all supported metals.

For methodology, tolerances, equations, and representative numerical results, see [`Validation_Report.md`](docs/Validation_Report.md).

---

## Performance

The performance suite measures the main computational kernels over multiple FCC system sizes. It reports neighbor-list construction time, force-evaluation time, Velocity Verlet integration time, total runtime scaling, and atom-step throughput.

The same benchmark workflow can be run with the Python and C++ implementations, making the effect of compiled kernels directly measurable rather than anecdotal.

<p align="center">
  <img src="screenshots/img_performance_scaling.png" alt="Performance scaling with system size" width="48%">
  <img src="screenshots/img_kernel_timing.png" alt="Kernel timing breakdown" width="48%">
</p>

Further parallelization and a dedicated performance report are planned extensions.

---

## Interactive interfaces

The Streamlit application provides the most complete interactive interface to the project. It exposes simulation controls, three-dimensional atomic visualization, analysis, validation, and performance results in a single web application.

```bash
streamlit run streamlit_app.py
```

A desktop interface is also available:

```bash
python gui.py
```

![Desktop GUI](screenshots/img_gui_dashboard.png)

For scripted workflows, the repository includes example drivers:

```bash
python examples/run_analysis.py
python examples/run_validation.py
python examples/run_performance.py
```

---

## Installation

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/adwendland/FCC-Molecular-Dynamics.git
cd FCC-Molecular-Dynamics

python -m pip install -r requirements.txt
python -m pip install -e .
```

Building the editable package also builds the optional pybind11 extension when a compatible C++ toolchain is available.

For development and testing:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m pytest
```

The GitHub Actions workflow runs the test suite with coverage on Python 3.11, 3.12, and 3.13 for pushes and pull requests.

---

## Project structure

```text
FCC-Molecular-Dynamics/
├── md/
│   ├── analysis/              # RDF, MSD, VACF, S(k), thermodynamics, transport
│   ├── performance/           # Kernel and scaling benchmarks
│   ├── validation/            # Conservation, convergence, structure, thermostat tests
│   ├── constants.py           # Metal and unit-system constants
│   ├── forces.py              # Pair-force evaluation
│   ├── integrator.py          # Velocity Verlet and thermostat integration
│   ├── lattice.py             # FCC lattice construction
│   ├── neighborlist.py        # Verlet neighbor lists
│   ├── system.py              # Simulation state and system utilities
│   ├── plotting.py
│   ├── viz.py
│   └── md_cpp.cpp             # pybind11 C++ kernels
│
├── docs/
│   ├── Analysis_Report.md
│   ├── Validation_Report.md
├── examples/
│   ├── run_analysis.py
│   ├── run_performance.py
│   └── run_validation.py
├── figures/
├── outputs/
│   ├── analysis/
│   ├── validation/
├── tests/                     # pytest unit, scientific, and smoke tests
├── screenshots/               # README/report figures
├── .github/workflows/tests.yml
│
├── streamlit_app.py
├── gui.py
├── TESTING.md
├── pyproject.toml
├── setup.py
└── README.md
```

---

## Testing

The pytest suite covers the numerical core, analysis routines, validation helpers, and short end-to-end scientific smoke tests. Coverage configuration is defined in `pyproject.toml`.

```bash
python -m pytest
python -m pytest --cov=md --cov-report=term-missing
```

See [`TESTING.md`](TESTING.md) for details.

---

## Current scope and next steps

The code is intentionally a compact molecular dynamics implementation rather than a replacement for a production package such as LAMMPS. The current Lennard–Jones model is useful for studying the algorithms and analysis pipeline in a transparent setting, while also making the limitations of the model explicit.

The next planned extensions are external comparison against LAMMPS, shared-memory parallelization of the computational kernels, and a dedicated scaling/performance study. Longer-term directions include many-body metallic potentials such as EAM, additional ensembles, larger simulations, and defect-focused calculations.

---

## References

[1] H. Heinz, R. A. Vaia, B. L. Farmer, and R. R. Naik,  
“Accurate simulation of surfaces and interfaces of face-centered cubic metals using 12–6 and 9–6 Lennard-Jones potentials,”  
*J. Phys. Chem. C* **112** (2008), no. 44, 17281–17290.

[2] L. Verlet,  
“Computer ‘experiments’ on classical fluids. I. Thermodynamical properties of Lennard-Jones molecules,”  
*Phys. Rev.* **159** (1967), 98–103.

[3] D. Frenkel and B. Smit,  
*Understanding Molecular Simulation: From Algorithms to Applications*, 2nd ed., Academic Press, San Diego, 2002.

[4] M. P. Allen and D. J. Tildesley,  
*Computer Simulation of Liquids*, 2nd ed., Oxford University Press, Oxford, 2017.

[5] H. J. C. Berendsen, J. P. M. Postma, W. F. van Gunsteren, A. DiNola, and J. R. Haak,  
“Molecular dynamics with coupling to an external bath,”  
*J. Chem. Phys.* **81** (1984), 3684–3690.

---

## License

MIT License.
