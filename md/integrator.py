# md/integrator.py

import numpy as np
from .forces import lj_forces

# Try to import the C++ extension
try:
    from . import md_cpp
    _HAVE_CPP = True
except ImportError:
    _HAVE_CPP = False

BACKENDS = ("python", "cpp")


def cpp_backend_available():
    """Return True when the optional pybind11/C++ extension is importable."""
    return _HAVE_CPP


def resolve_backend(backend):
    """Validate and normalize an explicitly selected integration backend."""
    name = str(backend).strip().lower()
    if name in {"c++", "pybind11"}:
        name = "cpp"

    if name not in BACKENDS:
        raise ValueError(
            f"backend must be one of {BACKENDS}; got {backend!r}. "
            "Automatic backend selection is intentionally not supported."
        )

    if name == "cpp" and not _HAVE_CPP:
        raise RuntimeError(
            "C++ backend requested, but md.md_cpp is not available. "
            "Build/install the pybind11 extension or select backend='python'."
        )

    return name


def velocity_verlet(system, dt, epsilon=1.0, sigma=1.0, rcut=2.5, backend="python"):
    """
    One Velocity–Verlet time integration step using Lennard–Jones forces.

    The backend is explicit: ``backend='python'`` always uses NumPy and
    ``backend='cpp'`` always uses the pybind11/C++ implementation.

    A requested C++ backend never silently falls back to Python.
    """
    backend = resolve_backend(backend)
    m = system.mass

    
    if backend == "cpp":
        if not np.isscalar(m):
            raise ValueError(
                "C++ backend requires a scalar particle mass."
            )

        if not hasattr(system, "nl") or system.nl is None:
            raise RuntimeError(
                "C++ backend requires an initialized neighbor list."
            )

        # 1. Update neighbor list at current positions
        system.nl.update(system.pos)
        pairs64 = np.ascontiguousarray(
            system.nl.pairs, dtype=np.int64
        )

        # 2. Compute forces at current positions
        md_cpp.lj_forces_cpp(
            system.pos,
            system.force,
            system.box,
            pairs64,
            float(epsilon),
            float(sigma),
            float(rcut),
        )

        # 3. First half-kick and full drift
        md_cpp.verlet_drift_cpp(
            system.pos,
            system.vel,
            system.force,
            system.box,
            float(m),
            float(dt),
        )

        # 4. IMPORTANT: update neighbor list at NEW positions
        system.nl.update(system.pos)

        # Obtain the updated pairs
        pairs64 = np.ascontiguousarray(
            system.nl.pairs, dtype=np.int64
        )

        # 5. Compute forces at new positions
        pe_new, virial_new = md_cpp.lj_forces_cpp(
            system.pos,
            system.force,
            system.box,
            pairs64,
            float(epsilon),
            float(sigma),
            float(rcut),
        )

        # 6. Second half-kick
        md_cpp.verlet_kick_cpp(
            system.vel,
            system.force,
            float(m),
            float(dt),
        )

        # 7. Update energies
        system.potential_energy = float(pe_new)
        system.virial = float(virial_new)
        system.update_energies()

        return

    return _velocity_verlet_python(system, dt, epsilon, sigma, rcut)


### Python version of velocity Verlet
def _velocity_verlet_python(system, dt, epsilon, sigma, rcut):
    m = system.mass

    # 1) Compute forces at current positions
    pe = system.compute_forces(
        lambda pos, box, pairs: lj_forces(pos, box, pairs, epsilon, sigma, rcut)
    )

    # 2) Half-step velocity update and full-step position update
    if np.isscalar(m):
        inv_m = 1.0 / m
        system.vel += 0.5 * dt * system.force * inv_m
        system.pos += dt * system.vel
    else:
        system.vel += 0.5 * dt * (system.force / m[:, None])
        system.pos += dt * system.vel

    # 3) Apply periodic boundary conditions
    system.pos %= system.box

    # 4) Recompute forces at new positions
    system.compute_forces(
        lambda pos, box, pairs: lj_forces(pos, box, pairs, epsilon, sigma, rcut)
    )   

    # 5) Second half-step velocity update
    if np.isscalar(m):
        inv_m = 1.0 / m
        system.vel += 0.5 * dt * system.force * inv_m
    else:
        system.vel += 0.5 * dt * (system.force / m[:, None])

    # 6) Update kinetic energy
    system.update_energies()

    


# ============================================================
#                    THERMOSTATS (for NVT)
# ============================================================

def berendsen_thermostat(system, T_target, tau_T, dt):
    """
    Berendsen weak-coupling thermostat.
    Scales velocities smoothly toward the target temperature.

    dT/dt = (T_target - T)/tau_T
    """
    T_inst = system.temperature()
    if T_inst <= 0.0:
        return

    # scaling factor
    lam2 = 1.0 + (dt / tau_T) * (T_target / T_inst - 1.0)
    if lam2 < 0.0:
        return

    lam = np.sqrt(lam2)
    system.vel *= lam
    system.update_energies()


def simple_rescale_thermostat(system, T_target):
    """
    Instant velocity-rescale thermostat.
    Brings temperature exactly to T_target in one step.
    Only use for equilibration
    """
    T_inst = system.temperature()
    if T_inst <= 0.0:
        return

    lam = np.sqrt(T_target / T_inst)
    system.vel *= lam
    system.update_energies()


# ============================================================
#                  USER-FRIENDLY INTEGRATION STEPS
# ============================================================

def step_nve(system, dt, epsilon=1.0, sigma=1.0, rcut=2.5, backend="python"):
    """Perform one NVE (microcanonical) MD step."""
    velocity_verlet(
        system,
        dt,
        epsilon=epsilon,
        sigma=sigma,
        rcut=rcut,
        backend=backend,
    )


def step_nvt_berendsen(system, dt, T_target, tau_T,
                       epsilon=1.0, sigma=1.0, rcut=2.5, backend="python"):
    """
    Perform one NVT step using:
        - velocity Verlet
        - Berendsen thermostat
    """
    velocity_verlet(
        system,
        dt,
        epsilon=epsilon,
        sigma=sigma,
        rcut=rcut,
        backend=backend,
    )
    berendsen_thermostat(system, T_target, tau_T, dt)
