# md/integrator.py

import numpy as np
from .forces import lj_forces

# Try to import the C++ extension
try:
    from . import md_cpp
    _HAVE_CPP = True
except ImportError:
    _HAVE_CPP = False

#_HAVE_CPP = False


def velocity_verlet(system, dt, epsilon=1.0, sigma=1.0, rcut=2.5):
    """
    One Velocity–Verlet time integration step using Lennard–Jones forces.

    If the C++ extension md_cpp is available and the mass is scalar,
    use the fast C++ implementation; otherwise fall back to pure Python.
    """
    m = system.mass

    # Use C++ path when possible
    if _HAVE_CPP and np.isscalar(m):
        # --- NEW: keep neighbor list in sync ---
        if hasattr(system, "nl") and system.nl is not None:
            system.nl.update(system.pos)
            pairs = system.nl.pairs
        else:
            # If for some reason there's no neighbor list, bail out to Python
            return _velocity_verlet_python(system, dt, epsilon, sigma, rcut)

        # Ensure int64 for C++
        pairs64 = pairs.astype(np.int64, copy=False)

        # Call C++ function: updates pos, vel, force in-place
        pe_new = md_cpp.velocity_verlet_lj_cpp(
            system.pos,
            system.vel,
            system.force,
            system.box,
            pairs64,
            float(m),
            float(dt),
            float(epsilon),
            float(sigma),
            float(rcut),
        )

        system.potential_energy = pe_new
        system.kinetic_energy()
        return
    
    # ----------------------------
    # Fallback: original Python/Numpy version
    # ----------------------------
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
    pe_new = system.compute_forces(
        lambda pos, box, pairs: lj_forces(pos, box, pairs, epsilon, sigma, rcut)
    )
    system.potential_energy = pe_new

    # 5) Second half-step velocity update
    if np.isscalar(m):
        inv_m = 1.0 / m
        system.vel += 0.5 * dt * system.force * inv_m
    else:
        system.vel += 0.5 * dt * (system.force / m[:, None])

    # 6) Update kinetic energy
    system.kinetic_energy()

    


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
    system.kinetic_energy()


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
    system.kinetic_energy()


# ============================================================
#                  USER-FRIENDLY INTEGRATION STEPS
# ============================================================

def step_nve(system, dt, epsilon=1.0, sigma=1.0, rcut=2.5):
    """Perform one NVE (microcanonical) MD step."""
    velocity_verlet(system, dt, epsilon=epsilon, sigma=sigma, rcut=rcut)


def step_nvt_berendsen(system, dt, T_target, tau_T,
                       epsilon=1.0, sigma=1.0, rcut=2.5):
    """
    Perform one NVT step using:
        - velocity Verlet
        - Berendsen thermostat
    """
    velocity_verlet(system, dt, epsilon=epsilon, sigma=sigma, rcut=rcut)
    berendsen_thermostat(system, T_target, tau_T, dt)
