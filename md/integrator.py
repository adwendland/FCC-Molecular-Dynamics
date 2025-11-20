# md/integrator.py

import numpy as np
from .forces import lj_forces


def velocity_verlet(system, dt, epsilon=1.0, sigma=1.0, rcut=2.5):
    """
    One Velocity–Verlet time integration step using Lennard–Jones forces.

    Updates the System in-place:
        system.pos
        system.vel
        system.force
        system.potential_energy
        system.kinetic_energy
    """
    # ---- 1) Compute forces at current positions ----
    pe = system.compute_forces(
            lambda pos, box, pairs: lj_forces(pos, box, pairs, epsilon, sigma, rcut)
        )

    # ---- 2) Half-step velocity update and full-step position update ----    
    # v(t+dt/2) = v + (dt/2)*(F/m)
    # r(t+dt) = r + dt*v(t+dt/2)
    m = system.mass

    if np.isscalar(m):
        inv_m = 1.0 / m
        system.vel += 0.5 * dt * system.force * inv_m      
        system.pos += dt * system.vel
    else:
        # m is per-particle masses
        system.vel += 0.5 * dt * (system.force / m[:, None])
        system.pos += dt * system.vel

    # ---- 3) Apply periodic boundary conditions ----
    system.pos %= system.box

    # ---- 4) Recompute forces at new positions ----
    pe_new = system.compute_forces(
            lambda pos, box, pairs: lj_forces(pos, box, pairs, epsilon, sigma, rcut)
        )
    system.potential_energy = pe_new
    forces_new = system.force

    # ---- 5) Second half-step velocity update ----
    # v(t+dt) = v(t+dt/2) + (dt/2)*(F/m)
    if np.isscalar(m):
        inv_m = 1.0 / m
        system.vel += 0.5 * dt * system.force * inv_m
    else:
        system.vel += 0.5 * dt * (system.force / m[:, None])

    # ---- 6) Update kinetic energy ----
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
    """
    Perform one NVE (microcanonical) MD step.
    """
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
