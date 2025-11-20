import numpy as np
import os
import sys
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from md.system import System
from md.constants import get_lattice_constant, get_amu, get_sigma, get_eps
from md.lattice import make_fcc_lattice
from md.integrator import step_nve, step_nvt_berendsen
from md.utils import write_xyz
from md.analysis import (
    compute_rdf,
    compute_msd,
    compute_vacf,
    compute_pressure,
    compute_heat_capacity_from_energy,
    compute_coordination_from_rdf,
    compute_structure_factor,
    compute_diffusion_from_msd,
    compute_diffusion_from_vacf,
)

# OVITO optional
try:
    from ovito.io import import_file
    from ovito.vis import Viewport
    HAVE_OVITO = True
except ImportError:
    HAVE_OVITO = False

# -----------------------------------------------------
# Simulation Inputs
# -----------------------------------------------------
metal = "Ni"
ensemble = "nve"
nx = ny = nz = 3
T_target = 1200.0

dt = 0.001
nsteps_equil = 1000
nsteps = 10000

sample_interval = 10
output_interval = 20
traj_file = "nve_Ni-1200K-10fs.xyz"

# -----------------------------------------------------
# Velocity initialization
# -----------------------------------------------------
kB = 8.617333262e-5  # eV/K

def initialize_velocities(system, T, seed=123):
    rng = np.random.default_rng(seed)
    N = system.N
    m = system.mass

    # Generate random velocities
    if np.isscalar(m):
        std = np.sqrt(kB * T / m)
    else:
        std = np.sqrt(kB * T / m)[:, None]

    v = rng.normal(0.0, 1.0, size=(N, 3)) * std
    v -= v.mean(axis=0)     # remove COM drift
    system.vel = v

    # Rescale exactly to target T
    system.kinetic_energy()
    T_inst = system.temperature()
    scale = np.sqrt(T / T_inst)
    system.vel *= scale
    system.kinetic_energy()

# -----------------------------------------------------
# Setup system
# -----------------------------------------------------
a = get_lattice_constant(metal)
mass = get_amu(metal) * 103.6427

epsilon = get_eps(metal)
sigma   = get_sigma(metal)
rcut    = 2.5 * sigma

pos, box = make_fcc_lattice(a, nx, ny, nz)
system = System(pos, mass, box, symbol=metal, cutoff=rcut, skin=0.3)

initialize_velocities(system, T_target)
system.remove_drift()

# -----------------------------------------------------
# Equilibration (always NVT Berendsen)
# -----------------------------------------------------
for step in range(nsteps_equil):
    step_nvt_berendsen(system, dt, T_target, 100 * dt,
                       epsilon=epsilon, sigma=sigma, rcut=rcut)

write_xyz(system, step=0, filename=traj_file)

# -----------------------------------------------------
# Production Trajectory
# -----------------------------------------------------
n_samples = nsteps // sample_interval + 1

positions_traj = np.zeros((n_samples, system.N, 3))
velocities_traj = np.zeros((n_samples, system.N, 3))
pressure_traj = np.zeros(n_samples)
energy_traj = np.zeros(n_samples)
temp_traj = np.zeros(n_samples)

sample_idx = 0

positions_traj[0] = system.pos.copy()
velocities_traj[0] = system.vel.copy()
pressure_traj[0] = compute_pressure(system)
energy_traj[0] = system.kinetic_energy() + system.potential_energy
temp_traj[0] = system.temperature()

print("# step        KE (eV)        PE (eV)      Etot (eV)      T (K)")

# -----------------------------------------------------
# Production Run
# -----------------------------------------------------
for step in range(1, nsteps + 1):

    step_nve(system, dt, epsilon=epsilon, sigma=sigma, rcut=rcut)

    if step % output_interval == 0:
        write_xyz(system, step, traj_file)

    if step % sample_interval == 0:
        sample_idx += 1

        positions_traj[sample_idx] = system.pos.copy()
        velocities_traj[sample_idx] = system.vel.copy()
        pressure_traj[sample_idx] = compute_pressure(system)

        KE = system.kinetic_energy()
        PE = system.potential_energy
        energy_traj[sample_idx] = KE + PE
        temp_traj[sample_idx] = system.temperature()

    if step % 200 == 0:
        KE = system.kinetic_energy()
        PE = system.potential_energy
        Etot = KE + PE
        Tcur = system.temperature()
        print(f"{step:6d}  {KE: .6e}  {PE: .6e}  {Etot: .6e}  {Tcur: .4f}")

print("Simulation complete.")



# Trim arrays
positions_traj  = positions_traj[:sample_idx+1]
velocities_traj = velocities_traj[:sample_idx+1]
pressure_traj   = pressure_traj[:sample_idx+1]
energy_traj     = energy_traj[:sample_idx+1]
temp_traj       = temp_traj[:sample_idx+1]

dt_sample = dt * sample_interval


# ================================================================
# ========================= ANALYSIS ==============================
# ================================================================

print("Computing RDF, MSD, VACF, CN, Cv, diffusion...")

# ---------- RDF ----------
r_max = 0.45 * min(box)
n_bins = 200
r, g_r = compute_rdf(positions_traj, system.box, r_max, n_bins)

# ---------- MSD ----------
t_msd, msd = compute_msd(positions_traj, system.box)
t_msd *= dt_sample

# ---------- VACF ----------
t_vacf, vacf = compute_vacf(velocities_traj)
t_vacf *= dt_sample

# ---------- Diffusion ----------
D_msd, slope = compute_diffusion_from_msd(t_msd, msd)
D_vacf, integral = compute_diffusion_from_vacf(t_vacf, vacf)

# ---------- Coordination number ----------
rho = system.N / system.volume()
idx_peak = np.argmax(g_r)
idx_min = idx_peak + np.argmin(g_r[idx_peak:])
r_cn = r[idx_min]
CN = compute_coordination_from_rdf(r, g_r, rho, r_cn)

# ---------- Heat capacity ----------
E_tail = energy_traj[len(energy_traj)//2:]
T_mean = temp_traj[len(temp_traj)//2:].mean()
Cv = compute_heat_capacity_from_energy(E_tail, T_mean)

# ---------- Structure factor ----------
k_vals = np.linspace(0.1, 12.0, 300)
k_vals, S_k = compute_structure_factor(k_vals, r, g_r, rho)


# ================================================================
# ======================== PRINT SUMMARY ==========================
# ================================================================
print("\n=== Summary ===")
print(f"Mean T: {T_mean:.3f} K")
print(f"Mean P: {np.mean(pressure_traj):.5e} eV/Å³")
print(f"CN (FCC ideal ~12): {CN:.3f}")
print(f"Cv: {Cv:.5e} eV/K")
print(f"D (MSD):  {D_msd:.5e}")
print(f"D (VACF): {D_vacf:.5e}")
print("=================\n")


# ================================================================
# ============================ PLOTS ==============================
# ================================================================
plt.figure(figsize=(7,5))
plt.plot(r, g_r)
plt.xlabel("r (Å)")
plt.ylabel("g(r)")
plt.title("RDF")
plt.axvline(r_cn, ls="--", alpha=0.6)
plt.tight_layout()

plt.figure(figsize=(7,5))
plt.plot(t_msd, msd)
plt.xlabel("t (fs)")
plt.ylabel("MSD (Å²)")
plt.title("MSD")
plt.tight_layout()

plt.figure(figsize=(7,5))
plt.plot(t_vacf, vacf)
plt.xlabel("t (fs)")
plt.ylabel("VACF")
plt.title("VACF")
plt.tight_layout()

plt.figure(figsize=(7,5))
plt.plot(k_vals, S_k)
plt.xlabel("k (1/Å)")
plt.ylabel("S(k)")
plt.title("Structure Factor")
plt.tight_layout()

plt.figure(figsize=(7,5))
plt.plot(temp_traj)
plt.xlabel("sample index")
plt.ylabel("T (K)")
plt.title("Temperature")
plt.tight_layout()

plt.figure(figsize=(7,5))
plt.plot(pressure_traj)
plt.xlabel("sample index")
plt.ylabel("P (eV/Å³)")
plt.title("Pressure")
plt.tight_layout()

plt.show()
