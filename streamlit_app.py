# streamlit_app.py
# FCC Molecular Dynamics – Web Version 

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from md.system import System
from md.constants import get_lattice_constant, get_amu, get_sigma, get_eps
from md.lattice import make_fcc_lattice
from md.forces import lj_forces
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

kB = 8.617333262145e-5  # eV/K


# ------------------------------------------------------------
# Velocity initialization 
# ------------------------------------------------------------
def initialize_velocities(system, T, seed=123):
    rng = np.random.default_rng(seed)
    N = system.N
    m = system.mass

    if np.isscalar(m):
        std = np.sqrt(kB * T / m)
    else:
        std = np.sqrt(kB * T / m)[:, None]

    v = rng.normal(0.0, 1.0, size=(N, 3)) * std
    v -= v.mean(axis=0)
    system.vel = v
    system.kinetic_energy()


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="FCC Molecular Dynamics", layout="wide")

st.title("FCC Molecular Dynamics - Web Version")

st.sidebar.header("Simulation Parameters")

metal = st.sidebar.selectbox(
    "Metal", ["Ag", "Al", "Au", "Cu", "Ni", "Pb", "Pd", "Pt"], index=3
)

nx = st.sidebar.number_input("nx", min_value=1, value=3, step=1)
ny = st.sidebar.number_input("ny", min_value=1, value=3, step=1)
nz = st.sidebar.number_input("nz", min_value=1, value=3, step=1)

T_target = st.sidebar.number_input("Temperature (K)", min_value=1.0, value=300.0)

ensemble = st.sidebar.selectbox("Ensemble", ["NVT (Berendsen)", "NVE"])

dt = st.sidebar.number_input("dt (fs)", min_value=1e-5, value=0.001, format="%.6f")

nsteps_equil = st.sidebar.number_input(
    "Equilibration steps", min_value=0, value=2000, step=100
)
nsteps = st.sidebar.number_input(
    "Production steps", min_value=1, value=2000, step=100
)
sample_interval = st.sidebar.number_input(
    "Sample interval", min_value=1, value=10, step=1
)
xyz_out_interval = st.sidebar.number_input(
    ".xyz output interval", min_value=1, value=100, step=1
)

run_btn = st.sidebar.button("Run Simulation")


# ------------------------------------------------------------
# Run simulation when button pressed
# ------------------------------------------------------------
if run_btn:
    st.write("### Running simulation…")

    # --- Build system ---
    a = get_lattice_constant(metal)
    mass = get_amu(metal) * 103.6427
    epsilon = get_eps(metal)
    sigma = get_sigma(metal)
    rcut = 2.5 * sigma

    pos, box = make_fcc_lattice(a, int(nx), int(ny), int(nz))
    system = System(pos, mass, box, symbol=metal, cutoff=rcut, skin=0.3)

    initialize_velocities(system, T_target)
    system.remove_drift()

    pe0 = system.compute_forces(
                lambda pos, box, pairs: lj_forces(pos, box, pairs,
                                                  epsilon=epsilon, sigma=sigma, rcut=rcut)
            )
    system.potential_energy = pe0

    # --- Equilibration (always NVT Berendsen) ---
    if nsteps_equil > 0:
        for _ in range(int(nsteps_equil)):
            step_nvt_berendsen(
                system, dt, T_target, 100 * dt, epsilon=epsilon, sigma=sigma, rcut=rcut
            )

    # --- Production trajectory storage ---
    nsteps = int(nsteps)
    sample_interval = int(sample_interval)
    xyz_out_interval = int(xyz_out_interval)

    n_samples = nsteps // sample_interval + 1
    N_atoms = system.N

    positions_traj = np.zeros((n_samples, N_atoms, 3))
    velocities_traj = np.zeros((n_samples, N_atoms, 3))
    pressure_traj = np.zeros(n_samples)
    energy_traj = np.zeros(n_samples)
    temp_traj = np.zeros(n_samples)

    sample_idx = 0

    # Compute initial forces & potential energy (important when equil steps = 0)
    system.compute_forces(
        lambda pos, box, pairs: lj_forces(pos, box, pairs, epsilon, sigma, rcut)
    )

    positions_traj[0] = system.pos.copy()
    velocities_traj[0] = system.vel.copy()
    pressure_traj[0] = compute_pressure(system)
    energy_traj[0] = system.kinetic_energy() + system.potential_energy
    temp_traj[0] = system.temperature()

    # write initial frame
    traj_filename = "traj_streamlit.xyz"
    write_xyz(system, step=0, filename=traj_filename)

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    # --- Production run ---
    for step in range(1, nsteps + 1):
        if ensemble.startswith("NVT"):
            step_nvt_berendsen(
                system, dt, T_target, 100 * dt, epsilon=epsilon, sigma=sigma, rcut=rcut
            )
        else:
            step_nve(system, dt, epsilon=epsilon, sigma=sigma, rcut=rcut)

        # Write .xyz periodically
        if step % xyz_out_interval == 0:
            write_xyz(system, step, filename=traj_filename)

        # Sample physical quantities
        if step % sample_interval == 0:
            sample_idx += 1
            positions_traj[sample_idx] = system.pos.copy()
            velocities_traj[sample_idx] = system.vel.copy()
            pressure_traj[sample_idx] = compute_pressure(system)
            energy_traj[sample_idx] = system.kinetic_energy() + system.potential_energy
            temp_traj[sample_idx] = system.temperature()

        if step % max(1, nsteps // 20) == 0:
            frac = step / nsteps
            progress_bar.progress(frac)
            status_text.text(f"Step {step}/{nsteps}")

    # Trim arrays
    positions_traj = positions_traj[: sample_idx + 1]
    velocities_traj = velocities_traj[: sample_idx + 1]
    pressure_traj = pressure_traj[: sample_idx + 1]
    energy_traj = energy_traj[: sample_idx + 1]
    temp_traj = temp_traj[: sample_idx + 1]

    dt_sample = dt * sample_interval

    # --------------------------------------------------------
    # Analysis (RDF, MSD, VACF, CN, Cv, diffusion, S(k))
    # --------------------------------------------------------
    st.write("### Computing analysis (RDF, MSD, VACF, S(k), diffusion, Cv)…")

    # RDF
    r_max = 0.45 * min(system.box)
    n_bins = 200
    r, g_r = compute_rdf(positions_traj, system.box, r_max, n_bins)

    # MSD
    t_msd, msd = compute_msd(positions_traj, system.box)
    t_msd *= dt_sample

    # VACF
    t_vacf, vacf = compute_vacf(velocities_traj)
    t_vacf *= dt_sample

    # Diffusion
    D_msd, _ = compute_diffusion_from_msd(t_msd, msd)
    D_vacf, _ = compute_diffusion_from_vacf(t_vacf, vacf)

    # Coordination number
    rho = system.N / system.volume()
    idx_peak = np.argmax(g_r)
    idx_min = idx_peak + np.argmin(g_r[idx_peak:])
    r_cn = r[idx_min]
    CN = compute_coordination_from_rdf(r, g_r, rho, r_cn)

    # Heat capacity
    E_tail = energy_traj[len(energy_traj) // 2 :]
    T_mean = temp_traj[len(temp_traj) // 2 :].mean()
    Cv = compute_heat_capacity_from_energy(E_tail, T_mean)

    # Structure factor
    k_vals = np.linspace(0.1, 12.0, 300)
    k_vals, S_k = compute_structure_factor(k_vals, r, g_r, rho)

    st.success("Simulation and analysis complete!")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    st.write("### Summary")
    st.write(f"Mean T: {T_mean:.3f} K")
    st.write(f"Mean P: {np.mean(pressure_traj):.5e} eV/Å³")
    st.write(f"CN (FCC ideal ~12): {CN:.3f}")
    st.write(f"Cv: {Cv:.5e} eV/K")
    st.write(f"D (MSD):  {D_msd:.5e} Å²/fs")
    st.write(f"D (VACF): {D_vacf:.5e} Å²/fs")

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    # RDF
    fig, ax = plt.subplots()
    ax.plot(r, g_r)
    ax.axvline(r_cn, ls="--", alpha=0.6)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("g(r)")
    ax.set_title("Radial Distribution Function")
    st.pyplot(fig)

    # MSD
    fig, ax = plt.subplots()
    ax.plot(t_msd, msd)
    ax.set_xlabel("t (fs)")
    ax.set_ylabel("MSD (Å²)")
    ax.set_title("Mean Squared Displacement")
    st.pyplot(fig)

    # VACF (normalized)
    fig, ax = plt.subplots()
    if vacf[0] != 0:
        vacf_norm = vacf / vacf[0]
    else:
        vacf_norm = vacf
    ax.plot(t_vacf, vacf_norm)
    ax.set_xlabel("t (fs)")
    ax.set_ylabel("Normalized VACF")
    ax.set_title("Velocity Autocorrelation Function")
    st.pyplot(fig)

    # Structure factor
    fig, ax = plt.subplots()
    ax.plot(k_vals, S_k)
    ax.set_xlabel("k (1/Å)")
    ax.set_ylabel("S(k)")
    ax.set_title("Static Structure Factor")
    st.pyplot(fig)

    # Temperature vs time (fs)
    t_temp = np.arange(len(temp_traj)) * dt_sample
    fig, ax = plt.subplots()
    ax.plot(t_temp, temp_traj)
    ax.set_xlabel("t (fs)")
    ax.set_ylabel("T (K)")
    ax.set_title("Temperature vs Time")
    st.pyplot(fig)

    # Pressure vs time (fs)
    t_pres = np.arange(len(pressure_traj)) * dt_sample
    fig, ax = plt.subplots()
    ax.plot(t_pres, pressure_traj)
    ax.set_xlabel("t (fs)")
    ax.set_ylabel("P (eV/Å³)")
    ax.set_title("Pressure vs Time")
    st.pyplot(fig)


    # --------------------------------------------------------
    # Trajectory download (.xyz)
    # --------------------------------------------------------
    try:
        with open(traj_filename, "rb") as f:
            xyz_bytes = f.read()

        st.download_button(
            label="Download trajectory (.xyz)",
            data=xyz_bytes,
            file_name=traj_filename,
            mime="chemical/x-xyz",
        )
    except FileNotFoundError:
        st.warning("No trajectory file found. Try rerunning the simulation.")