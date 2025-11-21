# streamlit_app.py
# FCC Molecular Dynamics – Web Version (persistent results + 3D viewer + plot carousel)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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
# Browser-native 3D atom visualizer (Plotly)
# ------------------------------------------------------------
def visualize_atoms_3d(positions, box, symbol="X"):
    pos = np.asarray(positions)
    x, y, z = pos.T
    Lx, Ly, Lz = np.asarray(box, dtype=float)

    fig = go.Figure()

    # Atoms
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(size=4, opacity=0.9),
            name=symbol,
        )
    )

    # Simulation box wireframe
    corners = np.array([
        [0, 0, 0],
        [Lx, 0, 0],
        [Lx, Ly, 0],
        [0, Ly, 0],
        [0, 0, Lz],
        [Lx, 0, Lz],
        [Lx, Ly, Lz],
        [0, Ly, Lz],
    ])

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for i, j in edges:
        fig.add_trace(
            go.Scatter3d(
                x=[corners[i, 0], corners[j, 0]],
                y=[corners[i, 1], corners[j, 1]],
                z=[corners[i, 2], corners[j, 2]],
                mode="lines",
                line=dict(width=2),
                showlegend=False,
            )
        )

    fig.update_layout(
        title=f"{symbol} Atom Configuration",
        scene=dict(
            xaxis=dict(title="x (Å)", range=[0, Lx], showbackground=False),
            yaxis=dict(title="y (Å)", range=[0, Ly], showbackground=False),
            zaxis=dict(title="z (Å)", range=[0, Lz], showbackground=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
    )
    return fig


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

dt = st.sidebar.number_input("dt (fs)", min_value=1e-5, value=0.001, format="%.3f")

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
# Run simulation when button pressed (store results in session_state)
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

    # Initial forces / potential energy
    pe0 = system.compute_forces(
        lambda pos, box, pairs: lj_forces(
            pos, box, pairs, epsilon=epsilon, sigma=sigma, rcut=rcut
        )
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

    # Compute forces before starting production (in case equil steps = 0)
    pe = system.compute_forces(
        lambda pos, box, pairs: lj_forces(
            pos, box, pairs, epsilon=epsilon, sigma=sigma, rcut=rcut
        )
    )
    system.potential_energy = pe

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
            # Update forces / PE for correct pressure/energy if needed
            pe = system.compute_forces(
                lambda pos, box, pairs: lj_forces(
                    pos, box, pairs, epsilon=epsilon, sigma=sigma, rcut=rcut
                )
            )
            system.potential_energy = pe

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
    # Analysis (RDF, MSD, VACF, S(k), diffusion, Cv)
    # --------------------------------------------------------
    st.write("### Computing analysis (RDF, MSD, VACF, S(k), D, C)…")

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

    # Trajectory bytes for download
    try:
        with open(traj_filename, "rb") as f:
            xyz_bytes = f.read()
    except FileNotFoundError:
        xyz_bytes = None

    # --------------------------------------------------------
    # Store everything in session_state so it persists
    # --------------------------------------------------------
    st.session_state["results"] = {
        "metal": metal,
        "box": system.box.copy(),
        "positions_traj": positions_traj,
        "velocities_traj": velocities_traj,
        "pressure_traj": pressure_traj,
        "energy_traj": energy_traj,
        "temp_traj": temp_traj,
        "dt_sample": dt_sample,
        "r": r,
        "g_r": g_r,
        "r_cn": r_cn,
        "t_msd": t_msd,
        "msd": msd,
        "t_vacf": t_vacf,
        "vacf": vacf,
        "k_vals": k_vals,
        "S_k": S_k,
        "D_msd": D_msd,
        "D_vacf": D_vacf,
        "CN": CN,
        "Cv": Cv,
        "T_mean": T_mean,
        "P_mean": float(np.mean(pressure_traj)),
        "traj_bytes": xyz_bytes,
        "traj_filename": traj_filename,
    }

    st.success("Simulation and analysis complete!")


# --------------------------------------------------------
# Display results
# --------------------------------------------------------
if "results" in st.session_state:
    res = st.session_state["results"]

    # -------- Row 1: Summary (left) + Plot carousel (right) --------
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Summary")
        st.write(f"**FCC Metal:** {res['metal']}")
        st.write(f"**Mean Temperature:** {res['T_mean']:.3f} K")
        st.write(f"**Mean Pressure:** {res['P_mean']:.5e} eV/Å³")
        st.write(f"**Coordination Number (FCC ideal ~12):** {res['CN']:.3f}")
        st.write(f"**Heat Capacity (C):** {res['Cv']:.5e} eV/K")
        st.write(f"**Diffusion Coefficient (D) [MSD]:**  {res['D_msd']:.5e} Å²/fs")
        st.write(f"**Diffusion Coefficient (D) [VACF]:** {res['D_vacf']:.5e} Å²/fs")

    with col2:
        st.write("### Plots")
        plot_choice = st.selectbox(
            "Select plot",
            [
                "Radial Distribution Function g(r)",
                "Mean Squared Displacement",
                "Velocity Autocorrelation Function",
                "Static Structure Factor S(k)",
                "Temperature vs Time",
                "Pressure vs Time",
            ],
            index=0,
            key="plot_choice",
        )

        fig, ax = plt.subplots()

        if plot_choice == "Radial Distribution Function g(r)":
            ax.plot(res["r"], res["g_r"])
            ax.axvline(res["r_cn"], ls="--", alpha=0.6)
            ax.set_xlabel("r (Å)")
            ax.set_ylabel("g(r)")
            ax.set_title("Radial Distribution Function")

        elif plot_choice == "Mean Squared Displacement":
            ax.plot(res["t_msd"], res["msd"])
            ax.set_xlabel("t (fs)")
            ax.set_ylabel("MSD (Å²)")
            ax.set_title("Mean Squared Displacement")

        elif plot_choice == "Velocity Autocorrelation Function":
            vacf = res["vacf"]
            if vacf[0] != 0:
                vacf_norm = vacf / vacf[0]
            else:
                vacf_norm = vacf
            ax.plot(res["t_vacf"], vacf_norm)
            ax.set_xlabel("t (fs)")
            ax.set_ylabel("Normalized VACF")
            ax.set_title("Velocity Autocorrelation Function")

        elif plot_choice == "Static Structure Factor S(k)":
            ax.plot(res["k_vals"], res["S_k"])
            ax.set_xlabel("k (1/Å)")
            ax.set_ylabel("S(k)")
            ax.set_title("Static Structure Factor")

        elif plot_choice == "Temperature vs Time":
            t_temp = np.arange(len(res["temp_traj"])) * res["dt_sample"]
            ax.plot(t_temp, res["temp_traj"])
            ax.set_xlabel("t (fs)")
            ax.set_ylabel("T (K)")
            ax.set_title("Temperature vs Time")

        elif plot_choice == "Pressure vs Time":
            t_pres = np.arange(len(res["pressure_traj"])) * res["dt_sample"]
            ax.plot(t_pres, res["pressure_traj"])
            ax.set_xlabel("t (fs)")
            ax.set_ylabel("P (eV/Å³)")
            ax.set_title("Pressure vs Time")

        st.pyplot(fig)

    # -------- Row 2: Visualizer full width below --------
    st.write(f"### 3D {res['metal']} Interactive Visualizer")
    nframes = len(res["positions_traj"])
    dt_samp = res["dt_sample"]  # fs per saved frame
    t_max = (nframes - 1) * dt_samp

    t_fs = st.slider(
        "Time (fs)",
        min_value=0.0,
        max_value=float(t_max),
        value=0.0,
        step=float(dt_samp),
        key="time_slider",
    )
    frame = int(round(t_fs / dt_samp))

    fig3d = visualize_atoms_3d(
        res["positions_traj"][frame], res["box"], symbol=res["metal"]
    )
    st.plotly_chart(fig3d, width="stretch")

    # -------- Downloads at bottom --------
    st.write("### Downloadable Data Files")

    # Trajectory (.xyz) from OVITO-style writer
    if res["traj_bytes"] is not None:
        st.download_button(
            label="Download trajectory (.xyz)",
            data=res["traj_bytes"],
            file_name=res["traj_filename"],
            mime="chemical/x-xyz",
        )
    else:
        st.warning("No trajectory file found. Try rerunning the simulation.")

    # Helper: arrays -> CSV bytes
    def to_csv(*cols, headers):
        import io, csv
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(headers)
        for row in zip(*cols):
            writer.writerow(row)
        return buf.getvalue().encode()

    # RDF (r, g(r))
    rdf_bytes = to_csv(
        res["r"], res["g_r"],
        headers=["r (A)", "g(r)"]
    )
    st.download_button(
        label="Download RDF (CSV)",
        data=rdf_bytes,
        file_name="rdf.csv",
        mime="text/csv",
    )

    # MSD
    msd_bytes = to_csv(
        res["t_msd"], res["msd"],
        headers=["t (fs)", "MSD (A^2)"]
    )
    st.download_button(
        label="Download MSD (CSV)",
        data=msd_bytes,
        file_name="msd.csv",
        mime="text/csv",
    )

    # VACF
    vacf_bytes = to_csv(
        res["t_vacf"], res["vacf"],
        headers=["t (fs)", "VACF"]
    )
    st.download_button(
        label="Download VACF (CSV)",
        data=vacf_bytes,
        file_name="vacf.csv",
        mime="text/csv",
    )

    # Structure factor S(k)
    sk_bytes = to_csv(
        res["k_vals"], res["S_k"],
        headers=["k (1/A)", "S(k)"]
    )
    st.download_button(
        label="Download S(k) (CSV)",
        data=sk_bytes,
        file_name="structure_factor.csv",
        mime="text/csv",
    )

    # Thermodynamic time series: T(t), P(t), E(t)
    t_temp = np.arange(len(res["temp_traj"])) * res["dt_sample"]
    thermo_bytes = to_csv(
        t_temp,
        res["temp_traj"],
        res["pressure_traj"],
        res["energy_traj"],
        headers=["t (fs)", "T(K)", "P(eV/A^3)", "E_total(eV)"]
    )
    st.download_button(
        label="Download Thermodynamic Data (CSV)",
        data=thermo_bytes,
        file_name="thermo.csv",
        mime="text/csv",
    )

    # Final atomic positions (last frame)
    final_positions = res["positions_traj"][-1]
    final_bytes = to_csv(
        final_positions[:, 0],
        final_positions[:, 1],
        final_positions[:, 2],
        headers=["x (A)", "y (A)", "z (A)"]
    )
    st.download_button(
        label="Download Final Snapshot (CSV)",
        data=final_bytes,
        file_name="final_snapshot.csv",
        mime="text/csv",
    )

    # Full simulation dump as NPZ (everything in res)
    import io
    npz_buf = io.BytesIO()
    np.savez(npz_buf, **res)
    st.download_button(
        label="Download Full Simulation (.npz)",
        data=npz_buf.getvalue(),
        file_name="simulation_data.npz",
        mime="application/octet-stream",
    )

else:
    st.info("Set parameters in the sidebar and click **Run Simulation** to begin.")
