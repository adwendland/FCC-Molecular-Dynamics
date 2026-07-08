# streamlit_app.py
# FCC Molecular Dynamics – Web Version

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from md.analysis.analysis_driver import (  # noqa: E402
    AVAILABLE_ANALYSES,
    print_analysis_report,
    run_analysis_suite,
)
from md.constants import get_eps, get_lattice_constant, get_sigma  # noqa: E402
from md.performance.performance_driver import (  # noqa: E402
    print_performance_report,
    run_performance_suite,
)
from md.validation.validation_driver import (  # noqa: E402
    AVAILABLE_TESTS,
    print_validation_report,
    run_validation_suite,
)

METALS = ["Ag", "Al", "Au", "Cu", "Ni", "Pb", "Pd", "Pt"]
SIZE_CHOICES = list(range(1, 10))
DT_CHOICES = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
THERMAL_DISPLACEMENT_CHOICES = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1]

DEFAULT_ANALYSES = [
    "thermo",
    "rdf",
    "coordination_number",
    "msd",
    "vacf",
]
DEFAULT_TESTS = [
    "energy_drift",
    "momentum",
    "timestep_refinement",
    "rdf_peaks",
    "coordination_number",
]

ANALYSIS_PLOTS = [
    "Temperature",
    "Energy",
    "Pressure",
    "RDF",
    "Structure factor",
    "MSD",
    "VACF",
]
VALIDATION_PLOTS = [
    "Energy conservation",
    "Momentum conservation",
    "Timestep convergence",
    "Temperature stability",
    "Component equipartition",
    "RDF validation",
]
PERFORMANCE_PLOTS = ["Performance scaling", "Kernel timing"]


# -----------------------------------------------------------------------------
# Page style
# -----------------------------------------------------------------------------
st.set_page_config(page_title="FCC MD Workbench", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    section[data-testid="stSidebar"] {min-width: 360px !important; width: 360px !important;}
    div[data-testid="stMetric"] {background: #f7f7f9; border: 1px solid #e5e7eb; border-radius: 0.5rem; padding: 0.65rem;}
    .small-muted {font-size: 0.85rem; color: #666;}
    .mono-report {font-family: Consolas, Menlo, Monaco, monospace; font-size: 0.86rem; white-space: pre-wrap;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def capture_report(fn, results: dict[str, Any]) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        fn(results)
    return buffer.getvalue()


def fmt(x: Any, precision: int = 4) -> str:
    try:
        x = float(x)
    except Exception:
        return "—"
    if not np.isfinite(x):
        return "—"
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e4:
        return f"{x:.{precision}e}"
    return f"{x:.{precision}f}"


def parse_sizes(text: str) -> tuple[tuple[int, int, int], ...]:
    sizes: list[tuple[int, int, int]] = []
    for chunk in text.replace(";", ",").split(","):
        chunk = chunk.strip().lower()
        if not chunk:
            continue
        parts = chunk.split("x")
        if len(parts) == 1:
            n = int(parts[0])
            sizes.append((n, n, n))
        elif len(parts) == 3:
            sizes.append(tuple(int(p) for p in parts))
        else:
            raise ValueError(f"Could not parse size '{chunk}'. Use e.g. 3x3x3,4x4x4,5.")
    if not sizes:
        raise ValueError("Enter at least one benchmark size.")
    return tuple(sizes)


def lammps_input_deck(
    *,
    metal: str,
    nx: int,
    ny: int,
    nz: int,
    ensemble: str,
    T0: float,
    dt: float,
    equil_steps: int,
    run_steps: int,
    sample_every: int,
    thermal_displacement: float,
) -> str:
    a = get_lattice_constant(metal)
    sigma = get_sigma(metal)
    eps = get_eps(metal)
    rcut = 2.5 * sigma
    atoms = 4 * nx * ny * nz
    tau = 100 * dt

    production_fix = "fix             prod all nve" if ensemble == "nve" else f"fix             prod all nvt temp {T0:g} {T0:g} {tau:g}"
    line = "#" + "=" * 76
    sub = "#" + "-" * 76
    return f"""{line}
# FCC MOLECULAR DYNAMICS INPUT SUMMARY
{line}

# ---- system ---------------------------------------------------------------
units           metal                 # app uses eV, Angstrom, fs internally
atom_style      atomic
boundary        p p p

lattice         fcc {a:.6f}            # metal = {metal}
region          simbox block 0 {nx} 0 {ny} 0 {nz}
create_box      1 simbox
create_atoms    1 simbox              # atoms = {atoms}
mass            1 <internal mass>      # app uses amu -> internal MD mass

{sub}
# Lennard-Jones model
{sub}
pair_style      lj/cut {rcut:.6f}
pair_coeff      1 1 {eps:.8f} {sigma:.8f}   # epsilon[eV], sigma[Angstrom]
neighbor        0.3 bin
neigh_modify    every 1 delay 0 check yes

{sub}
# Initialization
{sub}
timestep        {dt:g}                 # fs in this app
velocity        all create {T0:g} 123 mom yes dist gaussian
# random_displace amplitude = {thermal_displacement:g} lattice constants

{sub}
# Equilibration
{sub}
fix             eq all nvt temp {T0:g} {T0:g} {tau:g}
thermo          {sample_every}
run             {equil_steps}
unfix           eq

{sub}
# Production
{sub}
{production_fix}
thermo          {sample_every}
run             {run_steps}
unfix           prod

{line}
# End input summary
{line}
"""


def array_to_dat_bytes(*cols: np.ndarray, headers: list[str]) -> bytes:
    buf = io.StringIO()
    buf.write("# " + " ".join(headers) + "\n")
    for row in zip(*cols):
        buf.write(" ".join(f"{float(v):.10e}" for v in row) + "\n")
    return buf.getvalue().encode("utf-8")


def make_xyz_bytes(positions_traj: np.ndarray, symbol: str, box: np.ndarray, steps=None, times=None) -> bytes:
    out = io.StringIO()
    positions_traj = np.asarray(positions_traj)
    if steps is None:
        steps = np.arange(len(positions_traj))
    if times is None:
        times = [None] * len(positions_traj)
    Lx, Ly, Lz = np.asarray(box, dtype=float)
    lattice = f'Lattice="{Lx:.8f} 0 0  0 {Ly:.8f} 0  0 0 {Lz:.8f}" Properties=species:S:1:pos:R:3'
    for i, frame in enumerate(positions_traj):
        out.write(f"{frame.shape[0]}\n")
        time_part = "" if times[i] is None else f" Time_fs={float(times[i]):.8f}"
        out.write(f"Frame={i} Step={int(steps[i])}{time_part} {lattice}\n")
        for x, y, z in frame:
            out.write(f"{symbol} {x:.8f} {y:.8f} {z:.8f}\n")
    return out.getvalue().encode("utf-8")


def plot_atoms_3d(positions: np.ndarray, box: np.ndarray, symbol: str, marker_size: int = 7) -> go.Figure:
    pos = np.asarray(positions)
    Lx, Ly, Lz = np.asarray(box, dtype=float)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            mode="markers",
            marker=dict(size=marker_size, opacity=0.9),
            name=symbol,
            hovertemplate=f"{symbol}<br>x=%{{x:.3f}} Å<br>y=%{{y:.3f}} Å<br>z=%{{z:.3f}} Å<extra></extra>",
        )
    )
    corners = np.array([
        [0, 0, 0], [Lx, 0, 0], [Lx, Ly, 0], [0, Ly, 0],
        [0, 0, Lz], [Lx, 0, Lz], [Lx, Ly, Lz], [0, Ly, Lz],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        fig.add_trace(go.Scatter3d(
            x=[corners[i,0], corners[j,0]], y=[corners[i,1], corners[j,1]], z=[corners[i,2], corners[j,2]],
            mode="lines", line=dict(width=2), showlegend=False, hoverinfo="skip"
        ))
    fig.update_layout(
        title=f"{symbol} atomic configuration",
        scene=dict(
            xaxis=dict(title="x (Å)", range=[0, Lx], showbackground=False),
            yaxis=dict(title="y (Å)", range=[0, Ly], showbackground=False),
            zaxis=dict(title="z (Å)", range=[0, Lz], showbackground=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=35, b=0),
        showlegend=False,
        height=650,
        scene_camera=dict(eye=dict(x=1.8, y=1.6, z=1.2)),
    )
    return fig



def _mpl_figure(title: str, xlabel: str = "", ylabel: str = ""):
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=120)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.30)
    return fig, ax


def _no_data_figure(message: str):
    fig, ax = _mpl_figure("No data")
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def _as_array(x):
    if x is None:
        return np.array([])
    if isinstance(x, dict):
        return np.array(list(x.values()), dtype=float)
    return np.asarray(x)


def plot_analysis(results: dict[str, Any], choice: str):
    traj = results["trajectory"]

    if choice == "Temperature":
        fig, ax = _mpl_figure("Temperature vs Time", "Time (fs)", "Temperature (K)")
        ax.plot(traj["time_traj"], traj["temp_traj"], linewidth=1.8)

    elif choice == "Energy":
        fig, ax = _mpl_figure("Energy vs Time", "Time (fs)", "Energy (eV)")
        ax.plot(traj["time_traj"], traj["kinetic_traj"], label="Kinetic", linewidth=1.6)
        ax.plot(traj["time_traj"], traj["potential_traj"], label="Potential", linewidth=1.6)
        ax.plot(traj["time_traj"], traj["energy_traj"], label="Total", linewidth=1.8)
        ax.legend(frameon=False)

    elif choice == "Pressure":
        fig, ax = _mpl_figure("Pressure vs Time", "Time (fs)", "Pressure (eV/Å³)")
        ax.plot(traj["time_traj"], traj["pressure_traj"], linewidth=1.8)

    elif choice == "RDF" and "rdf" in results:
        rdf = results["rdf"]
        fig, ax = _mpl_figure("Radial Distribution Function", "r (Å)", "g(r)")
        ax.plot(rdf["r"], rdf["g_r"], linewidth=1.8)
        if "coordination_number" in results:
            r_cn = results["coordination_number"].get("r_cut_cn")
            if r_cn is not None:
                ax.axvline(r_cn, linestyle="--", linewidth=1.2, label="CN cutoff")
                ax.legend(frameon=False)

    elif choice == "Structure factor" and "structure_factor" in results:
        sk = results["structure_factor"]
        fig, ax = _mpl_figure("Structure Factor", "k (1/Å)", "S(k)")
        ax.plot(sk["k"], sk["S_k"], linewidth=1.8)

    elif choice == "MSD" and "msd" in results:
        msd = results["msd"]
        fig, ax = _mpl_figure("Mean Squared Displacement", "Time (fs)", "MSD (Å²)")
        ax.plot(msd["time"], msd["msd"], linewidth=1.8)

    elif choice == "VACF" and "vacf" in results:
        vacf = np.asarray(results["vacf"]["vacf"])
        y = vacf / vacf[0] if len(vacf) and vacf[0] != 0 else vacf
        fig, ax = _mpl_figure("Velocity Autocorrelation Function", "Time (fs)", "Normalized VACF")
        ax.plot(results["vacf"]["time"], y, linewidth=1.8)
        ax.axhline(0.0, linewidth=1.0, alpha=0.7)

    else:
        return _no_data_figure(f"No data available for {choice}.")

    fig.tight_layout()
    return fig


def plot_validation(results: dict[str, Any], choice: str):
    if choice == "Energy conservation" and "energy_drift" in results:
        d = results["energy_drift"]
        y = d.get("rel_drift", d.get("relative_drift", d.get("drift", [])))
        x = d.get("times", d.get("time", d.get("steps", np.arange(len(_as_array(y))))))
        fig, ax = _mpl_figure("Energy Conservation", "Time / step", "Relative energy drift")
        ax.plot(x, y, linewidth=1.8)
        ax.axhline(0.0, linewidth=1.0, alpha=0.7)

    elif choice == "Momentum conservation" and "momentum" in results:
        d = results["momentum"]
        y = d.get("normalized_momentum_drift", d.get("normalized_drift", d.get("momentum_norm", [])))
        fig, ax = _mpl_figure("Momentum Conservation", "Sample", "Normalized drift")
        ax.plot(y, linewidth=1.8)
        ax.axhline(0.0, linewidth=1.0, alpha=0.7)

    elif choice == "Timestep convergence" and "timestep_refinement" in results:
        d = results["timestep_refinement"]
        errors = d.get("errors", {})
        if isinstance(errors, dict):
            items = sorted((float(k), float(v)) for k, v in errors.items())
            dt_vals = np.array([k for k, _ in items])
            err_vals = np.array([v for _, v in items])
        else:
            dt_vals = np.asarray(d.get("dt_values", []), dtype=float)
            err_vals = np.asarray(errors, dtype=float)

        fig, ax = _mpl_figure("Timestep Convergence", "dt (fs)", "RMS position error (Å)")
        if len(dt_vals) and len(err_vals):
            order = d.get("order")
            label = "Position error" if order is None else f"Position error, order ≈ {float(order):.2f}"
            ax.loglog(dt_vals, err_vals, marker="o", linewidth=1.8, label=label)
            ax.set_xticks(dt_vals)
            ax.set_xticklabels([f"{v:g}" for v in dt_vals])
            ax.invert_xaxis()
            ax.legend(frameon=False)
        else:
            ax.text(0.5, 0.5, "No convergence data", ha="center", va="center", transform=ax.transAxes)

    elif choice == "Temperature stability" and "temperature_stability" in results:
        d = results["temperature_stability"]
        y = d.get("temperatures", d.get("temperature", []))
        x = d.get("times", np.arange(len(_as_array(y))))
        fig, ax = _mpl_figure("Temperature Stability", "Time / sample", "Temperature (K)")
        ax.plot(x, y, linewidth=1.8, label="Measured T")
        target = results.get("metadata", {}).get("T0")
        if target is not None:
            ax.axhline(float(target), linestyle="--", linewidth=1.2, label="Target T")
            ax.legend(frameon=False)

    elif choice == "Component equipartition" and ("component_equipartition" in results or "equipartition" in results):
        d = results.get("component_equipartition", results.get("equipartition", {}))
        fig, ax = _mpl_figure(
            "Component Equipartition",
            "Time (fs)",
            "Component kinetic energy (eV)",
        )

        # The validation driver returns time series named kinetic_x/y/z.
        # This matches the Tkinter GUI plot.
        if all(k in d for k in ("times", "kinetic_x", "kinetic_y", "kinetic_z")):
            t = np.asarray(d["times"], dtype=float)
            kx = np.asarray(d["kinetic_x"], dtype=float)
            ky = np.asarray(d["kinetic_y"], dtype=float)
            kz = np.asarray(d["kinetic_z"], dtype=float)

            ax.plot(t, kx, linewidth=1.5, label="Kx")
            ax.plot(t, ky, linewidth=1.5, label="Ky")
            ax.plot(t, kz, linewidth=1.5, label="Kz")

            expected = d.get("expected_component_energy")
            if expected is not None:
                ax.axhline(float(expected), linestyle="--", linewidth=1.2, label="Expected")

            ax.legend(frameon=False)

        # Fallback for older result dictionaries that only store means.
        else:
            vals = [d.get("mean_kinetic_x"), d.get("mean_kinetic_y"), d.get("mean_kinetic_z")]
            if any(v is None for v in vals):
                vals = [d.get("Kx"), d.get("Ky"), d.get("Kz")]
            vals = [np.nan if v is None else float(v) for v in vals]
            ax.set_xlabel("Component")
            ax.set_ylabel("Mean kinetic energy (eV)")
            ax.bar(["Kx", "Ky", "Kz"], vals)
            expected = d.get("expected_component_energy")
            if expected is not None:
                ax.axhline(float(expected), linestyle="--", linewidth=1.2, label="Expected")
                ax.legend(frameon=False)
            elif np.all(np.isfinite(vals)):
                ax.axhline(float(np.nanmean(vals)), linestyle="--", linewidth=1.2, label="Mean")
                ax.legend(frameon=False)

    elif choice == "RDF validation" and "rdf_peaks" in results:
        d = results["rdf_peaks"]
        r = d.get("r", d.get("r_values", []))
        g = d.get("g_r", d.get("rdf", []))
        fig, ax = _mpl_figure("RDF Validation", "r (Å)", "g(r)")
        ax.plot(r, g, linewidth=1.8, label="g(r)")
        measured = d.get("measured_peaks", [])
        expected = d.get("expected_peaks", [])
        if len(measured):
            ax.axvline(float(measured[0]), linestyle="--", linewidth=1.2, label="Measured first peak")
        if len(expected):
            ax.axvline(float(expected[0]), linestyle=":", linewidth=1.4, label="Expected first peak")
        ax.legend(frameon=False)

    else:
        return _no_data_figure(f"No data available for {choice}.")

    fig.tight_layout()
    return fig


def plot_performance(results: dict[str, Any], choice: str):
    cases = results.get("cases", [])
    labels = [f"{c['metadata']['nx']}x{c['metadata']['ny']}x{c['metadata']['nz']}" for c in cases]

    if choice == "Performance scaling":
        y = [c["integrator_nve"]["atom_steps_per_second"] for c in cases]
        fig, ax = _mpl_figure("Performance Scaling", "System size", "Atom-steps / s")
        ax.plot(labels, y, marker="o", linewidth=1.8)
        ax.tick_params(axis="x", rotation=25)

    elif choice == "Kernel timing":
        x = np.arange(len(labels))
        width = 0.35
        neighbor = [1000 * c["neighbor_build"]["mean_seconds"] for c in cases]
        force = [1000 * c["force_evaluation"]["mean_seconds"] for c in cases]
        fig, ax = _mpl_figure("Kernel Timing", "System size", "Time (ms)")
        ax.bar(x - width / 2, neighbor, width, label="Neighbor list")
        ax.bar(x + width / 2, force, width, label="Force")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25)
        ax.legend(frameon=False)

    else:
        return _no_data_figure(f"No data available for {choice}.")

    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Sidebar: Tkinter-style control panel
# -----------------------------------------------------------------------------
st.title("FCC Molecular Dynamics Workbench")
st.caption(
    "Web version"
)

with st.sidebar:
    st.header("Control panel")
    mode = st.radio("Task", ["Simulation / Analysis", "Validation", "Performance"], horizontal=False)

    # Safe defaults so downstream code never sees undefined names.
    ensemble_label = "NVE"
    thermal_displacement = 0.0
    save_trajectory = False
    save_plots = False
    sample_every = 10
    n_equil_steps = 0
    n_steps = 0
    nx = ny = nz = 3

    st.divider()

    if mode == "Simulation / Analysis":
        st.subheader("System")
        c1, c2 = st.columns(2)
        with c1:
            metal = st.selectbox("Metal", METALS, index=METALS.index("Ni"))
            nx = st.selectbox("nx", SIZE_CHOICES, index=2)
            ny = st.selectbox("ny", SIZE_CHOICES, index=2)
            nz = st.selectbox("nz", SIZE_CHOICES, index=2)
        with c2:
            ensemble_label = st.selectbox("Ensemble", ["NVE", "NVT"], index=0)
            T0 = st.number_input("T0 (K)", min_value=1.0, value=300.0, step=100.0)
            dt = st.selectbox("dt (fs)", DT_CHOICES, index=DT_CHOICES.index(0.1))
            thermal_displacement = st.selectbox("thermal disp.", THERMAL_DISPLACEMENT_CHOICES, index=2)

        st.caption(
            f"Ready | {metal} | {nx}x{ny}x{nz} | {4*nx*ny*nz} atoms | "
            f"{ensemble_label} | dt={dt:g} fs"
        )
        st.divider()
        st.subheader("Simulation / Analysis")
        n_equil_steps = st.number_input("Equilibration steps", min_value=0, value=2000, step=100)
        n_steps = st.number_input("Production steps", min_value=1, value=5000, step=500)
        
        if n_steps > 9999:
            st.warning("Large run: this may take several minutes in Streamlit.")

        sample_every = st.number_input("Sample every", min_value=1, value=10, step=1)
        analyses = st.multiselect(
            "Analyses",
            sorted(AVAILABLE_ANALYSES),
            default=[a for a in DEFAULT_ANALYSES if a in AVAILABLE_ANALYSES],
        )
        save_outputs = st.checkbox("Save .dat / report outputs", value=False)
        save_trajectory = st.checkbox("Save trajectory .npz and .xyz", value=True)
        save_plots = st.checkbox("Save plots", value=False)
        run_clicked = st.button("Run analysis", type="primary", width="stretch")

    elif mode == "Validation":
        st.subheader("Validation system")
        c1, c2 = st.columns(2)
        with c1:
            metal = st.selectbox("Metal", METALS, index=METALS.index("Ni"))
            nx = st.selectbox("nx", SIZE_CHOICES, index=3)
            ny = st.selectbox("ny", SIZE_CHOICES, index=3)
            nz = st.selectbox("nz", SIZE_CHOICES, index=3)
        with c2:
            T0 = st.number_input("T0 (K)", min_value=1.0, value=300.0, step=100.0)
            dt = st.selectbox("Production dt (fs)", DT_CHOICES, index=DT_CHOICES.index(0.1))
            refinement_dt = st.selectbox("Refinement dt (fs)", DT_CHOICES, index=DT_CHOICES.index(0.04))

        st.caption(
            f"Ready | {metal} | {nx}x{ny}x{nz} | {4*nx*ny*nz} atoms | "
            f"validation suite | dt={dt:g} fs"
        )
        st.divider()
        st.subheader("Validation")
        n_equil_steps = st.number_input("Equilibration steps", min_value=0, value=2000, step=500)
        n_steps = st.number_input("Validation steps", min_value=1, value=5000, step=500)

        if n_steps > 9999:
            st.warning("Large run: this may take several minutes in Streamlit.")

        sample_every = st.number_input("Sample every", min_value=1, value=10, step=1)
        refinement_steps = st.number_input("Refinement steps", min_value=1, value=500, step=100)
        tests = st.multiselect(
            "Tests",
            sorted(AVAILABLE_TESTS),
            default=AVAILABLE_TESTS,
        )
        st.caption("Validation uses fixed ensembles internally: NVE for conservation/convergence, NVT where thermostatted sampling is appropriate.")
        save_outputs = st.checkbox("Save validation outputs", value=False)
        save_plots = st.checkbox("Save validation plots", value=False)
        run_clicked = st.button("Run validation", type="primary", width="stretch")

    else:
        st.subheader("Benchmark system")
        c1, c2 = st.columns(2)
        with c1:
            metal = st.selectbox("Metal", METALS, index=METALS.index("Ni"))
            T0 = st.number_input("T0 (K)", min_value=1.0, value=300.0, step=100.0)
        with c2:
            dt = st.selectbox("dt (fs)", DT_CHOICES, index=DT_CHOICES.index(0.1))
            backend = st.selectbox("Backend", ["auto", "python", "serial-baseline"], index=0)

        st.caption(f"Ready | {metal} | performance benchmark | NVE timing kernel | dt={dt:g} fs")
        st.divider()
        st.subheader("Performance")
        sizes_text = st.text_input("Benchmark sizes", value="3x3x3,4x4x4,5x5x5")
        n_steps = st.number_input("Integrator steps / repeat", min_value=1, value=100, step=50)

        if n_steps > 9999:
            st.warning("Large run: this may take several minutes in Streamlit.")

        repeats = st.number_input("Repeats", min_value=1, value=5, step=1)
        warmup = st.number_input("Warmup repeats", min_value=0, value=2, step=1)
        save_outputs = st.checkbox("Save performance outputs", value=False)
        run_clicked = st.button("Run performance", type="primary", width="stretch")

    # st.divider()
    # st.subheader("Report preview")
    # preview = st.session_state.get("last_report", "Ready. Configure a run, validation suite, or benchmark, then click Run.")
    # st.text_area("", value=preview, height=260, label_visibility="collapsed")


# -----------------------------------------------------------------------------
# Execute selected workflow
# -----------------------------------------------------------------------------
if run_clicked:
    try:
        if mode == "Simulation / Analysis":
            with st.spinner("Running analysis..."):
                results = run_analysis_suite(
                    metal=metal,
                    nx=int(nx), ny=int(ny), nz=int(nz),
                    T0=float(T0),
                    ensemble=ensemble_label.lower(),
                    dt=float(dt),
                    n_equil_steps=int(n_equil_steps),
                    n_steps=int(n_steps),
                    sample_every=int(sample_every),
                    analyses=analyses,
                    seed=123,
                    thermal_displacement=float(thermal_displacement),
                    save_outputs=bool(save_outputs),
                    save_trajectory=bool(save_trajectory),
                    save_plots=bool(save_plots),
                    show_plots=False,
                )
            report = capture_report(print_analysis_report, results)
            st.session_state["last_kind"] = "analysis"
            st.session_state["last_results"] = results
            st.session_state["last_report"] = report
            st.success("Analysis complete.")

        elif mode == "Validation":
            with st.spinner("Running validation..."):
                results = run_validation_suite(
                    metal=metal,
                    nx=int(nx), ny=int(ny), nz=int(nz),
                    T0=float(T0),
                    dt=float(dt),
                    n_steps=int(n_steps),
                    n_equil_steps=int(n_equil_steps),
                    sample_every=int(sample_every),
                    refinement_dt=float(refinement_dt),
                    refinement_steps=int(refinement_steps),
                    tests=tests,
                    save_outputs=bool(save_outputs),
                    save_plots=bool(save_plots),
                    show_plots=False,
                )
            report = capture_report(print_validation_report, results)
            st.session_state["last_kind"] = "validation"
            st.session_state["last_results"] = results
            st.session_state["last_report"] = report
            st.success("Validation complete.")

        else:
            with st.spinner("Running performance suite..."):
                sizes = parse_sizes(sizes_text)
                results = run_performance_suite(
                    metal=metal,
                    sizes=sizes,
                    T0=float(T0),
                    dt=float(dt),
                    n_steps=int(n_steps),
                    repeats=int(repeats),
                    warmup=int(warmup),
                    seed=123,
                    thermal_displacement=0.0,
                    backend=backend,
                    save_outputs=bool(save_outputs),
                )
            report = capture_report(print_performance_report, results)
            st.session_state["last_kind"] = "performance"
            st.session_state["last_results"] = results
            st.session_state["last_report"] = report
            st.success("Performance suite complete.")
    except Exception as exc:
        st.error(f"Run failed: {exc}")
        st.exception(exc)


# -----------------------------------------------------------------------------
# Right-side notebook: task-aware report, plots, tables, deck, files
# -----------------------------------------------------------------------------
kind = st.session_state.get("last_kind")
results = st.session_state.get("last_results")
report = st.session_state.get("last_report", "Ready. Configure a run, validation suite, or benchmark, then click Run.")

if mode == "Simulation / Analysis":
    tab_report, tab_plot, tab_atoms, tab_tables, tab_deck, tab_files = st.tabs([
        "Report", "Plot", "Atom visualization", "Status / tables", "Input deck", "Files"
    ])

    with tab_report:
        st.code(report, language="text")

    with tab_plot:
        if results is None or kind != "analysis":
            st.info("Run an analysis job first.")
        else:
            choice = st.selectbox("Plot", ANALYSIS_PLOTS, index=0)
            st.pyplot(plot_analysis(results, choice), clear_figure=True)

    with tab_atoms:
        if kind != "analysis" or results is None or "trajectory" not in results:
            st.info("Atom visualization is available after a Simulation / Analysis job.")
        else:
            meta = results["metadata"]
            traj = results["trajectory"]
            positions_traj = traj["positions_traj"]
            frame_count = len(positions_traj)
            col_a, col_b, col_c = st.columns([2, 1, 1])
            with col_a:
                frame_idx = st.slider("Frame", 0, frame_count - 1, frame_count - 1)
            with col_b:
                marker_size = st.slider("Atom size", 2, 10, 7)
            with col_c:
                st.metric("Time (fs)", fmt(traj["time_traj"][frame_idx]))
            box = np.array([
                meta["nx"] * meta["lattice_constant"],
                meta["ny"] * meta["lattice_constant"],
                meta["nz"] * meta["lattice_constant"],
            ], dtype=float)
            st.plotly_chart(plot_atoms_3d(positions_traj[frame_idx], box, meta["metal"], marker_size), width="stretch")

    with tab_tables:
        if results is None or kind != "analysis":
            st.info("No analysis results yet.")
        else:
            meta = results["metadata"]
            summary = results.get("summary", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Atoms", int(meta["N"]))
            c2.metric("Mean T (K)", fmt(summary.get("mean_temperature")))
            c3.metric("CN", fmt(summary.get("coordination_number")))
            c4.metric("D MSD", fmt(summary.get("D_msd")))
            rows = [{"quantity": key, "value": fmt(value, 6)} for key, value in summary.items()]
            st.dataframe(rows, width="stretch", hide_index=True)
            if "trajectory" in results:
                traj = results["trajectory"]
                thermo_rows = np.column_stack((traj["time_traj"], traj["temp_traj"], traj["pressure_traj"], traj["energy_traj"]))
                st.write("Thermodynamic samples")
                st.dataframe(
                    [{"time_fs": r[0], "T_K": r[1], "P_eV_A3": r[2], "E_eV": r[3]} for r in thermo_rows[:500]],
                    width="stretch",
                    hide_index=True,
                )

    with tab_deck:
        st.code(
            lammps_input_deck(
                metal=metal,
                nx=int(nx), ny=int(ny), nz=int(nz),
                ensemble=ensemble_label.lower(),
                T0=float(T0),
                dt=float(dt),
                equil_steps=int(n_equil_steps),
                run_steps=int(n_steps),
                sample_every=int(sample_every),
                thermal_displacement=float(thermal_displacement),
            ),
            language="text",
        )

    with tab_files:
        if results is None or kind != "analysis":
            st.info("Run an analysis job first to enable downloads.")
        else:
            st.write("Saved files")
            saved = results.get("saved_files", {})
            if saved:
                for name, path in saved.items():
                    st.write(f"**{name}**: `{path}`")
            else:
                st.caption("No files were saved to disk for this run.")

            st.divider()
            meta = results["metadata"]
            traj = results["trajectory"]
            box = np.array([meta["nx"] * meta["lattice_constant"], meta["ny"] * meta["lattice_constant"], meta["nz"] * meta["lattice_constant"]])
            st.download_button("Download report (.txt)", data=report.encode("utf-8"), file_name="analysis_report.txt", mime="text/plain")
            st.download_button(
                "Download trajectory (.xyz)",
                data=make_xyz_bytes(traj["positions_traj"], meta["metal"], box, traj.get("step_traj"), traj.get("time_traj")),
                file_name="trajectory.xyz",
                mime="chemical/x-xyz",
            )
            npz_buf = io.BytesIO()
            np.savez_compressed(npz_buf, **traj)
            st.download_button("Download trajectory (.npz)", npz_buf.getvalue(), "trajectory.npz", "application/octet-stream")
            if "rdf" in results:
                st.download_button("Download RDF (.dat)", array_to_dat_bytes(results["rdf"]["r"], results["rdf"]["g_r"], headers=["r_A", "g_r"]), "rdf.dat", "text/plain")
            if "msd" in results:
                st.download_button("Download MSD (.dat)", array_to_dat_bytes(results["msd"]["time"], results["msd"]["msd"], headers=["time_fs", "msd_A2"]), "msd.dat", "text/plain")
            if "vacf" in results:
                st.download_button("Download VACF (.dat)", array_to_dat_bytes(results["vacf"]["time"], results["vacf"]["vacf"], headers=["time_fs", "vacf"]), "vacf.dat", "text/plain")
            if "structure_factor" in results:
                st.download_button("Download S(k) (.dat)", array_to_dat_bytes(results["structure_factor"]["k"], results["structure_factor"]["S_k"], headers=["k_1_per_A", "S_k"]), "structure_factor.dat", "text/plain")

elif mode == "Validation":
    tab_report, tab_plot, tab_passfail, tab_deck, tab_files = st.tabs([
        "Report", "Validation plots", "PASS / FAIL", "Input deck", "Files"
    ])

    with tab_report:
        st.code(report, language="text")

    with tab_plot:
        if results is None or kind != "validation":
            st.info("Run validation first.")
        else:
            choice = st.selectbox("Validation plot", VALIDATION_PLOTS, index=0)
            st.pyplot(plot_validation(results, choice), clear_figure=True)

    with tab_passfail:
        if results is None or kind != "validation":
            st.info("No validation results yet.")
        else:
            summary = results.get("summary", {})
            status_rows = [
                {"test": name, "status": "PASS" if bool(passed) else "FAIL"}
                for name, passed in summary.items()
            ]

            if status_rows:
                n_passed = sum(1 for row in status_rows if row["status"] == "PASS")
                n_total = len(status_rows)
                c1, c2, c3 = st.columns(3)
                c1.metric("Passed", n_passed)
                c2.metric("Total", n_total)
                c3.metric("Overall", "PASS" if n_passed == n_total else "FAIL")

                st.dataframe(status_rows, width="stretch", hide_index=True)

                st.divider()
                st.code(report, language="text")
            else:
                st.info("No PASS/FAIL summary was returned by the validation driver.")

    with tab_deck:
        st.code(
            lammps_input_deck(
                metal=metal,
                nx=int(nx), ny=int(ny), nz=int(nz),
                ensemble="nve",
                T0=float(T0),
                dt=float(dt),
                equil_steps=int(n_equil_steps),
                run_steps=int(n_steps),
                sample_every=int(sample_every),
                thermal_displacement=0.0,
            ),
            language="text",
        )

    with tab_files:
        if results is None or kind != "validation":
            st.info("Run validation first to enable downloads.")
        else:
            saved = results.get("saved_files", {})
            if saved:
                for name, path in saved.items():
                    st.write(f"**{name}**: `{path}`")
            else:
                st.caption("No files were saved to disk for this run.")
            st.download_button("Download report (.txt)", report.encode("utf-8"), "validation_report.txt", "text/plain")

else:
    tab_report, tab_plot, tab_table, tab_deck, tab_files = st.tabs([
        "Report", "Timing plots", "Benchmark table", "Input deck", "Files"
    ])

    with tab_report:
        st.code(report, language="text")

    with tab_plot:
        if results is None or kind != "performance":
            st.info("Run performance first.")
        else:
            choice = st.selectbox("Performance plot", PERFORMANCE_PLOTS, index=0)
            st.pyplot(plot_performance(results, choice), clear_figure=True)

    with tab_table:
        if results is None or kind != "performance":
            st.info("No performance results yet.")
        else:
            rows = []
            for case in results.get("cases", []):
                m = case["metadata"]
                rows.append({
                    "size": f"{m['nx']}x{m['ny']}x{m['nz']}",
                    "atoms": m["N"],
                    "pairs": case["force_evaluation"]["pairs"],
                    "neighbor_ms": 1000 * case["neighbor_build"]["mean_seconds"],
                    "force_ms": 1000 * case["force_evaluation"]["mean_seconds"],
                    "steps_per_s": case["integrator_nve"]["steps_per_second"],
                    "atom_steps_per_s": case["integrator_nve"]["atom_steps_per_second"],
                })
            st.dataframe(rows, width="stretch", hide_index=True)

    with tab_deck:
        first_size = parse_sizes(sizes_text)[0]
        st.code(
            lammps_input_deck(
                metal=metal,
                nx=int(first_size[0]), ny=int(first_size[1]), nz=int(first_size[2]),
                ensemble="nve",
                T0=float(T0),
                dt=float(dt),
                equil_steps=0,
                run_steps=int(n_steps),
                sample_every=10,
                thermal_displacement=0.0,
            ),
            language="text",
        )
        st.caption("Performance uses a fixed NVE timing kernel; size shown is the first benchmark size.")

    with tab_files:
        if results is None or kind != "performance":
            st.info("Run performance first to enable downloads.")
        else:
            saved = results.get("saved_files", {})
            if saved:
                for name, path in saved.items():
                    st.write(f"**{name}**: `{path}`")
            else:
                st.caption("No files were saved to disk for this run.")
            st.download_button("Download report (.txt)", report.encode("utf-8"), "performance_report.txt", "text/plain")
