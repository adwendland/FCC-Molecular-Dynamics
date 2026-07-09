### gui.py
### tkinter gui

from __future__ import annotations

import io
import queue
import sys
import threading
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter import font as tkfont

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:  # plotting stays optional
    plt = None
    FigureCanvasTkAgg = None

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
DEFAULT_ANALYSES = [
    "thermo",
    "rdf",
    "coordination_number",
    "msd",
    "vacf",
    "diffusion_msd",
    "diffusion_vacf",
    "structure_factor",
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
    "Total kinetic energy",
    "Component equipartition",
    "RDF validation",
]

PERFORMANCE_PLOTS = [
    "Performance scaling",
    "Kernel timing",
]


# -----------------------------------------------------------------------------
# Small helpers
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


def _analysis_output_dir_from_results(results: dict[str, Any], run_name: str | None = None) -> Path:
    """Choose the same default output folder style as the analysis driver."""
    meta = results.get("metadata", {})
    if not run_name:
        run_name = (
            f"{meta.get('metal', 'run')}_{meta.get('ensemble', 'nve')}_"
            f"{int(float(meta.get('T0', 0)))}K_"
            f"{meta.get('nx', '?')}x{meta.get('ny', '?')}x{meta.get('nz', '?')}"
        )
    out = PROJECT_ROOT / "outputs" / "analysis" / run_name
    out.mkdir(parents=True, exist_ok=True)
    return out

def center_positions_in_box(positions: np.ndarray, box) -> np.ndarray:
    pos = np.asarray(positions, dtype=float).copy()
    box = np.asarray(box, dtype=float)

    mins = pos.min(axis=0)
    maxs = pos.max(axis=0)
    center_atoms = 0.5 * (mins + maxs)
    center_box = 0.5 * box

    return pos + (center_box - center_atoms)

def write_xyz_snapshot(path: str | Path, positions: np.ndarray, element: str, box=None, comment: str = "Final configuration") -> Path:
    """Write one OVITO-compatible extended XYZ frame."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    positions = np.asarray(positions, dtype=float)

    if box is not None:
        positions = center_positions_in_box(positions, box)

    if box is not None:
        Lx, Ly, Lz = np.asarray(box, dtype=float)
        comment = (
            f'{comment} Lattice="{Lx:.8f} 0 0  0 {Ly:.8f} 0  0 0 {Lz:.8f}" '
            'Properties=species:S:1:pos:R:3'
        )

    with path.open("w", encoding="utf-8") as f:
        f.write(f"{len(positions)}\n")
        f.write(f"{comment}\n")
        for x, y, z in positions:
            f.write(f"{element} {x:.8f} {y:.8f} {z:.8f}\n")
    return path


def write_xyz_trajectory(path: str | Path, positions_traj: np.ndarray, element: str, times=None, steps=None, box=None) -> Path:
    """Write a multi-frame OVITO-compatible extended XYZ trajectory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    positions_traj = np.asarray(positions_traj, dtype=float)

    if times is None:
        times = [None] * len(positions_traj)
    if steps is None:
        steps = range(len(positions_traj))

    box_comment = ""
    if box is not None:
        Lx, Ly, Lz = np.asarray(box, dtype=float)
        box_comment = f' Lattice="{Lx:.8f} 0 0  0 {Ly:.8f} 0  0 0 {Lz:.8f}"'

    with path.open("w", encoding="utf-8") as f:
        for frame_idx, frame in enumerate(positions_traj):
            if box is not None:
                frame = center_positions_in_box(frame, box)
            step = int(steps[frame_idx]) if frame_idx < len(steps) else frame_idx
            time = times[frame_idx] if frame_idx < len(times) else None
            time_part = "" if time is None else f" Time={float(time):.8f}fs"
            f.write(f"{frame.shape[0]}\n")
            f.write(
                f"Frame={frame_idx} Step={step}{time_part}{box_comment} "
                "Properties=species:S:1:pos:R:3\n"
            )
            for x, y, z in frame:
                f.write(f"{element} {x:.8f} {y:.8f} {z:.8f}\n")
    return path


def save_gui_trajectory_outputs(results: dict[str, Any], run_name: str | None = None) -> dict[str, Path]:
    """Save both Python-native NPZ and OVITO-readable XYZ trajectory files."""
    if "trajectory" not in results:
        return {}

    meta = results.get("metadata", {})
    traj = results["trajectory"]
    out = _analysis_output_dir_from_results(results, run_name=run_name)
    element = str(meta.get("metal", "X"))
    a = float(meta.get("lattice_constant", 1.0))
    nx = int(meta.get("nx", 1))
    ny = int(meta.get("ny", 1))
    nz = int(meta.get("nz", 1))
    box = np.array([nx * a, ny * a, nz * a], dtype=float)

    positions = np.asarray(traj["positions_traj"])
    saved: dict[str, Path] = {}

    npz_path = out / "trajectory.npz"
    if not npz_path.exists():
        np.savez_compressed(npz_path, **traj)
    saved["trajectory_npz"] = npz_path

    saved["trajectory_xyz"] = write_xyz_trajectory(
        out / "trajectory.xyz",
        positions,
        element,
        times=traj.get("time_traj"),
        steps=traj.get("step_traj"),
        box=box,
    )
    saved["final_xyz"] = write_xyz_snapshot(
        out / "final.xyz",
        positions[-1],
        element,
        box=box,
        comment="Final configuration",
    )
    return saved


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
    """Build a readable, LAMMPS-like input summary for the current GUI state."""
    a = get_lattice_constant(metal)
    sigma = get_sigma(metal)
    eps = get_eps(metal)
    rcut = 2.5 * sigma
    atoms = 4 * nx * ny * nz
    tau = 100 * dt

    if ensemble == "nve":
        production_fix = "fix             prod all nve"
    else:
        production_fix = f"fix             prod all nvt temp {T0:g} {T0:g} {tau:g}"

    line = "#" + "=" * 76
    sub = "#" + "-" * 76
    return f"""{line}
# FCC MOLECULAR DYNAMICS INPUT SUMMARY
{line}

# ---- system ---------------------------------------------------------------
units           metal                 # eV, Angstrom, ps in LAMMPS convention
atom_style      atomic
boundary        p p p

lattice         fcc {a:.6f}            # metal = {metal}
region          simbox block 0 {nx} 0 {ny} 0 {nz}
create_box      1 simbox
create_atoms    1 simbox              # atoms = {atoms}
mass            1 <internal mass>      # app uses amu -> eV fs^2 / Angstrom^2

{sub}
# Lennard-Jones model
# Your code uses a 12-6 LJ potential with metal-specific epsilon and sigma.
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
# random_displace amplitude = {thermal_displacement:g} Angstrom

{sub}
# Equilibration
{sub}
fix             eq all nvt temp {T0:g} {T0:g} {tau:g}  # app thermostat stage
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


class LammpsMDGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FCC MD Workbench")
        self.geometry("1240x660+10+20")
        self.minsize(900, 600)

        self.result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.last_analysis: dict[str, Any] | None = None
        self.last_validation: dict[str, Any] | None = None
        self.last_performance: dict[str, Any] | None = None
        self.current_canvas = None

        self._setup_style()
        self._build_ui()
        self.after(100, self._poll_queue)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _setup_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Subheader.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Run.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Pass.TLabel", foreground="#0a7f32", font=("Segoe UI", 10, "bold"))
        style.configure("Fail.TLabel", foreground="#b00020", font=("Segoe UI", 10, "bold"))

    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=0)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)

        header = ttk.Frame(root)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Label(header, text="FCC Molecular Dynamics Workbench", style="Header.TLabel").pack(side="left")

        self.control_panel = ttk.Frame(root)
        self.control_panel.grid(row=1, column=0, sticky="ns", padx=(0, 10))

        right = ttk.Frame(root)
        right.grid(row=1, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self.output_notebook = ttk.Notebook(right)
        self.output_notebook.grid(row=0, column=0, sticky="nsew")

        self.report_tab = ttk.Frame(self.output_notebook)
        self.plot_tab = ttk.Frame(self.output_notebook)
        self.table_tab = ttk.Frame(self.output_notebook)
        self.deck_tab = ttk.Frame(self.output_notebook)
        self.output_notebook.add(self.report_tab, text="Report")
        self.output_notebook.add(self.plot_tab, text="Plot")
        self.output_notebook.add(self.table_tab, text="Status / tables")
        self.output_notebook.add(self.deck_tab, text="Input deck")

        self.report_box = self._make_text_box(self.report_tab, height=28, title=None)
        self.report_box["frame"].pack(fill="both", expand=True)

        self.plot_controls = ttk.Frame(self.plot_tab, padding=6)
        self.plot_controls.pack(fill="x")
        ttk.Label(self.plot_controls, text="Plot:").pack(side="left")
        self.plot_choice_var = tk.StringVar(value="Temperature")
        self.plot_choice = ttk.Combobox(
            self.plot_controls,
            textvariable=self.plot_choice_var,
            values=ANALYSIS_PLOTS + VALIDATION_PLOTS + PERFORMANCE_PLOTS,
            state="readonly",
            width=28,
        )
        self.plot_choice.pack(side="left", padx=6)
        ttk.Button(self.plot_controls, text="Show plot", command=self._show_selected_plot).pack(side="left")

        self.plot_frame = ttk.Frame(self.plot_tab, padding=6)
        self.plot_frame.pack(fill="both", expand=True)

        self.table_box = self._make_text_box(self.table_tab, height=28, title=None)
        self.table_box["frame"].pack(fill="both", expand=True)

        self.deck_box = self._make_text_box(self.deck_tab, height=28, title=None)
        self.deck_box["frame"].pack(fill="both", expand=True)

        self.control_panel.rowconfigure(0, weight=1)
        self.control_panel.columnconfigure(0, weight=1)
        self.tabs = ttk.Notebook(self.control_panel)
        self.tabs.grid(row=0, column=0, sticky="nsew")

        self.left_status = ttk.Label(
            self.control_panel,
            text="Ready | Ni | 4x4x4 | 256 atoms | NVE | dt=0.1 fs",
            anchor="w",
            relief="sunken",
            padding=(4, 2),
        )
        self.left_status.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        # Each left-side tab keeps the original stacked layout, but the tab body
        # is scrollable so the Output section and Run buttons are never clipped.
        self.run_page = ttk.Frame(self.tabs)
        self.validation_page = ttk.Frame(self.tabs)
        self.performance_page = ttk.Frame(self.tabs)
        self.run_tab = self._make_scrollable_control_tab(self.run_page)
        self.validation_tab = self._make_scrollable_control_tab(self.validation_page)
        self.performance_tab = self._make_scrollable_control_tab(self.performance_page)

        self.tabs.add(self.run_page, text="Run / Analysis")
        self.tabs.add(self.validation_page, text="Validation")
        self.tabs.add(self.performance_page, text="Performance")
        self.tabs.bind("<<NotebookTabChanged>>", lambda _e: (self._update_deck(), self._update_left_status("Ready")))

        self._build_run_controls()
        self._build_validation_controls()
        self._build_performance_controls()
        self._update_deck()
        self._write_report("Ready. Configure a run, validation suite, or benchmark, then click Run.")

    def _make_text_box(self, parent, *, height: int, title: str | None):
        frame = ttk.LabelFrame(parent, text=title, padding=5) if title else ttk.Frame(parent, padding=0)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        mono_family = "Consolas"
        try:
            mono = tkfont.Font(family=mono_family, size=10)
        except Exception:
            mono = tkfont.Font(family="Courier", size=10)
        text = tk.Text(frame, height=height, wrap="none", font=mono, state="disabled")
        yscroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        xscroll = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
        text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        text.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        return {"frame": frame, "text": text}

    def _make_scrollable_control_tab(self, parent):
        """Return a scrollable inner frame for a left-side control tab."""
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        canvas = tk.Canvas(parent, highlightthickness=0, borderwidth=0)
        yscroll = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, padding=8)

        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=yscroll.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")

        def _sync_scroll_region(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _sync_inner_width(event):
            canvas.itemconfigure(window_id, width=event.width)

        def _on_mousewheel(event):
            # Windows/macOS mouse wheel support.
            if event.delta:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_linux_scroll_up(_event):
            canvas.yview_scroll(-1, "units")

        def _on_linux_scroll_down(_event):
            canvas.yview_scroll(1, "units")

        def _bind_wheel(_event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Button-4>", _on_linux_scroll_up)
            canvas.bind_all("<Button-5>", _on_linux_scroll_down)

        def _unbind_wheel(_event):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        inner.bind("<Configure>", _sync_scroll_region)
        canvas.bind("<Configure>", _sync_inner_width)
        canvas.bind("<Enter>", _bind_wheel)
        canvas.bind("<Leave>", _unbind_wheel)

        return inner

    def _section(self, parent, title: str):
        box = ttk.LabelFrame(parent, text=title, padding=8)
        box.pack(fill="x", pady=(0, 8))
        box.columnconfigure(1, weight=1)
        return box

    def _entry(self, parent, row: int, label: str, var: tk.Variable, width=12):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ent = ttk.Entry(parent, textvariable=var, width=width)
        ent.grid(row=row, column=1, sticky="ew", pady=2, padx=(8, 0))
        ent.bind("<KeyRelease>", lambda _e: self._update_deck())
        return ent

    def _combo(self, parent, row, label, var, values, width=8):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")

        combo = ttk.Combobox(
            parent,
            textvariable=var,
            values=values,
            state="readonly",
            width=width,
        )
        combo.grid(row=row, column=1, sticky="ew")

    def _build_common_system_controls(self, parent, prefix: str, defaults: dict[str, Any]):
        vars_: dict[str, tk.Variable] = {}
        system = self._section(parent, "System")
        vars_["metal"] = tk.StringVar(value=defaults.get("metal", "Ni"))
        self._combo(system, 0, "metal", vars_["metal"], METALS)
        
        box_sizes = [str(i) for i in range(1, 10)]

        vars_["nx"] = tk.StringVar(value=str(defaults.get("nx", 4)))
        vars_["ny"] = tk.StringVar(value=str(defaults.get("ny", 4)))
        vars_["nz"] = tk.StringVar(value=str(defaults.get("nz", 4)))

        self._combo(system, 1, "nx", vars_["nx"], box_sizes)
        self._combo(system, 2, "ny", vars_["ny"], box_sizes)
        self._combo(system, 3, "nz", vars_["nz"], box_sizes)

        vars_["T0"] = tk.StringVar(value=str(defaults.get("T0", 300.0)))
        self._entry(system, 4, "temperature K", vars_["T0"])
        
        vars_["dt"] = tk.StringVar(value=str(defaults.get("dt", 0.1)))             
        dt_values = [
            "0.001",
            "0.002",
            "0.005",
            "0.01",
            "0.02",
            "0.05",
            "0.1",
            "0.2",
            "0.5",
        ]
        self._combo(system, 5, "timestep dt", vars_["dt"], dt_values)

        vars_["seed"] = tk.StringVar(value=str(defaults.get("seed", 123)))
        self._entry(system, 6, "seed", vars_["seed"])

        vars_["thermal_displacement"] = tk.StringVar(value=str(defaults.get("thermal_displacement", 0.01)))
        disp_values = [
            "0.0",
            "0.005",
            "0.01",
            "0.02",
            "0.03",
            "0.04",
            "0.05",
            "0.06",
            "0.07",
            "0.08",
            "0.09",
            "0.1"
        ]
        
        self._combo(system, 7, "thermal disp.", vars_["thermal_displacement"], disp_values)
        setattr(self, f"{prefix}_vars", vars_)
        return vars_

    def _build_run_controls(self):
        vars_ = self._build_common_system_controls(
            self.run_tab,
            "run",
            {"metal": "Ni", "nx": 4, "ny": 4, "nz": 4, "T0": 300.0, "dt": 0.1},
        )
        run = self._section(self.run_tab, "Run control")
        vars_["ensemble"] = tk.StringVar(value="nve")
        self._combo(run, 0, "ensemble", vars_["ensemble"], ["nve", "nvt"])
        vars_["n_equil_steps"] = tk.StringVar(value="20000")
        vars_["n_steps"] = tk.StringVar(value="100000")
        vars_["sample_every"] = tk.StringVar(value="500")
        self._entry(run, 1, "equil steps", vars_["n_equil_steps"])
        self._entry(run, 2, "production steps", vars_["n_steps"])
        self._entry(run, 3, "thermo/sample", vars_["sample_every"])

        analysis = self._section(self.run_tab, "Analysis computes")
        self.analysis_flags: dict[str, tk.BooleanVar] = {}
        for i, name in enumerate(sorted(AVAILABLE_ANALYSES)):
            v = tk.BooleanVar(value=name in DEFAULT_ANALYSES)
            self.analysis_flags[name] = v
            ttk.Checkbutton(analysis, text=name, variable=v).grid(row=i // 2, column=i % 2, sticky="w", padx=(0, 10))

        output = self._section(self.run_tab, "Output")
        vars_["save_outputs"] = tk.BooleanVar(value=False)
        vars_["save_trajectory"] = tk.BooleanVar(value=False)
        vars_["save_plots"] = tk.BooleanVar(value=False)
        vars_["run_name"] = tk.StringVar(value="")
        ttk.Checkbutton(output, text="save data", variable=vars_["save_outputs"]).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(output, text="save trajectory (.npz + .xyz)", variable=vars_["save_trajectory"]).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(output, text="save plots", variable=vars_["save_plots"]).grid(row=1, column=0, sticky="w")
        self._entry(output, 2, "run name", vars_["run_name"], width=18)

        ttk.Button(self.run_tab, text="Run analysis", style="Run.TButton", command=self.run_analysis).pack(fill="x", pady=(4, 0))

    def _build_validation_controls(self):
        vars_ = self._build_common_system_controls(
            self.validation_tab,
            "val",
            {"metal": "Ni", "nx": 4, "ny": 4, "nz": 4, "T0": 300.0, "dt": 0.1},
        )
        ctrl = self._section(self.validation_tab, "Validation control")
        vars_["n_equil_steps"] = tk.StringVar(value="20000")
        vars_["n_steps"] = tk.StringVar(value="100000")
        vars_["sample_every"] = tk.StringVar(value="500")
        vars_["refinement_dt"] = tk.StringVar(value="0.04")
        vars_["refinement_steps"] = tk.StringVar(value="500")
        self._entry(ctrl, 0, "equil steps", vars_["n_equil_steps"])
        self._entry(ctrl, 1, "test steps", vars_["n_steps"])
        self._entry(ctrl, 2, "sample every", vars_["sample_every"])
        self._entry(ctrl, 3, "refinement dt", vars_["refinement_dt"])
        self._entry(ctrl, 4, "refinement steps", vars_["refinement_steps"])

        tests = self._section(self.validation_tab, "Tests")
        self.test_flags: dict[str, tk.BooleanVar] = {}
        for i, name in enumerate(sorted(AVAILABLE_TESTS)):
            v = tk.BooleanVar(value=name in DEFAULT_TESTS)
            self.test_flags[name] = v
            ttk.Checkbutton(tests, text=name, variable=v).grid(row=i // 2, column=i % 2, sticky="w", padx=(0, 10))

        output = self._section(self.validation_tab, "Output")
        vars_["save_outputs"] = tk.BooleanVar(value=False)
        vars_["save_plots"] = tk.BooleanVar(value=False)
        vars_["run_name"] = tk.StringVar(value="")
        ttk.Checkbutton(output, text="save report/data", variable=vars_["save_outputs"]).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(output, text="save plots", variable=vars_["save_plots"]).grid(row=0, column=1, sticky="w")
        self._entry(output, 1, "run name", vars_["run_name"], width=18)

        ttk.Button(self.validation_tab, text="Run validation", style="Run.TButton", command=self.run_validation).pack(fill="x", pady=(4, 0))

    def _build_performance_controls(self):
        vars_: dict[str, tk.Variable] = {}
        self.perf_vars = vars_
        system = self._section(self.performance_tab, "System")
        vars_["metal"] = tk.StringVar(value="Ni")
        vars_["T0"] = tk.StringVar(value="300.0")
        vars_["dt"] = tk.StringVar(value="0.001")
        vars_["seed"] = tk.StringVar(value="123")
        vars_["thermal_displacement"] = tk.StringVar(value="0.01")
        self._combo(system, 0, "metal", vars_["metal"], METALS)
        self._entry(system, 1, "temperature K", vars_["T0"])
        self._entry(system, 2, "timestep fs", vars_["dt"])
        self._entry(system, 3, "seed", vars_["seed"])
        self._entry(system, 4, "thermal disp.", vars_["thermal_displacement"])

        ctrl = self._section(self.performance_tab, "Benchmark control")
        vars_["sizes"] = tk.StringVar(value="3x3x3,4x4x4,5x5x5,6x6x6")
        vars_["n_steps"] = tk.StringVar(value="100")
        vars_["repeats"] = tk.StringVar(value="5")
        vars_["warmup"] = tk.StringVar(value="2")
        vars_["backend"] = tk.StringVar(value="auto")
        self._entry(ctrl, 0, "sizes", vars_["sizes"], width=22)
        self._entry(ctrl, 1, "steps/repeat", vars_["n_steps"])
        self._entry(ctrl, 2, "repeats", vars_["repeats"])
        self._entry(ctrl, 3, "warmup", vars_["warmup"])
        self._combo(ctrl, 4, "backend", vars_["backend"], ["auto", "python", "cpp"], width=12)

        output = self._section(self.performance_tab, "Output")
        vars_["save_outputs"] = tk.BooleanVar(value=False)
        ttk.Checkbutton(output, text="save performance data", variable=vars_["save_outputs"]).grid(row=0, column=0, sticky="w")

        ttk.Button(self.performance_tab, text="Run benchmark", style="Run.TButton", command=self.run_performance).pack(fill="x", pady=(4, 0))

    # ------------------------------------------------------------------
    # Thread dispatch
    # ------------------------------------------------------------------
    def _start_worker(self, label: str, target, *args, **kwargs):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Run in progress", "A run is already in progress.")
            return
        self._write_report(f"Starting {label}...\n")
        self._write_table("")
        self._clear_plot()
        self.worker = threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True)
        self.worker.start()

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.result_queue.get_nowait()
                if kind == "analysis_done":
                    self.last_analysis = payload["results"]
                    self._write_report(payload["report"])
                    self._write_table(self._analysis_status_text(payload["results"]))
                    self._set_plot_choices(self._available_analysis_plots(payload["results"]))
                    self.output_notebook.select(self.report_tab)
                    self._update_left_status("Done")
                elif kind == "validation_done":
                    self.last_validation = payload["results"]
                    self._write_report(payload["report"])
                    self._write_table(self._validation_status_text(payload["results"]))
                    self._set_plot_choices(self._available_validation_plots(payload["results"]))
                    self.output_notebook.select(self.report_tab)
                    self._update_left_status("Done")
                elif kind == "performance_done":
                    self.last_performance = payload["results"]
                    self._write_report(payload["report"])
                    self._write_table(self._performance_table_text(payload["results"]))
                    self._set_plot_choices(PERFORMANCE_PLOTS)
                    self.output_notebook.select(self.table_tab)
                    self._update_left_status("Done")
                elif kind == "error":
                    self._write_report(payload)
                    self._update_left_status("Error")
                    messagebox.showerror("Run failed", "The run failed. See the Report tab for the traceback.")
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    # ------------------------------------------------------------------
    # Run actions
    # ------------------------------------------------------------------
    def run_analysis(self):
        try:
            v = self.run_vars
            analyses = [name for name, flag in self.analysis_flags.items() if flag.get()]
            if not analyses:
                raise ValueError("Select at least one analysis.")
            params = dict(
                metal=v["metal"].get(),
                nx=int(v["nx"].get()),
                ny=int(v["ny"].get()),
                nz=int(v["nz"].get()),
                T0=float(v["T0"].get()),
                ensemble=v["ensemble"].get(),
                dt=float(v["dt"].get()),
                n_equil_steps=int(v["n_equil_steps"].get()),
                n_steps=int(v["n_steps"].get()),
                sample_every=int(v["sample_every"].get()),
                analyses=analyses,
                seed=int(v["seed"].get()),
                thermal_displacement=float(v["thermal_displacement"].get()),
                save_outputs=bool(v["save_outputs"].get()),
                save_trajectory=bool(v["save_trajectory"].get()),
                save_plots=bool(v["save_plots"].get()),
                show_plots=False,
                run_name=v["run_name"].get().strip() or None,
            )
            self._update_deck()
        except Exception as exc:
            messagebox.showerror("Invalid input", str(exc))
            return
        self._start_worker("analysis", self._analysis_worker, params)

    def _analysis_worker(self, params: dict[str, Any]):
        try:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                results = run_analysis_suite(**params)

            # The analysis driver already saves trajectory.npz when both
            # save_outputs=True and save_trajectory=True. The GUI also writes
            # OVITO-readable XYZ files and ensures the NPZ exists when the
            # trajectory checkbox is selected.
            if params.get("save_trajectory", False):
                saved = save_gui_trajectory_outputs(results, run_name=params.get("run_name"))
                results.setdefault("saved_files", {}).update(saved)

            report = buffer.getvalue() + capture_report(print_analysis_report, results)
            self.result_queue.put(("analysis_done", {"results": results, "report": report}))
        except Exception:
            self.result_queue.put(("error", traceback.format_exc()))

    def run_validation(self):
        try:
            v = self.val_vars
            tests = [name for name, flag in self.test_flags.items() if flag.get()]
            if not tests:
                raise ValueError("Select at least one validation test.")
            params = dict(
                metal=v["metal"].get(),
                nx=int(v["nx"].get()),
                ny=int(v["ny"].get()),
                nz=int(v["nz"].get()),
                T0=float(v["T0"].get()),
                dt=float(v["dt"].get()),
                n_steps=int(v["n_steps"].get()),
                n_equil_steps=int(v["n_equil_steps"].get()),
                sample_every=int(v["sample_every"].get()),
                refinement_dt=float(v["refinement_dt"].get()),
                refinement_steps=int(v["refinement_steps"].get()),
                tests=tests,
                save_outputs=bool(v["save_outputs"].get()),
                save_plots=bool(v["save_plots"].get()),
                show_plots=False,
                run_name=v["run_name"].get().strip() or None,
            )
            self._update_deck()
        except Exception as exc:
            messagebox.showerror("Invalid input", str(exc))
            return
        self._start_worker("validation", self._validation_worker, params)

    def _validation_worker(self, params: dict[str, Any]):
        try:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                results = run_validation_suite(**params)
            report = buffer.getvalue() + capture_report(print_validation_report, results)
            self.result_queue.put(("validation_done", {"results": results, "report": report}))
        except Exception:
            self.result_queue.put(("error", traceback.format_exc()))

    def run_performance(self):
        try:
            v = self.perf_vars
            params = dict(
                metal=v["metal"].get(),
                sizes=parse_sizes(v["sizes"].get()),
                T0=float(v["T0"].get()),
                dt=float(v["dt"].get()),
                n_steps=int(v["n_steps"].get()),
                repeats=int(v["repeats"].get()),
                warmup=int(v["warmup"].get()),
                seed=int(v["seed"].get()),
                thermal_displacement=float(v["thermal_displacement"].get()),
                backend=v["backend"].get(),
                save_outputs=bool(v["save_outputs"].get()),
            )
            # Show a representative deck using first benchmark size.
            first = params["sizes"][0]
            self._write_deck(lammps_input_deck(
                metal=params["metal"], nx=first[0], ny=first[1], nz=first[2], ensemble="nve",
                T0=params["T0"], dt=params["dt"], equil_steps=0,
                run_steps=params["n_steps"], sample_every=max(1, params["n_steps"]),
                thermal_displacement=params["thermal_displacement"],
            ))
        except Exception as exc:
            messagebox.showerror("Invalid input", str(exc))
            return
        self._start_worker("performance benchmark", self._performance_worker, params)

    def _performance_worker(self, params: dict[str, Any]):
        try:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                results = run_performance_suite(**params)
            report = buffer.getvalue() + capture_report(print_performance_report, results)
            self.result_queue.put(("performance_done", {"results": results, "report": report}))
        except Exception:
            self.result_queue.put(("error", traceback.format_exc()))

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------
    def _write_text(self, box, text: str):
        widget = box["text"]
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.configure(state="disabled")

    def _write_report(self, text: str):
        self._write_text(self.report_box, text)

    def _write_table(self, text: str):
        self._write_text(self.table_box, text)

    def _write_deck(self, text: str):
        self._write_text(self.deck_box, text)

    def _active_left_vars(self):
        try:
            tab = self.tabs.index(self.tabs.select()) if hasattr(self, "tabs") else 0
        except Exception:
            tab = 0
        if tab == 0 and hasattr(self, "run_vars"):
            return self.run_vars, self.run_vars.get("ensemble", tk.StringVar(value="nve")).get()
        if tab == 1 and hasattr(self, "val_vars"):
            return self.val_vars, "nve"
        if tab == 2 and hasattr(self, "perf_vars"):
            return self.perf_vars, "nve"
        return None, "nve"

    def _update_left_status(self, prefix: str = "Ready"):
        if not hasattr(self, "left_status"):
            return
        vars_, ensemble = self._active_left_vars()
        if not vars_:
            return
        try:
            metal = vars_["metal"].get()
            nx = int(vars_.get("nx", tk.StringVar(value="4")).get())
            ny = int(vars_.get("ny", tk.StringVar(value="4")).get())
            nz = int(vars_.get("nz", tk.StringVar(value="4")).get())
            atoms = 4 * nx * ny * nz
            dt = float(vars_["dt"].get())
            self.left_status.config(
                text=f"{prefix} | {metal} | {nx}x{ny}x{nz} | {atoms} atoms | {ensemble.upper()} | dt={dt:g} fs"
            )
        except Exception:
            self.left_status.config(text=f"{prefix}")

    def _update_deck(self):
        self._update_left_status("Ready")
        try:
            tab = self.tabs.index(self.tabs.select()) if hasattr(self, "tabs") else 0
            if tab == 0 and hasattr(self, "run_vars"):
                v = self.run_vars
                self._write_deck(lammps_input_deck(
                    metal=v["metal"].get(), nx=int(v["nx"].get()), ny=int(v["ny"].get()), nz=int(v["nz"].get()),
                    ensemble=v["ensemble"].get(), T0=float(v["T0"].get()), dt=float(v["dt"].get()),
                    equil_steps=int(v["n_equil_steps"].get()), run_steps=int(v["n_steps"].get()),
                    sample_every=int(v["sample_every"].get()),
                    thermal_displacement=float(v["thermal_displacement"].get()),
                ))
            elif tab == 1 and hasattr(self, "val_vars"):
                v = self.val_vars
                self._write_deck(lammps_input_deck(
                    metal=v["metal"].get(), nx=int(v["nx"].get()), ny=int(v["ny"].get()), nz=int(v["nz"].get()),
                    ensemble="nve", T0=float(v["T0"].get()), dt=float(v["dt"].get()),
                    equil_steps=int(v["n_equil_steps"].get()), run_steps=int(v["n_steps"].get()),
                    sample_every=int(v["sample_every"].get()),
                    thermal_displacement=float(v["thermal_displacement"].get()),
                ))
        except Exception:
            # Avoid noisy errors while the user is halfway through typing.
            pass

    def _set_plot_choices(self, values: list[str]):
        self.plot_choice.configure(values=values)
        if values:
            self.plot_choice_var.set(values[0])

    def _analysis_status_text(self, results: dict[str, Any]) -> str:
        meta = results.get("metadata", {})
        summary = results.get("summary", {})
        saved = results.get("saved_files", {})
        lines = [
            "Run summary",
            "-" * 70,
            f"metal                      : {meta.get('metal')}",
            f"lattice                    : {meta.get('nx')} x {meta.get('ny')} x {meta.get('nz')}",
            f"atoms                      : {meta.get('N')}",
            f"ensemble                   : {meta.get('ensemble')}",
            f"dt                         : {fmt(meta.get('dt'))} fs",
            f"simulation time            : {fmt(results.get('simulation_time_fs'))} fs",
            f"runtime                    : {fmt(results.get('runtime_seconds'))} s",
            "",
            "Metrics",
            "-" * 70,
        ]
        for key in sorted(summary):
            lines.append(f"{key:28s}: {fmt(summary[key])}")
        if saved:
            lines += ["", "Saved files", "-" * 70]
            for key, path in saved.items():
                lines.append(f"{key:28s}: {path}")
        return "\n".join(lines)

    def _validation_status_text(self, results: dict[str, Any]) -> str:
        summary = results.get("summary", {})
        n_pass = sum(bool(v) for v in summary.values())
        lines = [
            "Validation status",
            "-" * 70,
            f"overall                    : {'PASS' if n_pass == len(summary) and summary else 'FAIL'} ({n_pass}/{len(summary)})",
            "",
        ]
        for name, passed in summary.items():
            lines.append(f"{name:28s}: {'PASS' if passed else 'FAIL'}")
        saved = results.get("saved_files", {})
        if saved:
            lines += ["", "Saved files", "-" * 70]
            for key, path in saved.items():
                lines.append(f"{key:28s}: {path}")
        return "\n".join(lines)

    def _performance_table_text(self, results: dict[str, Any]) -> str:
        lines = [
            "Performance table",
            "-" * 106,
            f"{'size':>8s} {'atoms':>8s} {'pairs':>10s} {'NL build ms':>14s} {'force ms':>12s} {'steps/s':>12s} {'atom-steps/s':>16s} {'ns/day':>12s}",
            "-" * 106,
        ]
        for case in results.get("cases", []):
            m = case["metadata"]
            nb = case["neighbor_build"]
            force = case["force_evaluation"]
            integ = case["integrator_nve"]
            size = f"{m['nx']}x{m['ny']}x{m['nz']}"
            lines.append(
                f"{size:>8s} {m['N']:8d} {force.get('pairs', 0):10d} "
                f"{1000 * nb.get('mean_seconds', np.nan):14.3f} "
                f"{1000 * force.get('mean_seconds', np.nan):12.3f} "
                f"{integ.get('steps_per_second', np.nan):12.2f} "
                f"{integ.get('atom_steps_per_second', np.nan):16.2e} "
                f"{integ.get('simulated_ns_per_day', np.nan):12.4f}"
            )
        saved = results.get("saved_files", {})
        if saved:
            lines += ["", "Saved files", "-" * 70]
            for key, path in saved.items():
                lines.append(f"{key:28s}: {path}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def _clear_plot(self):
        for child in self.plot_frame.winfo_children():
            child.destroy()
        self.current_canvas = None

    def _available_analysis_plots(self, results: dict[str, Any]) -> list[str]:
        """Return every analysis plot from md/plotting.py that has data available."""
        plots = ["Temperature", "Energy"]  # always available from trajectory
        if "rdf" in results:
            plots.append("RDF")
        if "structure_factor" in results:
            plots.append("Structure factor")
        if "msd" in results:
            plots.append("MSD")
        if "vacf" in results:
            plots.append("VACF")
        return plots

    def _available_validation_plots(self, results: dict[str, Any]) -> list[str]:
        """Return every validation plot from md/plotting.py that has data available."""
        plots = []
        if "energy_drift" in results:
            plots.append("Energy conservation")
        if "momentum" in results:
            plots.append("Momentum conservation")
        if "timestep_refinement" in results:
            plots.append("Timestep convergence")
        if "temperature_stability" in results:
            plots.append("Temperature stability")
        if "total_kinetic_energy" in results:
            plots.append("Total kinetic energy")
        if "component_equipartition" in results:
            plots.append("Component equipartition")
        if "rdf_peaks" in results:
            plots.append("RDF validation")
        return plots

    def _show_selected_plot(self):
        if plt is None or FigureCanvasTkAgg is None:
            messagebox.showinfo("Plotting unavailable", "matplotlib Tk support is not available.")
            return

        choice = self.plot_choice_var.get()
        fig = None

        try:
            if self.last_analysis and choice in ANALYSIS_PLOTS:
                fig = self._analysis_plot(self.last_analysis, choice)
            elif self.last_validation and choice in VALIDATION_PLOTS:
                fig = self._validation_plot(self.last_validation, choice)
            elif self.last_performance and choice in PERFORMANCE_PLOTS:
                fig = self._performance_plot(self.last_performance, choice)
        except Exception as exc:
            messagebox.showerror("Plot failed", str(exc))
            return

        if fig is None:
            messagebox.showinfo(
                "No plot",
                "That plot is not available for the most recent run. "
                "Run the matching suite and make sure the corresponding analysis/test was selected.",
            )
            return

        self._clear_plot()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.current_canvas = canvas
        self.output_notebook.select(self.plot_tab)

    def _meta_for_plot(self, results: dict[str, Any]) -> dict[str, Any]:
        meta = dict(results.get("metadata", {}))
        if "simulation_time_fs" in results:
            meta.setdefault("time_ps", float(results["simulation_time_fs"]) / 1000.0)
        else:
            traj = results.get("trajectory", {})
            times = traj.get("time_traj", [])
            if len(times):
                meta.setdefault("time_ps", float(np.asarray(times)[-1]) / 1000.0)
            else:
                meta.setdefault("time_ps", 0.0)
        return meta

    def _decorate_axes(self, fig, ax):
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        return fig

    def _analysis_plot(self, results: dict[str, Any], choice: str):
        """GUI versions of all analysis plots exposed by md/plotting.py."""
        traj = results["trajectory"]
        meta = self._meta_for_plot(results)
        fig, ax = plt.subplots(figsize=(7.8, 4.8))

        if choice == "Temperature":
            ax.plot(traj["time_traj"], traj["temp_traj"])
            ax.set_xlabel("Time (fs)")
            ax.set_ylabel("Temperature (K)")
            ax.set_title(
                f"Temperature (K) vs Time (fs)\n"
                f"{meta.get('metal', '')} ({meta.get('nx', '?')}×{meta.get('ny', '?')}×{meta.get('nz', '?')}) | "
                f"{meta.get('ensemble', '').upper()} | {float(meta.get('T0', 0.0)):.0f} K | "
                f"{float(meta.get('time_ps', 0.0)):g} ps"
            )

        elif choice == "Energy":
            ax.plot(traj["time_traj"], traj["kinetic_traj"], label="Kinetic")
            ax.plot(traj["time_traj"], traj["potential_traj"], label="Potential")
            ax.plot(traj["time_traj"], traj["energy_traj"], label="Total")
            ax.set_xlabel("Time (fs)")
            ax.set_ylabel("Energy (eV)")
            ax.set_title(
                f"Energy (eV) vs Time (fs)\n"
                f"{meta.get('metal', '')} ({meta.get('nx', '?')}×{meta.get('ny', '?')}×{meta.get('nz', '?')}) | "
                f"{meta.get('ensemble', '').upper()} | {float(meta.get('T0', 0.0)):.0f} K | "
                f"{float(meta.get('time_ps', 0.0)):g} ps"
            )
            ax.legend()

        elif choice == "RDF" and "rdf" in results:
            ax.plot(results["rdf"]["r"], results["rdf"]["g_r"])
            ax.set_xlabel("r (Å)")
            ax.set_ylabel("g(r)")
            ax.set_title(
                f"Radial Distribution Function\n"
                f"{meta.get('metal', '')} ({meta.get('nx', '?')}×{meta.get('ny', '?')}×{meta.get('nz', '?')}) | "
                f"{meta.get('ensemble', '').upper()} | {float(meta.get('T0', 0.0)):.0f} K | "
                f"{float(meta.get('time_ps', 0.0)):g} ps"
            )

        elif choice == "Structure factor" and "structure_factor" in results:
            ax.plot(results["structure_factor"]["k"], results["structure_factor"]["S_k"])
            ax.set_xlabel("k")
            ax.set_ylabel("S(k)")
            ax.set_title(
                f"Structure Factor\n"
                f"{meta.get('metal', '')} ({meta.get('nx', '?')}×{meta.get('ny', '?')}×{meta.get('nz', '?')}) | "
                f"{meta.get('ensemble', '').upper()} | {float(meta.get('T0', 0.0)):.0f} K | "
                f"{float(meta.get('time_ps', 0.0)):g} ps"
            )

        elif choice == "MSD" and "msd" in results:
            ax.plot(results["msd"]["time"], results["msd"]["msd"])
            ax.set_xlabel("Time (fs)")
            ax.set_ylabel("MSD (Å²)")
            ax.set_title(
                f"Mean Squared Displacement\n"
                f"{meta.get('metal', '')} ({meta.get('nx', '?')}×{meta.get('ny', '?')}×{meta.get('nz', '?')}) | "
                f"{meta.get('ensemble', '').upper()} | {float(meta.get('T0', 0.0)):.0f} K | "
                f"{float(meta.get('time_ps', 0.0)):g} ps"
            )

        elif choice == "VACF" and "vacf" in results:
            ax.plot(results["vacf"]["time"], results["vacf"]["vacf"])
            ax.set_xlabel("Time (fs)")
            ax.set_ylabel("VACF")
            ax.set_title(
                f"Velocity Autocorrelation Function\n"
                f"{meta.get('metal', '')} ({meta.get('nx', '?')}×{meta.get('ny', '?')}×{meta.get('nz', '?')}) | "
                f"{meta.get('ensemble', '').upper()} | {float(meta.get('T0', 0.0)):.0f} K | "
                f"{float(meta.get('time_ps', 0.0)):g} ps"
            )

        else:
            plt.close(fig)
            return None

        return self._decorate_axes(fig, ax)

    def _validation_plot(self, results: dict[str, Any], choice: str):
        """GUI versions of all validation plots exposed by md/plotting.py."""
        meta = self._meta_for_plot(results)
        tol = results.get("tolerances", {})
        fig, ax = plt.subplots(figsize=(7.8, 4.8))

        if choice == "Energy conservation" and "energy_drift" in results:
            r = results["energy_drift"]
            ax.plot(r["times"], r["rel_drift"], label="Relative energy drift")
            tolerance = tol.get("energy_drift", None)
            if tolerance is not None:
                ax.axhline(tolerance, linestyle="--", linewidth=1.2, alpha=0.7, label="Tolerance")
                ax.axhline(-tolerance, linestyle="--", linewidth=1.2, alpha=0.7)
            ax.set_xlabel("Time (fs)")
            ax.set_ylabel("Relative energy drift")
            ax.set_title(
                f"Energy Conservation\n"
                f"{meta.get('metal', '')} ({meta.get('nx', '?')}×{meta.get('ny', '?')}×{meta.get('nz', '?')}) | "
                f"{float(meta.get('T0', 0.0)):.0f} K | {float(meta.get('time_ps', 0.0)):g} ps"
            )
            ax.legend()

        elif choice == "Momentum conservation" and "momentum" in results:
            r = results["momentum"]
            y = r.get("normalized_momentum_drift", r.get("relative_momentum_drift", r.get("momentum_norm")))
            ax.plot(r["times"], y)
            ax.set_xlabel("Time (fs)")
            ax.set_ylabel("|P| / scale")
            ax.set_title(
                f"Total Momentum\n"
                f"{meta.get('metal', '')} ({meta.get('nx', '?')}×{meta.get('ny', '?')}×{meta.get('nz', '?')}) | "
                f"{float(meta.get('T0', 0.0)):.0f} K | {float(meta.get('time_ps', 0.0)):g} ps"
            )

        elif choice == "Timestep convergence" and "timestep_refinement" in results:
            r = results["timestep_refinement"]
            rows = sorted((float(dt), float(err)) for dt, err in r.get("errors", {}).items())
            if not rows:
                plt.close(fig)
                return None
            dt_values = np.asarray([row[0] for row in rows])
            errors = np.asarray([row[1] for row in rows])
            ax.loglog(dt_values, errors, "o-", lw=2, ms=6, label="Measured")
            if len(dt_values) > 0 and errors[-1] > 0:
                dt_ref = dt_values[-1]
                err_ref = errors[-1]
                ax.loglog(dt_values, err_ref * (dt_values / dt_ref), "--", lw=1.2, label="O(dt)")
                ax.loglog(dt_values, err_ref * (dt_values / dt_ref) ** 2, "--", lw=1.2, label="O(dt²)")
            ax.invert_xaxis()
            ax.text(
                0.03,
                0.05,
                f"Observed order = {float(r.get('order', np.nan)):.2f}",
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.8"),
            )
            ax.set_xlabel("Time step (fs)")
            ax.set_ylabel("Position error")
            ax.set_title(
                f"Time Step Convergence\n"
                f"{meta.get('metal', '')} ({meta.get('nx', '?')}×{meta.get('ny', '?')}×{meta.get('nz', '?')}) | "
                f"{float(meta.get('T0', 0.0)):.0f} K"
            )
            ax.legend()

        elif choice == "Temperature stability" and "temperature_stability" in results:
            r = results["temperature_stability"]
            ax.plot(r["times"], r["temperatures"])
            if "target_temperature" in r:
                ax.axhline(r["target_temperature"], linestyle="--", linewidth=1.2, label=f"Target ({r['target_temperature']:.0f} K)")
                ax.legend()
            ax.set_xlabel("Time (fs)")
            ax.set_ylabel("Temperature (K)")
            ax.set_title(
                f"Temperature Stability\n"
                f"{meta.get('metal', '')} ({meta.get('nx', '?')}×{meta.get('ny', '?')}×{meta.get('nz', '?')}) | "
                f"{float(meta.get('T0', 0.0)):.0f} K | {float(meta.get('time_ps', 0.0)):g} ps"
            )

        elif choice == "Total kinetic energy" and "total_kinetic_energy" in results:
            r = results["total_kinetic_energy"]
            ax.plot(r["times"], r["kinetic_energies"], label="Measured")
            ax.axhline(r["expected_kinetic_energy"], linestyle="--", label="Expected")
            ax.set_xlabel("Time (fs)")
            ax.set_ylabel("Total kinetic energy (eV)")
            ax.set_title(
                f"Total Kinetic Energy\n"
                f"{meta.get('metal', '')} ({meta.get('nx', '?')}×{meta.get('ny', '?')}×{meta.get('nz', '?')}) | "
                f"{float(meta.get('T0', 0.0)):.0f} K | {float(meta.get('time_ps', 0.0)):g} ps"
            )
            ax.legend()

        elif choice == "Component equipartition" and "component_equipartition" in results:
            r = results["component_equipartition"]
            ax.plot(r["times"], r["kinetic_x"], label="Kx")
            ax.plot(r["times"], r["kinetic_y"], label="Ky")
            ax.plot(r["times"], r["kinetic_z"], label="Kz")
            ax.axhline(r["expected_component_energy"], linestyle="--", label="Expected")
            ax.set_xlabel("Time (fs)")
            ax.set_ylabel("Component kinetic energy (eV)")
            ax.set_title(
                f"Component Equipartition\n"
                f"{meta.get('metal', '')} ({meta.get('nx', '?')}×{meta.get('ny', '?')}×{meta.get('nz', '?')}) | "
                f"{float(meta.get('T0', 0.0)):.0f} K | {float(meta.get('time_ps', 0.0)):g} ps"
            )
            ax.legend()

        elif choice == "RDF validation" and "rdf_peaks" in results:
            r = results["rdf_peaks"]
            ax.plot(r["r"], r["g_r"])
            expected_peak = None
            if "rdf_first_peak" in results:
                expected_peak = results["rdf_first_peak"].get("expected", None)
            if expected_peak is not None:
                ax.axvline(expected_peak, linestyle="--", linewidth=1.2, label="Expected FCC peak")
                ax.legend()
            ax.set_xlabel("r (Å)")
            ax.set_ylabel("g(r)")
            ax.set_title(
                f"RDF Validation\n"
                f"{meta.get('metal', '')} ({meta.get('nx', '?')}×{meta.get('ny', '?')}×{meta.get('nz', '?')}) | "
                f"{float(meta.get('T0', 0.0)):.0f} K | {float(meta.get('time_ps', 0.0)):g} ps"
            )

        else:
            plt.close(fig)
            return None

        return self._decorate_axes(fig, ax)

    def _performance_plot(self, results: dict[str, Any], choice: str):
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        cases = results.get("cases", [])
        atoms = np.array([c["metadata"]["N"] for c in cases])
        if choice == "Performance scaling":
            y = np.array([c["integrator_nve"].get("atom_steps_per_second", np.nan) for c in cases])
            ax.plot(atoms, y, marker="o")
            ax.set_xlabel("Atoms")
            ax.set_ylabel("atom-steps/s")
            ax.set_title("Integrator throughput")
        elif choice == "Kernel timing":
            nb = np.array([1000 * c["neighbor_build"].get("mean_seconds", np.nan) for c in cases])
            force = np.array([1000 * c["force_evaluation"].get("mean_seconds", np.nan) for c in cases])
            ax.plot(atoms, nb, marker="o", label="Neighbor list")
            ax.plot(atoms, force, marker="o", label="Force")
            ax.set_xlabel("Atoms")
            ax.set_ylabel("Time (ms)")
            ax.set_title("Kernel timing")
            ax.legend()
        else:
            plt.close(fig)
            return None
        return self._decorate_axes(fig, ax)


if __name__ == "__main__":
    app = LammpsMDGUI()
    app.mainloop()
