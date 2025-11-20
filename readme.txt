========================================================================
======================== readme.txt ====================================
========================================================================

FCC Molecular Dynamics Simulator
This project implements a functional molecular dynamics (MD) simulation
for 8 FCC metals (Ag, Al, Au, Cu, Ni, Pb, Pd, Pt). Forces are calculated
with 12-6 Lennard-Jones potentials, and the integration is done with a 
velocity Verlet "leapfrog" routine featuring a neighbor list. The code
is written in Python with a modular design with the goal of being easily
readable and extendable.

INCLUDED:
• Modular MD backend (velocity Verlet routine in integrator.py, LJ forces routine in forces.py, etc.)
• Implementation of physically-relevant 12-6 Lennard–Jones interatomic potentials
• NVE and NVT (Berendsen thermostat) ensembles
• FCC lattice generator (lattice.py) for FCC structure
• Complete physical analysis: RDF, MSD, VACF
• GUI (tkinter)
• Optional OVITO visualizer
• Plotting tools for saved analysis files



===================================================================================
1.  Project structure
main.py:	MD runner in terminal (full analysis)
gui.py:		GUI (tkinter) for user-friendly simulations

md/constants.py:	 Lattice constants, amus, LJ sigma/epsilon
md/lattice.py:		 FCC lattice generator
md/forces.py:		 Lennard–Jones force calculation
md/neighborlist.py:	 Neighbor list for integrator (with skin distance)
md/system.py:		 System state: positions, velocities, forces, box, energy
md/integrator.py:	 Velocity-Verlet (NVE) and NVT Berendsen integration
md/analysis.py:		 RDF, MSD, VACF, coordination number, S(k), diffusion, Cv
md/plot_analysis.py: 	 Loads rdf.dat, msd.dat, vacf.dat and produces plots
md/utils.py:		 .xyz trajectory writer for OVITO visualizer



=======================================================================================
2.  Features
• FCC lattice generation for any nx, ny, nz (num cells in x,y,z direction, resp.)
• Velocity initialization consistent with temperature (K) from Boltzmann distribution
• Periodic boundary conditions (PBC) to simulate "bulk" material
• Automatic neighbor-list rebuilds as needed using displacement criterion
• Velocity-Verlet integration for NVE
• Berendsen thermostat for NVT
• traj.xyz trajectory writer for OVITO
• Analysis of relevant physical quantities:
- Radial Distribution Function g(r)
- Mean Squared Displacement (MSD)
- Velocity Autocorrelation Function (VACF)
- Derived quantities.



=======================================================================================================
3.  Units
Physical units are used for all input/output.

• Distance: Angstrom (Å)
• Energy: eV
• Mass: amu (internally converted using 103.6427 → eV·fs²/Å²)
• Temperature: Kelvin
• Time step dt: femtoseconds (fs)
• Pressure: eV / Å³
• Diffusion: Å² / fs

LJ parameters (sigma in Å, epsilon in eV) for each metal are automatically loaded from constants.py.



=========================================================================================================
4. Running a simulation
A. Using gui.py

Run:
	python gui.py 	(or py gui.py)

Select desired inputs; click "Run simulation".

The GUI will:
• Run equilibration and production steps
• Display energies during run
• Write titled .xyz file
• Produce plots for RDF, MSD, normalized VACF, S(k), T(t), P(t)	


B. Using main.py
First select desired inputs in main.py. Then:

	python. main.py		(or py main.py)

This performs the same analysis as the GUI.


C. Use web app (streamlit_app.py)



=================================================================================
5.  Extending the code
• Add new metal: Update constants.py
• Add new crystal structure: Add routines to lattice.py
• Replace LJ with another potential (e.g., EAM): extend forces.py (add method)
• Other thermostats (e.g., Nose-Hoover): extend integrator.py (add method)
• Other physical analysis: extend analysis.py/plot_analysis.py (add method)



=================================================================================
6.  License
This project is released under the MIT License.
(See LICENSE file included in the repository.)



==================================================================================
7.  References

H. Heinz, R. A. Vaia, B. L. Farmer, R. R. Naik, Accurate simulation of surfaces and interfaces of face-centered cubic metals using 12-6 and 9-6 Lennard-Jones potentials, J. Phys. Chem. C (112), no. 44, October 2008.
• Relevant constants for LJ potential

Jarek Miller, Molecular Dynamics, Encylopedia of Life Sciences, Nature Publishing Group, 2001.
• General MD reference

University of Helsinki, Constructing a neighbor list (slides/lecture notes)
• Neighbor list reference



===================================================================================
8.  Additional notes.
• The 12-6 Lennard-Jones potential isn't used for most FCC solids since it is a pair
potential, meaning it only considers pairwise interactions between particles. In a
real solid, many particles interact simultaneously. More sophisticated models (such 
as EAM) have been used to improve physical accuracy. However, the LJ potentials are
the easiest to understand (at least coming from a traditional energy-minimization/
gradient flow/descent framework), which is why I chose them here in my first MD code.
The code will support a switch to EAM potentials easily; just add a new routine to 
forces.py.
	


