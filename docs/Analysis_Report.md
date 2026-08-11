# Analysis Report

**FCC Molecular Dynamics Simulator**  
**Author:** Alec D. Wendland

---

## Abstract

This report documents the physical analysis capabilities of the FCC Molecular Dynamics Simulator and summarizes representative results obtained from simulations of Ag, Al, Au, Cu, Ni, Pb, Pd, and Pt. The analysis suite computes thermodynamic, structural, and dynamical observables including temperature, pressure, energy, the radial distribution function (RDF), coordination number, static structure factor, mean squared displacement (MSD), velocity autocorrelation function (VACF), diffusion coefficients, and energy-fluctuation estimates of the heat capacity.

All eight supported FCC metals were analyzed at 300 K in both NVE and NVT production runs. In addition, Cu and Ni were simulated over a broad temperature range from 300 K to 12,000 K in order to examine temperature-dependent loss of crystalline order and the onset of diffusive motion. At low temperature, the RDF reproduces the expected FCC coordination shells, the first-neighbor peak agrees with the theoretical FCC nearest-neighbor distance to within approximately 0.56% for every metal, and the coordination number is 12.000. The corresponding MSD remains bounded, consistent with atoms vibrating about lattice sites rather than undergoing long-range diffusion.

The high-temperature Cu and Ni simulations exhibit a clear change from solid-like to liquid-like behavior. For Cu, the 4000 K simulations retain a bounded MSD and recognizable higher-order RDF structure, whereas the 6000 K simulations exhibit an approximately linear, multi-tens-of-Å$^2$ MSD together with a broad liquid-like RDF. For Ni, the analogous change occurs between 6000 K and 8000 K. These transition intervals are observed in both NVT and NVE production runs and are supported independently by structural and dynamical observables. They should not, however, be interpreted as predictions of experimental melting temperatures: the simulations employ a finite, defect-free periodic FCC cell, fixed volume, Lennard–Jones interactions, and short simulation times. The results instead demonstrate that the simulator and analysis framework reproduce the expected qualitative signatures of a solid-to-liquid transition.

---

## 1. Introduction

Molecular dynamics produces microscopic trajectories of atomic positions and velocities, but the trajectories themselves are only the starting point for physical interpretation. Useful information is obtained by reducing those trajectories to thermodynamic averages, structural correlation functions, and dynamical observables that describe how the simulated material is organized and how atoms move over time [1–4].

The FCC Molecular Dynamics Simulator therefore includes an automated analysis suite intended to complement the numerical verification described in the separate `Validation_Report.md`. The Validation Report addresses whether the equations of motion, force calculations, numerical integration, and equilibrium FCC structure are implemented correctly. The present report addresses a different question: **what physical information can be extracted from the resulting trajectories?**

The analysis is organized around three complementary classes of quantities:

- **Thermodynamic observables**, including temperature, pressure, total energy, and energy fluctuations;
- **Structural observables**, including the radial distribution function, coordination number, and static structure factor;
- **Dynamical observables**, including the mean squared displacement, velocity autocorrelation function, and diffusion coefficients.

No single observable is sufficient to characterize a phase or dynamical regime. An ordered solid, for example, should simultaneously exhibit sharp coordination shells in the RDF, strong oscillatory structure in $S(k)$, and a bounded long-time MSD. A liquid should retain short-range nearest-neighbor correlations while losing long-range crystalline order, and it should exhibit sustained growth of the MSD associated with atomic diffusion [1,3,5]. The central goal of this report is therefore to interpret the observables together rather than as isolated numerical outputs.

The dataset contains equilibrium simulations for all eight supported FCC metals at 300 K and extended temperature sweeps for Cu and Ni. The latter provide a useful demonstration of how the same analysis framework distinguishes low-temperature crystalline behavior from high-temperature liquid-like behavior.

---

## 2. Simulation Methodology and Analysis Protocol

### 2.1 Molecular Dynamics Model

The simulator evolves classical particles according to Newton's equations of motion,

```math
m_i\frac{d^2\mathbf{r}_i}{dt^2}=\mathbf{F}_i,
```

with pairwise forces obtained from the 12–6 Lennard–Jones potential,

```math
U(r)=4\varepsilon
\left[
\left(\frac{\sigma}{r}\right)^{12}
-
\left(\frac{\sigma}{r}\right)^6
\right].
```

Metal-specific lattice constants, atomic masses, and Lennard–Jones parameters are provided for Ag, Al, Au, Cu, Ni, Pb, Pd, and Pt using the parameterization described by Heinz *et al.* [6]. Periodic boundary conditions and the minimum-image convention are used throughout.

Particle trajectories are integrated with the Velocity Verlet method. NVT simulations use the Berendsen thermostat [7], while NVE production runs evolve without thermostatting after the initial equilibration stage. Further implementation and numerical-validation details are given in `Validation_Report.md`.

### 2.2 Simulation Conditions

All analysis datasets use the same system size and basic numerical parameters.

| Parameter | Value |
|---|---:|
| Crystal structure | FCC |
| Unit cells | $5\times5\times5$ |
| Number of atoms | 500 |
| Production timestep | $0.1$ fs |
| Equilibration steps | 20,000 |
| Production steps | 100,000 |
| Production time | 10 ps |
| Sampling interval | 500 steps = 50 fs |
| Saved production samples | 201 |
| Ensembles | NVE and NVT |
| Pair cutoff | $2.5\sigma$ |

Before production, each system is equilibrated for 20,000 steps using the Berendsen thermostat at the requested temperature. Production is then performed either in NVT, retaining the thermostat, or in NVE, removing the thermostat after equilibration.

All eight metals were simulated at 300 K in both ensembles. Cu and Ni were additionally simulated at

```math
T = 1200,\ 2000,\ 4000,\ 6000,\ 8000,\ 10000,\ 12000\ {\rm K},
```

again in both NVE and NVT. In total, the analysis archive contains 44 production datasets.

| Metals | Temperatures | Production ensembles |
|---|---|---|
| Ag, Al, Au, Pb, Pd, Pt | 300 K | NVE, NVT |
| Cu | 300–12,000 K | NVE, NVT |
| Ni | 300–12,000 K | NVE, NVT |

The controlled-temperature NVT runs are used primarily for temperature-dependent structural comparisons. The NVE runs provide a useful companion for dynamical quantities because the absence of continuous thermostat rescaling during production avoids direct thermostat modification of velocity correlations.

![Figure 1. Analysis workflow](../figures/analysis/analysis_workflow.png)

**Figure 1.** Analysis workflow for the FCC Molecular Dynamics Simulator. Each simulation is equilibrated before production, after which thermodynamic, structural, and dynamical observables are extracted from the saved trajectory.

---

## 3. Thermodynamic Analysis

### 3.1 Temperature

The instantaneous temperature is obtained from the kinetic energy,

```math
T=
\frac{2K}
{(3N-3)k_B},
```

where the three center-of-mass translational degrees of freedom are removed.

The NVT simulations remain close to their requested target temperatures across the complete dataset. Considering all NVT runs, including the Cu and Ni high-temperature sweeps, the time-averaged production temperature differs from the requested target by less than approximately **0.35%**. Representative values are

| System | Target $T$ (K) | Mean $T$ (K) | Std. dev. (K) |
|---|---:|---:|---:|
| Cu, NVT | 300 | 300.12 | 4.18 |
| Cu, NVT | 6000 | 5990.11 | 104.60 |
| Cu, NVT | 12000 | 12020.96 | 217.42 |
| Ni, NVT | 300 | 300.02 | 4.07 |
| Ni, NVT | 8000 | 7991.05 | 145.19 |
| Ni, NVT | 12000 | 12009.61 | 184.81 |

The increasing absolute magnitude of the fluctuations with temperature is expected, while the mean values remain tightly centered near the target temperatures. This provides a controlled basis for comparing structural observables across the temperature sweeps.

The NVE runs are initialized from the same thermostatted equilibration stage and then evolve at constant total energy. Their mean kinetic temperatures therefore need not remain exactly equal to the nominal initialization temperature, particularly when structural rearrangement redistributes energy between kinetic and potential modes. This distinction is important when interpreting high-temperature NVE results.

<p align="center">
  <img src="../figures/analysis/Cu_300K_temperature.png" alt="Cu temperature vs time distribution, 300 K" width="48%">
  <img src="../figures/analysis/Cu_12000K_temperature.png" alt="Cu temperature vs time distribution, 12,000 K" width="48%">
</p>

<p align="center">
  <img src="../figures/analysis/Ni_300K_temperature.png" alt="Ni temperature vs time distribution, 300 K" width="48%">
  <img src="../figures/analysis/Ni_12000K_temperature.png" alt="Ni temperature vs time distribution, 12,000 K" width="48%">
</p>

**Figure 2.** Representative temperature histories for low- and high-temperature NVT simulations for both Cu and Ni. The thermostat maintains stable fluctuations around the requested target temperature throughout production.

### 3.2 Pressure and Total Energy

Pressure is computed from the virial expression,

```math
P=
\frac{2K+W}{3V},
```

where $W$ is the configurational virial. At fixed simulation volume, increasing temperature produces a substantial increase in mean pressure for both Ni and Cu. For example, the mean NVT pressure of Cu increases from approximately $1.67\times10^{-2}$ eV/Å$^3$ at 300 K to $6.98\times10^{-1}$ eV/Å$^3$ at 12,000 K. For Ni, the corresponding increase is from approximately $2.53\times10^{-2}$ to $8.15\times10^{-1}$ eV/Å$^3$.

This trend is physically consistent with a fixed-volume simulation: the simulation cell is not allowed to thermally expand, so increasingly energetic particles generate larger kinetic and virial contributions to the pressure. The high-temperature simulations should therefore be interpreted as **fixed-density model states**, not as ambient-pressure heating experiments.

The total energy similarly increases with temperature. In NVE production, total energy is the conserved quantity and is treated primarily as a numerical diagnostic in the Validation Report. In NVT production, energy fluctuates as the thermostat exchanges energy with the simulated system. For the purposes of the present report, the most important thermodynamic role of these quantities is to establish stable production conditions before analyzing structural and dynamical trends.

<p align="center">
  <img src="../figures/analysis/Cu_300K_energy.png" alt="Cu energy vs time, 300 K" width="32%">
  <img src="../figures/analysis/Cu_4000K_energy.png" alt="Cu energy vs time, 4000 K" width="32%">
  <img src="../figures/analysis/Cu_12000K_energy.png" alt="Cu energy vs time, 12,000 K" width="32%">
</p>

<p align="center">
  <img src="../figures/analysis/Ni_300K_energy.png" alt="Ni energy vs time, 300 K" width="32%">
  <img src="../figures/analysis/Ni_4000K_energy.png" alt="Ni energy vs time, 4000 K" width="32%">
  <img src="../figures/analysis/Ni_12000K_energy.png" alt="Ni energy vs time, 12,000 K" width="32%">
</p>

**Figure 3.** Energy histories for Cu and Ni at 300 K, 4000 K, and 12,000 K under NVE production. The energy remains statistically stable after equilibration, providing a consistent thermodynamic baseline for the subsequent structural and dynamical analyses.

### 3.3 Heat Capacity

The analysis suite also reports the energy-fluctuation quantity

```math
C_V=
\frac{\langle E^2\rangle-\langle E\rangle^2}
{k_BT^2}.
```

This expression is formally associated with canonical-ensemble energy fluctuations [1,2]. The values are included in the machine-generated analysis outputs, but they are **not used as a primary quantitative result in this report**. The Berendsen thermostat is designed for efficient temperature relaxation and does not generate the exact canonical fluctuation distribution [1,2,7]. Consequently, fluctuation-derived heat capacities from these NVT trajectories should be regarded as exploratory diagnostics rather than precision thermodynamic measurements.

This distinction is useful more generally: the analysis suite is capable of computing a broad set of observables, but the physical interpretation of each quantity must remain consistent with the ensemble and sampling method used to generate the trajectory.

---

## 4. Structural Analysis

### 4.1 Radial Distribution Function

The radial distribution function $g(r)$ measures the probability of finding a second atom at separation $r$ relative to the probability expected for an uncorrelated system of the same number density. In an isotropic representation,

```math
g(r)
=
\frac{1}{4\pi r^2\rho N}
\left\langle
\sum_{i\ne j}\delta(r-r_{ij})
\right\rangle.
```

For a crystalline FCC solid, $g(r)$ contains a sequence of narrow peaks associated with successive coordination shells. As thermal disorder increases, these peaks broaden. A liquid retains a strong nearest-neighbor peak but lacks the persistent sequence of narrow long-range coordination shells characteristic of a crystal [1,3,5].

At 300 K, all eight supported metals exhibit the expected FCC shell structure. The first RDF peak also agrees closely with the theoretical FCC nearest-neighbor distance,

```math
r_{\rm nn}=\frac{a}{\sqrt{2}},
```

where $a$ is the lattice constant.

| Metal | $a$ (Å) | $a/\sqrt{2}$ (Å) | RDF first peak (Å) | Relative difference |
|---|---:|---:|---:|---:|
| Ag | 4.090 | 2.892 | 2.876 | 0.56% |
| Al | 4.050 | 2.864 | 2.848 | 0.56% |
| Au | 4.080 | 2.885 | 2.869 | 0.56% |
| Cu | 3.615 | 2.556 | 2.542 | 0.56% |
| Ni | 3.520 | 2.489 | 2.475 | 0.56% |
| Pb | 4.950 | 3.500 | 3.480 | 0.56% |
| Pd | 3.890 | 2.751 | 2.735 | 0.56% |
| Pt | 3.920 | 2.772 | 2.756 | 0.56% |

The nearly uniform sub-percent difference is consistent with the finite RDF bin width rather than a metal-dependent structural discrepancy. The result demonstrates that the same analysis routine correctly identifies the nearest-neighbor shell across all eight FCC parameter sets.

<p align="center">
  <img src="../figures/analysis/Ag_300K_rdf.png" alt="Ag radial distribution function, 300 K" width="48%">
  <img src="../figures/analysis/Al_300K_rdf.png" alt="Al radial distribution function, 300 K" width="48%">
</p>

<p align="center">
  <img src="../figures/analysis/Au_300K_rdf.png" alt="Au radial distribution function, 300 K" width="48%">
  <img src="../figures/analysis/Cu_300K_rdf.png" alt="Cu radial distribution function, 300 K" width="48%">
</p>

<p align="center">
  <img src="../figures/analysis/Ni_300K_rdf.png" alt="Ni radial distribution function, 300 K" width="48%">
  <img src="../figures/analysis/Pb_300K_rdf.png" alt="Pb radial distribution function, 300 K" width="48%">
</p>

<p align="center">
  <img src="../figures/analysis/Pd_300K_rdf.png" alt="Pd radial distribution function, 300 K" width="48%">
  <img src="../figures/analysis/Pt_300K_rdf.png" alt="Pt radial distribution function, 300 K" width="48%">
</p>

**Figure 4.** Radial distribution functions for the eight supported FCC metals at 300 K using NVT simulation. Each curve exhibits a sharp first-neighbor peak followed by well-defined higher-order coordination shells characteristic of crystalline FCC order.

### 4.2 Coordination Number

The first-shell coordination number is obtained by integrating the RDF to its first minimum,

```math
CN=
4\pi\rho
\int_0^{r_{\min}}
g(r)r^2\,dr.
```

An ideal FCC lattice has twelve nearest neighbors. At 300 K, the NVT analysis gives

```math
CN = 12.000
```

for every supported metal to the displayed numerical precision. The corresponding NVE runs produce the same first-shell coordination number.

This result complements the first-peak location. The RDF peak position verifies the characteristic nearest-neighbor distance, while the coordination number verifies that the correct number of atoms populate the first shell. Together they show that the low-temperature simulations preserve the expected local FCC geometry across the complete set of metals.

At very high temperatures, the automatically integrated coordination number increases above 12 for Cu and Ni, reaching approximately 13.1 in the 12,000 K NVT simulations. This should not be interpreted as a distorted FCC coordination number. Once the crystalline shell structure has been lost, the first minimum of a liquid-like RDF defines a broader local coordination region, and its integral measures the average local liquid coordination rather than the fixed nearest-neighbor count of an ideal FCC crystal.

### 4.3 Static Structure Factor

The analysis suite computes an isotropically averaged static structure factor from the RDF using

```math
S(k)
=
1+
4\pi\rho
\int_0^\infty
r^2[g(r)-1]
\frac{\sin(kr)}{kr}\,dr.
```

This reciprocal-space representation contains information complementary to $g(r)$. Low-temperature FCC states display pronounced oscillatory peaks associated with long-range order. As the lattice becomes increasingly disordered, the higher-order oscillations weaken and the spectrum approaches a broad principal peak with damped secondary structure.

For Ni, the low-temperature $S(k)$ curves retain several pronounced maxima through 4000 K and still show recognizable multi-peak structure at 6000 K. By 8000 K, the higher-order structure has largely collapsed into broad oscillations around unity. Cu exhibits the same qualitative progression at a lower sampled temperature: the 4000 K spectrum remains structured, whereas the 6000 K spectrum is already dominated by a broad principal feature and substantially weaker long-range oscillation.

The structure-factor trends therefore agree with the real-space RDF and provide an independent reciprocal-space signature of the same structural transition.

<p align="center">
  <img src="../figures/analysis/Cu_300K_structure_factor.png" alt="Cu structure factor S(k), 300 K" width="48%">
  <img src="../figures/analysis/Cu_4000K_structure_factor.png" alt="Cu structure factor S(k), 4000 K" width="48%">
</p>

<p align="center">
  <img src="../figures/analysis/Cu_6000K_structure_factor.png" alt="Cu structure factor S(k), 6000 K" width="48%">
  <img src="../figures/analysis/Cu_4000K_structure_factor.png" alt="Cu structure factor S(k), 12000 K" width="48%">
</p>

<p align="center">
  <img src="../figures/analysis/Ni_300K_structure_factor.png" alt="Ni structure factor S(k), 300 K" width="48%">
  <img src="../figures/analysis/Ni_6000K_structure_factor.png" alt="Ni structure factor S(k), 6000 K" width="48%">
</p>

<p align="center">
  <img src="../figures/analysis/Ni_8000K_structure_factor.png" alt="Ni structure factor S(k), 8000 K" width="48%">
  <img src="../figures/analysis/Ni_12000K_structure_factor.png" alt="Ni structure factor S(k), 12,000 K" width="48%">
</p>

**Figure 5.** Static structure factor S(k) for Cu and Ni at different temperatures during NVT production. Increasing temperature suppresses the higher-order oscillatory structure associated with crystalline order and leaves a broader liquid-like principal feature.

---

## 5. Dynamical Analysis

### 5.1 Mean Squared Displacement

The mean squared displacement measures how far atoms move from their initial positions,

```math
MSD(t)
=
\frac{1}{N}
\sum_{i=1}^{N}
\left|
\mathbf r_i(t)-\mathbf r_i(0)
\right|^2.
```

Periodic displacements are evaluated using the minimum-image convention in the current implementation. For the 10 ps trajectories considered here, the MSD provides a particularly clear distinction between localized solid-state motion and sustained liquid-like diffusion.

At 300 K, the MSD rapidly reaches a small plateau for every metal rather than growing steadily with time. In the NVT runs, the final MSD ranges from approximately $0.012$ Å$^2$ for Pt to $0.071$ Å$^2$ for Pb. The exact plateau level depends on the metal-specific mass and interaction parameters, but the common qualitative behavior is localization: atoms fluctuate around lattice sites without long-range translational motion.

The high-temperature Cu and Ni data show a fundamentally different regime. For Cu, the 4000 K NVT trajectory remains bounded near $0.27$ Å$^2$ at 10 ps, whereas the 6000 K trajectory reaches approximately $41.8$ Å$^2$ and grows roughly linearly over much of the production interval. For Ni, the 6000 K NVT MSD remains bounded near $0.36$ Å$^2$, while the 8000 K trajectory reaches approximately $50.3$ Å$^2$.

The change is therefore not merely a gradual increase in vibrational amplitude. Between adjacent sampled temperatures, the long-time behavior changes by roughly two orders of magnitude and switches from a plateau to sustained growth. This is one of the clearest dynamical signatures of the solid-to-liquid change observed in the dataset.

The corresponding NVE trajectories show the same transition brackets: Cu remains localized at 4000 K and diffusive at 6000 K, while Ni remains localized at 6000 K and diffusive at 8000 K. Agreement between the NVE and NVT classifications is important because it indicates that the qualitative transition is not an artifact of the production thermostat.

<p align="center">
  <img src="../figures/analysis/Cu_300K_msd.png" alt="Cu mean squared displacement, 300 K" width="48%">
  <img src="../figures/analysis/Cu_2000K_msd.png" alt="Cu mean squared displacement, 2000 K" width="48%">
</p>

<p align="center">
  <img src="../figures/analysis/Cu_4000K_msd.png" alt="Cu mean squared displacement, 4000 K" width="48%">
  <img src="../figures/analysis/Cu_6000K_msd.png" alt="Cu mean squared displacement, 6000 K" width="48%">
</p>

<p align="center">
  <img src="../figures/analysis/Cu_8000K_msd.png" alt="Cu mean squared displacement, 8000 K" width="48%">
  <img src="../figures/analysis/Cu_12000K_msd.png" alt="Cu mean squared displacement, 12000 K" width="48%">
</p>

**Figure 6.** Cu MSD temperature sweep using NVT simulation. The 300 K–4000 K trajectories remain bounded, while the 6000 K and higher-temperature trajectories exhibit the sustained long-time growth associated with diffusion.

<p align="center">
  <img src="../figures/analysis/Ni_300K_msd.png" alt="Ni mean squared displacement, 300 K" width="48%">
  <img src="../figures/analysis/Ni_2000K_msd.png" alt="Ni mean squared displacement, 2000 K" width="48%">
</p>

<p align="center">
  <img src="../figures/analysis/Ni_4000K_msd.png" alt="Ni mean squared displacement, 4000 K" width="48%">
  <img src="../figures/analysis/Ni_6000K_msd.png" alt="Ni mean squared displacement, 6000 K" width="48%">
</p>

<p align="center">
  <img src="../figures/analysis/Ni_8000K_msd.png" alt="Ni mean squared displacement, 8000 K" width="48%">
  <img src="../figures/analysis/Ni_12000K_msd.png" alt="Ni mean squared displacement, 12000 K" width="48%">
</p>

**Figure 7.** Ni MSD temperature sweep using NVT simulation. The MSD remains bounded through 6000 K and becomes strongly diffusive at 8000 K and above.

### 5.2 Velocity Autocorrelation Function

The velocity autocorrelation function is computed as

```math
C_{\rm vv}(t)
=
\left\langle
\mathbf v(0)\cdot\mathbf v(t)
\right\rangle,
```

with averaging over particles and multiple time origins. The center-of-mass velocity is removed from each saved frame before the correlation is evaluated.

The VACF provides information about the persistence of atomic velocities and supplies an independent route to the self-diffusion coefficient through the Green–Kubo relation [8,9]. Across the present simulations, the correlation decays rapidly from its initial value and then fluctuates near zero at longer lag times. The initial VACF magnitude grows strongly with temperature because the characteristic squared particle speed increases with kinetic energy.

The saved analysis frames are separated by 50 fs. This cadence is sufficient for the broad transport behavior examined here, but it is relatively coarse for resolving detailed short-time vibrational oscillations. The VACF is therefore used primarily as a transport diagnostic rather than as a high-resolution vibrational spectrum.

<p align="center">
  <img src="../figures/analysis/Cu_300K_vacf.png" alt="Cu velocity autocorrelation function, 300 K" width="48%">
  <img src="../figures/analysis/Cu_4000K_vacf.png" alt="Cu velocity autocorrelation function, 4000 K" width="48%">
</p>

<p align="center">
  <img src="../figures/analysis/Cu_6000K_vacf.png" alt="Cu velocity autocorrelation function, 6000 K" width="48%">
  <img src="../figures/analysis/Cu_12000K_vacf.png" alt="Cu velocity autocorrelation function, 12,000 K" width="48%">
</p>

<p align="center">
  <img src="../figures/analysis/Ni_300K_vacf.png" alt="Ni velocity autocorrelation function, 300 K" width="48%">
  <img src="../figures/analysis/Ni_6000K_vacf.png" alt="Ni velocity autocorrelation function, 6000 K" width="48%">
</p>

<p align="center">
  <img src="../figures/analysis/Ni_8000K_vacf.png" alt="Ni velocity autocorrelation function, 8000 K" width="48%">
  <img src="../figures/analysis/Ni_12000K_vacf.png" alt="Ni velocity autocorrelation function, 12,000 K" width="48%">
</p>

**Figure 8.** Representative velocity autocorrelation functions for Cu and Ni at 300 K for NVT production. The correlation decays rapidly from its initial value, while its time integral provides an independent estimate of self-diffusion.

### 5.3 Diffusion Coefficients

The self-diffusion coefficient is estimated in two ways. From the long-time MSD,

```math
D_{\rm MSD}
=
\lim_{t\rightarrow\infty}
\frac{MSD(t)}{6t},
```

and from the VACF using the Green–Kubo relation,

```math
D_{\rm VACF}
=
\frac{1}{3}
\int_0^\infty
C_{\rm vv}(t)\,dt.
```

The implemented MSD estimator fits the final third of the saved MSD curve. For a localized solid, the true long-time diffusion coefficient should vanish. Consequently, very small positive or negative fitted slopes in low-temperature runs should be interpreted as regression noise around a plateau rather than as physical negative diffusion.

This behavior is visible throughout the solid-like portion of the Cu and Ni sweeps. The fitted $D_{\rm MSD}$ values remain near zero through 4000 K for Cu and through 6000 K for Ni. Once the MSD becomes strongly diffusive, the estimate jumps to the order of $10^{-4}$–$10^{-3}$ Å$^2$/fs. For example,

| System | $D_{\rm MSD}$ (Å$^2$/fs) | $D_{\rm VACF}$ (Å$^2$/fs) |
|---|---:|---:|
| Cu, NVT, 6000 K | $7.36\times10^{-4}$ | $1.26\times10^{-3}$ |
| Cu, NVT, 8000 K | $6.85\times10^{-4}$ | $2.40\times10^{-3}$ |
| Cu, NVT, 12000 K | $5.47\times10^{-4}$ | $3.45\times10^{-3}$ |
| Ni, NVT, 8000 K | $6.92\times10^{-4}$ | $2.30\times10^{-3}$ |
| Ni, NVT, 10000 K | $4.76\times10^{-4}$ | $3.43\times10^{-3}$ |
| Ni, NVT, 12000 K | $5.09\times10^{-4}$ | $4.91\times10^{-3}$ |

The two estimators identify the same qualitative change from negligible solid-state transport to finite liquid-like transport, but they do not coincide quantitatively over these short trajectories. This is not unexpected. The MSD fit depends on the selected diffusive time window, while the VACF integral is sensitive to short-time sampling and long-lag noise. Production thermostatting can also alter dynamical correlations in NVT trajectories. For quantitative transport calculations, longer NVE production runs, finer VACF sampling, explicit trajectory unwrapping, and convergence studies with respect to the fitting/integration window would be appropriate.

Accordingly, diffusion is used here primarily to confirm the change in dynamical regime rather than to claim high-precision transport coefficients.

---

## 6. Temperature-Dependent Structural Evolution and Melting-Like Behavior

The Cu and Ni temperature sweeps provide the most direct demonstration that the analysis suite can distinguish qualitatively different material states. The important observation is not a single numerical threshold, but the simultaneous change in several independent observables.

### 6.1 Copper

The NVT Cu results are summarized below.

| Target $T$ (K) | Mean $T$ (K) | Mean pressure (eV/Å$^3$) | MSD at 10 ps (Å$^2$) | CN | Qualitative regime |
|---:|---:|---:|---:|---:|---|
| 300 | 300.12 | 0.0167 | 0.019 | 12.000 | FCC solid |
| 1200 | 1199.72 | 0.0723 | 0.078 | 12.003 | FCC solid |
| 2000 | 2002.41 | 0.1185 | 0.126 | 12.010 | FCC solid |
| 4000 | 4002.79 | 0.2286 | 0.274 | 12.230 | Strongly thermally broadened, solid-like |
| 6000 | 5990.11 | 0.4350 | 41.84 | 12.901 | Liquid-like |
| 8000 | 8013.96 | 0.5305 | 56.30 | 12.550 | Liquid-like |
| 10000 | 10034.40 | 0.6159 | 64.60 | 12.805 | Liquid-like |
| 12000 | 12020.96 | 0.6984 | 75.45 | 13.116 | Liquid-like |

From 300 to 4000 K, the Cu RDF peaks progressively broaden but remain identifiable over multiple coordination shells. The MSD also remains bounded, increasing from approximately $0.02$ Å$^2$ at 300 K to approximately $0.27$ Å$^2$ at 4000 K. This is consistent with increasingly large thermal vibration within a still-localized solid.

The 6000 K result is qualitatively different. The RDF no longer displays the sequence of narrow FCC shells; instead, it consists of a broad first-neighbor peak followed by damped oscillations characteristic of short-range order in a dense disordered phase. The corresponding structure factor loses most of its higher-order crystalline oscillation. Most decisively, the MSD grows throughout the production run and reaches approximately $41.8$ Å$^2$ after 10 ps.

The 8000 K–12,000 K trajectories retain this liquid-like behavior, with final MSD values between approximately $56$ and $75$ Å$^2$ and broad, strongly damped structural correlations. The NVE production runs exhibit the same qualitative separation between 4000 K and 6000 K.

Thus, within the sampled temperatures and present model, the Cu simulations bracket a solid-to-liquid transition between **4000 K and 6000 K**.

<p align="center">
  <img src="../figures/analysis/Cu_300K_rdf.png" alt="Cu radial distribution function, 300 K" width="32%">
  <img src="../figures/analysis/Cu_2000K_rdf.png" alt="Cu radial distribution function, 2000 K" width="32%">
  <img src="../figures/analysis/Cu_4000K_rdf.png" alt="Cu radial distribution function, 4000 K" width="32%">
</p>

<p align="center">
  <img src="../figures/analysis/Cu_6000K_rdf.png" alt="Cu radial distribution function, 6000 K" width="32%">
  <img src="../figures/analysis/Cu_8000K_rdf.png" alt="Cu radial distribution function, 8000 K" width="32%">
  <img src="../figures/analysis/Cu_12000K_rdf.png" alt="Cu radial distribution function, 12000 K" width="32%">
</p>

**Figure 9.** Cu RDF temperature sweep for NVT production. Crystalline coordination shells broaden with temperature and are replaced by a broad nearest-neighbor peak with damped long-range oscillations between 4000 K and 6000 K.

### 6.2 Nickel

The NVT Ni results are summarized below.

| Target $T$ (K) | Mean $T$ (K) | Mean pressure (eV/Å$^3$) | MSD at 10 ps (Å$^2$) | CN | Qualitative regime |
|---:|---:|---:|---:|---:|---|
| 300 | 300.02 | 0.0253 | 0.014 | 12.000 | FCC solid |
| 1200 | 1202.18 | 0.0862 | 0.057 | 12.001 | FCC solid |
| 2000 | 2000.38 | 0.1372 | 0.097 | 12.008 | FCC solid |
| 4000 | 3993.86 | 0.2570 | 0.195 | 12.090 | FCC solid |
| 6000 | 6009.14 | 0.3763 | 0.355 | 12.616 | Strongly thermally broadened, solid-like |
| 8000 | 7991.05 | 0.6196 | 50.28 | 12.600 | Liquid-like |
| 10000 | 10003.90 | 0.7217 | 60.51 | 12.840 | Liquid-like |
| 12000 | 12009.61 | 0.8150 | 63.86 | 13.141 | Liquid-like |

Ni remains structurally ordered over a somewhat broader temperature range than Cu in the present parameterization. The RDF at 6000 K is strongly broadened, but several higher-order features remain visible, and the MSD remains bounded near $0.36$ Å$^2$. The system is therefore highly thermally disordered but remains dynamically localized over the 10 ps observation window.

At 8000 K, the behavior changes sharply. The RDF has the broad first-neighbor maximum and weak long-range oscillation expected for a liquid-like state, the structure factor loses its multi-peak crystalline character, and the MSD grows to approximately $50.3$ Å$^2$. The 10,000 K and 12,000 K simulations remain in the same qualitative regime.

The NVE runs again reproduce the same transition bracket: the 6000 K production trajectory remains localized, whereas the 8000 K trajectory is strongly diffusive. Within the sampled temperatures, the Ni simulations therefore bracket the solid-to-liquid transition between **6000 K and 8000 K**.

<p align="center">
  <img src="../figures/analysis/Ni_300K_rdf.png" alt="Ni radial distribution function, 300 K" width="32%">
  <img src="../figures/analysis/Ni_2000K_rdf.png" alt="Ni radial distribution function, 2000 K" width="32%">
  <img src="../figures/analysis/Ni_4000K_rdf.png" alt="Ni radial distribution function, 4000 K" width="32%">
</p>

<p align="center">
  <img src="../figures/analysis/Ni_6000K_rdf.png" alt="Ni radial distribution function, 6000 K" width="32%">
  <img src="../figures/analysis/Ni_8000K_rdf.png" alt="Ni radial distribution function, 8000 K" width="32%">
  <img src="../figures/analysis/Ni_12000K_rdf.png" alt="Ni radial distribution function, 12000 K" width="32%">
</p>

**Figure 10.** Ni RDF NVT temperature sweep. The higher-order FCC shell structure persists through 6000 K but largely disappears by 8000 K, consistent with the simultaneous onset of diffusive MSD growth.

### 6.3 Combined Structural and Dynamical Interpretation

The strongest evidence for melting-like behavior is the agreement among independent observables.

For Cu:

- 4000 K retains several recognizable RDF coordination features;
- the 4000 K structure factor retains substantial higher-order structure;
- the 4000 K MSD remains bounded at approximately $0.27$ Å$^2$;
- at 6000 K, the RDF becomes liquid-like;
- the structure factor becomes broadly damped;
- the MSD increases to more than $40$ Å$^2$ and grows throughout the trajectory;
- finite self-diffusion estimates emerge.

<p align="center">
  <img src="../figures/analysis/Cu_4000K_rdf.png" alt="Cu radial distribution function, 4000 K" width="48%">
  <img src="../figures/analysis/Cu_6000K_rdf.png" alt="Cu radial distribution function, 6000 K" width="48%">
</p>

<p align="center">
  <img src="../figures/analysis/Cu_4000K_msd.png" alt="Cu mean squared displacement, 4000 K" width="48%">
  <img src="../figures/analysis/Cu_6000K_msd.png" alt="Cu mean squared displacement, 6000 K" width="48%">
</p>

**Figure 11.** Summary of the observed solid-to-liquid change for Cu. Notice that the transition is bracketed by 4000 K and 6000 K. Loss of higher-order RDF structure coincides with the onset of sustained MSD growth.

For Ni, the same sequence occurs between 6000 K and 8000 K.

The abrupt change in the MSD is particularly useful when interpreted alongside the RDF. Broadening of an RDF can arise simply from large thermal vibrations in a hot solid; it does not by itself prove melting. Conversely, a large MSD establishes atomic mobility but does not describe the accompanying loss of spatial order. Here, the structural and dynamical changes occur together. The system loses long-range FCC coordination at the same sampled temperature interval in which atoms become translationally mobile.

<p align="center">
  <img src="../figures/analysis/Ni_6000K_rdf.png" alt="Ni radial distribution function, 6000 K" width="48%">
  <img src="../figures/analysis/Ni_8000K_rdf.png" alt="Ni radial distribution function, 8000 K" width="48%">
</p>

<p align="center">
  <img src="../figures/analysis/Ni_6000K_msd.png" alt="Ni mean squared displacement, 6000 K" width="48%">
  <img src="../figures/analysis/Ni_8000K_msd.png" alt="Ni mean squared displacement, 8000 K" width="48%">
</p>

**Figure 12.** Summary of the observed solid-to-liquid change for Ni. Here, the transition occurs between 6000 K and 8000 K.



---

## 7. Discussion

The analysis results demonstrate that the simulator supports more than trajectory generation. The implemented observables provide mutually complementary views of equilibrium structure and dynamics and are capable of distinguishing qualitatively different physical regimes.

At 300 K, the analysis is highly consistent across the complete set of supported metals. All eight systems retain the expected FCC nearest-neighbor geometry, each first RDF peak lies within approximately 0.56% of $a/\sqrt{2}$, and every first-shell coordination number is 12.000. The MSD remains bounded for every metal, providing dynamical confirmation that the atoms remain localized about lattice sites. These observations extend the representative Ni structural checks in the Validation Report to the broader analysis dataset.

The Cu and Ni temperature sweeps provide a stronger test of physical interpretation because the system is driven far away from the low-temperature crystalline reference state. Rather than simply broadening continuously, the observables separate into two regimes. Below the transition bracket, the systems exhibit increasingly large but bounded thermal motion and retain recognizable higher-order structure. Above the bracket, the RDF and structure factor lose their crystalline signatures while the MSD becomes strongly time-dependent and diffusion estimates become finite. Both NVT and NVE production runs reproduce the same qualitative transition intervals.

These results are best viewed as **model-dependent melting behavior rather than melting-point predictions**. Several aspects of the present simulations shift the conditions under which a defect-free crystal loses stability:

1. **Pairwise Lennard–Jones interactions.** Real metallic bonding is many-body in character. The Lennard–Jones parameterization used here is useful for a compact classical model but is not expected to reproduce all finite-temperature thermodynamic properties of a realistic metal [6].

2. **Fixed simulation volume.** The simulations do not permit thermal expansion or pressure relaxation. The rapidly increasing pressure observed during the temperature sweeps reflects this constraint.

3. **Finite, periodic, defect-free crystals.** A perfect periodic lattice lacks surfaces, grain boundaries, and many heterogeneous nucleation sites that facilitate melting in real materials. Such systems can remain metastably ordered above a bulk melting temperature.

4. **Finite simulation duration.** Each production run spans 10 ps. A longer trajectory could reveal slower structural rearrangements or improve transport statistics near a transition.

5. **Thermostat choice.** The Berendsen thermostat is appropriate for equilibration and temperature control but does not rigorously generate canonical fluctuations and can modify dynamical correlations [1,2,7]. NVE production results are therefore especially useful when interpreting transport behavior.

6. **Transport sampling.** The 50 fs saved-frame interval is relatively coarse for detailed VACF analysis, and the present MSD calculation uses periodic minimum-image displacement relative to the initial configuration. Longer quantitative diffusion studies would benefit from explicitly unwrapped trajectories and finer temporal sampling.

These limitations do not weaken the primary purpose of the analysis suite as a scientific-computing demonstration. In fact, distinguishing between quantities that are reliable qualitative diagnostics and quantities that require more specialized sampling is an important part of molecular-dynamics analysis. The present calculations convincingly demonstrate structural ordering, thermal disordering, localization, diffusion, and model-dependent solid-to-liquid behavior while identifying clear directions for more quantitative future studies.

---

## 8. Conclusions

The FCC Molecular Dynamics Simulator includes an integrated analysis framework capable of extracting thermodynamic, structural, and dynamical information from atomistic trajectories.

The principal results of the analysis dataset are:

- NVT simulations remain tightly centered around their requested target temperatures over the complete temperature range studied.
- At 300 K, all eight supported metals reproduce the expected FCC nearest-neighbor structure.
- The first RDF peak agrees with the theoretical distance $a/\sqrt{2}$ to within approximately 0.56% for Ag, Al, Au, Cu, Ni, Pb, Pd, and Pt.
- The first-shell coordination number is 12.000 for every supported metal at 300 K.
- Low-temperature MSD curves remain bounded, demonstrating localization of atoms about crystalline lattice sites.
- Increasing temperature broadens the RDF and suppresses higher-order structure-factor oscillations.
- Cu changes from a strongly thermally broadened but localized state at 4000 K to a diffusive, liquid-like state at 6000 K.
- Ni remains localized at 6000 K but becomes strongly diffusive and liquid-like at 8000 K.
- The same Cu and Ni transition brackets are observed in both NVT and NVE production trajectories.
- MSD- and VACF-based diffusion estimates provide complementary evidence for finite transport in the high-temperature liquid-like regime.

Taken together, the RDF, coordination number, structure factor, MSD, VACF, and diffusion analyses provide a coherent description of both equilibrium FCC solids and high-temperature disordered states. The temperature sweeps demonstrate that the simulator reproduces the qualitative structural and dynamical signatures expected during a solid-to-liquid transition, while the ensemble and model limitations make clear why these results should be interpreted as behavior of the present Lennard–Jones model rather than as experimental melting-point predictions.

Combined with the separate numerical Validation Report, the analysis results show that the project provides both a verified molecular-dynamics implementation and a reproducible workflow for extracting physically meaningful information from simulated atomic trajectories.

---

## References

[1] M. P. Allen and D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed., Oxford University Press, Oxford, 2017.

[2] D. Frenkel and B. Smit, *Understanding Molecular Simulation: From Algorithms to Applications*, 2nd ed., Academic Press, San Diego, 2002.

[3] D. C. Rapaport, *The Art of Molecular Dynamics Simulation*, 2nd ed., Cambridge University Press, Cambridge, 2004.

[4] J. M. Haile, *Molecular Dynamics Simulation: Elementary Methods*, Wiley, New York, 1992.

[5] J.-P. Hansen and I. R. McDonald, *Theory of Simple Liquids*, 4th ed., Academic Press, Oxford, 2013.

[6] H. Heinz, R. A. Vaia, B. L. Farmer, and R. R. Naik, Accurate simulation of surfaces and interfaces of face-centered cubic metals using 12–6 and 9–6 Lennard-Jones potentials, *J. Phys. Chem. C* **112** (2008), no. 44, 17281–17290.

[7] H. J. C. Berendsen, J. P. M. Postma, W. F. van Gunsteren, A. DiNola, and J. R. Haak, Molecular dynamics with coupling to an external bath, *J. Chem. Phys.* **81** (1984), no. 8, 3684–3690.

[8] M. S. Green, Markoff random processes and the statistical mechanics of time-dependent phenomena. II. Irreversible processes in fluids, *J. Chem. Phys.* **22** (1954), 398–413.

[9] R. Kubo, Statistical-mechanical theory of irreversible processes. I. General theory and simple applications to magnetic and conduction problems, *J. Phys. Soc. Japan* **12** (1957), 570–586.
