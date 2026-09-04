# PySCF Resource for DragonBall

A small collection of PySCF-based utilities for preparing and running *ab initio* molecular dynamics calculations compatible with **DragonBall**.

The resource provides three main programs:

- `optimizer.py` — geometry optimization, harmonic-frequency analysis, and equilibrium Hessian generation.
- `runner.py` — Born–Oppenheimer molecular dynamics (BOMD), including adiabatic switching from a harmonic reference potential to the *ab initio* potential.
- `hessianator.py` — batch evaluation of Cartesian Hessians for a sequence of molecular geometries.

All three programs are configured through a common `config.py` file.

## Requirements

The scripts are written in Python and use:

- PySCF
- NumPy
- geomeTRIC, through the PySCF geometry-optimization interface

A typical environment therefore requires at least:

```bash
pip install pyscf numpy geometric
```

The electronic-structure calculations in the supplied scripts use density-fitted DFT.

## Files

```text
config.py
optimizer.py
runner.py
hessianator.py
```

### `config.py`

Central configuration file shared by the three programs. It defines file names, electronic-structure settings, optimization thresholds, dynamics parameters, switching options, and Hessianator settings.

### `optimizer.py`

Optimizes an input geometry with geomeTRIC, computes harmonic frequencies at the optimized structure, and writes the Cartesian Hessian in the lower-triangular format expected by DragonBall.

### `runner.py`

Runs a PySCF Born–Oppenheimer trajectory using velocity-Verlet integration. It can either propagate directly on the *ab initio* potential or gradually switch from a harmonic approximation to the full electronic-structure potential.

### `hessianator.py`

Reads a multi-frame XYZ file, computes a PySCF Hessian for every geometry, and appends the Hessians to a single database-style output file.

## Configuration

All user-adjustable parameters are collected in `config.py`.

The supplied configuration is:

```python
# --- INPUT FILES
xyzguess = "guess.xyz"              # Optimizer guess geometry
xyzfile = "geometry.xyz"            # Optimizer output / runner input geometry
velfile = "velocity.xyz"            # Initial velocity
eqxyz = "geometry.xyz"              # Equilibrium geometry
cnormfile = "none"                  # Hessian eigenproblem file
hessfile = "Hessian_flat.out"       # Hessian file

# --- OUTPUT FILES
traj_file = "traj.xyz"              # Trajectory
force_file = "forces.dat"           # Forces
md_file = "md.log"                  # Energies
final_geo = "final_geo_bomd.xyz"    # Final geometry
final_vel = "final_vel_bomd.xyz"    # Final velocity

# --- LEVEL OF THEORY
functional = "b3lyp"
dispersion = "none"
basis = "def2-SVP"
charge = 0
spin = 0
conv_tol = 1e-10
backend = "pyscf"

# --- OPTIMIZATION PARAMETERS (geomeTRIC)
opt_energy = 1e-6
opt_grms = 3e-4
opt_gmax = 4.5e-4
opt_drms = 1.2e-3
opt_dmax = 1.8e-3

# --- DYNAMICS PARAMETERS
dt_fs = 0.2
nsteps = 2500
switching_steps = nsteps
NROTRANSL = 5

# --- SWITCHING OPTIONS
switching_fun = "smooth"
harmonic_gen = "hessian"

# --- HESSIANATOR INPUT
geomfile = "geom_test_hessian.xyz"
hdbfile = "hdb.out"
```

`spin` follows the PySCF convention, i.e. the number of alpha electrons minus the number of beta electrons (`2S`).

Setting `dispersion = "none"` disables an explicit dispersion correction. Otherwise the value is assigned to the PySCF `disp` setting.

---

# 1. Harmonic Optimizer

Run:

```bash
python optimizer.py
```

The optimizer performs the following sequence:

1. Reads the initial structure from `xyzguess`.
2. Constructs a PySCF DFT calculation using the level of theory specified in `config.py`.
3. Optimizes the geometry with **geomeTRIC**.
4. Writes the optimized structure to `xyzfile`.
5. Computes the analytic Hessian at the optimized geometry.
6. Performs PySCF harmonic analysis and prints the harmonic frequencies in cm\(^{-1}\).
7. Writes the Cartesian Hessian to `hessfile`.

Closed-shell calculations (`spin = 0`) use density-fitted RKS; open-shell calculations use density-fitted UKS.

## Optimizer input

`xyzguess` is a standard XYZ geometry, for example:

```text
3
H2O
O   0.000000   0.000000   0.000000
H   0.000000   0.757000   0.586000
H   0.000000  -0.757000   0.586000
```

## Optimizer outputs

With the default configuration:

```text
geometry.xyz
Hessian_flat.out
```

`geometry.xyz` contains the optimized Cartesian geometry in Angstrom.

`Hessian_flat.out` contains the lower triangle of the Cartesian Hessian. The file starts with two blank lines, followed by one matrix element per line in Fortran-style `D` exponential notation:

```text
H(1,1)
H(2,1)
H(2,2)
H(3,1)
H(3,2)
H(3,3)
...
```

This representation is intended to be directly compatible with DragonBall.

---

# 2. BOMD Runner

Run:

```bash
python runner.py
```

The runner performs Born–Oppenheimer molecular dynamics using a **velocity-Verlet** integrator.

The initial geometry is read from `xyzfile`, while `eqxyz` defines the equilibrium geometry used to construct the harmonic reference potential. Initial velocities are read from `velfile`.

The time step is specified in femtoseconds through

```python
dt_fs = 0.2
```

and converted internally to atomic units.

## Initial velocity file

The velocity file is expected to have an XYZ-like two-line header followed by the atomic symbols and three velocity components:

```text
3
Initial velocities
O   vx   vy   vz
H   vx   vy   vz
H   vx   vy   vz
```

The velocity components are read directly as atomic units.

## Electronic-structure forces

At each step, the runner evaluates the PySCF energy and nuclear gradient through a gradient scanner. The current implementation supports

```python
backend = "pyscf"
```

and uses density-fitted RKS or UKS DFT according to `spin`.

## Adiabatic switching

The runner can interpolate between a harmonic reference potential and the full *ab initio* potential.

If $\lambda$ is the switching function, the force is

$$
F = (1-\lambda)F_\mathrm{harm} + \lambda F_\mathrm{real},
$$

and the potential energy reported by the trajectory is

$$
V = (1-\lambda)V_\mathrm{harm} + \lambda\left(E_\mathrm{real}-E_0\right),
$$

where $E_0$ is the *ab initio* energy at the equilibrium geometry.

The switching interval is controlled by:

```python
switching_steps = nsteps
```

### Switching functions

The following values of `switching_fun` are implemented:

| Value | Behaviour |
|---|---|
| `"none"` | No switching; use the *ab initio* potential |
| `"linear"` | Linear interpolation |
| `"sine"` | Smooth sine-based switching |
| `"smooth"` | Fifth-order smoothstep switching |

The `"smooth"` option uses


$$\lambda(x)=10x^3-15x^4+6x^5,  \qquad x=\frac{t}{T}.$$

## Harmonic reference

Two representations of the harmonic reference are supported.

### Cartesian Hessian

```python
harmonic_gen = "hessian"
```

The runner reads `hessfile` and evaluates

$$V_\mathrm{harm} = \frac{1}{2}\Delta x^\mathrm{T}H\Delta x,$$


with force


$$F_\mathrm{harm}=-H\Delta x.$$


The Hessian is read from the same flattened lower-triangular representation written by `optimizer.py`.

### Normal modes

```python
harmonic_gen = "cnorm"
```

The runner instead reads `cnormfile`, containing the scaled-Hessian eigenvectors and eigenvalues. The last `NROTRANSL` squared frequencies are set to zero before evaluating the harmonic potential and force.

For a nonlinear molecule normally use:

```python
NROTRANSL = 6
```

whereas a linear molecule has five rotational/translational modes:

```python
NROTRANSL = 5
```

## Runner outputs

The default files are:

```text
traj.xyz
forces.dat
md.log
final_geo_bomd.xyz
final_vel_bomd.xyz
```

### `traj.xyz`

Multi-frame XYZ trajectory. Each frame contains the Cartesian coordinates in Angstrom together with the current velocities. The comment line reports the step, time, total energy, kinetic energy, and potential energy.

### `forces.dat`

Stores the force vector for each atom at every trajectory step.

### `md.log`

Energy log containing:

```text
Step    Time    Epot (au)    Ekin (au)    Etot (au)    Etot (cm-1)
```

Energies are reported in Hartree and the total energy is additionally converted to cm\(^{-1}\).

### Final state

`final_geo_bomd.xyz` stores the final Cartesian geometry, while `final_vel_bomd.xyz` stores the final velocities in atomic units.

---

# 3. Hessianator

Run:

```bash
python hessianator.py
```

Hessianator computes Cartesian Hessians for a collection of geometries.

The input is controlled by:

```python
geomfile = "geom_test_hessian.xyz"
hdbfile = "hdb.out"
```

`geomfile` must be a multi-frame XYZ file:

```text
3
Geometry 1
O   ...
H   ...
H   ...
3
Geometry 2
O   ...
H   ...
H   ...
...
```

For the first geometry, Hessianator builds the PySCF molecule and DFT object. For subsequent geometries it updates the molecular coordinates and resets the existing mean-field object rather than reconstructing the entire calculation from scratch.

The atomic symbols and their ordering must remain identical in every frame. The program raises an error if this changes.

For each geometry it:

1. runs the SCF calculation;
2. computes the analytic PySCF Hessian;
3. converts the four-index PySCF Hessian to a \(3N\times3N\) Cartesian matrix;
4. writes its lower triangle to the output file.

The output file is recreated when the program starts. Hessian blocks are appended sequentially, with two blank lines preceding each Hessian. The numerical values are written consecutively without geometry labels, in `D` exponential notation, to retain compatibility with the DragonBall Hessian format.

---

# Typical Workflow

A complete workflow can be:

```text
             guess.xyz
                 |
                 v
          +---------------+
          | optimizer.py  |
          +---------------+
             |         |
             v         v
       geometry.xyz  Hessian_flat.out
             |         |
             +----+----+
                  |
           velocity.xyz
                  |
                  v
           +-------------+
           |  runner.py  |
           +-------------+
                  |
        +---------+---------+
        |         |         |
        v         v         v
     traj.xyz  forces.dat  md.log
        |
        | selected geometries
        v
 geom_test_hessian.xyz
        |
        v
 +----------------+
 | hessianator.py |
 +----------------+
        |
        v
      hdb.out
```

A common use is therefore:

```bash
python optimizer.py
python runner.py
python hessianator.py
```

The individual utilities can also be used separately when the required input files are already available.

The initial velocities can be generated with DragonBall's Vegeta utility. The list of geometries for Hessianator can be obtained using an Hessian Database.

## Units

The scripts use PySCF/atomic units internally. The principal external units are:

| Quantity | Unit |
|---|---|
| Input/output Cartesian geometries | Angstrom |
| Dynamics time step (`dt_fs`) | fs |
| Internal trajectory time | atomic units |
| Initial/final velocities | atomic units |
| Electronic and trajectory energies | Hartree |
| Harmonic frequencies | cm\(^{-1}\) |
| Cartesian Hessian | Hartree/Bohr² |

## Notes

- `optimizer.py`, `runner.py`, and `hessianator.py` all import their settings from `config.py`; keep the configuration file in the same working directory.
- The scripts assume PySCF-compatible functional, basis-set, and dispersion identifiers.
- `optimizer.py` and `hessianator.py` write Hessians in the flattened lower-triangular format used by DragonBall.
- Hessianator assumes the same atoms in the same order for every geometry.
- The runner currently implements only the `pyscf` electronic-structure backend.
