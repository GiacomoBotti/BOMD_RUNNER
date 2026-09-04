import re
import numpy as np
import config as inp
from pyscf import gto, scf, dft
import time

t0 = time.perf_counter()

AMU2AU = 1822.888486209
ANG2BOHR = 1.8897261246257702
BOHR2ANG = 1.0 / ANG2BOHR
FS2AU = 41.3413745758
AU2FS = 1.0 / FS2AU
CM1_TO_AU = 1.0 / 219474.6313705  # cm^-1 -> atomic units
AU_TO_CM1 = 219474.6313705

#----------------------------------------
# INPUT READING FROM INPUT.PY
#----------------------------------------
# --- INPUT FILES
eqxyz     = inp.eqxyz
xyzfile   = inp.xyzfile
velfile   = inp.velfile
cnormfile = inp.cnormfile
hessfile  = inp.hessfile

# --- OUTPUT FILES
traj_file  = inp.traj_file
force_file = inp.force_file
md_file    = inp.md_file
final_geo  = inp.final_geo
final_vel  = inp.final_vel

# --- LEVEL OF THEORY
functional = inp.functional
dispersion = inp.dispersion
basis      = inp.basis
charge     = inp.charge
spin       = inp.spin
conv_tol   = inp.conv_tol
backend    = inp.backend

# --- DYNAMICS PARAMETERS
dt_fs           = inp.dt_fs
nsteps          = inp.nsteps
switching_steps = inp.switching_steps
NROTRANSL        = inp.NROTRANSL

# --- SWITCHING OPTIONS
switching_fun   = inp.switching_fun
harmonic_gen    = inp.harmonic_gen

def read_flat_lower_hessian(filename):
    vals = []
    with open(filename, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            vals.append(float(s.replace("D", "E")))

    vals = np.asarray(vals, dtype=float)

    n = int((np.sqrt(8 * len(vals) + 1) - 1) / 2)
    if n * (n + 1) // 2 != len(vals):
        raise ValueError("File length is not a valid lower-triangular matrix.")

    H = np.zeros((n, n), dtype=float)
    tril = np.tril_indices(n)
    H[tril] = vals
    H = H + np.tril(H, -1).T
    return H

def read_cnorm(filename):
    """
    Read VEGETA cnorm.dat.

    Returns
    -------
    cnorm : ndarray
        Scaled Hessian eigenvectors.
    omega2_au : ndarray
        Squared frequencies (eigenvalues of the scaled Hessian) in a.u.
    """

    data = np.loadtxt(filename, comments="#")

    # In this file format:
    # first N rows  -> cnorm matrix (N x N)
    # last row      -> eigenvalues of scaled Hessian
    nmode = data.shape[1]
    cnorm = data[:nmode, :]
    omega2_au = data[-1, :]

    # Print frequencies in cm^-1
    # Use signed sqrt so imaginary/negative modes stay visible if present.
    omega_au = np.sign(omega2_au) * np.sqrt(np.abs(omega2_au))
    omega_cm1 = omega_au * AU_TO_CM1

    print("\nFrequencies (cm^-1)")
    for i, w in enumerate(omega_cm1, 1):
        print(f"{i:4d} {w:12.2f}")

    return cnorm, omega2_au


def harmonic_energy_cnorm(coords, coords0, harmonic):
    """
    Harmonic energy from normal modes.

    coords, coords0 : (natm, 3) in Bohr
    masses_au       : (natm,) or (natm,1) in electron masses
    cnorm           : normal-mode matrix
    omega2          : squared frequencies in a.u.
    """
    cnorm = harmonic["cnorm"]
    omega2 = harmonic["omega2"]
    masses_au = harmonic["masses_au"]

    dx = (np.asarray(coords, dtype=float) - np.asarray(coords0, dtype=float)).ravel()

    masses_au = np.asarray(masses_au, dtype=float).reshape(-1)
    sqrtm = np.repeat(np.sqrt(masses_au), 3)

    dx_mw = sqrtm * dx
    q = cnorm @ dx_mw

    return 0.5 * np.sum(omega2 * q * q)

def harmonic_force_cnorm(coords, coords0, harmonic):
    """
    coords, coords0 : (natm, 3) in Bohr
    masses_au       : (natm,) or (natm,1)
    cnorm           : full normal-mode matrix
    omega2          : squared frequencies in a.u., last NROTRASL entries already zero
    """

    cnorm = harmonic["cnorm"]
    omega2 = harmonic["omega2"]
    masses_au = harmonic["masses_au"]

    coords = np.asarray(coords, dtype=float)
    coords0 = np.asarray(coords0, dtype=float)
    masses_au = np.asarray(masses_au, dtype=float).reshape(-1)

    x = (coords - coords0).ravel()
    sqrtm = np.repeat(np.sqrt(masses_au), 3)
    xmw = sqrtm * x

    # Try this convention first:
    q = cnorm @ xmw
    fq = -omega2 * q
    fmw = cnorm.T @ fq
    f = (sqrtm * fmw).reshape(coords.shape)

    return f

def harmonic_energy_hessian(coords, coords0, harmonic):
    """
    Harmonic energy from the Cartesian Hessian.

    coords, coords0 : (natm, 3) in Bohr
    hessian         : (3N, 3N) Cartesian Hessian in Hartree/Bohr^2
    """
    H = harmonic["H"]
 
    dx = (np.asarray(coords, dtype=float) - np.asarray(coords0, dtype=float)).ravel()
    return 0.5 * dx @ H @ dx

def harmonic_force_hessian(coords, coords0, harmonic):
    """
    coords, coords0 : (natm, 3) in Bohr
    Hcart           : (3N, 3N) Cartesian Hessian in Hartree/Bohr^2
    """
 
    H = harmonic["H"]
 
    coords = np.asarray(coords, dtype=float)
    coords0 = np.asarray(coords0, dtype=float)

    dx      = (coords - coords0).ravel()
    F_cart    = (-H @ dx).reshape(coords.shape)

    return F_cart

def switching_harmonic(step, nsteps):
    return 0.0

def switching_none(step, nsteps):
    return 1.0

def switching_linear(step, nsteps):
    return step / nsteps

def switching_sine(t, T):
    x = t / T
    return x - np.sin(2.0 * np.pi * x) / (2.0 * np.pi)

def switching_smoothstep(t, T):
    x = t / T
    return 10.0*x**3 - 15.0*x**4 + 6.0*x**5 

def write_output(traj, forces, md, step, time, mol, vel, frc, epot, ekin):
    """
    Write one MD frame to three open files:
      - traj.xyz
      - forces.dat
      - md.log

    Coordinates are written in Angstrom.
    Velocities and forces are written as they are passed in.
    Energies are written in Hartree.
    """

    natm = mol.natm
    coords = mol.atom_coords(unit="Ang")
    symbols = [mol.atom_symbol(i) for i in range(natm)]

    # --- trajectory ---
    traj.write(f"{natm}\n")
    traj.write(
        f"Step = {step:6d}  Time = {time:16.8f}  "
        f"Total energy = {epot + ekin:20.12f}  "
        f"Kinetic energy = {ekin:20.12f}  "
        f"Potential energy = {epot:20.12f}\n"
    )
    for sym, r, v in zip(symbols, coords, vel):
        traj.write(
            f"{sym:2s}"
            f"{r[0]:16.8f}{r[1]:16.8f}{r[2]:16.8f}"
            f"{v[0]:16.8f}{v[1]:16.8f}{v[2]:16.8f}\n"
        )

    # --- forces ---
    forces.write(f"{natm}\n")
    forces.write(f"Step = {step:6d}  Time = {time:16.8f}\n")
    for sym, f in zip(symbols, frc):
        forces.write(
            f"{sym:2s}"
            f"{f[0]:16.8f}{f[1]:16.8f}{f[2]:16.8f}\n"
        )

    # --- md log ---
    md.write(
        f"{step:8d}"
        f"{time:16.8f}"
        f"{epot:20.12e}"
        f"{ekin:20.12e}"
        f"{(epot + ekin):20.12e}"
        f"{(epot + ekin)*AU_TO_CM1:20.12e}\n"
    )

    traj.flush()
    forces.flush()
    md.flush()

def write_final_files(final_geo, final_vel, mol, vel):
    natm = mol.natm
    symbols = [mol.atom_symbol(i) for i in range(natm)]
    coords = mol.atom_coords(unit="Ang")

    with open(final_geo, "w") as fg:
        fg.write(f"{natm}\n")
        fg.write("Final geometry\n")
        for sym, r in zip(symbols, coords):
            fg.write(f"{sym:2s} {r[0]:16.10f} {r[1]:16.10f} {r[2]:16.10f}\n")

    with open(final_vel, "w") as fv:
        fv.write(f"{natm}\n")
        fv.write("Final velocities (a.u.)\n")
        for sym, v in zip(symbols, vel):
            fv.write(f"{sym:2s} {v[0]:16.10e} {v[1]:16.10e} {v[2]:16.10e}\n")

def run_bomd():
    # -----------------------
    # Input
    # -----------------------
    dt = dt_fs*FS2AU


    # -----------------------
    # Molecule
    # -----------------------
    mol = gto.Mole()
    #mol.atom = xyzfile
    mol.atom = eqxyz #equilibrium geometry
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()

    coords = mol.atom_coords()  # Bohr
    coords0 = coords # Equilibrium geometry
    vel = np.loadtxt(velfile, skiprows=2, usecols=(1, 2, 3))

    if vel.shape != coords.shape:
        raise ValueError(
            f"Velocity file shape {vel.shape} does not match number of atoms {coords.shape[0]}"
        )

    mass = np.asarray(mol.atom_mass_list())[:, None] * AMU2AU

    # -----------------------
    # Electronic structure
    # -----------------------
    if spin == 0:
       mf = dft.RKS(mol).density_fit()
    else:
       mf = dft.UKS(mol).density_fit()

    mf.xc = functional
    if dispersion != "none":
        mf.disp = dispersion
    mf.conv_tol = conv_tol

    scanner = mf.nuc_grad_method().as_scanner()

    def compute_energy_gradient():

        if backend == "pyscf":
            epot, grad = scanner(mol)

        else:
            raise ValueError(f"Unsupported backend: {backend}")  
            exit(1)

        return epot, np.asarray(grad)

    # -----------------------
    # AS preparation
    #------------------------

    if inp.switching_fun == "sine":
       switching = switching_sine
       print(f" Using sine switching function")
    elif inp.switching_fun == "none":
       switching = switching_none
       print(f" Using no switching function")
    elif inp.switching_fun == "linear":
       switching = switching_linear
       print(f" Using linear switching function")
    elif inp.switching_fun == "smooth":
       switching = switching_smoothstep
       print(f" Using smoothstep switching function")
    else:
       switching = switching_none
       print(f" Unsupported switching:\n I am not switching at all")

    if inp.harmonic_gen == "cnorm":
       # --- cnorm reading and cleaning
       cnorm, omega2 = read_cnorm(cnormfile)
       omega2[-NROTRASL:] = 0.0
       harmonic = {"cnorm": cnorm, "omega2": omega2, "masses_au": mass}
       harmonic_force = harmonic_force_cnorm
       harmonic_energy = harmonic_energy_cnorm
       print(f" Using cnorm for harmonic job")
    elif inp.harmonic_gen == "hessian":
       H = read_flat_lower_hessian(hessfile)
       harmonic = {"H": H}
       harmonic_force = harmonic_force_hessian
       harmonic_energy = harmonic_energy_hessian
       print(f" Using hessian for harmonic job")
    else:
       harmonic_force = harmonic_force_hessian
       harmonic_energy = harmonic_energy_hessian
       print(f" Unsupported harmonic force:\n I am using the Hessian ")

    # -----------------------
    # Equilibrium energy
    # -----------------------

    E0, grad = compute_energy_gradient()

    # -----------------------
    # Output files
    # -----------------------

    with open(traj_file, "w") as traj, open(force_file, "w") as forces, open(md_file, "w") as md:

        md.write(f"{'Step':>8s}{'Time':>16s}{'Epot (au)':>20s}{'Ekin (au)':>20s}{'Etot (au)':>20s}{'Etot (cm-1)':>20s}\n")

        # -----------------------
        # Initial energy and force
        # -----------------------
        mol.set_geom_(xyzfile, unit="Ang")
        coords = mol.atom_coords(unit="Bohr").copy()
        e_real, grad = compute_energy_gradient()
        f_real = -np.asarray(grad)
        ekin = 0.5 * np.sum(mass * vel**2)
        time = 0.0
        step = 0
        # --- it always scales the energy
        lam = switching(step, switching_steps) 
        f_harm = harmonic_force(coords, coords0, harmonic)
        frc = (1.0 - lam) * f_harm + lam * f_real
        eharm = harmonic_energy(coords, coords0, harmonic)
        epot = (1.0 -lam)*eharm + lam*(e_real -E0)

        write_output(traj, forces, md, step, time, mol, vel, frc, epot, ekin)

        # DEBUG
        #print(f"step {step}")
        #print(f"  E0      = {E0:.15e}")
        #print(f"  e_real  = {e_real:.15e}")
        #print(f"  epot    = {epot:.15e}")
        #print(f"  |coord| = {np.linalg.norm(coords):.15e}")

        # -----------------------
        # Velocity Verlet
        # -----------------------
        for step in range(1, nsteps + 1):
            
            lam = switching(step, switching_steps) 
            print(f"{step:6d}  {lam:12.8f}", flush=True)
            acc = frc / mass

            coords = coords + vel * dt + 0.5 * acc * dt**2
            mol.set_geom_(coords, unit="Bohr")

            e_real, grad = compute_energy_gradient()
            f_real = -np.asarray(grad)
 
            f_harm = harmonic_force(coords, coords0, harmonic)
            frc_new = (1.0 - lam) * f_harm + lam * f_real 

            acc_new = frc_new / mass
            vel = vel + 0.5 * (acc + acc_new) * dt

            frc = frc_new
            time = step * dt
            ekin = 0.5 * np.sum(mass * vel**2)
            eharm = harmonic_energy(coords, coords0, harmonic)
            epot = (1.0 -lam)*eharm + lam*(e_real-E0)

            write_output(traj, forces, md, step, time, mol, vel, frc, epot, ekin)
            
            # DEBUG 
            #print(f"step {step}")
            #print(f"  E0      = {E0:.15e}")
            #print(f"  e_real  = {e_real:.15e}")
            #print(f"  epot    = {epot:.15e}")
            #print(f"  |coord| = {np.linalg.norm(coords):.15e}")

    write_final_files(final_geo, final_vel, mol, vel)

if __name__ == "__main__":

    print(r"""
    ██████╗  ██████╗ ███╗   ███╗██████╗
    ██╔══██╗██╔═══██╗████╗ ████║██╔══██╗
    ██████╔╝██║   ██║██╔████╔██║██║  ██║
    ██╔══██╗██║   ██║██║╚██╔╝██║██║  ██║
    ██████╔╝╚██████╔╝██║ ╚═╝ ██║██████╔╝
    ╚═════╝  ╚═════╝ ╚═╝     ╚═╝╚═════╝

    ██████╗ ██╗   ██╗███╗   ██╗███╗   ██╗███████╗██████╗
    ██╔══██╗██║   ██║████╗  ██║████╗  ██║██╔════╝██╔══██╗
    ██████╔╝██║   ██║██╔██╗ ██║██╔██╗ ██║█████╗  ██████╔╝
    ██╔══██╗██║   ██║██║╚██╗██║██║╚██╗██║██╔══╝  ██╔══██╗
    ██║  ██║╚██████╔╝██║ ╚████║██║ ╚████║███████╗██║  ██║
    ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝

          A PySCF-based trajectory integrator 
              compatible with DragonBall
    """)
    print("   Version 0.1")
    print("   G. Botti")
    print()
    #run_bomd()
    try:
        run_bomd()

    finally:
        elapsed = time.perf_counter() - t0
        print(f"\nTotal BOMD wall time: {elapsed:.2f} s", flush=True)
