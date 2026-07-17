import numpy as np
from pyscf import gto, scf, dft

AMU2AU = 1822.888486209
ANG2BOHR = 1.8897261246257702
BOHR2ANG = 1.0 / ANG2BOHR
FS2AU = 41.3413745758
AU2FS = 1.0 / FS2AU

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
        f"{epot:20.12f}"
        f"{ekin:20.12f}"
        f"{epot + ekin:20.12f}\n"
    )

    traj.flush()
    forces.flush()
    md.flush()

def run_bomd():
    # -----------------------
    # Input
    # -----------------------
    xyzfile = "geometry.xyz"
    velfile = "velocity.xyz"

    basis = "def2-TZVP"
    #basis = "def2-SVP"
    charge = 0
    spin = 0

    dt_fs = 0.2
    dt = dt_fs*FS2AU
    nsteps = 2500 
    conv_tol = 1e-10

    traj_file = "traj.xyz"
    force_file = "forces.dat"
    md_file = "md.log"

    # -----------------------
    # Molecule
    # -----------------------
    mol = gto.Mole()
    mol.atom = xyzfile
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()

    coords = mol.atom_coords()  # Bohr
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
        mf.xc = "b3lyp"
        mf.disp = "d4"
    else:
        mf = scf.UHF(mol).density_fit()

    mf.conv_tol = conv_tol
    backend='pyscf'
    scanner = mf.nuc_grad_method().as_scanner()

    def compute_energy_gradient():

        if backend == "pyscf":
            epot, grad = scanner(mol)

        else:
            raise ValueError(f"Unsupported backend: {backend}")  
            exit(1)

        return epot, np.asarray(grad)

    # -----------------------
    # Output files
    # -----------------------
    with open(traj_file, "w") as traj, open(force_file, "w") as forces, open(md_file, "w") as md:

        md.write(f"{'Step':>8s}{'Time':>16s}{'Epot':>20s}{'Ekin':>20s}{'Etot':>20s}\n")

        # -----------------------
        # Initial energy and force
        # -----------------------
        epot, grad = compute_energy_gradient()
        frc = -np.asarray(grad)
        ekin = 0.5 * np.sum(mass * vel**2)
        time = 0.0
        step = 0

        write_output(traj, forces, md, step, time, mol, vel, frc, epot, ekin)

        # -----------------------
        # Velocity Verlet
        # -----------------------
        for step in range(1, nsteps + 1):
            acc = frc / mass

            coords = coords + vel * dt + 0.5 * acc * dt**2
            mol.set_geom_(coords, unit="Bohr")

            epot, grad = compute_energy_gradient()
            frc_new = -np.asarray(grad)

            acc_new = frc_new / mass
            vel = vel + 0.5 * (acc + acc_new) * dt

            frc = frc_new
            time = step * dt
            ekin = 0.5 * np.sum(mass * vel**2)

            write_output(traj, forces, md, step, time, mol, vel, frc, epot, ekin)

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
    print("Version 0.1")
    print("G. Botti")
    print()
    run_bomd()
