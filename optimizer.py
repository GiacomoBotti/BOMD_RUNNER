import re
import numpy as np
import config as inp
from pyscf import gto, scf, dft
from pyscf.hessian import thermo

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
xyzguess = inp.xyzguess

# --- BERNY OPTIMIZATION PARAMETERS
#gradientmax = inp.opt_gradientmax
#gradientrms = inp.opt_gradientrms
#stepmax = inp.opt_stepmax
#steprms = inp.opt_steprms

# --- geomeTRIC OPTIMIZATION PARAMETERS
opt_energy = inp.opt_energy 
opt_grms = inp.opt_grms
opt_gmax = inp.opt_gmax
opt_drms = inp.opt_drms
opt_dmax = inp.opt_dmax

# --- OUTPUT FILES
xyzfile   = inp.xyzfile
hessfile  = inp.hessfile

# --- LEVEL OF THEORY
functional = inp.functional
dispersion = inp.dispersion
basis      = inp.basis
charge     = inp.charge
spin       = inp.spin
conv_tol   = inp.conv_tol
backend    = inp.backend

def write_optgeo(opt_geo, mol):
    natm = mol.natm
    symbols = [mol.atom_symbol(i) for i in range(natm)]
    coords = mol.atom_coords(unit="Ang")

    with open(opt_geo, "w") as og:
        og.write(f"{natm}\n")
        og.write("Final geometry\n")
        for sym, r in zip(symbols, coords):
            og.write(f"{sym:2s} {r[0]:16.10f} {r[1]:16.10f} {r[2]:16.10f}\n")

def write_hessian(hessfile, hess):
    with open(hessfile, "w") as hf:
        hf.write("\n\n")  # two blank lines

        ndim = hess.shape[0]

        for i in range(ndim):
            for j in range(i + 1):
                hf.write(f"{hess[i,j]:20.10E}".replace("E", "D") + "\n")

def run_optfreq():
    # -----------------------
    # Molecule
    # -----------------------
    mol = gto.Mole()
    mol.atom = xyzguess
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()

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

    # ------------------------------
    # Optimization with Berny (BAD)
    # ------------------------------
    #conv_params = {  # These are the default settings
    #    'gradientmax': gradientmax,  # Eh/[Bohr|rad]
    #    'gradientrms': gradientrms,  # Eh/[Bohr|rad]
    #    'stepmax': stepmax,       # [Bohr|rad]
    #    'steprms': steprms,       # [Bohr|rad]
    #}
 
    #mol = mf.Gradients().optimizer(solver='berny').kernel(conv_params)

    # ------------------------------
    # Optimization with geomeTRIC 
    # ------------------------------

    conv_params = {
    "convergence_energy": opt_energy,
    "convergence_grms": opt_grms,
    "convergence_gmax": opt_gmax,
    "convergence_drms": opt_drms,
    "convergence_dmax": opt_dmax,
    }

    mol = mf.Gradients().optimizer(solver="geomeTRIC").kernel(conv_params)
    write_optgeo(xyzfile,mol)

    # -----------------------
    # Frequencies 
    # -----------------------
    mf.mol = mol
    hess = mf.Hessian().kernel()
    freq_res = thermo.harmonic_analysis(mol, hess)

    print("\nHarmonic frequencies (cm^-1)")
    print("-" * 32)
    for i, freq in enumerate(freq_res["freq_wavenumber"], start=1):
        print(f"{i:3d} {freq:12.4f}")
    #print(hess.shape)
    natm = hess.shape[0]
    hess_cart = hess.transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm)
    write_hessian(hessfile, hess_cart)

if __name__ == "__main__":

    print(r"""
    ██╗  ██╗ █████╗ ██████╗ ███╗   ███╗ ██████╗ ███╗   ██╗██╗ ██████╗
    ██║  ██║██╔══██╗██╔══██╗████╗ ████║██╔═══██╗████╗  ██║██║██╔════╝
    ███████║███████║██████╔╝██╔████╔██║██║   ██║██╔██╗ ██║██║██║
    ██╔══██║██╔══██║██╔══██╗██║╚██╔╝██║██║   ██║██║╚██╗██║██║██║
    ██║  ██║██║  ██║██║  ██║██║ ╚═╝ ██║╚██████╔╝██║ ╚████║██║╚██████╗
    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝ ╚═════╝

     ██████╗ ██████╗ ████████╗██╗███╗   ███╗██╗███████╗███████╗██████╗
    ██╔═══██╗██╔══██╗╚══██╔══╝██║████╗ ████║██║╚══███╔╝██╔════╝██╔══██╗
    ██║   ██║██████╔╝   ██║   ██║██╔████╔██║██║  ███╔╝ █████╗  ██████╔╝
    ██║   ██║██╔═══╝    ██║   ██║██║╚██╔╝██║██║ ███╔╝  ██╔══╝  ██╔══██╗
    ╚██████╔╝██║        ██║   ██║██║ ╚═╝ ██║██║███████╗███████╗██║  ██║
     ╚═════╝ ╚═╝        ╚═╝   ╚═╝╚═╝     ╚═╝╚═╝╚══════╝╚══════╝╚═╝  ╚═╝

          A PySCF-based optimizer and frequency calculator 
              compatible with DragonBall
    """)
    print("   Version 0.1")
    print("   G. Botti")
    print()
    try:
        run_optfreq()

    finally:
        elapsed = time.perf_counter() - t0
        print(f"\nTotal optimization wall time: {elapsed:.2f} s", flush=True)
