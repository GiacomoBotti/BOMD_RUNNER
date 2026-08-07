import time
import numpy as np
from pathlib import Path

import config as inp
from pyscf import gto, dft

t0 = time.perf_counter()

# ------------------------
# INPUT FROM config.py
# ------------------------
geomfile   = inp.geomfile      # trajectory-like XYZ file
hessfile   = inp.hessfile      # output Hessian file
functional = inp.functional
dispersion = inp.dispersion
basis      = inp.basis
charge     = inp.charge
spin       = inp.spin
conv_tol   = inp.conv_tol


def iter_xyz_frames(xyzfile):
    """
    Iterate over a multi-frame XYZ file.

    Yields
    ------
    symbols : list[str]
        Atomic symbols in the frame.
    coords : np.ndarray
        Cartesian coordinates with shape (natm, 3), in Angstrom.
    """

    with open(xyzfile, "r") as f:
        while True:
            line = f.readline()

            # End of file
            if not line:
                return

            # Ignore blank lines between frames
            if not line.strip():
                continue

            natm = int(line)

            # Skip XYZ comment line
            f.readline()

            symbols = []
            coords = np.empty((natm, 3))

            for i in range(natm):
                fields = f.readline().split()

                symbols.append(fields[0])
                coords[i] = [
                    float(fields[1]),
                    float(fields[2]),
                    float(fields[3]),
                ]

            yield symbols, coords

def write_hessian_block(fh, hess4):
    """
    Write one Hessian in DragonBall format:
    - two blank lines
    - lower triangle of the Cartesian Hessian
    """
    natm = hess4.shape[0]
    hess_cart = hess4.transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm)

    fh.write("\n\n")
    ndim = 3 * natm
    for i in range(ndim):
        for j in range(i + 1):
            fh.write(f"{hess_cart[i, j]:20.10E}".replace("E", "D") + "\n")


def run_all():
    # Start fresh each time
    Path(hessfile).write_text("")

    # -----------------------
    # Frame generator
    # -----------------------
    frames = iter_xyz_frames(geomfile)

    # -----------------------
    # First geometry
    # -----------------------
    symbols, coords = next(frames)

    # Build once
    mol = gto.Mole()
    mol.atom = list(zip(symbols, coords))
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()

    if spin == 0:
        mf = dft.RKS(mol).density_fit()
    else:
        mf = dft.UKS(mol).density_fit()

    mf.xc = functional
    mf.conv_tol = conv_tol

    if dispersion != "none":
        mf.disp = dispersion

    with open(hessfile, "a") as hf:

        # First geometry
        mf.kernel()
        hess = mf.Hessian().kernel()
        write_hessian_block(hf, hess)

        # Remaining geometries
        for iframe, (new_symbols, coords) in enumerate(frames, start=2):

            if new_symbols != symbols:
                raise ValueError(
                    f"Atomic symbols/order changed in frame {iframe}"
                )

            print(f"Computing Hessian for geometry {iframe}")

            mol.set_geom_(coords, unit="Angstrom")

            mf.reset(mol)
            mf.kernel()
            hess = mf.Hessian().kernel()
            write_hessian_block(hf, hess)

            print(f"  SCF energy: {mf.e_tot:.12f}")

if __name__ == "__main__":
    print("Starting Hessian batch run...")
    run_all()
    elapsed = time.perf_counter() - t0
    print(f"\nTotal wall time: {elapsed:.2f} s")
