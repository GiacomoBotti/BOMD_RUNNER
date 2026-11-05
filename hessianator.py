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
hessfile   = inp.hdbfile      # output Hessian file
indexfile  = inp.indexfile 
functional = inp.functional
dispersion = inp.dispersion
basis      = inp.basis
charge     = inp.charge
spin       = inp.spin
conv_tol   = inp.conv_tol


def iter_xyz_frames(geomfile):
    """
    Iterate over a multi-frame XYZ file.

    Yields
    ------
    symbols : list[str]
        Atomic symbols in the frame.
    coords : np.ndarray
        Cartesian coordinates with shape (natm, 3), in Angstrom.
    """

    with open(geomfile, "r") as f:
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

def read_index(index_path):
    """Read 1-based representative indices for the full trajectory."""
    values = []
    with open(index_path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                values.append(int(text))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid index value on line {line_number}: {text!r}"
                ) from exc

    if not values:
        raise ValueError(f"Index file is empty: {index_path}")

    index = np.asarray(values, dtype=int)
    if np.any(index < 1):
        raise ValueError("Representative indices must be >= 1")
    return index


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
    use_index = indexfile not in (None, "", "none", "None")
    index = read_index(indexfile) if use_index else None
    frames = iter_xyz_frames(geomfile)

    try:
        symbols, coords = next(frames)
    except StopIteration as exc:
        raise ValueError(f"No geometries found in {geomfile}") from exc

    # Build the molecular object once.
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

    # Store representative Hessians only when reconstruction is requested.
    representative_hessians = [] if use_index else None

    with open(hessfile, "w") as hf:
        print("Computing Hessian for geometry 1")
        mf.kernel()
        hess = mf.Hessian().kernel()
        if use_index:
            representative_hessians.append(hess)
        else:
            write_hessian_block(hf, hess)
        print(f"  SCF energy: {mf.e_tot:.12f}")

        nrep = 1
        for nrep, (new_symbols, new_coords) in enumerate(frames, start=2):
            if new_symbols != symbols:
                raise ValueError(
                    f"Atomic symbols/order changed in geometry {nrep}"
                )
            print(f"Computing Hessian for geometry {nrep}")
            mol.set_geom_(new_coords, unit="Angstrom")
            mf.reset(mol)
            mf.kernel()
            hess = mf.Hessian().kernel()
            if use_index:
                representative_hessians.append(hess)
            else:
                write_hessian_block(hf, hess)
            print(f"  SCF energy: {mf.e_tot:.12f}")

        if use_index:
            if np.max(index) > nrep:
                raise ValueError(
                    f"Index refers to representative {np.max(index)}, "
                    f"but only {nrep} geometries were found"
                )
            for irep in index:
                write_hessian_block(hf, representative_hessians[irep - 1])
            print(f"Reconstructed {len(index)} Hessian frames "
                  f"from {nrep} representative geometries.")
        else:
            print(f"Wrote {nrep} Hessian frames (no index file).")

    print(f"Hessian output: {hessfile}")

if __name__ == "__main__":
    print("Starting Hessian batch run...")
    run_all()
    elapsed = time.perf_counter() - t0
    print(f"\nTotal wall time: {elapsed:.2f} s")
