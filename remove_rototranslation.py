#!/usr/bin/env python3
"""Project rigid translation and rotation out of an XYZ frame's velocities.

Coordinates in Angstrom; velocities in atomic units (Bohr/atomic time).
By default, reads the last frame of a multi-frame BOMD trajectory.
"""
import argparse
import numpy as np
from pyscf import gto

AMU2AU = 1822.888486209
ANG2BOHR = 1.8897261246257702
AU_TO_CM1 = 219474.6313705


def read_frame(path, index=-1):
    frames = []
    with open(path, encoding='utf-8') as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.strip():
                continue
            n = int(line.strip())
            comment = fh.readline().rstrip('\n')
            symbols, coords, velocities = [], [], []
            for _ in range(n):
                fields = fh.readline().split()
                if len(fields) < 7:
                    raise ValueError('Each atom must have symbol, x y z, vx vy vz')
                symbols.append(fields[0])
                coords.append([float(x) for x in fields[1:4]])
                velocities.append([float(x) for x in fields[4:7]])
            frames.append((symbols, np.array(coords), np.array(velocities), comment))
    if not frames:
        raise ValueError('No frames in input')
    return frames[index], len(frames)


def project(coords_bohr, vel, masses_au, rcond=1e-12):
    m = np.asarray(masses_au, float).reshape(-1)
    r = np.asarray(coords_bohr, float)
    v = np.asarray(vel, float)
    total_mass = m.sum()
    rcm = (m[:, None] * r).sum(axis=0) / total_mass
    vcm = (m[:, None] * v).sum(axis=0) / total_mass
    rho = r - rcm
    u = v - vcm
    L = np.sum(np.cross(rho, m[:, None] * u), axis=0)
    I = np.zeros((3, 3))
    for mi, ri in zip(m, rho):
        I += mi * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri))
    omega = np.linalg.pinv(I, rcond=rcond) @ L
    vnew = u - np.cross(omega, rho)
    t0 = 0.5 * np.sum(m[:, None] * v**2)
    tt = 0.5 * total_mass * np.dot(vcm, vcm)
    tr = 0.5 * np.dot(omega, L)
    tv = 0.5 * np.sum(m[:, None] * vnew**2)
    pnew = np.sum(m[:, None] * vnew, axis=0)
    lnew = np.sum(np.cross(rho, m[:, None] * vnew), axis=0)
    return vnew, dict(L=L, L_projected=lnew, P_projected=pnew,
                      omega=omega, T_total=t0, T_trans=tt, T_rot=tr,
                      T_vib=tv, residual=t0-tt-tr-tv)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('trajectory', help='XYZ with x y z (Angstrom), vx vy vz (a.u.)')
    ap.add_argument('--frame', type=int, default=-1, help='Frame index (default: last)')
    ap.add_argument('--output', default='projected_velocity.xyz')
    ap.add_argument('--masses', help='Optional masses in amu, one per atom (for isotopes)')
    args = ap.parse_args()
    (symbols, coords, vel, comment), nframes = read_frame(args.trajectory, args.frame)
    if args.masses:
        masses_amu = np.loadtxt(args.masses, ndmin=1)
    else:
        mol = gto.Mole()
        mol.atom = list(zip(symbols, coords.tolist()))
        mol.unit = 'Angstrom'
        #mol.basis = {}
        mol.verbose = 0
        mol.build()
        masses_amu = np.asarray(mol.atom_mass_list(), float)
    if len(masses_amu) != len(symbols):
        raise ValueError('Mass count does not match atom count')
    newvel, info = project(coords * ANG2BOHR, vel, masses_amu * AMU2AU)
    with open(args.output, 'w', encoding='utf-8') as out:
        out.write(f'{len(symbols)}\n')
        out.write(f'Projected velocities (a.u.); source frame {args.frame} of {nframes}\n')
        for sym, v in zip(symbols, newvel):
            out.write(f'{sym:2s} {v[0]:20.12e} {v[1]:20.12e} {v[2]:20.12e}\n')
    print(f'Input: {args.trajectory}; frames: {nframes}; selected: {args.frame}')
    print(f'Original angular momentum (hbar): {info["L"]}')
    print(f'Projected angular momentum (hbar): {info["L_projected"]}')
    print(f'Projected linear momentum (a.u.): {info["P_projected"]}')
    for key in ('T_total', 'T_trans', 'T_rot', 'T_vib', 'residual'):
        print(f'{key:>12}: {info[key]:.12e} Eh  ({info[key]*AU_TO_CM1:.8f} cm^-1)')
    print(f'Output: {args.output}')

if __name__ == '__main__':
    main()
