#!/usr/bin/env python3
"""Analyze rigid-body translation and rotation in BOMD XYZ trajectories.

Expected frame: N, comment with Step/Time, then N rows of
symbol x y z vx vy vz. Coordinates in Angstrom; velocities in a.u.
"""
import argparse
import re
import numpy as np

AMU2AU = 1822.888486209
ANG2BOHR = 1.8897261246257702
AU_TO_CM1 = 219474.6313705
FS2AU = 41.3413745758
def pyscf_masses(symbols, isotope_masses=None):
    """Return PySCF atomic masses in electron masses, matching runner.py."""
    from pyscf import gto
    mol = gto.Mole()
    mol.atom = [(symbol, (float(i) * 5.0, 0.0, 0.0))
                for i, symbol in enumerate(symbols)]
    mol.unit = 'Bohr'
    mol.verbose = 0
    mol.basis = 'sto-3g'
    mol.build(parse_arg=False, dump_input=False)
    masses = np.asarray(mol.atom_mass_list(), dtype=float)
    if isotope_masses is not None:
        masses = np.asarray(isotope_masses, dtype=float)
        if masses.shape != (len(symbols),):
            raise ValueError('Mass file must contain one mass (amu) per atom')
    return masses * AMU2AU

def frames(path):
    with open(path, encoding='utf-8') as fh:
        while line := fh.readline():
            if not line.strip():
                continue
            n = int(line.strip())
            comment = fh.readline()
            step_match = re.search(r'Step\s*=\s*(\d+)', comment)
            time_match = re.search(r'Time\s*=\s*([\d.eE+-]+)', comment)
            symbols, xyz, vel = [], [], []
            for _ in range(n):
                fields = fh.readline().split()
                if len(fields) < 7:
                    raise ValueError('Each atom needs symbol, x y z, vx vy vz')
                symbols.append(fields[0])
                xyz.append([float(v) for v in fields[1:4]])
                vel.append([float(v) for v in fields[4:7]])
            yield (int(step_match.group(1)) if step_match else None,
                   float(time_match.group(1)) if time_match else None,
                   symbols, np.array(xyz) * ANG2BOHR, np.array(vel))

def diagnostics(coords, velocities, masses):
    m = masses[:, None]
    total_mass = masses.sum()
    rcm = (m * coords).sum(axis=0) / total_mass
    vcm = (m * velocities).sum(axis=0) / total_mass
    r = coords - rcm
    v = velocities - vcm
    L = np.sum(m * np.cross(r, v), axis=0)
    I = np.einsum('i,ijk->jk', masses,
                  np.sum(r*r, axis=1)[:, None, None] * np.eye(3)
                  - r[:, :, None] * r[:, None, :])
    omega = np.linalg.pinv(I, rcond=1e-12) @ L
    vrot = np.cross(np.broadcast_to(omega, r.shape), r)
    vvib = v - vrot
    ttotal = 0.5 * np.sum(m * velocities**2)
    ttrans = 0.5 * total_mass * np.dot(vcm, vcm)
    trot = 0.5 * np.dot(omega, L)
    tvib = 0.5 * np.sum(m * vvib**2)
    return np.r_[L, np.linalg.norm(L), ttotal, ttrans, trot, tvib,
                 trot/ttotal if ttotal else np.nan,
                 np.linalg.norm((m * velocities).sum(axis=0)),
                 np.linalg.norm(vcm),
                 ttotal - ttrans - trot - tvib]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('trajectory', nargs='?', default='traj.xyz')
    ap.add_argument('--output', default='rotation_diagnostics.dat')
    ap.add_argument('--plot', default='rotation_diagnostics.png')
    ap.add_argument('--masses', help='Optional per-atom masses in amu (one per line), for isotopes')
    args = ap.parse_args()
    rows = []
    reference_symbols = None
    for index, (step, time, symbols, xyz, vel) in enumerate(frames(args.trajectory)):
        if reference_symbols is None:
            reference_symbols = symbols
            custom_masses = np.loadtxt(args.masses, ndmin=1) if args.masses else None
            masses = pyscf_masses(symbols, custom_masses)
        if symbols != reference_symbols:
            raise ValueError(f'Atom order changed at frame {index}')
        d = diagnostics(xyz, vel, masses)
        rows.append([index if step is None else step,
                     np.nan if time is None else time / FS2AU, *d])
    a = np.asarray(rows)
    names = ['step','time_fs','Lx_hbar','Ly_hbar','Lz_hbar','Lnorm_hbar',
             'Ttotal_Eh','Ttrans_Eh','Trot_Eh','Tvib_Eh','rot_fraction',
             'Pnorm_au','Vcm_au','energy_closure_Eh']
    np.savetxt(args.output, a, header=' '.join(names), fmt='%.12e')
    print(f'Frames: {len(a)}, atoms: {len(reference_symbols)} ({" ".join(reference_symbols)})')
    print(f'Time: {a[0,1]:.4f} to {a[-1,1]:.4f} fs')
    print('Atomic masses (amu):', ' '.join(f'{m/AMU2AU:.8f}' for m in masses))
    for label, j in [('initial',0),('final',-1)]:
        x = a[j]
        print(f'{label}: |L|={x[5]:.8g} hbar, Trot={x[8]*AU_TO_CM1:.6f} cm^-1, '
              f'Ttrans={x[7]*AU_TO_CM1:.6f} cm^-1, '
              f'Tvib={x[9]*AU_TO_CM1:.6f} cm^-1, '
              f'Trot/T={100*x[10]:.4f}%')
    print(f'Max |L|: {a[:,5].max():.8g} hbar; max Trot: {a[:,8].max()*AU_TO_CM1:.6f} cm^-1')
    print(f'Max energy closure residual: {np.abs(a[:,13]).max():.3e} Eh')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        for k, label in zip((2, 3, 4, 5), ('Lx', 'Ly', 'Lz', '|L|')):
            axes[0].plot(a[:,1], a[:,k], label=label)
        axes[0].legend(ncol=4); axes[0].set_ylabel(r'Angular momentum ($\hbar$)')
        axes[1].plot(a[:,1], a[:,8]*AU_TO_CM1, label='Rotation')
        axes[1].plot(a[:,1], a[:,7]*AU_TO_CM1, label='Translation')
        axes[1].legend(); axes[1].set_ylabel(r'Energy (cm$^{-1}$)')
        axes[2].plot(a[:,1], 100*a[:,10]); axes[2].set_ylabel('Rotational fraction (%)')
        axes[2].set_xlabel('Time (fs)')
        fig.tight_layout(); fig.savefig(args.plot, dpi=180)
        plt.close(fig)
        print(f'Wrote {args.output} and {args.plot}')
    except ImportError:
        print(f'Wrote {args.output}; matplotlib unavailable, plot skipped')

if __name__ == '__main__':
    main()
