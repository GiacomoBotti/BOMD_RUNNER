import numpy as np
from pyscf.hessian.thermo import harmonic_analysis

def make_qct_initial_velocities(
    mol,
    mf,
    occupied_quanta,
    hessian_file="Hessian.out",
    velocity_file="velocity.xyz",
    seed=None,
):
    """
    Compute the Hessian, print frequencies, write Hessian.out, and generate
    initial Cartesian velocities from occupied vibrational quanta.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Converged molecule.
    mf : pyscf.scf object
        Converged mean-field object.
    occupied_quanta : list[tuple[int, int]]
        List of (mode_index, quantum_number) pairs for vibrational modes.
        Mode indices are 1-based and refer to the vibrational modes after
        translations/rotations have been projected out.
    hessian_file : str
        Output file for the Hessian.
    velocity_file : str
        Output file for the initial velocities.
    seed : int or None
        Random seed for the phase sampling.

    Returns
    -------
    vel : (natm, 3) ndarray
        Cartesian velocities in atomic units.
    vib : dict
        Harmonic analysis dictionary from PySCF for the vibrational subspace.
    """

    # Analytical Hessian from the converged mean-field object
    hess = mf.Hessian().kernel()
    hess = np.asarray(hess)

    # Full 3N analysis: print all frequencies, including translational/rotational ones
    vib_all = harmonic_analysis(
        mol,
        hess,
        exclude_trans=False,
        exclude_rot=False,
        imaginary_freq=False,
    )

    print("\nFrequencies (cm^-1):")
    for i, w in enumerate(np.asarray(vib_all["freq_wavenumber"]).ravel(), 1):
        print(f"{i:4d}  {w:16.6f}")

    # Vibrational subspace analysis: this is the one used for QCT initialization
    vib = harmonic_analysis(
        mol,
        hess,
        exclude_trans=True,
        exclude_rot=True,
        imaginary_freq=False,
    )

    freq = np.asarray(vib["freq_au"], dtype=float)
    norm_mode = np.asarray(vib["norm_mode"], dtype=float)   # (nmodes, natm, 3)
    nmodes = freq.size
    natm = mol.natm

    if np.any(freq <= 0.0):
        raise ValueError(
            "Non-positive vibrational frequency encountered. "
            "The geometry is likely not a true minimum."
        )

    # Parse occupied quanta: (mode_index, quantum_number), 1-based mode index
    quanta = np.zeros(nmodes, dtype=int)
    for mode_index, n in occupied_quanta:
        if mode_index < 1 or mode_index > nmodes:
            raise ValueError(f"Mode index {mode_index} out of range for {nmodes} vibrational modes")
        if n < 0:
            raise ValueError("Quantum number must be non-negative")
        quanta[mode_index - 1] = int(n)

    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=nmodes)

    # Harmonic-oscillator amplitudes in mass-weighted normal coordinates
    # Q_i(t) = A_i sin(phi_i), P_i(t) = A_i * omega_i cos(phi_i)
    amp_q = np.sqrt((2 * quanta + 1) / freq)
    mom_nm = np.sqrt((2 * quanta + 1) * freq) * np.cos(phases)

    # Cartesian velocities from the normal-mode matrix
    # norm_mode has shape (nmodes, natm, 3)
    vel = np.tensordot(mom_nm, norm_mode, axes=(0, 0))

    # Write Hessian.out in atom-block form, with two blank lines on top
    with open(hessian_file, "w") as fh:
        fh.write("\n\n")
        for ia in range(natm):
            for row in range(3):
                parts = []
                for ja in range(natm):
                    block = hess[ia, ja]  # 3x3 block
                    parts.extend(f"{block[row, col]:16.8e}" for col in range(3))
                fh.write(" ".join(parts) + "\n")
            fh.write("\n")

    # Write velocity.xyz
    symbols = [mol.atom_symbol(i) for i in range(natm)]
    with open(velocity_file, "w") as fv:
        fv.write(f"{natm}\n")
        fv.write("Initial Cartesian velocities from PySCF harmonic analysis\n")
        for sym, v in zip(symbols, vel):
            fv.write(f"{sym:2s} {v[0]:16.8e} {v[1]:16.8e} {v[2]:16.8e}\n")

    return vel, vib

mf.kernel()
vel, vib = make_qct_initial_velocities(
    mol,
    mf,
    occupied_quanta=[(1, 1), (2, 0), (3, 1)],
    seed=1234,
)
