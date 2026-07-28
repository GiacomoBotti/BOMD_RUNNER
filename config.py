# --- INPUT FILES
xyzfile = "geometry.xyz"
velfile = "velocity.xyz"
cnormfile = "none"
hessfile = "Hessian_flat.out"

# --- OUTPUT FILES
traj_file = "traj.xyz"
force_file = "forces.dat"
md_file = "md.log"
final_geo = "final_geo.xyz"
final_vel = "final_vel.xyz"

# --- LEVEL OF THEORY
functional='b3lyp'
dispersion='d4'
basis = "def2-TZVP"
charge = 0 
spin = 0 
conv_tol = 1e-10
backend='pyscf'

# --- DYNAMICS PARAMETERS
dt_fs = 0.2 
nsteps = 2500 
switching_steps = nsteps 

NROTRASL = 5 

# --- SWITCHING OPTIONS 
switching_fun = "sine"
harmonic_gen = "hessian"

