# --- INPUT FILES
xyzguess = "guess.xyz"
xyzfile = "geometry.xyz"
velfile = "velocity.xyz"
cnormfile = "none"
hessfile = "Hessian_flat.out"

# --- OUTPUT FILES
traj_file = "traj.xyz"
force_file = "forces.dat"
md_file = "md.log"
final_geo = "final_geo_bomd.xyz"
final_vel = "final_vel_bomd.xyz"

# --- LEVEL OF THEORY
functional='b3lyp'
dispersion='none'
basis = "def2-SVP"
charge = 0 
spin = 0 
conv_tol = 1e-10
backend='pyscf'

# --- OPTIMIZATION PARAMETERS (BERNY)
opt_gradientmax = 0.45e-3
opt_gradientrms = 0.15e-3
opt_stepmax = 1.8e-3
opt_steprms = 1.2e-3

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
NROTRASL = 5 

# --- SWITCHING OPTIONS 
switching_fun = "smooth"
harmonic_gen = "hessian"

# HESSIANATOR INPUT
geomfile = "geom_test_hessian.xyz"
hessfile = "hdb.out"

