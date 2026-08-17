# --- INPUT FILES
xyzguess = "guess.xyz" 				# Optimizer initial velocity
xyzfile = "geometry.xyz" 			# Optimizer output/ runner input geometry
velfile = "velocity.xyz"			# Initial velocity
cnormfile = "none"				# Hessian eigenproblem file
hessfile = "Hessian_flat.out"			# Hessian file

# --- OUTPUT FILES
traj_file = "traj.xyz"				# Trajectory print
force_file = "forces.dat"			# Forces print
md_file = "md.log"				# Energies print
final_geo = "final_geo_bomd.xyz"		# Final geometry
final_vel = "final_vel_bomd.xyz"		# Final velocity

# --- LEVEL OF THEORY
functional='b3lyp'				# Functional
dispersion='none'				# Dispersions, none = skip
basis = "def2-SVP"				# Basis set
charge = 0 						
spin = 0 
conv_tol = 1e-10				# Convergence for SCF
backend='pyscf'					# Backend for trajectory energy/ gradient

# --- OPTIMIZATION PARAMETERS (BERNY)           # LEGACY
opt_gradientmax = 0.45e-3                       # not used anymore
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
dt_fs = 0.2 					# Time step size in fs
nsteps = 2500 					# Number of steps
switching_steps = nsteps 			# Number of switching steps
NROTRASL = 5 					# Number of rototraslational modes (5 for linear)

# --- SWITCHING OPTIONS 
switching_fun = "smooth"			# Switching fun: none (classical) sine or smooth
harmonic_gen = "hessian"			# hessian: cartesian harm forces; cnorm: normal modes

# HESSIANATOR INPUT
geomfile = "geom_test_hessian.xyz"		# File with geometries
hdbfile = "hdb.out"				# Output file

