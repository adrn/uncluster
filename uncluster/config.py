import astropy.cosmology as ac
import astropy.units as u

##############################################################################
# If you change these, you must also re-run `scripts/setup_potential.py`
#
cosmology = ac.Planck15
halo_concentration = 15. # MAGIC NUMBER - assumption
#
##############################################################################

# from Gnedin et al. 2014
f_gc = 0.012 # fraction of total stellar mass in GCs
M_min = 1E4 * u.Msun
M_max = 1E7 * u.Msun
M_tot = 5E10 * u.Msun

# amount of time to evolve clusters
z_max = 3.
t_max = cosmology.lookback_time(z_max)
