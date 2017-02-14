import astropy.cosmology as ac
import astropy.units as u

##############################################################################
# If you change these, you must also re-run `scripts/setup_potential.py`
#
cosmology = ac.Planck15
Mvir0 = 1E12 * u.Msun # MAGIC NUMBER - assumption
#
##############################################################################

# from Gnedin et al. 2014
M_min = 1E4 * u.Msun
M_max = 1E7 * u.Msun

# amount of time to evolve clusters
z_max = 3.
t_max = -cosmology.lookback_time(z_max)
