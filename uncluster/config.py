from math import log, sqrt
import astropy.units as u
from astropy.constants import G
import gala.potential as gp
from gala.units import galactic

# from Gnedin et al. 2014
f_gc = 0.012 # fraction of total stellar mass in GCs
M_min = 1E4 * u.Msun
M_max = 1E7 * u.Msun
M_tot = 5E10 * u.Msun
t_evolve = 11.5 * u.Gyr

# Background Milky Way potential
mw_potential = gp.CompositePotential()
mw_potential['disk'] = gp.MiyamotoNagaiPotential(m=4E10, a=3.5, b=0.14, units=galactic)
mw_potential['bulge'] = gp.HernquistPotential(m=1E10, c=1.1, units=galactic)

# for DM halo potential
M_h = 1E12 * u.Msun
rs_h = 20. * u.kpc
v_c = sqrt(((log(2.) - 0.5) * (G * M_h / rs_h)).decompose(galactic).value)
mw_potential['halo'] = gp.SphericalNFWPotential(v_c=v_c, r_s=rs_h, units=galactic)

