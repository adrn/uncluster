# Standard library
import os
if not os.path.exists(os.path.abspath("../data")):
    raise ValueError("This must be run from the 'scripts' directory of the "
                     "cloned 'uncluster' repository.")

# Third-party
import astropy.cosmology as ac
import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d

M_vir0 = 1E12 * u.Msun
cosmo = ac.Planck15
z_max = 4
n_grid = 256

# Use astropy.cosmology to compute the evolution of the virial mass and
#   virial radius of our halo potential. This script prints the grids so
#   the C code (uncluster/potential/components.c) can interpolate to get
#   the quantities at any input time.
t_grid = np.linspace(0, 1., n_grid) * cosmo.lookback_time(z_max)
z_grid = np.array([0.] + [ac.z_at_value(cosmo.lookback_time, tt) for tt in t_grid[1:]])
t_grid = t_grid[::-1]
z_grid = z_grid[::-1]

def Delta(z):
    """ An approximation thanks to Dekel & Birnboim 2006 (see appendix) """
    return (18*np.pi**2 - 82*cosmo.Ode(z) - 39*cosmo.Ode(z)**2) / cosmo.Om(z)

def M_vir(z):
    return M_vir0 * np.exp(-0.8*z)

def R_vir(z):
    _Delta = (Delta(z) / 200)**(-1/3.)
    _Om = (cosmo.Om(z) / 0.3)**(-1/3.)
    _h2 = (cosmo.h / 0.7)**(-2/3.)
    return 309 * (M_vir(z)/(1E12*u.Msun))**(1/3) * _Delta * _Om * _h2 / (1 + z) * u.kpc

lines = []
lines.append(str((-t_grid.to(u.Myr).value).tolist()))
lines.append(str(z_grid.tolist()))
lines.append(str(R_vir(z_grid).to(u.kpc).value.tolist()))
lines.append(str(M_vir(z_grid).to(u.Msun).value.tolist()))

# Now we need to make a grid for the stellar mass evolution. We use the
#   star formation rates from Behroozi et al. 2013 for a 10^12 solar mass
#   virial mass halo.
path = os.path.join(os.path.abspath("../data"), "sm_hist_rel_12.0.dat")
d = np.genfromtxt(path, names=['a', 'f_star'], usecols=[0,1])

ifunc = interp1d(1/d['a'] - 1, d['f_star'], kind='linear')
f_star = ifunc(z_grid)
lines.append(str(f_star.tolist()))

for i,line in enumerate(lines):
    line = line.replace("[", "{")
    line = line.replace("]", "}")
    lines[i] = "{};\n".format(line)

with open("../data/cosmo_arrays.c", "w") as f:
    f.writelines(lines)
