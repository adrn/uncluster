import astropy.units as u

# from Gnedin et al. 2014
f_gc = 0.012 # fraction of mass in GCs from G14
M_min = 1E4 * u.Msun
M_max = 1E7 * u.Msun
M_tot = 5E10 * u.Msun
r_max = 250. * u.kpc # changed from G14 value
t_evolve = 11.5 * u.Gyr

# for DM halo
M_h = 1E12 * u.Msun
rs_h = 20. * u.kpc

del u
