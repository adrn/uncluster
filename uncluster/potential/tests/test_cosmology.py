# Third-party
from astropy.utils.data import get_pkg_data_filename
import astropy.cosmology as ac
import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d

# Project
from ...config import cosmology, Mvir0
from .helpers import (_f_star, _redshift, _nu_relative_density, _inv_efunc_sq,
                      _M_vir, _R_vir)

n_grid = 128
z_max = 4.

def test_redshift():
    t_grid = np.linspace(0, 1., n_grid) * cosmology.lookback_time(z_max)
    z_grid = np.array([0.] + [ac.z_at_value(cosmology.lookback_time, tt) for tt in t_grid[1:]])
    z_check = _redshift(-t_grid.to(u.Myr).value)
    assert np.allclose(z_grid, z_check, rtol=2E-2)

def test_nu_relative_density():
    z_grid = np.linspace(0, z_max, n_grid)
    nu1 = cosmology.nu_relative_density(z_grid)
    nu2 = _nu_relative_density(z_grid)
    assert np.allclose(nu1, nu2)

def test_inv_efunc_sq():
    z_grid = np.linspace(0, z_max, n_grid)
    arr1 = cosmology.inv_efunc(z_grid)**2
    arr2 = _inv_efunc_sq(z_grid)
    assert np.allclose(arr1, arr2)

def test_fstar():
    z_grid = np.linspace(0, 3., n_grid)

    # Read actual Behroozi data table:
    fstar_file = get_pkg_data_filename('../../data/sm_hist_rel_12.0.dat')
    d = np.genfromtxt(fstar_file, names=['a', 'f_star'], usecols=[0,1])
    ifunc = interp1d(1/d['a'] - 1, d['f_star'], kind='linear')
    f_star = ifunc(z_grid)
    assert np.allclose(f_star, _f_star(z_grid), rtol=0.2)

# --------------------------------------------------------------

def Delta(z):
    """ An approximation thanks to Dekel & Birnboim 2006 (see appendix) """
    return (18*np.pi**2 - 82*cosmology.Ode(z) - 39*cosmology.Ode(z)**2) / cosmology.Om(z)

def M_vir(z):
    return Mvir0 * np.exp(-0.8*z)

def R_vir(z):
    _Delta = (Delta(z) / 200)**(-1/3.)
    _Om = (cosmology.Om(z) / 0.3)**(-1/3.)
    _h2 = (cosmology.h / 0.7)**(-2/3.)
    return 309 * (M_vir(z)/(1E12*u.Msun))**(1/3) * _Delta * _Om * _h2 / (1 + z) * u.kpc

def test_M_vir():
    z_grid = np.linspace(0, z_max, n_grid)
    arr1 = M_vir(z_grid).to(u.Msun).value
    arr2 = _M_vir(z_grid)
    assert np.allclose(arr1, arr2)

def test_R_vir():
    z_grid = np.linspace(0, z_max, n_grid)
    arr1 = R_vir(z_grid).to(u.kpc).value
    arr2 = _R_vir(z_grid)
    assert np.allclose(arr1, arr2)
