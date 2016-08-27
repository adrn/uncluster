# Third-party
from astropy.constants import G as _G
from astropy import log as logger
import astropy.units as u
import numpy as np
import gala.potential as gp
from gala.units import galactic

# Project
from ..distribution_function import SphericalIsotropicDF

def test_spherical_isotropic():
    G = _G.decompose(galactic).value

    # Hernquist density profile
    m_h = 1E12 # Msun
    c = 20. # kpc
    hernquist = gp.HernquistPotential(m=m_h, c=c, units=galactic)

    # Plummer background potential
    b = 2.5 # kpc
    m_p = 5E10 # Msun
    plummer = gp.PlummerPotential(m=m_p, b=b, units=galactic)

    def hernquist_r2(phi):
        return (G*m_h/phi)**2 + c**2 + 2*G*m_h*c/phi

    def plummer_density(phi):
        r2 = hernquist_r2(phi)
        return 3/(4*np.pi*b**3) * (1+r2/b**2)**-2.5

    curlyE = np.linspace(1E-2,5.1,256)
    energy_grid = -curlyE * G*m_h/c
    df = SphericalIsotropicDF(tracer=plummer_density, background=hernquist, energy_grid=energy_grid)
