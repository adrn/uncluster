# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import astropy.units as u
import numpy as np

def test_apw():
    from ..apw import sample_radii, sample_masses, gc_prob_density

    r = sample_radii(size=128)
    assert np.isfinite(r).all()

def test_gnedin():
    from ..gnedin import sample_radii, sample_masses, gc_prob_density

    r = sample_radii(size=128)
    assert np.isfinite(r).all()
