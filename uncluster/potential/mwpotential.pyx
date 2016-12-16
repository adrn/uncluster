# coding: utf-8
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from collections import OrderedDict

# Third-party
import astropy.units as u
import numpy as np
cimport numpy as np
np.import_array()
import cython
cimport cython

# Project
from gala.units import galactic
from gala.potential.potential.cpotential cimport CPotentialWrapper
from gala.potential.potential.cpotential import CPotentialBase

cdef extern from "src/funcdefs.h":
    ctypedef double (*densityfunc)(double t, double *pars, double *q, int n_dim, int n_dim) nogil
    ctypedef double (*energyfunc)(double t, double *pars, double *q, int n_dim, int n_dim) nogil
    ctypedef void (*gradientfunc)(double t, double *pars, double *q, int n_dim, int n_dim, double *grad) nogil
    ctypedef void (*hessianfunc)(double t, double *pars, double *q, int n_dim, int n_dim, double *hess) nogil

cdef extern from "potential/src/cpotential.h":
    enum:
        MAX_N_COMPONENTS = 16

    ctypedef struct CPotential:
        int n_components
        int n_dim
        densityfunc density[MAX_N_COMPONENTS]
        energyfunc value[MAX_N_COMPONENTS]
        gradientfunc gradient[MAX_N_COMPONENTS]
        hessianfunc hessian[MAX_N_COMPONENTS]
        int n_params[MAX_N_COMPONENTS]
        double *parameters[MAX_N_COMPONENTS]

cdef extern from "components.h":
    double growing_hernquist_value(double t, double *pars, double *q, int n_dim) nogil
    void growing_hernquist_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    double growing_hernquist_density(double t, double *pars, double *q, int n_dim) nogil

#

cdef class GrowingHernquistWrapper(CPotentialWrapper):

    def __init__(self, G, parameters):
        cdef CPotential cp

        # This is the only code that needs to change per-potential
        cp.value[0] = <energyfunc>(growing_hernquist_value)
        cp.density[0] = <densityfunc>(growing_hernquist_density)
        cp.gradient[0] = <gradientfunc>(growing_hernquist_gradient)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        cp.n_dim = 3
        cp.n_components = 1
        self._params = np.array([G] + list(parameters), dtype=np.float64)
        self._n_params = np.array([len(self._params)], dtype=np.int32)
        cp.n_params = &(self._n_params[0])
        cp.parameters[0] = &(self._params[0])

        self.cpotential = cp

class GrowingHernquistPotential(CPotentialBase):
    r"""
    GrowingHernquistPotential(m0, c0, units=None)

    Hernquist potential for a spheroid that grows with time.

    TODO:

    .. math::

        \Phi(r) = -\frac{G M}{r + c}

    See: http://adsabs.harvard.edu/abs/1990ApJ...356..359H

    Parameters
    ----------
    m0 : :class:`~astropy.units.Quantity`, numeric [mass]
        Mass at present day.
    c0 : :class:`~astropy.units.Quantity`, numeric [length]
        Core scale radius at present day.
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m0, c0, units=galactic):
        parameters = OrderedDict()
        ptypes = OrderedDict()

        parameters['m0'] = m0
        ptypes['m0'] = 'mass'

        parameters['c0'] = c0
        ptypes['c0'] = 'length'

        super(GrowingHernquistPotential, self).__init__(parameters=parameters,
                                                        parameter_physical_types=ptypes,
                                                        units=units,
                                                        Wrapper=GrowingHernquistWrapper)
