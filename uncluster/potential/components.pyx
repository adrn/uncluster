# coding: utf-8
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

from __future__ import division, print_function

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

cdef extern from "src/components.h":
    double growing_milkyway_value(double t, double *pars, double *q, int n_dim) nogil
    void growing_milkyway_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    double growing_milkyway_density(double t, double *pars, double *q, int n_dim) nogil

__all__ = ['GrowingMilkyWayPotential']

cdef class GrowingMilkyWayWrapper(CPotentialWrapper):

    def __init__(self, G, parameters):
        cdef CPotential cp

        # This is the only code that needs to change per-potential
        cp.value[0] = <energyfunc>(growing_milkyway_value)
        cp.density[0] = <densityfunc>(growing_milkyway_density)
        cp.gradient[0] = <gradientfunc>(growing_milkyway_gradient)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        cp.n_dim = 3
        cp.n_components = 1
        self._params = np.array([G] + list(parameters), dtype=np.float64)
        self._n_params = np.array([len(self._params)], dtype=np.int32)
        cp.n_params = &(self._n_params[0])
        cp.parameters[0] = &(self._params[0])

        self.cpotential = cp

class GrowingMilkyWayPotential(CPotentialBase):
    r"""
    GrowingMilkyWayPotential(m_n0, c_n0, m_h0, r_s, units=None)

    TODO

    Parameters
    ----------
    m_n0 : :class:`~astropy.units.Quantity`, numeric [mass]
        Nucleus mass at present day.
    c_n0 : :class:`~astropy.units.Quantity`, numeric [length]
        Nucleus scale radius at present day.
    m_h0 : :class:`~astropy.units.Quantity`, numeric [mass]
        Halo scale mass at present day.
    r_s : :class:`~astropy.units.Quantity`, numeric [length]
        Halo scale radius (fixed).
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m_n0, c_n0, m_h0, r_s, units=galactic):
        parameters = OrderedDict()
        ptypes = OrderedDict()

        parameters['m_n0'] = m_n0
        ptypes['m_n0'] = 'mass'

        parameters['c_n0'] = c_n0
        ptypes['c_n0'] = 'length'

        parameters['m_h0'] = m_h0
        ptypes['m_h0'] = 'mass'

        parameters['r_s'] = r_s
        ptypes['r_s'] = 'length'

        super(GrowingMilkyWayPotential, self).__init__(parameters=parameters,
                                                       parameter_physical_types=ptypes,
                                                       units=units,
                                                       Wrapper=GrowingMilkyWayWrapper)
