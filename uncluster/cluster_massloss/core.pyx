# coding: utf-8
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

# Third-party
import astropy.units as u
import numpy as np
cimport numpy as np
np.import_array()
import cython
cimport cython
from cython_gsl cimport *

from libc.math cimport sqrt, isnan

# Project
from ..potential import mw_potential as pot

cdef double _NAN = float("nan")

# Functions used to solve the differential equation for cluster mass M(t)
cdef double P(double r, double vc):
    return 207. * r / vc / 5. # Eq. 4

cdef double t_tid(double r, double M, double vc):
    """ Tidal disruption timescale """
    cdef double alpha = 2/3.
    return 10. * (M / 2E5)**alpha * P(r, vc) # Gyr, Eq. 4

cdef double t_iso(double M):
    """ Isolation disruption timescale (2-body evaporation)"""
    return 17. * (M / 2E5) # Gyr, Eq. 5

cdef void dy_dt(double M, double t, double r, double vc, double *out):
    cdef double tid, iso, min_t

    if M <= 0:
        out[0] = _NAN
        return

    tid = t_tid(r, M, vc)
    iso = t_iso(M)

    if tid < iso:
        min_t = tid
    else:
        min_t = iso

    out[0] = -M / min_t

cdef _solve_mass_evolution(double M0, double *t_grid, int n_times,
                           double *r_grid, double *vc_grid):
    cdef:
        double[::1] M = np.zeros(n_times)
        double dt = t_grid[1] - t_grid[0]
        int i

        # containers
        double[::1] y_dot = np.zeros(2)
        double[:,::1] q = np.zeros((1,3))

    # set initial conditions
    M[0] = M0

    # use a forward Euler method instead...
    for i in range(n_times-1):
        dy_dt(M[i], t_grid[i],
              r_grid[i], vc_grid[i], &y_dot[0])

        M[i+1] = M[i] + y_dot[0]*dt

        if isnan(y_dot[0]) or M[i+1]<=0:
            break

    return i, np.array(M)

cpdef solve_mass_evolution(_M0, _t_grid, _r_grid):
    """
    solve_mass_evolution(M0, t_grid, r_grid)
    """

    _t_grid = np.atleast_1d(_t_grid)
    _r_grid = np.atleast_1d(_r_grid)

    if _t_grid.ndim > 1 or _r_grid.ndim > 1:
        raise ValueError("input arrays must be 1d")

    if len(_t_grid) != len(_r_grid):
        raise ValueError("Time and radius grid must have same shape.")

    cdef:
        double M0 = float(_M0)

        # make memoryviews
        double[::1] t_grid = _t_grid
        double[::1] r_grid = _r_grid
        int n_times = len(t_grid)
        int i

        int disrupt_idx

        # compute vc at all radii, times
        double[:,::1] _q = np.zeros((3,n_times))
        double[::1] vc_grid

    for i in range(n_times):
        _q[0,i] = r_grid[i]
    vc_grid = pot.circular_velocity(_q, t=_t_grid).to(u.km/u.s).value

    disrupt_idx, M = _solve_mass_evolution(M0, &t_grid[0], n_times,
                                           &r_grid[0], &vc_grid[0])

    return disrupt_idx, np.array(M)
