# coding: utf-8
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

__author__ = "adrn <adrn@princeton.edu>"

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
from ..potential import potential as pot

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

cdef double t_df(double r, double M, double vc, double ecc):
    cdef double f_e = 0.5 # HACK: this f_e is WRONG -- should be computed from orbit...
    return 64. * r*r * vc/283. * (2E5/M) * f_e # Gyr, Eq. 8

cdef void dy_dt(double M, double r2, double ecc, double t,
                gsl_interp_accel *acc, gsl_interp *spline,
                double *rgrid, double *vcgrid,
                double *out):
    cdef double tid, iso, min_t, vc

    if M <= 0 or r2 <= 0:
        out[0] = _NAN
        out[1] = _NAN
        return

    r = sqrt(r2)

    vc = gsl_interp_eval(spline, rgrid, vcgrid, r, acc)
    tid = t_tid(r, M, vc)
    iso = t_iso(M)

    if tid < iso:
        min_t = tid
    else:
        min_t = iso

    out[0] = -M / min_t
    out[1] = -r2 / t_df(r, M, vc, ecc)

cdef _solve_mass_radius(double M0, double r0, double ecc, double *t_grid, int n_times,
                        gsl_interp_accel *acc, gsl_interp *spline,
                        double *rgrid, double *vcgrid):
    cdef:
        double[::1] M = np.zeros(n_times)
        double[::1] r2 = np.zeros(n_times)
        double dt = t_grid[1] - t_grid[0]

        int i

        # container
        double[::1] y_dot = np.zeros(2)
        double[:,::1] q = np.zeros((1,3))

    # set initial conditions
    M[0] = M0
    r2[0] = r0**2 # solve for r^2 not r

    # use a forward Euler method instead...
    for i in range(n_times-1):
        dy_dt(M[i], r2[i], ecc, t_grid[i], acc, spline,
              rgrid, vcgrid, &y_dot[0])

        M[i+1] = M[i] + y_dot[0]*dt
        r2[i+1] = r2[i] + y_dot[1]*dt

        if isnan(y_dot[0]) or isnan(y_dot[1]) or M[i+1]<=0:
            break

    return i, np.array(M), np.sqrt(np.array(r2))

cpdef solve_mass_radius(_M0s, _r0s, _eccs, _t_grid):
    """
    solve_mass_radius(M0s, r0s, eccs, t_grid)
    """

    _M0s = np.atleast_1d(_M0s)
    _r0s = np.atleast_1d(_r0s)
    _eccs = np.atleast_1d(_eccs)
    _t_grid = np.atleast_1d(_t_grid)

    if _M0s.ndim > 1 or _r0s.ndim > 1 or _eccs.ndim > 1 or _t_grid.ndim > 1:
        raise ValueError("input arrays must be 1d")

    cdef:
        # make memoryviews
        double[::1] M0s = _M0s
        double[::1] r0s = _r0s
        double[::1] t_grid = _t_grid
        double[::1] eccs = _eccs
        int n_clusters = len(_M0s)
        int n_times = len(t_grid)
        int i

        int[::1] i_disrupt = np.zeros(n_clusters, dtype=np.int32)

        # set up interpolation stuff
        int n_grid = 1024
        double[::1] r_grid = np.logspace(-3., 3, n_grid)
        double[:,::1] _q = np.zeros((3,len(r_grid)))

        double[::1] vc_grid

        # GSL interpolation
        gsl_interp_accel *acc
        gsl_interp *vc_func

    _q[0] = r_grid
    vc_grid = pot.circular_velocity(_q).to(u.km/u.s).value

    acc = gsl_interp_accel_alloc()
    vc_func = gsl_interp_alloc(gsl_interp_linear, n_grid)
    gsl_interp_init(vc_func, &r_grid[0], &vc_grid[0], n_grid)

    all_M = []
    all_r = []
    for i in range(n_clusters):
        idx, M, r = _solve_mass_radius(M0s[i], r0s[i], eccs[i],
                                       &t_grid[0], n_times,
                                       acc, vc_func, &r_grid[0], &vc_grid[0])
        i_disrupt[i] = idx
        all_M.append(M)
        all_r.append(r)

    gsl_interp_free(vc_func)
    gsl_interp_accel_free(acc)

    return np.array(i_disrupt), all_M, all_r
