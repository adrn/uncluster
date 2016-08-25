# coding: utf-8
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
cimport numpy as np
np.import_array()
import cython
cimport cython

# Project
from .gnedin_mass_radius import _sersic_frac_mass_enclosed_grid

# _sersic_frac_mass_enclosed_grid(rmax)

cdef extern from "gsl/gsl_interp.h":

    ctypedef struct gsl_interp_accel:
        size_t  cache
        size_t  miss_count
        size_t  hit_count

    ctypedef struct gsl_interp_type:
        char* name
        unsigned int min_size
        # skip the rest of the fields for now

    gsl_interp_type * gsl_interp_linear

    ctypedef struct gsl_interp:
        gsl_interp_type * type
        double  xmin
        double  xmax
        size_t  size
        void * state

    gsl_interp_accel * gsl_interp_accel_alloc()
    int gsl_interp_accel_reset (gsl_interp_accel * a)
    void gsl_interp_accel_free(gsl_interp_accel * a)

    gsl_interp * gsl_interp_alloc(gsl_interp_type* T, int n)
    void gsl_interp_free(gsl_interp * interp)

    double gsl_interp_eval(gsl_interp * obj, double* xa, double* ya, double x,
                           gsl_interp_accel * a)

    int gsl_interp_init(gsl_interp * obj, double* xa, double* ya, size_t size)
    char * gsl_interp_name(gsl_interp * interp)

def test():

    cdef:
        double[::1] x = np.array([1970, 1980, 1990, 2000])
        double[::1] y = np.array([12,   11,   14,   13])
        size_t size = 4

    # inialise and allocate the gsl objects
    gsl_interp *interpolation = gsl_interp_alloc(gsl_interp_linear, size)
    gsl_interp_init(interpolation, &x[0], &y[0], size)
    gsl_interp_accel * accelerator =  gsl_interp_accel_alloc()

    # get interpolation for x = 1981
    value = gsl_interp_eval(interpolation, &x[0], &y[0], 1981, accelerator)
    print(value)
