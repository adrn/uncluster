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

cdef extern from "../src/cosmology.h":
    double f_star(double z) nogil
    double redshift(double t_lb) nogil
    double nu_relative_density(double z) nogil
    double inv_efunc_sq(double z) nogil
    double M_vir(double z) nogil
    double R_vir(double z) nogil

__all__ = ['']

cpdef _f_star(z):
    cdef:
        double[::1] _z = np.atleast_1d(z)
        int i
        int n = len(_z)
        double[::1] arr = np.zeros(n)

    for i in range(n):
        arr[i] = f_star(_z[i])

    return np.array(arr)

cpdef _redshift(tlb):
    cdef:
        double[::1] _tlb = np.atleast_1d(tlb)
        int i
        int n = len(_tlb)
        double[::1] arr = np.zeros(n)

    for i in range(n):
        arr[i] = redshift(_tlb[i])

    return np.array(arr)

cpdef _nu_relative_density(z):
    cdef:
        double[::1] _z = np.atleast_1d(z)
        int i
        int n = len(_z)
        double[::1] arr = np.zeros(n)

    for i in range(n):
        arr[i] = nu_relative_density(_z[i])

    return np.array(arr)

cpdef _inv_efunc_sq(z):
    cdef:
        double[::1] _z = np.atleast_1d(z)
        int i
        int n = len(_z)
        double[::1] arr = np.zeros(n)

    for i in range(n):
        arr[i] = inv_efunc_sq(_z[i])

    return np.array(arr)

cpdef _M_vir(z):
    cdef:
        double[::1] _z = np.atleast_1d(z)
        int i
        int n = len(_z)
        double[::1] arr = np.zeros(n)

    for i in range(n):
        arr[i] = M_vir(_z[i])

    return np.array(arr)

cpdef _R_vir(z):
    cdef:
        double[::1] _z = np.atleast_1d(z)
        int i
        int n = len(_z)
        double[::1] arr = np.zeros(n)

    for i in range(n):
        arr[i] = R_vir(_z[i])

    return np.array(arr)
