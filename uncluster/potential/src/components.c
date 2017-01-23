#include <stdlib.h>
#include <math.h>
#include <potential/builtin/builtin_potentials.h>
#include "cosmology.h"

// TODO: make this a PR, note so slow, remove most of this shite and define
//       potential parameters hard-set in header file.

double growing_milkyway_value(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G - Gravitational constant
            - m_n0 - nucleus mass scale at z=0
            - c0 - nucleus scale radius at z=0
            - m_h0 - halo mass scale at z=0
            - r_s - halo scale radius (fixed)
    */
    double z = redshift(t);

    int n_pars = 3;// 3 parameters for each
    double *pars_t = (double*)malloc(sizeof(double)*n_pars);

    pars_t[1] = pars[1] * f_star(z);
    pars_t[2] = pars[2] * R_vir(z) / R_vir(0);

    double val = hernquist_value(t, pars_t, q, n_dim);
    free(pars_t);
    return val;
}


/* ===========================================================================
    The stuff below is too slow for generating mock streams because the
    cosmological functions are evaluated 4 times per gradient call!
*/

/* ---------------------------------------------------------------------------
    Hernquist sphere
*/
double growing_hernquist_value(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G - Gravitational constant
            - m0 - mass scale at z=0
            - c0 - length scale at z=0
    */
    int n_pars = 3;// 3 parameters
    double *pars_t = (double*)malloc(sizeof(double)*n_pars);
    memcpy(pars_t, pars, sizeof(double)*n_pars);
    double z = redshift(t);
    pars_t[1] = pars[1] * f_star(z);
    pars_t[2] = pars[2] * R_vir(z) / R_vir(0);

    double val = hernquist_value(t, pars_t, q, n_dim);
    free(pars_t);
    return val;
}

void growing_hernquist_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
    /*  pars:
            - G - Gravitational constant
            - m0 - mass scale at z=0
            - c0 - length scale at z=0
    */
    int n_pars = 3;// 3 parameters
    double *pars_t = (double*)malloc(sizeof(double)*n_pars);
    memcpy(pars_t, pars, sizeof(double)*n_pars);
    double z = redshift(t);
    pars_t[1] = pars[1] * f_star(z);
    pars_t[2] = pars[2] * R_vir(z) / R_vir(0);

    hernquist_gradient(t, pars_t, q, n_dim, grad);
    free(pars_t);
}

double growing_hernquist_density(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G - Gravitational constant
            - m0 - mass scale at z=0
            - c0 - length scale at z=0
    */
    int n_pars = 3;// 3 parameters
    double *pars_t = (double*)malloc(sizeof(double)*n_pars);
    memcpy(pars_t, pars, sizeof(double)*n_pars);
    double z = redshift(t);
    pars_t[1] = pars[1] * f_star(z);
    pars_t[2] = pars[2] * R_vir(z) / R_vir(0);

    double val = hernquist_density(t, pars_t, q, n_dim);
    free(pars_t);
    return val;
}

/* ---------------------------------------------------------------------------
    Miyamoto-Nagai flattened potential
*/
double growing_miyamotonagai_value(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G - Gravitational constant
            - m0 - mass scale at z=0
            - a0 - radial scale length at z=0
            - b - vertical scale length (constant)
    */
    int n_pars = 4;
    double *pars_t = (double*)malloc(sizeof(double)*n_pars);
    memcpy(pars_t, pars, sizeof(double)*n_pars);
    double z = redshift(t);
    pars_t[1] = pars[1] * f_star(z);
    pars_t[2] = pars[2] * R_vir(z) / R_vir(0);

    double val = miyamotonagai_value(t, pars_t, q, n_dim);
    free(pars_t);
    return val;
}

void growing_miyamotonagai_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
    /*  pars:
            - G - Gravitational constant
            - m0 - mass scale at z=0
            - a0 - radial scale length at z=0
            - b - vertical scale length (constant)
    */
    int n_pars = 4;
    double *pars_t = (double*)malloc(sizeof(double)*n_pars);
    memcpy(pars_t, pars, sizeof(double)*n_pars);
    double z = redshift(t);
    pars_t[1] = pars[1] * f_star(z);
    pars_t[2] = pars[2] * R_vir(z) / R_vir(0);

    miyamotonagai_gradient(t, pars_t, q, n_dim, grad);
    free(pars_t);
}

double growing_miyamotonagai_density(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G - Gravitational constant
            - m0 - mass scale at z=0
            - a0 - radial scale length at z=0
            - b - vertical scale length (constant)
    */

    int n_pars = 4;
    double *pars_t = (double*)malloc(sizeof(double)*n_pars);
    memcpy(pars_t, pars, sizeof(double)*n_pars);
    double z = redshift(t);
    pars_t[1] = pars[1] * f_star(z);
    pars_t[2] = pars[2] * R_vir(z) / R_vir(0);

    double val = miyamotonagai_density(t, pars_t, q, n_dim);
    free(pars_t);
    return val;
}


/* ---------------------------------------------------------------------------
    Spherical NFW

    Note: because R_vir() uses an approximation.
*/
double growing_sphericalnfw_value(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G - Gravitational constant
            - m0 - mass scale at z=0
            - rs - scale radius (constant)
    */
    int n_pars = 3;
    double *pars_t = (double*)malloc(sizeof(double)*n_pars);
    memcpy(pars_t, pars, sizeof(double)*n_pars);
    double z = redshift(t);
    double c = R_vir(z) / pars[2];
    pars_t[1] = M_vir(z) / (log(c+1) - c/(c+1));

    double val = sphericalnfw_value(t, pars_t, q, n_dim);
    free(pars_t);
    return val;
}

void growing_sphericalnfw_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
    /*  pars:
            - G - Gravitational constant
            - m0 - mass scale at z=0
            - rs - scale radius (constant)
    */
    int n_pars = 3;
    double *pars_t = (double*)malloc(sizeof(double)*n_pars);
    memcpy(pars_t, pars, sizeof(double)*n_pars);
    double z = redshift(t);
    double c = R_vir(z) / pars[2];
    pars_t[1] = M_vir(z) / (log(c+1) - c/(c+1));

    sphericalnfw_gradient(t, pars_t, q, n_dim, grad);
    free(pars_t);
}

double growing_sphericalnfw_density(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G - Gravitational constant
            - m0 - mass scale at z=0
            - rs - scale radius (constant)
    */
    int n_pars = 3;
    double *pars_t = (double*)malloc(sizeof(double)*n_pars);
    memcpy(pars_t, pars, sizeof(double)*n_pars);
    double z = redshift(t);
    double c = R_vir(z) / pars[2];
    pars_t[1] = M_vir(z) / (log(c+1) - c/(c+1));

    double val = sphericalnfw_density(t, pars_t, q, n_dim);
    free(pars_t);
    return val;
}
