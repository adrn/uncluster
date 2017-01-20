#include <stdlib.h>
#include <math.h>
#include <potential/builtin/builtin_potentials.h>
#include "cosmology.h"

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

    return hernquist_value(t, pars_t, q, n_dim);
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

    return hernquist_density(t, pars_t, q, n_dim);
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

    return miyamotonagai_value(t, pars_t, q, n_dim);
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

    return miyamotonagai_density(t, pars_t, q, n_dim);
}


/* ---------------------------------------------------------------------------
    Spherical NFW
*/
double growing_sphericalnfw_value(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G - Gravitational constant
            - m0 - mass scale at z=0
            - rs - scale radius (constant)
    */
    double c = R_vir(t) / pars[2];

    int n_pars = 3;
    double *pars_t = (double*)malloc(sizeof(double)*n_pars);
    memcpy(pars_t, pars, sizeof(double)*n_pars);
    double z = redshift(t);
    pars_t[1] = M_vir(z) / (log(c+1) - c/(c+1));

    return sphericalnfw_value(t, pars_t, q, n_dim);
}

void growing_sphericalnfw_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
    /*  pars:
            - G - Gravitational constant
            - m0 - mass scale at z=0
            - rs - scale radius (constant)
    */
    double c = R_vir(t) / pars[2];

    int n_pars = 3;
    double *pars_t = (double*)malloc(sizeof(double)*n_pars);
    memcpy(pars_t, pars, sizeof(double)*n_pars);
    double z = redshift(t);
    pars_t[1] = M_vir(z) / (log(c+1) - c/(c+1));

    sphericalnfw_gradient(t, pars_t, q, n_dim, grad);
}

double growing_sphericalnfw_density(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G - Gravitational constant
            - m0 - mass scale at z=0
            - rs - scale radius (constant)
    */

    double c = R_vir(t) / pars[2];

    int n_pars = 3;
    double *pars_t = (double*)malloc(sizeof(double)*n_pars);
    memcpy(pars_t, pars, sizeof(double)*n_pars);
    double z = redshift(t);
    pars_t[1] = M_vir(z) / (log(c+1) - c/(c+1));

    return sphericalnfw_density(t, pars_t, q, n_dim);
}
