#include <stdlib.h>
#include <math.h>
#include <potential/builtin/builtin_potentials.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include "cosmo_arrays.h"

/* -------- */
double f_star(double t) {
    double ret;
    gsl_interp *intp = gsl_interp_alloc(gsl_interp_linear, interp_grid_size);

    (void) gsl_interp_init(intp, &t_grid, &fstar_grid, interp_grid_size);
    ret = gsl_interp_eval(intp, &t_grid, &fstar_grid, t, NULL);
    gsl_interp_free(intp);

    return ret;
}

/* Get the virial radius, mass at time t */
double R_vir(double t) {
    double ret;
    gsl_interp *intp = gsl_interp_alloc(gsl_interp_linear, interp_grid_size);

    (void) gsl_interp_init(intp, &t_grid, &R_vir_grid, interp_grid_size);
    ret = gsl_interp_eval(intp, &t_grid, &R_vir_grid, t, NULL);
    gsl_interp_free(intp);

    return ret;
}

double M_vir(double t) {
    double ret;
    gsl_interp *intp = gsl_interp_alloc(gsl_interp_linear, interp_grid_size);

    (void) gsl_interp_init(intp, &t_grid, &M_vir_grid, interp_grid_size);
    ret = gsl_interp_eval(intp, &t_grid, &M_vir_grid, t, NULL);
    gsl_interp_free(intp);

    return ret;
}

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
    pars_t[1] = pars[1] * f_star(t);
    pars_t[2] = pars[2] * R_vir(t) / R_vir0;

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
    pars_t[1] = pars[1] * f_star(t);
    pars_t[2] = pars[2] * R_vir(t) / R_vir0;

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
    pars_t[1] = pars[1] * f_star(t);
    pars_t[2] = pars[2] * R_vir(t) / R_vir0;

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
    pars_t[1] = pars[1] * f_star(t);
    pars_t[2] = pars[2] * R_vir(t) / R_vir0;

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
    pars_t[1] = pars[1] * f_star(t);
    pars_t[2] = pars[2] * R_vir(t) / R_vir0;

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
    pars_t[1] = pars[1] * f_star(t);
    pars_t[2] = pars[2] * R_vir(t) / R_vir0;

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
    pars_t[1] = M_vir(t) / (log(c+1) - c/(c+1));

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
    pars_t[1] = M_vir(t) / (log(c+1) - c/(c+1));

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
    pars_t[1] = M_vir(t) / (log(c+1) - c/(c+1));

    return sphericalnfw_density(t, pars_t, q, n_dim);
}
