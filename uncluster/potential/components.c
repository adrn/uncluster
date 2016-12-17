#include <stdlib.h>
#include <math.h>
#include <potential/builtin/builtin_potentials.h>

/* ---------------------------------------------------------------------------
    Some global parameters that we set up here for speed
*/
static double const M_vir0 = 1e12; // Solar masses
static double const R_vir0 = 264.31575; // kpc, computed using Astropy for the above mass

/* ---------------------------------------------------------------------------
    Implements the function f* from Leitner 2012, which is related to the
    star formation rate at a given time. The one modification we make here is
    to slowly taper off the star formation rate after z=2.5 instead of the
    steep

    TODO: do I want t or z?
*/
double f_star(double t) {
    // TODO:
    return 1.;
}

/* Get the virial radius at time t */
double R_vir(double t) {
    // TODO:
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
    pars_t[1] = pars[1] + f_star(t);
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
    pars_t[1] = pars[1] + f_star(t);
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
    pars_t[1] = pars[1] + f_star(t);
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
    pars_t[1] = pars[1] + f_star(t);
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
    pars_t[1] = pars[1] + f_star(t);
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
    pars_t[1] = pars[1] + f_star(t);
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
    pars_t[1] = M_vir(t) / (np.log(c+1) - c/(c+1));

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
    pars_t[1] = M_vir(t) / (np.log(c+1) - c/(c+1));

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
    pars_t[1] = M_vir(t) / (np.log(c+1) - c/(c+1));

    return sphericalnfw_density(t, pars_t, q, n_dim);
}
