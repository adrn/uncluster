#include <stdlib.h>
#include <math.h>
#include <potential/builtin/builtin_potentials.h>

/* ---------------------------------------------------------------------------
    Hernquist sphere
*/
double growing_hernquist_value(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - c (length scale)
    */
    int n_pars = 3;// 3 parameters
    double *pars_t = (double*)malloc(sizeof(double)*n_pars);
    memcpy(pars_t, pars, sizeof(double)*n_pars);

    double dm_dt = 1000000.;
    pars_t[1] = pars[1] + dm_dt * t;

    return hernquist_value(t, pars_t, q, n_dim);
}

void growing_hernquist_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - c (length scale)
    */
    int n_pars = 3;// 3 parameters
    double *pars_t = (double*)malloc(sizeof(double)*n_pars);
    memcpy(pars_t, pars, sizeof(double)*n_pars);

    double dm_dt = 1000000.;
    pars_t[1] = pars[1] + dm_dt * t;

    hernquist_gradient(t, pars_t, q, n_dim, grad);
}

double growing_hernquist_density(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - c (length scale)
    */
    int n_pars = 3;// 3 parameters
    double *pars_t = (double*)malloc(sizeof(double)*n_pars);
    memcpy(pars_t, pars, sizeof(double)*n_pars);

    double dm_dt = 1000000.;
    pars_t[1] = pars[1] + dm_dt * t;

    return hernquist_density(t, pars_t, q, n_dim);
}

/* ---------------------------------------------------------------------------
    Spherical NFW
*/
// double growing_sphericalnfw_value(double t, double *pars, double *r) {
//     /*  pars:
//             - G (Gravitational constant)
//             - m0 (mass scale at z=0)
//             - r_s (scale radius)
//     */
//     double u, v_h2;
//     v_h2 = pars[1]*pars[1] / (log(2.) - 0.5);
//     u = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]) / pars[2];
//     return -v_h2 * log(1 + u) / u;
// }

// void growing_sphericalnfw_gradient(double t, double *pars, double *r, double *grad) {
//     /*  pars:
//             - G (Gravitational constant)
//             - m0 (mass scale at z=0)
//             - r_s (scale radius)
//     */
//     double fac, u, v_h2;
//     v_h2 = pars[1]*pars[1] / (log(2.) - 0.5);

//     u = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]) / pars[2];
//     fac = v_h2 / (u*u*u) / (pars[2]*pars[2]) * (log(1+u) - u/(1+u));

//     grad[0] = grad[0] + fac*r[0];
//     grad[1] = grad[1] + fac*r[1];
//     grad[2] = grad[2] + fac*r[2];
// }

// double growing_sphericalnfw_density(double t, double *pars, double *q) {
//     /*  pars:
//             - G (Gravitational constant)
//             - m0 (mass scale at z=0)
//             - r_s (scale radius)
//     */
//     double v_h2 = pars[1]*pars[1] / (log(2.) - 0.5);
//     double r, rho0;
//     r = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);

//     rho0 = v_h2 / (4*M_PI*pars[0]*pars[2]*pars[2]);
//     return rho0 / ((r/pars[2]) * pow(1+r/pars[2],2));
// }

// /* ---------------------------------------------------------------------------
//     Miyamoto-Nagai flattened potential
// */
// double growing_miyamotonagai_value(double t, double *pars, double *r) {
//     /*  pars:
//             - G (Gravitational constant)
//             - m0 (mass scale at z=0)
//             - a (length scale 1) TODO
//             - b (length scale 2) TODO
//     */
//     double zd;
//     zd = (pars[2] + sqrt(r[2]*r[2] + pars[3]*pars[3]));
//     return -pars[0] * pars[1] / sqrt(r[0]*r[0] + r[1]*r[1] + zd*zd);
// }

// void growing_miyamotonagai_gradient(double t, double *pars, double *r, double *grad) {
//     /*  pars:
//             - G (Gravitational constant)
//             - m0 (mass scale at z=0)
//             - a (length scale 1) TODO
//             - b (length scale 2) TODO
//     */
//     double sqrtz, zd, fac;

//     sqrtz = sqrt(r[2]*r[2] + pars[3]*pars[3]);
//     zd = pars[2] + sqrtz;
//     fac = pars[0]*pars[1] * pow(r[0]*r[0] + r[1]*r[1] + zd*zd, -1.5);

//     grad[0] = grad[0] + fac*r[0];
//     grad[1] = grad[1] + fac*r[1];
//     grad[2] = grad[2] + fac*r[2] * (1. + pars[2] / sqrtz);
// }

// double growing_miyamotonagai_density(double t, double *pars, double *q) {
//     /*  pars:
//             - G (Gravitational constant)
//             - m0 (mass scale at z=0)
//             - a (length scale 1) TODO
//             - b (length scale 2) TODO
//     */

//     double M, a, b;
//     M = pars[1];
//     a = pars[2];
//     b = pars[3];

//     double R2 = q[0]*q[0] + q[1]*q[1];
//     double sqrt_zb = sqrt(q[2]*q[2] + b*b);
//     double numer = (b*b*M / (4*M_PI)) * (a*R2 + (a + 3*sqrt_zb)*(a + sqrt_zb)*(a + sqrt_zb));
//     double denom = pow(R2 + (a + sqrt_zb)*(a + sqrt_zb), 2.5) * sqrt_zb*sqrt_zb*sqrt_zb;

//     return numer/denom;
// }
