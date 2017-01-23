#include <stdlib.h>
#include <math.h>
#include <potential/builtin/builtin_potentials.h>
#include "components.h"
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
    double val = 0;
    double z = redshift(t);
    double f_starz = f_star(z);
    double R_virz = R_vir(z);
    double R_vir_frac = R_virz / R_vir(0);

    int n_pars = 4;// max num of params is 4 (for Miyamoto-Nagai)
    double *pars_t = (double*)malloc(sizeof(double)*n_pars);

    // Bulge
    pars_t[0] = pars[0]; // G
    pars_t[1] = bulge_pars[1] * f_starz;
    pars_t[2] = bulge_pars[2] * R_vir_frac;
    val = val + hernquist_value(t, pars_t, q, n_dim);

    // Disk
    pars_t[1] = disk_pars[1] * f_starz;
    pars_t[2] = disk_pars[2] * R_vir_frac;
    pars_t[3] = disk_pars[3]; // scale-height time independent
    val = val + miyamotonagai_value(t, pars_t, q, n_dim);

    // Nucleus
    pars_t[1] = pars[1] * f_starz;
    pars_t[2] = pars[2] * R_vir_frac;
    val = val + hernquist_value(t, pars_t, q, n_dim); // nucleus

    // Halo
    double c = R_virz / pars[4];
    pars_t[1] = M_vir(z) / (log(c+1) - c/(c+1));
    pars_t[2] = pars[4]; // NFW scale radius time independent
    val = val + sphericalnfw_value(t, pars_t, q, n_dim); // nucleus

    free(pars_t);
    return val;
}

void growing_milkyway_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
    /*  pars:
            - G - Gravitational constant
            - m_n0 - nucleus mass scale at z=0
            - c0 - nucleus scale radius at z=0
            - m_h0 - halo mass scale at z=0
            - r_s - halo scale radius (fixed)
    */
    double val = 0;
    double z = redshift(t);
    double f_starz = f_star(z);
    double R_virz = R_vir(z);
    double R_vir_frac = R_virz / R_vir(0);

    int n_pars = 4;// max num of params is 4 (for Miyamoto-Nagai) - others will ignore
    double *pars_t = (double*)malloc(sizeof(double)*n_pars);

    // Bulge
    pars_t[0] = pars[0]; // G
    pars_t[1] = bulge_pars[1] * f_starz;
    pars_t[2] = bulge_pars[2] * R_vir_frac;
    hernquist_gradient(t, pars_t, q, n_dim, grad);

    // Disk
    pars_t[1] = disk_pars[1] * f_starz;
    pars_t[2] = disk_pars[2] * R_vir_frac;
    pars_t[3] = disk_pars[3]; // scale-height time independent
    miyamotonagai_gradient(t, pars_t, q, n_dim, grad);

    // Nucleus
    pars_t[1] = pars[1] * f_starz;
    pars_t[2] = pars[2] * R_vir_frac;
    hernquist_gradient(t, pars_t, q, n_dim, grad); // nucleus

    // Halo
    double c = R_virz / pars[4];
    pars_t[1] = M_vir(z) / (log(c+1) - c/(c+1));
    pars_t[2] = pars[4]; // NFW scale radius time independent
    sphericalnfw_gradient(t, pars_t, q, n_dim, grad); // nucleus

    free(pars_t);
}

double growing_milkyway_density(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G - Gravitational constant
            - m_n0 - nucleus mass scale at z=0
            - c0 - nucleus scale radius at z=0
            - m_h0 - halo mass scale at z=0
            - r_s - halo scale radius (fixed)
    */
    double val = 0;
    double z = redshift(t);
    double f_starz = f_star(z);
    double R_virz = R_vir(z);
    double R_vir_frac = R_virz / R_vir(0);

    int n_pars = 4;// max num of params is 4 (for Miyamoto-Nagai)
    double *pars_t = (double*)malloc(sizeof(double)*n_pars);

    // Bulge
    pars_t[0] = pars[0]; // G
    pars_t[1] = bulge_pars[1] * f_starz;
    pars_t[2] = bulge_pars[2] * R_vir_frac;
    val = val + hernquist_density(t, pars_t, q, n_dim);

    // Disk
    pars_t[1] = disk_pars[1] * f_starz;
    pars_t[2] = disk_pars[2] * R_vir_frac;
    pars_t[3] = disk_pars[3]; // scale-height time independent
    val = val + miyamotonagai_density(t, pars_t, q, n_dim);

    // Nucleus
    pars_t[1] = pars[1] * f_starz;
    pars_t[2] = pars[2] * R_vir_frac;
    val = val + hernquist_density(t, pars_t, q, n_dim); // nucleus

    // Halo
    double c = R_virz / pars[4];
    pars_t[1] = M_vir(z) / (log(c+1) - c/(c+1));
    pars_t[2] = pars[4]; // NFW scale radius time independent
    val = val + sphericalnfw_density(t, pars_t, q, n_dim); // nucleus

    free(pars_t);
    return val;
}
