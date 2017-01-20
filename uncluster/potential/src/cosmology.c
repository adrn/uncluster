#include <math.h>
#include "cosmology.h"

double f_star(double z) {
    /*
        This is based on a fit to the Behroozi+ 2013 fractional stellar
        mass vs. redshift relation. The error is <1% for z<2 and is
        ~10% at z=2-3.
    */
    return _fs_a / (1 + exp(-_fs_b*(z-_fs_c))) * exp(_fs_f*(z-_fs_g)) + _fs_d;
}

double redshift(double t_lb) {
    /*
        Lookback time (in Myr) to redshift - this is also an approximation
        ignoring radiation energy. Even at z=4, it's only ~1% wrong.
    */
    return pow(sqrt(Om0/Ode0) * sinh(1.5*sqrt(Ode0)*(t_lb/t_H - C)), -0.6666666667) - 1.;
}

double nu_relative_density(double z) {
    /*
        Neutrino density function relative to the energy density in photons.
        Ripped out of astropy.cosmology and hard-set values for Planck 2015
        cosmological parameters.
    */
    double curr_nu_y = nu_y / (1. + z);
    double rel_mass_per = pow(1.0 + pow(k * curr_nu_y, p), invp);
    double rel_mass = rel_mass_per + nmasslessnu;
    return prefac * neff_per_nu * rel_mass;
}

double inv_efunc_sq(double z) {
    double Or = Ogamma0 * (1 + nu_relative_density(z));
    double zp1 = 1 + z;
    return 1/(zp1*zp1*zp1 * (Or * zp1 + Om0) + Ode0);
}

double Om(double z) {
    double zp1 = 1 + z;
    return Om0 * zp1*zp1*zp1 * inv_efunc_sq(z);
}

double Ode(double z) {
    return Ode0 * inv_efunc_sq(z);
}

double Delta(double z) {
    /* An approximation thanks to Dekel & Birnboim 2006 (see appendix) */
    return (18*M_PI*M_PI - 82*Ode(z) - 39*pow(Ode(z),2)) / Om(z);
}

double M_vir(double z) {
    return M_vir0 * exp(-0.8*z);
}

double R_vir(double z) {
    double _Delta = pow(Delta(z) / 200, -1/3.);
    double _Om = pow(Om(z) / 0.3, -1/3.);
    double _h2 = pow(_h / 0.7, -2/3.);
    return 309. * pow(M_vir(z)/1E12,1/3.) * _Delta * _Om * _h2 / (1 + z);
}
