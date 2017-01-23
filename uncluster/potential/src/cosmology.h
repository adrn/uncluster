#include <math.h>
#include "cosmo_helper.h"

/*
    These parameters are based on a fit to the Behroozi+ 2013 fractional stellar
    mass vs. redshift relation. The error is <1% for z<2 and is ~10% at z=2-3.
*/
static const double _fs_a = -0.99209053;
static const double _fs_b = 3.1559676;
static const double _fs_c = 0.7968646;
static const double _fs_d = 1.08458397;
static const double _fs_f = 0.01762417;
static const double _fs_g = -1.27356273;

/*
    These are for Planck 2015 cosmological parameters!
*/
static const double Om0 = 0.3075; // Omega matter
static const double Ode0 = 0.69100993445944359; // Omega dark energy
static const double Ogamma0 = 5.388890478958946e-05; // Omega photons
static const double t_H = 14434.48806721477; // Myr
static const double _h = 0.6774; // hubble parameter
static const double C = -0.95776556521317;
static const double _rfac1 = 0.66708383142120198; // sqrt(Om0/Ode0)
static const double _rfac2 = 1.2469051096750499; // 1.5*sqrt(Ode0)

/*
    The following definitions are nedded for neutrino density. See
    astropy.cosmology.core comments in nu_relative_density() for more info
*/
static const double prefac = 0.22710731766;  // 7/8 (4/11)^4/3 -- see any cosmo book
static const double p = 1.83;
static const double invp = 0.54644808743;
static const double k = 0.3173;
static const double nu_y = 357.91215740033334;
static const double nmasslessnu = 2;
static const double neff_per_nu = 1.0153333333;

// Functions:
double f_star(double z);
double redshift(double t_lb);
double nu_relative_density(double z);
double inv_efunc_sq(double z);
double Om(double z);
double Ode(double z);
double Delta(double z);
double M_vir(double z);
double R_vir(double z);
