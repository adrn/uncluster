"""
1. Fit the potential model to Milky Way enclosed mass measurements compiled by Oleg Gnedin and
   Andreas Kuepper. The only free parameters are the halo mass, halo scale radius, nucleus mass,
   and nucleus scale radius. The best-fit values from this script are then dumped into
   uncluster/potential/core.py to be used for the rest of the project.

2. Generate arrays that specify the time-evolution of the potential parameters for fast
   interpolation later.

"""

# TODO: turn print statements into log statements

# Standard library
from os.path import exists, abspath, join

# Third-party
import astropy.cosmology as ac
from astropy.constants import G
from astropy.table import Table
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq
import gala.potential as gp
from gala.units import galactic
from scipy.interpolate import interp1d

##############################################################################
# These are the only configurable items - if you change them here, you must
# also change them in `uncluster/config.py`
#
cosmology = ac.Planck15
halo_concentration = 15. # MAGIC NUMBER - assumption
#
##############################################################################

# Project
def get_potential(log_M_h, log_r_s, log_M_n, log_a):
    mw_potential = gp.CCompositePotential()
    mw_potential['bulge'] = gp.HernquistPotential(m=5E9, c=1., units=galactic)
    mw_potential['disk'] = gp.MiyamotoNagaiPotential(m=6.8E10*u.Msun, a=3*u.kpc, b=280*u.pc,
                                                     units=galactic)
    mw_potential['nucl'] = gp.HernquistPotential(m=np.exp(log_M_n), c=np.exp(log_a)*u.pc,
                                                 units=galactic)
    mw_potential['halo'] = gp.NFWPotential(m=np.exp(log_M_h), r_s=np.exp(log_r_s), units=galactic)

    return mw_potential

def main(plot=False):
    data_path = abspath("../uncluster/data")
    if not exists(data_path):
        raise IOError("Could not find data path -- you should run this from the "
                      "'scripts' directory.")

    potential_path = abspath("../uncluster/potential/")
    if not exists(potential_path):
        raise IOError("Could not find path to potential subpackage -- you should "
                      "run this script from the 'scripts' directory.")

    ##########################################################################
    # 1. Get the best-fit potential parameters
    #

    # load tables of mass enclosed / circular velocity measurements
    gnedin_tbl = np.genfromtxt(join(data_path, "gnedin_mwmass_tbl.txt"),
                               delimiter=',', names=True, dtype=None)
    kuepper_tbl = np.genfromtxt(join(data_path, "kuepper15_mwmass_tbl.txt"),
                                delimiter=',', names=True, dtype=None)

    Menc1 = gnedin_tbl['Menc']
    Menc_l = gnedin_tbl['neg_err']
    Menc_u = gnedin_tbl['pos_err']
    Menc_err1 = np.max([Menc_u, Menc_l], axis=0) # HACK: take maximum uncertainty

    # Andreas compiled circular velocity measurements so need to turn into mass
    kuepper_tbl = kuepper_tbl[kuepper_tbl['source'].astype(str) != 'Gibbons et al. (2014)']
    Menc2 = (kuepper_tbl['radius']*u.kpc * (kuepper_tbl['v_c']*u.km/u.s)**2 / G).to(u.Msun).value
    Menc_l = Menc2 - (kuepper_tbl['radius']*u.kpc * ((kuepper_tbl['v_c']-kuepper_tbl['neg_err'])*u.km/u.s)**2 / G).to(u.Msun).value
    Menc_u = (kuepper_tbl['radius']*u.kpc * ((kuepper_tbl['v_c']+kuepper_tbl['pos_err'])*u.km/u.s)**2 / G).to(u.Msun).value - Menc2
    Menc_err2 = np.max([Menc_u, Menc_l], axis=0) # HACK: take maximum uncertainty

    # compile all mass measurements
    r = np.concatenate((gnedin_tbl['radius'], kuepper_tbl['radius']))
    Menc = np.concatenate((Menc1, Menc2))
    Menc_err = np.concatenate((Menc_err1, Menc_err2))

    # sort by radius and remove measurements at very small radii
    idx = r.argsort()
    obs_Menc = Menc[idx][2:]
    obs_Menc_err = Menc_err[idx][2:]
    obs_r = r[idx][2:]

    tbl = Table(dict(r=obs_r, Menc=obs_Menc, Menc_err=obs_Menc_err))
    tbl.write(join(data_path, 'apw_menc.txt'), format='csv')

    # Initial guess for the parameters:
    x0 = [np.log(6E11), np.log(20.), np.log(2E9), np.log(100.)] # a is in pc
    init_pot = get_potential(*x0)

    xyz = np.zeros((3, obs_r.size))
    xyz[0] = obs_r

    def f(p):
        pot = get_potential(*p)
        model_menc = pot.mass_enclosed(xyz).to(u.Msun).value
        return (model_menc - obs_Menc) / obs_Menc_err

    p_opt, ier = leastsq(f, x0=x0)
    assert ier in range(1,4+1)
    fit_potential = get_potential(*p_opt)

    # plot initial guess, fit profile:
    r = np.logspace(-3.5, 2.6, 1024)
    xyz = np.zeros((3,r.size))
    xyz[0] = r
    init_menc = init_pot.mass_enclosed(xyz*u.kpc)
    fit_menc = fit_potential.mass_enclosed(xyz*u.kpc)

    if plot:
        # observational points
        fig,ax = plt.subplots(1,1,figsize=(6,6))

        plt.errorbar(obs_r, obs_Menc, yerr=obs_Menc_err, marker='o', markersize=4,
                     color='#3182bd', alpha=1., ecolor='#9ecae1', capthick=0,
                     linestyle='none', elinewidth=1.)
        ax.loglog(r, init_menc.value, marker='', color="#444444",
                  linewidth=2, alpha=0.7, linestyle='--')
        ax.loglog(r, fit_menc.value, marker='', color="#444444",
                  linewidth=2, alpha=0.7)

        ax.set_xlim(5E-3, 10**2.6)
        ax.set_ylim(7E6, 10**12.25)

        ax.set_xlabel('$r$ [kpc]')
        ax.set_ylabel('$M(<r)$ [M$_\odot$]')

        fig.tight_layout()

        plt.show()

    m_h = np.exp(p_opt[0])
    r_s = np.exp(p_opt[1])
    m_n = np.exp(p_opt[2])
    a = np.exp(p_opt[3])

    lines = []
    lines.append("# vvv --- THIS IS AUTO-GENERATED CODE - see scripts/setup-potential.py --- vvv\n")
    lines.append("import astropy.units as u\n")
    lines.append("m_h = {:.7e} * u.Msun\n".format(m_h))
    lines.append("r_s = {:.7f} * u.kpc\n".format(r_s))
    lines.append("m_n = {:.7e} * u.Msun\n".format(m_n))
    lines.append("c_n = {:.7f} * u.pc\n".format(a))
    lines.append("# ^^^ --- THIS IS AUTO-GENERATED CODE - see scripts/setup-potential.py --- ^^^")

    with open(join(potential_path, "potential_config.py"), "w") as f:
        f.writelines(lines)

    ##########################################################################
    # 2. Now solve for the cosmological evolution of the parameters:
    #
    def F(c):
        return np.log(1 + c) - c/(1+c)

    # Global configuration
    M_vir0 = F(halo_concentration) * fit_potential['halo'].parameters['m']
    print("Mvir at z=0: {:.6e} Msun".format(M_vir0))
    print(halo_concentration, F(halo_concentration))

    lines = [
        '// vvv --- THIS IS AUTO-GENERATED CODE - see scripts/setup-potential.py --- vvv\n',
        'static double const M_vir0 = {:.8e};\n'.format(M_vir0.to(u.Msun).value),
        '// ^^^ --- THIS IS AUTO-GENERATED CODE - see scripts/setup-potential.py --- ^^^'
    ]
    with open(join(potential_path, "src", "cosmo_helper.h"), "w") as f:
        f.writelines(lines)

if __name__ == "__main__":
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")

    vq_group = parser.add_mutually_exclusive_group()
    parser.add_argument('-p', '--plot', action='store_true', dest='plot',
                        default=False, help='Show a plot of mass profile over data.')

    args = parser.parse_args()

    main(**vars(args))

