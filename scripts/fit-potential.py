"""
Fit the potential model to Milky Way enclosed mass measurements compiled by
Oleg Gnedin and Andreas Kuepper
"""

# Standard library
from os.path import exists, abspath, join

# Third-party
from astropy.constants import G
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq
import gala.potential as gp
from gala.units import galactic

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
    data_path = abspath("../data")
    if not exists(data_path):
        raise IOError("Could not find data path -- you should run this from the "
                      "'scripts' directory.")

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

    print("Best fit values (as code, to put in uncluster/potential/core.py):")
    print("-----------------------------------------------------------------")
    print("m_h = {:.2e} * u.Msun".format(m_h))
    print("r_s = {:.2f} * u.kpc".format(r_s))
    print("m_n = {:.2e} * u.Msun".format(m_n))
    print("c_n = {:.2f} * u.pc".format(a))

if __name__ == "__main__":
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")

    vq_group = parser.add_mutually_exclusive_group()
    parser.add_argument('-p', '--plot', action='store_true', dest='plot',
                        default=False, help='Show a plot of mass profile over data.')

    args = parser.parse_args()

    main(**vars(args))

