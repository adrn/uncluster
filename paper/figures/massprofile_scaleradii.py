# Standard library
from os.path import exists, abspath, join
import sys

# Third-party
from astropy.io import ascii
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

# Project
from uncluster.potential import mw_potential
from uncluster.config import cosmology

linestyle = dict(marker='', linewidth=2, alpha=0.7) # color="#444444",
datastyle = dict(marker='o', markersize=4, color='#3182bd', alpha=1.,
                 ecolor='#9ecae1', capthick=0, linestyle='none', elinewidth=1.)

def main():
    data_path = abspath("../data")
    figure_path = abspath("../paper/figures")
    if not exists(data_path) or not exists(figure_path):
        raise IOError("Could not find path -- you should run this from the "
                      "'scripts' directory.")

    tbl = ascii.read(join(data_path, 'apw_menc.txt'))

    r = np.logspace(-3.5, 2.6, 1024)
    xyz = np.zeros((3,r.size))
    xyz[0] = r
    xyz = xyz * u.kpc

    # observational points
    fig,axes = plt.subplots(1, 2, figsize=(8,4))

    # panel 1
    ax = axes[0]
    ax.errorbar(tbl['r'], tbl['Menc'], yerr=tbl['Menc_err'], **datastyle)
    menc = mw_potential.mass_enclosed(xyz, t=0.)
    ax.loglog(r, menc.value, **linestyle)

    ax.set_xlim(5E-3, 10**2.6)
    ax.set_ylim(7E6, 10**12.25)

    ax.set_xlabel('$r$ [kpc]')
    ax.set_ylabel('$M(<r)$ [M$_\odot$]')

    # panel 2
    r_s = mw_potential['halo'].parameters['r_s'].to(u.kpc).value
    xyz = [r_s, 0., 0.] * u.kpc
    ts = np.linspace(0, -12., 256) * u.Gyr
    Mencs = np.zeros_like(ts.value) * u.Msun
    for i,t in enumerate(ts):
        Mencs[i] = mw_potential.mass_enclosed(xyz, t=t)
    axes[1].semilogy(ts.value, Mencs.value, **linestyle)
    axes[1].set_xlabel(r'$t_{\rm lookback}$ [Gyr]')
    axes[1].set_ylabel('$M(<r_s)$ [M$_\odot$]')

    fig.tight_layout()

    # plt.show()
    fig.savefig(join(figure_path, 'mass-profile.pdf'))

if __name__ == "__main__":
    main()

