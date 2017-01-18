*********
Uncluster
*********

The broad overview of what happens here is:

1. Choose radial and mass distributions for the initial globular cluster
   population. (The radial distribution / density profile must be a
   :class:`~gala.potential.PotentialBase` subclass because we'll use it later to
   sample from a DF.) Generate samples from these distributions so that the
   total mass in globular clusters is some fraction of the total stellar mass of
   the Galaxy, generate (mean) orbital radii by sampling from the radial
   distribution.
2. Choose a DF to use (spherical isotropic, radially biased, tangentially
   biased, or accreted with a few satellite galaxies) and generate velocities
   for each cluster to match the DF embedded in a given background
   potential. Integrate the orbits of the clusters in the total background
   potential and determine eccentricities.
3. Solve for the mass-loss history of the cluster using the differential
   equations in Gnedin et al. (2014), including the eccentricities measured
   from the integrated cluster orbits. Run streakline-like models along each
   cluster orbit and post-process the samples to impose the solved mass-loss
   histories.
4. Paint stellar populations (and abundances???) on to the star particles in
   each stream. Generate mock images with a background...

Starting from scratch
=====================

Here I'll write all of the steps needed to reproduce the results starting from a
clean clone of this repository.

1. The first thing to do, even before installing the package, is to setup the
   metadata needed for the Milky Way mass model. Two things need to happen: (a)
   we need to generate best-fit parameters for the halo and nucleus components
   (we fix the disk and bulge models), and (b) we need to generate C arrays that
   specify the time evolution of the different potential parameters so we can
   interpolate to get the potential at any given time. Both of these are handled
   by the same script, ``setup_potential.py``. Run this script from the
   ``scripts`` directory. This will generate some code and place two files in
   the project directory: ``uncluster/potential/potential_config.py`` and
   ``uncluster/potential/src/cosmo_arrays.h``.

   Now you should install the package using::

      python setup.py install

2.

API
===

.. automodapi:: uncluster

.. automodapi:: uncluster.cluster_distributions.apw

.. automodapi:: uncluster.cluster_distributions.gnedin

.. automodapi:: uncluster.cluster_massloss
