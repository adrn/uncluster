Notes
=====

2016-08-16
----------

Scope of the paper should be:

* Predict number of globular cluster streams, lengths, densities
    - I'm a bit lost on how to do this. This somewhat requires matched filtering the simulated
      streams and modeling Carl Grillmair's visual cortex...

* Predict action-space / Energy-angular-momentum-space properties of the streams
  by mapping orbital properties + masses to number of stars and blobs in
  conserved quantities
    - In particular, do they show up over a smooth halo background in this space?

2016-09-13
----------

The plan, as sent to Oleg:

1) I've defined a potential model for the Milky Way that is a combination of a
   disk (Miyamoto-Nagai), bulge (Hernquist), and spherical halo (NFW). I
   roughly match the rotation curve measurements over the radii (4 kpc < R < 14
   kpc) -- I don't think the precise details matter too much, so this is
   something just to get started. For now, I assume the potential has fixed
   mass even at z=3 (when I form the clusters), but I might want to change that
   assumption, or possibly only consider the cluster orbits from z=2 onwards.

2) I choose a radial profile and mass distribution for the initial globular
   cluster population. For this, I use the same distributions you used in
   Gnedin et al. 2014 (G14) but I allow the radii to go out to 200 kpc. I
   follow the same prescription you use to set the number of clusters: I
   generate samples from these distributions so the total stellar mass in
   globular clusters is some fraction of the total stellar mass of the Galaxy,
   and I use the radii as the *mean* orbital radii of each cluster.

3) I turn the mean orbital radius into a 3D position assuming isotropy and,
   given a distribution function, I then generate velocities for each cluster
   by embedding the DF (as a tracer population) in the background MW potential.
   I integrate the orbits of the clusters in th MW potential for the total
   evolution time (for now, set to 11.5 yr, same as in G14) and compute the
   circularity of the orbits.

4) I solve for the mass-loss history of each cluster using the differential
   equations you use in G14, but I set f_e = exp(1.9*(J/J_c - 1)), since I have
   the circularities. For each cluster, I then use its orbit and mass evolution
   to generate a mock stellar stream evolved for the full evolution time. The
   prescription I use is to release star particles uniformly in time from the
   Lagrange points around the parent cluster orbit with dispersions set by the
   mass of the cluster at the time a star particle is released. If the mass of
   the cluster goes to 0, no more particles are released, but the
   previously-stripped debris stars are evolved to present day.

For the orbit distribution functions, I'll consider two cases: (1) all cluster
orbits are drawn from the same distribution function (the 'in situ' models),
and (2) the cluster orbits are small deviations away from a small number of
orbits drawn from the same distribution function (the 'all accreted' models).
In both cases, I will consider isotropic, radially-anisotropic, and
tangentially-anisotropic DFs.

The rationale behind Case 2, as I think we discussed, is that if all of the
clusters came in with a smaller number of galaxies that merged early on, then
there could be some relic of that in the orbits of the surviving clusters. Of
course, the problem is then figuring out how to pick the number of galaxies and
how to assign the clsuters to the galaxies...I'm assuming we could use merger
trees from simulations, abundance matching, and a conversion from stellar-mass
to number of clusters, but hoping you have some input there.

The end goal is then to take each of these simulated halos, "observe" a small
chunk with Gaia, and see if we can distinguish between the different formation
scenarios. Here I'm imagining that we can measure chemical abundances for all
of the halo stars and can separate out "stars that were formed in globular
clusters" vs. otherwise, that way I don't have to worry about super-imposing
the rest of the stellar halo on top of this. Does that sound crazy to you?

2016-09-22
----------

After thinking a bit about what we want to measure, some slight modifications to the above:

1) Define a potential model for the Milky Way that is a combination of a disk (Miyamoto-Nagai),
   bulge (Hernquist), and spherical halo (NFW) by roughly matching to rotation curve measurements
   over the radii (4 kpc < R < 14 kpc). For the last 10 Gyr, the disk should grow linearly or as
   some function of time consistent with the star formation rate, and the halo should grow
   according to accretion seen in simulations.

2) Consider the following scenarios for the initial population of globular clusters: (1) in place
   at t=-10 Gyr with a spherical, isotropic DF, (2) constant isotropic accretion, (3) accreted
   cosmologically with satellite galaxies with S_N = [2, 4, 8, 16]. In each of these cases, the
   prediction for the number of MW globular clusters is different because we need to end with the
   ~150 we see today. Therefore, the number disrupted and the number of streams will be different.
