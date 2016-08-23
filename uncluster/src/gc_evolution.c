// Written by Oleg Gnedin
// Last modified on May 28, 2013

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define Ntot 1000000  // maximum number of clusters
#define Nz   402      // number of points for cosmic t-z relation
#define Nir  55       // number of points for stellar debris vs radius
#define Nit  70       // number of points for stellar debris vs time

#define sqr(x)  ((x)*(x))
extern double ran1_();
extern void slinear_();
extern double gammp_();

int GALAXY, nsteps[Ntot], Nbh=0, idum=-123;
double ri_[Ntot], rf_[Ntot], Mi_[Ntot], Mf_[Ntot], Vc_[Ntot], ti_[Ntot], tacc_[Ntot], 
  tf_[Ntot], t_[Nz], z_[Nz], rdebris_[Nir], tdebris_[Nit], mdebris_[Nir][Nit], 
  mdebriss_[Nir][Nit], mdebris1_[Nir][Nit], mdebriss1_[Nir][Nit],
  mdebrisse_[Nir], mdebrisse1_[Nir], nser, rser, rmax, Ms, Mh, rs, 
  kms=2.07e-3,
  fGC0=0.01,  // initial cluster mass fraction
  beta=2.0,   // slope of the initial cluster mass function
  Mmin=1.e4,  // minimum cluster mass, in solar masses
  Mmax=1.e7,  // maximum cluster mass, in solar masses
  zi=6.,      // initial redshift for forming clusters
  zf=0.,      // final redshift for forming clusters
  ecc_df=1.;  // eccentricity correction for dynamical friction time

typedef struct {
  double m, r, rf, tacc;
} black_hole;

black_hole bh[100];


void EllModel( double z, double *Msz, double *nserz, double *rserz, 
	       double *an, double *bn ) {
  double ns;
  *Msz = Ms*pow(1.+z, -0.67);
  *nserz = ns = nser*pow(1.+z, -0.95);
  //*nserz = ns = nser*pow(1.+z, -0.3);
  *rserz = rser*pow(1.+z, -1.27);
  *an = 2.*ns;
  *bn = 2.*ns-1./3.+0.0098765/ns+0.0018/sqr(ns);
  return;
}


void GenerateClusters( int *Nformed ) {
  #define ni 501
  int i, idum, i1, i2, iacc, n=ni, nz=Nz, j, nj, ns=7;
  double r, f, an, bn, arg, argc, gamnorm, Msz, Msz1, nserz, rserz, t, ti, tf, z,
    m_in, tau_in = 1.1, tau_acc = 6., 
    f_in, f_in1, f_acc, f_acc1, df_in, df_acc, dMsz_in, dMsz_acc, reduction_factor, tsf,
    rser_[ni], mser_[ni], mser1_[ni], dmser_[ni], tsf_[ni], fsf_[ni],
    zs_[7] =   { 0., 0.4,  1.,  2.,   3.,   4.,   6. }, 
    rhos_[7] = { 1., 0.95, 0.8, 0.45, 0.12, 0.02, 0. };

  if(GALAXY>=3) { 
    nj = 20; 
    m_in = 0.52;
  } else { 
    nj = 1;
    m_in = 1.0;
    zi = 3.;
  }

  // check M87 model
  if(GALAXY==2 && 0) {
    double M3, M8;
    r = 1.;
    nser = 3.;
    EllModel(0., &Msz, &nserz, &rserz, &an, &bn);
    arg = bn*pow(r/rserz, 1./nserz);
    M3 = gammp_(&an, &arg);     
    nser = 8.;
    EllModel(0., &Msz, &nserz, &rserz, &an, &bn);
    arg = bn*pow(r/rserz, 1./nserz);
    M8 = gammp_(&an, &arg);
    r = 10.;
    EllModel(0., &Msz, &nserz, &rserz, &an, &bn);
    arg = bn*pow(r/rserz, 1./nserz);
    Msz = gammp_(&an, &arg);
    printf("M(n=8)/M(n=3) = %g at r=1, %g at r=%g\n", M8/M3, Msz/(M3+Msz-M8), r);
    exit(0);
  }

  ran1_(&idum);
  i2=0; Msz=f_in=f_acc=0.;
  for(i=0; i<ni; i++) mser_[i]=0.;
  
  slinear_(z_, t_, &nz, &zi, &ti);
  slinear_(z_, t_, &nz, &zf, &tf);
  
  // interpolate the growth rate by star formation
  for(i=0; i<ni; i++) {
    tsf_[i] = t = t_[0]*(double)i/(double)(ni-1);
    fsf_[i] = (1.-(1.+t/tau_in)*exp(-t/tau_in))/(1.-(1.+t_[0]/tau_in)*exp(-t_[0]/tau_in));
  }

  // split cluster formation over nj episodes from zf to zi, spread linearly in time

  for(j=0; j<nj; j++) {
    if(j==0) {
      t = ti; 
      z = zi;
    } else {
      t = ti + (tf-ti)*(double)j/(double)(nj-1);
      slinear_(t_, z_, &nz, &t, &z);
      if(z<0.) z=0.;
    }
    if(GALAXY<3) z=zf;  // only one episode of cluster formation

    // cumulative mass distribution for Sersic model at redshift z
    Msz1 = Msz;
    for(i=0; i<ni; i++) mser1_[i] = mser_[i];
    EllModel(z, &Msz, &nserz, &rserz, &an, &bn);
    argc = bn*pow(rmax/rserz, 1./nserz);
    gamnorm = gammp_(&an, &argc);
    for(i=0; i<ni; i++) {
      rser_[i] = r = pow(10., -3. + i/100.);
      arg = bn*pow(r/rserz, 1./nserz);
      mser_[i] = gammp_(&an, &arg)/gamnorm;
    }
    
    // split stellar mass growth into in-situ star formation and dissipationless accretion
    f_in1 = f_in; f_acc1 = f_acc;
    f_in = m_in*(1.-(1.+t/tau_in)*exp(-t/tau_in))/(1.-(1.+t_[0]/tau_in)*exp(-t_[0]/tau_in));
    f_acc = (1.-m_in)*(1.-(1.+t/tau_acc)*exp(-t/tau_acc))/(1.-(1.+t_[0]/tau_acc)*exp(-t_[0]/tau_acc));
    df_in = f_in - f_in1;
    df_acc = f_acc - f_acc1;
    dMsz_in = (Msz-Msz1)*df_in/(df_in+df_acc);
    dMsz_acc = (Msz-Msz1)*df_acc/(df_in+df_acc);
    
    // find where the new stellar profile exceeds the old one by 0.4% (= observed M_bh/M_* ratio)...
    for(i=0; i<ni; i++) dmser_[i] = mser_[i] - Msz1/Msz*mser1_[i];
    f = 0.004;
    slinear_(dmser_, rser_, &n, &f, &r);
    
    // ...and put an accreted satellite black hole there
    slinear_(zs_, rhos_, &ns, &z, &reduction_factor);
    if(reduction_factor > 1.) reduction_factor = 1.;
    if(reduction_factor < 0.) reduction_factor = 0.;
    bh[Nbh].m = 0.004*dMsz_acc*pow(dMsz_acc/1.e11,0.1)*reduction_factor;
    bh[Nbh].r = r;
    bh[Nbh].tacc = t;
    Nbh++;
    
    // number of clusters to form
    i1 = i2;  
    //i2 = i1 + (int)((Msz-Msz1)/M1+0.5);
    //if(i2 < i1) i2=i1;   
    //iacc = i1 + (int)(dMsz_in/M1+0.5);

    // draw cluster mass, to match the required mass of clusters forming in this episode
    double summz=0., sumM1, sumMacc, m;
    i=i1; iacc=-1;
    sumM1 = (Msz-Msz1)*fGC0;
    sumMacc = (dMsz_in-Msz1)*fGC0;
    while(summz < sumM1-Mmin && i < Ntot) {
      m = Mmin/pow(1. - ran1_(&idum)*(1.-pow(Mmin/Mmax,beta-1.)), 1./(beta-1.));
      if(summz+m < sumM1+Mmin) {
	summz += m;
	Mi_[i]=m; i++;
	if(iacc<0 && summz>sumMacc) iacc=i;
      }
    }
    i2=i;
    if(iacc<0) iacc=i2;
    if(i >= Ntot) printf("number of clusters exceeded the array size. increase Ntot\n");

    // draw cluster position
    for(i=i1; i<i2; i++) {
      f = ran1_(&idum);
      slinear_(mser_, rser_, &n, &f, &r);
      ri_[i] = r;
      if(i<iacc) {
	ti_[i] = t;
	tacc_[i] = 0.;
      } else {
	f = ran1_(&idum);
	slinear_(fsf_, tsf_, &n, &f, &tsf);
	if(tsf > t) tsf = t;
	ti_[i] = tsf;     // actual time of formation
	tacc_[i] = t;     // time of accretion
      }
    }

    printf("i1=%d i2=%d Nacc=%d f_in=%5.3f z=%5.3f t=%5.3f Msz=%9.3e Msz1=%9.3e dMsz_in=%g\n", 
	   i1, i2, i2-iacc, df_in/(df_in+df_acc), z, t, Msz, Msz1, dMsz_in);    
  }
  *Nformed = i2;
}


double Spline2D( double x, double y, int nx, int ny, double *x_, double *y_, double z_[Nir][Nit] )
{
  double x1, x2, y1, y2, dx, dy, s, xmax, xmin, ymax, ymin, add=1.0e-10, F11, F12, F21, F22;
  int i, j, xorder, yorder, ix0, iy0;

  if(x_[1]<x_[nx-2]) {xorder=1; ix0=0;} else {xorder=-1; ix0=nx-1;}
  if(y_[1]<y_[ny-2]) {yorder=1; iy0=0;} else {yorder=-1; iy0=ny-1;}

  xmax=x_[nx-1-ix0]; xmin=x_[ix0];
  ymax=y_[ny-1-iy0]; ymin=y_[iy0];

  if((x>xmax)&&(x-xmax<1.0e-6)) x=xmax;  /* to prevent inadequate exit */
  if((y>ymax)&&(y-ymax<1.0e-6)) y=ymax;  /* to prevent inadequate exit */

  if(x<xmin-add) {printf("Spline2D: out of data (x=%e < xmin=%e)\n", x, xmin); exit(1);}
  if(x>xmax+add) {printf("Spline2D: out of data (x=%e > xmax=%e)\n", x, xmax); exit(1);}
  if(y<ymin-add) {printf("Spline2D: out of data (y=%e < ymin=%e)\n", y, ymin); exit(1);}
  if(y>ymax+add) {printf("Spline2D: out of data (y=%e > ymax=%e)\n", y, ymax); exit(1);}

  i=ix0; x2=x_[i];
  while( x > x2 ) { i+=xorder; x2=x_[i]; }
  if((i<0)||(i>=nx)) printf("Spline2D: x-index out of order!\n");
  if(i==ix0) xorder=-xorder;
  x1=x_[i-xorder]; dx=x2-x1;
  j=iy0; y2=y_[j];
  while( y > y2 ) { j+=yorder; y2=y_[j]; }
  if((j<0)||(j>=ny)) printf("Spline2D: y-index out of order!\n");
  if(j==iy0) yorder=-yorder;
  y1=y_[j-yorder]; dy=y2-y1;
  F11=z_[i-xorder][j-yorder]; F12=z_[i-xorder][j];
  F21=z_[i][j-yorder]; F22=z_[i][j];
/* :::::::::::::::::::::::::::::::::::::::::::::::::::::
                        F12 * - - - - * F22
  y ^                       |         |
    |                       | .       |
    |                       |         |
    ---------> x        F11 * - - - - * F21
   ::::::::::::::::::::::::::::::::::::::::::::::::::::: */
  if(x1==x2) 
    s = ( (y2-y)*F21 + (y-y1)*F22 )/dy;
  else
    s = ( (x2-x)*(y2-y)*F11 + (x2-x)*(y-y1)*F12
	  + (x-x1)*(y2-y)*F21 + (x-x1)*(y-y1)*F22 )/dx/dy;
  return( s );
}


// Circular velocity for a Sersic model

void Potential( double r, double *Vcirc, double t ) {
  int nz=Nz;
  double an, bn, arg, Mstar, Mhalo, Mdeb, nserz, rserz, Msz, z;

  if(GALAXY>=3) {
    slinear_(t_, z_, &nz, &t, &z);
    if(z < 0.) z = 0.;
  } else
    z = 0.;
  // stellar mass
  EllModel(z, &Msz, &nserz, &rserz, &an, &bn);
  arg = bn*pow(r/rserz, 1./nserz);
  Mstar = Msz*gammp_(&an, &arg);
  // halo mass
  Mhalo = Mh*(log(1.+r/rs) - r/rs/(1.+r/rs));
  // accumulated debris mass (stars+gas+BHs)
  /* slinear_(tdebris_, mcdebris_, &nl, &t, &Mdeb); */
  Mdeb = Spline2D(r, t, Nir, Nit, rdebris_, tdebris_, mdebris_);
  // circular velocity
  *Vcirc = sqrt((Mstar+Mhalo+Mdeb)/r)*kms;
}


void skiplines( FILE *in, int n ) 
{
  int i; char ch;  
  for(i=0; i<n; i++) {ch='1'; while(ch != '\n') fscanf(in, "%c", &ch);}
}



int main( int argc, char *argv[] )
{
  int i, j, it, ir, ir2, irbh, Ncl, Nclf, nms, EVAP;
  double delta, slope, td, tdf, t, dt, dt0, m, r, Vc, age, fse, fse1, fse0, dm1, dm2, 
    dm1se, tol, P, t_tid, t_iso, mitot, mftot, logmitot, logmftot, miav, mfav, mcbh, 
    rhocl, flost_[33], mslife_[33];
  char filename[60];
  FILE *in, *out, *out2, *out3, *out4, *outs;

  if(argc<3 || argv[1]=="") {
    printf("Usage: gc galaxy(1=MW;2,3=M87;4,5=mini-M87) evap_model(0,1) f_GC(0.01) beta(2) Mmin(1e4) Mmax(1e7) ecc_df(1) name\n");
    return 1;
  }

  // Understanding input
  GALAXY = atoi(argv[1]);
  switch(GALAXY) {
  case 1: 
    printf("MW galaxy, "); 
    Ms = 5.e10;   // stellar mass, in solar masses
    nser = 2.2;   // stellar Sersic parameter 2.2
    rser = 4.;    // stellar Sersic effective radius, in kpc
    rmax = 99.;   // maximum Galactocentric radius for GCs, in kpc
    Mh = 1.e12;   // halo mass, in solar masses
    rs = 20.;     // halo scale radius, in kpc
    break;
  case 2:
  case 3:
    printf("M87 galaxy, "); 
    Ms = 8.e11;   // stellar mass, in solar masses
    nser = 8.;    // stellar Sersic parameter
    rser = 30.;   // stellar Sersic effective radius, in kpc
    rmax = 199.;  // maximum galactocentric radius for GCs, in kpc
    Mh = 2.7e13;  // halo mass, in solar masses
    rs = 50.;     // halo scale radius, in kpc
    break;
  case 4:
    printf("M87-mini1 galaxy, "); 
    Ms = 2.e11;   // stellar mass, in solar masses
    nser = 4.;    // stellar Sersic parameter
    rser = 8.6;   // stellar Sersic effective radius, in kpc
    rmax = 199.;  // maximum galactocentric radius for GCs, in kpc
    Mh = 5.e12;   // halo mass, in solar masses
    rs = 35.;     // halo scale radius, in kpc
    break;
  case 5:
    printf("M87-mini2 galaxy, "); 
    Ms = 5.e10;   // stellar mass, in solar masses
    nser = 2.;    // stellar Sersic parameter
    rser = 2.5;   // stellar Sersic effective radius, in kpc
    rmax = 199.;  // maximum galactocentric radius for GCs, in kpc
    Mh = 1.e12;   // halo mass, in solar masses
    rs = 20.;     // halo scale radius, in kpc
    break;
  default: 
    printf("galaxy must be 1, 2, 3, 4, or 5\n"); return 1;
  }

  EVAP = atoi(argv[2]);

  if(argc>3) fGC0 = atof(argv[3]);
  if(argc>4) beta = atof(argv[4]); if(beta<1.01) beta=1.01;
  if(argc>5) Mmin = atof(argv[5]);
  if(argc>6) Mmax = atof(argv[6]);
  if(argc>7) ecc_df = atof(argv[7]);
  printf("initial cluster fraction=%g  CMFslope=%g  Mmin=%g  Mmax=%g  ecc_df=%g\n", fGC0, beta, Mmin, Mmax, ecc_df);

  // cosmic time in Gyr vs. redshift
  double Om=0.272, h100=0.704, f;
  for(i=0; i<Nz; i++) {
    z_[i] = zi*(double)i/(double)(Nz-2);
    f = Om/(1.-Om)*pow(1.+z_[i],3.);
    t_[i] = 9.779/h100*2./3./sqrt(1.-Om)*log((1.+sqrt(1.+f))/sqrt(f));
  }

  // initial conditions for globular clusters
  GenerateClusters(&Ncl);

  // read cumulative stellar mass loss table
  if( (in = fopen( "massloss.dat", "r" )) == NULL )
    { printf("Can't open input file\n"); exit(1); }
  skiplines(in, 5);
  i=0;
  while(!feof(in)) {
    fscanf(in, "%le %le %le %le", &t, flost_+i, &t, mslife_+i);
    mslife_[i] *= 1.e-9;
    i++;
  }
  fclose(in);
  nms = i-1; //printf("n_ms = %d\n", nms);
  // make a mass loss rate table
  //for(i=nms-1; i>0; i--) flost_[i] = (flost_[i]-flost_[i-1])/(mslife_[i]-mslife_[i-1]);
  //for(i=0; i<nms; i++) printf("mslife=%f flost=%f\n", mslife_[i], flost_[i]);

  // radial grid for counting stellar debris
  for(ir=0; ir<Nir; ir++) {
    r = -3.0 + (double)(ir-1)*0.1;
    rdebris_[ir] = pow(10.,r);
  }
  rdebris_[0] = 1.e-6;

  // time grid for counting stellar debris
  for(it=0; it<Nit; it++)
    tdebris_[it] = (double)it*0.2;

  for(ir=0; ir<Nir; ir++) {
    for(it=0; it<Nit; it++)
      mdebris_[ir][it] = mdebriss_[ir][it] = 0.;
    mdebrisse_[ir] = 0.;
  }

  // dynamical friction of satellite black holes
  mcbh = 0.;
  for(i=0; i<Nbh; i++) {
    t = bh[i].tacc; dt0 = (t_[0]-t)*0.002;
    r = bh[i].r;  

    r = 1.e-6; //to test instantaneous deposition of satellite BHs at the center

    irbh=Nir-1;
    while(t<t_[0] && r>1.e-6) {
      Potential(r, &Vc, t);     
      tdf = 64.*sqr(r)*(Vc/283.)/(bh[i].m/2.e5)*ecc_df;
      // update time step
      dt = dt0;
      tol = 0.02;
      if(dt > tol*tdf && r > 1.e-6) dt = tol*tdf;
      t += dt;
      r *= (1. - dt/tdf/2.); if(r < 1.e-7) r = 1.e-7;

      // add satellite black hole to accumulated debris mass
      //Ncl=0;
      ir=Nir-1;
      while(r <= rdebris_[ir] && ir >= 0) ir--;
      if(irbh > ir) {
	for(it=0; it<Nit; it++) {
	  if(t <= tdebris_[it]) {
	    mdebris_[ir][it] += bh[i].m;
	    if(irbh == Nir-1) {
	      for(ir2=ir+1; ir2<Nir; ir2++)
		mdebris_[ir2][it] += bh[i].m;
	    }
	  }
	}
	irbh = ir;
      }
    }
    bh[i].rf = r;
    if(bh[i].rf < 1.e-2) mcbh += bh[i].m;
  }

  // parameters of the dynamical evolution model
  if(EVAP) delta = 1./9.; else delta = 1./3.;
  slope = (1.+3.*delta)/2.;

  // solving differential evolutions for each cluster

  for(i=0; i<Ncl; i++) {

    t = ti_[i]; dt0 = (t_[0]-t)*0.002; j=0; fse=0.;
    m = Mi_[i]; r = ri_[i];

    for(ir=0; ir<Nir; ir++) {
      for(it=0; it<Nit; it++)
	mdebris1_[ir][it] = mdebriss1_[ir][it] = 0.; // debris of one cluster
      mdebrisse1_[ir] = 0.;
    }
    it=0;

    while(t<t_[0] && m>1.) {

      Potential(r, &Vc, t);
      if(j==0) Vc_[i] = Vc;
      j++;
      
      // dynamical friction, until r = 1 pc
      tdf = 64.*sqr(r)*(Vc/283.)/(m/2.e5)*ecc_df;
      
      // tidal mass loss, until M = 1 Msun
      P = (r/5.)*(207./Vc);
      t_tid = 10.*pow(m/2.e5,slope)*P;

      // treat separately clusters in satellites before accretion
      if(t < tacc_[i]) {
	tdf = 1.e99;
	t_tid = 10.*pow(m/2.e5,slope);
      }
      
      t_iso = 17.*(m/2.e5);  // or 17.*pow(m/2.e5,0.87)
      if(t_tid < t_iso) td = t_tid; else td = t_iso;
      
      // update time step
      dt = dt0;
      tol = 0.02;
      if(dt > tol*tdf && r > 1.e-6) dt = tol*tdf;
      if(dt > tol*td) dt = tol*td;
      t += dt;
      
      // update radius
      r *= (1. - dt/tdf/2.); if(r < 1.e-7) r = 1.e-7;
      
      // update mass
      dm1 = m*dt/td;  if(dm1>m) dm1=m;

      // immediate disruption within Roche lobe
      // average density for model clusters, (3/8pi) M/Rh^3, in Msun pc^-3
      // mass scaling from Gieles et al. 2011, MNRAS 413, 2509
      // max cluster density set at 10^5 Msun/pc^3
      rhocl = 1.e3;
      if(m > 1.e5) rhocl = 1.e3*sqr(m/1.e5);
      if(rhocl > 1.e5) rhocl = 1.e5;
      if(rhocl < 0.16*sqr(Vc/kms/r)*1.e-9) dm1=m;

      m -= dm1;
      
      // stellar evolution mass loss
      fse1 = fse;
      age = t - ti_[i];
      slinear_(mslife_, flost_, &nms, &age, &fse);
      dm2 = m*(fse-fse1);  if(dm2>m) dm2=m; if(dm2<0) dm2=0.;
      m -= dm2;
      
      // stellar evolution mass loss of stripped stars
      age = t_[0] - ti_[i];
      slinear_(mslife_, flost_, &nms, &age, &fse0);
      dm1se = dm1*(fse0-fse1);  if(dm1se>dm1) dm1se=dm1; if(dm1se<0) dm1se=0.;

      // note: stripped stars should continue to lose mass by stellar evolution
      // what we calculate is the mass of stars when they were stripped from GCs

      // count stellar debris at a current radius at a given epoch
      while(t > tdebris_[it] && it < Nit-1) it++;
      ir=Nir-1;
      while(r <= rdebris_[ir] && ir >= 0) {
	mdebris1_[ir][it] += (dm1+dm2);   // stars + gas
	mdebriss1_[ir][it] += dm1;        // stars only
	mdebrisse1_[ir] += (dm1-dm1se);   // evolved to z=0 stars only
	ir--;
      }

      /*
      ir=Nir-1;
      while(r <= rdebris_[ir] && ir >= 0) {
	it=Nit-1;
	while(t <= tdebris_[it] && it >= 0) {
	  mdebris_[ir][it] += (dm1+dm2); // stars + gas
	  mdebriss_[ir][it] += dm1;      // stars only
	  it--;
	}
	ir--;
      }
      */
      // count stellar debris at the center vs time
	/*
      if(r <= rdebris_[0]) {
	for(l=0; l<Nl; l++)
	  if(t <= tdebris_[l]) mcdebris0_[l] += (dm1+dm2);
      }
      if(r <= rdebris_[1]) {
	for(l=0; l<Nl; l++)
	  if(t <= tdebris_[l]) mcdebris_[l] += (dm1+dm2);
	  }*/
    }
    if(m > 1.) Mf_[i] = m; else Mf_[i] = 0.;
    rf_[i] = r;
    tf_[i] = t;
    nsteps[i] = j;

    // sum one cluster debris over time
    for(ir=0; ir<Nir; ir++) {
      for(it=1; it<Nit; it++) {
	mdebris1_[ir][it] += mdebris1_[ir][it-1];
	mdebriss1_[ir][it] += mdebriss1_[ir][it-1];
      }
    }
    // sum debris over all clusters
    for(ir=0; ir<Nir; ir++) {
      for(it=0; it<Nit; it++) {
	mdebris_[ir][it] += mdebris1_[ir][it];
	mdebriss_[ir][it] += mdebriss1_[ir][it];
      }
      mdebrisse_[ir] += mdebrisse1_[ir];
    }
  }

  // statistics of cluster population
  mitot = mftot = logmitot = logmftot = 0.; Nclf=0;
  for(i=0; i<Ncl; i++) {
    mitot += Mi_[i]; 
    logmitot += log(Mi_[i]);
    mftot += Mf_[i];
    if(Mf_[i]>1.) { logmftot += log(Mf_[i]); Nclf++; }
  }
  miav = exp(logmitot/(double)Ncl);
  mfav = exp(logmftot/(double)Nclf);
  printf("cluster number: initial = %d  final = %d\n", Ncl, Nclf);
  printf("cluster mass: initial = %8.2e  final = %8.2e  <mi> = %8.2e  <mf> = %8.2e\n", 
	 mitot, mftot, miav, mfav);

  if( (outs = fopen( "runs.sum", "a" )) == NULL )
      { printf("Can't open summary output file\n"); exit(1); }
  fprintf(outs, "%d %5.3f %3.1f %7.1e %7.1e %5d %3d %8.2e %8.2e %8.2e %8.2e %8.2e %8.2e\n",
	  EVAP, fGC0, beta, Mmin, Mmax, Ncl, Nclf, mitot, mftot, miav, mfav, mdebris_[1][Nit-1], mcbh);
  fclose(outs);

  // output initial and final cluster masses and radii
  if(argc>8) sprintf(filename, "clusters_%s.dat", argv[8]); else sprintf(filename, "clusters.dat");
  if( (out = fopen( filename, "w" )) == NULL )
    { printf("Can't open file <%s>\n", filename); exit(1); }
  fprintf(out, "# Globular cluster system modeling\n");
  fprintf(out, "# galaxy=%d evap=%d init_cluster_fraction=%g CMFslope=%g Mmin=%g Mmax=%g\n", 
	  GALAXY, EVAP, fGC0, beta, Mmin, Mmax);
  fprintf(out, "# Ncl_i=%d Ncl_f=%d Mcltot_i=%8.2e Mcltot_f=%8.2e mav_i=%8.2e mav_f=%8.2e\n", 
	  Ncl, Nclf, mitot, mftot, miav, mfav);
  fprintf(out, "# columns: ri rf Mi Mf Vc nsteps ti tacc tf\n");
  for(i=0; i<Ncl; i++)
    fprintf(out, "%9.3e %9.3e %9.3e %9.3e %5.1f %d %5.2f %5.2f %5.2f\n", 
	    ri_[i], rf_[i], Mi_[i], Mf_[i], Vc_[i], nsteps[i], ti_[i], tacc_[i], tf_[i]);
  fclose(out);

  if(argc>8) sprintf(filename, "debris_%s.dat", argv[8]); else sprintf(filename, "debris.dat");
  if( (out2 = fopen( filename, "w" )) == NULL )
      { printf("Can't open file <%s>\n", filename); exit(1); }
  fprintf(out2, "# rdebris mdebris (t=13.8, 2, 4, 8 Gyr) star-only mdebris\n"); // t = i*0.2
  for(ir=1; ir<Nir; ir++)
    fprintf(out2, "%9.3e %9.3e %9.3e %9.3e %9.3e %9.3e %9.3e %9.3e %9.3e %9.3e\n", rdebris_[ir], mdebris_[ir][Nit-1], mdebris_[ir][10], mdebris_[ir][20], mdebris_[ir][40], mdebriss_[ir][Nit-1], mdebriss_[ir][10], mdebriss_[ir][20], mdebriss_[ir][40], mdebrisse_[ir]);
  fclose(out2);

  if(argc>8) sprintf(filename, "cdebris_%s.dat", argv[8]); else sprintf(filename, "cdebris.dat");
  if( (out3 = fopen( filename, "w" )) == NULL )
      { printf("Can't open file <%s>\n", filename); exit(1); }
  fprintf(out3, "# tdebris mdebris (r=1, 10, 100, 1000 pc) star-only mdebris\n"); // r = 10**(-3+(ir-1)*0.1)
  for(it=0; it<Nit; it++)
    fprintf(out3, "%4.1f %9.3e %9.3e %9.3e %9.3e %9.3e %9.3e %9.3e %9.3e\n", tdebris_[it], mdebris_[1][it], mdebris_[11][it], mdebris_[21][it], mdebris_[31][it], mdebriss_[1][it], mdebriss_[11][it], mdebriss_[21][it], mdebriss_[31][it]);
  fclose(out3);

  if(argc>8) sprintf(filename, "satbh_%s.dat", argv[8]); else sprintf(filename, "satbh.dat");
  if( (out4 = fopen( filename, "w" )) == NULL )
      { printf("Can't open file <%s>\n", filename); exit(1); }
  fprintf(out4, "# bh.r, bh.rf, bh.m, bh.tacc\n");
  for(i=0; i<Nbh; i++)
    fprintf(out4, "%9.3e %9.3e %9.3e %9.3e\n", bh[i].r, bh[i].rf, bh[i].m, bh[i].tacc);
  fclose(out4);

  return 0;
}
