static const double bulge_pars[] = {0., 5.E9, 1.}; // G, m0, c0
static const double disk_pars[] = {0., 6.8E10, 3., 0.28}; // G, m0, a0, b

double growing_milkyway_value(double t, double *pars, double *q, int n_dim);
void growing_milkyway_gradient(double t, double *pars, double *q, int n_dim, double *grad);
double growing_milkyway_density(double t, double *pars, double *q, int n_dim);
