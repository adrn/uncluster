double growing_hernquist_value(double t, double *pars, double *q, int n_dim);
void growing_hernquist_gradient(double t, double *pars, double *q, int n_dim, double *grad);
double growing_hernquist_density(double t, double *pars, double *q, int n_dim);

double growing_miyamotonagai_value(double t, double *pars, double *q, int n_dim);
void growing_miyamotonagai_gradient(double t, double *pars, double *q, int n_dim, double *grad);
double growing_miyamotonagai_density(double t, double *pars, double *q, int n_dim);

double growing_sphericalnfw_value(double t, double *pars, double *q, int n_dim);
void growing_sphericalnfw_gradient(double t, double *pars, double *q, int n_dim, double *grad);
double growing_sphericalnfw_density(double t, double *pars, double *q, int n_dim);
