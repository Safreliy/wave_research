/**
Two-phase VOF solitary wave shoaling and breaking on a plane beach.

This is a compact, reproducible downstream test for the future Euler--BIE to
VOF handoff.  It follows the nondimensional setup of the public Basilisk
shallow-water breaker example by Mostert and Deike.  It is not yet initialized
from a BIE archive; its purpose is to verify the receiving two-phase solver and
to produce an actual topology-changing impact sequence.
*/

#include "grid/quadtree.h"
#include "navier-stokes/centered.h"
#include "two-phase.h"
#include "navier-stokes/conserving.h"
#include "tension.h"
#include "reduced.h"

#ifndef LEVEL
# define LEVEL 10
#endif
#ifndef MINLEVEL
# define MINLEVEL 7
#endif
#ifndef AMPLITUDE
# define AMPLITUDE 0.30
#endif
#ifndef END_TIME
# define END_TIME 24.0
#endif
#ifndef FRAME_N
# define FRAME_N 1000
#endif
#ifndef FRAME_X_MIN
# define FRAME_X_MIN 0.0
#endif
#ifndef FRAME_X_MAX
# define FRAME_X_MAX 32.0
#endif
#ifndef FRAME_Y_MIN
# define FRAME_Y_MIN -1.35
#endif
#ifndef FRAME_Y_MAX
# define FRAME_Y_MAX 1.15
#endif

#define DENSITY_RATIO (1.0/850.0)
#define VISCOSITY_RATIO (17.4e-6/8.9e-4)
#define REYNOLDS 40000.0
#define BOND 1000.0

double depth = 1.0;
double gravity_value = 1.0;
double wave_center = 5.0;
double domain_length = 40.0;
double beach_toe = 10.0;
double beach_slope = 3.0*pi/180.0;
scalar beach[];

int main()
{
  size (domain_length);
  origin (0.0, -1.5);
  rho1 = 1.0;
  rho2 = DENSITY_RATIO;
  mu1 = 1.0/REYNOLDS;
  mu2 = mu1*VISCOSITY_RATIO;
  f.sigma = 1.0/BOND;
  G.y = -gravity_value;
  init_grid (1 << MINLEVEL);
  run();
}

double beach_height (double horizontal)
{
  return horizontal < beach_toe
    ? -depth
    : -depth + beach_slope*(horizontal - beach_toe);
}

double sech_value (double value)
{
  return 1.0/cosh(value);
}

double elevation (double horizontal)
{
  double k = sqrt(3.0*AMPLITUDE)/
    (2.0*depth*sqrt(depth*(1.0 + AMPLITUDE/depth)));
  return AMPLITUDE*sq(sech_value(k*horizontal));
}

double elevation_derivative (double horizontal)
{
  double k = sqrt(3.0*AMPLITUDE)/
    (2.0*depth*sqrt(depth*(1.0 + AMPLITUDE/depth)));
  return -2.0*AMPLITUDE*k*sq(sech_value(k*horizontal))*tanh(k*horizontal);
}

event init (i = 0)
{
  fraction (beach, beach_height(x) - y);
  fraction (f, elevation(x - wave_center) - y);
  double celerity = sqrt(gravity_value*depth*(1.0 + AMPLITUDE/depth));
  foreach() {
    double eta = elevation(x - wave_center);
    double deta = elevation_derivative(x - wave_center);
    double water_column = depth + eta;
    u.x[] = celerity*eta/water_column*f[]*(1.0 - beach[]);
    u.y[] = -(y + depth)*celerity*deta/water_column*
      (1.0 - eta/water_column)*f[]*(1.0 - beach[]);
  }
  boundary ((scalar *){u, f, beach});
}

event enforce_beach (i++)
{
  fraction (beach, beach_height(x) - y);
  foreach()
    foreach_dimension()
      u.x[] *= 1.0 - beach[];
  boundary ((scalar *){u, beach});
}

event adapt (i++)
{
  adapt_wavelet ((scalar *){f, u.x, u.y, beach},
                 (double[]){2e-4, 2e-3, 2e-3, 8e-3},
                 LEVEL, MINLEVEL);
}

event frames (t = 0.0; t += 0.08; t <= END_TIME)
{
  static int frame = 0;
  char name[128];
  sprintf (name, "frames/vof-%05d.ppm", frame++);
  FILE * image = fopen (name, "w");
  output_ppm (f, image, n = FRAME_N, min = 0.0, max = 1.0,
              linear = true, mask = beach,
              box = {{FRAME_X_MIN, FRAME_Y_MIN}, {FRAME_X_MAX, FRAME_Y_MAX}});
  fclose (image);
}

event diagnostics (t += 0.08; t <= END_TIME)
{
  double liquid = 0.0, kinetic = 0.0;
  foreach (reduction(+:liquid) reduction(+:kinetic)) {
    double active = f[]*(1.0 - beach[]);
    liquid += active*dv();
    kinetic += 0.5*rho[]*(sq(u.x[]) + sq(u.y[]))*active*dv();
  }
  fprintf (stderr, "%g %d %.12g %.12g\n", t, i, liquid, kinetic);
}

event stop (t = END_TIME)
{
  return 1;
}
