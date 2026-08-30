/**
Momentum-conserving wrapper for q_embed_vof.h.

This follows Basilisk's navier-stokes/conserving.h, but transports the two
phase momenta with q_embed_vof_advection(), whose finite-volume capacity is
the open cell and whose phase fluxes are double-PLIC apertures.
*/

#ifndef Q_EMBED_MOMENTUM_DIAGNOSTICS
# define Q_EMBED_MOMENTUM_DIAGNOSTICS 0
#endif
#ifndef Q_EMBED_MOMENTUM_DIAGNOSTIC_STOP_ITERATION
# define Q_EMBED_MOMENTUM_DIAGNOSTIC_STOP_ITERATION -1
#endif

#if TREE
static void q_embed_momentum_refine (Point point, scalar velocity)
{
  refine_bilinear (point, velocity);
  double momentum = 0., capacity = 0.;
  foreach_child() {
    momentum += cm[]*rho(f[])*velocity[];
    capacity += cm[]*rho(f[]);
  }
  double correction = velocity[] - momentum/(capacity + SEPS);
  foreach_child()
    velocity[] += correction;
}

static void q_embed_momentum_restriction (Point point, scalar velocity)
{
  double momentum = 0., capacity = 0.;
  foreach_child() {
    momentum += cm[]*rho(f[])*velocity[];
    capacity += cm[]*rho(f[]);
  }
  velocity[] = momentum/(capacity + SEPS);
}
#endif

event defaults (i = 0)
{
  stokes = true;
#if TREE
  /* The momentum callbacks require conditional f (and hence rho(f)) to be
     restricted/refined before velocity, as in Basilisk conserving.h. */
  int fraction_index = 0;
  while (all[fraction_index].i >= 0 && all[fraction_index].i != f.i)
    fraction_index++;
  while (fraction_index > 0 && all[fraction_index].i) {
    all[fraction_index] = all[fraction_index - 1];
    fraction_index--;
  }
  all[fraction_index] = f;
  foreach_dimension() {
    u.x.refine = u.x.prolongation = q_embed_momentum_refine;
    u.x.restriction = q_embed_momentum_restriction;
    u.x.depends = list_add (u.x.depends, f);
  }
#endif
}

event stability (i++)
{
  dtmax = timestep (uf, dtmax);
  dtmax = q_embed_stable_timestep (f, uf, dtmax);
}

foreach_dimension()
static double q_embed_boundary_q1_x
  (Point neighbor, Point point, scalar momentum, void * data)
{
  return clamp(f[],0.,1.)*rho1*u.x[];
}

foreach_dimension()
static double q_embed_boundary_q2_x
  (Point neighbor, Point point, scalar momentum, void * data)
{
  return (1. - clamp(f[],0.,1.))*rho2*u.x[];
}

#if TREE
foreach_dimension()
static void q_embed_prolongation_q1_x (Point point, scalar momentum)
{
  foreach_child()
    momentum[] = clamp(f[],0.,1.)*rho1*u.x[];
}

foreach_dimension()
static void q_embed_prolongation_q2_x (Point point, scalar momentum)
{
  foreach_child()
    momentum[] = (1. - clamp(f[],0.,1.))*rho2*u.x[];
}
#endif

static scalar * q_embed_interfaces_saved = NULL;

event vof (i++)
{
  vector q1[], q2[];
  vector velocity_before[];
  for (scalar momentum in {q1,q2}) {
    momentum.depends = list_add (momentum.depends, f);
    foreach_dimension()
      momentum.v.x.i = -1;
  }
  for (int boundary_index = 0; boundary_index < nboundary; boundary_index++)
    foreach_dimension() {
      q1.x.boundary[boundary_index] = q_embed_boundary_q1_x;
      q2.x.boundary[boundary_index] = q_embed_boundary_q2_x;
    }
#if TREE
  foreach_dimension() {
    q1.x.prolongation = q_embed_prolongation_q1_x;
    q2.x.prolongation = q_embed_prolongation_q2_x;
  }
#endif
  foreach()
    foreach_dimension() {
      double liquid = clamp(f[],0.,1.);
      velocity_before.x[] = u.x[];
      q1.x[] = liquid*rho1*u.x[];
      q2.x[] = (1. - liquid)*rho2*u.x[];
    }
  foreach_dimension() {
    q2.x.inverse = true;
    q1.x.gradient = q2.x.gradient = u.x.gradient;
  }
  scalar * tracers = f.tracers;
  f.tracers = list_concat (tracers, (scalar *){q1,q2});
  q_embed_vof_advection (f, i);
  free (f.tracers);
  f.tracers = tracers;
  foreach()
    foreach_dimension()
      u.x[] = (q1.x[] + q2.x[])/rho(f[]);
#if Q_EMBED_MOMENTUM_DIAGNOSTICS
  double velocity_delta2 = 0., velocity_norm2 = 0.;
  double kinetic_before = 0., kinetic_after = 0.;
  double maximum_speed_before = 0., maximum_speed_after = 0.;
  double maximum_q1 = 0., maximum_q2 = 0.;
  foreach (reduction(+:velocity_delta2) reduction(+:velocity_norm2)
           reduction(+:kinetic_before) reduction(+:kinetic_after)
           reduction(max:maximum_speed_before)
           reduction(max:maximum_speed_after)
           reduction(max:maximum_q1) reduction(max:maximum_q2)) {
    double capacity = rho(f[])*dv();
    foreach_dimension() {
      velocity_delta2 += capacity*sq(u.x[] - velocity_before.x[]);
      velocity_norm2 += capacity*sq(velocity_before.x[]);
      maximum_q1 = max(maximum_q1, fabs(q1.x[]));
      maximum_q2 = max(maximum_q2, fabs(q2.x[]));
    }
    kinetic_before += 0.5*capacity*
      (sq(velocity_before.x[]) + sq(velocity_before.y[]));
    kinetic_after += 0.5*capacity*(sq(u.x[]) + sq(u.y[]));
    maximum_speed_before = max(maximum_speed_before,
                               hypot(velocity_before.x[], velocity_before.y[]));
    maximum_speed_after = max(maximum_speed_after, hypot(u.x[], u.y[]));
  }
  fprintf (stderr,
           "Q_MOMENTUM %.12g %d %.12g %.12g %.12g %.12g %.12g %.12g %.12g\n",
           t, i,
           sqrt(velocity_delta2/max(velocity_norm2, 1e-30)),
           maximum_speed_before, maximum_speed_after,
           kinetic_before, kinetic_after, maximum_q1, maximum_q2);
#endif
  q_embed_interfaces_saved = interfaces;
  interfaces = NULL;
#if Q_EMBED_MOMENTUM_DIAGNOSTIC_STOP_ITERATION >= 0
  if (i == Q_EMBED_MOMENTUM_DIAGNOSTIC_STOP_ITERATION)
    return 1;
#endif
}

event tracer_advection (i++)
  interfaces = q_embed_interfaces_saved;
