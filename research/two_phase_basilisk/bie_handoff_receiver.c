/**
Two-phase receiver initialized from the conservative Euler--BIE handoff.

The generated header contains the interface fraction and either a cell-centred
state or a constrained MAC streamfunction.  The latter is the preferred path:
it transfers shared finite-volume fluxes, embeds momentum and bed constraints
in one solve, and lets the receiver pressure projection handle only the change
caused by cut-cell sampling.
*/

#include "grid/quadtree.h"
#include "embed.h"
#include "navier-stokes/centered.h"
#include "two-phase.h"
#ifdef CONSERVATIVE_Q_EMBED
# include "q_embed_vof.h"
# ifndef Q_EMBED_CONSERVING_MOMENTUM
#  define Q_EMBED_CONSERVING_MOMENTUM 1
# endif
# if Q_EMBED_CONSERVING_MOMENTUM
#  include "conserving_q_embed.h"
# else
#  include "q_embed_advection_only.h"
# endif
#else
# include "navier-stokes/conserving.h"
#endif
#include "tension.h"
#include "reduced.h"
#ifdef RECEIVER_CONSERVATIVE_Q_HANDOFF_N576_T20303125_X1024_L8
# include "bie_handoff_data_receiver_conservative_q_t20303125_l8.h"
#elif defined(RECEIVER_CONSERVATIVE_Q_HANDOFF_N576_T20303125_X1024_L11)
# include "bie_handoff_data_receiver_conservative_q_t20303125_l11.h"
#elif defined(RECEIVER_CONSERVATIVE_Q_HANDOFF_N576_T20303125_X1024_L10)
# include "bie_handoff_data_receiver_conservative_q_t20303125_l10.h"
#elif defined(RECEIVER_CONSERVATIVE_Q_HANDOFF_N576_T20303125_X1024)
# include "bie_handoff_data_receiver_conservative_q_t20303125.h"
#elif defined(RECEIVER_GEOMETRIC_Q_HANDOFF_N576_T20303125_X1024)
# include "bie_handoff_data_receiver_geometric_q_t20303125.h"
#elif defined(RECEIVER_CONSTRAINED_HANDOFF_N576_T20303125_X1024)
# include "bie_handoff_data_receiver_constrained_t20303125.h"
#elif defined(MAC_FACE_HANDOFF_N576_T20303125_X1024)
# include "bie_handoff_data_active_n576_t20303125_mac_face_x1024.h"
#elif defined(MAC_FACE_HANDOFF_N576_X1024)
# include "bie_handoff_data_active_n576_mac_face_x1024.h"
#elif defined(ACTIVE_HANDOFF_N576_X1024_SW64)
# include "bie_handoff_data_active_n576_x1024_c150e200_sw64.h"
#elif defined(ACTIVE_HANDOFF_N576_SW32_C200)
# include "bie_handoff_data_active_n576_sw32_c200.h"
#elif defined(PHYSICAL_HANDOFF_N256_SW32)
# include "bie_handoff_data_physical_n256_sw32.h"
#elif defined(PHYSICAL_HANDOFF_N256)
# include "bie_handoff_data_physical_n256.h"
#else
# include "bie_handoff_data.h"
#endif

#ifndef LEVEL
# define LEVEL 9
#endif
#ifndef MINLEVEL
# define MINLEVEL 7
#endif
#ifndef END_TIME
# define END_TIME 1.5
#endif

#define DENSITY_RATIO (1.0/850.0)
#define VISCOSITY_RATIO (17.4e-6/8.9e-4)
#define REYNOLDS 40000.0
#define BOND 1000.0
#ifndef SURFACE_TENSION
# define SURFACE_TENSION (1./BOND)
#endif
#ifndef PROJECTION_TOLERANCE
# define PROJECTION_TOLERANCE 1e-8
#endif
#ifndef COMPATIBLE_EMBEDDED_FLUX
# define COMPATIBLE_EMBEDDED_FLUX 0
#endif
#ifndef FRAME_N
# define FRAME_N 900
#endif
#ifndef FRAME_X_MIN
# define FRAME_X_MIN 0.
#endif
#ifndef FRAME_X_MAX
# define FRAME_X_MAX domain_length
#endif
#ifndef FRAME_Y_MIN
# define FRAME_Y_MIN -1.28
#endif
#ifndef FRAME_Y_MAX
# define FRAME_Y_MAX 1.25
#endif
#ifndef SHORE_DIAGNOSTIC_X_MIN
# define SHORE_DIAGNOSTIC_X_MIN 28.
#endif
#ifndef SHORE_DIAGNOSTIC_X_MAX
# define SHORE_DIAGNOSTIC_X_MAX 37.
#endif
#ifndef VORTICITY_COLOR_LIMIT
# define VORTICITY_COLOR_LIMIT 8.
#endif
#ifndef INITIAL_ADAPT_ITERATION
# define INITIAL_ADAPT_ITERATION 0
#endif
#ifndef ADAPT_F_TOLERANCE
# define ADAPT_F_TOLERANCE 2e-4
#endif
#ifndef ADAPT_U_TOLERANCE
# define ADAPT_U_TOLERANCE 2e-3
#endif
#ifndef ADAPT_CS_TOLERANCE
# define ADAPT_CS_TOLERANCE 8e-3
#endif
#ifndef ADAPT_VOLUME_LEDGER
# define ADAPT_VOLUME_LEDGER 0
#endif
#ifndef POST_ADAPT_REPROJECT
# define POST_ADAPT_REPROJECT 0
#endif
#ifndef POST_ADAPT_REPROJECT_TRIGGER
# define POST_ADAPT_REPROJECT_TRIGGER 0.
#endif
#ifndef POST_ADAPT_PRESERVE_FACE_FLUX
# define POST_ADAPT_PRESERVE_FACE_FLUX 0
#endif
#ifndef POST_ADAPT_VOLUME_CORRECTION
# define POST_ADAPT_VOLUME_CORRECTION 0
#endif
#ifndef POST_ADAPT_VOLUME_TOLERANCE
# define POST_ADAPT_VOLUME_TOLERANCE 1e-12
#endif
#ifndef POST_ADAPT_VOLUME_INVARIANT_COMPENSATION
# define POST_ADAPT_VOLUME_INVARIANT_COMPENSATION 0
#endif
#ifndef POST_ADAPT_VOLUME_TRANSFORM_FACE_FLUX
# define POST_ADAPT_VOLUME_TRANSFORM_FACE_FLUX 0
#endif
#ifndef RECEIVER_ADAPT_EVERY_STEP
# define RECEIVER_ADAPT_EVERY_STEP 1
#endif
#ifndef RECEIVER_ADAPT_INTERVAL
# define RECEIVER_ADAPT_INTERVAL 1
#endif
#if RECEIVER_ADAPT_INTERVAL < 1
# error "RECEIVER_ADAPT_INTERVAL must be positive"
#endif
#ifndef HANDOFF_DOMAIN_LENGTH
# define HANDOFF_DOMAIN_LENGTH (2.*pi)
#endif
#ifndef HANDOFF_VERTICAL_ORIGIN
# define HANDOFF_VERTICAL_ORIGIN -1.30
#endif
#ifndef HANDOFF_HAS_MAC_STREAMFUNCTION
# define HANDOFF_HAS_MAC_STREAMFUNCTION 0
#endif
#ifndef HANDOFF_MOMENTUM_EMBEDDED
# define HANDOFF_MOMENTUM_EMBEDDED 0
#endif
#ifndef RECEIVER_RESIDUAL_MOMENTUM_CORRECTION
# define RECEIVER_RESIDUAL_MOMENTUM_CORRECTION 0
#endif
#ifndef HANDOFF_RECEIVER_LEVEL
# define HANDOFF_RECEIVER_LEVEL 0
#endif
#if HANDOFF_MOMENTUM_EMBEDDED && HANDOFF_RECEIVER_LEVEL > 0 && \
    LEVEL != HANDOFF_RECEIVER_LEVEL && !RECEIVER_RESIDUAL_MOMENTUM_CORRECTION
# error "momentum-embedded handoff was fitted on another receiver LEVEL; enable residual correction or regenerate it"
#endif
#ifndef RECEIVER_EXPORT_INITIAL_STATE
# define RECEIVER_EXPORT_INITIAL_STATE 0
#endif
#ifndef RECEIVER_EXPORT_INITIAL_STATE_PATH
# define RECEIVER_EXPORT_INITIAL_STATE_PATH "receiver_initial_state.csv"
#endif
#ifndef RECEIVER_INITIALIZATION_ONLY
# define RECEIVER_INITIALIZATION_ONLY 0
#endif
#ifndef CONSERVATIVE_Q_EMBED
# define CONSERVATIVE_Q_EMBED 0
#endif

double domain_length = HANDOFF_DOMAIN_LENGTH;
double vertical_origin = HANDOFF_VERTICAL_ORIGIN;
scalar beach[];
double q_reference_mass = 0.;
#if (HANDOFF_HAS_STREAMFUNCTION || HANDOFF_HAS_MAC_STREAMFUNCTION) && COMPATIBLE_EMBEDDED_FLUX
double restart_shift_x = 0.;
double restart_shift_y = 0.;
#endif

u.n[embed] = dirichlet (0.);
u.t[embed] = neumann (0.);

static inline double lerp (double a, double b, double weight)
{
  return a + weight*(b - a);
}

double bottom_height (double horizontal);

double sample_grid (const double * field, double horizontal, double vertical)
{
  horizontal = fmod(horizontal, domain_length);
  if (horizontal < 0.)
    horizontal += domain_length;
  double gx = (horizontal - handoff_x0)/handoff_dx;
  double gz = (vertical - handoff_z0)/handoff_dz;
  if (gz < 0. || gz > HANDOFF_NZ - 1.)
    return 0.;
  int i0 = (int) floor(gx);
  int k0 = (int) floor(gz);
  double ax = gx - i0;
  double az = gz - k0;
  i0 = (i0 % HANDOFF_NX + HANDOFF_NX) % HANDOFF_NX;
  int i1 = (i0 + 1) % HANDOFF_NX;
  if (k0 < 0) k0 = 0;
  if (k0 >= HANDOFF_NZ - 1) k0 = HANDOFF_NZ - 2;
  int k1 = k0 + 1;
  double lower = lerp(field[k0*HANDOFF_NX + i0],
                      field[k0*HANDOFF_NX + i1], ax);
  double upper = lerp(field[k1*HANDOFF_NX + i0],
                      field[k1*HANDOFF_NX + i1], ax);
  return lerp(lower, upper, az);
}

#if CONSERVATIVE_Q_EMBED
/* Conservative cell-average remap of the physical liquid aperture supplied by
   the BIE raster.  Unlike fraction(liquid_level), this overlap integral does
   not discard a thin overturning sheet merely because all four receiver-cell
   vertices lie outside the liquid polygon.  The source and receiver domains
   share the same periodic horizontal extent; vertical source cells outside
   the exported strip contribute zero. */
double remap_handoff_aperture (double horizontal, double vertical,
                               double cell_size)
{
  double source_x_origin = handoff_x0 - .5*handoff_dx;
  double source_z_origin = handoff_z0 - .5*handoff_dz;
  double left = horizontal - .5*cell_size;
  double right = horizontal + .5*cell_size;
  double lower = vertical - .5*cell_size;
  double upper = vertical + .5*cell_size;
  int first_i = (int) floor((left - source_x_origin)/handoff_dx);
  int last_i = (int) ceil((right - source_x_origin)/handoff_dx) - 1;
  int first_k = max(0, (int) floor((lower - source_z_origin)/handoff_dz));
  int last_k = min(HANDOFF_NZ - 1,
                   (int) ceil((upper - source_z_origin)/handoff_dz) - 1);
  if (last_k < first_k)
    return 0.;
  double liquid_area = 0.;
  for (int raw_i = first_i; raw_i <= last_i; raw_i++) {
    double source_left = source_x_origin + raw_i*handoff_dx;
    double overlap_x = max(0., min(right, source_left + handoff_dx) -
                           max(left, source_left));
    if (overlap_x <= 0.)
      continue;
    int i = (raw_i%HANDOFF_NX + HANDOFF_NX)%HANDOFF_NX;
    for (int k = first_k; k <= last_k; k++) {
      double source_lower = source_z_origin + k*handoff_dz;
      double overlap_z = max(0., min(upper, source_lower + handoff_dz) -
                             max(lower, source_lower));
      if (overlap_z > 0.)
        liquid_area += overlap_x*overlap_z*
          handoff_fraction[k*HANDOFF_NX + i];
    }
  }
  return clamp(liquid_area/sq(cell_size), 0., 1.);
}
#endif

#if HANDOFF_HAS_MAC_STREAMFUNCTION
double sample_mac_streamfunction (double horizontal, double vertical)
{
  horizontal = fmod(horizontal, domain_length);
  if (horizontal < 0.)
    horizontal += domain_length;
  double gx = (horizontal - (handoff_x0 - 0.5*handoff_dx))/handoff_dx;
  double gz = (vertical - (handoff_z0 - 0.5*handoff_dz))/handoff_dz;
  if (gz < 0. || gz > HANDOFF_NZ)
    return 0.;
  int i0raw = (int) floor(gx);
  int i0 = (i0raw % HANDOFF_NX + HANDOFF_NX) % HANDOFF_NX;
  int i1 = (i0 + 1) % HANDOFF_NX;
  int k0 = (int) floor(gz);
  double ax = gx - i0raw;
  if (k0 < 0) k0 = 0;
  if (k0 >= HANDOFF_NZ) k0 = HANDOFF_NZ - 1;
  int k1 = k0 + 1;
  double az = clamp(gz - k0, 0., 1.);
  double lower = lerp(handoff_mac_streamfunction[k0*HANDOFF_NX + i0],
                      handoff_mac_streamfunction[k0*HANDOFF_NX + i1], ax);
  double upper = lerp(handoff_mac_streamfunction[k1*HANDOFF_NX + i0],
                      handoff_mac_streamfunction[k1*HANDOFF_NX + i1], ax);
  return lerp(lower, upper, az);
}
#endif

#if HANDOFF_HAS_STREAMFUNCTION || HANDOFF_HAS_MAC_STREAMFUNCTION
double sample_handoff_streamfunction (double horizontal, double vertical)
{
#if HANDOFF_HAS_MAC_STREAMFUNCTION
  return sample_mac_streamfunction(horizontal, vertical);
#else
  return sample_grid(handoff_streamfunction, horizontal, vertical);
#endif
}
#endif

#if (HANDOFF_HAS_STREAMFUNCTION || HANDOFF_HAS_MAC_STREAMFUNCTION) && COMPATIBLE_EMBEDDED_FLUX
void compatible_embedded_face_flux (double shift_x, double shift_y)
{
  /* uf stores volume flux per full face length.  Integrating
     (psi_y,-psi_x) over the open segment gives endpoint differences and makes
     the finite-volume divergence telescope in cut cells when psi is constant
     on the embedded bed. */
  foreach_face(x) {
    double open = fs.x[];
    double upper = y + Delta/2.;
    double lower = upper - open*Delta; /* liquid lies above the bed */
    uf.x[] = open > 0. ?
      (sample_handoff_streamfunction(x, upper) -
       sample_handoff_streamfunction(x, lower))/Delta +
      open*(handoff_velocity_shift_x + shift_x) : 0.;
  }
  foreach_face(y) {
    double open = fs.y[];
    double left = x - Delta/2., right = x + Delta/2.;
    if (open <= 0.)
      uf.y[] = 0.;
    else {
      bool liquid_left = y >= bottom_height(left);
      double liquid_a = liquid_left ? left : right - open*Delta;
      double liquid_b = liquid_left ? left + open*Delta : right;
      uf.y[] = -(sample_handoff_streamfunction(liquid_b, y) -
                 sample_handoff_streamfunction(liquid_a, y))/Delta +
        open*(handoff_velocity_shift_z + shift_y);
    }
  }
  boundary ((scalar *){uf});
}
#endif

double bottom_height (double horizontal)
{
  horizontal = fmod(horizontal, domain_length);
  if (horizontal < 0.) horizontal += domain_length;
  int upper = 1;
  while (upper < HANDOFF_NB && handoff_bottom_x[upper] < horizontal)
    upper++;
  if (upper == HANDOFF_NB) {
    double x0 = handoff_bottom_x[HANDOFF_NB - 1];
    double x1 = handoff_bottom_x[0] + domain_length;
    return lerp(handoff_bottom_z[HANDOFF_NB - 1], handoff_bottom_z[0],
                (horizontal - x0)/(x1 - x0));
  }
  return lerp(handoff_bottom_z[upper - 1], handoff_bottom_z[upper],
              (horizontal - handoff_bottom_x[upper - 1])/
              (handoff_bottom_x[upper] - handoff_bottom_x[upper - 1]));
}

void polygon_vertex (int index, double * horizontal, double * vertical)
{
  double start = handoff_surface_x[0];
  double end = start + domain_length;
  if (index < HANDOFF_NS) {
    *horizontal = handoff_surface_x[index];
    *vertical = handoff_surface_z[index];
  }
  else if (index == HANDOFF_NS) {
    *horizontal = end;
    *vertical = handoff_surface_z[0];
  }
  else if (index == HANDOFF_NS + 1) {
    *horizontal = end;
    *vertical = bottom_height(end);
  }
  else if (index < HANDOFF_NS + HANDOFF_NB + 2) {
    int bottom_index = HANDOFF_NB - 1 - (index - HANDOFF_NS - 2);
    *horizontal = handoff_bottom_x[bottom_index];
    *vertical = handoff_bottom_z[bottom_index];
  }
  else {
    *horizontal = start;
    *vertical = bottom_height(start);
  }
}

double liquid_level (double horizontal, double vertical)
{
  int vertices = HANDOFF_NS + HANDOFF_NB + 3;
  bool inside = false;
  double minimum_squared = HUGE;
  double ax, ay;
  polygon_vertex(vertices - 1, &ax, &ay);
  for (int index = 0; index < vertices; index++) {
    double bx, by;
    polygon_vertex(index, &bx, &by);
    if (((ay > vertical) != (by > vertical)) &&
        horizontal < (bx - ax)*(vertical - ay)/(by - ay) + ax)
      inside = !inside;
    double dx = bx - ax, dy = by - ay;
    double denominator = sq(dx) + sq(dy);
    double parameter = denominator > 0.
      ? clamp(((horizontal - ax)*dx + (vertical - ay)*dy)/denominator, 0., 1.)
      : 0.;
    double ex = horizontal - (ax + parameter*dx);
    double ey = vertical - (ay + parameter*dy);
    minimum_squared = min(minimum_squared, sq(ex) + sq(ey));
    ax = bx; ay = by;
  }
  return (inside ? 1. : -1.)*sqrt(minimum_squared);
}

int main()
{
  size (domain_length);
  origin (0., vertical_origin);
  periodic (right);
  rho1 = 1.;
  rho2 = DENSITY_RATIO;
  mu1 = 1./REYNOLDS;
  mu2 = mu1*VISCOSITY_RATIO;
  f.sigma = SURFACE_TENSION;
  G.y = -1.;
  TOLERANCE = PROJECTION_TOLERANCE;
  /* Sample the transferred state at the target resolution once; subsequent
     adaptation may coarsen it conservatively. */
  init_grid (1 << LEVEL);
  run();
}

#if CONSERVATIVE_Q_EMBED && TREE
/* q is the sole conservative liquid measure.  Conditional f is derived from
   q after adaptation, so no mutation of Basilisk's global scalar order is
   needed here. */
event q_embed_amr_defaults (i = 0, last)
{
  q_embed_configure_fraction_amr (f);
}
#endif

event init (i = 0)
{
  solid (cs, fs, y - bottom_height(x));
  fraction (beach, bottom_height(x) - y);
#if CONSERVATIVE_Q_EMBED
  double remap_raw_mass = 0., remap_clipped_mass = 0., remap_l1_clip = 0.;
  double remap_maximum_violation = 0.;
  foreach (reduction(+:remap_raw_mass) reduction(+:remap_clipped_mass)
           reduction(+:remap_l1_clip) reduction(max:remap_maximum_violation)) {
    double raw_q = remap_handoff_aperture (x, y, Delta);
    q_liquid[] = clamp(raw_q, 0., cs[]);
    f[] = cs[] > 0. ? q_liquid[]/cs[] : 0.;
    /* q is a full-cell aperture.  Basilisk's embedded dv() already contains
       cs, so q*dv() would apply the aperture twice. */
    remap_raw_mass += raw_q*sq(Delta);
    remap_clipped_mass += q_liquid[]*sq(Delta);
    remap_l1_clip += fabs(q_liquid[] - raw_q)*sq(Delta);
    remap_maximum_violation = max(remap_maximum_violation, raw_q - cs[]);
  }
  boundary ({f, q_liquid});
  fprintf (stderr, "Q_CONSERVATIVE_REMAP %.17g %.17g %.17g %.17g %.17g\n",
           remap_raw_mass, remap_clipped_mass, handoff_target_volume,
           remap_l1_clip, remap_maximum_violation);
#if TREE
  q_embed_configure_fraction_amr (f);
#endif
#else
  fraction (f, liquid_level(x, y));
#endif
  foreach() {
#if HANDOFF_HAS_MAC_STREAMFUNCTION || HANDOFF_HAS_STREAMFUNCTION
    /* Differentiate the same interpolated scalar on the receiver grid.  On
       regular same-level cells the centered derivatives commute, and the
       centered solver's subsequent cell-to-face averaging therefore retains
       a telescoping, divergence-free flux. */
    u.x[] = ((sample_handoff_streamfunction(x, y + Delta) -
              sample_handoff_streamfunction(x, y - Delta))/(2.*Delta) +
             handoff_velocity_shift_x);
    u.y[] = (-(sample_handoff_streamfunction(x + Delta, y) -
               sample_handoff_streamfunction(x - Delta, y))/(2.*Delta) +
             handoff_velocity_shift_z);
#else
    u.x[] = sample_grid(handoff_u, x, y);
    u.y[] = sample_grid(handoff_w, x, y);
#endif
  }
  boundary ((scalar *){u, f, beach, cs});

  /* Correct interpolation error on the actual quadtree using interface-cell
     capacity, then restore the two BIE momentum components by a uniform
     Galilean shift. */
  double mass = 0., initial_mass_repair_l1 = 0.;
  for (int correction = 0; correction < 5; correction++) {
    mass = 0.;
#if CONSERVATIVE_Q_EMBED
    foreach (reduction(+:mass))
      mass += q_liquid[]*sq(Delta);
    double difference = handoff_target_volume - mass;
    double capacity = 0.;
    foreach (reduction(+:capacity)) {
      double room = difference > 0. ? cs[] - q_liquid[] : q_liquid[];
      if (q_liquid[] > 0. && q_liquid[] < cs[] && room > 0.)
        capacity += room*sq(Delta);
    }
    if (fabs(difference) > 0. && capacity > 0.)
      foreach (reduction(+:initial_mass_repair_l1)) {
        double room = difference > 0. ? cs[] - q_liquid[] : q_liquid[];
        if (q_liquid[] > 0. && q_liquid[] < cs[] && room > 0.) {
          double old_q = q_liquid[];
          q_liquid[] = clamp(q_liquid[] + difference*room/capacity,
                             0., cs[]);
          initial_mass_repair_l1 +=
            fabs(q_liquid[] - old_q)*sq(Delta);
          f[] = cs[] > 0. ? q_liquid[]/cs[] : 0.;
        }
      }
#else
    foreach (reduction(+:mass))
      mass += f[]*dv();
    double difference = handoff_target_volume - mass;
    double capacity = 0.;
    foreach (reduction(+:capacity))
      if (f[] > 0. && f[] < 1.)
        capacity += (difference > 0. ? 1. - f[] : f[])*dv();
    if (fabs(difference) > 0. && capacity > 0.)
      foreach() {
        double room = difference > 0. ? 1. - f[] : f[];
        if (f[] > 0. && f[] < 1.)
          /* capacity already contains cs*dv; multiplying by cs again would
             under-correct embedded cut cells and make the identity
             resolution dependent. */
          f[] = clamp(f[] + difference*room/capacity, 0., 1.);
      }
#endif
  }
  boundary ({f
#if CONSERVATIVE_Q_EMBED
             , q_liquid
#endif
             });
#if CONSERVATIVE_Q_EMBED
  fprintf (stderr, "Q_INITIAL_MASS_REPAIR %.17g %.17g\n",
           initial_mass_repair_l1,
           initial_mass_repair_l1/max(handoff_target_volume, 1e-30));
#endif
  mass = 0.;
  double px = 0., py = 0.;
  foreach (reduction(+:mass) reduction(+:px) reduction(+:py)) {
    double active = f[];
    mass += active*dv();
    px += active*u.x[]*dv();
    py += active*u.y[]*dv();
  }
  q_reference_mass = mass;
#if RECEIVER_EXPORT_INITIAL_STATE
  /* Serial, pre-correction receiver quadrature for the offline constrained
     momentum audit.  q=f*cs is the representation currently consumed by the
     solver; the separate columns are retained so a geometric-q replacement
     can later be audited without changing the file schema. */
  FILE * receiver_state = fopen (RECEIVER_EXPORT_INITIAL_STATE_PATH, "w");
  if (!receiver_state) {
    perror (RECEIVER_EXPORT_INITIAL_STATE_PATH);
    exit (1);
  }
  fprintf (receiver_state, "x,y,delta,cell_area,f,cs,q,u_x,u_y\n");
  foreach (serial)
    if (f[]*cs[] > 1e-14)
      fprintf (receiver_state,
               "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
               x, y, Delta, sq(Delta), f[], cs[],
#if CONSERVATIVE_Q_EMBED
               q_liquid[],
#else
               f[]*cs[],
#endif
               u.x[], u.y[]);
  fclose (receiver_state);
#endif
  double shift_x = 0., shift_y = 0.;
#if !HANDOFF_MOMENTUM_EMBEDDED || RECEIVER_RESIDUAL_MOMENTUM_CORRECTION
  shift_x = (handoff_target_momentum_x - px)/mass;
  shift_y = (handoff_target_momentum_z - py)/mass;
#endif
  double kinetic_before_shift = 0.;
  foreach (reduction(+:kinetic_before_shift)) {
    double active = f[];
    kinetic_before_shift += 0.5*active*(sq(u.x[]) + sq(u.y[]))*dv();
  }
#if (HANDOFF_HAS_STREAMFUNCTION || HANDOFF_HAS_MAC_STREAMFUNCTION) && COMPATIBLE_EMBEDDED_FLUX
  restart_shift_x = shift_x;
  restart_shift_y = shift_y;
#endif
  foreach() {
    /* Apply the Galilean correction to both phases.  Restricting it to
       liquid cells would create an artificial velocity jump at the interface
       and destroy the streamfunction extension's solenoidal property. */
    if (cs[] > 0.) {
      u.x[] += shift_x;
      u.y[] += shift_y;
    }
  }
  boundary ((scalar *){u, f});

  double px_after = 0., py_after = 0., kinetic_after_shift = 0.;
  foreach (reduction(+:px_after) reduction(+:py_after)
           reduction(+:kinetic_after_shift)) {
    double active = f[];
    px_after += active*u.x[]*dv();
    py_after += active*u.y[]*dv();
    kinetic_after_shift += 0.5*active*(sq(u.x[]) + sq(u.y[]))*dv();
  }
  double target_momentum_scale =
    max(hypot(handoff_target_momentum_x, handoff_target_momentum_z), 1e-30);
  double velocity_norm = sqrt(max(2.*kinetic_before_shift, 1e-30));
  fprintf (stderr,
           "INIT_RECEIVER_CORRECTION %.12g %d %.12g %.12g %.12g %.12g %.12g %.12g %.12g %.12g %.12g %.12g\n",
           t, i,
           hypot(px - handoff_target_momentum_x,
                 py - handoff_target_momentum_z)/target_momentum_scale,
           hypot(px_after - handoff_target_momentum_x,
                 py_after - handoff_target_momentum_z)/target_momentum_scale,
           shift_x, shift_y,
           sqrt(mass*sq(shift_x) + mass*sq(shift_y))/velocity_norm,
           fabs(kinetic_after_shift - kinetic_before_shift)/
             max(kinetic_before_shift, 1e-30),
           px, py, px_after, py_after);

#if RECEIVER_INITIALIZATION_ONLY
  return 1;
#endif
}

#if (HANDOFF_HAS_STREAMFUNCTION || HANDOFF_HAS_MAC_STREAMFUNCTION) && COMPATIBLE_EMBEDDED_FLUX
/* This inherited hook executes immediately before centered.h's projection.
   The initial advection predictor uses the compatible cell field; the final
   face projection at i=0 receives exact open-segment streamfunction fluxes. */
event projection (i++, last)
{
  if (i == 0)
    compatible_embedded_face_flux (restart_shift_x, restart_shift_y);
}
#endif

event transfer_diagnostics (i = 0; i <= 2; i++)
{
  double mass = 0., px = 0., py = 0., div2 = 0., divmax = 0.;
  double face_div2 = 0., face_divmax = 0.;
  long cells = 0;
  foreach (reduction(+:mass) reduction(+:px) reduction(+:py)
           reduction(+:div2) reduction(max:divmax)
           reduction(+:face_div2) reduction(max:face_divmax)
           reduction(+:cells)) {
    double active = f[];
    mass += active*dv();
    px += active*u.x[]*dv();
    py += active*u.y[]*dv();
    if (active > 0.999) {
      double divergence = (u.x[1] - u.x[-1] + u.y[0,1] - u.y[0,-1])/(2.*Delta);
      div2 += sq(divergence)*dv();
      divmax = max(divmax, fabs(divergence));
      double face_divergence = (uf.x[1] - uf.x[] + uf.y[0,1] - uf.y[])/Delta;
      face_div2 += sq(face_divergence)*dv();
      face_divmax = max(face_divmax, fabs(face_divergence));
      cells++;
    }
  }
  fprintf (stderr, "TRANSFER %.12g %d %.12g %.12g %.12g %.12g %.12g %.12g %.12g %ld\n",
           t, i, mass, px, py, sqrt(div2/max(mass, 1e-30)), divmax,
           sqrt(face_div2/max(mass, 1e-30)), face_divmax, cells);
}

/* The ordinary transfer record at i=0 is deliberately before the first
   pressure projection and verifies interpolation plus conservative
   corrections.  This last event records the face field after all solver
   events at the same iteration, so the projection residual is not confused
   with event ordering. */
event projected_transfer_diagnostics (i++, last)
{
  if (i <= 2) {
    double mass = 0., px = 0., py = 0., div2 = 0., divmax = 0.;
    double face_div2 = 0., face_divmax = 0.;
    long cells = 0;
    foreach (reduction(+:mass) reduction(+:px) reduction(+:py)
             reduction(+:div2) reduction(max:divmax)
             reduction(+:face_div2) reduction(max:face_divmax)
             reduction(+:cells)) {
    double active = f[];
      mass += active*dv();
      px += active*u.x[]*dv();
      py += active*u.y[]*dv();
      if (active > 0.999) {
        double divergence =
          (u.x[1] - u.x[-1] + u.y[0,1] - u.y[0,-1])/(2.*Delta);
        div2 += sq(divergence)*dv();
        divmax = max(divmax, fabs(divergence));
        double face_divergence =
          (uf.x[1] - uf.x[] + uf.y[0,1] - uf.y[])/Delta;
        face_div2 += sq(face_divergence)*dv();
        face_divmax = max(face_divmax, fabs(face_divergence));
        cells++;
      }
    }
    fprintf (stderr,
             "PROJECTED %.12g %d %.12g %.12g %.12g %.12g %.12g %.12g %.12g %ld\n",
             t, i, mass, px, py, sqrt(div2/max(mass, 1e-30)), divmax,
             sqrt(face_div2/max(mass, 1e-30)), face_divmax, cells);
  }
}

/* A clean t=0 operator audit avoids entering a deliberately under-resolved
   physical step merely to verify generated receiver quadrature identities. */
event receiver_initialization_only_stop (i = 0, last)
{
#if RECEIVER_INITIALIZATION_ONLY
  return 1;
#endif
}

/* A tree adaptation changes the interpolation stencil of both cell and face
   velocities after centered.h's ordinary end-of-step projection.  For the
   MAC handoff audit we optionally apply a pure Helmholtz correction on the
   adapted mesh.  This is not another physical acceleration step: the
   provisional face field is reconstructed from the already advanced cell
   velocity, projected with unit pseudo-time, and the same pressure-only
   correction is averaged back to cell centres. */
static void reproject_after_adaptation (int iteration)
{
#if POST_ADAPT_REPROJECT
#if CONSERVATIVE_Q_EMBED
  scalar q_before_cleanup[];
  foreach()
    q_before_cleanup[] = q_liquid[];
#endif
  fractions_cleanup (cs, fs);
#if CONSERVATIVE_Q_EMBED
  double cleanup_clipped = 0., cleanup_bound = 0.;
  foreach (reduction(+:cleanup_clipped) reduction(max:cleanup_bound)) {
    double violation = max(-q_before_cleanup[], q_before_cleanup[] - cs[]);
    cleanup_bound = max(cleanup_bound, violation);
    double restored_q = clamp(q_before_cleanup[], 0., cs[]);
    cleanup_clipped += fabs(restored_q - q_before_cleanup[])*sq(Delta);
    f[] = cs[] > 0. ? restored_q/cs[] : 0.;
    q_liquid[] = restored_q;
  }
  boundary ({f, q_liquid});
  fprintf (stderr, "Q_CLEANUP %.12g %d %.12g %.12g\n",
           t, iteration, cleanup_clipped, cleanup_bound);
#endif
  event ("properties");
#if ADAPT_VOLUME_LEDGER
  double ledger_after_cleanup = 0.;
  foreach (reduction(+:ledger_after_cleanup))
    ledger_after_cleanup += f[]*dv();
  fprintf (stderr, "AMR_VOLUME_STAGE %.12g %d 2 %.17g %.17g\n",
           t, iteration, ledger_after_cleanup,
           (ledger_after_cleanup - handoff_target_volume)/handoff_target_volume);
#endif
#if !POST_ADAPT_PRESERVE_FACE_FLUX
  trash ({uf});
  foreach_face()
    uf.x[] = fm.x[]*face_value (u.x, 0);
  boundary ((scalar *){uf});
#endif
  double pre_face_div2 = 0., pre_face_divmax = 0., pre_liquid_volume = 0.;
  foreach (reduction(+:pre_face_div2) reduction(max:pre_face_divmax)
           reduction(+:pre_liquid_volume)) {
    double active = f[];
    if (active > 0.999) {
      double divergence =
        (uf.x[1] - uf.x[] + uf.y[0,1] - uf.y[])/Delta;
      pre_face_div2 += sq(divergence)*dv();
      pre_face_divmax = max(pre_face_divmax, fabs(divergence));
      pre_liquid_volume += dv();
    }
  }
  double pre_face_rms =
    sqrt(pre_face_div2/max(pre_liquid_volume, 1e-30));
  if (POST_ADAPT_REPROJECT_TRIGGER > 0. &&
      pre_face_rms <= POST_ADAPT_REPROJECT_TRIGGER &&
      pre_face_divmax <= POST_ADAPT_REPROJECT_TRIGGER) {
    fprintf (stderr, "POSTADAPT_SKIP %.12g %d %.12g %.12g\n",
             t, iteration, pre_face_rms, pre_face_divmax);
#if ADAPT_VOLUME_LEDGER
    fprintf (stderr, "AMR_VOLUME_STAGE %.12g %d 3 %.17g %.17g\n",
             t, iteration, ledger_after_cleanup,
             (ledger_after_cleanup - handoff_target_volume)/handoff_target_volume);
#endif
    return;
  }
  double momentum_x_before = 0., momentum_y_before = 0.;
  double kinetic_before = 0.;
  foreach (reduction(+:momentum_x_before) reduction(+:momentum_y_before)
           reduction(+:kinetic_before)) {
    double active = f[];
    momentum_x_before += active*u.x[]*dv();
    momentum_y_before += active*u.y[]*dv();
    kinetic_before += 0.5*active*(sq(u.x[]) + sq(u.y[]))*dv();
  }
  mgstats post = project (uf, p, alpha, 1., mgp.nrelax);
  face vector pressure_correction[];
  foreach_face()
    pressure_correction.x[] = -alpha.x[]*(p[] - p[-1])/Delta;
  foreach()
    foreach_dimension()
      u.x[] += (pressure_correction.x[] + pressure_correction.x[1])/
               (fm.x[] + fm.x[1] + SEPS);
  /* project() already leaves the face field in flux-compatible form.
     Reapplying generic face boundary prolongation here can destroy the
     coarse--fine cancellation just established by the Poisson solve. */
  boundary ((scalar *){u});
  double momentum_x_after = 0., momentum_y_after = 0.;
  double kinetic_after = 0.;
  foreach (reduction(+:momentum_x_after) reduction(+:momentum_y_after)
           reduction(+:kinetic_after)) {
    double active = f[];
    momentum_x_after += active*u.x[]*dv();
    momentum_y_after += active*u.y[]*dv();
    kinetic_after += 0.5*active*(sq(u.x[]) + sq(u.y[]))*dv();
  }
  double momentum_scale =
    max(hypot(momentum_x_before, momentum_y_before), 1e-30);
  double relative_momentum_change =
    hypot(momentum_x_after - momentum_x_before,
          momentum_y_after - momentum_y_before)/momentum_scale;
  double relative_kinetic_change =
    fabs(kinetic_after - kinetic_before)/max(kinetic_before, 1e-30);
  double face_div2 = 0., face_divmax = 0., liquid_volume = 0.;
  foreach (reduction(+:face_div2) reduction(max:face_divmax)
           reduction(+:liquid_volume)) {
    double active = f[];
    if (active > 0.999) {
      double divergence =
        (uf.x[1] - uf.x[] + uf.y[0,1] - uf.y[])/Delta;
      face_div2 += sq(divergence)*dv();
      face_divmax = max(face_divmax, fabs(divergence));
      liquid_volume += dv();
    }
  }
  fprintf (stderr,
           "POSTADAPT_REPROJECT %.12g %d %d %.12g %.12g %.12g %.12g %.12g %.12g %.12g\n",
           t, iteration, post.i, post.resa,
           sqrt(face_div2/max(liquid_volume, 1e-30)), face_divmax,
           relative_momentum_change, relative_kinetic_change,
           pre_face_rms, pre_face_divmax);
#if ADAPT_VOLUME_LEDGER
  double ledger_after_projection = 0.;
  foreach (reduction(+:ledger_after_projection))
    ledger_after_projection += f[]*dv();
  fprintf (stderr, "AMR_VOLUME_STAGE %.12g %d 3 %.17g %.17g\n",
           t, iteration, ledger_after_projection,
           (ledger_after_projection - handoff_target_volume)/handoff_target_volume);
#endif
#endif
}

/* Optional conservative repair for the liquid-volume defect introduced by
   quadtree restriction/prolongation near a steep VOF interface.  The target
   volume is fixed by the accepted BIE handoff.  Only mixed cells are changed,
   in proportion to their available room, and every induced momentum/energy
   change is reported so this cannot act as hidden interface forcing. */
static void correct_volume_after_adaptation (int iteration)
{
#if POST_ADAPT_VOLUME_CORRECTION
  double mass_before = 0., momentum_x_before = 0., momentum_y_before = 0.;
  double kinetic_before = 0.;
  foreach (reduction(+:mass_before) reduction(+:momentum_x_before)
           reduction(+:momentum_y_before) reduction(+:kinetic_before)) {
    double active = f[];
    mass_before += active*dv();
    momentum_x_before += active*u.x[]*dv();
    momentum_y_before += active*u.y[]*dv();
    kinetic_before += 0.5*active*(sq(u.x[]) + sq(u.y[]))*dv();
  }
  double l1_change = 0.;
  for (int correction = 0; correction < 5; correction++) {
    double mass = 0.;
    foreach (reduction(+:mass))
      mass += f[]*dv();
    double difference = handoff_target_volume - mass;
    if (fabs(difference) <=
        POST_ADAPT_VOLUME_TOLERANCE*handoff_target_volume)
      break;
    double capacity = 0.;
    foreach (reduction(+:capacity))
      if (f[] > 0. && f[] < 1.) {
        double room = difference > 0. ? 1. - f[] : f[];
        capacity += room*dv();
      }
    if (capacity <= 0.)
      break;
    foreach (reduction(+:l1_change))
      if (f[] > 0. && f[] < 1.) {
        double old_fraction = f[];
        double room = difference > 0. ? 1. - f[] : f[];
        f[] = clamp(f[] + difference*room/capacity, 0., 1.);
        l1_change += fabs(f[] - old_fraction)*dv();
      }
    boundary ({f});
  }
  double mass_after = 0., momentum_x_after = 0., momentum_y_after = 0.;
  double kinetic_after = 0.;
  foreach (reduction(+:mass_after) reduction(+:momentum_x_after)
           reduction(+:momentum_y_after) reduction(+:kinetic_after)) {
    double active = f[];
    mass_after += active*dv();
    momentum_x_after += active*u.x[]*dv();
    momentum_y_after += active*u.y[]*dv();
    kinetic_after += 0.5*active*(sq(u.x[]) + sq(u.y[]))*dv();
  }
  double momentum_scale =
    max(hypot(momentum_x_before, momentum_y_before), 1e-30);
  fprintf (stderr,
           "POSTADAPT_VOLUME %.12g %d %.12g %.12g %.12g %.12g %.12g %.12g %.12g\n",
           t, iteration, mass_before, mass_after, handoff_target_volume,
           l1_change/handoff_target_volume,
           fabs(mass_after - handoff_target_volume)/handoff_target_volume,
           hypot(momentum_x_after - momentum_x_before,
                 momentum_y_after - momentum_y_before)/momentum_scale,
           fabs(kinetic_after - kinetic_before)/max(kinetic_before, 1e-30));
#if POST_ADAPT_VOLUME_INVARIANT_COMPENSATION
  double mean_x_after = momentum_x_after/max(mass_after, 1e-30);
  double mean_y_after = momentum_y_after/max(mass_after, 1e-30);
  double target_mean_x = momentum_x_before/max(mass_after, 1e-30);
  double target_mean_y = momentum_y_before/max(mass_after, 1e-30);
  double variance_after =
    max(2.*kinetic_after - mass_after*(sq(mean_x_after) + sq(mean_y_after)),
        1e-30);
  double target_variance =
    2.*kinetic_before -
    mass_after*(sq(target_mean_x) + sq(target_mean_y));
  double velocity_scale =
    sqrt(max(target_variance, 0.)/variance_after);
  double velocity_delta2 = 0., velocity_norm2 = 0.;
  foreach (reduction(+:velocity_delta2) reduction(+:velocity_norm2)) {
    double active = f[];
    double new_x = velocity_scale*(u.x[] - mean_x_after) + target_mean_x;
    double new_y = velocity_scale*(u.y[] - mean_y_after) + target_mean_y;
    velocity_delta2 += active*(sq(new_x - u.x[]) + sq(new_y - u.y[]))*dv();
    velocity_norm2 += active*(sq(u.x[]) + sq(u.y[]))*dv();
    if (cs[] > 0.) {
      u.x[] = new_x;
      u.y[] = new_y;
    }
  }
#if POST_ADAPT_VOLUME_TRANSFORM_FACE_FLUX
  foreach_face(x)
    uf.x[] = velocity_scale*(uf.x[] - fm.x[]*mean_x_after) +
             fm.x[]*target_mean_x;
  foreach_face(y)
    uf.y[] = velocity_scale*(uf.y[] - fm.y[]*mean_y_after) +
             fm.y[]*target_mean_y;
#endif
  boundary ((scalar *){u});
  double momentum_x_compensated = 0., momentum_y_compensated = 0.;
  double kinetic_compensated = 0.;
  foreach (reduction(+:momentum_x_compensated)
           reduction(+:momentum_y_compensated)
           reduction(+:kinetic_compensated)) {
    double active = f[];
    momentum_x_compensated += active*u.x[]*dv();
    momentum_y_compensated += active*u.y[]*dv();
    kinetic_compensated += 0.5*active*(sq(u.x[]) + sq(u.y[]))*dv();
  }
  fprintf (stderr,
           "POSTADAPT_INVARIANTS %.12g %d %.12g %.12g %.12g %.12g %.12g %.12g\n",
           t, iteration, velocity_scale,
           target_mean_x - velocity_scale*mean_x_after,
           target_mean_y - velocity_scale*mean_y_after,
           sqrt(velocity_delta2/max(velocity_norm2, 1e-30)),
           hypot(momentum_x_compensated - momentum_x_before,
                 momentum_y_compensated - momentum_y_before)/momentum_scale,
           fabs(kinetic_compensated - kinetic_before)/
           max(kinetic_before, 1e-30));
#endif
#endif
}

event adapt (i++)
{
  if (i >= INITIAL_ADAPT_ITERATION)
    {
      if (i > INITIAL_ADAPT_ITERATION &&
          (!RECEIVER_ADAPT_EVERY_STEP ||
           (i - INITIAL_ADAPT_ITERATION) % RECEIVER_ADAPT_INTERVAL != 0))
        return 0;
#if ADAPT_VOLUME_LEDGER
      double ledger_before_adapt = 0.;
      foreach (reduction(+:ledger_before_adapt))
        ledger_before_adapt += f[]*dv();
      fprintf (stderr, "AMR_VOLUME_STAGE %.12g %d 0 %.17g %.17g\n",
               t, i, ledger_before_adapt,
               (ledger_before_adapt - handoff_target_volume)/handoff_target_volume);
#endif
#if CONSERVATIVE_Q_EMBED
      astats changed = adapt_wavelet ((scalar *){cs, q_liquid, u.x, u.y},
                                      (double[]){ADAPT_CS_TOLERANCE,
                                                 ADAPT_F_TOLERANCE,
                                                 ADAPT_U_TOLERANCE,
                                                 ADAPT_U_TOLERANCE},
                                      LEVEL, MINLEVEL);
      foreach()
        f[] = cs[] > 0. ? clamp(q_liquid[]/cs[], 0., 1.) : 0.;
      boundary ({f});
#else
      astats changed = adapt_wavelet ((scalar *){f, u.x, u.y, cs},
                                      (double[]){ADAPT_F_TOLERANCE,
                                                 ADAPT_U_TOLERANCE,
                                                 ADAPT_U_TOLERANCE,
                                                 ADAPT_CS_TOLERANCE},
                                      LEVEL, MINLEVEL);
#endif
#if ADAPT_VOLUME_LEDGER
      double ledger_after_adapt = 0.;
      foreach (reduction(+:ledger_after_adapt))
        ledger_after_adapt += f[]*dv();
      fprintf (stderr, "AMR_VOLUME_STAGE %.12g %d 1 %.17g %.17g\n",
               t, i, ledger_after_adapt,
               (ledger_after_adapt - handoff_target_volume)/handoff_target_volume);
#endif
      if (changed.nf || changed.nc) {
        correct_volume_after_adaptation(i);
        reproject_after_adaptation(i);
      }
    }
}

event frames (t = 0.; t += 0.02; t <= END_TIME)
{
  static int frame = 0;
  char name[128], vorticity_name[128];
  sprintf (name, "handoff_frames/vof-%05d.ppm", frame++);
  FILE * image = fopen (name, "w");
  output_ppm (f, image, n = FRAME_N, min = 0., max = 1.,
              linear = true, mask = beach,
              box = {{FRAME_X_MIN, FRAME_Y_MIN}, {FRAME_X_MAX, FRAME_Y_MAX}});
  fclose (image);

  scalar omega[];
  vorticity (u, omega);
  sprintf (vorticity_name, "handoff_vorticity/vorticity-%05d.ppm", frame - 1);
  FILE * vortex_image = fopen (vorticity_name, "w");
  output_ppm (omega, vortex_image, n = FRAME_N,
              min = -VORTICITY_COLOR_LIMIT, max = VORTICITY_COLOR_LIMIT,
              linear = true, mask = beach,
              box = {{FRAME_X_MIN, FRAME_Y_MIN}, {FRAME_X_MAX, FRAME_Y_MAX}});
  fclose (vortex_image);
}

event runtime_diagnostics (t += 0.02; t <= END_TIME)
{
  double mass = 0., kinetic = 0.;
  foreach (reduction(+:mass) reduction(+:kinetic)) {
    double active = f[];
    mass += active*dv();
    kinetic += 0.5*rho[]*(sq(u.x[]) + sq(u.y[]))*active*dv();
  }
  fprintf (stderr, "RUN %.12g %d %.12g %.12g\n", t, i, mass, kinetic);
#if CONSERVATIVE_Q_EMBED
  double lower_violation = 0., upper_violation = 0.;
  foreach (reduction(max:lower_violation) reduction(max:upper_violation)) {
    double q = f[]*cs[];
    lower_violation = max(lower_violation, -q);
    upper_violation = max(upper_violation, q - cs[]);
  }
  fprintf (stderr,
           "Q_RUN %.12g %d %.17g %.12g %.12g %.12g %.12g %.12g %ld\n",
           t, i, mass,
           (mass - q_reference_mass)/max(q_reference_mass, 1e-30),
           lower_violation, upper_violation,
           q_embed_diagnostics.maximum_open_flux_mismatch,
           q_embed_diagnostics.clipped_aperture,
           q_embed_diagnostics.corrected_cells);
  fprintf (stderr,
           "Q_CUTCELL_CFL %.12g %d %.12g %.12g %.12g %ld %ld\n",
           t, i,
           q_embed_stability_diagnostics.candidate_timestep,
           q_embed_stability_diagnostics.limited_timestep,
           q_embed_stability_diagnostics.maximum_cutcell_courant,
           q_embed_stability_diagnostics.limited_cells,
           q_embed_stability_diagnostics.active_cut_cells);
#endif
}

/* Track run-up independently of trapped-air topology.  The left shoreline is
   isolated from the periodic wet copy by the diagnostic window. */
event shore_impact_diagnostics (t += 0.02; t <= END_TIME)
{
  double wetting_front = SHORE_DIAGNOSTIC_X_MIN;
  double embedded_pressure_peak = 0., embedded_speed_peak = 0.;
  foreach (reduction(max:wetting_front)
           reduction(max:embedded_pressure_peak)
           reduction(max:embedded_speed_peak))
    if (x >= SHORE_DIAGNOSTIC_X_MIN && x <= SHORE_DIAGNOSTIC_X_MAX &&
        cs[] > 0.999 && f[] > 0.5 &&
        y >= bottom_height(x) && y - bottom_height(x) < 2.*Delta) {
      wetting_front = max(wetting_front, x);
      embedded_pressure_peak = max(embedded_pressure_peak, fabs(p[]));
      embedded_speed_peak = max(embedded_speed_peak, hypot(u.x[], u.y[]));
    }
  fprintf (stderr, "SHORE %.12g %d %.12g %.12g %.12g\n",
           t, i, wetting_front, embedded_pressure_peak, embedded_speed_peak);
}

event stop (t = END_TIME)
{
  return 1;
}
