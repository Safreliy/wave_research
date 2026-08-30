/**
Inclined-wall manufactured translation for q=cs*f.

A compact polygonal liquid layer translates parallel to a planar embedded
wall.  The exact geometry is known, both directional sweeps are active and
the inclined cut cells include arbitrarily small open apertures as the grid is
refined.  This complements q_embed_manufactured.c, whose horizontal wall does
not exercise the embedded small-cell flux restriction.
*/

#include "grid/quadtree.h"
#include "embed.h"
#include "run.h"
#include "vof.h"

scalar f[], * interfaces = {f};
face vector uf[];

#include "q_embed_vof.h"

#ifndef LEVEL
# define LEVEL 7
#endif
#ifndef MINLEVEL
# define MINLEVEL (LEVEL - 2)
#endif
#ifndef END_TIME
# define END_TIME 0.35
#endif
#ifndef USE_AMR
# define USE_AMR 0
#endif
#ifndef MANUFACTURED_CFL
# define MANUFACTURED_CFL 0.20
#endif
#ifndef USE_CUTCELL_CFL
# define USE_CUTCELL_CFL 0
#endif
static const double wall_slope = 0.37;
static const double wall_intercept = 0.12;
static const double translation_speed = 0.24;
static const double patch_left = 0.18;
static const double patch_right = 0.52;
static const double layer_lower = -0.025;
static const double layer_upper = 0.19;
double manufactured_dt;

static inline double inclined_wall (double horizontal, double vertical)
{
  return vertical - wall_slope*horizontal - wall_intercept;
}

static inline double exact_liquid_level
  (double horizontal, double vertical, double time)
{
  double translated = horizontal - translation_speed*time;
  double distance = inclined_wall(horizontal, vertical);
  return min(min(translated - patch_left, patch_right - translated),
             min(distance - layer_lower, layer_upper - distance));
}

static void prescribed_flux (void)
{
  foreach_face(x)
    uf.x[] = fs.x[]*translation_speed;
  foreach_face(y)
    uf.y[] = fs.y[]*wall_slope*translation_speed;
  boundary ((scalar *){uf});
}

static double stable_timestep (void)
{
#if USE_CUTCELL_CFL
  return q_embed_stable_timestep (f, uf, manufactured_dt);
#else
  return manufactured_dt;
#endif
}

int main()
{
  size (1.);
  origin (0., 0.);
  manufactured_dt = MANUFACTURED_CFL*L0/
    (translation_speed*(1 << LEVEL));
  DT = manufactured_dt;
  init_grid (1 << LEVEL);
  run();
}

event q_amr_defaults (i = 0, last)
{
#if TREE
  q_embed_configure_fraction_amr (f);
#endif
}

event init (i = 0)
{
  solid (cs, fs, inclined_wall(x, y));
  fraction (f, exact_liquid_level(x, y, 0.));
  boundary ({f, cs});
  q_embed_initialize_geometric_fraction (f);
#if TREE
  q_embed_configure_fraction_amr (f);
#endif
  interfaces = NULL;
  prescribed_flux();
}

event manufactured_stability (i++, last)
  dt = dtnext(stable_timestep());

event q_transport (i++)
{
  prescribed_flux();
  dt = dtnext(stable_timestep());
  q_embed_vof_advection (f, i);
}

event q_adapt (i++)
{
#if USE_AMR
  double mass_before = 0.;
  foreach (reduction(+:mass_before))
    mass_before += q_liquid[]*sq(Delta);
  adapt_wavelet ({cs, q_liquid}, (double[]){1e-7, 2e-4},
                 LEVEL, MINLEVEL);
  foreach()
    f[] = cs[] > 0. ? clamp(q_liquid[]/cs[], 0., 1.) : 0.;
  boundary ({f, q_liquid});
  double mass_after = 0., upper = 0.;
  foreach (reduction(+:mass_after) reduction(max:upper)) {
    mass_after += q_liquid[]*sq(Delta);
    upper = max(upper, q_liquid[] - cs[]);
  }
  if (i%10 == 0)
    fprintf (stderr, "Q_INCLINED_AMR %.12g %d %.17g %.17g %.12g\n",
             t, i, mass_before, mass_after, upper);
#endif
  prescribed_flux();
}

event diagnostics (i++)
{
  double mass = 0., lower = 0., upper = 0.;
  foreach (reduction(+:mass) reduction(max:lower) reduction(max:upper)) {
    mass += q_liquid[]*sq(Delta);
    lower = max(lower, -q_liquid[]);
    upper = max(upper, q_liquid[] - cs[]);
  }
  if (i%10 == 0 || q_embed_diagnostics.maximum_open_flux_mismatch > 1e-14)
    fprintf (stderr,
             "Q_INCLINED %.12g %d %.17g %.12g %.12g %.12g %.12g %.12g %ld\n",
             t, i, mass, lower, upper,
             q_embed_diagnostics.maximum_bound_violation,
             q_embed_diagnostics.maximum_open_flux_mismatch,
             q_embed_diagnostics.clipped_aperture,
             q_embed_diagnostics.corrected_cells);
}

event stop (t = END_TIME)
{
  scalar numerical_q[];
  foreach()
    numerical_q[] = q_liquid[];
  scalar exact[];
  fraction (exact, exact_liquid_level(x, y, t));
  boundary ({exact});
  q_embed_initialize_geometric_fraction (exact);
  double mass = 0., exact_mass = 0., l1 = 0.;
  foreach (reduction(+:mass) reduction(+:exact_mass) reduction(+:l1)) {
    mass += numerical_q[]*sq(Delta);
    exact_mass += q_liquid[]*sq(Delta);
    l1 += fabs(numerical_q[] - q_liquid[])*sq(Delta);
  }
  fprintf (stderr, "Q_INCLINED_FINAL %d %d %.17g %.17g %.12g %.12g\n",
           LEVEL, USE_AMR, mass, exact_mass,
           (mass - exact_mass)/max(exact_mass, 1e-30),
           l1/max(exact_mass, 1e-30));
  return 1;
}
