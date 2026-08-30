/**
Manufactured translation test for the conservative liquid aperture q=cs*f.

A thin periodic liquid layer is translated parallel to a static embedded
wall.  The exact interface is known at every time.  The test exercises the
same double-PLIC swept-aperture operator and q-aware AMR callbacks used by the
receiver without the large BIE handoff header.
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
# define END_TIME 0.5
#endif
#ifndef USE_AMR
# define USE_AMR 1
#endif
#ifndef ADAPT_EVERY_STEP
# define ADAPT_EVERY_STEP 1
#endif

static const double translation_speed = 0.20;
static const double wall_height = 0.15;
double manufactured_dt;

static inline double exact_liquid_level
  (double horizontal, double vertical, double time)
{
  double phase = 2.*pi*(horizontal - translation_speed*time);
  double surface = 0.225 + 0.075*sin(phase);
  return surface - vertical;
}

int main()
{
  size (1.);
  origin (0., 0.);
  periodic (right);
  manufactured_dt = 0.20*L0/(translation_speed*(1 << LEVEL));
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
  solid (cs, fs, y - wall_height);
  fraction (f, exact_liquid_level(x, y, 0.));
  boundary ({f, cs});
  q_embed_initialize_geometric_fraction (f);
#if TREE
  q_embed_configure_fraction_amr (f);
#endif
  interfaces = NULL;
  foreach_face(x)
    uf.x[] = fs.x[]*translation_speed;
  foreach_face(y)
    uf.y[] = 0.;
  boundary ((scalar *){uf});
}

event manufactured_stability (i++, last)
{
  dt = dtnext(manufactured_dt);
}

event q_transport (i++)
{
  dt = dtnext(manufactured_dt);
  foreach_face(x)
    uf.x[] = fs.x[]*translation_speed;
  foreach_face(y)
    uf.y[] = 0.;
  boundary ((scalar *){uf});
  q_embed_vof_advection (f, i);
}

event q_adapt (i++)
{
#if USE_AMR
  if (i > 0 && !ADAPT_EVERY_STEP)
    return 0;
  double mass_before = 0.;
  foreach (reduction(+:mass_before))
    mass_before += q_liquid[]*sq(Delta);
  adapt_wavelet ({cs, q_liquid},
                 (double[]){1e-6, 2e-4}, LEVEL, MINLEVEL);
  refine (level < LEVEL &&
          ((f[] > 1e-8 && f[] < 1. - 1e-8) ||
           (f[1] > 1e-8 && f[1] < 1. - 1e-8) ||
           (f[-1] > 1e-8 && f[-1] < 1. - 1e-8) ||
           (f[0,1] > 1e-8 && f[0,1] < 1. - 1e-8) ||
           (f[0,-1] > 1e-8 && f[0,-1] < 1. - 1e-8)));
  foreach()
    f[] = cs[] > 0. ? clamp(q_liquid[]/cs[], 0., 1.) : 0.;
  boundary ({f});
  double mass_after_adapt = 0.;
  foreach (reduction(+:mass_after_adapt))
    mass_after_adapt += q_liquid[]*sq(Delta);
  scalar q_before_cleanup[];
  foreach()
    q_before_cleanup[] = q_liquid[];
  fractions_cleanup (cs, fs);
  double cleanup_clipped = 0.;
  foreach (reduction(+:cleanup_clipped)) {
    double restored = clamp(q_before_cleanup[], 0., cs[]);
    cleanup_clipped += fabs(restored - q_before_cleanup[])*sq(Delta);
    f[] = cs[] > 0. ? restored/cs[] : 0.;
    q_liquid[] = restored;
  }
  boundary ({f, q_liquid});
  double mass_after_cleanup = 0.;
  foreach (reduction(+:mass_after_cleanup))
    mass_after_cleanup += q_liquid[]*sq(Delta);
  if (i%10 == 0 || Q_EMBED_DEBUG)
    fprintf (stderr, "Q_AMR %.12g %d %.17g %.17g %.17g %.12g %ld %ld\n",
             t, i, mass_before, mass_after_adapt, mass_after_cleanup,
             cleanup_clipped, q_embed_restriction_calls,
             q_embed_refine_calls);
#endif
  foreach_face(x)
    uf.x[] = fs.x[]*translation_speed;
  foreach_face(y)
    uf.y[] = 0.;
  boundary ((scalar *){uf});
}

event diagnostics (i += 10; i <= 100000)
{
  double mass = 0., lower = 0., upper = 0.;
  foreach (reduction(+:mass) reduction(max:lower) reduction(max:upper)) {
    double q = q_liquid[];
    mass += q*sq(Delta);
    lower = max(lower, -q);
    upper = max(upper, q - cs[]);
  }
  fprintf (stderr, "Q_MANUFACTURED %.12g %d %.17g %.12g %.12g %.12g %.12g %.12g %ld\n",
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
    double q = numerical_q[], q_exact = q_liquid[];
    mass += q*sq(Delta);
    exact_mass += q_exact*sq(Delta);
    l1 += fabs(q - q_exact)*sq(Delta);
  }
  fprintf (stderr, "Q_FINAL %d %d %.17g %.17g %.12g %.12g\n",
           LEVEL, USE_AMR, mass, exact_mass,
           (mass - exact_mass)/max(exact_mass, 1e-30),
           l1/max(exact_mass, 1e-30));
  return 1;
}
