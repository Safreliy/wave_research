/**
Conservative two-interface VOF transport for a static embedded boundary.

The transported quantity is the full-cell liquid aperture

  q = |K \cap Omega_f \cap Omega_l|/|K| = cs*f,

where f is the liquid fraction conditional on the open (cs) part of the
cell.  Standard Basilisk embedded VOF deliberately ignores the solid part of
mixed cells.  Here every directional flux is instead the intersection of a
swept open slab, the solid PLIC and the liquid PLIC.  The same swept aperture
is used for the phase-momentum tracers.

This header is intentionally two-dimensional and assumes a static embedded
boundary.  It is a research implementation: bound repairs and geometric flux
defects are reported rather than hidden.
*/

#if dimension != 2
# error "q_embed_vof.h currently implements the 2-D operator only"
#endif

#ifndef Q_EMBED_GEOMETRIC_TOLERANCE
# define Q_EMBED_GEOMETRIC_TOLERANCE 1e-12
#endif
#ifndef Q_EMBED_BOUND_TOLERANCE
# define Q_EMBED_BOUND_TOLERANCE 1e-10
#endif
#ifndef Q_EMBED_DEBUG
# define Q_EMBED_DEBUG 0
#endif
#ifndef Q_EMBED_CUTCELL_CFL
# define Q_EMBED_CUTCELL_CFL 0.45
#endif
#ifndef Q_EMBED_CONSERVATIVE_REDISTRIBUTION
# define Q_EMBED_CONSERVATIVE_REDISTRIBUTION 0
#endif
#ifndef Q_EMBED_REDISTRIBUTION_RADIUS
# define Q_EMBED_REDISTRIBUTION_RADIUS 2
#endif

typedef struct {
  double x, y;
} QEmbedVertex;

typedef struct {
  int n;
  QEmbedVertex v[24];
} QEmbedPolygon;

typedef struct {
  double clipped_aperture;
  double maximum_bound_violation;
  double maximum_open_flux_mismatch;
  double locally_redistributed_volume;
  double nonlocally_redistributed_volume;
  long corrected_cells;
} QEmbedDiagnostics;

typedef struct {
  double candidate_timestep;
  double limited_timestep;
  double maximum_cutcell_courant;
  long active_cut_cells;
  long limited_cells;
} QEmbedStabilityDiagnostics;

QEmbedDiagnostics q_embed_diagnostics = {0};
QEmbedStabilityDiagnostics q_embed_stability_diagnostics = {0};
scalar q_liquid[];

/**
Return a timestep which also satisfies the embedded small-cell restriction in
the one-cell band containing the liquid interface.

The ordinary VOF CFL is based on a full cell.  A cut cell instead has the
capacity cs*Delta^dimension, while uf already contains the open-face metric.
Without this additional bound, the requested swept open volume can exceed the
donor capacity.  Clamping that overflow is bounded but not conservative, and
phase tracers can be amplified by the subsequent division by cs.

Pure cells are deliberately excluded unless they touch the other phase.  The
geometric VOF pure-state shortcut transports the entire open flux in a
uniform-phase region, for which the discrete divergence cancellation is exact.
The one-cell halo is sufficient under the ordinary Cartesian VOF CFL because
an interface cannot cross more than one cell in a step.  This avoids letting
arbitrarily small, phase-uniform wall cells throttle the whole calculation.
*/
static inline double q_embed_stable_timestep
  (scalar fraction, face vector open_flux, double candidate)
{
  double limited = candidate, maximum_courant = 0.;
  long active_cut_cells = 0, limited_cells = 0;
  foreach (reduction(min:limited) reduction(max:maximum_courant)
           reduction(+:active_cut_cells) reduction(+:limited_cells))
    if (cs[] > Q_EMBED_GEOMETRIC_TOLERANCE && cs[] < 1.) {
      double phase = clamp(fraction[], 0., 1.);
      bool active = phase > Q_EMBED_GEOMETRIC_TOLERANCE &&
        phase < 1. - Q_EMBED_GEOMETRIC_TOLERANCE;
      if (!active && fs.x[] > Q_EMBED_GEOMETRIC_TOLERANCE)
        active = fabs(phase - clamp(fraction[-1], 0., 1.)) >
          Q_EMBED_GEOMETRIC_TOLERANCE;
      if (!active && fs.x[1] > Q_EMBED_GEOMETRIC_TOLERANCE)
        active = fabs(phase - clamp(fraction[1], 0., 1.)) >
          Q_EMBED_GEOMETRIC_TOLERANCE;
      if (!active && fs.y[] > Q_EMBED_GEOMETRIC_TOLERANCE)
        active = fabs(phase - clamp(fraction[0,-1], 0., 1.)) >
          Q_EMBED_GEOMETRIC_TOLERANCE;
      if (!active && fs.y[0,1] > Q_EMBED_GEOMETRIC_TOLERANCE)
        active = fabs(phase - clamp(fraction[0,1], 0., 1.)) >
          Q_EMBED_GEOMETRIC_TOLERANCE;
      if (active) {
        active_cut_cells++;
        double face_maximum = 0.;
        foreach_dimension()
          face_maximum = max(face_maximum,
                             max(fabs(open_flux.x[]),
                                 fabs(open_flux.x[1])));
        if (face_maximum > 0.) {
          double local = Q_EMBED_CUTCELL_CFL*Delta*cs[]/face_maximum;
          maximum_courant = max(maximum_courant,
                                candidate*face_maximum/(Delta*cs[]));
          if (local < candidate)
            limited_cells++;
          limited = min(limited, local);
        }
      }
    }
  q_embed_stability_diagnostics = (QEmbedStabilityDiagnostics) {
    candidate, limited, maximum_courant, active_cut_cells, limited_cells
  };
#if Q_EMBED_DEBUG
  fprintf (stderr, "Q_STABILITY %.12g %.12g %.12g %.12g %ld %ld\n",
           t, candidate, limited, maximum_courant,
           active_cut_cells, limited_cells);
#endif
  return limited;
}

static inline QEmbedPolygon q_embed_square (void)
{
  QEmbedPolygon polygon = {4, {{-.5,-.5}, {.5,-.5}, {.5,.5}, {-.5,.5}}};
  return polygon;
}

static inline double q_embed_signed_distance
  (QEmbedVertex vertex, double nx, double ny, double alpha)
{
  return nx*vertex.x + ny*vertex.y - alpha;
}

static QEmbedPolygon q_embed_clip
  (QEmbedPolygon input, double nx, double ny, double alpha)
{
  QEmbedPolygon output = {0};
  if (!input.n)
    return output;
  QEmbedVertex previous = input.v[input.n - 1];
  double previous_distance =
    q_embed_signed_distance (previous, nx, ny, alpha);
  bool previous_inside = previous_distance <= Q_EMBED_GEOMETRIC_TOLERANCE;
  for (int j = 0; j < input.n; j++) {
    QEmbedVertex current = input.v[j];
    double current_distance =
      q_embed_signed_distance (current, nx, ny, alpha);
    bool current_inside = current_distance <= Q_EMBED_GEOMETRIC_TOLERANCE;
    if (current_inside != previous_inside) {
      double denominator = previous_distance - current_distance;
      double weight = fabs(denominator) > 1e-30 ?
        previous_distance/denominator : 0.;
      assert (output.n < 24);
      output.v[output.n++] = (QEmbedVertex) {
        previous.x + weight*(current.x - previous.x),
        previous.y + weight*(current.y - previous.y)
      };
    }
    if (current_inside) {
      assert (output.n < 24);
      output.v[output.n++] = current;
    }
    previous = current;
    previous_distance = current_distance;
    previous_inside = current_inside;
  }
  return output;
}

static inline double q_embed_area (QEmbedPolygon polygon)
{
  if (polygon.n < 3)
    return 0.;
  double twice_area = 0.;
  for (int j = 0; j < polygon.n; j++) {
    QEmbedVertex a = polygon.v[j];
    QEmbedVertex b = polygon.v[(j + 1)%polygon.n];
    twice_area += a.x*b.y - a.y*b.x;
  }
  return fabs(twice_area)/2.;
}

static QEmbedPolygon q_embed_open_polygon
  (double cs_value, double nsx, double nsy, double alpha_s)
{
  QEmbedPolygon polygon = q_embed_square();
  if (cs_value <= 0.)
    polygon.n = 0;
  else if (cs_value < 1.)
    polygon = q_embed_clip (polygon, nsx, nsy, alpha_s);
  return polygon;
}

static double q_embed_intersection_area
  (double cs_value, double nsx, double nsy, double alpha_s,
   double nfx, double nfy, double alpha_f)
{
  QEmbedPolygon polygon =
    q_embed_open_polygon (cs_value, nsx, nsy, alpha_s);
  polygon = q_embed_clip (polygon, nfx, nfy, alpha_f);
  return q_embed_area (polygon);
}

static double q_embed_liquid_alpha
  (double target_q, double cs_value,
   double nsx, double nsy, double alpha_s,
   double nfx, double nfy)
{
  double norm = fabs(nfx) + fabs(nfy);
  if (norm <= 1e-30)
    return target_q >= .5*cs_value ? norm : -norm;
  double lower = -.5*norm - 2.*Q_EMBED_GEOMETRIC_TOLERANCE;
  double upper =  .5*norm + 2.*Q_EMBED_GEOMETRIC_TOLERANCE;
  for (int iteration = 0; iteration < 64; iteration++) {
    double middle = (lower + upper)/2.;
    double area = q_embed_intersection_area
      (cs_value, nsx, nsy, alpha_s, nfx, nfy, middle);
    if (area < target_q)
      lower = middle;
    else
      upper = middle;
  }
  return (lower + upper)/2.;
}

static void q_embed_reconstruction
  (scalar f, vector nf, scalar alpha_f, vector ns, scalar alpha_s)
{
  foreach() {
    if (cs[] > 0. && cs[] < 1.) {
      coord solid_normal = facet_normal (point, cs, fs);
      ns.x[] = solid_normal.x;
      ns.y[] = solid_normal.y;
      alpha_s[] = plane_alpha (cs[], solid_normal);
    }
    else {
      ns.x[] = ns.y[] = 0.;
      alpha_s[] = 0.;
    }
    double q = clamp(f[]*cs[], 0., cs[]);
    if (q <= 0.) {
      nf.x[] = nf.y[] = 0.;
      alpha_f[] = -1.;
    }
    else if (q >= cs[]) {
      nf.x[] = nf.y[] = 0.;
      alpha_f[] = 1.;
    }
    else {
      coord liquid_normal = interface_normal (point, f);
      double normal_norm = fabs(liquid_normal.x) + fabs(liquid_normal.y);
      if (normal_norm <= 1e-30) {
        liquid_normal.x = 0.;
        liquid_normal.y = 1.;
      }
      nf.x[] = liquid_normal.x;
      nf.y[] = liquid_normal.y;
      alpha_f[] = q_embed_liquid_alpha
        (q, cs[], ns.x[], ns.y[], alpha_s[], nf.x[], nf.y[]);
    }
  }
  boundary ((scalar *){nf, alpha_f, ns, alpha_s});
}

static QEmbedPolygon q_embed_slab
  (QEmbedPolygon polygon, int direction, int sign_velocity, double width)
{
  if (direction == 0) {
    if (sign_velocity > 0)
      polygon = q_embed_clip (polygon, -1., 0., width - .5);
    else
      polygon = q_embed_clip (polygon, 1., 0., width - .5);
  }
  else {
    if (sign_velocity > 0)
      polygon = q_embed_clip (polygon, 0., -1., width - .5);
    else
      polygon = q_embed_clip (polygon, 0., 1., width - .5);
  }
  return polygon;
}

typedef struct {
  double open_area;
  double liquid_area;
} QEmbedCellSweep;

typedef struct {
  QEmbedCellSweep donor;
  double open_mismatch;
} QEmbedSweptTransfer;

/** Intersect one cell with the part of its open polygon adjacent to the
    downstream face.  The requested measure, rather than a Cartesian width,
    is used so that the solid aperture and the phase flux stay consistent. */
static QEmbedCellSweep q_embed_cell_sweep
  (double requested_open_area, int direction, int sign_velocity,
   double cs_value, double nsx, double nsy, double alpha_s,
   double nfx, double nfy, double alpha_f)
{
  QEmbedCellSweep result = {0};
  if (requested_open_area <= 0. || cs_value <= 0.)
    return result;
  QEmbedPolygon open_polygon =
    q_embed_open_polygon (cs_value, nsx, nsy, alpha_s);
  double open_area = q_embed_area (open_polygon);
  double target = min(requested_open_area, open_area);
  double lower = 0., upper = 1.;
  for (int iteration = 0; iteration < 56; iteration++) {
    double middle = (lower + upper)/2.;
    double area = q_embed_area
      (q_embed_slab(open_polygon, direction, sign_velocity, middle));
    if (area < target)
      lower = middle;
    else
      upper = middle;
  }
  QEmbedPolygon swept = q_embed_slab
    (open_polygon, direction, sign_velocity, (lower + upper)/2.);
  result.open_area = q_embed_area (swept);
  swept = q_embed_clip (swept, nfx, nfy, alpha_f);
  result.liquid_area = min(q_embed_area(swept), result.open_area);
  return result;
}

static QEmbedSweptTransfer q_embed_swept_transfer
  (double target_open_area, int direction, int sign_velocity,
   double donor_cs, double donor_nsx, double donor_nsy,
   double donor_alpha_s, double donor_nfx, double donor_nfy,
   double donor_alpha_f)
{
  QEmbedSweptTransfer transfer = {0};
  transfer.donor = q_embed_cell_sweep
    (target_open_area, direction, sign_velocity,
     donor_cs, donor_nsx, donor_nsy, donor_alpha_s,
     donor_nfx, donor_nfy, donor_alpha_f);
  transfer.open_mismatch = max
    (target_open_area - transfer.donor.open_area, 0.);
  return transfer;
}

/**
Redistribute only the bound correction produced by one directional sweep.

The clipped state is used as the feasible starting point.  Positive residual
is placed into local open-cell room and negative residual is removed from local
liquid.  The support is the face-connected neighborhood of the offending cut
cells.  A domain-wide fallback is retained and diagnosed rather than silently
losing mass when that neighborhood has insufficient capacity.

This is a conservative state-redistribution pilot, not a replacement for the
phase-tracer reconciliation required when a material small-cell correction is
large.  In the coupled receiver regression the correction remains at roundoff
scale; the inclined q-only test is the intended finite-correction validation.
*/
#if Q_EMBED_CONSERVATIVE_REDISTRIBUTION
static void q_embed_redistribute_bound_residual (scalar residual)
{
  double positive_volume = 0., negative_volume = 0.;
  scalar positive_support[], negative_support[];
  scalar expanded_positive[], expanded_negative[];
  foreach (reduction(+:positive_volume) reduction(+:negative_volume)) {
    positive_volume += max(residual[], 0.)*sq(Delta);
    negative_volume += max(-residual[], 0.)*sq(Delta);
    positive_support[] = residual[] > Q_EMBED_BOUND_TOLERANCE ? 1. : 0.;
    negative_support[] = residual[] < -Q_EMBED_BOUND_TOLERANCE ? 1. : 0.;
  }
  if (positive_volume <= Q_EMBED_GEOMETRIC_TOLERANCE &&
      negative_volume <= Q_EMBED_GEOMETRIC_TOLERANCE)
    return;
  boundary ({positive_support, negative_support});
  for (int pass = 0; pass < Q_EMBED_REDISTRIBUTION_RADIUS; pass++) {
    foreach() {
      double positive = positive_support[], negative = negative_support[];
      if (fs.x[] > Q_EMBED_GEOMETRIC_TOLERANCE) {
        positive = max(positive, positive_support[-1]);
        negative = max(negative, negative_support[-1]);
      }
      if (fs.x[1] > Q_EMBED_GEOMETRIC_TOLERANCE) {
        positive = max(positive, positive_support[1]);
        negative = max(negative, negative_support[1]);
      }
      if (fs.y[] > Q_EMBED_GEOMETRIC_TOLERANCE) {
        positive = max(positive, positive_support[0,-1]);
        negative = max(negative, negative_support[0,-1]);
      }
      if (fs.y[0,1] > Q_EMBED_GEOMETRIC_TOLERANCE) {
        positive = max(positive, positive_support[0,1]);
        negative = max(negative, negative_support[0,1]);
      }
      expanded_positive[] = positive;
      expanded_negative[] = negative;
    }
    boundary ({expanded_positive, expanded_negative});
    foreach() {
      positive_support[] = expanded_positive[];
      negative_support[] = expanded_negative[];
    }
    boundary ({positive_support, negative_support});
  }

  double local_room = 0.;
  foreach (reduction(+:local_room))
    if (positive_support[] > 0.)
      local_room += max(cs[] - q_liquid[], 0.)*sq(Delta);
  double local_positive = min(positive_volume, local_room);
  double positive_fraction = local_room > 0. ? local_positive/local_room : 0.;
  foreach()
    if (positive_support[] > 0.)
      q_liquid[] += positive_fraction*max(cs[] - q_liquid[], 0.);

  double positive_remainder = positive_volume - local_positive;
  if (positive_remainder > Q_EMBED_GEOMETRIC_TOLERANCE) {
    double global_room = 0.;
    foreach (reduction(+:global_room))
      global_room += max(cs[] - q_liquid[], 0.)*sq(Delta);
    double global_fraction = global_room > 0. ?
      min(positive_remainder/global_room, 1.) : 0.;
    foreach()
      q_liquid[] += global_fraction*max(cs[] - q_liquid[], 0.);
    q_embed_diagnostics.nonlocally_redistributed_volume +=
      global_fraction*global_room;
  }

  double local_liquid = 0.;
  foreach (reduction(+:local_liquid))
    if (negative_support[] > 0.)
      local_liquid += max(q_liquid[], 0.)*sq(Delta);
  double local_negative = min(negative_volume, local_liquid);
  double negative_fraction = local_liquid > 0. ?
    local_negative/local_liquid : 0.;
  foreach()
    if (negative_support[] > 0.)
      q_liquid[] -= negative_fraction*max(q_liquid[], 0.);

  double negative_remainder = negative_volume - local_negative;
  if (negative_remainder > Q_EMBED_GEOMETRIC_TOLERANCE) {
    double global_liquid = 0.;
    foreach (reduction(+:global_liquid))
      global_liquid += max(q_liquid[], 0.)*sq(Delta);
    double global_fraction = global_liquid > 0. ?
      min(negative_remainder/global_liquid, 1.) : 0.;
    foreach()
      q_liquid[] -= global_fraction*max(q_liquid[], 0.);
    q_embed_diagnostics.nonlocally_redistributed_volume +=
      global_fraction*global_liquid;
  }
  q_embed_diagnostics.locally_redistributed_volume +=
    local_positive + local_negative;
  boundary ({q_liquid});
}
#endif

static void q_embed_update_cell
  (scalar f, scalar cc, face vector liquid_swept, face vector open_swept,
   scalar * tracers, scalar * tracer_swept, scalar * tracer_compression,
   int x_direction)
{
#if Q_EMBED_CONSERVATIVE_REDISTRIBUTION
  scalar bound_residual[];
#endif
  double maximum_bound_violation = 0., clipped_aperture = 0.;
  double mass_before = 0., mass_after = 0.;
  long corrected_cells = 0;
  foreach (reduction(max:maximum_bound_violation)
           reduction(+:clipped_aperture)
           reduction(+:mass_before) reduction(+:mass_after)
           reduction(+:corrected_cells)) {
    double old_q = clamp(q_liquid[], 0., cs[]);
    double liquid_left = x_direction ? liquid_swept.x[] : liquid_swept.y[];
    double liquid_right = x_direction ?
      liquid_swept.x[1] : liquid_swept.y[0,1];
    double open_left = x_direction ? open_swept.x[] : open_swept.y[];
    double open_right = x_direction ? open_swept.x[1] : open_swept.y[0,1];
    double directional_scale = dt/Delta;
    double raw_q = old_q + directional_scale*
      (liquid_left - liquid_right + cc[]*(open_right - open_left));
    double violation = max(-raw_q, raw_q - cs[]);
#if Q_EMBED_DEBUG
    static int debug_count = 0;
    if (violation > 1e-6 && debug_count < 24) {
      fprintf (stderr,
               "Q_DEBUG %d %.12g %.12g %.12g %.12g %.12g %.12g %.12g %.12g %.12g\n",
               x_direction, x, y, old_q, cs[], raw_q,
               liquid_left, liquid_right, open_left, open_right);
      debug_count++;
    }
#endif
    maximum_bound_violation = max(maximum_bound_violation, violation);
    double new_q = clamp(raw_q, 0., cs[]);
#if Q_EMBED_CONSERVATIVE_REDISTRIBUTION
    bound_residual[] = raw_q - new_q;
#endif
    mass_before += old_q*sq(Delta);
    mass_after += new_q*sq(Delta);
    if (fabs(new_q - raw_q) > 0.) {
      clipped_aperture += fabs(new_q - raw_q)*sq(Delta);
      corrected_cells++;
    }
    scalar tracer, swept, compression;
    for (tracer, swept, compression in
         tracers, tracer_swept, tracer_compression) {
      double tracer_left = swept[];
      double tracer_right = x_direction ? swept[1] : swept[0,1];
      double physical = cs[]*tracer[] + directional_scale*
        (tracer_left - tracer_right +
         compression[]*(open_right - open_left));
#if Q_EMBED_DEBUG
      static int tracer_debug_count = 0;
      if ((!isfinite(physical) || fabs(physical) > 1e6 ||
           fabs(tracer_left) > 1e6 || fabs(tracer_right) > 1e6) &&
          tracer_debug_count < 32) {
        fprintf (stderr,
                 "Q_TRACER_DEBUG %.12g %d %.12g %.12g %.12g %.12g %.12g %.12g %.12g %.12g %.12g %.12g\n",
                 t, x_direction, x, y, cs[], tracer[], physical,
                 tracer_left, tracer_right, compression[],
                 open_left, open_right);
        tracer_debug_count++;
      }
#endif
      tracer[] = cs[] > 0. ? physical/cs[] : 0.;
    }
    f[] = cs[] > 0. ? new_q/cs[] : 0.;
    q_liquid[] = new_q;
  }
#if Q_EMBED_CONSERVATIVE_REDISTRIBUTION
  boundary ({bound_residual, q_liquid});
  q_embed_redistribute_bound_residual (bound_residual);
#else
  boundary ({q_liquid});
#endif
  mass_after = 0.;
  foreach (reduction(+:mass_after)) {
    q_liquid[] = clamp(q_liquid[], 0., cs[]);
    f[] = cs[] > 0. ? q_liquid[]/cs[] : 0.;
    mass_after += q_liquid[]*sq(Delta);
  }
  boundary ({f, q_liquid});
  boundary (tracers);
  q_embed_diagnostics.maximum_bound_violation = max
    (q_embed_diagnostics.maximum_bound_violation, maximum_bound_violation);
  q_embed_diagnostics.clipped_aperture += clipped_aperture;
  q_embed_diagnostics.corrected_cells += corrected_cells;
#if Q_EMBED_DEBUG
  fprintf (stderr, "Q_SWEEP %d %.12g %.12g %.12g %.12g %.12g\n",
           x_direction, t, mass_before, mass_after,
           mass_after - mass_before, clipped_aperture);
#endif
}

/**
VOF-concentration slopes are only second-order corrections.  The standard
three-point helper does not know about the embedded mask and can pass
Basilisk's nodata=HUGE from a solid neighbor into the limiter.  Near such a
stencil we deliberately fall back to first-order upwinding.
*/
foreach_dimension()
static inline double q_embed_safe_concentration_gradient_x
  (Point point, scalar f, scalar tracer)
{
  double cs_center = cs[], cs_left = cs[-1], cs_right = cs[1];
  double f_center = f[], f_left = f[-1], f_right = f[1];
  double tracer_center = tracer[];
  double tracer_left = tracer[-1], tracer_right = tracer[1];
  if (!isfinite(cs_center) || !isfinite(cs_left) || !isfinite(cs_right) ||
      cs_center <= Q_EMBED_GEOMETRIC_TOLERANCE ||
      cs_left <= Q_EMBED_GEOMETRIC_TOLERANCE ||
      cs_right <= Q_EMBED_GEOMETRIC_TOLERANCE ||
      cs_center > 1. + Q_EMBED_BOUND_TOLERANCE ||
      cs_left > 1. + Q_EMBED_BOUND_TOLERANCE ||
      cs_right > 1. + Q_EMBED_BOUND_TOLERANCE ||
      !isfinite(f_center) || !isfinite(f_left) || !isfinite(f_right) ||
      !isfinite(tracer_center) || !isfinite(tracer_left) ||
      !isfinite(tracer_right) ||
      fabs(tracer_center) >= HUGE/2. ||
      fabs(tracer_left) >= HUGE/2. || fabs(tracer_right) >= HUGE/2.)
    return 0.;
  double gradient = vof_concentration_gradient_x (point, f, tracer);
  return isfinite(gradient) && fabs(gradient) < HUGE/2. ? gradient : 0.;
}

static void q_embed_sweep_x
  (scalar f, scalar cc, scalar * tracers, scalar * tracer_compression)
{
  vector nf[], ns[];
  scalar alpha_f[], alpha_s[];
  face vector liquid_swept[], open_swept[];
  scalar * gradients = NULL, * tracer_swept = NULL;
  for (scalar tracer in tracers) {
    scalar gradient = new scalar, swept = new scalar;
    gradients = list_append (gradients, gradient);
    tracer_swept = list_append (tracer_swept, swept);
  }
  foreach() {
    scalar tracer, gradient;
    for (tracer, gradient in tracers, gradients)
      gradient[] = q_embed_safe_concentration_gradient_x (point, f, tracer);
  }
  q_embed_reconstruction (f, nf, alpha_f, ns, alpha_s);
  double maximum_open_flux_mismatch = 0.;
  foreach_face(x, reduction(max:maximum_open_flux_mismatch)) {
    double open_transfer = uf.x[]*dt/Delta;
    int velocity_sign = sign(open_transfer);
    int donor = -(velocity_sign + 1.)/2.;
    double requested_open_area = fabs(open_transfer);
    QEmbedSweptTransfer transfer = {0};
    /* Match the standard geometric-VOF pure-cell shortcut.  A conditional
       phase fraction of one transports the entire open face flux even when a
       triangular donor cut-cell is smaller than the swept volume. */
    if (f[donor] <= Q_EMBED_GEOMETRIC_TOLERANCE)
      transfer.donor.open_area = requested_open_area;
    else if (f[donor] >= 1. - Q_EMBED_GEOMETRIC_TOLERANCE)
      transfer.donor.open_area = transfer.donor.liquid_area =
        requested_open_area;
    else
      transfer = q_embed_swept_transfer
        (requested_open_area, 0, velocity_sign,
         cs[donor], ns.x[donor], ns.y[donor], alpha_s[donor],
         nf.x[donor], nf.y[donor], alpha_f[donor]);
    double liquid_area = transfer.donor.liquid_area;
    liquid_swept.x[] = dt > 0. ?
      velocity_sign*liquid_area*Delta/dt : 0.;
    open_swept.x[] = uf.x[];
    maximum_open_flux_mismatch = max
      (maximum_open_flux_mismatch, transfer.open_mismatch);
    double un = fabs(open_transfer)/(fm.x[] + SEPS);
    scalar tracer, gradient, swept;
    for (tracer, gradient, swept in tracers, gradients, tracer_swept) {
      double donor_fraction = tracer.inverse ? 1. - f[donor] : f[donor];
      double donor_area = tracer.inverse ?
        max(transfer.donor.open_area - transfer.donor.liquid_area, 0.) :
        transfer.donor.liquid_area;
      double tracer_integral = 0.;
      if (donor_fraction > 1e-10 && isfinite(tracer[donor]) &&
          fabs(tracer[donor]) < HUGE/2.) {
        double concentration = tracer[donor]/donor_fraction;
        concentration += velocity_sign*min(1., 1. - un)*
          gradient[donor]*Delta/2.;
        tracer_integral += concentration*donor_area;
      }
      swept[] = dt > 0. ?
        velocity_sign*tracer_integral*Delta/dt : 0.;
#if Q_EMBED_DEBUG
      static int swept_x_debug_count = 0;
      if ((!isfinite(swept[]) || fabs(swept[]) > 1e6) &&
          swept_x_debug_count < 24) {
        double phase_area = donor_area;
        double concentration = phase_area > 0. ?
          tracer_integral/phase_area : 0.;
        fprintf (stderr,
                 "Q_SWEPT_X_DEBUG %.12g %.12g %.12g %d %.12g %.12g %.12g %.12g %.12g %.12g %.12g\n",
                 t, x, y, donor, donor_fraction, tracer[donor],
                 gradient[donor], concentration, phase_area, un, swept[]);
        swept_x_debug_count++;
      }
#endif
    }
  }
  q_embed_diagnostics.maximum_open_flux_mismatch = max
    (q_embed_diagnostics.maximum_open_flux_mismatch,
     maximum_open_flux_mismatch);
  q_embed_update_cell (f, cc, liquid_swept, open_swept, tracers,
                       tracer_swept, tracer_compression, true);
  delete (gradients); free (gradients);
  delete (tracer_swept); free (tracer_swept);
}

static void q_embed_sweep_y
  (scalar f, scalar cc, scalar * tracers, scalar * tracer_compression)
{
  vector nf[], ns[];
  scalar alpha_f[], alpha_s[];
  face vector liquid_swept[], open_swept[];
  scalar * gradients = NULL, * tracer_swept = NULL;
  for (scalar tracer in tracers) {
    scalar gradient = new scalar, swept = new scalar;
    gradients = list_append (gradients, gradient);
    tracer_swept = list_append (tracer_swept, swept);
  }
  foreach() {
    scalar tracer, gradient;
    for (tracer, gradient in tracers, gradients)
      gradient[] = q_embed_safe_concentration_gradient_y (point, f, tracer);
  }
  q_embed_reconstruction (f, nf, alpha_f, ns, alpha_s);
  double maximum_open_flux_mismatch = 0.;
  foreach_face(y, reduction(max:maximum_open_flux_mismatch)) {
    double open_transfer = uf.y[]*dt/Delta;
    int velocity_sign = sign(open_transfer);
    int donor = -(velocity_sign + 1.)/2.;
    double requested_open_area = fabs(open_transfer);
    QEmbedSweptTransfer transfer = {0};
    if (f[0,donor] <= Q_EMBED_GEOMETRIC_TOLERANCE)
      transfer.donor.open_area = requested_open_area;
    else if (f[0,donor] >= 1. - Q_EMBED_GEOMETRIC_TOLERANCE)
      transfer.donor.open_area = transfer.donor.liquid_area =
        requested_open_area;
    else
      transfer = q_embed_swept_transfer
        (requested_open_area, 1, velocity_sign,
         cs[0,donor], ns.x[0,donor], ns.y[0,donor], alpha_s[0,donor],
         nf.x[0,donor], nf.y[0,donor], alpha_f[0,donor]);
    double liquid_area = transfer.donor.liquid_area;
    liquid_swept.y[] = dt > 0. ?
      velocity_sign*liquid_area*Delta/dt : 0.;
    open_swept.y[] = uf.y[];
    maximum_open_flux_mismatch = max
      (maximum_open_flux_mismatch, transfer.open_mismatch);
    double un = fabs(open_transfer)/(fm.y[] + SEPS);
    scalar tracer, gradient, swept;
    for (tracer, gradient, swept in tracers, gradients, tracer_swept) {
      double donor_fraction =
        tracer.inverse ? 1. - f[0,donor] : f[0,donor];
      double donor_area = tracer.inverse ?
        max(transfer.donor.open_area - transfer.donor.liquid_area, 0.) :
        transfer.donor.liquid_area;
      double tracer_integral = 0.;
      if (donor_fraction > 1e-10 && isfinite(tracer[0,donor]) &&
          fabs(tracer[0,donor]) < HUGE/2.) {
        double concentration = tracer[0,donor]/donor_fraction;
        concentration += velocity_sign*min(1., 1. - un)*
          gradient[0,donor]*Delta/2.;
        tracer_integral += concentration*donor_area;
      }
      swept[] = dt > 0. ?
        velocity_sign*tracer_integral*Delta/dt : 0.;
#if Q_EMBED_DEBUG
      static int swept_y_debug_count = 0;
      if ((!isfinite(swept[]) || fabs(swept[]) > 1e6) &&
          swept_y_debug_count < 24) {
        double phase_area = donor_area;
        double concentration = phase_area > 0. ?
          tracer_integral/phase_area : 0.;
        fprintf (stderr,
                 "Q_SWEPT_Y_DEBUG %.12g %.12g %.12g %d %.12g %.12g %.12g %.12g %.12g %.12g %.12g\n",
                 t, x, y, donor, donor_fraction, tracer[0,donor],
                 gradient[0,donor], concentration, phase_area, un, swept[]);
        swept_y_debug_count++;
      }
#endif
    }
  }
  q_embed_diagnostics.maximum_open_flux_mismatch = max
    (q_embed_diagnostics.maximum_open_flux_mismatch,
     maximum_open_flux_mismatch);
  q_embed_update_cell (f, cc, liquid_swept, open_swept, tracers,
                       tracer_swept, tracer_compression, false);
  delete (gradients); free (gradients);
  delete (tracer_swept); free (tracer_swept);
}

static void q_embed_vof_advection (scalar f, int iteration)
{
  scalar cc[], * tracer_compression = NULL, * tracers = f.tracers;
  for (scalar tracer in tracers) {
    scalar compression = new scalar;
    tracer_compression = list_append (tracer_compression, compression);
  }
  foreach() {
    cc[] = f[] > .5;
    scalar tracer, compression;
    for (tracer, compression in tracers, tracer_compression) {
      if (tracer.inverse)
        compression[] = f[] < .5 ? tracer[]/(1. - f[] + SEPS) : 0.;
      else
        compression[] = f[] > .5 ? tracer[]/(f[] + SEPS) : 0.;
    }
  }
  q_embed_diagnostics = (QEmbedDiagnostics){0};
  if (iteration%2) {
    q_embed_sweep_y (f, cc, tracers, tracer_compression);
    q_embed_sweep_x (f, cc, tracers, tracer_compression);
  }
  else {
    q_embed_sweep_x (f, cc, tracers, tracer_compression);
    q_embed_sweep_y (f, cc, tracers, tracer_compression);
  }
  delete (tracer_compression); free (tracer_compression);
}

#if TREE
long q_embed_restriction_calls = 0, q_embed_refine_calls = 0;

static void q_embed_q_restriction (Point point, scalar q)
{
  q_embed_restriction_calls++;
  double sum = 0.;
  foreach_child()
    sum += q[];
  q[] = sum/(1 << dimension);
}

static void q_embed_q_refine (Point point, scalar q)
{
  q_embed_refine_calls++;
  double parent_cs = clamp(cs[], 0., 1.);
  double parent_q = clamp(q[], 0., parent_cs);
  coord solid_normal = {0., 0.};
  double alpha_s = 0.;
  if (parent_cs > 0. && parent_cs < 1.) {
    solid_normal = facet_normal (point, cs, fs);
    alpha_s = plane_alpha (parent_cs, solid_normal);
  }
  coord liquid_normal = {0., 0.};
  double alpha_f = parent_q <= 0. ? -1. : 1.;
  if (parent_q > 0. && parent_q < parent_cs) {
    liquid_normal = interface_normal (point, f);
    if (fabs(liquid_normal.x) + fabs(liquid_normal.y) <= 1e-30)
      liquid_normal.y = 1.;
    alpha_f = q_embed_liquid_alpha
      (parent_q, parent_cs, solid_normal.x, solid_normal.y, alpha_s,
       liquid_normal.x, liquid_normal.y);
  }
  double assigned = 0.;
  foreach_child() {
    QEmbedPolygon polygon = q_embed_open_polygon
      (parent_cs, solid_normal.x, solid_normal.y, alpha_s);
    polygon = q_embed_clip
      (polygon, liquid_normal.x, liquid_normal.y, alpha_f);
    if (child.x < 0)
      polygon = q_embed_clip (polygon, 1., 0., 0.);
    else
      polygon = q_embed_clip (polygon, -1., 0., 0.);
    if (child.y < 0)
      polygon = q_embed_clip (polygon, 0., 1., 0.);
    else
      polygon = q_embed_clip (polygon, 0., -1., 0.);
    q[] = 4.*q_embed_area(polygon);
    assigned += q[];
  }
  double target_sum = (1 << dimension)*parent_q;
  double scale = assigned > 0. ? target_sum/assigned : 0.;
  foreach_child()
    q[] = clamp(scale*q[], 0., 1.);
}

static void q_embed_fraction_restriction (Point point, scalar f)
{
  double physical_liquid = 0., open_volume = 0.;
  foreach_child() {
    physical_liquid += cs[]*f[];
    open_volume += cs[];
  }
  f[] = open_volume > 0. ? physical_liquid/open_volume : 0.;
}

static void q_embed_configure_fraction_amr (scalar f)
{
  /* q is the only transported liquid measure.  Refining conditional f
     independently creates a second, order-dependent geometry; callers
     reconstruct f=q/cs immediately after adaptation instead. */
  f.restriction = q_embed_fraction_restriction;
  q_liquid.refine = q_liquid.prolongation = q_embed_q_refine;
  q_liquid.restriction = q_embed_q_restriction;
}
#endif

/* Replace the independent product by the geometric intersection of the two
   PLIC reconstructions at initialization.  The incoming f must still be the
   ordinary full-cell liquid fraction when this routine is called. */
static void q_embed_initialize_geometric_fraction (scalar f)
{
  vector nf[];
  scalar alpha_f[];
  reconstruction (f, nf, alpha_f);
  double solid_reconstruction_l1 = 0., liquid_reconstruction_l1 = 0.;
  double product_intersection_l1 = 0.;
  long triple_cells = 0;
  foreach (reduction(+:solid_reconstruction_l1)
           reduction(+:liquid_reconstruction_l1)
           reduction(+:product_intersection_l1)
           reduction(+:triple_cells)) {
    if (cs[] <= 0.)
      f[] = 0.;
    else if (cs[] < 1. && f[] > 0. && f[] < 1.) {
      coord solid_normal = facet_normal (point, cs, fs);
      double alpha_s = plane_alpha (cs[], solid_normal);
      double reconstructed_solid = q_embed_area
        (q_embed_open_polygon(cs[], solid_normal.x, solid_normal.y, alpha_s));
      QEmbedPolygon liquid_polygon = q_embed_square();
      liquid_polygon = q_embed_clip
        (liquid_polygon, nf.x[], nf.y[], alpha_f[]);
      double reconstructed_liquid = q_embed_area(liquid_polygon);
      double q = q_embed_intersection_area
        (cs[], solid_normal.x, solid_normal.y, alpha_s,
         nf.x[], nf.y[], alpha_f[]);
      solid_reconstruction_l1 +=
        fabs(reconstructed_solid - cs[])*sq(Delta);
      liquid_reconstruction_l1 +=
        fabs(reconstructed_liquid - f[])*sq(Delta);
      product_intersection_l1 +=
        fabs(q - f[]*cs[])*sq(Delta);
      triple_cells++;
      f[] = clamp(q/cs[], 0., 1.);
    }
  }
  foreach()
    q_liquid[] = f[]*cs[];
  boundary ({f, q_liquid});
  fprintf (stderr, "Q_GEOMETRY %.12g %.12g %.12g %ld\n",
           solid_reconstruction_l1, liquid_reconstruction_l1,
           product_intersection_l1, triple_cells);
}

/* Use this path when the incoming fraction was constructed from an already
   closed physical liquid polygon which includes the solid boundary.  In that
   case the incoming full-cell fraction is q itself; intersecting it with cs a
   second time would double-count the bed cut. */
static void q_embed_initialize_physical_aperture (scalar f)
{
  double clipped = 0., maximum_violation = 0.;
  foreach (reduction(+:clipped) reduction(max:maximum_violation)) {
    double raw_q = f[];
    maximum_violation = max(maximum_violation, max(-raw_q, raw_q - cs[]));
    q_liquid[] = clamp(raw_q, 0., cs[]);
    clipped += fabs(q_liquid[] - raw_q)*sq(Delta);
    f[] = cs[] > 0. ? q_liquid[]/cs[] : 0.;
  }
  boundary ({f, q_liquid});
  fprintf (stderr, "Q_PHYSICAL_INIT %.12g %.12g\n",
           clipped, maximum_violation);
}
