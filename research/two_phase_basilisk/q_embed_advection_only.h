/**
Geometric q transport coupled to the standard centered velocity advection.

This is a diagnostic ablation.  It isolates the double-PLIC liquid-aperture
flux from the phase-momentum tracer extension in conserving_q_embed.h.
*/

static scalar * q_embed_interfaces_saved_advection_only = NULL;

event q_embed_cutcell_stability (i++, last)
  dtmax = q_embed_stable_timestep (f, uf, dtmax);

event vof (i++)
{
  q_embed_vof_advection (f, i);
  q_embed_interfaces_saved_advection_only = interfaces;
  interfaces = NULL;
}

event tracer_advection (i++)
  interfaces = q_embed_interfaces_saved_advection_only;
