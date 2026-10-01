"""Project liquid and gas phase-cell means to one receiver cell velocity.

The receiver stores a single velocity in a mixed cell. This projection
preserves its represented density-weighted momentum. It does not choose the
VOF or embedded-boundary geometry, and it does not change legacy transfer
paths unless a caller opts in.
"""

from __future__ import annotations

import numpy as np


def project_phase_means_to_mixture(
    liquid_mean: np.ndarray,
    gas_mean: np.ndarray,
    q: np.ndarray,
    cs: np.ndarray,
    rho_liquid: float,
    rho_gas: float,
) -> np.ndarray:
    """Return the density-weighted mixture mean on active receiver cells.

    ``q`` is the liquid aperture fraction, ``cs`` is the total fluid aperture
    fraction, and both have the spatial grid shape. Phase means have shape
    ``(components, *q.shape)``. Cells with ``cs=0`` return zero. Nonfinite
    values are permitted only in a phase absent from that cell; this lets
    source solvers leave unused phase means undefined without polluting the
    projected velocity.

    On each active cell the result satisfies
    ``(rho_l*q + rho_g*(cs-q))*u_mix = rho_l*q*u_l + rho_g*(cs-q)*u_g``
    to floating-point roundoff.
    """
    q = np.asarray(q, dtype=np.float64)
    cs = np.asarray(cs, dtype=np.float64)
    liquid_mean = np.asarray(liquid_mean, dtype=np.float64)
    gas_mean = np.asarray(gas_mean, dtype=np.float64)
    if q.shape != cs.shape or q.ndim < 1:
        raise ValueError("q and cs must have the same non-scalar grid shape")
    if (liquid_mean.shape != gas_mean.shape
            or liquid_mean.ndim != q.ndim + 1
            or liquid_mean.shape[1:] != q.shape
            or liquid_mean.shape[0] < 1):
        raise ValueError("phase means must share shape (components, *q.shape)")
    if not np.isfinite(q).all() or not np.isfinite(cs).all():
        raise ValueError("q and cs must be finite")
    if np.any(q < 0) or np.any(cs < 0) or np.any(q > cs) or np.any(cs > 1):
        raise ValueError("fractions must satisfy 0 <= q <= cs <= 1")
    try:
        rho_liquid = float(rho_liquid)
        rho_gas = float(rho_gas)
    except (TypeError, ValueError) as exc:
        raise ValueError("phase densities must be positive finite scalars") from exc
    if (not np.isfinite(rho_liquid) or not np.isfinite(rho_gas)
            or rho_liquid <= 0 or rho_gas <= 0):
        raise ValueError("phase densities must be positive finite scalars")

    gas_fraction = cs - q
    liquid_active = q > 0
    gas_active = gas_fraction > 0
    if (not np.isfinite(liquid_mean[:, liquid_active]).all()
            or not np.isfinite(gas_mean[:, gas_active]).all()):
        raise ValueError("present-phase velocity means must be finite")
    liquid_weight = rho_liquid * q
    gas_weight = rho_gas * gas_fraction
    mass = liquid_weight + gas_weight
    momentum = np.zeros_like(liquid_mean)
    np.multiply(liquid_mean, liquid_weight[None, ...],
                out=momentum, where=liquid_active[None, ...])
    gas_momentum = np.zeros_like(gas_mean)
    np.multiply(gas_mean, gas_weight[None, ...],
                out=gas_momentum, where=gas_active[None, ...])
    momentum += gas_momentum
    return np.divide(momentum, mass[None, ...], out=np.zeros_like(momentum),
                     where=(mass > 0)[None, ...])
