"""Decoupled gas continuation: never modify vertices supporting liquid cells.

Minimize discrete Dirichlet energy outside the fixed region, with a periodic
half-space Dirichlet-to-Neumann condition at the upper edge. This is a tested
initialization candidate, not a validated viscous two-phase boundary closure.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


def liquid_support_vertices(fraction):
    liquid = np.asarray(fraction) > 0
    horizontal = liquid | np.roll(liquid, 1, axis=1)
    fixed = np.zeros((liquid.shape[0] + 1, liquid.shape[1]), dtype=bool)
    fixed[:-1] |= horizontal
    fixed[1:] |= horizontal
    return fixed


def extend_gas(
    psi,
    fraction,
    dx,
    *,
    additional_fixed=None,
    rtol=1e-11,
    maxiter=10000,
    backend="numpy",
):
    psi = np.asarray(psi, dtype=float)
    fraction = np.asarray(fraction, dtype=float)
    if (
        fraction.ndim != 2
        or not np.isfinite(dx)
        or dx <= 0
        or psi.shape != (fraction.shape[0] + 1, fraction.shape[1])
    ):
        raise ValueError("incompatible shape or cell size")
    if (
        not np.isfinite(psi).all()
        or not np.isfinite(fraction).all()
        or np.any((fraction < 0) | (fraction > 1))
    ):
        raise ValueError("invalid input fields")
    fixed = liquid_support_vertices(fraction)
    if additional_fixed is not None:
        extra = np.asarray(additional_fixed, dtype=bool)
        if extra.shape != fixed.shape:
            raise ValueError("additional fixed mask has incompatible shape")
        fixed |= extra
    if backend not in ("numpy", "cupy"):
        raise ValueError("unknown gas extension backend")
    if not fixed.any():
        raise ValueError("no fixed liquid/reference vertices")
    unknown = ~fixed
    if not unknown.any():
        return psi.copy(), {
            "iterations": 0,
            "fixed_change_linf": 0.0,
            "relative_residual": 0.0,
        }
    xp, operator_class, solve = np, LinearOperator, cg
    if backend == "cupy":
        import cupy as xp
        from cupyx.scipy.sparse.linalg import LinearOperator as operator_class
        from cupyx.scipy.sparse.linalg import cg as solve
    device_psi = xp.asarray(psi)
    device_unknown = xp.asarray(unknown)
    _ny, nx = psi.shape
    wave = 2 * xp.pi * xp.fft.rfftfreq(nx, d=dx)

    def apply(values):
        # Graph Laplacian is the derivative of face-based kinetic energy.
        result = 2 * values - xp.roll(values, 1, axis=1) - xp.roll(values, -1, axis=1)
        # Half control volumes at the endpoints give a second-order DtN
        # boundary closure; full weights introduce a first-order bias.
        result[[0, -1]] *= 0.5
        jumps = values[1:] - values[:-1]
        result[1:] += jumps
        result[:-1] -= jumps
        result[-1] += dx * xp.fft.irfft(wave * xp.fft.rfft(values[-1]), n=nx)
        return result

    prescribed = xp.where(xp.asarray(fixed), device_psi, 0.0)
    rhs = -apply(prescribed)[device_unknown]

    def matvec(values):
        full = xp.zeros_like(device_psi)
        full[device_unknown] = values
        return apply(full)[device_unknown]

    operator = operator_class(
        (int(unknown.sum()), int(unknown.sum())), matvec=matvec, dtype=float
    )
    count = 0

    def callback(_):
        nonlocal count
        count += 1

    solution, info = solve(
        operator,
        rhs,
        x0=device_psi[device_unknown],
        atol=1e-14,
        maxiter=maxiter,
        callback=callback,
        rtol=rtol,
    )
    if info != 0:
        raise ValueError(f"gas extension failed to converge: {info}")
    result = psi.copy()
    result[unknown] = solution if backend == "numpy" else xp.asnumpy(solution)
    residual = float(xp.linalg.norm(matvec(solution) - rhs)) / max(
        float(xp.linalg.norm(rhs)), np.finfo(float).tiny
    )
    return result, {
        "backend": backend,
        "iterations": count,
        "fixed_change_linf": float(abs(result[fixed] - psi[fixed]).max()),
        "relative_residual": float(residual),
        "fixed_vertices": int(fixed.sum()),
        "gas_vertices": int(unknown.sum()),
        "top_closure": "periodic half-space DtN; not a solver pressure boundary",
    }
