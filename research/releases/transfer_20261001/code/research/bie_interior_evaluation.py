"""Oversampled analytic interior velocity for a periodic mixed BIE.

This evaluator does not change the solved boundary data. It refines the source
quadrature and differentiates the kernels analytically. Callers must compare
successive factors before treating near-boundary values as accurate.
"""

from __future__ import annotations

import numpy as np


def periodic_resample(values, factor):
    values = np.asarray(values, dtype=float)
    if factor < 1 or int(factor) != factor:
        raise ValueError("factor must be a positive integer")
    if factor == 1:
        return values.copy()
    spectrum = np.fft.rfft(values)
    if len(values) % 2 == 0:
        spectrum[-1] *= 0.5  # split the real Nyquist cosine into two modes
    return np.fft.irfft(spectrum, n=len(values) * factor) * factor


def boundary_quadrature(data, length, factor):
    arrays = [[] for _ in range(6)]
    for name, orientation in (("surface", 1), ("bottom", -1)):
        x, z = data[name + "_x"], data[name + "_z"]
        n = len(x)
        count = n * factor
        base = length * np.arange(n) / n
        fine_base = length * np.arange(count) / count
        fx = periodic_resample(x - base, factor) + fine_base
        fz = periodic_resample(z, factor)
        modes = np.fft.fftfreq(count) * count
        xa = (
            length / (2 * np.pi)
            + np.fft.ifft(1j * modes * np.fft.fft(fx - fine_base)).real
        )
        za = np.fft.ifft(1j * modes * np.fft.fft(fz)).real
        trace = periodic_resample(data[name + "_potential"], factor)
        q = periodic_resample(data[name + "_normal_derivative"], factor)
        weight = 2 * np.pi / count
        for dest, value in zip(
            arrays,
            (
                fx,
                fz,
                -orientation * za,
                orientation * xa,
                q * np.hypot(xa, za) * weight,
                trace * weight,
            ),
        ):
            dest.append(value)
    return tuple(np.concatenate(v) for v in arrays)


def interior_velocity(data, target_x, target_z, length, factor=8, *, xp=np, chunk=128):
    """Return analytic gradient of S q - D phi, with bounded target batches.

    ``xp=cupy`` enables float64 GPU evaluation without a second formula.
    Only zero background current is handled; the caller may add its constant
    current to the x component after supplying compatible boundary data.
    """
    sx, sz, nxds, nzds, qds, phi = [
        xp.asarray(v) for v in boundary_quadrature(data, length, factor)
    ]
    tx, tz = np.broadcast_arrays(target_x, target_z)
    result = np.empty((2, tx.size))
    scale = 2 * np.pi / length
    for start in range(0, tx.size, chunk):
        stop = min(start + chunk, tx.size)
        x = scale * (xp.asarray(tx.ravel()[start:stop, None]) - sx[None, :])
        z = scale * (xp.asarray(tz.ravel()[start:stop, None]) - sz[None, :])
        # Stable at small separations (unlike cosh(z)-cos(x)). Handoff
        # targets are close to the finite boundary, not at infinite height.
        denominator = 2 * (xp.sin(x / 2) ** 2 + xp.sinh(z / 2) ** 2)
        sinx, sinhz = xp.sin(x), xp.sinh(z)
        numerator = sinx * nxds + sinhz * nzds
        for direction, trig, second in (
            (0, sinx, xp.cos(x) * nxds),
            (1, sinhz, xp.cosh(z) * nzds),
        ):
            grad_s = -scale / (4 * np.pi) * (trig / denominator)
            grad_d = (
                scale**2
                / (4 * np.pi)
                * (second / denominator - numerator * trig / denominator**2)
            )
            velocity = grad_s @ qds - grad_d @ phi
            result[direction, start:stop] = (
                velocity if xp is np else xp.asnumpy(velocity)
            )
    if not np.isfinite(result).all():
        raise ValueError("non-finite interior velocity (possibly a boundary target)")
    return result.reshape((2,) + tx.shape)
