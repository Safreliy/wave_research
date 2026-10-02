"""One cubic streamfunction trace per straight physical boundary segment.

This operator is for the receiver's *polygonal* boundary.  It neither moves
the boundary nor claims that a chord is the smooth BIE contour.  Parameter t
is the fraction of the straight chord, not BIE arclength.
"""

from __future__ import annotations

import numpy as np


def cartesian_blend_weight(s):
    """C1 blend: zero Cartesian weight and zero slope at the boundary."""
    s = np.asarray(s)
    if np.any((s < 0) | (s > 1)):
        raise ValueError("edge blend coordinate must lie in [0, 1]")
    return 3 * s * s - 2 * s * s * s


class ParentChordTrace:
    """Cubic Hermite values and exact restricted integrals on parent chords."""

    def __init__(self, curves: dict, length: float = 64.0):
        self.curves = curves
        self.length = float(length)
        self.chords = {}
        for name in ("surface", "bottom"):
            curve = curves[name]
            a = np.column_stack((curve["x"], curve["z"]))
            b = np.roll(a, -1, axis=0)
            b[-1, 0] += length
            delta = b - a
            norm = np.linalg.norm(delta, axis=1)
            if np.any(norm <= 0):
                raise ValueError("degenerate physical boundary chord")
            tangent = delta / norm[:, None]
            normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
            gradient = np.column_stack((curve["psi_x"], curve["psi_z"]))
            next_gradient = np.roll(gradient, -1, axis=0)
            y0 = np.asarray(curve["psi"], dtype=float)
            y1 = np.roll(y0, -1)
            s0 = np.sum(gradient * delta, axis=1)
            s1 = np.sum(next_gradient * delta, axis=1)
            self.chords[name] = {
                "a": a, "b": b, "delta": delta, "norm": norm,
                "tangent": tangent, "normal": normal,
                "normal0": np.sum(gradient * normal, axis=1),
                "normal1": np.sum(next_gradient * normal, axis=1),
                "c0": y0, "c1": s0,
                "c2": 3 * (y1 - y0) - 2 * s0 - s1,
                "c3": 2 * (y0 - y1) + s0 + s1,
            }

    def value(self, name: str, segment, t):
        c = self.chords[name]
        return ((c["c3"][segment] * t + c["c2"][segment]) * t
                + c["c1"][segment]) * t + c["c0"][segment]

    def derivative(self, name: str, segment, t):
        c = self.chords[name]
        return (3 * c["c3"][segment] * t + 2 * c["c2"][segment]) * t + c["c1"][segment]

    def gradient(self, name: str, segment, t):
        c = self.chords[name]
        tangent = c["tangent"][segment]
        normal = c["normal"][segment]
        tangential = self.derivative(name, segment, t) / c["norm"][segment]
        normal_component = ((1 - t) * c["normal0"][segment]
                            + t * c["normal1"][segment])
        return tangential[..., None] * tangent + normal_component[..., None] * normal

    def normal_derivative(self, name: str, segment, t):
        c = self.chords[name]
        return (1 - t) * c["normal0"][segment] + t * c["normal1"][segment]

    def integral(self, name: str, segment, t0, t1):
        """Integrate H(t) dt; two-point Gauss is exact for a cubic.

        Direct antiderivative subtraction loses digits for tiny cut edges.
        """
        mid = 0.5 * (t0 + t1)
        half = 0.5 * (t1 - t0)
        offset = half / np.sqrt(3.0)
        return half * (self.value(name, segment, mid - offset)
                       + self.value(name, segment, mid + offset))

    def mean(self, name: str, segment, t0, t1):
        mid = 0.5 * (t0 + t1)
        offset = 0.5 * (t1 - t0) / np.sqrt(3.0)
        return 0.5 * (self.value(name, segment, mid - offset)
                      + self.value(name, segment, mid + offset))

    def momentum(self):
        result = np.zeros(2)
        for name, sign in (("surface", 1), ("bottom", -1)):
            chord = self.chords[name]
            n = len(chord["a"])
            integral = self.integral(name, np.arange(n), 0.0, 1.0)
            result += sign * np.sum(integral[:, None] * chord["delta"], axis=0)
        return result

    def flat_surface_jet_means(self, xedges):
        """Exact chordwise means of H, N, NN on receiver horizontal edges.

        N and NN are linearly interpolated *source boundary data*, not values
        from an analytic reference.  This deliberately supports only a flat
        periodic surface; normal Taylor at a curved cut is a separate problem.
        """
        c = self.chords["surface"]
        surface = self.curves["surface"]
        if np.ptp(surface["z"]) > 1e-10:
            raise ValueError("matched source-jet edges require a flat surface")
        lefts, rights = c["a"][:, 0], c["b"][:, 0]
        if (np.any(np.diff(lefts) <= 0) or np.any(rights <= lefts)
                or abs(lefts[0]) > 1e-10
                or abs(rights[-1] - self.length) > 1e-10):
            raise ValueError("surface chords must cover one monotone period")
        xedges = np.asarray(xedges, dtype=float)
        if (abs(xedges[0]) > 1e-10
                or abs(xedges[-1] - self.length) > 1e-10
                or np.any(np.diff(xedges) <= 0)):
            raise ValueError("receiver horizontal edges must cover one period")
        second0 = np.asarray(surface["psi_nn"])
        second1 = np.roll(second0, -1)
        result = np.zeros((3, len(xedges) - 1))
        j = 0
        for i, (left, right) in enumerate(zip(xedges[:-1], xedges[1:])):
            width = right - left
            covered = 0.0
            while j < len(lefts) and rights[j] <= left + 1e-13:
                j += 1
            current = j
            while current < len(lefts) and lefts[current] < right - 1e-13:
                lo = max(left, lefts[current])
                hi = min(right, rights[current])
                if hi > lo:
                    t0 = (lo - lefts[current]) / (rights[current] - lefts[current])
                    t1 = (hi - lefts[current]) / (rights[current] - lefts[current])
                    span = hi - lo
                    result[0, i] += (rights[current] - lefts[current]) * self.integral(
                        "surface", current, t0, t1
                    )
                    mean_t = 0.5 * (t0 + t1)
                    result[1, i] += span * (c["normal0"][current]
                                                  + mean_t * (c["normal1"][current]
                                                              - c["normal0"][current]))
                    result[2, i] += span * (second0[current]
                                                  + mean_t * (second1[current]
                                                              - second0[current]))
                    covered += span
                current += 1
            if abs(covered - width) > 1e-10 * max(1.0, width):
                raise ValueError("surface jet did not cover receiver edge")
            result[:, i] /= width
        return result


def gauge_to_source_horizontal_momentum(curves: dict, length: float = 64.0):
    """Shift only surface psi by a constant to preserve source Mx exactly.

    The periodic surface's total vertical displacement is zero, hence this
    constant cannot repair Mz.  Return an independent Mz discrepancy.
    """
    target = np.asarray(curves["surface"].get("source_momentum_target"), dtype=float)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("Hermite trace requires the BIE source momentum target")
    before = ParentChordTrace(curves, length).momentum()
    surface_dx = np.sum(ParentChordTrace(curves, length).chords["surface"]["delta"][:, 0])
    if not np.isclose(surface_dx, length, rtol=0, atol=1e-10):
        raise ValueError("periodic surface does not span the declared length")
    gauge = float((target[0] - before[0]) / surface_dx)
    adjusted = {name: dict(curve) for name, curve in curves.items()}
    adjusted["surface"]["psi"] = np.asarray(curves["surface"]["psi"]) + gauge
    trace = ParentChordTrace(adjusted, length)
    after = trace.momentum()
    if not np.isclose(after[0], target[0], rtol=1e-11, atol=1e-11):
        raise ValueError("horizontal source momentum gauge did not close")
    return adjusted, trace, {
        "surface_gauge": gauge,
        "source_momentum_target": target.tolist(),
        "hermite_momentum_before_gauge": before.tolist(),
        "hermite_momentum_after_gauge": after.tolist(),
        "vertical_source_momentum_defect": float(after[1] - target[1]),
    }
