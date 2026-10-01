"""Small invariant checks for the opt-in parent-chord trace operator."""

from __future__ import annotations

import numpy as np

from liquid_dirichlet_transfer import BoundarySegments
from physical_trace import (ParentChordTrace, cartesian_blend_weight,
                            gauge_to_source_horizontal_momentum)


def fixture():
    x = np.arange(8) * 8.0
    k = 2 * np.pi / 64
    curves = {}
    for name, z, psi in (("surface", 1.0, 1 + np.sin(k * x)),
                         ("bottom", 0.0, np.zeros_like(x))):
        curves[name] = {
            "x": x.copy(), "z": np.full_like(x, z), "psi": psi,
            "psi_x": k * np.cos(k * x) if name == "surface" else np.zeros_like(x),
            "psi_z": np.zeros_like(x), "psi_n": np.zeros_like(x),
            "psi_nn": np.zeros_like(x),
        }
    return curves


def main():
    curves = fixture()
    trace = ParentChordTrace(curves)
    for name in ("surface", "bottom"):
        c = trace.chords[name]
        j = np.arange(len(c["a"]))
        np.testing.assert_allclose(trace.value(name, j, 0.0), c["c0"], atol=1e-15)
        np.testing.assert_allclose(trace.value(name, j, 1.0),
                                   np.roll(c["c0"], -1), atol=1e-15)
        np.testing.assert_allclose(trace.derivative(name, j, 0.0), c["c1"], atol=1e-15)
        np.testing.assert_allclose(trace.derivative(name, j, 1.0),
                                   np.sum(np.roll(np.column_stack((curves[name]["psi_x"],
                                                                    curves[name]["psi_z"])),
                                                  -1, axis=0) * c["delta"], axis=1), atol=1e-15)
        # A tiny interval tests the cancellation-safe restricted integral.
        for a, b, split in ((0.02, 0.99, 0.31), (0.3, 0.30001, 0.300004)):
            direct = trace.integral(name, j, a, b)
            pieces = (trace.integral(name, j, a, split)
                      + trace.integral(name, j, split, b))
            np.testing.assert_allclose(direct, pieces, rtol=1e-12, atol=1e-15)
    xprobe = 1.23
    segment = BoundarySegments(curves, 64.0, trace=trace)
    theta, value, hit = segment.crossings(
        np.array([[xprobe, 0.75]]), np.array([[xprobe, 1.25]])
    )
    assert hit[0] and np.isclose(theta[0], 0.5)
    np.testing.assert_allclose(value[0], trace.value("surface", 0, xprobe / 8), atol=1e-14)
    continued, _ = segment.continue_normal(np.array([[xprobe, 1.1]]))
    np.testing.assert_allclose(continued[0], value[0], atol=1e-14)
    old = trace.momentum()
    curves["surface"]["source_momentum_target"] = old + [3.0, 0.0]
    adjusted, shifted, report = gauge_to_source_horizontal_momentum(curves)
    np.testing.assert_allclose(shifted.momentum(), old + [3.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(report["surface_gauge"], 3 / 64, atol=1e-15)
    np.testing.assert_allclose(adjusted["surface"]["psi"] - curves["surface"]["psi"],
                               3 / 64, atol=1e-15)
    constants = fixture()
    constants["surface"]["psi"][:] = 5.0
    constants["surface"]["psi_x"][:] = 0.0
    constants["surface"]["psi_z"][:] = 2.0
    constants["surface"]["psi_nn"][:] = 3.0
    jet = ParentChordTrace(constants)
    edge_means = jet.flat_surface_jet_means(np.arange(17) * 4.0)
    np.testing.assert_allclose(edge_means,
                               np.broadcast_to(np.array([5., 2., 3.])[:, None],
                                               edge_means.shape), atol=1e-14)
    # One parent chord reproduces a prescribed linear polynomial exactly.
    polynomial = fixture()
    polynomial["surface"]["psi"][0:2] = [1.0, 3.0]
    polynomial["surface"]["psi_x"][0:2] = 0.25
    linear = ParentChordTrace(polynomial)
    np.testing.assert_allclose(linear.value("surface", 0, np.array([.2, .7])),
                               [1.4, 2.4], atol=1e-14)
    np.testing.assert_allclose(linear.integral("surface", 0, .2, .7),
                               (.7-.2) + (.7**2-.2**2), atol=1e-14)
    eps = 1e-5
    np.testing.assert_allclose(cartesian_blend_weight([0, 1]), [0, 1], atol=0)
    assert cartesian_blend_weight(eps) / eps < 4 * eps
    assert (1 - cartesian_blend_weight(1-eps)) / eps < 4 * eps
    curved = fixture()
    curved["surface"]["z"][2] += 0.01
    try:
        ParentChordTrace(curved).flat_surface_jet_means(np.arange(17) * 4.0)
    except ValueError as error:
        assert "flat surface" in str(error)
    else:
        raise AssertionError("curved surface should fail fast")
    print("physical trace endpoints, slopes, subdivision, crossings, continuation, gauge, jets, blend: PASS")


if __name__ == "__main__":
    main()
