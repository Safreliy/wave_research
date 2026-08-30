import numpy as np

from cutcell_q_transport import (
    Geometry,
    clip_half_plane,
    integrate_interval_overlap,
    polygon_area,
    rectangle,
    run_resolution,
)


def test_half_plane_clip_triangle_area() -> None:
    triangle = clip_half_plane(rectangle(0.0, 1.0, 0.0, 1.0), (1.0, 1.0), 1.0)
    assert abs(polygon_area(triangle) - 0.5) < 1.0e-14


def test_affine_interval_overlap_integral() -> None:
    # [t, 1] has length 1-t on t in [0,1].
    value = integrate_interval_overlap([(1.0, 0.0)], [(0.0, 1.0)], 1.0)
    assert abs(value - 0.5) < 1.0e-14


def test_shared_q_flux_is_conservative_and_bounded() -> None:
    result, _ = run_resolution(64, 0.2, Geometry())
    assert abs(result.conservative_relative_volume_change) < 1.0e-13
    assert result.conservative_lower_bound_violation < 1.0e-12
    assert result.conservative_upper_bound_violation < 1.0e-12


def test_double_intersection_is_not_independent_product() -> None:
    result, _ = run_resolution(32, 0.2, Geometry())
    assert result.product_geometry_l1 > 1.0e-4
    assert abs(result.product_relative_volume_change) > 1.0e-8


def test_piecewise_linear_manufactured_case_is_exact() -> None:
    errors = [
        run_resolution(n, 0.2, Geometry())[0].conservative_transport_l1
        for n in (24, 48, 96)
    ]
    assert np.all(np.asarray(errors) < 1.0e-12), errors
