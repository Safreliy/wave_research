import numpy as np
from pathlib import Path

from project_mac_face_flux import (
    load_receiver_quadrature,
    receiver_cell_velocity_operators,
)
from receiver_momentum_correction import apply_profile_correction


def test_global_shift_is_l2_lower_bound() -> None:
    rng = np.random.default_rng(7)
    weights = rng.uniform(0.1, 2.0, 40)
    velocity = rng.normal(size=(40, 2))
    target = np.sum(weights[:, None] * velocity, axis=0) + np.array([0.3, -0.2])
    global_result, _ = apply_profile_correction(
        velocity, weights, target, np.ones(40), "global"
    )
    local_result, _ = apply_profile_correction(
        velocity, weights, target, rng.uniform(0.05, 1.0, 40), "local"
    )
    assert global_result.relative_momentum_residual < 1.0e-14
    assert local_result.relative_momentum_residual < 1.0e-14
    assert abs(global_result.norm_ratio_to_global_minimum - 1.0) < 1.0e-14
    assert local_result.norm_ratio_to_global_minimum >= 1.0


def test_hard_support_penalty() -> None:
    weights = np.ones(100)
    velocity = np.zeros((100, 2))
    target = np.array([1.0, 0.0])
    profile = np.zeros(100)
    profile[:25] = 1.0
    result, _ = apply_profile_correction(velocity, weights, target, profile, "quarter")
    assert abs(result.norm_ratio_to_global_minimum - 2.0) < 1.0e-14


def test_python_receiver_operator_matches_exported_c_sampling() -> None:
    directory = Path(__file__).resolve().parent
    archive = directory / "mac_t20303125_h200_m1e8.npz"
    receiver_csv = directory.parent / "two_phase_basilisk" / "receiver_initial_state.csv"
    if not archive.exists() or not receiver_csv.exists():
        return
    with np.load(archive) as loaded:
        x = loaded["grid_x"]
        z = loaded["grid_z"]
        streamfunction = loaded["mac_streamfunction"].ravel()
    dx = float(x[1] - x[0])
    dz = float(z[1] - z[0])
    x_edges = x - 0.5 * dx
    z_edges = np.concatenate(([z[0] - 0.5 * dz], z + 0.5 * dz))
    receiver = np.genfromtxt(receiver_csv, delimiter=",", names=True)
    operator_u, operator_w = receiver_cell_velocity_operators(
        x_edges,
        z_edges,
        receiver["x"],
        receiver["y"],
        receiver["delta"],
    )
    reconstructed = np.concatenate(
        (operator_u @ streamfunction, operator_w @ streamfunction)
    )
    exported = np.concatenate((receiver["u_x"], receiver["u_y"]))
    assert np.linalg.norm(reconstructed - exported) / np.linalg.norm(exported) < 1.0e-8


def test_receiver_q_quadrature_accepts_explicit_cell_area(tmp_path: Path) -> None:
    path = tmp_path / "receiver.csv"
    path.write_text(
        "x,y,delta,cell_area,f,cs,q,u_x,u_y\n"
        "0.25,0.25,0.5,0.25,0.5,0.4,0.2,1.0,-0.5\n",
        encoding="ascii",
    )
    receiver, area, schema = load_receiver_quadrature(path)
    assert len(receiver) == 1
    assert np.array_equal(area, np.array([0.25]))
    assert schema == "explicit-full-cell-area-v2"


def test_receiver_q_quadrature_rejects_embedded_dv(tmp_path: Path) -> None:
    path = tmp_path / "legacy.csv"
    path.write_text(
        "x,y,delta,dv,f,cs,q,u_x,u_y\n"
        "0.25,0.25,0.5,0.1,0.5,0.4,0.2,1.0,-0.5\n",
        encoding="ascii",
    )
    try:
        load_receiver_quadrature(path)
    except ValueError as error:
        assert "apply cs twice" in str(error)
    else:
        raise AssertionError("embedded dv must not be accepted as full cell area")
