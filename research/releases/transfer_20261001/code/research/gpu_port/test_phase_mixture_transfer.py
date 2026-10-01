"""Invariant tests and optional frozen-input regression for mixture projection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tarfile
from pathlib import Path

import numpy as np

from phase_mixture_transfer import project_phase_means_to_mixture


def expect_value_error(call) -> None:
    try:
        call()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_invariants() -> dict:
    rng = np.random.default_rng(20261001)
    cs = rng.uniform(0.01, 0.9, size=(4, 7))
    q = cs * rng.uniform(0.01, 0.99, size=cs.shape)
    liquid = rng.normal(size=(2, *cs.shape))
    gas = rng.normal(size=liquid.shape)
    rho_l, rho_g = 3.7, 0.12
    mixed = project_phase_means_to_mixture(liquid, gas, q, cs, rho_l, rho_g)
    expected_momentum = rho_l * q[None] * liquid + rho_g * (cs - q)[None] * gas
    actual_momentum = (rho_l * q + rho_g * (cs - q))[None] * mixed
    np.testing.assert_allclose(actual_momentum, expected_momentum, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(actual_momentum.sum(axis=(1, 2)),
                               expected_momentum.sum(axis=(1, 2)), rtol=1e-14, atol=1e-14)

    # Full liquid, full gas, empty solid, and a mixed cut cell with cs<1.
    q = np.array([[1.0, 0.0], [0.0, 0.2]])
    cs = np.array([[1.0, 1.0], [0.0, 0.8]])
    liquid = np.array([[[2.0, np.nan], [np.nan, 4.0]],
                       [[-3.0, np.nan], [np.nan, 1.0]]])
    gas = np.array([[[np.nan, 7.0], [np.nan, -2.0]],
                    [[np.nan, 5.0], [np.nan, 3.0]]])
    result = project_phase_means_to_mixture(liquid, gas, q, cs, 2.0, 0.5)
    np.testing.assert_allclose(result[:, 0, 0], [2.0, -3.0], atol=0)
    np.testing.assert_allclose(result[:, 0, 1], [7.0, 5.0], atol=0)
    np.testing.assert_allclose(result[:, 1, 0], [0.0, 0.0], atol=0)
    mix_weight = 2.0 * 0.2 + 0.5 * 0.6
    np.testing.assert_allclose(result[:, 1, 1],
                               [(2.0 * 0.2 * 4.0 + 0.5 * 0.6 * -2.0) / mix_weight,
                                (2.0 * 0.2 * 1.0 + 0.5 * 0.6 * 3.0) / mix_weight],
                               rtol=1e-15, atol=1e-15)
    expect_value_error(lambda: project_phase_means_to_mixture(
        liquid, gas, np.array([[1., 0.], [0., .9]]), cs, 2., .5))
    expect_value_error(lambda: project_phase_means_to_mixture(
        liquid, gas, q, cs, -2., .5))
    expect_value_error(lambda: project_phase_means_to_mixture(
        liquid, gas, q, cs, 2., math.inf))
    expect_value_error(lambda: project_phase_means_to_mixture(
        np.where(np.indices(liquid.shape)[1] == 0, np.nan, liquid), gas,
        q, cs, 2., .5))
    expect_value_error(lambda: project_phase_means_to_mixture(
        liquid[:, :1], gas, q, cs, 2., .5))
    return {"random_cellwise_and_global_momentum_identity": "passed",
            "full_liquid_gas_solid_and_cut_cell": "passed",
            "inactive_nan_allowed_and_active_nan_rejected": "passed",
            "invalid_fraction_density_shape_rejected": "passed"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_regression(archive: Path) -> dict:
    from linear_twofluid_wave_reference import Wave

    shape = 32, 512
    result = {}
    with tarfile.open(archive, "r:gz") as tar:
        for name, amplitude in (("A_mass", 2e-5), ("Ahalf_mass", 1e-5)):
            base = f"flat_native_L9_twofluid/{name}/"
            def raw(component):
                data = tar.extractfile(base + component + ".raw").read()
                if len(data) != 8 * math.prod(shape):
                    raise ValueError(f"wrong frozen field size: {name}/{component}")
                return np.frombuffer(data, dtype="<f8").reshape(shape)

            cells = Wave(potential_amplitude=amplitude).cells(512, 0)
            q, cs = raw("q"), raw("cs")
            velocity = project_phase_means_to_mixture(
                cells["liquid_mean"], cells["gas_mean"], q, cs,
                1.0, 1.0 / 850.0,
            )
            actual = np.stack((raw("vx"), raw("vy")))
            max_difference = float(np.max(np.abs(velocity - actual)))
            if max_difference > 2e-13:
                raise AssertionError(f"helper does not reproduce frozen {name}: {max_difference}")
            result[name] = {"max_absolute_velocity_difference": max_difference,
                            "mixed_cells": int(np.count_nonzero((q > 0) & (cs > q)))}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-inputs", type=Path,
                        help="Optional frozen two-fluid input tar for executed-case regression")
    parser.add_argument("--report", type=Path,
                        help="Write a compact machine-readable validation report")
    args = parser.parse_args()
    report = {"schema": "phase-mixture-transfer-verification-v1",
              "helper_sha256": sha256(Path(__file__).with_name("phase_mixture_transfer.py")),
              "tests": test_invariants()}
    if args.frozen_inputs:
        report["frozen_input_archive_sha256"] = sha256(args.frozen_inputs)
        report["frozen_regression"] = frozen_regression(args.frozen_inputs)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
