from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from two_phase_basilisk.analyze_topology import trapped_air
from two_phase_basilisk.render_bie_handoff import (
    connected_component_labels,
    persistent_onset,
    topology_record_from_liquid,
)


class BasiliskTopologyTests(unittest.TestCase):
    def _write_frame(self, path: Path, connected: bool, bubble: bool) -> None:
        height, width = 80, 320
        encoded = np.zeros((height, width, 3), dtype=np.uint8)
        encoded[:] = (0, 0, 143)  # air in the Basilisk output_ppm colour map
        encoded[40:, :] = (135, 0, 0)  # liquid
        if bubble:
            encoded[48:52, 150:160] = (0, 0, 143)
        if connected:
            encoded[:49, 154:157] = (0, 0, 143)
        Image.fromarray(encoded).save(path)

    def test_closed_air_component_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "closed.ppm"
            self._write_frame(path, connected=False, bubble=True)
            result = trapped_air(path, minimum_pixels=4)
            self.assertEqual(result["pocket_count"], 1)
            self.assertGreaterEqual(result["largest_pocket_pixels"], 40)

    def test_boundary_connected_air_is_not_impact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "open.ppm"
            self._write_frame(path, connected=True, bubble=True)
            result = trapped_air(path, minimum_pixels=4)
            self.assertEqual(result["pocket_count"], 0)

    def test_persistent_onset_rejects_one_frame_flicker(self) -> None:
        records = [
            {"pocket_count": count}
            for count in (0, 1, 0, 1, 1, 1, 0)
        ]
        self.assertEqual(persistent_onset(records, persistence_frames=3), 3)
        self.assertIsNone(persistent_onset(records, persistence_frames=4))

    def test_physical_area_gate_is_raster_independent(self) -> None:
        liquid = np.zeros((40, 80), dtype=bool)
        liquid[20:, :] = True
        liquid[24:28, 35:45] = False
        bottom_x = np.array([0.0, 2.0])
        bottom_z = np.array([-2.0, -2.0])
        rejected = topology_record_from_liquid(
            liquid, bottom_x, bottom_z, 2.0, 0.0, 2.0, -1.0, 1.0,
            minimum_pixels=1, minimum_area=0.06,
        )
        accepted = topology_record_from_liquid(
            liquid, bottom_x, bottom_z, 2.0, 0.0, 2.0, -1.0, 1.0,
            minimum_pixels=1, minimum_area=0.04,
        )
        self.assertEqual(rejected["pocket_count"], 0)
        self.assertEqual(accepted["pocket_count"], 1)

    def test_periodic_seam_component_is_counted_once(self) -> None:
        liquid = np.ones((40, 80), dtype=bool)
        liquid[:15, :] = False
        liquid[22:26, :3] = False
        liquid[22:26, -3:] = False
        bottom_x = np.array([0.0, 2.0])
        bottom_z = np.array([-2.0, -2.0])
        result = topology_record_from_liquid(
            liquid, bottom_x, bottom_z, 2.0, 0.0, 2.0, -1.0, 1.0,
            minimum_pixels=1, minimum_area=0.0,
        )
        self.assertEqual(result["unfiltered_pocket_count"], 1)

    def test_periodic_label_union_preserves_component_area(self) -> None:
        mask = np.zeros((6, 10), dtype=bool)
        mask[2:4, :2] = True
        mask[2:4, -3:] = True
        labels, count = connected_component_labels(mask, periodic_x=True)
        self.assertEqual(count, 1)
        self.assertEqual(np.count_nonzero(labels == 1), 10)

    def test_bed_guard_rejects_attached_component_not_only_guard_pixels(self) -> None:
        liquid = np.ones((40, 80), dtype=bool)
        liquid[:12, :] = False
        liquid[26:29, 30:45] = False
        bottom_x = np.array([0.0, 2.0])
        bottom_z = np.array([-0.5, -0.5])
        result = topology_record_from_liquid(
            liquid, bottom_x, bottom_z, 2.0, 0.0, 2.0, -1.0, 1.0,
            minimum_pixels=1, minimum_area=0.0, bed_guard_distance=0.1,
        )
        self.assertEqual(result["pocket_count"], 0)


if __name__ == "__main__":
    unittest.main()
