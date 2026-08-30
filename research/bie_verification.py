"""Generate convergence data and an SVG verification plate for periodic_bie."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from periodic_bie import PeriodicMirrorBIE, manufactured_harmonic_solution


def _polyline(x: np.ndarray, y: np.ndarray, map_x, map_y, color: str, width: float) -> str:
    points = " ".join(
        f"{map_x(float(xi)):.2f},{map_y(float(yi)):.2f}" for xi, yi in zip(x, y)
    )
    return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{width}"/>'


def run(output_prefix: Path) -> None:
    rows = []
    finest = None
    for n in (32, 64, 128, 256):
        x = np.arange(n) * 2.0 * math.pi / n
        eta = 0.08 * np.cos(x) + 0.025 * np.cos(2.0 * x + 0.3)
        solver = PeriodicMirrorBIE(eta, depth=1.0)
        trace, exact = manufactured_harmonic_solution(solver, mode=2)
        result = solver.dirichlet_to_neumann(trace)
        error = float(np.linalg.norm(result.dno - exact) / np.linalg.norm(exact))
        rows.append(
            {
                "n": n,
                "relative_l2_dno_error": error,
                "boundary_residual": result.boundary_residual,
                "normal_offset": result.offset,
            }
        )
        finest = (solver, result.dno, exact)

    for previous, current in zip(rows, rows[1:]):
        current["error_ratio_from_previous"] = (
            previous["relative_l2_dno_error"] / current["relative_l2_dno_error"]
        )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(
        json.dumps({"manufactured_mode": 2, "results": rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    assert finest is not None
    solver, computed, exact = finest
    width, height = 1100, 760
    left, right = 95.0, 1050.0
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="95" y="38" font-family="Segoe UI,Arial" font-weight="600" font-size="24" fill="#172433">Second-kind periodic BIE verification</text>',
        '<text x="95" y="64" font-family="Segoe UI,Arial" font-size="14" fill="#5b6b7b">Manufactured harmonic mode, flat Neumann bed enforced by mirror symmetry</text>',
    ]

    x = solver.x
    top_z = solver.eta
    mirror_z = -2.0 * solver.depth - solver.eta
    geo_top, geo_bottom = 92.0, 282.0
    z_min, z_max = float(np.min(mirror_z)) - 0.1, float(np.max(top_z)) + 0.1
    map_x = lambda value: left + value / solver.length * (right - left)
    map_geo = lambda value: geo_bottom - (value - z_min) / (z_max - z_min) * (geo_bottom - geo_top)
    elements.extend(
        [
            '<text x="95" y="88" font-family="Segoe UI,Arial" font-size="14" fill="#5b6b7b">Geometry and reflected boundary</text>',
            _polyline(x, top_z, map_x, map_geo, "#087eaa", 2.4),
            _polyline(x, mirror_z, map_x, map_geo, "#7a4fa3", 2.0),
            f'<line x1="{left}" y1="{map_geo(-solver.depth):.2f}" x2="{right}" y2="{map_geo(-solver.depth):.2f}" stroke="#d94841" stroke-dasharray="7,6" stroke-width="1.6"/>',
            f'<text x="{right-120}" y="{map_geo(-solver.depth)-7:.2f}" font-family="Segoe UI,Arial" font-size="12" fill="#d94841">Neumann bed</text>',
        ]
    )

    dno_top, dno_bottom = 345.0, 555.0
    d_min = float(min(np.min(computed), np.min(exact))) * 1.1
    d_max = float(max(np.max(computed), np.max(exact))) * 1.1
    map_dno = lambda value: dno_bottom - (value - d_min) / (d_max - d_min) * (dno_bottom - dno_top)
    elements.extend(
        [
            '<text x="95" y="325" font-family="Segoe UI,Arial" font-size="14" fill="#5b6b7b">Dirichlet-to-Neumann map at N=256</text>',
            f'<rect x="{left}" y="{dno_top}" width="{right-left}" height="{dno_bottom-dno_top}" fill="white" stroke="#d7e0e8"/>',
            _polyline(x, exact, map_x, map_dno, "#172433", 2.2),
            _polyline(x, computed, map_x, map_dno, "#d94841", 1.5),
            '<line x1="730" y1="320" x2="758" y2="320" stroke="#172433" stroke-width="2.2"/><text x="766" y="325" font-family="Segoe UI,Arial" font-size="12" fill="#172433">exact</text>',
            '<line x1="850" y1="320" x2="878" y2="320" stroke="#d94841" stroke-width="1.5"/><text x="886" y="325" font-family="Segoe UI,Arial" font-size="12" fill="#d94841">BIE</text>',
        ]
    )

    chart_left, chart_right = 150.0, 985.0
    chart_top, chart_bottom = 625.0, 710.0
    ns = np.asarray([row["n"] for row in rows], dtype=float)
    errors = np.asarray([row["relative_l2_dno_error"] for row in rows])
    log_n, log_e = np.log2(ns), np.log10(errors)
    map_n = lambda value: chart_left + (value - log_n[0]) / (log_n[-1] - log_n[0]) * (chart_right - chart_left)
    map_e = lambda value: chart_bottom - (value - np.min(log_e)) / (np.max(log_e) - np.min(log_e)) * (chart_bottom - chart_top)
    points = " ".join(f"{map_n(n):.2f},{map_e(e):.2f}" for n, e in zip(log_n, log_e))
    elements.extend(
        [
            '<text x="95" y="600" font-family="Segoe UI,Arial" font-size="14" fill="#5b6b7b">Relative L2 DNO error under grid refinement</text>',
            f'<line x1="{chart_left}" y1="{chart_bottom}" x2="{chart_right}" y2="{chart_bottom}" stroke="#9dacba"/>',
            f'<polyline points="{points}" fill="none" stroke="#087eaa" stroke-width="2.5"/>',
        ]
    )
    for n, e, row in zip(log_n, log_e, rows):
        px, py = map_n(n), map_e(e)
        elements.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="5" fill="#087eaa"/>')
        elements.append(f'<text x="{px-12:.2f}" y="{chart_bottom+22:.2f}" font-family="Segoe UI,Arial" font-size="12" fill="#5b6b7b">{row["n"]}</text>')
        elements.append(f'<text x="{px+8:.2f}" y="{py-7:.2f}" font-family="Segoe UI,Arial" font-size="11" fill="#172433">{row["relative_l2_dno_error"]:.2e}</text>')
    elements.append("</svg>")
    output_prefix.with_suffix(".svg").write_text("\n".join(elements), encoding="utf-8")


if __name__ == "__main__":
    run(Path("results/bie_verification"))
