"""Audit retained analytic-test arrays and recompute reported error norms.

The analytic means are additionally checked by independent Gaussian quadrature
of the velocity, rather than by the runner's antiderivative formula.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(summary_path):
    summary = json.loads(summary_path.read_text())
    nodes, weights = np.polynomial.legendre.leggauss(8)
    weights = weights / 2
    results = []
    for row in summary['rows']:
        path = summary_path.parent / row['arrays']
        if sha(path) != row['arrays_sha256']:
            raise ValueError(f'Array hash mismatch: {path}')
        with np.load(path) as data:
            if any(not np.isfinite(data[key]).all() for key in data.files):
                raise ValueError(f'Nonfinite array: {path}')
            q = data['q']
            ref = data['exact_phase_mean']
            h = 64 / row['nx']
            lo, hi = data['liquid_lower_z'], data['liquid_upper_z']
            if not np.allclose(q, ((hi - lo) / h)[:, None], rtol=0, atol=2e-10):
                raise ValueError('Stored q disagrees with analytic rectangle widths')
            if not np.allclose(q[-1], row['surface_liquid_fraction'], rtol=0, atol=2e-10):
                raise ValueError('Wrong prescribed surface fraction')
            k, depth = 2 * np.pi * 4 / 64, 4
            x_nodes = data['cell_center_x'][:, None] + h * nodes[None, :] / 2
            z_nodes = (hi + lo)[:, None] / 2 + (hi - lo)[:, None] * nodes[None, :] / 2
            sin_mean = np.sin(k * x_nodes) @ weights
            cos_mean = np.cos(k * x_nodes) @ weights
            cosh_mean = np.cosh(k * (z_nodes + depth)) @ weights
            sinh_mean = np.sinh(k * (z_nodes + depth)) @ weights
            quadrature = np.array([
                -k * cosh_mean[:, None] * sin_mean[None, :],
                k * sinh_mean[:, None] * cos_mean[None, :],
            ]) / np.cosh(k * depth)
            quadrature_error = float(np.linalg.norm(quadrature - ref) / np.linalg.norm(ref))
            if quadrature_error > 5e-10:
                raise ValueError('Analytic cell means fail independent quadrature check')
            cut = (q > 0) & (q < 1)
            checks = {}
            for field, prefix in (
                ('fitted_phase_mean', 'fitted'),
                ('arithmetic_mac_mean', 'baseline'),
                ('exact_centroid_oracle', 'centroid_oracle'),
            ):
                difference = data[field] - ref
                for suffix, error in (
                    ('all', np.linalg.norm(difference) / np.linalg.norm(ref)),
                    ('cut', np.linalg.norm(difference[:, cut]) / np.linalg.norm(ref[:, cut])),
                ):
                    key = f'{prefix}_relative_l2_{suffix}'
                    if not np.isclose(error, row[key], rtol=1e-11, atol=1e-14):
                        raise ValueError(f'Norm mismatch: {key}')
                    checks[key] = float(error)
        results.append({'arrays': path.name, 'arrays_sha256': sha(path),
                        'independent_quadrature_relative_error': quadrature_error,
                        'recomputed_norms': checks})
    return {'summary': str(summary_path), 'summary_sha256': sha(summary_path), 'cases': results}


def audit_native(root):
    loose_path, strict_path = root / 'resolved12b/comparison.npz', root / 'strict12/comparison.npz'
    assessment_path = root / 'strict12/assessment.json'
    assessment = json.loads(assessment_path.read_text())
    with np.load(loose_path) as loose, np.load(strict_path) as strict:
        if not np.array_equal(loose['indices'], strict['indices']):
            raise ValueError('Native reference selection changed')
        ref = strict['reference5']
        q = np.asarray(assessment['selected_q'])
        source_change = float(np.linalg.norm(ref - loose['reference5']) / np.linalg.norm(ref))
        groups = {}
        for label, mask in [('moderate', q < .95), ('nearly_full', q > .95), ('pooled', q > 0)]:
            groups[label] = {name: float(np.linalg.norm((strict[name] - ref)[:, mask]) / np.linalg.norm(ref[:, mask]))
                             for name in ('candidate', 'control')}
        for name, error in groups['pooled'].items():
            if not np.isclose(error, assessment[name + '_relative_l2'], rtol=1e-12, atol=1e-15):
                raise ValueError('Native reference norm disagrees with assessment')
        if source_change >= .01 * groups['pooled']['candidate']:
            raise ValueError('Source refinement change is material relative to fitted error')
        better_cells = int(np.sum(np.linalg.norm(strict['candidate'] - ref, axis=0)
                                  < np.linalg.norm(strict['control'] - ref, axis=0)))
    node_path = root / 'strict12/reference5.npz'
    with np.load(node_path) as nodes:
        if not nodes['accepted'].all() or not np.isfinite(nodes['values']).all():
            raise ValueError('Unresolved strict BIE reference nodes')
        accepted = int(nodes['accepted'].size)
    return {'sources': {str(p.relative_to(root)): sha(p) for p in
                        (loose_path, strict_path, assessment_path, node_path)},
            'source_refinement_relative_change': source_change,
            'groups': groups, 'fitted_better_cell_count': better_cells,
            'strict_accepted_quadrature_nodes': accepted,
            'scope': 'norm arithmetic and source-refinement sensitivity on 12 selected bed-cut cells; no full-domain accuracy claim'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summaries', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-reference-root', type=Path)
    args = parser.parse_args()
    reports = [audit(path) for path in args.summaries]
    result = {'schema': 'transfer-array-independent-audit-v1', 'passed': True,
              'scope': 'retained flat analytic arrays, hashes, norms and Gaussian-quadrature means; not a CFD convergence test',
              'code_sha256': sha(Path(__file__)), 'campaigns': reports}
    if args.native_reference_root:
        result['native_reference'] = audit_native(args.native_reference_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'cases': sum(len(r['cases']) for r in reports),
                      'maximum_quadrature_relative_error': max(
                          c['independent_quadrature_relative_error'] for r in reports for c in r['cases'])}))


if __name__ == '__main__':
    main()
