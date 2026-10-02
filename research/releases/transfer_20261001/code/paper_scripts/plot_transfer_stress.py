"""Plot actual flat-boundary stress-test records; no interpolated data points."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding='utf-8'))
    if data['schema'] != 'flat-harmonic-sliver-transfer-v1':
        raise ValueError('Unsupported input schema')
    fractions = sorted({row['surface_liquid_fraction'] for row in data['rows']})
    fig, axes = plt.subplots(1, len(fractions), figsize=(7.2, 2.8), sharey=True)
    series = [
        ('baseline_relative_l2_cut', 'Adjacent MAC-face average', '#b04b38', 's', '--'),
        ('fitted_relative_l2_cut', 'Shared-edge liquid moments', '#176a99', 'o', '-'),
        ('centroid_oracle_relative_l2_cut', 'Exact-field centroid', '#303030', '^', ':'),
    ]
    for ax, fraction in zip(axes, fractions):
        rows = sorted((r for r in data['rows'] if r['surface_liquid_fraction'] == fraction), key=lambda r: r['nx'])
        x = [r['nx'] for r in rows]
        for key, label, color, marker, style in series:
            ax.loglog(x, [r[key] for r in rows], color=color, marker=marker, linestyle=style, markersize=4, linewidth=1.2, label=label)
        ax.set_title(r'$q_{\mathrm{surface}}=' + f'{fraction:g}' + '$', fontsize=10)
        ax.set_xticks(x, labels=[str(n) for n in x], fontsize=8)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis='y', labelsize=8)
        ax.set_xlabel('Horizontal cells', fontsize=9)
        ax.grid(True, which='major', alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel('Cut-cell relative velocity error', fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ('pdf', 'png'):
        fig.savefig(args.output_dir / f'flat_sliver_comparison.{suffix}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    (args.output_dir / 'flat_sliver_comparison.provenance.json').write_text(json.dumps({
        'input_name': args.input.name,
        'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'plot_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'pdf_sha256': hashlib.sha256((args.output_dir / 'flat_sliver_comparison.pdf').read_bytes()).hexdigest(),
        'png_sha256': hashlib.sha256((args.output_dir / 'flat_sliver_comparison.png').read_bytes()).hexdigest(),
        'metric': 'unweighted relative Euclidean error over both surface and bed cut rows',
        'scope': 'flat analytic field, varied top liquid fraction; oracle has exact interior information',
    }, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
