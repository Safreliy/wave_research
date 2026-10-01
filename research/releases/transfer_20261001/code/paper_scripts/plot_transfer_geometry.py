"""Plot the actual archived pre-adjustment wave boundary, without smoothing."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.input) as data:
        geometry = {key: data[key] for key in ('surface_x', 'surface_z', 'bottom_x', 'bottom_z')}
    sx, sz, bx, bz = (geometry[k] for k in ('surface_x', 'surface_z', 'bottom_x', 'bottom_z'))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={'width_ratios': [2.5, 1]})
    # Periodic copies are drawn as separate paths so no artificial seam is joined.
    for ax in axes:
        for offset in (-64, 0, 64):
            ax.plot(sx + offset, sz, color='#176a99', linewidth=1.2,
                    label='Free surface' if offset == 0 else None)
            ax.plot(bx + offset, bz, color='#665548', linewidth=1.1,
                    label='Bed' if offset == 0 else None)
        ax.set_xlabel('$x$', fontsize=9)
        ax.set_ylabel('$z$', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.2)
    axes[0].set_xlim(0, 64)
    axes[0].set_ylim(-1.15, 0.9)
    axes[0].set_title('Periodic domain (vertical scale enlarged)', fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    crest_x = float(sx[np.argmax(sz)])
    axes[1].set_xlim(crest_x - 0.4, crest_x + 0.65)
    axes[1].set_ylim(-0.3, 0.9)
    axes[1].set_aspect('equal')
    axes[1].set_title('Crest detail (equal scales)', fontsize=9)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.005),
               ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.105, 1, 1), w_pad=1.7)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ('pdf', 'png'):
        fig.savefig(args.output_dir / f'fixed_wave_geometry.{suffix}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    np.savez_compressed(args.output_dir / 'fixed_wave_geometry.npz', **geometry)
    (args.output_dir / 'fixed_wave_geometry.provenance.json').write_text(json.dumps({
        'input_name': args.input.name,
        'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'plot_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'geometry_archive_sha256': hashlib.sha256((args.output_dir / 'fixed_wave_geometry.npz').read_bytes()).hexdigest(),
        'pdf_sha256': hashlib.sha256((args.output_dir / 'fixed_wave_geometry.pdf').read_bytes()).hexdigest(),
        'png_sha256': hashlib.sha256((args.output_dir / 'fixed_wave_geometry.png').read_bytes()).hexdigest(),
        'scope': 'actual v63 boundary polygon before native-bed adjustment; no inferred time trajectory',
        'smoothing': False,
    }, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
