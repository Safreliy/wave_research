"""Generate the complete nonzero-time physical-wave comparison table."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    lines = [r'\begin{table}[htbp]\centering\small',
             r'\caption{Depth-error RMS against the independent linear two-fluid wave,',
             r'normalised by the fixed analytic wave-amplitude RMS. All recorded',
             r'nonzero times are included. The last column is the relative reduction',
             r'from Bilinear (B) to Matched (M) transfer.}',
             r'\label{tab:physical-wave}',
             r'\begin{tabular}{rrrrr}\toprule',
             r'$t$ & Exact input & Matched (M) & Bilinear (B) & Reduction (\%)\\\midrule']
    reductions = []
    for row in data['series']:
        if row['time'] <= 0:
            continue
        values = [row['results'][method]['physical_H_error_normalized_fixed_amplitude']
                  for method in ('phase_exact_input', 'matched', 'practical')]
        reduction = 100 * (1 - values[1]/values[2])
        reductions.append(reduction)
        fields = [f"{row['time']:.5f}"] + [f'{v:.8f}' for v in values] + [f'{reduction:.2f}']
        lines.append(' & '.join(fields) + r'\\')
    lines += [r'\bottomrule\end{tabular}', r'\end{table}']
    args.output.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    report = {'source_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
              'native_archive_sha256': data['archive_sha256'],
              'nonzero_rows': len(reductions), 'reduction_percent': reductions,
              'all_recorded_nonzero_times_favour_matched': all(x > 0 for x in reductions)}
    args.output.with_suffix('.provenance.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
