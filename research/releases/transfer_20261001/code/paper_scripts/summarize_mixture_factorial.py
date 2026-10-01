"""Summarise all six endpoint cells of the physical-wave comparison."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.input.read_text())
    endpoint = data['series'][-1]
    values = endpoint['results']
    key = 'physical_H_error_normalized_fixed_amplitude'
    lines = [r'\begin{table}[htbp]\centering\small',
             r'\caption{Factorial comparison at $0.1T$ against the independent',
             r'linear two-fluid wave: normalised physical depth-error RMS.',
             r'Each column uses the same phase reconstruction under both',
             r'receiver assignments; geometry, gas phase and runtime are common.}',
             r'\label{tab:mixture-factorial}',
             r'\begin{tabular}{lrrr}\toprule',
             r'Receiver assignment & Exact phase data & Matched & Practical\\\midrule']
    records = {}
    for prefix, label in [('phase', 'Liquid mean in wet cells'), ('mass', 'Density-weighted mixture')]:
        fields = [values[prefix+'_'+suffix][key] for suffix in ('exact_input', 'matched', 'practical')]
        records[prefix] = fields
        lines.append(label+' & '+' & '.join(f'{v:.8f}' for v in fields)+r'\\')
    lines += [r'\bottomrule\end{tabular}', r'\end{table}']
    args.output.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    report = {'source_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
              'native_archive_sha256': data['archive_sha256'], 'time': endpoint['time'],
              'rows_exact_matched_practical': records,
              'matched_gain_within_mixture_assignment': records['mass'][2]/records['mass'][1],
              'combined_gain_over_original_practical_assignment': records['phase'][2]/records['mass'][1],
              'combined_reduction_percent': 100*(1-records['mass'][1]/records['phase'][2])}
    args.output.with_suffix('.provenance.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
