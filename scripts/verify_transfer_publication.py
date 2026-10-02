"""Verify the current transfer publication snapshot and its evidence manifest.

After intentional publication edits, --refresh updates the publication manifest.
Both scientific evidence manifests remain immutable to that operation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASES = [ROOT / ('research/releases/transfer_' + date)
            for date in ('20261001', '20261002')]
MANIFEST = ROOT / 'TRANSFER_PUBLICATION_SHA256SUMS.json'
RECEIPT = ROOT / 'manuscript/transfer_article.build.json'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def entry(path, root):
    return {'path': path.relative_to(root).as_posix(),
            'size': path.stat().st_size, 'sha256': sha(path)}


def verify(root, entries):
    for item in entries:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(item['path'])
        path = root / relative
        if not path.is_file() or path.stat().st_size != item['size'] or sha(path) != item['sha256']:
            raise AssertionError(item['path'])


def publication_files():
    names = ['CITATION.cff', 'README.md', 'DATA_AVAILABILITY.md', 'LICENSE.md',
             'RELEASE_v0.31.0.txt', 'RELEASE_v0.32.0.txt',
             'AUDITOR_REVISION_RESPONSE.txt', '.gitattributes', '.gitignore',
             'scripts/build_transfer_pdf.py', 'scripts/verify_transfer_publication.py',
             'manuscript/transfer_article.tex', 'manuscript/transfer_article.pdf',
             'manuscript/transfer_article.build.json',
             'research/releases/transfer_20261001/SHA256SUMS.json',
             'research/releases/transfer_20261002/SHA256SUMS.json']
    names += ['scripts/' + name for name in (
        'summarize_matched_boundary_extension.py', 'summarize_transfer_dynamics.py',
        'summarize_physical_wave.py', 'summarize_mixture_factorial.py',
        'plot_transfer_stress.py', 'plot_transfer_dynamics.py',
        'plot_matched_native_dynamics.py', 'summarize_auditor_revision.py',
        'plot_audited_dynamics.py')]
    source = (ROOT / 'manuscript/transfer_article.tex').read_text(encoding='utf-8')
    for relative in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', source):
        for suffix in ('.pdf', '.png', '.provenance.json'):
            names.append('manuscript/' + Path(relative).with_suffix(suffix).as_posix())
    for relative in re.findall(r'\\input\{([^}]+)\}', source):
        names.append('manuscript/' + relative)
        provenance = 'manuscript/' + Path(relative).with_suffix('.provenance.json').as_posix()
        if (ROOT / provenance).is_file():
            names.append(provenance)
    return [ROOT / name for name in sorted(set(names))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    for release in RELEASES:
        evidence = json.loads((release / 'SHA256SUMS.json').read_text(encoding='utf-8'))
        verify(release, evidence['files'])
    receipt = json.loads(RECEIPT.read_text(encoding='utf-8'))
    if receipt['pdf_sha256'] != sha(ROOT / 'manuscript/transfer_article.pdf'):
        raise AssertionError('PDF differs from the checked build receipt')
    for name, expected in receipt['input_sha256'].items():
        if sha(ROOT / name) != expected:
            raise AssertionError('Build input changed: ' + name)
    if args.refresh:
        MANIFEST.write_text(json.dumps({
            'schema': 'transfer-publication-snapshot-v2', 'version': '0.32.0',
            'scope': 'Current transfer manuscript, publication metadata and numerical evidence manifest; historical root paper.pdf is a separate study.',
            'files': [entry(p, ROOT) for p in publication_files()]}, indent=2)+'\n', encoding='utf-8')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    verify(ROOT, manifest['files'])
    if {item['path'] for item in manifest['files']} != {p.relative_to(ROOT).as_posix() for p in publication_files()}:
        raise AssertionError('Publication allowlist differs from current source dependencies')
    print(json.dumps({'publication_files_verified': len(manifest['files']),
                      'pdf_pages': receipt['pdf_pages'], 'version': manifest['version']}))


if __name__ == '__main__':
    main()
