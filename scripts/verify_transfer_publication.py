"""Verify the current transfer publication snapshot and its evidence manifest.

After intentional publication edits, --refresh updates both manifests. Frozen
scientific archives may not be changed by that operation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'research/releases/transfer_20261001'
MANIFEST = ROOT / 'TRANSFER_PUBLICATION_SHA256SUMS.json'


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
             'RELEASE_v0.31.0.txt', '.gitattributes', '.gitignore',
             'scripts/build_transfer_pdf.py', 'scripts/verify_transfer_publication.py',
             'manuscript/transfer_article.tex', 'manuscript/transfer_article.pdf',
             'research/releases/transfer_20261001/SHA256SUMS.json']
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
    evidence_manifest = RELEASE / 'SHA256SUMS.json'
    if args.refresh:
        old = json.loads(evidence_manifest.read_text(encoding='utf-8'))
        verify(RELEASE, [item for item in old['files'] if item['path'].endswith('.tar.gz')])
        files = [p for p in sorted(RELEASE.rglob('*'))
                 if p.is_file() and p != evidence_manifest and '__pycache__' not in p.parts
                 and p.suffix != '.pyc']
        evidence_manifest.write_text(json.dumps({
            'schema': 'transfer-20261001-explicit-allowlist-v1',
            'files': [entry(p, RELEASE) for p in files]}, indent=2)+'\n', encoding='utf-8')
        MANIFEST.write_text(json.dumps({
            'schema': 'transfer-publication-snapshot-v1', 'version': '0.31.0',
            'scope': 'Current transfer manuscript, publication metadata and numerical evidence manifest; historical root paper.pdf is a separate study.',
            'files': [entry(p, ROOT) for p in publication_files()]}, indent=2)+'\n', encoding='utf-8')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    verify(ROOT, manifest['files'])
    if {item['path'] for item in manifest['files']} != {p.relative_to(ROOT).as_posix() for p in publication_files()}:
        raise AssertionError('Publication allowlist differs from current source dependencies')
    receipt = json.loads((RELEASE / 'results/manuscript_build_20261002.json').read_text())
    if receipt['pdf_sha256'] != sha(ROOT / 'manuscript/transfer_article.pdf'):
        raise AssertionError('PDF differs from the checked build receipt')
    for name, expected in receipt['input_sha256'].items():
        if sha(ROOT / name) != expected:
            raise AssertionError('Build input changed: ' + name)
    print(json.dumps({'publication_files_verified': len(manifest['files']),
                      'pdf_pages': receipt['pdf_pages'], 'version': manifest['version']}))


if __name__ == '__main__':
    main()
