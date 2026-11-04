"""Verify scientific inputs and unpack an isolated, regenerable workspace."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent


def sha(content):
    return hashlib.sha256(content).hexdigest()


def prepare(output, overwrite=False):
    output = Path(output).expanduser().resolve()
    manifest = json.loads((ROOT / 'data/source_manifest.json').read_text(encoding='utf-8'))
    planned = []
    for item in manifest['inputs']:
        stored = (ROOT / item['file']).read_bytes()
        if sha(stored) != item['sha256_stored']:
            raise ValueError(f"Stored input hash mismatch: {item['file']}")
        raw = gzip.decompress(stored) if item['file'].endswith('.gz') else stored
        if sha(raw) != item['sha256_uncompressed']:
            raise ValueError(f"Uncompressed input hash mismatch: {item['file']}")
        planned.append((item['work_path'], raw))
    with zipfile.ZipFile(ROOT / 'data/frozen_models.zip') as archive:
        expected = {i['archive_member']: i['sha256'] for i in manifest['frozen_models']}
        if set(archive.namelist()) != set(expected):
            raise ValueError('Frozen-model archive member mismatch')
        for name, digest in expected.items():
            path = Path(name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Invalid model archive path')
            raw = archive.read(name)
            if sha(raw) != digest:
                raise ValueError(f'Model hash mismatch: {name}')
            planned.append((name, raw))
    # Detect conflicts before writing any file.
    for name, raw in planned:
        destination = output / name
        if destination.exists() and destination.read_bytes() != raw and not overwrite:
            raise FileExistsError(f'{destination} differs; choose a new work directory or explicitly use --overwrite')
    for name, raw in planned:
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    (output / '.prepared.json').write_text(json.dumps({
        'inputs': len(manifest['inputs']), 'model_files': len(manifest['frozen_models']),
        'source_manifest_sha256': sha((ROOT / 'data/source_manifest.json').read_bytes())
    }, indent=2) + '\n', encoding='utf-8')
    print(f'Verified and prepared: {output}', flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'work')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    prepare(args.output_dir, args.overwrite)
