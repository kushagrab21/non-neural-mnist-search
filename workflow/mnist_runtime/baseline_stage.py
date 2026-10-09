"""Freeze the existing development CNN; never train or open final datasets.

Hash checks establish identity, not a proof of historical training provenance.
The architecture is extracted from the already reviewed training source.
"""
import ast
import hashlib
from pathlib import Path

from .common import Blocked, atomic_write, canonical, digest, strict_json

MANIFEST_SHA = '44b9835b5b6c2bfe227a2059c93a5c1ad9e23c0ad36cd711f0169f559abffa43'
DEVELOPMENT_FILES = (
    'development/train_images.npy', 'development/train_labels.npy',
    'development/folds.npy', 'development/folds.json',
    'baselines/cnn/train_cnn.py', 'baselines/cnn/results/CNN_BASELINE.json',
    'baselines/cnn/results/cnn_fold0.pt',
)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stage(inputs, output, image_id):
    inputs, output = Path(inputs).resolve(), Path(output)
    if output.exists():
        raise Blocked('stage destination exists; preserve previous evidence')
    manifest_bytes = (inputs / 'MANIFEST_repair_v4.json').read_bytes()
    if digest(manifest_bytes) != MANIFEST_SHA:
        raise Blocked('development manifest identity changed')
    manifest = strict_json(manifest_bytes)
    verified = {}
    for name in DEVELOPMENT_FILES:
        path = inputs / name
        if path.is_symlink() or not path.is_file() or inputs not in path.resolve().parents:
            raise Blocked('development input is not a regular contained file')
        actual = {'sha256': file_hash(path), 'bytes': path.stat().st_size}
        if actual != manifest['files'][name]:
            raise Blocked('development integrity mismatch: ' + name)
        verified[name] = actual
    text = (inputs / 'baselines/cnn/train_cnn.py').read_text()
    classes = [n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == 'SmallCNN']
    if len(classes) != 1:
        raise Blocked('unique SmallCNN architecture missing')
    source = ('import torch\nimport torch.nn as nn\n\n' + ast.get_source_segment(text, classes[0]) + '\n').encode()
    weights = (inputs / 'baselines/cnn/results/cnn_fold0.pt').read_bytes()
    if digest(weights) != verified['baselines/cnn/results/cnn_fold0.pt']['sha256']:
        raise Blocked('checkpoint changed during staging')
    if digest(text.encode()) != verified['baselines/cnn/train_cnn.py']['sha256']:
        raise Blocked('training source changed during staging')
    record = {
        'format': 'development-cnn-stage/1', 'image_id': image_id,
        'source_sha256': digest(source), 'weights_sha256': digest(weights),
        'development_manifest_sha256': MANIFEST_SHA, 'verified_inputs': verified,
        'checkpoint': 'cnn_fold0.pt', 'preprocessing': 'float32(uint8)/255; reshape(1,1,28,28)',
        'training_scope_from_source': 'folds1to5; fold0 excluded; no learned preprocessing',
        'historical_fit_independently_proven': False, 'fresh_training': False,
        'final_data_access': False,
    }
    output.mkdir(parents=True)
    for name, data in [('architecture.py', source), ('weights.pt', weights), ('STAGE.json', canonical(record) + b'\n')]:
        atomic_write(output / name, data)
        (output / name).chmod(0o444)
    return validate_stage(output)


def validate_stage(folder):
    folder = Path(folder)
    for name in ('STAGE.json', 'architecture.py', 'weights.pt'):
        p = folder / name
        if p.is_symlink() or not p.is_file():
            raise Blocked('stage requires regular source/checkpoint/manifest files')
    record = strict_json((folder / 'STAGE.json').read_bytes())
    if record.get('format') != 'development-cnn-stage/1':
        raise Blocked('unknown baseline stage')
    for name, key in [('architecture.py', 'source_sha256'), ('weights.pt', 'weights_sha256')]:
        if file_hash(folder / name) != record.get(key):
            raise Blocked('staged baseline bytes changed: ' + name)
    return record
