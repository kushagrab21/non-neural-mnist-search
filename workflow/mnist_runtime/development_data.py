"""Parent-owned development data; only fit folds are exported to a build.

No final-data path, key, decryptor, download or model service is used here.
"""
from io import BytesIO
from pathlib import Path
import numpy as np

from .baseline_stage import MANIFEST_SHA
from .common import Blocked, canonical, digest, strict_json, atomic_write


def split(images, labels, folds, *, seed=20261007):
    if images.dtype != np.uint8 or images.ndim != 3 or images.shape[1:] != (28, 28):
        raise Blocked('uint8 (N,28,28) images required')
    n = len(images)
    if labels.shape != (n,) or labels.dtype.kind not in 'iu' or np.any(labels > 9) or np.any(labels < 0):
        raise Blocked('integer digit labels required')
    if folds.shape != (n,) or folds.dtype.kind not in 'iu' or set(np.unique(folds).tolist()) != set(range(6)):
        raise Blocked('six identified integer development folds required')
    fit_ids, evaluation_ids = np.flatnonzero(folds != 0), np.flatnonzero(folds == 0)
    if len(fit_ids) < 1074 or not 128 <= len(evaluation_ids) <= 10000:
        raise Blocked('cohort too small/large for the frozen comparison recipe')
    # Warmups and history fillers use fit images, selected once independently of
    # candidate behavior. No evaluation labels or arrays go to the build.
    selected = np.random.Generator(np.random.PCG64(seed)).choice(fit_ids, 1074, replace=False)
    result = {'fit_images': images[fit_ids].copy(), 'fit_labels': labels[fit_ids].copy(),
              'images': images[evaluation_ids].copy(), 'labels': labels[evaluation_ids].copy(),
              'warmups': images[selected[:50]].copy(), 'fillers': images[selected[50:]].copy()}
    for value in result.values(): value.flags.writeable = False
    result['identity'] = {'fit_folds': [1, 2, 3, 4, 5], 'feedback_fold': 0,
                          'fit_n': len(fit_ids), 'feedback_n': len(evaluation_ids),
                          'fit_indices_sha256': digest(fit_ids.astype('<i8').tobytes()),
                          'feedback_indices_sha256': digest(evaluation_ids.astype('<i8').tobytes()),
                          'context_indices_sha256': digest(selected.astype('<i8').tobytes()),
                          'context_seed': seed, 'preprocessing_fit_policy': 'fit folds only',
                          'adaptive_development_feedback': True, 'untouched_holdout_claim': False}
    return result


def load(inputs):
    inputs = Path(inputs).resolve()
    raw_manifest = (inputs / 'MANIFEST_repair_v4.json').read_bytes()
    if digest(raw_manifest) != MANIFEST_SHA:
        raise Blocked('development manifest changed')
    manifest = strict_json(raw_manifest)
    arrays, identities = {}, {}
    for name in ('train_images', 'train_labels', 'folds'):
        relative = 'development/' + name + '.npy'
        path = inputs / relative
        if path.is_symlink() or not path.is_file() or inputs not in path.resolve().parents:
            raise Blocked('development input escapes its root')
        expected = manifest['files'][relative]
        if path.stat().st_size != expected['bytes'] or expected['bytes'] > 64 * 1024 * 1024:
            raise Blocked('development input size mismatch')
        raw = path.read_bytes()
        if digest(raw) != expected['sha256']:
            raise Blocked('development payload hash mismatch: ' + relative)
        arrays[name] = np.load(BytesIO(raw), allow_pickle=False)
        identities[relative] = expected
    if arrays['train_images'].shape != (60000, 28, 28) or any(int(np.sum(arrays['folds'] == i)) != 10000 for i in range(6)):
        raise Blocked('recorded MNIST development cohort/fold sizes changed')
    result = split(arrays['train_images'], arrays['train_labels'], arrays['folds'])
    result['identity'].update(manifest_sha256=MANIFEST_SHA, source_files=identities)
    return result


def export_fit(data, output):
    """The build receives exactly these two arrays, never a development root."""
    output = Path(output)
    if output.exists(): raise Blocked('fit destination exists')
    output.mkdir(parents=True)
    files = {}
    for key in ('fit_images', 'fit_labels'):
        value = data[key]
        buffer = BytesIO(); np.save(buffer, value, allow_pickle=False)
        raw = buffer.getvalue(); name = key + '.npy'
        atomic_write(output / name, raw); (output / name).chmod(0o444)
        files[name] = {'bytes': len(raw), 'sha256': digest(raw)}
    # Provenance stays in the parent, outside the container's input mount.
    return {'format': 'fit-fold-export/1', 'files': files,
            'cohort': data['identity'], 'only_folds1to5': True}


def validate_fit_export(directory, manifest):
    directory = Path(directory)
    if manifest.get('format') != 'fit-fold-export/1' or manifest.get('only_folds1to5') is not True:
        raise Blocked('identified fit-only export required')
    expected = {'fit_images.npy', 'fit_labels.npy'}
    if set(manifest.get('files', {})) != expected or {p.name for p in directory.iterdir()} != expected:
        raise Blocked('only the two fit arrays may be mounted')
    for name in expected:
        p = directory / name
        if p.is_symlink() or not p.is_file(): raise Blocked('regular fit arrays required')
        record = manifest['files'][name]
        if p.stat().st_size != record['bytes'] or digest(p.read_bytes()) != record['sha256']:
            raise Blocked('fit export changed after split')
    return manifest
