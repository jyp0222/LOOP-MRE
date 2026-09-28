"""Experiment 1B: frozen ViT cache and final-test-only feature concatenation.

No vision model is imported or loaded by training. Alignment uses the same
source parser/IDs as the current caption adapter, but no captions are generated
or consumed. Cache and run snapshots contain no ground-truth labels.
"""
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from mre_captions import digest, image_paths, source_binding, source_images, write_json_atomic

DEFAULT_IMAGE_MODEL = 'google/vit-base-patch16-224-in21k'
REPRESENTATION = 'last_hidden_state[:, 0, :]'


def validate_features(features, rows=None, width=None):
    array = np.asarray(features)
    if (array.ndim != 2 or min(array.shape) < 1 or array.dtype.kind != 'f'
            or not np.isfinite(array).all() or np.any(np.linalg.norm(array, axis=1) <= 1e-12)):
        raise ValueError('Features must be a finite, nonzero floating-point matrix')
    if rows is not None and len(array) != rows:
        raise ValueError('Image/text feature row count mismatch')
    if width is not None and array.shape[1] != width:
        raise ValueError('Image feature dimension differs from cache metadata')
    return np.ascontiguousarray(array, dtype=np.float32)


def fuse_features(text_features, image_features):
    """Return [N, d_text + d_image]; unit-normalize each modality, no projection.

    The concatenation has norm sqrt(2), deliberately NOT normalized again.
    """
    import torch
    import torch.nn.functional as F
    text = validate_features(text_features)
    image = validate_features(image_features, rows=len(text))
    fused = torch.cat((F.normalize(torch.from_numpy(text), dim=-1),
                       F.normalize(torch.from_numpy(image), dim=-1)), dim=-1)
    return fused.numpy()


class FrozenViTEncoder:
    """HF ViT's final normalized CLS, before pooler/classifier: [B, hidden_size]."""
    def __init__(self, model_path, revision='main', device='cpu'):
        import torch
        import transformers
        from transformers import ViTModel
        # ViTFeatureExtractor works with the original older training environment.
        try:
            from transformers import ViTImageProcessor as Processor
        except ImportError:
            from transformers import ViTFeatureExtractor as Processor
        options = {'revision': revision, 'local_files_only': Path(model_path).is_dir()}
        self.processor = Processor.from_pretrained(str(model_path), **options)
        self.model = ViTModel.from_pretrained(str(model_path), **options).to(device).eval()
        self.model.requires_grad_(False)
        self.torch, self.device = torch, device
        self.provenance = {'torch': torch.__version__, 'transformers': transformers.__version__,
            'resolved_revision': getattr(self.model.config, '_commit_hash', None),
            'processor': self.processor.to_dict(), 'hidden_size': self.model.config.hidden_size,
            'representation': REPRESENTATION, 'frozen': True, 'device': device}

    def __call__(self, paths):
        from PIL import Image, ImageOps
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(ImageOps.exif_transpose(image).convert('RGB'))
        inputs = self.processor(images=images, return_tensors='pt')
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with self.torch.no_grad():
            # image_feature.shape = [batch_size, config.hidden_size], usually [B, 768].
            image_feature = self.model(**inputs).last_hidden_state[:, 0, :]
        return image_feature.detach().cpu().float().numpy()


def cache_manifest(cache_dir):
    cache = json.loads((Path(cache_dir) / 'manifest.json').read_text(encoding='utf-8'))
    if cache.get('schema') != 1 or not isinstance(cache.get('entries'), dict):
        raise ValueError('Unsupported ViT feature cache')
    return cache


def _vector_path(cache_dir, image_id):
    # Image IDs may include subdirectories; never use them as cache filenames.
    filename = hashlib.sha256(image_id.encode('utf-8')).hexdigest() + '.npy'
    return Path(cache_dir) / 'vectors' / filename


def _read_vector(cache_dir, image_id, entry, width=None):
    path = _vector_path(cache_dir, image_id)
    if digest(path) != entry['feature_sha256']:
        raise ValueError('Corrupt cached image feature: {}'.format(image_id))
    vector = np.load(str(path), allow_pickle=False)
    if vector.ndim != 1:
        raise ValueError('Cached image feature must be one-dimensional')
    return validate_features(vector[None, :], width=width)[0]


def generate_feature_cache(paths, cache_dir, model_path=DEFAULT_IMAGE_MODEL, revision='main',
                           device='cpu', batch_size=16, binding=None, encoder_factory=None,
                           image_root=None):
    """Resume offline inference. Persist one vector per unique image, then manifest.

    One preprocessing writer; training only reads. Repeated generation skips all
    existing vectors without loading ViT. Changed model/images/sources fail.
    """
    if type(batch_size) is not int or batch_size < 1 or not paths:
        raise ValueError('Positive image batch size and nonempty paths required')
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    spec = {'encoder': 'vit', 'model': str(model_path), 'revision': revision,
            'representation': REPRESENTATION, 'frozen': True, 'dtype': 'float32',
            'source_binding': binding, 'implementation': 'vit-cls-v1'}
    local = Path(model_path)
    if local.is_dir():
        spec['local_files_sha256'] = {p.relative_to(local).as_posix(): digest(p)
            for p in sorted(local.rglob('*')) if p.is_file() and not any(
                part.startswith('.') for part in p.relative_to(local).parts)}
    index = directory / 'manifest.json'
    cache = cache_manifest(directory) if index.exists() else {
        'schema': 1, 'encoder': spec, 'entries': {}, 'image_root': str(Path(image_root).resolve()) if image_root else None}
    if cache['encoder'] != spec:
        raise ValueError('ViT model/source/config changed; use a NEW image feature cache')
    pending, hits = [], 0
    for image_id, path in sorted(paths.items()):
        fingerprint = digest(path)
        previous = cache['entries'].get(image_id)
        if previous is not None:
            if previous['image_sha256'] != fingerprint:
                raise ValueError('Image changed: {}; use a NEW cache'.format(image_id))
            _read_vector(directory, image_id, previous, cache.get('feature_dim'))
            hits += 1
        else:
            pending.append((image_id, path, fingerprint))
    generated = 0
    if pending:
        encoder = (encoder_factory or FrozenViTEncoder)(model_path, revision, device)
        provenance = getattr(encoder, 'provenance', {})
        previous_revision = cache.get('runtime', {}).get('resolved_revision')
        if previous_revision and previous_revision != provenance.get('resolved_revision'):
            raise ValueError('Resolved ViT revision changed; use a NEW cache or pinned revision')
        cache['runtime'] = provenance
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            features = validate_features(encoder([row[1] for row in batch]), rows=len(batch),
                                         width=cache.get('feature_dim'))
            cache['feature_dim'] = features.shape[1]
            for (image_id, _, fingerprint), vector in zip(batch, features):
                target = _vector_path(directory, image_id)
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix('.tmp')
                with temporary.open('wb') as stream:
                    np.save(stream, vector, allow_pickle=False)
                temporary.replace(target)
                cache['entries'][image_id] = {'image_sha256': fingerprint, 'feature_sha256': digest(target)}
            write_json_atomic(index, cache)
            generated += len(batch)
            print('ViT features: {}/{} generated; {} cache hits'.format(generated, len(pending), hits), flush=True)
    return {'unique_images': len(paths), 'generated': generated, 'cache_hits': hits,
            'feature_dim': cache['feature_dim']}


def snapshot_image_features(config, run_dir, manifest):
    """Copy only final-test features in manifest order. No model inference/RNG use."""
    enabled = config.get('use_image_feature', False)
    print('Experiment: ViT Feature Fusion\nuse_image_feature = {}'.format(enabled), flush=True)
    if not enabled:
        print('Image encoder: not used\nFrozen: not required\nNumber of images: 0\nMissing images: 0 (not required)', flush=True)
        return
    if config.get('use_image_caption') or not config.get('freeze_image_encoder', True):
        raise ValueError('Experiment 1B requires frozen ViT and no image caption')
    if config.get('task_type') != 'relation':
        raise ValueError('Experiment 1B currently supports MRE relation samples only')
    rows, mapping, sources = source_images(config['source_dir'], config['source_files'],
        config['position_format'], config['image_field'])
    if sources != {name: info['sha256'] for name, info in manifest['sources'].items()}:
        raise ValueError('Source files changed after dataset preparation')
    cache_dir = Path(config['image_feature_cache'])
    cache = cache_manifest(cache_dir)
    spec = cache['encoder']
    if (spec.get('encoder') != config['image_encoder'] or spec.get('model') != config['image_model_path']
            or spec.get('frozen') is not True or spec.get('representation') != REPRESENTATION
            or spec.get('source_binding') != source_binding(sources, mapping, config['image_field'])):
        raise ValueError('ViT cache model/source/image alignment differs from experiment config')
    root = config.get('image_root') or cache.get('image_root')
    if not root:
        raise ValueError('Image root required: use --image-root')
    paths = image_paths(mapping, root)
    missing = sorted(set(mapping.values()) - set(cache['entries']))
    print('Image encoder: {} ({})\nFrozen: True\nNumber of images: {}\nMissing images: 0'.format(
        config['image_encoder'], config['image_model_path'], len(paths)), flush=True)
    if missing:
        raise ValueError('Missing image features: {} (first 10: {})'.format(len(missing), missing[:10]))
    test_path = Path(run_dir) / 'data' / manifest['files']['test']
    records = [json.loads(line) for line in test_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if [row['id'] for row in records] != manifest['splits']['test']['ids']:
        raise ValueError('Test sample order differs from manifest')
    unique_vectors = {}
    for image_id in sorted(set(mapping[row['id']] for row in records)):
        entry = cache['entries'][image_id]
        if digest(paths[image_id]) != entry['image_sha256']:
            raise ValueError('Image changed since offline ViT generation: {}'.format(image_id))
        unique_vectors[image_id] = _read_vector(cache_dir, image_id, entry, cache['feature_dim'])
    for row in records:
        if rows[row['id']]['text'] != row['text']:
            raise ValueError('Image source text differs from prepared sample: {}'.format(row['id']))
    features = np.stack([unique_vectors[mapping[row['id']]] for row in records])
    target = Path(run_dir) / 'image_features.npz'
    temporary = target.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, features=features,
            sample_ids=np.array([row['id'] for row in records]),
            image_ids=np.array([mapping[row['id']] for row in records]))
    temporary.replace(target)
    model_config = json.loads((Path(config['bert_model']) / 'config.json').read_text(encoding='utf-8'))
    text_dim, image_dim = model_config['hidden_size'], features.shape[1]
    print('Text feature shape (expected): {}\nImage feature shape: {}\nFused feature shape (expected): {}'.format(
        (len(records), text_dim), features.shape, (len(records), text_dim + image_dim)), flush=True)
    for row in random.Random(config['seed']).sample(records, min(3, len(records))):
        print('Sample: {}\nText: {}\nImage path: {}\nText feature shape (expected): {}\n'
              'Image feature shape: {}\nFused feature shape (expected): {}'.format(
            row['id'], row['text'], paths[mapping[row['id']]], (text_dim,), (image_dim,), (text_dim + image_dim,)), flush=True)
    write_json_atomic(Path(run_dir) / 'image_feature_audit.json', {
        'schema': 1, 'encoder': spec, 'runtime': cache.get('runtime', {}),
        'image_root': str(Path(root).resolve()), 'number_of_images': len(paths), 'missing_images': 0,
        'test_samples': len(records), 'text_dim_expected': text_dim, 'image_dim': image_dim,
        'fused_dim_expected': text_dim + image_dim, 'scope': 'final_test_kmeans_only'})
    config.update(image_features_sha256=digest(target),
        image_feature_audit_sha256=digest(Path(run_dir) / 'image_feature_audit.json'),
        image_cache_manifest_sha256=digest(cache_dir / 'manifest.json'),
        image_feature_dim=image_dim, image_feature_scope='final_test_kmeans_only', image_missing=0)


def load_image_features(config, run_dir, records):
    """No images, original cache, ViT or network required to reevaluate a saved run."""
    if not config.get('use_image_feature', False):
        return None
    path = Path(run_dir) / 'image_features.npz'
    if digest(path) != config.get('image_features_sha256'):
        raise ValueError('Run image feature snapshot hash mismatch')
    audit = Path(run_dir) / 'image_feature_audit.json'
    if digest(audit) != config.get('image_feature_audit_sha256'):
        raise ValueError('Run image feature audit hash mismatch')
    with np.load(str(path), allow_pickle=False) as snapshot:
        if snapshot['sample_ids'].tolist() != [row['id'] for row in records]:
            raise ValueError('Image features/test sample ID order mismatch')
        if snapshot['image_ids'].shape != (len(records),):
            raise ValueError('Image ID count mismatch')
        return validate_features(snapshot['features'], rows=len(records), width=config['image_feature_dim'])
