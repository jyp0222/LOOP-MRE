"""Experiment 1B: image validation and resumable frozen ViT feature generation."""
import argparse
import json
from pathlib import Path

from mre_captions import image_paths, source_binding, source_images
from mre_image_features import DEFAULT_IMAGE_MODEL, generate_feature_cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--source-files', nargs='+', default=['train.txt', 'val.txt', 'test.txt'])
    parser.add_argument('--position-format', choices=['half_open', 'indices'], default='half_open')
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--image-field', default='img_id')
    parser.add_argument('--image-encoder', '--image_encoder', choices=['vit'], default='vit')
    parser.add_argument('--image-model-path', '--image_model_path', default=DEFAULT_IMAGE_MODEL)
    parser.add_argument('--image-feature-cache', '--image_feature_cache', type=Path)
    parser.add_argument('--revision', default='main')
    parser.add_argument('--batch-size', type=int, default=16, help='offline ViT inference batch, NOT LOOP batch')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--check-images', action='store_true')
    args = parser.parse_args(argv)
    rows, mapping, sources = source_images(args.source_dir, args.source_files,
                                          args.position_format, args.image_field)
    paths = image_paths(mapping, args.image_root)
    from PIL import Image
    for image_id, path in paths.items():
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            raise ValueError('Unreadable image: {}'.format(image_id)) from exc
    print('Retained samples: {}; unique readable images: {}; missing images: 0'.format(len(rows), len(paths)))
    if args.check_images:
        return
    if args.image_feature_cache is None:
        parser.error('--image-feature-cache is required for generation')
    result = generate_feature_cache(paths, args.image_feature_cache, args.image_model_path,
        args.revision, args.device, args.batch_size,
        binding=source_binding(sources, mapping, args.image_field), image_root=args.image_root)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
