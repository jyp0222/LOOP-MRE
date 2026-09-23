"""Offline MRE image validation and resumable BLIP caption generation."""
import argparse
import json
from pathlib import Path

from mre_captions import DEFAULT_MODEL, generate_cache, image_paths, source_images, source_binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--source-files', nargs='+', default=['train.txt', 'val.txt', 'test.txt'])
    parser.add_argument('--position-format', choices=['half_open', 'indices'], default='half_open')
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--image-field', default='img_id')
    parser.add_argument('--caption-path', '--caption_path', type=Path)
    parser.add_argument('--caption-model', '--caption_model', default=DEFAULT_MODEL)
    parser.add_argument('--revision', default='main')
    parser.add_argument('--max-new-tokens', type=int, default=40)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--check-images', action='store_true', help='verify all image files; no model load')
    args = parser.parse_args()
    rows, mapping, sources = source_images(args.source_dir, args.source_files, args.position_format, args.image_field)
    paths = image_paths(mapping, args.image_root)
    # Validate decodability before generating anything, including cached images.
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
    if args.caption_path is None:
        parser.error('--caption-path is required for generation')
    print(json.dumps(generate_cache(paths, args.caption_path, args.caption_model, args.revision,
                                    args.max_new_tokens, args.device,
                                    binding=source_binding(sources, mapping, args.image_field)), indent=2))


if __name__ == '__main__':
    main()
