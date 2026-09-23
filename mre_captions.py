"""Experiment 1A: image-only offline captions; no training/model imports here."""
import hashlib
import json
import random
from pathlib import Path

from mre_protocol import _read_source

DEFAULT_MODEL = "Salesforce/blip-image-captioning-base"
SUFFIX = " The image caption is: "


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_images(source_dir, source_files, position_format, image_field="img_id"):
    """Reuse the exact relation parser and its one-based array/line sample IDs."""
    root = Path(source_dir).resolve()
    mapping, rows, sources = {}, {}, {}
    stems = set()
    for filename in source_files:
        path = (root / filename).resolve()
        if path.parent != root or path.stem in stems:
            raise ValueError("source files must be unique files directly inside source_dir")
        stems.add(path.stem)
        retained, _, info = _read_source(path, position_format, mapping, image_field)
        rows.update((row["id"], row) for row in retained)
        sources[path.stem] = info["sha256"]
    if not rows:
        raise ValueError("No retained relation samples")
    for sample_id, image_id in mapping.items():
        if not isinstance(image_id, str) or not image_id.strip():
            raise ValueError("{}: missing/non-string image field {!r}".format(sample_id, image_field))
    return rows, mapping, sources


def image_paths(mapping, image_root):
    """Use img_id verbatim; never add .jpg or guess a recursive file match."""
    root = Path(image_root).resolve()
    paths, missing = {}, []
    for image_id in sorted(set(mapping.values())):
        path = (root / image_id).resolve()
        if path == root or root not in path.parents:
            raise ValueError("Image ID escapes image_root: {!r}".format(image_id))
        if not path.is_file():
            missing.append(image_id)
        paths[image_id] = path
    if missing:
        raise FileNotFoundError("{} images missing under {}. First 10: {}".format(len(missing), root, missing[:10]))
    return paths


def validate_caption(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Caption must be nonempty text; no silent fallback")
    return " ".join(text.split())


def source_binding(sources, mapping, image_field):
    return {"sources_sha256": sources, "image_field": image_field,
            "sample_images_sha256": hashlib.sha256(json.dumps(mapping, sort_keys=True,
                ensure_ascii=False).encode("utf-8")).hexdigest()}


def read_cache(path):
    cache = json.loads(Path(path).read_text(encoding="utf-8"))
    if cache.get("schema") != 1 or not isinstance(cache.get("entries"), dict):
        raise ValueError("Unsupported caption cache")
    return cache


def generate_cache(paths, cache_path, model, revision="main", max_new_tokens=40,
                   device="cpu", captioner_factory=None, binding=None):
    """Resume per-image, atomically persisting each completed image.

    The generator is loaded lazily only if a caption is absent. One writer per
    cache; run preprocessing once, then share its read-only cache across seeds.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    spec = {"model": str(model), "revision": revision, "max_new_tokens": max_new_tokens,
            "do_sample": False, "num_beams": 1, "input": "image_only", "implementation": "blip-v1"}
    if binding is not None:
        spec["source_binding"] = binding
    local = Path(model)
    if local.is_dir():
        # Weights stay outside Git; fingerprints prevent stale local-model reuse.
        spec["local_files_sha256"] = {p.relative_to(local).as_posix(): digest(p)
            for p in sorted(local.rglob("*")) if p.is_file() and not any(
                part.startswith(".") for part in p.relative_to(local).parts)}
    path = Path(cache_path)
    cache = read_cache(path) if path.exists() else {"schema": 1, "generator": spec, "entries": {}}
    if cache["generator"] != spec:
        raise ValueError("Caption model/generation config changed; use a NEW cache path")
    generator, generated, hits = None, 0, 0
    for index, (image_id, image_path) in enumerate(paths.items(), 1):
        fingerprint = digest(image_path)
        previous = cache["entries"].get(image_id)
        if previous is not None:
            if previous.get("image_sha256") != fingerprint:
                raise ValueError("Image changed for {}: use a NEW cache path".format(image_id))
            validate_caption(previous.get("caption"))
            hits += 1
            continue
        if generator is None:
            generator = (captioner_factory or BlipCaptioner)(model, revision, device, max_new_tokens)
            provenance = getattr(generator, "provenance", {})
            previous_revision = cache.get("runtime", {}).get("resolved_revision")
            if previous_revision and previous_revision != provenance.get("resolved_revision"):
                raise ValueError("Resolved model revision changed; use the original revision or a NEW cache path")
            cache["runtime"] = provenance
        caption = validate_caption(generator(image_path))
        cache["entries"][image_id] = {"caption": caption, "image_sha256": fingerprint}
        write_json_atomic(path, cache)
        generated += 1
        print("Caption {}/{}: {} (saved)".format(index, len(paths), image_id), flush=True)
    return {"unique_images": len(paths), "generated": generated, "cache_hits": hits}


class BlipCaptioner:
    """Isolated offline tool only; NOT imported/loaded by run_mre.py."""
    def __init__(self, model, revision, device, max_new_tokens):
        import torch
        import transformers
        from transformers import BlipProcessor, BlipForConditionalGeneration
        self.torch, self.device, self.max_new_tokens = torch, device, max_new_tokens
        options = {"local_files_only": Path(model).is_dir(), "revision": revision}
        self.processor = BlipProcessor.from_pretrained(str(model), **options)
        self.model = BlipForConditionalGeneration.from_pretrained(str(model), **options).to(device).eval()
        self.provenance = {"torch": torch.__version__, "transformers": transformers.__version__,
                           "resolved_revision": getattr(self.model.config, "_commit_hash", None),
                           "device": device}

    def __call__(self, path):
        from PIL import Image, ImageOps
        with Image.open(path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            tokens = self.model.generate(**inputs, do_sample=False, num_beams=1,
                                         max_new_tokens=self.max_new_tokens)
        return self.processor.decode(tokens[0], skip_special_tokens=True)


def print_examples(records, seed):
    # No global Python/NumPy/Torch RNG changes from previews.
    for sample_id in random.Random(seed).sample(sorted(records), min(3, len(records))):
        row = records[sample_id]
        print("Sample: {}\nOriginal text: {}\nImage caption: {}\nAugmented text: {}".format(
            sample_id, row["original_text"], row["caption"] or "not used", row["text"]), flush=True)


def snapshot_inputs(config, run_dir, manifest):
    """Snapshot caption inputs separately; keep original split files byte-identical."""
    enabled = config.get("use_image_caption", False)
    print("Experiment: Image Caption Augmentation", flush=True)
    print("use_image_caption = {}".format(enabled), flush=True)
    print("caption model = {}".format(config.get("caption_model") if enabled else "not used"), flush=True)
    if not enabled:
        print("number of captions = 0\nmissing captions = 0 (not required)", flush=True)
        examples = {}
        for split in ("train_labeled", "validation", "test"):
            path = Path(run_dir) / "data" / manifest["files"][split]
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                examples[row["id"]] = {"original_text": row["text"], "caption": "", "text": row["text"]}
        print_examples(examples, config["seed"])
        return
    if config["task_type"] != "relation":
        raise ValueError("Experiment 1A currently supports MRE relation samples only")
    if not config.get("caption_path"):
        raise ValueError("--use-image-caption requires --caption-path (offline cache)")
    rows, mapping, sources = source_images(config["source_dir"], config["source_files"],
                                          config["position_format"], config["caption_image_field"])
    if sources != {name: info["sha256"] for name, info in manifest["sources"].items()}:
        raise ValueError("Source files changed after dataset preparation")
    cache_path = Path(config["caption_path"])
    cache = read_cache(cache_path)
    if cache["generator"]["model"] != config["caption_model"]:
        raise ValueError("--caption-model differs from the cached generator model")
    if cache["generator"].get("source_binding") != source_binding(sources, mapping, config["caption_image_field"]):
        raise ValueError("Caption cache belongs to different source files/image mapping; regenerate with prepare_image_captions.py")
    missing = sorted(set(mapping.values()) - set(cache["entries"]))
    print("number of captions = {}\nmissing captions = {}".format(
        len(set(mapping.values())) - len(missing), len(missing)), flush=True)
    if missing:
        raise ValueError("Missing captions for {} images; first 10: {}".format(len(missing), missing[:10]))
    records = {}
    for sample_id, row in rows.items():
        image_id = mapping[sample_id]
        entry = cache["entries"][image_id]
        caption = validate_caption(entry.get("caption"))
        records[sample_id] = {"original_text": row["text"], "image_id": image_id,
            "caption": caption, "text": row["text"] + SUFFIX + caption,
            "image_sha256": entry["image_sha256"]}
    print_examples(records, config["seed"])
    snapshot = Path(run_dir) / "caption_inputs.json"
    write_json_atomic(snapshot, {"schema": 1, "generator": cache["generator"],
        "runtime": cache.get("runtime", {}), "records": records})
    config["caption_inputs_sha256"] = digest(snapshot)
    config["caption_cache_sha256"] = digest(cache_path)
    config["caption_unique_images"] = len(set(mapping.values()))
    config["caption_samples"] = len(records)
    config["caption_missing"] = 0
    config["caption_llm_input"] = "original_tokenizer_visible_text"


def load_caption_texts(config, run_dir):
    if not config.get("use_image_caption", False):
        return None
    path = Path(run_dir) / "caption_inputs.json"
    if digest(path) != config.get("caption_inputs_sha256"):
        raise ValueError("Caption snapshot hash mismatch")
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def truncation_stats(tokenizer, rows, max_length, captions=None):
    original = [len(tokenizer.encode(row["text"], truncation=False)) for row in rows]
    lengths = original if captions is None else [
        len(tokenizer.encode(captions[row["id"]]["text"], truncation=False)) for row in rows]
    stats = {"total": len(rows), "truncated": sum(n > max_length for n in lengths),
             "maximum_tokens_before_truncation": max(lengths)}
    if captions is not None:
        stats.update(original_truncated=sum(n > max_length for n in original),
            newly_truncated_by_caption=sum(a <= max_length < b for a, b in zip(original, lengths)))
    return stats
