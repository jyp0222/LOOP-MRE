"""Server entry point. Paths/defaults live in mre_config.py, not shell exports."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import mre_config as defaults
from mre_prompts import prompt_settings


def experiment_config(args):
    names = ('seed', 'position_format', 'pretrain_epochs', 'train_epochs', 'patience',
             'labeled_batch_size', 'train_batch_size', 'eval_batch_size', 'max_length',
             'lr_pretrain', 'lr', 'warmup_proportion', 'grad_clip', 'temperature',
             'ce_weight', 'topk', 'update_every', 'query_pool_size', 'kmeans_n_init', 'name_clusters',
             'view_strategy', 'rtr_prob', 'experiment_variant')
    config = {name: getattr(defaults, name.upper()) for name in names}
    config.update(seed=args.seed, source_dir=str(args.source_dir), bert_model=str(args.bert_model),
                  source_files=list(args.source_files), expected_classes=args.expected_classes,
                  position_format=args.position_format, experiment_variant=args.experiment_variant,
                  tokenizer=str(args.tokenizer), llm_enabled=not args.no_llm,
                  model=args.model, reasoning_effort=args.reasoning_effort,
                  api_base=defaults.API_BASE, max_output_tokens=defaults.MAX_OUTPUT_TOKENS,
                  naming_model=defaults.NAMING_MODEL_NAME, llm_snapshot_verified=False,
                  task_type=args.task_type, prompt_version=prompt_settings(args.task_type)[0],
                  checkpoint_selection='last_epoch', evaluation='test_kmeans',
                  request_timeout=defaults.REQUEST_TIMEOUT, max_retries=defaults.MAX_RETRIES,
                  max_requests=args.max_requests, max_queries_per_refresh=None)
    config.update(use_image_caption=args.use_image_caption,
                  caption_path=str(args.caption_path) if args.caption_path else None,
                  caption_model=args.caption_model, caption_image_field=args.caption_image_field)
    if args.smoke:
        # Keep the normal ranking pools. Cap actual eligible anchors AFTER
        # intersection; top-5 vs top-5 can have an empty intersection.
        config.update(pretrain_epochs=2, train_epochs=2, max_queries_per_refresh=5, name_clusters=False)
    if args.name_clusters:
        config['name_clusters'] = True
    if args.no_name_clusters or args.no_llm:
        config['name_clusters'] = False
    if config['view_strategy'] not in ('rtr', 'none') or not 0 <= config['rtr_prob'] <= 1:
        raise ValueError('Invalid view augmentation settings')
    for name in ('pretrain_epochs', 'train_epochs', 'patience', 'labeled_batch_size',
                 'train_batch_size', 'eval_batch_size', 'max_length', 'topk', 'update_every', 'kmeans_n_init'):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError('{} must be a positive integer'.format(name))
    if (not 0 <= config['seed'] <= 2**32 - 1 or type(config['query_pool_size']) is not int
            or config['query_pool_size'] < 0):
        raise ValueError('Invalid seed or query_pool_size')
    for name in ('lr_pretrain', 'lr', 'grad_clip', 'temperature'):
        if not 0 < config[name] < float('inf'):
            raise ValueError('{} must be positive and finite'.format(name))
    if not 0 <= config['warmup_proportion'] < 1 or not 0 <= config['ce_weight'] < float('inf'):
        raise ValueError('Invalid warmup_proportion or ce_weight')
    return config


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=defaults.SOURCE_DIR)
    parser.add_argument('--task-type', choices=['relation', 'entity_type'],
                        default=getattr(defaults, 'TASK_TYPE', 'relation'))
    parser.add_argument('--source-files', nargs='+',
                        default=getattr(defaults, 'SOURCE_FILES', ('train.txt', 'test.txt')),
                        help='ordered source files to pool before the discovery split')
    parser.add_argument('--expected-classes', type=int, default=getattr(defaults, 'EXPECTED_CLASSES', 80),
                        help='expected class count AFTER filtering None/Other/none/NA')
    parser.add_argument('--position-format', choices=['indices', 'half_open'], default=defaults.POSITION_FORMAT)
    parser.add_argument('--experiment-variant', default=defaults.EXPERIMENT_VARIANT)
    parser.add_argument('--bert-model', type=Path, default=defaults.BERT_MODEL)
    parser.add_argument('--tokenizer', type=Path, default=defaults.TOKENIZER)
    parser.add_argument('--output-root', type=Path, default=defaults.OUTPUT_ROOT)
    parser.add_argument('--seed', type=int, default=defaults.SEED)
    parser.add_argument('--model', choices=['gpt-3.5-turbo-0301', 'gpt-3.5-turbo', 'gpt-3.5-turbo-0125', 'gpt-3.5-turbo-1106'],
                        default=defaults.MODEL_NAME)
    parser.add_argument('--reasoning-effort', choices=['none'], default=defaults.REASONING_EFFORT,
                        help='legacy compatibility only; GPT-3.5 has no reasoning mode')
    parser.add_argument('--api-key-file', type=Path, default=defaults.API_KEY_FILE)
    parser.add_argument('--max-requests', type=int, default=defaults.MAX_REQUESTS)
    parser.add_argument('--no-llm', action='store_true', help='same-protocol baseline, zero API calls')
    captions = parser.add_mutually_exclusive_group()
    captions.add_argument('--use-image-caption', '--use_image_caption', action='store_true',
                          default=getattr(defaults, 'USE_IMAGE_CAPTION', False))
    captions.add_argument('--no-image-caption', dest='use_image_caption', action='store_false')
    parser.add_argument('--caption-path', '--caption_path', type=Path,
                        default=getattr(defaults, 'CAPTION_PATH', None))
    parser.add_argument('--caption-model', '--caption_model',
                        default=getattr(defaults, 'CAPTION_MODEL', 'Salesforce/blip-image-captioning-base'))
    parser.add_argument('--caption-image-field', default=getattr(defaults, 'CAPTION_IMAGE_FIELD', 'img_id'))
    parser.add_argument('--smoke', action='store_true', help='2+2 epochs; at most 5 queried anchors per refresh')
    parser.add_argument('--name-clusters', action='store_true', help='extra paid naming calls after final metrics')
    parser.add_argument('--no-name-clusters', action='store_true', help='skip the optional post-test naming calls')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only', action='store_true', help='parse/audit/split only, no BERT or API')
    mode.add_argument('--check-data', action='store_true', help='also tokenize and check tensor shapes, no API')
    mode.add_argument('--check-llm', action='store_true', help='one small API query, no data/model training')
    mode.add_argument('--evaluate-run', type=Path, help='reevaluate a saved run using its recorded evaluation protocol; no API')
    return parser


def create_client(args, run_dir=None):
    from llm_client import LLMClient
    return LLMClient(
        model=args.model, naming_model=defaults.NAMING_MODEL_NAME,
        reasoning_effort=args.reasoning_effort, api_key_file=args.api_key_file,
        base_url=defaults.API_BASE, max_output_tokens=defaults.MAX_OUTPUT_TOKENS,
        timeout=defaults.REQUEST_TIMEOUT, max_retries=defaults.MAX_RETRIES,
        max_requests=args.max_requests,
        task_type=args.task_type,
        cache_path=run_dir / 'llm_cache.jsonl' if run_dir else None,
        log_path=run_dir / 'llm_calls.jsonl' if run_dir else None,
    )


def runtime_info():
    from importlib import metadata
    versions = {}
    for package in ('torch', 'transformers', 'numpy', 'scipy', 'scikit-learn', 'requests', 'faiss-cpu', 'faiss-gpu'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent,
                                         stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=Path(__file__).parent,
                                             stderr=subprocess.DEVNULL, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    root = Path(__file__).parent
    code_files = list(root.glob('*.py')) + list((root / 'utils').glob('*.py'))
    source_hashes = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(code_files)}
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': versions, 'git_commit': commit, 'git_dirty': dirty, 'source_sha256': source_hashes}


def load_data(config, run_dir, tokenizer_path=None):
    fingerprint = hashlib.sha256((run_dir / 'data' / 'manifest.json').read_bytes()).hexdigest()
    if config.get('manifest_sha256') != fingerprint:
        raise ValueError('Dataset manifest does not match this experiment config')
    from transformers import AutoTokenizer
    from mre_data import MREData
    from mre_captions import load_caption_texts, truncation_stats
    captions = load_caption_texts(config, run_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path or config['tokenizer']), local_files_only=True)
    data = MREData(run_dir / 'data', tokenizer, max_length=config['max_length'],
                   labeled_batch_size=config['labeled_batch_size'], train_batch_size=config['train_batch_size'],
                   eval_batch_size=config['eval_batch_size'], seed=config['seed'], caption_records=captions)
    # Caption affects BERT only. GPT retains the baseline original tokenized text.
    truncation = {}
    for name, rows in (('train_labeled', data.labeled_records), ('train_unlabeled', data.unlabeled_records),
                       ('validation', data.validation_records)):
        truncation[name] = truncation_stats(tokenizer, rows, config['max_length'], captions)
    if captions is not None:
        truncation['test'] = dict(truncation['train_unlabeled'])
        truncation['unique_samples'] = truncation_stats(tokenizer,
            data.semi_records + data.validation_records, config['max_length'], captions)
    (run_dir / 'tokenization_audit.json').write_text(json.dumps(truncation, indent=2) + '\n', encoding='utf-8')
    print('Tokenization audit: {}'.format(truncation), flush=True)
    return tokenizer, data


def evaluate_saved_run(run_dir):
    import numpy as np
    import torch
    from model import CLBert
    from mre_trainer import cluster_score_loader, score_loader, write_json
    config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    _, data = load_data(config, run_dir, run_dir / 'tokenizer')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CLBert(str(run_dir / 'backbone'), device, data.n_base).to(device)
    original = config.get('evaluation') == 'test_kmeans'
    checkpoint = torch.load(run_dir / ('last_model.pt' if original else 'best_model.pt'), map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    if original:
        metrics, predictions, _ = cluster_score_loader(model, data.test_loader, device,
            data.n_base, data.n_total, config['seed'], config['kmeans_n_init'])
    else:
        metrics, predictions = score_loader(model, data.test_loader, checkpoint['centers'].numpy(), device,
                                            data.n_base, data.n_total)
    saved_rows = [json.loads(line) for line in
                  (run_dir / 'predictions.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if ([(row['id'], row['label']) for row in saved_rows]
            != [(row['id'], row['label']) for row in data.test_records]):
        raise ValueError('Saved predictions and current evaluation sample IDs/labels differ')
    prior = [row['prediction'] for row in saved_rows]
    report = {'metrics': metrics, 'predictions_identical': bool(np.array_equal(prior, predictions))}
    write_json(run_dir / 'reevaluation.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.no_llm and args.check_llm:
        parser.error('--no-llm cannot be combined with --check-llm')
    if args.name_clusters and args.no_name_clusters:
        parser.error('--name-clusters and --no-name-clusters are mutually exclusive')
    if args.evaluate_run:
        evaluate_saved_run(args.evaluate_run.resolve())
        return
    if args.check_llm:
        client = create_client(args)
        if args.task_type == 'entity_type':
            query = 'Entity: Alice. Sentence: [ENTITY] Alice [/ENTITY] visited Paris.'
            choices = ['Entity: Bob. Sentence: [ENTITY] Bob [/ENTITY] visited Rome.',
                       'Entity: Paris. Sentence: Alice visited [ENTITY] Paris [/ENTITY].']
        else:
            query = 'Head: Alice. Tail: Paris. Sentence: Alice was born in Paris.'
            choices = ['Head: Bob. Tail: Rome. Sentence: Bob was born in Rome.',
                       'Head: Carol. Tail: London. Sentence: Carol works in London.']
        answer = client.choose_neighbor(query, choices)
        if client.last_query_fallback:
            raise RuntimeError('API check failed: Choice 1 was only the upstream error fallback; no valid LLM answer')
        print('Requested model: {}; response model: {}; answer: Choice {}'.format(
            client.model, client.last_response_model, answer + 1))
        print('Prompt version: {}'.format(prompt_settings(args.task_type)[0]))
        print('Provider snapshot provenance: UNVERIFIED (model field does not verify backend weights).')
        return
    config = experiment_config(args)
    from mre_protocol import prepare_dataset
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir = args.output_root.resolve() / '{}_seed{}_{}'.format(
        config['experiment_variant'], config['seed'], stamp)
    # prepare_dataset creates this unique run's immutable data directory.
    manifest = prepare_dataset(args.source_dir, run_dir / 'data', config['seed'], config['position_format'],
                               expected_total=config['expected_classes'], source_files=config['source_files'],
                               task_type=config['task_type'])
    print('Run directory: {}'.format(run_dir), flush=True)
    print('MRE random-half classes: {} base / {} novel (seed={})'.format(
        manifest['n_base'], manifest['n_novel'], config['seed']))
    print('Base:', manifest['base_classes'])
    print('Novel:', manifest['novel_classes'])
    print('Split counts: {}'.format({name: split['count'] for name, split in manifest['splits'].items()}))
    print('Source audit: {}'.format({name: {key: info[key] for key in
          ('retained_records', 'filtered_records', 'filtered_relations')}
          for name, info in manifest['sources'].items()}), flush=True)
    print('Data audit: {}'.format(manifest['audit']), flush=True)
    print('Experiment variant: {}; k={}; batch={}/{}/{}; view={}; last-epoch selection; test KMeans'.format(
        config['experiment_variant'], config['topk'], config['labeled_batch_size'],
        config['train_batch_size'], config['eval_batch_size'], config['view_strategy']), flush=True)
    from mre_captions import snapshot_inputs
    snapshot_inputs(config, run_dir, manifest)
    client = None
    if not (args.no_llm or args.prepare_only or args.check_data):
        client = create_client(args, run_dir)
        config['reasoning_effort'] = client.reasoning_effort
    config['data_protocol'] = manifest['protocol']
    config['runtime'] = runtime_info()
    config['manifest_sha256'] = hashlib.sha256((run_dir / 'data' / 'manifest.json').read_bytes()).hexdigest()
    (run_dir / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.prepare_only:
        return
    if not args.bert_model.is_dir():
        raise FileNotFoundError('Edit BERT_MODEL in mre_config.py: {}'.format(args.bert_model))
    tokenizer, data = load_data(config, run_dir)
    if args.check_data:
        batch = next(iter(data.semi_loader))
        print('Tensor shapes:', [tuple(tensor.shape) for tensor in batch])
        print('Data check passed; no API requests or training performed.')
        return
    from mre_trainer import MRETrainer
    trainer = MRETrainer(config, data, tokenizer, run_dir, client)
    try:
        trainer.train()
        if args.smoke and client and client.requests_made == 0 and client.cache_hits == 0:
            print('Smoke coverage incomplete: training finished, but no LLM neighbor query was exercised.',
                  flush=True)
        if client and client.fallback_count:
            print('LLM fallback warning: {} queries used upstream Choice 1 fallback; inspect llm_calls.jsonl.'.format(
                client.fallback_count), flush=True)
    finally:
        if client:
            (run_dir / 'llm_summary.json').write_text(json.dumps({
                'http_attempts': client.requests_made, 'cache_hits': client.cache_hits,
                'requested_model': client.model, 'reasoning_effort': client.reasoning_effort,
                'naming_model': client.naming_model, 'fallback_count': client.fallback_count,
                'prompt_version': config['prompt_version'],
                'response_models': sorted(client.response_models),
                'model_mismatch_count': client.model_mismatch_count,
                'snapshot_verified': False,
            }, indent=2) + '\n', encoding='utf-8')
    print('Finished! {}'.format(run_dir), flush=True)


if __name__ == '__main__':
    main()
