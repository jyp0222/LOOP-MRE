"""Server entry point. Paths/defaults live in mre_config.py, not shell exports."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import mre_config as defaults


def experiment_config(args):
    names = ('seed', 'position_format', 'pretrain_epochs', 'train_epochs', 'patience',
             'labeled_batch_size', 'train_batch_size', 'eval_batch_size', 'max_length',
             'lr_pretrain', 'lr', 'warmup_proportion', 'grad_clip', 'temperature',
             'ce_weight', 'topk', 'update_every', 'query_pool_size', 'kmeans_n_init', 'name_clusters')
    config = {name: getattr(defaults, name.upper()) for name in names}
    config.update(seed=args.seed, source_dir=str(args.source_dir), bert_model=str(args.bert_model),
                  tokenizer=str(args.tokenizer), llm_enabled=not args.no_llm,
                  model=args.model, reasoning_effort=args.reasoning_effort,
                  api_base=defaults.API_BASE, max_output_tokens=defaults.MAX_OUTPUT_TOKENS,
                  request_timeout=defaults.REQUEST_TIMEOUT, max_retries=defaults.MAX_RETRIES,
                  max_requests=args.max_requests)
    if args.smoke:
        config.update(pretrain_epochs=2, train_epochs=2, query_pool_size=5, name_clusters=False)
    if args.name_clusters:
        config['name_clusters'] = True
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
    parser.add_argument('--bert-model', type=Path, default=defaults.BERT_MODEL)
    parser.add_argument('--tokenizer', type=Path, default=defaults.TOKENIZER)
    parser.add_argument('--output-root', type=Path, default=defaults.OUTPUT_ROOT)
    parser.add_argument('--seed', type=int, default=defaults.SEED)
    parser.add_argument('--model', choices=['gpt-5.6-sol', 'gpt-5.6', 'gpt-6-astra'], default=defaults.MODEL_NAME)
    parser.add_argument('--reasoning-effort', default=defaults.REASONING_EFFORT)
    parser.add_argument('--api-key-file', type=Path, default=defaults.API_KEY_FILE)
    parser.add_argument('--max-requests', type=int, default=defaults.MAX_REQUESTS)
    parser.add_argument('--no-llm', action='store_true', help='same-protocol baseline, zero API calls')
    parser.add_argument('--smoke', action='store_true', help='2+2 epochs; at most 5 queried anchors per refresh')
    parser.add_argument('--name-clusters', action='store_true', help='extra paid naming calls after final metrics')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only', action='store_true', help='parse/audit/split only, no BERT or API')
    mode.add_argument('--check-data', action='store_true', help='also tokenize and check tensor shapes, no API')
    mode.add_argument('--check-llm', action='store_true', help='one small API query, no data/model training')
    mode.add_argument('--evaluate-run', type=Path, help='reevaluate saved weights and centers, no API or refitting')
    return parser


def create_client(args, run_dir=None):
    from llm_client import LLMClient
    return LLMClient(
        model=args.model, reasoning_effort=args.reasoning_effort, api_key_file=args.api_key_file,
        base_url=defaults.API_BASE, max_output_tokens=defaults.MAX_OUTPUT_TOKENS,
        timeout=defaults.REQUEST_TIMEOUT, max_retries=defaults.MAX_RETRIES,
        max_requests=args.max_requests,
        cache_path=run_dir / 'llm_cache.jsonl' if run_dir else None,
        log_path=run_dir / 'llm_calls.jsonl' if run_dir else None,
    )


def runtime_info():
    from importlib import metadata
    versions = {}
    for package in ('torch', 'transformers', 'numpy', 'scipy', 'scikit-learn', 'requests'):
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
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path or config['tokenizer']), local_files_only=True)
    data = MREData(run_dir / 'data', tokenizer, max_length=config['max_length'],
                   labeled_batch_size=config['labeled_batch_size'], train_batch_size=config['train_batch_size'],
                   eval_batch_size=config['eval_batch_size'], seed=config['seed'])
    # Report differing input visibility: GPT sees complete text; BERT has a
    # fixed token budget. No silent claim that all entity context survived.
    truncation = {}
    for name, rows in (('train_labeled', data.labeled_records), ('train_unlabeled', data.unlabeled_records),
                       ('validation', data.validation_records)):
        lengths = [len(tokenizer.encode(row['text'], truncation=False)) for row in rows]
        truncation[name] = {'total': len(rows), 'truncated': sum(n > config['max_length'] for n in lengths),
                            'maximum_tokens_before_truncation': max(lengths)}
    (run_dir / 'tokenization_audit.json').write_text(json.dumps(truncation, indent=2) + '\n', encoding='utf-8')
    print('Tokenization audit: {}'.format(truncation), flush=True)
    return tokenizer, data


def evaluate_saved_run(run_dir):
    import numpy as np
    import torch
    from model import CLBert
    from mre_trainer import score_loader, write_json
    config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    _, data = load_data(config, run_dir, run_dir / 'tokenizer')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CLBert(str(run_dir / 'backbone'), device, data.n_base).to(device)
    checkpoint = torch.load(run_dir / 'best_model.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
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
    if args.evaluate_run:
        evaluate_saved_run(args.evaluate_run.resolve())
        return
    if args.check_llm:
        client = create_client(args)
        answer = client.choose_neighbor(
            'Head: Alice. Tail: Paris. Sentence: Alice was born in Paris.',
            ['Head: Bob. Tail: Rome. Sentence: Bob was born in Rome.',
             'Head: Carol. Tail: London. Sentence: Carol works in London.'])
        print('Model: {}; effort: {}; answer: Choice {}'.format(client.model, client.reasoning_effort, answer + 1))
        return
    config = experiment_config(args)
    from mre_protocol import prepare_dataset
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir = args.output_root.resolve() / 'mre_seed{}_{}'.format(config['seed'], stamp)
    # prepare_dataset creates this unique run's immutable data directory.
    manifest = prepare_dataset(args.source_dir, run_dir / 'data', config['seed'], config['position_format'])
    print('Run directory: {}'.format(run_dir), flush=True)
    print('Fixed classes: {} base / {} novel'.format(manifest['n_base'], manifest['n_novel']))
    print('Split counts: {}'.format({name: split['count'] for name, split in manifest['splits'].items()}))
    print('Data audit: {}'.format(manifest['audit']), flush=True)
    client = None
    if not (args.no_llm or args.prepare_only or args.check_data):
        client = create_client(args, run_dir)
        config['reasoning_effort'] = client.reasoning_effort
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
    finally:
        if client:
            (run_dir / 'llm_summary.json').write_text(json.dumps({
                'http_attempts': client.requests_made, 'cache_hits': client.cache_hits,
                'requested_model': client.model, 'reasoning_effort': client.reasoning_effort,
            }, indent=2) + '\n', encoding='utf-8')
    print('Finished! {}'.format(run_dir), flush=True)


if __name__ == '__main__':
    main()
