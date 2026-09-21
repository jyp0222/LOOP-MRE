"""Offline, post-hoc neighbor audit. Python 3.8+ stdlib; no API, BERT or CUDA.

Evaluation labels are for diagnosis only. Never import this into the trainer
or use its gold-label findings to filter training neighbors.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path


def _json_lines(path):
    payload = path.read_bytes()
    rows = []
    for number, line in enumerate(payload.decode('utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError('{}:{}: invalid JSON'.format(path.name, number)) from exc
        if not isinstance(row, dict):
            raise ValueError('{}:{}: expected object'.format(path.name, number))
        rows.append(row)
    return rows, hashlib.sha256(payload).hexdigest()


def _load_records(run_dir):
    directory = run_dir / 'data'
    raw = (directory / 'manifest.json').read_bytes()
    manifest = json.loads(raw.decode('utf-8-sig'))
    protocol = manifest.get('protocol')
    if protocol not in ('mre_transductive_random_half', 'mre_transductive_fixed_base_novel'):
        raise ValueError('Expected a supported MRE manifest protocol')
    nb, nt, classes = manifest.get('n_base'), manifest.get('n_total'), manifest.get('classes')
    if (type(nb) is not int or type(nt) is not int or not 0 < nb < nt
            or not isinstance(classes, list) or len(classes) != nt
            or any(not isinstance(c, str) for c in classes) or len(set(classes)) != nt
            or manifest.get('base_classes') != classes[:nb]
            or manifest.get('novel_classes') != classes[nb:]
            or manifest.get('class_to_id') != {c: i for i, c in enumerate(classes)}):
        raise ValueError('Inconsistent manifest class order/counts')
    if protocol == 'mre_transductive_random_half' and nb != nt // 2:
        raise ValueError('Random-half manifest must have floor(n_total / 2) base classes')
    records, hashes = {}, {'data/manifest.json': hashlib.sha256(raw).hexdigest()}
    for split in ('train_labeled', 'test'):
        filename = manifest['files'][split]
        source = (directory / filename).resolve()
        if source.parent != directory.resolve():
            raise ValueError('Prepared files must be directly inside run_dir/data')
        rows, digest = _json_lines(source)
        if digest != manifest['file_sha256'][filename]:
            raise ValueError('Prepared file hash mismatch: {}'.format(filename))
        if (len(rows) != manifest['splits'][split]['count']
                or [r.get('id') for r in rows] != manifest['splits'][split]['ids']):
            raise ValueError('Prepared IDs/count mismatch: {}'.format(split))
        for row in rows:
            key, label, text = row.get('id'), row.get('label'), row.get('text')
            if (not isinstance(key, str) or not key or key in records
                    or type(label) is not int or not 0 <= label < (nb if split == 'train_labeled' else nt)
                    or not isinstance(text, str) or not text.strip()):
                raise ValueError('Invalid/repeated record in {}'.format(split))
            records[key] = dict(row, split=split)
        hashes['data/' + filename] = digest
    return manifest, records, hashes


def _ratio(n, d):
    return float(n) / d if d else None


def _metrics(rows):
    n = len(rows)
    one = [r for r in rows if r['same_count'] == 1]
    def rate(key):
        return _ratio(sum(r[key] for r in rows), n)
    return {
        'decision_count': n, 'unique_queries': len({r['query_id'] for r in rows}),
        'candidate_coverage': _ratio(sum(r['same_count'] > 0 for r in rows), n),
        'neither_candidate_correct': _ratio(sum(r['same_count'] == 0 for r in rows), n),
        'exactly_one_candidate_count': len(one),
        'accuracy_given_exactly_one': _ratio(sum(r['selected_correct'] for r in one), len(one)),
        'selected_same_label_rate': rate('selected_correct'),
        'random_two_choice_expected_rate': _ratio(sum(r['same_count'] / 2.0 for r in rows), n),
        'choice1_same_label_rate': rate('choice1_correct'),
        'selected_choice1_rate': rate('selected_choice1'),
        'selected_self_rate': rate('selected_self'), 'candidate_self_rate': rate('candidate_self'),
        'selected_same_text_rate': rate('selected_same_text'),
        'candidate_same_text_rate': rate('candidate_same_text'), 'fallback_rate': rate('fallback'),
    }


def _groups(rows):
    return {name: _metrics(subset) for name, subset in (
        ('overall', rows), ('base', [r for r in rows if r['query_group'] == 'base']),
        ('novel', [r for r in rows if r['query_group'] == 'novel']))}


def audit_run(run_dir):
    """Read one fixed or random-half run and return a report without writing inputs."""
    run_dir = Path(run_dir).resolve()
    manifest, records, hashes = _load_records(run_dir)
    query_path = run_dir / 'neighbor_queries.jsonl'
    if not query_path.is_file():
        raise FileNotFoundError('No neighbor_queries.jsonl in {}. Supply a GPT run; '
                                'no-LLM runs normally have no queries.'.format(run_dir))
    decisions, digest = _json_lines(query_path)
    hashes['neighbor_queries.jsonl'] = digest
    nb, rows = manifest['n_base'], []
    errors = {name: 0 for name in ('base_to_other_base', 'base_to_novel',
                                  'novel_to_base', 'novel_to_other_novel')}
    llm_errors, wrong_pairs = dict(errors), Counter()
    for number, item in enumerate(decisions, 1):
        qid, candidates = item.get('query_id'), item.get('candidate_ids')
        sid, fallback = item.get('selected_id'), item.get('fallback')
        if (not isinstance(qid, str) or not isinstance(candidates, list) or len(candidates) != 2
                or any(not isinstance(c, str) for c in candidates) or len(set(candidates)) != 2
                or not isinstance(sid, str) or sid not in candidates or type(fallback) is not bool):
            raise ValueError('Invalid query/candidates/selection/fallback in decision {}'.format(number))
        missing = [i for i in [qid] + candidates if i not in records]
        if missing:
            raise ValueError('Unknown sample IDs in decision {}: {}'.format(number, missing))
        query, selected = records[qid], records[sid]
        options = [records[i] for i in candidates]
        same = [c['label'] == query['label'] for c in options]
        qgroup = 'base' if query['label'] < nb else 'novel'
        sgroup = 'base' if selected['label'] < nb else 'novel'
        identity = query['text'].casefold()
        row = {
            'query_id': qid, 'query_label': query['label'], 'query_group': qgroup,
            'same_count': sum(same), 'selected_correct': selected['label'] == query['label'],
            'choice1_correct': same[0], 'selected_choice1': sid == candidates[0],
            'candidate_self': qid in candidates, 'selected_self': sid == qid,
            'candidate_same_text': any(c['text'].casefold() == identity for c in options),
            'selected_same_text': selected['text'].casefold() == identity, 'fallback': fallback,
        }
        rows.append(row)
        if not row['selected_correct']:
            key = qgroup + '_to_' + ('other_' if qgroup == sgroup else '') + sgroup
            errors[key] += 1
            if not fallback:
                llm_errors[key] += 1
                wrong_pairs[(query['label'], selected['label'])] += 1
    llm = [r for r in rows if not r['fallback']]
    strict = [r for r in llm if not r['candidate_self'] and not r['candidate_same_text']]
    counts = Counter(r['label'] for r in records.values())
    return {
        'schema_version': 1,
        'purpose': 'post-hoc diagnosis only; gold labels must never filter training neighbors',
        'run_dir': str(run_dir), 'seed': manifest.get('seed'), 'protocol': manifest['protocol'],
        'n_base': nb, 'n_total': manifest['n_total'], 'source_sha256': hashes,
        'scopes': {'all_decisions': _groups(rows), 'llm_answers_only': _groups(llm),
                   'llm_answers_without_self_or_duplicate': _groups(strict)},
        'errors_by_group': errors, 'llm_errors_by_group': llm_errors,
        'per_relation': [
            {'label': label, 'relation': relation, 'group': 'base' if label < nb else 'novel',
             'training_pool_count': counts[label],
             'metrics': _metrics([r for r in rows if r['query_label'] == label])}
            for label, relation in enumerate(manifest['classes'])],
        'llm_wrong_relation_pairs': [
            {'query_relation': manifest['classes'][a], 'selected_relation': manifest['classes'][b],
             'decision_count': count}
            for (a, b), count in sorted(wrong_pairs.items(), key=lambda kv: (-kv[1], kv[0]))],
        'limitations': [
            'Unit: logged decisions, not independent examples, HTTP calls or all training pairs.',
            'Repeated anchors across refreshes count again; cached reuse within a refresh is not logged.',
            'Legacy query logs have no epoch/refresh ID; per-refresh rates cannot be recovered.',
            'Random-two-choice is an expectation on these SAME two candidates, not a no-LLM training result.',
            'All-decision rates include API fallbacks; LLM-only and strict rates exclude them.',
            'Strict scope excludes decisions if EITHER candidate is self or has the same casefolded text.',
            'Text duplicates use prepared text, not BERT token IDs; conflicting labels are retained.',
            'The full graph is not saved here: top-k purity and unqueried pairs are not measured.',
            'Base/novel refer to query ground truth, used solely in this offline audit.',
            'Ratios are fractions; missing denominators are null. No causal or significance claim.',
        ],
    }


def _pct(value):
    return 'n/a' if value is None else '{:.2f}'.format(100 * value)


def summary_tsv(reports):
    """Plain tab-separated text for direct pasting into Tencent Docs."""
    lines = ['运行目录\tSeed\t统计范围\t查询组\t决策数\t不同Query数\t候选含同类(%)\t两项都异类(%)'
             '\t恰一项同类的决策数\t有唯一正确项时选对(%)\t所选同类(%)\t同候选随机期望(%)'
             '\t比随机高(百分点)\t候选含自身(%)\t候选含同文本(%)\t回退(%)']
    names = {'all_decisions': '全部决策', 'llm_answers_only': '排除回退',
             'llm_answers_without_self_or_duplicate': '排除回退/自身/同文本'}
    for report in reports:
        for scope, groups in report['scopes'].items():
            for group, m in groups.items():
                chosen, random = m['selected_same_label_rate'], m['random_two_choice_expected_rate']
                delta = 'n/a' if chosen is None else '{:+.2f}'.format((chosen - random) * 100)
                fields = [Path(report['run_dir']).name, str(report['seed']), names[scope], group,
                          str(m['decision_count']), str(m['unique_queries']),
                          _pct(m['candidate_coverage']), _pct(m['neither_candidate_correct']),
                          str(m['exactly_one_candidate_count']), _pct(m['accuracy_given_exactly_one']),
                          _pct(chosen), _pct(random), delta, _pct(m['candidate_self_rate']),
                          _pct(m['candidate_same_text_rate']), _pct(m['fallback_rate'])]
                lines.append('\t'.join(fields))
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, nargs='+', required=True,
                        help='One or more existing GPT output directories')
    parser.add_argument('--output-dir', type=Path,
                        help='NEW report directory; existing directories are never overwritten')
    args = parser.parse_args()
    try:
        paths = [p.resolve() for p in args.run_dir]
        if len(set(paths)) != len(paths):
            raise ValueError('A run directory was supplied twice')
        reports = [audit_run(p) for p in paths]
        destination = args.output_dir or Path('outputs') / ('neighbor_audit_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        destination.mkdir(parents=True, exist_ok=False)
        for i, report in enumerate(reports, 1):
            (destination / ('audit_{:02d}_seed{}.json'.format(i, report['seed']))).write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        summary = summary_tsv(reports)
        (destination / 'summary.txt').write_text(summary, encoding='utf-8-sig')
        print(summary, end='')
        print('\n审计完成：{}'.format(destination.resolve()))
        print('比随机高 = 与同一对候选等概率选择相比；不是与 no-LLM 模型准确率相比。')
        print('本次仅离线诊断，不训练、不调用 API。请将 summary.txt 内容发回分析。')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, 'Audit failed: {}\n'.format(exc))


if __name__ == '__main__':
    main()
