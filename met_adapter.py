"""MET JSON -> one marked text per target entity; no image or URL access."""
import hashlib
import json
from collections import Counter


def read_met_source(path, filtered_labels):
    raw = path.read_bytes()
    records = json.loads(raw.decode('utf-8-sig'))
    if not isinstance(records, list):
        raise ValueError('{}: MET root must be an array'.format(path.name))
    rows, classes, filtered = [], [], Counter()
    annotation_count = 0
    for index, record in enumerate(records, 1):
        context = '{}:record {}'.format(path.name, index)
        if (not isinstance(record, list) or len(record) != 4
                or not isinstance(record[0], str) or not record[0].strip()
                or not isinstance(record[3], list)):
            raise ValueError(context + ': expected [sentence, image, topic, entities]')
        sentence = record[0]
        for entity_index, entity in enumerate(record[3], 1):
            annotation_count += 1
            where = context + ':entity {}'.format(entity_index)
            if not isinstance(entity, list) or len(entity) < 4:
                raise ValueError(where + ': expected [name, type, start, end, ...]')
            name, label, start, end = entity[:4]
            if not isinstance(label, str) or not label.strip():
                raise ValueError(where + ': empty entity type')
            if (not isinstance(name, str) or not name.strip()
                    or type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(sentence)):
                raise ValueError(where + ': invalid character [start, end) span')
            if sentence[start:end] != name:
                raise ValueError(where + ': entity name does not match character slice')
            if label in filtered_labels:
                filtered[label] += 1
                continue
            marked = sentence[:start] + ' [ENTITY] ' + sentence[start:end] + ' [/ENTITY] ' + sentence[end:]
            text = 'Entity: {}. Sentence: {}'.format(' '.join(name.split()), ' '.join(marked.split()))
            if label not in classes:
                classes.append(label)
            rows.append({'id': '{}:{:08d}:{:04d}'.format(path.stem, index, entity_index),
                         'text': text, 'relation': label,
                         'source_record': '{}:{:08d}'.format(path.stem, index),
                         'sentence_identity': ' '.join(sentence.split()).casefold()})
    return rows, classes, {
        'file': path.name, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
        'source_records': len(records), 'annotation_count': annotation_count,
        'retained_records': len(rows), 'filtered_records': sum(filtered.values()),
        'filtered_relations': dict(sorted(filtered.items())),
        'blank_token_count': 0, 'records_with_blank_tokens': 0,
        'record_format': 'met_json_array',
        'id_index': 'one-based sentence record and entity annotation indices',
    }


def sentence_overlap_audit(rows, splits):
    lookup = {row['id']: row for row in rows}
    result = {}
    for identity in ('source_record', 'sentence_identity'):
        sets = {split: {lookup[row['id']][identity] for row in records}
                for split, records in splits.items()}
        result[identity] = {
            'labeled_validation': len(sets['train_labeled'] & sets['validation']),
            'labeled_test': len(sets['train_labeled'] & sets['test']),
            'validation_test': len(sets['validation'] & sets['test']),
        }
    return result
