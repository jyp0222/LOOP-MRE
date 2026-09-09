"""Offline integration: real tiny BERT, synthetic text, no downloaded weights/API."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import BertConfig, BertForMaskedLM, BertTokenizer

from mre_data import MREData
from mre_metrics import mre_accuracy
from mre_neighbors import adjacency_mask, mine_neighbors, predict_clusters
from mre_protocol import prepare_dataset
from mre_trainer import MRETrainer, NeighborPairs, score_loader
from run_mre import build_parser, experiment_config, evaluate_saved_run, load_data, main


class RunnerTests(unittest.TestCase):
    def test_contradictory_no_llm_check_never_calls_api(self):
        with patch('run_mre.create_client', side_effect=AssertionError('No paid calls')):
            with self.assertRaises(SystemExit) as error:
                main(['--check-llm', '--no-llm'])
        self.assertEqual(error.exception.code, 2)

    def test_invalid_hyperparameters_fail_before_pretraining(self):
        args = build_parser().parse_args([])
        for name, value in [('QUERY_POOL_SIZE', 1.5), ('CE_WEIGHT', float('nan')),
                            ('CE_WEIGHT', float('inf'))]:
            with patch('run_mre.defaults.' + name, value), self.assertRaises(ValueError):
                experiment_config(args)

    def test_manifest_fingerprint_rejected_before_tokenization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data').mkdir()
            (root / 'data' / 'manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'manifest'):
                load_data({'manifest_sha256': 'wrong'}, root)


class NeighborsTests(unittest.TestCase):
    def test_unlabeled_minus_one_is_not_a_positive_class(self):
        mask = adjacency_mask([0, 1, 2], [[0], [1], [2]], [-1, -1, -1])
        np.testing.assert_array_equal(mask, np.eye(3))
        mask = adjacency_mask([0, 1, 2], [[0], [1], [2]], [4, 4, -1])
        self.assertEqual(mask[0, 1], 1)
        self.assertEqual(mask[0, 2], 0)

    def test_self_first_even_when_inner_product_prefers_other(self):
        features = np.array([[1., 0.], [4., 0.], [0., 1.]])
        centers = features[[0, 2]]
        indices, queries = mine_neighbors(features, np.array([0, 0, 1]), centers, 2, 3)
        np.testing.assert_array_equal(indices[:, 0], np.arange(3))
        self.assertTrue(all(len(set(row)) == 3 for row in indices))
        self.assertEqual(set(queries), {0, 1, 2})

    def test_no_queries_when_budget_zero(self):
        features = np.eye(3)
        _, queries = mine_neighbors(features, np.arange(3), features, 20, 0)
        self.assertEqual(queries, [])

    def test_bad_features_and_pseudo_labels_fail(self):
        for bad in ([[float('nan'), 0.]], [[float('inf'), 0.]], [], [1, 2]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                predict_clusters(bad, [[0., 0.]])
        for labels in ([0, -1], [0., 1.], [0, 2], [[0, 1]]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                mine_neighbors(np.eye(2), labels, np.eye(2))


def make_source(path):
    path.mkdir()
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'head', 'tail', 'sentence',
             ':', '.', '[', ']', '/', 'alice', 'bob', 'paris', 'rome', 'born', 'works', 'in']
    for filename, relations, count in [('train.txt', ['birthplace', 'workplace'], 12),
                                       ('test.txt', ['visited'], 8)]:
        rows = []
        for rel in relations:
            for index in range(count):
                tokens = ['alice' if index % 2 else 'bob', rel, 'in', 'paris' if index % 3 else 'rome', str(index)]
                vocab.extend([rel, str(index)])
                rows.append({'tokens': tokens, 'h': {'pos': [0]}, 't': {'pos': [3]},
                             'relation': rel, 'img_id': 'image-does-not-exist.jpg'})
        (path / filename).write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
    return list(dict.fromkeys(vocab))


class TrainingIntegrationTests(unittest.TestCase):
    def test_real_tiny_bert_training_best_checkpoint_and_fixed_prediction(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocab = make_source(root / 'source')
            backbone = root / 'bert'
            backbone.mkdir()
            (backbone / 'vocab.txt').write_text('\n'.join(vocab) + '\n', encoding='utf-8')
            tokenizer = BertTokenizer(str(backbone / 'vocab.txt'), do_lower_case=True)
            tokenizer.save_pretrained(backbone)
            torch.manual_seed(1)
            BertForMaskedLM(BertConfig(vocab_size=len(vocab), hidden_size=768,
                                      num_hidden_layers=1, num_attention_heads=12,
                                      intermediate_size=32, max_position_embeddings=64)).save_pretrained(backbone)
            run_dir = root / 'run'
            prepare_dataset(root / 'source', run_dir / 'data', expected_base=2, expected_novel=1)
            data = MREData(run_dir / 'data', tokenizer, max_length=32,
                           labeled_batch_size=4, train_batch_size=12, eval_batch_size=12)
            args = build_parser().parse_args(['--no-llm', '--bert-model', str(backbone),
                                               '--tokenizer', str(backbone)])
            config = experiment_config(args)
            config.update(pretrain_epochs=1, train_epochs=2, max_length=32, train_batch_size=12,
                          labeled_batch_size=4, eval_batch_size=12, kmeans_n_init=2, update_every=1,
                          query_pool_size=3, name_clusters=False)
            config['manifest_sha256'] = hashlib.sha256((run_dir / 'data' / 'manifest.json').read_bytes()).hexdigest()
            (run_dir / 'config.json').write_text(json.dumps(config), encoding='utf-8')
            trainer = MRETrainer(config, data, tokenizer, run_dir)
            # Fail the test if either new HTTP transport or a legacy call is attempted.
            with patch('requests.post', side_effect=AssertionError('Network forbidden')):
                result = trainer.train()
                self.assertTrue((run_dir / 'best_model.pt').is_file())
                self.assertEqual(result['n_test'], 20)
                self.assertEqual(result['model'], None)
                self.assertIsNone(trainer.history[-1]['validation']['Novel'])
                self.assertNotIn('test', trainer.history[-1])
                with patch('mre_trainer.KMeans.fit', side_effect=AssertionError('Evaluation must not fit')):
                    metrics, predictions = score_loader(trainer.model, data.test_loader, trainer.centers,
                                                         trainer.device, data.n_base, data.n_total)
                    evaluate_saved_run(run_dir)
                self.assertEqual(metrics, {key: result[key] for key in ('Base', 'Novel', 'Overall')})
                replay = json.loads((run_dir / 'reevaluation.json').read_text())
                self.assertTrue(replay['predictions_identical'])
            stored = [json.loads(line) for line in (run_dir / 'predictions.jsonl').read_text().splitlines()]
            self.assertEqual([r['prediction'] for r in stored], predictions.tolist())
            self.assertEqual(metrics, mre_accuracy([r['label'] for r in stored],
                                                   [r['prediction'] for r in stored], 2, 3))
            # Every model-selection record contains base validation, never a novel score.
            self.assertTrue(all(entry['validation']['Novel'] is None for entry in trainer.history))
            stored[0]['label'] = (stored[0]['label'] + 1) % data.n_total
            (run_dir / 'predictions.jsonl').write_text('\n'.join(json.dumps(row) for row in stored))
            with self.assertRaisesRegex(ValueError, 'sample IDs/labels differ'):
                evaluate_saved_run(run_dir)

    def test_llm_pairs_receive_text_only_no_self_and_cache_per_graph(self):
        class Data:
            semi_records = [{'id': str(i), 'text': 'Head A Tail B sentence {}'.format(i)} for i in range(3)]
            semi_dataset = torch.utils.data.TensorDataset(
                torch.ones(3, 2, dtype=torch.long), torch.ones(3, 2, dtype=torch.long),
                torch.zeros(3, 2, dtype=torch.long), torch.tensor([0, -1, -1]))

        class Client:
            calls = []
            def choose_neighbor(self, query, choices):
                self.calls.append((query, choices))
                return 1

        with tempfile.TemporaryDirectory() as directory:
            client = Client()
            pairs = NeighborPairs(Data(), np.array([[0, 1, 2], [1, 0, 2], [2, 0, 1]]),
                                  [0], np.array([0, 0, 1]), client, 0, Path(directory) / 'queries.jsonl')
            first, second = pairs[0], pairs[0]
            self.assertEqual(len(client.calls), 1)
            query, choices = client.calls[0]
            self.assertNotIn(query, choices)
            self.assertEqual(choices, [Data.semi_records[1]['text'], Data.semi_records[2]['text']])
            torch.testing.assert_close(first['neighbor'][0], second['neighbor'][0])


if __name__ == '__main__':
    unittest.main()
