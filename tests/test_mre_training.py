"""Offline integration: real tiny BERT, synthetic text, no downloaded weights/API."""
import json
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
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
from mre_trainer import MRETrainer, NeighborPairs, cluster_score_loader
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

    def test_raw_inner_product_order_does_not_force_self_first(self):
        features = np.array([[1., 0.], [4., 0.], [0., 1.]])
        centers = features[[0, 2]]
        indices, queries = mine_neighbors(features, np.array([0, 0, 1]), centers, 2, 3)
        np.testing.assert_array_equal(indices[:, 0], [1, 1, 2])
        self.assertTrue(all(len(set(row)) == 3 for row in indices))
        self.assertEqual(set(queries), {0, 1, 2})

    def test_retrieval_and_lis_ranking_match_original_source(self):
        import faiss
        import torch.nn.functional as F
        from scipy.optimize import linear_sum_assignment
        original = Path(__file__).resolve().parents[1] / 'utils' / 'memory.py'
        tree = ast.parse(original.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MemoryBank')
        namespace = dict(torch=torch, np=np, F=F, linear_sum_assignment=linear_sum_assignment)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(original), 'exec'), namespace)
        rng = np.random.RandomState(7)
        features = rng.randn(620, 8).astype(np.float32)
        centers = features[::62].copy()
        pseudo = predict_clusters(features, centers)
        bank = namespace['MemoryBank'](620, 8, 10, .1)
        bank.features = torch.from_numpy(features)
        bank.targets = torch.zeros(620, dtype=torch.long)
        with patch.object(faiss, 'index_cpu_to_all_gpus', lambda index: index, create=True):
            old_indices, old_selected = bank.mine_nearest_neighbors(50, pseudo, centers)
            indices, selected = mine_neighbors(features, pseudo, centers, 50, 500)
        np.testing.assert_array_equal(indices, old_indices)
        self.assertEqual(selected, [int(i) for i in old_selected])

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
    def test_pretrain_equal_accuracy_keeps_first_checkpoint_and_stops_after_patience(self):
        class Toy(torch.nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(()))
                self.backbone = SimpleNamespace(config=SimpleNamespace(hidden_size=768))
                self.loss_ce = torch.nn.CrossEntropyLoss()
            def forward(self, inputs):
                # Always predict class 0: validation stays at exactly 50%.
                logits = torch.tensor([[2., 0.]], device=self.weight.device).repeat(len(inputs['input_ids']), 1)
                return {'logits': logits + self.weight * 0}
            def save_backbone(self, path):
                pass

        tensors = torch.utils.data.TensorDataset(torch.ones(2, 4, dtype=torch.long),
            torch.ones(2, 4, dtype=torch.long), torch.zeros(2, 4, dtype=torch.long), torch.tensor([0, 1]))
        loader = torch.utils.data.DataLoader(tensors, batch_size=2)
        data = SimpleNamespace(n_base=2, labeled_dataset=tensors, validation_dataset=tensors,
                               labeled_loader=loader, semi_loader=loader, validation_loader=loader)
        cfg = experiment_config(build_parser().parse_args(['--no-llm']))
        cfg.update(pretrain_epochs=8, patience=2, labeled_batch_size=2)
        def update(model, *args):
            with torch.no_grad():
                model.weight.add_(1)
        with tempfile.TemporaryDirectory() as directory:
            trainer = MRETrainer(cfg, data, None, directory)
            with patch('mre_trainer.BertForModel', Toy), patch.object(trainer, '_step', update), \
                    patch('mre_trainer.optimizer_for', return_value=(None, None)), \
                    patch('mre_trainer.mask_tokens', side_effect=lambda ids, *a, **k: (ids, torch.full_like(ids, -100))):
                model = trainer.pretrain()
            self.assertEqual(trainer.pretrain_best_epoch, 1)
            self.assertEqual(len(trainer.history), 3)
            self.assertEqual(model.weight.item(), 1.)
            self.assertEqual([r['validation_classifier_accuracy_percent'] for r in trainer.history], [50., 50., 50.])

    def test_real_tiny_bert_last_checkpoint_rtr_and_test_reclustering(self):
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
                self.assertTrue((run_dir / 'last_model.pt').is_file())
                self.assertFalse((run_dir / 'best_model.pt').exists())
                self.assertEqual(result['n_test'], 20)
                self.assertEqual(result['model'], None)
                loops = [record for record in trainer.history if record['stage'] == 'loop']
                self.assertEqual([row['epoch'] for row in loops], [1, 2])
                self.assertTrue(all('validation' not in row for row in loops))
                self.assertEqual(result['checkpoint_selection'], 'last_epoch')
                self.assertEqual(result['checkpoint_epoch'], 2)
                checkpoint = torch.load(run_dir / 'last_model.pt', map_location='cpu')
                self.assertEqual(checkpoint['epoch'], 2)
                for key, tensor in trainer.model.state_dict().items():
                    torch.testing.assert_close(tensor.cpu(), checkpoint['model_state'][key])
                metrics, predictions, _ = cluster_score_loader(trainer.model, data.test_loader,
                    trainer.device, data.n_base, data.n_total, config['seed'], config['kmeans_n_init'])
                with patch('mre_trainer.cluster_score_loader', wraps=cluster_score_loader) as recluster:
                    evaluate_saved_run(run_dir)
                    self.assertEqual(recluster.call_count, 1)
                self.assertEqual(metrics, {key: result[key] for key in ('Base', 'Novel', 'Overall')})
                replay = json.loads((run_dir / 'reevaluation.json').read_text())
                self.assertTrue(replay['predictions_identical'])
            stored = [json.loads(line) for line in (run_dir / 'predictions.jsonl').read_text().splitlines()]
            self.assertEqual([r['prediction'] for r in stored], predictions.tolist())
            self.assertEqual(metrics, mre_accuracy([r['label'] for r in stored],
                                                   [r['prediction'] for r in stored], 2, 3))
            self.assertEqual([r['epoch'] for r in trainer.history if r['stage'] == 'intermediate_test'], [1])
            stored[0]['label'] = (stored[0]['label'] + 1) % data.n_total
            (run_dir / 'predictions.jsonl').write_text('\n'.join(json.dumps(row) for row in stored))
            with self.assertRaisesRegex(ValueError, 'sample IDs/labels differ'):
                evaluate_saved_run(run_dir)

    def test_llm_pairs_decode_tokens_allow_self_and_cache_per_graph(self):
        class Data:
            semi_records = [{'id': str(i), 'text': 'Head A Tail B sentence {}'.format(i)} for i in range(3)]
            semi_dataset = torch.utils.data.TensorDataset(
                torch.arange(3).view(3, 1).expand(3, 2), torch.ones(3, 2, dtype=torch.long),
                torch.zeros(3, 2, dtype=torch.long), torch.tensor([0, -1, -1]))

        class Client:
            calls = []
            def choose_neighbor(self, query, choices):
                self.calls.append((query, choices))
                return 1

        class Tokenizer:
            def decode(self, ids, **kwargs):
                return 'decoded-{}'.format(int(ids[0]))

        with tempfile.TemporaryDirectory() as directory:
            client = Client()
            pairs = NeighborPairs(Data(), np.array([[0, 1, 2], [1, 0, 2], [2, 0, 1]]),
                                  [0], np.array([0, 0, 1]), client, Tokenizer(), Path(directory) / 'queries.jsonl')
            with patch('numpy.random.choice', side_effect=lambda values, size: values[:size]) as draw:
                first, second = pairs[0], pairs[0]
                self.assertEqual(draw.call_count, 4)  # q1/q2 consumed even for saved decisions
            self.assertEqual(len(client.calls), 1)
            query, choices = client.calls[0]
            self.assertIn(query, choices)
            self.assertEqual(choices, ['decoded-0', 'decoded-2'])
            torch.testing.assert_close(first['neighbor'][0], second['neighbor'][0])


if __name__ == '__main__':
    unittest.main()
