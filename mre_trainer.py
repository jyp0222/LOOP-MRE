"""LOOP CE+MLM pretraining and CE+RNCL training under the MRE protocol.

Pretraining selects on ordinary classifier accuracy. LOOP uses its last epoch
and reclusters the test features. Test scores never control model selection.
"""
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from model import BertForModel, CLBert
from mre_metrics import mre_accuracy
from mre_neighbors import (adjacency_mask, cap_query_candidates, mine_neighbors,
                           predict_clusters)
from utils.tools import mask_tokens, view_generator


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def model_inputs(batch, device):
    return {name: tensor.to(device) for name, tensor in zip(
        ('input_ids', 'attention_mask', 'token_type_ids'), batch[:3])}


def cpu_state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def optimizer_for(model, lr, epochs, steps_per_epoch, warmup):
    no_decay = ('bias', 'LayerNorm.bias', 'LayerNorm.weight')
    groups = [
        {'params': [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(x in n for x in no_decay)],
         'weight_decay': 0.0},
    ]
    optimizer = AdamW(groups, lr=lr)
    # Upstream uses floor(N / batch_size), despite keeping partial batches.
    steps = epochs * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * warmup), steps)
    return optimizer, scheduler


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    return np.concatenate([model(model_inputs(batch, device), output_hidden_states=True)
                           ['hidden_states'].cpu().numpy() for batch in loader], axis=0)


def fit_predictor(features, n_clusters, seed, n_init):
    if len(features) < n_clusters:
        raise ValueError('Training pool has fewer samples than the fixed number of classes')
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init).fit(features)
    return km.cluster_centers_.astype(np.float32)


def score_loader(model, loader, centers, device, n_base, n_total):
    features = extract_features(model, loader, device)
    predictions = predict_clusters(features, centers)
    labels = loader.dataset.tensors[3].numpy()
    return mre_accuracy(labels, predictions, n_base, n_total), predictions


def cluster_score_loader(model, loader, device, n_base, n_total, seed, n_init):
    """Fit KMeans on test features; ground-truth labels are used only to score."""
    features = extract_features(model, loader, device)
    km = KMeans(n_clusters=n_total, random_state=seed, n_init=n_init).fit(features)
    labels = loader.dataset.tensors[3].numpy()
    return (mre_accuracy(labels, km.labels_, n_base, n_total), km.labels_,
            km.cluster_centers_.astype(np.float32))


class NeighborPairs(Dataset):
    def __init__(self, data, indices, selected, pseudo_labels, client, tokenizer, log_path):
        self.data, self.indices = data, indices
        self.selected, self.pseudo = set(selected), pseudo_labels
        self.client, self.log_path = client, Path(log_path)
        self.tokenizer = tokenizer
        self.decisions = {}

    def __len__(self):
        return len(self.data.semi_dataset)

    def __getitem__(self, index):
        neighbors = self.indices[index]
        neighbor_pred = self.pseudo[neighbors]
        categories = list(dict.fromkeys(neighbor_pred.tolist()))[:2]
        # Upstream allows self and draws q1/q2 even on saved-decision hits.
        q1 = int(np.random.choice(neighbors[neighbor_pred == categories[0]], 1)[0])
        if self.client is None or index not in self.selected or len(categories) == 1:
            neighbor_index = int(np.random.choice(neighbors, 1)[0])
        else:
            q2 = int(np.random.choice(neighbors[neighbor_pred == categories[1]], 1)[0])
            if index not in self.decisions:
                choices = [q1, q2]
                def decode(i):
                    return self.tokenizer.decode(self.data.semi_dataset[i][0],
                        skip_special_tokens=True, clean_up_tokenization_spaces=True)
                answer = self.client.choose_neighbor(
                    decode(index), [decode(i) for i in choices])
                neighbor_index = choices[answer]
                self.decisions[index] = neighbor_index
                with self.log_path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps({
                        'query_id': self.data.semi_records[index]['id'],
                        'candidate_ids': [self.data.semi_records[i]['id'] for i in choices],
                        'selected_id': self.data.semi_records[neighbor_index]['id'],
                        'fallback': getattr(self.client, 'last_query_fallback', False),
                    }) + '\n')
            else:
                neighbor_index = self.decisions[index]
        anchor = self.data.semi_dataset[index]
        neighbor = self.data.semi_dataset[neighbor_index]
        return {'anchor': anchor[:3], 'neighbor': neighbor[:3],
                'neighbors': torch.from_numpy(neighbors), 'target': anchor[3], 'index': index}


class MRETrainer:
    def __init__(self, config, data, tokenizer, run_dir, client=None):
        self.config, self.data, self.tokenizer = config, data, tokenizer
        self.run_dir, self.client = Path(run_dir), client
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.history = []
        self.centers = None
        self.model = None
        seed_everything(config['seed'])

    def _record(self, record):
        self.history.append(record)
        write_json(self.run_dir / 'history.json', self.history)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def _step(self, model, loss, optimizer, scheduler):
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss; stopping before corrupting weights')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    def pretrain(self):
        cfg, data = self.config, self.data
        model = BertForModel(cfg['bert_model'], data.n_base, self.device).to(self.device)
        if model.backbone.config.hidden_size != 768:
            raise ValueError('Original LOOP pretraining requires BERT hidden_size=768')
        optimizer, scheduler = optimizer_for(model, cfg['lr_pretrain'], cfg['pretrain_epochs'],
                                             len(data.labeled_dataset) // cfg['labeled_batch_size'],
                                             cfg['warmup_proportion'])
        iterator = iter(data.semi_loader)
        best, stale, best_weights, best_epoch = 0.0, 0, None, None
        print('Pre-training begin: CE + MLM', flush=True)
        for epoch in range(cfg['pretrain_epochs']):
            model.train()
            total_loss = 0.0
            for batch in data.labeled_loader:
                try:
                    semi = next(iterator)
                except StopIteration:
                    iterator = iter(data.semi_loader)
                    semi = next(iterator)
                inputs = model_inputs(semi, self.device)
                ids, targets = mask_tokens(semi[0].clone(), self.tokenizer, mlm_probability=0.15)
                # The upstream masks the MLM input before both forward passes.
                logits = model(model_inputs(batch, self.device))['logits']
                ce = model.loss_ce(logits, batch[3].to(self.device))
                # A very short batch can contain no masked tokens. Its MLM term
                # is zero, avoiding CrossEntropyLoss(all ignore_index) -> NaN.
                if (targets != -100).any():
                    inputs['input_ids'] = ids.to(self.device)
                    mlm = model.mlmForward(inputs, targets.to(self.device))
                else:
                    mlm = ce.new_zeros(())
                loss = ce + mlm
                self._step(model, loss, optimizer, scheduler)
                total_loss += loss.item()
            model.eval()
            with torch.no_grad():
                predictions = np.concatenate([
                    model(model_inputs(batch, self.device))['logits'].argmax(1).cpu().numpy()
                    for batch in data.validation_loader])
            score = round(float(np.mean(predictions == data.validation_dataset.tensors[3].numpy())) * 100, 2)
            self._record({'stage': 'pretrain', 'epoch': epoch + 1,
                          'loss': total_loss / len(data.labeled_loader),
                          'validation_classifier_accuracy_percent': score})
            if score > best:
                best, stale, best_epoch = score, 0, epoch + 1
                best_weights = cpu_state(model)
            else:
                stale += 1
                if stale >= cfg['patience']:
                    break
        if best_weights is None:
            raise RuntimeError('No pretraining checkpoint exceeded 0% validation accuracy')
        model.load_state_dict(best_weights)
        self.pretrain_best_epoch = best_epoch
        write_json(self.run_dir / 'pretrain_selection.json', {
            'best_epoch': best_epoch, 'validation_classifier_accuracy_percent': best,
            'completed_epochs': epoch + 1, 'patience': cfg['patience'], 'ties': 'keep_earlier'})
        model.save_backbone(self.run_dir / 'pretrained_backbone')
        return model

    def train(self):
        cfg, data = self.config, self.data
        pretrained = self.pretrain()
        seed_everything(cfg['seed'])
        self.model = CLBert(cfg['bert_model'], self.device, data.n_base).to(self.device)
        self.model.backbone.load_state_dict(pretrained.backbone.state_dict())
        del pretrained
        optimizer, scheduler = optimizer_for(self.model, cfg['lr'], cfg['train_epochs'],
            len(data.semi_dataset) // cfg['train_batch_size'], cfg['warmup_proportion'])
        generator = view_generator(self.tokenizer, cfg['rtr_prob'], cfg['seed'])
        labeled_iterator = None
        print('Training begin: original LOOP CE + RNCL; use last epoch, no LOOP early stopping', flush=True)
        for epoch in range(cfg['train_epochs']):
            if epoch % cfg['update_every'] == 0:
                features = extract_features(self.model, data.semi_loader, self.device)
                # Use KMeans.labels_, just like upstream, for graph pseudo-labels.
                km = KMeans(n_clusters=data.n_total, random_state=cfg['seed'],
                            n_init=cfg['kmeans_n_init']).fit(features)
                neighbor_features = extract_features(self.model, data.semi_loader, self.device)
                indices, selected = mine_neighbors(neighbor_features, km.labels_, km.cluster_centers_,
                                                    cfg['topk'], cfg['query_pool_size'])
                ranked_count = len(selected)
                selected = cap_query_candidates(indices, km.labels_, selected, cfg.get('max_queries_per_refresh'))
                pairs = NeighborPairs(data, indices, selected, km.labels_, self.client,
                                      self.tokenizer, self.run_dir / 'neighbor_queries.jsonl')
                loader = DataLoader(pairs, batch_size=cfg['train_batch_size'], shuffle=True, num_workers=0)
                print('Neighbor refresh: {} candidate queries (ranked intersection: {}; query cap: {})'.format(
                    len(selected), ranked_count, cfg.get('max_queries_per_refresh')), flush=True)
            if labeled_iterator is None:
                labeled_iterator = iter(data.labeled_loader)
            self.model.train()
            total_loss = 0.0
            for batch in loader:
                mask = torch.from_numpy(adjacency_mask(batch['index'].numpy(), batch['neighbors'].numpy(),
                                                       batch['target'].numpy())).to(self.device)
                anchor = model_inputs(batch['anchor'], self.device)
                neighbor = model_inputs(batch['neighbor'], self.device)
                if cfg['view_strategy'] == 'rtr':
                    anchor['input_ids'] = generator.random_token_replace(batch['anchor'][0].clone()).to(self.device)
                    neighbor['input_ids'] = generator.random_token_replace(batch['neighbor'][0].clone()).to(self.device)
                elif cfg['view_strategy'] != 'none':
                    raise ValueError('view_strategy must be rtr or none')
                views = torch.stack([self.model(anchor)['features'], self.model(neighbor)['features']], dim=1)
                contrastive = self.model.loss_cl(views, mask=mask, temperature=cfg['temperature'])
                try:
                    labeled = next(labeled_iterator)
                except StopIteration:
                    labeled_iterator = iter(data.labeled_loader)
                    labeled = next(labeled_iterator)
                logits = self.model(model_inputs(labeled, self.device))['logits']
                loss = cfg['ce_weight'] * self.model.loss_ce(logits, labeled[3].to(self.device)) + contrastive
                self._step(self.model, loss, optimizer, scheduler)
                total_loss += loss.item()
            self._record({'stage': 'loop', 'epoch': epoch + 1, 'loss': total_loss / len(loader),
                          'llm_requests': self.client.requests_made if self.client else 0,
                          'llm_cache_hits': self.client.cache_hits if self.client else 0,
                          'llm_fallbacks': getattr(self.client, 'fallback_count', 0)})
            if (epoch + 1) % cfg['update_every'] == 0 and epoch + 1 != cfg['train_epochs']:
                metrics, _, _ = cluster_score_loader(self.model, data.test_loader, self.device,
                    data.n_base, data.n_total, cfg['seed'], cfg['kmeans_n_init'])
                # Original intermediate test reporting; NEVER selects a model.
                self._record({'stage': 'intermediate_test', 'epoch': epoch + 1, 'test': metrics})
        self.model.save_backbone(self.run_dir / 'backbone')
        self.tokenizer.save_pretrained(self.run_dir / 'tokenizer')
        self.model.backbone.config.save_pretrained(self.run_dir / 'tokenizer')
        metrics, predictions, self.centers = cluster_score_loader(self.model, data.test_loader, self.device,
            data.n_base, data.n_total, cfg['seed'], cfg['kmeans_n_init'])
        torch.save({'model_state': cpu_state(self.model), 'centers': torch.from_numpy(self.centers),
                    'epoch': cfg['train_epochs'], 'selection': 'last_epoch', 'evaluation': 'test_kmeans',
                    'n_base': data.n_base, 'n_total': data.n_total}, self.run_dir / 'last_model.pt')
        result = {'protocol': data.manifest['protocol'], 'seed': cfg['seed'],
                  'method': 'LOOP-MRE-GPT-FewRelPrompt' if self.client else 'LOOP-original-MRE-no-LLM',
                  'experiment_variant': cfg['experiment_variant'],
                  'prompt_version': cfg['prompt_version'] if self.client else None,
                  'model': self.client.model if self.client else None,
                  'reasoning_effort': None,
                  'checkpoint_selection': 'last_epoch', 'checkpoint_epoch': cfg['train_epochs'],
                  'pretrain_best_epoch': self.pretrain_best_epoch,
                  'llm_snapshot_verified': False if self.client else None,
                  'llm_fallbacks': getattr(self.client, 'fallback_count', 0),
                  'n_test': len(predictions), 'transductive': True, **metrics}
        write_json(self.run_dir / 'results.json', result)
        with (self.run_dir / 'results.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result))
            writer.writeheader()
            writer.writerow(result)
        with (self.run_dir / 'predictions.jsonl').open('w', encoding='utf-8') as handle:
            for record, prediction in zip(data.test_records, predictions):
                handle.write(json.dumps({'id': record['id'], 'label': record['label'],
                                         'prediction': int(prediction)}) + '\n')
        print('Final MRE test metrics (fractions): {}'.format(metrics), flush=True)
        if cfg['name_clusters'] and self.client is not None:
            self.name_clusters()
        return result

    def name_clusters(self):
        # Names only interpret the saved model; they never enter MRE matching.
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial.distance import cdist
        from llm_client import LLMError, LLMBudgetExceeded
        features = extract_features(self.model, self.data.semi_loader, self.device)
        labels = self.data.labeled_dataset.tensors[3].numpy()
        labeled_features = features[:len(labels)]
        # Upstream accumulates prototypes in float64 and matches Euclidean
        # distances. Squaring the assignment costs can change the matching.
        prototypes = np.stack([labeled_features[labels == c].astype(np.float64).mean(0)
                               for c in range(self.data.n_base)])
        # Upstream naming fits its own clusterer on the semi-supervised pool.
        centers = fit_predictor(features, self.data.n_total, self.config['seed'], self.config['kmeans_n_init'])
        _, assigned = linear_sum_assignment(cdist(prototypes, centers, 'euclidean'))
        novel = [i for i in range(self.data.n_total) if i not in set(assigned.tolist())]
        feat_tensor, center_tensor = torch.from_numpy(features), torch.from_numpy(centers[novel])
        distances = torch.sqrt((feat_tensor ** 2).sum(1).unsqueeze(1)
                               + (center_tensor ** 2).sum(1).unsqueeze(0)
                               - 2 * feat_tensor.mm(center_tensor.t())).t()
        _, nearest_rows = torch.sort(distances, dim=1)
        names = {}
        for cluster, nearest in zip(novel, nearest_rows[:, :3]):
            samples = [self.tokenizer.decode(self.data.semi_dataset[int(i)][0], skip_special_tokens=True,
                                            clean_up_tokenization_spaces=True) for i in nearest]
            try:
                names[str(cluster)] = self.client.name_cluster(samples)
            except LLMBudgetExceeded:
                raise
            except LLMError as exc:
                names[str(cluster)] = {'error': str(exc)}
        write_json(self.run_dir / 'cluster_names.json', names)
