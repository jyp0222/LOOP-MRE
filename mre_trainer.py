"""LOOP CE+MLM pretraining and CE+RNCL training under the MRE protocol.

The original BERT and contrastive-loss implementations are reused. Test gold
labels never enter losses, query selection, or checkpoint selection. All
intermediate selection uses base validation accuracy with MRE's matching.
"""
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
from transformers import get_linear_schedule_with_warmup

from model import BertForModel, CLBert
from mre_metrics import mre_accuracy
from mre_neighbors import adjacency_mask, mine_neighbors, predict_clusters, squared_distances
from utils.tools import mask_tokens


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    # epsilon matches the legacy transformers AdamW default.
    optimizer = torch.optim.AdamW(groups, lr=lr, eps=1e-6)
    steps = epochs * steps_per_epoch  # includes the final partial batch
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


class NeighborPairs(Dataset):
    def __init__(self, data, indices, selected, pseudo_labels, client, seed, log_path):
        self.data, self.indices = data, indices
        self.selected, self.pseudo = set(selected), pseudo_labels
        self.client, self.log_path = client, Path(log_path)
        self.rng = np.random.RandomState(seed)
        self.decisions = {}

    def __len__(self):
        return len(self.data.semi_dataset)

    def __getitem__(self, index):
        neighbors = self.indices[index]
        if index in self.decisions:
            neighbor_index = self.decisions[index]
        else:
            # Exclude the query itself from the two examples sent to GPT.
            candidates = neighbors[neighbors != index]
            categories = list(dict.fromkeys(self.pseudo[candidates].tolist()))[:2]
            if self.client is not None and index in self.selected and len(categories) == 2:
                choices = [int(self.rng.choice(candidates[self.pseudo[candidates] == category]))
                           for category in categories]
                answer = self.client.choose_neighbor(
                    self.data.semi_records[index]['text'],
                    [self.data.semi_records[i]['text'] for i in choices])
                neighbor_index = choices[answer]
                self.decisions[index] = neighbor_index
                with self.log_path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps({
                        'query_id': self.data.semi_records[index]['id'],
                        'candidate_ids': [self.data.semi_records[i]['id'] for i in choices],
                        'selected_id': self.data.semi_records[neighbor_index]['id'],
                    }) + '\n')
            else:
                # Retain upstream random-neighbor behavior for unqueried anchors.
                neighbor_index = int(self.rng.choice(neighbors))
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
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()

    def pretrain(self):
        cfg, data = self.config, self.data
        model = BertForModel(cfg['bert_model'], data.n_base, self.device).to(self.device)
        if model.backbone.config.hidden_size != 768:
            raise ValueError('Original LOOP pretraining requires BERT hidden_size=768')
        optimizer, scheduler = optimizer_for(model, cfg['lr_pretrain'], cfg['pretrain_epochs'],
                                             len(data.labeled_loader), cfg['warmup_proportion'])
        mixed = DataLoader(data.semi_dataset, batch_size=cfg['train_batch_size'], shuffle=True,
                           generator=torch.Generator().manual_seed(cfg['seed']), num_workers=0)
        iterator = iter(mixed)
        best, stale, best_weights = -float('inf'), 0, None
        print('Pre-training begin: CE + MLM', flush=True)
        for epoch in range(cfg['pretrain_epochs']):
            model.train()
            total_loss = 0.0
            for batch in data.labeled_loader:
                logits = model(model_inputs(batch, self.device))['logits']
                ce = model.loss_ce(logits, batch[3].to(self.device))
                try:
                    semi = next(iterator)
                except StopIteration:
                    iterator = iter(mixed)
                    semi = next(iterator)
                inputs = model_inputs(semi, self.device)
                ids, targets = mask_tokens(semi[0].clone(), self.tokenizer, mlm_probability=0.15)
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
            metrics = mre_accuracy(data.validation_dataset.tensors[3].numpy(), predictions,
                                   data.n_base, data.n_total)
            self._record({'stage': 'pretrain', 'epoch': epoch + 1,
                          'loss': total_loss / len(data.labeled_loader), 'validation': metrics})
            # MRE retains the later checkpoint on equal validation accuracy.
            if metrics['Base'] >= best:
                best, stale = metrics['Base'], 0
                best_weights = cpu_state(model)
            else:
                stale += 1
                if stale >= cfg['patience']:
                    break
        model.load_state_dict(best_weights)
        model.save_backbone(self.run_dir / 'pretrained_backbone')
        return model

    def train(self):
        cfg, data = self.config, self.data
        pretrained = self.pretrain()
        self.model = CLBert(cfg['bert_model'], self.device, data.n_base).to(self.device)
        self.model.backbone.load_state_dict(pretrained.backbone.state_dict())
        del pretrained
        optimizer, scheduler = optimizer_for(
            self.model, cfg['lr'], cfg['train_epochs'],
            math.ceil(len(data.semi_dataset) / cfg['train_batch_size']), cfg['warmup_proportion'])
        labeled_iterator = iter(data.labeled_loader)
        best, stale, best_epoch = -float('inf'), 0, None
        print('Training begin: CE + RNCL; validation selects checkpoints', flush=True)
        for epoch in range(cfg['train_epochs']):
            if epoch % cfg['update_every'] == 0:
                features = extract_features(self.model, data.semi_loader, self.device)
                centers = fit_predictor(features, data.n_total, cfg['seed'], cfg['kmeans_n_init'])
                pseudo = predict_clusters(features, centers)
                indices, selected = mine_neighbors(features, pseudo, centers, cfg['topk'],
                                                    cfg['query_pool_size'])
                pairs = NeighborPairs(data, indices, selected, pseudo, self.client,
                                      cfg['seed'] + epoch, self.run_dir / 'neighbor_queries.jsonl')
                loader = DataLoader(pairs, batch_size=cfg['train_batch_size'], shuffle=True,
                                    num_workers=0, generator=torch.Generator().manual_seed(cfg['seed'] + epoch))
                print('Neighbor refresh: {} candidate queries'.format(len(selected)), flush=True)
            self.model.train()
            total_loss = 0.0
            for batch in loader:
                mask = adjacency_mask(batch['index'].numpy(), batch['neighbors'].numpy(),
                                      batch['target'].numpy())
                mask = torch.from_numpy(mask).to(self.device)
                views = torch.stack([
                    self.model(model_inputs(batch['anchor'], self.device))['features'],
                    self.model(model_inputs(batch['neighbor'], self.device))['features'],
                ], dim=1)
                contrastive = self.model.loss_cl(views, mask=mask, temperature=cfg['temperature'])
                try:
                    labeled = next(labeled_iterator)
                except StopIteration:
                    labeled_iterator = iter(data.labeled_loader)
                    labeled = next(labeled_iterator)
                logits = self.model(model_inputs(labeled, self.device))['logits']
                ce = self.model.loss_ce(logits, labeled[3].to(self.device))
                loss = cfg['ce_weight'] * ce + contrastive
                self._step(self.model, loss, optimizer, scheduler)
                total_loss += loss.item()
            # Refit using TRAIN inputs only, with current model weights. Do NOT
            # fit separately on validation/test inputs inside score_loader.
            features = extract_features(self.model, data.semi_loader, self.device)
            centers = fit_predictor(features, data.n_total, cfg['seed'], cfg['kmeans_n_init'])
            metrics, _ = score_loader(self.model, data.validation_loader, centers, self.device,
                                      data.n_base, data.n_total)
            self._record({'stage': 'loop', 'epoch': epoch + 1,
                          'loss': total_loss / len(loader), 'validation': metrics,
                          'llm_requests': self.client.requests_made if self.client else 0,
                          'llm_cache_hits': self.client.cache_hits if self.client else 0})
            if metrics['Base'] >= best:
                best, stale, best_epoch = metrics['Base'], 0, epoch + 1
                torch.save({'model_state': cpu_state(self.model), 'centers': torch.from_numpy(centers),
                            'epoch': best_epoch, 'validation': metrics,
                            'n_base': data.n_base, 'n_total': data.n_total}, self.run_dir / 'best_model.pt')
            else:
                stale += 1
                if stale >= cfg['patience']:
                    break
        # This is our own checkpoint, not an untrusted downloaded pickle.
        checkpoint = torch.load(self.run_dir / 'best_model.pt', map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state'])
        self.centers = checkpoint['centers'].numpy()
        self.model.save_backbone(self.run_dir / 'backbone')
        self.tokenizer.save_pretrained(self.run_dir / 'tokenizer')
        # Older AutoTokenizer versions also consult the model config when
        # reopening a tokenizer directory. Keep this run self-contained.
        self.model.backbone.config.save_pretrained(self.run_dir / 'tokenizer')
        metrics, predictions = score_loader(self.model, data.test_loader, self.centers, self.device,
                                            data.n_base, data.n_total)
        result = {'protocol': data.manifest['protocol'], 'seed': cfg['seed'],
                  'method': 'LOOP-MRE-GPT' if self.client else 'LOOP-MRE-no-LLM',
                  'model': self.client.model if self.client else None,
                  'reasoning_effort': self.client.reasoning_effort if self.client else None,
                  'best_epoch': best_epoch, 'validation_Base': best,
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
        features = extract_features(self.model, self.data.semi_loader, self.device)
        labels = self.data.labeled_dataset.tensors[3].numpy()
        labeled_features = features[:len(labels)]
        prototypes = np.stack([labeled_features[labels == c].mean(0) for c in range(self.data.n_base)])
        _, assigned = linear_sum_assignment(squared_distances(prototypes, self.centers))
        novel = [i for i in range(self.data.n_total) if i not in set(assigned.tolist())]
        names = {}
        for cluster in novel:
            nearest = np.argsort(squared_distances(features, self.centers[cluster:cluster + 1])[:, 0])[:3]
            names[str(cluster)] = self.client.name_cluster([self.data.semi_records[int(i)]['text'] for i in nearest])
        write_json(self.run_dir / 'cluster_names.json', names)
