"""LOOP neighbor mining, without consuming any unlabelled ground-truth IDs."""
import numpy as np


def squared_distances(features, centers):
    features = np.asarray(features, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    if (features.ndim != 2 or centers.ndim != 2 or min(features.shape) == 0
            or min(centers.shape) == 0 or features.shape[1] != centers.shape[1]
            or not np.isfinite(features).all() or not np.isfinite(centers).all()):
        raise ValueError('Features and centers must be finite, nonempty matrices with equal width')
    return np.maximum((features ** 2).sum(1)[:, None]
                      + (centers ** 2).sum(1)[None, :]
                      - 2 * features.dot(centers.T), 0)


def predict_clusters(features, centers):
    """Fixed nearest-center prediction: this function never fits a clusterer."""
    return squared_distances(features, centers).argmin(axis=1)


def mine_neighbors(features, pseudo_labels, centers, topk=50, query_pool_size=500):
    """Upstream raw inner-product FAISS retrieval and Torch LIS ranking.

    Do not force self into slot zero: preserve the original search result.
    Ground-truth labels are deliberately absent from this interface.
    """
    import torch
    import torch.nn.functional as F
    try:
        import faiss
    except ImportError as exc:
        raise ImportError('Original LOOP retrieval requires faiss: install faiss-gpu in the server environment '
                          '(or faiss-cpu for CPU testing).') from exc
    features = np.ascontiguousarray(features, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    pseudo_labels = np.asarray(pseudo_labels)
    squared_distances(features, centers)  # validate shape/finite values
    n = len(features)
    if (n < 2 or topk < 1 or query_pool_size < 0 or pseudo_labels.shape != (n,)
            or pseudo_labels.dtype.kind not in 'iu' or np.any(pseudo_labels < 0)
            or np.any(pseudo_labels >= len(centers))):
        raise ValueError('Invalid neighbor mining dimensions or budget')
    k = min(topk, n - 1)
    index = faiss.IndexFlatIP(features.shape[1])
    if hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(features)
    _, indices = index.search(features, k + 1)
    tensor_features = torch.from_numpy(features)
    q = 1.0 / (1.0 + torch.sum((tensor_features.unsqueeze(1) - centers) ** 2, dim=2))
    q = q ** 2 / 2.0
    q = (q.t() / torch.sum(q, dim=1)).t()
    weight = q ** 2 / torch.sum(q, dim=0)
    p = (weight.t() / torch.sum(weight, dim=1)).t()
    prob = F.softmax(p, dim=-1)
    entropy = -torch.sum(prob * torch.log(prob), 1)
    _, entropy_order = torch.sort(entropy, descending=True)
    # Upstream drops the FIRST result for inconsistency, rather than finding self.
    inconsistent = torch.from_numpy(
        (pseudo_labels[indices[:, 1:]] != pseudo_labels[:, None]).sum(axis=1))
    _, inconsistency_order = torch.sort(inconsistent, descending=True)
    selected = [int(i) for i in inconsistency_order[:query_pool_size]
                if i in entropy_order[:query_pool_size]]
    return indices, selected


def cap_query_candidates(indices, pseudo_labels, selected, limit=None):
    """Bound smoke queries after normal selection, keeping its ranking order.

    Only anchors with two distinct neighbor pseudo-classes can reach the LLM
    in NeighborPairs. The full retrieved row (self eligible) is used. No gold
    labels are used, and uncapped formal runs keep the original selection.
    """
    if limit is None:
        return list(selected)
    if type(limit) is not int or limit < 0:
        raise ValueError('Query cap must be a nonnegative integer or None')
    indices, pseudo_labels = np.asarray(indices), np.asarray(pseudo_labels)
    capped = []
    for index in selected:
        if len(capped) >= limit:
            break
        if len(np.unique(pseudo_labels[indices[index]])) >= 2:
            capped.append(int(index))
    return capped


def adjacency_mask(indices, neighbors, targets):
    """LOOP RNCL positives plus genuinely labeled same-class examples only."""
    indices = np.asarray(indices)
    neighbors = np.asarray(neighbors)
    targets = np.asarray(targets)
    nearby = (indices[None, :, None] == neighbors[:, None, :]).any(axis=2)
    same_labeled = ((targets[:, None] == targets[None, :])
                    & (targets[:, None] >= 0) & (targets[None, :] >= 0))
    mask = nearby | same_labeled
    np.fill_diagonal(mask, True)
    return mask.astype(np.float32)
