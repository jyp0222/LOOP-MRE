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


def mine_neighbors(features, pseudo_labels, centers, topk=20, query_pool_size=500):
    """Retain upstream IP similarity, q/p formula and softmax-entropy ranking.

    Self is explicitly placed first, instead of assuming raw inner-product
    search necessarily retrieves self first. Blocked NumPy avoids a mandatory
    GPU FAISS dependency. No ground-truth label argument exists here.
    """
    features = np.asarray(features, dtype=np.float32)
    pseudo_labels = np.asarray(pseudo_labels)
    distances = squared_distances(features, centers)
    n = len(features)
    if (n < 2 or topk < 1 or query_pool_size < 0 or pseudo_labels.shape != (n,)
            or pseudo_labels.dtype.kind not in 'iu' or np.any(pseudo_labels < 0)
            or np.any(pseudo_labels >= len(centers))):
        raise ValueError('Invalid neighbor mining dimensions or budget')
    k = min(topk, n - 1)
    indices = np.empty((n, k + 1), dtype=np.int64)
    indices[:, 0] = np.arange(n)
    for start in range(0, n, 256):
        stop = min(start + 256, n)
        scores = features[start:stop].dot(features.T)
        scores[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        indices[start:stop, 1:] = np.argsort(-scores, axis=1, kind='stable')[:, :k]
    # This intentionally follows utils/memory.py, including its q**2 / 2 and
    # softmax(p) operations. Changing these to the paper formula is a separate
    # methodological ablation, not part of the requested protocol migration.
    q = 1.0 / (1.0 + distances)
    q = q ** 2 / 2.0
    q /= q.sum(axis=1, keepdims=True)
    weight = q ** 2 / np.maximum(q.sum(axis=0), np.finfo(np.float32).tiny)
    p = weight / weight.sum(axis=1, keepdims=True)
    softmax_p = np.exp(p - p.max(axis=1, keepdims=True))
    softmax_p /= softmax_p.sum(axis=1, keepdims=True)
    entropy = -(softmax_p * np.log(softmax_p)).sum(axis=1)
    inconsistent = (pseudo_labels[indices[:, 1:]] != pseudo_labels[:, None]).sum(axis=1)
    budget = min(query_pool_size, n)
    high_entropy = set(np.argsort(-entropy, kind='stable')[:budget].tolist())
    selected = [int(i) for i in np.argsort(-inconsistent, kind='stable')[:budget]
                if int(i) in high_entropy]
    return indices, selected


def cap_query_candidates(indices, pseudo_labels, selected, limit=None):
    """Bound smoke queries after normal selection, keeping its ranking order.

    Only anchors with two distinct neighbor pseudo-classes can reach the LLM
    in NeighborPairs. Self is excluded exactly as in that dataset. No gold
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
        others = indices[index][indices[index] != index]
        if len(np.unique(pseudo_labels[others])) >= 2:
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
