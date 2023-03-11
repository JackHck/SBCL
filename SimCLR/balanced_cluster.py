import warnings
from functools import partial

import numpy as np
import torch
from tqdm import trange, tqdm


def balanced_kmean(
        X,
        n_clusters,
        init='k-means++',
        device=torch.device('cpu'),
        tol=1e-4,
        iol=100,
        distance='cosine'

):
    '''
    X: the clustered feature
    n_clusters: the cluster number
    '''
    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    if distance == 'euclidean':
        pairwise_similarity_function = partial(pairwise_euclidean, device=device)
    elif distance == 'cosine':
        pairwise_similarity_function = partial(pairwise_cosine, device=device)
    else:
        raise NotImplementedError
    # initialize
    if init == 'random':
        centroids = initialize(X, n_clusters)
    elif init == 'k-means++':
        centroids, _ = _kmeans_plusplus(X,
                                        n_clusters,
                                        random_state=0,
                                        pairwise_similarity=pairwise_similarity_function,
                                        n_local_trials=None)


    else:
        raise NotImplementedError

    N = len(X)
    n_per_cluster = N // n_clusters
    n_left = N % n_clusters
    for i in trange(iol):
        similarity_matrix = pairwise_similarity_function(centroids, X)
        similarity_matrix = similarity_matrix / similarity_matrix.sum(dim=1, keepdim=True)
        cluster_assignment = torch.zeros(N, dtype=torch.long) - 1
        cluster_size = {c: 0 for c in range(n_clusters)}

        idx = torch.argsort(similarity_matrix.flatten(), descending=True)
        #print(idx)
       
        if n_left == 0:
            for labels in idx:
                labels = labels.item()
                des = labels % N
                label = labels // N
                if cluster_assignment[des] == -1 and cluster_size[label] < n_per_cluster:
                    cluster_assignment[des] = label
                    cluster_size[label] += 1
        else:
            for labels in idx:
                labels = labels.item()
                des = labels % N
                label = labels // N
                if cluster_assignment[des] == -1 and cluster_size[label] < n_per_cluster:
                    cluster_assignment[des] = label
                    cluster_size[label] += 1
                    similarity_matrix[:, des] = -100
            for _ in range(n_left):
                labels = torch.argmax(similarity_matrix).item()
                des = labels % N
                label = labels // N
                cluster_assignment[des] = label
                similarity_matrix[:, des] = -100
                cluster_size[label] += 1
                if cluster_size[label] >= n_per_cluster + 1:
                    similarity_matrix[label, :] = -100

        assert torch.all(cluster_assignment != -1)

        last_centroids = centroids.clone()
        for index in range(n_clusters):
            centroids[index] = X[cluster_assignment == index].mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((centroids - last_centroids) ** 2, dim=1)
            ))

        # update tqdm meter
        if center_shift ** 2 < tol:
            break

    return cluster_assignment.cpu().numpy(), centroids.cpu().numpy()


# def balanced_kmeans1(
#         X,
#         n_clusters,
#         init='k-means++',
#         device=torch.device('cpu'),
#         tol=1e-4,
#         iol=100,
#         distance='cosine'
#
# ):
#     '''
#     X: the clustered feature
#     n_clusters: the cluster number
#     '''
#     # convert to float
#     X = X.float()
#
#     # transfer to device
#     X = X.to(device)
#
#     if distance == 'euclidean':
#         pairwise_similarity_function = partial(pairwise_euclidean, device=device)
#     elif distance == 'cosine':
#         pairwise_similarity_function = partial(pairwise_cosine, device=device)
#     else:
#         raise NotImplementedError
#     # initialize
#     if init == 'random':
#         centroids = initialize(X, n_clusters)
#     elif init == 'k-means++':
#         centroids, _ = _kmeans_plusplus(X,
#                                         n_clusters,
#                                         random_state=0,
#                                         pairwise_distance=pairwise_similarity_function,
#                                         n_local_trials=None)
#
#
#     else:
#         raise NotImplementedError
#
#     # centroids = KMeans(n_clusters=n_clusters)._init_centroids(X.cpu().numpy(), x_squared_norms=None,
#     #                                                              init=init, random_state=np.random.RandomState(seed=0))
#     # centroids = torch.from_numpy(centroids).to(device)
#
#     N = len(X)
#     n_per_cluster = N // n_clusters
#     n_left = N % n_clusters
#     for i in trange(iol):
#         similarity_matrix = pairwise_similarity_function(centroids, X)
#         similarity_matrix = similarity_matrix / similarity_matrix.sum(dim=1, keepdim=True)
#         cluster_assignment = torch.zeros(N) - 1
#         cluster_size = {c: 0 for c in range(n_clusters)}
#
#         if n_left == 0:
#             for _ in range(len(X)):
#                 labels = torch.argmax(similarity_matrix).item()
#                 label = labels // len(X)
#                 des = labels % len(X)
#                 cluster_assignment[des] = label
#                 similarity_matrix[:, des] = -100
#                 cluster_size[label] += 1
#                 if cluster_size[label] >= n_per_cluster:
#                     similarity_matrix[label, :] = -100
#         else:
#             similarity_matrix_clone = similarity_matrix.clone()
#             for _ in range(n_per_cluster * n_clusters):
#                 labels = torch.argmax(similarity_matrix).item()
#                 label = labels // len(X)
#                 des = labels % len(X)
#                 cluster_assignment[des] = label
#                 similarity_matrix[:, des] = -100
#                 similarity_matrix_clone[:, des] = -100
#                 cluster_size[label] += 1
#                 if cluster_size[label] >= n_per_cluster:
#                     similarity_matrix[label, :] = -100
#             for _ in range(n_left):
#                 labels = torch.argmax(similarity_matrix_clone).item()
#                 label = labels // len(X)
#                 des = labels % len(X)
#                 cluster_assignment[des] = label
#                 similarity_matrix_clone[:, des] = -100
#                 cluster_size[label] += 1
#                 if cluster_size[label] >= n_per_cluster + 1:
#                     similarity_matrix_clone[label, :] = -100
#
#         last_centroids = centroids.clone()
#         for index in range(n_clusters):
#             centroids[index] = X[cluster_assignment == index].mean(dim=0)
#
#         center_shift = torch.sum(
#             torch.sqrt(
#                 torch.sum((centroids - last_centroids) ** 2, dim=1)
#             ))
#
#         # update tqdm meter
#         if center_shift ** 2 < tol:
#             break
#
#     return cluster_assignment.cpu().numpy(), centroids.cpu().numpy()


def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    data1, data2 = data1.to(device), data2.to(device)
    A = data1.unsqueeze(dim=1)
    B = data2.unsqueeze(dim=0)
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)
    cosine = A_normalized * B_normalized
    cosine_dis = cosine.sum(dim=-1).squeeze()
    return cosine_dis


def pairwise_euclidean(data1, data2, device=torch.device('cpu')):
    data1, data2 = data1.to(device), data2.to(device)
    A = data1.unsqueeze(dim=1)
    B = data2.unsqueeze(dim=0)
    dis = (A - B) ** 2.0
    dis = 1 / (dis.sum(dim=-1).squeeze() + 1e-4)
    return dis


def initialize(X, num_clusters):
    """
    initialize cluster centers
    """
    num_samples = X.shape[1]
    bs = X.shape[0]

    indices = torch.empty(X.shape[:-1], device=X.device, dtype=torch.long)
    for i in range(bs):
        indices[i] = torch.randperm(num_samples, device=X.device)
    initial_state = torch.gather(X, 1, indices.unsqueeze(-1).repeat(1, 1, X.shape[-1])).reshape(bs, num_clusters, -1, X.shape[-1]).mean(dim=-2)
    return initial_state


def stable_cumsum(arr, dim=None, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum.
    """
    if dim is None:
        arr = arr.flatten()
        dim = 0
    out = torch.cumsum(arr, dim=dim, dtype=torch.float64)
    expected = torch.sum(arr, dim=dim, dtype=torch.float64)
    if not torch.all(torch.isclose(out.take(torch.Tensor([-1]).long().to(arr.device)),
                                   expected, rtol=rtol,
                                   atol=atol, equal_nan=True)):
        warnings.warn('cumsum was found to be unstable: '
                      'its last element does not correspond to sum',
                      RuntimeWarning)
    return out


def _kmeans_plusplus(X,
                     n_clusters,
                     random_state,
                     pairwise_similarity,
                     n_local_trials=None):
    """Computational component for initialization of n_clusters by
    k-means++. Prior validation of data is assumed.
    """
    n_samples, n_features = X.shape

    generator = torch.Generator(device=str(X.device))
    generator.manual_seed(random_state)

    centers = torch.empty((n_clusters, n_features), dtype=X.dtype, device=X.device)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly and track index of point
    #     center_id = random_state.randint(n_samples)
    center_id = torch.randint(n_samples, (1,), generator=generator, device=X.device)

    indices = torch.full((n_clusters,), -1, dtype=torch.int, device=X.device)
    centers[0] = X[center_id]
    indices[0] = center_id

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = 1/pairwise_similarity(
        centers[0, None], X)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        #         rand_vals = random_state.random_sample(n_local_trials) * current_pot
        rand_vals = torch.rand(n_local_trials, generator=generator, device=X.device) * current_pot

        candidate_ids = torch.searchsorted(stable_cumsum(closest_dist_sq),
                                           rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        torch.clip(candidate_ids, None, closest_dist_sq.numel() - 1,
                   out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = 1/pairwise_similarity(
            X[candidate_ids], X)

        # update closest distances squared and potential for each candidate
        torch.minimum(closest_dist_sq, distance_to_candidates,
                      out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(dim=1)

        # Decide which candidate is the best
        best_candidate = torch.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        centers[c] = X[best_candidate]
        indices[c] = best_candidate

    return centers, indices
if __name__ == '__main__':
    X = torch.randn(6, 3)
    cluster_label,_ = balanced_kmean(X,n_clusters=3,init='k-means++',device=torch.device('cpu'),tol=1e-4,iol=100,
        distance='euclidean')
    print(cluster_label)
