import os
import pickle

import numpy as np

from support.rating_based_relevance_support import rating_based_relevance_support
from support.intra_list_diversity_support import intra_list_diversity_support
from support.popularity_complement_support import popularity_complement_support

def get_supports(users_partial_lists, items, extended_rating_matrix, distance_matrix, users_viewed_item, k):
    rel_supps = rating_based_relevance_support(extended_rating_matrix)
    div_supps = intra_list_diversity_support(users_partial_lists, items, distance_matrix, k)
    nov_supps = popularity_complement_support(users_viewed_item, num_users=users_partial_lists.shape[0])
    return np.stack([rel_supps, div_supps, nov_supps])

class RLPropWrapper:
    def __init__(
        self, items, extended_rating_matrix, distance_matrix, users_viewed_item,
        normalization_factory, mandate_allocation, unseen_items_mask, cache_dir, discount_sequences,
        n_users
    ):
        
        self.cache_dir = cache_dir or "" # None gets replaced by ""
        self.mandate_allocation = mandate_allocation
        self.discount_sequences = discount_sequences
        self.items = items
        self.extended_rating_matrix = extended_rating_matrix
        self.distance_matrix = distance_matrix
        self.users_viewed_item = users_viewed_item
        self.normalization_factory = normalization_factory
        self.unseen_items_mask = unseen_items_mask
        self.seed = 42
        self.shift = 0.0
        self.baseline = "baseline"
        self.diversity = "cf"
        self.n_users = n_users

    def _load_cache(self, cache_path):
        print(f"Loading cache from: {cache_path}")
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        return cache

    def _save_cache(self, cache_path, cache):
        print(f"Saving cache to: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)

    def _build_normalization(self, shift):
        if shift:
            return self.normalization_factory(shift)
        else:
            return self.normalization_factory()

    def _prepare_normalization(self):
        cache_path = os.path.join(self.cache_dir, f"sup_norm_{self.normalization_factory.__name__}_{self.shift}_{self.seed}_{self.baseline}_{self.diversity}.pckl")
        if self.cache_dir and os.path.exists(cache_path):
            cache = self._load_cache(cache_path)
            norm_relevance = cache["norm_relevance"]
            norm_diversity = cache["norm_diversity"]
            norm_novelty = cache["norm_novelty"]
        else:
            relevance_data_points = self.extended_rating_matrix.T
            
            upper_triangular_indices = np.triu_indices(self.distance_matrix.shape[0], k=1)
            upper_triangular_nonzero = self.distance_matrix[upper_triangular_indices]
                
            diversity_data_points = np.expand_dims(upper_triangular_nonzero, axis=1)
            novelty_data_points = np.expand_dims(1.0 - self.users_viewed_item / self.n_users, axis=1)

            norm_relevance = self._build_normalization(self.shift)
            norm_relevance.train(relevance_data_points)
            norm_diversity = self._build_normalization(self.shift)
            norm_diversity.train(diversity_data_points)
            norm_novelty = self._build_normalization(self.shift)
            norm_novelty.train(novelty_data_points)
            if self.cache_dir:
                cache = {
                    "norm_relevance": norm_relevance,
                    "norm_diversity": norm_diversity,
                    "norm_novelty": norm_novelty
                }
                self._save_cache(cache_path, cache)

        return [norm_relevance, norm_diversity, norm_novelty]

    def init(self):
        self.normalizations = self._prepare_normalization()
    def __call__(self, k, shuffle=True):
        # Assume recommending for a single user
        users_partial_lists = np.full((self.extended_rating_matrix.shape[0], k), -1, dtype=np.int32)

        # Masking already recommended users and SEEN items
        mask = self.unseen_items_mask.copy()
        for i in range(k):
            # Calculate support values
            supports = get_supports(users_partial_lists, self.items, self.extended_rating_matrix, self.distance_matrix, self.users_viewed_item, k=i+1)
            
            # Normalize the supports
            assert supports.shape[0] == 3, "expecting 3 objectives, if updated, update code below"
            
            supports[0, :, :] = self.normalizations[0](supports[0].T).T * self.discount_sequences[0][i]
            supports[1, :, :] = self.normalizations[1](supports[1].reshape(-1, 1)).reshape((supports.shape[1], -1)) * self.discount_sequences[1][i]
            supports[2, :, :] = self.normalizations[2](supports[2].reshape(-1, 1)).reshape((supports.shape[1], -1)) * self.discount_sequences[2][i]
            
            # Mask out the already recommended items
            np.put_along_axis(mask, users_partial_lists[:, :i], 0, 1)

            # Get the per-user top-k recommendations
            users_partial_lists[:, i] = self.mandate_allocation(mask, supports)

        if shuffle:
            np.random.shuffle(users_partial_lists.T)

        return users_partial_lists