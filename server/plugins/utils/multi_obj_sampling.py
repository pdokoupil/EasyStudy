import numpy as np

# Samples from N buckets where each bucket corresponds to a single objective
class MultiObjectiveSamplingFromBucketsElicitation:
    # Objectives is a dictionary mapping objective name to its implementation (e.g. we can use different implementations of diversity etc..)
    def __init__(self, rating_matrix, similarity_matrix, n_relevance_buckets, n_diversity_buckets, n_novelty_buckets, n_samples_per_bucket, k=1.0, **kwargs):
        
        self.rating_matrix = rating_matrix
        self.similarity_matrix = similarity_matrix
        self.n_buckets = {
            "relevance": n_relevance_buckets,
            "diversity": n_diversity_buckets,
            "novelty": n_novelty_buckets
        }
        self.n_samples_per_bucket = {
            "relevance": [n_samples_per_bucket] * n_relevance_buckets,
            "diversity": [n_samples_per_bucket] * n_diversity_buckets,
            "novelty": [n_samples_per_bucket] * n_novelty_buckets
        }
        self.k = k

    def _calculate_item_popularities(self, rating_matrix):
        return np.power(np.sum(rating_matrix > 0.0, axis=0) / rating_matrix.shape[0], self.k)

    # Relevances are simply mean ratings of each item
    def _calculate_item_relevances(self, rating_matrix):
        return np.power(rating_matrix.mean(axis=0), self.k) # Beware that zeroes (non-rated) items are included as well

    # Novelties are inverse popularity
    def _calculate_item_novelties(self, rating_matrix):
        return -self._calculate_item_popularities(rating_matrix)

    def get_initial_data(self, movie_indices_to_ignore=[]):

        # We first sample relevance bucket
        # Then we sample novelty bucket
        # And in the very end, we CALCULATE (based on already sampled items) diversity and sample diversity bucket

        movie_indices_to_ignore_np = np.array(movie_indices_to_ignore)

        relevances = self._calculate_item_relevances(self.rating_matrix)
        novelties = self._calculate_item_novelties(self.rating_matrix)

        if movie_indices_to_ignore:
            relevances[movie_indices_to_ignore_np] = 0.0 # This will cause that ignore items wont be sampled
            novelties[movie_indices_to_ignore_np] = 0.0 # This will cause that ignore items wont be sampled

        relevance_indices = np.argsort(-relevances)
        sorted_relevances = relevances[relevance_indices]
        sorted_items_by_relevance = np.arange(relevances.shape[0])[relevance_indices]
        assert sorted_relevances.ndim == sorted_items_by_relevance.ndim

        n_items_total = sum([sum(l) for l in self.n_samples_per_bucket.values()])
        result = np.zeros((n_items_total, ), dtype=np.int32)
        extra_data = []
        bucket_idx = 0

        offset = 0

        # Fill in relevance buckets
        for items_bucket, relevances_bucket, n_samples in zip(
            np.array_split(sorted_items_by_relevance, self.n_buckets["relevance"]),
            np.array_split(sorted_relevances, self.n_buckets["relevance"]),
            self.n_samples_per_bucket["relevance"]
        ):
            samples = np.random.choice(items_bucket, size=n_samples, p=relevances_bucket/relevances_bucket.sum(), replace=False)
            result[offset:offset+n_samples] = samples
            offset += n_samples

            extra_data.extend([f"relevance_bucket with idx={bucket_idx + 1}/{self.n_buckets['relevance']}"] * n_samples)
            bucket_idx += 1


        # Zero everything selected in relevance sampling
        novelties[result[:offset]] = 0.0 # This will cause that ignore items wont be sampled

        novelty_indices = np.argsort(-novelties)
        sorted_novelties = novelties[novelty_indices]
        sorted_items_by_novelty = np.arange(novelties.shape[0])[novelty_indices]
        assert sorted_novelties.ndim == sorted_items_by_novelty.ndim

        bucket_idx = 0

        # Fill in novelty buckets
        for items_bucket, novelties_bucket, n_samples in zip(
            np.array_split(sorted_items_by_novelty, self.n_buckets["novelty"]),
            np.array_split(sorted_novelties, self.n_buckets["novelty"]),
            self.n_samples_per_bucket["novelty"]
        ):
            samples = np.random.choice(items_bucket, size=n_samples, p=novelties_bucket/novelties_bucket.sum(), replace=False)
            result[offset:offset+n_samples] = samples
            offset += n_samples

            extra_data.extend([f"novelty_bucket with idx={bucket_idx + 1}/{self.n_buckets['novelty']}"] * n_samples)
            bucket_idx += 1

        # Set selected so far to use it as a filter below
        selected_so_far = result[:offset]

        # Calculate diversities
        #similarity_matrix = np.float32(squareform(pdist(self.rating_matrix.T, "cosine")))
        distance_matrix = 1.0 - self.similarity_matrix

        accums = np.add.accumulate(self.n_samples_per_bucket["diversity"])

        # For the total number of items we have to sample across all diversity buckets
        for i in range(sum(self.n_samples_per_bucket["diversity"])):
            # Compute "approximate" diversity of each item to the list we have so far
            diversities = distance_matrix[result[:offset]].sum(axis=0)
            if movie_indices_to_ignore:
                diversities[movie_indices_to_ignore_np] = 0.0 # This will cause that ignore items wont be sampled
            diversities[selected_so_far] = 0.0 # Filter out movies selected so far
            diversities /= diversities.sum() # Normalize to 1
            
            # Prepare buckets based on diversities of all items w.r.t. CURRENT set of sampled items
            diversity_indices = np.argsort(-diversities)
            sorted_diversities = diversities[diversity_indices]
            sorted_items_by_diversity = np.arange(diversities.shape[0])[diversity_indices]
            assert sorted_diversities.ndim == sorted_items_by_diversity.ndim

            # Find the corresponding bucket for the given item
            current_target_bucket = np.searchsorted(accums, i, "right")

            items_bucket = np.array_split(sorted_items_by_diversity, self.n_buckets["diversity"])[current_target_bucket]
            diversities_bucket = np.array_split(sorted_diversities, self.n_buckets["diversity"])[current_target_bucket]
            
            result[offset:offset+1] = np.random.choice(items_bucket, size=1, p=diversities_bucket/diversities_bucket.sum(), replace=False)
            offset += 1

            extra_data.append(f"diversity_bucket with idx={current_target_bucket + 1}/{self.n_buckets['diversity']}")

        # # For each diversity bucket
        # for n_samples in self.n_samples_per_bucket["diversity"]:
        #     # For each item in individual bucket
        #     for _ in range(n_samples):
        #         # Compute "approximate" diversity of each item to the list we have so far
        #         diversities = distance_matrix[result[:offset]].sum(axis=0)
        #         diversities /= diversities.sum() # Normalize to 1
        #         result[offset:offset+1] = np.random.choice()
        #         offset += 1
        
        # diversities = self._calculate_item_diversities(self.rating_matrix, result[:offset])
        
        # diversity_indices = np.argsort(-diversities)
        # sorted_diversities = diversities[diversity_indices]
        # sorted_items_by_diversity = np.arange(diversities.shape[0])[diversity_indices]
        # assert sorted_diversities.ndim == sorted_items_by_diversity.ndim

        # # Fill in diversity buckets
        # for items_bucket, diversities_bucket, n_samples in zip(
        #     np.array_split(sorted_items_by_diversity, self.n_buckets["diversity"]),
        #     np.array_split(sorted_diversities, self.n_buckets["diversity"]),
        #     self.n_samples_per_bucket["diversity"]
        # ):
        #     samples = np.random.choice(items_bucket, size=n_samples, p=diversities_bucket/diversities_bucket.sum(), replace=False)
        #     result[offset:offset+n_samples] = samples
        #     offset += n_samples
        
        #np.random.shuffle(result)
        assert len(extra_data) == result.shape[0], f"{len(extra_data)}!={result.shape[0]}"
        assert result.shape[0] == n_items_total, f"{result.shape[0]}!={n_items_total}"

        p = np.random.permutation(len(extra_data))
        extra_data = np.array(extra_data)
        return result[p] #, extra_data[p]