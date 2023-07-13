import numpy as np

class PopularitySamplingFromBucketsElicitation:
    def __init__(self, ratings_df, n_buckets, n_samples_per_bucket, k, **kwargs):
        #assert n_buckets == len(n_samples_per_bucket)
        self.ratings_df = ratings_df
        self.n_buckets = n_buckets
        self.n_samples_per_bucket = [n_samples_per_bucket] * n_buckets
        self.k = k
        self.popularities = None
        self.n_users = self.ratings_df.user.unique().size

    def _calculate_item_popularities(self, ratings_df):
        return np.power(ratings_df.groupby("item").count().values[:, 0] / self.n_users, self.k)
        #return np.power(np.sum(rating_matrix > 0.0, axis=0) / rating_matrix.shape[0], self.k)

    def get_initial_data(self, movie_indices_to_ignore=[]):
        if self.popularities is None:
            self.popularities = self._calculate_item_popularities(self.ratings_df)

        if movie_indices_to_ignore:
            self.popularities[np.array(movie_indices_to_ignore)] = 0.0 # This will cause that ignore items wont be sampled
        indices = np.argsort(-self.popularities)
        sorted_popularities = self.popularities[indices]
        sorted_items = np.arange(self.popularities.shape[0])[indices]
        assert sorted_popularities.ndim == sorted_items.ndim

        n_items_total = sum(self.n_samples_per_bucket)
        result = np.zeros((n_items_total, ), dtype=np.int32)

        offset = 0
        for items_bucket, popularities_bucket, n_samples in zip(
            np.array_split(sorted_items, self.n_buckets),
            np.array_split(sorted_popularities, self.n_buckets),
            self.n_samples_per_bucket
        ):
            samples = np.random.choice(items_bucket, size=n_samples, p=popularities_bucket/popularities_bucket.sum(), replace=False)
            result[offset:offset+n_samples] = samples
            offset += n_samples
            
        
        np.random.shuffle(result)
        return result

# Popularity-sampling based implementation of preference elicitation
class PopularitySamplingElicitation:
    
    def __init__(self, ratings_df, n_samples, k, **kwargs):
        self.ratings_df = ratings_df
        self.n_samples = n_samples
        self.k = k
        self.popularities = None
        self.n_users = self.ratings_df.user.unique().size

    def _calculate_item_popularities(self, ratings_df):
        return np.power(ratings_df.groupby("item").count().values[:, 0] / self.n_users, self.k)
        #return np.power(np.sum(rating_matrix > 0.0, axis=0) / rating_matrix.shape[0], self.k)

    # Returns data to be shown to the user
    def get_initial_data(self, movie_indices_to_ignore=[]):
        if self.popularities is None:
            self.popularities = self._calculate_item_popularities(self.ratings_df)
        
        if movie_indices_to_ignore:
            self.popularities[np.array(movie_indices_to_ignore)] = 0.0 # This will cause that ignore items wont be sampled
        p_popularities = self.popularities / self.popularities.sum()
        s = np.random.choice(np.arange(p_popularities.shape[0]), p=p_popularities, size=self.n_samples, replace=False)
        np.random.shuffle(s)
        return s
