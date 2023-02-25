
from plugins.fastcompare.algo.algorithm_base import Parameter, ParameterType, PreferenceElicitationBase
from plugins.utils.multi_obj_sampling import MultiObjectiveSamplingFromBucketsElicitation
from plugins.utils.popularity_sampling import PopularitySamplingElicitation, PopularitySamplingFromBucketsElicitation

from scipy.spatial.distance import squareform, pdist

class MultiObjectiveSamplingFromBucketsElicitationWrapper(PreferenceElicitationBase):
    # Objectives is a dictionary mapping objective name to its implementation (e.g. we can use different implementations of diversity etc..)
    def __init__(self, loader, *args, **kwargs):
        self.ratings_df = loader.ratings_df
        self.elicitation = None
        self.args = args
        self.kwargs = kwargs

    def get_initial_data(self, movie_indices_to_ignore=[]):
        return self.elicitation.get_initial_data(movie_indices_to_ignore)

    def fit(self):
        if "rating" not in self.ratings_df:
            self.ratings_df.loc[:, "rating"] = 1
        rating_matrix = self.ratings_df.pivot(index='user', columns='item', values="rating").fillna(0).values
        similarity_matrix = np.float32(squareform(pdist(rating_matrix.T, "cosine")))
        self.elicitation = MultiObjectiveSamplingFromBucketsElicitation(rating_matrix, similarity_matrix, *self.args, **self.kwargs)

    @classmethod
    def name(cls):
        return "Multi Objective Sampling from buckets"

    @classmethod
    def parameters(cls):
        return [
            Parameter("n_relevance_buckets", ParameterType.INT, 2, help_key="n_relevance_buckets"),
            Parameter("n_diversity_buckets", ParameterType.INT, 2, help_key="n_diversity_buckets"),
            Parameter("n_novelty_buckets", ParameterType.INT, 2, help_key="n_novelty_buckets"),
            Parameter("n_samples_per_bucket", ParameterType.INT, 4, help_key="n_samples_per_bucket"),
        ]

import numpy as np
from plugins.fastcompare.algo.algorithm_base import PreferenceElicitationBase, Parameter, ParameterType

class PopularitySamplingFromBucketsElicitationWrapper(PreferenceElicitationBase):
    def __init__(self, loader, *args, **kwargs):
        self.elicitation = PopularitySamplingFromBucketsElicitation(loader.ratings_df, *args, **kwargs)
    
    def get_initial_data(self, movie_indices_to_ignore=[]):
        return self.elicitation.get_initial_data(movie_indices_to_ignore)

    def fit(self):
        # This elicitation is light weight, there is no expensive internal state that would have to be prepared upfront
        pass

    @classmethod
    def name(cls):
        return "Popularity Sampling from buckets"

    @classmethod
    def parameters(cls):
        return [
            Parameter("n_buckets", ParameterType.INT, 5, help_key="n_buckets"),
            Parameter("n_samples_per_bucket", ParameterType.INT, 4, help_key="n_samples_per_bucket"),
            Parameter("k", ParameterType.FLOAT, 1.0, help_key="exp_k")
        ]

# Popularity-sampling based implementation of preference elicitation
class PopularitySamplingElicitationWrapper(PreferenceElicitationBase):
    
    def __init__(self, loader, *args, **kwargs):
        self.elicitation = PopularitySamplingElicitation(loader.ratings_df, *args, **kwargs)

    def _calculate_item_popularities(self, rating_matrix):
        return np.power(np.sum(rating_matrix > 0.0, axis=0) / rating_matrix.shape[0], self.k)

    def fit(self):
        # This elicitation is light weight, there is no expensive internal state that would have to be prepared upfront
        pass

    def get_initial_data(self, movie_indices_to_ignore=[]):
        return self.elicitation.get_initial_data(movie_indices_to_ignore)

    @classmethod
    def name(cls):
        return "Popularity Sampling"

    @classmethod
    def parameters(cls):
        return [
            Parameter("n_samples", ParameterType.INT, 10, help_key="n_samples"),
            Parameter("k", ParameterType.FLOAT, 1.0, help_key="exp_k")
        ]