# Lenskit wrapper
# Ensure this code work even if in different directory (so that others who whill add new wrappers for different frameworks in the future have a freedom where to put that code?)

# !! TODO note that there is a restriction such that all python files containing algorithm declarations should be placed in module directory (i.e. __init__.py should be present at their level)
from abc import ABC, abstractmethod
import json
import numpy as np

import pandas as pd
from plugins.fastcompare.algo.algorithm_base import AlgorithmBase, Parameter, ParameterType
from plugins.fastcompare.loading import is_algorithm

from lenskit.algorithms import als, item_knn, user_knn, Recommender, basic

class Algo55(AlgorithmBase):
    def fit(self):
        pass

    def predict(self, selected_items, filter_out_items, k):
        pass

class LenskitWrapper(AlgorithmBase, ABC):
    def __init__(self, loader, algo, **kwargs):
        self.ratings_df = loader.ratings_df
        self.algo = Recommender.adapt(algo)
        self.all_items = self.ratings_df.item.unique()
    
    # One-time fitting of the algorithm for a predefined number of iterations
    def fit(self):
        self.algo = self.algo.fit(self.ratings_df) # Todo make this generic assumption and let all load_ml_dataset and other wrappers adhere to this notation
        self.new_user = self.ratings_df.user.max() + 1

    # Predict for the user, we can also pass-in ratings to update user's preferences (a sort of fine-tune)
    def predict(self, selected_items, filter_out_items, k):
        rat = pd.DataFrame({"item" :selected_items}).set_index("item", drop=False)
        # Appropriately filter out what was seen and what else should be filtered
        candidates = np.setdiff1d(self.all_items, rat.item.unique())
        candidates = np.setdiff1d(candidates, filter_out_items)
        if not selected_items:
            # Nothing was selected, since the new_user was unknown during training, Lenskit algorithm would simply recommended nothing
            # to avoid empty recommendation, we just sample random candidates
            return np.random.choice(candidates, size=k, replace=False).tolist()
        return self.algo.recommend(user=self.new_user, candidates=candidates, n=k, ratings=rat.item).item.tolist()


class MostPopularItem(LenskitWrapper):
    def __init__(self, loader, **kwargs):
        super().__init__(loader, basic.PopScore(**kwargs))
    
    @classmethod
    def name(cls):
        return "Most Popular Items Baseline"

    @classmethod
    def parameters(cls):
        return []
    
class RandomItem(LenskitWrapper):
    def __init__(self, loader, **kwargs):
        super().__init__(loader, basic.Random(**kwargs))

    @classmethod
    def name(cls):
        return "Random Items Baseline"
    
    @classmethod
    def parameters(cls):
        return []

class ImplicitMF(LenskitWrapper):
    def __init__(self, loader, **kwargs):
        super().__init__(loader, als.ImplicitMF(**kwargs))
    
    @classmethod
    def name(cls):
        return "Implicit Matrix Factorization"

    @classmethod
    def parameters(cls):
        return [
            Parameter("features", ParameterType.INT, 50, help_key="implicit_mf_features_help"),
            Parameter("iterations", ParameterType.INT, 20, help_key="implicit_mf_iterations_help"),
        ]

class UserKNN(LenskitWrapper):
    def __init__(self, loader, **kwargs):
        super().__init__(loader, user_knn.UserUser(**kwargs))
    
    @classmethod
    def name(cls):
        return "UserKNN"

    @classmethod
    def parameters(cls):
        return [
            Parameter("nnbrs", ParameterType.INT, None, help_key="user_knn_nnbrs_help"),
            Parameter("feedback", ParameterType.OPTIONS, "implicit", help_key="user_knn_feedback_help", options=["implicit"]),
        ]

class ItemKNN(LenskitWrapper):
    def __init__(self, loader, **kwargs):
        super().__init__(loader, item_knn.ItemItem(**kwargs))
    
    @classmethod
    def name(cls):
        return "ItemKNN"

    @classmethod
    def parameters(cls):
        return [
            Parameter("nnbrs", ParameterType.INT, None, help_key="item_knn_nnbrs_help"),
            Parameter("feedback", ParameterType.OPTIONS, "implicit", help_key="item_knn_feedback_help", options=["implicit"]),
            Parameter("param_1", ParameterType.BOOL, True, "Some bool parameter"),
            Parameter("param_2", ParameterType.FLOAT, 3.14, "Some float parameter"),
            Parameter("param_3", ParameterType.STRING, "Hello world 345", "Some string parameter"),
            Parameter("param_4", ParameterType.OPTIONS, True, "Some options simulating True/False single choice", options=[True, False])
        ]