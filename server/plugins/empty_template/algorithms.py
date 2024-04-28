import numpy as np
from plugins.fastcompare.algo.algorithm_base import AlgorithmBase, PreferenceElicitationBase, DataLoaderBase, Parameter, ParameterType


# class SomeAlgorithm(AlgorithmBase):
#     def __init__(self, loader, **kwargs):
#         self.all_items = loader.ratings_df.item.unique()
    
#     def fit(self):
#         pass

#     def predict(self, selected_items, filter_out_items, k):
#         candidates = np.setdiff1d(self.all_items, filter_out_items)
#         candidates = np.setdiff1d(candidates, selected_items)
#         return np.random.choice(candidates, size=k)

#     @classmethod
#     def name(cls):
#         return "Some algorithm unique name"

#     @classmethod
#     def parameters(cls):
#         return [
            
#         ]
    
# class SomePreferenceElicitation(PreferenceElicitationBase):
#     def __init__(self, loader, n_items, **kwargs):
#         self.all_items = loader.ratings_df.item.unique()
#         self.n_items = n_items

#     def get_initial_data(self, movie_indices_to_ignore=[]):
#         candidates = np.setdiff1d(self.all_items, movie_indices_to_ignore)
#         return np.random.choice(candidates, size=self.n_items)

#     def fit(self):
#         pass

#     @classmethod
#     def name(cls):
#         return "Random (uniform) sampling elicitation"

#     @classmethod
#     def parameters(cls):
#         return [
#             Parameter("n_items", ParameterType.INT, 42, "Help text")
#         ]
    

            