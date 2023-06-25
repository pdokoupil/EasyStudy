import os
import numpy as np
from plugins.fastcompare.algo.algorithm_base import AlgorithmBase, PreferenceElicitationBase, DataLoaderBase, Parameter, ParameterType

# Tensorflow setting
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import keras

# Microsoft Recommenders includes
from recommenders.utils.timer import Timer
from recommenders.datasets import movielens
from recommenders.datasets.split_utils import min_rating_filter_pandas
from recommenders.datasets.python_splitters import numpy_stratified_split
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.utils.constants import SEED as DEFAULT_SEED

from recommenders.datasets.sparse import AffinityMatrix
from recommenders.utils.python_utils import binarize
from recommenders.models.vae.standard_vae import StandardVAE



class BasicVAE(AlgorithmBase):
    # VAE parameters
    TOP_K = 100

    # Select MovieLens data size: 100k, 1m, 10m, or 20m
    MOVIELENS_DATA_SIZE = '1m'

    # Model parameters
    HELDOUT_USERS = 600 # CHANGE FOR DIFFERENT DATASIZE
    INTERMEDIATE_DIM = 200
    LATENT_DIM = 70
    EPOCHS = 10 #400
    BATCH_SIZE = 100
    EVAL_K = 10

    # temporary Path to save the optimal model's weights
    tmp_dir = './tmp_dir' #TemporaryDirectory().name
    WEIGHTS_PATH = os.path.join(tmp_dir, "svae_weights.hdf5")

    SEED = 98765


    def __init__(self, loader, **kwargs):
        self.all_items = loader.ratings_df.item.unique()

        self.model = StandardVAE(n_users=loader.rating_matrix.shape[0], # Number of unique users in the training set
                                   original_dim=loader.rating_matrix.shape[1], # Number of unique items in the training set
                                   intermediate_dim=BasicVAE.INTERMEDIATE_DIM, 
                                   latent_dim=BasicVAE.LATENT_DIM, 
                                   n_epochs=BasicVAE.EPOCHS, 
                                   batch_size=BasicVAE.BATCH_SIZE, 
                                   k=BasicVAE.EVAL_K,
                                   verbose=0,
                                   seed=BasicVAE.SEED,
                                   save_path=BasicVAE.WEIGHTS_PATH,
                                   drop_encoder=0.5,
                                   drop_decoder=0.5,
                                   annealing=False,
                                   beta=1.0
                                   )
        
        self.am_train = AffinityMatrix(df=self.loader.train_df, items_list=self.all_items)
    
    def fit(self):
        self.model.fit(
            x_train=self.loader.rating_matrix,
            x_valid=self.loader.rating_matrix[:1], # We do not care about validation results
            x_val_tr=self.loader.rating_matrix[:1],
            x_val_te=self.loader.rating_matrix[:1],
            mapper=self.am_train # We do not care about Metrics evaluation and mapper is only used in metrics, so pass-in whatever that will make this call to pass
        )
        


    def predict(self, selected_items, filter_out_items, k):
        return None #self.model.
        #candidates = np.setdiff1d(self.all_items, filter_out_items)
        #candidates = np.setdiff1d(candidates, selected_items)
        #return np.random.choice(candidates, size=k)

    @classmethod
    def name(cls):
        return "Basic VAE"

    @classmethod
    def parameters(cls):
        return [
            
        ]
    
    def save(self, instance_cache_path, class_cache_path):
        self.model.model.save_weights(instance_cache_path)

    def load(self, instance_cache_path, class_cache_path):
        self.model.model.load_weights(instance_cache_path)

            