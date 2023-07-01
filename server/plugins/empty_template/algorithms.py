from tempfile import TemporaryDirectory
import time
import numpy as np
from plugins.fastcompare.algo.algorithm_base import AlgorithmBase, Parameter, ParameterType

# Tensorflow setting
import tensorflow as tf


from recommenders.datasets.sparse import AffinityMatrix
from recommenders.utils.python_utils import binarize
from recommenders.models.vae.standard_vae import StandardVAE
from recommenders.models.vae.multinomial_vae import Mult_VAE
from recommenders.models.rbm.rbm import RBM


class StandardVaeWrapper(AlgorithmBase):
    # VAE parameters
    TOP_K = 100

    # Model parameters
    INTERMEDIATE_DIM = 200
    BATCH_SIZE = 100
    EVAL_K = 10

    MIN_POSITIVE_RATING = 3.5

    @staticmethod
    def _vae_loss(x, x_bar, original_dim, z_log_var, z_mean, beta):
        """Calculate negative ELBO (NELBO)."""
        # Reconstruction error: logistic log likelihood
        reconst_loss = original_dim * tf.keras.losses.binary_crossentropy(x, x_bar)

        # Kullback–Leibler divergence
        kl_loss = 0.5 * tf.keras.backend.sum(
            -1 - z_log_var + tf.keras.backend.square(z_mean) + tf.keras.backend.exp(z_log_var), axis=-1
        )

        return reconst_loss + beta * kl_loss

    def __init__(self, loader, epochs, latent_dim, beta, **kwargs):
        self.epochs = epochs
        self.latent_dim = latent_dim

        #self.rating_matrix = loader.ratings_df.pivot(index='user', columns='item', values="rating").fillna(0).values
        self.am_train = AffinityMatrix(df=loader.ratings_df.rename(columns={"user": "userID", "item": "itemID"}), items_list=loader.ratings_df.item.unique())
        self.rating_matrix, _, _ = self.am_train.gen_affinity_matrix()
        self.rating_matrix = binarize(self.rating_matrix, StandardVaeWrapper.MIN_POSITIVE_RATING)
        #self.all_items = np.arange(self.rating_matrix.shape[1]) # loader.ratings_df.item.unique()
        self.model = StandardVAE(n_users=self.rating_matrix.shape[0], # Number of unique users in the training set
                                   original_dim=self.rating_matrix.shape[1], # Number of unique items in the training set
                                   intermediate_dim=StandardVaeWrapper.INTERMEDIATE_DIM, 
                                   latent_dim=self.latent_dim, 
                                   n_epochs=self.epochs, 
                                   batch_size=StandardVaeWrapper.BATCH_SIZE, 
                                   k=StandardVaeWrapper.EVAL_K,
                                   verbose=1,
                                   save_path=TemporaryDirectory().name,
                                   drop_encoder=0.5,
                                   drop_decoder=0.5,
                                   annealing=False,
                                   beta=beta
                                   )
        
        # https://stackoverflow.com/questions/68754599/tensorflow-use-input-in-loss-function
        self.model.model.add_loss( StandardVaeWrapper._vae_loss( 
                            self.model.x, self.model.x_decoded,
                            self.model.original_dim, self.model.z_log_var,
                            self.model.z_mean, self.model.beta
                    ) )
        # Try legacy adam tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
        self.model.model.compile(optimizer=self.model.model.optimizer, loss=None)
        
        
        
        #self.am_train.gen_affinity_matrix()
        
    def fit(self):
        start_time = time.perf_counter()
        self.model.fit(
            x_train=self.rating_matrix,
            x_valid=self.rating_matrix[:1], # We do not care about validation results
            x_val_tr=self.rating_matrix[:1],
            x_val_te=self.rating_matrix[:1],
            mapper=self.am_train # We do not care about Metrics evaluation and mapper is only used in metrics, so pass-in whatever that will make this call to pass
        )
        print(f"Fitting took: {time.perf_counter() - start_time}")


    def predict(self, selected_items, filter_out_items, k):
        #x = np.zeros(shape=(self.all_items.size, ), dtype=np.int32)
        #x[np.unique(selected_items + filter_out_items)] = 1
        
        # res = self.model.recommend_k_items(np.expand_dims(x, axis=0), k, remove_seen=True)[0]
        # return np.argsort(-res)[:k].tolist()
        
        x = np.zeros(shape=(len(self.am_train.map_items), ), dtype=np.int32)
        x[np.unique(list(map(lambda y: self.am_train.map_items[y], selected_items + filter_out_items)))] = 1
        x = np.expand_dims(x, axis=0)
        score = self.model.model.predict(x)
        seen_mask = np.not_equal(x, 0)
        score[seen_mask] = 0
        return [self.am_train.map_back_items[z] for z in np.argpartition(-score, range(k), axis=1)[0, :k]]

    @classmethod
    def name(cls):
        return "StandardVAE"

    @classmethod
    def parameters(cls):
        return [
            Parameter("epochs", ParameterType.INT, 50, help="Number of epochs"),
            Parameter("latent_dim", ParameterType.INT, 70, help="Latent space dimension"),
            Parameter("beta", ParameterType.FLOAT, 1.0, help="Beta in VAE loss")
        ]
    
    def save(self, instance_cache_path, class_cache_path):
        self.model.save_path = instance_cache_path
        self.model.model.save_weights(instance_cache_path)

    def load(self, instance_cache_path, class_cache_path):
        self.model.save_path = instance_cache_path
        self.model.model.load_weights(instance_cache_path)




class MultVaeWrapper(AlgorithmBase):
    # VAE parameters
    TOP_K = 100

    # Model parameters
    INTERMEDIATE_DIM = 200
    BATCH_SIZE = 100
    EVAL_K = 10

    MIN_POSITIVE_RATING = 3.5

    @staticmethod
    def _mult_vae_loss(x, x_bar, z_log_var, z_mean, beta):
        """Calculate negative ELBO (NELBO)."""
        log_softmax_var = tf.nn.log_softmax(x_bar)
        neg_ll = -tf.reduce_mean(
            input_tensor=tf.reduce_sum(input_tensor=log_softmax_var * x, axis=-1)
        )
        a = tf.keras.backend.print_tensor(neg_ll)  # noqa: F841
        # calculate positive Kullback–Leibler divergence  divergence term
        kl_loss = tf.keras.backend.mean(
            0.5
            * tf.keras.backend.sum(
                -1 - z_log_var + tf.keras.backend.square(z_mean) + tf.keras.backend.exp(z_log_var),
                axis=-1,
            )
        )

        # obtain negative ELBO
        neg_ELBO = neg_ll + beta * kl_loss

        return neg_ELBO

    def __init__(self, loader, epochs, latent_dim, beta, **kwargs):
        self.epochs = epochs
        self.latent_dim = latent_dim

        self.am_train = AffinityMatrix(df=loader.ratings_df.rename(columns={"user": "userID", "item": "itemID"}), items_list=loader.ratings_df.item.unique())
        self.rating_matrix, _, _ = self.am_train.gen_affinity_matrix()
        self.rating_matrix = binarize(self.rating_matrix, MultVaeWrapper.MIN_POSITIVE_RATING)
        
        self.model = Mult_VAE(n_users=self.rating_matrix.shape[0], # Number of unique users in the training set
                                original_dim=self.rating_matrix.shape[1], # Number of unique items in the training set
                                intermediate_dim=MultVaeWrapper.INTERMEDIATE_DIM, 
                                latent_dim=self.latent_dim, 
                                n_epochs=self.epochs, 
                                batch_size=MultVaeWrapper.BATCH_SIZE, 
                                k=MultVaeWrapper.TOP_K,
                                verbose=1,
                                save_path=TemporaryDirectory().name,
                                drop_encoder=0.5,
                                drop_decoder=0.5,
                                annealing=False,
                                beta=beta
                                )


        # https://stackoverflow.com/questions/68754599/tensorflow-use-input-in-loss-function
        self.model.model.add_loss( MultVaeWrapper._mult_vae_loss( 
                            self.model.x, self.model.x_decoded,
                            self.model.z_log_var,
                            self.model.z_mean, self.model.beta
                    ) )
        # Try legacy adam tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
        self.model.model.compile(optimizer=self.model.model.optimizer, loss=None)
        
        
        
        #self.am_train.gen_affinity_matrix()
        
    def fit(self):
        start_time = time.perf_counter()
        self.model.fit(
            x_train=self.rating_matrix,
            x_valid=self.rating_matrix[:1], # We do not care about validation results
            x_val_tr=self.rating_matrix[:1],
            x_val_te=self.rating_matrix[:1],
            mapper=self.am_train # We do not care about Metrics evaluation and mapper is only used in metrics, so pass-in whatever that will make this call to pass
        )
        print(f"Fitting took: {time.perf_counter() - start_time}")


    def predict(self, selected_items, filter_out_items, k):
        x = np.zeros(shape=(len(self.am_train.map_items), ), dtype=np.int32)
        x[np.unique(list(map(lambda y: self.am_train.map_items[y], selected_items + filter_out_items)))] = 1
        x = np.expand_dims(x, axis=0)
        score = self.model.model.predict(x)
        seen_mask = np.not_equal(x, 0)
        score[seen_mask] = 0
        return [self.am_train.map_back_items[z] for z in np.argpartition(-score, range(k), axis=1)[0, :k]]

    @classmethod
    def name(cls):
        return "MultVAE"

    @classmethod
    def parameters(cls):
        return [
            Parameter("epochs", ParameterType.INT, 50, help="Number of epochs"),
            Parameter("latent_dim", ParameterType.INT, 70, help="Latent space dimension"),
            Parameter("beta", ParameterType.FLOAT, 1.0, help="Beta in VAE loss")
        ]
    
    def save(self, instance_cache_path, class_cache_path):
        self.model.save_path = instance_cache_path
        self.model.model.save_weights(instance_cache_path)

    def load(self, instance_cache_path, class_cache_path):
        self.model.save_path = instance_cache_path
        self.model.model.load_weights(instance_cache_path)



class RbmWrapper(AlgorithmBase):
    
    BATCH_SIZE = 350

    def __init__(self, loader, epochs, hidden_units, **kwargs):
        self.epochs = epochs
        
        self.am_train = AffinityMatrix(df=loader.ratings_df.rename(columns={"user": "userID", "item": "itemID"}), items_list=loader.ratings_df.item.unique())
        self.rating_matrix, _, _ = self.am_train.gen_affinity_matrix()

        self.model = RBM(
            possible_ratings=np.setdiff1d(np.unique(self.rating_matrix), np.array([0])),
            visible_units=self.rating_matrix.shape[1],
            hidden_units=hidden_units,
            training_epoch=epochs,
            minibatch_size=RbmWrapper.BATCH_SIZE,
            with_metrics=False
        )

    def fit(self):
        start_time = time.perf_counter()
        self.model.fit(self.rating_matrix)
        print(f"Fitting took: {time.perf_counter() - start_time}")


    def predict(self, selected_items, filter_out_items, k):
        x = np.zeros(shape=(len(self.am_train.map_items), ), dtype=np.int32)
        x[np.unique(list(map(lambda y: self.am_train.map_items[y], selected_items + filter_out_items)))] = self.rating_matrix.max()
        x = np.expand_dims(x, axis=0)

        v_, pvh_ = self.model.eval_out()
        vp, pvh = self.model.sess.run([v_, pvh_], feed_dict={self.model.vu: x})
        pv = np.max(pvh, axis=2)

        # evaluate the score
        score = np.multiply(vp, pv)
        # Remove seen
        seen_mask = np.not_equal(x, 0)
        vp[seen_mask] = 0
        pv[seen_mask] = 0
        score[seen_mask] = 0

        return [self.am_train.map_back_items[z] for z in np.argpartition(-score, range(k), axis=1)[0, :k]]

    @classmethod
    def name(cls):
        return "RBM"

    @classmethod
    def parameters(cls):
        return [
            Parameter("epochs", ParameterType.INT, 30, help="Number of epochs"),
            Parameter("hidden_units", ParameterType.INT, 1200, help="Number of hidden units")
        ]
    
    def save(self, instance_cache_path, class_cache_path):
        self.model.save(instance_cache_path)

    def load(self, instance_cache_path, class_cache_path):
        self.model.load(instance_cache_path)
        self.model.init_training_session(self.rating_matrix)