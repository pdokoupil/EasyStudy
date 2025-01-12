import functools
import itertools
import json
import pickle
import random
import secrets
import sys
import os
import time
from cachetools import cached

import flask
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler

from plugins.fastcompare.algo.wrappers.data_loadering import MLGenomeDataLoader, GoodBooksFilteredDataLoader

from plugins.grs2024.aggregators import AggregationStrategy

[sys.path.append(i) for i in ['.', '..']]
[sys.path.append(i) for i in ['../.', '../..', '../../.']]


import numpy as np

from common import get_tr, load_languages, multi_lang, load_user_study_config
from flask import Blueprint, jsonify, make_response, request, redirect, render_template, url_for, session

from plugins.utils.interaction_logging import log_interaction
from plugins.utils.helpers import stable_unique, cos_sim_np
from plugins.fastcompare.loading import load_data_loaders
from plugins.fastcompare import elicitation_ended, filter_params, load_data_loader_cached, search_for_item

from plugins.fastcompare import get_semi_local_cache_name, get_cache_path

#from memory_profiler import profile

NEG_INF = int(-10e6)

from app import rds

# Load available language mutations
languages = load_languages(os.path.dirname(__file__))


BASIC_ALGORITHMS = ["MPL", "LMS", "MUL"]

##### USER STUDY VARIABLES ######

### BETWEEN USER ################

# Each user can either be or not be part of the groups used in the experiments
USER_PART_OF_GROUP = [True, False]

# Join these two groups together into one group [Extra explanations ON/OFF]
# SHOW_MEMBERS_PREFERENCES = [True, False]
# SHOW_HISTORICAL_RECOMMENDATIONS = [True, False]
EXTENDED_EXPLANATIONS = [True, False]

THIRD_ALGORITHM = BASIC_ALGORITHMS

# So that in total we would have just 2 x 2 x 3 variables = 12 groups

### WITHIN USER #################

# Group types per iteration (is shuffled later on)
# GROUP_TYPES = ["similar", "divergent", "random", "outlier"]
GROUP_TYPES = ["similar", "outlier", "divergent"]

#################################

# Possible GRS Algorithms
NON_BASIC_ALGORITHMS = ["RLProp", "GFAR"]



# The mapping is not fixed between users
ALGORITHM_ANON_NAMES = ["BETA", "GAMMA", "DELTA"]

# For each group member, we show 5 items they liked in the past
# These items are sampled randomly from those items perceived positively
# by the group members, where positivly means rating >= 3
POSITIVE_RATING_THRESHOLD = 3.0



###### STUDY SCHEMA #######

# We have 3 algorithms and always show 2 which is 6 / 2 = 3 combinations
# A A B
# B C C
# denoted as C1, C2, and C3.
# 
# Then we have multiple groups, one per flavor (similar, outlier)
# If we had 2 iterations per group and per algorithm combination
# We would end up with 3 * 2 * 2 = 12 iterations
# Do it in a way that we run
# Randomly permutate order of flavors, say we have Group 1 and Group 2
# Randomly permute order of algorithms, say we have C1, C2, and C3
# Do:
# 1. (Group 1, C1, 0)
# 2. (Group 1, C1, 1)
# 3. (Group 1, C2, 0)
# 4. (Group 1, C2, 1)
# 5. (Group 1, C3, 0)
# 6. (Group 1, C3, 1)
# -> After block questionnaire
# 1. (Group 2, C1, 0)
# 2. (Group 2, C1, 1)
# 3. (Group 2, C2, 0)
# 4. (Group 2, C2, 1)
# 5. (Group 2, C3, 0)
# 6. (Group 2, C3, 1)

######


ENABLE_DEBUG = False


## GRS Algorihm Implementations ##
##### Taken from GRS Tutorial ####

##################################

# Implementation of this function can differ among plugins
def get_lang():
    default_lang = "en"
    if "lang" in session and session['lang'] and session['lang'] in languages:
        return session['lang']
    return default_lang

# Global, plugin related stuff
__plugin_name__ = "grs2024"
__version__ = "0.1.0"
__author__ = "Patrik Dokoupil"
__author_contact__ = "Patrik.Dokoupil@matfyz.cuni.cz"
__description__ = "Complex plugin for a very customized user study that we have used for GRS user study concerned with fairness perception"

bp = Blueprint(__plugin_name__, __plugin_name__, url_prefix=f"/{__plugin_name__}")

##### Redis helpers #######
# TODO decide if should be moved somewhere else
# Return user key for a given user (by session)
# We will key nearly all the data based on unique user session ID
# to ensure values stored to different users are separated
def get_uname():
    return f"user:{session['pid_long']}"

# Wrapper for setting values, performs serialization via pickle
def set_val(key, val):
    name = get_uname()
    rds.hset(name, key, value=pickle.dumps(val))

# Sets redis mapping (dict), serializing every value via pickle
def set_mapping(name, mapping):
    rds.hset(name, mapping={x: pickle.dumps(v) for x, v in mapping.items()})

# Wrapper for getting values, performs deserialization via pickle
def get_val(key):
    name = get_uname()
    return pickle.loads(rds.hget(name, key))

# Return all data we have for a given key
def get_all(name):
    res = {str(x, encoding="utf-8") : pickle.loads(v) for x, v in rds.hgetall(name).items()}
    return res

# Increment a given key
def incr(key):
    x = get_val(key)
    set_val(key, x + 1)

# Plugin specific version of enrich_results (transforming list of item indices into structured dict with additional item metadata)
# Here we are sure that we are inside this particular plugin
# thus we have a particular data loader and can use some of its internals
def enrich_results(top_k, loader, support=None):
    # This plugin is currently supposed to work with MLGenomeDataLoader and GoodBooksFilteredDataLoader
    # this is just a temporary safety check, should be removed in the future as there are no valid reasons
    # why this plugin should not work with other datasets
    assert isinstance(loader, MLGenomeDataLoader) or isinstance(loader, GoodBooksFilteredDataLoader), f"Loader name: {loader.name()} type: {type(loader)}"
    top_k_ids = [loader.get_item_id(movie_idx) for movie_idx in top_k]
    top_k_description = [loader.items_df_indexed.loc[movie_id].title for movie_id in top_k_ids]
    top_k_genres = [loader.get_item_id_categories(movie_id) for movie_id in top_k_ids]
    top_k_genres = [x if x != ["(no genres listed)"] else [] for x in top_k_genres]
    top_k_url = [loader.get_item_index_image_url(movie_idx) for movie_idx in top_k]
    top_k_trailers = [""] * len(top_k)
    top_k_plots = [loader.get_item_id_plot(movie_id) for movie_id in top_k_ids]


    if support:
        top_k_supports = [
            {
                "relevance": np.round(support["relevance"][i], 4),
                "diversity": np.round(support["diversity"][i], 4),
                "novelty": np.round(support["novelty"][i], 4),
                "raw_rating": np.round(support["raw_rating"][i], 4),
                "raw_distance": np.squeeze(np.round(support["raw_distance"][i], 4)).tolist(),
                "raw_users_viewed_item": support["raw_users_viewed_item"][i],
            }
            for i in range(len(top_k))
        ]
        return [
            {
            "movie": movie,
            "url": url,
            "movie_idx": str(movie_idx),
            "movie_id": movie_id,
            "genres": genres,
            "support": support,
            "trailer_url": trailer_url,
            "plot": plot,
            "rank": rank
            }
            for movie, url, movie_idx, movie_id, genres, support, trailer_url, plot, rank in
                zip(top_k_description, top_k_url, top_k, top_k_ids, top_k_genres, top_k_supports, top_k_trailers, top_k_plots, range(len(top_k_ids)))
        ]
    return [
        {
            "movie": movie,
            "url": url,
            "movie_idx": str(movie_idx),
            "movie_id": movie_id,
            "genres": genres,
            "trailer_url": trailer_url,
            "plot": plot,
            "rank": rank
        }
        for movie, url, movie_idx, movie_id, genres, trailer_url, plot, rank in
            zip(top_k_description, top_k_url, top_k, top_k_ids, top_k_genres, top_k_trailers, top_k_plots, range(len(top_k_ids)))
    ]

# Get content based distance matrix from a given path
# this function is cached, so the matrix file is read just once
@functools.lru_cache(maxsize=None)
def get_distance_matrix_cb(path):
    return np.load(path)

# For a given distance_matrix_cb, we calculate similarity_matrix, cached by path to the distance matrix (if it changes, cache should be invalid)
@functools.lru_cache(maxsize=None)
def get_similarity_matrix_cb(distance_matrix_cb_path):
    distance_matrix_cb = get_distance_matrix_cb(distance_matrix_cb_path)
    return 1.0 - distance_matrix_cb

# Get similarity_matrix (CF) -> cached version to prevent huge allocations when called for multiple users
# Keyed by loader name as it just depends on the loader's distance_matrix
@cached(cache={}, key=lambda loader: loader.name())
def get_similarity_matrix_cf(loader):
    return 1.0 - loader.distance_matrix

# Get cdf cache for a given metric
# this function is cached, so the cdf file is read just once
@functools.lru_cache(maxsize=None)
def load_cdf_cache(base_path, metric_name):
    with open(os.path.join(base_path, "cdf", f"{metric_name}.pckl"), "rb") as f:
        return pickle.load(f)

#################################################################################
###########################   ALGORITHMS   ######################################
#################################################################################
# TODO move algorithms to shared common

class EASER_pretrained:
    def __init__(self, all_items, **kwargs):
        self.all_items = all_items

    # We do this to hijack the cache so that there is one shared cache instance
    # instead of per algorithm instance cache (that causes leaks)
    def __eq__(self, other) -> bool:
        if not isinstance(other, EASER_pretrained):
            return False
        
        # We use arange for items, so just comparing the size should be fine for our purpose
        return self.all_items.size == other.all_items.size

    def __hash__(self) -> int:
        return self.all_items.size

    @functools.lru_cache(maxsize=None)
    def load(self, path):
        self.item_item = np.load(path)
        assert self.item_item.shape[0] == self.item_item.shape[1] == self.all_items.size
        return self

    def predict_with_score(self, selected_items, filter_out_items, k):
        candidates = np.setdiff1d(self.all_items, selected_items)
        candidates = np.setdiff1d(candidates, filter_out_items)
        user_vector = np.zeros(shape=(self.all_items.size,), dtype=self.item_item.dtype)
        if selected_items.size == 0:
            return np.zeros_like(user_vector), user_vector, np.random.choice(candidates, size=k, replace=False).tolist()
        user_vector[selected_items] = 1
        probs = np.dot(user_vector, self.item_item)

        # Here the NEG_INF used for masking must be STRICTLY smaller than probs predicted by the algorithms
        # So that the masking works properly
        assert NEG_INF < probs.min()
        # Mask out selected items
        probs[selected_items] = NEG_INF
        # Mask out items to be filtered
        probs[filter_out_items] = NEG_INF
        return probs, user_vector, np.argsort(-probs)[:k].tolist()

    # Predict for the user
    def predict(self, selected_items, filter_out_items, k):
        return self.predict_with_score(selected_items, filter_out_items, k)[2]



# @cached(cache={}, key=lambda loader: loader.name())
# def get_user_similarity_matrix(loader):
#     return cos_sim_np(loader.rating_matrix)

@cached(cache={}, key=lambda loader: loader.name())
def get_normalized_rating_matrix(loader):
    # We use the same thresholds on which the EASE was pretrained
    return np.where(loader.rating_matrix >= 3.0, 1, 0).astype(np.uint8)

@cached(cache={}, key=lambda loader: loader.name())
def get_normalized_rating_matrix_for_cos_sim(loader):
    # We use the same thresholds on which the EASE was pretrained
    x = get_normalized_rating_matrix(loader)
    # We need this for cosine similarity in group construction and need it to be as quick as possible
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).T


# Returns past positive history for a given user
def get_past_positive_history(user_idx, rating_matrix, history_length_limit):
    if user_idx == -1:
        return np.array([], dtype=np.int32)
    positive_feedback = np.where(rating_matrix[user_idx] >= POSITIVE_RATING_THRESHOLD)[0]
    np.random.shuffle(positive_feedback)
    return positive_feedback[:history_length_limit]

# User embedding is vector of shape (n_items,) (if specified)
# Each element being either 0 or 1 (implicit feedback)
# The loader.rating_matrix may not be normalized to [0, 1] so we do it here
def generate_similar_group(loader, group_size, user_embedding, filt_out, history_length_limit):
    start_time = time.perf_counter()
    rm_normed = get_normalized_rating_matrix(loader)
    per_user_ratings = rm_normed.sum(axis=1)
    n_users, n_items = rm_normed.shape
    indices = []
    # Do it incrementally, start with a embedding user (either random, or the real user)
    if user_embedding is not None:
        assert user_embedding.shape == (n_items, ), f"user_embedding.shape={user_embedding.shape}"
        prefix_embeddings = user_embedding
        indices.append(-1)
    else:
        non_zero_users = np.where(per_user_ratings > 0)[0]
        rnd_idx = np.random.choice(non_zero_users)
        prefix_embeddings = rm_normed[rnd_idx]
        filt_out.append(rnd_idx)
        indices.append(rnd_idx.item())

    assert prefix_embeddings.sum() > 0, f"We expect at least some history for the user: {prefix_embeddings.sum()}"

    assert prefix_embeddings.shape == (n_items, ), f"shape={prefix_embeddings.shape}"
    prefix_embeddings = prefix_embeddings[np.newaxis, :]

    # rm_normed_t = (rm_normed / np.linalg.norm(rm_normed, axis=1, keepdims=True)).T
    rm_normed_t = get_normalized_rating_matrix_for_cos_sim(loader)

    for i in range(group_size - 1):
        # These newaxis are here to ensure proper broadcasting
        # Basically, prefix_embeddings are converted from (i+1, N) to (i+1, 1, N) and rm_normed is from (M, N) to (1, M, N)
        # Therefore i+1 broadcasts (thanks to 1 in first dimension of rating matrix) and same for M in the rating matrix
        # Effectively we end up with (i+1, M) which means for each prefix embedding and each candidate we have a distance between the two
        # distances = np.linalg.norm(prefix_embeddings[:, np.newaxis, :] - rm_normed[np.newaxis, :, :], axis=2, )
        # For cosine distance it is slightly different
        distances = 1.0 - np.dot(prefix_embeddings / np.linalg.norm(prefix_embeddings, axis=1, keepdims=True), rm_normed_t)
        assert distances.shape == (i + 1, n_users), f"distances.shape={distances.shape}"
        non_nan_max = distances[~np.isnan(distances)].max()
        # Then we need to aggregate this to have one distance for each candidate user
        # What we do here is to simply take random users whose mean similarity to the rest of the (partially generated) group
        # is above certain threshold
        # We also mask already selected users
        if filt_out:
            distances[:, filt_out] = non_nan_max + 1

        # Similarly, we filter out NaNs (these may occurr whenever rm_normed row is full of 0)
        zero_users = np.where(per_user_ratings == 0)[0]
        if zero_users.size > 0:
            distances[:, zero_users] = non_nan_max + 1
        
        distances = distances.mean(axis=0)
        assert distances.shape == (n_users, ), f"distances.shape after mean = {distances.shape}"

        # We take 20th percentile as a threshold (we go <= not >=)
        threshold = np.percentile(distances, 20)
        candidates = np.where(distances <= threshold)[0]
        #print(f"### threshold={threshold}, candidates={candidates.tolist()}")
        selected_candidate = np.random.choice(candidates)
        # Mask it out
        filt_out.append(selected_candidate)
        indices.append(selected_candidate.item())
        # Append it to current group
        prefix_embeddings = np.concatenate([prefix_embeddings, rm_normed[selected_candidate][np.newaxis, :]], axis=0) #TODO rm_normed_t ???
    
    assert prefix_embeddings.shape == (group_size, n_items), f"prefix_embeddings.shape={prefix_embeddings.shape}"

    group_sim = cos_sim_np(prefix_embeddings)
    assert group_sim.shape == (group_size, group_size), f"group_sim.shape={group_sim.shape}"

    print(f"SIMILAR GROUP TOOK: {time.perf_counter() - start_time}")
    return {
        "embeddings": prefix_embeddings,
        "indices": indices,
        "extra_data": {
            "mean_sim": (group_sim[np.triu_indices(group_size, k=1)].sum() / ((group_size * (group_size - 1)) / 2)).item(),
            "past_positive_history": {user_idx : get_past_positive_history(user_idx, loader.rating_matrix, history_length_limit).tolist() for user_idx in indices}
        }
    }

def generate_divergent_group(loader, group_size, user_embedding, filt_out, history_length_limit):
    start_time = time.perf_counter()
    rm_normed = get_normalized_rating_matrix(loader)
    per_user_ratings = rm_normed.sum(axis=1)
    n_users, n_items = rm_normed.shape
    indices = []

    # Do it incrementally, start with a embedding user (either random, or the real user)
    if user_embedding is not None:
        prefix_embeddings = user_embedding
        indices.append(-1)
    else:
        non_zero_users = np.where(per_user_ratings > 0)[0]
        rnd_idx = np.random.choice(non_zero_users)
        prefix_embeddings = rm_normed[rnd_idx]
        filt_out.append(rnd_idx)
        indices.append(rnd_idx.item())

    assert prefix_embeddings.sum() > 0, f"We expect at least some history for the user: {prefix_embeddings.sum()}"
    assert prefix_embeddings.shape == (n_items, ), f"shape={prefix_embeddings.shape}"
    prefix_embeddings = prefix_embeddings[np.newaxis, :]

    # rm_normed_t = (rm_normed / np.linalg.norm(rm_normed, axis=1, keepdims=True)).T
    rm_normed_t = get_normalized_rating_matrix_for_cos_sim(loader)

    for i in range(group_size - 1):
        # These newaxis are here to ensure proper broadcasting
        # Basically, prefix_embeddings are converted from (i+1, N) to (i+1, 1, N) and rm_normed is from (M, N) to (1, M, N)
        # Therefore i+1 broadcasts (thanks to 1 in first dimension of rating matrix) and same for M in the rating matrix
        # Effectively we end up with (i+1, M) which means for each prefix embedding and each candidate we have a distance between the two
        # distances = np.linalg.norm(prefix_embeddings[:, np.newaxis, :] - loader.rm_normed[np.newaxis, :, :], axis=2)
        # For cosine distance it is slightly different
        distances = 1.0 - np.dot(prefix_embeddings / np.linalg.norm(prefix_embeddings, axis=1, keepdims=True), rm_normed_t)
        assert distances.shape == (i + 1, n_users), f"distances.shape={distances.shape}"
        non_nan_min = distances[~np.isnan(distances)].min()
        # Then we need to aggregate this to have one distance for each candidate user
        # What we do here is to simply take random users whose mean similarity to the rest of the (partially generated) group
        # is above certain threshold
        # We also mask already selected users
        if filt_out:
            distances[:, filt_out] = non_nan_min - 1
        
        # Similarly, we filter out NaNs (these may occurr whenever rm_normed row is full of 0)
        zero_users = np.where(per_user_ratings == 0)[0]
        if zero_users.size > 0:
            distances[:, zero_users] = non_nan_min - 1
        
        distances = distances.mean(axis=0)
        assert distances.shape == (n_users, ), f"distances.shape after mean = {distances.shape}"

        # We take 80th percentile as a threshold (we go >= not <=)
        threshold = np.percentile(distances, 80)
        candidates = np.where(distances >= threshold)[0]
        #print(f"### threshold={threshold}, candidates={candidates.tolist()}")
        selected_candidate = np.random.choice(candidates)
        # Mask it out
        filt_out.append(selected_candidate)
        indices.append(selected_candidate.item())
        # Append it to current group
        prefix_embeddings = np.concatenate([prefix_embeddings, rm_normed[selected_candidate][np.newaxis, :]], axis=0)
    
    assert prefix_embeddings.shape == (group_size, n_items), f"prefix_embeddings.shape={prefix_embeddings.shape}"

    group_sim = cos_sim_np(prefix_embeddings)
    assert group_sim.shape == (group_size, group_size), f"group_sim.shape={group_sim.shape}"

    print(f"DIVERGENT GROUP TOOK: {time.perf_counter() - start_time}")
    return {
        "embeddings": prefix_embeddings,
        "indices": indices,
        "extra_data": {
            "mean_sim": (group_sim[np.triu_indices(group_size, k=1)].sum() / ((group_size * (group_size - 1)) / 2)).item(),
            "past_positive_history": {user_idx : get_past_positive_history(user_idx, loader.rating_matrix, history_length_limit).tolist() for user_idx in indices}
        }
    }

def generate_outlier_group(loader, group_size, user_embedding, history_length_limit):
    start_time = time.perf_counter()
    rm_normed = get_normalized_rating_matrix(loader)
    per_user_ratings = rm_normed.sum(axis=1)
    n_users, n_items = rm_normed.shape
    indices = []
    # Outlier embedding is either the user, or randomly sampled vector from the rating matrix
    # Depending on the study variables
    if user_embedding is not None:
        prefix_embeddings = user_embedding
        filt_out = []
        indices.append(-1)
    else:
        non_zero_users = np.where(per_user_ratings > 0)[0]
        rnd_idx = np.random.choice(non_zero_users)
        prefix_embeddings = rm_normed[rnd_idx]
        filt_out = [rnd_idx]
        indices.append(rnd_idx.item())

    assert prefix_embeddings.shape == (n_items, ), f"shape={prefix_embeddings.shape}"
    prefix_embeddings = prefix_embeddings[np.newaxis, :]

    outlier_index = 0

    # Once we have the outlier, we essentially generate divergent group of size 2 (including the outlier)
    # Around that outlier. This basically means -> find single, divergent group member for the outlier
    # Once we have it, we basically just generate similar group around the divergent member, this time, of size group_size - 1
    # We pass filt out so that it does not happen that second selected member is the same as the randomly generated outlier
    # Although this should never happen given that we generate divergent group and similarity of user to itself is 0
    div_group = generate_divergent_group(loader, 2, prefix_embeddings[outlier_index], filt_out=filt_out, history_length_limit=history_length_limit)
    assert div_group['embeddings'].shape == (2, n_items), f"div_group.shape={div_group['embeddings'].shape}"

    # We need to replace the -1 inside div_group since we passed prefix_embeddings[outlier_index] as the user embedding but it may not be user embedding actually
    if user_embedding is None:
        idx = div_group["indices"].index(-1)
        if idx >= 0:
            div_group["indices"][idx] = indices[outlier_index]

    if group_size > 2:
        # We still need group_size - 1 to be >= 2 that is why the "if" above
        # Now we generate similar group so it is much more important to properly pass filt_out
        # so that we do not include a user twice
        sim_group = generate_similar_group(loader, group_size - 1, div_group['embeddings'][1], filt_out=filt_out, history_length_limit=history_length_limit)
        # We need to replace the -1 inside div_group since we passed prefix_embeddings[outlier_index] as the user embedding but it may not be user embedding actually
        if user_embedding is None:
            idx = sim_group["indices"].index(-1)
            if idx >= 0:
                sim_group["indices"][idx] = div_group['indices'][1]
        outlier_group = np.concatenate([prefix_embeddings, sim_group['embeddings']], axis=0)
        assert outlier_group.shape == (group_size, n_items), f"outlier_group.shape={outlier_group.shape}"
        assert len(indices) + len(sim_group['indices']) == group_size, f"{len(indices)} + {len(sim_group['indices'])} != {group_size}"
        
        group_sim_homogeneous = cos_sim_np(sim_group['embeddings'])
        assert group_sim_homogeneous.shape == (group_size - 1, group_size - 1), f"group_sim.shape={group_sim_homogeneous.shape}"

        mean_outlier_sim = np.dot(
            prefix_embeddings / np.linalg.norm(prefix_embeddings, axis=1, keepdims=True),
            (sim_group['embeddings'] / np.linalg.norm(sim_group['embeddings'], axis=1, keepdims=True)).T
        )
        assert mean_outlier_sim.shape == (1, group_size - 1), f"distances.shape={mean_outlier_sim.shape}"

        print(f"OUTLIER GROUP TOOK: {time.perf_counter() - start_time}")
        return {
            "embeddings": outlier_group,
            "indices": indices + sim_group['indices'],
            "extra_data": {
                "mean_sim_homogeneous_part": (group_sim_homogeneous[np.triu_indices(group_size - 1, k=1)].sum() / (((group_size - 1) * (group_size - 2)) / 2)).item(),
                "mean_outlier_sim": mean_outlier_sim.mean().item(),
                "past_positive_history": {user_idx : get_past_positive_history(user_idx, loader.rating_matrix, history_length_limit).tolist() for user_idx in indices + sim_group['indices']}
            }}
    else:
        group_sim = cos_sim_np(div_group["embeddings"])
        assert group_sim.shape == (group_size, group_size), f"group_sim.shape={group_sim.shape}"
        div_group["extra_data"] = {
            "mean_outlier_sim": (group_sim[np.triu_indices(group_size, k=1)].sum() / ((group_size * (group_size - 1)) / 2)).item(),
            "past_positive_history": {user_idx : get_past_positive_history(user_idx, loader.rating_matrix, history_length_limit).tolist() for user_idx in div_group['indices']}
        }
        return div_group




# What is outcome of group generation
# Basically a list of |G| vectors each of size N (number of items)
# Corresponding to "user" embeddings. The term user here is vague, we do not deal
# with user IDs and instead, only care about the embeddings

def generate_groups(loader, group_types_permutation, group_size, user_embedding, history_length_limit):
    assert group_size >= 2, f"group_size={group_size}"
    rm_normed = get_normalized_rating_matrix(loader)
    n_users, n_items = rm_normed.shape

    #similarity_matrix = get_user_similarity_matrix(loader)

    resulting_groups = []

    for gtype in group_types_permutation:
        assert gtype in GROUP_TYPES, f"gtype={gtype}"
        if gtype == "similar":
            resulting_groups.append(generate_similar_group(loader, group_size, user_embedding, [], history_length_limit))
        elif gtype == "divergent":
            resulting_groups.append(generate_divergent_group(loader, group_size, user_embedding, [], history_length_limit))
        elif gtype == "random":
            # Random is so simple that we do it inline here
            if user_embedding is not None:
                # We have group_size - 1 users and combine it with input user embedding (the real user doing the study)
                members = np.random.choice(n_users, group_size - 1, replace=False)
                emb = np.concatenate([rm_normed[members], user_embedding], axis=0)
                group_sim = cos_sim_np(emb)
                assert group_sim.shape == (group_size, group_size), f"group_sim.shape={group_sim.shape}"
                resulting_groups.append({
                    "embeddings": emb,
                    "indices": [-1] + members.tolist(),
                    "extra_data": {
                        "mean_sim": group_sim[np.triu_indices(group_size, k=1)].sum() / ((group_size * (group_size - 1)) / 2),
                        "past_positive_history": {user_idx : get_past_positive_history(user_idx, loader.rating_matrix, history_length_limit).tolist() for user_idx in [-1] + members.tolist()}
                    }
                })
            else:
                # We just sample all the embeddings
                rnd_members = np.random.choice(n_users, group_size, replace=False)
                emb = rm_normed[rnd_members]
                group_sim = cos_sim_np(emb)
                assert group_sim.shape == (group_size, group_size), f"group_sim.shape={group_sim.shape}"
                resulting_groups.append({
                    "embeddings": emb,
                    "indices": rnd_members.tolist(),
                    "extra_data": {
                        "mean_sim": group_sim[np.triu_indices(group_size, k=1)].sum() / ((group_size * (group_size - 1)) / 2),
                        "past_positive_history": {user_idx : get_past_positive_history(user_idx, loader.rating_matrix, history_length_limit).tolist() for user_idx in rnd_members}
                    }
                })
        elif gtype == "outlier":
            resulting_groups.append(generate_outlier_group(loader, group_size, user_embedding, history_length_limit))

        print(resulting_groups[-1]['extra_data']['past_positive_history'])
        assert len(resulting_groups[-1]['indices']) == len(resulting_groups[-1]['extra_data']['past_positive_history'])
    return resulting_groups

##### End of algorithms #####


#################################################################################
###########################   ENDPOINTS   #######################################
#################################################################################


# Render grs2024 plugin study creation page
@bp.route("/create")
@multi_lang
def create():

    tr = get_tr(languages, get_lang())

    params = {}
    params["contacts"] = tr("footer_contacts")
    params["contact"] = tr("footer_contact")
    params["charles_university"] = tr("footer_charles_university")
    params["cagliari_university"] = tr("footer_cagliari_university")
    params["t1"] = tr("footer_t1")
    params["t2"] = tr("footer_t2")
    params["about_placeholder"] = tr("fastcompare_create_about_placeholder")
    params["override_informed_consent"] = tr("fastcompare_create_override_informed_consent")
    params["override_about"] = tr("fastcompare_create_override_about")
    params["show_final_statistics"] = tr("fastcompare_create_show_final_statistics")
    params["override_algorithm_comparison_hint"] = tr("fastcompare_create_override_algorithm_comparison_hint")
    params["algorithm_comparison_placeholder"] = tr("fastcompare_create_algorithm_comparison_placeholder")
    params["informed_consent_placeholder"] = tr("fastcompare_create_informed_consent_placeholder")

    # TODO add tr(...) to make it translatable
    params["disable_demographics"] = "Disable demographics"
    params["extended_explanations"] = "Extended explanations"
    params["user_part_of_group"] = "User part of group"
    params["basic_grs"] = "Please select basic GRS algorithm"
    params["choice_model"] = "Select model used to simulate group choice"
    params["group_source"] = "Select source of groups (when 'User part of group is set', the only option is 'Online')"
    params["filter_option"] = "What items should be filtered between iterations"

    return render_template("grs_create.html", **params)

# Public facing endpoint
@bp.route("/join", methods=["GET"])
def join():
    assert "guid" in request.args, "guid must be available in arguments"

    pid_long = secrets.token_urlsafe(16) # once per join, not once per browser session like uuid is
    session["pid_long"] = pid_long
    session.modified = True

    return redirect(url_for("utils.join", continuation_url=url_for("grs2024.on_joined"), **request.args))

# Callback once user has joined we forward to pre-study questionnaire
@bp.route("/on-joined", methods=["GET", "POST"])
def on_joined():
    # The user study configuration for each user is generated right after joining (because even pref. elicitation)
    # Usually this was done later, in "after-preference-elicitation" step, but in this study
    # some portion of users won't get through pref elicitation at all

    # Retrieve configuration
    conf = load_user_study_config(session['user_study_id'])
    user_part_of_group = conf["user_part_of_group"]
    extended_explanations = conf["extended_explanations"]
    basic_grs = conf["basic_grs"]
    choice_model = conf["choice_model"]
    filter_option = conf["filter_option"]

    assert user_part_of_group in (["Random", True, False]), f"user_part_of_group={user_part_of_group}"
    assert extended_explanations in (["Random", True, False]), f"extended_explanations={extended_explanations}"
    assert basic_grs in (["Random"] + BASIC_ALGORITHMS), f"basic_grs={basic_grs}"
    assert choice_model in ["Random", "RandomChoice", "Same", "None"] + BASIC_ALGORITHMS + NON_BASIC_ALGORITHMS, f"choice_model={choice_model}"
    assert filter_option in ["Selection", "Impressions"], f"filter_option={filter_option}"

    selected_user_part_of_group = user_part_of_group if user_part_of_group != "Random" else np.random.choice(USER_PART_OF_GROUP)
    selected_extended_explanations = extended_explanations if extended_explanations != "Random" else np.random.choice(EXTENDED_EXPLANATIONS)
    selected_basic_grs = basic_grs if basic_grs != "Random" else np.random.choice(BASIC_ALGORITHMS)
    selected_choice_model = choice_model if choice_model != "Random" else np.random.choice(["Same", "RandomChoice"] + BASIC_ALGORITHMS + NON_BASIC_ALGORITHMS)

    # Build the configuration
    # We know we have several group types, each of them gets repeated ITERS_PER_GROUP_TYPE times
    # Order of blocks is random
    selected_group_types = GROUP_TYPES[:]
    np.random.shuffle(selected_group_types)

    ## Here we have e.g. (Group 1), (Group 1), (Group 2), (Group 2)

    # Also generate order of algorithms for the user
    # We always fix that columns are sorted by alphabet, e.g.
    # BETA, GAMMA
    # BETA, DELTA
    # GAMMA, DELTA
    # However, the assignment of actual algorithms to these labels is random for each user
    # After the random assignment is established for the user, it is used throughout whole study for him/her
    all_user_algorithms = NON_BASIC_ALGORITHMS + [selected_basic_grs]
    np.random.shuffle(all_user_algorithms)
    # Here we have e.g. (AB), (AC), (BC)
    # all_user_algorithms[0] is BETA
    # all_user_algorithms[1] is GAMMA
    # all_user_algorithms[2] is DELTA
    algorithm_combinations = [
        ('BETA', 'GAMMA'), #(all_user_algorithms[0], all_user_algorithms[1]),
        ('BETA', 'DELTA'), #(all_user_algorithms[0], all_user_algorithms[2]),
        ('GAMMA', 'DELTA'), #(all_user_algorithms[1], all_user_algorithms[2]),
    ]
    algorithm_name_assignment = {
        key : algo for key, algo in zip(ALGORITHM_ANON_NAMES, all_user_algorithms)
    }

    # Old code that was producing configurations such that we had blocks comparing two algorithms
    # New code generates blocks per single group type 
    # # For each iteration, capture the group type used in that iteration
    # generated_iters = [{"group_type": x} for x in selected_group_types for _ in range(conf['iters_per_group_type'])]
    # # Now for every algorithm combination, we enumerate all group, each of them for given number of iterations
    # # See the SCHEMA at the top of this file for details
    # final_iterations = []
    # for algo_1, algo_2 in algorithm_combinations:
    #     for i in range(len(generated_iters)):
    #         final_iterations.append({
    #             "group_type": generated_iters[i]['group_type'],
    #             "algorithms": {
    #                 algo_1: algorithm_name_assignment[algo_1],
    #                 algo_2: algorithm_name_assignment[algo_2]
    #             }
    #         })
    final_iterations = []
    for group_type in selected_group_types:
        for _ in range(conf['iters_per_group_type']):
            for algo_1, algo_2 in algorithm_combinations:
                final_iterations.append({
                    "group_type": group_type,
                    "algorithms": {
                        algo_1: algorithm_name_assignment[algo_1],
                        algo_2: algorithm_name_assignment[algo_2]
                    }
                })

    print(f"### Final iterations:")
    print(final_iterations)
    print("###")

    expected_len = len(GROUP_TYPES) * conf['iters_per_group_type'] * len(algorithm_combinations)
    assert len(final_iterations) == expected_len, f"{len(final_iterations)} != {expected_len}"


    generated_iters = final_iterations
    # Store the configuration
    generated_config = {
        "selected_user_part_of_group": selected_user_part_of_group,
        "selected_extended_explanations": selected_extended_explanations,
        "selected_basic_grs": selected_basic_grs,
        "selected_group_types": selected_group_types,
        "generated_iters": generated_iters,
        "algorithm_name_assignment": algorithm_name_assignment,
        "selected_choice_model": selected_choice_model,
        "filter_option": filter_option,
        "pid_long": session["pid_long"],
    }

    log_interaction(session["participation_id"], "generated-configuration", **generated_config)

    set_mapping(get_uname(), generated_config)

    return redirect(url_for("grs2024.pre_study_questionnaire"))

# Endpoint for pre-study questionnaire
@bp.route("/pre-study-questionnaire", methods=["GET", "POST"])
def pre_study_questionnaire():
    params = {
        "continuation_url": url_for("grs2024.pre_study_questionnaire_done"),
        "header": "Pre-study questionnaire",
        "hint": "Please answer the questions below before starting the user study.",
        "finish": "Proceed to user study",
        "title": "Pre-study questionnaire"
    }
    return render_template("grs_pre_study_questionnaire.html", **params)

# Endpoint that should be called once pre-study-questionnaire is done
@bp.route("/pre-study-questionnaire-done", methods=["GET", "POST"])
def pre_study_questionnaire_done():

    data = {}
    data.update(**request.form)

    # We just log the question answers as there is no other useful data gathered during pre-study-questionnaire
    log_interaction(session["participation_id"], "pre-study-questionnaire", **data)

    user_data = get_all(get_uname())
    is_user_part_of_group = user_data['selected_user_part_of_group']

    if is_user_part_of_group:
        # Step through the preference elicitaiton
        return redirect(url_for("utils.preference_elicitation", continuation_url=url_for("grs2024.send_feedback"),
                consuming_plugin=__plugin_name__,
                initial_data_url=url_for('fastcompare.get_initial_data'),
                search_item_url=url_for('journal.item_search')))
    else:
        # Continue directly to group generation
        return redirect(url_for("grs2024.group_gen"))

# Receives arbitrary feedback (typically from preference elicitation) and generates recommendation
@bp.route("/send-feedback", methods=["GET"])
def send_feedback():
    # We read k from configuration of the particular user study
    conf = load_user_study_config(session['user_study_id'])

    # Some future steps (outside of this plugin) may relay on presence of "iteration" key in session
    # we just have to set it, not keep it updated
    session["iteration"] = 0

    # Get a loader
    loader_factory = load_data_loaders()[conf["selected_data_loader"]]
    loader = loader_factory(**filter_params(conf["data_loader_parameters"], loader_factory))
    loader = load_data_loader_cached(loader, session['user_study_guid'], loader_factory.name(), get_semi_local_cache_name(loader))

    # Movie indices of selected movies
    selected_movies = request.args.get("selectedMovies")
    selected_movies = selected_movies.split(",") if selected_movies else []
    selected_movies = [int(m) for m in selected_movies]

    # Indices of items shown during preference elicitation
    elicitation_shown_items = stable_unique([int(movie["movie_idx"]) for movie in session['elicitation_movies']])

    set_mapping(get_uname(), {
        'iteration': 0, # Start with zero, because at the very beginning, mors_feedback is called, not mors and that generates recommendations for first iteration, but at the same time, increases the iteration
        'elicitation_selected_movies': selected_movies,
        'selected_movie_indices': [],
        'elicitation_shown_movies': elicitation_shown_items
    })

    elicitation_ended(
        session['elicitation_movies'],
        selected_movies,
        ease_selections=selected_movies,
        ease_filter_out=selected_movies,
        elicitation_shown_items=elicitation_shown_items.tolist()
    )

    return redirect(url_for("grs2024.group_gen"))


def generate_group_recommendations(conf, loader, iteration, shown_items, group_choices):
    user_data = get_all(get_uname())
    start_time = time.perf_counter()

    # At the same time, we generate recommendations
    # Since this needs to be done in "hidden" endpoint
    # so that page refresh do not lead to repeated recommendations

    # List where each entry corresponds to data set for a single iteration
    # So each entry is a dict with "algorithms" and "group_type" keys
    generated_iters = user_data["generated_iters"]

    group_recommendations = {}

    selected_group_types = user_data["selected_group_types"]

    items = np.arange(loader.rating_matrix.shape[1])
    ease = EASER_pretrained(items)
    ease = ease.load(os.path.join(get_cache_path(get_semi_local_cache_name(loader)), "item_item.npy"))

    print(f"Until EASE got loaded it took: {time.perf_counter() - start_time}")

    # groups contains 1 group per group type
    # the whole configuration looks like selected_group_types where each element is repeated
    groups = get_val('groups')
    current_group_type = generated_iters[iteration]['group_type']
    print(f"current_group_type: {current_group_type}, selected_group_types: {selected_group_types}, groups: {groups}")
    current_group = groups[selected_group_types.index(current_group_type)]

    group_members = current_group['indices']

    # Instead of iterating algorithms first, we first iterate over users
    # Since all GRS can reuse output of single EASE call
    n_per_user_selections = []
    user_ids = []
    items_ids = []
    ratings_raw = []
    ratings_normed = []
    ratings_lists = []
    for g_user in group_members:
        member_selections = np.where(loader.rating_matrix[g_user] > 0.0)[0]
        # conf['k'] is actually not needed since we only care about scores not the top_k
        scores, user_vector, top_k = ease.predict_with_score(member_selections, member_selections, conf['k'])
        scores_list = scores.tolist()

        non_neg_inf_score_indices = scores > NEG_INF
        neg_inf_score_indices = scores <= NEG_INF
        qt = MinMaxScaler().fit(scores[non_neg_inf_score_indices].reshape(-1, 1))
        scores_transformed = qt.transform(scores.reshape(-1, 1)).reshape(scores.shape)
        scores_transformed[neg_inf_score_indices] = NEG_INF
        ratings_raw.extend(scores_list)
        ratings_normed.extend(scores_transformed.tolist())
        ratings_lists.append(scores_transformed)
        items_ids.extend(items.tolist())
        user_ids.extend([g_user] * items.size)
        n_per_user_selections.append(len(member_selections))
        assert len(member_selections) == scores[scores == NEG_INF].size, f"NEG_INF={NEG_INF} occurrences not matching {member_selections}, {scores_list}"

    group_ratings_full = pd.DataFrame({
        "user": user_ids,
        "item": items_ids,
        "predicted_rating_raw" : ratings_raw,
        "predicted_rating": ratings_normed,
    })

    old_size = len(group_ratings_full)
    neg_inf_rating_items = group_ratings_full[group_ratings_full['predicted_rating'] <= NEG_INF]['item'].unique()
    # We need to drop all the rows that contain item rated NEG_INF by ANY of the group members
    group_ratings = group_ratings_full[~group_ratings_full['item'].isin(neg_inf_rating_items)]
    #group_ratings = group_ratings_full[group_ratings_full.predicted_rating != NEG_INF]
    # Assertion no longer hold since if only one group mamber gave NEG_INF, we filter it out from all the users
    #assert old_size - len(group_ratings) == sum(n_per_user_selections), f"Number of selections per user ({n_per_user_selections}) not reflected in size drop: {old_size}, {len(group_ratings)}"


    print(f"Until we get the predictions for group members: {time.perf_counter() - start_time}")
    group_ratings_rm = np.stack(ratings_lists, axis=0)
    group_ratings_rm[:, neg_inf_rating_items] = NEG_INF

    # Recommendations from all available algorithms, not just those selected for current iteration
    # Only used for logging purposes
    # We do index by real name here
    full_recs = dict()
    full_recs_data = {
        "iteration": iteration,
        "iteration_data": generated_iters[iteration],
    }

    selected_choice_model = user_data["selected_choice_model"]
    filter_option = user_data["filter_option"]
    if selected_choice_model == "None":
        filter_option = "Impressions"

    algo_anon_to_name = dict()

    for order, (algo_anon, algo_name) in enumerate(generated_iters[iteration]["algorithms"].items()):
        
        algo_anon_to_name[algo_anon] = algo_name

        if algo_name in BASIC_ALGORITHMS:
            # These are part of "BASE" aggregator
            algo = AggregationStrategy.getAggregator("BASE")
        else:
            algo = AggregationStrategy.getAggregator(algo_name)

        if filter_option == "Impressions":
            filter_out_items = shown_items
        elif filter_option == "Selection":
            filter_out_items = group_choices

        all_filter_out_items = list(itertools.chain(*filter_out_items[algo_anon][current_group_type]))

        group_ratings_rm_copy = group_ratings_rm.copy()
        group_ratings_rm_copy[:, all_filter_out_items] = NEG_INF

        # Build the group_ratings dataframe in the format that is expected
        # by the algorithms, so basically DF with the following columns: [user, item, predicted_rating]
        if algo_name == "RLProp":
            # We use precomputed rating matrix to save on time (since we would pivot inside RLProp and that would take unneccessary time)
            rec_list = algo.generate_group_recommendations_for_group(group_ratings,
                                                                     recommendations_number=conf['k'],
                                                                     group_ratings_rm=group_ratings_rm_copy,
                                                                     past_recommendations=filter_out_items[algo_anon][current_group_type])
        else:
            rec_list = algo.generate_group_recommendations_for_group(group_ratings[~group_ratings.item.isin(all_filter_out_items)],
                                                                     recommendations_number=conf['k'])
        rec_list = rec_list[algo_name]
        rec_list = enrich_results(rec_list, loader)

        if selected_choice_model == "Same":
            rec_list[0]["group_choice"] = True

        # Add information about predicted rating w.r.t. each of the group members
        for idx in range(len(rec_list)):
            movie_idx = int(rec_list[idx]["movie_idx"])
            predicted_ratings = group_ratings[group_ratings.item == movie_idx].set_index("user")
            assert len(predicted_ratings) == len(group_members), f"{len(predicted_ratings)} != {len(group_members)}"

            group_member_ratings_raw, group_member_ratings_normed, group_member_ratings_normed_percentage = dict(), dict(), dict()
            for g_uid in group_members:
                group_member_ratings_raw[g_uid] = predicted_ratings.loc[g_uid].predicted_rating_raw
                group_member_ratings_normed[g_uid] = predicted_ratings.loc[g_uid].predicted_rating
                group_member_ratings_normed_percentage[g_uid] = round(group_member_ratings_normed[g_uid] * 100)

            rec_list[idx]["extra_data"] = {
                "group_member_ratings_raw": group_member_ratings_raw,
                "group_member_ratings_normed": group_member_ratings_normed,
                "group_member_ratings_normed_percentage": group_member_ratings_normed_percentage,
            }

        group_recommendations[algo_anon] = {
            "movies": rec_list,
            "order": order,
        }

        full_recs[algo_name] = {
            "movies": rec_list,
            "order": order,
            "filter_out_items": filter_out_items,
            "all_filter_out_items": all_filter_out_items,
        }

        print(f"Until {algo_name} recommendations got generated it took: {time.perf_counter() - start_time}")

    # Cover rest of the algorithms
    for algo_name in NON_BASIC_ALGORITHMS + BASIC_ALGORITHMS:
        if algo_name in full_recs:
            continue
        
        if algo_name in BASIC_ALGORITHMS:
            # These are part of "BASE" aggregator
            algo = AggregationStrategy.getAggregator("BASE")
        else:
            algo = AggregationStrategy.getAggregator(algo_name)

        if filter_option == "Impressions":
            filter_out_items = shown_items
        elif filter_option == "Selection":
            filter_out_items = group_choices

        all_filter_out_items = list(itertools.chain(*filter_out_items[algo_anon][current_group_type]))

        group_ratings_rm_copy = group_ratings_rm.copy()
        group_ratings_rm_copy[:, all_filter_out_items] = NEG_INF

        # Build the group_ratings dataframe in the format that is expected
        # by the algorithms, so basically DF with the following columns: [user, item, predicted_rating]
        if algo_name == "RLProp":
            # We use precomputed rating matrix to save on time (since we would pivot inside RLProp and that would take unneccessary time)
            rec_list = algo.generate_group_recommendations_for_group(group_ratings,
                                                                     recommendations_number=conf['k'],
                                                                     group_ratings_rm=group_ratings_rm_copy,
                                                                     past_recommendations=filter_out_items[algo_anon][current_group_type])
        else:
            rec_list = algo.generate_group_recommendations_for_group(group_ratings[~group_ratings.item.isin(all_filter_out_items)],
                                                                     recommendations_number=conf['k'])
        rec_list = rec_list[algo_name]
        rec_list = enrich_results(rec_list, loader)

        if selected_choice_model == "Same":
            rec_list[0]["group_choice"] = True

        # Add information about predicted rating w.r.t. each of the group members
        for idx in range(len(rec_list)):
            movie_idx = int(rec_list[idx]["movie_idx"])
            predicted_ratings = group_ratings[group_ratings.item == movie_idx].set_index("user")
            assert len(predicted_ratings) == len(group_members), f"{len(predicted_ratings)} != {len(group_members)}"
            
            group_member_ratings_raw, group_member_ratings_normed, group_member_ratings_normed_percentage = dict(), dict(), dict()
            for g_uid in group_members:
                group_member_ratings_raw[g_uid] = predicted_ratings.loc[g_uid].predicted_rating_raw
                group_member_ratings_normed[g_uid] = predicted_ratings.loc[g_uid].predicted_rating
                group_member_ratings_normed_percentage = round(group_member_ratings_normed[g_uid] * 100)

            rec_list[idx]["extra_data"] = {
                "group_member_ratings_raw": group_member_ratings_raw,
                "group_member_ratings_normed": group_member_ratings_normed,
                "group_member_ratings_normed_percentage": group_member_ratings_normed_percentage,
            }

        full_recs[algo_name] = {
            "movies": rec_list,
            "order": order,
            "filter_out_items": filter_out_items,
            "all_filter_out_items": all_filter_out_items,
        }

    # "Same" was already handled above
    # "None" does not need any handling
    if selected_choice_model not in ["Same", "None"]:
        for algo_anon, recs in group_recommendations.items():

            algo_name = algo_anon_to_name[algo_anon]

            # Get the original algorithm name
            rec_list = recs["movies"]
            rec_list_indices = [int(x["movie_idx"]) for x in rec_list]

            # Limit ourselves to items that are present in top-k (rec_list) as these are those that we are selecting from
            ratings_to_choose_from = group_ratings[group_ratings.item.isin(rec_list_indices)].copy()

            group_ratings_rm_copy = group_ratings_rm.copy()
            all_items = np.arange(group_ratings_rm.shape[1])
            group_ratings_rm_copy[:, np.setdiff1d(all_items, rec_list_indices)] = NEG_INF

            if selected_choice_model == "RLProp":
                # We use precomputed rating matrix to save on time (since we would pivot inside RLProp and that would take unneccessary time)
                [the_choice] = algo.generate_group_recommendations_for_group(ratings_to_choose_from,
                                                                        recommendations_number=1,
                                                                        group_ratings_rm=group_ratings_rm_copy)
            else:
                [the_choice] = algo.generate_group_recommendations_for_group(ratings_to_choose_from,
                                                                         recommendations_number=1)
            idx = rec_list_indices.index(the_choice)
            assert idx >= 0, f"rec_list={rec_list_indices}, the_choice={the_choice}"
            full_recs[algo_name]["choice"] = the_choice
            group_recommendations[algo_anon]["movies"][idx]["group_choice"] = True

    full_recs_data["recommendations"] = full_recs
    full_recs_data["algorithm_name_mapping"] = algo_anon_to_name
    full_recs_data["shown_items"] = shown_items
    full_recs_data["group_choices"] = group_choices

    log_interaction(session["participation_id"], "generate-group-recommendations", **full_recs_data)

    # If extra explanations are disabled, then we deleted group_member_ratings so that they do not leak to the participants
    extended_explanations = user_data["selected_extended_explanations"]
    if not extended_explanations:
        for algo_name, vals in group_recommendations.items():
            for it in vals["movies"]:
                it["extra_data"]["group_member_ratings"] = None
                it["extra_data"]["group_member_ratings_normed"] = None
                it["extra_data"]["group_member_ratings_normed_percentage"] = None

    return group_recommendations

@bp.route("/group-gen", methods=['GET'])
def group_gen():
    start_time = time.perf_counter()
    user_data = get_all(get_uname())

    conf = load_user_study_config(session['user_study_id'])

    if conf["group_source"] == "Pre-generated":
        print("USING PRE-GENERATED GROUPS")

    is_user_part_of_group = user_data['selected_user_part_of_group']

    # Get a loader
    loader_factory = load_data_loaders()[conf["selected_data_loader"]]
    loader = loader_factory(**filter_params(conf["data_loader_parameters"], loader_factory))
    loader = load_data_loader_cached(loader, session['user_study_guid'], loader_factory.name(), get_semi_local_cache_name(loader))

    print(f"Until we got the data loader it took: {time.perf_counter() - start_time}")

    n_items = loader.rating_matrix.shape[1]

    history_length_limit = conf["user_history_length"]

    if is_user_part_of_group:
        user_embedding = np.zeros(shape=(n_items, ), dtype=np.int8)
        user_embedding[user_data['elicitation_selected_movies']] = 1
        assert sum(user_embedding) > 0, f"{len(user_embedding)} <= 0, {user_data['elicitation_selected_movies']}"
        groups = generate_groups(loader, user_data['selected_group_types'], conf['group_size'], user_embedding, history_length_limit)
    else:
        groups = generate_groups(loader, user_data['selected_group_types'], conf['group_size'], user_embedding=None, history_length_limit=history_length_limit)

    print(f"Until end of group generation it took: {time.perf_counter() - start_time}")

    # We only store indices and no embeddings as these could be easily retrieved later
    # from the pre-loaded rating matrix
    groups_indices = [groups[idx]['indices'] for idx, _ in enumerate(user_data['selected_group_types'])]
    groups_indices_named = {gtype: groups[idx]['indices'] for idx, gtype in enumerate(user_data['selected_group_types'])}
    groups_extra_data = {gtype: groups[idx]['extra_data'] for idx, gtype in enumerate(user_data['selected_group_types'])}
    initial_iteration = 0
    new_data = {
        "groups_indices": groups_indices,
        "groups_indices_named": groups_indices_named,
        "groups_extra_data": groups_extra_data,
        "iteration": initial_iteration # Mark the beginning here
    }

    set_mapping(get_uname(), new_data)
    set_val('groups', groups)

    log_interaction(session["participation_id"], "group-gen", **new_data)

    # We are about to start the very first iteration, so no items were shown yet
    shown_items = { algo_name: { gtype: [] for gtype in user_data['selected_group_types'] } for algo_name in ALGORITHM_ANON_NAMES }
    group_choices = { algo_name: { gtype: [] for gtype in user_data['selected_group_types'] } for algo_name in ALGORITHM_ANON_NAMES }
    set_val('shown_items', shown_items)
    set_val('group_choices', group_choices)
    # Group type for which we generate in initial_iteration
    target_group_type = user_data['generated_iters'][initial_iteration]['group_type']
    group_recommendations = generate_group_recommendations(conf, loader, initial_iteration, shown_items, group_choices)

    print(f"## Shown before: {shown_items}, choices before: {group_choices}")
    for anon_name, recs in group_recommendations.items():
        shown_items[anon_name][target_group_type].append([int(x['movie_idx']) for x in recs['movies']])
        group_choices[anon_name][target_group_type].append([int(x['movie_idx']) for x in recs['movies'] if 'group_choice' in x and x['group_choice']])
    print(f"## Shown after: {shown_items}, choices after: {group_choices}")
    set_val('shown_items', shown_items)
    set_val('group_choices', group_choices)

    print(f"Until all recommendations got generated it took: {time.perf_counter() - start_time}")

    set_mapping(get_uname() + ":grs_movies", {
        "movies": group_recommendations
    })

    print(f"Generated recommendations: {group_recommendations}")

    data = {
        "recommendations": group_recommendations,
        "iteration": initial_iteration
    }
    log_interaction(session["participation_id"], "group-recommendation-started", **data)

    return redirect(url_for("grs2024.compare_grs"))


@bp.route("/compare-grs", methods=["GET", "POST"])
def compare_grs():
    tr = get_tr(languages, get_lang())

    user_data = get_all(get_uname())

    #params = session['alpha_movies']
    u_key = get_uname()
    params = get_all(u_key + ":grs_movies")
    params["continuation_url"] = url_for("grs2024.grs_feedback")
    params["hint"] = tr("grs2024_compare_grs_hint")
    params["header"] = tr("grs2024_compare_grs_header")
    params["title"] = tr("grs2024_compare_grs_title")
    params["drag"] = tr("grs2024_compare_grs_drag")
    params["n_iterations"] = len(user_data["generated_iters"]) # We have 1 iteration for actual metric assessment followed bz N_ALPHA_ITERS iterations for comparing alphas
    params["iteration"] = user_data["iteration"]
    if ENABLE_DEBUG:
        params["grs2024_debug"] = f"DEBUG: {user_data['generated_iters'][user_data['iteration']]}"
    # -1 to make it zero based, another -1 because we count 2 N_ALPHA_ITERS and one for metric-assessment, together we have -2
    params["algorithm_offset"] = 0 #(get_val("alphas_iteration") - 2) * 3
    return render_template("compare_grs.html", **params)

@bp.route("/grs-feedback", methods=["GET", "POST"])
def grs_feedback():

    user_data = get_all(get_uname())

    curr_iter = user_data["iteration"]
    print(f"Finished with curr_iter = {curr_iter}")

    drag_and_drop_positions = json.loads(request.form.get("drag_and_drop_positions"))
    dropzone_position = json.loads(request.form.get("dropzone_position"))

    feedback_data = {
        'iteration': curr_iter,
        'drag_and_drop_positions': drag_and_drop_positions,
        'dropzone_position': dropzone_position
    }

    print(f"Feedback data: {feedback_data}")

    log_interaction(session["participation_id"], "group-recommendation-ended", **feedback_data)


    conf = load_user_study_config(session['user_study_id'])

    # Get a loader
    loader_factory = load_data_loaders()[conf["selected_data_loader"]]
    loader = loader_factory(**filter_params(conf["data_loader_parameters"], loader_factory))
    loader = load_data_loader_cached(loader, session['user_study_guid'], loader_factory.name(), get_semi_local_cache_name(loader))

    shown_items = get_val('shown_items')
    group_choices = get_val('group_choices')
    print(f"@@ Shown items before: {shown_items}, group_choices before: {group_choices}")
    

    # Iterations are zero-based
    # We have len(GROUP_TYPES) * conf['iters_per_group_type'] * len(algorithm_combinations)
    # iterations and questionnaire is shown when changing algorithms (i.e. after len(GROUP_TYPES) * conf['iters_per_group_type'])
    if (curr_iter + 1) % (len(GROUP_TYPES) * conf['iters_per_group_type']) == 0:
        
        if curr_iter + 1 >= len(user_data["generated_iters"]):
            print("End of study soon")
            pass # do not generate recommendations, its end of study now
        else:
            # Group type for which we generate in initial_iteration
            target_group_type = user_data['generated_iters'][curr_iter + 1]['group_type']
            # generate recommendations that will be shown later, after finishing block questionnaire
            group_recommendations = generate_group_recommendations(conf, loader, curr_iter + 1, shown_items, group_choices)
            print(f"## Generated recommendations: {group_recommendations}")
            set_mapping(get_uname() + ":grs_movies", {
                "movies": group_recommendations
            })

            for anon_name, recs in group_recommendations.items():
                shown_items[anon_name][target_group_type].append([int(x['movie_idx']) for x in recs['movies']])
                group_choices[anon_name][target_group_type].append([int(x['movie_idx']) for x in recs['movies'] if 'group_choice' in x and x['group_choice']])


        print("End of block")
        # Redirect to after block questionnaire
        next_page = "grs2024.block_questionnaire"
    else:
        # Redirect to next iteration
        print(f"Not end of block")
        # Group type for which we generate in initial_iteration
        target_group_type = user_data['generated_iters'][curr_iter + 1]['group_type']
        # Generate group recommendations
        group_recommendations = generate_group_recommendations(conf, loader, curr_iter + 1, shown_items, group_choices)

        for anon_name, recs in group_recommendations.items():
            shown_items[anon_name][target_group_type].append([int(x['movie_idx']) for x in recs['movies']])
            group_choices[anon_name][target_group_type].append([int(x['movie_idx']) for x in recs['movies'] if 'group_choice' in x and x['group_choice']])

        print(f"## Generated recommendations: {group_recommendations}")

        set_mapping(get_uname() + ":grs_movies", {
            "movies": group_recommendations
        })

        data = {
            "recommendations": group_recommendations,
            "iteration": curr_iter + 1
        }

        log_interaction(session["participation_id"], "group-recommendation-started", **data)
        next_page = "grs2024.compare_grs"

    print(f"@@ Shown items after: {shown_items}, group choices after: {group_choices}")
    set_val('shown_items', shown_items)
    set_val('group_choices', group_choices)
    incr("iteration")
    continuation_url = url_for('grs2024.grs_feedback')
    return redirect(url_for(next_page, continuation_url=continuation_url))


# Endpoint for block questionnaire
@bp.route("/block-questionnaire", methods=["GET", "POST"])
def block_questionnaire():
    user_data = get_all(get_uname())
    conf = load_user_study_config(session['user_study_id'])

    it = user_data["iteration"]
    n_iterations_per_block = (len(GROUP_TYPES) * conf['iters_per_group_type'])
    n_blocks = len(user_data["generated_iters"]) // n_iterations_per_block
    cur_block = (int(it) - 1) // n_iterations_per_block
    params = {
        "continuation_url": url_for("grs2024.block_questionnaire_done"),
        "header": f"After-recommendation block questionnaire for block: ({cur_block + 1}/{n_blocks})",
        "hint": f"Before proceeding to the next step, please answer questions specific to the recent block (last {n_iterations_per_block} iterations) of recommendations.",
        "finish": "Continue",
        "title": "Questionnaire"
    }
    return render_template("grs_block_questionnaire.html", **params)

# Endpoint that should be called once block questionnaire is done
@bp.route("/block-questionnaire-done", methods=["GET", "POST"])
def block_questionnaire_done():
    user_data = get_all(get_uname())
    conf = load_user_study_config(session['user_study_id'])

    it = user_data["iteration"]
    print(f"Iteration = {it}")
    n_iterations_per_block = (len(GROUP_TYPES) * conf['iters_per_group_type'])
    n_blocks = len(user_data["generated_iters"]) // n_iterations_per_block
    cur_block = (int(it) - 1) // n_iterations_per_block
    # Take from last iteration of the block (that is why we have -1 since it was already incremented)
    current_data = user_data["generated_iters"][it - 1]

    # Log the iteration block, algorithm as well as responses to all the questions
    data = {
        "block": cur_block,
        "group_data": current_data,
        "iteration": it - 1 # Last iteration of the block
    }
    data.update(**request.form)

    if cur_block == n_blocks - 1:
        # We are done, do not forget to mark last iteration as ended
        log_interaction(session["participation_id"], "after-block-questionnaire", **data)
        return redirect(url_for("grs2024.done"))
    else:
        # Otherwise continue with next block
        log_interaction(session["participation_id"], "after-block-questionnaire", **data)
        # We already generated group recommendations, but have not logged the corresponding
        # group-recommendation-started interaction, do it now
        u_key = get_uname()
        group_recommendations = get_all(u_key + ":grs_movies")
        data_rec = {
            "recommendations": group_recommendations,
            "iteration": it # First iteration of the new block, so no decrement
        }
        log_interaction(session["participation_id"], "group-recommendation-started", **data_rec)
        return redirect(url_for("grs2024.compare_grs"))

@bp.route("/done", methods=["GET", "POST"])
def done():
    #return "DONE - TODO"
    #return redirect(url_for("journal.final_questionnaire"))
    return redirect(url_for("grs2024.finish_user_study"))

# Endpoint for final questionnaire
@bp.route("/final-questionnaire", methods=["GET", "POST"])
def final_questionnaire():
    params = {
        "continuation_url": url_for("journal.finish_user_study"),
        "header": "Final questionnaire",
        "hint": "Please answer the questions below before finishing the user study. Note that these questions are related to the whole user study.",
        "finish": "Finish",
        "title": "Final questionnaire"
    }
    return render_template("journal_final_questionnaire.html", **params)

@bp.route("/finish-user-study", methods=["GET", "POST"])
def finish_user_study():
    # Handle final questionnaire feedback, logging all the answers
    #data = {}
    #data.update(**request.form)
    #log_interaction(session["participation_id"], "final-questionnaire", **data)

    session["iteration"] = int(get_val("iteration"))
    session.modified = True
    return redirect(url_for("utils.finish"))

# Long-running initialization
@bp.route("/initialize", methods=["GET"])
def initialize():
    guid = request.args.get("guid")
    cont_url = request.args.get("continuation_url")
    return redirect(url_for("fastcompare.initialize", guid=guid, continuation_url=cont_url, consuming_plugin=__plugin_name__))

#################################################################################
########################### HELPER ENDPOINTS  ###################################
#################################################################################

@bp.route("/get-block-questions", methods=["GET"])
def get_block_questions():
   
    user_data = get_all(get_uname())
    conf = load_user_study_config(session['user_study_id'])
    it = user_data["iteration"]
    n_iterations_per_block = (len(GROUP_TYPES) * conf['iters_per_group_type'])
    n_blocks = len(user_data["generated_iters"]) // n_iterations_per_block
    cur_block = (int(it) - 1) // n_iterations_per_block
    #cur_algorithm = user_data["selected_algorithms"][cur_block]

    conf = load_user_study_config(session['user_study_id'])

    extended_explanations = user_data["selected_extended_explanations"]

    if "Goodbooks" in conf["selected_data_loader"]:
        item_text = "books"
        popular_text = "bestsellers"
        do_text = "read"
    else:
        item_text = "movies"
        popular_text = "blockbusters"
        do_text = "watch"

    questions = [
        {
            "text": f"The recommendations were generally fair to all group members.",
            "name": "q1",
            "icon": "grid"
        },
        {
            "text": "Assessing the fairness of recommendations was challenging.",
            "name": "q2",
            "icon": "grid",
        },
        {
            "text": "The provided information was sufficient to evaluate the fairness of recommendations.",
            "name": "q3",
            "icon": "grid",
        },
        {
            "text": "Information about group members' history / previously liked movies helped evaluate the recommendations' fairness.",
            "name": "q4",
            "icon": "grid",
        },
        {
            "text": "There were group members whose preferences were often overlooked in the recommendations.",
            "name": "q5",
            "icon": "grid",
        },
    ]

    if extended_explanations:
        questions.append(
            {
                "text": "Knowing the estimated preference of users (on a 0% - 100% scale) towards recommended items simplified evaluating the fairness of the recommendations.",
                "name": "q6",
                "icon": "grid",
            }
        )


    questions.extend([
        {
            "text": "The highlighted group's choices were generally fair to all members.",
            "name": "q7",
            "icon": "ui-checks-grid",
        },
        {
            "text": "The highlighted group's choices were generally the best option among the recommended items.",
            "name": "q8",
            "icon": "ui-checks-grid",
        },
        {
            "text": "Assessing the fairness of the group's choices was challenging.",
            "name": "q9",
            "icon": "ui-checks-grid",
        },
        {
            "text": "The provided information was sufficient to evaluate the fairness of the group's choices.",
            "name": "q10",
            "icon": "ui-checks-grid",
        },
        {
            "text": "Information about group members' history / previously liked movies helped to evaluate the fairness of group choice.",
            "name": "q11",
            "icon": "ui-checks-grid",
        },
        {
            "text": "There were group members whose preferences were often overlooked in the group choices.",
            "name": "q12",
            "icon": "ui-checks-grid",
        },
    ])

    if extended_explanations:
        questions.append(
            {
                "text": "Knowing the estimated preference of users (on a 0% - 100% scale) towards recommended items simplified evaluating the fairness of the group choices.",
                "name": "q13",
                "icon": "ui-checks-grid",
            }
        )

    questions.extend([
        {
            "text": "Did you notice any changes in the fairness of recommendations across iterations?",
            "name": "q14",
            "icon": "arrow90deg-right",
        },
        {
            "text": "The recommendations included options for every group member (i.e., each user had at least one preferred item in the recommendations).",
            "name": "q15",
            "icon": "grid",
        },
    ])


    attention_checks = [
        {
            "text": "I believe recommender systems can be very useful to people. To answer ",
            "text2": "this attention check question correctly, you must select 'Agree'.",
            "name": "qs1",
            "icon": "grid"
        },
        {
            "text": "Using this recommender system was entertaining and I would recommend it to my friends. To answer ",
            "text2": "this attention check question correctly, you must select 'Strongly Disagree'.",
            "name": "qs2",
            "icon": "ui-checks-grid"
        },
        {
            "text": "This group recommender system provided many tips for interesting computer games.",
            "name": "qs3",
            "icon": "ui-checks-grid",
            "atn": "true"
        },
        {
            "text": "The group recommendations produced by this system provided great recipes for exotic cuisines.",
            "name": "qs4",
            "icon": "ui-checks-grid",
            "atn": "true"
        }
    ]

    # Mapping blocks to indices at which we place attention checks
    atn_check_indices = {
        0: 4,
        1: 6,
        2: 10,
        3: 9,
    }

    # if cur_algorithm == "RELEVANCE-BASED":
    #     # If this is the case, we only show first 9 questions
    #     questions = questions[:9]
    #     # However, this also means that we have to redistribute attention checks!
    #     atn_check_indices = {
    #         0: 1,
    #         1: 6,
    #         2: 9,
    #         3: 8
    #     }
    #     # No sliders icon for relevance-only so having different icon for attention check would make it too easy
    #     attention_checks[3]["icon"] = "grid"
    #     attention_checks[2]["icon"] = "grid"
    #     attention_checks[1]["icon"] = "grid"

    # Use first attention check
    questions.insert(atn_check_indices[cur_block], attention_checks[cur_block])
    if cur_block == 1:
        # For second block we have two attention checks
        questions.insert(atn_check_indices[3], attention_checks[3])
    assert cur_block >= 0 and cur_block < n_blocks, f"cur_block == {cur_block}"

    return jsonify(questions)

def is_books(*args):
    return False

@bp.route("/get-group-member-histories", methods=["GET"])
def get_group_member_histories():
    user_data = get_all(get_uname())

    print("Getting group member histories")
    g_data = user_data["groups_extra_data"]
    #print(f"Got data: {g_data}")
    #print(f"Got iteration: {get_val('iteration')}")
    gen_iters = user_data["generated_iters"]
    #print(f"Generated iters: {gen_iters}")

    current_iter_data = gen_iters[user_data['iteration']]
    #print(f"Current iter: {current_iter_data}")

    current_group_type = current_iter_data['group_type']
    #print(f"Current group type: {current_group_type} and histories: {g_data[current_group_type]}")
    #print(f"Group indices: {get_val('groups_indices')}")
    #print(f"Group indices named: {get_val('groups_indices_named')}")
    
    group_indices_named = user_data['groups_indices_named'] #get_val('groups_indices_named')

    # Get a loader
    conf = load_user_study_config(session['user_study_id'])
    loader_factory = load_data_loaders()[conf["selected_data_loader"]]
    loader = loader_factory(**filter_params(conf["data_loader_parameters"], loader_factory))
    loader = load_data_loader_cached(loader, session['user_study_guid'], loader_factory.name(), get_semi_local_cache_name(loader))

    # We know that each block corresponds to single group type
    # Thus we map names based on blocks
    # We do 'A' + group_size * Block + user_idx
    n_iterations_per_block = (len(GROUP_TYPES) * conf['iters_per_group_type'])
    cur_block = (int(user_data['iteration'])) // n_iterations_per_block

    result = dict()
    #member_to_idx = dict()
    member_to_name = dict()
    for idx, member in enumerate(group_indices_named[current_group_type]):
        #print(f"\tg_data={g_data}, current_group_type={current_group_type}")
        #print(f"\tg_data[current_group_type]={g_data[current_group_type]}")
        #print(f"\tg_data[current_group_type]['past_positive_history']={g_data[current_group_type]['past_positive_history']}")
        #print(f"\tMember: {member}, type={type(member)}")
        history_item_indices = g_data[current_group_type]['past_positive_history'][member]
        result[member] = enrich_results(history_item_indices, loader)
        print(ord('A'), idx, cur_block, conf['group_size'], "Result:", ord('A') + idx + cur_block * conf['group_size'])
        member_to_name[member] = str(chr(ord('A') + idx + cur_block * conf['group_size']))
        #member_to_idx[member] = idx

    #print(f"Final result: {result}")
    print(member_to_name)
    return jsonify({
        "histories": result,
        "member_mapping": member_to_name,
    })


@bp.route("/get-instruction-bullets", methods=["GET"])
def get_instruction_bullets():
    page = request.args.get("page")
    if not page:
        return jsonify([])

    conf = load_user_study_config(session['user_study_id'])

    user_data = get_all(get_uname())
    extended_explanations = user_data["selected_extended_explanations"]
    
    # Iteration may not be present when this is called for pre-study-questionnaire:
    if "iteration" in user_data:
        it = user_data["iteration"]
        n_iterations_per_block = (len(GROUP_TYPES) * conf['iters_per_group_type'])
        n_blocks = len(user_data["generated_iters"]) // n_iterations_per_block
        
        if page == "compare-grs":
            cur_block = (int(it)) // n_iterations_per_block
            first_iter_of_block = cur_block * n_iterations_per_block
            prev_algos = list(user_data["generated_iters"][first_iter_of_block]["algorithms"].keys())
        elif page == "block-questionnaire":
            cur_block = (int(it) - 1) // n_iterations_per_block
            first_iter_of_block = cur_block * n_iterations_per_block
            prev_algos = list(user_data["generated_iters"][first_iter_of_block]["algorithms"].keys())

    if page == "mors":
        if is_books(conf):
            bullets = [
                "Books are selected by clicking on them; each selected book is highlighted by a green frame."
                "If you do not like any of the recommended books, there is a button at the bottom of this page that you should check.",
                "When a mouse cursor is placed over a book, its title (including authors), description, and genres will be shown.",
                "Completion of each step is final, and you cannot return to previous pages.",
                "Also note that each book will be displayed only once within the block (i.e., you need to make an immediate decision)."
            ]
        else:
            bullets = [
                "Movies are selected by clicking on them; each selected movie is highlighted by a green frame.",
                "If you do not like any of the recommended movies, there is a button at the bottom of this page that you should check.",
                "When a mouse cursor is placed over a movie, its title (including release year), description, and genres will be shown.",
                "Completion of each step is final, and you cannot return to previous pages.",
                "Also note that each movie will be displayed only once within the block (i.e., you need to make an immediate decision)."
            ]
    elif page == "block-questionnaire":
        bullets = [
            f"Important: These questions are about the recent recommendations, specifically recommendations from the last block (meaning the last {n_iterations_per_block} iterations).",
            f"These {n_iterations_per_block} iterations covered a single group receiving {conf['iters_per_group_type']} iteration(s) from each of the 3 group recommenders.",
            "Please answer them before moving on to the next step in the study.",
            "If any question is unclear, choose 'I don't understand' as your response for that specific question."
        ]
    elif page == "pre-study-questionnaire":
        bullets = [
            "Please answer these questions before moving on to the next step in the study.",
            "For question 6, pick the answer that most aligns with your personal understanding of fairness in context of group recommender systems.",
            "Considering that different people may have different views/interpretations, you can choose 'Other' and share your personal thoughts in the text box below each question."
        ]
    elif page == "final-questionnaire":
        bullets = [
            "Important: These questions are about the experience during the WHOLE user study.",
            "Please answer these questions before finishing the study."
        ]
    elif page == "metric-assessment":
        items_name = "books" if is_books(conf) else "movies"
        extra_info = "authors" if is_books(conf) else "release year"
        bullets = [
            f"This page displays three lists of recommendations: A, B, C, which were created for you based on the {items_name} you chose at the start of this study as part of stating your preferences (preference elicitation).",
            "Your goal is to choose the one list that you think is the most DIVERSE (based on how you understand or interpret diversity).",
            f"You can choose a list by clicking on any of its {items_name}.",
            f"If you want more information about the displayed {items_name}, simply hover your mouse cursor over it, and the description, genres, and name (including {extra_info}) will all appear."
        ]
    elif page == "compare-alphas":
        items_name = "books" if is_books(conf) else "movies"
        extra_info = "authors" if is_books(conf) else "release year"
        bullets = [
            f"This page once again presents three recommendation lists, but they differ from the previous step and are now labeled either 1, 2, 3 (in the first step) or 4, 5, 6 (in the second step).",
            "Your goal is to order the recommendation lists from least diverse to most diverse.",
            "To order them, drag and drop the colorful rectangles located at the bottom of this page into the adjacent gray area.",
            "The further right you position the lists, the more diversity you perceive in them.",
            f"If you want more information about the displayed {items_name}, simply hover your mouse cursor over it, and the description, genres, and name (including {extra_info}) will all appear."
        ]
    elif page == "compare-grs":
        items_name = "books" if is_books(conf) else "movies"
        extra_info = "authors" if is_books(conf) else "release year"
        bullets = [
            f"The top part of the screen shows the group for which the recommendation is generated, consisting of three users. You can view the movies each user liked in the past.",
            "Below that are two recommendation lists, each generated by a different algorithm for the given group.",
            "For each recommendation list, one movie represents a group choice. These are highlighted in the green border.",
            "Your task is to place the four rectangles into the gray drag-and-drop area based on their fairness (less fair on the left, more fair on the right).",
            "These rectangles represent the two recommendation lists and their corresponding group choices.",
            "Hovering over a movie item reveals additional details, including its name, genres, and plot that could help you with this task.",
        ]

        if extended_explanations:
            bullets.append("Additionally, below each recommended movie, you can see each user's preference for that movie on a 0% to 100% scale")
    else:
        bullets = []

    return jsonify(bullets)


# Plugin related functionality
def register():
    return {
        "bep": dict(blueprint=bp, prefix=None),
    }

@bp.context_processor
def plugin_name():
    return {
        "plugin_name": __plugin_name__
    }
