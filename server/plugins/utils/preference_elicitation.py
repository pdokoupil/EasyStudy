# If running from python, do it as python -m preference_elicitation to avoid issues with relative imports !!!!

import os
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import pickle


import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.config.set_visible_devices([], 'GPU') # Disable GPU because of adagrad issues
# tf.random.set_seed(42)
# np.random.seed(42)
# random.seed(42)


from popularity_sampling import PopularitySamplingElicitation, PopularitySamplingFromBucketsElicitation
from multi_obj_sampling import MultiObjectiveSamplingFromBucketsElicitation
from tfrs_model import get_model_mf

import time
from sklearn.preprocessing import QuantileTransformer


from rlprop_wrapper import RLPropWrapper
from normalization.identity import identity
from normalization.cdf import cdf
from normalization.standardization import standardization
from mandate_allocation.exactly_proportional_fuzzy_dhondt_2 import exactly_proportional_fuzzy_dhondt_2
from mandate_allocation.weighted_average_strategy import weighted_average_strategy

from data_loading import load_ml_dataset, MLDataLoader

from common import get_abs_project_root_path

MOST_RATED_MOVIES_THRESHOLD = 200
USERS_RATING_RATIO_THRESHOLD = 0.75
NUM_TAGS_PER_GROUP = 3
NUM_CLUSTERS = 6
NUM_CLUSTERS_TO_PICK = 1
NUM_MOVIES_PER_TAG = 2
MIN_NUM_TAG_OCCURRENCES = 50 # NUM_MOVIES_PER_TAG * NUM_TAGS_PER_GROUP # Calculation based because of the Deny list #10

MIN_RATINGS_PER_USER = 500
MIN_RATINGS_PER_MOVIE = 500
MIN_TAGS_PER_MOVIE = 50


USE_LOCAL_IMAGES = True

import os
import functools
cluster_data = None

basedir = os.path.abspath(os.path.dirname(__file__))

groups = None

result_layout_variants = [
    "columns",
    "column-single",
    "rows",
    "row-single",
    "row-single-scrollable",
    "max-columns"
]

# Decorator that will cache the result of the decorated function after first call
# difference from functools.cache is that it ignores the value of the parameters
def compute_once(func):
    result = None
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal result
        if not result:
            result = func(*args, **kwargs)
        return result
    return wrapper

@compute_once
def prepare_tf_data(loader):
    ratings_df = loader.ratings_df.copy()

    # Add item_title
    ratings_df.loc[:, "item_title"] = ratings_df.movieId.map(loader.movies_df_indexed.title)

    # Rename column and cast to string
    ratings_df = ratings_df.rename(columns={"userId": "user_id"})
    ratings_df.user_id = ratings_df.user_id.astype(str)

    ratings = tf.data.Dataset.from_tensor_slices(dict(ratings_df[["user_id", "item_title"]]))
    movies = tf.data.Dataset.from_tensor_slices(dict(loader.movies_df.rename(columns={"title": "item_title"})[["item_title"]])).map(lambda x: x["item_title"])

    import numpy as np
    tf.random.set_seed(42)
    shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

    train_size = int(ratings_df.shape[0] * 0.85)

    # Take everything as train
    train = shuffled

    movie_titles = movies.batch(1_000)
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles))) # loader.movies_df.title.str.encode("utf-8").unique()
    unique_user_ids = np.unique(np.concatenate(list(user_ids))) # ratings_df.user_id.str.encode("utf-8").unique()

    new_user = str(max([int(x) for x in unique_user_ids]) + 1)
    unique_user_ids = np.concatenate([unique_user_ids, np.array([new_user])])

    cached_train = train.shuffle(100_000).batch(8192).cache()

    return unique_user_ids, unique_movie_titles, movies, cached_train, train

def prepare_tf_model(loader):

    unique_user_ids, unique_movie_titles, movies, cached_train, train = prepare_tf_data(loader)
    model = get_model_mf(unique_user_ids, unique_movie_titles, movies)
    #cache_path = os.path.join(Path(__file__).parent.absolute(), 'cache', 'utils', 'ml-latest', 'tf_weights_cache')
    cache_path = os.path.join(get_abs_project_root_path(), 'cache', 'utils', 'ml-latest', 'tf_weights_cache')

    # Try load
    try:
        model.load_weights(cache_path)
    except tf.errors.NotFoundError as ex:
        model.fit(cached_train, epochs=5)
        model.save_weights(cache_path)

    return model, train

def load_data(loader, elicitation, elicitation_movies):
    data = elicitation.get_initial_data([int(x["movie_idx"]) for x in elicitation_movies])
    return enrich_results(data, loader)

def load_data_1(elicitation_movies):
    loader = load_ml_dataset()

    # Get list of items
    data = MultiObjectiveSamplingFromBucketsElicitation(
        loader.rating_matrix,
        loader.distance_matrix,
        2, 2, 2, 4, 1.0
    ).get_initial_data([int(x["movie_idx"]) for x in elicitation_movies])

    # TODO user enrich_results instead
    

    # Result is a list of movies, each movie being a dict (JSON object)
    return enrich_results(data, loader)

def load_data_2(elicitation_movies):
    
    loader = load_ml_dataset()

    # Get list of items
    data = PopularitySamplingElicitation(loader.rating_matrix, n_samples=16).get_initial_data([int(x["movie_idx"]) for x in elicitation_movies])

    # TODO use unrich_results instead
    res = [loader.movie_index_to_description[movie_idx] for movie_idx in data]
    res_url = [loader.get_image(movie_idx) for movie_idx in data]
    result = [{"movie": movie, "url": url, "movie_idx": str(movie_idx)} for movie, url, movie_idx in zip(res, res_url, data)]
    # Result is a list of movies, each movie being a dict (JSON object)
    return result

def load_data_3(elicitation_movies):
    loader = load_ml_dataset()
    # Get list of items
    # TODO use enrich_results instead
    data = PopularitySamplingFromBucketsElicitation(loader.rating_matrix, 5, [4]*5).get_initial_data([int(x["movie_idx"]) for x in elicitation_movies])
    res = [loader.movie_index_to_description[movie_idx] for movie_idx in data]
    res_url = [loader.get_image(movie_idx) for movie_idx in data]
    result = [{"movie": movie, "url": url, "movie_idx": str(movie_idx)} for movie, url, movie_idx in zip(res, res_url, data)]
    # Result is a list of movies, each movie being a dict (JSON object)
    return result

def calculate_weight_estimate(loader, selected_movies, elicitation_movies, return_supports=False):
    if not selected_movies:
        x = np.array([1.0, 1.0, 1.0])
        return x / x.sum(), {}

    #loader = load_ml_dataset()

    rel = 0
    div = 0
    nov = 0

    selected_relevances = []
    selected_diversities = []
    selected_novelties = []

    #distance_matrix = 1.0 - loader.similarity_matrix

    
    if type(elicitation_movies[0]) == int:
        movie_indices = np.unique(elicitation_movies)
    elif type(elicitation_movies[0]) == dict:
        movie_indices = np.unique([int(movie["movie_idx"]) for movie in elicitation_movies])
    else:
        assert False

    diversities = loader.distance_matrix[np.ix_(movie_indices, movie_indices)]
    raw_diversities = diversities
    #diversities = diversities[np.triu_indices(diversities.shape[0])].reshape((-1, 1)) # Train diversity on this
    diversities = (diversities.sum(axis=0) / (diversities.shape[0] - 1)).reshape(-1,1)

    # relevances = loader.rating_matrix[:, movie_indices].T # Train other CDF on this
    relevances = loader.rating_matrix[:, movie_indices] # Train other CDF on this
    relevances = relevances.mean(axis=0)
    # relevances = relevances[relevances > 0.0]
    relevances = relevances.reshape((-1, 1))
    users_viewed_item = loader.rating_matrix.astype(bool).sum(axis=0)
    novelties = 1.0 - (users_viewed_item / loader.rating_matrix.shape[0])
    novelties = novelties[movie_indices].reshape((-1, 1))

    # for movie_idx in movie_indices:
    #     if movie_idx in selected_movies:
    #         r = loader.rating_matrix[:, movie_idx]
    #         #r = r[r > 0.0]
    #         r = r.mean(axis=0)
    #         #r = r.mean()
    #         selected_relevances.append(r) # TODO try without transposition
    #         # selected_diversities.append(distance_matrix[movie_idx][movie_indices].mean())
    #         d = distance_matrix[movie_idx][movie_indices].sum() / (len(movie_indices) - 1)
    #         selected_diversities.append(d[d > 0].reshape(-1, 1))
    #         selected_novelties.append(1.0 - (loader.rating_matrix[:, movie_idx].astype(bool).sum() / loader.rating_matrix.shape[0]))

    for movie_idx in selected_movies:
        r = loader.rating_matrix[:, movie_idx]
        r = r.mean(axis=0)
        selected_relevances.append(r)
        d = loader.distance_matrix[movie_idx][movie_indices].sum() / (len(movie_indices) - 1)
        selected_diversities.append(d[d > 0].reshape(-1, 1))
        selected_novelties.append(1.0 - (loader.rating_matrix[:, movie_idx].astype(bool).sum() / loader.rating_matrix.shape[0]))

    
    if (not selected_relevances) or (not selected_diversities) or (not selected_novelties):
        print(f"Something weird again: {movie_indices}, {selected_movies}")
        x = np.array([1.0, 1.0, 1.0])
        return x / x.sum(), {}

    # Take relevance values of all items and train CDF on it
    # Calculate CDF for relevances of all selected items and take their average
    rel_cdf = QuantileTransformer()
    rel_cdf.fit(relevances)
    x = np.stack(selected_relevances).reshape(-1, 1)
    rel = np.mean(rel_cdf.transform(x))


    # Take diversity values of all items shown during elicitation and train CDF on it
    # Calculate CDF for diversities of all selected items and take their average
    div_cdf = QuantileTransformer()
    div_cdf.fit(diversities)
    div = np.mean(div_cdf.transform(np.stack(selected_diversities).reshape(-1, 1)))

    # Take novelty values of all items and train CDF on it
    # Calculate CDF for novelties of all selected items and take their average
    nov_cdf = QuantileTransformer()
    nov_cdf.fit(novelties)
    nov = np.mean(nov_cdf.transform(np.stack(selected_novelties).reshape(-1, 1)))

    result = np.array([rel, div, nov])

    if return_supports:
        supports = {
            "elicitation_movies": np.squeeze(movie_indices),
            "relevances": np.squeeze(relevances),
            "diversities": np.squeeze(diversities),
            "novelties": np.squeeze(novelties),
            "normalized_relevances": np.squeeze(rel_cdf.transform(relevances)),
            "normalized_diversities": np.squeeze(div_cdf.transform(diversities)),
            "normalized_novelties": np.squeeze(nov_cdf.transform(novelties)),
            "raw_rating": np.squeeze(relevances),
            "raw_distance": np.squeeze(raw_diversities.flatten()),
            "raw_users_viewed_item": np.squeeze(users_viewed_item[movie_indices]),
            "rating_matrix_shape": np.array(loader.rating_matrix.shape)
        }
        return result / result.sum(), supports

    return result / result.sum()


@functools.lru_cache(maxsize=None)
def prepare_wrapper_once():
    loader = load_ml_dataset()

    items = np.arange(loader.rating_matrix.shape[1])
    distance_matrix = 1.0 - loader.similarity_matrix
    
    users_viewed_item = loader.rating_matrix.astype(bool).sum(axis=0)

    movie_title_to_idx = dict()
    for movie_id, row in loader.movies_df_indexed.iterrows():
        movie_title_to_idx[bytes(row.title, "UTF-8")] = loader.movie_id_to_index[movie_id]

    return loader, items, distance_matrix, users_viewed_item, movie_title_to_idx#, algo, ratings_df

# Define enrich_results on each loader?
# TODO make this to be custom implementation in multiobjective plugin
def enrich_results(top_k, loader, support=None):
    print(loader)
    if type(loader) is MLDataLoader:
        top_k_ids = [loader.movie_index_to_id[movie_idx] for movie_idx in top_k]
        top_k_description = [loader.movies_df_indexed.loc[movie_id].title for movie_id in top_k_ids]
        top_k_genres = [loader.movies_df_indexed.loc[movie_id].genres.split("|") for movie_id in top_k_ids]
        top_k_genres = [x if x != ["(no genres listed)"] else [] for x in top_k_genres]
        top_k_url = [loader.get_image(movie_idx) for movie_idx in top_k]
        top_k_trailers = [loader.get_trailer_url(movie_idx) for movie_idx in top_k]
        top_k_plots = [loader.get_plot(movie_idx) for movie_idx in top_k]
    else:
        top_k_ids = [loader.get_item_id(movie_idx) for movie_idx in top_k]
        top_k_description = [loader.items_df_indexed.loc[movie_id].title for movie_id in top_k_ids]
        if hasattr(loader.items_df_indexed, "genres"):
            top_k_genres = [loader.items_df_indexed.loc[movie_id].genres.split("|") for movie_id in top_k_ids]
        else:
            top_k_genres = ['' for movie_id in top_k_ids]
        top_k_genres = [x if x != ["(no genres listed)"] else [] for x in top_k_genres]
        top_k_url = [loader.get_item_index_image_url(movie_idx) for movie_idx in top_k]
        top_k_trailers = [""] * len(top_k)
        top_k_plots = [""] * len(top_k)
        
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

def prepare_wrapper(selected_movies, model, mandate_allocation_factory, obj_weights, filter_out_movies = [], k=10, norm_f=cdf):
    loader, items, distance_matrix, users_viewed_item, movie_title_to_idx = prepare_wrapper_once()

    max_user = loader.ratings_df.userId.max()
    new_user = tf.constant(str(max_user + 1))
    
    # We want to reuse the relevance based model instead
    # model, train = prepare_tf_model(loader)
    
    seen_movies_tensor = tf.stack(
        [tf.constant(loader.movies_df.loc[x].title) for x in selected_movies]
        +
        [tf.constant(loader.movies_df.loc[x].title) for x in filter_out_movies]
    )
    scores, x = model.predict_all_unseen(new_user, seen_movies_tensor, n_items=items.size) #model.predict_for_user(new_user, ratings2, k=2000)
    scores, x = tf.squeeze(scores).numpy(), tf.squeeze(x).numpy()
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    top_k = [movie_title_to_idx[t] for t in x]

    
    rating_vector = np.zeros((items.size, ), dtype=np.float32)
    rating_vector[top_k] = scores # Set the predicted scores
    rating_vector[selected_movies] = 1.0 # Set selected movies to 1.0
    rating_vector = np.expand_dims(rating_vector, axis=0) # Convert it to 1xN rating matrix
    extended_rating_matrix = rating_vector #loader.rating_matrix[:1] #rating_vector

    normalization_factory = norm_f
    cache_dir = None #"."

    mandate_allocation = mandate_allocation_factory(obj_weights, -1e6)
    
    unseen_items_mask = np.ones(extended_rating_matrix.shape, dtype=np.bool8)
    if filter_out_movies:
        unseen_items_mask[:, np.array(filter_out_movies)] = 0 # Mask out the items

    discount_sequences = [[1.0] * k, [1.0] * k, [1.0] * k]

    n_users = loader.rating_matrix.shape[0] # Calculate number of users on the full rating matrix not just on the single user vector
    return loader, RLPropWrapper(items, extended_rating_matrix, distance_matrix, users_viewed_item, normalization_factory, mandate_allocation, unseen_items_mask, cache_dir, discount_sequences, n_users)

def rlprop(selected_movies, model, weights, filter_out_movies = [], k=10, norm_f=cdf, include_support=False):
    obj_weights = weights
    obj_weights /= obj_weights.sum()

    loader, wrapper = prepare_wrapper(selected_movies, model, exactly_proportional_fuzzy_dhondt_2, obj_weights, filter_out_movies, k, norm_f)
    wrapper.init()

    if include_support:
        x, support = wrapper(k, return_support=include_support)
        return enrich_results(x[0], loader, {key: value for key,value in support.items()})

    x = wrapper(k, return_support=include_support)
    return enrich_results(x[0], loader)

def get_objective_importance(loader, selected_movie_indices, shown_movies):
    if not selected_movie_indices:
        return None
    importances = calculate_weight_estimate(loader, selected_movie_indices, shown_movies)
    return {
        "relevance": importances[0],
        "diversity": importances[1],
        "novelty": importances[2]
    }

def weighted_average(selected_movies, model, weights, filter_out_movies = [], k=10, norm_f=cdf, include_support=False):
    obj_weights = weights
    obj_weights /= obj_weights.sum()

    loader, wrapper = prepare_wrapper(selected_movies, model, weighted_average_strategy, obj_weights, filter_out_movies, k, norm_f)
    wrapper.init()

    if include_support:
        x, support = wrapper(k, return_support=include_support)
        return enrich_results(x[0], loader, {key: value for key,value in support.items()})

    x = wrapper(k, return_support=include_support)
    return enrich_results(x[0], loader)


def recommend_2_3(selected_movies, filter_out_movies = [], return_model = False, k = 10):
    loader = load_ml_dataset()

    max_user = loader.ratings_df.userId.max()

    ################ TF specific ################
    model, train = prepare_tf_model(loader)

    
    new_user = tf.constant(str(max_user + 1))
    def data_gen():
        for x in selected_movies:
            yield {
                "item_title": tf.constant(loader.movies_df.loc[x].title),
                "user_id": new_user,
            }
    ratings2 = tf.data.Dataset.from_generator(data_gen, output_signature={
        "item_title": tf.TensorSpec(shape=(), dtype=tf.string),
        "user_id": tf.TensorSpec(shape=(), dtype=tf.string)
    })



    # Finetune
    model.fit(ratings2.concatenate(train.take(100)).batch(256), epochs=2)

    seen_movies_tensor = tf.stack(
        [tf.constant(loader.movies_df.loc[x].title) for x in selected_movies]
        +
        [tf.constant(loader.movies_df.loc[x].title) for x in filter_out_movies]
    )
    predictions = tf.squeeze(model.predict_for_user(new_user, seen_movies_tensor, k)).numpy()
    
    top_k = [loader.movie_id_to_index[loader.movies_df[loader.movies_df.title == x.decode("UTF-8")].movieId.values[0]] for x in predictions]

    if return_model:
        return enrich_results(top_k, loader), model

    return enrich_results(top_k, loader)

# Takes rating matrix and returns dense copy
def gen_dense_rating_matrix(rating_matrix):
    # Number of times each item was rated
    ratings_per_item = np.sum(rating_matrix > 0.0, axis=0)
    most_rated_items = np.argsort(-ratings_per_item, kind="stable")
    # Take only MOST_RATED_MOVIES_THRESHOLD movies
    most_rated_items_subset = most_rated_items[:MOST_RATED_MOVIES_THRESHOLD]
    # Restrict rating matrix to the subset of items
    dense_rating_matrix = rating_matrix[:, most_rated_items_subset]
    # Restrict rating matrix to the subset of users
    n = np.minimum(MOST_RATED_MOVIES_THRESHOLD, most_rated_items_subset.size)
    per_user_rating_ratios = np.sum(dense_rating_matrix > 0, axis=1) / n
    selected_users = np.where(per_user_rating_ratios >= USERS_RATING_RATIO_THRESHOLD)[0]
    dense_rating_matrix = dense_rating_matrix[selected_users, :]
    assert dense_rating_matrix.ndim == rating_matrix.ndim, f"Dense rating matrix should preserve ndim: {dense_rating_matrix.shape}"
    assert np.all(dense_rating_matrix.shape <= rating_matrix.shape), f"Making dense rating matrix should not increase dimensions: {dense_rating_matrix.shape} vs. {rating_matrix.shape}"
    return dense_rating_matrix, most_rated_items_subset


def gen_groups(rating_matrix, n_groups):
    similarities = cosine_similarity(rating_matrix.T)
    similarities = (similarities + 1) / 2
    clustering = SpectralClustering(n_groups, random_state=0, affinity="precomputed")
    groups = clustering.fit_predict(similarities)
    return groups

def tags_in_cluster(group_items, tags_per_item):
    tags = set()
    for item in group_items:
        tags.update(tags_per_item[item])
    return tags

# Relevance of tag to cluster/group of movies
def tag_relevance(tag, group_items, tag_counts_per_movie):
    rel = 0.0
    for item in group_items:
        rel += tag_counts_per_movie[item][tag]
    return rel

def most_relevant_movies(group, movie_to_group, deny_list, tag, tag_counts_per_movie, loader):
    movie_counts = dict()
    for movie, tag_counts in tag_counts_per_movie.items():
        if movie in deny_list or movie not in movie_to_group or movie_to_group[movie] != group:
            pass #movie_counts[movie] = -1
        else:
            movie_counts[movie] = tag_counts[tag]
    return sorted(movie_counts.keys(), key=lambda x: movie_counts[x], reverse=True)

def acc_per_cluster_tag_relevance(tag, group_to_items, tag_counts_per_movie):
    acc = 0.0
    for group, items in group_to_items.items():
        acc += tag_relevance(tag, items, tag_counts_per_movie)
    return acc

def acc_per_tag_tag_relevance(group_items, tag_counts_per_movie):
    acc = 0.0
    # for tag in group_tags:
    group_tags = set()
    for movie in group_items:
        for tag, tag_count in tag_counts_per_movie[movie].items():
            if tag_count > 0:
                group_tags.add(tag)

    for tag in group_tags:
        acc += tag_relevance(tag, group_items, tag_counts_per_movie)
    return acc

# Prepares description for each of the groups
def label_groups(group_assignment, tags, tag_counts_per_movie):
    group_to_items = dict()
    tags_per_group = dict()
    
    for item, group in group_assignment.items():
        if group not in group_to_items:
            tags_per_group[group] = set()
            group_to_items[group] = []
        group_to_items[group].append(item)

        for tag in tags:
            if tag in tag_counts_per_movie[item] and tag_counts_per_movie[item][tag] > 0:
                tags_per_group[group].add(tag)

    best_group_tags = dict()
    tag_deny_list = set()
    for group in set(group_assignment.values()):
        tag_prod = dict()
        for tag in tags_per_group[group]: #tags:
            if tag in tag_deny_list:
                pass
            else:
                d1 = acc_per_cluster_tag_relevance(tag, group_to_items, tag_counts_per_movie)
                if d1 == 0:
                    uniqueness = 0.0
                else:
                    uniqueness = tag_relevance(tag, group_to_items[group], tag_counts_per_movie) / d1
                d2 = acc_per_tag_tag_relevance(group_to_items[group], tag_counts_per_movie)
                if d2 == 0:
                    relevance = 0.0
                else:
                    relevance = tag_relevance(tag, group_to_items[group], tag_counts_per_movie) / d2
                tag_prod[tag] = uniqueness * relevance
        best_tags = sorted(tag_prod.keys(), key=lambda x: tag_prod[x], reverse=True)
        best_group_tags[group] = best_tags[:NUM_TAGS_PER_GROUP]
    return best_group_tags

def search_for_movie(attrib, pattern, tr=None):
    # Map to 
    if attrib == "movie":
        attrib = "title"

    loader = load_ml_dataset()

    # If we have a translate function
    if tr:
        found_movies = loader.movies_df[loader.movies_df.movieId.astype(str).map(tr).str.contains(pattern, case=False)]
    else:
        found_movies = loader.movies_df[loader.movies_df.title.str.contains(pattern, case=False)]
    
    movie_indices = [loader.movie_id_to_index[movie_id] for movie_id in found_movies.movieId.values]
    res_url = [loader.get_image(movie_idx) for movie_idx in movie_indices]
    result = [{"movie": movie, "url": url, "movie_idx": str(movie_idx)} for movie, url, movie_idx in zip(found_movies.title.values, res_url, movie_indices)]
    return result

