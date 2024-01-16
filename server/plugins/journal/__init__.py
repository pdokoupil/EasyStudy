import functools
import json
import pickle
import random
import sys
import os
import time

import flask
from sklearn.preprocessing import QuantileTransformer

from plugins.fastcompare.algo.wrappers.data_loadering import MLGenomeDataLoader, GoodBooksFilteredDataLoader



[sys.path.append(i) for i in ['.', '..']]
[sys.path.append(i) for i in ['../.', '../..', '../../.']]


import numpy as np

from common import get_tr, load_languages, multi_lang, load_user_study_config
from flask import Blueprint, jsonify, make_response, request, redirect, render_template, url_for, session

from plugins.utils.preference_elicitation import enrich_results
from plugins.utils.interaction_logging import log_interaction
from plugins.fastcompare.loading import load_data_loaders
from plugins.fastcompare import elicitation_ended, filter_params, load_data_loader_cached, search_for_item

from plugins.fastcompare import get_semi_local_cache_name, get_cache_path
from plugins.journal.metrics import binomial_diversity, intra_list_diversity, item_popularity, popularity_based_novelty, exploration, exploitation

#from memory_profiler import profile

from plugins.journal.algorithms import NEG_INF, evolutionary_exact, evolutionary_max, greedy_exact, greedy_max, item_wise_exact, item_wise_max, unit_normalized_diversification

from app import rds

# Load available language mutations
languages = load_languages(os.path.dirname(__file__))

##### Global variables block #####
N_BLOCKS = 3
N_ITERATIONS = 6 # 6 iterations per block

# Algorithm matrix
# Algorithms are divided in two dimensions: [MAX, EXACT] x [ITEM-WISE, GREEDY, EVOLUTIONARY]
# We always select one row per user, so each user gets either ITEM-WISE, GREEDY or EVOLUTIONARY (in both MAX and EXACT variants) + Relevance baseline
ALGORITHMS = [
    "ITEM-WISE-MAX-1W", "ITEM-WISE-MAX-2W", "ITEM-WISE-EXACT-1W",
    "GREEDY-MAX-1W", "GREEDY-MAX-2W", "GREEDY-EXACT-1W",
    "EVOLUTIONARY-MAX-1W", "EVOLUTIONARY-MAX-2W", "EVOLUTIONARY-EXACT-1W",
    "RELEVANCE-BASED"
]

# What kind of optimization should the algorithm apply
# "MAX" means that we maximize the objectives while ensuring proportionality w.r.t. given weights
# "EXACT" means we do not care about maximization and only care about proportionality w.r.t. given weights
OPTIMIZATION_TYPE = ["MAX", "EXACT"]

# Different algorithm families
ALGORITHM_FAMILY = [
    "ITEM-WISE",
    "GREEDY",
    "EVOLUTIONARY"
]

# "Backend slider" used for "MAX" variants of the algorithms
# Note that for EXACT, these two are essentially same, se we ignore this (and always use 1W)
BACKEND_SLIDER = [
    "1W", # 1-way slider -> we use 4 objectives for the algorithm
    "2W"  # 2-way slider -> we use all 7 objectives for the algorithm
]

# The mapping is not fixed between users
ALGORITHM_ANON_NAMES = ["ALPHA", "BETA", "GAMMA"]

# Possible alpha values to use during metric assessment
# Since we have two steps, comparing 3 alphas in each, we sample 6 alphas randomly

POSSIBLE_ALPHAS = [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
# one-way means there is single objective per slider, going from 0 to 1
# two-way means there are two objectives, one at each side of the slider
SLIDER_VERSIONS = ["TWO-WAY", "ONE-WAY"]

# Number of alpha iterations
N_ALPHA_ITERS = 2

METRIC_ASSESSMENT_ALPHA = 0.1

N_ITEMS_SUBSET=500

# Given algorithm family, optimization type and backend slider, generate unique algorithm name
def algo_name(family, optim_type, backend_slider):
    name = f"{family}-{optim_type}-{backend_slider}"
    assert name in ALGORITHMS, f"name={name} not in {ALGORITHMS}"
    return name

# Implementation of this function can differ among plugins
def get_lang():
    default_lang = "en"
    if "lang" in session and session['lang'] and session['lang'] in languages:
        return session['lang']
    return default_lang

# Global, plugin related stuff
__plugin_name__ = "journal"
__version__ = "0.1.0"
__author__ = "Patrik Dokoupil"
__author_contact__ = "Patrik.Dokoupil@matfyz.cuni.cz"
__description__ = "Complex plugin for a very customized user study that we have used for journal paper"

bp = Blueprint(__plugin_name__, __plugin_name__, url_prefix=f"/{__plugin_name__}")

##### Redis helpers #######
# TODO decide if should be moved somewhere else
# Return user key for a given user (by session)
# We will key nearly all the data based on unique user session ID
# to ensure values stored to different users are separated
def get_uname():
    return f"user:{session['uuid']}"

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

# Helper that checks if the user study is configured with Books dataset
def is_books(conf):
    return "Goodbooks" in conf["selected_data_loader"]

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



# Some common abstraction about "morsification" (like diversification, but with multiple objectives)
# Runs diversification on a relevance based recommendation
# w.r.t. algorithm passed as algo
def morsify(k, rel_scores,
            algo, items, objective_fs,
            rating_row, filter_out_items, n_items_subset=None,
            do_normalize=False, rnd_mixture=False):
   
    assert rel_scores.ndim == 1
    assert rating_row.ndim == 1
    
    start_time = time.perf_counter()
    top_k_list = np.zeros(shape=(k, ), dtype=np.int32)
    
    # Hold marginal gain for each item, objective pair
    mgains = np.zeros(shape=(len(objective_fs), items.size if n_items_subset is None else n_items_subset), dtype=np.float32)
    # marginal gains for relevance are calculated separate, this is optimization and brings some savings when compared to naive calculation

    # Sort relevances
    # Filter_out_items are already propageted into rel_scores (have lowest score)
    sorted_relevances = np.argsort(-rel_scores, axis=-1)
   

    #print(f"User_idx = {user_idx}, user={random_user}")
    
    # If n_items_subset is specified, we take subset of items
    if n_items_subset is None:
        # Mgain masking will ensure we do not select items in filter_out_items set
        source_items = items
    else:
        if rnd_mixture:
            assert n_items_subset % 2 == 0, f"When using random mixture we expect n_items_subset ({n_items_subset}) to be divisible by 2"
            # Here we need to ensure that we do not include already seen items among source_items
            # so we have to filter out 'filter_out_items' out of the set

            # We know items from filter_out_items have very low relevances
            # so here we are safe w.r.t. filter_out_movies because those will be at the end of the sorted list
            relevance_half = sorted_relevances[:n_items_subset//2]
            # However, for the random half, we have to ensure we do not sample movies from filter_out_movies because this can lead to issues
            # especially when n_items_subset is small and filter_out_items is large (worst case is that we will sample exactly those items that should have been filtered out)
            random_candidates = np.setdiff1d(sorted_relevances[n_items_subset//2:], filter_out_items)
            random_half = np.random.choice(random_candidates, n_items_subset//2, replace=False)
            source_items = np.concatenate([
                relevance_half, 
                random_half
            ])
        else:
            source_items = sorted_relevances[:n_items_subset]

    # Default number of quantiles is 1000, however, if n_samples is smaller than n_quantiles, then n_samples is used and warning is raised
    # to get rid of the warning, we calculates quantiles straight away
    n_quantiles = min(1000, mgains.shape[1])

    # Mask-out seen items by multiplying with zero
    # i.e. 1 is unseen
    # 0 is seen
    # Lets first set zeros everywhere
    seen_items_mask = np.zeros(shape=(source_items.size, ), dtype=np.int8)
    # And only put 1 to UNSEEN items in CANDIDATE (source_items) list
    seen_items_mask[rating_row[source_items] <= 0.0] = 1
    # print(f"### Unseen: {seen_items_mask.sum()} out of: {seen_items_mask.size}")
    
    # Build the recommendation incrementally
    for i in range(k):
        for objective_index, objective_f in enumerate(objective_fs):
            # Cache f_prev
            f_prev = objective_f(top_k_list[:i])
            
            objective_cdf_train_data = []
            # For every source item, try to add it and calculate its marginal gain
            for j, item in enumerate(source_items):
                top_k_list[i] = item # try extending the list
                objective_cdf_train_data.append(objective_f(top_k_list[:i+1]) - f_prev)
                mgains[objective_index, j] = objective_cdf_train_data[-1]
                
            # If we should normalize, use cdf_div to normalize marginal gains
            if do_normalize:
                # Reshape to N examples with single feature
                mgains[objective_index] = QuantileTransformer(n_quantiles=n_quantiles).fit_transform(mgains[objective_index].reshape(-1, 1)).reshape(mgains[objective_index].shape)
    
        # Calculate scores
        #print(f"@@ Mgains shape: {mgains.shape}, seen_items_mask shape: {seen_items_mask}")
        best_item_idx = algo(mgains, seen_items_mask)
        best_item = source_items[best_item_idx]
            
        # Select the best item and append it to the recommendation list            
        top_k_list[i] = best_item
        # Mask out the item so that we do not recommend it again
        seen_items_mask[best_item_idx] = 0

        # if i == 9:
        #     print(f"Rel scores shape: {rel_scores.shape}")
        #     print(f"### best item = {best_item}, mgains: {mgains[:, best_item_idx]}, rel: {rel_scores[best_item]}")

    # print(f"Diversification took: {time.perf_counter() - start_time}")
    return top_k_list

# A simple wrapper class for a recommendation objective (adds name to it)
class ObjectiveWrapper:
    def __init__(self, f, name):
        self.f = f
        self.obj_name = name
        
    def __call__(self, rec_list, *args, **kwargs):
        return self.f(rec_list, *args, **kwargs)
    
    def name(self):
        return self.obj_name

##### End of algorithms #####


#################################################################################
###########################   ENDPOINTS   #######################################
#################################################################################


# Render journal plugin study creation page
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
    params["disable_relative_comparison"] = "Disable relative comparison"
    params["disable_demographics"] = "Disable demographics"
    params["separate_training_data"] = "Separate training data"

    return render_template("journal_create.html", **params)

# Public facing endpoint
@bp.route("/join", methods=["GET"])
def join():
    assert "guid" in request.args, "guid must be available in arguments"
    return redirect(url_for("utils.join", continuation_url=url_for("journal.on_joined"), **request.args))

# Callback once user has joined we forward to pre-study questionnaire
@bp.route("/on-joined", methods=["GET", "POST"])
def on_joined():
    return redirect(url_for("journal.pre_study_questionnaire"))

# Endpoint for pre-study questionnaire
@bp.route("/pre-study-questionnaire", methods=["GET", "POST"])
def pre_study_questionnaire():
    params = {
        "continuation_url": url_for("journal.pre_study_questionnaire_done"),
        "header": "Pre-study questionnaire",
        "hint": "Please answer the questions below before starting the user study.",
        "finish": "Proceed to user study",
        "title": "Pre-study questionnaire"
    }
    return render_template("journal_pre_study_questionnaire.html", **params)

# Endpoint that should be called once pre-study-questionnaire is done
@bp.route("/pre-study-questionnaire-done", methods=["GET", "POST"])
def pre_study_questionnaire_done():

    data = {}
    data.update(**request.form)

    # We just log the question answers as there is no other useful data gathered during pre-study-questionnaire
    log_interaction(session["participation_id"], "pre-study-questionnaire", **data)

    return redirect(url_for("utils.preference_elicitation", continuation_url=url_for("journal.send_feedback"),
            consuming_plugin=__plugin_name__,
            initial_data_url=url_for('fastcompare.get_initial_data'),
            search_item_url=url_for('journal.item_search')))

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


    # Proceed to weights estimation, use CF-ILD, popularity novelty, MER
    # and exploration
    configured_metric_name = conf["mors_diversity_metric"]
    if configured_metric_name == "CF-ILD":
        diversity_f = intra_list_diversity(loader.distance_matrix)
        distance_matrix = loader.distance_matrix
    elif configured_metric_name == "CB-ILD":
        distance_matrix_cb = get_distance_matrix_cb(os.path.join(get_cache_path(get_semi_local_cache_name(loader)), "distance_matrix_text.npy"))
        diversity_f = intra_list_diversity(distance_matrix_cb)
        distance_matrix = distance_matrix_cb
    else:
        assert False, f"Unknown configured diversity metric: {configured_metric_name}"
    novelty_f = popularity_based_novelty(loader.rating_matrix)
    #relevances = loader.rating_matrix.mean(axis=0)

    items = np.arange(loader.rating_matrix.shape[1])
    algo = EASER_pretrained(items)
    algo = algo.load(os.path.join(get_cache_path(get_semi_local_cache_name(loader)), "item_item.npy"))

    def relevance_f(top_k_list, relevances):
        return relevances[top_k_list].sum()

    def relevance_w(top_k_list, relevances, **kwargs):
        return relevance_f(top_k_list, relevances)
    
    # Wrapped diversity
    def diversity_w(top_k_list, *args, **kwargs):
        return diversity_f(top_k_list)
    
    def novelty_w(top_k_list, *args, **kwargs):
        return novelty_f(top_k_list)

    def exploration_w(top_k_list, user_vector_list, *args, **kwargs):
        f = exploration(np.array([]), distance_matrix)
        f.user_vector = np.array(user_vector_list, dtype=np.int32) # fixup
        return f(top_k_list)

    objectives = [
        ObjectiveWrapper(relevance_w, "relevance"),
        ObjectiveWrapper(diversity_w, "diversity"),
        ObjectiveWrapper(novelty_w, "novelty"),
        ObjectiveWrapper(exploration_w, "exploration")
    ]

    # Calculate weights based on selection and shown movies during preference elicitation
    start_time = time.perf_counter()
    weights, supports = calculate_weight_estimate_generic(algo, objectives, selected_movies, elicitation_shown_items, return_supports=True)
    print(f"Weights initialized to {weights}, took: {time.perf_counter() - start_time}")

    set_mapping(get_uname(), {
        'initial_weights': weights,
        'iteration': 0, # Start with zero, because at the very beginning, mors_feedback is called, not mors and that generates recommendations for first iteration, but at the same time, increases the iteration
        'elicitation_selected_movies': selected_movies,
        'selected_movie_indices': [],
        'elicitation_shown_movies': elicitation_shown_items
    })

    ### Initialize stuff related to alpha comparison (after metric assessment step) ###
    # Permutation over alphas
    possible_alphas = POSSIBLE_ALPHAS[:]
    np.random.shuffle(possible_alphas)
    selected_alphas = possible_alphas[:6]

    ### Initialize MORS related stuff ###
    # "Algorithm family" is between-user variable
    selected_algorithm_family = np.random.choice(ALGORITHM_FAMILY)
    # We always select RELEVANCE-BASED, then EXACT (thus EXACT-1W as we only consider 1W for EXACT) from the given family
    # and then randomly either MAX-1W or MAX-2W
    selected_algorithms = [
        algo_name(selected_algorithm_family, "EXACT", "1W"),
        np.random.choice([
            algo_name(selected_algorithm_family, "MAX", "1W"),
            algo_name(selected_algorithm_family, "MAX", "2W")
        ]),
        "RELEVANCE-BASED"
    ]
    # Select sliders
    # "Front-end slider" is between-user variable
    selected_slider_versions = [np.random.choice(SLIDER_VERSIONS)] * N_BLOCKS
    # Shuffle algorithms
    np.random.shuffle(selected_algorithms)
    # Shuffle algorithm names
    algorithm_names = ALGORITHM_ANON_NAMES[:]
    np.random.shuffle(algorithm_names)

    #########################################################################
    # Here we need to precompute data to show in the metric-assessment #
    # since this may take a long time, we log elicitation ended right here #
    weights_with_list = {}
    weights_with_list["values"] = {key: val.astype(float) for key, val in weights["values"].items()}
    weights_with_list["vec"] = weights["vec"].tolist()
    elicitation_ended(
        session['elicitation_movies'],
        selected_movies,
        supports={key: np.round(value.astype(float), 4).tolist() for key, value in supports.items()},
        alphas_p=[selected_alphas[:3], selected_alphas[3:]],
        algorithm_family=selected_algorithm_family,
        selected_algorithms=selected_algorithms,
        selected_slider_versions=selected_slider_versions,
        initial_weights=weights_with_list,
        ease_selections=selected_movies,
        ease_filter_out=selected_movies,
        elicitation_shown_items=elicitation_shown_items.tolist()
    )

    assessment_k = 8 # We use k=8 instead of k=10 so that items fit to screen easily
    start_time = time.perf_counter()
    
    # Movie indices of selected movies
    elicitation_selected = np.array(selected_movies)
    rel_scores, user_vector, ease_pred = algo.predict_with_score(elicitation_selected, elicitation_selected, assessment_k)

    distance_matrix_cb = get_distance_matrix_cb(os.path.join(get_cache_path(get_semi_local_cache_name(loader)), "distance_matrix_text.npy"))

    cb_ild = intra_list_diversity(distance_matrix_cb)
    cf_ild = intra_list_diversity(loader.distance_matrix)
    bin_div = binomial_diversity(loader.all_categories, loader.get_item_index_categories, loader.rating_matrix, 0.0, loader.name())

    alpha = METRIC_ASSESSMENT_ALPHA
    # relevance wrapper for morsify function below
    relevance_wrapper = ObjectiveWrapper(functools.partial(relevance_f, relevances=rel_scores), "relevance")
    # basic diversification algorithm (1 - alpha) * relevance + alpha * diversity
    diversification_algo = unit_normalized_diversification(alpha)

    # ease_baseline = enrich_results(ease_pred, loader)
    print(f"Predicting r1 took: {time.perf_counter() - start_time}")

    r2 = morsify(assessment_k, rel_scores, diversification_algo,
                 items, [relevance_wrapper, ObjectiveWrapper(cf_ild, "cf-ild")],
                 rating_row=user_vector, filter_out_items=elicitation_selected,
                 n_items_subset=N_ITEMS_SUBSET, do_normalize=True, rnd_mixture=True)
    r2_indices = r2
    r2 = enrich_results(r2_indices, loader)
    print(f"Predicting r2 took: {time.perf_counter() - start_time}")
    
    r3 = morsify(assessment_k, rel_scores, diversification_algo,
                 items, [relevance_wrapper, ObjectiveWrapper(cb_ild, "cb-ild")],
                 rating_row=user_vector, filter_out_items=elicitation_selected,
                 n_items_subset=N_ITEMS_SUBSET, do_normalize=True, rnd_mixture=True)
    r3_indices = r3
    r3 = enrich_results(r3_indices, loader)
    print(f"Predicting r3 took: {time.perf_counter() - start_time}")

    r4 = morsify(assessment_k, rel_scores, diversification_algo,
                 items, [relevance_wrapper, ObjectiveWrapper(bin_div, "bin-div")],
                 rating_row=user_vector, filter_out_items=elicitation_selected,
                 n_items_subset=N_ITEMS_SUBSET, do_normalize=True, rnd_mixture=True)
    r4_indices = r4
    r4 = enrich_results(r4_indices, loader)
    print(f"Predicting r4 took: {time.perf_counter() - start_time}")

    # Mapping is implicit, position 0 means "LIST A", position 2 is "LIST C"
    # We just shuffle the algorithms so that what is displayed below "LIST A" is random
    lists = ["CB-ILD", "CF-ILD", "BIN-DIV"]
    list_name_to_rec = {
        "CF-ILD": r2,
        "CB-ILD": r3,
        "BIN-DIV": r4
    }
    np.random.shuffle(lists)
    algorithm_name_mapping = {
        lists[0]: "LIST A",
        lists[1]: "LIST B",
        lists[2]: "LIST C"
    }

    params = {
        "movies": {
            # "EASE": {
            #     "movies": ease_baseline,
            #     "order": 3
            # },
            algorithm_name_mapping[lists[0]]: {
                "movies": list_name_to_rec[lists[0]],
                "order": 0
            },
            algorithm_name_mapping[lists[1]]: {
                "movies": list_name_to_rec[lists[1]],
                "order": 1
            },
            algorithm_name_mapping[lists[2]]: {
                "movies": list_name_to_rec[lists[2]],
                "order": 2
            }
        }
    }

    # We need to store inverse mapping
    # in the form of "list name" : "diversity name"
    set_val("metric_assessment_list_to_diversity", { list_name : diversity_name for diversity_name, list_name in algorithm_name_mapping.items() })
    #########################################################################

    set_mapping(get_uname(), {
        "alphas_iteration": 1,
        "alphas_p": [selected_alphas[:3], selected_alphas[3:]],
        "algorithm_family": selected_algorithm_family,
        "selected_algorithms": selected_algorithms,
        "selected_slider_versions": selected_slider_versions,
        "recommendations": {
           algo: [] for algo in selected_algorithms # For each algorithm and each iteration we hold the recommendation
        },
        "selected_items": {
           algo: [] for algo in selected_algorithms # For each algorithm and each iteration we hold the selected items
        },
        "shown_items": {
            algo: [] for algo in selected_algorithms # For each algorithm and each iteration we hold the IDs of recommended items (for quick filtering)
        },
        "slider_values": {
            "slider_relevance": [],
            "slider_exploitation_exploration": [],
            "slider_uniformity_diversity": [],
            "slider_popularity_novelty": []
        },
        "assessment_recommendations": params
    })

    objectives = {
        "CF-ILD": {
            "indices": r2_indices.tolist(),
            "cf_ild": cf_ild(r2_indices),
            "cb_ild": cb_ild(r2_indices),
            "bin_div": bin_div(r2_indices),
            "relevance": relevance_f(r2_indices, rel_scores).item()
        },
        "CB-ILD": {
            "indices": r3_indices.tolist(),
            "cf_ild": cf_ild(r3_indices),
            "cb_ild": cb_ild(r3_indices),
            "bin_div": bin_div(r3_indices),
            "relevance": relevance_f(r3_indices, rel_scores).item()
        },
        "BIN-DIV": {
            "indices": r4_indices.tolist(),
            "cf_ild": cf_ild(r4_indices),
            "cb_ild": cb_ild(r4_indices),
            "bin_div": bin_div(r4_indices),
            "relevance": relevance_f(r4_indices, rel_scores).item()
        }
    }
    
    data = {
        "list_permutation": lists,
        "algorithm_name_mapping": algorithm_name_mapping,
        "objectives": objectives,
        "list_name_to_rec": list_name_to_rec
    }
    log_interaction(session["participation_id"], "metric-assessment-started", **data)

    #return redirect(url_for("multiobjective.compare_and_refine"))
    return redirect(url_for("journal.metric_assessment"))

# Rendering endpoint for metric-assessment
@bp.route("/metric-assessment", methods=["GET"])
def metric_assessment():
    conf =  load_user_study_config(session["user_study_id"])
    params = get_val("assessment_recommendations")    

    tr = get_tr(languages, get_lang())
    params["contacts"] = tr("footer_contacts")
    params["contact"] = tr("footer_contact")
    params["charles_university"] = tr("footer_charles_university")
    params["cagliari_university"] = tr("footer_cagliari_university")
    params["t1"] = tr("footer_t1")
    params["t2"] = tr("footer_t2")
    params["title"] = tr("journal_metric_assessment_title")
    params["header"] = tr("journal_metric_assessment_header")
    params["hint"] = tr("journal_metric_assessment_hint")
    params["continuation_url"] = request.args.get("continuation_url")
    params["finish"] = tr("metric_assessment_finish")
    params["iteration"] = 1
    params["n_iterations"] = 1 + N_ALPHA_ITERS # We have 1 iteration for actual metric assessment followed bz N_ALPHA_ITERS iterations for comparing alphas
    
    # Handle overrides
    params["footer_override"] = None
    if "text_overrides" in conf:
        if "footer" in conf["text_overrides"]:
            params["footer_override"] = conf["text_overrides"]["footer"]
    
    return render_template("metric_assessment.html", **params)

# Called as continuation of compare-alphas / metric-assessment, redirects for compare-alphas (next step)
# This is where we generate the results, compare-alphas then just shows them
@bp.route("/metric-feedback", methods=["POST"])
def metric_feedback():
    user_data = get_all(get_uname())

    # We are before first iteration
    cur_iter = user_data['alphas_iteration']

    # Take alpha values to be shown in the current step
    current_alphas = user_data['alphas_p'][cur_iter - 1]

    print(f"METRIC FEEDBACK step 1 = {cur_iter}")
    if cur_iter == 1:
        # We need to map back from "hidden name", e.g. "List A" to actual name of the diversity metric
        selected_metric_name = request.form.get("selected_metric_name")
        mapping = get_val("metric_assessment_list_to_diversity")
        selected_metric_name = mapping[selected_metric_name]

        #selected_metric_index = request.form.get("selected_metric_index")
        set_val('selected_metric_name', selected_metric_name)
        # Mark end of metric assessment here
        log_interaction(session["participation_id"], "metric-assessment-ended", selected_metric_name=selected_metric_name)
    else:
        selected_metric_name = user_data['selected_metric_name']
        # Here we first mark end of previous iteration, then start of current iteration
        drag_and_drop_positions = json.loads(request.form.get("drag_and_drop_positions"))
        dropzone_position = json.loads(request.form.get("dropzone_position"))
        log_interaction(session["participation_id"],
                        "compare-alphas-ended",
                        iteration=cur_iter - 1,
                        drag_and_drop_positions=drag_and_drop_positions,
                        dropzone_position=dropzone_position)
    
    if cur_iter >= N_ALPHA_ITERS:
        continuation_url = url_for("journal.mors_feedback")
    else:
        continuation_url = url_for("journal.metric_feedback")


    #@profile
    #def compare_alphas_x():
    conf = load_user_study_config(session['user_study_id'])
    params = {}

    # Get a loader
    loader_factory = load_data_loaders()[conf["selected_data_loader"]]
    loader = loader_factory(**filter_params(conf["data_loader_parameters"], loader_factory))
    loader = load_data_loader_cached(loader, session['user_study_guid'], loader_factory.name(), get_semi_local_cache_name(loader))

    # Mapping is implicit, position 0 means "current_alphas[0]", position 2 is "current_alphas[2]"
    # as the alphas are already shuffled
    # We just shuffle the algorithms so that what is displayed below "LIST A" is random
    algorithm_name_mapping = {
        current_alphas[0]: f"LIST {(cur_iter - 1) * 3 + 1}", # Either 1, 4, ..
        current_alphas[1]: f"LIST {(cur_iter - 1) * 3 + 2}", # Either 2, 5, ..
        current_alphas[2]: f"LIST {(cur_iter - 1) * 3 + 3}" # # Either 3, 6, ..
    }

    # We prepare relevance for logging
    def relevance_f(top_k_list, relevances):
        return relevances[top_k_list].sum()

    # We construct all objectives as we want to use them for logging
    cf_ild = intra_list_diversity(loader.distance_matrix)
    distance_matrix_cb = get_distance_matrix_cb(os.path.join(get_cache_path(get_semi_local_cache_name(loader)), "distance_matrix_text.npy"))
    cb_ild = intra_list_diversity(distance_matrix_cb)
    bin_div = binomial_diversity(loader.all_categories, loader.get_item_index_categories, loader.rating_matrix, 0.0, loader.name())

    # Eventually we select just a single objective for diversification
    if selected_metric_name == "CF-ILD":
        div_f = cf_ild
    elif selected_metric_name == "CB-ILD":
        div_f = cb_ild
    elif selected_metric_name == "BIN-DIV":
        div_f = bin_div
    else:
        assert False

    params["movies"] = {}

    k = 8 # We use k=8 instead of k=10 so that items fit to screen easily
    items = np.arange(loader.rating_matrix.shape[1])
    algo = EASER_pretrained(items)
    algo = algo.load(os.path.join(get_cache_path(get_semi_local_cache_name(loader)), "item_item.npy"))
    elicitation_selected = np.array(user_data['elicitation_selected_movies'])
    rel_scores, user_vector, _ = algo.predict_with_score(elicitation_selected, elicitation_selected, k)
    # relevance wrapper for morsify function below
    objective_fs = [
        ObjectiveWrapper(functools.partial(relevance_f, relevances=rel_scores), "relevance"),
        ObjectiveWrapper(div_f, selected_metric_name)
    ]
    rec_lists = dict()
    objectives = dict()
    for alpha_order, alpha in enumerate(current_alphas):

        diversification_algo = unit_normalized_diversification(alpha)
        rec_list = morsify(k, rel_scores, diversification_algo, items,
                           objective_fs, user_vector, filter_out_items=elicitation_selected,
                           n_items_subset=N_ITEMS_SUBSET, do_normalize=True, rnd_mixture=True)

        rec_lists[algorithm_name_mapping[alpha]] = rec_list
        rec_list = enrich_results(rec_list, loader)

        params["movies"][algorithm_name_mapping[alpha]] = {
            "movies": rec_list,
            "order": str(alpha_order)
        }

        objectives[algorithm_name_mapping[alpha]] = {
            "indices": rec_lists[algorithm_name_mapping[alpha]].tolist(),
            "cf_ild": cf_ild(rec_lists[algorithm_name_mapping[alpha]]),
            "cb_ild": cb_ild(rec_lists[algorithm_name_mapping[alpha]]),
            "bin_div": bin_div(rec_lists[algorithm_name_mapping[alpha]]),
            "relevance": relevance_f(rec_lists[algorithm_name_mapping[alpha]], rel_scores).item()
        }

    #session['alpha_movies'] = params
    set_mapping(get_uname() + ":alpha_movies", {
        "movies": params["movies"]
    })

    # Mark start of the first iteration for compare-alphas
    log_interaction(session["participation_id"], "compare-alphas-started",
                    alphas=current_alphas,
                    iteration=cur_iter,
                    objectives=objectives,
                    algorithm_name_mapping=algorithm_name_mapping,
                    selected_metric_name=selected_metric_name,
                    elicitation_selected=user_data['elicitation_selected_movies'],
                    ease_selections=user_data['elicitation_selected_movies'],
                    ease_filter_out=user_data['elicitation_selected_movies'])

    cur_iter += 1
    #session['alphas_iteration'] = cur_iter
    set_val('alphas_iteration', cur_iter)

    return redirect(url_for("journal.compare_alphas", continuation_url=continuation_url))

# Rendering endpoint for compare-alphas
@bp.route("/compare-alphas", methods=["GET", "POST"])
def compare_alphas():
    continuation_url = request.args.get("continuation_url")
    tr = get_tr(languages, get_lang())
    #params = session['alpha_movies']
    u_key = f"user:{session['uuid']}"
    params = get_all(u_key + ":alpha_movies")
    params["continuation_url"] = continuation_url
    params["hint"] = tr("journal_compare_alphas_hint")
    params["header"] = tr("journal_compare_alphas_header")
    params["title"] = tr("journal_compare_alphas_title")
    params["drag"] = tr("journal_compare_alphas_drag")
    params["n_iterations"] = 1 + N_ALPHA_ITERS # We have 1 iteration for actual metric assessment followed bz N_ALPHA_ITERS iterations for comparing alphas
    params["iteration"] = get_val("alphas_iteration")
    # -1 to make it zero based, another -1 because we count 2 N_ALPHA_ITERS and one for metric-assessment, together we have -2
    params["algorithm_offset"] = (get_val("alphas_iteration") - 2) * 3
    return render_template("compare_alphas.html", **params)


# Called after every MORS step, taking feedback from user selections and fine-tuning during MORS
@bp.route("/mors-feedback", methods=["GET", "POST"])
def mors_feedback():
    user_data = get_all(get_uname())
    it = user_data["iteration"]

    if it == 0:
        # At the very beginning, we mark end of compare-alphas step
        drag_and_drop_positions = json.loads(request.form.get("drag_and_drop_positions"))
        dropzone_position = json.loads(request.form.get("dropzone_position"))
        log_interaction(session["participation_id"],
                        "compare-alphas-ended",
                        iteration=user_data['alphas_iteration'],
                        drag_and_drop_positions=drag_and_drop_positions,
                        dropzone_position=dropzone_position)
    else:
        log_interaction(session["participation_id"], "mors-recommendation-ended", iteration=it - 1)

    cur_block = it // N_ITERATIONS
    cur_algorithm = user_data["selected_algorithms"][cur_block]
    cur_slider_version = user_data["selected_slider_versions"][cur_block]

    # Get selected items
    selected_items = request.form.get("selected_items")
    selected_items = selected_items.split(",") if selected_items else []
    selected_items = [int(m) for m in selected_items]
    selected_items_history = get_val('selected_items')
    selected_items_history[cur_algorithm].append(selected_items)
    set_val('selected_items', selected_items_history)
    
    # Store those items in redis
    
    conf = load_user_study_config(session['user_study_id'])

    # Get a loader
    loader_factory = load_data_loaders()[conf["selected_data_loader"]]
    loader = loader_factory(**filter_params(conf["data_loader_parameters"], loader_factory))
    loader = load_data_loader_cached(loader, session['user_study_guid'], loader_factory.name(), get_semi_local_cache_name(loader))


    #print(f"MORS-FEEDBACK IT before = {session['iteration']}")
    if it % N_ITERATIONS == 0:
        # Very first iteration of the block, we get to mors-feedback before sliders are shown for the first time
        # TODO calculate weight estimate?
        # or just use relevance only baseline?
        weights = get_val("initial_weights")
        relevance_weight = weights["values"]["relevance"]
        diversity_weight = weights["values"]["diversity"]
        novelty_weight = weights["values"]["novelty"]
        exploration_weight = weights["values"]["exploration"]
        print(f"Using initial weights again: {relevance_weight, diversity_weight, novelty_weight, exploration_weight}")
        slider_relevance = slider_uniformity_diversity = slider_popularity_novelty = slider_exploitation_exploration = None

        # No sliders were shown, we have to set False to any objective ignoring for now
        ignore_exploitation_exploration = ignore_uniformity_diversity = ignore_popularity_novelty = ignore_relevance = False
    else:
        print("@@@@ Received following params:")
        print(request.form.get("selected_items"))
        slider_relevance = float(request.form.get("slider_relevance"))
        slider_exploitation_exploration = float(request.form.get("slider_exploitation_exploration"))
        slider_uniformity_diversity = float(request.form.get("slider_uniformity_diversity"))
        slider_popularity_novelty = float(request.form.get("slider_popularity_novelty"))
        print(slider_relevance, slider_exploitation_exploration, slider_uniformity_diversity, slider_popularity_novelty)
        print("@@@@")
    
        # Map the slider values to actual criteria weights
        relevance_weight = slider_relevance
        diversity_weight = slider_uniformity_diversity
        novelty_weight = slider_popularity_novelty
        exploration_weight = slider_exploitation_exploration

        # Get options regarding objective ignoring
        ignore_exploitation_exploration = request.form.get("ignore_exploitation_exploration") == "true"
        ignore_uniformity_diversity = request.form.get("ignore_uniformity_diversity") == "true"
        ignore_popularity_novelty = request.form.get("ignore_popularity_novelty") == "true"
        ignore_relevance = request.form.get("ignore_relevance") == "true"

        print(f"Ignoring options: [{ignore_relevance}, {ignore_uniformity_diversity}, {ignore_popularity_novelty}, {ignore_exploitation_exploration}]")

    # Set weights for inverse objectives
    uniformity_weight = 1.0 - diversity_weight
    popularity_weight = 1.0 - novelty_weight
    exploitation_weight = 1.0 - exploration_weight

    # Generate the actual recommendations
    recommendation = {}

    items = np.arange(loader.rating_matrix.shape[1])
    selected_metric_name = user_data['selected_metric_name']


    #cdf_div = load_cdf_cache(get_cache_path(get_semi_local_cache_name(loader)), selected_metric_name)

    k = conf["k"]


    # Generate recommendation for relevance only EASE
    ease = EASER_pretrained(items)
    ease = ease.load(os.path.join(get_cache_path(get_semi_local_cache_name(loader)), "item_item.npy"))
    elicitation_selected = np.array(user_data['elicitation_selected_movies'],  dtype=np.int32)
    # Train the algorithm on all movies selected during preference elicitation
    # and all movies previously selected during recommendations made by current algorithm
    training_selections = np.concatenate([elicitation_selected] + [np.array(x, dtype=np.int32) for x in selected_items_history[cur_algorithm]], dtype=np.int32)
    print(f"Training selections: {training_selections}, dtype={training_selections.dtype}, elicitation selected: {elicitation_selected.dtype}, history: {selected_items_history[cur_algorithm]}")
    print(f"Shown items so far: {user_data['shown_items']}")
    print(f"Shown items from current algorithm so far: {user_data['shown_items'][cur_algorithm]}")
    all_recommendations = sum(user_data["shown_items"][cur_algorithm], [])
    # Note that "all_recommendations" already includes items selected during the study (except those selected during elicitattion which we are thus adding)
    filter_out_items = np.concatenate([all_recommendations, elicitation_selected]).astype(np.int32)
    rel_scores, user_vector, relevance_top_k = ease.predict_with_score(training_selections, filter_out_items, k)
    relevance_top_k = np.array(relevance_top_k)

    # Needed for proper working of evolutionary algorithms
    # as here we need the objective values to be non-negative
    rel_scores_normed = (rel_scores - rel_scores.min()) / (rel_scores.max() - rel_scores.min())

    def relevance_f(top_k_list):
        return rel_scores[top_k_list].sum()
    
        # We construct all objectives as we want to use them for logging
    cf_ild = intra_list_diversity(loader.distance_matrix)
    cf_uniformity = intra_list_diversity(1.0 - loader.distance_matrix)
    cf_exploration = exploration(user_vector, loader.distance_matrix)
    cf_exploitation = exploitation(user_vector, 1.0 - loader.distance_matrix)
    
    distance_matrix_cb = get_distance_matrix_cb(os.path.join(get_cache_path(get_semi_local_cache_name(loader)), "distance_matrix_text.npy"))
    cb_ild = intra_list_diversity(distance_matrix_cb)
    cb_uniformity = intra_list_diversity(1.0 - distance_matrix_cb)
    cb_exploration = exploration(user_vector, distance_matrix_cb)
    cb_exploitation = exploitation(user_vector, 1.0 - distance_matrix_cb)

    bin_div = binomial_diversity(loader.all_categories, loader.get_item_index_categories, loader.rating_matrix, 0.0, loader.name())
    
    configured_metric_name = conf["mors_diversity_metric"]
    if configured_metric_name == "CF-ILD":
        div_f = cf_ild
        uniformity_f = cf_uniformity
        exploration_f = cf_exploration
        exploitation_f = cf_exploitation
        distance_matrix = loader.distance_matrix
    elif configured_metric_name == "CB-ILD":
        div_f = cb_ild
        uniformity_f = cb_uniformity
        exploration_f = cb_exploration
        exploitation_f = cb_exploitation
        distance_matrix = distance_matrix_cb
    else:
        assert False, f"Uknown metric name encountered in configuration: {configured_metric_name}"

    # if selected_metric_name == "CF-ILD":
    #     div_f = cf_ild
    # elif selected_metric_name == "CB-ILD":
    #     div_f = cb_ild
    # elif selected_metric_name == "BIN-DIV":
    #     div_f = bin_div

    popularity_f = item_popularity(loader.rating_matrix)
    novelty_f = popularity_based_novelty(loader.rating_matrix)

    if "EXACT" in cur_algorithm or "1W" in cur_algorithm:
        # We use one-way backend version of the slider and just 4 objectives
        target_weights = [
            relevance_weight    if not ignore_relevance else None,
            diversity_weight    if not ignore_uniformity_diversity else None,
            novelty_weight      if not ignore_popularity_novelty else None,
            exploration_weight  if not ignore_exploitation_exploration else None
        ]
        objectives = [
            ObjectiveWrapper(relevance_f, "relevance")      if not ignore_relevance else None,
            ObjectiveWrapper(div_f, "diversity")            if not ignore_uniformity_diversity else None,
            ObjectiveWrapper(novelty_f, "novelty")          if not ignore_popularity_novelty else None,
            ObjectiveWrapper(exploration_f, "exploration")  if not ignore_exploitation_exploration else None
        ]
    elif "MAX" in cur_algorithm:
        # We use two-way backend version of the slider and all 4 objectives
        target_weights = [
            relevance_weight    if not ignore_relevance else None,
            diversity_weight    if not ignore_uniformity_diversity else None,
            novelty_weight      if not ignore_popularity_novelty else None,
            exploration_weight  if not ignore_exploitation_exploration else None,
            uniformity_weight   if not ignore_uniformity_diversity else None,
            popularity_weight   if not ignore_popularity_novelty else None,
            exploitation_weight if not ignore_exploitation_exploration else None
        ]
        objectives = [
            ObjectiveWrapper(relevance_f, "relevance")          if not ignore_relevance else None,
            ObjectiveWrapper(div_f, "diversity")                if not ignore_uniformity_diversity else None,
            ObjectiveWrapper(novelty_f, "novelty")              if not ignore_popularity_novelty else None,
            ObjectiveWrapper(exploration_f, "exploration")      if not ignore_exploitation_exploration else None,
            ObjectiveWrapper(uniformity_f, "uniformity")        if not ignore_uniformity_diversity else None,
            ObjectiveWrapper(popularity_f, "popularity")        if not ignore_popularity_novelty else None,
            ObjectiveWrapper(exploitation_f, "exploitation")    if not ignore_exploitation_exploration else None
        ]
    else:
        # Not really used explicitly, we only keep it for logging that
        # takes place at the end of this function
        target_weights = [1.0]
        objectives = [
            ObjectiveWrapper(relevance_f, "relevance")
        ]

    # Do the actual filtering of objectives -> get rid of "None" places
    objectives = [obj for obj in objectives if obj is not None]
    target_weights = np.array([w for w in target_weights if w is not None])

    if "EVOLUTIONARY" in cur_algorithm:
        # We need to prepare mgain_cdf
        # Unfortunately calculate_normalization requires special wrappers over the objective
        # that support incrementally built user history

        def relevance_w(top_k_list, relevances, **kwargs):
            return relevances[top_k_list].sum()
            
        def diversity_w(top_k_list, *args, **kwargs):
            return div_f(top_k_list)
        
        def novelty_w(top_k_list, *args, **kwargs):
            return novelty_f(top_k_list)

        def exploration_w(top_k_list, user_vector_list, *args, **kwargs):
            f = exploration(np.array([]), distance_matrix)
            f.user_vector = np.array(user_vector_list, dtype=np.int32) # fixup
            return f(top_k_list)
        
        def exploitation_w(top_k_list, user_vector_list, *args, **kwargs):
            f = exploitation(np.array([]), 1.0 - distance_matrix)
            f.user_vector = np.array(user_vector_list, dtype=np.int32) # fixup
            return f(top_k_list)
        
        def popularity_w(top_k_list, *args, **kwargs):
            return popularity_f(top_k_list)
        
        def uniformity_w(top_k_list, *args, **kwargs):
            return uniformity_f(top_k_list)

        start_time = time.perf_counter()
        elicitation_shown = stable_unique(np.concatenate([get_val('elicitation_shown_movies'), elicitation_selected]))
        if "EXACT" in cur_algorithm or "1W" in cur_algorithm:
            # Only care about relevance, diversity, novelty and exploration. Minus everything that user wants to ignore
            wrapped_objectives = [
                ObjectiveWrapper(relevance_w, "relevance") if not ignore_relevance else None,
                ObjectiveWrapper(diversity_w, "diversity") if not ignore_uniformity_diversity else None,
                ObjectiveWrapper(novelty_w, "novelty") if not ignore_popularity_novelty else None,
                ObjectiveWrapper(exploration_w, "exploration") if not ignore_exploitation_exploration else None
            ]
        elif "MAX" in cur_algorithm:
            # Care about all 7 objectives, minus what the user wants to ignore
            wrapped_objectives = [
                ObjectiveWrapper(relevance_w, "relevance") if not ignore_relevance else None,
                ObjectiveWrapper(diversity_w, "diversity") if not ignore_uniformity_diversity else None,
                ObjectiveWrapper(novelty_w, "novelty") if not ignore_popularity_novelty else None,
                ObjectiveWrapper(exploration_w, "exploration") if not ignore_exploitation_exploration else None,
                ObjectiveWrapper(uniformity_w, "uniformity") if not ignore_uniformity_diversity else None,
                ObjectiveWrapper(popularity_w, "popularity") if not ignore_popularity_novelty else None,
                ObjectiveWrapper(exploitation_w, "exploitation") if not ignore_exploitation_exploration else None
            ]
        else:
            assert False, f"Unknown EVOLUTIONARY algorithm: {cur_algorithm}"
        
        wrapped_objectives = [obj for obj in wrapped_objectives if obj is not None]
        mgain_cdf, _, _ = calculate_normalization(ease, wrapped_objectives, np.concatenate([elicitation_shown, all_recommendations]).astype(np.int32), training_selections, k)
        print(f"Calculating marginal gain CDF took: {time.perf_counter() - start_time}")
        print(f"Inputs: {np.concatenate([elicitation_shown, all_recommendations]).astype(np.int32)}, {training_selections}")

    start_time = time.perf_counter()
    if cur_algorithm == "GREEDY-EXACT-1W":
        algo = greedy_exact(target_weights)
        top_k = morsify(
            k, rel_scores, algo,
            items, objectives, user_vector, filter_out_items, n_items_subset=N_ITEMS_SUBSET,
            do_normalize=True, rnd_mixture=True
        )
    elif cur_algorithm == "ITEM-WISE-EXACT-1W":
        algo = item_wise_exact(target_weights)
        top_k = morsify(
            k, rel_scores, algo,
            items, objectives, user_vector, filter_out_items, n_items_subset=N_ITEMS_SUBSET,
            do_normalize=True, rnd_mixture=True
        )
    elif cur_algorithm == "EVOLUTIONARY-EXACT-1W":
        # The exact variant takes rel_scores instead of rel_scores_normed, because objectives are not passed to jmetalpy framework directly
        # and thus does not have to be scales to >= 0. At the same time, mgain_cdf needs unnormalized rel_scores, so this actually makes it simpler
        #objectives[0] = ObjectiveWrapper(lambda x: rel_scores_normed[x].sum(), "relevance-normed")
        algo = evolutionary_exact(mgain_cdf, target_weights, rel_scores, user_vector, objectives, relevance_top_k, filter_out_items, k=k, time_limit_seconds=4)
        top_k = algo()
    elif cur_algorithm == "GREEDY-MAX-1W":
        algo = greedy_max(target_weights)
        top_k = morsify(
            k, rel_scores, algo,
            items, objectives, user_vector, filter_out_items, n_items_subset=N_ITEMS_SUBSET,
            do_normalize=True, rnd_mixture=True
        )
    elif cur_algorithm == "GREEDY-MAX-2W":
        algo = greedy_max(target_weights)
        top_k = morsify(
            k, rel_scores, algo,
            items, objectives, user_vector, filter_out_items, n_items_subset=N_ITEMS_SUBSET,
            do_normalize=True, rnd_mixture=True
        )
    elif cur_algorithm == "ITEM-WISE-MAX-1W":
        algo = item_wise_max(target_weights)
        top_k = morsify(
            k, rel_scores, algo,
            items, objectives, user_vector, filter_out_items, n_items_subset=N_ITEMS_SUBSET,
            do_normalize=True, rnd_mixture=True
        )
    elif cur_algorithm == "ITEM-WISE-MAX-2W":
        algo = item_wise_max(target_weights)
        top_k = morsify(
            k, rel_scores, algo,
            items, objectives, user_vector, filter_out_items, n_items_subset=N_ITEMS_SUBSET,
            do_normalize=True, rnd_mixture=True
        )
    elif cur_algorithm == "EVOLUTIONARY-MAX-1W":
        # For max, we need to distinguish between objectives passed to jmetalpy (have to be >= 0)
        # and our internal, raw objectives (does not have to be >= 0)
        # For us this is just about relevance, when can be negative and can cause problems
        # So we only care when ignore_relevance is False
        framework_objectives = objectives[:]
        if not ignore_relevance:
            framework_objectives[0] = ObjectiveWrapper(lambda x: rel_scores_normed[x].sum(), "relevance-normed")
        algo = evolutionary_max(mgain_cdf, target_weights, rel_scores,
                                user_vector, relevance_top_k, filter_out_items,
                                our_objectives=objectives, framework_objectives=framework_objectives, k=k, time_limit_seconds=4)
        top_k = algo()
    elif cur_algorithm == "EVOLUTIONARY-MAX-2W":
        # For max, we need to distinguish between objectives passed to jmetalpy (have to be >= 0)
        # and our internal, raw objectives (does not have to be >= 0)
        # For us this is just about relevance, when can be negative and can cause problems
        # So we only care when ignore_relevance is False
        framework_objectives = objectives[:]
        if not ignore_relevance:
            objectives[0] = ObjectiveWrapper(lambda x: rel_scores_normed[x].sum(), "relevance-normed")
        algo = evolutionary_max(mgain_cdf, target_weights, rel_scores,
                                user_vector, relevance_top_k, filter_out_items,
                                our_objectives=objectives, framework_objectives=framework_objectives, k=k, time_limit_seconds=4)
        top_k = algo()
    elif cur_algorithm == "RELEVANCE-BASED":
        top_k = relevance_top_k
    else:
        assert False, f"Unknown algorithm: {cur_algorithm} for it={it}"

    print(f"@@@ Morsify took: {time.perf_counter() - start_time}")

    algorithm_anon_name = f"ALGORITHM {ALGORITHM_ANON_NAMES[cur_block]}"
    recommendation[algorithm_anon_name] = {
        'movies': enrich_results(top_k, loader),
        'order': 0
    }

    # Increment the iteration
    incr("iteration")

    # Here we should generate recommendation and update the current list
    recommendations = get_val('recommendations')
    recommendations[cur_algorithm].append(recommendation)
    set_val('recommendations', recommendations)

    set_val('recommendation', recommendation)
    
    shown_items = get_val('shown_items')
    top_k_list = top_k.tolist()
    shown_items[cur_algorithm].append(top_k_list)
    set_val('shown_items', shown_items)

    slider_values = get_val("slider_values")
    slider_values["slider_relevance"].append(slider_relevance)
    slider_values["slider_exploitation_exploration"].append(slider_exploitation_exploration)
    slider_values["slider_uniformity_diversity"].append(slider_uniformity_diversity)
    slider_values["slider_popularity_novelty"].append(slider_popularity_novelty)
    set_val("slider_values", slider_values)

    #print(f"MORS feedback iter after: {session['iteration']} but should have been {it + 1}")
    #assert session.modified == True
    #session.modified = True

    # We log beginning of mors iteration
    data = {
        "iteration": it,
        "cur_block": cur_block,
        "cur_algorithm": cur_algorithm,
        "cur_slider_version": cur_slider_version,
        "selected_items_history": selected_items_history,
        "algorithm_anon_name": algorithm_anon_name,
        "top_k": top_k_list,
        "weights": target_weights.tolist(),
        "objective_names": [obj.name() for obj in objectives],
        "selected_metric_name": selected_metric_name,
        "configured_metric_name": configured_metric_name,
        "raw_slider_values": {
            "relevance": slider_relevance,
            "diversity": slider_uniformity_diversity,
            "novelty": slider_popularity_novelty,
            "exploration": slider_exploitation_exploration
        },
        "ignore_objectives": {
            "relevance": ignore_relevance,
            "diversity": ignore_uniformity_diversity,
            "novelty": ignore_popularity_novelty,
            "exploration": ignore_exploitation_exploration
        },
        "objectives": {
            "relevance": relevance_f(top_k).item(),
            "relevance-normed": rel_scores_normed[top_k].sum().item(),
            "cf_ild": cf_ild(top_k).item(),
            "cb_ild": cb_ild(top_k).item(),
            "bin_div": bin_div(top_k).item(),
            "novelty": novelty_f(top_k).item(),
            "cf_exploration": cf_exploration(top_k).item(),
            "cb_exploration": cb_exploration(top_k).item(),
            "cf_uniformity": cf_uniformity(top_k).item(),
            "cb_uniformity": cb_uniformity(top_k).item(),
            "cf_exploitation": cf_exploitation(top_k).item(),
            "cb_exploitation": cb_exploitation(top_k).item(),
            "popularity": popularity_f(top_k).item()
        },
        "ease_selections": training_selections.tolist(),
        "ease_filter_out": filter_out_items.tolist(),
        "ease_top_k": relevance_top_k.tolist()
    }
    log_interaction(session["participation_id"], "mors-recommendation-started", **data)
    return flask.Flask.redirect(flask.current_app, url_for("journal.mors"))

# Rendering endpoint showing the recommendations
@bp.route("/mors", methods=["GET", "POST"])
def mors():
    #it = session['iteration']
    user_data = get_all(get_uname())
    it = user_data["iteration"]

    print(f"MORS IT={it}, mod={int(it) - 1 % N_ITERATIONS}")

    cur_block = (int(it) - 1) // N_ITERATIONS
    cur_algorithm = user_data["selected_algorithms"][cur_block]
    cur_slider_version = user_data["selected_slider_versions"][cur_block]
    show_modal = int(it) - 1 == 0 # Only show the modal in the very first iteration

    if (int(it) - 1) % N_ITERATIONS == 0:
        # Initialize sliders on the frontend to estimated weights
        weights = get_val("initial_weights")
        slider_relevance = weights["values"]["relevance"]
        slider_uniformity_diversity = weights["values"]["diversity"]
        slider_popularity_novelty = weights["values"]["novelty"]
        slider_exploitation_exploration = weights["values"]["exploration"]
        print(f"Setting sliders to initial estimate {[slider_relevance, slider_uniformity_diversity, slider_popularity_novelty, slider_exploitation_exploration]}")
    else:
        # Otherwise get values previously set by the user
        slider_values = get_val("slider_values")
        slider_relevance = slider_values["slider_relevance"][-1]
        slider_exploitation_exploration = slider_values["slider_exploitation_exploration"][-1]
        slider_uniformity_diversity = slider_values["slider_uniformity_diversity"][-1]
        slider_popularity_novelty = slider_values["slider_popularity_novelty"][-1]
        print(f"Setting sliders to previous values: {[slider_relevance, slider_uniformity_diversity, slider_popularity_novelty, slider_exploitation_exploration]}")

    print(f"Algorithm = {cur_algorithm}")

    conf = load_user_study_config(session['user_study_id'])

    # We are at the end of block
    if it > 0 and it % N_ITERATIONS == 0:
        continuation_url = url_for("journal.block_questionnaire")
    else:
        continuation_url = url_for("journal.mors_feedback")

    tr = get_tr(languages, get_lang())
    # TODO replace random with user_data['movies'] which in turn should be filled in by recommendation algorithms
    params = {
        "continuation_url": continuation_url,
        "iteration": (it - 1) % N_ITERATIONS + 1, #it,
        "n_iterations": N_ITERATIONS,
        "block": cur_block + 1,
        "n_blocks": N_BLOCKS,
        "movies": get_val('recommendation'),
        "like_nothing": tr("compare_like_nothing"),
        "can_fine_tune": cur_algorithm != "RELEVANCE-BASED",
        "slider_version": cur_slider_version,
        "slider_relevance": slider_relevance,
        "slider_uniformity_diversity": slider_uniformity_diversity,
        "slider_popularity_novelty": slider_popularity_novelty,
        "slider_exploitation_exploration": slider_exploitation_exploration,
        "show_modal": show_modal
    }

    if is_books(conf):
        params["title"] = tr("journal_recommendation_title_books")
        params["header"] = tr("journal_recommendation_header_books")
        params["hint"] = tr("journal_recommendation_hint_books")
        params["highly_popular_help"] = tr("journal_highly_popular_help_books")
        params["highly_novel_help"] = tr("journal_highly_novel_help_books")
        params["highly_diverse_help"] = tr("journal_highly_diverse_help_books")
        params["highly_uniform_help"] = tr("journal_highly_uniform_help_books")
        params["highly_explorative_help"] = tr("journal_highly_explorative_help_books")
        params["highly_exploitative_help"] = tr("journal_highly_exploitative_help_books")
        params["highly_relevant_help"] = tr("journal_highly_relevant_help_books")
        params["less_relevant_help"] = tr("journal_less_relevant_help_books")
        params["less_novel_help"] = tr("journal_less_novel_help_books")
        params["less_diverse_help"] = tr("journal_less_diverse_help_books")
        params["less_explorative_help"] = tr("journal_less_explorative_help_books")
        params["more_relevant_help"] = tr("journal_more_relevant_help_books")
        params["more_novel_help"] = tr("journal_more_novel_help_books")
        params["more_diverse_help"] = tr("journal_more_diverse_help_books")
        params["more_explorative_help"] = tr("journal_more_explorative_help_books")
        params["block_start_header"] = tr("journal_block_start_header_books")
        params["block_start_detail"] = tr("journal_block_start_detail_books")
    else:
        params["title"] = tr("journal_recommendation_title_movies")
        params["header"] = tr("journal_recommendation_header_movies")
        params["hint"] = tr("journal_recommendation_hint_movies")
        params["highly_popular_help"] = tr("journal_highly_popular_help_movies")
        params["highly_novel_help"] = tr("journal_highly_novel_help_movies")
        params["highly_diverse_help"] = tr("journal_highly_diverse_help_movies")
        params["highly_uniform_help"] = tr("journal_highly_uniform_help_movies")
        params["highly_explorative_help"] = tr("journal_highly_explorative_help_movies")
        params["highly_exploitative_help"] = tr("journal_highly_exploitative_help_movies")
        params["highly_relevant_help"] = tr("journal_highly_relevant_help_movies")
        params["less_relevant_help"] = tr("journal_less_relevant_help_movies")
        params["less_novel_help"] = tr("journal_less_novel_help_movies")
        params["less_diverse_help"] = tr("journal_less_diverse_help_movies")
        params["less_explorative_help"] = tr("journal_less_explorative_help_movies")
        params["more_relevant_help"] = tr("journal_more_relevant_help_movies")
        params["more_novel_help"] = tr("journal_more_novel_help_movies")
        params["more_diverse_help"] = tr("journal_more_diverse_help_movies")
        params["more_explorative_help"] = tr("journal_more_explorative_help_movies")
        params["block_start_header"] = tr("journal_block_start_header_movies")
        params["block_start_detail"] = tr("journal_block_start_detail_movies")

    #assert session.modified == False
    return render_template("mors.html", **params)


# Endpoint for block questionnaire
@bp.route("/block-questionnaire", methods=["GET", "POST"])
def block_questionnaire():
    user_data = get_all(get_uname())
    it = user_data["iteration"]
    cur_block = (int(it) - 1) // N_ITERATIONS
    params = {
        "continuation_url": url_for("journal.block_questionnaire_done"),
        "header": f"After-recommendation block questionnaire for block: ({cur_block + 1}/{N_BLOCKS})",
        "hint": "Before proceeding to the next step, please answer questions specific to the recent block (last 6 iterations) of recommendations.",
        "finish": "Continue",
        "title": "Questionnaire"
    }
    return render_template("journal_block_questionnaire.html", **params)

# Endpoint that should be called once block questionnaire is done
@bp.route("/block-questionnaire-done", methods=["GET", "POST"])
def block_questionnaire_done():
    user_data = get_all(get_uname())
    it = user_data["iteration"]
    cur_block = (int(it) - 1) // N_ITERATIONS
    cur_algorithm = user_data["selected_algorithms"][cur_block]

    # Log the iteration block, algorithm as well as responses to all the questions
    data = {
        "block": cur_block,
        "algorithm": cur_algorithm,
        "iteration": it
    }
    data.update(**request.form)

    if cur_block == N_BLOCKS - 1:
        # We are done, do not forget to mark last iteration as ended
        log_interaction(session["participation_id"], "mors-recommendation-ended", iteration=it - 1)
        log_interaction(session["participation_id"], "after-block-questionnaire", **data)
        return redirect(url_for("journal.done"))
    else:
        # Otherwise continue with next block
        log_interaction(session["participation_id"], "after-block-questionnaire", **data)
        return redirect(url_for("journal.mors_feedback"))

@bp.route("/done", methods=["GET", "POST"])
def done():
    return redirect(url_for("journal.final_questionnaire"))

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
    data = {}
    data.update(**request.form)
    log_interaction(session["participation_id"], "final-questionnaire", **data)

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
    it = user_data["iteration"]
    cur_block = (int(it) - 1) // N_ITERATIONS
    cur_algorithm = user_data["selected_algorithms"][cur_block]

    conf = load_user_study_config(session['user_study_id'])

    if "Goodbooks" in conf["selected_data_loader"]:
        item_text = "books"
        popular_text = "bestsellers"
    else:
        item_text = "movies"
        popular_text = "blockbusters"

    questions = [
        {
            "text": f"The {item_text} recommended to me matched my interests.",
            "name": "q1",
            "icon": "grid"
        },
        {
            "text": f"The recommended {item_text} were mostly novel to me.",
            "name": "q2",
            "icon": "grid"
        },
        {
            "text": f"The recommended {item_text} were highly different from each other.",
            "name": "q3",
            "icon": "grid"
        },
        {
            "text": f"The recommended {item_text} were unexpected yet interesting to me.",
            "name": "q4",
            "icon": "grid"
        },
        {
            "text": f"The recommended {item_text} differed from my usual choices.",
            "name": "q5",
            "icon": "grid"
        },
        {
            "text": f"The recommended {item_text} were mostly similar to what I usually watch.",
            "name": "q6",
            "icon": "grid"
        },
        {
            "text": f"The recommended {item_text} were mostly popular (i.e., {popular_text}).",
            "name": "q7",
            "icon": "grid"
        },
        {
            "text": f"The recommended {item_text} were mostly similar to each other",
            "name": "q8",
            "icon": "grid"
        },
        {
            "text": f"Overall, I am satisfied with the recommended {item_text}.",
            "name": "q9",
            "icon": "grid"
        },
        {
            "text": "The initial values of sliders already provided good recommendations.",
            "name": "q10",
            "icon": "sliders"
        },
        {
            "text": "Being able to change objective criteria (relevance, diversity, etc.) was useful for me.",
            "name": "q11",
            "icon": "sliders"
        },
        {
            "text": "Overall, after modifying the objective criteria, recommendations improved.",
            "name": "q12",
            "icon": "sliders"
        },
        {
            "text": "The mechanism (slider) was sufficient to tell the system what recommendations I wanted.",
            "name": "q13",
            "icon": "sliders"
        },
        {
            "text": "I was able to describe my preferences w.r.t. supported objective criteria.",
            "name": "q14",
            "icon": "sliders"
        },
        {
            "text": "Setting appropriate values for the objective criteria was straightforward.",
            "name": "q15",
            "icon": "sliders"
        }
    ]

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
            "icon": "sliders"
        },
        {
            "text": "This recommender system provided me with many tips for interesting computer games.",
            "name": "qs3",
            "icon": "sliders",
            "atn": "true"
        },
        {
            "text": "The recommendations I got from this system provided me with great recipes for exotic cuisines.",
            "name": "qs4",
            "icon": "sliders",
            "atn": "true"
        }
    ]

    # Mapping blocks to indices at which we place attention checks
    atn_check_indices = {
        0: 7,
        1: 11,
        2: 15,
        3: 14
    }

    if cur_algorithm == "RELEVANCE-BASED":
        # If this is the case, we only show first 9 questions
        questions = questions[:9]
        # However, this also means that we have to redistribute attention checks!
        atn_check_indices = {
            0: 1,
            1: 6,
            2: 9,
            3: 8
        }
        # No sliders icon for relevance-only so having different icon for attention check would make it too easy
        attention_checks[3]["icon"] = "grid"
        attention_checks[2]["icon"] = "grid"
        attention_checks[1]["icon"] = "grid"

    # Use first attention check
    questions.insert(atn_check_indices[cur_block], attention_checks[cur_block])
    if cur_block == 1:
        # For second block we have two attention checks
        questions.insert(atn_check_indices[3], attention_checks[3])
    assert cur_block >= 0 and cur_block < N_BLOCKS, f"cur_block == {cur_block}"

    return jsonify(questions)

@bp.route("/get-final-questions", methods=["GET"])
def get_final_questions():
    conf = load_user_study_config(session['user_study_id'])

    if "Goodbooks" in conf["selected_data_loader"]:
        item_text = "books"
        production = "Amazon, GoodReads"
        ic = "book"
    else:
        item_text = "movies"
        production = "Netflix, Disney+"
        ic = "collection-play"

    questions = [
        {
            "text": f"The information provided for the recommended {item_text} was sufficient to judge whether I gonna like them.",
            "name": "q1",
            "icon": "chat-left-text"
        },
        {
            "text": "The description of objective criteria (relevance, diversity, etc.) was clear and sufficient.",
            "name": "q2",
            "icon": "sliders"
        },
        {
            "text": "I understood the purpose of tweaking objective criteria.",
            "name": "q3",
            "icon": "sliders"
        },
        {
            "text": f"I would like to have a similar objective tweaking mechanism in production systems (e.g., {production}) too.",
            "name": "q4",
            "icon": ic
        }
    ]
    return jsonify(questions)

@bp.route("/get-engagement-question", methods=["GET"])
def get_engagement_question():
    conf = load_user_study_config(session['user_study_id'])

    if is_books(conf):
        text= "How frequently do you find yourself reading books?"
        options = [
            "Non-Reader / Rare Reader (read books rarely/never)",
            "Occasional Reader (read books infrequently, perhaps few times a year)",
            "Regular Reader (read one or two books a month)",
            "Enthusiastic Readers (read several books a more a month)"
        ]
    else:
        text = "How often do you typically engage in watching movies?"
        options = [
            "Non-Watcher / Infrequent Viewer (watch movies rarely/never)",
            "Casual Viewer (watch movies occasionally, maybe a few times a week)",
            "Frequent Viewer (watch one or two movies per week)",
            "Enthusiastic Viewer (watch several movies a week)"
        ]

    question = {
        "text": text,
        "options": options
    }
    return jsonify(question)

@bp.route("/get-instruction-bullets", methods=["GET"])
def get_instruction_bullets():
    page = request.args.get("page")
    if not page:
        return jsonify([])

    conf = load_user_study_config(session['user_study_id'])

    if page == "mors":
        if is_books(conf):
            bullets = [
                "Books are selected by clicking on them; each selected book is highlighted by a green frame."
                "If you do not like any of the recommended books, there is a button at the bottom of this page that you should check.",
                "When a mouse cursor is placed over a book, its title and description will be shown.",
                "Completion of each step is final, and you cannot return to previous pages.",
                "Also note that each book will be displayed only once within the block (i.e., you need to make an immediate decision)."
            ]
        else:
            bullets = [
                "Movies are selected by clicking on them; each selected movie is highlighted by a green frame.",
                "If you do not like any of the recommended movies, there is a button at the bottom of this page that you should check.",
                "When a mouse cursor is placed over a movie, its title and description will be shown.",
                "Completion of each step is final, and you cannot return to previous pages.",
                "Also note that each movie will be displayed only once within the block (i.e., you need to make an immediate decision)."
            ]
    elif page == "block-questionnaire":
        bullets = [
            "Important: These questions are about the recent recommendations, specifically recommendations from the last block (meaning the last 6 iterations).",
            "Please answer them before moving on to the next step in the study.",
            "If any question is unclear, choose 'I don't understand' as your response for that specific question."
        ]
    elif page == "pre-study-questionnaire":
        bullets = [
            "Please answer these questions before moving on to the next step in the study.",
            "For questions 7 to 11, pick the answer that most aligns with your personal understanding of the specific recommendation mentioned in each question.",
            "Considering that different people may have different views/interpretations, you can choose 'Other' and share your personal thoughts in the text box below each question."
        ]
    elif page == "final-questionnaire":
        bullets = [
            "Important: These questions are about the experience during the WHOLE user study.",
            "Please answer these questions before finishing the study."
        ]
    elif page == "metric-assessment":
        items_name = "books" if is_books(conf) else "movies"
        bullets = [
            f"This page displays three lists of recommendations: A, B, C, which were created for you based on the {items_name} you chose at the start of this study as part of stating your preferences (preference elicitation).",
            "Your goal is to choose the one list that you think is the most DIVERSE (based on how you understand or interpret diversity).",
            f"You can choose a list by clicking on any of its {items_name}.",
            f"If you want more information about the displayed {items_name}, simply hover your mouse cursor over it, and both the description and its name will appear."
        ]
    elif page == "compare-alphas":
        items_name = "books" if is_books(conf) else "movies"
        bullets = [
            f"This page once again presents three recommendation lists, but they differ from the previous step and are now labeled either 1, 2, 3 (in the first step) or 4, 5, 6 (in the second step).",
            "Your goal is to order the recommendation lists from least diverse to most diverse.",
            "To order them, drag and drop the colorful rectangles located at the bottom of this page into the adjacent gray area.",
            "The further right you position the lists, the more diversity you perceive in them.",
            f"If you want more information about the displayed {items_name}, simply hover your mouse cursor over it, and both the description and its name will appear."
        ]
    else:
        bullets = []

    return jsonify(bullets)
get_instruction_bullets

# Endpoint used to search items
@bp.route("/item-search", methods=["GET"])
def item_search():
    pattern = request.args.get("pattern")
    if not pattern:
        return make_response("", 404)

    conf = load_user_study_config(session['user_study_id'])

    ## TODO get_loader helper
    loader_factory = load_data_loaders()[conf["selected_data_loader"]]
    loader = loader_factory(**filter_params(conf["data_loader_parameters"], loader_factory))
    loader = load_data_loader_cached(loader, session['user_study_guid'], loader_factory.name(), get_semi_local_cache_name(loader))

    res = search_for_item(pattern, loader, tr=None)

    return jsonify(res)


# Given list of shown_items and selected_items
# train CDF for every objective
# Shown_items are partitioned into N parts of length part_length
# so when we iteratively calculate mgains, we always calculate it w.r.t. partition (to avoid mgain calculation against very long lists)
def calculate_normalization(algo, objectives, shown_items, selected_items, part_length):

    # n_parts = np.ceil(shown_items.size / part_length).astype(int)
    parts = np.array_split(shown_items, np.arange(part_length, shown_items.size, part_length))
    #print(f"Parts = {parts}")
    user_vector_incremental = []
    mgains = np.zeros(shape=(len(objectives), len(shown_items)), dtype=np.float32)
    selected_mgains = [[] for _ in objectives]

    items = algo.all_items
    # items = np.arange(loader.rating_matrix.shape[1])
    # algo = EASER_pretrained(items)
    # algo = algo.load(os.path.join(get_cache_path(get_semi_local_cache_name(loader)), "item_item.npy"))

    # To be consistent with exploration, that returns 0 when no user history is available
    # we say items have 0 relevancy when we have no user history
    relevances = np.zeros(shape=(items.size, ), dtype=np.float32)

    for obj_idx, obj in enumerate(objectives):
        for idx, item_idx in enumerate(shown_items):
            # Get index of the part to which current item belongs to
            cur_part = parts[idx // part_length]
            within_part_idx = idx % cur_part.size

            # Calculate objective of old list (before adding item_idx, thus using old relevances and old user_vector_incremental)
            obj_old = obj(cur_part[:within_part_idx], user_vector_list=user_vector_incremental, relevances=relevances)

            if item_idx in selected_items:
                # We will end up in this place exactly len(objectives)-times, so we only append for the first time
                if obj_idx == 0:
                    print(f"Item idx = {item_idx}, idx = {idx} is selected, updating relevance")
                    user_vector_incremental.append(item_idx)
                    # Update relevance only prediction
                    user_selections = np.array(user_vector_incremental)
                    user_vector = np.zeros(shape=(items.size,), dtype=algo.item_item.dtype)
                    user_vector[user_selections] = 1
                    relevances = np.dot(user_vector, algo.item_item)
                    print(f"Selected items: {user_selections} and their relevances: {relevances[user_selections]}")
                    #relevances = (relevances - np.min(relevances)) / (np.max(relevances) - np.min(relevances))

                # Since we added new item, we have to take updated user vector into an account
                if obj_idx == 0:
                    print(f"Selected 2 items: {user_selections} and their relevances: {relevances[user_selections]}")
                obj_new = obj(cur_part[:within_part_idx+1], user_vector_list=user_vector_incremental, relevances=relevances)
                mgains[obj_idx, idx] = obj_new - obj_old
                if obj_idx == 0:
                    print(f"obj_new -> {obj_new}, obj_old -> {obj_old}")
                selected_mgains[obj_idx].append(mgains[obj_idx, idx])
            else:
                # We did not select anything, so user vector stays intact
                mgains[obj_idx, idx] = obj(cur_part[:within_part_idx+1], user_vector_list=user_vector_incremental, relevances=relevances) - obj_old

            if obj_idx == 0:
                print(f"Mgains at index: obj={obj_idx}, idx={idx}, item_idx={item_idx}, within_part_idx={within_part_idx}, part_idx={idx // part_length}, cur_part={cur_part}: {mgains[obj_idx, idx]}")

    # Default number of quantiles is 1000, however, if n_samples is smaller than n_quantiles, then n_samples is used and warning is raised
    # to get rid of the warning, we calculates quantiles straight away
    obj_cdf = QuantileTransformer(n_quantiles=min(1000, mgains.shape[1])).fit(mgains.T)
    return obj_cdf, mgains, selected_mgains

# Stable implementation of np.unique
# meaning that elements are not sorted as in np.unique
# but keep stable order of appearance
def stable_unique(x):
    if type(x) is not np.ndarray:
        x = np.array(x)
    _, index = np.unique(x, return_index=True)
    return x[np.sort(index)]

# If exact = True, then we are in "exact" algorithm world as apposed to "max", this also means that we do not normalize weights to sum to 1
# selected_movies -> movies selected during elicitation
# elicitation_movies -> movies displayed during elicitation
# Note that we DO NOT NORMALIZE the WEIGHTS here
def calculate_weight_estimate_generic(algo, objectives, selected_movies, elicitation_movies, return_supports=False):
    
    n_objectives = len(objectives)
    if not selected_movies:
        x = {
            "vec": np.array([1.0] * n_objectives, dtype=np.float32),
            "values": {
                obj.name(): np.float32(1.0) for obj in objectives
            }
        }
        if return_supports:
            return x, {}
        else:
            return x

    # If movies are selected after search, they are not correctly propagated through elicitation_movies
    # therefore we append them here
    # We need to keep order of appearance instead of sorted order as is done by np.unique alone
    movie_indices = stable_unique(np.concatenate([elicitation_movies, selected_movies]))

    obj_cdf, mgains, selected_mgains = calculate_normalization(algo, objectives, movie_indices, selected_movies, len(selected_movies))
    selected_mgains = np.array(selected_mgains)
    if selected_mgains.size == 0:
        x = {
            "vec": np.array([1.0] * n_objectives, dtype=np.float32),
            "values": {
                obj.name(): np.float32(1.0) for obj in objectives
            }
        }
        if return_supports:
            return x, {}
        else:
            return x


    selected_mgains_normed = obj_cdf.transform(selected_mgains.T)
    result_weight_vec = selected_mgains_normed.mean(axis=0)
    result = {
            "vec": result_weight_vec,
            "values": {
                obj.name(): result_weight_vec[obj_idx] for obj_idx, obj in enumerate(objectives)
            }
        }

    if return_supports:
        supports = {
            "elicitation_movies": np.squeeze(movie_indices),
            "selected_movies": np.array(selected_movies),
            "selected_mgains": selected_mgains,
            "selected_mgains_normed": selected_mgains_normed,
            "mgains": mgains
        }
        return result, supports

    return result

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