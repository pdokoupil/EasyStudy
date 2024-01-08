from abc import ABC
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

from models import UserStudy

from multiprocessing import Process
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import numpy as np

from common import get_tr, load_languages, multi_lang, load_user_study_config
from flask import Blueprint, jsonify, make_response, request, redirect, render_template, url_for, session

from plugins.utils.preference_elicitation import recommend_2_3, rlprop, weighted_average, get_objective_importance, prepare_tf_model, calculate_weight_estimate, load_ml_dataset, enrich_results
from plugins.utils.interaction_logging import log_interaction, log_message
from plugins.fastcompare.loading import load_data_loaders
from plugins.fastcompare import elicitation_ended, filter_params, iteration_started, iteration_ended, load_data_loader, load_data_loader_cached, search_for_item
#from plugins.multiobjective import prepare_recommendations
from plugins.fastcompare import prepare_recommendations, get_semi_local_cache_name, get_cache_path
from plugins.journal.metrics import binomial_diversity, intra_list_diversity, item_popularity, popularity_based_novelty, exploration, exploitation

#from memory_profiler import profile

from plugins.journal.algorithms import NEG_INF, evolutionary_exact, evolutionary_max, greedy_exact, greedy_max, item_wise_exact, item_wise_max

from app import rds

languages = load_languages(os.path.dirname(__file__))
N_BLOCKS = 3
N_ITERATIONS = 6 # 6 iterations per block
#ALGORITHMS = ["WA", "RLPROP", "MOEA-RS"]
# Algorithm matrix
# Algorithms are divided in two dimensions: [MAX, EXACT] x [ITEM-WISE, GREEDY, EVOLUTIONARY]
# We always select one row per user, so each user gets either ITEM-WISE, GREEDY or EVOLUTIONARY (in both MAX and EXACT variants) + Relevance baseline
ALGORITHMS = [
    "ITEM-WISE-MAX-1W", "ITEM-WISE-MAX-2W", "ITEM-WISE-EXACT-1W",
    "GREEDY-MAX-1W", "GREEDY-MAX-2W", "GREEDY-EXACT-1W",
    "EVOLUTIONARY-MAX-1W", "EVOLUTIONARY-MAX-2W", "EVOLUTIONARY-EXACT-1W",
    "RELEVANCE-BASED"
]

# ALGORITHM_ROWS = {
#     "ITEM-WISE": ["ITEM-WISE-MAX", "ITEM-WISE-EXACT"],
#     "GREEDY": ["GREEDY-MAX", "GREEDY-EXACT"],
#     "EVOLUTIONARY": ["EVOLUTIONARY-MAX", "EVOLUTIONARY-EXACT"]
# }

OPTIMIZATION_TYPE = ["MAX", "EXACT"]

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
POSSIBLE_ALPHAS = [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
# one-way means there is single objective per slider, going from 0 to 1
# two-way means there are two objectives, one at each side of the slider
SLIDER_VERSIONS = ["TWO-WAY", "ONE-WAY"]


HIDE_LAST_K = 100000
N_ALPHA_ITERS = 2


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

__plugin_name__ = "journal"
__version__ = "0.1.0"
__author__ = "Patrik Dokoupil"
__author_contact__ = "Patrik.Dokoupil@matfyz.cuni.cz"
__description__ = "Complex plugin for a very customized user study that we have used for journal paper"

bp = Blueprint(__plugin_name__, __plugin_name__, url_prefix=f"/{__plugin_name__}")

# Redis helpers
# TODO deice if should be moved somewhere else
# Return user key for a given user (by session)
def get_uname():
    return f"user:{session['uuid']}"

# Wrapper for setting values, performs serialization via pickle
def set_val(key, val):
    name = get_uname()
    print("Called setval with=", pickle.dumps(val))
    rds.hset(name, key, value=pickle.dumps(val))

def set_mapping(name, mapping):
    rds.hset(name, mapping={x: pickle.dumps(v) for x, v in mapping.items()})

# Wrapper for getting values, performs deserialization via pickle
def get_val(key):
    name = get_uname()
    return pickle.loads(rds.hget(name, key))

def get_all(name):
    res = {str(x, encoding="utf-8") : pickle.loads(v) for x, v in rds.hgetall(name).items()}
    return res

def incr(key):
    x = get_val(key)
    set_val(key, x + 1)

def is_books(conf):
    return "Goodbooks" in conf["selected_data_loader"]

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

# Callback once user has joined we forward to preference elicitation
@bp.route("/on-joined", methods=["GET", "POST"])
def on_joined():
    return redirect(url_for("journal.pre_study_questionnaire")
    )

displyed_name_mapping = {
    "relevance_based": "BETA",
    "weighted_average": "GAMMA",
    "rlprop": "DELTA"
}

refinement_layouts = {
    "default": "0",
    "default_shifted": "8",
    "buttons": "7",
    "options": "4"
}

@bp.context_processor
def plugin_name():
    return {
        "plugin_name": __plugin_name__
    }

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

@bp.route("/block-questionnaire", methods=["GET", "POST"])
def block_questionnaire():
    params = {
        "continuation_url": url_for("journal.block_questionnaire_done"),
        "header": "After-recommendation block questionnaire",
        "hint": "Please answer the questions below before proceeding with the next step of the user study.",
        "finish": "Continue",
        "title": "Questionnaire"
    }
    return render_template("journal_block_questionnaire.html", **params)

@bp.route("/block-questionnaire-done", methods=["GET", "POST"])
def block_questionnaire_done():
    user_data = get_all(get_uname())
    it = user_data["iteration"]
    cur_block = (int(it) - 1) // N_ITERATIONS
    cur_algorithm = user_data["selected_algorithms"][cur_block]

    # TODO log interaction
    data = {
        "block": cur_block,
        "algorithm": cur_algorithm,
        "iteration": it
    }
    data.update(**request.form)

    log_interaction(session["participation_id"], "after-block-questionnaire", **data)

    if cur_block == N_BLOCKS - 1:
        # We are done
        return redirect(url_for("journal.done"))
    else:
        # Otherwise continue with next block
        return redirect(url_for("journal.mors_feedback"))


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

@bp.route("/pre-study-questionnaire-done", methods=["GET", "POST"])
def pre_study_questionnaire_done():

    data = {}
    data.update(**request.form)

    log_interaction(session["participation_id"], "pre-study-questionnaire", **data)

    return redirect(url_for("utils.preference_elicitation", continuation_url=url_for("journal.send_feedback"),
            consuming_plugin=__plugin_name__,
            initial_data_url=url_for('fastcompare.get_initial_data'),
            search_item_url=url_for('journal.item_search')))

@bp.route("/final-questionnaire", methods=["GET", "POST"])
def final_questionnaire():
    params = {
        "continuation_url": url_for("journal.finish_user_study"),
        "header": "Final questionnaire",
        "hint": "Please answer the questions below before finishing the user study.",
        "finish": "Finish",
        "title": "Final questionnaire"
    }
    return render_template("journal_final_questionnaire.html", **params)

@bp.route("/finish-user-study", methods=["GET", "POST"])
def finish_user_study():
    # TODO handle final questionnaire feedback

    data = {}
    data.update(**request.form)
    log_interaction(session["participation_id"], "final-questionnaire", **data)

    session["iteration"] = get_val("iteration")
    return redirect(url_for("utils.finish"))

# Receives arbitrary feedback (typically from preference elicitation) and generates recommendation
@bp.route("/send-feedback", methods=["GET"])
def send_feedback():
    # We read k from configuration of the particular user study
    conf = load_user_study_config(session['user_study_id'])
    k = conf["k"]

    # Get a loader
    loader_factory = load_data_loaders()[conf["selected_data_loader"]]
    loader = loader_factory(**filter_params(conf["data_loader_parameters"], loader_factory))
    loader = load_data_loader_cached(loader, session['user_study_guid'], loader_factory.name(), get_semi_local_cache_name(loader))

    # Movie indices of selected movies
    selected_movies = request.args.get("selectedMovies")
    selected_movies = selected_movies.split(",") if selected_movies else []
    selected_movies = [int(m) for m in selected_movies]

    # Calculate weights based on selection and shown movies during preference elicitation
    diversity_f = intra_list_diversity(loader.distance_matrix)
    novelty_f = popularity_based_novelty(loader.rating_matrix)
    relevances = loader.rating_matrix.mean(axis=0)

    def relevance_f(top_k_list):
        return relevances[top_k_list].sum()
    
    def relevance_w(top_k_list, *args, **kwargs):
        return relevance_f(top_k_list)
    
    # Wrapped diversity
    def diversity_w(top_k_list, *args, **kwargs):
        return diversity_f(top_k_list)
    
    def novelty_w(top_k_list, *args, **kwargs):
        return novelty_f(top_k_list)

    def exploration_w(top_k_list, user_vector_list, *args, **kwargs):
        f = exploration(np.array([]), loader.distance_matrix)
        f.user_vector = np.array(user_vector_list, dtype=np.int32) # fixup
        return f(top_k_list)

    objectives = [
        ObjectiveWrapper(relevance_w, "relevance"),
        ObjectiveWrapper(diversity_w, "diversity"),
        ObjectiveWrapper(novelty_w, "novelty"),
        ObjectiveWrapper(exploration_w, "exploration")
    ]
    weights, supports = calculate_weight_estimate_generic(loader, objectives, selected_movies, session['elicitation_movies'], return_supports=True)
    # weights /= weights.sum()
    #weights = weights.tolist()
    # weights = {
    #     algo_displayed_name: weights for algo_displayed_name in displyed_name_mapping.values()
    # }
    #rds.hset(f"user:weights:{session['uuid']}:", mapping=weights)
    print(f"Weights initialized to {weights}, supports: {supports}")

    # Add default entries so that even the non-chosen algorithm has an empty entry
    # to unify later access
    # recommendations = {
    #     x["displayed_name"]: [[]] for x in conf["algorithm_parameters"]
    # }
    # initial_weights_recommendation = {
    #     x["displayed_name"]: [[]] for x in conf["algorithm_parameters"]
    # }
    
    # We filter out everything the user has selected during preference elicitation.
    # However, we allow future recommendation of SHOWN, NOT SELECTED (during elicitation, not comparison) altough these are quite rare
    filter_out_movies = selected_movies

    #prepare_recommendations(loader, conf, recommendations, selected_movies, filter_out_movies, k)
    #prepare_recommendations(weights, recommendations, initial_weights_recommendation, selected_movies, filter_out_movies, k)
    #print(f"Recommendations={recommendations}")
    
    set_mapping(get_uname(), {
        # 'movies': recommendations,
        # 'initial_weights_recommendation': initial_weights_recommendation,
        'initial_weights': weights,
        'iteration': 0, # Start with zero, because at the very beginning, mors_feedback is called, not mors and that generates recommendations for first iteration, but at the same time, increases the iteration
        'elicitation_selected_movies': selected_movies,
        'selected_movie_indices': []
    })

    # Set randomly generated refinement layout
    if "refinement_layout" in conf and conf["refinement_layout"]:
        print("Using configured refinement layout")
        refinement_layout_name = conf["refinement_layout"]
    else:
        print(f"Using random refinement layout")
        refinement_layout_name = np.random.choice(list(refinement_layouts.keys()))

    p = [1, 0, 2]    
    set_val("refinement_layout", refinement_layouts[refinement_layout_name])

    #TODO log those into elicitation-ended
    ### Initialize stuff related to alpha comparison (after metric assesment step) ###
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
    # Prepare mapping
    algorithm_name_mapping = {
        algorithm: algorithm_name for algorithm_name, algorithm in zip(algorithm_names, selected_algorithms)
    }

    set_mapping(get_uname(), {
        "alphas_iteration": 1,
        "alphas_p": [selected_alphas[:3], selected_alphas[3:]],
        "algorithm_family": selected_algorithm_family,
        "selected_algorithms": selected_algorithms,
        "selected_slider_versions": selected_slider_versions,
        "algorithm_name_mapping": algorithm_name_mapping,
        "recommendations": {
           algo: [] for algo in algorithm_name_mapping.keys() # For each algorithm and each iteration we hold the recommendation
        },
        "selected_items": {
           algo: [] for algo in algorithm_name_mapping.keys() # For each algorithm and each iteration we hold the selected items
        },
        "shown_items": {
            algo: [] for algo in algorithm_name_mapping.keys() # For each algorithm and each iteration we hold the IDs of recommended items (for quick filtering)
        },
        "slider_values": {
            "slider_relevance": [],
            "slider_exploitation_exploration": [],
            "slider_uniformity_diversity": [],
            "slider_popularity_novelty": []
        }
    })

    weights_with_list = {}
    weights_with_list["values"] = {key: val.astype(float) for key, val in weights["values"].items()}
    weights_with_list["vec"] = weights["vec"].tolist()
    elicitation_ended(
        session['elicitation_movies'],
        selected_movies,
        orig_permutation=p,
        displyed_name_mapping=displyed_name_mapping,
        refinement_layout=refinement_layout_name,
        supports={key: np.round(value.astype(float), 4).tolist() for key, value in supports.items()},
        alphas_p=[selected_alphas[:3], selected_alphas[3:]],
        algorithm_family=selected_algorithm_family,
        selected_algorithms=selected_algorithms,
        selected_slider_versions=selected_slider_versions,
        algorithm_name_mapping=algorithm_name_mapping,
        initial_weights=weights_with_list
    )  

    #return redirect(url_for("multiobjective.compare_and_refine"))
    return redirect(url_for("journal.metric_assesment"))

class RandomBoundedRecommendationGenertor:
    
    def __init__(self, k, lower_bound, upper_bound):
        self.k = k
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    
    def __iter__(self):
        return self
    
    def __next__(self):
        print("Next got called")
        yield 0
    
class RandomMaximizingRecommendationGenertor:
    
    def __init__(self, objective, k, items, max_iters = 100000):
        self.objective = objective
        self.k = k
        self.items = items
        self.max_iters = max_iters
    
    def __iter__(self):
        return self
    
    def __next__(self):
        max_obj_rec = None
        max_obj_val = None
        obj_vals = []
        for i in range(self.max_iters):
            top_k = np.random.choice(self.items, self.k, replace=False)
            obj_val = self.objective(top_k)
            obj_vals.append(obj_val)
            if max_obj_val is None or obj_val > max_obj_val:
                max_obj_val = obj_val
                max_obj_rec = top_k

        obj_vals = np.array(obj_vals)
        print(f"Min: {obj_vals.min()}, Max: {obj_vals.max()}, Mean: {obj_vals.mean()}, Percentiles: {np.percentile(obj_vals, [50, 70, 80, 90, 95, 99])}")
        return max_obj_val, max_obj_rec
    
# Greedily choose highest marginal gain items
class GreedyMaximizingRecommendationGenertor:
    
    def __init__(self, objective, k, items):
        self.objective = objective
        self.k = k
        self.items = items
    
    def __iter__(self):
        return self
    
    def __next__(self):
        max_obj_rec = None
        max_obj_val = None
        
        top_k = []
        for i in range(self.k):
            max_mgain_item = None
            max_mgain = None
            for item in self.items:
                if item in top_k:
                    continue
                mgain = self.objective(top_k + [item]) - self.objective(top_k)
                if max_mgain_item is None or mgain > max_mgain:
                    max_mgain = mgain
                    max_mgain_item = item
            top_k.append(max_mgain_item)
        return self.objective(top_k), top_k

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


# If all_items is false, we only look into 1000 items with highest relevance score
# do_normalize -> whether we normalize the diversity support
def get_diversified_top_k_lists(k, random_users, rel_scores, rel_scores_normed,
                                alpha, items, diversity_function, diversity_cdf,
                                rating_matrix,
                                n_items_subset=None, do_normalize=False, unit_normalize=False, rnd_mixture=False):
    ####TODO REMOVE SEED SETTING!!!
    random.seed(42)
    np.random.seed(42)
    ### TODO
    
    start_time = time.perf_counter()
    top_k_lists = np.zeros(shape=(random_users.size, k), dtype=np.int32)
    scores = np.zeros(shape=(items.size if n_items_subset is None else n_items_subset, ), dtype=np.float32)
    mgains = np.zeros(shape=(items.size if n_items_subset is None else n_items_subset, ), dtype=np.float32)

    assert rel_scores.shape == rel_scores_normed.shape
    
    # User normalized relevance scores
    if do_normalize:
        rel_scores = rel_scores_normed
    
    # Sort relevances
    sorted_relevances = np.argsort(-rel_scores, axis=-1)
   
    divs = []
   
    # Iterate over the random users sample
    for user_idx, random_user in enumerate(random_users):
        
        #print(f"User_idx = {user_idx}, user={random_user}")
        
        # If n_items_subset is specified, we take subset of items
        if n_items_subset is None:
            source_items = items
        else:
            if rnd_mixture:
                #print(f"Using random mixture")
                assert n_items_subset % 2 == 0, f"When using random mixture we expect n_items_subset ({n_items_subset}) to be divisible by 2"
                source_items = np.concatenate([sorted_relevances[user_idx, :n_items_subset//2], np.random.choice(sorted_relevances[user_idx, n_items_subset:], n_items_subset//2, replace=False)])
            else:
                source_items = sorted_relevances[user_idx, :n_items_subset]
            
        #print(f"Source items are: {source_items}")
        
        # Mask-out seen items by multiplying with zero
        # i.e. 1 is unseen
        # 0 is seen
        seen_items_mask = np.ones(shape=(random_users.size, source_items.size), dtype=np.int8)
        seen_items_mask[rating_matrix[np.ix_(random_users, source_items)] > 0.0] = 0
        
        #print(f"Seen items mask: {seen_items_mask}")
        
        # Build the recommendation incrementally
        for i in range(k):
            # Cache f_prev
            st = time.perf_counter()
            f_prev = diversity_function(top_k_lists[user_idx, :i])
            #print(f"\ti={i}, f_prev={f_prev}")
           
            # For every source item, try to add it and calculate its marginal gain
            st = time.perf_counter()
            for j, item in enumerate(source_items):
                top_k_lists[user_idx, i] = item # try extending the list
                mgains[j] = (diversity_function(top_k_lists[user_idx, :i+1]) - f_prev)
                
            # If we should normalize, use cdf_div to normalize marginal gains
            if do_normalize:
                # Reshape to N examples with single feature
                mgains = diversity_cdf.transform(mgains.reshape(-1, 1)).reshape(mgains.shape)
    
            # Calculate scores
            if unit_normalize:
                # If we do unit normalization, we multiply with coefficients that sum to 1
                scores = (1.0 - alpha) * rel_scores[user_idx, source_items] + alpha * mgains
            else:
                # Otherwise we just take relevance + diversity
                scores = rel_scores[user_idx, source_items] + alpha * mgains
            # assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
            # Ensure seen items get lowest score of 0
            # Just multiplying by zero does not work when scores are not normalized to be always positive
            # because masked-out items will not have smallest score (some valid, non-masked ones can be negative)
            # scores = scores * seen_items_mask[user_idx]
            # So instead we do scores = scores * seen_items_mask[user_idx] + NEG_INF * (1 - seen_items_mask[user_idx])
            min_score = scores.min()
            # Unlike in predict_with_score, here we do not mandate NEG_INF to be strictly smaller
            # because rel_scores may already contain some NEG_INF that was set by predict_with_score
            # called previously -> so we allow <=.
            assert NEG_INF <= min_score, f"min_score ({min_score}) is not smaller than NEG_INF ({NEG_INF})"
            scores = scores * seen_items_mask[user_idx] + NEG_INF * (1 - seen_items_mask[user_idx])

            # Get item with highest score
            # But beware that this is index inside "scores"
            # which has same size as subset of rnd_items
            # so we need to map item index to actual item later on
            best_item_idx = scores.argmax()
            best_item = source_items[best_item_idx]
            #print(f"Best item idx = {best_item_idx}, best item: {best_item}, best item score={scores[best_item_idx]}, mgain: {mgains[best_item_idx]}, rel: {rel_scores[user_idx, best_item_idx]}")
            
            # Select the best item and append it to the recommendation list            
            top_k_lists[user_idx, i] = best_item
            # Mask out the item so that we do not recommend it again
            seen_items_mask[user_idx, best_item_idx] = 0

    print(f"Diversification took: {time.perf_counter() - start_time}")
    return top_k_lists

@functools.lru_cache(maxsize=None)
def get_distance_matrix_cb(path):
    return np.load(path)

@functools.lru_cache(maxsize=None)
def load_cdf_cache(base_path, metric_name):
    with open(os.path.join(base_path, "cdf", f"{metric_name}.pckl"), "rb") as f:
        return pickle.load(f)

# Plugin specific version of enrich_results
# Here we are sure that we are inside this particular plugin
# thus we have a particular data loader and can use some of its internals
def enrich_results(top_k, loader, support=None):
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

# TODO remove
@bp.route("/reset", methods=["GET", "POST"])
def reset():
    set_val("iteration", 0)
    return redirect(url_for("journal.mors_feedback"))

@bp.route("/done", methods=["GET", "POST"])
def done():
    # #it = session['iteration']
    # #it += 1
    # #session.modified = True
    # print(f"Sess id={request.cookies['something']}")
    # # Start with zero, because at the very beginning, mors_feedback is called, not mors and that generates recommendations for first iteration, but at the same time, increases the iteration
    # set_val("iteration", 0)

    # return f"DONE, it={get_val('iteration')}"
    return redirect(url_for("journal.final_questionnaire"))

#####    Algorithms     #####
# TODO move algorithms to shared common

# Some common abstraction
# Runs diversification on a relevance based recommendation
# w.r.t. algorithm passed as algo
def morsify(k, rel_scores,
            algo, items, objective_fs,
            rating_row, filter_out_items, n_items_subset=None,
            do_normalize=False, rnd_mixture=False):
    ####TODO REMOVE SEED SETTING!!!
    random.seed(42)
    np.random.seed(42)
    ### TODO
    
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


    # Mask-out seen items by multiplying with zero
    # i.e. 1 is unseen
    # 0 is seen
    # Lets first set zeros everywhere
    seen_items_mask = np.zeros(shape=(source_items.size, ), dtype=np.int8)
    # And only put 1 to UNSEEN items in CANDIDATE (source_items) list
    seen_items_mask[rating_row[source_items] <= 0.0] = 1
    print(f"### Unseen: {seen_items_mask.sum()} out of: {seen_items_mask.size}")
    
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
                mgains[objective_index] = QuantileTransformer().fit_transform(mgains[objective_index].reshape(-1, 1)).reshape(mgains[objective_index].shape)
    
        # Calculate scores
        print(f"@@ Mgains shape: {mgains.shape}, seen_items_mask shape: {seen_items_mask}")
        best_item_idx = algo(mgains, seen_items_mask)
        best_item = source_items[best_item_idx]
            
        # Select the best item and append it to the recommendation list            
        top_k_list[i] = best_item
        # Mask out the item so that we do not recommend it again
        seen_items_mask[best_item_idx] = 0

        if i == 9:
            print(f"Rel scores shape: {rel_scores.shape}")
            print(f"### best item = {best_item}, mgains: {mgains[:, best_item_idx]}, rel: {rel_scores[best_item]}")

    print(f"Diversification took: {time.perf_counter() - start_time}")
    return top_k_list



# Does not get throough morsify as this is end-to-end approach to MORS, not "diversification" or "morsification"
def moea():
    pass

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
    if selected_metric_name == "CF-ILD":
        div_f = intra_list_diversity(loader.distance_matrix)
    elif selected_metric_name == "CB-ILD":
        distance_matrix_cb = get_distance_matrix_cb(os.path.join(get_cache_path(get_semi_local_cache_name(loader)), "distance_matrix_text.npy"))
        div_f = intra_list_diversity(distance_matrix_cb)
    elif selected_metric_name == "BIN-DIV":
        div_f = binomial_diversity(loader.all_categories, loader.get_item_index_categories, loader.rating_matrix, 0.0, loader.name())


    #cdf_div = load_cdf_cache(get_cache_path(get_semi_local_cache_name(loader)), selected_metric_name)

    k = 10

    # Generate recommendation for relevance only EASE
    ease = EASER_pretrained(items)
    ease = ease.load(os.path.join(get_cache_path(get_semi_local_cache_name(loader)), "item_item.npy"))
    elicitation_selected = np.array(user_data['elicitation_selected_movies'])
    # Train the algorithm on all movies selected during preference elicitation
    # and all movies previously selected during recommendations made by current algorithm
    training_selections = np.concatenate([elicitation_selected] + [np.array(x, dtype=np.int32) for x in selected_items_history[cur_algorithm]], dtype=np.int32)
    print(f"Training selections: {training_selections}, dtype={training_selections.dtype}, elicitation selected: {elicitation_selected.dtype}, history: {selected_items_history[cur_algorithm]}")
    print(f"Shown items so far: {user_data['shown_items']}")
    print(f"Shown items from current algorithm so far: {user_data['shown_items'][cur_algorithm]}")
    all_recommendations = sum(user_data["shown_items"][cur_algorithm], [])
    rel_scores, user_vector, relevance_top_k = ease.predict_with_score(training_selections, all_recommendations, k)
    cdf_rel = load_cdf_cache(get_cache_path(get_semi_local_cache_name(loader)), "REL")
    rel_scores_normed = cdf_rel.transform(rel_scores.reshape(-1, 1)).reshape(rel_scores.shape)
    print(f"Rel scores normed: {rel_scores_normed}")
    print(f"@@ Items shape={items.shape}, size={items.size}")
    

    # Note that "all_recommendations" already includes items selected during the study (except those selected during elicitattion which we are thus adding)
    filter_out_items = np.concatenate([all_recommendations, elicitation_selected])

    def relevance_f(top_k_list):
        return rel_scores[top_k_list].sum()

    uniformity_f = lambda x: -1 * div_f(x)
    
    popularity_f = item_popularity(loader.rating_matrix)
    novelty_f = popularity_based_novelty(loader.rating_matrix)
    exploration_f = exploration(user_vector, loader.distance_matrix)
    exploitation_f = exploitation(user_vector, 1.0 - loader.distance_matrix)

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

    start_time = time.perf_counter()
    if cur_algorithm == "GREEDY-EXACT-1W":
        algo = greedy_exact(target_weights)
        top_k = morsify(
            10, rel_scores, algo,
            items, objectives, user_vector, filter_out_items, n_items_subset=500,
            do_normalize=True, rnd_mixture=True
        )
    elif cur_algorithm == "ITEM-WISE-EXACT-1W":
        algo = item_wise_exact(target_weights)
        top_k = morsify(
            10, rel_scores, algo,
            items, objectives, user_vector, filter_out_items, n_items_subset=500,
            do_normalize=True, rnd_mixture=True
        )
    elif cur_algorithm == "EVOLUTIONARY-EXACT-1W":
        objectives[0] = ObjectiveWrapper(lambda x: rel_scores_normed[x].sum(), "relevance-normed")
        algo = evolutionary_exact(target_weights, rel_scores, user_vector, objectives, np.array(relevance_top_k), filter_out_items, k=k, time_limit_seconds=4)
        top_k = algo()
    elif cur_algorithm == "GREEDY-MAX-1W":
        algo = greedy_max(target_weights)
        top_k = morsify(
            10, rel_scores, algo,
            items, objectives, user_vector, filter_out_items, n_items_subset=500,
            do_normalize=True, rnd_mixture=True
        )
    elif cur_algorithm == "GREEDY-MAX-2W":
        algo = greedy_max(target_weights)
        top_k = morsify(
            10, rel_scores, algo,
            items, objectives, user_vector, filter_out_items, n_items_subset=500,
            do_normalize=True, rnd_mixture=True
        )
    elif cur_algorithm == "ITEM-WISE-MAX-1W":
        algo = item_wise_max(target_weights)
        top_k = morsify(
            10, rel_scores, algo,
            items, objectives, user_vector, filter_out_items, n_items_subset=500,
            do_normalize=True, rnd_mixture=True
        )
    elif cur_algorithm == "ITEM-WISE-MAX-2W":
        algo = item_wise_max(target_weights)
        top_k = morsify(
            10, rel_scores, algo,
            items, objectives, user_vector, filter_out_items, n_items_subset=500,
            do_normalize=True, rnd_mixture=True
        )
    elif cur_algorithm == "EVOLUTIONARY-MAX-1W":
        objectives[0] = ObjectiveWrapper(lambda x: rel_scores_normed[x].sum(), "relevance-normed")
        algo = evolutionary_max(target_weights, rel_scores, user_vector, objectives, np.array(relevance_top_k), filter_out_items, k=k, time_limit_seconds=4)
        top_k = algo()
    elif cur_algorithm == "EVOLUTIONARY-MAX-2W":
        objectives[0] = ObjectiveWrapper(lambda x: rel_scores_normed[x].sum(), "relevance-normed")
        algo = evolutionary_max(target_weights, rel_scores, user_vector, objectives, np.array(relevance_top_k), filter_out_items, k=k, time_limit_seconds=4)
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
    top_k_list = top_k if isinstance(top_k, list) else top_k.tolist()
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
        }
    }
    log_interaction(session["participation_id"], "mors-recommendation-started", **data)
    return flask.Flask.redirect(flask.current_app, url_for("journal.mors"))

@bp.route("/mors", methods=["GET", "POST"])
def mors():
    #it = session['iteration']
    user_data = get_all(get_uname())
    it = user_data["iteration"]

    print(f"MORS IT={it}, mod={int(it) - 1 % N_ITERATIONS}")

    cur_block = (int(it) - 1) // N_ITERATIONS
    cur_algorithm = user_data["selected_algorithms"][cur_block]
    cur_slider_version = user_data["selected_slider_versions"][cur_block]

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

    # if cur_algorithm == "RLPROP":
    #     pass
    # elif cur_algorithm == "WA":
    #     pass
    # elif cur_algorithm == "MOEA-RS":
    #     pass
    # else:
    #     assert False, f"Unknown algorithm: {cur_algorithm} for it={it}"

    print(f"Algorithm = {cur_algorithm}")

    conf = load_user_study_config(session['user_study_id'])
    # if it >= N_ITERATIONS * len(ALGORITHMS):
    #     continuation_url = url_for("journal.done")
    # else:
    #     continuation_url = url_for("journal.mors_feedback")

    # We are at the end of block
    if it > 0 and it % N_ITERATIONS == 0:
        continuation_url = url_for("journal.block_questionnaire")
    else:
        continuation_url = url_for("journal.mors_feedback")

    tr = get_tr(languages, get_lang())
    # TODO replace random with user_data['movies'] which in turn should be filled in by recommendation algorithms
    params = {
        "continuation_url": continuation_url,
        "iteration": it,
        "n_iterations": N_ITERATIONS * N_BLOCKS,
        "movies": get_val('recommendation'),
        "like_nothing": tr("compare_like_nothing"),
        "can_fine_tune": cur_algorithm != "RELEVANCE-BASED",
        "slider_version": cur_slider_version,
        "slider_relevance": slider_relevance,
        "slider_uniformity_diversity": slider_uniformity_diversity,
        "slider_popularity_novelty": slider_popularity_novelty,
        "slider_exploitation_exploration": slider_exploitation_exploration,
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

    #assert session.modified == False
    return render_template("mors.html", **params)

# TODO remove, temporary endpoint
@bp.route("/greedy", methods=["GET", "POST"])
def gr():
    conf = load_user_study_config(session['user_study_id'])
    
    # Get a loader
    loader_factory = load_data_loaders()[conf["selected_data_loader"]]
    loader = loader_factory(**filter_params(conf["data_loader_parameters"], loader_factory))
    loader = load_data_loader_cached(loader, session['user_study_guid'], loader_factory.name(), get_semi_local_cache_name(loader))

    potter_movies = np.array([3161, 3948, 4882, 5524, 6452, 7193, 7538])
    items = np.arange(loader.rating_matrix.shape[1])
    algo = EASER_pretrained(items)
    algo = algo.load(os.path.join(get_cache_path(get_semi_local_cache_name(loader)), "item_item.npy"))
    rel_scores, user_vector, ease_pred = algo.predict_with_score(potter_movies, potter_movies, k=10)
    ease_baseline = enrich_results(ease_pred, loader)
    
    cdf_rel = load_cdf_cache(get_cache_path(get_semi_local_cache_name(loader)), "REL")
    #rel_scores_normed = cdf_rel.transform(rel_scores.reshape(-1, 1)).reshape(rel_scores.shape)

    #diversity_f = intra_list_diversity(loader.distance_matrix)
    diversity_f = binomial_diversity(loader.all_categories, loader.get_item_index_categories, loader.rating_matrix, 0.0, loader.name())
    #cdf_diversity = load_cdf_cache(get_cache_path(get_semi_local_cache_name(loader)), "CF-ILD")
    
    uniformity_f = intra_list_diversity(1.0 - loader.distance_matrix)
    #cdf_uniformity = load_cdf_cache

    popularity_f = item_popularity(loader.rating_matrix)
    novelty_f = popularity_based_novelty(loader.rating_matrix)
    exploration_f = exploration(user_vector, loader.distance_matrix)
    exploitation_f = exploitation(user_vector, 1.0 - loader.distance_matrix)


    def relevance_f(top_k_list):
        return rel_scores[top_k_list].sum()

    objectives = [
        relevance_f,

        diversity_f,
        uniformity_f,

        popularity_f,
        novelty_f,

        exploration_f,
        exploitation_f
    ]

    start_time = time.perf_counter()
    algo = item_wise_exact(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]))
    greedy_top_k = morsify(
        10, rel_scores, algo,
        items, objectives, user_vector, n_items_subset=500,
        do_normalize=True, rnd_mixture=True
    )
    print(f"Greedy morsify took: {time.perf_counter() - start_time}")

    print(f"@@@ Greedy top k = {greedy_top_k}")


    params = {
        "movies": {
            "EASE": {
                "movies": ease_baseline,
                "order": 0
            },
            "Greedy": {
                "movies": enrich_results(greedy_top_k, loader),
                "order": 1
            }
        }
    }
    return render_template("metric_assesment.html", **params)

# Called as continuation of compare-alphas, redirects for compare-alphas (next step)
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
        mapping = get_val("metric_assesment_list_to_diversity")
        selected_metric_name = mapping[selected_metric_name]

        #selected_metric_index = request.form.get("selected_metric_index")
        set_val('selected_metric_name', selected_metric_name)
        # Mark end of metric assesment here
        log_interaction(session["participation_id"], "metric-assesment-ended", selected_metric_name=selected_metric_name)
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
        current_alphas[0]: "LIST 1",
        current_alphas[1]: "LIST 2",
        current_alphas[2]: "LIST 3"
    }

    if selected_metric_name == "CF-ILD":
        div_f = intra_list_diversity(loader.distance_matrix)
    elif selected_metric_name == "CB-ILD":
        distance_matrix_cb = get_distance_matrix_cb(os.path.join(get_cache_path(get_semi_local_cache_name(loader)), "distance_matrix_text.npy"))
        div_f = intra_list_diversity(distance_matrix_cb)
    elif selected_metric_name == "BIN-DIV":
        div_f = binomial_diversity(loader.all_categories, loader.get_item_index_categories, loader.rating_matrix, 0.0, loader.name())
    else:
        assert False

    params["movies"] = {}

    k = 8 # We use k=8 instead of k=10 so that items fit to screen easily
    items = np.arange(loader.rating_matrix.shape[1])
    algo = EASER_pretrained(items)
    algo = algo.load(os.path.join(get_cache_path(get_semi_local_cache_name(loader)), "item_item.npy"))
    elicitation_selected = np.array(user_data['elicitation_selected_movies'])
    rel_scores, user_vector, _ = algo.predict_with_score(elicitation_selected, elicitation_selected, k)
    rel_scores = rel_scores[np.newaxis, :]
    cdf_rel = load_cdf_cache(get_cache_path(get_semi_local_cache_name(loader)), "REL")
    rel_scores_normed = cdf_rel.transform(rel_scores.reshape(-1, 1)).reshape(rel_scores.shape)
    user_vector = user_vector[np.newaxis, :]


    cdf_div = load_cdf_cache(get_cache_path(get_semi_local_cache_name(loader)), selected_metric_name)
    rec_lists = dict()
    for alpha_order, alpha in enumerate(current_alphas):

        rec_list = get_diversified_top_k_lists(k, np.array([0]), rel_scores, rel_scores_normed,
                                        alpha=alpha, items=items, diversity_function=div_f, diversity_cdf=cdf_div,
                                        rating_matrix=user_vector,
                                        n_items_subset=500, do_normalize=True, unit_normalize=True, rnd_mixture=True)

        rec_list = enrich_results(rec_list[0], loader)
        rec_lists[algorithm_name_mapping[alpha]] = rec_list[0]

        params["movies"][algorithm_name_mapping[alpha]] = {
            "movies": rec_list,
            "order": str(alpha_order)
        }

    #session['alpha_movies'] = params
    set_mapping(get_uname() + ":alpha_movies", {
        "movies": params["movies"]
    })

    # Mark start of the first iteration for compare-alphas
    log_interaction(session["participation_id"], "compare-alphas-started",
                    alphas=current_alphas,
                    iteration=cur_iter,
                    algorithm_name_mapping=algorithm_name_mapping,
                    selected_metric_name=selected_metric_name,
                    elicitation_selected=user_data['elicitation_selected_movies'])

    cur_iter += 1
    #session['alphas_iteration'] = cur_iter
    set_val('alphas_iteration', cur_iter)

    return redirect(url_for("journal.compare_alphas", continuation_url=continuation_url))

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
    params["n_iterations"] = 1 + N_ALPHA_ITERS # We have 1 iteration for actual metric assesment followed bz N_ALPHA_ITERS iterations for comparing alphas
    params["iteration"] = get_val("alphas_iteration")
    return render_template("compare_alphas.html", **params)

@bp.route("/metric-assesment", methods=["GET"])
def metric_assesment():
    start_time = time.perf_counter()
    conf = load_user_study_config(session['user_study_id'])

    # Get a loader
    loader_factory = load_data_loaders()[conf["selected_data_loader"]]
    loader = loader_factory(**filter_params(conf["data_loader_parameters"], loader_factory))
    loader = load_data_loader_cached(loader, session['user_study_guid'], loader_factory.name(), get_semi_local_cache_name(loader))

    # TODO get TRUE CB
    k = 8 # We use k=8 instead of k=10 so that items fit to screen easily
    items = np.arange(loader.rating_matrix.shape[1])

    algo = EASER_pretrained(items)
    algo = algo.load(os.path.join(get_cache_path(get_semi_local_cache_name(loader)), "item_item.npy"))
    print(f"Took 2+:", time.perf_counter() - start_time)
    #print(f"@@ {loader.get_item_id(270)}, {loader.get_item_index(333)}, {loader.get_item_index_description(270)}, {loader.get_item_id_description(333)}")
    # Movie indices of selected movies
    elicitation_selected = np.array(get_val("elicitation_selected_movies"))
    rel_scores, user_vector, ease_pred = algo.predict_with_score(elicitation_selected, elicitation_selected, k)
    rel_scores = rel_scores[np.newaxis, :]
    cdf_rel = load_cdf_cache(get_cache_path(get_semi_local_cache_name(loader)), "REL")
    rel_scores_normed = cdf_rel.transform(rel_scores.reshape(-1, 1)).reshape(rel_scores.shape)
    user_vector = user_vector[np.newaxis, :]
    
    print(f"Elicitation selected: {elicitation_selected}")
    print(f"Ease pred: {ease_pred}")
    print(f"Rel scores: {rel_scores[0, ease_pred]}")

    # print(f"## Elicitation selected movies = {elicitation_selected}")
    # user_row = np.zeros(shape=(1, items.size), dtype=np.int32)
    # if elicitation_selected.size > 0:
    #     user_row[0, elicitation_selected] = 1
    # rel_scores = np.dot(user_row, algo.item_item)
    # rel_scores_normed = rel_scores
    print(f"Took 2++:", time.perf_counter() - start_time)
    distance_matrix_cb = get_distance_matrix_cb(os.path.join(get_cache_path(get_semi_local_cache_name(loader)), "distance_matrix_text.npy"))
    print(f"Distance matrix CB loading took: {time.perf_counter() - start_time}")

    cb_ild = intra_list_diversity(distance_matrix_cb)
    cf_ild = intra_list_diversity(loader.distance_matrix)
    bin_div = binomial_diversity(loader.all_categories, loader.get_item_index_categories, loader.rating_matrix, 0.0, loader.name())

    print(f"Took 3:", time.perf_counter() - start_time)

    try:
        alpha = float(request.args.get("alpha"))
    except:
        alpha = 1.0
    # r1 = enrich_results(next(rnd_cb_ild)[1], loader)
    # r2 = enrich_results(next(rnd_cf_ild)[1], loader)
    # r3 = enrich_results(next(rnd_bin_div)[1], loader)
    ease_baseline = enrich_results(ease_pred, loader)
    print(f"Predicting r1 took: {time.perf_counter() - start_time}")
    cdf_cf_div = load_cdf_cache(get_cache_path(get_semi_local_cache_name(loader)), "CF-ILD")
    r2 = get_diversified_top_k_lists(k, np.array([0]), rel_scores, rel_scores_normed,
                                alpha=alpha, items=items, diversity_function=cf_ild, diversity_cdf=cdf_cf_div,
                                rating_matrix=user_vector,
                                n_items_subset=500, do_normalize=True, unit_normalize=True, rnd_mixture=True)
    r2 = enrich_results(r2[0], loader)
    print(f"Predicting r2 took: {time.perf_counter() - start_time}")
    cdf_cb_div = load_cdf_cache(get_cache_path(get_semi_local_cache_name(loader)), "CB-ILD")
    r3 = get_diversified_top_k_lists(k, np.array([0]), rel_scores, rel_scores_normed,
                                alpha=alpha, items=items, diversity_function=cb_ild, diversity_cdf=cdf_cb_div,
                                rating_matrix=user_vector,
                                n_items_subset=500, do_normalize=True, unit_normalize=True, rnd_mixture=True)
    r3 = enrich_results(r3[0], loader)
    print(f"Predicting r3 took: {time.perf_counter() - start_time}")
    cdf_bin_div = load_cdf_cache(get_cache_path(get_semi_local_cache_name(loader)), "BIN-DIV")
    r4 = get_diversified_top_k_lists(k, np.array([0]), rel_scores, rel_scores_normed,
                                alpha=alpha, items=items, diversity_function=bin_div, diversity_cdf=cdf_bin_div,
                                rating_matrix=user_vector,
                                n_items_subset=500, do_normalize=True, unit_normalize=True, rnd_mixture=True)
    r4 = enrich_results(r4[0], loader)
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
            #     "order": 0
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

    data = {
        "list_permutation": lists,
        "algorithm_name_mapping": algorithm_name_mapping,
        "list_name_to_rec": list_name_to_rec
    }
    log_interaction(session["participation_id"], "metric-assesment-started", **data)

    # We need to store inverse mapping
    # in the form of "list name" : "diversity name"
    set_val("metric_assesment_list_to_diversity", { list_name : diversity_name for diversity_name, list_name in algorithm_name_mapping.items() })

    print(f"Took 4:", time.perf_counter() - start_time)
    tr = get_tr(languages, get_lang())
    params["contacts"] = tr("footer_contacts")
    params["contact"] = tr("footer_contact")
    params["charles_university"] = tr("footer_charles_university")
    params["cagliari_university"] = tr("footer_cagliari_university")
    params["t1"] = tr("footer_t1")
    params["t2"] = tr("footer_t2")
    params["title"] = tr("journal_metric_assesment_title")
    params["header"] = tr("journal_metric_assesment_header")
    params["hint"] = tr("journal_metric_assesment_hint")
    params["continuation_url"] = request.args.get("continuation_url")
    params["finish"] = tr("metric_assesment_finish")
    params["iteration"] = 1
    params["n_iterations"] = 1 + N_ALPHA_ITERS # We have 1 iteration for actual metric assesment followed bz N_ALPHA_ITERS iterations for comparing alphas
    print(f"Took 5:", time.perf_counter() - start_time)
    # Handle overrides
    params["footer_override"] = None
    if "text_overrides" in conf:
        if "footer" in conf["text_overrides"]:
            params["footer_override"] = conf["text_overrides"]["footer"]
    print(f"Took 6:", time.perf_counter() - start_time)
    params["alpha"] = alpha
    return render_template("metric_assesment.html", **params)

# from plugins.layoutshuffling import long_initialization

@bp.route("/initialize", methods=["GET"])
def initialize():
    guid = request.args.get("guid")
    # p = Process(
    #     target=long_initialization,
    #     daemon=True,
    #     args=(guid, )
    # )
    # p.start()
    # print("Going to redirect back")

    #return redirect(request.args.get("continuation_url"))
    cont_url = request.args.get("continuation_url")
    return redirect(url_for("fastcompare.initialize", guid=guid, continuation_url=cont_url, consuming_plugin=__plugin_name__))

def register():
    return {
        "bep": dict(blueprint=bp, prefix=None),
    }

@bp.route("/get-block-questions", methods=["GET"])
def get_block_questions():
   
    user_data = get_all(get_uname())
    it = user_data["iteration"]
    cur_block = (int(it) - 1) // N_ITERATIONS
    cur_algorithm = user_data["selected_algorithms"][cur_block]
    print(f"IT={it}, cur_block={cur_block}, cur_algorithm={cur_algorithm}")

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
            "text": f"The recommended {item_text} were mostly different from what I usually watch.",
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
            "text": "Appropriate values of the objective criteria were easy to set.",
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



# If exact = True, then we are in "exact" algorithm world as apposed to "max", this also means that we do not normalize weights to sum to 1
# selected_movies -> movies selected during elicitation
# elicitation_movies -> movies displayed during elicitation
# Note that we DO NOT NORMALIZE the WEIGHTS here
def calculate_weight_estimate_generic(loader, objectives, selected_movies, elicitation_movies, return_supports=False):
    
    n_objectives = len(objectives)
    if not selected_movies:
        x = {
            "vec": np.array([1.0] * n_objectives, dtype=np.float32),
            "values": {
                obj.name(): 1.0 for obj in objectives
            }
        }
        if return_supports:
            return x, {}
        else:
            return x

    if type(elicitation_movies[0]) == int:
        movie_indices = np.unique(elicitation_movies)
    elif type(elicitation_movies[0]) == dict:
        movie_indices = np.unique([int(movie["movie_idx"]) for movie in elicitation_movies])
    else:
        assert False

    # We treat movie_indices as one long recommendation list and try to reconstruct it, calculating
    # intermediate marginal gains for all the items
    # We then estimate the gains from the weights
    # We opt for optimization by doing the expansion (incrementally adding) s
    mgains = np.zeros(shape=(len(objectives), len(movie_indices)), dtype=np.float32)
    selected_mgains = [[] for _ in objectives]
    # As the users are selecting the movies during elicitation, we can treat this as incrementally
    # growing user profile (vector of selections)
    user_vector_incremental = []
    objective_index_to_name = []
    for obj_idx, obj in enumerate(objectives):
        objective_index_to_name.append(obj.name())
        for idx, movie_idx in enumerate(movie_indices):
            if movie_idx in selected_movies:
                # We will end up in this place exactly len(objectives)-times, so we only append for the first time
                if obj_idx == 0:
                    user_vector_incremental.append(movie_idx)
                # Since we added new item, we have to take updated user vector into an account
                mgains[obj_idx, idx] = obj(movie_indices[:idx+1], user_vector_incremental) - obj(movie_indices[:idx], user_vector_incremental[:-1])
                selected_mgains[obj_idx].append(mgains[obj_idx, idx])
            else:
                # We did not select anything, so user vector stays intact
                mgains[obj_idx, idx] = obj(movie_indices[:idx+1], user_vector_incremental) - obj(movie_indices[:idx], user_vector_incremental)

    obj_cdf = QuantileTransformer().fit(mgains.T)
    selected_mgains = np.array(selected_mgains)
    if selected_mgains.size == 0:
        x = {
            "vec": np.array([1.0] * n_objectives, dtype=np.float32),
            "values": {
                obj.name(): 1.0 for obj in objectives
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
            "mgains": mgains,
            "rating_matrix_shape": np.array(loader.rating_matrix.shape)
        }
        return result, supports

    return result