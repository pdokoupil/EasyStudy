from abc import ABC
import functools
import random
import sys
import os
import time



[sys.path.append(i) for i in ['.', '..']]
[sys.path.append(i) for i in ['../.', '../..', '../../.']]

from models import UserStudy

from multiprocessing import Process
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import numpy as np

from common import get_tr, load_languages, multi_lang, load_user_study_config
from flask import Blueprint, request, redirect, render_template, url_for, session

from plugins.utils.preference_elicitation import recommend_2_3, rlprop, weighted_average, get_objective_importance, prepare_tf_model, calculate_weight_estimate, load_ml_dataset, enrich_results
from plugins.utils.interaction_logging import log_interaction, log_message
from plugins.fastcompare.loading import load_data_loaders
from plugins.fastcompare import elicitation_ended, filter_params, iteration_started, iteration_ended, load_data_loader, load_data_loader_cached
#from plugins.multiobjective import prepare_recommendations
from plugins.fastcompare import prepare_recommendations, get_semi_local_cache_name, get_cache_path
from plugins.journal.metrics import binomial_diversity, intra_list_diversity

languages = load_languages(os.path.dirname(__file__))

N_ITERATIONS = 6 # 6 iterations
HIDE_LAST_K = 100000
NEG_INF = int(-10e6)

# Implementation of this function can differ among plugins
def get_lang():
    default_lang = "en"
    if "lang" in session and session["lang"] and session["lang"] in languages:
        return session["lang"]
    return default_lang

__plugin_name__ = "journal"
__version__ = "0.1.0"
__author__ = "Patrik Dokoupil"
__author_contact__ = "Patrik.Dokoupil@matfyz.cuni.cz"
__description__ = "Complex plugin for a very customized user study that we have used for journal paper"

bp = Blueprint(__plugin_name__, __plugin_name__, url_prefix=f"/{__plugin_name__}")

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
    params["about_placeholder"] = tr("moo_create_about_placeholder")
    params["override_informed_consent"] = tr("moo_create_override_informed_consent")
    params["override_about"] = tr("moo_create_override_about")
    params["show_final_statistics"] = tr("moo_create_show_final_statistics")
    params["override_algorithm_comparison_hint"] = tr("moo_create_override_algorithm_comparison_hint")
    params["algorithm_comparison_placeholder"] = tr("moo_create_algorithm_comparison_placeholder")
    params["informed_consent_placeholder"] = tr("moo_create_informed_consent_placeholder")

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
    return redirect(url_for("utils.preference_elicitation",
            continuation_url=url_for("journal.send_feedback"),
            consuming_plugin=__plugin_name__,
            initial_data_url=url_for('fastcompare.get_initial_data'),
            search_item_url=url_for('fastcompare.item_search')
        )
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

# Receives arbitrary feedback (typically from preference elicitation) and generates recommendation
@bp.route("/send-feedback", methods=["GET"])
def send_feedback():
    # We read k from configuration of the particular user study
    conf = load_user_study_config(session["user_study_id"])
    k = conf["k"]
    session["rec_k"] = k

    # Get a loader
    loader_factory = load_data_loaders()[conf["selected_data_loader"]]
    loader = loader_factory(**filter_params(conf["data_loader_parameters"], loader_factory))
    loader = load_data_loader_cached(loader, session["user_study_guid"], loader_factory.name(), get_semi_local_cache_name(loader))

    # Movie indices of selected movies
    selected_movies = request.args.get("selectedMovies")
    selected_movies = selected_movies.split(",") if selected_movies else []
    selected_movies = [int(m) for m in selected_movies]

    # Calculate weights based on selection and shown movies during preference elicitation
    weights, supports = calculate_weight_estimate(loader, selected_movies, session["elicitation_movies"], return_supports=True)
    weights /= weights.sum()
    weights = weights.tolist()
    weights = {
        algo_displayed_name: weights for algo_displayed_name in displyed_name_mapping.values()
    }
    session["weights"] = weights
    print(f"Weights initialized to {weights}")


    # Add default entries so that even the non-chosen algorithm has an empty entry
    # to unify later access
    recommendations = {
        x["displayed_name"]: [[]] for x in conf["algorithm_parameters"]
    }
    initial_weights_recommendation = {
        x["displayed_name"]: [[]] for x in conf["algorithm_parameters"]
    }
    
    # We filter out everything the user has selected during preference elicitation.
    # However, we allow future recommendation of SHOWN, NOT SELECTED (during elicitation, not comparison) altough these are quite rare
    filter_out_movies = selected_movies

    prepare_recommendations(loader, conf, recommendations, selected_movies, filter_out_movies, k)
    #prepare_recommendations(weights, recommendations, initial_weights_recommendation, selected_movies, filter_out_movies, k)
    print(f"Recommendations={recommendations}")

    slider_state = {
        algo_displayed_name: {} for algo_displayed_name in displyed_name_mapping.values()
    }
    
    session["movies"] = recommendations
    session["initial_weights_recommendation"] = initial_weights_recommendation
    session["iteration"] = 1
    session["elicitation_selected_movies"] = selected_movies
    session["selected_movie_indices"] = [] #dict() # For each iteration, we can store selected movies
    session["selected_variants"] = []
    session["nothing"] = []
    session["cmp"] = []
    session["a_r"] = []
    session["slider_state"] = slider_state

    # Build permutation
    p = []
    for i in range(N_ITERATIONS):
        orders = dict()
        available_orders = [0, 1] # We compare two algorithms
        for j, algorithm_displayed_name in enumerate(conf["selected_algorithms"]):
            if algorithm_displayed_name == displyed_name_mapping["weighted_average"]: # Weighted average is not displayed
                continue
            order_idx = np.random.randint(len(available_orders))
            orders[algorithm_displayed_name] = available_orders[order_idx]
            del available_orders[order_idx]

        p.append(orders)

    session["permutation"] = p
    session["orig_permutation"] = p
    session["algorithms_to_show"] = [displyed_name_mapping["rlprop"]] * N_ITERATIONS

    # Set randomly generated refinement layout
    if "refinement_layout" in conf and conf["refinement_layout"]:
        print("Using configured refinement layout")
        refinement_layout_name = conf["refinement_layout"]
        session["refinement_layout"] = refinement_layouts[refinement_layout_name]
    else:
        print(f"Using random refinement layout")
        refinement_layout_name = np.random.choice(list(refinement_layouts.keys()))
        session["refinement_layout"] =  refinement_layouts[refinement_layout_name]
    
    elicitation_ended(
        session["elicitation_movies"], session["elicitation_selected_movies"],
        orig_permutation=p, displyed_name_mapping=displyed_name_mapping, refinement_layout=refinement_layout_name,
        supports={key: np.round(value.astype(float), 4).tolist() for key, value in supports.items()}
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

    @functools.lru_cache(maxsize=None)
    def load(self, path):
        self.item_item = np.load(path)
        assert self.item_item.shape[0] == self.item_item.shape[1] == self.all_items.size

    def predict_with_score(self, selected_items, filter_out_items, k):
        candidates = np.setdiff1d(self.all_items, selected_items)
        candidates = np.setdiff1d(candidates, filter_out_items)
        if selected_items.size == 0:
            return np.random.choice(candidates, size=k, replace=False).tolist()
        user_vector = np.zeros(shape=(self.all_items.size,), dtype=self.item_item.dtype)
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

@bp.route("/metric-assesment", methods=["GET"])
def metric_assesment():
    start_time = time.perf_counter()
    conf = load_user_study_config(session["user_study_id"])
    print(f"Took 1:", time.perf_counter() - start_time)

    # Get a loader
    loader_factory = load_data_loaders()[conf["selected_data_loader"]]
    print(f"Took 1+:", time.perf_counter() - start_time)
    loader = loader_factory(**filter_params(conf["data_loader_parameters"], loader_factory))
    print(f"Took 1++:", time.perf_counter() - start_time)
    loader = load_data_loader_cached(loader, session["user_study_guid"], loader_factory.name(), get_semi_local_cache_name(loader))
    print(f"Took 2:", time.perf_counter() - start_time)

    # TODO get TRUE CB
    k = 8 # We use k=8 instead of k=10 so that items fit to screen easily
    items = np.array(list(loader.item_index_to_id.keys()))

    algo = EASER_pretrained(items)
    algo.load(os.path.join(get_cache_path(get_semi_local_cache_name(loader)), "item_item.npy"))
    print(f"Took 2+:", time.perf_counter() - start_time)
    print(f"@@ {loader.get_item_id(270)}, {loader.get_item_index(333)}, {loader.get_item_index_description(270)}, {loader.get_item_id_description(333)}")
    # Movie indices of selected movies
    elicitation_selected = np.array(session["elicitation_selected_movies"])
    rel_scores, user_vector, ease_pred = algo.predict_with_score(elicitation_selected, elicitation_selected, k)
    rel_scores = rel_scores[np.newaxis, :]
    rel_scores_normed = rel_scores
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
    r2 = get_diversified_top_k_lists(k, np.array([0]), rel_scores, rel_scores_normed,
                                alpha=alpha, items=items, diversity_function=cf_ild, diversity_cdf=None,
                                rating_matrix=user_vector,
                                n_items_subset=500, do_normalize=False, unit_normalize=True, rnd_mixture=True)
    r2 = enrich_results(r2[0], loader)
    print(f"Predicting r2 took: {time.perf_counter() - start_time}")
    r3 = get_diversified_top_k_lists(k, np.array([0]), rel_scores, rel_scores_normed,
                                alpha=alpha, items=items, diversity_function=cb_ild, diversity_cdf=None,
                                rating_matrix=user_vector,
                                n_items_subset=500, do_normalize=False, unit_normalize=True, rnd_mixture=True)
    r3 = enrich_results(r3[0], loader)
    print(f"Predicting r3 took: {time.perf_counter() - start_time}")
    r4 = get_diversified_top_k_lists(k, np.array([0]), rel_scores, rel_scores_normed,
                                alpha=alpha, items=items, diversity_function=bin_div, diversity_cdf=None,
                                rating_matrix=user_vector,
                                n_items_subset=500, do_normalize=False, unit_normalize=True, rnd_mixture=True)
    r4 = enrich_results(r4[0], loader)
    print(f"Predicting r4 took: {time.perf_counter() - start_time}")

    params = {
        "movies": {
            "EASE": {
                "movies": ease_baseline,
                "order": 0
            },
            "CB-ILD": {
                "movies": r2,
                "order": 1
            },
            "CF-ILD": {
                "movies": r3,
                "order": 2
            },
            "Binomial": {
                "movies": r4,
                "order": 3
            }
         }
    }
    print(f"Took 4:", time.perf_counter() - start_time)
    tr = get_tr(languages, get_lang())
    params["contacts"] = tr("footer_contacts")
    params["contact"] = tr("footer_contact")
    params["charles_university"] = tr("footer_charles_university")
    params["cagliari_university"] = tr("footer_cagliari_university")
    params["t1"] = tr("footer_t1")
    params["t2"] = tr("footer_t2")
    params["title"] = tr("metric_assesment_title")
    params["header"] = tr("metric_assesment_header")
    params["hint"] = tr("metric_assesment_hint")
    params["continuation_url"] = request.args.get("continuation_url")
    params["finish"] = tr("metric_assesment_finish")
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
