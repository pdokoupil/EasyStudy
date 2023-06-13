import random
import sys
import os

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

from plugins.fastcompare import elicitation_ended, iteration_started, iteration_ended
from plugins.multiobjective import prepare_recommendations

languages = load_languages(os.path.dirname(__file__))

N_ITERATIONS = 6 # 6 iterations
HIDE_LAST_K = 100000

# Implementation of this function can differ among plugins
def get_lang():
    default_lang = "en"
    if "lang" in session and session["lang"] and session["lang"] in languages:
        return session["lang"]
    return default_lang

__plugin_name__ = "slidershuffling"
__version__ = "0.1.0"
__author__ = "Patrik Dokoupil"
__author_contact__ = "Patrik.Dokoupil@matfyz.cuni.cz"
__description__ = "Compare different layouts for fine-tuning."

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

    return render_template("slidershuffling_create.html", **params)

# Public facing endpoint
@bp.route("/join", methods=["GET"])
def join():
    assert "guid" in request.args, "guid must be available in arguments"
    return redirect(url_for("utils.join", continuation_url=url_for("slidershuffling.on_joined"), **request.args))

# Callback once user has joined we forward to preference elicitation
@bp.route("/on-joined", methods=["GET", "POST"])
def on_joined():
    return redirect(url_for("utils.preference_elicitation",
            continuation_url=url_for("slidershuffling.send_feedback"),
            consuming_plugin=__plugin_name__,
            initial_data_url=url_for('utils.get_initial_data'),
            search_item_url=url_for('utils.movie_search')
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

    # Movie indices of selected movies
    selected_movies = request.args.get("selectedMovies")
    selected_movies = selected_movies.split(",") if selected_movies else []
    selected_movies = [int(m) for m in selected_movies]

    # Calculate weights based on selection and shown movies during preference elicitation
    weights, supports = calculate_weight_estimate(selected_movies, session["elicitation_movies"], return_supports=True)
    weights /= weights.sum()
    weights = weights.tolist()
    weights = {
        algo_displayed_name: weights for algo_displayed_name in displyed_name_mapping.values()
    }
    session["weights"] = weights
    print(f"Weights initialized to {weights}")

    algorithms = list(displyed_name_mapping.values())
    

    # Add default entries so that even the non-chosen algorithm has an empty entry
    # to unify later access
    recommendations = {
        algo: [[]] for algo in algorithms
    }
    initial_weights_recommendation = {
        algo: [[]] for algo in algorithms
    }
    
    # We filter out everything the user has selected during preference elicitation.
    # However, we allow future recommendation of SHOWN, NOT SELECTED (during elicitation, not comparison) altough these are quite rare
    filter_out_movies = selected_movies

    prepare_recommendations(weights, recommendations, initial_weights_recommendation, selected_movies, filter_out_movies, k)
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
        for j, algorithm_displayed_name in enumerate(algorithms):
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

    return redirect(url_for("multiobjective.compare_and_refine"))

from plugins.layoutshuffling import long_initialization

@bp.route("/initialize", methods=["GET"])
def initialize():
    guid = request.args.get("guid")
    p = Process(
        target=long_initialization,
        daemon=True,
        args=(guid, )
    )
    p.start()
    print("Going to redirect back")
    return redirect(request.args.get("continuation_url"))

def register():
    return {
        "bep": dict(blueprint=bp, prefix=None),
    }
