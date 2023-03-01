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

from plugins.utils.preference_elicitation import recommend_2_3, rlprop, weighted_average, get_objective_importance, prepare_tf_model, calculate_weight_estimate

from plugins.fastcompare import elicitation_ended, iteration_started, iteration_ended

languages = load_languages(os.path.dirname(__file__))

N_ITERATIONS = 10
HIDE_LAST_K = 100000

# Implementation of this function can differ among plugins
def get_lang():
    default_lang = "en"
    if "lang" in session and session["lang"] and session["lang"] in languages:
        return session["lang"]
    return default_lang

__plugin_name__ = "multiobjective"
__version__ = "0.1.0"
__author__ = "Patrik Dokoupil"
__author_contact__ = "Patrik.Dokoupil@matfyz.cuni.cz"
__description__ = "Compare RLprop to Weighted Average strategy and allow fine-tuning."

bp = Blueprint(__plugin_name__, __plugin_name__, url_prefix=f"/{__plugin_name__}")

@bp.route("/create")
def create():
    return render_template("multiobjective_create.html")

# Public facing endpoint
@bp.route("/join", methods=["GET"])
def join():
    assert "guid" in request.args, "guid must be available in arguments"
    return redirect(url_for("utils.join", continuation_url=url_for("multiobjective.on_joined"), **request.args))

# Callback once user has joined we forward to preference elicitation
@bp.route("/on-joined", methods=["GET", "POST"])
def on_joined():
    return redirect(url_for("utils.preference_elicitation",
            continuation_url=url_for("multiobjective.send_feedback"),
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

@bp.context_processor
def plugin_name():
    return {
        "plugin_name": __plugin_name__
    }

# We received feedback from compare_algorithms.html
@bp.route("/algorithm-feedback")
def algorithm_feedback():
    # TODO do whatever with the passed parameters and set session variable

    # conf = load_user_study_config(session["user_study_id"])
    
    selected_movies = request.args.get("selected_movies")
    selected_movies = selected_movies.split(",") if selected_movies else []

    selected_variants = request.args.get("selected_movie_variants")
    selected_variants = selected_variants.split(",") if selected_variants else []
    selected_variants = [int(x) for x in selected_variants]

    algorithm_ratings = []
    for i in range(len(displyed_name_mapping)):
        algorithm_ratings.append(int(request.args.get(f"ar_{i + 1}")))

    dont_like_anything = request.args.get("nothing")
    if dont_like_anything == "true":
        dont_like_anything = True
    else:
        dont_like_anything = False
    algorithm_comparison = request.args.get("cmp")
    order = session["permutation"][0]
    ordered_ratings = {}
    for algo_name, idx in order.items():
        ordered_ratings[algo_name] = algorithm_ratings[idx]

    t1 = session["nothing"]
    t1.append(dont_like_anything)
    session["nothing"] = t1

    t2 = session["cmp"]
    t2.append(algorithm_comparison)
    session["cmp"] = t2

    t3 = session["a_r"]
    t3.append(ordered_ratings)
    session["a_r"] = t3


    assert len(selected_variants) == len(selected_movies), f"selected_movies={selected_movies}, selected_variants={selected_variants}"

    selected_movies = [int(m) for m in selected_movies]
    x = session["selected_movie_indices"]
    x.append(selected_movies)
    session["selected_movie_indices"] = x
    
    y = session["selected_variants"]
    y.append(selected_variants)
    session["selected_variants"] = y

    return redirect(url_for("multiobjective.refinement_feedback", refine_results_url=url_for("multiobjective.refine_results")))

@bp.route("/refinement-feedback")
def refinement_feedback():
    version = request.args.get("version") or session["refinement_layout"] #"1"
    return render_template("refinement_feedback.html", iteration=session["iteration"], version=version,
        metrics={
            "relevance": session["weights"][0],
            "diversity": session["weights"][1],
            "novelty": session["weights"][2]
        },
        refine_results_url=request.args.get("refine_results_url")
    )

def prepare_recommendations(weights, recommendations, selected_movies, filter_out_movies, k):
    # Order of insertion should be preserved
    print(f"Called Prepare recommendations")
    recommended_items, model = recommend_2_3(selected_movies, filter_out_movies, return_model=True, k=k)
    for algorithm, algorithm_displayed_name in displyed_name_mapping.items():
        if algorithm == "relevance_based":
            pass
        elif algorithm == "rlprop":
            recommended_items = rlprop(selected_movies, model, weights, filter_out_movies, k=k)
        elif algorithm == "weighted_average":
            recommended_items = weighted_average(selected_movies, model, weights, filter_out_movies, k=k)
        else:
            assert False
        recommendations[algorithm_displayed_name] = [recommended_items]

@bp.route("/refine-results")
def refine_results():
    
    # Get new weights
    new_weights = request.args.get("new_weights")
    new_weights = [float(x) for x in new_weights.split(",")]
    session["weights"] = new_weights

    # Increase iteration
    session["iteration"] += 1
    ### And generate new recommendations ###
    recommendations = session["movies"]

    lengths = []
    for x in displyed_name_mapping.values():
        recommendations[x].append([])
        lengths.append(len(recommendations[x]))

    assert len(set(lengths)), "All algorithms should share the number of iterations"
    n_iterations = lengths[0] # Since all have same number of iteration, pick the first one

    mov_indices = []
    for i in range(n_iterations):
        indices = set()
        for algo_displayed_name in displyed_name_mapping.values():
            indices.update([int(y["movie_idx"]) for y in recommendations[algo_displayed_name][i]])
        mov_indices.append(list(indices))

    
    filter_out_movies = session["elicitation_selected_movies"] + sum(mov_indices[:HIDE_LAST_K], [])
    selected_movies = session["elicitation_selected_movies"] + sum(session["selected_movie_indices"], [])
    
    prepare_recommendations(np.array(new_weights), recommendations, selected_movies, filter_out_movies, k=session["rec_k"])

    session["movies"] = recommendations
    ### End generation ###


    # And shift the permutation
    permutation = session["permutation"]
    permutation = permutation[1:] + permutation[:1] # Move first item to the end
    session["permutation"] = permutation

    iteration_ended(session["iteration"], session["selected_movie_indices"], session["selected_variants"], session["nothing"], session["cmp"], session["a_r"])
    return redirect(url_for(f"{__plugin_name__}.compare_algorithms"))


@bp.route("/compare-algorithms", methods=["GET"])
def compare_algorithms():
    
    if session["iteration"] == 1:
        elicitation_ended(session["elicitation_movies"], session["elicitation_selected_movies"])    

    conf = load_user_study_config(session["user_study_id"])
    algorithm_assignment = {}
    movies = {}

    p = session["permutation"][0]
    
    for i, (algorithm, algorithm_displayed_name) in enumerate(displyed_name_mapping.items()):
        if session["movies"][algorithm_displayed_name][-1]:
            # Only non-empty makes it to the results
            movies[algorithm_displayed_name] = {
                "movies": session["movies"][algorithm_displayed_name][-1],
                "order": p[algorithm_displayed_name]
            }
            algorithm_assignment[str(i)] = {
                "algorithm": algorithm,
                "name": algorithm_displayed_name,
                "order": p[algorithm_displayed_name]
            }

    result_layout = "column-single"

    # In some sense, we can treat this as iteration start
    # TODO fix that we have two algorithms, add weights and fix algorithm_assignment (randomly assigning with each iteration)
    shown_movie_indices = {}
    for algo_name, movie_lists in session["movies"].items():
        shown_movie_indices[algo_name] = [[int(x["movie_idx"]) for x in movie_list] for movie_list in movie_lists]
        
    session["refinement_layout"] = "3"

    iteration_started(session["iteration"], movies, algorithm_assignment, result_layout, shown_movie_indices)

    tr = get_tr(languages, get_lang())
    for d in movies.values():
        x = d["movies"]
        for i in range(len(x)):
            input_name = f"{x[i]['movie_id']}"
            x[i]["movie"] = tr(input_name, x[i]['movie']) + " " + "|".join([tr(f"genre_{y.lower()}") for y in x[i]["genres"]])

    params = {
        "movies": movies,
        "iteration": session["iteration"],
        "result_layout": result_layout,
        "MIN_ITERATION_TO_CANCEL": len(session["permutation"]),
        "consuming_plugin": __plugin_name__,
    }
   
    params["contacts"] = tr("footer_contacts")
    params["contact"] = tr("footer_contact")
    params["charles_university"] = tr("footer_charles_university")
    params["cagliari_university"] = tr("footer_cagliari_university")
    params["t1"] = tr("footer_t1")
    params["t2"] = tr("footer_t2")
    params["title"] = tr("compare_title")
    params["header"] = tr("compare_header")
    params["note"] = tr("note")
    params["algorithm"] = tr("algorithm")
    params["note_text"] = tr("compare_note_text")
    params["hint"] = tr("compare_hint")
    params["hint_lead"] = tr("compare_hint_lead")
    params["algorithm_satisfaction"] = tr("compare_algorithm_satisfaction")
    params["like_nothing"] = tr("compare_like_nothing")
    params["significantly"] = tr("compare_significantly")
    params["slightly"] = tr("compare_slightly")
    params["same"] = tr("compare_same")
    params["next"] = tr("next")
    params["finish"] = tr("compare_finish")
    params["algorithm_how_compare"] = tr("compare_algorithm_how_compare")


    return render_template("compare_algorithms.html", **params)

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
    weights = calculate_weight_estimate(selected_movies, session["elicitation_movies"])
    weights /= weights.sum()
    session["weights"] = weights.tolist()

    algorithms = list(displyed_name_mapping.values())
    # Add default entries so that even the non-chosen algorithm has an empty entry
    # to unify later access
    recommendations = {
        algo: [[]] for algo in algorithms
    }
    
    # We filter out everything the user has selected during preference elicitation.
    # However, we allow future recommendation of SHOWN, NOT SELECTED (during elicitation, not comparison) altough these are quite rare
    filter_out_movies = selected_movies

    prepare_recommendations(weights, recommendations, selected_movies, filter_out_movies, k)

    
    session["movies"] = recommendations
    session["iteration"] = 1
    session["elicitation_selected_movies"] = selected_movies
    session["selected_movie_indices"] = [] #dict() # For each iteration, we can store selected movies
    session["selected_variants"] = []
    session["nothing"] = []
    session["cmp"] = []
    session["a_r"] = []

    # Build permutation
    p = []
    for i in range(N_ITERATIONS):
        orders = dict()
        available_orders = list(range(len(algorithms)))
        for algorithm_displayed_name in algorithms:
            
            #if conf["shuffle_algorithms"]:
            order_idx = np.random.randint(len(available_orders))
            #else:
            #    order_idx = 0

            orders[algorithm_displayed_name] = available_orders[order_idx]
            del available_orders[order_idx]

        p.append(orders)
    session["permutation"] = p
    session["orig_permutation"] = p

    print(f"Recommendations={recommendations}")
    return redirect(url_for("multiobjective.compare_algorithms"))



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
