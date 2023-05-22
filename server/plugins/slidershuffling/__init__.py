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
    params["override_about"] = tr("moo_create_override_about")
    params["show_final_statistics"] = tr("moo_create_show_final_statistics")
    params["override_algorithm_comparison_hint"] = tr("moo_create_override_algorithm_comparison_hint")
    params["algorithm_comparison_placeholder"] = tr("moo_create_algorithm_comparison_placeholder")

    params["disable_relative_comparison"] = "Disable relative comparison"
    params["disable_demographics"] = "Disable demographics"

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

# def get_all_recommended_items(n_iterations, recommendations):
#     mov_indices = []
#     for i in range(n_iterations):
#         indices = set()
#         for algo_displayed_name in displyed_name_mapping.values():
#             if algo_displayed_name not in session["orig_permutation"][i]:
#                 #print(f"@@ Ignoring {algo_displayed_name} for iteration={i}, orders = {session['orig_permutation']}")
#                 continue
#             indices.update([int(y["movie_idx"]) for y in recommendations[algo_displayed_name][i]])
#         mov_indices.append(list(indices))
#     return mov_indices

# @bp.route("/algorithm-feedback")
# def algorithm_feedback():
#     # TODO do whatever with the passed parameters and set session variable

#     # conf = load_user_study_config(session["user_study_id"])
    
#     selected_movies = request.args.get("selected_movies")
#     selected_movies = selected_movies.split(",") if selected_movies else []

#     selected_variants = request.args.get("selected_movie_variants")
#     selected_variants = selected_variants.split(",") if selected_variants else []
#     selected_variants = [int(x) for x in selected_variants]

#     order = session["permutation"][0]

#     algorithm_ratings = []
#     for i in range(len(order)): # Skip algorithms with order being < 0
#         algorithm_ratings.append(int(request.args.get(f"ar_{i + 1}")))

#     not_shown_algorithm = [x for x in displyed_name_mapping.values() if x not in order]
#     assert len(not_shown_algorithm) == 1, f"{not_shown_algorithm}"
#     not_shown_algorithm = not_shown_algorithm[0]
#     print(f"Hidden algorithm is: {not_shown_algorithm}")

#     dont_like_anything = request.args.get("nothing")
#     if dont_like_anything == "true":
#         dont_like_anything = True
#     else:
#         dont_like_anything = False
#     algorithm_comparison = request.args.get("cmp")
    
#     ordered_ratings = {}
#     for algo_name, idx in order.items():
#         ordered_ratings[algo_name] = algorithm_ratings[idx]

#     t1 = session["nothing"]
#     t1.append(dont_like_anything)
#     session["nothing"] = t1

#     t2 = session["cmp"]
#     t2.append(algorithm_comparison)
#     session["cmp"] = t2

#     t3 = session["a_r"]
#     t3.append(ordered_ratings)
#     session["a_r"] = t3


#     assert len(selected_variants) == len(selected_movies), f"selected_movies={selected_movies}, selected_variants={selected_variants}"

#     selected_movies = [int(m) for m in selected_movies]
#     x = session["selected_movie_indices"]
#     x.append(selected_movies)
#     session["selected_movie_indices"] = x
    
#     y = session["selected_variants"]
#     y.append(selected_variants)
#     session["selected_variants"] = y

#     # Get new weights
#     new_weights = request.args.get("new_weights").split(";") # Get new weights for each algorithm
#     transformed_weights = []
#     for i, weights in enumerate(new_weights):
#         if weights:
#             transformed_weights.append([float(x) for x in weights.split(",")])
#         else:
#             transformed_weights.append([])
#     #new_weights = [float(x) if x else '' for y in new_weights for x in y.split(",")]
#     new_weights = transformed_weights
#     ordered_weights = {}
#     for algo_name, idx in order.items():
#         ordered_weights[algo_name] = new_weights[idx]
#     old_weights = session["weights"]
#     ordered_weights[displyed_name_mapping["relevance_based"]] = old_weights[displyed_name_mapping["relevance_based"]]
#     ordered_weights[not_shown_algorithm] = old_weights[not_shown_algorithm]
#     print(f"Old weights = {old_weights}, new weights = {new_weights}")
#     print(f"Ordered weights after fixup = {ordered_weights}")
#     session["weights"] = ordered_weights # Take first non-empty weights TODO fix and set weights to each algorithm separately

#     # Log end of iteration here
#     iteration_ended(
#         session["iteration"], session["selected_movie_indices"], session["selected_variants"],
#         session["nothing"], session["cmp"], session["a_r"],
#         old_weights = old_weights, new_weights = session["weights"]
#     )

#     if session["iteration"] >= N_ITERATIONS:
#         # We are done, continue with another step
#         # This happens (intentionally) after logging iteration_ended
#         return redirect(url_for(f"{__plugin_name__}.compare_done"))


#     ### And generate new recommendations ###
#     recommendations = session["movies"]
#     initial_weights_recommendation = session["initial_weights_recommendation"]

#     lengths = []
#     for x in displyed_name_mapping.values():
#         recommendations[x].append([])
#         lengths.append(len(recommendations[x]))

#     assert len(set(lengths)), "All algorithms should share the number of iterations"
#     n_iterations = lengths[0] # Since all have same number of iteration, pick the first one

#     mov_indices = get_all_recommended_items(n_iterations, recommendations)

#     filter_out_movies = session["elicitation_selected_movies"] + sum(mov_indices[:HIDE_LAST_K], [])
#     selected_movies = session["elicitation_selected_movies"] + sum(session["selected_movie_indices"], [])

#     prepare_recommendations(ordered_weights, recommendations, initial_weights_recommendation, selected_movies, filter_out_movies, k=session["rec_k"])

#     session["movies"] = recommendations
#     session["initial_weights_recommendation"] = initial_weights_recommendation
#     ### End generation ###


#     # And shift the permutation
#     permutation = session["permutation"]
#     permutation = permutation[1:] + permutation[:1] # Move first item to the end
#     session["permutation"] = permutation

#     # Increase iteration
#     session["iteration"] += 1

#     return redirect(url_for("multiobjective.compare_and_refine"))


def prepare_recommendations(weights, recommendations, initial_weights_recommendation, selected_movies, filter_out_movies, k):
    # Order of insertion should be preserved
    print(f"Called Prepare recommendations")
    recommended_items, model = recommend_2_3(selected_movies, filter_out_movies, return_model=True, k=k)
    initial_weights_recommended_items = recommended_items
    for algorithm, algorithm_displayed_name in displyed_name_mapping.items():
        if algorithm == "relevance_based":
            pass
        elif algorithm == "rlprop":
            recommended_items = rlprop(selected_movies, model, np.array(weights[algorithm_displayed_name]), filter_out_movies, k=k, include_support=True)
            initial_weights_recommended_items = rlprop(selected_movies, model, np.array(weights[displyed_name_mapping["relevance_based"]]), filter_out_movies, k=k, include_support=True)
        elif algorithm == "weighted_average":
            recommended_items = weighted_average(selected_movies, model, np.array(weights[algorithm_displayed_name]), filter_out_movies, k=k, include_support=True)
            initial_weights_recommended_items = weighted_average(selected_movies, model, np.array(weights[displyed_name_mapping["relevance_based"]]), filter_out_movies, k=k, include_support=True)
        else:
            assert False
        recommendations[algorithm_displayed_name][-1] = recommended_items
        initial_weights_recommendation[algorithm_displayed_name][-1] = initial_weights_recommended_items

# @bp.route("/compare-done")
# def compare_done():
#     # Prepare questions for final questionnaire
#     # All recommended movies
#     for recommendations in session["movies"].values():
#         assert len(recommendations) == N_ITERATIONS
#     all_recommended = sum(get_all_recommended_items(N_ITERATIONS, session["movies"]),  [])
#     all_elicited = [x['movie_idx'] for x in session["elicitation_movies"]]
#     all_selected = sum(session["selected_movie_indices"], [])
    
#     loader = load_ml_dataset()
#     all_indices = set(loader.movie_index_to_id.keys())

#     unseen_indices = all_indices.difference(all_recommended).difference(all_elicited)
    
#     orders = [0, 1, 2]
#     np.random.shuffle(orders)

#     not_recommended_movie = random.sample(list(unseen_indices), 1)[0]
#     not_selected_movie = random.sample(all_recommended, 1)[0]

#     if len(all_selected) == 0:
#         # It may (very rarely) happen that user does not select anything
#         # In that case, we show some other item that was just recommended but not selected
#         selected_recommended_movie = random.sample(all_recommended, 1)[0]
#     else:
#         selected_recommended_movie = random.sample(all_selected, 1)[0]

#     attention_movies = [-1] * len(orders)
#     attention_movies[orders[0]] = not_recommended_movie
#     attention_movies[orders[1]] = not_selected_movie
#     attention_movies[orders[2]] = selected_recommended_movie

#     attention_movies_enriched = enrich_results(attention_movies, loader)
#     attention_movies_enriched[orders[0]]["selected"] = False
#     attention_movies_enriched[orders[0]]["recommended"] = False
#     attention_movies_enriched[orders[0]]["not_recommended"] = True
#     attention_movies_enriched[orders[0]]["order"] = orders[0]

#     attention_movies_enriched[orders[1]]["selected"] = False
#     attention_movies_enriched[orders[1]]["recommended"] = True
#     attention_movies_enriched[orders[1]]["not_recommended"] = False
#     attention_movies_enriched[orders[1]]["order"] = orders[1]

#     attention_movies_enriched[orders[2]]["selected"] = len(all_selected) > 0 # If nothing was selected, set to False
#     attention_movies_enriched[orders[2]]["recommended"] = True
#     attention_movies_enriched[orders[2]]["not_recommended"] = False
#     attention_movies_enriched[orders[2]]["order"] = orders[2]

#     session["attention_check"] = attention_movies_enriched
    
#     log_interaction(session["participation_id"], "attention-check-input", attention_check_input=session["attention_check"])

#     return redirect(url_for(f"{__plugin_name__}.final_questionnaire"))

# @bp.route("/final-questionnaire")
# @multi_lang
# def final_questionnaire():
#     if "iteration" not in session:
#         print(f"This should no happen, session: {session}", flush=True)
#         part = session["participation_id"] if "participation_id" in session else None
#         data = {
#             "message": "Iteration not found",
#             "accessed": session.accessed,
#             "modified": session.modified,
#             "session_kv": dict(session.items())
#         }
#         log_message(part, **data)
#     else:
#         part = session["participation_id"] if "participation_id" in session else None
#         log_message(part, **{"message": "Everything is OK", "Check": session["attention_check"]})

#     params = {
#         "continuation_url": url_for(f'{__plugin_name__}.finish_user_study')
#     }

#     tr = get_tr(languages, get_lang())
#     params["contacts"] = tr("footer_contacts")
#     params["contact"] = tr("footer_contact")
#     params["charles_university"] = tr("footer_charles_university")
#     params["cagliari_university"] = tr("footer_cagliari_university")
#     params["t1"] = tr("footer_t1")
#     params["t2"] = tr("footer_t2")
#     params["title"] = tr("final_title")
#     params["header"] = tr("final_header")
#     params["finish"] = tr("final_finish")
#     params["hint"] = tr("final_hint")

#     params["attention_movies"] = session["attention_check"]   

#     return render_template("final_questionnaire.html", **params)

# @bp.route("/finish-user-study")
# def finish_user_study():

#     gold_attention_check = session["attention_check"]

#     real_attention_check = []
#     for x in gold_attention_check:

#         state_key = f"{x['movie_idx']}ch"

#         not_recommended = request.args.get(state_key) == "0"
#         recommended = request.args.get(state_key) == "1"
#         selected = request.args.get(state_key) == "2"
#         not_sure = request.args.get(state_key) == "3"

#         hits = 0
#         if not not_sure:
#             # If both are either selected or not selected, add hit
#             if selected == x["selected"]:
#                 hits += 1
#             virtual_recommended = recommended or selected # Selected automatically imply it was recommended
#             if virtual_recommended == x["recommended"]:
#                 hits += 1

#         real_attention_check.append({
#             "movie_idx": x["movie_idx"],
#             "order": x["order"],
#             "gold": {
#                 "recommended": x["recommended"],
#                 "selected": x["selected"]
#             },
#             "real": {
#                 "not_recommended": not_recommended,
#                 "recommended": recommended,
#                 "selected": selected,
#                 "not_sure": not_sure,
#                 "raw_value": request.args.get(state_key)
#             },
#             "hits": hits, # User may hit either recommended, selected, or both
#         })

#     data = {
#         "attention_check": real_attention_check
#     }
#     data.update(**request.args)

#     log_interaction(session["participation_id"], "final-questionnaire", **data)
#     return redirect(url_for("utils.finish"))


# @bp.route("/compare-and-refine", methods=["GET"])
# def compare_and_refine():
    
#     # if session["iteration"] == 1:
#     #     elicitation_ended(session["elicitation_movies"], session["elicitation_selected_movies"])    

#     conf = load_user_study_config(session["user_study_id"])
#     algorithm_assignment = {}
#     movies = {}

#     p = session["permutation"][0]
#     refinement_algorithms = [-1 for _ in range(len(p))] #[-1 for _ in displyed_name_mapping]
#     for i, (algorithm, algorithm_displayed_name) in enumerate(displyed_name_mapping.items()):
#         if session["movies"][algorithm_displayed_name][-1]:
#             order_idx = p[algorithm_displayed_name] if algorithm_displayed_name in p else -1
#             # Only non-empty makes it to the results
#             movies[algorithm_displayed_name] = {
#                 "movies": session["movies"][algorithm_displayed_name][-1],
#                 "order": order_idx
#             }
#             algorithm_assignment[str(i)] = {
#                 "algorithm": algorithm,
#                 "name": algorithm_displayed_name,
#                 "order": order_idx
#             }
#             if order_idx != -1: # Those with negative indices are skipped
#                 refinement_algorithms[order_idx] = int(algorithm != "relevance_based")
            

#     result_layout = "column-single"

#     # In some sense, we can treat this as iteration start
#     # TODO fix that we have two algorithms, add weights and fix algorithm_assignment (randomly assigning with each iteration)
#     shown_movie_indices = {}
#     for algo_name, movie_lists in session["movies"].items():
#         shown_movie_indices[algo_name] = [[int(x["movie_idx"]) for x in movie_list] for movie_list in movie_lists]
        
#     iteration_started(
#         session["iteration"], movies, algorithm_assignment,
#         result_layout, shown_movie_indices, weights = session["weights"],
#         initial_weights_recommendation=session["initial_weights_recommendation"]
#     )

#     tr = get_tr(languages, get_lang())
#     for d in movies.values():
#         x = d["movies"]
#         for i in range(len(x)):
#             input_name = f"{x[i]['movie_id']}"
#             x[i]["movie"] = tr(input_name, x[i]['movie']) + " " + "|".join([tr(f"genre_{y.lower()}") for y in x[i]["genres"]])

#     params = {
#         "movies": { algo: rec for algo, rec in movies.items() if algo in p}, # Only show a subset of algorithms
#         "iteration": session["iteration"],
#         "result_layout": result_layout,
#         "MIN_ITERATION_TO_CANCEL": len(session["permutation"]),
#         "consuming_plugin": __plugin_name__,
#         "refinement_layout": session['refinement_layout'],
#         "metrics" : {
#             "relevance": {algo_name: round(weights[0], 2) for algo_name, weights in session["weights"].items()},
#             "diversity": {algo_name: round(weights[1], 2) for algo_name, weights in session["weights"].items()},
#             "novelty": {algo_name: round(weights[2], 2) for algo_name, weights in session["weights"].items()}
#         },
#         "refinement_algorithms": refinement_algorithms,
#         "continuation_url": url_for(f"{__plugin_name__}.algorithm_feedback")
#     }
   
#     params["contacts"] = tr("footer_contacts")
#     params["contact"] = tr("footer_contact")
#     params["charles_university"] = tr("footer_charles_university")
#     params["cagliari_university"] = tr("footer_cagliari_university")
#     params["t1"] = tr("footer_t1")
#     params["t2"] = tr("footer_t2")
#     params["title"] = tr("compare_title")
#     params["header"] = tr("compare_header")
#     params["note"] = tr("note")
#     params["algorithm"] = tr("algorithm")
#     params["note_text"] = tr("compare_note_text")
#     params["hint"] = tr("compare_hint")
#     params["hint_lead"] = tr("compare_hint_lead")
#     params["algorithm_satisfaction"] = tr("compare_algorithm_satisfaction")
#     params["like_nothing"] = tr("compare_like_nothing")
#     params["significantly"] = tr("compare_significantly")
#     params["slightly"] = tr("compare_slightly")
#     params["same"] = tr("compare_same")
#     params["next"] = tr("next")
#     params["finish"] = tr("compare_finish")
#     params["algorithm_how_compare"] = tr("compare_algorithm_how_compare")
#     params["relevance_explanation"] = tr("refine_relevance_explanation")
#     params["diversity_explanation"] = tr("refine_diversity_explanation")
#     params["novelty_explanation"] = tr("refine_novelty_explanation")

#     # Handle textual overrides
#     params["comparison_hint_override"] = None
#     params["footer_override"] = None
#     if "text_overrides" in conf:
#         if "comparison_hint" in conf["text_overrides"]:
#             params["comparison_hint_override"] = conf["text_overrides"]["comparison_hint"]

#         if "footer" in conf["text_overrides"]:
#             params["footer_override"] = conf["text_overrides"]["footer"]

#     return render_template("compare_and_refine.html", **params)

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
    
    session["movies"] = recommendations
    session["initial_weights_recommendation"] = initial_weights_recommendation
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
    refinement_layout_name = np.random.choice(list(refinement_layouts.keys()))
    session["refinement_layout"] = refinement_layouts[refinement_layout_name]
    
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
