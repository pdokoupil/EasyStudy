# -*- coding: utf-8 -*-

import json
import os
from flask import Blueprint, request, redirect, url_for, render_template, session
from sklearn.preprocessing import QuantileTransformer

from plugins.utils.preference_elicitation import recommend_2_3, rlprop, weighted_average, result_layout_variants, get_objective_importance, prepare_tf_model
from plugins.utils.data_loading import load_ml_dataset
from plugins.utils.interaction_logging import log_interaction, study_ended

from models import Interaction, Participation, UserStudy
from app import db
from common import get_tr, load_languages, multi_lang, load_user_study_config

import numpy as np

__plugin_name__ = "layoutshuffling"
__version__ = "0.1.0"
__author__ = "Patrik Dokoupil"
__author_contact__ = "Patrik.Dokoupil@matfyz.cuni.cz"
__description__ = "Simple plugin comparing RLprop with Matrix Factorization while shuffling result layouts (for illustration purposes only)."

bp = Blueprint(__plugin_name__, __plugin_name__, url_prefix=f"/{__plugin_name__}")

NUM_TO_SELECT = 5

MIN_ITERATION_TO_CANCEL = 5
TOTAL_ITERATIONS = 8

HIDE_LAST_K = 1000000 # Effectively hides everything

languages = load_languages(os.path.dirname(__file__))

# Map internal algorithm names to those displayed to user
algorithm_name_mapping = {
    #"rlprop": "beta",
    "relevance_based": "gamma",
    "weighted_average": "delta"
}

# Implementation of this function can differ among plugins
def get_lang():
    default_lang = "en"
    if "lang" in session and session["lang"] and session["lang"] in languages:
        return session["lang"]
    return default_lang

@bp.context_processor
def plugin_name():
    return {
        "plugin_name": __plugin_name__
    }

@bp.route("/create")
def create():
    return render_template("create.html")

@bp.route("/num-to-select")
def get_num_to_select():
    return {
        'num_to_select': NUM_TO_SELECT
    }

# Public facing endpoint
@bp.route("/join", methods=["GET"])
@multi_lang
def join():
    assert "guid" in request.args, "guid must be available in arguments"
    #guid = request.args.get("guid")
    return redirect(url_for("utils.join", continuation_url=url_for("layoutshuffling.on_joined"), **request.args))

# Callback once user has joined we forward to preference elicitation
@bp.route("/on-joined", methods=["GET", "POST"])
def on_joined():
    return redirect(url_for("utils.preference_elicitation",
            continuation_url=url_for("layoutshuffling.send_feedback"),
            consuming_plugin="layoutshuffling",
            initial_data_url=url_for('utils.get_initial_data'),
            search_item_url=url_for('utils.movie_search')
        )
    )

@bp.route("/compare-algorithms", methods=["GET"])
def compare_algorithms():
    
    if session["iteration"] == 1:
        elicitation_ended(session["elicitation_movies"], session["elicitation_selected_movies"])    

    k_param = request.args.get("k")
    if k_param:
        x = session["selected_movie_indices"]
        y = session["selected_variants"]
        if not x:
            x.append([])
            y.append([])
            session["selected_movie_indices"] = x
            session["selected_variants"] = y
            prepare_recommendations(k=int(k_param))
            session["selected_movie_indices"] = x[:-1]
            session["selected_variants"] = y[:-1]
        else:
            prepare_recommendations(k=int(k_param))
    p = session["permutation"][0]

    algorithm_assignment = {}
    algorithms = list(algorithm_name_mapping.keys())
    movies = {}
    for i, algorithm in enumerate(algorithms):
        if session["movies"][algorithm][-1]:
            # Only non-empty makes it to the results
            movies[algorithm_name_mapping[algorithm]] = {
                "movies": session["movies"][algorithm][-1],
                "order": p["order"][algorithm]
            }
            algorithm_assignment[str(i)] = {
                "algorithm": algorithm,
                "name": algorithm_name_mapping[algorithm],
                "order": p["order"][algorithm]
            }


    #result_layout = request.args.get("result_layout") or "rows"
    result_layout = result_layout_variants[p["result_layout"]]

    # Decide on next refinement layout
    refinement_layout = "3" # Use version 3
    session["refinement_layout"] = refinement_layout

    # In some sense, we can treat this as iteration start
    # TODO fix that we have two algorithms, add weights and fix algorithm_assignment (randomly assigning with each iteration)
    shown_movie_indices = {}
    for algo_name, movie_lists in session["movies"].items():
        shown_movie_indices[algo_name] = [[int(x["movie_idx"]) for x in movie_list] for movie_list in movie_lists]
        
    iteration_started(session["iteration"], session["weights"], movies, algorithm_assignment, result_layout, refinement_layout, shown_movie_indices)

    tr = get_tr(languages, get_lang())
    for d in movies.values():
        x = d["movies"]
        for i in range(len(x)):
            x[i]["movie"] = tr(str(x[i]["movie_id"])) + " " + "|".join([tr(f"genre_{y.lower()}") for y in x[i]["genres"]])

    params = {
        "movies": movies,
        "iteration": session["iteration"],
        "result_layout": result_layout,
        "MIN_ITERATION_TO_CANCEL": len(session["permutation"]),
        "consuming_plugin": __plugin_name__
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

@bp.route("/refinement-feedback", methods=["GET"])
def refinement_feedback():
    version = request.args.get("version") or session["refinement_layout"] #"1"
    return render_template("refinement_feedback.html", iteration=session["iteration"], version=version,
        metrics={
            "relevance": session["weights"][0],
            "diversity": session["weights"][1],
            "novelty": session["weights"][2]
        }
    )


# We received feedback from compare_algorithms.html
@bp.route("/algorithm-feedback")
def algorithm_feedback():
    # TODO do whatever with the passed parameters and set session variable

    selected_movies = request.args.get("selected_movies")
    selected_movies = selected_movies.split(",") if selected_movies else []

    selected_variants = request.args.get("selected_movie_variants")
    selected_variants = selected_variants.split(",") if selected_variants else []
    selected_variants = [int(x) for x in selected_variants]

    algorithm_1_rating = float(request.args.get("ar_1"))
    algorithm_2_rating = float(request.args.get("ar_2"))
    dont_like_anything = request.args.get("nothing")
    if dont_like_anything == "true":
        dont_like_anything = True
    else:
        dont_like_anything = False
    algorithm_comparison = request.args.get("cmp")
    order = session["permutation"][0]["order"]
    ordered_ratings = {}
    for algo_name, idx in order.items():
        ordered_ratings[algorithm_name_mapping[algo_name]] = algorithm_1_rating if idx == 0 else algorithm_2_rating

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


    # return redirect(url_for("layoutshuffling.refinement_feedback")) # TODO uncomment for main user study
    ##### START OF NEW, SHORTER VERSION
    # Since we never get to refine-results, we have to move some of the stuff here
    # E.g. we should call iteration ended here, weights are kept the same
    # TODO store dont_like_anything, algorithm_comparison, and ordered_ratings in session
    iteration_ended(session["iteration"], session["selected_movie_indices"], session["selected_variants"], session["weights"], session["nothing"], session["cmp"], session["a_r"])    
    # Increase iteration
    session["iteration"] += 1
    # And generate new recommendations
    prepare_recommendations(k=session["rec_k"])
    # And shift the permutation
    permutation = session["permutation"]
    permutation = permutation[1:] + permutation[:1] # Move first item to the end
    session["permutation"] = permutation
    return redirect(url_for("layoutshuffling.compare_algorithms"))
    ##### END OF NEW, SHORTER VERSION

def prepare_recommendations(k=10):
    mov = session["movies"]

    # Randomly chose two algorithms
    algorithms = ["relevance_based", "weighted_average"]
    assert len(mov[algorithms[0]]) == len(mov[algorithms[1]]), "All algorithms should share the number of iterations"
    
    for algorithm in algorithms:
        mov[algorithm].append([])

    mov_indices = []
    for i in range(len(mov[algorithms[0]])):
        indices = set()
        for algo in algorithms:
            indices.update([int(y["movie_idx"]) for y in mov[algo][i]])
        mov_indices.append(list(indices))

    
    filter_out_movies = session["elicitation_selected_movies"] + sum(mov_indices[:HIDE_LAST_K], [])

    # Always generate recommendation via relevance based algorithm because we need to get the model (we use it as a baseline)
    selected_movies = sum(session["selected_movie_indices"], [])
    recommended_items, model = recommend_2_3(session["elicitation_selected_movies"] + selected_movies, filter_out_movies, return_model=True, k=k)    

    # Order of insertion should be preserved
    for algorithm in algorithms:
        if algorithm == "relevance_based":
            pass
        elif algorithm == "rlprop":
            recommended_items = rlprop(session["elicitation_selected_movies"] + selected_movies, model, np.array(session['weights']), filter_out_movies, k=k)
        elif algorithm == "weighted_average":
            recommended_items = weighted_average(session["elicitation_selected_movies"] + selected_movies, model, np.array(session['weights']), filter_out_movies, k=k)
        else:
            assert False
        mov[algorithm][-1] = recommended_items

    session["movies"] = mov

# We receive feedback from refinement_feedback.html
@bp.route("/refine-results")
def refine_results():
    # Get new weights
    new_weights = request.args.get("new_weights")
    new_weights = [float(x) for x in new_weights.split(",")]
    session["weights"] = new_weights

    # Go back to compare algorithms
    session["iteration"] += 1
    # Generate new recommendations
    mov = session["movies"]
    
    # Randomly chose two algorithms
    algorithms = ["relevance_based", "weighted_average"]
    assert len(mov[algorithms[0]]) == len(mov[algorithms[1]]), "All algorithms should share the number of iterations"
    
    for algorithm in algorithms:
        mov[algorithm].append([])

    mov_indices = []
    for i in range(len(mov[algorithms[0]])):
        indices = set()
        for algo in algorithms:
            indices.update([int(y["movie_idx"]) for y in mov[algo][i]])
        mov_indices.append(list(indices))

    
    filter_out_movies = session["elicitation_selected_movies"] + sum(mov_indices[:HIDE_LAST_K], [])

    # Always generate recommendation via relevance based algorithm because we need to get the model (we use it as a baseline)
    selected_movies = sum(session["selected_movie_indices"], [])
    recommended_items, model = recommend_2_3(session["elicitation_selected_movies"] + selected_movies, filter_out_movies, return_model=True)    

    # Order of insertion should be preserved
    for algorithm in algorithms:
        if algorithm == "relevance_based":
            pass
        elif algorithm == "rlprop":
            recommended_items = rlprop(session["elicitation_selected_movies"] + selected_movies, model, np.array(new_weights), filter_out_movies)
        elif algorithm == "weighted_average":
            recommended_items = weighted_average(session["elicitation_selected_movies"] + selected_movies, model, np.array(new_weights), filter_out_movies)
        else:
            assert False
        mov[algorithm][-1] = recommended_items

    session["movies"] = mov

    # In some sense, session ended here
    iteration_ended(session["iteration"] - 1, session["selected_movie_indices"], session["selected_variants"], new_weights)    

    return redirect(url_for("layoutshuffling.compare_algorithms"))

@bp.route("/final-questionare")
@multi_lang
def final_questionare():
    params = {
        "iteration": session["iteration"]
    }

    tr = get_tr(languages, get_lang())
    params["contacts"] = tr("footer_contacts")
    params["contact"] = tr("footer_contact")
    params["charles_university"] = tr("footer_charles_university")
    params["cagliari_university"] = tr("footer_cagliari_university")
    params["t1"] = tr("footer_t1")
    params["t2"] = tr("footer_t2")
    params["title"] = tr("final_title")
    params["header"] = tr("final_header")
    params["finish"] = tr("final_finish")
    params["hint"] = tr("final_hint")

    params["premature"] = False
    if session["iteration"] < TOTAL_ITERATIONS:
        params["premature"] = True

    return render_template("final_questionare.html", **params)

def elicitation_ended(elicitation_movies, elicitation_selected_movies):
    data = {
        "elicitation_movies": elicitation_movies,
        "elicitation_selected_movies": elicitation_selected_movies
    }
    log_interaction(session["participation_id"], "elicitation-ended", **data)

def iteration_started(iteration, weights, movies, algorithm_assignment, result_layout, refinement_layout, shown_movie_indices):
    data = {
        "iteration": iteration,
        "weights": weights,
        "movies": movies,
        "algorithm_assignment": algorithm_assignment,
        "result_layout": result_layout,
        "refinement_layout": refinement_layout,
        "shown": shown_movie_indices
    }
    log_interaction(session["participation_id"], "iteration-started", **data)

def iteration_ended(iteration, selected, selected_variants, new_weights, dont_like_anything, algorithm_comparison, ordered_ratings):
    data = {
        "iteration": iteration,
        "selected": selected,
        "new_weights": new_weights,
        "selected_variants": selected_variants,
        "dont_like_anything": dont_like_anything,
        "algorithm_comparison": algorithm_comparison,
        "ratings": ordered_ratings
    }
    log_interaction(session["participation_id"], "iteration-ended", **data)


@bp.route("/finish-user-study")
@multi_lang
def finish_user_study():
    # TODO once we get back to full user study, we remove this call because it will be called before refinement feedback and we only keep study_ended
    iteration_ended(session["iteration"], session["selected_movie_indices"], session["selected_variants"], session["weights"], session["nothing"], session["cmp"], session["a_r"])
    study_ended(session["participation_id"], iteration=session["iteration"])

    params = {}
    tr = get_tr(languages, get_lang())
    params["contacts"] = tr("footer_contacts")
    params["contact"] = tr("footer_contact")
    params["charles_university"] = tr("footer_charles_university")
    params["cagliari_university"] = tr("footer_cagliari_university")
    params["t1"] = tr("footer_t1")
    params["t2"] = tr("footer_t2")
    params["title"] = tr("finish_title")
    params["header"] = tr("finish_header")
    params["hint"] = tr("finish_hint")
    params["statistics"] = tr("finish_statistics")
    params["in_total"] = tr("finish_in_total")
    params["movies_out_of"] = tr("finish_movies_out_of")
    params["rec_to_you"] = tr("finish_rec_to_you")
    params["out_of_the"] = tr("finish_out_of_the")
    params["you_have_selected"] = tr("finish_you_have_selected")
    params["and"] = tr("and")
    params["were_rec_by"] = tr("finish_were_rec_by")
    params["therefore_you_liked"] = tr("finish_therefore_you_liked")
    params["auto_redirect"] = tr("finish_auto_redirect")
    params["finish_user_study"] = tr("finish_finish_user_study")
    params["therefore_same"] = tr("finish_therefore_same")
    params["more"] = tr("more")
    params["out_of"] = tr("finish_out_of")
    params["than_those"] = tr("finish_than_those")
    params["stars"] = tr("finish_stars")
    params["movies_that_were"] = tr("finish_movies_that_were")
    params["during_pref"] = tr("finish_during_pref")
    params["on_avg"] = tr("finish_on_avg")
    params["rec_from_algo"] = tr("finish_rec_from_algo")
    params["rel_more"] = tr("finish_rel_more")
    params["avg_importance"] = tr("finish_avg_importance")
    params["for_rel_div_nov"] = tr("finish_for_rel_div_nov")
    params["of_other"] = tr("finish_of_other")
    params["for_div_nov"] = tr("finish_for_div_nov")
    params["gave_importance"] = tr("finish_gave_importance")
    params["importance_of"] = tr("finish_importance_of")
    params["to_rel"] = tr("finish_to_rel")
    params["to_div"] = tr("finish_to_div")
    params["to_nov"] = tr("finish_to_nov")
    params["relevance_explanation"] = tr("finish_relevance_explanation")
    params["diversity_explanation"] = tr("finish_diversity_explanation")
    params["novelty_explanation"] = tr("finish_novelty_explanation")


    # Prolific stuff
    if "PROLIFIC_PID" in session:
        conf = load_user_study_config(session["user_study_id"])
        params["prolific_pid"] = session["PROLIFIC_PID"]
        params["prolific_url"] = f"https://app.prolific.co/submissions/complete?cc={conf['prolific_code']}"
    else:
        params["prolific_pid"] = None

    # Statistics
    params["n_selected"] = sum([len(x) for x in session["selected_movie_indices"]])
    params["n_recommended"] = int(session["iteration"]) * session["rec_k"] * 2 # TODO specific for 2 variants


    counts = {
        name: 0 for name in algorithm_name_mapping.values()
    }

    for variants, p in zip(session["selected_variants"], session["orig_permutation"]):
        # Convert "relevance_based":0 format to 0:"gamma"
        algo_idx_to_name = {idx: algorithm_name_mapping[algo_name] for algo_name, idx in p["order"].items()}
        for v in variants:
            counts[algo_idx_to_name[v]] += 1


    params["n_gamma"] = counts["gamma"]
    params["n_delta"] = counts["delta"]


    params["avg_rating_gamma"] = 0
    params["avg_rating_delta"] = 0

    for ratings in session["a_r"]:
        params["avg_rating_gamma"] += ratings["gamma"]
        params["avg_rating_delta"] += ratings["delta"]
    
    params["avg_rating_gamma"] = round(params["avg_rating_gamma"] / len(session["a_r"]), 1)
    params["avg_rating_delta"] = round(params["avg_rating_delta"] / len(session["a_r"]), 1)

    params["n_selected_elicitation"] = len(session["elicitation_selected_movies"])
    params["n_shown_elicitation"] = len(session["elicitation_movies"])


    all_selected_movie_indices = sum(session["selected_movie_indices"], [])
    all_shown_movies = []
    for shown_movie_lists in session["movies"].values():
        all_shown_movies.extend(sum(shown_movie_lists, []))
    objective_importance = get_objective_importance(all_selected_movie_indices, all_shown_movies)
    
    if objective_importance:
        params["relevance"] = round(objective_importance["relevance"], 1)
        params["diversity"] = round(objective_importance["diversity"], 1)
        params["novelty"] = round(objective_importance["novelty"], 1)

        per_user_selected = []
        per_user_importances = {
            "relevance": [],
            "diversity": [],
            "novelty": []
        }

        other_participations = Participation.query.filter((Participation.id != session["participation_id"]) & (Participation.time_finished != None) & (Participation.user_study_id == session["user_study_id"])).all()
        for participation in other_participations:
            # Get all interactions for the given participation
            study_ended_interaction = Interaction.query.filter((Interaction.participation == participation.id) & (Interaction.interaction_type == "study-ended")).order_by(Interaction.id.desc()).first()
            if not study_ended_interaction:
                continue
            ended_iteration = json.loads(study_ended_interaction.data)["iteration"]
            iterations_ended = Interaction.query.filter((Interaction.participation == participation.id) & (Interaction.interaction_type == "iteration-ended")).order_by(Interaction.id.desc()).all()
            iterations_started = Interaction.query.filter((Interaction.participation == participation.id) & (Interaction.interaction_type == "iteration-started")).order_by(Interaction.id.desc()).all()
            if iterations_ended:

                for it in iterations_ended:
                    if json.loads(it.data)["iteration"] == ended_iteration:
                        last_iteration_ended = it
                        break

                for it in iterations_started:
                    if it.id < last_iteration_ended.id:
                        last_iteration_started = it

                if last_iteration_ended and last_iteration_started:
                    if last_iteration_ended.id <= last_iteration_started.id:
                        print(f"Something weird happens for user with participation id={participation.id}")
                    else:
                        data = json.loads(last_iteration_ended.data)
                        user_selected = sum(data["selected"], [])
                        
                        start_data = json.loads(last_iteration_started.data)
                        user_shown = []

                        for _, res in start_data["shown"].items():
                            user_shown.extend(sum(res, []))

                        per_user_selected.append(user_selected) # Convert 2d list to 1d list and append
                        print(f"!!!! participation_id={participation.id}")
                        user_importances = get_objective_importance(user_selected, user_shown)
                        if user_importances:
                            for obj_name, importance in user_importances.items():
                                per_user_importances[obj_name].append(importance)
                else:
                    print(f"Wird for participation: {participation.id}")



        rel_input = np.array(per_user_importances["relevance"] + [objective_importance["relevance"]])
        div_input = np.array(per_user_importances["diversity"] + [objective_importance["diversity"]])
        nov_input = np.array(per_user_importances["novelty"] + [objective_importance["novelty"]])

        if len(per_user_importances["relevance"]) and len(per_user_importances["diversity"]) and len(per_user_importances["novelty"]) and rel_input.size and div_input.size and nov_input.size:
            rel_input = rel_input.reshape(-1, 1)
            rel = QuantileTransformer().fit_transform(rel_input)
            params["relevance_percent"] = round(rel[-1, 0].item() * 100.0, 1)

            div_input = div_input.reshape(-1, 1)
            div = QuantileTransformer().fit_transform(div_input)
            params["diversity_percent"] = round(div[-1, 0].item() * 100.0, 1)

            nov_input = nov_input.reshape(-1, 1)
            nov = QuantileTransformer().fit_transform(nov_input)
            params["novelty_percent"] = round(nov[-1, 0].item() * 100.0, 1)

            params["avg_relevance"] = round(rel_input.mean(), 1)
            params["avg_diversity"] = round(div_input.mean(), 1)
            params["avg_novelty"] = round(nov_input.mean(), 1)


            params["show_extra_statistics"] = True
        else:
            params["show_extra_statistics"] = False

    else:
        params["show_extra_statistics"] = False
    
    return render_template("finished_user_study.html", **params)

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
    weights = np.array([0.33, 0.33, 0.33]) #calculate_weight_estimate(selected_movies, flask.session["elicitation_movies"])
    weights /= weights.sum()
    session["weights"] = weights.tolist()

    algorithms = ["relevance_based", "weighted_average"]
    # Add default entries so that even the non-chosen algorithm has an empty entry
    # to unify later access
    recommendations = {
        algo: [[]] for algo in algorithms
    }
    
    # We filter out everything the user has selected during preference elicitation.
    # However, we allow future recommendation of SHOWN, NOT SELECTED (during elicitation, not comparison) altough these are quite rare
    filter_out_movies = selected_movies

    # Order of insertion should be preserved
    recommended_items, model = recommend_2_3(selected_movies, filter_out_movies, return_model=True, k=k)
    for algorithm in algorithms:
        if algorithm == "relevance_based":
            pass
        elif algorithm == "rlprop":
            recommended_items = rlprop(selected_movies, model, weights, filter_out_movies, k=k)
        elif algorithm == "weighted_average":
            recommended_items = weighted_average(selected_movies, model, weights, filter_out_movies, k=k)
        else:
            assert False
        recommendations[algorithm] = [recommended_items]

    
    session["movies"] = recommendations

    session["iteration"] = 1
    # TODO store all these information in DB as well
    session["elicitation_selected_movies"] = selected_movies

    # TODO This is layoutshuffling specific, move it there
    session["selected_movie_indices"] = [] #dict() # For each iteration, we can store selected movies
    session["selected_variants"] = []
    session["nothing"] = []
    session["cmp"] = []
    session["a_r"] = []

    # Build permutation
    # We choose 4 out of 6 result layout variants
    # Randomly generate order of algorithms for each of them and append second list with inverses.
    p = np.random.permutation(len(result_layout_variants))[:4]
    r = np.random.randint(size=p.shape, low=0, high=2)
    rnd_order = r.tolist() + (1 - r).tolist()
    p = p.tolist() * 2
    assert len(algorithms) == 2 # This implementation only works for 2 algorithms
    algo_order = []
    for j in range(len(rnd_order)):
        d = {}
        for i, algorithm in enumerate(algorithms):
            if i == 0:
                d[algorithm] = rnd_order[j]
            else:
                d[algorithm] = 1 - rnd_order[j]
        algo_order.append({
            "result_layout": p[j],
            "order": d
        })

    session["permutation"] = algo_order #np.random.permutation(len(result_layout_variants)).tolist()
    session["orig_permutation"] = algo_order # Backup, read-only
    return redirect(url_for("layoutshuffling.compare_algorithms"))

from multiprocessing import Process
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

def long_initialization(guid):
    # Activate the user study once the initialization is done
    # We have to use SQLAlchemy directly because we are outside of the Flask context (since we are running on a daemon thread)
    engine = create_engine('sqlite:///instance/db.sqlite')
    session = Session(engine)
    q = session.query(UserStudy).filter(UserStudy.guid == guid).first()
    
    # Do a single call to load_ml_dataset and prepare_tf_data to force cache population
    loader = load_ml_dataset()
    prepare_tf_model(loader)
    
    q.initialized = True
    q.active = True
    session.commit()
    session.expunge_all()
    session.close()

@bp.route("/initialize", methods=["GET"])
def initialize():
    guid = request.args.get("guid")
    heavy_process = Process(
        target=long_initialization,
        daemon=True,
        args=(guid, )
    )
    heavy_process.start()
    return redirect(request.args.get("continuation_url"))

# Plugin specific disposal procedure
# E.g. removing plugin-specific cache etc.
# Leaving empty if no disposal is needed
@bp.route("/dispose", methods=["DELETE"])
def dispose():
    return "OK"

def register():
    return {
        "bep": dict(blueprint=bp, prefix=None),
        # "hep": dict(before_request=limit_handler)
    }
