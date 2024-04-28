# This plugin is for a user study which aims at comparing 2 or 3 algoritms
# We currently support comparison of all algorithms implementing our AlgorithmBase interface
# We also provide a wrapper for Lenskit which automatically makes all Lenskit algorithms supported
# If anyone adds new algorithm, it is enough to implement AlgorithmBase
# Also there can be new wrappers that will provide algorithms from different libraries

import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
[sys.path.append(i) for i in ['.', '..']]
[sys.path.append(i) for i in ['../.', '../..', '../../.']]

from plugins.utils.preference_elicitation import load_data, enrich_results
from plugins.fastcompare.loading import load_algorithms, load_preference_elicitations, load_data_loaders
from plugins.utils.interaction_logging import log_interaction, study_ended

from models import UserStudy


import os
from flask import Blueprint, jsonify, request, redirect, url_for, make_response, render_template, session

from common import get_tr, load_languages, multi_lang, load_user_study_config


__plugin_name__ = "fastcompare"
__version__ = "0.1.0"
__author__ = "Patrik Dokoupil"
__author_contact__ = "Patrik.Dokoupil@matfyz.cuni.cz"
__description__ = "Fast and easy comparison of 2 or 3 RS algorithms on implicit feedback datasets."

bp = Blueprint(__plugin_name__, __plugin_name__, url_prefix=f"/{__plugin_name__}")

languages = load_languages(os.path.dirname(__file__))


HIDE_LAST_K = 1000000 # Effectively hides everything

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
    tr = get_tr(languages, get_lang())
    params = {
        "select_num_algorithms": tr("fastcompare_create_select_num_algorithms"),
        "select_result_layout": tr("fastcompare_create_select_result_layout"),
        "shuffle_recommendations": tr("fastcompare_create_shuffle_recommendations"),
        "shuffle_algorithms": tr("fastcompare_create_shuffle_algorithms"),
        "select_algorithm": tr("fastcompare_create_select_algorithm"),
        "create": tr("fastcompare_create_create"),
        "cancel": tr("fastcompare_create_cancel"),
        "rows": tr("fastcompare_create_rows"),
        "row_single": tr("fastcompare_create_row_single"),
        "row_single_scrollable": tr("fastcompare_create_row_single_scrollable"),
        "columns": tr("fastcompare_create_columns"),
        "column_single": tr("fastcompare_create_column_single"),
        "max_columns": tr("fastcompare_create_max_columns"),
        "settings": tr("fastcompare_create_settings"),
        "recommendation_layout": tr("fastcompare_create_recommendation_layout"),
        "recommendation_layout_hint": tr("fastcompare_create_recommendation_layout_hint"),
        "number_of_iterations": tr("fastcompare_create_number_of_iterations"),
        "number_of_iterations_hint": tr("fastcompare_create_number_of_iterations_hint"),
        "please_enter_k": tr("fastcompare_create_please_enter_k"),
        "please_enter_prolific": tr("fastcompare_create_please_enter_prolific"),
        "please_enter_n_iterations": tr("fastcompare_create_please_enter_n_iterations"),
        "number_of_algorithms": tr("fastcompare_create_number_of_algorithms"),
        "number_of_algorithms_hint": tr("fastcompare_create_number_of_algorithms_hint"),
        "prolific_code": tr("fastcompare_create_prolific_code"),
        "prolific_code_hint": tr("fastcompare_create_prolific_code_hint"),
        "recommendation_size": tr("fastcompare_create_recommendation_size"),
        "recommendation_size_hint": tr("fastcompare_create_recommendation_size_hint"),
        "more_info": tr("fastcompare_create_more_info"),
        "administration": tr("fastcompare_create_administration"),
        "preference_elicitation": tr("fastcompare_create_preference_elicitation"),
        "preference_elicitation_hint": tr("fastcompare_create_preference_elicitation_hint"),
        "data_loader": tr("fastcompare_create_data_loader"),
        "data_loader_hint": tr("fastcompare_create_data_loader_hint"),
        "select_elicitation": tr("fastcompare_create_select_elicitation"),
        "select_data_loader": tr("fastcompare_create_select_data_loader"),
        "displayed_name": tr("fastcompare_create_displayed_name"),
        "displayed_name_help": tr("fastcompare_create_displayed_name_help"),
        "override_about": tr("fastcompare_create_override_about"),
        "override_informed_consent": tr("fastcompare_create_override_informed_consent"),
        "override_preference_elicitation_hint": tr("fastcompare_create_override_preference_elicitation_hint"),
        "override_algorithm_comparison_hint": tr("fastcompare_create_override_algorithm_comparison_hint"),
        "override_finished_text": tr("fastcompare_create_override_finished_text"),
        "provide_footer": tr("fastcompare_create_provide_footer"),
        "show_final_statistics": tr("fastcompare_create_show_final_statistics"),
        "footer_placeholder": tr("fastcompare_create_footer_placeholder"),
        "about_placeholder": tr("fastcompare_create_about_placeholder"),
        "informed_consent_placeholder": tr("fastcompare_create_informed_consent_placeholder"),
        "preference_elicitation_placeholder": tr("fastcompare_create_preference_elicitation_placeholder"),
        "algorithm_comparison_placeholder": tr("fastcompare_create_algorithm_comparison_placeholder"),
        "finished_text_placeholder": tr("fastcompare_create_finished_text_placeholder")
    }
    return render_template("fastcompare_create.html", **params)

@bp.route("/available-algorithms")
def available_algorithms():
    res = [
        {
            "name": x.name(),
            "parameters": x.parameters()
        }
        for x in load_algorithms().values()
    ]
    tr = get_tr(languages, get_lang())
    for data in res:
        for p in data["parameters"]:
            if "help_key" in p and p.help_key:
                setattr(p, "help", tr(p.help_key)) # Translate help
                p["help"] = p.help
    return res

@bp.route("/available-preference-elicitations")
def available_preference_elicitations():
    res = [
        {
            "name": x.name(),
            "parameters": x.parameters()
        }
        for x in load_preference_elicitations().values()
    ]
    tr = get_tr(languages, get_lang())
    for data in res:
        for p in data["parameters"]:
            if "help_key" in p and p.help_key:
                setattr(p, "help", tr(p.help_key)) # Translate help
                p["help"] = p.help
    return res

@bp.route("/available-data-loaders")
def available_data_loaders():
    res = [
        {
            "name": x.name(),
            "parameters": x.parameters()
        }
        for x in load_data_loaders().values()
    ]
    tr = get_tr(languages, get_lang())
    for data in res:
        for p in data["parameters"]:
            if "help_key" in p and p.help_key:
                setattr(p, "help", tr(p.help_key)) # Translate help
                p["help"] = p.help
    return res    

@bp.route("/get-initial-data", methods=["GET"])
def get_initial_data():
    # Get already shown movies
    el_movies = session["elicitation_movies"]

    config = load_user_study_config(session["user_study_id"])
    
    loader_factory = load_data_loaders()[config["selected_data_loader"]]
    loader = loader_factory(**filter_params(config["data_loader_parameters"], loader_factory))
    load_data_loader(loader, session["user_study_guid"], loader_factory.name())
    
    elicitation_factory = load_preference_elicitations()[config["selected_preference_elicitation"]]
    elicitation = elicitation_factory(
        loader=loader,
        **filter_params(config["preference_elicitation_parameters"], elicitation_factory)
    )
    load_preference_elicitation(elicitation, session["user_study_guid"], elicitation_factory.name())

    x = load_data(loader, elicitation, el_movies)

    tr = get_tr(languages, get_lang())

    for i in range(len(x)):
        input_name = f"{config['selected_data_loader']}_{x[i]['movie_id']}"
        x[i]["movie"] = tr(input_name, x[i]['movie']) + " " + "|".join([tr(f"genre_{y.lower()}") for y in x[i]["genres"]])
    
    el_movies.extend(x)
    session["elicitation_movies"] = el_movies

    # TODO to do lazy loading, return just X and update rows & items in JS directly
    return jsonify(el_movies)

# Public facing endpoint
@bp.route("/join", methods=["GET"])
@multi_lang
def join():
    assert "guid" in request.args, "guid must be available in arguments"
    return redirect(url_for("utils.join", continuation_url=url_for(f"{__plugin_name__}.on_joined"), **request.args))

# Callback once user has joined we forward to preference elicitation
@bp.route("/on-joined", methods=["GET", "POST"])
def on_joined():
    return redirect(url_for(
        "utils.preference_elicitation",
        continuation_url=url_for(f"{__plugin_name__}.send_feedback"),
        consuming_plugin=__plugin_name__,
        initial_data_url=url_for(f"{__plugin_name__}.get_initial_data"),
        search_item_url=url_for(f'{__plugin_name__}.item_search')
    ))

def search_for_item(pattern, tr=None):
    conf = load_user_study_config(session["user_study_id"])
    
    ## TODO get_loader helper
    loader_factory = load_data_loaders()[conf["selected_data_loader"]]
    loader = loader_factory(**filter_params(conf["data_loader_parameters"], loader_factory))
    load_data_loader(loader, session["user_study_guid"], loader_factory.name())

    # If we have a translate function
    if tr:
        found_items = loader.items_df[loader.items_df.item_id.astype(str).map(tr).str.contains(pattern, case=False)]
    else:
        found_items = loader.items_df[loader.items_df.title.str.contains(pattern, case=False)]
    
    item_indices = [loader.get_item_index(item_id) for item_id in found_items.item_id.values]
    return enrich_results(item_indices, loader)

@bp.route("/item-search", methods=["GET"])
def item_search():
    print('caalled')
    pattern = request.args.get("pattern")
    if not pattern:
        return make_response("", 404)
    
    lang = get_lang()
    if lang == "en":
        tr = None
    else:
        tr = get_tr(languages, lang)
    res = search_for_item(pattern, tr)

    return jsonify(res)

def prepare_recommendations(loader, conf, recommendations, selected_movies, filter_out_movies, k):
    algorithm_factories = load_algorithms()
    algorithms = conf["selected_algorithms"]
    print(f"Algorithm factories = {algorithm_factories}")
    for algorithm_idx, algorithm_name in enumerate(algorithms):
        # Construct the algorithm
        algorithm_displayed_name = conf["algorithm_parameters"][algorithm_idx]["displayed_name"]
        factory = algorithm_factories[algorithm_name]
        algorithm = factory(loader, **filter_params(conf["algorithm_parameters"][algorithm_idx], factory))
        load_algorithm(algorithm, session["user_study_guid"], algorithm_displayed_name)
        recommended_items = algorithm.predict(selected_movies, filter_out_movies, k=k)

        if conf["shuffle_recommendations"]:
            np.random.shuffle(recommended_items)

        recommendations[algorithm_displayed_name][-1] = enrich_results(recommended_items, loader)

    return recommendations

# Receives arbitrary feedback (CALLED from preference elicitation) and generates recommendation
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

    # Add default entries so that even the non-chosen algorithm has an empty entry
    # to unify later access
    recommendations = {
        x["displayed_name"]: [[]] for x in conf["algorithm_parameters"]
    }
    
    # We filter out everything the user has selected during preference elicitation.
    # However, we allow future recommendation of SHOWN, NOT SELECTED (during elicitation, not comparison) altough these are quite rare
    filter_out_movies = selected_movies

    loader_factory = load_data_loaders()[conf["selected_data_loader"]]
    loader = loader_factory(**filter_params(conf["data_loader_parameters"], loader_factory))
    load_data_loader(loader, session["user_study_guid"], loader_factory.name())
    prepare_recommendations(loader, conf, recommendations, selected_movies, filter_out_movies, k)

    
    # Initialize session data
    print(f"SEtting session movies with: {recommendations}")
    session["movies"] = recommendations
    session["iteration"] = 1
    session["elicitation_selected_movies"] = selected_movies
    session["selected_movie_indices"] = [] #dict() # For each iteration, we can store selected movies
    session["selected_variants"] = []
    session["nothing"] = []
    session["cmp"] = []
    session["a_r"] = []

    ### Prepare permutation ###
    # Possible order values
    p = []
    for i in range(conf["n_iterations"]):
        orders = dict()
        available_orders = list(range(conf["n_algorithms_to_compare"]))
        for i, algorithm in enumerate(conf["selected_algorithms"]):
            algorithm_displayed_name = conf["algorithm_parameters"][i]["displayed_name"]
            
            if conf["shuffle_algorithms"]:
                order_idx = np.random.randint(len(available_orders))
            else:
                order_idx = 0

            orders[algorithm_displayed_name] = available_orders[order_idx]
            del available_orders[order_idx]

        p.append(orders)
    session["permutation"] = p
    session["orig_permutation"] = p
    return redirect(url_for(f"{__plugin_name__}.compare_algorithms"))

def elicitation_ended(elicitation_movies, elicitation_selected_movies):
    data = {
        "elicitation_movies": elicitation_movies,
        "elicitation_selected_movies": elicitation_selected_movies
    }
    log_interaction(session["participation_id"], "elicitation-ended", **data)

def iteration_started(iteration, movies, algorithm_assignment, result_layout, shown_movie_indices):
    data = {
        "iteration": iteration,
        "movies": movies,
        "algorithm_assignment": algorithm_assignment,
        "result_layout": result_layout,
        "shown": shown_movie_indices
    }
    log_interaction(session["participation_id"], "iteration-started", **data)

def iteration_ended(iteration, selected, selected_variants, dont_like_anything, algorithm_comparison, ordered_ratings):
    data = {
        "iteration": iteration,
        "selected": selected,
        "selected_variants": selected_variants,
        "dont_like_anything": dont_like_anything,
        "algorithm_comparison": algorithm_comparison,
        "ratings": ordered_ratings
    }
    log_interaction(session["participation_id"], "iteration-ended", **data)

@bp.route("/compare-algorithms", methods=["GET"])
def compare_algorithms():
    
    if session["iteration"] == 1:
        # TODO move to utils
        elicitation_ended(session["elicitation_movies"], session["elicitation_selected_movies"])    
        pass

    conf = load_user_study_config(session["user_study_id"])
    algorithm_assignment = {}
    movies = {}

    p = session["permutation"][0]

    
    for i, algorithm in enumerate(conf["selected_algorithms"]):
        algorithm_displayed_name = conf["algorithm_parameters"][i]["displayed_name"]
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

    result_layout = conf["result_layout"]

    # In some sense, we can treat this as iteration start
    # TODO fix that we have two algorithms, add weights and fix algorithm_assignment (randomly assigning with each iteration)
    shown_movie_indices = {}
    for algo_name, movie_lists in session["movies"].items():
        shown_movie_indices[algo_name] = [[int(x["movie_idx"]) for x in movie_list] for movie_list in movie_lists]
        
    iteration_started(session["iteration"], movies, algorithm_assignment, result_layout, shown_movie_indices)

    tr = get_tr(languages, get_lang())
    for d in movies.values():
        x = d["movies"]
        for i in range(len(x)):
            input_name = f"{conf['selected_data_loader']}_{x[i]['movie_id']}"
            x[i]["movie"] = tr(input_name, x[i]['movie']) + " " + "|".join([tr(f"genre_{y.lower()}") for y in x[i]["genres"]])
            #x[i]["movie"] = tr(str(x[i]["movie_id"])) + " " + "|".join([tr(f"genre_{y.lower()}") for y in x[i]["genres"]])

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

    # Handle textual overrides
    params["comparison_hint_override"] = None
    params["footer_override"] = None
    if "text_overrides" in conf:
        if "comparison_hint" in conf["text_overrides"]:
            params["comparison_hint_override"] = conf["text_overrides"]["comparison_hint"]

        if "footer" in conf["text_overrides"]:
            params["footer_override"] = conf["text_overrides"]["footer"]

    return render_template("compare_algorithms.html", **params)


# We received feedback from compare_algorithms.html
@bp.route("/algorithm-feedback")
def algorithm_feedback():
    # TODO do whatever with the passed parameters and set session variable

    conf = load_user_study_config(session["user_study_id"])
    
    ## TODO get_loader helper
    loader_factory = load_data_loaders()[conf["selected_data_loader"]]
    loader = loader_factory(**filter_params(conf["data_loader_parameters"], loader_factory))
    load_data_loader(loader, session["user_study_guid"], loader_factory.name())

    selected_movies = request.args.get("selected_movies")
    selected_movies = selected_movies.split(",") if selected_movies else []

    selected_variants = request.args.get("selected_movie_variants")
    selected_variants = selected_variants.split(",") if selected_variants else []
    selected_variants = [int(x) for x in selected_variants]

    algorithm_ratings = []
    for i in range(conf["n_algorithms_to_compare"]):
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


    iteration_ended(session["iteration"], session["selected_movie_indices"], session["selected_variants"], session["nothing"], session["cmp"], session["a_r"])    
    # Increase iteration
    session["iteration"] += 1
    ### And generate new recommendations ###
    
    mov = session["movies"]

    lengths = []
    algorithm_displayed_names = []
    for x in conf["algorithm_parameters"]:
        mov[x["displayed_name"]].append([])
        lengths.append(len(mov[x["displayed_name"]]))
        algorithm_displayed_names.append(x["displayed_name"])

    assert len(set(lengths)), "All algorithms should share the number of iterations"
    n_iterations = lengths[0] # Since all have same number of iteration, pick the first one

    mov_indices = []
    for i in range(n_iterations):
        indices = set()
        for algo_displayed_name in algorithm_displayed_names:
            indices.update([int(y["movie_idx"]) for y in mov[algo_displayed_name][i]])
        mov_indices.append(list(indices))

    
    filter_out_movies = session["elicitation_selected_movies"] + sum(mov_indices[:HIDE_LAST_K], [])
    selected_movies = session["elicitation_selected_movies"] + sum(session["selected_movie_indices"], [])
    
    mov = prepare_recommendations(loader, conf, mov, selected_movies, filter_out_movies, k=session["rec_k"])

    session["movies"] = mov
    ### End generation ###


    # And shift the permutation
    permutation = session["permutation"]
    permutation = permutation[1:] + permutation[:1] # Move first item to the end
    session["permutation"] = permutation
    return redirect(url_for(f"{__plugin_name__}.compare_algorithms"))

from multiprocessing import Process
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


#### Loading extensible components from the cache ######
def get_cache_path(guid, name=""):
    return os.path.join("cache", __plugin_name__, guid, name)

def load_algorithm(algorithm, guid, algorithm_displayed_name):
    algorithm.load(get_cache_path(guid, algorithm_displayed_name), get_cache_path("", algorithm_displayed_name))

# Elicitation may have some internal state as well, so we load it as well
def load_preference_elicitation(elicitation, guid, elicitation_name):
    elicitation.load(get_cache_path(guid, elicitation_name), get_cache_path("", elicitation_name))

# Dataset loaders also may have internal state, load them as well
def load_data_loader(data_loader, guid, loader_name):
    data_loader.load(get_cache_path(guid, loader_name), get_cache_path("", loader_name))

# There could be a missmatch in-between parameters passed here and what is declared by actual component factory
# Filter out all temporary params (e.g. displayed_name) that are not directly connected to the underlying algorithm class
def filter_params(actual_parameters, factory):
    params = set([x.name for x in factory.parameters()])
    return {k: v for k, v in actual_parameters.items() if k in params}

### Long running initialization is here ####
def long_initialization(guid):
    # Activate the user study once the initialization is done
    # We have to use SQLAlchemy directly because we are outside of the Flask context (since we are running on a daemon thread)
    engine = create_engine('sqlite:///instance/db.sqlite')
    session = Session(engine)
    q = session.query(UserStudy).filter(UserStudy.guid == guid).first()
    try:
        conf = json.loads(q.settings)

        # Ensure cache directory exists
        Path(get_cache_path(guid)).mkdir(parents=True, exist_ok=True)
        
        # Prepare data loader first
        loader_factory = load_data_loaders()[conf["selected_data_loader"]]
        loader = loader_factory(**filter_params(conf["data_loader_parameters"], loader_factory))
        loader.load_data() # Actually load the data
        loader.save(get_cache_path(guid, loader.name()), get_cache_path("", loader.name())) # Save the data loader itself to the cache

        # Then preference elicitation
        elicitation_factory = load_preference_elicitations()[conf["selected_preference_elicitation"]]
        elicitation = elicitation_factory(
            loader=loader, # Pass in the loader
            **filter_params(conf["preference_elicitation_parameters"], elicitation_factory)
        )
        elicitation.fit()
        elicitation.save(get_cache_path(guid, elicitation.name()), get_cache_path("", elicitation.name()))

        # Prepare algorithms
        algorithms = conf["selected_algorithms"]
        algorithm_factories = load_algorithms()
        for algorithm_idx, algorithm_name in enumerate(algorithms):
            # Construct the algorithm with parameters from config
            # And construct the algorithm
            factory = algorithm_factories[algorithm_name]
            algorithm = factory(loader, **filter_params(conf["algorithm_parameters"][algorithm_idx], factory))
            algorithm_displayed_name = conf["algorithm_parameters"][algorithm_idx]["displayed_name"]
            
            print(f"Training algorithm: {algorithm_displayed_name}")
            algorithm.fit()
            print(f"Done training algorithm: {algorithm_displayed_name}")

            # Save the algorithm
            algorithm.save(get_cache_path(guid, algorithm_displayed_name), get_cache_path("", algorithm_displayed_name))

        q.initialized = True
        q.active = True
    except Exception as e:
        q.initialization_error = traceback.format_exc()

    session.commit()
    session.expunge_all()
    session.close()


@bp.route("/initialize", methods=["GET"])
def initialize():
    print("Called here, do whatever initialization")
    guid = request.args.get("guid")
    heavy_process = Process(
        target=long_initialization,
        daemon=True,
        args=(guid, )
    )
    heavy_process.start()
    print("Going to redirect back")
    return redirect(request.args.get("continuation_url"))

# Plugin specific disposal procedure
@bp.route("/dispose", methods=["DELETE"])
def dispose():
    guid = request.args.get("guid")
    p = get_cache_path(guid)
    if os.path.exists(p):
        shutil.rmtree(p)
    return "OK"

@bp.route("/finish-user-study")
@multi_lang
def finish_user_study():
    # Last iteration has ended here
    iteration_ended(session["iteration"], session["selected_movie_indices"], session["selected_variants"], session["nothing"], session["cmp"], session["a_r"])
    return redirect(url_for("utils.finish"))

def register():
    return {
        "bep": dict(blueprint=bp, prefix=None),
        # "hep": dict(before_request=limit_handler)
    }
