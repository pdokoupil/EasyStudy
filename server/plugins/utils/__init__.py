# -*- coding: utf-8 -*-

import os
import secrets
from flask import Blueprint, jsonify, request, url_for, make_response, render_template
from flask_login import current_user
from common import get_tr, load_languages, multi_lang, load_user_study_config, load_user_study_config_by_guid
import flask
import json

import datetime

from models import Interaction, Participation, Message, UserStudy
from app import db

from .interaction_logging import study_ended

__plugin_name__ = "utils"
__description__ = "Plugin containing common, shared functionality that can be used from other plugins."
__version__ = "0.1.0"
__author__ = "Patrik Dokoupil"
__author_contact__ = "Patrik.Dokoupil@matfyz.cuni.cz"

bp = Blueprint(__plugin_name__, __plugin_name__, url_prefix=f"/{__plugin_name__}")

from .preference_elicitation import load_data_1, load_data_2, load_data_3, recommend_2_3, search_for_movie, rlprop, weighted_average, calculate_weight_estimate, result_layout_variants, load_data

NUM_TO_SELECT = 5

@bp.context_processor
def plugin_name():
    return {
        "plugin_name": __plugin_name__
    }

languages = load_languages(os.path.dirname(__file__))

def get_lang():
    default_lang = "en"
    if "lang" in flask.session and flask.session["lang"] and flask.session["lang"] in languages:
        return flask.session["lang"]
    return default_lang

# Shared implementation of "/join" phase of the user study
# Expected input is continuation_url
# Expected output is 
@bp.route("/join", methods=["GET"])
@multi_lang
def join():
    assert "continuation_url" in request.args, f"Continuation url must be available: {request.args}"
    assert "guid" in request.args, f"Guid must be available: {request.args}"

    params = dict(request.args)
    params["email"] = current_user.email if current_user.is_authenticated else ""
    params["lang"] = get_lang()

    tr = get_tr(languages, get_lang())
    params["title"] = tr("join_title")
    params["contacts"] = tr("footer_contacts")
    params["contact"] = tr("footer_contact")
    params["charles_university"] = tr("footer_charles_university")
    params["cagliari_university"] = tr("footer_cagliari_university")
    params["t1"] = tr("footer_t1")
    params["t2"] = tr("footer_t2")
    params["participant_details"] = tr("join_participant_details")
    params["please_enter_details"] = tr("join_please_enter_details")
    params["about_study"] = tr("join_about_study")
    params["study_details"] = tr("join_study_details")
    params["enter_email"] = tr("join_enter_email")
    params["enter_email_hint"] = tr("join_enter_email_hint")
    params["enter_gender"] = tr("join_enter_gender")
    params["enter_gender_hint"] = tr("join_enter_gender_hint")
    params["enter_age"] = tr("join_enter_age")
    params["enter_age_hint"] = tr("join_enter_age_hint")
    params["enter_education"] = tr("join_enter_education")
    params["enter_education_hint"] = tr("join_enter_education_hint")
    params["enter_ml_familiar"] = tr("join_enter_ml_familiar")
    params["enter_ml_familiar_hint"] = tr("join_enter_ml_familiar_hint")
    params["gender_male"] = tr("join_gender_male")
    params["gender_female"] = tr("join_gender_female")
    params["gender_other"] = tr("join_gender_other")
    params["education_no_formal"] = tr("join_education_no_formal")
    params["education_primary"] = tr("join_education_primary")
    params["education_high"] = tr("join_education_high")
    params["education_bachelor"] = tr("join_education_bachelor")
    params["education_master"] = tr("join_education_master")
    params["education_doctoral"] = tr("join_education_doctoral")
    params["yes"] = tr("yes")
    params["no"] = tr("no")
    params["informed_consent_header"] = tr("join_informed_consent_header")
    params["informed_consent_p1"] = tr("join_informed_consent_p1")
    params["informed_consent_p2"] = tr("join_informed_consent_p2")
    params["informed_consent_p3"] = tr("join_informed_consent_p3")
    params["informed_consent_p31"] = tr("join_informed_consent_p31")
    params["informed_consent_p32"] = tr("join_informed_consent_p32")
    params["informed_consent_p33"] = tr("join_informed_consent_p33")
    params["informed_consent_p4"] = tr("join_informed_consent_p4")
    params["informed_consent_p5"] = tr("join_informed_consent_p5")
    params["informed_consent_p6"] = tr("join_informed_consent_p6")
    params["start_user_study"] = tr("join_start_user_study")
    params["guid_not_found"] = tr("join_guid_not_found")
    params["server_error"] = tr("join_server_error")
    params["min_resolution_error"] = tr("join_min_resolution_error")
    params["czech"] = tr("join_czech")
    params["english"] = tr("join_english")

    study = UserStudy.query.filter(UserStudy.guid == request.args.get("guid")).first()
    if not study.initialized:
        params["hint_lead"] = tr("error_hint_lead")
        params["header"] = tr("error_header")
        params["hint"] = tr("error_hint")
        params["alert_text"] = tr("error_not_initialized")
        return render_template("error.html", **params)
    if not study.active:
        params["hint_lead"] = tr("error_hint_lead")
        params["header"] = tr("error_header")
        params["hint"] = tr("error_hint")
        params["alert_text"] = tr("error_not_active")
        return render_template("error.html", **params)

    if "uuid" not in flask.session:
        flask.session["uuid"] = secrets.token_urlsafe(16)

    # Handle prolific parameters
    if "PROLIFIC_PID" in request.args:
        flask.session["PROLIFIC_PID"] = request.args.get("PROLIFIC_PID")
        flask.session["PROLIFIC_STUDY_ID"] = request.args.get("STUDY_ID")
        flask.session["PROLIFIC_SESSION_ID"] = request.args.get("SESSION_ID")
    else:
        if "PROLIFIC_PID" in flask.session:
            del flask.session["PROLIFIC_PID"]
        if "PROLIFIC_STUDY_ID" in flask.session:    
            del flask.session["PROLIFIC_STUDY_ID"]
        if "PROLIFIC_SESSION_ID" in flask.session:
            del flask.session["PROLIFIC_SESSION_ID"]

    # Handle textual overrides
    params["informed_consent_override"] = None
    params["about_override"] = None
    params["footer_override"] = None
    config = load_user_study_config_by_guid(request.args.get("guid"))
    if "text_overrides" in config:
        if "informed_consent" in config["text_overrides"]:
            params["informed_consent_override"] = config["text_overrides"]["informed_consent"]
        
        if "about" in config["text_overrides"]:
            params["about_override"] = config["text_overrides"]["about"]

        if "footer" in config["text_overrides"]:
            params["footer_override"] = config["text_overrides"]["footer"]
    
    return render_template("join.html", **params)

@bp.route("/preference-elicitation", methods=["GET", "POST"])
@multi_lang # TODO remove? and keep only in layoutshuffling
def preference_elicitation():

    assert 'continuation_url' in request.args, 'Continuation url must be provided by the consumer'
    assert 'initial_data_url' in request.args, 'Initial data url must be provided by the consumer'
    assert 'search_item_url' in request.args, 'Search item url must be provided by the consumer'

    config = load_user_study_config(flask.session["user_study_id"])
    
    impl = config["selected_preference_elicitation"] if "selected_preference_elicitation" in config else ""

    flask.session["elicitation_movies"] = []
    
    params = {
        "impl": impl,
        "consuming_plugin": request.args.get("consuming_plugin")
    }
    
    tr = get_tr(languages, get_lang())
    params["contacts"] = tr("footer_contacts")
    params["contact"] = tr("footer_contact")
    params["charles_university"] = tr("footer_charles_university")
    params["cagliari_university"] = tr("footer_cagliari_university")
    params["t1"] = tr("footer_t1")
    params["t2"] = tr("footer_t2")
    params["load_more"] = tr("elicitation_load_more")
    params["finish"] = tr("elicitation_finish")
    params["search"] = tr("elicitation_search")
    params["cancel_search"] = tr("elicitation_cancel_search")
    params["enter_name"] = tr("elicitation_enter_name")
    params["header"] = tr("elicitation_header")
    params["hint_lead"] = tr("elicitation_hint_lead")
    params["hint"] = tr("elicitation_hint")
    params["title"] = tr("elicitation_title")
    params["continuation_url"] = request.args.get("continuation_url") # Continuation url must be specified
    params["initial_data_url"] = request.args.get("initial_data_url")
    params["search_item_url"] = request.args.get("search_item_url")

    # Handle textual overrides
    params["elicitation_hint_override"] = None
    params["footer_override"] = None
    if "text_overrides" in config:
        if "elicitation_hint" in config["text_overrides"]:
            params["elicitation_hint_override"] = config["text_overrides"]["elicitation_hint"]    

        if "footer" in config["text_overrides"]:
            params["footer_override"] = config["text_overrides"]["footer"]

    return render_template("preference_elicitation.html", **params) # TODO remove hardcoded consuming plugin

@bp.route("/get-initial-data", methods=["GET"])
def get_initial_data():
    # Default implementation
    return cluster_data_1()

@bp.route("/cluster-data-1", methods=["GET"])
def cluster_data_1():
    #return json.dumps(load_data_1())
    el_movies = flask.session["elicitation_movies"]
    
    x = load_data_1(el_movies)

    tr = get_tr(languages, get_lang())
    

    for i in range(len(x)):
        x[i]["movie"] = tr(str(x[i]["movie_id"])) + " " + "|".join([tr(f"genre_{y.lower()}") for y in x[i]["genres"]])
    
    el_movies.extend(x)
    flask.session["elicitation_movies"] = el_movies

    # TODO to do lazy loading, return just X and update rows & items in JS directly
    return jsonify(el_movies)

@bp.route("/cluster-data-2", methods=["GET"])
def cluster_data_2():
    el_movies = flask.session["elicitation_movies"]
    
    x = load_data_2(el_movies)
    el_movies.extend(x)
    flask.session["elicitation_movies"] = el_movies
    return jsonify(el_movies)

@bp.route("/cluster-data-3", methods=["GET"])
def cluster_data_3():
    el_movies = flask.session["elicitation_movies"]
    
    x = load_data_3(el_movies)
    el_movies.extend(x)
    flask.session["elicitation_movies"] = el_movies

    return jsonify(el_movies)


@bp.route("/changed-viewport", methods=["POST"])
def changed_viewport():
    x = Interaction(
        participation = Participation.query.filter(Participation.id == flask.session["participation_id"]).first().id,
        interaction_type = "changed-viewport", #InteractionType.query.filter(InteractionType.name == "changed-viewport").first(),
        time = datetime.datetime.utcnow(),
        data = json.dumps(request.get_json(), ensure_ascii=False)
    )
    db.session.add(x)
    db.session.commit()

    return "OK"

@bp.route("/selected-item", methods=["POST"])
def selected_item():
    x = Interaction(
        participation = Participation.query.filter(Participation.id == flask.session["participation_id"]).first().id,
        interaction_type = "selected-item", #InteractionType.query.filter(InteractionType.name == "selected-item").first(),
        time = datetime.datetime.utcnow(),
        data = json.dumps(request.get_json(), ensure_ascii=False)
    )
    db.session.add(x)
    db.session.commit()
    return "OK"

@bp.route("/deselected-item", methods=["POST"])
def deselected_item():
    x = Interaction(
        participation = Participation.query.filter(Participation.id == flask.session["participation_id"]).first().id,
        interaction_type = "deselected-item", #InteractionType.query.filter(InteractionType.name == "deselected-item").first(),
        time = datetime.datetime.utcnow(),
        data = json.dumps(request.get_json(), ensure_ascii=False)
    )
    db.session.add(x)
    db.session.commit()
    return "OK"

@bp.route("/loaded-page", methods=["POST"])
def loaded_page():
    x = Interaction(
        participation = Participation.query.filter(Participation.id == flask.session["participation_id"]).first().id,
        interaction_type = "loaded-page", #InteractionType.query.filter(InteractionType.name == "loaded-page").first(),
        time = datetime.datetime.utcnow(),
        data = json.dumps(request.get_json(), ensure_ascii=False)
    )
    db.session.add(x)
    db.session.commit()
    return "OK"

@bp.route("/on-input", methods=["POST"])
def on_input():
    x = Interaction(
        participation = Participation.query.filter(Participation.id == flask.session["participation_id"]).first().id,
        interaction_type = "on-input",
        time = datetime.datetime.utcnow(),
        data = json.dumps(request.get_json(), ensure_ascii=False)
    )
    db.session.add(x)
    db.session.commit()
    return "OK"

@bp.route("/on-message", methods=["POST"])
def on_message():
    x = Message(
        time = datetime.datetime.utcnow(),
        data = json.dumps(request.get_json(), ensure_ascii=False)
    )

    if "participation_id" in flask.session:
        x.participation = Participation.query.filter(Participation.id == flask.session["participation_id"]).first().id

    db.session.add(x)
    db.session.commit()
    return "OK"

# TODO use from layoutshuffling statistics implementation as well
def prepare_basic_statistics(n_algorithms, algorithm_names):
    res = dict()

    counts = {
        x: 0 for x in algorithm_names
    }
    for variants, p in zip(flask.session["selected_variants"], flask.session["orig_permutation"]):
        # Convert "relevance_based":0 format to 0:"gamma"
        algo_idx_to_name = {idx: algo_name for algo_name, idx in p.items()}
        for v in variants:
            counts[algo_idx_to_name[v]] += 1

    avg_ratings = {
        x: 0.0 for x in algorithm_names
    }
    for ratings in flask.session["a_r"]:
        print(f"ratings={ratings}")
        for algo_name, rating in ratings.items():
            avg_ratings[algo_name] += rating
    avg_ratings = {algo_name : round(sum_ratings / len(flask.session["a_r"]), 1) if len(flask.session["a_r"]) else 0 for algo_name, sum_ratings in avg_ratings.items()}

    res["n_selected"] = sum([len(x) for x in flask.session["selected_movie_indices"]])
    res["n_recommended"] = int(flask.session["iteration"]) * flask.session["rec_k"] * n_algorithms
    res["n_selected_per_algorithm"] = counts
    res["n_avg_rating_per_algorithm"] = avg_ratings
    res["n_selected_elicitation"] = len(flask.session["elicitation_selected_movies"])
    res["n_total_elicitation"] = len(flask.session["elicitation_movies"])

    return res

# Shared implementation of "/finish" phase of the user study
@bp.route("/finish", methods=["GET", "POST"])
def finish():
    conf = load_user_study_config(flask.session["user_study_id"])
    study_ended(flask.session["participation_id"], iteration=flask.session["iteration"])

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
    params["selected_items"] = tr("finish_selected_items")
    params["out_of"] = tr("out_of")
    params["selected_per_algorithm"] = tr("finish_selected_per_algorithm")
    params["avg_rating_per_algorithm"] = tr("finish_avg_rating_per_algorithm")
    params["selected_during_elicitation"] = tr("finish_selected_during_elicitation")
    params["table_algo_name"] = tr("finish_table_algo_name")
    params["table_n_selected"] = tr("finish_table_n_selected")
    params["table_n_shown"] = tr("finish_table_n_shown")
    params["table_avg_rating"] = tr("finish_table_avg_rating")

    # Prolific stuff
    if "PROLIFIC_PID" in flask.session:
        params["prolific_pid"] = flask.session["PROLIFIC_PID"]
        params["prolific_url"] = f"https://app.prolific.co/submissions/complete?cc={conf['prolific_code']}"
    else:
        params["prolific_pid"] = None

    # Handle overrides
    params["finished_text_override"] = None
    params["footer_override"] = None
    if "text_overrides" in conf:
        if "finished_text" in conf["text_overrides"]:
            params["finished_text_override"] = conf["text_overrides"]["finished_text"]

        if "footer" in conf["text_overrides"]:
            params["footer_override"] = conf["text_overrides"]["footer"]

    # Handle statistics
    params["show_final_statistics"] = conf["show_final_statistics"]
    if conf["show_final_statistics"]:
        # Prepare statistics
        algorithm_names = [x["displayed_name"] for x in conf["algorithm_parameters"]]
        params.update(prepare_basic_statistics(conf["n_algorithms_to_compare"], algorithm_names))

    return render_template("finish.html", **params)
    
@bp.route("/movie-search", methods=["GET"])
def movie_search():
    attrib = flask.request.args.get("attrib")
    pattern = flask.request.args.get("pattern")
    if not attrib or attrib not in ["movie"]: # TODO extend search support
        return make_response("", 404)
    if not pattern:
        return make_response("", 404)
    
    lang = get_lang()
    if lang == "en":
        tr = None
    else:
        tr = get_tr(languages, lang)
    res = search_for_movie(attrib, pattern, tr)

    return flask.jsonify(res)

def register():
    return {
        "bep": dict(blueprint=bp, prefix=None),
        #"hep": dict(before_request=limit_handler)
    }