import json
import secrets
import sys

from plugins.utils.interaction_logging import log_interaction, study_ended
from models import Interaction

[sys.path.append(i) for i in ['.', '..']]
[sys.path.append(i) for i in ['../.', '../..', '../../.']]

from models import UserStudy

from common import get_abs_project_root_path, load_user_study_config


from multiprocessing import Process
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from flask import Blueprint, jsonify, request, redirect, render_template, url_for, session

import functools
import os
import numpy as np

from app import rds

import pickle


__plugin_name__ = "visualrepresentation2024"
__version__ = "0.1.0"
__author__ = "Patrik Dokoupil"
__author_contact__ = "Patrik.Dokoupil@matfyz.cuni.cz"
__description__ = "Comparing different visual representations of data"

bp = Blueprint(__plugin_name__, __plugin_name__, url_prefix=f"/{__plugin_name__}")

MIN_ITERATIONS = 5
N_ITERATIONS = 100

def get_uname():
    return f"user:{__plugin_name__}-{session['uuid']}"

# Wrapper for setting values, performs serialization via pickle
def set_val(key, val):
    name = get_uname()
    rds.hset(name, key, value=pickle.dumps(val))

# Wrapper for getting values, performs deserialization via pickle
def get_val(key):
    name = get_uname()
    return pickle.loads(rds.hget(name, key))


# Uncomment this endpoint to make plugin visible in the administration
@bp.route("/create")
def create():
    params = {
        "footer_placeholder": "",
        "about_placeholder": "",
        "informed_consent_placeholder": "",
        "algorithm_comparison_placeholder": "",
        "finished_text_placeholder": "",
        "override_footer": "Override Footer",
        "override_about": "Override About",
        "override_informed_consent": "Override Informed Consent",
        "override_algorithm_comparison_hint": "Override Selection Instructions",
        "override_finished_text": "Override Final Text"
    }
    return render_template("visual_representation_create.html", **params)

from flask import send_file, current_app
from jinja2 import FileSystemLoader, Environment
from markupsafe import Markup

loader = FileSystemLoader('./')


def include_file(name):
    print("OK")
    return Markup(loader.get_source(env, name)[0])

env = Environment(loader=loader)
env.globals['include_file'] = include_file

@bp.context_processor
def plugin_name():
    return {
        "plugin_name": __plugin_name__,
        "include_file": include_file
    }

@bp.route("/table-data-example", methods=["GET"])
def table_data_example():
    example_name = request.args.get('name')
    p = session["permutation"][session["iteration"]]
    
    if p.method != "table-t":
        return "", 404
    
    if example_name != p.example.name:
        return "", 404

    return include_file('static/datasets/vizualizations/' + p.example.path.split(f'vizualizations/')[1])

@bp.route("/table-data", methods=["GET"])
def table_data():
    class_name = request.args.get('name')

    p = session["permutation"][session["iteration"]]
    if p.method != "table-t":
        return "", 404

    shown_classes = p.shown_classes
    for cls in shown_classes:
        if cls.name == class_name:
            return include_file('static/datasets/vizualizations/' + cls.class_image_path.split(f'vizualizations/')[1])

    return "", 404
#current_app.jinja_env.globals.update(include_file=include_file)

# Helpers
def iteration_ended(iteration, payload):
    data = {
        "iteration": iteration
    }
    data.update(**payload)
    log_interaction(session["participation_id"], "iteration-ended", **data)

def iteration_started(iteration, payload):
    data = {
        "iteration": iteration
    }
    data.update(**payload)
    log_interaction(session["participation_id"], "iteration-started", **data)



# Public facing endpoint
@bp.route("/join", methods=["GET"])
def join():
    assert "guid" in request.args, "guid must be available in arguments"
    return redirect(url_for("utils.join", continuation_url=url_for("visualrepresentation2024.on_joined"), **request.args))

# Callback once user has joined we forward to preference elicitation
@bp.route("/on-joined", methods=["GET", "POST"])
def on_joined():

    if "uuid" not in session:
        session["uuid"] = secrets.token_urlsafe(16)

    set_val("iteration", 0)

    data = load_configuration_json()
    selected_user = np.random.choice(list(data.keys()))

    set_val("selected_user", selected_user)
    set_val("selections", {})

    log_interaction(session["participation_id"], "user-selected", selected_user=selected_user, uuid=session["uuid"])

    return redirect(url_for(f"{__plugin_name__}.compare_visualizations",
            consuming_plugin=__plugin_name__
        )
    )

@bp.route("/compare-visualizations")
def compare_visualizations():
    iteration = get_val("iteration")
    selected_user = get_val("selected_user")
    data = load_configuration_json()

    iteration_data = data[selected_user][iteration]

    params = {
        "continuation_url": url_for(f"{__plugin_name__}.handle_feedback"),
        "finish_url": url_for(f"{__plugin_name__}.finish_user_study"),
        "iteration": iteration,
        "iteration_data": iteration_data,
        "n_iterations": len(data[selected_user])
    }

    conf = load_user_study_config(session["user_study_id"])
    if "text_overrides" in conf:
        if "comparison_hint" in conf["text_overrides"]:
            params["comparison_hint_override"] = conf["text_overrides"]["comparison_hint"]

        if "footer" in conf["text_overrides"]:
            params["footer_override"] = conf["text_overrides"]["footer"]

    payload = {
        "iteration_data": iteration_data
    }
    iteration_started(iteration, payload)

    return render_template("compare_visualizations.html", **params)

@bp.route("/submit-selections", methods=["POST"])
def submit_selections():
    it = get_val("iteration")

    selections = get_val("selections")
    selections[it] = request.get_json()
    set_val("selections", selections)

    return "OK"

@bp.route("/handle-feedback", methods=["POST", "GET"])
def handle_feedback():
    selected_user = get_val("selected_user")
    data = load_configuration_json()
    it = get_val("iteration")
    shown_data = data[selected_user][it]
    iteration_selections = get_val("selections")[it]
    it = it + 1
    set_val("iteration", it)

    payload = {
        "selected": iteration_selections,
        "shown": shown_data
    }

    iteration_ended(it - 1, payload)

    if it >= len(data[selected_user]):
        return redirect(url_for(f"{__plugin_name__}.finish_user_study"))

    return redirect(url_for(f"{__plugin_name__}.compare_visualizations"))


@bp.route("/finish-user-study")
def finish_user_study():

    conf = load_user_study_config(session["user_study_id"])
    params = {}
    # Handle overrides
    params["finished_text_override"] = None
    params["footer_override"] = None
    if "text_overrides" in conf:
        if "finished_text" in conf["text_overrides"]:
            params["finished_text_override"] = conf["text_overrides"]["finished_text"]

        if "footer" in conf["text_overrides"]:
            params["footer_override"] = conf["text_overrides"]["footer"]

    # Prolific stuff
    if "PROLIFIC_PID" in session:
        params["prolific_pid"] = session["PROLIFIC_PID"]
        params["prolific_url"] = f"https://app.prolific.co/submissions/complete?cc={conf['prolific_code']}"
    else:
        params["prolific_pid"] = None

    iterations = Interaction.query.filter((Interaction.participation == session["participation_id"]) & (Interaction.interaction_type == "iteration-ended")).all()
    n_identified = 0
    for it in iterations:
        d = json.loads(it.data)
        n_identified += d["selected"]["class_name"] == d["selected"]["example_class_name"]

    params["n_identified"] = min(n_identified, len(session["permutation"]))
    params["n_shown"] = len(session["permutation"])

    study_ended(session["participation_id"], iteration=session["iteration"])

    return render_template("visual_representation_finish.html", **params)

@bp.route("/get-image-data", methods=["GET"])
def get_image_data():
    selected_user = get_val("selected_user")
    data = load_configuration_json()
    it = get_val("iteration")
    iteration_data = data[selected_user][it]

    dataset = iteration_data["dataset"]
    viz_method = iteration_data["vizMethod"]

    candidate_list = []
    negative_samples = []

    for neg in iteration_data["negativeSamples"]:
        [class_name, file_name] = neg.split("/")
        negative_samples.append({
            "class_name": class_name,
            "dataset": dataset,
            "viz_method": viz_method,
            "file_path": url_for('static', filename=f'datasets/vizualizations2024/{dataset}/{viz_method}/{class_name}/{file_name}.png')
        })

    for candidate in iteration_data["candidateList"]:
        [class_name, file_name] = candidate.split("/")
        candidate_list.append({
            "class_name": class_name,
            "dataset": dataset,
            "viz_method": viz_method,
            "file_path": url_for('static', filename=f'datasets/vizualizations2024/{dataset}/{viz_method}/{class_name}/{file_name}.png')
        })

    [target_class_name, target_file_name] = iteration_data["target"].split("/")
    target = {
        "class_name": target_class_name,
        "file_path": url_for('static', filename=f'datasets/vizualizations2024/{dataset}/{viz_method}/{target_class_name}/{target_file_name}.png'),
        "dataset": dataset,
        "viz_method": viz_method,
    }

    paths = {
        "target": target,
        "candidate_list": candidate_list,
        "negative_samples": negative_samples
    }
    return jsonify(paths)


@bp.route("/get-instruction-bullets", methods=["GET"])
def get_instruction_bullets():
    page = request.args.get("page")
    if not page:
        return jsonify([])

    if page == "compare-visualizations":
        bullets = [
            "Instruction A With some text",
            "Instruction B",
            "Instruction C",
        ]
    else:
        bullets = []

    return jsonify(bullets)

@functools.cache
def load_configuration_json():
    # Load the JSON with configurations
    json_path = os.path.join(get_abs_project_root_path(), 'static', 'datasets', 'vizualizations2024', 'studyUsersRaw.json')
    with open(json_path, "r") as f:
        return json.load(f)

### Long running initialization is here ####
def long_initialization(guid):
    engine = create_engine('sqlite:///instance/db.sqlite')
    session = Session(engine)
    q = session.query(UserStudy).filter(UserStudy.guid == guid).first()
    
    # Just to populate the cache
    _ = load_configuration_json()

    q.initialized = True
    q.active = True
    session.commit()
    session.expunge_all()
    session.close()


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