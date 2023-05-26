import json
import sys

from plugins.utils.interaction_logging import log_interaction

[sys.path.append(i) for i in ['.', '..']]
[sys.path.append(i) for i in ['../.', '../..', '../../.']]

from models import UserStudy

from common import load_user_study_config


from multiprocessing import Process
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from flask import Blueprint, request, redirect, render_template, url_for, session

from plugins.visual_representation.utils import build_permutation, dumper

import functools
import os
import numpy as np

__plugin_name__ = "visualrepresentation"
__version__ = "0.1.0"
__author__ = "Patrik Dokoupil"
__author_contact__ = "Patrik.Dokoupil@matfyz.cuni.cz"
__description__ = "Comparing different visual representations of data"

bp = Blueprint(__plugin_name__, __plugin_name__, url_prefix=f"/{__plugin_name__}")

MIN_ITERATIONS = 5
N_ITERATIONS = 100


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





# Public facing endpoint
@bp.route("/join", methods=["GET"])
def join():
    assert "guid" in request.args, "guid must be available in arguments"
    return redirect(url_for("utils.join", continuation_url=url_for("visualrepresentation.on_joined"), **request.args))

# Callback once user has joined we forward to preference elicitation
@bp.route("/on-joined", methods=["GET", "POST"])
def on_joined():
    
    session["iteration"] = 0
    session["permutation"] = build_permutation()

    log_interaction(session["participation_id"], "permutation-generated", permutation=json.loads(json.dumps(session["permutation"], default=dumper)))

    return redirect(url_for(f"{__plugin_name__}.compare_visualizations",
            consuming_plugin=__plugin_name__
        )
    )

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

def study_ended(iteration, payload):
    data = {
        "iteration": iteration
    }
    data.update(**payload)
    log_interaction(session["participation_id"], "study-ended", **data)

@bp.route("/compare-visualizations")
def compare_visualizations():
    iteration_data = json.loads(json.dumps(session["permutation"][session["iteration"]], default=dumper))
    params = {
        "continuation_url": url_for(f"{__plugin_name__}.handle_feedback"),
        "finish_url": url_for(f"{__plugin_name__}.finish_user_study"),
        "iteration": session["iteration"],
        "iteration_data": iteration_data,
        "n_iterations": len(session["permutation"])
    }

    conf = load_user_study_config(session["user_study_id"])
    if "text_overrides" in conf:
        if "comparison_hint" in conf["text_overrides"]:
            params["comparison_hint_override"] = conf["text_overrides"]["comparison_hint"]

        if "footer" in conf["text_overrides"]:
            params["footer_override"] = conf["text_overrides"]["footer"]

    payload = {
        "shown": iteration_data
    }
    iteration_started(session["iteration"], payload)

    return render_template("compare_visualizations.html", **params)

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

    study_ended(session["iteration"], {})

    return render_template("visual_representation_finish.html", **params)

@bp.route("/handle-feedback")
def handle_feedback():
    it = session['iteration']
    shown_data = json.loads(json.dumps(session["permutation"][it], default=dumper))
    it = it + 1
    session['iteration'] = it
    
    method = request.args.get("method")
    dataset = request.args.get("dataset")
    selection_id = request.args.get("selection_id")
    class_name = request.args.get("class_name")
    example_name = request.args.get("example_name")
    example_class_name = request.args.get("example_class_name")

    payload = {
        "selected": {
            "method": method,
            "dataset": dataset,
            "selection_id": selection_id,
            "class_name": class_name,
            "example_name": example_name,
            "example_class_name": example_class_name
        },
        "shown": shown_data
    }

    iteration_ended(it - 1, payload)

    if it >= len(session["permutation"]):
        return redirect(url_for(f"{__plugin_name__}.finish_user_study"))

    return redirect(url_for(f"{__plugin_name__}.compare_visualizations"))

### Long running initialization is here ####
def long_initialization(guid):
    engine = create_engine('sqlite:///instance/db.sqlite')
    session = Session(engine)
    q = session.query(UserStudy).filter(UserStudy.guid == guid).first()
    
    # TODO long running initialization

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