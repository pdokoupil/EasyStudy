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
from collections import Counter
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
        "override_finished_text": "Override Final Text",
        "disable_demographics": "Disable demographics"
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
    session["iteration"] = 0
    data = load_configuration_json()
    selected_user = np.random.choice(list(data.keys()))

    set_val("selected_user", selected_user)
    set_val("selections", {})

    log_interaction(session["participation_id"], "user-selected", selected_user=selected_user, uuid=session["uuid"])

    return redirect(url_for(f"{__plugin_name__}.pre_study_questionnaire"))

@bp.route("/pre-study-questionnaire", methods=["GET", "POST"])
def pre_study_questionnaire():
    params = {
        "questions_url": url_for(f"{__plugin_name__}.get_pre_study_questions"),
        "continuation_url": url_for(f"{__plugin_name__}.pre_study_questionnaire_done"),
        "instructions_url": url_for(f"{__plugin_name__}.get_instruction_bullets", page="pre_study_questionnaire"),
        "finish": "Continue",
        "header": "Pre-study questionnaire",
        "hint": "Please fill in the questionnaire before proceeding to the following steps."
    }
    params["footer_override"] = None
    conf = load_user_study_config(session["user_study_id"])
    if "text_overrides" in conf:
        if "footer" in conf["text_overrides"]:
            params["footer_override"] = conf["text_overrides"]["footer"]
    return render_template("generic_questionnaire.html", **params)

@bp.route("/pre-study-questionnaire-done", methods=["GET", "POST"])
def pre_study_questionnaire_done():

    data = {}
    data.update(**request.form)

    # We just log the question answers as there is no other useful data gathered during pre-study-questionnaire
    log_interaction(session["participation_id"], "pre-study-questionnaire", **data)

    return redirect(url_for(f"{__plugin_name__}.compare_visualizations",
            consuming_plugin=__plugin_name__
        )
    )

@bp.route("/after-block-questionnaire", methods=["GET", "POST"])
def after_block_questionnaire():
    params = {
        "questions_url": url_for(f"{__plugin_name__}.get_after_block_questions"),
        "continuation_url": url_for(f"{__plugin_name__}.after_block_questionnaire_done"),
        "instructions_url": url_for(f"{__plugin_name__}.get_instruction_bullets", page="after_block_questionnaire"),
        "finish": "Continue",
        "hint": "Note that the questions relate only to the <b>current visualization method</b>, i.e., only to the last few tasks you completed.",
        "header": "Please fill in the questionnaire about the visualization method"
    }
    conf = load_user_study_config(session["user_study_id"])
    params["footer_override"] = None
    if "text_overrides" in conf:
        if "footer" in conf["text_overrides"]:
            params["footer_override"] = conf["text_overrides"]["footer"]
    return render_template("generic_questionnaire.html", **params)

@bp.route("/after-block-questionnaire-done", methods=["GET", "POST"])
def after_block_questionnaire_done():
    it = get_val("iteration")
    selected_user = get_val("selected_user")
    data = load_configuration_json()

    q_data = {}
    q_data.update(**request.form)

    # We just log the question answers as there is no other useful data gathered during after-block-questionnaire
    log_interaction(session["participation_id"], "after-block-questionnaire", **q_data)
 
    if it >= len(data[selected_user]):
        return redirect(url_for(f"{__plugin_name__}.finish_user_study"))

    return redirect(url_for(f"{__plugin_name__}.compare_visualizations"))

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
        "n_iterations": len(data[selected_user]),
        "hint": "Below, there is a visual representation of one data record belonging to the target class (positive example) and several representations o negative examples (items belonging to different classes).<br>Please select all candidates, which, you think, belong to the same class as the positive example (i.e., they seem close enough to it). Note that in several cases, the question may be an attention check, where the exact duplicate of the positive example is displayed.",
        "header": "Select all candidates that belong to the same class as the positive example."
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

    # If the visualization method is going to change, we mark end of block and show after block questionnaire
    # before proceeding further
    if it >= len(data[selected_user]) or shown_data["vizMethod"] != data[selected_user][it]['vizMethod']:
        return redirect(url_for(f"{__plugin_name__}.after_block_questionnaire"))

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


    selected_user = get_val("selected_user")
    data = load_configuration_json()

    iteration_data = data[selected_user]
    selections = get_val("selections")

    if len(iteration_data) != len(selections):
        print(f"Warning, lengths differ: {len(iteration_data) != len(selections)}, selections={selections}")

    #iterations = Interaction.query.filter((Interaction.participation == session["participation_id"]) & (Interaction.interaction_type == "iteration-ended")).all()
    n_identified = 0
    n_failed_checks = 0
    n_total_positive = 0
    n_total_candidates = 0
    n_selections = 0
    for iteration, iteration_selections in selections.items():
        n_selections += len(iteration_selections)
        current_iteration_data = iteration_data[iteration]
        n_total_candidates += len(current_iteration_data["candidateList"])
        if current_iteration_data["attentionCheck"] == 1:
            cnts_real = dict(Counter([x['example_name'] for x in iteration_selections]))
            cnts_goal = dict(Counter([x for x in current_iteration_data["candidateList"] if x == current_iteration_data["target"]]))
            if cnts_goal != cnts_real:
                n_failed_checks += 1
        failed_check = 0
        for candidate_name, is_correct in zip(current_iteration_data["candidateList"], current_iteration_data["correct"]):

            if is_correct:
                n_identified += len([x for x in iteration_selections if x["example_name"] == candidate_name]) > 0
                n_total_positive += 1

        n_failed_checks += 1 if failed_check else 0

    precision = round(n_identified / n_selections, 2)
    params["n_shown"] = n_total_candidates
    params["n_selections"] = n_selections
    params["n_identified"] = n_identified
    params["n_total_positive"] = n_total_positive
    params["precision"] = precision


    study_ended(session["participation_id"],
                iteration=get_val("iteration"),
                n_failed_checks=n_failed_checks,
                n_identified=n_identified,
                n_total_positive=n_total_positive,
                precision=precision,
                n_total_candidates=n_total_candidates,
                n_selections=n_selections
    )

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
            "example_name": neg,
            "file_path": url_for('static', filename=f'datasets/vizualizations2024/{dataset}/{viz_method}/{class_name}/{file_name}.png')
        })

    for idx, candidate in enumerate(iteration_data["candidateList"]):
        [class_name, file_name] = candidate.split("/")
        candidate_list.append({
            "class_name": class_name,
            "dataset": dataset,
            "viz_method": viz_method,
            "example_name": candidate,
            "file_path": url_for('static', filename=f'datasets/vizualizations2024/{dataset}/{viz_method}/{class_name}/{file_name}.png'),
            'idx': idx
        })

    [target_class_name, target_file_name] = iteration_data["target"].split("/")
    target = {
        "class_name": target_class_name,
        "file_path": url_for('static', filename=f'datasets/vizualizations2024/{dataset}/{viz_method}/{target_class_name}/{target_file_name}.png'),
        "dataset": dataset,
        "example_name": iteration_data["target"],
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
            "You can select the candidate by simply clicking on it",
            "If needed, you can de-select the candidate by clicking again",
            "All selected candidates are highlighted by a green border",
            "Once done, click on 'Continue' button. Note that you cannot go back after that."
        ]
    else:
        bullets = []

    return jsonify(bullets)

@bp.route("/get-pre-study-questions", methods=["GET"])
def get_pre_study_questions():
    q = [
        {
            "text": "Are you familiar with machine learning (ML)?",
            "name": "q1",
            "type": "select",
            "options": [
                "Not familiar at all",
                "User knowledge (I have an idea what ML is; I sometimes use its outputs, e.g., ChatGPT)",
                "Substantive knowledge (I understand how some ML algorithms work, I know its limitations)"
            ]
        },
        {
            "text": "Are you familiar with visualization techniques (VT)?",
            "name": "q2",
            "type": "select",
            "options": [
                "Not familiar at all",
                "User knowledge (I have an idea what VT is; I sometimes use basic methods, e.g., bar charts or line charts)",
                "Substantive knowledge (I have a good overview of different VT techniques, I can tune or adapt them to my needs)"
            ]
        },
        {
            "text": "Do you have any visual impairment?",
            "name": "q3",
            "type": "select",
            "options": [
                "None",
                "Glasses or contact lenses",
                "Color blindness or similar conditions affecting the perception of colors",
                "Other (please specify)"
            ]
        }
    ]

    return q

@bp.route("/get-after-block-questions", methods=["GET"])
def get_after_block_questions():
    q = [
        {
            "text": "It was easy to perceive differences/similarities using these visualizations.",
            "name": "q1",
            "type": "likert7",
            "neutral": True
        },
        {
            "text": "Using these visualizations took a lot of work and required substantial effort.",
            "name": "q2",
            "type": "likert7",
            "neutral": True
        },
        {
            "text": "The visualizations were rather disturbing.",
            "name": "q3",
            "type": "likert7",
            "neutral": True
        },
        {
            "text": "Overall, visualizations displayed enough information to distinguish positive vs. negative examples.",
            "name": "q4",
            "type": "likert7",
            "neutral": True
        },
        {
            "text": "Overall, I am confident that the candidates I selected are indeed from the same classes as the positive examples.",
            "name": "q5",
            "type": "likert7",
            "neutral": True
        },
        {
            "text": "I would feel unpleasant if I had to see these visualizations more often.",
            "name": "q6",
            "type": "likert7",
            "neutral": True
        },
    ]
    return q

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
