import sys

from plugins.utils.interaction_logging import log_interaction

[sys.path.append(i) for i in ['.', '..']]
[sys.path.append(i) for i in ['../.', '../..', '../../.']]

from models import UserStudy

from multiprocessing import Process
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from flask import Blueprint, request, redirect, render_template, url_for, session

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
    return render_template("visual_representation_create.html")

@bp.context_processor
def plugin_name():
    return {
        "plugin_name": __plugin_name__
    }

# Public facing endpoint
@bp.route("/join", methods=["GET"])
def join():
    assert "guid" in request.args, "guid must be available in arguments"
    return redirect(url_for("utils.join", continuation_url=url_for("visualrepresentation.on_joined"), **request.args))

# Callback once user has joined we forward to preference elicitation
@bp.route("/on-joined", methods=["GET", "POST"])
def on_joined():
    session["iteration"] = 1
    return redirect(url_for(f"{__plugin_name__}.compare_visualizations",
            consuming_plugin=__plugin_name__
        )
    )

def iteration_ended(iteration, selected_visualization, shown_visualizations):
    data = {
        "iteration": iteration,
        "selected_visualization": selected_visualization,
        "shown_visualizations": shown_visualizations
    }
    log_interaction(session["participation_id"], "iteration-ended", **data)

@bp.route("/compare-visualizations")
def compare_visualizations():
    params = {
        "continuation_url": url_for(f"{__plugin_name__}.handle_feedback"),
        "finish_url": url_for(f"{__plugin_name__}.finish_user_study"),
        "iteration": session["iteration"],
        "n_iterations": N_ITERATIONS,
        "MIN_ITERATIONS_TO_CANCEL": MIN_ITERATIONS
    }
    return render_template("compare_visualizations.html", **params)

@bp.route("/finish-user-study")
def finish_user_study():
    return render_template("visual_representation_finish.html")

@bp.route("/handle-feedback")
def handle_feedback():
    it = session["iteration"]
    
    it += 1
    session["iteration"] = it

    
    selected_visualization = int(request.args.get("selected_visualization"))
    iteration_ended(it - 1, selected_visualization, shown_visualizations=[])

    if it - 1 >= MIN_ITERATIONS:
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
