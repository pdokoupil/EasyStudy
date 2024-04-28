import sys

[sys.path.append(i) for i in ['.', '..']]
[sys.path.append(i) for i in ['../.', '../..', '../../.']]

from models import UserStudy

from multiprocessing import Process
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from flask import Blueprint, request, redirect, render_template

__plugin_name__ = "emptytemplate"
__version__ = "0.1.0"
__author__ = "Patrik Dokoupil"
__author_contact__ = "Patrik.Dokoupil@matfyz.cuni.cz"
__description__ = "Empty template, can be used as a starting point for creating plugins."

bp = Blueprint(__plugin_name__, __plugin_name__, url_prefix=f"/{__plugin_name__}")

# Uncomment this endpoint to make plugin visible in the administration
# @bp.route("/create")
# def create():
#     return render_template("empty_template_create.html")

# Public facing endpoint
@bp.route("/join", methods=["GET"])
def join():
    assert "guid" in request.args, "guid must be available in arguments"
    return render_template("empty_template_join.html")

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

# Plugin specific disposal procedure
# E.g. removing plugin-specific cache etc.
# Leaving empty if no disposal is needed
@bp.route("/dispose", methods=["DELETE"])
def dispose():
    return "OK"

def register():
    return {
        "bep": dict(blueprint=bp, prefix=None),
    }
