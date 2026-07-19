import os
import sys

[sys.path.append(i) for i in ['.', '..']]
[sys.path.append(i) for i in ['../.', '../..', '../../.']]

from models import UserStudy

from multiprocessing import Process
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from flask import Blueprint, request, redirect, render_template

__plugin_name__ = "vae"
__version__ = "0.1.0"
__author__ = "Anonymous Author"
__author_contact__ = "Anonymous@Author.com"
__description__ = "Plugin with VAE algorithm implementations (wrappers) for the fastcompare plugin."

bp = Blueprint(__plugin_name__, __plugin_name__, url_prefix=f"/{__plugin_name__}")

# Public facing endpoint
@bp.route("/join", methods=["GET"])
def join():
    return "Not supported"

### Long running initialization is here ####
def long_initialization(guid):
    # Same DATABASE_URL config.py/create_app() use — a hardcoded relative path here would
    # silently open/create a DIFFERENT, empty database whenever the app isn't run with CWD ==
    # server/ (e.g. the easystudy CLI, which points DATABASE_URL at the user's project dir).
    engine = create_engine(os.environ.get("DATABASE_URL", "sqlite:///instance/db.sqlite"))
    session = Session(engine)
    q = session.query(UserStudy).filter(UserStudy.guid == guid).first()

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
    return redirect(request.args.get("continuation_url"))

def register():
    return {
        "bep": dict(blueprint=bp, prefix=None),
    }
