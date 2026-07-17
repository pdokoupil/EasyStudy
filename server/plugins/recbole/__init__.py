import sys

[sys.path.append(i) for i in ['.', '..']]
[sys.path.append(i) for i in ['../.', '../..', '../../.']]

from models import UserStudy

from multiprocessing import Process
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from flask import Blueprint, request, redirect

__plugin_name__ = "recbole"
__version__ = "0.1.0"
__author__ = "EasyStudy"
__author_contact__ = ""
__description__ = "PyTorch/RecBole algorithm wrappers (BPR, LightGCN, NeuMF, …) for fastcompare."

bp = Blueprint(__plugin_name__, __plugin_name__, url_prefix=f"/{__plugin_name__}")


# Like the `vae` plugin, this only ships shared algorithm implementations for fastcompare —
# it is not a study template, so /join is unsupported.
@bp.route("/join", methods=["GET"])
def join():
    return "Not supported"


def long_initialization(guid):
    engine = create_engine('sqlite:///instance/db.sqlite')
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
    p = Process(target=long_initialization, daemon=True, args=(guid, ))
    p.start()
    return redirect(request.args.get("continuation_url"))


def register():
    return {
        "bep": dict(blueprint=bp, prefix=None),
    }
