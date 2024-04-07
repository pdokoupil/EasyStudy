import flask

from flask_login import current_user, login_required

import secrets

import sqlalchemy

from app import pm, db
from models import Participation, UserStudy
from common import gen_url_prefix

import json

import datetime

main = flask.Blueprint('main', __name__)

@main.route("/administration")
# @login_required
def administration():
    if current_user.is_authenticated:
        return flask.render_template("administration.html", current_user=current_user.email)
    else:
        return flask.redirect(flask.url_for('auth.login'))
    
# @main.route("/", methods=["GET"])
# def index():
#     if current_user.is_authenticated:
#         current_email = current_user.email
#     else:
#         current_email = ""
#     return flask.render_template("index.html", is_authenticated=current_user.is_authenticated, current_user=current_email)

@main.route("/notify")
# @login_required
def notify():
    if current_user.is_authenticated:
        return flask.render_template("notify.html", guid=flask.request.args.get("guid"))
    else:
        return flask.redirect(flask.url_for('auth.login'))


def get_loaded_plugins():
    endpoints = {str(p) for p in flask.current_app.url_map.iter_rules()}
    return [{
        "plugin_name": p["plugin_name"],
        "plugin_description": p["plugin_description"],
        "plugin_version": p["plugin_version"],
        "plugin_author": p["plugin_author"],
        "create_url": f"/{p['plugin_name']}/create"
    } for p in pm.get_enabled_plugins if f"/{p['plugin_name']}/create" in endpoints]


def get_loaded_plugin_names():
    return {p["plugin_name"] for p in get_loaded_plugins()}

# Returns a list of dicts (JSON) containing information about loaded plugins
# Only enabled plugins that also has pluginName/create endpoint defined, are listed
@main.route("/loaded-plugins")
def loaded_plugins():
    return get_loaded_plugins()

# Returns ids of all existing (both current and past) user studies
@main.route("/existing-user-studies")
@login_required
def existing_user_studies():
    result = db.session.query(UserStudy, sqlalchemy.func.count(Participation.participant_email)).outerjoin(Participation, UserStudy.id==Participation.user_study_id).group_by(UserStudy.id).all()
    
    # Filtering condition
    # Admin users see all the user studies while
    # normal users see only user studies created by them
    def filter_cond(x):
        if current_user.is_admin():
            return True
        return x.creator == current_user.get_id()

    return flask.jsonify([{
            "id": x.id,
            "creator": x.creator,
            "guid": x.guid,
            "parent_plugin": x.parent_plugin,
            "settings": x.settings,
            "time_created": x.time_created,
            "participants": c,
            "join_url": gen_user_study_invitation_url(x.parent_plugin, x.guid),
            "active": x.active,
            "initialized": x.initialized,
            "error": x.initialization_error
        } for x, c in result if filter_cond(x)])

def gen_user_study_url(guid):
    return f"/user-study/{guid}"

def gen_user_study_invitation_url(parent_plugin, guid):
    return f"{gen_url_prefix()}/{parent_plugin}/join?guid={guid}"

def get_vars(x):
    return {name: value for name, value in vars(x).items() if not name.startswith("_")}

@main.route("/user-study", methods=["GET"])
def get_user_study():
    user_study_id = flask.request.args.get("user_study_id")
    studies = UserStudy.query.filter(UserStudy.id == user_study_id).all()
    assert len(studies) <= 1
    if studies:
        return flask.jsonify(get_vars(studies[0]))
    else:
        return "Not found", 404

@main.route("/user-study/<id>", methods=["DELETE"])
def delete_user_study(id):
    x = UserStudy.query.filter(UserStudy.id == id)
    guid = x.first().guid
    parent_plugin = x.first().parent_plugin
    x.delete()
    db.session.commit()
    # Trigger plugin-specific disposal procedure 
    return flask.redirect(flask.url_for(f"{parent_plugin}.dispose", guid=guid))

@main.route("/user-study-active", methods=["POST"])
def set_user_study_active():
    data = flask.request.get_json()
    user_study_id = data["user_study_id"]
    new_state = bool(data["active"])
    study = UserStudy.query.filter(UserStudy.id == user_study_id).first()
    if study is None:
        return "Not found", 404
    if not study.initialized:
        return "Cannot activate study that was not initialized yet", 500
    study.active = new_state
    db.session.commit()
    return "OK"

@main.route("/user-studies", methods=["GET"])
def get_user_studies():
    studies = UserStudy.query.all()
    return flask.jsonify([get_vars(x) for x in studies])

@main.route("/participations", methods=["GET"])
def get_participations():
    participations = Participation.query.all()
    return flask.jsonify([get_vars(x) for x in participations])

@main.route("/user-study-participants", methods=["GET"])
def get_user_study_participants():
    user_study_id = flask.request.args.get("user_study_id")
    participants = Participation.query.filter(Participation.user_study_id == user_study_id).with_entities(Participation.participant_email).all()
    return flask.jsonify([{"participant_email": x[0]} for x in participants])

# Returns user studies in which the given user participated
@main.route("/user-participated-user-studies", methods=["GET"])
def get_user_participated_user_studies():
    user_email = flask.request.args.get("user_email")
    studies = Participation.query.filter(Participation.participant_email == user_email).with_entities(Participation.user_study_id)
    return flask.jsonify([{"user_study_id": x[0]} for x in studies])

# Adds a record that user starts participation in a user study
@main.route("/add-participant", methods=["POST"])
def add_participant():
    json_data = flask.request.get_json()
    
    user_study = UserStudy.query.filter(UserStudy.guid == json_data["user_study_guid"]).first()

    if not user_study:
        return "GUID not found", 404
    
    user_study_id = user_study.id

    extra_data = {}
    if "PROLIFIC_PID" in flask.session:
        extra_data["PROLIFIC_PID"] = flask.session["PROLIFIC_PID"]
        extra_data["PROLIFIC_STUDY_ID"] = flask.session["PROLIFIC_STUDY_ID"]
        extra_data["PROLIFIC_SESSION_ID"] = flask.session["PROLIFIC_SESSION_ID"]

    participation = Participation(
        participant_email=json_data["user_email"],
        user_study_id=user_study_id,
        time_joined=datetime.datetime.utcnow(),
        time_finished=None,
        age_group=json_data["age_group"],
        gender=json_data["gender"],
        education=json_data["education"],
        ml_familiar=json_data["ml_familiar"],
        language=json_data["lang"],
        uuid=flask.session["uuid"],
        extra_data=json.dumps(extra_data)
    )
    db.session.add(participation)
    db.session.commit()
    
    flask.session["participation_id"] = participation.id
    flask.session["user_study_id"] = user_study_id
    flask.session["user_study_guid"] = json_data["user_study_guid"]

    return "OK"

# Global create handler - takes user study settings and creates an user study from it
# Usually called from the individual plugins' create handlers
@main.route("/create-user-study", methods=["POST"])
@login_required
def create_user_study():
    guid = secrets.token_urlsafe(24)
    json_data = flask.request.get_json()

    if "parent_plugin" not in json_data:
        return "Bad Request - parent plugin was not specified", 400

    if json_data["parent_plugin"] not in get_loaded_plugin_names():
        return "Bad Request - invalid parent plugin", 400

    if "config" not in json_data:
        # No config was specified for the user study
        json_data["config"] = dict()

    study = UserStudy(
        creator=current_user.email,
        guid=guid,
        parent_plugin=json_data["parent_plugin"],
        settings = json.dumps(json_data["config"]),
        time_created = datetime.datetime.utcnow(),
        active=False, # Activation is responsibility of the plugin!,
        initialized=False,
        initialization_error=None
    )
    
    db.session.add(study)
    db.session.commit()

    # Trigger initialize 
    return flask.redirect(flask.url_for(f"{json_data['parent_plugin']}.initialize", continuation_url=flask.url_for('main.administration'), guid=guid), Response={
        "status": "success",
        "url": gen_user_study_url(guid)
    })

    

# if __name__ == "__main__":
#     print(os.getcwd())
#     print(os.listdir(os.getcwd()))
#     print(os.listdir("plugins"))
#     print(f"Starting, all plugins: {pm.get_all_plugins}")
#     app.run(debug=True, host="0.0.0.0")