import datetime
import json
from models import Interaction, Participation
from app import db

def log_interaction(participation_id, interaction_type, **kwargs):
    x = Interaction(
        participation = Participation.query.filter(Participation.id == participation_id).first().id,
        interaction_type = interaction_type,
        time = datetime.datetime.utcnow(),
        data = json.dumps(kwargs, ensure_ascii=False)
    )
    db.session.add(x)
    db.session.commit() 

def study_ended(participation_id, **kwargs):

    p = Participation.query.filter(Participation.id == participation_id).first()
    if p.time_finished:
        # This one was already marked as finished
        return

    x = Interaction(
        participation = p.id,
        interaction_type = "study-ended",
        time = datetime.datetime.utcnow(),
        data = json.dumps(kwargs, ensure_ascii=False)
    )

    db.session.add(x)
    db.session.commit()

    Participation.query.filter(Participation.id == participation_id).update({"time_finished": datetime.datetime.utcnow()})
    db.session.commit()