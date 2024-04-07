from app import db

from wtforms import PasswordField, EmailField
from wtforms.validators import DataRequired, Length
from flask_wtf import FlaskForm

# Represents authenticated user (Different from user study participant who were may not be authenticated)
class User(db.Model):
    __tablename__ = 'user'

    email = db.Column(db.String, primary_key=True)
    password = db.Column(db.String)
    authenticated = db.Column(db.Boolean, default=False)
    admin = db.Column(db.Boolean, default=False)

    def is_active(self):
        return True

    def get_id(self):
        return self.email

    @property
    def is_authenticated(self):
        return self.authenticated

    def is_anonymous(self):
        return False

    def is_admin(self):
        return self.admin


# Single instance of a user study/template/plugin
class UserStudy(db.Model):
    __tablename__ = "userstudy"

    id = db.Column(db.Integer, primary_key=True)
    creator = db.Column(db.String, db.ForeignKey('user.email')) # User who created the user study
    guid = db.Column(db.String) # Used for link creation
    parent_plugin = db.Column(db.String)
    settings = db.Column(db.String) # Settings of the user study
    time_created = db.Column(db.DateTime) # Date and time where the user study was created
    active = db.Column(db.Boolean)
    initialized = db.Column(db.Boolean)
    initialization_error = db.Column(db.String, default=None) # Initialization may stop due to an error

    def __str__(self):
        return f"id={self.id},creator={self.creator},guid={self.guid},time_created={self.time_created},settings={self.settings}"


# Relation between participants and user studies
class Participation(db.Model):
    __tablename__ = "participation"

    id = db.Column(db.Integer, primary_key=True)
    # It is not foreign key as the participant could be an anonymous user
    participant_email = db.Column(db.String)
    age_group = db.Column(db.String)
    gender = db.Column(db.String)
    education = db.Column(db.String)
    ml_familiar = db.Column(db.Boolean)
    user_study_id = db.Column(db.Integer, db.ForeignKey('userstudy.id'))
    time_joined = db.Column(db.DateTime)
    time_finished = db.Column(db.DateTime)
    uuid = db.Column(db.String)
    language = db.Column(db.String)
    extra_data = db.Column(db.String)

class LoginForm(FlaskForm):
    email = EmailField('email', validators=[DataRequired("missing mail")])
    password = PasswordField('password', validators=[DataRequired("missing password"), Length(6, 128, "short password")])
    #submit = SubmitField('Log in')

class SignupForm(FlaskForm):
    email = EmailField('email', validators=[DataRequired("missing mail")])
    password = PasswordField('password', validators=[DataRequired("missing password"), Length(6, 128, "short password")])
    #submit = SubmitField('Sign up')

# class InteractionType(db.Model):
#     __tablename__ = "interactiontype"
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String)

class Interaction(db.Model):
    __tablename__ = "interaction"

    id = db.Column(db.Integer, primary_key=True)
    # Interactions are tied to participations
    participation = db.Column(db.Integer, db.ForeignKey("participation.id"))
    # interaction_type = db.Column(db.Integer, db.ForeignKey("interactiontype.id"))
    interaction_type = db.Column(db.String)

    # Time when interaction has occurred
    time = db.Column(db.DateTime)
    data = db.Column(db.String)

class Message(db.Model):
    __tablename__ = "message"

    id = db.Column(db.Integer, primary_key=True)
    # Not mandatory
    participation = db.Column(db.Integer, nullable=True)
    # Time when massage has been received
    time = db.Column(db.DateTime)
    # Arbitrary data
    data = db.Column(db.String)