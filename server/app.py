import flask
from flask_pluginkit import PluginManager
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect

from flask_session import Session

from sqlalchemy import MetaData, event
from sqlalchemy.engine import Engine

naming_convention = {
    "ix": 'ix_%(column_0_label)s',
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}

db = SQLAlchemy(metadata = MetaData(naming_convention=naming_convention))
migrate = Migrate()
pm = PluginManager(plugins_folder="plugins")
csrf = CSRFProtect()

sess = Session()

from models import *

# This is needed to ensure foreign keys and corresponding cascade deletion work as
# expected when SQLite is used as backend for SQLAlchemy
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

# Insert/set all values that have to be set once (e.g. insert interaction types into DB)
def initialize_db_tables():
    pass
    # from models import InteractionType

    # # If it has not been inserted yet, insert selected-item interaction type
    # if db.session.query(
    #     db.session.query(InteractionType).filter_by(name='selected-item').exists()
    # ).scalar():
    #     x = InteractionType()
    #     x.name = "selected-item"
    #     db.session.add(x)

    # # If it has not been inserted yet, insert deselected-item interaction type
    # if db.session.query(
    #     db.session.query(InteractionType).filter_by(name='deselected-item').exists()
    # ).scalar():
    #     x = InteractionType()
    #     x.name = "deselected-item"
    #     db.session.add(x)

    # # If it has not been inserted yet, insert changed-viewport type
    # if db.session.query(
    #     db.session.query(InteractionType).filter_by(name='changed-viewport').exists()
    # ).scalar():
    #     x = InteractionType()
    #     x.name = "changed-viewport"
    #     db.session.add(x)

    # # If it has not been inserted yet, insert clicked-button type
    # if db.session.query(
    #     db.session.query(InteractionType).filter_by(name='clicked-button').exists()
    # ).scalar():
    #     x = InteractionType()
    #     x.name = "clicked-button"
    #     db.session.add(x)

def create_app():
    app = flask.Flask(__name__)

    app.config['SECRET_KEY'] = '8bf29bd88d0bfb94509f5fb0'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
    app.config['SESSION_COOKIE_NAME'] = "something"
    app.config["SESSION_TYPE"] = "filesystem"

    sess.init_app(app)

    db.init_app(app)

    migrate.init_app(app, db, render_as_batch=True)

    csrf.init_app(app)

    login_manager = LoginManager(app)
    
    pm.init_app(app)


    @login_manager.user_loader
    def user_loader(user_id):
        """Given *user_id*, return the associated User object.

        :param unicode user_id: user_id (email) user to retrieve

        """
        return User.query.get(user_id)

    from main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    from auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint)

    with app.app_context():
        db.create_all()
        initialize_db_tables()

    return app