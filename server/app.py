import time
import os
import random
import sys

import flask
import numpy as np
from flask_pluginkit import PluginManager
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect
from flask_session import Session

from sqlalchemy import MetaData, event
from sqlalchemy.engine import Engine
from werkzeug.security import generate_password_hash
import click

#from werkzeug.middleware.profiler import ProfilerMiddleware

from config import Config

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
    # Only SQLite needs (and understands) this pragma; skip it for other backends
    # (e.g. Postgres) so a production DATABASE_URL does not error on connect.
    if dbapi_connection.__class__.__module__.startswith("sqlite3"):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

# Insert/set all values that have to be set once (e.g. insert interaction types into DB)
def initialize_db_tables():
    pass


def _seed_user(email, password, admin=True):
    """Idempotently create an EasyStudy login user. Returns True if newly created.

    Must be called within an application context. Passwords are hashed (never stored plain).
    """
    from models import User
    if User.query.get(email):
        return False
    db.session.add(User(email=email, password=generate_password_hash(password), admin=admin))
    db.session.commit()
    return True

def create_app():
    app = flask.Flask(__name__)
    #app.wsgi_app = ProfilerMiddleware(app.wsgi_app)

    app.config.from_object(Config)
    # Flask-Session (sqlalchemy backend) needs a handle to the db instance
    if app.config.get("SESSION_TYPE") == "sqlalchemy":
        app.config["SESSION_SQLALCHEMY"] = db

    # IMPORTANT: db must be initialized BEFORE the session. Flask-Session >= 0.8's
    # sqlalchemy backend accesses `db.engine` during its own init_app(), which raises
    # "The current Flask app is not registered with this 'SQLAlchemy' instance" if db
    # has not been registered with this app yet.
    db.init_app(app)

    sess.init_app(app)

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

        # Optional zero-friction dev login: set EASYSTUDY_DEV_USER="email:password" to
        # auto-create an admin on startup. Unset by default (safety). NEVER use in production.
        dev_user = os.environ.get("EASYSTUDY_DEV_USER")
        if dev_user and ":" in dev_user:
            email, password = dev_user.split(":", 1)
            if _seed_user(email, password, admin=True):
                print(f"[EASYSTUDY_DEV_USER] created admin {email!r} — DO NOT use in production",
                      file=sys.stderr)

    # `flask create-user EMAIL PASSWORD [--admin/--no-admin]` — safe, explicit account creation
    # (no default credentials are ever baked in). Run e.g.:
    #   docker compose exec app flask create-user me@example.com secret
    @app.cli.command("create-user")
    @click.argument("email")
    @click.argument("password")
    @click.option("--admin/--no-admin", default=True, help="Grant admin rights (default: yes).")
    def create_user_command(email, password, admin):
        """Create an EasyStudy user for logging into the administration UI."""
        created = _seed_user(email, password, admin=admin)
        click.echo(f"{'Created' if created else 'Already exists'}: {email} (admin={admin})")

    # Seed setting in the case we use --preload with multiple workers and want to improve randomization on the first iteration
    # Otherwise we can just assume that this will be random enough given that users are distributed to workers randomly
    time_int = int(time.time())
    seed = os.getpid() + time_int
    random.seed(seed)
    np.random.seed(seed)
    # TensorFlow is an optional (heavy) extra; only seed it if it is installed.
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    print(f"Seeding with: {seed} ({time_int}, {os.getpid()})", file=sys.stderr)

    return app
