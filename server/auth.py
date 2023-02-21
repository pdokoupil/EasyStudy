import flask

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_required, current_user, login_user, logout_user
from is_safe_url import is_safe_url

from werkzeug.security import generate_password_hash, check_password_hash

from app import db
from models import User, LoginForm, SignupForm

auth = flask.Blueprint('auth', __name__)

@auth.route('/login', methods=['POST'])
def login_post():
    # Here we use a class of some kind to represent and validate our
    # client-side form data. For example, WTForms is a library that will
    # handle this for us, and we use a custom LoginForm to validate.
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.get(form.email.data)
        print(f"A1, user={user}, email={form.email.data}")
        if user:
            print("A2")
            if check_password_hash(user.password, form.password.data):
                print("A3")
                user.authenticated = True
                db.session.add(user)
                db.session.commit()
                login_user(user, remember=True)

            next = flask.request.args.get('next')
            # is_safe_url should check if the url is safe for redirects.
            # See http://flask.pocoo.org/snippets/62/ for an example.
            print(f"next={next}")
            if next and not is_safe_url(next, {}):
                return flask.abort(400)

            return flask.redirect(next or flask.url_for('main.administration'))
    else:
        print(f"Errors = {form.errors}")
    print("A5")
    return flask.render_template('login.html', form=form)

@auth.route('/login')
def login():
    print("Login html is called")
    return flask.render_template('login.html', authenticated=current_user.is_authenticated)

@auth.route("/logout")
@login_required
def logout():
    """Logout the current user."""
    user = current_user
    user.authenticated = False
    db.session.add(user)
    db.session.commit()
    logout_user()
    return flask.redirect(flask.url_for('auth.login'))


@auth.route('/signup')
def signup():
    return flask.render_template('signup.html')

@auth.route('/signup', methods=['POST'])
def signup_post():

    form = SignupForm()
    print(f"fm={form.email.data}, f={form.password.data}")
    if form.validate_on_submit():
        print("Passed")
        user = User.query.filter_by(email=form.email.data).first() # if this returns a user, then the email already exists in database

        if user: # if a user is found, we want to redirect back to signup page so user can try again  
            #flask.flash('Email address already exists')
            #print("AA")
            return "The user already exists" #flask.redirect(flask.url_for('auth.signup'))

        # create new user with the form data. Hash the password so plaintext version isn't saved.
        new_user = User(email=form.email.data, password=generate_password_hash(form.password.data, method='sha256'))

        # add the new user to the database
        db.session.add(new_user)
        db.session.commit()

        return flask.redirect(flask.url_for('auth.login'))
    else:
        print("Not passed")
        print(form.errors)
        return "The password is too short"