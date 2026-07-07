# Write your first plugin

A **study plugin** defines a whole study flow (not just a component). Start from the shipped
[`empty_template`](https://github.com/pdokoupil/EasyStudy/tree/main/server/plugins/empty_template).

## 1. Copy the template

```bash
cp -r server/plugins/empty_template server/plugins/myplugin
```

A plugin folder is a Python package with an `__init__.py` and a `templates/` directory.

## 2. Declare the plugin

In `myplugin/__init__.py`, set the metadata and a Flask `Blueprint` whose `url_prefix` is your plugin
name:

```python
from flask import Blueprint, request, redirect, render_template

__plugin_name__ = "myplugin"
__version__ = "0.1.0"
__author__ = "You"
__description__ = "My custom study flow."

bp = Blueprint(__plugin_name__, __plugin_name__, url_prefix=f"/{__plugin_name__}")
```

## 3. Implement the four lifecycle endpoints

See [Concepts → plugin lifecycle](concepts.md#the-plugin-lifecycle-endpoints). Minimum viable set:

```python
# (a) /create — render the parameter form. Uncomment to make the plugin appear in Administration.
@bp.route("/create")
def create():
    return render_template("myplugin_create.html")

# (b) /initialize — runs after the study is created. Do heavy setup in a BACKGROUND process so the
#     request returns immediately, then mark the study initialized & active.
from multiprocessing import Process
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from models import UserStudy

def long_initialization(guid):
    session = Session(create_engine('sqlite:///instance/db.sqlite'))
    study = session.query(UserStudy).filter(UserStudy.guid == guid).first()
    # ... train models / prepare data ...
    study.initialized = True
    study.active = True
    session.commit(); session.close()

@bp.route("/initialize")
def initialize():
    guid = request.args.get("guid")
    Process(target=long_initialization, daemon=True, args=(guid,)).start()
    return redirect(request.args.get("continuation_url"))

# (c) /join — entry point when a participant opens the study URL. You drive the flow from here.
@bp.route("/join", methods=["GET"])
def join():
    assert "guid" in request.args, "guid must be present"
    return render_template("myplugin_join.html")
```

!!! important "The `/create` → `/create-user-study` contract"
    Your `/create` form must ultimately POST to the framework's `/create-user-study` endpoint (the
    template in `empty_template` shows how). That is what actually persists the `UserStudy` row and
    triggers `/initialize`.

Optionally add `/results` to provide a custom evaluation view; if you omit it, EasyStudy falls back to
the default results page from `utils`.

## 4. Reuse the shared building blocks

Your `/join` flow can reuse `utils` endpoints instead of reinventing them:

- Preference elicitation, participant-details and final pages (GUIs).
- Interaction logging: call `utils`' logging so your data lands in the standard `Interaction` table
  (see [Data model](data-model.md)) — keeping your study analyzable with the same tools as everyone
  else.

## 5. Run it

Restart the server; your plugin appears in the Administration UI (once `/create` is enabled). Create an
instance, open the join URL, and verify the flow.

## Tips

- Long/blocking work (training, downloads) must go in the background daemon, never inside the request.
- Set `study.initialization_error` if setup fails, so the admin UI can surface it.
- Keep heavy dependencies optional (import lazily) so your plugin doesn't break the lightweight core.
