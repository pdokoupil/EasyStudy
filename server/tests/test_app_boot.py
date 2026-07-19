"""Smoke test: the LIGHTWEIGHT core must import and boot without TensorFlow/LensKit.

This guards the promise that `pip install easystudy` (no extras) yields a runnable app:
plugin discovery must skip modules whose optional deps are missing rather than crash.
Run from the `server/` directory: `pytest -q`.
"""
import os
import tempfile

import pytest


# Module-scoped: create_app() uses module-level singletons (db/sess/pm), so it must be
# called once per process, not per test.
@pytest.fixture(scope="module")
def app():
    # Isolate DB/secret so the test never touches a real deployment.
    os.environ["SECRET_KEY"] = "test-secret"
    tmp = tempfile.mkdtemp()
    os.environ["DATABASE_URL"] = f"sqlite:///{os.path.join(tmp, 'test.sqlite')}"

    from app import create_app  # imported here so env is set first

    application = create_app()
    application.config.update(TESTING=True)
    return application


def test_app_boots(app):
    assert app is not None


def test_login_page_served(app):
    client = app.test_client()
    resp = client.get("/login")
    # 200 (page) or a redirect are both fine; a 500 would mean the core is broken.
    assert resp.status_code < 500


def test_plugin_discovery_is_resilient_without_extras():
    # load_* must not raise even when TF/LensKit algorithms can't be imported.
    from plugins.fastcompare.loading import (
        load_algorithms,
        load_data_loaders,
        load_preference_elicitations,
        load_evaluation_metrics,
    )

    # These call walk_packages over all plugins; on a lightweight install some modules
    # (TFRS/VAE/LensKit) are skipped. The call must succeed and return a dict.
    assert isinstance(load_algorithms(), dict)
    assert isinstance(load_data_loaders(), dict)
    assert isinstance(load_preference_elicitations(), dict)
    assert isinstance(load_evaluation_metrics(), dict)


# --- long_initialization must honor DATABASE_URL, not a hardcoded path -------------------
#
# Regression coverage for a real bug: every study-init plugin's `long_initialization(guid)`
# runs in a spawned subprocess with no Flask app context, so it reconnects to the DB by hand
# with its own `create_engine(...)`. That used to be a hardcoded `sqlite:///instance/db.sqlite`
# — only correct when the process's CWD happens to be `server/`. With DATABASE_URL pointing
# elsewhere (exactly what the `app` fixture above sets up, and exactly what the `easystudy`
# CLI does for every real install, since it runs from the user's own project directory), the
# subprocess silently opened a DIFFERENT, empty database and found no matching row — surfacing
# as `AttributeError: 'NoneType' object has no attribute 'settings'` well after study creation
# looked like it had succeeded. None of the existing tests caught this: test_creation.py
# exercises data loaders/algorithms/elicitations directly (never through long_initialization's
# own DB round trip), and nothing else calls it either.
_LONG_INIT_PLUGINS = ["fastcompare", "layoutshuffling", "recbole", "vae", "empty_template"]

# layoutshuffling's long_initialization loads the full ml-latest dataset unconditionally,
# *before* it ever looks at the study row — in this test's deliberately empty/uncached
# environment that fails with FileNotFoundError first, which would mask the very regression
# this test exists to catch (a false pass, not a real one). Skip it here rather than pretend
# to cover it; ml-latest isn't fetched in CI on purpose (this suite avoids downloads).
_ML_LATEST_RATINGS = os.path.join(
    os.path.dirname(__file__), "..", "static", "datasets", "ml-latest", "ratings.csv")


@pytest.mark.parametrize("plugin_name", _LONG_INIT_PLUGINS)
def test_long_initialization_finds_its_own_userstudy_row(app, plugin_name):
    import importlib
    import uuid

    from app import db
    from models import UserStudy

    if plugin_name == "layoutshuffling" and not os.path.exists(_ML_LATEST_RATINGS):
        pytest.skip("needs the ml-latest dataset (not fetched in CI) before it touches the DB")

    mod = importlib.import_module(f"plugins.{plugin_name}")

    with app.app_context():
        guid = str(uuid.uuid4())
        # Deliberately empty settings: we're not testing that a study can fully train here
        # (that needs a real dataset — see test_creation.py), only that long_initialization
        # connects to the SAME database the row was inserted into and actually finds it.
        study = UserStudy(creator=None, guid=guid, parent_plugin=plugin_name,
                          settings="{}", active=False, initialized=False)
        db.session.add(study)
        db.session.commit()

    # Plugins differ in how "thorough" their long_initialization is: fastcompare/
    # layoutshuffling wrap it in try/except and record failures on the row itself; the
    # simpler ones (recbole/vae/empty_template) have no settings-parsing at all and would
    # raise directly on a missing row. Either way, a NoneType AttributeError means the
    # wrong database — that's the one thing this test must never see, regardless of which
    # shape of failure/success a given plugin produces otherwise.
    raised = None
    try:
        mod.long_initialization(guid)  # runs synchronously when called directly
    except Exception as e:
        raised = e

    def _is_wrong_db_error(exc_or_msg) -> bool:
        return "'NoneType' object has no attribute" in str(exc_or_msg)

    if raised is not None:
        assert not _is_wrong_db_error(raised), (
            f"{plugin_name}.long_initialization couldn't find its own study row: {raised}"
        )
        return

    with app.app_context():
        db.session.expire_all()  # force a real re-read; the row was written via a separate engine
        study = db.session.query(UserStudy).filter(UserStudy.guid == guid).first()
        assert study is not None, (
            f"{plugin_name}.long_initialization lost track of its own study row entirely"
        )
        if study.initialization_error is not None:
            assert not _is_wrong_db_error(study.initialization_error)
