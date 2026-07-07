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
