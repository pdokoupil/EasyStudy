"""Locate the vendored `server/`/`scripts/` bundle and make it importable.

EasyStudy's application code (`server/`) predates this packaging and uses flat,
CWD/sys.path-relative imports throughout (``from app import db``, ``from plugins.fastcompare
import ...``) rather than a dotted ``easystudy.*`` namespace. Rewriting ~40 files across every
plugin to a namespaced import style is a large, invasive change for comparatively little
benefit, so instead we ship `server/` and `scripts/` **verbatim** as vendored sub-packages
(`easystudy._vendor_server`, `easystudy._vendor_scripts`) and, at runtime, insert their
directories directly onto `sys.path` — exactly what already happens today when a developer
runs `cd server && flask run`. This keeps every existing file byte-for-byte unmodified.
"""
import os
import sys

_bootstrapped = False


def _pkg_dir(module) -> str:
    return os.path.dirname(os.path.abspath(module.__file__))


def server_dir() -> str:
    """Filesystem path of the vendored `server/` bundle (the Flask app + plugins)."""
    import easystudy._vendor_server as m
    return _pkg_dir(m)


def scripts_dir() -> str:
    """Filesystem path of the vendored `scripts/` bundle (fetch_data.py, fetch_images.py)."""
    import easystudy._vendor_scripts as m
    return _pkg_dir(m)


def ensure_server_on_path() -> str:
    """Insert the vendored server/ directory at the front of sys.path (idempotent).

    After calling this, ``from app import create_app``, ``from models import User``, etc.
    resolve exactly as they do when running from a source checkout.
    """
    global _bootstrapped
    path = server_dir()
    if path not in sys.path:
        sys.path.insert(0, path)
    _bootstrapped = True
    return path
