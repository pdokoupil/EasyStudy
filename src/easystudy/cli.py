"""Console entry point for the `easystudy` command.

    pip install "easystudy[recbole]"
    cd my-study-project                    # data (db, fetched datasets) lands HERE
    easystudy fetch-data --dataset ml-latest-small
    easystudy create-user me@example.com my-password
    easystudy serve --debug

Each subcommand that touches the actual application (`serve`, `create-user`, `fetch-data`,
`fetch-images`) first puts the vendored `server/` directory on `sys.path` (see
`easystudy._bootstrap`) and imports it exactly like running from a source checkout, then sets
``EASYSTUDY_DATA_ROOT`` / ``EASYSTUDY_DATASETS_DIR`` to the current working directory — so
fetched datasets, images, and the SQLite DB land in *your* project directory (wherever you
invoke `easystudy` from), never inside the installed package. Static assets that ship with the
code (templates, JS/CSS, tinymce, language files) are unaffected and always resolve from the
installed package, as they should.

Known limitation: Flask-Migrate schema migrations (`flask db migrate`/`upgrade`) are a
maintainer-only workflow and aren't wired into this CLI — fresh installs get their schema via
`db.create_all()`, which `serve` already runs automatically.
"""
import argparse
import os
import runpy
import sys


def _use_cwd_as_data_root() -> str:
    """Point the app's data-directory resolution AND its SQLite DB at the current working
    directory, so `fetch-data`/`fetch-images`/`serve`/`create-user` all agree on where
    datasets, images, and the DB live: the user's project directory, never inside the
    installed package.

    The DB URL needs special care: Flask-SQLAlchemy 3.x resolves a *relative* sqlite URI
    (the app's own default, ``sqlite:///db.sqlite``) against ``app.instance_path``, which for
    an installed package is inside site-packages — not writable/appropriate, and wiped on
    every reinstall. An *absolute* URI bypasses that resolution entirely, so we build one
    from the CWD.
    """
    cwd = os.getcwd()
    os.environ.setdefault("EASYSTUDY_DATA_ROOT", cwd)
    os.environ.setdefault("EASYSTUDY_DATASETS_DIR", os.path.join(cwd, "static", "datasets"))
    os.environ.setdefault("DATABASE_URL", f"sqlite:///{os.path.join(cwd, 'db.sqlite')}")
    return cwd


def _run_vendored_script(script_name: str, argv: list) -> None:
    from ._bootstrap import scripts_dir

    _use_cwd_as_data_root()
    script_path = os.path.join(scripts_dir(), script_name)
    old_argv = sys.argv
    sys.argv = [script_path] + list(argv)
    try:
        runpy.run_path(script_path, run_name="__main__")
    finally:
        sys.argv = old_argv


def _cmd_serve(args):
    _use_cwd_as_data_root()
    from ._bootstrap import ensure_server_on_path
    ensure_server_on_path()
    from app import create_app  # the vendored server/app.py, now importable
    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


def _cmd_create_user(args):
    _use_cwd_as_data_root()
    from ._bootstrap import ensure_server_on_path
    ensure_server_on_path()
    from app import create_app, _seed_user
    app = create_app()
    with app.app_context():
        created = _seed_user(args.email, args.password, admin=not args.no_admin)
    print(f"{'Created' if created else 'Already exists'}: {args.email} "
          f"(admin={not args.no_admin})")


def _cmd_version(_args):
    from . import __version__
    print(f"easystudy {__version__}")


#: Commands whose flags are simply forwarded, verbatim, to a vendored script's own argparse
#: parser. argparse's subparsers + `nargs=REMAINDER` interact unreliably for this (a
#: long-standing, documented argparse quirk), so these are dispatched by hand, before the
#: main parser ever sees the remaining argv.
_PASSTHROUGH_COMMANDS = {
    "fetch-data": "fetch_data.py",
    "fetch-images": "fetch_images.py",
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)

    if argv and argv[0] in _PASSTHROUGH_COMMANDS:
        _run_vendored_script(_PASSTHROUGH_COMMANDS[argv[0]], argv[1:])
        return

    parser = argparse.ArgumentParser(prog="easystudy", description="EasyStudy CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser(
        "serve", help="Run the development server in the current directory")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=5000)
    p_serve.add_argument("--debug", action="store_true")
    p_serve.set_defaults(func=_cmd_serve)

    # Listed here only so they show up in --help; actually handled above.
    sub.add_parser("fetch-data", help="Download a dataset (CSVs + item images) — "
                   "accepts the same flags as scripts/fetch_data.py, e.g. --dataset")
    sub.add_parser("fetch-images", help="Pre-fetch item poster images for a dataset — "
                   "accepts the same flags as scripts/fetch_images.py, e.g. --dataset")

    p_user = sub.add_parser("create-user", help="Create an administration login")
    p_user.add_argument("email")
    p_user.add_argument("password")
    p_user.add_argument("--no-admin", action="store_true", help="create a non-admin user")
    p_user.set_defaults(func=_cmd_create_user)

    p_version = sub.add_parser("version", help="Print the installed easystudy version")
    p_version.set_defaults(func=_cmd_version)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
