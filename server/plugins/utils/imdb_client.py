"""Thin, optional IMDb client used to fetch cover images / metadata for movie datasets.

Historically this used ``cinemagoer`` (IMDbPY), but cinemagoer **removed IMDb web-page
parsing in April 2026**, so ``get_movie(id)["full-size cover url"]`` no longer works. We
now use `imdbinfo <https://pypi.org/project/imdbinfo/>`_, which scrapes IMDb and returns a
structured object with a ``.image`` (poster) URL.

``imdbinfo`` is an **optional** dependency (``pip install easystudy[imdb]``). It is imported
lazily so the lightweight core doesn't need it; if it's missing or a lookup fails, these
helpers degrade gracefully (empty cover / safe defaults) with a one-time hint.
"""
from __future__ import annotations

_warned = False


def _get_movie(imdb_id):
    """Return an imdbinfo movie object for a MovieLens-style numeric ``imdbId``, or None.

    MovieLens ``links.csv`` stores IMDb ids as bare numbers (e.g. ``114709``); imdbinfo
    expects the canonical ``tt``-prefixed, zero-padded form (``tt0114709``).
    """
    global _warned
    try:
        # Broad except: imdbinfo may be missing (ImportError) *or* installed-but-broken
        # (e.g. a stale pydantic v1 in the env) — either way, degrade gracefully.
        from imdbinfo import get_movie
    except Exception as e:
        if not _warned:
            print(f"[imdb_client] imdbinfo unavailable ({type(e).__name__}) — cover images/"
                  "metadata disabled. Install/repair with: pip install easystudy[imdb]")
            _warned = True
        return None
    try:
        return get_movie(f"tt{int(imdb_id):07d}")
    except Exception as e:  # network error, not found, parse change, …
        print(f"[imdb_client] lookup failed for imdbId={imdb_id}: {e}")
        return None


def get_cover_url(imdb_id) -> str:
    """Return the poster/cover image URL for the given IMDb id, or ``""`` on failure."""
    movie = _get_movie(imdb_id)
    return getattr(movie, "image", "") or "" if movie is not None else ""


def get_movie_subset(imdb_id) -> dict:
    """Return a small, stable metadata subset for the given IMDb id.

    Shape matches what the data-loader wrappers expect:
    ``{plot, cast, genres, rating, year, cover}``. Missing pieces fall back to safe
    defaults so callers never have to guard individual fields.
    """
    movie = _get_movie(imdb_id)
    if movie is None:
        return {"plot": [], "cast": [], "genres": [], "rating": -1, "year": -1, "cover": None}

    def _safe(attr, default):
        try:
            val = getattr(movie, attr, default)
            return default if val is None else val
        except Exception:
            return default

    # cast entries are objects with a ``.name`` (or plain strings); be defensive.
    cast = []
    try:
        for person in _safe("cast", []) or []:
            name = getattr(person, "name", person)
            if name:
                cast.append(name)
    except Exception:
        cast = []

    return {
        "plot": _safe("plot", []),
        "cast": cast,
        "genres": _safe("genres", []),
        "rating": _safe("rating", -1),
        "year": _safe("year", -1),
        "cover": _safe("image", None),
    }
