"""Optional client for movie cover images / metadata, used by the MovieLens loaders.

Two providers, in precedence order:

1. **TMDB** — only if a ``TMDB_API_KEY`` environment variable is set. TMDB has cleaner,
   more stable data, but the API requires a (free) key, so we never *require* it — it's a
   pure opt-in quality upgrade. Image files themselves come from the token-free
   ``image.tmdb.org`` CDN.
2. **imdbinfo** — the token-free default. Scrapes IMDb and returns a structured object;
   the poster URL is on ``cover_url`` (a token-free ``media-amazon.com`` CDN link).

Historically this used ``cinemagoer`` (IMDbPY), but cinemagoer removed IMDb web-page
parsing in April 2026. ``imdbinfo`` is an **optional** dependency
(``pip install easystudy[imdb]``), imported lazily; if neither provider is available or a
lookup fails, these helpers degrade gracefully (empty cover / safe defaults).
"""
from __future__ import annotations

import os

_warned = False


# --- imdbinfo (token-free default) ------------------------------------------------------

def _get_movie(imdb_id):
    """Return an imdbinfo movie object for a MovieLens numeric ``imdbId``, or None.

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
    except Exception as e:  # network error, not found, IMDb layout change, …
        print(f"[imdb_client] imdbinfo lookup failed for imdbId={imdb_id}: {e}")
        return None


# --- TMDB (optional, only with an API key) ----------------------------------------------

def _tmdb_cover(imdb_id) -> str:
    """Return a TMDB poster URL for the given IMDb id, or ``""``.

    No-op unless ``TMDB_API_KEY`` is set. Uses TMDB's ``/find`` endpoint to map an IMDb id
    to a movie and its ``poster_path``; the image itself is served token-free from the CDN.
    """
    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        return ""
    try:
        import requests
        r = requests.get(
            f"https://api.themoviedb.org/3/find/tt{int(imdb_id):07d}",
            params={"api_key": api_key, "external_source": "imdb_id"},
            timeout=10,
        )
        results = r.json().get("movie_results", []) if r.ok else []
        poster = results[0].get("poster_path") if results else None
        return f"https://image.tmdb.org/t/p/w342{poster}" if poster else ""
    except Exception as e:
        print(f"[imdb_client] TMDB lookup failed for imdbId={imdb_id}: {e}")
        return ""


# --- public API -------------------------------------------------------------------------

def get_cover_url(imdb_id) -> str:
    """Return the poster/cover image URL for the given IMDb id, or ``""`` on failure.

    Prefers TMDB (if ``TMDB_API_KEY`` is set), otherwise falls back to imdbinfo.
    """
    return _tmdb_cover(imdb_id) or (getattr(_get_movie(imdb_id), "cover_url", "") or "")


def get_movie_subset(imdb_id) -> dict:
    """Return a small, stable metadata subset for the given IMDb id.

    Shape: ``{plot, cast, genres, rating, year, cover}``. Missing pieces fall back to safe
    defaults so callers never have to guard individual fields. Metadata comes from imdbinfo;
    the cover prefers TMDB when a key is configured.
    """
    movie = _get_movie(imdb_id)
    tmdb_cover = _tmdb_cover(imdb_id)
    if movie is None:
        return {"plot": [], "cast": [], "genres": [], "rating": -1, "year": -1,
                "cover": tmdb_cover or None}

    def _safe(attr, default):
        try:
            val = getattr(movie, attr, default)
            return default if val is None else val
        except Exception:
            return default

    # imdbinfo exposes billed cast as ``stars`` (Person objects with ``.name``).
    cast = []
    try:
        for person in _safe("stars", []) or []:
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
        "cover": tmdb_cover or _safe("cover_url", None),
    }
