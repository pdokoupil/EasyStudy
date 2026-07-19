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

import concurrent.futures
import os
import re

_warned = False

# In-memory memo so repeated lookups for the same movie (across study iterations) don't
# re-hit the network. Keyed by (imdb_id, width) for covers and imdb_id for subsets.
_cover_cache: dict = {}
_subset_cache: dict = {}

# imdbinfo's get_movie() takes no timeout argument, and internally makes a plain
# `niquests.get(url, ...)` call with none set either — a slow/hung response (IMDb's WAF, a
# network hiccup) blocks forever with nothing to stop it except gunicorn's own worker-timeout
# SIGKILL, which takes down the whole request (and, with the default single worker, the app)
# with it. Running the call in a bounded thread means a hang just times out here instead —
# same graceful-degradation-to-None this module already does for every other failure mode.
_IMDB_TIMEOUT_S = float(os.environ.get("IMDB_REQUEST_TIMEOUT", "8"))
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="imdb-fetch")


def _resize_amazon(url: str, width: int) -> str:
    """Downscale a media-amazon cover URL by rewriting its ``._V1_….jpg`` size suffix.

    imdbinfo's ``cover_url`` points at the full-resolution image (often several MB). IMDb's
    CDN supports on-the-fly resizing via the filename suffix, so ``._V1_SX300.jpg`` yields a
    ~40 KB, 300px-wide thumbnail. Non-amazon URLs (e.g. TMDB, already sized) are left as-is.
    """
    if not url or "media-amazon.com" not in url:
        return url
    resized = re.sub(r"\._V1_[^/]*?\.jpg$", f"._V1_SX{width}.jpg", url)
    # If the URL had no recognizable suffix, append one so we still get a thumbnail.
    return resized if resized != url else f"{url}._V1_SX{width}.jpg"


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
        future = _executor.submit(get_movie, f"tt{int(imdb_id):07d}")
        return future.result(timeout=_IMDB_TIMEOUT_S)
    except concurrent.futures.TimeoutError:
        print(f"[imdb_client] imdbinfo lookup timed out after {_IMDB_TIMEOUT_S}s for "
              f"imdbId={imdb_id} — giving up on this cover/metadata for now")
        return None
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

def get_cover_url(imdb_id, width: int = 300) -> str:
    """Return a (thumbnail-sized) poster URL for the given IMDb id, or ``""`` on failure.

    Prefers TMDB (if ``TMDB_API_KEY`` is set), otherwise falls back to imdbinfo, downscaled
    to ``width`` px. Results are memoized in-process. Pass ``width=0`` for full resolution.
    """
    key = (str(imdb_id), width)
    if key in _cover_cache:
        return _cover_cache[key]
    tmdb = _tmdb_cover(imdb_id)  # TMDB URLs are already sized (w342)
    if tmdb:
        _cover_cache[key] = tmdb
        return tmdb
    cover = getattr(_get_movie(imdb_id), "cover_url", "") or ""
    if cover and width:
        cover = _resize_amazon(cover, width)
    _cover_cache[key] = cover
    return cover


def get_movie_subset(imdb_id) -> dict:
    """Return a small, stable metadata subset for the given IMDb id.

    Shape: ``{plot, cast, genres, rating, year, cover}``. Missing pieces fall back to safe
    defaults so callers never have to guard individual fields. Metadata comes from imdbinfo;
    the cover prefers TMDB when a key is configured.
    """
    key = str(imdb_id)
    if key in _subset_cache:
        return _subset_cache[key]
    movie = _get_movie(imdb_id)
    tmdb_cover = _tmdb_cover(imdb_id)
    if movie is None:
        result = {"plot": [], "cast": [], "genres": [], "rating": -1, "year": -1,
                  "cover": tmdb_cover or None}
        _subset_cache[key] = result
        return result

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

    imdb_cover = _safe("cover_url", None)
    result = {
        "plot": _safe("plot", []),
        "cast": cast,
        "genres": _safe("genres", []),
        "rating": _safe("rating", -1),
        "year": _safe("year", -1),
        "cover": tmdb_cover or (_resize_amazon(imdb_cover, 300) if imdb_cover else None),
    }
    _subset_cache[key] = result
    return result
