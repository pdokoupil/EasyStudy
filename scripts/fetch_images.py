#!/usr/bin/env python3
"""Bulk-download MovieLens cover images locally, so studies show posters instantly.

By default EasyStudy fetches a movie's poster from IMDb the first time it's shown (via
imdbinfo), which is slow on the first render. This script pre-fetches them all into
``server/static/datasets/<dataset>/img/<movieId>.jpg`` so the app serves them from disk.

It is **resumable** (skips images already on disk) and **threaded**.

Usage:
    python scripts/fetch_images.py --dataset ml-latest-small          # ~1.2k after filtering
    python scripts/fetch_images.py --dataset ml-latest --workers 16
    python scripts/fetch_images.py --dataset ml-latest-small --limit 500   # just the first N

Dependencies: only **imdbinfo** (`pip install imdbinfo`, or it's in the EasyStudy core) — the
rest is the Python standard library. Requests images at ~200px straight from IMDb's CDN, so no
Pillow/requests needed. If you'd rather not install anything, run it inside the container:
    docker compose run --rm app python scripts/fetch_images.py --dataset ml-latest-small

Note: IMDb may throttle a large burst; the run is resumable, so just re-run to continue.

If you've ALSO installed the `easystudy` CLI and use `easystudy serve` from this same
checkout directory: use `easystudy fetch-images` instead of this raw script from then on. This
script always writes to server/static/datasets (relative to itself); the CLI instead redirects
everything to your current directory, so mixing the two silently produces two different image
caches — `easystudy serve` won't find anything this script fetched, and re-fetches from IMDb
live (slowly) on every render instead.
"""
import argparse
import csv
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
# EASYSTUDY_DATASETS_DIR lets the installed `easystudy` CLI point this at the bundled
# server/static/datasets directory (a different relative layout than a repo checkout);
# unset (the default, repo-checkout usage) keeps today's sibling-relative behavior.
DATASETS_DIR = os.environ.get("EASYSTUDY_DATASETS_DIR") or os.path.join(HERE, "..", "server", "static", "datasets")


def _resize_amazon(url, width):
    """Rewrite a media-amazon cover URL to request a ~width-px thumbnail (small download)."""
    if not url or "media-amazon.com" not in url:
        return url
    resized = re.sub(r"\._V1_[^/]*?\.jpg$", f"._V1_SX{width}.jpg", url)
    return resized if resized != url else f"{url}._V1_SX{width}.jpg"


def _cover_url(imdb_id, width):
    from imdbinfo import get_movie
    movie = get_movie(f"tt{int(imdb_id):07d}")
    return _resize_amazon(getattr(movie, "cover_url", "") or "", width)


def _download(url):
    req = urllib.request.Request(url, headers={"User-Agent": "easystudy-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310 (documented CDN URL)
        return resp.read()


def _fetch_one(movie_id, imdb_id, img_dir, width):
    out_path = os.path.join(img_dir, f"{movie_id}.jpg")
    if os.path.exists(out_path):
        return movie_id, "skip"
    try:
        url = _cover_url(imdb_id, width)
        if not url:
            return movie_id, "no-poster"
        data = _download(url)  # already ~width px wide from the CDN — no local resize needed
        with open(out_path, "wb") as f:
            f.write(data)
        return movie_id, "ok"
    except Exception as e:  # noqa: BLE001
        return movie_id, f"err:{type(e).__name__}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="ml-latest-small",
                   help="dataset dir under server/static/datasets/ (must contain links.csv)")
    p.add_argument("--workers", type=int, default=8, help="parallel downloads (default 8)")
    p.add_argument("--width", type=int, default=200, help="poster width in px to request/save")
    p.add_argument("--limit", type=int, default=0, help="only the first N movies (0 = all)")
    p.add_argument("--min-ratings", type=int, default=0,
                   help="only movies with >= this many positive (>=4) ratings — matches the "
                        "'MovieLens Latest Small (demo)' loader with --min-ratings 10 (~1.2k). "
                        "0 = every movie in links.csv (~9.7k for ml-latest-small).")
    args = p.parse_args(argv)

    ds_dir = os.path.normpath(os.path.join(DATASETS_DIR, args.dataset))
    links_path = os.path.join(ds_dir, "links.csv")
    img_dir = os.path.join(ds_dir, "img")
    if not os.path.exists(links_path):
        sys.exit(f"links.csv not found at {links_path} — run scripts/fetch_data.py --dataset {args.dataset} first")
    os.makedirs(img_dir, exist_ok=True)

    # Optionally restrict to the popular subset the demo loader actually shows (so we don't
    # fetch ~9.7k posters when the demo only uses ~1.2k). Counts positive ratings per movie.
    keep_ids = None
    if args.min_ratings > 0:
        ratings_path = os.path.join(ds_dir, "ratings.csv")
        if not os.path.exists(ratings_path):
            sys.exit(f"--min-ratings needs ratings.csv at {ratings_path}")
        from collections import Counter
        counts_by_movie = Counter()
        with open(ratings_path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    if float(r["rating"]) >= 4.0:
                        counts_by_movie[r["movieId"]] += 1
                except (KeyError, ValueError):
                    pass
        keep_ids = {mid for mid, c in counts_by_movie.items() if c >= args.min_ratings}

    # links.csv columns: movieId,imdbId,tmdbId
    rows = []
    with open(links_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("imdbId") and (keep_ids is None or r["movieId"] in keep_ids):
                rows.append((r["movieId"], r["imdbId"]))
    if args.limit:
        rows = rows[:args.limit]

    total = len(rows)
    subset = f" (filtered to >={args.min_ratings} positive ratings)" if args.min_ratings else ""
    # Split into already-cached vs to-fetch up front so the progress bar reflects *real work*
    # (and a fully-cached folder finishes instantly with a clear message, not a confusing
    # "800/1182 skip" partial count).
    todo = [(mid, iid) for mid, iid in rows
            if not os.path.exists(os.path.join(img_dir, f"{mid}.jpg"))]
    cached = total - len(todo)
    print(f"[{args.dataset}] {total} movies{subset}: {cached} already cached, {len(todo)} to fetch "
          f"-> {img_dir} ({args.workers} workers)")
    if not todo:
        print("all posters already on disk — nothing to do.")
        return

    # imdbinfo is only needed to look up the posters we still have to fetch.
    try:
        import imdbinfo  # noqa: F401
    except Exception as e:  # noqa: BLE001
        sys.exit(f"imdbinfo unavailable ({e}); install with: pip install imdbinfo   (or run "
                 f"this inside the container: docker compose run --rm app python scripts/fetch_images.py …)")

    n = len(todo)
    counts, done = {}, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_fetch_one, mid, iid, img_dir, args.width) for mid, iid in todo]
        for fut in as_completed(futs):
            _, status = fut.result()
            key = status.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            done += 1
            if done % 100 == 0 or done == n:
                print(f"\r  {done}/{n} fetched  {counts}", end="", flush=True)
    print(f"\ndone: {counts}")


if __name__ == "__main__":
    main()
