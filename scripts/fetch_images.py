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
"""
import argparse
import csv
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(HERE, "..", "server", "static", "datasets")


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
    args = p.parse_args(argv)

    ds_dir = os.path.normpath(os.path.join(DATASETS_DIR, args.dataset))
    links_path = os.path.join(ds_dir, "links.csv")
    img_dir = os.path.join(ds_dir, "img")
    if not os.path.exists(links_path):
        sys.exit(f"links.csv not found at {links_path} — run scripts/fetch_data.py --dataset {args.dataset} first")
    os.makedirs(img_dir, exist_ok=True)

    try:
        import imdbinfo  # noqa: F401
    except Exception as e:  # noqa: BLE001
        sys.exit(f"imdbinfo unavailable ({e}); install with: pip install imdbinfo   (or run "
                 f"this inside the container: docker compose run --rm app python scripts/fetch_images.py …)")

    # links.csv columns: movieId,imdbId,tmdbId
    rows = []
    with open(links_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("imdbId"):
                rows.append((r["movieId"], r["imdbId"]))
    if args.limit:
        rows = rows[:args.limit]

    total = len(rows)
    print(f"[{args.dataset}] {total} movies -> {img_dir} ({args.workers} workers)")
    counts, done = {}, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_fetch_one, mid, iid, img_dir, args.width) for mid, iid in rows]
        for fut in as_completed(futs):
            _, status = fut.result()
            key = status.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            done += 1
            if done % 100 == 0 or done == total:
                print(f"\r  {done}/{total}  {counts}", end="", flush=True)
    print(f"\ndone: {counts}")


if __name__ == "__main__":
    main()
