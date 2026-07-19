#!/usr/bin/env python3
"""Download the datasets (CSVs) and item images EasyStudy needs to run studies.

These used to be shipped inside the git repo via git-LFS, which caused "over data
quota" errors on clone. They now live outside git and are fetched with this script.

Usage:
    python scripts/fetch_data.py --dataset all           # everything (default)
    python scripts/fetch_data.py --dataset ml-latest     # just MovieLens
    python scripts/fetch_data.py --dataset goodbooks-10k # just goodbooks
    python scripts/fetch_data.py --dataset all --no-images   # CSVs only
    python scripts/fetch_data.py --dataset all --force       # re-download

Only the Python standard library is used, so this runs before any pip install.
"""
import argparse
import io
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
# EASYSTUDY_DATASETS_DIR lets the installed `easystudy` CLI point this at the bundled
# server/static/datasets directory (a different relative layout than a repo checkout);
# unset (the default, repo-checkout usage) keeps today's sibling-relative behavior.
DATASETS_DIR = os.environ.get("EASYSTUDY_DATASETS_DIR") or os.path.join(HERE, "..", "server", "static", "datasets")

# dataset -> config
SOURCES = {
    "ml-latest": {
        "csv_url": "https://files.grouplens.org/datasets/movielens/ml-latest.zip",
        # the movielens archive nests everything under ml-latest/
        "csv_strip_prefix": "ml-latest/",
        "img_url": "http://herkules.ms.mff.cuni.cz/ligan/easystudy/ml_latest_img.zip",
    },
    "ml-latest-small": {
        # ~1 MB demo dataset. Posters are fetched on demand via imdbinfo, so there is no
        # image archive — img_url=None just creates an empty img/ dir for runtime caching.
        "csv_url": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
        "csv_strip_prefix": "ml-latest-small/",
        "img_url": None,
    },
    "goodbooks-10k": {
        "csv_url": "https://github.com/zygmuntz/goodbooks-10k/archive/refs/heads/master.zip",
        "csv_strip_prefix": "goodbooks-10k-master/",
        "img_url": "http://herkules.ms.mff.cuni.cz/ligan/easystudy/goodbooks_img.zip",
    },
}


def _download(url):
    print(f"  downloading {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "easystudy-fetch/1.0"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310 (trusted, documented URLs)
        total = int(resp.headers.get("Content-Length", 0))
        buf = io.BytesIO()
        read = 0
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            buf.write(chunk)
            read += len(chunk)
            if total:
                pct = 100 * read / total
                print(f"\r    {read/1e6:6.1f} MB / {total/1e6:.1f} MB ({pct:4.1f}%)",
                      end="", flush=True)
        print()
        buf.seek(0)
        return buf


def _extract_csvs(zip_bytes, dest, strip_prefix):
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zip_bytes) as zf:
        for member in zf.namelist():
            if not member.lower().endswith(".csv"):
                continue
            name = member[len(strip_prefix):] if member.startswith(strip_prefix) else os.path.basename(member)
            name = name.lstrip("/")
            if not name:
                continue
            target = os.path.join(dest, name)
            os.makedirs(os.path.dirname(target) or dest, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            print(f"    -> {os.path.relpath(target)}")


def _extract_images(zip_bytes, dataset_dir):
    """Image archives already contain an img/ directory; extract in place."""
    os.makedirs(dataset_dir, exist_ok=True)
    with zipfile.ZipFile(zip_bytes) as zf:
        zf.extractall(dataset_dir)
    img_dir = os.path.join(dataset_dir, "img")
    n = len(os.listdir(img_dir)) if os.path.isdir(img_dir) else 0
    print(f"    -> {os.path.relpath(img_dir)} ({n} images)")


def fetch(dataset, images=True, force=False):
    cfg = SOURCES[dataset]
    dataset_dir = os.path.normpath(os.path.join(DATASETS_DIR, dataset))
    print(f"[{dataset}]")

    csv_marker = os.path.join(dataset_dir, ".csv_done")
    if force or not os.path.exists(csv_marker):
        _extract_csvs(_download(cfg["csv_url"]), dataset_dir, cfg["csv_strip_prefix"])
        open(csv_marker, "w").close()
    else:
        print("  CSVs already present (use --force to re-download)")

    if images:
        img_dir = os.path.join(dataset_dir, "img")
        if not cfg.get("img_url"):
            # No prebuilt image archive: images are fetched on demand at runtime (imdbinfo).
            os.makedirs(img_dir, exist_ok=True)
            print("  images fetched on demand at runtime (no archive); created empty img/")
        elif force or not os.path.isdir(img_dir) or not os.listdir(img_dir):
            _extract_images(_download(cfg["img_url"]), dataset_dir)
        else:
            print("  images already present (use --force to re-download)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=list(SOURCES) + ["all"], default="all")
    p.add_argument("--no-images", dest="images", action="store_false", help="skip item images (CSVs only)")
    p.add_argument("--force", action="store_true", help="re-download even if already present")
    args = p.parse_args(argv)

    datasets = list(SOURCES) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        try:
            fetch(ds, images=args.images, force=args.force)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR fetching {ds}: {exc}", file=sys.stderr)
            print("  (you can retry, or download manually — see README 'Setup')", file=sys.stderr)
    print("done.")


if __name__ == "__main__":
    main()
