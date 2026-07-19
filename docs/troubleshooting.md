# Troubleshooting

## `git clone` fails with a git-LFS "over data quota" / smudge error
Fixed in current `main` — dataset images are no longer tracked by git-LFS. If you're on an **older
checkout** that still has the LFS filter, clone with LFS smudge disabled:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/pdokoupil/EasyStudy.git
```
then fetch images with `python scripts/fetch_data.py`.

## A study won't start / "ratings.csv is missing"
You haven't fetched the dataset. Run:
```bash
python scripts/fetch_data.py --dataset all
# or a single one: python scripts/fetch_data.py --dataset ml-latest
```
Files land in `server/static/datasets/<dataset>/`.

## An algorithm I expected isn't in the create-study dropdown
Its optional dependency isn't installed. RecBole, TensorFlow-based (VAE, RBM), and LensKit algorithms
need extras — install from a clone with `uv sync --extra recbole` (or `--extra tensorflow` /
`--extra lenskit`; pip equivalent `pip install -e ".[recbole]"`). If you're running via Docker, rebuild
with `EASYSTUDY_EXTRAS="recbole" docker compose up --build` (the `recbole` extra needs
`PYTHON_VERSION=3.11` too — see the README).

The server log prints a line like `[loading] skipping 'plugins…' — optional dependency missing: …` for
each skipped module. **EASE**, **Popularity**, and **Random** are always available in the lightweight core.

## `ImportError: TFRS-based recommenders require the optional 'tensorflow' extra`
You selected a TensorFlow-based component without the extra installed. Install the `tensorflow` extra
(or pick EASE / a non-TF method).

## Item images don't show up, or show a "No poster" placeholder
For MovieLens loaders, posters are fetched from IMDb on first view and cached to
`server/static/datasets/<dataset>/img/`; pre-fetch them all with
`python scripts/fetch_images.py --dataset <dataset>`. A small number of items have no usable IMDb cover
(dead/invalid URL) — those show the shared `no_poster.svg` placeholder by design, not an error.
Custom datasets must return valid `get_item_*_image_url(...)` (a `static` URL is fastest; remote
`http://…` works but is slow).

## Pre-fetched images/data don't seem to be used — everything is slow, like it's fetching live
You likely ran `python scripts/fetch_data.py`/`fetch_images.py` directly in the same checkout
directory where you also use `easystudy serve`. The raw scripts always write to
`server/static/datasets`; the CLI instead redirects data to your *current directory*, a
different location — so `easystudy serve` never sees what the raw scripts fetched, and
re-fetches everything live (slowly) instead. `easystudy serve`/`create-user`/etc. print a
warning when this situation is detected. Fix: use `easystudy fetch-data`/`easystudy
fetch-images` instead of the raw scripts once you're using the CLI from that directory.

## Sessions reset / users logged out after restart
Set a fixed `SECRET_KEY` (see [Deployment](deployment.md)); the default is randomized per boot.

## The app slows down / stalls with many concurrent participants
This is a known limitation of the current single-worker design (per-participant model state lives in
worker memory). See the concurrency note in [Deployment](deployment.md) and the scalability items in
the technical backlog. Short term: keep `GUNICORN_WORKERS=1`, run multiple instances behind a
sticky-session load balancer, and rate-limit recruitment.

## Tests error with missing dataset
`pytest` skips dataset-dependent tests automatically when data isn't present. To run them, fetch the
MovieLens dataset first: `python scripts/fetch_data.py --dataset ml-latest`.

## Still stuck?
Open an issue with the bug template (OS, Python version, install extras, branch/commit, traceback).
