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
Its optional dependency isn't installed. TensorFlow-based (VAE, TFRS) and LensKit algorithms need
extras:
```bash
pip install "easystudy[tensorflow]"   # or [lenskit]
```
The server log prints a line like `[loading] skipping 'plugins…' — optional dependency missing: …` for
each skipped module. **EASE** and the popularity/multi-objective elicitations are always available in
the lightweight core.

## `ImportError: TFRS-based recommenders require the optional 'tensorflow' extra`
You selected a TensorFlow-based component without the extra installed. Install
`easystudy[tensorflow]` (or pick EASE / a non-TF method).

## Item images don't show up
Fetch images (`fetch_data.py` without `--no-images`) so they exist under
`server/static/datasets/<dataset>/img/`. Custom datasets must return valid `get_item_*_image_url(...)`
(a `static` URL is fastest; remote `http://…` works but is slow).

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
