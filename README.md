<h1 align="center">EasyStudy</h1>

<p align="center">
  <b>Open-source framework for building, running, and analyzing interactive user studies —<br>
  for recommender systems, HCI, and any item-based experiment.</b>
</p>

<p align="center">
  <a href="https://doi.org/10.1145/3604915.3610640"><img alt="Paper" src="https://img.shields.io/badge/RecSys'23-paper-b31b1b"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green"></a>
  <a href="https://github.com/pdokoupil/EasyStudy/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/pdokoupil/EasyStudy/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pdokoupil.github.io/EasyStudy/"><img alt="Docs" src="https://github.com/pdokoupil/EasyStudy/actions/workflows/docs.yml/badge.svg"></a>
</p>

EasyStudy is a **modular, highly extensible** framework for deploying customizable user studies. It ships
out-of-the-box functionality for every phase of a study — data loading & presentation, preference
elicitation, baseline algorithms, questionnaires, and results comparison — so a simple study can be
deployed **in a few clicks from a web UI**, while complex designs stay fully programmable via plugins.

Built for recommender-systems research, EasyStudy is **domain-generic**: it has already been used for
recommender-systems studies, **data-visualization** studies, and **university teaching**. If your study
shows people *items*, records how they *interact*, and asks them *questions*, EasyStudy fits.

> Credo: *"Make simple things fast and hard things possible."*

## Who is this for?

- **RS / IR / HCI researchers** who want user studies without rebuilding UI, tracking, and elicitation each time.
- **Instructors & students** — try algorithms live, or implement and evaluate an extension in a single plugin.

## Used in research

EasyStudy has powered the user studies behind peer-reviewed publications in recommender systems and data
visualization — see **[Studies built with EasyStudy](https://pdokoupil.github.io/EasyStudy/publications/)**.

**Using EasyStudy in your research or teaching?** We'd genuinely love to know — it helps us understand
where it's useful (labs, courses, companies) and what to improve:

- 📄 Add your paper: open a [pull request or issue](https://github.com/pdokoupil/EasyStudy/issues/new).
- 💬 Share feedback, ask questions, or say hi in [GitHub Discussions](https://github.com/pdokoupil/EasyStudy/discussions).
- ✉️ Or email <patrik.dokoupil@matfyz.cuni.cz>.

## Quickstart

Not on PyPI yet, so install from a clone — the core install is deliberately lightweight (Flask +
pandas/numpy/scikit-learn); heavy backends live behind extras so nobody installs TensorFlow or
PyTorch just to run a books-vs-movies study:

```bash
git clone https://github.com/pdokoupil/EasyStudy.git
cd EasyStudy
pip install -e ".[dev]"                             # lightweight core + dev tools
easystudy fetch-data --dataset ml-latest-small       # small, fast demo dataset (~1 MB of CSVs)
easystudy create-user you@example.com yourpassword
easystudy serve --debug                              # http://localhost:8000
```

Open **http://localhost:8000**, log in with the account you just created, and create a study from
the *fastcompare* template. `easystudy fetch-data`/`create-user`/`serve` all use whichever directory
you run them from for the dataset/DB — no need to `cd` into `server/`.

`easystudy serve` uses Flask's own dev server (single-process, auto-reloads with `--debug`) —
good for trying things out, not for a real deployment. Gunicorn (same server Docker below uses)
is already installed alongside the core package; run `easystudy serve --prod` to use it directly,
no Docker required.

Item posters are fetched from IMDb on first view and cached to disk; to pre-fetch them all up front,
run `easystudy fetch-images --dataset ml-latest-small`.

By default only the lightweight-core algorithms appear (**EASE**, **Popularity**, **Random**).
Install an extra to unlock more:
```bash
pip install -e ".[recbole]"       # RecBole model zoo: BPR, LightGCN, NGCF, NeuMF, DMF
pip install -e ".[tensorflow]"    # VAE, RBM
pip install -e ".[lenskit]"       # ItemKNN, UserKNN, ImplicitMF
pip install -e ".[all]"           # everything
```
> The `recbole` extra pulls a dependency with no Python 3.12+ wheels yet — use Python 3.10 or 3.11
> for that one (`python3.11 -m venv .venv && ...`).

Once published, all of the above becomes clone-free: `pip install "easystudy[recbole]"`.

Configuration is via environment variables (see [`.env.example`](.env.example)): `SECRET_KEY`,
`DATABASE_URL` (SQLite by default, Postgres for scale), `SESSION_TYPE`/`REDIS_URL`, `PORT`.

### Also works with `uv`

This repo is itself developed with [uv](https://docs.astral.sh/uv/) — `uv sync` (backed by the
committed `uv.lock`) is what our own CI uses. It's a drop-in, much faster alternative to pip + venv,
and does a couple of things pip doesn't:
- **Speed** — dependency resolution/installs are typically 10–100× faster.
- **One lockfile** — `uv.lock` pins exact versions across *every* extra, resolved together; there's
  no equivalent to that with plain pip.
- **Manages Python itself** — `uv venv --python 3.11` downloads that Python for you if you don't
  have it (handy for the `recbole` extra above); plain pip just uses whichever Python is already active.

```bash
git clone https://github.com/pdokoupil/EasyStudy.git
cd EasyStudy
uv sync --extra dev                              # creates .venv from uv.lock
uv run easystudy fetch-data --dataset ml-latest-small
uv run easystudy create-user you@example.com yourpassword
uv run easystudy serve --debug
```
Extras work the same way: `uv sync --extra recbole`, etc.

### Docker

Useful if you'd rather not install any Python locally, or want something closer to a real
deployment (gunicorn, optional redis-backed sessions):

```bash
git clone https://github.com/pdokoupil/EasyStudy.git
cd EasyStudy
python scripts/fetch_data.py --dataset ml-latest-small
docker compose up --build                          # http://localhost:8000
```
Create an account at `/signup`, or `docker compose exec app flask create-user you@example.com yourpassword`.

To unlock more algorithms, rebuild with extras (same Python 3.11 caveat for `recbole` applies):
```bash
EASYSTUDY_EXTRAS="recbole" docker compose up --build
PYTHON_VERSION=3.11 EASYSTUDY_EXTRAS="recbole" docker compose up --build
```

### pip/uv vs. Docker — what actually differs

All three paths run the identical app; they only differ in *how much Docker sets up for you*:

| | `easystudy serve` | `easystudy serve --prod` | Docker |
|---|---|---|---|
| WSGI server | Flask dev server (not for production) | gunicorn | gunicorn |
| Redis-backed sessions | run your own `redis-server` + set `SESSION_TYPE`/`REDIS_URL` | same | `docker compose --profile redis up` starts one for you |
| Heavy extras | `pip install -e ".[recbole]"` / `uv sync --extra recbole` | same | `EASYSTUDY_EXTRAS=recbole` build arg |
| Default port | 8000 (`--port` to change) | 8000 | 8000 (`PORT` env var to change) |

Gunicorn is a core dependency either way — Docker doesn't add capability, it just automates
the setup (and gives you an isolated container). Redis works identically regardless of how you
installed EasyStudy; it's just an environment-variable switch.

## Datasets

Datasets and item images are **fetched on demand** (they are intentionally not committed to git). Use
`python scripts/fetch_data.py` before installing anything (it's stdlib-only), or `easystudy fetch-data`
after `pip install`/`uv sync` — same script either way:

```bash
python scripts/fetch_data.py --dataset ml-latest-small   # small, fast demo (recommended first try)
python scripts/fetch_data.py --dataset all                # MovieLens + goodbooks, CSVs + images
python scripts/fetch_data.py --dataset ml-latest --no-images
python scripts/fetch_images.py --dataset ml-latest-small  # pre-cache all posters (optional, ~1 min)
```

Once you've `pip install`-ed and use `easystudy serve` from this same checkout directory, switch to
`easystudy fetch-data`/`easystudy fetch-images` for anything after that. The raw scripts always write
to `server/static/datasets`; the CLI redirects everything (including these) to your current directory
instead — mixing the two silently produces two different data directories, and `easystudy serve` won't
see anything the raw scripts fetched (it'll look empty and re-fetch slowly from IMDb/MovieLens on
every request instead of using the cache).

Sources: [ml-latest-small / ml-latest](https://files.grouplens.org/datasets/movielens/),
[goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k). Images mirror:
[ml_latest_img.zip](http://herkules.ms.mff.cuni.cz/ligan/easystudy/ml_latest_img.zip),
[goodbooks_img.zip](http://herkules.ms.mff.cuni.cz/ligan/easystudy/goodbooks_img.zip).

## Plugins

**See the [paper](https://doi.org/10.1145/3604915.3610640) for the detailed description of plugins.**

- [fastcompare](./server/plugins/fastcompare/) — compare 2–3 algorithms on an implicit-feedback dataset.
  Extend it with new data loaders, algorithms, preference-elicitation methods, or metrics via subclassing.
- [utils](./server/plugins/utils/) — shared functionality (not a study template).
- [layoutshuffling](./server/plugins/layoutshuffling/) — a plugin from one of our internal studies (illustrative).
- [vae](./server/plugins/vae/) — MultVAE / StandardVAE / RBM algorithms for *fastcompare* (needs `[tensorflow]`).
- [recbole](./server/plugins/recbole/) — BPR / LightGCN / NGCF / NeuMF / DMF for *fastcompare* (needs `[recbole]`).
- [empty_template](./server/plugins/empty_template/) — the minimal working plugin; start here.

## Development

### Extending fastcompare
Add a class subclassing the appropriate base — either directly in `plugins/fastcompare/algo/*.py` or in a
separate plugin `plugins/<yourplugin>/*.py`:
- **Datasets** → subclass `DataLoaderBase` (see `GoodbooksDataLoader`); put data under
  [static/datasets](./server/static/datasets/) or ship it inside your plugin and use `pm.emit_assets(...)`.
- **Algorithms** → subclass `AlgorithmBase`.
- **Preference elicitation** → subclass `PreferenceElicitationBase`.
- **Evaluation metrics** → subclass `EvaluationMetricBase`.

All abstract methods/properties are documented in the base-class docstrings. Algorithm parameters annotated
on your class automatically become clickable, configurable fields in the study-creation UI.

> Heads-up: heavy backends (RecBole, TensorFlow, LensKit) are optional extras. Plugin discovery **skips**
> modules whose optional dependency isn't installed, so the lightweight core runs fine without them —
> install the matching extra to enable those algorithms.

### Adding a plugin
Create a folder under `server/plugins/`. A study plugin should expose:
- `/create` — renders the parameter page; must ultimately call `/create-user-study`.
- `/initialize` — post-creation setup (do long work in a background daemon; then mark the study
  `initialized=True`, `active=True`).
- `/join` — entry point when a participant follows the study URL.
- `/results` *(optional)* — custom evaluation view; falls back to `utils`' default if omitted.

Start from [empty_template](./server/plugins/empty_template). See [CONTRIBUTING.md](CONTRIBUTING.md).

## Links

- **[Documentation](https://pdokoupil.github.io/EasyStudy/)** — quickstart, concepts, guides, API reference.
- **[Studies built with EasyStudy](https://pdokoupil.github.io/EasyStudy/publications/)** — peer-reviewed
  publications whose user studies were run on EasyStudy (recommender systems & data visualization).
- **[Live demo & hosted instance](https://pdokoupil.github.io/EasyStudy/live-demo/)** — hosted
  administration/database (access details in the paper) and the walkthrough video. Hosted URLs can
  change, so they live on that page rather than in short links here.
- **[Walkthrough recording](https://youtu.be/xogcaJDOcFw)** (YouTube).

## Citation

If you use EasyStudy in your research, please cite the RecSys'23 paper (see [CITATION.cff](CITATION.cff)):

```bibtex
@inproceedings{dokoupil2023easystudy,
author = {Dokoupil, Patrik and Peska, Ladislav},
title = {EasyStudy: Framework for Easy Deployment of User Studies on Recommender Systems},
year = {2023},
isbn = {9798400702419},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3604915.3610640},
doi = {10.1145/3604915.3610640},
booktitle = {Proceedings of the 17th ACM Conference on Recommender Systems},
pages = {1196–1199},
numpages = {4},
keywords = {Recommender systems, evaluation frameworks, user centric, user studies},
location = {Singapore, Singapore},
series = {RecSys '23}
}
```

## License

[MIT](LICENSE). Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Contact
[Patrik Dokoupil](mailto:patrik.dokoupil@matfyz.cuni.cz) · [Ladislav Peska](mailto:ladislav.peska@matfyz.cuni.cz)
