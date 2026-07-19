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

## Quickstart (Docker, recommended)

> **No local setup?** The badge below opens a free, temporary browser-based VS Code (GitHub Codespaces)
> with the lightweight Python environment already installed and a small demo dataset fetched. Once it's
> ready, run `cd server && flask --debug run` in its terminal and open the forwarded port.
>
> [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/pdokoupil/EasyStudy?quickstart=1)

```bash
git clone https://github.com/pdokoupil/EasyStudy.git
cd EasyStudy
python scripts/fetch_data.py --dataset ml-latest-small   # small, fast demo dataset (~1MB of CSVs)
docker compose up --build                                # http://localhost:8000
```

Open **http://localhost:8000** — it redirects straight to the administration UI. Create an account at
`/signup` (or run `docker compose exec app flask create-user you@example.com yourpassword`), log in, and
create a study.

Item posters are fetched from IMDb on first view and cached to disk; to pre-fetch them all up front
(so the first study has no delay), run `python scripts/fetch_images.py --dataset ml-latest-small`.

By default only the lightweight-core algorithms appear (**EASE**, **Popularity**, **Random**). To unlock
more — RecBole's model zoo, TensorFlow-based VAE/RBM, or LensKit baselines — rebuild with extras:
```bash
EASYSTUDY_EXTRAS="recbole" docker compose up --build       # BPR, LightGCN, NGCF, NeuMF, DMF
# the `recbole` extra needs a dependency with no Python 3.12 wheel yet, so build it on 3.11:
PYTHON_VERSION=3.11 EASYSTUDY_EXTRAS="recbole" docker compose up --build
```

<details>
<summary><b>Run from source (for development)</b></summary>

```bash
uv sync --extra dev                    # creates .venv from uv.lock (core + dev tools)
python scripts/fetch_data.py --dataset ml-latest-small
cd server && uv run flask --debug run  # http://localhost:5000
```
No [uv](https://docs.astral.sh/uv/)? `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"` works too.
</details>

## Installation & dependency extras

EasyStudy isn't published on PyPI yet — install from a clone. The **core install is deliberately
lightweight** (Flask + pandas/numpy/scikit-learn). Heavy or optional backends live behind extras —
nobody installs TensorFlow or PyTorch to run a books-vs-movies study:

```bash
uv sync                        # lightweight core (default)
uv sync --extra recbole        # + RecBole model zoo (BPR, LightGCN, NGCF, NeuMF, DMF)
uv sync --extra tensorflow     # + TensorFlow / TF-Recommenders algorithms (VAE, RBM)
uv sync --extra lenskit        # + LensKit baseline wrappers
uv sync --extra redis          # + redis-backed sessions
uv sync --extra all            # everything
```
(Or the pip equivalent: `pip install -e ".[recbole]"`, etc.)

Configuration is via environment variables (see [`.env.example`](.env.example)): `SECRET_KEY`,
`DATABASE_URL` (SQLite by default, Postgres for scale), `SESSION_TYPE`/`REDIS_URL`, `PORT`.

## Datasets

Datasets and item images are **fetched on demand** (they are intentionally not committed to git):

```bash
python scripts/fetch_data.py --dataset ml-latest-small   # small, fast demo (recommended first try)
python scripts/fetch_data.py --dataset all                # MovieLens + goodbooks, CSVs + images
python scripts/fetch_data.py --dataset ml-latest --no-images
python scripts/fetch_images.py --dataset ml-latest-small  # pre-cache all posters (optional, ~1 min)
```

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
  title     = {EasyStudy: Framework for Easy Deployment of User Studies on Recommender Systems},
  author    = {Dokoupil, Patrik and Peska, Ladislav},
  booktitle = {Proceedings of the 17th ACM Conference on Recommender Systems (RecSys '23)},
  pages     = {1196--1199},
  year      = {2023},
  doi       = {10.1145/3604915.3610640}
}
```

## License

[MIT](LICENSE). Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Contact
[Patrik Dokoupil](mailto:patrik.dokoupil@matfyz.cuni.cz) · [Ladislav Peska](mailto:ladislav.peska@matfyz.cuni.cz)
