# Quickstart (15 minutes)

Goal: clone EasyStudy, start it, create a study comparing two recommenders from the web UI, join it as a
participant, and find your collected data. You need only **Docker** (recommended) or **Python 3.10+**.

## 1. Get the code

```bash
git clone https://github.com/pdokoupil/EasyStudy.git
cd EasyStudy
```

!!! success "No more LFS errors"
    Datasets/images are no longer stored in git, so the clone is fast and never hits the old
    "git-LFS over data quota" error.

## 2. Fetch a dataset (first time only)

`fetch_data.py` is pure standard-library, so it runs even before any install. Start with
**`ml-latest-small`** — a small (~1 MB), fast MovieLens subset that's ideal for a first study:

```bash
python scripts/fetch_data.py --dataset ml-latest-small
```

This populates `server/static/datasets/ml-latest-small/`. Other options:
`goodbooks-10k`, `ml-latest` (the full, much larger MovieLens dump), or `all`.

!!! tip "Pre-fetch item posters (optional)"
    Posters are fetched from IMDb and cached to disk the first time each item is shown. To avoid that
    first-view delay, pre-fetch them all up front (~1 minute for `ml-latest-small`):
    `python scripts/fetch_images.py --dataset ml-latest-small`

## 3. Start the server

=== "Docker"
    ```bash
    docker compose up --build
    ```
    Open **<http://localhost:8000>**. To unlock more algorithms (RecBole, TensorFlow, LensKit), rebuild
    with extras — see the [README Quickstart](https://github.com/pdokoupil/EasyStudy#quickstart-docker-recommended).
=== "Python (uv, recommended)"
    ```bash
    uv sync --extra dev                    # creates .venv from uv.lock (core + dev tools)
    cd server && uv run flask --debug run
    ```
    Open **<http://localhost:5000>**.
=== "Python (pip)"
    ```bash
    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"                # lightweight core + dev tools
    cd server && flask --debug run
    ```
    Open **<http://localhost:5000>**.

## 4. Create an administrator account

The root URL redirects straight to the administration UI, which redirects to `/login` if you're not
signed in yet. Either:

- click **Sign up** and create an account through the web form, or
- (Docker) run `docker compose exec app flask create-user you@example.com yourpassword`.

## 5. Create a study with `fastcompare`

`fastcompare` is the built-in template for comparing 2–3 recommenders within-subject.

1. Click **Create** on the *fastcompare* plugin.
2. Fill the form (every field has a `?` help tooltip):
   - **Data loader** — e.g. *MovieLens Latest Small (demo)*.
   - **Preference elicitation** — e.g. *Popularity Sampling* (works in the lightweight core).
   - **Algorithms** — pick 2–3, e.g. **EASE** and **Popularity**. Algorithm parameters appear as
     editable fields automatically.
   - Recommendation size, number of iterations, layout, prompts, optional Prolific code.
3. Submit. The study **initializes in the background** (training the algorithms); when done it becomes
   **active** and you get a **join URL**.

!!! note "Lightweight core"
    With just the core install (no extras), only **EASE**, **Popularity**, and **Random** appear.
    Install `easystudy[recbole]` / `[tensorflow]` / `[lenskit]` to unlock more algorithms.

## 6. Take the study as a participant

Open the **join URL** (in a private window if you like). You'll go through: consent & details →
preference elicitation → *N* recommendation rounds (select items you'd engage with) → final page /
optional questionnaire.

## 7. Find your data

Back in the admin UI, open **Results** for your study. Under the hood, every action is stored in the
database (`Interaction`, `Participation`, `Message` tables — see
[Data model & exporting results](data-model.md)). You can also query the SQLite DB directly at
`server/instance/db.sqlite`.

## Next steps

- [Concepts](concepts.md) — how plugins, studies, and instances relate.
- [Create a study (no code)](create-a-study.md) — the admin UI in depth.
- [Extending fastcompare](extending.md) — add your own dataset, algorithm, or metric.
