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

## 2. Fetch datasets (first time only)

The framework ships loaders for **MovieLens** and **goodbooks-10k**; the data itself is downloaded on
demand:

=== "Docker"
    ```bash
    docker compose run --rm fetch          # CSVs + item images
    ```
=== "Python"
    ```bash
    python scripts/fetch_data.py --dataset all
    # or just one: python scripts/fetch_data.py --dataset goodbooks-10k
    ```

This populates `server/static/datasets/{ml-latest,goodbooks-10k}/`.

!!! tip "Faster first run"
    MovieLens `ml-latest` is large. To try things quickly, fetch only goodbooks:
    `python scripts/fetch_data.py --dataset goodbooks-10k`.

## 3. Start the server

=== "Docker"
    ```bash
    docker compose up --build
    ```
=== "Python"
    ```bash
    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"                # lightweight core + dev tools
    cd server && flask --debug run
    ```

Open **http://localhost:5000**.

## 4. Create an administrator account

On first run there are no users. Go to **`/signup`**, create an account, then log in at **`/login`**.
You land on the **administration** page, which lists study templates (plugins) and existing studies.

## 5. Create a study with `fastcompare`

`fastcompare` is the built-in template for comparing 2–3 recommenders within-subject.

1. Click **Create** on the *fastcompare* plugin.
2. Fill the form (every field has a `?` help tooltip):
   - **Data loader** — e.g. *Goodbooks-10k dataset*.
   - **Preference elicitation** — e.g. *Popularity Sampling* (works in the lightweight core).
   - **Algorithms** — pick 2–3, e.g. **EASE** and a baseline. Algorithm parameters appear as
     editable fields automatically.
   - Recommendation size, number of iterations, layout, prompts, optional Prolific code.
3. Submit. The study **initializes in the background** (training the algorithms); when done it becomes
   **active** and you get a **join URL**.

!!! note "Lightweight core"
    With just `pip install easystudy` (no extras), TensorFlow/LensKit algorithms won't appear —
    **EASE** and the popularity/multi-objective elicitations are available out of the box. Install
    `easystudy[tensorflow]` or `[lenskit]` to unlock the rest.

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
