# Concepts

A quick mental model of how EasyStudy is put together.

## Framework, plugins, studies, instances

```
EasyStudy (Flask app)
 ├─ Administration UI        ← researchers create/monitor studies
 ├─ Plugins                  ← code that defines *what a study is*
 │   ├─ utils                (shared: preference elicitation, tracking, common GUIs) — not a template
 │   ├─ fastcompare          (a study TEMPLATE: compare 2–3 recommenders)
 │   ├─ vae / …              (extra algorithms for fastcompare)
 │   └─ your-plugin          (your custom study flow)
 └─ Database (SQLite/Postgres)  ← users, studies, participations, interactions
```

- **Plugin** — a self-contained package of backend endpoints + UI. Two kinds:
    - **Utility plugins** (e.g. `utils`, `vae`) — reusable building blocks; *cannot* create a study.
    - **Study plugins** (e.g. `fastcompare`) — **templates** you instantiate into concrete studies.
- **User study (instance)** — one concrete study created from a template, with its own settings and its
  own **join URL**. You can create many instances from one template with different parameters.
- **Participation** — one participant's run through one study instance (may be anonymous).

## The fastcompare study flow

`fastcompare` implements the most common flow (configurable when you create the instance):

1. **Initialization** — participant details + informed consent.
2. **Preference elicitation** — collect signal to personalize (e.g. popularity sampling).
3. **Comparison rounds ×N** — the participant sees recommendations from each algorithm and selects items
   they'd engage with; optionally rates each algorithm. Previously shown items are filtered out.
4. **Final page** — usage stats + optional post-study questionnaire.

Behind the scenes fastcompare trains the chosen algorithms on the dataset, then per participant
personalizes from the elicitation and each round's feedback.

## The plugin lifecycle (endpoints)

A study plugin exposes a small set of endpoints that EasyStudy calls at the right time:

| Endpoint | When | Responsibility |
|----------|------|----------------|
| `/create` | researcher clicks *Create* | render the parameter form; must end by calling `/create-user-study` |
| `/initialize` | after creation | do setup (train models…), ideally in a **background daemon**; then mark the study `initialized=True`, `active=True` |
| `/join` | participant opens the join URL | take over and drive the participant's flow |
| `/results` *(optional)* | researcher clicks *Results* | custom evaluation view (falls back to `utils` default) |

See [Write your first plugin](first-plugin.md) for a walkthrough.

## Extensible components of fastcompare

Without writing a whole plugin, you can extend `fastcompare` by subclassing one base class:

| Base class | Adds a… | Guide |
|------------|---------|-------|
| `DataLoaderBase` | dataset / domain | [Extending](extending.md#datasets) |
| `AlgorithmBase` | recommender algorithm | [Extending](extending.md#algorithms) |
| `PreferenceElicitationBase` | elicitation method | [Extending](extending.md#preference-elicitation) |
| `EvaluationMetricBase` | evaluation metric | [Extending](extending.md#metrics) |

Discovery is automatic: EasyStudy scans the `plugins/` tree for subclasses. Modules whose **optional
dependency** (e.g. TensorFlow) isn't installed are skipped, so the lightweight core still runs.

## Lightweight core vs extras

`pip install easystudy` gives a small pure-Python install. Heavy backends are opt-in:

```bash
pip install "easystudy[tensorflow]"   # VAE / TF-Recommenders algorithms
pip install "easystudy[lenskit]"      # LensKit baselines
pip install "easystudy[redis]"        # redis-backed sessions
```

An algorithm whose extra is missing simply won't appear in the study-creation UI.

## Item ids vs. item indices (and users)

This distinction trips up everyone writing a data loader, so learn it once:

| Term | What it is | Example | Used for |
|------|-----------|---------|----------|
| **`item_id`** | The identifier from the **underlying dataset** — arbitrary, often non-contiguous, not necessarily zero-based. | a MovieLens `movieId` like `260`, a Goodreads book id | image file paths (`img/{item_id}.jpg`), joining external metadata, anything user-facing/original |
| **`item_index`** (a.k.a. `item_idx`, and the `item` column) | A **dense, zero-based** integer EasyStudy assigns so items map to rows/columns of NumPy arrays. | `0, 1, 2, …, n-1` | all algorithm math: the rating matrix, distance matrix, model weights — anything indexed into an array |

Why both exist: algorithms work with NumPy matrices, which need contiguous 0-based indices, but
datasets ship messy real ids. So EasyStudy keeps a **bijection** between them:

- `get_item_index(item_id) -> item_index`
- `get_item_id(item_index) -> item_id`

(Internally these are backed by maps like `movie_id_to_index` / `movie_index_to_id`.) **Users** follow
the same pattern: the raw `userId` from the dataset vs. a zero-based `user_index` used for the rating
matrix rows.

Rules of thumb when implementing a `DataLoaderBase`:

- `ratings_df` exposes the **dense** space: a `user` column and an `item` column (both zero-based
  indices) plus `rating`. This is what algorithms like EASE pivot into a matrix.
- Return **`item_id`** from image/description helpers keyed on ids, and convert with `get_item_index` /
  `get_item_id` whenever you cross between the two worlds.
- Never assume `item_id == item_index`; the [test harness](testing-components.md)'s `TinyDataLoader`
  deliberately uses ids `100..107` for indices `0..7` to catch that bug.
