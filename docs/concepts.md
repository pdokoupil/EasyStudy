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
