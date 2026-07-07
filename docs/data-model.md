# Data model & exporting results

Everything a study collects lives in the application database (SQLite by default at
`server/instance/db.sqlite`, or your `DATABASE_URL`). Defined in `server/models.py`.

## Tables

### `UserStudy` — one study instance
| Column | Meaning |
|--------|---------|
| `id` | primary key |
| `creator` | admin email (FK → `user.email`) |
| `guid` | opaque id used in the join/link URLs |
| `parent_plugin` | template the study was created from (e.g. `fastcompare`) |
| `settings` | **JSON string** of all study parameters chosen at creation |
| `time_created` | creation timestamp |
| `active` / `initialized` | lifecycle flags set by the plugin |
| `initialization_error` | populated if background setup failed |

### `Participation` — one participant's run
| Column | Meaning |
|--------|---------|
| `id` | primary key |
| `participant_email` | may be empty (anonymous) |
| `age_group`, `gender`, `education`, `ml_familiar` | demographics from the intake form |
| `user_study_id` | FK → `userstudy.id` (cascade delete) |
| `time_joined`, `time_finished` | timing (finished is null until completion) |
| `uuid`, `language`, `extra_data` | session id, locale, free JSON |

### `Interaction` — every tracked event
| Column | Meaning |
|--------|---------|
| `id` | primary key |
| `participation` | FK → `participation.id` (cascade delete) |
| `interaction_type` | string, e.g. `selected-item`, `deselected-item`, `changed-viewport`, `clicked-button` |
| `time` | timestamp |
| `data` | **JSON string** payload specific to the interaction type |

### `Message` — free-form per-participant records
`id`, `participation` (FK, nullable), `time`, `data` (JSON). Used for questionnaire answers and other
arbitrary payloads.

!!! note "Two JSON columns to know"
    `UserStudy.settings` and `Interaction.data` (and `Message.data`) are JSON **stored as strings** —
    `json.loads` them when analyzing. Standardizing/indexing these is a v2 improvement (see the
    technical backlog).

## Getting your data out

### From the admin UI
Open **Results** for a study. `fastcompare` shows built-in metrics (nDCG@K, Precision@K, ILD, Selection
Count) and summaries.

### Directly from the database
Because it's plain SQL, you can query with anything. Example with pandas:

```python
import sqlite3, pandas as pd, json

con = sqlite3.connect("server/instance/db.sqlite")

studies = pd.read_sql("SELECT * FROM userstudy", con)
parts   = pd.read_sql("SELECT * FROM participation WHERE user_study_id = ?", con, params=[STUDY_ID])
inter   = pd.read_sql(
    "SELECT i.* FROM interaction i JOIN participation p ON i.participation = p.id "
    "WHERE p.user_study_id = ?", con, params=[STUDY_ID])

# expand the JSON payloads
inter["payload"] = inter["data"].apply(lambda s: json.loads(s) if s else {})
```

Common recipes:

- **Attention-check pass rate** — filter `interaction_type` for your check events and aggregate per
  participation.
- **Questionnaire answers** — read from `message` (or the relevant `interaction_type`), `json.loads`
  the `data`.
- **Selections per round** — filter `interaction_type == "selected-item"`, group by participation.

!!! tip "Roadmap"
    A first-class `easystudy export <study_id> --csv` helper and an in-admin analytics dashboard
    (completion funnel, attention-check rates, answer distributions, click heatmaps) are planned —
    see the v2 features/technical backlogs. Until then, the SQL recipes above are the supported path.
